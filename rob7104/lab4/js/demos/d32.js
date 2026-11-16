// 3.2 Two kicks cancel: a Franka arm carries a glass of water 0.30 m in two half-steps.
// The delay between the half-steps is the viewer's choice; half a damped slosh period leaves the water flat.
// The robot is the nonlinear slosh pendulum; the theory picture uses the linear model.
import * as S from '../sim.js';
import { CSS, val, onInput, onClick, readout, ctl, seg, plot2d, stepper, deg } from '../common.js';
import { OM, ZE, WD, PERIOD, SPILL, PEND, loadWaiter, amplitude, setSlider, Ax, label } from '../waiter_common.js';

const H = 1e-3;                                   // integration step (s)
const D = 0.30, TM = 0.60, T0 = 0.10, X0 = -0.15; // travel, duration of the plain (minimum-jerk) move, start time
const TEND = 1.9;                                 // length of the time axis of the theory picture
const REC = 4;
const ZV = S.zvShaper(OM, ZE);                    // amplitudes [0.516, 0.484], ideal delay pi / wd = 0.138 s
const [A1, A2] = ZV.amps;
const IDEAL = ZV.times[1];

const pos = (s) => { const u = Math.max(0, Math.min(1, s / TM)); return D * u * u * u * (10 - 15 * u + 6 * u * u); };
const vel = (s) => { const u = Math.max(0, Math.min(1, s / TM)); return D / TM * 30 * u * u * (1 - u) * (1 - u); };
const acc = (s) => S.smoothStepAccel(D, TM, s);
const handAcc = (t, d) => A1 * acc(t - T0) + A2 * acc(t - T0 - d);
const handPos = (t, d) => X0 + A1 * pos(t - T0) + A2 * pos(t - T0 - d);
const handVel = (t, d) => A1 * vel(t - T0) + A2 * vel(t - T0 - d);
const tStop = (d) => T0 + TM + d;

const lin = new S.LinearSim(PEND.A, PEND.B, H);

/** One step of both models under the hand acceleration of delay d, from time t to t + H. */
function stepBoth(st, t, d) {
  const a0 = handAcc(t, d), a1 = handAcc(t + H, d);
  st.nl = S.pendStep(OM, ZE, st.nl, a0, a1, H);
  st.ln = lin.step(st.ln, 0.5 * (a0 + a1));
}

/** Whole move offline: peaks, and the largest tilt during the first slosh period after the hand stops. */
function offline(d) {
  const st = { nl: [0, 0], ln: [0, 0] }, ts = tStop(d), n = Math.ceil((ts + PERIOD) / H);
  const o = { delay: d, peakNl: 0, peakLin: 0, resNl: 0, resLin: 0, ampNl: 0, ampLin: 0 };
  for (let k = 0; k < n; k++) {
    stepBoth(st, k * H, d);
    const t = (k + 1) * H, a = Math.abs(st.nl[0]), b = Math.abs(st.ln[0]);
    o.peakNl = Math.max(o.peakNl, a); o.peakLin = Math.max(o.peakLin, b);
    if (t >= ts - 1e-9 && t <= ts + PERIOD + 1e-9) {
      if (!o.atStop) { o.atStop = true; o.ampNl = amplitude(st.nl); o.ampLin = amplitude(st.ln); }   // envelope at the stop
      o.resNl = Math.max(o.resNl, a); o.resLin = Math.max(o.resLin, b);
    }
  }
  o.spilled = o.peakNl > SPILL;
  return o;
}
const inDeg = (o) => ({ delay: o.delay, peakNlDeg: deg(o.peakNl), peakLinDeg: deg(o.peakLin), residNlDeg: deg(o.resNl),
  residLinDeg: deg(o.resLin), ampAtStopNlDeg: deg(o.ampNl), ampAtStopLinDeg: deg(o.ampLin), spilled: o.spilled });

// tilt of the linear model for the plain move starting at time 0, sampled every H
const NB = Math.round((TEND + 0.1) / H);
const base = new Float64Array(NB + 1);
{
  let x = [0, 0];
  for (let k = 0; k < NB; k++) { x = lin.step(x, 0.5 * (acc(k * H) + acc((k + 1) * H))); base[k + 1] = x[0]; }
}
const baseAt = (s) => {
  if (s <= 0) return 0;
  const u = Math.min(s / H, NB - 1e-9), i = Math.floor(u);
  return base[i] + (u - i) * (base[i + 1] - base[i]);
};

// residual against delay, for the lower plot
const TABLE = { delays: [], lin: [], nl: [] };
for (let i = 0; i <= 140; i++) {
  const o = offline(i * 0.0025);
  TABLE.delays.push(o.delay); TABLE.lin.push(deg(o.resLin)); TABLE.nl.push(deg(o.resNl));
}
const REF = { ideal: offline(IDEAL), zero: offline(0) };
let best = { delay: 0, resNl: Infinity };
for (let d = 0.11; d <= 0.16; d += 0.001) { const o = offline(d); if (o.resNl < best.resNl) best = o; }

export async function init(panel) {
  const W = await loadWaiter(panel);
  const P = plot2d(panel.querySelector('canvas.plot'));
  const slider = ctl(panel, 'delay');
  const advance = stepper(H);
  let speed = 0.25;
  seg(panel, 'speed', (s) => { speed = Number(s); });

  let delay = val(panel, 'delay');
  let theory = null;                              // curves of the linear model for the current delay
  let run = null;                                 // the robot's current move

  function setDelay(d) {
    delay = d;
    const n = Math.round(TEND / (REC * H)) + 1;
    const th = { t: new Array(n), a1: new Array(n), a2: new Array(n), as: new Array(n), th1: new Array(n), th2: new Array(n), sum: new Array(n) };
    for (let i = 0; i < n; i++) {
      const t = i * REC * H;
      th.t[i] = t;
      th.a1[i] = A1 * acc(t - T0); th.a2[i] = A2 * acc(t - T0 - d); th.as[i] = th.a1[i] + th.a2[i];
      th.th1[i] = deg(A1 * baseAt(t - T0)); th.th2[i] = deg(A2 * baseAt(t - T0 - d)); th.sum[i] = th.th1[i] + th.th2[i];
    }
    // the sum of the two shifted responses against the linear model driven by the whole command
    let x = [0, 0], err = 0;
    for (let k = 0; k < Math.round(TEND / H); k++) {
      x = lin.step(x, 0.5 * (handAcc(k * H, d) + handAcc((k + 1) * H, d)));
      const t = (k + 1) * H;
      err = Math.max(err, Math.abs(x[0] - A1 * baseAt(t - T0) - A2 * baseAt(t - T0 - d)));
    }
    th.sumErr = err;
    th.off = offline(d);
    theory = th;
  }

  function go() {
    run = { delay, k: 0, nl: [0, 0], ln: [0, 0], peak: 0, resid: 0, done: false, spilled: false, trace: { t: [0], th: [0] } };
    W.clearDrops();
  }

  function step() {
    stepBoth(run, run.k * H, run.delay);
    run.k++;
    const t = run.k * H, a = Math.abs(run.nl[0]), ts = tStop(run.delay);
    run.peak = Math.max(run.peak, a);
    if (a > SPILL) run.spilled = true;
    if (t >= ts - 1e-9 && t <= ts + PERIOD + 1e-9) run.resid = Math.max(run.resid, a);
    if (t > ts + PERIOD + 1e-9) run.done = true;
    if (run.k % REC === 0 && t <= TEND + 1e-9) { run.trace.t.push(t); run.trace.th.push(deg(run.nl[0])); }
  }

  onInput(panel, 'delay', () => setDelay(val(panel, 'delay')));
  onClick(panel, 'go', go);
  onClick(panel, 'one', () => { setSlider(slider, 0); setDelay(0); go(); });
  onClick(panel, 'ideal', () => { setSlider(slider, IDEAL); setDelay(IDEAL); go(); });
  setDelay(delay);
  go();

  const SMALL = '10px Inter, system-ui, sans-serif';
  function draw() {
    P.clear();
    const g = P.g, w = P.w, h = P.h, L = 40, R = 10, iw = w - L - R;
    if (iw < 60 || h < 200) return;
    const avail = h - 24 - 36 - 50 - 40;
    const hA = Math.round(0.19 * avail), hB = Math.round(0.52 * avail), hC = avail - hA - hB;
    const yA = 24, yB = yA + hA + 36, yC = yB + hB + 50;
    const xr = [0, TEND], th = theory, ts = tStop(delay);
    const xt = [[0, '0'], [0.5, '0.5'], [1, '1.0'], [1.5, '1.5']];

    // the two half-steps of the hand's acceleration
    const A = new Ax(g, L, yA, iw, hA, xr, [-5.6, 5.6]);
    label(g, 'hand acceleration (m/s\u00b2): two half-steps', L, yA - 8);
    A.clip(() => {
      A.line(th.t, th.as, CSS.grey, 1, { dash: [3, 3] });
      A.line(th.t, th.a1, CSS.green, 1.8);
      A.line(th.t, th.a2, CSS.red, 1.8);
    });
    A.frame(xt.map(([v]) => [v, '']), [[-4, '−4'], [0, '0'], [4, '4']]);

    // response to each half-step and their sum (linear model); the robot's tilt as it runs
    const B = new Ax(g, L, yB, iw, hB, xr, [-56, 56]);
    label(g, 'first', L, yB - 8, CSS.green);
    label(g, '+ second', L + 30, yB - 8, CSS.red);
    label(g, '= sum (linear model)', L + 86, yB - 8, CSS.yellow);
    label(g, 'robot', L + iw, yB - 8, CSS.blue, 'right');
    B.clip(() => {
      g.fillStyle = 'rgba(242, 239, 233, 0.07)';
      g.fillRect(B.X(ts), B.y, B.X(ts + PERIOD) - B.X(ts), B.h);
      B.hline(deg(SPILL), CSS.red, [5, 4], 0.8);
      B.hline(-deg(SPILL), CSS.red, [5, 4], 0.8);
      B.vline(ts, CSS.grey);
      B.line(th.t, th.th1, CSS.green, 1.3);
      B.line(th.t, th.th2, CSS.red, 1.3);
      B.line(th.t, th.sum, CSS.yellow, 2.6);
      if (run && run.delay === delay) B.line(run.trace.t, run.trace.th, CSS.blue, 1.6);
    });
    label(g, 'spills', L + 4, B.Y(deg(SPILL)) - 3, CSS.red, 'left', SMALL);
    label(g, 'hand stops', Math.min(B.X(ts) + 4, L + iw - 54), yB + hB - 5, CSS.grey, 'left', SMALL);
    B.frame(xt, [[-40, '−40°'], [0, '0'], [40, '40°']]);
    label(g, 'time (s)', L + iw, yB + hB + 15, CSS.grey, 'right', '11px Inter, system-ui, sans-serif');
    if (run && run.delay === delay && run.k * H < TEND) { A.vline(run.k * H, CSS.white, [], 0.45); B.vline(run.k * H, CSS.white, [], 0.45); }

    // slosh left after the hand stops, against the delay
    const C = new Ax(g, L, yC, iw, hC, [0, 0.35], [0, 34]);
    label(g, 'slosh left after the hand stops', L, yC - 8);
    C.clip(() => {
      C.vline(IDEAL, CSS.grey, [2, 3]);
      C.line(TABLE.delays, TABLE.lin, CSS.yellow, 2);
      C.line(TABLE.delays, TABLE.nl, CSS.blue, 1.5);
      C.vline(delay, CSS.white, [], 0.8);
    });
    for (const [y, c] of [[deg(th.off.resLin), CSS.yellow], [deg(th.off.resNl), CSS.blue]]) {
      g.fillStyle = c;
      g.beginPath(); g.arc(C.X(delay), C.Y(Math.min(y, 34)), 3.5, 0, 2 * Math.PI); g.fill();
    }
    label(g, 'half a period', C.X(IDEAL) + 5, yC + 10, CSS.grey, 'left', SMALL);
    label(g, 'linear model', L + iw - 2, yC + hC - 20, CSS.yellow, 'right', SMALL);
    label(g, 'robot', L + iw - 2, yC + hC - 7, CSS.blue, 'right', SMALL);
    C.frame([[0, '0'], [0.1, '0.1'], [0.2, '0.2'], [0.3, '0.3']], [[0, '0'], [15, '15°'], [30, '30°']]);
    label(g, 'delay of the second half-step (s)', L + iw / 2, h - 6, CSS.grey, 'center', '11px Inter, system-ui, sans-serif');
  }

  const sgn = (r) => `${r >= 0 ? '+' : '−'}${Math.abs(deg(r)).toFixed(1)}°`;
  const refText = `&nbsp;&nbsp;\u00b7&nbsp;&nbsp; one step leaves <b>${deg(REF.zero.resNl).toFixed(1)}\u00b0</b>, ` +
    `the ideal delay <b>${deg(REF.ideal.resNl).toFixed(1)}\u00b0</b>`;
  let shown = '';
  W.v.onFrame((dt) => {
    const k0 = run.k;
    advance(dt, speed, step);
    const t = run.k * H;
    W.show(handPos(t, run.delay), run.nl[0], handVel(t, run.delay), (run.k - k0) * H);
    draw();
    const left = run.done ? `<b>${deg(run.resid).toFixed(1)}\u00b0</b>` : (t < tStop(run.delay) ? 'hand moving' : 'measuring');
    const s = `run at delay <b>${run.delay.toFixed(3)} s</b>: tilt <b>${sgn(run.nl[0])}</b> &nbsp; peak <b>${deg(run.peak).toFixed(1)}\u00b0</b>` +
      ` &nbsp; left after the stop: ${left}` + (run.spilled ? ' &nbsp; <b class="wbad">spilled</b>' : '') + refText;
    if (s !== shown) { shown = s; readout(panel, 'info', s); }
  });

  panel._state = () => ({
    delay, idealDelay: IDEAL, amps: ZV.amps, omega: OM, wd: WD, period: PERIOD, zeta: ZE, spillDeg: deg(SPILL), speed,
    move: { D, TM, T0 },
    run: { t: run.k * H, delay: run.delay, x: handPos(run.k * H, run.delay), tiltDeg: deg(run.nl[0]), tiltLinDeg: deg(run.ln[0]),
      peakDeg: deg(run.peak), residDeg: run.done ? deg(run.resid) : null, done: run.done, spilled: run.spilled },
    theory: Object.assign(inDeg(theory.off), { sumErrDeg: deg(theory.sumErr) }),
    ideal: inDeg(REF.ideal), zero: inDeg(REF.zero), best: inDeg(best),
    table: { delays: TABLE.delays, linDeg: TABLE.lin, nlDeg: TABLE.nl },
  });
  // for tests: tilt of both models at time t (multiple of 1 ms) for a given delay, computed offline
  panel._tiltAt = (d, t) => {
    const st = { nl: [0, 0], ln: [0, 0] };
    for (let k = 0; k < Math.round(t / H); k++) stepBoth(st, k * H, d);
    return { nlDeg: deg(st.nl[0]), linDeg: deg(st.ln[0]), x: handPos(t, d), sumDeg: deg(A1 * baseAt(t - T0) + A2 * baseAt(t - T0 - d)) };
  };
}
