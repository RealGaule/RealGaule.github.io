// 2.1 A sum of kicks: a Franka arm moves a glass of water sideways; the viewer drives the hand with a slider.
// The robot is the nonlinear slosh pendulum. The theory picture cuts the hand's acceleration into slices,
// shows the decaying slosh each slice starts (linear model), and their running sum on top of the linear model.
import * as S from '../sim.js';
import { CSS, val, onInput, onClick, readout, ctl, seg, plot2d, stepper, deg } from '../common.js';
import { OM, ZE, PERIOD, SPILL, PEND, PINK, loadWaiter, freeResponse, amplitude, setSlider, Ax, label, symTicks } from '../waiter_common.js';

const H = 1e-3;                       // integration step (s)
const NS = 60;                        // steps per slice: slices are 0.06 s wide (0.22 slosh periods)
const SUB = 3;                        // a slice keeps every 3rd sample of its own response while it is being driven
const REC = 4;                        // history is recorded every 4 ms
const WIN = 2.0;                      // the theory picture shows the last 2 s
const LIMIT = 1.45;                   // the pendulum is stopped at 83 degrees (far beyond spilling at 40.6)
const X0 = -0.15;
// standard move, as in the video scene s21_kicks: push, coast, brake; 0.30 m in 0.93 s
const STD = { D: 0.30, T0: 0.10, TL: 0.42, TR: 0.03, TC: 0.09 };
STD.A0 = STD.D / ((STD.TL - STD.TR) * (STD.TL + STD.TC));
STD.TMOVE = 2 * STD.TL + STD.TC;
const lobe = (s) => (s < 0 || s > STD.TL ? 0 : Math.min(1, Math.min(s, STD.TL - s) / STD.TR));
const stdAccel = (t) => STD.A0 * (lobe(t - STD.T0) - lobe(t - STD.T0 - STD.TL - STD.TC));
// the slider is a target; the hand follows it as x'' = wf^2 (target - x) - 2 zeta wf x', target speed limited
const FOLLOW = { wf: 12, zeta: 1, vmax: 1.0 };

class KickSim {
  constructor() {
    const { wf, zeta } = FOLLOW;
    this.lin = new S.LinearSim(PEND.A, PEND.B, H);
    this.fol = new S.LinearSim([[0, 1], [-wf * wf, -2 * zeta * wf]], [0, wf * wf], H);
    this.reset(X0);
  }

  reset(x) {
    this.k = 0;
    this.x = x; this.v = 0; this.a = 0; this.xt = x; this.target = x;
    this.mode = 'manual'; this.kStd = 0;
    this.nl = [0, 0];                 // the robot: nonlinear pendulum
    this.ln = [0, 0];                 // linear model driven by the whole acceleration
    this.peak = 0; this.tPeak = 0; this.peakLin = 0; this.spilled = false;
    this.slices = []; this.cur = null;
    this.old = [0, 0];                // sum of the slices that are no longer drawn
    this.sum = 0; this.sumErr = 0;
    this.hist = { t: [], a: [], lin: [], nl: [], sum: [] };
  }

  get t() { return this.k * H; }
  startStd() { this.reset(X0); this.mode = 'std'; }
  setTarget(x) {
    if (this.mode === 'std') { this.mode = 'manual'; this.xt = this.x; }
    this.target = x;
  }

  step() {
    let a0, a1;
    if (this.mode === 'std') {
      const ts = this.kStd * H;
      a0 = stdAccel(ts); a1 = stdAccel(ts + H);
      const v1 = this.v + 0.5 * (a0 + a1) * H;
      this.x += 0.5 * (this.v + v1) * H; this.v = v1;
      this.kStd++;
      if (this.kStd * H > STD.T0 + STD.TMOVE + 1e-9) { this.mode = 'manual'; this.v = 0; this.xt = this.target = this.x; }
    } else {
      const { wf, zeta, vmax } = FOLLOW;
      this.xt += Math.max(-vmax * H, Math.min(vmax * H, this.target - this.xt));
      a0 = wf * wf * (this.xt - this.x) - 2 * zeta * wf * this.v;
      [this.x, this.v] = this.fol.step([this.x, this.v], this.xt);
      a1 = wf * wf * (this.xt - this.x) - 2 * zeta * wf * this.v;
    }
    const um = 0.5 * (a0 + a1);
    this.a = a1;
    this.nl = S.pendStep(OM, ZE, this.nl, a0, a1, H);
    if (Math.abs(this.nl[0]) > LIMIT) {
      const sg = Math.sign(this.nl[0]);
      this.nl[0] = sg * LIMIT;
      if (this.nl[1] * sg > 0) this.nl[1] = 0;
    }
    this.ln = this.lin.step(this.ln, um);
    // the slice that is being driven now; all earlier slices move freely
    if (!this.cur) {
      this.cur = { idx: this.k / NS, k0: this.k, n: 0, sum: 0, mean: 0, s: [0, 0], live: false, done: false,
        samp: new Float32Array(NS / SUB + 1) };
      this.slices.push(this.cur);
    }
    const c = this.cur;
    c.s = this.lin.step(c.s, um);
    c.sum += um; c.n++; c.mean = c.sum / c.n;
    if (um !== 0) c.live = true;
    if (c.n % SUB === 0) c.samp[c.n / SUB] = c.s[0];
    this.old = this.lin.step(this.old, 0);
    this.k++;
    if (this.k % NS === 0) {
      c.done = true; c.kEnd = this.k; c.amp = amplitude(c.s);
      if (!c.live) this.slices.pop();
      this.cur = null;
    }
    const th = Math.abs(this.nl[0]);
    if (th > this.peak) { this.peak = th; this.tPeak = this.t; }
    this.peakLin = Math.max(this.peakLin, Math.abs(this.ln[0]));
    if (th > SPILL) this.spilled = true;
    if (this.k % REC === 0) this.record();
  }

  /** Sum of all slice responses now, kept apart from the direct linear solution so the two can be compared. */
  record() {
    const t = this.t;
    let sum = this.old[0];
    for (let i = this.slices.length - 1; i >= 0; i--) {
      const s = this.slices[i];
      if (!s.done) { sum += s.s[0]; continue; }
      const age = (this.k - s.kEnd) * H;
      const now = freeResponse(s.s, age);
      if (age > 6 || s.amp * Math.exp(-ZE * OM * age) < 0.003) {          // faded: fold it into `old`
        this.old = [this.old[0] + now[0], this.old[1] + now[1]];
        this.slices.splice(i, 1);
      }
      sum += now[0];
    }
    this.sum = sum;
    this.sumErr = Math.max(this.sumErr, Math.abs(sum - this.ln[0]));
    const h = this.hist;
    h.t.push(t); h.a.push(this.a); h.lin.push(this.ln[0]); h.nl.push(this.nl[0]); h.sum.push(sum);
    if (h.t.length > 2 * WIN / (REC * H)) {
      const drop = h.t.length - Math.round(1.2 * WIN / (REC * H));
      for (const key in h) h[key].splice(0, drop);
    }
  }
}

/** The standard move run offline with the same code as the live demo; numbers for checking. */
function standardFacts() {
  const o = new KickSim();
  o.startStd();
  const at = {};
  for (let k = 1; k <= 2300; k++) {
    o.step();
    if (k === 500 || k === 1000 || k === 1500) at[k / 1000] = { nlDeg: deg(o.nl[0]), linDeg: deg(o.ln[0]), sumDeg: deg(o.sum) };
  }
  return { tiltAt: at, peakNlDeg: deg(o.peak), tPeakNl: o.tPeak, peakLinDeg: deg(o.peakLin), travel: o.x - X0,
    sumErrDeg: deg(o.sumErr), plateauAccel: STD.A0, duration: STD.TMOVE, spilled: o.spilled };
}

export async function init(panel) {
  const W = await loadWaiter(panel);
  const P = plot2d(panel.querySelector('canvas.plot'));
  const hand = ctl(panel, 'hand');
  const sim = new KickSim();
  const std = standardFacts();
  const advance = stepper(H);
  let speed = 0.25;
  seg(panel, 'speed', (s) => { speed = Number(s); });
  onInput(panel, 'hand', () => sim.setTarget(val(panel, 'hand') / 100));
  onClick(panel, 'std', () => { W.clearDrops(); sim.startStd(); });
  onClick(panel, 'reset', () => {
    const x = sim.x, v = sim.v, xt = sim.xt, target = sim.target, mode = sim.mode, kStd = sim.kStd;
    sim.reset(x);                                                        // water flat, plots cleared; the hand keeps going
    Object.assign(sim, { v, xt, target, mode, kStd });
    W.clearDrops();
  });
  sim.startStd();

  const sliceColor = (s) => (s.mean >= 0 ? (s.idx % 2 ? CSS.green : CSS.blue) : (s.idx % 2 ? PINK : CSS.red));
  const fmt = (r) => `${r >= 0 ? '+' : '−'}${Math.abs(deg(r)).toFixed(1)}°`;

  function draw() {
    P.clear();
    const g = P.g, w = P.w, h = P.h, L = 40, R = 10, iw = w - L - R;
    if (iw < 60 || h < 200) return;
    const t = sim.t, t0 = Math.max(0, t - WIN), xr = [t0, t0 + WIN], hi = sim.hist;
    let i0 = 0;
    while (i0 < hi.t.length && hi.t[i0] < t0 - 2 * REC * H) i0++;
    let amax = 0, thmax = 0;
    for (let i = i0; i < hi.t.length; i++) {
      amax = Math.max(amax, Math.abs(hi.a[i]));
      thmax = Math.max(thmax, Math.abs(hi.lin[i]), Math.abs(hi.nl[i]));
    }
    const ar = Math.max(2.2, 1.2 * amax), thr = Math.min(95, Math.max(14, 1.18 * deg(thmax)));
    const avail = h - 24 - 34 - 34 - 38;
    const hA = Math.round(0.25 * avail), hR = Math.round(0.36 * avail), hS = avail - hA - hR;
    const yA = 24, yR = yA + hA + 34, yS = yR + hR + 34;
    const xt = [];
    for (let v = Math.ceil(t0 / 0.5 - 1e-9) * 0.5; v <= xr[1] + 1e-9; v += 0.5) xt.push([v, v.toFixed(1)]);
    const xtBlank = xt.map(([v]) => [v, '']);
    const degs = hi.t.slice(i0);

    // hand acceleration: slice bars and the acceleration itself
    const A = new Ax(g, L, yA, iw, hA, xr, [-ar, ar]);
    label(g, 'hand acceleration (m/s\u00b2), cut into slices', L, yA - 8);
    A.clip(() => {
      for (const s of sim.slices) {
        const ta = s.k0 * H, tb = s.done ? s.kEnd * H : t;
        if (tb < xr[0] || Math.abs(s.mean) < 1e-6) continue;
        const xa = A.X(ta), xb = A.X(tb), y0 = A.Y(0), y1 = A.Y(s.mean);
        g.fillStyle = sliceColor(s);
        g.globalAlpha = s.done ? 0.9 : 0.55;
        g.fillRect(xa + 0.5, Math.min(y0, y1), Math.max(1, xb - xa - 1), Math.abs(y1 - y0));
      }
      g.globalAlpha = 1;
      A.line(degs, hi.a.slice(i0), CSS.grey, 1.2);
    });
    A.frame(xtBlank, symTicks(ar));

    // the slosh each slice starts (linear model driven by that slice alone)
    const Rr = new Ax(g, L, yR, iw, hR, xr, [-thr, thr]);
    label(g, 'each slice starts its own slosh', L, yR - 8);
    Rr.clip(() => {
      g.lineWidth = 1.1; g.lineJoin = 'round';
      for (const s of sim.slices) {
        if (s.done && s.amp < 0.0035) continue;
        const ta = s.k0 * H;
        g.strokeStyle = sliceColor(s);
        g.globalAlpha = 0.8;
        g.beginPath();
        g.moveTo(Rr.X(ta), Rr.Y(0));
        const nIn = Math.floor(s.n / SUB);
        for (let j = 1; j <= nIn; j++) g.lineTo(Rr.X(ta + j * SUB * H), Rr.Y(deg(s.samp[j])));
        if (s.done) {
          const te = s.kEnd * H;
          for (let tau = Math.max(0, xr[0] - te); te + tau <= t; tau += 0.006) g.lineTo(Rr.X(te + tau), Rr.Y(deg(freeResponse(s.s, tau)[0])));
          g.lineTo(Rr.X(t), Rr.Y(deg(freeResponse(s.s, t - te)[0])));
        }
        g.stroke();
      }
      g.globalAlpha = 1;
    });
    Rr.frame(xtBlank, symTicks(thr, '°'));

    // running sum of the slices on the linear model, and the robot
    const Sx = new Ax(g, L, yS, iw, hS, xr, [-thr, thr]);
    label(g, 'sum of slices', L, yS - 8, CSS.yellow);
    label(g, 'linear model', L + 88, yS - 8, CSS.white);
    label(g, 'robot', L + iw, yS - 8, CSS.blue, 'right');
    Sx.clip(() => {
      Sx.hline(deg(SPILL), CSS.red, [5, 4], 0.8);
      Sx.hline(-deg(SPILL), CSS.red, [5, 4], 0.8);
      Sx.line(degs, hi.lin.slice(i0).map(deg), CSS.white, 5, { alpha: 0.85 });
      Sx.line(degs, hi.sum.slice(i0).map(deg), CSS.yellow, 2);
      Sx.line(degs, hi.nl.slice(i0).map(deg), CSS.blue, 1.5);
    });
    if (thr > deg(SPILL)) label(g, 'spills', L + 4, Sx.Y(deg(SPILL)) - 3, CSS.red, 'left', '10px Inter, system-ui, sans-serif');
    Sx.frame(xt, symTicks(thr, '°'));
    label(g, 'time (s)', L + iw / 2, h - 6, CSS.grey, 'center', '11px Inter, system-ui, sans-serif');
    if (t < xr[1]) for (const ax of [A, Rr, Sx]) ax.vline(t, CSS.white, [], 0.45);
  }

  let shown = '';
  W.v.onFrame((dt) => {
    const k0 = sim.k, wasStd = sim.mode === 'std';
    advance(dt, speed, () => sim.step());
    if (wasStd) setSlider(hand, (sim.x * 100).toFixed(1));
    W.show(sim.x, sim.nl[0], sim.v, (sim.k - k0) * H);
    draw();
    const s = `tilt <b>${fmt(sim.nl[0])}</b> &nbsp; peak <b>${deg(sim.peak).toFixed(1)}°</b> &nbsp; ` +
      (sim.spilled ? '<b class="wbad">spilled</b>' : '<b class="wok">not spilled</b>');
    if (s !== shown) { shown = s; readout(panel, 'info', s); }
  });

  panel._state = () => ({
    t: sim.t, mode: sim.mode, speed, x: sim.x, v: sim.v, a: sim.a, target: sim.target,
    tiltDeg: deg(sim.nl[0]), tiltLinDeg: deg(sim.ln[0]), sumDeg: deg(sim.sum), peakDeg: deg(sim.peak), peakLinDeg: deg(sim.peakLin),
    spilled: sim.spilled, sumErrDeg: deg(sim.sumErr), nSlices: sim.slices.length,
    omega: OM, period: PERIOD, zeta: ZE, spillDeg: deg(SPILL), follower: FOLLOW, sliceWidth: NS * H, std,
  });
  // for tests: run a slider gesture offline (target positions in metres, one per millisecond) and return the tilts
  panel._gesture = (targets) => {
    const o = new KickSim();
    o.reset(targets[0]);
    const out = [];
    targets.forEach((x, k) => { o.setTarget(x); o.step(); if ((k + 1) % 100 === 0) out.push([o.x, o.a, o.nl[0], o.ln[0], o.sum]); });
    return { peakDeg: deg(o.peak), peakLinDeg: deg(o.peakLin), spilled: o.spilled, sumErrDeg: deg(o.sumErr), samples: out };
  };
}
