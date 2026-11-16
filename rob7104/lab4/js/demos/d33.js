// 3.3 Steering a crane: the gantry carries the two-link load 2 m and must stop it.
// Three commands designed on the linear 6-state model (trolley + two-link load) and played on the nonlinear
// robot (dpStep), each held constant over a 1 ms step: accelerate-then-brake, the same command passed through
// a zero-vibration shaper for each of the two swing modes, and the minimum-energy input from the Gramian.
// A fourth mode lets the viewer drive the trolley with the arrow keys and get the same score.
import * as S from '../sim.js';
import { makeViewer, Trail, COL, CSS, val, ctl, onInput, onClick, readout, seg, plot2d, stepper, deg } from '../common.js';
import * as GC from '../gantry_common.js';

const D1 = 0.05, D2 = 0.03;     // hinge damping of the load
const DIST = 2.0;               // travel (m)
const H = 1e-3;                 // integration step (s)
const RES_WIN = 2.6;            // the residual sway is the largest sway within this long after arrival (s)
const TAIL = 0.4;               // playback goes on this long after the window
// drive it yourself
const A_KEY = 0.8;              // acceleration while a key is held (m/s^2)
const K_BRAKE = 2.5;            // with no key held the trolley brakes gently: u = -K_BRAKE v, at most A_KEY
const SOFT_LO = -0.25, SOFT_HI = 2.25, K_BUF = 10, C_BUF = 6.3;   // spring buffers at the ends of the rail
const TOL_P = 0.05, TOL_V = 0.01;   // "parked": within 5 cm of the mark, slower than 1 cm/s, no key held
const KINDS = ['bang', 'shaped', 'minE'];
const NAME = { bang: 'accelerate, brake', shaped: 'input shaping', minE: 'minimum energy', drive: 'you' };
const TINT = { bang: CSS.red, shaped: CSS.blue, minE: CSS.green, drive: CSS.yellow };
const TINT3 = { bang: COL.red, shaped: COL.blue, minE: COL.green, drive: COL.yellow };
const KEYMAP = { ArrowLeft: 'left', a: 'left', A: 'left', ArrowRight: 'right', d: 'right', D: 'right' };

// complex numbers as [re, im], for the damped modes
const cadd = (a, b) => [a[0] + b[0], a[1] + b[1]];
const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const cdiv = (a, b) => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };

/** Natural frequencies and damping ratios: roots of det(M s^2 + C s + K) = 0, Newton from the undamped ones. */
function dampedModes({ M, C, K }) {
  const out = { w: [], zeta: [] };
  for (const w0 of S.modes({ M, K }).w) {
    let s = [0, w0];
    for (let it = 0; it < 20; it++) {
      const s2 = cmul(s, s);
      const a = (i, j) => [M[i][j] * s2[0] + C[i][j] * s[0] + K[i][j], M[i][j] * s2[1] + C[i][j] * s[1]];
      const da = (i, j) => [2 * M[i][j] * s[0] + C[i][j], 2 * M[i][j] * s[1]];
      const det = cadd(cmul(a(0, 0), a(1, 1)), cmul(a(0, 1), a(1, 0)).map((x) => -x));
      const ddet = cadd(cadd(cmul(da(0, 0), a(1, 1)), cmul(a(0, 0), da(1, 1))),
        cadd(cmul(da(0, 1), a(1, 0)), cmul(a(0, 1), da(1, 0))).map((x) => -x));
      const step = cdiv(det, ddet);
      s = [s[0] - step[0], s[1] - step[1]];
    }
    const mag = Math.hypot(s[0], s[1]);
    out.w.push(mag); out.zeta.push(-s[0] / mag);
  }
  return out;
}

export async function init(panel) {
  const v = await makeViewer(panel, 'gantry_crane.mjb', { lookat: [1.0, 0, 1.52], distance: 4.7, azimuth: 80, elevation: -5 });
  const trail = new Trail(v.scene, COL.red, 90);
  const targetLine = GC.plumb(v.scene, GC.PIV_CRANE, 0xffffff, 0.35);
  targetLine.position.x = DIST;
  const underTrolley = GC.plumb(v.scene, GC.PIV_CRANE, 0x9a97a3, 0.35);
  const plot = plot2d(panel.querySelector('canvas.plot'));
  const pane = new GC.Pane(plot);
  const advance = stepper(H);

  // ---------------------------------------------------------------- model
  const mats = S.dpMatrices(S.P, D1, D2);
  const sys = S.withTrolley(S.stateSpace(mats));
  const rank = S.rank(S.ctrb(sys.A, sys.B));
  const md = dampedModes(mats);
  const shapers = md.w.map((w, i) => S.zvShaper(w, md.zeta[i]));
  const shLen = shapers[0].times[1] + shapers[1].times[1];
  const lin = new S.LinearSim(sys.A, sys.B, H);
  const stepA = S.expm(S.scale(sys.A, H)), halfA = S.expm(S.scale(sys.A, H / 2));
  const xf = [DIST, 0, 0, 0, 0, 0];
  const offset = (th1, th2) => S.P.l1 * Math.sin(th1) + S.P.l2 * Math.sin(th2);   // load relative to the trolley (m)

  /** One command for travel time T: its samples at the step midpoints, then played on the robot and scored. */
  function command(kind, T) {
    const arrive = kind === 'shaped' ? T + shLen : T;
    const n = Math.round((arrive + RES_WIN + TAIL) / H);
    const u = new Float64Array(n);
    const extra = {};
    if (kind === 'bang') {
      for (let k = 0; k < n; k++) u[k] = S.bangBangAccel(DIST, T, (k + 0.5) * H);
    } else if (kind === 'shaped') {
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
        const amp = shapers[0].amps[i] * shapers[1].amps[j], delay = shapers[0].times[i] + shapers[1].times[j];
        for (let k = 0; k < n; k++) u[k] += amp * S.bangBangAccel(DIST, T, (k + 0.5) * H - delay);
      }
      extra.least_energy_for_its_time = S.minEnergy(sys.A, sys.B, [0, 0, 0, 0, 0, 0], xf, arrive, 800).energy;
    } else {
      const me = S.minEnergy(sys.A, sys.B, [0, 0, 0, 0, 0, 0], xf, T, 800);
      let b = S.mulVec(halfA, sys.B);                      // e^{A (T - t)} B at the last midpoint, then backwards
      for (let k = Math.round(T / H) - 1; k >= 0; k--) { u[k] = S.dot(b, me.lambda); b = S.mulVec(stepA, b); }
      extra.energy_closed_form = me.energy;
    }
    // robot (nonlinear) and linear model side by side
    const p = new Float64Array(n + 1), th1 = new Float64Array(n + 1), th2 = new Float64Array(n + 1);
    let x = [0, 0, 0, 0], xl = [0, 0, 0, 0, 0, 0], pp = 0, vv = 0;
    const kA = Math.round(arrive / H), kW = Math.min(n, kA + Math.round(RES_WIN / H));
    let energy = 0, peak = 0, sway = 0, swayLin = 0, maxAng = 0, pArr = 0, vArr = 0;
    for (let k = 0; k <= n; k++) {
      p[k] = pp; th1[k] = x[0]; th2[k] = x[1];
      maxAng = Math.max(maxAng, Math.abs(x[0]), Math.abs(x[1]));
      if (k === kA) { pArr = pp; vArr = vv; }
      if (k >= kA && k <= kW) {
        sway = Math.max(sway, Math.abs(offset(x[0], x[1])));
        swayLin = Math.max(swayLin, Math.abs(S.P.l1 * xl[2] + S.P.l2 * xl[3]));
      }
      if (k === n) break;
      const a = u[k];
      energy += a * a * H; peak = Math.max(peak, Math.abs(a));
      x = S.dpStep(S.P, D1, D2, x, a, a, H);
      xl = lin.step(xl, a);
      pp += vv * H + 0.5 * a * H * H; vv += a * H;
    }
    return Object.assign({ kind, T, arrive, n, u, p, th1, th2, energy, peak, sway_cm: 100 * sway, sway_lin_cm: 100 * swayLin,
      max_angle_deg: deg(maxAng), p_arrive: pArr, v_arrive: vArr }, extra);
  }

  // ---------------------------------------------------------------- state
  let kind = 'bang', T = val(panel, 'T');
  let cmds = {}, board = {}, uLim = 1;
  let play = { tp: 0, running: false };
  let drv = null;
  const keys = { left: false, right: false };
  let visible = false, listening = false;

  function design() {
    T = val(panel, 'T');
    cmds = {};
    for (const k of KINDS) cmds[k] = command(k, T);
    uLim = GC.niceLimit(1.08 * Math.max(...KINDS.map((k) => cmds[k].peak)), [0.5, 0.75, 1, 1.5, 2, 3, 4]);
    for (const k of KINDS) delete board[k];
    reset();
  }

  function newDrive() {
    return { t: 0, p: 0, v: 0, u: 0, x: [0, 0, 0, 0], started: false, t0: 0, energy: 0, peak: 0, arr: null, sway: 0, result: null,
      rec: new GC.Record(1, 1e9), nstep: 0 };
  }

  function reset() {
    play = { tp: 0, running: false };
    drv = newDrive();
    trail.clear();
    trail.color.set(TINT3[kind]);
  }

  function go() {
    reset();
    if (kind !== 'drive') play.running = true;
  }

  function driveStep(h) {
    const d = drv, c = (keys.right ? 1 : 0) - (keys.left ? 1 : 0);
    let a = c !== 0 ? c * A_KEY : -Math.max(-A_KEY, Math.min(A_KEY, K_BRAKE * d.v));
    if (d.p > SOFT_HI) a += -K_BUF * (d.p - SOFT_HI) - C_BUF * d.v;
    if (d.p < SOFT_LO) a += -K_BUF * (d.p - SOFT_LO) - C_BUF * d.v;
    if (!d.started && c !== 0) { d.started = true; d.t0 = d.t; }
    d.x = S.dpStep(S.P, D1, D2, d.x, a, a, h);
    d.p += d.v * h + 0.5 * a * h * h; d.v += a * h;
    d.u = a; d.t += h; d.nstep++;
    if (d.started && !d.result) {
      d.energy += a * a * h; d.peak = Math.max(d.peak, Math.abs(a));
      const off = Math.abs(offset(d.x[0], d.x[1]));
      if (d.arr === null) {
        if (c === 0 && Math.abs(d.p - DIST) <= TOL_P && Math.abs(d.v) <= TOL_V) { d.arr = d.t; d.sway = off; d.eArr = d.energy; }
      } else if (c !== 0) {
        d.arr = null;                                       // moved again: not parked yet
      } else {
        d.sway = Math.max(d.sway, off);
        if (d.t - d.arr >= RES_WIN) {
          d.result = { arrive: d.arr - d.t0, sway_cm: 100 * d.sway, energy: d.eArr, peak: d.peak, miss_cm: 100 * (d.p - DIST) };
          board.drive = d.result;
        }
      }
    }
    if (d.nstep % 20 === 0 && d.started) d.rec.push(d.t - d.t0, a);
  }

  // ---------------------------------------------------------------- keyboard (only while this panel is visible and in drive mode)
  function setKey(side, down) { keys[side] = down; ctl(panel, side).classList.toggle('held', down); }
  const typing = (e) => { const el = e.target; return el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName) && el.type !== 'range'); };
  const onDown = (e) => { const s = KEYMAP[e.key]; if (!s || typing(e) || e.ctrlKey || e.metaKey || e.altKey) return; e.preventDefault(); if (!e.repeat) setKey(s, true); };
  const onUp = (e) => { const s = KEYMAP[e.key]; if (s) setKey(s, false); };
  function listen() {
    const want = visible && kind === 'drive';
    if (want === listening) return;
    listening = want;
    if (want) { window.addEventListener('keydown', onDown); window.addEventListener('keyup', onUp); }
    else { window.removeEventListener('keydown', onDown); window.removeEventListener('keyup', onUp); setKey('left', false); setKey('right', false); }
  }
  window.addEventListener('blur', () => { setKey('left', false); setKey('right', false); });
  for (const side of ['left', 'right']) {
    const b = ctl(panel, side);
    b.addEventListener('pointerdown', (e) => { e.preventDefault(); b.setPointerCapture(e.pointerId); setKey(side, true); });
    for (const ev of ['pointerup', 'pointercancel', 'lostpointercapture']) b.addEventListener(ev, () => setKey(side, false));
  }

  // ---------------------------------------------------------------- controls
  const getKind = seg(panel, 'cmd', (x) => {
    kind = x;
    const drive = kind === 'drive';
    ctl(panel, 'T').disabled = drive;
    ctl(panel, 'keys').classList.toggle('on', drive);
    ctl(panel, 'go').textContent = drive ? 'start over' : 'go';
    if (drive && document.activeElement && document.activeElement.blur) document.activeElement.blur();
    reset();
    listen();
  });
  kind = getKind();
  onInput(panel, 'T', design);
  onClick(panel, 'go', go);
  GC.watchVisible(panel, (vis) => { visible = vis; listen(); });
  design();

  // ---------------------------------------------------------------- pose, text, 2D picture
  const f2 = (x) => x.toFixed(2);
  function pose() {
    let p, th1, th2;
    if (kind === 'drive') { p = drv.p; th1 = drv.x[0]; th2 = drv.x[1]; }
    else { const c = cmds[kind], k = Math.min(c.n, Math.round(play.tp / H)); p = c.p[k]; th1 = c.th1[k]; th2 = c.th2[k]; }
    v.setMocap('trolley', [p, 0, GC.PIV_CRANE]);
    v.setJoint('c_q1', th1);
    v.setJoint('c_q2', th2 - th1);
    underTrolley.position.x = p;
    v.forward();
    if (kind === 'drive' ? drv.started : play.tp > 0) trail.push(v.sitePos('c_m2'));
    return { p, th1, th2 };
  }

  function text(now) {
    GC.hud(panel, 'rank', `trolley + two-link load: 6 states<br>controllability rank <b class="big ${rank === 6 ? 'g-ok' : 'g-bad'}">${rank} of 6</b>`);
    const sway = 100 * offset(now.th1, now.th2);
    let s;
    if (kind === 'drive') {
      const d = drv;
      GC.hud(panel, 'note', `<span class="g-key">←</span> <span class="g-key">→</span> accelerate (also A, D) · no key: gentle brake · park under the white mark`);
      if (d.result) {
        s = `<span class="g-ok">parked</span> after <b>${d.result.arrive.toFixed(1)} s</b>, ${Math.abs(d.result.miss_cm).toFixed(1)} cm off the mark · residual sway <b>±${d.result.sway_cm.toFixed(1)} cm</b> · energy <b>${f2(d.result.energy)}</b> m²/s³ · peak <b>${f2(d.result.peak)}</b> m/s²`;
      } else if (!d.started) {
        s = 'hold <b>→</b> to set off; the clock starts with the first key';
      } else {
        s = `${(d.t - d.t0).toFixed(1)} s · trolley at <b>${f2(d.p)} m</b> of ${DIST.toFixed(2)} · speed ${f2(d.v)} m/s · sway ${sway.toFixed(1)} cm · energy so far <b>${f2(d.energy)}</b> m²/s³` + (d.arr !== null ? ' · <span class="g-ok">parked, watching the sway</span>' : '');
      }
    } else {
      const c = cmds[kind];
      GC.hud(panel, 'note', `t = <b>${Math.min(play.tp, c.n * H).toFixed(1)} s</b> · load <b>${sway >= 0 ? '+' : '−'}${Math.abs(sway).toFixed(1)} cm</b> from the point under the trolley`);
      const when = kind === 'shaped' ? `arrives at <b>${f2(c.arrive)} s</b> = ${c.T.toFixed(1)} s + ${f2(shLen)} s of shaper` : `arrives at <b>${f2(c.arrive)} s</b>`;
      if (play.tp < c.arrive) {
        s = `${when} · energy <b>${f2(c.energy)}</b> m²/s³ · peak <b>${f2(c.peak)}</b> m/s²` + (kind === 'minE' ? ' · the least energy of any input that does it in this time' : '');
      } else {
        const k1 = Math.min(c.n, Math.round(play.tp / H)), kA = Math.round(c.arrive / H);
        let m = 0;
        for (let k = kA; k <= Math.min(k1, kA + Math.round(RES_WIN / H)); k += 5) m = Math.max(m, Math.abs(offset(c.th1[k], c.th2[k])));
        const done = play.tp >= c.arrive + RES_WIN;
        const sw = done ? c.sway_cm : 100 * m;
        s = `arrived at <b>${f2(c.arrive)} s</b> · residual sway <b class="${sw < 1 ? 'g-ok' : 'g-bad'}">±${sw.toFixed(1)} cm</b>${done ? '' : ' so far'} · energy <b>${f2(c.energy)}</b> m²/s³ · peak <b>${f2(c.peak)}</b> m/s²`
          + (kind === 'shaped' && done ? ` · least possible for ${f2(c.arrive)} s: ${f2(c.least_energy_for_its_time)}` : '');
      }
    }
    readout(panel, 'info', s);
  }

  function draw() {
    plot.clear();
    const g = plot.g, W = plot.w, Hh = plot.h, left = 40, right = 12;
    const h1 = Math.round(Hh * 0.38);
    if (kind === 'drive') {
      const d = drv, tt = d.started ? d.t - d.t0 : 0, tMax = Math.max(8, Math.ceil(tt + 1));
      pane.set(left, 28, W - left - right, h1, 0, tMax, -1.1, 1.1);
      pane.frame({ title: 'your trolley acceleration u(t)', yticks: [-0.8, 0, 0.8], fmt: (x) => x.toFixed(1), xticks: [0, tMax / 2, tMax], xfmt: (x) => x + ' s' });
      pane.line(d.rec.t, d.rec.s[0], TINT.drive, 1.8);
    } else {
      const c = cmds[kind], tMax = Math.ceil(2 * (T + shLen + 0.3)) / 2;
      pane.set(left, 28, W - left - right, h1, 0, tMax, -uLim, uLim);
      pane.frame({ title: 'trolley acceleration u(t)  (m/s²)', yticks: [-uLim, 0, uLim], fmt: (x) => String(x), xticks: [0, tMax / 2, tMax], xfmt: (x) => x + ' s' });
      const kEnd = Math.min(c.n, Math.round(tMax / H)), ts = [], us = [];
      for (let k = 0; k < kEnd; k += 10) { ts.push((k + 0.5) * H); us.push(c.u[k]); }
      pane.line(ts, us, CSS.grey, 1.5, { alpha: 0.6 });
      const iNow = Math.min(ts.length, Math.ceil(play.tp / (10 * H)));
      pane.line(ts, us, TINT[kind], 2.4, { i1: iNow });
      pane.vline(c.arrive, CSS.white, { alpha: 0.7 });
      GC.label(g, 'arrives', pane.X(c.arrive) - 4, 28 + 12, CSS.grey, { align: 'right' });
      if (play.tp > 0 && play.tp < tMax) pane.vline(play.tp, TINT[kind], { dash: [], alpha: 0.5 });
    }
    // scoreboard
    const y0 = 28 + h1 + 44, rowH = 25;
    const cols = [left - 28, W - 178, W - 122, W - 66, W - right];
    g.save(); g.strokeStyle = CSS.faint; g.beginPath(); g.moveTo(cols[0], y0 + 21.5); g.lineTo(W - right, y0 + 21.5); g.stroke(); g.restore();
    GC.label(g, `score at ${T.toFixed(1)} s travel time`, cols[0], y0 - 14, CSS.white, { font: GC.FONT_B });
    const head = [['arrives', 's'], ['sway', '± cm'], ['energy', 'm²/s³'], ['peak', 'm/s²']];
    head.forEach(([a, b], i) => {
      GC.label(g, a, cols[i + 1], y0 + 4, CSS.grey, { align: 'right' });
      GC.label(g, b, cols[i + 1], y0 + 16, CSS.faint, { align: 'right', font: '10px Inter, Arial, sans-serif' });
    });
    [...KINDS, 'drive'].forEach((k, i) => {
      const y = y0 + 22 + (i + 1) * rowH - 8, r = board[k];
      g.save(); g.fillStyle = TINT[k]; g.fillRect(cols[0], y - 10, 3, 13); g.restore();
      GC.label(g, NAME[k], cols[0] + 9, y, k === kind ? CSS.white : CSS.grey, { font: k === kind ? GC.FONT_B : GC.FONT });
      if (r) {
        [r.arrive.toFixed(2), r.sway_cm.toFixed(1), r.energy.toFixed(2), r.peak.toFixed(2)].forEach((txt, j) => GC.label(g, txt, cols[j + 1], y, CSS.white, { align: 'right', font: GC.MONO }));
      } else {
        GC.label(g, k === 'drive' ? 'not driven yet' : 'not run yet', W - right, y, CSS.faint, { align: 'right' });
      }
    });
    GC.label(g, 'sway: farthest the load gets from under the trolley', cols[0], Hh - 22, CSS.faint);
    GC.label(g, `within ${RES_WIN} s after arrival · energy = ∫u² dt`, cols[0], Hh - 8, CSS.faint);
  }

  function tick(sec) {
    if (kind === 'drive') return;
    if (play.running) {
      const c = cmds[kind];
      const before = play.tp;
      play.tp += sec;
      if (before < c.arrive + RES_WIN && play.tp >= c.arrive + RES_WIN) board[kind] = { arrive: c.arrive, sway_cm: c.sway_cm, energy: c.energy, peak: c.peak };
      if (play.tp >= c.n * H) { play.tp = c.n * H; play.running = false; }
    }
  }

  v.onFrame((dt) => {
    if (kind === 'drive') advance(dt, 1, driveStep); else tick(dt);
    const now = pose();
    text(now);
    draw();
  });

  // numbers for the checks; _run(seconds) advances the playback or the driving simulation without drawing
  panel._run = (sec) => {
    if (kind === 'drive') { const n = Math.round(sec / H); for (let i = 0; i < n; i++) driveStep(H); } else tick(sec);
    return panel._state();
  };
  panel._key = (side, down) => setKey(side, down);
  const brief = (c) => { const o = {}; for (const k of Object.keys(c)) if (!(c[k] instanceof Float64Array)) o[k] = c[k]; return o; };
  panel._state = () => ({
    kind, T, distance: DIST, rank, residual_window_s: RES_WIN,
    modes: md, shaper: { amps: shapers.map((s) => s.amps), times: shapers.map((s) => s.times), length: shLen },
    cmds: Object.fromEntries(KINDS.map((k) => [k, brief(cmds[k])])),
    board, play: { t: play.tp, running: play.running },
    drive: { t: drv.t, p: drv.p, v: drv.v, u: drv.u, th_deg: [deg(drv.x[0]), deg(drv.x[1])], started: drv.started, energy: drv.energy, peak: drv.peak,
      parked_at: drv.arr, result: drv.result },
    listening, keys: { ...keys },
  });
}
