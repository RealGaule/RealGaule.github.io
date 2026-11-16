// 1.3 Stability: hinge friction moves the eigenvalues of A to the left.
// Robot = nonlinear double pendulum with hinge damping d1 (upper) and d2 = RATIO * d1 (lower).
// Right: the four eigenvalues of A (roots of det(M s^2 + C s + K)) with the path they take over the slider
// range, and below it ||e^{At}||_2 against time with its bound c e^{alpha t} and the robot's own ||x(t)|| / ||x0||.
import * as S from '../sim.js';
import { makeViewer, Trail, COL, CSS, val, onInput, onClick, readout, ctl, plot2d, stepper, rad } from '../common.js';
import { setChain, showChain, dpEigs, norm2, Frame, runWhenVisible } from '../drone_util.js';

const RATIO = 0.1;                         // lower hinge friction / upper hinge friction
const RUN = 12, HOLD = 0.6, DTN = 0.05;    // seconds per release; still pose before it; sampling of the norm curve
const MIX = [12, -8];                      // release: 12 deg of mode 1 minus 8 deg of mode 2 (largest link angle of each)

export async function init(panel) {
  const v = await makeViewer(panel, 'drone.mjb', { lookat: [0, 0, 1.5], distance: 4.5, azimuth: 80, elevation: -8 });
  runWhenVisible(panel, v);
  showChain(v, 'g1', false); showChain(v, 'g2', false);
  const { V } = S.modes(S.dpMatrices());
  const shape = V.map((m) => { const s = Math.max(Math.abs(m[0]), Math.abs(m[1])); return m.map((q) => q / s); });
  const th0 = [0, 1].map((k) => rad(MIX[0]) * shape[0][k] + rad(MIX[1]) * shape[1][k]);
  const n0 = Math.hypot(th0[0], th0[1]);
  const trail = new Trail(v.scene, COL.yellow, 70);
  const advance = stepper(1e-3);
  const plot = plot2d(panel.querySelector('canvas.plot'));
  const slider = ctl(panel, 'd');
  const dMax = Number(slider.max), dStep = Number(slider.step);

  // ---- the path of the eigenvalues over the whole slider range, each branch followed by continuity
  const PERMS = [];
  (function perm(a, rest) { if (!rest.length) PERMS.push(a); rest.forEach((r, i) => perm([...a, r], rest.filter((_, j) => j !== i))); })([], [0, 1, 2, 3]);
  const eigsAt = (d) => dpEigs(S.dpMatrices(S.P, d, RATIO * d));
  function follow(prev, cur) {
    let best = null, cost = Infinity;
    for (const p of PERMS) {
      let c = 0;
      for (let i = 0; i < 4; i++) c += Math.hypot(cur[p[i]][0] - prev[i][0], cur[p[i]][1] - prev[i][1]);
      if (c < cost) { cost = c; best = p; }
    }
    return best.map((i) => cur[i]);
  }
  const locus = [eigsAt(0)];                // sorted by imaginary part: +w2, +w1, -w1, -w2
  for (let k = 1; k <= Math.round(dMax / dStep); k++) locus.push(follow(locus[k - 1], eigsAt(k * dStep)));
  const BRANCH = [CSS.red, CSS.green, CSS.green, CSS.red];
  let reMin = 0;
  for (const set of locus) for (const z of set) reMin = Math.min(reMin, z[0]);
  const reRange = [Math.floor(reMin - 0.6), 2];

  /** ||e^{A t}||_2 on the grid t = 0, DTN, ..., RUN. */
  function normCurve(A) {
    const Ad = S.expm(S.scale(A, DTN));
    let E = S.eye(4);
    const out = [];
    for (let k = 0; k <= Math.round(RUN / DTN); k++) { out.push(norm2(E)); E = S.mul(Ad, E); }
    return out;
  }
  const normMax = Math.max(...normCurve(S.stateSpace(S.dpMatrices()).A));

  let d, x, t, hold, lam, alpha, cBound, curve, trace;
  function release() { x = [th0[0], th0[1], 0, 0]; t = 0; hold = HOLD; trace = [[0, 1]]; trail.clear(); }
  function setFriction() {
    d = val(panel, 'd');
    const mats = S.dpMatrices(S.P, d, RATIO * d);
    lam = follow(locus[Math.round(d / dStep)], dpEigs(mats));
    alpha = Math.max(...lam.map((z) => z[0]));
    if (Math.abs(alpha) < 1e-9) alpha = 0;
    curve = normCurve(S.stateSpace(mats).A);
    cBound = Math.max(...curve.map((nv, k) => nv * Math.exp(-alpha * k * DTN)));
    readout(panel, 'info', alpha < 0
      ? `max Re λ = <b>${alpha.toFixed(3)}</b> · swing halves every <b>${(Math.LN2 / -alpha).toFixed(1)} s</b>`
      : 'max Re λ = <b>0</b> · <b>the swing never dies</b>');
    release();
  }
  function step(h) { x = S.dpStep(S.P, d, RATIO * d, x, 0, 0, h); t += h; }
  onInput(panel, 'd', setFriction);
  onClick(panel, 'release', release);
  setFriction();

  const place = (name, px, py) => { const el = ctl(panel, name); el.style.left = `${px}px`; el.style.top = `${py}px`; };
  const YR = [0.01, 30], lg = Math.log10;                    // the norm is drawn on a logarithmic axis
  function draw() {
    plot.clear();
    const g = plot.g, left = 44, width = plot.w - left - 16;
    const hTop = Math.round((plot.h - 122) * 0.61), hBot = plot.h - 122 - hTop;
    // ---- complex plane
    const f = new Frame(g, left, 32, width, hTop, reRange, [-5.6, 5.6]);
    g.fillStyle = 'rgba(252, 98, 85, 0.13)';
    g.fillRect(f.X(0), f.top, f.X(reRange[1]) - f.X(0), f.height);
    f.seg(reRange[0], 0, reRange[1], 0, CSS.faint);
    f.seg(0, -5.6, 0, 5.6, CSS.grey, 1.2);
    const xt = [];
    for (let r = Math.ceil(reRange[0] / 2) * 2; r <= 0; r += 2) xt.push([r, r === 0 ? '0' : `−${-r}`]);
    f.axes(xt, [[-4, '−4i'], [0, '0'], [4, '4i']], null, null);
    f.text('Re λ (1/s)', f.left + 6, f.Y(0) - 7, CSS.grey);
    f.text('unstable', f.X(0) + 5, f.Y(0) + 19, CSS.red);
    f.text('stable', f.X(0) - 6, f.Y(0) + 19, CSS.green, 'right');
    f.clip(() => {
      for (let b = 0; b < 4; b++) {
        f.line(locus.map((s) => s[b][0]), locus.map((s) => s[b][1]), BRANCH[b] === CSS.red ? 'rgba(252,98,85,0.5)' : 'rgba(131,193,103,0.5)', 1.3);
      }
      const a = f.X(alpha);
      g.save(); g.strokeStyle = CSS.accent; g.setLineDash([3, 3]); g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(a, f.top); g.lineTo(a, f.top + f.height); g.stroke(); g.restore();
      lam.forEach((z, b) => f.dot(z[0], z[1], 5, BRANCH[b], '#000', 1));
    });
    f.text('α', Math.max(f.left + 12, f.X(alpha) - 5), f.top + 12, CSS.accent, 'right');
    place('lab-eig', left + 6, 7);
    // ---- norm against time
    const top2 = 32 + hTop + 68;
    const q = new Frame(g, left, top2, width, hBot, [0, RUN], [lg(YR[0]), lg(YR[1])]);
    const ts = curve.map((_, k) => k * DTN);
    const ly = (y) => lg(Math.min(Math.max(y, YR[0] / 10), YR[1] * 10));
    q.seg(0, 0, RUN, 0, CSS.faint, 1, [2, 4]);
    q.axes([[0, '0'], [3, '3'], [6, '6'], [9, '9'], [12, '12 s']], [[-2, '0.01'], [-1, '0.1'], [0, '1'], [1, '10']], null, null);
    q.clip(() => {
      q.line(ts, ts.map((s) => ly(cBound * Math.exp(alpha * s))), CSS.accent, 1.2, [5, 4]);
      q.line(ts, curve.map(ly), CSS.white, 1.8);
      q.line(trace.map((p) => p[0]), trace.map((p) => ly(p[1])), CSS.yellow, 1.6);
      const tm = Math.min(t, RUN), k = Math.min(curve.length - 1, Math.round(tm / DTN));
      q.seg(tm, q.yr[0], tm, q.yr[1], 'rgba(242,239,233,0.35)');
      q.dot(tm, ly(curve[k]), 3.6, CSS.white, '#000', 1);
      q.dot(trace[trace.length - 1][0], ly(trace[trace.length - 1][1]), 3.6, CSS.yellow, '#000', 1);
    });
    place('lab-norm', left + 6, top2 - 43);
    place('lab-bound', left + 6, top2 - 23);
    place('lab-robot', left + width - ctl(panel, 'lab-robot').offsetWidth, top2 - 43);
  }

  v.onFrame((dt) => {
    if (hold > 0) hold -= dt;
    else {
      advance(dt, 1, step);
      trace.push([t, Math.hypot(x[0], x[1], x[2], x[3]) / n0]);
      if (t >= RUN) release();
    }
    setChain(v, 'main', x);
    v.forward();
    if (hold <= 0) trail.push(v.sitePos('main_m2'));
    draw();
  });
  panel._state = () => ({ t, d, d2: RATIO * d, x, eig: lam, alpha, cBound, normNow: curve[Math.min(curve.length - 1, Math.round(Math.min(t, RUN) / DTN))],
    norm: [1, 2, 4, 8].map((s) => curve[Math.round(s / DTN)]), normMax, ratio: Math.hypot(x[0], x[1], x[2], x[3]) / n0, th0 });
  panel._run = (T) => { hold = 0; while (t < T - 1e-9) step(1e-3); return panel._state(); };
}
