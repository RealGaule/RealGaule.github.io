// 8.1 Itô chain rule: robots released together wander by pure diffusion, dX = sqrt(2D) dW in the plane.
// Their distance r from a mark obeys dr = (D/r) dt + sqrt(2D) dW_r (Itô), so E r^2 = r0^2 + 4 D t and the mean
// distance grows although the mean position does not move. The ordinary chain rule drops D/r and predicts E r = r0.
// Same defaults as the notebook video s08_chain_rule (D = 0.004 m^2/s, dt = 0.02 s, 16 s, 20000 robots, 800 drawn).
import { rng } from '../sim3.js';
import { makeViewer, placeLamp, SwarmMesh, ledColour, Trail, THREE, COL, CSS, val, onInput, onClick, readout, seg, ctl, plot2d } from '../common3.js';
import { hudStyle, hud, sw, dot, FlatLine, Cross, Pane, legend, label, niceMax, ticks, tfmt } from '../plane_common.js';

const N_ALL = 20000, N_SHOW = 800;
const DT = 0.02;                 // s
const T_MAX = 16;                // s, the notebook's run length at D = 0.004
const R2_END = 4 * 0.004 * 16;   // a run stops when 4 D t reaches this (rms spread 0.51 m), so walls stay far
const WALL = 1.27;               // reflecting walls (table half-width minus a margin)
const PLAY = 10;                 // s of real time per run
const HL = [7, 19, 42];          // highlighted robots with tracks
const NB = 30;                   // histogram bins
const Z = 0.036;                 // height of the rings (above the robots)

// ---------------------------------------------------------------- exact distance law
/** exp(-x) I0(x) for x >= 0 (Abramowitz & Stegun 9.8.1-2). */
function i0e(x) {
  if (x < 3.75) {
    const t = (x / 3.75) ** 2;
    return Math.exp(-x) * (1 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813))))));
  }
  const t = 3.75 / x;
  return (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281 + t * (-0.02057706
    + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))))) / Math.sqrt(x);
}
/** Density of r = |X - mark| when X ~ N(start, 2Dt I) and |start - mark| = r0 (Rayleigh for r0 = 0, Rice otherwise). */
function rPdf(r, r0, D, t) {
  const s2 = 2 * D * t;
  if (r0 === 0) return r / s2 * Math.exp(-r * r / (2 * s2));
  return r / s2 * Math.exp(-((r - r0) ** 2) / (2 * s2)) * i0e(r * r0 / s2);
}
/** E r: sqrt(pi D t) for r0 = 0, numerical integral of the Rice density otherwise. */
function meanR(r0, D, t) {
  if (t <= 0) return r0;
  if (r0 === 0) return Math.sqrt(Math.PI * D * t);
  const s = Math.sqrt(2 * D * t), hi = r0 + 9 * s, n = 600, h = hi / n;
  let m = 0;
  for (let i = 1; i < n; i++) { const r = i * h; m += (i % 2 ? 4 : 2) * r * rPdf(r, r0, D, t); }
  return m * h / 3;
}

export async function init(panel) {
  hudStyle();
  const v = await makeViewer(panel, 'arena.mjb', { lookat: [0, -0.05, 0], distance: 3.35, azimuth: 90, elevation: -62 });
  for (let i = 0; i < 4; i++) placeLamp(v, i, null);
  v.forward();
  const swarm = new SwarmMesh(v.scene, N_SHOW);
  const trails = HL.map(() => new Trail(v.scene, COL.pink, 120));
  const ringMean = new FlatLine(v.scene, COL.yellow, { width: 0.014, z: Z });
  const ringRms = new FlatLine(v.scene, COL.yellow, { width: 0.011, z: Z });
  const ringOrd = new FlatLine(v.scene, COL.red, { width: 0.014, z: Z + 0.001 });
  const mark = new Cross(v.scene, COL.white, { size: 0.04, width: 0.012, z: Z + 0.002 });
  const startMark = new FlatLine(v.scene, COL.grey, { width: 0.006, z: 0.0047, maxPts: 64 });
  const plot = plot2d(panel.querySelector('canvas.plot'));
  const top = new Pane(plot), bot = new Pane(plot);

  const X = new Float64Array(N_ALL), Y = new Float64Array(N_ALL);
  const touched = new Uint8Array(N_ALL);
  const xs = new Float64Array(N_SHOW), ys = new Float64Array(N_SHOW);
  const led = ledColour(0), pink = [0xd1 / 255, 0x47 / 255, 0xbd / 255];
  let R, D, r0, tEnd, t, k, nSteps, stats, theory, ordinary = false, hist = new Float64Array(NB), Rmax;
  let seed = 8, acc = 0, mx = 0, my = 0, sr = 0, sr2 = 0, sinv = 0;

  function release() {
    D = val(panel, 'D');
    r0 = Number(getStart());
    tEnd = Math.min(T_MAX, R2_END / (4 * D));
    nSteps = Math.round(tEnd / DT);
    R = rng(seed++);
    X.fill(0); Y.fill(0); touched.fill(0);
    t = 0; k = 0; acc = 0;
    stats = { t: [0], Er: [r0], rms: [r0] };
    sr = r0; sr2 = r0 * r0; mx = 0; my = 0; sinv = 0;
    // theory curves over the whole run
    theory = { t: [], Er: [], rms: [] };
    for (let i = 0; i <= 160; i++) {
      const s = tEnd * i / 160;
      theory.t.push(s); theory.Er.push(meanR(r0, D, s)); theory.rms.push(Math.sqrt(r0 * r0 + 4 * D * s));
    }
    Rmax = niceMax(r0 + 3.3 * Math.sqrt(2 * D * tEnd));
    trails.forEach((tr) => tr.clear());
    // the mark is r0 to the left of the release point (the table centre)
    mark.at(-r0, 0);
    startMark.visible = r0 > 0;
    if (r0 > 0) startMark.circle(0, 0, 0.03, {}, 32);
    computeHist();
  }

  function step() {
    const s = Math.sqrt(2 * D * DT);
    let a = 0, b = 0, c = 0, ex = 0, ey = 0;
    for (let i = 0; i < N_ALL; i++) {
      let x = X[i] + s * R.normal(), y = Y[i] + s * R.normal();
      if (x > WALL) { x = 2 * WALL - x; touched[i] = 1; } else if (x < -WALL) { x = -2 * WALL - x; touched[i] = 1; }
      if (y > WALL) { y = 2 * WALL - y; touched[i] = 1; } else if (y < -WALL) { y = -2 * WALL - y; touched[i] = 1; }
      X[i] = x; Y[i] = y;
      const dx = x + r0, r2 = dx * dx + y * y, r = Math.sqrt(r2);
      a += r; b += r2; c += 1 / r; ex += x; ey += y;
    }
    k++; t = k * DT;
    sr = a / N_ALL; sr2 = b / N_ALL; sinv = c / N_ALL; mx = ex / N_ALL; my = ey / N_ALL;
    if (k % 20 === 0) { stats.t.push(t); stats.Er.push(sr); stats.rms.push(Math.sqrt(sr2)); }
  }

  function computeHist() {
    hist.fill(0);
    const w = Rmax / NB;
    for (let i = 0; i < N_ALL; i++) {
      const r = Math.hypot(X[i] + r0, Y[i]), j = Math.floor(r / w);
      if (j >= 0 && j < NB) hist[j]++;
    }
    for (let j = 0; j < NB; j++) hist[j] /= N_ALL * w;
  }

  // ---------------------------------------------------------------- controls
  const getStart = seg(panel, 'start', release);
  onInput(panel, 'D', release);
  onClick(panel, 'restart', release);
  onClick(panel, 'ord', () => { ordinary = !ordinary; ctl(panel, 'ord').classList.toggle('on', ordinary); });
  release();

  // ---------------------------------------------------------------- drawing
  function draw3d() {
    for (let i = 0; i < N_SHOW; i++) { xs[i] = X[i]; ys[i] = Y[i]; }
    swarm.update(xs, ys, (i) => (HL.includes(i) ? pink : led));
    if (k > 0) HL.forEach((i, j) => trails[j].push([X[i], Y[i], 0.034]));
    const Er = meanR(r0, D, t), rms = Math.sqrt(r0 * r0 + 4 * D * t);
    ringMean.visible = Er > 0.004; ringRms.visible = rms > 0.004;
    if (Er > 0.004) ringMean.circle(-r0, 0, Er);
    if (rms > 0.004) ringRms.circle(-r0, 0, rms, { dash: 0.03 });
    ringOrd.visible = ordinary;
    if (ordinary) ringOrd.circle(-r0, 0, Math.max(r0, 0.022), r0 > 0 ? { dash: 0.05 } : {});
    hud(panel, 'note', `${sw(CSS.yellow)}mean distance E r (Itô) &nbsp;${sw(CSS.yellow, true)}rms √E r² ${ordinary ? `&nbsp;${sw(CSS.red, r0 > 0)}ordinary chain rule` : ''}`
      + `<br>${dot(CSS.white)}mark (r measured from here)${r0 > 0 ? ` &nbsp;${dot(CSS.grey)}release point` : ''} &nbsp;${dot(CSS.pink)}three tracks`);
    hud(panel, 'clock', `t = <b>${t.toFixed(1)} s</b> of ${tEnd.toFixed(1)}<br>mean position (${fmtS(mx)}, ${fmtS(my)}) m`);
  }
  const fmtS = (x) => (Math.abs(x) < 0.005 ? '' : x > 0 ? '+' : '−') + Math.abs(x).toFixed(2);

  function draw2d() {
    plot.clear();
    const g = plot.g, W = plot.w, H = plot.h, L = 46, Rt = 14;
    label(g, 'Itô:  dr = (D/r) dt + √(2D) dW', W / 2, 22, CSS.yellow, { align: 'center', font: '600 14px Inter, Arial, sans-serif' });
    // histogram of r
    const h1 = Math.round(H * 0.33), y1 = 64;
    const peak = t > 0 ? pdfPeak() : 1;
    const ymax = niceMax(1.2 * Math.max(peak, ...hist), [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1]);
    top.set(L, y1, W - L - Rt, h1, 0, Rmax, 0, ymax);
    top.frame({ title: 'distance from the mark r: histogram vs exact', yticks: ticks(0, ymax, 3), fmt: tfmt,
      xticks: ticks(0, Rmax, 4), xfmt: tfmt, xlabel: 'r (m)' });
    const w = Rmax / NB, cx = Array.from({ length: NB }, (_, j) => (j + 0.5) * w);
    top.bars(cx, hist, CSS.blue, w * 0.9, { alpha: 0.7 });
    if (t > 0) {
      const rr = [], pp = [];
      for (let i = 0; i <= 200; i++) { const r = Rmax * i / 200; rr.push(r); pp.push(rPdf(r, r0, D, t)); }
      top.line(rr, pp, CSS.yellow, 2.2);
    }
    let lx = L + 4;
    lx = legend(g, lx, y1 + h1 + 42, CSS.blue, `${N_ALL} robots`, 'box');
    legend(g, lx, y1 + h1 + 42, CSS.yellow, r0 > 0 ? 'Rice density (exact)' : 'Rayleigh density (exact)');
    // distance against time
    const y2 = y1 + h1 + 84, h2 = H - y2 - 60;
    const dmax = niceMax(1.12 * Math.sqrt(r0 * r0 + 4 * D * tEnd), [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.2]);
    bot.set(L, y2, W - L - Rt, h2, 0, tEnd, 0, dmax);
    bot.frame({ title: 'mean and rms distance over time (m)', yticks: ticks(0, dmax, 4), fmt: tfmt,
      xticks: ticks(0, tEnd, 4), xfmt: (x) => tfmt(x), xlabel: 't (s)' });
    if (ordinary) bot.line([0, tEnd], [r0, r0], CSS.red, 2.2, r0 > 0 ? { dash: [7, 5] } : {});
    bot.line(theory.t, theory.rms, CSS.yellow, 2, { dash: [6, 4] });
    bot.line(theory.t, theory.Er, CSS.yellow, 2.2);
    bot.dots(stats.t, stats.rms, CSS.blue, 2.4, 0.95, { hollow: true });
    bot.dots(stats.t, stats.Er, CSS.blue, 2.2, 0.95);
    if (t > 0 && t < tEnd) bot.vline(t, CSS.grey);
    lx = L + 4;
    const yl = y2 + h2 + 42;
    lx = legend(g, lx, yl, CSS.yellow, 'E r');
    lx = legend(g, lx, yl, CSS.yellow, '√E r²', 'dash');
    lx = legend(g, lx, yl, CSS.blue, 'robots', 'dot');
    if (ordinary) legend(g, lx, yl, CSS.red, 'ordinary rule', r0 > 0 ? 'dash' : 'line');
  }
  function pdfPeak() {
    let m = 0;
    for (let i = 1; i <= 200; i++) m = Math.max(m, rPdf(Rmax * i / 200, r0, D, t));
    return m;
  }

  function text() {
    const th = r0 * r0 + 4 * D * t;
    const lhs = r0 > 0 ? `r0² + 4Dt = ${th.toFixed(3)}` : `4Dt = ${th.toFixed(3)}`;
    readout(panel, 'info', `E r² <b>${sr2.toFixed(3)}</b> m² (robots) vs <b>${lhs}</b> · E r <b>${sr.toFixed(3)}</b> vs ${meanR(r0, D, t).toFixed(3)} m`
      + (ordinary ? ` · ordinary rule: E r = <span class="c-red">${r0.toFixed(2)}</span>` : ''));
  }

  function advance(sec) {
    acc += sec;
    let n = 0;
    while (acc >= DT && k < nSteps && n < 400) { step(); acc -= DT; n++; }
    if (k >= nSteps) acc = 0;
  }

  v.onFrame((dt) => {
    advance(dt * tEnd / PLAY);
    computeHist();
    draw3d();
    draw2d();
    text();
  });

  // checks: _run(seconds of simulated time) advances without drawing
  panel._run = (sec) => { const n = Math.min(nSteps - k, Math.round(sec / DT)); for (let i = 0; i < n; i++) step(); computeHist(); return panel._state(); };
  panel._state = () => ({
    D, r0, t, t_end: tEnd, dt: DT, robots: N_ALL,
    E_r2_sim: sr2, E_r2_theory: r0 * r0 + 4 * D * t,
    E_r_sim: sr, E_r_theory: meanR(r0, D, t),
    ito_drift_D_E_inv_r: D * sinv,
    mean_position: [mx, my],
    fraction_touched_wall: touched.reduce((a, b) => a + b, 0) / N_ALL,
    ordinary_chain_rule_E_r: r0, ordinary_shown: ordinary,
  });
}
