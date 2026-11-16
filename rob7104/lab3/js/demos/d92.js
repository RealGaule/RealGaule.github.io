// 9.2 Four lamps: robots drift up the light and wander at random, dX = -grad U dt + sqrt(2T) dW with U = -kappa I
// (Langevin dynamics, Euler-Maruyama, reflecting walls). The stationary density is Boltzmann, p ~ e^{-U/T}, so the
// share of robots under each lamp tends to the integral of e^{kappa I / T} over that lamp's pool. Annealing lowers
// T geometrically from T_hi to T_lo; slow cooling gathers the swarm under the brightest lamp, fast cooling freezes
// it where it is. Defaults as the notebook video s09_four_lamps.
import { rng, Lamps2D } from '../sim3.js';
import { makeViewer, placeLamp, SwarmMesh, ledColour, LightPool, THREE, COL, CSS, val, onInput, onClick, readout, seg, ctl, plot2d } from '../common3.js';
import { hudStyle, hud, sw, dot, FlatLine, Cross, Pane, legend, label, floorLabel, niceMax, ticks, tfmt, budget } from '../plane_common.js';

const CENTRES = [[-0.62, -0.6], [0.62, -0.6], [-0.62, 0.6], [0.62, 0.6]];
const BRIGHT = [1.0, 0.6, 0.45, 0.3];
const WIDTH = 0.26;                 // m
const LABELS = ['100%', '60%', '45%', '30%', 'dark'];
const POOL_R = 0.45;                // "under lamp k" = within 0.45 m of its centre
const WALL = 1.27;
const T0 = 0.004;                   // m^2/s, default randomness
const KAPPA = 0.02;                 // m^2/s, phototaxis gain: U = -KAPPA I (= 5 T0)
const DT = 0.05;                    // s
const T_HI = 0.008, T_LO = 0.0015;  // annealing range
const N_ALL = 4000, N_SHOW = 600;
const REC = 5;                      // s between recorded points
const LAMPS = new Lamps2D(CENTRES, BRIGHT, WIDTH);

function region(x, y) {
  let best = 4, dmin = POOL_R * POOL_R;
  for (let k = 0; k < 4; k++) { const dx = x - CENTRES[k][0], dy = y - CENTRES[k][1], d = dx * dx + dy * dy; if (d < dmin) { dmin = d; best = k; } }
  return best;
}

// Boltzmann shares on a grid over the table (midpoint rule), cached per temperature
const NG = 261, GRID = (() => {
  const reg = new Uint8Array(NG * NG), lu = new Float64Array(NG * NG), h = 2 * WALL / NG;
  for (let i = 0; i < NG; i++) for (let j = 0; j < NG; j++) {
    const x = -WALL + (i + 0.5) * h, y = -WALL + (j + 0.5) * h;
    reg[i * NG + j] = region(x, y); lu[i * NG + j] = LAMPS.I(x, y);
  }
  return { reg, lu };
})();
const bzCache = new Map();
function boltzmann(T) {
  const key = T.toPrecision(6);
  if (bzCache.has(key)) return bzCache.get(key);
  const w = [0, 0, 0, 0, 0];
  let s = 0;
  for (let i = 0; i < NG * NG; i++) { const e = Math.exp(KAPPA * (GRID.lu[i] - 1) / T); w[GRID.reg[i]] += e; s += e; }
  const out = w.map((x) => x / s);
  if (bzCache.size > 400) bzCache.clear();
  bzCache.set(key, out);
  return out;
}

export async function init(panel) {
  hudStyle();
  const v = await makeViewer(panel, 'arena.mjb', { lookat: [0, -0.12, 0], distance: 3.75, azimuth: 90, elevation: -60 });
  CENTRES.forEach((c, i) => placeLamp(v, i, c));
  v.forward();
  // the lamp fixtures should not throw shadows of the viewer's sun onto the table
  const lampBodies = new Set(CENTRES.map((_, i) => v.id('body', `lamp${i}`)));
  for (let gi = 0; gi < v.model.ngeom; gi++) if (lampBodies.has(v.model.geom_bodyid[gi]) && v.meshes[gi]) v.meshes[gi].castShadow = false;
  CENTRES.forEach((c, i) => {
    const pool = new LightPool(v.scene, c[0], c[1], WIDTH);
    pool.setLevel(BRIGHT[i]);
    pool.mesh.position.z = 0.0045;               // just above the table top (z = 0.002), avoids z-fighting
  });
  CENTRES.forEach((c, i) => floorLabel(v.scene, `lamp ${LABELS[i]}`, c[0], c[1] + 0.52, CSS.yellow, { height: 0.1, z: 0.005 }));
  const poolRings = CENTRES.map((c) => { const f = new FlatLine(v.scene, COL.grey, { width: 0.005, z: 0.005, maxPts: 200, opacity: 0.45 }); f.circle(c[0], c[1], POOL_R, { dash: 0.04 }); return f; });
  const dropRing = new FlatLine(v.scene, COL.pink, { width: 0.012, z: 0.036, maxPts: 80 });
  const dropCross = new Cross(v.scene, COL.pink, { size: 0.035, width: 0.012, z: 0.037 });
  const swarm = new SwarmMesh(v.scene, N_SHOW);
  const plot = plot2d(panel.querySelector('canvas.plot'));
  const top = new Pane(plot), bot = new Pane(plot);

  const X = new Float64Array(N_ALL), Y = new Float64Array(N_ALL);
  const xs = new Float64Array(N_SHOW), ys = new Float64Array(N_SHOW), cols = new Array(N_SHOW);
  let R, T, t, k, mode, tau, frac, rec, drop, results = {}, seed = 91, acc = 0, lastRate = 0;

  // ---------------------------------------------------------------- dynamics
  function Tnow(s) { return mode === 'anneal' ? T_HI * Math.pow(T_LO / T_HI, Math.min(s / tau, 1)) : T; }
  function step() {
    const Ts = Tnow((k + 0.5) * DT), sg = Math.sqrt(2 * Ts * DT), a = KAPPA * DT;
    const c0 = CENTRES[0], c1 = CENTRES[1], c2 = CENTRES[2], c3 = CENTRES[3], iw2 = 1 / (WIDTH * WIDTH), h2 = -0.5 * iw2;
    for (let i = 0; i < N_ALL; i++) {
      const x = X[i], y = Y[i];
      let gx = 0, gy = 0, dx, dy, e;
      dx = x - c0[0]; dy = y - c0[1]; e = BRIGHT[0] * Math.exp(h2 * (dx * dx + dy * dy)); gx -= e * dx; gy -= e * dy;
      dx = x - c1[0]; dy = y - c1[1]; e = BRIGHT[1] * Math.exp(h2 * (dx * dx + dy * dy)); gx -= e * dx; gy -= e * dy;
      dx = x - c2[0]; dy = y - c2[1]; e = BRIGHT[2] * Math.exp(h2 * (dx * dx + dy * dy)); gx -= e * dx; gy -= e * dy;
      dx = x - c3[0]; dy = y - c3[1]; e = BRIGHT[3] * Math.exp(h2 * (dx * dx + dy * dy)); gx -= e * dx; gy -= e * dy;
      // grad I = (gx, gy) / w^2; drift -grad U = KAPPA grad I
      let nx = x + a * iw2 * gx + sg * R.normal(), ny = y + a * iw2 * gy + sg * R.normal();
      if (nx > WALL) nx = 2 * WALL - nx; else if (nx < -WALL) nx = -2 * WALL - nx;
      if (ny > WALL) ny = 2 * WALL - ny; else if (ny < -WALL) ny = -2 * WALL - ny;
      X[i] = nx; Y[i] = ny;
    }
    k++; t = k * DT;
    if (k % Math.round(REC / DT) === 0) { const f = fractions(); rec.t.push(t); rec.f0.push(f[0]); rec.bz0.push(boltzmann(Tnow(t))[0]); }
    if (mode === 'anneal' && !results[tau]?.done && t >= tau - 1e-9) {
      frac = fractions();
      results[tau] = { done: true, under_brightest: frac[0], fractions: frac.slice(), boltzmann_T_lo: boltzmann(T_LO) };
    }
  }
  function fractions() {
    const f = [0, 0, 0, 0, 0];
    for (let i = 0; i < N_ALL; i++) f[region(X[i], Y[i])]++;
    return f.map((c) => c / N_ALL);
  }
  function scatter(cx, cy) {
    for (let i = 0; i < N_ALL; i++) {
      let x = cx + 0.04 * R.normal(), y = cy + 0.04 * R.normal();
      X[i] = Math.max(-WALL, Math.min(WALL, x)); Y[i] = Math.max(-WALL, Math.min(WALL, y));
    }
  }
  function begin(newMode) {
    mode = newMode;
    R = rng(seed++);
    drop = [val(panel, 'dx'), val(panel, 'dy')];
    scatter(drop[0], drop[1]);
    t = 0; k = 0; acc = 0;
    if (mode === 'anneal') { tau = Number(getTau()); delete results[tau]; }
    frac = fractions();
    rec = { t: [0], f0: [frac[0]], bz0: [boltzmann(Tnow(0))[0]] };
    ctl(panel, 'T').disabled = mode === 'anneal';
    ctl(panel, 'anneal').classList.toggle('on', mode === 'anneal');
  }

  // ---------------------------------------------------------------- controls
  function showDrop() {
    const x = val(panel, 'dx'), y = val(panel, 'dy');
    dropRing.circle(x, y, 0.07, {}, 48); dropCross.at(x, y);
  }
  function setT(x) {
    const el = ctl(panel, 'T');
    el.value = x;
    const out = el.parentElement.querySelector('output');
    if (out) out.textContent = Number(x).toFixed(4) + (el.dataset.unit || '');
  }
  onInput(panel, 'T', () => {
    T = val(panel, 'T');
    if (mode === 'anneal') mode = 'fixed';          // (the slider is disabled while cooling; kept for safety)
  });
  onInput(panel, 'dx', showDrop);
  onInput(panel, 'dy', showDrop);
  onClick(panel, 'drop', () => { T = val(panel, 'T'); begin('fixed'); });
  onClick(panel, 'anneal', () => begin('anneal'));
  onClick(panel, 'restart', () => {
    ctl(panel, 'dx').value = CENTRES[3][0]; ctl(panel, 'dy').value = CENTRES[3][1];
    ['dx', 'dy'].forEach((n) => ctl(panel, n).dispatchEvent(new Event('input', { bubbles: true })));
    setT(T0); T = T0;
    begin('fixed');
  });
  const getTau = seg(panel, 'tau', () => {});
  const getSpeed = seg(panel, 'speed', () => {});
  T = val(panel, 'T');
  showDrop();
  begin('fixed');

  // ---------------------------------------------------------------- drawing
  function draw3d() {
    for (let i = 0; i < N_SHOW; i++) { xs[i] = X[i]; ys[i] = Y[i]; cols[i] = ledColour(LAMPS.I(X[i], Y[i]) / 0.8); }
    swarm.update(xs, ys, (i) => cols[i]);
    const Tc = Tnow(t);
    hud(panel, 'clock', `t = <b>${t.toFixed(0)} s</b> · running at ${Math.round(lastRate)} s per second<br>`
      + `T = <b>${Tc.toFixed(4)}</b> m²/s${mode === 'anneal' ? (t < tau ? ` · <span style="color:${CSS.accent}">cooling</span>, ${Math.round(100 * t / tau)}% done` : ' · cooled') : ''}`);
    hud(panel, 'note', `${dot(CSS.yellow)}light pools, brightness 100 / 60 / 45 / 30% &nbsp;${sw(CSS.grey, true)}counted as under a lamp<br>${dot(CSS.pink)}drop point &nbsp;LEDs: ${dot(CSS.blue)}dark ${dot(CSS.yellow)}lit`);
  }

  function draw2d() {
    plot.clear();
    const g = plot.g, W = plot.w, H = plot.h, L = 44, Rt = 14;
    label(g, 'dX = −∇U dt + √(2T) dW,   U = −κ I', W / 2, 22, CSS.yellow, { align: 'center', font: '600 14px Inter, Arial, sans-serif' });
    const Tc = Tnow(t), bz = boltzmann(Tc);
    // bars
    const y1 = 66, h1 = Math.round(H * 0.36);
    top.set(L, y1, W - L - Rt, h1, -0.6, 4.6, 0, 1);
    top.frame({ title: `share of robots under each lamp, T = ${Tc.toFixed(4)}`, yticks: [0, 0.25, 0.5, 0.75, 1], fmt: tfmt });
    const xsB = [0, 1, 2, 3, 4];
    top.bars(xsB, frac, CSS.blue, 0.56, { alpha: 0.8 });
    top.bars(xsB, bz, CSS.yellow, 0.78, { alpha: 1, outline: true, lw: 2 });
    xsB.forEach((i) => {
      label(g, frac[i].toFixed(2), top.X(i), Math.max(y1 + 13, top.Y(frac[i]) - 6), CSS.white, { align: 'center' });
      label(g, LABELS[i], top.X(i), y1 + h1 + 15, i < 4 ? CSS.white : CSS.grey, { align: 'center' });
    });
    let lx = L;
    lx = legend(g, lx, y1 + h1 + 38, CSS.blue, `${N_ALL} robots`, 'box');
    legend(g, lx, y1 + h1 + 38, CSS.yellow, 'Boltzmann ∝ e^(−U/T)', 'outline');
    // share under the brightest lamp over time
    const y2 = y1 + h1 + 82, h2 = H - y2 - 64;
    const tmax = mode === 'anneal' ? Math.max(tau * 1.15, t) : niceMax(Math.max(400, 1.05 * t), [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1]);
    bot.set(L, y2, W - L - Rt, h2, 0, tmax, 0, 1);
    bot.frame({ title: 'share under the brightest lamp', yticks: [0, 0.5, 1], fmt: tfmt, xticks: ticks(0, tmax, 4), xfmt: tfmt, xlabel: 't (s)' });
    bot.line(rec.t, rec.bz0, CSS.yellow, 1.8, { dash: [6, 4] });
    bot.line(rec.t, rec.f0, CSS.blue, 2.2);
    if (mode === 'anneal') { bot.vline(tau, CSS.white, { alpha: 0.6 }); label(g, 'cooling ends', bot.X(tau) - 4, y2 + 13, CSS.grey, { align: 'right' }); }
    const yl = y2 + h2 + 40;
    lx = L;
    lx = legend(g, lx, yl, CSS.blue, 'robots');
    legend(g, lx, yl, CSS.yellow, 'Boltzmann at the current T', 'dash');
    // anneal results
    const done = Object.keys(results).map(Number).filter((k) => results[k].done).sort((a, b) => a - b);
    if (done.length) label(g, 'cooled over ' + done.map((d) => `${d} s → ${results[d].under_brightest.toFixed(2)}`).join(' · '), L, yl + 18, CSS.white);
  }

  function text() {
    const bz = boltzmann(Tnow(t));
    const gap = Math.max(...frac.map((f, i) => Math.abs(f - bz[i])));
    readout(panel, 'info', `largest gap to Boltzmann <b>${gap.toFixed(3)}</b> · Boltzmann under the brightest <b>${bz[0].toFixed(3)}</b>`
      + (mode === 'anneal' && results[tau]?.done ? ` · cooled over ${tau} s: <b>${results[tau].under_brightest.toFixed(2)}</b> under the brightest` : ''));
  }

  v.onFrame((dt) => {
    const sp = getSpeed(), fast = sp === 'max';
    acc += fast ? 1e9 : dt * Number(sp);
    const n = Math.floor(acc / DT);
    const done = budget(n, fast ? 24 : 14, step);
    acc = n > done ? 0 : acc - done * DT;          // if the budget ran out, drop the backlog (the run just goes slower)
    lastRate = 0.9 * lastRate + 0.1 * (done * DT / Math.max(dt, 1e-3));
    frac = fractions();
    if (mode === 'anneal') setT(Tnow(t));
    ctl(panel, 'T').disabled = mode === 'anneal' && t < tau;
    draw3d();
    draw2d();
    text();
  });

  panel._run = (sec) => { const n = Math.round(sec / DT); for (let i = 0; i < n; i++) step(); frac = fractions(); return panel._state(); };
  panel._begin = (m) => begin(m);
  panel._state = () => {
    const Tc = Tnow(t), bz = boltzmann(Tc);
    return { t, dt: DT, mode, T: Tc, kappa: KAPPA, tau: mode === 'anneal' ? tau : null, drop, robots: N_ALL,
      fractions: frac, boltzmann: bz, max_gap: Math.max(...frac.map((f, i) => Math.abs(f - bz[i]))), anneal_results: results };
  };
}
