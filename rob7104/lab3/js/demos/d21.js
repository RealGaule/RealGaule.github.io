// 2.2 Ito or Stratonovich: robots that slow down in light, v(x) = v0 (1 - s I(x)), flip direction at rate a.
// Three corridors: (1) the run-and-tumble robots, (2) dX = sqrt(2 D(X)) dW, D = v^2 / (2a), read as Ito
// (Euler-Maruyama), (3) read as Stratonovich (Heun). Walls: histogram, 1/v (yellow), 1/D (pink dashed).
// Same model and defaults as video s02_interpretation (v0 = 1 m/s, s = 0.6, a = 4 /s, lamp width 0.3 m).
import { rng, Lamps1D, emStep, heunStep, runTumbleStep } from '../sim3.js';
import { makeViewer, placeLamp, SwarmMesh, DensityWall, LightPool, ledColour, COL, CSS, ctl, val, onInput, onClick, seg, readout } from '../common3.js';
import { Ribbon, Tags, binDensity, noLampShadows } from '../noise_kit.js';

const V0 = 1.0, LAMP_W = 0.30;
const LO = -1.47, HI = 1.47;                 // reflecting walls for robot centres (corridor 3 m)
const DT = 1 / 240;                          // s, as in the video
const NPOP = 2500, NDRAW = 300;
const NB = 30, BW = (HI - LO) / NB;
const SAMPLE = 24;                           // steps between histogram samples (0.1 s)
const WINDOW = 60;                           // samples averaged for the gap readout (6 s)
const BUDGET = 10;                           // ms of simulation per animation frame at most
const WALL = { dy: 0.30, z0: 0.06, height: 0.40 };
const ROWS = [
  { key: 'robots', y: 0.9, name: '1 · real robots', sub: 'run and tumble', col: CSS.blue },
  { key: 'ito', y: 0.0, name: '2 · Itô SDE', sub: 'Euler–Maruyama', col: CSS.white },
  { key: 'strat', y: -0.9, name: '3 · Stratonovich SDE', sub: 'Heun', col: CSS.white },
];
const lamp = new Lamps1D([0], [1], LAMP_W);

export async function init(panel) {
  const v = await makeViewer(panel, 'corridor3.mjb', { lookat: [0, 0.12, 0.05], distance: 4.55, azimuth: 90, elevation: -43, fov: 40 });
  ROWS.forEach((r, i) => placeLamp(v, i, [0, r.y], 2.7));     // high, so the fixtures do not hide the far walls
  v.forward();
  noLampShadows(v);

  const R = rng(21);
  let s = val(panel, 's'), a = val(panel, 'a'), speed = 3;
  const vel = (x) => V0 * (1 - s * lamp.I(x));
  // the steppers call v(x) and g(x) = sqrt(2 D(x)) = v(x) / sqrt(a) for every robot at every step: use tables
  // (linear interpolation on a 1.2 mm grid; relative error below 1e-5)
  const LN = 4096, L0 = LO - 0.6, LH = (HI - LO + 1.2) / (LN - 1);
  const vTab = new Float64Array(LN), gTab = new Float64Array(LN);
  const lut = (T) => (x) => {
    let u = (x - L0) / LH;
    if (u < 0) u = 0; else if (u > LN - 1.0001) u = LN - 1.0001;
    const i = u | 0;
    return T[i] + (u - i) * (T[i + 1] - T[i]);
  };
  const vFast = lut(vTab), gFast = lut(gTab);
  function fillTables() { for (let i = 0; i < LN; i++) { const x = L0 + i * LH; vTab[i] = vel(x); gTab[i] = vel(x) / Math.sqrt(a); } }
  const zero = () => 0;

  // populations
  const pop = {}, dir = new Float64Array(NPOP);
  ROWS.forEach((r) => { pop[r.key] = new Float64Array(NPOP); });
  const ry = ROWS.map(() => Float64Array.from({ length: NDRAW }, () => (R.uni() - 0.5) * 2 * (0.16 - 0.035)));

  // scene objects
  const pools = ROWS.map((r) => new LightPool(v.scene, 0, r.y, LAMP_W));
  const walls = ROWS.map((r) => new DensityWall(v.scene, { lo: LO, hi: HI, nb: NB, y0: r.y + WALL.dy, z0: WALL.z0, height: WALL.height, vmax: 1, curveColours: [] }));
  const curveV = ROWS.map(() => new Ribbon(v.scene, COL.yellow, { width: 0.013 }));
  const curveD = ROWS.map(() => new Ribbon(v.scene, COL.pink, { width: 0.010, dash: 0.035 }));
  const swarms = ROWS.map(() => new SwarmMesh(v.scene, NDRAW));
  const tags = new Tags(panel.querySelector('.cell'), v);
  ROWS.forEach((r) => {
    r.tagL = tags.add([LO - 0.07, r.y + 0.1, 0.12], { anchor: 'right', valign: 'middle' });
    r.tagR = tags.add([HI + 0.07, r.y + 0.1, 0.12], { anchor: 'left', valign: 'middle' });
    tags.text(r.tagL, `<b style="color:${r.col}">${r.name}</b><br><span style="color:${CSS.grey}">${r.sub}</span>`);
  });

  // predictions (bin averages for the gaps; fine curves for the walls)
  let pred = null;
  const xg = Float64Array.from({ length: 241 }, (_, i) => LO + (HI - LO) * i / 240);
  function setModel() {
    s = val(panel, 's'); a = val(panel, 'a');
    fillTables();
    const bv = binDensity((x) => 1 / vel(x), LO, HI, NB), bd = binDensity((x) => 1 / vel(x) ** 2, LO, HI, NB);
    const cv = Float64Array.from(xg, (x) => 1 / vel(x) / bv.total), cd = Float64Array.from(xg, (x) => 1 / vel(x) ** 2 / bd.total);
    const vmax = Math.max(0.8, 1.08 * Math.max(...cd));
    pred = { binV: bv.bins, binD: bd.bins, cv, cd, vmax, maxV: Math.max(...bv.bins), maxD: Math.max(...bd.bins) };
    const sc = WALL.height / vmax;
    ROWS.forEach((r, i) => {
      walls[i].vmax = vmax;
      curveV[i].set(xg, Float64Array.from(cv, (p) => WALL.z0 + p * sc), r.y + WALL.dy - 0.008);
      curveD[i].set(xg, Float64Array.from(cd, (p) => WALL.z0 + p * sc), r.y + WALL.dy - 0.012);
    });
    clearAverage();
  }

  // time-averaged histograms (ring of the last WINDOW samples) for the gap readout
  const ring = ROWS.map(() => new Float64Array(WINDOW * NB));
  let nSamp = 0, sinceChange = 0;
  function clearAverage() { nSamp = 0; sinceChange = 0; }
  function counts(x, out) {
    out.fill(0);
    for (let i = 0; i < x.length; i++) { const k = Math.floor((x[i] - LO) / BW); if (k >= 0 && k < NB) out[k]++; }
    for (let k = 0; k < NB; k++) out[k] /= x.length * BW;
    return out;
  }
  const tmp = new Float64Array(NB);
  function sample() {
    const slot = nSamp % WINDOW;
    ROWS.forEach((r, i) => ring[i].set(counts(pop[r.key], tmp), slot * NB));
    nSamp++;
  }
  function averaged(i) {
    const n = Math.min(nSamp, WINDOW), h = new Float64Array(NB);
    for (let j = 0; j < n; j++) for (let k = 0; k < NB; k++) h[k] += ring[i][j * NB + k];
    for (let k = 0; k < NB; k++) h[k] /= n || 1;
    return h;
  }
  const gap = (h, p, pmax) => { let m = 0; for (let k = 0; k < NB; k++) m = Math.max(m, Math.abs(h[k] - p[k])); return m / pmax; };

  let t = 0, steps = 0;
  function restart() {
    for (let i = 0; i < NPOP; i++) {
      const x0 = LO + (HI - LO) * R.uni();
      ROWS.forEach((r) => { pop[r.key][i] = x0; });
      dir[i] = R.uni() < 0.5 ? -1 : 1;
    }
    t = 0; steps = 0;
    clearAverage();
  }

  function step() {
    runTumbleStep(pop.robots, dir, vFast, a, DT, R, LO, HI);
    emStep(pop.ito, zero, gFast, DT, R, LO, HI);
    heunStep(pop.strat, zero, gFast, DT, R, LO, HI);
    t += DT; steps++; sinceChange += DT;
    if (steps % SAMPLE === 0) sample();
  }

  onInput(panel, 's', setModel);
  onInput(panel, 'a', setModel);
  seg(panel, 'speed', (x) => { speed = Number(x); });
  onClick(panel, 'restart', restart);
  setModel();
  restart();

  const hist = new Float64Array(NB);
  const ys = ROWS.map((r, i) => Float64Array.from(ry[i], (u) => r.y + u));
  let gaps = null, rate = { sim: 0, wall: 0, shown: speed };
  function show() {
    gaps = {};
    ROWS.forEach((r, i) => {
      const x = pop[r.key], xs = x.subarray(0, NDRAW);
      swarms[i].update(xs, ys[i], (k) => ledColour(lamp.I(xs[k])), NDRAW);
      walls[i].update(counts(x, hist));
      const h = averaged(i), gv = gap(h, pred.binV, pred.maxV), gd = gap(h, pred.binD, pred.maxD);
      gaps[r.key] = { toV: gv, toD: gd, peakOverEdge: (h[NB / 2 - 1] + h[NB / 2]) / (h[0] + h[NB - 1]) };
      const bv = gv <= gd ? 'font-weight:700' : 'opacity:0.75', bd = gd < gv ? 'font-weight:700' : 'opacity:0.75';
      tags.text(r.tagR, nSamp ? `<span style="color:${CSS.yellow};${bv}">gap to 1/v ${gv.toFixed(2)}</span><br>`
        + `<span style="color:${CSS.pink};${bd}">gap to 1/D ${gd.toFixed(2)}</span>` : `<span style="color:${CSS.grey}">collecting…</span>`);
    });
    const avgS = Math.min(nSamp, WINDOW) * SAMPLE * DT;
    const slow = rate.shown < 0.8 * speed ? `<span style="color:${CSS.red}">&ensp;(this computer manages ${rate.shown.toFixed(1)}×)</span>` : '';
    readout(panel, 'info', `t = <b>${t.toFixed(1)} s</b>${slow}&ensp;·&ensp;gaps: histogram averaged over the last <b>${avgS.toFixed(1)} s</b>`);
  }

  v.onFrame((dt) => {
    const t0 = performance.now(), want = dt * speed;
    let n = 0, need = Math.round(want / DT);
    while (n < need) { step(); n++; if ((n & 7) === 0 && performance.now() - t0 > BUDGET) break; }
    rate.sim += n * DT; rate.wall += dt;
    if (rate.wall > 1) { rate.shown = rate.sim / rate.wall; rate.sim = 0; rate.wall = 0; }
    show();
    tags.update();
  });
  show();

  /** Test hook: simulate `sec` seconds at once (no drawing). */
  panel._advance = (sec) => { const n = Math.round(sec / DT); for (let i = 0; i < n; i++) step(); show(); return t; };
  panel._state = () => {
    const c40 = [-0.5, 0.5].map((k) => k * (HI - LO) / 40), e40 = [LO + 0.5 * (HI - LO) / 40, HI - 0.5 * (HI - LO) / 40];
    const ratio = (q) => (q(c40[0]) + q(c40[1])) / (q(e40[0]) + q(e40[1]));
    return { t, s, a, speed, robotsPerCorridor: NPOP, dt: DT, bins: NB, averagedSeconds: Math.min(nSamp, WINDOW) * SAMPLE * DT,
      sinceChange, gaps, predPeakOverEdge40bins: { ito: ratio((x) => 1 / vel(x) ** 2), strat: ratio((x) => 1 / vel(x)) },
      predGapBetweenModels: gap(pred.binV, pred.binD, pred.maxD) };
  };
}
