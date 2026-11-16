// 4.1 Two lamps: robots with phototaxis dX = kappa I'(X) dt + sigma dW under two lamps at -0.75 and +0.75 m (a double
// well). Wall: the robots' histogram (blue), the Fokker-Planck solution (cyan) and the stationary density
// p ~ exp(2 kappa I / sigma^2) (pink, dashed). Above each half: the share of robots / Fokker-Planck / stationary.
// Same parameters as the notebook clip s04_fokker_planck.
import { rng, Lamps1D, FP1D, histogram } from '../sim3.js';
import { makeViewer, placeLamp, SwarmMesh, ledColour, LightPool, DensityWall, COL, CSS, val, onInput, onClick, readout, seg } from '../common3.js';
import { Overlay, Ribbon, wallCurve, binAverage, robotY, legendItem, bulbLevel } from '../dens_common.js';

const CX = 0.75, WL = 0.35, KAPPA = 0.08;
const DT = 1 / 60;                       // Euler-Maruyama step (s), as in the clip
const NPOP = 4000, NDRAW = 400;
const LO = -1.47, HI = 1.47, NB = 36, NCELL = 360;
const FP_H = 0.05;                       // largest implicit Fokker-Planck step (s)
const YW = 0.32, ZW = 0.06, HW = 0.6, VMAX = 2.0;
const LAMP_Z = 0.98;

export async function init(panel) {
  const v = await makeViewer(panel, 'corridor.mjb', { lookat: [0, 0.18, 0.36], distance: 2.75, azimuth: 90, elevation: -18 });
  placeLamp(v, 0, [-CX, 0], LAMP_Z); placeLamp(v, 1, null); placeLamp(v, 2, [CX, 0], LAMP_Z);
  v.forward();
  const lamps = new Lamps1D([-CX, CX], [0.8, 0.8], WL);
  const pools = [new LightPool(v.scene, -CX, 0, 0.26), new LightPool(v.scene, CX, 0, 0.26)];
  const swarm = new SwarmMesh(v.scene, NDRAW);
  const wall = new DensityWall(v.scene, { lo: LO, hi: HI, nb: NB, y0: YW, z0: ZW, height: HW, vmax: VMAX, curveColours: [] });
  const fpCurve = new Ribbon(v.scene, COL.cyan, { y0: YW - 0.008, width: 0.013 });
  const stCurve = new Ribbon(v.scene, COL.pink, { y0: YW - 0.012, width: 0.011, dash: 0.035 });
  const ov = new Overlay(panel, v);
  ov.box('legend', `${legendItem(CSS.blue, `histogram of all ${NPOP} robots (${NDRAW} drawn)`, 'bar')}<br>` +
    `${legendItem(CSS.cyan, 'Fokker–Planck solution')}<br>${legendItem(CSS.pink, 'stationary  p ∝ exp(2κI/σ²)', 'dash')}`);
  const corner = ov.box('corner');
  const side = [ov.label([-CX, YW, ZW + HW + 0.1]), ov.label([CX, YW, ZW + HW + 0.1])];

  const p = { b: [0.8, 0.8], sigma: 0.2 };
  const f = (x) => KAPPA * lamps.dI(x);
  const g = () => p.sigma;
  const fp = new FP1D(f, g, LO, HI, NCELL);
  const probe = new FP1D(f, g, LO, HI, NCELL);          // for the slowest relaxation time
  const hx = (HI - LO) / NCELL, xs = Array.from(fp.x);
  const ry = robotY(NDRAW, rng(41));
  let X, R, t, acc, speed = 4, pst = new Float64Array(NCELL), tauHop = NaN;

  /** Slowest relaxation time of the density (hops between the lamps): inverse iteration with implicit Euler steps. */
  function hopTime() {
    probe.setModel(f, g);
    const H = 10;
    let q = Float64Array.from(xs, (x) => x);                 // any start with zero mass after deflation
    let rate = 0;
    for (let it = 0; it < 60; it++) {
      let m = 0; for (let i = 0; i < NCELL; i++) m += q[i] * hx;
      for (let i = 0; i < NCELL; i++) q[i] -= m * pst[i];     // remove the stationary part (mass is conserved)
      let nrm = 0; for (let i = 0; i < NCELL; i++) nrm += q[i] * q[i];
      nrm = Math.sqrt(nrm);
      for (let i = 0; i < NCELL; i++) q[i] /= nrm;
      probe.p = q; probe.step(H); q = probe.p;
      let dot = 0; for (let i = 0; i < NCELL; i++) dot += q[i] * q[i];
      const mu = Math.sqrt(dot);                               // ~ 1 / (1 - H lambda2)
      if (Math.abs(mu - rate) < 1e-10) break;
      rate = mu;
    }
    return H / (1 / rate - 1);                                 // -1 / lambda2
  }

  function readParams() {
    p.b = [val(panel, 'bl'), val(panel, 'br')]; p.sigma = val(panel, 'sigma');
    lamps.b = p.b.slice();
    fp.setModel(f, g);
    // stationary density exp(2 kappa I / sigma^2), normalised on the grid
    const e = xs.map((x) => 2 * KAPPA * lamps.I(x) / p.sigma ** 2), m = Math.max(...e);
    let s = 0;
    for (let i = 0; i < NCELL; i++) { pst[i] = Math.exp(e[i] - m); s += pst[i] * hx; }
    for (let i = 0; i < NCELL; i++) pst[i] /= s;
    tauHop = hopTime();
    p.b.forEach((b, i) => { pools[i].setLevel(Math.min(1, b / 1.1)); pools[i].setWidth(0.2 + 0.07 * b); bulbLevel(v, 2 * i, b / 1.1); });
  }

  function restart() {
    R = rng(31);
    X = new Float64Array(NPOP);
    for (let i = 0; i < NPOP; i++) X[i] = 0.1 * R.normal();
    fp.setDensity((x) => Math.exp(-(x * x) / (2 * 0.01)));
    t = 0; acc = 0;
  }

  function step() {
    const sq = p.sigma * Math.sqrt(DT);
    for (let i = 0; i < NPOP; i++) {
      let y = X[i] + KAPPA * lamps.dI(X[i]) * DT + sq * R.normal();
      if (y < LO) y = 2 * LO - y; if (y > HI) y = 2 * HI - y;
      X[i] = y;
    }
    t += DT;
  }
  function advanceFp(T) {
    const n = Math.ceil(T / FP_H - 1e-9);
    for (let k = 0; k < n; k++) fp.step(T / n);
  }

  function stats() {
    const hist = histogram(X, LO, HI, NB), pb = binAverage(fp.p, NB);
    let gap = 0, peak = 0, rRob = 0, rFp = 0, rSt = 0;
    for (let k = 0; k < NB; k++) { gap = Math.max(gap, Math.abs(hist[k] - pb[k])); peak = Math.max(peak, pb[k]); }
    for (let i = 0; i < NPOP; i++) if (X[i] > 0) rRob++;
    for (let i = 0; i < NCELL; i++) if (xs[i] > 0) { rFp += fp.p[i] * hx; rSt += pst[i] * hx; }
    return { hist, gap: gap / peak, right: { robots: rRob / NPOP, fp: rFp, stationary: rSt } };
  }

  for (const k of ['bl', 'br', 'sigma']) onInput(panel, k, readParams);
  onClick(panel, 'restart', restart);
  seg(panel, 'speed', (s) => { speed = Number(s); });
  readParams();
  restart();

  const colour = (i) => ledColour(lamps.I(X[i]) / 0.9);
  const pct = (q) => `${(100 * q).toFixed(0)}%`;
  let S = stats(), shown = '';
  v.onFrame((dt) => {
    acc += dt * speed;
    let n = 0;
    while (acc >= DT && n < 20000) { step(); acc -= DT; n++; }
    if (n) { advanceFp(n * DT); S = stats(); }
    swarm.update(X, ry, colour, NDRAW);
    wall.update(S.hist);
    wallCurve(wall, fpCurve, xs, fp.p);
    wallCurve(wall, stCurve, xs, pst);
    const r = S.right;
    [[1 - r.robots, 1 - r.fp, 1 - r.stationary, 'left half', 0], [r.robots, r.fp, r.stationary, 'right half', 1]].forEach(([a, b, c, name, i]) => {
      side[i].set(`<div style="text-align:center"><b>${name}</b> &nbsp;<span style="color:${CSS.grey}">lamp ${p.b[i].toFixed(2)}</span><br>` +
        `<span style="color:${CSS.blue}">robots ${pct(a)}</span> &nbsp;<span style="color:${CSS.cyan}">FP ${pct(b)}</span>` +
        ` &nbsp;<span style="color:${CSS.pink}">stationary ${pct(c)}</span></div>`);
    });
    corner.set(`t = <b>${t.toFixed(1)} s</b> &nbsp;·&nbsp; played ${speed}× faster<br>` +
      `<span style="color:${CSS.yellow}">dX = κ I′(X) dt + σ dW</span>`);
    ov.update();
    const s = `gap |hist − FP| / peak <b>${S.gap.toFixed(2)}</b> &nbsp; hops between the lamps settle in about <b>${tauHop.toFixed(0)} s</b>` +
      ` (play at 50×)`;
    if (s !== shown) { shown = s; readout(panel, 'info', s); }
  });

  panel._state = () => {
    const s = stats();
    return { t, speed, params: { kappa: KAPPA, sigma: p.sigma, brightness: p.b, lampsX: [-CX, CX], lampWidth: WL, dt: DT, robots: NPOP },
      gap: s.gap, rightShare: s.right, hopTimeS: tauHop, fpMass: fp.p.reduce((a, b) => a + b, 0) * hx };
  };
  panel._advance = (T) => { const n = Math.round(T / DT); for (let k = 0; k < n; k++) step(); advanceFp(n * DT); S = stats(); return panel._state(); };
}
