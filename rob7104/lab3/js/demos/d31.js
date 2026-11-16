// 3.1 Euler-Maruyama on a swarm: robots steer up the light of one lamp, dX = kappa I'(X) dt + sigma dW, simulated by EM
// with step dt. The density wall shows their histogram against the Fokker-Planck solution (cyan), solved on a fine grid
// with small implicit steps, so it does not depend on the EM step. Same parameters as the notebook clip s03_em.
import { rng, Lamps1D, FP1D, histogram } from '../sim3.js';
import { makeViewer, placeLamp, SwarmMesh, ledColour, LightPool, DensityWall, COL, CSS, val, onInput, onClick, readout, seg, THREE } from '../common3.js';
import { Overlay, Ribbon, wallCurve, binAverage, robotY, legendItem } from '../dens_common.js';

const LAMP_W = 0.45;                 // lamp width (m)
const X0 = -0.6, S0 = 0.06;          // start: a narrow group left of the lamp
const NPOP = 3000, NDRAW = 400;
const LO = -1.47, HI = 1.47;         // reflecting walls (the corridor is 3 m long)
const NB = 30, NCELL = 300;          // histogram bins; Fokker-Planck cells (10 per bin)
const FP_H = 0.02;                   // largest Fokker-Planck time step (s)
const YW = 0.32, ZW = 0.06, HW = 0.62;   // density wall: plane, base, height
const LAMP_Z = 0.98;                 // lamp height in the picture (lower than the real 1.6 m so it stays in view)

export async function init(panel) {
  const v = await makeViewer(panel, 'corridor.mjb', { lookat: [0, 0.18, 0.36], distance: 2.75, azimuth: 90, elevation: -18 });
  placeLamp(v, 1, null); placeLamp(v, 2, null);
  const lamps = new Lamps1D([0.5], [1.0], LAMP_W);
  const pool = new LightPool(v.scene, 0.5, 0, 0.3);
  const swarm = new SwarmMesh(v.scene, NDRAW);
  const wall = new DensityWall(v.scene, { lo: LO, hi: HI, nb: NB, y0: YW, z0: ZW, height: HW, vmax: 2.5, curveColours: [] });
  const fpCurve = new Ribbon(v.scene, COL.cyan, { y0: YW - 0.008, width: 0.013 });
  const ov = new Overlay(panel, v);
  ov.box('legend', `${legendItem(CSS.blue, `histogram of all ${NPOP} robots (${NDRAW} drawn)`, 'bar')}<br>` +
    `${legendItem(CSS.cyan, 'Fokker–Planck solution')}<br>${legendItem(CSS.pink, 'one robot followed step by step', 'bar')}`);
  const corner = ov.box('corner');
  const lampLabel = ov.label([0.62, 0, LAMP_Z + 0.02], 'l');
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.04, 0.058, 40), new THREE.MeshBasicMaterial({ color: COL.pink, toneMapped: false, depthTest: false }));
  ring.position.z = 0.04;
  ring.renderOrder = 3;
  v.scene.add(ring);

  const p = { kappa: 0, sigma: 0, dt: 0, xl: 0.5 };
  const f = (x) => p.kappa * lamps.dI(x);
  const g = () => p.sigma;
  const fp = new FP1D(f, g, LO, HI, NCELL);
  const hx = (HI - LO) / NCELL;
  const ry = robotY(NDRAW, rng(77));
  let X, prev, R, t, acc, last, speed = 3, vmax = 2.5;

  function readParams() {
    p.kappa = val(panel, 'kappa'); p.sigma = val(panel, 'sigma'); p.dt = val(panel, 'dt'); p.xl = val(panel, 'lamp');
    lamps.c = [p.xl];
    placeLamp(v, 0, [p.xl, 0], LAMP_Z);
    v.forward();
    pool.setPos(p.xl, 0);
    fp.setModel(f, g);
    // wall scale from the stationary density exp(2 kappa I / sigma^2)
    let m = -Infinity, s = 0;
    const e = fp.x.map((x) => 2 * p.kappa * lamps.I(x) / p.sigma ** 2);
    const emax = Math.max(...e);
    for (const z of e) { const q = Math.exp(z - emax); s += q * hx; m = Math.max(m, q); }
    vmax = Math.min(5, Math.max(1.2, 1.2 * m / s));
    wall.vmax = vmax;
  }

  function restart() {
    R = rng(8);
    X = new Float64Array(NPOP);
    for (let i = 0; i < NPOP; i++) X[i] = X0 + S0 * R.normal();
    prev = Float64Array.from(X.subarray(0, NDRAW));
    fp.setDensity((x) => Math.exp(-((x - X0) ** 2) / (2 * S0 * S0)));
    t = 0; acc = 0; last = null;
  }

  /** One EM step for every robot; the Fokker-Planck solution advances by the same time in small implicit steps. */
  function step() {
    const h = p.dt, sq = Math.sqrt(h);
    prev.set(X.subarray(0, NDRAW));
    for (let i = 0; i < NPOP; i++) {
      const drift = f(X[i]) * h, kick = p.sigma * sq * R.normal();
      let y = X[i] + drift + kick;
      if (y < LO) y = 2 * LO - y; if (y > HI) y = 2 * HI - y;
      if (i === 0) last = { drift, kick, from: X[0], to: y };
      X[i] = y;
    }
    const n = Math.ceil(h / FP_H - 1e-9);
    for (let k = 0; k < n; k++) fp.step(h / n);
    t += h;
  }

  function stats() {
    const hist = histogram(X, LO, HI, NB);
    const pb = binAverage(fp.p, NB);
    let gap = 0, peak = 0, mr = 0, mf = 0, vr = 0, vf = 0, near = 0, nearFp = 0;
    for (let k = 0; k < NB; k++) { gap = Math.max(gap, Math.abs(hist[k] - pb[k])); peak = Math.max(peak, pb[k]); }
    for (let i = 0; i < NPOP; i++) { mr += X[i]; vr += X[i] * X[i]; if (Math.abs(X[i] - p.xl) < 0.3) near++; }
    for (let i = 0; i < NCELL; i++) { const w = fp.p[i] * hx, x = fp.x[i]; mf += w * x; vf += w * x * x; if (Math.abs(x - p.xl) < 0.3) nearFp += w; }
    mr /= NPOP; vr = vr / NPOP - mr * mr; vf -= mf * mf;
    return { hist, pb, gap: gap / peak, meanRobots: mr, meanFp: mf, sdRobots: Math.sqrt(vr), sdFp: Math.sqrt(vf),
      nearLampRobots: near / NPOP, nearLampFp: nearFp };
  }

  for (const k of ['kappa', 'sigma', 'dt', 'lamp']) onInput(panel, k, () => { readParams(); acc = Math.min(acc, p.dt); });
  onClick(panel, 'restart', restart);
  seg(panel, 'speed', (s) => { speed = Number(s); });
  readParams();
  restart();

  const xs = Array.from(fp.x);
  const colour = (i) => (i === 0 ? [0.82, 0.28, 0.74] : ledColour(lamps.I(dispX[i])));
  const dispX = new Float64Array(NDRAW);
  let S = stats(), shown = '';
  v.onFrame((dt) => {
    acc += dt * speed;
    let n = 0;
    while (acc >= p.dt && n < 4000) { step(); acc -= p.dt; n++; }
    if (n === 4000) acc = 0;
    if (n) S = stats();
    const fr = Math.min(1, acc / p.dt), tween = p.dt > 0.2;   // long steps: draw robots part-way along their last step
    for (let i = 0; i < NDRAW; i++) dispX[i] = tween ? prev[i] + (X[i] - prev[i]) * fr : X[i];
    swarm.update(dispX, ry, colour);
    wall.update(S.hist);
    wallCurve(wall, fpCurve, xs, fp.p);
    lampLabel.move([p.xl + 0.12, 0, LAMP_Z + 0.02]);
    ring.position.x = dispX[0]; ring.position.y = ry[0];
    lampLabel.set(`<span style="color:${CSS.accent}">lamp at x = ${p.xl.toFixed(2)} m</span>`);
    corner.set(`t = <b>${t.toFixed(1)} s</b> &nbsp;·&nbsp; played ${speed}× faster<br>` +
      `<span style="color:${CSS.yellow}">X ← X + κ I′(X) Δt + σ √Δt Z</span>`);
    ov.update();
    const st = last ? `pink robot's last step: drift <b class="ye">${(100 * last.drift >= 0 ? '+' : '−')}${Math.abs(100 * last.drift).toFixed(1)} cm</b>` +
      ` + kick <b class="pk">${(last.kick >= 0 ? '+' : '−')}${Math.abs(100 * last.kick).toFixed(1)} cm</b>` : '';
    const s = `gap |hist − FP| / peak <b>${S.gap.toFixed(2)}</b> &nbsp; mean x: robots <b class="bl">${S.meanRobots.toFixed(2)} m</b>` +
      ` FP <b class="cy">${S.meanFp.toFixed(2)} m</b> &nbsp; within 0.3 m of the lamp: robots <b class="bl">${(100 * S.nearLampRobots).toFixed(0)}%</b>` +
      ` FP <b class="cy">${(100 * S.nearLampFp).toFixed(0)}%</b><br>${st}`;
    if (s !== shown) { shown = s; readout(panel, 'info', s); }
  });

  panel._state = () => {
    const s = stats();
    return { t, params: { ...p, lampWidth: LAMP_W, x0: X0, s0: S0, robots: NPOP, bins: NB, fpCells: NCELL }, speed,
      gap: s.gap, meanRobots: s.meanRobots, meanFp: s.meanFp, sdRobots: s.sdRobots, sdFp: s.sdFp,
      nearLampRobots: s.nearLampRobots, nearLampFp: s.nearLampFp, fpMass: fp.p.reduce((a, b) => a + b, 0) * hx, lastStep: last, vmax };
  };
  // for tests: advance the simulation by T seconds of simulated time at once
  panel._advance = (T) => { const n = Math.round(T / p.dt); for (let k = 0; k < n; k++) step(); S = stats(); return panel._state(); };
}
