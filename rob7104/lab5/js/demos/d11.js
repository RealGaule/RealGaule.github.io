// 1.1 Predict and correct: the lecture's position-velocity Kalman filter (L10, at length scale 1:10, as the clip
// s11_tunnel) stepped by hand. "predict" drives the true cart half a second and runs the filter's prediction on the
// fine grid (exact, Q is exact); "correct" takes one position reading, except while the true cart is inside the
// tunnel. The blue samples carry the filter's belief exactly (sim/kf.py simulate_cloud: each sample moves with its own
// process noise and is shifted by K (z + r_i - p_i) at a reading). Numerics: sim5.js (pvModel, KalmanFilter, RNG).
import {
  S, THREE, COL, CSS, makeViewer, setCart, Wheels, loadGhost, CartCloud, Ribbon, keyboard, ctl, val, onInput,
  onClick, readout, plot2d, scales, drawAxes, drawCurve, drawEllipse,
} from '../common.js';

// model of the clip (out/s11_tunnel_facts.json "model")
const Q_C = 5e-4, R_Z = 0.01, SIG_Z = 0.1, DT = 0.5, SUB = 45, FINE = DT / SUB;
const X0 = [0, 0.25], P0 = [[0.01, 0], [0, 0.01]];
const N_CLOUD = 300, N_BODIES = 24;
// stage: the cart starts at world x = START (p = 0); the tunnel entrance is fixed, its length is the slider
const START = -2.2, END = 2.25, X_IN = -1.2;
// stated rule for a run (the clip's condition (a)): the true speed stays in [0.12, 0.4] m/s until the end of the track
const V_RANGE = [0.12, 0.4], T_MAX = 45;
const MORPH_S = 0.7, AUTO_HOLD_S = 0.35;

/* ---------------- the true run ---------------- */
function simulateTruth(seed) {
  const rng = new S.RNG(seed);
  const { F, Q } = S.pvModel(FINE, Q_C);
  const Lq = S.gaussianFactor(Q);
  const x0 = S.sampleGaussian(X0, P0, 1, rng);
  const nMax = Math.round(T_MAX / FINE);
  const p = [x0[0]], v = [x0[1]];
  let ok = v[0] >= V_RANGE[0] && v[0] <= V_RANGE[1];
  for (let n = 0; ok && n < nMax && START + p[n] < END; n++) {
    const x = S.mulVec(F, [p[n], v[n]]), a = rng.normal(), b = rng.normal();
    p.push(x[0] + Lq[0][0] * a + Lq[0][1] * b);
    v.push(x[1] + Lq[1][0] * a + Lq[1][1] * b);
    if (v[n + 1] < V_RANGE[0] || v[n + 1] > V_RANGE[1]) ok = false;
  }
  ok = ok && START + p[p.length - 1] >= END;
  return ok ? { p: Float64Array.from(p), v: Float64Array.from(v) } : null;
}

/* ---------------- small 3D pieces ---------------- */
function arrowMeshes(color, dir, { shaftR, headR, headL }) {
  // geometry along +x (dir 'x') or pointing down (dir '-z'), base at the origin, unit shaft length
  const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.98, depthWrite: true });
  const shaftG = new THREE.CylinderGeometry(shaftR, shaftR, 1, 14); shaftG.translate(0, 0.5, 0);
  const headG = new THREE.ConeGeometry(headR, headL, 20); headG.translate(0, headL / 2, 0);
  if (dir === 'x') { shaftG.rotateZ(-Math.PI / 2); headG.rotateZ(-Math.PI / 2); } else { shaftG.rotateX(-Math.PI / 2); headG.rotateX(-Math.PI / 2); }
  const shaft = new THREE.Mesh(shaftG, mat), head = new THREE.Mesh(headG, mat), g = new THREE.Group();
  g.add(shaft, head);
  return { g, shaft, head, mat, headL };
}

/** Yellow "stretch of track" [p - 2 sigma, p + 2 sigma] as a rounded box on the floor (its width has no meaning). */
class StretchBand {
  constructor(scene, color, halfWidth = 0.17, n = 72) {
    this.n = n; this.b = halfWidth;
    this.band = new Ribbon(scene, color, { n, z: 0.0045, opacity: 0.95 });
    this.fpos = new Float32Array(3 * (n + 1));
    const idx = [];
    for (let i = 0; i < n; i++) idx.push(0, 1 + i, 1 + ((i + 1) % n));
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.fpos, 3));
    g.setIndex(idx);
    this.fill = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false }));
    this.fill.frustumCulled = false;
    scene.add(this.fill);
  }
  set(x, half) {
    const pts = [], a = Math.max(half, 0.01), b = this.b;
    for (let i = 0; i < this.n; i++) {
      const t = 2 * Math.PI * i / this.n, c = Math.cos(t), s = Math.sin(t);
      pts.push([x + a * Math.sign(c) * Math.abs(c) ** 0.35, b * Math.sign(s) * Math.abs(s) ** 0.35]);
    }
    this.band.set(pts, 0.012);
    this.fpos.set([x, 0, 0.004], 0);
    pts.forEach((p, i) => this.fpos.set([p[0], p[1], 0.004], 3 * (i + 1)));
    this.fill.geometry.attributes.position.needsUpdate = true;
  }
}

/* ---------------- 2D helpers ---------------- */
function niceTicks(a, b, n = 4) {
  const raw = (b - a) / n, mag = 10 ** Math.floor(Math.log10(raw)), f = raw / mag;
  const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag, out = [];
  for (let x = Math.ceil(a / step) * step; x <= b + 1e-9; x += step) out.push(Number(x.toFixed(10)));
  return { ticks: out, digits: Math.max(0, -Math.floor(Math.log10(step) + 1e-9)) };
}
const pdf = (x, m, var_) => Math.exp(-0.5 * (x - m) ** 2 / var_) / Math.sqrt(2 * Math.PI * var_);
const lerp = (a, b, t) => a + (b - a) * t;
const lerpM = (A, B, t) => A.map((r, i) => r.map((x, j) => lerp(x, B[i][j], t)));

export async function init(panel) {
  const v = await makeViewer(panel, 'tunnel.mjb', { lookat: [START + 0.45, 0, 0.08], distance: 1.9, azimuth: 64, elevation: -21 });
  const geo = await loadGhost();
  const scene = v.scene;
  const flag = ctl(panel, 'flag');

  /* ---- brighten the translucent estimate cart (dim at a distance in the viewer) ---- */
  const bodyName = (b) => v.mujoco.mj_id2name(v.model, v.mujoco.mjtObj.mjOBJ_BODY.value, b) || '';
  for (let g = 0; g < v.model.ngeom; g++) {
    const m = v.meshes[g];
    if (m && /^est/.test(bodyName(v.model.geom_bodyid[g]))) {
      m.material = m.material.clone();
      m.material.emissive = new THREE.Color(COL.est);
      m.material.emissiveIntensity = 0.28;
    }
  }

  /* ---- tunnel of adjustable length: hide the model's tunnel, draw clones of its parts ---- */
  setCart(v, 'cart', [START, 0, 0], [0, 0]); setCart(v, 'est', [START, 0, 0], [0, 0]);
  v.forward();
  const xp = v.data.geom_xpos, xm = v.data.geom_xmat;
  const rib = [], glowIn = [], glowOut = [], long = [];
  for (let g = 0; g < v.model.ngeom; g++) {
    const m = v.meshes[g];
    if (!m || v.model.geom_bodyid[g] !== 0) continue;
    const x = xp[3 * g], y = xp[3 * g + 1], z = xp[3 * g + 2], sz = v.model.geom_size.slice(3 * g, 3 * g + 3);
    const name = v.mujoco.mj_id2name(v.model, v.mujoco.mjtObj.mjOBJ_GEOM.value, g) || '';
    const floorBox = v.model.geom_type[g] === v.mujoco.mjtGeom.mjGEOM_BOX.value && sz[1] > 0.3 && Math.abs(sz[0] - 0.4) < 1e-3;
    if (Math.abs(x) > 0.45 || Math.abs(y) > 0.34 || (z < 0.0025 && !floorBox)) continue;
    m.visible = false;
    const M0 = m.matrix.clone();
    if (name.startsWith('tun_glow0')) glowIn.push(M0);
    else if (name.startsWith('tun_glow2')) glowOut.push(M0);
    else if (floorBox || sz[0] > 0.39 || Math.abs(xm[9 * g + 2]) > 0.9) long.push({ m, M0 });   // roof, rails, floor
    else if (Math.abs(x) < 1e-4) rib.push({ m, M0 });                                            // the middle arch
  }
  const glowMesh = v.meshes[v.id('geom', 'tun_glow0_0')];
  const glowMat = glowMesh.material.clone(), glowBase = glowMat.emissiveIntensity || 0.5;
  const tGroup = new THREE.Group(); scene.add(tGroup);
  const clone = (m, mat) => { const c = new THREE.Mesh(m.geometry, mat || m.material); c.matrixAutoUpdate = false; c.castShadow = m.castShadow; c.receiveShadow = m.receiveShadow; tGroup.add(c); return c; };
  const MAX_RIBS = 7;
  const ribSets = Array.from({ length: MAX_RIBS }, () => rib.map(({ m, M0 }) => ({ c: clone(m), M0 })));
  const glowSets = [glowIn, glowOut].map((list, i) => list.map((M0) => ({ c: clone(glowMesh, glowMat), M0, i })));
  const longSet = long.map(({ m, M0 }) => ({ c: clone(m), M0 }));
  let L = val(panel, 'L');
  const tunnelX = () => [X_IN, X_IN + L];
  function layoutTunnel() {
    const [x0, x1] = tunnelX(), nr = Math.max(2, Math.round(L / 0.4) + 1), T = new THREE.Matrix4();
    ribSets.forEach((set, i) => set.forEach(({ c, M0 }) => {
      c.visible = i < nr;
      if (c.visible) c.matrix.copy(T.makeTranslation(x0 + (x1 - x0) * i / (nr - 1), 0, 0).multiply(M0));
    }));
    glowSets.forEach((set) => set.forEach(({ c, M0, i }) => c.matrix.copy(T.makeTranslation(i === 0 ? x0 + 0.4 : x1 - 0.4, 0, 0).multiply(M0))));
    const Sx = new THREE.Matrix4().makeScale(L / 0.8, 1, 1);
    longSet.forEach(({ c, M0 }) => c.matrix.copy(T.makeTranslation((x0 + x1) / 2, 0, 0).multiply(Sx).multiply(M0)));
  }
  layoutTunnel();

  /* ---- belief and truth objects ---- */
  const cloud = new CartCloud(scene, geo, { max: N_CLOUD, bodies: N_BODIES, color: COL.cloud, opacity: 0.3, dot: 0.022 });
  const cloudPose = new Float64Array(3 * N_CLOUD);
  const jit = new S.RNG(7);
  const jitterY = Float64Array.from({ length: N_CLOUD }, (_, i) => (i < N_BODIES ? 0 : (jit.uniform() - 0.5) * 0.2));
  const band = new StretchBand(scene, COL.est);
  const truthArrow = arrowMeshes(COL.truth, '-z', { shaftR: 0.007, headR: 0.022, headL: 0.05 });
  scene.add(truthArrow.g);
  truthArrow.shaft.scale.set(1, 1, 0.06); truthArrow.head.position.set(0, 0, -0.06);   // tail 0.36 m, tip 0.25 m (as the clip)
  const velArrow = arrowMeshes(COL.est, 'x', { shaftR: 0.006, headR: 0.02, headL: 0.045 });
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 1, 8).rotateX(Math.PI / 2).translate(0, 0, 0.5), velArrow.mat);
  scene.add(velArrow.g, stem);
  // readings: a green light line at the reading, a green bar on the track, the 2-sigma likelihood stretch
  const greenMat = new THREE.MeshBasicMaterial({ color: COL.reading, transparent: true, opacity: 0.95, depthWrite: false });
  const readLine = new THREE.Mesh(new THREE.BoxGeometry(0.007, 0.007, 0.55).translate(0, 0, 0.275), greenMat);
  const likeBand = new THREE.Mesh(new THREE.PlaneGeometry(4 * SIG_Z, 0.5), new THREE.MeshBasicMaterial({ color: COL.reading, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide }));
  likeBand.position.z = 0.0035;
  scene.add(readLine, likeBand);
  const bars = Array.from({ length: 10 }, () => {
    const b = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.56, 0.002), new THREE.MeshBasicMaterial({ color: COL.reading, transparent: true, opacity: 0, depthWrite: false }));
    b.position.z = 0.003; b.visible = false; scene.add(b); return b;
  });
  let barAge = bars.map(() => Infinity), barNext = 0;

  /* ---- run state ---- */
  const { F: Ff, Q: Qf } = S.pvModel(FINE, Q_C);
  const Lq = S.gaussianFactor(Qf);
  let seed = 0, tried = 0, accepted = 0, triesThis = 0;
  let truth, n, k, kf, cl, rngC, rngZ, wheelsT, wheelsE, corrected, anim, queue, ghost, corr, last;
  let missed, blind, report, lastPostSigma, lastK, counts, autoOn = false, flashUntil = 0, flashMsg = '';

  function newRun() {
    triesThis = 0;
    do { seed++; tried++; triesThis++; truth = simulateTruth(seed); } while (!truth);
    accepted++;
    n = 0; k = 0;
    kf = new S.KalmanFilter({ F: Ff, H: S.H_POS, Q: Qf, R: [[R_Z]], x0: X0, P0 });
    rngC = new S.RNG(seed + 1000); rngZ = new S.RNG(seed + 2000);
    cl = S.sampleGaussian(X0, P0, N_CLOUD, rngC);
    wheelsT = new Wheels([START + truth.p[0], 0, 0]); wheelsE = new Wheels([START + X0[0], 0, 0]);
    corrected = true;                       // no reading at t = 0 (the filter starts from (x0, P0))
    anim = null; queue = []; ghost = null; corr = null; last = 'start';
    missed = 0; blind = null; report = null; lastPostSigma = Math.sqrt(P0[0][0]); lastK = null;
    counts = { predict: 0, correct: 0, blocked: 0 };
    barAge = barAge.map(() => Infinity);
    view.ready = false;
  }

  const pTrue = () => truth.p[n], vTrue = () => truth.v[n];
  const xWorld = () => START + pTrue();
  const inTunnel = () => { const [a, b] = tunnelX(), x = xWorld(); return x > a && x < b; };
  const atEnd = () => n + SUB >= truth.p.length;
  function flash(msg) { flashMsg = msg; flashUntil = performance.now() + 2200; }   // wall clock: frames are slow in tests

  /* ---- the two steps ---- */
  function startPredict() {
    if (atEnd()) { flash('end of the track: new run'); autoOn = false; ctl(panel, 'auto').classList.remove('on'); return false; }
    ghost = { x: kf.x.slice(), P: S.copyMat(kf.P) };
    corr = null; last = 'predict';
    anim = { type: 'p', left: SUB, acc: 0 };
    return true;
  }
  function fineStep() {
    n++;
    kf.predict();
    for (let i = 0; i < N_CLOUD; i++) {
      const p = cl[2 * i], vv = cl[2 * i + 1], a = rngC.normal(), b = rngC.normal();
      cl[2 * i] = p + FINE * vv + Lq[0][0] * a + Lq[0][1] * b;
      cl[2 * i + 1] = vv + Lq[1][0] * a + Lq[1][1] * b;
    }
  }
  function finishPredict() {
    k++; counts.predict++; corrected = false; anim = null;
    if (inTunnel()) {
      missed++;
      if (!blind) blind = { sigma0: lastPostSigma, Kbefore: lastK, missed: 0 };
      blind.missed++;
    }
  }
  function startCorrect(quiet = false) {
    if (corrected) { if (!quiet) flash(k === 0 ? 'predict first: no reading at t = 0' : 'one reading per step: predict first'); return false; }
    if (inTunnel()) { counts.blocked++; if (!quiet) flash('no readings inside the tunnel'); return false; }
    const z = pTrue() + SIG_Z * rngZ.normal();
    const xPrior = kf.x.slice(), PPrior = S.copyMat(kf.P);
    const r = kf.correct([z]);
    const K = [r.K[0][0], r.K[1][0]], cPre = Float64Array.from(cl), d = new Float64Array(2 * N_CLOUD);
    for (let i = 0; i < N_CLOUD; i++) { const e = z + SIG_Z * rngC.normal() - cl[2 * i]; d[2 * i] = K[0] * e; d[2 * i + 1] = K[1] * e; }
    corr = { z, xPrior, PPrior, xPost: kf.x.slice(), PPost: S.copyMat(kf.P), K, nis: r.nis, cPre, d };
    corrected = true; counts.correct++; last = 'correct';
    const sPrior = Math.sqrt(PPrior[0][0]), sPost = Math.sqrt(kf.P[0][0]);
    if (blind) { report = { Kbefore: blind.Kbefore, growth: sPrior / blind.sigma0, shrink: sPrior / sPost, missed: blind.missed, sigmaBefore: blind.sigma0, sigmaExitPrior: sPrior, sigmaExitPost: sPost, blindSeconds: (blind.missed + 1) * DT }; blind = null; }
    lastPostSigma = sPost; lastK = K;
    anim = { type: 'c', beta: 0 };
    bars[barNext].position.x = START + z; barAge[barNext] = 0; barNext = (barNext + 1) % bars.length;
    return true;
  }
  function setCloudBeta(beta) {
    for (let i = 0; i < 2 * N_CLOUD; i++) cl[i] = corr.cPre[i] + beta * corr.d[i];
  }
  function finishAnim() {
    if (!anim) return;
    if (anim.type === 'p') { while (anim.left > 0) { fineStep(); anim.left--; } finishPredict(); } else if (anim.type === 'c') { setCloudBeta(1); anim = null; } else anim = null;
  }
  function begin(a) {
    if (a === 'p') return startPredict();
    if (a === 'c') return startCorrect(false);
    if (a === 'c?') return startCorrect(true);
    if (a === 'w') { anim = { type: 'w', left: AUTO_HOLD_S }; return true; }
    return false;
  }
  function advance(dt) {
    let budget = dt;
    for (let guard = 0; guard < 8 && budget > 0; guard++) {
      if (!anim) {
        if (!queue.length && autoOn) queue.push('p', 'c?', 'w');
        if (!queue.length) break;
        begin(queue.shift());
        continue;
      }
      if (anim.type === 'p') {
        anim.acc += budget; budget = 0;
        while (anim.acc >= FINE && anim.left > 0) { fineStep(); anim.left--; anim.acc -= FINE; }
        if (anim.left === 0) finishPredict();
      } else if (anim.type === 'c') {
        anim.beta = Math.min(1, anim.beta + budget / MORPH_S); budget = 0;
        setCloudBeta(anim.beta);
        if (anim.beta >= 1) anim = null;
      } else {
        anim.left -= budget; budget = 0;
        if (anim.left <= 0) anim = null;
      }
    }
  }

  /* ---- what is on screen ---- */
  function displayed() {
    if (anim && anim.type === 'c') return { x: corr.xPrior.map((a, i) => lerp(a, corr.xPost[i], anim.beta)), P: lerpM(corr.PPrior, corr.PPost, anim.beta), beta: anim.beta };
    return { x: kf.x, P: kf.P, beta: 1 };
  }

  // 2D picture
  const plot = plot2d(panel.querySelector('canvas.plot'));
  const view = { ready: false, cp: 0, cv: 0.25, hp: 0.3, hv: 0.12, ymax: 6 };
  function draw2d(dt) {
    plot.clear();
    const o = plot, g = o.g, D = displayed();
    const P = D.P, sp = Math.sqrt(P[0][0]), sv = Math.sqrt(P[1][1]);
    const refP = [P[0][0], ghost ? ghost.P[0][0] : 0, corr ? corr.PPrior[0][0] : 0];
    const refV = [P[1][1], ghost ? ghost.P[1][1] : 0, corr ? corr.PPrior[1][1] : 0];
    const tp = Math.max(0.2, 2.9 * Math.sqrt(Math.max(...refP)), 1.3 * Math.abs(pTrue() - D.x[0]), corr ? 1.2 * Math.abs(corr.z - D.x[0]) + 0.25 : 0);
    const tv = Math.max(0.08, 2.9 * Math.sqrt(Math.max(...refV)), 1.3 * Math.abs(vTrue() - D.x[1]));
    const a = view.ready ? Math.min(1, dt * 4) : 1;
    view.hp += (tp - view.hp) * a; view.hv += (tv - view.hv) * a;
    view.cp += (D.x[0] - view.cp) * (view.ready ? Math.min(1, dt * 8) : 1);
    view.cv += (D.x[1] - view.cv) * (view.ready ? Math.min(1, dt * 8) : 1);
    const H = o.h, lowH = Math.round(H * 0.27);
    const pr = [view.cp - view.hp, view.cp + view.hp], vr = [view.cv - view.hv, view.cv + view.hv];
    const s = scales(o, pr, vr, [50, 14, 58, lowH + 34]);
    const [l, t, W, Hh] = s.box;

    // header: predict / correct
    const blocked = inTunnel();
    const head = (x, word, formula, on, colr) => {
      g.font = '600 15px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'top';
      g.fillStyle = on ? colr : CSS.faint; g.fillText(word, x, 10);
      g.font = '12px Inter, sans-serif'; g.fillStyle = on ? CSS.white : CSS.faint; g.fillText(formula, x, 31);
    };
    const pOn = (anim && anim.type === 'p') || (!anim && last === 'predict');
    const cOn = (anim && anim.type === 'c') || (!anim && last === 'correct');
    head(l, 'predict', 'P⁻ = F P Fᵀ + Q', pOn, CSS.cloud);
    head(l + W * 0.52, blocked ? 'correct: no reading' : 'correct', 'P⁺ = (I − K H) P⁻', cOn || blocked, blocked ? CSS.fail : CSS.reading);

    // grid
    const xt = niceTicks(pr[0], pr[1], 4), yt = niceTicks(vr[0], vr[1], 4);
    g.save(); g.strokeStyle = 'rgba(255,255,255,0.06)'; g.lineWidth = 1;
    xt.ticks.forEach((x) => { g.beginPath(); g.moveTo(s.X(x), t); g.lineTo(s.X(x), t + Hh); g.stroke(); });
    yt.ticks.forEach((y) => { g.beginPath(); g.moveTo(l, s.Y(y)); g.lineTo(l + W, s.Y(y)); g.stroke(); });
    g.beginPath(); g.rect(l, t, W, Hh); g.clip();
    // reading: a vertical band (it measures p only)
    if (corr) {
      g.fillStyle = 'rgba(131,193,103,0.13)'; g.fillRect(s.X(corr.z - 2 * SIG_Z), t, s.X(corr.z + 2 * SIG_Z) - s.X(corr.z - 2 * SIG_Z), Hh);
      g.strokeStyle = CSS.reading; g.lineWidth = 1.5; g.beginPath(); g.moveTo(s.X(corr.z), t); g.lineTo(s.X(corr.z), t + Hh); g.stroke();
    }
    // samples
    g.fillStyle = 'rgba(88,196,221,0.75)';
    for (let i = 0; i < N_CLOUD; i++) { g.beginPath(); g.arc(s.X(cl[2 * i]), s.Y(cl[2 * i + 1]), 1.6, 0, 2 * Math.PI); g.fill(); }
    // ellipses: the start of the last step dashed, the current belief solid
    const ref = corr ? { x: corr.xPrior, P: corr.PPrior } : ghost;
    if (ref) { g.setLineDash([5, 4]); drawEllipse(o, s, ref.x, ref.P, 2, (corr ? 'rgba(88,196,221,0.85)' : 'rgba(255,235,64,0.55)'), null, 1.4); g.setLineDash([]); }
    drawEllipse(o, s, D.x, P, 2, CSS.est, 'rgba(255,235,64,0.10)', 2.2);
    if (corr) {                                                  // the shift K nu moves p AND v
      g.strokeStyle = CSS.est; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(s.X(corr.xPrior[0]), s.Y(corr.xPrior[1])); g.lineTo(s.X(D.x[0]), s.Y(D.x[1])); g.stroke();
    }
    g.fillStyle = CSS.est; g.beginPath(); g.arc(s.X(D.x[0]), s.Y(D.x[1]), 3.5, 0, 2 * Math.PI); g.fill();
    g.strokeStyle = CSS.white; g.lineWidth = 2; g.beginPath(); g.arc(s.X(pTrue()), s.Y(vTrue()), 5, 0, 2 * Math.PI); g.stroke();
    g.restore();
    drawAxes(o, s, { xticks: xt.ticks, digits: xt.digits });
    drawAxes(o, s, { yticks: yt.ticks, digits: yt.digits, ylabel: 'velocity v (m/s)' });
    const rho = P[0][1] / (sp * sv);
    g.font = '12px Inter, sans-serif'; g.fillStyle = CSS.grey; g.textAlign = 'right'; g.textBaseline = 'top';
    { const tw = g.measureText(`tilt ρ = ${rho.toFixed(2)}`).width; g.fillStyle = 'rgba(0,0,0,0.7)'; g.fillRect(l + W - tw - 8, t + 2, tw + 8, 17); g.fillStyle = CSS.grey; }
    g.fillText(`tilt ρ = ${rho.toFixed(2)}`, l + W - 4, t + 4);

    // lower plot: densities over p
    const s2 = scales(o, pr, [0, view.ymax * 1.15], [50, 14, H - lowH - 6, 28]);
    const xs = Array.from({ length: 161 }, (_, i) => pr[0] + (pr[1] - pr[0]) * i / 160);
    const curves = [];
    let peak = 1 / Math.sqrt(2 * Math.PI * P[0][0]);
    if (corr) {
      peak = Math.max(peak, 1 / Math.sqrt(2 * Math.PI * corr.PPrior[0][0]), 1 / Math.sqrt(2 * Math.PI * R_Z));
      curves.push([xs.map((x) => pdf(x, corr.xPrior[0], corr.PPrior[0][0])), 'rgba(88,196,221,0.85)', 1.5, [5, 4]]);
      curves.push([xs.map((x) => pdf(x, corr.z, R_Z)), CSS.reading, 1.8, null]);
    } else if (ghost) {
      peak = Math.max(peak, 1 / Math.sqrt(2 * Math.PI * ghost.P[0][0]));
      curves.push([xs.map((x) => pdf(x, ghost.x[0], ghost.P[0][0])), 'rgba(255,235,64,0.55)', 1.4, [5, 4]]);
    }
    view.ymax += (1.12 * peak - view.ymax) * a;
    const ys = xs.map((x) => pdf(x, D.x[0], P[0][0]));
    g.save(); g.beginPath(); g.rect(s2.box[0], s2.box[1] - 2, s2.box[2], s2.box[3] + 2); g.clip();
    g.fillStyle = 'rgba(255,235,64,0.14)'; g.beginPath(); g.moveTo(s2.X(xs[0]), s2.Y(0));
    xs.forEach((x, i) => g.lineTo(s2.X(x), s2.Y(ys[i]))); g.lineTo(s2.X(xs[160]), s2.Y(0)); g.fill();
    curves.forEach(([yy, c, w, dash]) => drawCurve(o, s2, xs, yy, c, w, dash));
    drawCurve(o, s2, xs, ys, CSS.est, 2.2);
    g.restore();
    drawAxes(o, s2, { xticks: xt.ticks, digits: xt.digits, xlabel: 'position p (m)' });
    g.strokeStyle = CSS.white; g.lineWidth = 2; g.beginPath(); g.arc(s2.X(pTrue()), s2.box[1] + s2.box[3], 4.5, 0, 2 * Math.PI); g.stroke();
    if (corr) {
      g.font = '12px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'top';
      let x = s2.box[0] + 4; const y = s2.box[1] - 2;
      [['prior', CSS.cloud], [' × ', CSS.grey], ['reading', CSS.reading], [' ∝ ', CSS.grey], ['posterior', CSS.est]].forEach(([w, c]) => { g.fillStyle = c; g.fillText(w, x, y); x += g.measureText(w).width; });
    }
    view.ready = true;
  }

  // 3D picture
  function draw3d(dt) {
    const D = displayed();
    const xt = xWorld(), xe = START + D.x[0];
    setCart(v, 'cart', [xt, 0, 0], wheelsT.update([xt, 0, 0]));
    setCart(v, 'est', [xe, 0, 0], wheelsE.update([xe, 0, 0]));
    for (let i = 0; i < N_CLOUD; i++) { cloudPose[3 * i] = START + cl[2 * i]; cloudPose[3 * i + 1] = jitterY[i]; cloudPose[3 * i + 2] = 0; }
    cloud.set(cloudPose, N_CLOUD);
    band.set(xe, 2 * Math.sqrt(D.P[0][0]));
    truthArrow.g.position.set(xt, 0, 0.36);
    const vl = Math.max(0.02, D.x[1] * 1.6);                       // estimated velocity: 1.6 s of travel
    velArrow.g.position.set(xe + 0.02, 0, 0.22);
    velArrow.shaft.scale.set(Math.max(vl - velArrow.headL, 0.001), 1, 1); velArrow.head.position.set(Math.max(vl - velArrow.headL, 0.001), 0, 0);
    stem.position.set(xe + 0.02, 0, 0.11); stem.scale.set(1, 1, 0.11);
    // readings
    readLine.visible = likeBand.visible = !!corr;               // the latest reading, until the next predict
    if (corr) { readLine.position.set(START + corr.z, 0, 0); likeBand.position.x = START + corr.z; }
    bars.forEach((b, i) => { barAge[i] += dt; b.visible = barAge[i] < 6; if (b.visible) b.material.opacity = 0.85 * Math.exp(-barAge[i] / 2.2); });
    // tunnel portals: bright red while the true cart is inside (no readings)
    const inside = inTunnel();
    glowMat.emissiveIntensity = glowBase * (inside ? 2 : 1);
    const flashing = performance.now() < flashUntil;
    flag.textContent = flashing ? flashMsg : 'no readings';
    flag.classList.toggle('show', inside || flashing);
    ctl(panel, 'correct').classList.toggle('blocked', inside);
    // camera follows the cart along the track (orbit and zoom stay with the user)
    if (follow) {
      const target = xt + 0.35;
      const dx = (target - v.controls.target.x) * Math.min(1, dt * 2.5);
      v.controls.target.x += dx; v.camera.position.x += dx;
    }
    v.forward();
  }

  function info() {
    const P = kf.P, sp = Math.sqrt(P[0][0]);
    let h = `t <b>${(n * FINE).toFixed(1)} s</b> · σ<sub>p</sub> <b>${sp.toFixed(3)} m</b>`;
    if (corr) h += ` · K <b>[${corr.K[0].toFixed(2)}, ${corr.K[1].toFixed(2)}]</b>`;
    if (blind) h += ` · <span class="c-red">blind: <b>${blind.missed}</b> readings missed</span>`;
    else if (report) h += ` · tunnel: <b>${((report.missed + 1) * DT).toFixed(1)} s</b> blind, σ<sub>p</sub> <b>×${report.growth.toFixed(1)}</b>, then <b>÷${report.shrink.toFixed(1)}</b> by the first reading`;
    readout(panel, 'info', h);
  }

  /* ---- controls ---- */
  let follow = true;
  const press = (a) => { queue.push(a); };
  onClick(panel, 'predict', () => press('p'));
  onClick(panel, 'correct', () => press('c'));
  onClick(panel, 'auto', () => { autoOn = !autoOn; ctl(panel, 'auto').classList.toggle('on', autoOn); if (!autoOn) queue = queue.filter((a) => a !== 'w' && a !== 'c?'); });
  onClick(panel, 'reset', () => { autoOn = false; ctl(panel, 'auto').classList.remove('on'); newRun(); snapCamera(); info(); });
  onClick(panel, 'follow', () => { follow = !follow; ctl(panel, 'follow').classList.toggle('on', follow); });
  onInput(panel, 'L', () => { L = val(panel, 'L'); layoutTunnel(); autoOn = false; ctl(panel, 'auto').classList.remove('on'); newRun(); snapCamera(); info(); });
  const kb = keyboard(panel, ['p', 'P', 'c', 'C']);
  ['p', 'P'].forEach((key) => kb.onPress(key, () => press('p')));
  ['c', 'C'].forEach((key) => kb.onPress(key, () => press('c')));
  function snapCamera() {
    const dx = xWorld() + 0.35 - v.controls.target.x;
    v.controls.target.x += dx; v.camera.position.x += dx;
  }

  newRun();
  v.onFrame((dt) => {
    advance(dt);
    draw3d(dt);
    draw2d(dt);
    info();
    if (!anim && !queue.length) panel.dataset.idle = '1'; else delete panel.dataset.idle;
  });
  draw3d(0); draw2d(0); info();

  /* ---- for the tests ---- */
  // run actions at once, without the animation: 'p' predict, 'c' correct, e.g. panel._advance('pcpcpc')
  panel._advance = (str) => {
    finishAnim(); queue = [];
    for (const ch of str) { if (begin(ch === 'p' ? 'p' : ch === 'c' ? 'c' : '')) finishAnim(); }
    return panel._state();
  };
  panel._state = () => {
    const P = kf.P, sp = Math.sqrt(P[0][0]), sv = Math.sqrt(P[1][1]);
    let mp = 0, mv = 0; for (let i = 0; i < N_CLOUD; i++) { mp += cl[2 * i]; mv += cl[2 * i + 1]; }
    mp /= N_CLOUD; mv /= N_CLOUD;
    let vp = 0, vv = 0; for (let i = 0; i < N_CLOUD; i++) { vp += (cl[2 * i] - mp) ** 2; vv += (cl[2 * i + 1] - mv) ** 2; }
    vp /= N_CLOUD - 1; vv /= N_CLOUD - 1;
    const e = [pTrue() - kf.x[0], vTrue() - kf.x[1]];
    const Pi = S.inv(P), nees = e[0] * (Pi[0][0] * e[0] + Pi[0][1] * e[1]) + e[1] * (Pi[1][0] * e[0] + Pi[1][1] * e[1]);
    return {
      seed, triesThisRun: triesThis, runRule: `true v in [${V_RANGE}] m/s to the end of the track`, acceptedShare: accepted / tried,
      t: n * FINE, k, fineIndex: n, pTrue: pTrue(), vTrue: vTrue(), xWorldTrue: xWorld(),
      xhat: kf.x.slice(), P: S.copyMat(P), sigma_p: sp, sigma_v: sv, rho: P[0][1] / (sp * sv), nees,
      lastCorrection: corr ? { z: corr.z, K: corr.K, nis: corr.nis, priorSigmaP: Math.sqrt(corr.PPrior[0][0]) } : null,
      tunnel: { x0: X_IN, x1: X_IN + L, L }, inTunnel: inTunnel(), missed, blind: blind ? { ...blind } : null, report,
      cloud: { n: N_CLOUD, meanErrP: mp - kf.x[0], meanErrV: mv - kf.x[1], varPOverP: vp / P[0][0], varVOverP: vv / P[1][1] },
      counts: { ...counts }, anim: anim ? anim.type : null, queue: queue.length, auto: autoOn, follow, atEnd: atEnd(),
      camera: { target: v.controls.target.toArray(), position: v.camera.position.toArray() },
      tunnelClones: { ribs: ribSets.length * rib.length, ribGeomsPerArch: rib.length, glowIn: glowIn.length, glowOut: glowOut.length, long: long.length },
    };
  };
}
