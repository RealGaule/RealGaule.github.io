// 3.1 EKF: linearise at the mean (clip s31_ekf).
// The command "1 m straight ahead in 4 s" with the lab's noisy wheels. A beacon at (1.75, 0.55) gives range and bearing
// every 0.5 s; for the last `tb` seconds of the run there are no readings. Three things move together:
//   true cart   one run of the cart SDE (sim5.cartStep, step 1/300 s, as the clip)
//   EKF         sim5.EKF, step 0.1 s (40 steps), textbook step model, Joseph update (as nlfilters / the clip)
//   cloud       "where the cart can be": sim5.ParticleFilter with the same SDE (method 'em', step 1/300 s), weighted by
//               the same readings and resampled after each one; blind for the whole run it is the clip's run-2 cloud.
// Panel: the EKF's heading belief pushed through cos(theta) by its tangent (yellow) and by the curve (blue), the
// cloud's headings under the belief, and the position along the track (EKF Gaussian, cloud histogram, end mark).
// Separate random streams for truth, readings and cloud: the same seed gives the same true path for every blind time.
import { S, THREE, COL, CSS, makeViewer, setCart, parkBody, Wheels, loadGhost, CartCloud, FloorEllipse, RangeRing,
  FloorArrow, Ribbon, val, onInput, onClick, seg, readout, plot2d, scales, stepper, fmt } from '../common.js';

const LENGTH = 1.0, T_RUN = 4.0, V = LENGTH / T_RUN;   // the command
const H_TRUE = 1 / 300;                                // truth and cloud step (s)
const TICK = 1 / 30, SUB = 10;                         // one tick = 10 truth steps
const EKF_EVERY = 3;                                   // filter step 0.1 s
const READ_EVERY = 15;                                 // a reading every 0.5 s
const N_TICKS = Math.round(T_RUN / TICK);              // 120
const BEACON = [1.75, 0.55], SIGMA_R = 0.03, SIGMA_B = 0.05;
const SD0 = 1e-3;                                      // start known to 1 mm and 1 mrad (scenario sd0)
const N_CLOUD = 4000, N_BODIES = 24;
const NSIG95 = Math.sqrt(S.CHI2_95_2);                 // the 95 % ellipse in (x, y)
const PAINT = 0xd9d6cf;

export async function init(panel) {
  const v = await makeViewer(panel, 'beacons.mjb', { lookat: [0.8, 0.0, 0.05], distance: 2.35, azimuth: 78, elevation: -34 });
  const geo = await loadGhost();

  // ---------------------------------------------------------------- stage
  v.setMocap('b0', [BEACON[0], BEACON[1], 0]);
  parkBody(v, 'b1'); parkBody(v, 'ukf');
  v.setMocap('shelf', [0.25, 2.0, 0]);
  const bodyName = (b) => v.mujoco.mj_id2name(v.model, v.mujoco.mjtObj.mjOBJ_BODY.value, b) || '';
  for (let g = 0; g < v.model.ngeom; g++) { const m = v.meshes[g]; if (m && /^ekf/.test(bodyName(v.model.geom_bodyid[g]))) { m.material = m.material.clone(); m.material.emissive = new THREE.Color(COL.ekf); m.material.emissiveIntensity = 0.45; m.material.opacity = Math.max(m.material.opacity, 0.6); } }
  v.forward();
  const lampG = v.id('geom', 'b0_lamp');
  const lampMat = v.meshes[lampG] && v.meshes[lampG].material;
  const lampPos = Array.from(v.data.geom_xpos.slice(3 * lampG, 3 * lampG + 3));
  const setLamp = (on) => {
    if (!lampMat) return;
    const c = new THREE.Color(on ? COL.reading : COL.fail);
    lampMat.color.copy(c); lampMat.emissive = c.clone(); lampMat.emissiveIntensity = on ? 0.9 : 0.35;
  };

  const seg2 = (a, b, w = 0.016) => { const r = new Ribbon(v.scene, PAINT, { n: 2, closed: false, opacity: 0.75 }); r.set([a, b], w); return r; };
  for (let x = 0; x < LENGTH - 0.02; x += 0.11) seg2([x, 0], [Math.min(x + 0.06, LENGTH), 0], 0.012);   // the command
  seg2([0, -0.1], [0, 0.1], 0.018);                                                                     // start mark
  const bx0 = 0.825, bx1 = 1.115, bh = 0.145;                                                           // the bay (as the clip)
  seg2([bx0, -bh], [bx1, -bh]); seg2([bx1, -bh], [bx1, bh]); seg2([bx1, bh], [bx0, bh]); seg2([bx0, bh], [bx0, -bh]);

  const cloud = new CartCloud(v.scene, geo, { max: N_CLOUD, bodies: N_BODIES, opacity: 0.26, dot: 0.022 });
  const ell = new FloorEllipse(v.scene, COL.ekf, { nsig: NSIG95, width: 0.014, fill: 0.10 });
  const arrow = new FloorArrow(v.scene, COL.ekf, { width: 0.02, head: 0.07, z: 0.135 });
  const ring = new RangeRing(v.scene, COL.reading);
  ring.visible = false;
  const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 1, 8, 1, true),
    new THREE.MeshBasicMaterial({ color: COL.reading, transparent: true, opacity: 0, depthWrite: false }));
  beam.frustumCulled = false; v.scene.add(beam);

  // white down-arrow above the true cart; pins at the EKF mean (yellow) and the cloud's mean (blue)
  const truthMark = new THREE.Group();
  const wm = new THREE.MeshBasicMaterial({ color: COL.truth });
  const cone = new THREE.Mesh(new THREE.ConeGeometry(0.035, 0.08, 20), wm); cone.rotation.x = -Math.PI / 2; cone.position.z = 0.23;
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.011, 0.11, 12), wm); shaft.rotation.x = Math.PI / 2; shaft.position.z = 0.32;
  truthMark.add(cone, shaft); v.scene.add(truthMark);
  const pin = (color, h) => {
    const g = new THREE.Group(), m = new THREE.MeshBasicMaterial({ color });
    const s = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, h, 8), m); s.rotation.x = Math.PI / 2; s.position.z = h / 2;
    const b = new THREE.Mesh(new THREE.SphereGeometry(0.018, 16, 12), m); b.position.z = h;
    g.add(s, b); g.visible = false; v.scene.add(g); return g;
  };
  const pinEkf = pin(COL.ekf, 0.46), pinCloud = pin(COL.cloud, 0.36);

  // ---------------------------------------------------------------- state
  let seed = 3, DT = 7, D = DT / T_RUN, tb = val(panel, 'tb'), speed = 1;
  let k, truth, rngT, rngM, pf, ekf, wheelT, wheelE, nRead, lastRead, flash, finished, cache;
  const tSee = () => T_RUN - tb;
  const seeing = (kk) => tSee() > 1e-9 && kk * TICK <= tSee() + 1e-9;

  function reading() {
    const zc = S.measure(truth, [BEACON], true);
    const z = [zc[0] + SIGMA_R * rngM.normal(), S.wrapAngle(zc[1] + SIGMA_B * rngM.normal())];
    ekf.update(z, [BEACON], SIGMA_R, { bearing: true, sigmaB: SIGMA_B });
    pf.update(z, [BEACON], SIGMA_R, { bearing: true, sigmaB: SIGMA_B, resample: false });
    pf.resample(true);
    nRead++; lastRead = { t: k * TICK, range: z[0], at: truth.slice() }; flash = 1;
  }

  function reset() {
    k = 0; finished = false; nRead = 0; lastRead = null; flash = 0;
    truth = new Float64Array([0, 0, 0]);
    rngT = new S.RNG(seed); rngM = new S.RNG(seed + 1000);
    const P0 = S.diag([SD0 * SD0, SD0 * SD0, SD0 * SD0]);
    ekf = new S.EKF([0, 0, 0], P0, D);
    pf = new S.ParticleFilter([0, 0, 0], P0, D, { nParticles: N_CLOUD, seed: seed + 2000, dtSub: H_TRUE, method: 'em' });
    wheelT = new Wheels([0, 0, 0]); wheelE = new Wheels([0, 0, 0]);
    if (seeing(0)) reading();
    ring.visible = false;
    summarise();
  }

  function tick() {
    if (k >= N_TICKS) { finished = true; return; }
    for (let i = 0; i < SUB; i++) S.cartStep(truth, V, 0, D, H_TRUE, rngT, { n: 1 });
    pf.predict(V, 0, TICK);
    k++;
    if (k % EKF_EVERY === 0) ekf.predict(V, 0, EKF_EVERY * TICK);
    if (k % READ_EVERY === 0 && seeing(k)) reading();
    if (k >= N_TICKS) finished = true;
  }

  /** Numbers of the present moment (also what _state() returns). */
  function summarise() {
    const frac = (k % EKF_EVERY) * TICK;                      // the EKF prediction over the part of the step
    const e = frac > 0 && !finished ? S.ekfPredict(ekf.m, ekf.P, V, 0, D, frac) : { m: ekf.m, P: ekf.P };
    const c = pf.meanCov();
    const th = e.m[2], dir = [Math.cos(th), Math.sin(th)];
    const along = (e.m[0] - c.m[0]) * dir[0] + (e.m[1] - c.m[1]) * dir[1];
    const blindFor = Math.max(0, k * TICK - Math.max(tSee(), 0));
    const lagTheory = blindFor > 0 ? V * blindFor - S.meanForwardDistance(V, D, blindFor) : 0;
    const cover = S.ellipseCoverage(pf.x, e.m, e.P, 0.95);
    const sTh = Math.sqrt(Math.max(e.P[2][2], 0));
    const err = S.errorStats(Array.from(truth), e.m, e.P);
    cache = { e, c, along, blindFor, lagTheory, cover, sTh, err,
      cosTangent: Math.cos(th), cosCurve: Math.cos(th) * Math.exp(-0.5 * sTh * sTh) };
    return cache;
  }

  // ---------------------------------------------------------------- controls
  onInput(panel, 'tb', () => { tb = val(panel, 'tb'); reset(); });
  seg(panel, 'DT', (x) => { DT = Number(x); D = DT / T_RUN; reset(); });
  seg(panel, 'speed', (x) => { speed = Number(x); });
  onClick(panel, 'go', reset);
  onClick(panel, 'new', () => { seed++; reset(); });
  reset();

  // ---------------------------------------------------------------- 3D
  function draw3d() {
    const { e, c } = cache, blind = !seeing(k) || (finished && tb > 0);
    setCart(v, 'cart', truth, wheelT.update(truth));
    setCart(v, 'ekf', e.m, wheelE.update(e.m));
    truthMark.position.set(truth[0] - 0.045 * Math.cos(truth[2]), truth[1] - 0.045 * Math.sin(truth[2]), 0);
    cloud.set(pf.x, N_CLOUD);
    cloud.visible = k * TICK > tSee() + 1e-9 || tSee() <= 1e-9;        // the cloud is drawn once the readings stop (as the clip)
    ell.set(e.m, e.P);
    arrow.set([e.m[0], e.m[1]], [0.3 * Math.cos(e.m[2]), 0.3 * Math.sin(e.m[2])]);
    const showPins = cache.blindFor > 0;
    pinEkf.visible = pinCloud.visible = showPins;
    pinEkf.position.set(e.m[0], e.m[1], 0); pinCloud.position.set(c.m[0], c.m[1], 0);
    setLamp(!blind);
    if (lastRead) {
      ring.visible = true;
      ring.set(BEACON, lastRead.range, 0.012 + 0.02 * flash);
      ring.opacity = 0.25 + 0.65 * flash;
      const a = new THREE.Vector3(lastRead.at[0], lastRead.at[1], 0.11), b = new THREE.Vector3(...lampPos);
      beam.position.copy(a).add(b).multiplyScalar(0.5);
      beam.scale.set(1, a.distanceTo(b), 1);
      beam.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
      beam.material.opacity = 0.8 * flash;
      if (blind) ring.opacity = 0.12;
    } else { ring.visible = false; beam.material.opacity = 0; }
    v.forward();
  }

  // ---------------------------------------------------------------- 2D panel
  const o = plot2d(panel.querySelector('canvas.plot'));
  const TH = [-2.0, 2.0], CY = [-0.15, 1.1], XR = [-0.1, 1.3];
  const Y_EDGES = Array.from({ length: 52 }, (_, i) => -0.1625 + 0.025 * i);   // [0.9875, 1.0125] centred on 1
  const QN = 241, QZ = Array.from({ length: QN }, (_, i) => -4 + 8 * i / (QN - 1));
  const QW = QZ.map((z) => Math.exp(-0.5 * z * z)); const QS = QW.reduce((a, b) => a + b, 0); QW.forEach((w, i) => { QW[i] = w / QS; });
  const rgba = (hex, a) => { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; };

  function text(g, s, x, y, col = CSS.grey, align = 'left', base = 'alphabetic', font = '11px Inter, sans-serif') {
    g.font = font; g.fillStyle = col; g.textAlign = align; g.textBaseline = base; g.fillText(s, x, y);
  }

  function draw2d() {
    o.clear();
    const g = o.g, W = o.w, Hh = o.h, { e, c } = cache;
    const th = e.m[2], sTh = cache.sTh;
    // header
    text(g, 'EKF', 12, 20, CSS.est, 'left', 'alphabetic', '600 15px Inter, sans-serif');
    text(g, 'same formulas; F, H = Jacobians at the mean', 46, 20, CSS.white, 'left', 'alphabetic', '12px Inter, sans-serif');
    g.fillStyle = rgba(CSS.est, 0.35); g.fillRect(12, 32, 10, 9); g.strokeStyle = CSS.est; g.strokeRect(12.5, 32.5, 9, 8);
    text(g, 'via the tangent (EKF)', 27, 41, CSS.est);
    g.fillStyle = rgba(CSS.cloud, 0.6); g.fillRect(160, 32, 10, 9);
    text(g, 'via the curve (exact)', 175, 41, CSS.cloud);

    // layout
    const hC = Math.max(96, Math.round(Hh * 0.24)), hB = Math.max(58, Math.round(Hh * 0.14));
    const topA = 52, botA = Hh - hC - hB - 34;
    const colL = 36, colR = 92;                                  // the output column
    const sA = scales(o, TH, CY, [colR + 8, 12, topA, Hh - botA]);
    const [l, t, w, h] = sA.box;
    // grid
    g.strokeStyle = 'rgba(255,255,255,0.07)'; g.lineWidth = 1;
    [0, 0.5, 1].forEach((y) => { g.beginPath(); g.moveTo(colL, sA.Y(y)); g.lineTo(l + w, sA.Y(y)); g.stroke(); text(g, y.toFixed(1), colL - 4, sA.Y(y), CSS.grey, 'right', 'middle'); });
    g.save(); g.translate(9, (t + t + h) / 2); g.rotate(-Math.PI / 2); text(g, 'forward share of a step, cos θ', 0, 0, CSS.grey, 'center', 'middle', '10.5px Inter, sans-serif'); g.restore();
    // belief band +-2 sd
    const b0 = Math.max(TH[0], th - 2 * sTh), b1 = Math.min(TH[1], th + 2 * sTh);
    g.fillStyle = rgba(CSS.est, 0.08); g.fillRect(sA.X(b0), t, Math.max(1, sA.X(b1) - sA.X(b0)), h);
    // the curve
    g.save(); g.beginPath(); g.rect(l, t, w, h); g.clip();
    g.strokeStyle = CSS.white; g.lineWidth = 2.2; g.beginPath();
    for (let i = 0; i <= 200; i++) { const a = TH[0] + (TH[1] - TH[0]) * i / 200, px = sA.X(a), py = sA.Y(Math.cos(a)); if (i) g.lineTo(px, py); else g.moveTo(px, py); }
    g.stroke();
    text(g, 'cos θ', sA.X(1.55), sA.Y(Math.cos(1.55)) - 8, CSS.white, 'left', 'bottom', '12px Inter, sans-serif');
    // the tangent at the mean
    const slope = -Math.sin(th);
    g.strokeStyle = CSS.est; g.lineWidth = 2; g.beginPath();
    g.moveTo(sA.X(TH[0]), sA.Y(Math.cos(th) + slope * (TH[0] - th))); g.lineTo(sA.X(TH[1]), sA.Y(Math.cos(th) + slope * (TH[1] - th))); g.stroke();
    g.restore();
    // the two means, carried to the output column
    const yT = cache.cosTangent, yC = cache.cosCurve;
    g.setLineDash([4, 4]); g.lineWidth = 1.2;
    g.strokeStyle = CSS.est; g.beginPath(); g.moveTo(colL, sA.Y(yT)); g.lineTo(sA.X(th), sA.Y(yT)); g.stroke();
    g.strokeStyle = CSS.cloud; g.beginPath(); g.moveTo(colL, sA.Y(yC)); g.lineTo(sA.X(th), sA.Y(yC)); g.stroke();
    g.setLineDash([]);
    g.strokeStyle = 'rgba(255,235,64,0.5)'; g.beginPath(); g.moveTo(sA.X(th), t); g.lineTo(sA.X(th), t + h); g.stroke();
    g.fillStyle = CSS.cloud; g.beginPath(); g.arc(sA.X(th), sA.Y(yC), 4.5, 0, 2 * Math.PI); g.fill();
    g.fillStyle = CSS.est; g.beginPath(); g.arc(sA.X(th), sA.Y(yT), 4.5, 0, 2 * Math.PI); g.fill();

    // output column: distribution of cos(theta) under the EKF's heading belief, by the curve and by the tangent
    const nb = Y_EDGES.length - 1, mc = new Float64Array(nb), mt = new Float64Array(nb);
    for (let q = 0; q < QN; q++) {
      const y = Math.cos(th + sTh * QZ[q]), j = Math.floor((y - Y_EDGES[0]) / 0.025);
      if (j >= 0 && j < nb) mc[j] += QW[q];
    }
    const sdT = Math.max(Math.abs(Math.sin(th)) * sTh, 1e-9);
    for (let j = 0; j < nb; j++) mt[j] = S.normalCdf((Y_EDGES[j + 1] - yT) / sdT) - S.normalCdf((Y_EDGES[j] - yT) / sdT);
    const pc = Math.max(...mc, 1e-12), pt = Math.max(...mt, 1e-12), cw = colR - colL;
    g.strokeStyle = CSS.faint; g.beginPath(); g.moveTo(colR + 0.5, t); g.lineTo(colR + 0.5, t + h); g.stroke();
    for (let j = 0; j < nb; j++) {
      const y0 = sA.Y(Y_EDGES[j + 1]), y1 = sA.Y(Y_EDGES[j]);
      if (mc[j] > 0) { g.fillStyle = rgba(CSS.cloud, 0.75); g.fillRect(colR - cw * mc[j] / pc, y0, cw * mc[j] / pc, y1 - y0); }
    }
    g.strokeStyle = CSS.est; g.lineWidth = 1.6; g.beginPath();
    for (let j = 0; j < nb; j++) {
      const y0 = sA.Y(Y_EDGES[j + 1]), y1 = sA.Y(Y_EDGES[j]), x = colR - cw * mt[j] / pt;
      if (j) g.lineTo(x, y1); else g.moveTo(x, y1);
      g.lineTo(x, y0);
    }
    g.stroke();

    // B: belief about the heading: EKF Gaussian (yellow) against the cloud's headings (blue histogram)
    const sB = scales(o, TH, [0, 1], [l, 12, t + h + 8, Hh - (t + h + 8 + hB)]);
    const [, tB, , hBb] = sB.box;
    const nh = 80, hw = (TH[1] - TH[0]) / nh, hist = new Float64Array(nh);
    for (let p = 0; p < pf.Np; p++) { const j = Math.floor((pf.x[3 * p + 2] - TH[0]) / hw); if (j >= 0 && j < nh) hist[j]++; }
    const dens = Array.from(hist, (n) => n / (pf.Np * hw));
    const sG = Math.max(sTh, 1e-4), pdf = (a) => Math.exp(-0.5 * ((a - th) / sG) ** 2) / (sG * Math.sqrt(2 * Math.PI));
    const dmax = Math.max(...dens, 1e-9), peak = Math.max(dmax, Math.min(pdf(th), 1.5 * dmax)) * 1.05;
    g.fillStyle = rgba(CSS.cloud, 0.7);
    for (let j = 0; j < nh; j++) if (dens[j] > 0) { const hh = hBb * dens[j] / peak; g.fillRect(sB.X(TH[0] + j * hw), tB + hBb - hh, sB.X(TH[0] + (j + 1) * hw) - sB.X(TH[0] + j * hw) - 0.5, hh); }
    g.save(); g.beginPath(); g.rect(l, tB - 2, w, hBb + 2); g.clip();
    g.strokeStyle = CSS.est; g.lineWidth = 1.8; g.beginPath();
    for (let i = 0; i <= 400; i++) { const a = TH[0] + (TH[1] - TH[0]) * i / 400, py = tB + hBb - hBb * pdf(a) / peak; if (i) g.lineTo(sB.X(a), py); else g.moveTo(sB.X(a), py); }
    g.stroke(); g.restore();
    g.strokeStyle = CSS.faint; g.lineWidth = 1; g.beginPath(); g.moveTo(l, tB + hBb + 0.5); g.lineTo(l + w, tB + hBb + 0.5); g.stroke();
    [-90, -45, 0, 45, 90].forEach((d) => text(g, `${d}°`, sB.X(d * Math.PI / 180), tB + hBb + 4, CSS.grey, 'center', 'top'));
    text(g, 'belief about the heading θ', l + w / 2, tB + hBb + 17, CSS.white, 'center', 'top', '11.5px Inter, sans-serif');

    // C: position along the track
    const sC = scales(o, XR, [0, 1], [26, 16, Hh - hC + 22, 34]);
    const [lC, tC, wC, hCb] = sC.box;
    const nx = 70, xw = (XR[1] - XR[0]) / nx, hx = new Float64Array(nx);
    for (let p = 0; p < pf.Np; p++) { const j = Math.floor((pf.x[3 * p] - XR[0]) / xw); if (j >= 0 && j < nx) hx[j]++; }
    const dx = Array.from(hx, (n) => n / (pf.Np * xw));
    const sX = Math.sqrt(Math.max(e.P[0][0], 1e-10)), pdfX = (x) => Math.exp(-0.5 * ((x - e.m[0]) / sX) ** 2) / (sX * Math.sqrt(2 * Math.PI));
    const xmax = Math.max(...dx, 1e-9), pk = Math.max(xmax, Math.min(pdfX(e.m[0]), 1.5 * xmax)) * 1.08;
    g.fillStyle = rgba(CSS.cloud, 0.7);
    for (let j = 0; j < nx; j++) if (dx[j] > 0) { const hh = Math.min(hCb, hCb * dx[j] / pk); g.fillRect(sC.X(XR[0] + j * xw), tC + hCb - hh, sC.X(XR[0] + (j + 1) * xw) - sC.X(XR[0] + j * xw) - 0.5, hh); }
    g.save(); g.beginPath(); g.rect(lC, tC - 2, wC, hCb + 2); g.clip();
    g.strokeStyle = CSS.est; g.lineWidth = 1.8; g.beginPath();
    for (let i = 0; i <= 500; i++) { const x = XR[0] + (XR[1] - XR[0]) * i / 500, py = tC + hCb - hCb * pdfX(x) / pk; if (i) g.lineTo(sC.X(x), py); else g.moveTo(sC.X(x), py); }
    g.stroke(); g.restore();
    g.strokeStyle = CSS.faint; g.lineWidth = 1; g.beginPath(); g.moveTo(lC, tC + hCb + 0.5); g.lineTo(lC + wC, tC + hCb + 0.5); g.stroke();
    [0, 0.5, 1].forEach((x) => text(g, x.toFixed(1), sC.X(x), tC + hCb + 4, CSS.grey, 'center', 'top'));
    text(g, 'position along the track x (m)', lC + wC / 2, tC + hCb + 17, CSS.white, 'center', 'top', '11.5px Inter, sans-serif');
    // end mark, true cart, the two means
    g.setLineDash([3, 3]); g.strokeStyle = CSS.white; g.beginPath(); g.moveTo(sC.X(1), tC - 6); g.lineTo(sC.X(1), tC + hCb); g.stroke(); g.setLineDash([]);
    text(g, 'end mark', sC.X(1) + 4, tC - 8, CSS.grey, 'left', 'bottom', '10.5px Inter, sans-serif');
    const tri = (x, y, col, down = true) => { g.fillStyle = col; g.beginPath(); g.moveTo(x - 5, y); g.lineTo(x + 5, y); g.lineTo(x, y + (down ? 8 : -8)); g.closePath(); g.fill(); };
    tri(sC.X(truth[0]), tC + hCb + 1, CSS.white, false);
    tri(sC.X(e.m[0]), tC + 4, CSS.est); tri(sC.X(c.m[0]), tC + 4, CSS.cloud);
    if (cache.blindFor > 0 && Math.abs(cache.along) >= 0.005) {
      const xa = sC.X(c.m[0]), xb = sC.X(e.m[0]), yb = tC - 2;
      g.strokeStyle = CSS.white; g.lineWidth = 1.2; g.beginPath(); g.moveTo(xa, yb + 4); g.lineTo(xa, yb); g.lineTo(xb, yb); g.lineTo(xb, yb + 4); g.stroke();
      text(g, `${fmt(cache.along, 2)} m`, Math.min(xa, xb) - 6, yb + 2, CSS.white, 'right', 'bottom', '600 13px Inter, sans-serif');
    }
  }

  function info() {
    const blind = !seeing(k) || (finished && tb > 0);
    const phase = tSee() <= 1e-9 ? '<span class="c-red">no readings</span>'
      : blind ? `<span class="c-red">no readings</span> after ${fmt(tSee(), 1)} s` : '<span class="c-green">a reading every 0.5 s</span>';
    let s = `t = <b>${fmt(k * TICK, 1)} s</b> · ${phase}`;
    if (cache.blindFor > 0) s += ` · EKF ahead of the cloud's mean <b>${fmt(cache.along, 3)} m</b> (theory ${fmt(cache.lagTheory, 3)} m${tSee() > 1e-9 ? ', from an exactly known start' : ''})`;
    s += ` · cloud in the EKF's 95 % ellipse <b>${Math.round(100 * cache.cover)} %</b>`;
    readout(panel, 'info', s);
  }

  // ---------------------------------------------------------------- loop
  const advance = stepper(TICK);
  let lastK = -1;
  v.onFrame((dt) => {
    if (!finished) advance(dt, speed, () => { tick(); });
    flash = Math.max(0, flash - dt * speed / 0.4);
    if (k !== lastK || flash > 0) { summarise(); lastK = k; }
    draw3d(); draw2d(); info();
  });
  summarise(); draw3d(); draw2d(); info();

  /** Test hook: advance the simulation by `sec` seconds of run time at once. */
  panel._advance = (sec) => { const n = Math.round(sec / TICK); for (let i = 0; i < n && !finished; i++) tick(); summarise(); draw3d(); draw2d(); info(); return k * TICK; };
  panel._state = () => {
    const { e, c } = cache, sd = (P) => [0, 1, 2].map((i) => Math.sqrt(Math.max(P[i][i], 0)));
    return {
      t: k * TICK, finished, seed, DT, D, tBlind: tb, tSee: tSee(), seeing: seeing(k), nReadings: nRead,
      truth: Array.from(truth), ekf: { m: e.m, sd: sd(e.P) }, cloud: { n: pf.Np, m: c.m, sd: sd(c.P) },
      blindFor: cache.blindFor, ekfAheadOfCloudMean: cache.along, lagTheoryKnownStart: cache.lagTheory,
      cover95: cache.cover, neesXY: cache.err.neesXY, posErr: cache.err.pos,
      meanCos: { tangent: cache.cosTangent, curve: cache.cosCurve }, speed,
      camera: v.camera.position.toArray().map((x) => +x.toFixed(3)),
    };
  };
}
