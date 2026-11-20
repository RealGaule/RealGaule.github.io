// 1.2 The gain: trust the model or the sensor (clip s12a_gain).
// One drive shown on three lanes: the three white carts are the same true cart. Every 0.5 s one position reading
// (green line across the lanes) feeds three linear Kalman filters that differ only in the Q and R they assume:
//   far lane    "trusts the sensor"  Q x 100   (K_p = 0.72)
//   middle lane "your filter"        Q and R from the sliders (default = truth: the lecture gain [0.33, 0.13])
//   near lane   "trusts the model"   Q x 0.01  (K_p = 0.12)
// The model and every number are those of the clip (out/s12a_gain_facts.json): the lecture example of Lecture 10
// at length scale 1:10, q = 5e-4 m^2/s^3, R = 0.01 m^2, dt = 0.5 s. The keys push the true cart (an acceleration
// the filters do not know about). Numerics: sim5.js (pvModel, KalmanFilter, steadyState, trueErrorCovariance).
// The 2D panel: top, the (p, v) belief of your filter (2-sigma ellipse, samples, tilt axis, the gain splitting the
// innovation); bottom, stationary theory against the assumed Q/R ratio (steady K_p and the true RMS error).
import { S, THREE, COL, CSS, makeViewer, setCart, Wheels, FloorEllipse, FloorArrow, Ribbon, keyboard, val, onInput,
  onClick, seg, toggle, readout, ctl, plot2d, scales, drawEllipse, stepper, fmt } from '../common.js';

const Q_TRUE = 5e-4, R_TRUE = 0.01, DT = 0.5, H = S.H_POS, H_FINE = 1 / 90;
const X0 = [-0.3, 0.25], WRAP_AT = 1.7, WRAP_BY = 2.0;       // wrap by 2 m: a multiple of the 0.5 m ticks AND of the
                                                               // viewer's floor checker (1 m cells, period 2 m)
const TICK = 0.5, LINE_HALF = 10;                              // endless track: lane lines 20 m long, ticks re-placed
const ACCEL = 0.4, V_LO = 0.08, V_HI = 0.55, V_MAX = 0.75;     // pushes (m/s^2) and the speed band of the drive
const N_SAMP = 150, ARROW_T = 1.2, CORR_T = 0.35, RMS_N = 20;
const ELL_Y = 0.09;                                            // half-width / 2 of the lane ellipse (m): 2 sigma = 0.18

const LANES = [
  { key: 'model', title: 'trusts the model', y: -0.7, cart: 'cart0', est: 'est0', qf: 0.01, rf: 1 },
  { key: 'yours', title: 'your filter', y: 0.0, cart: 'cart1', est: 'est1', qf: 1, rf: 1 },
  { key: 'sensor', title: 'trusts the sensor', y: 0.7, cart: 'cart2', est: 'est2', qf: 100, rf: 1 },
];

const lerp = (a, b, t) => a + (b - a) * t;
const nf = (f) => String(Number(f.toPrecision(2)));
const smooth = (t) => t * t * (3 - 2 * t);
const lerpVec = (a, b, t) => a.map((x, i) => lerp(x, b[i], t));
const lerpMat = (A, B, t) => A.map((r, i) => r.map((x, j) => lerp(x, B[i][j], t)));

/** Mean and covariance of a (p, v) belief carried forward by tau seconds under the assumed q (sim5 kfPredict). */
function predictTo(x, P, tau, q) {
  if (tau <= 0) return { x, P };
  const m = S.pvModel(tau, q);
  return S.kfPredict(x, P, m.F, m.Q);
}

/**
 * Make the 3.5 m lanes of lanes.mjb look endless, so that the wrap is invisible: the two edge lines of every lane are
 * stretched to 2 * LINE_HALF m, and the 8 ticks of a lane are re-placed every frame on the 0.5 m grid around the camera.
 * The geoms stay the model's own (same paint), only their size / position in the model change.
 */
function endlessTrack(v) {
  const m = v.model, T = v.mujoco.mjtGeom.mjGEOM_BOX.value, lines = [], ticks = [];
  for (let g = 0; g < m.ngeom; g++) {
    if (m.geom_bodyid[g] !== 0 || m.geom_type[g] !== T) continue;
    const sx = m.geom_size[3 * g];
    if (sx > 1) lines.push(g);
    else if (sx < 0.1) ticks.push(g);
  }
  for (const g of lines) { m.geom_size[3 * g] = LINE_HALF; v.refreshGeom(g); }
  const lanes = new Map();                                     // tick geoms grouped by lane (their y)
  for (const g of ticks) {
    const y = Math.round(m.geom_pos[3 * g + 1] * 100);
    if (!lanes.has(y)) lanes.set(y, []);
    lanes.get(y).push(g);
  }
  const info = { lines: lines.length, ticks: ticks.length, tickX: [NaN, NaN],
    lineX: lines.length ? [m.geom_pos[3 * lines[0]] - LINE_HALF, m.geom_pos[3 * lines[0]] + LINE_HALF] : null };
  info.update = (cx) => {
    const base = TICK * Math.floor((cx - 1.6) / TICK);         // ticks cover about cx - 1.6 .. cx + 1.9 m
    for (const gs of lanes.values()) gs.forEach((g, k) => { m.geom_pos[3 * g] = base + TICK * k; });
    const n = Math.max(0, ...[...lanes.values()].map((gs) => gs.length));
    info.tickX = [base, base + TICK * (n - 1)];
  };
  return info;
}

/** Steady filter for an assumed Q = qf * Q_TRUE and R = rf * R_TRUE. */
function steady(qf, rf) {
  const m = S.pvModel(DT, Q_TRUE * qf);
  return S.steadyState(m.F, H, m.Q, [[R_TRUE * rf]]);
}

/** A text label that always faces the camera (canvas texture on a sprite). */
function textSprite(scene, w = 0.5, h = 0.125) {
  const c = document.createElement('canvas');
  c.width = 512; c.height = 128;
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  sp.scale.set(w, h, 1);
  sp.renderOrder = 5;
  scene.add(sp);
  let last = '';
  sp.draw = (l1, l2) => {
    if (l1 + '|' + l2 === last) return;
    last = l1 + '|' + l2;
    const g = c.getContext('2d');
    g.clearRect(0, 0, 512, 128);
    g.textBaseline = 'middle';
    g.font = '600 44px Inter, sans-serif'; g.fillStyle = CSS.white; g.fillText(l1, 8, 36);
    g.font = '40px Inter, sans-serif'; g.fillStyle = CSS.yellow; g.fillText(l2, 8, 92);
    tex.needsUpdate = true;
  };
  return sp;
}

function niceTicks(lo, hi, n = 4) {
  const raw = (hi - lo) / n, p = 10 ** Math.floor(Math.log10(raw)), r = raw / p;
  const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * p;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-12; t += step) out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  return { ticks: out, digits: Math.max(0, -Math.floor(Math.log10(step) + 1e-9)) };
}

export async function init(panel) {
  const v = await makeViewer(panel, 'lanes.mjb', { lookat: [X0[0] + 0.15, -0.12, 0.0], distance: 2.45, azimuth: 78, elevation: -42 });
  const bodyName = (b) => v.mujoco.mj_id2name(v.model, v.mujoco.mjtObj.mjOBJ_BODY.value, b) || '';
  for (let gi = 0; gi < v.model.ngeom; gi++) {
    const m = v.meshes[gi];
    if (m && /^est/.test(bodyName(v.model.geom_bodyid[gi]))) {
      m.material = m.material.clone();
      m.material.emissive = new THREE.Color(COL.est);
      m.material.emissiveIntensity = 0.4;
    }
  }
  const track = endlessTrack(v);
  const rng = new S.RNG(97);
  const advance = stepper(H_FINE);
  const kb = keyboard(panel, ['ArrowUp', 'ArrowDown']);
  const plotCanvas = panel.querySelector('canvas.plot');
  const o = plot2d(plotCanvas);

  // ---------- precomputed theory curve (stationary, drive without pushes) ----------
  const fineModel = S.pvModel(DT, Q_TRUE);
  const curve = { l: [], kp: [], kv: [], rms: [] };
  for (let l = -3; l <= 3.0001; l += 0.05) {
    const st = steady(10 ** l, 1);
    const tr = S.trueErrorCovariance(fineModel.F, H, fineModel.Q, [[R_TRUE]], st.K);
    curve.l.push(l); curve.kp.push(st.K[0][0]); curve.kv.push(st.K[1][0]); curve.rms.push(Math.sqrt(tr.SigmaPost[0][0]));
  }
  const theoryAt = (qf, rf) => {
    const st = steady(qf, rf), tr = S.trueErrorCovariance(fineModel.F, H, fineModel.Q, [[R_TRUE]], st.K);
    return { Kp: st.K[0][0], Kv: st.K[1][0], rms: Math.sqrt(tr.SigmaPost[0][0]), sigmaClaimed: Math.sqrt(st.Ppost[0][0]) };
  };

  // ---------- 3D objects ----------
  const yLine = [LANES[0].y - 0.36, LANES[2].y + 0.36];
  const reads = [];                                            // { x, age, ribbon }
  const readPool = Array.from({ length: 6 }, () => { const r = new Ribbon(v.scene, COL.reading, { n: 2, closed: false, z: 0.003, opacity: 0 }); r.visible = false; return r; });
  for (const L of LANES) {
    L.ell = new FloorEllipse(v.scene, COL.est, { width: 0.012, fill: 0.10 });
    L.arT = new FloorArrow(v.scene, COL.truth, { width: 0.022, head: 0.065, z: 0.17 });
    L.arE = new FloorArrow(v.scene, COL.est, { width: 0.022, head: 0.065, z: 0.19 });
    L.label = textSprite(v.scene);
    L.wE = new Wheels([X0[0], L.y, 0]);
  }
  const wT = new Wheels([X0[0], 0, 0]);

  // ---------- state ----------
  let t, tNext, x, nReads, wraps, lastZ, readAge, followX = X0[0] + 0.15, rate = 1, others = true;
  let samp, sampPri, sampW;                                    // ensemble that carries the belief of your filter
  const span = { p: 0.3, v: 0.15 };
  const push = { up: false, down: false };

  function laneFilter(L) {
    const m = S.pvModel(DT, Q_TRUE * L.qf), st = steady(L.qf, L.rf);
    L.kf = new S.KalmanFilter({ F: m.F, H, Q: m.Q, R: [[R_TRUE * L.rf]], x0: X0, P0: st.Ppost });
    L.xPri = X0.slice(); L.PPri = st.Ppost; L.xPost = X0.slice(); L.PPost = st.Ppost;
    L.K = [st.K[0][0], st.K[1][0]]; L.nu = 0; L.err2 = [];
    L.steady = theoryAt(L.qf, L.rf);
  }

  function drawSamples(mean, P) {
    const Lc = S.cholesky(P);
    samp = []; sampPri = []; sampW = [];
    for (let i = 0; i < N_SAMP; i++) {
      const n = [rng.normal(), rng.normal()];
      const s = [mean[0] + Lc[0][0] * n[0], mean[1] + Lc[1][0] * n[0] + Lc[1][1] * n[1]];
      samp.push(s); sampPri.push(s.slice()); sampW.push([rng.normal(), rng.normal()]);
    }
  }
  /** Ensemble member i carried forward by tau under the assumed q (exact in law: F s + chol(Q(tau)) w). */
  function carrier(tau, q) {
    if (tau <= 1e-9) return null;
    const m = S.pvModel(tau, q);
    return { F: m.F, L: S.cholesky(m.Q) };
  }
  function carry(s, w, c) {
    if (!c) return s.slice();
    const y = S.mulVec(c.F, s);
    return [y[0] + c.L[0][0] * w[0], y[1] + c.L[1][0] * w[0] + c.L[1][1] * w[1]];
  }

  function reset() {
    t = 0; tNext = DT; nReads = 0; wraps = 0; lastZ = null; readAge = 99;
    const tuned = steady(1, 1).Ppost, Lc = S.cholesky(tuned), n = [rng.normal(), rng.normal()];
    x = [X0[0] + Lc[0][0] * n[0], X0[1] + Lc[1][0] * n[0] + Lc[1][1] * n[1]];   // true start ~ N(x0, tuned steady P+)
    LANES.forEach(laneFilter);
    drawSamples(X0, LANES[1].PPost);
    reads.length = 0;
    readPool.forEach((r) => { r.visible = false; });
    wT.reset([x[0], 0, 0]);
    LANES.forEach((L) => L.wE.reset([X0[0], L.y, 0]));
  }

  function setYours() {
    const L = LANES[1];
    L.qf = 10 ** val(panel, 'Q'); L.rf = 10 ** val(panel, 'R');
    const m = S.pvModel(DT, Q_TRUE * L.qf);
    L.kf.Q = m.Q; L.kf.R = [[R_TRUE * L.rf]];
    const st = steady(L.qf, L.rf);                              // the new setting starts at its own steady belief
    L.kf.P = st.Ppost.map((r) => r.slice());
    L.PPost = L.kf.P; L.PPri = L.kf.P; L.xPri = L.xPost.slice();
    L.K = [st.K[0][0], st.K[1][0]];
    drawSamples(L.xPost, L.PPost);
    L.steady = theoryAt(L.qf, L.rf);
    const show = (name, f) => { const out = ctl(panel, name).parentElement.querySelector('output'); if (out) out.textContent = '×' + nf(f); };
    show('Q', L.qf); show('R', L.rf);
  }

  function reading() {
    const z = x[0] + Math.sqrt(R_TRUE) * rng.normal();
    nReads++; lastZ = z; readAge = 0;
    for (const L of LANES) {
      L.kf.predict();
      L.xPri = L.kf.x.slice(); L.PPri = L.kf.P;
      const c = L.kf.correct([z]);
      L.xPost = L.kf.x.slice(); L.PPost = L.kf.P;
      L.K = [c.K[0][0], c.K[1][0]]; L.nu = c.nu[0];
      L.err2.push((L.xPost[0] - x[0]) ** 2); if (L.err2.length > RMS_N) L.err2.shift();
    }
    // the ensemble of your filter: carried to the reading, then conditioned on it (perturbed reading, same K)
    const Y = LANES[1], cDT = carrier(DT, Q_TRUE * Y.qf), sr = Math.sqrt(R_TRUE * Y.rf);
    for (let i = 0; i < N_SAMP; i++) {
      const pri = carry(samp[i], sampW[i], cDT);
      const innov = z + sr * rng.normal() - pri[0];
      sampPri[i] = pri;
      samp[i] = [pri[0] + Y.K[0] * innov, pri[1] + Y.K[1] * innov];
      sampW[i] = [rng.normal(), rng.normal()];
    }
    reads.unshift({ x: z, age: 0 });
    if (reads.length > readPool.length) reads.pop();
  }

  function wrap() {
    x[0] -= WRAP_BY; wraps++;
    if (lastZ !== null) lastZ -= WRAP_BY;
    for (const L of LANES) { L.kf.x[0] -= WRAP_BY; L.xPri[0] -= WRAP_BY; L.xPost[0] -= WRAP_BY; L.wE.prev[0] -= WRAP_BY; }
    for (let i = 0; i < N_SAMP; i++) { samp[i][0] -= WRAP_BY; sampPri[i][0] -= WRAP_BY; }
    reads.forEach((r) => { r.x -= WRAP_BY; });
    wT.prev[0] -= WRAP_BY;
  }

  /** Displayed belief of lane L, tau after the last reading: the correction is drawn over CORR_T s (prior + beta K nu). */
  function belief(L, tau) {
    const q = Q_TRUE * L.qf, post = predictTo(L.xPost, L.PPost, tau, q);
    if (nReads === 0 || tau >= CORR_T) return { x: post.x, P: post.P, beta: 1 };
    const pri = predictTo(L.xPri, L.PPri, tau, q), b = smooth(tau / CORR_T);
    return { x: lerpVec(pri.x, post.x, b), P: lerpMat(pri.P, post.P, b), beta: b, pri: pri.x };
  }

  // ---------- controls ----------
  onInput(panel, 'Q', setYours);
  onInput(panel, 'R', setYours);
  seg(panel, 'rate', (r) => { rate = Number(r); });
  toggle(panel, 'others', (on) => { others = on; });
  onClick(panel, 'reset', reset);
  for (const [name, k] of [['up', 'up'], ['down', 'down']]) {
    const b = ctl(panel, name);
    const on = (e) => { e.preventDefault(); push[k] = true; b.classList.add('on'); };
    const off = () => { push[k] = false; b.classList.remove('on'); };
    b.addEventListener('pointerdown', on);
    ['pointerup', 'pointerleave', 'pointercancel'].forEach((ev) => b.addEventListener(ev, off));
  }
  reset();
  setYours();

  let accel = 0;
  const Lfine = S.cholesky(S.pvModel(H_FINE, Q_TRUE).Q), Ffine = S.pvModel(H_FINE, Q_TRUE).F;
  function truthStep(h) {
    const up = push.up || kb.down('ArrowUp'), dn = push.down || kb.down('ArrowDown');
    let u = ACCEL * ((up ? 1 : 0) - (dn ? 1 : 0));
    if (!up && !dn) {                                          // keep the drive in a band of speeds when nobody pushes
      if (x[1] < V_LO) u = (V_LO - x[1]) * 1.0;
      else if (x[1] > V_HI) u = (V_HI - x[1]) * 1.0;
    }
    accel = u;
    const n = [rng.normal(), rng.normal()], y = S.mulVec(Ffine, x);
    x = [y[0] + 0.5 * u * h * h + Lfine[0][0] * n[0], y[1] + u * h + Lfine[1][0] * n[0] + Lfine[1][1] * n[1]];
    if (up || dn) x[1] = Math.min(Math.max(x[1], 0), V_MAX);
    t += h;
    if (t >= tNext - 1e-9) { reading(); tNext += DT; }
    if (x[0] > WRAP_AT) wrap();
  }

  // ---------- one frame ----------
  let frames = 0, lastBel = null;
  v.onFrame((dt) => {
    frames++;
    advance(dt, rate, truthStep);
    readAge += dt * rate;
    const tau = Math.max(0, t - (tNext - DT));

    // camera follows the true cart along the lanes; the user's orbit, zoom and pan are kept
    const want = x[0] + 0.15, d = want - followX;
    followX = want;
    v.camera.position.x += d; v.controls.target.x += d; v.homeView.lookat[0] = followX;
    track.update(followX);

    const phiT = wT.update([x[0], 0, 0]);
    const bels = LANES.map((L) => belief(L, tau));
    LANES.forEach((L, i) => {
      const show = others || L.key === 'yours', b = bels[i];
      if (show) {
        setCart(v, L.cart, [x[0], L.y, 0], phiT);
        setCart(v, L.est, [b.x[0], L.y, 0], L.wE.update([b.x[0], L.y, 0]));
        L.ell.set([b.x[0], L.y], [[b.P[0][0], 0], [0, ELL_Y * ELL_Y]]);
        L.arT.set([x[0], L.y + 0.04], [ARROW_T * x[1], 0]);
        L.arE.set([b.x[0], L.y - 0.04], [ARROW_T * b.x[1], 0]);
        L.label.position.set(followX - 0.72, L.y + 0.24, 0.05);
        const dd = v.camera.position.distanceTo(L.label.position); L.label.scale.set(0.21 * dd, 0.0525 * dd, 1);
        const qtxt = L.key === 'yours' ? `Q×${nf(L.qf)} R×${nf(L.rf)}` : `Q×${nf(L.qf)}`;
        L.label.draw(L.title, `${qtxt} · Kp = ${fmt(L.steady.Kp, 2)}`);
      } else {
        v.setMocap(L.cart, [0, 0, -5]); v.setMocap(L.est, [0, 0, -5]);
        L.arT.set([0, 0], [0, 0]); L.arE.set([0, 0], [0, 0]);
      }
      L.ell.visible = show; L.label.visible = show;
      if (!show) { L.arT.visible = false; L.arE.visible = false; }
    });
    v.forward();

    // readings: green lines across the lanes, the newest bright, older ones faint
    readPool.forEach((r, i) => {
      const rd = reads[i];
      if (!rd) { r.visible = false; return; }
      r.visible = true;
      const age = i === 0 ? readAge : 99;
      r.mesh.material.opacity = i === 0 ? lerp(0.95, 0.45, Math.min(1, age / 1.0)) : Math.max(0.08, 0.32 - 0.06 * i);
      r.set([[rd.x, yLine[0]], [rd.x, yLine[1]]], i === 0 ? 0.022 : 0.014);
    });

    lastBel = bels[1];
    drawPlot(bels[1], tau);
    if (frames % 6 === 1) info();
  });

  function info() {
    const r = (L) => (L.err2.length ? Math.sqrt(L.err2.reduce((a, b) => a + b, 0) / L.err2.length) : NaN);
    const Y = LANES[1];
    const parts = others
      ? `<span class="c-yellow">${fmt(r(LANES[2]), 3)}</span> / <span class="c-yellow"><b>${fmt(r(Y), 3)}</b></span> / <span class="c-yellow">${fmt(r(LANES[0]), 3)}</span> m (sensor / yours / model)`
      : `<b class="c-yellow">${fmt(r(Y), 3)}</b> m`;
    readout(panel, 'info', `your gain <b class="c-yellow">K = [${fmt(Y.K[0], 2)}, ${fmt(Y.K[1], 2)}]</b> · speed <b>${fmt(x[1], 2)} m/s</b>${accel ? ' <span class="c-white">(pushed)</span>' : ''} · position RMS, last ${RMS_N} readings: ${parts}`);
  }

  // ---------- the 2D picture ----------
  function drawPlot(b, tau) {
    o.clear();
    if (o.w < 50 || o.h < 80) return;
    const g = o.g, Y = LANES[1];
    const H1 = Math.round(o.h * 0.57);

    // top: belief of your filter in the (p, v) plane, window centred on the estimate
    const sp = Math.sqrt(b.P[0][0]), sv = Math.sqrt(b.P[1][1]);
    const reach = Math.max(3.3 * sp, 0.14, 1.5 * Math.abs(x[0] - b.x[0]), lastZ === null ? 0 : 1.25 * Math.abs(lastZ - b.x[0]));
    span.p = lerp(span.p, reach, 0.15);
    span.v = lerp(span.v, Math.max(3.3 * sv, 0.07, 1.5 * Math.abs(x[1] - b.x[1])), 0.15);
    const top = { g, w: o.w, h: H1 };
    const s = scales(top, [b.x[0] - span.p, b.x[0] + span.p], [b.x[1] - span.v, b.x[1] + span.v], [46, 12, 30, 30]);
    const [l, tp, W, Hh] = s.box;
    const xt = niceTicks(s.xr[0], s.xr[1], 4), yt = niceTicks(s.yr[0], s.yr[1], 3);
    g.save();
    g.font = '11px Inter, sans-serif'; g.strokeStyle = 'rgba(255,255,255,0.07)'; g.lineWidth = 1;
    xt.ticks.forEach((xx) => { const px = s.X(xx); g.beginPath(); g.moveTo(px, tp); g.lineTo(px, tp + Hh); g.stroke(); });
    yt.ticks.forEach((yy) => { const py = s.Y(yy); g.beginPath(); g.moveTo(l, py); g.lineTo(l + W, py); g.stroke(); });
    g.strokeStyle = CSS.faint; g.beginPath(); g.moveTo(l, tp); g.lineTo(l, tp + Hh); g.lineTo(l + W, tp + Hh); g.stroke();
    g.fillStyle = CSS.grey; g.textAlign = 'center'; g.textBaseline = 'top';
    xt.ticks.forEach((xx) => g.fillText(xx.toFixed(xt.digits), s.X(xx), tp + Hh + 4));
    g.textAlign = 'right'; g.textBaseline = 'middle';
    yt.ticks.forEach((yy) => g.fillText(yy.toFixed(yt.digits), l - 5, s.Y(yy)));
    g.fillStyle = CSS.white; g.font = '12px Inter, sans-serif';
    g.textAlign = 'right'; g.textBaseline = 'bottom'; g.fillText('position p (m)', l + W, tp + Hh - 3);
    g.textAlign = 'left'; g.textBaseline = 'top'; g.fillText('velocity v (m/s)', l + 5, tp + 3);
    g.font = '600 12.5px Inter, sans-serif'; g.textBaseline = 'middle'; g.fillStyle = CSS.white;
    g.fillText('belief of your filter', 8, 13);
    g.font = '12.5px Inter, sans-serif'; g.fillStyle = CSS.yellow; g.textAlign = 'right';
    g.fillText(`Kₚ ${fmt(Y.K[0], 2)} · Kᵥ ${fmt(Y.K[1], 2)}/s · ρ ${fmt(b.P[0][1] / (sp * sv), 2)}`, o.w - 8, 13);

    g.beginPath(); g.rect(l, tp, W, Hh); g.clip();
    // the reading
    if (lastZ !== null) {
      const a = lerp(0.95, 0.4, Math.min(1, readAge / 1.0)), px = s.X(lastZ);
      g.strokeStyle = `rgba(131,193,103,${a})`; g.lineWidth = 2.5;
      g.beginPath(); g.moveTo(px, tp); g.lineTo(px, tp + Hh); g.stroke();
      g.fillStyle = `rgba(131,193,103,${a})`; g.font = '11px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'bottom';
      g.fillText('reading', px + 4, tp + Hh - 16);
    }
    // samples that carry the belief
    const q = Q_TRUE * Y.qf, cT = carrier(tau, q);
    g.fillStyle = 'rgba(88,196,221,0.75)';
    for (let i = 0; i < N_SAMP; i++) {
      const post = carry(samp[i], sampW[i], cT);
      const pt = b.beta < 1 ? lerpVec(carry(sampPri[i], sampW[i], cT), post, b.beta) : post;
      g.beginPath(); g.arc(s.X(pt[0]), s.Y(pt[1]), 1.7, 0, 2 * Math.PI); g.fill();
    }
    // tilt axis through the mean, slope P_pv / P_pp
    const slope = b.P[0][1] / b.P[0][0];
    g.save(); g.setLineDash([2, 4]); g.strokeStyle = 'rgba(255,235,64,0.75)'; g.lineWidth = 1.3;
    g.beginPath(); g.moveTo(s.X(s.xr[0]), s.Y(b.x[1] + slope * (s.xr[0] - b.x[0]))); g.lineTo(s.X(s.xr[1]), s.Y(b.x[1] + slope * (s.xr[1] - b.x[0]))); g.stroke();
    g.restore();
    drawEllipse(o, s, b.x, b.P, 2, CSS.est, 'rgba(255,235,64,0.07)', 2.2);
    // the gain splits the innovation: hollow prediction -> K_p nu along p, K_v nu along v
    const sinceRead = readAge;
    if (nReads > 0 && sinceRead < 1.6) {
      const pri = predictTo(Y.xPri, Y.PPri, tau, q).x, post = predictTo(Y.xPost, Y.PPost, tau, q).x;
      const fade = sinceRead < 1.1 ? 1 : 1 - (sinceRead - 1.1) / 0.5, bb = b.beta;
      const cx = lerp(pri[0], post[0], bb), cy = lerp(pri[1], post[1], bb);
      g.globalAlpha = fade;
      g.strokeStyle = CSS.yellow; g.lineWidth = 2;
      g.beginPath(); g.moveTo(s.X(pri[0]), s.Y(pri[1])); g.lineTo(s.X(cx), s.Y(pri[1])); g.lineTo(s.X(cx), s.Y(cy)); g.stroke();
      g.fillStyle = '#000'; g.beginPath(); g.arc(s.X(pri[0]), s.Y(pri[1]), 4.5, 0, 2 * Math.PI); g.fill(); g.stroke();
      g.fillStyle = CSS.yellow; g.font = 'italic 12px Inter, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'bottom';
      g.fillText('Kₚν', s.X((pri[0] + cx) / 2), s.Y(pri[1]) - 5);
      g.textAlign = 'left'; g.textBaseline = 'middle';
      g.fillText('Kᵥν', s.X(cx) + 6, s.Y((pri[1] + cy) / 2));
      g.globalAlpha = 1;
    }
    // estimate and truth
    g.fillStyle = CSS.yellow; g.beginPath(); g.arc(s.X(b.x[0]), s.Y(b.x[1]), 3.6, 0, 2 * Math.PI); g.fill();
    g.fillStyle = CSS.white; g.strokeStyle = '#000'; g.lineWidth = 1.5;
    g.beginPath(); g.arc(s.X(x[0]), s.Y(x[1]), 5, 0, 2 * Math.PI); g.fill(); g.stroke();
    g.restore();

    // bottom: stationary theory against the assumed Q/R (drive without pushes)
    const bot = scales(o, [-3, 3], [0, 1], [46, 46, H1 + 30, 34]);
    const [bl, bt, bw, bh] = bot.box, RM = 0.3, Yr = (r) => bot.Y(Math.min(r, RM) / RM);
    g.save();
    g.strokeStyle = 'rgba(255,255,255,0.12)'; g.beginPath(); g.moveTo(0, H1 + 4); g.lineTo(o.w, H1 + 4); g.stroke();
    g.font = '600 12.5px Inter, sans-serif'; g.fillStyle = CSS.white; g.textAlign = 'left'; g.textBaseline = 'middle';
    g.fillText('theory, no pushes', 8, H1 + 17);
    g.font = '12px Inter, sans-serif'; g.textAlign = 'right';
    g.fillStyle = CSS.white; g.fillText('true RMS error of p (m) →', o.w - 8, H1 + 17);
    const wE = g.measureText('true RMS error of p (m) →').width;
    g.fillStyle = CSS.yellow; g.fillText('← steady Kₚ   ', o.w - 8 - wE, H1 + 17);
    g.strokeStyle = CSS.faint; g.lineWidth = 1;
    g.beginPath(); g.moveTo(bl, bt); g.lineTo(bl, bt + bh); g.lineTo(bl + bw, bt + bh); g.lineTo(bl + bw, bt); g.stroke();
    g.font = '11px Inter, sans-serif'; g.fillStyle = CSS.grey; g.textAlign = 'center'; g.textBaseline = 'top';
    ['0.001', '0.01', '0.1', '1', '10', '100', '1000'].forEach((lab, i) => {
      const px = bot.X(i - 3);
      g.strokeStyle = i === 3 ? 'rgba(255,255,255,0.18)' : 'rgba(255,255,255,0.06)';
      g.beginPath(); g.moveTo(px, bt); g.lineTo(px, bt + bh); g.stroke();
      g.fillText(lab, px, bt + bh + 4);
    });
    g.fillStyle = CSS.grey; g.textAlign = 'center';
    g.fillText('(Q/R assumed) ÷ (Q/R true)', bl + bw / 2, bt + bh + 18);
    g.textAlign = 'right'; g.textBaseline = 'middle'; g.fillStyle = CSS.yellow;
    [0, 0.5, 1].forEach((k) => g.fillText(k.toFixed(1), bl - 5, bot.Y(k)));
    g.textAlign = 'left'; g.fillStyle = CSS.white;
    [0, 0.1, 0.2, 0.3].forEach((r) => g.fillText(r.toFixed(1), bl + bw + 5, Yr(r)));
    // bare sensor
    g.save(); g.setLineDash([5, 4]); g.strokeStyle = 'rgba(131,193,103,0.85)'; g.lineWidth = 1.4;
    g.beginPath(); g.moveTo(bl, Yr(Math.sqrt(R_TRUE))); g.lineTo(bl + bw, Yr(Math.sqrt(R_TRUE))); g.stroke(); g.restore();
    g.fillStyle = CSS.green; g.textAlign = 'right'; g.textBaseline = 'bottom'; g.fillText('bare sensor 0.10 m', bl + bw - 5, Yr(Math.sqrt(R_TRUE)) - 2);
    g.beginPath(); g.rect(bl, bt, bw, bh); g.clip();
    const pathOf = (ys, map, css, w) => { g.strokeStyle = css; g.lineWidth = w; g.beginPath(); curve.l.forEach((ll, i) => { const px = bot.X(ll), py = map(ys[i]); if (i) g.lineTo(px, py); else g.moveTo(px, py); }); g.stroke(); };
    pathOf(curve.kp, (k) => bot.Y(k), CSS.yellow, 2.2);
    pathOf(curve.rms, Yr, CSS.white, 2.2);
    g.restore();
    // markers: the three lanes
    const marks = others ? [LANES[0], LANES[2], LANES[1]] : [LANES[1]];
    marks.forEach((L) => {
      const lx = Math.log10(L.qf / L.rf), px = bot.X(Math.max(-3, Math.min(3, lx))), yk = bot.Y(L.steady.Kp), yr = Yr(L.steady.rms);
      const mine = L.key === 'yours';
      g.strokeStyle = mine ? 'rgba(255,235,64,0.55)' : 'rgba(255,255,255,0.22)'; g.lineWidth = 1;
      g.beginPath(); g.moveTo(px, Math.min(yk, yr)); g.lineTo(px, Math.max(yk, yr)); g.stroke();
      g.fillStyle = CSS.yellow; g.beginPath(); g.arc(px, yk, mine ? 5 : 3.5, 0, 2 * Math.PI); g.fill();
      g.fillStyle = CSS.white; g.beginPath(); g.arc(px, yr, mine ? 5 : 3.5, 0, 2 * Math.PI); g.fill();
      if (mine) { g.strokeStyle = '#000'; g.lineWidth = 1.2; g.stroke(); }
      g.font = mine ? '600 11.5px Inter, sans-serif' : '11px Inter, sans-serif';
      g.fillStyle = mine ? CSS.yellow : CSS.grey; g.textAlign = 'center'; g.textBaseline = 'bottom';
      if (!mine && Math.abs(Math.log10(L.qf / L.rf) - Math.log10(Y.qf / Y.rf)) < 0.4) return;
      const name = mine ? 'you' : L.key;
      g.fillText(name, px, Math.min(yk, yr) - 7);
    });
    g.restore();
  }

  panel._state = () => {
    const bel = lastBel || belief(LANES[1], 0);
    return {
      t: +t.toFixed(3), frames, rate, wraps, readings: nReads, lastZ,
      truth: { p: x[0], v: x[1] }, accel, keysOn: panel.classList.contains('keys-on'),
      held: [...kb.held], others,
      lanes: LANES.map((L) => ({
        lane: L.key, qf: L.qf, rf: L.rf, ratio: L.qf / L.rf,
        K: L.K, Ksteady: [L.steady.Kp, L.steady.Kv], sigmaPclaimed: L.steady.sigmaClaimed, rmsTheory: L.steady.rms,
        est: L.kf.x, sigma: [Math.sqrt(L.kf.P[0][0]), Math.sqrt(L.kf.P[1][1])], corr: L.kf.P[0][1] / Math.sqrt(L.kf.P[0][0] * L.kf.P[1][1]),
        err: [L.kf.x[0] - x[0], L.kf.x[1] - x[1]],
        rmsLast: L.err2.length ? Math.sqrt(L.err2.reduce((a, c) => a + c, 0) / L.err2.length) : null, nErr: L.err2.length,
      })),
      plot: { mean: bel.x, P: bel.P, tiltSlope: bel.P[0][1] / bel.P[0][0], nSamples: samp.length,
        sampleMean: [samp.reduce((a, s) => a + s[0], 0) / samp.length, samp.reduce((a, s) => a + s[1], 0) / samp.length] },
      track: { lines: track.lines, ticks: track.ticks, lineX: track.lineX, tickX: track.tickX },
      camera: { followX, target: [v.controls.target.x, v.controls.target.y, v.controls.target.z] },
    };
  };
}
