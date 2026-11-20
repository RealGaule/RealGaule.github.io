// Demo 3.2: optimization on manifolds (clip s32_manifold, render/s32_manifold.py).
// The camera on the tool must look at the target. Act A: the viewing direction x is a point on the unit sphere,
// cost f(x) = -t.x. Act B: the whole tool orientation X is a rotation, cost f(X) = -tr(R_t^T X). Three steps:
//   plain              x <- x - alpha g                          (S.sphereStep 'euclidean',  S.manifoldStep 'euclidean')
//   tangent only       x <- x - alpha (I - x x^T) g              ('project_only', projection 'sym')
//   tangent + retract  x <- normalise(x - alpha (I - x x^T) g)   ('riemannian'; rotations: QR retraction)
// Same start pose, target rule (distance 0.25 m, roll 60 degrees), step size and display rules as the clip. The arm
// holds the tool point still (inverse kinematics, sim6) and turns the tool to x / |x| (act A) or to the rotation
// closest to X (act B); the gripper is drawn with the matrix X itself, so a non-rotation shows as a sheared gripper.
// All numerics are sim6.js; nothing is random.
import { S, THREE, makeArmViewer, marker, Line3D, COL, CSS, world, val, onInput, onClick, readout, seg, plot2d,
  fmt, deg, rad } from '../common.js';

const Q_START = [Math.PI, -1.9, 1.7, -Math.PI / 2 + 0.2 - 0.15, -Math.PI / 2 + 0.45, Math.PI / 2];
const T_RAW = [1.0, 0.35, 0.15];         // direction tool point -> target of the clip (base frame)
const ROLL = rad(60);                    // target orientation: looking at the target, rolled 60 degrees
const N = 10;                            // iterations of every run (as in the clip)
const R3 = 0.25;                         // act A: radius of the drawn sphere = distance of the target (m)
const RB = 0.11;                         // act B: drawn length of a unit column (m)
const U_T = 0.55;                        // share of a retraction step shown as the move along the tangent
const RES_FLOOR = 1e-17;                 // exact zeros are drawn here in the residual plot
const DUR = 0.85, DUR_RUN = 0.55, DUR_BACK = 0.6;   // seconds per displayed step / rewind
const SPH = 0x6b94cc, SPH_RGB = '107, 148, 204';    // the sphere (clip colour C_SPH)
const METHODS = ['plain', 'tangent', 'retract'];
const MODE = { plain: 'euclidean', tangent: 'project_only', retract: 'riemannian' };
const LABEL = { plain: 'plain gradient step', tangent: 'tangent step only', retract: 'tangent step + retraction' };
const MCSS = { plain: CSS.red, tangent: CSS.grey, retract: CSS.white };
const M3 = { plain: COL.red, tangent: COL.grey, retract: COL.white };

// Numbers of the clip's case (Lab6_Optimization/build/out/s32_manifold_facts.json), compared in _state().vsClip.
const CLIP = {
  theta_A_deg: 81.76596867706922, theta_B_deg: 98.19759595262637,
  sphere_plain_norm_3: 2.180729341468262, sphere_plain_residual_10: 37.71860153580199,
  sphere_tangent_steps_to_1deg: 6, sphere_tangent_max_residual: 3.3306690738754696e-16,
  sphere_tangent_only_norm_1_3_10: [1.1630202156036873, 1.0177076964331757, 1.0000000812107457],
  so3_plain_residual_3: 7.852084178569007, so3_plain_residual_10: 68.23096322189889,
  so3_tangent_steps_to_1deg: 7, so3_tangent_max_residual: 1.0174308724868857e-15,
  so3_tangent_only_residual_1_3_10: [0.4987659416959837, 0.20571220762365228, 2.3850585413786308e-08],
};

// ------------------------------------------------------------------------------------------------ small helpers
const unit = (v) => S.scale(v, 1 / S.norm(v));
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const ease = (x) => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
const lerp = (a, b, e) => a.map((v, i) => v + e * (b[i] - v));
const lerpM = (A, B, e) => A.map((r, i) => lerp(r, B[i], e));
const polar = (X) => S.retractPolar(X, S.zerosM(3, 3));            // closest rotation (det > 0)
const maxOf = (a) => a.reduce((m, x) => Math.max(m, x), 0);
const stepsTo1 = (ang) => { const i = ang.findIndex((a) => a < rad(1)); return i < 0 ? null : i; };
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
/** Upper bound with one digit, rounded up: 3.3e-16 -> '4×10⁻¹⁶'. */
const upBound = (r) => {
  if (!(r > 0)) return '0';
  let e = Math.floor(Math.log10(r)), m = Math.ceil(r / 10 ** e - 1e-9);
  if (m >= 10) { m = 1; e += 1; }
  return `${m}×10${String(e).split('').map((c) => SUP[c]).join('')}`;
};
const fr = (r) => (r === 0 ? '0' : r < 1e-3 ? r.toExponential(1) : r < 10 ? r.toFixed(3) : r.toFixed(1));

/** Smallest rotation taking the direction of a to the direction of b (null if opposite). */
function rotBetween(a, b) {
  const ua = unit(a), ub = unit(b), w = S.cross(ua, ub), s = S.norm(w), c = S.dot(ua, ub);
  if (s < 1e-14) return c > 0 ? S.eye(3) : null;
  return S.so3Exp(S.scale(w, Math.atan2(s, c) / s));
}
/** Great-circle point between unit vectors a and b. */
function slerp(a, b, u) {
  const th = S.angleBetween(a, b);
  if (th < 1e-9) return lerp(a, b, u);
  const sa = Math.sin((1 - u) * th) / Math.sin(th), sb = Math.sin(u * th) / Math.sin(th);
  return a.map((v, i) => sa * v + sb * b[i]);
}
function dirFromAngles(azDeg, elDeg) {
  const a = rad(azDeg), e = rad(elDeg);
  return [Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e)];
}

// ------------------------------------------------------------------------------------------------ the problem
/** Every run of both acts for target direction t and step size alpha (S.sphereGD, S.manifoldGD). */
function problem(t, alpha, R0) {
  const x0 = S.column(R0, 2);
  const RA = S.matmul(rotBetween(x0, t), R0);
  const Rt = polar(S.matmul(S.so3Exp(S.scale(t, ROLL)), RA));
  const g = S.scale(t, -1), G = S.scaleM(Rt, -1);                 // ordinary gradients (constant)
  const A = {}, B = {};
  for (const m of METHODS) {
    const xs = S.sphereGD(() => g, x0, alpha, N, MODE[m]);
    A[m] = { xs, V: xs.slice(0, N).map((x) => S.scale(S.sphereProject(x, g), -alpha)),
      res: xs.map((x) => S.orthResidual(x)), ang: xs.map((x) => S.angleBetween(x, t)), nrm: xs.map((x) => S.norm(x)) };
    const Xs = S.manifoldGD(() => G, R0, alpha, N, { retraction: 'qr', mode: MODE[m], projection: 'sym' });
    const dets = Xs.map((X) => S.det(X));
    B[m] = { Xs, det: dets, res: Xs.map((X) => S.orthResidual(X)), sv: Xs.map((X) => S.singularValues(X)),
      ang: Xs.map((X, k) => (dets[k] > 0 ? S.rotationAngle(polar(X), Rt) : NaN)) };
  }
  B.retract.logs = B.retract.Xs.slice(0, N).map((X, k) => S.so3Log(S.matmul(S.transpose(X), B.retract.Xs[k + 1])));
  return { t, alpha, x0, Rt, g, thetaA: S.angleBetween(x0, t), thetaB: S.rotationAngle(R0, Rt), A, B };
}

// what is on screen between two iterates (display only; the iterates themselves are exact)
function dispA(P, m, s) {
  const R = P.A[m]; s = clamp(s, 0, N);
  const k = Math.min(Math.floor(s), N - 1), u = s - k;
  if (m !== 'retract') return lerp(R.xs[k], R.xs[k + 1], ease(u));
  if (u <= U_T) return S.axpy(R.xs[k], ease(u / U_T), R.V[k]);      // along the tangent ...
  const w = S.add(R.xs[k], R.V[k]), e = ease((u - U_T) / (1 - U_T));
  return S.scale(w, (1 - e) + e / S.norm(w));                         // ... then radially back (the retraction)
}
function dispB(P, m, s) {
  const R = P.B[m]; s = clamp(s, 0, N);
  const k = Math.min(Math.floor(s), N - 1), u = s - k;
  if (m !== 'retract') return lerpM(R.Xs[k], R.Xs[k + 1], ease(u));  // the plain step IS a straight line in R^{3x3}
  return S.matmul(R.Xs[k], S.so3Exp(S.scale(R.logs[k], ease(u))));   // geodesic: every displayed matrix a rotation
}
function pathA(P, m, upto) {
  const R = P.A[m], K = Math.min(N, Math.floor(upto + 1e-9)), pts = [R.xs[0]];
  for (let k = 0; k < K; k++) {
    if (m === 'retract') for (let j = 1; j <= 12; j++) pts.push(slerp(R.xs[k], R.xs[k + 1], j / 12));
    else pts.push(R.xs[k + 1]);
  }
  if (upto - K > 1e-6 && K < N) pts.push(dispA(P, m, upto));
  return pts;
}
function pathB(P, m, upto, i) {
  const R = P.B[m], K = Math.min(N, Math.floor(upto + 1e-9)), pts = [S.column(R.Xs[0], i)];
  for (let k = 0; k < K; k++) {
    if (m === 'retract') for (let j = 1; j <= 8; j++) pts.push(S.column(dispB(P, m, k + j / 8 - 1e-12), i));
    else pts.push(S.column(R.Xs[k + 1], i));
  }
  if (upto - K > 1e-6 && K < N) pts.push(S.column(dispB(P, m, upto), i));
  return pts;
}

// ------------------------------------------------------------------------------------------------ 3D objects
class Arrow3D {
  constructor(scene, color, { r = 0.005, head = 0.034, headR = 0.013, opacity = 1 } = {}) {
    const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.45, roughness: 0.45,
      transparent: opacity < 1, opacity, depthWrite: opacity >= 1 });
    this.head = head;
    this.shaft = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 1, 14), mat);
    this.cone = new THREE.Mesh(new THREE.ConeGeometry(headR, head, 18), mat);
    for (const o of [this.shaft, this.cone]) { o.castShadow = opacity >= 1; scene.add(o); }
    this._y = new THREE.Vector3(0, 1, 0); this._d = new THREE.Vector3();
  }
  set(a, b) {
    const d = this._d.set(b[0] - a[0], b[1] - a[1], b[2] - a[2]), len = d.length();
    if (len < 1e-5) { this.visible = false; return; }
    this.visible = true;
    d.multiplyScalar(1 / len);
    const h = Math.min(this.head, 0.45 * len), ls = len - h;
    this.shaft.quaternion.setFromUnitVectors(this._y, d); this.cone.quaternion.copy(this.shaft.quaternion);
    this.shaft.scale.set(1, ls, 1); this.shaft.position.set(a[0] + d.x * ls / 2, a[1] + d.y * ls / 2, a[2] + d.z * ls / 2);
    this.cone.scale.set(h / this.head, h / this.head, h / this.head);
    this.cone.position.set(a[0] + d.x * (ls + h / 2), a[1] + d.y * (ls + h / 2), a[2] + d.z * (ls + h / 2));
  }
  set visible(on) { this.shaft.visible = on; this.cone.visible = on; }
}

// The gripper with its camera, as in stage6._ur5e_spec (tool frame: z out of the flange, fingers apart along x,
// camera body on +y). Sizes: box half extents; cylinder (radius, half height) along z.
const DARK = 0x212429, LIGHT = 0xcccfd6, LENS = 0x58c4dd;
const GRIPPER = [['cyl', [0.044, 0.009], [0, 0, -0.011], DARK], ['box', [0.046, 0.013, 0.004], [0, 0, 0.002], DARK],
  ['box', [0.006, 0.012, 0.030], [0.040, 0, 0.034], LIGHT], ['box', [0.006, 0.012, 0.030], [-0.040, 0, 0.034], LIGHT],
  ['box', [0.019, 0.014, 0.017], [0, 0.058, -0.004], DARK], ['cyl', [0.010, 0.006], [0, 0.058, 0.018], LENS]];

function makeGripper(scene, { color = null, opacity = 1 } = {}) {
  const grp = new THREE.Group();
  grp.matrixAutoUpdate = false;
  for (const [kind, size, pos, col] of GRIPPER) {
    let geo;
    if (kind === 'box') geo = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
    else { geo = new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 32); geo.rotateX(Math.PI / 2); }
    const c = color === null ? col : color;
    const mat = new THREE.MeshStandardMaterial({ color: c, roughness: 0.45, metalness: 0.15,
      emissive: color === null ? 0x000000 : c, emissiveIntensity: color === null ? 0 : 0.35,
      transparent: opacity < 1, opacity, depthWrite: opacity >= 1 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(...pos);
    mesh.castShadow = opacity >= 1;
    grp.add(mesh);
  }
  scene.add(grp);
  grp.setX = (X, p) => {   // any 3 x 3 matrix: a rotation gives the rigid gripper, anything else a sheared one
    grp.matrix.set(X[0][0], X[0][1], X[0][2], p[0], X[1][0], X[1][1], X[1][2], p[1], X[2][0], X[2][1], X[2][2], p[2], 0, 0, 0, 1);
    grp.matrixWorldNeedsUpdate = true;
  };
  return grp;
}

function circlePts(c, a, b, r, n = 96) {
  return Array.from({ length: n + 1 }, (_, k) => {
    const th = 2 * Math.PI * k / n;
    return [0, 1, 2].map((j) => c[j] + r * (Math.cos(th) * a[j] + Math.sin(th) * b[j]));
  });
}
function line3(scene, color, max, opacity) {
  const l = new Line3D(scene, color, max, { opacity: 0.999 });
  l.line.material.opacity = opacity;
  l.line.renderOrder = 3;
  return l;
}

// ------------------------------------------------------------------------------------------------ the demo
export async function init(panel) {
  const fk0 = S.fk(Q_START), p0 = fk0.p, R0 = polar(fk0.R), x0 = S.column(R0, 2);
  const tool = world(p0);
  const T_DEFAULT = unit(T_RAW);

  // cameras of the two acts (the clip's azimuth and elevation; the tool point a little left of the centre)
  const camFor = (act) => {
    const az = rad(-102), right = [Math.sin(az), -Math.cos(az), 0];
    const sh = act === 'A' ? 0.08 : 0.07, dz = act === 'A' ? -0.1 : -0.02;
    return { lookat: [tool[0] + sh * right[0], tool[1] + sh * right[1], tool[2] + dz],
      distance: act === 'A' ? 1.85 : 1.05, azimuth: -102, elevation: -8 };
  };
  const v = await makeArmViewer(panel, { camera: camFor('A'), q: Q_START });

  // the model's own gripper is replaced by one drawn with the matrix X
  let hiddenToolGeoms = 0;
  try {
    const bid = v.id('body', 'tool');
    for (let gi = 0; gi < v.model.ngeom; gi++) {
      if (v.model.geom_bodyid[gi] === bid && v.meshes[gi]) { v.meshes[gi].visible = false; hiddenToolGeoms++; }
    }
  } catch (e) { hiddenToolGeoms = -1; }
  const grip = makeGripper(v.scene);
  const gripT = makeGripper(v.scene, { color: COL.green, opacity: 0.32 });

  // act A objects
  const bigSphere = new THREE.Mesh(new THREE.SphereGeometry(R3, 48, 32), new THREE.MeshStandardMaterial({
    color: SPH, roughness: 0.3, transparent: true, opacity: 0.10, depthWrite: false, side: THREE.DoubleSide }));
  bigSphere.position.set(...tool); bigSphere.renderOrder = 1; v.scene.add(bigSphere);
  const rings = [[[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 0, 1]]].map(([a, b]) => {
    const l = line3(v.scene, SPH, 100, 0.45); l.set(circlePts(tool, a, b, R3)); return l;
  });
  const target = marker(v.scene, COL.green, 0.02);
  const halo = marker(v.scene, COL.green, 0.05, 0.22);
  const toTarget = line3(v.scene, COL.green, 2, 0.35);
  const xArrow = new Arrow3D(v.scene, COL.white, { r: 0.0055 });
  const xTip = marker(v.scene, COL.white, 0.011);
  const pathsA = Object.fromEntries(METHODS.map((m) => [m, line3(v.scene, M3[m], 160, 1)]));
  const anaG = new Arrow3D(v.scene, COL.red, { r: 0.003, head: 0.024, headR: 0.009 });
  const anaV = new Arrow3D(v.scene, COL.white, { r: 0.003, head: 0.024, headR: 0.009 });
  const anaR = line3(v.scene, COL.grey, 2, 0.9);
  // act B objects
  const smallSphere = new THREE.Mesh(new THREE.SphereGeometry(RB, 40, 28), new THREE.MeshStandardMaterial({
    color: SPH, roughness: 0.3, transparent: true, opacity: 0.07, depthWrite: false, side: THREE.DoubleSide }));
  smallSphere.position.set(...tool); smallSphere.renderOrder = 1; v.scene.add(smallSphere);
  const ringB = line3(v.scene, SPH, 100, 0.4); ringB.set(circlePts(tool, [1, 0, 0], [0, 1, 0], RB));
  const colArrows = [0, 1, 2].map(() => new Arrow3D(v.scene, COL.white, { r: 0.003, head: 0.02, headR: 0.008 }));
  const colTargets = [0, 1, 2].map(() => marker(v.scene, COL.green, 0.008));
  const pathsB = [0, 1, 2].map(() => line3(v.scene, COL.white, 120, 1));
  const groupA = [bigSphere, target, halo, xTip], groupB = [smallSphere, gripT, ...colTargets];

  // ---------------------------------------------------------------------------------------------- state
  let t = T_DEFAULT, P = problem(t, val(panel, 'alpha'), R0);
  const st = { act: 'A', method: 'plain', pos: 0, goal: 0, dur: DUR, back: null,
    reached: { plain: 0, tangent: 0, retract: 0 } };
  let dirty = true;
  const plotCanvas = panel.querySelector('canvas.plot'), P2 = plot2d(plotCanvas);
  new ResizeObserver(() => { dirty = true; }).observe(plotCanvas);

  const current = () => {
    if (st.back) { const e = ease(st.back.u); return st.act === 'A' ? lerp(st.back.from, P.x0, e) : lerpM(st.back.from, R0, e); }
    return st.act === 'A' ? dispA(P, st.method, st.pos) : dispB(P, st.method, st.pos);
  };
  const kNow = () => Math.min(N, Math.floor(st.pos + 1e-9));
  const animating = () => !!st.back || st.pos < st.goal;
  function rewind() {
    if (st.pos > 0 || st.back) st.back = { from: current(), u: 0 };
    st.pos = 0; st.goal = 0; dirty = true;
  }
  function recompute() {
    P = problem(t, val(panel, 'alpha'), R0);
    st.pos = 0; st.goal = 0; st.back = null;
    for (const m of METHODS) st.reached[m] = 0;
    target.position.set(...S.axpy(tool, R3, t)); halo.position.copy(target.position);
    toTarget.set([tool, S.axpy(tool, R3, t)]);
    dirty = true;
  }
  recompute();

  // ---------------------------------------------------------------------------------------------- arm (IK)
  let q = Q_START.slice(), Rcur = R0;
  const ik = { posErr: 0, rotErr: 0, sigmaMin: Infinity, solves: 0 };
  function trackFrame(Rgoal) {
    if (!Rgoal) return;
    const ang = S.rotationAngle(Rcur, Rgoal);
    if (!(ang > 1e-9)) return;
    const n = Math.max(1, Math.ceil(ang / 0.15));            // small pieces keep the arm on one branch
    const L = S.so3Log(S.matmul(S.transpose(Rcur), Rgoal));
    for (let i = 1; i <= n; i++) {
      const Ri = i === n ? Rgoal : S.matmul(Rcur, S.so3Exp(S.scale(L, i / n)));
      const r = S.ikDampedLeastSquares(q, p0, Ri, { damping: 1e-4, rotWeight: 1.0, tol: 1e-10, maxIter: 60 });
      q = r.q[r.q.length - 1];
      ik.posErr = r.errPos[r.errPos.length - 1]; ik.rotErr = r.errRot[r.errRot.length - 1];
      ik.sigmaMin = Math.min(ik.sigmaMin, r.sigmaMin[r.sigmaMin.length - 1]); ik.solves++;
    }
    Rcur = S.fk(q).R;
    v.setArm(q); v.forward();
  }
  const armFrame = (d) => {
    if (st.act === 'A') { if (S.norm(d) < 1e-9) return null; const Rb = rotBetween(x0, d); return Rb && S.matmul(Rb, R0); }
    return S.det(d) > 1e-9 ? polar(d) : null;
  };

  // ---------------------------------------------------------------------------------------------- 3D update
  const tip = (x, r = R3) => S.axpy(tool, r, x);
  function update3D(d) {
    const A = st.act === 'A';
    for (const o of groupA) o.visible = A;
    for (const o of groupB) o.visible = !A;
    for (const l of [...rings, toTarget, anaR]) l.visible = A;
    ringB.visible = !A;
    xArrow.visible = A;
    for (const m of METHODS) pathsA[m].visible = A;
    colArrows.forEach((a) => { a.visible = !A; });
    pathsB.forEach((l) => { l.visible = !A; });
    if (A) {
      grip.setX(Rcur, tool);
      xArrow.set(tool, tip(d)); xTip.position.set(...tip(d));
      for (const m of METHODS) {
        const active = m === st.method, upto = active ? (st.back ? st.reached[m] : st.pos) : st.reached[m];
        pathsA[m].set(pathA(P, m, upto).map((x) => tip(x)));
        pathsA[m].line.material.opacity = active && !st.back ? 1 : 0.4;
      }
      // anatomy of one step: -alpha g (red), its tangent part (white), the pull back (grey)
      const ka = st.back ? -1 : (st.pos < st.goal || st.pos % 1 > 1e-9 ? Math.floor(st.pos) : kNow() - 1);
      const showA = ka >= 0 && ka < N;
      anaG.visible = showA; anaV.visible = showA && st.method !== 'plain'; anaR.visible = showA && st.method === 'retract';
      if (showA) {
        const R = P.A[st.method], xk = R.xs[ka];
        anaG.set(tip(xk), tip(S.axpy(xk, -P.alpha, P.g)));
        if (st.method !== 'plain') anaV.set(tip(xk), tip(S.add(xk, R.V[ka])));
        if (st.method === 'retract') anaR.set([tip(S.add(xk, R.V[ka])), tip(R.xs[ka + 1])]);
      }
    } else {
      anaG.visible = false; anaV.visible = false; anaR.visible = false;
      grip.setX(d, tool);
      gripT.setX(P.Rt, tool);
      for (let i = 0; i < 3; i++) {
        colArrows[i].set(tool, tip(S.column(d, i), RB));
        colTargets[i].position.set(...tip(S.column(P.Rt, i), RB));
        pathsB[i].set(pathB(P, st.method, st.back ? 0 : st.pos, i).map((c) => tip(c, RB)));
        pathsB[i].line.material.color.setHex(M3[st.method]);
      }
    }
  }

  // ---------------------------------------------------------------------------------------------- 2D panel
  let lastBasis = '';
  const basis = () => {
    const e = v.camera.matrixWorld.elements;
    return { r: [e[0], e[1], e[2]], u: [e[4], e[5], e[6]], b: [e[8], e[9], e[10]] };
  };
  function drawPanel(d) {
    P2.clear();
    const g = P2.g, W = P2.w, H = P2.h, topH = Math.round(H * 0.6), Bs = basis();
    const cx = W / 2, cy = 34 + (topH - 34) / 2, Rp = Math.max(30, Math.min(W * 0.3, (topH - 50) * 0.43));
    const pr = (x) => [cx + Rp * S.dot(x, Bs.r), cy - Rp * S.dot(x, Bs.u)];
    const behind = (x) => S.dot(x, Bs.b) < 0 && Math.hypot(S.dot(x, Bs.r), S.dot(x, Bs.u)) < 1;   // hidden by the sphere
    const poly = (pts, color, { w = 2, a = 1, dash = null } = {}) => {
      g.save(); g.strokeStyle = color; g.lineWidth = w; g.lineJoin = 'round'; g.lineCap = 'round';
      if (dash) g.setLineDash(dash);
      for (let i = 0; i + 1 < pts.length; i++) {
        const mid = lerp(pts[i], pts[i + 1], 0.5);
        g.globalAlpha = a * (behind(mid) ? 0.3 : 1);
        const p = pr(pts[i]), q2 = pr(pts[i + 1]);
        g.beginPath(); g.moveTo(...p); g.lineTo(...q2); g.stroke();
      }
      g.restore();
    };
    const arrow = (a3, b3, color, { w = 2.2, head = 9, a = 1 } = {}) => {
      const [x1, y1] = pr(a3), [x2, y2] = pr(b3), L = Math.hypot(x2 - x1, y2 - y1);
      if (L < 1) return;
      const ux = (x2 - x1) / L, uy = (y2 - y1) / L, h = Math.min(head, 0.5 * L);
      g.save(); g.globalAlpha = a; g.strokeStyle = color; g.fillStyle = color; g.lineWidth = w;
      g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2 - ux * h * 0.8, y2 - uy * h * 0.8); g.stroke();
      g.beginPath(); g.moveTo(x2, y2); g.lineTo(x2 - ux * h - uy * h * 0.45, y2 - uy * h + ux * h * 0.45);
      g.lineTo(x2 - ux * h + uy * h * 0.45, y2 - uy * h - ux * h * 0.45); g.closePath(); g.fill(); g.restore();
    };
    const dot = (x3, color, r = 4, a = 1, ring = false) => {
      const [x, y] = pr(x3); g.save(); g.globalAlpha = a;
      if (ring) { g.strokeStyle = color; g.lineWidth = 1.8; g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.stroke(); }
      else { g.fillStyle = color; g.strokeStyle = '#000'; g.lineWidth = 1.2; g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill(); g.stroke(); }
      g.restore();
    };
    const text = (s, x, y, color, { size = 12, align = 'left', base = 'middle', bold = false } = {}) => {
      g.save(); g.fillStyle = color; g.font = `${bold ? '600 ' : ''}${size}px Inter, sans-serif`;
      g.textAlign = align; g.textBaseline = base; g.fillText(s, x, y); g.restore();
    };
    const boxes = [];
    if (!st.back && kNow() > 0) boxes.push([0, topH - 22, W, topH - 2]);
    const label3 = (x3, s, color, off = 13) => {
      const [x, y] = pr(x3), dx = x - cx, dy = y - cy, L = Math.hypot(dx, dy) || 1;
      const px = clamp(x + off * dx / L, 14, W - 14), py0 = clamp(y + off * dy / L, 50, topH - 30);
      g.font = '12px Inter, sans-serif'; const hw = g.measureText(s).width / 2 + 3;
      for (const ddy of [0, 14, -14, 28, -28]) {
        const py = py0 + ddy; if (py < 50 || py > topH - 30) continue; const q = [px - hw, py - 8, px + hw, py + 8];
        if (boxes.some(b => !(b[0] > q[2] || b[2] < q[0] || b[1] > q[3] || b[3] < q[1]))) continue;
        boxes.push(q); text(s, px, py, color, { align: 'center' }); return;
      }
    };

    // --- the sphere, seen from the 3D camera
    g.save(); g.beginPath(); g.rect(0, 42, W, topH - 42); g.clip();
    const gr = g.createRadialGradient(cx - 0.3 * Rp, cy - 0.35 * Rp, 0.1 * Rp, cx, cy, Rp);
    gr.addColorStop(0, 'rgba(32, 52, 84, 1)'); gr.addColorStop(1, 'rgba(10, 18, 32, 1)');
    g.fillStyle = gr; g.beginPath(); g.arc(cx, cy, Rp, 0, 2 * Math.PI); g.fill();
    const grid = [circlePts([0, 0, 0], [1, 0, 0], [0, 1, 0], 1, 72)];
    for (const az of [0, 60, 120]) grid.push(circlePts([0, 0, 0], dirFromAngles(az, 0), [0, 0, 1], 1, 72));
    for (const c of grid) poly(c, `rgba(${SPH_RGB}, 0.55)`, { w: 1 , a: 0.8 });
    g.strokeStyle = `rgba(${SPH_RGB}, 0.95)`; g.lineWidth = 1.5; g.beginPath(); g.arc(cx, cy, Rp, 0, 2 * Math.PI); g.stroke();

    const k = kNow(), m = st.method, A = st.act === 'A';
    if (A) {
      poly([[0, 0, 0], P.t], CSS.green, { w: 1.2, a: 0.6, dash: [3, 4] });
      const [tx, ty] = pr(P.t);
      g.save(); g.fillStyle = 'rgba(131, 193, 103, 0.25)'; g.beginPath(); g.arc(tx, ty, 11, 0, 2 * Math.PI); g.fill(); g.restore();
      dot(P.t, CSS.green, 5, behind(P.t) ? 0.5 : 1); label3(P.t, 't', CSS.green); label3(d, 'x', CSS.white, 14);
      for (const mm of METHODS) {
        const active = mm === m && !st.back;
        if (!active && st.reached[mm] === 0) continue;
        const pts = pathA(P, mm, active ? st.pos : st.reached[mm]);
        poly(pts, MCSS[mm], { w: active ? 2.2 : 1.6, a: active ? 1 : 0.4, dash: mm === 'tangent' ? [5, 3] : null });
        const K = Math.min(N, Math.floor((active ? st.pos : st.reached[mm]) + 1e-9));
        for (let j = 1; j <= K; j++) dot(P.A[mm].xs[j], MCSS[mm], 2.6, active ? 0.9 : 0.4);
      }
      const ka = st.back ? -1 : (st.pos < st.goal || st.pos % 1 > 1e-9 ? Math.floor(st.pos) : k - 1);
      if (ka >= 0 && ka < N) {
        const R = P.A[m], xk = R.xs[ka], gStep = S.axpy(xk, -P.alpha, P.g);
        arrow(xk, gStep, CSS.red, { w: 1.6, head: 7, a: 0.9 }); label3(lerp(xk, gStep, 0.5), '−αg', CSS.red, 16);
        if (m !== 'plain') {
          const w = S.add(xk, R.V[ka]);
          arrow(xk, w, CSS.white, { w: 1.6, head: 7, a: 0.9 });
          { const [a1, b1] = pr(xk), [a2, b2] = pr(w); if ((ka === k - 1 || st.pos < st.goal) && Math.hypot(a2 - a1, b2 - b1) > 24) label3(lerp(xk, w, 0.5), 'project', CSS.white, -16); }
          if (m === 'retract') { poly([w, R.xs[ka + 1]], CSS.grey, { w: 1.4, dash: [3, 3] }); { const [a1, b1] = pr(w), [a2, b2] = pr(R.xs[ka + 1]); if (Math.hypot(a2 - a1, b2 - b1) > 12) label3(w, 'retract', CSS.grey, 30); } }
        }
      }
      arrow([0, 0, 0], d, CSS.white, { w: 2.6, head: 11 });
      dot(d, CSS.white, 3.5);
    } else {
      for (let i = 0; i < 3; i++) {
        const c = S.column(P.Rt, i);
        poly([[0, 0, 0], c], CSS.green, { w: 1, a: 0.45, dash: [3, 4] });
        dot(c, CSS.green, 6, behind(c) ? 0.5 : 1, true);
      }
      if (!st.back) for (let i = 0; i < 3; i++) poly(pathB(P, m, st.pos, i), MCSS[m], { w: 2, a: 0.9, dash: m === 'tangent' ? [5, 3] : null });
      for (let i = 0; i < 3; i++) {
        const c = S.column(d, i);
        arrow([0, 0, 0], c, CSS.white, { w: 2.4, head: 10 }); dot(c, CSS.white, 3);
        label3(c, ['x₁', 'x₂', 'x₃'][i], CSS.white, 14);
      }
    }
    g.restore();
    text(A ? 'pointing direction x,  xᵀx = 1' : 'orientation X = [x₁ x₂ x₃],  XᵀX = I', 12, 14, CSS.mute);
    text(`${LABEL[m]}`, 12, 32, MCSS[m], { bold: true });
    g.font = '600 12px Inter, sans-serif';
    text(`step ${k}`, 12 + g.measureText(LABEL[m]).width + 12, 32, CSS.mute);
    const res = (A ? P.A : P.B)[m].res[k];
    if (!st.back && k > 0) {
      const ok = res < 1e-12;
      text(ok ? (A ? 'every iterate is a unit vector' : 'every iterate is a rotation')
        : (A ? `x is off the sphere:  |xᵀx − 1| = ${fr(res)}` : `X is no longer a rotation:  ‖XᵀX − I‖ = ${fr(res)}`), 12, topH - 12, ok ? CSS.white : MCSS[m]);
    }

    // --- constraint residual against the iteration (log scale; exact zeros on the bottom line)
    const L = 54, Rr = 16, T = topH + 30, Bm = 30, pw = W - L - Rr, ph = H - T - Bm;
    const X = (kk) => L + kk / N * pw, Y = (r) => T + clamp((3 - Math.log10(Math.max(r, RES_FLOOR))) / 20, 0, 1) * ph;
    text(A ? 'constraint residual  |xᵀx − 1|' : 'constraint residual  ‖XᵀX − I‖', L, topH + 14, CSS.mute);
    g.save(); g.strokeStyle = 'rgba(97, 97, 107, 1)'; g.lineWidth = 1; g.strokeRect(L + 0.5, T + 0.5, pw - 1, ph - 1);
    for (const [e, s] of [[0, '1'], [-4, '10⁻⁴'], [-8, '10⁻⁸'], [-12, '10⁻¹²'], [-16, '10⁻¹⁶'], [-17, '0']]) {
      const y = Y(10 ** e);
      g.strokeStyle = 'rgba(74, 76, 85, 0.6)'; g.beginPath(); g.moveTo(L, y); g.lineTo(L + pw, y); g.stroke();
      if (e !== -16) text(s, L - 6, y, CSS.mute, { size: 10, align: 'right' });
    }
    for (let kk = 0; kk <= N; kk += 2) text(String(kk), X(kk), T + ph + 9, CSS.mute, { size: 10, align: 'center' });
    text('iteration k', L + pw / 2, T + ph + 22, CSS.mute, { size: 11, align: 'center' });
    g.restore();
    const curve = (rs, upto, color, { w = 2, a = 1, dash = null, dots = true } = {}) => {
      g.save(); g.globalAlpha = a; g.strokeStyle = color; g.fillStyle = color; g.lineWidth = w; if (dash) g.setLineDash(dash);
      g.beginPath(); for (let j = 0; j <= upto; j++) { const x = X(j), y = Y(rs[j]); if (j) g.lineTo(x, y); else g.moveTo(x, y); }
      g.stroke(); g.setLineDash([]);
      if (dots) for (let j = 0; j <= upto; j++) { g.beginPath(); g.arc(X(j), Y(rs[j]), 2.4, 0, 2 * Math.PI); g.fill(); }
      g.restore();
    };
    const runs = A ? P.A : P.B;
    curve(runs[m].res, N, MCSS[m], { w: 1.2, a: 0.3, dash: [4, 4], dots: false });   // computed, not yet stepped
    let ly = T + 12;
    for (const mm of METHODS) {
      const active = mm === m, upto = active ? k : st.reached[mm];
      if (!active && upto === 0) continue;
      if (active && st.reached[mm] > k) curve(runs[mm].res, st.reached[mm], MCSS[mm], { w: 1.4, a: 0.4 });
      curve(runs[mm].res, upto, MCSS[mm], { w: active ? 2.2 : 1.4, a: active ? 1 : 0.4 });
      text(LABEL[mm], L + pw - 6, ly, MCSS[mm], { size: 11, align: 'right' });
      ly += 15;
    }
    const [hx, hy] = [X(k), Y(runs[m].res[k])];
    g.save(); g.fillStyle = MCSS[m]; g.strokeStyle = '#000'; g.lineWidth = 1.5; g.beginPath(); g.arc(hx, hy, 4.5, 0, 2 * Math.PI); g.fill(); g.stroke(); g.restore();
    const big = maxOf(runs.retract.res);
    if (m === 'retract' && k > 0) text(`≤ ${upBound(big)}`, L + pw - 6, Y(RES_FLOOR) - 10, CSS.white, { size: 11, align: 'right' });
  }

  function info() {
    const k = kNow(), m = st.method;
    if (st.act === 'A') {
      const R = P.A[m];
      return `k = <b>${k}</b> · |x| = <b>${(R.res[k] > 1e-12 && Math.abs(R.nrm[k] - 1) < 5e-4 ? R.nrm[k].toFixed(9) : fmt(R.nrm[k], 3))}</b> · |xᵀx − 1| = <b>${fr(R.res[k])}</b> · angle to t <b>${fmt(deg(R.ang[k]), 1)}°</b>`;
    }
    const R = P.B[m];
    return `k = <b>${k}</b> · σ(X) = <b>${R.sv[k].map((s) => s.toFixed(2)).join(', ')}</b> · det <b>${fmt(R.det[k], 2)}</b> · ‖XᵀX − I‖ = <b>${fr(R.res[k])}</b> · angle <b>${fmt(deg(R.ang[k]), 1)}°</b>`;
  }

  // ---------------------------------------------------------------------------------------------- controls
  seg(panel, 'act', (a) => {
    st.act = a; st.pos = 0; st.goal = 0; st.back = null;
    for (const mm of METHODS) st.reached[mm] = 0;
    const cam = camFor(a); v.homeView = cam; v.setCamera(cam); dirty = true;
  });
  seg(panel, 'method', (mm) => { rewind(); st.method = mm; });
  onInput(panel, 'alpha', recompute);
  const onTarget = () => { t = unit(dirFromAngles(val(panel, 'az'), val(panel, 'el'))); recompute(); };
  onInput(panel, 'az', onTarget);
  onInput(panel, 'el', onTarget);
  onClick(panel, 'step', () => {
    if (st.back) st.back = null;
    if (st.pos >= st.goal && st.goal < N) { st.goal = Math.min(N, Math.floor(st.pos + 1e-9) + 1); st.dur = DUR; dirty = true; }
  });
  onClick(panel, 'run', () => { if (st.back) st.back = null; st.goal = N; st.dur = DUR_RUN; dirty = true; });
  onClick(panel, 'reset', rewind);

  // ---------------------------------------------------------------------------------------------- frame loop
  let last = performance.now();
  v.onFrame(() => {
    const now = performance.now(), dt = Math.min((now - last) / 1000, 0.25);   // wall clock: steps keep their pace
    last = now;
    if (st.back) { st.back.u += dt / DUR_BACK; if (st.back.u >= 1) st.back = null; dirty = true; }
    else if (st.pos < st.goal) {
      st.pos = Math.min(st.goal, st.pos + dt / st.dur);
      st.reached[st.method] = Math.max(st.reached[st.method], kNow());
      dirty = true;
    }
    const e = v.camera.matrixWorld.elements, key = [0, 1, 2, 4, 5, 6].map((i) => e[i].toFixed(4)).join();
    if (!dirty && key === lastBasis) return;
    const d = current();
    if (dirty) { trackFrame(armFrame(d)); update3D(d); readout(panel, 'info', info()); }
    drawPanel(d);
    lastBasis = key; dirty = false;
  });

  // ---------------------------------------------------------------------------------------------- checks
  const ref = problem(T_DEFAULT, 0.6, R0);
  const vsClip = {
    source: 'Lab6_Optimization/build/out/s32_manifold_facts.json (browser value, clip value)',
    theta_A_deg: [deg(ref.thetaA), CLIP.theta_A_deg], theta_B_deg: [deg(ref.thetaB), CLIP.theta_B_deg],
    sphere_plain_norm_3: [ref.A.plain.nrm[3], CLIP.sphere_plain_norm_3],
    sphere_plain_residual_10: [ref.A.plain.res[10], CLIP.sphere_plain_residual_10],
    sphere_tangent_steps_to_1deg: [stepsTo1(ref.A.retract.ang), CLIP.sphere_tangent_steps_to_1deg],
    sphere_tangent_max_residual: [maxOf(ref.A.retract.res), CLIP.sphere_tangent_max_residual],
    sphere_tangent_only_norm_1_3_10: [[1, 3, 10].map((j) => ref.A.tangent.nrm[j]), CLIP.sphere_tangent_only_norm_1_3_10],
    so3_plain_residual_3: [ref.B.plain.res[3], CLIP.so3_plain_residual_3],
    so3_plain_residual_10: [ref.B.plain.res[10], CLIP.so3_plain_residual_10],
    so3_tangent_steps_to_1deg: [stepsTo1(ref.B.retract.ang), CLIP.so3_tangent_steps_to_1deg],
    so3_tangent_max_residual: [maxOf(ref.B.retract.res), CLIP.so3_tangent_max_residual],
    so3_tangent_only_residual_1_3_10: [[1, 3, 10].map((j) => ref.B.tangent.res[j]), CLIP.so3_tangent_only_residual_1_3_10],
  };

  panel._state = () => {
    const k = kNow(), A = P.A[st.method], B = P.B[st.method];
    return {
      act: st.act, method: st.method, alpha: P.alpha, t: P.t, k, goal: st.goal, animating: animating(),
      reached: { ...st.reached },
      startAngleDeg: deg(P.thetaA), rotationAngleDeg: deg(P.thetaB),
      sphere: { norm: A.nrm[k], residual: A.res[k], angleDeg: deg(A.ang[k]), residuals: A.res, stepsTo1deg: stepsTo1(A.ang) },
      rotation: { residual: B.res[k], det: B.det[k], singularValues: B.sv[k], angleDeg: deg(B.ang[k]), residuals: B.res,
        stepsTo1deg: stepsTo1(B.ang) },
      maxResidualRetraction: { sphere: maxOf(P.A.retract.res), rotation: maxOf(P.B.retract.res) },
      ik: { ...ik, toolPointDrift_m: S.norm(S.sub(S.fk(q).p, p0)), viewerToolError_m: S.norm(S.sub(v.tool(), tool)) },
      q: q.slice(), hiddenToolGeoms, vsClip,
    };
  };
}
