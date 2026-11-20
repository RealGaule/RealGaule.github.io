// sim6.js: the numerics of ROB7104 Lab 6 (Optimization) for the browser.
//
// One ES module, no dependencies. It mirrors the Python core of the lab (sim/arm.py, sim/solvers.py,
// sim/structure.py, sim/experiments.py) function by function, and is checked against it by web/run_tests.py
// (fixtures written by web/export_fixtures.py). Notation follows Lecture 4: iterate x_k, step size alpha,
// gradient g, Hessian H, curvature bounds mu and L, noise sigma_g, penalty weight lambda.
//
// Conventions
//   vectors   plain arrays of numbers
//   matrices  arrays of rows (A[i][j])
//   angles    radians, MuJoCo joint angles of the UR5e in the order of JOINTS
//   lengths   metres; positions are in the arm's base frame (base at the origin, z up), as in sim/arm.py.
//             The 3D stage puts the arm on a pedestal: add the pedestal height to z before drawing.
//
// Sections: 1 random numbers, 2 linear algebra, 3 rotations, 4 six-joint arm, 5 two-joint arm,
//           6 solvers, 7 inverse kinematics, 8 noise, 9 l1 and sparsity, 10 paths (calculus of variations),
//           11 sphere and SO(3).

export const VERSION = '2026-11-20';
const EPS = 2.220446049250313e-16;
const PI = Math.PI;

// =====================================================================================================================
// 1. Seeded random numbers
// =====================================================================================================================

/** xoshiro128** generator, seeded through splitmix32. The stream is NOT numpy's: results of stochastic
 *  experiments agree with the Python core only in distribution (see the stochastic tests). */
export class RNG {
  constructor(seed = 0) {
    let s = seed >>> 0;
    const sm = () => {
      s = (s + 0x9e3779b9) >>> 0;
      let z = s;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
      return (z ^ (z >>> 16)) >>> 0;
    };
    this.s = new Uint32Array([sm(), sm(), sm(), sm()]);
    this._spare = null;
  }

  /** Next 32-bit unsigned integer. */
  nextU32() {
    const s = this.s;
    const x = Math.imul(s[1], 5);
    const r = Math.imul((x << 7) | (x >>> 25), 9) >>> 0;
    const t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = (s[3] << 11) | (s[3] >>> 21);
    return r;
  }

  /** Uniform in [0, 1) with 53 random bits. */
  uniform() {
    const a = this.nextU32() >>> 5, b = this.nextU32() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  }

  /** Uniform in [lo, hi). */
  range(lo, hi) { return lo + (hi - lo) * this.uniform(); }

  /** Integer in 0 .. n-1. */
  int(n) { return Math.floor(this.uniform() * n); }

  /** Standard normal (Box-Muller, both values used). */
  normal() {
    if (this._spare !== null) { const v = this._spare; this._spare = null; return v; }
    const r = Math.sqrt(-2 * Math.log(1 - this.uniform())), th = 2 * PI * this.uniform();
    this._spare = r * Math.sin(th);
    return r * Math.cos(th);
  }

  normalArray(n) { const out = new Array(n); for (let i = 0; i < n; i++) out[i] = this.normal(); return out; }

  /** k distinct integers from 0 .. n-1 (partial Fisher-Yates). */
  choice(n, k) {
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < k; i++) { const j = i + this.int(n - i); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    return idx.slice(0, k);
  }
}

// =====================================================================================================================
// 2. Small dense linear algebra
// =====================================================================================================================
export const zeros = (n) => new Array(n).fill(0);
export const zerosM = (n, m) => Array.from({ length: n }, () => new Array(m).fill(0));
export const eye = (n) => Array.from({ length: n }, (_, i) => { const r = new Array(n).fill(0); r[i] = 1; return r; });
export const copyM = (A) => A.map((r) => r.slice());
export function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
export const norm = (a) => Math.sqrt(dot(a, a));
export const normInf = (a) => { let m = 0; for (const v of a) m = Math.max(m, Math.abs(v)); return m; };
export const add = (a, b) => a.map((v, i) => v + b[i]);
export const sub = (a, b) => a.map((v, i) => v - b[i]);
export const scale = (a, s) => a.map((v) => v * s);
/** a + s b */
export const axpy = (a, s, b) => a.map((v, i) => v + s * b[i]);
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const transpose = (A) => A[0].map((_, j) => A.map((r) => r[j]));
export const matvec = (A, x) => A.map((r) => dot(r, x));
/** A^T x without forming the transpose. */
export function matTvec(A, x) {
  const n = A[0].length, out = new Array(n).fill(0);
  for (let i = 0; i < A.length; i++) { const r = A[i], xi = x[i]; for (let j = 0; j < n; j++) out[j] += r[j] * xi; }
  return out;
}
export function matmul(A, B) {
  const n = A.length, m = B[0].length, k = B.length, C = zerosM(n, m);
  for (let i = 0; i < n; i++) {
    const Ai = A[i], Ci = C[i];
    for (let l = 0; l < k; l++) { const a = Ai[l]; if (a === 0) continue; const Bl = B[l]; for (let j = 0; j < m; j++) Ci[j] += a * Bl[j]; }
  }
  return C;
}
export const addM = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
export const subM = (A, B) => A.map((r, i) => r.map((v, j) => v - B[i][j]));
export const scaleM = (A, s) => A.map((r) => r.map((v) => v * s));
export const outer = (a, b) => a.map((x) => b.map((y) => x * y));
export const trace = (A) => { let s = 0; for (let i = 0; i < A.length; i++) s += A[i][i]; return s; };
export const frob = (A) => { let s = 0; for (const r of A) for (const v of r) s += v * v; return Math.sqrt(s); };
export const column = (A, j) => A.map((r) => r[j]);
export const isFiniteVec = (a) => a.every(Number.isFinite);

/** LU factorisation with partial pivoting. Returns {LU, piv, sign, singular}. */
export function lu(A) {
  const n = A.length, LU = copyM(A), piv = Array.from({ length: n }, (_, i) => i);
  let sign = 1, singular = false;
  for (let k = 0; k < n; k++) {
    let p = k, best = Math.abs(LU[k][k]);
    for (let i = k + 1; i < n; i++) { const v = Math.abs(LU[i][k]); if (v > best) { best = v; p = i; } }
    if (!(best > 0)) { singular = true; continue; }
    if (p !== k) { [LU[p], LU[k]] = [LU[k], LU[p]]; [piv[p], piv[k]] = [piv[k], piv[p]]; sign = -sign; }
    const pk = LU[k][k], rk = LU[k];
    for (let i = k + 1; i < n; i++) {
      const ri = LU[i], l = ri[k] / pk;
      ri[k] = l;
      if (l !== 0) for (let j = k + 1; j < n; j++) ri[j] -= l * rk[j];
    }
  }
  return { LU, piv, sign, singular };
}

function luSolveVec(f, b) {
  const { LU, piv } = f, n = LU.length, x = new Array(n);
  for (let i = 0; i < n; i++) { let s = b[piv[i]]; const r = LU[i]; for (let j = 0; j < i; j++) s -= r[j] * x[j]; x[i] = s; }
  for (let i = n - 1; i >= 0; i--) { let s = x[i]; const r = LU[i]; for (let j = i + 1; j < n; j++) s -= r[j] * x[j]; x[i] = s / r[i]; }
  return x;
}

/** Solve A x = b (b a vector) or A X = B (B a matrix). Throws Error('singular matrix') like numpy.linalg.solve. */
export function solve(A, b) {
  const f = lu(A);
  if (f.singular) throw new Error('singular matrix');
  if (Array.isArray(b[0])) return transpose(transpose(b).map((col) => luSolveVec(f, col)));
  return luSolveVec(f, b);
}
export function det(A) {
  const f = lu(A);
  if (f.singular) return 0;
  let d = f.sign;
  for (let i = 0; i < A.length; i++) d *= f.LU[i][i];
  return d;
}
export const inv = (A) => solve(A, eye(A.length));

/** Cholesky factor L (lower, A = L L^T) of a symmetric matrix, or null if A is not positive definite. */
export function cholesky(A) {
  const n = A.length, L = zerosM(n, n);
  for (let j = 0; j < n; j++) {
    let s = A[j][j];
    for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k];
    if (!(s > 0)) return null;
    const d = Math.sqrt(s);
    L[j][j] = d;
    for (let i = j + 1; i < n; i++) {
      let t = A[i][j];
      for (let k = 0; k < j; k++) t -= L[i][k] * L[j][k];
      L[i][j] = t / d;
    }
  }
  return L;
}
export function choleskySolve(L, b) {
  const n = L.length, y = new Array(n), x = new Array(n);
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
  return x;
}

/** Symmetric eigen-decomposition by cyclic Jacobi rotations.
 *  Returns {values (ascending), vectors}: column k of `vectors` is the unit eigenvector of values[k]. */
export function eigSym(A) {
  const n = A.length, a = copyM(A), V = eye(n);
  const tiny = 1e-3 * EPS * Math.max(frob(A), 1e-300);
  for (let sweep = 0; sweep < 60; sweep++) {
    let rotated = false;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p][q];
        if (Math.abs(apq) <= tiny) { a[p][q] = a[q][p] = 0; continue; }
        rotated = true;
        const theta = (a[q][q] - a[p][p]) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
        for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
        for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
      }
    }
    if (!rotated) break;
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i][i] - a[j][j]);
  return { values: order.map((i) => a[i][i]), vectors: V.map((r) => order.map((i) => r[i])) };
}

/** Eigenvalues [min, max] of a symmetric 2 x 2 matrix (closed form, as in arm.TwoJointArm.hess_eig). */
export function eig2(H) {
  const m = 0.5 * (H[0][0] + H[1][1]);
  const h = 0.5 * (H[0][0] - H[1][1]);
  const d = Math.sqrt(h * h + H[0][1] * H[0][1]);
  return [m - d, m + d];
}

/** Thin singular value decomposition A = U diag(s) V^T by one-sided Jacobi rotations (Hestenes).
 *  A is m x n; with k = min(m, n): U is m x k, s has k values (largest first), V is n x k. */
export function svd(A) {
  const m = A.length, n = A[0].length;
  if (m < n) { const r = svd(transpose(A)); return { U: r.V, s: r.s, V: r.U }; }
  const U = copyM(A), V = eye(n);
  for (let sweep = 0; sweep < 80; sweep++) {
    let rotated = false;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        let al = 0, be = 0, ga = 0;
        for (let i = 0; i < m; i++) { const up = U[i][p], uq = U[i][q]; al += up * up; be += uq * uq; ga += up * uq; }
        if (ga === 0 || Math.abs(ga) <= EPS * Math.sqrt(al * be)) continue;
        rotated = true;
        const zeta = (be - al) / (2 * ga);
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t), s = c * t;
        for (let i = 0; i < m; i++) { const up = U[i][p], uq = U[i][q]; U[i][p] = c * up - s * uq; U[i][q] = s * up + c * uq; }
        for (let i = 0; i < n; i++) { const vp = V[i][p], vq = V[i][q]; V[i][p] = c * vp - s * vq; V[i][q] = s * vp + c * vq; }
      }
    }
    if (!rotated) break;
  }
  const sv = new Array(n);
  for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < m; i++) s += U[i][j] * U[i][j]; sv[j] = Math.sqrt(s); }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => sv[j] - sv[i]);
  const smax = sv[order[0]];
  const Uo = zerosM(m, n), Vo = zerosM(n, n), so = order.map((j) => sv[j]);
  for (let c = 0; c < n; c++) {
    const j = order[c];
    for (let i = 0; i < n; i++) Vo[i][c] = V[i][j];
    if (sv[j] > 1e-300 && sv[j] > 1e3 * EPS * EPS * smax) for (let i = 0; i < m; i++) Uo[i][c] = U[i][j] / sv[j];
    else so[c] = sv[j], completeColumn(Uo, c);
  }
  return { U: Uo, s: so, V: Vo };
}

// Fill column c of U with a unit vector orthogonal to columns 0 .. c-1 (for a zero singular value).
function completeColumn(U, c) {
  const m = U.length;
  for (let e = 0; e < m; e++) {
    const v = new Array(m).fill(0);
    v[e] = 1;
    for (let pass = 0; pass < 2; pass++) {
      for (let j = 0; j < c; j++) { let d = 0; for (let i = 0; i < m; i++) d += U[i][j] * v[i]; for (let i = 0; i < m; i++) v[i] -= d * U[i][j]; }
    }
    const nv = norm(v);
    if (nv > 0.3) { for (let i = 0; i < m; i++) U[i][c] = v[i] / nv; return; }
  }
}

/** Singular values, largest first. */
export const singularValues = (A) => svd(A).s;
/** Spectral norm |A|_2. */
export const norm2 = (A) => svd(A).s[0];

/** Minimum-norm least-squares solution of A x = b (numpy.linalg.lstsq with rcond=None:
 *  singular values below eps * max(m, n) * s_max count as zero). */
export function lstsq(A, b) {
  const { U, s, V } = svd(A), m = A.length, n = A[0].length;
  const cut = EPS * Math.max(m, n) * s[0], x = new Array(n).fill(0);
  for (let k = 0; k < s.length; k++) {
    if (!(s[k] > cut)) continue;
    let c = 0;
    for (let i = 0; i < m; i++) c += U[i][k] * b[i];
    c /= s[k];
    for (let j = 0; j < n; j++) x[j] += c * V[j][k];
  }
  return x;
}

/** Nelder-Mead minimiser (used to polish grid searches). Returns {x, f, iters}. */
export function nelderMead(fun, x0, { xatol = 1e-10, fatol = 1e-14, maxIter = 2000 } = {}) {
  const n = x0.length;
  let S = [x0.slice()];
  for (let i = 0; i < n; i++) { const y = x0.slice(); y[i] = y[i] !== 0 ? 1.05 * y[i] : 0.00025; S.push(y); }
  let F = S.map(fun), it = 0;
  for (; it < maxIter; it++) {
    const order = Array.from({ length: n + 1 }, (_, i) => i).sort((i, j) => F[i] - F[j]);
    S = order.map((i) => S[i]); F = order.map((i) => F[i]);
    let dx = 0, df = 0;
    for (let i = 1; i <= n; i++) { dx = Math.max(dx, normInf(sub(S[i], S[0]))); df = Math.max(df, Math.abs(F[i] - F[0])); }
    if (dx <= xatol && df <= fatol) break;
    const c = zeros(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c[j] += S[i][j] / n;
    const xr = axpy(c, 1, sub(c, S[n])), fr = fun(xr);
    if (fr < F[0]) {
      const xe = axpy(c, 2, sub(c, S[n])), fe = fun(xe);
      if (fe < fr) { S[n] = xe; F[n] = fe; } else { S[n] = xr; F[n] = fr; }
    } else if (fr < F[n - 1]) { S[n] = xr; F[n] = fr; } else {
      const outside = fr < F[n];
      const xc = outside ? axpy(c, 0.5, sub(xr, c)) : axpy(c, 0.5, sub(S[n], c)), fc = fun(xc);
      if (fc < (outside ? fr : F[n])) { S[n] = xc; F[n] = fc; } else {
        for (let i = 1; i <= n; i++) { S[i] = axpy(S[0], 0.5, sub(S[i], S[0])); F[i] = fun(S[i]); }
      }
    }
  }
  let b = 0;
  for (let i = 1; i <= n; i++) if (F[i] < F[b]) b = i;
  return { x: S[b], f: F[b], iters: it };
}

// =====================================================================================================================
// 3. Rotations
// =====================================================================================================================

/** Wrap an angle (or each entry of an array) to [-pi, pi), with Python's modulo. */
export function wrap(q) {
  if (Array.isArray(q)) return q.map(wrap);
  let m = (q + PI) % (2 * PI);
  if (m < 0) m += 2 * PI;
  return m - PI;
}
/** 3-vector -> skew-symmetric matrix. */
export const hat = (w) => [[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]];
/** Skew-symmetric matrix -> 3-vector. */
export const vee = (W) => [W[2][1], W[0][2], W[1][0]];
export const sym = (A) => A.map((r, i) => r.map((v, j) => 0.5 * (v + A[j][i])));
export const skew = (A) => A.map((r, i) => r.map((v, j) => 0.5 * (v - A[j][i])));

/** Unit quaternion (w, x, y, z; normalised here) -> rotation matrix. */
export function quatToMat(qu) {
  const n = Math.sqrt(qu[0] * qu[0] + qu[1] * qu[1] + qu[2] * qu[2] + qu[3] * qu[3]);
  const w = qu[0] / n, x = qu[1] / n, y = qu[2] / n, z = qu[3] / n;
  return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]];
}

/** Rotation by angle th about the unit axis a (Rodrigues). */
export function axisAngle(a, th) {
  const K = hat(a), K2 = matmul(K, K), s = Math.sin(th), c = 1 - Math.cos(th);
  return K.map((r, i) => r.map((v, j) => (i === j ? 1 : 0) + s * v + c * K2[i][j]));
}

/** Exponential map of SO(3): rotation vector (axis * angle) -> rotation matrix.
 *  (arm.matrix_from_rotvec and structure.so3_exp are the same function.) */
export function so3Exp(w) {
  const th = norm(w), K = hat(w), K2 = matmul(K, K);
  const a = th < 1e-8 ? 1 : Math.sin(th) / th, b = th < 1e-8 ? 0.5 : (1 - Math.cos(th)) / (th * th);
  return K.map((r, i) => r.map((v, j) => (i === j ? 1 : 0) + a * v + b * K2[i][j]));
}

/** Logarithm of SO(3): rotation matrix -> rotation vector, angle in [0, pi] (arm.rotvec_from_matrix). */
export function so3Log(R) {
  const v = [0.5 * (R[2][1] - R[1][2]), 0.5 * (R[0][2] - R[2][0]), 0.5 * (R[1][0] - R[0][1])];
  const c = Math.min(1, Math.max(-1, 0.5 * (trace(R) - 1)));
  const s = norm(v), th = Math.atan2(s, c);
  if (th < 1e-8) return v;
  if (th < PI - 1e-3) return scale(v, th / s);
  // near pi the vector v is tiny: use the symmetric part, (R + R^T)/2 = cos I + (1 - cos) a a^T
  const cth = Math.cos(th);
  const M = R.map((r, i) => r.map((x, j) => (0.5 * (x + R[j][i]) - (i === j ? cth : 0)) / (1 - cth)));
  let k = 0;
  for (let i = 1; i < 3; i++) if (M[i][i] > M[k][k]) k = i;
  let a = scale(column(M, k), 1 / Math.sqrt(M[k][k]));
  a = scale(a, 1 / norm(a));
  if (dot(a, v) < 0) a = scale(a, -1);
  return scale(a, th);
}

/** Rotation vector (world frame) of the rotation that carries R onto Rt: log(Rt R^T). */
export const orientationError = (R, Rt) => so3Log(matmul(Rt, transpose(R)));

/** Angle (rad) of the rotation Ra^T Rb, accurate near 0 and near pi. */
export function rotationAngle(Ra, Rb) {
  const E = matmul(transpose(Ra), Rb);
  return Math.atan2(norm(vee(skew(E))), 0.5 * (trace(E) - 1));
}

// =====================================================================================================================
// 4. Six-joint arm (UR5e, lengths of the MuJoCo Menagerie model)
// =====================================================================================================================
export const JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'];

// MuJoCo convention: child frame = parent frame * Trans(pos) * Rot(quat) * Rot(axis, q); numbers from ur5e.xml.
const LINK_POS = [[0, 0, 0.163], [0, 0.138, 0], [0, -0.131, 0.425], [0, 0, 0.392], [0, 0.127, 0], [0, 0, 0.1]];
const LINK_QUAT = [[1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0]];
const LINK_AXIS = [[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]];
const LINK_ROT = LINK_QUAT.map(quatToMat);
const BASE_ROT = quatToMat([0, 0, 0, -1]);
const SITE_POS = [0, 0.1, 0];
const SITE_ROT = quatToMat([-1, 1, 0, 0]);

/** Model constants. Joint limits are those of the model; the solvers of the lab treat angles as periodic and do
 *  not enforce them. */
export const UR5E = {
  n: 6,
  lower: [-6.28319, -6.28319, -3.1415, -6.28319, -6.28319, -6.28319],
  upper: [6.28319, 6.28319, 3.1415, 6.28319, 6.28319, 6.28319],
  home: [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0],          // keyframe "home" of the model
  Q_PLANE: [PI, -1.0, 1.3, -PI / 2, 0, 0],                        // a pose of the two-joint problem
  Q_HOME: [PI, -1.9, 1.7, -1.4, -PI / 2, 0],                      // the six-joint start pose of the lab (tool down)
  linkPos: LINK_POS, linkQuat: LINK_QUAT, linkAxis: LINK_AXIS, sitePos: SITE_POS,
};

function chain(q) {
  let R = BASE_ROT, o = [0, 0, 0];
  const origins = [], axes = [];
  for (let i = 0; i < 6; i++) {
    o = add(o, matvec(R, LINK_POS[i]));
    R = matmul(R, LINK_ROT[i]);
    origins.push(o);
    axes.push(matvec(R, LINK_AXIS[i]));
    R = matmul(R, axisAngle(LINK_AXIS[i], q[i]));
  }
  return { p: add(o, matvec(R, SITE_POS)), R: matmul(R, SITE_ROT), origins, axes };
}

/** Tool position p (3) and orientation R (3 x 3) in the base frame. */
export function fk(q) { const c = chain(q); return { p: c.p, R: c.R }; }

/** {p, R, Jp, Jr}: Jp and Jr are 3 x 6, tool velocity = Jp qdot, angular velocity = Jr qdot (world frame). */
export function fkJac(q) {
  const c = chain(q), Jp = zerosM(3, 6), Jr = zerosM(3, 6);
  for (let i = 0; i < 6; i++) {
    const col = cross(c.axes[i], sub(c.p, c.origins[i]));
    for (let k = 0; k < 3; k++) { Jp[k][i] = col[k]; Jr[k][i] = c.axes[i][k]; }
  }
  return { p: c.p, R: c.R, Jp, Jr };
}

/** Positions of the six link frames followed by the tool point, 7 x 3 (for drawing a skeleton). */
export function jointPositions(q) { const c = chain(q); return [...c.origins, c.p]; }

/** Link frames for drawing: [{p, R}] for the six links (after their joint rotation) and the tool. */
export function linkFrames(q) {
  let R = BASE_ROT, o = [0, 0, 0];
  const out = [];
  for (let i = 0; i < 6; i++) {
    o = add(o, matvec(R, LINK_POS[i]));
    R = matmul(matmul(R, LINK_ROT[i]), axisAngle(LINK_AXIS[i], q[i]));
    out.push({ p: o, R });
  }
  out.push({ p: add(o, matvec(R, SITE_POS)), R: matmul(R, SITE_ROT) });
  return out;
}

/** Task error e (desired minus current) and its Jacobian J, so that e(q + dq) ~ e(q) - J dq.
 *  Position only (Rt null): e = pt - p (3), J = Jp. With orientation: e = [pt - p ; w log(Rt R^T)] (6),
 *  J = [Jp ; w Jr], w = rotWeight (metres per radian). The cost 1/2 |e|^2 has gradient -J^T e. */
export function poseError(q, pt, Rt = null, rotWeight = 1.0) {
  const k = fkJac(q), ep = sub(pt, k.p);
  if (!Rt) return { e: ep, J: k.Jp };
  const er = scale(orientationError(k.R, Rt), rotWeight);
  return { e: [...ep, ...er], J: [...k.Jp, ...scaleM(k.Jr, rotWeight)] };
}

/** (p, J) for position targets: the generic callable of the l1 / l2 reach code. */
export function fkJacPos(q) { const k = fkJac(q); return { p: k.p, J: k.Jp }; }

/** Singular values of Jp (3) or of [Jp ; w Jr] (6), largest first. */
export function armSingularValues(q, orientation = false, rotWeight = 1.0) {
  const k = fkJac(q);
  return singularValues(orientation ? [...k.Jp, ...scaleM(k.Jr, rotWeight)] : k.Jp);
}
/** Yoshikawa measure of the position task: product of the singular values of Jp. */
export const armManipulability = (q) => armSingularValues(q).reduce((a, b) => a * b, 1);
export const withinLimits = (q) => q.every((v, i) => v >= UR5E.lower[i] && v <= UR5E.upper[i]);

// =====================================================================================================================
// 5. Two-joint arm: shoulder_lift and elbow free, the other four joints fixed
// =====================================================================================================================

/** An exact planar two-link arm cut out of the six-joint arm (arm.TwoJointArm).
 *  q = [shoulder_lift, elbow]; targets t are plane coordinates [x, z] relative to the shoulder axis.
 *  Default fixed pose: pan = pi, wrist = (-pi/2, 0, 0): l1 = 0.425, l2 = 0.492, reach 0.067 .. 0.917 m,
 *  plane y = 0.234, NEGATIVE shoulder_lift raises the arm. */
export class TwoJointArm {
  constructor(pan = PI, wrist = [-PI / 2, 0, 0]) {
    this.qFixed = [pan, 0, 0, wrist[0], wrist[1], wrist[2]];
    this.lower = [UR5E.lower[1], UR5E.lower[2]];
    this.upper = [UR5E.upper[1], UR5E.upper[2]];
    const P = (a, b) => fk(this.fullQ([a, b])).p;
    const h = PI / 2;
    const p00 = P(0, 0), p10 = P(h, 0), p20 = P(PI, 0), p01 = P(0, h), p02 = P(0, PI);
    const c = scale(add(p00, p20), 0.5);
    const sCol1 = scale(sub(p00, p20), 0.5), sCol2 = sub(p10, c);
    const bCol1 = scale(sub(p00, p02), 0.5);
    const aCol1 = sub(sCol1, bCol1);
    const bCol2 = sub(sub(p01, c), aCol1);
    const aCol2 = sub(sCol2, bCol2);
    let normal = cross(aCol1, aCol2);
    normal = scale(normal, 1 / norm(normal));
    const ez = [0, 0, 1];
    if (Math.abs(dot(normal, ez)) > 1e-9) throw new Error('plane of motion is not vertical');
    let ex = axpy(aCol1, -dot(aCol1, ez), ez);
    if (norm(ex) < 1e-9) ex = axpy(aCol2, -dot(aCol2, ez), ez);
    ex = scale(ex, 1 / norm(ex));
    this.ex = ex; this.ez = ez; this.normal = cross(ez, ex);
    this.origin = c;
    // 2 x 2 matrices: column 0 is the link vector at angle 0, column 1 at angle pi/2 (plane coordinates)
    this.A = [[dot(ex, aCol1), dot(ex, aCol2)], [dot(ez, aCol1), dot(ez, aCol2)]];
    this.B = [[dot(ex, bCol1), dot(ex, bCol2)], [dot(ez, bCol1), dot(ez, bCol2)]];
    this.l1 = Math.sqrt(this.A[0][0] ** 2 + this.A[1][0] ** 2);
    this.l2 = Math.sqrt(this.B[0][0] ** 2 + this.B[1][0] ** 2);
    this.reachMax = this.l1 + this.l2;
    this.reachMin = Math.abs(this.l1 - this.l2);
    this.sgn = Math.sign(this.A[0][0] * this.A[1][1] - this.A[0][1] * this.A[1][0]);
    this.phi1 = Math.atan2(this.A[1][0], this.A[0][0]);
    this.phi2 = Math.atan2(this.B[1][0], this.B[0][0]);
    this.elbowStraight = wrap(this.sgn * (this.phi1 - this.phi2));
  }

  /** [shoulder_lift, elbow] -> the six joint angles that the 3D stage should display. */
  fullQ(q) { const out = this.qFixed.slice(); out[1] = q[0]; out[2] = q[1]; return out; }
  /** Plane coordinates [x, z] -> position in the base frame. */
  toWorld(xz) { return [0, 1, 2].map((k) => this.origin[k] + xz[0] * this.ex[k] + xz[1] * this.ez[k]); }
  /** Position in the base frame -> {xz, off}: plane coordinates and the out-of-plane distance. */
  toPlane(p) { const d = sub(p, this.origin); return { xz: [dot(d, this.ex), dot(d, this.ez)], off: dot(d, this.normal) }; }

  // link vectors v1, v2 and their derivatives d1 = dv1/dq1, d2 = dv2/dq1 = dv2/dq2 (same operation order as arm.py)
  _parts(q) {
    const a = q[0], b = a + q[1];
    const c1 = Math.cos(a), s1 = Math.sin(a), c12 = Math.cos(b), s12 = Math.sin(b), A = this.A, B = this.B;
    return [c1 * A[0][0] + s1 * A[0][1], c1 * A[1][0] + s1 * A[1][1],
      c12 * B[0][0] + s12 * B[0][1], c12 * B[1][0] + s12 * B[1][1],
      -s1 * A[0][0] + c1 * A[0][1], -s1 * A[1][0] + c1 * A[1][1],
      -s12 * B[0][0] + c12 * B[0][1], -s12 * B[1][0] + c12 * B[1][1]];
  }

  /** Tool point in plane coordinates. */
  p(q) { const s = this._parts(q); return [s[0] + s[2], s[1] + s[3]]; }
  /** Elbow point in plane coordinates. */
  elbow(q) { const s = this._parts(q); return [s[0], s[1]]; }
  pWorld(q) { return this.toWorld(this.p(q)); }
  /** Jacobian dp/dq, 2 x 2. */
  jac(q) { const s = this._parts(q); return [[s[4] + s[6], s[6]], [s[5] + s[7], s[7]]]; }

  /** f(q) = 1/2 |p(q) - t|^2, in residual form (never expand it: see the Armijo finding in PROGRESS.md). */
  cost(q, t) {
    const s = this._parts(q), rx = s[0] + s[2] - t[0], rz = s[1] + s[3] - t[1];
    return 0.5 * (rx * rx + rz * rz);
  }

  /** Gradient J^T (p - t). */
  grad(q, t) {
    const s = this._parts(q), rx = s[0] + s[2] - t[0], rz = s[1] + s[3] - t[1];
    return [(s[4] + s[6]) * rx + (s[5] + s[7]) * rz, s[6] * rx + s[7] * rz];
  }

  /** Hessian J^T J + sum_k r_k grad^2 p_k (at a solution r = 0 and H = J^T J). */
  hess(q, t) {
    const s = this._parts(q), rx = s[0] + s[2] - t[0], rz = s[1] + s[3] - t[1];
    const jx = s[4] + s[6], jz = s[5] + s[7];
    const rv2 = rx * s[2] + rz * s[3];
    const h11 = (jx * jx + jz * jz) - (rx * (s[0] + s[2]) + rz * (s[1] + s[3]));
    const h12 = (jx * s[6] + jz * s[7]) - rv2;
    const h22 = (s[6] * s[6] + s[7] * s[7]) - rv2;
    return [[h11, h12], [h12, h22]];
  }

  /** Hessian eigenvalues [min, max]. */
  hessEig(q, t) { return eig2(this.hess(q, t)); }

  /** [min eigenvalue, max eigenvalue] of the Hessian over all joint angles (grid search plus polish).
   *  L = max(|min|, |max|); a fixed step alpha <= 1/L decreases the cost at every step from every start. */
  curvatureBounds(t, n = 360) {
    let lo = Infinity, hi = -Infinity, qlo = null, qhi = null;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const q = [-PI + (i + 0.5) * 2 * PI / n, -PI + (j + 0.5) * 2 * PI / n], ev = this.hessEig(q, t);
        if (ev[0] < lo) { lo = ev[0]; qlo = q; }
        if (ev[1] > hi) { hi = ev[1]; qhi = q; }
      }
    }
    return [nelderMead((x) => this.hessEig(x, t)[0], qlo).f, -nelderMead((x) => -this.hessEig(x, t)[1], qhi).f];
  }

  _qFromLinkAngles(th1, th2) {
    const q1 = this.sgn * (th1 - this.phi1), q12 = this.sgn * (th2 - this.phi2);
    return [wrap(q1), wrap(q12 - q1)];
  }

  reachable(t) { const rho = Math.sqrt(t[0] * t[0] + t[1] * t[1]); return this.reachMin <= rho && rho <= this.reachMax; }

  /** Closed-form solutions of p(q) = t: 0, 1 or 2 wrapped joint vectors (elbow above the shoulder-target line first). */
  ik(t) {
    const rho = Math.sqrt(t[0] * t[0] + t[1] * t[1]);
    if (rho < 1e-12 || !this.reachable(t)) return [];
    const phi = Math.atan2(t[1], t[0]);
    const cg = (rho * rho + this.l1 * this.l1 - this.l2 * this.l2) / (2 * this.l1 * rho);
    const gam = Math.acos(Math.min(1, Math.max(-1, cg)));
    return (gam < 1e-12 ? [1] : [1, -1]).map((s) => {
      const th1 = phi + s * gam;
      const w = [t[0] - this.l1 * Math.cos(th1), t[1] - this.l1 * Math.sin(th1)];
      return this._qFromLinkAngles(th1, Math.atan2(w[1], w[0]));
    });
  }

  /** All stationary points of f on the torus of joint angles, with their types, sorted by cost.
   *  Each: {name, q, f, eig, type ('min' | 'saddle' | 'max' | 'degenerate'), tool, gradNorm, withinLimits}.
   *  The two minima both have cost zero to rounding; they are listed elbow up first. */
  stationaryPoints(t, eigTol = 1e-9) {
    const rho = Math.sqrt(t[0] * t[0] + t[1] * t[1]);
    if (rho < 1e-12) throw new Error('target at the shoulder axis: the stationary points are not isolated');
    const phi = Math.atan2(t[1], t[0]), cands = [], sols = this.ik(t);
    if (sols.length === 2) {
      const up = this.elbow(sols[0])[1] >= this.elbow(sols[1])[1] ? 0 : 1;
      cands.push(['elbow up', sols[up]], ['elbow down', sols[1 - up]]);
    } else if (sols.length === 1) cands.push(['on the reach limit', sols[0]]);
    cands.push(['stretched towards the target', this._qFromLinkAngles(phi, phi)],
      ['stretched away from the target', this._qFromLinkAngles(phi + PI, phi + PI)],
      ['folded, upper arm towards the target', this._qFromLinkAngles(phi, phi + PI)],
      ['folded, upper arm away from the target', this._qFromLinkAngles(phi + PI, phi)]);
    const out = cands.map(([name, q], order) => {
      const ev = this.hessEig(q, t);
      const type = ev[0] > eigTol ? 'min' : ev[1] < -eigTol ? 'max' : (ev[0] < -eigTol && ev[1] > eigTol) ? 'saddle' : 'degenerate';
      return { name, q, f: this.cost(q, t), eig: ev, type, tool: this.p(q), gradNorm: norm(this.grad(q, t)),
        withinLimits: q.every((v, i) => v > this.lower[i] && v < this.upper[i]), order };
    });
    out.sort((a, b) => (Math.abs(a.f - b.f) < 1e-20 ? a.order - b.order : a.f - b.f));
    return out;
  }

  /** Singular values of the 2 x 2 Jacobian, largest first. */
  singularValues(q) { return singularValues(this.jac(q)); }
  /** Condition number of J^T J = (sigma_max / sigma_min)^2. */
  condJtJ(q) { const s = this.singularValues(q); return (s[0] / s[1]) ** 2; }
  /** |det J| = l1 l2 |sin(elbow - elbowStraight)|. */
  manipulability(q) { const J = this.jac(q); return Math.abs(J[0][0] * J[1][1] - J[0][1] * J[1][0]); }

  /** Manipulability ellipse at the tool: {sigmas [2], axesPlane [2][2], axesWorld [2][3]}; row k is the unit
   *  direction of the k-th principal axis, of half length sigmas[k] (tool velocities reachable with |qdot| <= 1). */
  ellipse(q) {
    const { U, s } = svd(this.jac(q)), axes = transpose(U);
    return { sigmas: s, axesPlane: axes, axesWorld: axes.map((a) => [0, 1, 2].map((k) => a[0] * this.ex[k] + a[1] * this.ez[k])) };
  }

  withinLimits(q) { return q[0] >= this.lower[0] && q[0] <= this.upper[0] && q[1] >= this.lower[1] && q[1] <= this.upper[1]; }
}

/** f_w(q) = 1/2 |p(q) - t|^2 - w z_elbow(q): the reach cost with a preference for a high elbow
 *  (experiments.ReachCost, here in residual form). w = 0 gives the plain cost. */
export class ReachCost {
  constructor(two, t, w = 0) {
    this.two = two; this.t = t.slice(); this.w = w;
    this.cost = (q) => two.cost(q, this.t) - this.w * two._parts(q)[1];
    this.grad = (q) => { const g = two.grad(q, this.t); g[0] -= this.w * two._parts(q)[5]; return g; };
    this.hess = (q) => { const H = two.hess(q, this.t); H[0][0] += this.w * two._parts(q)[1]; return H; };
  }

  zElbow(q) { return this.two._parts(q)[1]; }
  /** Distance of the tool point from the target. */
  miss(q) { const p = this.two.p(q); return Math.hypot(p[0] - this.t[0], p[1] - this.t[1]); }

  /** Stationary points of f_w, continued from those of the plain cost by Newton's method; sorted by cost. */
  stationary() {
    const out = this.two.stationaryPoints(this.t).map((sp) => {
      let q = sp.q.slice();
      for (let k = 0; k < 50; k++) {
        const g = this.grad(q);
        if (norm(g) <= 1e-15) break;
        let p;
        try { p = solve(this.hess(q), scale(g, -1)); } catch (e) { break; }
        q = add(q, p);
        if (norm(p) <= 1e-16) break;
      }
      q = wrap(q);
      const ev = eig2(this.hess(q));
      const type = ev[0] > 1e-9 ? 'min' : ev[1] < -1e-9 ? 'max' : 'saddle';
      return { name: sp.name, q, f: this.cost(q), eig: ev, type, gradNorm: norm(this.grad(q)), miss: this.miss(q), zElbow: this.zElbow(q) };
    });
    out.sort((a, b) => a.f - b.f);
    return out;
  }

  /** Largest |Hessian eigenvalue| over the torus (grid estimate). */
  L(n = 360) {
    let m = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const ev = eig2(this.hess([-PI + (i + 0.5) * 2 * PI / n, -PI + (j + 0.5) * 2 * PI / n]));
      m = Math.max(m, Math.abs(ev[0]), Math.abs(ev[1]));
    }
    return m;
  }
}

export const DEFAULT_TARGET = [0.45, 0.40];   // plane coordinates (x, z) relative to the shoulder axis, metres
export const SAFEGUARD_DELTA = 1e-3;          // smallest eigenvalue allowed for H + tau I
export const PREF_W = 0.04;                   // default weight of the elbow-height preference, metres

// =====================================================================================================================
// 6. Solvers (Lecture 4 p43-64)
// =====================================================================================================================
// Every single-start solver returns
//   {method, x: [x_0 .. x_K], f: [..], gNorm: [..], alpha: [K step sizes], converged, nIter: K, status, extra}
// x[k] is the iterate before step k; status is 'ok', 'max_iter', 'singular' or 'not_finite'.

function finish(method, xs, fs, gs, al, tol, status, extra = {}) {
  const last = gs[gs.length - 1], converged = Number.isFinite(last) && last <= tol;
  if (status === 'ok' && !converged) status = 'max_iter';
  return { method, x: xs, f: fs, gNorm: gs, alpha: al, converged, nIter: xs.length - 1, status, extra };
}

/** x_{k+1} = x_k - alpha g_k with a fixed step. Stops when |g| <= tol. */
export function gradientDescent(f, grad, x0, alpha, { maxIter = 1000, tol = 1e-9 } = {}) {
  let x = x0.slice(), status = 'ok';
  const xs = [x.slice()], fs = [f(x)], gs = [], al = [];
  for (let it = 0; it <= maxIter; it++) {
    const g = grad(x);
    gs.push(norm(g));
    if (!Number.isFinite(gs[gs.length - 1])) { status = 'not_finite'; break; }
    if (gs[gs.length - 1] <= tol || xs.length > maxIter) break;
    x = axpy(x, -alpha, g);
    xs.push(x.slice()); fs.push(f(x)); al.push(alpha);
  }
  return finish('gradient descent', xs, fs, gs, al, tol, status);
}

/** Backtracking (p46): shrink alpha until f(x + alpha p) <= f(x) + c alpha g.p. Returns {alpha, n}. */
export function armijo(f, x, fx, g, p, { alpha0 = 1.0, c = 1e-4, shrink = 0.5, maxBacktracks = 60 } = {}) {
  const slope = dot(g, p);
  if (slope >= 0) throw new Error('p is not a descent direction (g.p >= 0)');
  let a = alpha0;
  for (let n = 0; n <= maxBacktracks; n++) {
    if (f(axpy(x, a, p)) <= fx + c * a * slope) return { alpha: a, n };
    a *= shrink;
  }
  return { alpha: a, n: maxBacktracks };
}

/** Gradient descent with Armijo backtracking: each step starts from the trial step alpha0. */
export function gradientDescentArmijo(f, grad, x0, { alpha0 = 1.0, c = 1e-4, shrink = 0.5, maxIter = 1000, tol = 1e-9 } = {}) {
  let x = x0.slice();
  const xs = [x.slice()], fs = [f(x)], gs = [], al = [], bt = [];
  for (let it = 0; it <= maxIter; it++) {
    const g = grad(x);
    gs.push(norm(g));
    if (gs[gs.length - 1] <= tol || xs.length > maxIter) break;
    const r = armijo(f, x, fs[fs.length - 1], g, scale(g, -1), { alpha0, c, shrink });
    x = axpy(x, -r.alpha, g);
    xs.push(x.slice()); fs.push(f(x)); al.push(r.alpha); bt.push(r.n);
  }
  return finish('gradient descent + Armijo', xs, fs, gs, al, tol, 'ok', { backtracks: bt });
}

/** Heavy-ball momentum (p54): x_{k+1} = x_k - alpha g_k + beta (x_k - x_{k-1}), x_{-1} = x_0. */
export function heavyBall(f, grad, x0, alpha, beta, { maxIter = 1000, tol = 1e-9 } = {}) {
  let x = x0.slice(), xPrev = x0.slice(), status = 'ok';
  const xs = [x.slice()], fs = [f(x)], gs = [], al = [];
  for (let it = 0; it <= maxIter; it++) {
    const g = grad(x);
    gs.push(norm(g));
    if (!Number.isFinite(gs[gs.length - 1])) { status = 'not_finite'; break; }
    if (gs[gs.length - 1] <= tol || xs.length > maxIter) break;
    const xn = x.map((v, i) => v - alpha * g[i] + beta * (v - xPrev[i]));
    xPrev = x; x = xn;
    xs.push(x.slice()); fs.push(f(x)); al.push(alpha);
  }
  return finish('heavy ball', xs, fs, gs, al, tol, status);
}

/** Polyak's parameters for a quadratic with curvatures in [mu, L]: {alpha, beta, rate}. */
export function heavyBallOptimal(mu, L) {
  const a = Math.sqrt(L), b = Math.sqrt(mu);
  return { alpha: 4 / (a + b) ** 2, beta: ((a - b) / (a + b)) ** 2, rate: (a - b) / (a + b) };
}

/** Pure Newton: solve H p = -g, x <- x + p. It goes to whatever stationary point its quadratic model points at. */
export function newton(f, grad, hess, x0, { maxIter = 100, tol = 1e-9 } = {}) {
  let x = x0.slice(), status = 'ok';
  const xs = [x.slice()], fs = [f(x)], gs = [], al = [];
  for (let it = 0; it <= maxIter; it++) {
    const g = grad(x);
    gs.push(norm(g));
    if (!Number.isFinite(gs[gs.length - 1])) { status = 'not_finite'; break; }
    if (gs[gs.length - 1] <= tol || xs.length > maxIter) break;
    let p;
    try { p = solve(hess(x), scale(g, -1)); } catch (e) { status = 'singular'; break; }
    x = add(x, p);
    xs.push(x.slice()); fs.push(f(x)); al.push(1.0);
  }
  return finish('Newton', xs, fs, gs, al, tol, status);
}

/** tau >= 0 that makes H + tau I sufficiently positive definite (p64).
 *  'floor':  tau = max(0, delta - lam_min);  'mirror' (default): tau = max(0, delta - lam_min, -2 lam_min). */
export function curvatureShift(H, delta, shift = 'mirror') {
  const lam = H.length === 2 ? eig2(H)[0] : eigSym(H).values[0];
  const tau = Math.max(0, delta - lam);
  return shift === 'mirror' ? Math.max(tau, -2 * lam) : tau;
}

/** Safeguarded Newton (p64): (H + tau I) p = -g, then Armijo backtracking from alpha = 1.
 *  extra.tau and extra.backtracks hold the shift and the number of step halvings per iteration. */
export function safeguardedNewton(f, grad, hess, x0, { delta = 1e-3, shift = 'mirror', c = 1e-4, shrink = 0.5, maxIter = 200, tol = 1e-9 } = {}) {
  let x = x0.slice();
  const n = x.length, xs = [x.slice()], fs = [f(x)], gs = [], al = [], taus = [], bt = [];
  for (let it = 0; it <= maxIter; it++) {
    const g = grad(x);
    gs.push(norm(g));
    if (gs[gs.length - 1] <= tol || xs.length > maxIter) break;
    const H = hess(x), tau = curvatureShift(H, delta, shift);
    const Ht = H.map((r, i) => r.map((v, j) => v + (i === j ? tau : 0)));
    const p = solve(Ht, scale(g, -1));
    const r = armijo(f, x, fs[fs.length - 1], g, p, { alpha0: 1.0, c, shrink });
    x = axpy(x, r.alpha, p);
    xs.push(x.slice()); fs.push(f(x)); al.push(r.alpha); taus.push(tau); bt.push(r.n);
  }
  return finish('safeguarded Newton', xs, fs, gs, al, tol, 'ok', { tau: taus, backtracks: bt });
}

/** Lecture 4 p55-56: f = 1/2 (x1^2 + 9 x2^2), x0 = (1, 1). */
export function lectureQuadratic() {
  const D = [1, 9];
  return { f: (x) => 0.5 * (D[0] * x[0] * x[0] + D[1] * x[1] * x[1]), grad: (x) => [D[0] * x[0], D[1] * x[1]],
    hess: () => [[D[0], 0], [0, D[1]]], x0: [1, 1], eigenvalues: D };
}
/** Lecture 4 p60: f = 1/4 (x1 - 1)^2 + (x2 - 1)^2, x0 = (0, 0), minimiser (1, 1). */
export function lectureNewtonExample() {
  const D = [0.5, 2], xs = [1, 1];
  return { f: (x) => 0.5 * (D[0] * (x[0] - xs[0]) ** 2 + D[1] * (x[1] - xs[1]) ** 2),
    grad: (x) => [D[0] * (x[0] - xs[0]), D[1] * (x[1] - xs[1])], hess: () => [[D[0], 0], [0, D[1]]], x0: [0, 0], xStar: xs };
}
/** Per-step multipliers 1 - alpha lambda_i of the eigen-components under fixed-step gradient descent. */
export const quadraticGDFactors = (eigenvalues, alpha) => eigenvalues.map((l) => 1 - alpha * l);

// ---- the two-joint problem, many starts (basin maps) ------------------------------------------------------------
export const TWO_METHODS = ['gd', 'gd_armijo', 'heavy_ball', 'newton', 'safeguarded_newton'];

/** One run of one method on the two-joint reach problem without storing the path (solvers.batch_solve, one start).
 *  Returns {q (unwrapped), iters, converged, failed (non-finite iterate), leftLimits, maxStep}. */
export function solveTwo(method, two, t, q0, { alpha = null, beta = 0, delta = 1e-3, shift = 'mirror', alpha0 = 1.0, c = 1e-4,
  shrink = 0.5, maxIter = 3000, tol = 1e-9 } = {}) {
  let x = [q0[0], q0[1]], xp = [q0[0], q0[1]], iters = 0, converged = false, failed = false, maxStep = 0;
  let left = !two.withinLimits(x);
  const lineSearch = (g, p, a0) => {
    const f0 = two.cost(x, t), slope = g[0] * p[0] + g[1] * p[1];
    let a = a0;
    for (let n = 0; n <= 60; n++) {
      if (two.cost([x[0] + a * p[0], x[1] + a * p[1]], t) <= f0 + c * a * slope) break;
      a *= shrink;
    }
    return [a * p[0], a * p[1]];
  };
  for (let k = 0; k <= maxIter; k++) {
    const g = two.grad(x, t), gn = Math.sqrt(g[0] * g[0] + g[1] * g[1]);
    iters = k;
    if (!Number.isFinite(gn)) { failed = true; break; }
    if (gn <= tol) { converged = true; break; }
    if (k === maxIter) break;
    let step;
    if (method === 'gd') step = [-alpha * g[0], -alpha * g[1]];
    else if (method === 'heavy_ball') step = [-alpha * g[0] + beta * (x[0] - xp[0]), -alpha * g[1] + beta * (x[1] - xp[1])];
    else if (method === 'gd_armijo') step = lineSearch(g, [-g[0], -g[1]], alpha0);
    else {
      const H = two.hess(x, t);
      let a = H[0][0], d = H[1][1];
      const b = H[0][1];
      if (method === 'safeguarded_newton') {
        const lamMin = 0.5 * (a + d) - Math.sqrt((0.5 * (a - d)) ** 2 + b * b);
        let tau = Math.max(0, delta - lamMin);
        if (shift === 'mirror') tau = Math.max(tau, -2 * lamMin);
        a += tau; d += tau;
      }
      const dt = a * d - b * b;
      const p = [-(d * g[0] - b * g[1]) / dt, -(-b * g[0] + a * g[1]) / dt];
      step = method === 'safeguarded_newton' ? lineSearch(g, p, 1.0) : p;
    }
    xp = x;
    x = [x[0] + step[0], x[1] + step[1]];
    const sn = Math.sqrt(step[0] * step[0] + step[1] * step[1]);
    if (sn > maxStep) maxStep = sn;
    if (!two.withinLimits(x)) left = true;
  }
  return { q: x, iters, converged, failed, leftLimits: left, maxStep };
}

/** Index of the stationary point a run ended at (angles compared on the torus); -1 if none or not converged. */
export function classify(q, converged, stationary, tol = 1e-3) {
  if (!converged) return -1;
  for (let i = 0; i < stationary.length; i++) {
    const d0 = wrap(q[0] - stationary[i].q[0]), d1 = wrap(q[1] - stationary[i].q[1]);
    if (Math.sqrt(d0 * d0 + d1 * d1) < tol) return i;
  }
  return -1;
}

/** Starting angles along one axis: the n cell centres of [-pi, pi) (centred) or -pi + 2 pi i / n. */
export const startGrid = (n, centred = true) => Array.from({ length: n }, (_, i) => -PI + (i + (centred ? 0.5 : 0)) * 2 * PI / n);

export const TYPE_CODE = { min: 0, saddle: 1, max: 2, degenerate: 3 };

/** Basin map of one method: n x n grid of starts (q1[i] = shoulder_lift, q2[j] = elbow), row-major typed arrays.
 *  label: index into `stationary`, -1 not converged, -2 non-finite; kind: 0 min, 1 saddle, 2 max, -1 none. */
export function basinMap(method, two, t, n = 128, opts = {}) {
  const stationary = opts.stationary || two.stationaryPoints(t), g = startGrid(n, opts.centred !== false);
  const label = new Int8Array(n * n), kind = new Int8Array(n * n), iters = new Int32Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const r = solveTwo(method, two, t, [g[i], g[j]], opts), k = i * n + j;
      const l = r.failed ? -2 : classify(r.q, r.converged, stationary);
      label[k] = l; kind[k] = l >= 0 ? TYPE_CODE[stationary[l].type] : -1; iters[k] = r.iters;
    }
  }
  return { q1: g, q2: g, n, label, kind, iters, stationary };
}

// =====================================================================================================================
// 7. Inverse kinematics on the six-joint arm (Lecture 4 p64-66)
// =====================================================================================================================
// Each solver returns {method, q: [q_0 .. q_K], err, errPos, errRot, sigmaMin (K+1 values each), dqNorm, dqMax
// (K values), converged (|e| <= tol), stalled (|dq| <= tolStep with |e| > tol), nIter: K, extra}.

function ikLog(e, J, hasRot, rotWeight, L) {
  L.err.push(norm(e));
  L.errPos.push(norm(e.slice(0, 3)));
  L.errRot.push(hasRot ? norm(e.slice(3)) / rotWeight : 0);
  L.sigmaMin.push(Math.min(...singularValues(J)));
}

function ikLoop(method, q0, pt, Rt, rotWeight, stepFn, maxIter, tol, tolStep, clip) {
  let q = q0.slice(), stalled = false;
  const L = { q: [q.slice()], err: [], errPos: [], errRot: [], sigmaMin: [], dqNorm: [], dqMax: [] };
  for (let it = 0; it <= maxIter; it++) {
    const { e, J } = poseError(q, pt, Rt, rotWeight);
    ikLog(e, J, !!Rt, rotWeight, L);
    if (L.err[L.err.length - 1] <= tol || L.q.length > maxIter) break;
    let qn = add(q, stepFn(e, J));
    if (clip) qn = qn.map((v, i) => Math.min(Math.max(v, UR5E.lower[i]), UR5E.upper[i]));
    if (!isFiniteVec(qn)) break;
    const step = sub(qn, q);
    q = qn;
    L.q.push(q.slice()); L.dqNorm.push(norm(step)); L.dqMax.push(normInf(step));
    if (L.dqNorm[L.dqNorm.length - 1] <= tolStep) {
      const r = poseError(q, pt, Rt, rotWeight);
      ikLog(r.e, r.J, !!Rt, rotWeight, L);
      stalled = L.err[L.err.length - 1] > tol;
      break;
    }
  }
  return { method, ...L, converged: L.err[L.err.length - 1] <= tol, stalled, nIter: L.q.length - 1, extra: {} };
}

/** Jacobian-transpose IK = gradient descent on 1/2 |e|^2: dq = alpha J^T e.
 *  alpha = 'auto' uses alpha = |J^T e|^2 / |J J^T e|^2; a number gives a fixed step. */
export function ikJacobianTranspose(q0, pt, Rt = null, { alpha = 'auto', rotWeight = 0.3, maxIter = 2000, tol = 1e-6, tolStep = 1e-10, clip = false } = {}) {
  const step = (e, J) => {
    const g = matTvec(J, e);
    let a = alpha;
    if (alpha === 'auto') { const Jg = matvec(J, g), den = dot(Jg, Jg); a = den > 0 ? dot(g, g) / den : 0; }
    return scale(g, a);
  };
  return ikLoop('Jacobian transpose', q0, pt, Rt, rotWeight, step, maxIter, tol, tolStep, clip);
}

/** Gauss-Newton / pseudo-inverse IK: dq = step * pinv(J) e. */
export function ikGaussNewton(q0, pt, Rt = null, { step = 1.0, rotWeight = 0.3, maxIter = 200, tol = 1e-6, tolStep = 1e-10, clip = false } = {}) {
  return ikLoop('Gauss-Newton (pseudoinverse)', q0, pt, Rt, rotWeight, (e, J) => scale(lstsq(J, e), step), maxIter, tol, tolStep, clip);
}

/** The damped step for one (J, e): J^T (J J^T + damping^2 I)^-1 e. */
export function dlsDirection(J, e, damping) {
  const JJt = matmul(J, transpose(J));
  for (let i = 0; i < JJt.length; i++) JJt[i][i] += damping * damping;
  return matTvec(J, solve(JJt, e));
}

/** Damped least squares with a fixed damping. Finding of the lab: at an unreachable target a fixed damping settles
 *  only if it is above sqrt(h / 2), h = curvature of the cost along the lost direction. */
export function ikDampedLeastSquares(q0, pt, Rt = null, { damping = 0.05, rotWeight = 0.3, maxIter = 500, tol = 1e-6, tolStep = 1e-10, clip = false } = {}) {
  return ikLoop('damped least squares', q0, pt, Rt, rotWeight, (e, J) => dlsDirection(J, e, damping), maxIter, tol, tolStep, clip);
}

/** Damped least squares with adaptive damping (Levenberg-Marquardt): a step is accepted only if it reduces |e|;
 *  then the damping is divided by `factor`, otherwise multiplied by it. extra.damping, extra.rejected. */
export function ikLevenbergMarquardt(q0, pt, Rt = null, { damping0 = 0.05, factor = 3.0, dampingMin = 1e-4, rotWeight = 0.3,
  maxIter = 500, tol = 1e-6, tolStep = 1e-10, clip = false } = {}) {
  let q = q0.slice(), lam = damping0, rejected = 0, stalled = false;
  let { e, J } = poseError(q, pt, Rt, rotWeight);
  const L = { q: [q.slice()], err: [], errPos: [], errRot: [], sigmaMin: [], dqNorm: [], dqMax: [] }, lams = [];
  ikLog(e, J, !!Rt, rotWeight, L);
  while (L.err[L.err.length - 1] > tol && L.q.length <= maxIter) {
    let qn = add(q, dlsDirection(J, e, lam));
    if (clip) qn = qn.map((v, i) => Math.min(Math.max(v, UR5E.lower[i]), UR5E.upper[i]));
    const step = sub(qn, q);
    if (norm(step) <= tolStep || lam > 1e8) { stalled = true; break; }
    const r = poseError(qn, pt, Rt, rotWeight);
    if (norm(r.e) < L.err[L.err.length - 1]) {
      L.dqNorm.push(norm(step)); L.dqMax.push(normInf(step)); lams.push(lam);
      q = qn; e = r.e; J = r.J;
      L.q.push(q.slice()); ikLog(e, J, !!Rt, rotWeight, L);
      lam = Math.max(lam / factor, dampingMin);
    } else { rejected += 1; lam *= factor; }
  }
  return { method: 'Levenberg-Marquardt', ...L, converged: L.err[L.err.length - 1] <= tol, stalled, nIter: L.q.length - 1,
    extra: { damping: lams, rejected } };
}

// =====================================================================================================================
// 8. Gradient descent with Gaussian noise, Langevin steps (Lecture 4 p85-99)
// =====================================================================================================================

/** Stationary variance of x+ = x - alpha (lam x + sigma_g xi) (p89, p99): alpha sigma_g^2 / (lam (2 - alpha lam)). */
export const stationaryVariance = (alpha, sigmaG, lam) => alpha * sigmaG * sigmaG / (lam * (2 - alpha * lam));
/** The same with sigma_g^2 = 2 T / alpha (p93-94): 2 T / (lam (2 - alpha lam)), which tends to T / lam. */
export const langevinVariance = (alpha, T, lam) => 2 * T / (lam * (2 - alpha * lam));

/** Temperature per step: 'geometric' (constant ratio), 'linear', or 'constant' (Tlo throughout). */
export function schedule(kind, Thi, Tlo, K) {
  return Array.from({ length: K }, (_, k) => {
    const x = k / Math.max(K - 1, 1);
    if (kind === 'geometric') return Thi * (Tlo / Thi) ** x;
    if (kind === 'linear') return Thi + (Tlo - Thi) * x;
    if (kind === 'constant') return Tlo;
    throw new Error(kind);
  });
}

const perStep = (v, k) => (typeof v === 'function' ? v(k) : Array.isArray(v) ? v[k] : v);

/** Displacement noise amplitude of step k: alpha sigma_g, or sqrt(2 T alpha) with Langevin scaling (T given). */
export function noiseAmplitude(alpha, sigmaG, T = null) {
  return T !== null && T !== undefined ? Math.sqrt(2 * T * alpha) : alpha * sigmaG;
}

/** One noisy step in place: x <- x - alpha g - amp xi. */
export function noisyStep(x, g, alpha, amp, xi) {
  for (let i = 0; i < x.length; i++) { x[i] -= alpha * g[i]; if (amp !== 0) x[i] -= amp * xi[i]; }
  return x;
}

/** Gradient descent with Gaussian gradient noise for many runs at once (experiments.noisy_gd):
 *      x_{k+1} = x_k - alpha_k (grad f(x_k) + sigma_{g,k} xi_k),  xi_k ~ N(0, I) fresh for every run and step.
 *  Q0: array of N start points. alpha, sigmaG, T: a number, an array of length `steps`, or a function of k.
 *  With T the noise has Langevin scaling sigma_g^2 = 2 T / alpha. Noise source: opts.rng (an RNG), or opts.xi,
 *  a function (k, run) -> noise vector (used by the tests to replay numpy's draws). callback(k, Q) after step k
 *  (1-based) every callbackEvery steps. Returns {Q, traj, kRec}. */
export function noisyGD(grad, Q0, alpha, { sigmaG = 0, steps = 1000, T = null, rng = null, seed = 0, xi = null,
  recordEvery = 0, callback = null, callbackEvery = 1 } = {}) {
  const Q = Q0.map((q) => q.slice()), n = Q[0].length, N = Q.length;
  const gen = xi ? null : (rng || new RNG(seed));
  const traj = recordEvery ? [Q.map((q) => q.slice())] : null, kRec = recordEvery ? [0] : null;
  const z = new Array(n);
  for (let k = 0; k < steps; k++) {
    const a = perStep(alpha, k), Tk = T === null ? null : perStep(T, k);
    const amp = noiseAmplitude(a, perStep(sigmaG, k), Tk);
    for (let r = 0; r < N; r++) {
      const q = Q[r], g = grad(q);
      let noise = null;
      if (amp !== 0) {
        if (xi) noise = xi(k, r);
        else { for (let i = 0; i < n; i++) z[i] = gen.normal(); noise = z; }
      }
      for (let i = 0; i < n; i++) { q[i] -= a * g[i]; if (noise) q[i] -= amp * noise[i]; }
    }
    if (recordEvery && (k + 1) % recordEvery === 0) { traj.push(Q.map((q) => q.slice())); kRec.push(k + 1); }
    if (callback && (k + 1) % callbackEvery === 0) callback(k + 1, Q);
  }
  return { Q, traj, kRec };
}

/** Distance on the torus of joint angles. */
export const torusDist = (q, p) => Math.hypot(...q.map((v, i) => wrap(v - p[i])));

// =====================================================================================================================
// 9. l1 and sparsity (Lecture 4 p76-83, p118-122)
// =====================================================================================================================

/** S_tau(v): proximal map of tau |x| (p80). Above tau subtract tau, below -tau add tau, else exactly 0. */
export function softThreshold(v, tau) {
  if (Array.isArray(v)) return v.map((x) => softThreshold(x, tau));
  const a = Math.abs(v) - tau;
  return a > 0 ? Math.sign(v) * a : 0;
}

/** x_{k+1} = prox_{alpha r}(x_k - alpha grad h(x_k)) (p81). proxR(v, alpha) is the proximal map.
 *  Stops when the gradient mapping |x_k - x_{k+1}| / alpha <= tol. Returns {x, iters, xs}. */
export function proxGradient(gradH, proxR, x0, alpha, { iters = 1000, tol = 0, record = false } = {}) {
  let x = x0.slice(), k = 0;
  const xs = record ? [x.slice()] : null;
  for (k = 1; k <= iters; k++) {
    const xn = proxR(axpy(x, -alpha, gradH(x)), alpha), gm = norm(sub(xn, x)) / alpha;
    x = xn;
    if (record) xs.push(x.slice());
    if (gm <= tol) break;
  }
  return { x, iters: Math.min(k, iters), xs };
}

export function lassoObjective(A, b, lam, x) {
  const r = sub(matvec(A, x), b);
  let l1 = 0;
  for (const v of x) l1 += Math.abs(v);
  return 0.5 * dot(r, r) + lam * l1;
}

/** ISTA for F(x) = 1/2 |A x - b|^2 + lam |x|_1 with step 1/L, L = |A|_2^2 (p82, p121); accelerated = FISTA.
 *  Returns {x, iters, alpha, L, objective, xs}. */
export function ista(A, b, lam, { x0 = null, alpha = null, iters = 5000, tol = 1e-10, accelerated = false, record = false, L = null } = {}) {
  if (L === null) L = norm2(A) ** 2;
  if (alpha === null) alpha = 1 / L;
  let x = x0 ? x0.slice() : zeros(A[0].length), y = x.slice(), tk = 1, k = 0;
  const obj = [lassoObjective(A, b, lam, x)], xs = record ? [x.slice()] : null;
  for (k = 1; k <= iters; k++) {
    const base = accelerated ? y : x;
    const xn = softThreshold(axpy(base, -alpha, matTvec(A, sub(matvec(A, base), b))), alpha * lam);
    const gm = norm(sub(xn, base)) / alpha;
    if (accelerated) {
      const tn = 0.5 * (1 + Math.sqrt(1 + 4 * tk * tk));
      y = axpy(xn, (tk - 1) / tn, sub(xn, x));
      tk = tn;
    }
    x = xn;
    obj.push(lassoObjective(A, b, lam, x));
    if (record) xs.push(x.slice());
    if (gm <= tol) break;
  }
  return { x, iters: Math.min(k, iters), alpha, L, objective: obj, xs };
}

/** Minimum-l2-norm solution of A x = b (pseudo-inverse). */
export const minL2Norm = (A, b) => lstsq(A, b);
/** Joint displacement with the smallest l2 norm that produces the tool displacement dp to first order. */
export const minL2Step = (J, dp) => lstsq(J, dp);

/** Joint displacement with the smallest l1 norm that produces dp to first order: min |d|_1 s.t. J d = dp.
 *  The optimum is a vertex: at most m joints move. All subsets of m usable columns are tried with an exact
 *  m x m solve and the cheapest is returned; resting joints are exactly 0. Returns null if J has no
 *  well-conditioned m x m block (the Python core then calls a linear-programming solver). */
export function minL1Step(J, dp) {
  const m = J.length, n = J[0].length;
  if (dp.every((v) => v === 0)) return zeros(n);
  const norms = Array.from({ length: n }, (_, j) => norm(column(J, j))), nmax = Math.max(...norms, 1e-300);
  const cols = [];
  for (let j = 0; j < n; j++) if (norms[j] > 1e-12 * nmax) cols.push(j);
  if (cols.length < m) return null;
  let best = null, bestCost = Infinity, bestSub = null;
  const sub_ = [];
  const rec = (start) => {
    if (sub_.length === m) {
      const Bm = J.map((r) => sub_.map((j) => r[j])), s = singularValues(Bm);
      if (!(s[m - 1] > 0) || s[0] / s[m - 1] >= 1e10) return;
      const x = solve(Bm, dp);
      let c = 0;
      for (const v of x) c += Math.abs(v);
      if (c < bestCost) { bestCost = c; best = x; bestSub = sub_.slice(); }
      return;
    }
    for (let i = start; i < cols.length; i++) { sub_.push(cols[i]); rec(i + 1); sub_.pop(); }
  };
  rec(0);
  if (!best) return null;
  const xm = normInf(best), d = zeros(n);
  bestSub.forEach((j, i) => { d[j] = Math.abs(best[i]) <= 1e-13 * xm ? 0 : best[i]; });
  return d;
}

/** Smallest lambda for which d = 0 is stationary for the l1-penalised reach: |J(q0)^T (p0 - t)|_inf. */
export function reachLambdaMax(fkJacFn, q0, t) {
  const { p, J } = fkJacFn(q0);
  return normInf(matTvec(J, sub(p, t)));
}

/** Stationarity residual (max norm) at displacement d. l1: |d - S_lam(d - g)|_inf, g = J^T (p - t); l2: |g + lam d|_inf. */
export function reachKKT(fkJacFn, q0, t, lam, d, kind = 'l1') {
  const { p, J } = fkJacFn(add(q0, d)), g = matTvec(J, sub(p, t));
  if (kind === 'l1') return normInf(d.map((v, i) => v - softThreshold(v - g[i], lam)));
  return normInf(g.map((v, i) => v + lam * d[i]));
}

function supportHessian(fkJacFn, q0, t, d, S, eps = 1e-6) {
  const F = (x) => {
    const dd = zeros(d.length);
    S.forEach((j, i) => { dd[j] = x[i]; });
    const { p, J } = fkJacFn(add(q0, dd)), g = matTvec(J, sub(p, t));
    return S.map((j) => g[j]);
  };
  const k = S.length, H = zerosM(k, k), x0 = S.map((j) => d[j]);
  for (let j = 0; j < k; j++) {
    const xp = x0.slice(), xm = x0.slice();
    xp[j] += eps; xm[j] -= eps;
    const fp = F(xp), fm = F(xm);
    for (let i = 0; i < k; i++) H[i][j] = (fp[i] - fm[i]) / (2 * eps);
  }
  return sym(H);
}

// Newton finish on a guessed support (structure._reach_polish; the Python core uses MINPACK's hybrid method
// for the same equations, so the two agree at the solution but not along the way).
function reachPolish(fkJacFn, q0, t, lam, d, kind) {
  const n = d.length;
  const cost = (dd) => {
    const { p } = fkJacFn(add(q0, dd)), r = sub(p, t);
    let pen = 0;
    for (const v of dd) pen += kind === 'l1' ? lam * Math.abs(v) : 0.5 * lam * v * v;
    return 0.5 * dot(r, r) + pen;
  };
  const attempt = (S0, sgnAll, xAll) => {
    let S = S0.slice(), dn;
    for (;;) {
      dn = zeros(n);
      if (S.length === 0) break;
      const sgn = S.map((j) => sgnAll[j]);
      const F = (x) => {
        const dd = zeros(n);
        S.forEach((j, i) => { dd[j] = x[i]; });
        const { p, J } = fkJacFn(add(q0, dd)), g = matTvec(J, sub(p, t));
        return S.map((j, i) => g[j] + (kind === 'l1' ? lam * sgn[i] : lam * x[i]));
      };
      let x = S.map((j) => xAll[j]), Fx = F(x), ok = true;
      for (let it = 0; it < 60 && normInf(Fx) > 1e-15; it++) {
        const dd = zeros(n);
        S.forEach((j, i) => { dd[j] = x[i]; });
        const H = supportHessian(fkJacFn, q0, t, dd, S);
        if (kind === 'l2') for (let i = 0; i < S.length; i++) H[i][i] += lam;
        let step;
        try { step = solve(H, scale(Fx, -1)); } catch (e) { ok = false; break; }
        const xn = add(x, step), Fn = F(xn);
        if (!isFiniteVec(Fn)) { ok = false; break; }
        if (normInf(Fn) >= normInf(Fx) && it > 3) break;
        x = xn; Fx = Fn;
      }
      if (!ok) return null;
      S.forEach((j, i) => { dn[j] = x[i]; });
      if (kind === 'l1' && x.some((v, i) => v * sgn[i] <= 0)) {
        let worst = 0;
        for (let i = 1; i < x.length; i++) if (x[i] * sgn[i] < x[worst] * sgn[worst]) worst = i;
        S.splice(worst, 1);
        continue;
      }
      break;
    }
    if (reachKKT(fkJacFn, q0, t, lam, dn, kind) > 1e-10 || cost(dn) > cost(d) + 1e-13) return null;
    if (S.length) {
      const H = supportHessian(fkJacFn, q0, t, dn, S);
      if (eigSym(H).values[0] + (kind === 'l2' ? lam : 0) <= 0) return null;
    }
    return dn;
  };
  const all = Array.from({ length: n }, (_, i) => i);
  if (kind === 'l2') { const dn = attempt(all, d.map(Math.sign), d); return dn ? { d: dn, guess: 1 } : null; }
  let dn = attempt(all.filter((j) => d[j] !== 0), d.map(Math.sign), d);
  if (dn) return { d: dn, guess: 1 };
  const { p, J } = fkJacFn(add(q0, d));
  const x = ista(J, sub(matvec(J, d), sub(p, t)), lam, { x0: d, iters: 20000, tol: 1e-13, accelerated: true }).x;
  dn = attempt(all.filter((j) => x[j] !== 0), x.map(Math.sign), x);
  return dn ? { d: dn, guess: 2 } : null;
}

/** Penalised reaching by proximal gradient with backtracking (structure.reach_penalised):
 *      kind 'l1':  min_d 1/2 |p(q0 + d) - t|^2 + lam |d|_1         prox = soft threshold (exact zeros)
 *      kind 'l2':  min_d 1/2 |p(q0 + d) - t|^2 + lam/2 |d|_2^2     prox = v / (1 + alpha lam) (no zeros)
 *  fkJacFn(q) -> {p, J} (use fkJacPos for the UR5e). Step 1/L with L = |J|_2^2, doubled until the quadratic upper
 *  bound holds. polish = false (default) is the pure lecture algorithm; polish = true solves the stationarity
 *  equations on the current support every `chunk` iterations and stops as soon as that succeeds.
 *  Returns {d, q, p, miss, nnz, moved, iters, polished, kkt, cost, stoppedAtCap, history}.
 *  A run that stopped at the cap is NOT a solution: its nnz over-counts the moving joints. */
export function reachPenalised(fkJacFn, q0, t, lam, { kind = 'l1', d0 = null, maxIter = 20000, tol = 1e-9, polish = false, chunk = 200, record = false } = {}) {
  let d = d0 ? d0.slice() : zeros(q0.length);
  const prox = kind === 'l1' ? (v, a) => softThreshold(v, a * lam) : (v, a) => v.map((x) => x / (1 + a * lam));
  const h = (dd) => {
    const { p, J } = fkJacFn(add(q0, dd)), r = sub(p, t);
    return { hv: 0.5 * dot(r, r), g: matTvec(J, r), J };
  };
  const hist = record ? [d.slice()] : null;
  let cur = h(d), it = 0, done = false, polished = 0;
  while (it < maxIter && !done) {
    const stop = polish ? Math.min(it + chunk, maxIter) : maxIter;
    while (it < stop) {
      it += 1;
      let L = Math.max(eigSym(matmul(cur.J, transpose(cur.J))).values.pop(), 1e-12), dn, nxt, diff;
      for (let b = 0; b < 60; b++) {
        dn = prox(axpy(d, -1 / L, cur.g), 1 / L);
        nxt = h(dn);
        diff = sub(dn, d);
        if (nxt.hv <= cur.hv + dot(cur.g, diff) + 0.5 * L * dot(diff, diff) + 1e-12 * Math.abs(cur.hv) + 1e-20) break;
        L *= 2;
      }
      const gm = L * norm(diff);
      d = dn; cur = nxt;
      if (record) hist.push(d.slice());
      if (gm <= tol) { done = true; break; }
    }
    if (polish) {
      const r = reachPolish(fkJacFn, q0, t, lam, d, kind);
      if (r) { d = r.d; cur = h(d); done = true; polished = r.guess; if (record) hist.push(d.slice()); }
    }
  }
  const q = add(q0, d), p = fkJacFn(q).p;
  let pen = 0;
  for (const v of d) pen += kind === 'l1' ? lam * Math.abs(v) : 0.5 * lam * v * v;
  const moved = d.map((v) => v !== 0);
  return { d, q, p, miss: norm(sub(p, t)), nnz: moved.filter(Boolean).length, moved, iters: it, polished,
    kkt: reachKKT(fkJacFn, q0, t, lam, d, kind), cost: cur.hv + pen, stoppedAtCap: !done, history: hist };
}

/** Regularisation path (structure.reach_path): solve for every lambda, from the largest down, each warm-started at
 *  the previous solution, so the path follows one branch continuously from d = 0. The finishing solve is on by
 *  default here. Returns {lams, D, miss, nnz, kkt} in the order of `lams`. */
export function reachPath(fkJacFn, q0, t, lams, opts = {}) {
  const order = lams.map((_, i) => i).sort((a, b) => lams[b] - lams[a]), m = lams.length;
  const out = { lams: lams.slice(), D: new Array(m), miss: new Array(m), nnz: new Array(m), kkt: new Array(m) };
  let d = zeros(q0.length);
  for (const i of order) {
    const r = reachPenalised(fkJacFn, q0, t, lams[i], { polish: true, ...opts, d0: d });
    d = r.d;
    out.D[i] = d; out.miss[i] = r.miss; out.nnz[i] = r.nnz; out.kkt[i] = r.kkt;
  }
  return out;
}

/** Compressed-sensing instance (p118-122) drawn with this module's generator: Gaussian Phi (m x n, entries
 *  N(0, 1/m)), k-sparse signal with magnitudes uniform in `amp` and random signs. Returns {Phi, s, y}. */
export function csInstance(rng, n = 128, m = 48, k = 6, amp = [0.8, 2.0]) {
  const Phi = Array.from({ length: m }, () => Array.from({ length: n }, () => rng.normal() / Math.sqrt(m)));
  const s = zeros(n);
  for (const j of rng.choice(n, k)) s[j] = rng.range(amp[0], amp[1]) * (rng.uniform() < 0.5 ? -1 : 1);
  return { Phi, s, y: matvec(Phi, s) };
}

// =====================================================================================================================
// 10. Calculus of variations: a discretised path (Lecture 4 p100-113)
// =====================================================================================================================

/** Smooth obstacle potential around a vertical pillar: U(y) = w exp(-rho^2 / (2 sigma^2)), rho = horizontal
 *  distance from the pillar's axis (first two coordinates). It is a potential, not a constraint. */
export class Pillar {
  constructor(center, sigma, weight) {
    this.c = [center[0], center[1]]; this.s = sigma; this.w = weight;
    this.curvBound = weight / (sigma * sigma);
  }

  U(y) { const dx = y[0] - this.c[0], dy = y[1] - this.c[1]; return this.w * Math.exp(-(dx * dx + dy * dy) / (2 * this.s * this.s)); }

  grad(y) {
    const dx = y[0] - this.c[0], dy = y[1] - this.c[1], e = this.U(y), g = zeros(y.length), s2 = this.s * this.s;
    g[0] = -e * dx / s2; g[1] = -e * dy / s2;
    return g;
  }

  hess(y) {
    const D = [y[0] - this.c[0], y[1] - this.c[1]], e = this.U(y), H = zerosM(y.length, y.length), s2 = this.s * this.s;
    for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) H[a][b] = e * (D[a] * D[b] / (s2 * s2) - (a === b ? 1 : 0) / s2);
    return H;
  }

  distance(y) { return Math.hypot(y[0] - this.c[0], y[1] - this.c[1]); }
}

/** U(y) = c/2 |y|^2 (the p113 exercise has c = 1). */
export class QuadraticPotential {
  constructor(c = 1) { this.c = c; this.curvBound = Math.abs(c); }
  U(y) { return 0.5 * this.c * dot(y, y); }
  grad(y) { return scale(y, this.c); }
  hess(y) { return scaleM(eye(y.length), this.c); }
}

/** A path y(t), 0 <= t <= T, from A to B, discretised with M segments (nodes y_0 .. y_M, h = T/M):
 *      J_h(Y) = sum_i |y_{i+1} - y_i|^2 / (2h) + sum_i w_i U(y_i),  w_i = h (h/2 at the two ends).
 *  Fixed end: the unknowns are the M-1 interior nodes. Free end (shelf = [B0, tau]): the last node is
 *  y_M = B0 + s tau and s is one more unknown (shelfBounds clips s; only gradient descent honours the bounds).
 *  Euler-Lagrange: y'' = grad U(y); natural condition at a free end: y'(T) . tau = 0.
 *  Paths Y are arrays of M+1 points; z is the vector of free variables. */
export class PathProblem {
  constructor(A, B, M, { potential = null, T = 1.0, shelf = null, shelfBounds = null } = {}) {
    this.A = A.slice(); this.d = A.length; this.M = M; this.T = T; this.h = T / M; this.pot = potential;
    this.shelf = null; this.shelfBounds = shelfBounds;
    if (shelf) { const nt = norm(shelf[1]); this.shelf = [shelf[0].slice(), scale(shelf[1], 1 / nt)]; this.B = shelf[0].slice(); } else this.B = B.slice();
    this.nInt = (M - 1) * this.d;
    this.nFree = this.nInt + (this.shelf ? 1 : 0);
    this.wts = new Array(M + 1).fill(this.h);
    this.wts[0] = this.wts[M] = 0.5 * this.h;
    this.t = Array.from({ length: M + 1 }, (_, i) => (i === M ? T : i * (T / M)));
  }

  nodes(z) {
    const Y = [this.A.slice()], d = this.d;
    for (let i = 1; i < this.M; i++) Y.push(z.slice((i - 1) * d, i * d));
    Y.push(this.shelf ? axpy(this.shelf[0], z[z.length - 1], this.shelf[1]) : this.B.slice());
    return Y;
  }

  pack(Y) {
    const z = [];
    for (let i = 1; i < this.M; i++) z.push(...Y[i]);
    if (this.shelf) z.push(dot(sub(Y[this.M], this.shelf[0]), this.shelf[1]));
    return z;
  }

  /** The straight line from A to B (or to the given end point). */
  straight(B = null) {
    const E = B || this.B;
    return this.t.map((ti) => this.A.map((a, k) => a + (ti / this.T) * (E[k] - a)));
  }

  cost(Y) {
    let J = 0;
    for (let i = 0; i < this.M; i++) for (let k = 0; k < this.d; k++) { const v = Y[i + 1][k] - Y[i][k]; J += v * v; }
    J /= 2 * this.h;
    if (this.pot) for (let i = 0; i <= this.M; i++) J += this.wts[i] * this.pot.U(Y[i]);
    return J;
  }

  /** dJ_h/dy_i for every node (row 0 belongs to the fixed start and is not used). */
  gradNodes(Y) {
    const G = zerosM(this.M + 1, this.d);
    for (let i = 0; i < this.M; i++) for (let k = 0; k < this.d; k++) { const v = (Y[i + 1][k] - Y[i][k]) / this.h; G[i][k] -= v; G[i + 1][k] += v; }
    if (this.pot) for (let i = 0; i <= this.M; i++) { const g = this.pot.grad(Y[i]); for (let k = 0; k < this.d; k++) G[i][k] += this.wts[i] * g[k]; }
    return G;
  }

  reduce(G) {
    const g = [];
    for (let i = 1; i < this.M; i++) g.push(...G[i]);
    if (this.shelf) g.push(dot(G[this.M], this.shelf[1]));
    return g;
  }

  grad(Y) { return this.reduce(this.gradNodes(Y)); }

  /** Euler-Lagrange residual y'' - grad U(y) at the interior nodes, (M-1) x d. It equals -dJ_h/dy_i / h. */
  elResidual(Y) {
    const R = [];
    for (let i = 1; i < this.M; i++) {
      const r = Y[i].map((v, k) => (Y[i + 1][k] - 2 * v + Y[i - 1][k]) / (this.h * this.h));
      R.push(this.pot ? sub(r, this.pot.grad(Y[i])) : r);
    }
    return R;
  }

  /** Hessian of J_h in the free variables as a symmetric band: band[k][i] = H[i][i + k], k = 0 .. d. */
  hessBand(Y) {
    const { M, d, h } = this, n = this.nFree, band = Array.from({ length: d + 1 }, () => new Array(n).fill(0));
    for (let i = 1; i < M; i++) {
      const base = (i - 1) * d;
      for (let a = 0; a < d; a++) { band[0][base + a] = 2 / h; if (i + 1 < M) band[d][base + a] = -1 / h; }
      if (this.pot) {
        const Hu = this.pot.hess(Y[i]);
        for (let a = 0; a < d; a++) for (let b = a; b < d; b++) band[b - a][base + a] += this.wts[i] * Hu[a][b];
      }
    }
    if (this.shelf) {
      const tau = this.shelf[1], s = this.nInt;
      let hss = dot(tau, tau) / h;                                 // tau^T (I / h) tau, |tau| = 1
      if (this.pot) hss += this.wts[M] * dot(tau, matvec(this.pot.hess(Y[M]), tau));
      band[0][s] = hss;
      if (M >= 2) for (let a = 0; a < d; a++) band[d - a][s - d + a] = -tau[a] / h;
    }
    return band;
  }

  /** The same Hessian as a dense matrix. */
  hess(Y) {
    const band = this.hessBand(Y), n = this.nFree, H = zerosM(n, n);
    for (let k = 0; k < band.length; k++) for (let i = 0; i + k < n; i++) { H[i][i + k] = band[k][i]; H[i + k][i] = band[k][i]; }
    return H;
  }

  /** Upper bound on the largest Hessian eigenvalue: 4/h from the energy plus h times the curvature of U. */
  lipschitz() { return 4 / this.h + (this.pot ? this.h * this.pot.curvBound : 0); }

  /** One relaxation step on the free variables: z <- z - alpha grad J_h (with the shelf bounds). Returns the new z. */
  gdStep(z, alpha, Y = null) {
    const g = this.reduce(this.gradNodes(Y || this.nodes(z))), zn = z.map((v, i) => v - alpha * g[i]);
    if (this.shelf && this.shelfBounds) zn[zn.length - 1] = Math.min(Math.max(zn[zn.length - 1], this.shelfBounds[0]), this.shelfBounds[1]);
    return zn;
  }

  /** Relaxation z <- z - alpha grad J_h, default alpha = 1/L. snapshots: iteration numbers at which the path is
   *  stored. Stops when max |Euler-Lagrange residual| <= tol or when stop(Y) is true. Returns {Y, iters, alpha, snaps}. */
  gradientDescent(Y0, { alpha = null, iters = 10000, tol = 0, snapshots = null, stop = null } = {}) {
    if (alpha === null) alpha = 1 / this.lipschitz();
    let z = this.pack(Y0), Y = this.nodes(z), k = 0;
    const want = new Set(snapshots || []), snaps = {};
    if (want.has(0)) snaps[0] = copyM(Y);
    for (k = 1; k <= iters; k++) {
      z = this.gdStep(z, alpha, Y);
      Y = this.nodes(z);
      if (want.has(k)) snaps[k] = copyM(Y);
      if (tol > 0) { let m = 0; for (const r of this.elResidual(Y)) m = Math.max(m, normInf(r)); if (m <= tol) break; }
      if (stop && stop(Y)) break;
    }
    return { Y, iters: Math.min(k, iters), alpha, snaps };
  }

  /** Newton's method on J_h. safeguard = true (p64): add tau I until the banded Cholesky factorisation succeeds,
   *  then backtrack (Armijo). safeguard = false: pure Newton, full step (exact in one step for a quadratic cost).
   *  Stops when |grad J_h|_inf <= tol. Returns {Y, iters, gradNorm, shifts, history}. */
  newton(Y0, { maxIter = 100, tol = 1e-10, safeguard = true, record = false } = {}) {
    let z = this.pack(Y0), Y = this.nodes(z), shifts = 0, k = 0, g = this.grad(Y);
    const hist = record ? [copyM(Y)] : null;
    for (k = 1; k <= maxIter; k++) {
      if (!safeguard) z = add(z, solve(this.hess(Y), scale(g, -1)));
      else {
        const band = this.hessBand(Y);
        let tau = 0, Lb;
        for (;;) {
          Lb = bandCholesky(tau === 0 ? band : [band[0].map((v) => v + tau), ...band.slice(1)]);
          if (Lb) break;
          tau = Math.max(2 * tau, 1e-3 * 4 / this.h);
        }
        if (tau > 0) shifts += 1;
        const p = bandCholeskySolve(Lb, scale(g, -1)), J0 = this.cost(Y), slope = dot(g, p);
        let step = 1.0;
        while (-slope > 1e-9 * Math.max(1, Math.abs(J0)) && step > 1e-12 && this.cost(this.nodes(axpy(z, step, p))) > J0 + 1e-4 * step * slope) step *= 0.5;
        z = axpy(z, step, p);
      }
      Y = this.nodes(z);
      if (record) hist.push(copyM(Y));
      g = this.grad(Y);
      if (normInf(g) <= tol) break;
    }
    return { Y, iters: Math.min(k, maxIter), gradNorm: normInf(g), shifts, history: hist };
  }

  /** 1/2 |y'|^2 - U(y) at the segment midpoints; constant along an exact Euler-Lagrange path. */
  firstIntegral(Y) {
    const U = this.pot ? Y.map((y) => this.pot.U(y)) : null, E = [];
    for (let i = 0; i < this.M; i++) {
      let e = 0;
      for (let k = 0; k < this.d; k++) { const v = (Y[i + 1][k] - Y[i][k]) / this.h; e += v * v; }
      E.push(0.5 * e - (U ? 0.5 * (U[i] + U[i + 1]) : 0));
    }
    return E;
  }

  /** y'(T) by a one-sided difference: order 1 uses two nodes, order 2 three nodes. */
  endVelocity(Y, order = 2) {
    const M = this.M;
    if (order === 1) return Y[M].map((v, k) => (v - Y[M - 1][k]) / this.h);
    return Y[M].map((v, k) => (3 * v - 4 * Y[M - 1][k] + Y[M - 2][k]) / (2 * this.h));
  }

  /** Angle between the arrival velocity and the shelf direction, degrees (90 = perpendicular). */
  arrivalAngleDeg(Y, order = 2) {
    const v = this.endVelocity(Y, order), c = dot(v, this.shelf[1]) / norm(v);
    return Math.acos(Math.min(1, Math.max(-1, c))) * 180 / PI;
  }

  minHessianEig(Y) { return eigSym(this.hess(Y)).values[0]; }
}

/** Cholesky factor of a symmetric positive-definite band matrix (band[k][i] = H[i][i + k]).
 *  Returns the factor in the same layout (Lb[k][i] = L[i + k][i]) or null if H is not positive definite. */
export function bandCholesky(band) {
  const u = band.length - 1, n = band[0].length, Lb = Array.from({ length: u + 1 }, () => new Array(n).fill(0));
  for (let j = 0; j < n; j++) {
    let s = band[0][j];
    for (let k = Math.max(0, j - u); k < j; k++) { const l = Lb[j - k][k]; s -= l * l; }
    if (!(s > 0)) return null;
    const dj = Math.sqrt(s);
    Lb[0][j] = dj;
    for (let i = j + 1; i <= Math.min(n - 1, j + u); i++) {
      let t = band[i - j][j];
      for (let k = Math.max(0, i - u); k < j; k++) t -= Lb[i - k][k] * Lb[j - k][k];
      Lb[i - j][j] = t / dj;
    }
  }
  return Lb;
}
export function bandCholeskySolve(Lb, b) {
  const u = Lb.length - 1, n = b.length, y = new Array(n), x = new Array(n);
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = Math.max(0, i - u); k < i; k++) s -= Lb[i - k][k] * y[k]; y[i] = s / Lb[0][i]; }
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k <= Math.min(n - 1, i + u); k++) s -= Lb[k - i][i] * x[k]; x[i] = s / Lb[0][i]; }
  return x;
}

/** Orthonormal eigenvectors of tridiag(-1, 2, -1) of size M-1: column j-1 is sqrt(2/M) sin(i j pi / M). */
export function sineModes(M) {
  return Array.from({ length: M - 1 }, (_, i) => Array.from({ length: M - 1 }, (_, j) => Math.sqrt(2 / M) * Math.sin((i + 1) * (j + 1) * PI / M)));
}
/** Amplitude of each sine mode in a path difference E (interior nodes, (M-1) x d): norm over the coordinates. */
export function modeAmplitudes(E) {
  const S = sineModes(E.length + 1), C = matmul(transpose(S), E);
  return C.map(norm);
}
/** Eigenvalues of the Hessian of sum |y_{i+1} - y_i|^2 / (2h) with fixed ends: (4/h) sin^2(j pi / 2M). */
export const energyEigenvalues = (M, T = 1) => Array.from({ length: M - 1 }, (_, j) => 4 / (T / M) * Math.sin((j + 1) * PI / (2 * M)) ** 2);
/** Condition number of the discretised energy with fixed ends: cot^2(pi / 2M). */
export const energyCondition = (M) => 1 / Math.tan(PI / (2 * M)) ** 2;

// default scene of the path problems (metres, seconds); the pillar is a little off the straight line
export const PATH = {
  A2: [-0.40, 0.0], B2: [0.40, 0.0], A3: [-0.40, 0.0, 0.20], B3: [0.40, 0.0, 0.45],
  pillarCenter: [0.0, 0.03], pillarSigma: 0.08, pillarWeight: 1.5, pillarRadius: 0.05,
  shelfB0: [0.40, 0.0], shelfTau: [0.5, Math.sqrt(3) / 2],
};
export const defaultPillar = (weight = PATH.pillarWeight, center = PATH.pillarCenter) => new Pillar(center, PATH.pillarSigma, weight);

/** 'A curve drawn by hand': the straight line plus one long bend to one side and a fine wiggle. */
export function wigglyStart(prob, { bend = 0.25, wiggle = 0.04, wiggleMode = 9, side = 1.0, B = null } = {}) {
  const Y = prob.straight(B);
  prob.t.forEach((ti, i) => { const s = ti / prob.T; Y[i][1] += side * bend * Math.sin(PI * s) + wiggle * Math.sin(wiggleMode * PI * s); });
  return Y;
}

// =====================================================================================================================
// 11. Optimisation on manifolds: the unit sphere and SO(3) (Lecture 4 p123-129)
// =====================================================================================================================

/** Constraint residual |X^T X - I|_F (X a matrix, or a vector treated as one column). */
export function orthResidual(X) {
  const Xm = Array.isArray(X[0]) ? X : X.map((v) => [v]);
  const G = matmul(transpose(Xm), Xm);
  for (let i = 0; i < G.length; i++) G[i][i] -= 1;
  return frob(G);
}

/** Tangent projection on the sphere: (I - x x^T) g (p125). */
export const sphereProject = (x, g) => axpy(g, -dot(x, g), x);
/** Retraction by normalisation (p126). */
export function sphereRetract(x, v) { const y = add(x, v); return scale(y, 1 / norm(y)); }
/** Exponential map of the sphere: follow the great circle from x in the tangent direction v for the length |v|. */
export function sphereExp(x, v) {
  const nv = norm(v);
  if (nv < 1e-12) return sphereRetract(x, v);
  return x.map((xi, i) => Math.cos(nv) * xi + Math.sin(nv) * v[i] / nv);
}

/** One step for f restricted to the unit sphere; g is the ordinary gradient at x.
 *  'riemannian': normalise(x - alpha (I - x x^T) g); 'project_only': tangent step, no retraction;
 *  'euclidean': x - alpha g (the naive step). */
export function sphereStep(x, g, alpha, mode = 'riemannian') {
  if (mode === 'riemannian') return sphereRetract(x, scale(sphereProject(x, g), -alpha));
  if (mode === 'project_only') return axpy(x, -alpha, sphereProject(x, g));
  if (mode === 'euclidean') return axpy(x, -alpha, g);
  throw new Error(mode);
}
/** Gradient descent on the sphere; returns the iterates (iters + 1 points). */
export function sphereGD(egrad, x0, alpha, iters, mode = 'riemannian') {
  let x = x0.slice();
  const xs = [x.slice()];
  for (let k = 0; k < iters; k++) { x = sphereStep(x, egrad(x), alpha, mode); xs.push(x.slice()); }
  return xs;
}
/** Angle between two vectors of any length (accurate near 0 and near pi). */
export function angleBetween(x, t) {
  const u = scale(x, 1 / norm(x)), v = scale(t, 1 / norm(t));
  return 2 * Math.atan2(norm(sub(u, v)), norm(add(u, v)));
}

/** Tangent projection at a matrix X with orthonormal columns (p125).
 *  'sym': G - X sym(X^T G) (the slide's formula); 'skew': X skew(X^T G) (square X only). For a rotation the two are
 *  the same matrix; they differ once X has left the manifold. */
export function stiefelProject(X, G, form = 'sym') {
  const XtG = matmul(transpose(X), G);
  if (form === 'sym') return subM(G, matmul(X, sym(XtG)));
  if (form === 'skew') return matmul(X, skew(XtG));
  throw new Error(form);
}

/** Thin QR factor of X + V with positive diagonal of R (p126). */
export function retractQR(X, V) {
  const A = addM(X, V), m = A.length, n = A[0].length, Q = zerosM(m, n);
  for (let j = 0; j < n; j++) {
    let v = column(A, j);
    for (let pass = 0; pass < 2; pass++) for (let k = 0; k < j; k++) { const qk = column(Q, k); v = axpy(v, -dot(qk, v), qk); }
    const nv = norm(v);
    for (let i = 0; i < m; i++) Q[i][j] = v[i] / nv;
  }
  return Q;
}
/** Polar factor of X + V: the closest matrix with orthonormal columns. */
export function retractPolar(X, V) { const r = svd(addM(X, V)); return matmul(r.U, transpose(r.V)); }
/** X expm(X^T V) for a rotation matrix X and a tangent vector V = X Omega (Omega skew). */
export const retractExp = (X, V) => matmul(X, so3Exp(vee(skew(matmul(transpose(X), V)))));
export const RETRACTIONS = { exp: retractExp, qr: retractQR, polar: retractPolar };

/** One step for f over matrices with orthonormal columns; G is the ordinary gradient at X.
 *  'riemannian': Retr_X(-alpha (G - X sym(X^T G))); 'project_only': X - alpha grad (no retraction);
 *  'euclidean': X - alpha G (the naive step: the matrix stops being a rotation). */
export function manifoldStep(X, G, alpha, { retraction = 'exp', mode = 'riemannian', projection = 'sym' } = {}) {
  if (mode === 'riemannian') return RETRACTIONS[retraction](X, scaleM(stiefelProject(X, G), -alpha));
  if (mode === 'project_only') return subM(X, scaleM(stiefelProject(X, G, projection), alpha));
  if (mode === 'euclidean') return subM(X, scaleM(G, alpha));
  throw new Error(mode);
}
/** Gradient descent over rotations (or matrices with orthonormal columns); returns the iterates. */
export function manifoldGD(egrad, X0, alpha, iters, opts = {}) {
  let X = copyM(X0);
  const Xs = [copyM(X)];
  for (let k = 0; k < iters; k++) { X = manifoldStep(X, egrad(X), alpha, opts); Xs.push(copyM(X)); }
  return Xs;
}

/** How far a 3 x 3 matrix is from a rigid rotation: {residual |X^T X - I|_F, singularValues, det}. */
export const distortion = (X) => ({ residual: orthResidual(X), singularValues: singularValues(X), det: det(X) });

/** Wahba's problem: B = sum_i w_i b_i a_i^T; the cost is 1/2 sum w_i |b_i - R a_i|^2 = const - tr(R^T B). */
export function wahbaB(a, b, w) {
  const B = zerosM(3, 3);
  for (let i = 0; i < a.length; i++) for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) B[r][c] += w[i] * b[i][r] * a[i][c];
  return B;
}
export function wahbaCost(R, a, b, w) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const r = sub(b[i], matvec(R, a[i])); s += w[i] * dot(r, r); }
  return 0.5 * s;
}
/** Closed-form minimiser of Wahba's problem over SO(3) (Kabsch / SVD). */
export function wahbaSVD(B) {
  const { U, V } = svd(B), Vt = transpose(V), sgn = Math.sign(det(matmul(U, Vt)));
  return matmul(matmul(U, [[1, 0, 0], [0, 1, 0], [0, 0, sgn]]), Vt);
}
