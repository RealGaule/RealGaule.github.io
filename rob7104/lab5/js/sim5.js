// Lab 5 numerics (Kalman filtering and the stochastic cart) in plain JavaScript: one ES module, no dependencies.
// Mirrors build/sim/cart.py, build/sim/kf.py and build/sim/nlfilters.py. Every function here is compared with
// the Python core by build/web/run_tests.py (fixtures written by build/web/export_fixtures.py).
//
// Conventions
//   vector            plain array (or Float64Array) of numbers
//   matrix            array of rows, A[i][j]
//   pose   g          [x, y, theta]      theta is NOT wrapped by the simulation
//   twist  xi         [v1, v2, alpha]    exponential coordinates, g = exp(v1 E1 + v2 E2 + alpha E3)
//   distribution      g = mu o exp(xi), xi ~ N(0, Sigma)   (perturbation on the right)
//   cloud of N carts  Float64Array of length 3 N, interleaved [x0, y0, th0, x1, y1, th1, ...]
//   beacons           array of [x, y]
//   measurement       [rho_1 .. rho_J] or, with bearing, [rho_1 .. rho_J, beta_1 .. beta_J]
//   lecture notation  F, H, Q, R, nu (innovation), S, K, P
// Wheel 1 of the SDE is the right wheel. Wheel radius r = 0.033 m, axle length l = 0.200 m.

export const R_WHEEL = 0.033;
export const AXLE = 0.200;
const TWO_PI = 2 * Math.PI;

// ================================================================================================ random numbers
/** Seeded generator: sfc32 state filled by splitmix32. `uniform()` in [0, 1) with 53 random bits, `normal()` by
 *  the Box-Muller transform (the second value of each pair is kept for the next call).
 *  The same seed gives the same stream in every browser up to the last bits of Math.log / cos / sin.
 *  build/web/export_fixtures.py holds a line-by-line Python copy (class MirrorRNG). */
export class RNG {
  constructor(seed = 0) { this.seed(seed); }

  seed(seed) {
    let s = Number(seed) >>> 0;
    const next = () => {
      s = (s + 0x9e3779b9) | 0;
      let z = s;
      z ^= z >>> 16; z = Math.imul(z, 0x21f0aaad);
      z ^= z >>> 15; z = Math.imul(z, 0x735a2d97);
      z ^= z >>> 15;
      return z >>> 0;
    };
    this.a = next(); this.b = next(); this.c = next(); this.d = 1;
    this._has = false; this._spare = 0;
    for (let i = 0; i < 15; i++) this.uint32();
    return this;
  }

  uint32() {
    const t = (((this.a + this.b) | 0) + this.d) | 0;
    this.d = (this.d + 1) | 0;
    this.a = this.b ^ (this.b >>> 9);
    this.b = (this.c + (this.c << 3)) | 0;
    this.c = (this.c << 21) | (this.c >>> 11);
    this.c = (this.c + t) | 0;
    return t >>> 0;
  }

  /** Uniform on [0, 1). Uses two 32-bit outputs. */
  uniform() {
    const hi = this.uint32() >>> 5, lo = this.uint32() >>> 6;
    return (hi * 67108864 + lo) / 9007199254740992;
  }

  /** Standard normal. */
  normal() {
    if (this._has) { this._has = false; return this._spare; }
    const u1 = 1 - this.uniform(), u2 = this.uniform();
    const rad = Math.sqrt(-2 * Math.log(u1)), ang = TWO_PI * u2;
    this._spare = rad * Math.sin(ang); this._has = true;
    return rad * Math.cos(ang);
  }

  fillUniform(out, n = out.length) { for (let i = 0; i < n; i++) out[i] = this.uniform(); return out; }
  fillNormal(out, n = out.length) { for (let i = 0; i < n; i++) out[i] = this.normal(); return out; }
}

const asRng = (x) => (x instanceof RNG ? x : new RNG(x === undefined ? 0 : x));

/** Round to the nearest whole number, halves to the even one, as Python's round() does: step counts such as
 *  round(T / dt) must be the same here and in the Python core. */
export function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

// ================================================================================================ small matrices
export const zeros = (n, m = n) => Array.from({ length: n }, () => new Array(m).fill(0));
export const eye = (n) => zeros(n, n).map((r, i) => { r[i] = 1; return r; });
export const diag = (d) => zeros(d.length, d.length).map((r, i) => { r[i] = d[i]; return r; });
export const copyMat = (A) => A.map((r) => Array.from(r));
export const transpose = (A) => A[0].map((_, j) => A.map((r) => r[j]));
export const add = (A, B, s = 1) => A.map((r, i) => r.map((v, j) => v + s * B[i][j]));
export const sub = (A, B) => add(A, B, -1);
export const scale = (A, s) => A.map((r) => r.map((v) => v * s));
export const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
export const mulVec = (A, x) => A.map((r) => dot(r, x));
export const addVec = (a, b, s = 1) => Array.from(a, (v, i) => v + s * b[i]);
export const outer = (a, b) => Array.from(a, (x) => Array.from(b, (y) => x * y));
export const symmetrize = (A) => A.map((r, i) => r.map((v, j) => 0.5 * (v + A[j][i])));

/** Matrix product A B (any sizes that fit). */
export function mul(A, B) {
  const n = A.length, k = B.length, m = B[0].length, C = zeros(n, m);
  for (let i = 0; i < n; i++) {
    const Ai = A[i], Ci = C[i];
    for (let l = 0; l < k; l++) {
      const a = Ai[l];
      if (a === 0) continue;
      const Bl = B[l];
      for (let j = 0; j < m; j++) Ci[j] += a * Bl[j];
    }
  }
  return C;
}

/** A B A^T. */
export const sandwich = (A, B) => mul(mul(A, B), transpose(A));

/** Lower Cholesky factor L (A = L L^T), or null when A is not positive definite. */
export function cholesky(A) {
  const n = A.length, L = zeros(n, n);
  for (let j = 0; j < n; j++) {
    let d = A[j][j];
    for (let k = 0; k < j; k++) d -= L[j][k] * L[j][k];
    if (!(d > 0) || !Number.isFinite(d)) return null;
    const ljj = Math.sqrt(d);
    L[j][j] = ljj;
    for (let i = j + 1; i < n; i++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = s / ljj;
    }
  }
  return L;
}

/** Cholesky factor that always exists: a matrix that is not positive definite (round-off in a filter that has
 *  lost track) is repaired by clipping its eigenvalues, as `_chol` of nlfilters.py does. `info.repaired` is set. */
export function choleskyRepair(A, info = null) {
  const L = cholesky(A);
  if (L) return L;
  if (info) info.repaired = (info.repaired || 0) + 1;
  const n = A.length, { values, vectors } = eigSym(symmetrize(A));
  let wmax = 0;
  for (const w of values) wmax = Math.max(wmax, Math.abs(w));
  const floor = 1e-12 * Math.max(wmax, 1e-300);
  const B = zeros(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let s = 0;
    for (let k = 0; k < n; k++) s += vectors[i][k] * Math.max(values[k], floor) * vectors[j][k];
    B[i][j] = s + (i === j ? 1e-15 : 0);
  }
  return cholesky(B);
}

/** Solve A x = b by Gaussian elimination with partial pivoting. b: vector. */
export function solve(A, b) { return solveMat(A, b.map((v) => [v])).map((r) => r[0]); }

/** Solve A X = B for a matrix right-hand side. */
export function solveMat(A, B) {
  const n = A.length, m = B[0].length, M = A.map((r, i) => [...r, ...B[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (p !== c) [M[c], M[p]] = [M[p], M[c]];
    const piv = M[c][c];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / piv;
      if (f === 0) continue;
      for (let k = c; k < n + m; k++) M[r][k] -= f * M[c][k];
    }
  }
  const X = zeros(n, m);
  for (let j = 0; j < m; j++) for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n + j];
    for (let k = i + 1; k < n; k++) s -= M[i][k] * X[k][j];
    X[i][j] = s / M[i][i];
  }
  return X;
}

export const inv = (A) => solveMat(A, eye(A.length));

/** Eigen-decomposition of a symmetric 2x2 matrix [[a, b], [b, d]] in closed form (one Jacobi rotation).
 *  Returns { values: [w0 <= w1], vectors }: column k of `vectors` is the unit eigenvector of values[k]; its
 *  component of largest size is positive. */
export function eigSym2(A) {
  const a = A[0][0], b = 0.5 * (A[0][1] + A[1][0]), d = A[1][1];
  let c = 1, s = 0, w0 = a, w1 = d;
  if (b !== 0) {
    const th = (d - a) / (2 * b);
    const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
    c = 1 / Math.sqrt(t * t + 1); s = t * c;
    w0 = a - t * b; w1 = d + t * b;
  }
  let V = [[c, s], [-s, c]];
  let values = [w0, w1];
  if (w0 > w1) { values = [w1, w0]; V = [[s, c], [c, -s]]; }
  return { values, vectors: canonicalSigns(V) };
}

/** Eigen-decomposition of a symmetric n x n matrix by cyclic Jacobi rotations (use it for 3x3; fine up to a few
 *  dozen rows). Returns { values ascending, vectors } with eigenvectors in the columns, signs as in eigSym2. */
export function eigSym(A) {
  const n = A.length, a = symmetrize(A), V = eye(n);
  for (let sweep = 0; sweep < 80; sweep++) {
    let off = 0, tot = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { tot += a[i][j] * a[i][j]; if (i !== j) off += a[i][j] * a[i][j]; }
    if (off === 0 || off <= 1e-36 * tot) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      const apq = a[p][q];
      if (apq === 0) continue;
      const th = (a[q][q] - a[p][p]) / (2 * apq);
      const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
      }
      a[p][q] = a[q][p] = 0;
      for (let k = 0; k < n; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i][i] - a[j][j]);
  const values = order.map((i) => a[i][i]);
  const vectors = V.map((row) => order.map((i) => row[i]));
  return { values, vectors: canonicalSigns(vectors) };
}

function canonicalSigns(V) {
  const n = V.length;
  for (let k = 0; k < n; k++) {
    let big = 0;
    for (let i = 0; i < n; i++) if (Math.abs(V[i][k]) > Math.abs(big) * (1 + 1e-12)) big = V[i][k];
    if (big < 0) for (let i = 0; i < n; i++) V[i][k] = -V[i][k];
  }
  return V;
}

// ================================================================================================ chi-square
/** ln Gamma(x), x > 0 (Lanczos, g = 7). */
export function gammaln(x) {
  const g = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - gammaln(1 - x);
  x -= 1;
  let a = g[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += g[i] / (x + i);
  return 0.5 * Math.log(TWO_PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Regularised lower incomplete gamma function P(a, x). */
export function gammaP(a, x) {
  if (!(x > 0)) return 0;
  const lead = Math.exp(-x + a * Math.log(x) - gammaln(a));
  if (x < a + 1) {
    let sum = 1 / a, term = sum;
    for (let n = 1; n < 100000; n++) {
      term *= x / (a + n); sum += term;
      if (Math.abs(term) < Math.abs(sum) * 1e-17) break;
    }
    return lead * sum;
  }
  const tiny = 1e-300;
  let b = x + 1 - a, c = 1 / tiny, d = 1 / b, h = d;
  for (let i = 1; i < 100000; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b; if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c; if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-16) break;
  }
  return 1 - lead * h;
}

/** Chi-square distribution with k degrees of freedom: cumulative probability at x. */
export const chi2Cdf = (x, k) => gammaP(0.5 * k, 0.5 * x);

/** Chi-square quantile: the x with chi2Cdf(x, k) = p. */
export function chi2Quantile(p, k) {
  if (!(p > 0)) return 0;
  if (!(p < 1)) return Infinity;
  if (k === 2) return -2 * Math.log1p(-p);
  // Wilson-Hilferty start, then Newton on the cumulative probability (bracketed)
  const zq = normalQuantile(p), h = 2 / (9 * k);
  let x = k * Math.pow(Math.max(1 - h + zq * Math.sqrt(h), 1e-3), 3);
  let lo = 0, hi = Infinity;
  const a = 0.5 * k, lg = gammaln(a);
  for (let it = 0; it < 200; it++) {
    const f = gammaP(a, 0.5 * x) - p;
    if (f > 0) hi = x; else lo = x;
    const pdf = 0.5 * Math.exp((a - 1) * Math.log(0.5 * x) - 0.5 * x - lg);
    let xn = x - f / pdf;
    if (!(xn > lo) || !(xn < hi) || !Number.isFinite(xn)) xn = Number.isFinite(hi) ? 0.5 * (lo + hi) : 2 * x;
    if (Math.abs(xn - x) <= 1e-15 * Math.abs(xn)) { x = xn; break; }
    x = xn;
  }
  return x;
}

/** Standard normal quantile (Acklam's rational approximation, refined by one Halley step; error below 1e-15). */
export function normalQuantile(p) {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  let x;
  if (p < 0.02425) {
    const q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p > 1 - 0.02425) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else {
    const q = p - 0.5, r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  const e = normalCdf(x) - p, u = e * Math.sqrt(TWO_PI) * Math.exp(0.5 * x * x);
  return x - u / (1 + 0.5 * x * u);
}

/** Standard normal cumulative probability. */
export function normalCdf(x) {
  const half = 0.5 * gammaP(0.5, 0.5 * x * x);
  return x >= 0 ? 0.5 + half : 0.5 - half;
}

/** Interval that holds, with probability `level`, the average of nAvg independent chi-square(dof) variables
 *  (kf.chi2_bounds). For the NIS of one scalar reading averaged over a window of n readings: chi2Bounds(1, n). */
export function chi2Bounds(dof, nAvg = 1, level = 0.95) {
  const a = 0.5 * (1 - level), k = dof * nAvg;
  return [chi2Quantile(a, k) / nAvg, chi2Quantile(1 - a, k) / nAvg];
}

export const CHI2_95_2 = chi2Quantile(0.95, 2);      // 5.991: the "95 %" xy ellipse
export const CHI2_95_3 = chi2Quantile(0.95, 3);      // 7.815
export const CHI2_9999_2 = chi2Quantile(0.9999, 2);  // 18.42: the ellipse of the "lost" rule

// ================================================================================================ SE(2)
/** Wrap an angle to (-pi, pi]. */
export function wrapAngle(a) {
  let m = (Math.PI - a) % TWO_PI;
  if (m < 0) m += TWO_PI;
  return Math.PI - m;
}

const sinc = (a) => (a === 0 ? 1 : Math.sin(a) / a);                  // sin(a) / a
const cosc = (a) => Math.sin(0.5 * a) * sinc(0.5 * a);                // (1 - cos a) / a, no cancellation

/** Exponential map: twist [v1, v2, alpha] -> pose [x, y, theta]. */
export function se2Exp(xi) {
  const A = sinc(xi[2]), B = cosc(xi[2]);
  return [A * xi[0] - B * xi[1], B * xi[0] + A * xi[1], xi[2]];
}

/** Logarithm: pose -> twist. The angle is wrapped to (-pi, pi] first (wrap = true). */
export function se2Log(g, wrap = true) {
  const al = wrap ? wrapAngle(g[2]) : g[2];
  const a = Math.cos(0.5 * al) / sinc(0.5 * al), b = 0.5 * al;
  return [a * g[0] + b * g[1], -b * g[0] + a * g[1], al];
}

/** Group product a o b (first a, then b expressed in the frame of a). */
export function se2Compose(a, b) {
  const c = Math.cos(a[2]), s = Math.sin(a[2]);
  return [a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]];
}

export function se2Inverse(g) {
  const c = Math.cos(g[2]), s = Math.sin(g[2]);
  return [-(c * g[0] + s * g[1]), -(-s * g[0] + c * g[1]), -g[2]];
}

/** Relative pose a^-1 o b: pose b seen from the frame of a. */
export function se2Relative(a, b) {
  const c = Math.cos(a[2]), s = Math.sin(a[2]), dx = b[0] - a[0], dy = b[1] - a[1];
  return [c * dx + s * dy, -s * dx + c * dy, b[2] - a[2]];
}

/** 3x3 homogeneous matrix of a pose. */
export function se2Matrix(g) {
  const c = Math.cos(g[2]), s = Math.sin(g[2]);
  return [[c, -s, g[0]], [s, c, g[1]], [0, 0, 1]];
}

/** Adjoint matrix: g exp(xi) g^-1 = exp(Ad(g) xi). */
export function se2Ad(g) {
  const c = Math.cos(g[2]), s = Math.sin(g[2]);
  return [[c, -s, g[1]], [s, c, -g[0]], [0, 0, 1]];
}

/** Little adjoint of a twist h: Ad(exp(t h)) = expm(t ad(h)). */
export const se2ad = (h) => [[0, -h[2], h[1]], [h[2], 0, -h[0]], [0, 0, 0]];

// ================================================================================================ the cart
/** Wheel speeds [w1, w2] (rad/s) that give forward speed v and turn rate omega. */
export const bodyToWheel = (v, omega, r = R_WHEEL, l = AXLE) => [(v + omega * l / 2) / r, (v - omega * l / 2) / r];
/** Forward speed and turn rate [v, omega] produced by wheel speeds w1, w2. */
export const wheelToBody = (w1, w2, r = R_WHEEL, l = AXLE) => [r / 2 * (w1 + w2), r / l * (w1 - w2)];
/** Q = D diag(r^2/2, 0, 2 r^2/l^2): body-frame diffusion per unit time. */
export const noiseCov = (D, r = R_WHEEL, l = AXLE) => [[D * r * r / 2, 0, 0], [0, 0, 0], [0, 0, 2 * D * r * r / (l * l)]];
export const straightCommand = (length = 1, T = 1) => [length / T, 0];
export const arcCommand = (radius = 1, angle = Math.PI / 2, T = 1) => [radius * angle / T, angle / T];

let _scratch = new Float64Array(0);

/** One time step of the SDE for a cloud of carts, in place.
 *  pose: Float64Array(3 N). v, omega: commanded forward speed and turn rate. D: wheel noise strength (rad^2/s).
 *  opt.method  "em": Euler-Maruyama, heading frozen during the step (default; the truth of the lab).
 *              "geo": the body-frame increment is applied through the exponential map (used by the particle filter).
 *  opt.wheel   Float64Array(2 N): the wheel angle increments of this step are ADDED to it ([w1 of all carts, then w2]).
 *  opt.n       number of carts to move (default pose.length / 3).
 *  Random numbers are used in the order of cart.em_step: N normals for wheel 1, then N for wheel 2. */
export function cartStep(pose, v, omega, D, dt, rng, opt = {}) {
  const r = opt.r === undefined ? R_WHEEL : opt.r, l = opt.l === undefined ? AXLE : opt.l;
  const N = opt.n === undefined ? (pose.length / 3) | 0 : opt.n;
  const geo = opt.method === 'geo', wheel = opt.wheel || null;
  if (_scratch.length < 2 * N) _scratch = new Float64Array(2 * N);
  const z = _scratch;
  for (let i = 0; i < 2 * N; i++) z[i] = rng.normal();
  const sd = Math.sqrt(D * dt);
  const w1dt = (v + omega * l / 2) / r * dt, w2dt = (v - omega * l / 2) / r * dt;
  const hr = r / 2, rl = r / l;
  for (let i = 0, j = 0; i < N; i++, j += 3) {
    const d1 = sd * z[i] + w1dt, d2 = sd * z[N + i] + w2dt;
    const ds = hr * (d1 + d2), dth = rl * (d1 - d2), th = pose[j + 2];
    const c = Math.cos(th), s = Math.sin(th);
    if (geo) {
      const A = sinc(dth), B = cosc(dth);
      pose[j] += ds * (c * A - s * B);
      pose[j + 1] += ds * (s * A + c * B);
    } else {
      pose[j] += ds * c;
      pose[j + 1] += ds * s;
    }
    pose[j + 2] = th + dth;
    if (wheel) { wheel[i] += d1; wheel[N + i] += d2; }
  }
  return pose;
}

/** Simulate N independent carts from the start pose opt.x0 (default the origin, heading along +x).
 *  v, omega: numbers (constant command; give T) or arrays with one entry per step (then T is ignored).
 *  opt: dt (default 1e-3), seed or rng, x0, method, onStep(k, pose, t) called after every step.
 *  Returns { pose: Float64Array(3 N), dt, nSteps }. */
export function simulateCart(N, v, omega, D, T, opt = {}) {
  const rng = asRng(opt.rng !== undefined ? opt.rng : opt.seed);
  let dt = opt.dt === undefined ? 1e-3 : opt.dt, n;
  const arr = typeof v !== 'number' || typeof omega !== 'number';
  if (arr) n = typeof v !== 'number' ? v.length : omega.length;
  else { n = roundHalfEven(T / dt); dt = T / n; }
  const pose = new Float64Array(3 * N);
  if (opt.x0) for (let i = 0; i < N; i++) { pose[3 * i] = opt.x0[0]; pose[3 * i + 1] = opt.x0[1]; pose[3 * i + 2] = opt.x0[2]; }
  for (let k = 0; k < n; k++) {
    cartStep(pose, typeof v === 'number' ? v : v[k], typeof omega === 'number' ? omega : omega[k], D, dt, rng, opt);
    if (opt.onStep) opt.onStep(k, pose, (k + 1) * dt);
  }
  return { pose, dt, nSteps: n };
}

/** Sample mean (d) and covariance (d x d, divisor N - 1) of N points stored interleaved in Z (length d N). */
export function meanCov(Z, d, N = (Z.length / d) | 0) {
  const m = new Array(d).fill(0), C = zeros(d, d);
  for (let i = 0; i < N; i++) for (let a = 0; a < d; a++) m[a] += Z[d * i + a];
  for (let a = 0; a < d; a++) m[a] /= N;
  for (let i = 0; i < N; i++) for (let a = 0; a < d; a++) {
    const da = Z[d * i + a] - m[a];
    for (let b = a; b < d; b++) C[a][b] += da * (Z[d * i + b] - m[b]);
  }
  for (let a = 0; a < d; a++) for (let b = a; b < d; b++) { C[a][b] /= (N - 1); C[b][a] = C[a][b]; }
  return { mean: m, cov: C };
}

/** Mean and covariance of (x, y, theta) of a cloud: the Gaussian fitted in the ground coordinates. */
export const fitCartesianGaussian = (pose, N = (pose.length / 3) | 0) => meanCov(pose, 3, N);

/** Mean on the group: the pose mu with sum_i log(mu^-1 g_i) = 0 (fixed-point iteration). pose: Float64Array(3 N). */
export function groupMean(pose, N = (pose.length / 3) | 0, tol = 1e-12, maxIter = 100) {
  let sx = 0, sy = 0, ss = 0, sc = 0;
  for (let i = 0; i < N; i++) { sx += pose[3 * i]; sy += pose[3 * i + 1]; ss += Math.sin(pose[3 * i + 2]); sc += Math.cos(pose[3 * i + 2]); }
  let mu = [sx / N, sy / N, Math.atan2(ss / N, sc / N)];
  for (let it = 0; it < maxIter; it++) {
    const m = [0, 0, 0];
    for (let i = 0; i < N; i++) {
      const xi = se2Log(se2Relative(mu, [pose[3 * i], pose[3 * i + 1], pose[3 * i + 2]]));
      m[0] += xi[0]; m[1] += xi[1]; m[2] += xi[2];
    }
    m[0] /= N; m[1] /= N; m[2] /= N;
    mu = se2Compose(mu, se2Exp(m));
    if (Math.hypot(m[0], m[1], m[2]) < tol) break;
  }
  return mu;
}

/** Group mean mu, covariance Sigma of xi_i = log(mu^-1 g_i), and the xi_i (Float64Array(3 N)). */
export function groupMeanCov(pose, N = (pose.length / 3) | 0) {
  const mu = groupMean(pose, N), xi = new Float64Array(3 * N);
  for (let i = 0; i < N; i++) {
    const x = se2Log(se2Relative(mu, [pose[3 * i], pose[3 * i + 1], pose[3 * i + 2]]));
    xi[3 * i] = x[0]; xi[3 * i + 1] = x[1]; xi[3 * i + 2] = x[2];
  }
  return { mu, Sigma: meanCov(xi, 3, N).cov, xi };
}

// ---------------------------------------------------------------- closed-form mean and covariance
/** mu(t) = exp(t (v, 0, omega)): the noise-free path for a constant command. */
export const meanPath = (v, omega, t) => se2Exp([v * t, 0, omega * t]);

function gaussLegendre(n) {
  const x = new Array(n), w = new Array(n);
  for (let i = 0; i < n; i++) {
    let z = Math.cos(Math.PI * (i + 0.75) / (n + 0.5)), pp = 1;
    for (let it = 0; it < 100; it++) {
      let p1 = 1, p2 = 0;
      for (let j = 1; j <= n; j++) { const p3 = p2; p2 = p1; p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j; }
      pp = n * (z * p1 - p2) / (z * z - 1);
      const dz = p1 / pp;
      z -= dz;
      if (Math.abs(dz) < 1e-16) break;
    }
    x[i] = z; w[i] = 2 / ((1 - z * z) * pp * pp);
  }
  return { x, w };
}
const GL = gaussLegendre(12);

/** Covariance of the exponential coordinates after driving the constant command (v, omega) for time t, to first
 *  order in the noise (Lecture 11+12 p93):
 *      Sigma(t) = int_0^t Ad(exp(-s h)) Q Ad(exp(-s h))^T ds,   h = (v, 0, omega),   Q = noiseCov(D).
 *  The pose is then close to mu(t) o exp(xi), xi ~ N(0, Sigma(t)). Evaluated by Gauss-Legendre quadrature on
 *  panels of at most 1 rad of turning, which is exact to round-off for every omega (the integrand is a
 *  trigonometric polynomial of degree 2) and has no cancellation at omega = 0.
 *  Sigma_11 of this first-order formula is too small (see sigma11SecondOrderStraight). */
export function sigmaClosedForm(v, omega, D, t, r = R_WHEEL, l = AXLE) {
  const q1 = D * r * r / 2, q3 = 2 * D * r * r / (l * l);
  const panels = Math.max(1, Math.ceil(Math.abs(omega * t)));
  let s11 = 0, s12 = 0, s13 = 0, s22 = 0, s23 = 0, s33 = 0;
  for (let p = 0; p < panels; p++) {
    const a = t * p / panels, half = 0.5 * t / panels;
    for (let k = 0; k < GL.x.length; k++) {
      const s = a + half * (GL.x[k] + 1), w = half * GL.w[k];
      const th = omega * s, c = Math.cos(th), sn = Math.sin(th);
      const mx = sinc(th) * v * s, my = cosc(th) * v * s;       // mu(s)
      const a3x = sn * mx - c * my, a3y = c * mx + sn * my;       // third column of Ad(mu(s)^-1); the first is (c, -sn, 0)
      s11 += w * (q1 * c * c + q3 * a3x * a3x);
      s12 += w * (-q1 * c * sn + q3 * a3x * a3y);
      s13 += w * q3 * a3x;
      s22 += w * (q1 * sn * sn + q3 * a3y * a3y);
      s23 += w * q3 * a3y;
      s33 += w * q3;
    }
  }
  return [[s11, s12, s13], [s12, s22, s23], [s13, s23, s33]];
}

/** The closed form written out for a straight run (omega = 0). */
export function sigmaStraight(v, D, t, r = R_WHEEL, l = AXLE) {
  const q1 = D * r * r / 2, q3 = 2 * D * r * r / (l * l);
  return [[q1 * t, 0, 0], [0, q3 * v * v * t * t * t / 3, q3 * v * t * t / 2], [0, q3 * v * t * t / 2, q3 * t]];
}

/** sigma_theta^2(t) = 2 D r^2 t / l^2. */
export const headingVariance = (D, t, r = R_WHEEL, l = AXLE) => 2 * D * r * r * t / (l * l);

/** Along-track variance Sigma_11 of a STRAIGHT run to second order in the noise:
 *  D r^2 t / 2 + sigma_theta^4 (v t)^2 / 48. An approximation: within 2.5 % of the sample value up to D T = 16. */
export const sigma11SecondOrderStraight = (v, D, t, r = R_WHEEL, l = AXLE) =>
  D * r * r * t / 2 + headingVariance(D, t, r, l) ** 2 * (v * t) ** 2 / 48;

/** sigmaStraight with Sigma_11 replaced by its second-order value (straight runs only). */
export function sigmaStraightSecondOrder(v, D, t, r = R_WHEEL, l = AXLE) {
  const S = sigmaStraight(v, D, t, r, l);
  S[0][0] = sigma11SecondOrderStraight(v, D, t, r, l);
  return S;
}

/** One step of the particle-free propagation through a piecewise-constant command:
 *  mu' = mu o exp(h dt),  Sigma' = Ad(exp(-h dt)) Sigma Ad(exp(-h dt))^T + Sigma_step(h, dt). Returns { mu, Sigma }. */
export function propagateStep(mu, Sigma, v, omega, dt, D, r = R_WHEEL, l = AXLE) {
  const step = meanPath(v, omega, dt), Adi = se2Ad(se2Inverse(step));
  return { mu: se2Compose(mu, step), Sigma: add(sandwich(Adi, Sigma), sigmaClosedForm(v, omega, D, dt, r, l)) };
}

/** propagateStep along command arrays v[k], omega[k]. Returns { mus (n + 1), Sigmas (n + 1) }. */
export function propagate(v, omega, dt, D, mu0 = [0, 0, 0], Sigma0 = zeros(3, 3), r = R_WHEEL, l = AXLE) {
  const n = typeof v === 'number' ? omega.length : v.length;
  const mus = [Array.from(mu0)], Sigmas = [copyMat(Sigma0)];
  let mu = mus[0], S = Sigmas[0];
  for (let k = 0; k < n; k++) {
    ({ mu, Sigma: S } = propagateStep(mu, S, typeof v === 'number' ? v : v[k], typeof omega === 'number' ? omega : omega[k], dt, D, r, l));
    mus.push(mu); Sigmas.push(S);
  }
  return { mus, Sigmas };
}

/** E[x(T)] of a straight run: v T (1 - e^-a) / a, a = D r^2 T / l^2 (exact; shorter than v T). */
export function meanForwardDistance(v, D, T, r = R_WHEEL, l = AXLE) {
  const a = D * r * r * T / (l * l);
  return v * T * (Math.abs(a) < 1e-12 ? 1 : -Math.expm1(-a) / a);
}

/** Exact Cartesian mean [E x, E y] for a constant command: v int_0^t exp((i omega - D r^2 / l^2) s) ds. */
export function meanPosition(v, omega, D, t, r = R_WHEEL, l = AXLE) {
  const a = -D * r * r / (l * l) * t, b = omega * t, n2 = a * a + b * b;
  if (Math.sqrt(n2) < 1e-12) return [v * t, 0];
  const ea = Math.exp(a), sh = Math.sin(0.5 * b);
  const re = Math.expm1(a) * Math.cos(b) - 2 * sh * sh, im = ea * Math.sin(b);    // exp(a + i b) - 1
  return [v * t * (re * a + im * b) / n2, v * t * (im * a - re * b) / n2];
}

// ---------------------------------------------------------------- sampling
/** A matrix L with L L^T = S: the Cholesky factor, or an eigen-factor when S is only positive semi-definite. */
export function gaussianFactor(S) {
  const L = cholesky(S);
  if (L) return L;
  const { values, vectors } = eigSym(S);
  return vectors.map((row) => row.map((v, k) => v * Math.sqrt(Math.max(values[k], 0))));
}

/** N poses g_i = mu o exp(xi_i), xi_i ~ N(0, Sigma): a Gaussian in exponential coordinates. Float64Array(3 N).
 *  opt.xi (Float64Array(3 N)) receives the xi_i. Three normals per pose, in order. */
export function sampleExpGaussian(mu, Sigma, N, rng, opt = {}) {
  rng = asRng(rng);
  const L = gaussianFactor(Sigma), out = new Float64Array(3 * N), z = [0, 0, 0];
  for (let i = 0; i < N; i++) {
    z[0] = rng.normal(); z[1] = rng.normal(); z[2] = rng.normal();
    const xi = mulVec(L, z), g = se2Compose(mu, se2Exp(xi));
    out[3 * i] = g[0]; out[3 * i + 1] = g[1]; out[3 * i + 2] = g[2];
    if (opt.xi) { opt.xi[3 * i] = xi[0]; opt.xi[3 * i + 1] = xi[1]; opt.xi[3 * i + 2] = xi[2]; }
  }
  return out;
}

/** N samples of N(mean, cov) in d dimensions, interleaved Float64Array(d N). For d = 3: poses (x, y, theta) of a
 *  Gaussian in the ground coordinates. */
export function sampleGaussian(mean, cov, N, rng) {
  rng = asRng(rng);
  const d = mean.length, L = gaussianFactor(cov), out = new Float64Array(d * N), z = new Array(d);
  for (let i = 0; i < N; i++) {
    for (let a = 0; a < d; a++) z[a] = rng.normal();
    for (let a = 0; a < d; a++) { let s = mean[a]; for (let b = 0; b < d; b++) s += L[a][b] * z[b]; out[d * i + a] = s; }
  }
  return out;
}
export const sampleCartesianGaussian = sampleGaussian;

// ---------------------------------------------------------------- pictures and fit metrics
/** n points [x, y] of the ellipse (p - mean)^T cov^-1 (p - mean) = nsig^2. cov: 2x2 (or the xy block of a larger one). */
export function ellipsePoints(mean, cov, nsig = 2, n = 96) {
  const { values, vectors } = eigSym2([[cov[0][0], cov[0][1]], [cov[1][0], cov[1][1]]]);
  const s0 = nsig * Math.sqrt(Math.max(values[0], 0)), s1 = nsig * Math.sqrt(Math.max(values[1], 0));
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const a = TWO_PI * i / n, c = s0 * Math.cos(a), s = s1 * Math.sin(a);
    out[i] = [mean[0] + c * vectors[0][0] + s * vectors[0][1], mean[1] + c * vectors[1][0] + s * vectors[1][1]];
  }
  return out;
}

/** Half-axes and direction of the nsig ellipse: { a, b (a >= b), angle of the long axis }. */
export function ellipseAxes(cov, nsig = 2) {
  const { values, vectors } = eigSym2([[cov[0][0], cov[0][1]], [cov[1][0], cov[1][1]]]);
  return { a: nsig * Math.sqrt(Math.max(values[1], 0)), b: nsig * Math.sqrt(Math.max(values[0], 0)), angle: Math.atan2(vectors[1][1], vectors[0][1]) };
}

/** Share of the samples Z (interleaved, `stride` numbers per sample, the first d are used) inside the
 *  probability-p ellipsoid of N(mean, cov). Equals p when the samples are that Gaussian. */
export function gaussianCoverage(Z, stride, d, mean, cov, p = 0.95, N = (Z.length / stride) | 0) {
  const C = zeros(d, d);
  for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] = cov[a][b];
  const Ci = inv(C), lim = chi2Quantile(p, d), e = new Array(d);
  let inside = 0;
  for (let i = 0; i < N; i++) {
    for (let a = 0; a < d; a++) e[a] = Z[stride * i + a] - mean[a];
    let d2 = 0;
    for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) d2 += e[a] * Ci[a][b] * e[b];
    if (d2 <= lim) inside++;
  }
  return inside / N;
}

/** Share of the carts of a cloud (Float64Array(3 N)) inside the probability-p covariance ellipse in (x, y).
 *  p = 0.95 is the 2.45-sigma ellipse, p = 0.8647 the 2-sigma ellipse. */
export const ellipseCoverage = (pose, mean, cov, p = 0.95, N = (pose.length / 3) | 0) => gaussianCoverage(pose, 3, 2, mean, cov, p, N);

function quantileSorted(sorted, q) {
  const h = q * (sorted.length - 1), i = Math.floor(h), f = h - i;
  return i + 1 < sorted.length ? sorted[i] + f * (sorted[i + 1] - sorted[i]) : sorted[i];
}

/** Grid limits [[xmin, xmax], [ymin, ymax]] for tvDistanceXY from reference clouds (each Float64Array with
 *  `stride` numbers per sample, x and y first): the box between the q and 1 - q quantiles of the pooled samples,
 *  widened by `pad` of its size on each side. */
export function tvLimits(clouds, stride = 3, q = 5e-4, pad = 0.05) {
  if (!Array.isArray(clouds)) clouds = [clouds];
  const lim = [];
  for (let a = 0; a < 2; a++) {
    let n = 0;
    for (const c of clouds) n += (c.length / stride) | 0;
    const col = new Float64Array(n);
    let k = 0;
    for (const c of clouds) for (let i = 0; i < ((c.length / stride) | 0); i++) col[k++] = c[stride * i + a];
    col.sort();
    const lo = quantileSorted(col, q), hi = quantileSorted(col, 1 - q), d = (hi - lo) * pad;
    lim.push([lo - d, hi + d]);
  }
  return lim;
}

/** Normalised (x, y) histogram (Float64Array(bins * bins), index ix * bins + iy). Samples outside the limits go
 *  into the edge cells, so no probability mass is dropped. */
export function xyHistogram(Z, lim, bins = 40, stride = 3, N = (Z.length / stride) | 0) {
  const h = new Float64Array(bins * bins), idx = [0, 0];
  for (let i = 0; i < N; i++) {
    for (let a = 0; a < 2; a++) {
      const lo = lim[a][0], hi = lim[a][1], eps = 1e-9 * (hi - lo), step = (hi - lo) / bins;
      const x = Math.min(Math.max(Z[stride * i + a], lo + eps), hi - eps);
      let b = Math.min(bins - 1, Math.max(0, Math.floor((x - lo) / step)));
      while (b > 0 && x < lo + b * step) b--;                              // same edges as numpy.linspace
      while (b < bins - 1 && x >= lo + (b + 1) * step) b++;
      idx[a] = b;
    }
    h[idx[0] * bins + idx[1]] += 1;
  }
  for (let i = 0; i < h.length; i++) h[i] /= N;
  return h;
}

/** Total-variation distance between the (x, y) distributions of two sample sets: 0.5 sum |p_a - p_b| over a
 *  bins x bins grid. It has a positive floor from the finite samples: quote it next to the value for two sets
 *  drawn from the same distribution. opt: lim (default tvLimits of both), bins (40), stride (3). */
export function tvDistanceXY(A, B, opt = {}) {
  const stride = opt.stride || 3, bins = opt.bins || 40, lim = opt.lim || tvLimits([A, B], stride);
  const ha = xyHistogram(A, lim, bins, stride), hb = xyHistogram(B, lim, bins, stride);
  let s = 0;
  for (let i = 0; i < ha.length; i++) s += Math.abs(ha[i] - hb[i]);
  return 0.5 * s;
}

// ================================================================================================ linear Kalman filter
/** Exact discretisation of dp = v dt, dv = sqrt(q) dW (Lecture 10 p52). Returns { F, Q }. */
export const pvModel = (dt, q) => ({ F: [[1, dt], [0, 1]], Q: [[q * dt ** 3 / 3, q * dt ** 2 / 2], [q * dt ** 2 / 2, q * dt]] });
/** Continuous model dx = A x dt + G dW, G G^T = Qc, position output C. Returns { A, Qc, C }. */
export const pvContinuous = (q) => ({ A: [[0, 1], [0, 0]], Qc: [[0, 0], [0, q]], C: [[1, 0]] });
export const H_POS = [[1, 0]];
export const LECTURE = { q: 0.05, R: 1.0, dt: 0.5, T: 20.0 };

/** Prediction: x- = F x (+ B u), P- = F P F^T + Q. Returns { x, P }. */
export function kfPredict(x, P, F, Q, B = null, u = null) {
  let xn = mulVec(F, x);
  if (B && u) xn = addVec(xn, mulVec(B, u));
  return { x: xn, P: symmetrize(add(sandwich(F, P), Q)) };
}

/** Correction with the reading z (vector): nu = z - H x-, S = H P- H^T + R, K = P- H^T S^-1, x+ = x- + K nu,
 *  P+ = (I - K H) P- (I - K H)^T + K R K^T (Joseph form; joseph = false gives (I - K H) P-).
 *  Returns { x, P, nu, S, K, nis } with nis = nu^T S^-1 nu. */
export function kfCorrect(x, P, z, H, R, joseph = true) {
  const n = x.length, Ht = transpose(H);
  const nu = addVec(z, mulVec(H, x), -1);
  const S = add(mul(mul(H, P), Ht), R);
  const K = transpose(solveMat(S, mul(H, P)));                    // S K^T = H P  (S, P symmetric)
  const nisK = dot(nu, solve(S, nu));
  const IKH = sub(eye(n), mul(K, H));
  const Pn = joseph ? add(sandwich(IKH, P), sandwich(K, R)) : mul(IKH, P);
  return { x: addVec(x, mulVec(K, nu)), P: symmetrize(Pn), nu, S, K, nis: nisK };
}

/** Linear Kalman filter object. kf = new KalmanFilter({ F, H, Q, R, x0, P0 }); kf.predict(); kf.correct([z]);
 *  kf.step(z) predicts and then corrects unless z is null, undefined or NaN. State in kf.x, kf.P; the last
 *  correction in kf.last ({ nu, S, K, nis }). */
export class KalmanFilter {
  constructor({ F, H, Q, R, x0, P0, B = null, joseph = true }) {
    Object.assign(this, { F, H, Q, R, B, joseph });
    this.x = Array.from(x0); this.P = copyMat(P0); this.last = null;
  }

  predict(u = null, F = this.F, Q = this.Q) {
    ({ x: this.x, P: this.P } = kfPredict(this.x, this.P, F, Q, this.B, u));
    return this;
  }

  correct(z, H = this.H, R = this.R) {
    const zz = typeof z === 'number' ? [z] : z;
    const r = kfCorrect(this.x, this.P, zz, H, R, this.joseph);
    this.x = r.x; this.P = r.P;
    this.last = { nu: r.nu, S: r.S, K: r.K, nis: r.nis };
    return this.last;
  }

  step(z = null, u = null) {
    this.predict(u);
    const zz = typeof z === 'number' ? [z] : z;
    if (zz === null || zz === undefined || zz.some((v) => v === null || Number.isNaN(v))) return null;
    return this.correct(zz);
  }
}

/** Steady-state filter: solution of the discrete algebraic Riccati equation
 *  P- = F (P- - P- H^T (H P- H^T + R)^-1 H P-) F^T + Q, by the structure-preserving doubling algorithm.
 *  Returns { Pprior, Ppost, K, S, iterations }. */
export function steadyState(F, H, Q, R) {
  const n = F.length, I = eye(n);
  let A = transpose(F), G = mul(transpose(H), solveMat(R, H)), X = copyMat(Q), it = 0;
  for (; it < 100; it++) {
    const W = add(I, mul(G, X));                                   // I + G X
    const WiA = solveMat(W, A), WiG = solveMat(W, G);
    const An = mul(A, WiA);
    const Gn = add(G, mul(mul(A, WiG), transpose(A)));
    const Xn = add(X, mul(mul(transpose(A), X), WiA));
    let dx = 0, nx = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { dx = Math.max(dx, Math.abs(Xn[i][j] - X[i][j])); nx = Math.max(nx, Math.abs(Xn[i][j])); }
    A = An; G = symmetrize(Gn); X = symmetrize(Xn);
    if (dx <= 1e-16 * nx) break;
  }
  const Pm = X, Ht = transpose(H);
  const S = add(mul(mul(H, Pm), Ht), R);
  const K = transpose(solveMat(S, mul(H, Pm)));
  const IKH = sub(I, mul(K, H));
  return { Pprior: Pm, Ppost: add(sandwich(IKH, Pm), sandwich(K, R)), K, S, iterations: it + 1 };
}

/** Actual steady error covariance of a filter that uses the fixed gain K (optimal or not) when the world has
 *  Qtrue, Rtrue: Sigma+ = A Sigma+ A^T + (I - K H) Q (I - K H)^T + K R K^T, A = (I - K H) F.
 *  Returns { SigmaPrior, SigmaPost, Strue (actual innovation covariance), innovLag1 }. */
export function trueErrorCovariance(F, H, Qtrue, Rtrue, K) {
  const n = F.length, IKH = sub(eye(n), mul(K, H)), A = mul(IKH, F);
  const W = add(sandwich(IKH, Qtrue), sandwich(K, Rtrue));
  const M = zeros(n * n, n * n), w = new Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    w[i * n + j] = W[i][j];
    for (let k = 0; k < n; k++) for (let l = 0; l < n; l++) M[i * n + j][k * n + l] = (i === k && j === l ? 1 : 0) - A[i][k] * A[j][l];
  }
  const s = solve(M, w), Sp = zeros(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Sp[i][j] = s[i * n + j];
  const Sm = add(sandwich(F, Sp), Qtrue), Ht = transpose(H);
  const Strue = add(mul(mul(H, Sm), Ht), Rtrue);
  const innovLag1 = mul(mul(H, F), sub(mul(Sm, Ht), mul(K, Strue)));
  return { SigmaPrior: Sm, SigmaPost: Sp, Strue, innovLag1 };
}

/** Right-hand side of the Kalman-Bucy Riccati equation: dP/dt = A P + P A^T + Qc - K Rc K^T, K = P C^T Rc^-1. */
export function riccatiRhs(P, A, C, Qc, Rc) {
  const K = transpose(solveMat(Rc, mul(C, P)));
  const AP = mul(A, P);
  return sub(add(add(AP, transpose(AP)), Qc), sandwich(K, Rc));
}

/** One classical Runge-Kutta step of length h of the Riccati equation. Returns the new P (symmetric). */
export function riccatiStep(P, A, C, Qc, Rc, h) {
  const k1 = riccatiRhs(P, A, C, Qc, Rc);
  const k2 = riccatiRhs(add(P, k1, h / 2), A, C, Qc, Rc);
  const k3 = riccatiRhs(add(P, k2, h / 2), A, C, Qc, Rc);
  const k4 = riccatiRhs(add(P, k3, h), A, C, Qc, Rc);
  return symmetrize(P.map((r, i) => r.map((v, j) => v + h / 6 * (k1[i][j] + 2 * k2[i][j] + 2 * k3[i][j] + k4[i][j]))));
}

/** Kalman-Bucy gain K = P C^T Rc^-1. */
export const kalmanBucyGain = (P, C, Rc) => transpose(solveMat(Rc, mul(C, P)));

/** Steady state of the Riccati equation for the position-velocity robot, in closed form:
 *  P_pv = sqrt(q Rc), P_pp = sqrt(2 Rc P_pv), P_vv = P_pp P_pv / Rc. */
export function kalmanBucySteadyPV(q, Rc) {
  const ppv = Math.sqrt(q * Rc), ppp = Math.SQRT2 * q ** 0.25 * Rc ** 0.75, pvv = Math.SQRT2 * q ** 0.75 * Rc ** 0.25;
  return [[ppp, ppv], [ppv, pvv]];
}

/** Discrete filter with sampling interval dt and reading variance R = Rc / dt (same information per second for
 *  every dt). Returns { t, Pprior, Ppost, K (index 0 = P0), tSaw, PSaw (the plotted sawtooth: at each sample time
 *  first the prior, then the posterior), dt, R }. */
export function sawtooth(dt, q, Rc, P0, T) {
  const n = roundHalfEven(T / dt), { F, Q } = pvModel(dt, q), R = [[Rc / dt]];
  let P = copyMat(P0);
  const t = [0], Pprior = [copyMat(P)], Ppost = [copyMat(P)], K = [[[0], [0]]], tSaw = [0], PSaw = [copyMat(P)];
  for (let k = 1; k <= n; k++) {
    const pm = kfPredict([0, 0], P, F, Q).P, c = kfCorrect([0, 0], pm, [0], H_POS, R);
    P = c.P;
    t.push(k * dt); Pprior.push(pm); Ppost.push(P); K.push(c.K);
    tSaw.push(k * dt, k * dt); PSaw.push(pm, P);
  }
  return { t, Pprior, Ppost, K, tSaw, PSaw, dt, R: Rc / dt };
}

/** Normalised innovation squared nu^T S^-1 nu (also the NEES with nu = error, S = P). */
export const nis = (nu, S) => dot(nu, solve(S, nu));

// ================================================================================================ nonlinear filters
/** One filter step of the cart. x = [x, y, theta]; n = [n1, n2] = wheel-angle noise of the step (rad), or null.
 *  ds = v dt + r/2 (n1 + n2), dth = omega dt + r/l (n1 - n2), the cart moves ds along the heading theta + dth/2. */
export function motion(x, v, omega, dt, n = null, r = R_WHEEL, l = AXLE) {
  let ds = v * dt, dth = omega * dt;
  if (n) { ds += r / 2 * (n[0] + n[1]); dth += r / l * (n[0] - n[1]); }
  const phi = x[2] + dth / 2;
  return [x[0] + ds * Math.cos(phi), x[1] + ds * Math.sin(phi), x[2] + dth];
}

/** Jacobians of the step at n = 0: { F = d step / dx (3x3), G = d step / dn (3x2) }. */
export function motionJacobians(x, v, omega, dt, r = R_WHEEL, l = AXLE) {
  const ds = v * dt, phi = x[2] + omega * dt / 2, c = Math.cos(phi), s = Math.sin(phi), k = ds * r / (2 * l);
  return {
    F: [[1, 0, -ds * s], [0, 1, ds * c], [0, 0, 1]],
    G: [[r / 2 * c - k * s, r / 2 * c + k * s], [r / 2 * s + k * c, r / 2 * s - k * c], [r / l, -r / l]],
  };
}

/** The sideways noise of one step that the two wheel-noise terms cannot produce: (v dt)^2 s2 / 12 n n^T,
 *  s2 = 2 D dt r^2 / l^2, n = (-sin phi, cos phi, 0), phi = theta + omega dt / 2 (option withinStep of the filters). */
export function withinStepCov(theta, v, omega, dt, D, r = R_WHEEL, l = AXLE) {
  const phi = theta + omega * dt / 2, n = [-Math.sin(phi), Math.cos(phi), 0];
  const k = (v * dt) ** 2 * 2 * D * dt * (r / l) ** 2 / 12;
  return outer(n, n).map((row) => row.map((e) => k * e));
}

/** Noise-free measurement: ranges to the beacons, then (bearing = true) the bearings of the beacons seen from the
 *  cart, atan2(by - y, bx - x) - theta, wrapped. */
export function measure(x, beacons, bearing = false) {
  const J = beacons.length, z = new Array(bearing ? 2 * J : J);
  for (let j = 0; j < J; j++) {
    const dx = beacons[j][0] - x[0], dy = beacons[j][1] - x[1];
    z[j] = Math.hypot(dx, dy);
    if (bearing) z[J + j] = wrapAngle(Math.atan2(dy, dx) - x[2]);
  }
  return z;
}

/** H = d measure / dx at x: J x 3, or 2J x 3 with bearing. */
export function measureJacobian(x, beacons, bearing = false) {
  const J = beacons.length, H = zeros(bearing ? 2 * J : J, 3);
  for (let j = 0; j < J; j++) {
    const dx = beacons[j][0] - x[0], dy = beacons[j][1] - x[1], q = dx * dx + dy * dy, rho = Math.sqrt(q);
    H[j][0] = -dx / rho; H[j][1] = -dy / rho;
    if (bearing) { H[J + j][0] = dy / q; H[J + j][1] = -dx / q; H[J + j][2] = -1; }
  }
  return H;
}

/** R: diagonal covariance of the measurement vector. */
export function measCov(nBeacons, sigmaR, bearing = false, sigmaB = 0.05) {
  const d = new Array(nBeacons).fill(sigmaR * sigmaR);
  if (bearing) for (let j = 0; j < nBeacons; j++) d.push(sigmaB * sigmaB);
  return diag(d);
}

/** a - b for measurement vectors; components from index nRange on are angles and are wrapped. */
export function residual(a, b, nRange = Infinity) {
  return Array.from(a, (v, i) => (i >= nRange ? wrapAngle(v - b[i]) : v - b[i]));
}

/** EKF prediction: m- = step(m), P- = F P F^T + D dt G G^T (+ withinStepCov when opt.withinStep). Returns { m, P }. */
export function ekfPredict(m, P, v, omega, D, dt, opt = {}) {
  const r = opt.r === undefined ? R_WHEEL : opt.r, l = opt.l === undefined ? AXLE : opt.l;
  const { F, G } = motionJacobians(m, v, omega, dt, r, l);
  let Pn = add(sandwich(F, P), mul(G, transpose(G)), D * dt);
  if (opt.withinStep) Pn = add(Pn, withinStepCov(m[2], v, omega, dt, D, r, l));
  return { m: motion(m, v, omega, dt, null, r, l), P: symmetrize(Pn) };
}

/** EKF correction with H evaluated at the predicted mean (Joseph form unless opt.joseph === false).
 *  Returns { m, P, nu, S, K }. */
export function ekfUpdate(m, P, z, beacons, R, opt = {}) {
  const bearing = !!opt.bearing, H = measureJacobian(m, beacons, bearing), Ht = transpose(H);
  const nu = residual(z, measure(m, beacons, bearing), beacons.length);
  const S = add(mul(mul(H, P), Ht), R);
  const K = transpose(solveMat(S, mul(H, P)));
  const mn = addVec(m, mulVec(K, nu));
  let Pn;
  if (opt.joseph === false) Pn = sub(P, sandwich(K, S));
  else { const A = sub(eye(3), mul(K, H)); Pn = add(sandwich(A, P), sandwich(K, R)); }
  return { m: mn, P: symmetrize(Pn), nu, S, K };
}

/** Scaling of the unscented transform. Default of the lab: alpha = 1, beta = 0, kappa = 1 (all weights positive). */
export const UT_DEFAULT = Object.freeze({ alpha: 1, beta: 0, kappa: 1 });

/** Weights of the 2n + 1 sigma points: { Wm, Wc, lam }, lam = alpha^2 (n + kappa) - n. */
export function utWeights(n, ut = UT_DEFAULT) {
  const lam = ut.alpha * ut.alpha * (n + ut.kappa) - n;
  const Wm = new Array(2 * n + 1).fill(0.5 / (n + lam)), Wc = Wm.slice();
  Wm[0] = lam / (n + lam);
  Wc[0] = Wm[0] + 1 - ut.alpha * ut.alpha + ut.beta;
  return { Wm, Wc, lam };
}

/** Sigma points { X ((2n + 1) rows of n), Wm, Wc }: the mean, then mean + columns, then mean - columns of the
 *  lower Cholesky factor of (n + lam) P. */
export function sigmaPoints(m, P, ut = UT_DEFAULT) {
  const n = m.length, { Wm, Wc, lam } = utWeights(n, ut);
  const L = choleskyRepair(scale(P, n + lam));
  const X = [Array.from(m)];
  for (let k = 0; k < n; k++) X.push(Array.from(m, (v, i) => v + L[i][k]));
  for (let k = 0; k < n; k++) X.push(Array.from(m, (v, i) => v - L[i][k]));
  return { X, Wm, Wc };
}

/** Weighted mean and covariance of propagated sigma points Y. Components from index nLinear on are angles: their
 *  mean is taken relative to the centre point and differences are wrapped. Returns { mean, cov, dY }. */
export function unscentedMeanCov(Y, Wm, Wc, nLinear = Infinity) {
  const d = Y[0].length, mean = new Array(d).fill(0);
  let dY;
  if (nLinear >= d) {
    for (let s = 0; s < Y.length; s++) for (let i = 0; i < d; i++) mean[i] += Wm[s] * Y[s][i];
    dY = Y.map((y) => y.map((v, i) => v - mean[i]));
  } else {
    for (let s = 0; s < Y.length; s++) { const e = residual(Y[s], Y[0], nLinear); for (let i = 0; i < d; i++) mean[i] += Wm[s] * e[i]; }
    for (let i = 0; i < d; i++) mean[i] += Y[0][i];
    dY = Y.map((y) => residual(y, mean, nLinear));
  }
  const cov = zeros(d, d);
  for (let s = 0; s < Y.length; s++) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cov[i][j] += Wc[s] * dY[s][i] * dY[s][j];
  return { mean, cov, dY };
}

/** UKF prediction for the cart. The state is augmented with the two wheel-noise terms, (x, y, theta, n1, n2) with
 *  covariance blockdiag(P, D dt I): 11 sigma points go through the step model, noise included.
 *  Returns { m, P, X (11 x 5, before the step), Y (11 x 3, after), Wm, Wc }. opt: ut, withinStep, r, l. */
export function ukfPredict(m, P, v, omega, D, dt, opt = {}) {
  const r = opt.r === undefined ? R_WHEEL : opt.r, l = opt.l === undefined ? AXLE : opt.l;
  const Pa = zeros(5, 5);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) Pa[i][j] = P[i][j];
  Pa[3][3] = Pa[4][4] = D * dt;
  const { X, Wm, Wc } = sigmaPoints([m[0], m[1], m[2], 0, 0], Pa, opt.ut || UT_DEFAULT);
  const Y = X.map((x) => motion(x, v, omega, dt, [x[3], x[4]], r, l));
  const { mean, cov } = unscentedMeanCov(Y, Wm, Wc);
  const Pn = opt.withinStep ? add(cov, withinStepCov(m[2], v, omega, dt, D, r, l)) : cov;
  return { m: mean, P: symmetrize(Pn), X, Y, Wm, Wc };
}

/** UKF correction with range (and bearing) readings; sigma points are drawn afresh from (m, P).
 *  Returns { m, P, nu, S, K, zbar, X }. opt: bearing, ut. */
export function ukfUpdate(m, P, z, beacons, R, opt = {}) {
  const bearing = !!opt.bearing, J = beacons.length;
  const { X, Wm, Wc } = sigmaPoints(m, P, opt.ut || UT_DEFAULT);
  const Z = X.map((x) => measure(x, beacons, bearing));
  const u = unscentedMeanCov(Z, Wm, Wc, J);
  const S = add(u.cov, R), dz = Z[0].length;
  const Cxz = zeros(3, dz);
  for (let s = 0; s < X.length; s++) for (let i = 0; i < 3; i++) for (let j = 0; j < dz; j++) Cxz[i][j] += Wc[s] * (X[s][i] - m[i]) * u.dY[s][j];
  const K = transpose(solveMat(S, transpose(Cxz)));
  const nu = residual(z, u.mean, J);
  return { m: addVec(m, mulVec(K, nu)), P: symmetrize(sub(P, sandwich(K, S))), nu, S, K, zbar: u.mean, X };
}

class GaussFilter {
  constructor(m0, P0, D, opt = {}) {
    this.m = Array.from(m0); this.P = copyMat(P0); this.D = D; this.opt = opt; this.info = null;
  }

  update(z, beacons, sigmaR, opt = {}) {
    const R = measCov(beacons.length, sigmaR, !!opt.bearing, opt.sigmaB);
    const out = this._update(z, beacons, R, { ...this.opt, ...opt });
    this.m = out.m; this.P = out.P; this.info = out;
    return out;
  }
}

/** Extended Kalman filter for the cart. f = new EKF(m0, P0, D, { withinStep }); f.predict(v, omega, dt);
 *  f.update(z, beacons, sigmaR, { bearing, sigmaB }); state in f.m, f.P. */
export class EKF extends GaussFilter {
  predict(v, omega, dt) { ({ m: this.m, P: this.P } = ekfPredict(this.m, this.P, v, omega, this.D, dt, this.opt)); return this; }
  _update(z, beacons, R, opt) { return ekfUpdate(this.m, this.P, z, beacons, R, opt); }
}

/** Unscented Kalman filter for the cart (augmented prediction). After predict, f.points = { X, Y, Wm, Wc } holds
 *  the 11 sigma points before and after the step, for drawing. Options: ut, withinStep. */
export class UKF extends GaussFilter {
  predict(v, omega, dt) {
    const p = ukfPredict(this.m, this.P, v, omega, this.D, dt, this.opt);
    this.m = p.m; this.P = p.P; this.points = { X: p.X, Y: p.Y, Wm: p.Wm, Wc: p.Wc };
    return this;
  }

  _update(z, beacons, R, opt) { return ukfUpdate(this.m, this.P, z, beacons, R, opt); }
}

/** Systematic resampling: weights w (sum 1) -> Int32Array(nOut) of indices. u0 is ONE uniform number in [0, 1);
 *  particle p is copied floor(nOut w_p) or ceil(nOut w_p) times. */
export function systematicIndices(w, nOut, u0) {
  const P = w.length, idx = new Int32Array(nOut);
  let cs = Math.min(w[0], 1), i = 0;
  if (P === 1) cs = 1;
  for (let j = 0; j < nOut; j++) {
    const u = (u0 + j) / nOut;
    while (cs < u && i < P - 1) { i++; cs = i === P - 1 ? 1 : Math.min(cs + w[i], 1); }
    idx[j] = i;
  }
  return idx;
}

/** Bootstrap particle filter for one run. Particles move by the cart SDE itself (cartStep with sub-steps of about
 *  dtSub), are weighted by the likelihood of each reading, and are resampled (systematic) when the effective
 *  sample size falls below essFrac of the particles. No roughening: diversity comes from the wheel noise alone.
 *  With about 1000 particles this filter loses the cart in a few per cent of hard runs (long blind drive, precise
 *  range reading); the lab's reference numbers use far more particles.
 *  pf = new ParticleFilter(m0, P0, D, { nParticles, seed | rng, dtSub, essFrac, method });
 *  pf.predict(v, omega, dt); pf.update(z, beacons, sigmaR, { bearing, sigmaB, resample }); pf.meanCov().
 *  pf.x: Float64Array(3 Np) particles; pf.logw: Float64Array(Np) unnormalised log-weights. */
export class ParticleFilter {
  constructor(m0, P0, D, opt = {}) {
    this.Np = opt.nParticles === undefined ? 2000 : opt.nParticles;
    this.rng = asRng(opt.rng !== undefined ? opt.rng : opt.seed);
    this.D = D;
    this.dtSub = opt.dtSub === undefined ? 0.01 : opt.dtSub;
    this.essFrac = opt.essFrac === undefined ? 0.5 : opt.essFrac;
    this.stepOpt = { method: opt.method || 'geo', r: opt.r, l: opt.l };
    this.nResampled = 0;
    const L = choleskyRepair(P0), Np = this.Np;
    this.x = new Float64Array(3 * Np); this.logw = new Float64Array(Np);
    for (let p = 0; p < Np; p++) {
      const z0 = this.rng.normal(), z1 = this.rng.normal(), z2 = this.rng.normal();
      for (let i = 0; i < 3; i++) this.x[3 * p + i] = m0[i] + (L[i][0] * z0 + L[i][1] * z1 + L[i][2] * z2);
    }
  }

  predict(v, omega, dt) {
    const n = Math.max(1, roundHalfEven(dt / this.dtSub));
    for (let k = 0; k < n; k++) cartStep(this.x, v, omega, this.D, dt / n, this.rng, this.stepOpt);
    return this;
  }

  /** Normalised weights (Float64Array(Np)). */
  weights() {
    const Np = this.Np, w = new Float64Array(Np);
    let mx = -Infinity, s = 0;
    for (let p = 0; p < Np; p++) if (this.logw[p] > mx) mx = this.logw[p];
    for (let p = 0; p < Np; p++) { w[p] = Math.exp(this.logw[p] - mx); s += w[p]; }
    for (let p = 0; p < Np; p++) w[p] /= s;
    return w;
  }

  /** Effective sample size 1 / sum w^2. */
  ess(w = this.weights()) { let s = 0; for (let p = 0; p < this.Np; p++) s += w[p] * w[p]; return 1 / s; }

  /** Weight the particles with the reading z ([rho_j ...] or [rho_j ..., beta_j ...]); then resample if needed
   *  (opt.resample === false leaves that to the caller). */
  update(z, beacons, sigmaR, opt = {}) {
    const bearing = !!opt.bearing, sigmaB = opt.sigmaB === undefined ? 0.05 : opt.sigmaB, J = beacons.length, x = this.x;
    for (let p = 0; p < this.Np; p++) {
      let q = 0;
      for (let j = 0; j < J; j++) {
        const dx = beacons[j][0] - x[3 * p], dy = beacons[j][1] - x[3 * p + 1];
        const e = (z[j] - Math.hypot(dx, dy)) / sigmaR;
        q += e * e;
        if (bearing) { const eb = wrapAngle(z[J + j] - wrapAngle(Math.atan2(dy, dx) - x[3 * p + 2])) / sigmaB; q += eb * eb; }
      }
      this.logw[p] -= 0.5 * q;
    }
    if (opt.resample !== false) this.resample();
    return this;
  }

  /** Systematic resampling when the effective sample size is below the threshold (or always, force = true).
   *  Returns true when the particles were resampled. */
  resample(force = false) {
    const w = this.weights(), Np = this.Np;
    if (!force && !(this.ess(w) < this.essFrac * Np)) return false;
    const idx = systematicIndices(w, Np, this.rng.uniform()), xn = new Float64Array(3 * Np);
    for (let p = 0; p < Np; p++) { const s = 3 * idx[p]; xn[3 * p] = this.x[s]; xn[3 * p + 1] = this.x[s + 1]; xn[3 * p + 2] = this.x[s + 2]; }
    this.x = xn; this.logw.fill(0); this.nResampled++;
    return true;
  }

  /** Weighted mean and covariance of the cloud (the heading is not wrapped). Returns { m, P }. */
  meanCov() {
    const w = this.weights(), Np = this.Np, x = this.x, m = [0, 0, 0], P = zeros(3, 3);
    for (let p = 0; p < Np; p++) for (let i = 0; i < 3; i++) m[i] += w[p] * x[3 * p + i];
    for (let p = 0; p < Np; p++) for (let i = 0; i < 3; i++) {
      const di = x[3 * p + i] - m[i];
      for (let j = i; j < 3; j++) P[i][j] += w[p] * di * (x[3 * p + j] - m[j]);
    }
    for (let i = 0; i < 3; i++) for (let j = 0; j < i; j++) P[i][j] = P[j][i];
    return { m, P };
  }

  /** n particles drawn according to the weights (for drawing an equal-weight cloud). Float64Array(3 n). */
  equalWeightCloud(n = this.Np, rng = this.rng) {
    const idx = systematicIndices(this.weights(), n, rng.uniform()), out = new Float64Array(3 * n);
    for (let p = 0; p < n; p++) { const s = 3 * idx[p]; out[3 * p] = this.x[s]; out[3 * p + 1] = this.x[s + 1]; out[3 * p + 2] = this.x[s + 2]; }
    return out;
  }
}

/** Errors of a Gaussian summary (m, P) against the true pose: { nees, neesXY, pos, err } (heading error wrapped). */
export function errorStats(xTrue, m, P) {
  const e = [xTrue[0] - m[0], xTrue[1] - m[1], wrapAngle(xTrue[2] - m[2])];
  const ridge = 1e-12 * (P[0][0] + P[1][1] + P[2][2]) + 1e-300;
  const Pr = P.map((r, i) => r.map((v, j) => v + (i === j ? ridge : 0)));
  const exy = [e[0], e[1]], Pxy = [[Pr[0][0], Pr[0][1]], [Pr[1][0], Pr[1][1]]];
  return { nees: dot(e, solve(Pr, e)), neesXY: dot(exy, solve(Pxy, exy)), pos: Math.hypot(e[0], e[1]), err: e };
}

/** The lab's "lost" rule: position error above lostDist (0.15 m) AND outside the filter's own 99.99 % xy ellipse:
 *  the filter is far away and does not know it. */
export function isLost(xTrue, m, P, lostDist = 0.15) {
  const s = errorStats(xTrue, m, P);
  return s.pos > lostDist && s.neesXY > CHI2_9999_2;
}

/** EKF and UKF predictions only, from an exactly known start, for a unit command of `length` in time T split into
 *  nSteps filter steps (nlfilters.dead_reckoning). Returns { ekf: { m, P }, ukf: { m, P } }. */
export function deadReckoning(D, T = 1, length = 1, omega = 0, nSteps = 40, opt = {}) {
  const v = length / T, dt = T / nSteps, sd0 = opt.sd0 === undefined ? 1e-6 : opt.sd0;
  let e = { m: [0, 0, 0], P: diag([sd0 * sd0, sd0 * sd0, sd0 * sd0]) }, u = { m: [0, 0, 0], P: diag([sd0 * sd0, sd0 * sd0, sd0 * sd0]) };
  for (let k = 0; k < nSteps; k++) {
    e = ekfPredict(e.m, e.P, v, omega, D, dt, opt);
    u = ukfPredict(u.m, u.P, v, omega, D, dt, opt);
  }
  return { ekf: { m: e.m, P: e.P }, ukf: { m: u.m, P: u.P } };
}
