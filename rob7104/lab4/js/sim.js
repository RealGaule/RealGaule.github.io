// Lab 4 numerics in plain JavaScript. Mirrors build/sim/linsys.py; the page's numbers are checked
// against the Python results by build/web/check_sim.html.
//
// Conventions: angles from the downward vertical, positive toward +x. The support (drone, hand, trolley)
// moves along x with acceleration a(t). Double pendulum state x = [th1, th2, w1, w2] with ABSOLUTE angles.

export const G = 9.81;
export const P = { m1: 5.0, m2: 1.0, l1: 0.9, l2: 0.7 };      // heavy upper crate, light lower parcel

// ---------------------------------------------------------------- small dense linear algebra
export const zeros = (n, m) => Array.from({ length: n }, () => new Array(m).fill(0));
export const eye = (n) => zeros(n, n).map((r, i) => { r[i] = 1; return r; });
export const transpose = (A) => A[0].map((_, j) => A.map((r) => r[j]));
export const add = (A, B, s = 1) => A.map((r, i) => r.map((v, j) => v + s * B[i][j]));
export const scale = (A, s) => A.map((r) => r.map((v) => v * s));
export function mul(A, B) {
  const n = A.length, m = B[0].length, k = B.length, C = zeros(n, m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    let s = 0;
    for (let l = 0; l < k; l++) s += A[i][l] * B[l][j];
    C[i][j] = s;
  }
  return C;
}
export const mulVec = (A, x) => A.map((r) => r.reduce((s, v, j) => s + v * x[j], 0));
export const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

/** Solve A x = b by Gaussian elimination with partial pivoting. */
export function solve(A, b) {
  const n = A.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

/** Numerical rank (row echelon with full pivoting, relative tolerance). */
export function rank(A, tol = 1e-9) {
  const M = A.map((r) => [...r]);
  const n = M.length, m = M[0].length;
  let big = 0;
  for (const r of M) for (const v of r) big = Math.max(big, Math.abs(v));
  let rk = 0;
  const used = new Array(m).fill(false);
  for (let r = 0; r < n; r++) {
    let pr = -1, pc = -1, best = tol * big;
    for (let i = r; i < n; i++) for (let j = 0; j < m; j++) {
      if (!used[j] && Math.abs(M[i][j]) > best) { best = Math.abs(M[i][j]); pr = i; pc = j; }
    }
    if (pr < 0) break;
    [M[r], M[pr]] = [M[pr], M[r]];
    used[pc] = true;
    for (let i = r + 1; i < n; i++) {
      const f = M[i][pc] / M[r][pc];
      for (let j = 0; j < m; j++) M[i][j] -= f * M[r][j];
    }
    rk++;
  }
  return rk;
}

/** Matrix exponential: scaling and squaring with a Taylor series. */
export function expm(A) {
  const n = A.length;
  let norm = 0;
  for (const r of A) norm = Math.max(norm, r.reduce((s, v) => s + Math.abs(v), 0));
  const sq = Math.max(0, Math.ceil(Math.log2(Math.max(norm, 1e-12) / 0.25)));
  const As = scale(A, 1 / 2 ** sq);
  let E = eye(n), term = eye(n);
  for (let k = 1; k <= 14; k++) { term = scale(mul(term, As), 1 / k); E = add(E, term); }
  for (let i = 0; i < sq; i++) E = mul(E, E);
  return E;
}

// ---------------------------------------------------------------- models
export function dpMatrices(p = P, d1 = 0, d2 = 0) {
  const { m1, m2, l1, l2 } = p;
  return {
    M: [[(m1 + m2) * l1 * l1, m2 * l1 * l2], [m2 * l1 * l2, m2 * l2 * l2]],
    K: [[(m1 + m2) * G * l1, 0], [0, m2 * G * l2]],
    C: [[d1 + d2, -d2], [-d2, d2]],
    h: [(m1 + m2) * l1, m2 * l2],
  };
}

function inv2(M) {
  const d = M[0][0] * M[1][1] - M[0][1] * M[1][0];
  return [[M[1][1] / d, -M[0][1] / d], [-M[1][0] / d, M[0][0] / d]];
}

/** x = (th, th'), x' = A x + B a. */
export function stateSpace({ M, C, K, h }) {
  const Mi = inv2(M), MK = mul(Mi, K), MC = mul(Mi, C), Mh = mulVec(Mi, h);
  return {
    A: [[0, 0, 1, 0], [0, 0, 0, 1],
      [-MK[0][0], -MK[0][1], -MC[0][0], -MC[0][1]], [-MK[1][0], -MK[1][1], -MC[1][0], -MC[1][1]]],
    B: [0, 0, -Mh[0], -Mh[1]],
  };
}

/** Natural frequencies and mass-normalized mode shapes of M q'' + K q = 0 (2 degrees of freedom). */
export function modes({ M, K }) {
  const a = M[0][0] * M[1][1] - M[0][1] * M[1][0];
  const b = -(K[0][0] * M[1][1] + K[1][1] * M[0][0] - K[0][1] * M[1][0] - K[1][0] * M[0][1]);
  const c = K[0][0] * K[1][1] - K[0][1] * K[1][0];
  const disc = Math.sqrt(b * b - 4 * a * c);
  const lam = [(-b - disc) / (2 * a), (-b + disc) / (2 * a)];
  const V = lam.map((l) => {
    let v = [-(K[0][1] - l * M[0][1]), K[0][0] - l * M[0][0]];
    const nrm = Math.sqrt(dot(v, mulVec(M, v)));
    v = v.map((x) => x / nrm);
    return v[0] < 0 ? v.map((x) => -x) : v;
  });
  return { w: lam.map(Math.sqrt), V };            // V[i] is mode i as [th1, th2]
}

export function singlePendulum(omega, zeta) {
  const l = G / (omega * omega);
  return { A: [[0, 1], [-omega * omega, -2 * zeta * omega]], B: [0, -1 / l], l };
}

/** Two independent pendulums on one support: x = (th1, th2, w1, w2). */
export function twinPendulums(l1, l2, zeta = 0) {
  const w = [Math.sqrt(G / l1), Math.sqrt(G / l2)];
  return {
    A: [[0, 0, 1, 0], [0, 0, 0, 1], [-w[0] * w[0], 0, -2 * zeta * w[0], 0], [0, -w[1] * w[1], 0, -2 * zeta * w[1]]],
    B: [0, 0, -1 / l1, -1 / l2],
  };
}

/** Prepend the trolley double integrator: x = (p, v, pendulum state), u = trolley acceleration. */
export function withTrolley({ A, B }) {
  const n = A.length, Af = zeros(n + 2, n + 2), Bf = new Array(n + 2).fill(0);
  Af[0][1] = 1; Bf[1] = 1;
  for (let i = 0; i < n; i++) { Bf[i + 2] = B[i]; for (let j = 0; j < n; j++) Af[i + 2][j + 2] = A[i][j]; }
  return { A: Af, B: Bf };
}

// ---------------------------------------------------------------- simulation
/** Exact stepping of x' = A x + B u for an input held constant over each step. */
export class LinearSim {
  constructor(A, B, dt) {
    const n = A.length, Z = zeros(n + 1, n + 1);
    for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) Z[i][j] = A[i][j] * dt; Z[i][n] = B[i] * dt; }
    const E = expm(Z);
    this.Ad = E.slice(0, n).map((r) => r.slice(0, n));
    this.Bd = E.slice(0, n).map((r) => r[n]);
    this.dt = dt;
  }
  step(x, u = 0) { return mulVec(this.Ad, x).map((v, i) => v + this.Bd[i] * u); }
}

function rk4(f, x, a0, a1, dt) {
  const am = 0.5 * (a0 + a1);
  const k1 = f(x, a0);
  const k2 = f(x.map((v, i) => v + 0.5 * dt * k1[i]), am);
  const k3 = f(x.map((v, i) => v + 0.5 * dt * k2[i]), am);
  const k4 = f(x.map((v, i) => v + dt * k3[i]), a1);
  return x.map((v, i) => v + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

/** One RK4 step of the full nonlinear double pendulum under support acceleration a0 -> a1. */
export function dpStep(p, d1, d2, x, a0, a1, dt) {
  const { m1, m2, l1, l2 } = p;
  const f = ([t1, t2, w1, w2], acc) => {
    const c = Math.cos(t1 - t2), s = Math.sin(t1 - t2);
    const M = [[(m1 + m2) * l1 * l1, m2 * l1 * l2 * c], [m2 * l1 * l2 * c, m2 * l2 * l2]];
    const r = [
      -m2 * l1 * l2 * s * w2 * w2 - (m1 + m2) * G * l1 * Math.sin(t1) - (m1 + m2) * l1 * Math.cos(t1) * acc - d1 * w1 + d2 * (w2 - w1),
      m2 * l1 * l2 * s * w1 * w1 - m2 * G * l2 * Math.sin(t2) - m2 * l2 * Math.cos(t2) * acc - d2 * (w2 - w1)];
    const Mi = inv2(M);
    return [w1, w2, Mi[0][0] * r[0] + Mi[0][1] * r[1], Mi[1][0] * r[0] + Mi[1][1] * r[1]];
  };
  return rk4(f, x, a0, a1, dt);
}

/** One RK4 step of the nonlinear single pendulum th'' = -w^2 sin th - 2 zeta w th' - (a/l) cos th. */
export function pendStep(omega, zeta, x, a0, a1, dt) {
  const l = G / (omega * omega);
  const f = ([th, w], acc) => [w, -omega * omega * Math.sin(th) - 2 * zeta * omega * w - acc / l * Math.cos(th)];
  return rk4(f, x, a0, a1, dt);
}

// ---------------------------------------------------------------- control
export function ctrb(A, B) {
  const n = A.length, cols = [B];
  for (let k = 1; k < n; k++) cols.push(mulVec(A, cols[k - 1]));
  return transpose(cols);
}

/** W(T) = int_0^T e^{As} B B^T e^{A^T s} ds, composite Simpson with n intervals (n even). */
export function gramian(A, B, T, n = 400) {
  const dim = A.length, h = T / n, step = expm(scale(A, h));
  let v = [...B], W = zeros(dim, dim);
  for (let k = 0; k <= n; k++) {
    const wgt = (k === 0 || k === n) ? 1 : (k % 2 ? 4 : 2);
    for (let i = 0; i < dim; i++) for (let j = 0; j < dim; j++) W[i][j] += wgt * v[i] * v[j];
    v = mulVec(step, v);
  }
  return scale(W, h / 3);
}

/** Minimum-energy input steering x0 to xf in time T. Returns { u(t), energy, lambda }. */
export function minEnergy(A, B, x0, xf, T, n = 400) {
  const W = gramian(A, B, T, n);
  const eAT = expm(scale(A, T));
  const d = xf.map((v, i) => v - dot(eAT[i], x0));
  const lam = solve(W, d);
  const At = transpose(A);
  return {
    energy: dot(lam, mulVec(W, lam)),
    lambda: lam,
    u: (t) => (t < 0 || t > T ? 0 : dot(B, mulVec(expm(scale(At, T - t)), lam))),
    /** Samples of u on a uniform grid with step dt (cheaper than calling u(t) repeatedly). */
    sample: (dt) => {
      const N = Math.round(T / dt), back = expm(scale(At, dt));
      const out = new Array(N + 1);
      let v = [...lam];
      for (let k = N; k >= 0; k--) { out[k] = dot(B, v); v = mulVec(back, v); }
      return out;
    },
  };
}

/** Two impulses that cancel a mode: { amps: [A1, A2], times: [0, t2] }. */
export function zvShaper(omega, zeta) {
  const wd = omega * Math.sqrt(1 - zeta * zeta);
  const Kf = Math.exp(-zeta * Math.PI / Math.sqrt(1 - zeta * zeta));
  return { amps: [1 / (1 + Kf), Kf / (1 + Kf)], times: [0, Math.PI / wd] };
}

/** Acceleration of a minimum-jerk move of distance D in time Tm, at time t. */
export function smoothStepAccel(D, Tm, t) {
  if (t < 0 || t > Tm) return 0;
  const s = t / Tm;
  return D / (Tm * Tm) * (60 * s - 180 * s * s + 120 * s * s * s);
}

/** Accelerate for Tm/2, brake for Tm/2. */
export function bangBangAccel(D, Tm, t) {
  if (t < 0 || t >= Tm) return 0;
  return (t < Tm / 2 ? 1 : -1) * 4 * D / (Tm * Tm);
}
