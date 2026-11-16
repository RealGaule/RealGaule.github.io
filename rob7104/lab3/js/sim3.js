// Lab 3 numerics in plain JavaScript (mirrors build/sim/swarm.py).
// Corridor: 1-D position x (m). Arena: 2-D (x, y). Lamps are Gaussian light spots.

// ---------------------------------------------------------------- random numbers (seeded)
export function rng(seed = 1) {
  let s = seed >>> 0;
  const uni = () => {                       // mulberry32
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  const normal = () => {                    // Box-Muller
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0, v = 0;
    while (u === 0) u = uni();
    v = uni();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
  return { uni, normal };
}

// ---------------------------------------------------------------- lamps
export class Lamps1D {
  constructor(centres, brightness, widths) {
    this.c = centres; this.b = brightness;
    this.w = Array.isArray(widths) ? widths : centres.map(() => widths);
  }
  I(x) { let s = 0; for (let k = 0; k < this.c.length; k++) { const d = x - this.c[k]; s += this.b[k] * Math.exp(-d * d / (2 * this.w[k] ** 2)); } return s; }
  dI(x) { let s = 0; for (let k = 0; k < this.c.length; k++) { const d = x - this.c[k], w2 = this.w[k] ** 2; s -= this.b[k] * d / w2 * Math.exp(-d * d / (2 * w2)); } return s; }
}

export class Lamps2D {
  constructor(centres, brightness, widths) {
    this.c = centres; this.b = brightness;
    this.w = Array.isArray(widths) ? widths : centres.map(() => widths);
  }
  I(x, y) { let s = 0; for (let k = 0; k < this.c.length; k++) { const dx = x - this.c[k][0], dy = y - this.c[k][1]; s += this.b[k] * Math.exp(-(dx * dx + dy * dy) / (2 * this.w[k] ** 2)); } return s; }
  grad(x, y) {
    let gx = 0, gy = 0;
    for (let k = 0; k < this.c.length; k++) {
      const dx = x - this.c[k][0], dy = y - this.c[k][1], w2 = this.w[k] ** 2;
      const e = this.b[k] * Math.exp(-(dx * dx + dy * dy) / (2 * w2)) / w2;
      gx -= e * dx; gy -= e * dy;
    }
    return [gx, gy];
  }
}

// ---------------------------------------------------------------- one step for many particles (in place)
/** Euler-Maruyama on a Float64Array x: x += f(x) dt + g(x) dW. */
export function emStep(x, f, g, dt, R, lo = -Infinity, hi = Infinity) {
  const s = Math.sqrt(dt);
  for (let i = 0; i < x.length; i++) {
    let v = x[i] + f(x[i]) * dt + g(x[i]) * s * R.normal();
    if (v < lo) v = 2 * lo - v; if (v > hi) v = 2 * hi - v;
    x[i] = v;
  }
}

/** Stochastic Heun (Stratonovich). */
export function heunStep(x, f, g, dt, R, lo = -Infinity, hi = Infinity) {
  const s = Math.sqrt(dt);
  for (let i = 0; i < x.length; i++) {
    const dW = s * R.normal(), x0 = x[i];
    const xp = x0 + f(x0) * dt + g(x0) * dW;
    let v = x0 + 0.5 * (f(x0) + f(xp)) * dt + 0.5 * (g(x0) + g(xp)) * dW;
    if (v < lo) v = 2 * lo - v; if (v > hi) v = 2 * hi - v;
    x[i] = v;
  }
}

/** Run-and-tumble robots: speed v(x), direction dir[i] = +-1 flips at rate a. */
export function runTumbleStep(x, dir, v, a, dt, R, lo, hi) {
  for (let i = 0; i < x.length; i++) {
    let p = x[i] + dir[i] * v(x[i]) * dt;
    if (p < lo) { p = 2 * lo - p; dir[i] = -dir[i]; }
    if (p > hi) { p = 2 * hi - p; dir[i] = -dir[i]; }
    if (R.uni() < a * dt) dir[i] = -dir[i];
    x[i] = p;
  }
}

// ---------------------------------------------------------------- histogram
export function histogram(x, lo, hi, nb) {
  const h = new Float64Array(nb), w = (hi - lo) / nb;
  let n = 0;
  for (let i = 0; i < x.length; i++) {
    const k = Math.floor((x[i] - lo) / w);
    if (k >= 0 && k < nb) { h[k]++; n++; }
  }
  const total = x.length * w;
  for (let k = 0; k < nb; k++) h[k] /= total;
  return h;                                  // density units (integrates to the fraction inside [lo, hi])
}

// ---------------------------------------------------------------- Fokker-Planck in 1-D (zero flux walls)
/**
 * Cell-centred finite volumes with the Scharfetter-Gummel flux on q = D p (exact for the stationary profile).
 * Ito: dp/dt = -d/dx J, J = f p - d/dx (D p), D = g^2/2. reading 'strat' adds 1/2 g g' to f.
 * Time stepping: implicit Euler (unconditionally stable) with the Thomas algorithm.
 */
export class FP1D {
  constructor(f, g, lo, hi, n = 200, reading = 'ito') {
    this.n = n; this.lo = lo; this.hi = hi; this.h = (hi - lo) / n;
    this.x = Float64Array.from({ length: n }, (_, i) => lo + (i + 0.5) * this.h);
    this.p = new Float64Array(n);
    this.setModel(f, g, reading);
  }
  setModel(f, g, reading = 'ito') {
    const n = this.n, h = this.h, x = this.x;
    const D = Float64Array.from(x, (v) => 0.5 * g(v) ** 2);
    const B = (z) => (Math.abs(z) < 1e-8 ? 1 - z / 2 : z / Math.expm1(z));
    this.lower = new Float64Array(n); this.diag = new Float64Array(n); this.upper = new Float64Array(n);
    for (let i = 0; i < n - 1; i++) {
      const xf = 0.5 * (x[i] + x[i + 1]);
      let fe = f(xf);
      if (reading === 'strat') { const e = 1e-6; fe += 0.5 * g(xf) * (g(xf + e) - g(xf - e)) / (2 * e); }
      const De = 0.5 * g(xf) ** 2, z = fe / De * h;
      const ai = B(-z) * D[i] / h, aj = B(z) * D[i + 1] / h;      // J = ai p_i - aj p_{i+1}
      this.diag[i] -= ai / h; this.upper[i] += aj / h;
      this.lower[i + 1] += ai / h; this.diag[i + 1] -= aj / h;
    }
  }
  setDensity(fn) {
    let s = 0;
    for (let i = 0; i < this.n; i++) { this.p[i] = Math.max(0, fn(this.x[i])); s += this.p[i] * this.h; }
    for (let i = 0; i < this.n; i++) this.p[i] /= s || 1;
  }
  /** Advance by dt with implicit Euler: (I - dt A) p_new = p. */
  step(dt) {
    const n = this.n, a = new Float64Array(n), b = new Float64Array(n), c = new Float64Array(n), d = Float64Array.from(this.p);
    for (let i = 0; i < n; i++) { a[i] = -dt * this.lower[i]; b[i] = 1 - dt * this.diag[i]; c[i] = -dt * this.upper[i]; }
    for (let i = 1; i < n; i++) { const m = a[i] / b[i - 1]; b[i] -= m * c[i - 1]; d[i] -= m * d[i - 1]; }
    this.p[n - 1] = d[n - 1] / b[n - 1];
    for (let i = n - 2; i >= 0; i--) this.p[i] = (d[i] - c[i] * this.p[i + 1]) / b[i];
  }
}

/** Stationary density of a 1-D SDE with zero-flux walls, on grid x (array). */
export function stationary(f, g, x, reading = 'ito') {
  const n = x.length, phi = new Float64Array(n), p = new Float64Array(n);
  const drift = (v) => {
    let d = f(v);
    if (reading === 'strat') { const e = 1e-6; d += 0.5 * g(v) * (g(v + e) - g(v - e)) / (2 * e); }
    return d / (0.5 * g(v) ** 2);
  };
  for (let i = 1; i < n; i++) phi[i] = phi[i - 1] + 0.5 * (drift(x[i]) + drift(x[i - 1])) * (x[i] - x[i - 1]);
  const m = Math.max(...phi);
  let s = 0;
  for (let i = 0; i < n; i++) { p[i] = Math.exp(phi[i] - m) / (0.5 * g(x[i]) ** 2); if (i) s += 0.5 * (p[i] + p[i - 1]) * (x[i] - x[i - 1]); }
  for (let i = 0; i < n; i++) p[i] /= s;
  return p;
}

export const gaussPdf = (x, m, v) => Math.exp(-((x - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
