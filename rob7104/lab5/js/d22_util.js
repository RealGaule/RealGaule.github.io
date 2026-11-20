// Helpers of demo d22 (exponential coordinates). Display geometry only: the cloud, the fits and the closed form
// come from sim5.js. This file ports the picture devices of render/s23_expcoords.py:
//   * expDensity: density of q = M_s(alpha) (v1, v2) for (v1, v2, alpha) ~ N(0, Sigma), by quadrature over alpha
//     (class ExpDensity of the clip): s = 0 gives the (x, y) marginal of mu exp(xi) in the frame of the mean
//     (the banana), s = 1 the plane Gaussian of (v1, v2). Its highest-density regions holding 39.3 / 86.5 / 98.9 %
//     are the contours drawn on the floor and in the panel.
//   * contour segments by marching squares, ellipsoid wires and surfaces, Mardia skewness, quantiles.
import * as S from './sim5.js';

export const MASSES = [1 - Math.exp(-0.5), 1 - Math.exp(-2), 1 - Math.exp(-4.5)];   // 1, 2, 3 sigma of a plane Gaussian

const sinc = (a) => (a === 0 ? 1 : Math.sin(a) / a);

/** M_s(alpha) = (1 - s) V(alpha) + s I = [[p, -q], [q, p]]; V maps (v1, v2) to (x, y) in the frame of the mean. */
export function mix(al, s) {
  const A = sinc(al), B = Math.sin(al / 2) * sinc(al / 2);
  return [(1 - s) * A + s, (1 - s) * B];
}

function fibSphere(n) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const z = 1 - 2 * (i + 0.5) / n, r = Math.sqrt(1 - z * z), ph = (i + 0.5) * Math.PI * (3 - Math.sqrt(5));
    out.push([r * Math.cos(ph), r * Math.sin(ph), z]);
  }
  return out;
}
const SPHERE = fibSphere(600);

/** Density grid of the deformed exponential Gaussian (frame of the mean). Returns
 *  { a0, da, na, b0, db, nb, dens (Float64Array nb * na, row = b), levels [3], mass, s }. */
export function expDensity(Sigma, s = 0, { n = 104, nal = 65, kBox = 3.7, kAl = 4.5 } = {}) {
  const Sg = S.symmetrize(Sigma);
  const L = S.gaussianFactor(Sg);
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const u of SPHERE) {
    const v = S.mulVec(L, [kBox * u[0], kBox * u[1], kBox * u[2]]);
    const al = Math.max(-Math.PI, Math.min(Math.PI, v[2]));
    const [p, q] = mix(al, s);
    const qa = p * v[0] - q * v[1], qb = q * v[0] + p * v[1];
    lo = [Math.min(lo[0], qa), Math.min(lo[1], qb)]; hi = [Math.max(hi[0], qa), Math.max(hi[1], qb)];
  }
  const pad = [0.06 * (hi[0] - lo[0]) + 1e-9, 0.06 * (hi[1] - lo[1]) + 1e-9];
  lo = [lo[0] - pad[0], lo[1] - pad[1]]; hi = [hi[0] + pad[0], hi[1] + pad[1]];
  const ratio = (hi[0] - lo[0]) / (hi[1] - lo[1]);
  const na = Math.max(40, Math.min(300, Math.round(n * Math.sqrt(ratio))));
  const nb = Math.max(40, Math.min(300, Math.round(n / Math.sqrt(ratio))));
  const da = (hi[0] - lo[0]) / (na - 1), db = (hi[1] - lo[1]) / (nb - 1);
  const s33 = Sg[2][2], aL = Math.min(kAl * Math.sqrt(s33), Math.PI);
  const c13 = Sg[0][2], c23 = Sg[1][2];
  const C11 = Sg[0][0] - c13 * c13 / s33, C12 = Sg[0][1] - c13 * c23 / s33, C22 = Sg[1][1] - c23 * c23 / s33;
  const dC = C11 * C22 - C12 * C12;
  if (!(dC > 0)) throw new Error('d22: conditional covariance is singular');
  const I11 = C22 / dC, I12 = -C12 / dC, I22 = C11 / dC;
  const dens = new Float64Array(na * nb);
  let wsum = 0;
  const nodes = [];
  for (let k = 0; k < nal; k++) {
    const al = -aL + 2 * aL * k / (nal - 1), w = Math.exp(-0.5 * al * al / s33);
    wsum += w;
    const [p, q] = mix(al, s);
    nodes.push({ w, p, q, det: p * p + q * q, m1: al / s33 * c13, m2: al / s33 * c23 });
  }
  for (const nd of nodes) {
    const c = nd.w / wsum / nd.det;
    for (let j = 0; j < nb; j++) {
      const B = lo[1] + j * db;
      for (let i = 0; i < na; i++) {
        const A = lo[0] + i * da;
        const u1 = (nd.p * A + nd.q * B) / nd.det - nd.m1, u2 = (-nd.q * A + nd.p * B) / nd.det - nd.m2;
        dens[j * na + i] += c * Math.exp(-0.5 * (I11 * u1 * u1 + 2 * I12 * u1 * u2 + I22 * u2 * u2));
      }
    }
  }
  const norm = 1 / (2 * Math.PI * Math.sqrt(dC));
  let tot = 0;
  for (let i = 0; i < dens.length; i++) { dens[i] *= norm; tot += dens[i]; }
  const sorted = Float64Array.from(dens).sort().reverse();
  const levels = [];
  let acc = 0, m = 0;
  for (let i = 0; i < sorted.length && m < MASSES.length; i++) {
    acc += sorted[i];
    while (m < MASSES.length && acc / tot >= MASSES[m]) { levels.push(sorted[i]); m++; }
  }
  while (levels.length < MASSES.length) levels.push(sorted[sorted.length - 1]);
  return { a0: lo[0], da, na, b0: lo[1], db, nb, dens, levels, mass: tot * da * db, s };
}

/** Bilinear density at (qa, qb) in the frame of the mean (0 outside the grid). */
export function densityAt(D, qa, qb) {
  const fi = (qa - D.a0) / D.da, fj = (qb - D.b0) / D.db;
  const i = Math.floor(fi), j = Math.floor(fj);
  if (i < 0 || j < 0 || i >= D.na - 1 || j >= D.nb - 1) return 0;
  const u = fi - i, v = fj - j, d = D.dens, na = D.na, k = j * na + i;
  return (1 - u) * (1 - v) * d[k] + u * (1 - v) * d[k + 1] + (1 - u) * v * d[k + na] + u * v * d[k + na + 1];
}

/** Marching squares: line segments [x1, y1, x2, y2] (frame of the mean) of the level set dens = level. */
export function contourSegments(D, level) {
  const out = [], { dens: d, na, nb, a0, b0, da, db } = D;
  const X = (i) => a0 + i * da, Y = (j) => b0 + j * db;
  for (let j = 0; j < nb - 1; j++) {
    for (let i = 0; i < na - 1; i++) {
      const v0 = d[j * na + i], v1 = d[j * na + i + 1], v2 = d[(j + 1) * na + i + 1], v3 = d[(j + 1) * na + i];
      const code = (v0 >= level) | ((v1 >= level) << 1) | ((v2 >= level) << 2) | ((v3 >= level) << 3);
      if (code === 0 || code === 15) continue;
      const t = (a, b) => (level - a) / (b - a);
      const e = [
        () => [X(i) + t(v0, v1) * da, Y(j)],          // bottom
        () => [X(i + 1), Y(j) + t(v1, v2) * db],      // right
        () => [X(i) + t(v3, v2) * da, Y(j + 1)],      // top
        () => [X(i), Y(j) + t(v0, v3) * db],          // left
      ];
      const seg = (p, q) => { const a = e[p](), b = e[q](); out.push([a[0], a[1], b[0], b[1]]); };
      const centre = 0.25 * (v0 + v1 + v2 + v3) >= level;
      switch (code) {
        case 1: case 14: seg(3, 0); break;
        case 2: case 13: seg(0, 1); break;
        case 3: case 12: seg(3, 1); break;
        case 4: case 11: seg(1, 2); break;
        case 6: case 9: seg(0, 2); break;
        case 7: case 8: seg(3, 2); break;
        case 5: if (centre) { seg(0, 1); seg(2, 3); } else { seg(3, 0); seg(1, 2); } break;
        case 10: if (centre) { seg(3, 0); seg(1, 2); } else { seg(0, 1); seg(2, 3); } break;
        default: break;
      }
    }
  }
  return out;
}

/** Frame of the mean -> display coordinates (as ExpDensity.to_display): rotate by (1 - s) mu_theta, add mu. */
export function toDisplay(segs, mu, s) {
  const ph = (1 - s) * mu[2], c = Math.cos(ph), sn = Math.sin(ph);
  return segs.map(([x1, y1, x2, y2]) => [mu[0] + c * x1 - sn * y1, mu[1] + sn * x1 + c * y1, mu[0] + c * x2 - sn * y2, mu[1] + sn * x2 + c * y2]);
}

/** Wire curves of the unit sphere: `nlat` latitude rings and `nmer` meridians, `n` points each. */
export function sphereWires(nlat = 5, nmer = 6, n = 72) {
  const curves = [];
  for (let k = 1; k <= nlat; k++) {
    const z = -1 + 2 * k / (nlat + 1), r = Math.sqrt(1 - z * z), c = [];
    for (let i = 0; i <= n; i++) { const a = 2 * Math.PI * i / n; c.push([r * Math.cos(a), r * Math.sin(a), z]); }
    curves.push(c);
  }
  for (let k = 0; k < nmer; k++) {
    const ph = Math.PI * k / nmer, c = [];
    for (let i = 0; i <= n; i++) { const a = 2 * Math.PI * i / n; c.push([Math.sin(a) * Math.cos(ph), Math.sin(a) * Math.sin(ph), Math.cos(a)]); }
    curves.push(c);
  }
  return curves;
}

/** Mardia's multivariate skewness b1 = (1/N^2) sum_ij (d_i' S^-1 d_j)^3 (S with divisor N) of N points in 3D. */
export function mardiaSkew3(Z, N = (Z.length / 3) | 0) {
  const m = [0, 0, 0];
  for (let i = 0; i < N; i++) for (let a = 0; a < 3; a++) m[a] += Z[3 * i + a] / N;
  const C = S.zeros(3, 3);
  for (let i = 0; i < N; i++) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) C[a][b] += (Z[3 * i + a] - m[a]) * (Z[3 * i + b] - m[b]) / N;
  const L = S.cholesky(C);
  if (!L) return NaN;
  const W = S.inv(L), Y = new Float64Array(3 * N);
  for (let i = 0; i < N; i++) {
    const d = [Z[3 * i] - m[0], Z[3 * i + 1] - m[1], Z[3 * i + 2] - m[2]];
    for (let a = 0; a < 3; a++) Y[3 * i + a] = W[a][0] * d[0] + W[a][1] * d[1] + W[a][2] * d[2];
  }
  let sum = 0;
  for (let i = 0; i < N; i++) {
    const y0 = Y[3 * i], y1 = Y[3 * i + 1], y2 = Y[3 * i + 2];
    sum += (y0 * y0 + y1 * y1 + y2 * y2) ** 3;
    for (let j = i + 1; j < N; j++) { const g = y0 * Y[3 * j] + y1 * Y[3 * j + 1] + y2 * Y[3 * j + 2]; sum += 2 * g * g * g; }
  }
  return sum / (N * N);
}

/** q-quantile (linear interpolation) of the k-th coordinate of interleaved points (stride 3). */
export function quantile3(Z, k, q, N = (Z.length / 3) | 0) {
  const a = new Float64Array(N);
  for (let i = 0; i < N; i++) a[i] = Z[3 * i + k];
  a.sort();
  const h = q * (N - 1), i = Math.floor(h), f = h - i;
  return i + 1 < N ? a[i] + f * (a[i + 1] - a[i]) : a[i];
}
