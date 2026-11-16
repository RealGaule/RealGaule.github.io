// Helpers shared by the drone demos d11b, d12, d13, d22: a few numerics that sim.js does not have
// (polynomial roots, eigenvalues of the damped double pendulum, matrix 2-norm, frequency response),
// chain helpers for the 3D viewer and a small axis frame for the 2D theory pictures.
// The numbers are checked against numpy/scipy (see the report of the drone demos).
import * as S from './sim.js';
import { THREE, CSS } from './common.js';

// ---------------------------------------------------------------- complex numbers as [re, im]
const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const csub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const cadd = (a, b) => [a[0] + b[0], a[1] + b[1]];
function cdiv(a, b) {
  const d = b[0] * b[0] + b[1] * b[1];
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
}
export const cabs = (a) => Math.hypot(a[0], a[1]);

/**
 * All roots of c[0] + c[1] s + ... + c[n] s^n (real coefficients, c[n] != 0) by the Durand-Kerner iteration,
 * each polished with a few Newton steps. Returns [re, im] pairs sorted by imaginary part (largest first),
 * then by real part.
 */
export function polyRoots(c) {
  const n = c.length - 1;
  const a = c.map((v) => v / c[n]);
  const horner = (z) => { let p = [1, 0]; for (let k = n - 1; k >= 0; k--) p = cadd(cmul(p, z), [a[k], 0]); return p; };
  let bound = 0;
  for (let k = 0; k < n; k++) bound = Math.max(bound, Math.abs(a[k]));
  bound = 1 + bound;
  let rad = 0;
  for (let k = 0; k < n; k++) rad = Math.max(rad, Math.pow(Math.abs(a[k]), 1 / (n - k)));
  rad = Math.min(bound, 2 * rad) || 1;
  let z = Array.from({ length: n }, (_, k) => { const ang = 2 * Math.PI * k / n + 0.4; return [rad * Math.cos(ang), rad * Math.sin(ang)]; });
  for (let it = 0; it < 400; it++) {
    let move = 0;
    for (let i = 0; i < n; i++) {
      let den = [1, 0];
      for (let j = 0; j < n; j++) if (j !== i) den = cmul(den, csub(z[i], z[j]));
      if (cabs(den) < 1e-300) continue;
      const dz = cdiv(horner(z[i]), den);
      z[i] = csub(z[i], dz);
      move = Math.max(move, cabs(dz));
    }
    if (move < 1e-14 * rad) break;
  }
  const deriv = (zz) => { let p = [n, 0]; for (let k = n - 1; k >= 1; k--) p = cadd(cmul(p, zz), [k * a[k], 0]); return p; };
  z = z.map((zz) => {
    for (let it = 0; it < 3; it++) {
      const d = deriv(zz);
      if (cabs(d) < 1e-9 * Math.pow(rad, n - 1)) break;          // (nearly) repeated root: leave it
      zz = csub(zz, cdiv(horner(zz), d));
    }
    if (Math.abs(zz[1]) < 1e-9 * rad) zz = [zz[0], 0];
    return zz;
  });
  return z.sort((p, q) => (q[1] - p[1]) || (q[0] - p[0]));
}

/** Coefficients (low order first) of det(M s^2 + C s + K) for 2 x 2 matrices: a quartic in s. */
export function dpCharPoly({ M, C, K }) {
  const q = (i, j) => [K[i][j], C[i][j], M[i][j]];
  const pm = (a, b) => { const r = new Array(a.length + b.length - 1).fill(0); a.forEach((x, i) => b.forEach((y, j) => { r[i + j] += x * y; })); return r; };
  const p = pm(q(0, 0), q(1, 1)), m = pm(q(0, 1), q(1, 0));
  return p.map((v, i) => v - m[i]);
}

/** The four eigenvalues of A = [[0, I], [-M^-1 K, -M^-1 C]] as [re, im] pairs. */
export const dpEigs = (mats) => polyRoots(dpCharPoly(mats));

/** Largest singular value of a real matrix: square root of the largest eigenvalue of A^T A (cyclic Jacobi). */
export function norm2(A) {
  const n = A[0].length;
  const B = S.mul(S.transpose(A), A);
  for (let sweep = 0; sweep < 12; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += B[p][q] * B[p][q];
    let tr = 0;
    for (let p = 0; p < n; p++) tr += B[p][p] * B[p][p];
    if (off < 1e-26 * tr) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(B[p][q]) < 1e-300) continue;
      const th = (B[q][q] - B[p][p]) / (2 * B[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const x = B[k][p], y = B[k][q]; B[k][p] = c * x - s * y; B[k][q] = s * x + c * y; }
      for (let k = 0; k < n; k++) { const x = B[p][k], y = B[q][k]; B[p][k] = c * x - s * y; B[q][k] = s * x + c * y; }
    }
  }
  let big = 0;
  for (let p = 0; p < n; p++) big = Math.max(big, B[p][p]);
  return Math.sqrt(Math.max(big, 0));
}

/** Singular values s1 >= s2 of a 2 x 2 matrix and the unit direction u1 of the longest semi-axis of its image. */
export function svd2(E) {
  const a = E[0][0] * E[0][0] + E[0][1] * E[0][1];           // E E^T = [[a, b], [b, d]]
  const b = E[0][0] * E[1][0] + E[0][1] * E[1][1];
  const d = E[1][0] * E[1][0] + E[1][1] * E[1][1];
  const mean = (a + d) / 2, dev = Math.hypot((a - d) / 2, b);
  const ang = 0.5 * Math.atan2(2 * b, a - d);
  return { s1: Math.sqrt(mean + dev), s2: Math.sqrt(Math.max(mean - dev, 0)), u1: [Math.cos(ang), Math.sin(ang)] };
}

/**
 * Steady-state response of M th'' + C th' + K th = -h x''(t) to x = X sin(w t), per metre of X:
 * th(t) = Im(Th e^{i w t}) with (K - w^2 M + i w C) Th = w^2 h. Returns [Th1, Th2] as [re, im] pairs.
 */
export function freqResp({ M, C, K, h }, w) {
  const e = (i, j) => [K[i][j] - w * w * M[i][j], w * C[i][j]];
  const a = e(0, 0), b = e(0, 1), c = e(1, 0), d = e(1, 1);
  const det = csub(cmul(a, d), cmul(b, c));
  const r = [[w * w * h[0], 0], [w * w * h[1], 0]];
  return [cdiv(csub(cmul(d, r[0]), cmul(b, r[1])), det), cdiv(csub(cmul(a, r[1]), cmul(c, r[0])), det)];
}

// ---------------------------------------------------------------- the slung load in the viewer
/** Sideways and vertical offsets of crate and parcel from the pivot, for absolute link angles th. */
export function loadXZ(th, p = S.P) {
  const x1 = p.l1 * Math.sin(th[0]), z1 = -p.l1 * Math.cos(th[0]);
  return { crate: [x1, z1], parcel: [x1 + p.l2 * Math.sin(th[1]), z1 - p.l2 * Math.cos(th[1])] };
}

/** Write absolute link angles to a two-link chain (hinges hold relative angles). */
export function setChain(v, name, th) { v.setJoint(`${name}_q1`, th[0]); v.setJoint(`${name}_q2`, th[1] - th[0]); }

/** Geom ids of all bodies whose name starts with `<chain>_`. */
export function chainGeoms(v, chain) {
  const out = [];
  for (let g = 0; g < v.model.ngeom; g++) {
    const name = v.mujoco.mj_id2name(v.model, v.mujoco.mjtObj.mjOBJ_BODY.value, v.model.geom_bodyid[g]) || '';
    if (name.startsWith(chain + '_')) out.push(g);
  }
  return out;
}

export function showChain(v, chain, on) { chainGeoms(v, chain).forEach((g) => { if (v.meshes[g]) v.meshes[g].visible = on; }); }

/**
 * Turn one of the ghost chains of drone.mjb into a translucent "linear model" halo: recolour it and make its
 * two spheres large enough to wrap the solid crate and parcel, so that model and robot can be told apart
 * only when they disagree.
 */
export function haloChain(v, chain, color, radii = [0.135, 0.125], opacity = 0.32) {
  const T = v.mujoco.mjtGeom;
  let k = 0;
  chainGeoms(v, chain).forEach((g) => {
    const mesh = v.meshes[g];
    if (!mesh) return;
    const ball = v.model.geom_type[g] === T.mjGEOM_SPHERE.value;
    mesh.material.color.set(color);
    mesh.material.emissive = new THREE.Color(color);
    mesh.material.emissiveIntensity = ball ? 0.35 : 0.6;
    mesh.material.opacity = ball ? opacity : 0.6;
    mesh.renderOrder = 2;
    if (ball && k < radii.length) {
      mesh.geometry.dispose();
      mesh.geometry = new THREE.SphereGeometry(radii[k++], 32, 20);
    }
  });
}

/**
 * The shared site script only watches canvases that carry data-live when the page loads; a demo canvas gets it
 * later, so the viewer would keep drawing while scrolled out of view. Watch it here instead.
 */
export function runWhenVisible(panel, v) {
  if (!('IntersectionObserver' in window)) return;
  new IntersectionObserver((entries) => entries.forEach((en) => {
    if (en.isIntersecting && panel.classList.contains('active')) v.start(); else v.stop();
  }), { threshold: 0.05 }).observe(v.canvas);
}

/** A straight line between two moving points (for the gap between robot and model). */
export class Link3 {
  constructor(scene, color) {
    this.pos = new Float32Array(6);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true }));
    this.line.frustumCulled = false;
    this.line.renderOrder = 5;
    scene.add(this.line);
  }
  set(a, b) { this.pos.set(a, 0); this.pos.set(b, 3); this.line.geometry.attributes.position.needsUpdate = true; }
}

// ---------------------------------------------------------------- 2D pictures
export const FONT = '500 11px Inter, system-ui, sans-serif';

/** Axis frame inside a plot2d canvas: data (x, y) -> pixels, with ticks, polylines, dots and labels. */
export class Frame {
  constructor(g, left, top, width, height, xr, yr) {
    Object.assign(this, { g, left, top, width, height, xr, yr });
  }
  X(x) { return this.left + (x - this.xr[0]) / (this.xr[1] - this.xr[0]) * this.width; }
  Y(y) { return this.top + this.height - (y - this.yr[0]) / (this.yr[1] - this.yr[0]) * this.height; }
  clip(fn) {
    const g = this.g;
    g.save(); g.beginPath(); g.rect(this.left, this.top, this.width, this.height); g.clip();
    fn(); g.restore();
  }
  seg(x0, y0, x1, y1, color, width = 1, dash = []) {
    const g = this.g;
    g.save(); g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash);
    g.beginPath(); g.moveTo(this.X(x0), this.Y(y0)); g.lineTo(this.X(x1), this.Y(y1)); g.stroke(); g.restore();
  }
  line(xs, ys, color, width = 1.5, dash = []) {
    const g = this.g;
    g.save(); g.strokeStyle = color; g.lineWidth = width; g.lineJoin = 'round'; g.setLineDash(dash);
    g.beginPath();
    for (let i = 0; i < xs.length; i++) { if (i) g.lineTo(this.X(xs[i]), this.Y(ys[i])); else g.moveTo(this.X(xs[i]), this.Y(ys[i])); }
    g.stroke(); g.restore();
  }
  dot(x, y, r, fill, stroke = null, lw = 1.5) {
    const g = this.g;
    g.beginPath(); g.arc(this.X(x), this.Y(y), r, 0, 2 * Math.PI);
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.strokeStyle = stroke; g.lineWidth = lw; g.stroke(); }
  }
  /** Left and bottom axis lines with tick labels. xt, yt: arrays of [value, label]. */
  axes(xt, yt, xlabel, ylabel) {
    const g = this.g, b = this.top + this.height;
    g.save();
    g.strokeStyle = CSS.faint; g.lineWidth = 1;
    g.beginPath(); g.moveTo(this.left, this.top); g.lineTo(this.left, b); g.lineTo(this.left + this.width, b); g.stroke();
    g.font = FONT; g.fillStyle = CSS.grey;
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (const [x, s] of xt) {
      g.beginPath(); g.moveTo(this.X(x), b); g.lineTo(this.X(x), b + 4); g.stroke();
      g.fillText(s, this.X(x), b + 6);
    }
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const [y, s] of yt) {
      g.beginPath(); g.moveTo(this.left - 4, this.Y(y)); g.lineTo(this.left, this.Y(y)); g.stroke();
      g.fillText(s, this.left - 7, this.Y(y));
    }
    g.fillStyle = CSS.white;
    if (xlabel) { g.textAlign = 'center'; g.textBaseline = 'top'; g.fillText(xlabel, this.left + this.width / 2, b + 21); }
    if (ylabel) {
      g.translate(this.left - 34, this.top + this.height / 2); g.rotate(-Math.PI / 2);
      g.textAlign = 'center'; g.textBaseline = 'bottom'; g.fillText(ylabel, 0, 0);
    }
    g.restore();
  }
  text(s, px, py, color = CSS.grey, align = 'left', base = 'alphabetic', font = FONT) {
    const g = this.g;
    g.save(); g.font = font; g.fillStyle = color; g.textAlign = align; g.textBaseline = base; g.fillText(s, px, py); g.restore();
  }
}
