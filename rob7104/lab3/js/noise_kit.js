// Helpers for the Part 1-2 demos (d11, d12, d21): thick curves on density walls, labels that follow 3-D points,
// small 2-D axes (linear or log), and a few numbers (erf, bin-averaged Gaussian, least-squares slope).
import { THREE, CSS } from './common3.js';

export const FONT = '11px Inter, system-ui, sans-serif';
export const FONT_B = '600 12px Inter, system-ui, sans-serif';

// ---------------------------------------------------------------- scene helpers
/** Lamp fixtures hang high above the floor; their shadows would litter the floor between the corridors. */
export function noLampShadows(v) {
  for (let g = 0; g < v.model.ngeom; g++) if (v.meshes[g] && v.data.geom_xpos[3 * g + 2] > 0.5) v.meshes[g].castShadow = false;
}

/** Hide the end walls of the corridor model(s), so robots can leave through the ends (d11). */
export function hideEndWalls(v) {
  const s = v.model.geom_size;
  for (let g = 0; g < v.model.ngeom; g++) {
    if (Math.abs(s[3 * g] - 0.01) < 1e-6 && s[3 * g + 1] > 0.1 && v.meshes[g]) v.meshes[g].visible = false;
  }
}

/** A thick curve (flat ribbon) in the vertical plane y = y0. WebGL lines are 1 px wide; this is not. */
export class Ribbon {
  constructor(scene, color, { width = 0.012, dash = 0, opacity = 1 } = {}) {
    this.width = width; this.dash = dash;
    this.geo = new THREE.BufferGeometry();
    this.mat = new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, toneMapped: false, transparent: opacity < 1, opacity, depthWrite: opacity >= 1 });
    this.mesh = new THREE.Mesh(this.geo, this.mat);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 2;
    scene.add(this.mesh);
  }
  /** Points (xs[i], zs[i]); keep[i] === false drops the piece from point i to i + 1 (gaps, clipping). */
  set(xs, zs, y0, keep = null) {
    const n = xs.length, pos = new Float32Array(6 * n), idx = [], hw = this.width / 2;
    let arc = 0;
    for (let i = 0; i < n; i++) {
      const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
      let tx = xs[b] - xs[a], tz = zs[b] - zs[a];
      const l = Math.hypot(tx, tz) || 1; tx /= l; tz /= l;
      pos.set([xs[i] + tz * hw, y0, zs[i] - tx * hw, xs[i] - tz * hw, y0, zs[i] + tx * hw], 6 * i);
      if (i < n - 1) {
        const seg = Math.hypot(xs[i + 1] - xs[i], zs[i + 1] - zs[i]), mid = arc + seg / 2;
        arc += seg;
        if (keep && keep[i] === false) continue;
        if (this.dash > 0 && Math.floor(mid / this.dash) % 2 === 1) continue;
        const k = 2 * i;
        idx.push(k, k + 1, k + 2, k + 1, k + 3, k + 2);
      }
    }
    this.geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.geo.setIndex(idx);
    this.geo.attributes.position.needsUpdate = true;
  }
  set visible(on) { this.mesh.visible = on; }
}

/** HTML labels pinned to 3-D points, drawn over the canvas (they follow the orbiting camera). */
export class Tags {
  constructor(cell, v) { this.cell = cell; this.v = v; this.items = []; this._p = new THREE.Vector3(); }
  add(pos, { anchor = 'left', valign = 'bottom', size = 12 } = {}) {
    const el = document.createElement('div');
    el.style.cssText = 'position:absolute;left:0;top:0;pointer-events:none;white-space:nowrap;padding:3px 8px;border-radius:3px;'
      + `background:rgba(8,9,12,0.66);color:${CSS.white};font:${size}px/1.45 var(--mono, monospace);z-index:2;`;
    this.cell.appendChild(el);
    const it = { el, pos: new THREE.Vector3(...pos), anchor, valign, html: '' };
    this.items.push(it);
    return it;
  }
  text(it, html) { if (it.html !== html) { it.el.innerHTML = html; it.html = html; } }
  /** Place every label; a label that would cover one placed before it is hidden (seen from odd orbit angles). */
  update() {
    const c = this.v.canvas, w = c.clientWidth, h = c.clientHeight, placed = [];
    for (const it of this.items) {
      const p = this._p.copy(it.pos).project(this.v.camera);
      const off = p.z > 1 || p.x < -1.3 || p.x > 1.3 || p.y < -1.3 || p.y > 1.3;
      it.el.style.display = off ? 'none' : 'block';
      if (off) continue;
      let px = (p.x + 1) / 2 * w, py = (1 - p.y) / 2 * h;
      const ew = it.el.offsetWidth, eh = it.el.offsetHeight;
      if (it.anchor === 'right') px -= ew;
      else if (it.anchor === 'centre') px -= ew / 2;
      py -= it.valign === 'middle' ? eh / 2 : it.valign === 'top' ? 0 : eh;
      px = Math.max(4, Math.min(w - ew - 4, px)); py = Math.max(4, Math.min(h - eh - 4, py));
      if (placed.some((r) => px < r[2] && px + ew > r[0] && py < r[3] && py + eh > r[1])) { it.el.style.display = 'none'; continue; }
      placed.push([px, py, px + ew, py + eh]);
      it.el.style.transform = `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px)`;
    }
  }
}

// ---------------------------------------------------------------- 2-D axes (linear or log10)
export class Ax {
  constructor(g, x, y, w, h, xr, yr, { xlog = false, ylog = false } = {}) {
    Object.assign(this, { g, x, y, w, h, xr, yr, xlog, ylog });
  }
  _u(v, r, log) { return log ? (Math.log10(v) - Math.log10(r[0])) / (Math.log10(r[1]) - Math.log10(r[0])) : (v - r[0]) / (r[1] - r[0]); }
  X(v) { return this.x + this._u(v, this.xr, this.xlog) * this.w; }
  Y(v) { return this.y + this.h - this._u(v, this.yr, this.ylog) * this.h; }
  invX(px) {
    const u = (px - this.x) / this.w;
    return this.xlog ? 10 ** (Math.log10(this.xr[0]) + u * (Math.log10(this.xr[1]) - Math.log10(this.xr[0]))) : this.xr[0] + u * (this.xr[1] - this.xr[0]);
  }
  frame(xt = [], yt = [], { xlab = '', ylab = '' } = {}) {
    const g = this.g;
    g.save();
    g.lineWidth = 1; g.strokeStyle = CSS.faint; g.fillStyle = CSS.grey; g.font = FONT;
    g.beginPath();
    g.moveTo(this.x, this.y); g.lineTo(this.x, this.y + this.h); g.lineTo(this.x + this.w, this.y + this.h);
    if (!this.ylog && this.yr[0] < 0 && this.yr[1] > 0) { g.moveTo(this.x, this.Y(0)); g.lineTo(this.x + this.w, this.Y(0)); }
    g.stroke();
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const [v, s] of yt) {
      g.beginPath(); g.moveTo(this.x - 3, this.Y(v)); g.lineTo(this.x, this.Y(v)); g.stroke();
      g.fillText(s, this.x - 5, this.Y(v));
    }
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (const [v, s] of xt) {
      g.beginPath(); g.moveTo(this.X(v), this.y + this.h); g.lineTo(this.X(v), this.y + this.h + 3); g.stroke();
      if (s) g.fillText(s, this.X(v), this.y + this.h + 5);
    }
    if (xlab) { g.textAlign = 'right'; g.textBaseline = 'bottom'; g.fillText(xlab, this.x + this.w, this.y + this.h - 3); }
    if (ylab) { g.textAlign = 'left'; g.textBaseline = 'top'; g.fillText(ylab, this.x + 5, this.y + 1); }
    g.restore();
  }
  clip(fn) {
    const g = this.g;
    g.save(); g.beginPath(); g.rect(this.x, this.y - 1, this.w, this.h + 2); g.clip(); fn(); g.restore();
  }
  line(xs, ys, color, width = 1.5, { dash = null, alpha = 1, i0 = 0, i1 = xs.length } = {}) {
    const g = this.g;
    if (i1 - i0 < 2) return;
    g.save();
    g.strokeStyle = color; g.lineWidth = width; g.globalAlpha = alpha; g.lineJoin = 'round';
    if (dash) g.setLineDash(dash);
    g.beginPath(); g.moveTo(this.X(xs[i0]), this.Y(ys[i0]));
    for (let i = i0 + 1; i < i1; i++) g.lineTo(this.X(xs[i]), this.Y(ys[i]));
    g.stroke(); g.restore();
  }
  dot(x, y, r, color, { ring = false } = {}) {
    const g = this.g;
    g.save(); g.beginPath(); g.arc(this.X(x), this.Y(y), r, 0, 2 * Math.PI);
    if (ring) { g.strokeStyle = color; g.lineWidth = 1.5; g.stroke(); } else { g.fillStyle = color; g.fill(); }
    g.restore();
  }
  vline(x, color, width = 2, alpha = 1) {
    const g = this.g;
    g.save(); g.strokeStyle = color; g.lineWidth = width; g.globalAlpha = alpha;
    g.beginPath(); g.moveTo(this.X(x), this.y); g.lineTo(this.X(x), this.y + this.h); g.stroke(); g.restore();
  }
}

export function label(g, s, x, y, color = CSS.white, align = 'left', font = FONT_B) {
  g.save(); g.font = font; g.fillStyle = color; g.textAlign = align; g.textBaseline = 'alphabetic'; g.fillText(s, x, y); g.restore();
}

// ---------------------------------------------------------------- numbers
/** erf, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7). */
export function erf(x) {
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
export const normCdf = (x, m, sd) => 0.5 * (1 + erf((x - m) / (sd * Math.SQRT2)));

/** Bin averages of N(m, sd^2) on equal bins of [lo, hi]. */
export function binGauss(lo, hi, nb, m, sd) {
  const w = (hi - lo) / nb, out = new Float64Array(nb);
  let c0 = normCdf(lo, m, sd);
  for (let k = 0; k < nb; k++) { const c1 = normCdf(lo + (k + 1) * w, m, sd); out[k] = (c1 - c0) / w; c0 = c1; }
  return out;
}

/** Bin averages of an unnormalised density q on [lo, hi], normalised to integrate to 1 there (fine trapezoid). */
export function binDensity(q, lo, hi, nb, sub = 40) {
  const w = (hi - lo) / nb, h = w / sub, out = new Float64Array(nb);
  let tot = 0;
  for (let k = 0; k < nb; k++) {
    let s = 0;
    for (let j = 0; j < sub; j++) { const a = lo + k * w + j * h; s += 0.5 * (q(a) + q(a + h)) * h; }
    out[k] = s / w; tot += s;
  }
  for (let k = 0; k < nb; k++) out[k] /= tot;
  return { bins: out, total: tot };
}

/** Least-squares slope of log y against log x. */
export function logSlope(xs, ys) {
  const n = xs.length; let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { const a = Math.log(xs[i]), b = Math.log(ys[i]); sx += a; sy += b; sxx += a * a; sxy += a * b; }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

export function median(a) {
  const b = Array.from(a).sort((p, q) => p - q), n = b.length;
  return n % 2 ? b[(n - 1) / 2] : 0.5 * (b[n / 2 - 1] + b[n / 2]);
}

/** Format a number with 3 significant digits, no exponent for the ranges used here. */
export function sig3(v) {
  if (v === 0 || !isFinite(v)) return String(v);
  const d = Math.max(0, 2 - Math.floor(Math.log10(Math.abs(v))));
  return v.toFixed(Math.min(d, 6));
}
