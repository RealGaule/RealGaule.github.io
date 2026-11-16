// Helpers for the Lab 3 demos d31, d41, d51, d71 (Parts 3-7): labels anchored to 3-D points, thick curves on a
// density wall, a robot y-offset generator, bin averages, and small 2-D axes (linear or log) for the theory plots.
import { THREE, CSS } from './common3.js';

let styled = false;
function injectStyle() {
  if (styled) return;
  styled = true;
  const s = document.createElement('style');
  s.textContent = `
.dens-ov { position: absolute; inset: 0; pointer-events: none; overflow: hidden; }
.dens-ov .lab { position: absolute; white-space: nowrap; font-family: var(--mono, monospace); font-size: 12px; line-height: 1.45;
  color: #f2efe9; text-shadow: 0 0 4px #000, 0 0 2px #000; }
.dens-ov .lab b { font-weight: 600; }
.dens-ov .legend { position: absolute; left: 12px; bottom: 10px; padding: 5px 9px; border-radius: 4px; background: rgba(8, 9, 12, 0.66);
  font-family: var(--sans, sans-serif); font-size: 11.5px; line-height: 1.6; color: #9a97a3; }
.dens-ov .legend i { display: inline-block; width: 16px; height: 3px; margin: 0 7px 3px 0; vertical-align: middle; border-radius: 1px; }
.dens-ov .legend i.bar { height: 9px; opacity: 0.75; }
.dens-ov .legend i.dash { background-image: linear-gradient(90deg, currentColor 55%, transparent 45%) !important; background-size: 6px 3px; }
.dens-ov .corner { position: absolute; right: 12px; top: 10px; padding: 4px 9px; border-radius: 4px; background: rgba(8, 9, 12, 0.66);
  font-family: var(--mono, monospace); font-size: 12px; color: #9a97a3; text-align: right; line-height: 1.55; }
.dens-ov .corner b { color: #f2efe9; font-weight: 600; }
.controls .readout .pk { color: ${CSS.pink}; } .controls .readout .cy { color: ${CSS.cyan}; }
.controls .readout .bl { color: ${CSS.blue}; } .controls .readout .ye { color: ${CSS.yellow}; }
.controls .readout .gr { color: ${CSS.green}; } .controls .readout .rd { color: ${CSS.red}; }
.c-pink { color: ${CSS.pink}; } .c-cyan { color: ${CSS.cyan}; }
.dens-wide .controls .readout { flex-basis: 100%; margin-left: 0; }
`;
  document.head.appendChild(s);
}

/** HTML labels over the 3-D canvas, anchored to world points and re-placed every frame (they follow the orbit). */
export class Overlay {
  constructor(panel, v, cell = panel.querySelector('canvas.view3d').parentElement) {
    injectStyle();
    this.v = v;
    this.canvas = cell.querySelector('canvas.view3d');
    this.root = document.createElement('div');
    this.root.className = 'dens-ov';
    cell.appendChild(this.root);
    this.items = [];
    this._p = new THREE.Vector3();
  }
  /** A label at world point p; anchor 'c' (centre, above), 'cb' (centre, below), 'l' (left edge), 'r' (right edge). */
  label(p, anchor = 'c') {
    const el = document.createElement('div');
    el.className = 'lab';
    this.root.appendChild(el);
    const it = { el, p, anchor, html: '' };
    this.items.push(it);
    return {
      set: (html) => { if (html !== it.html) { it.html = html; el.innerHTML = html; } },
      move: (q) => { it.p = q; },
      show: (on) => { el.style.display = on ? '' : 'none'; },
      el,
    };
  }
  /** A fixed box: 'legend' (bottom left) or 'corner' (top right). */
  box(kind, html = '') {
    const el = document.createElement('div');
    el.className = kind;
    el.innerHTML = html;
    this.root.appendChild(el);
    let cur = html;
    return { el, set: (h) => { if (h !== cur) { cur = h; el.innerHTML = h; } } };
  }
  update() {
    const cam = this.v.camera, w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    for (const it of this.items) {
      this._p.set(it.p[0], it.p[1], it.p[2]).project(cam);
      if (this._p.z > 1) { it.el.style.visibility = 'hidden'; continue; }
      it.el.style.visibility = '';
      const x = (this._p.x + 1) / 2 * w, y = (1 - this._p.y) / 2 * h;
      const tx = it.anchor === 'l' ? '0' : it.anchor === 'r' ? '-100%' : '-50%';
      it.el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) translate(${tx}, ${it.anchor === 'cb' ? '0' : '-100%'})`;
    }
  }
}

export const legendItem = (colour, text, kind = 'line') =>
  `<span style="color:${colour}"><i class="${kind}" style="background:${colour}"></i></span>${text}`;

/**
 * A thick curve (flat ribbon) in the vertical plane y = y0. set(xs, zs) with plain arrays; dashed draws every other
 * stretch of length `dash` (m). Lines in WebGL are 1 px wide, so the density curves are drawn as ribbons.
 */
export class Ribbon {
  constructor(scene, colour, { y0 = 0.3, width = 0.012, dash = 0, maxPts = 1200, opacity = 1 } = {}) {
    Object.assign(this, { y0, width, dash });
    this.pos = new Float32Array(maxPts * 6 * 3);
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.mesh = new THREE.Mesh(this.geo, new THREE.MeshBasicMaterial({ color: colour, side: THREE.DoubleSide, toneMapped: false,
      transparent: opacity < 1, opacity, depthWrite: opacity >= 1 }));
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 2;
    scene.add(this.mesh);
  }
  set(xs, zs) {
    const n = xs.length, P = this.pos, hw = this.width / 2, y = this.y0;
    // vertex normals in the x-z plane
    const nx = new Float32Array(n), nz = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
      const dx = xs[b] - xs[a], dz = zs[b] - zs[a], L = Math.hypot(dx, dz) || 1;
      nx[i] = -dz / L; nz[i] = dx / L;
    }
    let k = 0, run = 0;
    const cap = this.pos.length / 18;
    for (let i = 0; i < n - 1 && k < cap; i++) {
      const seg = Math.hypot(xs[i + 1] - xs[i], zs[i + 1] - zs[i]);
      const on = !this.dash || Math.floor((run + seg / 2) / this.dash) % 2 === 0;
      run += seg;
      if (!on) continue;
      const ax = xs[i] - nx[i] * hw, az = zs[i] - nz[i] * hw, bx = xs[i] + nx[i] * hw, bz = zs[i] + nz[i] * hw;
      const cx = xs[i + 1] - nx[i + 1] * hw, cz = zs[i + 1] - nz[i + 1] * hw, dx = xs[i + 1] + nx[i + 1] * hw, dz = zs[i + 1] + nz[i + 1] * hw;
      P.set([ax, y, az, bx, y, bz, dx, y, dz, ax, y, az, dx, y, dz, cx, y, cz], 18 * k);
      k++;
    }
    this.geo.setDrawRange(0, 6 * k);
    this.geo.attributes.position.needsUpdate = true;
    this.mesh.visible = k > 0;
  }
  show(on) { this.mesh.visible = on; }
}

/** Curve on a DensityWall: density values -> heights in the wall's plane, clipped like the bars. */
export function wallCurve(wall, ribbon, xs, ys) {
  const s = wall.height / wall.vmax, top = wall.height * 1.25, zs = new Array(xs.length);
  for (let i = 0; i < xs.length; i++) zs[i] = wall.z0 + Math.min(Math.max(ys[i], 0) * s, top);
  ribbon.set(xs, zs);
}

/** Step outline ("stairs") of a histogram for the wall. */
export function stairs(lo, hi, h) {
  const nb = h.length, w = (hi - lo) / nb, xs = [], ys = [];
  for (let k = 0; k < nb; k++) { xs.push(lo + k * w, lo + (k + 1) * w); ys.push(h[k], h[k]); }
  return { xs, ys };
}

/** Average a fine grid (cells) down to nb bins; requires cells % nb === 0. */
export function binAverage(p, nb) {
  const r = p.length / nb, out = new Float64Array(nb);
  for (let k = 0; k < nb; k++) { let s = 0; for (let j = 0; j < r; j++) s += p[k * r + j]; out[k] = s / r; }
  return out;
}

/** Positions across the corridor for drawn robots (fixed per robot), uniform in [-half, half]. */
export function robotY(n, R, half = 0.12, y0 = 0) {
  return Float64Array.from({ length: n }, () => y0 + (2 * R.uni() - 1) * half);
}

/** Set a lamp's bulb brightness (0..1) in the viewer: colour of geom bulb{i}. */
export function bulbLevel(v, i, level) {
  let g;
  try { g = v.id('geom', `bulb${i}`); } catch (e) { return; }
  const m = v.meshes[g];
  if (!m) return;
  const l = Math.max(0, Math.min(1, level));
  m.material.color.setRGB(0.25 + 0.75 * l, 0.22 + 0.68 * l, 0.18 + 0.47 * l, THREE.SRGBColorSpace);
  if (m.material.emissive) m.material.emissive.copy(m.material.color);
}

/** Hide all geoms whose world position lies within `test(x, y, z)` (e.g. an unused corridor). */
export function hideGeoms(v, test) {
  const xp = v.data.geom_xpos;
  for (let g = 0; g < v.meshes.length; g++) {
    if (v.meshes[g] && test(xp[3 * g], xp[3 * g + 1], xp[3 * g + 2])) v.meshes[g].visible = false;
  }
}

// ---------------------------------------------------------------- 2-D axes on a plot2d canvas
export const FONT = '11px Inter, system-ui, sans-serif';
export const SMALL = '10px Inter, system-ui, sans-serif';

export function text(g, s, x, y, colour = CSS.white, align = 'left', font = FONT) {
  g.font = font; g.fillStyle = colour; g.textAlign = align; g.textBaseline = 'alphabetic';
  g.fillText(s, x, y);
}

export class Ax {
  constructor(g, x, y, w, h, xr, yr, { xlog = false, ylog = false } = {}) {
    Object.assign(this, { g, x, y, w, h, xr, yr, xlog, ylog });
  }
  _t(v, r, log) { return log ? (Math.log(v) - Math.log(r[0])) / (Math.log(r[1]) - Math.log(r[0])) : (v - r[0]) / (r[1] - r[0]); }
  X(v) { return this.x + this._t(v, this.xr, this.xlog) * this.w; }
  Y(v) { return this.y + this.h - this._t(v, this.yr, this.ylog) * this.h; }
  clip(fn) {
    const g = this.g;
    g.save(); g.beginPath(); g.rect(this.x, this.y, this.w, this.h); g.clip(); fn(); g.restore();
  }
  line(xs, ys, colour, lw = 1.5, dash = []) {
    const g = this.g;
    g.strokeStyle = colour; g.lineWidth = lw; g.setLineDash(dash); g.beginPath();
    let started = false;
    for (let i = 0; i < xs.length; i++) {
      if (!Number.isFinite(ys[i]) || (this.ylog && ys[i] <= 0)) { started = false; continue; }
      const px = this.X(xs[i]), py = this.Y(ys[i]);
      if (!started) { g.moveTo(px, py); started = true; } else g.lineTo(px, py);
    }
    g.stroke(); g.setLineDash([]);
  }
  dots(xs, ys, colour, r = 3.5, hollow = false) {
    const g = this.g;
    for (let i = 0; i < xs.length; i++) {
      if (!Number.isFinite(ys[i]) || (this.ylog && ys[i] <= 0)) continue;
      g.beginPath(); g.arc(this.X(xs[i]), this.Y(ys[i]), r, 0, 2 * Math.PI);
      if (hollow) { g.strokeStyle = colour; g.lineWidth = 1.8; g.stroke(); } else { g.fillStyle = colour; g.fill(); }
    }
  }
  vline(v, colour, dash = [], lw = 1) { this.line([v, v], [this.ylog ? this.yr[0] : this.yr[0], this.yr[1]], colour, lw, dash); }
  hline(v, colour, dash = [], lw = 1) { this.line([this.xr[0], this.xr[1]], [v, v], colour, lw, dash); }
  frame(xt = [], yt = [], { grid = true } = {}) {
    const g = this.g;
    if (grid) {
      g.strokeStyle = 'rgba(242, 239, 233, 0.07)'; g.lineWidth = 1;
      for (const [v] of xt) { const p = Math.round(this.X(v)) + 0.5; g.beginPath(); g.moveTo(p, this.y); g.lineTo(p, this.y + this.h); g.stroke(); }
      for (const [v] of yt) { const p = Math.round(this.Y(v)) + 0.5; g.beginPath(); g.moveTo(this.x, p); g.lineTo(this.x + this.w, p); g.stroke(); }
    }
    g.strokeStyle = 'rgba(242, 239, 233, 0.35)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(this.x + 0.5, this.y); g.lineTo(this.x + 0.5, this.y + this.h + 0.5); g.lineTo(this.x + this.w, this.y + this.h + 0.5); g.stroke();
    for (const [v, s] of xt) text(g, s, this.X(v), this.y + this.h + 14, CSS.grey, 'center', SMALL);
    for (const [v, s] of yt) text(g, s, this.x - 5, this.Y(v) + 3.5, CSS.grey, 'right', SMALL);
  }
}
