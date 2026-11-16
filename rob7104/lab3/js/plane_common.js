// Helpers for the 2-D arena demos d81, d91, d92 (owned by those demos; not shared with the rest of the page).
// Flat thick lines on the table (three.js lines are 1 px wide in WebGL), floor text labels, a HUD box,
// and a small 2-D plotting pane on top of common3's plot2d.
import { THREE, CSS } from './common3.js';

// ---------------------------------------------------------------- HUD style (injected once; lab3.css is not ours)
let styled = false;
export function hudStyle() {
  if (styled) return;
  styled = true;
  const s = document.createElement('style');
  s.textContent = `
.pl-hud { position: absolute; padding: 6px 10px; border-radius: 4px; background: rgba(8, 9, 12, 0.74);
  color: var(--mute, #9a97a3); font-family: var(--mono, monospace); font-size: 11.5px; line-height: 1.55; pointer-events: none; }
.pl-hud:empty { display: none; }
.pl-hud b { color: var(--ink, #f2efe9); font-weight: 600; }
.pl-hud.tr { top: 10px; right: 10px; text-align: right; }
.pl-hud.bl { left: 10px; bottom: 10px; max-width: calc(100% - 20px); }
.pl-sw { display: inline-block; width: 16px; height: 0; border-top: 2px solid; vertical-align: middle; margin-right: 5px; }
.pl-sw.dash { border-top-style: dashed; }
.pl-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; vertical-align: middle; margin-right: 5px; }
.scene .duo.pl-duo .cell + .cell { margin-top: 0; }
.scene .duo.pl-duo .cell canvas { height: 440px; min-height: 0; }
@media (max-width: 820px) { .scene .duo.pl-duo .cell canvas { height: 320px; } }
.controls.pl-sub { margin-top: 6px; }
.controls button:disabled, .controls input:disabled { opacity: 0.45; cursor: default; }
`;
  document.head.appendChild(s);
}
export function hud(panel, name, html) {
  const el = panel.querySelector(`[data-hud="${name}"]`);
  if (el && el.innerHTML !== html) el.innerHTML = html;
}
export const sw = (col, dash = false) => `<span class="pl-sw${dash ? ' dash' : ''}" style="border-color:${col}"></span>`;
export const dot = (col) => `<span class="pl-dot" style="background:${col}"></span>`;

// ---------------------------------------------------------------- flat thick polylines on the table
/** A ribbon of constant width lying in the plane z = const. set(xs, ys, {closed, dash}) rebuilds it. */
export class FlatLine {
  constructor(scene, color, { width = 0.012, z = 0.03, maxPts = 720, opacity = 1 } = {}) {
    this.width = width; this.z = z; this.max = maxPts;
    this.pos = new Float32Array(maxPts * 6 * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    geo.setDrawRange(0, 0);
    this.mat = new THREE.MeshBasicMaterial({ color, transparent: opacity < 1, opacity, depthWrite: false, side: THREE.DoubleSide, toneMapped: false });
    this.mesh = new THREE.Mesh(geo, this.mat);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 5;
    scene.add(this.mesh);
  }
  set visible(b) { this.mesh.visible = b; }
  get visible() { return this.mesh.visible; }
  /** dash: length of dash and of gap (m), 0 for solid. */
  set(xs, ys, { closed = false, dash = 0 } = {}) {
    const n = Math.min(xs.length, this.max), w = this.width / 2, z = this.z, P = this.pos;
    let k = 0, s = 0;
    const nseg = closed ? n : n - 1;
    for (let i = 0; i < nseg; i++) {
      const j = (i + 1) % n;
      const x0 = xs[i], y0 = ys[i], x1 = xs[j], y1 = ys[j];
      const L = Math.hypot(x1 - x0, y1 - y0);
      if (L < 1e-9) continue;
      const mid = s + L / 2; s += L;
      if (dash > 0 && Math.floor(mid / dash) % 2 === 1) continue;
      // extend each segment by w/2 at both ends so that joints close
      const ux = (x1 - x0) / L, uy = (y1 - y0) / L, e = Math.min(w, L) * 0.5;
      const ax = x0 - ux * e, ay = y0 - uy * e, bx = x1 + ux * e, by = y1 + uy * e;
      const nx = -uy * w, ny = ux * w;
      const q = [ax + nx, ay + ny, ax - nx, ay - ny, bx + nx, by + ny, bx - nx, by - ny];
      for (const [a, b] of [[0, 1], [2, 3], [4, 5], [2, 3], [6, 7], [4, 5]]) { P[k++] = q[a]; P[k++] = q[b]; P[k++] = z; }
      if (k >= P.length - 18) break;
    }
    const g = this.mesh.geometry;
    g.setDrawRange(0, k / 3);
    g.attributes.position.needsUpdate = true;
  }
  circle(cx, cy, r, opts = {}, n = 160) {
    const xs = new Float64Array(n), ys = new Float64Array(n);
    for (let i = 0; i < n; i++) { const a = 2 * Math.PI * i / n; xs[i] = cx + r * Math.cos(a); ys[i] = cy + r * Math.sin(a); }
    this.set(xs, ys, Object.assign({ closed: true }, opts));
  }
  /** Ellipse {x: (x - m)^T S^{-1} (x - m) = ns^2} for a 2x2 covariance S = [[a, b], [b, c]]. */
  ellipse(m, S, ns, opts = {}, n = 160) {
    const { xs, ys } = ellipsePts(m, S, ns, n);
    this.set(xs, ys, Object.assign({ closed: true }, opts));
  }
}

/** Points on the ns-sigma ellipse of covariance S = [[a, b], [b, c]] around m. */
export function ellipsePts(m, S, ns, n = 160) {
  const a = S[0][0], b = S[0][1], c = S[1][1];
  const tr = (a + c) / 2, det = Math.sqrt(Math.max(0, ((a - c) / 2) ** 2 + b * b));
  const l1 = Math.max(0, tr + det), l2 = Math.max(0, tr - det);
  const th = 0.5 * Math.atan2(2 * b, a - c);
  const c1 = Math.cos(th), s1 = Math.sin(th), r1 = ns * Math.sqrt(l1), r2 = ns * Math.sqrt(l2);
  const xs = new Float64Array(n), ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const t = 2 * Math.PI * i / n, u = r1 * Math.cos(t), v = r2 * Math.sin(t);
    xs[i] = m[0] + c1 * u - s1 * v; ys[i] = m[1] + s1 * u + c1 * v;
  }
  return { xs, ys };
}

/** An x-shaped cross made of two flat lines. */
export class Cross {
  constructor(scene, color, { size = 0.05, width = 0.012, z = 0.03 } = {}) {
    this.a = new FlatLine(scene, color, { width, z, maxPts: 4 });
    this.b = new FlatLine(scene, color, { width, z, maxPts: 4 });
    this.size = size;
  }
  set visible(v) { this.a.visible = v; this.b.visible = v; }
  at(x, y) {
    const s = this.size;
    this.a.set([x - s, x + s], [y - s, y + s]);
    this.b.set([x - s, x + s], [y + s, y - s]);
  }
}

/** Flat text on the table (canvas texture), readable from the default camera. */
export function floorLabel(scene, text, x, y, color = CSS.yellow, { height = 0.09, z = 0.004, rot = 0 } = {}) {
  const c = document.createElement('canvas'), g = c.getContext('2d');
  const font = '600 64px Inter, Arial, sans-serif';
  g.font = font;
  const w = Math.ceil(g.measureText(text).width) + 24;
  c.width = w; c.height = 96;
  g.font = font; g.fillStyle = color; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(text, w / 2, 50);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(height * w / 96, height),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, toneMapped: false }));
  mesh.position.set(x, y, z); mesh.rotation.z = rot;
  scene.add(mesh);
  return mesh;
}

// ---------------------------------------------------------------- 2-D plotting pane
export const FONT = '12px Inter, Arial, sans-serif';
export const FONT_B = '600 12.5px Inter, Arial, sans-serif';
export const MONO = '11.5px "JetBrains Mono", Menlo, Consolas, monospace';

export function label(g, s, x, y, color = CSS.grey, { align = 'left', font = FONT, base = 'alphabetic' } = {}) {
  g.save(); g.font = font; g.fillStyle = color; g.textAlign = align; g.textBaseline = base; g.fillText(s, x, y); g.restore();
}

/** Legend entry: a short sample then text; returns the x where the next entry may start. kind: line | dash | dot | box */
export function legend(g, x, y, color, text, kind = 'line', textColor = CSS.grey) {
  g.save();
  g.strokeStyle = color; g.fillStyle = color; g.lineWidth = 2;
  if (kind === 'dot') { g.beginPath(); g.arc(x + 6, y - 4, 3.5, 0, 2 * Math.PI); g.fill(); }
  else if (kind === 'box') { g.globalAlpha = 0.85; g.fillRect(x, y - 9, 12, 10); }
  else if (kind === 'outline') { g.strokeRect(x + 0.5, y - 9.5, 12, 10); }
  else { if (kind === 'dash') g.setLineDash([5, 4]); g.beginPath(); g.moveTo(x, y - 4); g.lineTo(x + 18, y - 4); g.stroke(); }
  g.restore();
  const tx = x + (kind === 'dot' || kind === 'box' || kind === 'outline' ? 17 : 23);
  label(g, text, tx, y, textColor);
  g.save(); g.font = FONT; const w = g.measureText(text).width; g.restore();
  return tx + w + 14;
}

export class Pane {
  constructor(plot) { this.o = plot; }
  get g() { return this.o.g; }
  set(x, y, w, h, x0, x1, y0, y1) { Object.assign(this, { x, y, w, h, x0, x1, y0, y1 }); return this; }
  X(t) { return this.x + this.w * (t - this.x0) / (this.x1 - this.x0); }
  Y(v) { return this.y + this.h * (1 - (v - this.y0) / (this.y1 - this.y0)); }
  frame({ title = '', yticks = [], fmt = (v) => String(v), xticks = [], xfmt = (v) => String(v), xlabel = '', ylabel = '', zero = true } = {}) {
    const g = this.g;
    g.save();
    g.lineWidth = 1; g.strokeStyle = CSS.faint;
    g.strokeRect(this.x + 0.5, this.y + 0.5, this.w, this.h);
    g.font = FONT; g.fillStyle = CSS.grey; g.textBaseline = 'middle'; g.textAlign = 'right';
    for (const v of yticks) {
      const yy = Math.round(this.Y(v)) + 0.5;
      g.globalAlpha = (zero && v === 0) ? 0.9 : 0.35;
      g.beginPath(); g.moveTo(this.x, yy); g.lineTo(this.x + this.w, yy); g.stroke();
      g.globalAlpha = 1;
      g.fillText(fmt(v), this.x - 5, yy);
    }
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (const v of xticks) {
      const xx = Math.round(this.X(v)) + 0.5;
      g.globalAlpha = (zero && v === 0) ? 0.9 : 0.35;
      g.beginPath(); g.moveTo(xx, this.y); g.lineTo(xx, this.y + this.h); g.stroke();
      g.globalAlpha = 1;
      g.fillText(xfmt(v), xx, this.y + this.h + 4);
    }
    if (xlabel) { g.textAlign = 'right'; g.fillText(xlabel, this.x + this.w, this.y + this.h + 19); }
    if (ylabel) { g.save(); g.translate(this.x - 32, this.y + this.h / 2); g.rotate(-Math.PI / 2); g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(ylabel, 0, 0); g.restore(); }
    if (title) { g.font = FONT_B; g.fillStyle = CSS.white; g.textAlign = 'left'; g.textBaseline = 'alphabetic'; g.fillText(title, this.x, this.y - 8); }
    g.restore();
  }
  clip() { const g = this.g; g.save(); g.beginPath(); g.rect(this.x, this.y, this.w, this.h); g.clip(); }
  line(xs, ys, color, width = 2, { dash = null, alpha = 1, i0 = 0, i1 = xs.length, closed = false } = {}) {
    if (i1 - i0 < 2) return;
    const g = this.g;
    this.clip();
    g.strokeStyle = color; g.lineWidth = width; g.globalAlpha = alpha; g.lineJoin = 'round';
    if (dash) g.setLineDash(dash);
    g.beginPath();
    for (let i = i0; i < i1; i++) { const px = this.X(xs[i]), py = this.Y(ys[i]); if (i === i0) g.moveTo(px, py); else g.lineTo(px, py); }
    if (closed) g.closePath();
    g.stroke();
    g.restore();
  }
  dots(xs, ys, color, r = 1.6, alpha = 1, { hollow = false, i1 = xs.length } = {}) {
    const g = this.g;
    this.clip();
    g.fillStyle = color; g.strokeStyle = color; g.globalAlpha = alpha; g.lineWidth = 1.3;
    for (let i = 0; i < i1; i++) {
      g.beginPath(); g.arc(this.X(xs[i]), this.Y(ys[i]), r, 0, 2 * Math.PI);
      if (hollow) g.stroke(); else g.fill();
    }
    g.restore();
  }
  hline(v, color, { dash = [5, 4], alpha = 0.9, width = 1.5 } = {}) {
    const g = this.g, yy = Math.round(this.Y(v)) + 0.5;
    this.clip();
    g.strokeStyle = color; g.globalAlpha = alpha; g.lineWidth = width; g.setLineDash(dash);
    g.beginPath(); g.moveTo(this.x, yy); g.lineTo(this.x + this.w, yy); g.stroke();
    g.restore();
  }
  vline(t, color, { dash = [4, 4], alpha = 0.7, width = 1 } = {}) {
    const g = this.g, xx = Math.round(this.X(t)) + 0.5;
    this.clip();
    g.strokeStyle = color; g.globalAlpha = alpha; g.lineWidth = width; g.setLineDash(dash);
    g.beginPath(); g.moveTo(xx, this.y); g.lineTo(xx, this.y + this.h); g.stroke();
    g.restore();
  }
  /** Bars from y = 0: centres xs, heights ys, width in data units. */
  bars(xs, ys, color, width, { alpha = 0.85, outline = false, lw = 2 } = {}) {
    const g = this.g;
    this.clip();
    g.globalAlpha = alpha; g.fillStyle = color; g.strokeStyle = color; g.lineWidth = lw;
    for (let i = 0; i < xs.length; i++) {
      const xa = this.X(xs[i] - width / 2), xb = this.X(xs[i] + width / 2), ya = this.Y(0), yb = this.Y(ys[i]);
      if (outline) g.strokeRect(xa, yb, xb - xa, ya - yb); else g.fillRect(xa, yb, xb - xa, ya - yb);
    }
    g.restore();
  }
}

/** Nice upper limit for an axis. */
export function niceMax(v, steps = [0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1]) {
  if (!(v > 0)) return 1;
  const e = Math.pow(10, Math.floor(Math.log10(v)));
  for (const s of steps) if (s * 10 * e >= v - 1e-12) return s * 10 * e;
  return 10 * e;
}
export function ticks(lo, hi, n = 4) {
  const raw = (hi - lo) / n, e = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * e).find((s) => s >= raw - 1e-12);
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(Math.abs(v) < 1e-12 ? 0 : +v.toPrecision(10));
  return out;
}
export const tfmt = (v) => (Math.abs(v) >= 1 || v === 0 ? String(+v.toFixed(2)) : String(+v.toPrecision(2)));

/** Run frame work within a time budget: calls step() until n steps or ms elapsed; returns steps done. */
export function budget(n, ms, step) {
  const t0 = performance.now();
  let k = 0;
  while (k < n) { step(); k++; if ((k & 7) === 0 && performance.now() - t0 > ms) break; }
  return k;
}
