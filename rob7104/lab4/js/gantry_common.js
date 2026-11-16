// Shared pieces of the three gantry demos (d31, d31b, d33): three.js pendulums whose cable length can change,
// the trolley follower, the load-cell reading, a discrete Kalman filter, a seeded noise source and small
// helpers for the 2D canvases. Numerics that already exist in sim.js are used from there.
import * as S from './sim.js';
import { THREE, COL, CSS } from './common.js';

export const PIV_TWIN = 2.30 - 0.16;           // pivot height of gantry.mjb (rail at 2.30 m)
export const PIV_CRANE = 2.40 - 0.16;          // pivot height of gantry_crane.mjb
export const DY = 0.10;                        // the two hoists hang this far to either side of the rail
export const DIM = 0x383b42;                   // colour of a "hidden" load

// ------------------------------------------------------------------ three.js pendulums
function material(color, opacity) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05,
    transparent: opacity < 1, opacity, depthWrite: opacity >= 1 });
}

function restyle(mesh, color, opacity) {
  const m = mesh.material;
  if (color !== undefined) m.color.set(color);
  if (opacity !== undefined) {
    m.transparent = opacity < 1; m.opacity = opacity; m.depthWrite = opacity >= 1;
    mesh.castShadow = opacity >= 1;
    mesh.visible = opacity > 0;
    m.needsUpdate = true;
  }
}

/** One cable with a load, hanging from `parent` (a group at the pivot). Angle positive toward +x. */
export class Pend {
  constructor(parent, { y = 0, color = COL.yellow, shape = 'box', r = 0.085, cableR = 0.008, opacity = 1 } = {}) {
    this.pivot = new THREE.Group();
    this.pivot.position.set(0, y, 0);
    const cg = new THREE.CylinderGeometry(cableR, cableR, 1, 12);
    cg.rotateX(Math.PI / 2);
    cg.translate(0, 0, -0.5);                                  // unit cable from the pivot down to z = -1
    this.cable = new THREE.Mesh(cg, material(COL.white, opacity));
    const lg = shape === 'box' ? new THREE.BoxGeometry(2 * r, 2 * r, 2 * r) : new THREE.SphereGeometry(r, 28, 18);
    this.load = new THREE.Mesh(lg, material(color, opacity));
    for (const m of [this.cable, this.load]) { m.castShadow = opacity >= 1; m.receiveShadow = opacity >= 1; }
    this.pivot.add(this.cable, this.load);
    parent.add(this.pivot);
    this.setLength(1);
  }
  setLength(l) { this.l = l; this.cable.scale.z = l; this.load.position.z = -l; }
  setAngle(th) { this.pivot.rotation.y = -th; }
  /** Colour and opacity of the load, opacity of the cable (opacity 0 hides). */
  setLook(color, opacity = 1, cableOpacity = opacity) { restyle(this.load, color, opacity); restyle(this.cable, undefined, cableOpacity); }
}

/** The twin hoist: two pendulums (a: blue, cable l1; b: yellow, cable l2) on a group that follows the trolley. */
export class TwinRig {
  constructor(scene, z = PIV_TWIN) {
    this.group = new THREE.Group();
    this.group.position.set(0, 0, z);
    scene.add(this.group);
    this.a = new Pend(this.group, { y: -DY, color: COL.blue });
    this.b = new Pend(this.group, { y: DY, color: COL.yellow });
  }
  /** Translucent second pair (for an estimate). */
  addGhost(color = COL.green, opacity = 0.5) {
    this.ga = new Pend(this.group, { y: -DY, color, shape: 'sphere', r: 0.11, cableR: 0.005, opacity });
    this.gb = new Pend(this.group, { y: DY, color, shape: 'sphere', r: 0.11, cableR: 0.005, opacity });
    return this;
  }
  /** Small block under the trolley standing for the load cell. */
  addLoadCell(color = COL.green) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.30, 0.07), material(color, 1));
    m.position.set(0, 0, -0.035);
    m.castShadow = true;
    this.group.add(m);
    return this;
  }
  setX(x) { this.group.position.x = x; }
}

/** Thin vertical line from height z down to the floor (a plumb line or a target mark). */
export function plumb(scene, z, color = 0xffffff, opacity = 0.3) {
  const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, z), new THREE.Vector3(0, 0, 0.01)]);
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  scene.add(line);
  return line;
}

// ------------------------------------------------------------------ trolley
/**
 * Smooth follower with limited acceleration: a critically damped second-order lag toward `target`,
 * clipped at amax. Returns the acceleration at the END of a step of length h that started at (p, v, u).
 */
export function followerAccel(target, p, v, u, h, wf = 1.5, amax = 0.4) {
  const pp = p + v * h + 0.5 * u * h * h, vv = v + u * h;
  return Math.max(-amax, Math.min(amax, wf * wf * (target - pp) - 2 * wf * vv));
}

/** Exact trolley update over a step in which the acceleration goes linearly from u0 to u1. Returns [p, v]. */
export function trolleyStep(p, v, u0, u1, h) {
  return [p + v * h + (2 * u0 + u1) * h * h / 6, v + 0.5 * (u0 + u1) * h];
}

// ------------------------------------------------------------------ sensing and estimation
/** Exact horizontal force of one cable on the trolley (N): tension m (l w^2 + g cos th - u sin th) times sin th. */
export function cableForceX(l, m, th, w, u = 0) {
  return m * (l * w * w + S.G * Math.cos(th) - u * Math.sin(th)) * Math.sin(th);
}

/** Observability matrix [C; CA; ...; CA^(n-1)] for one output row C. */
export function obsv(A, C) {
  const At = S.transpose(A), rows = [C];
  for (let k = 1; k < A.length; k++) rows.push(S.mulVec(At, rows[k - 1]));
  return rows;
}

/** Seeded standard normal numbers (mulberry32 + Box-Muller, cosine branch only). */
export function gaussSource(seed) {
  let a = seed >>> 0;
  const uni = () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => { const u1 = uni(), u2 = uni(); return Math.sqrt(-2 * Math.log(u1 + 1e-300)) * Math.cos(2 * Math.PI * u2); };
}

/**
 * Discrete Kalman filter for x' = A x + B a (a: unknown acceleration jitter, std sigA per sample),
 * one scalar reading z = H x + noise (std sigR), sample period Ts. Predict, then correct, as in the lecture.
 */
export class Kalman {
  constructor(A, B, H, Ts, sigA, sigR, x0, P0diag) {
    const n = A.length, Z = S.zeros(n + 1, n + 1);
    for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) Z[i][j] = A[i][j] * Ts; Z[i][n] = B[i] * Ts; }
    const E = S.expm(Z);
    this.F = E.slice(0, n).map((r) => r.slice(0, n));
    const Gd = E.slice(0, n).map((r) => r[n]);
    this.Q = Gd.map((gi) => Gd.map((gj) => gi * gj * sigA * sigA));
    this.H = H; this.R = sigR * sigR; this.n = n;
    this.x = [...x0];
    this.P = S.zeros(n, n).map((r, i) => { r[i] = P0diag[i]; return r; });
    this.first = true;
  }
  /** One reading: predict (except before the very first reading), then correct. */
  update(z) {
    const { F, H, n } = this;
    if (!this.first) {
      this.x = S.mulVec(F, this.x);
      this.P = S.add(S.mul(S.mul(F, this.P), S.transpose(F)), this.Q);
    }
    this.first = false;
    const PH = S.mulVec(this.P, H);                            // P H^T
    const s = S.dot(H, PH) + this.R;
    const K = PH.map((v) => v / s);
    const innov = z - S.dot(H, this.x);
    this.x = this.x.map((v, i) => v + K[i] * innov);
    const HP = S.mulVec(S.transpose(this.P), H);               // H P
    this.P = this.P.map((r, i) => r.map((v, j) => v - K[i] * HP[j]));
    return this.x;
  }
}

// ------------------------------------------------------------------ 2D canvas helpers
export const FONT = '11px Inter, "Helvetica Neue", Arial, sans-serif';
export const FONT_B = '600 11.5px Inter, "Helvetica Neue", Arial, sans-serif';
export const MONO = '11.5px "SF Mono", ui-monospace, Menlo, Consolas, monospace';

/** A plotting rectangle inside a plot2d canvas: maps data (x, y) to pixels, draws a frame, lines and dots. */
export class Pane {
  constructor(o) { this.o = o; this.g = o.g; }
  set(x, y, w, h, x0, x1, y0, y1) { Object.assign(this, { x, y, w, h, x0, x1, y0, y1 }); return this; }
  X(t) { return this.x + this.w * (t - this.x0) / (this.x1 - this.x0); }
  Y(v) { return this.y + this.h * (1 - (v - this.y0) / (this.y1 - this.y0)); }
  /** Frame with a title above, y tick labels on the left and optional x tick labels below. */
  frame({ title = '', yticks = [], fmt = (v) => String(v), xticks = [], xfmt = (v) => String(v), zero = true, xlabel = '' } = {}) {
    const g = this.g;
    g.save();
    g.lineWidth = 1;
    g.strokeStyle = CSS.faint;
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
      g.globalAlpha = 0.35;
      g.beginPath(); g.moveTo(xx, this.y); g.lineTo(xx, this.y + this.h); g.stroke();
      g.globalAlpha = 1;
      g.fillText(xfmt(v), xx, this.y + this.h + 4);
    }
    if (xlabel) { g.textAlign = 'right'; g.fillText(xlabel, this.x + this.w, this.y + this.h + 18); }
    if (title) { g.font = FONT_B; g.fillStyle = CSS.white; g.textAlign = 'left'; g.textBaseline = 'alphabetic'; g.fillText(title, this.x, this.y - 7); }
    g.restore();
  }
  _clip() { const g = this.g; g.save(); g.beginPath(); g.rect(this.x, this.y, this.w, this.h); g.clip(); }
  /** Polyline through (xs[i], ys[i]) for i in [i0, i1). */
  line(xs, ys, color, width = 2, { dash = null, alpha = 1, i0 = 0, i1 = xs.length } = {}) {
    if (i1 - i0 < 2) return;
    const g = this.g;
    this._clip();
    g.strokeStyle = color; g.lineWidth = width; g.globalAlpha = alpha; g.lineJoin = 'round';
    if (dash) g.setLineDash(dash);
    g.beginPath();
    for (let i = i0; i < i1; i++) { const px = this.X(xs[i]), py = this.Y(ys[i]); if (i === i0) g.moveTo(px, py); else g.lineTo(px, py); }
    g.stroke();
    g.restore();
  }
  dots(xs, ys, color, r = 1.5, alpha = 1) {
    const g = this.g;
    this._clip();
    g.fillStyle = color; g.globalAlpha = alpha;
    for (let i = 0; i < xs.length; i++) { g.beginPath(); g.arc(this.X(xs[i]), this.Y(ys[i]), r, 0, 2 * Math.PI); g.fill(); }
    g.restore();
  }
  vline(t, color, { dash = [4, 4], alpha = 0.8, width = 1 } = {}) {
    const g = this.g, xx = Math.round(this.X(t)) + 0.5;
    this._clip();
    g.strokeStyle = color; g.globalAlpha = alpha; g.lineWidth = width; g.setLineDash(dash);
    g.beginPath(); g.moveTo(xx, this.y); g.lineTo(xx, this.y + this.h); g.stroke();
    g.restore();
  }
  marker(t, v, color, r = 4) {
    const g = this.g;
    g.save(); g.fillStyle = color; g.beginPath(); g.arc(this.X(t), this.Y(v), r, 0, 2 * Math.PI); g.fill(); g.restore();
  }
}

/** Text on a plot2d canvas. */
export function label(g, s, x, y, color = CSS.grey, { align = 'left', font = FONT, base = 'alphabetic' } = {}) {
  g.save(); g.font = font; g.fillStyle = color; g.textAlign = align; g.textBaseline = base; g.fillText(s, x, y); g.restore();
}

/** Short legend entry: a line sample followed by text. Returns the x where the next entry may start. */
export function legend(g, x, y, color, text, { dash = null, width = 2, alpha = 1 } = {}) {
  g.save();
  g.strokeStyle = color; g.lineWidth = width; g.globalAlpha = alpha;
  if (dash) g.setLineDash(dash);
  g.beginPath(); g.moveTo(x, y - 3.5); g.lineTo(x + 18, y - 3.5); g.stroke();
  g.restore();
  label(g, text, x + 23, y, CSS.grey);
  g.save(); g.font = FONT; const w = g.measureText(text).width; g.restore();
  return x + 23 + w + 12;
}

/** Rolling record of samples: push(t, a, b, ...) keeps the last `window` seconds in parallel arrays. */
export class Record {
  constructor(nseries, window) { this.window = window; this.t = []; this.s = Array.from({ length: nseries }, () => []); }
  push(t, ...vals) {
    this.t.push(t);
    vals.forEach((v, i) => this.s[i].push(v));
    while (this.t.length && this.t[0] < t - this.window) { this.t.shift(); this.s.forEach((a) => a.shift()); }
  }
  clear() { this.t.length = 0; this.s.forEach((a) => { a.length = 0; }); }
  maxAbs(i) { let m = 0; for (const v of this.s[i]) m = Math.max(m, Math.abs(v)); return m; }
}

/** "nice" symmetric axis limit at or above v, from a short list of steps. */
export function niceLimit(v, steps = [1, 2, 2.5, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]) {
  for (const s of steps) if (v <= s) return s;
  return steps[steps.length - 1];
}

/** Write HTML into an overlay of the 3D cell: <div class="g-hud" data-hud="name">. */
export function hud(panel, name, html) {
  const el = panel.querySelector(`[data-hud="${name}"]`);
  if (el && el._html !== html) { el.innerHTML = html; el._html = html; }
}

/** Run fn(visible) whenever the panel becomes visible (active tab and on screen) or stops being so. */
export function watchVisible(panel, fn) {
  let tab = panel.classList.contains('active'), screen = true, last = null;
  const emit = () => { const vis = tab && screen && !document.hidden; if (vis !== last) { last = vis; fn(vis); } };
  panel.addEventListener('panel:show', () => { tab = true; emit(); });
  panel.addEventListener('panel:hide', () => { tab = false; emit(); });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver((en) => { screen = en[en.length - 1].isIntersecting; emit(); }, { threshold: 0.15 }).observe(panel);
  }
  document.addEventListener('visibilitychange', emit);
  emit();
}
