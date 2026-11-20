// Helpers shared by the Lab 6 demos. Lab 4's helpers (same names, same behaviour) plus what this lab needs:
// the UR5e viewer, the joint-space landscape, translucent ghost arms, 3D markers, ellipses and lines, keys.
// The numerics are sim6.js (re-exported as S); do not compute anything here that sim6.js already has.
// Contract and examples: Lab6_Optimization/build/web/README_DEV.md.
import { MjViewer, Trail, THREE } from '../../assets/mjviewer.js';
import * as S from './sim6.js';
export { MjViewer, Trail, THREE, S };

// ---------------------------------------------------------------------------------------------------- colours
// Same colours and meanings as the clips (STAGE3_SPEC.md): green = minimum "elbow up" and the target;
// blue = minimum "elbow down" and the l2 penalty; yellow = saddles, the l1 penalty, results; red = the maximum,
// failure, the gap to the target; pink = Newton's method and its paths; white = safeguarded Newton and the
// current iterate; grey = gradient descent paths (white-grey), rings of joints that did not move.
export const COL = { green: 0x83c167, blue: 0x58c4dd, yellow: 0xffeb40, red: 0xfc6255, pink: 0xd147bd,
  white: 0xeeeeee, grey: 0x8c8c94, accent: 0xe8b44f };
export const CSS = { green: '#83c167', blue: '#58c4dd', yellow: '#ffeb40', red: '#fc6255', pink: '#d147bd',
  white: '#eeeeee', grey: '#8c8c94', ink: '#f2efe9', mute: '#9a97a3', faint: '#4a4c55', accent: '#e8b44f' };

// ---------------------------------------------------------------------------------------------------- the arm
/** Height of the pedestal: sim6.js positions are in the arm's base frame; the models stand on a 1.0 m column. */
export const PEDESTAL = 1.0;
/** Base-frame position (sim6.js) -> world position (viewer). */
export const world = (p) => [p[0], p[1], p[2] + PEDESTAL];
/** Plane coordinates [x, z] of the two-joint problem -> world position. */
export const planeToWorld = (two, xz) => world(two.toWorld(xz));
export const TOOL_SITE = 'attachment_site';

/** Cameras (MuJoCo free-camera convention, world frame). `six` frames the six-joint arm (scene 1.3). */
export const CAM = {
  six: { lookat: [0.4, 0.0, 1.34], distance: 1.85, azimuth: -75, elevation: -12 },
  props: { lookat: [0.25, 0.1, 1.38], distance: 2.9, azimuth: -52, elevation: -14 },
};
/** Camera looking straight at the plane of the two-joint arm, plane x to the right and z up.
 *  `at` is the plane point at the centre of the view; `yaw` (degrees) turns the view for a three-quarter look. */
export function planeCamera(two, { at = [0.2, 0.15], distance = 2.4, elevation = -6, yaw = 0 } = {}) {
  const f = two.normal;                               // view direction: f x z_up = plane x  =>  f = z x x = normal
  return { lookat: planeToWorld(two, at), distance, elevation, azimuth: Math.atan2(f[1], f[0]) * 180 / Math.PI + yaw };
}

/** Create a viewer on the panel's canvas.view3d, load a model, and run it only while visible (Lab 4). */
export async function makeViewer(panel, modelUrl, camera) {
  const canvas = panel.querySelector('canvas.view3d');
  const veil = panel.querySelector('.loading');
  const v = new MjViewer(canvas, Object.assign({ preserveDrawingBuffer: /[?&]shot/.test(location.search) }, camera));
  await v.load(new URL(`../models/${modelUrl}`, import.meta.url).href);
  if (veil) veil.classList.add('gone');
  v.autoRun();
  panel.addEventListener('panel:hide', () => v.stop());
  panel.addEventListener('panel:show', () => v.start());
  v.start();
  return v;
}

/** The UR5e on its pedestal. props = true loads the model with the pillar and the shelf (mocap bodies
 *  'pillar' (origin at its foot) and 'shelf' (origin at the centre of its top face); move them with
 *  v.setMocap(name, worldPos)). Adds v.setArm(q6) (six angles, order S.JOINTS; call v.forward() after) and
 *  v.tool() (world position of the tool point). */
export async function makeArmViewer(panel, { props = false, camera = CAM.six, q = S.UR5E.Q_HOME } = {}) {
  const v = await makeViewer(panel, props ? 'lab6_arm_props.mjb' : 'lab6_arm.mjb', camera);
  v.setArm = (q6) => { for (let i = 0; i < 6; i++) v.setJoint(S.JOINTS[i], q6[i]); };
  v.tool = () => v.sitePos(TOOL_SITE);
  v.setArm(q);
  v.forward();
  return v;
}

// ---------------------------------------------------------------------------------------------------- 3D objects
/** A sphere in the scene: marker(v.scene, COL.green, 0.02).position.set(...world(p)). */
export function marker(scene, color, radius = 0.02, opacity = 1) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(radius, 20, 14),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35, roughness: 0.5,
      transparent: opacity < 1, opacity, depthWrite: opacity >= 1 }));
  scene.add(m);
  return m;
}

/** Many translucent stick arms at once (one InstancedMesh for links, one for joints), e.g. 200 noisy runs.
 *  set(i, q6) poses arm i (six angles); show(k) draws the first k arms. Positions from S.jointPositions. */
export class GhostArms {
  constructor(scene, n, { color = COL.white, opacity = 0.22, radius = 0.016 } = {}) {
    this.n = n; this.r = radius; this.shown = n;
    const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.6, transparent: true, opacity, depthWrite: false });
    this.links = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 10, 1), mat, 6 * n);
    this.joints = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 8), mat, 6 * n);
    for (const m of [this.links, this.joints]) { m.frustumCulled = false; m.renderOrder = 2; scene.add(m); }
    this._m = new THREE.Matrix4(); this._q = new THREE.Quaternion(); this._a = new THREE.Vector3();
    this._b = new THREE.Vector3(); this._d = new THREE.Vector3(); this._s = new THREE.Vector3(); this._y = new THREE.Vector3(0, 1, 0);
    this.points = Array.from({ length: n }, () => null);
  }
  set(i, q6) {
    const P = S.jointPositions(q6).map(world);
    this.points[i] = P;
    for (let k = 0; k < 6; k++) {
      this._a.set(...P[k]); this._b.set(...P[k + 1]);
      this._d.subVectors(this._b, this._a);
      const len = this._d.length();
      if (len > 1e-9) this._q.setFromUnitVectors(this._y, this._d.multiplyScalar(1 / len)); else this._q.identity();
      this._a.add(this._b).multiplyScalar(0.5);
      this._m.compose(this._a, this._q, this._s.set(this.r, len, this.r));
      this.links.setMatrixAt(6 * i + k, this._m);
      this._m.compose(this._a.set(...P[k + 1]), this._q.identity(), this._s.set(1.25 * this.r, 1.25 * this.r, 1.25 * this.r));
      this.joints.setMatrixAt(6 * i + k, this._m);
    }
    this.links.instanceMatrix.needsUpdate = true;
    this.joints.instanceMatrix.needsUpdate = true;
  }
  /** World position of the tool point of arm i (last set). */
  tool(i) { return this.points[i] && this.points[i][6]; }
  show(k) { this.shown = Math.max(0, Math.min(this.n, k)); this.links.count = 6 * this.shown; this.joints.count = 6 * this.shown; }
  set visible(on) { this.links.visible = on; this.joints.visible = on; }
}

/** A filled translucent ellipse with an outline, e.g. the manipulability ellipse at the tool.
 *  set(centre, axes, half): centre (world), axes = two unit 3-vectors, half = two half lengths. */
export class Ellipse3D {
  constructor(scene, color = COL.pink, { n = 72, opacity = 0.22 } = {}) {
    this.n = n;
    this.pos = new Float32Array(3 * n);
    const line = new THREE.BufferGeometry();
    line.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.outline = new THREE.LineLoop(line, new THREE.LineBasicMaterial({ color }));
    this.fan = new Float32Array(3 * (n + 2));
    const fill = new THREE.BufferGeometry();
    fill.setAttribute('position', new THREE.BufferAttribute(this.fan, 3));
    fill.setIndex(Array.from({ length: n }, (_, k) => [0, k + 1, k + 2]).flat());
    this.fill = new THREE.Mesh(fill, new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false }));
    for (const o of [this.outline, this.fill]) { o.frustumCulled = false; scene.add(o); }
  }
  set(c, axes, half) {
    for (let k = 0; k <= this.n; k++) {
      const t = 2 * Math.PI * k / this.n, a = half[0] * Math.cos(t), b = half[1] * Math.sin(t);
      const p = [0, 1, 2].map((j) => c[j] + a * axes[0][j] + b * axes[1][j]);
      if (k < this.n) this.pos.set(p, 3 * k);
      this.fan.set(p, 3 * (k + 1));
    }
    this.fan.set(c, 0);
    this.outline.geometry.attributes.position.needsUpdate = true;
    this.fill.geometry.attributes.position.needsUpdate = true;
  }
  set visible(on) { this.outline.visible = on; this.fill.visible = on; }
}

/** A polyline in 3D with up to max points: set(points) (world positions). For paths, strings of beads, traces. */
export class Line3D {
  constructor(scene, color = COL.white, max = 512, { opacity = 1 } = {}) {
    this.max = max;
    this.pos = new Float32Array(3 * max);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    g.setDrawRange(0, 0);
    this.line = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }));
    this.line.frustumCulled = false;
    scene.add(this.line);
  }
  set(points) {
    const n = Math.min(points.length, this.max);
    for (let k = 0; k < n; k++) this.pos.set(points[k], 3 * k);
    this.line.geometry.setDrawRange(0, n);
    this.line.geometry.attributes.position.needsUpdate = true;
  }
  set visible(on) { this.line.visible = on; }
}

// ---------------------------------------------------------------------------------------------------- controls (Lab 4)
/** Read a control by its data-ctl name. Sliders return numbers through val(). */
export function ctl(panel, name) {
  const el = panel.querySelector(`[data-ctl="${name}"]`);
  if (!el) throw new Error(`no control ${name}`);
  return el;
}
export const val = (panel, name) => Number(ctl(panel, name).value);
export function onInput(panel, name, fn) { ctl(panel, name).addEventListener('input', fn); }
export function onClick(panel, name, fn) { ctl(panel, name).addEventListener('click', fn); }
export function readout(panel, name, html) { ctl(panel, name).innerHTML = html; }

/** Segmented buttons: <span class="seg" data-ctl="x"><button data-v="a" class="on">..</button>..</span>.
 *  Returns a getter of the current value. */
export function seg(panel, name, fn) {
  const box = ctl(panel, name);
  box.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    box.querySelectorAll('button').forEach((o) => o.classList.toggle('on', o === b));
    fn(b.dataset.v);
  }));
  return () => box.querySelector('button.on').dataset.v;
}

/** On/off button: <button data-ctl="x" class="on">..</button>. fn(on) after each click; returns a getter. */
export function toggle(panel, name, fn) {
  const b = ctl(panel, name);
  b.addEventListener('click', () => { b.classList.toggle('on'); fn(b.classList.contains('on')); });
  return () => b.classList.contains('on');
}

/** Set a slider from code and fire its 'input' event (so its <output> and listeners update). */
export function setVal(panel, name, value) {
  const el = ctl(panel, name);
  el.value = value;
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

/** Keys held down while the pointer is over the panel (or focus is inside it) and the panel is the active tab.
 *  Returns a Set of e.key values; arrow keys and the listed keys do not scroll the page while used. */
export function heldKeys(panel, keys = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown']) {
  const held = new Set();
  const mine = () => panel.classList.contains('active') && (panel.matches(':hover') || panel.contains(document.activeElement));
  window.addEventListener('keydown', (e) => {
    if (!keys.includes(e.key) || !mine()) return;
    if (document.activeElement && /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName) && !e.key.startsWith('Arrow')) return;
    e.preventDefault();
    held.add(e.key);
  });
  window.addEventListener('keyup', (e) => held.delete(e.key));
  window.addEventListener('blur', () => held.clear());
  return held;
}

// ---------------------------------------------------------------------------------------------------- 2D drawing (Lab 4)
/** 2D plot canvas with device-pixel scaling. Returns { g, w, h, r, fit, clear } in CSS pixels. */
export function plot2d(canvas) {
  const ctx = canvas.getContext('2d');
  const o = { g: ctx, w: 0, h: 0, r: 1 };
  o.fit = () => {
    const r = Math.min(window.devicePixelRatio || 1, 2);
    o.r = r; o.w = canvas.clientWidth; o.h = canvas.clientHeight;
    if (canvas.width !== Math.round(o.w * r) || canvas.height !== Math.round(o.h * r)) {
      canvas.width = Math.round(o.w * r); canvas.height = Math.round(o.h * r);
    }
    ctx.setTransform(r, 0, 0, r, 0, 0);
  };
  o.clear = () => { o.fit(); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, o.w, o.h); };
  return o;
}

/** Colour maps of the clips' landscape panel (render/stage6.py): stops [x, [r, g, b] in 0..1]. */
export const LAND_LINE = [[0, [0.345, 0.769, 0.867]], [0.35, [0.514, 0.757, 0.404]], [0.7, [1.0, 0.92, 0.25]], [1, [0.988, 0.384, 0.333]]];
export const LAND_FILL = [[0, [0.02, 0.07, 0.12]], [0.35, [0.05, 0.12, 0.09]], [0.7, [0.15, 0.13, 0.04]], [1, [0.20, 0.07, 0.06]]];
/** Interpolate a colour map at x in [0, 1]; returns [r, g, b] in 0..255. */
export function cmap(stops, x) {
  x = Math.max(0, Math.min(1, x));
  let k = 0;
  while (k < stops.length - 2 && x > stops[k + 1][0]) k++;
  const [x0, c0] = stops[k], [x1, c1] = stops[k + 1], u = (x - x0) / (x1 - x0 || 1);
  return c0.map((c, j) => Math.round(255 * (c + u * (c1[j] - c)))) ;
}

const TICKS = [[-Math.PI, '−π'], [-Math.PI / 2, '−π/2'], [0, '0'], [Math.PI / 2, 'π/2'], [Math.PI, 'π']];
const MARK = { min: CSS.green, saddle: CSS.yellow, max: CSS.red, degenerate: CSS.grey };

/** Stationary points of the two-joint cost as landscape marks: [{q, type, name, color}] (elbow down = blue). */
export const marksFor = (two, t) => two.stationaryPoints(t).map((s) => ({ q: s.q, type: s.type, name: s.name,
  color: s.name === 'elbow down' ? CSS.blue : undefined }));

/** Joint-space landscape of a two-joint cost on [-pi, pi)^2, drawn like the clips' panel: dark filled bands
 *  (levels vmax * (k / levels)^2) with coloured level lines, x = shoulder q1, y = elbow q2, stationary points
 *  as marks (circle = minimum, green elbow up / blue elbow down; yellow cross = saddle; red triangle = maximum).
 *
 *    const land = new Landscape(panel.querySelector('canvas.land'));
 *    land.setCost((q) => two.cost(q, t), marksFor(two, t));
 *    land.onPick((q) => { start = q; });             // click / tap on the map
 *    function draw() { land.draw(); land.path(run.path, CSS.grey); land.dot(q, CSS.white, 5); }
 *
 *  draw() returns the 2D context (CSS pixels) for extra drawing; toPx(q) / fromPx(x, y) convert. The raster is
 *  computed once per setCost() and canvas size (about 0.05 s), never per frame. */
export class Landscape {
  constructor(canvas, { levels = 16, vmax = null, labels = ['shoulder q₁', 'elbow q₂'] } = {}) {
    this.canvas = canvas; this.P = plot2d(canvas); this.levels = levels; this.vmax0 = vmax; this.labels = labels;
    this.f = null; this.marks = []; this.img = null; this.field = null;
    this.box = { x: 0, y: 0, s: 1 };
  }
  setCost(f, marks = [], { vmax = this.vmax0 } = {}) { this.f = f; this.marks = marks; this.vmax = vmax; this.img = null; }
  _layout() {
    this.P.fit();
    const L = 46, R = 10, T = 10, B = 36, s = Math.max(10, Math.min(this.P.w - L - R, this.P.h - T - B));
    this.box = { x: L + (this.P.w - L - R - s) / 2, y: T + (this.P.h - T - B - s) / 2, s };
  }
  toPx(q) {
    const w = S.wrap(q), b = this.box;
    return [b.x + (w[0] + Math.PI) / (2 * Math.PI) * b.s, b.y + (Math.PI - w[1]) / (2 * Math.PI) * b.s];
  }
  fromPx(x, y) {
    const b = this.box, u = (x - b.x) / b.s, v = (y - b.y) / b.s;
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    return [-Math.PI + 2 * Math.PI * u, Math.PI - 2 * Math.PI * v];
  }
  _raster() {
    const n = Math.max(32, Math.min(480, Math.round(this.box.s * this.P.r)));
    if (this.img && this.img.width === n) return this.img;
    const F = new Float64Array(n * n);
    let vmax = this.vmax || 0;
    for (let j = 0; j < n; j++) {
      const q2 = Math.PI - (j + 0.5) * 2 * Math.PI / n;
      for (let i = 0; i < n; i++) {
        const v = this.f([-Math.PI + (i + 0.5) * 2 * Math.PI / n, q2]);
        F[j * n + i] = v;
        if (!this.vmax && v > vmax) vmax = v;
      }
    }
    const K = this.levels, band = new Uint8Array(n * n);
    for (let k = 0; k < n * n; k++) {
      const x = Math.sqrt(Math.max(0, F[k]) / vmax);           // level k: F <= vmax ((k + 1) / K)^2
      band[k] = Math.min(K - 1, Math.max(0, Math.ceil(x * K) - 1));
    }
    const fill = Array.from({ length: K }, (_, k) => cmap(LAND_FILL, (k + 1) / K));
    const line = Array.from({ length: K }, (_, k) => cmap(LAND_LINE, (k + 1) / K).map((c, j) => Math.round(0.6 * c + 0.4 * fill[k][j])));
    const c = document.createElement('canvas'); c.width = n; c.height = n;
    const g = c.getContext('2d'), im = g.createImageData(n, n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const k = j * n + i, b = band[k];
      const edge = (i + 1 < n && band[k + 1] !== b) || (j + 1 < n && band[k + n] !== b);
      const col = edge ? line[Math.min(b, band[i + 1 < n ? k + 1 : k], band[j + 1 < n ? k + n : k])] : fill[b];
      im.data.set([col[0], col[1], col[2], 255], 4 * k);
    }
    g.putImageData(im, 0, 0);
    this.vmaxUsed = vmax;
    return (this.img = c);
  }
  /** An n x n picture over joint space, colorAt(q) -> [r, g, b, a] (0..255), e.g. a basin map. Shown by draw()
   *  over the landscape until setField(null). */
  setField(n, colorAt) {
    if (!colorAt) { this.field = null; return; }
    const c = document.createElement('canvas'); c.width = n; c.height = n;
    const g = c.getContext('2d'), im = g.createImageData(n, n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      im.data.set(colorAt([-Math.PI + (i + 0.5) * 2 * Math.PI / n, Math.PI - (j + 0.5) * 2 * Math.PI / n]), 4 * (j * n + i));
    }
    g.putImageData(im, 0, 0);
    this.field = c;
  }
  draw() {
    this._layout();
    const g = this.P.g, b = this.box;
    g.fillStyle = '#000'; g.fillRect(0, 0, this.P.w, this.P.h);
    if (this.f) { g.imageSmoothingEnabled = false; g.drawImage(this._raster(), b.x, b.y, b.s, b.s); }
    if (this.field) { g.imageSmoothingEnabled = false; g.drawImage(this.field, b.x, b.y, b.s, b.s); }
    g.strokeStyle = 'rgba(97, 97, 107, 1)'; g.lineWidth = 1; g.strokeRect(b.x + 0.5, b.y + 0.5, b.s - 1, b.s - 1);
    g.fillStyle = CSS.mute; g.font = '11px Inter, sans-serif';
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (const [v, t] of TICKS) g.fillText(t, b.x + (v + Math.PI) / (2 * Math.PI) * b.s, b.y + b.s + 4);
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const [v, t] of TICKS) g.fillText(t, b.x - 5, b.y + (Math.PI - v) / (2 * Math.PI) * b.s);
    g.textAlign = 'center'; g.textBaseline = 'bottom';
    g.fillText(this.labels[0], b.x + b.s / 2, this.P.h - 1);
    g.save(); g.translate(11, b.y + b.s / 2); g.rotate(-Math.PI / 2); g.textBaseline = 'middle';
    g.fillText(this.labels[1], 0, 0); g.restore();
    for (const m of this.marks) this.mark(m.q, m.type, 6, m.color);
    return g;
  }
  /** Mark of a stationary point: 'min' circle (green; pass CSS.blue for elbow down), 'saddle' cross, 'max' triangle. */
  mark(q, type, size = 6, color = null) {
    const g = this.P.g, [x, y] = this.toPx(q), c = color || MARK[type] || CSS.grey;
    g.save(); g.strokeStyle = c; g.fillStyle = c; g.lineWidth = 2;
    g.beginPath();
    if (type === 'saddle') { g.moveTo(x - size, y - size); g.lineTo(x + size, y + size); g.moveTo(x + size, y - size); g.lineTo(x - size, y + size); g.stroke(); }
    else if (type === 'max') { g.moveTo(x, y - size * 1.15); g.lineTo(x + size, y + size * 0.75); g.lineTo(x - size, y + size * 0.75); g.closePath(); g.fill(); }
    else { g.arc(x, y, size * 0.85, 0, 2 * Math.PI); g.fill(); }
    g.restore();
  }
  /** Polyline through joint vectors; broken where it crosses the periodic boundary. dots: draw each iterate. */
  path(qs, color = CSS.white, { width = 1.6, alpha = 1, dots = false, r = 2.2 } = {}) {
    if (!qs || !qs.length) return;
    const g = this.P.g;
    g.save(); g.globalAlpha = alpha; g.strokeStyle = color; g.fillStyle = color; g.lineWidth = width; g.lineJoin = 'round';
    g.beginPath();
    let prev = null;
    for (const q of qs) {
      const w = S.wrap(q), [x, y] = this.toPx(w);
      if (prev && Math.max(Math.abs(w[0] - prev[0]), Math.abs(w[1] - prev[1])) > Math.PI) g.moveTo(x, y);
      else if (prev) g.lineTo(x, y); else g.moveTo(x, y);
      prev = w;
    }
    g.stroke();
    if (dots) for (const q of qs) { const [x, y] = this.toPx(q); g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill(); }
    g.restore();
  }
  /** A point (filled disc with a dark rim). */
  dot(q, color = CSS.white, r = 4.5) {
    const g = this.P.g, [x, y] = this.toPx(q);
    g.save(); g.fillStyle = color; g.strokeStyle = '#000'; g.lineWidth = 1.5;
    g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill(); g.stroke(); g.restore();
  }
  /** Many points (e.g. 200 noisy runs) as small discs; qs = array of joint vectors. */
  cloud(qs, color = CSS.white, { r = 1.8, alpha = 0.6 } = {}) {
    const g = this.P.g;
    g.save(); g.globalAlpha = alpha; g.fillStyle = color;
    for (const q of qs) { const [x, y] = this.toPx(q); g.fillRect(x - r, y - r, 2 * r, 2 * r); }
    g.restore();
  }
  /** fn(q) when the map is clicked or tapped inside the square. */
  onPick(fn) {
    this.canvas.addEventListener('pointerdown', (e) => {
      const rc = this.canvas.getBoundingClientRect(), q = this.fromPx(e.clientX - rc.left, e.clientY - rc.top);
      if (q) fn(q);
    });
  }
}

// ---------------------------------------------------------------------------------------------------- numbers
/** Fixed-step accumulator (Lab 4): calls step(h) enough times to advance by dt * speed (at most 400 per call). */
export function stepper(h) {
  let acc = 0;
  return (dt, speed, step) => {
    acc += dt * speed;
    let n = 0;
    while (acc >= h && n < 400) { step(h); acc -= h; n++; }
    if (n === 400) acc = 0;
  };
}

/** Clamp a plane target [x, z] strictly inside the reach of the two-joint arm (finding F3 of webcore: exactly on
 *  the limit the number of solutions depends on the last bit). */
export function clampReach(two, t, margin = 1e-6) {
  const r = Math.hypot(t[0], t[1]);
  if (r < 1e-12) return [two.reachMin + margin, 0];
  const c = Math.min(two.reachMax - margin, Math.max(two.reachMin + margin, r));
  return [t[0] * c / r, t[1] * c / r];
}

/** Number for a readout: fixed digits, '—' when not finite. */
export const fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '—');
export const deg = (r) => r * 180 / Math.PI;
export const rad = (d) => d * Math.PI / 180;
