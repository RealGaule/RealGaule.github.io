// Helpers shared by the Lab 5 demos (documented in build/web/README_DEV.md).
// Lab 4's helpers (makeViewer, ctl, val, onInput, onClick, readout, seg, plot2d, stepper, deg, rad) plus what the
// cart demos need: cart poses and wheels, an instanced cloud of carts, floor ellipses, range rings, floor arrows,
// keyboard input, and small 2D plotting helpers. Numerics live in sim5.js (re-exported as S); do not copy them here.
import { MjViewer, Trail, THREE } from '../../assets/mjviewer.js';
import * as S from './sim5.js';
export { MjViewer, Trail, THREE, S };

/* ---------- colours: the meanings of the clips (STAGE3_SPEC.md) ---------- */
export const COL = {
  truth: 0xeeeeee, cloud: 0x58c4dd, est: 0xffeb40, ekf: 0xffeb40, ukf: 0xf0a45d, reading: 0x83c167,
  fail: 0xfc6255, pink: 0xd147bd,
  green: 0x83c167, red: 0xfc6255, blue: 0x58c4dd, yellow: 0xffeb40, orange: 0xf0a45d, white: 0xeeeeee, grey: 0x8c8c94, accent: 0xe8b44f,
};
export const CSS = {
  truth: '#eeeeee', cloud: '#58c4dd', est: '#ffeb40', ekf: '#ffeb40', ukf: '#f0a45d', reading: '#83c167',
  fail: '#fc6255', pink: '#d147bd',
  green: '#83c167', red: '#fc6255', blue: '#58c4dd', yellow: '#ffeb40', orange: '#f0a45d', white: '#f2efe9',
  grey: '#9a97a3', faint: '#4a4c55', accent: '#e8b44f',
};

/* ---------- viewer ---------- */

/** Create a viewer on the panel's canvas.view3d (or opts.canvas), load models/<modelFile>, run it only while visible.
 *  camera: MuJoCo free-camera terms { lookat, distance, azimuth, elevation }, plus any MjViewer option. */
export async function makeViewer(panel, modelFile, camera, opts = {}) {
  const canvas = opts.canvas || panel.querySelector('canvas.view3d');
  const veil = canvas.parentElement.querySelector('.loading');
  const v = new MjViewer(canvas, Object.assign({ preserveDrawingBuffer: /[?&]shot/.test(location.search) }, camera));
  await v.load(new URL(`../models/${modelFile}`, import.meta.url).href);
  if (veil) veil.classList.add('gone');
  v.autoRun();
  panel.addEventListener('panel:hide', () => v.stop());
  panel.addEventListener('panel:show', () => v.start());
  v.start();
  return v;
}

/** Fetch lab5/data/<name> (JSON). */
export async function loadJSON(name) {
  return (await fetch(new URL(`../data/${name}`, import.meta.url).href)).json();
}

/* ---------- carts ---------- */

export const quatYaw = (th) => [Math.cos(th / 2), 0, 0, Math.sin(th / 2)];

/** Put the mocap cart `name` (cart, est, ekf, ukf, cart0, ...) at pose [x, y, theta]; phi = [right, left] wheel angles. */
export function setCart(v, name, pose, phi) {
  v.setMocap(name, [pose[0], pose[1], 0], undefined, quatYaw(pose[2]));
  if (phi) { v.setJoint(`${name}_w1`, phi[0]); v.setJoint(`${name}_w2`, phi[1]); }
}

/** Hide a mocap body (cart, beacon, shelf) below the floor; setCart / setMocap brings it back. */
export function parkBody(v, name) { v.setMocap(name, [0, 0, -5]); }

/** Wheel angles from successive poses (rolling without slip, as in the model): phi1 right, phi2 left. */
export class Wheels {
  constructor(pose = [0, 0, 0]) { this.reset(pose); }
  reset(pose) { this.prev = [pose[0], pose[1], pose[2]]; this.phi = [0, 0]; return this.phi; }
  update(pose) {
    const p = this.prev, dth = pose[2] - p[2], thm = p[2] + dth / 2;
    const ds = (pose[0] - p[0]) * Math.cos(thm) + (pose[1] - p[1]) * Math.sin(thm);
    this.phi[0] += (ds + dth * S.AXLE / 2) / S.R_WHEEL;
    this.phi[1] += (ds - dth * S.AXLE / 2) / S.R_WHEEL;
    this.prev = [pose[0], pose[1], pose[2]];
    return this.phi;
  }
}

let ghostPromise = null;
/** The merged cart mesh of models/cart_ghost.json as a THREE.BufferGeometry (cart frame: origin on the floor under
 *  the middle of the axle, x forward). Loaded once per page. */
export function loadGhost() {
  if (!ghostPromise) {
    ghostPromise = fetch(new URL('../models/cart_ghost.json', import.meta.url).href).then((r) => r.json()).then((g) => {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(g.vertices.flat(), 3));
      geo.setIndex(g.faces.flat());
      geo.computeVertexNormals();
      return geo;
    });
  }
  return ghostPromise;
}

/** A cloud of carts: every cart is a light dot, the first `bodies` carts are also drawn as translucent cart bodies
 *  (the convention of the clips). set(pose, n) takes a Float64Array(3N) of [x, y, theta] as sim5.js uses. */
export class CartCloud {
  constructor(scene, ghostGeo, { max = 4000, bodies = 60, color = COL.cloud, opacity = 0.32, dot = 0.025, dotZ = 0.11, dotOpacity = 0.9 } = {}) {
    this.max = max; this.bodies = Math.min(bodies, max); this.n = 0;
    this.inst = new THREE.InstancedMesh(ghostGeo, new THREE.MeshStandardMaterial({ color, transparent: true, opacity, depthWrite: false, roughness: 0.6 }), Math.max(this.bodies, 1));
    this.inst.count = 0;
    this.inst.frustumCulled = false;
    this.pos = new Float32Array(3 * max);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    g.setDrawRange(0, 0);
    this.points = new THREE.Points(g, new THREE.PointsMaterial({ color, size: dot, sizeAttenuation: true, transparent: true, opacity: dotOpacity, depthWrite: false }));
    this.points.frustumCulled = false;
    this.dotZ = dotZ;
    this.group = new THREE.Group();
    this.group.add(this.inst, this.points);
    scene.add(this.group);
    this._m = new THREE.Matrix4();
  }
  set(pose, n = pose.length / 3) {
    n = Math.min(n | 0, this.max); this.n = n;
    for (let i = 0; i < n; i++) { this.pos[3 * i] = pose[3 * i]; this.pos[3 * i + 1] = pose[3 * i + 1]; this.pos[3 * i + 2] = this.dotZ; }
    this.points.geometry.setDrawRange(0, n);
    this.points.geometry.attributes.position.needsUpdate = true;
    const nb = Math.min(this.bodies, n);
    for (let i = 0; i < nb; i++) { this._m.makeRotationZ(pose[3 * i + 2]).setPosition(pose[3 * i], pose[3 * i + 1], 0); this.inst.setMatrixAt(i, this._m); }
    this.inst.count = nb;
    this.inst.instanceMatrix.needsUpdate = true;
  }
  get visible() { return this.group.visible; }
  set visible(b) { this.group.visible = b; }
}

/** Flat band along a closed loop of floor points [[x, y], ...] (WebGL lines are 1 px, so bands are meshes). */
export class Ribbon {
  constructor(scene, color, { n = 96, z = 0.003, opacity = 0.95, closed = true } = {}) {
    this.n = n; this.z = z; this.closed = closed;
    this.pos = new Float32Array(6 * n);
    const idx = [], m = closed ? n : n - 1;
    for (let i = 0; i < m; i++) { const a = 2 * i, b = a + 1, c = 2 * ((i + 1) % n), d = c + 1; idx.push(a, b, c, b, d, c); }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    g.setIndex(idx);
    this.mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false }));
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 2;
    scene.add(this.mesh);
  }
  /** pts: n points [x, y]; width in metres. */
  set(pts, width) {
    const n = this.n, h = width / 2;
    for (let i = 0; i < n; i++) {
      const a = pts[this.closed ? (i - 1 + n) % n : Math.max(i - 1, 0)], b = pts[this.closed ? (i + 1) % n : Math.min(i + 1, n - 1)];
      let tx = b[0] - a[0], ty = b[1] - a[1];
      const L = Math.hypot(tx, ty);
      if (L < 1e-12) { tx = 1; ty = 0; } else { tx /= L; ty /= L; }
      const p = pts[i];
      this.pos.set([p[0] + ty * h, p[1] - tx * h, this.z, p[0] - ty * h, p[1] + tx * h, this.z], 6 * i);
    }
    this.mesh.geometry.attributes.position.needsUpdate = true;
  }
  get visible() { return this.mesh.visible; }
  set visible(b) { this.mesh.visible = b; }
}

/** The filter's summary on the floor: the nsig ellipse of a 2x2 (or the xy block of a 3x3) covariance, as a band
 *  with a faint fill. set(mean [x, y, ...], cov). */
export class FloorEllipse {
  constructor(scene, color = COL.est, { nsig = 2, n = 96, width = 0.01, z = 0.004, fill = 0.10, opacity = 0.95 } = {}) {
    this.nsig = nsig; this.n = n; this.width = width;
    this.band = new Ribbon(scene, color, { n, z: z + 0.0005, opacity });
    this.fpos = new Float32Array(3 * (n + 1));
    const idx = [];
    for (let i = 0; i < n; i++) idx.push(0, 1 + i, 1 + ((i + 1) % n));
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.fpos, 3));
    g.setIndex(idx);
    this.fill = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: fill, side: THREE.DoubleSide, depthWrite: false }));
    this.fill.frustumCulled = false;
    this.fill.visible = fill > 0;
    this.z = z;
    scene.add(this.fill);
  }
  set(mean, cov) {
    const c2 = [[cov[0][0], cov[0][1]], [cov[1][0], cov[1][1]]];
    const pts = S.ellipsePoints([mean[0], mean[1]], c2, this.nsig, this.n);
    this.band.set(pts, this.width);
    this.fpos.set([mean[0], mean[1], this.z], 0);
    for (let i = 0; i < this.n; i++) this.fpos.set([pts[i][0], pts[i][1], this.z], 3 * (i + 1));
    this.fill.geometry.attributes.position.needsUpdate = true;
  }
  get visible() { return this.band.visible; }
  set visible(b) { this.band.visible = b; this.fill.visible = b; }
}

/** A range reading drawn on the floor: a ring of radius r around the beacon, band width `width` (e.g. 4 sigma_r). */
export class RangeRing {
  constructor(scene, color = COL.reading, { n = 160, z = 0.002, opacity = 0.7 } = {}) {
    this.n = n;
    this.band = new Ribbon(scene, color, { n, z, opacity });
  }
  set(center, r, width) {
    const pts = [];
    for (let i = 0; i < this.n; i++) { const a = 2 * Math.PI * i / this.n; pts.push([center[0] + r * Math.cos(a), center[1] + r * Math.sin(a)]); }
    this.band.set(pts, width);
  }
  get visible() { return this.band.visible; }
  set visible(b) { this.band.visible = b; }
  set opacity(a) { this.band.mesh.material.opacity = a; }
}

/** A flat arrow on the floor (velocity, heading). set(from [x, y], vec [dx, dy]); a zero vector hides it. */
export class FloorArrow {
  constructor(scene, color = COL.est, { width = 0.014, head = 0.05, z = 0.006, opacity = 0.95 } = {}) {
    this.width = width; this.head = head; this.z = z;
    this.pos = new Float32Array(7 * 3);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    g.setIndex([0, 1, 2, 0, 2, 3, 4, 5, 6]);
    this.mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false }));
    this.mesh.frustumCulled = false;
    scene.add(this.mesh);
  }
  set(from, vec) {
    const L = Math.hypot(vec[0], vec[1]);
    this.mesh.visible = L > 1e-6;
    if (!this.mesh.visible) return;
    const ux = vec[0] / L, uy = vec[1] / L, nx = -uy, ny = ux, w = this.width / 2;
    const hl = Math.min(this.head, 0.6 * L), sx = from[0] + ux * (L - hl), sy = from[1] + uy * (L - hl), z = this.z;
    this.pos.set([from[0] + nx * w, from[1] + ny * w, z, sx + nx * w, sy + ny * w, z, sx - nx * w, sy - ny * w, z, from[0] - nx * w, from[1] - ny * w, z,
      sx + nx * 2.4 * w, sy + ny * 2.4 * w, z, from[0] + vec[0], from[1] + vec[1], z, sx - nx * 2.4 * w, sy - ny * 2.4 * w, z]);
    this.mesh.geometry.attributes.position.needsUpdate = true;
  }
  get visible() { return this.mesh.visible; }
  set visible(b) { this.mesh.visible = b; }
}

/* ---------- controls (as Lab 4) ---------- */

/** Read a control by its data-ctl name. */
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

/** On/off button (<button data-ctl="x" class="on">): calls fn(state) on click; returns a getter. */
export function toggle(panel, name, fn) {
  const b = ctl(panel, name);
  let on = b.classList.contains('on');
  b.addEventListener('click', () => { on = !on; b.classList.toggle('on', on); fn(on); });
  return () => on;
}

/** Keyboard input for one panel. Keys count only while the panel is the active tab and mostly on screen and no text
 *  field has focus; the listed keys then do not scroll the page. The panel gets class "keys-on" while listening.
 *  Returns { held: Set, down(key), onPress(key, fn) }. Tests can send keys with pw.py ["key", "ArrowUp", ms]. */
export function keyboard(panel, keys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']) {
  const held = new Set(), press = {};
  let onScreen = false;
  const live = () => onScreen && panel.classList.contains('active');
  const mark = () => { panel.classList.toggle('keys-on', live()); if (!live()) held.clear(); };
  new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; mark(); }, { threshold: 0.35 }).observe(panel);
  panel.addEventListener('panel:show', mark);
  panel.addEventListener('panel:hide', mark);
  const typing = (t) => t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable);
  window.addEventListener('keydown', (e) => {
    if (!live() || !keys.includes(e.key) || typing(e.target)) return;
    e.preventDefault();
    if (!held.has(e.key) && press[e.key]) press[e.key](e);
    held.add(e.key);
  });
  window.addEventListener('keyup', (e) => held.delete(e.key));
  window.addEventListener('blur', () => held.clear());
  return { held, down: (k) => held.has(k), onPress: (k, fn) => { press[k] = fn; } };
}

/* ---------- 2D pictures ---------- */

/** 2D plot canvas with device-pixel scaling. Returns { g, w, h, fit, clear } in CSS pixels (as Lab 4). */
export function plot2d(canvas) {
  const ctx = canvas.getContext('2d');
  const o = { g: ctx, w: 0, h: 0 };
  o.fit = () => {
    const r = Math.min(window.devicePixelRatio || 1, 2);
    o.w = canvas.clientWidth; o.h = canvas.clientHeight;
    if (canvas.width !== Math.round(o.w * r) || canvas.height !== Math.round(o.h * r)) {
      canvas.width = Math.round(o.w * r); canvas.height = Math.round(o.h * r);
    }
    ctx.setTransform(r, 0, 0, r, 0, 0);
  };
  o.clear = () => { o.fit(); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, o.w, o.h); };
  return o;
}

/** Linear scales for a plot2d: data ranges xr = [x0, x1], yr = [y0, y1]; pad = [left, right, top, bottom] px.
 *  Returns { X(x), Y(y), invX(px), invY(py), box: [left, top, width, height] }. Call after o.clear(). */
export function scales(o, xr, yr, pad = [40, 12, 12, 30]) {
  const [l, r, t, b] = pad, W = o.w - l - r, H = o.h - t - b;
  return {
    X: (x) => l + (x - xr[0]) / (xr[1] - xr[0]) * W,
    Y: (y) => t + (1 - (y - yr[0]) / (yr[1] - yr[0])) * H,
    invX: (px) => xr[0] + (px - l) / W * (xr[1] - xr[0]),
    invY: (py) => yr[0] + (1 - (py - t) / H) * (yr[1] - yr[0]),
    box: [l, t, W, H], xr, yr,
  };
}

/** Axes with ticks and labels in the faint grey of the clips. opt: { xticks, yticks, xlabel, ylabel, digits }. */
export function drawAxes(o, s, opt = {}) {
  const g = o.g, [l, t, W, H] = s.box;
  g.save();
  g.strokeStyle = CSS.faint; g.lineWidth = 1; g.fillStyle = CSS.grey; g.font = '11px Inter, sans-serif';
  g.beginPath(); g.moveTo(l, t); g.lineTo(l, t + H); g.lineTo(l + W, t + H); g.stroke();
  const d = opt.digits === undefined ? 1 : opt.digits;
  g.textAlign = 'center'; g.textBaseline = 'top';
  (opt.xticks || []).forEach((x) => { const px = s.X(x); g.beginPath(); g.moveTo(px, t + H); g.lineTo(px, t + H + 4); g.stroke(); g.fillText(x.toFixed(d), px, t + H + 6); });
  g.textAlign = 'right'; g.textBaseline = 'middle';
  (opt.yticks || []).forEach((y) => { const py = s.Y(y); g.beginPath(); g.moveTo(l - 4, py); g.lineTo(l, py); g.stroke(); g.fillText(y.toFixed(d), l - 6, py); });
  g.fillStyle = CSS.white; g.font = '12px Inter, sans-serif';
  if (opt.xlabel) { g.textAlign = 'right'; g.textBaseline = 'bottom'; g.fillText(opt.xlabel, l + W, t + H - 4); }
  if (opt.ylabel) { g.textAlign = 'left'; g.textBaseline = 'top'; g.fillText(opt.ylabel, l + 6, t + 2); }
  g.restore();
}

/** Polyline through (xs[i], ys[i]) in data units. */
export function drawCurve(o, s, xs, ys, stroke, width = 2, dash = null) {
  const g = o.g;
  g.save(); g.strokeStyle = stroke; g.lineWidth = width; if (dash) g.setLineDash(dash);
  g.beginPath();
  for (let i = 0; i < xs.length; i++) { const px = s.X(xs[i]), py = s.Y(ys[i]); if (i) g.lineTo(px, py); else g.moveTo(px, py); }
  g.stroke(); g.restore();
}

/** The nsig ellipse of a 2x2 covariance in data units; fill is an rgba string or null. */
export function drawEllipse(o, s, mean, cov, nsig, stroke, fill = null, width = 2) {
  const pts = S.ellipsePoints(mean, cov, nsig, 96), g = o.g;
  g.save(); g.beginPath();
  pts.forEach((p, i) => { const px = s.X(p[0]), py = s.Y(p[1]); if (i) g.lineTo(px, py); else g.moveTo(px, py); });
  g.closePath();
  if (fill) { g.fillStyle = fill; g.fill(); }
  g.strokeStyle = stroke; g.lineWidth = width; g.stroke(); g.restore();
}

/* ---------- time and units ---------- */

/** Fixed-step accumulator: calls step(h) enough times to advance by dt * speed (at most 400 steps per frame). */
export function stepper(h) {
  let acc = 0;
  return (dt, speed, step) => {
    acc += dt * speed;
    let n = 0;
    while (acc >= h && n < 400) { step(h); acc -= h; n++; }
    if (n === 400) acc = 0;
  };
}

export const deg = (r) => r * 180 / Math.PI;
export const rad = (d) => d * Math.PI / 180;
export const fmt = (x, d = 2) => Number(x).toFixed(d);
