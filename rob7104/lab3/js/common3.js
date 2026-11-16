// Helpers shared by the Lab 3 demos: viewer setup, an instanced robot swarm, a density wall, light pools, 2-D plots.
import { MjViewer, Trail, THREE } from '../../assets/mjviewer.js';
export { MjViewer, Trail, THREE };

export const COL = { blue: 0x58c4dd, yellow: 0xffeb40, cyan: 0x33d9d9, pink: 0xd147bd, green: 0x83c167, red: 0xfc6255, white: 0xeeeeee, grey: 0x8c8c94, warm: 0xffdb8c };
export const CSS = { blue: '#58c4dd', yellow: '#ffeb40', cyan: '#33d9d9', pink: '#d147bd', green: '#83c167', red: '#fc6255', white: '#f2efe9', grey: '#9a97a3', faint: '#4a4c55', accent: '#e8b44f' };

/** Create a viewer on the panel's canvas.view3d with one of corridor.mjb / corridor3.mjb / arena.mjb. */
export async function makeViewer(panel, model, camera) {
  const canvas = panel.querySelector('canvas.view3d');
  const veil = panel.querySelector('.loading');
  const v = new MjViewer(canvas, Object.assign({ preserveDrawingBuffer: /[?&]shot/.test(location.search) }, camera));
  await v.load(new URL(`../models/${model}`, import.meta.url).href);
  if (veil) veil.classList.add('gone');
  v.autoRun();
  panel.addEventListener('panel:hide', () => v.stop());
  panel.addEventListener('panel:show', () => v.start());
  v.start();
  return v;
}

/** Move lamp i (mocap body "lamp{i}") to (x, y) at ceiling height, or hide it with null. */
export function placeLamp(v, i, xy, z = 1.6) {
  v.setMocap(`lamp${i}`, xy ? [xy[0], xy[1], z] : [0, 0, -10]);
}

/** Stop the lamp fixtures (bodies lamp0..) from casting shadows onto the floor. */
export function noLampShadows(v) {
  for (let g = 0; g < v.model.ngeom; g++) {
    const b = v.model.geom_bodyid[g];
    const name = v.mujoco.mj_id2name(v.model, v.mujoco.mjtObj.mjOBJ_BODY.value, b) || '';
    if (/^lamp\d/.test(name) && v.meshes[g]) v.meshes[g].castShadow = false;
  }
}

/** Instanced disc robots: body + glowing LED. update(xs, ys, ledRGB?) with typed arrays. */
export class SwarmMesh {
  constructor(scene, n, { r = 0.026, h = 0.022 } = {}) {
    this.n = n; this.r = r; this.h = h;
    const bodyGeo = new THREE.CylinderGeometry(r, r, h, 20); bodyGeo.rotateX(Math.PI / 2);
    const ledGeo = new THREE.CylinderGeometry(0.72 * r, 0.72 * r, 0.006, 20); ledGeo.rotateX(Math.PI / 2);
    this.body = new THREE.InstancedMesh(bodyGeo, new THREE.MeshStandardMaterial({ color: 0x4d525e, roughness: 0.5, metalness: 0.2 }), n);
    this.led = new THREE.InstancedMesh(ledGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false }), n);
    this.body.castShadow = true;
    this.led.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(3 * n), 3);
    scene.add(this.body); scene.add(this.led);
    this._m = new THREE.Matrix4();
    this.count = n;
  }
  /** xs, ys positions; colour(i) -> [r,g,b] in 0..1, or a single THREE.Color for all. */
  update(xs, ys, colour, count = this.n) {
    const m = this._m;
    this.body.count = this.led.count = count;
    const c = new THREE.Color();
    for (let i = 0; i < count; i++) {
      m.makeTranslation(xs[i], ys[i], 0.006 + this.h / 2); this.body.setMatrixAt(i, m);
      m.makeTranslation(xs[i], ys[i], 0.006 + this.h + 0.003); this.led.setMatrixAt(i, m);
      if (typeof colour === 'function') { const [r, g, b] = colour(i); c.setRGB(r, g, b, THREE.SRGBColorSpace); }
      else if (Array.isArray(colour)) c.setRGB(colour[0], colour[1], colour[2], THREE.SRGBColorSpace);
      else c.copy(colour);
      this.led.instanceColor.setXYZ(i, c.r, c.g, c.b);
    }
    this.body.instanceMatrix.needsUpdate = true; this.led.instanceMatrix.needsUpdate = true;
    this.led.instanceColor.needsUpdate = true;
  }
}

/** LED colour from light level t in 0..1: blue in the dark, yellow in the light. */
export function ledColour(t) {
  t = Math.max(0, Math.min(1, t));
  return [0.345 + (1 - 0.345) * t, 0.769 + (0.922 - 0.769) * t, 0.867 + (0.251 - 0.867) * t];
}

/** A soft warm pool of light on the floor (radial-gradient sprite plane). setLevel(0..1). */
export class LightPool {
  constructor(scene, x, y, width) {
    const c = document.createElement('canvas'); c.width = c.height = 256;
    const g = c.getContext('2d'), grd = g.createRadialGradient(128, 128, 0, 128, 128, 128);
    grd.addColorStop(0, 'rgba(255,225,160,0.85)'); grd.addColorStop(0.35, 'rgba(255,215,140,0.45)');
    grd.addColorStop(0.7, 'rgba(255,205,120,0.12)'); grd.addColorStop(1, 'rgba(255,200,120,0)');
    g.fillStyle = grd; g.fillRect(0, 0, 256, 256);
    const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
    this.mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false });
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), this.mat);
    this.mesh.position.set(x, y, 0.0045);   // above the table top (avoids z-fighting)
    scene.add(this.mesh);
    this.setWidth(width); this.setLevel(1);
  }
  setWidth(w) { this.mesh.scale.set(4.4 * w, 4.4 * w, 1); }
  setPos(x, y) { this.mesh.position.x = x; this.mesh.position.y = y; }
  setLevel(a) { this.mat.opacity = Math.max(0, Math.min(1, a)); this.mesh.visible = a > 0.01; }
}

/**
 * A vertical density wall beside a corridor, in the plane y = y0: histogram bars plus curves.
 * update(hist, curves) with hist (length nb) and curves = [{ x: Float64Array, y: Float64Array }] in density units.
 */
export class DensityWall {
  constructor(scene, { lo = -1.5, hi = 1.5, nb = 40, y0 = 0.3, z0 = 0.06, height = 0.55, vmax = 2, curveColours = [COL.yellow] } = {}) {
    Object.assign(this, { lo, hi, nb, y0, z0, height, vmax });
    const w = (hi - lo) / nb;
    const geo = new THREE.BoxGeometry(w * 0.92, 0.008, 1);
    this.bars = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({ color: COL.blue, transparent: true, opacity: 0.6, toneMapped: false }), nb);
    scene.add(this.bars);
    this.curves = curveColours.map((col) => {
      const line = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: col, toneMapped: false }));
      line.frustumCulled = false; scene.add(line); return line;
    });
    const base = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(lo, y0, z0), new THREE.Vector3(hi, y0, z0)]),
      new THREE.LineBasicMaterial({ color: COL.grey }));
    scene.add(base);
    this._m = new THREE.Matrix4();
  }
  update(hist, curves = []) {
    const s = this.height / this.vmax, w = (this.hi - this.lo) / this.nb, m = this._m;
    for (let k = 0; k < this.nb; k++) {
      const hgt = Math.max(1e-4, Math.min(hist[k] * s, this.height * 1.25));
      m.makeScale(1, 1, hgt).setPosition(this.lo + (k + 0.5) * w, this.y0, this.z0 + hgt / 2);
      this.bars.setMatrixAt(k, m);
    }
    this.bars.instanceMatrix.needsUpdate = true;
    curves.forEach((c, j) => {
      if (!this.curves[j]) return;
      const pts = [];
      for (let i = 0; i < c.x.length; i++) pts.push(new THREE.Vector3(c.x[i], this.y0 - 0.006, this.z0 + Math.min(c.y[i] * s, this.height * 1.25)));
      this.curves[j].geometry.setFromPoints(pts);
      this.curves[j].visible = true;
    });
    for (let j = curves.length; j < this.curves.length; j++) this.curves[j].visible = false;
  }
}

// ---------------------------------------------------------------- controls and plots (same contract as Lab 4)
export function ctl(panel, name) {
  const el = panel.querySelector(`[data-ctl="${name}"]`);
  if (!el) throw new Error(`no control ${name}`);
  return el;
}
export const val = (panel, name) => Number(ctl(panel, name).value);
export function onInput(panel, name, fn) { ctl(panel, name).addEventListener('input', fn); }
export function onClick(panel, name, fn) { ctl(panel, name).addEventListener('click', fn); }
export function readout(panel, name, html) { ctl(panel, name).innerHTML = html; }
export function seg(panel, name, fn) {
  const box = ctl(panel, name);
  box.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    box.querySelectorAll('button').forEach((o) => o.classList.toggle('on', o === b));
    fn(b.dataset.v);
  }));
  return () => box.querySelector('button.on').dataset.v;
}
export function plot2d(canvas) {
  const ctx = canvas.getContext('2d');
  const o = { g: ctx, w: 0, h: 0 };
  o.fit = () => {
    const r = Math.min(window.devicePixelRatio || 1, 2);
    o.w = canvas.clientWidth; o.h = canvas.clientHeight;
    if (canvas.width !== Math.round(o.w * r) || canvas.height !== Math.round(o.h * r)) { canvas.width = Math.round(o.w * r); canvas.height = Math.round(o.h * r); }
    ctx.setTransform(r, 0, 0, r, 0, 0);
  };
  o.clear = () => { o.fit(); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, o.w, o.h); };
  return o;
}
export function stepper(h) {
  let acc = 0;
  return (dt, speed, step) => {
    acc += dt * speed; let n = 0;
    while (acc >= h && n < 2000) { step(h); acc -= h; n++; }
    if (n === 2000) acc = 0;
  };
}
