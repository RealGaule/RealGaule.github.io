// Shared by the two "waiter" demos (d21, d32): a Franka Panda carries a glass of water sideways.
//
// The water is NOT a fluid simulation. It is the first slosh mode, modelled as an equivalent pendulum
//   th'' + 2 zeta w th' + w^2 th = -a / l ,   w = sqrt(g * 1.841 / R * tanh(1.841 * h / R)),   l = g / w^2,
// where a is the hand's acceleration along x. The liquid surface is flat and tilted by th (higher on the +x side
// when th > 0). It spills when the surface reaches the rim.
import * as S from './sim.js';
import { makeViewer, THREE, COL, CSS } from './common.js';

export const GLASS = { R: 0.035, H: 0.12, fill: 0.09 };
export const ZE = 0.02;
export const OM = Math.sqrt(S.G * 1.841 / GLASS.R * Math.tanh(1.841 * GLASS.fill / GLASS.R));   // about 22.7 rad/s
export const WD = OM * Math.sqrt(1 - ZE * ZE);
export const PERIOD = 2 * Math.PI / OM;                                                          // about 0.277 s
export const LEQ = S.G / (OM * OM);
export const SPILL = Math.atan((GLASS.H - GLASS.fill) / GLASS.R);                                // about 40.6 degrees
export const PEND = S.singlePendulum(OM, ZE);
export const PINK = '#d147bd';
export const CAMERA = { lookat: [0, -0.50, 0.80], distance: 0.85, azimuth: 68, elevation: -14, minDistance: 0.2 };

/** Free motion of the linear slosh model: state [th, w] after a time tau without forcing (closed form). */
export function freeResponse(s, tau) {
  const e = Math.exp(-ZE * OM * tau), c = Math.cos(WD * tau), sn = Math.sin(WD * tau), k = ZE * OM / WD;
  return [e * (s[0] * (c + k * sn) + s[1] / WD * sn), e * (s[1] * (c - k * sn) - s[0] * OM * OM / WD * sn)];
}

/** Envelope amplitude of the free linear slosh that starts from state s = [th, w]. */
export const amplitude = (s) => Math.hypot(s[0], (s[1] + ZE * OM * s[0]) / WD);

/** Set a slider's value and its <output> without firing an input event. */
export function setSlider(el, value) {
  el.value = value;
  const out = el.parentElement.querySelector('output');
  if (out) out.textContent = Number(el.value).toFixed(Number(el.dataset.fmt || 2)) + (el.dataset.unit || '');
}

// ---------------------------------------------------------------- liquid with a flat, tilted surface
/** Water in the glass: a cylinder whose top is the plane z = fill + x tan(th), cut off at the rim and the bottom. */
export class Liquid {
  constructor(scene, n = 72) {
    this.n = n;
    const R = GLASS.R - 0.0004;
    this.R = R;
    this.cos = new Float32Array(n + 1);
    const side = new Float32Array((n + 1) * 6), sideN = new Float32Array((n + 1) * 6), sideI = [];
    for (let j = 0; j <= n; j++) {
      const c = Math.cos(2 * Math.PI * j / n), s = Math.sin(2 * Math.PI * j / n);
      this.cos[j] = c;
      side.set([R * c, R * s, 0.0008, R * c, R * s, GLASS.fill], 6 * j);
      sideN.set([c, s, 0, c, s, 0], 6 * j);
      if (j < n) sideI.push(2 * j, 2 * j + 2, 2 * j + 3, 2 * j, 2 * j + 3, 2 * j + 1);
    }
    const top = new Float32Array(n * 3), topN = new Float32Array(n * 3), topI = [];
    for (let j = 0; j < n; j++) top.set([R * this.cos[j], R * Math.sin(2 * Math.PI * j / n), GLASS.fill], 3 * j);
    // vertices j and n - j have the same x, hence the same height: strips between them are exactly planar
    for (let j = 0; j < n / 2; j++) {
      const a = j, b = j + 1, c = n - j - 1, d = (n - j) % n;
      if (a !== d) topI.push(a, c, d);
      if (b !== c) topI.push(a, b, c);
    }
    const mk = (pos, nor, idx) => {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
      geo.setIndex(idx);
      return geo;
    };
    this.side = new THREE.Mesh(mk(side, sideN, sideI), new THREE.MeshStandardMaterial({
      color: 0x2f9fd0, roughness: 0.3, metalness: 0, emissive: 0x1d6f96, emissiveIntensity: 0.55 }));
    this.top = new THREE.Mesh(mk(top, topN, topI), new THREE.MeshStandardMaterial({
      color: 0xbfeeff, roughness: 0.2, metalness: 0, emissive: 0x58c4dd, emissiveIntensity: 0.5, side: THREE.DoubleSide }));
    const ring = (radius, z, color, opacity) => {
      const p = new Float32Array(n * 3);
      for (let j = 0; j < n; j++) p.set([radius * this.cos[j], radius * Math.sin(2 * Math.PI * j / n), z], 3 * j);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(p, 3));
      return new THREE.LineLoop(geo, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }));
    };
    this.edge = ring(R + 0.0003, GLASS.fill, 0xffffff, 0.9);                 // outline of the surface
    this.rest = ring(GLASS.R + 0.0036, GLASS.fill, 0xffeb40, 0.75);          // level at rest, on the outside of the glass
    this.rim = ring(GLASS.R + 0.0036, GLASS.H + 0.0028, 0xffffff, 0.8);      // rim of the glass
    this.group = new THREE.Group();
    this.group.add(this.side, this.top, this.edge, this.rest, this.rim);
    for (const o of this.group.children) o.frustumCulled = false;
    this.side.castShadow = true;
    scene.add(this.group);
    this.set([0, 0, 0], 0);
  }

  /** centre: centre of the glass; th: tilt of the surface (radians). */
  set(centre, th) {
    const { n, R } = this, tan = Math.tan(Math.max(-1.25, Math.min(1.25, th)));
    const sp = this.side.geometry.attributes.position, tp = this.top.geometry.attributes.position;
    const tn = this.top.geometry.attributes.normal, ep = this.edge.geometry.attributes.position;
    const nz = 1 / Math.hypot(1, tan);
    for (let j = 0; j <= n; j++) {
      const z = Math.max(0.004, Math.min(GLASS.H, GLASS.fill + R * this.cos[j] * tan));
      sp.array[6 * j + 5] = z;
      if (j < n) {
        tp.array[3 * j + 2] = z;
        ep.array[3 * j + 2] = z + 0.0004;
        tn.array[3 * j] = -tan * nz; tn.array[3 * j + 1] = 0; tn.array[3 * j + 2] = nz;
      }
    }
    sp.needsUpdate = tp.needsUpdate = tn.needsUpdate = ep.needsUpdate = true;
    this.group.position.set(centre[0], centre[1], centre[2] - GLASS.H / 2);
  }
}

// ---------------------------------------------------------------- spilled droplets on ballistic paths
export class Droplets {
  constructor(scene, max = 160) {
    this.max = max;
    this.mesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.0052, 10, 8),
      new THREE.MeshStandardMaterial({ color: COL.blue, roughness: 0.25, emissive: COL.blue, emissiveIntensity: 0.35 }), max);
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = true;
    this.p = new Float32Array(max * 3);
    this.v = new Float32Array(max * 3);
    this.m = new THREE.Matrix4();
    this.clear();
    scene.add(this.mesh);
  }

  clear() { this.count = 0; this.next = 0; this.mesh.count = 0; }

  emit(p, v) {
    const i = this.next;
    this.next = (this.next + 1) % this.max;
    this.count = Math.min(this.count + 1, this.max);
    this.p.set(p, 3 * i);
    this.v.set(v, 3 * i);
  }

  step(dt) {
    const floor = 0.0052;
    for (let i = 0; i < this.count; i++) {
      const a = 3 * i;
      if (this.p[a + 2] > floor) {
        this.v[a + 2] -= S.G * dt;
        for (let k = 0; k < 3; k++) this.p[a + k] += this.v[a + k] * dt;
        if (this.p[a + 2] < floor) this.p[a + 2] = floor;
      }
      this.m.makeTranslation(this.p[a], this.p[a + 1], this.p[a + 2]);
      this.mesh.setMatrixAt(i, this.m);
    }
    this.mesh.count = this.count;
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}

// ---------------------------------------------------------------- the arm with the glass
/**
 * Load the viewer, the model and the inverse-kinematics table.
 * Returns { v, show(x, th, vx, dtSim), clearDrops(), xRange }.
 */
export async function loadWaiter(panel, camera = CAMERA) {
  const ik = await (await fetch(new URL('../data/waiter_ik.json', import.meta.url))).json();
  const v = await makeViewer(panel, 'waiter.mjb', camera);
  // the glass of the model is very faint on the black background: make its wall catch more light
  const cup = v.id('body', ik.cup_body);
  for (let g = 0; g < v.model.ngeom; g++) {
    const mesh = v.meshes[g];
    if (!mesh || v.model.geom_bodyid[g] !== cup) continue;
    mesh.material.opacity = Math.min(0.6, mesh.material.opacity + 0.12);
    mesh.material.roughness = 0.08;
    mesh.material.emissive = new THREE.Color(0x9fc8e8);
    mesh.material.emissiveIntensity = 0.22;
    mesh.renderOrder = 2;                                           // after the liquid inside it
  }
  const liquid = new Liquid(v.scene);
  const drops = new Droplets(v.scene);
  const xMax = ik.x0 + ik.dx * (ik.n - 1);
  let spillAcc = 0, seed = 1;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };   // repeatable scatter

  /** Put the glass centre at x: joint angles by linear interpolation in the table (hand moves on a straight line). */
  function place(x) {
    const u = Math.max(0, Math.min(ik.n - 1, (x - ik.x0) / ik.dx));
    const i = Math.min(Math.floor(u), ik.n - 2), f = u - i;
    for (let k = 0; k < 7; k++) v.setJoint(ik.joints[k], ik.q[i][k] * (1 - f) + ik.q[i + 1][k] * f);
    for (const name of ik.fingers) v.setJoint(name, ik.finger_value);
    v.setMocap(ik.cup_body, [x, ik.cup_y, ik.cup_z]);
  }

  /** Draw one frame: arm and glass at x, surface tilted by th; vx and dtSim (simulated seconds) move the droplets. */
  function show(x, th, vx = 0, dtSim = 0) {
    const xc = Math.max(ik.x0, Math.min(xMax, x));
    place(xc);
    v.forward();
    const centre = [xc, ik.cup_y, ik.cup_z];
    liquid.set(centre, th);
    if (Math.abs(th) > SPILL && dtSim > 0) {
      const sx = Math.sign(th);                                   // water goes over the rim on the high side
      spillAcc += dtSim;
      while (spillAcc > 0.005) {
        spillAcc -= 0.005;
        drops.emit([xc + sx * GLASS.R, ik.cup_y + 0.03 * rnd(), ik.cup_z + GLASS.H / 2 + 0.004],
          [vx + sx * (0.45 + 0.3 * rnd()), 0.25 * rnd(), 0.55 + 0.3 * rnd()]);
      }
    }
    drops.step(dtSim);
  }

  place(-0.15);
  v.forward();
  liquid.set([-0.15, ik.cup_y, ik.cup_z], 0);
  return { v, ik, show, clearDrops: () => { drops.clear(); spillAcc = 0; }, xRange: [ik.x0, xMax] };
}

// ---------------------------------------------------------------- small 2D plotting helpers
export const FONT = '11px Inter, system-ui, sans-serif';

/** Axes box in CSS pixels with data ranges xr, yr. */
export class Ax {
  constructor(g, x, y, w, h, xr, yr) { Object.assign(this, { g, x, y, w, h, xr, yr }); }
  X(v) { return this.x + (v - this.xr[0]) / (this.xr[1] - this.xr[0]) * this.w; }
  Y(v) { return this.y + this.h - (v - this.yr[0]) / (this.yr[1] - this.yr[0]) * this.h; }

  /** Left and bottom spines, zero line, ticks. xt / yt: arrays of [value, label]. */
  frame(xt = [], yt = [], unit = '') {
    const g = this.g;
    g.save();
    g.lineWidth = 1; g.strokeStyle = CSS.faint; g.fillStyle = CSS.grey; g.font = FONT;
    g.beginPath();
    g.moveTo(this.x, this.y); g.lineTo(this.x, this.y + this.h); g.lineTo(this.x + this.w, this.y + this.h);
    if (this.yr[0] < 0 && this.yr[1] > 0) { g.moveTo(this.x, this.Y(0)); g.lineTo(this.x + this.w, this.Y(0)); }
    g.stroke();
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const [v, s] of yt) {
      g.beginPath(); g.moveTo(this.x - 3, this.Y(v)); g.lineTo(this.x, this.Y(v)); g.stroke();
      g.fillText(s, this.x - 5, this.Y(v));
    }
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (const [v, s] of xt) {
      if (v < this.xr[0] - 1e-9 || v > this.xr[1] + 1e-9) continue;
      g.beginPath(); g.moveTo(this.X(v), this.y + this.h); g.lineTo(this.X(v), this.y + this.h + 3); g.stroke();
      if (s) g.fillText(s, this.X(v), this.y + this.h + 5);
    }
    if (unit) { g.textAlign = 'left'; g.textBaseline = 'top'; g.fillText(unit, this.x + 4, this.y + 1); }
    g.restore();
  }

  /** Run fn with drawing clipped to the box. */
  clip(fn) {
    const g = this.g;
    g.save();
    g.beginPath(); g.rect(this.x, this.y - 1, this.w, this.h + 2); g.clip();
    fn();
    g.restore();
  }

  /** Polyline through (xs[i], ys[i]) for i in [i0, i1). */
  line(xs, ys, color, width = 1.5, { dash = null, alpha = 1, i0 = 0, i1 = xs.length } = {}) {
    const g = this.g;
    if (i1 - i0 < 2) return;
    g.save();
    g.strokeStyle = color; g.lineWidth = width; g.globalAlpha = alpha; g.lineJoin = 'round';
    if (dash) g.setLineDash(dash);
    g.beginPath();
    g.moveTo(this.X(xs[i0]), this.Y(ys[i0]));
    for (let i = i0 + 1; i < i1; i++) g.lineTo(this.X(xs[i]), this.Y(ys[i]));
    g.stroke();
    g.restore();
  }

  vline(x, color, dash = [4, 3], alpha = 1) {
    const g = this.g;
    if (x < this.xr[0] || x > this.xr[1]) return;
    g.save();
    g.strokeStyle = color; g.lineWidth = 1; g.globalAlpha = alpha; g.setLineDash(dash);
    g.beginPath(); g.moveTo(this.X(x), this.y); g.lineTo(this.X(x), this.y + this.h); g.stroke();
    g.restore();
  }

  hline(y, color, dash = [4, 3], alpha = 1) {
    const g = this.g;
    if (y < this.yr[0] || y > this.yr[1]) return;
    g.save();
    g.strokeStyle = color; g.lineWidth = 1; g.globalAlpha = alpha; g.setLineDash(dash);
    g.beginPath(); g.moveTo(this.x, this.Y(y)); g.lineTo(this.x + this.w, this.Y(y)); g.stroke();
    g.restore();
  }
}

export function label(g, s, x, y, color = CSS.white, align = 'left', font = '12px Inter, system-ui, sans-serif') {
  g.save();
  g.font = font; g.fillStyle = color; g.textAlign = align; g.textBaseline = 'alphabetic';
  g.fillText(s, x, y);
  g.restore();
}

/** Symmetric tick marks for a range [-r, r]: [[-s, '-s'], [0, '0'], [s, 's']] with a round step s. */
export function symTicks(r, unit = '') {
  let s = 1;
  for (const c of [0.5, 1, 2, 5, 10, 15, 20, 30, 45, 60, 90]) if (c <= 0.8 * r) s = c;
  return [[-s, `−${s}${unit}`], [0, '0'], [s, `${s}${unit}`]];
}
