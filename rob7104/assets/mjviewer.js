// Shared 3D viewer for the ROB7104 lab pages: a MuJoCo model (WebAssembly) drawn with three.js,
// with mouse orbit / zoom / pan. See README.md in this folder for the interface.
//
// Pages must declare an import map so that the bare specifier "three" resolves:
//   <script type="importmap">{"imports": {"three": "../assets/vendor/three.module.min.js"}}</script>

import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

export { THREE };

const HERE = new URL('.', import.meta.url);
let mujocoPromise = null;

/** Load the MuJoCo WebAssembly module once (about 10 MB, 2.5 MB gzipped). */
export function loadMujoco() {
  if (!mujocoPromise) {
    mujocoPromise = import(new URL('vendor/mujoco/mujoco.js', HERE).href).then((m) => m.default());
  }
  return mujocoPromise;
}

let fileCounter = 0;

/** Fetch a compiled model (.mjb, same MuJoCo version as the module) or an .xml without external assets. */
export async function loadModel(url) {
  const mujoco = await loadMujoco();
  const res = await fetch(url);
  if (!res.ok) throw new Error(`cannot fetch ${url}: ${res.status}`);
  if (/\.xml($|\?)/.test(url)) return mujoco.MjModel.from_xml_string(await res.text());
  const bytes = new Uint8Array(await res.arrayBuffer());
  const name = `model_${fileCounter++}.mjb`;
  const vfs = new mujoco.MjVFS();
  vfs.addBuffer(name, bytes);
  const model = mujoco.MjModel.from_binary_path(name, vfs);
  vfs.deleteFile(name);
  vfs.delete();
  return model;
}

function gridTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#23262e';
  g.fillRect(0, 0, 512, 512);
  g.fillStyle = '#1e2128';
  g.fillRect(0, 0, 256, 256);
  g.fillRect(256, 256, 256, 256);
  g.strokeStyle = '#4a5261';
  g.lineWidth = 3;
  g.strokeRect(0, 0, 512, 512);
  g.strokeRect(0, 0, 256, 256);
  g.strokeRect(256, 256, 256, 256);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 8;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export class MjViewer {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} opts  { lookat:[x,y,z], distance, azimuth, elevation (degrees, MuJoCo convention), fov,
   *                         floor: true, floorSize: 40, shadows: true, minDistance, maxDistance }
   */
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.opts = Object.assign({ lookat: [0, 0, 1], distance: 5, azimuth: 90, elevation: -10, fov: 38,
      floor: true, floorSize: 40, shadows: true }, opts);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: !!opts.preserveDrawingBuffer });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = this.opts.shadows;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000);
    this.scene.fog = new THREE.Fog(0x000000, 14, 38);

    this.camera = new THREE.PerspectiveCamera(this.opts.fov, 16 / 9, 0.02, 80);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = this.opts.minDistance || 0.3;
    this.controls.maxDistance = this.opts.maxDistance || 25;
    this.controls.maxPolarAngle = Math.PI * 0.497;          // stay above the floor
    this.setCamera(this.opts);
    this.homeView = { lookat: [...this.opts.lookat], distance: this.opts.distance,
      azimuth: this.opts.azimuth, elevation: this.opts.elevation };

    this.scene.add(new THREE.HemisphereLight(0xe6ecff, 0x2a2d38, 1.7));
    this.sun = new THREE.DirectionalLight(0xfff4e0, 2.8);
    this.sun.position.set(-2.5, -4, 7);
    this.sun.castShadow = this.opts.shadows;
    this.sun.shadow.mapSize.set(2048, 2048);
    const sc = this.sun.shadow.camera;
    sc.left = -7; sc.right = 7; sc.top = 7; sc.bottom = -7; sc.near = 0.5; sc.far = 25;
    this.sun.shadow.bias = -0.0004;
    this.scene.add(this.sun);
    this.scene.add(this.sun.target);
    const fill = new THREE.DirectionalLight(0x8fa4ff, 0.55);
    fill.position.set(4, -3, 4);
    this.scene.add(fill);

    if (this.opts.floor) {
      const tex = gridTexture();
      const s = this.opts.floorSize;
      tex.repeat.set(s / 2, s / 2);
      this.floor = new THREE.Mesh(new THREE.PlaneGeometry(s, s),
        new THREE.MeshStandardMaterial({ map: tex, roughness: 0.55, metalness: 0.1 }));
      this.floor.receiveShadow = true;
      this.scene.add(this.floor);
    }

    this.meshes = [];
    this.frameCallbacks = [];
    this.running = false;
    this.time = 0;
    this._names = {};
    this._loop = this._loop.bind(this);
    this._resize();
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(canvas);
    canvas.addEventListener('dblclick', () => this.setCamera(this.homeView));
  }

  /** Camera in MuJoCo free-camera terms. Azimuth 90 looks along +y; elevation is negative when looking down. */
  setCamera({ lookat, distance, azimuth, elevation }) {
    const az = THREE.MathUtils.degToRad(azimuth), el = THREE.MathUtils.degToRad(elevation);
    const f = new THREE.Vector3(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el));
    const target = new THREE.Vector3(...lookat);
    this.camera.position.copy(target).addScaledVector(f, -distance);
    this.controls.target.copy(target);
    this.controls.update();
  }

  _resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Load a model and build the three.js objects for its visible geoms. */
  async load(url) {
    this.mujoco = await loadMujoco();
    this.model = await loadModel(url);
    this.data = new this.mujoco.MjData(this.model);
    this._build();
    this.forward();
    return this;
  }

  _material(g) {
    const m = this.model;
    let rgba = Array.from(m.geom_rgba.slice(4 * g, 4 * g + 4));
    let emission = 0, texture = null, rough = 0.55, metal = 0.05;
    const mat = m.geom_matid[g];
    if (mat >= 0) {
      const mr = Array.from(m.mat_rgba.slice(4 * mat, 4 * mat + 4));
      const def = rgba[0] === 0.5 && rgba[1] === 0.5 && rgba[2] === 0.5 && rgba[3] === 1;
      if (def) rgba = mr;
      emission = m.mat_emission[mat];
      rough = 1 - 0.75 * m.mat_shininess[mat];
      metal = Math.min(0.6, m.mat_specular[mat] * 0.6);
      const nrole = m.mat_texid.length / m.nmat;
      const tid = m.mat_texid[mat * nrole + 1];                   // mjTEXROLE_RGB
      if (tid >= 0) texture = this._texture(tid);
    }
    const material = new THREE.MeshStandardMaterial({
      color: texture ? 0xffffff : new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace),
      map: texture, roughness: rough, metalness: metal,
      transparent: rgba[3] < 1, opacity: rgba[3], depthWrite: rgba[3] >= 1,
    });
    if (emission > 0) {
      material.emissive = texture ? new THREE.Color(0xffffff) : material.color.clone();
      material.emissiveMap = texture;
      material.emissiveIntensity = emission * 0.5;
    }
    return material;
  }

  _texture(tid) {
    this._tex = this._tex || {};
    if (this._tex[tid]) return this._tex[tid];
    const m = this.model;
    const w = m.tex_width[tid], h = m.tex_height[tid], nc = m.tex_nchannel[tid], adr = Number(m.tex_adr[tid]);
    const src = m.tex_data.slice(adr, adr + w * h * nc);
    const rgba = new Uint8Array(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      rgba[4 * i] = src[nc * i];
      rgba[4 * i + 1] = src[nc * i + (nc > 1 ? 1 : 0)];
      rgba[4 * i + 2] = src[nc * i + (nc > 2 ? 2 : 0)];
      rgba[4 * i + 3] = nc > 3 ? src[nc * i + 3] : 255;
    }
    const t = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat);
    t.colorSpace = THREE.SRGBColorSpace;
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.magFilter = THREE.LinearFilter;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.generateMipmaps = true;
    t.needsUpdate = true;
    this._tex[tid] = t;
    return t;
  }

  _meshGeometry(id) {
    this._geo = this._geo || {};
    if (this._geo[id]) return this._geo[id];
    const m = this.model;
    const va = m.mesh_vertadr[id], fa = m.mesh_faceadr[id], nf = m.mesh_facenum[id];
    const na = m.mesh_normaladr[id], ta = m.mesh_texcoordadr[id];
    const V = m.mesh_vert, N = m.mesh_normal, F = m.mesh_face, FN = m.mesh_facenormal;
    const T = m.mesh_texcoord, FT = m.mesh_facetexcoord;
    const pos = new Float32Array(nf * 9), nor = new Float32Array(nf * 9);
    const uv = ta >= 0 ? new Float32Array(nf * 6) : null;
    for (let f = 0; f < nf; f++) {
      for (let k = 0; k < 3; k++) {
        const vi = va + F[3 * (fa + f) + k], ni = na + FN[3 * (fa + f) + k];
        pos.set([V[3 * vi], V[3 * vi + 1], V[3 * vi + 2]], 9 * f + 3 * k);
        nor.set([N[3 * ni], N[3 * ni + 1], N[3 * ni + 2]], 9 * f + 3 * k);
        if (uv) {
          const ti = ta + FT[3 * (fa + f) + k];
          uv.set([T[2 * ti], T[2 * ti + 1]], 6 * f + 2 * k);
        }
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    if (uv) geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    this._geo[id] = geo;
    return geo;
  }

  _geometry(g) {
    const m = this.model, T = this.mujoco.mjtGeom;
    const type = m.geom_type[g];
    const s = m.geom_size.slice(3 * g, 3 * g + 3);
    if (type === T.mjGEOM_SPHERE.value) return new THREE.SphereGeometry(s[0], 32, 20);
    if (type === T.mjGEOM_CAPSULE.value) {
      const geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 20);
      geo.rotateX(Math.PI / 2);
      return geo;
    }
    if (type === T.mjGEOM_CYLINDER.value) {
      const geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 48);
      geo.rotateX(Math.PI / 2);
      return geo;
    }
    if (type === T.mjGEOM_ELLIPSOID.value) {
      const geo = new THREE.SphereGeometry(1, 32, 20);
      geo.scale(s[0], s[1], s[2]);
      return geo;
    }
    if (type === T.mjGEOM_BOX.value) return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    if (type === T.mjGEOM_MESH.value) return this._meshGeometry(m.geom_dataid[g]);
    return null;                                             // planes are replaced by the viewer's own floor
  }

  _build() {
    for (const o of this.meshes) if (o) this.scene.remove(o);
    this.meshes = [];
    const m = this.model;
    for (let g = 0; g < m.ngeom; g++) {
      let mesh = null;
      if (m.geom_group[g] < 3 && m.geom_rgba[4 * g + 3] > 0) {
        const geo = this._geometry(g);
        if (geo) {
          mesh = new THREE.Mesh(geo, this._material(g));
          const solid = !mesh.material.transparent;
          mesh.castShadow = solid;
          mesh.receiveShadow = solid;
          mesh.matrixAutoUpdate = false;
          this.scene.add(mesh);
        }
      }
      this.meshes.push(mesh);
    }
  }

  /** Rebuild the geometry of one geom after its size was changed in the model. */
  refreshGeom(g) {
    const mesh = this.meshes[g];
    if (!mesh) return;
    mesh.geometry.dispose();
    mesh.geometry = this._geometry(g);
  }

  id(type, name) {
    const key = type + ':' + name;
    if (!(key in this._names)) {
      const t = this.mujoco.mjtObj['mjOBJ_' + type.toUpperCase()].value;
      this._names[key] = this.mujoco.mj_name2id(this.model, t, name);
      if (this._names[key] < 0) throw new Error(`no ${type} named ${name}`);
    }
    return this._names[key];
  }

  setJoint(name, value) { this.data.qpos[this.model.jnt_qposadr[this.id('joint', name)]] = value; }
  getJoint(name) { return this.data.qpos[this.model.jnt_qposadr[this.id('joint', name)]]; }

  /** pos: [x,y,z]; pitch (rotation about +y, radians) or quat [w,x,y,z] optional. */
  setMocap(name, pos, pitch, quat) {
    const i = this.model.body_mocapid[this.id('body', name)];
    if (pos) this.data.mocap_pos.set(pos, 3 * i);
    if (quat) this.data.mocap_quat.set(quat, 4 * i);
    else if (pitch !== undefined) this.data.mocap_quat.set([Math.cos(pitch / 2), 0, Math.sin(pitch / 2), 0], 4 * i);
  }

  sitePos(name) {
    const i = this.id('site', name);
    return Array.from(this.data.site_xpos.slice(3 * i, 3 * i + 3));
  }

  forward() { this.mujoco.mj_forward(this.model, this.data); this.sync(); }
  step() { this.mujoco.mj_step(this.model, this.data); }

  /** Copy geom poses from MuJoCo to three.js. */
  sync() {
    const xp = this.data.geom_xpos, xm = this.data.geom_xmat;
    for (let g = 0; g < this.meshes.length; g++) {
      const mesh = this.meshes[g];
      if (!mesh) continue;
      const a = 9 * g, b = 3 * g;
      mesh.matrix.set(xm[a], xm[a + 1], xm[a + 2], xp[b], xm[a + 3], xm[a + 4], xm[a + 5], xp[b + 1],
        xm[a + 6], xm[a + 7], xm[a + 8], xp[b + 2], 0, 0, 0, 1);
    }
  }

  /** cb(dt, t) runs once per animation frame before drawing; dt is capped at 1/20 s. */
  onFrame(cb) { this.frameCallbacks.push(cb); return this; }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  _loop(now) {
    if (!this.running) return;
    const dt = Math.min((now - (this._last || now)) / 1000, 0.05);
    this._last = now;
    this.time += dt;
    for (const cb of this.frameCallbacks) cb(dt, this.time);
    this.render();
    requestAnimationFrame(this._loop);
  }

  start() {
    if (this.running) return;
    this.running = true;
    this._last = 0;
    requestAnimationFrame(this._loop);
  }

  stop() { this.running = false; }

  /** Start/stop automatically with the page's "live:on"/"live:off" and tab events (see site.js). */
  autoRun() {
    this.canvas.dataset.live = '1';
    this.canvas.addEventListener('live:on', () => this.start());
    this.canvas.addEventListener('live:off', () => this.stop());
    return this;
  }
}

/** Fading trail behind a moving point (line with vertex colours). */
export class Trail {
  constructor(scene, color, length = 60) {
    this.n = length;
    this.count = 0;
    this.color = new THREE.Color(color);
    this.pos = new Float32Array(length * 3);
    this.col = new Float32Array(length * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(this.col, 3));
    this.line = new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true }));
    this.line.frustumCulled = false;
    scene.add(this.line);
  }

  push(p) {
    this.pos.copyWithin(0, 3);
    this.pos.set(p, (this.n - 1) * 3);
    this.count = Math.min(this.count + 1, this.n);
    for (let i = 0; i < this.n; i++) {
      const a = i < this.n - this.count ? 0 : Math.pow((i - (this.n - this.count) + 1) / this.count, 1.5);
      this.col.set([this.color.r * a, this.color.g * a, this.color.b * a], 3 * i);
    }
    if (this.count < this.n) for (let i = 0; i < this.n - this.count; i++) this.pos.set(p, 3 * i);
    this.line.geometry.attributes.position.needsUpdate = true;
    this.line.geometry.attributes.color.needsUpdate = true;
  }

  clear() { this.count = 0; }
}
