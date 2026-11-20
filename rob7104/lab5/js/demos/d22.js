// 2.2 Exponential coordinates (clip s23_expcoords). Many carts drive the same 1 m straight command; their poses
// (x, y, theta) are lifted into a box (heading as height), where the cloud is a curved sheet, then deformed into
// exponential coordinates xi = log(mu^-1 g) + mu, where it is close to an ellipsoid. Yellow: the Gaussian fitted in
// (x, y, theta); pink: the Gaussian in exponential coordinates (fitted to the carts, or the closed form mu(t),
// Sigma(t) without samples), drawn on the floor as the highest-density regions of its (x, y) marginal (banana
// contours, 39.3 / 86.5 / 98.9 %). Numerics: sim5.js (simulateCart, groupMeanCov, sigmaClosedForm, ...); the picture
// devices are those of render/s23_expcoords.py (see js/d22_util.js). The deformation is the straight-line blend
// (1 - s) (x, y, theta) + s (xi + mu): only its two ends have a meaning.
import { S, THREE, COL, CSS, makeViewer, setCart, parkBody, loadGhost, CartCloud, FloorEllipse, Ribbon,
  val, onInput, onClick, seg, toggle, readout, ctl, plot2d } from '../common.js';
import * as U from '../d22_util.js';

// Total-variation distances to the cloud in (x, y), quoted (not recomputed): out/s23_expcoords_facts.json
// tv_shown.other_settings (= out/cart_facts.json main table), N = 200000, 40 x 40 bins, mean of 3 seeds.
const TV = {
  1: { floor: 0.029608, cart: 0.168013, fit: 0.066988, cf: 0.057652 },
  4: { floor: 0.025495, cart: 0.295215, fit: 0.148710, cf: 0.107950 },
  7: { floor: 0.027148, cart: 0.350010, fit: 0.202492, cf: 0.141525 },
};
const N_SIM = 12000, N_SHOW = 2500, N_SKEW = 6000, N_BODIES = 24, SEED = 23, SIM_DT = 1e-3, STORE = 10;
const H_THETA = 1.6, Z0 = 0.003, PUCK_X = -0.045, Z_LIGHT = 0.114;
const SOLID_MS = 700, DRIVE_MS = 2600, T_MIN_CF = 0.03;
const STEPS = [0.25, 0.5, 0.5];                      // tick steps of x, y, theta in the box

export async function init(panel) {
  // camera: the lift slider moves the target and the distance between a floor view and a box view (azimuth and
  // elevation stay as the viewer left them); drag, wheel and double-click work as everywhere
  const VIEW_FLOOR = { lookat: [0.72, 0, 0.08], distance: 2.45 }, VIEW_BOX = { lookat: [0.88, 0, 0.72], distance: 4.3 };
  const v = await makeViewer(panel, 'warehouse.mjb', { lookat: VIEW_FLOOR.lookat, distance: VIEW_FLOOR.distance, azimuth: 62, elevation: -32 });
  parkBody(v, 'est');
  parkBody(v, 'cart');
  v.forward();
  const geo = await loadGhost();
  // the shelf row (y about 2.15) hides the cloud when the camera is on its side, and clutters the box's tick numbers
  const shelfMeshes = v.meshes.filter((m, g) => m && v.data.geom_xpos[3 * g + 1] > 1.6);
  const P2 = plot2d(panel.querySelector('canvas.plot'));

  /* ---------------------------------------------------------------- data of one noise level (cached) */
  const cache = {};
  function build(DT) {
    const [vel, om] = S.straightCommand(1, 1);
    const snaps = [new Float32Array(3 * N_SHOW)];
    const run = S.simulateCart(N_SIM, vel, om, DT, 1, { dt: SIM_DT, seed: SEED, onStep: (k, pose) => {
      if ((k + 1) % STORE === 0) snaps.push(Float32Array.from(pose.subarray(0, 3 * N_SHOW)));
    } });
    const g = run.pose;
    const cart = S.fitCartesianGaussian(g);
    const { mu, Sigma, xi } = S.groupMeanCov(g);
    const muCf = S.meanPath(vel, om, 1), SigCf = S.sigmaClosedForm(vel, om, DT, 1);
    const Pc = new Float64Array(3 * N_SIM), Pe = new Float64Array(3 * N_SIM);
    for (let i = 0; i < N_SIM; i++) {
      Pc[3 * i] = g[3 * i]; Pc[3 * i + 1] = g[3 * i + 1]; Pc[3 * i + 2] = mu[2] + S.wrapAngle(g[3 * i + 2] - mu[2]);
      for (let k = 0; k < 3; k++) Pe[3 * i + k] = xi[3 * i + k] + mu[k];
    }
    // the box: 0.5 %-99.5 % range of every coordinate in both systems and both 2-sigma boxes, in steps of 0.25,
    // symmetric about the mean in y and theta (as the clip)
    const muTick = mu.map((x, k) => Math.round(x / STEPS[k]) * STEPS[k]);   // round to the tick step so the xi labels are round numbers
    const lo = [0, 0, 0], hi = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(U.quantile3(Pc, k, 0.005), U.quantile3(Pe, k, 0.005), cart.mean[k] - 2 * Math.sqrt(cart.cov[k][k]), mu[k] - 2 * Math.sqrt(Sigma[k][k]));
      hi[k] = Math.max(U.quantile3(Pc, k, 0.995), U.quantile3(Pe, k, 0.995), cart.mean[k] + 2 * Math.sqrt(cart.cov[k][k]), mu[k] + 2 * Math.sqrt(Sigma[k][k]));
      lo[k] = Math.floor(lo[k] / 0.25) * 0.25; hi[k] = Math.ceil(hi[k] / 0.25) * 0.25;
    }
    for (const k of [1, 2]) { const h = Math.max(muTick[k] - lo[k], hi[k] - muTick[k]); lo[k] = muTick[k] - h; hi[k] = muTick[k] + h; }
    const kTh = H_THETA / (hi[2] - lo[2]);
    // the shown carts (the first N_SHOW), those inside the box in both systems first
    const inb = (i) => [0, 1, 2].every((k) => Pc[3 * i + k] >= lo[k] && Pc[3 * i + k] <= hi[k] && Pe[3 * i + k] >= lo[k] && Pe[3 * i + k] <= hi[k]);
    const order = [];
    for (let i = 0; i < N_SHOW; i++) if (inb(i)) order.push(i);
    const nIn = order.length;
    for (let i = 0; i < N_SHOW; i++) if (!inb(i)) order.push(i);
    const pick = (A) => { const o = new Float64Array(3 * N_SHOW); order.forEach((j, i) => { o[3 * i] = A[3 * j]; o[3 * i + 1] = A[3 * j + 1]; o[3 * i + 2] = A[3 * j + 2]; }); return o; };
    const pc = pick(Pc), pe = pick(Pe);
    const sn = snaps.map((a) => { const o = new Float32Array(3 * N_SHOW); order.forEach((j, i) => { o[3 * i] = a[3 * j]; o[3 * i + 1] = a[3 * j + 1]; o[3 * i + 2] = a[3 * j + 2]; }); return o; });
    // live checks: skewness of the shown carts; share of all carts in the 86.5 % regions (clip facts: 0.892 / 0.918 / 0.896 at DT = 7)
    const skew = { cart: U.mardiaSkew3(Pc, N_SKEW), exp: U.mardiaSkew3(Pe, N_SKEW) };
    const share = (m, Sg) => {
      const D = U.expDensity(Sg, 0), lv = D.levels[1];
      let k = 0;
      for (let i = 0; i < N_SIM; i++) { const q = S.se2Relative(m, [g[3 * i], g[3 * i + 1], g[3 * i + 2]]); if (U.densityAt(D, q[0], q[1]) >= lv) k++; }
      return { share: k / N_SIM, mass: D.mass };
    };
    const shFit = share(mu, Sigma), shCf = share(muCf, SigCf);
    const cover = { yellowEllipse: S.ellipseCoverage(g, cart.mean, cart.cov, U.MASSES[1]), pinkFitted: shFit.share, pinkClosedForm: shCf.share, gridMass: [shFit.mass, shCf.mass] };
    return { DT, cart, mu, Sigma, muCf, SigCf, pc, pe, snaps: sn, lo, hi, kTh, muTick, nIn, skew, cover, dens: {} };
  }

  /* ---------------------------------------------------------------- 3D objects */
  const sc = v.scene;
  const basic = (color, opacity = 1, extra = {}) => new THREE.MeshBasicMaterial(Object.assign({ color, transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide, toneMapped: false }, extra));
  const quads = (rects, color, opacity) => {           // flat rectangles on the floor: [[x0, y0, x1, y1, width], ...]
    const pos = [], idx = [];
    rects.forEach(([x0, y0, x1, y1, w]) => {
      const L = Math.hypot(x1 - x0, y1 - y0), nx = -(y1 - y0) / L * w / 2, ny = (x1 - x0) / L * w / 2, b = pos.length / 3;
      pos.push(x0 + nx, y0 + ny, 0.002, x1 + nx, y1 + ny, 0.002, x1 - nx, y1 - ny, 0.002, x0 - nx, y0 - ny, 0.002);
      idx.push(b, b + 1, b + 2, b, b + 2, b + 3);
    });
    const gm = new THREE.BufferGeometry();
    gm.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); gm.setIndex(idx);
    const m = new THREE.Mesh(gm, basic(color, opacity)); sc.add(m); return m;
  };
  // the commanded path (dashed) and the start mark
  const dashes = [];
  for (let x = 0.06; x < 1.0; x += 0.09) dashes.push([x, 0, Math.min(x + 0.05, 1), 0, 0.012]);
  dashes.push([0, -0.07, 0, 0.07, 0.014], [1, -0.05, 1, 0.05, 0.01]);
  quads(dashes, 0xeeeeee, 0.55);

  const cloud = new CartCloud(sc, geo, { max: N_SHOW, bodies: N_BODIES, opacity: 0.3 });
  cloud.points.visible = false;                        // the dots are drawn below (they rise into the box)
  const BODY_OP = 0.3;

  const disc = (() => {
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const g = c.getContext('2d'), gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.55, 'rgba(255,255,255,0.95)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(c);
  })();
  const dotPos = new Float32Array(3 * N_SHOW);
  const dotGeo = new THREE.BufferGeometry();
  dotGeo.setAttribute('position', new THREE.BufferAttribute(dotPos, 3));
  const dots = new THREE.Points(dotGeo, new THREE.PointsMaterial({ color: COL.cloud, size: 0.028, map: disc, transparent: true, opacity: 0.85, depthWrite: false, alphaTest: 0.05, toneMapped: false }));
  dots.frustumCulled = false; dots.renderOrder = 4; sc.add(dots);

  // floor: yellow 1, 2, 3 sigma ellipses of the (x, y) block; pink highest-density regions of 39.3, 86.5, 98.9 %
  const yel2 = new FloorEllipse(sc, COL.est, { nsig: 2, width: 0.013, fill: 0.05 });
  const yel13 = [new Ribbon(sc, COL.est, { n: 96, opacity: 0.6 }), new Ribbon(sc, COL.est, { n: 96, opacity: 0.6 })];
  class SegBand {                                       // many short floor segments drawn as thin quads
    constructor(color, width, max = 5000, z = 0.006) {
      this.width = width; this.max = max; this.z = z;
      this.pos = new Float32Array(12 * max);
      const idx = new Uint32Array(6 * max);
      for (let i = 0; i < max; i++) idx.set([4 * i, 4 * i + 1, 4 * i + 2, 4 * i, 4 * i + 2, 4 * i + 3], 6 * i);
      const gm = new THREE.BufferGeometry();
      gm.setAttribute('position', new THREE.BufferAttribute(this.pos, 3)); gm.setIndex(new THREE.BufferAttribute(idx, 1));
      this.mesh = new THREE.Mesh(gm, basic(COL.pink, 0.95)); this.mesh.material.color.set(color);
      this.mesh.frustumCulled = false; this.mesh.renderOrder = 3; sc.add(this.mesh);
    }
    set(segs) {
      const n = Math.min(segs.length, this.max), h = this.width / 2, z = this.z;
      for (let i = 0; i < n; i++) {
        const [x1, y1, x2, y2] = segs[i], L = Math.hypot(x2 - x1, y2 - y1) || 1;
        const tx = (x2 - x1) / L * h * 0.5, ty = (y2 - y1) / L * h * 0.5, nx = -ty * 2, ny = tx * 2;
        this.pos.set([x1 - tx + nx, y1 - ty + ny, z, x2 + tx + nx, y2 + ty + ny, z, x2 + tx - nx, y2 + ty - ny, z, x1 - tx - nx, y1 - ty - ny, z], 12 * i);
      }
      this.mesh.geometry.setDrawRange(0, 6 * n);
      this.mesh.geometry.attributes.position.needsUpdate = true;
    }
    set visible(b) { this.mesh.visible = b; }
  }
  const pinkBands = [new SegBand(COL.pink, 0.006), new SegBand(COL.pink, 0.014), new SegBand(COL.pink, 0.006)];
  pinkBands[0].mesh.material.opacity = 0.65; pinkBands[2].mesh.material.opacity = 0.65;

  // the box (heading as height): grid, walls, axes and their numbers; built per noise level
  const boxGroup = new THREE.Group(); sc.add(boxGroup);
  const lineMat = new THREE.LineBasicMaterial({ color: 0x8c8c94, transparent: true, opacity: 0.3, depthWrite: false, toneMapped: false });
  const axisMat = basic(0xeeeeee, 0.9);
  const labelsA = [], labelsB = [];
  function textSprite(text, color, h = 0.07, italic = false) {
    const c = document.createElement('canvas'), g = c.getContext('2d'), px = 64;
    g.font = `${italic ? 'italic ' : ''}${px}px Inter, sans-serif`;
    c.width = Math.ceil(g.measureText(text).width) + 16; c.height = px + 20;
    g.font = `${italic ? 'italic ' : ''}${px}px Inter, sans-serif`; g.fillStyle = color; g.textBaseline = 'middle'; g.fillText(text, 8, c.height / 2);
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false, toneMapped: false }));
    sp.scale.set(h * c.width / c.height, h, 1); sp.renderOrder = 6;
    return sp;
  }
  const tickTxt = (x) => { const r = Math.round(x * 100) / 100; return (Math.abs(r) < 1e-9 ? 0 : r).toString().replace('-', '−'); };
  function buildBox(d) {
    boxGroup.clear(); labelsA.length = 0; labelsB.length = 0;
    const { lo, hi, kTh } = d, W = (p) => [p[0], p[1], Z0 + (p[2] - lo[2]) * kTh];
    const ticks = (k) => { const t = []; for (let x = Math.ceil(lo[k] / STEPS[k] - 1e-9) * STEPS[k]; x <= hi[k] + 1e-9; x += STEPS[k]) t.push(x); return t; };
    const tx = ticks(0), ty = ticks(1), tz = ticks(2), segs = [];
    const add = (a, b) => segs.push(...W(a), ...W(b));
    tx.forEach((x) => { add([x, lo[1], lo[2]], [x, hi[1], lo[2]]); add([x, hi[1], lo[2]], [x, hi[1], hi[2]]); });
    ty.forEach((y) => { add([lo[0], y, lo[2]], [hi[0], y, lo[2]]); add([hi[0], y, lo[2]], [hi[0], y, hi[2]]); });
    tz.forEach((z) => { add([lo[0], hi[1], z], [hi[0], hi[1], z]); add([hi[0], lo[1], z], [hi[0], hi[1], z]); });
    const gl = new THREE.BufferGeometry(); gl.setAttribute('position', new THREE.Float32BufferAttribute(segs, 3));
    boxGroup.add(new THREE.LineSegments(gl, lineMat));
    // three axes: x and y from the front-left corner, the vertical one on the back-left edge
    const arrow = (a, b) => {
      const A = new THREE.Vector3(...a), B = new THREE.Vector3(...b), dir = B.clone().sub(A), L = dir.length();
      const grp = new THREE.Group();
      const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, L - 0.05, 8), axisMat);
      shaft.position.y = (L - 0.05) / 2;
      const head = new THREE.Mesh(new THREE.ConeGeometry(0.018, 0.05, 12), axisMat);
      head.position.y = L - 0.025;
      grp.add(shaft, head);
      grp.position.copy(A); grp.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
      boxGroup.add(grp);
    };
    const o = W([lo[0], lo[1], lo[2]]), ob = W([lo[0], hi[1], lo[2]]);
    arrow(o, [o[0] + 1.1 * (hi[0] - lo[0]), o[1], o[2]]);
    arrow(o, [o[0], o[1] + 1.1 * (hi[1] - lo[1]), o[2]]);
    arrow(ob, [ob[0], ob[1], ob[2] + 1.05 * H_THETA]);
    const put = (set, sp, p) => { sp.position.set(...p); boxGroup.add(sp); set.push(sp); };
    const names = [['x', 'y', 'θ'], ['v₁', 'v₂', 'α']];
    [labelsA, labelsB].forEach((set, b) => {
      const sh = b ? d.muTick : [0, 0, 0], col = b ? CSS.pink : '#c9c6cf';
      tx.forEach((x) => put(set, textSprite(tickTxt(x - sh[0]), col, 0.075), [x, lo[1] - 0.1, 0.01]));
      ty.forEach((y) => put(set, textSprite(tickTxt(y - sh[1]), col, 0.075), [hi[0] + 0.12, y, 0.01]));
      tz.forEach((z) => put(set, textSprite(tickTxt(z - sh[2]), col, 0.075), [lo[0] - 0.02, hi[1] + 0.12, W([0, 0, z])[2]]));
      const nc = b ? CSS.pink : '#f2efe9';
      put(set, textSprite(names[b][0], nc, 0.14, true), [o[0] + 1.17 * (hi[0] - lo[0]), o[1], 0.05]);
      put(set, textSprite(names[b][1], nc, 0.14, true), [o[0], o[1] + 1.15 * (hi[1] - lo[1]), 0.05]);
      put(set, textSprite(names[b][2], nc, 0.14, true), [ob[0], ob[1], ob[2] + 1.12 * H_THETA]);
    });
  }

  // ellipsoids (2 sigma): translucent surface plus wire rings
  const WIRES = U.sphereWires(3, 4, 72);
  const sph = new THREE.SphereGeometry(1, 36, 18), unit = Float32Array.from(sph.attributes.position.array);
  function makeEllipsoid(color) {
    const surf = new THREE.Mesh(sph.clone(), basic(color, 0.1)); surf.frustumCulled = false; surf.renderOrder = 2;
    const wires = WIRES.map((c) => {
      const gm = new THREE.BufferGeometry(); gm.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3 * c.length), 3));
      const ln = new THREE.Line(gm, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85, depthWrite: false, toneMapped: false }));
      ln.frustumCulled = false; ln.renderOrder = 5; sc.add(ln); return ln;
    });
    sc.add(surf);
    return { surf, wires, set(map, op) {               // map: unit-sphere point -> world point
      const vis = op > 0.01;
      surf.visible = vis; wires.forEach((w) => { w.visible = vis; });
      if (!vis) return;
      surf.material.opacity = 0.1 * op;
      const P = surf.geometry.attributes.position;
      for (let i = 0; i < P.count; i++) P.setXYZ(i, ...map([unit[3 * i], unit[3 * i + 1], unit[3 * i + 2]]));
      P.needsUpdate = true;
      wires.forEach((w, k) => { const a = w.geometry.attributes.position; WIRES[k].forEach((u, i) => a.setXYZ(i, ...map(u))); a.needsUpdate = true; w.material.opacity = 0.7 * op; });
    } };
  }
  const ellY = makeEllipsoid(COL.est), ellP = makeEllipsoid(COL.pink);

  /* ---------------------------------------------------------------- state and controls */
  const st = { DT: 7, s: 0, lift: 0, kind: 'fit', yellow: true, drive: null, tRun: 1, segs: null };
  let d = null, dirty = true;
  v.onFrame(() => { const show = v.camera.position.y < 0.3 && st.lift < 0.3; shelfMeshes.forEach((m) => { m.visible = show; }); });
  const gaussOf = (kind) => (kind === 'cf' ? { mu: d.muCf, Sig: d.SigCf } : { mu: d.mu, Sig: d.Sigma });
  function densFor(kind, s, t = 1) {
    const key = `${kind}|${s.toFixed(2)}|${t.toFixed(2)}`;
    if (!d.dens[key]) {
      if (Object.keys(d.dens).length > 80) d.dens = {};
      let mu, Sig;
      if (t < 1) { const [vel, om] = S.straightCommand(1, 1); mu = S.meanPath(vel, om, t); Sig = S.sigmaClosedForm(vel, om, d.DT, t); }
      else ({ mu, Sig } = gaussOf(kind));
      const D = U.expDensity(Sig, s);
      d.dens[key] = { D, mu, lines: D.levels.map((lv) => U.toDisplay(U.contourSegments(D, lv), mu, s)) };
    }
    return d.dens[key];
  }
  const smooth = (a, b, x) => { const u = Math.min(Math.max((x - a) / (b - a), 0), 1); return u * u * (3 - 2 * u); };

  function setDT(DT) {
    readout(panel, 'info', '<span class="c-blue">simulating 12 000 carts …</span>');
    st.DT = DT;
    setTimeout(() => {
      if (!cache[DT]) cache[DT] = build(DT);
      d = cache[DT];
      buildBox(d);
      info();
      dirty = true;
    }, 30);
  }
  function info() {
    readout(panel, 'info', `skewness (Mardia, ${N_SKEW} carts; 0 for a Gaussian): <span class="c-yellow">${d.skew.cart.toFixed(2)}</span> in (<i>x, y, θ</i>), `
      + `<span class="c-pink">${d.skew.exp.toFixed(2)}</span> in exponential coordinates: smaller, not zero`);
  }

  onInput(panel, 's', () => { st.s = val(panel, 's'); dirty = true; });
  onInput(panel, 'lift', () => {
    st.lift = val(panel, 'lift'); dirty = true;
    const L = st.lift, mix3 = (a, b) => a.map((x, i) => x + L * (b[i] - x));
    const tgt = new THREE.Vector3(...mix3(VIEW_FLOOR.lookat, VIEW_BOX.lookat));
    const dir = v.camera.position.clone().sub(v.controls.target).normalize();
    v.controls.target.copy(tgt);
    v.camera.position.copy(tgt).addScaledVector(dir, VIEW_FLOOR.distance + L * (VIEW_BOX.distance - VIEW_FLOOR.distance));
    v.controls.update();
    v.homeView.lookat = tgt.toArray(); v.homeView.distance = VIEW_FLOOR.distance + L * (VIEW_BOX.distance - VIEW_FLOOR.distance);
  });
  seg(panel, 'gauss', (k) => { st.kind = k; dirty = true; });
  toggle(panel, 'yellow', (on) => { st.yellow = on; dirty = true; });
  seg(panel, 'DT', (x) => setDT(Number(x)));
  onClick(panel, 'drive', () => {
    const sl = ctl(panel, 's');
    if (Number(sl.value) !== 0) { sl.value = 0; sl.dispatchEvent(new Event('input')); }
    st.drive = { t0: performance.now() };
  });

  /* ---------------------------------------------------------------- drawing */
  const Pcur = new Float64Array(3 * N_SHOW);
  function frame() {
    let phase = 'rest';
    if (st.drive) {
      const el = performance.now() - st.drive.t0;
      if (el < SOLID_MS) { phase = 'solid'; st.tRun = 0; } else { phase = 'run'; st.tRun = Math.min((el - SOLID_MS) / DRIVE_MS, 1); }
      if (st.tRun >= 1 && el >= SOLID_MS + DRIVE_MS + 300) { st.drive = null; phase = 'rest'; st.tRun = 1; }
    }
    const driving = phase !== 'rest';
    const s = driving ? 0 : st.s, L = st.lift, n = N_SHOW;
    // current display coordinates
    if (driving) {
      const f = st.tRun * (d.snaps.length - 1), k = Math.min(Math.floor(f), d.snaps.length - 2), a = f - k, A = d.snaps[k], B = d.snaps[k + 1];
      for (let i = 0; i < 3 * n; i++) Pcur[i] = (1 - a) * A[i] + a * B[i];
      for (let i = 0; i < n; i++) Pcur[3 * i + 2] = d.mu[2] + S.wrapAngle(Pcur[3 * i + 2] - d.mu[2]);
    } else {
      for (let i = 0; i < 3 * n; i++) Pcur[i] = (1 - s) * d.pc[i] + s * d.pe[i];
    }
    // solid cart: only before the run starts (a solid cart means a known true state)
    if (phase === 'solid') setCart(v, 'cart', [0, 0, 0]); else parkBody(v, 'cart');
    // bodies of the carts (true poses on the floor) fade as the picture leaves the floor
    const bodyOp = BODY_OP * (1 - Math.max(L, s)) * (phase === 'solid' ? 0 : 1);
    cloud.inst.material.opacity = bodyOp; cloud.inst.visible = bodyOp > 0.01;
    cloud.set(driving ? Pcur : d.pc, n);
    // dots: from the cart's top light (L = 0) to height proportional to the heading (L = 1)
    const { lo, kTh } = d;
    for (let i = 0; i < n; i++) {
      const x = Pcur[3 * i], y = Pcur[3 * i + 1], th = Pcur[3 * i + 2], thc = driving ? th : d.pc[3 * i + 2];
      const ax = x + PUCK_X * Math.cos(thc), ay = y + PUCK_X * Math.sin(thc), wz = Z0 + (th - lo[2]) * kTh;
      dotPos[3 * i] = ax + L * (x - ax); dotPos[3 * i + 1] = ay + L * (y - ay); dotPos[3 * i + 2] = Z_LIGHT + L * (wz - Z_LIGHT);
    }
    dotGeo.attributes.position.needsUpdate = true;
    dotGeo.setDrawRange(0, L > 0.5 ? d.nIn : n);
    dots.visible = phase !== 'solid';
    dots.material.opacity = 0.55 + 0.35 * L;
    dots.material.size = 0.02 + 0.018 * L;
    // the box
    boxGroup.visible = L > 0.01;
    lineMat.opacity = 0.3 * L; axisMat.opacity = 0.9 * L;
    const oa = L * (1 - smooth(0.35, 0.5, s)), ob = L * smooth(0.5, 0.65, s);
    labelsA.forEach((sp) => { sp.material.opacity = oa; sp.visible = oa > 0.01; });
    labelsB.forEach((sp) => { sp.material.opacity = ob; sp.visible = ob > 0.01; });
    // yellow: Gaussian in (x, y, theta), fades out as the coordinates change
    const yOp = st.yellow && !driving ? 1 - s : 0;
    yel2.visible = yOp > 0.01; yel13.forEach((r) => { r.visible = yOp > 0.01; });
    if (yOp > 0.01) {
      yel2.set(d.cart.mean, d.cart.cov);
      yel2.band.mesh.material.opacity = 0.95 * yOp; yel2.fill.material.opacity = 0.05 * yOp;
      [1, 3].forEach((ns, j) => { yel13[j].set(S.ellipsePoints(d.cart.mean, d.cart.cov, ns, 96), 0.006); yel13[j].mesh.material.opacity = 0.6 * yOp; });
    }
    const Ly = S.gaussianFactor(d.cart.cov), W3 = (p) => [p[0], p[1], Z0 + L * (p[2] - lo[2]) * kTh + 0.004];
    ellY.set((u) => W3(S.addVec(d.cart.mean, S.mulVec(Ly, [2 * u[0], 2 * u[1], 2 * u[2]]))), yOp * L);
    // pink: Gaussian in exponential coordinates; floor contours of the deformed variable, ellipsoid in the box
    let pk = null;
    if (driving) { if (st.tRun >= T_MIN_CF) pk = densFor('cf', 0, Math.round(st.tRun * 50) / 50); } else pk = densFor(st.kind, s);
    pinkBands.forEach((b, j) => { b.visible = !!pk; if (pk) b.set(pk.lines[j]); });
    st.segs = pk;
    if (!driving) {
      const { mu, Sig } = gaussOf(st.kind), Lp = S.gaussianFactor(Sig);
      ellP.set((u) => {
        const c = S.mulVec(Lp, [2 * u[0], 2 * u[1], 2 * u[2]]), gc = S.se2Compose(mu, S.se2Exp(c));
        return W3([(1 - s) * gc[0] + s * (c[0] + mu[0]), (1 - s) * gc[1] + s * (c[1] + mu[1]), (1 - s) * gc[2] + s * (c[2] + mu[2])]);
      }, L);
    } else ellP.set(null, 0);
    v.forward();
    draw2d(driving, s);
  }

  /* ---------------------------------------------------------------- 2D: top view (= base of the box) and the distances */
  function draw2d(driving, s) {
    const o = P2, g = o.g;
    o.clear();
    const barH = 118, top = 26, left = 46, right = 14, bottom = barH + 34;
    const Wd = o.w - left - right, Hd = o.h - top - bottom;
    const { lo, hi, muTick } = d;
    const yh = Math.max(hi[1] - muTick[1], 3.3 * Math.sqrt(Math.max(d.Sigma[1][1], d.cart.cov[1][1])));
    let xr = [Math.min(lo[0], -0.15), hi[0]], yr = [muTick[1] - yh, muTick[1] + yh];
    const k = Math.min(Wd / (xr[1] - xr[0]), Hd / (yr[1] - yr[0]));
    const cx = (xr[0] + xr[1]) / 2, cy = (yr[0] + yr[1]) / 2;
    xr = [cx - Wd / 2 / k, cx + Wd / 2 / k]; yr = [cy - Hd / 2 / k, cy + Hd / 2 / k];
    const X = (x) => left + (x - xr[0]) * k, Y = (y) => top + Hd - (y - yr[0]) * k;
    // grid and axes (numbers of (v1, v2) are those of (x, y) minus the mean, as in the box)
    const exp = s > 0.5;
    g.save();
    g.font = '11px Inter, sans-serif'; g.lineWidth = 1;
    for (let x = Math.ceil(xr[0] * 2) / 2; x <= xr[1]; x += 0.5) {
      g.strokeStyle = 'rgba(140,140,148,0.22)'; g.beginPath(); g.moveTo(X(x), top); g.lineTo(X(x), top + Hd); g.stroke();
      g.fillStyle = exp ? CSS.pink : CSS.grey; g.textAlign = 'center'; g.textBaseline = 'top'; g.fillText(tickTxt(x - (exp ? muTick[0] : 0)), X(x), top + Hd + 4);
    }
    for (let y = Math.ceil(yr[0] * 2) / 2; y <= yr[1]; y += 0.5) {
      g.strokeStyle = 'rgba(140,140,148,0.22)'; g.beginPath(); g.moveTo(left, Y(y)); g.lineTo(left + Wd, Y(y)); g.stroke();
      g.fillStyle = exp ? CSS.pink : CSS.grey; g.textAlign = 'right'; g.textBaseline = 'middle'; g.fillText(tickTxt(y - (exp ? muTick[1] : 0)), left - 5, Y(y));
    }
    g.fillStyle = exp ? CSS.pink : CSS.white; g.font = 'italic 13px Inter, sans-serif';
    g.textAlign = 'right'; g.textBaseline = 'bottom'; g.fillText(exp ? 'v₁ (m)' : 'x (m)', left + Wd - 2, top + Hd - 3);
    g.textAlign = 'left'; g.textBaseline = 'top'; g.fillText(exp ? 'v₂ (m)' : 'y (m)', left + 4, top + 2);
    g.font = '12px Inter, sans-serif'; g.fillStyle = CSS.grey; g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    g.fillText(driving ? `run time ${st.tRun.toFixed(2)} s` : (s <= 0 ? 'top view: the carts on the floor' : s >= 1 ? 'exponential coordinates (v₁, v₂)' : `blend s = ${s.toFixed(2)} (only the ends have a meaning)`), left, 16);
    // the command
    if (!exp) { g.strokeStyle = 'rgba(238,238,238,0.5)'; g.setLineDash([6, 5]); g.beginPath(); g.moveTo(X(0), Y(0)); g.lineTo(X(1), Y(0)); g.stroke(); g.setLineDash([]); }
    // carts
    g.fillStyle = 'rgba(88,196,221,0.55)';
    for (let i = 0; i < N_SHOW; i++) { const px = X(Pcur[3 * i]), py = Y(Pcur[3 * i + 1]); g.fillRect(px - 1, py - 1, 2, 2); }
    // yellow ellipses
    if (st.yellow && !driving && s < 0.99) {
      g.globalAlpha = 1 - s;
      [1, 2, 3].forEach((ns) => {
        const pts = S.ellipsePoints(d.cart.mean, d.cart.cov, ns, 96);
        g.strokeStyle = CSS.est; g.lineWidth = ns === 2 ? 2.2 : 1; g.beginPath();
        pts.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])))); g.closePath(); g.stroke();
      });
      g.globalAlpha = 1;
    }
    // pink contours
    if (st.segs) {
      st.segs.lines.forEach((segs, j) => {
        g.strokeStyle = CSS.pink; g.lineWidth = j === 1 ? 2.4 : 1; g.lineCap = 'round'; g.globalAlpha = j === 1 ? 1 : 0.75;
        g.beginPath(); segs.forEach(([x1, y1, x2, y2]) => { g.moveTo(X(x1), Y(y1)); g.lineTo(X(x2), Y(y2)); }); g.stroke();
      });
      g.globalAlpha = 1;
      const m = st.segs.mu; g.fillStyle = CSS.pink; g.beginPath(); g.arc(X(m[0]), Y(m[1]), 3.5, 0, 2 * Math.PI); g.fill();
    }
    // the distances (quoted)
    const tv = TV[d.DT], by = o.h - barH + 4, bx = left, bw = Math.max(80, Wd - 50), scl = bw / 0.4;
    g.font = '12px Inter, sans-serif'; g.fillStyle = CSS.white; g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    g.fillText(`total-variation distance to the cloud in (x, y), DT = ${d.DT}`, bx, by);
    g.font = '10.5px Inter, sans-serif'; g.fillStyle = CSS.grey;
    g.fillText('N = 200 000 carts, 40 × 40 bins (from the clip’s facts file)', bx, by + 14);
    const rows = [['Gaussian in (x, y, θ)', tv.cart, CSS.est, 'fill', st.yellow], ['exp. coords, fitted', tv.fit, CSS.pink, 'fill', !driving && st.kind === 'fit'],
      ['exp. coords, closed form (no samples)', tv.cf, CSS.pink, 'line', (driving || st.kind === 'cf')]];
    rows.forEach(([lab, x, col, how, on], j) => {
      const y = by + 24 + j * 25;
      g.globalAlpha = on ? 1 : 0.4;
      g.fillStyle = col; g.font = '11px Inter, sans-serif'; g.fillText(lab, bx, y + 8);
      if (how === 'fill') { g.fillRect(bx, y + 11, x * scl, 9); } else { g.strokeStyle = col; g.lineWidth = 1.5; g.strokeRect(bx + 0.75, y + 11.75, x * scl - 1.5, 7.5); }
      g.fillStyle = CSS.white; g.fillText(x.toFixed(3), bx + x * scl + 6, y + 19);
      g.globalAlpha = 1;
    });
    const fx = bx + tv.floor * scl;
    g.strokeStyle = CSS.grey; g.setLineDash([2, 3]); g.lineWidth = 1; g.beginPath(); g.moveTo(fx, by + 26); g.lineTo(fx, by + 98); g.stroke(); g.setLineDash([]);
    g.fillStyle = CSS.grey; g.font = '10.5px Inter, sans-serif'; g.fillText(`sampling floor ${tv.floor.toFixed(3)}`, fx + 4, by + 106);
    g.restore();
  }

  /* ---------------------------------------------------------------- start */
  d = cache[7] = build(7);
  buildBox(d);
  info();
  v.onFrame(() => { if (d && (dirty || st.drive)) { dirty = false; frame(); } });
  window.addEventListener('resize', () => { dirty = true; });
  panel.addEventListener('panel:show', () => { dirty = true; });

  panel._state = () => ({
    DT: d.DT, nSim: N_SIM, nShow: N_SHOW, nInBox: d.nIn, s: st.s, lift: st.lift, gauss: st.kind, yellow: st.yellow,
    driving: !!st.drive, tRun: st.tRun,
    cartesianMean: d.cart.mean, cartesianCov: d.cart.cov,
    groupMean: d.mu, SigmaFitted: d.Sigma, muClosedForm: d.muCf, SigmaClosedForm: d.SigCf,
    SigmaDiagRatioFittedOverClosed: [0, 1, 2].map((k) => d.Sigma[k][k] / d.SigCf[k][k]),
    box: { lo: d.lo, hi: d.hi, metresPerRad: d.kTh },
    mardiaSkew: d.skew, share865: d.cover,
    pinkContourSegments: st.segs ? st.segs.lines.map((l) => l.length) : null,
    tvQuoted: TV[d.DT],
  });
}
