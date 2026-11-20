// Lab 6, demo 1.2: Newton's method and curvature.
// The UR5e reaches the default target with two joints (shoulder lift, elbow). Pick a start on the joint-space map
// (or the clip's starts A, B, C), then step: Newton's method jumps to the stationary point of the local quadratic
// model, drawn as a surface beside the arm (bowl green, dome red, saddle yellow) and as contour lines on the map.
// Safeguarded Newton uses H + tau I ("mirror" shift, delta = 1e-3) with Armijo backtracking; gradient descent uses
// the fixed step 1/L. The basin map colours every start of a 160 x 160 grid by where the chosen method ends.
// Numerics: sim6.js only (gradientDescent, newton, safeguardedNewton, basinMap, classify, curvatureShift).
// Finding F1 (webcore.md): no mid-path iterate digits of pure Newton are shown, only step counts, end points,
// model types and coarse numbers (two digits, whole millimetres).
import { S, THREE, makeArmViewer, planeCamera, Landscape, marksFor, marker, planeToWorld, COL, CSS, onClick, seg,
  toggle, readout, fmt, GhostArms, Line3D } from '../common.js';

const STARTS = { A: [-0.9, 1.3], B: [1.3, 0.3], C: [-0.2, 0.5] };   // out/s12_newton_facts.json, "starts"
const METH = {
  gd: { name: 'gradient descent', css: CSS.grey, col: COL.grey, dur: 0.03, hold: 0 },
  newton: { name: "Newton's method", css: CSS.pink, col: COL.pink, dur: 0.9, hold: 0.5 },
  safe: { name: 'safeguarded Newton', css: CSS.white, col: COL.white, dur: 0.9, hold: 0.5 },
};
const KIND = { bowl: [COL.green, CSS.green], dome: [COL.red, CSS.red], saddle: [COL.yellow, CSS.yellow],
  flat: [COL.grey, CSS.grey] };
const END = { 'elbow up': CSS.green, 'elbow down': CSS.blue, 'stretched towards the target': CSS.yellow,
  'folded, upper arm away from the target': CSS.yellow, 'folded, upper arm towards the target': CSS.yellow,
  'stretched away from the target': CSS.red };
const TYPE_WORD = { min: 'minimum', saddle: 'saddle', max: 'maximum', degenerate: 'degenerate point' };
const BASIN_N = 160;
const MODEL_AT = [1.0, 0.62];    // plane point (m) where the 3D model floats: 1.28 m from the shoulder, outside the reach
const MODEL_R3 = 0.2;           // radius of the 3D model disc (m)
const MODEL_H3 = 0.13;          // largest height of the 3D model surface (m): drawn to shape, height not to scale
const TILT = 32 * Math.PI / 180;
const NR = 12, NT = 48, MAX_GHOSTS = 24, TOL_1MM = 1e-3;

const rgba = (css, a) => { const n = parseInt(css.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255, a]; };
const smooth = (u) => u * u * (3 - 2 * u);

export async function init(panel) {
  const two = new S.TwoJointArm(), t = S.DEFAULT_TARGET;
  const f = (q) => two.cost(q, t), g = (q) => two.grad(q, t), h = (q) => two.hess(q, t);
  const st = two.stationaryPoints(t);
  const cb = two.curvatureBounds(t), L = Math.max(-cb[0], cb[1]), alphaGD = 1 / L;   // once per target

  const v = await makeArmViewer(panel, {
    camera: planeCamera(two, { at: [0.12, 0.08], distance: 2.75, elevation: -9, yaw: -10 }), q: two.fullQ(STARTS.A) });
  const land = new Landscape(panel.querySelector('canvas.land'));
  land.setCost(f, marksFor(two, t));

  // ---- 3D objects: target, tool path, earlier iterates, the next iterate, the model surface
  const tW = planeToWorld(two, t);
  marker(v.scene, COL.green, 0.026).position.set(...tW);
  marker(v.scene, COL.green, 0.06, 0.16).position.set(...tW);
  const trails = {}, ghosts = {}, nexts = {};
  for (const m of Object.keys(METH)) {
    trails[m] = new Line3D(v.scene, METH[m].col, 4000);
    ghosts[m] = new GhostArms(v.scene, MAX_GHOSTS, { color: METH[m].col, opacity: m === 'safe' ? 0.12 : 0.16, radius: 0.014 });
    nexts[m] = new GhostArms(v.scene, 1, { color: METH[m].col, opacity: 0.4, radius: 0.022 });
    ghosts[m].show(0); nexts[m].show(0);
  }

  const ex = new THREE.Vector3(...two.ex), nn = new THREE.Vector3(...two.normal), ez = new THREE.Vector3(0, 0, 1);
  const ay = nn.clone().multiplyScalar(Math.cos(TILT)).addScaledVector(ez, Math.sin(TILT));   // d2: away and up
  const az = ez.clone().multiplyScalar(Math.cos(TILT)).addScaledVector(nn, -Math.sin(TILT));  // height: up, to the camera
  const group = new THREE.Group();
  group.matrixAutoUpdate = false;
  group.matrix.makeBasis(ex, ay, az).setPosition(...planeToWorld(two, MODEL_AT));
  group.matrixWorldNeedsUpdate = true;
  v.scene.add(group);
  const nV = 1 + NR * NT, vid = (i, j) => (i === 0 ? 0 : 1 + (i - 1) * NT + (j % NT));
  const surfPos = new THREE.BufferAttribute(new Float32Array(3 * nV), 3);
  const surfGeo = new THREE.BufferGeometry();
  surfGeo.setAttribute('position', surfPos);
  const tri = [], seg2 = [];
  for (let j = 0; j < NT; j++) {
    tri.push(0, vid(1, j), vid(1, j + 1));
    for (let i = 1; i < NR; i++) tri.push(vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j), vid(i + 1, j + 1), vid(i, j + 1));
    for (const i of [4, 8, 12]) seg2.push(vid(i, j), vid(i, j + 1));
    if (j % 6 === 0) for (let i = 0; i < NR; i++) seg2.push(vid(i, j), vid(i + 1, j));
  }
  surfGeo.setIndex(tri);
  const surfMat = new THREE.MeshStandardMaterial({ color: COL.green, emissive: COL.green, emissiveIntensity: 0.3,
    roughness: 0.6, metalness: 0, transparent: true, opacity: 0.5, side: THREE.DoubleSide, depthWrite: false });
  const surf = new THREE.Mesh(surfGeo, surfMat);
  const wireGeo = new THREE.BufferGeometry();
  wireGeo.setAttribute('position', surfPos);
  wireGeo.setIndex(seg2);
  const wireMat = new THREE.LineBasicMaterial({ color: COL.green, transparent: true, opacity: 0.85 });
  const wire = new THREE.LineSegments(wireGeo, wireMat);
  const here = new THREE.Mesh(new THREE.SphereGeometry(0.013, 16, 10), new THREE.MeshBasicMaterial({ color: COL.white }));
  const centreMat = new THREE.MeshBasicMaterial({ color: COL.pink });
  const centre = new THREE.Mesh(new THREE.SphereGeometry(0.013, 16, 10), centreMat);
  const stepPos = new Float32Array(3 * 25);
  const stepGeo = new THREE.BufferGeometry();
  stepGeo.setAttribute('position', new THREE.BufferAttribute(stepPos, 3));
  const stepMat = new THREE.LineBasicMaterial({ color: COL.pink });
  const stepLine = new THREE.Line(stepGeo, stepMat);
  for (const o of [surf, wire, here, centre, stepLine]) { o.frustumCulled = false; group.add(o); }
  const tag = panel.querySelector('[data-d12="model"]'), legend3d = panel.querySelector('[data-d12="legend3d"]');

  // ---- state
  let method = 'newton', start = STARTS.A.slice(), startName = 'A';
  let r = null, k = 0, anim = null, hold = 0, playing = false, M = null, modelImg = null, end = null, to1mm = -1;
  let showModel = true, showBasin = false, dirtyPose = true, dirty2D = true;
  const trailPts = [], basins = {};

  function solve() {
    const q0 = start.slice();
    if (method === 'gd') r = S.gradientDescent(f, g, q0, alphaGD, { maxIter: 5000 });
    else if (method === 'newton') r = S.newton(f, g, h, q0);
    else r = S.safeguardedNewton(f, g, h, q0, { delta: S.SAFEGUARD_DELTA, shift: 'mirror' });
    const last = r.x[r.nIter], i = S.classify(last, r.converged, st);
    end = i >= 0 ? st[i] : null;
    to1mm = r.x.findIndex((q) => Math.hypot(...S.sub(two.p(q), t)) <= TOL_1MM);
    k = 0; anim = null; hold = 0; playing = false;
    trailPts.length = 0;
    for (const m of Object.keys(METH)) { trails[m].set([]); ghosts[m].show(0); nexts[m].show(0); }
    restAt();
    if (showBasin) setBasin();
    legend3d.innerHTML = `<span style="color:${CSS.green}">●</span> target &nbsp; <span style="color:${METH[method].css}">—</span> `
      + `tool path, ${METH[method].name}` + (method === 'gd' ? '' : ' &nbsp; translucent arms: earlier and next iterates');
  }

  // the local quadratic model at iterate k: m(d) = f + g.d + d.Hm d / 2, Hm = H (+ tau I for the safeguard)
  function modelAt(kk) {
    const q = r.x[kk], gq = g(q), H = h(q);
    let tau = 0;
    if (method === 'safe') tau = kk < r.extra.tau.length ? r.extra.tau[kk] : S.curvatureShift(H, S.SAFEGUARD_DELTA, 'mirror');
    const Hm = [[H[0][0] + tau, H[0][1]], [H[1][0], H[1][1] + tau]];
    const ev = S.eig2(Hm), eps = 1e-12;
    const kind = ev[0] > eps ? 'bowl' : ev[1] < -eps ? 'dome' : (ev[0] < -eps && ev[1] > eps) ? 'saddle' : 'flat';
    const det = Hm[0][0] * Hm[1][1] - Hm[0][1] * Hm[1][0];
    const p = Math.abs(det) > 1e-14 ? [-(Hm[1][1] * gq[0] - Hm[0][1] * gq[1]) / det, -(-Hm[1][0] * gq[0] + Hm[0][0] * gq[1]) / det] : null;
    const pn = p ? Math.hypot(p[0], p[1]) : 0;
    const R = Math.min(1.4, Math.max(0.5, 1.25 * pn));
    const val = (d) => gq[0] * d[0] + gq[1] * d[1] + 0.5 * (Hm[0][0] * d[0] * d[0] + 2 * Hm[0][1] * d[0] * d[1] + Hm[1][1] * d[1] * d[1]);
    return { q, qw: S.wrap(q), g: gq, Hm, tau, ev, eigH: S.eig2(H), kind, p, pn, R, val };
  }

  function setSurface() {
    const s = MODEL_R3 / M.R, z = new Float64Array(nV);
    let zmax = 0;
    for (let i = 0; i <= NR; i++) for (let j = 0; j < (i === 0 ? 1 : NT); j++) {
      const rr = M.R * i / NR, th = 2 * Math.PI * j / NT, d = [rr * Math.cos(th), rr * Math.sin(th)];
      const id = vid(i, j), zz = M.val(d);
      z[id] = zz; zmax = Math.max(zmax, Math.abs(zz));
      surfPos.setXYZ(id, d[0] * s, d[1] * s, 0);
    }
    const kz = zmax > 1e-15 ? MODEL_H3 / zmax : 0;
    for (let id = 0; id < nV; id++) surfPos.setZ(id, z[id] * kz);
    surfPos.needsUpdate = true;
    surfGeo.computeVertexNormals();
    surfGeo.computeBoundingSphere();
    const [c] = KIND[M.kind];
    surfMat.color.setHex(c); surfMat.emissive.setHex(c); wireMat.color.setHex(c);
    const mc = METH[method].col;
    centreMat.color.setHex(mc); stepMat.color.setHex(mc);
    const showStep = method !== 'gd' && M.p && M.pn > 1e-9;
    centre.visible = showStep && M.pn <= M.R;
    stepLine.visible = showStep;
    if (showStep) {
      const umax = Math.min(1, M.R / M.pn);
      for (let n = 0; n < 25; n++) {
        const d = [M.p[0] * umax * n / 24, M.p[1] * umax * n / 24];
        stepPos.set([d[0] * s, d[1] * s, M.val(d) * kz + 0.002], 3 * n);
      }
      stepGeo.attributes.position.needsUpdate = true;
      if (centre.visible) centre.position.set(M.p[0] * s, M.p[1] * s, M.val(M.p) * kz);
    }
    here.position.set(0, 0, 0);
    tag.innerHTML = `local quadratic model${method === 'safe' && M.tau > 0 ? ', modified curvature H + τI' : ''}: `
      + `<b style="color:${KIND[M.kind][1]}">${M.kind}</b>`;
    tag.style.color = KIND[M.kind][1];
  }

  // contour picture of the model on the map: n x n pixels over [q - R, q + R]^2, transparent outside the disc
  function buildModelImage() {
    const n = 120, c = document.createElement('canvas'); c.width = n; c.height = n;
    const gc = c.getContext('2d'), im = gc.createImageData(n, n), V = new Float64Array(n * n), inside = new Uint8Array(n * n);
    let vmax = 0;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const d = [((i + 0.5) / n * 2 - 1) * M.R, (1 - (j + 0.5) / n * 2) * M.R], kk = j * n + i;
      if (d[0] * d[0] + d[1] * d[1] > M.R * M.R) continue;
      inside[kk] = 1; V[kk] = M.val(d); vmax = Math.max(vmax, Math.abs(V[kk]));
    }
    const step = (vmax || 1) / 7, band = (x) => Math.floor(x / step);
    const line = rgba(KIND[M.kind][1], 235), fill = rgba(KIND[M.kind][1], 46);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const kk = j * n + i;
      if (!inside[kk]) continue;
      const b = band(V[kk]);
      const edge = (i + 1 < n && inside[kk + 1] && band(V[kk + 1]) !== b) || (j + 1 < n && inside[kk + n] && band(V[kk + n]) !== b);
      im.data.set(edge ? line : fill, 4 * kk);
    }
    gc.putImageData(im, 0, 0);
    return c;
  }

  function restAt() {
    M = modelAt(k);
    modelImg = buildModelImage();
    setSurface();
    const gh = ghosts[method], n = Math.min(k, MAX_GHOSTS);
    if (method !== 'gd') { for (let i = 0; i < n; i++) gh.set(i, two.fullQ(r.x[k - n + i])); gh.show(n); }
    const nx = nexts[method];
    if (method !== 'gd' && k < r.nIter) { nx.set(0, two.fullQ(r.x[k + 1])); nx.show(1); } else nx.show(0);
    dirtyPose = true; dirty2D = true;
    info();
  }

  const modelVisible = () => showModel && !anim && !(playing && method === 'gd');
  function beginStep() {
    if (k >= r.nIter) { playing = false; info(); return; }
    anim = { u: 0 };
    nexts[method].show(0);
    dirty2D = true;
  }
  function finishStep() {
    anim = null; k += 1;
    restAt();
    if (playing) { if (k < r.nIter) hold = METH[method].hold || 1e-9; else playing = false; }
    info();
  }
  const currentQ = () => {
    if (!anim) return r.x[k];
    const a = r.x[k], b = r.x[k + 1], u = method === 'gd' ? anim.u : smooth(anim.u);
    return [a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])];
  };

  // ---- the map
  const pxU = (q) => { const b = land.box; return [b.x + (q[0] + Math.PI) / (2 * Math.PI) * b.s, b.y + (Math.PI - q[1]) / (2 * Math.PI) * b.s]; };
  // a straight segment in joint space from a (wrapped) by the vector d, drawn on the torus (copies shifted by 2 pi)
  function torusSegs(gx, segs) {
    const b = land.box;
    gx.save(); gx.beginPath(); gx.rect(b.x, b.y, b.s, b.s); gx.clip();
    let K = 1;
    for (const [, d] of segs) K = Math.max(K, Math.ceil(Math.max(Math.abs(d[0]), Math.abs(d[1])) / (2 * Math.PI)) + 1);
    K = Math.min(K, 6);
    gx.beginPath();
    for (let sx = -K; sx <= K; sx++) for (let sy = -K; sy <= K; sy++) {
      for (const [a, d] of segs) {
        const p0 = pxU([a[0] + 2 * Math.PI * sx, a[1] + 2 * Math.PI * sy]), p1 = pxU([a[0] + d[0] + 2 * Math.PI * sx, a[1] + d[1] + 2 * Math.PI * sy]);
        gx.moveTo(p0[0], p0[1]); gx.lineTo(p1[0], p1[1]);
      }
    }
    gx.stroke(); gx.restore();
  }
  function draw2D() {
    const gx = land.draw(), b = land.box, col = METH[method].css;
    // local model (contours in a disc around the iterate) and the Newton step to its centre
    if (modelVisible()) {
      const hw = M.R / (2 * Math.PI) * b.s;
      gx.save(); gx.beginPath(); gx.rect(b.x, b.y, b.s, b.s); gx.clip();
      gx.imageSmoothingEnabled = true;
      for (const sx of [-1, 0, 1]) for (const sy of [-1, 0, 1]) {
        const [cx, cy] = pxU([M.qw[0] + 2 * Math.PI * sx, M.qw[1] + 2 * Math.PI * sy]);
        if (cx + hw < b.x || cx - hw > b.x + b.s || cy + hw < b.y || cy - hw > b.y + b.s) continue;
        gx.drawImage(modelImg, cx - hw, cy - hw, 2 * hw, 2 * hw);
        gx.strokeStyle = KIND[M.kind][1]; gx.lineWidth = 1.2; gx.globalAlpha = 0.9;
        gx.beginPath(); gx.arc(cx, cy, hw, 0, 2 * Math.PI); gx.stroke(); gx.globalAlpha = 1;
      }
      gx.restore();
      if (method !== 'gd' && M.p && M.pn > 1e-9) {
        gx.save(); gx.strokeStyle = col; gx.lineWidth = 1.6; gx.setLineDash([4, 3]);
        torusSegs(gx, [[M.qw, M.p]]);
        gx.restore();
        if (M.pn <= 6 * Math.PI) {
          const [cx, cy] = pxU(S.wrap([M.qw[0] + M.p[0], M.qw[1] + M.p[1]]));
          gx.save(); gx.strokeStyle = col; gx.lineWidth = 2; gx.beginPath(); gx.arc(cx, cy, 6, 0, 2 * Math.PI); gx.stroke(); gx.restore();
        }
      }
    }
    // the path so far (the iterates) and the current point
    const segs = [];
    for (let i = 0; i < k; i++) segs.push([S.wrap(r.x[i]), S.sub(r.x[i + 1], r.x[i])]);
    const q = currentQ();
    if (anim) segs.push([S.wrap(r.x[k]), S.sub(q, r.x[k])]);
    gx.save(); gx.strokeStyle = col; gx.lineWidth = method === 'gd' ? 1.3 : 1.8; gx.lineJoin = 'round';
    torusSegs(gx, segs);
    gx.restore();
    if (method !== 'gd') for (let i = 0; i <= k; i++) land.dot(r.x[i], col, 2.6);
    const [sx0, sy0] = land.toPx(r.x[0]);
    gx.save(); gx.strokeStyle = CSS.white; gx.lineWidth = 1.4; gx.beginPath(); gx.arc(sx0, sy0, 7.5, 0, 2 * Math.PI); gx.stroke(); gx.restore();
    land.dot(q, CSS.white, 5);
    // the end point, once reached
    if (k === r.nIter && !anim && end) {
      const [ex0, ey0] = land.toPx(end.q);
      gx.save(); gx.strokeStyle = END[end.name]; gx.lineWidth = 2; gx.beginPath(); gx.arc(ex0, ey0, 10, 0, 2 * Math.PI); gx.stroke(); gx.restore();
    }
  }

  // ---- basin map: every start of a 160 x 160 grid coloured by where the chosen method ends
  function setBasin() {
    if (!basins[method]) {
      const B = S.basinMap(method === 'safe' ? 'safeguarded_newton' : method, two, t, BASIN_N, { alpha: alphaGD, stationary: st, maxIter: 5000 });
      const share = {}, n2 = BASIN_N * BASIN_N;
      for (const l of B.label) { const key = l >= 0 ? st[l].name : 'none'; share[key] = (share[key] || 0) + 1 / n2; }
      basins[method] = { B, share };
    }
    const { B } = basins[method], n = BASIN_N, cols = st.map((s) => rgba(END[s.name], 200));
    land.setField(n, (q) => {
      const i = Math.min(n - 1, Math.max(0, Math.floor((q[0] + Math.PI) / (2 * Math.PI) * n)));
      const j = Math.min(n - 1, Math.max(0, Math.floor((q[1] + Math.PI) / (2 * Math.PI) * n)));
      const l = B.label[i * n + j];
      return l >= 0 ? cols[l] : [60, 60, 66, 200];
    });
    basinInfo();
  }
  function basinInfo() {
    if (!showBasin || !basins[method]) { readout(panel, 'shares', ''); return; }
    const sh = basins[method].share, pc = (x) => `${Math.round(100 * (x || 0))}&nbsp;%`;
    const sad = st.filter((s) => s.type === 'saddle').reduce((a, s) => a + (sh[s.name] || 0), 0);
    readout(panel, 'shares', `basin map, ${METH[method].name}, ${BASIN_N}&times;${BASIN_N} starts end at: `
      + `<b style="color:${CSS.green}">elbow up ${pc(sh['elbow up'])}</b> · <b style="color:${CSS.blue}">elbow down ${pc(sh['elbow down'])}</b> · `
      + `<b style="color:${CSS.yellow}">saddle ${pc(sad)}</b> · <b style="color:${CSS.red}">maximum ${pc(sh['stretched away from the target'])}</b>`
      + (sh.none ? ` · not settled ${pc(sh.none)}` : ''));
  }

  // ---- readout
  function info() {
    const q = r.x[k], err = 1000 * Math.hypot(...S.sub(two.p(q), t));
    let s = `<b style="color:${METH[method].css}">${METH[method].name}</b> · step ${k} of ${r.nIter} · `;
    s += `model <b style="color:${KIND[M.kind][1]}">${M.kind}</b> (λ ${fmt(M.ev[0], 2)}, ${fmt(M.ev[1], 2)}`
      + (method === 'safe' ? `; τ ${fmt(M.tau, 2)}` : '') + ') · ';
    s += `tool ${err < 1 ? '&lt; 1' : Math.round(err)} mm from the target`;
    if (k === r.nIter) {
      s += end ? ` · ends at the <b style="color:${END[end.name]}">${TYPE_WORD[end.type]}</b> (${end.name}) after ${r.nIter} steps (‖g‖ ≤ 1e-9)`
        + (to1mm >= 0 ? `, ${to1mm} to 1 mm` : ', never within 1 mm')
        : ` · <b style="color:${CSS.red}">did not settle</b> in ${r.nIter} steps`;
    }
    readout(panel, 'info', s);
  }

  // ---- controls
  seg(panel, 'method', (m) => { method = m; solve(); });
  const startSeg = panel.querySelector('[data-ctl="start"]');
  seg(panel, 'start', (name) => { start = STARTS[name].slice(); startName = name; solve(); });
  land.onPick((q0) => {
    start = q0; startName = 'picked';
    startSeg.querySelectorAll('button').forEach((b) => b.classList.remove('on'));
    solve();
  });
  onClick(panel, 'step', () => { playing = false; hold = 0; if (!anim) beginStep(); });
  onClick(panel, 'run', () => {
    if (k >= r.nIter && !anim) { solve(); }
    playing = true; hold = 0;
    if (!anim) beginStep();
  });
  onClick(panel, 'reset', () => solve());
  toggle(panel, 'model', (on) => { showModel = on; dirty2D = true; });
  toggle(panel, 'basin', (on) => {
    showBasin = on;
    if (on) setBasin(); else { land.setField(BASIN_N, null); basinInfo(); }
    dirty2D = true;
  });
  window.addEventListener('resize', () => { dirty2D = true; });
  panel.addEventListener('panel:show', () => { dirty2D = true; });

  // ---- frame loop
  const tmp = new THREE.Vector3();
  v.onFrame((dt) => {
    if (anim) {
      anim.u += dt / METH[method].dur;
      dirtyPose = true; dirty2D = true;
      while (anim && anim.u >= 1) {             // gradient descent may take several short steps in one frame
        const left = anim.u - 1;
        anim.u = 1; finishStep();
        if (playing && method === 'gd' && k < r.nIter) { beginStep(); anim.u = left; hold = 0; }
      }
    } else if (hold > 0) {
      hold -= dt;
      if (hold <= 0) { hold = 0; if (playing) beginStep(); }
    }
    if (dirtyPose) {
      const q = currentQ();
      v.setArm(two.fullQ(q)); v.forward();
      const p = v.tool(), last = trailPts[trailPts.length - 1];
      if (!last || Math.hypot(p[0] - last[0], p[1] - last[1], p[2] - last[2]) > 0.003) {
        if (trailPts.length >= 4000) trailPts.splice(0, 1000);
        trailPts.push([p[0], p[1], p[2]]);
        trails[method].set(trailPts);
      }
      dirtyPose = false;
    }
    const mv = modelVisible();
    group.visible = mv;
    if (dirty2D) { draw2D(); dirty2D = false; }
    // the label of the 3D model follows the camera
    tmp.set(0, -MODEL_R3 * 1.1, 0).applyMatrix4(group.matrix).project(v.camera);
    const cv = v.canvas, tw = tag.offsetWidth, th = tag.offsetHeight;
    const lx = Math.min(cv.clientWidth - tw / 2 - 6, Math.max(tw / 2 + 6, ((tmp.x + 1) / 2) * cv.clientWidth));
    const ly = Math.min(cv.clientHeight - th - 34, Math.max(8, ((1 - tmp.y) / 2) * cv.clientHeight + 6));
    tag.style.opacity = mv && tmp.z < 1 ? '1' : '0';
    tag.style.left = `${lx}px`;
    tag.style.top = `${ly}px`;
  });

  solve();

  panel._state = () => {
    const q = currentQ(), tool = v.tool(), pw = planeToWorld(two, two.p(q));
    return {
      method, start: startName, q0: r.x[0], k, nIter: r.nIter, converged: r.converged, status: r.status,
      animating: !!anim, playing,
      endsAt: end ? { name: end.name, type: end.type } : null, stepsTo1mm: to1mm,
      toolErrorMm: 1000 * Math.hypot(...S.sub(two.p(q), t)),
      viewerVsSim6Mm: 1000 * Math.hypot(tool[0] - pw[0], tool[1] - pw[1], tool[2] - pw[2]),
      model: M ? { kind: M.kind, eig: M.ev, eigH: M.eigH, tau: M.tau, stepNorm: M.pn, radius: M.R, visible: modelVisible() } : null,
      L, alphaGD, curvatureBounds: cb,
      basin: showBasin && basins[method] ? basins[method].share : null,
      ghosts: method === 'gd' ? 0 : ghosts[method].shown, trailPoints: trailPts.length,
    };
  };
}
