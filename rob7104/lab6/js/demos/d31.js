// 3.1 Calculus of variations: the unknown is the whole path of the hand.
// A string of beads from A to B (or to anywhere on the shelf) with a pillar's smooth penalty in the way; relax it
// by gradient descent (time-lapse) or by safeguarded Newton steps (PathProblem of sim6.js, the numerics of
// render/s31_variational.py); the UR5e's finger tips follow the current path (inverse kinematics per frame,
// tool axis vertical). Right: the same string seen from above, beads coloured by the Euler-Lagrange residual,
// and the largest bead distance to the optimal path against the number of steps.
import { S, THREE, makeArmViewer, marker, Line3D, COL, CSS, onInput, onClick, readout, seg, toggle, plot2d, world, fmt } from '../common.js';

// ---- the scene of the clip (render/s31_variational.py; out/s31_variational_facts.json "problem" and "room")
const A2 = S.PATH.A2, B2 = S.PATH.B2;                         // (-0.40, 0), (0.40, 0): (along A-B, lateral), metres
const SIGMA = S.PATH.pillarSigma, WEIGHT = S.PATH.pillarWeight, PIL_R = S.PATH.pillarRadius;
const D0 = 0.62, Z_TIP = 0.15, TIP = 0.064;                   // line A-B in front of the base; path plane; finger tips
const SHELF_DEG = 80, SHELF_S = [-0.42, 0.10], SHELF_DEPTH = 0.16;
const TAU = [Math.cos(SHELF_DEG * Math.PI / 180), Math.sin(SHELF_DEG * Math.PI / 180)];
const HAND = { bend: 0.30, skew: 0.07, wrinkles: [[11, 0.030], [17, 0.015]], along: [[2, 0.030], [13, 0.008]] };
const R_TOOL = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]];           // tool axis down
const Q_SEED = [Math.PI, -1.0, 1.4, -2.0, -Math.PI / 2, 0.8]; // elbow up, tool down (pan set per target)
const BEAD_R = 0.0105, LINK_R = 0.0032, PILLAR_ABOVE = 0.11;
const ZONES = [[1.0, 0.20], [1.75, 0.12], [2.5, 0.07], [3.25, 0.035]];
const RES_LO = 0.03, RES_HI = 30;                             // residual colour scale (log), m / T^2
const TOL = 1e-3;                                             // stopping rule: every bead within 1 mm of the optimum
const WRINKLE_J = 11;                                         // the start's first wrinkle (11 half-waves)
const CAMERA = { lookat: [0.36, 0.14, 1.20], distance: 2.0, azimuth: 131, elevation: -31 };
const GD_RATE = 1.1, GD_V0 = 4;                               // time-lapse: steps(t) = V0/RATE (e^{RATE t} - 1)
const PASS = 4.0, REST = 0.8, RAMP = 0.08;                    // hand travel: seconds per pass, pause at the ends

const uvToWorld = (y) => world([D0 + y[1], y[0], Z_TIP]);   // path (along, lateral) -> room (as the clip)
const siteTarget = (y) => [D0 + y[1], y[0], Z_TIP + TIP];     // tool point (base frame) that puts the tips on y
const lerp = (a, b, u) => a + (b - a) * u;
const ease = (x) => { x = Math.max(0, Math.min(1, x)); return x * x * (3 - 2 * x); };
const C_CALM = [230, 232, 240], C_YEL = [255, 235, 64], C_RED = [252, 98, 85], C_GREEN = [131, 193, 103];
const C_GD = 'rgb(184, 189, 199)';

function resColor(r) {
  const t = Math.max(0, Math.min(1, Math.log10(Math.max(r, 1e-300) / RES_LO) / Math.log10(RES_HI / RES_LO)));
  if (t <= 0.45) return C_CALM.map((c, k) => c + (t / 0.45) * (C_YEL[k] - c));
  return C_YEL.map((c, k) => c + ((t - 0.45) / 0.55) * (C_RED[k] - c));
}
const rgb = (c, a = 1) => `rgba(${c.map((v) => Math.round(v)).join(',')},${a})`;

function travel(u) {                                           // constant speed with short ramps (as the clip)
  u = Math.max(0, Math.min(1, u));
  const v = 1 / (1 - RAMP);
  if (u < RAMP) return v * u * u / (2 * RAMP);
  if (u > 1 - RAMP) return 1 - v * (1 - u) ** 2 / (2 * RAMP);
  return v * (u - RAMP / 2);
}

function beadDist(Y, Yr) {
  let m = 0;
  for (let i = 0; i < Y.length; i++) m = Math.max(m, Math.hypot(Y[i][0] - Yr[i][0], Y[i][1] - Yr[i][1]));
  return m;
}

function makeProblem(M, lat, free) {
  const pot = new S.Pillar([0, lat], SIGMA, WEIGHT);
  return free ? new S.PathProblem(A2, null, M, { potential: pot, shelf: [B2, TAU], shelfBounds: SHELF_S })
    : new S.PathProblem(A2, B2, M, { potential: pot });
}

/** The clip's "path drawn by hand": straight line + a skewed bend towards the robot + two wrinkles + unevenness. */
function handDrawn(prob) {
  const Y = prob.straight();
  prob.t.forEach((ti, i) => {
    const s = ti / prob.T;
    Y[i][1] += -(HAND.bend * Math.sin(Math.PI * s) + HAND.skew * Math.sin(2 * Math.PI * s));
    for (const [j, a] of HAND.wrinkles) Y[i][1] += a * Math.sin(j * Math.PI * s);
    for (const [j, a] of HAND.along) Y[i][0] += a * Math.sin(j * Math.PI * s);
  });
  return prob.nodes(prob.pack(Y));
}

/** Residual per bead: |y'' - grad U| inside; 0 at a fixed end; at a free end |dJ/ds| (the natural condition). */
function residuals(prob, Y) {
  const r = new Array(Y.length).fill(0);
  prob.elResidual(Y).forEach((v, i) => { r[i + 1] = Math.hypot(v[0], v[1]); });
  if (prob.shelf) { const G = prob.gradNodes(Y); r[prob.M] = Math.abs(G[prob.M][0] * TAU[0] + G[prob.M][1] * TAU[1]); }
  return r;
}

export async function init(panel) {
  const v = await makeArmViewer(panel, { camera: CAMERA, q: Q_SEED });
  const sc = v.scene;

  // ---------------------------------------------------------------- 3D objects (own three.js meshes)
  const MAXB = 81;
  const beads = new THREE.InstancedMesh(new THREE.SphereGeometry(BEAD_R, 16, 12),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.35, metalness: 0, emissive: 0x2a2a2a }), MAXB);
  const links = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 8, 1),
    new THREE.MeshStandardMaterial({ color: 0xb8bdc7, roughness: 0.5, transparent: true, opacity: 0.85 }), MAXB);
  for (const m of [beads, links]) { m.frustumCulled = false; sc.add(m); }
  const tmpC = new THREE.Color(), tmpM = new THREE.Matrix4(), tmpQ = new THREE.Quaternion();
  const vA = new THREE.Vector3(), vB = new THREE.Vector3(), vD = new THREE.Vector3(), vS = new THREE.Vector3(), yUp = new THREE.Vector3(0, 1, 0);
  for (let i = 0; i < MAXB; i++) beads.setColorAt(i, tmpC.setRGB(1, 1, 1));
  const endA = marker(sc, COL.white, 1.35 * BEAD_R);
  const endB = marker(sc, COL.green, 1.35 * BEAD_R);
  const glowB = [marker(sc, COL.green, 0.030, 0.26), marker(sc, COL.green, 0.050, 0.10)];
  glowB.forEach((m) => m.position.set(...uvToWorld(B2)));
  endA.position.set(...uvToWorld(A2));
  const refLine = new Line3D(sc, COL.white, 200, { opacity: 0.35 });
  const corner = new Line3D(sc, COL.yellow, 3);

  // pillar (red column, bright cap) and its penalty zone (soft red discs under the path plane)
  const pillar = new THREE.Group();
  const planeZ = uvToWorld([0, 0])[2];
  const pilH = planeZ + PILLAR_ABOVE;
  const col = new THREE.Mesh(new THREE.CylinderGeometry(PIL_R, PIL_R, pilH, 32),
    new THREE.MeshStandardMaterial({ color: new THREE.Color(COL.red).multiplyScalar(0.55), roughness: 0.75 }));
  col.rotation.x = Math.PI / 2; col.position.z = pilH / 2; col.castShadow = true;
  const cap = new THREE.Mesh(new THREE.CylinderGeometry(PIL_R + 0.004, PIL_R + 0.004, 0.02, 32),
    new THREE.MeshStandardMaterial({ color: COL.red, emissive: COL.red, emissiveIntensity: 0.6, roughness: 0.5 }));
  cap.rotation.x = Math.PI / 2; cap.position.z = pilH;
  pillar.add(col, cap);
  ZONES.forEach(([k, al], i) => {
    const d = new THREE.Mesh(new THREE.CircleGeometry(k * SIGMA, 48),
      new THREE.MeshBasicMaterial({ color: COL.red, transparent: true, opacity: al, depthWrite: false, side: THREE.DoubleSide }));
    d.position.z = planeZ - BEAD_R - 0.004 - 0.0012 * i;
    d.renderOrder = 1;
    pillar.add(d);
  });
  sc.add(pillar);

  // shelf: a board along tau (direction in the room: x = lateral, y = along), posts to the floor, a strip on top
  const shelf = new THREE.Group();
  const sMid = 0.5 * (SHELF_S[0] + SHELF_S[1]), sLen = SHELF_S[1] - SHELF_S[0];
  const top = planeZ - BEAD_R - 0.002;
  const board = new THREE.Mesh(new THREE.BoxGeometry(sLen, SHELF_DEPTH, 0.024),
    new THREE.MeshStandardMaterial({ color: new THREE.Color(0.33, 0.30, 0.27), roughness: 0.8 }));
  board.position.z = -0.012; board.castShadow = true; board.receiveShadow = true;
  const postMat = new THREE.MeshStandardMaterial({ color: 0x2a2b30, roughness: 0.7 });
  for (const sx of [-1, 1]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.028, 0.028, top - 0.024), postMat);
    post.position.set(sx * (sLen / 2 - 0.03), SHELF_DEPTH / 2 - 0.03, -0.024 - (top - 0.024) / 2);
    post.castShadow = true;
    shelf.add(post);
  }
  const strip = new THREE.Mesh(new THREE.BoxGeometry(sLen, 0.015, 0.0044),
    new THREE.MeshStandardMaterial({ color: 0x6b665e, emissive: 0x000000, roughness: 0.6 }));
  strip.position.z = 0.0022;
  shelf.add(board, strip);
  const cW = uvToWorld([B2[0] + sMid * TAU[0], B2[1] + sMid * TAU[1]]);
  shelf.position.set(cW[0], cW[1], top);
  shelf.rotation.z = Math.atan2(TAU[0], TAU[1]);            // room direction of tau: (x, y) = (tau_lat, tau_along)
  sc.add(shelf);

  // A / B labels over the 3D view
  const labA = panel.querySelector('.d31-lab[data-lab="A"]'), labB = panel.querySelector('.d31-lab[data-lab="B"]');
  const canvas3 = panel.querySelector('canvas.view3d');
  const proj = new THREE.Vector3();
  function place(el, p, dx, dy) {
    proj.set(...p).project(v.camera);
    if (proj.z > 1 || Math.abs(proj.x) > 1.05 || Math.abs(proj.y) > 1.05) { el.style.display = 'none'; return; }
    el.style.display = '';
    el.style.left = `${(proj.x + 1) / 2 * canvas3.clientWidth + dx}px`;
    el.style.top = `${(1 - proj.y) / 2 * canvas3.clientHeight + dy}px`;
  }

  // ---------------------------------------------------------------- state
  let M = 40, lat = 0.03, free = false;
  let prob, Y, z, Yref, refIters = 0, Yshow, Yfrom = null, morph = 1, linkWhite = false;
  const run = { gd: { k: 0, t: 0, on: false, done: false, why: '' }, nw: { n: 0, conv: false } };
  let series = { gd: null, nw: null };                     // {pts: [[k, dist, bend, wrinkle]], stale}
  let nextRec = 0;
  let dirty = true;
  let origin = 'a path drawn by hand';

  function solveRef() {
    const r = prob.newton(Y, { tol: 1e-11, safeguard: true, maxIter: 100 });
    Yref = r.Y; refIters = r.iters;
  }
  function setPath(Ynew, animate) {
    Yfrom = animate ? (Yshow || Y) : null; morph = animate ? 0 : 1;
    Y = Ynew; z = prob.pack(Y);
    if (!animate) Yshow = Y;
    dirty = true;
  }
  function clearRuns(keepStale) {
    run.gd = { k: 0, t: 0, on: false, done: false, why: '' };
    run.nw = { n: 0, conv: false };
    delete panel.dataset.gd;
    if (keepStale) { for (const kk of ['gd', 'nw']) if (series[kk]) series[kk].stale = true; } else series = { gd: null, nw: null };
    gdBtn.classList.remove('on');
  }
  function newProblem(resetPath) {
    prob = makeProblem(M, lat, free);
    const fresh = resetPath || !Y || Y.length !== M + 1; const Yp = fresh ? handDrawn(prob) : prob.nodes(prob.pack(Y)); origin = fresh ? 'a path drawn by hand' : 'start: the previous path';
    setPath(Yp, false);
    solveRef();
    clearRuns(false);
  }

  // ---------------------------------------------------------------- controls
  const gdBtn = panel.querySelector('[data-ctl="gd"]');
  onClick(panel, 'gd', () => {
    if (run.gd.on) { run.gd.on = false; gdBtn.classList.remove('on'); return; }
    if (!series.gd || series.gd.stale || run.gd.done) {      // a new series from the current path (else: resume)
      series.gd = { pts: [], stale: false };
      run.gd = { k: 0, t: 0, on: false, done: false, why: '' };
      nextRec = 0; record();
      solveRef();
    }
    if (series.nw && !series.nw.stale) series.nw.stale = true;
    run.nw = { n: 0, conv: false };
    delete panel.dataset.gd;
    morph = 1; Yshow = Y; z = prob.pack(Y);
    run.gd.on = true; gdBtn.classList.add('on'); linkWhite = false;
  });
  onClick(panel, 'newton', () => {
    if (run.gd.on) { run.gd.on = false; gdBtn.classList.remove('on'); }
    if (!series.nw || series.nw.stale) {
      series.nw = { pts: [[0, beadDist(Y, Yref)]], stale: false };
      run.nw = { n: 0, conv: false };
      if (series.gd) series.gd.stale = true;
      run.gd = { k: 0, t: 0, on: false, done: false, why: '' };
      delete panel.dataset.gd;
    }
    if (run.nw.conv) return;
    const r = prob.newton(Y, { maxIter: 1, tol: 0, safeguard: true });
    run.nw.n += 1;
    run.nw.conv = r.gradNorm <= 1e-10;
    setPath(r.Y, true);
    series.nw.pts.push([run.nw.n, beadDist(Y, Yref)]);
    linkWhite = true;
  });
  onClick(panel, 'reset', () => {
    origin = 'a path drawn by hand';
    setPath(handDrawn(prob), true);
    solveRef();
    clearRuns(true);
    linkWhite = false;
  });
  const follow = toggle(panel, 'follow', () => {});
  onInput(panel, 'pillar', (e) => {
    lat = Number(e.target.value);
    const wasOn = run.gd.on;
    prob = makeProblem(M, lat, free);
    setPath(prob.nodes(prob.pack(Y)), false);
    origin = 'start: the previous path';
    solveRef();
    clearRuns(false);
    if (wasOn) { series.gd = { pts: [], stale: false }; nextRec = 0; record(); run.gd.on = true; gdBtn.classList.add('on'); }
  });
  seg(panel, 'end', (val) => { free = val === 'free'; newProblem(false); linkWhite = false; });
  seg(panel, 'beads', (val) => { M = Number(val); newProblem(true); linkWhite = false; });

  // ---------------------------------------------------------------- numbers of the current path
  function record() {
    const d = beadDist(Y, Yref);
    let bend = NaN, wr = NaN;
    if (!free) {
      const E = [];
      for (let i = 1; i < M; i++) E.push([Y[i][0] - Yref[i][0], Y[i][1] - Yref[i][1]]);
      const amp = S.modeAmplitudes(E), f = Math.sqrt(2 / M);
      bend = f * amp[0]; wr = f * amp[WRINKLE_J - 1];
    }
    series.gd.pts.push([run.gd.k, d, bend, wr]);
    nextRec = Math.max(run.gd.k + 1, Math.floor(run.gd.k * 1.06));
    return d;
  }

  function gdAdvance(dt) {
    const g = run.gd;
    g.t += dt;
    const want = Math.floor(GD_V0 / GD_RATE * (Math.exp(GD_RATE * g.t) - 1));
    const alpha = 1 / prob.lipschitz(), t0 = performance.now();
    let n = 0;
    while (g.k < want && n < 4000 && performance.now() - t0 < 14) {
      z = prob.gdStep(z, alpha, Y);
      Y = prob.nodes(z);
      g.k += 1; n += 1;
      const d = beadDist(Y, Yref);
      if (g.k >= nextRec || d <= TOL) record();
      if (d <= TOL) { g.on = false; g.done = true; g.why = '1mm'; break; }
      if (g.k >= 200000) { g.on = false; g.done = true; g.why = 'cap'; break; }
      if (n % 50 === 0 && S.normInf(prob.grad(Y)) <= 1e-10) { g.on = false; g.done = true; g.why = 'other'; record(); break; }
    }
    if (g.done) { gdBtn.classList.remove('on'); panel.dataset.gd = 'done'; }
    Yshow = Y; dirty = true;
  }

  // ---------------------------------------------------------------- hand and arm
  let q = Q_SEED.slice(), handU = 0, clock = 0, ikErr = 0, tipErr = 0, ikIters = 0;
  {
    const p = siteTarget(A2);
    q[0] = Math.PI + Math.atan2(p[1], p[0]);
    const r = S.ikLevenbergMarquardt(q, p, R_TOOL, { rotWeight: 0.3, tol: 1e-10, maxIter: 200 });
    q = r.q[r.q.length - 1];
  }
  function pointAt(Yp, u) {                                  // path point at the share u of the path's time
    const x = u * M, i = Math.min(M - 1, Math.floor(x)), f = x - i;
    return [lerp(Yp[i][0], Yp[i + 1][0], f), lerp(Yp[i][1], Yp[i + 1][1], f)];
  }
  function moveHand(dt, Yp) {
    if (follow()) {
      clock += dt;
      const Tc = 2 * (PASS + REST), tc = clock % Tc;
      handU = tc < REST ? 0 : tc < REST + PASS ? travel((tc - REST) / PASS) : tc < 2 * REST + PASS ? 1 : 1 - travel((tc - 2 * REST - PASS) / PASS);
    } else {
      handU = Math.max(0, handU - dt / PASS); clock = 0;
    }
    const y = pointAt(Yp, handU), p = siteTarget(y);
    const r = S.ikLevenbergMarquardt(q, p, R_TOOL, { rotWeight: 0.3, tol: 1e-9, maxIter: 40 });
    const qn = r.q[r.q.length - 1];
    if (qn.every(Number.isFinite)) q = qn;
    ikErr = r.err[r.err.length - 1]; ikIters = r.nIter;
    const f = S.fk(q), tip = [0, 1, 2].map((k) => f.p[k] + f.R[k][2] * TIP), want = [D0 + y[1], y[0], Z_TIP];
    tipErr = Math.hypot(tip[0] - want[0], tip[1] - want[1], tip[2] - want[2]);
    v.setArm(q); v.forward();
    return y;
  }

  // ---------------------------------------------------------------- 3D update
  function draw3d(Yp, r) {
    const n = M + 1;
    for (let i = 0; i < MAXB; i++) {
      if (i >= 1 && i < n - 1) {
        vA.set(...uvToWorld(Yp[i]));
        tmpM.compose(vA, tmpQ.identity(), vS.set(1, 1, 1));
        beads.setMatrixAt(i, tmpM);
        const c = resColor(r[i]);
        beads.setColorAt(i, tmpC.setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace));
      } else {
        tmpM.makeScale(0, 0, 0); beads.setMatrixAt(i, tmpM);
      }
      if (i < n - 1) {
        vA.set(...uvToWorld(Yp[i])); vB.set(...uvToWorld(Yp[i + 1]));
        vD.subVectors(vB, vA); const len = vD.length();
        if (len > 1e-9) tmpQ.setFromUnitVectors(yUp, vD.multiplyScalar(1 / len)); else tmpQ.identity();
        vA.add(vB).multiplyScalar(0.5);
        tmpM.compose(vA, tmpQ, vS.set(LINK_R, len, LINK_R));
        links.setMatrixAt(i, tmpM);
      } else { tmpM.makeScale(0, 0, 0); links.setMatrixAt(i, tmpM); }
    }
    beads.instanceMatrix.needsUpdate = true; beads.instanceColor.needsUpdate = true; links.instanceMatrix.needsUpdate = true;
    links.material.color.set(linkWhite ? 0xeeeeee : 0xb8bdc7);
    endB.position.set(...uvToWorld(Yp[M]));
    if (free) { const c = resColor(r[M]); endB.material.color.setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace); endB.material.emissive.copy(endB.material.color); }
    else { endB.material.color.set(COL.green); endB.material.emissive.set(COL.green); }
    glowB.forEach((m) => { m.visible = !free; });
    refLine.set(Yref.map((y) => { const w = uvToWorld(y); return [w[0], w[1], w[2] - 0.004]; }));
    refLine.visible = beadDist(Yp, Yref) > 2e-3;
    strip.material.color.set(free ? COL.green : 0x6b665e);
    strip.material.emissive.set(free ? COL.green : 0x000000);
    strip.material.emissiveIntensity = free ? 0.6 : 0;
    pillar.position.set(...uvToWorld([0, lat]).slice(0, 2), 0);
    // right-angle mark at a free end
    if (free) {
      const E = uvToWorld(Yp[M]), P = uvToWorld(Yp[M - 1]);
      const da = [P[0] - E[0], P[1] - E[1]], na = Math.hypot(...da) || 1;
      const db = [-TAU[1], -TAU[0]], k = 0.045;           // room direction of -tau
      const a = [da[0] / na, da[1] / na];
      corner.set([[E[0] + k * a[0], E[1] + k * a[1], E[2] + 0.003], [E[0] + k * (a[0] + db[0]), E[1] + k * (a[1] + db[1]), E[2] + 0.003],
        [E[0] + k * db[0], E[1] + k * db[1], E[2] + 0.003]]);
    }
    corner.visible = free;
    place(labA, uvToWorld(A2), -22, -2);
    place(labB, uvToWorld(free ? Yp[M] : B2), 10, -8);
    labB.style.color = free ? CSS.yellow : CSS.green;
    labB.textContent = free ? 'end' : 'B';
  }

  // ---------------------------------------------------------------- 2D: plan view and chart
  const P = plot2d(panel.querySelector('canvas.plot'));
  const AR = [-0.52, 0.56], LR = [-0.48, 0.30];
  let L = {};
  function layout() {
    P.fit();
    const w = P.w, h = P.h;
    const s = Math.min((w - 20) / (AR[1] - AR[0]), (h * 0.55) / (LR[1] - LR[0]));
    const pw = s * (AR[1] - AR[0]), ph = s * (LR[1] - LR[0]);
    L = { s, x0: (w - pw) / 2, y0: 6, pw, ph };
    L.bar = L.y0 + ph + 8;
    L.ch = { x: 52, y: L.bar + 56, w: w - 52 - 24, h: Math.max(80, h - (L.bar + 56) - 30) };
  }
  const X = (y) => L.x0 + (y[0] - AR[0]) * L.s;
  const Yp_ = (y) => L.y0 + (y[1] - LR[0]) * L.s;

  function drawPlan(Yp, r, hand) {
    const g = P.g;
    g.save();
    g.beginPath(); g.rect(L.x0, L.y0, L.pw, L.ph); g.clip();
    // penalty glow
    const cx = X([0, lat]), cy = Yp_([0, lat]);
    const grd = g.createRadialGradient(cx, cy, 0, cx, cy, 3.25 * SIGMA * L.s);
    grd.addColorStop(0, 'rgba(252,98,85,0.42)'); grd.addColorStop(0.35, 'rgba(252,98,85,0.18)'); grd.addColorStop(1, 'rgba(252,98,85,0)');
    g.fillStyle = grd; g.beginPath(); g.arc(cx, cy, 3.25 * SIGMA * L.s, 0, 2 * Math.PI); g.fill();
    g.strokeStyle = CSS.red; g.lineWidth = 1.6; g.fillStyle = 'rgba(139,54,47,0.9)';
    g.beginPath(); g.arc(cx, cy, PIL_R * L.s, 0, 2 * Math.PI); g.fill(); g.stroke();
    g.fillStyle = CSS.red; g.font = '12px Inter, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'top';
    g.fillText('penalty U(y)', cx, cy + PIL_R * L.s + 0.06 * L.s);
    // shelf
    const nrm = [-TAU[1], TAU[0]], d = SHELF_DEPTH / 2;
    const cs = [[SHELF_S[0], -d], [SHELF_S[1], -d], [SHELF_S[1], d], [SHELF_S[0], d]].map(([s, w]) => [B2[0] + s * TAU[0] + w * nrm[0], B2[1] + s * TAU[1] + w * nrm[1]]);
    g.fillStyle = 'rgba(84,77,69,0.75)'; g.strokeStyle = 'rgba(140,130,118,0.8)'; g.lineWidth = 1;
    g.beginPath(); cs.forEach((p, i) => (i ? g.lineTo(X(p), Yp_(p)) : g.moveTo(X(p), Yp_(p)))); g.closePath(); g.fill(); g.stroke();
    const e0 = [B2[0] + SHELF_S[0] * TAU[0], B2[1] + SHELF_S[0] * TAU[1]], e1 = [B2[0] + SHELF_S[1] * TAU[0], B2[1] + SHELF_S[1] * TAU[1]];
    g.strokeStyle = free ? CSS.green : 'rgba(150,145,138,0.7)'; g.lineWidth = free ? 2.2 : 1.2;
    g.beginPath(); g.moveTo(X(e0), Yp_(e0)); g.lineTo(X(e1), Yp_(e1)); g.stroke();
    g.fillStyle = CSS.mute; g.textAlign = 'left'; g.textBaseline = 'bottom';
    g.fillText('shelf', X(e1) + 8, Yp_(e1) + 4);
    // optimal path (dashed) while the current one is not on it
    if (beadDist(Yp, Yref) > 2e-3) {
      g.setLineDash([4, 4]); g.strokeStyle = 'rgba(238,238,238,0.45)'; g.lineWidth = 1.2;
      g.beginPath(); Yref.forEach((y, i) => (i ? g.lineTo(X(y), Yp_(y)) : g.moveTo(X(y), Yp_(y)))); g.stroke(); g.setLineDash([]);
    }
    // current path
    g.strokeStyle = linkWhite ? 'rgba(238,238,238,0.9)' : C_GD; g.lineWidth = 1.3;
    g.beginPath(); Yp.forEach((y, i) => (i ? g.lineTo(X(y), Yp_(y)) : g.moveTo(X(y), Yp_(y)))); g.stroke();
    const br = Math.max(2.2, Math.min(3.6, 0.009 * L.s));
    for (let i = 1; i < M; i++) { g.fillStyle = rgb(resColor(r[i])); g.beginPath(); g.arc(X(Yp[i]), Yp_(Yp[i]), br, 0, 2 * Math.PI); g.fill(); }
    g.fillStyle = CSS.white; g.beginPath(); g.arc(X(A2), Yp_(A2), br * 1.5, 0, 2 * Math.PI); g.fill();
    g.fillStyle = free ? rgb(resColor(r[M])) : CSS.green; g.beginPath(); g.arc(X(Yp[M]), Yp_(Yp[M]), br * 1.5, 0, 2 * Math.PI); g.fill();
    if (free) { g.strokeStyle = CSS.green; g.lineWidth = 1.2; g.beginPath(); g.arc(X(B2), Yp_(B2), br * 1.3, 0, 2 * Math.PI); g.stroke(); }
    // hand
    g.strokeStyle = CSS.white; g.lineWidth = 1.8; g.beginPath(); g.arc(X(hand), Yp_(hand), br * 2.4, 0, 2 * Math.PI); g.stroke();
    // arrival angle at a free end
    if (free) {
      const E = Yp[M], Pm = Yp[M - 1], ang = prob.arrivalAngleDeg(Yp, 1);
      const ok = Math.abs(ang - 90) < 0.05, c = ok ? CSS.yellow : CSS.red;
      const a0 = Math.atan2(Yp_(Pm) - Yp_(E), X(Pm) - X(E)), a1 = Math.atan2(-TAU[1], -TAU[0]);
      g.strokeStyle = c; g.lineWidth = 1.8; g.beginPath();
      let da = a1 - a0; while (da > Math.PI) da -= 2 * Math.PI; while (da < -Math.PI) da += 2 * Math.PI;
      g.arc(X(E), Yp_(E), 16, a0, a0 + da, da < 0); g.stroke();
      g.fillStyle = c; g.font = '600 13px Inter, sans-serif'; g.textAlign = 'right'; g.textBaseline = 'middle';
      g.fillText(`${fmt(ang, 1)}°`, X(E) - 20, Yp_(E) + 22);
    }
    // labels
    g.font = '600 14px Inter, sans-serif'; g.fillStyle = CSS.white; g.textAlign = 'right'; g.textBaseline = 'middle';
    g.fillText('A', X(A2) - 9, Yp_(A2));
    g.fillStyle = CSS.green; g.textAlign = 'left'; g.fillText('B', X(B2) + 9, Yp_(B2) + (free ? 12 : 0));
    g.restore();
    g.font = '11px Inter, sans-serif'; g.fillStyle = CSS.mute; g.textAlign = 'right'; g.textBaseline = 'top';
    g.fillText('seen from above', L.x0 + L.pw - 2, L.y0 + 2);
    g.textAlign = 'left'; g.fillText('↑ robot', L.x0 + 4, L.y0 + 2);
    // residual colour bar
    const bw = Math.min(140, Math.round(0.38 * L.pw)), bx = L.x0 + L.pw - bw - 4, by = L.bar + 6;
    for (let i = 0; i < bw; i++) { const rr = RES_LO * (RES_HI / RES_LO) ** (i / (bw - 1)); g.fillStyle = rgb(resColor(rr)); g.fillRect(bx + i, by, 1, 7); }
    g.fillStyle = CSS.mute; g.textAlign = 'center'; g.textBaseline = 'top'; g.font = '10px Inter, sans-serif';
    for (const t of [0.1, 1, 10]) g.fillText(String(t), bx + Math.log10(t / RES_LO) / Math.log10(RES_HI / RES_LO) * (bw - 1), by + 10);
    g.textAlign = 'left'; g.textBaseline = 'middle'; g.font = '11px Inter, sans-serif';
    g.fillText('Euler–Lagrange residual', L.x0 + 4, by + 4);
  }

  function drawChart() {
    const g = P.g, c = L.ch;
    let kmax = 1e4;
    if (series.gd) for (const p of series.gd.pts) kmax = Math.max(kmax, p[0] * 1.3);
    const lx = Math.log10(kmax + 1), ylo = -4, yhi = 0;
    const cxp = (k) => c.x + Math.log10(k + 1) / lx * c.w;
    const cyp = (d) => c.y + (yhi - Math.max(ylo, Math.min(yhi, Math.log10(Math.max(d, 1e-12))))) / (yhi - ylo) * c.h;
    g.save();
    g.fillStyle = CSS.ink; g.font = '12px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'bottom';
    g.fillText('largest bead distance to the optimal path', c.x - 40, c.y - 8);
    g.strokeStyle = 'rgba(97,97,107,1)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(c.x, c.y); g.lineTo(c.x, c.y + c.h); g.lineTo(c.x + c.w, c.y + c.h); g.stroke();
    g.fillStyle = CSS.mute; g.font = '10.5px Inter, sans-serif'; g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const [e, t] of [[-4, '0.1 mm'], [-3, '1 mm'], [-2, '1 cm'], [-1, '10 cm'], [0, '1 m']]) g.fillText(t, c.x - 5, cyp(10 ** e));
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (const [k, t] of [[0, '0'], [1, '1'], [10, '10'], [100, '100'], [1000, '1 000'], [10000, '10 000'], [100000, '100 000']]) if (k <= kmax) g.fillText(t, cxp(k), c.y + c.h + 4);
    g.fillText('steps (log scale)', c.x + c.w / 2, c.y + c.h + 16);
    // 1 mm line
    g.setLineDash([3, 4]); g.strokeStyle = 'rgba(255,235,64,0.55)';
    g.beginPath(); g.moveTo(c.x, cyp(TOL)); g.lineTo(c.x + c.w, cyp(TOL)); g.stroke(); g.setLineDash([]);
    const line = (pts, col2, idx, width, alpha, dash) => {
      g.save(); g.globalAlpha = alpha; g.strokeStyle = col2; g.lineWidth = width; if (dash) g.setLineDash(dash);
      g.beginPath(); let on = false;
      for (const p of pts) { if (!Number.isFinite(p[idx])) continue; const x = cxp(p[0]), y = cyp(p[idx]); if (on) g.lineTo(x, y); else { g.moveTo(x, y); on = true; } }
      g.stroke(); g.restore();
    };
    if (series.gd && series.gd.pts.length) {
      const a = series.gd.stale ? 0.35 : 1, pts = series.gd.pts, last = pts[pts.length - 1];
      if (!free) {
        line(pts, CSS.mute, 2, 1, 0.8 * a, [5, 3]);
        line(pts, CSS.mute, 3, 1, 0.8 * a, [1.5, 2.5]);
        const p0 = pts[0];
        g.globalAlpha = a; g.fillStyle = CSS.mute; g.font = '10px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'bottom';
        if (Number.isFinite(p0[2])) g.fillText('large bend', cxp(0) + 4, cyp(p0[2]) - 2);
        if (Number.isFinite(p0[3])) { g.textBaseline = 'top'; g.fillText(`wrinkle (${WRINKLE_J} half-waves)`, cxp(0) + 4, cyp(p0[3]) + 3); }
        g.globalAlpha = 1;
      }
      line(pts, C_GD, 1, 2.2, a);
      g.globalAlpha = a; g.fillStyle = C_GD; g.beginPath(); g.arc(cxp(last[0]), cyp(last[1]), 3, 0, 2 * Math.PI); g.fill();
      g.font = '600 11px Inter, sans-serif'; g.textAlign = 'right'; g.textBaseline = 'bottom';
      g.fillText(`gradient descent ${last[0].toLocaleString('en-US').replace(/,/g, ' ')}`, Math.min(c.x + c.w, cxp(last[0]) + 40), cyp(last[1]) - 6);
      g.globalAlpha = 1;
    }
    if (series.nw && series.nw.pts.length) {
      const a = series.nw.stale ? 0.35 : 1, pts = series.nw.pts, last = pts[pts.length - 1];
      line(pts, CSS.white, 1, 1.6, a);
      g.globalAlpha = a; g.fillStyle = CSS.white;
      for (const p of pts) { const x = cxp(p[0]), y = cyp(p[1]); g.beginPath(); g.moveTo(x, y - 4); g.lineTo(x + 4, y); g.lineTo(x, y + 4); g.lineTo(x - 4, y); g.closePath(); g.fill(); }
      g.font = '600 11px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'top';
      g.fillText(`Newton ${last[0]}`, cxp(last[0]) + 7, Math.min(c.y + c.h - 14, cyp(last[1]) + 4));
      g.globalAlpha = 1;
    }
    g.restore();
  }

  // ---------------------------------------------------------------- readout
  let lastInfo = '';
  function info() {
    const parts = [];
    const d = beadDist(Y, Yref);
    if (run.gd.k > 0 || run.gd.on) {
      let s = `<span class="c-grey">gradient descent</span> <b>${run.gd.k.toLocaleString('en-US').replace(/,/g, ' ')}</b> steps`;
      if (run.gd.done && run.gd.why === '1mm') s += ' to 1 mm';
      if (run.gd.done && run.gd.why === 'other') s += ' (settled on another path)';
      parts.push(s);
    }
    if (run.nw.n > 0) parts.push(`<span class="c-white">Newton</span> <b>${run.nw.n}</b> iteration${run.nw.n > 1 ? 's' : ''}${run.nw.conv ? ' (|∇J| ≤ 1e-10)' : ''}`);
    if (!parts.length) parts.push(origin);
    parts.push(`largest bead distance to the optimal path <b>${d >= 0.01 ? fmt(100 * d, 1) + ' cm' : fmt(1000 * d, 2) + ' mm'}</b>`);
    parts.push(`J = <b>${fmt(prob.cost(Y), 4)}</b> (optimum ${fmt(prob.cost(Yref), 4)})`);
    if (free) {
      const ang = prob.arrivalAngleDeg(Y, 1);
      parts.push(`arrival at the shelf <b class="${Math.abs(ang - 90) < 0.05 ? 'c-yellow' : 'c-red'}">${fmt(ang, 1)}°</b>`);
    }
    const html = parts.join(' · ');
    if (html !== lastInfo) { readout(panel, 'info', html); lastInfo = html; }
  }

  // ---------------------------------------------------------------- start and loop
  newProblem(true);
  layout();
  new ResizeObserver(() => { layout(); dirty = true; }).observe(P.g.canvas);

  v.onFrame((dt) => {
    if (run.gd.on) gdAdvance(dt);
    if (morph < 1) {
      morph = Math.min(1, morph + dt / 0.5);
      const u = ease(morph);
      Yshow = Yfrom ? Yfrom.map((y, i) => [lerp(y[0], Y[i][0], u), lerp(y[1], Y[i][1], u)]) : Y;
      if (morph >= 1) Yshow = Y;
      dirty = true;
    }
    const r = residuals(prob, Yshow);
    const hand = moveHand(dt, Yshow);
    draw3d(Yshow, r);
    layout();
    P.clear();
    drawPlan(Yshow, r, hand);
    drawChart();
    info();
    dirty = false;
  });

  panel._state = () => {
    const f = S.fk(q);
    return {
      M, end: free ? 'free' : 'fixed', pillar_lateral_m: lat,
      gd_steps: run.gd.k, gd_running: run.gd.on, gd_done: run.gd.done, gd_stop: run.gd.why,
      newton_iterations: run.nw.n, newton_converged: run.nw.conv,
      bead_distance_to_optimum_m: beadDist(Y, Yref),
      cost: prob.cost(Y), cost_optimum: prob.cost(Yref), ref_newton_iterations: refIters,
      start_cost: prob.cost(handDrawn(prob)),
      el_residual_max: Math.max(...residuals(prob, Y).slice(1, M)),
      grad_inf: S.normInf(prob.grad(Y)),
      optimum_min_distance_to_pillar_axis_m: Math.min(...Yref.map((y) => Math.hypot(y[0], y[1] - lat))),
      optimum_lateral_at_middle_m: Yref[M / 2][1],
      end_s_m: (Y[M][0] - B2[0]) * TAU[0] + (Y[M][1] - B2[1]) * TAU[1],
      arrival_angle_deg_two_point: prob.shelf ? prob.arrivalAngleDeg(Y, 1) : null,
      arrival_angle_deg_three_point: prob.shelf ? prob.arrivalAngleDeg(Y, 2) : null,
      chart_points: { gd: series.gd ? series.gd.pts.length : 0, newton: series.nw ? series.nw.pts.length : 0 },
      follow: follow(), hand_u: handU,
      ik_error: ikErr, ik_iterations: ikIters, tip_to_path_m: tipErr,
      tool_world_vs_fk_m: (() => { const t = v.tool(), w = world(f.p); return Math.hypot(t[0] - w[0], t[1] - w[1], t[2] - w[2]); })(),
      q: q.map((x) => +x.toFixed(4)),
    };
  };
}
