// 2.1 Gradient descent with noise (clips s21a_noise and s21b_annealing).
// Up to 200 translucent arms run noisy gradient descent on the two-joint reach cost (sim6.noisyGD); the solid arm
// runs the same iteration without noise from the same start. The 2D panel shows the essence in one of three views:
//   landscape  - the runs as a point cloud on the joint-space map (same colours as the clip);
//   zoom       - the cloud near the minimum against the lecture's variance formula v_i = a s^2 / (l_i (2 - a l_i));
//   share vs T - (temperature mode) the share in the deeper well against the prediction from exp(-f/T), and the
//                path of a cooling run.
// Numbers that sim6.js does not port (Boltzmann prediction, 2000-run references) come from ../../data/d21_reference.json.
import { S, THREE, makeArmViewer, planeCamera, Landscape, marker, planeToWorld, GhostArms, Ellipse3D, COL, CSS,
  LAND_LINE, cmap, val, onInput, onClick, readout, ctl, seg, setVal, fmt } from '../common.js';

const TAU = 2 * Math.PI;
const BUDGET_MS = 14;              // simulation time per frame at most
const CHUNK = 25;                  // steps per call of noisyGD
const SETTLE = 4000;               // noise-free steps at the end of a cooling run (as in the Python experiments)
const AT_MIN = 0.8;                // rad: a run "is at" a solution (clip s21a's definition)
const AT_MAX = 0.3;                // rad: a run has not yet left the maximum (clip s21a's escape definition)
const NMAX = 200;
const KCOL = { 1000: CSS.red, 10000: '#f58a4b', 100000: '#f9bb45', 1000000: CSS.yellow };

export async function init(panel) {
  const REF = await (await fetch(new URL('../../data/d21_reference.json', import.meta.url))).json();
  const AN = REF.annealing;
  const two = new S.TwoJointArm(), t = S.DEFAULT_TARGET;
  const v = await makeArmViewer(panel, { camera: planeCamera(two, { at: [-0.11, 0.05], distance: 2.5, yaw: 194, elevation: -8 }) });
  marker(v.scene, COL.green, 0.024).position.set(...planeToWorld(two, t));

  // ------------------------------------------------------------------ the two costs and their special points
  const sp = two.stationaryPoints(t);
  const byName = (list, n) => list.find((s) => s.name === n);
  const plain = { cost: (q) => two.cost(q, t), grad: (q) => two.grad(q, t), hess: (q) => two.hess(q, t),
    up: byName(sp, 'elbow up').q, down: byName(sp, 'elbow down').q, max: sp.find((s) => s.type === 'max').q,
    marks: sp.map((s) => ({ q: s.q, type: s.type, color: s.name === 'elbow down' ? CSS.blue : undefined })) };
  const rc = new S.ReachCost(two, t, AN.w_m);
  const st = rc.stationary();
  const pref = { cost: rc.cost, grad: rc.grad, hess: rc.hess, up: byName(st, 'elbow up').q, down: byName(st, 'elbow down').q,
    max: st.find((s) => s.type === 'max').q,
    marks: st.map((s) => ({ q: s.q, type: s.type, color: s.name === 'elbow down' ? CSS.blue : undefined })) };
  const deepIsUp = byName(st, 'elbow up').f < byName(st, 'elbow down').f;    // expected: elbow up is deeper
  // Relaxation between the wells at a fixed T (Arrhenius form, scaled to the Kramers time at T = 0.012 from the
  // Python facts): tau(T) = tau_0.012 exp(dE (1/T - 1/0.012)), dE = lowest saddle minus the shallow minimum.
  // Checked against sim6 runs (1000 runs from elbow down): tau about 8000-9000 at T = 0.012 (model 7313) and
  // about 2100-2300 at T = 0.02 (model 2230); the share is within a few % of the prediction after 3 tau.
  const shallowF = Math.max(byName(st, 'elbow up').f, byName(st, 'elbow down').f);
  const dE = Math.min(...st.filter((s) => s.type === 'saddle').map((s) => s.f)) - shallowF;
  const tauOf = (T) => AN.kramers_relaxation_steps_at_T_0012 * Math.exp(dE * (1 / T - 1 / 0.012));
  const round2 = (x) => { const e = 10 ** Math.max(0, Math.floor(Math.log10(x)) - 1); return (Math.round(x / e) * e).toLocaleString('en'); };

  // Basins of the preference cost: where plain gradient descent from each cell ends (computed once, on first use).
  let basin = null;
  const NB = 100;
  function buildBasins() {
    const a = 0.9 / rc.L(120), lab = new Uint8Array(NB * NB);
    for (let j = 0; j < NB; j++) for (let i = 0; i < NB; i++) {
      let x = [-Math.PI + (i + 0.5) * TAU / NB, -Math.PI + (j + 0.5) * TAU / NB], l = 255;
      for (let k = 0; k < 4000 && l === 255; k++) {
        const g = pref.grad(x); x = [x[0] - a * g[0], x[1] - a * g[1]];
        if (k % 10 === 9) { if (S.torusDist(x, pref.up) < 0.02) l = 0; else if (S.torusDist(x, pref.down) < 0.02) l = 1; }
      }
      if (l === 255) l = S.torusDist(x, pref.up) <= S.torusDist(x, pref.down) ? 0 : 1;
      lab[j * NB + i] = l;
    }
    basin = lab;
  }
  const cell = (x) => Math.min(NB - 1, Math.max(0, Math.floor((S.wrap(x) + Math.PI) / TAU * NB)));
  const basinUp = (q) => basin[cell(q[1]) * NB + cell(q[0])] === 0;          // true: basin of elbow up

  // ------------------------------------------------------------------ 3D objects
  const ghosts = new GhostArms(v.scene, NMAX, { color: COL.white, opacity: 0.2, radius: 0.019 });
  const C3 = { up: new THREE.Color(COL.green), down: new THREE.Color(COL.blue), max: new THREE.Color(COL.white),
    mid: new THREE.Color(COL.grey) };
  const label = new Array(NMAX).fill('');
  function paint(i, key) {
    if (label[i] === key) return;
    label[i] = key;
    for (let k = 0; k < 6; k++) { ghosts.links.setColorAt(6 * i + k, C3[key]); ghosts.joints.setColorAt(6 * i + k, C3[key]); }
    ghosts.links.instanceColor.needsUpdate = true; ghosts.joints.instanceColor.needsUpdate = true;
  }
  for (let i = 0; i < NMAX; i++) paint(i, 'max');
  const ring = new Ellipse3D(v.scene, COL.yellow, { opacity: 0.12 });

  // ------------------------------------------------------------------ state
  const land = new Landscape(panel.querySelector('canvas.land'));
  const P = land.P;
  let mode = 'noise', view = 'map', C = plain;
  let N = val(panel, 'arms'), Q = [], qf = null, q0 = null, startName = 'stretched away';
  let rng = new S.RNG(21), step = 0, sinceChange = 0, acc = 0, lastNow = 0, paused = false, dirty = true;
  let cool = null;                                     // {K, sched, k, ks, phase, trace, rec, end}
  const traces = {};                                   // last finished cooling path per K
  let vs = null;                                       // variance accumulator {n, sx, sy, sxx, sxy, syy}
  let theory = null;                                   // formula numbers for the current alpha, sigma
  let Kpick = 1000;
  let exitStep = null;                                 // step at which the noise-free arm is 0.3 rad from its start
  let exitLaw = null;                                  // rounding-law steps at the alpha in use when the arm left
  let p0 = null;                                       // share in the deeper well when the fixed T last started (null: unknown)

  const alpha = () => val(panel, 'alpha');
  const sigma = () => val(panel, 'sigma');
  const Tset = () => val(panel, 'T');
  const curT = () => (cool && cool.phase === 'cool' ? cool.sched[Math.min(cool.k, cool.K - 1)] : cool && cool.phase !== 'cool' ? 0 : Tset());

  // Formula of the noise floor at the minima of the plain cost (both minima have the same Hessian, J^T J).
  function computeTheory() {
    const a = alpha(), s = sigma(), H = two.hess(plain.up, t), E = S.eigSym(H);
    const lam = E.values, V = E.vectors;                                    // ascending; columns
    const vv = lam.map((l) => S.stationaryVariance(a, s, l));
    const Cq = [[0, 0], [0, 0]];
    for (let k = 0; k < 2; k++) for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) Cq[i][j] += vv[k] * V[i][k] * V[j][k];
    let Cp = [[0, 0], [0, 0]];                                              // tool covariance, mean of the two solutions
    for (const qm of [plain.up, plain.down]) {
      const J = two.jac(qm), M = S.matmul(S.matmul(J, Cq), S.transpose(J));
      Cp = S.addM(Cp, S.scaleM(M, 0.5));
    }
    const Ep = S.eigSym(Cp);
    theory = { lam, V, v: vv, Cq, Cp, toolRms: Math.sqrt(S.trace(Cp) / 2),
      ringHalf: Ep.values.map((x) => 2 * Math.sqrt(Math.max(x, 0))),
      ringAxes: [0, 1].map((k) => { const a2 = [Ep.vectors[0][k], Ep.vectors[1][k]]; return [0, 1, 2].map((m) => a2[0] * two.ex[m] + a2[1] * two.ez[m]); }),
      stable: a * lam[1] < 2 };
    const w = Math.max(...vv.map((x) => 2.6 * Math.sqrt(x)));
    theory.W = Math.max(0.3, Math.ceil(w * 10) / 10);
    ring.set(planeToWorld(two, t), theory.ringAxes, theory.ringHalf);
    ring.visible = mode === 'noise' && s > 0;
    vs = { n: 0, sx: 0, sy: 0, sxx: 0, sxy: 0, syy: 0 };
    sinceChange = 0;
  }
  const burn = () => Math.ceil(5 / (alpha() * theory.lam[0]));

  // ------------------------------------------------------------------ runs
  function reset(qs, name) {
    q0 = qs.slice(); startName = name;
    Q = Array.from({ length: N }, () => q0.slice());
    qf = q0.slice();
    step = 0; sinceChange = 0; acc = 0; paused = false; cool = null; dirty = true; exitStep = null; exitLaw = null;
    p0 = name === 'elbow down' ? (deepIsUp ? 0 : 1) : null;
    if (vs) vs = { n: 0, sx: 0, sy: 0, sxx: 0, sxy: 0, syy: 0 };
    label.fill('');
  }
  function resize(n) {
    if (n < Q.length) Q.length = n;
    else for (let i = Q.length; i < n; i++) Q.push(Q[i % Math.max(1, Q.length)].slice());
    N = n; ghosts.show(N); dirty = true;
  }

  function accumulate(_, QQ) {                       // noise floor statistics (plain cost, after burn-in)
    if (sinceChange < burn()) { sinceChange++; return; }
    sinceChange++;
    for (const q of QQ) {
      const m = S.torusDist(q, plain.up) < AT_MIN ? plain.up : S.torusDist(q, plain.down) < AT_MIN ? plain.down : null;
      if (!m) continue;
      const d = S.wrap([q[0] - m[0], q[1] - m[1]]);
      vs.n++; vs.sx += d[0]; vs.sy += d[1]; vs.sxx += d[0] * d[0]; vs.sxy += d[0] * d[1]; vs.syy += d[1] * d[1];
    }
  }

  function shareDeep() {
    let k = 0;
    for (const q of Q) if (basinUp(q) === deepIsUp) k++;
    return k / Q.length;
  }

  // Advance c steps (c is already cut at phase boundaries).
  function run(c) {
    const a = alpha();
    if (mode === 'noise') {
      Q = S.noisyGD(C.grad, Q, a, { sigmaG: sigma(), steps: c, rng, callback: accumulate }).Q;
    } else if (cool && cool.phase === 'cool') {
      const k0 = cool.k, rec = cool.rec;
      Q = S.noisyGD(C.grad, Q, a, { T: (k) => cool.sched[k0 + k], steps: c, rng,
        callback: (kk, QQ) => { if ((k0 + kk) % rec === 0) { Q = QQ; cool.trace.push([cool.sched[k0 + kk - 1], shareDeep()]); } } }).Q;
      cool.k += c;
      if (cool.k >= cool.K) cool.phase = 'settle';
    } else if (cool && cool.phase === 'settle') {
      Q = S.noisyGD(C.grad, Q, a, { steps: c }).Q;
      cool.ks += c;
      if (cool.ks >= SETTLE) finishCooling();
    } else {
      Q = S.noisyGD(C.grad, Q, a, { T: Tset(), steps: c, rng }).Q;
      sinceChange += c;
    }
    qf = S.noisyGD(C.grad, [qf], a, { steps: c,                          // the solid arm: no noise
      callback: (k, QQ) => { if (exitStep === null && S.torusDist(QQ[0], q0) > AT_MAX) { exitStep = step + k; exitLaw = escapeLaw().steps; } } }).Q[0];
    step += c;
  }

  function finishCooling() {
    const deep = deepIsUp ? pref.up : pref.down;
    let k = 0;
    for (const q of Q) if (S.torusDist(q, deep) < 0.05) k++;
    cool.phase = 'done';
    cool.end = k / Q.length;
    cool.trace.push([AN.T_lo, cool.end]);
    traces[cool.K] = { trace: cool.trace, end: cool.end };
    paused = true;
    lockT(false);
  }

  function stepsWanted(dt) {
    if (cool && cool.phase === 'cool') return Math.max(125, cool.K / 8) * dt;
    if (cool && cool.phase === 'settle') return 2000 * dt;
    return 10 ** val(panel, 'speed') * dt;
  }

  function advance(dt) {
    if (paused) return;
    acc += stepsWanted(dt);
    let n = Math.floor(acc), done = 0;
    const t0 = performance.now();
    while (done < n && performance.now() - t0 < BUDGET_MS) {
      let c = Math.min(CHUNK, n - done);
      if (cool && cool.phase === 'cool') c = Math.min(c, cool.K - cool.k);
      if (cool && cool.phase === 'settle') c = Math.min(c, SETTLE - cool.ks);
      if (c <= 0) break;
      run(c); done += c;
      if (paused) break;
    }
    acc = done < n ? 0 : acc - n;                    // over budget: drop the rest (the simulation slows down)
    if (done) dirty = true;
  }

  // ------------------------------------------------------------------ counting and colours
  function classify(q) {
    if (mode === 'noise') {
      if (S.torusDist(q, plain.max) < AT_MAX) return 'max';
      return S.torusDist(q, plain.up) <= S.torusDist(q, plain.down) ? 'up' : 'down';
    }
    return basinUp(q) ? 'up' : 'down';
  }
  function counts() {
    const c = { up: 0, down: 0, max: 0, between: 0 };
    for (const q of Q) {
      if (mode === 'noise') {
        if (S.torusDist(q, plain.up) < AT_MIN) c.up++;
        else if (S.torusDist(q, plain.down) < AT_MIN) c.down++;
        else if (S.torusDist(q, plain.max) < AT_MAX) c.max++;
        else c.between++;
      } else if (basinUp(q)) c.up++; else c.down++;
    }
    return c;
  }
  function predictedShare(T) {
    const X = AN.prediction_curve.T, Y = AN.prediction_curve.share;
    if (T <= X[0]) return Y[0];
    if (T >= X[X.length - 1]) return Y[Y.length - 1];
    let k = 0;
    while (X[k + 1] < T) k++;
    const u = (Math.log(T) - Math.log(X[k])) / (Math.log(X[k + 1]) - Math.log(X[k]));
    return Y[k] + u * (Y[k + 1] - Y[k]);
  }
  function fixedT() {                               // fixed-T relaxation: null while cooling
    if (mode !== 'temp' || cool) return null;
    const T_ = Tset(), tau = tauOf(T_), peq = predictedShare(T_);
    return { T: T_, tau, peq, settled: sinceChange >= 3 * tau,
      expected: p0 === null ? null : peq + (p0 - peq) * Math.exp(-sinceChange / tau) };
  }
  function measured() {
    if (!vs || vs.n < 50) return null;
    const mx = vs.sx / vs.n, my = vs.sy / vs.n;
    const Cm = [[vs.sxx / vs.n - mx * mx, vs.sxy / vs.n - mx * my], [vs.sxy / vs.n - mx * my, vs.syy / vs.n - my * my]];
    const V = theory.V;
    const along = [0, 1].map((k) => { const e = [V[0][k], V[1][k]]; return S.dot(e, S.matvec(Cm, e)); });
    return { C: Cm, ratio: along.map((x, k) => x / theory.v[k]), n: vs.n };
  }

  // ------------------------------------------------------------------ 2D views
  function drawMap() {
    const g = land.draw();
    const groups = { up: [], down: [], max: [], mid: [] };
    Q.forEach((q, i) => groups[label[i] || 'max'].push(q));
    land.cloud(groups.max, CSS.white, { r: 2, alpha: 0.85 });
    land.cloud(groups.up, CSS.green, { r: 2, alpha: 0.75 });
    land.cloud(groups.down, CSS.blue, { r: 2, alpha: 0.75 });
    land.dot(qf, CSS.white, 5.5);
    const b = land.box;
    g.save(); g.font = '12px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'top'; g.fillStyle = CSS.ink;
    g.fillText(mode === 'noise' ? 'x ← x − α(∇f + σ_g ξ)' : 'x ← x − α∇f_w + √(2Tα) ξ', b.x + 8, b.y + 7);
    g.fillStyle = CSS.mute;
    g.fillText(`step ${step.toLocaleString('en')} · ${N} runs · ● no noise`, b.x + 8, b.y + 24);
    g.restore();
  }

  function ellipsePath(g, toPx, axes, half) {
    g.beginPath();
    for (let k = 0; k <= 96; k++) {
      const th = TAU * k / 96, a = half[0] * Math.cos(th), c = half[1] * Math.sin(th);
      const [x, y] = toPx([a * axes[0][0] + c * axes[1][0], a * axes[0][1] + c * axes[1][1]]);
      if (k) g.lineTo(x, y); else g.moveTo(x, y);
    }
  }

  function drawZoom() {
    P.fit();
    const g = P.g, L = 46, R = 10, T = 34, B = 36;
    g.fillStyle = '#000'; g.fillRect(0, 0, P.w, P.h);
    const s = Math.max(10, Math.min(P.w - L - R, P.h - T - B)), bx = L + (P.w - L - R - s) / 2, by = T + (P.h - T - B - s) / 2;
    const W = theory.W, toPx = (d) => [bx + (d[0] + W) / (2 * W) * s, by + (W - d[1]) / (2 * W) * s];
    const ax = [0, 1].map((k) => [theory.V[0][k], theory.V[1][k]]), lam = theory.lam;
    g.save(); g.beginPath(); g.rect(bx, by, s, s); g.clip();
    g.fillStyle = 'rgb(5, 15, 22)'; g.fillRect(bx, by, s, s);
    for (let k = 1; k <= 9; k++) {                                         // level lines of the quadratic model
      const c = lam[0] * (k * W / 3.2) ** 2, col = cmap(LAND_LINE, k / 14);
      g.strokeStyle = `rgba(${col[0]}, ${col[1]}, ${col[2]}, 0.55)`; g.lineWidth = 1;
      ellipsePath(g, toPx, ax, lam.map((l) => Math.sqrt(c / l))); g.stroke();
    }
    g.setLineDash([3, 4]); g.strokeStyle = 'rgba(200, 200, 210, 0.35)';
    for (const e of ax) { const p1 = toPx([-2 * W * e[0], -2 * W * e[1]]), p2 = toPx([2 * W * e[0], 2 * W * e[1]]); g.beginPath(); g.moveTo(...p1); g.lineTo(...p2); g.stroke(); }
    g.setLineDash([]);
    for (const q of Q) {                                                   // the cloud, each run at its own minimum
      const up = S.torusDist(q, plain.up) < AT_MIN, down = !up && S.torusDist(q, plain.down) < AT_MIN;
      if (!up && !down) continue;
      const m = up ? plain.up : plain.down, d = S.wrap([q[0] - m[0], q[1] - m[1]]), [x, y] = toPx(d);
      g.fillStyle = up ? CSS.green : CSS.blue; g.globalAlpha = 0.85; g.fillRect(x - 2, y - 2, 4, 4);
    }
    g.globalAlpha = 1;
    if (sigma() > 0 && theory.stable) {
      g.strokeStyle = CSS.yellow; g.lineWidth = 2.2;
      ellipsePath(g, toPx, ax, theory.v.map((x) => 2 * Math.sqrt(x))); g.stroke();
    }
    const M = measured();
    if (M) {
      const E = S.eigSym(M.C);
      g.setLineDash([5, 4]); g.strokeStyle = CSS.white; g.lineWidth = 1.5;
      ellipsePath(g, toPx, [0, 1].map((k) => [E.vectors[0][k], E.vectors[1][k]]), E.values.map((x) => 2 * Math.sqrt(Math.max(x, 0))));
      g.stroke(); g.setLineDash([]);
    }
    g.font = '11px Inter, sans-serif'; g.fillStyle = CSS.mute; g.textBaseline = 'middle';
    for (let k = 0; k < 2; k++) {
      const p = toPx([0.82 * W * ax[k][0] * Math.sign(ax[k][0] || 1), 0.82 * W * ax[k][1] * Math.sign(ax[k][0] || 1)]);
      g.textAlign = 'center'; g.fillText(`λ${k ? '₂' : '₁'} = ${fmt(lam[k], 3)}`, p[0], p[1] - 10);
    }
    g.restore();
    g.strokeStyle = 'rgb(97, 97, 107)'; g.lineWidth = 1; g.strokeRect(bx + 0.5, by + 0.5, s - 1, s - 1);
    g.font = '11px Inter, sans-serif'; g.fillStyle = CSS.mute; g.textAlign = 'center'; g.textBaseline = 'top';
    for (const u of [-W / 2, 0, W / 2]) g.fillText(fmt(u, 2), toPx([u, 0])[0], by + s + 4);
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const u of [-W / 2, 0, W / 2]) g.fillText(fmt(u, 2), bx - 5, toPx([0, u])[1]);
    g.textAlign = 'center'; g.textBaseline = 'bottom'; g.fillText('shoulder q₁ − q₁*', bx + s / 2, P.h - 1);
    g.save(); g.translate(11, by + s / 2); g.rotate(-Math.PI / 2); g.textBaseline = 'middle'; g.fillText('elbow q₂ − q₂*', 0, 0); g.restore();
    g.textAlign = 'left'; g.textBaseline = 'top'; g.font = '12px Inter, sans-serif'; g.fillStyle = CSS.ink;
    g.fillText('variance of the cloud:  vᵢ = ασ_g² / (λᵢ (2 − αλᵢ))', bx, 6);
    g.font = '11px Inter, sans-serif';
    g.fillStyle = CSS.yellow; g.fillText('— formula, 2 SD', bx, 20);
    g.fillStyle = CSS.white; g.fillText('- - measured, 2 SD', bx + 110, 20);
  }

  function drawShare() {
    P.fit();
    const g = P.g, L = 46, R = 12, T = 54, B = 40;
    g.fillStyle = '#000'; g.fillRect(0, 0, P.w, P.h);
    const w = P.w - L - R, h = Math.min(P.h - T - B, w), by = T, bx = L;
    const x0 = 0.0015, x1 = 0.034;
    const X = (T_) => bx + (T_ - x0) / (x1 - x0) * w, Y = (s_) => by + (1 - s_) * h;
    g.strokeStyle = 'rgb(60, 62, 70)'; g.lineWidth = 1;
    for (const s_ of [0, 0.5, 1]) { g.beginPath(); g.moveTo(bx, Y(s_)); g.lineTo(bx + w, Y(s_)); g.stroke(); }
    g.font = '11px Inter, sans-serif'; g.fillStyle = CSS.mute; g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const s_ of [0, 0.5, 1]) g.fillText(`${s_ * 100} %`, bx - 5, Y(s_));
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (const T_ of [0.002, 0.008, 0.016, 0.024, 0.032]) g.fillText(String(T_), X(T_), by + h + 4);
    g.fillText('cold', X(0.002), by + h + 17); g.fillText('hot', X(0.032), by + h + 17);
    g.textBaseline = 'bottom'; g.fillText('temperature T', bx + w / 2, by + h + 34);
    g.save(); g.translate(11, by + h / 2); g.rotate(-Math.PI / 2); g.textBaseline = 'middle'; g.fillStyle = CSS.green;
    g.fillText('share in the deeper well (elbow up)', 0, 0); g.restore();
    g.save(); g.beginPath(); g.rect(bx, by - 6, w, h + 12); g.clip();
    g.strokeStyle = CSS.white; g.lineWidth = 1.6; g.beginPath();                          // prediction from exp(-f/T)
    for (let k = 0; k <= 120; k++) { const T_ = x0 + (x1 - x0) * k / 120; const p = [X(T_), Y(predictedShare(T_))]; if (k) g.lineTo(...p); else g.moveTo(...p); }
    g.stroke();
    g.lineWidth = 1.4;
    for (const r of AN.fixed_T_measured) { g.beginPath(); g.arc(X(r.T), Y(r.share), 4, 0, TAU); g.fillStyle = '#000'; g.fill(); g.stroke(); }
    const drawTrace = (tr, col) => {
      g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
      tr.forEach((p, k) => { const q = [X(p[0]), Y(p[1])]; if (k) g.lineTo(...q); else g.moveTo(...q); });
      g.stroke();
    };
    for (const K of Object.keys(traces)) if (!cool || String(cool.K) !== K || cool.phase === 'done') {
      drawTrace(traces[K].trace, KCOL[K]);
      const e = traces[K].end; g.fillStyle = KCOL[K]; g.beginPath(); g.arc(X(AN.T_lo), Y(e), 4, 0, TAU); g.fill();
      g.font = '11px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'bottom'; g.fillText(`${Math.round(100 * e)} %`, X(AN.T_lo) + 3, Y(e) - 5);
    }
    if (cool && cool.phase !== 'done') drawTrace(cool.trace, KCOL[cool.K]);
    if (!cool || cool.phase === 'cool') {                                              // now: share of the live runs
      const T_ = curT(), p = shareDeep(), se = Math.sqrt(Math.max(p * (1 - p), 1e-9) / Q.length);
      const F = fixedT(), settling = F && !F.settled, col = settling ? CSS.mute : CSS.accent;
      if (F && F.expected !== null) {                                                  // faint: expected approach so far
        g.save(); g.globalAlpha = 0.45; g.strokeStyle = CSS.white; g.lineWidth = 1; g.setLineDash([2, 3]);
        g.beginPath(); g.moveTo(X(T_), Y(p0)); g.lineTo(X(T_), Y(F.peq)); g.stroke(); g.setLineDash([]);
        g.lineWidth = 1.5; g.beginPath(); g.moveTo(X(T_) - 8, Y(F.expected)); g.lineTo(X(T_) + 8, Y(F.expected)); g.stroke();
        g.restore();
      }
      g.strokeStyle = col; g.lineWidth = 1.5; g.beginPath(); g.moveTo(X(T_), Y(p - 2 * se)); g.lineTo(X(T_), Y(p + 2 * se)); g.stroke();
      g.beginPath(); g.arc(X(T_), Y(p), 5, 0, TAU);
      if (settling) { g.fillStyle = '#000'; g.fill(); g.lineWidth = 1.6; g.stroke(); } else { g.fillStyle = CSS.accent; g.fill(); }
      if (settling) {
        g.font = '11px Inter, sans-serif'; g.fillStyle = CSS.mute; g.textBaseline = 'middle';
        const right = X(T_) < bx + w - 70;
        g.textAlign = right ? 'left' : 'right'; g.fillText('settling', X(T_) + (right ? 10 : -10), Y(p));
      }
    }
    g.restore();
    g.font = '12px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'top'; g.fillStyle = CSS.ink;
    g.fillText('f_w = ½|p(q) − t|² − w z_elbow(q),  w = 0.04 m', bx, 4);
    g.font = '11px Inter, sans-serif';
    g.fillStyle = CSS.white; g.fillText('— prediction from e^(−f/T)', bx, 20);
    g.fillStyle = CSS.accent; g.fillText(`● now, ${Q.length} runs, ±2 SE`, bx + 165, 20);
    g.fillStyle = CSS.white; g.fillText('○ measured at fixed T, 2000 runs', bx, 35);
    g.fillStyle = KCOL[1000]; g.fillText('— cooling paths', bx + 190, 35);
  }

  function draw() {
    if (view === 'zoom' && mode === 'noise') drawZoom();
    else if (view === 'share' && mode === 'temp' && basin) drawShare();
    else drawMap();
  }

  // ------------------------------------------------------------------ readouts
  // Without noise, at the maximum: rounding leaves a gradient of about 1e-15 instead of 0, and the unstable
  // direction amplifies the offset by 1 + alpha |lambda_min| per step (PROGRESS.md: ln(0.3/delta)/ln(1 + alpha |lambda_min|)).
  function escapeLaw() {
    const a = alpha(), g = S.norm(C.grad(q0)), lmin = S.eig2(C.hess(q0))[0], delta = a * g;
    if (!(lmin < 0) || !(delta > 0)) return { delta, lmin, steps: Infinity };
    return { delta, lmin, steps: Math.log(AT_MAX / delta) / Math.log(1 + a * Math.abs(lmin)) };
  }
  function freeText() {
    if (startName !== 'stretched away') return `solid arm, no noise: moved <b>${movedText()}</b> rad`;
    const law = escapeLaw(), lawSteps = exitStep !== null && exitLaw !== null ? exitLaw : law.steps;
    const lawTxt = Number.isFinite(lawSteps) ? `rounding law ≈ <b>${Math.round(lawSteps)}</b>` : 'gradient exactly 0';
    return exitStep === null
      ? `solid arm, no noise: still at the maximum after <b>${step}</b> steps (moved ${movedText()} rad; ${lawTxt})`
      : `solid arm, no noise: left the maximum (${AT_MAX} rad away) at step <b>${exitStep}</b>; computed |∇f| at the start ${S.norm(C.grad(q0)).toExponential(1)}, ${lawTxt}`;
  }
  function movedText() {
    const d = S.torusDist(qf, q0);
    return d === 0 ? '0' : d < 0.01 ? d.toExponential(1) : fmt(d, 2);
  }
  function info() {
    const c = counts();
    if (mode === 'noise') {
      readout(panel, 'info', `step <b>${step}</b> · <span class="c-green">elbow up <b>${c.up}</b></span> · ` +
        `<span class="c-blue">elbow down <b>${c.down}</b></span> · <span class="c-red">at the maximum <b>${c.max}</b></span>` +
        (c.between ? ` · on the way <b>${c.between}</b>` : '') +
        ` · ${freeText()}`);
      const M = measured();
      readout(panel, 'theory', `<span class="c-yellow">tool scatter (formula)</span> <b>±${fmt(100 * theory.toolRms, 1)} cm</b> rms` +
        (M ? ` · variance ÷ formula <b>${fmt(M.ratio[0], 2)}</b> (λ₁), <b>${fmt(M.ratio[1], 2)}</b> (λ₂), ${M.n.toLocaleString('en')} samples`
          : sigma() > 0 ? ' · measuring after the runs settle' : ''));
    } else {
      const p = shareDeep(), ref = AN.cooling[String(cool ? cool.K : Kpick)];
      let s = '';
      if (cool && cool.phase === 'cool') s = `cooling, <b>${cool.K.toLocaleString('en')}</b> steps: step <b>${cool.k.toLocaleString('en')}</b>, T = <b>${fmt(curT(), 4)}</b>`;
      else if (cool && cool.phase === 'settle') s = `noise off: settling, <b>${cool.ks}</b> of ${SETTLE} steps`;
      else if (cool) s = `end of cooling in <b>${cool.K.toLocaleString('en')}</b> steps: <span class="c-green"><b>${fmt(100 * cool.end, 1)} %</b> in the deeper well</span> (${Q.length} runs)`;
      else {
        const F = fixedT();
        s = `T = <b>${fmt(F.T, 3)}</b> for <b>${sinceChange.toLocaleString('en')}</b> steps · <span class="c-green">deeper well <b>${fmt(100 * p, 0)} %</b></span> · ` +
          `prediction once settled <b>${fmt(100 * F.peq, 1)} %</b>` +
          (F.settled ? ' (settled)' : F.expected !== null ? `, expected by now ≈ <b>${fmt(100 * F.expected, 0)} %</b>` : '') +
          ` · crossing between the wells takes about <b>${round2(F.tau)}</b> steps; settled after about <b>${round2(3 * F.tau)}</b>`;
      }
      readout(panel, 'info', s);
      readout(panel, 'theory', ref ? `cooling in ${Number(cool ? cool.K : Kpick).toLocaleString('en')} steps, reference (α = 0.25, 2000 runs): <b>${fmt(100 * ref.share_deep, 1)} %</b> in the deeper well` : '');
    }
  }

  // ------------------------------------------------------------------ controls
  function lockT(on) { const el = ctl(panel, 'T'); el.disabled = on; el.closest('label').style.opacity = on ? 0.45 : 1; }
  function applyMode() {
    const noise = mode === 'noise';
    const sl = ctl(panel, 'sigma'); sl.disabled = !noise; sl.closest('label').style.opacity = noise ? 1 : 0.45;
    ctl(panel, 'T').disabled = noise; ctl(panel, 'T').closest('label').style.opacity = noise ? 0.45 : 1;
    for (const n of ['K', 'cool']) { ctl(panel, n).style.opacity = noise ? 0.45 : 1; ctl(panel, n).style.pointerEvents = noise ? 'none' : ''; }
    const vb = ctl(panel, 'view');
    vb.querySelector('[data-v=zoom]').style.display = noise ? '' : 'none';
    vb.querySelector('[data-v=share]').style.display = noise ? 'none' : '';
    ring.visible = noise && sigma() > 0;
  }
  const pickView = (name) => ctl(panel, 'view').querySelector(`[data-v=${name}]`).click();

  seg(panel, 'mode', (m) => {
    if (m === mode) return;
    mode = m;
    if (mode === 'temp') {
      if (!basin) buildBasins();
      C = pref; land.setCost(pref.cost, pref.marks);
      setVal(panel, 'alpha', 0.25); setVal(panel, 'speed', 4);
      reset(pref.down, 'elbow down'); applyMode(); pickView('share');
    } else {
      C = plain; land.setCost(plain.cost, plain.marks);
      setVal(panel, 'alpha', 1); setVal(panel, 'speed', 1);
      reset(plain.max, 'stretched away'); applyMode(); pickView('map');
    }
    lockT(mode === 'noise');
  });
  seg(panel, 'view', (w) => { view = w; dirty = true; });
  seg(panel, 'K', (k) => { Kpick = Number(k); dirty = true; });
  onClick(panel, 'away', () => reset(C.max, 'stretched away'));
  onClick(panel, 'down', () => reset(C.down, 'elbow down'));
  onClick(panel, 'cool', () => {
    if (mode !== 'temp') return;
    reset(pref.down, 'elbow down');
    cool = { K: Kpick, sched: S.schedule(AN.schedule, AN.T_hi, AN.T_lo, Kpick), k: 0, ks: 0, phase: 'cool', trace: [],
      rec: Math.max(1, Math.round(Kpick / 400)) };
    cool.trace.push([AN.T_hi, shareDeep()]);
    lockT(true);
    if (view !== 'share') pickView('share');
  });
  const speedOut = ctl(panel, 'speed').parentElement.querySelector('output');
  const showSpeed = () => { speedOut.textContent = `${Math.round(10 ** val(panel, 'speed')).toLocaleString('en')}/s`; };
  onInput(panel, 'speed', showSpeed); showSpeed();
  onInput(panel, 'alpha', () => { computeTheory(); dirty = true; });
  onInput(panel, 'sigma', () => { computeTheory(); dirty = true; });
  onInput(panel, 'T', () => {
    if (cool && cool.phase !== 'done') return;
    cool = null; paused = false; sinceChange = 0; dirty = true;
    p0 = mode === 'temp' && basin ? shareDeep() : p0;
  });
  onInput(panel, 'arms', () => resize(val(panel, 'arms')));

  // ------------------------------------------------------------------ start
  land.setCost(plain.cost, plain.marks);
  computeTheory();
  reset(plain.max, 'stretched away');
  ghosts.show(N);
  applyMode(); lockT(true);
  let lastInfo = 0;
  v.onFrame(() => {
    const now = performance.now(), dt = lastNow ? Math.min(0.25, (now - lastNow) / 1000) : 0;
    lastNow = now;
    advance(dt);
    for (let i = 0; i < N; i++) { ghosts.set(i, two.fullQ(Q[i])); paint(i, classify(Q[i])); }
    v.setArm(two.fullQ(qf)); v.forward();
    if (dirty) { draw(); dirty = false; }
    if (now - lastInfo > 200) { info(); lastInfo = now; }
  });
  window.addEventListener('resize', () => { dirty = true; });

  panel._state = () => {
    const c = counts(), M = mode === 'noise' ? measured() : null;
    return {
      mode, view, start: startName, step, runs: Q.length, alpha: alpha(), sigmaG: sigma(), T: mode === 'temp' ? curT() : null,
      counts: c,
      noNoise: { moved_rad: S.torusDist(qf, q0), steps: step, gradAtStart: S.norm(C.grad(q0)), exitStep,
        lawSteps: exitLaw ?? escapeLaw().steps },
      formula: mode === 'noise' ? { lambda: theory.lam, v: theory.v, toolRms_m: theory.toolRms, ringHalf_m: theory.ringHalf } : null,
      measured: M ? { ratio: M.ratio, samples: M.n } : null,
      share: mode === 'temp' && basin ? { deep: shareDeep(), predicted: predictedShare(curT() || Tset()), deepIsUp, sinceChange,
        tau: fixedT()?.tau ?? null, settled: fixedT()?.settled ?? null, expected: fixedT()?.expected ?? null, dE } : null,
      cooling: cool ? { K: cool.K, k: cool.k, settle: cool.ks, phase: cool.phase, end: cool.end ?? null,
        reference: AN.cooling[String(cool.K)].share_deep } : null,
      finished: Object.fromEntries(Object.entries(traces).map(([K, r]) => [K, r.end])),
    };
  };
}
