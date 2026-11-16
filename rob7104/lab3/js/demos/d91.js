// 9.1 Holding a spot: a robot runs a spring-and-damping controller towards a marked spot while random pushes act
// on it (stronger along one direction). State z = (p, v), p relative to the spot, unit mass:
//   dp = v dt,  dv = (-k p - c v) dt + G dW,   Q = G G^T.
// Many runs start from the same state. Linear SDE => each run's law stays Gaussian; the mean follows the noise-free
// damped spiral and the covariance solves dS/dt = A S + S A^T + B, settling at the Lyapunov solution
// S_pp = Q/(2ck), S_vv = Q/(2c), S_pv = 0. Defaults as the notebook video s09_hold_point.
import { rng } from '../sim3.js';
import { makeViewer, placeLamp, SwarmMesh, ledColour, THREE, COL, CSS, val, onInput, onClick, readout, ctl, plot2d } from '../common3.js';
import { hudStyle, hud, sw, dot, FlatLine, Cross, Pane, legend, label, ellipsePts, niceMax, ticks, tfmt } from '../plane_common.js';

const SPOT = [0.0, -0.1];
const P0 = [0.90, 0.40], V0 = [-0.35, 0.75];        // start offset (m) and velocity (m/s): the runs set off sideways
const TH_PUSH = 30 * Math.PI / 180;                  // pushes strongest along this direction
const Q_ALONG = 0.30, Q_ACROSS = 0.13;               // m/s^1.5 per unit noise setting
const DT = 0.002, N_ALL = 4000, N_SHOW = 200;
const PLAY = 12;                                     // s of real time for the plotted window
const REC = 100;                                    // record sample variances every REC steps (0.2 s)
const E_AL = [Math.cos(TH_PUSH), Math.sin(TH_PUSH)], E_AC = [-Math.sin(TH_PUSH), Math.cos(TH_PUSH)];
const Z = 0.036;

/** Right-hand side of the moment equations of one axis with unit noise: mean (mp, mv), covariance (spp, spv, svv). */
function rhs(k, c, y) {
  const [mp, mv, spp, spv, svv] = y;
  return [mv, -k * mp - c * mv, 2 * spv, svv - k * spp - c * spv, -2 * k * spv - 2 * c * svv + 1];
}
function rk4(k, c, y, h) {
  const a = rhs(k, c, y), b = rhs(k, c, y.map((v, i) => v + h / 2 * a[i]));
  const d = rhs(k, c, y.map((v, i) => v + h / 2 * b[i])), e = rhs(k, c, y.map((v, i) => v + h * d[i]));
  return y.map((v, i) => v + h / 6 * (a[i] + 2 * b[i] + 2 * d[i] + e[i]));
}

export async function init(panel) {
  hudStyle();
  const v = await makeViewer(panel, 'arena.mjb', { lookat: [0.12, 0.02, 0], distance: 3.2, azimuth: 90, elevation: -62 });
  for (let i = 0; i < 4; i++) placeLamp(v, i, null);
  v.forward();
  const swarm = new SwarmMesh(v.scene, N_SHOW);
  // static marks: the spot, the push line with arrow heads
  const spotRing = new FlatLine(v.scene, COL.pink, { width: 0.012, z: 0.0047, maxPts: 80 });
  spotRing.circle(SPOT[0], SPOT[1], 0.07, {}, 64);
  new Cross(v.scene, COL.pink, { size: 0.045, width: 0.014, z: Z }).at(SPOT[0], SPOT[1]);
  const pushLine = new FlatLine(v.scene, COL.grey, { width: 0.008, z: 0.0047, maxPts: 4, opacity: 0.55 });
  pushLine.set([SPOT[0] - 0.95 * E_AL[0], SPOT[0] + 0.95 * E_AL[0]], [SPOT[1] - 0.95 * E_AL[1], SPOT[1] + 0.95 * E_AL[1]]);
  for (const sg of [-1, 1]) {
    const tip = [SPOT[0] + sg * 0.95 * E_AL[0], SPOT[1] + sg * 0.95 * E_AL[1]];
    const head = new FlatLine(v.scene, COL.grey, { width: 0.008, z: 0.0047, maxPts: 4, opacity: 0.55 });
    const b1 = [tip[0] - sg * 0.08 * E_AL[0] + 0.045 * E_AC[0], tip[1] - sg * 0.08 * E_AL[1] + 0.045 * E_AC[1]];
    const b2 = [tip[0] - sg * 0.08 * E_AL[0] - 0.045 * E_AC[0], tip[1] - sg * 0.08 * E_AL[1] - 0.045 * E_AC[1]];
    head.set([b1[0], tip[0], b2[0]], [b1[1], tip[1], b2[1]]);
  }
  const statEll = new FlatLine(v.scene, COL.pink, { width: 0.011, z: Z });
  const ell1 = new FlatLine(v.scene, COL.yellow, { width: 0.011, z: Z + 0.002 });
  const ell2 = new FlatLine(v.scene, COL.yellow, { width: 0.011, z: Z + 0.002 });
  const meanPath = new FlatLine(v.scene, COL.white, { width: 0.009, z: Z + 0.001, maxPts: 2000, opacity: 0.9 });
  const meanDot = new THREE.Mesh(new THREE.SphereGeometry(0.022, 20, 14), new THREE.MeshBasicMaterial({ color: COL.yellow, toneMapped: false }));
  v.scene.add(meanDot);
  const plot = plot2d(panel.querySelector('canvas.plot'));
  const top = new Pane(plot), bot = new Pane(plot);

  const PX = new Float64Array(N_ALL), PY = new Float64Array(N_ALL), VX = new Float64Array(N_ALL), VY = new Float64Array(N_ALL);
  const xs = new Float64Array(N_SHOW), ys = new Float64Array(N_SHOW);
  const led = ledColour(0);
  let R, K, C, NS, G, QQ, tPlot, t, k, acc, curve, rec, ranges, momState, seed = 9;

  function release() {
    K = val(panel, 'k'); C = val(panel, 'c'); NS = val(panel, 'noise');
    // G = [e_al e_ac] diag(q_al, q_ac) * noise ; Q = G G^T
    G = [[E_AL[0] * Q_ALONG * NS, E_AC[0] * Q_ACROSS * NS], [E_AL[1] * Q_ALONG * NS, E_AC[1] * Q_ACROSS * NS]];
    QQ = [[G[0][0] ** 2 + G[0][1] ** 2, G[0][0] * G[1][0] + G[0][1] * G[1][1]], [0, G[1][0] ** 2 + G[1][1] ** 2]];
    QQ[1][0] = QQ[0][1];
    // slowest decay rate of the mean (real part of the slowest eigenvalue of [[0,1],[-k,-c]])
    const disc = C * C - 4 * K, slow = disc < 0 ? C / 2 : (C - Math.sqrt(disc)) / 2;
    tPlot = Math.min(40, Math.max(12, 5 * Math.ceil(4.5 / slow / 5)));
    R = rng(seed++);
    for (let i = 0; i < N_ALL; i++) { PX[i] = P0[0]; PY[i] = P0[1]; VX[i] = V0[0]; VY[i] = V0[1]; }
    t = 0; k = 0; acc = 0;
    // theory over the plotted window, every 0.05 s
    curve = { t: [], mx: [], my: [], ual: [], wal: [], spp: [], spv: [], svv: [] };
    let y = [1, 0, 0, 0, 0], yv = [0, 1, 0, 0, 0];
    const n = Math.round(tPlot / DT);
    for (let j = 0; j <= n; j++) {
      if (j % 25 === 0) pushCurve(j * DT, y, yv);
      y = rk4(K, C, y, DT); yv = rk4(K, C, yv, DT);
    }
    rec = { t: [0], al: [0], ac: [0] };
    // fixed axes for the phase plane (along the push line): mean path and 2.4 sigma of every time
    let u0 = 0, u1 = 0, w0 = 0, w1 = 0;
    curve.t.forEach((_, j) => {
      const su = 2.4 * Q_ALONG * NS * Math.sqrt(curve.spp[j]), sv = 2.4 * Q_ALONG * NS * Math.sqrt(curve.svv[j]);
      u0 = Math.min(u0, curve.ual[j] - su); u1 = Math.max(u1, curve.ual[j] + su);
      w0 = Math.min(w0, curve.wal[j] - sv); w1 = Math.max(w1, curve.wal[j] + sv);
    });
    const ua = niceMax(Math.max(-u0, u1)), wa = niceMax(Math.max(-w0, w1));
    ranges = { u0: -ua, u1: ua, w0: -wa, w1: wa,
      vmax: niceMax(1.15 * Math.max(...curve.spp) * Q_ALONG ** 2 * NS * NS || 0.01, [0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]) };
    // stationary 2 sigma ellipse (Lyapunov): S_pp = Q / (2 c k)
    const s = 1 / (2 * C * K);
    statEll.visible = NS > 0;
    if (NS > 0) statEll.ellipse(SPOT, [[QQ[0][0] * s, QQ[0][1] * s], [QQ[1][0] * s, QQ[1][1] * s]], 2, { dash: 0.035 });
    // the white path of the mean over the window
    meanPath.set(curve.mx.map((x) => x + SPOT[0]), curve.my.map((y) => y + SPOT[1]));
    meanPath.mesh.geometry.setDrawRange(0, 0);
    momState = { y: [1, 0, 0, 0, 0], yv: [0, 1, 0, 0, 0] };
  }
  function pushCurve(s, y, yv) {
    const mx = P0[0] * y[0] + V0[0] * yv[0], my = P0[1] * y[0] + V0[1] * yv[0];
    const vx = P0[0] * y[1] + V0[0] * yv[1], vy = P0[1] * y[1] + V0[1] * yv[1];
    curve.t.push(s); curve.mx.push(mx); curve.my.push(my);
    curve.ual.push(mx * E_AL[0] + my * E_AL[1]); curve.wal.push(vx * E_AL[0] + vy * E_AL[1]);
    curve.spp.push(y[2]); curve.spv.push(y[3]); curve.svv.push(y[4]);
  }
  /** Exact moments at the current time (integrated in step with the robots). */
  function moments() {
    const { y, yv } = momState;
    return { m: [P0[0] * y[0] + V0[0] * yv[0], P0[1] * y[0] + V0[1] * yv[0]],
      mv: [P0[0] * y[1] + V0[0] * yv[1], P0[1] * y[1] + V0[1] * yv[1]], spp: y[2], spv: y[3], svv: y[4] };
  }

  function step() {
    const sq = Math.sqrt(DT), g00 = G[0][0], g01 = G[0][1], g10 = G[1][0], g11 = G[1][1];
    for (let i = 0; i < N_ALL; i++) {
      const w1 = sq * R.normal(), w2 = sq * R.normal();
      const px = PX[i], py = PY[i], vx = VX[i], vy = VY[i];
      PX[i] = px + vx * DT; PY[i] = py + vy * DT;
      VX[i] = vx + (-K * px - C * vx) * DT + g00 * w1 + g01 * w2;
      VY[i] = vy + (-K * py - C * vy) * DT + g10 * w1 + g11 * w2;
    }
    momState.y = rk4(K, C, momState.y, DT); momState.yv = rk4(K, C, momState.yv, DT);
    k++; t = k * DT;
    if (k % REC === 0 && t <= tPlot + 1e-9) { const s = sampleCov(); rec.t.push(t); rec.al.push(s.al); rec.ac.push(s.ac); }
  }

  function sampleCov() {
    let ax = 0, ay = 0;
    for (let i = 0; i < N_ALL; i++) { ax += PX[i]; ay += PY[i]; }
    ax /= N_ALL; ay /= N_ALL;
    let sxx = 0, sxy = 0, syy = 0;
    for (let i = 0; i < N_ALL; i++) { const dx = PX[i] - ax, dy = PY[i] - ay; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; }
    const n1 = N_ALL - 1;
    sxx /= n1; sxy /= n1; syy /= n1;
    const q = (e) => e[0] * e[0] * sxx + 2 * e[0] * e[1] * sxy + e[1] * e[1] * syy;
    return { mean: [ax, ay], S: [[sxx, sxy], [sxy, syy]], al: q(E_AL), ac: q(E_AC) };
  }

  onInput(panel, 'k', release);
  onInput(panel, 'c', release);
  onInput(panel, 'noise', release);
  onClick(panel, 'release', release);
  release();

  function draw3d() {
    for (let i = 0; i < N_SHOW; i++) { xs[i] = PX[i] + SPOT[0]; ys[i] = PY[i] + SPOT[1]; }
    swarm.update(xs, ys, () => led);
    const M = moments(), m = [M.m[0] + SPOT[0], M.m[1] + SPOT[1]];
    const S = [[QQ[0][0] * M.spp, QQ[0][1] * M.spp], [QQ[1][0] * M.spp, QQ[1][1] * M.spp]];
    const show = t > 0 && NS > 0;
    ell1.visible = ell2.visible = show;
    if (show) { ell1.ellipse(m, S, 1); ell2.ellipse(m, S, 2); }
    meanDot.position.set(m[0], m[1], Z + 0.01);
    const j = Math.min(curve.t.length, Math.floor(t / 0.05) + 1);
    meanPath.mesh.geometry.setDrawRange(0, 6 * Math.max(0, j - 1));
    hud(panel, 'note', `${sw(CSS.yellow)}predicted 1σ and 2σ ellipses, mean &nbsp;${sw(CSS.white)}path of the mean<br>${sw(CSS.pink, true)}stationary 2σ (Lyapunov) &nbsp;${dot(CSS.pink)}spot &nbsp;${dot(CSS.blue)}${N_SHOW} of ${N_ALL} runs`);
    hud(panel, 'clock', `t = <b>${t.toFixed(1)} s</b><br>pushes strongest along the grey line`);
  }

  function draw2d() {
    plot.clear();
    const g = plot.g, W = plot.w, H = plot.h, L = 50, Rt = 14;
    label(g, 'dv = (−k p − c v) dt + G dW', W / 2, 22, CSS.yellow, { align: 'center', font: '600 14px Inter, Arial, sans-serif' });
    const qa = Q_ALONG * NS, M = moments();
    // phase plane along the push line
    const y1 = 62, h1 = Math.round(H * 0.4);
    top.set(L, y1, W - L - Rt, h1, ranges.u0, ranges.u1, ranges.w0, ranges.w1);
    top.frame({ title: 'phase plane along the push line', yticks: ticks(ranges.w0, ranges.w1, 4), fmt: tfmt,
      xticks: ticks(ranges.u0, ranges.u1, 4), xfmt: tfmt, xlabel: 'position (m)', ylabel: 'velocity (m/s)' });
    const n = Math.min(N_ALL, 1500), us = new Float64Array(n), ws = new Float64Array(n);
    for (let i = 0; i < n; i++) { us[i] = PX[i] * E_AL[0] + PY[i] * E_AL[1]; ws[i] = VX[i] * E_AL[0] + VY[i] * E_AL[1]; }
    top.dots(us, ws, CSS.blue, 1.3, 0.55);
    const j = Math.min(curve.t.length, Math.floor(t / 0.05) + 1);
    top.line(curve.ual, curve.wal, CSS.white, 1.4, { alpha: 0.85, i1: j });
    if (NS > 0) {
      const st = ellipsePts([0, 0], [[qa * qa / (2 * C * K), 0], [0, qa * qa / (2 * C)]], 2, 120);
      top.line(st.xs, st.ys, CSS.pink, 2, { dash: [6, 4], closed: true });
      if (t > 0) {
        const mu = [M.m[0] * E_AL[0] + M.m[1] * E_AL[1], M.mv[0] * E_AL[0] + M.mv[1] * E_AL[1]];
        for (const ns of [1, 2]) {
          const e = ellipsePts(mu, [[qa * qa * M.spp, qa * qa * M.spv], [qa * qa * M.spv, qa * qa * M.svv]], ns, 120);
          top.line(e.xs, e.ys, CSS.yellow, 2, { closed: true });
        }
      }
    }
    const mu = [M.m[0] * E_AL[0] + M.m[1] * E_AL[1], M.mv[0] * E_AL[0] + M.mv[1] * E_AL[1]];
    top.dots([mu[0]], [mu[1]], CSS.yellow, 4);
    // variances over time
    const y2 = y1 + h1 + 78, h2 = H - y2 - 58;
    bot.set(L, y2, W - L - Rt, h2, 0, tPlot, 0, ranges.vmax);
    bot.frame({ title: 'position variance (m²)', yticks: ticks(0, ranges.vmax, 3), fmt: tfmt,
      xticks: ticks(0, tPlot, 4), xfmt: tfmt, xlabel: 't (s)' });
    const qal = Q_ALONG * Q_ALONG * NS * NS, qac = Q_ACROSS * Q_ACROSS * NS * NS;
    bot.hline(qal / (2 * C * K), CSS.pink, { dash: [6, 4] });
    bot.hline(qac / (2 * C * K), CSS.pink, { dash: [6, 4] });
    bot.line(curve.t, curve.spp.map((s) => s * qal), CSS.yellow, 2);
    bot.line(curve.t, curve.spp.map((s) => s * qac), CSS.yellow, 2);
    bot.dots(rec.t, rec.al, CSS.blue, 2, 0.95);
    bot.dots(rec.t, rec.ac, CSS.blue, 2, 0.95);
    if (t < tPlot) bot.vline(t, CSS.grey);
    label(g, 'along', bot.X(tPlot) - 4, bot.Y(qal / (2 * C * K)) - 5, CSS.white, { align: 'right' });
    label(g, 'across', bot.X(tPlot) - 4, bot.Y(qac / (2 * C * K)) - 5, CSS.white, { align: 'right' });
    const yl = y2 + h2 + 42;
    let lx = L - 30;
    lx = legend(g, lx, yl, CSS.yellow, 'exact');
    lx = legend(g, lx, yl, CSS.pink, 'Lyapunov', 'dash');
    lx = legend(g, lx, yl, CSS.blue, 'robots', 'dot');
    legend(g, lx, yl, CSS.white, 'mean path');
  }

  function text() {
    const s = sampleCov(), sal = Q_ALONG ** 2 * NS * NS / (2 * C * K), sac = Q_ACROSS ** 2 * NS * NS / (2 * C * K);
    readout(panel, 'info', `stationary variance along <b>${sal.toFixed(4)}</b> m², across <b>${sac.toFixed(4)}</b> (Lyapunov)`
      + ` · robots now <b>${s.al.toFixed(4)}</b>, <b>${s.ac.toFixed(4)}</b> · ζ = ${(C / (2 * Math.sqrt(K))).toFixed(2)}`);
  }

  v.onFrame((dt) => {
    acc += dt * tPlot / PLAY;
    let n = 0;
    while (acc >= DT && n < 600) { step(); acc -= DT; n++; }
    if (n === 600) acc = 0;
    draw3d();
    draw2d();
    text();
  });

  panel._run = (sec) => { const n = Math.round(sec / DT); for (let i = 0; i < n; i++) step(); return panel._state(); };
  panel._state = () => {
    const s = sampleCov(), M = moments();
    const lyap = [[QQ[0][0] / (2 * C * K), QQ[0][1] / (2 * C * K)], [QQ[1][0] / (2 * C * K), QQ[1][1] / (2 * C * K)]];
    return {
      k: K, c: C, noise: NS, damping_ratio: C / (2 * Math.sqrt(K)), t, dt: DT, runs: N_ALL, t_plot: tPlot,
      lyapunov_position_cov: lyap, lyapunov_var_along: Q_ALONG ** 2 * NS * NS / (2 * C * K), lyapunov_var_across: Q_ACROSS ** 2 * NS * NS / (2 * C * K),
      exact_position_cov_now: [[QQ[0][0] * M.spp, QQ[0][1] * M.spp], [QQ[1][0] * M.spp, QQ[1][1] * M.spp]],
      sim_position_cov_now: s.S, sim_var_along: s.al, sim_var_across: s.ac,
      exact_mean_now: M.m, sim_mean_now: s.mean,
      exact_mean_distance: Math.hypot(M.m[0], M.m[1]), sim_mean_distance: Math.hypot(s.mean[0], s.mean[1]),
    };
  };
}
