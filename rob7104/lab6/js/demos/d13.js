// Lab 6, demo d13 (section 1.3): inverse kinematics is optimization.
// The UR5e must reach a target position AND orientation (the task of clip s13_ik). One solver iteration per tick,
// computed with sim6.js (ikJacobianTranspose / ikGaussNewton / ikDampedLeastSquares with maxIter 1, the same
// formulas as the clip's sim/solvers.py). Right panel: the gain of each solver against the singular value of the
// Jacobian, with the current singular values on it, and the residual / joint step of the last 60 iterations.
// "loop" replays the clip's task (target moved along the clip's loop, 60 iterations from the clip's start pose)
// and compares the browser's residuals with the Python run (data/d13_ik.json, copied from out/s13_ik_facts.json).
import { S, Trail, makeArmViewer, marker, Line3D, world, COL, CSS, val, onInput, onClick, readout, seg,
  toggle, setVal, heldKeys, plot2d, fmt } from '../common.js';

const RW = 0.3;                                       // metres per radian of orientation error (clip and sim6 default)
const COLOR = { jt: CSS.grey, gn: CSS.pink, dls: CSS.white };
const HEX = { jt: COL.grey, gn: COL.pink, dls: COL.white };
const LABEL = { jt: 'J transpose', gn: 'Gauss–Newton', dls: 'damped' };
const CAMERA = { lookat: [0.45, -0.06, 1.30], distance: 2.0, azimuth: -75, elevation: -12 };
const HIST = 60;                                      // iterations shown in the history strips
const BIG_STEP = 1.0;                                 // rad: "big step" as in the facts (steps_above_1_rad)

const col = (M, i) => M.map((r) => r[i]);

export async function init(panel) {
  const D = await (await fetch(new URL('../../data/d13_ik.json', import.meta.url))).json();
  const Rt = D.target_orientation, L = D.loop, SCHED = D.target_parameter_each_iteration;
  const [S0, S1] = D.loop_parameter_at_reach_crossings;
  const PHASE = Object.entries(D.schedule_iterations).flatMap(([name, n]) => Array(n).fill(name));
  let loopsDone = 0;                                  // for tests: data-d13loop = loops finished, data-d13phase
  const loopPoint = (s) => {
    const c = Math.cos(2 * Math.PI * s), sn = Math.sin(2 * Math.PI * s);
    return [0, 1, 2].map((j) => L.centre_m[j] - L.semi_axes_m[0] * c * L.long_axis[j] + L.semi_axes_m[1] * sn * L.short_axis[j]);
  };

  const v = await makeArmViewer(panel, { camera: CAMERA, q: D.start_pose_rad });

  // ------------------------------------------------------------------ state
  let method = 'dls', lam = val(panel, 'lam'), rate = val(panel, 'rate'), running = true;
  let q = D.start_pose_rad.slice();
  let target = loopPoint(0);
  let mode = 'free', loopK = 0, rec = null, check = null;
  let iter = 0, last = { r: NaN, dq: NaN, sigmaMin: NaN };
  let closest = { residual: NaN, posGap: NaN, reachable: null, ms: NaN, source: 'none yet' };
  const hist = [];                                    // {r, dq, m, out, rstar}
  let qFrom = q.slice(), qTo = q.slice(), tIter = performance.now(), tNext = tIter;
  let dirty = true, closestTimer = null;
  let ana = null;                                     // singular values, error components, alpha at (q, target)

  // ------------------------------------------------------------------ 3D objects
  const tMark = marker(v.scene, COL.green, 0.022, 0.95);
  const tAppr = new Line3D(v.scene, COL.green, 2), tSide = new Line3D(v.scene, COL.green, 2);
  const hAppr = new Line3D(v.scene, COL.white, 2), hSide = new Line3D(v.scene, COL.white, 2);
  const gap = new Line3D(v.scene, COL.red, 2);
  const loopIn = new Line3D(v.scene, COL.green, 260, { opacity: 0.55 });
  const loopOut = new Line3D(v.scene, COL.red, 140, { opacity: 0.7 });
  const trail = new Trail(v.scene, HEX[method], 90);
  {                                                   // the clip's loop: green where reachable, red beyond the reach
    const arc = (a, b, n) => Array.from({ length: n + 1 }, (_, k) => world(loopPoint(a + (b - a) * k / n)));
    loopIn.set(arc(S1, 1 + S0, 240));                 // loopPoint is periodic in s: one arc through s = 0
    loopOut.set(arc(S0, S1, 120));
  }
  const frameLines = (p, R, appr, side, la, ls) => {
    const P = world(p);
    appr.set([P, [0, 1, 2].map((j) => P[j] + la * R[j][2])]);
    side.set([P, [0, 1, 2].map((j) => P[j] + ls * R[j][0])]);
  };
  const placeTarget = () => { tMark.position.set(...world(target)); frameLines(target, Rt, tAppr, tSide, 0.13, 0.06); };
  placeTarget();

  // ------------------------------------------------------------------ the 2D panel
  const P = plot2d(panel.querySelector('canvas.plot'));
  const lx = (x, a, b) => (Math.log10(x) - Math.log10(a)) / (Math.log10(b) - Math.log10(a));
  const gain = (m, s) => (m === 'gn' ? 1 / s : m === 'jt' ? ana.alpha * s : s / (s * s + lam * lam));

  function analyse() {
    const { e, J } = S.poseError(q, target, Rt, RW);
    const sv = S.svd(J), ne = S.norm(e) || 1;
    const dirs = sv.s.map((s, i) => ({ s, c: S.dot(col(sv.U, i), e) / ne }));
    dirs.sort((a, b) => b.s - a.s);
    const g = S.matTvec(J, e), Jg = S.matvec(J, g), den = S.dot(Jg, Jg);
    ana = { dirs, alpha: den > 0 ? S.dot(g, g) / den : 0, r: S.norm(e), posErr: S.norm(e.slice(0, 3)), rotErr: S.norm(e.slice(3)) / RW };
  }

  function drawPlot() {
    P.clear();
    const g = P.g, W = P.w, H = P.h, x0 = 50, x1 = W - 14;
    g.font = '11px Inter, sans-serif'; g.textBaseline = 'alphabetic';
    // ---- gains
    const top = { y0: 34, y1: Math.round(H * 0.55) - 22 };
    const SX = [1e-3, 3], SY = [0.01, 1000];
    const X = (s) => x0 + (x1 - x0) * lx(s, SX[0], SX[1]);
    const Y = (y) => top.y1 - (top.y1 - top.y0) * lx(Math.min(Math.max(y, SY[0] * 0.5), SY[1] * 2), SY[0], SY[1]);
    g.fillStyle = CSS.mute; g.textAlign = 'left';
    g.fillText('gain per error direction · dots: current σᵢ (size: share of the error)', x0 - 36, 14);
    g.strokeStyle = 'rgba(97,97,107,1)'; g.lineWidth = 1;
    g.strokeRect(x0 + 0.5, top.y0 + 0.5, x1 - x0, top.y1 - top.y0);
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const y of [0.01, 0.1, 1, 10, 100, 1000]) {
      g.fillStyle = CSS.faint; g.fillRect(x0 + 1, Y(y), x1 - x0 - 1, 0.6);
      g.fillStyle = CSS.mute; g.fillText(y >= 1 ? String(y) : y.toString(), x0 - 5, Y(y));
    }
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (const s of [1e-3, 1e-2, 1e-1, 1]) g.fillText(s === 1 ? '1' : `10${['⁻¹', '⁻²', '⁻³'][Math.round(-Math.log10(s)) - 1]}`, X(s), top.y1 + 4);
    g.fillText('singular value σ of the Jacobian', (x0 + x1) / 2, top.y1 + 16);
    g.save(); g.translate(12, (top.y0 + top.y1) / 2); g.rotate(-Math.PI / 2); g.textBaseline = 'middle';
    g.fillText('joint step per unit of error', 0, 0); g.restore();
    g.save(); g.beginPath(); g.rect(x0, top.y0, x1 - x0, top.y1 - top.y0); g.clip();
    for (const m of ['jt', 'gn', 'dls']) {
      g.strokeStyle = COLOR[m]; g.globalAlpha = m === method ? 1 : 0.28; g.lineWidth = m === method ? 2.2 : 1.3;
      g.beginPath();
      for (let k = 0; k <= 160; k++) {
        const s = SX[0] * Math.pow(SX[1] / SX[0], k / 160), px = X(s), py = Y(gain(m, s));
        if (k) g.lineTo(px, py); else g.moveTo(px, py);
      }
      g.stroke();
    }
    g.globalAlpha = 1;
    if (method === 'dls') {                         // the cap 1/(2 lambda) at sigma = lambda
      g.setLineDash([4, 4]); g.strokeStyle = CSS.mute; g.lineWidth = 1;
      g.beginPath(); g.moveTo(x0, Y(1 / (2 * lam))); g.lineTo(X(lam), Y(1 / (2 * lam))); g.lineTo(X(lam), top.y1); g.stroke();
      g.setLineDash([]);
      g.fillStyle = CSS.mute; g.textAlign = 'left'; g.textBaseline = 'bottom';
      g.fillText('1/(2λ)', x0 + 4, Y(1 / (2 * lam)) - 2);
    }
    for (const d of ana.dirs) {                       // the current singular directions; size = share of the error
      const s = Math.max(d.s, SX[0] * 1.05), px = X(s);
      const py = Math.min(top.y1 - 4, Math.max(top.y0 + 4, Y(gain(method, Math.max(d.s, 1e-12)))));
      const rr = 2.5 + 5.5 * Math.abs(d.c);
      g.fillStyle = COLOR[method]; g.strokeStyle = '#000'; g.lineWidth = 1.2;
      g.beginPath(); g.arc(px, py, rr, 0, 2 * Math.PI); g.fill(); g.stroke();
    }
    g.restore();
    g.font = '11.5px Inter, sans-serif'; g.textAlign = 'right'; g.textBaseline = 'top';
    const leg = [['jt', 'gradient  α σ'], ['gn', 'Gauss–Newton  1/σ'], ['dls', `damped  σ/(σ²+λ²), λ = ${fmt(lam, 3)}`]];
    leg.forEach(([m, t], i) => { g.fillStyle = COLOR[m]; g.globalAlpha = m === method ? 1 : 0.5; g.fillText(t, x1 - 6, top.y0 + 6 + 15 * i); });
    g.globalAlpha = 1;

    // ---- history: residual and joint step of the last HIST iterations
    const strip = (y0, y1, R, ticks, key, label) => {
      const Ys = (y) => y1 - (y1 - y0) * Math.min(1, Math.max(0, lx(y, R[0], R[1])));
      const Xs = (i) => x0 + (x1 - x0) * (i + 0.5) / HIST;
      const off = HIST - hist.length;
      hist.forEach((h, i) => { if (h.out) { g.fillStyle = 'rgba(252,98,85,0.16)'; g.fillRect(Xs(off + i) - (x1 - x0) / HIST / 2, y0, (x1 - x0) / HIST + 0.5, y1 - y0); } });
      g.strokeStyle = 'rgba(97,97,107,1)'; g.lineWidth = 1; g.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0, y1 - y0);
      g.font = '11px Inter, sans-serif'; g.textAlign = 'right'; g.textBaseline = 'middle'; g.fillStyle = CSS.mute;
      for (const [y, t] of ticks) { g.fillText(t, x0 - 5, Ys(y)); g.fillStyle = CSS.faint; g.fillRect(x0 + 1, Ys(y), x1 - x0 - 1, 0.6); g.fillStyle = CSS.mute; }
      g.save(); g.translate(12, (y0 + y1) / 2); g.rotate(-Math.PI / 2); g.textAlign = 'center'; g.fillText(label, 0, 0); g.restore();
      if (key === 'r') {                            // closest possible residual (dashed green)
        g.strokeStyle = CSS.green; g.setLineDash([3, 3]); g.lineWidth = 1.2;
        hist.forEach((h, i) => { if (h.rstar > 1e-6) { g.beginPath(); g.moveTo(Xs(off + i) - (x1 - x0) / HIST / 2, Ys(h.rstar)); g.lineTo(Xs(off + i) + (x1 - x0) / HIST / 2, Ys(h.rstar)); g.stroke(); } });
        g.setLineDash([]);
      } else {                                      // half a turn
        g.strokeStyle = CSS.mute; g.setLineDash([2, 4]); g.beginPath(); g.moveTo(x0, Ys(Math.PI)); g.lineTo(x1, Ys(Math.PI)); g.stroke(); g.setLineDash([]);
        g.fillStyle = CSS.mute; g.textAlign = 'right'; g.textBaseline = 'bottom'; g.fillText('half a turn', x1 - 4, Ys(Math.PI) - 1);
      }
      g.lineWidth = 1.8;
      for (let i = 1; i < hist.length; i++) {
        const a = hist[i - 1][key], b = hist[i][key];
        if (!(a > 0) || !(b > 0)) continue;
        g.strokeStyle = COLOR[hist[i].m]; g.beginPath(); g.moveTo(Xs(off + i - 1), Ys(a)); g.lineTo(Xs(off + i), Ys(b)); g.stroke();
      }
      if (hist.length) { const h = hist[hist.length - 1]; if (h[key] > 0) { g.fillStyle = COLOR[h.m]; g.beginPath(); g.arc(Xs(HIST - 1), Ys(h[key]), 3.2, 0, 2 * Math.PI); g.fill(); } }
    };
    const b0 = Math.round(H * 0.55) + 14, mid = Math.round(H * 0.55 + (H * 0.45 - 30) / 2) + 6;
    strip(b0, mid - 6, [1e-6, 2], [[1e-6, '10⁻⁶'], [1e-3, '10⁻³'], [1, '1']], 'r', 'residual');
    strip(mid + 4, H - 22, [1e-4, 100], [[1e-4, '10⁻⁴'], [1e-2, '10⁻²'], [1, '1'], [100, '100']], 'dq', 'joint step');
    g.fillStyle = CSS.mute; g.textAlign = 'center'; g.textBaseline = 'top'; g.font = '11px Inter, sans-serif';
    g.font = (W < 520 ? '10px' : '11px') + ' Inter, sans-serif';
    g.fillText(`last ${HIST} iterations · red: out of reach · green dashes: closest possible`, (x0 + x1) / 2, H - 16);
  }

  // ------------------------------------------------------------------ reachability (free target): best of local solves
  function scheduleClosest() {
    const was = closest.reachable;
    closest = { residual: NaN, posGap: NaN, reachable: was, ms: NaN, source: 'pending' };
    clearTimeout(closestTimer);
    closestTimer = setTimeout(() => {
      const t0 = performance.now();
      let best = null;
      for (const q0 of [q.slice(), ...D.configuration_start_poses_rad]) {
        const r = S.ikLevenbergMarquardt(q0, target, Rt, { rotWeight: RW, maxIter: 150 });
        const k = r.err.length - 1;
        if (!best || r.err[k] < best.residual) best = { residual: r.err[k], posGap: r.errPos[k] };
        if (best.residual <= 1e-6) break;
      }
      closest = { ...best, reachable: best.residual <= 1e-6, ms: performance.now() - t0,
        source: 'Levenberg–Marquardt from the current pose and the 8 configurations of the clip' };
      dirty = true;
    }, 250);
  }
  const outNow = () => (mode === 'loop' ? null : closest.reachable === false);

  // ------------------------------------------------------------------ one iteration
  function iterate(now) {
    let pt = target, s = null;
    if (mode === 'loop') {
      s = SCHED[loopK]; pt = target = loopPoint(s); placeTarget();
      panel.dataset.d13phase = PHASE[loopK];
      ['tx', 'ty', 'tz'].forEach((n, j) => ctl_set(n, target[j]));
    }
    const o = { maxIter: 1, tol: 0, tolStep: 0, rotWeight: RW };
    const res = method === 'jt' ? S.ikJacobianTranspose(q, pt, Rt, o)
      : method === 'gn' ? S.ikGaussNewton(q, pt, Rt, o) : S.ikDampedLeastSquares(q, pt, Rt, { ...o, damping: lam });
    const qn = res.q[res.q.length - 1];
    const r0 = res.err[0], r = res.err[res.err.length - 1], dq = res.dqNorm.length ? res.dqNorm[0] : 0;
    const disp = displayQ(now);
    qFrom = disp; qTo = qn.slice(); q = qn.slice(); tIter = now;
    iter++;
    last = { r, dq, sigmaMin: res.sigmaMin[0] };
    let out, rstar;
    if (mode === 'loop') {
      out = s > S0 && s < S1;
      rstar = s === 0.5 ? D.far_point.closest_residual : NaN;
      if (rec) { if (!rec.r.length) rec.r.push(r0); rec.r.push(r); rec.dq.push(dq); }
      loopK++;
      if (loopK >= SCHED.length) finishLoop();
    } else {
      out = outNow();
      rstar = closest.reachable === false ? closest.residual : NaN;
    }
    hist.push({ r, dq, m: method, out: !!out, rstar });
    if (hist.length > HIST) hist.shift();
    dirty = true;
  }

  function runKey() {
    if (method === 'jt') return 'jt';
    if (method === 'gn') return 'gn';
    if (Math.abs(lam - 0.1) < 1e-9) return 'dls';
    if (Math.abs(lam - 0.05) < 1e-9) return 'dls_small';
    return null;
  }
  function finishLoop() {
    mode = 'free';
    if (rec && rec.valid) {
      const key = runKey(), py = key ? D.runs[key] : null;
      const big = (a) => a.filter((x) => x > BIG_STEP).length;
      const calm = (a) => { let k = -1; a.forEach((x, i) => { if (x > BIG_STEP) k = i; }); return k + 1; };
      check = { method: rec.method, damping: rec.damping, pythonRun: key, n: rec.dq.length,
        residualBrowserAt: { 0: rec.r[0], 8: rec.r[8], 16: rec.r[16], 38: rec.r[38], 60: rec.r[60] },
        stepsAbove1RadBrowser: big(rec.dq), calmFromIterationBrowser: calm(rec.dq) };
      if (py) {
        const R = py.residual_after_each_iteration, d = (n) => Math.max(...rec.r.slice(0, n).map((x, i) => Math.abs(x - R[i])));
        Object.assign(check, {
          residualPythonAt: { 0: R[0], 8: R[8], 16: R[16], 38: R[38], 60: R[60] },
          maxAbsDiffResidualFirst17: d(17), maxAbsDiffResidualAll: d(61),
          stepsAbove1RadPython: py.steps_above_1_rad, calmFromIterationPython: py.calm_from_iteration });
      }
    }
    rec = null;
    panel.dataset.d13phase = 'free';
    panel.dataset.d13loop = String(++loopsDone);
    const t = loopPoint(1);
    for (const [n, x] of [['tx', t[0]], ['ty', t[1]], ['tz', t[2]]]) ctl_set(n, x);
    target = t; placeTarget(); scheduleClosest();
  }
  let silent = false;
  function ctl_set(n, x) { silent = true; setVal(panel, n, x); silent = false; }

  function displayQ(now) {
    const per = 1000 / rate, u = Math.min(1, (now - tIter) / (0.75 * per)), w = u * u * (3 - 2 * u);
    return qFrom.map((a, i) => a + w * S.wrap(qTo[i] - a));
  }

  // ------------------------------------------------------------------ controls
  seg(panel, 'method', (m) => { method = m; trail.color.set(HEX[m]); if (rec) rec.valid = false; dirty = true; });
  onInput(panel, 'lam', () => { lam = val(panel, 'lam'); if (rec && method === 'dls') rec.valid = false; dirty = true; });
  onInput(panel, 'rate', () => { rate = val(panel, 'rate'); });
  toggle(panel, 'run', (on) => { running = on; tNext = performance.now(); });
  onClick(panel, 'step', () => { if (!running) iterate(performance.now()); });
  const onTarget = () => {
    if (silent) return;
    if (mode === 'loop') { mode = 'free'; rec = null; panel.dataset.d13phase = 'free'; }
    target = [val(panel, 'tx'), val(panel, 'ty'), val(panel, 'tz')];
    placeTarget(); scheduleClosest(); dirty = true;
  };
  for (const n of ['tx', 'ty', 'tz']) onInput(panel, n, onTarget);
  function restart() {
    q = D.start_pose_rad.slice(); qFrom = q.slice(); qTo = q.slice();
    hist.length = 0; iter = 0; trail.clear(); tIter = tNext = performance.now();
  }
  onClick(panel, 'loop', () => {
    restart();
    const t = loopPoint(0);
    for (const [n, x] of [['tx', t[0]], ['ty', t[1]], ['tz', t[2]]]) ctl_set(n, x);
    target = t; placeTarget();
    mode = 'loop'; loopK = 0; check = null;
    rec = { method, damping: method === 'dls' ? lam : null, r: [], dq: [], valid: true };
    if (!running) { ctl('run'); }
    dirty = true;
  });
  function ctl(n) { const b = panel.querySelector(`[data-ctl="${n}"]`); if (!b.classList.contains('on')) b.click(); }
  onClick(panel, 'reset', () => {
    mode = 'free'; rec = null; restart(); panel.dataset.d13phase = 'free';
    const t = loopPoint(0);
    for (const [n, x] of [['tx', t[0]], ['ty', t[1]], ['tz', t[2]]]) ctl_set(n, x);
    target = t; placeTarget(); scheduleClosest(); dirty = true;
  });
  const held = heldKeys(panel);
  let qShown = q.slice();
  let tKey = performance.now();

  // ------------------------------------------------------------------ frame loop
  scheduleClosest();
  v.onFrame(() => {
    const now = performance.now(), dtk = Math.min(0.1, (now - tKey) / 1000);
    tKey = now;
    if (held.size) {
      const sp = 0.25 * dtk, k = (a, b) => (held.has(a) ? 1 : 0) - (held.has(b) ? 1 : 0);
      const dx = k('ArrowLeft', 'ArrowRight'), dz = k('ArrowUp', 'ArrowDown');
      if (dx) setVal(panel, 'tx', val(panel, 'tx') + sp * dx);
      if (dz) setVal(panel, 'tz', val(panel, 'tz') + sp * dz);
    }
    if (running) {
      const per = 1000 / rate;
      let n = 0;
      while (now >= tNext && n < 4) { iterate(now); tNext += per; n++; }
      if (now - tNext > per) tNext = now + per;
    }
    const qd = displayQ(now);
    qShown = qd;
    v.setArm(qd); v.forward();
    const f = S.fk(qd);
    frameLines(f.p, f.R, hAppr, hSide, 0.13, 0.06);
    const pw = world(f.p), tw = world(target);
    trail.push(pw);
    gap.visible = Math.hypot(pw[0] - tw[0], pw[1] - tw[1], pw[2] - tw[2]) > 0.005;
    gap.set([pw, tw]);
    if (dirty) {
      analyse(); drawPlot(); dirty = false;
      const reach = mode === 'loop'
        ? (SCHED[Math.min(loopK, SCHED.length - 1)] > S0 && SCHED[Math.min(loopK, SCHED.length - 1)] < S1
          ? '<span class="c-red">target out of reach</span>' : '<span class="c-green">target reachable</span>')
        : closest.reachable === null ? '<span class="c-grey">checking reach…</span>'
          : closest.reachable ? '<span class="c-green">target reachable</span>'
            : `<span class="c-red">out of reach: closest pose ${fmt(100 * closest.posGap, 1)} cm away</span>`;
      readout(panel, 'info', `${mode === 'loop' ? `loop ${loopK}/${SCHED.length} · ` : ''}iteration ${iter} · residual ${last.r > 0 ? last.r.toExponential(1) : '—'}`
        + ` · step ${fmt(last.dq, 3)} rad · σ<sub>min</sub> ${fmt(last.sigmaMin, 3)} · ${reach}`);
    }
  });
  window.addEventListener('resize', () => { dirty = true; });

  panel._state = () => {
    const f = S.fk(qShown);
    return {
      method, damping: lam, rate, running, mode, loopK, iter, target,
      residual: last.r, jointStep: last.dq, sigmaMin: last.sigmaMin,
      posErrM: ana && ana.posErr, rotErrRad: ana && ana.rotErr, alphaJT: ana && ana.alpha,
      sigmas: ana && ana.dirs.map((d) => d.s), errorShare: ana && ana.dirs.map((d) => d.c * d.c),
      closest, toolShownWorld: world(f.p), toolViewer: v.tool(), q,
      loopStartErrM: Math.hypot(...loopPoint(0).map((x, i) => x - L.start_m[i])),
      loopFarErrM: Math.hypot(...loopPoint(0.5).map((x, i) => x - L.far_point_m[i])),
      check,
    };
  };
}
