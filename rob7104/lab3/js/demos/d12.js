// 1.2 A slice is a density: M robots released at x = 0 diffuse, dX = sqrt(2D) dW, D = 0.01 m^2/s, for 12 s.
// The wall shows the histogram of all M robots at the slice time against N(0, 2Dt). Right: 40 paths with the
// slice (drag it), and the histogram error against M with the 1/sqrt(M) line. Same numbers as video s01_slice.
import { rng, gaussPdf } from '../sim3.js';
import { makeViewer, placeLamp, SwarmMesh, DensityWall, THREE, COL, CSS, ctl, val, onInput, onClick, readout, plot2d } from '../common3.js';
import { Ribbon, Ax, label, binGauss, logSlope, median, sig3 } from '../noise_kit.js';

const D = 0.01, T = 12, FPS = 30, NS = T * FPS;           // snapshots every 1/30 s (exact Gaussian increments)
const LO = -1.5, HI = 1.5, NB = 30, W = (HI - LO) / NB;   // 0.1 m bins, as in the video
const VMAX = 2.4, HEIGHT = 0.95, Y0 = 0.32;      // the video's wall is 0.75 m tall; taller reads better in a narrow view
const MS = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 20000];
const MDRAW = 300, NPATH = 40;
const STUDY_M = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600];

/** L1 distance between a histogram (density units) and the bin-averaged exact density. */
function l1(h, pbar) { let e = 0; for (let k = 0; k < NB; k++) e += Math.abs(h[k] - pbar[k]) * W; return e; }

function histOf(x, out) {
  out.fill(0);
  for (let i = 0; i < x.length; i++) { const k = Math.floor((x[i] - LO) / W); if (k >= 0 && k < NB) out[k]++; }
  for (let k = 0; k < NB; k++) out[k] /= x.length * W;
  return out;
}

const PBAR = Array.from({ length: NS + 1 }, (_, f) => (f ? binGauss(LO, HI, NB, 0, Math.sqrt(2 * D * f / FPS)) : null));

/** Median histogram error at t = T against M (as in the video: reps = clip(200000 / M, 20, 400)). */
function errorStudy() {
  const R = rng(11), sd = Math.sqrt(2 * D * T), pbar = PBAR[NS], h = new Float64Array(NB);
  return STUDY_M.map((m) => {
    const reps = Math.min(400, Math.max(20, Math.floor(200000 / m))), x = new Float64Array(m), e = [];
    for (let r = 0; r < reps; r++) { for (let i = 0; i < m; i++) x[i] = sd * R.normal(); e.push(l1(histOf(x, h), pbar)); }
    return median(e);
  });
}

export async function init(panel) {
  const v = await makeViewer(panel, 'corridor.mjb', { lookat: [0, 0.2, 0.42], distance: 3.55, azimuth: 90, elevation: -21, fov: 40 });
  for (let i = 0; i < 3; i++) placeLamp(v, i, null);
  v.forward();
  const wall = new DensityWall(v.scene, { lo: LO, hi: HI, nb: NB, y0: Y0, z0: 0.06, height: HEIGHT, vmax: VMAX, curveColours: [] });
  const exactCurve = new Ribbon(v.scene, COL.yellow, { width: 0.014 });
  const swarm = new SwarmMesh(v.scene, MDRAW);
  const R0 = rng(5);
  const ry = Float64Array.from({ length: MDRAW }, () => (R0.uni() - 0.5) * 2 * (0.16 - 0.035));
  const blue = new THREE.Color(CSS.blue);
  const P = plot2d(panel.querySelector('canvas.plot'));
  const study = errorStudy();
  const slope = logSlope(STUDY_M, study);

  let run = null, t = 0, playing = true;

  function simulate(M) {
    const R = rng(7 + M), x = new Float64Array(M), s = Math.sqrt(2 * D / FPS);
    const md = Math.min(M, MDRAW);
    const run = { M, md, hist: new Float32Array((NS + 1) * NB), vars: new Float64Array(NS + 1), err: new Float64Array(NS + 1), drawn: new Float32Array((NS + 1) * md) };
    const h = new Float64Array(NB);
    for (let f = 0; f <= NS; f++) {
      if (f) for (let i = 0; i < M; i++) x[i] += s * R.normal();
      histOf(x, h); run.hist.set(h, f * NB);
      let m = 0, q = 0; for (let i = 0; i < M; i++) { m += x[i]; q += x[i] * x[i]; }
      m /= M; run.vars[f] = q / M - m * m;
      run.err[f] = f ? l1(h, PBAR[f]) : NaN;
      for (let i = 0; i < md; i++) run.drawn[f * md + i] = x[i];
    }
    return run;
  }

  const mSlider = ctl(panel, 'm'), tSlider = ctl(panel, 't');
  function setM() {
    run = simulate(MS[val(panel, 'm')]);
    ctl(panel, 'mout').textContent = String(run.M);
  }
  function setPlaying(on) { playing = on; ctl(panel, 'play').classList.toggle('on', on); ctl(panel, 'play').textContent = on ? 'pause' : 'play'; }
  function setT(tt, fromSlider = false) {
    t = Math.max(0, Math.min(T, tt));
    if (!fromSlider) { tSlider.value = t; tSlider.parentElement.querySelector('output').textContent = t.toFixed(1) + ' s'; }
  }

  onInput(panel, 'm', setM);
  onInput(panel, 't', () => { setPlaying(false); setT(val(panel, 't'), true); });
  onClick(panel, 'play', () => { if (!playing && t >= T) setT(0); setPlaying(!playing); });
  onClick(panel, 'replay', () => { setT(0); setPlaying(true); });

  // drag the slice on the paths plot
  let pathAx = null, dragging = false;
  const pc = panel.querySelector('canvas.plot');
  const fromPointer = (e) => { if (!pathAx) return; const r = pc.getBoundingClientRect(); setT(pathAx.invX(e.clientX - r.left)); };
  pc.addEventListener('pointerdown', (e) => {
    const r = pc.getBoundingClientRect(), y = e.clientY - r.top;
    if (!pathAx || y > pathAx.y + pathAx.h + 24) return;
    dragging = true; try { pc.setPointerCapture(e.pointerId); } catch (_) { /* synthetic events */ } setPlaying(false); fromPointer(e);
  });
  pc.addEventListener('pointermove', (e) => { if (dragging) fromPointer(e); });
  pc.addEventListener('pointerup', () => { dragging = false; });

  const xs = Float64Array.from({ length: 241 }, (_, i) => LO + (HI - LO) * i / 240);
  const zs = new Float64Array(xs.length), keep = new Array(xs.length);
  const px = new Float64Array(MDRAW);
  const tg = Float64Array.from({ length: NS + 1 }, (_, f) => f / FPS);

  function frameIndex() { return Math.min(NS, Math.round(t * FPS)); }

  function show3d() {
    const u = t * FPS, f0 = Math.min(NS - 1, Math.floor(u)), fr = Math.min(1, u - f0), md = run.md;
    for (let i = 0; i < md; i++) {
      const a = run.drawn[f0 * md + i], b = run.drawn[(f0 + 1) * md + i];
      px[i] = a + fr * (b - a);
    }
    swarm.update(px, ry, blue, md);
    const f = frameIndex();
    wall.update(run.hist.subarray(f * NB, (f + 1) * NB));
    if (t > 1e-6) {
      const s = HEIGHT / VMAX, cap = 1.25 * VMAX;
      for (let i = 0; i < xs.length; i++) { const p = gaussPdf(xs[i], 0, 2 * D * t); zs[i] = 0.06 + Math.min(p, cap) * s; keep[i] = p <= cap && gaussPdf(xs[i + 1] ?? xs[i], 0, 2 * D * t) <= cap; }
      exactCurve.set(xs, zs, Y0 - 0.008, keep); exactCurve.visible = true;
    } else exactCurve.visible = false;
    const vr = run.vars[f], ex = 2 * D * f / FPS;
    readout(panel, 'info', `t = <b>${(f / FPS).toFixed(1)} s</b>&ensp; Var x: robots <b>${vr.toFixed(4)} m²</b>, 2Dt = <b>${ex.toFixed(4)} m²</b>`
      + `&ensp; <span style="color:${CSS.pink}">histogram error now ${f ? run.err[f].toFixed(3) : '–'}</span>&ensp;(${run.M} robots, ${run.md} drawn)`);
  }

  function draw() {
    P.clear();
    const g = P.g, w = P.w, h = P.h, L = 46, Rm = 14, iw = w - L - Rm;
    if (iw < 80 || h < 260) return;
    const hA = Math.round((h - 120) * 0.5), hB = h - 120 - hA;
    const yA = 30, yB = yA + hA + 62;
    const f = frameIndex();
    // paths
    const A = new Ax(g, L, yA, iw, hA, [0, T], [-1.5, 1.5]);
    pathAx = A;
    label(g, `${NPATH} paths x(t); drag the pink slice`, L - 36, yA - 12);
    A.frame([[0, '0'], [2, '2'], [4, '4'], [6, '6'], [8, '8'], [10, '10'], [12, '12 s']], [[-1, '−1'], [0, '0'], [1, '1 m']]);
    A.clip(() => {
      const ys = new Float64Array(NS + 1), md = run.md;
      for (let i = 0; i < Math.min(NPATH, md); i++) {
        for (let k = 0; k <= NS; k++) ys[k] = run.drawn[k * md + i];
        A.line(tg, ys, CSS.blue, 0.8, { alpha: 0.16 });
        A.line(tg, ys, CSS.blue, 0.9, { alpha: 0.8, i1: f + 1 });
      }
      const tt = [], up = [], dn = [];
      for (let k = 0; k <= 120; k++) { const s = k / 120 * T; tt.push(s); up.push(2 * Math.sqrt(2 * D * s)); dn.push(-2 * Math.sqrt(2 * D * s)); }
      A.line(tt, up, CSS.yellow, 1.2, { dash: [4, 3] }); A.line(tt, dn, CSS.yellow, 1.2, { dash: [4, 3] });
    });
    A.vline(f / FPS, CSS.pink, 2.2);
    g.save(); g.fillStyle = CSS.pink; g.beginPath(); g.arc(A.X(f / FPS), yA - 1, 5, 0, 2 * Math.PI); g.fill(); g.restore();
    label(g, '±2 sd = ±2√(2Dt)', A.x + A.w - 4, A.y + 12, CSS.yellow, 'right', '11px Inter, system-ui, sans-serif');
    // error against M
    const B = new Ax(g, L, yB, iw, hB, [18, 36000], [0.012, 1.2], { xlog: true, ylog: true });
    label(g, `histogram error at t = 12 s  (fitted slope ${slope.toFixed(2)})`, L - 36, yB - 12);
    B.frame([[100, '100'], [1000, '1000'], [10000, '10 000']], [[0.02, '0.02'], [0.1, '0.1'], [1, '1']]);
    label(g, 'number of robots M', B.x + 8, B.y + B.h - 8, CSS.grey, 'left', '11px Inter, system-ui, sans-serif');
    B.clip(() => {
      const i4 = STUDY_M.indexOf(400);
      const ref = STUDY_M.map((m) => study[i4] * Math.sqrt(400 / m));
      B.line(STUDY_M, ref, CSS.grey, 1.2, { dash: [5, 4] });
      B.line(STUDY_M, study, CSS.blue, 1.8);
      STUDY_M.forEach((m, i) => B.dot(m, study[i], 2.8, CSS.blue));
      B.dot(run.M, run.err[NS], 5.5, CSS.pink);
    });
    label(g, '1/√M', B.X(9000), B.Y(study[STUDY_M.indexOf(400)] * Math.sqrt(400 / 9000)) - 8, CSS.grey, 'left', '11px Inter, system-ui, sans-serif');
    const ex = B.X(run.M), ey = B.Y(run.err[NS]);
    label(g, 'this run', ex + (run.M > 3000 ? -2 : 9), ey + (run.M > 3000 ? -12 : -8), CSS.pink, run.M > 3000 ? 'right' : 'left', '600 11px Inter, system-ui, sans-serif');
  }

  setM();
  setT(0);
  setPlaying(true);
  v.onFrame((dt) => {
    if (playing) { setT(t + dt); if (t >= T) setPlaying(false); }
    show3d();
    draw();
  });
  show3d(); draw();

  panel._state = () => {
    const f = frameIndex();
    return { M: run.M, t: f / FPS, varRobots: run.vars[f], var2Dt: 2 * D * f / FPS, errNow: run.err[f], errAtT12: run.err[NS],
      studyM: STUDY_M, studyMedianL1: study, studySlope: slope };
  };
}
