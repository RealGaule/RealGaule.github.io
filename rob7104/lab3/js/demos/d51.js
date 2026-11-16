// 5.1 Strong and weak: robots that slow down in light (Part 5, as in the clip s05_strong_weak). Stratonovich SDE in Ito
// form dX = 1/2 g g'(X) dt + g(X) dW, g = v(x)/sqrt(a), v = v0 (1 - s I(x)), reflecting walls. The same Brownian increments
// drive a coarse Euler-Maruyama run (step dt, the slider) and a fine Milstein reference (1/64 s).
// Front corridor: eight robots of the coarse run (pink) next to their reference selves (yellow ghosts); the pink bar is the
// strong error. Middle corridor and wall: the swarm, coarse histogram (blue) against reference (yellow outline).
// Plot: strong and weak errors at t = 40 s against dt, precomputed by make_trust_orders.py (data/trust_orders.json).
import { rng } from '../sim3.js';
import { makeViewer, placeLamp, SwarmMesh, ledColour, LightPool, DensityWall, COL, CSS, THREE, Trail, onClick, readout, seg, plot2d } from '../common3.js';
import { Overlay, Ribbon, wallCurve, stairs, robotY, legendItem, hideGeoms, Ax, text, FONT, SMALL } from '../dens_common.js';

const V0 = 0.15, A_TURN = 1.0, S_SLOW = 0.85, LW = 0.25;
const FINE = 64, H = 1 / FINE;              // reference step (s)
const T_END = 40;
const M = 20000, NDRAW = 300, NHI = 8;
const LO = -1.47, HI = 1.47, HALF = 1.5, NB = 30;
const YA = -0.9, YB = 0.0;                   // front corridor (one robot at a time), middle corridor (the swarm)
const YW = 0.33, ZW = 0.05, HW = 0.45, VMAX = 1.25;
const SQA = Math.sqrt(A_TURN), W2 = LW * LW;

/** Drift f = 1/2 g g', noise g and its slope g' at x, with one exponential. */
function model(x) {
  const e = Math.exp(-x * x / (2 * W2));
  const g = V0 * (1 - S_SLOW * e) / SQA, dg = V0 * S_SLOW * x / W2 * e / SQA;
  return [0.5 * g * dg, g, dg];
}
const reflect = (y) => (y < LO ? 2 * LO - y : y > HI ? 2 * HI - y : y);

export async function init(panel) {
  const v = await makeViewer(panel, 'corridor3.mjb', { lookat: [0, -0.34, 0.1], distance: 3.55, azimuth: 90, elevation: -37 });
  placeLamp(v, 0, [0, YA], 1.15); placeLamp(v, 1, [0, YB], 1.15); placeLamp(v, 2, null);
  v.forward();
  hideGeoms(v, (x, y, z) => Math.abs(y - 0.9) < 0.3 && z < 0.2);          // the third corridor is not used here
  [YA, YB].forEach((y) => new LightPool(v.scene, 0, y, 0.2).setLevel(0.75));
  const swarm = new SwarmMesh(v.scene, NDRAW);
  const hiMesh = new SwarmMesh(v.scene, NHI);
  const ghost = new SwarmMesh(v.scene, NHI);
  ghost.body.material = new THREE.MeshStandardMaterial({ color: COL.yellow, transparent: true, opacity: 0.32, depthWrite: false });
  ghost.led.material.transparent = true; ghost.led.material.opacity = 0.5;
  const bars = Array.from({ length: NHI }, () => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(1, 0.008, 0.008), new THREE.MeshBasicMaterial({ color: COL.pink, toneMapped: false }));
    v.scene.add(m); return m;
  });
  const trails = Array.from({ length: NHI }, () => new Trail(v.scene, COL.yellow, 90));
  const wall = new DensityWall(v.scene, { lo: -HALF, hi: HALF, nb: NB, y0: YW, z0: ZW, height: HW, vmax: VMAX, curveColours: [] });
  const refCurve = new Ribbon(v.scene, COL.yellow, { y0: YW - 0.01, width: 0.01 });
  const ov = new Overlay(panel, v);
  ov.box('legend', `${legendItem(CSS.pink, 'robot, coarse step Δt', 'bar')} &nbsp; ${legendItem(CSS.yellow, 'its reference self (1/64 s)', 'bar')}<br>` +
    `${legendItem(CSS.blue, `coarse histogram, all ${M} robots`, 'bar')} &nbsp; ${legendItem(CSS.yellow, 'reference')}`);
  const labA = ov.label([0, YA - 0.2, 0], 'cb');
  const labB = ov.label([0, YW, ZW + HW + 0.2]);

  const yDraw = robotY(NDRAW, rng(51), 0.12, YB);
  const yHi = robotY(NHI, rng(52), 0.1, YA);
  const P = plot2d(panel.querySelector('canvas.plot'));
  let D = null;
  fetch(new URL('../../data/trust_orders.json', import.meta.url)).then((r) => r.json()).then((d) => { D = d; }).catch((e) => console.error(e));

  let dt = 2, Rk, speed = 4;
  let R, x0, xr, xc, acc, k, hi, hist, snapA, snapB, td, refHi;

  /** Stats of both runs at a coarse grid time, and the drawn positions. */
  function snapshot(t) {
    const hc = new Float64Array(NB), hr = new Float64Array(NB), w = 2 * HALF / NB;
    let err = 0, uc = 0, ur = 0;
    for (let i = 0; i < M; i++) {
      err += Math.abs(xc[i] - xr[i]);
      if (Math.abs(xc[i]) < LW) uc++;
      if (Math.abs(xr[i]) < LW) ur++;
      const a = Math.floor((xc[i] + HALF) / w), b = Math.floor((xr[i] + HALF) / w);
      if (a >= 0 && a < NB) hc[a]++; if (b >= 0 && b < NB) hr[b]++;
    }
    let l1 = 0;
    for (let j = 0; j < NB; j++) { hc[j] /= M * w; hr[j] /= M * w; l1 += Math.abs(hc[j] - hr[j]) * w; }
    return { t, hc, hr, errCm: 100 * err / M, underCoarse: uc / M, underRef: ur / M, histL1: l1,
      draw: Float64Array.from(xc.subarray(0, NDRAW)), hi: hi.map((i) => xc[i]), hiErr: hi.map((i) => Math.abs(xc[i] - xr[i])) };
  }

  function restart() {
    Rk = Math.round(dt * FINE);
    R = rng(5);
    x0 = new Float64Array(M);
    for (let i = 0; i < M; i++) x0[i] = -1.2 + 2.4 * R.uni();
    xr = Float64Array.from(x0); xc = Float64Array.from(x0); acc = new Float64Array(M);
    // highlighted robots: start positions spread along the corridor (chosen before any error is seen)
    hi = [];
    for (let q = 0; q < NHI; q++) {
      const target = -1.05 + 2.1 * q / (NHI - 1);
      let best = 0;
      for (let i = 0; i < NDRAW; i++) if (!hi.includes(i) && Math.abs(x0[i] - target) < Math.abs(x0[best] - target)) best = i;
      hi.push(best);
    }
    k = 0; td = 0;
    refHi = [hi.map((i) => xr[i])];                // reference positions of the highlighted robots, every fine step
    snapA = snapshot(0); snapB = null;
    trails.forEach((tr) => tr.clear());
    advanceTo(dt);
  }

  /** Run the fine reference (and the coarse run at every Rk-th fine step) until the coarse grid passes time T. */
  function advanceTo(T) {
    const s = Math.sqrt(H);
    while (k * H < T - 1e-9 && k < T_END * FINE) {
      for (let i = 0; i < M; i++) {
        const dW = s * R.normal(), x = xr[i];
        const [f, g, dg] = model(x);
        xr[i] = reflect(x + f * H + g * dW + 0.5 * g * dg * (dW * dW - H));     // Milstein
        acc[i] += dW;
      }
      k++;
      refHi.push(hi.map((i) => xr[i]));
      if (k % Rk === 0) {
        for (let i = 0; i < M; i++) {
          const [f, g] = model(xc[i]);
          xc[i] = reflect(xc[i] + f * dt + g * acc[i]);                          // Euler-Maruyama, the same noise
          acc[i] = 0;
        }
        if (snapB) snapA = snapB;
        snapB = snapshot(k * H);
      }
    }
  }

  seg(panel, 'dt', (s) => { dt = Number(s); restart(); });
  onClick(panel, 'restart', restart);
  seg(panel, 'speed', (s) => { speed = Number(s); });
  restart();

  const dispX = new Float64Array(NDRAW), hiX = new Float64Array(NHI), ghX = new Float64Array(NHI);
  const lampLed = (x) => ledColour(Math.exp(-x * x / (2 * W2)));
  const pinkLed = new THREE.Color(COL.pink), yellowLed = new THREE.Color(COL.yellow);
  const pct = (q) => `${(100 * q).toFixed(1)}%`;
  let shown = '';

  v.onFrame((fdt) => {
    td = Math.min(T_END, td + fdt * speed);
    // the coarse run must have reached the grid time after td; snapA is the last grid state at or before td
    while (snapB && snapB.t < td - 1e-9 && k < T_END * FINE) advanceTo(snapB.t + dt);
    if (snapB && snapB.t <= td + 1e-9) { snapA = snapB; }
    const A = snapA, B = snapB && snapB.t > A.t ? snapB : A;
    const fr = B === A ? 0 : (td - A.t) / (B.t - A.t);
    for (let i = 0; i < NDRAW; i++) dispX[i] = A.draw[i] + (B.draw[i] - A.draw[i]) * fr;
    const kf = Math.min(refHi.length - 1, Math.round(td * FINE));
    for (let q = 0; q < NHI; q++) {
      hiX[q] = A.hi[q] + (B.hi[q] - A.hi[q]) * fr;
      ghX[q] = refHi[kf][q];
      const a = Math.min(hiX[q], ghX[q]), b = Math.max(hiX[q], ghX[q]);
      bars[q].scale.x = Math.max(1e-3, b - a);
      bars[q].position.set((a + b) / 2, yHi[q], 0.05);
      trails[q].push([ghX[q], yHi[q], 0.006]);
    }
    swarm.update(dispX, yDraw, (i) => lampLed(dispX[i]));
    hiMesh.update(hiX, yHi, pinkLed);
    ghost.update(ghX, yHi, yellowLed);
    wall.update(A.hc);
    const st = stairs(-HALF, HALF, A.hr);
    wallCurve(wall, refCurve, st.xs, st.ys);
    const tg = A.t.toFixed(A.t % 1 ? 2 : 0);
    labA.set(`<div style="text-align:center;color:${CSS.pink}"><b>track one robot (strong)</b><br>mean error per robot: <b>${A.errCm.toFixed(1)} cm</b></div>`);
    labB.set(`<div style="text-align:center"><b style="color:${CSS.blue}">predict the swarm (weak)</b><br>` +
      `<span style="color:${CSS.blue}">under the lamp: coarse <b>${pct(A.underCoarse)}</b></span>` +
      ` &nbsp;<span style="color:${CSS.yellow}">reference <b>${pct(A.underRef)}</b></span></div>`);
    ov.update();
    draw(A);
    const s = `t = <b>${td.toFixed(1)} s</b>${td >= T_END ? ' (end)' : ''} &nbsp; at the last coarse step, t = ${tg} s: mean error per robot <b class="pk">${A.errCm.toFixed(1)} cm</b> &nbsp;` +
      ` under the lamp: coarse <b class="bl">${pct(A.underCoarse)}</b> reference <b class="ye">${pct(A.underRef)}</b>` +
      ` &nbsp; histogram difference (L1) <b>${A.histL1.toFixed(3)}</b>`;
    if (s !== shown) { shown = s; readout(panel, 'info', s); }
  });

  // ---------------------------------------------------------------- the 2-D plot
  function draw(A) {
    P.clear();
    const g = P.g, w = P.w, h = P.h;
    if (!D || w < 120 || h < 150) return;
    const L = 46, Rm = 14, top = 92, bottom = 74;
    const ax = new Ax(g, L, top, w - L - Rm, h - top - bottom, [1 / 22, 5.6], [0.015, 30], { xlog: true, ylog: true });
    text(g, 'Errors at t = 40 s against the step', L - 36, 18, CSS.white, 'left', '600 12px Inter, system-ui, sans-serif');
    text(g, 'same noise, 20 000 robots; weak error without sampling noise', L - 36, 33, CSS.grey, 'left', SMALL);
    const dts = D.dts, sE = D.strong.mean_err_cm, wE = D.weak.err_pp;
    ax.clip(() => {
      const d0 = dts[0], d1 = dts[dts.length - 1];
      ax.line([d0, d1], [sE[0] * 0.55, sE[0] * 0.55 * Math.sqrt(d1 / d0)], CSS.grey, 1.2, [5, 4]);
      ax.line([d0, d1], [wE[0] * 0.28, wE[0] * 0.28 * (d1 / d0)], CSS.grey, 1.2, [5, 4]);
      ax.vline(dt, 'rgba(242,239,233,0.55)', [], 1);
      ax.line(dts, sE, CSS.pink, 1.6); ax.dots(dts, sE, CSS.pink, 3.2);
      ax.line(dts, wE, CSS.blue, 1.6); ax.dots(dts, wE, CSS.blue, 3.2);
      const j = dts.findIndex((d) => Math.abs(d - dt) < 1e-9);
      if (j >= 0) { ax.dots([dt], [sE[j]], CSS.pink, 6); ax.dots([dt], [wE[j]], CSS.blue, 6); }
      if (A.t >= T_END - 1e-9) {                                     // this run, measured live
        ax.dots([dt], [A.errCm], CSS.white, 8, true);
        ax.dots([dt], [Math.max(0.016, 100 * Math.abs(A.underCoarse - A.underRef))], CSS.white, 8, true);
      }
    });
    ax.frame([[1 / 16, '1/16'], [1 / 8, '1/8'], [1 / 4, '1/4'], [1 / 2, '1/2'], [1, '1'], [2, '2'], [4, '4']],
      [[0.1, '0.1'], [1, '1'], [10, '10']]);
    text(g, 'step Δt (s)', L + ax.w, top + ax.h + 28, CSS.grey, 'right', SMALL);
    text(g, 'slope ½', ax.X(0.09), ax.Y(sE[0] * 0.55 * Math.sqrt(0.09 / dts[0])) + 13, CSS.grey, 'left', SMALL);
    text(g, 'slope 1', ax.X(0.16), ax.Y(wE[0] * 0.28 * (0.16 / dts[0])) + 13, CSS.grey, 'left', SMALL);
    const xr = L + ax.w - 4, yb = top + ax.h, x0 = L - 36;
    g.fillStyle = CSS.pink; g.beginPath(); g.arc(x0 + 4, 48, 3.5, 0, 2 * Math.PI); g.fill();
    text(g, `strong: mean error per robot (cm), order ${D.strong.order_small_steps.toFixed(2)}`, x0 + 12, 52, CSS.pink, 'left', SMALL);
    g.fillStyle = CSS.blue; g.beginPath(); g.arc(x0 + 4, 63, 3.5, 0, 2 * Math.PI); g.fill();
    text(g, 'weak: error of the share under the lamp', x0 + 12, 67, CSS.blue, 'left', SMALL);
    text(g, `(percentage points), order ${D.weak.order_small_steps.toFixed(2)}`, x0 + 12, 80, CSS.blue, 'left', SMALL);
    text(g, A.t >= T_END - 1e-9 ? 'rings: this run, measured' : 'rings appear at t = 40 s', xr, yb - 8, CSS.white, 'right', SMALL);
    text(g, `Higham's test equation: orders ${D.higham.strong_order.toFixed(2)} and ${D.higham.weak_order.toFixed(2)}`, L - 36, h - 26, CSS.grey, 'left', SMALL);
    text(g, 'orders fitted for Δt ≤ 1/2 s', L - 36, h - 12, CSS.grey, 'left', SMALL);
  }

  panel._state = () => {
    const A = snapA;
    return { tDisplay: td, tGrid: A.t, dt, refSteps: k, robots: M, speed,
      meanErrCm: A.errCm, underCoarse: A.underCoarse, underRef: A.underRef, histL1: A.histL1,
      highlightedErrCm: A.hiErr.map((e) => 100 * e),
      table: D ? { dts: D.dts, strongCm: D.strong.mean_err_cm, weakPp: D.weak.err_pp, shareFp: D.weak.share_fp } : null };
  };
  // for tests: run to the end at once
  panel._finish = () => { advanceTo(T_END); td = T_END; snapA = snapB; return panel._state(); };
}
