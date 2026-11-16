// 7.1 Stability: robots steer to a steep lamp with a linear pull f(x) = -k x (k = 0.5 /s) plus noise sqrt(2D) dW, as in
// the clip s07_stability. The SDE is stable for every k; Euler-Maruyama multiplies the distance to the lamp by (1 - k dt)
// each step, so it is stable only for dt < 2/k = 4 s. Three corridors, the same robots and the same noise (every coarse
// increment is a sum of 0.1 s increments): back 0.1 s, middle 0.9 x the limit, front the slider. No walls in the
// simulation: unstable robots really leave the corridor. Plot: Higham's mean-square stability region of EM on the test
// equation dX = lambda X dt + mu X dW in the plane (dt lambda, dt mu^2), with the three steps of the robots
// (lambda = -k, mu = 0: their noise does not depend on x).
import { rng } from '../sim3.js';
import { makeViewer, placeLamp, SwarmMesh, ledColour, LightPool, COL, CSS, val, onInput, onClick, readout, seg, plot2d } from '../common3.js';
import { Overlay, robotY, legendItem, Ax, text, SMALL } from '../dens_common.js';

const K = 0.5, D = 0.003, LW = 0.25;
const LIMIT = 2 / K;                         // 4 s
const FINE = 0.1;                            // noise is drawn on this grid (s)
const T_END = 44;
const M = 10000, NDRAW = 160;
const HALF = 1.5;
const VAR0 = 2.6 ** 2 / 12;                  // variance of the uniform start on [-1.3, 1.3]
const YS = [0.9, 0.0, -0.9];                 // back, middle, front
const SQ2D = Math.sqrt(2 * D);

const trueSd = (t) => Math.sqrt(VAR0 * Math.exp(-2 * K * t) + D / K * (1 - Math.exp(-2 * K * t)));
const emSd = (dt) => (K * dt < 2 ? Math.sqrt(D / (K * (1 - K * dt / 2))) : Infinity);

export async function init(panel) {
  const v = await makeViewer(panel, 'corridor3.mjb', { lookat: [0, -0.1, 0], distance: 3.75, azimuth: 90, elevation: -50 });
  YS.forEach((y, i) => placeLamp(v, 2 - i, [0, y], 2.6));
  v.forward();
  YS.forEach((y) => new LightPool(v.scene, 0, y, 0.2).setLevel(0.8));
  const meshes = YS.map(() => new SwarmMesh(v.scene, NDRAW));
  const ys = YS.map((y, i) => robotY(NDRAW, rng(70 + i), 0.12, y));
  const ov = new Overlay(panel, v);
  const labs = YS.map((y) => ov.label([-HALF, y + 0.2, 0], 'l'));
  ov.box('legend', `robots steer to the lamp: f(x) = −k x, k = ${K} /s<br>${M} robots per corridor (${NDRAW} drawn), the same noise`);
  const P = plot2d(panel.querySelector('canvas.plot'));

  let frac = val(panel, 'frac'), dts, runs, td = 0, speed = 4, pending = false;

  /** Simulate the three corridors to T_END; keep every grid state of the drawn robots and the stats of all. */
  function simulate() {
    dts = [0.1, 0.9 * LIMIT, Math.round(frac * LIMIT / FINE) / Math.round(1 / FINE)];
    const R = rng(11), x0 = new Float64Array(M);
    for (let i = 0; i < M; i++) x0[i] = -1.3 + 2.6 * R.uni();
    runs = dts.map((dt) => ({ dt, n: Math.round(dt / FINE), x: Float64Array.from(x0), acc: new Float64Array(M), snaps: [] }));
    const snap = (r, t) => {
      let s = 0, s2 = 0, out = 0;
      for (let i = 0; i < M; i++) { const x = r.x[i]; s += x; s2 += x * x; if (Math.abs(x) > HALF) out++; }
      const m = s / M;
      r.snaps.push({ t, draw: Float64Array.from(r.x.subarray(0, NDRAW)), sd: Math.sqrt(Math.max(0, s2 / M - m * m)), out: out / M,
        medAbs: null });
    };
    runs.forEach((r) => snap(r, 0));
    const nf = Math.round(T_END / FINE), sq = Math.sqrt(FINE);
    for (let j = 1; j <= nf; j++) {
      for (let i = 0; i < M; i++) {
        const dW = sq * R.normal();
        runs[0].acc[i] += dW; runs[1].acc[i] += dW; runs[2].acc[i] += dW;
      }
      for (const r of runs) {
        if (j % r.n) continue;
        for (let i = 0; i < M; i++) { r.x[i] += -K * r.x[i] * r.dt + SQ2D * r.acc[i]; r.acc[i] = 0; }   // Euler-Maruyama
        snap(r, j * FINE);
      }
    }
    td = 0;
  }

  /** Grid state at or before t, the next one, and the fraction between them. */
  function at(r, t) {
    const k = Math.min(Math.floor(t / r.dt + 1e-9), r.snaps.length - 1), a = r.snaps[k], b = r.snaps[Math.min(k + 1, r.snaps.length - 1)];
    return { a, b, fr: b === a ? 0 : Math.min(1, (t - a.t) / r.dt) };
  }

  onInput(panel, 'frac', () => {
    frac = val(panel, 'frac');
    if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; simulate(); }); }
  });
  onClick(panel, 'restart', () => { td = 0; });
  seg(panel, 'speed', (s) => { speed = Number(s); });
  simulate();

  const disp = new Float64Array(NDRAW), hide = new Float64Array(NDRAW);
  const colours = [CSS.green, CSS.yellow, null];
  let shown = '';
  v.onFrame((dt) => {
    td = Math.min(T_END, td + dt * speed);
    runs.forEach((r, c) => {
      const { a, b, fr } = at(r, td);
      for (let i = 0; i < NDRAW; i++) { const x = a.draw[i] + (b.draw[i] - a.draw[i]) * fr; disp[i] = Math.abs(x) < 8 ? x : 1e3; hide[i] = x; }
      meshes[c].update(disp, ys[c], (i) => ledColour(Math.exp(-hide[i] * hide[i] / (2 * LW * LW)) / 0.8));
      const col = c < 2 ? colours[c] : (r.dt >= LIMIT ? CSS.red : CSS.pink);
      const head = `Δt = ${+r.dt.toFixed(1)} s (${(r.dt / LIMIT).toFixed(r.dt / LIMIT < 0.1 ? 3 : 2)} × limit)`;
      const msg = r.dt < LIMIT ? `spread <b>${(100 * a.sd).toFixed(0)} cm</b>, true ${(100 * trueSd(a.t)).toFixed(0)} cm`
        : `<b>${(100 * a.out).toFixed(0)}%</b> thrown out of the corridor`;
      labs[c].set(`<span style="color:${col}"><b>${head}</b>: ${msg}</span>`);
    });
    ov.update();
    drawPlot();
    const r = runs[2];
    const s = `t = <b>${td.toFixed(1)} s</b>${td >= T_END ? ' (end)' : ''} &nbsp; front corridor: one step multiplies the distance to the lamp by ` +
      `<b>1 − kΔt = ${(1 - K * r.dt).toFixed(2)}</b>; ` + (r.dt < LIMIT ? `settled spread of the simulation ${(100 * emSd(r.dt)).toFixed(0)} cm (true ${(100 * Math.sqrt(D / K)).toFixed(1)} cm)`
      : 'the spread grows without bound');
    if (s !== shown) { shown = s; readout(panel, 'info', s); }
  });

  function drawPlot() {
    P.clear();
    const g = P.g, w = P.w, h = P.h;
    if (w < 120 || h < 150) return;
    const L = 40, Rm = 12, top = 46, bottom = 66;
    const ax = new Ax(g, L, top, w - L - Rm, h - top - bottom, [-4.3, 0.5], [-0.25, 2.3]);
    text(g, 'Where Euler–Maruyama is stable', L - 30, 18, CSS.white, 'left', '600 12px Inter, system-ui, sans-serif');
    text(g, 'test equation dX = λX dt + μX dW; x = Δt λ, y = Δt μ²', L - 30, 33, CSS.grey, 'left', SMALL);
    ax.clip(() => {
      // the SDE is mean-square stable for lambda + mu^2/2 < 0, i.e. y < -2 x
      g.fillStyle = 'rgba(131, 193, 103, 0.16)';
      g.beginPath(); g.moveTo(ax.X(0), ax.Y(0)); g.lineTo(ax.X(-1.15), ax.Y(2.3)); g.lineTo(ax.X(-4.3), ax.Y(2.3)); g.lineTo(ax.X(-4.3), ax.Y(0)); g.closePath(); g.fill();
      // EM is mean-square stable for (1 + x)^2 + y < 1
      g.fillStyle = 'rgba(88, 196, 221, 0.32)';
      g.beginPath();
      for (let i = 0; i <= 80; i++) { const x = -2 * i / 80; const y = -x * (2 + x); i ? g.lineTo(ax.X(x), ax.Y(y)) : g.moveTo(ax.X(x), ax.Y(y)); }
      g.closePath(); g.fill();
      const xs = [], yv = [];
      for (let i = 0; i <= 80; i++) { const x = -2 * i / 80; xs.push(x); yv.push(-x * (2 + x)); }
      ax.line(xs, yv, CSS.blue, 2);
      ax.line([0, -1.15], [0, 2.3], CSS.green, 1.5);
      ax.hline(0, 'rgba(242,239,233,0.3)');
      // Higham's example lambda = -3, mu^2 = 3: limit dt = 1/3
      ax.line([0, -3 * 1.2], [0, 3 * 1.2], CSS.grey, 1, [4, 4]);
      ax.dots([-1], [1], CSS.grey, 3);
      // the robots: (-k dt, 0)
      runs.forEach((r, c) => {
        const col = c < 2 ? colours[c] : (r.dt >= LIMIT ? CSS.red : CSS.pink);
        ax.dots([-K * r.dt], [0], col, c === 2 ? 6.5 : 5);
      });
    });
    ax.frame([[-4, '−4'], [-3, '−3'], [-2, '−2'], [-1, '−1'], [0, '0']], [[0, '0'], [1, '1'], [2, '2']]);
    text(g, 'EM stable', ax.X(-1.3), ax.Y(0.42), CSS.white, 'center', SMALL);
    text(g, '(1 + x)² + y < 1', ax.X(-1.3), ax.Y(0.42) + 12, CSS.white, 'center', SMALL);
    text(g, 'SDE stable: y < −2x', ax.X(-4.2), ax.Y(2.05), CSS.green, 'left', SMALL);
    text(g, 'Higham: λ = −3, μ² = 3', ax.X(-1.3), ax.Y(1.3), CSS.grey, 'right', SMALL);
    text(g, 'limit Δt = 1/3', ax.X(-1.3), ax.Y(1.3) + 12, CSS.grey, 'right', SMALL);
    text(g, 'x = Δt λ', L + ax.w, top + ax.h + 28, CSS.grey, 'right', SMALL);
    const r = runs[2];
    text(g, `robots: λ = −k, μ = 0, so the point is (−kΔt, 0); front: ${(-K * r.dt).toFixed(2)}`, L - 30, h - 26, CSS.grey, 'left', SMALL);
    text(g, `EM leaves the blue region at x = −2, i.e. Δt = 2/k = ${LIMIT} s`, L - 30, h - 12, CSS.grey, 'left', SMALL);
  }

  panel._state = () => ({
    tDisplay: td, k: K, D, limitS: LIMIT, trueSpreadM: Math.sqrt(D / K), robots: M, speed,
    runs: runs.map((r) => {
      const last = r.snaps[r.snaps.length - 1], now = at(r, td).a;
      return { dt: r.dt, overLimit: r.dt / LIMIT, factor: 1 - K * r.dt, steps: r.snaps.length - 1, tLast: last.t,
        spreadEndM: last.sd, outsideEnd: last.out, spreadEmTheoryM: emSd(r.dt), trueSdAtEndM: trueSd(last.t),
        spreadNowM: now.sd, outsideNow: now.out };
    }),
  });
}
