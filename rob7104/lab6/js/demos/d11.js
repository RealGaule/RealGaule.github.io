// 1.1 Gradient descent and step size, on the two-joint arm (shoulder lift, elbow) of the UR5e.
// 3D: the arm runs fixed-step gradient descent (or heavy ball) from a start towards the green target; the tool
//     leaves a trace coloured by the outcome (grey: cost falls every step; yellow: overshoots but arrives;
//     red: never settles); a faint ghost marks the start; the pink ellipse = tool motions from unit joint moves.
// 2D: the cost over joint angles (zoomed on the valley, or the whole torus) with the path, the start ring and the
//     two curvature directions at the elbow-up minimum; below it the factor per step |1 - alpha lambda_i| against
//     the step size, with 2/lambda_2 (local limit) and 2/L (L over all joint angles) marked. Click it to set alpha.
// Numerics: sim6.js (TwoJointArm, gradientDescent, heavyBall, curvatureBounds). Same rules as clip s11a / s11b:
// start (-0.9, 1.3), presets alpha * lambda_2 = 0.2, 1.9, 2.25, momentum beta = (1 - sqrt(alpha mu))^2,
// "N steps to 1 mm" = first iterate after which the tool stays within 1 mm, convergence |g| <= 1e-9.
import { S, makeArmViewer, Landscape, marksFor, marker, planeToWorld, world, GhostArms, Ellipse3D, Line3D,
  COL, CSS, LAND_LINE, LAND_FILL, cmap, plot2d, val, onInput, onClick, readout, toggle, setVal, clampReach, fmt, ctl } from '../common.js';

const PI = Math.PI;
const START = [-0.9, 1.3];                          // start A of clip 1.2, also the start of clip 1.1
const MAX_ITER = 4000;                              // the smallest slider step needs about 2500 to |g| <= 1e-9
const PRESET = { small: 0.2, near: 1.9, beyond: 2.25 };   // alpha * lambda_2 at the solution, as in the clip
const ALPHA_MAX = 6;                                // slider range and x range of the factor chart
const TARGET_Z = 0.40;                              // the target slides outwards at the default target's height
const ZOOM_HALF = 1.0;                              // half width (rad) of the zoomed window
const SPACING = { zoom: 0.04, full: 0.10 };         // contour spacing in tool-to-target distance (m); clip: 4 cm
const ELLIPSE_SCALE = 0.2;                          // m of ellipse per unit singular value
const BIG_STEP = 0.15;                              // rad: steps larger than this are shown slowly (3.5 per s)
const OUTCOME = {
  grey: { css: CSS.grey, hex: COL.grey, text: 'cost falls every step' },
  yellow: { css: CSS.yellow, hex: COL.yellow, text: 'overshoots, still arrives' },
  red: { css: CSS.red, hex: COL.red, text: 'never settles' },
};

const betaRule = (a, mu) => (a * mu < 1 ? (1 - Math.sqrt(a * mu)) ** 2 : 0);
/** Per-step factor of one eigen-direction under heavy ball (spectral radius of the 2 x 2 recursion). */
function rhoHB(a, lam, b) {
  const tr = 1 + b - a * lam, disc = tr * tr - 4 * b;
  if (disc < 0) return Math.sqrt(b);
  const r = Math.sqrt(disc);
  return Math.max(Math.abs(tr + r), Math.abs(tr - r)) / 2;
}
const signed = (x) => (x < 0 ? `(−${Math.abs(x).toFixed(2)})` : x.toFixed(2));
function tickText(v) {
  const k = Math.round(v / (PI / 4));
  if (k === 0) return '0';
  const a = Math.abs(k), gcd = a % 4 === 0 ? 4 : a % 2 === 0 ? 2 : 1, num = a / gcd, den = 4 / gcd;
  return `${k < 0 ? '−' : ''}${num === 1 ? '' : num}π${den === 1 ? '' : '/' + den}`;
}

/** The common landscape with a view window: centre c and half width h (h = pi, c = 0 is the whole torus).
 *  Contours at fixed steps of the tool-to-target distance, coloured over the window as in the clip. */
class ValleyMap extends Landscape {
  constructor(canvas) {
    super(canvas, { labels: ['shoulder angle q₁', 'elbow angle q₂'] });
    this.view = { c: [0, 0], h: PI };
    this.spacing = SPACING.full;
  }
  setView(c, h, spacing) { this.view = { c: c.slice(), h }; this.spacing = spacing; this.img = null; }
  rel(q) { const c = this.view.c; return [c[0] + S.wrap(q[0] - c[0]), c[1] + S.wrap(q[1] - c[1])]; }
  toPx(q) {
    const { c, h } = this.view, b = this.box, w = this.rel(q);
    return [b.x + (w[0] - c[0] + h) / (2 * h) * b.s, b.y + (c[1] + h - w[1]) / (2 * h) * b.s];
  }
  fromPx(x, y) {
    const { c, h } = this.view, b = this.box, u = (x - b.x) / b.s, v = (y - b.y) / b.s;
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    return S.wrap([c[0] - h + 2 * h * u, c[1] + h - 2 * h * v]);
  }
  _raster() {
    const n = Math.max(32, Math.min(360, Math.round(this.box.s * this.P.r)));
    if (this.img && this.img.width === n) return this.img;
    const { c, h } = this.view, D = new Float64Array(n * n);
    let dmax = 0;
    for (let j = 0; j < n; j++) {
      const q2 = c[1] + h - (j + 0.5) * 2 * h / n;
      for (let i = 0; i < n; i++) {
        const d = Math.sqrt(2 * Math.max(0, this.f([c[0] - h + (i + 0.5) * 2 * h / n, q2])));
        D[j * n + i] = d;
        if (d > dmax) dmax = d;
      }
    }
    const K = Math.max(2, Math.ceil(dmax / this.spacing)), band = new Uint16Array(n * n);
    for (let k = 0; k < n * n; k++) band[k] = Math.min(K - 1, Math.floor(D[k] / this.spacing));
    const fill = Array.from({ length: K }, (_, k) => cmap(LAND_FILL, (k + 1) / K));
    const line = Array.from({ length: K }, (_, k) => cmap(LAND_LINE, (k + 1) / K).map((v, j) => Math.round(0.7 * v + 0.3 * fill[k][j])));
    const cv = document.createElement('canvas'); cv.width = n; cv.height = n;
    const g = cv.getContext('2d'), im = g.createImageData(n, n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const k = j * n + i, b = band[k];
      const edge = (i + 1 < n && band[k + 1] !== b) || (j + 1 < n && band[k + n] !== b);
      const col = edge ? line[Math.min(b, band[i + 1 < n ? k + 1 : k], band[j + 1 < n ? k + n : k])] : fill[b];
      im.data.set([col[0], col[1], col[2], 255], 4 * k);
    }
    g.putImageData(im, 0, 0);
    return (this.img = cv);
  }
  clip() { const g = this.P.g, b = this.box; g.beginPath(); g.rect(b.x, b.y, b.s, b.s); g.clip(); }
  draw() {
    this._layout();
    const g = this.P.g, b = this.box, { c, h } = this.view;
    g.fillStyle = '#000'; g.fillRect(0, 0, this.P.w, this.P.h);
    if (this.f) { g.imageSmoothingEnabled = false; g.drawImage(this._raster(), b.x, b.y, b.s, b.s); }
    g.strokeStyle = 'rgba(97, 97, 107, 1)'; g.lineWidth = 1; g.strokeRect(b.x + 0.5, b.y + 0.5, b.s - 1, b.s - 1);
    g.fillStyle = CSS.mute; g.font = '11px Inter, sans-serif';
    const step = h > 2 ? PI / 2 : PI / 4;
    const ticks = (m) => { const out = []; for (let k = Math.ceil((m - h) / step - 1e-9); k * step <= m + h + 1e-9; k++) out.push(k * step); return out; };
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (const v of ticks(c[0])) g.fillText(tickText(v), b.x + (v - c[0] + h) / (2 * h) * b.s, b.y + b.s + 4);
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const v of ticks(c[1])) g.fillText(tickText(v), b.x - 5, b.y + (c[1] + h - v) / (2 * h) * b.s);
    g.textAlign = 'center'; g.textBaseline = 'bottom';
    g.fillText(this.labels[0], b.x + b.s / 2, this.P.h - 1);
    g.save(); g.translate(11, b.y + b.s / 2); g.rotate(-PI / 2); g.textBaseline = 'middle'; g.fillText(this.labels[1], 0, 0); g.restore();
    g.save(); this.clip();
    for (const m of this.marks) this.mark(m.q, m.type, 6, m.color);
    g.restore();
    return g;
  }
  /** Polyline in window coordinates, broken where consecutive points are more than pi apart; clipped to the map. */
  path(qs, color = CSS.white, { width = 1.6, alpha = 1, dots = false, r = 2.2 } = {}) {
    if (!qs || !qs.length) return;
    const g = this.P.g;
    g.save(); this.clip();
    g.globalAlpha = alpha; g.strokeStyle = color; g.fillStyle = color; g.lineWidth = width; g.lineJoin = 'round';
    g.beginPath();
    let prev = null;
    for (const q of qs) {
      const w = this.rel(q), [x, y] = this.toPx(q);
      if (prev && Math.max(Math.abs(w[0] - prev[0]), Math.abs(w[1] - prev[1])) > PI) g.moveTo(x, y);
      else if (prev) g.lineTo(x, y); else g.moveTo(x, y);
      prev = w;
    }
    g.stroke();
    if (dots) for (const q of qs) { const [x, y] = this.toPx(q); g.beginPath(); g.arc(x, y, r, 0, 2 * PI); g.fill(); }
    g.restore();
  }
}

export async function init(panel) {
  const two = new S.TwoJointArm(), R = two.reachMax;
  const st = { t: S.DEFAULT_TARGET.slice(), start: START.slice(), alpha: 0, mom: false, zoom: true };
  const DEFAULT_DIST = Math.hypot(...S.DEFAULT_TARGET) / R;
  let geo = null, run = null, s = 0, q = st.start.slice(), trace = [], lastKey = '', lastHud = -1;
  let dirtyTarget = false, dirtyRun = false;

  const v = await makeArmViewer(panel, {
    camera: { lookat: [0.28, 0.19, 1.45], distance: 2.3, azimuth: -76, elevation: -8 },   // the clip's view, a bit wider
    q: two.fullQ(st.start) });
  const land = new ValleyMap(panel.querySelector('canvas.land'));
  const facCanvas = panel.querySelector('canvas.fac'), fac = plot2d(facCanvas);
  const hud = ctl(panel, 'hud');

  const target = marker(v.scene, COL.green, 0.022);
  const halo = marker(v.scene, COL.green, 0.065, 0.16);
  const ghost = new GhostArms(v.scene, 1, { color: COL.white, opacity: 0.14, radius: 0.022 });
  const ell = new Ellipse3D(v.scene, COL.pink, { opacity: 0.2 });
  const line = new Line3D(v.scene, COL.grey, 6000);

  // ---------------------------------------------------------------- per target: curvature, limits, view window
  function setTarget(t) {
    st.t = t;
    const sp = two.stationaryPoints(t), up = sp.find((p) => p.name === 'elbow up');
    const E = S.eigSym(two.hess(up.q, t));                    // at a solution H = J^T J
    const cb = two.curvatureBounds(t);                         // once per target (webcore)
    geo = { sp, up, mu: E.values[0], lmax: E.values[1],
      vflat: [E.vectors[0][0], E.vectors[1][0]], vsteep: [E.vectors[0][1], E.vectors[1][1]],
      L: Math.max(Math.abs(cb[0]), Math.abs(cb[1])), sigma: two.singularValues(up.q) };
    geo.cond = geo.lmax / geo.mu;
    geo.centre = [up.q[0] + 0.35 * S.wrap(START[0] - up.q[0]), up.q[1] + 0.35 * S.wrap(START[1] - up.q[1])];
    land.setCost((x) => two.cost(x, t), marksFor(two, t));
    applyView();
    const p = planeToWorld(two, t);
    target.position.set(...p); halo.position.set(...p);
    readout(panel, 'info', `cond <b>${geo.cond.toFixed(geo.cond < 10 ? 2 : 1)}</b> · 2/λ₂ <b>${fmt(2 / geo.lmax)}</b> · 2/L <b>${fmt(2 / geo.L)}</b>`);
  }
  function applyView() {
    if (st.zoom) land.setView(geo.centre, ZOOM_HALF, SPACING.zoom); else land.setView([0, 0], PI, SPACING.full);
    lastKey = '';
  }

  // ---------------------------------------------------------------- per run: the whole path, then shown step by step
  function solve() {
    const t = st.t, f = (x) => two.cost(x, t), g = (x) => two.grad(x, t);
    const beta = st.mom ? betaRule(st.alpha, geo.mu) : 0;
    const r = st.mom ? S.heavyBall(f, g, st.start, st.alpha, beta, { maxIter: MAX_ITER, tol: 1e-9 })
      : S.gradientDescent(f, g, st.start, st.alpha, { maxIter: MAX_ITER, tol: 1e-9 });
    const err = r.x.map((x) => { const p = two.p(x); return Math.hypot(p[0] - t[0], p[1] - t[1]); });
    let n1mm = null;
    if (r.converged) { let k = err.length - 1; while (k >= 0 && err[k] <= 1e-3) k--; n1mm = k + 1; }
    let monotone = true;
    for (let k = 1; k < r.f.length; k++) if (r.f[k] > r.f[k - 1]) { monotone = false; break; }
    let period = null;
    if (!r.converged) {
      const K = r.x.length - 1;
      for (const p of [1, 2, 4, 8, 16, 32]) {
        let ok = K >= 2 * p;
        for (let j = 0; ok && j < p; j++) {
          const a = r.x[K - j], b = r.x[K - j - p];
          if (Math.hypot(S.wrap(a[0] - b[0]), S.wrap(a[1] - b[1])) > 1e-7) ok = false;
        }
        if (ok) { period = p; break; }
      }
    }
    const idx = S.classify(r.x[r.x.length - 1], r.converged, geo.sp);
    const out = r.converged ? (monotone ? 'grey' : 'yellow') : 'red';
    run = { xs: r.x, f: r.f, err, nIter: r.nIter, converged: r.converged, status: r.status, monotone, n1mm, period,
      endsAt: idx >= 0 ? geo.sp[idx].name : null, outcome: out, beta };
    line.line.material.color.setHex(OUTCOME[out].hex);
    restart();
    drawFactors();
  }
  function restart() { s = 0; q = st.start.slice(); trace = []; lastKey = ''; lastHud = -1; }
  const at = (u) => {
    const xs = run.xs, K = xs.length - 1;
    if (u >= K) return xs[K].slice();
    const k = Math.floor(u), w = u - k;
    return [xs[k][0] + w * (xs[k + 1][0] - xs[k][0]), xs[k][1] + w * (xs[k + 1][1] - xs[k][1])];
  };

  // ---------------------------------------------------------------- the factor chart
  function factors(a) {
    if (!st.mom) return { flat: 1 - a * geo.mu, steep: 1 - a * geo.lmax };
    const b = betaRule(a, geo.mu);
    return { flat: rhoHB(a, geo.mu, b), steep: rhoHB(a, geo.lmax, b) };
  }
  const FM = { l: 40, r: 64, t: 18, b: 30 };
  const fx = (a) => FM.l + a / ALPHA_MAX * (fac.w - FM.l - FM.r);
  const fy = (y) => FM.t + (1 - y / 1.5) * (fac.h - FM.t - FM.b);
  function drawFactors() {
    fac.clear();
    const g = fac.g, x0 = fx(0), x1 = fx(ALPHA_MAX), y0 = fy(0), yTop = fy(1.5);
    g.fillStyle = 'rgba(252, 98, 85, 0.11)'; g.fillRect(x0, yTop, x1 - x0, fy(1) - yTop);
    g.strokeStyle = 'rgba(242, 239, 233, 0.35)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(x0, fy(1)); g.lineTo(x1, fy(1)); g.moveTo(x0, y0); g.lineTo(x1, y0); g.moveTo(x0, y0); g.lineTo(x0, yTop); g.stroke();
    g.font = '10.5px Inter, sans-serif'; g.fillStyle = CSS.mute;
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (let a = 0; a <= ALPHA_MAX; a++) g.fillText(String(a), fx(a), y0 + 3);
    g.fillText('step size α', (x0 + x1) / 2, y0 + 15);
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const y of [0, 0.5, 1, 1.5]) g.fillText(String(y), x0 - 4, fy(y));
    g.save(); g.translate(10, (y0 + yTop) / 2); g.rotate(-PI / 2); g.textAlign = 'center';
    g.fillText('factor per step', 0, 0); g.restore();
    g.textAlign = 'left'; g.textBaseline = 'top';

    // limits: 2/lambda_2 (local, dashed white) and 2/L over all joint angles (dotted grey)
    const lim = [[2 / geo.lmax, '2/λ₂', CSS.white, [5, 4]], [2 / geo.L, '2/L', CSS.grey, [2, 3]]];
    const close = Math.abs(fx(2 / geo.lmax) - fx(2 / geo.L)) < 34;
    for (const [a, txt, col, dash] of lim) {
      if (a > ALPHA_MAX) continue;
      g.strokeStyle = col; g.setLineDash(dash); g.beginPath(); g.moveTo(fx(a), y0); g.lineTo(fx(a), yTop); g.stroke(); g.setLineDash([]);
      g.fillStyle = col; g.textBaseline = 'bottom'; if (close) { const left = txt === '2/L'; g.textAlign = left ? 'right' : 'left'; g.fillText(txt, fx(a) + (left ? -2 : 2), yTop - 2); } else { g.textAlign = 'center'; g.fillText(txt, fx(a), yTop - 2); }
    }

    // curves: steep solid, flat dashed; with momentum the plain-gradient curves stay faint underneath
    g.save(); g.beginPath(); g.rect(x0, yTop - 1, x1 - x0, y0 - yTop + 2); g.clip();
    const curve = (fn, col, dash, alpha) => {
      g.strokeStyle = col; g.globalAlpha = alpha; g.lineWidth = 1.8; g.setLineDash(dash); g.beginPath();
      for (let i = 0; i <= 240; i++) { const a = ALPHA_MAX * i / 240, y = Math.min(1.6, Math.abs(fn(a))); i ? g.lineTo(fx(a), fy(y)) : g.moveTo(fx(a), fy(y)); }
      g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    };
    const gdFlat = (a) => 1 - a * geo.mu, gdSteep = (a) => 1 - a * geo.lmax;
    if (st.mom) {
      curve(gdSteep, CSS.white, [], 0.25); curve(gdFlat, CSS.white, [6, 4], 0.25);
      curve((a) => factors(a).steep, CSS.yellow, [], 1); curve((a) => factors(a).flat, CSS.yellow, [6, 4], 1);
    } else { curve(gdSteep, CSS.white, [], 1); curve(gdFlat, CSS.white, [6, 4], 1); }
    g.restore();
    g.textAlign = 'left'; g.textBaseline = 'top'; for (const [txt, col, y] of [['↑ grows', CSS.red, yTop + 2], ['↓ shrinks', CSS.mute, fy(1) + 3]]) { const w = g.measureText(txt).width; g.fillStyle = 'rgba(0,0,0,0.78)'; g.fillRect(x0 + 4, y - 1, w + 5, 13); g.fillStyle = col; g.fillText(txt, x0 + 6, y); }

    // the current step size
    const col = OUTCOME[run ? run.outcome : 'grey'].css, F = factors(st.alpha), a = st.alpha;
    g.strokeStyle = col; g.lineWidth = 1.5; g.beginPath(); g.moveTo(fx(a), y0); g.lineTo(fx(a), yTop); g.stroke();
    for (const y of [F.flat, F.steep]) {
      if (Math.abs(y) > 1.5) continue;
      g.fillStyle = col; g.strokeStyle = '#000'; g.beginPath(); g.arc(fx(a), fy(Math.abs(y)), 4, 0, 2 * PI); g.fill(); g.stroke();
    }
    // right-hand labels at the ends of the curves (kept apart)
    let ys = Math.min(1.5, Math.abs(factors(ALPHA_MAX).steep)), yf = Math.min(1.5, Math.abs(factors(ALPHA_MAX).flat));
    let ps = fy(ys), pf = fy(yf);
    if (Math.abs(ps - pf) < 13) { if (ps < pf) { ps -= 7; pf += 7; } else { ps += 7; pf -= 7; } }
    g.textAlign = 'left'; g.textBaseline = 'middle'; g.fillStyle = st.mom ? CSS.yellow : CSS.white;
    g.fillText('steep λ₂', x1 + 5, ps); g.fillText('flat λ₁', x1 + 5, pf);
  }

  // ---------------------------------------------------------------- the 2D map and the HUD
  function drawMap() {
    land.draw();
    const g = land.P.g, h = land.view.h, len = 0.55 * Math.min(h, ZOOM_HALF);
    // curvature directions at the elbow-up minimum: steep solid, flat dashed (white, as in the clip)
    g.save(); land.clip();
    for (const [vec, dash, txt] of [[geo.vsteep, [], 'steep'], [geo.vflat, [6, 4], 'flat']]) {
      const a = land.toPx([geo.up.q[0] - len * vec[0], geo.up.q[1] - len * vec[1]]);
      const b = land.toPx([geo.up.q[0] + len * vec[0], geo.up.q[1] + len * vec[1]]);
      g.strokeStyle = 'rgba(238, 238, 238, 0.55)'; g.lineWidth = 1.3; g.setLineDash(dash);
      g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke(); g.setLineDash([]);
      if (st.zoom) {
        const e = b[1] < a[1] ? a : b;
        g.fillStyle = 'rgba(238, 238, 238, 0.8)'; g.font = '11px Inter, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'bottom';
        g.fillText(txt, e[0], e[1] - 3);
      }
    }
    g.restore();
    const k = Math.min(Math.floor(s), run.xs.length - 1), col = OUTCOME[run.outcome].css;
    const shown = run.xs.slice(Math.max(0, k - 1500), k + 1);
    shown.push(q);
    land.path(shown, col, { width: 1.5, dots: k < 400, r: 2 });
    g.save(); land.clip();
    const sp = land.toPx(st.start);
    g.strokeStyle = CSS.white; g.lineWidth = 1.6; g.beginPath(); g.arc(sp[0], sp[1], 6, 0, 2 * PI); g.stroke();
    g.restore();
    g.save(); land.clip(); land.dot(q, CSS.white, 5); g.restore();
  }
  function drawHud(k) {
    const o = OUTCOME[run.outcome], F = factors(st.alpha);
    const fac1 = st.mom ? `flat ×${F.flat.toFixed(2)} · steep ×${F.steep.toFixed(2)} <span style="color:var(--mute)">(heavy ball, β = ${run.beta.toFixed(2)})</span>`
      : `flat ×${signed(F.flat)} · steep ×${signed(F.steep)}`;
    let end;
    if (run.converged) end = `${run.n1mm} steps to 1 mm · ${run.nIter} to |g| ≤ 1e-9`;
    else if (run.period && run.period > 1) end = `no convergence: swings between ${run.period} poses`;
    else end = `no convergence in ${run.nIter} steps`;
    const err = (() => { const p = two.p(q); return 1000 * Math.hypot(p[0] - st.t[0], p[1] - st.t[1]); })();
    hud.innerHTML = `<b style="color:${o.css}">α = ${st.alpha.toFixed(2)}</b> = ${(st.alpha * geo.lmax).toFixed(2)} / λ₂ · ${fac1}<br>`
      + `<span style="color:var(--ink)">step ${k}</span> · tool ${err < 1 ? err.toFixed(2) : err.toFixed(0)} mm from the target<br>`
      + `<span style="color:${o.css}">${o.text}</span> · ${end}`;
  }

  // ---------------------------------------------------------------- controls
  const distTarget = () => {
    const d = val(panel, 'dist');
    if (Math.abs(d - DEFAULT_DIST) < 1e-9) return S.DEFAULT_TARGET.slice();
    const r = d * R;
    return clampReach(two, [Math.sqrt(Math.max(0, r * r - TARGET_Z * TARGET_Z)), TARGET_Z], 1e-6);
  };
  onInput(panel, 'alpha', () => { st.alpha = val(panel, 'alpha'); dirtyRun = true; });
  onInput(panel, 'dist', () => { dirtyTarget = true; });
  for (const name of Object.keys(PRESET)) onClick(panel, name, () => setVal(panel, 'alpha', Math.min(ALPHA_MAX, PRESET[name] / geo.lmax)));
  toggle(panel, 'mom', (on) => { st.mom = on; dirtyRun = true; });
  toggle(panel, 'zoom', (on) => { st.zoom = on; applyView(); });
  onClick(panel, 'restart', () => restart());
  land.onPick((q0) => { st.start = q0; ghost.set(0, two.fullQ(q0)); dirtyRun = true; });
  facCanvas.addEventListener('pointerdown', (e) => {
    const rc = facCanvas.getBoundingClientRect(), x = e.clientX - rc.left;
    const a = (x - FM.l) / (fac.w - FM.l - FM.r) * ALPHA_MAX;
    if (a >= 0.1 && a <= ALPHA_MAX) setVal(panel, 'alpha', a);
  });

  setTarget(st.t);
  ghost.set(0, two.fullQ(st.start));
  st.alpha = PRESET.near / geo.lmax;
  ctl(panel, 'alpha').value = st.alpha;
  ctl(panel, 'alpha').dispatchEvent(new Event('input', { bubbles: true }));   // fills the <output>
  dirtyRun = false;
  solve();

  // ---------------------------------------------------------------- the frame loop
  v.onFrame((dt) => {
    if (dirtyTarget) { dirtyTarget = false; setTarget(distTarget()); dirtyRun = true; }
    if (dirtyRun) { dirtyRun = false; solve(); }
    const K = run.xs.length - 1, s0 = s;
    if (s < K) {
      const k = Math.floor(s), d = Math.hypot(run.xs[k + 1][0] - run.xs[k][0], run.xs[k + 1][1] - run.xs[k][1]);
      const rate = d > BIG_STEP ? 3.5 : Math.min(60, 2.5 * (1 + s / 6));
      s = Math.min(K, s + dt * rate);
    }
    q = at(s);
    // tool trace: sample the interpolated motion so the arcs stay round at any frame rate
    if (s > s0 || trace.length === 0) {
      const n = Math.min(64, Math.max(1, Math.ceil((s - s0) / 0.125)));
      for (let i = 1; i <= n; i++) trace.push(world(two.pWorld(at(s0 + (s - s0) * i / n))));
      if (trace.length > 6000) trace = trace.slice(trace.length - 6000);
      line.set(trace);
    }
    v.setArm(two.fullQ(q)); v.forward();
    const el = two.ellipse(q);
    ell.set(world(two.pWorld(q)), el.axesWorld, el.sigmas.map((x) => x * ELLIPSE_SCALE));
    const key = `${s}|${st.zoom}|${land.P.w}`;
    if (key !== lastKey) { lastKey = key; drawMap(); }
    const k = Math.floor(s);
    if (k !== lastHud) { lastHud = k; drawHud(k); }
  });
  window.addEventListener('resize', () => { drawFactors(); lastKey = ''; });

  panel._set = (o = {}) => {                       // for tests: set the start exactly (the map click is approximate)
    if (o.start) { st.start = o.start.slice(); ghost.set(0, two.fullQ(st.start)); }
    solve();
    return panel._state();
  };
  panel._state = () => {
    const p = two.p(q), F = factors(st.alpha);
    return {
      target: st.t, distOverReach: Math.hypot(...st.t) / R, start: st.start, alpha: st.alpha,
      alphaTimesLambda2: st.alpha * geo.lmax, momentum: st.mom, beta: run.beta,
      lambda: [geo.mu, geo.lmax], cond: geo.cond, sigma: geo.sigma, limitLocal: 2 / geo.lmax, L: geo.L, limitGlobal: 2 / geo.L,
      factors: [F.flat, F.steep],
      run: { nIter: run.nIter, converged: run.converged, status: run.status, monotone: run.monotone, stepsTo1mm: run.n1mm,
        endsAt: run.endsAt, period: run.period, outcome: OUTCOME[run.outcome].text, last: run.xs[run.xs.length - 1] },
      shown: { step: Math.floor(s), q, toolErrorMm: 1000 * Math.hypot(p[0] - st.t[0], p[1] - st.t[1]) },
      zoom: st.zoom, view: land.view, tracePoints: trace.length,
    };
  };
}
