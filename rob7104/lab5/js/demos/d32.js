// 3.2 UKF: sigma points travel for the distribution.
// Driving blind, straight 1 m in 1 s, wheel noise DT. A cloud of 4000 carts (sim5 simulateCart, Euler-Maruyama,
// dt = 1 ms) against the EKF and the UKF (sim5 EKF / UKF, 40 prediction steps of 25 ms, start sd 1e-6, as in
// out/s32_ukf_facts.json). The UKF's 11 sigma points after each step are drawn as small orange carts.
// Right: why the means differ (the weighted average of cos(theta) over the sigma points against the cloud and
// against cos of the mean), and the along-track spread (cloud histogram against the two Gaussians).
import {
  S, THREE, COL, CSS, makeViewer, setCart, parkBody, loadGhost, CartCloud, FloorEllipse, FloorArrow, Ribbon, RangeRing,
  ctl, val, onInput, onClick, seg, toggle, readout, plot2d, scales, fmt,
} from '../common.js';

const N = 4000;            // carts in the cloud
const NSTEP = 40;          // filter steps
const H = 1 / NSTEP;       // filter step (s)
const SUB = 25;            // simulation steps of 1 ms per filter step
const V = 1;               // commanded speed (m/s): 1 m in T = 1 s
const Y0 = 0.5;            // the drive runs along y = Y0 (inside the painted work area of beacons.mjb)
const NSIG = Math.sqrt(S.CHI2_95_2);   // 95 % ellipse in (x, y)
const SIGMA_SCALE = 0.55;  // sigma-point carts are small carts
const PLAY_SPEED = 0.3;    // simulated seconds per wall-clock second while driving

export async function init(panel) {
  const v = await makeViewer(panel, 'beacons.mjb', { lookat: [0.55, Y0 + 0.05, 0.02], distance: 2.55, azimuth: 32, elevation: -38 });
  const geo = await loadGhost();
  parkBody(v, 'b0'); parkBody(v, 'b1');
  v.setMocap('shelf', [0.55, 2.35, 0], undefined, [1, 0, 0, 0]);   // the shelf stands behind the drive as a backdrop

  /* ---------- static floor marks: dashed commanded track, start and end bars ---------- */
  for (let x = 0.04; x < 0.98; x += 0.1) {
    const r = new Ribbon(v.scene, COL.white, { n: 2, closed: false, opacity: 0.55 });
    r.set([[x, Y0], [Math.min(x + 0.06, 1), Y0]], 0.012);
  }
  [0, 1].forEach((x) => { const r = new Ribbon(v.scene, COL.white, { n: 2, closed: false, opacity: 0.8 }); r.set([[x, Y0 - 0.13], [x, Y0 + 0.13]], 0.016); });

  /* ---------- dynamic objects ---------- */
  const cloud = new CartCloud(v.scene, geo, { max: N, bodies: 12, color: COL.cloud, opacity: 0.28, dot: 0.022, dotZ: 0.012 });
  const red = new CartCloud(v.scene, geo, { max: N, bodies: 0, color: COL.fail, dot: 0.026, dotZ: 0.016 });
  red.points.renderOrder = 3;
  const meanRing = new RangeRing(v.scene, COL.cloud, { n: 48, opacity: 0.95 });

  // white down-arrow above the true cart; pins at the EKF mean (yellow) and the cloud's mean (blue)
  const truthMark = new THREE.Group();
  const wm = new THREE.MeshBasicMaterial({ color: COL.truth });
  const cone = new THREE.Mesh(new THREE.ConeGeometry(0.035, 0.08, 20), wm); cone.rotation.x = -Math.PI / 2; cone.position.z = 0.23;
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.011, 0.11, 12), wm); shaft.rotation.x = Math.PI / 2; shaft.position.z = 0.32;
  truthMark.add(cone, shaft); v.scene.add(truthMark);
  const pin = (color, h) => {
    const g = new THREE.Group(), m = new THREE.MeshBasicMaterial({ color });
    const s = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, h, 8), m); s.rotation.x = Math.PI / 2; s.position.z = h / 2;
    const b = new THREE.Mesh(new THREE.SphereGeometry(0.018, 16, 12), m); b.position.z = h;
    g.add(s, b); g.visible = false; v.scene.add(g); return g;
  };
  const pinE = pin(COL.ekf, 0.46), pinU = pin(COL.ukf, 0.46), pinC = pin(COL.cloud, 0.36);
  const ellE = new FloorEllipse(v.scene, COL.ekf, { nsig: NSIG, width: 0.014 });
  const ellU = new FloorEllipse(v.scene, COL.ukf, { nsig: NSIG, width: 0.014 });
  const sigMat = new THREE.MeshStandardMaterial({ color: COL.ukf, roughness: 0.5, metalness: 0.05, emissive: COL.ukf, emissiveIntensity: 0.18, transparent: true, opacity: 0.95 });
  const sigInst = new THREE.InstancedMesh(geo, sigMat, 11);
  sigInst.frustumCulled = false;
  v.scene.add(sigInst);
  const sigArrows = [], sigLinks = [];
  for (let s = 0; s < 11; s++) {
    sigArrows.push(new FloorArrow(v.scene, COL.ukf, { width: 0.009, head: 0.035, opacity: 0.85 }));
    sigLinks.push(new Ribbon(v.scene, COL.ukf, { n: 2, closed: false, opacity: 0.6, z: 0.005 }));
  }
  const M4 = new THREE.Matrix4(), SC = new THREE.Vector3(SIGMA_SCALE, SIGMA_SCALE, SIGMA_SCALE);

  /* ---------- the computation for one DT and seed: cloud snapshots and both filters ---------- */
  let D = val(panel, 'DT'), seed = 32, run = null;
  function compute() {
    const t0 = performance.now();
    const snaps = [], wheels = [];
    const x0 = [0, Y0, 0];
    const wheel = new Float64Array(2 * N);
    const start = new Float64Array(3 * N);
    for (let i = 0; i < N; i++) start[3 * i + 1] = Y0;
    snaps.push(start); wheels.push([0, 0]);
    S.simulateCart(N, V, 0, D, 1, {
      dt: 1e-3, seed, x0, wheel,
      onStep: (k, pose) => { if ((k + 1) % SUB === 0) { snaps.push(Float64Array.from(pose)); wheels.push([wheel[0], wheel[N]]); } },
    });
    const P0 = S.diag([1e-12, 1e-12, 1e-12]);           // start sd 1e-6, as nlfilters.dead_reckoning
    const ekf = new S.EKF(x0, P0, D), ukf = new S.UKF(x0, P0, D);
    const E = [{ m: x0.slice(), P: S.copyMat(P0) }], U = [{ m: x0.slice(), P: S.copyMat(P0), pts: null }];
    for (let k = 1; k <= NSTEP; k++) {
      ekf.predict(V, 0, H); ukf.predict(V, 0, H);
      E.push({ m: ekf.m.slice(), P: S.copyMat(ekf.P) });
      U.push({ m: ukf.m.slice(), P: S.copyMat(ukf.P), pts: ukf.points });
    }
    U[0].pts = { X: U[1].pts.X, Y: U[1].pts.X.map((x) => x.slice(0, 3)), Wm: U[1].pts.Wm, Wc: U[1].pts.Wc };   // before the first step
    run = { D, seed, snaps, wheels, E, U, stats: new Array(NSTEP + 1).fill(null), ms: performance.now() - t0 };
    dirty = true;
  }

  /** Statistics of step k (cached): cloud mean and covariance, coverage of both 95 % ellipses, carts outside. */
  function stats(k) {
    if (run.stats[k]) return run.stats[k];
    const pose = run.snaps[k], mc = S.meanCov(pose, 3);
    let cosSum = 0;
    for (let i = 0; i < N; i++) cosSum += Math.cos(pose[3 * i + 2]);
    const outside = (P, m) => {
      const det = P[0][0] * P[1][1] - P[0][1] * P[1][0];
      const idx = [];
      if (!(det > 0)) return idx;
      const a = P[1][1] / det, b = -P[0][1] / det, c = P[0][0] / det;
      for (let i = 0; i < N; i++) {
        const ex = pose[3 * i] - m[0], ey = pose[3 * i + 1] - m[1];
        if (a * ex * ex + 2 * b * ex * ey + c * ey * ey > S.CHI2_95_2) idx.push(i);
      }
      return idx;
    };
    const st = {
      mean: mc.mean, cov: mc.cov, cosMean: cosSum / N,
      covE: S.ellipseCoverage(pose, run.E[k].m, run.E[k].P), covU: S.ellipseCoverage(pose, run.U[k].m, run.U[k].P),
      outE: outside(run.E[k].P, run.E[k].m), outU: outside(run.U[k].P, run.U[k].m),
    };
    run.stats[k] = st;
    return st;
  }

  /* ---------- controls ---------- */
  let tau = NSTEP, playing = false, dirty = true, syncing = false;
  const filt = seg(panel, 'filter', () => { dirty = true; panel.querySelector('[data-ctl="sigma"]').disabled = filt() === 'ekf'; });
  const showSigma = toggle(panel, 'sigma', () => { dirty = true; });
  const showRed = toggle(panel, 'red', () => { dirty = true; });
  const tSlider = ctl(panel, 't');
  function syncSlider() {
    syncing = true;
    tSlider.value = (Math.floor(tau + 1e-9) * H).toFixed(3);
    tSlider.dispatchEvent(new Event('input', { bubbles: true }));
    syncing = false;
  }
  onInput(panel, 't', () => { if (syncing) return; playing = false; tau = Math.round(val(panel, 't') / H); dirty = true; });
  let pend = null;
  onInput(panel, 'DT', () => {
    clearTimeout(pend);
    pend = setTimeout(() => { D = val(panel, 'DT'); compute(); }, 120);
  });
  onClick(panel, 'go', () => { if (tau >= NSTEP - 1e-9) tau = 0; playing = true; dirty = true; });
  onClick(panel, 'step', () => {
    playing = false;
    const k = Math.floor(tau + 1e-9);
    tau = k >= NSTEP ? 1 : k + 1;
    syncSlider(); dirty = true;
  });
  onClick(panel, 'reseed', () => { seed += 1; compute(); });

  compute();

  /* ---------- 2D picture ---------- */
  const plot = plot2d(panel.querySelector('canvas.plot'));
  let lastSize = '';

  function frac12(w) {
    const n = Math.round(w * 12);
    if (Math.abs(w * 12 - n) > 1e-6) return fmt(w, 3);
    const g = (a, b) => (b ? g(b, a % b) : a), d = g(n, 12);
    return n / d === 12 / d ? '1' : `${n / d}/${12 / d}`;
  }

  function drawPlot(k) {
    plot.clear();
    const o = plot, g = o.g, st = stats(k), U = run.U[k], E = run.E[k], pts = U.pts;
    const W = pts.Wm, th = pts.Y.map((y) => y[2]);
    const cosU = th.reduce((a, t, s) => a + W[s] * Math.cos(t), 0), cosE = Math.cos(E.m[2]);
    const t = k * H, exact = Math.exp(-S.headingVariance(run.D, t) / 2);
    const split = Math.round(o.h * 0.62);

    // header: all text of the top panel lives here (word-wrapped), so nothing is written over the plot's lines
    g.textBaseline = 'top'; g.textAlign = 'left';
    const HX0 = 12, HX1 = o.w - 12;
    let tx = HX0, ty = 8, lh = 16;
    const F1 = '12px Inter, sans-serif', F2 = '11px Inter, sans-serif';
    // a phrase is a list of [text, colour] drawn on one line; it moves to the next line as a whole when it does not fit
    const phrase = (parts, font, gap = '   ') => {
      g.font = font;
      const w = g.measureText(parts.map((p) => p[0]).join('')).width, wg = g.measureText(gap).width;
      if (tx > HX0 && tx + wg + w > HX1) { tx = HX0; ty += lh; } else if (tx > HX0) tx += wg;
      parts.forEach(([t, c]) => { g.fillStyle = c; g.fillText(t, tx, ty); tx += g.measureText(t).width; });
    };
    const nl = (h) => { tx = HX0; ty += lh; lh = h; };
    phrase([['mean progress E[cos θ]:', CSS.grey]], F1, '');
    phrase([[`cloud ${fmt(st.cosMean, 3)}`, CSS.cloud]], F1, '  ');
    phrase([[`UKF ${fmt(cosU, 3)}`, CSS.ukf]], F1, '  ');
    phrase([[`EKF ${fmt(cosE, 3)}`, CSS.ekf]], F1, '  ');
    nl(14);
    phrase([['UKF', CSS.ukf], [' = weighted average over the sigma points', CSS.grey]], F2);
    phrase([['EKF', CSS.ekf], [' = cos of the mean', CSS.grey]], F2);
    nl(14);
    phrase([['dots: sigma points, area ∝ weight', CSS.grey]], F2);
    phrase([[`(centre ${frac12(W[0])}, others ${frac12(W[1])})`, CSS.grey]], F2, ' ');
    phrase([['blue: headings in the cloud', CSS.cloud]], F2);
    const headH = ty + lh + 10;

    // top: forward progress cos(theta) against heading
    const top = { g, w: o.w, h: split };
    const s = scales(top, [-Math.PI / 2, Math.PI / 2], [0, 1.12], [46, 14, headH, 36]);
    const [l, tp, Wd, Hd] = s.box;
    g.save();
    g.strokeStyle = CSS.faint; g.lineWidth = 1;
    g.beginPath(); g.moveTo(l, tp); g.lineTo(l, tp + Hd); g.lineTo(l + Wd, tp + Hd); g.stroke();
    g.fillStyle = CSS.grey; g.font = '11px Inter, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'top';
    [[-Math.PI / 2, '−π/2'], [-Math.PI / 4, '−π/4'], [0, '0'], [Math.PI / 4, 'π/4'], [Math.PI / 2, 'π/2']].forEach(([a, lab]) => g.fillText(lab, s.X(a), tp + Hd + 5));
    g.textAlign = 'right'; g.textBaseline = 'middle';
    [0, 0.5, 1].forEach((y) => { g.fillText(String(y), l - 6, s.Y(y)); g.strokeStyle = 'rgba(255,255,255,0.06)'; g.beginPath(); g.moveTo(l, s.Y(y)); g.lineTo(l + Wd, s.Y(y)); g.stroke(); });
    g.fillStyle = CSS.white; g.font = '12px Inter, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'top';
    g.fillText('heading θ (rad)', l + Wd / 2, tp + Hd + 19);
    g.save(); g.translate(13, tp + Hd / 2); g.rotate(-Math.PI / 2); g.textBaseline = 'middle'; g.fillText('forward progress  cos θ', 0, 0); g.restore();

    // headings of the cloud: histogram, scaled into the lower part
    const nb = 60, hist = new Float64Array(nb), a0 = -Math.PI / 2, bw = Math.PI / nb, pose = run.snaps[k];
    for (let i = 0; i < N; i++) { const b = Math.floor((pose[3 * i + 2] - a0) / bw); if (b >= 0 && b < nb) hist[b]++; }
    const hmax = Math.max(...hist, 1), hs = 0.36 / hmax;
    g.beginPath(); g.moveTo(s.X(a0), s.Y(0));
    for (let b = 0; b < nb; b++) { g.lineTo(s.X(a0 + b * bw), s.Y(hist[b] * hs)); g.lineTo(s.X(a0 + (b + 1) * bw), s.Y(hist[b] * hs)); }
    g.lineTo(s.X(Math.PI / 2), s.Y(0)); g.closePath();
    g.fillStyle = 'rgba(88,196,221,0.28)'; g.fill(); g.strokeStyle = CSS.cloud; g.lineWidth = 1.2; g.stroke();

    // sigma points: heading on the axis, image on the curve, dashed line to the axis of progress
    const order = th.map((a, i) => i).sort((i, j) => th[i] - th[j]);
    const groups = [];
    order.forEach((i) => { const gq = groups[groups.length - 1]; if (gq && Math.abs(th[i] - gq.th) < 0.004) { gq.w += W[i]; gq.n++; } else groups.push({ th: th[i], w: W[i], n: 1 }); });

    // cos curve; its label goes right of the curve at a heading where no sigma-point line or dot comes near
    g.beginPath();
    for (let i = 0; i <= 120; i++) { const a = -Math.PI / 2 + Math.PI * i / 120; const px = s.X(a), py = s.Y(Math.cos(a)); if (i) g.lineTo(px, py); else g.moveTo(px, py); }
    g.strokeStyle = 'rgba(242,239,233,0.85)'; g.lineWidth = 2; g.stroke();
    g.fillStyle = CSS.white; g.textAlign = 'left'; g.textBaseline = 'middle'; g.font = '12px Inter, sans-serif';
    const cw = g.measureText('cos θ').width;
    const clear = (a) => groups.every((q) => {
      const x0 = s.X(a) + 4, y0 = s.Y(Math.cos(a)), qx = s.X(q.th), qy = s.Y(Math.cos(q.th));
      return qx < x0 - 8 || qy > y0 + 13 || (qy < y0 - 13 && qx > x0 + cw + 8);
    });
    const aL = [1.12, 0.95, 1.3, 0.75, 0.55].find(clear);
    if (aL !== undefined) g.fillText('cos θ', s.X(aL) + 4, s.Y(Math.cos(aL)));
    g.setLineDash([4, 3]); g.strokeStyle = 'rgba(240,164,93,0.75)'; g.lineWidth = 1;
    groups.forEach((q) => { const c = Math.cos(q.th); g.beginPath(); g.moveTo(l, s.Y(c)); g.lineTo(s.X(q.th), s.Y(c)); g.stroke(); });
    g.setLineDash([]); g.strokeStyle = CSS.ukf; g.lineWidth = 1.4;
    groups.forEach((q) => {
      const c = Math.cos(q.th), px = s.X(Math.max(-Math.PI / 2, Math.min(Math.PI / 2, q.th)));
      g.beginPath(); g.moveTo(px, s.Y(0)); g.lineTo(px, s.Y(c)); g.stroke();
      const r = 11 * Math.sqrt(Math.max(q.w, 0));   // dot area proportional to the (merged) weight
      g.fillStyle = CSS.ukf;
      [0, c].forEach((y) => { g.beginPath(); g.arc(px, s.Y(y), r, 0, 2 * Math.PI); g.fill(); });
    });
    // the three means on the progress axis: coloured triangle and short line, no text (the header names them);
    // lines of decreasing length drawn longest first, so all three colours stay visible when the means coincide
    const mark = (y, col, len) => {
      const py = s.Y(y);
      g.fillStyle = col; g.beginPath(); g.moveTo(l - 1, py); g.lineTo(l - 9, py - 5); g.lineTo(l - 9, py + 5); g.closePath(); g.fill();
      g.strokeStyle = col; g.lineWidth = 2.5; g.beginPath(); g.moveTo(l, py); g.lineTo(l + len, py); g.stroke();
    };
    mark(st.cosMean, CSS.cloud, 30); mark(cosU, CSS.ukf, 20); mark(cosE, CSS.ekf, 10);
    g.restore();

    // bottom: along the track, x - v t
    const bot = { g, w: o.w, h: o.h - split };
    g.save(); g.translate(0, split);
    const xr = [-0.5, 0.22];
    const sb = scales(bot, xr, [0, 1], [46, 14, 26, 42]);
    const [lb, tb, Wb, Hb] = sb.box;
    g.strokeStyle = CSS.faint; g.lineWidth = 1;
    g.beginPath(); g.moveTo(lb, tb + Hb); g.lineTo(lb + Wb, tb + Hb); g.stroke();
    g.fillStyle = CSS.white; g.font = '12px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'top';
    g.fillText('along the track:  x − commanded  (m)', lb, 4);
    g.fillStyle = CSS.grey; g.font = '11px Inter, sans-serif'; g.textAlign = 'center';
    [-0.4, -0.2, 0, 0.2].forEach((xv) => { g.fillText(xv === 0 ? '0' : xv.toFixed(1).replace('-', '−'), sb.X(xv), tb + Hb + 5); g.beginPath(); g.moveTo(sb.X(xv), tb + Hb); g.lineTo(sb.X(xv), tb + Hb + 3); g.stroke(); });
    const cmd = V * t;
    const nbx = 72, hx = new Float64Array(nbx), bx = (xr[1] - xr[0]) / nbx;
    for (let i = 0; i < N; i++) { const b = Math.floor((pose[3 * i] - cmd - xr[0]) / bx); if (b >= 0 && b < nbx) hx[b]++; }
    const sdE = Math.sqrt(Math.max(E.P[0][0], 1e-14)), sdU = Math.sqrt(Math.max(U.P[0][0], 1e-14)), sdC = Math.sqrt(st.cov[0][0]);
    const dens = Array.from(hx, (c) => c / (N * bx)), dmax = Math.max(...dens, 1e-9), ky = 0.86 / Math.max(dmax, 1 / (Math.max(sdE, 0.004) * Math.sqrt(2 * Math.PI)), 1 / (Math.max(sdU, 0.004) * Math.sqrt(2 * Math.PI)));
    g.save(); g.beginPath(); g.rect(lb, tb, Wb, Hb + 1); g.clip();
    g.beginPath(); g.moveTo(sb.X(xr[0]), sb.Y(0));
    for (let b = 0; b < nbx; b++) { g.lineTo(sb.X(xr[0] + b * bx), sb.Y(dens[b] * ky)); g.lineTo(sb.X(xr[0] + (b + 1) * bx), sb.Y(dens[b] * ky)); }
    g.lineTo(sb.X(xr[1]), sb.Y(0)); g.closePath();
    g.fillStyle = 'rgba(88,196,221,0.28)'; g.fill(); g.strokeStyle = CSS.cloud; g.lineWidth = 1.2; g.stroke();
    const gauss = (m, sd, col) => {
      g.beginPath();
      for (let i = 0; i <= 240; i++) {
        const xv = xr[0] + (xr[1] - xr[0]) * i / 240, z = (xv - m) / sd;
        const y = Math.exp(-0.5 * z * z) / (sd * Math.sqrt(2 * Math.PI)) * ky;
        const px = sb.X(xv), py = sb.Y(Math.min(y, 1.2));
        if (i) g.lineTo(px, py); else g.moveTo(px, py);
      }
      g.strokeStyle = col; g.lineWidth = 2; g.stroke();
    };
    gauss(E.m[0] - cmd, sdE, CSS.ekf);
    gauss(U.m[0] - cmd, sdU, CSS.ukf);
    g.restore();
    const vline = (xv, col, dash) => { g.save(); g.setLineDash(dash); g.strokeStyle = col; g.lineWidth = 1.5; g.beginPath(); g.moveTo(sb.X(xv), tb); g.lineTo(sb.X(xv), tb + Hb); g.stroke(); g.restore(); };
    vline(st.mean[0] - cmd, CSS.cloud, [3, 3]);
    vline(U.m[0] - cmd, CSS.ukf, [6, 4]);
    vline(E.m[0] - cmd, CSS.ekf, [2, 5]);
    g.font = '11.5px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'top';
    let xx = lb;
    const put2 = (txt, col) => { g.fillStyle = col; g.fillText(txt, xx, tb + Hb + 22); xx += g.measureText(txt).width; };
    put2('sd along track  ', CSS.grey); put2(`cloud ${fmt(sdC, 3)}   `, CSS.cloud); put2(`UKF ${fmt(sdU, 3)}   `, CSS.ukf); put2(`EKF ${fmt(sdE, 3)} m`, CSS.ekf);
    g.restore();
    return { cosU, cosE, exact, cosCloud: st.cosMean, sdAlong: { cloud: sdC, ukf: sdU, ekf: sdE }, groups: groups.map((q) => ({ theta: q.th, n: q.n, w: q.w })) };
  }

  /* ---------- per frame ---------- */
  const disp = new Float64Array(3 * N), redPose = new Float64Array(3 * N);
  let last = null, frameMs = 0;

  v.onFrame((dt) => {
    const f0 = performance.now();
    if (playing) {
      tau = Math.min(NSTEP, tau + Math.min(dt, 0.1) * PLAY_SPEED / H);
      if (tau >= NSTEP) playing = false;
      syncSlider();
      dirty = true;
    }
    const size = `${plot.g.canvas.clientWidth}x${plot.g.canvas.clientHeight}`;
    if (size !== lastSize) { lastSize = size; dirty = true; }
    if (!dirty) return;
    dirty = false;

    const k = Math.floor(tau + 1e-9), fr = Math.min(1, tau - k), k1 = Math.min(NSTEP, k + 1);
    const A = run.snaps[k], B = run.snaps[k1];
    for (let i = 0; i < 3 * N; i++) disp[i] = A[i] + fr * (B[i] - A[i]);
    cloud.set(disp, N);
    const st = stats(k), f = filt();
    const wa = run.wheels[k], wb = run.wheels[k1];
    setCart(v, 'cart', [disp[0], disp[1], disp[2]], [wa[0] + fr * (wb[0] - wa[0]), wa[1] + fr * (wb[1] - wa[1])]);
    const E = run.E[k], U = run.U[k];
    const isU = f === 'ukf';
    if (isU) { setCart(v, 'ukf', U.m, [U.m[0] / S.R_WHEEL, U.m[0] / S.R_WHEEL]); parkBody(v, 'ekf'); } else { setCart(v, 'ekf', E.m, [E.m[0] / S.R_WHEEL, E.m[0] / S.R_WHEEL]); parkBody(v, 'ukf'); }
    ellE.visible = !isU; ellU.visible = isU;
    if (isU) ellU.set(U.m, U.P); else ellE.set(E.m, E.P);
    meanRing.set([st.mean[0], st.mean[1]], 0.035, 0.012);
    truthMark.position.set(disp[0] - 0.045 * Math.cos(disp[2]), disp[1] - 0.045 * Math.sin(disp[2]), 0); pinC.visible = true; pinC.position.set(st.mean[0], st.mean[1], 0); pinE.visible = !isU; pinU.visible = isU; (isU ? pinU : pinE).position.set((isU ? U.m : E.m)[0], (isU ? U.m : E.m)[1], 0);

    // carts outside the shown ellipse
    const out = isU ? st.outU : st.outE;
    red.visible = showRed();
    if (red.visible) {
      out.forEach((i, j) => { redPose[3 * j] = disp[3 * i]; redPose[3 * j + 1] = disp[3 * i + 1]; redPose[3 * j + 2] = disp[3 * i + 2]; });
      red.set(redPose, out.length);
    }

    // sigma points after step k: small carts, each with its own heading, linked to the weighted mean
    const sig = isU && showSigma();
    sigInst.visible = sig;
    const Y = U.pts.Y;
    for (let s = 0; s < 11; s++) {
      const y = Y[s];
      if (sig) {
        M4.makeRotationZ(y[2]).scale(SC).setPosition(y[0], y[1], 0);
        sigInst.setMatrixAt(s, M4);
        const c = Math.cos(y[2]), sn = Math.sin(y[2]), nose = 0.14 * SIGMA_SCALE;
        sigArrows[s].set([y[0] + c * nose, y[1] + sn * nose], [0.2 * c, 0.2 * sn]);
        sigLinks[s].set([[y[0], y[1]], [U.m[0], U.m[1]]], 0.005);
      } else sigArrows[s].visible = false;
      sigLinks[s].visible = sig;
    }
    sigInst.instanceMatrix.needsUpdate = true;
    v.forward();

    last = drawPlot(k);
    const t = k * H;
    readout(panel, 'info',
      `t <b>${fmt(t, 3)} s</b> · mean x: <span class="c-blue">cloud <b>${fmt(st.mean[0], 3)}</b></span> ` +
      `<span class="c-orange">UKF <b>${fmt(U.m[0], 3)}</b></span> <span class="c-yellow">EKF <b>${fmt(E.m[0], 3)}</b></span> m · ` +
      `inside 95 % ellipse: <span class="c-yellow">EKF <b>${Math.round(100 * st.covE)} %</b></span> <span class="c-orange">UKF <b>${Math.round(100 * st.covU)} %</b></span>`);
    frameMs = performance.now() - f0;
  });

  panel._state = () => {
    const k = Math.floor(tau + 1e-9), st = stats(k), E = run.E[k], U = run.U[k];
    const sd = (P) => [0, 1, 2].map((i) => Math.sqrt(Math.max(P[i][i], 0)));
    const dr = S.deadReckoning(run.D);          // the reference path of nlfilters.dead_reckoning
    const cen = U.pts.Y.filter((y) => Math.hypot(y[0] - U.m[0], y[1] - U.m[1]) < 1e-4 && Math.abs(y[2] - U.m[2]) < 1e-4).length;
    return {
      t: k * H, k, tau, playing, DT: run.D, seed: run.seed, N, filter: filt(), sigmaShown: showSigma(), redShown: showRed(),
      truth: [run.snaps[k][0], run.snaps[k][1] - Y0, run.snaps[k][2]],
      cloud: { mean: [st.mean[0], st.mean[1] - Y0, st.mean[2]], sd: sd(st.cov), meanXse: Math.sqrt(st.cov[0][0] / N), lag: V * k * H - st.mean[0] },
      ekf: { m: [E.m[0], E.m[1] - Y0, E.m[2]], sd: sd(E.P) },
      ukf: { m: [U.m[0], U.m[1] - Y0, U.m[2]], sd: sd(U.P) },
      coverage95: { ekf: st.covE, ukf: st.covU, outsideEkf: st.outE.length / N, outsideUkf: st.outU.length / N },
      cosMean: last ? { cloud: last.cosCloud, ukf: last.cosU, ekf: last.cosE, exact: last.exact } : null,
      sdAlongTrack: last ? last.sdAlong : null,
      sigma: { n: U.pts.Y.length, Wm: U.pts.Wm, theta: U.pts.Y.map((y) => y[2]), nearMean: cen, groups: last ? last.groups : null },
      check: { deadReckoningUkfX: dr.ukf.m[0], ukfXAtEnd: run.U[NSTEP].m[0], diff: Math.abs(dr.ukf.m[0] - run.U[NSTEP].m[0]), ekfXAtEnd: run.E[NSTEP].m[0] },
      computeMs: run.ms, frameMs,
    };
  };
}
