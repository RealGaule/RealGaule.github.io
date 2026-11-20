// 3.3 When a Gaussian is not enough. The cart drives the lab's arc (v = 1 m/s, omega = pi/4 rad/s) or is driven
// with the keyboard. As in the clip, the blind phase is set by time, not by a line of sight: the beacon is dark for
// t_blind (mild: D = 1, 0.5 s; severe: D = 8, 1.5 s, the clip's two cells; in keyboard mode t_blind of driving from
// the first key press), then gives a range reading every 0.05 s (first reading at t = t_blind, as nlfilters.Scenario).
// No shelf stands between the path and the beacon: at the severe setting the blind cloud covers that whole area, so
// carts would drive through it (s33_beyond_gaussian.py docstring). The shelf is background only, at a place no cart
// of the auto arc comes within 0.12 m of (2000 carts per setting, sim/cart.py simulate). EKF (yellow), UKF (orange) and a bootstrap particle filter (blue cloud)
// get the same commands and readings. Left: the robot scene. Right: the same from above, and the clip's map of how
// often each filter loses the cart (out/s33_beyond_gaussian_facts.json, copied to data/d33_map.json).
//
// Numerics: sim5.js only. Truth = S.cartStep (Euler-Maruyama, 1 ms, as nlfilters.simulate_truth); EKF / UKF =
// S.EKF / S.UKF at the lab's filter step 0.05 s (between two steps they are drawn with their own prediction over
// the elapsed part of the step); particle filter = S.ParticleFilter (sub-steps 0.01 s), resampled at every reading
// (systematic, survivors keep their slot so that a drawn cart is one particle that moves, as in the clip).
// Lost rules of the lab: S.isLost for EKF / UKF; particle filter: error > 0.15 m and the cart farther from the
// cloud mean than every particle.
import { S, THREE, COL, CSS, makeViewer, setCart, loadGhost, CartCloud, FloorEllipse, RangeRing, Wheels,
  parkBody, keyboard, ctl, val, onClick, seg, toggle, readout, plot2d, loadJSON, fmt, quatYaw } from '../common.js';

const BEACON = [0.5, 1.5];                    // layout 1R of the lab
const SIGMA_R = 0.03;                         // range noise (m)
const DT_F = 0.05, SUB = 0.01, NSUB = 5;      // filter step; particle sub-step = drawing tick; sub-steps per step
const DT_TRUTH = 0.001, NTRUTH = 10;          // truth step; truth steps per tick
const SD0 = 1e-3;                             // the start is known to 1 mm and 1 mrad
const NP = 4000;                              // particles
const ARC = { v: 1, w: Math.PI / 4 };        // the lab's command; the run lasts t_blind + T_SEE
const T_SEE = 1;                              // seconds of readings after the blind phase (auto mode)
const SHELF = { pos: [0.6, 3.3], yaw: 0 };    // background, beyond the floor and the keyboard arena
const ARENA = [-1.0, 2.6, -0.9, 3.0];         // keyboard runs end outside this box (x0, x1, y0, y1)
const SET = { mild: { D: 1, tBlind: 0.5 }, severe: { D: 8, tBlind: 1.5 } };
const V_MAX = 1, W_KEYS = Math.PI / 2, ACCEL = 0.8;

/** Resample at every reading (systematic, S.systematicIndices); a surviving particle keeps its slot, the extra
 *  copies go to the slots of the dead ones. Same distribution as pf.resample(true), up to the order of particles. */
function resampleKeep(pf) {
  const N = pf.Np, idx = S.systematicIndices(pf.weights(), N, pf.rng.uniform()), cnt = new Int32Array(N);
  for (let j = 0; j < N; j++) cnt[idx[j]]++;
  const xn = new Float64Array(pf.x), dead = [];
  for (let p = 0; p < N; p++) if (cnt[p] === 0) dead.push(p);
  let di = 0;
  for (let p = 0; p < N; p++) for (let c = 1; c < cnt[p]; c++) { const q = dead[di++]; xn[3 * q] = pf.x[3 * p]; xn[3 * q + 1] = pf.x[3 * p + 1]; xn[3 * q + 2] = pf.x[3 * p + 2]; }
  pf.x = xn; pf.logw.fill(0); pf.nResampled++;
  return dead.length;
}

export async function init(panel) {
  const v = await makeViewer(panel, 'beacons.mjb', { lookat: [1.0, 0.9, 0.05], distance: 3.8, azimuth: 90, elevation: -58 });
  const geo = await loadGhost();
  const map = await loadJSON('d33_map.json');

  // ---------- scene objects
  parkBody(v, 'b1');
  v.setMocap('shelf', [SHELF.pos[0], SHELF.pos[1], 0], undefined, quatYaw(SHELF.yaw));
  const b0 = v.id('body', 'b0'), lamps = [];
  for (let g = 0; g < v.model.ngeom; g++) {
    const m = v.meshes[g];
    if (v.model.geom_bodyid[g] === b0 && m && m.material && m.material.emissiveIntensity > 0) {
      lamps.push({ m: m.material, color: m.material.color.clone(), emissive: m.material.emissive.clone(), k: m.material.emissiveIntensity });
    }
  }
  const lampOn = (on) => lamps.forEach((l) => {
    if (on) { l.m.color.copy(l.color); l.m.emissive.copy(l.emissive); l.m.emissiveIntensity = l.k; }
    else { l.m.color.setRGB(0.16, 0.17, 0.2); l.m.emissiveIntensity = 0; }
  });
  const cloud = new CartCloud(v.scene, geo, { max: NP, bodies: 40, opacity: 0.28, dot: 0.022 });
  const ellE = new FloorEllipse(v.scene, COL.ekf, { width: 0.014, fill: 0.08 });
  const ellU = new FloorEllipse(v.scene, COL.ukf, { width: 0.014, fill: 0.08, z: 0.005 });
  const ring = new RangeRing(v.scene, COL.reading, { opacity: 0.75 });
  // the true path as a plain line (one point per 0.01 s tick)
  const trail = { n: 4000, count: 0, pos: new Float32Array(3 * 4000) };
  trail.geo = new THREE.BufferGeometry();
  trail.geo.setAttribute('position', new THREE.BufferAttribute(trail.pos, 3));
  trail.geo.setDrawRange(0, 0);
  trail.line = new THREE.Line(trail.geo, new THREE.LineBasicMaterial({ color: COL.truth, transparent: true, opacity: 0.55 }));
  trail.line.frustumCulled = false;
  v.scene.add(trail.line);
  trail.clear = () => { trail.count = 0; trail.geo.setDrawRange(0, 0); };
  trail.push = (p) => {
    if (trail.count === trail.n) { trail.pos.copyWithin(0, 3); trail.count--; }
    trail.pos.set(p, 3 * trail.count); trail.count++;
    trail.geo.setDrawRange(0, trail.count); trail.geo.attributes.position.needsUpdate = true;
  };
  const wheels = new Wheels([0, 0, 0]);
  // white down-arrow above the true cart (as in the clips)
  const arrow = new THREE.Group();
  const wmat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, 0.12, 10), wmat);
  shaft.rotation.x = Math.PI / 2; shaft.position.z = 0.36;
  const head = new THREE.Mesh(new THREE.ConeGeometry(0.03, 0.07, 16), wmat);
  head.rotation.x = -Math.PI / 2; head.position.z = 0.265;
  arrow.add(shaft, head);
  v.scene.add(arrow);

  // HTML tags on the 3D view: "no readings" over the cart, "EKF lost" / "UKF lost" at the estimates
  const cell3d = panel.querySelector('canvas.view3d').parentElement;
  const mkTag = (html, color) => {
    const el = document.createElement('div');
    el.innerHTML = html;
    Object.assign(el.style, { position: 'absolute', left: '0', top: '0', pointerEvents: 'none', font: '600 13px Inter, sans-serif',
      color, background: 'rgba(0,0,0,0.62)', padding: '2px 7px', borderRadius: '4px', whiteSpace: 'nowrap', display: 'none' });
    cell3d.appendChild(el);
    return el;
  };
  const tagBlind = mkTag('no readings', CSS.fail);
  const tagE = mkTag(`<span style="color:${CSS.ekf}">EKF</span>&nbsp; lost`, CSS.fail);
  const tagU = mkTag(`<span style="color:${CSS.ukf}">UKF</span>&nbsp; lost`, CSS.fail);
  const pv = new THREE.Vector3();
  function place(el, p, show, dy = 0) {
    if (!show) { el.style.display = 'none'; return; }
    pv.set(p[0], p[1], p[2]).project(v.camera);
    if (pv.z > 1 || Math.abs(pv.x) > 1.05 || Math.abs(pv.y) > 1.05) { el.style.display = 'none'; return; }
    const c = v.canvas, x = c.offsetLeft + (pv.x + 1) / 2 * c.clientWidth, y = c.offsetTop + (1 - pv.y) / 2 * c.clientHeight;
    el.style.display = 'block';
    el.style.transform = `translate(${Math.round(Math.max(c.offsetLeft + 2, Math.min(c.offsetLeft + c.clientWidth - el.offsetWidth - 2, x - el.offsetWidth / 2)))}px, ${Math.round(y - el.offsetHeight + dy)}px)`;
  }

  // ---------- controls
  let setting = 'mild', mode = 'auto', show = { ekf: true, ukf: true, pf: true };
  const tally = { mild: { runs: 0, ekf: 0, ukf: 0, pf: 0 }, severe: { runs: 0, ekf: 0, ukf: 0, pf: 0 } };
  seg(panel, 'noise', (s) => { setting = s; newRun(); });
  toggle(panel, 'ekf', (on) => { show.ekf = on; });
  toggle(panel, 'ukf', (on) => { show.ukf = on; });
  toggle(panel, 'pf', (on) => { show.pf = on; });
  const setMode = (m) => { mode = m; ctl(panel, 'auto').classList.toggle('on', m === 'auto'); };
  onClick(panel, 'auto', () => { setMode('auto'); newRun(); });
  onClick(panel, 'again', () => newRun());
  const kb = keyboard(panel, ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']);
  let vKeys = 0;
  ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].forEach((k) => kb.onPress(k, () => {
    if (mode === 'auto') { vKeys = ended ? 0 : cmd[0]; setMode('keys'); }
    if (ended) newRun();
  }));

  // ---------- one run
  let runNo = 0, D, tBlind, rngT, rngZ, x, ekf, ukf, pf, nTick, kSub, cmd, los, lastZ, zAge, nZ, blindStart, lastBlind, ended, endWall,
    lostE, lostU, lostP, pfMean, pfErr, essAtReading, reborn, wall = 0;
  function newRun() {
    runNo++;
    D = SET[setting].D; tBlind = SET[setting].tBlind;
    rngT = new S.RNG(1000 + runNo); rngZ = new S.RNG(50000 + runNo);
    x = new Float64Array([SD0 * rngT.normal(), SD0 * rngT.normal(), SD0 * rngT.normal()]);
    const P0 = S.diag([SD0 * SD0, SD0 * SD0, SD0 * SD0]);
    ekf = new S.EKF([0, 0, 0], P0, D);
    ukf = new S.UKF([0, 0, 0], P0, D);
    pf = new S.ParticleFilter([0, 0, 0], P0, D, { nParticles: NP, seed: 90000 + runNo, dtSub: SUB });
    nTick = 0; kSub = 0; cmd = [0, 0]; nZ = 0; lastZ = null; zAge = Infinity; ended = false; endWall = 0;
    los = false; blindStart = 0; lastBlind = 0;   // the beacon is dark from the start
    lostE = lostU = lostP = false; essAtReading = null; reborn = 0;
    vKeys = 0;
    wheels.reset([x[0], x[1], x[2]]); trail.clear();
    pfStats();
  }
  function pfStats() {
    const { m } = pf.meanCov();
    pfMean = m;
    pfErr = Math.hypot(x[0] - m[0], x[1] - m[1]);
    let far = 0;
    for (let p = 0; p < NP; p++) far = Math.max(far, Math.hypot(pf.x[3 * p] - m[0], pf.x[3 * p + 1] - m[1]));
    lostP = pfErr > 0.15 && pfErr > far;
  }
  const t = () => nTick * SUB;
  function command() {
    if (mode === 'auto') return [ARC.v, ARC.w];
    const turn = (kb.down('ArrowLeft') ? 1 : 0) - (kb.down('ArrowRight') ? 1 : 0);
    return [vKeys, turn * W_KEYS];
  }
  /** One tick of 0.01 s: truth, particles; every fifth tick the EKF / UKF step and, from t = t_blind on, a reading. */
  function tick() {
    if (kSub === 0) cmd = command();
    for (let i = 0; i < NTRUTH; i++) S.cartStep(x, cmd[0], cmd[1], D, DT_TRUTH, rngT);
    pf.predict(cmd[0], cmd[1], SUB);
    nTick++; kSub++; zAge += SUB;
    trail.push([x[0], x[1], 0.012]);
    if (kSub < NSUB) return;
    kSub = 0;
    ekf.predict(cmd[0], cmd[1], DT_F);
    ukf.predict(cmd[0], cmd[1], DT_F);
    los = t() >= tBlind - 1e-9;
    if (los) {
      if (blindStart !== null) { lastBlind = t() - blindStart; blindStart = null; }
      const z = [Math.hypot(BEACON[0] - x[0], BEACON[1] - x[1]) + SIGMA_R * rngZ.normal()];
      ekf.update(z, [BEACON], SIGMA_R);
      ukf.update(z, [BEACON], SIGMA_R);
      pf.update(z, [BEACON], SIGMA_R, { resample: false });
      essAtReading = pf.ess();
      reborn = resampleKeep(pf);
      lastZ = z[0]; zAge = 0; nZ++;
    } else if (blindStart === null) blindStart = t();
    lostE = S.isLost(x, ekf.m, ekf.P);
    lostU = S.isLost(x, ukf.m, ukf.P);
    pfStats();
    const out = x[0] < ARENA[0] || x[0] > ARENA[1] || x[1] < ARENA[2] || x[1] > ARENA[3];
    if ((mode === 'auto' && t() >= tBlind + T_SEE - 1e-9) || (mode === 'keys' && (out || t() >= 30))) endRun();
  }
  function endRun() {
    ended = true; endWall = wall;
    const s = tally[setting];
    s.runs++; s.ekf += lostE; s.ukf += lostU; s.pf += lostP;
  }
  /** EKF / UKF drawn between two filter steps: their own prediction over the elapsed part of the step. */
  function shown(f, isU) {
    if (kSub === 0) return { m: f.m, P: f.P };
    const tau = kSub * SUB;
    return isU ? S.ukfPredict(f.m, f.P, cmd[0], cmd[1], D, tau) : S.ekfPredict(f.m, f.P, cmd[0], cmd[1], D, tau);
  }

  // ---------- 2D picture: top view and the clip's map
  const o = plot2d(panel.querySelector('canvas.plot'));
  const XR = [-0.9, 2.7], YR = [-0.75, 2.85];
  function draw2d(dE, dU) {
    o.clear();
    const g = o.g, W = o.w, H = o.h, mapH = Math.min(168, 0.38 * H), topH = H - mapH - 8;
    const ppm = Math.min((W - 44) / (XR[1] - XR[0]), (topH - 30) / (YR[1] - YR[0]));
    const ox = 34 + (W - 44 - ppm * (XR[1] - XR[0])) / 2, oy = 22;
    const X = (u) => ox + (u - XR[0]) * ppm, Y = (w) => oy + (YR[1] - w) * ppm;
    g.save();
    g.beginPath(); g.rect(X(XR[0]), Y(YR[1]), ppm * (XR[1] - XR[0]), ppm * (YR[1] - YR[0])); g.clip();
    // grid
    g.strokeStyle = 'rgba(255,255,255,0.07)'; g.lineWidth = 1;
    for (let u = -0.5; u <= 2.51; u += 0.5) { g.beginPath(); g.moveTo(X(u), Y(YR[0])); g.lineTo(X(u), Y(YR[1])); g.stroke(); }
    for (let w = -0.5; w <= 2.51; w += 0.5) { g.beginPath(); g.moveTo(X(XR[0]), Y(w)); g.lineTo(X(XR[1]), Y(w)); g.stroke(); }
    // range reading: ring with its +-2 sigma band
    const ringOn = zAge < DT_F + 1e-9 && lastZ !== null;
    if (ringOn) {
      g.strokeStyle = 'rgba(131,193,103,0.22)'; g.lineWidth = Math.max(2, 4 * SIGMA_R * ppm);
      g.beginPath(); g.arc(X(BEACON[0]), Y(BEACON[1]), lastZ * ppm, 0, 2 * Math.PI); g.stroke();
      g.strokeStyle = CSS.reading; g.lineWidth = 1.5;
      g.beginPath(); g.arc(X(BEACON[0]), Y(BEACON[1]), lastZ * ppm, 0, 2 * Math.PI); g.stroke();
    }
    // true path
    g.strokeStyle = 'rgba(238,238,238,0.35)'; g.lineWidth = 1.2; g.beginPath();
    const tp = trail.pos, n0 = 0;
    for (let i = 0; i < 3 * trail.count; i += 3) { const px = X(tp[i]), py = Y(tp[i + 1]); if (i === n0) g.moveTo(px, py); else g.lineTo(px, py); }
    g.stroke();
    // particles
    if (show.pf) {
      g.fillStyle = 'rgba(88,196,221,0.75)';
      const pxs = pf.x;
      for (let p = 0; p < NP; p++) g.fillRect(X(pxs[3 * p]) - 1, Y(pxs[3 * p + 1]) - 1, 2, 2);
    }
    // ellipses (2 sigma) and means
    const ell = (d, css, fill) => {
      const pts = S.ellipsePoints([d.m[0], d.m[1]], [[d.P[0][0], d.P[0][1]], [d.P[1][0], d.P[1][1]]], 2, 96);
      g.beginPath(); pts.forEach((q, i) => (i ? g.lineTo(X(q[0]), Y(q[1])) : g.moveTo(X(q[0]), Y(q[1])))); g.closePath();
      g.fillStyle = fill; g.fill(); g.strokeStyle = css; g.lineWidth = 2; g.stroke();
      g.fillStyle = css; g.beginPath(); g.arc(X(d.m[0]), Y(d.m[1]), 3, 0, 2 * Math.PI); g.fill();
    };
    if (show.ekf) ell(dE, CSS.ekf, 'rgba(255,235,64,0.07)');
    if (show.ukf) ell(dU, CSS.ukf, 'rgba(240,164,93,0.07)');
    // beacon
    g.fillStyle = los ? CSS.reading : '#5a5d66';
    g.beginPath(); g.arc(X(BEACON[0]), Y(BEACON[1]), 5, 0, 2 * Math.PI); g.fill();
    // true cart: white ring with a heading tick
    g.strokeStyle = '#fff'; g.lineWidth = 2; g.fillStyle = '#000';
    g.beginPath(); g.arc(X(x[0]), Y(x[1]), 5.5, 0, 2 * Math.PI); g.fill(); g.stroke();
    g.beginPath(); g.moveTo(X(x[0]), Y(x[1])); g.lineTo(X(x[0] + 0.12 * Math.cos(x[2])), Y(x[1] + 0.12 * Math.sin(x[2]))); g.stroke();
    // labels
    g.font = '600 12px Inter, sans-serif'; g.textBaseline = 'middle'; g.textAlign = 'left';
    const lab = (d, txt, css, dy) => { const w1 = g.measureText(txt).width, w = w1 + 5 + g.measureText('lost').width; let x0 = X(d.m[0]) + 12; if (x0 + w > X(XR[1]) - 4) x0 = X(d.m[0]) - 12 - w; g.fillStyle = css; g.fillText(txt, x0, Y(d.m[1]) + dy); g.fillStyle = CSS.fail; g.fillText('lost', x0 + w1 + 5, Y(d.m[1]) + dy); };
    if (show.ekf && lostE) lab(dE, 'EKF', CSS.ekf, -8);
    if (show.ukf && lostU) lab(dU, 'UKF', CSS.ukf, 8);
    if (blindStart !== null && !ended) { g.fillStyle = CSS.fail; const wn = g.measureText('no readings').width; let xn = X(x[0]) + 10; if (xn + wn > X(XR[1]) - 4) xn = X(x[0]) - 10 - wn; g.fillText('no readings', xn, Y(x[1]) - 14); }
    g.font = '11px Inter, sans-serif'; g.fillStyle = CSS.grey; g.fillText('beacon', X(BEACON[0]) + 9, Y(BEACON[1]) - 9);
    g.restore();
    // frame, ticks and title
    g.strokeStyle = CSS.faint; g.lineWidth = 1; g.strokeRect(X(XR[0]), Y(YR[1]), ppm * (XR[1] - XR[0]), ppm * (YR[1] - YR[0]));
    g.fillStyle = CSS.grey; g.font = '10px Inter, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'top';
    for (let u = 0; u <= 2.01; u += 1) g.fillText(u.toFixed(0), X(u), Y(YR[0]) + 3);
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (let w = 0; w <= 2.01; w += 1) g.fillText(w.toFixed(0), X(XR[0]) - 4, Y(w));
    g.textAlign = 'left'; g.textBaseline = 'alphabetic'; g.fillStyle = CSS.white; g.font = '12px Inter, sans-serif';
    g.fillText('top view (m)', X(XR[0]), 14);
    g.textAlign = 'right'; g.fillStyle = CSS.grey; g.font = '11px Inter, sans-serif';
    g.fillText(`t = ${fmt(t(), 2)} s`, X(XR[1]), 14);
    drawMap(g, W, H - mapH, mapH);
  }

  function drawMap(g, W, y0, mH) {
    const names = [['EKF', CSS.ekf, map.ekf_lost_end_pct], ['UKF', CSS.ukf, map.ukf_lost_end_pct], ['particle filter', CSS.cloud, map.pf_outside_cloud_end_pct]];
    const rows = map.rows_D, cols = map.cols_t_blind, nr = rows.length, nc = cols.length;
    const left = 34, gap = 12, cw = Math.min(22, (W - left - 8 - 2 * gap) / (3 * nc)), chh = Math.min(cw, (mH - 50) / nr);
    const gw = cw * nc, top = y0 + 30;
    g.textBaseline = 'alphabetic'; g.textAlign = 'left'; g.fillStyle = CSS.white; g.font = '12px Inter, sans-serif';
    g.fillText('share of runs that lose the cart (%), the clip\'s arc', left - 26, y0 + 12);
    // the cell of this run: D and t_blind of the setting
    const iD = rows.indexOf(D), iT = cols.indexOf(tBlind);
    names.forEach(([nm, css, M], f) => {
      const gx = left + f * (gw + gap);
      g.fillStyle = css; g.font = '600 11px Inter, sans-serif'; g.textAlign = 'center';
      g.fillText(nm, gx + gw / 2, top - 5);
      for (let i = 0; i < nr; i++) for (let j = 0; j < nc; j++) {
        const pct = M[i][j], cx = gx + j * cw, cy = top + (nr - 1 - i) * chh;
        g.fillStyle = pct > 0 ? `rgba(252,98,85,${0.12 + 0.75 * Math.min(1, pct / 60)})` : '#0d0e12';
        g.fillRect(cx + 1, cy + 1, cw - 2, chh - 2);
        if (cw >= 16) { g.fillStyle = pct > 0 ? '#fff' : '#555'; g.font = '9px Inter, sans-serif'; g.textBaseline = 'middle'; g.fillText(String(pct), cx + cw / 2, cy + chh / 2 + 0.5); g.textBaseline = 'alphabetic'; }
      }
      if (iD >= 0 && iT >= 0) { g.strokeStyle = '#fff'; g.lineWidth = 2; g.strokeRect(gx + iT * cw + 0.5, top + (nr - 1 - iD) * chh + 0.5, cw - 1, chh - 1); }
    });
    g.fillStyle = CSS.grey; g.font = '9px Inter, sans-serif'; g.textAlign = 'right'; g.textBaseline = 'middle';
    rows.forEach((r, i) => g.fillText(String(r), left - 4, top + (nr - 1 - i) * chh + chh / 2));
    g.textAlign = 'left'; g.save(); g.translate(left - 24, top + nr * chh / 2); g.rotate(-Math.PI / 2); g.textAlign = 'center'; g.fillText('wheel noise D', 0, 0); g.restore();
    g.textAlign = 'center'; g.textBaseline = 'top';
    cols.forEach((c, j) => g.fillText(String(c), left + j * cw + cw / 2, top + nr * chh + 2));
    g.fillText(`time without readings (s) · runs per cell ${map.runs_per_cell.ekf}, ${map.runs_per_cell.ukf}, ${map.runs_per_cell.pf}`, left + (3 * gw + 2 * gap) / 2, top + nr * chh + 14);
  }

  // ---------- frame loop
  let acc = 0, frames = 0;
  newRun();
  v.onFrame((dt) => {
    wall += dt; frames++;
    if (mode === 'keys') {
      if (kb.down('ArrowUp')) vKeys = Math.min(V_MAX, vKeys + ACCEL * dt);
      if (kb.down('ArrowDown')) vKeys = Math.max(0, vKeys - ACCEL * dt);
    }
    const speed = val(panel, 'speed');
    if (ended) {
      acc = 0;
      if (mode === 'auto' && wall - endWall > 2.5) newRun();
    } else {
      acc += dt * speed;
      let n = 0;
      while (acc >= SUB && n < 40) {
        if (kSub === 0 && mode === 'keys') { const c = command(); if (c[0] === 0 && c[1] === 0) { acc = 0; break; } }
        tick(); acc -= SUB; n++;
        if (ended) break;
      }
      if (n === 40) acc = 0;
    }
    // 3D
    const dE = shown(ekf, false), dU = shown(ukf, true);
    setCart(v, 'cart', [x[0], x[1], x[2]], wheels.update([x[0], x[1], x[2]]));
    arrow.position.set(x[0], x[1], 0);
    if (show.ekf) { setCart(v, 'ekf', dE.m); ellE.set(dE.m, dE.P); } else parkBody(v, 'ekf');
    if (show.ukf) { setCart(v, 'ukf', dU.m); ellU.set(dU.m, dU.P); } else parkBody(v, 'ukf');
    ellE.visible = show.ekf; ellU.visible = show.ukf;
    cloud.visible = show.pf;
    if (show.pf) cloud.set(pf.x, NP);
    const ringOn = zAge < DT_F + 1e-9 && lastZ !== null;
    ring.visible = ringOn;
    if (ringOn) ring.set(BEACON, lastZ, 4 * SIGMA_R);
    lampOn(los);
    v.forward();
    place(tagBlind, [x[0], x[1], 0.5], blindStart !== null && !ended);
    place(tagE, [dE.m[0], dE.m[1], 0.3], show.ekf && lostE, -6);
    place(tagU, [dU.m[0], dU.m[1], 0.3], show.ukf && lostU, 20);
    draw2d(dE, dU);
    // readouts
    const blindNow = blindStart !== null ? t() - blindStart : 0;
    const e = (d) => Math.hypot(x[0] - d.m[0], x[1] - d.m[1]);
    const lost = (b) => (b ? ' <span class="c-red">lost</span>' : '');
    readout(panel, 'info', `${blindStart !== null ? `<span class="c-red">no readings ${fmt(blindNow, 2)} s</span>` : `readings <b>${nZ}</b> (blind ${fmt(lastBlind, 2)} s)`}`
      + ` · error <span class="c-yellow">EKF ${fmt(e(ekf), 2)} m</span>${lost(lostE)}, <span class="c-orange">UKF ${fmt(e(ukf), 2)} m</span>${lost(lostU)},`
      + ` <span class="c-blue">PF ${fmt(pfErr, 2)} m</span>${lost(lostP)}${mode === 'keys' && !ended && vKeys === 0 ? ' · <b>press ↑ to drive</b>' : ''}${ended ? ' · end of run' : ''}`);
    const s = tally[setting], pc = (a) => (s.runs ? Math.round(100 * a / s.runs) : 0);
    readout(panel, 'tally', `${setting} runs on this page: <b>${s.runs}</b> · lost at the end: <span class="c-yellow">EKF ${s.ekf}</span> (${pc(s.ekf)} %),`
      + ` <span class="c-orange">UKF ${s.ukf}</span> (${pc(s.ukf)} %), <span class="c-blue">PF ${s.pf}</span> (${pc(s.pf)} %)`);
  });

  const sd = (P) => [Math.sqrt(P[0][0]), Math.sqrt(P[1][1]), Math.sqrt(P[2][2])];
  panel._state = () => ({
    t: t(), run: runNo, mode, setting, D, tBlind, tEnd: tBlind + T_SEE, ended, cmd, los, readings: nZ, lastZ,
    blindNow: blindStart !== null ? t() - blindStart : 0, lastBlind,
    truth: Array.from(x),
    ekf: { m: ekf.m, sd: sd(ekf.P), err: Math.hypot(x[0] - ekf.m[0], x[1] - ekf.m[1]), neesXY: S.errorStats(x, ekf.m, ekf.P).neesXY, lost: lostE },
    ukf: { m: ukf.m, sd: sd(ukf.P), err: Math.hypot(x[0] - ukf.m[0], x[1] - ukf.m[1]), neesXY: S.errorStats(x, ukf.m, ukf.P).neesXY, lost: lostU },
    pf: { n: NP, mean: pfMean, err: pfErr, lost: lostP, essAtLastReading: essAtReading, rebornAtLastReading: reborn, resamples: pf.nResampled },
    tally: JSON.parse(JSON.stringify(tally)), show: { ...show }, speed: val(panel, 'speed'), frames,
    keysLive: panel.classList.contains('keys-on'),
  });
}
