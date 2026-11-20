// Section 0, opening: where is the cart?  (clip s00_opening)
// A pool of carts gets the same command, "1 m straight ahead in 1 s" (cart SDE of the lab, Euler-Maruyama, dt = 1 ms,
// sim5.cartStep). The cloud bends into a banana; the yellow Gaussian summary (fitted mean and 2-sigma ellipse) has the
// wrong shape for it. Range readings of the true cart to one or two beacons (sigma_r = 3 cm) reweight the pool with
// the bootstrap particle filter's update (sim5.ParticleFilter.update, no resampling): one ring leaves an arc, two
// rings a small lump. As in the clip, a reading is faded in as the tempered likelihood L^beta, beta from 0 to 1
// (exactly ParticleFilter.update with sigma_r / sqrt(beta)); the ring growing from the beacon is drawing only.
import { S, THREE, COL, CSS, makeViewer, setCart, Wheels, loadGhost, CartCloud, FloorEllipse, RangeRing, Ribbon,
  parkBody, val, onInput, onClick, seg, toggle, readout, plot2d, scales, drawEllipse, stepper } from '../common.js';

const N_POOL = 10000;                 // simulated carts (the posterior needs a pool: two rings keep a few per cent)
const N_SHOW = 2000;                  // carts drawn (dots), the first BODIES also as translucent bodies
const BODIES = 40;
const V = 1, T = 1, H = 1e-3, N_STEPS = S.roundHalfEven(T / H);
const SIGMA_R = 0.03;                 // range noise of the lab's sensor (s00_opening_facts.json)
const BEACONS = [[2.4, 0.0], [0.7, 1.9]];   // the clip's layout: first beacon ahead of the cart, second at the side
const DRIVE_SPEED = 0.4;              // simulated seconds per second of the page (1 s of driving takes 2.5 s)
const RISE = 1.0, FALL = 3.0;         // per second: a reading = ring 1 s (drawing) + weighting 1 s (lev 0 -> 2)
const SHOW_SEED = 77;                 // fixed stream for drawing N_SHOW carts by weight (no flicker)

export async function init(panel) {
  const v = await makeViewer(panel, 'beacons.mjb', { lookat: [1.15, 0.5, 0], distance: 4.2, azimuth: 72, elevation: -44 });
  const geo = await loadGhost();

  // ---------- scene
  ['b0', 'b1'].forEach((b, k) => {
    const i = v.model.body_mocapid[v.id('body', b)], z0 = v.data.mocap_pos[3 * i + 2];
    v.setMocap(b, [BEACONS[k][0], BEACONS[k][1], z0]);
  });
  parkBody(v, 'shelf'); parkBody(v, 'ukf');
  const lamps = ['b0_lamp', 'b1_lamp'].map((n) => v.meshes[v.id('geom', n)]);
  const lampLook = lamps.map((m) => m && { color: m.material.color.clone(), ei: m.material.emissiveIntensity || 0 });
  const DARK = new THREE.Color(0x2a2d33);
  function setLamp(k, on) {
    const m = lamps[k];
    if (!m) return;
    m.material.color.copy(on ? lampLook[k].color : DARK);
    if (m.material.emissive) { m.material.emissive.copy(on ? lampLook[k].color : DARK); m.material.emissiveIntensity = on ? Math.max(lampLook[k].ei, 0.8) : 0; }
  }

  // the command: a dashed line from the start mark to the commanded end, 1 m ahead
  const grey = 0xb9bcc6;
  for (let i = 0; i < 10; i++) {
    const d = new Ribbon(v.scene, grey, { n: 2, closed: false, opacity: 0.7 });
    d.set([[0.1 * i + 0.02, 0], [0.1 * i + 0.08, 0]], 0.014);
  }
  new Ribbon(v.scene, grey, { n: 2, closed: false, opacity: 0.85 }).set([[-0.005, -0.09], [-0.005, 0.09]], 0.018);
  new Ribbon(v.scene, grey, { n: 2, closed: false, opacity: 0.55 }).set([[1.0, -0.05], [1.0, 0.05]], 0.012);

  // white down-arrow over the true cart (the clips' marker)
  const arrow = new THREE.Group();
  const white = new THREE.MeshBasicMaterial({ color: COL.truth });
  const head = new THREE.Mesh(new THREE.ConeGeometry(0.035, 0.07, 20), white); head.rotation.x = Math.PI; head.position.z = 0.035;
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.011, 0.09, 12), white); shaft.rotation.x = Math.PI / 2; shaft.position.z = 0.115;
  arrow.add(head, shaft);
  v.scene.add(arrow);

  const cloud = new CartCloud(v.scene, geo, { max: N_SHOW, bodies: BODIES, opacity: 0.3, dot: 0.022 });
  const ell = new FloorEllipse(v.scene, COL.est, { width: 0.016, fill: 0.08 });
  const ringBand = [0, 1].map(() => new RangeRing(v.scene, COL.reading, { opacity: 0.2 }));
  const ringLine = [0, 1].map(() => new RangeRing(v.scene, COL.reading, { opacity: 0.9, z: 0.0025 }));

  // ---------- state
  const pf = new S.ParticleFilter([0, 0, 0], S.diag([1e-12, 1e-12, 1e-12]), 1, { nParticles: N_POOL, seed: 1 });
  const truth = new Float64Array(3);
  const wheels = new Wheels([0, 0, 0]);
  const advance = stepper(H);
  let run = 0, D = 7, k = 0, rngPool, rngTrue, z = [NaN, NaN], lev = [0, 0];
  let mode = 0, gaussOn = true, cur = null, bars = [0, NaN, NaN], show = null, view = null;

  function startRun(fresh) {
    if (fresh) run++;
    D = val(panel, 'DT') / T;
    rngPool = new S.RNG(500 + run); rngTrue = new S.RNG(10000 + run);
    pf.x.fill(0); pf.logw.fill(0);
    truth.fill(0); wheels.reset([0, 0, 0]);
    k = 0; z = [NaN, NaN]; lev = [0, 0];
    refresh();
  }

  function finishDrive() {               // the carts stand: the true cart's readings arrive (noise sigma_r)
    const rngRead = new S.RNG(20000 + run);
    z = S.measure(Array.from(truth), BEACONS).map((r) => r + SIGMA_R * rngRead.normal());
  }

  function stepAll(h) {
    if (k >= N_STEPS) return;
    S.cartStep(pf.x, V, 0, D, h, rngPool);
    S.cartStep(truth, V, 0, D, h, rngTrue, { n: 1 });
    k++;
    if (k === N_STEPS) finishDrive();
  }

  const beta = (l) => Math.min(Math.max(l - 1, 0), 1);
  /** Weights of the pool for tempered readings b0, b1 (0..1): ParticleFilter.update with sigma_r / sqrt(beta). */
  function weigh(b0, b1) {
    pf.logw.fill(0);
    if (b0 > 0) pf.update([z[0]], [BEACONS[0]], SIGMA_R / Math.sqrt(b0), { resample: false });
    if (b1 > 0) pf.update([z[1]], [BEACONS[1]], SIGMA_R / Math.sqrt(b1), { resample: false });
    const { m, P } = pf.meanCov();
    return { m, P, spread: Math.sqrt(P[0][0] + P[1][1]), ess: pf.ess() };
  }

  /** Recompute the belief (and the bars) after the cloud or the readings changed. */
  function refresh() {
    const b0 = beta(lev[0]), b1 = beta(lev[1]);
    bars[0] = weigh(0, 0).spread;
    bars[1] = lev[0] > 1 ? weigh(b0, 0).spread : NaN;
    bars[2] = lev[1] > 1 ? weigh(b0, b1).spread : NaN;
    cur = weigh(b0, b1);                  // leaves the current weights in pf.logw
    show = pf.equalWeightCloud(N_SHOW, new S.RNG(SHOW_SEED));
    cloud.set(show, N_SHOW);
  }

  // ---------- controls
  onInput(panel, 'DT', () => startRun(false));
  onClick(panel, 'drive', () => startRun(true));
  seg(panel, 'beacons', (x) => { mode = Number(x); });
  toggle(panel, 'gauss', (on) => { gaussOn = on; });

  // ---------- 2D map (seen from above, moves with the 3D picture) and the spread bars
  const plot = panel.querySelector('canvas.plot');
  const o = plot2d(plot);
  const VIEW_BLIND = [-0.25, 1.45, -1.05, 1.05], VIEW_WIDE = [-0.35, 2.65, -1.15, 2.15];
  function viewTarget() {
    if (lev[1] >= 1.95 && cur) { const r = 0.32; return [cur.m[0] - r, cur.m[0] + r, cur.m[1] - r, cur.m[1] + r]; }
    return (mode > 0 || lev[0] > 0 || lev[1] > 0) ? VIEW_WIDE : VIEW_BLIND;
  }

  function header() {
    if (k < N_STEPS) return ['one command, ' + N_SHOW.toLocaleString('en-US') + ' carts drawn', CSS.white];
    if (lev[1] > 1) return ['two range readings: one lump', CSS.reading];
    if (lev[0] > 1) return ['one range reading: an arc along the ring', CSS.reading];
    if (!gaussOn) return ['one command, ' + N_SHOW.toLocaleString('en-US') + ' carts drawn', CSS.white];
    return D * T >= 4 ? ['Gaussian summary: wrong shape', CSS.est] : ['Gaussian summary: low noise, close fit', CSS.est];
  }

  function drawMap(dt) {
    o.clear();
    const g = o.g, W = o.w, Hh = o.h;
    if (W < 50 || Hh < 120) return;
    const tgt = viewTarget();
    if (!view) view = tgt.slice();
    const a = 1 - Math.exp(-4 * dt);
    for (let i = 0; i < 4; i++) view[i] += a * (tgt[i] - view[i]);
    // equal scale in x and y
    const pad = [46, 14, 52, 166], bw = W - pad[0] - pad[1], bh = Hh - pad[2] - pad[3];
    const cx = (view[0] + view[1]) / 2, cy = (view[2] + view[3]) / 2;
    const sc = Math.min(bw / (view[1] - view[0]), bh / (view[3] - view[2]));
    const s = scales(o, [cx - bw / sc / 2, cx + bw / sc / 2], [cy - bh / sc / 2, cy + bh / sc / 2], pad);
    const [l, t, bW, bH] = s.box;

    // title
    const [title, tcol] = header();
    g.font = '600 14px Inter, sans-serif'; g.fillStyle = tcol; g.textAlign = 'left'; g.textBaseline = 'top';
    g.fillText(title, 14, 10);

    // grid and ticks
    const span = Math.max(s.xr[1] - s.xr[0], s.yr[1] - s.yr[0]);
    const step = span > 2.2 ? 0.5 : span > 0.9 ? 0.25 : 0.1, dig = step < 0.25 ? 1 : 2;
    g.save();
    g.strokeStyle = 'rgba(242,239,233,0.07)'; g.lineWidth = 1; g.fillStyle = CSS.grey; g.font = '10.5px Inter, sans-serif';
    for (let x = Math.ceil(s.xr[0] / step) * step; x <= s.xr[1]; x += step) {
      const px = s.X(x); g.beginPath(); g.moveTo(px, t); g.lineTo(px, t + bH); g.stroke();
      g.textAlign = 'center'; g.textBaseline = 'top'; g.fillText((Math.abs(x) < 1e-9 ? 0 : x).toFixed(dig), px, t + bH + 5);
    }
    for (let y = Math.ceil(s.yr[0] / step) * step; y <= s.yr[1]; y += step) {
      const py = s.Y(y); g.beginPath(); g.moveTo(l, py); g.lineTo(l + bW, py); g.stroke();
      g.textAlign = 'right'; g.textBaseline = 'middle'; g.fillText((Math.abs(y) < 1e-9 ? 0 : y).toFixed(dig), l - 5, py);
    }
    g.strokeStyle = CSS.faint; g.strokeRect(l, t, bW, bH);
    g.fillStyle = CSS.grey; g.font = 'italic 11px Inter, sans-serif';
    g.textAlign = 'right'; g.textBaseline = 'top'; g.fillText('x (m)', l + bW, t + bH + 18);
    g.textAlign = 'left'; g.textBaseline = 'bottom'; g.fillText('y (m)', l + 4, t - 3);

    g.beginPath(); g.rect(l, t, bW, bH); g.clip();
    // the command
    g.strokeStyle = 'rgba(200,203,212,0.6)'; g.lineWidth = 1.2; g.setLineDash([5, 4]);
    g.beginPath(); g.moveTo(s.X(0), s.Y(0)); g.lineTo(s.X(1), s.Y(0)); g.stroke(); g.setLineDash([]);
    g.beginPath(); g.moveTo(s.X(0), s.Y(-0.06)); g.lineTo(s.X(0), s.Y(0.06)); g.stroke();
    // rings
    for (let j = 0; j < 2; j++) {
      if (!(lev[j] > 0) || !isFinite(z[j])) continue;
      const r = z[j] * Math.min(lev[j], 1), bx = s.X(BEACONS[j][0]), by = s.Y(BEACONS[j][1]);
      g.strokeStyle = 'rgba(131,193,103,0.16)'; g.lineWidth = Math.max(2, 4 * SIGMA_R * sc);
      g.beginPath(); g.arc(bx, by, r * sc, 0, 2 * Math.PI); g.stroke();
      g.strokeStyle = 'rgba(131,193,103,0.85)'; g.lineWidth = 1.3;
      g.beginPath(); g.arc(bx, by, r * sc, 0, 2 * Math.PI); g.stroke();
    }
    // beacons
    for (let j = 0; j < 2; j++) {
      const lit = lev[j] > 0 || (j === 0 ? mode >= 1 : mode >= 2), bx = s.X(BEACONS[j][0]), by = s.Y(BEACONS[j][1]);
      g.beginPath(); g.arc(bx, by, 5, 0, 2 * Math.PI);
      if (lit) { g.fillStyle = CSS.reading; g.fill(); } else { g.strokeStyle = CSS.grey; g.lineWidth = 1.5; g.stroke(); }
      g.fillStyle = lit ? CSS.reading : CSS.grey; g.font = '11px Inter, sans-serif'; g.textBaseline = 'bottom';
      const right = bx > l + bW - 70; g.textAlign = right ? 'right' : 'left';
      g.fillText('beacon ' + (j + 1), right ? bx - 8 : bx + 8, by - 4);
    }
    // cloud
    if (show) {
      g.fillStyle = 'rgba(88,196,221,0.55)';
      const dsz = sc > 600 ? 3 : 2;
      for (let i = 0; i < N_SHOW; i++) g.fillRect(s.X(show[3 * i]) - dsz / 2, s.Y(show[3 * i + 1]) - dsz / 2, dsz, dsz);
    }
    // Gaussian summary
    if (gaussOn && cur && k > 0) {
      const c2 = [[cur.P[0][0], cur.P[0][1]], [cur.P[1][0], cur.P[1][1]]];
      if (c2[0][0] > 1e-10 && c2[1][1] > 1e-10) drawEllipse(o, s, [cur.m[0], cur.m[1]], c2, 2, CSS.est, 'rgba(255,235,64,0.07)', 2);
      g.fillStyle = CSS.est; g.beginPath(); g.arc(s.X(cur.m[0]), s.Y(cur.m[1]), 3.5, 0, 2 * Math.PI); g.fill();
    }
    // true cart
    g.fillStyle = '#000'; g.strokeStyle = CSS.truth; g.lineWidth = 2;
    g.beginPath(); g.arc(s.X(truth[0]), s.Y(truth[1]), 5.5, 0, 2 * Math.PI); g.fill(); g.stroke();
    g.fillStyle = CSS.truth; g.beginPath(); g.arc(s.X(truth[0]), s.Y(truth[1]), 2.2, 0, 2 * Math.PI); g.fill();
    g.restore();

    // bars: spread of the cloud
    const y0 = Hh - pad[3] + 38, barMax = 0.5, bx0 = 16, bwid = Math.max(60, W - bx0 - 80);
    g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    g.fillStyle = CSS.white; g.font = '13px Inter, sans-serif'; g.fillText('spread of the cloud', bx0, y0);
    g.fillStyle = CSS.grey; g.font = '10.5px Inter, sans-serif'; g.fillText('rms distance of the carts from their mean', bx0, y0 + 14);
    const rows = [['command only', bars[0]], ['one range reading', bars[1]], ['two range readings', bars[2]]];
    rows.forEach(([lab, x], i) => {
      const yy = y0 + 34 + 26 * i;
      g.fillStyle = isFinite(x) ? CSS.white : CSS.faint; g.font = '11px Inter, sans-serif'; g.fillText(lab, bx0, yy);
      if (!isFinite(x)) return;
      const len = Math.max(2, Math.min(1, x / barMax) * bwid);
      g.fillStyle = '#4eb3cc'; g.fillRect(bx0, yy + 4, len, 8);
      g.fillStyle = CSS.white; g.font = '11px ' + 'ui-monospace, monospace'; g.fillText(x.toFixed(2) + ' m', bx0 + len + 6, yy + 12);
    });
    if (isFinite(bars[1]) && bars[1] > bars[0]) { g.fillStyle = CSS.grey; g.font = '10.5px Inter, sans-serif'; g.fillText('one ring keeps carts on both sides: no smaller', bx0, y0 + 34 + 26 * 3); }
  }

  // ---------- frame
  let lastInfo = '';
  v.onFrame((dt) => {
    let changed = false;
    if (k < N_STEPS) {
      advance(dt, DRIVE_SPEED, stepAll);
      changed = true;
    } else {
      const tg = [mode >= 1 ? 2 : 0, mode >= 2 ? 2 : 0];
      for (let j = 0; j < 2; j++) {
        if (lev[j] === tg[j]) continue;
        if (tg[j] > lev[j]) { if (j === 1 && lev[0] < 2) continue; lev[j] = Math.min(tg[j], lev[j] + RISE * dt); }
        else lev[j] = Math.max(tg[j], lev[j] - FALL * dt);
        changed = true;
      }
    }
    if (changed) refresh();
    for (let j = 0; j < 2; j++) {
      const on = lev[j] > 0 && isFinite(z[j]);
      setLamp(j, on || (j === 0 ? mode >= 1 : mode >= 2) && k >= N_STEPS);
      ringBand[j].visible = ringLine[j].visible = on;
      if (on) {
        const r = Math.max(0.02, z[j] * Math.min(lev[j], 1));
        ringBand[j].set(BEACONS[j], r, 4 * SIGMA_R);
        ringLine[j].set(BEACONS[j], r, 0.016);
        ringBand[j].opacity = 0.2 * Math.min(lev[j], 1); ringLine[j].opacity = 0.9 * Math.min(lev[j], 1);
      }
    }
    // carts
    const pose = [truth[0], truth[1], truth[2]];
    setCart(v, 'cart', pose, wheels.update(pose));
    arrow.position.set(truth[0], truth[1], 0.2);
    const showG = gaussOn && k > 0 && cur;
    ell.visible = !!showG;
    if (showG) { ell.set(cur.m, cur.P); setCart(v, 'ekf', cur.m, [0, 0]); } else parkBody(v, 'ekf');
    v.forward();
    drawMap(dt);
    const err = cur ? Math.hypot(truth[0] - cur.m[0], truth[1] - cur.m[1]) : 0;
    const info = `t <b>${(k * H).toFixed(2)} s</b> · spread <b>${cur ? cur.spread.toFixed(2) : '-'} m</b> · true cart to mean <b>${err.toFixed(2)} m</b> · effective sample <b>${cur ? Math.round(cur.ess) : '-'}</b> of ${N_POOL} simulated (2000 drawn)`;
    if (info !== lastInfo) { readout(panel, 'info', info); lastInfo = info; }
  });

  startRun(true);

  /** Test hook: finish the drive and the reading animations now. */
  panel._skip = () => {
    while (k < N_STEPS) stepAll(H);
    lev = [mode >= 1 ? 2 : 0, mode >= 2 ? 2 : 0];
    refresh();
    return panel._state();
  };
  panel._state = () => ({
    run, DT: D * T, t: k * H, driving: k < N_STEPS, mode, lev: lev.slice(), z: z.slice(), gauss: gaussOn,
    truth: Array.from(truth), mean: cur && cur.m, covXY: cur && [cur.P[0][0], cur.P[0][1], cur.P[1][1]],
    spread: { command: bars[0], one: bars[1], two: bars[2] }, ess: cur && cur.ess, nPool: N_POOL, nShow: N_SHOW,
    errMean: cur && Math.hypot(truth[0] - cur.m[0], truth[1] - cur.m[1]),
    meanXTheory: S.meanForwardDistance(V, D, T),
    view: view && view.map((x) => +x.toFixed(3)),
  });
}
