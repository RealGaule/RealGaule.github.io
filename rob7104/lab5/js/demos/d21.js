// 2.1 The cart with noisy wheels. N carts drive the same command (straight 1 m, or a quarter arc with 1 m of path)
// through the lab's SDE (sim5.js cartStep: Euler-Maruyama, 1 ms, the numerics of the clips). The solid white cart is
// run 0 (its wheels turn by the simulated, noisy wheel angles); the other runs are the blue cloud with trails.
// Right: heading error of the first runs inside +-2 sigma_theta, heading variance on the line 2 D r^2 t / l^2, and
// the distance of the mean position behind the command against theory (DT = 1, 4, 7 for reference).
import { S, THREE, COL, CSS, makeViewer, setCart, parkBody, loadGhost, CartCloud, Ribbon, RangeRing,
  val, onInput, onClick, seg, toggle, readout, plot2d, scales, drawCurve, fmt } from '../common.js';

const T = 1, H = 1e-3, NSTEP = S.roundHalfEven(T / H);       // 1000 steps of 1 ms, as cart.simulate
const STORE = 10, NS = NSTEP / STORE + 1;                     // statistics every 10 ms: 101 stored times
const TEVERY = 20, NTP = NSTEP / TEVERY + 1;                  // trail points every 20 ms: 51 per run
const MAX_N = 8000, MAX_TRAILS = 150, MAX_HEAD = 60;
const CMDS = { straight: { v: 1, omega: 0 }, arc: { v: 1, omega: Math.PI / 2 } };   // arc = arc_unit (radius 2/pi)
const REF_DT = [1, 4, 7];

const ts = Float64Array.from({ length: NS }, (_, j) => j * STORE * H);

/** Distance of the mean position behind the command at time t (theory, exact for a constant command). */
function behindTheory(c, D, t) {
  const E = S.meanPosition(c.v, c.omega, D, t);
  if (c.omega === 0) return c.v * t - E[0];
  const m = S.meanPath(c.v, c.omega, t);
  return Math.hypot(m[0] - E[0], m[1] - E[1]);
}

/** One flat mesh of dashes along the noise-free path of a command (the command line on the floor). */
function dashMesh(c, width = 0.014, dash = 0.035, gap = 0.03, z = 0.003) {
  const pos = [], idx = [];
  const pt = (t) => S.meanPath(c.v, c.omega, t);
  for (let s0 = 0; s0 < c.v * T - 1e-9; s0 += dash + gap) {
    const s1 = Math.min(s0 + dash, c.v * T), n = 6, base = pos.length / 3;
    for (let q = 0; q <= n; q++) {
      const p = pt((s0 + (s1 - s0) * q / n) / c.v), nx = -Math.sin(p[2]), ny = Math.cos(p[2]);
      pos.push(p[0] + nx * width / 2, p[1] + ny * width / 2, z, p[0] - nx * width / 2, p[1] - ny * width / 2, z);
      if (q) { const a = base + 2 * (q - 1); idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0xdedede, transparent: true, opacity: 0.85, side: THREE.DoubleSide, depthWrite: false }));
  m.frustumCulled = false;
  return m;
}

function niceStep(max) {
  for (const s of [0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2]) if (max / s <= 4.2) return s;
  return 5;
}
const ticksTo = (lo, hi) => { const s = niceStep(Math.max(Math.abs(lo), Math.abs(hi))), out = []; for (let y = Math.ceil(lo / s) * s; y <= hi + 1e-12; y += s) out.push(+y.toFixed(6)); return out; };
const digitsOf = (ticks) => Math.max(0, ...ticks.map((y) => { const s = String(y); return s.includes('.') ? s.split('.')[1].length : 0; }));

export async function init(panel) {
  const v = await makeViewer(panel, 'warehouse.mjb', { lookat: [0.48, 0.28, 0], distance: 2.6, azimuth: 68, elevation: -42 });
  parkBody(v, 'est');
  const geo = await loadGhost();
  const cloud = new CartCloud(v.scene, geo, { max: MAX_N, bodies: 30, opacity: 0.26, dot: 0.02 });

  // trails: one LineSegments buffer, time-major (block b = segments from trail point b to b + 1 of every trailed run)
  const tPos = new Float32Array(MAX_TRAILS * (NTP - 1) * 6), tCol = new Float32Array(MAX_TRAILS * (NTP - 1) * 6);
  const tGeo = new THREE.BufferGeometry();
  tGeo.setAttribute('position', new THREE.BufferAttribute(tPos, 3));
  tGeo.setAttribute('color', new THREE.BufferAttribute(tCol, 3));
  tGeo.setDrawRange(0, 0);
  const trails = new THREE.LineSegments(tGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55, depthWrite: false }));
  trails.frustumCulled = false;
  v.scene.add(trails);

  // floor marks: command line, start bar, end bar, the command point (white ring) and the mean of the runs (blue disc)
  let dashes = null;
  const startBar = new Ribbon(v.scene, 0xdedede, { n: 2, closed: false });
  const endBar = new Ribbon(v.scene, 0xdedede, { n: 2, closed: false, opacity: 0.7 });
  startBar.set([[0, -0.13], [0, 0.13]], 0.03);
  const cmdRing = new RangeRing(v.scene, COL.truth, { n: 48, z: 0.012, opacity: 0.95 });
  const meanDisc = new THREE.Mesh(new THREE.CircleGeometry(0.026, 40), new THREE.MeshBasicMaterial({ color: COL.cloud, transparent: true, opacity: 0.95, depthWrite: false }));
  meanDisc.renderOrder = 3;
  v.scene.add(meanDisc);
  const meanRim = new RangeRing(v.scene, COL.truth, { n: 48, z: 0.013, opacity: 0.9 });
  const link = new Ribbon(v.scene, COL.truth, { n: 2, closed: false, opacity: 0.6, z: 0.011 });
  const trail0 = new Ribbon(v.scene, COL.truth, { n: NTP, closed: false, opacity: 0.95, z: 0.010 }); trail0.mesh.material.depthTest = false; trail0.mesh.renderOrder = 9; const tr0 = [];
  // the markers stay visible through the translucent cloud
  [cmdRing.band.mesh, meanRim.band.mesh, link.mesh, meanDisc].forEach((m, q) => { m.material.depthTest = false; m.renderOrder = 10 + q; });

  // screen labels at the two markers (the clip's "command" and "mean")
  const layer = panel.querySelector('.d21-labels');
  const mkLabel = (text, color) => {
    const el = document.createElement('span');
    el.textContent = text;
    el.style.cssText = `position:absolute;left:0;top:0;padding:2px 7px;border-radius:4px;background:rgba(0,0,0,0.72);color:${color};white-space:nowrap;font-family:Inter,sans-serif;`;
    layer.appendChild(el);
    return el;
  };
  const lab = { cmd: mkLabel('command', CSS.white), mean: mkLabel('mean', CSS.cloud) };
  const pv = new THREE.Vector3();
  function placeLabel(el, x, y, dx, dy) {
    pv.set(x, y, 0.02).project(v.camera);
    const W = v.canvas.clientWidth, Hh = v.canvas.clientHeight;
    el.style.display = pv.z < 1 ? '' : 'none';
    el.style.transform = `translate(${((pv.x + 1) / 2 * W + dx).toFixed(1)}px, ${((1 - pv.y) / 2 * Hh + dy).toFixed(1)}px)`;
  }

  // controls
  let seed = 1, speed = 0.3, cmdName = 'straight', nCarts = 2000;
  const trailsOn = toggle(panel, 'trails', (on) => { trails.visible = on; trail0.visible = on; });
  seg(panel, 'cmd', (x) => { cmdName = x; restart(); });
  seg(panel, 'n', (x) => { nCarts = Number(x); restart(); });
  seg(panel, 'speed', (x) => { speed = Number(x); });
  onInput(panel, 'DT', () => restart());
  onClick(panel, 'go', () => { seed++; restart(); });

  const plotCanvas = panel.querySelector('canvas.plot');
  const o = plot2d(plotCanvas);
  let dirty = true;
  new ResizeObserver(() => { dirty = true; }).observe(plotCanvas);

  let run = null, prev = null, frames = 0, last = 0;
  const cb = new THREE.Color(COL.cloud), cw = new THREE.Color(COL.truth);   // linear values, as the renderer expects
  const BLUE_LIN = [cb.r, cb.g, cb.b], WHITE_LIN = [cw.r, cw.g, cw.b];

  function restart() {
    if (run && run.done && run.N >= 100) prev = { cmd: run.cmd, DT: run.DT, hv: run.hv };
    const c = CMDS[cmdName], DT = val(panel, 'DT'), N = nCarts;
    run = {
      cmd: cmdName, c, DT, D: DT / T, N, seed, k: 0, acc: 0, done: false,
      rng: new S.RNG(seed + (cmdName === 'arc' ? 100003 : 0)), pose: new Float64Array(3 * N), wheel: new Float64Array(2 * N),
      NT: Math.min(N, MAX_TRAILS), NH: Math.min(N, MAX_HEAD),
      mx: new Float64Array(NS), my: new Float64Array(NS), hv: new Float64Array(NS).fill(NaN),
      behind: new Float64Array(NS), bse: new Float64Array(NS).fill(NaN), head: new Float32Array(NS * Math.min(N, MAX_HEAD)),
      last: new Float32Array(2 * Math.min(N, MAX_TRAILS)), j: 0,
    };
    // theory curves for the 2D picture
    run.thVar = Float64Array.from(ts, (t) => S.headingVariance(run.D, t));
    run.thBehind = Float64Array.from(ts, (t) => behindTheory(c, run.D, t));
    run.refBehind = REF_DT.map((d) => Float64Array.from(ts, (t) => behindTheory(c, d / T, t)));
    // trail colours: run 0 white-blue and bright, the others blue
    for (let b = 0; b < NTP - 1; b++) for (let i = 0; i < run.NT; i++) {
      const col = BLUE_LIN;
      const base = (b * run.NT + i) * 6;
      tCol.set(col, base); tCol.set(col, base + 3);
    }
    tGeo.attributes.color.needsUpdate = true;
    tGeo.setDrawRange(0, 0);
    if (dashes) { v.scene.remove(dashes); dashes.geometry.dispose(); }
    dashes = dashMesh(c);
    v.scene.add(dashes);
    const e = S.meanPath(c.v, c.omega, T), nx = -Math.sin(e[2]), ny = Math.cos(e[2]);
    endBar.set([[e[0] - 0.09 * nx, e[1] - 0.09 * ny], [e[0] + 0.09 * nx, e[1] + 0.09 * ny]], 0.02);
    record(0);
    show();
    dirty = true;
  }

  // statistics over all N runs at stored time j (k = j * STORE)
  function record(j) {
    const r = run, N = r.N, p = r.pose, t = ts[j], w = r.c.omega * t;
    let sx = 0, sy = 0, se = 0;
    for (let i = 0; i < N; i++) { sx += p[3 * i]; sy += p[3 * i + 1]; se += p[3 * i + 2] - w; }
    const mx = sx / N, my = sy / N, me = se / N;
    let cxx = 0, cyy = 0, cxy = 0, cee = 0;
    for (let i = 0; i < N; i++) {
      const dx = p[3 * i] - mx, dy = p[3 * i + 1] - my, de = p[3 * i + 2] - w - me;
      cxx += dx * dx; cyy += dy * dy; cxy += dx * dy; cee += de * de;
    }
    r.mx[j] = mx; r.my[j] = my;
    if (N > 1) { cxx /= N - 1; cyy /= N - 1; cxy /= N - 1; r.hv[j] = cee / (N - 1); }
    const m = S.meanPath(r.c.v, r.c.omega, t);
    if (r.c.omega === 0) {
      r.behind[j] = r.c.v * t - mx;
      if (N > 1) r.bse[j] = Math.sqrt(cxx / N);
    } else {
      // signed component of (command - mean) along the direction from the theory mean to the command
      const E = S.meanPosition(r.c.v, r.c.omega, r.D, t), ex = m[0] - E[0], ey = m[1] - E[1], en = Math.hypot(ex, ey);
      const ux = en > 1e-15 ? ex / en : -Math.sin(m[2]), uy = en > 1e-15 ? ey / en : Math.cos(m[2]);
      r.behind[j] = ux * (m[0] - mx) + uy * (m[1] - my);
      if (N > 1) r.bse[j] = Math.sqrt((ux * ux * cxx + 2 * ux * uy * cxy + uy * uy * cyy) / N);
    }
    for (let i = 0; i < r.NH; i++) r.head[j * r.NH + i] = p[3 * i + 2] - w;
    r.j = j;
  }

  function recordTrail(b) {
    const r = run, p = r.pose;
    if (b === 0) tr0.length = 0; tr0.push([run.pose[0], run.pose[1]]); const pts = tr0.slice(); while (pts.length < NTP) pts.push(pts[pts.length - 1]); trail0.set(pts, 0.012);
    for (let i = 0; i < r.NT; i++) {
      const x = p[3 * i], y = p[3 * i + 1];
      if (b > 0) {
        const base = ((b - 1) * r.NT + i) * 6;
        tPos[base] = r.last[2 * i]; tPos[base + 1] = r.last[2 * i + 1]; tPos[base + 2] = 0.008;
        tPos[base + 3] = x; tPos[base + 4] = y; tPos[base + 5] = 0.008;
      }
      r.last[2 * i] = x; r.last[2 * i + 1] = y;
    }
    if (b > 0) { tGeo.attributes.position.needsUpdate = true; tGeo.setDrawRange(0, b * r.NT * 2); }
  }

  function step() {
    const r = run;
    S.cartStep(r.pose, r.c.v, r.c.omega, r.D, H, r.rng, { wheel: r.wheel });
    r.k++;
    if (r.k % STORE === 0) record(r.k / STORE);
    if (r.k % TEVERY === 0) recordTrail(r.k / TEVERY);
    if (r.k >= NSTEP) r.done = true;
  }

  // put the 3D picture in the state of the run
  function show() {
    const r = run, p = r.pose, N = r.N, t = r.k * H;
    if (r.k === 0) recordTrail(0);
    setCart(v, 'cart', [p[0], p[1], p[2]], [r.wheel[0], r.wheel[N]]);
    cloud.visible = N > 1;
    const many = N > 1; meanDisc.visible = many; meanRim.visible = many; link.visible = many;
    if (N > 1) cloud.set(p.subarray(3), N - 1);
    const m = S.meanPath(r.c.v, r.c.omega, t), j = r.j;
    cmdRing.set([m[0], m[1]], 0.042, 0.012);
    meanDisc.position.set(r.mx[j], r.my[j], 0.012);
    meanRim.set([r.mx[j], r.my[j]], 0.028, 0.007);
    link.set([[m[0], m[1]], [r.mx[j], r.my[j]]], 0.005);
    v.forward();
  }

  // the 2D picture: three plots over the run time, in step with the carts
  function draw() {
    const r = run, j = r.j;
    o.clear();
    const g = o.g, rowH = o.h / 3, left = 52, right = 70;
    const box = (row, yr) => scales(o, [0, T], yr, [left, right, row * rowH + 30, o.h - (row + 1) * rowH + 20]);
    const title = (s, text, formula) => {
      g.save(); g.fillStyle = CSS.white; g.font = '13px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'bottom';
      g.fillText(text, s.box[0], s.box[1] - 8);
      if (formula) { g.fillStyle = CSS.grey; g.font = 'italic 12px Inter, serif'; g.textAlign = 'right'; g.fillText(formula, s.box[0] + s.box[2] + right - 6, s.box[1] - 8); }
      g.restore();
    };
    const axes = (s, yt, xlab) => {
      const [l, t, W, Hh] = s.box, d = digitsOf(yt);
      g.save(); g.strokeStyle = CSS.faint; g.lineWidth = 1; g.fillStyle = CSS.grey; g.font = '11px Inter, sans-serif';
      g.beginPath(); g.moveTo(l, t); g.lineTo(l, t + Hh); g.lineTo(l + W, t + Hh); g.stroke();
      g.textAlign = 'right'; g.textBaseline = 'middle';
      yt.forEach((y) => { const py = s.Y(y); g.globalAlpha = 0.35; g.beginPath(); g.moveTo(l, py); g.lineTo(l + W, py); g.stroke(); g.globalAlpha = 1; g.fillText(y.toFixed(d), l - 6, py); });
      g.textAlign = 'center'; g.textBaseline = 'top';
      [0, 0.5, 1].forEach((x) => g.fillText(String(x), s.X(x), t + Hh + 4));
      if (xlab) { g.fillStyle = CSS.white; g.textAlign = 'left'; g.fillText('t (s)', l + W + 10, t + Hh + 4); }
      g.restore();
    };
    const tsj = ts.subarray(0, j + 1);
    const sig = Math.sqrt(S.headingVariance(r.D, T));

    // 1. heading error of the first runs, inside the +-2 sigma_theta envelope
    const ym = Math.max(0.05, 1.3 * 2 * sig);
    let s = box(0, [-ym, ym]);
    title(s, 'heading error (rad)', 'dθ = √D (r/l)(dW₁ − dW₂)');
    axes(s, ticksTo(-ym, ym));
    for (let i = r.NH - 1; i >= 0; i--) {
      const ys = new Float64Array(j + 1);
      for (let q = 0; q <= j; q++) ys[q] = r.head[q * r.NH + i];
      if (i === 0) drawCurve(o, s, tsj, ys, 'rgba(240,248,255,0.95)', 1.6);
      else drawCurve(o, s, tsj, ys, 'rgba(88,196,221,0.38)', 1);
    }
    const env = Float64Array.from(ts, (t) => 2 * Math.sqrt(S.headingVariance(r.D, t)));
    drawCurve(o, s, ts, env, CSS.white, 1.6);
    drawCurve(o, s, ts, env.map((x) => -x), CSS.white, 1.6);
    g.save(); g.fillStyle = CSS.white; g.font = '12px Inter, sans-serif'; g.textBaseline = 'middle'; g.fillText('±2σθ', s.X(1) + 6, s.Y(2 * sig)); g.restore();

    // 2. variance of the heading error: theory line and the average over all N runs
    const vmax = Math.max(0.01, 1.15 * S.headingVariance(r.D, T));
    s = box(1, [0, vmax]);
    title(s, 'heading variance (rad²)');
    axes(s, ticksTo(0, vmax));
    if (prev && prev.cmd !== r.cmd && prev.DT === r.DT) {
      g.save(); g.strokeStyle = CSS.grey; g.lineWidth = 1;
      for (let q = 0; q < NS; q += 4) { g.beginPath(); g.arc(s.X(ts[q]), s.Y(prev.hv[q]), 3.2, 0, 2 * Math.PI); g.stroke(); }
      g.restore();
    }
    drawCurve(o, s, ts, r.thVar, CSS.white, 1.8);
    g.save(); g.fillStyle = CSS.cloud;
    for (let q = 0; q <= j; q += 4) if (Number.isFinite(r.hv[q])) {
      if (r.hv[q] > vmax * 1.05) { const px = s.X(ts[q]); g.beginPath(); g.moveTo(px, s.Y(vmax*1.05) - 4); g.lineTo(px - 3.5, s.Y(vmax*1.05) + 2); g.lineTo(px + 3.5, s.Y(vmax*1.05) + 2); g.closePath(); g.fill(); }
      else { g.beginPath(); g.arc(s.X(ts[q]), s.Y(r.hv[q]), 2.8, 0, 2 * Math.PI); g.fill(); }
    }
    // legend
    const lx = s.box[0] + 10; let ly = s.box[1] + 26;
    g.fillStyle = CSS.grey; g.font = 'italic 12px Inter, serif'; g.textBaseline = 'middle'; g.fillText('σθ²(t) = 2Dr²t / l²', lx, s.box[1] + 8);
    g.font = '11.5px Inter, sans-serif'; g.textBaseline = 'middle';
    g.strokeStyle = CSS.white; g.lineWidth = 1.8; g.beginPath(); g.moveTo(lx, ly); g.lineTo(lx + 16, ly); g.stroke();
    g.fillStyle = CSS.white; g.fillText('theory', lx + 24, ly);
    ly += 16;
    if (r.N > 1) { g.fillStyle = CSS.cloud; g.beginPath(); g.arc(lx + 8, ly, 2.8, 0, 2 * Math.PI); g.fill(); g.fillText(`average of ${r.N} carts`, lx + 24, ly); }
    else { g.fillStyle = CSS.grey; g.fillText('one cart: no variance', lx + 24, ly); }
    if (prev && prev.cmd !== r.cmd && prev.DT === r.DT) {
      ly += 16; g.strokeStyle = CSS.grey; g.lineWidth = 1; g.beginPath(); g.arc(lx + 8, ly, 3.2, 0, 2 * Math.PI); g.stroke();
      g.fillStyle = CSS.grey; g.fillText(prev.cmd === 'straight' ? 'straight run' : 'quarter arc', lx + 24, ly);
    }
    g.restore();

    // 3. mean position behind the command: theory (DT = 1, 4, 7 faint) and the average over all N runs, +- 1 se
    const DTs = Math.max(r.DT, 7), bmax = Math.max(1e-3, 1.2 * behindTheory(r.c, DTs / T, T));
    s = box(2, [-0.08 * bmax, bmax]);
    title(s, 'mean behind the command (m)');
    axes(s, ticksTo(0, bmax), true);
    REF_DT.forEach((d, q) => {
      if (d === r.DT) return;
      drawCurve(o, s, ts, r.refBehind[q], 'rgba(154,151,163,0.55)', 1.2, [4, 4]);
      g.save(); g.fillStyle = CSS.grey; g.font = '11px Inter, sans-serif'; g.textBaseline = 'middle';
      g.fillText(`DT = ${d}`, s.X(1) + 6, s.Y(r.refBehind[q][NS - 1])); g.restore();
    });
    drawCurve(o, s, ts, r.thBehind, CSS.white, 1.8);
    g.save(); g.fillStyle = CSS.white; g.font = '11px Inter, sans-serif'; g.textBaseline = 'middle';
    g.fillText(`DT = ${r.DT}`, s.X(1) + 6, s.Y(r.thBehind[NS - 1])); g.restore();
    g.save(); const dc = r.N > 1 ? CSS.cloud : CSS.white; g.fillStyle = dc; g.strokeStyle = dc; g.lineWidth = 1;
    const clampY = (y) => Math.max(-0.08 * bmax, Math.min(bmax * 1.02, y));
    for (let q = 0; q <= j; q += 4) {
      const px = s.X(ts[q]);
      if (Number.isFinite(r.bse[q]) && r.bse[q] > 0) { g.beginPath(); g.moveTo(px, s.Y(clampY(r.behind[q] - r.bse[q]))); g.lineTo(px, s.Y(clampY(r.behind[q] + r.bse[q]))); g.stroke(); }
      if (r.behind[q] > bmax * 1.02) { g.beginPath(); g.moveTo(px, s.Y(bmax*1.02) - 4); g.lineTo(px - 3.5, s.Y(bmax*1.02) + 2); g.lineTo(px + 3.5, s.Y(bmax*1.02) + 2); g.closePath(); g.fill(); } else { g.beginPath(); g.arc(px, s.Y(clampY(r.behind[q])), 2.8, 0, 2 * Math.PI); g.fill(); }
    }
    g.restore();
    // numbers at the current time
    const th = r.thBehind[j], b = r.behind[j], se = r.bse[j];
    const lines = r.c.omega === 0
      ? [['mean forward distance (of 1 m)', CSS.white], [`theory   ${fmt(r.c.v * ts[j] - th, 3)} m`, CSS.white],
         [`carts     ${fmt(r.mx[j], 3)}${Number.isFinite(se) ? ` ± ${fmt(se, 3)}` : ''} m`, r.N > 1 ? CSS.cloud : CSS.white]]
      : [['behind the command', CSS.white], [`theory   ${fmt(th, 3)} m`, CSS.white],
         [`carts     ${fmt(b, 3)}${Number.isFinite(se) ? ` ± ${fmt(se, 3)}` : ''} m`, r.N > 1 ? CSS.cloud : CSS.white]];
    if (r.N === 1) lines.push(['(one run: no average)', CSS.grey]);
    lines.push(['E[cos θ] = e^(−σθ²/2) < 1', CSS.grey]);
    g.save(); g.font = '12px Inter, sans-serif'; g.textBaseline = 'top';
    g.fillStyle = 'rgba(0,0,0,0.75)';
    g.fillRect(s.box[0] + 4, s.box[1], Math.max(...lines.map(([txt]) => g.measureText(txt).width)) + 12, 16 * lines.length + 4);
    lines.forEach(([txt, col], q) => { g.fillStyle = col; g.fillText(txt, s.box[0] + 10, s.box[1] + 2 + 16 * q); });
    g.restore();
  }

  function info() {
    const r = run;
    readout(panel, 'info', `t <b>${fmt(r.k * H, 2)} s</b> · <b>${r.N}</b> cart${r.N > 1 ? 's' : ''} · σ<sub>θ</sub>(1 s) <b>${fmt(Math.sqrt(S.headingVariance(r.D, T)) * 180 / Math.PI, 0)}°</b>`);
  }

  v.onFrame(() => {
    frames++;
    const now = performance.now(), dt = last ? Math.min((now - last) / 1000, 0.25) : 0;
    last = now;
    const r = run;
    if (!r.done) {
      r.acc += dt * speed;
      const cap = Math.max(10, Math.floor(400000 / r.N));
      let n = 0;
      while (r.acc >= H && !r.done && n < cap) { step(); r.acc -= H; n++; }
      if (n === cap) r.acc = 0;
      if (n) { show(); dirty = true; }
    }
    if (dirty) { draw(); info(); dirty = false; }
    const j = r.j, m = S.meanPath(r.c.v, r.c.omega, r.k * H);
    placeLabel(lab.cmd, m[0], m[1], 10, -30);
    if (r.N > 1) { placeLabel(lab.mean, r.mx[j], r.my[j], 12, 10); lab.mean.textContent = `mean of ${r.N}`; } else lab.mean.style.display = 'none';
  });
  panel.addEventListener('panel:show', () => { dirty = true; last = 0; });

  restart();

  panel._state = () => {
    const r = run, j = r.j, t = ts[j];
    return {
      t: r.k * H, done: r.done, cmd: r.cmd, DT: r.DT, N: r.N, seed: r.seed, speed, trails: trailsOn(), frames,
      cart0: [r.pose[0], r.pose[1], r.pose[2]], wheel0: [r.wheel[0], r.wheel[r.N]],
      statsAt: t,
      headingVar: { theory: S.headingVariance(r.D, t), carts: r.hv[j] },
      meanXY: [r.mx[j], r.my[j]], meanXYTheory: S.meanPosition(r.c.v, r.c.omega, r.D, t),
      behind: { theory: r.thBehind[j], carts: r.behind[j], se: r.bse[j] },
      meanForward: r.c.omega === 0 ? { theory: S.meanForwardDistance(r.c.v, r.D, t), carts: r.mx[j] } : null,
      cloud: cloud.visible ? cloud.n : 0, trailSegments: tGeo.drawRange.count / 2,
    };
  };
}
