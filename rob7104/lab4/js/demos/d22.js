// 2.2 Resonance: the drone is shaken sideways, x(t) = X sin(phase), phase' = w. The forcing of the load is
// the drone's acceleration a = x'' in M th'' + C th' + K th = -h a.
// Solid load = nonlinear robot (dpStep). Blue halo = steady state of the linear model at the current frequency.
// Right: sideways travel (world frame) of parcel and crate in the linear steady state against frequency,
// with dots for what the robot is doing now (half the peak-to-peak travel over the last shake period).
import * as S from '../sim.js';
import { makeViewer, THREE, COL, CSS, val, onInput, onClick, readout, ctl, plot2d, stepper, deg } from '../common.js';
import { setChain, showChain, haloChain, freqResp, cabs, Frame, runWhenVisible, FONT } from '../drone_util.js';

const X = 0.03;                            // shake amplitude (m)
const D1 = 4.0, D2 = 0.1;                  // hinge friction: damping ratios 0.074 (mode 1) and 0.097 (mode 2)
const TAU = 0.3;                           // the shake frequency follows the slider with this time constant (s)
const Z0 = 2.35, SAMPLE = 0.005;

export async function init(panel) {
  const v = await makeViewer(panel, 'drone.mjb', { lookat: [0, 0, 1.56], distance: 3.7, azimuth: 84, elevation: -7 });
  runWhenVisible(panel, v);
  showChain(v, 'g2', false);
  haloChain(v, 'g1', COL.blue);
  const plumb = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0.03, 0), new THREE.Vector3(0, 0.03, Z0 + 0.45)]),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.28 }));
  v.scene.add(plumb);                       // fixed in the world, through the drone's rest position
  const zr = Z0 - 0.16, tick = 0.035;       // the shake range, fixed in the world: a bar from -X to +X with end ticks
  const range = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints([
    [-X, zr], [X, zr], [-X, zr - tick], [-X, zr + tick], [X, zr - tick], [X, zr + tick]].map(([px, pz]) => new THREE.Vector3(px, -0.02, pz))),
  new THREE.LineBasicMaterial({ color: COL.accent }));
  v.scene.add(range);

  const mats = S.dpMatrices(S.P, D1, D2);
  const { w: wn, V } = S.modes(mats);
  const shape = V.map((m) => { const s = Math.max(Math.abs(m[0]), Math.abs(m[1])); return m.map((q) => q / s); });
  const det = shape[0][0] * shape[1][1] - shape[0][1] * shape[1][0];
  /** How much of each mode is in the link angles th (largest link angle of that mode's part, radians). */
  const content = (th) => [(th[0] * shape[1][1] - th[1] * shape[1][0]) / det, (th[1] * shape[0][0] - th[0] * shape[0][1]) / det];
  /** Linear steady state at frequency w: complex link angles and sideways travel of crate and parcel, per metre. */
  function linear(w) {
    const Th = freqResp(mats, w);
    const crate = [1 + S.P.l1 * Th[0][0], S.P.l1 * Th[0][1]];
    const parcel = [crate[0] + S.P.l2 * Th[1][0], crate[1] + S.P.l2 * Th[1][1]];
    return { Th, crate: cabs(crate), parcel: cabs(parcel) };
  }
  const slider = ctl(panel, 'r');
  const R0 = Number(slider.min), R1 = Number(slider.max);
  const rs = Array.from({ length: 501 }, (_, k) => R0 + (R1 - R0) * k / 500);
  const lin = rs.map((r) => linear(r * wn[0]));
  const crateCm = lin.map((l) => 100 * X * l.crate), parcelCm = lin.map((l) => 100 * X * l.parcel);
  let rStill = 1.2, best = Infinity;        // where the crate moves least, between the two modes
  for (let r = 1.05; r < wn[1] / wn[0] - 0.05; r += 0.0005) { const c = linear(r * wn[0]).crate; if (c < best) { best = c; rStill = r; } }
  const JUMP = { m1: 1, still: Math.round(rStill * 1000) / 1000, m2: Math.round(wn[1] / wn[0] * 1000) / 1000 };

  const advance = stepper(1e-3);
  const plot = plot2d(panel.querySelector('canvas.plot'));
  let x = [0, 0, 0, 0], t = 0, phase = 0, wt = val(panel, 'r') * wn[0], ws = wt, buf = [], nextSample = 0, meas = null;

  const accel = () => X * ((wt - ws) / TAU * Math.cos(phase) - ws * ws * Math.sin(phase));
  function step(h) {
    const a0 = accel();
    const w1 = wt + (ws - wt) * Math.exp(-h / TAU);
    phase += h * (ws + w1) / 2;
    if (phase > 2 * Math.PI) phase -= 2 * Math.PI;
    ws = w1;
    x = S.dpStep(S.P, D1, D2, x, a0, accel(), h);
    t += h;
    if (t >= nextSample) {
      nextSample += SAMPLE;
      const xc = X * Math.sin(phase) + S.P.l1 * Math.sin(x[0]), c = content(x);
      buf.push([t, xc, xc + S.P.l2 * Math.sin(x[1]), c[0], c[1]]);
    }
  }
  /** Half the peak-to-peak value of each recorded signal over the last shake period. */
  function measure() {
    const from = t - 2 * Math.PI / ws;
    while (buf.length > 2 && buf[0][0] < from) buf.shift();
    const lo = [0, 1e9, 1e9, 1e9, 1e9], hi = [0, -1e9, -1e9, -1e9, -1e9];
    for (const s of buf) for (let k = 1; k < 5; k++) { if (s[k] < lo[k]) lo[k] = s[k]; if (s[k] > hi[k]) hi[k] = s[k]; }
    const amp = (k) => (buf.length ? (hi[k] - lo[k]) / 2 : 0);
    meas = { crate: amp(1), parcel: amp(2), mode: [amp(3), amp(4)] };
  }

  const buttons = [...ctl(panel, 'jump').querySelectorAll('button')];
  function setFrequency() {
    const r = val(panel, 'r');
    wt = r * wn[0];
    buttons.forEach((b) => b.classList.toggle('on', Math.abs(JUMP[b.dataset.v] - r) < 0.0015));
  }
  onInput(panel, 'r', setFrequency);
  buttons.forEach((b) => b.addEventListener('click', () => {
    slider.value = JUMP[b.dataset.v];
    slider.dispatchEvent(new Event('input', { bubbles: true }));
  }));
  const rnd = (a) => (2 * Math.random() - 1) * a;
  onClick(panel, 'random', () => { x = [rnd(0.5), rnd(0.5), rnd(1.5), rnd(1.5)]; });
  setFrequency();

  const yMax = 1.1 * Math.max(...parcelCm);
  const MARK = [[JUMP.m1, 'mode 1', CSS.green, 13], [JUMP.still, 'crate still', CSS.blue, 26], [JUMP.m2, 'mode 2', CSS.red, 13]];
  function draw(now) {
    plot.clear();
    const g = plot.g;
    const f = new Frame(g, 48, 36, plot.w - 48 - 14, plot.h - 36 - 46, [R0, R1], [0, yMax]);
    for (const [r, name, col, py] of MARK) {
      f.seg(r, 0, r, yMax, col, 1, [3, 3]);
      f.text(name, f.X(r), py, col, 'center', 'middle');
    }
    const yt = [];
    for (let y = 0; y <= yMax; y += 10) yt.push([y, String(y)]);
    f.axes([[0.5, '0.5'], [1, '1'], [1.5, '1.5'], [2, '2']], yt, 'shake frequency / ω₁', 'sideways travel (± cm)');
    f.seg(R0, 100 * X, R1, 100 * X, CSS.grey, 1.2, [5, 4]);
    f.clip(() => {
      f.line(rs, parcelCm, CSS.yellow, 2);
      f.line(rs, crateCm, CSS.white, 2);
      const r = ws / wn[0];
      f.seg(r, 0, r, yMax, CSS.accent, 1.2);
      f.dot(r, 100 * meas.parcel, 5, CSS.yellow, '#000', 1.2);
      f.dot(r, 100 * meas.crate, 5, CSS.white, '#000', 1.2);
    });
    // legend, in the empty upper right corner
    const lx = f.X(1.72), ly = f.top + 14;
    const row = (k, col, dash, name) => {
      g.save(); g.strokeStyle = col; g.lineWidth = 2; g.setLineDash(dash);
      g.beginPath(); g.moveTo(lx, ly + 16 * k); g.lineTo(lx + 20, ly + 16 * k); g.stroke(); g.restore();
      f.text(name, lx + 26, ly + 16 * k, col, 'left', 'middle');
    };
    row(0, CSS.yellow, [], 'parcel');
    row(1, CSS.white, [], 'crate');
    row(2, CSS.grey, [5, 4], `drone, ±${(100 * X).toFixed(0)} cm`);
    f.text('lines: linear steady state', lx, ly + 16 * 3 + 4, CSS.grey, 'left', 'middle');
    g.beginPath(); g.arc(lx + 4, ly + 16 * 4 + 6, 4, 0, 2 * Math.PI); g.fillStyle = CSS.grey; g.fill();
    f.text('dots: the robot now', lx + 13, ly + 16 * 4 + 6, CSS.grey, 'left', 'middle');
    void now;
  }

  v.onFrame((dt) => {
    advance(dt, 1, step);
    measure();
    const now = linear(ws);
    const s = Math.sin(phase), c = Math.cos(phase);
    v.setMocap('drone', [X * s, 0, Z0]);
    setChain(v, 'main', x);
    setChain(v, 'g1', now.Th.map((z) => X * (z[0] * s + z[1] * c)));
    v.forward();
    draw(now);
    readout(panel, 'info', `in the swing now: <span class="c-green">mode 1 <b>${deg(meas.mode[0]).toFixed(1)}°</b></span> · <span class="c-red">mode 2 <b>${deg(meas.mode[1]).toFixed(1)}°</b></span> · parcel travels <b>±${(100 * meas.parcel).toFixed(1)} cm</b>, linear model ±${(100 * X * now.parcel).toFixed(1)} cm`);
  });
  panel._state = () => {
    measure();
    const now = linear(ws);
    return { t, r: ws / wn[0], w: ws, X, d: [D1, D2], x, phase, xDrone: v.data.mocap_pos[0], wn, jump: JUMP,
      lin: { crate: X * now.crate, parcel: X * now.parcel, thDeg: now.Th.map((z) => deg(X * cabs(z))) },
      meas: { crate: meas.crate, parcel: meas.parcel, modeDeg: meas.mode.map(deg) } };
  };
  /** Test hook: jump to frequency ratio r (no smoothing) and integrate T seconds without drawing. */
  panel._run = (r, T) => {
    slider.value = r; slider.dispatchEvent(new Event('input', { bubbles: true }));
    wt = r * wn[0]; ws = wt;
    const end = t + T;
    while (t < end - 1e-9) step(1e-3);
    return panel._state();
  };
  void FONT;
}
