// 3.1 Controllability: a gantry with a twin hoist (two independent pendulums on one trolley).
// The robot is the nonlinear pair of pendulums driven by the trolley acceleration u. Next to it the same pair
// is simulated with the trolley parked, to show what the difference th1 - th2 would have done on its own.
// Inputs are designed on the linear model: rank of [B AB ... A^5 B], and the minimum-energy input to rest.
import * as S from '../sim.js';
import { makeViewer, CSS, val, ctl, onInput, onClick, readout, plot2d, stepper, deg, rad } from '../common.js';
import * as GC from '../gantry_common.js';

const L1 = 1.0;                 // first cable (m); the second one is on the slider
const ZETA = 0.01;              // light friction of each hoist (damping ratio)
const AMP = rad(8);             // release: +AMP and -AMP
const T_REST = 5.0;             // duration of the input that brings everything to rest (s)
const U_MAX = 1.5;              // the trolley motor: largest acceleration it can give (m/s^2)
const P_MAX = 1.5;              // usable half-length of the rail (m)
const H = 1e-3;                 // integration step (s)
const WINDOW = 12;              // seconds shown in the trace
const COARSE = 0.025;           // time step of the quick look-ahead used for the live energy and peak
const MOTIONS = {               // slider targets over time: [seconds after the click, position]
  m1: [[0, 0.4], [2.5, -0.4], [5.5, 0]],
  m2: [[0, -0.5], [2.5, 0.5], [5.5, 0]],
  m3: [[0, 1.0], [5, -1.0], [11, 0]],
};

/** Linear model for a second cable l2: rank, Gramian, and b(t) = e^{A (T - t)} B on a coarse time grid. */
function design(l2) {
  const sys = S.withTrolley(S.twinPendulums(L1, l2, ZETA));
  const d = { l2, sys, rank: S.rank(S.ctrb(sys.A, sys.B)) };
  if (d.rank === 6) {
    d.W = S.gramian(sys.A, sys.B, T_REST, 500);
    d.eAT = S.expm(S.scale(sys.A, T_REST));
    d.rows = rowsOf(sys, COARSE);
  }
  return d;
}

function rowsOf(sys, dt) {
  const N = Math.round(T_REST / dt), back = S.expm(S.scale(sys.A, dt)), rows = new Array(N + 1);
  let b = [...sys.B];
  for (let k = N; k >= 0; k--) { rows[k] = b; b = S.mulVec(back, b); }
  return rows;
}

/** Minimum-energy input from x0 to rest at the origin: u_k = b(t_k) . lambda, with its energy, peak and trolley range. */
function plan(d, x0, rows = d.rows) {
  const N = rows.length - 1, dt = T_REST / N;
  const lam = S.solve(d.W, S.mulVec(d.eAT, x0).map((v) => -v));
  const us = new Float64Array(N + 1);
  let peak = 0, p = x0[0], v = x0[1], pmin = p, pmax = p;
  for (let k = 0; k <= N; k++) {
    us[k] = S.dot(rows[k], lam);
    peak = Math.max(peak, Math.abs(us[k]));
    if (k > 0) { [p, v] = GC.trolleyStep(p, v, us[k - 1], us[k], dt); pmin = Math.min(pmin, p); pmax = Math.max(pmax, p); }
  }
  return { us, energy: S.dot(lam, S.mulVec(d.W, lam)), peak, pmin, pmax };
}

export async function init(panel) {
  const v = await makeViewer(panel, 'gantry.mjb', { lookat: [0, 0, 1.68], distance: 4.5, azimuth: 80, elevation: -4 });
  const rig = new GC.TwinRig(v.scene);
  const plot = plot2d(panel.querySelector('canvas.plot'));
  const pane1 = new GC.Pane(plot), pane2 = new GC.Pane(plot);
  const advance = stepper(H);
  const rec = new GC.Record(2, WINDOW);
  const w1 = Math.sqrt(S.G / L1);
  const xRelease = [0, 0, AMP, -AMP, 0, 0];

  // energy and peak of the stopping input from the release state, against the second cable's length
  const curve = [];
  for (let i = 0; i <= 80; i++) {
    const l = 0.6 + 0.01 * i;
    if (Math.abs(l - L1) < 1e-9) { curve.push({ l, energy: Infinity, peak: Infinity }); continue; }
    const pl = plan(design(l), xRelease);
    curve.push({ l, energy: pl.energy, peak: pl.peak });
  }

  let l2, des, w2;
  let t = 0, nstep = 0, p = 0, vel = 0, u = 0, target = 0;
  let xa, xb, ra, rb;                       // robot and parked reference: [angle, rate] of each load
  let mode = 'free', script = null, rest = null, lastRest = null, gapMax = 0, gapOpen = true, live = null;
  let sliderDirty = false, syncing = false, liveClock = 0;

  const state6 = () => [p, vel, xa[0], xb[0], xa[1], xb[1]];
  const swingAmp = () => Math.max(Math.hypot(xa[0], xa[1] / w1), Math.hypot(xb[0], xb[1] / w2));

  function release() {
    t = 0; nstep = 0;
    xa = [AMP, 0]; xb = [-AMP, 0]; ra = [AMP, 0]; rb = [-AMP, 0];
    gapMax = 0; gapOpen = true; lastRest = null;
    rec.clear();
    rec.push(0, deg(2 * AMP), deg(2 * AMP));
  }

  function setLength() {
    l2 = val(panel, 'l2');
    w2 = Math.sqrt(S.G / l2);
    des = design(l2);
    rig.b.setLength(l2);
    if (mode === 'rest') endRest(false);
    release();
    live = lookAhead();
  }

  /** What the stopping input from the present state would cost, and whether the trolley can deliver it. */
  function lookAhead() {
    if (des.rank < 6) return { ok: false, why: 'rank' };
    const pl = plan(des, state6());
    const why = pl.peak > U_MAX ? 'peak' : (pl.pmin < -P_MAX || pl.pmax > P_MAX) ? 'rail' : '';
    return { ok: !why, why, energy: pl.energy, peak: pl.peak, pmin: pl.pmin, pmax: pl.pmax };
  }

  function startRest() {
    if (mode === 'rest') return;
    live = lookAhead();
    if (!live.ok) return;
    const x0 = state6();
    const pl = plan(des, x0, rowsOf(des.sys, H));
    rest = { us: pl.us, k: 0, x0, energy: pl.energy, peak: pl.peak, pmin: pl.pmin, pmax: pl.pmax };
    mode = 'rest'; script = null; gapOpen = false; lastRest = null;
    u = rest.us[0];
    ctl(panel, 'pos').disabled = true;
  }

  function endRest(finished) {
    if (finished) {
      lastRest = { x0: rest.x0, energy: rest.energy, peak: rest.peak, residual_deg: deg(swingAmp()), p: p, v: vel,
        th_deg: [deg(xa[0]), deg(xb[0])], w_deg_s: [deg(xa[1]), deg(xb[1])] };
    }
    mode = 'free'; rest = null; target = finished ? 0 : p; sliderDirty = true;
    ctl(panel, 'pos').disabled = false;
  }

  function step(h) {
    let u1;
    if (mode === 'rest') {
      u1 = rest.us[rest.k + 1];
    } else {
      while (script && t - script.t0 >= script.items[script.i][0]) {
        target = script.items[script.i][1]; sliderDirty = true;
        if (++script.i >= script.items.length) script = null;
      }
      u1 = GC.followerAccel(target, p, vel, u, h);
    }
    xa = S.pendStep(w1, ZETA, xa, u, u1, h);
    xb = S.pendStep(w2, ZETA, xb, u, u1, h);
    ra = S.pendStep(w1, ZETA, ra, 0, 0, h);
    rb = S.pendStep(w2, ZETA, rb, 0, 0, h);
    [p, vel] = GC.trolleyStep(p, vel, u, u1, h);
    u = u1; t += h; nstep++;
    if (mode === 'rest' && ++rest.k >= rest.us.length - 1) endRest(true);
    const d = deg(xa[0] - xb[0]), d0 = deg(ra[0] - rb[0]);
    if (gapOpen) gapMax = Math.max(gapMax, Math.abs(d - d0));
    if (nstep % 20 === 0) rec.push(t, d, d0);
  }

  // ---------------------------------------------------------------- controls
  onInput(panel, 'pos', () => {
    if (syncing) return;
    script = null; lastRest = null;
    if (mode !== 'rest') target = val(panel, 'pos');
  });
  for (const key of Object.keys(MOTIONS)) {
    onClick(panel, key, () => { if (mode !== 'rest') { script = { t0: t, items: MOTIONS[key], i: 0 }; lastRest = null; } });
  }
  onInput(panel, 'l2', setLength);
  onClick(panel, 'swing', () => { if (mode === 'rest') endRest(false); release(); });
  onClick(panel, 'rest', startRest);
  setLength();

  function syncSlider() {
    if (!sliderDirty) return;
    sliderDirty = false;
    const el = ctl(panel, 'pos');
    el.value = target;
    syncing = true; el.dispatchEvent(new Event('input', { bubbles: true })); syncing = false;
  }

  // ---------------------------------------------------------------- text
  const f2 = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2));
  function text() {
    const full = des.rank === 6;
    GC.hud(panel, 'rank', `cables ${L1.toFixed(2)} m and ${l2.toFixed(2)} m<br>controllability rank <b class="big ${full ? 'g-ok' : 'g-bad'}">${des.rank} of 6</b>`);
    const btn = ctl(panel, 'rest');
    btn.disabled = mode === 'rest' || !live || !live.ok;
    let s;
    if (mode === 'rest') {
      s = `steering to rest: <b>${Math.max(0, T_REST - rest.k * H).toFixed(1)} s</b> left · energy <b>${f2(rest.energy)}</b> m²/s³ · peak <b>${f2(rest.peak)}</b> m/s²`;
    } else if (lastRest) {
      s = `<span class="g-ok">at rest at the origin</span> · energy spent <b>${f2(lastRest.energy)}</b> m²/s³ · swing left on the robot <b>${lastRest.residual_deg.toFixed(1)}°</b> (the input was planned on the linear model)`;
    } else if (!full) {
      s = `<span class="g-bad">cannot be brought to rest:</span> rank ${des.rank} of 6, no trolley input reaches the opposite swing`;
    } else if (live.why === 'peak') {
      s = `to rest in ${T_REST} s: energy <b>${f2(live.energy)}</b> m²/s³, peak <b class="g-bad">${f2(live.peak)}</b> m/s² · <span class="g-bad">more than the motor's ${U_MAX} m/s²</span>`;
    } else if (live.why === 'rail') {
      s = `to rest in ${T_REST} s: energy <b>${f2(live.energy)}</b> m²/s³ · <span class="g-bad">the trolley would run off the rail</span>`;
    } else {
      s = `to rest in ${T_REST} s from now: energy <b>${f2(live.energy)}</b> m²/s³, peak <b>${f2(live.peak)}</b> m/s²`;
    }
    readout(panel, 'info', s);
    const d = deg(xa[0] - xb[0]), d0 = deg(ra[0] - rb[0]);
    const sg = (x) => (x < 0 ? '−' : '+') + Math.abs(x).toFixed(1) + '°';
    GC.hud(panel, 'note', `θ₁ − θ₂ <b>${sg(d)}</b> · parked <b>${sg(d0)}</b>` + (gapOpen ? ` · largest gap <b>${gapMax.toFixed(1)}°</b>` : ''));
  }

  // ---------------------------------------------------------------- 2D picture
  function draw() {
    plot.clear();
    const g = plot.g, W = plot.w, Hh = plot.h, left = 40, right = 12;
    // trace of the difference
    const h1 = Math.round(Hh * 0.43);
    const lim = GC.niceLimit(1.08 * Math.max(deg(2 * AMP), rec.maxAbs(0), rec.maxAbs(1)), [20, 30, 40, 60, 90, 120, 180]);
    const tEnd = Math.max(t, WINDOW);
    pane1.set(left, 28, W - left - right, h1, tEnd - WINDOW, tEnd, -lim, lim);
    pane1.frame({ title: 'difference of the two angles, θ₁ − θ₂', yticks: [-lim, -lim / 2, 0, lim / 2, lim], fmt: (x) => x + '°' });
    pane1.line(rec.t, rec.s[1], CSS.grey, 6, { alpha: 0.5 });
    pane1.line(rec.t, rec.s[0], CSS.yellow, 1.8);
    const ly = 28 + h1 + 18;
    let lx = GC.legend(g, left, ly, CSS.yellow, 'robot');
    GC.legend(g, lx, ly, CSS.grey, 'the same loads, trolley parked', { width: 6, alpha: 0.5 });
    GC.label(g, `last ${WINDOW} s`, W - right, ly, CSS.faint, { align: 'right' });

    // energy of the stopping input against the second cable's length (log scale)
    const y2 = ly + 38, h2 = Hh - y2 - 36;
    const LO = -2, HI = 4, lg = (e) => Math.max(LO, Math.min(HI + 0.3, Math.log10(e)));
    pane2.set(left, y2, W - left - right, h2, 0.6, 1.4, LO, HI);
    // lengths at which the stopping input from the release state needs more than the motor has
    g.save(); g.fillStyle = 'rgba(252, 98, 85, 0.13)';
    for (let i = 0; i < curve.length; i++) {
      if (curve[i].peak > U_MAX) g.fillRect(pane2.X(curve[i].l - 0.005), y2, pane2.X(curve[i].l + 0.005) - pane2.X(curve[i].l - 0.005), h2);
    }
    g.restore();
    pane2.frame({ title: `energy ∫u² dt to stop the release swing in ${T_REST} s`, yticks: [-2, 0, 2, 4],
      fmt: (x) => ({ '-2': '0.01', 0: '1', 2: '100', 4: '10⁴' })[x], xticks: [0.6, 0.8, 1.0, 1.2, 1.4], xfmt: (x) => x.toFixed(1),
      zero: false, xlabel: 'second cable (m)' });
    for (const [a, b] of [[0, 40], [41, 81]]) {
      pane2.line(curve.slice(a, b).map((c) => c.l), curve.slice(a, b).map((c) => lg(c.energy)), CSS.accent, 2);
    }
    pane2.vline(L1, CSS.red, { alpha: 0.9 });
    GC.label(g, '∞', pane2.X(L1) + 5, y2 + 13, CSS.red, { font: GC.FONT_B });
    GC.label(g, 'red band: beyond the motor', left + 5, y2 + h2 - 6, CSS.red);
    const i = Math.max(0, Math.min(80, Math.round((l2 - 0.6) / 0.01))), c = curve[i];
    if (Number.isFinite(c.energy)) {
      pane2.marker(c.l, lg(c.energy), CSS.white, 4);
      const tx = pane2.X(c.l), ty = pane2.Y(lg(c.energy));
      GC.label(g, f2(c.energy), tx + (c.l > 1.25 ? -8 : 8), ty - 7, CSS.white, { align: c.l > 1.25 ? 'right' : 'left', font: GC.FONT_B });
    } else {
      pane2.marker(L1, HI, CSS.red, 4);
    }
  }

  v.onFrame((dt) => {
    advance(dt, 1, step);
    syncSlider();
    liveClock += dt;
    if (liveClock > 0.2 && mode !== 'rest') { liveClock = 0; live = lookAhead(); }
    v.setMocap('trolley', [p, 0, GC.PIV_TWIN]);
    rig.setX(p);
    rig.a.setAngle(xa[0]);
    rig.b.setAngle(xb[0]);
    v.forward();
    text();
    draw();
  });

  // numbers for the checks; _run(seconds) advances the simulation without drawing
  panel._run = (sec) => { const n = Math.round(sec / H); for (let i = 0; i < n; i++) step(H); syncSlider(); live = mode === 'rest' ? live : lookAhead(); return panel._state(); };
  panel._state = () => ({
    t, l1: L1, l2, zeta: ZETA, rank: des.rank, mode, target, x: state6(), u,
    diff_deg: deg(xa[0] - xb[0]), diff_parked_deg: deg(ra[0] - rb[0]), gap_max_deg: gapMax,
    rest_now: live, release_cost: curve[Math.round((l2 - 0.6) / 0.01)], last_rest: lastRest,
    T_rest: T_REST, u_max: U_MAX,
  });
  panel._ranks = () => curve.map((c) => [c.l, S.rank(S.ctrb(...Object.values(S.withTrolley(S.twinPendulums(L1, c.l, ZETA)))))]);
}
