// 3.1b Observability: the twin hoist on a parked trolley, seen through one sensor, a load cell that measures
// the total horizontal pull of the two cables. Two releases are simulated side by side (blue load at 12 degrees,
// or yellow load at 12 degrees); a discrete Kalman filter sees only the noisy reading of the one on screen.
import * as S from '../sim.js';
import { makeViewer, COL, CSS, ctl, onClick, readout, seg, plot2d, stepper, deg, rad } from '../common.js';
import * as GC from '../gantry_common.js';

const L1 = 1.0, L2_UNEQUAL = 0.7, MASS = 1.0;
const REL = rad(12);            // release angle of the one load that is let go
const H = 1e-3;                 // integration step (s)
const TS = 0.02;                // sensor sample period (s)
const SIG_R = 0.3;              // sensor noise (N, std per sample)
const SIG_A = 0.1;              // process noise of the filter: unknown trolley jitter (m/s^2, std per sample)
const P0 = rad(15);             // initial uncertainty of each angle; rates: P0 * omega
const WINDOW = 8;               // seconds shown in the strips
const LATE = 15;                // the steady error is averaged from this time on (s)
const EVERY = Math.round(TS / H);

export async function init(panel) {
  const v = await makeViewer(panel, 'gantry.mjb', { lookat: [0, 0, 1.62], distance: 3.5, azimuth: 68, elevation: -5 });
  const rig = new GC.TwinRig(v.scene).addGhost().addLoadCell();
  const plot = plot2d(panel.querySelector('canvas.plot'));
  const pane1 = new GC.Pane(plot), pane2 = new GC.Pane(plot);
  const advance = stepper(H);
  const rec = new GC.Record(6, WINDOW);         // reading, other reading, noisy sample, filter's fit, gap, filter error
  const Hrow = [S.G * MASS, S.G * MASS, 0, 0];
  const w1 = Math.sqrt(S.G / L1);

  let cables = 'equal', start = 'a', hide = false, seed = 0;
  let l2, w2, rank, t, nstep, worlds, kf, noise, stats;

  const reading = (wd) => GC.cableForceX(L1, MASS, wd.a[0], wd.a[1]) + GC.cableForceX(l2, MASS, wd.b[0], wd.b[1]);
  const cur = () => worlds[start === 'a' ? 0 : 1];
  const other = () => worlds[start === 'a' ? 1 : 0];

  function sample() {
    const y = reading(cur()), yo = reading(other());
    const z = y + SIG_R * noise();
    const xh = kf.update(z);
    const c = cur();
    const e = [xh[0] - c.a[0], xh[1] - c.b[0], xh[2] - c.a[1], xh[3] - c.b[1]];
    const amp = deg(Math.max(Math.hypot(e[0], e[2] / w1), Math.hypot(e[1], e[3] / w2)));
    stats.y = y; stats.yo = yo; stats.z = z; stats.err = amp; stats.e_deg = [deg(e[0]), deg(e[1])];
    stats.gapMax = Math.max(stats.gapMax, Math.abs(y - yo));
    stats.yMax = Math.max(stats.yMax, Math.abs(y));
    if (t >= LATE - 1e-9) { stats.sq += e[0] * e[0] + e[1] * e[1]; stats.n += 2; stats.errLateMax = Math.max(stats.errLateMax, amp); }
    rec.push(t, y, yo, z, S.dot(Hrow, xh), y - yo, amp);
  }

  function release() {
    l2 = cables === 'equal' ? L1 : L2_UNEQUAL;
    w2 = Math.sqrt(S.G / l2);
    const A = S.twinPendulums(L1, l2).A, B = S.twinPendulums(L1, l2).B;
    rank = S.rank(GC.obsv(A, Hrow));
    t = 0; nstep = 0;
    worlds = [{ a: [REL, 0], b: [0, 0] }, { a: [0, 0], b: [REL, 0] }];
    seed += 1;
    noise = GC.gaussSource(seed);
    kf = new GC.Kalman(A, B, Hrow, TS, SIG_A, SIG_R, [0, 0, 0, 0], [P0 * P0, P0 * P0, (P0 * w1) ** 2, (P0 * w2) ** 2]);
    stats = { gapMax: 0, yMax: 0, sq: 0, n: 0, errLateMax: 0 };
    rec.clear();
    rig.b.setLength(l2); rig.gb.setLength(l2);
    look();
    sample();
  }

  function step(h) {
    for (const wd of worlds) {
      wd.a = S.pendStep(w1, 0, wd.a, 0, 0, h);
      wd.b = S.pendStep(w2, 0, wd.b, 0, 0, h);
    }
    t += h; nstep++;
    if (nstep % EVERY === 0) sample();
  }

  function look() {
    if (hide) {
      rig.a.setLook(GC.DIM, 1, 0.3); rig.b.setLook(GC.DIM, 1, 0.3);
      rig.ga.setLook(COL.green, 0.55, 0.5); rig.gb.setLook(COL.green, 0.55, 0.5);
    } else {
      rig.a.setLook(COL.blue, 1); rig.b.setLook(COL.yellow, 1);
      rig.ga.setLook(COL.green, 0); rig.gb.setLook(COL.green, 0);
    }
    ctl(panel, 'hide').classList.toggle('on', hide);
  }

  seg(panel, 'cables', (x) => { cables = x; release(); });
  seg(panel, 'start', (x) => { start = x; release(); });
  onClick(panel, 'hide', () => { hide = !hide; release(); });
  onClick(panel, 'again', release);
  release();

  const sg = (x, d = 1) => (x < 0 ? '−' : '+') + Math.abs(x).toFixed(d);
  function text() {
    const full = rank === 4;
    GC.hud(panel, 'rank', `cables ${L1.toFixed(1)} m and ${l2.toFixed(1)} m<br>observability rank <b class="big ${full ? 'g-ok' : 'g-bad'}">${rank} of 4</b>`);
    const c = cur();
    if (hide) {
      GC.hud(panel, 'note', 'dark: the hidden loads · <span class="g-ok">green: the filter\'s estimate</span>');
      const good = stats.err < 1;
      readout(panel, 'info', `filter error now <b class="${good ? 'g-ok' : 'g-bad'}">${stats.err.toFixed(1)}°</b> · estimate <b>${sg(deg(kf.x[0]))}°</b> and <b>${sg(deg(kf.x[1]))}°</b> · hidden truth <b>${sg(deg(c.a[0]))}°</b> and <b>${sg(deg(c.b[0]))}°</b> · ${t.toFixed(1)} s since release`);
    } else {
      GC.hud(panel, 'note', '<span class="g-ok">green block under the trolley: the load cell</span>');
      const same = stats.gapMax < 0.02 * Math.max(stats.yMax, 1e-9);
      readout(panel, 'info', `reading <b>${sg(stats.y, 2)} N</b> · the other release would read <b>${sg(stats.yo, 2)} N</b> · largest gap so far <b class="${same ? 'g-ok' : 'g-warn'}">${stats.gapMax.toFixed(2)} N</b>` + (same ? ' · <span class="g-ok">the sensor cannot tell them apart</span>' : ''));
    }
  }

  function draw() {
    plot.clear();
    const g = plot.g, W = plot.w, Hh = plot.h, left = 34, right = 12;
    const tEnd = Math.max(t, WINDOW), t0 = tEnd - WINDOW;
    const h1 = Math.round(Hh * 0.36);
    pane1.set(left, 28, W - left - right, h1, t0, tEnd, -3.2, 3.2);
    pane1.frame({ title: 'load cell reading (N)', yticks: [-2, 0, 2], fmt: (x) => (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x) });
    const ly = 28 + h1 + 18;
    if (hide) {
      pane1.dots(rec.t, rec.s[2], CSS.grey, 1.4, 0.8);
      pane1.line(rec.t, rec.s[3], CSS.green, 2);
      g.save(); g.fillStyle = CSS.grey; g.beginPath(); g.arc(left + 4, ly - 3.5, 2, 0, 2 * Math.PI); g.arc(left + 12, ly - 3.5, 2, 0, 2 * Math.PI); g.fill(); g.restore();
      GC.label(g, 'noisy samples', left + 20, ly);
      GC.legend(g, left + 104, ly, CSS.green, 'what the filter expects');
    } else {
      pane1.line(rec.t, rec.s[1], CSS.grey, 7, { alpha: 0.5 });
      pane1.line(rec.t, rec.s[0], CSS.green, 1.8);
      const lx = GC.legend(g, left, ly, CSS.green, 'this release');
      GC.legend(g, lx, ly, CSS.grey, 'the other release', { width: 7, alpha: 0.5 });
    }
    const y2 = ly + 36, h2 = Hh - y2 - 26;
    if (hide) {
      pane2.set(left, y2, W - left - right, h2, t0, tEnd, 0, 16);
      pane2.frame({ title: 'error of the estimate (degrees of swing)', yticks: [0, 5, 10, 15], fmt: (x) => x + '°' });
      pane2.line(rec.t, rec.s[5], rank === 4 ? CSS.green : CSS.red, 2);
    } else {
      pane2.set(left, y2, W - left - right, h2, t0, tEnd, -4.8, 4.8);
      pane2.frame({ title: 'gap between the two readings (N)', yticks: [-4, 0, 4], fmt: (x) => (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x) });
      pane2.line(rec.t, rec.s[4], rank === 4 ? CSS.accent : CSS.green, 2);
    }
    GC.label(g, `last ${WINDOW} s`, W - right, Hh - 8, CSS.faint, { align: 'right' });
  }

  v.onFrame((dt) => {
    advance(dt, 1, step);
    const c = cur();
    rig.a.setAngle(c.a[0]); rig.b.setAngle(c.b[0]);
    rig.ga.setAngle(kf.x[0]); rig.gb.setAngle(kf.x[1]);
    v.forward();
    text();
    draw();
  });

  panel._run = (sec) => { const n = Math.round(sec / H); for (let i = 0; i < n; i++) step(H); return panel._state(); };
  panel._state = () => {
    const c = cur(), o = other();
    return {
      t, cables, l1: L1, l2, start, hide, rank, seed, release_deg: deg(REL),
      truth_deg: [deg(c.a[0]), deg(c.b[0])], other_deg: [deg(o.a[0]), deg(o.b[0])],
      reading_N: stats.y, other_reading_N: stats.yo, gap_max_N: stats.gapMax, reading_max_N: stats.yMax,
      noisy_sample_N: stats.z, estimate_deg: [deg(kf.x[0]), deg(kf.x[1])], estimate_rate_deg_s: [deg(kf.x[2]), deg(kf.x[3])],
      error_amp_deg: stats.err, angle_error_deg: stats.e_deg,
      late_rms_angle_error_deg: stats.n ? deg(Math.sqrt(stats.sq / stats.n)) : null, late_max_error_amp_deg: stats.errLateMax,
      filter: { Ts: TS, sigma_r: SIG_R, sigma_a: SIG_A, p0_deg: deg(P0), late_from_s: LATE },
    };
  };
}
