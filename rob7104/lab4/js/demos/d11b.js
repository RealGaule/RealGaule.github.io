// 1.1b The flow e^{At}: one drone, 25 single-link loads, one per initial condition on a 5 x 5 grid in
// (angle, angular velocity / omega). The linear model is the subject here, so the loads and the phase
// plane are both drawn from x(t) = e^{At} x(0).
import * as S from '../sim.js';
import { makeViewer, CSS, val, onInput, onClick, readout, ctl, plot2d, rad } from '../common.js';
import { svd2, Frame, runWhenVisible } from '../drone_util.js';

const LEN = 1.2;                           // cable length (m), as in models/drone_flow.mjb
const GRID = [-12, -6, 0, 6, 12];          // degrees, for the angle and for angular velocity / omega
const R = 12, LIM = 20.5;                  // circle of initial conditions; axis limit (degrees)
const GREEN = [0.514, 0.757, 0.404], BLUE = [0.345, 0.769, 0.867];

/** Same colour rule as build/web/export_drone.py: green -> blue along the angle, dark -> bright along the velocity. */
function colour(i, j) {
  let c = GREEN.map((g, k) => ((1 - i / 4) * g + (i / 4) * BLUE[k]) * (0.5 + 0.5 * j / 4));
  if (j === 4) c = c.map((x) => 0.8 * x + 0.2);
  return `rgb(${c.map((x) => Math.round(255 * Math.min(1, x))).join(',')})`;
}

export async function init(panel) {
  const v = await makeViewer(panel, 'drone_flow.mjb', { lookat: [0, 0, 1.74], distance: 3.3, azimuth: 57, elevation: -13 });
  runWhenVisible(panel, v);
  const omega = Math.sqrt(S.G / LEN);
  const pts = [];                           // slot n = 5 j + i
  for (let j = 0; j < 5; j++) for (let i = 0; i < 5; i++) pts.push({ i, j, z0: [GRID[i], GRID[j]], col: colour(i, j) });
  const plot = plot2d(panel.querySelector('canvas.plot'));
  const circle = Array.from({ length: 121 }, (_, k) => [R * Math.cos(2 * Math.PI * k / 120), R * Math.sin(2 * Math.PI * k / 120)]);
  let t = 0, zeta = val(panel, 'zeta'), playing = true, hold = 0, E = S.eye(2), sv = svd2(E);

  /** e^{At} written in the plotted coordinates z = (theta, theta'/omega): S e^{At} S^-1 with S = diag(1, 1/omega). */
  function flow(time) {
    const Er = S.expm(S.scale(S.singlePendulum(omega, zeta).A, time));
    return [[Er[0][0], Er[0][1] * omega], [Er[1][0] / omega, Er[1][1]]];
  }
  const map = (z) => [E[0][0] * z[0] + E[0][1] * z[1], E[1][0] * z[0] + E[1][1] * z[1]];

  function restart() { t = 0; hold = 0.5; }
  function setPlaying(on) {
    playing = on;
    ctl(panel, 'play').classList.toggle('on', on);
    ctl(panel, 'play').textContent = on ? 'pause' : 'play';
  }
  onInput(panel, 'zeta', () => { zeta = val(panel, 'zeta'); restart(); setPlaying(true); });
  onClick(panel, 'play', () => setPlaying(!playing));
  onClick(panel, 'restart', () => { restart(); setPlaying(true); });

  function draw() {
    plot.clear();
    const g = plot.g;
    const side = Math.max(120, Math.min(plot.w - 76, plot.h - 104));
    const f = new Frame(g, (plot.w - side) / 2 + 20, 40, side, side, [-LIM, LIM], [-LIM, LIM]);
    f.seg(-LIM, 0, LIM, 0, CSS.faint); f.seg(0, -LIM, 0, LIM, CSS.faint);
    const lo = GRID[0], hi = GRID[4];
    for (const a of GRID) { f.seg(a, lo, a, hi, '#34363d'); f.seg(lo, a, hi, a, '#34363d'); }      // where the grid started
    f.line(circle.map((p) => p[0]), circle.map((p) => p[1]), '#6b6c75', 1, [4, 4]);
    f.axes([[-12, '−12°'], [0, '0'], [12, '12°']], [[-12, '−12°'], [0, '0'], [12, '12°']], 'angle θ', 'angular velocity / ω');
    f.clip(() => {
      GRID.forEach((a, k) => {
        const p = map([lo, a]), q = map([hi, a]);
        f.seg(p[0], p[1], q[0], q[1], `rgba(255,255,255,${0.30 + 0.09 * k})`, 1.4);                  // constant velocity
      });
      GRID.forEach((a, k) => {
        const p = map([a, lo]), q = map([a, hi]);
        f.seg(p[0], p[1], q[0], q[1], colour(k, 3), 1.4);                                            // constant angle
      });
      const c = circle.map(map);
      f.line(c.map((p) => p[0]), c.map((p) => p[1]), CSS.white, 2);
      if (sv.s1 - sv.s2 > 0.004) {
        const sgn = sv.u1[0] + 0.3 * sv.u1[1] >= 0 ? 1 : -1;
        f.seg(0, 0, sgn * R * sv.s1 * sv.u1[0], sgn * R * sv.s1 * sv.u1[1], CSS.yellow, 3);
      }
      for (const p of pts) { const z = map(p.z0); f.dot(z[0], z[1], 4.2, p.col, '#000', 1); }
    });
  }

  v.onFrame((dt) => {
    if (playing) {
      if (hold > 0) hold -= dt; else t += dt;
    }
    E = flow(t);
    sv = svd2(E);
    if (playing && zeta > 0 && sv.s1 < 0.03) restart();            // everything has settled: start again
    pts.forEach((p, n) => v.setJoint(`p${n}_q1`, rad(map(p.z0)[0])));
    v.forward();
    draw();
    readout(panel, 'norm', sv.s1.toFixed(2));
    readout(panel, 'info', `t = <b>${t.toFixed(1)} s</b> · eigenvalues <b>${(-zeta * omega).toFixed(2)} ± ${(omega * Math.sqrt(1 - zeta * zeta)).toFixed(2)}i</b>`);
  });
  panel._state = () => ({ t, zeta, omega, playing, E, norm: sv.s1, s2: sv.s2, u1: sv.u1, z: pts.map((p) => map(p.z0)),
    angles: pts.map((_, n) => v.getJoint(`p${n}_q1`)) });
  panel._seek = (time) => { t = time; hold = 0; setPlaying(false); E = flow(t); sv = svd2(E); };
}
