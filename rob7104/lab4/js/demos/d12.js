// 1.2 When linear fails: the two-link load released from rest with both links at the same angle.
// Solid load = nonlinear robot (dpStep). Blue halo = linear model e^{At} x0 from the same start,
// written with the two modes: th(t) = sum_i c_i v_i cos(w_i t). Yellow line = gap between the two parcels.
import * as S from '../sim.js';
import { makeViewer, Trail, COL, val, onInput, onClick, readout, stepper, rad } from '../common.js';
import { setChain, showChain, haloChain, loadXZ, Link3, runWhenVisible } from '../drone_util.js';

const RUN = 12, HOLD = 0.6;                // seconds of motion per release, and of still pose before it

export async function init(panel) {
  const v = await makeViewer(panel, 'drone.mjb', { lookat: [0, 0, 1.72], distance: 5.3, azimuth: 86, elevation: -6 });
  runWhenVisible(panel, v);
  showChain(v, 'g2', false);
  haloChain(v, 'g1', COL.blue);
  const mats = S.dpMatrices();
  const { w, V } = S.modes(mats);
  const trail = new Trail(v.scene, COL.yellow, 80), trailLin = new Trail(v.scene, COL.blue, 80);
  const gapLine = new Link3(v.scene, COL.yellow);
  const advance = stepper(1e-3);
  let x, c, t, hold, gap, peak, lastPeak = null, a0;

  /** Linear prediction of the two link angles at time t. */
  const linear = (time) => [0, 1].map((k) => c[0] * V[0][k] * Math.cos(w[0] * time) + c[1] * V[1][k] * Math.cos(w[1] * time));
  function measure() {
    const r = loadXZ(x).parcel, l = loadXZ(linear(t)).parcel;
    gap = Math.hypot(r[0] - l[0], r[1] - l[1]);
    peak = Math.max(peak, gap);
  }
  function release() {
    a0 = rad(val(panel, 'angle'));
    x = [a0, a0, 0, 0];
    c = V.map((m) => S.dot(m, S.mulVec(mats.M, [a0, a0])));         // modal coordinates (V is mass-normalized)
    t = 0; hold = HOLD; peak = 0;
    measure();
    trail.clear(); trailLin.clear();
  }
  function step(h) { x = S.dpStep(S.P, 0, 0, x, 0, 0, h); t += h; measure(); }

  onInput(panel, 'angle', release);
  onClick(panel, 'release', release);
  release();

  v.onFrame((dt) => {
    if (hold > 0) hold -= dt;
    else {
      advance(dt, 1, step);
      if (t >= RUN) { lastPeak = peak; release(); }
    }
    setChain(v, 'main', x);
    setChain(v, 'g1', linear(t));
    v.forward();
    const pr = v.sitePos('main_m2'), pl = v.sitePos('g1_m2');
    gapLine.set(pr, pl);
    if (hold <= 0) { trail.push(pr); trailLin.push(pl); }
    readout(panel, 'info', `gap now <b>${(100 * gap).toFixed(1)} cm</b> · peak so far <b>${(100 * peak).toFixed(1)} cm</b> · t = <b>${t.toFixed(1)} s</b>`);
  });
  panel._state = () => ({ t, angle: a0, x, lin: linear(t), gap, peak, lastPeak, w, c });
  /** Test hook: integrate without drawing until time T of the current release. */
  panel._run = (T) => { hold = 0; while (t < T - 1e-9) step(1e-3); return panel._state(); };
}
