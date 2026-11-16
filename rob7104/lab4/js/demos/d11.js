// 1.1 Normal modes: a drone with a two-link slung load. The yellow load is the nonlinear robot;
// the green and red ghosts are the two modal components of the linear solution.
import * as S from '../sim.js';
import { makeViewer, Trail, COL, val, onInput, onClick, readout, ctl, stepper, deg } from '../common.js';

export async function init(panel) {
  const v = await makeViewer(panel, 'drone.mjb', { lookat: [0, 0, 1.45], distance: 4.3, azimuth: 78, elevation: -9 });
  const mats = S.dpMatrices();
  const { w, V } = S.modes(mats);
  const AMP = [12, 8].map((d) => d * Math.PI / 180);            // largest link angle of each pure mode at slider = 1
  const shape = V.map((m, i) => { const s = Math.max(Math.abs(m[0]), Math.abs(m[1])); return m.map((x) => x / s * AMP[i]); });
  const trail = new Trail(v.scene, COL.yellow, 70);
  const advance = stepper(1e-3);
  let x, t, c;

  function release() {
    c = [val(panel, 'a1'), val(panel, 'a2')];
    x = [c[0] * shape[0][0] + c[1] * shape[1][0], c[0] * shape[0][1] + c[1] * shape[1][1], 0, 0];
    t = 0;
    trail.clear();
  }

  function setChain(name, th) { v.setJoint(`${name}_q1`, th[0]); v.setJoint(`${name}_q2`, th[1] - th[0]); }

  let ghosts = true;
  const ghostGeoms = [];
  for (let g = 0; g < v.model.ngeom; g++) {
    const body = v.model.geom_bodyid[g];
    const name = v.mujoco.mj_id2name(v.model, v.mujoco.mjtObj.mjOBJ_BODY.value, body) || '';
    if (/^g[12]_/.test(name)) ghostGeoms.push(g);
  }
  onClick(panel, 'ghosts', () => {
    ghosts = !ghosts;
    ctl(panel, 'ghosts').classList.toggle('on', ghosts);
    ghostGeoms.forEach((g) => { if (v.meshes[g]) v.meshes[g].visible = ghosts; });
  });
  onClick(panel, 'release', release);
  onInput(panel, 'a1', release);
  onInput(panel, 'a2', release);
  release();
  readout(panel, 'info', `periods <b>${(2 * Math.PI / w[0]).toFixed(2)} s</b> and <b>${(2 * Math.PI / w[1]).toFixed(2)} s</b>`);

  v.onFrame((dt) => {
    advance(dt, 1, (h) => { x = S.dpStep(S.P, 0, 0, x, 0, 0, h); t += h; });
    setChain('main', x);
    setChain('g1', shape[0].map((s) => s * c[0] * Math.cos(w[0] * t)));
    setChain('g2', shape[1].map((s) => s * c[1] * Math.cos(w[1] * t)));
    v.forward();
    trail.push(v.sitePos('main_m2'));
  });
  panel._state = () => ({ t, x, w, lin: [0, 1].map((i) => c[0] * shape[0][i] * Math.cos(w[0] * t) + c[1] * shape[1][i] * Math.cos(w[1] * t)) });
}
