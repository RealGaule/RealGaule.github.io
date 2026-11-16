// 1.1 Why sqrt(dt): three corridors of robots take N random steps of size c dt^p Z on [0, 1] s, p = 1, 1/2, 1/4.
// Same construction as the notebook video s01_sqrtdt: one set of fine normals (512 per robot) is summed into
// N coarse steps, so every N and every corridor uses the same random numbers. c = 0.22 m.
import { rng } from '../sim3.js';
import { makeViewer, placeLamp, SwarmMesh, THREE, COL, CSS, ctl, val, onInput, onClick, readout } from '../common3.js';
import { hideEndWalls, Tags, sig3 } from '../noise_kit.js';

const C = 0.22;                       // m
const NMAX = 512;
const MSTAT = 2000;                   // robots per corridor in the variance
const MDRAW = 200;                    // robots per corridor drawn (as in the video)
const WALK = 1.0, HOLD = 0.8;         // s of real time: the walk is shown in real time, then held
const ROWS = [
  { p: 1.0, y: 0.9, lab: 'steps ∝ Δt', law: 'Var X(1) ∝ Δt', verb: 'dies' },
  { p: 0.5, y: 0.0, lab: 'steps ∝ √Δt', law: 'Var X(1) ∝ 1', verb: 'stays' },
  { p: 0.25, y: -0.9, lab: 'steps ∝ Δt<sup>1/4</sup>', law: 'Var X(1) ∝ Δt<sup>−1/2</sup>', verb: 'explodes' },
];
const varTheory = (p, N, t = 1) => C * C * t * (1 / N) ** (2 * p - 1);

export async function init(panel) {
  const v = await makeViewer(panel, 'corridor3.mjb', { lookat: [0, -0.05, 0], distance: 4.25, azimuth: 90, elevation: -54, fov: 40 });
  for (let i = 0; i < 3; i++) placeLamp(v, i, null);
  hideEndWalls(v);                                   // the corridors are open: robots may walk out of the ends
  v.forward();

  // fine normals, shared by all N and all three corridors
  const R = rng(12);
  const Zf = new Float32Array(MSTAT * NMAX);
  for (let i = 0; i < Zf.length; i++) Zf[i] = R.normal();
  const S = new Float64Array(MSTAT);                 // sum of the fine normals of each robot
  for (let i = 0; i < MSTAT; i++) { let s = 0; for (let k = 0; k < NMAX; k++) s += Zf[i * NMAX + k]; S[i] = s; }
  const ry = ROWS.map(() => Float64Array.from({ length: MDRAW }, () => (R.uni() - 0.5) * 2 * (0.16 - 0.035)));

  // markers: +-2 sd of X(t) from the formula, one pair per corridor
  const markGeo = new THREE.BoxGeometry(0.014, 0.42, 0.006);
  const markMat = new THREE.MeshBasicMaterial({ color: COL.yellow, toneMapped: false, transparent: true, opacity: 0.9 });
  const marks = ROWS.map((r) => [-1, 1].map(() => { const m = new THREE.Mesh(markGeo, markMat); m.position.set(0, r.y, 0.006); v.scene.add(m); return m; }));

  const swarm = new SwarmMesh(v.scene, 3 * MDRAW);
  const xs = new Float64Array(3 * MDRAW), ys = new Float64Array(3 * MDRAW);
  ROWS.forEach((r, j) => { for (let i = 0; i < MDRAW; i++) ys[j * MDRAW + i] = r.y + ry[j][i]; });
  const blue = new THREE.Color(CSS.blue);

  const tags = new Tags(panel.querySelector('.cell'), v);
  ROWS.forEach((r) => { r.tag = tags.add([-1.5, r.y + 0.2, 0.03]); r.end = tags.add([1.5, r.y + 0.2, 0.03], { anchor: 'right' }); });

  let N = 0, path = null, stats = null, t = 0, sweeping = false;

  function setN(n) {
    N = n;
    const B = NMAX / N;
    // coarse cumulative sums (unscaled) for the drawn robots: path[i * (N + 1) + j]
    path = new Float64Array(MDRAW * (N + 1));
    for (let i = 0; i < MDRAW; i++) {
      let s = 0;
      for (let j = 0; j < N; j++) {
        let b = 0;
        for (let k = 0; k < B; k++) b += Zf[i * NMAX + j * B + k];
        s += b / Math.sqrt(B);
        path[i * (N + 1) + j + 1] = s;
      }
    }
    // Var X(1) over all MSTAT robots: X(1) = c dt^p S / sqrt(B)
    let m = 0, q = 0;
    for (let i = 0; i < MSTAT; i++) { m += S[i]; q += S[i] * S[i]; }
    m /= MSTAT; const varS = q / MSTAT - m * m;
    let md = 0, qd = 0;
    for (let i = 0; i < MDRAW; i++) { const e = path[i * (N + 1) + N]; md += e; qd += e * e; }
    md /= MDRAW; const varD = qd / MDRAW - md * md;
    stats = ROWS.map((r) => {
      const k = C * (1 / N) ** r.p;
      return { p: r.p, varTheory: varTheory(r.p, N), varRobots: k * k * varS / B, varDrawn: k * k * varD };
    });
    ctl(panel, 'nout').textContent = String(N);
    t = 0;
  }

  function show() {
    const u = Math.min(t, WALK) / WALK * N, j = Math.min(Math.floor(u), N - 1), fr = u - j;
    ROWS.forEach((r, c) => {
      const k = C * (1 / N) ** r.p;
      for (let i = 0; i < MDRAW; i++) {
        const a = path[i * (N + 1) + j], b = path[i * (N + 1) + j + 1];
        xs[c * MDRAW + i] = k * (a + fr * (b - a));
      }
      const tt = Math.min(t, WALK) / WALK, sd = Math.sqrt(varTheory(r.p, N, tt));
      marks[c].forEach((m, s) => { m.position.x = (s ? 1 : -1) * 2 * sd; m.visible = tt > 0 && 2 * sd < 3.2; });
      const st = stats[c], done = t >= WALK;
      tags.text(r.tag, `<b style="color:${CSS.white}">${r.lab}</b>&ensp;<span style="color:${CSS.grey}">${r.law}</span><br>`
        + `<span style="color:${CSS.yellow}">formula ${sig3(st.varTheory)} m²</span>`
        + `&ensp;<span style="color:${CSS.blue}">robots ${done ? sig3(st.varRobots) + ' m²' : '…'}</span>`);
      tags.text(r.end, `<span style="color:${CSS.pink}">spread ${r.verb}</span>`);
      r.end.el.style.visibility = done && N >= 64 ? 'visible' : 'hidden';
    });
    swarm.update(xs, ys, blue);
    readout(panel, 'info', `N = <b>${N}</b>, Δt = 1/${N} s&ensp; t = <b>${Math.min(t, WALK).toFixed(2)} s</b>`);
  }

  const slider = ctl(panel, 'n');
  onInput(panel, 'n', () => { sweeping = false; ctl(panel, 'sweep').classList.remove('on'); setN(2 ** val(panel, 'n')); });
  onClick(panel, 'replay', () => { t = 0; });
  onClick(panel, 'sweep', () => {
    sweeping = !sweeping;
    ctl(panel, 'sweep').classList.toggle('on', sweeping);
    if (sweeping) { slider.value = 1; setN(2); }
  });
  setN(2 ** val(panel, 'n'));

  v.onFrame((dt) => {
    t += dt;
    if (sweeping && t > WALK + HOLD) {
      if (N < NMAX) { slider.value = Math.log2(N) + 1; setN(N * 2); }
      else { sweeping = false; ctl(panel, 'sweep').classList.remove('on'); }
    }
    show();
    tags.update();
  });
  show();

  panel._state = () => ({ N, t: Math.min(t, WALK), c: C, robotsPerCorridor: MSTAT, corridors: stats.map((s) => ({ ...s })) });
}
