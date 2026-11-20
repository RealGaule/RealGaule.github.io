// Lab 6, Section 0 (clip s00_opening): one arm, one target, four methods.
// The problem is the clip's (render/s00_opening.py): two-joint UR5e (shoulder lift, elbow), target at 90 % of the
// reach and 50 degrees above the horizontal, start (1.45, 0.30) rad (the arm hangs down). One button per method:
// gradient descent with the clip's large fixed step (0.94 * 2 / lambda_max at the solution), gradient descent with
// Gaussian gradient noise (step 1/L, sigma_g = 0.03), safeguarded Newton (mirror shift, Armijo), pure Newton.
// All numerics are sim6.js (same as Python); the deterministic runs are computed once at start-up, the noisy run
// draws its noise live (sim6 RNG, not NumPy's stream: a different draw from the clip's run, same law).
// Display only: joint angles are interpolated linearly between iterates, with the clip's per-method clocks.
import { S, THREE, makeArmViewer, planeCamera, Landscape, marksFor, marker, planeToWorld, world, Line3D, COL, CSS,
  val, onClick, readout, toggle, seg, fmt } from '../common.js';

// ---- the problem (constants of render/s00_opening.py)
const START = [1.45, 0.30];
const TARGET_FRACTION = 0.90, TARGET_ELEVATION_DEG = 50;
const GD_STEP_FRACTION = 0.94;            // alpha_gd = 0.94 * 2 / lambda_max(H at the solution)
const NOISE_SIGMA = 0.03;                 // sigma_g; noisy step alpha = 1 / L
const TOL_ARRIVE = 1e-3;                  // "steps to 1 mm": first iterate from which the tool stays within 1 mm

// ---- display clocks (as the clip)
const GD_SCHEDULE = [2.0, 0.36, 0.037];   // iterate shown after t s: a t + b t^2 + c t^3
const NOISE_SCHEDULE = [2.0, 7.0, 1.5];   // iterates per second rise from r0 to r1 with time constant tau
const SAFE_STEP_S = [0.9, 0.9, 0.7, 0.6, 0.5, 0.4];
const NEWTON_STEP_S = [1.1, 0.9, 0.6];
const MOVE_FRACTION = 0.7;                // part of a Newton step's time in which the arm moves
const RETURN_S = 0.9;                     // back to the start pose before a run
const NOISE_MAX = 6000;                   // the noisy run holds after this many steps
const NOISE_TRAIL = 120;                  // iterates of the noisy run drawn as the 3D tool trail
const SUB = 6;                            // samples per step of the 3D tool path

const M = {
  gd: { label: 'gradient descent', css: '#b8bcc6', col: 0xb8bcc6 },
  noise: { label: 'gradient descent + noise', css: '#a2a6b0', col: 0xa2a6b0 },
  safe: { label: 'safeguarded Newton', css: CSS.white, col: COL.white },
  newton: { label: "Newton's method", css: CSS.pink, col: COL.pink },
};
const ORDER = ['gd', 'noise', 'safe', 'newton'];

const ease = (x) => { x = Math.max(0, Math.min(1, x)); return x * x * (3 - 2 * x); };
function interp(xs, s) {
  if (xs.length === 1) return xs[0].slice();
  s = Math.max(0, Math.min(xs.length - 1, s));
  const k = Math.min(Math.floor(s), xs.length - 2), u = s - k;
  return [xs[k][0] + u * (xs[k + 1][0] - xs[k][0]), xs[k][1] + u * (xs[k + 1][1] - xs[k][1])];
}
function stepwise(tr, durs, n) {
  let T = 0;
  for (let k = 0; k < n; k++) {
    const d = durs[Math.min(k, durs.length - 1)];
    if (tr < T + d) return k + ease((tr - T) / (d * MOVE_FRACTION));
    T += d;
  }
  return n;
}
function distText(d) {
  if (d < 0.01) return `${fmt(1000 * d, 1)} mm`;
  if (d < 1) return `${fmt(100 * d, 1)} cm`;
  return `${fmt(d, 3)} m`;
}

export async function init(panel) {
  // ---- numerics (sim6.js)
  const two = new S.TwoJointArm();
  const el = TARGET_ELEVATION_DEG * Math.PI / 180, rho = TARGET_FRACTION * two.reachMax;
  const t = [rho * Math.cos(el), rho * Math.sin(el)];
  const f = (q) => two.cost(q, t), g = (q) => two.grad(q, t), h = (q) => two.hess(q, t);
  const err = (q) => { const p = two.p(q); return Math.hypot(p[0] - t[0], p[1] - t[1]); };
  const sp = two.stationaryPoints(t);
  const sol = sp.find((s) => s.name === 'elbow down');
  const lamSol = sol.eig;                                    // [min, max] at the solution
  const bounds = two.curvatureBounds(t);                     // once per target (webcore)
  const L = Math.max(Math.abs(bounds[0]), Math.abs(bounds[1]));
  const alphaGD = GD_STEP_FRACTION * 2 / lamSol[1], alphaN = 1 / L;

  const runs = {
    gd: S.gradientDescent(f, g, START, alphaGD, { maxIter: 2000 }),
    safe: S.safeguardedNewton(f, g, h, START),
    newton: S.newton(f, g, h, START),
  };
  const sum = {};
  for (const m of ['gd', 'safe', 'newton']) {
    const xs = runs[m].x, e = xs.map(err);
    let k1 = null;
    for (let k = e.length - 1; k >= 0 && e[k] <= TOL_ARRIVE; k--) k1 = k;
    const i = S.classify(xs[xs.length - 1], runs[m].converged, sp);
    sum[m] = { nIter: runs[m].nIter, converged: runs[m].converged, stepsTo1mm: k1,
      endsAt: i >= 0 ? sp[i].name : null, endType: i >= 0 ? sp[i].type : null, toolErrorEnd: e[e.length - 1],
      largestStep: Math.max(0, ...xs.slice(1).map((x, k) => Math.hypot(x[0] - xs[k][0], x[1] - xs[k][1]))) };
  }

  // the noisy run: drawn live, a new draw for each run
  let seed = 0;
  const noise = { xs: [START.slice()], rng: new S.RNG(seed) };
  const resetNoise = () => { seed += 1; noise.xs = [START.slice()]; noise.rng = new S.RNG(seed); };
  const extendNoise = (n) => {
    n = Math.min(n, NOISE_MAX);
    while (noise.xs.length - 1 < n) {
      const q = noise.xs[noise.xs.length - 1].slice();
      S.noisyStep(q, g(q), alphaN, alphaN * NOISE_SIGMA, [noise.rng.normal(), noise.rng.normal()]);
      noise.xs.push(q);
    }
  };
  seed = -1; resetNoise();                                   // first run: seed 0
  const pathOf = (m) => (m === 'noise' ? noise.xs : runs[m].x);
  const nOf = (m) => (m === 'noise' ? NOISE_MAX : runs[m].nIter);
  const progress = (m, tau) => {
    if (m === 'gd') { const [a, b, c] = GD_SCHEDULE; return Math.min(runs.gd.nIter, a * tau + b * tau * tau + c * tau ** 3); }
    if (m === 'noise') { const [r0, r1, T] = NOISE_SCHEDULE; return Math.min(NOISE_MAX, r1 * tau - (r1 - r0) * T * (1 - Math.exp(-tau / T))); }
    return stepwise(tau, m === 'safe' ? SAFE_STEP_S : NEWTON_STEP_S, runs[m].nIter);
  };

  // ---- 3D: the arm square to its plane, seen from the side the clip shows (target up and to the left)
  const v = await makeArmViewer(panel, { camera: planeCamera(two, { at: [-0.02, 0.02], distance: panel.clientWidth < 600 ? 3.45 : 3.05, elevation: -7, yaw: 186 }),
    q: two.fullQ(START) });
  const tW = planeToWorld(two, t);
  marker(v.scene, COL.green, 0.026).position.set(...tW);
  marker(v.scene, COL.green, 0.06, 0.16).position.set(...tW);
  {                                                          // reach limit: dashed arc, as in the clip
    const pts = [];
    for (let a = 6; a < 100; a += 4) {
      for (const b of [a, a + 2.4]) {
        const r = b * Math.PI / 180;
        pts.push(...planeToWorld(two, [two.reachMax * Math.cos(r), two.reachMax * Math.sin(r)]));
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    v.scene.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: COL.grey, transparent: true, opacity: 0.75 })));
  }
  const toolPath = {};
  for (const m of ORDER) { toolPath[m] = new Line3D(v.scene, M[m].col, 3200); toolPath[m].visible = false; }
  const gap = new Line3D(v.scene, COL.red, 2);
  gap.visible = false;
  const beads = Array.from({ length: 12 }, () => { const b = marker(v.scene, COL.white, 0.012); b.visible = false; return b; });

  // ---- 2D: the landscape over the two joint angles
  const land = new Landscape(panel.querySelector('canvas.land'));
  land.setCost(f, marksFor(two, t));

  // ---- state and controls
  let cur = 'gd', tau = -0.6, from = START.slice(), q = START.slice(), s = 0, done = false;
  const shown = { gd: 0, noise: 0, safe: 0, newton: 0 };    // progress reached by each method's last run
  let allPaths = toggle(panel, 'all', () => { landKey = ''; });
  const start = (m) => {
    if (m === 'noise') resetNoise();
    from = q.slice(); cur = m; done = false; s = 0; shown[m] = 0;
    const back = Math.hypot(...S.wrap([START[0] - from[0], START[1] - from[1]]));
    tau = back < 1e-9 ? 0 : -RETURN_S;
    for (const k of ORDER) toolPath[k].visible = false;
    showRule();
  };
  seg(panel, 'method', start);
  onClick(panel, 'replay', () => start(cur));
  window.addEventListener('keydown', (e) => {
    if (!'1234'.includes(e.key) || e.key.length !== 1 || !panel.matches(':hover')) return;
    panel.querySelectorAll('[data-ctl="method"] button')[Number(e.key) - 1].click();
  });

  const cond = lamSol[1] / lamSol[0];
  const showRule = () => {
    const head = `Target at ${Math.round(100 * TARGET_FRACTION)}&nbsp;% of the reach; curvatures at the solution ` +
      `${fmt(lamSol[0], 3)} and ${fmt(lamSol[1], 3)} (condition number ${fmt(cond, 1)}). `;
    const rule = {
      gd: `<b>Gradient descent:</b> fixed step <i>α</i> = ${fmt(alphaGD, 2)}, 0.94 of the local limit 2/<i>λ</i><sub>max</sub>; ` +
        `<i>αL</i> = ${fmt(alphaGD * L, 2)}, above the global limit 2/<i>L</i>, chosen to show the zig-zag.`,
      noise: `<b>Gradient descent + noise:</b> step <i>α</i> = 1/<i>L</i> = ${fmt(alphaN, 3)}, Gaussian gradient noise ` +
        `<i>σ</i><sub>g</sub> = ${NOISE_SIGMA}, a new draw on every run.`,
      safe: `<b>Safeguarded Newton:</b> solve (<i>H</i> + <i>τI</i>)<i>p</i> = −<i>g</i> with the curvature shifted positive, ` +
        `then halve the step until the cost drops.`,
      newton: `<b>Newton's method:</b> <i>p</i> = −<i>H</i><sup>−1</sup><i>g</i>, the stationary point of the local quadratic model, whatever its type.`,
    }[cur];
    readout(panel, 'rule', head + rule);
  };
  showRule();

  // ---- drawing
  let landKey = '', infoText = '';
  const slice = (m, s1) => { const xs = pathOf(m), k = Math.floor(s1); return [...xs.slice(0, k + 1), interp(xs, s1)]; };
  function drawLand() {
    const key = `${q[0].toFixed(5)},${q[1].toFixed(5)},${cur},${allPaths()}`;
    if (key === landKey) return;
    landKey = key;
    const G = land.draw();
    if (allPaths()) {
      for (const m of ORDER) if (m !== cur && shown[m] > 0) land.path(slice(m, shown[m]), M[m].css, { width: 1.1, alpha: 0.38 });
    }
    if (tau >= 0) {
      const dots = cur === 'safe' || cur === 'newton';
      land.path(slice(cur, s), M[cur].css, { width: cur === 'noise' ? 1.0 : 1.8, alpha: 0.95, dots, r: 2.6 });
    }
    const [x0, y0] = land.toPx(START);
    G.save(); G.strokeStyle = CSS.white; G.lineWidth = 1.5; G.beginPath(); G.arc(x0, y0, 5.5, 0, 2 * Math.PI); G.stroke(); G.restore();
    land.dot(q, CSS.white, 5);
  }
  function draw3D() {
    v.setArm(two.fullQ(q)); v.forward();
    const P = toolPath[cur];
    if (tau < 0) { P.visible = false; gap.visible = false; beads.forEach((b) => { b.visible = false; }); return; }
    const xs = pathOf(cur), k = Math.floor(s), k0 = cur === 'noise' ? Math.max(0, k - NOISE_TRAIL) : 0, pts = [];
    for (let j = k0; j < k && j < xs.length - 1; j++) for (let i = 0; i < SUB; i++) pts.push(world(two.pWorld(interp(xs, j + i / SUB))));
    pts.push(world(two.pWorld(q)));
    P.set(pts); P.visible = true;
    const nb = cur === 'safe' || cur === 'newton' ? Math.min(k + 1, beads.length) : 0;
    beads.forEach((b, i) => {
      b.visible = i < nb;
      if (b.visible) { b.position.set(...world(two.pWorld(xs[i]))); b.material.color.setHex(M[cur].col); b.material.emissive.setHex(M[cur].col); }
    });
    const far = done && err(q) > 0.01;
    gap.visible = far;
    if (far) gap.set([world(two.pWorld(q)), tW]);
  }
  function info() {
    let html;
    const e = err(q), lab = `<b style="color:${M[cur].css}">${M[cur].label}</b>`;
    if (tau < 0) html = `${lab}: back to the start`;
    else if (cur === 'noise') html = `${lab}: step ${Math.floor(s)} · tool ${distText(e)} from the target · never at rest`;
    else {
      const r = sum[cur], k = Math.floor(s);
      if (done && r.endType === 'max') {
        html = `${lab}: ${r.nIter} steps · <span class="c-red">stops at the maximum</span>, tool ${distText(r.toolErrorEnd)} from the target`;
      } else if (r.stepsTo1mm !== null && k >= r.stepsTo1mm) {
        html = `${lab}: <b>${r.stepsTo1mm} steps to 1 mm</b>` + (done ? ` · ${r.nIter} to |<i>g</i>| ≤ 10<sup>−9</sup> · ${r.endsAt}` : ` · step ${k}`);
      } else html = `${lab}: step ${k} · tool ${distText(e)} from the target`;
    }
    if (html !== infoText) { infoText = html; readout(panel, 'info', html); }
  }

  v.onFrame((dt) => {
    if (!done) tau += dt * val(panel, 'speed');
    if (tau < 0) {
      const u = ease(1 + tau / RETURN_S), d = S.wrap([START[0] - from[0], START[1] - from[1]]);
      q = [from[0] + u * d[0], from[1] + u * d[1]];
    } else {
      s = progress(cur, tau);
      if (cur === 'noise') extendNoise(Math.ceil(s) + 1);
      shown[cur] = s;
      q = interp(pathOf(cur), s);
      if (s >= nOf(cur)) done = true;
    }
    draw3D();
    drawLand();
    info();
  });
  draw3D(); drawLand(); info();

  panel._state = () => {
    const ne = noise.xs.slice(-100).map(err);
    return {
      method: cur, tau, progress: s, step: Math.floor(s), done, q, toolError: err(q), speed: val(panel, 'speed'),
      allPaths: allPaths(), shown: { ...shown },
      target: t, targetDistance: rho, reach: two.reachMax, start: START,
      eigAtSolution: lamSol, condAtSolution: cond, curvatureBounds: bounds, L,
      alphaGD, alphaGDtimesL: alphaGD * L, alphaNoise: alphaN, sigmaG: NOISE_SIGMA,
      runs: sum,
      noise: { seed, steps: noise.xs.length - 1, rmsToolErrorLast100: Math.sqrt(ne.reduce((a, b) => a + b * b, 0) / ne.length) },
      stationary: sp.map((p) => ({ name: p.name, type: p.type, q: p.q })),
      armToolVsSim: (() => { const a = v.tool(), b = world(two.pWorld(q)); return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); })(),
      camera: v.camera.position.toArray().map((x) => +x.toFixed(3)),
    };
  };
}
