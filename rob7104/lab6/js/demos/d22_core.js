// Demo 2.2 (l1 and sparsity): the numerics of the demo, all through sim6.js (structure.reach_penalised).
//
//   l1:  min_d 1/2 |p(q0 + d) - t|^2 + lam |d|_1           (exact zeros: a few joints do all the work)
//   l2:  min_d 1/2 |p(q0 + d) - t|^2 + mu/2 |d|_2^2        (every joint that can help moves a little)
//
// As in the clip (render/s22_l1.py), the l2 weight mu is chosen so that the l2 solution leaves the SAME distance to
// the target as the l1 solution: the two penalties are then compared at equal accuracy.
// Per target (analyse): lambda_max, the l1 path on a grid of lambda / lambda_max (warm-started from the largest
// lambda down, finishing solve on, as sim6.reachPath), the weights at which each joint stops (bisection), and a
// table of l2 solutions over mu (for the l2 curves of the path plot, interpolated in log gap; display only).
// Per slider value (solveAt): the exact l1 solution and the exact gap-matched l2 solution.
import * as S from '../sim6.js';

export const Q0 = S.UR5E.Q_HOME;                       // start pose of scene 2.2 (stage6.Q_HOME)
export const T_DEFAULT = [0.37, 0.355, 0.74];          // target of scene 2.2, arm frame (m)
export const FK = S.fkJacPos;
export const TOL = 1e-9;                                // |d_j| <= TOL rad is drawn as "did not move" (joint 6 in l2)
export const F_MIN = 0.001, F_MAX = 1.05;               // slider range of lambda / lambda_max
const N_GRID = 64;
const MUS = Array.from({ length: 46 }, (_, i) => 10 ** (-6 + 0.2 * i));   // l2 weights 1e-6 .. 1e3

/** Numbers of out/s22_l1_facts.json (the clip's case: default target), for side-by-side checks in _state(). */
export const REF = {
  source: 'Lab6_Optimization/build/out/s22_l1_facts.json',
  lambda_max: 0.09487087999748235, lambda_max_joint: 3,
  l1_at_0p001: { d_deg: [28.6287, 23.1143, -43.6367, 0, 0, 0], gap_mm: 0.6773015484422845, moving: 3 },
  l2_matched: { mu: 0.00020440093824950802, d_deg: [26.7151, 18.9583, -34.096, -18.0048, 9.4614, 0], moving: 5 },
  stops_over_lambda_max: { 1: 0.7855474219135429, 2: 0.2795509485951219, 3: 1.0000000000000007 },
  order_of_stopping: [2, 1, 3],
  distance_start_to_target_mm: 294.759762972498,
};

const zeros6 = () => [0, 0, 0, 0, 0, 0];
const solve = (t, kind, w, d0) => S.reachPenalised(FK, Q0, t, w, { kind, polish: true, chunk: 40, d0 });

/** Everything that depends on the target only (call once per target, about 0.1-0.5 s). */
export function analyse(t) {
  const t0 = performance.now();
  const { p: p0, J: J0 } = FK(Q0);
  const g0 = S.matTvec(J0, S.sub(p0, t)).map(Math.abs);
  const lamMax = S.reachLambdaMax(FK, Q0, t);
  const dist0 = S.norm(S.sub(p0, t));
  const F = Array.from({ length: N_GRID }, (_, i) => F_MIN + (F_MAX - F_MIN) * i / (N_GRID - 1));

  // l1 path, from the largest weight down (each solve warm-started at the previous solution).
  const D1 = new Array(N_GRID), G1 = new Array(N_GRID), K1 = new Array(N_GRID);
  let d = zeros6();
  for (let i = N_GRID - 1; i >= 0; i--) {
    const r = solve(t, 'l1', F[i] * lamMax, d);
    d = r.d; D1[i] = d; G1[i] = r.miss; K1[i] = r.kkt;
  }

  // Weight at which each joint stops (largest lambda / lambda_max with d_j != 0), by bisection.
  const stops = {};
  for (let j = 0; j < 6; j++) {
    let i = -1;
    for (let k = N_GRID - 1; k >= 0; k--) if (D1[k][j] !== 0) { i = k; break; }
    if (i < 0) continue;
    if (i === N_GRID - 1) { stops[j + 1] = F_MAX; continue; }
    let lo = F[i], hi = F[i + 1], warm = D1[i];
    for (let it = 0; it < 18; it++) {
      const mid = 0.5 * (lo + hi), r = solve(t, 'l1', mid * lamMax, warm);
      if (r.d[j] !== 0) { lo = mid; warm = r.d; } else hi = mid;
    }
    stops[j + 1] = 0.5 * (lo + hi);
  }
  const order = Object.keys(stops).map(Number).sort((a, b) => stops[a] - stops[b]);

  // l2 table over mu (largest first, warm-started), gap ascending with mu.
  const M = MUS.length, D2T = new Array(M), G2T = new Array(M);
  d = zeros6();
  for (let k = M - 1; k >= 0; k--) {
    const r = solve(t, 'l2', MUS[k], d);
    d = r.d; D2T[k] = d; G2T[k] = r.miss;
  }

  // l2 curves of the path plot: the l2 solution with the gap of each l1 grid point (interpolated in log gap).
  const D2 = G1.map((g) => interpGap(G2T, D2T, g, dist0));

  const A = { t: t.slice(), lamMax, lamMaxJoint: 1 + g0.indexOf(Math.max(...g0)), dist0, F, D1, G1, K1, stops, order,
    MU: MUS, D2T, G2T, D2, ms: 0 };
  // Ring brightness scale: the largest displacement of the two solutions at the smallest weight (as the clip).
  const s0 = solveAt(A, F_MIN);
  A.fullGlow = Math.max(...s0.l1.d.map(Math.abs), ...s0.l2.d.map(Math.abs), 1e-9);
  A.ms = performance.now() - t0;
  return A;
}

function interpGap(G, D, g, dist0) {
  if (g >= dist0) return zeros6();
  let k = 0;
  while (k < G.length - 2 && G[k + 1] < g) k++;
  const a = Math.log(Math.max(G[k], 1e-300)), b = Math.log(Math.max(G[k + 1], 1e-300));
  const u = Math.max(0, Math.min(1, (Math.log(g) - a) / ((b - a) || 1)));
  return D[k].map((v, j) => v + u * (D[k + 1][j] - v));
}

/** The l2 solution whose distance to the target equals gStar (Illinois method on log mu, bracketed by the table). */
export function l2Matched(A, gStar) {
  const t = A.t;
  if (!(gStar < A.dist0 * (1 - 1e-12))) {
    return { d: zeros6(), mu: Infinity, miss: A.dist0, kkt: 0, evals: 0 };
  }
  const M = A.MU.length, G = A.G2T;
  let k = -1;
  for (let i = 0; i < M; i++) if (G[i] <= gStar) k = i;
  const ev = (x, warm) => { const r = solve(t, 'l2', Math.exp(x), warm); return { r, h: Math.log(r.miss / gStar) }; };
  let evals = 0, a, b, ra, rb;
  if (k < 0) { ra = ev(Math.log(A.MU[0]) - 6, A.D2T[0]); a = Math.log(A.MU[0]) - 6; evals++; }
  else { a = Math.log(A.MU[k]); ra = { r: { d: A.D2T[k], miss: G[k], kkt: 0 }, h: Math.log(G[k] / gStar) }; }
  if (k >= M - 1) { b = Math.log(A.MU[M - 1]) + 6; rb = ev(b, A.D2T[M - 1]); evals++; }
  else { b = Math.log(A.MU[k + 1]); rb = { r: { d: A.D2T[k + 1], miss: G[k + 1], kkt: 0 }, h: Math.log(G[k + 1] / gStar) }; }
  let best = Math.abs(ra.h) < Math.abs(rb.h) ? { x: a, ...ra } : { x: b, ...rb }, side = 0;
  for (let it = 0; it < 40 && Math.abs(best.h) > 1e-11; it++) {
    if (!(ra.h < 0 && rb.h > 0) && !(ra.h > 0 && rb.h < 0)) break;      // no sign change: keep the closest
    const x = (a * rb.h - b * ra.h) / (rb.h - ra.h);
    const c = ev(x, Math.abs(x - a) < Math.abs(x - b) ? ra.r.d : rb.r.d);
    evals++;
    if (Math.abs(c.h) < Math.abs(best.h)) best = { x, ...c };
    if (c.h * rb.h > 0) { b = x; rb = c; if (side === 1) ra = { ...ra, h: ra.h / 2 }; side = 1; }
    else { a = x; ra = c; if (side === -1) rb = { ...rb, h: rb.h / 2 }; side = -1; }
  }
  // the table entries carry no solve result; make sure the returned one is a fresh solve
  if (best.r.kkt === 0 && !best.r.q) { best = { x: best.x, ...ev(best.x, best.r.d) }; evals++; }
  return { d: best.r.d, mu: Math.exp(best.x), miss: best.r.miss, kkt: best.r.kkt, evals };
}

/** Exact l1 solution at lambda = f lambda_max and the gap-matched l2 solution. */
export function solveAt(A, f) {
  const t0 = performance.now();
  const i = Math.max(0, Math.min(A.F.length - 1, Math.round((f - F_MIN) / (F_MAX - F_MIN) * (A.F.length - 1))));
  const r1 = solve(A.t, 'l1', f * A.lamMax, A.D1[i]);
  const r2 = l2Matched(A, r1.miss);
  const out = {
    f, lam: f * A.lamMax,
    l1: { d: r1.d, miss: r1.miss, kkt: r1.kkt, moving: r1.d.filter((v) => v !== 0).length, polished: r1.polished, iters: r1.iters },
    l2: { d: r2.d, mu: r2.mu, miss: r2.miss, kkt: r2.kkt, moving: r2.d.filter((v) => Math.abs(v) > TOL).length, evals: r2.evals },
  };
  out.ms = performance.now() - t0;
  return out;
}
