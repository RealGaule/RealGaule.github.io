// Demo 2.2 - l1 and sparsity. The UR5e reaches the same target with an l1 or an l2 penalty on the joint
// displacement; a ring on every joint shows how far that joint moved (yellow l1, blue l2, grey = exactly zero).
// Right: the two-variable picture of Lecture 4 (l1 diamond touches the line at a corner, l2 circle anywhere),
// moving with the same lambda / lambda_max. Below: the path of every joint against the weight, and the bars.
// Numerics: d22_core.js -> sim6.js (reachPenalised, reachLambdaMax). Same colours as out/s22_l1.mp4.
import { S, THREE, makeArmViewer, marker, GhostArms, Line3D, world, COL, CSS, val, onInput, onClick, readout, seg,
  toggle, setVal, plot2d, fmt, deg } from '../common.js';
import { Q0, T_DEFAULT, TOL, F_MIN, F_MAX, REF, analyse, solveAt } from './d22_core.js';

const CAMERA = { lookat: [0.16, 0.18, 1.46], distance: 1.62, azimuth: -24, elevation: -17 };   // the clip's view, closer
// Ring and cover plate of each joint (render/stage6.py _RING, _CAP): offset along the joint axis, radius (m).
const RING = [[-0.064, 0.060], [-0.064, 0.060], [0.053, 0.058], [0.053, 0.039], [0.054, 0.039], [0.053, 0.039]];
const CAP = [[0.0715, 0.050], [0.0715, 0.050], [0.2025, 0.050], [-0.0555, 0.032], [-0.0555, 0.032], [-0.0565, 0.032]];
const REST = 0x9ea3b0;                                   // ring of a joint that did not move (light grey, no glow)
const FLOOR = 0.20;                                      // glow of a ring that moved at all (clip rule)
const SWEEP_S = 9;                                       // seconds for a sweep of the weight from 0 to 1.05
const L2D = Math.sqrt(0.8 * 0.8 + 0.4 * 0.4);           // radius of the l2 circle in the two-variable picture

export async function init(panel) {
  const v = await makeArmViewer(panel, { camera: CAMERA, q: Q0 });
  const cell3d = panel.querySelector('canvas.view3d').parentElement;
  const hud = panel.querySelector('.d22-hud'), gapTag = panel.querySelector('.d22-gap');
  const pic = plot2d(panel.querySelector('canvas.d22-pic'));
  const strip = plot2d(panel.querySelector('canvas.d22-strip'));

  // ------------------------------------------------------------------------------------------- 3D objects
  const tgtBall = marker(v.scene, COL.green, 0.022);
  const tgtHalo = marker(v.scene, COL.green, 0.048, 0.16);
  const gapLine = new Line3D(v.scene, COL.red, 2);
  const ghost = new GhostArms(v.scene, 1, { color: COL.white, opacity: 0.32, radius: 0.014 });
  ghost.set(0, Q0); ghost.visible = false;
  const rings = RING.map(() => {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.07, 0.006, 10, 56),
      new THREE.MeshStandardMaterial({ color: REST, roughness: 0.5, metalness: 0.05, emissive: 0x000000 }));
    const cap = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 0.003, 40),
      new THREE.MeshStandardMaterial({ color: REST, roughness: 0.85, metalness: 0.0 }));
    v.scene.add(ring); v.scene.add(cap);
    return { ring, cap, glow: -1, color: -1 };
  });
  const zUp = new THREE.Vector3(0, 0, 1), yUp = new THREE.Vector3(0, 1, 0), ax = new THREE.Vector3();

  // ------------------------------------------------------------------------------------------- state
  let target = T_DEFAULT.slice(), A = analyse(target), f = val(panel, 'lam'), sol = solveAt(A, f);
  let kind = 'l1', qShow = Q0.slice(), qGoal = Q0.slice(), sweeping = false, busy = false, pending = null;
  let dirty2d = true, ringsDirty = true, lastWall = performance.now();
  const showGhost = toggle(panel, 'ghost', (on) => { ghost.visible = on; });

  const active = () => sol[kind];
  function setGoal() { qGoal = S.add(Q0, active().d); ringsDirty = true; dirty2d = true; }
  setGoal();

  function placeTarget() {
    const w = world(target);
    tgtBall.position.set(...w); tgtHalo.position.set(...w);
  }
  placeTarget();

  // ring look of the active solution: colour by method, glow by |d_j| (grey when the joint did not move)
  function ringStyle() {
    const d = active().d, col = kind === 'l1' ? COL.yellow : COL.blue;
    return d.map((x) => {
      const a = Math.abs(x);
      if (a <= TOL || (kind === 'l1' && x === 0)) return { color: REST, glow: 0 };
      return { color: col, glow: FLOOR + (1 - FLOOR) * Math.min(1, a / A.fullGlow) };
    });
  }
  function updateRings(q) {
    const P = S.jointPositions(q), Jr = S.fkJac(q).Jr, st = ringsDirty ? ringStyle() : null;
    for (let j = 0; j < 6; j++) {
      const r = rings[j], o = world(P[j]), a = [Jr[0][j], Jr[1][j], Jr[2][j]], k = j < 3 ? 1.0 : 0.8;
      ax.set(...a);
      if (st && (st[j].glow !== r.glow || st[j].color !== r.color)) {
        const g = st[j].glow, rad = RING[j][1] + (0.009 + 0.004 * g) * k, tube = 0.0062 * (0.8 + 0.7 * g) * k;
        r.ring.geometry.dispose();
        r.ring.geometry = new THREE.TorusGeometry(rad, tube, 10, 56);
        r.ring.material.color.setHex(st[j].color);
        r.ring.material.emissive.setHex(g > 0 ? st[j].color : 0x000000);
        r.ring.material.emissiveIntensity = 0.25 + 0.75 * g;
        r.cap.visible = g > 0.02;
        r.cap.material.color.setHex(st[j].color).multiplyScalar(0.9);
        const rc = 0.86 * CAP[j][1];
        r.cap.scale.set(rc, 1, rc);
        r.glow = g; r.color = st[j].color;
      }
      r.ring.position.set(o[0] + a[0] * RING[j][0], o[1] + a[1] * RING[j][0], o[2] + a[2] * RING[j][0]);
      r.ring.quaternion.setFromUnitVectors(zUp, ax);
      const off = CAP[j][0] + Math.sign(CAP[j][0]) * 0.0035;
      r.cap.position.set(o[0] + a[0] * off, o[1] + a[1] * off, o[2] + a[2] * off);
      r.cap.quaternion.setFromUnitVectors(yUp, ax);
    }
    ringsDirty = false;
  }

  // ------------------------------------------------------------------------------------------- 2D: two variables
  // Lecture 4: min |d|_1 or |d|_2 subject to 2 d1 + d2 = 2. With the penalty and lambda = f lambda_max (lambda_max = 4
  // for this problem) the l1 solution is (1 - f, 0); the l2 solution with the same residual is (1 - f) (0.8, 0.4).
  // Both lie on the level line 2 d1 + d2 = 2 (1 - f): the diamond touches it at a corner, the circle anywhere.
  function drawPic() {
    pic.clear();
    const g = pic.g, W = pic.w, H = pic.h, lo = -1.25, hi = 1.6, pad = 14;
    const s = Math.min(W, H) - 2 * pad, x0 = (W - s) / 2, y0 = (H - s) / 2;
    const X = (d1) => x0 + (d1 - lo) / (hi - lo) * s, Y = (d2) => y0 + (hi - d2) / (hi - lo) * s;
    const sc = (len) => len / (hi - lo) * s;
    const k = Math.max(0, 1 - Math.min(f, 1)), on1 = kind === 'l1' ? 1 : 0.35, on2 = kind === 'l2' ? 1 : 0.35;
    g.save();
    g.beginPath(); g.rect(x0, y0, s, s); g.clip();
    // axes
    g.strokeStyle = 'rgba(242,239,233,0.35)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(X(lo), Y(0)); g.lineTo(X(hi), Y(0)); g.moveTo(X(0), Y(lo)); g.lineTo(X(0), Y(hi)); g.stroke();
    // constraint line 2 d1 + d2 = 2 and the level line through the two solutions
    g.strokeStyle = CSS.white; g.lineWidth = 1.6;
    g.beginPath(); g.moveTo(X((2 - hi) / 2), Y(hi)); g.lineTo(X((2 - lo) / 2), Y(lo)); g.stroke();
    if (f > 0.01 && k > 0) {
      g.setLineDash([5, 5]); g.strokeStyle = 'rgba(238,238,238,0.55)'; g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(X((2 * k - hi) / 2), Y(hi)); g.lineTo(X((2 * k - lo) / 2), Y(lo)); g.stroke();
      g.setLineDash([]);
    }
    // l2 circle, l1 diamond
    g.globalAlpha = on2;
    g.fillStyle = 'rgba(88,196,221,0.13)'; g.strokeStyle = CSS.blue; g.lineWidth = 2;
    g.beginPath(); g.arc(X(0), Y(0), sc(L2D * k), 0, 2 * Math.PI); g.fill(); g.stroke();
    g.globalAlpha = on1;
    g.fillStyle = 'rgba(255,235,64,0.16)'; g.strokeStyle = CSS.yellow;
    g.beginPath(); g.moveTo(X(k), Y(0)); g.lineTo(X(0), Y(k)); g.lineTo(X(-k), Y(0)); g.lineTo(X(0), Y(-k)); g.closePath();
    g.fill(); g.stroke();
    g.globalAlpha = 1;
    // the two solutions
    const p2 = [0.8 * k, 0.4 * k], p1 = [k, 0];
    g.font = '12px Inter, sans-serif'; g.textBaseline = 'bottom';
    g.globalAlpha = on2; g.fillStyle = CSS.blue;
    g.beginPath(); g.arc(X(p2[0]), Y(p2[1]), 4.5, 0, 2 * Math.PI); g.fill();
    g.textAlign = 'left'; g.fillText(`(${p2[0].toFixed(2)}, ${p2[1].toFixed(2)})`, X(p2[0]) + 7, Y(p2[1]) - 4);
    g.globalAlpha = on1; g.fillStyle = CSS.yellow;
    g.beginPath(); g.arc(X(p1[0]), Y(p1[1]), 4.5, 0, 2 * Math.PI); g.fill();
    g.textBaseline = 'top'; g.fillText(`(${p1[0].toFixed(2)}, 0)`, X(p1[0]) + 7, Y(0) + 5);
    g.globalAlpha = 1;
    // labels
    g.fillStyle = CSS.mute; g.textAlign = 'right'; g.textBaseline = 'bottom'; g.font = 'italic 13px Inter, sans-serif';
    g.fillText('d₁', X(hi) - 4, Y(0) - 3);
    g.textAlign = 'right'; g.fillText('d₂', X(0) - 5, Y(hi) + 16);
    g.font = '12px Inter, sans-serif'; g.fillStyle = CSS.white; g.textAlign = 'left'; g.textBaseline = 'middle';
    g.fillText('2d₁ + d₂ = 2', X((2 - 1.3) / 2) + 9, Y(1.3));
    if (k > 0.15) {
      g.fillStyle = CSS.yellow; g.globalAlpha = on1; g.textAlign = 'right';
      g.textBaseline = 'top'; g.fillText('ℓ₁', X(-0.5 * k) - 4, Y(-0.5 * k) + 2); g.textBaseline = 'middle';
      g.fillStyle = CSS.blue; g.globalAlpha = on2; g.textAlign = 'left';
      g.fillText('ℓ₂', X(-0.71 * L2D * k) - 16, Y(0.71 * L2D * k) - 6);
      g.globalAlpha = 1;
    }
    g.restore();
    g.fillStyle = CSS.mute; g.font = '11px Inter, sans-serif'; g.textAlign = 'left'; g.textBaseline = 'bottom';
    g.fillText(`two-variable example · same λ/λmax = ${f.toFixed(3)}`, x0 + 6, y0 + s - 6);
    g.strokeStyle = 'rgba(97,97,107,1)'; g.strokeRect(x0 + 0.5, y0 + 0.5, s - 1, s - 1);
  }

  // ------------------------------------------------------------------------------------------- 2D: path and bars
  function drawStrip() {
    strip.clear();
    const g = strip.g, W = strip.w, H = strip.h, stack = W < 620;
    const boxes = stack
      ? [{ x: 44, y: 22, w: W - 92, h: H / 2 - 50 }, { x: 44, y: H / 2 + 26, w: W - 60, h: H / 2 - 56 }]
      : [{ x: 46, y: 22, w: 0.6 * W - 100, h: H - 62 }, { x: 0.6 * W + 30, y: 22, w: 0.4 * W - 46, h: H - 62 }];
    const dmax = Math.max(10, Math.ceil(deg(A.fullGlow) / 10) * 10);
    drawPath(g, boxes[0], dmax);
    drawBars(g, boxes[1], dmax);
  }
  function axisText(g, txt, x, y, align, color = CSS.mute, base = 'middle') {
    g.fillStyle = color; g.textAlign = align; g.textBaseline = base; g.fillText(txt, x, y);
  }
  function drawPath(g, b, dmax) {
    const X = (x) => b.x + x / F_MAX * b.w, Y = (dd) => b.y + b.h - dd / dmax * b.h;
    const gmax = Math.max(1, Math.ceil(A.dist0 * 100 / 10) * 10), YG = (m) => b.y + b.h - m * 100 / gmax * b.h;
    g.font = '11px Inter, sans-serif'; g.lineWidth = 1;
    g.strokeStyle = 'rgba(242,239,233,0.25)';
    g.beginPath(); g.moveTo(b.x, b.y); g.lineTo(b.x, b.y + b.h); g.lineTo(b.x + b.w, b.y + b.h); g.lineTo(b.x + b.w, b.y); g.stroke();
    for (const x of [0, 0.25, 0.5, 0.75, 1]) axisText(g, String(x), X(x), b.y + b.h + 5, 'center', CSS.mute, 'top');
    axisText(g, 'penalty weight  λ / λmax', b.x + b.w / 2, b.y + b.h + 19, 'center', CSS.mute, 'top');
    for (let k = 0; k <= dmax; k += dmax > 40 ? 20 : 10) axisText(g, `${k}°`, b.x - 5, Y(k), 'right');
    for (let k = 0; k <= gmax; k += gmax > 30 ? 10 : 5) axisText(g, `${k}`, b.x + b.w + 5, YG(k / 100), 'left', CSS.red);
    axisText(g, 'joint displacement |dⱼ|', b.x, b.y - 12, 'left', CSS.ink);
    axisText(g, 'gap to target (cm)', b.x + b.w + 30, b.y - 12, 'right', CSS.red);
    // the stopping weights
    g.setLineDash([2, 4]); g.strokeStyle = 'rgba(255,235,64,0.35)';
    for (const j of A.order) { g.beginPath(); g.moveTo(X(A.stops[j]), b.y); g.lineTo(X(A.stops[j]), b.y + b.h); g.stroke(); }
    g.setLineDash([]);
    const curve = (D, j, color, width, alpha) => {
      g.save(); g.strokeStyle = color; g.lineWidth = width; g.globalAlpha = alpha; g.beginPath();
      A.F.forEach((x, i) => { const y = Y(Math.abs(deg(D[i][j]))); if (i) g.lineTo(X(x), y); else g.moveTo(X(x), y); });
      g.stroke(); g.restore();
    };
    for (let j = 0; j < 6; j++) if (A.D2.some((d) => Math.abs(d[j]) > TOL)) curve(A.D2, j, CSS.blue, 1.3, kind === 'l2' ? 0.9 : 0.45);
    for (let j = 0; j < 6; j++) if (A.D1.some((d) => d[j] !== 0)) curve(A.D1, j, CSS.yellow, 2.2, kind === 'l1' ? 1 : 0.5);
    // gap (the same for both: the l2 weight is matched to it)
    g.save(); g.strokeStyle = CSS.red; g.lineWidth = 1.6; g.beginPath();
    A.F.forEach((x, i) => { if (i) g.lineTo(X(x), YG(A.G1[i])); else g.moveTo(X(x), YG(A.G1[i])); });
    g.stroke(); g.restore();
    // joint numbers at the left end of the l1 curves
    g.font = '600 10px Inter, sans-serif';
    for (let j = 0; j < 6; j++) {
      if (!A.D1.some((d) => d[j] !== 0)) continue;
      const y = Y(Math.abs(deg(A.D1[0][j])));
      g.fillStyle = '#000'; g.strokeStyle = CSS.yellow; g.lineWidth = 1.3;
      g.beginPath(); g.arc(b.x + 9, y, 7, 0, 2 * Math.PI); g.fill(); g.stroke();
      axisText(g, String(j + 1), b.x + 9, y + 0.5, 'center', CSS.yellow);
    }
    // cursor at the current weight, with the exact solutions
    const xc = X(Math.min(f, F_MAX));
    g.strokeStyle = 'rgba(238,238,238,0.75)'; g.lineWidth = 1.2;
    g.beginPath(); g.moveTo(xc, b.y); g.lineTo(xc, b.y + b.h); g.stroke();
    const dots = (d, color, r) => d.forEach((x, j) => {
      if (Math.abs(x) <= TOL) return;
      g.fillStyle = color; g.beginPath(); g.arc(xc, Y(Math.abs(deg(x))), r, 0, 2 * Math.PI); g.fill();
    });
    dots(sol.l2.d, CSS.blue, 2.8); dots(sol.l1.d, CSS.yellow, 3.4);
    g.fillStyle = CSS.red; g.beginPath(); g.arc(xc, YG(sol.l1.miss), 3.2, 0, 2 * Math.PI); g.fill();
  }
  function drawBars(g, b, dmax) {
    const Y = (dd) => b.y + b.h - dd / dmax * b.h, slot = b.w / 6, bw = Math.min(14, slot * 0.3);
    g.font = '11px Inter, sans-serif';
    g.strokeStyle = 'rgba(242,239,233,0.25)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(b.x, b.y); g.lineTo(b.x, b.y + b.h); g.lineTo(b.x + b.w, b.y + b.h); g.stroke();
    for (let k = 0; k <= dmax; k += dmax > 40 ? 20 : 10) axisText(g, `${k}°`, b.x - 5, Y(k), 'right');
    axisText(g, 'joint', b.x + b.w / 2, b.y + b.h + 19, 'center', CSS.mute, 'top');
    axisText(g, 'now', b.x, b.y - 12, 'left', CSS.ink);
    g.fillStyle = CSS.blue; g.fillRect(b.x + b.w - 74, b.y - 17, 9, 9); axisText(g, 'ℓ₂', b.x + b.w - 61, b.y - 12, 'left', CSS.blue);
    g.fillStyle = CSS.yellow; g.fillRect(b.x + b.w - 36, b.y - 17, 9, 9); axisText(g, 'ℓ₁', b.x + b.w - 23, b.y - 12, 'left', CSS.yellow);
    for (let j = 0; j < 6; j++) {
      const cx = b.x + slot * (j + 0.5);
      axisText(g, String(j + 1), cx, b.y + b.h + 5, 'center', CSS.mute, 'top');
      [[sol.l2.d[j], CSS.blue, -1, kind === 'l2'], [sol.l1.d[j], CSS.yellow, 1, kind === 'l1']].forEach(([x, color, side, on]) => {
        const xb = cx + side * (bw / 2 + 1) - bw / 2;
        g.globalAlpha = on ? 1 : 0.45;
        if (Math.abs(x) <= TOL || x === 0) {
          g.strokeStyle = CSS.grey; g.lineWidth = 1.3;
          g.beginPath(); g.arc(xb + bw / 2, b.y + b.h - 5, 3.6, 0, 2 * Math.PI); g.stroke();
        } else {
          const y = Y(Math.min(dmax, Math.abs(deg(x))));
          g.fillStyle = color; g.fillRect(xb, y, bw, b.y + b.h - y);
        }
        g.globalAlpha = 1;
      });
    }
  }

  // ------------------------------------------------------------------------------------------- readouts
  const mm = (m) => (m < 0.01 ? `${fmt(m * 1000, 2)} mm` : `${fmt(m * 100, 1)} cm`);
  function texts() {
    const L1 = `<b class="c-yellow">ℓ₁</b> moves <b>${sol.l1.moving}</b> joint${sol.l1.moving === 1 ? '' : 's'}`;
    const L2 = `<b class="c-blue">ℓ₂</b> moves <b>${sol.l2.moving}</b>`;
    readout(panel, 'info', `${L1} · ${L2} · gap <b class="c-red">${mm(sol.l1.miss)}</b>`);
    const st = A.order.map((j) => `${j}: ${fmt(A.stops[j], 2)}`).join(', ');
    readout(panel, 'tinfo', busy ? 'computing the path…'
      : `λmax <b>${fmt(A.lamMax, 4)}</b> · distance <b>${fmt(A.dist0 * 100, 1)} cm</b> · joints stop at λ/λmax ${st}`);
    hud.innerHTML = kind === 'l1'
      ? `<b class="c-yellow">ℓ₁ penalty</b><span>λ = ${sol.lam >= 1e-3 ? sol.lam.toFixed(4) : sol.lam.toExponential(2)}</span>`
      : `<b class="c-blue">ℓ₂ penalty</b><span>${Number.isFinite(sol.l2.mu) ? `weight ${sol.l2.mu.toPrecision(3)} (same gap)` : 'weight ∞ (nothing moves)'}</span>`;
  }

  function resolve() {
    sol = solveAt(A, f);
    setGoal();
    texts();
  }

  // ------------------------------------------------------------------------------------------- controls
  onInput(panel, 'lam', () => { f = val(panel, 'lam'); resolve(); });
  const getKind = seg(panel, 'kind', (k) => { kind = k; setGoal(); texts(); });
  kind = getKind();
  const sweepBtn = panel.querySelector('[data-ctl="sweep"]');
  onClick(panel, 'sweep', () => {
    sweeping = !sweeping;
    sweepBtn.classList.toggle('on', sweeping);
    if (sweeping && f >= F_MAX - 1e-9) setVal(panel, 'lam', F_MIN);
  });
  function retarget() {
    pending = null;
    A = analyse(target);
    busy = false;
    resolve();
  }
  for (const [i, name] of ['tx', 'ty', 'tz'].entries()) {
    onInput(panel, name, () => {
      target[i] = val(panel, name);
      placeTarget();
      busy = true; texts();
      if (pending) clearTimeout(pending);
      pending = setTimeout(retarget, 250);           // recompute the path once the slider rests
    });
  }
  onClick(panel, 'reset', () => {
    ['tx', 'ty', 'tz'].forEach((name, i) => { const el = panel.querySelector(`[data-ctl="${name}"]`); el.value = T_DEFAULT[i]; el.dispatchEvent(new Event('input', { bubbles: true })); });
  });
  texts();

  // ------------------------------------------------------------------------------------------- frame loop
  const proj = new THREE.Vector3();
  v.onFrame(() => {
    const now = performance.now(), dtw = Math.min(0.5, (now - lastWall) / 1000);   // wall clock (display only)
    lastWall = now;
    if (sweeping && !busy) {
      const nf = Math.min(F_MAX, f + Math.min(0.03, dtw * (F_MAX - F_MIN) / SWEEP_S));
      setVal(panel, 'lam', nf);                      // fires 'input' -> exact solve at the new weight
      if (nf >= F_MAX - 1e-9) { sweeping = false; sweepBtn.classList.remove('on'); }
    }
    // the arm travels to the solution (display only: the angles are interpolated, as in the clip)
    let moved = false;
    const a = 1 - Math.exp(-dtw / 0.16);
    for (let j = 0; j < 6; j++) {
      const e = qGoal[j] - qShow[j];
      if (Math.abs(e) > 1e-6) { qShow[j] += a * e; moved = true; } else if (e !== 0) { qShow[j] = qGoal[j]; moved = true; }
    }
    if (moved || ringsDirty) { v.setArm(qShow); v.forward(); updateRings(qShow); }
    // gap from the tool to the target (red), labelled with the solution's gap
    const tool = v.tool(), tw = world(target), gap = active().miss;
    if (gap > 0.002) {
      gapLine.visible = true; gapLine.set([tool, tw]);
      proj.set((tool[0] + tw[0]) / 2, (tool[1] + tw[1]) / 2, (tool[2] + tw[2]) / 2).project(v.camera);
      const W = cell3d.clientWidth, H = cell3d.clientHeight;
      gapTag.style.display = proj.z < 1 ? 'block' : 'none';
      gapTag.style.left = `${((proj.x + 1) / 2) * W}px`;
      gapTag.style.top = `${((1 - proj.y) / 2) * H}px`;
      gapTag.textContent = mm(gap);
    } else { gapLine.visible = false; gapTag.style.display = 'none'; }
    if (dirty2d) { drawPic(); drawStrip(); dirty2d = false; }
  });
  window.addEventListener('resize', () => { dirty2d = true; });

  // ------------------------------------------------------------------------------------------- for tests
  panel._state = () => {
    const r = (x, k = 4) => +x.toFixed(k);
    const toolSim = world(S.fk(qShow).p), toolView = v.tool();
    return {
      target, f, kind, busy, sweeping, ghost: showGhost(),
      lambdaMax: A.lamMax, lambdaMaxJoint: A.lamMaxJoint, distanceStartToTarget_mm: A.dist0 * 1000,
      stops_over_lambdaMax: A.stops, orderOfStopping: A.order, analyse_ms: r(A.ms, 1), solve_ms: r(sol.ms, 1),
      l1: { lambda: sol.lam, d_deg: sol.l1.d.map((x) => r(deg(x))), moving: sol.l1.moving, gap_mm: sol.l1.miss * 1000,
        kkt: sol.l1.kkt, polished: sol.l1.polished },
      l2: { mu: sol.l2.mu, d_deg: sol.l2.d.map((x) => r(deg(x))), moving: sol.l2.moving, gap_mm: sol.l2.miss * 1000,
        kkt: sol.l2.kkt, evals: sol.l2.evals },
      picture2d: { l1: [r(1 - Math.min(f, 1)), 0], l2: [r(0.8 * (1 - Math.min(f, 1))), r(0.4 * (1 - Math.min(f, 1)))] },
      shown: { q_deg: qShow.map((x) => r(deg(x), 3)), arrived: S.norm(S.sub(qShow, qGoal)) < 1e-4,
        toolViewerMinusSim_m: S.norm(S.sub(toolView, toolSim)) },
      ringsLit: rings.filter((x) => x.glow > 0).length,
      ringColours: rings.map((x) => (x.glow > 0 ? (x.color === COL.yellow ? 'yellow' : 'blue') : 'grey')),
      ref: REF,
    };
  };
}
