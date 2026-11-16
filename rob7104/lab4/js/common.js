// Helpers shared by the Lab 4 demos.
import { MjViewer, Trail, THREE } from '../../assets/mjviewer.js';
export { MjViewer, Trail, THREE };

export const COL = { green: 0x83c167, red: 0xfc6255, blue: 0x58c4dd, yellow: 0xffeb40, white: 0xeeeeee, grey: 0x8c8c94, accent: 0xe8b44f };
export const CSS = { green: '#83c167', red: '#fc6255', blue: '#58c4dd', yellow: '#ffeb40', white: '#f2efe9', grey: '#9a97a3', faint: '#4a4c55', accent: '#e8b44f' };

/** Create a viewer on the panel's canvas.view3d, load a model, and run it only while visible. */
export async function makeViewer(panel, modelUrl, camera) {
  const canvas = panel.querySelector('canvas.view3d');
  const veil = panel.querySelector('.loading');
  const v = new MjViewer(canvas, Object.assign({ preserveDrawingBuffer: /[?&]shot/.test(location.search) }, camera));
  await v.load(new URL(`../models/${modelUrl}`, import.meta.url).href);
  if (veil) veil.classList.add('gone');
  v.autoRun();
  panel.addEventListener('panel:hide', () => v.stop());
  panel.addEventListener('panel:show', () => v.start());
  v.start();
  return v;
}

/** Read a control by its data-ctl name. Sliders return numbers. */
export function ctl(panel, name) {
  const el = panel.querySelector(`[data-ctl="${name}"]`);
  if (!el) throw new Error(`no control ${name}`);
  return el;
}
export const val = (panel, name) => Number(ctl(panel, name).value);
export function onInput(panel, name, fn) { ctl(panel, name).addEventListener('input', fn); }
export function onClick(panel, name, fn) { ctl(panel, name).addEventListener('click', fn); }
export function readout(panel, name, html) { ctl(panel, name).innerHTML = html; }

/** Segmented buttons: <span class="seg" data-ctl="x"><button data-v="a" class="on">..</button>..</span> */
export function seg(panel, name, fn) {
  const box = ctl(panel, name);
  box.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    box.querySelectorAll('button').forEach((o) => o.classList.toggle('on', o === b));
    fn(b.dataset.v);
  }));
  return () => box.querySelector('button.on').dataset.v;
}

/** 2D plot canvas with device-pixel scaling. Returns { g, w, h, clear } in CSS pixels. */
export function plot2d(canvas) {
  const ctx = canvas.getContext('2d');
  const o = { g: ctx, w: 0, h: 0 };
  o.fit = () => {
    const r = Math.min(window.devicePixelRatio || 1, 2);
    o.w = canvas.clientWidth; o.h = canvas.clientHeight;
    if (canvas.width !== Math.round(o.w * r) || canvas.height !== Math.round(o.h * r)) {
      canvas.width = Math.round(o.w * r); canvas.height = Math.round(o.h * r);
    }
    ctx.setTransform(r, 0, 0, r, 0, 0);
  };
  o.clear = () => { o.fit(); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, o.w, o.h); };
  return o;
}

/** Fixed-step accumulator: calls step(h) enough times to advance by dt * speed. */
export function stepper(h) {
  let acc = 0;
  return (dt, speed, step) => {
    acc += dt * speed;
    let n = 0;
    while (acc >= h && n < 400) { step(h); acc -= h; n++; }
    if (n === 400) acc = 0;
  };
}

export const deg = (r) => r * 180 / Math.PI;
export const rad = (d) => d * Math.PI / 180;
