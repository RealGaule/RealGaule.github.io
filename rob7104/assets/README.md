# Shared assets for the ROB7104 lab pages

Owned by the Lab 4 session. Lab pages (`../lab3/`, `../lab4/`) link these files; they do not copy or edit them.

## Files

- `site.css`, `site.js`: page look and tab/autoplay behaviour, following the template page
  https://alaya-lab.github.io/pwm/ (hero, link pills, about cards, section heads, pill tabs with panels,
  media cells, control bars, collapsible "why" blocks, gallery grid, summary table, footer).
- `mjviewer.js`: live 3D viewer, a MuJoCo model (WebAssembly) drawn with three.js, with mouse orbit, zoom and pan.
- `vendor/three.module.min.js`, `vendor/OrbitControls.js`, `vendor/OBJLoader.js`: three@0.160.0 (MIT).
- `vendor/mujoco/mujoco.js`, `vendor/mujoco/mujoco.wasm`: `@mujoco/mujoco@3.12.0`, single-threaded build (Apache-2.0).
  It needs no special HTTP headers, so it runs on GitHub Pages.

## Page skeleton

```html
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet" />
<link rel="stylesheet" href="../assets/site.css" />
<script type="importmap">{"imports": {"three": "../assets/vendor/three.module.min.js"}}</script>
...
<body><div class="glow" aria-hidden="true"></div><main class="wrap"> ... </main>
<script src="../assets/site.js"></script>
```

Tabs: a `.feature-tabs` group switches the `.feature-panel` elements of its own `<section>`
(`data-feature` on the tab and on the panel must match). A panel receives `panel:show` / `panel:hide` events.
Elements with `data-live` receive `live:on` / `live:off` when they scroll into or out of view.

## `mjviewer.js`

```js
import { MjViewer, Trail, THREE, loadMujoco, loadModel } from '../assets/mjviewer.js';

const v = new MjViewer(canvas, { lookat: [0, 0, 1.5], distance: 4.6, azimuth: 70, elevation: -12 });
await v.load('./models/drone.mjb');          // compiled model; see "Models" below
v.onFrame((dt, t) => {                       // once per animation frame, before drawing
  v.setJoint('main_q1', angle);              // write qpos by joint name
  v.setMocap('drone', [x, 0, 2.35], pitch);  // move a mocap body
  v.forward();                               // mj_forward + copy geom poses to three.js
});
v.autoRun();                                 // run only while the canvas is on screen (uses site.js events)
```

- Camera options use MuJoCo's free-camera convention (`lookat`, `distance`, `azimuth`, `elevation` in degrees).
  The viewer is z-up. Drag = orbit, wheel = zoom, right-drag = pan, double-click = back to the start view.
- `v.model`, `v.data`, `v.mujoco` are the MuJoCo objects; `v.step()` calls `mj_step`. Call `v.sync()` after
  stepping yourself to copy poses to the meshes, or just call `v.forward()`.
- `v.scene`, `v.camera`, `v.renderer`, `v.controls` are the three.js objects, for adding your own objects
  (surfaces, particles, instanced meshes). `v.meshes[g]` is the mesh of geom `g` (or `null` if not drawn).
- Geoms in groups 3 and above and fully transparent geoms are not drawn. Plane geoms are not drawn either:
  the viewer supplies its own dark grid floor (`floor: false` to turn it off).
- `v.id('joint' | 'body' | 'site' | 'geom', name)`, `v.sitePos(name)`, `v.getJoint(name)`,
  `v.refreshGeom(g)` (after changing `model.geom_size`).
- `new Trail(v.scene, 0xffeb40, 60).push([x, y, z])`: fading line behind a moving point.
- Integer address arrays that are 64-bit in MuJoCo (for example `tex_adr`) arrive as BigInt: wrap in `Number()`.

## Models

Compile the model in Python with the SAME MuJoCo version (3.12.0) and save it with
`mujoco.mj_saveModel(model, 'name.mjb')`. The `.mjb` holds meshes and textures, so the page needs no other
asset files. `loadModel(url)` also accepts an `.xml` that has no external assets.

## Testing without a GPU

`google-chrome --headless=new --no-sandbox --use-angle=swiftshader --enable-unsafe-swiftshader ...` renders
WebGL in software. The Lab 4 build folder has `build/web/shot.sh` and `serve.py` (static server plus
`POST /report/<name>` so a test page can send back a canvas PNG and a JSON of numbers).

Checked on 2026-11-16 (headless Chrome, software rendering): module load plus drone model about 2 s,
textured mesh, capsules, boxes, spheres, transparent ghosts, shadows and trails draw correctly.
Not checked: Firefox, Safari, phones, and mouse interaction itself (headless).
