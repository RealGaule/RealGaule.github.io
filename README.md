# Project website

Static project site with two routes:

- `/`: project entry page.
- `/mtod/`: **Sample, Then Refine: Training-Free Diffusion for Offline Global Trajectory Planning**.

The project page displays its title, a short English Motivation, one continuous
BRHP schematic, an illustrative single-denoise particle animation, and three
saved-trajectory GIFs labeled only by task name.
Required template/license credits remain at the bottom. There are no project-author
names, affiliations, JavaScript, or analytics.
This does not anonymize the GitHub repository or its commit history.

## Local preview

From the repository root, run:

```sh
python -m http.server --bind 127.0.0.1 8000
```

Open <http://127.0.0.1:8000/> or <http://127.0.0.1:8000/mtod/>.
The site is plain HTML/CSS; no build step is required.

The diagram is editable vector artwork at `mtod/static/images/brhp-overview.svg`.
The figure is one left-to-right tree with a central selected lineage, omitted
intermediate expansion, and schematic objective landscapes at the first and final
layers. Its short labels are in English; click it to open the full-size SVG.

The particle animation illustrates the proposed **single-denoise** MToD variant
on a one-dimensional analytic mixture, not a robot task or benchmark result.
The same particles first sample stochastically and then undergo deterministic,
annealed mean-shift refinement along one continuously decreasing noise schedule.
There is no particle restart or second noise schedule at the phase boundary.
The right panel distinguishes the empirical particle density from the smoothed
target density; mode-seeking refinement need not preserve the target distribution.
This illustration does not change the optimizer: the current benchmark MToD
implementation still uses its two-process sample/refine design.
The layout is inspired by the density-evolution illustrations on the
[Temporal Score Rescaling project page](https://temporalscorerescaling.github.io/);
the simulation and rendered assets here are original, not copied from that site.
Recreate it with `python tools/render_mtod_density.py` after installing NumPy,
Matplotlib, and Pillow. The generator and its JSON manifest record the toy target,
seed, noise schedule, switch point, and particle updates.

The GIFs replay saved MPPI-BRHP consistency-test trajectories for Double Cart-Pole,
Walker, and Push T; they are not new optimization runs or claims of task success.
They are stored in `mtod/static/gifs/`; clicking a GIF opens its full-size version.

## GitHub Pages and domains

Pages is enabled. In **Settings → Pages**, verify **Deploy from a branch** with
the **main** branch and **/ (root)** folder. The `.nojekyll` file keeps this a
plain static site.

A custom domain can be added later through Pages settings and the domain's DNS
configuration. The project path remains `/mtod/`; relative asset links do not
need changing. No custom domain or `CNAME` file is preset here.

The earlier template repository is retained as a reference and does not need a
separate Pages deployment.

## Attribution and licenses

The `/mtod/` page is adapted from Eliahu Horwitz's
[Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template),
which incorporates parts of the [Nerfies project page](https://nerfies.github.io/).
This adaptation removes demo content and scripts and adds the project title
and an original BRHP diagram. The adapted project page is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

The bundled [Bulma](https://github.com/jgthms/bulma) stylesheet is separately
licensed under the [MIT License](mtod/static/css/BULMA-LICENSE.txt).
Retain the applicable attribution and license notices when reusing these files.
