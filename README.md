# Project website

Static project site with two routes:

- `/`: project entry page.
- `/mtod/`: **Sample, Then Refine: Training-Free Diffusion for Offline Global Trajectory Planning**.

The project page displays its title, a single annotated BRHP diagram, and required
template/license credits. It includes no project-author names, affiliations,
videos, JavaScript, or analytics.
This does not anonymize the GitHub repository or its commit history.

## Local preview

From the repository root, run:

```sh
python -m http.server --bind 127.0.0.1 8000
```

Open <http://127.0.0.1:8000/> or <http://127.0.0.1:8000/mtod/>.
The site is plain HTML/CSS; no build step is required.

The diagram is editable vector artwork at `mtod/static/images/brhp-overview.svg`.
All explanatory labels are inside the figure; click it to open the full-size SVG.

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
