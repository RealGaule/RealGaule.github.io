"""Figures for the note "one iteration, four quantities, three levers".

levers.svg      three ways a sample or prior can enter a sampling-based
                optimiser: move the proposal (seed), reshape its covariance
                (prior), change the space (hierarchy).
quantities.svg  the four quantities that decide what the iteration does:
                (a) weight collapse boundary, (b) per-step progress cap,
                (c) discovery as Gaussian measure with the product law for
                conjunctive requirements, (d) mode retention of warm starts.
Panel (b) re-runs the quadratic-basin Monte Carlo (seconds); (c) and (d)
use the numbers measured in hydrax on 2026-10-13/16 (hard-coded below).
Run:  python3 tools/render_unified_view.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Ellipse, FancyArrowPatch
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content/posts/unified-view-samples-priors"
OUT.mkdir(parents=True, exist_ok=True)

cjk = [f.name for f in font_manager.fontManager.ttflist if "Noto Sans CJK" in f.name]
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": (["Noto Sans CJK SC"] if "Noto Sans CJK SC" in cjk else cjk)
        + ["DejaVu Sans"],
        "svg.fonttype": "path",
        "font.size": 9.5,
        "axes.titlesize": 10,
        "mathtext.fontset": "dejavusans",
    }
)
BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#777777"
rng = np.random.default_rng(0)


# ----------------------------------------------------------------- levers
def draw_cloud(ax, centre, cov, color, levels=(1.0, 2.0), lw=1.2, label=None):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    for i, k in enumerate(levels):
        ax.add_patch(
            Ellipse(
                centre,
                2 * k * np.sqrt(vals[0]),
                2 * k * np.sqrt(vals[1]),
                angle=angle,
                fill=False,
                ec=color,
                lw=lw,
                ls="-" if i == 0 else "--",
                label=label if i == 0 else None,
            )
        )
    ax.plot(*centre, "o", color=color, ms=4)


def draw_target(ax, centre, size, color=GREEN, label=None):
    ax.add_patch(
        Ellipse(centre, size[0], size[1], fc=color, ec="none", alpha=0.35, label=label)
    )


fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), constrained_layout=True)
for ax in axes:
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

ax = axes[0]
draw_target(ax, (2.0, 1.6), (0.9, 0.7), label="good region")
draw_cloud(ax, (-1.2, -0.8), np.diag([0.5, 0.5]), GREY, label="proposal before")
draw_cloud(ax, (2.0, 1.4), np.diag([0.12, 0.12]), BLUE, label="proposal after")
ax.add_patch(FancyArrowPatch((-1.0, -0.6), (1.75, 1.2), arrowstyle="-|>", mutation_scale=12, color=BLUE, lw=1.2))
ax.text(0.1, -0.1, "seed", color=BLUE, ha="center", va="top")
ax.set_title("(1) seed: move the proposal, enter below $\\sigma_e^*$")
ax.legend(loc="lower left", frameon=False, fontsize=7.6)

ax = axes[1]
draw_target(ax, (2.1, 1.7), (0.9, 0.7))
draw_cloud(ax, (-0.6, -0.5), np.diag([0.5, 0.5]), GREY, label="isotropic")
R = np.array([[np.cos(0.68), -np.sin(0.68)], [np.sin(0.68), np.cos(0.68)]])
draw_cloud(ax, (-0.6, -0.5), R @ np.diag([2.6, 0.25]) @ R.T, VERM, label="prior-aligned")
ax.text(1.0, -1.9, "same centre, mass moved\nalong the prior's directions", color=VERM, ha="center", fontsize=8.2)
ax.set_title("(2) prior: reshape the covariance")
ax.legend(loc="lower left", frameon=False, fontsize=7.6)

ax = axes[2]
draw_target(ax, (1.9, 1.5), (0.22, 0.16))
ax.text(1.9, 2.0, "control space\n(volume $\\to 0$)", ha="center", fontsize=8.2, color=GREEN)
draw_target(ax, (-1.3, -1.1), (2.4, 1.9))
ax.text(-1.3, -1.1, "object space\n(volume $O(1)$)", ha="center", va="center", fontsize=8.2, color="#1f5f3f")
draw_cloud(ax, (0.3, 0.2), np.diag([0.6, 0.6]), GREY)
ax.set_title("(3) space: change what is sampled")
fig.savefig(OUT / "levers.svg")
fig.savefig("/tmp/levers.png", dpi=110)
plt.close(fig)


# ------------------------------------------------------------- quantities
def chi(R):
    return (1 + R) / np.sqrt(1 + 2 * R)


fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.4), constrained_layout=True)

# (a) collapse boundary
ax = axes[0]
Rs = np.logspace(-2, 2, 400)
K = 256
for D, c in ((2, GREEN), (10, BLUE), (40, VERM)):
    ess = np.minimum(1.0, chi(Rs) ** (-D))
    ax.plot(Rs, ess, color=c, label=f"$D={D}$")
for Kstar, ls in ((256, "-"), (10**6, ":")):
    x = Kstar ** (1 / 40)
    Rstar = (x**2 - 1) + np.sqrt((x**2 - 1) * x**2)
    ax.axvline(Rstar, color=VERM, ls=ls, lw=0.9)
    ax.text(Rstar * 1.08, 0.55 if Kstar == 256 else 0.3, f"$R^*$, $K={Kstar:,}$".replace(",", "\\,"), color=VERM, fontsize=7.8, rotation=90, va="center")
ax.axvspan(5, 100, color=GREY, alpha=0.12)
ax.text(22, 0.85, "PushT chains\n(ESS = 1)", ha="center", fontsize=8, color=GREY)
ax.set_xscale("log")
ax.set_xlabel("$R=\\sigma^2 a/\\lambda$")
ax.set_ylabel("ESS / $K$  ($K=256$)")
ax.set_title("(a) weight collapse: ESS $\\approx K\\,\\chi(R)^{-D}$")
ax.legend(frameon=False, fontsize=8, loc="center left")

# (b) progress cap (re-run the quadratic-basin Monte Carlo)
ax = axes[1]
D, K, sigma, a = 40, 256, 1.0, 1.0
rho = 3.0 * np.sqrt(D)
U = np.zeros(D)
U[0] = rho
lams = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
mc, exact = [], []
for lam in lams:
    prog = []
    for _ in range(300):
        V = U + sigma * rng.standard_normal((K, D))
        C = 0.5 * a * np.sum(V**2, axis=1)
        lw = -C / lam
        w = np.exp(lw - lw.max())
        w /= w.sum()
        prog.append(rho - np.linalg.norm(w @ V))
    mc.append(np.mean(prog))
    Rv = sigma**2 * a / lam
    exact.append(rho - rho / (1 + Rv))
c1k = np.mean([np.max(rng.standard_normal(K)) for _ in range(20000)])
cap = sigma * c1k
ax.plot(lams, exact, "o-", color=GREY, label="exact mean shift $T$")
ax.plot(lams, mc, "s-", color=BLUE, label="Monte Carlo $\\hat T_K$")
ax.axhline(cap, color=VERM, ls="--", lw=1, label="$\\sigma\\,c_{1,K}$ (hull cap)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$\\lambda$")
ax.set_ylabel("progress per step")
ax.set_title(f"(b) progress cap ($D=40$, $K=256$, $\\sigma=1$)")
ax.legend(frameon=False, fontsize=7.8, loc="lower left")

# (c) discovery: Gaussian measure and the product law
ax = axes[2]
ds = np.linspace(0.0, 3.5, 300)
p1 = norm.sf(ds)
for m, c, lab in ((1, GREEN, "one direction"), (2, BLUE, "two at once"), (3, VERM, "three at once")):
    ax.plot(ds, 1.0 / p1**m, color=c, label=lab)
# toy measurements: threshold 2 sigma, isotropic d/sigma = 2 -> 21 iterations of 64;
# aligned (sigma doubled) d/sigma = 1 -> 1 iteration
ax.plot([2.0], [21 * 64], "s", color=BLUE, ms=6)
ax.plot([1.0], [1 * 64], "s", color=BLUE, ms=6)
ax.annotate("toy: isotropic,\n21 iterations x 64", (2.0, 21 * 64), (2.2, 2e4), fontsize=7.6, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))
ax.annotate("aligned variance\n(1 iteration)", (1.0, 64), (0.35, 900), fontsize=7.6, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))
ax.set_yscale("log")
ax.set_xlabel("distance to the region / $\\sigma$")
ax.set_ylabel("expected proposals to first hit")
ax.set_title("(c) discovery: $1/\\bar\\Phi(d/\\sigma)^m$")
ax.legend(frameon=False, fontsize=8, loc="upper left")

# (d) retention of warm starts (hydrax, kinematic PushT, 2026-10-16)
ax = axes[3]
sig_e = np.array([4.0, 1.212, 0.495, 0.273, 0.15])
kept = {
    "seed in the wide basin (below)": ([3, 3, 4, 4, 4], GREEN),
    "seed in the narrow basin (above)": ([1, 2, 2, 4, 4], BLUE),
    "seed in a stalled basin": ([0, 1, 2, 4, 4], VERM),
}
for lab, (v, c) in kept.items():
    ax.plot(sig_e, np.array(v) / 4.0, "o-", color=c, label=lab)
ax.axvspan(0.1, 0.3, color=GREY, alpha=0.12)
ax.text(0.17, 0.08, "refine", ha="center", fontsize=8, color=GREY)
ax.text(1.5, 0.08, "re-explore", ha="center", fontsize=8, color=GREY)
ax.set_xscale("log")
ax.set_xlabel("entry bandwidth $\\sigma_e$ (normalized)")
ax.set_ylabel("fraction keeping the seed's mode")
ax.set_ylim(-0.05, 1.08)
ax.set_title("(d) retention of a seed's mode")
ax.legend(frameon=False, fontsize=7.6, loc="center left")

fig.savefig(OUT / "quantities.svg")
fig.savefig("/tmp/quantities.png", dpi=110)
print("c_{1,256} =", round(float(c1k), 3), "| MC progress:", np.round(mc, 2).tolist(), "| exact:", np.round(exact, 2).tolist())
print("written", OUT / "levers.svg", OUT / "quantities.svg")
