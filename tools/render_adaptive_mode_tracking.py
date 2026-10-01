"""Two-basin example for the adaptive mode-tracking note.

Model (well-separated double well, mixture approximation):
    C(v) = min(c1 + A (v-a)^2 / 2,  c2 + A (v+a)^2 / 2),  dC = c2 - c1 >= 0
    pi_lambda ≈ w N(a, s^2) + (1-w) N(-a, s^2),  s^2 = lambda / A,
    log(w / (1-w)) = dC / lambda
    p_{sigma,lambda} = pi_lambda * N(0, sigma^2)  (same mixture, v = s^2 + sigma^2)
    M(z) = Cov[V | z] / sigma^2 = 1 + sigma^2 d^2/dz^2 log p(z)

Panels: (a) critical points of p versus sigma (pitchfork vs fold),
(b) M along each branch, (c) where p is bimodal in the (lambda, sigma) plane.
The script also checks the Robertson-Fryer bimodality condition against a
grid count of modes.
Run:  python3 tools/render_adaptive_mode_tracking.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content/posts/adaptive-mode-tracking/two-basin.svg"

# ---------------------------------------------------------------- parameters
A_HALF = 1.0     # basin centres at +a (deep) and -a (shallow)
CURV = 20.0      # basin curvature A
DC = 2.0         # depth gap dC = c2 - c1
LAM = 1.0        # temperature used in panels (a) and (b)

# offset so that no grid point sits exactly on a basin centre (exact zeros)
Z = np.linspace(-2.0, 2.0, 16001) + 1.234567e-5


def mixture_terms(z, sigma, lam, dc):
    """Responsibility r of the deep (+a) basin, and v = s^2 + sigma^2."""
    v = lam / CURV + sigma**2
    logit_w = dc / lam
    # log [w N(z; a, v)] - log [(1-w) N(z; -a, v)] = logit_w + 2 a z / v
    r = 1.0 / (1.0 + np.exp(-(logit_w + 2.0 * A_HALF * z / v)))
    return r, v


def dlogp(z, sigma, lam, dc):
    r, v = mixture_terms(z, sigma, lam, dc)
    return ((2.0 * r - 1.0) * A_HALF - z) / v


def m_ratio(z, sigma, lam, dc):
    """M = 1 + sigma^2 d2 log p = s^2/v + 4 a^2 sigma^2 r(1-r) / v^2."""
    r, v = mixture_terms(z, sigma, lam, dc)
    return 1.0 + sigma**2 * (-1.0 / v + 4.0 * A_HALF**2 * r * (1.0 - r) / v**2)


def critical_points(sigma, lam, dc):
    """Sorted critical points of log p and whether each one is a mode."""
    g = dlogp(Z, sigma, lam, dc)
    idx = np.nonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]
    pts = []
    for i in idx:
        zc = brentq(dlogp, Z[i], Z[i + 1], args=(sigma, lam, dc))
        pts.append((zc, m_ratio(zc, sigma, lam, dc) < 1.0))
    return pts


def g_rf(d):
    """Robertson-Fryer: bimodal iff d > 1 and |logit w| < g(d)."""
    return 2.0 * d * np.sqrt(d * d - 1.0) - 2.0 * np.arccosh(d)


def sigma_fold(lam, dc):
    """Bandwidth at which the shallow mode is born (fold), or nan."""
    target = dc / lam
    if target <= 0.0:
        d = 1.0
    else:
        d = brentq(lambda x: g_rf(x) - target, 1.0 + 1e-12, 1e4)
    s2 = A_HALF**2 / d**2 - lam / CURV
    return np.sqrt(s2) if s2 > 0 else np.nan


def check_robertson_fryer(n=400, seed=0):
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(n):
        lam = rng.uniform(0.1, 3.0)
        sigma = rng.uniform(0.02, 1.3)
        dc = rng.uniform(0.0, 4.0)
        n_modes = sum(is_mode for _, is_mode in critical_points(sigma, lam, dc))
        v = lam / CURV + sigma**2
        d = A_HALF / np.sqrt(v)
        bimodal = d > 1.0 and dc / lam < g_rf(d)
        bad += int((n_modes == 2) != bimodal)
    return bad, n


# ---------------------------------------------------------------- branches
sigma_c = np.sqrt(A_HALF**2 - LAM / CURV)
sigma_f = sigma_fold(LAM, DC)
# dense near the fold so the shallow branch is drawn down to M = 1
sig = np.unique(np.concatenate([np.linspace(1.35, 0.03, 900),
                                sigma_f - np.geomspace(1e-9, 0.02, 200)]))[::-1]
branches = {}
for case, dc in (("sym", 0.0), ("asym", DC)):
    deep = np.full(sig.shape, np.nan)
    shallow = np.full(sig.shape, np.nan)
    saddle = np.full(sig.shape, np.nan)
    m_deep = np.full(sig.shape, np.nan)
    m_shallow = np.full(sig.shape, np.nan)
    for k, s in enumerate(sig):
        pts = critical_points(s, LAM, dc)
        if len(pts) == 1:
            deep[k] = pts[0][0]
        elif len(pts) == 3:
            shallow[k], saddle[k], deep[k] = (p[0] for p in pts)
            m_shallow[k] = m_ratio(shallow[k], s, LAM, dc)
        m_deep[k] = m_ratio(deep[k], s, LAM, dc)
    branches[case] = (deep, shallow, saddle, m_deep, m_shallow)

bad, n = check_robertson_fryer()
print(f"sigma_c = {sigma_c:.3f}, sigma_f = {sigma_f:.3f} (lambda={LAM}, dC={DC})")
print(f"Robertson-Fryer check: {bad}/{n} mismatches")
mid = sig > 0.3
print(f"max M on deep branch for sigma > 0.3 (asym) = {np.nanmax(branches['asym'][3][mid]):.3f}")

# ---------------------------------------------------------------- plotting
cjk = [f.name for f in font_manager.fontManager.ttflist if "Noto Sans CJK" in f.name]
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": (["Noto Sans CJK SC"] if "Noto Sans CJK SC" in cjk else cjk)
        + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "path",
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "dejavusans",
    }
)
# Okabe-Ito colorblind-safe palette
BLUE, VERM, GREEN, GREY, ORANGE = "#0072B2", "#D55E00", "#009E73", "#7f7f7f", "#E69F00"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
fig.patch.set_facecolor("white")

# (a) critical points versus sigma
deep, shallow, saddle, _, _ = branches["sym"]
ax1.plot(sig, deep, color=GREY, lw=1.3, label="对称：峰")
ax1.plot(sig, shallow, color=GREY, lw=1.3)
ax1.plot(sig, saddle, color=GREY, lw=1.0, ls=":", label="对称：鞍点")
deep, shallow, saddle, _, _ = branches["asym"]
ax1.plot(sig, deep, color=BLUE, lw=1.9, label="非对称：深盆地的峰")
ax1.plot(sig, shallow, color=VERM, lw=1.9, label="非对称：浅盆地的峰")
ax1.plot(sig, saddle, color=VERM, lw=1.1, ls="--", label="非对称：鞍点")
ax1.axvline(sigma_c, color=GREY, lw=0.7, ls=":")
ax1.axvline(sigma_f, color=VERM, lw=0.7, ls=":")
ax1.text(sigma_c, -1.22, r" $\sigma_c$", color=GREY, ha="left", va="bottom", fontsize=9)
ax1.text(sigma_f, -1.22, r" $\sigma_f$", color=VERM, ha="left", va="bottom", fontsize=9)
ax1.set_xlim(sig[0], 0.0)
ax1.set_ylim(-1.25, 1.95)
ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
ax1.set_xlabel(r"带宽 $\sigma$（向右为降噪方向）")
ax1.set_ylabel(r"临界点位置 $z$")
ax1.legend(loc="upper center", frameon=False, fontsize=7.4, ncol=2, columnspacing=0.8,
           handlelength=1.6)
ax1.text(-0.16, 1.02, "(a)", transform=ax1.transAxes, fontsize=11, fontweight="bold", va="bottom")
ax1.text(0.5, 1.02, "叉形分叉与折叠分叉", transform=ax1.transAxes, fontsize=10,
         ha="center", va="bottom")

# (b) M along the branches
_, _, _, m_deep_sym, _ = branches["sym"]
m_centre = np.array([m_ratio(0.0, s, LAM, 0.0) for s in sig])
pre = sig >= sigma_c
ax2.plot(sig[pre], m_centre[pre], color=GREY, lw=1.3, label=r"对称：追踪点 $z=0$")
ax2.plot(sig[~pre], m_deep_sym[~pre], color=GREY, lw=1.3, ls="-.", label="对称：分叉后的峰")
_, _, _, m_deep, m_shallow = branches["asym"]
ax2.plot(sig, m_deep, color=BLUE, lw=1.9, label="非对称：深盆地的峰")
ax2.plot(sig, m_shallow, color=VERM, lw=1.9, label="非对称：浅盆地的峰")
ax2.axhline(1.0, color="black", lw=0.7)
ax2.axvline(sigma_c, color=GREY, lw=0.7, ls=":")
ax2.axvline(sigma_f, color=VERM, lw=0.7, ls=":")
ax2.set_xlim(sig[0], 0.0)
ax2.set_ylim(0.0, 1.3)
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_xlabel(r"带宽 $\sigma$")
ax2.set_ylabel(r"$M=\mathrm{Cov}[V\mid z]/\sigma^2$")
ax2.legend(loc="upper left", frameon=False, fontsize=7.4, ncol=2, columnspacing=0.8,
           handlelength=1.6, bbox_to_anchor=(0.0, 1.0))
ax2.annotate("σ 远小于盆地宽度时\n任何峰的 M 都趋于 1", xy=(0.05, 0.9), xytext=(0.2, 0.04),
             fontsize=7.6, color=GREY, ha="center",
             arrowprops=dict(arrowstyle="->", color=GREY, lw=0.7))
ax2.text(-0.16, 1.02, "(b)", transform=ax2.transAxes, fontsize=11, fontweight="bold", va="bottom")
ax2.text(0.5, 1.02, r"沿分支的 $M$", transform=ax2.transAxes, fontsize=10,
         ha="center", va="bottom")

# (c) bimodal region in (lambda, sigma)
lams = np.linspace(0.05, 3.0, 300)
sc = np.sqrt(np.clip(A_HALF**2 - lams / CURV, 0, None))
sf = np.array([sigma_fold(l, DC) for l in lams])
ax3.plot(lams, sc, color=GREY, lw=1.4, label=r"$\Delta C=0$：$\sigma_c=\sqrt{a^2-\lambda/A}$")
ax3.plot(lams, sf, color=VERM, lw=1.9, label=rf"$\Delta C={DC:g}$：$\sigma_f(\lambda)$")
ax3.fill_between(lams, 0, sf, color=VERM, alpha=0.10, lw=0)
ax3.axvline(LAM, color=BLUE, lw=0.7, ls=":")
ax3.text(2.1, 0.33, "双峰", color=VERM, fontsize=9.5, ha="center")
ax3.text(2.1, 0.8, "单峰", color=GREY, fontsize=9.5, ha="center")
ax3.set_xlim(0, lams[-1])
ax3.set_ylim(0, 1.45)
ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
ax3.set_xlabel(r"温度 $\lambda$")
ax3.set_ylabel(r"带宽 $\sigma$")
ax3.legend(loc="upper right", frameon=False, fontsize=7.4)
ax3.text(-0.16, 1.02, "(c)", transform=ax3.transAxes, fontsize=11, fontweight="bold", va="bottom")
ax3.text(0.5, 1.02, r"$(\lambda,\sigma)$ 平面上的双峰区", transform=ax3.transAxes, fontsize=10,
         ha="center", va="bottom")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, format="svg", facecolor="white", bbox_inches="tight")
print(f"wrote {OUT}")
