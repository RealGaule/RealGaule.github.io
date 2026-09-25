"""Gaussian convolution of a Gibbs distribution = Gibbs distribution of the
KL-regularized free energy (1-D numerical illustration).

    C(v)   = -log( exp(-(v+2)^2 / (2 s1^2)) + A2 exp(-(v-2)^2 / (2 s2^2)) )
    pi(v)  = exp(-C(v)/lam) / Z_C
    p      = N(0, sigma^2)
    F(u)   = -lam log  int p(eps) exp(-C(u+eps)/lam) d eps
    (pi * p)(u) = exp(-F(u)/lam) / Z_C
    r*_u(v) ∝ pi(v) N(v; u, sigma^2)

Narrow deep well near v=-2 (s1 small), wide shallow well near v=+2.
Run:  python3 tools/render_gibbs_convolution.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notes-src/docs/assets/images/gibbs-convolution.svg"

# ---------------------------------------------------------------- parameters
LAM = 1.0              # temperature lambda
SIGMA = 0.8            # std of reference p = N(0, sigma^2)
M1, S1 = -2.0, 0.15    # narrow, deep well
M2, S2 = 2.0, 1.0      # wide, shallow well
A2 = np.exp(-1.0)      # wide-well depth offset: its minimum is ~1 above the narrow one
U_STAR = 0.0           # point u at which the optimal r*_u is drawn


def cost(v):
    """C(v) = -log mixture of two unnormalized Gaussian bumps (smooth double well)."""
    a = -((v - M1) ** 2) / (2 * S1**2)
    b = np.log(A2) - ((v - M2) ** 2) / (2 * S2**2)
    return -np.logaddexp(a, b)


def gauss(x, m, s):
    return np.exp(-((x - m) ** 2) / (2 * s**2)) / (np.sqrt(2 * np.pi) * s)


# ---------------------------------------------------------------- numerics
dv = 0.002
v = np.arange(-10.0, 10.0 + dv / 2, dv)
C = cost(v)
Z_C = np.trapz(np.exp(-C / LAM), v)
pi = np.exp(-C / LAM) / Z_C

# (1) convolution pi * p on the grid (FFT-free direct discrete convolution)
k = np.arange(-int(8 * SIGMA / dv), int(8 * SIGMA / dv) + 1) * dv
conv = np.convolve(pi, gauss(k, 0.0, SIGMA), mode="same") * dv

# (2) free energy F(u) by independent quadrature over eps (log-sum-exp)
u = v[np.abs(v) <= 6.0]
eps = np.arange(-8 * SIGMA, 8 * SIGMA + dv / 2, dv)
logw = -(eps**2) / (2 * SIGMA**2) - np.log(np.sqrt(2 * np.pi) * SIGMA) + np.log(dv)
F = np.empty_like(u)
for i, ui in enumerate(u):
    x = logw - cost(ui + eps) / LAM
    mx = x.max()
    F[i] = -LAM * (mx + np.log(np.exp(x - mx).sum()))

gibbsF = np.exp(-F / LAM) / Z_C
conv_u = np.interp(u, v, conv)

# (3) closed form (pi is exactly a Gaussian mixture when LAM = 1)
w1, w2 = S1, A2 * S2
exact = (w1 * gauss(u, M1, np.hypot(S1, SIGMA)) + w2 * gauss(u, M2, np.hypot(S2, SIGMA))) / (w1 + w2)

err_id = np.max(np.abs(conv_u - gibbsF))
err_exact = np.max(np.abs(gibbsF - exact))

argminC = v[np.argmin(C)]
argminF = u[np.argmin(F)]
print(f"Z_C = {Z_C:.8f}  (closed form {np.sqrt(2*np.pi)*(S1 + A2*S2):.8f})")
print(f"argmin C = {argminC:+.4f}   min C = {C.min():.4f}")
print(f"argmin F = {argminF:+.4f}   min F = {F.min():.4f}")
print(f"max |(pi*p) - exp(-F/lam)/Z_C| = {err_id:.3e}")
print(f"max |exp(-F/lam)/Z_C - closed form| = {err_exact:.3e}")

# optimal r*_u(v) ∝ pi(v) N(v; u, sigma^2)
r = pi * gauss(v, U_STAR, SIGMA)
r /= np.trapz(r, v)

# ---------------------------------------------------------------- figure
cjk = [f.name for f in font_manager.fontManager.ttflist if "Noto Sans CJK" in f.name]
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": (["Noto Sans CJK SC"] if "Noto Sans CJK SC" in cjk else cjk) + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "path",
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "dejavusans",
    }
)

# Okabe-Ito colorblind-safe palette
BLUE, ORANGE, GREEN, VERM, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#7f7f7f"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
fig.patch.set_facecolor("white")
XL = (-5, 5)

# (a) cost and free energy
mv = np.abs(v) <= 5
ax1.plot(v[mv], C[mv] - C.min(), color=BLUE, lw=1.8, label=r"$C(v)-\min\,C$")
ax1.plot(u, F - F.min(), color=VERM, lw=1.8, label=r"$F(u)-\min\,F$")
ax1.plot(argminC, 0, "o", color=BLUE, ms=6, zorder=5, clip_on=False, ls="none",
         label=rf"$\arg\min\,C={{{argminC:+.2f}}}$")
ax1.plot(argminF, 0, "s", color=VERM, ms=6, zorder=5, clip_on=False, ls="none",
         label=rf"$\arg\min\,F={{{argminF:+.2f}}}$")
ax1.text(M1 - 0.5, 4.45, "窄而深", ha="right", va="center", fontsize=9, color=GREY)
ax1.text(M2, 0.5, "宽而浅", ha="center", va="center", fontsize=9, color=GREY)
ax1.set_xlim(*XL)
ax1.set_ylim(0, 5.0)
ax1.set_xlabel(r"$v$ 或 $u$")
ax1.set_ylabel("相对值（减去各自最小值）")
ax1.legend(loc="upper center", bbox_to_anchor=(0.7, 1.0), frameon=False, fontsize=8.5,
           handlelength=1.6, borderaxespad=0.2)
ax1.text(-0.13, 1.02, "(a)", transform=ax1.transAxes, fontsize=11, fontweight="bold", va="bottom")
ax1.text(0.5, 1.02, "代价与自由能", transform=ax1.transAxes, fontsize=10, ha="center", va="bottom")

# (b) distributions
ax2.fill_between(v[mv], pi[mv], color=BLUE, alpha=0.12, lw=0)
ax2.plot(v[mv], pi[mv], color=BLUE, lw=1.5, label=r"$\pi(v)\propto e^{-C(v)/\lambda}$")
ax2.plot(u, gibbsF, color=VERM, lw=1.8,
         label=r"$(\pi*p)(u)=e^{-F(u)/\lambda}/Z_C$")
ax2.plot(v[mv], r[mv], color=GREEN, lw=1.6, ls="--",
         label=rf"$r^*_u(v)\propto\pi(v)\,\mathcal{{N}}(v;u,\sigma^2)$, $u={U_STAR:g}$")
ax2.axvline(U_STAR, color=GREY, lw=0.8, ls=":")
ax2.text(U_STAR + 0.1, 0.02, r"$u$", color=GREY, fontsize=9, transform=ax2.get_xaxis_transform())
ax2.set_xlim(*XL)
ax2.set_ylim(0, 1.05 * max(pi.max(), r.max()))
ax2.set_xlabel(r"$v$ 或 $u$")
ax2.set_ylabel("概率密度")
ax2.legend(loc="upper right", frameon=False, fontsize=8.5)
ax2.text(-0.13, 1.02, "(b)", transform=ax2.transAxes, fontsize=11, fontweight="bold", va="bottom")
ax2.text(0.5, 1.02, "分布", transform=ax2.transAxes, fontsize=10, ha="center", va="bottom")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, format="svg", facecolor="white", bbox_inches="tight")
print(f"wrote {OUT}")
