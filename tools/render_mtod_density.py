"""Original illustrative single-schedule Sample -> Refine particle animation.

This toy is not the production MToD algorithm or a benchmark result.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SEED = 260914
PARTICLES = 240
STEPS = 720
FRAMES = 161
SWITCH = 0.625
MEANS = np.array([-2.4, 0.0, 2.35])
WEIGHTS = np.array([0.30, 0.38, 0.32])
BASE_STD = 0.30
SIGMA_MAX = 1.65
SIGMA_MIN = 0.035
REFINE_RATE = 12.0
KDE_BANDWIDTH = 0.14
COLORS = np.array(["#2563eb", "#0d9488", "#c77912"])


def sigma_squared(progress):
    return SIGMA_MIN**2 + (SIGMA_MAX**2 - SIGMA_MIN**2) * (1.0 - progress) ** 3


def mixture_quantities(x, noise_variance):
    variance = BASE_STD**2 + noise_variance
    difference = np.asarray(x)[..., None] - MEANS
    log_components = np.log(WEIGHTS) - difference**2 / (2.0 * variance)
    shifted = np.exp(log_components - np.max(log_components, axis=-1, keepdims=True))
    responsibilities = shifted / shifted.sum(axis=-1, keepdims=True)
    mean_shift_target = responsibilities @ MEANS
    score = (mean_shift_target - x) / variance
    density = np.exp(log_components).sum(axis=-1) / np.sqrt(2.0 * np.pi * variance)
    return score, mean_shift_target, density


def simulate():
    rng = np.random.default_rng(SEED)
    progress = np.linspace(0.0, 1.0, STEPS + 1)
    variances = sigma_squared(progress)
    # A single initialization drawn exactly from the broad smoothed toy target.
    # No component assignment is used after this initial distribution draw.
    components = rng.choice(len(MEANS), size=PARTICLES, p=WEIGHTS)
    initial = MEANS[components] + np.sqrt(BASE_STD**2 + variances[0]) * rng.normal(size=PARTICLES)
    history = np.empty((STEPS + 1, PARTICLES))
    history[0] = initial
    innovations = np.zeros((STEPS, PARTICLES))
    switch_step = int(round(SWITCH * STEPS))
    for step in range(STEPS):
        x = history[step]
        score, mean_shift_target, _ = mixture_quantities(x, variances[step])
        if step < switch_step:
            # Euler-Maruyama for the reverse variance-exploding diffusion,
            # parameterized by decreasing forward diffusion variance.
            variance_drop = variances[step] - variances[step + 1]
            innovations[step] = np.sqrt(variance_drop) * rng.normal(size=PARTICLES)
            history[step + 1] = x + variance_drop * score + innovations[step]
        else:
            # Damped mean-shift ascent on the SAME continuously annealed p_sigma.
            # This is mode refinement, not an exact probability-flow sampler.
            amount = -np.expm1(-REFINE_RATE / STEPS)
            history[step + 1] = x + amount * (mean_shift_target - x)
    assert np.isfinite(history).all()
    assert np.all(np.diff(variances) < 0.0)
    assert np.all(innovations[switch_step:] == 0.0)
    return progress, variances, history, innovations, switch_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "mtod/static/gifs")
    parser.add_argument("--preview-dir", type=Path, default=None)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = args.preview_dir.resolve() if args.preview_dir else Path(tempfile.mkdtemp(prefix="mtod-density-preview."))
    work.mkdir(parents=True, exist_ok=True)
    progress, variances, history, innovations, switch_step = simulate()
    # Colors identify the resulting endpoint neighborhoods, retrospectively.
    # They never influence the update, score, noise or particle assignment.
    endpoint_groups = np.argmin(abs(history[-1, :, None] - MEANS), axis=-1)
    mode_counts = np.bincount(endpoint_groups, minlength=len(MEANS))
    particle_colors = COLORS[endpoint_groups]
    y_extent = float(max(6.0, np.ceil(np.max(np.abs(history)) * 2.0) / 2.0 + 0.3))
    coordinates = np.linspace(-y_extent, y_extent, 900)
    frame_steps = np.rint(np.linspace(0, STEPS, FRAMES)).astype(int)
    assert switch_step in frame_steps
    all_kdes = []
    all_targets = []
    for step in frame_steps:
        z = (coordinates[:, None] - history[step][None, :]) / KDE_BANDWIDTH
        all_kdes.append(np.exp(-0.5 * z * z).mean(axis=1) / (np.sqrt(2.0 * np.pi) * KDE_BANDWIDTH))
        all_targets.append(mixture_quantities(coordinates, variances[step])[2])
    density_extent = max(float(np.max(all_kdes)), float(np.max(all_targets))) * 1.10

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": "#475569", "text.color": "#172033", "xtick.color": "#64748b", "ytick.color": "#64748b"})
    fig = plt.figure(figsize=(11.2, 5.2), dpi=100, facecolor="white")
    ax = fig.add_axes([0.076, 0.205, 0.622, 0.566])
    density_ax = fig.add_axes([0.776, 0.205, 0.187, 0.566], sharey=ax)
    fig.text(0.075, 0.935, "MToD · Sample, then Refine", fontsize=23, fontweight="bold")
    fig.text(0.076, 0.883, "Schematic · 1-D example", fontsize=11, color="#64748b")
    fig.text(0.076 + 0.622 * SWITCH / 2, 0.802, "SAMPLE · stochastic", ha="center", fontsize=12, fontweight="bold", color="#2563eb")
    fig.text(0.076 + 0.622 * (SWITCH + (1-SWITCH)/2), 0.802, "REFINE · deterministic", ha="center", fontsize=12, fontweight="bold", color="#0d9488")
    fig.text(0.776, 0.802, "Density", fontsize=12, fontweight="bold")
    ax.axvspan(0.0, SWITCH, facecolor="#eff6ff", alpha=0.7, zorder=0)
    ax.axvspan(SWITCH, 1.0, facecolor="#f0fdfa", alpha=0.85, zorder=0)
    ax.axvline(SWITCH, color="#94a3b8", linewidth=1.3, linestyle=(0, (4, 4)), zorder=1)
    ax.set(xlim=(-0.008, 1.008), ylim=(-y_extent, y_extent), xlabel="Denoising progress", ylabel="Scalar control coordinate")
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks(np.arange(-int(y_extent)//2*2, int(y_extent)+1, 2))
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.7, alpha=0.65)
    ax.xaxis.labelpad = 12
    density_ax.set(xlim=(0, density_extent), xlabel="Probability density")
    density_ax.set_xticks([0.0, 0.5, 1.0])
    density_ax.xaxis.labelpad = 12
    density_ax.tick_params(axis="y", left=False, labelleft=False)
    density_ax.grid(axis="x", color="#e2e8f0", linewidth=0.7)
    for axis in [ax, density_ax]:
        axis.spines[["top", "right"]].set_visible(False)
        for spine in axis.spines.values():
            spine.set_color("#cbd5e1")

    trail_rgba = [to_rgba(color, 0.17) for color in particle_colors]
    trails = LineCollection([], colors=trail_rgba, linewidths=0.65, zorder=2)
    ax.add_collection(trails)
    # A small subset receives stronger trails for readable individual motion.
    highlight_ids = np.linspace(0, PARTICLES - 1, 18).astype(int)
    highlights = LineCollection([], colors=[to_rgba(particle_colors[i], 0.65) for i in highlight_ids], linewidths=1.0, zorder=3)
    ax.add_collection(highlights)
    particles = ax.scatter(np.zeros(PARTICLES), history[0], s=13, c=particle_colors, alpha=0.80, linewidths=0.22, edgecolors="white", zorder=5)
    cursor = ax.axvline(0, color="#334155", linewidth=1, alpha=0.28, zorder=4)
    (kde_line,) = density_ax.plot([], [], color="#7c3aed", linewidth=2.2, label="Particle KDE")
    (target_line,) = density_ax.plot([], [], color="#475569", linewidth=1.8, linestyle=(0, (4, 2.5)), label="Smoothed target")
    density_ax.legend(loc="lower left", bbox_to_anchor=(-0.05, 1.145), frameon=False, fontsize=10, borderaxespad=0, handlelength=2.3, labelspacing=0.65)
    fill = None
    noise_label = fig.text(0.077, 0.061, "", fontsize=11, color="#475569")
    fig.text(0.428, 0.061, "240 particles · no restart", fontsize=11, color="#475569")
    fig.text(0.775, 0.061, "KDE bandwidth: 0.14", fontsize=10, color="#64748b")

    images = []
    keyframes = {0: "first", switch_step: "switch", STEPS: "last"}
    for frame_index, step in enumerate(frame_steps):
        # Retain a decimated full history; points always show the current state.
        shown_steps = np.unique(np.r_[np.arange(0, step + 1, 3), step])
        segments = [np.column_stack((progress[shown_steps], history[shown_steps, particle])) for particle in range(PARTICLES)]
        trails.set_segments(segments)
        highlights.set_segments([segments[i] for i in highlight_ids])
        particles.set_offsets(np.column_stack((np.full(PARTICLES, progress[step]), history[step])))
        cursor.set_xdata([progress[step], progress[step]])
        kde_line.set_data(all_kdes[frame_index], coordinates)
        target_line.set_data(all_targets[frame_index], coordinates)
        if fill is not None:
            fill.remove()
        fill = density_ax.fill_betweenx(coordinates, 0, all_kdes[frame_index], color="#ede9fe", alpha=0.8, zorder=-1)
        noise_label.set_text(f"Noise scale σ = {np.sqrt(variances[step]):.3f}    ·    {progress[step]:.0%}")
        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        if step in keyframes:
            frame.save(work / f"{keyframes[step]}.png")
        images.append(frame)
        if frame_index % 40 == 0:
            print(f"Rendered {frame_index+1}/{FRAMES} frames", flush=True)
    durations = [60] * len(images)
    durations[0] = 550
    durations[-1] = 1100
    # A shared palette avoids frame-to-frame color flicker.
    palette_source = Image.new("RGB", (1120, 520 * 3))
    for position, frame_id in enumerate([0, int(SWITCH * (FRAMES - 1)), FRAMES - 1]):
        palette_source.paste(images[frame_id], (0, 520 * position))
    palette = palette_source.quantize(colors=192, method=Image.Quantize.MEDIANCUT)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in images]
    gif_path = out / "mtod-single-denoise.gif"
    quantized[0].save(gif_path, save_all=True, append_images=quantized[1:], duration=durations, loop=0, optimize=False, disposal=1)
    poster_path = out / "mtod-single-denoise-poster.png"
    images[-1].save(poster_path, optimize=True)
    plt.close(fig)

    manifest = {
        "description": "Original illustrative 1-D single-process Sample then Refine variant; not the production two-process MToD algorithm, not a benchmark, not an exact-sampling claim.",
        "files": {"gif": gif_path.name, "poster": poster_path.name, "generator": "tools/render_mtod_density.py"},
        "reference_layout_only": "https://temporalscorerescaling.github.io/static/images/animated_1d/uncond_v_TSR_k1.0.gif; no reference artwork, frames, code, or data reused.",
        "size": [1120, 520], "frames": len(images), "duration_ms": sum(durations), "loop": True,
        "seed": SEED, "particles": PARTICLES, "integration_steps": STEPS,
        "target": {"means": MEANS.tolist(), "weights": WEIGHTS.tolist(), "component_std": BASE_STD, "p_sigma": "sum_j w_j Normal(x; mu_j, base_std^2 + sigma^2)", "toy_cost": "J(x) = -log p_0(x)"},
        "schedule": {"sigma_max": SIGMA_MAX, "sigma_min": SIGMA_MIN, "sigma_squared": "sigma_min^2 + (sigma_max^2 - sigma_min^2) * (1-progress)^3", "switch_progress": SWITCH, "switch_sigma": float(np.sqrt(variances[switch_step]))},
        "equations": {"responsibility": "r_j(x,sigma) proportional to w_j exp(-(x-mu_j)^2/(2*(base_std^2+sigma^2)))", "score": "grad log p_sigma(x) = (sum_j r_j mu_j - x)/(base_std^2+sigma^2)", "sample": "x_next = x + (sigma_k^2-sigma_next^2) * score(x,sigma_k) + sqrt(sigma_k^2-sigma_next^2) * xi; xi ~ N(0,1)", "refine": "x_next = x + (1-exp(-12*delta_progress)) * (sum_j r_j(x,sigma_k)*mu_j - x)", "interpretation": "reverse-VE Euler-Maruyama sampling followed by damped annealed mean-shift on the same analytic smoothed density; no schedule reset, no particle reinitialization, no resampling; refinement changes the law and is not exact target sampling"},
        "initialization": "One draw from the analytic broad smoothed target p_sigma_max. No latent mixture assignments are used during particle updates.",
        "visualization": {"particle_density": "Gaussian KDE with fixed bandwidth 0.14", "smoothed_target": "Analytic p_sigma at the current continuously decreasing sigma", "colors": "Nearest-mean neighborhood of final endpoints, assigned retrospectively for visual identity only; never used by dynamics", "trails": "All 240 particles with 18 stronger paths for legibility; points are not synthetic or manually assigned to modes."},
        "checks": {"all_finite": True, "strictly_decreasing_noise_variance": True, "zero_injected_noise_after_switch": True, "particle_count_constant": PARTICLES, "final_neighborhood_counts": mode_counts.tolist(), "final_within_neighborhood_std": [float(history[-1, endpoint_groups == j].std()) for j in range(3)], "max_abs_coordinate": float(np.max(np.abs(history))), "sigma_before_at_after_switch": np.sqrt(variances[switch_step-1:switch_step+2]).tolist()},
        "parameter_status": "All values and switch timing are illustrative choices, not tuned or validated performance conclusions.",
    }
    (out / "mtod-single-denoise.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"gif": str(gif_path), "bytes": gif_path.stat().st_size, "duration_ms": sum(durations), "counts": mode_counts.tolist(), "range": y_extent, "manifest": str(out / "mtod-single-denoise.manifest.json"), "previews": str(work)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
