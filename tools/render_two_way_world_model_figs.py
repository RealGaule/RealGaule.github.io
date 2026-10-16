# Generates the four SVG figures for the two-way world model project page.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "content", "two-way-world-model")
BLUE, ORANGE, GRAY, INK, MUTED, FROZEN = "#3b5bdb", "#c2410c", "#6b7280", "#222222", "#555555", "#eceff3"

STYLE = f"""<style>
text{{font-family:Roboto,Helvetica,Arial,sans-serif;font-size:12.5px;fill:{INK}}}
.h{{font-size:13.5px;font-weight:600}} .H{{font-size:15px;font-weight:600}}
.s{{font-size:11.5px;fill:{MUTED}}} .m{{text-anchor:middle}} .e{{text-anchor:end}}
.cb{{fill:{BLUE}}} .co{{fill:{ORANGE}}} .cg{{fill:{GRAY}}} .w{{font-weight:600}}
.box{{fill:#fff;stroke-width:1.6}} .b{{stroke:{BLUE}}} .o{{stroke:{ORANGE}}} .g{{stroke:{GRAY}}}
.frz{{fill:{FROZEN};stroke:{GRAY};stroke-width:1.4}}
.lb{{stroke:{BLUE};stroke-width:1.8;fill:none}} .lo{{stroke:{ORANGE};stroke-width:1.8;fill:none}}
.lg{{stroke:{GRAY};stroke-width:1.4;fill:none}} .dash{{stroke-dasharray:5 4}}
</style>"""

def defs():
    m = ""
    for i, c in (("ab", BLUE), ("ao", ORANGE), ("ag", GRAY)):
        m += (f'<marker id="{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
              f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
    return f"<defs>{m}</defs>"

def T(parts):
    """parts: list of str or (sub,) tuples -> text content with dy subscripts."""
    out, pending = "", False
    for p in parts:
        if isinstance(p, tuple):
            out += f'<tspan dy="3" font-size="9">{p[0]}</tspan>'
            pending = True
        elif pending:
            out += f'<tspan dy="-3">{p}</tspan>'
            pending = False
        else:
            out += p
    return out

def text(x, y, s, cls="", extra=""):
    return f'<text x="{x}" y="{y}" class="{cls}" {extra}>{s}</text>\n'

def rect(x, y, w, h, cls, r=4):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" class="{cls}"/>\n'

def arrow(d, color="b", dash=False):
    return f'<path d="{d}" class="l{color}{" dash" if dash else ""}" marker-end="url(#a{color})"/>\n'

def svg(w, h, title, desc, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-labelledby="t d">\n<title id="t">{title}</title>\n<desc id="d">{desc}</desc>\n'
            f'{STYLE}\n{defs()}\n<rect width="{w}" height="{h}" fill="#fff"/>\n{body}</svg>\n')

# ---------------------------------------------------------------- figure 1: motivation
def fig1():
    b = ""
    b += text(80, 28, "(a) Tracked-point error against horizon", "H")
    X = lambda s: 80 + s * 36            # 0..10 s -> 80..440
    Y = lambda mm: 270 - mm * (220 / 70)  # 0..70 mm -> 270..50
    for mm in (20, 40, 60):
        b += f'<line x1="80" y1="{Y(mm):.1f}" x2="440" y2="{Y(mm):.1f}" stroke="#e3e3e3"/>\n'
    b += f'<path d="M80,50 V270 H440" class="lg" style="stroke:#444"/>\n'
    for s in (2, 4, 6, 8):
        b += f'<line x1="{X(s)}" y1="270" x2="{X(s)}" y2="275" stroke="#444"/>\n' + text(X(s), 290, str(s), "s m")
    for mm in (0, 20, 40, 60):
        b += text(72, Y(mm) + 4, str(mm), "s e")
    b += text(260, 312, "horizon (s)", "m")
    b += f'<text x="28" y="160" class="m" transform="rotate(-90 28 160)">tracked-point error (mm)</text>\n'
    # series: 3D-native 23 -> 56 mm ; video + depth 31 -> 46 mm (2 s and 8 s)
    pts = {"b": [(2, 23), (8, 56)], "o": [(2, 31), (8, 46)]}
    for c, ps in pts.items():
        (s1, m1), (s2, m2) = ps
        b += f'<line x1="{X(s1)}" y1="{Y(m1):.1f}" x2="{X(s2)}" y2="{Y(m2):.1f}" class="l{c}" style="stroke-width:2.6"/>\n'
        for s, mm in ps:
            b += f'<circle cx="{X(s)}" cy="{Y(mm):.1f}" r="5" class="c{c}"/>\n'
    b += text(X(2) + 9, Y(23) + 16, "23 mm", "s cb w") + text(X(2) + 9, Y(31) - 8, "31 mm", "s co w")
    b += text(X(8) + 9, Y(56) + 4, "56 mm", "s cb w") + text(X(8) + 9, Y(46) + 4, "46 mm", "s co w")
    # legend
    b += f'<line x1="96" y1="66" x2="124" y2="66" class="lb" style="stroke-width:2.6"/>' + text(132, 70, "3D-native point model (PointWorld)", "s")
    b += f'<line x1="96" y1="86" x2="124" y2="86" class="lo" style="stroke-width:2.6"/>' + text(132, 90, "video model with depth (DepthWorld)", "s")
    b += text(240, 256, "3D: precise early, drifts late", "s cb")
    b += text(240, 240, "video: coarser early, stable late", "s co")
    # panel b
    b += text(500, 28, "(b) Where each stream is strong and weak", "H")
    cols = [(500, "Video prior", "co",
             ["appearance and texture", "occluded and new content", "contact outcomes seen in video", "internet-scale motion prior"],
             ["metric scale, along-ray depth", "action faithfulness", "optimistic on failures", "no explicit state to plan on"]),
            (700, "Metric 3D state", "cb",
             ["metric, robot fixed by FK", "cheap rollouts (0.1 s)", "explicit, plannable, editable", "short-horizon precision"],
             ["no appearance", "contact it has not seen", "occluded points unsupervised", "drift by 8 s"])]
    for x, name, c, strong, weak in cols:
        b += text(x, 62, name, f"h {c}")
        b += text(x, 88, "strong at", "s w")
        for i, t in enumerate(strong):
            b += text(x + 10, 108 + 18 * i, "· " + t)
        b += text(x, 196, "weak at", "s w")
        for i, t in enumerate(weak):
            b += text(x + 10, 216 + 18 * i, "· " + t)
    b += f'<line x1="500" y1="74" x2="880" y2="74" stroke="#ddd"/>\n'
    return svg(900, 330, "Video and 3D prediction err at different horizons and on different things",
               "Left: tracked-point error against horizon for a 3D-native point model and a video model with depth; "
               "the point model is lower at 2 s and higher at 8 s. Right: what each stream is strong and weak at.", b)

# ---------------------------------------------------------------- figure 2: coupling patterns
def fig2():
    b = ""
    panels = [
        ("(a) auxiliary 3D branch", "Video DiT", "3D branch", "frz", "box g dash",
         [("fwd", "features (train only)", True)], ["GEM-4D, HF-WM, DreamWAM,", "Geometry Forcing, GaussianWAM"],
         "3D shapes training, then vanishes", None),
        ("(b) 3D as a readout", "Video DiT", "3D head", "frz", "box g",
         [("fwd", "read out", False)], ["DepthWorld, TesserAct"],
         "3D is an output, never fed back", "down"),
        ("(c) 3D as conditioning", "3D input", "Video DiT", "box b", "frz",
         [("fwd", "render, flow", False)], ["Hydra-0, RoboRender, SimForcing;", "World from Motion (per-video", "camera loop, no actions)"],
         "3D is an input, never corrected", None),
        ("(d) this proposal", "3D state", "Video DiT", "box b", "frz",
         [("fwd", "render, condition", False), ("back", "read out, fuse", False)], ["predictor-corrector loop"],
         "two-way, every chunk, at inference", "io"),
    ]
    for k, (ttl, L, R, lcls, rcls, arrows, ex, verdict, extra) in enumerate(panels):
        x0 = 15 + k * 237
        b += text(x0 + 110, 30, ttl, "H m")
        lx, rx, y, w, h = x0 + 6, x0 + 134, 70, 80, 44
        b += rect(lx, y, w, h, lcls) + rect(rx, y, w, h, rcls)
        b += text(lx + w / 2, y + 27, L, "m h" if len(L) < 14 else "m s") + text(rx + w / 2, y + 27, R, "m h")
        for kind, lab, dashed in arrows:
            if kind == "fwd":
                yy = y + 14 if len(arrows) > 1 else y + 22
                col = "b" if k == 3 else "g"
                b += arrow(f"M{lx + w},{yy} L{rx},{yy}", col, dashed)
                b += text((lx + w + rx) / 2, y - 8, lab, f"s m {'cb' if k == 3 else ''}")
            else:
                yy = y + 32
                b += arrow(f"M{rx},{yy} L{lx + w},{yy}", "o")
                b += text((lx + w + rx) / 2, y + h + 16, lab, "s m co w")
        if k == 0:
            b += text(rx + w / 2, y + h + 18, "removed at inference", "s m co")
        if extra == "down":
            b += arrow(f"M{rx + w / 2},{y + h} L{rx + w / 2},{y + h + 26}", "g")
            b += text(rx + w / 2, y + h + 38, "3D output", "s m")
        if extra == "io":
            b += arrow(f"M{lx + w / 2},{y - 26} L{lx + w / 2},{y - 2}", "b")
            b += text(lx + w / 2 - 6, y - 12, "action", "s e")
            b += arrow(f"M{rx + w / 2},{y + h} L{rx + w / 2},{y + h + 24}", "o")
            b += text(rx + w / 2 + 6, y + h + 20, "video", "s")
        for i, t in enumerate(ex):
            b += text(x0 + 110, 176 + 16 * i, t, "s m")
        b += text(x0 + 110, 246, verdict, "m " + ("co w" if k == 3 else ""))
        if k < 3:
            b += f'<line x1="{x0 + 232}" y1="20" x2="{x0 + 232}" y2="262" stroke="#e3e3e3"/>\n'
    return svg(960, 275, "How published models couple video and 3D, and the proposed two-way loop",
               "Four panels. (a) an auxiliary 3D branch receives video features during training and is removed at inference. "
               "(b) a 3D head reads geometry out of the video model as an output. (c) 3D renders or flow condition the video model. "
               "(d) the proposal: the 3D state renders and conditions the video model, and the video model is read out and fused back "
               "into the state, every chunk, at inference.", b)

# ---------------------------------------------------------------- figure 3: architecture
def fig3():
    b = ""
    # action
    b += text(280, 22, T(["action a", ("t",), ": robot point flow ΔX", ("rob",), " from forward kinematics"]), "m")
    b += arrow("M280,30 L280,68", "b")
    # state
    b += rect(15, 120, 150, 120, "box b")
    b += text(90, 142, T(["State S", ("t",)]), "m h") + text(90, 161, "4096 scene, 512 robot pts", "m s")
    b += text(90, 177, "X, f (DINOv3), skin, Σ", "m s") + text(90, 193, "robot points move by FK", "m s")
    b += text(90, 209, "Gaussian skin for rendering", "m s") + text(90, 225, "the state is the memory", "m s")
    # A predictor
    b += rect(200, 70, 160, 100, "box b")
    b += text(280, 90, T(["A. 3D predictor f", ("θ",)]), "m h") + text(280, 107, "PTv3, 12 blocks, d 512", "m s")
    b += text(280, 122, "SE(3) clusters + residual", "m s") + text(280, 137, "aleatoric variance, contact", "m s")
    b += text(280, 157, T(["→ μ", ("f",), ", Σ", ("f",), " for 8 steps"]), "m")
    # render
    b += rect(400, 70, 120, 100, "box g")
    b += text(460, 90, "Render", "m h") + text(460, 107, T(["gsplat skin after μ", ("f",)]), "m s")
    b += text(460, 122, "+ URDF robot render", "m s")
    b += text(460, 142, "C: RGB, log-depth,", "m") + text(460, 158, "robot, mask (8 ch)", "m")
    # DiT
    b += rect(560, 30, 290, 220, "frz")
    b += text(705, 50, "B. Wan2.1 DiT 1.3B, frozen", "m h") + text(705, 66, "30 blocks, d 1536, LoRA r64 on attn + FFN", "m s")
    b += f'<rect x="585" y="80" width="26" height="150" fill="#fff" stroke="{GRAY}"/>\n'
    for i in range(1, 15):
        b += f'<line x1="585" y1="{80 + i * 10}" x2="611" y2="{80 + i * 10}" stroke="#cfd3d8"/>\n'
    b += text(598, 245, "blocks 0–29", "s m")
    b += f'<line x1="632" y1="80" x2="632" y2="200" class="lo"/>\n'
    for blk in (0, 6, 12, 18, 24):
        yy = 80 + blk * 5
        b += arrow(f"M632,{yy} L614,{yy}", "o")
        b += text(638, yy + 4, str(blk), "s")
    b += rect(662, 150, 180, 44, "box o")
    b += text(752, 167, "condition trunk (trainable)", "m") + text(752, 184, "4 DiT blocks, zero-init adds", "m s")
    b += f'<line x1="662" y1="186" x2="633" y2="186" class="lo"/>\n'
    b += text(662, 96, "input: noisy latent ⊕ C latents", "") + text(662, 111, "channel concat, zero-init projection", "s")
    b += text(662, 127, "anchor = last observed frame", "s")
    b += text(662, 218, "4-step distilled sampler", "") + text(662, 234, "trainable: LoRA + trunk only", "s")
    # video out
    b += rect(870, 90, 120, 70, "box g")
    b += text(930, 112, "video chunk", "m h") + text(930, 130, "8 frames at 5 Hz", "m s") + text(930, 146, "480 × 640", "m s")
    # features
    b += rect(560, 340, 290, 56, "frz")
    b += text(705, 360, "clean pass at t = 0 → block-15 features", "m") + text(705, 378, "m: 3 × 30 × 40 × 1536 (3600 tokens)", "m s")
    # readout
    b += rect(330, 340, 195, 125, "box o")
    b += text(427, 360, T(["C. Readout h", ("ψ",)]), "m h") + text(427, 377, T(["queries: f, current (u, v); no μ", ("f",)]), "m s")
    b += text(427, 392, "4 cross-attn blocks, 2 iterations", "m s") + text(427, 407, "deformable sampling on m", "m s")
    b += text(427, 422, "DPT depth head → spawn points", "m s")
    b += text(427, 444, T(["→ μ", ("h",), ", Σ", ("h",), ", visibility v"]), "m")
    # fusion
    b += rect(165, 340, 145, 140, "box b")
    b += text(237, 360, "D. Fusion", "m h") + text(237, 377, T(["gate α from Σ", ("f",), ", Σ", ("h",), ", v"]), "m s")
    b += text(237, 392, "precision-weighted update", "m s") + text(237, 407, "robot points := FK", "m s")
    b += text(237, 422, "ARAP projection", "m s") + text(237, 437, "skin: 5 Adam steps", "m s")
    b += text(237, 459, T(["→ S", ("t+1",), ", Σ + Q"]), "m")
    # arrows top row
    b += arrow("M165,150 L198,122", "b") + text(130, 112, "point tokens", "s m")
    b += arrow("M360,120 L398,120", "b") + text(380, 112, T(["μ", ("f",)]), "s m")
    b += arrow("M520,120 L558,120", "b") + text(540, 112, "C", "s m")
    b += arrow("M850,125 L868,125", "o") + text(859, 117, "x₀", "s m")
    # arrows bottom row
    b += arrow("M705,250 L705,338", "o") + text(712, 298, T(["x", ("0",), " (generated)"]), "s")
    b += arrow("M560,368 L527,368", "o") + text(543, 360, "k, v", "s m")
    b += arrow("M330,350 L312,350", "o") + text(320, 331, T(["μ", ("h",), ", Σ", ("h",), ", v"]), "s m")
    b += arrow("M240,170 L240,338", "b") + text(247, 262, T(["μ", ("f",), ", Σ", ("f",)]), "s")
    b += arrow("M165,362 H100 V242", "b") + text(92, 300, T(["S", ("t+1",)]), "s e")
    b += arrow("M60,240 V540 H427 V467", "g") + text(250, 553, "queries: point feature f and current projection (u, v), not the prediction", "s m")
    b += arrow("M237,480 L237,508", "b") + text(237, 523, T(["disagreement d", ("t",), " = ‖μ", ("f",), " − μ", ("h",), "‖"]), "s m")
    # legend
    y = 585
    b += rect(20, y - 10, 16, 12, "frz", 2) + text(42, y, "frozen", "s")
    b += rect(100, y - 10, 16, 12, "box b", 2) + text(122, y, "trainable, 3D side", "s")
    b += rect(230, y - 10, 16, 12, "box o", 2) + text(252, y, "trainable, video side", "s")
    b += arrow(f"M380,{y - 4} L412,{y - 4}", "b") + text(418, y, "3D → video", "s")
    b += arrow(f"M500,{y - 4} L532,{y - 4}", "o") + text(538, y, "video → 3D", "s")
    b += text(650, y, T(["one chunk = 9 frames at 5 Hz; f", ("θ",), " ≈ 0.1 s, g", ("φ",), " ≈ 1 s on one H100"]), "s")
    return svg(1000, 600, "Architecture of the predictor-corrector world model, one chunk",
               "Top row, 3D to video: the point state and the action enter the 3D predictor, its prediction is rendered into an "
               "eight-channel condition video and injected into a frozen Wan2.1 DiT through channel concatenation and a condition "
               "trunk at five blocks; a four-step sampler emits the video chunk. Bottom row, video to 3D: a clean pass exposes "
               "block-15 features, a readout head queried by the state returns per-point displacement, covariance and visibility, "
               "and the fusion module combines it with the prediction, fixes robot points by kinematics, projects with ARAP and "
               "writes the next state.", b)

# ---------------------------------------------------------------- figure 4: training
def fig4():
    b = ""
    mods = [T(["A  predictor f", ("θ",)]), "B  LoRA + trunk", T(["C  readout h", ("ψ",)]), "D  gate"]
    stages = [
        ("Stage 1: predictor", ["train", "unused", "unused", "unused"],
         ["DROID-3D + RoboTwin 2.0", "2D tracks, FK, no 3D labels"],
         ["reprojection NLL", "+ ARAP + FK anchor"]),
        ("Stage 2: renderer", ["frozen", "train", "unused", "unused"],
         ["same clips, conditions", "rendered from stage-1 states"],
         ["flow matching", "per-block cond. dropout"]),
        ("Stage 3: readout", ["unused", "frozen", "train", "unused"],
         ["pseudo 3D tracks (ST-V2,", "TrackCraft3R), metric via DROID-3D"],
         ["NLL + BCE(v)", "+ 2D reprojection"]),
        ("Stage 4: closed loop", ["train", "train", "train", "train"],
         ["N-chunk unrolls", "+ never-executed actions a′"],
         [T(["L", ("track",), " + λ₁L", ("corr",), " + λ₂ agree"]), T(["+ λ₃L", ("cf",), " (backbone frozen)"])]),
    ]
    colw, gap = 215, 22
    for k, (ttl, st, data, loss) in enumerate(stages):
        x0 = 20 + k * (colw + gap)
        b += text(x0 + colw / 2, 30, ttl, "H m")
        for i, (m, s) in enumerate(zip(mods, st)):
            y = 48 + i * 34
            cls = {"train": "box " + ("b" if i in (0, 3) else "o"), "frozen": "frz", "unused": "box g dash"}[s]
            b += rect(x0, y, colw, 26, cls)
            b += text(x0 + 10, y + 17, m, "")
            tag = {"train": ("train", "co" if i in (1, 2) else "cb"), "frozen": ("frozen", "cg"), "unused": ("unused", "cg")}[s]
            b += text(x0 + colw - 8, y + 17, tag[0], f"s e w {tag[1]}")
        b += text(x0, 204, "data", "s w")
        for i, t in enumerate(data):
            b += text(x0 + 8, 220 + 15 * i, t, "s")
        b += text(x0, 262, "loss", "s w")
        for i, t in enumerate(loss):
            b += text(x0 + 8, 278 + 15 * i, t, "s")
    # stage-4 detail band
    b += f'<line x1="20" y1="312" x2="940" y2="312" stroke="#ddd"/>\n'
    b += text(20, 332, "Stage 4 in detail: unroll the loop, supervise both directions, teach counterfactuals", "h")
    y = 372
    xs = [40, 215, 390, 565]
    for i, x in enumerate(xs):
        b += rect(x, y - 16, 62, 32, "box b") + text(x + 31, y + 5, T(["S", (str(i),)]), "m h")
        if i < 3:
            b += arrow(f"M{x + 62},{y} L{xs[i + 1] - 2},{y}", "b")
            b += text((x + 62 + xs[i + 1]) / 2, y - 8, "f → g → h → F", "s m")
    b += text(660, y + 5, "…", "h")
    # losses attached
    b += text(40, 408, T(["L", ("track",), ": fused state reprojected onto real 2D tracks (no 3D labels)"]), "s")
    b += text(40, 424, T(["L", ("corr",), ": fused state reprojected onto the generated frames (action faithfulness)"]), "s")
    b += text(40, 440, T(["agree: ‖μ", ("f",), " − sg(μ", ("h",), ")‖² on visible points; gradients on the video side reach LoRA, trunk, readout, gate only"]), "s")
    # teacher
    b += rect(720, 340, 220, 60, "box o")
    b += text(830, 360, T(["teacher: g", ("φ",), " in robot-only mode"]), "m")
    b += text(830, 376, "skin dropped, URDF render of a′", "m s") + text(830, 391, "imagines the scene response to a′", "m s")
    b += arrow("M720,372 L700,372", "o")
    b += text(698, 376, T(["L", ("cf",)]), "s e co w")
    return svg(960, 455, "Four training stages and the closed-loop stage in detail",
               "Top: four columns, one per stage, each listing which of the four modules is trained, frozen or unused, the data and the loss. "
               "Bottom: the closed-loop stage unrolls the chunk loop over several states, supervises the fused state with real tracks and "
               "with the generated frames, adds an agreement term, and distils counterfactual responses from the video model in robot-only mode.", b)

for name, fn in (("fig1-motivation.svg", fig1), ("fig2-couplings.svg", fig2), ("fig3-architecture.svg", fig3), ("fig4-training.svg", fig4)):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(fn())
    print("wrote", name)
