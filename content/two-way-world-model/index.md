---
title: "A Predictor-Corrector World Model: Two-Way Coupling of a Video Prior and a Metric 3D State"
date: 2026-10-16
description: "Research proposal (October 2026): a robot-manipulation world model in which a metric point state predicts, a frozen video generator renders and corrects, and the two streams improve each other at inference, not only during training."
summary: "Proposal: a metric 3D point state and a frozen video DiT correct each other every chunk through rendering, a readout head and precision-weighted fusion."
tags: ["世界模型", "研究规划", "视频生成"]
---

<style>.post-content figure > figcaption::before { content: "Figure " counter(fig) ". "; }</style>

This page is a proposal written in October 2026, not a report of results. Every number quoted below comes from the cited paper.

## 1. Motivation

Learned simulators for robot manipulation come in two families. **Video world models** fine-tune a large video generator so that, given the current view and the commanded robot motion, it produces the future video [1, 2]. They carry an internet-scale motion prior, fill in appearance, occluded content and contact outcomes, and already rank policies with a correlation of about $0.96$ to real success [2]. But their geometry is only up to scale and drifts, and they are optimistic: on execution-verified failure counterfactuals, released models predict success about $80\%$ of the time [6]. **3D world models** predict the future as the motion of every point in the scene, from a point cloud and the robot's own point flow [4, 5]. They are metric, grounded in the robot's kinematics, cheap to roll out and directly usable for planning, but they have no appearance, cannot anticipate contact they were not trained on, and leave occluded points unsupervised.

The one head-to-head comparison in the literature shows that the two profiles are complementary along the horizon (Figure 1). On moving points of DROID episodes, the point model PointWorld is more precise at $2$ s and worse at $8$ s, and the video model DepthWorld the reverse [3].

{{< figure src="fig1-motivation.svg" alt="Tracked-point error against horizon for a 3D-native point model and a video model with depth, and lists of what each stream is strong and weak at" caption=`(a) Tracked-point error on moving points of DROID episodes, measured at 2 s and 8 s in the DepthWorld paper [3] for its own model and for PointWorld [4]; the line between the two measurements is a guide, not data. (b) The strengths and weaknesses of the two streams are nearly disjoint, which is the condition under which fusing two estimators helps.` >}}

Every published way of combining the two streams is one-way or training-only (Figure 2). Either geometry shapes the video model during training and is then discarded, or geometry is an output that is never fed back, or geometry is an input that is never corrected. The only published closed loop, World from Motion [15], is camera-driven, has no actions, and optimises per video.

{{< figure src="fig2-couplings.svg" alt="Four coupling patterns: auxiliary 3D branch dropped at inference, 3D as a readout, 3D as conditioning, and the proposed two-way loop" caption=`How published models couple video and 3D. (a) An auxiliary geometry branch reads video features during training and is removed at inference [7, 8, 9, 10, 11]. (b) A head reads depth or point maps out of the video model as an output [3, 12]. (c) Renders, flow or simulator depth condition the video model [2, 13, 14, 15]. (d) This proposal: the state renders and conditions the video model, and the video model is read out and fused back into the state, every chunk, at inference.` >}}

| Pattern | Examples | Direction | At inference |
|---|---|---|---|
| auxiliary 3D branch | GEM-4D, HF-WM, DreamWAM, Geometry Forcing, GaussianWAM | 3D shapes video | branch removed |
| 3D as readout | DepthWorld, TesserAct | video emits 3D | no feedback |
| 3D as conditioning | Hydra-0, RoboRender, SimForcing | 3D drives video | no correction |
| per-video loop | World from Motion | both | no actions, test-time optimisation |
| 3D-native dynamics | PointWorld, UNITAS, PointCast | none | no appearance |

**Proposal.** Close the loop under robot actions. A metric point state predicts the next chunk; the video model observes that prediction as a rendering and generates the chunk; a small readout head reads the video's own 3D motion back out; a fusion step combines prediction and readout and writes the next state. The direction 3D to video is conditioning, the direction video to 3D is readout plus fusion, and both run every chunk at inference. The fusion is not an average of two errors. The predictor errs late in the horizon and at contact it has not seen; the readout errs along the viewing ray and in scale; the kinematic anchor and an as-rigid-as-possible projection remove exactly those components. The claim to test is that one model can have PointWorld's early precision, DepthWorld's late stability and a failure sensitivity that neither has.

## 2. Proposed architecture

**Setting.** One chunk is $9$ frames at $5$ Hz: the last observed frame plus $8$ generated ones, at $480\times640$, one external view for the pilot. Through the Wan2.1 VAE and patchifier this is a latent grid of $3\times30\times40$, so $3600$ tokens per chunk.

**State.** A set of metric control points in the robot base frame,
$$
S_t=\big\{(\mathbf X^k_t,\ \mathbf f^k,\ \mathbf c^k,\ \Sigma^k_t)\big\}_{k=1}^{N},\qquad N=4096\ \text{scene}+512\ \text{robot points},
$$
where $\mathbf X^k_t\in\mathbb R^3$ is the position, $\mathbf f^k\in\mathbb R^{384}$ a DINOv3 feature fixed when the point is spawned, $\mathbf c^k$ the Gaussian-skin parameters (colour, log-scale, rotation, opacity) and $\Sigma^k_t$ a diagonal covariance. Robot points are attached to links and move by forward kinematics, so the action is the robot point flow $\Delta\mathbf X^{\rm rob}_{t:t+8}$, as in PointWorld and UNITAS [4, 5]. The state is also the memory; there is no separate memory module. Figure 3 shows one chunk.

{{< figure src="fig3-architecture.svg" alt="Architecture diagram: state, 3D predictor, renderer, frozen video DiT with condition trunk, video chunk, clean-pass features, readout head, fusion, and the loop back to the state" caption=`One chunk of the predictor-corrector loop. Top row, 3D to video: the state and the action enter the 3D predictor $f_\theta$, its prediction is rendered into an eight-channel condition video $C$ and injected into a frozen Wan2.1 DiT through channel concatenation and a condition trunk at five blocks; a four-step sampler emits the video chunk. Bottom row, video to 3D: a clean pass exposes block-15 features $\mathbf m$, the readout $h_\psi$, queried by the state, returns per-point displacement, covariance and visibility, and the fusion $\mathcal F$ combines it with the prediction, fixes robot points by kinematics, projects with ARAP and writes $S_{t+1}$.` >}}

**A. 3D predictor $f_\theta$.** One token per point, built from the last three positions, the feature and the point type; robot tokens also embed the commanded flow, and one global token carries the end-effector pose and gripper command. The backbone is Point Transformer V3 [22] (12 blocks, width 512 in the pilot; PointWorld shows the family scales to 1B [4]). Five heads output a soft assignment to $16$ SE(3) clusters, a per-cluster SE(3) delta per step, a per-point residual, a per-point log-variance and a contact logit:
$$
\mu^k_f=\sum_c w_{kc}\big(\mathbf R_c\mathbf X^k_t+\mathbf t_c-\mathbf X^k_t\big)+\mathbf r^k,\qquad \Sigma^k_f=\mathrm{diag}\big(e^{\mathbf s^k}\big).
$$

**B. Video model $g_\phi$.** Wan2.1 T2V 1.3B [21], 30 blocks of width 1536, kept frozen, with a rank-64 LoRA on attention and feed-forward weights. The condition video has eight channels per frame: the skin rendered in RGB after applying $\mu_f$, its log-depth, the URDF render of the commanded robot motion, and a coverage mask. The frozen VAE encodes them, and they enter the DiT in two places: concatenated along the channel axis with the noisy latent through a new zero-initialised input projection, as in Wan's image-to-video variant, and through a light condition trunk of four DiT blocks whose tokens are added by zero-initialised linears at blocks $0, 6, 12, 18, 24$, which is the multi-block injection of SimForcing [14]. Each injection point is dropped with probability $0.3$, the skin channels are dropped jointly with probability $0.2$ (the robot-only mode used as a teacher in Section 3), and all conditions are dropped with probability $0.1$ for classifier-free guidance. The last observed frame occupies the first latent frame under a temporal mask. Inference uses a four-step distilled sampler.

**C. Readout $h_\psi$.** After sampling, one clean-latent forward pass at zero noise exposes the block-15 hidden states $\mathbf m\in\mathbb R^{3\times30\times40\times1536}$, the mid-level clean features that Gen4P found most informative [16]. Each control point becomes a query anchored at its current projected location in the anchor frame, built from $\mathbf f^k$ and a positional encoding. The query never sees $\mu_f$, so the readout cannot simply echo the predictor. Four cross-attention blocks over $\mathbf m$, with deformable sampling at the previous iterate's projected positions and two refinement iterations in the style of CoTracker and TAPIP3D [24, 25], output a displacement in the reference camera, a log-variance and a visibility per step, $(\mu^k_h,\Sigma^k_h,v^k)$. The calibrated extrinsic maps them to the base frame. A DPT depth head initialised from VGGT [23], as in DepthWorld [3], lifts pixels not covered by the state into new points with large covariance.

**D. Fusion $\mathcal F$.** A three-layer gate takes both covariances, visibility, contact and type and outputs $\alpha^k\in[0,1]^3$. The update is
$$
\hat{\Delta\mathbf X}^k=\big(\Sigma_f^{-1}+\alpha^k\Sigma_h^{-1}\big)^{-1}\big(\Sigma_f^{-1}\mu^k_f+\alpha^k\Sigma_h^{-1}\mu^k_h\big),
$$
with robot points overwritten by forward kinematics. An as-rigid-as-possible projection then solves
$$
\min_{\mathbf Y}\ \sum_k\big\|\mathbf Y^k-(\mathbf X^k_t+\hat{\Delta\mathbf X}^k)\big\|^2_{\hat\Sigma_k^{-1}}+\lambda\sum_{(k,j)\in\mathcal N}\big\|(\mathbf Y^k-\mathbf Y^j)-\mathbf R_k(\mathbf X^k_t-\mathbf X^j_t)\big\|^2,
$$
with local rotations $\mathbf R_k$ from Procrustes, three alternations and a sparse differentiable solve. The posterior covariance is $(\Sigma_f^{-1}+\alpha\Sigma_h^{-1})^{-1}$ plus process noise. Skin colour, opacity and scale take five Adam steps against the generated frames through a differentiable splatter, masked by visibility. Points spawn where coverage is low and die after several invisible chunks with inflated covariance.

**Per-chunk flow and cost.** Predictor, then rendering with the splatter and the URDF, then four DiT steps, then the clean pass and readout, then fusion. The predictor runs in about $0.1$ s and the video stage in about $1$ s on one H100, so the predictor can run several chunks between video corrections (every $K$ chunks, or whenever $\Sigma_f$ spikes). Each chunk emits the video, the state, and a per-point disagreement $d^k_t=\|\mu^k_f-\mu^k_h\|_{\Sigma}$, which is a free failure and novelty signal.

| Module | Backbone | Trainable | Output |
|---|---|---|---|
| A. 3D predictor $f_\theta$ | PTv3, 12 blocks, width 512 | 100M (1B full) | $\mu_f,\Sigma_f$, clusters, contact |
| B. video model $g_\phi$ | Wan2.1 1.3B, frozen | LoRA 50M + trunk 60M | 8 frames |
| C. readout $h_\psi$ | 4 cross-attention blocks + DPT head | 25M | $\mu_h,\Sigma_h,v$, new points |
| D. fusion $\mathcal F$ | gate MLP + closed form + ARAP | 0.1M | $S_{t+1}$, $d_t$ |

## 3. Training method

Training has four stages (Figure 4). Each stage reuses a published recipe, and only the last one is new.

{{< figure src="fig4-training.svg" alt="Four training stages showing which module is trained, frozen or unused, with data and loss per stage, and a detail of the closed-loop stage with the counterfactual teacher" caption=`Top: the four stages, with the module that each one trains, its data and its loss. Bottom: the closed-loop stage unrolls the chunk loop over several states, supervises the fused state with real 2D tracks and with the generated frames, adds an agreement term, and distils counterfactual responses from the video model in robot-only mode. On the video side only the LoRA, the trunk, the readout and the gate receive gradients.` >}}

**Stage 1, predictor, no 3D labels.** On DROID-3D [3] and RoboTwin 2.0, the predictor is trained by reprojecting its prediction onto 2D tracks, with the covariance projected into the image, plus the as-rigid-as-possible regulariser and a kinematic term that ties robot points to forward kinematics:
$$
\mathcal L_f=\sum_{k,h}-\log\mathcal N\!\Big(\mathbf p^k_{t+h};\ \pi\big(\mathbf X^k_t+\mu^k_{f,h}\big),\ \mathbf J\Sigma^k_{f,h}\mathbf J^{\!\top}\Big)+\lambda_{\rm ARAP}\mathcal L_{\rm ARAP}+\lambda_{\rm FK}\big\|\mu^{\rm rob}_f-\Delta\mathbf X^{\rm FK}\big\|^2 .
$$
The robot body supplies the metric scale and the camera placement, which is what DROID-3D and PointWorld already exploit [3, 4].

**Stage 2, renderer.** Standard flow matching on the LoRA and the trunk, with conditions rendered from stage-1 states and the per-block dropout above:
$$
\mathcal L_{\rm FM}=\mathbb E_{\tau,x_0,x_1}\big\|v_\phi(x_\tau,\tau,C)-(x_1-x_0)\big\|^2 .
$$
This is the RoboRender and SimForcing recipe [13, 14], with the state in place of a simulator.

**Stage 3, readout.** The head is trained on the frozen DiT with pseudo 3D tracks from SpatialTrackerV2 or TrackCraft3R [19, 17], made metric by the DROID-3D calibration, following MotionForesight [18]:
$$
\mathcal L_h=\sum_{k,h}-\log\mathcal N\big(\mathbf y^k_h;\ \mu^k_{h},\ \Sigma^k_{h}\big)+\mathrm{BCE}\big(v^k_h\big)+\mathcal L_{\rm reproj} .
$$

**Stage 4, closed loop.** The loop is unrolled over $N$ chunks and four terms are summed:
$$
\mathcal L=\mathcal L_{\rm track}(\hat S)+\lambda_1\,\mathcal L_{\rm corr}(\text{generated frames},\hat S)+\lambda_2\sum_k v^k\big\|\mu^k_f-\mathrm{sg}(\mu^k_h)\big\|^2+\lambda_3\,\mathcal L_{\rm cf} .
$$
The track term reprojects the fused state onto real 2D tracks. The correspondence term reprojects the fused state onto the generated frames, which is the correspondence loss of GEM-4D [7] with the state as the source of truth; it is what makes the video action-faithful. The agreement term pulls the predictor toward the readout on visible points. The counterfactual term samples a never-executed action $a'$, runs the video model in robot-only mode (skin channels dropped, URDF render of $a'$) as a teacher that imagines the response of the scene, reads its tracks out with $h_\psi$, and trains the predictor to match them, weighted by cross-sample consistency. On the video side only the LoRA, the trunk, the readout and the gate receive gradients; the DiT backbone stays frozen so that a noisy readout cannot corrupt the prior, which is the lesson GEM-4D learned with its asymmetric coupling [7]. Trainable parameters in the pilot total about $235$M against a frozen 1.3B DiT.

## 4. Evaluation plan and kill criteria

The headline table measures both directions, which no existing paper does.

| Direction | Metric | Ablation |
|---|---|---|
| video to 3D | moving-region ADE/FDE at 2, 4, 8 s | $f_\theta$ alone vs fused |
| 3D to video | kinematic alignment [20], object-motion error [2], counterfactual optimism [6] | $g_\phi$ with vs without state |
| loop | drift over 20 s; disagreement $d_t$ against verified failures | $K=1$ vs $K=\infty$ |

Three risks decide whether this becomes a paper, and each has a cheap pilot.

1. **The readout may be too noisy.** MotionForesight under-predicts motion magnitude with a ratio of about $0.72$ [18]. Pilot: bin readout error and predictor error by horizon and by contact on about $1\,000$ DROID-3D clips. If the readout is never the more precise source in any bin, fusion cannot help and the design collapses to a readout model.
2. **The loop may be circular.** If the video merely echoes the state it was conditioned on, the readout agrees with $\mu_f$ even when $\mu_f$ is wrong. Pilot: perturb $\mu_f$ and require the readout to disagree. This is also the counterfactual test a reviewer will ask for.
3. **Latency.** Four-step sampling of a 1.3B DiT costs about a second per chunk, fine for policy evaluation and the $K$-chunk schedule, not for $10$ Hz model-predictive control.

## 5. References

1. Ctrl-World. [arXiv:2510.10125](https://arxiv.org/abs/2510.10125)
2. Hydra-0: Action Flow for Generalist World Modeling and Control. [arXiv:2608.18077](https://arxiv.org/abs/2608.18077)
3. DepthWorld: 3D World Model for Robot Manipulation. [arXiv:2610.08780](https://arxiv.org/abs/2610.08780)
4. PointWorld. [arXiv:2601.03782](https://arxiv.org/abs/2601.03782)
5. UNITAS. [arXiv:2610.12099](https://arxiv.org/abs/2610.12099)
6. CureWM. [arXiv:2610.09134](https://arxiv.org/abs/2610.09134)
7. GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation. [arXiv:2605.22882](https://arxiv.org/abs/2605.22882)
8. HF-WM. [arXiv:2610.13522](https://arxiv.org/abs/2610.13522)
9. DreamWAM. [arXiv:2608.04996](https://arxiv.org/abs/2608.04996)
10. Geometry Forcing. [arXiv:2507.07982](https://arxiv.org/abs/2507.07982)
11. GaussianWAM. [arXiv:2608.24714](https://arxiv.org/abs/2608.24714)
12. TesserAct. [arXiv:2504.20995](https://arxiv.org/abs/2504.20995)
13. RoboRender. [arXiv:2610.09254](https://arxiv.org/abs/2610.09254)
14. SimForcing. [arXiv:2610.06598](https://arxiv.org/abs/2610.06598)
15. World from Motion. [arXiv:2607.01202](https://arxiv.org/abs/2607.01202)
16. Gen4P: Video Generators Are 4D Perceivers. [arXiv:2610.22371](https://arxiv.org/abs/2610.22371)
17. TrackCraft3R. [arXiv:2605.12587](https://arxiv.org/abs/2605.12587)
18. MotionForesight. [arXiv:2607.16192](https://arxiv.org/abs/2607.16192)
19. SpatialTrackerV2. [arXiv:2507.12462](https://arxiv.org/abs/2507.12462)
20. R2-WAM. [arXiv:2610.04913](https://arxiv.org/abs/2610.04913)
21. Wan: Open and Advanced Large-Scale Video Generative Models. [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)
22. Point Transformer V3. [arXiv:2312.10035](https://arxiv.org/abs/2312.10035)
23. VGGT. [arXiv:2503.11651](https://arxiv.org/abs/2503.11651)
24. CoTracker3. [arXiv:2410.11831](https://arxiv.org/abs/2410.11831)
25. TAPIP3D. [arXiv:2504.14717](https://arxiv.org/abs/2504.14717)
