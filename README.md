<div align="center">

# Awesome World Models for Embodied AI [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

**Two schools of world models — one that understands the world by *generating* it, one that understands it by *acting* in it — and the papers, architectures, simulators and companies behind each.**

[English](README.md) · [中文](README.zh-CN.md)

</div>

---

## Why this list is organized this way

Every world model answers the same question — learn a dynamics you can condition and roll out:

$$p(s_{t+1} \mid s_{\le t},\, a_{\le t})$$

The disagreement is never about that equation. It is about **what `s` is** and **where the supervision comes from**. That single split produces two schools with different data, different backbones, different failure modes — and different companies.

| | **Generative** (generate the world) | **Action-centric** (act in the world) |
| :--- | :--- | :--- |
| Criterion | Predicting/rendering the future = understanding | Acting correctly = understanding |
| State `s` | Pixels / video latent / 3D geometry | Compact task-relevant features, often implicit |
| Supervision | Reconstruction, denoising, next-token (self-supervised) | Action supervision (BC), reward (RL) |
| Data | Internet video, multi-view photos — abundant, **no action labels** | Teleop / real robots / sim — scarce, **with action labels** |
| Backbone | DiT · causal Transformer · RSSM | VLM + action expert |
| Output | Simulators, data engines, evaluators | A policy `π` |
| Weakness | Drift, no physical guarantees, expensive | Narrow generalization, expensive data, no long-horizon imagination |

Notation used throughout: **route numbers** (1.1–1.5, 2.1–2.3) are consistent across both language versions.

## Contents

- [1. Generative World Models](#1-generative-world-models)
  - [0. Foundations](#0-foundations--the-shared-parts-bin)
  - [1.1 Video / Latent World Models](#11-video--latent-world-models)
  - [1.2 Autoregressive Token World Models](#12-autoregressive-token-world-models)
  - [1.3 Compact Latent-State World Models](#13-compact-latent-state-world-models)
  - [1.4a Feed-Forward 3D Reconstruction](#14a-feed-forward-3d-reconstruction)
  - [1.4b 3D / Scene Generation](#14b-3d--scene-generation)
  - [1.4c From Viewable to Interactable](#14c-from-viewable-to-interactable)
  - [1.5 Neural + Physics Hybrids](#15-neural--physics-hybrids)
- [2. Action World Models](#2-action-world-models)
  - [2.1 VLA](#21-vla--vision-language-action)
  - [2.2 WAM](#22-wam--world-action-models)
  - [2.3 Hierarchical](#23-hierarchical--llm-planner--low-level-skills)
- [3. Simulators & Benchmarks](#3-simulators--benchmarks)
- [4. Surveys](#4-surveys)
- [5. Companies & Products](#5-companies--products)
- [Open Questions](#open-questions)
- [Contributing](#contributing)

---

## 1. Generative World Models

> *Bet: dynamics can be learned self-supervised from "predict the next frame / next view", so it can eat internet-scale unlabeled video.*

### 0. Foundations — the shared parts bin

Every world model in this list is assembled from these. Skip if you already know them.

- **[ViT](https://arxiv.org/abs/2010.11929)** `2020` · *Google* — Patchify + Transformer. Every video/3D world model still tokenizes space this way.
- **[DDPM](https://arxiv.org/abs/2006.11239)** `2020` · *UC Berkeley* — The denoising objective that underlies nearly every generative world model.
- **[Latent Diffusion](https://arxiv.org/abs/2112.10752)** `2021` · *LMU Munich* — Diffuse in a VAE latent instead of pixels — the reason video world models are affordable.
- **[DiT](https://arxiv.org/abs/2212.09748)** `2022` · *Meta / NYU* — Replaces the U-Net with a Transformer and shows clean scaling laws. The default backbone of video and 3D world models.
- **[Flow Matching](https://arxiv.org/abs/2210.02747)** `2022` · *Meta* — Straight-line probability paths; now the standard training objective over DDPM for both video and action experts.
- **[VQ-VAE](https://arxiv.org/abs/1711.00937)** `2017` · *DeepMind* — Discrete codebooks — the mechanism behind both visual tokenizers and latent action models.
- **[VQGAN](https://arxiv.org/abs/2012.09841)** `2020` · *Heidelberg* — Adversarial visual tokenizer that made autoregressive image/video modeling practical.
- **[NeRF](https://arxiv.org/abs/2003.08934)** `2020` · *UC Berkeley* — Implicit volumetric scene as an MLP. Started the modern 3D reconstruction wave.
- **[3DGS](https://arxiv.org/abs/2308.04079)** `2023` · *Inria* — Explicit, real-time, differentiable rendering primitive — the substrate of today's 3D world generators.

### 1.1 Video / Latent World Models

Compress the world into a spatio-temporal latent, then generate conditioned on actions. **Causal 3D VAE → DiT → flow matching** is the near-universal recipe. The line between "a video model" and "a world model" is exactly one thing: *action conditioning*.

- **[VDM](https://arxiv.org/abs/2204.03458)** `2022` · *Google* — First diffusion model over the time axis; introduces the 3D U-Net factorization everything inherited.
- **[Sora](https://openai.com/index/video-generation-models-as-world-simulators/)** · *OpenAI* — Tech report that made 'video generation = world simulator' a mainstream claim; spacetime patches + DiT at scale.
- **[Genie](https://arxiv.org/abs/2402.15391)** `2024` · *Google DeepMind* — **The key idea of this route**: an unsupervised *latent action model* learned from unlabeled video, turning the internet into interaction data.
- **[Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)** · *Google DeepMind* — Real-time interactive world model at 720p/24fps with minute-scale environment memory. No paper — blog only.
- **[Cosmos](https://arxiv.org/abs/2501.03575)** `2025` · *NVIDIA* — Open world foundation models (diffusion + autoregressive) plus tokenizers, explicitly positioned as a data engine for robotics.
- **[Cosmos-Reason](https://arxiv.org/abs/2503.15558)** `2025` · *NVIDIA* — The reasoning VLM of the Cosmos stack — physical common sense, data curation, and policy evaluation.
- **[GameNGen](https://arxiv.org/abs/2408.14837)** `2024` · *Google* — DOOM simulated at 20 FPS by a diffusion model conditioned on keypresses. The existence proof for playable world models.
- **[Oasis](https://oasis-model.github.io/)** · *Decart / Etched* — Open real-time interactive Minecraft world model; the reference point for latency-first world modeling.
- **[The Matrix](https://arxiv.org/abs/2412.03568)** `2024` · *Alibaba et al.* — Infinite-horizon streaming generation with real-time movement control.
- **[Matrix-Game](https://arxiv.org/abs/2506.18701)** `2025` · *Skywork AI* — Open interactive world foundation model for game environments, with an action-controllable benchmark.
- **[NWM](https://arxiv.org/abs/2412.03572)** `2024` · *Meta / NYU* — Conditional diffusion transformer over navigation trajectories; plans by imagining and scoring futures.
- **[Diffusion Forcing](https://arxiv.org/abs/2407.01392)** `2024` · *MIT* — Per-token noise levels unify autoregression and diffusion — the standard trick for causal, variable-length rollout.
- **[Self Forcing](https://arxiv.org/abs/2506.08009)** `2025` · *Adobe / UT Austin* — Train on your own rollouts to kill exposure bias; real-time streaming video generation.
- **[CausVid](https://arxiv.org/abs/2412.07772)** `2024` · *MIT / Adobe* — Distills a bidirectional video diffusion model into a fast causal one for interactive use.
- **[LAPA](https://arxiv.org/abs/2410.11758)** `2024` · *KAIST / Microsoft* — Learns discrete latent actions from action-free human video, then fine-tunes to real robot actions.
- **[LAPO](https://arxiv.org/abs/2312.10812)** `2023` · *Oxford / UCL* — Recovers a latent action space purely from observations — the clean formulation of the idea.
- **[Moto](https://arxiv.org/abs/2412.04445)** `2024` · *HKU / ByteDance* — Latent motion tokens as the shared language between video pretraining and robot control.
- **[GAIA-1](https://arxiv.org/abs/2309.17080)** `2023` · *Wayve* — Driving world model conditioned on video, text and ego-actions; the first credible industrial world model.
- **[GAIA-2](https://arxiv.org/abs/2503.20523)** `2025` · *Wayve* — Multi-camera, structurally controllable successor built for scenario generation and safety testing.
- **[Vista](https://arxiv.org/abs/2405.17398)** `2024` · *OpenDriveLab* — Open high-fidelity driving world model with multi-modal action control.
- **[HunyuanWorld](https://arxiv.org/abs/2507.21809)** `2025` · *Tencent* — Open panorama-to-mesh pipeline producing explorable, semantically layered 3D worlds.
- **[Wan](https://arxiv.org/abs/2503.20314)** `2025` · *Alibaba* — Strongest fully open video foundation model family; the de-facto academic base model for world-model research.
- **[CogVideoX](https://arxiv.org/abs/2408.06072)** `2024` · *Zhipu / Tsinghua* — Open DiT video model with a 3D causal VAE and expert adaptive LayerNorm.

### 1.2 Autoregressive Token World Models

Quantize frames into discrete tokens and run next-token prediction. Structurally identical to an LLM — which is why action tokens can be mixed into the same sequence (see [WAM](#22-wam--world-action-models)).

- **[IRIS](https://arxiv.org/abs/2209.00588)** `2022` · *Geneva* — Discrete autoencoder + causal Transformer world model; agent trained purely inside imagination.
- **[MAGVIT-v2](https://arxiv.org/abs/2310.05737)** `2023` · *Google* — Lookup-free quantization makes an LLM-style video tokenizer beat diffusion on quality.
- **[VideoPoet](https://arxiv.org/abs/2312.14125)** `2023` · *Google* — One decoder-only LLM over interleaved image/video/audio tokens.
- **[Emu3](https://arxiv.org/abs/2409.18869)** `2024` · *BAAI* — Pure next-token prediction across text, image and video in a single vocabulary.
- **[LWM](https://arxiv.org/abs/2402.08268)** `2024` · *UC Berkeley* — Million-token context over video and text; long-horizon world memory as a context-length problem.

### 1.3 Compact Latent-State World Models

Don't render pixels — keep a small latent state and train the policy *inside imagination*. The model-based RL lineage, plus LeCun's JEPA counter-proposal.

- **[World Models (Ha & Schmidhuber)](https://arxiv.org/abs/1809.01999)** `2018` · *Google Brain* — The paper that named the field: VAE + RNN dynamics + tiny controller trained in a dream.
- **[PlaNet](https://arxiv.org/abs/1811.04551)** `2018` · *Google* — Introduces the RSSM (deterministic + stochastic latent) and latent-space MPC.
- **[Dreamer](https://arxiv.org/abs/1912.01603)** `2019` · *Google / Toronto* — Actor-critic trained entirely on imagined latent rollouts.
- **[DreamerV2](https://arxiv.org/abs/2010.02193)** `2020` · *Google* — Categorical latents make the world model work on Atari at human level.
- **[DreamerV3](https://arxiv.org/abs/2301.04104)** `2023` · *DeepMind* — One hyperparameter set across 150+ tasks; collects diamonds in Minecraft from scratch.
- **[TD-MPC](https://arxiv.org/abs/2203.04955)** `2022` · *UC San Diego* — Task-oriented latent dynamics + online planning, no observation reconstruction.
- **[TD-MPC2](https://arxiv.org/abs/2310.16828)** `2023` · *UC San Diego* — Decoder-free world model that scales to a single agent across 100+ continuous-control tasks.
- **[MuZero](https://arxiv.org/abs/1911.08265)** `2019` · *DeepMind* — Learns only what matters for value and policy — the purest 'implicit world model'.
- **[I-JEPA](https://arxiv.org/abs/2301.08243)** `2023` · *Meta* — Predict in representation space, not pixel space. LeCun's counter-proposal to generative world models.
- **[V-JEPA](https://arxiv.org/abs/2404.08471)** `2024` · *Meta* — The video version: masked feature prediction with an EMA target encoder.
- **[V-JEPA 2 / V-JEPA 2-AC](https://arxiv.org/abs/2506.09985)** `2025` · *Meta* — Internet-scale video pretraining, then a small action-conditioned head enabling zero-shot robot planning by MPC.
- **[DayDreamer](https://arxiv.org/abs/2206.14176)** `2022` · *UC Berkeley* — Dreamer trained directly on physical robots — world models under real-world sample budgets.

### 1.4a Feed-Forward 3D Reconstruction

Real world → geometry, in one forward pass. Reconstruction stopped being optimization and became inference.

- **[DUSt3R](https://arxiv.org/abs/2312.14132)** `2023` · *Naver Labs* — Regresses pointmaps directly from image pairs — no camera poses, no SfM pipeline.
- **[MASt3R](https://arxiv.org/abs/2406.09756)** `2024` · *Naver Labs* — Adds dense matching, turning DUSt3R into a full localization + reconstruction stack.
- **[VGGT](https://arxiv.org/abs/2503.11651)** `2025` · *Oxford / Meta* — One feed-forward transformer predicting poses, depth, pointmaps and tracks in a single pass.
- **[LRM](https://arxiv.org/abs/2311.04400)** `2023` · *Adobe / ANU* — Single-image 3D in 5 seconds by scaling a transformer over triplanes — reconstruction as inference, not optimization.
- **[GS-LRM](https://arxiv.org/abs/2404.19702)** `2024` · *Adobe* — Feed-forward 3D Gaussians from sparse views.

### 1.4b 3D / Scene Generation

Geometry out of nothing. The field moved from SDS distillation (slow) to **native 3D latent generation** (3D VAE + latent DiT / rectified flow).

- **[DreamFusion](https://arxiv.org/abs/2209.14988)** `2022` · *Google* — Score Distillation Sampling — the optimization-based era of text-to-3D.
- **[Zero-1-to-3](https://arxiv.org/abs/2303.11328)** `2023` · *Columbia* — Camera-conditioned novel view synthesis as the bridge from 2D diffusion priors to 3D.
- **[MVDream](https://arxiv.org/abs/2308.16512)** `2023` · *ByteDance* — Generate consistent multi-view sets first, then lift — fixes the Janus problem.
- **[CAT3D](https://arxiv.org/abs/2405.10314)** `2024` · *Google* — Multi-view diffusion + fast 3D fitting; minutes instead of hours.
- **[CLAY](https://arxiv.org/abs/2406.13897)** `2024` · *Deemos / ShanghaiTech* — Large-scale native 3D latent diffusion over neural fields; the backbone behind Rodin.
- **[TRELLIS](https://arxiv.org/abs/2412.01506)** `2024` · *Microsoft / Tsinghua* — Structured latents (SLAT) decode into meshes, 3DGS or radiance fields from one model.
- **[Hunyuan3D 2.0](https://arxiv.org/abs/2501.12202)** `2025` · *Tencent* — Open two-stage shape-then-texture generation; the most widely deployed open 3D asset model.
- **[TripoSR](https://arxiv.org/abs/2403.02151)** `2024` · *Tripo / Stability* — Open LRM-style reconstruction in under a second on a consumer GPU.
- **[TripoSG](https://arxiv.org/abs/2502.06608)** `2025` · *Tripo / VAST* — Large-scale rectified-flow shape generation; a common open reference implementation.
- **[MIDI](https://arxiv.org/abs/2412.03558)** `2024` · *VAST* — Generates a scene as **separable object instances** — the step that makes a scene editable and simulatable.

### 1.4c From Viewable to Interactable

**The bottleneck of the whole field.** A pretty mesh is not a world model. To simulate you also need parts, joints, mass, friction, collision geometry and a URDF/MJCF/USD export. Fewest papers, highest leverage.

- **[Articulate-Anything](https://arxiv.org/abs/2410.13882)** `2024` · *Michigan / NVIDIA* — VLM writes the articulation program: turns images/videos into jointed URDF objects.
- **[URDFormer](https://arxiv.org/abs/2405.11656)** `2024` · *UW* — Predicts kinematic structure and URDF parameters straight from a single real photo.
- **[Real2Code](https://arxiv.org/abs/2406.08474)** `2024` · *Columbia* — Represents articulation as generated code, giving exact joint semantics instead of a soup of meshes.
- **[PhysGen](https://arxiv.org/abs/2409.18964)** `2024` · *UIUC* — Infers physical properties from an image and runs a real rigid-body simulator to drive generation.
- **[EmbodiedGen](https://github.com/HorizonRobotics/EmbodiedGen)** · *Horizon Robotics* — Generative 3D world engine that outputs physically plausible, **simulation-ready** assets and scenes (URDF/MJCF/USD).
- **[EmbodiedGen (paper)](https://arxiv.org/abs/2506.10600)** `2025` · *Horizon Robotics* — The paper: a full asset-to-simulation toolchain, closing the gap between 3D generation and physics engines.
- **[RialTo](https://arxiv.org/abs/2403.03949)** `2024` · *MIT* — Scan the real scene, make it a simulator, RL there, deploy back — the canonical real2sim2real loop.
- **[Infinigen](https://arxiv.org/abs/2306.09310)** `2023` · *Princeton* — Fully procedural, fully annotated worlds — the non-neural answer to synthetic data.

### 1.5 Neural + Physics Hybrids

Learn dynamics with physical structure baked in: graph-network particles, neural PDE operators, differentiable simulators. Extrapolates and stays verifiable; hard to feed with video.

- **[GNS](https://arxiv.org/abs/2002.09405)** `2020` · *DeepMind* — Graph network particle simulator that generalizes far outside its training distribution.
- **[DPI-Net](https://arxiv.org/abs/1810.01566)** `2018` · *MIT* — Learned particle dynamics used directly for model-based manipulation control.
- **[FNO](https://arxiv.org/abs/2010.08895)** `2020` · *Caltech* — Learns solution operators for PDE families — the core of industrial surrogate models.
- **[PINN](https://arxiv.org/abs/1711.10561)** `2017` · *Brown* — Bakes the PDE residual into the loss; physics as a regularizer.
- **[DiffTaichi](https://arxiv.org/abs/1910.00935)** `2019` · *MIT* — Differentiable physics as a programming language — gradients through contact and deformation.
- **[Genesis](https://github.com/Genesis-Embodied-AI/Genesis)** · *Genesis-Embodied-AI* — GPU-parallel, differentiable, multi-physics engine; the fastest-growing open simulator in embodied research.

---

## 2. Action World Models

> *Bet: you do not have to render the world. The world model can be a by-product of the policy, or live implicitly in the weights.*

### 2.1 VLA — Vision-Language-Action

All of these are the same shape: $\pi(a_{t:t+H} \mid o_t, \ell)$, predicting an **action chunk**. The only real disagreement is how the action head is modeled — discrete bins, compressed tokens, diffusion, or flow matching.

- **[RT-1](https://arxiv.org/abs/2212.06817)** `2022` · *Google* — Tokenized actions + a Transformer over 130k real episodes. The template for everything after.
- **[RT-2](https://arxiv.org/abs/2307.15818)** `2023` · *Google DeepMind* — Treats actions as text tokens inside a VLM — web semantics transfer to manipulation.
- **[Open X-Embodiment / RT-X](https://arxiv.org/abs/2310.08864)** `2023` · *Consortium* — 1M+ trajectories across 22 embodiments; proved positive transfer across robot bodies.
- **[Octo](https://arxiv.org/abs/2405.12213)** `2024` · *UC Berkeley* — Open transformer policy with a diffusion head and swappable observation/action interfaces.
- **[OpenVLA](https://arxiv.org/abs/2406.09246)** `2024` · *Stanford et al.* — 7B open VLA (SigLIP+DINOv2+Llama-2) with discretized action bins; the standard academic baseline.
- **[OpenVLA-OFT](https://arxiv.org/abs/2502.19645)** `2025` · *Stanford* — Careful study of what actually matters when adapting a VLA: parallel decoding, action chunking, continuous heads.
- **[Diffusion Policy](https://arxiv.org/abs/2303.04137)** `2023` · *Columbia / TRI* — Models the *distribution* of action sequences — solves multimodality that regression averages away.
- **[ACT / ALOHA](https://arxiv.org/abs/2304.13705)** `2023` · *Stanford* — Action chunking + a \$20k bimanual rig; the reason 'action chunk' is now universal.
- **[π0](https://arxiv.org/abs/2410.24164)** `2024` · *Physical Intelligence* — PaliGemma-3B backbone + a ~300M **flow-matching action expert** at 50Hz. The dominant modern VLA architecture.
- **[π0-FAST](https://arxiv.org/abs/2501.09747)** `2025` · *Physical Intelligence* — DCT + BPE compression of action sequences, letting autoregressive VLAs reach high control rates.
- **[π0.5](https://arxiv.org/abs/2504.16054)** `2025` · *Physical Intelligence* — Hierarchical inference (high-level subtask → low-level action) + heterogeneous co-training; cleans never-before-seen homes.
- **[RDT-1B](https://arxiv.org/abs/2410.07864)** `2024` · *Tsinghua* — 1.2B diffusion transformer with a unified action space across heterogeneous bimanual robots.
- **[GR00T N1](https://arxiv.org/abs/2503.14734)** `2025` · *NVIDIA* — The open reference for the **dual-system** design: VLM System 2 + DiT action System 1.
- **[Gemini Robotics](https://arxiv.org/abs/2503.20020)** `2025` · *Google DeepMind* — Gemini-based VLA plus a separate embodied-reasoning (ER) model for spatial grounding and planning.
- **[SmolVLA](https://arxiv.org/abs/2506.01844)** `2025` · *Hugging Face* — Small, community-data-trained VLA that runs on consumer hardware; asynchronous inference.
- **[GO-1 / AgiBot World](https://arxiv.org/abs/2503.06669)** `2025` · *AgiBot* — Large open real-robot dataset plus a latent-planner VLA trained on human and robot video.
- **[LBM](https://arxiv.org/abs/2507.05331)** `2025` · *Toyota Research Institute* — Rigorous evidence that multitask pretraining beats single-task policies — the industrial validation of diffusion policies.
- **[ECoT](https://arxiv.org/abs/2407.08693)** `2024` · *UC Berkeley* — Makes the policy emit subtasks, object grounding and gripper poses before acting.
- **[PerAct](https://arxiv.org/abs/2209.05451)** `2022` · *UW / NVIDIA* — Voxelized 3D observations + Perceiver; actions as 3D keyframes rather than pixels.
- **[Act3D](https://arxiv.org/abs/2306.17817)** `2023` · *CMU* — Coarse-to-fine 3D feature fields for sample-efficient manipulation.
- **[3D Diffuser Actor](https://arxiv.org/abs/2402.10885)** `2024` · *CMU* — Combines diffusion policies with explicit 3D scene tokens.
- **[RVT-2](https://arxiv.org/abs/2406.08545)** `2024` · *NVIDIA* — Multi-view re-rendering instead of voxels — fast, precise, few-shot.
- **[SpatialVLA](https://arxiv.org/abs/2501.15830)** `2025` · *Shanghai AI Lab* — Ego3D position encoding and adaptive action grids for cross-embodiment spatial transfer.
- **[UMI](https://arxiv.org/abs/2402.10329)** `2024` · *Stanford* — Handheld gripper data collection in the wild — the cheapest scalable source of action-labeled data.
- **[HIL-SERL](https://arxiv.org/abs/2410.21845)** `2024` · *UC Berkeley* — Real-world RL with human corrections reaching near-100% success on contact-rich tasks.

### 2.2 WAM — World-Action Models

Predict the future *and* act. Action labels are expensive; predicting the next frame is free — so use prediction as pretraining, as a subgoal generator, or as a planner.

- **[UniPi](https://arxiv.org/abs/2302.00111)** `2023` · *Google / MIT* — Generate the future video, then invert it into actions with an inverse dynamics model.
- **[UniSim](https://arxiv.org/abs/2310.06114)** `2023` · *UC Berkeley / Google* — An action-conditioned video simulator of the real world, used to train policies that transfer zero-shot.
- **[GR-1](https://arxiv.org/abs/2312.13139)** `2023` · *ByteDance* — Video prediction pretraining, then a light action head — prediction as free supervision.
- **[GR-2](https://arxiv.org/abs/2410.06158)** `2024` · *ByteDance* — Scales the recipe to web video and multi-task generalization.
- **[3D-VLA](https://arxiv.org/abs/2403.09631)** `2024` · *UMass* — Generative world model that imagines future point clouds and images before acting.
- **[WorldVLA](https://arxiv.org/abs/2506.21539)** `2025` · *Alibaba* — One autoregressive sequence containing both image and action tokens, each predicting the other.
- **[iVideoGPT](https://arxiv.org/abs/2405.15223)** `2024` · *Tsinghua* — Scalable interactive autoregressive world model for model-based RL and visual control.
- **[SuSIE](https://arxiv.org/abs/2310.10639)** `2023` · *UC Berkeley* — Image-editing diffusion proposes visual subgoals; a low-level policy chases them.
- **[AVDC](https://arxiv.org/abs/2310.08576)** `2023` · *MIT* — Extracts actions from generated video via dense correspondence — no action labels at all.
- **[Genie Envisioner](https://arxiv.org/abs/2508.05635)** `2025` · *AgiBot* — Unified platform where the same video world model serves as policy, simulator and evaluator.

### 2.3 Hierarchical — LLM Planner + Low-Level Skills

Keep the reasoning in a frozen LLM/VLM and let a classical planner or a skill library close the loop. Interpretable and zero-shot composable; limited by the skill library and low control rates.

- **[SayCan](https://arxiv.org/abs/2204.01691)** `2022` · *Google* — LLM proposes, a learned value function checks feasibility.
- **[Code as Policies](https://arxiv.org/abs/2209.07753)** `2022` · *Google* — The LLM writes executable policy code calling perception and control APIs.
- **[VoxPoser](https://arxiv.org/abs/2307.05973)** `2023` · *Stanford* — LLM+VLM compose 3D affordance and constraint maps; a motion planner solves the rest. No robot data needed.
- **[ReKep](https://arxiv.org/abs/2409.01652)** `2024` · *Stanford* — Expresses tasks as relational keypoint constraints, turning manipulation into a solvable optimization.

---

### 3. Simulators & Benchmarks

The seam between the two schools. Generated assets land here; policies are born here.

- **[Isaac Sim / Isaac Lab](https://github.com/isaac-sim/IsaacLab)** · *NVIDIA* — The de-facto industrial standard: USD scenes, GPU-parallel RL, photoreal rendering.
- **[MuJoCo / MJX](https://github.com/google-deepmind/mujoco)** · *Google DeepMind* — The most trusted contact dynamics; MJX brings it to GPU-scale RL.
- **[SAPIEN](https://arxiv.org/abs/2003.08515)** `2020` · *UC San Diego* — Part-level articulated objects (PartNet-Mobility) — the standard testbed for interacting with joints.
- **[ManiSkill3](https://arxiv.org/abs/2410.00425)** `2024` · *UC San Diego* — GPU-parallel simulation *and* rendering, tens of thousands of FPS for visual RL.
- **[RoboCasa](https://arxiv.org/abs/2406.02523)** `2024` · *UT Austin / NVIDIA* — Generative-asset-populated kitchens; a bridge between 3D generation and robot learning.
- **[BEHAVIOR-1K](https://arxiv.org/abs/2403.09227)** `2024` · *Stanford* — 1,000 human-chosen household activities with rich physical state (heat, wetness, soiling).
- **[Habitat 3.0](https://arxiv.org/abs/2310.13724)** `2023` · *Meta* — Fast photorealistic simulation for navigation and human-robot cohabitation.
- **[Orbit (→ Isaac Lab)](https://arxiv.org/abs/2301.04195)** `2023` · *ETH / NVIDIA* — The framework paper behind Isaac Lab's task/environment abstractions.

### 4. Surveys

- **[World Models Survey](https://arxiv.org/abs/2411.14499)** `2024` — Broad survey covering both the generative and the decision-making lineages.
- **[Driving World Models Survey](https://arxiv.org/abs/2501.11260)** `2025` — The most mature vertical application of world models.
- **[VLA Survey](https://arxiv.org/abs/2405.14093)** `2024` — Taxonomy of VLA architectures, data and benchmarks.

---

## 5. Companies & Products

Where the money and the products actually sit. `Route` refers to the numbering above.

### Generative — 3D assets (route 1.4b → 1.4c)

| Company / Product | Route | What they actually do |
| :--- | :--- | :--- |
| **[Meshy](https://www.meshy.ai/)** | 1.4b | Native 3D latent generation + multi-view PBR texture diffusion. Art-pipeline oriented; no physics. |
| **[Tripo AI](https://www.tripo3d.ai/)** (VAST) | 1.4b → 1.4c | Feed-forward reconstruction (TripoSR) → rectified-flow shape generation (TripoSG); pushing part-level and multi-instance scenes. Most aggressive open-sourcer. |
| **Seed3D** (ByteDance Seed) | 1.4b → 1.4c | Single image → **simulation-ready** asset. Explicitly optimizes watertightness, real scale and collision quality, not just looks. |
| **[Tencent Hunyuan3D](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)** | 1.4b | Largest open shape+texture model family; HunyuanWorld extends it to explorable scenes. |
| **[Rodin](https://hyper3d.ai/)** (Deemos) | 1.4b | CLAY-based; film-grade characters and clean topology. |
| **CSM** | 1.4c | Explicitly targets simulation-ready assets and world models. |
| **[Polycam](https://poly.cam/) / [KIRI](https://www.kiriengine.app/)** | 1.4a | Consumer scanning — the real-world → 3D on-ramp. |

### Generative — worlds, video & scenes (route 1.1 / 1.4b)

| Company / Product | Route | What they actually do |
| :--- | :--- | :--- |
| **[Google DeepMind](https://deepmind.google/)** | **1.1 + 2.1** | **The only full-stack player**: Genie 3 (interactive worlds) + Veo (video) + SIMA (agents inside generated worlds) + Gemini Robotics (VLA). The generate→act loop already closes here. |
| **[World Labs](https://www.worldlabs.ai/)** | 1.4b (scene) | Marble: image → **persistent, walkable 3D Gaussian world**, exportable to splat/mesh. Founded by Fei-Fei Li, Justin Johnson, Christoph Lassner, Ben Mildenhall (NeRF). Stores the world as *state*, which sidesteps route 1.1's consistency problem. |
| **[NVIDIA Cosmos](https://github.com/nvidia-cosmos)** | 1.1 (serving 2.x) | Predict (future video) / Transfer (sim→photoreal, i.e. domain randomization on steroids) / Reason (physics-aware VLM for curation & eval). Sold as an entire **data engine** with Omniverse + Isaac + GR00T. |
| **[Decart](https://www.decart.ai/)** | 1.1 | Oasis, Mirage — latency-first real-time interactive video. |
| **[Odyssey](https://odyssey.ml/)** | 1.1 | Streaming interactive video world models. |
| **[Luma AI](https://lumalabs.ai/)** | 1.1 | NeRF-scanning heritage → Dream Machine / Ray. One of the few with both geometry and video know-how. |
| **[Kling](https://klingai.com/)** (Kuaishou) | 1.1 | 3D causal VAE + DiT; differentiates on controllability (keyframes, camera moves, motion brush) — weak-form action conditioning. |
| **[Runway](https://runwayml.com/)** | 1.1 | Gen-4; published a "General World Models" agenda early. |
| **[Wayve](https://wayve.ai/)** | 1.1 + 2.1 | GAIA-1/2 driving world models + end-to-end driving. The most complete vertical loop. |
| **[Waabi](https://waabi.ai/)** | 1.1 / 1.5 | Waabi World neural simulator; verification-first. |
| **[Niantic Spatial](https://nianticspatial.com/)** | 1.4a | Large Geographic Model — planet-scale geometry from crowdsourced scans. |
| **Open video bases** | 1.1 | [Wan](https://github.com/Wan-Video) · [CogVideoX](https://github.com/THUDM/CogVideo) · [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) — what most academic world-model work is actually built on. |

### Action-centric — robot foundation models (route 2.1 / 2.2)

| Company / Product | Route | What they actually do |
| :--- | :--- | :--- |
| **[Physical Intelligence (π)](https://www.pi.website/)** | 2.1 | The standard-setter: π0 → π0-FAST → π0.5. VLM backbone + flow-matching action expert; heavily open ([openpi](https://github.com/Physical-Intelligence/openpi)). |
| **[NVIDIA GR00T](https://github.com/NVIDIA/Isaac-GR00T)** | 2.1 | Open humanoid foundation model; the public reference for the dual-system design. |
| **[Generalist AI](https://generalistai.com/)** | 2.1 | End-to-end, hardware-agnostic policies; bets on real-data scale and dexterity. |
| **[Skild AI](https://www.skild.ai/)** | 2.1 | "Skild Brain", omni-bodied — one brain across wheeled, quadruped, humanoid, arms. |
| **[Figure](https://www.figure.ai/)** | 2.1 | Helix: S2 semantics (~7–9Hz) + S1 control (200Hz), full-stack humanoid. |
| **[1X](https://www.1x.tech/)** | 2.1 | NEO home humanoid; teleop-driven data flywheel. |
| **[Toyota Research Institute](https://www.tri.global/) + Boston Dynamics** | 2.1 | Large Behavior Models — the industrial validation of diffusion policies. |
| **[Dyna Robotics](https://www.dyna.co/)** | 2.1 | Contrarian: single-task, industrial-grade reliability first. |
| **Covariant** | 2.2 | RFM-1 was the earliest industrial world-model+action system; team absorbed by Amazon in 2024 — the cautionary tale of this school. |
| **China cohort** | 2.1 | [AgiBot](https://www.zhiyuan-robot.com/) (GO-1, open AgiBot World) · [Galbot](https://www.galbot.com/) (GraspVLA, synthetic-data-only grasping) · Galaxea · X Square (WALL-OSS) · Noematrix · Spirit AI |

### Cross-school & infrastructure

| Company / Product | Route | What they actually do |
| :--- | :--- | :--- |
| **Genesis AI** | 1.5 + 2.1 | Owns a GPU-parallel, differentiable, multi-physics engine; the play is **synthetic data at scale → generalist policy**, bypassing the real-data bottleneck. Not the same thing as the open-source Genesis engine, and unrelated to Google's Genie. |
| **[Lightwheel](https://lightwheel.ai/)** | 1.4c | Sim-ready assets and synthetic data as a product — the closest commercial analogue to route 1.4c. |
| **[Applied Intuition](https://www.appliedintuition.com/)** | 3 | Simulation for AV and defense, turned into a business. |
| **[PhysicsX](https://www.physicsx.ai/)** | 1.5 | **Industrial, not embodied.** Neural surrogates for FEA/CFD — optimizes for numerical accuracy and conservation, a different evaluation regime entirely. Peers: Neural Concept, Emmi AI, Ansys SimAI, PasteurLabs. |

### The one-line map

> **Assets** (Meshy / Tripo / Seed3D / Hunyuan3D) → **Worlds & scenes** (World Labs / Genie / Cosmos) → **Simulation & data** (Isaac / Genesis / MuJoCo / Lightwheel) → **Brains** (π / GR00T / Gemini Robotics / Skild / Figure).
>
> Route **1.4c** — turning a good-looking mesh into a simulatable one — is the thinnest section of this list and the least crowded part of the map.

## Open Questions

1. **Pixel prediction vs representation prediction.** Generating pixels spends capacity on render-irrelevant detail (LeCun's objection), but pixels are the only self-supervised signal that scales to internet video.
2. **Implicit neural dynamics vs explicit geometry + physics engines.** The former generalizes but conserves nothing; the latter is exact but cannot author the long tail. Route 1.4c is the attempt to have both.
3. **World model at training time or at inference time?** Train a policy inside imagination (cheap), or plan online with MPC (self-correcting)?
4. **Who owns evaluation?** If a world model is also the evaluator, what stops the two from being wrong in the same direction?

## Contributing

PRs welcome. Please keep entries to **one line**, state *what is architecturally new*, not what benchmark it won, and add the paper to **both** `README.md` and `README.zh-CN.md` so the two stay in sync.

## License

[CC0](LICENSE) — to the extent possible under law, the authors have waived all copyright to this list.
