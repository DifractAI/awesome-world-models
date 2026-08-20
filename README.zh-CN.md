<div align="center">

# Awesome World Models for Embodied AI [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

**世界模型的两大流派——一派通过「生成世界」来理解世界，一派通过「在世界中行动」来理解世界——以及各自的论文、架构、仿真器与公司。**

[English](README.md) · [中文](README.zh-CN.md)

</div>

---

## 这份列表为什么这样组织

所有世界模型回答的是同一个问题——学一个可以被条件化、可以外推的动力学：

$$p(s_{t+1} \mid s_{\le t},\, a_{\le t})$$

分歧从来不在这个式子，而在于 **`s` 取什么** 以及 **监督信号从哪来**。就这一个分叉，长出了两个数据来源不同、主干不同、失效模式不同、公司也不同的流派。

| | **生成派**（generate the world） | **行动派**（act in the world） |
| :--- | :--- | :--- |
| 判据 | 能预测/渲染未来 = 理解了世界 | 能做对动作 = 理解了世界 |
| 状态 `s` | 像素 / 视频 latent / 3D 几何 | 任务相关的紧凑表征，常常是隐式的 |
| 主监督 | 重建、去噪、next-token（自监督） | 行为克隆的动作监督、RL 回报 |
| 数据 | 互联网视频、多视角照片——海量，**无动作标签** | 遥操作 / 真机 / 仿真——稀缺，**有动作标签** |
| 主干 | DiT · causal Transformer · RSSM | VLM + action expert |
| 产物 | 仿真器、数据引擎、评估器 | 策略 `π` |
| 软肋 | 漂移、无物理保证、算力贵 | 泛化窄、数据贵、缺长程想象 |

全文使用统一的**路线编号**（1.1–1.5、2.1–2.3），中英文版本一一对应。

## 目录

- [1. 生成派世界模型](#1-生成派世界模型)
  - [0. 基础骨架](#0-基础骨架--两派共用的零件)
  - [1.1 视频 · 隐空间世界模型](#11-视频--隐空间世界模型)
  - [1.2 自回归 token 世界模型](#12-自回归-token-世界模型)
  - [1.3 紧凑隐状态世界模型](#13-紧凑隐状态世界模型)
  - [1.4a 前馈三维重建](#14a-前馈三维重建)
  - [1.4b 三维 · 场景生成](#14b-三维--场景生成)
  - [1.4c 从「能看」到「能交互」](#14c-从能看到能交互)
  - [1.5 神经 + 物理混合](#15-神经--物理混合)
- [2. 行动派世界模型](#2-行动派世界模型)
  - [2.1 VLA](#21-vla--视觉-语言-动作模型)
  - [2.2 WAM](#22-wam--世界模型与策略耦合)
  - [2.3 分层](#23-分层--llm-规划器--底层技能)
- [3. 仿真器与基准](#3-仿真器与基准)
- [4. 综述](#4-综述)
- [5. 公司与产品](#5-公司与产品)
- [尚未定论的问题](#尚未定论的问题)
- [贡献](#贡献)

---

## 1. 生成派世界模型

> *赌注：动力学可以从「预测下一帧 / 下一视角」里自监督地学出来，因此能吃互联网规模的无标注视频。*

### 0. 基础骨架 —— 两派共用的零件

本列表里所有模型都是用这些零件拼出来的。已经熟悉可以直接跳过。

- **[ViT](https://arxiv.org/abs/2010.11929)** `2020` · *Google* — 把图像切 patch 喂 Transformer。今天所有视频/3D 世界模型的空间 tokenize 方式都源于此。
- **[DDPM](https://arxiv.org/abs/2006.11239)** `2020` · *UC Berkeley* — 去噪生成目标，几乎所有生成式世界模型的训练目标源头。
- **[Latent Diffusion](https://arxiv.org/abs/2112.10752)** `2021` · *LMU Munich* — 在 VAE latent 而非像素上扩散——视频世界模型算得起的前提。
- **[DiT](https://arxiv.org/abs/2212.09748)** `2022` · *Meta / NYU* — 用 Transformer 替掉 U-Net，给出干净的 scaling law。视频与 3D 世界模型的默认主干。
- **[Flow Matching](https://arxiv.org/abs/2210.02747)** `2022` · *Meta* — 直线概率路径，现已取代 DDPM 成为视频生成与动作专家的标准训练目标。
- **[VQ-VAE](https://arxiv.org/abs/1711.00937)** `2017` · *DeepMind* — 离散码本。视觉 tokenizer 和 latent action model 共用的机制。
- **[VQGAN](https://arxiv.org/abs/2012.09841)** `2020` · *Heidelberg* — 带对抗损失的视觉 tokenizer，让自回归图像/视频建模变得可用。
- **[NeRF](https://arxiv.org/abs/2003.08934)** `2020` · *UC Berkeley* — 用 MLP 表示隐式体积场景，开启现代三维重建浪潮。
- **[3DGS](https://arxiv.org/abs/2308.04079)** `2023` · *Inria* — 显式、实时、可微的渲染基元，今天 3D 世界生成的主流载体。

### 1.1 视频 · 隐空间世界模型

把世界压进时空 latent，再以动作为条件生成。**Causal 3D VAE → DiT → flow matching** 是几乎通用的配方。「视频模型」与「世界模型」的分界线只有一条：*有没有动作条件*。

- **[VDM](https://arxiv.org/abs/2204.03458)** `2022` · *Google* — 第一个在时间轴上做扩散的模型，提出被后来者继承的 3D U-Net 分解。
- **[Sora](https://openai.com/index/video-generation-models-as-world-simulators/)** · *OpenAI* — 把「视频生成 = 世界模拟器」变成主流叙事的技术报告；时空 patch + 大规模 DiT。
- **[Genie](https://arxiv.org/abs/2402.15391)** `2024` · *Google DeepMind* — **这条路线最关键的一招**：从无标注视频里无监督学出 latent action，让互联网视频第一次能当交互数据用。
- **[Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)** · *Google DeepMind* — 720p/24fps 实时可交互世界模型，分钟级环境记忆。无论文，仅博客。
- **[Cosmos](https://arxiv.org/abs/2501.03575)** `2025` · *NVIDIA* — 开源世界基础模型（扩散 + 自回归）与配套 tokenizer，明确定位为机器人的数据引擎。
- **[Cosmos-Reason](https://arxiv.org/abs/2503.15558)** `2025` · *NVIDIA* — Cosmos 栈里的推理 VLM：物理常识、数据筛选与策略评估。
- **[GameNGen](https://arxiv.org/abs/2408.14837)** `2024` · *Google* — 用按键条件扩散模型以 20FPS 模拟 DOOM。可玩世界模型的存在性证明。
- **[Oasis](https://oasis-model.github.io/)** · *Decart / Etched* — 开源实时可交互 Minecraft 世界模型，延迟优先路线的参照系。
- **[The Matrix](https://arxiv.org/abs/2412.03568)** `2024` · *Alibaba et al.* — 无限时长流式生成 + 实时移动控制。
- **[Matrix-Game](https://arxiv.org/abs/2506.18701)** `2025` · *Skywork AI* — 面向游戏环境的开源交互式世界基础模型，附动作可控性基准。
- **[NWM](https://arxiv.org/abs/2412.03572)** `2024` · *Meta / NYU* — 在导航轨迹上做条件扩散 Transformer，通过想象未来并打分来规划。
- **[Diffusion Forcing](https://arxiv.org/abs/2407.01392)** `2024` · *MIT* — 每个 token 独立噪声级，统一自回归与扩散——因果、变长 rollout 的标准手法。
- **[Self Forcing](https://arxiv.org/abs/2506.08009)** `2025` · *Adobe / UT Austin* — 训练时喂自己的 rollout 以消除 exposure bias，实现实时流式视频生成。
- **[CausVid](https://arxiv.org/abs/2412.07772)** `2024` · *MIT / Adobe* — 把双向视频扩散模型蒸馏成快速因果模型，用于交互式生成。
- **[LAPA](https://arxiv.org/abs/2410.11758)** `2024` · *KAIST / Microsoft* — 从无动作标签的人类视频学离散 latent action，再微调到真实机器人动作。
- **[LAPO](https://arxiv.org/abs/2312.10812)** `2023` · *Oxford / UCL* — 纯从观测里恢复潜在动作空间，这一想法最干净的形式化。
- **[Moto](https://arxiv.org/abs/2412.04445)** `2024` · *HKU / ByteDance* — 用 latent motion token 作为视频预训练与机器人控制之间的桥梁语言。
- **[GAIA-1](https://arxiv.org/abs/2309.17080)** `2023` · *Wayve* — 以视频、文本、自车动作为条件的驾驶世界模型，第一个可信的工业级世界模型。
- **[GAIA-2](https://arxiv.org/abs/2503.20523)** `2025` · *Wayve* — 多相机、结构可控的后继版本，面向场景生成与安全测试。
- **[Vista](https://arxiv.org/abs/2405.17398)** `2024` · *OpenDriveLab* — 开源高保真驾驶世界模型，支持多模态动作控制。
- **[HunyuanWorld](https://arxiv.org/abs/2507.21809)** `2025` · *Tencent* — 开源全景图 → mesh 流水线，产出可漫游、语义分层的 3D 世界。
- **[Wan](https://arxiv.org/abs/2503.20314)** `2025` · *Alibaba* — 最强的完全开源视频基础模型族，事实上的世界模型学术底座。
- **[CogVideoX](https://arxiv.org/abs/2408.06072)** `2024` · *Zhipu / Tsinghua* — 开源 DiT 视频模型，3D 因果 VAE + expert AdaLN。

### 1.2 自回归 token 世界模型

把帧量化成离散 token 做 next-token 预测。结构上与 LLM 完全同构，因此动作 token 可以混进同一序列（见 [WAM](#22-wam)）。

- **[IRIS](https://arxiv.org/abs/2209.00588)** `2022` · *Geneva* — 离散自编码器 + 因果 Transformer 世界模型，智能体完全在想象中训练。
- **[MAGVIT-v2](https://arxiv.org/abs/2310.05737)** `2023` · *Google* — 无查表量化让 LLM 式视频 tokenizer 在质量上反超扩散。
- **[VideoPoet](https://arxiv.org/abs/2312.14125)** `2023` · *Google* — 单一 decoder-only LLM 处理交错的图像/视频/音频 token。
- **[Emu3](https://arxiv.org/abs/2409.18869)** `2024` · *BAAI* — 文本、图像、视频统一到一个词表里做纯 next-token 预测。
- **[LWM](https://arxiv.org/abs/2402.08268)** `2024` · *UC Berkeley* — 百万 token 的视频+文本上下文，把长时世界记忆当作上下文长度问题解。

### 1.3 紧凑隐状态世界模型

不渲染像素，只维护低维隐状态，**在想象里训练策略**。model-based RL 的正统血脉，外加 LeCun 的 JEPA 反提案。

- **[World Models (Ha & Schmidhuber)](https://arxiv.org/abs/1809.01999)** `2018` · *Google Brain* — 给这个领域命名的论文：VAE + RNN 动力学 + 在梦里训练的小控制器。
- **[PlaNet](https://arxiv.org/abs/1811.04551)** `2018` · *Google* — 提出 RSSM（确定 + 随机隐状态）与隐空间 MPC。
- **[Dreamer](https://arxiv.org/abs/1912.01603)** `2019` · *Google / Toronto* — 完全在想象的隐空间 rollout 上训练 actor-critic。
- **[DreamerV2](https://arxiv.org/abs/2010.02193)** `2020` · *Google* — 分类型隐变量让世界模型在 Atari 上达到人类水平。
- **[DreamerV3](https://arxiv.org/abs/2301.04104)** `2023` · *DeepMind* — 一套超参通吃 150+ 任务，从零在 Minecraft 里挖到钻石。
- **[TD-MPC](https://arxiv.org/abs/2203.04955)** `2022` · *UC San Diego* — 面向任务的隐空间动力学 + 在线规划，不重建观测。
- **[TD-MPC2](https://arxiv.org/abs/2310.16828)** `2023` · *UC San Diego* — 无解码器世界模型，单一智能体覆盖 100+ 连续控制任务。
- **[MuZero](https://arxiv.org/abs/1911.08265)** `2019` · *DeepMind* — 只学与价值和策略相关的东西，最纯粹的「隐式世界模型」。
- **[I-JEPA](https://arxiv.org/abs/2301.08243)** `2023` · *Meta* — 在表征空间而非像素空间预测。LeCun 对生成式世界模型的反提案。
- **[V-JEPA](https://arxiv.org/abs/2404.08471)** `2024` · *Meta* — 视频版本：掩码特征预测 + EMA 目标编码器。
- **[V-JEPA 2 / V-JEPA 2-AC](https://arxiv.org/abs/2506.09985)** `2025` · *Meta* — 互联网规模视频预训练 + 小型动作条件头，用 MPC 实现零样本机器人规划。
- **[DayDreamer](https://arxiv.org/abs/2206.14176)** `2022` · *UC Berkeley* — 直接在真机上训 Dreamer，验证世界模型在真实采样预算下可行。

### 1.4a 前馈三维重建

真实世界 → 几何，一次前馈完成。重建从「优化」变成了「推理」。

- **[DUSt3R](https://arxiv.org/abs/2312.14132)** `2023` · *Naver Labs* — 直接从图像对回归 pointmap，不要位姿、不要 SfM 流水线。
- **[MASt3R](https://arxiv.org/abs/2406.09756)** `2024` · *Naver Labs* — 加上稠密匹配，把 DUSt3R 变成完整的定位 + 重建栈。
- **[VGGT](https://arxiv.org/abs/2503.11651)** `2025` · *Oxford / Meta* — 一次前馈同时预测位姿、深度、pointmap 与轨迹的单一 Transformer。
- **[LRM](https://arxiv.org/abs/2311.04400)** `2023` · *Adobe / ANU* — 在 triplane 上放大 Transformer，5 秒单图出 3D——把重建从优化变成推理。
- **[GS-LRM](https://arxiv.org/abs/2404.19702)** `2024` · *Adobe* — 从稀疏视角前馈预测 3D 高斯。

### 1.4b 三维 · 场景生成

无中生有造几何。这条线已从 SDS 蒸馏（慢）转向**原生 3D latent 生成**（3D VAE + latent DiT / rectified flow）。

- **[DreamFusion](https://arxiv.org/abs/2209.14988)** `2022` · *Google* — 提出 SDS，开启基于优化的 text-to-3D 时代。
- **[Zero-1-to-3](https://arxiv.org/abs/2303.11328)** `2023` · *Columbia* — 相机条件的新视角合成，把 2D 扩散先验桥接到 3D。
- **[MVDream](https://arxiv.org/abs/2308.16512)** `2023` · *ByteDance* — 先生成一致的多视角集合再抬升，解决 Janus 多面问题。
- **[CAT3D](https://arxiv.org/abs/2405.10314)** `2024` · *Google* — 多视角扩散 + 快速三维拟合，从小时级降到分钟级。
- **[CLAY](https://arxiv.org/abs/2406.13897)** `2024` · *Deemos / ShanghaiTech* — 在神经场上做大规模原生 3D latent 扩散，Rodin 背后的主干。
- **[TRELLIS](https://arxiv.org/abs/2412.01506)** `2024` · *Microsoft / Tsinghua* — 结构化 latent（SLAT）可从同一模型解码出 mesh、3DGS 或辐射场。
- **[Hunyuan3D 2.0](https://arxiv.org/abs/2501.12202)** `2025` · *Tencent* — 开源两阶段「先几何后贴图」生成，部署最广的开源 3D 资产模型。
- **[TripoSR](https://arxiv.org/abs/2403.02151)** `2024` · *Tripo / Stability* — 开源 LRM 式重建，消费级 GPU 上亚秒出结果。
- **[TripoSG](https://arxiv.org/abs/2502.06608)** `2025` · *Tripo / VAST* — 大规模 rectified flow 形状生成，常用的开源参考实现。
- **[MIDI](https://arxiv.org/abs/2412.03558)** `2024` · *VAST* — 把场景生成为**可分离的物体实例**——让场景可编辑、可仿真的关键一步。

### 1.4c 从「能看」到「能交互」

**整个领域真正的瓶颈。** 好看的 mesh 不是世界模型。要能仿真，还需要部件、关节、质量、摩擦、碰撞体，以及 URDF/MJCF/USD 导出。论文最少，杠杆最高。

- **[Articulate-Anything](https://arxiv.org/abs/2410.13882)** `2024` · *Michigan / NVIDIA* — 让 VLM 写出铰接程序，把图像/视频变成带关节的 URDF 物体。
- **[URDFormer](https://arxiv.org/abs/2405.11656)** `2024` · *UW* — 直接从单张真实照片预测运动学结构与 URDF 参数。
- **[Real2Code](https://arxiv.org/abs/2406.08474)** `2024` · *Columbia* — 把铰接结构表示为生成的代码，得到精确关节语义而非一堆 mesh。
- **[PhysGen](https://arxiv.org/abs/2409.18964)** `2024` · *UIUC* — 从图像推断物理属性，用真实刚体仿真器驱动生成。
- **[EmbodiedGen](https://github.com/HorizonRobotics/EmbodiedGen)** · *Horizon Robotics* — 生成式 3D 世界引擎，输出物理合理、**可直接仿真**的资产与场景（URDF/MJCF/USD）。
- **[EmbodiedGen (paper)](https://arxiv.org/abs/2506.10600)** `2025` · *Horizon Robotics* — 论文版：完整的资产到仿真工具链，弥合 3D 生成与物理引擎之间的断层。
- **[RialTo](https://arxiv.org/abs/2403.03949)** `2024` · *MIT* — 扫描真实场景 → 变成仿真器 → 在里面 RL → 部署回真机，real2sim2real 的范式化流程。
- **[Infinigen](https://arxiv.org/abs/2306.09310)** `2023` · *Princeton* — 完全程序化、完全带标注的世界，合成数据的非神经解法。

### 1.5 神经 + 物理混合

把物理结构写进学习目标：图网络粒子、神经 PDE 算子、可微仿真。能外推、可验证，但吃不动视频数据。

- **[GNS](https://arxiv.org/abs/2002.09405)** `2020` · *DeepMind* — 图网络粒子模拟器，能在训练分布之外大幅泛化。
- **[DPI-Net](https://arxiv.org/abs/1810.01566)** `2018` · *MIT* — 学习到的粒子动力学直接用于基于模型的操作控制。
- **[FNO](https://arxiv.org/abs/2010.08895)** `2020` · *Caltech* — 学习偏微分方程族的解算子，工业代理模型的核心。
- **[PINN](https://arxiv.org/abs/1711.10561)** `2017` · *Brown* — 把 PDE 残差写进损失，用物理当正则。
- **[DiffTaichi](https://arxiv.org/abs/1910.00935)** `2019` · *MIT* — 把可微物理做成编程语言，梯度可以穿过接触与形变。
- **[Genesis](https://github.com/Genesis-Embodied-AI/Genesis)** · *Genesis-Embodied-AI* — GPU 并行、可微、多物理场引擎，具身研究里增长最快的开源仿真器。

---

## 2. 行动派世界模型

> *赌注：不需要把世界画出来。世界模型可以只是策略的副产品，甚至隐式存在于权重里。*

### 2.1 VLA —— 视觉-语言-动作模型

这些模型形状完全一样：$\pi(a_{t:t+H} \mid o_t, \ell)$，预测一个**动作块**。真正的分歧只在动作头怎么建模——离散分箱、压缩 token、扩散，还是流匹配。

- **[RT-1](https://arxiv.org/abs/2212.06817)** `2022` · *Google* — 动作 token 化 + Transformer，13 万条真机数据。之后所有工作的模板。
- **[RT-2](https://arxiv.org/abs/2307.15818)** `2023` · *Google DeepMind* — 把动作当成 VLM 里的文本 token，网页语义迁移到操作。
- **[Open X-Embodiment / RT-X](https://arxiv.org/abs/2310.08864)** `2023` · *Consortium* — 22 种本体、百万级轨迹，证明了跨本体的正迁移。
- **[Octo](https://arxiv.org/abs/2405.12213)** `2024` · *UC Berkeley* — 开源 Transformer 策略，扩散动作头 + 可插拔的观测/动作接口。
- **[OpenVLA](https://arxiv.org/abs/2406.09246)** `2024` · *Stanford et al.* — 7B 开源 VLA（SigLIP+DINOv2+Llama-2），离散动作分箱，学术界标准基线。
- **[OpenVLA-OFT](https://arxiv.org/abs/2502.19645)** `2025` · *Stanford* — 系统研究 VLA 微调里真正重要的因素：并行解码、动作分块、连续动作头。
- **[Diffusion Policy](https://arxiv.org/abs/2303.04137)** `2023` · *Columbia / TRI* — 对动作序列的**分布**建模，解决回归会做平均的多模态问题。
- **[ACT / ALOHA](https://arxiv.org/abs/2304.13705)** `2023` · *Stanford* — 动作分块 + 两万美元双臂平台，「action chunk」由此普及。
- **[π0](https://arxiv.org/abs/2410.24164)** `2024` · *Physical Intelligence* — PaliGemma-3B 主干 + 约 300M **flow matching 动作专家**，50Hz。现代 VLA 的主流架构。
- **[π0-FAST](https://arxiv.org/abs/2501.09747)** `2025` · *Physical Intelligence* — 用 DCT + BPE 压缩动作序列，让自回归 VLA 也能上高频控制。
- **[π0.5](https://arxiv.org/abs/2504.16054)** `2025` · *Physical Intelligence* — 分层推理（高层子任务 → 低层动作）+ 异构数据 co-training，能打扫没见过的真实民居。
- **[RDT-1B](https://arxiv.org/abs/2410.07864)** `2024` · *Tsinghua* — 1.2B 扩散 Transformer，为异构双臂机器人设计统一动作空间。
- **[GR00T N1](https://arxiv.org/abs/2503.14734)** `2025` · *NVIDIA* — **双系统**设计的开源参考实现：VLM 当 System 2 + DiT 动作头当 System 1。
- **[Gemini Robotics](https://arxiv.org/abs/2503.20020)** `2025` · *Google DeepMind* — 基于 Gemini 的 VLA，外加独立的具身推理（ER）模型负责空间定位与规划。
- **[SmolVLA](https://arxiv.org/abs/2506.01844)** `2025` · *Hugging Face* — 小体量、社区数据训练的 VLA，消费级硬件可跑，支持异步推理。
- **[GO-1 / AgiBot World](https://arxiv.org/abs/2503.06669)** `2025` · *AgiBot* — 大规模开源真机数据集 + 用人类与机器人视频训练的 latent planner VLA。
- **[LBM](https://arxiv.org/abs/2507.05331)** `2025` · *Toyota Research Institute* — 严谨验证多任务预训练优于单任务策略，扩散策略路线的工业化背书。
- **[ECoT](https://arxiv.org/abs/2407.08693)** `2024` · *UC Berkeley* — 让策略在动作之前先吐出子任务、物体定位与夹爪位姿。
- **[PerAct](https://arxiv.org/abs/2209.05451)** `2022` · *UW / NVIDIA* — 体素化 3D 观测 + Perceiver，动作表示为 3D 关键帧而非像素。
- **[Act3D](https://arxiv.org/abs/2306.17817)** `2023` · *CMU* — 由粗到细的 3D 特征场，提升操作的样本效率。
- **[3D Diffuser Actor](https://arxiv.org/abs/2402.10885)** `2024` · *CMU* — 把扩散策略与显式 3D 场景 token 结合。
- **[RVT-2](https://arxiv.org/abs/2406.08545)** `2024` · *NVIDIA* — 用多视图重渲染替代体素，快速、精确、少样本。
- **[SpatialVLA](https://arxiv.org/abs/2501.15830)** `2025` · *Shanghai AI Lab* — Ego3D 位置编码 + 自适应动作网格，实现跨本体的空间迁移。
- **[UMI](https://arxiv.org/abs/2402.10329)** `2024` · *Stanford* — 手持夹爪野外采数，最便宜的可规模化带动作标签数据来源。
- **[HIL-SERL](https://arxiv.org/abs/2410.21845)** `2024` · *UC Berkeley* — 带人类纠正的真机 RL，在高接触任务上逼近 100% 成功率。

### 2.2 WAM —— 世界模型与策略耦合

既预测未来，又输出动作。动作标签很贵，而预测下一帧免费——于是把预测当预训练、当子目标生成器、或当规划器。

- **[UniPi](https://arxiv.org/abs/2302.00111)** `2023` · *Google / MIT* — 先生成未来视频，再用逆动力学模型反解成动作。
- **[UniSim](https://arxiv.org/abs/2310.06114)** `2023` · *UC Berkeley / Google* — 动作条件的真实世界视频模拟器，训出的策略可零样本迁移。
- **[GR-1](https://arxiv.org/abs/2312.13139)** `2023` · *ByteDance* — 先做视频预测预训练，再接轻量动作头——把预测当免费监督。
- **[GR-2](https://arxiv.org/abs/2410.06158)** `2024` · *ByteDance* — 把同一配方放大到网页视频与多任务泛化。
- **[3D-VLA](https://arxiv.org/abs/2403.09631)** `2024` · *UMass* — 在动作之前先想象未来点云与图像的生成式世界模型。
- **[WorldVLA](https://arxiv.org/abs/2506.21539)** `2025` · *Alibaba* — 把图像 token 与动作 token 放进同一个自回归序列，互相预测。
- **[iVideoGPT](https://arxiv.org/abs/2405.15223)** `2024` · *Tsinghua* — 可扩展的交互式自回归世界模型，用于基于模型的 RL 与视觉控制。
- **[SuSIE](https://arxiv.org/abs/2310.10639)** `2023` · *UC Berkeley* — 用图像编辑扩散生成视觉子目标，底层策略负责追踪。
- **[AVDC](https://arxiv.org/abs/2310.08576)** `2023` · *MIT* — 用稠密对应从生成视频里抽动作，完全不需要动作标签。
- **[Genie Envisioner](https://arxiv.org/abs/2508.05635)** `2025` · *AgiBot* — 统一平台：同一个视频世界模型同时充当策略、仿真器与评估器。

### 2.3 分层 —— LLM 规划器 + 底层技能

推理留在冻结的 LLM/VLM 里，闭环交给经典规划器或技能库。可解释、零样本可组合；受限于技能库与低控制频率。

- **[SayCan](https://arxiv.org/abs/2204.01691)** `2022` · *Google* — LLM 提议，学到的价值函数判断可行性。
- **[Code as Policies](https://arxiv.org/abs/2209.07753)** `2022` · *Google* — LLM 直接写可执行的策略代码，调用感知与控制 API。
- **[VoxPoser](https://arxiv.org/abs/2307.05973)** `2023` · *Stanford* — LLM+VLM 组合出 3D 可供性与约束图，剩下交给运动规划器。零机器人数据。
- **[ReKep](https://arxiv.org/abs/2409.01652)** `2024` · *Stanford* — 把任务表达为关系型关键点约束，将操作转化为可求解的优化问题。

---

### 3. 仿真器与基准

两派的接缝。生成出来的资产落在这里，策略从这里出生。

- **[Isaac Sim / Isaac Lab](https://github.com/isaac-sim/IsaacLab)** · *NVIDIA* — 事实上的工业标准：USD 场景、GPU 并行 RL、照片级渲染。
- **[MuJoCo / MJX](https://github.com/google-deepmind/mujoco)** · *Google DeepMind* — 接触动力学最可信的引擎，MJX 把它带到 GPU 规模的 RL。
- **[SAPIEN](https://arxiv.org/abs/2003.08515)** `2020` · *UC San Diego* — part 级铰接物体（PartNet-Mobility），铰接交互的标准测试场。
- **[ManiSkill3](https://arxiv.org/abs/2410.00425)** `2024` · *UC San Diego* — 仿真与渲染同时 GPU 并行，视觉 RL 可达数万 FPS。
- **[RoboCasa](https://arxiv.org/abs/2406.02523)** `2024` · *UT Austin / NVIDIA* — 用生成式资产填充的厨房环境，3D 生成与机器人学习之间的桥。
- **[BEHAVIOR-1K](https://arxiv.org/abs/2403.09227)** `2024` · *Stanford* — 1000 个真人票选的家务活动，带丰富物理状态（温度、潮湿、脏污）。
- **[Habitat 3.0](https://arxiv.org/abs/2310.13724)** `2023` · *Meta* — 面向导航与人机共处的高速照片级仿真。
- **[Orbit (→ Isaac Lab)](https://arxiv.org/abs/2301.04195)** `2023` · *ETH / NVIDIA* — Isaac Lab 任务/环境抽象背后的框架论文。

### 4. 综述

- **[World Models Survey](https://arxiv.org/abs/2411.14499)** `2024` — 同时覆盖生成派与决策派两条脉络的综述。
- **[Driving World Models Survey](https://arxiv.org/abs/2501.11260)** `2025` — 世界模型最成熟的垂直落地场景。
- **[VLA Survey](https://arxiv.org/abs/2405.14093)** `2024` — VLA 架构、数据与基准的分类梳理。

---

## 5. 公司与产品

钱和产品真正落在哪里。`路线` 对应上文编号。

### 生成派 —— 3D 资产（路线 1.4b → 1.4c）

| 公司 / 产品 | 路线 | 实际在做什么 |
| :--- | :--- | :--- |
| **[Meshy](https://www.meshy.ai/)** | 1.4b | 原生 3D latent 生成 + 多视角 PBR 贴图扩散。面向美术管线，不碰物理。 |
| **[Tripo AI](https://www.tripo3d.ai/)**（VAST） | 1.4b → 1.4c | 前馈重建（TripoSR）→ rectified flow 形状生成（TripoSG），正在推 part-level 与多实例场景。开源最狠的一家。 |
| **Seed3D**（字节 Seed） | 1.4b → 1.4c | 单图 → **可直接仿真**的资产。优化目标是水密性、真实尺度、碰撞体质量，而不只是好看。 |
| **[腾讯混元3D](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)** | 1.4b | 最大的开源「形状+贴图」模型族；HunyuanWorld 把它扩展到可漫游场景。 |
| **[Rodin](https://hyper3d.ai/)**（Deemos） | 1.4b | 基于 CLAY，影视级角色与干净拓扑。 |
| **CSM** | 1.4c | 明确瞄准 simulation-ready 资产与世界模型。 |
| **[Polycam](https://poly.cam/) / [KIRI](https://www.kiriengine.app/)** | 1.4a | 消费级扫描，「真实世界 → 3D」的入口。 |

### 生成派 —— 世界、视频与场景（路线 1.1 / 1.4b）

| 公司 / 产品 | 路线 | 实际在做什么 |
| :--- | :--- | :--- |
| **[Google DeepMind](https://deepmind.google/)** | **1.1 + 2.1** | **唯一全栈通吃的玩家**：Genie 3（可交互世界）+ Veo（视频）+ SIMA（在生成世界里行动的智能体）+ Gemini Robotics（VLA）。「生成 → 行动」的闭环已经在这里跑通。 |
| **[World Labs](https://www.worldlabs.ai/)** | 1.4b（场景） | Marble：单图 → **持久、可漫游的 3D 高斯世界**，可导出 splat/mesh。创始人为李飞飞、Justin Johnson、Christoph Lassner、Ben Mildenhall（NeRF 一作）。把世界存成**状态**，绕开了路线 1.1 的一致性难题。 |
| **[NVIDIA Cosmos](https://github.com/nvidia-cosmos)** | 1.1（服务于 2.x） | Predict（预测未来视频）/ Transfer（仿真图 → 照片级，本质是加强版域随机化）/ Reason（懂物理的 VLM，做数据筛选与评估）。配合 Omniverse + Isaac + GR00T，卖的是**整条数据引擎**。 |
| **[Decart](https://www.decart.ai/)** | 1.1 | Oasis、Mirage——延迟优先的实时可交互视频。 |
| **[Odyssey](https://odyssey.ml/)** | 1.1 | 流式可交互视频世界模型。 |
| **[Luma AI](https://lumalabs.ai/)** | 1.1 | NeRF 扫描起家 → Dream Machine / Ray。少数同时掌握几何与视频两套 know-how 的公司。 |
| **[可灵 Kling](https://klingai.com/)**（快手） | 1.1 | 3D causal VAE + DiT；差异化在可控性（首尾帧、运镜、运动笔刷）——本质是弱化版动作条件。 |
| **[Runway](https://runwayml.com/)** | 1.1 | Gen-4；很早发过 "General World Models" 纲领。 |
| **[Wayve](https://wayve.ai/)** | 1.1 + 2.1 | GAIA-1/2 驾驶世界模型 + 端到端自驾。垂直闭环做得最完整。 |
| **[Waabi](https://waabi.ai/)** | 1.1 / 1.5 | Waabi World 神经仿真器，可验证性优先。 |
| **[Niantic Spatial](https://nianticspatial.com/)** | 1.4a | Large Geographic Model——靠众包扫描做地球级几何。 |
| **开源视频底座** | 1.1 | [Wan](https://github.com/Wan-Video) · [CogVideoX](https://github.com/THUDM/CogVideo) · [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo)——学术界的世界模型工作基本都长在这上面。 |

### 行动派 —— 机器人基础模型（路线 2.1 / 2.2）

| 公司 / 产品 | 路线 | 实际在做什么 |
| :--- | :--- | :--- |
| **[Physical Intelligence (π)](https://www.pi.website/)** | 2.1 | 标准制定者：π0 → π0-FAST → π0.5。VLM 主干 + flow matching 动作专家，开源力度大（[openpi](https://github.com/Physical-Intelligence/openpi)）。 |
| **[NVIDIA GR00T](https://github.com/NVIDIA/Isaac-GR00T)** | 2.1 | 开源人形基础模型，双系统设计的公开参考实现。 |
| **[Generalist AI](https://generalistai.com/)** | 2.1 | 端到端、硬件无关的策略，押注真机数据规模与灵巧操作。 |
| **[Skild AI](https://www.skild.ai/)** | 2.1 | "Skild Brain"，omni-bodied——一个大脑跨轮式/四足/人形/机械臂。 |
| **[Figure](https://www.figure.ai/)** | 2.1 | Helix：S2 语义（~7–9Hz）+ S1 控制（200Hz），全栈自研人形。 |
| **[1X](https://www.1x.tech/)** | 2.1 | NEO 家用人形，遥操作驱动的数据飞轮。 |
| **[Toyota Research Institute](https://www.tri.global/) + 波士顿动力** | 2.1 | Large Behavior Model——扩散策略路线的工业化验证。 |
| **[Dyna Robotics](https://www.dyna.co/)** | 2.1 | 反共识：先把单任务做到工业级可靠性。 |
| **Covariant** | 2.2 | RFM-1 是最早的工业级「世界模型 + 动作」系统；团队 2024 年被亚马逊吸收，是这一派最重要的前车之鉴。 |
| **国内梯队** | 2.1 | [智元 AgiBot](https://www.zhiyuan-robot.com/)（GO-1、开源 AgiBot World）· [银河通用 Galbot](https://www.galbot.com/)（GraspVLA，纯合成数据训抓取）· 星海图 · 自变量（WALL-OSS）· 穹彻 · 千寻智能 |

### 跨派与基础设施

| 公司 / 产品 | 路线 | 实际在做什么 |
| :--- | :--- | :--- |
| **Genesis AI** | 1.5 + 2.1 | 手握 GPU 并行、可微、多物理场引擎；打法是**大规模合成数据 → 通用策略**，绕开真机数据瓶颈。注意它 ≠ 开源 Genesis 引擎，也和 Google 的 Genie 无关。 |
| **[光轮智能 Lightwheel](https://lightwheel.ai/)** | 1.4c | 把 sim-ready 资产与合成数据做成产品，商业上最接近路线 1.4c 的一家。 |
| **[Applied Intuition](https://www.appliedintuition.com/)** | 3 | 把自驾/国防仿真做成了生意。 |
| **[PhysicsX](https://www.physicsx.ai/)** | 1.5 | **工业向，不是具身。** 用神经网络做 FEA/CFD 代理模型，追求数值精度与守恒，评价体系完全不同。同赛道：Neural Concept、Emmi AI、Ansys SimAI、PasteurLabs。 |

### 一句话地图

> **资产层**（Meshy / Tripo / Seed3D / 混元3D）→ **世界与场景层**（World Labs / Genie / Cosmos）→ **仿真与数据层**（Isaac / Genesis / MuJoCo / 光轮）→ **大脑层**（π / GR00T / Gemini Robotics / Skild / Figure）。
>
> 路线 **1.4c**——把好看的 mesh 变成能仿真的资产——是本列表里论文最少的一节，也是这张地图上最空的一段。

## 尚未定论的问题

1. **像素预测 vs 表征预测。** 生成像素把算力花在与渲染无关的细节上（LeCun 的反对意见），但像素是唯一能规模化到互联网视频的自监督信号。
2. **隐式神经动力学 vs 显式几何 + 物理引擎。** 前者通用但不守恒，后者精确但造不出长尾世界。路线 1.4c 就是想两头都要。
3. **世界模型该在训练时用还是推理时用？** 在想象里训策略（省算力），还是用 MPC 在线规划（可纠错）？
4. **谁来评估？** 如果世界模型同时也是评估器，什么能阻止两者朝同一个方向错？

## 贡献

欢迎 PR。请把每条控制在**一行**，说明**架构上新在哪里**，而不是刷了什么榜；并且同时更新 `README.md` 和 `README.zh-CN.md`，保持两版同步。

## License

[CC0](LICENSE) —— 在法律允许范围内，作者放弃对本列表的一切版权。
