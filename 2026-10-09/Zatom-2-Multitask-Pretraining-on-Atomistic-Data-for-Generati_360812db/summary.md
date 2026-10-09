---
title: "Zatom-2-Multitask-Pretraining-on-Atomistic-Data-for-Generati"
source: https://arxiv.org/pdf/2610.11454v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:10:26"
field: "原子级/分子级生成建模"
keywords: ["原子级生成模型", "多任务预训练", "条件流匹配", "力条件", "跨域迁移", "蛋白质生成", "统一原子建模"]
innovations: ["统一 atom1+atom14 多域 Transformer 架构支持分子/材料/蛋白质生成与预测", "联合生成-预测（MLIP）多任务预训练提升跨域迁移能力", "力范数全局条件实现可控弛豫态生成"]
benchmarks: ["GEOM-Drugs", "QM9", "MP20", "OMol25-4M", "OMat24-1M", "SCOPe-2k"]
---

# 论文速读：Zatom-2: Multitask Pretraining on Atomistic Data for Generative Modeling across Domains

## 一句话总结
Zatom-2 是一个统一原子级生成模型，在约 500 万 OMol25（分子）和 OMat24（材料）结构上进行多任务预训练，结合条件流匹配（conditional flow matching）与力条件（force conditioning），支持分子/材料生成、结构预测和能量/力预测；其联合生成-预测预训练策略成功迁移至低数据蛋白质生成任务，将长度外推场景下的可设计性从 67.8% 提升至 74.8%。

## 研究问题与动机
1. **领域割裂问题**：现有原子级生成模型高度专业化，化学/材料与生物学方法互不通用，无法利用跨域共享的局部原子环境信息进行统一建模。
2. **缺乏统一预训练框架**：现有方法未同时利用大规模有机分子（小分子）和无机组晶（材料）数据用于通用预训练，限制了表示的泛化能力。
3. **数据多样性未被充分挖掘**：OMol25/OMat24 等大规模量子化学数据集包含丰富的构象多样性（高力/弛豫态混合），但现有生成模型未显式利用这些物理信息。
4. **迁移学习潜力待探索**：如何从数据丰富的化学/材料域学习有效表示，并迁移至数据稀缺的蛋白质生成任务，尚缺乏系统研究。

## 核心贡献（创新点）
1. **统一多域原子级架构**：提出 atom1 + atom14 混合 tokenization 方案，在单一 Transformer 架构下支持分子、周期性材料和全原子蛋白质的生成与预测，无需各模态独立扩散轨迹。
2. **联合生成-预测多任务预训练**：将生成（flow matching）、结构预测与辅助力/能量预测（MLIP）任务统一训练，验证了预测目标可塑造更优生成表示。
3. **力条件控制生成**：引入全局平均力范数作为条件信号（$s_F = \log_{10}(\frac{1}{L}\sum_\ell \|F_\ell\|_2)$），实现低/高力 regime 的可控采样，AUROC 达 0.76–1.00。
4. **跨域迁移至蛋白质生成**：在无蛋白质预训练数据的情况下，通过 OMol25/OMat24 预训练 + 仅 2,000 蛋白质域微调，长度外推（129–256 残基）可设计性从 67.8% 提升至 74.8%，超越 RFdiffusion3。

## 方法详解

### 3.1 多尺度 Transformer 主干
- **输入初始化器**：嵌入域标签、电荷/自旋、系统力标签、固定化学身份、相对 token 位置和键信息，与坐标及流时间嵌入融合，初始化 atom ($Q_L$) 和 token ($Q_I$) 表征。
- **U 形原子-令牌-原子结构**：Atom encoder（局部注意力，128 keys）→ Downcast（cross-attention 至 token）→ Token trunk（全局 attention）→ Upcast → Atom decoder。所有 Transformer 块使用 pair-biased attention 和 DiT-style adaptive normalization。
- **Task-specific readout heads**：Atom 特征预测归一化笛卡尔速度和分数坐标；Token 特征预测元素身份和序列类型 logits；Lattice 参数由系统内 token pooling 预测；可选 property 模块聚合为每原子残差能量并预测力向量。物理能量重构公式：$\widehat{E}_b = \sum_{\ell \in b} \varepsilon_{d,z_\ell} - n_b(\mu_d + s_d \widehat{e}_b)$。
- **Recycling 机制**：将 $\widehat{\mathbf{v}}_\theta$ 输出的干净坐标估计转化为 token 级 distogram（65-bin），更新 token-pair 条件以进行第二遍前向传播（默认 $R=2$ 次）。

### 3.2 条件流匹配（Conditional Flow Matching）
- 采用速度预测范式：$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$，目标速度 $\mathbf{v}_t = \frac{\mathbf{x}_1-\mathbf{x}_0}{\sigma_{\text{data}}}$，$\sigma_{\text{data}}=16$ Å。
- 训练目标：$\mathbb{E}[\|\widehat{\mathbf{v}}_\theta(\mathbf{x}_t,t) - \mathbf{v}_t\|_2^2]$。
- 推理时采用 EDM-style 采样：$t = \frac{\sigma_{\text{data}}}{\sigma_{\text{data}}+\sigma}$，$\mathbf{x}_t = t\mathbf{y}_\sigma$，保留标准流匹配训练目标的同时获得改进的样本质量。

### 3.3 力条件（Force Conditioning）
- 使用系统平均力范数对数作为全局描述符：$s_F = \log_{10}\left(\frac{1}{L}\sum_{\ell=1}^L \|F_\ell\|_2\right)$，作为残差嵌入注入初始化器，使模型区分相同组成但弛豫程度不同的构型。

### 3.4 多任务损失
$$\mathcal{L} = \lambda_{\text{FM}}\mathcal{L}_{\text{FM}} + \lambda_{\text{lDDT}}\mathcal{L}_{\text{lDDT}} + g_{1\mathring{A}}(t)\sum_{k\in\mathcal{K}}\lambda_k\mathcal{L}_k + m_E\lambda_E\mathcal{L}_E + m_F\lambda_F\mathcal{L}_F + \mathbf{1}_{\text{MLIP}}\lambda_{\text{eq}}\mathcal{L}_{\text{eq}}$$
- $\mathcal{L}_{\text{FM}}$：流匹配损失；$\mathcal{L}_{\text{lDDT}}$：平滑 lDDT 损失；$\mathcal{L}_E, \mathcal{L}_F$：Huber 损失。
- **时间门控机制** $g_\delta(t)$：仅在近端点（$t\to1$）启用结构预测等辅助任务，防止 target-force 泄漏。
- **Latent equivariance loss** $\mathcal{L}_{\text{eq}}$：两图层网络从旋转视角预测池化表征，增强力的预测精度。
- 关键超参：Adam(lr=1e-4), bf16 混合精度, $\sigma_{\text{data}}=16$Å, 训练分布为 Beta(1.9,1)+2% Uniform。

## 实验与结果

### 预训练数据
- OMol25-4M：3,986,754 个分子结构（83 种原子类型）
- OMat24-1M：1,009,850 个周期材料结构（88 种元素）

### 预训练消融（Table 1）
- Zatom-2 (VI)-final（Large, 270M）在所有评估指标上表现最佳：RDKit Valid 0.850，AFD 0.016，OMat24 Valid. 0.748，力条件 MAE 0.616，AUROC 0.817。
- 引入结构预测（III）基本保持生成质量；加入 OMat24+MLIP 监督（IV→V→VI）逐步提升力一致性和整体性能。

### 基准生成（Table 2）
- **GEOM-Drugs**：Validity 96.90%，Valid & PB-valid 88.63%，接近/优于基线（TABASCO 89.40%）。
- **QM9**：Validity 95.72%。
- **MP20（MetaSUN）**：Zatom-2 达到 **4.92%**，显著超越所有对比方法（ADiT 1.0，Crystalformer 3.1，Zatom-1-L 0.64）。

### 缩放研究（Figure 3）
- 全数据集 + 更大模型在 AFD、结构预测 match rate、力预测 MAE 四项指标上均一致优于小数据集/小模型配置，数据覆盖与模型容量提供互补增益。

### 蛋白质迁移（Table 3）
- **分布内（50–128 残基）**：Zatom-2 (V) Designability 89.09%，Diversity 453.7 clusters，Novelty 16.77%。
- **分布外长度外推（129–256 残基）**：Zatom-2 (V) Designability **74.80%**（vs. 无预训练 67.81%，+6.99pp），显著超越 RFdiffusion3（54.04%）及其他初始化。
- 力条件（FC）对迁移增益至关重要：无 FC 的预训练甚至劣于从头训练。

### 结构预测
- OMol25 结构预测 Mean RMSD ≈ 1.72 Å，AUROC（力分类）0.818；MP20 match rate 64.00%（接近 Crystalite 66.09%）。

## 相关工作脉络
1. **Zatom-1**（Morehead et al., 2026）：同源分子/材料生成模型，Zatom-2 在其基础上扩展了结构预测、力条件和能量/力预测多任务，并验证了向蛋白质的跨域迁移。
2. **RFdiffusion3**（Butcher et al., 2025）：蛋白质专用 all-atom 生成模型，使用 residue-level tokenization；Zatom-2 证明跨域原子级预训练可在低数据蛋白生成中取得更强迁移效果（尤其长度外推）。
3. **La-Proteina**（Geffner et al., 2026）：编码序列和侧链信息到 latent variables；Zatom-2 的不同在于统一生成-预测预训练， latent 空间展现出更清晰的氨基酸聚类分离。
4. **EMyx**（Williams et al., 2026）：快速全原子蛋白质生成模型；Zatom-2 在计算效率相近的同时支持更多任务（MLIP 预测、力条件）。
5. **All-atom Diffusion Transformers / TABASCO**：分别面向分子/材料单独建模；Zatom-2 统一分子+材料+蛋白的全原子生成范式，无需各模态独立扩散轨迹。
6. **MLIP 预训练相关工作**（REPA, CrystalREPA, UMA）：将 MLIP 表示对齐到生成模型；Zatom-2 的差异化在于将 MLIP 作为联合多任务之一直接在生成训练中优化，而非后验对齐。

## 局限性与未来方向
1. 预训练仅使用 OMol25/OMat24 的 4M/1M 子集（全集各 100M），未充分利用完整数据规模。
2. 力条件在当前形式下难以在 regime 内部精确匹配具体力值，未来需探索 per-atom 力向量条件。
3. 直接力预测 head 未强制能量-梯度一致性（energy-gradient consistency）和精确旋转等变性，限制了在高精度分子动力学中的应用。
4. 蛋白质实验仅在 SCOPe-2k（2,000 域）小规模数据集上验证，需扩展至大规模蛋白质数据集。
5. MLIP 预测精度远低于专业势函数（如 UMA、eSEN），非为替代 MLIP 而设计。

## 研究启发与可借鉴点
1. **联合生成-预测预训练范式**：将能量/力预测作为辅助任务与生成目标联合训练，可显著提升下游跨域迁移效果——该策略可推广至其他科学领域（如聚合物、表面科学）的生成模型设计。
2. **物理条件信号控制生成**：利用力范数等物理可观测量作为全局条件，实现生成样本弛豫态的可控调节；类似思路可用于控制晶体缺陷密度、分子张力状态等。
3. **跨域统一 tokenization（atom1 + atom14）**：用同一骨架处理不同尺度的原子系统，避免了多模型分别训练的成本，为"基础原子模型"提供了可行路径。
4. **Recycling 机制用于自条件**：从干净坐标估计派生 distogram 反馈到 token-pair 条件，提升了结构预测精度——该 self-conditioning 策略可直接迁移到其他 diffusion/flow 生成架构。
5. **时间门控训练策略**：通过 $g_\delta(t)$ 在近端点启用辅助任务，有效防止目标泄漏，这一技术对多任务扩散模型训练具有通用参考价值。

## 关键术语表
**Conditional Flow Matching**：一种生成建模框架，通过学习从噪声到数据的连续 ODE 轨迹来生成样本，以速度预测为目标函数。
**Atom1 / Atom14 Tokenization**：Atom1 为每个原子作为一个 token（分子/材料），Atom14 为每个残基占据 14 个坐标槽位（蛋白质），二者统一于同一架构。
**Force Conditioning**：将系统的平均力范数对数值作为全局条件信号输入模型，使其能区分并控制生成样本的弛豫程度。
**MLIP（Machine Learning Interatomic Potential）**：基于机器学习的原子间势能面模型，可同时预测体系和能量与原子受力。
**lDDT（local Distance Difference Test）**：一种无需全局对齐的蛋白质结构相似度评估指标，本文使用其平滑版本作为结构预测损失。
**MetaSUN**：Materials Generation 基准指标，衡量生成的材料同时满足有效性（valid）、独特性（unique）、新颖性（novel）且处于亚稳态（$0 < E_{\text{hull}} \leq 0.1$ eV/atom）的比例。
**Recycling**：将一次前向传播得到的干净坐标估计转化为 distogram，反馈至第二轮前向传播以更新 token-pair 条件，属于自条件（self-conditioning）机制。
**EDM-style Sampling**：Elucidating Diffusion Models 的采样调度策略，在流匹配框架下使用改进的噪声调度以获得更高样本质量。

## 可复现要素
- **数据集**：OMol25（CC BY 4.0）、OMat24（CC BY 4.0）、QM9（CC0 1.0）、MP20（MIT/CC BY 4.0）、GEOM-Drugs（CC0 1.0）、SCOPe-2k（自由可用）。
- **代码**：开源，https://github.com/Zatom-AI/nucleus，含全部文档、数据加载基础设施和训练/推理材料。
- **权重**：论文声明提供预训练权重（Project Page: https://zatomai.github.io/nucleus）。
- **关键超参**：模型规模 70M/140M/270M；Adam(lr=1e-4, β₁=0.9, β₂=0.95, ε=1e-8)；bf16 混合精度；σ_data=16Å；80 epoch 预训练；200 sampling steps（OMol25/OMat24）/100 steps（QM9/MP20）；EDM exponent=7。
