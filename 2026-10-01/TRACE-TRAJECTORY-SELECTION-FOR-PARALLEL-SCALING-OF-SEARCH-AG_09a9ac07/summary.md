---
title: "TRACE-TRAJECTORY-SELECTION-FOR-PARALLEL-SCALING-OF-SEARCH-AG"
source: https://arxiv.org/pdf/2609.39912v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:04:55"
field: "Agent with Search and Reasoning"
keywords: ["trajectory selection", "parallel scaling", "search agents", "graph neural networks", "test-time scaling", "evidence reasoning"]
innovations: ["提出出现保留的跨rollout证据图表示，将轨迹整合定义为选择而非生成", "设计轻量级可迁移选择器，单检查点跨六种rollout策略和多个长时序生成器泛化", "证明更好选择可减少rollout预算，K=8达到K=64投票性能，效率提升10-258倍"]
benchmarks: ["WebQA (NQ, HotpotQA, TriviaQA, PopQA, 2Wiki, MuSiQue, Bamboogle)", "BrowseComp-Plus", "FRAMES", "GAIA"]
---

# 论文速读：TRACE - Trajectory Selection for Parallel Scaling of Search Agents

## 一句话总结
论文提出了 **TRACE（Trajectory Ranking with Aggregated Cross-Rollout Evidence）**，一个轻量级可迁移的轨迹选择器，通过构建保留查询-证据图谱并在跨rollout传播共享证据信息，从K个已完成搜索轨迹中选出最优候选答案，避免了投票丢弃搜索上下文或生成式聚合引入额外自回归计算的问题。

## 研究问题与动机
- **并行搜索的整合瓶颈**：对同一问题采样K条独立搜索轨迹后，正确答案可能已在候选池中，但基于最终答案的一致性投票可能遗漏它，因为投票忽略了产生答案的搜索过程和证据上下文。
- **现有方法的信息损失**：答案级投票（如Majority Voting）仅看最终答案一致性，丢弃了支撑答案的检索路径和证据来源；生成式聚合（如AggAgent）虽能综合多轨迹信息，但需额外调用大语言模型进行自回归合成，当正确答案已存在时引入了不必要的计算成本和格式错误风险。
- **跨rollout证据共享的价值未被充分利用**：不同轨迹可能通过不同子查询检索到相同文本片段，或浏览同一文档的不同段落，这些共享证据提供了判断轨迹质量的强信号，但直接合并重复观察会抹除查询上下文和检索来源。
- **选择器与生成器的解耦需求**：现有工作往往将整合策略绑定到特定生成器，缺乏跨不同 rollout 策略和 agent 骨架的通用选择能力。

## 核心贡献（创新点）
1. **将 rollout 后整合重新形式化为轨迹选择问题**：直接对 K 个已完成的搜索轨迹打分并返回已有候选，与轨迹生成解耦，区别于投票（丢弃过程信息）和生成式聚合（引入额外自回归阶段）。
2. **提出出现保留的跨rollout证据推理机制**：构建保留每个查询和证据发生的图，通过共享证据（WebQA中的chunk identity，长时序中的document identity）连接不同轨迹，使跨rollout信息传播不抹除检索溯源信息。
3. **设计轻量级、可迁移的选择器架构**：基于冻结文本嵌入（Qwen3-Embedding-8B）和关系特定GNN，配合answer-conditioned QFormer读入，单检查点在六种WebQA rollout策略和多个长时序数据集-骨架组合上无需微调即可泛化。
4. **展示选择效率优势**：在Base WebQA上仅用8条轨迹即可达到与64条轨迹多数投票相当的性能（43.3% vs 43.6% EM），后处理吞吐量比三种LLM聚合器高至少10倍。

## 方法详解
**整体流程**：给定问题q和K条完成轨迹$\mathcal{T}_K$，TRACE构造查询-证据图→关系特定图传播→answer-conditioned QFormer读入→答案打分→返回最高分候选。

**1. 出现保留的查询-证据图构建**
- 节点对应具体发生：每个Search调用产生Subquery节点，每个检索到的证据产生Evidence节点；两条轨迹检索到相同文本仍保留为独立节点。
- **WebQA**：每个Search返回Top-3 ranked chunk，Rank-specific边连接Subquery与Evidence，保留顺序信息；相同normalized chunk通过双向边连接（区分rollout内/外匹配）。
- **长时序浏览**：引入Doc节点表示文档身份，所有来自同一来源的Evidence通过两跳路径$e_{i,t,j} \rightarrow d_m \rightarrow e_{i',t',j'}$连接，支持"不同段落同源"的关联。

**2. 跨rollout证据传播（关系特定GraphSAGE）**
- 冻结Qwen3-Embedding-8B（4096维）提供文本嵌入， trainable type-specific encoder映射到$d=256$维初始状态$\mathbf{h}_v^{(0)}$。
- 答案初始状态额外融合投票特征：$\mathbf{h}_{a_i}^{(0)} = \phi_A(\mathbf{x}_{a_i}) + \mathbf{W}_{vote}\mathbf{f}_i$，其中$\mathbf{f}_i = [\log(1+c_i), c_i/R_q]^\top$。
- 使用$L=4$层关系特定GraphSAGE：
  $$\mathbf{m}_v^{(\ell)} = \sum_r \left(\mathbf{W}_r^{(\ell)} \mathrm{Mean}_{u \in \mathcal{N}_r(v)} \mathbf{h}_u^{(\ell)} + \mathbf{b}_r^{(\ell)}\right), \quad \mathbf{h}_v^{(\ell+1)} = \mathrm{LN}\left(\mathbf{h}_v^{(\ell)} + \mathrm{Dropout}(\mathrm{GELU}(\mathbf{m}_v^{(\ell)}))\right)$$
- 仅Subquery和Evidence节点参与消息传递；Query和Answer节点不参与图传播，保证最终决策保留轨迹本地溯源。

**3. Answer-conditioned QFormer读入**
- 每个候选答案$a_i$通过cross-attention查询其自身轨迹$\tau_i$内的Subquery/Evidence更新状态$\mathbf{H}_i$：
  $$\alpha_{i,v} = \mathrm{softmax}_v\left(\frac{(\mathbf{W}_Q \mathbf{h}_{a_i}^{(0)})^\top (\mathbf{W}_K \tilde{\mathbf{h}}_v)}{\sqrt{d_h}}\right), \quad \mathbf{z}_i = \sum_{v \in \mathcal{C}_i} \alpha_{i,v} \mathbf{W}_V \tilde{\mathbf{h}}_v$$
- 四头attention后接FFN得到候选表示$\mathbf{h}_{a_i}^\star$，实现"局部溯源+全局感知"。

**4. 打分与损失**
- 问题投影$\mathbf{z}_q = \rho_Q(\mathbf{h}_q^{(0)})$，答案投影$\rho_A(\mathbf{h}_{a_i}^\star)$，余弦相似度打分：
  $$\mathrm{score}_i = \gamma \cdot \cos(\mathbf{z}_q, \rho_A(\mathbf{h}_{a_i}^\star)), \quad \hat{i} = \arg\max_{i \in \mathcal{I}_q} \mathrm{score}_i$$
- 训练损失$\mathcal{L} = \mathcal{L}_{\mathrm{BCE}} + \mathcal{L}_{\mathrm{list}} + \mathcal{L}_{\mathrm{hard}}$：
  - BCE：候选级二分类（考虑正负样本权重）
  - Listwise：聚焦正例轨迹的概率质量
  - Hard ranking：拉开最高分错误候选与最强正确候选的距离
- 文本嵌入和rollout生成策略保持冻结，仅训练选择器组件。

## 实验与结果
**数据集与设置**
- **WebQA**：3,125题（NQ、HotpotQA、TriviaQA、PopQA、2Wiki、MuSiQue、Bamboogle），使用Qwen2.5-7B/14B的Base/SFT/RL六种rollout策略。
- **长时序搜索**：BrowseComp-Plus（830题）、FRAMES（500题）、GAIA text-only（103题），使用OpenResearcher-30B-A3B和gpt-oss-120B两个生成器。
- 评估指标：WebQA为question-weighted EM/F1；长时序为accuracy（取Qwen3-32B判定的正确性）。

**主要结果（K=16）**
| 设置 | TRACE | 最佳基线 | 提升 |
|------|-------|----------|------|
| WebQA Base (Qwen2.5-14B) | 45.2% EM | SolAgg 43.9% | +1.3pp |
| WebQA SFT (Qwen2.5-14B) | 49.2% EM | SolAgg 48.0% | +1.2pp |
| 长时序平均（OR+OSS） | 78.6% | Majority Voting 75.5% | +3.1pp |
| gpt-oss-120B BrowseComp-Plus | 74.3% | Fewest Tools 72.7% | +1.6pp |

**效率对比**（3,125题SFT WebQA池，H20 GPU）
- TRACE：0.271 GPU-hours（16.2分钟）
- SolAgg：3.28（12×）/ SummAgg：32.96（122×）/ AggAgent：69.93（258×）

**预算缩放**（固定检查点，无需重训）
- WebQA Base K=8时TRACE达43.3% EM，接近 Majority Voting K=64时的43.6%，节省8倍rollout开销。

**消融验证**
- 移除跨rollout通信：WebQA 45.2%→43.9%，长时序 78.6%→72.7%（最大降幅）
- 移除GNN：WebQA 43.7%，长时序 75.5%
- 固定query读入：WebQA 44.7%，长时序 75.6%
- 合并相同证据：44.1%/77.1%，验证出现保留的重要性

## 相关工作脉络
1. **Parallel scaling of search agents**：Brown et al. (2024)提出重复采样提升推理，Zhu et al. (2025)、Zeng et al. (2025)将其扩展到工具增强轨迹；TRACE将整合阶段重新定义为选择而非生成。
2. **Answer-level voting**：Wang et al. (2023)的Self-Consistency；投票仅看最终答案一致性，丢弃搜索过程信息，TRACE通过图结构利用跨rollout证据关联。
3. **Learned verifiers**：Cobbe et al. (2021)、Montgomery et al. (2025)独立打分候选；TRACE将轨迹质量视为关系性——一个轨迹的得分可依赖其他轨迹遇到的证据。
4. **Generative aggregation**：Lee et al. (2026)的AggAgent、Li et al. (2025)的ParallelMuse；调用另一个LLM综合多轨迹；TRACE返回已有候选，避免额外自回归推理和格式错误。
5. **Graph-structured evidence reasoning**：Zhou et al. (2019)的GEAR、Fang et al. (2020)的多跳QA图网络；TRACE的独特之处在于保留每个查询-证据的独立出现并连接跨rollout共享证据。

## 局限性与未来方向
- **生成器依赖性**：Rollout生成策略影响候选池质量和分布；RL策略产生高重叠证据（Jaccard 0.87），导致TRACE相对于投票的提升幅度减小（0.7pp vs Base的3.0pp），说明选择器对候选多样性敏感。
- **评估范围**：仅在Qwen系列模型和特定搜索工具集（Search/Open/Find）上验证；未测试其他agent骨架（如Code interpreter、API调用）或更复杂的工具交互模式。
- **精确内容匹配局限**：WebQA依赖exact chunk matching，长时序依赖URL-level document identity；对于paraphrase或语义等价但未精确匹配的证据，当前图连接可能遗漏。
- **训练数据规模**：WebQA训练集约10万题，长时序仅2,655题，较小样本下learned selector的泛化能力有待进一步验证。
- **未来方向**：可扩展至paraphrase-based证据匹配、多模态搜索轨迹（图像/表格）、以及online setting下的增量图更新。

## 研究启发与可借鉴点
1. **Occurrence-preserving图表示的可迁移性**：保留每个事件独立出现的图设计可推广到代码生成（保留每次尝试的编译错误）、数学推理（保留每次推导步骤）等场景，避免信息合并导致的上下文丢失。
2. **两阶段设计（图传播+answer-conditioned读入）的结构价值**：先跨轨迹共享证据，再由候选独立读取自身上下文，这一分离既利用全局信息又保留局部溯源，可应用于多agent协作评审、ensemble selection等任务。
3. **选择器与生成器解耦的工程收益**：单检查点跨多种rollout策略和模型骨架通用，大幅降低部署成本；可作为"即插即用"模块集成到现有搜索agent pipeline中。
4. **预算缩放的理论洞见**：TRACE在K=8时达到Majority Voting K=64的性能，验证了"更好的选择可比更多采样更有效"，为test-time scaling资源分配提供新视角。
5. **融合投票特征的轻量设计**：答案频率特征（log频次、比例）直接编码到初始状态，以极低成本捕捉部分投票信号，兼顾判别力与效率。

## 关键术语表
**Trajectory Selection**：从K个已完成搜索轨迹中直接选择一个已有候选答案，而非生成新答案的整合范式。
**Occurrence-Preserving Graph**：保留每个查询调用和证据检索作为独立节点的图表示，相同内容通过边连接而非节点合并。
**Cross-Rollout Communication**：通过共享证据（chunk/document identity）在不同搜索轨迹间传播信息的机制。
**Answer-Conditioned QFormer**：以候选答案embedding为query、轨迹内过程状态为key/value的cross-attention模块，实现本地溯源+全局感知的读入。
**Relation-Specific GraphSAGE**：按关系类型分别学习的图神经网络层，支持WebQA的直接chunk匹配和长时序的两跳Evidence-Doc-Evidence路径。
**Pass@K**：在K次rollout中至少获得一次正确答案的概率，衡量候选覆盖率。
**Generative Aggregation**：调用另一个LLM综合多个轨迹的搜索过程和证据以合成最终答案的方法（如AggAgent、SolAgg）。
**Long-Horizon Browsing**：涉及多轮Search/Open/Find工具调用的复杂搜索任务，轨迹长度远超单次检索。

## 可复现要素
- **数据集**：WebQA（NQ、HotpotQA、TriviaQA、PopQA、2Wiki、MuSiQue、Bamboogle）为标准公开数据集；长时序使用OpenResearcher/web-bench中的BrowseComp-Plus、FRAMES、GAIA split，部分数据需申请访问。
- **代码开源**：GitHub https://github.com/Jaasssoooonnnnn/TRACE 包含实现、训练配置和评估脚本。
- **模型依赖**：冻结Qwen3-Embedding-8B作为文本编码器（HuggingFace可下载）；Qwen2.5-7B/14B Base/SFT/RL为rollout生成器。
- **关键超参**：embed_dim=4096，hidden_dim d=256，GNN层数L=4，QFormer四头宽度64，AdamW lr=3e-4，weight_decay=1e-4，dropout=0.1，batch_size=256（WebQA）/4（长时序），训练3 epoch。
- **训练数据量**：WebQA 101,323训练/5,309验证题；长时序 2,655训练/132验证题。
- **评估硬件**：NVIDIA H20 GPU。
