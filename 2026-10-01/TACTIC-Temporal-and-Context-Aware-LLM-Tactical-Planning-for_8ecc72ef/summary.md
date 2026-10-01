---
title: "TACTIC-Temporal-and-Context-Aware-LLM-Tactical-Planning-for"
source: https://arxiv.org/pdf/2609.39969v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:04:47"
field: "自动驾驶对抗性攻击与安全评估"
keywords: ["LiDAR 攻击", "多模态大语言模型", "自动驾驶安全", "战术规划", "场景感知", "物理对抗", "灰盒威胁模型", "CARLA"]
innovations: ["将物理LiDAR攻击形式化为场景依赖的战术规划问题，联合多原语互补选择与连续参数自适应", "提出multimodal scene-graph推理与physics-grounded约束落地，分离度量估计与语义关系推断", "设计异步generate-while-execute架构与∆路径刷新，将场景突变响应延迟从7.4s降至2.0s"]
benchmarks: ["CARLA 0.9.14 Town07", "280 randomized multi-vehicle trials", "Wilson 95% CI + Fisher exact test"]
---

# 论文速读：TACTIC-Temporal-and-Context-Aware-LLM-Tactical-Planning-for

## 一句话总结
论文提出 TACTIC 框架，将物理 LiDAR 攻击执行建模为**场景依赖的战术规划问题**，利用多模态大语言模型（MLLM）结合路边感知与图像语义推理出相对交通关系，动态选择并配置互补的攻击原语（push-away / phantom-obstacle braking），在 CARLA 中实现 100% 碰撞成功率，显著优于固定规则（35%）与随机策略（60%）。

## 研究问题与动机
1. **物理 LiDAR 攻击评估脱离动态交通**：现有工作多用固定原语与手动选定参数在简化场景（直道、预设转向）下验证攻击可行性，缺乏对周围多车交互演化的适应性分析。
2. **攻击有效性强依赖相对交通拓扑**：同一路边感知原语在不同跟随/并行/相邻车道结构下成败迥异，需要基于 TTC、间距、相对速度等时序特征进行在线规划。
3. **单原语攻击无法覆盖多类失效模式**：移除类（spoofing）与注入类（point injection）攻击分别依赖不同车辆关系，单一固定策略易被 AEB 等防御机制规避。
4. **MLLM 推理延迟与高速交通状态不匹配**：直接端到端 LLM 规划难以满足 20 Hz 场景更新需求，需在语义推理与高频执行间解耦。

## 核心贡献（创新点）
1. **将物理 LiDAR 攻击形式化为场景依赖的战术规划问题**：区别于以往单次原语演示，本文以多车相对拓扑与时效约束为核心，定义"选择原语 + 连续参数"的两层决策空间。
2. **提出 multimodal scene-graph 推理模块**：MLLM 联合 SigLIP-2 视觉 token 与序列化车辆状态，自回归解码为结构化 JSON 语义图，使 LLM 专注"关系理解"而将度量估计留给感知后端。
3. **构建 physics-grounded tactical policy**：引入离线校准的 h-group consequence 度量（rear-end / braking 双组加权 TTC、间距、加速度），并结合实测可行边界（如 push-away 的 Δd ∈ [11, 50−o] m、ρ ≤ 3.0 m/s）限定 LLM 输出，保证战术可执行。
4. **设计异步 perception–LLM 架构与 ∆ 路径刷新**：基于 TypeFly generate-while-execute 思路，20 Hz 局部告警器检测到结构级场景突变（间距变化 >8 m 或 40%、车辆增减）才触发重规划；拓扑不变时仅更新边属性，将响应延迟从 7.4 s 降至 2.0 s。
5. **系统性消融与 dose–response 标定**：揭示 push-away 在 Δd≈10.5–11 m 处的锐利可行性相变，并证明联合物理 + 图像输入达到 100% 成功率，单模态分别降至 65%/75%。

## 方法详解
### 威胁模型与攻击目标
- **灰盒威胁**：攻击者部署独立路边感知栈（LiDAR + RGB 相机，20 Hz），可获知传感器/车辆控制配置，但无法读取受害 LiDAR 原生点云、滤波状态或车辆消息。
- 目标车辆 T 被指定，攻击目标是诱发 T 与相邻交通参与者发生碰撞。
- 每轮决策输出 $\mathcal{P}_t = (m, \theta_m)$，其中 $m \in \{\text{push-away, phantom-braking, hold}\}$，单试验最多 $R=3$ 轮。

### 互补攻击原语
| 原语 | 机制 | 参数 |
|---|---|---|
| **push-away** | 偏移前车 A1 的感知距离，使 T 误判前向间距扩大而实际距 A1 缩短 | $\Delta d$（位移）、$\rho$（分离斜率）、$o$（点火相位） |
| **phantom-obstacle braking** | 在 T 前方注入虚拟墙，使其感知 TTC 低于 1.6 s 全制动阈值，由后车 A2 追尾 | $w$（墙距，$w(v)=1.5v$, $w\in[5,15]$ m）、$\tau$（发射持续时间） |
| **hold** | 维持上一轮 $\mathcal{P}_{t-1}$，避免频繁切换 | — |

### 物理与时序可行性边界（III-C）
- **push-away**：$\Delta d_{\min}=11$ m、$\Delta d_{\max}=50-o$ m；$\frac{\Delta d}{t_{\text{arr}}} \le \rho \le 3.0$ m/s。
- **braking**：虚拟墙 $w(v)=1.5v$；持续时间需覆盖 T 的制动响应 + A2 的逼近窗口。
- **时序可行性**：等待会改变间距与 TTC 窗口，attack timing 由连续测量动态确定而非固定时间表。

### 多模态语义场景图生成（IV-A）
- **视觉编码**：SigLIP-2 ViT + lightweight merger 将 $2\times2$ patch 压缩为视觉 token $Z=\{z_1,\dots,z_M\}$。
- **文本编码**：跟踪车辆状态、输出 schema、缓存拓扑序列化为文本 token $X=\{x_1,\dots,x_n\}$。
- **联合解码**：MLLM 使用 interleaved MRoPE 位置 + DeepStack 中阶视觉特征注入，自回归生成结构化 JSON 场景图 $G_t=(V_t,E_t)$。
- **稳态 ∆ 刷新**：若车集合与拓扑不变，仅更新间距/TTC 等度量边属性；否则触发全量重生成。

### Physics-grounded Tactical Policy（IV-B）
1. **h-group 后果度量**（离线网格搜索标定）：
   - rear-end：gap 0.15 / acc 0.70 / TTC 0.15，二次组合 duration 0.10。
   - braking：acc 0.85 / TTC 0.15，二次组合 duration 0.10。
   - 分离 margin：rear-end 96.5×、braking 13.6×（攻击 vs 良性）。
2. **可行性落地**：LLM 同时接收关系结构 + 当前场景数值化边界；确定性代码将输出投影回合法集 $\Phi(S_t)$。
3. **Decision sources 对比基线**：
   - llm policy：全模式 + 全参数选择
   - llm restricted：仅选模式，参数用默认值
   - rule：固定 push-away，参数每试验随机采样一次
   - random：均匀采样模式 + 参数，整试验不变

### 异步场景监控与重规划（IV-C）
- **局部告警器**：20 Hz 比较 tracker 与缓存图。
- **场景突变判定**：车辆出现/消失；间距变化 >8 m 或 >40%；道路上下文变化。
- **最小重规划间隔**：0.5 s。
- **决策流**：告警挂起执行 → 拓扑检查 → ∆ 路径或全量重生成 → 策略从更新后的图与可行性状态恢复。

## 实验与结果
### 实验设置
- **仿真平台**：CARLA 0.9.14，Town07，20 Hz 同步仿真。
- **车辆配置**：target T + 前车 A1 + 后车 A2 + 两辅助参与者 A3/A4；智能驾驶员模型（SEIDM）+ 两级 AEB（TTC=3.0 s partial / 1.6 s full）。
- **路边感知**： pole-mounted LiDAR + RGB 相机，有效范围 ~50 m。
- **MLLM 后端**：Qwen3-VL（基线），Kimi K2（泛化性验证）。
- **Monte Carlo**：T–A1 初始间距均匀采样 [18,28] m，每配置 20 trials，最多 3 轮；成功定义为中心距 <5.0 m 且残余撞击速度 ≥1.5 m/s。
- **总试验数**：280 次主试验 + 20 次 benign control。

### 主要结果
| 指标 | 数值 |
|---|---|
| **LLM policy（全）** | 20/20 = **100%**（[84,100] 95% CI） |
| LLM restricted | 15/20 = 75% |
| Fixed rule（固定 push-away） | 7/20 = 35% |
| Random | 12/20 = 60% |
| Fisher p（policy vs rule） | $1.3\times10^{-5}$ |
| **物理+图像输入** | 20/20 = 100% |
| 仅物理输入 | 13/20 = 65% |
| 仅图像输入 | 15/20 = 75% |
| **异步 ∆ 刷新延迟** | 2.0 s（同步 7.4 s → 异步 3.7 s） |
| 立即执行 vs 等待脆弱窗口 | 100% vs 45%，时间 34.5 s vs 43.7 s |
| Benign control | 0/20 碰撞，最小间距 17.9 m |

### 可行性标定
- **push-away**：Δd=5 m → 0/20；Δd=10 m → 5/20；Δd=10.5 m → 12/20；Δd≥11 m → 20/20。锐利相变位于 10.5–11 m。
- **phantom-braking**：2–10 s 持续时间内成功率 ≥80%，主导约束为 A2–T 间距而非持续时间。
- **ramp 一致性**：ρ > 3.0 m/s 在 3/3 次被检测；ρ ≤ 3.0 m/s 在 0/10 次被检出异常。

### 额外消融
- **Kimi 后端**：full=20/20，restricted=16/20，与 Qwen 差异不显著（Fisher $p=1.0$）。
- **战术 duty cycle & 剂量**：full policy duty=0.39，dose=3.9；restricted duty=0.65，dose=9.7，说明参数自适应同时提升有效性与信号效率。

## 相关工作脉络
1. **物理 LiDAR 攻击**（Cao 2019; Jin 2023; Sato 2025）：聚焦单原语移除/注入，本文将其扩展为**多原语互补 + 场景在线切换**。
2. **系统综述 Sok**（Zhang et al. 2025, arXiv:2509.11120）：指出 joint perception–decision 攻击与 scene-aware hybrid attack chains 为开放方向，本文直接回应该呼吁。
3. **Attack hardness 离线搜索**（Kim et al. 2024 SP）：仅离线识别可行条件；本文在**线上实时测算**并动态接地 LLM 决策。
4. **LLM 对抗场景生成**（Mei et al. 2025, LLM-Attacker）：生成新驾驶场景而非协调在线物理攻击；本文针对**既有交通流的战术规划**。
5. **自动驾驶 LLM/MLLM 推理**（GPT-Driver, LMDrive, DriveLM）：面向控制与预测的顺向任务；本文**逆向迁移**至攻击战术，证明同类架构可被安全研究员复用。
6. **低延迟 LLM 规划**（TypeFly 2025）：generate-while-execute 范式被本文引入路边 LiDAR 攻击的异步监控架构。

## 局限性与未来方向
1. **仿真环境局限**：所有试验在 CARLA 中进行，未部署真实 roadside 硬件与真实 LiDAR/相机噪声。
2. **场景复杂度有限**：仅考虑单车道前后车（T-A1-A2）结构，未覆盖多 lane 换道、交叉路口或复杂合流。
3. **固定两类原语**：仅使用 push-away 与 phantom-braking，未探索更多原语（如联合 lateral 扰动、多车协同攻击）。
4. **MLLM 后端单一验证**：虽测试 Qwen 与 Kimi，但未系统评估不同规模/架构模型的泛化与成本。
5. **防御视角缺失**：未讨论 AEB 参数自适应、感知冗余或多传感器融合对攻击可行性的削弱效应。
6. **未来方向**（作者自述）：扩展至真实硬件平台、学习型感知栈、更丰富的序列级战术。

## 研究启发与可借鉴点
1. **"度量估计 ≠ 语义推理" 的分工范式**：将低层状态估计（位置、速度、TTC）保留给专用感知模块，MLLM 仅负责关系推断与战术选择，可显著降低幻觉风险并提高可解释性；该范式可直接迁移至**自动驾驶规划/预测**任务。
2. **Physics-grounded LLM 决策**：用离线 h-group 度量与实测可行边界双重约束 LLM 输出，并通过确定性投影保证物理一致性；适用于任何"LLM 控制物理系统"的场景（如机器人、无人车）。
3. **异步 generate-while-execute + ∆ 路径刷新**：以 20 Hz 局部监控捕捉结构突变，稳态下仅增量更新边属性，将 LLM 调用次数从每次全量降至 ~2.5 次/试验；可推广至无人机/无人车实时决策。
4. **Dose–response 锐利相变发现**：push-away 在 10.5–11 m 处存在可行性跃迁，提示**参数敏感性分析**应在任何攻击/控制策略部署前进行；可复用于其他物理对抗任务的阈值标定。
5. **多模态互补性验证**：物理测量（精确性）与图像语义（车道/走向上下文）联合达到 100% vs 单模态 65%/75%，说明**异构观测融合**对复杂交互任务至关重要；对多传感器自动驾驶安全评估有借鉴价值。

## 关键术语表
**TACTIC**：Temporal and Context-Aware LLM Tactical Planning，本文提出的场景感知路边 LiDAR 攻击战术规划框架。
**Gray-box threat model**：灰盒威胁模型，攻击者无法访问受害传感器原生数据或内部状态，但可观测物理感知输出并了解系统配置。
**Push-away attack**：通过偏移前车感知距离，使目标车误判前向间距而缩短真实间距，诱发自撞。
**Phantom-obstacle braking**：在目标车前方注入虚拟障碍，使其 TTC 低于制动阈值触发紧急刹车，由后车追尾。
**Semantic scene graph**：以节点表示交通参与者、边表示同向/相邻/跟随等语义关系的结构化图表示。
**h-group measure**：离线校准的后果度量，加权组合归一化间距、加速度与 TTC，用于在攻击者与受害者视角下量化潜在危害严重性。
**∆ 路径刷新**：在车辆集合与拓扑不变时，仅增量更新场景图的度量边属性，避免全量 MLLM 重生成。
**Generate-while-execute**：源于 TypeFly，指推理通道与执行通道异步重叠，感知的高频监控不阻塞 LLM 的低频推理。

## 可复现要素
- **数据集**：CARLA 0.9.14，Town07；初始 T–A1 间距均匀采样 [18,28] m；种子随机化；**已开源仿真环境**，但非公开 benchmark 数据集。
- **代码**：项目网站已公开，源代码已开源（论文未提供具体 repo URL，仅标注 source code released）。
- **权重**：MLLM 使用 Qwen3-VL（开源权重）与 Kimi K2（开源权重），Vision backbone 为 SigLIP-2。
- **关键超参**：LiDAR 20 Hz；感知范围 ~50 m；巡航速度 ~6.0 m/s（闭环 ~5.5 m/s）；AEB 阈值 3.0 s / 1.6 s；Δd_min=11 m；ρ_max=3.0 m/s；w(v)=1.5v，w∈[5,15] m；间距突变阈值 >8 m 或 >40%；最小重规划间隔 0.5 s；最大轮数 R=3。
- **评估指标**：成功率、t_90、d<7、R<1、发射 Gini G、剂量效率 η、95% Wilson CI、Fisher 精确检验。
