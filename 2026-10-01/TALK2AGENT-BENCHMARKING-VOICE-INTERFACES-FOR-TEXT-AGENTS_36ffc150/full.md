# TALK2AGENT: BENCHMARKING VOICE INTERFACES FOR TEXT AGENTS

Terumi Chiba<sup>⋆</sup>

Guangzhi Sun<sup>†</sup>

Zheqi Yuan<sup>⋆</sup>

Chao Zhang<sup>⋆</sup>

<sup>⋆</sup>Department of Electronic Engineering, Tsinghua University <sup>†</sup>University of Cambridge

## ABSTRACT

Large language model (LLM) computer-use agents are typically evaluated with clean written instructions, despite speech being an increasingly popular interface for interacting with such systems. Speech input introduces an additional failure point: transcription errors can alter task-critical entities, constraints, or targets before the agent begins reasoning, while conventional ASR metrics do not directly measure whether the information required for successful execution has been preserved. We introduce Talk2Agent, a benchmark for evaluating how effectively voice interfaces convey humanspoken instructions to LLM-based computer-use agents. Talk2Agent builds human-spoken versions of tasks from WildClawBench and OSWorld and evaluates a range of voice interfaces, including dedicated ASR models, audio-capable LLMs, contextual biasing, and LLM-based ontology repair. Because repeatedly executing longhorizon computer-use tasks is costly and stochastic, we further propose an execution-free, task-conditioned evaluation framework that projects the original task grader onto prompt-addressable intentions and measures how much task-relevant information is retained after the voice interface. On WildClawBench, Talk2Agent’s executionfree native projection provides a practical, execution-grounded measure of voice-interface quality, correlating with downstream task completion and improving Pearson correlation by 0.246 over WER/CER on 32 hours of real human speech.

Index Terms— ASR, voice interfaces, computer-use agents, task-conditioned evaluation, speech benchmarks

## 1. INTRODUCTION

Large language models (LLMs) are increasingly capable of performing agentic tasks, and benchmarks such as WebArena [1] and OS-World [2] assess them through functional outcomes in interactive environments. However, these benchmarks assume clean written user instructions. In practice, speech is becoming a natural interface for interacting with intelligent systems, introducing an additional and underexplored source of failure before an agent begins reasoning or acting. This is particularly important for computer-use tasks, where instructions often contain filenames, application names, paths, numbers, entities, and other task-critical details that are difficult to recognize reliably from speech. A transcription may appear largely correct while altering a single constraint or target that determines task success. Existing lexical or semantic ASR metrics therefore provide only an indirect measure of whether a spoken instruction remains useful to an agent, while evaluating every interface through full agent execution is costly and noisy, especially for long-horizon tasks.

We introduce Talk2Agent, a benchmark for evaluating how effectively voice interfaces convey human-spoken instructions to LLM-based computer-use agents. Talk2Agent builds spoken versions of tasks from WildClawBench (WCB) [3] and OSWorld [2], covering both command-line and graphical computer interaction, and evaluates the resulting instructions with fixed downstream agents. This setup isolates the voice interface from the agent itself, and allows us to study how different interface designs preserve task-critical information. We evaluate both dedicated ASR systems and audio-capable LLMs as voice interfaces. Together with these interfaces, we investigate contextual biasing mechanisms that incorporate task-relevant environmental information, and LLM-based ontology repair that attempts to recover corrupted entities and terminology. In this way, Talk2Agent measures not only transcription quality, but the extent to which different voice interfaces preserve the information needed for successful agent execution.

Full execution, however, is expensive as a routine evaluation mechanism. Agent reliability can require repeated trials to characterize [4]; benchmarks such as WCB additionally involve long interaction trajectories and environment resets. We therefore additionally propose an execution-free evaluation framework that reuses the grading logic of the original text-agent benchmark. We decompose each task into prompt-addressable intentions and measure whether those intentions are retained after passing through the voice interface, weighting them according to their contribution to the original task grader. This produces task-conditioned retention measures that can diagnose interface errors without executing the downstream agent, while remaining grounded in what the benchmark ultimately rewards. We validate these measures against actual agent execution; on WCB, Native projection improves Pearson correlation coefficient (PCC) with task performance by an absolute 0.246 over WER/CER. Main contributions are summarized as follows:

• We propose Talk2Agent, an execution-grounded benchmark that isolates voice-interface quality for fixed LLM computer-use agents. Talk2Agent contains 31.82 hours of human-recorded instructions from WCB and OSWorld.

• A systematic study of voice interfaces for computer-use agents is performed, covering dedicated ASR models, audio-capable LLMs, contextual biasing, and LLM-based ontology repair.

• An execution-free, task-conditioned evaluation framework that projects an existing agent benchmark’s grading logic onto instruction intentions, enabling efficient evaluation of voice interfaces without repeatedly executing expensive long-horizon tasks.

## 2. RELATED WORK

Earlier spoken language understanding benchmarks evaluate intent and entity recognition [5] and compositional semantic parsing [6]. Talk2Agent instead targets the information needed for computer-use task execution.

Voice-agent benchmarks span spoken function calling and tool use [7, 8, 9], entity-sensitive task completion and conversation quality [10], and full-duplex voice agent tasks [11]. A closely related framework converts existing text tool-calling benchmarks into synthetic audio while retaining their tool schemas and labels [12]. These works primarily evaluate tool calling or end-to-end agents, where recognition, reasoning, and action jointly determine outcomes. In contrast, Talk2Agent uses real human speech for computer-use tasks and isolates the voice interface before a fixed downstream text agent while retaining task execution as the endpoint.

Table 1. Talk2Agent speech collections. Coverage is tasks × recordings per task = total recordings.
<table><tr><td>Source</td><td>Coverage</td><td>Hours</td><td>Median [IQR] (s)</td></tr><tr><td>WCB</td><td> $6 0 \times 1 0 = 6 0 0$ </td><td>24.84</td><td>113.66 [55.01–212.31]</td></tr><tr><td>OSWorld</td><td> $3 5 1 \times 5 = 1 , 7 5 5$ </td><td>6.98</td><td>11.54 [7.00–18.28]</td></tr></table>

Beyond lexical WER/CER, ASR metrics assess semantic similarity [13, 14, 15, 16], error severity [17], downstream answer preservation [18], and atomic requirements [19]. SeMaScore [20] additionally weights aligned segments by their reference-context importance, while generative-LLM evaluation studies hypothesis preference and semantic error assessment [21]. These approaches improve semantic or application-oriented assessment but do not directly encode the source computer-use benchmark’s grading logic. Talk2Agent projects that grader onto prompt-addressable intentions, retaining its task-specific weights and non-additive interactions.

## 3. TALK2AGENT BENCHMARK

## 3.1. Benchmark construction

Talk2Agent is constructed from two source benchmarks with complementary interaction modalities: WildClawBench (WCB) provides command-line tasks, and OSWorld provides graphical desktop tasks. We retain each source task and its original downstream evaluator so that an interface can be assessed by actual agent execution rather than transcript similarity alone. WCB contributes 60 formal tasks, excluding templates, and OSWorld contributes 351 Englishlanguage tasks.

Written agent prompts are not read verbatim. GPT-5.4 rewrites each prompt into a speakable script: prose becomes natural speech, lists and JSON are verbalized by sequence and fields rather than punctuation, and paths, filenames, and other exact tokens are spoken explicitly. The rewrite preserves execution-critical entities, constraints, literal symbols, and requested outputs; human review verifies that presentation changes do not alter task intent. Each WCB task is then recorded by ten Chinese–English proficient speakers (five women and five men). Each OSWorld task has five recordings; its participant pool contains 41 women and 57 men. Table 1 summarizes the resulting 31.82 hours of human speech.

WCB recordings used smartphone or computer microphones in quiet indoor settings and were manually checked for completeness and intelligibility, with rerecording as needed. OSWorld recordings were collected through Prolific in 20-task batches under the same device and script-fidelity requirements.

Figure 1 shows the evaluation pipeline. Human speech is processed by a voice interface, whose text output is passed unchanged to a fixed downstream agent. The source benchmark evaluator then scores the resulting execution. This endpoint preserves the operational meaning of each task, but it also combines voice-interface information loss with agent capability, environment failures, and run stochasticity. Section 4 therefore introduces a complementary execution-free diagnostic that evaluates the interface before downstream action.

## 3.2. Voice-interface methods

The benchmark compares vanilla ASR, audio-capable LLMs, and two agent-oriented interventions. Contextual biasing supplies tasklocal terms from agent-visible workspace content, including GUI OCR, to improve recognition of rare and domain-specific entities [22, 23, 24]. Ontology repair applies Typeless-inspired LLM post-processing [25] to restore punctuation and technical terms only when supported by task-local context. We evaluate generative error correction [26] and contextual entity correction [27, 28] as controlled benchmark interventions rather than propose a new correction algorithm. Original and colloquialized prompts serve as controls; model configurations are given in Sec. 5.

## 4. EXECUTION-FREE EVALUATION

Direct execution is the benchmark’s final endpoint, but it is expensive and noisy: observed score differences conflate voice-interface loss with downstream planning errors, limited agent capability, environment failures, and run-to-run variation. We therefore estimate the task-relevant information retained by an interface under an idealized downstream agent, while remaining grounded in the source benchmark’s own grading logic.

For task t, we compile its evaluator into Boolean criteria $P _ { t } =$ $I _ { t } \cup F _ { t } . \quad I _ { t }$ contains the smallest prompt-addressable intentions, while $F _ { t }$ contains fixed assessment conditions that affect grading but cannot be changed by the voice interface. Let $v _ { t } ( C )$ be the sourceevaluator score when exactly criteria $C \subseteq P _ { t }$ are satisfied. For a retained intention set $\mathcal { R } \subseteq I _ { t }$ , we condition on the fixed criteria and remove their baseline:

$$
g _ { t } ( \mathcal { R } ) = v _ { t } ( F _ { t } \cup \mathcal { R } ) - v _ { t } ( F _ { t } ) .\tag{1}
$$

This compilation preserves the evaluator’s sums, products, gates, thresholds, caps, and floors; continuous criteria are represented by their unsatisfied and fully satisfied endpoints. Using the Mobius ¨ representation of set functions [29], the conditioned game expands as $\begin{array} { r } { g _ { t } ( \mathcal { R } ) ~ = ~ \sum _ { T \subset \mathcal { R } } c _ { T } } \end{array}$ . Its exact Shapley values are $\begin{array} { r } { \phi _ { t i } = \sum _ { T \ni i } c _ { T } / | T | } \end{array}$ and satisfy efficiency, $\begin{array} { r } { \sum _ { i \in I _ { t } } \phi _ { t i } = g _ { t } ( I _ { t } ) - } \end{array}$ $g _ { t } ( \emptyset )$ [30].

We report three complementary diagnostics. Lost intentions counts the breadth of information loss. Native projection

$$
N _ { t } ( \mathcal { R } ) = g _ { t } ( \mathcal { R } ) / g _ { t } ( I _ { t } )\tag{2}
$$

substitutes retained intentions into the original grader and therefore preserves its joint logic. Shapley retention

$$
S _ { t } ( \mathcal { R } ) = \sum _ { i \in \mathcal { R } } \phi _ { t i } \Big / \sum _ { i \in I _ { t } } \phi _ { t i }\tag{3}
$$

uses these values to additively allocate benchmark value to intentions. Native is an execution-oriented surrogate rather than a success probability; Shapley instead measures how much attributed task value survives, so neither score subsumes the other.

Each transcript intention is judged retained, changed, omitted, or unresolved; changed and omitted are treated as lost. For either $M _ { t } \in \{ N _ { t } , S _ { t } \}$ , unresolved intentions $\mathcal { U } _ { t }$ produce the exact interval

$$
[ M _ { t , L } , M _ { t , U } ] = \left[ \operatorname* { m i n } _ { \mathscr { Q } \subseteq \mathscr { U } _ { t } } M _ { t } ( \mathscr { R } _ { t } \cup \mathscr { Q } ) , \operatorname* { m a x } _ { \mathscr { Q } \subseteq \mathscr { U } _ { t } } M _ { t } ( \mathscr { R } _ { t } \cup \mathscr { Q } ) \right] .\tag{4}
$$

![](images/a3b0370d8281fc259b1d9c4d792c78eca4742ebea42dd2e22aee326356ee769b.jpg)  
Fig. 1. Talk2Agent dataset construction and evaluation. Written agent tasks are colloquialized to spoken forms and recorded by human speakers. Voice-interface outputs are evaluated either by a fixed downstream text agent and the source task grader or by execution-free projection of that grader onto retained task intentions.

Midpoints are used only when a single value is required for aggregate analysis. Execution outcomes are never exposed during intention extraction, weighting, or retention judgment.

## 5. EXPERIMENTAL SETUP

## 5.1. Interface and agent configurations

On WCB, we adopt ten interfaces cross Parakeet and Gemini recognition with contextual biasing and GPT-5.4 ontology repair, alongside original-text and colloquialized-text controls. Task-local bias terms are extracted by a simple prompt from agent-visible workspace crawls; scoring secrets are excluded. All WCB executions use GPT-5.6-Luna as the fixed downstream agent, with three runs per task–interface pair. To keep full execution tractable and control the acoustic input across interfaces, WCB execution uses TTS versions of the spoken prompts. Executing all ten human recordings per task would multiply the downstream-agent cost tenfold; these recordings are therefore more naturally analyzed using execution-free evaluation. On OSWorld, the interface comparison includes GPT-4o, Whisper [31], Parakeet, a Qwen recognizer, and Gemini end-to-end audio input. GPT-4o and Whisper ontology repair use GPT-5.4. Bias terms combine agent-visible workspace content with Apple Vision OCR in fast mode with language correction enabled. We use Gemini-3.5-flash and Qwen-3.5-9B as proprietary and open-source downstream agents, respectively, following the official OSWorld harness [2].

## 5.2. Execution-free validation protocol

Before observing execution results, we audit each WCB prompt, programmatic grader, LLM-judge instruction, and aggregation rule. Requirements are split at the smallest independently scored criterion; reference-derived criteria remain linked to their parent requirement, and repetitions are not double counted. The 60 tasks yield 549 prompt-addressable intentions, with 2 to 46 per task (mean 9.15; median 6). GPT-5.6-Luna assigns retention labels using a fixed prompt without access to executions or scores. Unresolved labels account for 3.00% of judgments; suspected errors are independently re-reviewed, and remaining uncertainty is propagated through Eq. 4.

Correlation validation uses 50 non-Safety WCB tasks under ten matched conditions, for 500 task–interface observations. Safety tasks are excluded because missing an unsafe source intention can improve a safety-aligned execution score, reversing the ordinary retention-success relationship. We compare execution-free scores with direct execution and higher-is-better versions of WER/CER, SemDist, AER, Atomic Rubric, and Semantic WER. PCC is reported over 500 task–interface pairs, 50 task-type–interface pairs, and ten interface means. Task-grouped bootstrap intervals resample tasks while preserving all interfaces from each sampled task.

## 6. RESULTS

## 6.1. Operational benchmark results

Table 2 reports the frozen 60-task WCB execution artifact, comprising ten matched interfaces and three runs per task–interface pair, with equal weighting across tasks and runs. All executions use GPT-5.6-Luna. These measurements are the operational endpoint, but they combine interface loss with downstream planning, environment failures, and run stochasticity. No intervention is uniformly beneficial in the full cohort. Parakeet changes little across settings.

Table 2. Results on Talk2Agent WCB split. Mean ± std. deviation reported across three round-level interface means; each round equally averages the 60 tasks. Voice rows use TTS audio.
<table><tr><td>Input / interface</td><td>Biasing</td><td>Ontology Repair</td><td>Execution (%)</td></tr><tr><td>Original text</td><td>一</td><td>一</td><td> $4 0 . 7 5 \pm 1 . 6 7$ </td></tr><tr><td>Spoken-form text</td><td>一</td><td>一</td><td> $3 6 . 3 9 \pm 3 . 5 6$ </td></tr><tr><td>Parakeet</td><td>x</td><td>x</td><td> $2 4 . 2 0 \pm 2 . 7 6$ </td></tr><tr><td>Parakeet</td><td>x</td><td>√</td><td> $2 5 . 3 2 \pm 1 . 1 3$ </td></tr><tr><td>Parakeet</td><td>√</td><td>x</td><td> $2 5 . 2 8 \pm 3 . 3 9$ </td></tr><tr><td>Parakeet</td><td>√</td><td>√</td><td> $2 5 . 0 3 \pm 2 . 9 5$ </td></tr><tr><td>Gemini</td><td>X</td><td>x</td><td> $4 0 . 8 4 \pm 3 . 3 6$ </td></tr><tr><td>Gemini</td><td>x</td><td>√</td><td> $3 7 . 3 2 \pm 1 . 3 3$ </td></tr><tr><td>Gemini</td><td>√</td><td>x</td><td> $4 3 . 1 5 \pm 1 . 9 8$ </td></tr><tr><td>Gemini</td><td>√</td><td>√</td><td> $3 8 . 6 8 \pm 2 . 2 2$ </td></tr></table>

OSWorld supplies a distinct GUI operating point using human recordings (Table 3). Ontology repair raises the GPT-4o and Whisper execution means by 0.79 and 0.70 points with the Gemini agent, while Parakeet biasing changes them by only +0.13 with Gemini and −0.11 with Qwen. Thus execution effects depend on both the recognizer and downstream agent; OSWorld is not used in the WCB correlation analysis.

Table 3. OSWorld execution from the frozen summary. Values are means and SDs over five speaker recordings per task
<table><tr><td>Agent / input</td><td>Biasing</td><td>Ontology Repair</td><td>Mean</td><td>SD</td></tr><tr><td>Gemini agent</td><td></td><td></td><td></td><td></td></tr><tr><td>Text</td><td>一</td><td>一</td><td>61.76</td><td>0.37</td></tr><tr><td>GPT-40</td><td>X</td><td>x</td><td>58.49</td><td>0.64</td></tr><tr><td>GPT-40</td><td>x</td><td>√</td><td>59.28</td><td>1.09</td></tr><tr><td>Whisper</td><td>x</td><td>x</td><td>56.07</td><td>1.34</td></tr><tr><td>Whisper</td><td>x</td><td>√</td><td>56.77</td><td>1.50</td></tr><tr><td>Parakeet</td><td>x</td><td>x</td><td>56.26</td><td>1.59</td></tr><tr><td>Parakeet</td><td>√</td><td>x</td><td>56.39</td><td>1.22</td></tr><tr><td>Qwen recognizer</td><td>x</td><td>x</td><td>55.44</td><td>1.68</td></tr><tr><td>Gemini end-to-end</td><td>x</td><td>x</td><td>58.87</td><td>1.57</td></tr><tr><td>Qwen agent</td><td></td><td></td><td></td><td></td></tr><tr><td>Text</td><td>一</td><td>一</td><td>41.71</td><td>0.88</td></tr><tr><td>Parakeet</td><td>x</td><td>x</td><td>36.69</td><td>1.50</td></tr><tr><td>Parakeet</td><td>√</td><td>x</td><td>36.58</td><td>0.87</td></tr></table>

## 6.2. Does execution-free scoring track execution?

At the task–interface level, Native exceeds AER, the best generic metric, by 0.180 (Table 4). Shapley and Native exceed AER in 98.95% and 99.89% of task-grouped bootstrap replicates by PCC, and in 93.46% and 99.13% by Spearman correlation; they also rank first and second after task-type aggregation. At the interfacemean level, AER is numerically higher than Shapley (0.916 versus 0.902), but not significantly so (95% CI for $\boldsymbol { r } _ { \mathrm { A E R } } \ : - \ : \boldsymbol { r } _ { \mathrm { S h a p l e y } } ;$ [−0.060, 0.072]), and is higher in only 53.35% of replicates. Thus, task-conditioned metrics better track execution at the task–interface level, while remaining statistically comparable to the best generic metric at the interface-mean level.

Table 4. PCC against WCB execution over 500 task–interface pairs, 50 task-type–interface aggregates, and ten interface means. Flat equally weights intentions; other metrics are in Sec. 5.
<table><tr><td>Metric</td><td>Task-interface</td><td>Type-interface</td><td>Interface</td></tr><tr><td>1-WER/CER</td><td>0.220</td><td>0.176</td><td>0.514</td></tr><tr><td>1-SemDist</td><td>0.218</td><td>0.178</td><td>0.606</td></tr><tr><td>1-AER</td><td>0.286</td><td>0.297</td><td>0.916</td></tr><tr><td>Atomic Rubric</td><td>0.259</td><td>0.243</td><td>0.787</td></tr><tr><td>1—Semantic WER</td><td>0.058</td><td>0.143</td><td>0.834</td></tr><tr><td>Flat retention</td><td>0.349</td><td>0.441</td><td>0.895</td></tr><tr><td>Shapley retention</td><td>0.399</td><td>0.476</td><td>0.902</td></tr><tr><td>Native projection</td><td>0.466</td><td>0.475</td><td>0.870</td></tr></table>

## 6.3. What does execution-free diagnosis reveal?

We next test a mechanism-specific operating point. Before observing intervention outcomes, we selected the 15 WCB tasks with the most task-relevant workspace content because bias lists are extracted from that workspace. As shown in Table5, Ontology repair improves Parakeet execution by 11.85% without biasing and 16.09% with biasing; biasing itself has no consistent benefit. The same direction appears in Shapley and Native projection. Across all 50 tasks, ontology repair still recovers substantial interface-level value and joint grader logic, while end-to-end execution changes by less than 0.3%. Thus the targeted probe exposes the operational effect where the intervention is relevant, whereas full-cohort retention isolates an interface improvement obscured by downstream execution variability.

Table 5. Parakeet diagnosis on WCB. Targeted split comprises the 15 non-Safety tasks in the preselected 16-task workspace-intensive probe. Full comprises all 50 non-Safety WCB tasks. Bold indicates improvements from ontology repair under matched biasing. All values are percentages.
<table><tr><td>Cohort</td><td>Biasing</td><td>Ont. fix</td><td>Shapley</td><td>Native</td><td>Execution</td></tr><tr><td rowspan="4">Targeted (n = 15)</td><td>x</td><td>x</td><td>39.93</td><td>14.00</td><td>40.60</td></tr><tr><td>x</td><td>√</td><td>69.33</td><td>34.00</td><td>52.45</td></tr><tr><td>√</td><td>x</td><td>41.30</td><td>14.67</td><td>33.97</td></tr><tr><td>√</td><td>√</td><td>59.25</td><td>33.67</td><td>50.06</td></tr><tr><td rowspan="4">Full (n = 50)</td><td>x</td><td>x</td><td>33.39</td><td>7.73</td><td>22.64</td></tr><tr><td>x</td><td>√</td><td>54.84</td><td>23.24</td><td>22.68</td></tr><tr><td>√</td><td>x</td><td>33.44</td><td>7.33</td><td>22.38</td></tr><tr><td>√</td><td>√</td><td>46.30</td><td>16.33</td><td>22.64</td></tr></table>

Table 6. OSWorld execution-free transfer, mean ± SD across five paired Whisper takes. Improvements are percentage points.
<table><tr><td>Measure</td><td>Raw +Ont.</td></tr><tr><td>Whisper  $5 4 . 9 3 \pm 1 . 1 8$ </td><td>∆  $6 2 . 8 9 \pm 2 . 0 5$   $+ 7 . 9 6 \pm 1 . 5 2$ </td></tr></table>

Moreover, Shapley retention and Native projection provide complementary views: Shapley attributes grader value to individual intentions, whereas Native evaluates them under the grader’s original composition. On the full cohort without biasing, ontology repair yields Shapley retention of 54.84% but Native projection of 23.24%, suggesting that many individually valuable requirements are restored but some joint requirements are not satisfied.

Finally, the execution-free evaluation transfers from WCB’s command-line tasks to OSWorld’s graphical tasks. Across five paired Whisper takes, ontology repair improves the intent accuracy with 7.96-point gain (Table 6). The smaller directionally consistent execution gains in Table 3 reflect the additional downstream-agent factors absent from interface-level retention.

## 7. CONCLUSION

We presented Talk2Agent, a human-spoken benchmark and an execution-free evaluation framework for voice interfaces to LLM agents. By projecting source-benchmark graders onto promptaddressable intentions, its Native and Shapley measures complementary diagnosis of task-relevant information loss without repeatedly executing the downstream agent, while remaining operationally grounded through execution-based validation. The study also exposes several open challenges. Long, written prompts do not necessarily reflect natural voice use. Speakers may omit, paraphrase, or interactively resolve URLs, file paths, code, and other structured content rather than verbalize them in full. Moreover, bias lists constructed from workspace did not consistently improve agentic task execution, leaving open how relevant context should be selected and divided between interface-level recognition and agent-level error recovery.

## 8. REFERENCES

[1] Shuyan Zhou et al., “WebArena: A Realistic Web Environment for Building Autonomous Agents,” in Proc. ICLR, Vienna, 2024.

[2] Tianbao Xie et al., “OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments,” in Proc. NeurIPS Datasets and Benchmarks Track, Vancouver, 2024.

[3] Shuangrui Ding et al., “WildClawBench: A Benchmark for Real-World, Long-Horizon Agent Evaluation,” arXiv preprint arXiv:2605.10912, 2026.

[4] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan, “τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains,” in Proc. ICLR, Singapore, 2025.

[5] Emanuele Bastianelli, Andrea Vanzo, Pawel Swietojanski, and Verena Rieser, “SLURP: A Spoken Language Understanding Resource Package,” in Proc. EMNLP, 2020.

[6] Paden Tomasello et al., “STOP: A dataset for Spoken Task Oriented Semantic Parsing,” in Proc. SLT, Doha, 2023.

[7] Huanzhi Mao et al., “BFCL Audio: An Audio Function Calling Evaluation for Large Language Models,” in Proc. ICML, Seoul, 2026.

[8] Dhruv Jain, Harshit Shukla, Gautam Rajeev, Ashish Kulkarni, Chandra Khatri, and Shubham Agarwal, “VoiceAgentBench: Are Voice Assistants ready for agentic tasks?,” arXiv preprint arXiv:2510.07978, 2025.

[9] Ramit Pahwa et al., “Audio2Tool: Speak, Call, Act – A Dataset for Benchmarking Speech Tool Use,” in Proc. Interspeech, Sydney, 2026.

[10] Tara Bogavelli et al., “EVA-Bench: A New End-to-end Framework for Evaluating Voice Agents,” arXiv preprint arXiv:2605.13841, 2026.

[11] Soham Ray, Keshav Dhandhania, Victor Barres, and Karthik Narasimhan, “τ-Voice: Benchmarking Full-Duplex Voice Agents on Real-World Domains,” arXiv preprint arXiv:2603.13686, 2026.

[12] Md Tahmid Rahman Laskar, Xue-Yong Fu, Seyyed Saeed Sarfjoo, Quinten McNamara, Jonas Robertson, and Shashi Bhushan TN, “From Text to Voice: A Reproducible and Verifiable Framework for Evaluating Tool Calling LLM Agents,” arXiv preprint arXiv:2605.15104, 2026.

[13] Suyoun Kim et al., “Semantic Distance: A New Metric for ASR Performance Analysis Towards Spoken Language Understanding,” in Proc. Interspeech, Brno, 2021.

[14] Janine Rugayan, Giampiero Salvi, and Torbjørn Svendsen, “Perceptual and Task-Oriented Assessment of a Semantic Metric for ASR Evaluation,” in Proc. Interspeech, Dublin, 2023.

[15] Somnath Roy, “Semantic-wer: A unified metric for the evaluation of asr transcript for end usability,” arXiv preprint arXiv:2106.02016, 2021.

[16] Pipecat-AI, “Pipecat STT Benchmark,” Software, 2026, Opensource speech-to-text benchmarking toolkit; accessed 2026- 09-02.

[17] Ryan Whetten and Casey Kennington, “Evaluating and Improving Automatic Speech Recognition using Severity,” in Proc. ACL BioNLP Workshop, Toronto, 2023.

[18] Sujith Pulikodan, Sahapthan K, Prasanta Kumar Ghosh, Visruth Sanka, and Nihar Desai, “An approach to measuring the performance of Automatic Speech Recognition(ASR) models in the context of Large Language Model(LLM) powered applications,” in Proc. Interspeech, Rotterdam, 2025.

[19] Zixuan Jiang, Binghao Qiang, Jiaying Chi, Yanqiao Zhu, Kai Yu, and Xie Chen, “AgenticASR: Refining Speech Recognition in Real-World Scenarios via an Agentic Approach,” arXiv preprint arXiv:2607.28175, 2026.

[20] Zitha Sasindran, Harsha Yelchuri, and T. V. Prabhakar, “Se-MaScore: A new evaluation metric for automatic speech recognition tasks,” in Proc. Interspeech, Kos, 2024.

[21] Thibault Baneras-Roux et al., “Evaluation of Automatic˜ Speech Recognition Using Generative Large Language Models,” arXiv preprint arXiv:2604.21928, 2026.

[22] Golan Pundak, Tara N. Sainath, Rohit Prabhavalkar, Anjuli Kannan, and Ding Zhao, “Deep Context: End-to-End Contextual Speech Recognition,” in Proc. SLT, Athens, 2018.

[23] Guangzhi Sun, Xianrui Zheng, Chao Zhang, and Philip C. Woodland, “Can Contextual Biasing Remain Effective with Whisper and GPT-2?,” in Proc. Interspeech, Dublin, 2023.

[24] Christian Huber and Alexander Waibel, “How to Recognize New Words: A Comparison Between Context Biasing Methods and Speech LLMs,” arXiv preprint arXiv:2608.05759, 2026.

[25] Typeless, “Typeless: AI Voice Dictation,” 2026, Commercial AI voice-dictation system; accessed 2026-09-10.

[26] Chen Chen, Yuchen Hu, Chao-Han Huck Yang, Sabato Marco Siniscalchi, Pin-Yu Chen, and Eng Siong Chng, “HyPoradise: An Open Baseline for Generative Speech Recognition with Large Language Models,” in Proc. NeurIPS Datasets and Benchmarks Track, New Orleans, 2023.

[27] Gan Song et al., “Contextual Spelling Correction with Large Language Models,” in Proc. ASRU, Taipei, 2023.

[28] Solee Im, Wonjun Lee, JinMyeong An, Yunsu Kim, Jungseul Ok, and Gary Geunbae Lee, “DeRAGEC: Denoising Named Entity Candidates with Synthetic Rationale for ASR Error Correction,” in Proc. ACL Findings, Vienna, 2025.

[29] Michel Grabisch, Jean-Luc Marichal, and Marc Roubens, “Equivalent Representations of Set Functions,” Mathematics ofOperations Research, vol. 25, no. 2, pp. 157–178, 2000.

[30] Lloyd S. Shapley, “A Value for n-Person Games,” in Contributions to the Theory ofGames II, vol. 28 of Annals ofMathematics Studies, pp. 307–317. Princeton University Press, 1953.

[31] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine Mcleavey, and Ilya Sutskever, “Robust Speech Recognition via Large-Scale Weak Supervision,” in Proc. ICML, Honolulu, 2023.