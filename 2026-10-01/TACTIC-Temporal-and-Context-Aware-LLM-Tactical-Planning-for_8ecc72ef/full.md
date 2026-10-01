# TACTIC: Temporal and Context-Aware LLM Tactical Planning for Roadside LiDAR Attacks

Yiming Gao<sup>1</sup> and Shaocheng Luo<sup>2</sup>\*

Abstract— Physical LiDAR attacks are often evaluated using fixed primitives and manually selected parameters, despite their strong dependence on surrounding traffic. We present TAC-TIC, a scene-aware framework that uses a multimodal large language model (MLLM) to coordinate state-adaptive roadside LiDAR attacks. Under a gray-box threat model, TACTIC relies only on an attacker-operated roadside perception stack, without accessing the victim LiDAR’s native point clouds or internal processing. Local perception provides metric vehicle states, while the MLLM combines these measurements with roadside imagery to infer relational traffic context and construct a semantic scene graph. Based on this representation, TACTIC selects and configures two complementary primitives: push-away, which shifts the perceived range of a lead vehicle, and phantomobstacle braking, which triggers emergency braking through obstacle injection. Measured traffic states and empirically calibrated constraints ground the generated tactics in physically feasible operating regions. To accommodate MLLM latency, TACTIC overlaps reasoning and execution asynchronously while high-rate local perception detects scene changes and triggers replanning. Across 280 randomized CARLA trials, the full policy achieves a 100% collision rate, compared with 35% for a fixed rule, 60% for random selection, and 75% for a restricted LLM using mode selection with default parameters. Joint physical-and-image input achieves 100% success, versus 65% with physical measurements alone and 75% with imagery alone, while asynchronous ∆ refresh reduces scene-mutation response from 7.4 s to 2.0 s. These results show that scenedependent tactical planning can expose context-sensitive LiDAR failure modes that fixed attack policies may miss.

## I. INTRODUCTION

LiDAR is widely used in autonomous driving but remains vulnerable to physical attacks that suppress genuine returns, inject phantom objects, and induce unsafe vehicle behavior [1], [2], [3]. Studying these attacks in controlled environments is essential for identifying safety-critical failure modes, evaluating countermeasures, and improving protection for passengers and other road users.

Existing physical LiDAR attacks mainly demonstrate individual attack primitives. A target may be removed through spoofing [4], while a nonexistent obstacle may be introduced through point injection [2]. However, these attacks are typically evaluated using manually defined rules and fixed parameters in simplified traffic settings, such as straight-road driving [5], predefined turning scenarios [6], or pre-assumed vehicle maneuvers [7]. Such evaluations establish whether an individual primitive can work, but provide limited insight into how attack effectiveness changes with surrounding traffic scenes.

We focus on vehicle-to-vehicle interactions because they directly expose these dynamic dependencies and are central to autonomous-driving safety. Unlike collisions with walls or road dividers, which are dominated by static geometry and may additionally depend on map availability, localization, or lateral control, interactions among vehicles depend on continuously evolving traffic relationships. A nearby vehicle may be the target’s leader or follower in the same lane, or an unrelated participant in an adjacent lane; the same perturbation can therefore produce very different outcomes. Physical attack execution is consequently a scene-dependent planning problem that requires reasoning over traffic topology, spacing, relative motion, acceleration, and time-tocollision (TTC), rather than applying a fixed primitive or parameter set.

We address this gap with TACTIC, a temporal and context-aware framework for state-adaptive, LLMorchestrated roadside LiDAR attacks, illustrated in Fig. 1. Under a gray-box threat model, the roadside attacker independently observes traffic without accessing the victim LiDAR’s native point clouds, filtering parameters, or internal processing. TACTIC combines object-level local perception with roadside imagery and uses a multimodal large language model (MLLM) for relational scene reasoning and contextdependent tactical planning.

Specifically, local perception detects and tracks individual vehicles and provides measurable physical states such as position, velocity, spacing, and TTC, while the MLLM jointly considers these measurements and visual road context to infer same-lane, adjacent-lane, and leader–follower relationships. These relations form a semantic scene graph for tactical decision making. Based on this graph, TACTIC selects and configures two complementary collision primitives: a push-away attack, which shifts the preceding vehicle’s perceived range, and a phantom-obstacle braking attack, which triggers emergency braking through obstacle injection. Because the primitives exploit different vehicle relationships and feasibility regions, their effectiveness depends on the current scene.

Introducing an LLM into the loop creates two additional challenges: generated tactics may violate physical feasibility, and LLM inference may lag behind rapidly changing traffic. TACTIC addresses the former using measured vehicle states, calibrated operating bounds, and deterministic verification, leaving the MLLM to focus on relational scene understanding and tactical selection. To address latency, reasoning and execution overlap asynchronously, while high-rate local perception detects scene changes and triggers replanning only when needed.

![](images/eff40731273c97dc76c5db515e7910cb277a00c9fd9a2bbf79d0d8033b9459e0.jpg)  
Fig. 1: TACTIC overview. The attacker-operated roadside perception stack observes local traffic with co-located LiDAR and camera and provides sensing state S to the multimodal LLM agent. The agent constructs a semantic scene graph, selects a feasible tactic P, and updates the policy when significant scene changes occur. Cached topology and asynchronous reasoning reduce repeated scene generation and replanning latency.

We evaluate TACTIC in CARLA using randomized multivehicle scenarios, a realistic driver model, and a two-stage automatic emergency braking controller. Across 280 primary CARLA trials spanning feasibility, policy, perception, timing, and benign-control evaluations, the full LLM policy achieves a 100% collision rate, compared with 35% for a fixed rule and 60% for random selection. A restricted LLM that selects the attack mode but uses default parameters reaches 75%, showing that joint mode and parameter adaptation provides additional benefit. Joint physical and visual input sustains 100% success, whereas physical-only and image-only inputs drop to 65% and 75%, respectively. Physical calibration reveals a sharp push-away feasibility transition near 10.5– 11 m, while asynchronous replanning reduces scene-change response latency from 7.4 s to 2.0 s. The project website<sup>1</sup> is accessible, and the source code<sup>2</sup> is released.

The main contributions are as follows:

• We formulate physical LiDAR attack execution as a scene-dependent tactical planning problem, focusing on dynamic vehicle-to-vehicle interactions whose feasibility depends on evolving relational traffic context.

• We develop TACTIC, which combines object-level local perception and roadside imagery with multimodal LLM reasoning to construct semantic scene graphs and select complementary physical attack primitives and parameters.

• We ground LLM tactical decisions with measured vehicle states and calibrated physical constraints and introduce an asynchronous perception–LLM architecture for high-rate scene-change detection and replanning. Experiments demonstrate higher collision success than fixed and random policies and faster adaptation to changing scenes.

## II. RELATED WORK

## A. Physical LiDAR Attacks on Autonomous Vehicles

Physical LiDAR attacks largely follow two directions. Removal attacks suppress genuine target returns through adversarial objects or physical spoofing [1], [8], [4], [9], whereas injection attacks introduce nonexistent obstacles to manipulate downstream perception and control [2], [10]. Recent work improves operational realism by attacking moving targets at longer range [5] and extending physical manipulation to localization, odometry, and sensor fusion [6], [7]. A recent systematization further characterizes how sensor perturbations propagate through autonomous-driving pipelines [3].

Despite these advances, attack execution remains largely scenario-specific: individual primitives are commonly evaluated with predefined conditions and parameters tailored to particular road configurations or maneuvers [5], [6], [7]. How to select and configure physical attacks as multi-vehicle traffic evolves has received substantially less attention. TACTIC addresses this gap by formulating physical attack execution as a scene-dependent tactical planning problem.

## B. Context-Aware Attack Planning

Context-dependent coordination of physical sensor attacks remains comparatively unexplored, with existing approaches typically executing individual attacks independently or using handcrafted rules that map predefined conditions to fixed actions. Attack-hardness analysis [11] shows that physical attack feasibility depends strongly on situation-specific conditions, but identifies these conditions primarily through offline search. LLM-based adversarial scenario generation [12] instead varies simulated driving scenarios rather than coordinating online physical attacks. Zhang et al. [3] further identify joint perception–decision attacks and scene-aware hybrid attack chains as open directions.

TACTIC moves this reasoning online: measured traffic states and calibrated physical constraints define the feasible action space, while relational scene understanding guides context-dependent selection and parameterization of complementary physical attack primitives.

## C. LLM-Based Scene and Tactical Reasoning

LLMs and multimodal LLMs (MLLMs) are increasingly used for high-level reasoning in autonomous driving. GPT-Driver reformulates motion planning as language modeling over structured driving states [13]; LMDrive integrates multimodal sensor observations and language for closed-loop driving [14]; and DriveLM introduces graph-structured visual reasoning across perception, prediction, and planning [15]. These studies demonstrate the potential of language models to reason over heterogeneous observations and relationships among traffic participants.

TACTIC extends this capability to physical-attack planning: local perception provides metric states, the MLLM infers relational context and selects tactics, while physical feasibility and execution remain deterministic. To reduce latency, TACTIC overlaps reasoning with execution following TypeFly’s generate-while-execute principle [16].

## III. SCENE-DEPENDENT ATTACK FORMULATION

We focus on vehicle-to-vehicle interactions and stress tests because they require reasoning over dynamic traffic relationships, whereas collisions with walls or road dividers are dominated by static geometry and may additionally depend on map availability, localization, and lateral control. These factors can confound the study of physical LiDAR attacks and obscure the role of surrounding traffic. Our work provides a direct setting for studying how dynamic relationships, including leader–follower structure, spacing, and relative motion, affect attack feasibility.

This objective makes physical attack execution inherently scene dependent. As shown in Fig. 1, a nearby vehicle may be the target’s leader or follower in the same lane, or an unrelated participant in an adjacent lane. The same physical primitive can therefore succeed or fail depending on the current traffic topology and vehicle dynamics. We formulate this scene-dependent attack space below; Sec. IV then introduces TACTIC to reason over the scene and select among feasible attack policies.

## A. Threat Model and Attack Objective

We consider a roadside attacker equipped with LiDAR and an RGB camera, following a deployment setting similar to the Moving Vehicle Spoofing System in [5]. The roadside perception stack detects and tracks surrounding vehicles at 20 Hz. Vehicles follow an intelligent driver model [17], [18] together with a two-stage AEB controller that applies partial and full braking at perceived TTC thresholds of 3.0 s and 1.6 s, respectively [19].

We adopt a gray-box threat model in which the attacker interacts with the victim only through the physical sensing channel. The attacker independently observes the traffic scene and may know relevant sensor and vehicle-control configurations, but has no access to the victim’s native point clouds, internal filtering states, or vehicle messages. A target vehicle T is designated, and the objective is to induce a collision between $T$ and a neighboring traffic participant.

Let $S _ { t }$ denote the roadside observation at time t, comprising vehicle positions, velocities, accelerations, inter-vehicle spacing, TTC, and the corresponding roadside image. Because these object-level measurements do not by themselves encode how traffic participants relate to one another, Sec. IV further interprets $S _ { t }$ into a relational traffic context for tactical decision making. At each decision round, the policy outputs

$$
\mathcal { P } _ { t } = ( m , \theta _ { m } ) ,\tag{1}
$$

where m ∈ {push-away,phantom-braking,hold} denotes the selected physical attack primitive and $\theta _ { m }$ contains its mode-specific continuous parameters. Specifically, $\theta _ { m } = ( \Delta d , \rho , o )$ for push-away and $\theta _ { m } = \left( w , \tau \right)$ for phantom-braking, as defined in Sec. III-B. A round corresponds to one decision–execution–replanning cycle of the policy in Sec. IV, and each trial is limited to at most R = 3 such rounds.

## B. Complementary Physical Attack Primitives

Pointcloud push-away: Consider a target T following a lead vehicle A1, with another vehicle A2 behind T in the same lane. The attacker shifts the perceived range of A1, causing T to perceive a larger leading gap while the true T– A1 distance contracts. This primitive, denoted push-away, is parameterized by the induced range displacement ∆d, separation ramp ρ, and ignition onset o.

Phantom-obstacle braking: The second primitive injects a virtual obstacle at distance w ahead of T. When its perceived TTC falls below the full-braking threshold, T performs emergency braking and the trailing vehicle A2 may become the collision agent. This primitive, denoted phantom-braking, is parameterized by phantom wall placement distance w and emission duration τ.

Hold: The third primitive hold means keeping the previous primitive $\mathcal { P } _ { t - 1 }$ . This primitive avoids frequent policy changes during attacking.

The three primitives are complementary because they exploit different traffic relationships. push-away depends primarily on the T–A1 leader–follower geometry, whereas braking exploits the A2–T relationship and the target’s deceleration. This complementarity motivates scene-aware tactical selection rather than a fixed attack rule.

## C. Physical and Temporal Feasibility

The available action space is further bounded by physicsinformed, empirically calibrated constraints. These constraints determine whether a candidate primitive can operate under the current scene, leaving the scene-aware policy in Sec. IV to choose among feasible alternatives.

Push-away feasibility: The induced displacement must be large enough to produce the required closing behavior while remaining inside roadside sensing coverage: $\Delta d _ { \mathrm { m i n } } =$ 11 m, $\Delta d _ { \mathrm { m a x } } ( o ) = 5 0 - o \textrm { m }$ . The displacement must also develop before the target reaches the lead vehicle while remaining below the measured consistency threshold: $\begin{array} { r } { \frac { \Delta d } { t _ { \mathrm { a r r } } } \leq } \end{array}$ $\rho \leq 3 . 0 ~ \mathrm { m / s }$ , where $t _ { \mathrm { a r r } }$ denotes the remaining arrival time.

![](images/de813ab3884618f1b7bac2b9947dcdb8dcea8223f8fa57545cf957cadce28737.jpg)  
Fig. 2: Multimodal scene-graph generation in TACTIC. Roadside imagery is encoded into visual tokens, while tracked vehicle states are serialized as text tokens. The MLLM jointly conditions on both modalities to infer semantic traffic relations and decodes them as a structured scene graph. When the topology remains unchanged, the steadystate manager updates only metric edge attributes without full multimodal regeneration.

Braking feasibility: For the braking primitive, the injected obstacle must drive perceived TTC below the 1.6 s full-braking threshold. We therefore place the virtual wall according to $w ( \nu ) = 1 . 5 \nu , w \in [ 5 , 1 5 ]$ m, where v is the measured target speed. The emission duration is calibrated to cover both the target’s braking response and the follower’s subsequent approach.

Temporalfeasibility: Feasibility depends jointly on traffic geometry and execution time. Waiting can enlarge or shrink relevant inter-vehicle gaps, consume the remaining push-away displacement envelope, or change whether a trailing vehicle can reach the target after braking. Attack timing is therefore determined from the continuously measured spacing, TTC, and vehicle motion rather than by a fixed schedule. Sec. IV describes how TACTIC combines these feasibility conditions with relational scene understanding for state-adaptive tactical planning.

## IV. TACTIC: SCENE-AWARE LLM TACTICAL PLANNING

TACTIC operationalizes the scene-dependent attack space defined in Sec. III. Its design separates three responsibilities: local perception measures the traffic state, a multimodal LLM performs relational scene understanding and tactical selection, and deterministic modules enforce the physical constraints in Sec. III-C and execute the selected primitive. Because perception operates much faster than LLM inference, reasoning and execution proceed asynchronously, with a high-rate local monitor triggering re-planning when the cached scene becomes invalid.

## A. Multimodal Semantic Scene Reasoning

The perception front end operates at the LiDAR’s native 20 Hz rate, clustering detections and associating them into persistent tracks. It provides object-level measurements including longitudinal and lateral position, velocity, acceleration, spacing, and TTC. These measurements accurately describe individual vehicle states but do not explicitly encode their semantic relationships.

To recover this relational context, each MLLM query combines the structured vehicle states with the corresponding roadside camera frame. The image provides complementary road semantics, such as lane structure and driving direction, while the structured measurements retain precise metric information. From these inputs, the MLLM constructs a semantic scene graph

$$
G _ { t } = ( V _ { t } , E _ { t } ) ,\tag{2}
$$

where nodes $V _ { t }$ represent tracked traffic participants and edges $E _ { t }$ encode relations such as same-lane leader–follower and adjacent- or opposite-lane interactions. Edge attributes retain metric quantities supplied by the perception stack, including spacing, relative motion, and TTC. This division keeps physical-state estimation outside the MLLM while assigning it the higher-level task of interpreting how traffic participants relate to one another [20], [21].

MLLM scene-graph generator: Figure 2 illustrates the multimodal generation process. The roadside image I<sub>t</sub> is patchified and encoded by a SigLIP-2 vision transformer, followed by a lightweight merger that compresses each $2 \times 2$ block of patch embeddings into a visual token sequence,

$$
Z = { \mathrm { M e r g e r } } ( { \mathrm { V i T } } ( I _ { t } ) ) = \{ z _ { 1 } , \dots , z _ { M } \} .\tag{3}
$$

The tracked vehicle states, output schema, and cached topology are serialized into text tokens $X = \{ x _ { 1 } , \ldots , x _ { n } \}$ . Visual and text tokens are jointly processed by the multimodal decoder using interleaved MRoPE positions, while mid-level visual features are injected into early decoder layers through DeepStack. This allows relational fields to condition jointly on visual road context and precise object-state measurements.

The resulting graph is autoregressively decoded into a constrained JSON representation. Letting $y _ { 1 } , \ldots , y _ { K }$ denote its serialized output tokens,

$$
p \mathopen { } \mathclose \bgroup \left( G _ { t } \aftergroup \egroup \right) = \prod _ { k = 1 } ^ { K } p \mathopen { } \mathclose \bgroup \left( y _ { k } \mid y _ { < k } , Z , X \aftergroup \egroup \right) .\tag{4}
$$

Semantic fields, such as lane association and leader–follower relations, are inferred from the multimodal context, whereas metric fields retain measurements supplied by the local perception stack. This separation prevents the MLLM from replacing low-level state estimation while allowing it to resolve traffic relationships needed for tactical planning.

Scene-graph generation runs asynchronously following the generate-while-execute principle of TypeFly [16]. The initial query performs full multimodal scene interpretation. During steady state, subsequent queries reuse the cached graph and recent object states. If the vehicle set and relational topology remain unchanged, the ∆ path refreshes only dynamic edge attributes such as spacing and TTC; otherwise, a vehicle-set consistency check triggers full multimodal regeneration. The updated graph atomically replaces the cached topology once generation completes.

## B. Physics-Grounded Tactical Policy

Given the semantic graph $G _ { t }$ , TACTIC determines which physically feasible primitive best matches the current traffic configuration and how its continuous parameters should be configured. The key design principle is to separate consequence, feasibility, and tactical selection rather than asking the LLM to infer all three from scratch.

H-group measure: Before each tactical decision, both primitives are scored using an offline-calibrated h-group measure. For the rear-end primitive, normalized gap, acceleration, and TTC terms use weights 0.15/0.70/0.15, with attack duration introduced at a second combination level with weight 0.10. For braking, acceleration and TTC use weights 0.85/0.15, again with a 0.10 duration term. Gap is omitted from the braking score because braking severity is driven by the target’s deceleration rather than a monotonic function of inter-vehicle distance. Grid-search calibration yields mean attack-to-benign separation margins of 96.5× and 13.6× for the rear-end and braking measures, respectively. The score is defined from the victim’s perspective: larger values indicate more severe consequences and do not encode attacker cost or feasibility.

Feasibility grounding: A high h-group score does not imply that the corresponding primitive is physically executable. TACTIC therefore supplies the LLM with the measured traffic state together with the mode-specific feasibility bounds derived in Sec. III-C. For push-away, these include the minimum and coverage-limited displacement, $1 1 \leq \Delta d \leq$ $5 0 - o$ , and the admissible ramp band, $\Delta d / t _ { \mathrm { a r r } } \le \rho \le 3 . 0 \ \mathrm { m / s }$

For braking, the wall placement follows the measured target speed, $w ( \nu ) = 1$ .5v within [5,15] m, while the minimum emission duration depends on the current follower gap. These values are recomputed from the current scene and provided numerically at every tactical decision.

The LLM therefore reasons over three complementary inputs: relational scene structure, potential consequence, and remaining feasibility margin. A lower-h-group primitive may be selected when the traffic geometry makes it substantially more feasible. Before execution, deterministic code projects the generated parameters back into the admissible set $\Phi ( S _ { t } )$ , ensuring that physical bounds are system constraints rather than prompt suggestions. In short, measured states and calibrated constraints determine what can be executed, while the LLM determines which tactic best matches the scene.

Tactical policy: At round t, the LLM produces $\mathcal { P } _ { t } =$ $\left( m , \theta _ { m } \right)$ , where m and $\theta _ { m }$ are defined in Sec. III-A. The full policy jointly selects the primitive and its parameters and additionally returns an estimated success likelihood and a short rationale used for logging and analysis. Execution may continue for at most $R = 3$ rounds, allowing the tactic to be revised when the previous round fails or the traffic relationship changes.

Attack timing is handled separately from tactical selection. The high-rate controller continuously evaluates spacing, TTC, and vehicle motion against the feasible spatio-temporal region defined in Sec. III-C. It determines whether the cached policy remains executable, should be launched, or must be suspended and reconsidered. The LLM therefore selects and configures the tactic, but does not replace high-rate temporal monitoring.

Decision sources: To isolate the contribution of LLM tactical reasoning, we compare four decision sources in Sec. V. llm policy jointly selects the primitive and its mode-specific continuous parameters, returning the primitive choice m together with its continuous parameters $( ( \Delta d , \rho , o )$ for push-away, (w, τ) for the phantom wall), with a success estimate and rationale; llm selects only the primitive in a single shot without the policy layer, with the continuous parameters fixed at system defaults; rule always selects push-away and draws its parameters at random once per trial, exempt from the analytic floors, and reads nothing; and random commits to one uniformly sampled tactic and parameterization for the trial. The random group is not resampled after each round, avoiding multiple independent chances that would confound tactical quality.

## C. Asynchronous Scene Monitoring and Replanning

LLM inference takes seconds, whereas the traffic state evolves at the perception rate. TACTIC therefore decouples high-rate scene monitoring from lower-rate semantic and tactical reasoning. While a cached policy is valid, execution continues concurrently with generation of the next scene interpretation. A lightweight local alerter compares the 20 Hz tracker state against the cached graph on every frame.

A scene mutation is declared when a vehicle appears or disappears, an inter-vehicle gap changes by more than 8 m or 40%, or the inferred road context changes. Because ordinary car-following gaps evolve gradually, these thresholds target structural scene changes rather than frame-level noise. A minimum 0.5 s replanning interval prevents repeated alerts from saturating the LLM scheduler.

When a mutation invalidates the current tactical assumptions, the alerter suspends execution and triggers a scene update. If the vehicle set and relational structure remain valid, the lightweight ∆ path updates the graph; otherwise, a full multimodal regeneration is requested. Tactical planning then resumes from the updated graph and feasibility state. The alerter only detects and localizes scene changes— semantic interpretation and policy regeneration remain the responsibility of the LLM.

This architecture allows perception to react at sensor rate without requiring the LLM to operate at 20 Hz. The resulting latency and scene-change response are evaluated in Sec. V.

## V. EXPERIMENTAL EVALUATION

We evaluate TACTIC along three dimensions: (i) whether the physical constraints in Sec. III-C correctly characterize feasible operating regions; (ii) whether LLM-based tactical planning improves attack effectiveness and efficiency over fixed and random policies; and (iii) how multimodal scene reasoning and asynchronous replanning affect performance under changing traffic conditions. We first establish the experimental setting and physical operating bounds, then evaluate tactical decision quality, and finally isolate the contribution of individual system components.

![](images/bd3e86ed219758637e944285d39d530d415abb1d3f4da4ebf73b94f22a3dc1f4.jpg)  
Fig. 3: Example phantom-obstacle braking attack. At T<sub>1</sub> the system monitors a nominal scene; execution begins at $T _ { 2 } =$ 8.0 s and collision occurs at $T _ { 3 } = 1 0 . 4 \ : \mathrm { s }$ . Windows $\mathrm { A } { - } \mathrm { C }$ show the driving scene, target perception, and scene graph.

## A. Experimental Setup

We implement TACTIC in CARLA 0.9.14 [22] using Town07 with synchronous simulation at 20 Hz, as illustrated in Figs. 3 and 4. Each trial contains the target T, a lead vehicle A1, a follower A2, and two additional traffic participants A3/A4. To isolate dynamic vehicle interactions from occlusion, we target the lane closest to the roadside perception stack. The attacker uses pole-mounted LiDAR and an RGB camera in a setting similar to MVS [5], with an effective observation range of approximately 50 m. The longitudinal vehicles cruise at 6.0 m/s nominally (closedloop ≈ 5.5 m/s), maintaining approximately stationary gaps without attack. Qwen [23] serves as the primary MLLM backend.

Monte Carlo protocol: The initial T–A1 gap is sampled uniformly from 18–28 m, spanning the measured push-away feasibility transition. Each configuration contains 20 trials with distinct seeds and at most three attack rounds. Success requires a physical collision, defined as a center-to-center distance below 5.0 m with residual impact speed of at least 1.5 m/s. We additionally run 20 attack-free control trials for 30 s and report Wilson 95% confidence intervals and Fisher’s exact test. Relevant secondary metrics are t<sub>90</sub>, the time to reach 90% of commanded displacement; d<7, trials with minimum T–A1 distance below 7 m; $R { < } 1 .$ , displacementtracking RMSE below 1 m; emission Gini G; and dose efficiency η. Together, these metrics capture not only collision success but also physical execution quality and attack efficiency.

## B. Feasibility Characterization and Attack Effectiveness

We first validate the physical constraints used to ground tactical decisions. These experiments establish the operating regions within which the LLM is allowed to reason, after which we evaluate whether scene-aware policy selection improves attack effectiveness.

Push-away feasibility: A locked-displacement sweep reveals a sharp transition: ∆d = 5 and 10 m succeed in 0/20 and 5/20 trials, whereas 15 and 20 m both succeed in 20/20 (Table I). A finer sweep gives 1/20 at 8 m, 0/20 at 9 m, 5/20 at 10 m, 12/20 at 10.5 m, and 20/20 at 11–12 m, motivating the sufficient floor $\Delta d _ { \operatorname* { m i n } } = 1 1$ 1 m. The upper bound follows sensing coverage, $\Delta d _ { \mathrm { { m a x } } } = 5 0 - o ;$ beyond the sufficient floor, larger displacement mainly increases exposure rather than improving success.

TABLE I: Locked-displacement response of the push-away primitive (20 trials per level; ramp 3.0 m/s).
<table><tr><td>∆d/m</td><td>Succ.</td><td> $d { < } 7$ </td><td> $R { < } 1$ </td><td> $t 9 0 \mathrm { / s }$ </td><td> $\Delta d / \mathrm { m }$ </td><td>Succ.</td><td> $d { < } 7$ </td><td> $R { < } 1$ </td><td> $t 9 0 \mathrm { / s }$ </td></tr><tr><td>5</td><td>0/20</td><td>0/20</td><td>0/20</td><td>5.83</td><td>15</td><td>20/20</td><td>20/20</td><td>18/20</td><td>4.87</td></tr><tr><td>10</td><td>5/20</td><td>20/20</td><td>9/20</td><td>4.30</td><td>20</td><td>20/20</td><td>20/20</td><td>6/20</td><td>6.20</td></tr></table>

The ramp must also complete before contact while remaining below the tracking-consistency limit. Instantaneous steps are detected in 10/10 trials and ramps above 3.0 m/s in 3/3, whereas no tested ramp at or below 3.0 m/s is flagged (0/10). Together with $\rho \ge \Delta d / t _ { \mathrm { a r r } }$ , this defines the feasible band in Fig. 5. The full policy operates at $\Delta d \approx 1 7 . 4$ m and $\rho \approx 2 . 9 7$ m/s, placing its typical operating point safely inside this region.

Phantom-braking feasibility: We next characterize the complementary braking primitive. A locked sweep over emission duration at a fixed 13–15 m A2–T gap shows no clear duration threshold: success remains high from 2–10 s. Thus, within this tested range, the binding factor is the follower gap rather than emission duration. The virtual wall follows $w ( \nu ) = 1 . 5 \nu$ , keeping perceived TTC below the 1.6 s full-braking threshold. These results confirm that the two primitives are limited by different aspects of scene geometry and therefore occupy distinct state-dependent feasibility regions.

Dose response: Figure 6 summarizes the resulting dose–response behavior of both primitives. Push-away success rises sharply around 10.5–11 m and saturates thereafter, validating the 11 m operating floor. In contrast, phantombraking remains at or above 80% success (16/20–20/20) across 2–10 s emission durations at a fixed 13–15 m gap, further indicating that follower geometry rather than duration is the dominant constraint. The contrasting response profiles motivate selecting the primitive according to the current traffic state rather than applying one fixed policy.

Decision-source comparison: Having established the feasible action space, we next evaluate how different decision mechanisms use it. We compare four decision sources under the hybrid implementation, where both push-away and phantom-braking are available: the full LLM policy, restricted LLM, fixed rule, and committed random policy defined in Sec. IV-B. The full policy achieves 20/20 successes, significantly exceeding the fixed rule’s 7/20 and random policy’s 12/20 (Fisher $p = 1 . 3 \times 1 0 ^ { - 5 }$ and $p = 0 . 0 0 3 3 ;$ Table II). The fixed rule always chooses push-away, leaving success dependent on sampled geometry; all 13 failures are AEB-arrested near-misses. Random selection can choose either primitive but cannot jointly adapt its parameters to the observed scene.

The restricted LLM achieves 15/20, significantly below the full policy $( p = 0 . 0 4 7 )$ , indicating that joint mode and parameter selection improves over mode selection alone. The full policy also has the lowest duty cycle (0.39) and emission dose (3.9), showing that adaptive parameterization improves both effectiveness and signal efficiency. Overall, these results support the central design choice of coupling scene-level primitive selection with continuous parameter adaptation.

![](images/49957148e732cff5f9646f2f1e2d7764e47d183181e49dd1c79ea1bae5c2eb25.jpg)  
Fig. 4: Example scene change and replanning. Execution is reconsidered at $T _ { 2 } = 3 . 9 \mathrm { s }$ and $T _ { 4 } = 1 3 . 3 \mathrm { s }$ as traffic evolves, ultimately causing a collision between T and lead vehicle A1.

![](images/61a363285f232166ff9503e94b73a966db6af8893ef9a720bfbe54efd6f00fbd.jpg)  
(a)

![](images/08a4f620049f07a793b1d6739ee7b3cc5e5e236e4c7b964879dd37ee2c31d670.jpg)  
(b)

Fig. 5: Measured push-away feasibility bounds. (a) Displacement range and policy operating points. (b) Ramp rates bounded by completion and the $\rho _ { \mathrm { m a x } } = 3 . 0$ m/s consistency limit.  
![](images/4d891502016d4e5435c7dfb8ea13b0a74dd7b0799528926fe2419bb5c0612ea4.jpg)  
(a)

![](images/13f913cc42dbd66b1b7b0d5b70c5b319bec76d60111cc5e088d2af0480191803.jpg)  
(b)  
Fig. 6: Dose–response of the two primitives (20 trials per level; 95% Wilson CI). (a) push-away success versus relay displacement. (b) phantom-braking success versus emission duration at a fixed 13–15 m gap.

Case studies: The quantitative results above are further illustrated by two representative executions. Figure 3 shows a phantom-braking episode. At $T _ { 2 } = 8 . 0 \mathrm { s } .$ , the $A 2 { - } T$ geometry becomes feasible; the cached policy is released, an 8 m phantom wall triggers emergency braking, and A2 rear-ends T at $T _ { 3 } = 1 0 . 4 \ : \mathrm { s } .$ . Figure 4 shows replanning under changing traffic. A push-away policy becomes feasible at $T _ { 2 } = 3 . 9 \mathrm { s }$ , but lead-vehicle acceleration invalidates the cached conditions and suspends execution. When feasibility is restored at $T _ { 4 } = 1 3 . 3 \mathrm { s } ,$ , the scene graph and policy are updated and execution resumes, eventually causing a $T { \mathrm { - } } A 1$ collision. Together, these examples illustrate how relational scene reasoning, primitive selection, and high-rate feasibility monitoring interact during execution.

Benign control: As a sanity check, no collision occurs in 20 attack-free trials; the minimum separation is 17.9 m and TTC never enters the hazard region. The observed collisions therefore arise from attack execution rather than the nominal traffic dynamics.

## C. Ablation and Runtime Analysis

We finally isolate the contributions of multimodal perception, execution timing, and asynchronous scene updates. These experiments evaluate the components that keep the tactical policy responsive as traffic conditions evolve.

Perception input: Removing either input channel significantly reduces success (Table IIIa): physical-only input reaches 13/20 $( p = 0 . 0 0 8 3 )$ and image-only input 15/20 $( p = 0 . 0 4 7 )$ ), versus 20/20 with joint input. The two ablations are not significantly different from each other $( p = 0 . 7 3 )$ , indicating that structured measurements and visual context provide complementary information. The joint representation therefore improves performance by combining precise metric state with semantic road and traffic context.

Attack timing: We next test whether deliberately waiting for a nominally vulnerable state improves execution. Immediate execution achieves 20/20 successes, versus 9/20 when waiting for a predefined vulnerable window (gap < 12 m and closing speed > 0.5 m/s within 4 s), with Fisher $p = 1 . 5 \times 1 0 ^ { - 4 }$ (Table IIIc). Waiting also increases mean trial time from 34.5 to 43.7 s. Thus, delaying execution can consume the spatio-temporal feasibility envelope before the desired condition is reached; we therefore execute once the current state satisfies the primitive constraints.

Asynchronous replanning: Finally, we evaluate whether asynchronous processing reduces the latency introduced by MLLM reasoning. The reasoning channel makes approximately 2.5 LLM calls per trial. During steady traffic, all ten ∆-path probes correctly preserve the existing topology while updating metric edge attributes. Under injected scene mutations, the local alerter suspends execution within 0.05 s. Synchronous regeneration requires 7.4 s before execution resumes, asynchronous regeneration reduces this to 3.7 s, and asynchronous ∆ refresh further reduces it to 2.0 s when topology remains unchanged (Table IIIb). Decoupling highrate monitoring from lower-rate MLLM reasoning therefore substantially reduces scene-update latency while preserving rapid local response.

Backend generality: To test whether these results depend on a particular MLLM, we repeat both LLM decision groups using Kimi [24]. Qwen [23] achieves 20/20 and 15/20 successes for the full and restricted policies, respectively, while Kimi achieves 20/20 and 16/20. The differences are not significant (Fisher $p = 1 . 0 )$ , suggesting that the observed advantage is not specific to one MLLM backend.

TABLE II: Decision-source comparison under the hybrid implementation (20 trials per group).
<table><tr><td>Decision source</td><td>Attack decision</td><td>Succ. [95% CI]</td><td></td><td> $t { \mathrm { 9 0 } } / { \mathrm { s } }$ </td><td>G</td><td>η</td><td>LLM calls</td><td>Duty</td><td>Dose</td><td>Fisher p</td></tr><tr><td>LLM policy</td><td>LLM-selected mode and  $( \Delta d , \rho , o , w , \tau )$ </td><td>20/20 [84,100]</td><td></td><td>3.6</td><td>0.61</td><td>0.26</td><td>2.1</td><td>0.39</td><td>3.9</td><td>(ref)</td></tr><tr><td>LLM restricted</td><td>LLM-selected mode, default parameters</td><td>15/20 [53,89]</td><td></td><td>3.6</td><td>0.35</td><td>0.08</td><td>2.6</td><td>0.65</td><td>9.7</td><td> $4 . 7 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Fixed rule</td><td>fixed push-away, per-trial random dose</td><td>7/20 [18,57]</td><td></td><td>4.3</td><td>0.33</td><td>0.07</td><td>一</td><td>0.67</td><td>4.7</td><td> $1 . 3 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Random</td><td>committed draw over mode and parameters</td><td>12/20 [39,78]</td><td></td><td>4.2</td><td>0.39</td><td>0.15</td><td>一</td><td>0.62</td><td>4.0</td><td> $3 . 3 \times 1 0 ^ { - 3 }$ </td></tr></table>

TABLE III: Ablation and runtime results.

(a) Perception-input ablation.  
(b) Scene-mutation latency.
<table><tr><td>Input</td><td>Succ. [95% CI]</td><td>Duty</td><td>Dose</td><td>Time/s</td><td>Fisher p</td></tr><tr><td>Physical+image</td><td>20/20 [84,100]</td><td>0.39</td><td>3.9</td><td>34.5</td><td>(ref)</td></tr><tr><td>Physical only</td><td>13/20 [43,82]</td><td>0.49</td><td>3.2</td><td>37.8</td><td> $8 . 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Image only</td><td>15/20 [53,89]</td><td>0.96</td><td>10.0</td><td>20.6</td><td> $4 . 7 \times 1 0 ^ { - 2 }$ </td></tr></table>

<table><tr><td>Configuration</td><td>Latency/s</td></tr><tr><td>Synchronous</td><td>7.4</td></tr><tr><td>Asynchronous</td><td>3.7</td></tr><tr><td>Async. + ∆ refresh</td><td>2.0</td></tr></table>

(c) Attack-timing ablation.
<table><tr><td>Timing</td><td>Succ. [95% CI]</td><td>Time/s</td><td>Fisher p</td></tr><tr><td>Immediate</td><td>20/20 [84,100]</td><td>34.5</td><td>(ref)</td></tr><tr><td>Wait for window</td><td>9/20 [26,66]</td><td>43.7</td><td>1.5×10−4</td></tr></table>

## VI. CONCLUSION

This paper presents TACTIC, a scene-dependent framework for physical LiDAR attack planning in dynamic traffic. TACTIC combines multimodal LLM-based relational reasoning with calibrated physical constraints and asynchronous replanning to select feasible attack tactics from evolving scene context. Across 280 CARLA trials in five ablation experiments, the full policy achieves a 100% collision rate across both primitives, significantly above the fixed-rule (35%) and random (60%) baselines. Joint physical-andimage perception sustains 100%, whereas single-channel ablations drop to 65% and 75%. Asynchronous replanning reduces scene-change response latency from 7.4 s to 2.0 s. These results demonstrate that scene-aware tactical planning can expose failure modes missed by static attack evaluation. Future work will extend TACTIC to hardware platforms, learned perception stacks, and richer sequence-level tactics.

## REFERENCES

[1] Y. Cao, C. Xiao, B. Cyr, Y. Zhou, W. Park, S. Rampazzi, Q. A. Chen, K. Fu, and Z. M. Mao, “Adversarial sensor attack on lidarbased perception in autonomous driving,” in Proceedings of the 2019 ACM SIGSAC conference on computer and communications security, 2019, pp. 2267–2281.

[2] Z. Jin, X. Ji, Y. Cheng, B. Yang, C. Yan, and W. Xu, “Pla-lidar: Physical laser attacks against lidar-based 3d object detection in autonomous vehicle,” in 2023 IEEE Symposium on Security and Privacy (SP). IEEE, 2023, pp. 1822–1839.

[3] Q. Zhang, S. Luo, Z. M. Mao, M. Pajic, and M. K. Reiter, “Sok: How sensor attacks disrupt autonomous vehicles: An end-to-end analysis, challenges, and missed threats,” arXiv preprint arXiv:2509.11120, 2025.

[4] Y. Cao, S. H. Bhupathiraju, P. Naghavi, T. Sugawara, Z. M. Mao, and S. Rampazzi, “You can’t see me: Physical removal attacks on {lidarbased} autonomous vehicles driving frameworks,” in 32nd USENIX security symposium (USENIX Security 23), 2023, pp. 2993–3010.

[5] T. Sato, R. Suzuki, Y. Hayakawa, K. Ikeda, O. Sako, R. Nagata, R. Yoshida, Q. A. Chen, and K. Yoshioka, “On the realism of lidar spoofing attacks against autonomous driving vehicle at high speed and long distance,” in Network and Distributed System Security Symposium (NDSS), 2025.

[6] J. Zhang, S. Cheng, L. Hu, J. Zhang, C. Shi, X. Han, T. Zhang, Y. Cheng, and W. Zhang, “The ghost navigator: Revisiting the hidden vulnerability of localization in autonomous driving,” in 34th USENIX Security Symposium (USENIX Security 25), 2025, pp. 3979–3998.

[7] Z. Song, X. Chen, Z. Zhang, K. Zhang, J. Lu, and W. Li, “Gradientbased adversarial attacks on deep lidar odometry,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 15 188–15 194.

[8] J. Tu, M. Ren, S. Manivasagam, M. Liang, B. Yang, R. Du, F. Cheng, and R. Urtasun, “Physically realizable adversarial examples for lidar object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 13 716– 13 725.

[9] J. Sun, Y. Cao, Q. A. Chen, and Z. M. Mao, “Towards robust lidarbased perception in autonomous driving: General black-box adversarial sensor attack and countermeasures,” in 29th USENIX Security Symposium, 2020, pp. 877–894.

[10] B. Nassi, Y. Mirsky, D. Nassi, R. Ben-Netanel, O. Drokin, and Y. Elovici, “Phantom of the adas: Securing advanced driver-assistance systems from split-second phantom attacks,” in Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security, 2020, pp. 293–308.

[11] H. Kim, R. Bandyopadhyay, M. O. Ozmen, Z. B. Celik, A. Bianchi, Y. Kim, and D. Xu, “A systematic study of physical sensor attack hardness,” in 2024 IEEE Symposium on Security and Privacy (SP). IEEE, 2024.

[12] Y. Mei, T. Nie, J. Sun, and Y. Tian, “Llm-attacker: Enhancing closedloop adversarial scenario generation for autonomous driving with large language models,” arXiv preprint arXiv:2501.15850, 2025.

[13] J. Mao, Y. Qian, J. Ye, H. Zhao, and Y. Wang, “Gpt-driver: Learning to drive with gpt,” arXiv preprint arXiv:2310.01415, 2023.

[14] H. Shao, Y. Hu, L. Wang, G. Song, S. L. Waslander, Y. Liu, and H. Li, “Lmdrive: Closed-loop end-to-end driving with large language models,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 15 120–15 130.

[15] C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beißwenger, P. Luo, A. Geiger, and H. Li, “Drivelm: Driving with graph visual question answering,” in European conference on computer vision. Springer, 2024, pp. 256–274.

[16] G. Chen, X. Yu, N. Ling, and L. Zhong, “Typefly: Low-latency drone planning with large language models,” IEEE Transactions on Mobile Computing, vol. 24, no. 9, pp. 9068–9079, 2025.

[17] A. Kesting, M. Treiber, and D. Helbing, “Enhanced intelligent driver model to access the impact of driving strategies on traffic capacity,” Philosophical Transactions: Mathematical, Physical and Engineering Sciences, pp. 4585–4605, 2010.

[18] Y. Yao and S. Luo, “SEIDM: A safe and efficient intelligent driver model for autonomous driving behavior,” arXiv preprint arXiv:2605.23915, 2026.

[19] European New Car Assessment Programme (Euro NCAP), “Test protocol – aeb/lss vru systems,” v4.5.1, Feb. 2024, 2024.

[20] J. Johnson, R. Krishna, M. Stark, L.-J. Li, D. A. Shamma, M. Bernstein, and L. Fei-Fei, “Image retrieval using scene graphs,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 3668–3678.

[21] D. Xu, Y. Zhu, C. B. Choy, and L. Fei-Fei, “Scene graph generation by iterative message passing,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 5410– 5419.

[22] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, “Carla:

An open urban driving simulator,” in Proceedings of the 1st Annual Conference on Robot Learning (CoRL), 2017, pp. 1–16.

[23] Qwen Team, “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[24] K. Team, Y. Bai, Y. Bao, Y. Charles, C. Chen, G. Chen, H. Chen, H. Chen, J. Chen, N. Chen et al., “Kimi k2: Open agentic intelligence,” arXiv preprint arXiv:2507.20534, 2025.