# TTLab at Daleel 2026: STAR-Ar, Sequence Tagging for Argument Recognition in Arabic

Bhuvanesh Verma, Ali Abusaleh, Alexander Mehler

Text Technology Lab (TTLab),

Goethe University Frankfurt

{verma,a.abusaleh,mehler}@em.uni-frankfurt.de

## Abstract

Argument Mining (AM) is a critical NLP task that remains significantly under-resourced in Arabic. This paper presents STAR-Ar, a BERT-BiLSTM-CRF architecture for argument discourse detection and classification, as our system for Daleel 2026, the inaugural Arabic argument mining shared task. The task requires the identification and classification of argumentative discourse units (ADUs) in debate and editorial texts. We jointly model these two objectives as a token-level sequence labeling task using a BERT-BiLSTM-CRF architecture that combines contextual transformer embeddings with structural transition constraints to support accurate span detection. STAR-Ar achieves an F1-score of 72.69 on validation and 73.7 on test data. Our domain-specific analysis shows that models trained exclusively on editorials underperform those trained on debates, a disparity we primarily attribute to the smaller size of the editorial dataset. The code for STAR-Ar is available at  TTLab at Daleel 2026

## 1 Introduction

Argument Mining (AM) is the automated extraction of argumentative structures and reasoning from natural language, with extensive applications across legal analysis, scientific literature, and public debates (Lawrence and Reed, 2019). While various theoretical frameworks govern AM research (Teufel, 1999; Walton et al., 2008), the Toulmin Model of Argument (Toulmin, 1958) remains a foundational paradigm. In this model, the primary components are claims, premises, and warrants, with the extraction of claims and premises serving as the primary objective for most modern AM systems.

The fundamental building block in these systems is the Argumentative Discourse Unit (ADU), a term formalized by Peldszus and Stede (2013) to describe the minimal atomic text span of an argument. An ADU can vary significantly in length, ranging from a short dependent clause to several consecutive sentences. Consequently, argument extraction inherently requires both precise boundary detection, identifying the exact span of the ADU within a broader text, and subsequent classification to assign that span to a specific argumentative function (e.g., distinguishing between an author’s own claim and a background claim in scientific literature).

Despite significant algorithmic advancements, the vast majority of AM research has been strictly restricted to the English language. Addressing this critical linguistic gap, the Daleel 2026 shared task (Nabhani et al., 2026) introduces a pioneering benchmark for Argument Discourse Mining in Arabic. The shared task is divided into two distinct but related challenges operating over the same text: Task 1 requires paragraph-level multi-label classification to identify all ADU types present within a given text, while Task 2 demands precise token-level span detection and classification of those ADUs. We focus only on Task 2 and utilize a sequence tagging architecture. We employ a spanbased classification approach inspired by Binder et al. (2022) to directly extract and classify ADU boundaries using contextualized representations from a pre-trained Arabic language model. Instead of building a complex system for a severely underresourced and imbalanced dataset, we estbalished a highly stable, mathematically constrained baseline. Our proposed system, STAR-Ar, demonstrates robust cross-domain performance on both debate and editorial texts, ultimately achieving 3rd rank among 14 participating teams in the final Daleel 2026 shared task evaluation.

The paper is organized as follows: Section 2 provides the task description, dataset detail, and related work. Section 3 provides the details regarding the proposed system architecture. In Section 4, we present the experimental setup for the shared task. Section 5 provides a comprehensive analysis of our experimental results alongside ablation studies. Finally, Section 6 concludes the paper.

![](images/e331fbba1ce756ca6e2d8a3274c7fea4f7525639fe56a837ecbcdd5c81a3a19e.jpg)  
Figure 1: Overview of STAR-Ar, a BERT–BiLSTM–CRF sequence tagger for Task 2 (ADU span detection and classification) of Daleel 2026.

## 2 Background

Argument mining (AM) focuses on automatically identifying and extracting reasoning structures from natural language, typically through sentencelevel classification or span-level sequence tagging paradigms. Although this task have been widely studied in English, Arabic remains under-resourced. To this end, the Daleel 2026 introduced two shared tasks which focuses on the identification and classification of argumentative discourse units (ADUs) in Arabic Text.

## 2.1 Task 1: ADU Paragraph Classification

Task 1 is a multi-label classification challenge. Given a paragraph, the system must predict the presence of one or more ADU classes: Common ground, Assumption, Testimony, Statistics, Anecdote and Other. The training labels exhibit strong class imbalance; for instance, Assumption is the most frequent label (appearing in 75.5% of training paragraphs), while Statistics is the rarest (4.6%). On average, each paragraph is annotated with 2.0 labels.

## 2.2 Task 2: ADU Span Detection

Task 2 is a token-level sequence tagging task. Given a paragraph, systems must identify the exact character span of an ADU and classify it into its corresponding class. A single text can contain multiple ADU spans of the same or different classes. The training set consists of 2,975 ground-truth spans with exceptionally dense coverage, 84.6% of training paragraphs have between 80% and 100% of their text volume bound to a labeled span.

The dataset for the shared tasks is partitioned into a training set (n = 612), a development set (n = 217) and a final evaluation test set (n = 213). All splits share highly consistent domain distributions. However, distribution of labels although fairly consistent but highly skewed in both train and dev set. Common grounds and Statistics have the least amount of data instances (see Table 1).

We worked exclusively with task 2 under closed track rules which is essentially a sequence tagging task. Early sequence-based methods in argument mining used conditional random fields (CRFs) over BIO-encoded tokens to identify premises (Goudas et al., 2014). This approach was later extended to the joint extraction of major claims, claims, and premises from text (Stab and Gurevych, 2017). To capture complex long-range sequential dependencies more effectively, modern systems use deep learning and contextualized language models for end-to-end span extraction. Eger et al. (2017) advanced neural end-to-end argument mining by investigating BiLSTM-CRF sequence tagging models for identifying exact argument boundaries. Building on this line of work, Binder et al. (2022) introduced a BERT-BiLSTM-CRF architecture for identifying and classifying scientific argumentative discourse units (ADUs). Because the Daleel 2026 shared task requires precise extraction and classification of ADUs within Arabic texts, the framework established by Binder et al. (2022) serves as the foundational basis for our methodology.

<table><tr><td>Split</td><td>Genre</td><td>Paragraphs</td><td>AS</td><td>OT</td><td>AN</td><td>TE</td><td>CO</td><td>ST</td></tr><tr><td rowspan="3">Train</td><td>Overall</td><td>612</td><td>462 (75.5%)</td><td>222 (36.3%)</td><td>182 (29.7%)</td><td>160 (26.1%)</td><td>36 (5.9%)</td><td>28 (4.6%)</td></tr><tr><td>Debate</td><td>357</td><td>287 (80.4%)</td><td>202 (56.6%)</td><td>69 (19.3%)</td><td>100 (28.0%)</td><td>22 (6.2%)</td><td>6 (1.7%)</td></tr><tr><td>Editorial</td><td>255</td><td>175 (68.6%)</td><td>20 (7.8%)</td><td>113 (44.3%)</td><td>60 (23.5%)</td><td>14 (5.5%)</td><td>22 (8.6%)</td></tr><tr><td rowspan="3">Dev</td><td>Overall</td><td>217</td><td>157 (72.4%)</td><td>63 (29.0%)</td><td>68 (31.3%)</td><td>76 (35.0%)</td><td>12 (5.5%)</td><td>9 (4.1%)</td></tr><tr><td>Debate</td><td>129</td><td>106 (82.2%)</td><td>52 (40.3%)</td><td>33 (25.6%)</td><td>43 (33.3%)</td><td>7 (5.4%)</td><td>1 (0.8%)</td></tr><tr><td>Editorial</td><td>88</td><td>51 (58.0%)</td><td>11 (12.5%)</td><td>35 (39.8%)</td><td>33 (37.5%)</td><td>5 (5.7%)</td><td>8 (9.1%)</td></tr></table>

Table 1: Paragraph-level label coverage in the Daleel 2026 dataset. Each value reports the number and percentage of paragraphs containing at least one span of the corresponding label. Percentages are computed with respect to the number of paragraphs in each row and do not sum to 100% because a paragraph may contain multiple labels.

## 3 System Overview

To account for the highly imbalanced and structured nature of ADU detection and classification, we formulate the problem as a sequence tagging task using a BIO (begin, inside, outside) tagging scheme. For this task, we propose STAR-Ar, a pretrained transformer architecture augmented with a recurrent layer and a structured prediction head.

## 3.1 Model Architecture

The core architecture uses a BERT-based text encoder pre-trained on Arabic corpora to extract contextualized representations of the input sequence. To capture longer-range dependencies and sequential argumentative structures, the transformer’s final hidden states are passed to a bidirectional long short-term memory (BiLSTM) network. A linear classifier then projects the BiLSTM outputs into the BIO-encoded label space, producing emission scores. Finally, a conditional random field (CRF) layer is applied on top of these emission scores. To guarantee the structural validity of the predicted boundaries, we explicitly initialize the CRF transition matrix to heavily penalize invalid BIO sequences. Invalid paths such as sequence initializations starting directly with an I-tag, O → I transitions, and cross-label continuations (e.g., B-AS → I-TE), are assigned a mathematically prohibitive transition score (−10000.0). This strictly forces the Viterbi decoder to output structurally coherent label sequences.

## 3.2 Optimization and Loss Formulation

To directly counteract the severe class imbalance and density of the dataset, STAR-Ar bypasses standard uniform optimization in favor of a custom, cost-sensitive composite loss formulation. The primary objective is the negative log-likelihood of the CRF sequence $( L _ { C R F } )$ . We supplement this with an auxiliary token-level Cross-Entropy loss $( L _ { C E } )$ applied directly to the BiLSTM emissions. The CE loss utilizes inverse-square-root class weights, with the O tag weight capped at 0.3 to prevent it from overwhelming the ADU classes. The final loss is defined as:

$$
L _ { t o t a l } = L _ { C R F } + \lambda _ { a u x } L _ { C E }
$$

where $\lambda _ { a u x }$ is an auxiliary weighting parameter.

## 4 Experimental Setup

To ensure robust evaluation and generalization, we utilize a 5-fold stratified cross-validation strategy. Stratification is based on the least frequent label present within a document to guarantee that rare classes are distributed evenly across folds. During training, we apply a weighted random sampler at the document level to over-sample paragraphs containing minority ADU classes.

To determine the optimal contextual representation for this task, we evaluated three distinct pretrained Arabic text encoders. We found MARBERTv2 consistently outperformed both AraBERT variants across multi seed experiments, so we used MARBERTv2 as text encoder for the final system (see Appendix A). Regarding other hyperparameters, we apply a transformer dropout rate of 0.1, and the two-layer BiLSTM utilizes a hidden size of 256 per direction. The network is optimized using AdamW for up to 20 epochs with a patience of 5 epochs for early stopping. We employ differential learning rates: $2 \times 1 0 ^ { - 5 }$ for the BERT encoder and $1 \times 1 0 ^ { - \bar { 3 } }$ for the randomly initialized top layers (BiLSTM, CRF, and linear heads), alongside a linear learning rate scheduler with a 10% warm-up phase and gradient clipping at 1.0.

<table><tr><td>Split</td><td>Evaluation</td><td>Editorial</td><td>Debate</td><td>Both</td></tr><tr><td rowspan="3">Dev</td><td>Editorial</td><td>60.80</td><td>51.30</td><td>66.52</td></tr><tr><td>Debate</td><td>58.28</td><td>75.29</td><td>75.04</td></tr><tr><td>Both</td><td>60.15</td><td>69.25</td><td>72.69</td></tr><tr><td rowspan="3">Test</td><td>Editorial</td><td>62.35</td><td>54.52</td><td>63.56</td></tr><tr><td>Debate</td><td>60.07</td><td>76.17</td><td>77.84</td></tr><tr><td>Both</td><td>61.25</td><td>70.68</td><td>73.74</td></tr></table>

Table 2: F1-score (%) for evaluation on the Daleel development and test sets. Columns denote the training domain, while rows denote the evaluation split and domain.

The final system for test evaluation is trained with both train and dev set using same 5-fold stratifed cross validation strategy. For predictions, we leverage an ensembling technique at the emission level. Rather than applying majority voting to the final decoded label sequences, we compute tokenwise averages of the raw CRF emission probabilities across all trained folds and perform a single Viterbi decoding step on the averaged emissions to produce the final predicted spans.

## 5 Results and Discussion

Table 2 presents STAR-Ar’s performance on the development set as well as test set for Task 2. Overall, we see model performing better on test set which is likely because we added development data to train the final system. Furthermore, we observe that the model trained on both domains performs better on debates than on editorials which is understandable given the distribution of data across domains. Training on debate domain alone yields performance close to training on the both domains, suggesting that debate data captures many of the linguistic patterns needed for ADU identification. On the other hand, models trained only on editorials benefit more from incorporating debate data, likely because the editorial subset is smaller and exhibits greater label imbalance. Consequently, the additional debate examples provide useful supervision that improves generalization to editorials.

As detailed in Table 1, the dataset exhibits severe class imbalance. To evaluate the system’s span-level boundary resolution across these skewed distributions, we analyze the character-overlap confusion matrix in Figure 2. Overall, the model exhibits a strong predictive bias toward the majority class, Assumption (AS). While AS achieves the highest recall (88.49%), the model systematically overpredicts this label across all other categories. This absorption effect is most severe for Common ground (CO), where 74.22% of gold characters are erroneously captured by AS predictions, reducing CO’s recall to just 12.26%. Mid-frequency classes such as Anecdote (AN), Testimony (TE), and Other (OT) demonstrate moderate robustness, maintaining recall rates between 45% and 55%. However, they still lose a substantial portion of their span volume (ranging from 31% to 40%) to the majority AS class. Interestingly, despite being the rarest class, Statistics (ST) performs robustly as the secondbest category with a recall of 64.77%. Because its explicit quantitative surface features provide a highly separable linguistic signal, it largely resists absorption into the majority class, with its primary misclassifications splitting between TE (22.06%) and AS (11.11%).

![](images/ee8dcbeba977e7426478b7ce5180cfe23cda663af83f2bb01a75233f068c0b6f.jpg)  
Figure 2: Cross-domain span-level confusion matrix. Rows are gold classes; cells show mean character-wise overlap with predictions. The diagonal represents spanlevel recall; ’NONE’ captures false negatives.

## 6 Conclusion

In this work, we present STAR-Ar, a BERT-BiLSTM-CRF framework tailored for argumentative discourse unit (ADU) extraction and classification in Arabic. On the shared task final test set, our system achieved a $\mathrm { F _ { 1 } }$ score of 73.74 on the both domain setting, ranking 3rd out of 14 participating teams. Our findings demonstrate that sequence labeling remains a competitive, highly efficient paradigm for joint span detection and classification in argument mining. Qualitative analysis reveals that while class re-balancing yields nonuniform performance gains, label frequency alone fails to predict class-wise difficulty. Finally, crossdomain evaluations indicate a high sensitivity to domain shifts, highlighting the necessity for larger, more structurally diverse annotated corpora to further advance Arabic argument mining.

## Limitations

We used small language models with 512-subword limit which truncates the long inputs. Although we verified that it affect negligible portion of the dataset (1.14% in train, 0.46% in dev and 0.47% in test), but it still constrains applicability of our system to longer documents. Another limitation of this work is its generalization capabilities across other Arabic text types and dialectical variations.

## Acknowledgments

This research is funded by the German Research Foundation within the project (Semi-)Automated thematic text classification as a basis for corpuslinguistic value-added services (Project number: 531750631) and within the Infrastructure Priority Programme New Data Spaces for the Social Sciences (SPP 2431), Research-driven Infrastructure for Advanced Survey-related Data (CIRCLET) measure project number 539634240.

## References

Arne Binder, Leonhard Hennig, and Bhuvanesh Verma. 2022. Full-text argumentation mining on scientific publications. In Proceedings ofthe First Workshop on Information Extraction from Scientific Publications, pages 54–66, Online. Association for Computational Linguistics.

Steffen Eger, Johannes Daxenberger, and Iryna Gurevych. 2017. Neural end-to-end learning for computational argumentation mining. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11–22.

Theodosis Goudas, Christos Louizos, Georgios Petasis, and Vangelis Karkaletsis. 2014. Argument extraction from news, blogs, and social media. In Hellenic conference on artificial intelligence, pages 287–299. Springer.

John Lawrence and Chris Reed. 2019. Argument mining: A survey. Computational linguistics, 45(4):765– 818.

Sara Nabhani, Nahla Bassyouni, Ali Al-Zawqari, Mohammad Khader, and Khalid Al-Khatib. 2026.

Daleel 2026 shared task on arabic argumentative discourse mining. Budapest, Hungary. Association for Computational Linguistics.

Andreas Peldszus and Manfred Stede. 2013. From argument diagrams to argumentation mining in texts: A survey. International Journal ofCognitive Informatics and Natural Intelligence (IJCINI), 7(1):1–31.

Christian Stab and Iryna Gurevych. 2017. Parsing argumentation structures in persuasive essays. Computational Linguistics, 43(3):619–659.

Simone Teufel. 1999. Argumentative zoning: Information extraction from scientific text. Ph.D. thesis, University of Edinburgh Edinburgh, Scotland.

Stephen E. Toulmin. 1958. The Uses ofArgument. Cambridge University Press, Cambridge, England.

Douglas Walton, Christopher Reed, and Fabrizio Macagno. 2008. Argumentation Schemes. Cambridge University Press, Cambridge, England.

## A Encoder Comparison

In order to decide which encoder is best for our system, we performed multi seed experiment using three different Arabic encoders. We ran same 5 fold stratified cross validation on train set across three different seeds to obtain results shown in Table 3.

<table><tr><td>Encoder Model</td><td>Task 2 Overall F1 (%)</td></tr><tr><td>AraBERTv02-Twitter-b</td><td> $6 3 . 7 5 \pm 0 . 4 3$ </td></tr><tr><td>AraBERTv02-Twitter-1</td><td> $6 5 . 8 2 \pm 2 . 1 7$ </td></tr><tr><td>MARBERTv2</td><td> $6 7 . 5 3 \pm 1 . 5 3$ </td></tr></table>

Table 3: Ablation Study: Impact of different pre-trained text encoders on overall performance (models trained on the combined dataset).

## B Paragraph Level Error Analysis

To evaluate the efficacy of our proposed system for the class imbalance, we analyze the development set confusion matrix (Figure 3). Overall, the model robustly identifies the majority class, Assumption (AS), alongside mid-frequency classes, Testimony (TE) and Other (OT). Misclassifications among minority classes are predominantly driven by false negatives (omissions) rather than cross-label confusion. Specifically, Anecdote (AN) and Common ground (CO) are most frequently misclassified as None rather than assigned competing labels.

Despite being the rarest class, Statistics (ST) achieves 100% precision (recovering 6 of 9 instances without false positives), as its explicit quantitative surface features provide a highly separable signal that cost-sensitive weighting exploits.

![](images/7df2bddef0a5b80e3c709cd94b024664e884f47eaa1afc227d097fa328bac4e3.jpg)  
Figure 3: Paragraph-level dev-set heatmap (crossdomain trained). Rows list gold labels with paragraph support; the diagonal gives recall. Off-diagonal cells show spurious predictions on missed gold paragraphs.

Conversely, CO yields the lowest recall (2 of 12 instances), indicating that class-re-balancing techniques fail to surface minority classes that lack distinct lexical or structural cues. This pattern remains consistent in single-domain configurations, underscoring the necessity of multi-domain training which mitigates cross-domain sparsity by leveraging complementary label distributions.