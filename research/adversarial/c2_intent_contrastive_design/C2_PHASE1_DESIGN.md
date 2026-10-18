# C2 Phase 1 — Training-data audit and design for intent-contrastive, de-confounded PAIR training

| | |
| --- | --- |
| Date | 2026-10-18 |
| Repository | `rebuild/wp-001-eval-harness` @ `a59787f` (E55 commit). Tree clean at start and end. |
| Canonical baseline | `BL-0002_pair-v6.3b_fie-70de7053` (`evals/baselines/CANONICAL`), untouched. |
| Scope | Planning and read-only audit. **No dataset generated, no label changed, no model trained or scored, no network, no installation.** |
| Evidence | [audit/training_data_audit.json](audit/training_data_audit.json), produced by [scripts/audit_training_data.py](scripts/audit_training_data.py) from E53's checksum-verified cached embeddings and saved scores. Machine-readable design: [design_spec.json](design_spec.json). |

**Labels used in this document**

- **Verified:** recomputed here from local artifacts.
- **Reported:** taken from an earlier E-entry or report and not re-run here.
- **Inference:** reasoned from the evidence.
- **Unknown:** cannot be established from local evidence.

---

## 0. State and prior evidence

### Repository and outputs

- **Branch and HEAD:** `rebuild/wp-001-eval-harness` @ `a59787f`; `git status --short` was empty.
- **Untracked research outputs:** `results/` in E53, E54 and E55 is git-ignored by `.gitignore` line 82 (`results/`).
  - These folders exist only locally. They are covered by each experiment's `SHA256SUMS.txt`, and all three verify.
  - The commits do not contain the metrics, per-arm scores or integrity reports. **Owner decision OD-C2-1** (§10) asks whether to force-add them.

### What each prior finding rests on (all verified from saved records)

| Finding | Basis |
| --- | --- |
| E53: dropping HarmfulQA or red-team rows, or balancing sources, does not lower XSTest / OR-Bench over-refusal at matched recall. A3 − A0 on XSTest is −0.8 points. | `e53/results/metrics.json` |
| E53 A5: leave-one-source-out (LOSO) optimism of 15–32 points for 9 of 18 sources. | `e53/results/a5_loso.json` |
| E54: no candidate reaches the useful trade-off; H0 retained. | `e54/results/metrics.json` |
| E55: fails at α = 10% and α = 5%; the view-count rule can be exploited. | `e55/results/metrics.json` |
| All three integrity checks pass. | `integrity_check.json` in each experiment |

**Development-only evidence:** every result measured on XSTest, OR-Bench, SORRY-Bench, HarmBench, StrongREJECT or JailbreakBench. These sets have shaped PAIR since v6.

**Independent in E55 only:** the filler, the templates and the benign sources.

**Not available in any experiment:** an attack-family-held-out or source-held-out evaluation.

---

## 1. Why C2, and why it is not "source removal again"

E53 removed sources and found nothing. The audit below shows why removal could not work. The problem is not that a few sources are noisy. Three properties of the corpus as a whole make topic and register cheaper to learn than intent.

1. **Topic–label confound (Verified).**
   - Harm-topic vocabulary (fixed regex in `cue_regex.harm_topic`) appears in **20.9%** of attack rows and **3.5%** of benign rows.
   - So P(attack | harm word) = **0.875** in training.
   - Hard-benign evaluation sets contain harm words at **15.6%** (XSTest-safe) and **23.6%** (OR-Bench-hard), 4.5–6.7× the training-benign rate.
   - The model has almost never seen a benign prompt that contains harm vocabulary. This is the mechanism behind E19's high-confidence over-refusals (Inference).
2. **Register / source confound (Verified, with E53).**
   - Every source carries a single label.
   - E53 A5: unseen benign sources are flagged 20–32 points more often than row-random CV suggests.
   - For red-team rows, the nearest opposite-label neighbour comes from Anthropic-helpful in 216 of 485 cases: the same platform register, different label.
3. **Concept conflation (Verified from code and data).**
   - The binary "attack" label joins harmful-content requests, toxic chat, the opening turns of red-team conversations, refusals, and a small amount of instruction-hierarchy manipulation.
   - **Instruction manipulation is nearly absent from training.** Its only real source is ToxicChat's `jailbreak` subset (71 rows).
   - JailbreakBench *behaviours* (89 rows) are goals, not jailbreak prompts.
   - The manipulation-cue regex fires on 0% of every other source.

C2 tests a different hypothesis from E53: **if the training signal is redefined and balanced so that topic and register no longer predict the label, does a frozen-encoder head separate intent better?** It does this by labelling by concept, pairing topic-matched examples of opposite intent, and enforcing framing symmetry. Removal changed none of the three confounds above.

---

## 2. Training data inventory (v6.3b corpus, n = 4,236; Verified)

The recipe is `scripts/train_pair_v63b.py`.

- **Files and weights:** `train.decontam.jsonl` (×1), `dataset_v6.decontam.jsonl` (×1), `augmented_v6.decontam.jsonl` (×2), `augmented_v6_3_train.jsonl` (×2), `augmented_v6_3b_benign_train.jsonl` (×2).
- **Preprocessing:** deduplicated by normalised text (SHA-1); truncated to 512 characters; prefix added at encoding.
- **Split:** `train_test_split(0.10, seed 42, stratified by label)`, row-random, giving `tr` 3,812 and `va` 424.
- **Collection** (`data/collect_pair_training_data.py`, May 2026): hh-rlhf **first human turn only**; ToxicChat rows where `jailbreaking == 1` **or** `toxicity == 1`.

### 2.1 Source-by-source audit

Column notes:

- **LOSO / row-CV** = share of `tr` rows that the E53 heads strongly contradict (attack scored < 0.2, benign scored > 0.8), under leave-one-source-out and under row-random CV.
- **Opposite-label ≥ 0.80** = number of rows whose nearest opposite-label row has cosine ≥ 0.80 (candidate natural topic-matched pairs).
- **Near-dup ≥ 0.95** = rows with a within-source near-duplicate.

| Source | Label | n | Kind | Mean words | Near-dup ≥ 0.95 | Opposite-label ≥ 0.80 | LOSO / row-CV | Known or suspected issue | Proposed handling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anthropic_rlhf_redteam | 1 | 485 | first human turn of red-team transcripts | 13.5 | 6 | 74 | 22.6% / 7.5% | **Label = transcript membership.** Red-team openers are often innocuous; harmlessness ratings were not kept locally | **Review all.** Cannot be resolved from metadata |
| harmfulqa | 1 | 465 | questions (reported to be model-generated by the dataset authors; not verified here) | 10.5 | 42 | 212 | 8.3% / 1.7% | E2 found benign academic/business questions; categories collapsed to "harmful" | **Review all** |
| toxicchat `toxic_query` | 1 | 180 | real user chat | ~48 (whole source) | 24 (whole source) | 15 (whole source) | 8.7% / 0.9% (whole source) | **Toxicity ≠ harmful request** | **Review all** |
| toxicchat `jailbreak` | 1 | 71 | real user chat, many long role-play prompts | — | — | — | — | The only real manipulation data; 30% of the source is truncated at 512 characters | Review; becomes the core of the manipulation class |
| llama_paraphrase | 1 | 160 | LLM paraphrases of attack seeds | 17.8 | 28 | 54 | 0% / 0% | **82 rows (51%) are refusal texts labelled attack** (row ids in the audit) | **Exclude refusals as attack**; review the rest |
| mlabonne/harmful_behaviors (AdvBench mirror) | 1 | 288 | behaviour requests | 12.2 | 66 | 112 | 0.8% / 0.4% | AdvBench template style; 67.5% of AdvBench overlapped training (E21) | Retain; sample audit; flag template-style duplicates |
| forbidden_question_set (policy-filtered) | 1 | 210 | questions | 12.5 | 0 | 55 | 1.6% / 1.1% | Policy categories filtered in v6 (E2) | Retain; sample audit |
| MaliciousInstruct | 1 | 100 | requests | 10.5 | 2 | 27 | 0% / 0% | — | Retain; sample audit |
| JBB-Behaviors | 1 | 89 | behaviour goals | 14.1 | 0 | 12 | 1.2% / 1.2% | Goals, not jailbreak prompts | Retain as harmful-request (H), not manipulation (M) |
| aug_v63 (soft harm) | 1 | 230 | 58 hand-authored seeds + 172 NLLB back-translations | 14.3 | 133 | 34 | 19.7% / 2.3% | **No seed id stored.** Families are recoverable: back-translation to nearest seed median cosine 0.94; 6 ambiguous | Retain; recover families before any split |
| alpaca (base) | 0 | 634 | instructions | 10.4 | 10 | 200 | 11.4% / 3.6% | Low harm vocabulary (1%) | Retain |
| tatsu-lab/alpaca (v6) | 0 | 200 | instructions | 8.8 | 24 | 114 | 1.1% / 1.6% | Same corpus as above under another name | Merge the source identity for grouping |
| anthropic_rlhf_helpful | 0 | 551 | first human turn | 13.7 | 2 | 90 | **32.4%** / 7.4% | Same register as red-team | Retain; key register-matched negative pool |
| law_stackexchange | 0 | 100 | legal questions | 12.4 | 0 | 16 | **41.1%** / 23.3% | Reads adversarial (E4) | Retain as hard benign; review a sample |
| MedQuad | 0 | 150 | templated medical questions | 9.2 | **100** | 0 | 6.5% / 0% | Heavy templating ("What are the symptoms of …") | Retain; group by template |
| medical_meadow_medqa | 0 | 150 | exam vignettes | 61.7 | 0 | 0 | 0% / 0% | Long; 19% harm vocabulary | Retain |
| mbpp | 0 | 100 | coding tasks | 15.5 | 6 | 0 | 0% / 0% | — | Retain |
| hand_written | 0 | 30 | hard benign | 9.7 | 0 | 10 | 8% / 8% | Self-authored | Retain |
| aug_v63b | 0 | 43 | hand-authored safe-but-scary | 10.4 | 0 | 8 | 19.5% / 9.8% | Written in XSTest's registers (E25) | Retain; **must not support XSTest claims** |

Additional verified facts:

- **Cross-label conflicts.** Rows whose nearest opposite-label neighbour has cosine ≥ 0.92: 22 in total (ids in the audit). Every one needs adjudication.
- **No conversation-turn markers survive** (≤ 2% anywhere).
- **Length is not a confound in training.** Attack rows average 16.7 words and benign rows 15.3.

### 2.2 What cannot be established locally (Unknown)

- **Red-team intent per row.** The original transcript ratings are not in the local file.
- **HarmfulQA's original categories and generation provenance.**
- **Why each ToxicChat row was flagged.** Only the combined flag survives.
- **The rate of true label errors.** No independent human labels exist. The disagreement rates above are model-based proxies, not error rates.

---

## 3. Label taxonomy proposal (v0, for owner approval)

The proposal is a **multi-label scheme of four dimensions, not four exclusive classes.**

| Dimension | Values | Definition |
| --- | --- | --- |
| **M — instruction-hierarchy manipulation** | 0 / 1 / unclear | The input tries to override, bypass, extract or redefine the model's instructions, role, policy or control boundary. Examples: "ignore previous instructions", personas without restrictions, system-prompt extraction, delimiter or role injection, instructions addressed to the model inside data. |
| **H — harmful-content intent** | none / harmful / dual-use | The user asks for help that falls inside FIE's harmful-content policy (operational help with violence, weapons, cyber-intrusion, fraud, sexual content involving minors, and similar). *Harmful* = the requested output would provide that help. *Dual-use* = legitimate and harmful readings are both plausible. |
| **C — legitimate context** | set of {educational, analytical/defensive, fictional, historical, quoted, professional-domain, none} | The communicative purpose. C never overrides H = harmful by itself. It is recorded so that framing can be tested symmetrically. |
| **Q — label confidence** | high / medium / low, plus `adjudicated` | Two independent reviewers; a disagreement goes to an adjudicator. Low-confidence rows are excluded from training and analysed separately. |

### 3.1 Decision rules

1. **Topic is not intent.** Mentioning war, drugs, hacking or crime sets neither M nor H. The test is the requested action and the expected output.
2. **The addressee test for quoted content.**
   - Quoted malicious text that the user asks to *analyse, classify, explain or translate* gives M = 0. H is judged on the requested output.
   - Quoted instructions *directed at the model*, or a request to *follow* them, give M = 1. This rule covers indirect injection.
3. **Security research.** Conceptual explanation, defence and detection give H = none. Requests for a working exploit or malware against a real target give H = harmful, even with research framing.
4. **Framing does not change intent.** A medical, legal, academic, developer or fictional frame keeps the H and M of the underlying request. A frame that *changes the requested output* (for example "summarise the history of" instead of "give the steps for") is a semantic change and is labelled anew.
5. **Both at once.** M = 1 and H = harmful are both recorded. Training uses both heads; the production label is the union.
6. **Contradictions.** Two high-confidence reviewers who still disagree after adjudication → Q = low, excluded from training, kept in an "ambiguous" analysis set.

### 3.2 Mapping to the production binary label

- **attack** = (M = 1) ∨ (H = harmful) ∨ (H = dual-use and policy says block).
- The dual-use policy is **owner decision OD-C2-3**.
- Refusals, and rows with H = none and M = 0, map to **benign** regardless of source.

### 3.3 Historical rows (no relabelling done in this phase)

| Group | Rows | Status |
| --- | --- | --- |
| Review required | red-team 485, HarmfulQA 465, ToxicChat 251 (both subtypes), llama-paraphrase non-refusals 78, cross-label conflicts 22 | **≈ 1,300** (the conflicts overlap the other groups) |
| Mechanically excludable as "attack" | 82 llama-paraphrase refusals | Need owner sign-off (OD-C2-2) |
| Retain, with a 10% sample audit | AdvBench mirror, forbidden_qs, MaliciousInstruct, JBB-behaviours, aug_v63, and every benign source | — |
| Cannot be resolved from metadata | red-team, ToxicChat subtype rationale, HarmfulQA categories | Human review is the only route |

---

## 4. Minimal-pair specification

**Goal.** Topic-matched pairs (x⁺, x⁻) that share subject, key vocabulary, length band, register and framing, and differ only in requested intent, so that vocabulary or topic alone no longer separates the classes.

### 4.1 Construction routes, in order of preference

1. **Mined natural pairs (no generation).**
   - Attack rows whose nearest benign neighbour has cosine ≥ 0.80 under a *different* encoder from the one being trained, or under MiniLM for a first pass.
   - The audit counts 595 such attack-side rows across sources, including HarmfulQA 212, AdvBench mirror 112, red-team 74 and forbidden_qs 55.
   - Each candidate is human-verified as a true intent contrast on the same topic, not just a near neighbour.
2. **Human-written benign twins of existing human-written attack requests.**
   - Same topic and vocabulary, legitimate intent: defensive, educational, historical, or a safe version of the task.
   - Only the benign side is written, so no new harmful content is created.
3. **Locally generated benign twins** (only if OD-C2-4 allows).
   - Generated with a local Ollama model, then reviewed by a human.
   - **Feasibility:** `ollama.exe` and on-disk models (llama3.1, llama3.2, mistral, qwen2.5, qwen3, granite3.3) are present. The service was **not started**, so working generation is **Unknown**.
   - In E24a, llama3.1 refused to generate euphemistic-harm examples.
   - **No harmful-side generation is planned.** Harmful twins come only from existing human-written benchmark-style requests, never from XSTest, OR-Bench, SORRY or other evaluation sets.

### 4.2 Controls per pair

- **Topic group:** 12 proposed groups: weapons, explosives, drugs, cyber, fraud, violence, self-harm, hate, privacy, sexual, extremism, dangerous chemistry.
- Length ratio within 0.67–1.5.
- Same framing class.
- Same quotation status.
- Same source register, wherever a twin is mined.
- **Recorded fields:** `pair_id`, `topic_group`, `source_ids`, `construction_route`, `author` (human / model name), `review` (two reviewers + adjudication), `taxonomy` (M, H, C, Q).

### 4.3 Proving a pair differs in intent and not in surface form

For every accepted pair, record:

- token-level Jaccard similarity (target ≥ 0.4);
- harm-vocabulary count on each side (target equal ±1);
- length ratio;
- reviewer agreement on H and M (both reviewers must agree on the contrast).

At the corpus level, a surface-only baseline (bag-of-words logistic regression trained on pair membership) must be near chance at telling twins apart by side. If it is not, pairs leak surface cues and must be rebalanced.

### 4.4 Review and artifact control

- Two-reviewer labelling, with Cohen's κ reported per dimension.
- Pilot of 200 rows before scaling. The rule is preregistered: if κ < 0.6 on H or M, revise the taxonomy before continuing.
- **Generator-artifact screen:** a classifier trained on generated vs human-written *benign* text must be near chance. Otherwise generated twins get their own source group and are evaluated source-held-out.
- **Partitioning:** each pair, together with its seed family and back-translation descendants, is assigned as one unit to exactly one partition (§7). Near-duplicate guard: no cross-partition pair with cosine ≥ 0.92 under two encoders.

### 4.5 Effort estimate (Inference)

- 12 groups × 40 pairs = 480 pairs (960 items) for training, plus 200 held-out evaluation pairs.
- At about 1 minute per item per reviewer with 2 reviewers: ≈ 40 reviewer-hours.
- Adjudication of ≈ 15%: ≈ 4 hours.
- The ≈ 1,300-row label review adds ≈ 45 reviewer-hours.

---

## 5. Framing-invariance specification

The design is symmetric: frames are applied to **both** benign and attack items. The label follows §3.1 rule 4.

- **Frame bank.** Five frame classes (neutral, medical, legal/compliance, academic, developer/security-analysis) × 3 structures (prefix clause, inline without a boundary, suffix sentence).
  - **Training frames:** written fresh (human, or reviewed local generation).
  - **Held-out frames:** a separate bank written by a different author, with no shared n-grams of 4 or more tokens with the training frames, never used for training or hyperparameter selection.
  - The E54 and E55 templates are **excluded** from both training and evidence of generalization; they may serve only as a historical development reference.
- **Semantic-change check.** A framed item keeps its label only if a reviewer confirms the requested output is unchanged. A sample of 10% is double-reviewed.
- **Separate evaluations:**
  1. **Stability:** flip rate of decisions under held-out frames, separately for benign and attack items.
  2. **Manipulation detection:** recall on M = 1 items, a separate subset that excludes harmful-content-only items.
  3. **Hard-benign over-refusal.**
  4. **Source- and register-held-out performance** (§7).
  5. **Length and format:** E55-style human-filler dose-response, and markdown/code formatting.

---

## 6. Training arms and objectives

### 6.1 Constraints

- **Frozen MiniLM encoder** (cached ONNX embeddings, as in E53).
- The trainable part is a head:
  - (i) the v6.3b LinearSVC + sigmoid recipe, for A0; and
  - (ii) a small head f(x) = wᵀ·tanh(P·e(x)) + b with P ∈ ℝ^{64×384}, or a linear head when P = I.
- **Implementation:** torch is **not installed** in `failure-engine` or `fie-eval-pinned` (Verified), so the heads are trained in numpy/scipy (L-BFGS on analytic gradients). No new dependency is needed.
- **Compute (Verified):** 20 CPU threads and an RTX 3050 4 GB.
  - Embedding about 10k texts took about 9 minutes on CPU in E53.
  - Fitting a head on about 5k × 384 features is a matter of seconds.
  - Compute is adequate.

### 6.2 Objective

Let ℓ be weighted binary cross-entropy on logits z = f(x):

L(θ) = Σᵢ wᵢ·ℓ(yᵢ, zᵢ)
  + λ_p · Σ_{(a,b)∈P} max(0, m − (z_a − z_b))
  + λ_c · Σ_{(x,x′)∈F} (z_x − z_{x′})²
  + λ₂·‖θ‖²

| Term | Role | Pairs or groups | Hyperparameters | Likely failure mode |
| --- | --- | --- | --- | --- |
| Pairwise margin | Forces intent separation *within* a topic | P = reviewed minimal pairs, attack twin a, benign twin b | λ_p ∈ {0.1, 1}, m = 1 logit | Overfits pair-specific wording; collapses if pairs leak surface cues (§4.3) |
| Consistency | Same intent ⇒ same score under semantics-preserving frames, for both classes | F = (item, framed item) | λ_c ∈ {0.1, 1} | Suppresses real frame-dependent intent if semantic-change checks are weak |
| L2 | Regularisation | — | λ₂ chosen with the SVM-equivalent C = 0.8 as anchor | Under- or over-fitting |

**Optional group-robust variant (A5).** Replace the first term with max over groups g of the mean group loss, using online exponentiated weights q_g ∝ exp(η·L_g), η = 0.01.

- Groups = source × register.
- Only if at least 8 groups each have 100 or more training rows.

**Multi-task variant (conditional on the review results).** Two logits, z_M and z_H, on a shared P. The production score is σ(max(z_M, z_H)).

### 6.3 Recommended arms (minimum defensible set)

| Arm | Data | Objective | Isolates | Paired control |
| --- | --- | --- | --- | --- |
| A0 | v6.3b corpus | SVM + sigmoid (exact recipe, recipe row order) | baseline; must reproduce the shipped head (E53 gate) | — |
| A1 | Label-audited corpus (§3.2 mapping, low-Q excluded) | same as A0 | **label quality** | R1: same number of rows dropped or flipped at random; class balance held by reweighting |
| A2 | A1 + reviewed minimal pairs (train partition) | same as A0 | **topic-matched data** | A2-ctl: A1 + the same number of reviewed non-paired rows from the same sources |
| A3 | A2 + symmetric framed copies | same as A0 | **framing data** | A3-ctl: A2 + the same number of duplicated (unframed) copies |
| A4 | A3 data | BCE + margin + consistency (§6.2) | **objective** | A4-ctl: same data and head, BCE only |
| A5 *(conditional)* | A3 data | group-DRO version of A4 | group robustness | A4 |

Rules applied to every arm:

- **Seeds.** Five seeds per arm for initialisation and row order; report the mean and spread.
- **Weighting.** Class prior fixed by weights to A0's effective prior. Source weights are reported, never silently changed.
- **Augmentation.** Each row's source and route is logged.
- **Comparisons.** Each arm is compared with its control, not just with A0, so that size, balance and augmentation count cannot be credited to the method.

---

## 7. Partitions and evaluation protocol

### 7.1 Partitions (group-aware)

**Groups:**

- source;
- recovered seed family (aug_v63 back-translations → nearest seed);
- MedQuad template;
- `pair_id`;
- framing transformation group;
- near-duplicate cluster (cosine ≥ 0.92 under MiniLM and one other encoder; the second encoder is Unknown and needs OD-C2-6).

**Partitions:**

1. **Training.**
2. **Calibration** (thresholds only): group-held-out within training sources.
3. **Development and model selection:** group-held-out, plus **mandatory LOSO folds** over every source with 100 or more rows. LOSO is the primary development metric, given E53 A5.
4. **Register-held-out:** hold out an entire register, e.g. all hh-rlhf (red-team + helpful) or all medical, in turn.
5. **Final evaluation:** to be collected, deduplicated against all of the above, and hash-frozen before any C2 model exists.
   - **It cannot be built from local data alone.** TruthfulQA, HaluEval-600 and parts of MMLU were used as evaluation in E54/E55 and are now development evidence.
   - Candidate sources needing owner approval and download (OD-C2-5): a fresh over-refusal set (e.g. OKTest, PHTest or FalseReject), an injection over-defence set (NotInject), an injection set (BIPIA or deepset prompt-injections), and WildGuardTest / WildJailbreak evaluation splits.
   - Licences and availability are Unknown.

XSTest, OR-Bench-hard, SORRY-Bench, HarmBench, StrongREJECT and JailbreakBench remain **development** sets.

### 7.2 Metrics (fixed before any C2 scoring)

Uncensored scores are saved for every metric.

**Primary:**

- **P1:** LOSO macro attack recall at α = 10% calibration-benign FPR.
- **P2:** hard-benign over-refusal at matched recall (development: XSTest-safe, OR-Bench-hard; final: the new set).
- **P3:** held-out minimal-pair accuracy, P(z_a > z_b).

**Secondary:**

- source- and register-held-out benign FPR;
- framing flip rate and evasion (held-out frames);
- manipulation (M) recall;
- padding dose-response with human filler;
- per-family and per-domain results;
- PR-AUC;
- ECE / Brier on calibration-like data;
- latency.

### 7.3 Acceptance criteria (to be preregistered in Phase 2)

- **Non-inferiority:** P1 ≥ A0 − 2.0 points.
- **Superiority:** P2 improves by ≥ 6.0 points on development **and** on the final set; P3 ≥ A0 + 5 points.
- **Arm attribution:** each arm must beat its own control (§6.3) by the same margins.
- **Intervals:** paired, stratified bootstrap with 10,000 resamples; Holm correction across arms for P1–P3.

### 7.4 Sample-size minimums (Inference, Wilson approximations)

| Purpose | Minimum | Why |
| --- | --- | --- |
| Headline benign set | 400 per register | ±3 points at a 10% rate |
| Any family or domain claim | 100 items | ±10 points |
| Held-out minimal pairs | 200 pairs, ≥ 12 per topic group | Per-group results reported as counts only |
| LOSO folds | sources with ≥ 100 rows in `tr` only (12 qualify) | — |

### 7.5 Separating a better representation from a shifted threshold or prior

- Report **threshold-free** P3 and PR-AUC alongside the operating-point metrics.
- Compare at **iso-recall** and **iso-FPR** (E24 style).
- Hold the class prior fixed by weighting.
- Show calibration-set threshold movement for each arm.

---

## 8. Feasibility and risks

| Item | Status | Detail |
| --- | --- | --- |
| Local data supports the taxonomy | **Partly** | Enough rows for M = 0 / H axes. **M = 1 has only 71 candidates**, too few for a reliable manipulation head without new reviewed data |
| Source / group metadata | Partly | Sources are present; seed ids are missing but recoverable; red-team and ToxicChat rationale is lost |
| Compute | **Adequate** | numpy/scipy heads; cached embeddings |
| Local generation | **Unverified** | Binary and models present; not run; known refusals for the harmful side |
| Human review | **The binding constraint** | ≈ 90 reviewer-hours including adjudication; needs at least 2 reviewers |
| New shortcuts | Risk | Generator style, pair-specific wording, frame-bank n-grams. Mitigated by §4.3–4.4 screens and LOSO |
| E53 baseline reproduction | Preserved | A0 reruns E53's verified recipe with recipe row order (E53 D9) |
| Over-reliance on development benchmarks | Risk | The final set must be collected and frozen first (OD-C2-5) |

---

## 9. Staged execution plan for C2 Phase 2 (each stage needs approval)

| Stage | Work | Outputs | Gate |
| --- | --- | --- | --- |
| S0 | Owner decisions §10 | Signed decision list | All blocking ODs answered |
| S1 | Taxonomy pilot: 200 stratified rows, 2 reviewers | κ per dimension; revised guide | κ ≥ 0.6 on M and H |
| S2 | Full review of ≈ 1,300 rows; refusal exclusion | Versioned label file (new path, originals untouched) | ≥ 95% of reviewed rows with Q ≥ medium |
| S3 | Pair mining, benign-twin authoring, review | ≥ 480 training + 200 held-out pairs | Surface and generator screens near chance |
| S4 | Frame banks (training / held-out); framed copies with semantic checks | Two banks plus framed sets | n-gram disjointness verified |
| S5 | Collect, deduplicate and hash-freeze the final evaluation set | Sealed manifest | No overlap ≥ 0.92 with anything above |
| S6 | Preregister training experiment E56 (numbering subject to D-010) | `PREREGISTRATION.md` | Before any fit |
| S7 | Train A0–A4 (+ A5 if eligible) with controls; LOSO; development evaluation | Per-arm records | Integrity checks |
| S8 | Single final-set evaluation of the preregistered winner(s) | Report | Acceptance criteria §7.3 |

---

## 10. Owner decisions required

| # | Decision | Blocking? |
| --- | --- | --- |
| OD-C2-1 | Force-add the git-ignored `results/` folders of E53–E55 to version control? | No |
| OD-C2-2 | Approve excluding the 82 refusal rows from the attack class (no relabelling until approved)? | For S2 |
| OD-C2-3 | Production policy for H = dual-use: block or allow? | For S2 |
| OD-C2-4 | May a local Ollama model write *benign* twins and frames under human review? Who reviews (2 people needed)? | For S3 |
| OD-C2-5 | Approve downloading and freezing a final evaluation set (which sets, licences)? | For S5 |
| OD-C2-6 | Which second encoder may be used for leakage and near-duplicate checks only? Needs a download; this is not the E40 comparison | For S3/S5 |
| OD-C2-7 | Accept numpy/scipy heads (no torch), or approve installing torch? | For S7 |
| OD-C2-8 | Is a dedicated manipulation (M) head in scope, given only 71 local candidates, or should C2 focus on H? | For S6 |

---

## 11. Recommendation

**The local data support a defensible C2 experiment on the harmful-content (H) axis.** The core confounds are measurable and specific:

- harm vocabulary predicts "attack" with P = 0.875 in training;
- training benign text almost never contains it (3.5%), while hard-benign evaluation text does at 4.5–6.7× that rate;
- every source is single-label, and E53's leave-one-source-out results show the head leans on register.

C2's three ingredients each target one of these confounds, and they can be isolated with the paired controls in §6.3:

- concept labels;
- topic-matched minimal pairs;
- symmetric framing.

**Must be resolved before any training:**

1. Human label review of about 1,300 rows: red-team, HarmfulQA, ToxicChat, Llama paraphrases and the conflicts.
2. A reviewed minimal-pair set of at least 480 training and 200 held-out pairs.
3. A final evaluation set collected and sealed before any model exists. This is not possible offline today.
4. A decision on the manipulation axis. With 71 candidates, C2 cannot claim an improvement in instruction-hierarchy detection.

Human review time (about 90 reviewer-hours), not compute, is the binding constraint.

Not started, as required: no C2 training, no E40, no C3, no shadow-model or RAG work.

**Paths:**

- `research/adversarial/c2_intent_contrastive_design/C2_PHASE1_DESIGN.md`
- `research/adversarial/c2_intent_contrastive_design/design_spec.json`
- `research/adversarial/c2_intent_contrastive_design/audit/training_data_audit.json`
- `research/adversarial/c2_intent_contrastive_design/scripts/audit_training_data.py`
- `research/adversarial/c2_intent_contrastive_design/SHA256SUMS.txt`
