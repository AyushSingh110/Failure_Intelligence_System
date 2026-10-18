# C2 Phase 1.1 — Final-test options (for OD-C2-5; nothing downloaded)

**Constraints.**

- C2's final test must be collected, deduplicated and hash-frozen **before any C2 candidate is trained**.
- It is never used to choose hyperparameters, arms or thresholds.
- Everything local is development evidence: XSTest, OR-Bench-hard, SORRY-Bench, HarmBench, StrongREJECT,
  JailbreakBench, AdvBench, TruthfulQA, HaluEval-600, and the MMLU subsets used in E54/E55.

**Status of every candidate below:** named in `C2_PHASE1_DESIGN.md` §7.1. **None has been checked online in
this phase.**

- Their existence, licence, current version, split structure and size are **Unknown until verified at
  download time.**
- Descriptions are from the project's prior documents and general knowledge, not from inspected files.

## 1. Which research question each candidate would test

| Candidate (as named in the design) | Primary capability measured | C2 question | Overlap risk with FIE data | Must be established before freezing |
| --- | --- | --- | --- | --- |
| An over-refusal set not yet used: OKTest, PHTest or FalseReject (choose ≥ 1) | Over-refusal on benign but sensitive prompts | P2: does concept or topic de-confounding reduce hard-benign over-refusal at matched recall? | Medium: built in the same safe-but-scary style as XSTest / OR-Bench, which shaped aug_v63b (E25) | Licence; version or commit; benign/unsafe labels; category field; whether any items derive from XSTest or OR-Bench |
| WildGuardTest (evaluation split) | Harmful-intent classification of prompts, including adversarial phrasings, with benign contrasts | P1/P3: harmful-content intent generalisation outside the development benchmarks | **High** unless checked: WildJailbreak / WildGuard draw on sources that overlap HarmBench / JBB-style behaviours; ToxicChat-like chat may also overlap | Licence and access terms; prompt-harmfulness label field; source of each item; exact split; deduplication against all FIE sources |
| WildJailbreak evaluation split | Adversarial (jailbreak-style) harmful requests plus adversarial benign contrasts | Framing and template robustness of H; over-refusal on adversarial-looking benign prompts | High: the training collector (`data/collect_pair_training_data.py`) has a `collect_wildjailbreak` function; whether any rows entered v6.3b must be checked (the audit found no `wildjailbreak` source in the corpus) | Same as above, plus confirming no train-split leakage |
| NotInject | Over-defence of prompt-injection detectors (benign text containing trigger words) | M-axis false positives; trigger-word shortcut | Low against FIE training (no injection training data) | Licence; version; label mapping |
| BIPIA or deepset prompt-injections | Instruction-hierarchy manipulation / indirect injection | M recall (evaluation only, per OD-C2-8 default A) | Low | Licence; which subset is a user-input prompt and which needs an external-content channel FIE does not have |

## 2. Is one dataset enough?

**No.** C2's primary metrics need three different things:

- **P1:** harmful-intent recall on new sources;
- **P2:** benign over-refusal on prompts that are sensitive in topic but benign in intent;
- **OD-C2-8 default A:** an M-axis slice that is evaluated but makes no claims.

**Recommended composition (owner to choose):**

1. **One harmful-intent set with benign contrasts** (WildGuardTest-type), for P1 and P3-like analysis.
2. **One over-refusal set not used before** (one of OKTest / PHTest / FalseReject), for P2.
3. **One injection or over-defence pair** (NotInject plus one injection set), as an M-axis slice. It is
   reported, not used for C2 acceptance.

**Size targets** (design §7.4): ≥ 400 benign items per register for headline over-refusal; ≥ 100 per family
for family claims. A candidate that cannot meet these is reported as a pilot.

## 3. Freeze procedure (to run in the approved download window, before S3)

1. **Download** into a new directory (e.g. `research/adversarial/c2_final_test/raw/`), recording URL, version
   or commit, licence text, retrieval time and file SHA-256.
2. **Normalise** to `{id, prompt, label_harmful, label_benign_contrast, category, source, licence}` with a
   versioned mapping script. Ambiguous source labels map to `unclear` and are reported, never forced.
3. **Deduplicate:**
   - exact-normalised matching against every FIE training, calibration, development and E53–E55 item;
   - character 5-gram MinHash (Jaccard ≥ 0.8);
   - cosine ≥ 0.92 under MiniLM **and** under the deduplication-only encoder if OD-C2-6 = A.
   - All removals are logged.
4. **Freeze:** write `FINAL_TEST_MANIFEST.json` with content hashes per file and per row set, the label
   mapping, the deduplication report, and the SHA-256 of the evaluation script. Commit it, or have the
   owner sign it, **before** S7.
5. **Keep it out of selection:**
   - training and selection code must refuse to read the final directory (path guard plus a test);
   - the final set is scored once per preregistered candidate in S8;
   - a re-score needs a logged reason.
6. **Evaluation script:** written and hashed at freeze time. It reads the frozen thresholds from calibration
   and computes the preregistered metrics only.

## 4. Trade-offs for the owner

| Choice | Benefit | Cost |
| --- | --- | --- |
| Smallest set (one over-refusal set) | Fast, cheap | Cannot support P1 or any H-generalisation claim |
| Recommended three-part set | Covers P1, P2 and the M slice | Larger download and licence review; more deduplication |
| Also hold out a register from the local data (e.g. all hh-rlhf) | No download | Not independent of FIE's development history; development-only |
