# Audit of the v2 results handoff

Written 2026-10-30, the day the handoff was written, on branch `docs/cia-roadmap-v3`. Subject:
`research/cia_roadmap_v2/RESULTS_HANDOFF.md` at master `c885c13f`. The handoff now carries the corrections below.

**In plain words.**
1. Almost every number in the handoff matches its source. Three things are wrong: the experiment design, the power
   figure, and the claim about the published paper.
2. The serious one: the handoff says the published paper's protection result came from shared noise. The evidence
   does not support that. Shared noise was a flaw in this project's own earlier measurement, and it made leakage look
   *higher*, not lower. The audit paper's headline has to change.

## 1. Method

- **Numbers.**
  - Every numeric token was extracted from the handoff, except plain small integers.
  - Each was checked against the report(s) cited for its section in the handoff's Section 8 and the packages'
    `analysis.json`, by script.
  - Result: 260 tokens. 249 are found in the cited source, 9 in another committed source and 2 in neither. All 11
    were resolved by hand; none is wrong.
  - Table: `tables/handoff_number_check.csv`.
- **Design statements and framing.** Checked against the reports, the run manifests (`manifest.json` of every M2,
  B1 and C1 run), A1's power analysis and the project's notes on the paper's evaluation.
- **Fairness at matched accuracy.** Lowest-client accuracy was interpolated linearly in mean accuracy, in the same
  way as A1's AUC statistic. The interpolated AUCs reproduce the reported ones exactly (composite 0.606 and 0.573,
  metric privacy 0.702 and 0.681, global DP 0.702 and 0.666). Table: `tables/fairness_matched_accuracy.csv`.

## 2. Corrections

| # | Handoff location | Handoff says | Evidence | Correction |
|---|---|---|---|---|
| C1 | §1.3 item 2 | 16 candidates, each IN in half the replicates; the other 32 always take part; 512 trials; top decile plus *eleven* by size quantile | M0 report §2 (lines 50-53). Manifests of every M2, B1 and C1 configuration: 32 candidates, 16 IN per replicate, each candidate IN in exactly 16 of 32 replicates, 32 of 48 slots active | **32 candidates** (top decile 23, 7, 40, 18, 33 plus 27 by size quantile). Each replicate includes 16 of them; the other 16 clients always take part; 32 active of 48 slots. **1,024 trials per configuration.** The handoff copied ROADMAP §2's pre-M0 plan. |
| C2 | §1.3 item 5; §7 "Power" | Minimum detectable difference at `R = 32` about 0.05 (all candidates), 0.09 (top decile) | A1 `power.py` (A1 report §2): detectable difference at 80% power for the paired matched-accuracy statistic 0.027 / 0.037 (all, at 0.740 / 0.660) and 0.091 / 0.096 (top decile). M0 / M2: 0.087 for *independent* configurations | Paired matched-accuracy differences: detectable at about **0.03-0.04** (all) and **0.09-0.10** (top decile). Unpaired comparisons: about 0.09. 0.05 is the gate's effect-size threshold, not the power. |
| C3 | §0 first bullet; §5.1 thesis bullet 2 | "The protection result was an evaluation artifact. It came from an evaluation in which the compared worlds shared their noise draws." / "Its published protection came from IN/OUT worlds sharing noise (A-1)." | M1 report §1: the coupled design is **v1's** consensus-KL step 3 in this repository. The paper is not mentioned. Sharing noise made leakage *insensitive* to noise (paired score 0.79 at 1-4x; 0.516 decoupled at 4x), i.e. it **inflated** measured leakage. The paper reports near-chance AUCs. Its weakness, per `research/literature_review/evaluation_requirements.md` §1: a pooled ROC over 20 IN and 20 OUT checkpoint scores from single trajectories, in a 3-client federation; intervals overlapping 0.5 read as protection; global DP at 0.397, below chance. | A-1 is a **methodological finding about this project's earlier measurement**, not an explanation of the paper's numbers. The paper's protection claim is unsupported for a different reason: its estimand. That reason is not yet tested by re-measuring the paper's own setting under standard v2. |
| C4 | §3 D-4 readings; §0 | Logit adjustment adds 3-10 pp accuracy at the same noise | C1 `analysis.json` against B1 `analysis.json` (`q = 0.25`), main runs at shared levels: +3.7 pp (`r` 0.000773), +2.5 pp (0.003092), +6.0 pp (0.006184). 3-10 pp is the 2-run pilot (C1 `STATE.md`, 2026-10-29 18:01) | **+2.5 to +6.0 pp** in the main runs (3-10 pp in the 2-run pilot). |
| C5 | §0 bullet 2 | "leaks exactly like global DP" | A1: equivalent within ±0.03 at 0.740 (-0.000 [-0.019, 0.018]); at 0.660 only "no detectable difference" (+0.015 [-0.016, 0.045]) | "Equivalent to global DP within ±0.03 at 0.740 accuracy; no detectable difference at 0.660." |
| C6 | §0 defense bullet 2 | Subsampling lowers AUC "by about 0.10" | B1: -0.102 at 0.740, -0.059 at 0.660 | "by 0.10 at 0.740 and 0.06 at 0.660." |
| C7 | §0 defense bullet 3 | Beats metric privacy by about 0.10, "against two attackers who know the defenses" | C1 report §2 and §6: attacker (a), label-matched surrogates, is compared against **global DP** under the same attacker, because A1's metric-privacy runs did not score surrogates | Against metric privacy: standard and defense-aware attackers. Label-only attacker: against global DP only. |
| C8 | §3 D-4 readings; §5.2 item 5 | Fairness: the lowest client accuracy drops to 0.33 | At matched accuracy (table below), full-participation mechanisms are at 0.36-0.37 at 0.660 and about 0.50 at 0.740 | Lowest-client accuracy is **4-9 pp below** the full-participation mechanisms at matched accuracy (point estimates, no interval). The absolute 0.33 is mostly the accuracy target itself. |

**Lowest-client accuracy and AUC at matched accuracy** (linear interpolation in mean final accuracy, all candidates):

| Mechanism | Lowest client, `a*` 0.740 | Lowest client, `a*` 0.660 | AUC 0.740 | AUC 0.660 |
|---|---|---|---|---|
| Global DP (M2) | 0.512 | 0.366 | 0.702 | 0.666 |
| Metric privacy (A1) | 0.497 | 0.361 | 0.702 | 0.681 |
| Private median (A1) | 0.500 | 0.356 | 0.696 | 0.656 |
| Subsampling `q = 0.25` (B1) | 0.426 | 0.306 | 0.600 | 0.607 |
| Composite (C1) | 0.443 | 0.325 | 0.606 | 0.573 |

## 3. Open issues the handoff understates

These are not errors. They are the questions a reviewer will raise, and they set the next steps.

| # | Issue | Why it matters | Where it is handled |
|---|---|---|---|
| I1 | **Sampling secrecy.** B1's and C1's gain needs the attacker not to know when the candidate was sampled. | Amplification by sampling assumes the sample is secret, and many deployments leak it (the server knows it; cohort announcements). If the gain disappears for a sampling-aware attacker, the defense claim narrows to "secret sampling". | Re-analysis (a), on committed runs: `client-ids` are logged per round |
| I2 | **Subsampling against fewer contributions.** At `q = 0.25` each client trains in about 12.5 of 50 rounds. Early stopping and fewer contributions are known membership-inference defenses. | If an early-stopped full-participation run matches the subsampling curve, the contribution is "fewer contributions", not sampling. | Re-analysis (b), approximate: truncated trajectories |
| I3 | **Accuracy ceiling.** Subsampling reaches at most 0.746 (0.772 with logit adjustment), against 0.803 without noise. | There is no comparison above 0.772, where most users would operate. More rounds at `q = 0.25` might lift the ceiling, at a cost. | A v3 package (GPU) |
| I4 | **One dataset with real leakage, one layout.** Alzheimer leaks at chance even without noise. | Both papers need a second leaking dataset. The 2026-10-10 decision excluded CIFAR-10. | A v3 package; owner decision |
| I5 | **The paper's own setting** (3 clients, Alzheimer) has not been measured under standard v2. | It is the most direct test of the published claim (C3). | A v3 package (small, owner-PC scale) |
| I6 | **Attacker class.** The gated attacker uses shadow losses at 11 of 51 checkpoints and sees only global models. | An attacker using all checkpoints or per-round model deltas could be stronger. Subsampled runs may be more exposed to per-round attacks (I1). | Noted in the v3 measurement standard |
| I7 | **The candidate panel is fixed.** The gate interval is conditional on the 32 candidates. | Claims generalise over clients only through the two-way bootstrap, which is wider. | Reported as is; v3 keeps both intervals |
| I8 | **"Not computable" `mu`.** At noise multipliers below 0.05 the accountant does not finish. | This is a tool limit, not an unknown guarantee: at those multipliers any accountant gives a vacuous bound. The wording should say "vacuous (accountant out of range)". | Wording in the corrected handoff |

## 4. Files

- `tables/handoff_number_check.csv`: every checked token, its handoff line, its status and how it was resolved.
- `tables/fairness_matched_accuracy.csv`: the table above.
- The corrected handoff: `research/cia_roadmap_v2/RESULTS_HANDOFF.md`, with a change note at its top.
