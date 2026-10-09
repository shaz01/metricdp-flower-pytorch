# Comparison with the paper's server-side mechanisms: findings

Protocol (rules fixed first; one addendum before any evaluation cell): [2026-10-09_baseline_comparison.md](2026-10-09_baseline_comparison.md). 1,000 new evaluation cells (Part 1: 450 = global DP 180, metric privacy 180, vanilla 90 on exactly the confirmation cells; Part 2: 550 = 250 + 250 baseline frontier worlds + 50 noise-free vanilla worlds) plus 660 development cells (tuning/calibration), 256 independent releases per evaluation cell, the repository's real `make_strategy("fedavg", ...)` global-DP and metric-privacy classes, same public base, data, gate, recorder and attack as the stacked construction. Reports: `comparison_part1_report`, `comparison_part2_report`, `baseline_tuning.json` in `results/stacked_head/`.

## Headline

**At matched calibrated risk, in the same pipeline and with the same public validation gate, the stacked construction is NOT better than the paper's server-side mechanisms.** Pre-stated verdicts: **mixed against global DP** (global DP better in 7 of 10 groups, ours better in 2, no clear difference in 1) and **mixed against metric privacy** (5 / 2 / 3). The Part 2 claim ("ours above the baseline's utility-versus-practical-AUC curve at 4 of 5 risk levels in both tasks") is **false for both** baselines. At risk .80 the baselines win in all five tasks; at risk .65 the result is task dependent and small.

## Part 1: paired gain in held-out CE over the public-only control (gated V0), fresh public sets and test-split images

D = ours - baseline, mean over public sets with a bootstrap 95% interval over sets. Realized oracle AUC is the known-alternative attack AUC actually realized by the baseline (ours is calibrated to the risk target by construction; global DP realizes the target within .01 because almost every update is clipped).

| Task | Risk | Sets | Ours | Global DP | D vs GDP | Metric priv. | D vs MDP | MDP realized AUC | Non-private (vanilla) |
|---|---|---:|---:|---:|---|---:|---|---:|---:|
| Fashion-MNIST 4-7 | .65 | 10 | +.0202 | +.0311 | −.0109 [−.0192, −.0030] | +.0268 | −.0067 [−.0149, +.0045] | .655 | +.1063 |
| Fashion-MNIST 4-7 | .80 | 10 | +.0208 | +.0634 | −.0426 [−.0618, −.0256] | +.0596 | −.0387 [−.0567, −.0234] | .792 | +.1063 |
| KMNIST 0-3 | .65 | 30 | +.0430 | +.0646 | −.0216 [−.0322, −.0120] | +.0685 | −.0254 [−.0392, −.0129] | .666 | +.2337 |
| KMNIST 0-3 | .80 | 30 | +.0508 | +.1503 | −.0996 [−.1259, −.0758] | +.1496 | −.0988 [−.1273, −.0724] | .803 | +.2337 |
| KMNIST 4-7 | .65 | 30 | +.0787 | +.0763 | +.0025 [−.0037, +.0089] | +.0832 | −.0045 [−.0127, +.0034] | .682 | +.2741 |
| KMNIST 4-7 | .80 | 30 | +.0924 | +.1704 | −.0780 [−.0992, −.0580] | +.1866 | −.0943 [−.1205, −.0702] | .840 | +.2741 |
| MNIST 0-3 | .65 | 10 | +.0220 | +.0164 | **+.0057** [+.0028, +.0085] | +.0159 | **+.0062** [+.0036, +.0087] | .626 | +.0410 |
| MNIST 0-3 | .80 | 10 | +.0251 | +.0328 | −.0077 [−.0141, −.0020] | +.0292 | −.0042 [−.0093, +.0010] | .736 | +.0410 |
| MNIST 4-7 | .65 | 10 | +.0192 | +.0175 | **+.0017** [+.0006, +.0030] | +.0160 | **+.0032** [+.0015, +.0049] | .621 | +.0452 |
| MNIST 4-7 | .80 | 10 | +.0226 | +.0348 | −.0123 [−.0174, −.0081] | +.0307 | −.0081 [−.0132, −.0033] | .729 | +.0452 |

Share of the non-private (vanilla FedAvg of the same local updates) gain that each mechanism retains, mean over task-sets: **ours .25 (q.65) / .29 (q.80); global DP .27 / .64; metric privacy .28 / .62.** At risk .65 the three are similar; at risk .80 the baselines keep about twice as much of the available gain. Baselines have no cell with a negative gated gain (0 of 180 each).

## Exploratory (not pre-stated) observations

- **Metric privacy equals global DP once leakage is matched.** The paired difference MDP − GDP is within about ±.004 CE in 8 of 10 groups (half of the ten intervals include 0). Where it is larger (KMNIST 4-7, +.007 / +.016), MDP's realized AUC was higher than the target (.682 / .840 vs .65 / .80, because its noise multiplier was calibrated on a different dataset). In every group whose interval excludes 0 (five), the sign of the gain difference equals the sign of the realized-AUC difference (MDP leaks more and gains more on KMNIST 4-7, leaks less and gains less on MNIST): gains track realized leakage. In this matched head-update setting the metric-based calibration gives no utility advantage beyond rescaling the noise. Because MDP's realized leakage deviates from the target (.62-.84), its comparison with ours is looser than global DP's.
- **The public validation gate helps the paper's mechanism a lot.** Global DP's mean gain is +.054 gated versus +.034 ungated (the mechanism as published, multiplier 1) at risk .65 and +.122 versus +.052 at .80. The gate is post-processing of the released aggregate and public data, so this is a general add-on for server-side mechanisms and is the part of this project's work that transfers to them; it requires public labelled validation data.
- **Likely reasons for the gap (untested):** each baseline client runs 20-50 local steps and its update is clipped to norm C, while ours sends a single projected, class-balanced gradient direction; baselines noise and use all 132 head parameters, ours 51 projected dimensions; ours also releases an aggregate with the documented 8/7 noise variance at the same calibrated risk (about 14% more variance, 7% more standard deviation), which slightly favours the baselines. Which factor matters was not isolated.

## Part 2: practical-attack frontier (KMNIST, 32 images, sets 0-4, 4 targets, calibrated own-records AUC)

Practical AUC and mean gain by risk target (ours / global DP / metric privacy); noise-free = vanilla.

| Task | Level | Ours | Global DP | Metric privacy |
|---|---|---|---|---|
| KMNIST 0-3 | .55 | .513, +.0067 | .506, +.0070 | .504, +.0068 |
| | .65 | .544, +.0157 | .526, +.0224 | .521, +.0213 |
| | .80 | .585, +.0185 | .549, +.0552 | .545, +.0514 |
| | .90 | .632, +.0198 | .578, +.0499 | .571, +.0474 |
| | .95 | .645, +.0202 | .588, +.0527 | .581, +.0509 |
| | noise-free | .750, +.0213 | .750, +.1003 | .750, +.1003 |
| KMNIST 4-7 | .55 | .503, +.0194 | .508, +.0164 | .509, +.0175 |
| | .65 | .511, +.0409 | .528, +.0432 | .527, +.0480 |
| | .80 | .519, +.0468 | .549, +.0963 | .549, +.1048 |
| | .90 | .530, +.0487 | .550, +.0966 | .547, +.1003 |
| | .95 | .532, +.0493 | .556, +.1040 | .550, +.1071 |
| | noise-free | .600, +.0508 | .600, +.1510 | .600, +.1510 |

On KMNIST 0-3 the baselines have both higher utility and a LOWER practical AUC than ours at every level above .55 (e.g. at .80: .549 versus .585); ours is above the baseline curve at 0 of 5 levels. On KMNIST 4-7 ours is above at 3 of 5 levels, all in a narrow AUC band (.50-.53) where the practical attack barely separates the worlds. The claim is false for both baselines. (Noise-free AUCs are the degenerate single-release ordering and coincide.)

## Checks and limits

- Checks: the baseline strategies are the repository's own (noise scale z*C/n verified by test and in every result file; metric-privacy noise equals the global-DP noise divided by the client distance, verified by test); vanilla release equals the plain average of local updates (test); oracle-AUC formula tested; the freeze, calibration and comparison are reproducible from committed files; a release audit over all 1,980 cells in the results directory passes (15,840 checks), after making the audit tie-aware (one baseline cell had two multipliers within 1e-6 in validation CE, which float32 storage resolves differently from the float64 pipeline; the audit now accepts any pick within float32 resolution and checks the reported value against those bounds); a client-feature cache added mid-run for speed was verified to leave every release array bit-identical.
- Tuning (development only): global DP and vanilla were tuned on Fashion-MNIST 0-3 with 60 / 12 configurations (after a pre-evaluation addendum extended the grid because the first freeze hit the edge); global DP at q.65 selected (lr .03, 20 steps, C .1) and at q.80 (lr .03, 50 steps, C .1); vanilla (lr .03, 50 steps). Steps = 50, the largest grid value, is still an edge selection for q.80 and vanilla (disclosed, not extended). The stacked construction's own freeze had a larger grid (360 configurations per risk) and the same selection rule.
- Scope: the baselines run as a ONE-SHOT head update on the same public base, isolating clipping and noise calibration; the paper's mechanism is a multi-round full-model protocol, which was not run. Baseline clients do 20-50 local steps (more client compute than ours). Offline tuning and public-set choice are unaccounted for both sides. 4-class tasks, small CNN body, Gaussian noise, 8 clients, replicate rounds are independent releases from one base, the practical attacker is weak, the validation set is extra public labelled data for every mechanism.

## What this means for the research question

Your question was whether client-side noise built per client can beat server-side metric calibration. Three results now answer it within this study: (1) per-client noise shapes and a different noise density gave no benefit (earlier phases); (2) the construction that did work against public-only models (stacked head step) is NOT better than the paper's server-side mechanisms at matched calibrated risk, and is clearly worse at risk .80; (3) the paper's metric calibration itself adds nothing over global DP once leakage is matched. What survives is the public-validation gate as a post-processing step that improves any of these mechanisms, and the limited-public findings. Any claim that the stacked construction or client-side noise protects better than the original mechanism is not supported.
