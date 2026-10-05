# Frozen slot profiles: reproduce the oracle, fail roster reassignment

2026-10-05. Owner: “lets move to that”. This completes the approved bounded diagnostic, not the overall project experiment.

The fixed configuration **exactly reproduces** the raw-information oracle in all three original-roster primary cases. No fresh profile construction or radius/shrink tuning is needed for that equality. However, the fresh-test gains are **0.000938–0.000998 CE**, below the unchanged 0.001 feasibility gate. Moving the same clients to different registered slots turns the fixed configuration harmful relative to the tuned shared profile. The previous positive result therefore does not establish that a private distribution estimator is needed or worthwhile; its slot-label alignment is a substantive confound.

## Protocol and reproducibility

Read the [protocol frozen before fresh evaluation](2026-10-05_public_slot_protocol.md), [independent mathematical/result review](2026-10-05_public_slot_math_review.md), and preceding [broader oracle findings](2026-10-05_broader_oracle_findings.md). Calculator: [public_slot_profile_probe.py](../calculations/public_slot_profile_probe.py). Artifacts: [settings/measurements](../../results/client_specific_noise/public_slot_profile_probe.json) and [paired trials and evaluation indices](../../results/client_specific_noise/public_slot_profile_trials.npz).

Freeze assignment **[0,2,0,1,0,2,0,1]**, radius **0.3**, shrink **1**, from the previous three-bias label-stress/round 20/label 8 cases. Four arms distinguish construction from calibration:

| Arm | Profile assignment | Radius/shrink |
|---|---|---|
| Frozen configuration | Prior fixed per-slot pattern | Fixed at 0.3/1 |
| Frozen assignment, tuned calibration | Same fixed pattern | Raw-development optimized |
| Tuned shared | One common profile | Raw-development optimized |
| Tuned oracle | Exact 4⁸ assignment search | Raw-development optimized |

The frozen configuration reports its development score only descriptively; the score does not change its parameters. “Public/frozen” means fixed for this comparison, **not** certified data-independent calibration: its original discovery used private raw diagnostics. Source trajectories, weights and the three tuning arms remain unprotected; noise labels 4/8/16 are fixed-profile counterfactuals, not complete-protocol privacy budgets.

Use the same cached Fashion-MNIST reduced classifier, three seeds, balanced/quantity/explicit label-stress partitions and rounds 5/20. Only three fixed class-bias contrasts change. The next official test slice, per-class positions 512–767, is disjoint from both earlier test slices. Use 2048 paired Laplace draws per four-arm comparison. Training data/checkpoints/test examples remain shared across seeds; intervals capture conditional perturbation-noise uncertainty, not independent population uncertainty.

Original roster gives 18 seed/partition/checkpoint records. For label stress, six additional records cyclically move clients through order[1,2,3,4,5,6,7,0], carrying their updates and weights but retaining fixed profiles at the registered slots. The model checkpoint and raw aggregate learning step remain unchanged. Total: **24 records,72 noise-label comparisons,288 arm evaluations**. Shared/oracle development optima are checked invariant to this reassignment. Separate roster rows have different sampled innovations, so only their within-row contrasts are paired comparisons.

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m research.calculations.public_slot_profile_probe --self-check`, then the same command without `--self-check`. `--cache-dir PATH` overrides the offline Arrow cache. Source hashes must match the saved audit; no training, data download or CIA run occurs.

## Primary fresh-test results

Primary cases were specified before evaluation: label stress, original roster, round 20, noise label 8. Positive gain means tuned-shared CE minus frozen CE.

| Seed | Tuned shared CE | Frozen = oracle CE | Frozen gain | Noise-only 95% gain interval | Frozen minus oracle |
|---|---:|---:|---:|---|---:|
| 42 | 1.227707433 | 1.226748016 | 0.000959417 | [0.000931143,0.000987690] | 0 exactly |
| 43 | 1.225843982 | 1.224845655 | 0.000998326 | [0.000968926,0.001027727] | 0 exactly |
| 44 | 1.229866168 | 1.228928059 | 0.000938109 | [0.000910504,0.000965715] | 0 exactly |

All three matching gaps satisfy the predeclared 0.0001 absolute frozen-oracle tolerance: paired loss arrays are identical. This is equality of output laws/configurations in these cases, not a confidence-based approximate match. All three frozen configurations, tuned fixed assignments and oracle configurations use the same assignment/radius/shrink. Equality is expected once their selected settings match and shared innovations are used.

**Zero of three primary point gains passes 0.001**, and no primary noise-only interval lies wholly above that threshold. Across all 72 comparisons, no frozen-configuration point gain reaches 0.001. This does not mean there is zero utility benefit: a small original-roster gain persists. It means the earlier materiality gate does not survive this fresh evaluation. Do not lower the gate or search another test slice to recover the favorable headline.

## Roster-shift control

At round 20/label 8, reassigning the same stress clients produces:

| Seed | Frozen minus shared CE (positive is worse) | Tuned fixed assignment minus shared CE | Frozen minus oracle CE |
|---|---:|---:|---:|
| 42 | +0.001747878 | +0.000613194 | +0.002683190 |
| 43 | +0.001768534 | +0.000636399 | +0.002774719 |
| 44 | +0.001782625 | +0.000558923 | +0.002757198 |

Development-optimal oracle assignments rotate to[2,0,1,0,2,0,1,0]. Their development optimum and noiseless evaluated update remain invariant, as expected under mere client relabeling. The frozen slot assignment does not follow clients, so its clipping mean changes while its noise penalty stays the same in this equally weighted stress population. Its noiseless loss is worse too: this is a genuine mean/clipping mismatch, not solely noise Monte Carlo variation.

Tuning radius/shrink while keeping the mismatched profile assignment partly reduces harm, but cannot recover the oracle effect; all three remain worse than tuned shared. The primary original-versus-shift intervention isolates attachment of profiles to **slots**, not a new dataset, model trajectory or client population. One deliberately chosen cyclic shift does not quantify arbitrary roster robustness, but suffices to expose failure of this fixed pattern under this reassignment.

## All frozen-control outcomes

Each cell below contains 18 comparisons across seeds, two checkpoints and three noise labels. Values are frozen minus tuned-shared CE; positive values indicate harm.

| Partition / roster | Minimum | Maximum | Mean |
|---|---:|---:|---:|
| Balanced / original | +0.000063462 | +0.001424448 | +0.000568530 |
| Quantity skew / original | +0.000080113 | +0.001613038 | +0.000629923 |
| Label stress / original | −0.000998326 | −0.000096462 | −0.000677400 |
| Label stress / shifted | +0.001747878 | +0.002548348 | +0.002054223 |

The frozen pattern benefits every original label-stress comparison, harms every balanced/quantity comparison, and harms every shifted label-stress comparison in this grid. It is not a generally beneficial public heterogeneity policy. These descriptive multiple looks are exploratory, not corrected significance tests.

## Meaning for construction research

The broader oracle's same-profile-versus-personalized comparison omitted a relevant control: publicly frozen heterogeneity. In its aligned primary cases that control now reproduces the oracle without any new private estimator. The difference can be explained by the constructed relationship between client labels and registered slot positions; it cannot yet support a unique client noise distribution or the value of covariance estimation.

The shifted oracle still adapts its assignment to client content, so a client-sensitive rule may have an opportunity after the slot shortcut is removed. That is a **new feasibility question**, not an implemented defense. The gain is small, below the fresh threshold, and must cover the cost of obtaining/privatizing the selection statistic and the shared utility information. A bias-only correction also cannot leave the remaining model transcript unprotected while claiming dataset-contribution privacy.

Next proposed step: a bounded **construction and accounting design review** of a permutation-equivariant client-local rule. For example, ask whether a protected coarse client statistic can route among public profiles, with the frozen per-slot pattern and tuned shared controls retained. Establish the complete observer law and a legitimate shared calibration signal; estimate the extra protection cost against this measured headroom before any estimator implementation. Existing separately paid RR diagnostics were unfavorable, so a renewed selector needs a concrete reason to avoid that failure. Do not launch another sampler or CIA sweep merely because an oracle advantage exists.

No mechanism is selected here, no novel density or privacy proof is claimed, and no experiment-completion decision is inferred. The frontier remains the eventual model-utility/leakage comparator, with its established evaluation qualifications.
