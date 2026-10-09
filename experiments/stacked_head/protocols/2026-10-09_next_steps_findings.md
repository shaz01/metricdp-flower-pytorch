# Next steps A-D: accuracy-aware gate, public-budget map, cohorts, practical-attack frontier

Protocols (rules fixed before the cells they govern): [accuracy-aware gate](2026-10-09_accuracy_aware_gate.md), [budgets and cohorts](2026-10-09_budgets_and_cohorts.md), [model-based CIA frontier](2026-10-09_model_based_cia_frontier.md) (two amendments, below). 980 cells in total (180 dev reruns, 180 confirmation, 200 budget map, 120 cohorts, 300 frontier), 256 independent one-shot releases per cell (1 for noise-free), real ServerApp/ClientApp code, in-process backend (bit-identical to the Ray simulation on the cells compared earlier). Checks: the 180 dev reruns reproduce the previously committed results exactly (0 differ, max per-release difference 0); `audit_releases.py` passes for every cell (180 + 500 + 300 cells; 1,440 + 4,000 + 2,400 checks); the practical-attack AUC of one frontier group was recomputed independently with a Mann-Whitney statistic (0.585, equal to the report); the attack statistics are unit-tested against a direct recomputation; the data layout refactor is pinned by a test (earlier roles unchanged) and a re-run of a committed cell was identical. Reports: `gate_dev_report`, `gate_confirmation_report`, `budget_map_report`, `cohort_report`, `frontier_report` (`.json`/`.txt`) in `results/stacked_head/`.

## A. Accuracy-aware gate

Variants (all post-processing of validation scores, same privacy accounting as V0): V0 argmin validation CE; V1 non-inferior validation accuracy then argmin CE; V2 highest validation accuracy among CE-improving multipliers; V3 non-inferiority with a 1-point slack.

**Development (180 cells, data already seen):**

| Variant | Groups reliable | Cells with accuracy delta < -0.005 | Worst accuracy delta | Smallest per-group median-gain retention vs V0 |
|---|---:|---:|---:|---:|
| V0 | 10/10 | 11/180 (6.1%) | -2.63 pts | 1.00 |
| V1 | 10/10 | 2/180 (1.1%) | -1.28 pts | 0.42 |
| V2 | 10/10 | 1/180 (0.6%) | -0.80 pts | 0.42 |
| V3 | 10/10 | 5/180 (2.8%) | -1.55 pts | 0.74 |

The pre-stated rule requires a variant to keep at least 80% of V0's median CE gain in every group; none does (V3 comes closest at 74%). **Selected: V0, no accuracy-aware gate adopted.** The variants buy fewer accuracy dips at a large price in CE gain.

**Confirmation (fresh public sets 30-59 and 10-19, fresh test-split evaluation images, 180 cells; looked at only after the choice):** V0 meets the reliability rule in **10 of 10 groups** (confirmed; 9 required): p_win .80-1.00, p_loss .00, medians +.009 to +.058 CE (KMNIST 0-3 .032/.037, KMNIST 4-7 .051/.058, MNIST .026/.029 and .012/.016, Fashion-MNIST .011/.009 at q.65/.80). Descriptive: V0 accuracy dips 16/180 (8.9%, worst -2.46 pts); V1 6.7%, V2 5.6%, V3 6.7%. The development-time dip reductions (V1 -82%, V2 -91%) largely did not replicate on fresh data (-25%, -37%), which supports not adopting them. The headline claim stands on data that played no role in any design choice.

## B. Public-budget map (5 tasks x 10 sets x 2 risks per budget, cohort A)

| Public images | Groups reliable | Pooled p_win | Pooled p_loss | Median gain (CE) | Mean gain |
|---:|---:|---:|---:|---:|---:|
| 32 | 9/10 (reliable) | .91 | .01 | +.0155 | +.0299 |
| 128 | 10/10 (reliable) | .85 | .00 | +.0045 | +.0077 |
| 512 | 0/10 (not reliable) | .33 | .02 | +.0003 | +.0009 |

Fade budget (smallest budget with pooled p_win < .5): **512**. The median gain falls from +.0155 at 32 images to +.0045 at 128 (3.4x) and to +.0003 at 512 (15x further); at 128 images it is still reliable but small (+.0025 to +.0113 per group median). The one 32-image group that misses the rule on these 10 sets is KMNIST 4-7 at q.80 (p_win .70, p_loss .10; it passes with 30 sets).

## C. Cohorts (KMNIST, 32 images, 10 sets x 4 cohorts)

Variance components of the per-cell gain: **public set 98-99%**, cohort about 0% (cohort/set variance ratio .000-.004), residual (set x cohort interaction plus release noise) 1-2%. The pre-stated claim (cohort variation at most 10% of public-set variation in all four groups) holds. Across all 40 (set x cohort) cells per group p_win is .82-.90 and p_loss .00-.03. For this label-stress design the earlier single-cohort sweeps are representative of the private-data side; what varies is the public base.

## D. Practical model-only attack and the utility-leakage frontier (KMNIST, 32 images, sets 0-4, 4 targets, constant frozen q.65 configuration at all risks)

Protocol amendments, both made BEFORE any frontier cell was run, after two 128-release sanity pairs (KMNIST 0-3, set 0, risk .95, target 0): the class-mix statistic gave AUC 0.42 and the raw own-records statistic 0.43 (both below chance: queries are class-balanced by design and IN/OUT releases differ in overall quality), so the primary statistic became the calibrated own-records statistic (CE decrease on the target's records minus that on class-mix-matched public shadow records), which gave 0.575 on the same pair. Those two pairs are the only attack data seen before the grid was fixed.

| Task / risk target | Calibrated AUC (95% CI over sets) | Utility gain over control | Accuracy delta |
|---|---|---:|---:|
| KMNIST 0-3 / .55 | .513 [.509, .520] | +.0067 | +0.21 pts |
| KMNIST 0-3 / .65 | .544 [.533, .557] | +.0157 | +0.42 |
| KMNIST 0-3 / .80 | .585 [.560, .607] | +.0185 | +0.49 |
| KMNIST 0-3 / .90 | .632 [.584, .683] | +.0198 | +0.50 |
| KMNIST 0-3 / .95 | .645 [.583, .707] | +.0202 | +0.52 |
| KMNIST 0-3 / noise-free | .750 [.550, .950] (single deterministic release per world) | +.0213 | +0.58 |
| KMNIST 4-7 / .55 | .503 [.495, .512] | +.0194 | +0.47 |
| KMNIST 4-7 / .65 | .511 [.494, .531] | +.0409 | +0.81 |
| KMNIST 4-7 / .80 | .519 [.482, .569] | +.0468 | +0.98 |
| KMNIST 4-7 / .90 | .530 [.470, .590] | +.0487 | +1.04 |
| KMNIST 4-7 / .95 | .532 [.452, .611] | +.0493 | +1.07 |
| KMNIST 4-7 / noise-free | .600 [.500, .800] (single deterministic release) | +.0508 | +1.12 |

Pre-stated claims: (1) **the calibrated target upper-bounds the practical attack at every risk on both tasks: true** (every bootstrap upper limit is below the target; the bound is very loose in practice, e.g. .65 vs .544 and .95 vs .645); (2) the practical attack is statistically detectable (interval excludes 0.5) on KMNIST 0-3 at every level, including .55 (.513 [.509, .520], practically negligible), and on none of the KMNIST 4-7 levels. (3) Frontier: utility rises with the risk target and saturates; for KMNIST 0-3 about a third of the noise-free gain is available at .55 and 74% at .65, while the practical attacker's AUC at .65 is .544 versus .645 at .95 and .750 noise-free.

## Caveats

- **Accuracy:** CE improves reliably but accuracy dips (worse than -0.5 pts) in about 6-9% of cells and by up to about -2.5 pts; no variant in the tested family fixes this without sacrificing the CE gain. The gate still depends on public validation data (512 images; 128 was adequate earlier).
- **Attack evidence is weak by construction.** One natural model-only statistic (the strongest practical attacker was not searched for: no shadow models, no repeated observation of one federation), releases are independent draws from one base, noise-free AUC is a degenerate single-release ordering, and the attack design was shaped by two sanity runs. Absence of detectability is not a privacy guarantee; the guarantee is the calibrated conditional contract.
- Frontier uses the q.65 configuration at all risks (the frozen q.55 entry has a different cap/step); 5 public sets and 4 targets per task.
- Scope unchanged: 4-class tasks, small CNN head step, Gaussian noise only, single cohort per set in A/B, in-process replicate releases (not a multi-round protocol), no secure aggregation, offline tuning and base selection unaccounted. Nothing here addresses non-Gaussian or client-specific noise densities (no benefit was found earlier).
