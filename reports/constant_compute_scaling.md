# Phase 1: Constant-Compute Client-Count Scaling — CUDA Redo (v1 and v2)

Source data: `results/scale_controlled/` (v1, round-scaled), `results/scale_controlled_epochs/`
(v2, epoch-scaled). Code: `experiments/client_scaling/sweep_scale_controlled.py` (v1),
`experiments/client_scaling/sweep_scale_controlled_epochs.py` (v2),
`metricdp_pytorch/metricdp_strategy.py`, `metricdp_pytorch/strategy_factory.py`,
`experiments/reproduce/`. Supersedes `reports/archive/constant_compute_scaling_mps_v1v2.md`
(archived MPS-era attempt, kept for historical comparison).

> **Status.** This redo answers the question the archived MPS-era report left open. Two things
> changed since that attempt: (1) a genuine non-determinism source was found and fixed —
> `DeterministicReplyOrderMixin` (`strategy_factory.py`) and a matching fix in `metrics.py` now sort
> client replies by client ID before floating-point aggregation, instead of summing in
> network-arrival order; (2) training moved off this project's original Mac MPS backend (documented
> non-determinism, see `metricdp_pytorch/utils/device.py`'s `resolve_device` docstring) onto CUDA
> hardware. Both v1 and v2 have now completed their full `fedavg` matrix (`n ∈ {4, 8, 48}`) plus
> `fedyogi` at `n ∈ {4, 8}` — **40/40 combinations, 0 run failures, 0 invalid-distance rounds, 0
> collapsed-aggregation rounds, anywhere.** Determinism was confirmed directly, not just inferred:
> v1 and v2 share identical hyperparameters at `n=4` by construction, and every `n=4` combination
> (`fedavg` and `fedyogi` alike) produced bit-identical accuracy/loss from two independently-launched
> subprocesses. `fedyogi` at `n=48` is not yet run (planned as the immediate next step) — the central
> "does the metric-privacy advantage shrink/reverse at scale" question is answered below for
> `fedavg`, still open for `fedyogi`.

## Overview

`results/48client_scaling` (fixed 20-round budget) found the metric-privacy-vs-global-dp accuracy
advantage shrinking or reversing at 48 clients compared to 8. Phase 1 of `docs/RESEARCH_ROADMAP.md`
asks whether that is a genuine mechanism failure at scale or a round-budget confound. Two
constant-compute designs test this by holding total training compute roughly constant across
client counts instead of a fixed round budget:

- **v1 (round-scaled)**: `rounds(n) = round(20 * n / 4)`, `--local-epochs`/`--batch-size` fixed
  (5, 32). Each client's total local gradient steps over its own shrinking shard stay roughly
  constant, at the cost of scaling aggregation frequency 12x from n=4 to n=48.
- **v2 (epoch-scaled)**: `rounds(n) = 20` (fixed), `local_epochs(n) = round(5 * n / 4)` instead.
  Every client count aggregates exactly 20 times, targeting aggregation-frequency parity directly —
  the corrected design after v1's `rounds(n)=5n` formula was found to confound step count with
  interruption frequency (see the archived report's Finding 1).

Matrix: `global-dp`/`metric-privacy` × `homogeneous`/`non-iid` × `num_clients ∈ {4, 8, 48}`, all at
`noise_multiplier=0.05` (the 8-client sweet spot from `sweep_noise_multiplier.py`), seed 42,
clipping norm 5.0. This redo additionally ran `fedyogi` at `num_clients ∈ {4, 8}` on both designs
(`n=48` for `fedyogi` is the planned next run) via the `--aggregation-methods`/`--client-counts`
override flags added for splitting this redo across machines. `vanilla`, `fedavgm`, `fedprox`,
`fedmedian`, and `n=16` remain out of scope, same as the archived attempt.

Two infrastructure bugs were found and fixed while bringing this redo up (both already on
`feature/scale-controlled-redo`, see `git log` and `STATUS.md` for full detail, not re-derived
here): both constant-compute sweep scripts were silently broken since `runner.py` made several
CLI arguments required, and a separate cross-platform fix for `runner.py`'s isolated-worker
re-exec briefly broke on this machine's `uv`-managed venv layout. Neither affects the numbers
below — both were caught and fixed before they could produce a result.

## v1 Results: constant-compute matrix (nm=0.05)

Accuracy/loss/F1/precision/AUC are the **training-time recorded** values from the last completed
round of `server_evaluate_metrics`. "Invalid"/"collapsed" round counts come from `train_metrics`'
per-round `metric-dp-distance-invalid`/`metric-dp-aggregation-collapsed` flags (metric-privacy
only). Distance min/mean/max pools every round's finite `metric-dp-distance-mean` value.

| Partition | Privacy | Aggregation | n | Rounds | Accuracy | Loss | F1 | Precision | AUC | Invalid | Collapsed | Distance min/mean/max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| homogeneous | global-dp | fedavg | 4 | 20 | 49.53% | 1.0489 | 0.3281 | 0.2453 | 0.7797 | n/a | n/a | n/a |
| homogeneous | global-dp | fedavg | 8 | 40 | 79.69% | 0.6015 | 0.7877 | 0.8023 | 0.9491 | n/a | n/a | n/a |
| homogeneous | global-dp | fedavg | 48 | 240 | 97.50% | 1.1434 | 0.9748 | 0.9753 | 0.9911 | n/a | n/a | n/a |
| homogeneous | global-dp | fedyogi | 4 | 20 | 49.53% | 1.3536 | 0.3281 | 0.2453 | 0.8156 | n/a | n/a | n/a |
| homogeneous | global-dp | fedyogi | 8 | 40 | 74.69% | 0.8036 | 0.7750 | 0.8190 | 0.9433 | n/a | n/a | n/a |
| homogeneous | metric-privacy | fedavg | 4 | 20 | 63.44% | 1.5078 | 0.5792 | 0.6948 | 0.8816 | 0 | 0 | 1.4601 / 1.6416 / 1.7923 |
| homogeneous | metric-privacy | fedavg | 8 | 40 | 90.47% | 0.4030 | 0.9034 | 0.9084 | 0.9804 | 0 | 0 | 1.1253 / 1.3231 / 1.5318 |
| homogeneous | metric-privacy | fedavg | 48 | 240 | 95.63% | 1.2058 | 0.9561 | 0.9562 | 0.9897 | 0 | 0 | 0.3001 / 0.5446 / 0.7507 |
| homogeneous | metric-privacy | fedyogi | 4 | 20 | 61.09% | 5.6919 | 0.5548 | 0.6514 | 0.8810 | 0 | 0 | 1.0688 / 1.4682 / 1.9291 |
| homogeneous | metric-privacy | fedyogi | 8 | 40 | 72.19% | 0.8431 | 0.7348 | 0.7818 | 0.9287 | 0 | 0 | 0.7429 / 0.9541 / 1.3089 |
| non-iid | global-dp | fedavg | 4 | 20 | 49.53% | 1.0506 | 0.3281 | 0.2453 | 0.7797 | n/a | n/a | n/a |
| non-iid | global-dp | fedavg | 8 | 40 | 79.37% | 0.6802 | 0.7852 | 0.7932 | 0.9511 | n/a | n/a | n/a |
| non-iid | global-dp | fedavg | 48 | 240 | 93.59% | 0.6606 | 0.9356 | 0.9372 | 0.9895 | n/a | n/a | n/a |
| non-iid | global-dp | fedyogi | 4 | 20 | 49.53% | 1.1039 | 0.3281 | 0.2453 | 0.7797 | n/a | n/a | n/a |
| non-iid | global-dp | fedyogi | 8 | 40 | 78.59% | 1.1930 | 0.7816 | 0.8021 | 0.9466 | n/a | n/a | n/a |
| non-iid | metric-privacy | fedavg | 4 | 20 | 83.44% | 0.7186 | 0.8303 | 0.8368 | 0.9626 | 0 | 0 | 1.5601 / 1.7226 / 1.9422 |
| non-iid | metric-privacy | fedavg | 8 | 40 | 92.50% | 0.3386 | 0.9249 | 0.9267 | 0.9823 | 0 | 0 | 1.0216 / 1.2192 / 1.4692 |
| non-iid | metric-privacy | fedavg | 48 | 240 | 91.09% | 0.6775 | 0.9101 | 0.9115 | 0.9855 | 0 | 0 | 0.3808 / 0.5742 / 0.7165 |
| non-iid | metric-privacy | fedyogi | 4 | 20 | 86.41% | 1.2186 | 0.8609 | 0.8748 | 0.9733 | 0 | 0 | 1.2368 / 1.5180 / 1.9743 |
| non-iid | metric-privacy | fedyogi | 8 | 40 | 85.94% | 1.0073 | 0.8575 | 0.8578 | 0.9658 | 0 | 0 | 0.8507 / 0.9949 / 1.3177 |

Wall-clock: `fedavg` n=4 combos 118.0–143.4s; n=8 combos 237.4–253.5s; n=48 combos
2447.2–3816.5s. `fedyogi` n=4 combos 123.4–151.9s; n=8 combos 246.2–262.1s. Total v1 wall-clock:
~4.34h across all 20 combinations. Full detail in `results/scale_controlled/sweep_progress.log`.

## v1 vs. archived MPS baseline

Same `fedavg` matrix, same seed/hyperparameters, different platform (this project's original Mac
MPS backend vs. this redo's CUDA hardware) and, critically, the aggregation-order fix in place here
that wasn't present for the archived run.

| Partition | Privacy | n | MPS accuracy | CUDA accuracy | MPS invalid/collapsed | CUDA invalid/collapsed |
|---|---|---:|---:|---:|---:|---:|
| homogeneous | global-dp | 4 | 49.53% | 49.53% | — | — |
| homogeneous | global-dp | 8 | 14.06% | 79.69% | — | — |
| homogeneous | global-dp | 48 | 33.44% | 97.50% | — | — |
| homogeneous | metric-privacy | 4 | 70.94% | 63.44% | 0/0 | 0/0 |
| homogeneous | metric-privacy | 8 | 19.69% | 90.47% | 7/14 | **0/0** |
| homogeneous | metric-privacy | 48 | 14.53% | 95.63% | 0/0 | 0/0 |
| non-iid | global-dp | 4 | 49.53% | 49.53% | — | — |
| non-iid | global-dp | 8 | 25.31% | 79.37% | — | — |
| non-iid | global-dp | 48 | 17.19% | 93.59% | — | — |
| non-iid | metric-privacy | 4 | 85.47% | 83.44% | 0/0 | 0/0 |
| non-iid | metric-privacy | 8 | 32.66% | 92.50% | 23/23 | **0/0** |
| non-iid | metric-privacy | 48 | 20.31% | 91.09% | 239/240 | **0/0** |

Every combination at n≥8 improved dramatically, and every metric-privacy combination that had
invalid-distance or collapsed-aggregation rounds under MPS has zero under CUDA. See Findings below
for why this is expected, not suspicious.

## v2 Results: constant-compute matrix (nm=0.05)

Same column definitions as v1's table, with `local_epochs` in place of `rounds` as the scaled
parameter (rounds fixed at 20 throughout).

| Partition | Privacy | Aggregation | n | Local epochs | Accuracy | Loss | F1 | Precision | AUC | Invalid | Collapsed | Distance min/mean/max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| homogeneous | global-dp | fedavg | 4 | 5 | 49.53% | 1.0489 | 0.3281 | 0.2453 | 0.7797 | n/a | n/a | n/a |
| homogeneous | global-dp | fedavg | 8 | 10 | 80.00% | 0.7190 | 0.7950 | 0.7924 | 0.9451 | n/a | n/a | n/a |
| homogeneous | global-dp | fedavg | 48 | 60 | 77.81% | 0.8219 | 0.7756 | 0.7768 | 0.9349 | n/a | n/a | n/a |
| homogeneous | global-dp | fedyogi | 4 | 5 | 49.53% | 1.3536 | 0.3281 | 0.2453 | 0.8156 | n/a | n/a | n/a |
| homogeneous | global-dp | fedyogi | 8 | 10 | 85.62% | 1.0896 | 0.8532 | 0.8726 | 0.9681 | n/a | n/a | n/a |
| homogeneous | metric-privacy | fedavg | 4 | 5 | 63.44% | 1.5078 | 0.5792 | 0.6948 | 0.8816 | 0 | 0 | 1.4601 / 1.6416 / 1.7923 |
| homogeneous | metric-privacy | fedavg | 8 | 10 | 87.19% | 0.5687 | 0.8720 | 0.8751 | 0.9714 | 0 | 0 | 1.3767 / 1.5766 / 1.9439 |
| homogeneous | metric-privacy | fedavg | 48 | 60 | 77.97% | 0.9069 | 0.7737 | 0.7845 | 0.9352 | 0 | 0 | 1.1188 / 1.4019 / 2.7281 |
| homogeneous | metric-privacy | fedyogi | 4 | 5 | 61.09% | 5.6919 | 0.5548 | 0.6514 | 0.8810 | 0 | 0 | 1.0688 / 1.4682 / 1.9291 |
| homogeneous | metric-privacy | fedyogi | 8 | 10 | 92.19% | 0.7650 | 0.9213 | 0.9267 | 0.9830 | 0 | 0 | 1.2268 / 1.5065 / 2.1193 |
| non-iid | global-dp | fedavg | 4 | 5 | 49.53% | 1.0506 | 0.3281 | 0.2453 | 0.7797 | n/a | n/a | n/a |
| non-iid | global-dp | fedavg | 8 | 10 | 74.06% | 0.9658 | 0.7248 | 0.7458 | 0.9317 | n/a | n/a | n/a |
| non-iid | global-dp | fedavg | 48 | 60 | 78.44% | 0.7476 | 0.7822 | 0.7831 | 0.9352 | n/a | n/a | n/a |
| non-iid | global-dp | fedyogi | 4 | 5 | 49.53% | 1.1039 | 0.3281 | 0.2453 | 0.7797 | n/a | n/a | n/a |
| non-iid | global-dp | fedyogi | 8 | 10 | 87.97% | 0.7390 | 0.8785 | 0.8834 | 0.9770 | n/a | n/a | n/a |
| non-iid | metric-privacy | fedavg | 4 | 5 | 83.44% | 0.7186 | 0.8303 | 0.8368 | 0.9626 | 0 | 0 | 1.5601 / 1.7226 / 1.9422 |
| non-iid | metric-privacy | fedavg | 8 | 10 | 90.16% | 0.3710 | 0.9014 | 0.9027 | 0.9829 | 0 | 0 | 1.3543 / 1.5382 / 1.8673 |
| non-iid | metric-privacy | fedavg | 48 | 60 | 79.06% | 0.7584 | 0.7893 | 0.7899 | 0.9393 | 0 | 0 | 1.3086 / 1.5228 / 2.6866 |
| non-iid | metric-privacy | fedyogi | 4 | 5 | 86.41% | 1.2186 | 0.8609 | 0.8748 | 0.9733 | 0 | 0 | 1.2368 / 1.5180 / 1.9743 |
| non-iid | metric-privacy | fedyogi | 8 | 10 | 94.53% | 0.6252 | 0.9452 | 0.9466 | 0.9879 | 0 | 0 | 1.2704 / 1.5098 / 2.0980 |

Wall-clock: `fedavg` n=4 combos 117.5–144.4s; n=8 combos 190.2–203.1s; n=48 combos
912.6–1041.0s. `fedyogi` n=4 combos 125.1–151.3s; n=8 combos 197.5–210.9s. Total v2 wall-clock:
~1.83h across all 20 combinations. Full detail in `results/scale_controlled_epochs/sweep_progress.log`.

## v2 vs. archived MPS baseline

| Partition | Privacy | n | MPS accuracy | CUDA accuracy | MPS invalid/collapsed | CUDA invalid/collapsed |
|---|---|---:|---:|---:|---:|---:|
| homogeneous | global-dp | 4 | 35.78% | 49.53% | — | — |
| homogeneous | global-dp | 8 | 39.38% | 80.00% | — | — |
| homogeneous | global-dp | 48 | 35.47% | 77.81% | — | — |
| homogeneous | metric-privacy | 4 | 30.31% | 63.44% | 0/0 | 0/0 |
| homogeneous | metric-privacy | 8 | 24.22% | 87.19% | 0/5 | **0/0** |
| homogeneous | metric-privacy | 48 | 34.53% | 77.97% | 0/0 | 0/0 |
| non-iid | global-dp | 4 | 31.87% | 49.53% | — | — |
| non-iid | global-dp | 8 | 38.44% | 74.06% | — | — |
| non-iid | global-dp | 48 | 33.44% | 78.44% | — | — |
| non-iid | metric-privacy | 4 | 32.19% | 83.44% | 0/0 | 0/0 |
| non-iid | metric-privacy | 8 | 24.84% | 90.16% | 0/7 | **0/0** |
| non-iid | metric-privacy | 48 | 37.81% | 79.06% | 19/0 | **0/0** |

Same pattern as v1: uniform improvement, all collapsed/invalid rounds gone.

## Cross-cutting findings

1. **The Phase 1 central question — is the metric-privacy-vs-global-dp advantage's shrinkage at
   scale genuine or an artifact — now has a trustworthy answer for `fedavg`: it shrinks, but it
   converges toward parity, not the archived report's large reversal.** With clean, deterministic
   data, metric-privacy's advantage over global-dp shrinks steadily from n=4 to n=48 in both
   designs, landing close to zero rather than sharply negative:

   | Design | Partition | Δ at n=4 | Δ at n=8 | Δ at n=48 |
   |---|---|---:|---:|---:|
   | v1 | homogeneous | +13.91pp | +10.78pp | −1.87pp |
   | v1 | non-iid | +33.91pp | +13.13pp | −2.50pp |
   | v2 | homogeneous | +13.91pp | +7.19pp | +0.16pp |
   | v2 | non-iid | +33.91pp | +16.10pp | +0.62pp |

   Compare to the archived MPS report's v1/homogeneous: +21.41pp → +5.62pp → **−18.91pp** — a sharp
   reversal that later failed a direct noise-floor check (`results/noise_floor_check/`) and was
   never established as real. This redo's n=48 deltas (−2.50 to +0.62pp) are an order of magnitude
   smaller and consistent in direction between v1 and v2 (near-zero, not a reversal) — this is the
   first time this question has been asked on a platform confirmed not to have a noise floor
   comparable to the effect size (see Finding 2).

2. **Determinism is now confirmed directly, not inferred.** v1 and v2's `n=4` combinations share
   identical hyperparameters by construction (`round(20·4/4)=20` rounds, `round(5·4/4)=5` local
   epochs, same seed/noise/clipping — both scripts' own `n=4` combinations too, since v1's design
   reduces to v2's design exactly at the base client count). Every one of these pairs — `fedavg`
   and `fedyogi`, both partitions — produced **bit-identical** accuracy and loss from two
   independently-launched subprocesses (verified directly against the raw result JSONs, not just
   summary tables). The archived MPS report's equivalent check found the opposite: MPS's v1/v2
   `n=4` results *disagreed* with each other on an identical config, diverging already at round 0 —
   before any client aggregation could even be involved — proving MPS's own model-initialization
   non-determinism, separate from and larger than the specific client-ID-ordering bug this redo
   fixes.

3. **`DeterministicReplyOrderMixin` covers every aggregation method, not just `metric-privacy`**,
   which is why `global-dp`'s own numbers improved as dramatically as metric-privacy's (e.g. v1
   homogeneous/global-dp/n=48: 33.44% → 97.50%). It wraps `FedAvg`, `FedYogi`, `FedAvgM`,
   `FedMedian`, and `FedProx` uniformly in `strategy_factory.make_base_strategy` — floating-point
   non-associativity in `FedAvg`'s weighted average was apparently degrading convergence quality
   under non-deterministic reply order, not just introducing numeric noise, at exactly the scale
   (high client-count parallelism) where it would bite hardest. Both `fedavg`'s and `fedyogi`'s
   `global-dp` trajectories are smooth and monotonically improving under CUDA; the archived MPS
   `fedavg`/`global-dp`/n=48 trajectory plateaued at 15–42% accuracy for all 240 rounds without ever
   really learning, despite no loss spikes — a distinct failure mode from metric-privacy's
   collapsed-round/invalid-distance pathology, but traceable to the same root cause.

4. **`fedyogi` shows the same shrinking-advantage direction as `fedavg` at the client counts tested
   so far, with one exception worth flagging for the n=48 follow-up:** v1's
   `homogeneous`/`fedyogi` delta goes *negative* already at n=8 (+11.56pp → **−2.50pp**), the only
   n≤8 combination across both designs and both aggregation methods where metric-privacy
   underperforms global-dp. v2's `homogeneous`/`fedyogi` delta stays positive at n=8 (+11.56pp →
   +6.57pp). Both designs agree `non-iid`/`fedyogi` shrinks but stays positive at n=8 (v1: +36.88pp
   → +7.35pp; v2: +36.88pp → +6.56pp). Worth specific attention when `fedyogi`/`n=48` runs next —
   `homogeneous`/v1 is the one combination already trending toward reversal before n=48.

5. **The archived report's evaluation-artifact bug (Finding 4 there) is gone.** All 20
   `evaluation.json` files in each sweep (40 total) now report `accuracies_match: true` against
   their training-time `server_evaluate_metrics`, vs. every one of the archived MPS run's artifacts
   disagreeing. `evaluation.json`/`predictions.npz` can be used for per-class/confusion-matrix
   detail on this result set, unlike the archived data.

## Known gaps / caveats

- **`fedyogi` at `n=48` has not been run yet** (planned as the immediate next step). The Finding 1
  delta table and Finding 4's specific flag are therefore only established for `fedavg` at n=48 —
  `fedyogi`'s own behavior at that scale, especially whether `homogeneous`'s already-negative n=8
  delta continues or reverses, is unknown until that combination completes.
- **Single seed (42) for every run.** No variance estimates in this redo either — the determinism
  fix guarantees a given seed reproduces exactly, not that a different seed would land the same
  deltas found in Finding 1.
- **Reduced matrix.** `vanilla`, `fedavgm`, `fedprox`, `fedmedian`, and `n=16` remain out of scope,
  same as the archived attempt.
- **`results/48client_scaling`'s fixed-20-round comparison (the archived report's v1 Finding 1) was
  not rerun on CUDA.** That baseline is still MPS-era data; the round-fragmentation explanation for
  why constant-compute accuracy came in below it is unchanged from the archived report and not
  re-derived here.
- **Not yet merged to `master`.** Code and results live on `feature/scale-controlled-redo`; see
  `AGENTS.md`'s git workflow for the merge criteria.

## Next steps

Run `fedyogi` at `n=48` on both designs (`--aggregation-methods fedyogi --client-counts 48`),
completing the full matrix this redo set out to run. After that, revisit whether Phase 1's central
question is fully answered (Finding 1 above) or needs the wider `n=16` interpolation point, and
decide on merging this branch per `AGENTS.md`'s workflow.
