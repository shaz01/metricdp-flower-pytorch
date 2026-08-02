# Phase 1 Diagnosis: Constant-Compute Client-Count Scaling

Source data: `results/scale_controlled/`.
Code: `experiments/client_scaling/sweep_scale_controlled.py`, `metricdp_pytorch/metricdp_strategy.py`,
`experiments/reproduce/`.

## Overview

`results/48client_scaling` (fixed 20-round budget, reused from the 8-client `nm=0.05` sweet spot)
found the metric-privacy-vs-global-dp accuracy advantage shrinking or reversing at 48 clients
compared to 8. Phase 1 of `docs/RESEARCH_ROADMAP.md` asks whether that is a genuine mechanism
failure at scale, or an artifact of holding rounds fixed while each client's per-round data shrinks
roughly as `1/num_clients`. This sweep disentangles the two by scaling `--rounds` proportionally
with client count instead:

```
rounds(n) = round(BASE_ROUNDS * n / BASE_NUM_CLIENTS), BASE_ROUNDS=20, BASE_NUM_CLIENTS=4
```

so each client's total local gradient steps over its own data stay roughly constant across client
counts. `--local-epochs`/`--batch-size` (5, 32) are held fixed, matching the paper's architecture,
learning rate 0.001, clipping norm 5.0, seed 42.

The matrix actually run is deliberately narrower than the full paper-reproduction grid: only
`global-dp` and `metric-privacy` (not `vanilla`, which isn't part of the gap being measured), only
`fedavg` (not `fedyogi`/`fedavgm`/`fedprox`/`fedmedian`/`fedopt`), at `noise_multiplier=0.05` (the
8-client sweet spot from `sweep_noise_multiplier.py`), across `num_clients ∈ {4, 8, 48}` — `n=16`
was deferred as an interpolation point not needed for the yes/no confound question. This reduction
was a deliberate time-budget tradeoff (see the 2026-08-01 note in `docs/RESEARCH_ROADMAP.md` Phase
1 item 1), not a methodology choice motivated by the results below. 12 combinations total: 2
partition modes × 2 privacy modes × 3 client counts, `fedavg` only.

This sweep also surfaced and fixed a chain of infrastructure bugs unrelated to the underlying
research question (an MPS unified-memory leak, per-round dataset-reload races under concurrent
multi-actor access, and an uncaught `ZeroDivisionError` in Flower's own clipping code on a
collapsed-update round) — those are `feature/scaling-diagnosis`'s earlier commits and are not
re-derived here. All 12 combinations below are from each combo's final, post-fix, successful
attempt; earlier failed attempts during the same sweep are not reflected in the wall-clock timings.

## Results: constant-compute matrix (nm=0.05, fedavg)

Accuracy/loss/F1/precision/AUC are the **training-time recorded** values from the last completed
round of `server_evaluate_metrics` — see "Evaluation artifacts are not usable for this sweep"
below for why the separate `evaluation.json`/`predictions.npz` postprocessed numbers are not used.
"Invalid" and "collapsed" round counts come from `train_metrics`' per-round
`metric-dp-distance-invalid`/`metric-dp-aggregation-collapsed` flags (metric-privacy only).
Distance min/mean/max pools every round's `metric-dp-distance`/`metric-dp-distance-mean` value,
excluding rounds where that value was itself non-finite.

| Partition | Privacy | n | Rounds | Accuracy | Loss | F1 | Precision | AUC | Invalid dist. rounds | Collapsed rounds | Distance min/mean/max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| homogeneous | global-dp | 4 | 20 | 49.53% | 1.0493 | 0.3281 | 0.2453 | 0.7797 | n/a | n/a | n/a |
| homogeneous | global-dp | 8 | 40 | 14.06% | 1.2760 | 0.0347 | 0.0198 | 0.6048 | n/a | n/a | n/a |
| homogeneous | global-dp | 48 | 240 | 33.44% | 0.4933 | 0.1676 | 0.1118 | 0.7991 | n/a | n/a | n/a |
| homogeneous | metric-privacy | 4 | 20 | 70.94% | 1.1314 | 0.6913 | 0.6985 | 0.9180 | 0 | 0 | 1.4563 / 1.6679 / 2.1084 |
| homogeneous | metric-privacy | 8 | 40 | 19.69% | 54.8583 | 0.0648 | 0.0388 | 0.4928 | 7 | 14 | 0.0000 / 0.3592 / 1.4450 |
| homogeneous | metric-privacy | 48 | 240 | 14.53% | 0.5857 | 0.1419 | 0.1786 | 0.6738 | 0 | 0 | 0.3423 / 0.3699 / 1.0934 |
| non-iid | global-dp | 4 | 20 | 49.53% | 1.0484 | 0.3281 | 0.2453 | 0.7797 | n/a | n/a | n/a |
| non-iid | global-dp | 8 | 40 | 25.31% | 1.3124 | 0.1023 | 0.0641 | 0.6466 | n/a | n/a | n/a |
| non-iid | global-dp | 48 | 240 | 17.19% | 1.1151 | 0.0504 | 0.0295 | 0.6084 | n/a | n/a | n/a |
| non-iid | metric-privacy | 4 | 20 | 85.47% | 0.5427 | 0.8516 | 0.8573 | 0.9707 | 0 | 0 | 1.7263 / 1.6163 / 2.2542 |
| non-iid | metric-privacy | 8 | 40 | 32.66% | 23.4717 | 0.1608 | 0.1066 | 0.6754 | 23 | 23 | 0.0000 / 0.2645 / 1.6907 |
| non-iid | metric-privacy | 48 | 240 | 20.31% | 0.4509 | 0.0686 | 0.0413 | 0.7089 | 239 | 0 | 1.2603 / n/a / 1.2603 |

Wall-clock for the final successful attempt of each combo: n=4 combos 825.9–830.5s; n=8 combos
409.4–420.2s (`homogeneous`/`non-iid` × `global-dp`/`metric-privacy`); n=48 combos 3036.9–3757.9s
(`non-iid/global-dp` fastest, `homogeneous/metric-privacy` slowest). Full detail in
`results/scale_controlled/sweep_progress.log`.

## Comparison against the fixed-20-round n=48 baseline

`results/48client_scaling` ran the same `nm=0.05`, `fedavg` combinations at 48 clients but with the
fixed 20-round budget every prior sweep used, instead of this sweep's proportionally-scaled 240
rounds.

| Partition | Privacy | Fixed 20 rounds (accuracy) | Constant-compute 240 rounds (accuracy) |
|---|---|---:|---:|
| homogeneous | global-dp | 62.03% | 33.44% |
| homogeneous | metric-privacy | 62.66% | 14.53% |
| non-iid | global-dp | 61.56% | 17.19% |
| non-iid | metric-privacy | 63.75% | 20.31% |

| Partition | Δ (metric-privacy − global-dp), fixed 20 rounds | Δ, constant-compute 240 rounds |
|---|---:|---:|
| homogeneous | +0.63pp | **−18.91pp** |
| non-iid | +2.19pp | +3.12pp |

## Findings

1. **Scaling rounds to compensate for less per-client data made accuracy worse, not better, for
   both mechanisms.** The constant-compute design's premise was that 240 rounds at 48 clients would
   let each client complete roughly the same total local gradient steps as 20 rounds at 4 clients,
   isolating "does the mechanism degrade with n" from "is there just less data per round." Instead,
   accuracy at 48 clients *dropped* substantially versus the fixed-20-round baseline for every
   partition/privacy combination — global-dp fell 62%→33% (homogeneous) and 62%→17% (non-iid);
   metric-privacy fell 63%→15% (homogeneous) and 64%→20% (non-iid). The most likely explanation:
   DP noise is added once per round regardless of round count
   (`MetricPrivacyServerSideFixedClipping._add_noise_to_aggregated_arrays`,
   `metricdp_pytorch/metricdp_strategy.py`), so 240 rounds injects roughly 12x the total noise of 20
   rounds. The constant-compute design held local training steps fixed but let total accumulated
   privacy noise scale freely — a real confound the original Phase 1 plan did not account for, and
   one that dominates any signal from the round-budget question the sweep set out to answer.

2. **The metric-privacy advantage over global-dp shrinks monotonically from n=4 to n=8 in both
   partitions, then diverges sharply at n=48.** Homogeneous: +21.41pp (n=4) → +5.62pp (n=8) →
   **−18.91pp** (n=48). Non-iid: +35.94pp (n=4) → +7.34pp (n=8) → +3.12pp (n=48). The non-iid
   partition retains a small positive advantage all the way to 48 clients under constant compute;
   the homogeneous partition reverses into a large deficit. This is a stronger, partition-dependent
   version of the shrinking-advantage pattern already seen under the fixed-round design.

3. **Two distinct metric-privacy failure modes are visible in the round-level data, and neither
   fully explains the homogeneous/n=48 reversal.** At n=8, both partitions hit frequent
   zero-norm-update collapse rounds (caught non-fatally by the fix in `metricdp_strategy.py`'s
   `aggregate_train`, previously an uncaught `ZeroDivisionError`) — 14/40 rounds (homogeneous),
   23/40 rounds (non-iid) — alongside catastrophic loss blowup (54.86, 23.47) even though the run
   survives to report a final accuracy. But `homogeneous/metric-privacy/n=48` shows **zero**
   collapsed rounds and **zero** invalid-distance rounds — a fully numerically stable run by both
   of the metrics this sweep tracks — and still ends up 18.9pp behind global-dp. Whatever is
   degrading that specific combination is not the zero-norm collapse or the non-finite-distance
   fallback; it is unexplained by the diagnostics currently in place. Conversely,
   `non-iid/metric-privacy/n=48` spent 239 of 240 rounds on the non-finite-distance fallback
   (essentially the entire run past round 1) yet still slightly *outperformed* global-dp — the
   fallback (reuse the last valid distance) can apparently keep training usable even when the
   underlying pairwise-distance signal is broken almost throughout, which is itself worth noting as
   a partial validation of that fallback's design.

4. **The post-hoc detailed-evaluation artifacts are not usable for this sweep and were not used in
   this report.** Every one of the 8 n=8/n=48 combinations shows `evaluation.json`'s postprocessed
   `server_final_test.accuracy` diverging from the training-time recorded accuracy that produced it
   (`validated_against_run_json.accuracies_match: false` in every case) — 6 of the 8 land on exactly
   0.134375 (172/1280, "always predict class 0" on the fixed-composition server test set) regardless
   of the actual trained model. This looked at first like the per-round dataset-reload race fixed in
   `290a5b0`/`d6b9a06`, but that fix doesn't obviously apply here: `evaluate_state_dict` reuses the
   same already-loaded `data_module` instance passed in from `server.py`, so it shouldn't be
   re-triggering a fresh, racy dataset load at all. The root cause of this specific divergence is
   unresolved (see "Known gaps" below). Every number in this report's tables comes from
   `server_evaluate_metrics` (the training-time centralized evaluation), never from
   `evaluation.json`.

## Known gaps / caveats

- **Single seed (42) for every run.** No variance estimates; a single bad/lucky seed could shift any
  of the deltas above, especially the ones close to zero (e.g. non-iid's +3.12pp at n=48).
- **Reduced matrix.** `vanilla`, `fedyogi`, `fedavgm`, `fedprox`, `fedmedian`, and `n=16` are not
  covered by this sweep — see "Overview" above. The homogeneous/n=48 reversal and the eval-artifact
  bug (Finding 4) are both strong candidates for follow-up at `n=16` to see whether they emerge
  gradually or specifically at 48 clients.
- **`evaluation.json`/`predictions.npz` are not trustworthy for `results/scale_controlled`.** Do not
  use them for per-class/confusion-matrix/macro-averaged detail on this result set without first
  resolving Finding 4 — they were deliberately not used for anything in this report beyond
  documenting that they disagree with the training-time numbers.
- **Global-dp's own scaling behavior is out of scope here.** Its accuracy also drops substantially
  and non-monotonically across n (49.5% → 14–25% → 17–33%); this report only uses it as the
  comparison baseline for metric-privacy and does not attempt to explain it. `Finding 1` in
  `reports/client_count_scaling.md` covers FedAvg's own scaling behavior under the fixed-round
  design.
- **The 4-client rows are this sweep's own base case, not directly comparable to other reports.**
  They use `nm=0.05` (not the paper's `nm=0.01`) and `fedavg` only, run specifically as the anchor
  for the `rounds(n)` proportional-scaling formula, not as a paper-reproduction data point.
