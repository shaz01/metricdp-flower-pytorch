# Phase 1 Diagnosis: Constant-Compute Client-Count Scaling (v1 and v2)

Source data: `results/scale_controlled/` (v1), `results/scale_controlled_epochs/` (v2).
Code: `experiments/client_scaling/sweep_scale_controlled.py` (v1),
`experiments/client_scaling/sweep_scale_controlled_epochs.py` (v2),
`metricdp_pytorch/metricdp_strategy.py`, `experiments/reproduce/`.

> **Status.** Two constant-compute designs exist: v1 (round-scaled, below) and v2 (rounds-fixed,
> epoch-scaled — the corrected design, added after a methodology flaw was found in v1). Both have
> completed their full 12-combination matrix. **Neither design's results are currently trustworthy
> at face value.** Comparing v1's and v2's *identical* n=4 baseline configuration (same seed, same
> hyperparameters) directly proved this machine's MPS backend is non-deterministic across process
> launches under concurrent multi-actor training — accuracy deltas up to 53.28pp on a run that
> should be bit-for-bit reproducible (`metricdp_pytorch/utils/device.py`'s `resolve_device`
> docstring has the full investigation; see v2's Findings below for the exact numbers). v1's own
> two headline n=48 findings were separately noise-checked by repeated-rep spot checks and both
> failed to hold up (`results/noise_floor_check/`, `results/noise_floor_check_noniid/`). v2 has not
> been checked the same way. **This report documents what each design found, corrected in place
> where a claim didn't hold up — it does not yet answer this phase's central question** (is the
> metric-privacy scaling failure genuine, or a round-budget confound). See "Overall status" at the
> end.
>
> v1's original Finding 1 was wrong and has been corrected in place (see that section): the
> `rounds(n) = 5n` design scales aggregation/noise-injection frequency 12x from n=4 to n=48 while
> *also* shrinking per-client shard size 12x, at fixed `--local-epochs`/`--batch-size` — meaning
> n=48 clients get only 2–4 mini-batches per local epoch, interrupted by aggregation 12x more often
> than n=4 clients, despite the total gradient-step count being roughly preserved. That confound,
> not "more accumulated DP noise" as v1 originally claimed, is the best-supported explanation for
> why its 240-round accuracy came in so far below the fixed-20-round baseline. v2 (below) is the
> redesign that holds rounds fixed and scales `--local-epochs` instead, targeting
> aggregation-frequency parity directly rather than only total-step-count parity.

## v1 Overview

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

## v1 Results: constant-compute matrix (nm=0.05, fedavg)

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

## v1 Comparison against the fixed-20-round n=48 baseline

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

## v1 Findings

1. **[Corrected] Scaling rounds to compensate for less per-client data made accuracy worse, not
   better, for both mechanisms — but not because of accumulated DP noise magnitude, which was
   verified to *not* grow the way originally claimed here.** Accuracy at 48 clients dropped
   substantially versus the fixed-20-round baseline for every combination — global-dp fell 62%→33%
   (homogeneous) and 62%→17% (non-iid); metric-privacy fell 63%→15% (homogeneous) and 64%→20%
   (non-iid). This report originally attributed that to "240 rounds injects ~12x the noise of 20,"
   which is wrong: Flower's `compute_stdv` scales per-round sigma as
   `noise_multiplier * clipping_norm / num_clients`, and under this sweep's exact
   `rounds(n) = 5n` scaling, `rounds(n) * sigma(n) = 5n * (0.25/n) = 1.25` — an exact constant,
   verified directly from the formula, independent of n. Metric-privacy's calibrated noise
   (`noise_multiplier / distance`) isn't 1/n-scaled the same way, but its logged
   `metric-dp-noise-stdv` trajectory is *not* monotonically worse at higher n either: total summed
   sigma across all rounds is higher at n=8 than n=48 for both partitions (homogeneous: 2.06 vs
   2.87 — comparable; non-iid: 1.60 vs 0.99 — n=48 is *lower*), and `homogeneous/n=48` — the combo
   with the single worst accuracy of the whole sweep — has the *most* stable, slowest-growing noise
   trajectory of the three metric-privacy n values (σ moves from 0.0048 to only 0.0132 over 240
   rounds). Accumulated noise magnitude does not explain the accuracy drop.

   What does line up with the drop: per-client training-set size at n=48 collapses to 86 samples
   (homogeneous) or 34–137 (non-iid), which at `batch_size=32` means 2–4 mini-batches per local
   epoch (some non-iid clients get 1). The `rounds(n)=5n` formula does roughly preserve *total*
   gradient-step count across the whole run (~3200 at n=4 vs. ~3600 at n=48, homogeneous), but it
   delivers that count as 240 rounds of 2–4 tiny batches each — interrupted by aggregation and
   noise-injection 240 times — instead of 20 rounds of 32 substantial batches each, interrupted 20
   times. Total step count parity does not imply comparable training dynamics when the same step
   count is fragmented into 12x more, much shorter local phases, each drawn from a much smaller and
   (for non-iid) much noisier local sample. This reframing is why the redesigned control (see the
   note at the top of this report) holds *rounds* fixed and scales `--local-epochs` instead of
   scaling rounds — it targets aggregation-frequency/batch-count parity directly, rather than only
   total-step-count parity.

2. **[Corrected] The metric-privacy advantage over global-dp shrinks monotonically from n=4 to n=8
   in both partitions; the homogeneous/n=48 reversal below is directionally weak evidence, not a
   reliable magnitude.** Homogeneous: +21.41pp (n=4) → +5.62pp (n=8) → **−18.91pp** (n=48). Non-iid:
   +35.94pp (n=4) → +7.34pp (n=8) → +3.12pp (n=48). The −18.91pp homogeneous/n=48 figure was later
   checked against MPS's own run-to-run noise (`results/noise_floor_check/`, 2 extra reps each of
   both privacy modes at identical settings, given `metricdp_pytorch/utils/device.py`'s confirmed
   MPS non-determinism — see `resolve_device`'s docstring). Both accuracies swing enormously on
   their own: global-dp ranged 33.44–63.12% and metric-privacy 14.53–39.38% across just 3 runs each.
   Paired within the same run generation, the delta stayed negative in all 3 cases (−18.91, −34.22,
   −17.97pp) — some directional consistency — but crossing generations (e.g. comparing one run's
   global-dp against a *different* run's metric-privacy, which is what a single-seed comparison risks
   doing) it ranges from −48.59pp to **+5.94pp**, including a sign flip. Treat "metric-privacy
   underperforms at homogeneous/n=48" as weakly supported direction, not an established effect, and
   do not cite −18.91pp as a precise magnitude.

   The non-iid n=48 result was noise-checked the same way (`results/noise_floor_check_noniid/`,
   2026-08-04) and fails for a different reason: individual accuracies are far more stable there
   (global-dp 16.41–17.97%, metric-privacy 14.69–21.41% — a few pp of spread, not tens) but the
   +3.12pp delta itself is small enough to be within that smaller noise anyway. Paired deltas were
   +3.12, +5.00, and **−3.28pp** — the sign flips on the third pair. Small effect, comparable-sized
   noise: same "not established" verdict as homogeneous, via the opposite mechanism (there, a large
   effect was swamped by even larger noise; here, a small effect is swamped by smaller but
   still-comparable noise).

3. **Two distinct metric-privacy failure modes are visible in the round-level data, and neither
   fully explains the homogeneous/n=48 reversal** — though per Finding 2's correction, that
   reversal's magnitude is itself not established, so "explaining" its exact size isn't a
   well-posed question yet; the finding below stands as "these two known failure modes weren't
   active in this specific run," not as an open mystery requiring a specific-sized effect to
   explain. At n=8, both partitions hit frequent
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

## v1 Known gaps / caveats

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

---

## v2 Overview

v1's Finding 1 (above) identified a confound: scaling *rounds* at fixed local epochs and batch
size also scales *aggregation frequency* 12x from n=4 to n=48, while shrinking each client's
per-round shard 12x — at 48 clients, clients see only 2–4 mini-batches per local epoch and are
interrupted by aggregation and noise-injection 240 times instead of 20. Total gradient-step count
is preserved, but the training dynamics are not comparable: the same step count delivered as many
short, frequently-interrupted local phases is not equivalent to delivering it as few, substantial
local phases.

v2 corrects this by holding rounds fixed at the paper's value and scaling *local epochs* instead:

```
rounds(n) = 20 (constant)
local_epochs(n) = round(BASE_LOCAL_EPOCHS * n / BASE_NUM_CLIENTS), BASE_LOCAL_EPOCHS=5, BASE_NUM_CLIENTS=4
```

so every combination aggregates exactly 20 times regardless of client count, targeting
aggregation-frequency parity directly rather than only total-step-count parity. Same reduced
12-combination matrix as v1 (`global-dp`/`metric-privacy` × `fedavg` × `homogeneous`/`non-iid` ×
`n ∈ {4, 8, 48}`, `noise_multiplier=0.05`, seed 42), run via
`experiments/client_scaling/sweep_scale_controlled_epochs.py`, results in
`results/scale_controlled_epochs/`.

## v2 Results: constant-compute matrix (nm=0.05, fedavg)

Same column definitions as the v1 table above (training-time recorded values from the last
completed round; invalid/collapsed counts from `train_metrics`' per-round flags; distance
min/mean/max pools every finite per-round `metric-dp-distance`/`metric-dp-distance-mean` value).

| Partition | Privacy | n | Local epochs | Accuracy | Loss | F1 | Precision | AUC | Invalid dist. rounds | Collapsed rounds | Distance min/mean/max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| homogeneous | global-dp | 4 | 5 | 35.78% | 0.3699 | 0.1886 | 0.1280 | 0.8252 | n/a | n/a | n/a |
| homogeneous | global-dp | 8 | 10 | 39.38% | 0.5683 | 0.2225 | 0.1550 | 0.7931 | n/a | n/a | n/a |
| homogeneous | global-dp | 48 | 60 | 35.47% | 0.6191 | 0.2787 | 0.2318 | 0.7955 | n/a | n/a | n/a |
| homogeneous | metric-privacy | 4 | 5 | 30.31% | 0.5530 | 0.1445 | 0.1072 | 0.7539 | 0 | 0 | 1.0874 / 1.4350 / 1.7755 |
| homogeneous | metric-privacy | 8 | 10 | 24.22% | 35.8899 | 0.0944 | 0.0587 | 0.5722 | 0 | 5 | 0.0635 / 1.0526 / 1.7843 |
| homogeneous | metric-privacy | 48 | 60 | 34.53% | 0.6054 | 0.2576 | 0.2155 | 0.7898 | 0 | 0 | 1.1169 / 1.5433 / 2.1823 |
| non-iid | global-dp | 4 | 5 | 31.87% | 0.5600 | 0.1541 | 0.1016 | 0.7579 | n/a | n/a | n/a |
| non-iid | global-dp | 8 | 10 | 38.44% | 0.5443 | 0.2134 | 0.1477 | 0.7855 | n/a | n/a | n/a |
| non-iid | global-dp | 48 | 60 | 33.44% | 0.6816 | 0.1676 | 0.1118 | 0.7403 | n/a | n/a | n/a |
| non-iid | metric-privacy | 4 | 5 | 32.19% | 0.5797 | 0.2073 | 0.1530 | 0.7738 | 0 | 0 | 1.2980 / 1.7332 / 2.2858 |
| non-iid | metric-privacy | 8 | 10 | 24.84% | 41.6213 | 0.1658 | 0.1265 | 0.5566 | 0 | 7 | 0.0321 / 1.0894 / 1.9455 |
| non-iid | metric-privacy | 48 | 60 | 37.81% | 0.5829 | 0.2075 | 0.1430 | 0.7838 | 19 | 0 | 1.9969 / 1.9969 / 1.9969 |

Wall-clock for the final successful attempt of each combo: n=4 combos 206.8–215.0s; n=8 combos
390.0–398.0s; n=48 combos 2246.1–2361.7s. Full detail in
`results/scale_controlled_epochs/sweep_progress.log`.

## v2 Comparison against baselines

**Against the fixed-20-round n=48 baseline** (`results/48client_scaling`, same as v1's comparison):

| Partition | Privacy | Fixed 20 rounds (accuracy) | v2 constant-compute (accuracy) |
|---|---|---:|---:|
| homogeneous | global-dp | 62.03% | 35.47% |
| homogeneous | metric-privacy | 62.66% | 34.53% |
| non-iid | global-dp | 61.56% | 33.44% |
| non-iid | metric-privacy | 63.75% | 37.81% |

**Metric-privacy − global-dp deltas, v1 vs. v2** (different designs, not a reproducibility check —
v1 uses `rounds(n)=5n`/fixed `local_epochs=5`; v2 uses fixed `rounds=20`/`local_epochs(n)`):

| Partition | v1 Δ at n=4 | v2 Δ at n=4 | v1 Δ at n=8 | v2 Δ at n=8 | v1 Δ at n=48 | v2 Δ at n=48 |
|---|---:|---:|---:|---:|---:|---:|
| homogeneous | +21.41pp | −5.47pp | +5.62pp | −15.16pp | −18.91pp | −0.94pp |
| non-iid | +35.94pp | +0.32pp | +7.34pp | −13.60pp | +3.12pp | +4.37pp |

## v2 Findings

1. **The single most important result of v2 is not an accuracy pattern — it's direct proof that
   this machine's MPS backend cannot currently be trusted for any of these numbers.** v2's `n=4`
   baseline uses an *identical* configuration to v1's `n=4` baseline by construction
   (`round(5·4/4) = 5`): same seed (42), same `rounds=20`, same `local_epochs=5`, same
   `noise_multiplier=0.05`, same everything — metadata for both runs was diffed directly and
   confirmed identical. These should be bit-for-bit reproducible. They are not:

   | Combination | v1 accuracy | v2 accuracy | Δ |
   |---|---:|---:|---:|
   | homogeneous / global-dp | 49.53% | 35.78% | −13.75pp |
   | homogeneous / metric-privacy | 70.94% | 30.31% | −40.62pp |
   | non-iid / global-dp | 49.53% | 31.87% | −17.66pp |
   | non-iid / metric-privacy | 85.47% | 32.19% | −53.28pp |

   This directly confirmed the root cause: PyTorch's MPS backward pass is non-deterministic across
   process launches specifically when multiple Ray client actors train concurrently on the shared
   device, even with every seed fixed (full investigation in
   `metricdp_pytorch/utils/device.py`'s `resolve_device` docstring). An opt-in
   `METRICDP_FORCE_CPU=1` override now exists for reproducibility-critical runs, at roughly 3x the
   wall-clock cost. **This finding supersedes everything below it** — points 2–4 describe what v2's
   numbers show, not what should be believed about the underlying mechanism.

2. **v2's own metric-privacy-vs-global-dp deltas do not resemble v1's pattern, even qualitatively —
   which is itself further evidence neither should be trusted at face value.** v1 showed a
   monotonic shrink from a large positive advantage at n=4 down to a reversal at n=48
   (homogeneous: +21.41 → +5.62 → −18.91pp; non-iid: +35.94 → +7.34 → +3.12pp). v2 shows no such
   pattern: homogeneous is negative or near-zero at every n (−5.47 / −15.16 / −0.94pp), and non-iid
   is small and non-monotonic (+0.32 / −13.60 / +4.37pp). If both designs were measuring the same
   underlying signal reliably, their qualitative shapes should agree even if magnitudes differ
   between a round-scaled and an epoch-scaled control. They don't — consistent with, not
   independent evidence beyond, Finding 1's proof that the platform itself is the problem.

3. **The same n=8 fragility seen in v1 recurs in v2.** Both partitions' metric-privacy/n=8 runs hit
   frequent collapsed-update rounds (5/20 homogeneous, 7/20 non-iid) alongside catastrophic loss
   blowup (35.89, 41.62) — comparable in kind to v1's n=8 collapse pattern (14/40, 23/40 rounds).
   Separately, `non-iid/metric-privacy/n=48` spent 19 of 20 rounds on the non-finite-distance
   fallback — proportionally similar to v1's same combination (239/240 rounds). This specific
   pattern — non-iid/metric-privacy struggling with distance validity at n=48 — now shows up
   under *two* different constant-compute designs, which makes it a more interesting candidate for
   follow-up than a single-design observation would be. It is not yet a verified finding: the
   underlying accuracy numbers for both designs are subject to Finding 1's caveat, and round-level
   collapse/invalid counts are diagnostics, not accuracy — they have not been separately
   noise-checked either.

4. **The same evaluation-artifact bug from v1's Finding 4 is present in v2, unchanged.** All 12 of
   12 `results/scale_controlled_epochs/*.evaluation.json` files show
   `validated_against_run_json.accuracies_match: false`; several land on exactly 0.134375 (the
   "always predict class 0" degenerate value seen in v1). This confirms the bug generalizes across
   both constant-compute designs rather than being specific to v1's round-scaling, but its root
   cause remains unresolved (see v1's Finding 4 discussion). As with v1, no number in this section
   uses `evaluation.json`; everything above is from `server_evaluate_metrics`.

## v2 Known gaps / caveats

- **Same structural caveats as v1**: single seed (42), reduced matrix (`vanilla`/`fedyogi`/
  `fedavgm`/`fedprox`/`fedmedian`/`n=16` not covered), `evaluation.json`/`predictions.npz` not
  trustworthy for this result set (Finding 4), global-dp's own scaling behavior out of scope.
- **v2's `n=8`/`n=48` deltas have not been individually noise-floor-checked.** Only the `n=4` point
  has direct proof of unreliability (Finding 1, an exact reproducibility comparison against v1).
  The `n=8` and `n=48` rows have not been re-run under repeated MPS reps or
  `METRICDP_FORCE_CPU=1` the way v1's two n=48 headline findings were
  (`results/noise_floor_check/`, `results/noise_floor_check_noniid/`). Given Finding 1's proof
  that the platform itself diverges by tens of pp on an identical config, there is no reason to
  expect v2's n=8/n=48 rows are any more reliable — this is stated as a caveat, not confirmed by a
  dedicated check on those specific rows.

## Overall status

Both constant-compute designs — the thing Phase 1's first roadmap item asked for — are built and
have each completed a full sweep. That is not the same as having answered Phase 1's actual
question (is the metric-privacy scaling failure at 48 clients genuine, or an artifact of the fixed
round budget every earlier sweep used). It hasn't been answered yet: v1's two headline findings
failed direct noise checks, v2 was never separately checked and its own numbers disagree with
v1's even in shape, and Finding 1 above proves the measurement platform itself has a noise floor
comparable to the effect sizes in question. Getting a trustworthy answer requires either rerunning
one of these designs (v2, being the corrected one, is the natural candidate) under
`METRICDP_FORCE_CPU=1`, or exhaustively noise-checking v2's remaining findings the way v1's were —
neither has been started. See `reports/progress_report_phase1.tex`/`.pdf` for the fuller writeup of
this tradeoff, and `STATUS.md` for the current decision status.
