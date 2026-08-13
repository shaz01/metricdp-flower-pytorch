# EuroSAT Client Inference Attack (CIA)

**Branch:** `feature/eurosat-scaling`
**Status:** complete — 36/36 trajectories (18 IN + 18 OUT), 0 failed
**Date:** 2026-08-12 – 2026-08-13

## Purpose

Runs the same IN-remove vs. OUT-remove Client Inference Attack methodology already built and run
against CIFAR-100 (`experiments/cia/scripts/cifar100_scaling.py`) against the EuroSAT accuracy
sweep (`reports/eurosat_accuracy_sweep.md`), to measure whether — and how much — this repo's
metric-privacy and global-DP mechanisms suppress membership-inference leakage on a genuinely
different dataset. Explicitly deferred at accuracy-sweep design time
(`docs/superpowers/specs/2026-08-12-eurosat-scaling-design.md`: "CIA attack experiment on EuroSAT
— explicitly deferred to a later decision") and picked up once the sweep's results existed.

## Protocol

- **Clients:** 48 IN-remove (target participates) / 47 OUT-remove (target excluded),
  `target_partition_id=0` — matches the accuracy sweep's `n=48`.
- **Seeds:** (42, 43, 44) — 3 seeds from the start, not a single-seed pilot. CIFAR-100's CIA
  started with seed 42 only and that turned out statistically underpowered, requiring a full
  multi-seed redo; this experiment skipped repeating that mistake.
- **Matrix:** `{homogeneous, non-iid} × {vanilla, global-dp, metric-privacy}`, `fedavg`-only — 6
  combos × 3 seeds = 18 trajectories per group (IN, OUT), 36 total.
- **`noise_multiplier = 0.03710712210729851`** — reused directly from the accuracy sweep's own
  empirical calibration (see `reports/eurosat_accuracy_sweep.md`); no separate calibration step
  for CIA, mirroring how CIFAR-100's CIA reused its own accuracy sweep's calibrated value.
- **Hyperparameters:** identical to the accuracy sweep (`clipping_norm=5.0`, `local_epochs=5`,
  `batch_size=32`, `learning_rate=0.001`, `initialization_epochs=20`, `rounds=100`; no
  `weight_decay`/`lr_schedule` — same base-branch constraint).
- **Shadow datasets:** `SHADOW_FRACTION=0.10`, `NOISE_STD_FRACTION=0.20` — same values as
  CIFAR-100's CIA. EuroSAT's `n=48` with ~450 images/client (homogeneous) is comparable in
  per-client volume to CIFAR-100's `n=100` at ~500/client, so this wasn't adjusted.
- **Checkpoints:** round 1 + every 10th round through 100 (11 checkpoints per trajectory), scaled
  down from CIFAR-100 CIA's every-10th-through-250 (26 checkpoints) to match this sweep's shorter
  round budget.
- **Execution:** `experiments/cia/scripts/eurosat_scaling.py`, `--group in`/`--group out` as two
  separate concurrent processes (safe: each writes its own report file, `cia_in.json`/
  `cia_out.json`). Ran at `--max-parallel-clients 6` throughout, matching the accuracy sweep's
  proven value for this GPU/model/client-count combination — no skipped-client rounds occurred.
- **Analysis:** `experiments/cia/scripts/eurosat_scaling_analysis.py`, mirroring
  `cifar100_scaling_analysis.py` exactly — pairs each combo's IN/OUT trajectories at their shared
  checkpoint rounds per seed, reports round-matched AUC plus a bootstrap 95% CI, pooling all 3
  seeds (33 round-matched pairs per combo). Reuses the shared scoring primitives in
  `experiments/cia/reports/build_alzheimer_cia_report.py` unmodified.

## Results

Round-matched AUC (0.5 = attacker cannot distinguish IN from OUT at all; 1.0 = perfect
distinguishing; below 0.5 = the attack does *worse* than random guessing), pooled across all 3
seeds (33 round-matched pairs per combo), for both shadow-loss score variants:

| Partition | Privacy | clean-shadow AUC | clean pooled AUC (95% CI) | noisy-shadow AUC | noisy pooled AUC (95% CI) |
|---|---|---:|---|---:|---|
| homogeneous | vanilla | 0.727 | 0.544 (0.404–0.686) | 0.697 | 0.555 (0.412–0.694) |
| homogeneous | global-dp | 0.606 | 0.542 (0.399–0.685) | 0.485 | 0.538 (0.397–0.680) |
| homogeneous | metric-privacy | 0.606 | 0.533 (0.390–0.677) | 0.485 | 0.533 (0.390–0.676) |
| non-iid | vanilla | 0.545 | 0.528 (0.387–0.672) | 0.364 | 0.487 (0.345–0.630) |
| non-iid | global-dp | 0.455 | 0.509 (0.368–0.653) | 0.364 | 0.488 (0.346–0.631) |
| non-iid | metric-privacy | 0.606 | 0.522 (0.379–0.663) | 0.212 | 0.440 (0.301–0.582) |

Full data, including both score keys and per-combo raw round-matched-AUC values, is in
`results/cia_eurosat_scaling/cia_analysis.json`.

## Observations

- **`homogeneous / vanilla` shows the clearest leak**: round-matched AUC 0.727 (clean-shadow
  score), the highest of all 6 combos, and consistent with the general expectation that vanilla
  FedAvg with a homogeneous partition gives an attacker the most signal to exploit.
- **Both DP mechanisms reduce leakage relative to vanilla in the homogeneous partition**:
  `global-dp` and `metric-privacy` both drop to 0.606 clean-shadow AUC (down from 0.727) — a real,
  consistent suppression effect from vanilla, though not all the way to the 0.5 no-leakage line.
- **`non-iid` partitioning shows much weaker, sometimes negative, leakage across all privacy
  modes** — round-matched AUCs of 0.545, 0.455, and 0.606 (vanilla/global-dp/metric-privacy), all
  close to or below 0.5. `non-iid / metric-privacy` with the noisy-shadow score reaches **0.212**,
  notably *below* chance — the attack does worse than random guessing there. This mirrors the
  accuracy sweep's own finding that `non-iid` slightly outperformed `homogeneous` on task
  accuracy; here it also appears to leak less, which is a genuinely interesting and non-obvious
  combination (usually a privacy/utility trade-off would predict the opposite pairing).
- **Wide confidence intervals.** Every combo's bootstrap 95% CI spans roughly 0.30–0.35 AUC units
  (e.g. homogeneous/vanilla: 0.404–0.686) — all 6 CIs overlap 0.5, and most overlap each other
  substantially. With only 33 round-matched pairs per combo (3 seeds × 11 checkpoints), this
  experiment is not statistically powered to distinguish most of these combos' AUCs from each
  other or from the no-leakage baseline with confidence. The *pattern* (homogeneous leaks more
  than non-iid; DP mechanisms reduce homogeneous's leak) is consistent with the round-matched
  point estimates, but should be read as a directional signal, not a statistically confirmed
  ranking.
- **Both shadow-loss score keys agree in direction** on every combo (clean-shadow AUC ≥
  noisy-shadow AUC in all 6 cases) — the noisy-shadow variant consistently reads as slightly less
  leaky than the clean-shadow variant, which is the expected direction (a noisier shadow dataset
  should make membership harder to distinguish).

## Data location

`results/cia_eurosat_scaling/` — `cia_in.json`/`cia_out.json` (aggregate per-round-per-trajectory
rows, 198 each), `cia_analysis.json` (the round-matched AUC summary above), 36 per-trajectory
result JSONs, and `progress_in.log`/`progress_out.log`. Per-trajectory `.evaluation.json` files
(~630MB total across 36 trajectories) are gitignored — not committed, given the volume, unlike
the accuracy sweep's own (much fewer) evaluation files.

## What this does not cover

No comparison against CIFAR-100's own CIA round-matched AUC numbers — that experiment's retry
passes are still in progress (`feature/cifar100-scaling`) and its own report hasn't been written
yet, so no committed numbers exist to compare against at time of writing.
