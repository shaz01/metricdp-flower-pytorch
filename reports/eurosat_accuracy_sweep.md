# EuroSAT Accuracy Sweep

**Branch:** `feature/eurosat-scaling`
**Status:** complete — 6/6 combinations, 0 failed
**Date:** 2026-08-12 – 2026-08-13

## Purpose

A comparison point for the paper-reproduction methodology on a dataset genuinely different from
the paper's own (Alzheimer MRI) and from this repo's other extensions (CIFAR-10, CIFAR-100,
Fashion-MNIST): EuroSAT is 10-class satellite land-use imagery, not digit/object recognition.
CIFAR-10 was explicitly out of scope (a teammate's part of the project), and the MNIST family was
ruled out for being too easy/well-covered already. Design doc:
`docs/superpowers/specs/2026-08-12-eurosat-scaling-design.md` (gitignored, not in this repo).

## Protocol

- **Dataset:** `tanganke/eurosat` (Hugging Face) — 21,600 train / 2,700 test images, RGB 64×64,
  10 classes, all used (no subsetting). `experiments/reproduce/dataset/eurosat.py`.
- **Model:** a 3-block Conv–GroupNorm–ReLU CNN (channels 32→64→128), 289,194 parameters, 0
  buffers — deliberately lighter than the CIFAR-100 extension's model, since EuroSAT is an
  easier 10-way task on similarly-sized images. `experiments/reproduce/eurosat_cnn.py`.
- **Clients:** n=48 (not CIFAR-100's n=100) — chosen so per-client data volume (~450 images at
  homogeneous partitioning) is comparable to CIFAR-100's ~500/client at n=100, rather than
  starving it further given EuroSAT's smaller total dataset.
- **Matrix:** `{homogeneous, non-iid} × {vanilla, global-dp, metric-privacy} × fedavg` — 6
  combinations. `fedavg`-only (not the full 6-aggregator matrix), matching the CIFAR-100
  extension's own scope.
- **Hyperparameters:** `clipping_norm=5.0`, `local_epochs=5`, `batch_size=32`,
  `learning_rate=0.001`, `initialization_epochs=20` (inert for `fedavg`; only
  `fedavgm`/`fedopt`/`fedyogi` use validation-set initialization — confirmed in the committed run
  metadata: `initialization_pretrained: false`), `rounds=100`, `seed=42`.
- **`noise_multiplier = 0.03710712210729851`** — empirically calibrated, not guessed. Measured a
  real `dp-signal-update-norm` of 2.0786 at round 3 of a diagnostic `global-dp` run (n=48,
  homogeneous), then solved `noise_multiplier = signal_norm × num_sampled_clients / (clipping_norm
  × √param_count)`. A 20-round verification run confirmed a round-3 noise-to-signal ratio of
  ~0.998 (on target) and healthy convergence (loss 2.275→0.703, accuracy 19.5%→74.4%); the ratio
  drifts to ~2.63 by round 20 as the signal naturally shrinks with convergence — an expected
  limitation of single-early-round calibration, not a bug.
- **No `weight_decay`/`lr_schedule` fields** — this branch is based on `master`, whose
  `Hyperparams` dataclass has neither (`feature/cifar100-scaling` added them independently and
  this branch deliberately doesn't duplicate that). Training runs at PyTorch `Adam`'s own default
  `weight_decay=0.0`.
- **Execution:** `experiments/eurosat_scaling/sweep_eurosat_scaling.py`, resumable/skip-if-complete,
  continues past failures. First launch at `--max-parallel-clients 16` hit repeated skipped-client
  rounds under shared-GPU contention (a real correctness concern — non-48/48 aggregation changes
  what the round actually trains on); restarted clean at `--max-parallel-clients 6`, which
  eliminated skipped-client rounds entirely for the rest of the sweep.

## Results

All 6 combinations converged cleanly; none diverged.

| Partition | Privacy | Final accuracy | Final loss | F1 | Precision | AUC |
|---|---|---:|---:|---:|---:|---:|
| homogeneous | vanilla | 89.9% | 0.323 | 0.897 | 0.899 | 0.993 |
| homogeneous | global-dp | 88.7% | 0.341 | 0.886 | 0.887 | 0.993 |
| homogeneous | metric-privacy | 87.5% | 0.381 | 0.874 | 0.876 | 0.991 |
| non-iid | vanilla | 90.7% | 0.272 | 0.906 | 0.907 | 0.995 |
| non-iid | global-dp | 90.3% | 0.294 | 0.901 | 0.903 | 0.995 |
| non-iid | metric-privacy | 88.5% | 0.350 | 0.884 | 0.886 | 0.992 |

**Convergence trajectory** (accuracy at round 0 / 10 / 50 / 100), all six combos from the same
random-baseline starting point:

| Combo | r0 | r10 | r50 | r100 |
|---|---:|---:|---:|---:|
| homogeneous / vanilla | 11.9% | 65.3% | 84.8% | 89.9% |
| homogeneous / global-dp | 11.9% | 64.2% | 84.2% | 88.7% |
| homogeneous / metric-privacy | 11.9% | 66.1% | 82.1% | 87.5% |
| non-iid / vanilla | 11.9% | 68.7% | 86.5% | 90.7% |
| non-iid / global-dp | 11.9% | 69.0% | 85.3% | 90.3% |
| non-iid / metric-privacy | 11.9% | 68.7% | 83.0% | 88.5% |

## Observations

- **DP cost is small.** Relative to `vanilla`, `global-dp` costs 1.2 points of accuracy
  (homogeneous) / 0.4 points (non-iid); `metric-privacy` costs 2.4 points (homogeneous) / 2.2
  points (non-iid). Both mechanisms stay within ~2.5 points of the vanilla baseline in every
  partition mode.
- **`non-iid` partitioning outperformed `homogeneous`** across all three privacy modes (by 0.8–1.8
  points), which is not the usual expectation for federated learning — quantity-skewed non-iid
  partitioning is generally expected to hurt, not help, convergence. This is a real, consistent
  pattern across all three privacy modes on this dataset/model/client-count combination, not a
  single anomalous run — worth flagging as a finding rather than dismissing as noise, though a
  single seed per combo means it isn't statistically confirmed here.
- **No divergence, no accuracy collapse, no early-freeze behavior** in any of the 6 combos —
  unlike CIFAR-100's model history (a documented clipping-norm freeze and a
  vanilla/homogeneous correlated-update freeze), nothing analogous appeared here.

## Data location

`results/eurosat_scaling/` — one `{run_name}.json` (round-by-round metric history) and
`{run_name}.evaluation.json` (detailed per-class evaluation) per combo, plus `sweep_progress.log`.
An earlier 3-round smoke-test combo (`...n48__r3.*`) is also present, kept for provenance; it is
not part of the real 100-round sweep and should not be read as a result.

## What this does not cover

No Client Inference Attack was run as part of this sweep — see `reports/eurosat_cia.md` for that,
a separate follow-on experiment built once this sweep's results existed (it reuses this sweep's
calibrated `noise_multiplier` directly, without a separate calibration step).
