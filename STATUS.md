# Project Status

**Branch:** `master`
**Last updated:** 2026-08-13, CUDA workstation (`feature/eurosat-scaling` and
`feature/cifar100-scaling` are both complete, merged into `master`, and deleted — see `git log`
for anything more recent

This file is a short, git-tracked pickup point for any Claude Code session — this machine or
another — starting work on this repo. It reflects the branch it's committed on; check out the
branch you're working on before trusting it. Treat it as a pointer, not the source of truth: for
full narrative detail see `reports/`, for raw run data see `results/`, for chronological detail
use `git log`. Update it whenever a branch merges into `master` or `master`-level state otherwise
changes — keep it short, don't turn it into a changelog. Exception: the Active work
section (including the Currently running table) updates more often, at "worth a commit"
granularity — see `AGENTS.md`'s "Working across machines" section.

## Active work

Nothing currently running. Both `feature/eurosat-scaling` (the EuroSAT accuracy sweep and its CIA
attack) and `feature/cifar100-scaling` (the CIFAR-100 accuracy sweep and its CIA attack) are
complete, merged into `master`, and deleted (locally and on `origin`) — see "What's established"
below and `reports/eurosat_accuracy_sweep.md`/`reports/eurosat_cia.md`/
`reports/cifar-100_and_eurosat_results.tex` for the full writeups.

**CIFAR-100 CIA multi-seed retry was dropped.** The seed-42 single-seed CIA run (6/6 combos, 0
failures) was complete and reportable. A follow-on rerun to add seeds 43/44 — for a properly
3-seed-pooled analysis matching the Alzheimer/CIFAR-10/Fashion-MNIST/EuroSAT CIA protocols —
launched 2026-08-10 20:08 but never finished: the large 100-client/4.6M-parameter model combos hit
persistent, severe GPU VRAM contention on this shared machine. Four separate retry attempts (each
recovering 1-3 of the failing combos, with diminishing returns, despite reducing
`--max-parallel-clients` 16→6→10 and fixing a real `dp_diagnostics.py` crash-on-client-error bug
along the way) never reached a complete 3-seed run. Decision: drop seeds 43/44 entirely and keep
only the complete seed-42 data. `results/cia_cifar100_scaling/cia_in.json`/`cia_out.json`/
`cia_analysis.json` and the 12 per-trajectory result JSONs now hold seed-42-only data for all 6
combos; the 22 seed-43/44 per-trajectory JSONs were deleted.
`reports/cifar-100_and_eurosat_results.tex`'s CIFAR-100 CIA table reflects this (a single
seed-42-only table, all 6 combos scored, replacing the earlier 3-seed-pooled table that had two
"in progress" `global-dp` rows).

**Model history** (this repo went through several CIFAR-100 architectures before settling):
v1-v3 was a plain 3-block CNN (v2 briefly added a 4th conv block, reverted after its natural
per-round update magnitude exceeded `clipping_norm=5.0` and froze every clipping privacy mode).
v4 replaced it with a DenseNet+SELU architecture (553,220 params, concatenative skip connections,
GroupNorm(8), SELU with LeCun-normal init and AlphaDropout), built and verified for robustness to
that clipping-related freeze — and a second, independent freeze mode was found and fixed in that
generation too (`vanilla`, no DP at all, froze at n=128/homogeneous from many highly-correlated
client updates reinforcing rather than averaging out). **Per project-owner direction, v4 was itself
replaced** with the current model (v5): an adaptation of the project supervisor's own `CNNCIFAR100`
reference architecture (3 blocks of 2x[Conv3x3-GroupNorm-ReLU], channels 128/256/512,
global-average-pooled classifier, 4,631,268 params, 0 buffers — see
`experiments/cifar100_scaling/sweep_cifar100_scaling.py`'s docstring for the full model-history
record and `experiments/reproduce/cifar100_cnn.py`'s docstring for the adaptation details). This
consolidation also removed the separate `feature/cifar100-scaling-supervisor` branch/worktree that
had briefly held this model in isolation (merged into this branch, then deleted) and cleared every
prior CIFAR-100 result (v1-v4 sweep data, the supervisor model's own earlier narrower grid) —
none were kept, since all described a model, grid, or directory layout no longer in use.

GroupNorm has no running-stats buffers — unlike BatchNorm — so this still needs no
`metricdp_pytorch/metricdp_strategy.py` changes, same as v1's "no normalization at all" workaround.
Training supports weight decay (`WEIGHT_DECAY = 5e-4`) and an opt-in cosine LR schedule; the
schedule was tried and dropped — a decaying LR eventually drops client updates below
`clipping_norm`, so clipping stops binding late in each run and the DP noise-to-signal ratio drifts
back up against `noise_multiplier`, fighting the whole point of tuning it — so the sweep uses a
fixed LR instead. `noise_multiplier=0.0182`, calibrated specifically for this model at n=100 (the
only client count this sweep runs — see the `NOISE_MULTIPLIER` comment in
`experiments/cifar100_scaling/sweep_cifar100_scaling.py` for the full derivation), confirmed via a
verification run: noise-to-signal ratio 1.001 at n=100.

Full sweep and CIA results (tables, protocol, discussion) are in "What's established" below and
`reports/cifar-100_and_eurosat_results.tex` — not repeated here to avoid drifting out of sync with
those. Raw data: sweep JSONs at `results/cifar100_scaling/*.json`; CIA raw/analysis JSONs at
`results/cia_cifar100_scaling/`. Both experiments' `.evaluation.json`/`.predictions.npz`
artifacts stay local-only, gitignored (see `.gitignore` comment) — they blow past GitHub's 100MB
push limit.

Separately, on `master`: nothing currently running. `feature/scale-controlled-redo` (Phase 1 items
1 and 2 of `docs/RESEARCH_ROADMAP.md`) merged into `master` 2026-08-06 and was deleted — see
"What's established" below for what it left behind. The natural next steps there, not yet started:
`fedyogi` at `n=48` (the redo's matrix only covers `n=4/8` for `fedyogi`), and Phase 1's remaining
item (NaN/failure-mode logging in `runner.py`, motivated directly by the zero-norm-update crashes
found during the redo). After that, Phase 2 (mechanism redesign) is the next major phase.

### Currently running

Update this table whenever a machine picks up new work: add a row, edit the Status column
in place (e.g. `running` -> `done`), and leave a finished row for one update cycle before removing
it, so machine-to-results provenance isn't lost; see `AGENTS.md`'s "Working across machines"
section.

| Command | What | Status |
| --- | --- | --- |
| _(none)_ | Nothing currently running. | — |

## What's established on `master`

- The metric-privacy mechanism reproduces the source paper at 4 clients — the effect is barely
  visible at the paper's `noise_multiplier=0.01` (`reports/paper_reproduction.md`).
- A genuine, previously unpublished effect exists at 8 clients: metric-privacy beats global-DP by
  +6.9pp (homogeneous) / +12.2pp (non-IID) at `noise_multiplier=0.05`
  (`reports/client_count_scaling.md`).
- `MetricPrivacyServerSideFixedClipping.aggregate_train` (`metricdp_pytorch/metricdp_strategy.py`)
  no longer aborts a whole run on a non-finite/non-positive client-model distance or a
  `ZeroDivisionError` from Flower's own clipping code on zero-norm updates — both fall back
  gracefully (last-valid distance, or skip-and-keep-previous-round respectively) so a run's full
  round-by-round history survives instead of being discarded on one bad round. Richer per-round
  diagnostics also landed: full pairwise client-model distance distribution, per-pair client IDs,
  min/median/mean/count, not just the single max used for calibration.
  `LoggedGlobalDPServerSideFixedClipping` (`metricdp_pytorch/globaldp_strategy.py`) has the same
  zero-norm-update guard as of the scale-controlled redo — it never had one before, and 12/96 runs
  in `results/noise_by_clients/` hit exactly this crash before the fix landed.
- **A genuine client-reply-ordering non-determinism was found and fixed**: Flower's own weighted
  aggregation summed replies in network-arrival order, not deterministically — floating-point
  addition isn't associative, so this compounded into real numeric drift over rounds. Fixed via
  `DeterministicReplyOrderMixin` (`metricdp_pytorch/strategy_factory.py`, covers every aggregation
  method, not just metric-privacy) and a matching fix in `metricdp_pytorch/metrics.py`.
- **Phase 1 items 1 and 2 of `docs/RESEARCH_ROADMAP.md` are done**, redone on CUDA hardware (moved
  off this project's original Mac MPS backend, whose own non-determinism made every earlier result
  on these questions untrustworthy — see `reports/archive/constant_compute_scaling_mps_v1v2.md`).
  - **Constant-compute client-count scaling** (`reports/constant_compute_scaling.md`/`.tex`):
    `fedavg` at `n=4/8/48` + `fedyogi` at `n=4/8` — 40/40 combinations, 0 failures, 0
    invalid-distance/collapsed-aggregation rounds anywhere. The metric-privacy-vs-global-dp
    advantage shrinks from `n=4` to `n=48` but converges toward parity (`fedavg`: -2.5pp to
    +0.6pp at `n=48`), not the large reversal the MPS-era attempt found (which never survived its
    own noise-floor check). `fedyogi` at `n=48` not yet run.
  - **Noise-multiplier x client-count sweep** (`reports/noise_by_clients.md`): `n ∈ {8,16,32,48}` x
    6 noise multipliers, `fedavg` only — 96/96, 0 failures. The noise ceiling genuinely shifts up
    with client count (a fixed `noise_multiplier` is relatively less noisy at higher `n`, since
    `compute_stdv` divides by `num_sampled_clients`) — `nm=0.1` collapses training at `n=8` but
    stays healthy through `n=48`. Separately, metric-privacy's own calibrated noise goes unstable
    right at each client count's collapse boundary and underperforms global-dp there by as much as
    -18pp, worse at higher `n` — a real, more precisely localized version of the original
    `results/48client_scaling` scaling concern, and a lead for Phase 2's mechanism redesign.
- `reports/first_round_cia.md` is stale — says "no result data yet," but `results/cia_client_scaling/`
  has real trained models and partial attack scores. Needs a rewrite, not done yet. The Flower-1.32
  port-equivalence check (`reports/port_equivalence.md`) still has no committed result data.
- **CIFAR-100 accuracy sweep + CIA** (`reports/cifar-100_and_eurosat_results.tex`): `n=100`
  clients, 250 rounds, `fedavg`, `noise_multiplier=0.0182`. Accuracy sweep: 6/6 combos, 0 failed,
  21.4–29.1% accuracy (100-class task, ~1% random-baseline) — metric-privacy roughly ties
  global-dp (within ~0.6pp either way), both DP modes ~7–8pp below vanilla, partition mode barely
  moves the numbers. CIA: seed-42-only (a multi-seed 43/44 rerun was attempted for proper 3-seed
  pooling but dropped after repeated GPU VRAM contention on this large 4.6M-parameter model
  prevented it from ever completing — see `git log` on the deleted `feature/cifar100-scaling` for
  the retry history), 26 round-matched pairs per combo, all 6/6 combos scored. Leakage is highest
  for `homogeneous/vanilla` (0.846) and lowest for `non-iid/vanilla` (0.500, the no-leakage line);
  `global-dp` shows more leakage than `metric-privacy` in every partition/shadow combination on
  this seed. Every 95% CI includes 0.5 (single-seed, underpowered), so read this as a directional
  pattern, not a statistically confirmed ranking.
- **EuroSAT accuracy sweep + CIA** (`reports/eurosat_accuracy_sweep.md`, `reports/eurosat_cia.md`):
  a comparison point on satellite land-use imagery (10-class, genuinely different domain from
  CIFAR-10/CIFAR-100/Fashion-MNIST/Alzheimer), `n=48`. Accuracy sweep: 6/6 combos, 0 failed,
  87.5–90.7% accuracy across all combos; `non-iid` partitioning slightly *outperformed*
  `homogeneous` in every privacy mode, and DP mechanisms cost only 0.4–2.4pp versus vanilla. CIA:
  36/36 trajectories (18 IN + 18 OUT), 0 failed, 3 seeds from the start (CIFAR-100's CIA needed a
  post-hoc multi-seed redo after an underpowered single-seed pilot — this one skipped that
  mistake). Round-matched AUC shows the clearest leak at `homogeneous/vanilla` (0.727), both DP
  mechanisms suppress it there (to 0.606), and `non-iid` leaks much less across every privacy mode
  (one combo's noisy-shadow AUC drops to 0.212, below chance) — but confidence intervals are wide
  (~0.30–0.35 AUC units, all overlapping 0.5), so read this as a directional pattern, not a
  statistically confirmed ranking. Along the way: found and fixed a real bug in `server.py`
  (`_require_trained_arrays`) where a run whose every single round failed to aggregate crashed
  with a confusing "Missing key(s) in state_dict" error instead of a clear one — Flower's
  `Strategy.start()` only assigns `result.arrays` on a successful round, with no fallback to the
  initial model.

## Where to look

- `docs/RESEARCH_ROADMAP.md` — canonical multi-session research plan (gitignored — not on every
  machine by default; copy it manually if a fresh checkout is missing it).
- `reports/*.md`, `reports/*.tex` — narrative writeups; source of truth over this file for
  anything beyond a one-line summary.
- `results/<name>/` — raw run data; `results/archive/` — superseded data kept for comparison.
- `AGENTS.md` — repo conventions, including the branch-per-experiment workflow that explains why
  most in-progress work isn't here yet.
- `git branch -a` — see which experiment branches are currently active.
