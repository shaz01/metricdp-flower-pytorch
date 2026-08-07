# Project Status

**Branch:** `feature/cifar100-scaling`
**Last updated:** 2026-08-07, CUDA workstation (`feature/cifar100-scaling` active — v2 model
(GroupNorm + 4th conv block), retuned noise_multiplier, augmentation/weight-decay added, cosine LR
schedule tried and dropped, 12/72 v1 combos already run and superseded) — see `git log` for
anything more recent

This file is a short, git-tracked pickup point for any Claude Code session — this machine or
another — starting work on this repo. It reflects the branch it's committed on; check out the
branch you're working on before trusting it. Treat it as a pointer, not the source of truth: for
full narrative detail see `reports/`, for raw run data see `results/`, for chronological detail
use `git log`. Update it whenever a branch merges into `master` or `master`-level state otherwise
changes — keep it short, don't turn it into a changelog. Exception: the Active work
section (including the Currently running table) updates more often, at "worth a commit"
granularity — see `AGENTS.md`'s "Working across machines" section.

## Active work

`feature/cifar100-scaling` (not yet merged) adds a CIFAR-100 dataset/model plugin pair and a
client-count/round-budget sweep script: `experiments/reproduce/dataset/cifar100.py` (full
100-class CIFAR-100 data plugin — unlike every other dataset plugin in this repo, which subsets to
4 classes — now with train-only data augmentation, random crop + horizontal flip), and
`experiments/reproduce/cifar100_cnn.py`, now on its v2 architecture: GroupNorm after each conv plus
a 4th conv block (~2.76M params, up from v1's ~2.6M, 3 blocks, no normalization). GroupNorm has no
running-stats buffers — unlike BatchNorm — so it still needs no `metricdp_pytorch/metricdp_strategy.py`
changes, same as v1's "no normalization at all" workaround. Training now also supports weight decay
(`WEIGHT_DECAY = 5e-4` in the sweep) and an opt-in cosine LR schedule; the schedule was tried for
this sweep and dropped — a decaying LR eventually drops client updates below `clipping_norm`, so
clipping stops binding late in each run and the DP noise-to-signal ratio drifts back up against the
retuned `noise_multiplier`, fighting the whole point of retuning it — so the sweep uses a fixed LR
instead (see `experiments/cifar100_scaling/sweep_cifar100_scaling.py`'s docstring/comments). For the
new v2 model, `noise_multiplier` was recalibrated from v1's `0.05` down to `0.0025`, verified via a
smoke run (`clipping_norm=5.0` unchanged). `experiments/cifar100_scaling/sweep_cifar100_scaling.py`
is the resumable sweep script: client counts 8/64/128/256 x rounds 20/60/120 x privacy
vanilla/global-dp/metric-privacy x partition homogeneous/non-iid x `fedavg` = 72 combos. 12 of
those 72 combos were actually run under the earlier v1 model/`noise_multiplier=0.05`, and their
(now-superseded) result files sit in `results/cifar100_scaling/` under the same run names;
`is_complete()` treats them as done, so relaunching the sweep on v2 requires `--force` to overwrite
them. Next step: launch (or relaunch with `--force`) the full 72-combo sweep
(`uv run python -m experiments.cifar100_scaling.sweep_cifar100_scaling`) — a separate, multi-day
action, not started.

Separately, on `master`: nothing currently running. `feature/scale-controlled-redo` (Phase 1 items
1 and 2 of `docs/RESEARCH_ROADMAP.md`) merged into `master` 2026-08-06 and was deleted — see
"What's established" below for what it left behind. The natural next steps there, not yet started:
`fedyogi` at `n=48` (the redo's matrix only covers `n=4/8` for `fedyogi`), and Phase 1's remaining
item (NaN/failure-mode logging in `runner.py`, motivated directly by the zero-norm-update crashes
found during the redo). After that, Phase 2 (mechanism redesign) is the next major phase.

### Currently running

Nothing. Update this table whenever a machine picks up new work: add a row, edit the Status column
in place (e.g. `running` -> `done`), and leave a finished row for one update cycle before removing
it, so machine-to-results provenance isn't lost; see `AGENTS.md`'s "Working across machines"
section.

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

## Where to look

- `docs/RESEARCH_ROADMAP.md` — canonical multi-session research plan (gitignored — not on every
  machine by default; copy it manually if a fresh checkout is missing it).
- `reports/*.md`, `reports/*.tex` — narrative writeups; source of truth over this file for
  anything beyond a one-line summary.
- `results/<name>/` — raw run data; `results/archive/` — superseded data kept for comparison.
- `AGENTS.md` — repo conventions, including the branch-per-experiment workflow that explains why
  most in-progress work isn't here yet.
- `git branch -a` — see which experiment branches are currently active.
