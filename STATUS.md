# Project Status

**Branch:** `feature/cifar100-scaling`
**Last updated:** 2026-08-09, CUDA workstation (`feature/cifar100-scaling` active — the 6-combo
100-client/250-round sweep on the supervisor-derived model finished: 6/6 attempted, 0 failed, total
wall clock ~18h20m 2026-08-08 17:46 to 2026-08-09 12:05. Results committed. Next up: a CIA attack
experiment on CIFAR-100) — see `git log` for anything more recent

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
sweep script: `experiments/reproduce/dataset/cifar100.py` (full 100-class CIFAR-100 data plugin —
unlike every other dataset plugin in this repo, which subsets to 4 classes — with train-only data
augmentation, random crop + horizontal flip), and `experiments/reproduce/cifar100_cnn.py`.

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

`experiments/cifar100_scaling/sweep_cifar100_scaling.py` is the resumable sweep script: a single
client count (100) and round budget (250) — narrowed from earlier client-count/round-count grids
per project-owner direction, judged too short — x privacy vanilla/global-dp/metric-privacy x
partition homogeneous/non-iid x `fedavg` = 6 combos. **Finished 2026-08-09, 12:05, 6/6 attempted, 0
failed.** Final-round (250) server-side accuracy/loss (100-class task, so ~1% is random-baseline):

| Partition | Privacy | Accuracy | Loss | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| homogeneous | vanilla | 29.14% | 2.8118 | 0.2688 | 0.9249 |
| homogeneous | global-dp | 21.42% | 3.2273 | 0.1882 | 0.8919 |
| homogeneous | metric-privacy | 22.00% | 3.2137 | 0.1952 | 0.8929 |
| non-IID | vanilla | 28.28% | 2.8552 | 0.2593 | 0.9225 |
| non-IID | global-dp | 21.40% | 3.1908 | 0.1904 | 0.8960 |
| non-IID | metric-privacy | 21.44% | 3.1939 | 0.1888 | 0.8956 |

Metric-privacy roughly ties global-dp here (+0.58pp homogeneous, +0.04pp non-IID) rather than
beating it, both DP modes ~7-8pp below vanilla; homogeneous vs. non-IID partitioning barely moves
the numbers at this scale. Notably lower absolute accuracy than the Alzheimer-dataset scaling
sweeps (77-98%) — expected, since CIFAR-100 is a much harder 100-way task and this budget may be
short of what full convergence needs; not yet interpreted/written up, no report exists for this
experiment yet — see `AGENTS.md`'s "don't decide unilaterally an experiment is finished" rule.
Run-result JSONs are committed at `results/cifar100_scaling/*.json`; the `.evaluation.json` (full
per-class ROC, 230-400MB each) and `.predictions.npz` artifacts stay local-only, gitignored (see
`.gitignore` comment) since they blow past GitHub's 100MB push limit.

Next up, per project-owner direction: a CIA (Client Inference Attack) experiment on CIFAR-100,
building on `experiments/cia/` with fresh federated-learning trajectories using the same combos
and hyperparameters as the sweep (the sweep's own checkpoints are gone, so new trajectories must
be trained).

This plan adds two new scripts under `experiments/cia/scripts/` to execute that CIA experiment:
`cifar100_scaling.py` trains fresh federated-learning IN/OUT trajectories for each privacy mode
and partition combo, collecting per-checkpoint loss values for the attack. `cifar100_scaling_analysis.py`
reads the loss records from both groups, computes round-matched AUC and bootstrap confidence
intervals per (partition, privacy) combo, and writes the results to a JSON summary. Both scripts
are ready to launch (not yet run). Execution is two-group: `uv run python -m experiments.cia.scripts.cifar100_scaling --group in`
and `--group out`, intended to run concurrently (e.g. on separate GPUs or tmux sessions). Expected
wall-clock per group is roughly the sweep's own ~18h; run concurrently, the total time should land
somewhere between that single-group duration and half of it, depending on hardware contention
(no guaranteed 2x speedup).

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
| `experiments.cifar100_scaling.sweep_cifar100_scaling` (tmux session `cifar100sweep`, `CUDA_VISIBLE_DEVICES=1`) | 6-combo CIFAR-100 sweep (n=100, r=250, 3 privacy modes x 2 partitions) on the current (supervisor-derived) model | done, 2026-08-09 |

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
