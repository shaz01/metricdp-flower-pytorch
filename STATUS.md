# Project Status

**Branch:** `master`
**Last updated:** 2026-08-05, commit `fa32977` (merge of `feature/scaling-diagnosis`) plus
follow-on fixes `eb98bd2`/`757fac5` (client-ID-ordered aggregation)

This file is a short, git-tracked pickup point for any Claude Code session — this machine or
another — starting work on this repo. It reflects the branch it's committed on; check out the
branch you're working on before trusting it. Treat it as a pointer, not the source of truth: for
full narrative detail see `reports/`, for raw run data see `results/`, for chronological detail
use `git log`. Update it whenever a branch merges into `master` or `master`-level state otherwise
changes — keep it short, don't turn it into a changelog.

## Active work

`feature/scale-controlled-redo` (not yet merged) is redoing Phase 1 Part 1 (the constant-compute
control sweep — `docs/RESEARCH_ROADMAP.md`) from scratch on CUDA hardware, now that two things
changed since the last attempt:

1. **A genuine non-determinism source was found and fixed on master**: Flower's own weighted
   aggregation (`aggregate_arrayrecords`/`aggregate_metricrecords`) summed client replies in
   network-arrival order, not a deterministic order — floating-point addition isn't associative,
   so run-to-run arrival jitter produced small numeric drift that compounds over rounds. Fixed via
   a `DeterministicReplyOrderMixin` (`metricdp_pytorch/strategy_factory.py`) and a matching fix in
   `metricdp_pytorch/metrics.py`, sorting by the already-present per-reply `client-id` metric
   before aggregating.
2. Runs are moving off this Mac's MPS backend entirely, which has its own separate, unfixable
   non-determinism (see `metricdp_pytorch/utils/device.py:resolve_device()`'s docstring) — onto
   CUDA hardware instead.

The old MPS-era v1 (rounds-fixed)/v2 (epoch-scaled) sweep attempts are archived at
`results/archive/scale_controlled_mps_v1v2/` (see its `README.md`) rather than deleted, and
`reports/archive/constant_compute_scaling_mps_v1v2.md` is the write-up they supported — both
superseded, kept for historical comparison only.

`sweep_scale_controlled.py`/`sweep_scale_controlled_epochs.py` gained `--client-counts` and
`--aggregation-methods` override flags so the redo can be split across multiple machines instead
of running the whole matrix sequentially on one box.

### Currently running

| Machine role | Task | Status |
|---|---|---|
| GPU workstation | fedavg matrix (client counts 4/8/48), `sweep_scale_controlled(_epochs)` | running |
| GPU laptop | fedyogi at n=4/n=8, `sweep_scale_controlled(_epochs)` | running |

Neither finished yet — don't treat `results/scale_controlled*/` as complete. Update this table
(add/remove/edit rows) whenever what's running changes; see `AGENTS.md`'s "Working across
machines" section.

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
