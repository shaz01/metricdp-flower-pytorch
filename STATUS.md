# Project Status

**Branch:** `feature/scaling-diagnosis` (not yet merged to `master`)
**Last updated:** 2026-08-04, commit `94a4542`

This file is a short, git-tracked pickup point for any Claude Code session — this machine or
another — starting work on this repo. It reflects the branch it's committed on; check out the
branch you're working on before trusting it. Treat it as a pointer, not the source of truth: for
full narrative detail see `reports/`, for raw run data see `results/`, for chronological detail
use `git log`. Update the relevant sections here whenever you finish a unit of work that changes
what's established or what's next — keep it short, don't turn it into a changelog.

## Active work

Diagnosing why the metric-privacy noise-calibration mechanism's accuracy advantage over
global-DP — real at 8 clients (`reports/client_count_scaling.md`: +6.9pp homogeneous, +12.2pp
non-IID at `noise_multiplier=0.05`) — shrinks or reverses at 48 clients
(`results/48client_scaling`), and whether that's a genuine mechanism failure or a confound with
the fixed 20-round training budget every earlier sweep used regardless of client count.

Full writeup: `reports/constant_compute_scaling.md` (detailed, v1 design + corrections) and
`reports/progress_report_phase1.tex`/`.pdf` (compiled, supervisor-facing progress report covering
both sweep designs and the MPS finding below — also committed on `master`).

## What's established (safe to build on)

- Two constant-compute experimental controls exist and have each completed a full 12-combination
  sweep: v1 (`experiments/client_scaling/sweep_scale_controlled.py`, scales rounds with client
  count) and v2 (`sweep_scale_controlled_epochs.py`, holds rounds fixed and scales local epochs
  instead — the corrected design, after v1 was found to confound round count with aggregation
  frequency).
- Apple MPS's backward pass is non-deterministic across process launches when multiple Ray client
  actors train concurrently on the shared device, even with every seed fixed. Directly proven, not
  inferred: re-running the identical v1/v2 `n=4` baseline config (same seed, same hyperparameters)
  produced accuracy deltas of 13.75–53.28pp across the four combinations
  (`metricdp_pytorch/utils/device.py`'s `resolve_device` docstring has the full investigation). An
  opt-in `METRICDP_FORCE_CPU=1` override exists for runs that need exact reproducibility, at ~3x
  the wall-clock cost.
- Five real infrastructure bugs (MPS unified-memory leak, process orphaning, DataLoader worker
  churn, a per-round dataset-reload race, an uncaught `ZeroDivisionError` on model collapse) are
  fixed and covered by tests. `uv run pytest` passes (47 passed, 5 reproducibility-marker tests
  deselected by default, as of this commit).

## What's NOT established (do not cite as signal)

- **Every single-seed accuracy delta produced on this machine's MPS backend is suspect** until
  individually noise-checked, given the proven noise floor above. Two have been checked so far via
  repeated-rep spot checks, and **both failed to hold up as originally stated**:
  - Homogeneous/n=48 reversal (v1 reported −18.91pp): weak directional support only, magnitude not
    trustworthy (`results/noise_floor_check/`).
  - Non-IID/n=48 delta (v1 reported +3.12pp): not established, sign flips across reps
    (`results/noise_floor_check_noniid/`).
- Everything else remains unverified: v1's `n=8` points, all of v2
  (`results/scale_controlled_epochs/`, never checked), and the original
  `8client_scaling`/`48client_scaling`/`noise_sweep` result sets that motivated this whole
  investigation in the first place. The noise magnitude found for one finding does **not**
  generalize to another (the two checked so far failed via two different mechanisms — large effect
  swamped by larger noise vs. small effect swamped by comparable noise) — each remaining
  single-seed number needs its own check before being cited.

## Immediate next decision (not yet made)

Three options are on the table, not yet chosen: (1) keep noise-checking existing findings one at a
time (~3.5–4.5h each, purely retrospective), (2) proceed to new diagnostic work (per-client-count
noise-multiplier re-sweep, pairwise-distance-distribution instrumentation, failure-mode logging)
on the current MPS pipeline, accepting the same unresolved noise on the new numbers too, or (3)
switch to `METRICDP_FORCE_CPU=1` for all future reproducibility-critical runs before doing either.
See `reports/progress_report_phase1.tex`, section "Next Steps", for the full tradeoff writeup.

## Where to look

- `reports/*.md`, `reports/*.tex` — narrative writeups; always the source of truth over this file
  for anything beyond a one-line summary.
- `results/<name>/` — raw run JSONs backing every number above.
- `AGENTS.md` — repo conventions (branch-per-experiment, testing, running experiments).
- `git log` — chronological detail; this file intentionally does not repeat it.
