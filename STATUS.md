# Project Status

**Branch:** `master`
**Last updated:** 2026-08-04

This file is a short, git-tracked pickup point for any Claude Code session — this machine or
another — starting work on this repo. It reflects the branch it's committed on; check out the
branch you're working on before trusting it. Treat it as a pointer, not the source of truth: for
full narrative detail see `reports/`, for raw run data see `results/`, for chronological detail
use `git log`. Update it whenever a branch merges into `master` or `master`-level state otherwise
changes — keep it short, don't turn it into a changelog.

## Active work

`feature/scaling-diagnosis` (not yet merged) is diagnosing why the metric-privacy
noise-calibration mechanism's accuracy advantage over global-DP — real at 8 clients (+6.9pp
homogeneous, +12.2pp non-IID at `noise_multiplier=0.05`, see below) — shrinks or reverses at 48
clients, and whether that's a genuine mechanism failure or a confound with the fixed round budget
every earlier sweep used. That branch's own `STATUS.md` has the detailed current state; short
version: two experimental controls are built and have both been run, but a proven
non-determinism in this machine's MPS backend (noise floor comparable to the effect sizes being
measured) means no single-run result from either control can currently be trusted at face value.
See `reports/progress_report_phase1.tex`/`.pdf` — a supervisor-facing progress report, committed
here ahead of the rest of that branch (which stays on `feature/scaling-diagnosis` until finished,
per `AGENTS.md`'s branch-per-experiment convention).

## What's established on `master`

- The metric-privacy mechanism reproduces the source paper at 4 clients — the effect is barely
  visible at the paper's `noise_multiplier=0.01` (`reports/paper_reproduction.md`).
- A genuine, previously unpublished effect exists at 8 clients: metric-privacy beats global-DP by
  +6.9pp (homogeneous) / +12.2pp (non-IID) at `noise_multiplier=0.05`
  (`reports/client_count_scaling.md`).
- First-round CIA and the Flower-1.32 port-equivalence check both have status writeups but no (or
  partial) result data yet — see `reports/first_round_cia.md`, `reports/port_equivalence.md` for
  exact scope.

## Where to look

- `reports/*.md`, `reports/*.tex` — narrative writeups; source of truth over this file for
  anything beyond a one-line summary.
- `results/<name>/` — raw run data.
- `AGENTS.md` — repo conventions, including the branch-per-experiment workflow that explains why
  most in-progress work isn't here yet.
- `git branch -a` — see which experiment branches are currently active.
