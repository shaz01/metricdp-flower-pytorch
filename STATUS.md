# Project Status

**Branch:** `master`
**Last updated:** 2026-08-04, commit `cf0e29c` (merge of a large collaborator-authored batch of
work — runner refactor, model-module abstraction, CIA client-scaling — see below)

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
- `reports/first_round_cia.md` is **stale as of this merge** — it says "no result data yet," but
  `results/cia_client_scaling/` now has real trained models and partial attack scores (see below).
  Needs a rewrite; not done as part of this merge. The Flower-1.32 port-equivalence check
  (`reports/port_equivalence.md`) is unaffected by this merge and still has no committed result
  data.

## What changed in this merge (previously only on other branches/machines)

Landed via merging `origin/master`, authored by `atahakancildas` and a collaborator (`olcay`):

- **Runner refactor**: the old `experiments/reproduce/matrix_runner.py` is now
  `experiments/reproduce/matrix/` (a proper package, "Matrix API"), shared by both the CIA and
  client-scaling sweep scripts via new `sweep_runner.py`/`iter_combos.py` helpers.
- **Pluggable model layer**: `metricdp_pytorch/model_module.py` + a `--model-module` CLI flag —
  the model is now swappable the same way the data module already was. First non-Alzheimer
  dataset/model pair added: Fashion-MNIST (`experiments/reproduce/dataset/fashion_mnist.py`,
  `fashion_mnist_cnn.py`).
- **CIA experiment overhaul**: `attack.py` → `result.py`; new `experiments/cia/client_scaling.py`
  runs a 48-client CIA variant that attacks a single trained trajectory at checkpoints (round 1,
  round 20) instead of retraining per attack. Real data now exists at
  `results/cia_client_scaling/`: 18 models trained (full first-round matrix + a
  `noise_multiplier=0.12`/FedYogi-specific set), but only 6 attacks are actually scored in
  `cia_client_scaling.json` so far — all `homogeneous / fedyogi / nm=0.12`, split between
  first-round and post-convergence checkpoints. The other 12 trained combos (fedavg, non-IID,
  default `nm=0.05`) don't have attack scores yet — **this experiment is in progress, not
  finished**, don't treat it as a complete result set.
- New diagnostics: `metricdp_pytorch/dp_diagnostics.py`, `globaldp_strategy.py`, deterministic-mode
  settings added to `paper_training.py`.
- Test suite grew from 47 to 71 passing tests (still 5 reproducibility-marker tests deselected by
  default).

## Where to look

- `reports/*.md`, `reports/*.tex` — narrative writeups; source of truth over this file for
  anything beyond a one-line summary.
- `results/<name>/` — raw run data.
- `AGENTS.md` — repo conventions, including the branch-per-experiment workflow that explains why
  most in-progress work isn't here yet.
- `git branch -a` — see which experiment branches are currently active.
