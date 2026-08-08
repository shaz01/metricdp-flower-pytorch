# Agent Instructions

This file is the canonical source of working conventions for this repo. `CLAUDE.md` is a symlink to this file — edit `AGENTS.md`, not `CLAUDE.md`.

## Project overview

Federated learning + differential privacy research repo built on Flower and PyTorch, reproducing and extending a published metric-privacy noise-calibration mechanism. See `README.md` for the algorithm description and aggregator hyperparameters — don't re-derive that here.

## Environment & setup

- Install with `uv sync`. Python is pinned via `.python-version` (3.13; `pyproject.toml` requires `>=3.11,<3.14`).
- Run everything through `uv run <cmd>` (e.g. `uv run pytest`, `uv run python -m ...`).
- Device selection is automatic: CUDA → MPS → CPU, via `metricdp_pytorch/utils/device.py:resolve_device()`.
- `experiments/reproduce/detailed_evaluation.py` used to have a narrower inline device check that skipped MPS; fixed 2026-08-01 after it caused a real failure — its postprocessed-vs-recorded accuracy consistency check compares against a training-time evaluation that runs on MPS via `resolve_device()`, so evaluating on CPU there could diverge enough on borderline/near-random-accuracy models to fail that check outright. `experiments/cia/runner.py` had the same issue but no longer exists (refactored away 2026-08-03 into `experiments/cia/attack_runner.py` and `experiments/cia/scripts/*.py`); every current entry point under `experiments/cia/scripts/` now imports and uses `resolve_device()` directly, so this inconsistency no longer applies anywhere in the repo.

## Testing

- Run the suite with `uv run pytest` from the repo root. The default `addopts` deselects the `reproducibility` marker (5 cross-version port-equivalence tests that need an isolated legacy environment).
- To include those: `uv run pytest -m reproducibility experiments/port_equivalence/test_equivalence.py`.
- Current test locations: `tests/`, `experiments/reproduce/tests/`, `experiments/cia/tests/`, `experiments/port_equivalence/test_equivalence.py`.
- `experiments/client_scaling/` has no tests yet — it's sweep/analysis scripts only.

## Linting

No enforced or configured lint pipeline exists (no `[tool.ruff]`, no `ruff.toml`, ruff isn't in `uv.lock`). `.ruff_cache/` is leftover from ad hoc `uvx ruff` runs, not a project dependency. Don't assume a lint gate exists, and don't invent lint config unprompted.
See `README.md` and each experiment's own `README.md` (where present) for full flag references.

## Repo structure

- `metricdp_pytorch/` — core library (privacy mechanism, strategy factory, data/device utils). Shared across all experiments.
- `experiments/<name>/` — one folder per experiment (`reproduce`, `cia`, `client_scaling`, `port_equivalence`), each with its own code and, where applicable, its own `tests/`.
- `results/<name>/` — every experiment's real output data lives here on `master`, organized to mirror the experiment folder names. Raw logs, one-off shell scripts, and lock/status files don't belong here — only real result data (run JSONs, evaluation summaries, READMEs).
- `papers/` — reference PDFs, not experiment code.
- `docs/` — gitignored, local-only notes. Don't expect it to exist on a fresh clone, and don't treat its absence as a problem.

## Experiment reports

- Once an experiment is finished, write a detailed Markdown report under the root-level `reports/` directory (one file per experiment, e.g. `reports/paper_reproduction.md`). Raw run data (JSONs, evaluation summaries, predictions, checkpoints) still lives in `results/<name>/`; `reports/` holds the narrative writeup that interprets that data — protocol, per-run/aggregate tables, anomalies, and conclusions.
- Reports must be built from real numbers pulled from the committed result artifacts, never fabricated or estimated.
- **Don't decide unilaterally that an experiment is finished.** Only the project owner makes that call. When it looks like an experiment has wrapped up, ask before writing its report (or before treating it as done in any other way) — don't just go write one.

## Git workflow

- Each experiment gets its own branch: `feature/<experiment-name>`, branched from `master`.
- While an experiment is active, its code and results stay on that branch — don't merge early.
- Once an experiment is finished: merge its branch into `master` (organizing its code into `experiments/<name>/` and its results into `results/<name>/` if not already structured that way), run the test suite to confirm the merge is clean, then delete the branch both locally and on `origin` (`git branch -d`, `git push origin --delete`). Don't leave fully-merged branches lying around.
- Prefer a real `git merge` over rebase/squash for finished experiment branches — it preserves each experiment's actual commit history.
- **Never add a `Co-Authored-By` trailer to commits.** This is a strict, explicit rule from the project owner.

## Working across machines

- This project runs across multiple machines (this repo has no built-in cross-machine session sync — each machine's Claude Code install is independent, keyed by its own local project path). Compensate with a habit, not a tool.
- **Session start**: before making any changes, read `STATUS.md` and skim recent history (`git log --oneline -10`) to pick up state left by other machines/sessions.
- **Session end**: when a meaningful chunk of work wraps up (same granularity as "worth a commit" — not every message), update `STATUS.md`'s Active work section — current state, what's running where (the Currently running table), next steps — refresh the "Last updated" line, then commit and push. This Active-work-section update is more frequent than `STATUS.md`'s own top-level "whenever a branch merges into master" rule, which still governs the rest of the file.
- `docs/RESEARCH_ROADMAP.md` is gitignored and doesn't travel via `git pull`; if it changes, copy it to other machines manually.
- Track "what's running where" in `STATUS.md`'s Currently running table using generic machine-role labels (e.g. "CUDA workstation", "CUDA laptop") — never hostnames, IPs, or usernames, in this or any other committed doc.
