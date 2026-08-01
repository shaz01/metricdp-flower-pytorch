# Agent Instructions

This file is the canonical source of working conventions for this repo. `CLAUDE.md` is a symlink to this file — edit `AGENTS.md`, not `CLAUDE.md`.

## Project overview

Federated learning + differential privacy research repo built on Flower and PyTorch, reproducing and extending a published metric-privacy noise-calibration mechanism. See `README.md` for the algorithm description and aggregator hyperparameters — don't re-derive that here.

## Environment & setup

- Install with `uv sync`. Python is pinned via `.python-version` (3.13; `pyproject.toml` requires `>=3.11,<3.14`).
- Run everything through `uv run <cmd>` (e.g. `uv run pytest`, `uv run python -m ...`).
- Device selection is automatic: CUDA → MPS → CPU, via `metricdp_pytorch/utils/device.py:resolve_device()`.
- Known inconsistency: `experiments/cia/runner.py` and `experiments/reproduce/detailed_evaluation.py` use a narrower inline device check that skips MPS, so those two entry points fall back to CPU on Apple Silicon instead of using `resolve_device()`.

## Testing

- Run the suite with `uv run pytest` from the repo root. The default `addopts` deselects the `reproducibility` marker (5 cross-version port-equivalence tests that need an isolated legacy environment).
- To include those: `uv run pytest -m reproducibility experiments/port_equivalence/test_equivalence.py`.
- Current test locations: `tests/`, `experiments/reproduce/tests/`, `experiments/cia/tests/`, `experiments/cia_client_scaling/tests/`, `experiments/port_equivalence/test_equivalence.py`.
- `experiments/client_scaling/` has no tests yet — it's sweep/analysis scripts only.

## Linting

No enforced or configured lint pipeline exists (no `[tool.ruff]`, no `ruff.toml`, ruff isn't in `uv.lock`). `.ruff_cache/` is leftover from ad hoc `uvx ruff` runs, not a project dependency. Don't assume a lint gate exists, and don't invent lint config unprompted.

## Running experiments

- Smoke test: `uv run python -m experiments.reproduce.runner --smoke`
- Full run: `uv run python -m experiments.reproduce.runner --partition <homogeneous|non-iid> --privacy <vanilla|global-dp|metric-privacy> --aggregation <fedavg|fedavgm|fedmedian|fedprox|fedopt|fedyogi> --rounds 20 --local-epochs 5`
- Matrix run: `uv run python -m experiments.reproduce.matrix_runner ...`
- CIA experiment: `uv run python -m experiments.cia.runner ...`
- Registered Flower App: `uv run flwr run . --stream`

See `README.md` and each experiment's own `README.md` (where present) for full flag references.

## Repo structure

- `metricdp_pytorch/` — core library (privacy mechanism, strategy factory, data/device utils). Shared across all experiments.
- `experiments/<name>/` — one folder per experiment (`reproduce`, `cia`, `client_scaling`, `cia_client_scaling`, `port_equivalence`), each with its own code and, where applicable, its own `tests/`.
- `results/<name>/` — every experiment's real output data lives here on `master`, organized to mirror the experiment folder names. Raw logs, one-off shell scripts, and lock/status files don't belong here — only real result data (run JSONs, evaluation summaries, READMEs).
- `papers/` — reference PDFs, not experiment code.
- `docs/` — gitignored, local-only notes. Don't expect it to exist on a fresh clone, and don't treat its absence as a problem.

## Experiment reports

- Once an experiment is finished, write a detailed Markdown report under `experiments/<name>/reports/`. Raw run data (JSONs, evaluation summaries, predictions, checkpoints) still lives in `results/<name>/`; `experiments/<name>/reports/` holds the narrative writeup that interprets that data — protocol, per-run/aggregate tables, anomalies, and conclusions.
- Reports must be built from real numbers pulled from the committed result artifacts, never fabricated or estimated.
- **Don't decide unilaterally that an experiment is finished.** Only the project owner makes that call. When it looks like an experiment has wrapped up, ask before writing its report (or before treating it as done in any other way) — don't just go write one.

## Git workflow

- Each experiment gets its own branch: `feature/<experiment-name>`, branched from `master`.
- While an experiment is active, its code and results stay on that branch — don't merge early.
- Once an experiment is finished: merge its branch into `master` (organizing its code into `experiments/<name>/` and its results into `results/<name>/` if not already structured that way), run the test suite to confirm the merge is clean, then delete the branch both locally and on `origin` (`git branch -d`, `git push origin --delete`). Don't leave fully-merged branches lying around.
- Prefer a real `git merge` over rebase/squash for finished experiment branches — it preserves each experiment's actual commit history.
- **Never add a `Co-Authored-By` trailer to commits.** This is a strict, explicit rule from the project owner.
