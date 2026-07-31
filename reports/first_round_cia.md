# First-Round Single-Shot Client Inference Attack (CIA) — Status Report

Code: `experiments/cia/` (`dataset.py`, `attack.py`, `runner.py`). No result data is committed
anywhere on `master` for this experiment (`results/cia/` does not exist, and the runner's own
default output, `experiments/cia/results/first_round_cia.json`, is not present either). This
report therefore documents scope and design only — there are no numbers to report yet.

## Objective

Reproduces Section 5 and 7.4.1 of Sáinz-Pardo Díaz et al. (2026): a semi-honest client (the
attacker) infers whether another client (the target) participated in training by comparing the
aggregated model's loss on a public test set against its loss on a shadow sample drawn from the
target's own training data. Scope is **first-round, single-shot only** (paper Tables 9–12); the
multi-round AUC-based variant (Section 7.4.2, Table 13) is not implemented.

## Design

1. **Partitioning** (`dataset.py`): splits the Alzheimer MRI train set into 3 clients per the
   paper's Table 9 (client 1 = attacker, client 2 = bystander, client 3 = target). Table 9 as
   published has a typo in client 2's row (1591/180 clients); the code uses the corrected values
   (1491/80), justified by an arithmetic reconciliation against Table 1's row/column/grand totals
   documented in the module's docstring.
2. **Training** (`runner.py`): for each of the 18 `(privacy_mode, aggregation)` combinations, runs
   one real 1-round, 3-client Flower simulation with `local-epochs=20` (the paper's CIA-specific
   value), reusing `experiments.reproduce.runner` unmodified via
   `experiments.cia.dataset:create_cia_data_module`.
3. **Evaluation**: the resulting model's loss is measured on the global held-out test set and on a
   stratified 10% shadow sample of the target's train indices — which overlaps with, rather than
   is excluded from, what the target trains on, matching the paper's stated "strong adversarial
   assumption."
4. **Scoring** (`attack.py`): `relative_difference(aggregated_loss, target_loss)` computes
   `(target_loss - aggregated_loss) / target_loss * 100`, matching Tables 10–12's structure. The
   formula's direction was checked against the paper's own worked example (aggregated=1.032,
   target=1.182 → 12.69%, matching the published 12.719% within table-rounding noise); dividing by
   `aggregated_loss` instead gives 14.5%, which does not match, confirming the denominator choice.

## Comparison target

Running the experiment is meant to produce `experiments/cia/results/first_round_cia.json`, whose
per-`(privacy, aggregation)` rows are comparable to the paper's Table 10 (FedAvg), Table 11
(FedYogi), and Table 12 (test loss only, across all six aggregation strategies). Exact numeric
parity isn't expected (different hardware/library versions, and the paper doesn't specify all
stochastic-seed details); the qualitative pattern to check for is metric-privacy loss lower than
global-DP loss, with comparable CIA protection.

## Test coverage

`experiments/cia/tests/` (`test_dataset.py`, `test_attack.py`, `test_runner.py`) unit-tests the
partitioning arithmetic, the attack-score formula, and the runner's orchestration logic — these
pass, but they exercise the code path, not the paper comparison itself. No CIA simulation has been
run to completion and captured on `master`.

## What's needed for a results report

Run `uv run python -m experiments.cia.runner --output-dir experiments/cia/results` (18 real
1-round, 3-client training runs) and commit the resulting `results/cia/` artifacts, then this
report can be replaced with the per-combination loss table and paper comparison. Related
in-progress work (a separate client-count-scaling variant of this attack) exists on the
`feature/cia-client-scaling` branch, but that work is not yet finished or merged, so it's out of
scope for this report.
