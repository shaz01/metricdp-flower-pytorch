# First-round single-shot Client Inference Attack (CIA)

Reproduces Section 5 and 7.4.1 of Sáinz-Pardo Díaz et al. (2026): a
semi-honest client (attacker) infers whether another client (target)
participated in training by comparing the aggregated model's loss on a
public test set against its loss on a shadow sample drawn from the target's
own training data.

Scope: **first-round single-shot only** (Table 9-12). The multi-round
AUC-based variant (Section 7.4.2, Table 13) is not implemented here.

## What this does

1. Partitions the Alzheimer MRI train set into 3 clients using the paper's
   Table 9 distribution (client 1 = attacker, client 2 = bystander, client 3
   = target). Table 9 as published has a typo in Client 2's row (1591/180);
   the corrected values (1491/80) are used here -- see
   `experiments/cia/dataset.py`'s module docstring for the arithmetic proof
   (row sum, column sums against Table 1, and the grand total all reconcile
   only with the corrected values).
2. For each of the 18 `(privacy_mode, aggregation)` combinations, runs one
   real 1-round, 3-client Flower simulation with `local-epochs=20` (the
   paper's CIA-specific value), reusing `experiments.reproduce.runner`
   unmodified via `experiments.cia.dataset:create_cia_data_module`.
3. Evaluates the resulting model's loss on the global held-out test set and
   on a stratified 10% shadow sample of the target's train indices (which
   overlaps with, not excluded from, what the target trains on -- matching
   the paper's stated "strong adversarial assumption").
4. Reports `(target_loss - aggregated_loss) / target_loss * 100` per
   combination, matching Tables 10-12's structure.

## Running it

```bash
uv run python -m experiments.cia.runner --output-dir experiments/cia/results
```

This takes a while: 18 real training runs, each downloading/reusing the
cached Alzheimer MRI dataset and training 3 clients for 20 local epochs.
Results are written to `experiments/cia/results/first_round_cia.json` and
printed to stdout.

## Comparing against the paper

The paper's Table 10 (FedAvg) and Table 11 (FedYogi) report, per privacy
mode: aggregated test loss, target shadow loss, and the relative
difference. Table 12 reports test loss only, across all six aggregation
strategies. Compare the corresponding rows in
`experiments/cia/results/first_round_cia.json` against those tables. Exact
numeric parity is not expected (different hardware/library versions, and
the paper doesn't specify all stochastic-seed details), but the qualitative
pattern -- metric-privacy loss lower than global-DP loss, with comparable
CIA protection -- should hold.
