# Client Inference Attack (CIA) at 48-client scale

Extends the paper's first-round single-shot Client Inference Attack
(`experiments/cia/`, Sáinz-Pardo Díaz et al. 2026, Section 5 and 7.4.1) to
the 48-client setting used in the concluded client-scaling experiment
(`experiments/client_scaling/`, `results/48client_scaling/`), which never
evaluated CIA risk.

## What this does

For each of 12 `(partition_mode, privacy, aggregation)` combinations —
`{homogeneous, non-iid} x {vanilla, global-dp, metric-privacy} x {fedavg,
fedyogi}` — runs two timing variants via the existing, unmodified
`experiments.reproduce.runner`:

1. **first-round**: 1 round, local-epochs=20, noise-multiplier=0.01,
   clipping-norm=5.0 — the exact hyperparameters `experiments/cia/runner.py`
   uses, just with `--num-clients 48` instead of 3. Directly comparable to
   `experiments/cia/results/first_round_cia.json`.
2. **post-convergence**: 20 rounds, local-epochs=5, noise-multiplier=0.05,
   clipping-norm=5.0 — the exact hyperparameters
   `experiments/client_scaling/sweep_48_clients.py` used, with `--save-model`
   added (the original sweep never saved checkpoints). Directly comparable
   to `results/48client_scaling/`.

For each combo/timing, evaluates the resulting saved model's loss on the
global held-out test set and on a fixed target client's (`partition_id=0`)
shadow split — a stratified 10% of that client's own train indices, which
overlaps with (not excluded from) what it actually trained on, matching the
paper's "strong adversarial assumption". Reports
`(target_loss - aggregated_loss) / target_loss * 100` per combination,
mirroring `experiments/cia/`'s Table 12 structure.

## Running it

```bash
uv run python -m experiments.cia_client_scaling.runner --output-dir results/cia_client_scaling
```

This is resumable: rerunning skips any combo/timing whose training result
JSON already shows the expected number of completed rounds. Pass `--force`
to ignore existing results and rerun everything, or `--timings first-round`
(or `--timings post-convergence`) to run only one variant — useful given
the post-convergence variant is a full 20-round, 48-client training per
combo and the more expensive half of this sweep.

Results are written to
`results/cia_client_scaling/cia_client_scaling.json` (one record per
combo/timing) and printed to stdout. Per-combo raw training JSONs and saved
model checkpoints (`*.pt`) live alongside it in the same directory.

## Comparing results

- Compare `timing="first-round"` rows against
  `experiments/cia/results/first_round_cia.json` to see whether the
  3-client paper-scale CIA protection pattern (metric-privacy loss lower
  than global-DP loss, comparable attack difference) holds at 48 clients
  under identical hyperparameters.
- Compare `timing="post-convergence"` rows against the accuracy findings in
  `results/48client_scaling/` to see what membership-leakage risk
  accompanies the accuracy numbers actually reported for that scaling
  study.
