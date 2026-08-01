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

**Caveat on shadow-sample size at this scale:** at 48 clients, 10% of the
target client's own (already much smaller) per-client partition works out
to roughly 8-15 images, versus ~150 in the original 3-client
`experiments/cia/` experiment. The 10% fraction is kept as-is to stay
faithful to the paper's methodology, but `difference_pct` computed from a
handful of images is noisier than the 3-client numbers it's tempting to
compare it against. Every result row includes `shadow_size` for exactly
this reason — read `difference_pct` alongside it, and don't compare
`difference_pct` directly against `experiments/cia/`'s numbers without
accounting for the sample-size gap.

## Running it

```bash
uv run python -m experiments.cia_client_scaling.runner --output-dir results/cia_client_scaling
```

This is resumable: rerunning skips any combo/timing whose training result
JSON already shows the expected number of completed rounds *and* whose
`.pt` checkpoint is actually present on disk. Pass `--force` to ignore
existing results and rerun everything, or `--timings first-round` (or
`--timings post-convergence`) to run only one variant — useful given the
post-convergence variant is a full 20-round, 48-client training per combo
and the more expensive half of this sweep.

At 48 clients, per-round work (and Ray actor memory) is substantially
higher than the smaller-scale experiments this one builds on. `--max-parallel-clients`
(default 4) controls how many client processes run concurrently; lower it
if you hit memory pressure on your machine, at the cost of slower rounds.

Results are written to `results/cia_client_scaling/cia_client_scaling.json`
and printed to stdout. The report file is a `{"results": [...], "failed":
[...]}` object, not a flat list: `results` holds one record per
combo/timing (each including `shadow_size` — see the caveat above), and
`failed` lists the run names that errored out. Because the report is keyed
by `(timing, partition_mode, privacy, aggregation)` and rewritten after
every combo, running `--timings first-round` and `--timings
post-convergence` as two separate invocations merges both sets of results
into the same file instead of the second invocation overwriting the
first, and a crash mid-sweep loses at most the one combo in flight. Per-combo
raw training JSONs and saved model checkpoints (`*.pt`) live alongside the
report in the same directory.

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
