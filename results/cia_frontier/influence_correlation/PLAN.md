# Runs

Question: which definition of a client's influence predicts how exposed that client is to the
client inference attack?

Stage C only scores 7 of 48 clients per setting. Here the federation is small so every client
gets an attack score.

## Setup
- 16 clients, 450 images each (a fixed 7,200-image stratified pool of the Stage B datasets).
  Same data per client as Stage B/C.
- cifar10_cnn, Dirichlet α = 0.3, 50 rounds (Stage C's plateau result), CIFAR-10 and EuroSAT 32×32.
- Arms: vanilla, global-DP 0.004, metric-privacy 0.004 (one ratio in the gap region).
- Seed 42 first.
- Per (dataset, arm, seed): 1 IN, 16 OUT runs
- = 2 × 3 × 17 = 102 runs per seed, about 45 units (each run is a third of a Stage C run).

## Order
1. Seed 42 (102 runs). Review the correlations.
2. Seed 43 (another 102 runs) only if seed 42 shows something worth confirming.

## What gets correlated
Attack score per client: fraction of rounds where its clean-shadow loss is lower under IN than
under its OUT (`per_client_score.py`; 0.5 = no signal).

Influence definitions, all free from the IN run and the partition:
- client size (`train_records`), label skew (distance from the pooled class mix)
- update size before clipping (mean, sum), fraction of rounds clipped
- distance from the average update, cosine with it, leave-one-out influence (mean, sum)
- aggregation weight, mean training loss

`analyze.py` prints Spearman/Pearson per definition, pooled and per setting; 96 (client, setting)
pairs per seed. Clients in the same IN model are not independent, so treat the pooled numbers as a ranking
of definitions, not as significance tests.

## Launch
```
uv run python -m results.cia_frontier.influence_correlation.runner --all-out   # seed 42
```
Shard with `--datasets`, `--privacy`, `--out-targets` / `--no-in`; one output subdir per shard
under `results/`. Plan without `--execute` first.

## Not covered
Fed-Influence (Xue 2021) and Hu's local-vs-others loss gap need extra estimators or every local
model; left out of this pass.

## Planned, not launched: seeds 43 and 44 (equal weights)
Same 6 settings, equal weights, 16 clients, all scored: 2 × 51 = 102 runs, ~60 units. Pooled with
seed 42 that gives 48 clients per setting, narrowing the ρ uncertainty from about ±0.5 to ±0.3.

Before launching, add one influence definition: **metric-privacy's client-model distance**, logged
for every privacy mode (vanilla and global-DP too, not only metric-privacy). Log the full pairwise
matrix each round, measured the way metric-privacy measures it (raw models before clipping, mean of
per-layer L2 distances, with client ids), then derive per client: mean distance to the others, max
distance to the others, and the share of rounds it is in the max pair (i.e. sets d). Today only the
metric-privacy runs log this matrix (`metric-dp-pairwise-*`); on seed 42 the per-client mean distance
tracked client size almost exactly (ρ +0.96 to +1.00), and the same two clients set d in most rounds.
