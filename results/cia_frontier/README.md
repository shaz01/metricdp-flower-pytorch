# CIA frontier (Sep 22–24, 2026, Olcay)

Follows [`contest_at_scale`](../contest_at_scale/), which scored the attack on one client. Here it
is scored across ten clients, on a really label-skewed EuroSAT split (Dirichlet α = 0.3), without
folding. Not finished yet.

| Experiment | What it found |
|---|---|
| [`eurosat_frontier/`](eurosat_frontier/) | At the noise the old sweep "landed" on, neither defense leaks less than vanilla |
| [`influence_noise/`](influence_noise/) | Steering noise toward client-influence directions costs accuracy and doesn't lower the attack |

Why metric-privacy adds less noise than global-DP on CIFAR-10 but more on EuroSAT and Alzheimer:
[`metric_privacy_noise.md`](metric_privacy_noise.md).

## Caveats

- **Only one noise level was tested.** The upward noise sweep and the training-seed variance runs
  never ran.
