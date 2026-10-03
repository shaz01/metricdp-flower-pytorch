# EuroSAT at 48 clients (Aug 12–13, 2026, Ata)

## Question

On an easier dataset from a different domain (10-class satellite images) at 48 clients, what do
global-DP and metric-privacy cost in accuracy, and do they reduce what the client inference attack
can learn?

## Setup

EuroSAT, 48 clients, 100 rounds, FedAvg. Homogeneous and "non-IID" splits × vanilla / global-DP /
metric-privacy. One fixed noise multiplier, 0.0371. Attack: client 0 in vs out, seeds 42–44,
scored at 11 checkpoints per seed (33 pairs). The score is the share of pairs where the target's
loss was lower with it in training (0.5 = no signal).

Full write-ups: [`eurosat_accuracy_sweep.md`](reports/eurosat_accuracy_sweep.md),
[`eurosat_cia.md`](reports/eurosat_cia.md).

## Result

**The defenses are almost free.** They cost 0.4–2.4 points against ~90% vanilla accuracy.

**But the attack is weak even without them, so there's little to reduce.** Vanilla scores 0.73 on
the homogeneous split and 0.55 on the "non-IID" one. Both defenses bring the homogeneous score
to 0.61; on "non-IID" they land at 0.45–0.61, around vanilla. None of this separates from chance.

The later per-client rescoring on a real label-skewed EuroSAT split found the same: at twice this
noise level, neither defense scores below vanilla. See
[`cia_frontier/eurosat_frontier`](../../cia_frontier/eurosat_frontier/README.md).

## Caveats

- **One target client.** Every 95% interval in `cia_analysis.json` includes 0.5.
- **The "non-IID" split isn't label-skewed;** it only varies client size. So "non-IID beats
  homogeneous", which the reports call a finding, compares two near-identical splits. See
  [`dirichlet_comparison`](../dirichlet_comparison/README.md).
- **The noise multiplier is fixed,** not a ratio.
