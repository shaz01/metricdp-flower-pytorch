# CIFAR-100 at 100 clients (Aug 9–13, 2026, Ata)

## Question

On a much harder dataset (100 classes) at 100 clients, what do global-DP and metric-privacy cost
in accuracy, and do they reduce what the client inference attack can learn?

## Setup

CIFAR-100, 100 clients, 250 rounds, FedAvg. Homogeneous and "non-IID" splits × vanilla /
global-DP / metric-privacy. One fixed noise multiplier, 0.0182, for every mode. Attack: client 0
in vs out, seed 42, scored at 26 checkpoints (round 1 and every 10th). The score is the share of
checkpoints where the target's loss was lower with it in training (0.5 = no signal).

## Result

**Both defenses cost the same 7–8 points.** Vanilla reaches 28–29% and both defenses 21–22%, in
either split. Metric-privacy keeps no accuracy edge here.

**The attack results can't answer the second half.** On the homogeneous split the defenses lower
the score (vanilla 0.85, global-DP 0.73, metric-privacy 0.62). On the "non-IID" split it's the
other way round (vanilla 0.50, defenses 0.65–0.69). The two splits have the same class mix, so
that flip is just how much a single client's score swings.

## Caveats

- **One seed, one target client.** Seeds 43–44 were planned but never committed.
- **The "non-IID" split isn't label-skewed;** it only varies client size. See
  [`dirichlet_comparison`](../dirichlet_comparison/README.md).
- **The noise multiplier was set from an early-round update norm,** which the script's docstring
  itself flags as unresolved. It is also fixed rather than a ratio, so it isn't comparable with
  the ratio runs in [`cifar10`](../cifar10/README.md).
