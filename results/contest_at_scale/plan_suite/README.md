# Plan suite (Aug 5–7, 2026, Olcay)

## Question

[`PLAN.md`](PLAN.md) asks four things about the paper's metric-privacy claims:

1. Can we reproduce the paper's accuracy (Alzheimer, 4 clients)?
2. Does its 3-client client inference attack hold?
3. Does the attack transfer to new datasets?
4. What happens as clients are added?

It also introduced the **noise ratio** (multiplier / clients), so noise stays comparable across
client counts.

## Setup

FedAvg, 20 rounds, seeds 42–44, vanilla / global-DP / metric-privacy, noise multiplier 0.01 unless
noted. The 3-client attack uses the paper's own client split (Table 9). New datasets:
Fashion-MNIST and 4-class CIFAR-10. Scaling: 4-class CIFAR-10 at 3, 8 and 16 clients, noise
ratios 0.0025, 0.0033 and 0.00625. The attack score is the share of rounds where the target's
loss was lower with it in training (0.5 = no signal).

## Result

1. **Yes, and our accuracy is higher.** FedAvg reaches 95.1 / 94.2 / 94.5% (vanilla / global-DP
   / metric-privacy) against the paper's 90.9 / 88.4 / 89.4%. The ordering is the same.
2. **The attack holds, but the paper's defense numbers don't.** On Alzheimer the attack wins
   92–100% of rounds in every mode, so neither defense stops it at multiplier 0.01. The paper
   reports first-round loss gaps of ~25% under both defenses; we get 3–4%.
3. **The attack transfers.** It wins 95–98% of rounds on Fashion-MNIST and 4-class CIFAR-10,
   with or without a defense.
4. **More clients weaken the attack only a little, and only with noise.** Vanilla stays at
   93–95% from 3 to 16 clients. The defenses pull it down only at 16 clients and the highest
   ratio (global-DP 77%, metric-privacy 83%), where global-DP also costs more accuracy (71% vs
   74%; vanilla 82%).

The noise calibration picked ratio 0.0025 as nearly free. At 0.00625, global-DP loses 12.5
points at 3 clients while metric-privacy loses none; at 48 clients both lose 14–15.

## Caveats

- **One target client.** Every attack score is for client 0 only.
- **The 48-client attack runs were planned but never run.** Scaling stops at 16 clients.
- **The PDF reports pool all rounds together** when scoring the scaling runs, which mixes early
  and late rounds. Their numbers differ from the ones here.
- **The noise calibration is one seed (42).**
