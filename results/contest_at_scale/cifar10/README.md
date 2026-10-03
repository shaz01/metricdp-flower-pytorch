# CIFAR-10 at 8–100 clients (Aug 11–14, 2026, Olcay)

## Question

The plan suite tested 4-class CIFAR-10 at up to 16 clients. Here the full 10-class CIFAR-10 goes up
to 100 clients.

Does the client inference attack get weaker as clients are added, and do global-DP and
metric-privacy help on top of that?

## Setup

10-class CIFAR-10, "non-IID" split, FedAvg, 20 rounds, seed 42. Client 0 is the target (IN/OUT
removal). The attack score is the share of the 20 rounds where the target's loss was lower with it
in training (0.5 = no signal). Done in three runs:

1. **Fixed noise multiplier 0.0182** at 8, 16, 48 and 100 clients (Aug 11).
2. **Ratio calibration** (Aug 12): measured how much noise each run actually got, then swept
   noise ratios for accuracy.
3. **Fixed noise ratios 0.0025, 0.004 and 0.00625** at 8, 48 and 100 clients (Aug 13–14).

<details>
<summary>Multiplier vs ratio</summary>

Global-DP's noise is `multiplier × clip / clients`, so a fixed multiplier adds less noise the more
clients there are. A fixed ratio (`multiplier / clients`, defined in
[`PLAN.md`](../plan_suite/PLAN.md)) keeps the noise the same at every client count.

</details>

## Result

**More clients hide the target; the defenses barely add to that.** At 8–48 clients the attack wins
in most rounds in every mode (vanilla 85–100%). At 100 clients it drops to near chance
(vanilla 0.55), again in every mode, so the drop comes from the crowd, not from the noise.

**A fixed multiplier made the noise do nothing.** In run 1, all three modes stayed within ~2
points of each other at every client count, and the attack scores matched vanilla.

**With a fixed ratio, the noise starts to matter, and metric-privacy keeps more accuracy.** At the
highest ratio, global-DP falls to 42–47% while metric-privacy holds 47–70%. But the defenses lower
the attack only where they also cost accuracy: global-DP brings it to 0.75 at 8 and 48 clients,
at 47% and 42% accuracy.

## Caveats

- **One seed, one target client.** A single client's score swings a lot, so treat these as
  directional.
- **The "non-IID" split isn't label-skewed.** It only varies client size; every client keeps the
  global class mix. See [`dirichlet_comparison`](../dirichlet_comparison/README.md).
- **Run 2's calibration sweep output was never committed;** only the noise diagnostics are.
- **The presentations in `reports/` fold the attack score** with max(AUC, 1−AUC), so their
  numbers read higher than the ones here.
