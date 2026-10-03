# Influence-directed noise pilot (Sep 24, 2026, Olcay)

## Question

Global-DP spreads its noise evenly over every direction of the model update, most of which no
client uses. Can putting a share *f* of the same noise on the directions clients actually push
the model protect them better, without costing more accuracy?

## Setup

Same as [`eurosat_frontier`](../eurosat_frontier/README.md): EuroSAT α = 0.3, 48 clients, 100
rounds, noise ratio 0.001546, targets 0–9, seed 42. Three arms with equal total noise energy:
f = 0 (ordinary isotropic noise, the control), f = 0.05 and f = 0.5. The mechanism is in
`metricdp_pytorch/influence_noise.py`; [`PROTOCOL.md`](PROTOCOL.md) has the details. No DP claim.

## Result

**No benefit.** f = 0.05 costs 0.8 points of accuracy against the control (85.2% vs 86.0%) and
doesn't lower the attack score (0.58 vs 0.55).

**Concentrating more noise breaks the model.** f = 0.5 loses 15 points of accuracy, so it was
stopped after 2 targets.

The directions clients push the model are also the directions it learns from, so noise there costs
accuracy directly.

## Caveats

- **The noise level was too low to test the idea.** At this level, plain global-DP doesn't lower
  the attack either, so there was little protection to improve on.
- **The arms were compared at equal noise energy,** not equal accuracy, which is the fairer test.
- **One seed.**
