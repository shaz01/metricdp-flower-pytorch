# WP2 escalation: G2 inconclusive after the declared extension (2026-10-14)

Roadmap Section 3.2 and PROTOCOL.md Section 6: if the gate still straddles after its single extension, mark it
`inconclusive`, stop the package and escalate. This file is that escalation; no further runs have been started.

## What happened

- Seed 42 (`analysis.json`, 60 trajectories, 5 clusters): five of the six candidates fail G2 and `fixed-steps` is open
  on both parts. `consensus-kl-0.3` cuts the clean-view score by 0.148 [0.048, 0.224] (AUC by 0.170 [0.065, 0.238]) and
  beats both controls at matched accuracy, but its accuracy is 1.95 pp below vanilla (limit 1 pp), so it fails part 1.
- Extension (`analysis_seeds42-43.json`, seed 43 with the seed-42 layout for `vanilla`, `weight-decay-5e-4`,
  `local-epochs-1`, `fixed-steps`; 10 clusters, PROTOCOL.md Amendment 1): `fixed-steps` score 0.768 vs vanilla 0.812,
  reduction 0.044 [-0.034, 0.118] (AUC 0.041 [-0.027, 0.103]) at accuracy 0.820 vs 0.821. Part 1 is still open (the
  interval straddles 0.05; the point estimate is below it). Part 2 is open too: versus weight decay 0.042
  [-0.039, 0.120], versus one local epoch 0.044 [-0.034, 0.118].
- G2 = **inconclusive**. Cost: 21.1 h of trajectory wall time (15.5 h + 5.6 h) of the 30 t-h cap.

## Options

1. **Treat Stage A as not supporting D1 (recommended).** No candidate reached the bar: the only open one has a point
   reduction below 0.05 with an interval reaching 0, and the one large reduction (consensus KL beta 0.3) costs 2 pp
   accuracy. Write `FINDINGS.md` for WP2 recording the inconclusive gate and these numbers; D1 stops and this agent
   (Agent B role) moves to WP6 as the roadmap prescribes on a G2 fail. Owner's call, because the gate itself is not a
   fail.
2. **Carry consensus KL into WP4 by owner decision.** Its leakage cut is the largest and is significant, and the 1 pp
   accuracy condition is a screen without DP; WP4 would test it under the noise law at matched leakage, where its
   accuracy cost is what counts. This changes how the gate is applied, so only the owner can decide it.
3. **More seeds for `fixed-steps`.** Not allowed by the roadmap ("Never add further runs to push a result over a
   threshold") unless the owner explicitly asks.

Evidence: `analysis.json`, `analysis_seeds42-43.json`, `STATE.md` log, trajectories under `results/stage_a/`.

## After Amendment 2 (2026-10-15)

The owner chose option 3. Seeds 44 and 45 for `fixed-steps`, `vanilla` and both controls (48 trajectories, 13.39 h;
WP2 total 34.5 h, 15% over the cap) gave, over seeds 42-45 (`analysis_seeds42-45.json`, 20 clusters):

| Configuration | Clean score [95%] | AUC | Accuracy |
|---|---|---|---|
| vanilla | 0.762 [0.659, 0.854] | 0.822 | 0.828 |
| fixed-steps | 0.763 [0.704, 0.822] | 0.830 | 0.825 |
| weight-decay-5e-4 | 0.763 [0.666, 0.851] | 0.824 | 0.816 |
| local-epochs-1 | 0.848 [0.780, 0.911] | 0.917 | 0.685 |

`fixed-steps` reduction: -0.001 [-0.079, +0.076]; versus the controls at matched accuracy -0.001 and +0.001. The point
estimate fell from 0.024 (seed 42) and 0.044 (seeds 42-43) to zero, but the interval did not narrow (seeds 44-45 vary
a lot between targets), so by the rule both parts are still open: G2 stays `inconclusive`. As Amendment 2 fixed, no
further seeds without a new owner decision.

Remaining options: **1 (recommended now more strongly)**: record Stage A as not supporting D1 (no candidate reaches the
bar; the only open one shows no effect at four seeds), write `FINDINGS.md`, stop D1, Agent B role moves to WP6.
**2**: carry consensus KL beta 0.3 into WP4 by owner decision (its seed-42 cut of 0.148 [0.048, 0.224] at -1.95 pp
accuracy is the only clear leakage reduction; it was never re-run at other seeds).
