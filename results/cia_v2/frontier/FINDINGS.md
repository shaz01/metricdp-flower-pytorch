# M2 findings: corrected leakage-accuracy curves for global DP

Package M2 of [research/cia_roadmap_v2/ROADMAP.md](../../../research/cia_roadmap_v2/ROADMAP.md). Design, statistic
and gate are in [STATE.md](STATE.md), the fixed run order in [PROTOCOL.md](PROTOCOL.md). All numbers come from
[analysis.json](analysis.json), which this command regenerates from the committed runs in `results/`:

```
python -m results.cia_v2.frontier.analyze --configs vanilla=0 gdp-r1x=0.001546 gdp-r2x=0.003092 \
    gdp-r4x=0.006184 gdp-r8x=0.012368 gdp-r16x=0.024736 --out results/cia_v2/frontier/analysis.json
```

## Verdict

**G-M2: no useful regime.** No noise level meets the three conditions together. The all-candidate AUC's upper
gate-interval end stays above 0.6 at every level, down to 16x, where mean accuracy is 0.199.

| Level | All-candidate AUC upper end (condition: 0.6 or below) | Top-decile AUC (condition: 0.6 or below) | Accuracy lower end (condition: 0.70 or above) |
|---|---|---|---|
| 1x | 0.778, fails | 0.741, fails | 0.780, meets |
| 2x | 0.761, fails | 0.746, fails | 0.731, meets |
| 4x | 0.731, fails | 0.741, fails | 0.646, fails |
| 8x | 0.721, fails | 0.724, fails | 0.524, fails |
| 16x | 0.642, fails | 0.603, fails | 0.179, fails |

Consequence (ROADMAP.md Section 4): "Escalate with the recommendation that B4 lead; A1 still runs (the audit claim
needs it)". B4 has since closed with a G-B4 fail, so the recommendation it names is no longer available as written.
The escalation is in [ESCALATION.md](ESCALATION.md).

## Results

Randomized inclusion, EuroSAT alpha 0.3, 32 candidates (16 IN per replicate), 50 rounds, `R = 32` replicates per
configuration (`design_seed = 2`), 192 runs. Calibrated loss attacker on the 11-checkpoint shadow-loss trajectory.
Intervals: pair jackknife at 97.5% (gate) and the two-way bootstrap (reported alongside); accuracy with a 95%
bootstrap over replicates. Formal `mu_total = sqrt(50) / (32 r)`.

| Config | `mu_total` | AUC, all candidates (97.5%) | Two-way bootstrap | Top-decile AUC (97.5%) | TPR at 5% FPR | Final-round AUC | Mean final accuracy (95%) | Lowest client accuracy |
|---|---|---|---|---|---|---|---|---|
| Vanilla | none | 0.723 (0.674-0.773) | 0.653-0.787 | 0.738 (0.600-0.875) | 0.234 | 0.656 | 0.803 (0.799-0.808) | 0.589 |
| 1x (0.001546) | 142.9 | 0.720 (0.663-0.778) | 0.657-0.781 | 0.741 (0.629-0.853) | 0.170 | 0.650 | 0.786 (0.780-0.792) | 0.573 |
| 2x (0.003092) | 71.5 | 0.702 (0.643-0.761) | 0.626-0.768 | 0.746 (0.643-0.849) | 0.174 | 0.611 | 0.740 (0.731-0.750) | 0.512 |
| 4x (0.006184) | 35.7 | 0.666 (0.602-0.731) | 0.595-0.732 | 0.741 (0.650-0.832) | 0.186 | 0.607 | 0.660 (0.646-0.673) | 0.366 |
| 8x (0.012368) | 17.9 | 0.671 (0.622-0.721) | 0.603-0.735 | 0.724 (0.593-0.854) | 0.193 | 0.577 | 0.543 (0.524-0.561) | 0.177 |
| 16x (0.024736) | 8.9 | 0.593 (0.545-0.642) | 0.522-0.659 | 0.603 (0.496-0.710) | 0.094 | 0.553 | 0.199 (0.179-0.220) | 0.003 |

`analysis.json` also holds TPR at 1% FPR, the uncalibrated threshold attacker and the surrogate-attacker AUC (B5's
input) for both candidate sets.

## Reading

- **Leakage barely moves until the model breaks.** From vanilla to 2x the all-candidate AUC goes from 0.723 to 0.702
  while accuracy falls by 0.06. It reaches 0.593 only at 16x, where accuracy is 0.199 on a 10-class task.
- **With independent randomness, global DP's leakage does fall with noise, but slowly.** The decline from vanilla to
  16x is 0.13 AUC. Each step from 1x to 8x is within the 0.087 detectable difference at this design's power, so
  adjacent levels are not distinguishable from each other.
- **The largest clients stay exposed.** The top-decile AUC is 0.72-0.75 from vanilla to 8x. Only at 16x does it
  reach 0.603.
- **The formal guarantee is vacuous wherever accuracy is usable.** `mu_total` is 71-143 at the levels with accuracy
  of 0.70 or above. It is 8.9 only at 16x.
- **Limits.**
  - One dataset, one layout seed, one heterogeneity level.
  - The audit regime conditions on the 32-candidate panel; the two-way bootstrap, which adds the panel's variation,
    gives somewhat wider intervals with the same verdict at every level.
  - The realized-noise-energy correlation across configurations (STATE.md, "Shared-noise check") is not computed in
    `analyze.py`. The pairing comes from equal `(client_seed, noise_seed)` per replicate by construction
    (`design_seed = 2`), and no statistic here compares configurations.

## Operating points for A1

The protocol's rule for "no useful regime" (A1 STATE.md, Noise levels): the two noise levels around 0.70 accuracy.
They are **2x (accuracy 0.740) and 4x (0.660)**.

## Cost

| Stage | Runs | GPU-h |
|---|---|---|
| Stage 1 (vanilla, 1x, 4x, 16x; two workers on GPU 1) | 64 | about 6.9 |
| Stage 2 and the 2x and 8x fill-ins (two workers on GPU 0, 2026-10-18 12:46 to 2026-10-19 00:51) | 128 | about 12.1 |
| Total | 192 | about 19.0 of the 64 GPU-h cap |
