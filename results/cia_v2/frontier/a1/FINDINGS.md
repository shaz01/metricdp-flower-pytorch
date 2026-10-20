# A1 findings: metric privacy against global DP at matched accuracy

Package A1 of [research/cia_roadmap_v2/ROADMAP.md](../../../../research/cia_roadmap_v2/ROADMAP.md). Design, statistic
and gate are in [STATE.md](STATE.md), levels and run order in [PROTOCOL.md](PROTOCOL.md). All numbers come from
[analysis.json](analysis.json), which PROTOCOL.md Section 3's command regenerates from the committed runs in
`results/` and M2's global-DP runs in `../results/`.

## Verdict

**G-A1: at matched accuracy, neither metric privacy mechanism leaks detectably less than global DP.**

| Mechanism | `a* = 0.740` | `a* = 0.660` |
|---|---|---|
| Paper's metric privacy | **equivalent**: -0.000, gate interval [-0.019, 0.018] | no detectable difference at this power: +0.015 [-0.016, 0.045] |
| Private-median metric privacy | no detectable difference at this power: -0.006 [-0.042, 0.031] | no detectable difference at this power: -0.011 [-0.051, 0.029] |

All candidates; difference `AUC_mechanism(a*) - AUC_GDP(a*)`, pair jackknife at 97.5% (the gate interval).

- **Under the two-way bootstrap** (reported alongside), the paper's mechanism at 0.740 is [-0.028, 0.031]. That is
  just outside ±0.03, so under that interval the outcome is "no detectable difference at this power", not
  "equivalent". All other outcomes are the same under both intervals.
- **No interval excludes 0**, so "better" is ruled out at both targets for both mechanisms, at this power.

Consequence (ROADMAP.md Section 4): "Recorded; the outcome decides the audit claim."

## Results

`R = 32` per configuration on M2's design (`design_seed = 2`), so replicate `j` has the same inclusion and seeds in
every configuration. Calibrated loss attacker; AUC with the 97.5% pair-jackknife interval; accuracy with a 95%
bootstrap over replicates. Formal `mu_total`: global DP `sqrt(50) / (32 r)`, private median its accountant (update
and quantile-release parts), the paper's mechanism none.

| Config | `mu_total` | AUC, all candidates | Top-decile AUC | Final-round AUC | Mean final accuracy |
|---|---|---|---|---|---|
| Global DP 1x (M2) | 142.9 | 0.720 (0.663-0.778) | 0.741 | 0.650 | 0.786 (0.780-0.792) |
| Global DP 2x (M2) | 71.5 | 0.702 (0.643-0.761) | 0.746 | 0.611 | 0.740 (0.731-0.750) |
| Global DP 4x (M2) | 35.7 | 0.666 (0.602-0.731) | 0.741 | 0.607 | 0.660 (0.646-0.673) |
| Global DP 8x (M2) | 17.9 | 0.671 (0.622-0.721) | 0.724 | 0.577 | 0.543 (0.524-0.561) |
| Global DP 16x (M2) | 8.9 | 0.593 (0.545-0.642) | 0.603 | 0.553 | 0.199 (0.179-0.220) |
| Metric privacy 1x | none | 0.706 (0.652-0.760) | 0.727 | 0.626 | 0.752 (0.744-0.760) |
| Metric privacy 2x | none | 0.679 (0.625-0.732) | 0.749 | 0.620 | 0.678 (0.667-0.688) |
| Metric privacy 4x | none | 0.687 (0.637-0.738) | 0.753 | 0.612 | 0.609 (0.594-0.625) |
| Private median 0.5x | 301.3 | 0.702 (0.649-0.755) | 0.756 | 0.622 | 0.749 (0.740-0.758) |
| Private median 1x | 150.7 | 0.661 (0.601-0.720) | 0.733 | 0.605 | 0.683 (0.671-0.695) |
| Private median 2x | 75.3 | 0.646 (0.596-0.697) | 0.683 | 0.567 | 0.617 (0.600-0.635) |
| Private median 4x | 37.7 | 0.612 (0.564-0.660) | 0.635 | 0.576 | 0.466 (0.441-0.490) |

Matched-accuracy differences, both subsets:

| Mechanism | Subset | `a*` | Mechanism AUC | Global-DP AUC | Difference | Gate interval (97.5%) | Two-way bootstrap (95%) | Outcome |
|---|---|---|---|---|---|---|---|---|
| Metric privacy | all | 0.740 | 0.702 | 0.702 | -0.000 | [-0.019, 0.018] | [-0.028, 0.031] | equivalent |
| Metric privacy | all | 0.660 | 0.681 | 0.666 | +0.015 | [-0.016, 0.045] | [-0.017, 0.046] | no detectable difference |
| Metric privacy | top decile | 0.740 | 0.730 | 0.746 | -0.016 | [-0.093, 0.062] | [-0.077, 0.062] | no detectable difference |
| Metric privacy | top decile | 0.660 | 0.750 | 0.741 | +0.009 | [-0.031, 0.049] | [-0.088, 0.085] | no detectable difference |
| Private median | all | 0.740 | 0.696 | 0.702 | -0.006 | [-0.042, 0.031] | [-0.046, 0.040] | no detectable difference |
| Private median | all | 0.660 | 0.656 | 0.666 | -0.011 | [-0.051, 0.029] | [-0.059, 0.039] | no detectable difference |
| Private median | top decile | 0.740 | 0.753 | 0.746 | +0.007 | [-0.090, 0.103] | [-0.125, 0.134] | worse or inconclusive |
| Private median | top decile | 0.660 | 0.716 | 0.741 | -0.026 | [-0.159, 0.108] | [-0.217, 0.101] | worse or inconclusive |

The top decile is reported, not gated.

## Reading

- **At matched accuracy the three mechanisms leak about the same.** At 0.740 accuracy all three sit at an AUC of
  about 0.70, and at 0.660 between 0.656 and 0.681. The paper's mechanism is within ±0.02 of global DP at 0.740.
- **v1's advantage for metric privacy does not survive.** It rested on paired, shared-noise measurements (M1). With
  independent randomness and comparison at equal accuracy, metric privacy is no better than global DP.
- **The formal guarantee does not follow the leakage.** At about 0.75 accuracy private median's `mu_total` is 301,
  against 71-143 for global DP at neighbouring accuracies; the paper's mechanism has none. Their measured leakage
  is the same, so the empirical audit cannot tell them apart while the guarantees differ by a factor of 2-4.
- **The largest clients stay exposed under every mechanism.** Top-decile AUCs are 0.68-0.76 at every level with
  accuracy above 0.6.

## Limits

- One dataset, one layout seed, one heterogeneity level; the calibrated loss attacker only.
- The bracketing margins are thin. The stage-1 accuracies closest to a target were 0.666 (metric privacy 2x, target
  0.660) and 0.748 (private median 0.5x, target 0.740). The replacement rule did not trigger, but 4 and 38 of the
  2,000 bootstrap resamples at `a* = 0.740` did not bracket the target and are left out of that interval.
- The replacement rule was applied after stage 2 had already run, because the chain went straight on. It used
  stage-1 accuracy only, as written, and no leakage statistic had been computed. No replacement was needed.
- Equivalence holds only for the paper's mechanism at 0.740, and only under the gate interval.

## Cost

| Step | Runs | GPU-h |
|---|---|---|
| Accuracy calibration (`design_seed = 3`, both GPUs, 2026-10-19 20:31-21:32) | 20 | about 2.0 |
| Main runs (both GPUs, two workers each, 2026-10-19 21:34 to 2026-10-20 08:26) | 224 | about 21.7 |
| Total | 244 | about 23.7 of the 30 GPU-h cap |
