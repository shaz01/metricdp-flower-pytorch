# CIA roadmap v2, A1: metric privacy against global DP at matched accuracy

Package A1 of `research/cia_roadmap_v2/ROADMAP.md` (Agent A, branch `feature/cia-v2-frontier`, 2026-10-17 to
2026-10-20). The owner closed the package on 2026-10-20.

All numbers below come from committed artifacts in `results/cia_v2/frontier/a1/`:

- `STATE.md` (hypothesis, statistic, gate) and `PROTOCOL.md` (levels and run order, fixed before any leakage run).
- `FINDINGS.md`, and `analysis.json` from the 224 runs in `results/` and M2's global-DP runs in
  `results/cia_v2/frontier/results/`. PROTOCOL.md Section 3 has the command that regenerates it.
- `calibration/`: the 20 accuracy-calibration runs.
- `power.py`: the detectable difference, from M2's runs.

## 1. Question

v1 found metric privacy leaking less than global DP. M1 showed that v1's DP measurements shared noise between the
compared worlds, and M2 measured global DP's real leakage-accuracy curve. A1 asks the audit question: at the same
accuracy, does metric privacy leak less than global DP?

Two mechanisms:

- **The paper's metric privacy** (`metric-privacy`): noise calibrated by a pairwise client distance, with no formal
  guarantee.
- **WP3's private-median metric privacy** (`private-median`): the distance is replaced by a privately tracked median
  (`q_0 = 0.3` in `[0.03, 1.0]`, 32 public slots), with a GDP accountant.

Hypothesis (STATE.md): at matched accuracy metric privacy leaks no less than global DP; private median sits between
the two.

## 2. Protocol

**Federation and design.** M2's: randomized inclusion with 32 candidates (16 IN per replicate, 32 active of 48),
EuroSAT alpha 0.3, the seed-42 layout, 50 rounds, equal weights, WP2 hyperparameters, `R = 32`, `design_seed = 2`.
Replicate `j` has M2's inclusion and `(client_seed, noise_seed)` in every configuration (checked for replicates 0 and
17), so A1 pairs with M2's global-DP curve. Surrogates are not scored.

**Targets.** M2 found no useful regime, so the protocol's fallback applies: M2's two levels around 0.70 accuracy,
**0.740** (global DP 2x) and **0.660** (4x).

**Statistic.** Per mechanism, AUC against mean final accuracy, linear between its levels; global DP's curve is M2's
1x-16x. At each target `a*` the statistic is `AUC_mechanism(a*) - AUC_GDP(a*)`. A target outside a curve's accuracy
range is not extrapolated. Every resample draws the same pairs, candidates and replicates in all configurations and
recomputes both curves (`a1/analyze.py`, 12 synthetic tests).

**Intervals.** Gate: pair jackknife at 97.5%. Reported alongside: the two-way bootstrap (2,000 resamples).

**Gate G-A1** (ROADMAP.md, report, no kill), per mechanism and target on all candidates:

- better: the interval excludes 0 (below it);
- equivalent: the interval lies within ±0.03;
- worse, or inconclusive.

An interval inside ±0.10 that contains 0 is labelled "no detectable difference at this power". The top decile is
reported with the same labels but not gated.

**Power** (`a1/power.py`). Splitting M2's six configurations into two disjoint global-DP curves (vanilla, 2x, 8x
against 1x, 4x, 16x) gives a difference with A1's structure:

| Subset | `a* = 0.740` | `a* = 0.660` |
|---|---|---|
| All candidates | 0.027 (SE 0.008) | 0.037 (SE 0.011) |
| Top decile | 0.091 (SE 0.027) | 0.096 (SE 0.029) |

Detectable difference at 80% power for the 97.5% gate interval. Pairing across configurations makes differences far
more precise than M0's independent-configuration table (0.087), so the ±0.03 band is reachable for all candidates.

## 3. Choosing the levels: accuracy calibration

v1 had the paper's mechanism at one noise level only (100 rounds, 48 clients), so the levels came from an accuracy
calibration: both mechanisms at several multiples of 0.001546, 2 runs each, `design_seed = 3` (never analyzed), only
final accuracy read. 4x was added after the first 16 runs because neither mechanism had reached 0.660.

| Multiple | 0.25x | 0.5x | 1x | 2x | 4x |
|---|---|---|---|---|---|
| Metric privacy | 0.802 | 0.797 | 0.769 | 0.711 | 0.636 |
| Private median | 0.803 | 0.768 | 0.708 | 0.680 | 0.506 |
| Global DP (M2, `R = 32`) | | | 0.786 | 0.740 | 0.660 |

**Levels fixed in PROTOCOL.md:**

- Metric privacy: 1x, 2x, 4x.
- Private median: 0.5x, 1x, 2x, 4x. Four levels, because its two calibration runs differed by up to 0.07 at one level
  (0.744 and 0.673 at 1x).

**Replacement rule** (accuracy only): if a mechanism's stage-1 accuracies do not bracket both targets, its outermost
level on the wrong side is replaced once.

## 4. Runs and the replacement check

224 runs (7 configurations x 32), stage 1 then stage 2, on both GPUs with two workers each. Stage-1 mean accuracies:

| Mechanism | Levels | Stage-1 accuracy |
|---|---|---|
| Metric privacy | 1x, 2x, 4x | 0.753, 0.666, 0.597 |
| Private median | 0.5x, 1x, 2x, 4x | 0.748, 0.680, 0.616, 0.466 |

Both targets are bracketed by both mechanisms, so no level was replaced. Private median's fourth level was needed: at
1x it was already below 0.740.

## 5. Main result

| Config | `mu_total` | AUC, all candidates (97.5%) | Top-decile AUC | Final-round AUC | Mean final accuracy (95%) |
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

`mu_total`: global DP `sqrt(50) / (32 r)`; private median its accountant (update and quantile-release parts); the
paper's mechanism none.

**Matched-accuracy differences** (`AUC_mechanism(a*) - AUC_GDP(a*)`):

| Mechanism | Subset | `a*` | Mechanism AUC | Global-DP AUC | Difference | Gate interval (97.5%) | Two-way bootstrap (95%) | Outcome |
|---|---|---|---|---|---|---|---|---|
| Metric privacy | all | 0.740 | 0.702 | 0.702 | -0.000 | [-0.019, 0.018] | [-0.028, 0.031] | **equivalent** |
| Metric privacy | all | 0.660 | 0.681 | 0.666 | +0.015 | [-0.016, 0.045] | [-0.017, 0.046] | no detectable difference |
| Metric privacy | top decile | 0.740 | 0.730 | 0.746 | -0.016 | [-0.093, 0.062] | [-0.077, 0.062] | no detectable difference |
| Metric privacy | top decile | 0.660 | 0.750 | 0.741 | +0.009 | [-0.031, 0.049] | [-0.088, 0.085] | no detectable difference |
| Private median | all | 0.740 | 0.696 | 0.702 | -0.006 | [-0.042, 0.031] | [-0.046, 0.040] | no detectable difference |
| Private median | all | 0.660 | 0.656 | 0.666 | -0.011 | [-0.051, 0.029] | [-0.059, 0.039] | no detectable difference |
| Private median | top decile | 0.740 | 0.753 | 0.746 | +0.007 | [-0.090, 0.103] | [-0.125, 0.134] | worse or inconclusive |
| Private median | top decile | 0.660 | 0.716 | 0.741 | -0.026 | [-0.159, 0.108] | [-0.217, 0.101] | worse or inconclusive |

**G-A1 verdict: at matched accuracy, neither metric privacy mechanism leaks detectably less than global DP.**

- The paper's mechanism is **equivalent** to global DP at 0.740 under the gate interval. Under the two-way bootstrap
  its interval ends at 0.031, just outside ±0.03, so the outcome there is "no detectable difference".
- Every other gated outcome is "no detectable difference at this power", the same under both intervals.
- No interval lies below 0: "better" is ruled out at this power for both mechanisms at both targets.

## 6. Reading

- **At matched accuracy the three mechanisms leak about the same.** At 0.740 accuracy all three are at an AUC of
  about 0.70; at 0.660, between 0.656 and 0.681.
- **v1's advantage for metric privacy does not survive** independent randomness and comparison at equal accuracy.
- **At equal noise multiple the mechanisms differ, at equal accuracy they do not.** Metric privacy adds somewhat more
  noise than global DP at the same `r` (accuracy 0.752 against 0.786 at 1x), private median more still (0.683). Their
  leakage follows their accuracy.
- **The formal guarantee does not follow the leakage.** Near 0.75 accuracy private median's `mu_total` is 301, against
  71-143 for global DP at neighbouring accuracies; the paper's mechanism has none. The empirical audit cannot tell
  them apart while the guarantees differ by a factor of 2-4.
- **The largest clients stay exposed** under every mechanism: top-decile AUCs of 0.68-0.76 wherever accuracy is above
  0.6.

## 7. Anomalies and corrections

- **The replacement rule was applied late.** It was due after stage 1, but the chain went straight on to stage 2
  overnight. It was applied the next morning, on stage-1 accuracy only and before any leakage statistic; no
  replacement was needed.
- **Unbalanced GPU split.** GPU 0 had 128 runs and GPU 1 had 96. A helper moved GPU 1 onto GPU 0's remaining runs
  when its own chain finished (06:41), so neither GPU sat idle.
- **Cap revised down.** STATE.md first proposed a 48 GPU-h cap at the M0 cost run's figure. With M2's measured cost
  the plan fit ROADMAP.md's 30 GPU-h, and the proposal was withdrawn before the protocol.

## 8. Caveats

- One dataset, one layout seed, one heterogeneity level; the calibrated loss attacker only.
- **Thin bracketing margins.** The stage-1 accuracies closest to a target were 0.666 (metric privacy 2x, target 0.660)
  and 0.748 (private median 0.5x, target 0.740). 4 and 38 of the 2,000 bootstrap resamples at 0.740 did not bracket
  the target and are left out of that interval.
- **Equivalence** holds only for the paper's mechanism at 0.740, and only under the gate interval.
- **Linear interpolation** in accuracy between levels up to 0.08 apart; M2's curve is close to linear in that range.
- The attacker is not robust to diverged trajectories (found in A2). Below about 0.6 accuracy that could understate
  leakage; both targets are above it.

## 9. Cost

| Step | Runs | GPU-h |
|---|---|---|
| Accuracy calibration (`design_seed = 3`, both GPUs, 2026-10-19 20:31-21:32) | 20 | about 2.0 |
| Main runs (both GPUs, two workers each, 2026-10-19 21:34 to 2026-10-20 08:26) | 224 | about 21.7 |
| Total | 244 | about 23.7 of the 30 GPU-h cap |

## 10. Conclusions

1. **G-A1:** at matched accuracy neither the paper's metric privacy nor private-median metric privacy leaks
   detectably less than global DP. The paper's mechanism is equivalent at 0.740 accuracy (gate interval
   [-0.019, 0.018]).
2. **The audit claim** (ROADMAP.md Section 4: "the outcome decides the audit claim"): metric privacy's apparent
   advantage was a measurement artifact; at equal accuracy it offers the same empirical protection as global DP,
   with no formal guarantee for the paper's version and a 2-4 times weaker one for private median.
3. **Closed by the owner** on 2026-10-20 together with M0, M1, M2 and A2.
