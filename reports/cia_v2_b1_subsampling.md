# CIA roadmap v2, B1: client subsampling against global DP at matched accuracy

Package B1 of `research/cia_roadmap_v2/ROADMAP.md`, as amended by Section 10 (decisions 9, 10 and 13). Branch
`feature/cia-v2-mechanisms`. The owner assigned Agent B's packages to the agent that ran Agent A's, and closed B1 on
2026-10-29, opening C1 the same day.

All numbers below come from committed artifacts in `results/cia_v2/mechanisms/b1_subsampling/`: `STATE.md` (design,
log), `PROTOCOL.md` (fixed 2026-10-27, before any leakage run), `FINDINGS.md`, `analysis.json` from the 256 runs in
`results/` paired with M2's global-DP runs, and the accuracy pilot in `pilot/` (never analyzed for leakage).

## 1. Question

Sampling each client into a round with probability `q` amplifies a DP guarantee and lets a candidate influence only
about `q` of the rounds. At matched accuracy, does client subsampling lower the attacker's AUC below full-participation
global DP?

## 2. Mechanism and design

- **`PoissonSampledGlobalDP`** (`metricdp_pytorch/subsampled_dp_strategy.py`). Each round every active client joins
  independently with probability `q`, drawn from a generator seeded by the run's noise seed (the sample is private).
  The release is `w + sum_{j in S} clip(u_j, C) / (q N) + N(0, (r C)^2 I)` with the public denominator `q N`
  (`N = 32`), so neither scaling nor noise depends on who joined; a round with no sampled client releases noise only.
  The noise on the released average is `r C`, as M2's global DP; the noise multiplier is `z = r q N`.
- **Regime:** M2's (EuroSAT alpha 0.3, randomized inclusion with 32 candidates and 32 active of 48, 50 rounds, equal
  weights, `R = 32`, `design_seed = 2`).
- **Comparison: A1's matched-accuracy statistic.** Per `q`, AUC against mean final accuracy, linear between its levels,
  against M2's global-DP curve at the targets **0.740 and 0.660** (M2's accuracy at 2x and 4x).
- **Levels** from an accuracy pilot (18 runs, `design_seed = 3`): `q = 0.5` at `r` = 0.001546, 0.003092, 0.006184,
  0.012368; `q = 0.25` at 0.0001, 0.000773, 0.003092, 0.006184. An automatic accuracy-only replacement check after
  stage 1 found both targets bracketed.
- **G-B1:** all-candidate AUC at least 0.05 below global DP at matched accuracy, gate interval (pair jackknife at 97.5%)
  below 0. The amplified formal `mu` is reported, not gated (decision 13).
- **Safety:** a run stops at the first round with a failed or missing sampled client (`--require-all-clients`), and the
  run check refuses short rounds; every run passed.

## 3. Results

| `q` | `a* = 0.740` | `a* = 0.660` | G-B1 |
|---|---|---|---|
| **0.25** | **-0.102 [-0.164, -0.040]**; two-way [-0.166, -0.040] | **-0.059 [-0.096, -0.022]**; two-way [-0.105, -0.011] | **pass at both** |
| 0.5 | -0.036 [-0.064, -0.008]; two-way [-0.075, 0.012] | -0.023 [-0.054, 0.009]; two-way [-0.066, 0.021] | fail at both |

| Config | Amplified `mu` | AUC, all candidates (97.5%) | Top-decile AUC | TPR at 5% FPR | Mean final accuracy (95%) |
|---|---|---|---|---|---|
| q = 0.5, r = 0.001546 | not computable | 0.647 (0.600-0.695) | 0.741 | 0.121 | 0.768 (0.761-0.776) |
| q = 0.5, r = 0.003092 | not computable | 0.673 (0.617-0.730) | 0.750 | 0.111 | 0.730 (0.720-0.739) |
| q = 0.5, r = 0.006184 | not computable | 0.635 (0.584-0.686) | 0.717 | 0.145 | 0.640 (0.626-0.655) |
| q = 0.5, r = 0.012368 | 29.7 | 0.650 (0.591-0.710) | 0.737 | 0.146 | 0.519 (0.498-0.539) |
| q = 0.25, r = 0.0001 | not computable | 0.585 (0.540-0.630) | 0.615 | 0.066 | 0.746 (0.734-0.757) |
| q = 0.25, r = 0.000773 | not computable | 0.610 (0.552-0.667) | 0.662 | 0.107 | 0.736 (0.723-0.747) |
| q = 0.25, r = 0.003092 | not computable | 0.600 (0.549-0.651) | 0.622 | 0.105 | 0.698 (0.682-0.715) |
| q = 0.25, r = 0.006184 | not computable | 0.617 (0.567-0.668) | 0.646 | 0.115 | 0.602 (0.585-0.618) |
| Global DP 2x (M2) | 71.5 | 0.702 (0.643-0.761) | 0.746 | 0.174 | 0.740 |
| Global DP 4x (M2) | 35.7 | 0.666 (0.602-0.731) | 0.741 | 0.186 | 0.660 |

Top decile at matched accuracy (reported, not gated): `q = 0.25` -0.103 [-0.196, -0.009] at 0.740 and -0.110
[-0.220, 0.001] at 0.660; `q = 0.5` +0.002 and -0.017.

## 4. Reading

- **Sampling a quarter of the clients per round cuts leakage by about 0.10 AUC at 0.740 accuracy and 0.06 at 0.660**,
  under both intervals, and the top decile gains too.
- **The reduction comes from sampling, not noise:** at `q = 0.25` the AUC stays at 0.585-0.617 over a 60-fold range of
  `r`. A candidate reaches the model in about a quarter of the rounds, so its label signal (B5's finding) arrives less
  often.
- **It caps accuracy:** with almost no noise `q = 0.25` reaches 0.746, against about 0.80 at full participation.
- **The formal guarantee is not better:** at the same noise on the average the noise multiplier is `q` times smaller;
  the amplified `mu` is below the accountant's range at most levels and, where it computes (29.7), exceeds global DP's
  at the same `r` (17.9).
- **Same channel as B5:** logit adjustment removes the label prior from local training (-0.100 at 2x); subsampling
  dilutes it across rounds (-0.10 at matched accuracy). Their combination is C1's natural candidate.

## 5. Anomalies and corrections

- **Accountant limits.** At `z` far below the accountant's supported minimum (0.05) its privacy loss distribution did not
  finish (the first analysis timed out after 50 minutes); such levels are reported as not computable. A reporting fix
  makes the formal condition read "not computable" rather than false.
- **Run check and collapsed rounds.** The check introduced after the Alzheimer out-of-memory incident first rejected
  collapsed rounds; both fixes (sampled clients, collapsed rounds) were merged before B1's runs.
- **Branch hygiene.** `feature/cia-v2-alzheimer` was merged into this branch twice to bring the safety fixes in.

## 6. Caveats

- Thin bracketing at 0.740 for `q = 0.25` (best level 0.746): 280 of 2,000 bootstrap resamples did not bracket the
  target and are left out of that interval.
- One dataset, one layout; the calibrated loss attacker only, no adaptive attacker.
- Amplification assumes the per-round sample stays secret.

## 7. Cost

About 10.9 GPU-h of the 15 GPU-h cap (pilot about 1, main runs about 9.9; three workers on GPU 1, 2026-10-27
10:49-20:42).

## 8. Conclusions and owner decisions (2026-10-29)

1. **G-B1 passes for `q = 0.25`** at 0.740 and 0.660 (about -0.10 and -0.06 AUC); `q = 0.5` fails.
2. **Closed by the owner;** the branch (B5 and B1) merges into master.
3. **C1 opened** by the owner the same day: the composite of the passed mechanisms against metric privacy.
