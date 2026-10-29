# B1 findings: client subsampling against full-participation global DP at matched accuracy

Package B1 of `research/cia_roadmap_v2/ROADMAP.md`, as amended by Section 10 (decisions 9, 10 and 13). Design in
[STATE.md](STATE.md), levels, order and gate in [PROTOCOL.md](PROTOCOL.md) (fixed before any leakage run). Numbers come
from [analysis.json](analysis.json); the command is in PROTOCOL.md Section 3.

## Verdict

**G-B1 passes for `q = 0.25` at both targets and fails for `q = 0.5`.**

| `q` | `a* = 0.740` | `a* = 0.660` | G-B1 |
|---|---|---|---|
| **0.25** | **-0.102 [-0.164, -0.040]**; two-way [-0.166, -0.040] | **-0.059 [-0.096, -0.022]**; two-way [-0.105, -0.011] | **pass at both** |
| 0.5 | -0.036 [-0.064, -0.008]; two-way [-0.075, 0.012] | -0.023 [-0.054, 0.009]; two-way [-0.066, 0.021] | fail at both (reduction below 0.05) |

All candidates; `AUC_B1(a*) - AUC_GDP(a*)` against M2's global-DP curve, pair jackknife at 97.5% (gate), two-way
bootstrap (95%) alongside. For `q = 0.25` the outcome is the same under both intervals; for `q = 0.5` at 0.740 the gate
interval excludes 0 (A1's label: "better") but the reduction is below the gate's 0.05 and the bootstrap includes 0.
The formal condition is reported, not gated (decision 13).

## Results

Randomized inclusion as M2 (32 candidates, 16 IN per replicate, 32 active of 48), EuroSAT alpha 0.3, 50 rounds, `R = 32`
on M2's design. Each round every active client joins with probability `q`; the release divides by `q N` and adds noise
`r C` to the average, as M2's global DP. Every run passed the run check (every sampled client trained in every round).

| Config | Amplified `mu` | AUC, all candidates (97.5%) | Top-decile AUC | TPR at 5% FPR | Mean final accuracy (95%) | Lowest client accuracy |
|---|---|---|---|---|---|---|
| q = 0.5, r = 0.001546 | not computable | 0.647 (0.600-0.695) | 0.741 | 0.121 | 0.768 (0.761-0.776) | 0.520 |
| q = 0.5, r = 0.003092 | not computable | 0.673 (0.617-0.730) | 0.750 | 0.111 | 0.730 (0.720-0.739) | 0.462 |
| q = 0.5, r = 0.006184 | not computable | 0.635 (0.584-0.686) | 0.717 | 0.145 | 0.640 (0.626-0.655) | 0.307 |
| q = 0.5, r = 0.012368 | 29.7 | 0.650 (0.591-0.710) | 0.737 | 0.146 | 0.519 (0.498-0.539) | 0.152 |
| q = 0.25, r = 0.0001 | not computable | 0.585 (0.540-0.630) | 0.615 | 0.066 | 0.746 (0.734-0.757) | 0.453 |
| q = 0.25, r = 0.000773 | not computable | 0.610 (0.552-0.667) | 0.662 | 0.107 | 0.736 (0.723-0.747) | 0.408 |
| q = 0.25, r = 0.003092 | not computable | 0.600 (0.549-0.651) | 0.622 | 0.105 | 0.698 (0.682-0.715) | 0.355 |
| q = 0.25, r = 0.006184 | not computable | 0.617 (0.567-0.668) | 0.646 | 0.115 | 0.602 (0.585-0.618) | 0.231 |
| Global DP 2x (M2) | 71.5 | 0.702 (0.643-0.761) | 0.746 | 0.174 | 0.740 | 0.512 |
| Global DP 4x (M2) | 35.7 | 0.666 (0.602-0.731) | 0.741 | 0.186 | 0.660 | 0.366 |

"Not computable": the noise multiplier `z = r q N` is below the accountant's supported minimum of 0.05 (its privacy loss
distribution does not finish there). Top decile, matched accuracy (reported, not gated): `q = 0.25` -0.103
[-0.196, -0.009] at 0.740 and -0.110 [-0.220, 0.001] at 0.660; `q = 0.5` +0.002 and -0.017.

**Replacement check** (automatic, stage-1 accuracy only): q = 0.5 0.762 / 0.724 / 0.633 / 0.515 and q = 0.25 0.741 /
0.730 / 0.685 / 0.588 bracket both targets; nothing replaced.

## Reading

- **Sampling a quarter of the clients per round cuts leakage by about 0.10 AUC at 0.740 accuracy and 0.06 at 0.660.**
  The reduction holds under both intervals, and it reaches the top decile too (-0.10 to -0.11, not gated). At `q = 0.5`
  it is smaller (-0.02 to -0.04) and below the gate.
- **The reduction comes from sampling, not from noise.** At `q = 0.25` the AUC is 0.585-0.617 across a 60-fold range of
  `r`, while accuracy falls from 0.746 to 0.602. A candidate influences only about a quarter of the rounds, so the label
  signal B5 identified reaches the model less often.
- **It costs peak accuracy.** With almost no noise (`r = 0.0001`) `q = 0.25` reaches 0.746, against 0.786-0.803 for
  global DP at low noise and vanilla. Matched-accuracy comparison is possible only up to about 0.745.
- **The formal guarantee does not follow.** At the same noise on the average, subsampling divides by `q N`, so its noise
  multiplier is `q` times smaller; its amplified `mu` is not computable at most levels and, where it is (q = 0.5,
  r = 0.012368: 29.7), it is larger than global DP's at the same `r` (17.9). The empirical improvement is not a
  stronger formal guarantee here.
- **Compared with B5:** logit adjustment gave -0.100 at 2x (accuracy +1.1 pp) by removing the label prior from local
  training; subsampling gives a similar -0.10 at matched accuracy by diluting it across rounds. They act on the same
  channel by different means, which makes their combination the natural candidate for C1.

## Limits

- **Thin bracketing at 0.740 for `q = 0.25`:** its most accurate level reached 0.746 (stage 1: 0.741). 280 of the 2,000
  bootstrap resamples did not bracket the target and are left out of that interval; the jackknife was unaffected.
- One dataset (EuroSAT), one layout, one heterogeneity level; the calibrated loss attacker only.
- The sample is assumed secret from the attacker (amplification's threat model); a server that reveals who joined each
  round would also reveal participation directly.
- Two levels per `q` were not in the pilot (q = 0.5, r = 0.012368; q = 0.25, r = 0.006184).

## Cost

| Step | Runs | GPU-h (occupancy) |
|---|---|---|
| Accuracy pilot and extension (GPU 1, 2026-10-26) | 18 | about 1 |
| Main runs (three workers on GPU 1, 2026-10-27 10:49-20:42) | 256 | about 9.9 |
| Total | 274 | about 10.9 of the 15 GPU-h cap |
