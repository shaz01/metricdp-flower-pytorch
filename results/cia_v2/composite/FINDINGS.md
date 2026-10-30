# C1 findings: the composite (subsampling + logit adjustment) against metric privacy at matched accuracy

Package C1 of `research/cia_roadmap_v2/ROADMAP.md`, as amended by Section 10 (decisions 9, 13, 14, 15 and 16). Design
in [STATE.md](STATE.md), levels, order and gate in [PROTOCOL.md](PROTOCOL.md) (fixed before any leakage run). Numbers
come from [analysis.json](analysis.json); the command is in PROTOCOL.md Section 3.

## Verdict

**G-C1 passes at both targets, under the gate interval and under the two-way bootstrap.**

| `a*` | Composite AUC | Metric privacy AUC | Difference (pair jackknife, 97.5%) | Two-way bootstrap (95%) | G-C1 |
|---|---|---|---|---|---|
| 0.740 | 0.606 | 0.702 | **-0.095 [-0.137, -0.053]** | [-0.154, -0.036] | **pass** |
| 0.660 | 0.573 | 0.681 | **-0.108 [-0.163, -0.053]** | [-0.183, -0.035] | **pass** |

All candidates; `AUC_composite(a*) - AUC_MP(a*)` against A1's metric-privacy curve on the same design. The formal clause
is reported, not gated (decision 15): the amplified `mu` is not computable at the three lower levels (`z` below the
accountant's minimum, as B1) and is 37.1 at `r = 0.012368`.

**But the gain over subsampling alone is not detectable.** At matched accuracy the composite's AUC matches B1's
`q = 0.25` curve (+0.006 [-0.048, 0.061] at 0.740, -0.034 [-0.084, 0.016] at 0.660). Logit adjustment buys accuracy at a
given noise (3-10 pp), but at matched accuracy it does not lower leakage further at this power.

## Results

Randomized inclusion as M2 (32 candidates, 16 IN per replicate, 32 active of 48), EuroSAT alpha 0.3, 50 rounds, `R = 32`
on M2's design (`design_seed = 2`). The composite is B1's Poisson-sampled global DP at `q = 0.25` with B5's logit-adjusted
local loss (`tau = 1`). All 128 runs passed the run check (every sampled client trained in every round); no
out-of-memory failure and no run stopped by `--require-all-clients`. Stage 1's accuracies bracketed both targets, so the
replacement check passed with no replacement.

### Per level (all candidates, gated attacker unless noted)

| `r` | Accuracy | Lowest client acc. | AUC [95%] | TPR@5% | TPR@1% | Top-decile AUC | Surrogate AUC (a) | Defense-aware AUC (b) | Amplified `mu` |
|---|---|---|---|---|---|---|---|---|---|
| 0.000773 | 0.772 | 0.497 | 0.592 [0.538, 0.646] | 0.072 | 0.020 | 0.659 | 0.563 | 0.619 | not computable |
| 0.003092 | 0.723 | 0.415 | 0.614 [0.570, 0.658] | 0.088 | 0.008 | 0.691 | 0.599 | 0.598 | not computable |
| 0.006184 | 0.661 | 0.327 | 0.574 [0.520, 0.627] | 0.127 | 0.053 | 0.696 | 0.587 | 0.588 | not computable |
| 0.012368 | 0.559 | 0.200 | 0.537 [0.505, 0.569] | 0.092 | 0.016 | 0.619 | 0.532 | 0.530 | 37.1 |

The AUC is not monotone in the noise between the two lowest levels (0.592 then 0.614); the intervals overlap fully.

### Matched accuracy, reported comparisons (all candidates)

| Composite against | `a*` = 0.740 | `a*` = 0.660 |
|---|---|---|
| Private median (A1) | -0.090 [-0.139, -0.041]; two-way [-0.149, -0.025] | -0.083 [-0.131, -0.034]; two-way [-0.150, -0.018] |
| Global DP (M2) | -0.096 [-0.148, -0.043]; two-way [-0.159, -0.027] | -0.093 [-0.157, -0.030]; two-way [-0.170, -0.018] |
| Subsampling alone, `q = 0.25` (B1) | +0.006 [-0.048, 0.061]; two-way [-0.061, 0.078] | -0.034 [-0.084, 0.016]; two-way [-0.109, 0.042] |

Top decile against metric privacy: -0.050 [-0.138, 0.037] at 0.740 and -0.055 [-0.141, 0.030] at 0.660, so no detectable
difference there (the top decile has a tenth of the pairs).

### Adaptive attackers (reported)

| Attacker | `a*` = 0.740 | `a*` = 0.660 |
|---|---|---|
| (b) Defense-aware (logit-adjusted loss) on the composite, against metric privacy under the standard attacker | -0.096 [-0.136, -0.057]; two-way [-0.154, -0.038] | -0.094 [-0.143, -0.044]; two-way [-0.170, -0.024] |
| (a) Label-matched surrogates, composite against global DP under the same attacker | -0.093 [-0.134, -0.053]; two-way [-0.161, -0.026] | -0.066 [-0.122, -0.010]; two-way [-0.135, -0.002] |

Neither adaptive attacker recovers the leakage. The defense-aware loss scores the composite at AUC 0.605 and 0.587 at the
two targets, against 0.606 and 0.573 for the standard attacker. Knowing the client trained on a logit-adjusted loss does
not help. The surrogate attacker, which holds only a candidate's label mix, stays well below global DP's surrogate
curve.

## Reading

- The composite beats every full-participation mechanism measured on this design (metric privacy, private median,
  global DP) by about 0.08-0.11 AUC at matched accuracy, and the reduction holds against both adaptive attackers.
- Most or all of that reduction comes from subsampling: B1's `q = 0.25` alone sits on the same curve. Logit adjustment
  moves the composite along the curve, giving higher accuracy at the same noise, and adds no measured privacy at matched
  accuracy. At the low-noise end it does extend the reachable accuracy (0.772 against B1's best of 0.746).
- B5's logit-adjustment gain (-0.100 at 2x) was measured against unadjusted training at the same noise, not at matched
  accuracy against subsampling. C1 does not contradict it; the two effects do not add up on this design.

## Limits

- EuroSAT only (decision 15); on Alzheimer leakage without noise is already near chance.
- The formal guarantee is weaker than global DP's and mostly not computable (B1's limit carries over).
- The lowest-client accuracy drops to 0.33 at the 0.660 target level (0.20 at the noisiest level); fairness cost is not
  gated.
- One design (`R = 32`, alpha 0.3, layout 42); the comparison against subsampling alone is underpowered for differences
  below about 0.05.

## Cost

GPU 1 was shared with another user's job (owner approval); GPU 0 was left to other users. Pilot: 12 runs, 1.3 run-hours.
Main runs: 128 runs, 2026-10-29 20:15 to 2026-10-30 06:24, about 10.2 hours of GPU 1 time with two workers in parallel
(15.6 run-hours summed per run). In GPU-occupancy terms that is about 11.5 GPU-h against the 15 GPU-h cap (decision 16).
Summed per-run time is 16.9 h: over the cap but under the 18.75 stop line. The scaler never added workers, because the
other user's job stayed on GPU 1 throughout.
