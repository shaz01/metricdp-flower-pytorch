# B5 findings: contribution caps and a prior-neutral local loss against global DP

Package B5 of `research/cia_roadmap_v2/ROADMAP.md`, as amended by Section 10 (decisions 9 and 10). Design in
[STATE.md](STATE.md), runs and gate in [PROTOCOL.md](PROTOCOL.md). Numbers come from [analysis.json](analysis.json):

```
python -m results.cia_v2.mechanisms.b5_large_clients.analyze \
    --pairs b5-cap326-r2x=gdp-r2x:0.003092 b5-cap182-r2x=gdp-r2x:0.003092 b5-la-r2x=gdp-r2x:0.003092 \
    b5-la-r4x=gdp-r4x:0.006184 --out results/cia_v2/mechanisms/b5_large_clients/analysis.json
```

## Verdict

**G-B5 passes for logit adjustment at 2x and fails at 4x; both contribution caps fail.**

| Configuration | All-candidate AUC vs global DP | Difference, gate interval (97.5%) | Two-way bootstrap | Accuracy vs global DP | G-B5 |
|---|---|---|---|---|---|
| Cap 326, 2x | 0.698 vs 0.702 | -0.004 [-0.041, 0.033] | [-0.044, 0.038] | 0.704 vs 0.740 (-3.6 pp) | fail |
| Cap 182, 2x | 0.696 vs 0.702 | -0.006 [-0.042, 0.031] | [-0.055, 0.044] | 0.657 vs 0.740 (-8.3 pp) | fail |
| **Logit adjustment, 2x** | **0.601 vs 0.702** | **-0.100 [-0.152, -0.049]** | **[-0.172, -0.035]** | **0.751 vs 0.740 (+1.1 pp [+0.4, +1.7])** | **pass** |
| Logit adjustment, 4x | 0.630 vs 0.666 | -0.036 [-0.074, 0.002] | [-0.090, 0.017] | 0.673 vs 0.660 (+1.3 pp [+0.1, +2.4]) | fail |

Each configuration is paired with M2's global-DP run at the same noise (same inclusion and seeds per replicate). The
gate: all-candidate AUC at least 0.05 lower, gate interval below 0, accuracy loss at most 1 pp.

## Item 1: the leakage is mostly label-distribution inference

In M2's runs, an attacker scoring label-matched public surrogates instead of a candidate's records reaches nearly the
true AUC: within 0.009-0.025 for all candidates and within 0.05 for the top decile, at every noise level (STATE.md
table). That is why item 3 (a prior-neutral local loss) ran.

## Secondary metrics

| Configuration | Top-decile AUC vs global DP (difference, gate interval) | TPR at 5% FPR, all | TPR at 1% FPR, all | Lowest client accuracy |
|---|---|---|---|---|
| Cap 326, 2x | 0.691 vs 0.746 (-0.055 [-0.163, 0.053]) | 0.188 vs 0.174 | 0.039 vs 0.066 | 0.449 vs 0.512 |
| Cap 182, 2x | 0.703 vs 0.746 (-0.043 [-0.129, 0.042]) | 0.123 vs 0.174 | 0.045 vs 0.066 | 0.352 vs 0.512 |
| Logit adjustment, 2x | 0.710 vs 0.746 (-0.036 [-0.119, 0.047]) | 0.168 vs 0.174 | 0.051 vs 0.066 | 0.505 vs 0.512 |
| Logit adjustment, 4x | 0.738 vs 0.741 (-0.003 [-0.089, 0.083]) | 0.115 vs 0.186 | 0.043 vs 0.039 | 0.377 vs 0.366 |

Validity checks for logit adjustment at 2x (STATE.md log): the final-round attacker also drops (0.611 to 0.563) and the
threshold attacker slightly (0.568 to 0.555); no run diverged (largest final shadow loss 2.27 against 2.34); the mean
final-loss gap between included and excluded candidates shrinks from 0.075 to 0.058.

## Reading

- **Removing the client's label prior from local training removes much of the leakage, at no accuracy cost.** At 2x
  the attacker's AUC falls by 0.10 and accuracy rises by 1.1 pp, under the same noise and the same formal guarantee
  (logit adjustment is local computation on the client's own data, covered by client-level DP). This fits item 1:
  the attack mostly reads each client's label mix through the model's class priors.
- **The gain is in the bulk, not the tail.** TPR at 5% FPR is unchanged at 2x (0.168 vs 0.174) and TPR at 1% falls only
  from 0.066 to 0.051. The candidates the attacker identifies confidently stay identifiable.
- **It weakens with noise.** At 4x the reduction is 0.036 and its interval reaches 0.002. More noise already removes
  part of the label signal that the adjustment targets (global DP's own AUC is 0.666 there).
- **The largest clients gain least.** Top-decile reductions are -0.036 (2x) and -0.003 (4x), neither significant; their
  exposure is not mostly about label mix.
- **Contribution caps do not work here.** Capping large clients' records per round costs 3.6-8.3 pp of accuracy and
  leaves the AUC unchanged; it does not touch the label mix a client contributes.

## Limits

- One dataset (EuroSAT), one layout, one heterogeneity level; the calibrated loss attacker family.
- One `tau` (1.0) and no tuning; label calibration (Zhang et al. 2022) was not run.
- An attacker who knows the clients use logit adjustment could adapt, for example by scoring with adjusted logits.
  Not tested.
- Stage B ran only the intervention that met the stage-B rule, as fixed in the protocol.

## Cost

| Step | Runs | GPU-h (occupancy) |
|---|---|---|
| Stage A at 2x (two workers on GPU 1, shared by owner approval, 2026-10-20 10:55-19:44) | 96 | about 8.8 |
| Stage B at 4x (two workers on GPU 1, shared by owner approval, 2026-10-21 16:47-20:29) | 32 | about 3.7 |
| Total | 128 | about 12.5 of the 15 GPU-h cap |
