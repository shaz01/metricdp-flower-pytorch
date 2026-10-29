# CIA roadmap v2, B5: exposure of large clients (label-distribution leakage, caps, logit adjustment)

Package B5 of `research/cia_roadmap_v2/ROADMAP.md`, as amended by Section 10 (decisions 9 and 10, 2026-10-20). Branch
`feature/cia-v2-mechanisms`. The owner assigned Agent B's packages to the agent that ran Agent A's, and closed B5 on
2026-10-23. C1, which a passed B gate would open, stays waiting by the owner's decision of the same day.

All numbers below come from committed artifacts in `results/cia_v2/mechanisms/b5_large_clients/`: `STATE.md` (design and
item 1's diagnostic), `PROTOCOL.md` (fixed before any B5 run), `FINDINGS.md` and `analysis.json` from the 128 runs in
`results/`, paired with M2's global-DP runs in `results/cia_v2/frontier/results/`. The command is in PROTOCOL.md
Section 2 and FINDINGS.md.

## 1. Question

M2 found no noise level at which global DP silences the attacker at usable accuracy, and the top decile stayed exposed
at every level. B5 asks what the leakage is made of, and whether a defense aimed at it lowers the attacker's AUC at the
same noise without costing accuracy.

## 2. Item 1: the leakage is mostly label-distribution inference

M2's runs also scored, for every candidate, a label-matched public surrogate set: public test images resampled to the
candidate's label histogram, none of its records. An attacker using only those reaches nearly the true AUC:

| Config | All candidates: true | surrogate | difference | Top decile: true | surrogate | difference |
|---|---|---|---|---|---|---|
| Vanilla | 0.723 | 0.714 | +0.009 | 0.738 | 0.700 | +0.037 |
| Global DP 1x | 0.720 | 0.697 | +0.024 | 0.741 | 0.726 | +0.015 |
| Global DP 2x | 0.702 | 0.680 | +0.022 | 0.746 | 0.726 | +0.020 |
| Global DP 4x | 0.666 | 0.652 | +0.014 | 0.741 | 0.697 | +0.044 |
| Global DP 8x | 0.671 | 0.646 | +0.025 | 0.724 | 0.677 | +0.047 |
| Global DP 16x | 0.593 | 0.584 | +0.009 | 0.603 | 0.608 | -0.005 |

Within the roadmap's 0.05 everywhere, so by its rule the leakage is mostly label-distribution inference (Wainakh et
al. 2022) and a prior-neutral local loss is worth a screen. This also accounts for M2 (the class prior is a strong
low-dimensional signal that noise barely blurs) and A1 (equal accuracy, equal leakage).

## 3. Protocol

- **Regime:** M2's (EuroSAT alpha 0.3, randomized inclusion with 32 candidates and 32 active of 48, 50 rounds, equal
  weights, `R = 32`, `design_seed = 2`), so each B5 run is paired with M2's global-DP run of the same replicate.
- **Interventions** (client-side; server, noise and formal guarantee are global DP's at the same noise):
  - Contribution caps: each round a client trains on at most `m` records, a fresh uniform draw from its round seed;
    `m` = 326 (median training-set size; caps 24 clients) and 182 (25th percentile; caps 36).
  - Logit adjustment (Menon et al. 2020): local loss `CE(softmax(log p + tau log pi_k))` with the client's
    add-one-smoothed label prior `pi_k`, `tau = 1`; prediction unchanged.
- **Stages:** A, all three at 2x (`r = 0.003092`); B, at 4x (0.006184) for interventions with a point reduction of at
  least 0.03 and an accuracy loss of at most 1 pp at 2x (logit adjustment only).
- **Statistic:** paired difference `AUC_B5 - AUC_GDP`, pair jackknife at 97.5% (gate), two-way difference bootstrap
  alongside; accuracy difference with a paired bootstrap over replicates.
- **G-B5 (decision 9):** all-candidate AUC at least 0.05 below global DP at the same noise, gate interval below 0,
  accuracy loss at most 1 pp. Top decile reported, not gated.

## 4. Results

| Configuration | All-candidate AUC vs global DP | Difference, gate interval | Two-way | Accuracy vs global DP | G-B5 |
|---|---|---|---|---|---|
| Cap 326, 2x | 0.698 vs 0.702 | -0.004 [-0.041, 0.033] | [-0.044, 0.038] | 0.704 vs 0.740 (-3.6 pp) | fail |
| Cap 182, 2x | 0.696 vs 0.702 | -0.006 [-0.042, 0.031] | [-0.055, 0.044] | 0.657 vs 0.740 (-8.3 pp) | fail |
| **Logit adjustment, 2x** | **0.601 vs 0.702** | **-0.100 [-0.152, -0.049]** | **[-0.172, -0.035]** | **0.751 vs 0.740 (+1.1 pp)** | **pass** |
| Logit adjustment, 4x | 0.630 vs 0.666 | -0.036 [-0.074, 0.002] | [-0.090, 0.017] | 0.673 vs 0.660 (+1.3 pp) | fail |

| Configuration | Top-decile AUC vs global DP (difference, gate interval) | TPR at 5% FPR | TPR at 1% FPR | Lowest client accuracy |
|---|---|---|---|---|
| Cap 326, 2x | 0.691 vs 0.746 (-0.055 [-0.163, 0.053]) | 0.188 vs 0.174 | 0.039 vs 0.066 | 0.449 vs 0.512 |
| Cap 182, 2x | 0.703 vs 0.746 (-0.043 [-0.129, 0.042]) | 0.123 vs 0.174 | 0.045 vs 0.066 | 0.352 vs 0.512 |
| Logit adjustment, 2x | 0.710 vs 0.746 (-0.036 [-0.119, 0.047]) | 0.168 vs 0.174 | 0.051 vs 0.066 | 0.505 vs 0.512 |
| Logit adjustment, 4x | 0.738 vs 0.741 (-0.003 [-0.089, 0.083]) | 0.115 vs 0.186 | 0.043 vs 0.039 | 0.377 vs 0.366 |

**Validity checks for logit adjustment at 2x:** the final-round attacker also drops (0.611 to 0.563) and the
uncalibrated threshold attacker slightly (0.568 to 0.555); no run diverged (largest final shadow loss 2.27 against
2.34); the mean included-minus-excluded final-loss gap shrinks from 0.075 to 0.058.

## 5. Reading

- **Factoring the client's own label prior out of local training removes much of the leakage at no accuracy cost.**
  At 2x the AUC falls by 0.10 while accuracy rises by 1.1 pp; logit adjustment is local computation on the client's
  data, so global DP's client-level guarantee is unchanged. It is the roadmap's first mechanism to pass its gate, and
  it acts on exactly the channel item 1 identified.
- **The gain is in the bulk, not the tail.** TPR at 5% FPR is unchanged at 2x and TPR at 1% falls only from 0.066 to
  0.051: confidently identified candidates stay identified.
- **It weakens with noise:** -0.036 at 4x, where noise already blurs part of the label signal.
- **The largest clients gain least** (-0.036 and -0.003, not significant): their exposure is not mostly label mix.
- **Caps fail:** they cost 3.6-8.3 pp and leave the AUC unchanged, since they do not change the label mix a client
  contributes.

## 6. Anomalies and corrections

- **Shared GPU.** Both stages ran on GPU 1 shared with another user's process by owner approval. A later scan of every
  B5 run (2026-10-22) found every client present in every round; the out-of-memory failures that hit the Alzheimer
  runs did not occur here.
- **Ownership.** B5 changes `experiments/reproduce/client.py` and `paper_training.py` (Agent B's files) and the
  randomized-inclusion runner (Agent A's); both options are default-off and M2's 437 manifests regenerate unchanged.

## 7. Caveats

- One dataset (EuroSAT), one layout, one heterogeneity level; the calibrated loss attacker family.
- One `tau` (1.0), untuned; label calibration (Zhang et al. 2022) not run.
- No adaptive attacker: an attacker who knows clients use logit adjustment might score with adjusted logits.
- Stage B covered only the intervention that met the stage-B rule.

## 8. Cost

| Step | Runs | GPU-h (occupancy) |
|---|---|---|
| Stage A at 2x (2026-10-20 10:55-19:44) | 96 | about 8.8 |
| Stage B at 4x (2026-10-21 16:47-20:29) | 32 | about 3.7 |
| Total | 128 | about 12.5 of the 15 GPU-h cap |

## 9. Conclusions and owner decisions (2026-10-23)

1. **EuroSAT's leakage is mostly label-distribution inference** (surrogate attacker within 0.009-0.025 AUC).
2. **G-B5 passes for logit adjustment at 2x** (-0.100 [-0.152, -0.049], +1.1 pp accuracy) and fails at 4x; caps fail.
3. **Closed by the owner.** By ROADMAP.md Section 4 a passed B gate opens C1; the owner keeps C1 waiting.
4. The branch merges into master after B1, which is in progress on the same branch.
