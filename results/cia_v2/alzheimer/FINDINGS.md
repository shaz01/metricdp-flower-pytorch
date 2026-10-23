# AZ findings, part 1: M2 on Alzheimer MRI

Owner decision 8, as amended by the owner's choices recorded in [STATE.md](STATE.md) and [ESCALATION.md](ESCALATION.md)
(alpha 1.0, 100 rounds; M2 first within the cap). Protocol: [PROTOCOL.md](PROTOCOL.md) part 1, fixed before any
leakage run. Numbers come from [analysis.json](analysis.json):

```
python -m results.cia_v2.frontier.analyze results/cia_v2/alzheimer/results \
    --configs az-vanilla=0 az-gdp-r0.0025=0.0025 az-gdp-r0.003=0.003 az-gdp-r0.0035=0.0035 \
    --out results/cia_v2/alzheimer/analysis.json
```

## Verdict

**G-M2: useful empirical regime at `r = 0.0025`.** All three conditions hold there: the all-candidate AUC's upper gate
end is 0.568 (at most 0.6), the top-decile point estimate 0.586 (at most 0.6), and the lower end of mean accuracy 0.777
(at least 0.70).

This does **not** replicate EuroSAT's "no useful regime", and the reason is the baseline: on Alzheimer in this setup the
attacker is weak even without noise (vanilla AUC 0.564, upper end 0.620). Global DP's regime is "useful" mainly because
there is little leakage to remove.

## Results

Randomized inclusion (32 candidates, 16 IN per replicate, 32 active of 48), Alzheimer MRI, label-Dirichlet alpha 1.0,
100 rounds, `paper_cnn`, shadow samples at 25%, 11 checkpoints. `R = 32`, `design_seed = 2`. Every round of every run
had all 32 clients (`check_every_client_trained`).

| Config | `mu_total` | AUC, all candidates (97.5%) | Two-way bootstrap | Top-decile AUC (97.5%) | TPR at 5% FPR | Final-round AUC | Threshold AUC | Mean final accuracy (95%) | Lowest client accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla | none | 0.564 (0.507-0.620) | 0.492-0.634 | 0.576 (0.429-0.722) | 0.078 | 0.576 | 0.513 | 0.815 (0.809-0.822) | 0.533 |
| Global DP 0.0025 | 125.0 | 0.540 (0.512-0.568) | 0.476-0.599 | 0.586 (0.471-0.702) | 0.041 | 0.489 | 0.479 | 0.784 (0.777-0.791) | 0.433 |
| Global DP 0.003 | 104.2 | 0.523 (0.490-0.556) | 0.463-0.583 | 0.502 (0.406-0.598) | 0.061 | 0.547 | 0.498 | 0.732 (0.724-0.741) | 0.341 |
| Global DP 0.0035 | 89.3 | 0.546 (0.510-0.582) | 0.471-0.616 | 0.472 (0.335-0.609) | 0.086 | 0.509 | 0.507 | 0.670 (0.658-0.681) | 0.279 |

`mu_total = sqrt(100) / (32 r)`.

| Level | All-candidate upper end (at most 0.6) | Top-decile point (at most 0.6) | Accuracy lower end (at least 0.70) |
|---|---|---|---|
| 0.0025 | 0.568, meets | 0.586, meets | 0.777, meets |
| 0.003 | 0.556, meets | 0.502, meets | 0.724, meets |
| 0.0035 | 0.582, meets | 0.472, meets | 0.658, fails |

## Reading

- **Leakage is near chance across the board.** Vanilla's AUC is 0.564 and the noisy levels 0.523-0.546; no difference
  between levels is distinguishable at this power. TPR at 5% FPR is 0.04-0.09, against 0.17-0.23 on EuroSAT.
- **Why weaker than EuroSAT:** B5's diagnostic found EuroSAT's leakage to be mostly label-distribution inference, and
  this setup has milder label skew (alpha 1.0 against 0.3; chosen because alpha 0.3 did not train). That is the
  likeliest reason, but this package did not test it (no surrogate scoring here).
- **The guarantee is still vacuous** in the useful regime: `mu_total` 104-125.
- **For the paper:** "no useful regime" is setup-dependent. With strong label skew (EuroSAT, alpha 0.3) global DP
  cannot silence the attacker at usable accuracy; with milder skew (Alzheimer, alpha 1.0) the attacker is weak to begin
  with, and DP's added protection is not measurable.

## Limits

- One layout, one skew level; the calibrated loss attacker only.
- Different model, rounds, skew and shadow fraction from EuroSAT, so absolute AUCs do not compare across datasets.
- The operating points are `r_0.6` = 0.0025 only: the grid has no global-DP level below it, so "the next-lower noise"
  does not exist (vanilla).

## Cost

| Step | GPU-h (occupancy) |
|---|---|
| Diagnosis and calibration (stopped alpha-0.3 batch, diagnosis, batches 1 and 2) | about 10 |
| First launch, 2026-10-22 11:05-13:44 (three workers on each GPU, then on GPU 1; runs with failed clients quarantined) | about 2.7 |
| Relaunch, 2026-10-22 13:47 to 2026-10-23 20:55 (two workers on GPU 1, shared by owner approval) | about 31.1 |
| Total | about 44 of the 40 GPU-h cap (stop at 50) |
