# AZ escalation: Alzheimer does not train to useful accuracy in the decided setup

Raised 2026-10-20 by the agent running owner decision 8. Calibration stopped after 8 of 18 runs; nothing else ran.

## Finding

Accuracy calibration (accuracy only, `design_seed = 3`, 2 runs per level), the decided setup: Alzheimer MRI, 48 slots,
alpha 0.3, at least 10 records per client, 32 active clients, equal weights, `paper_cnn`, v1's hyperparameters, 50
rounds.

| Configuration | Final accuracy (2 runs) | Mean |
|---|---|---|
| Vanilla | 0.547, 0.561 | 0.554 |
| Global DP, r = 0.000625 | 0.541, 0.597 | 0.569 |
| Global DP, r = 0.00125 | 0.591, 0.655 | 0.623 |

The majority class (Non_Demented) is 50.1% of the training split. Vanilla's server accuracy every 5 rounds (replicate
0): 0.011, 0.495, 0.495, 0.495, 0.495, 0.495, 0.528, 0.511, 0.542, 0.548, 0.547. The model predicts the majority class
for about 25 rounds and then improves slowly.

**Why:** at 48 slots and alpha 0.3 the clients are small and skewed (14-372 records, median 90; about 72 training
images for a median client), each trains 5 local epochs with Adam, and equal-weight averaging of these divergent
updates makes little progress. v1's 0.907 at 48 clients used v1's non-IID partition mode (not label-Dirichlet 0.3)
and 100 rounds.

## Consequence

The replication cannot test what it was decided for:

- G-M2's accuracy condition (lower end at least 0.70) fails for every configuration, including vanilla, so "no useful
  regime" would hold trivially.
- A1's targets are global-DP levels around 0.70 accuracy, which do not exist.

## Options

1. **Diagnose first (recommended).** Six accuracy-only vanilla runs (2 each, `design_seed = 3`) to find a setup that
   trains: alpha 1.0; the homogeneous (IID) partition; alpha 0.3 with 100 rounds. About 45 minutes on a shared GPU,
   about 0.5 GPU-h. The owner then picks the setup.
2. **Homogeneous partition.** v1 reached 0.913 there, and the roadmap's risk table already plans it for Alzheimer in
   C1. But identical label mixes remove the channel B5's diagnostic found to carry most of the leakage on EuroSAT
   (label-distribution inference), so this replicates the conclusions under a different leakage source.
3. **Another second dataset.** Fashion-MNIST at 48 clients (v1: vanilla 0.955 in v1's non-IID mode) with the same
   design, alpha 0.3. Runner support is a new dataset profile.
4. **Drop the replication** and state the single-dataset scope in the paper.

Cost so far: 8 runs, about 1 GPU-h of shared occupancy.

## Diagnosis and resolution (owner, 2026-10-21)

Option 1 ran (vanilla, 2 runs each, final accuracy): alpha 1.0 with 50 rounds 0.738 and 0.742 (still rising: 0.666 at
round 40); alpha 100 (near-IID) with 50 rounds 0.787 and 0.772; alpha 0.3 with 100 rounds 0.595 and 0.652.

The owner chose **alpha 1.0 with 100 rounds**: it keeps label skew (the channel that carries most of the leakage on
EuroSAT) and should leave room above G-M2's 0.70 floor, which the calibration checks. STATE.md records the amended
design.

