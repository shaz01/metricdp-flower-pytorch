# Accuracy-aware gate: development, selection and fresh-data confirmation

Owner (2026-10-09): "move to the next steps one by one, do all of them autonomously; the report at the very end". Rules below are fixed before the confirmation cells are generated or looked at.

## Why

The validation gate optimises cross-entropy. In the public-set sweep and transfer, mean accuracy rose (+0.3 to +1.1 points per group) but 29/120 sweep cells and 12/60 transfer cells had a negative accuracy delta (worst -2.6 points). Can a gate that also looks at validation accuracy remove most of those dips without giving up the CE gain?

## Gate variants (fixed here; all use the 512-image public validation scores of the 8 step multipliers, m=0 is the public base)

- **V0** (current): choose argmin validation CE.
- **V1** (non-inferiority): among multipliers whose validation accuracy >= the base's (m=0 always qualifies), argmin validation CE.
- **V2** (accuracy-first): among multipliers whose validation CE <= the base's, the one with the highest validation accuracy (ties: lower validation CE); m=0 if none qualify.
- **V3** (relaxed non-inferiority): among multipliers whose validation accuracy >= the base's minus 0.01, argmin validation CE.

Each release applies the variant independently (the validation scores are post-processing of the released aggregate and public data only, so every variant keeps the same privacy accounting as V0).

## Development (data already seen)

Cells: the 120 KMNIST sweep cells (sets 0-29, 2 tasks, q.65/.80) and the 60 transfer cells (3 tasks, sets 0-9), train-split evaluation images, re-run with validation accuracy stored (deterministic; the re-run must reproduce the committed numbers exactly). Per cell and variant: mean held-out CE gain over the control and mean accuracy delta.

Per group (task, risk) the reliability rule of the earlier protocol (p_win >= .80 with gain > .001, p_loss <= .10 with gain < -.003, median > 0) and the pooled **accuracy-dip share** = share of dev cells with accuracy delta < -0.005.

**Selection rule:** a variant is eligible iff (i) the reliability rule holds in all 10 groups and (ii) in every group its median CE gain >= 0.8 x V0's. Among eligible variants take the one with the lowest accuracy-dip share. It is adopted only if its dip share <= 0.7 x V0's (a relative reduction of at least 30%); otherwise V0 is kept and the finding is "no accuracy-aware gate of this family helps".

## Confirmation (fresh public sets and fresh evaluation images)

Cells: KMNIST sets 30-59 (2 tasks, q.65/.80; 120 cells) and MNIST/Fashion-MNIST sets 10-19 (3 tasks, q.65/.80; 60 cells), cohort A, 256 releases, held-out CE and accuracy on a fixed 2,000-image subset of the dataset's TEST split (500 per class), which no role or earlier result ever used. The confirmation cells are generated while development analysis runs but are NOT inspected until the variant is chosen. All four variants are computed offline from the stored per-release arrays; the pre-selected variant (or V0 if none adopted) is the confirmation, the others are descriptive.

**Confirmation success:** on the fresh cells the reliability rule holds for the chosen variant in at least 9 of the 10 groups, AND (if a variant other than V0 was adopted) its accuracy-dip share is lower than V0's by at least 30% relative. V0's own fresh-data result is reported regardless: it confirms or refutes the headline claim on data that played no role in any design choice.

## Limits

Development data double as the source of the variants' motivation (the dips were observed there); only the confirmation is clean. Validation accuracy on 512 images has a standard error of about 0.017, so non-inferiority tests are noisy by construction. Single cohort per set; small CNN; Gaussian; same caveats as the earlier protocol.
