# Public-set sweep and cross-dataset transfer: findings

Protocol (decision rules fixed before any new cell was run): [2026-10-09_public_set_sweep_and_transfer.md](2026-10-09_public_set_sweep_and_transfer.md). Data: `results/stacked_head/*_sweep.json` (120 cells) and `*_transfer.json` (60 cells) with `.releases.npz` sidecars; reports `sweep_report_sweep.{json,txt}` and `sweep_report_transfer.{json,txt}`. Every cell is 256 independent one-shot releases through the real ServerApp/ClientApp code (in-process backend; on the two cells compared, set 0 of KMNIST 0-3 at both risks, per-release held-out CE and step picks were bit-identical to the Ray simulation, max difference exactly 0). Consistency audit `audit_releases.py`: 180 cells, 1,440 checks (every gate pick, gated/alternative/fixed gain and accuracy delta recomputed from the stored per-release arrays). The base-model/query/noise code is the code validated against the research probe (1e-8) in the previous phase; this phase's new code (extra public sets, per-release diagnostics, report) has unit tests plus the audit above.

## Rule and outcome

"Reliable at 32 public images" for a task and risk iff p_win (share of public sets with gain > .001 CE) >= 0.80, p_loss (gain < -.003) <= 0.10 and median gain > 0.

| Dataset / task | Risk | Sets | p_win (95% CI) | p_loss | Median gain (CE) | p10 | Mean accuracy delta | Verdict |
|---|---|---:|---|---:|---:|---:|---:|---|
| KMNIST 0-3 | .65 | 30 | .90 (.74-.97) | .00 | +.0341 | +.0015 | +0.61 pts | reliable |
| KMNIST 0-3 | .80 | 30 | .87 (.70-.95) | .00 | +.0367 | +.0007 | +0.77 pts | reliable |
| KMNIST 4-7 | .65 | 30 | .90 (.74-.97) | .00 | +.0303 | +.0011 | +0.61 pts | reliable |
| KMNIST 4-7 | .80 | 30 | .87 (.70-.95) | .03 | +.0421 | +.0008 | +0.70 pts | reliable |
| MNIST 0-3 | .65 | 10 | 1.00 (.72-1.00) | .00 | +.0099 | +.0021 | +0.33 pts | reliable |
| MNIST 0-3 | .80 | 10 | 1.00 (.72-1.00) | .00 | +.0127 | +.0042 | +0.43 pts | reliable |
| MNIST 4-7 | .65 | 10 | .90 (.60-.98) | .00 | +.0282 | +.0082 | +0.92 pts | reliable |
| MNIST 4-7 | .80 | 10 | .90 (.60-.98) | .00 | +.0357 | +.0091 | +1.13 pts | reliable |
| Fashion-MNIST 4-7 | .65 | 10 | 1.00 (.72-1.00) | .00 | +.0257 | +.0048 | +0.43 pts | reliable |
| Fashion-MNIST 4-7 | .80 | 10 | 1.00 (.72-1.00) | .00 | +.0268 | +.0054 | +0.32 pts | reliable |

Pre-stated dataset-level verdicts: KMNIST 4/4 pairs reliable (transfer within dataset holds with 30 sets); **MNIST 4/4 and Fashion-MNIST 2/2 reliable, so both transfer** under the rule. Losses: 1 of 120 sweep cells below -.003 (KMNIST 4-7, q.80, public set 2: gain -.0062, the same set that failed in the 3-set gate matrix), 0 of 60 transfer cells; the smallest transfer gain is +.0004.

## What the gate does (important)

- **Without the validation gate the frozen step is destructive.** Fixed step multiplier 1 (no gating) has median gain between -0.05 and -0.98 CE and p_loss between .87 and 1.00 in every task and risk. The frozen step length was tuned on pooled-pixel linear models and is 3-10x too large for the CNN head; the public validation gate picks a step of 1/3 (57%/53% of releases in sweep/transfer) or 1/10 (31%/36%) of it, never 3x or more. The reliability above is therefore the property of construction + public-validation gating, not of the frozen step alone.
- **A 128-image validation set is adequate** under the rule: it is reliable in all ten task/risk pairs, as the 512-image gate is; its p_win is lower in two pairs (KMNIST 4-7 q.65 by .03, MNIST 0-3 q.65 by .10), higher in one (KMNIST 4-7 q.80 by .03) and equal in the rest, and its p_loss is .03 instead of .00 in one pair (KMNIST 0-3 q.80).
- The one failure is a gate error, not a missing step: on that set a fixed multiplier of 1/10 would have gained +.0142 on held-out data, but the 512-image validation picked a larger step. Single set, so not generalisable beyond "gate selection noise can cost about .006 CE in rare sets".

## Caveats the numbers do not remove

- **Accuracy is not guaranteed to improve.** The gate optimises cross-entropy. Mean accuracy changes are positive (+0.3 to +1.1 points per group) but 29 of 120 sweep cells and 12 of 60 transfer cells have a negative accuracy delta (worst -2.6 points; 8 and 3 cells respectively worse than -0.5 points).
- **Weak base-strength dependence, not established.** Spearman correlation of gain with control CE is positive but not significant for KMNIST (rho +.06 to +.28, p >= .14); MNIST 4-7 shows rho +.78 (p .008, n=10, two risks are the same sets so one finding): weaker public bases gain more there.
- With only 10 sets per transfer task the Wilson upper bound on p_loss is .28; the rule is a point-estimate rule and these results do not certify p_loss <= .10 statistically.
- One cohort per public set (cohort A); 4-class tasks; small CNN head-only step; Gaussian only; the public validation set (512 images, 128 also tested) is extra labelled public data beyond the 32 training images; replicate rounds are independent releases from one base, not a multi-round protocol; no secure aggregation; offline tuning and base selection unaccounted; Fashion-MNIST classes 0-3 deliberately excluded (the construction was tuned on that class set's development pools); MNIST/Fashion-MNIST pools were used in earlier linear diagnostics but never to tune this construction.
- Conditional peer/known-alternative attack contract only (one real-path check of the attack in the previous phase); not a general CIA defense or a privacy accounting of the tuning.

## What this supports

On four datasets-by-class-split tasks (KMNIST 0-3/4-7, MNIST 0-3/4-7, Fashion-MNIST 4-7) with 32 public training images, the stacked, validation-gated client-query step improves held-out cross-entropy over the best public-only model in at least 87% of public sets at a calibrated attack AUC of .65 and .80, with losses in 1 of 180 cells. It does not support claims about larger public budgets (the earlier 128/512-image results stand), accuracy gains for every set, other model classes, or any benefit of non-Gaussian or client-specific noise densities.
