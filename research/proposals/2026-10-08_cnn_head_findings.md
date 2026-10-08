# CNN-head port of the stacked, validation-gated constructor: findings

Protocol: [CNN-head protocol](2026-10-08_cnn_head_protocol.md) (gate fixed before the run). Code `research/calculations/cnn_head_probe.py`; artifacts `results/client_specific_noise/cnn_head.{json,npz}`; audit `audit_cnn_head.py` (148 checks, max gated-CE draw error 5.6e-14: roles and class counts, CNN retraining reproduces the base selection and control CE, and an independent recomputation of head queries, public Hessian, clipping, noise, 8 multipliers x 512 draws, validation gating and gains). The CNN training code itself is shared with the probe, so the audit verifies determinism and the construction around the CNN, not the CNN code independently.

## Setup

Small CNN (two conv layers, 32 ReLU features, 4-class head) trained on the public set only (6 candidates, base chosen on public validation CE). The private step acts on the frozen-feature head (99 identifiable parameters) with the FROZEN stacked configuration (target_center, d=51, cap .003, eta 100 x multiplier, risk .65/.80); multiplier grid widened a priori to (0, 1/30, 1/10, 1/3, 1, 3, 10, 30) because CNN feature scale differs. Fresh-pool KMNIST classes 0-3 and 4-7, budgets 32 and 128, 3 public sets x 2 cohorts, 512 draws.

## Result

Gain in held-out CE over the strongest public-only CNN (gated release), cells > .001 / mean / worst cell:

| Task | Budget | q.65 | q.80 |
|---|---|---|---|
| KMNIST 0-3 | 32 | 6/6, +.0114, +.0027 | 6/6, +.0153, +.0045 |
| KMNIST 4-7 | 32 | 5/6, +.0158, −.0001 | 4/6, +.0202, **−.0059** |
| KMNIST 0-3 | 128 | 4/6, +.0019, −.0009 | 4/6, +.0025, −.0011 |
| KMNIST 4-7 | 128 | 6/6, +.0074, +.0036 | 6/6, +.0098, +.0038 |

Mean accuracy change at 32: +0.2 points (0-3), +1.1 to +1.3 points (4-7). Pre-stated primary gate (budget 32, per task and risk: >= 5/6 cells > .001, mean > 0, no cell below −.003): **met in 3 of 4 task/risk pairs; not met for KMNIST 4-7 at q.80** (4/6 cells, worst −.0059). Overall: PARTIAL.

The CNN control is weaker than the pooled-pixel linear base (control CE .56-.92 at 32 examples), so there is more room, and the gains are in the same range as the linear model (+.011 to +.020 CE).

## Diagnostics

- The frozen step length was too large for CNN head scale. Mean CE by multiplier is minimised at 1/10 to 1/3 of the frozen eta in almost every cell; multipliers >= 3 give CE 1.3 to 150 (catastrophic). The widened gate absorbed this: in 47 of 48 cell/risk combinations the gate never picks a multiplier >= 3, and in the 48th only 0.2% of draws do. The gate is doing its job as a scale adapter, which is why a frozen (cap, eta) tuned on pooled pixels transfers at all.
- The failed cell (KMNIST 4-7, public set 2, cohort A, q.80) is NOT caused by catastrophic picks: the gate picks the full frozen step (m=1) in 97.7% of draws, while held-out CE is better at m=1/3 (.787 vs .809 at m=1; control .801). This is finite-validation-set error (512 images, CE gaps of about .02 between adjacent multipliers) for a base trained 300 epochs; its sibling cohort B gains +.0146. A fixed m=1/3 would have gained .014 there, but choosing it now would be post-hoc tuning on these cells and is not adopted.
- At 128 public examples gains shrink (+.002 to +.010), as with the linear model; one cell per risk at 0-3 is slightly negative (≈ −.001).

## Limits

Head-only private step; the feature extractor is trained from the small public set only and is not updated with private data; one-shot release, not a multi-round Flower federation; Gaussian only; 4-class tasks; the attack calibration depends only on cap and risk target and was not repeated; the validation set is extra public labelled data; offline tuning, base selection and artifact release unaccounted; widening the multiplier grid was the single a-priori change from the linear protocol.

## Next

The remaining gap to the original objective is an actual Flower strategy (server noise addition, public-validation gating, round state) integrated with the repo's CIA pipeline, plus an a-priori decision on whether to bound the multiplier at 1 and enlarge the validation set to reduce selection error; either is a new protocol requiring fresh data (all KMNIST classes 0-7 are now used).
