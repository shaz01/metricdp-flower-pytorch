# Matched-strength noise-law comparison: findings

Protocol: [noise-law protocol](2026-10-08_noise_law_protocol.md) (gate stated before data were opened). Code `research/calculations/noise_law_probe.py`; artifacts `results/client_specific_noise/noise_law_radial_freeze.json`, `noise_law_comparison.{json,npz}`. Checks: the Gaussian arm reproduces the independently audited gated-step Gaussian gains to 1e-9 in all 12 cells x 2 risks (asserted in the run); a unit test shows the radial sampler equals the earlier matched-CIA sampler draw-for-draw and has the expected 8/7 (d+1) scale^2 variance. The radial arm itself has no separate independent reconstruction beyond that.

## Result

Radial-Laplace freeze (same six development cells and grid): both risks select exactly the Gaussian configuration (target_center, d=51, cap .003, eta 100); pooled selection CE .5954/.5935 versus Gaussian .5951/.5933 (marginally worse).

Fresh KMNIST, budget 32, validation-gated release, 6 cells per task:

| Task | q | Gaussian gain | Radial gain | cells where radial better / worse by > .001 | mean radial advantage |
|---|---|---:|---:|---|---:|
| KMNIST 0-3 | .65 | +.0149 | +.0149 | 0 / 0 | −.00000 |
| KMNIST 0-3 | .80 | +.0174 | +.0174 | 0 / 0 | +.00003 |
| KMNIST 4-7 | .65 | +.0117 | +.0118 | 0 / 0 | −.00008 |
| KMNIST 4-7 | .80 | +.0144 | +.0144 | 0 / 0 | −.00001 |

Largest single-cell difference .00025 CE. Pre-stated verdict: **no consistent benefit of the radial Laplace density over the Gaussian at matched AUC.**

Matched strength was verified (65,536 draws per world, optimal statistic, d=51): Gaussian AUC .6523/.7999, radial .6527/.7970 for targets .65/.80. Low-FPR operating points differ slightly: at q.80 radial TPR at FPR 0.1%/1%/5% is .0235/.1169/.3129 versus Gaussian .0254/.1233/.3227 (radial a few percent lower, about 4 standard errors at the 1% point); at q.65 they are within noise (.0363 vs .0378 at 1%, .1378 vs .1361 at 5%). So the radial law is no better on utility and at most marginally better against the low-FPR attacker at the stronger setting; matched AUC does not equal matched low-FPR risk, and this small difference is not a practical benefit.

## Why this is expected

At the selected d=51, the radial law's norm is concentrated (Gamma shape 26), and the one-step quadratic utility depends on the noise through its covariance, which the calibration makes essentially equal. Consistent with the earlier matched-CIA finding and the analytic work (common-ball shaping cannot improve quadratic utility).

## Limits

Only Gaussian and radial Laplace; per-client-varying or anisotropic laws were not tested (they would need a new accounting and the earlier worst-case argument already rules out spherical-ball shaping); conditional known-alternative contract; linear model on pooled 4x4 features; same offline-tuning and artifact-release caveats as before.

## What this means for the research question

Within the present construction and contract, no distribution change improves the utility/attack trade-off: the useful, reproducible ingredient is the per-client clip/normalize + limited-public stacking + public validation gating, not the noise density. Whether a client-specific distribution can help a different construction (e.g., one where the attacker's shift is not norm-bounded the same way) remains open but is not supported by any result so far.
