# Tonight's rotated-profile and residual-aware comparison

2026-10-04. The owner requested work on this comparison tonight and then explicitly selected **“Stop after this comparison”**. Scope ends with this comparison, its reproducible artifacts and handoff. No follow-on diagnostic, image-model run or CIA experiment is launched.

## Result

Updating the selector from the first upload Y_i,0 to the residual proxy Y_i,0-w_1 usually improves it relative to a stale selector, but **does not establish a useful general advantage over tuned public geometry**. Against the tuned static rotated bank, the residual-aware mechanism is worse in five of six cells and about 0.39% better in one cell (E=16,N=8). The public one-release control performs better in every cell.

This comparison isolates an important limitation: knowing something about the initial client update is not enough to construct a useful next-round distribution after the shared model changes. It does not prove that all protected-history or client-specific mechanisms fail. The isolated high-epsilon gain is too narrow to justify a deep-model sweep or a better-CIA-defense claim.

## Design held fixed

Use the same full-vector changing-gradient quadratic objective as the [previous spike](2026-10-04_dynamic_quadratic_findings.md): F_i(w)=0.5||w-theta_i||², theta_i=0.8e_1/e_2 independently with equal probability, equal-weight registered slots, w_0=0, w_1=mean(Y_i,0), w_2=w_1+mean(Y_i,1). The second raw update is theta_i-w_1; it includes both coordinates.

The first clipping/noise mechanism is a public isotropic diamond. The new second-round bank has diamond semiaxes (L,S), L>=S>0, rotated to 0°,45°,90°,135°. For a rotation R, transform u to R^T u, clip radially in the L1 diamond, and rotate back. Fresh noise is R times independent Laplace coordinates of scales 2L/eta_1 and 2S/eta_1.

All bank profiles must protect every possible input, not just inputs that prefer them. Conditional covariance is 8 R diag(L²,S²)R^T/eta_1². Trace is angle-invariant, but **the clipping and full noise law are not**. Even L=S gives two different diamond geometries: the 0°/90° body and the 45°/135° body. Equal radii therefore do not make this rotated selector identical to a fixed public profile, unlike the previous axis-only bank.

At each client, pick the profile minimizing Euclidean clipping distortion of the declared proxy. This is equivalent to maximizing its radial clipping factor. Ties use the first public angle. Three fixed rules:

- Stale: proxy=Y_i,0.
- Residual: proxy=Y_i,0-w_1.
- Clipping-corrected residual: proxy=max(1,0.8/a_0)Y_i,0-w_1. This additional diagnostic uses the known synthetic input norm, not a general estimator available without assumptions.

The third proxy is permitted by the publicly known toy setup but would require justification under a real client population. In the chosen residual configurations a_0=0.8, so residual and corrected-residual rules coincide. No benefit is claimed for a correction that was effectively unused.

The construction uses only already privatized upload information, the released global model and public constants. Under the hypothetical full upload transcript and uniform conditional per-profile bounds, the reference privacy cost remains eta_0+eta_1=E. A model-dependent selector is post-processing of the protected interactive history; it is not permission to use raw private local history without accounting. No physical-connection secrecy or finite-precision sampler certification is claimed.

## Fair controls and evaluation

Controls: fixed public rotation from the same bank, independent uniformly random public-profile selection, same-probe tuned static profiles, one-release public diamond using all E, and the known public population mean. A fixed public profile can exploit the same common diagonal geometry as the private selector. Different covariance alone cannot establish different privacy or utility laws.

Per cell, tune on 1,024 development federations and evaluate on 8,192 newly drawn federations. New development/evaluation seeds are separate from the preceding spike. Refit every arm for this comparison; do not compare gains directly against old runs with different samples and objectives. Radius and budget grids remain the same. Static candidates include all four fixed angles; adaptive candidates select among all four. Same-probe controls fix the adaptive arm's first radius/budget and tune their second profile using development only.

Primary endpoint integrates independent second-round noise analytically:

    conditional expected excess
      =0.5||w_1+mean(v_i)-theta_bar||²
       +4(L²+S²)/(N eta_1²).

This reduces tuning noise without omitting selection/model dependence. It is evaluated per held-out theta/probe realization, and is a utility measurement, not an attacker-visible feature or attack score. Secondary measurements retain sampled second-round noise and paired full-trial losses. Common innovations couple arms only for statistical comparison; protecting randomness is not made known to an attacker.

## Held-out primary measurements

Mean conditional expected **excess quadratic loss**:

| Total epsilon | Clients | Tuned public static | Stale selector | Residual selector | Public one-release |
|---|---:|---:|---:|---:|---:|
| 4 | 8 | 0.0580185 | 0.0586740 | 0.0587181 | 0.0398372 |
| 4 | 48 | 0.0118692 | 0.0120424 | 0.0119945 | 0.0067382 |
| 8 | 8 | 0.0170112 | 0.0172106 | 0.0170932 | 0.0100314 |
| 8 | 48 | 0.0030065 | 0.0030545 | 0.0030164 | 0.0016881 |
| 16 | 8 | 0.0045199 | 0.0045672 | 0.0045024 | 0.0024861 |
| 16 | 48 | 0.0007732 | 0.0007732 | 0.0008068 | 0.0004180 |

Residual selection improves on stale selection in four cells, worsens it in two, and beats the tuned static profile only at E=16,N=8. In that cell the residual-minus-static contrast is -0.00001743, with exploratory paired normal 95% interval [-0.00002378,-0.00001108], corresponding to 0.39% lower mean loss. E=16 is a large whole-input privacy budget; a small utility advantage there does not demonstrate practically strong protection.

At E=16,N=48, the residual selector slightly improves on the static profile matched to its own first probe, but loses to the static arm that also optimizes the first probe/budget. This distinction is retained in the artifact; a matched-probe improvement is not overall mechanism superiority.

All primary residual-versus-static paired intervals have the stated sign. They are unadjusted for cells/contrasts and support a bounded exploratory finding, not a broad confirmatory claim. Sampled full-trial endpoints are also saved; they can have wider uncertainty and should not be silently substituted for the lower-variance conditional endpoint.

## Verification and limitations

Independent review checks the rotated density, row/column conventions, clipping sensitivity, conditional covariance and interactive privacy argument. A separate 100,000-draw sampler sanity check gives conditional sampled excess 0.0538043 versus formula 0.0537605, standard error 0.0001198, within the prespecified six-standard-error tolerance. Rotation orthogonality, body containment and the equal-radius rotated-diamond distinction are checked numerically.

These are finite-grid, known-population toy results. The constructor minimizes proxy clipping distortion, not downstream loss, contribution leakage, curvature or mutual information. Its profile bank and public tuning are not globally optimized. Public one-release tuning retains the initial isotropic diamond family; improving that public control could further strengthen the negative comparison. The simple quadratic task is still solvable by one unclipped noiseless step, so it cannot alone justify multiple privacy-consuming learning rounds.

No CIA attack is run. Empty-contribution learning and the registered-slot metadata contract remain obligations for a later attack protocol. Experiment completion remains the owner's decision; this note is the authorized spike readout, not a final branch report.

## Pickup after tonight

Stop after documenting/pushing this comparison, per the owner's reply. When work resumes, read this note and the prior failure cases before selecting another constructor. A useful next decision is whether there is a realistic setting with distinct, persistent client utility geometry that public controls cannot capture. Establish that reason before adding learned densities or neural samplers merely to seek a positive result.

Reproduce with `uv run python research/calculations/rotated_residual_probe.py`.

- [Code](../calculations/rotated_residual_probe.py)
- [Settings and primary/secondary uncertainty](../../results/client_specific_noise/rotated_residual_probe.json)
- [Held-out trial losses](../../results/client_specific_noise/rotated_residual_trial_losses.npz)
- [Independent math review](2026-10-04_rotated_residual_math_review.md)

No systematic-review counts or novelty assessments are changed by this diagnostic.
