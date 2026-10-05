# Paid protected-history routing: findings and pickup point

2026-10-05. This owner-approved bounded comparison is documented; the overall research experiment remains active. Read the [frozen protocol](2026-10-05_protected_real_history_protocol.md) and [independent mathematical/result review](2026-10-05_protected_real_history_math_review.md).

## Finding

A protected early update can carry useful information for selecting a later client-specific profile. On the saved label-skew geometry, adaptation beats a globally optimized static two-release control in 12 of 54 cases and a stronger full-budget one-release control in six. All six one-release improvements occur at budget label 8, both replay paths and all three seeds. Gains are only **0.000109382–0.000246798 in expected development quadratic loss**; **zero cases reach the retained 0.001 material-improvement gate**. These numbers are neither measured cross-entropy improvements nor CIA scores.

This is narrower than a defense result: the queries, model anchors, weights and raw development calibration come from an unprotected trajectory. The calculation shows limited construction headroom under fixed anchors, not a certified private learning mechanism or superiority to metric privacy.

## How the candidate works

Each registered client sends an early clipped update plus Laplace noise. It reconstructs a coarse four-class bias direction from its own protected three-dimensional upload, then maps that direction to one of three fixed clipping/noise profiles. A later update is clipped inside that selected weighted L1 body and receives independent coordinate Laplace noise. Equal-axis and compressed-axis profiles come from the earlier public bank; this test does not invent a new noise density.

The selector reads only the protected early upload. It does not read fresh raw labels or class counts. Unlike the separately paid randomized-response selector, this construction reuses information in a release that can also contribute to learning. The probe still consumes privacy budget and adds noise.

For public probe radius C0, Y0 = clip(u0,C0) + Lap(C0/eps0). For selected profile radii r, Y1 = clip(u1,r) + Lap(r/eps1), coordinatewise. An empty dataset uses zero queries and the same traffic and decoder. Conditional on genuinely public or previously protected histories/calibration, contribution-versus-dummy transcript accounting is eps0+eps1=E; arbitrary replacement is bounded by 2E. The saved raw histories and data-dependent configuration fitting do not satisfy those preconditions automatically.

The final correction is a A+b(B+Z1), where A is the weighted probe aggregate and B the weighted selected clipped main aggregate, with a,b in [0,1]. We retain AHA, AHB and BHB moments, including the dependence between probe noise and selected profile. Independent main noise is integrated exactly: its Hessian trace is 2 sum_i alpha_i² sum_j r_ij² H_jj / eps1². Both coefficients therefore affect the signal, cross term and noise cost correctly.

## Comparison and evaluation

Three saved Fashion-MNIST reduced-classifier seeds, three partition regimes, paths 0→5 and 5→20, and budgets 4/8/16 give 18 records and 54 cases. Only the three identifiable bias contrasts are used. The later query is taken from the original raw trajectory; it was not produced by training from this noisy probe.

Settings and coefficients were fitted on 4,096 probe draws. Frozen choices were evaluated using 16,384 fresh probe draws at the same development-data gradient/Hessian. Main noise is analytically integrated. Pointwise normal intervals measure conditional probe Monte Carlo uncertainty, not held-out-data uncertainty or multiple-comparison-adjusted confirmation. Innovations are paired within a case and reused across some budgets/partition cells, so cell errors are not independent replications.

Controls all receive the same raw calibration information:

- Same-probe static: identical selected probe radius and budget split; independently optimized public main profile/radius and both coefficients.
- Optimized static: additionally optimizes the whole probe/split grid, including all four public profiles.
- One release: full budget, early/current/average query, all four profiles and 287 common public mixtures, the same main-radius grid, and output shrink in [0,2] to match the possible two-stage coefficient sum. This control changes the message schedule and integrates its quadratic expectation exactly.

All include no-op. The main radius grid is 0.001/0.005/0.02/0.1/0.3/1/3; probe radius is 0.02/0.1/0.3 and probe fractions are 0.125/0.25/0.5. Finite searches are not universal optimality claims.

| Comparison | Adaptive wins / 54 | Location | Passes 0.001 against one release |
|---|---:|---|---:|
| Same-probe static | 18 | Label stress, all three budgets | — |
| Optimized static | 12 | Label stress, budgets 8 and 16 | — |
| Full-budget one release | 6 | Label stress, budget 8 | 0 |

Every negative paired contrast has a negative upper endpoint in its pointwise noise-only interval. Adaptation loses to one release in all balanced and quantity cases. In label stress it also loses at budget 4 and budget 16, despite improvements over weaker two-release controls.

| Label-stress budget | Adaptive minus optimized static, range | Adaptive minus one release, range |
|---|---:|---:|
| 4 | +0.000149100 to +0.000186574 | +0.000307908 to +0.000499231 |
| 8 | −0.000780333 to −0.000518679 | −0.000246798 to −0.000109382 |
| 16 | −0.000924579 to −0.000443458 | +0.000002337 to +0.000309260 |

Negative values favor adaptation. One weaker same-probe comparison improves by more than 0.001; that cannot replace the registered stronger-control gate.

## Strongest case

Seed 44, label stress, path 5→20, E=8 selects probe/main budgets 4/4, radii 0.3/0.3, and correction coefficients [0.46346855,1].

| Arm | Expected quadratic loss change |
|---|---:|
| Adaptive | −0.001691703106 |
| Same-probe static | −0.000786848681 |
| Optimized static | −0.000911370433 |
| One release | −0.001444904875 |

Adaptive minus one release is −0.000246798230, with pointwise noise-only 95% interval [−0.000258243573,−0.000235352888]. This is a change around the same raw development anchor, not total loss or a measured CE difference. Protected uploads select the stress-regime target profiles in approximately 84–95% of draws per client in this case. That is profile agreement in this manufactured label-stress diagnostic, not general class-inference accuracy or a privacy measurement.

## Reproduction and verification

Run `uv run python -m research.calculations.protected_real_history_probe --self-check` for mathematical checks, or omit the flag to regenerate the comparison from the saved geometry archive and cached training Arrow. No test Arrow or new training is used. [Calculator](../calculations/protected_real_history_probe.py), [settings/results](../../results/client_specific_noise/protected_real_history_probe.json), and [fresh-probe conditional trial values](../../results/client_specific_noise/protected_real_history_trials.npz) are committed together.

Checks cover convex box optimization versus a dense grid, fitted-versus-evaluated objective consistency, slot permutation with paired innovations, and empirical independent-main covariance. Independent review recomputed all 162 arm means/standard errors and 162 paired contrasts/intervals from the archive, agreeing within 1e-15. No core library or training code changed.

## Research decision and next proposed step

Keep protected-history reuse as a possible construction ingredient: it can provide useful geometry without a fresh RR channel. Do not promote this particular bias-profile replay to a full FL implementation from these small gains. The current bank primarily changes clipping signal, and stronger controls remove most apparent headroom.

Before another sampler or sweep, review the **observer-specific protection contract**: distinguish the original model-observer CIA game from an observer who sees every individual upload, then determine which client-side construction and aggregate-output law can legitimately protect dataset contribution under the intended observer. Keep metadata and weighting explicit. Compare any proposed cost sharing or aggregate-noise route under stated trust assumptions; do not silently introduce secure aggregation or a trusted server. This is the next proposed design/literature question, not an authorized new experiment or a selected defense. Whole-trajectory accounting and independent model/CIA evaluation remain prerequisites for a paper claim.
