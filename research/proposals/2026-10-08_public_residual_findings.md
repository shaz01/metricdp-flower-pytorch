# Public-reference residual result and a distribution-aware construction lead

2026-10-08, `feature/client-specific-noise`. Owner: “continue with this step dont stop until you find something”. This records a bounded verified research chunk, not completion of the overall experiment or a deployed defense.

## What changed our direction

**The selected public-model residuals fail the near-chance protected-utility gate.** On512fresh images, the apparent projected-model gain is reproduced and slightly exceeded by a public-only decoder adjustment. The noisy private release adds no demonstrated value to that adjustment. Class-balanced local fine-tuning residuals lose to it in all three seeds at conditional CIA ceiling0.55.

There is a more concrete construction lead from a separate DEVELOPMENT-ONLY diagnostic: remove each client's gradient component predictable from its class mixture under a public class-conditional reference. Mean client vector norms fall84–88%, while the all-IN unbounded aggregate gradient is preserved to numerical precision in these balanced-prior federations. This concerns an internally constructed query, not a validated noise distribution or protected utility result. It is different from the failed fixed-model parameter subtraction.

These two findings should steer the roadmap: first preserve useful aggregate information while removing client-specific nuisance components, then determine the complete noise/accounting law. Do not credit public pretraining or offset calibration to privacy, and do not start another density sweep on a query that fails even without noise.

## Primary-source foundation and technique explanations

The [methods tutorial](../literature_review/public_reference_methods.md) explains GEP, projected DP-SGD, public pretraining/private fine-tuning, PATE and SCAFFOLD, with exact read scopes and privacy-transfer limits. GEP protects both embedding and residual; projected DP-SGD discards residual under structural assumptions. Public pretraining supplies utility but needs a public-only control. PATE protects released answers, not disclosed teachers. SCAFFOLD motivates correction of client drift using control variates; our public class-conditional construction is not its algorithm and has no inherited DP theorem. [Exact search/access log](../literature_review/followup_public_reference_log.json): five query executions and scoped primary reads; frozen31-family systematic counts unchanged.

## Frozen protocol and actual channel

[Original protocol](2026-10-08_public_residual_protocol.md): eight registered weighted slots with retained dummy noise, peer knows its own coins, whole dataset contribution versus empty. Conditional attacker additionally knows target/background datasets. The information contract now grants every method512LABELLED public examples; it differs from the original unlabelled-prototype pilot.

The public reference is trained for80/320/1280steps with six logit multipliers and selected using OLDdevelopment labels.1280steps×1.5was chosen. Public-data CE Hessian eigenvectors supply nested1/3/12/51-dimensional spaces; Euclidean/Fisher metrics differ in coordinate scaling. Geometry is public/fixed CONDITIONAL on the offline reference selection, not solely512-derived for the entire research algorithm. Private transforms/configuration selection remain unaccounted.

Absolute projected teachers, centered zero-initialized teachers and constrained public-initialized fine-tuning were compared with five previously selected legacy descriptor families, reoptimized over public caps/gains. Absolute decoder preserves the public reference outside the subspace. Before clipping its all-IN learner equals the centered learner; OUT absolute imputes0inside the subspace while centered/fine imputes the public reference. This changes what is learned, not merely where noise is placed.

Joint clip the local encoded vector to PUBLIC C, then weight1/8. Independent Gaussian shares have variance sigma²/7, sigma=C/(8sqrt(2)Phi^-1(q)). Seven unknown shares give conditional optimal AUC Phi(norm(target)/(sqrt(2)sigma))<=q for any fixed bounded contribution. Unlike the preceding study, noise is not calibrated to a private maximum norm. This is a fixed-channel dataset-versus-empty certificate; arbitrary dataset replacement has twice the sensitivity. Full aggregate variance is8/7sigma². The central identical-query control has the same unknown residual sigma, but smaller total variance. Individually observed client uploads are a different contract.

Development searched **16,758settings**,6rows×2,793,32draws per each of eight primary target/world strata. All48family choices were frozen. An audit-directed [16-record public-only offset supplement](2026-10-08_public_residual_control_addendum.md) was selected on old development BEFORE reserve extraction; it was not originally predeclared. It independently searches B−aP_dB and scalar offsets, including duplicates. The chosen d3/a0.3adjustment uses no private contribution. All model/config choices and hash dependencies preceded confirmation.

Confirmation: **512new reserve images**,128perclass; **456remain unused**. Same saved private bank/seeds, not new client populations.54arms/432target rows, all8target alternatives,4,096fresh attack draws perworld and256separate utility draws perworld. Primary targets0–3cover four dominant classes; peer7excepttarget7uses6. No fresh test choices or law changes.

## Fresh protected-utility comparison

CE/accuracy average equal OUT/IN and four primary targets, then three seeds. AUC averages target-wise ordinary oriented ROC values; not pooled population AUC. Public-only outputs are independent of target contribution, hence AUC0.5.

| Target | Arm | Mean CE | Accuracy | Mean measured AUC |
| ---: | --- | ---: | ---: | ---: |
| 0.55 | Fine-tuned residual | 0.451109 | 85.75% | 0.551598 |
| 0.55 | Projected absolute | 0.447609 | 86.51% | 0.550602 |
| 0.55 | Centered old teacher | 0.450218 | 85.78% | 0.549153 |
| 0.55 | Legacy model | 0.455047 | 85.72% | 0.548750 |
| 0.55 | Legacy logits | 0.455078 | 85.73% | 0.552678 |
| 0.55 | Legacy probabilities | 0.455202 | 85.65% | 0.553544 |
| 0.55 | Legacy votes | 0.455229 | 85.63% | 0.552682 |
| 0.55 | Central same residual | 0.451030 | 85.75% | 0.551768 |
| 0.65 | Fine-tuned residual | 0.448510 | 85.89% | 0.648985 |
| 0.65 | Projected absolute | 0.447598 | 86.52% | 0.645282 |
| 0.65 | Centered old teacher | 0.448482 | 85.94% | 0.652275 |
| 0.65 | Legacy model | 0.455049 | 85.74% | 0.653706 |
| 0.65 | Legacy logits | 0.455067 | 85.74% | 0.649298 |
| 0.65 | Legacy probabilities | 0.455162 | 85.70% | 0.649489 |
| 0.65 | Legacy votes | 0.455143 | 85.69% | 0.648784 |
| 0.65 | Central same residual | 0.448325 | 85.91% | 0.649557 |
| — | Public B | 0.451861 | 85.74% | 0.500000 |
| — | Frozen public decoder offset | 0.447582 | 86.52% | 0.500000 |

The high accuracy comes largely from labelled public learning. At0.55the projected absolute arm has CE0.447609 versus public-only offset0.447582, and essentially the same86.52%accuracy. It loses0.000024–0.000034CE in every seed; all conditional noise intervals favor the deterministic public offset. Its gain over B therefore establishes no incremental private information benefit.

Positive gain below means reference CE minus fine CE. Intervals average eight fixed strata's paired noise variances; they are not population or selection-adjusted intervals.

| Target | Seed | Fine gain vs public B | Fine gain vs absolute [95% interval] | Fine gain vs stronger public offset [95% interval] |
| ---: | ---: | ---: | --- | --- |
| 0.55 | 42 | 0.000994 | -0.003251 [-0.003377, -0.003126] | -0.003286 [-0.003419, -0.003152] |
| 0.55 | 43 | 0.000305 | -0.003950 [-0.004029, -0.003872] | -0.003974 [-0.004061, -0.003888] |
| 0.55 | 44 | 0.000957 | -0.003299 [-0.003394, -0.003205] | -0.003323 [-0.003429, -0.003217] |
| 0.65 | 42 | 0.004562 | 0.000299 [0.000145, 0.000452] | 0.000282 [0.000126, 0.000438] |
| 0.65 | 43 | 0.001782 | -0.002483 [-0.002649, -0.002317] | -0.002498 [-0.002667, -0.002329] |
| 0.65 | 44 | 0.003712 | -0.000550 [-0.000652, -0.000448] | -0.000568 [-0.000673, -0.000462] |

The original gate—fine beats both B and strongest alternative in all three seeds, lower interval>0.001—fails at0.55. Strongest alternative was absolute in all six development rows. At0.65, seed42's gain versus the stronger public offset is0.000282, interval[0.000126,0.000438], below0.001; the other seeds lose. No robust positive gate.

All432theoretical Gaussian AUCs obey the public-cap ceiling. The empirical fine ranges across all8targets are0.533121–0.559551atq0.55and0.634011–0.662353atq0.65. These finite draw deviations are not uniform exact empirical bounds; maximum Gaussian expectation deviation is3.036standard errors. Central noise improves fine CE slightly; it does not rescue the primary gate or show a client-placement advantage.

## A useful mathematical stopping rule

For the three q0.55SELECTED clipped fine queries, even noiseless equal-world CE is0.449834/0.451203/0.450399, worse than deterministic public offset0.447582. Their decoder is affine. Conditional on fixed data/query/decoder, any additive parameter noise with zero conditional mean and finite first moment satisfies

`E[CE(theta_noiseless + L*noise)] >= CE(theta_noiseless)`.

CE is convex in linear classifier parameters; applying Jensen perworld/image set and averaging strata preserves the inequality. Consequently, changing ONLY the zero-mean additive density cannot repair THESE selected queries' expected CE deficit. Private-data-dependent covariance does not evade this conclusion if conditional mean stays zero, and its privacy selection cost is a separate obligation. This is not an impossibility result for other queries, gains, nonlinear decoders, deliberate bias, selections, information contracts or client counts. At0.65some selected noiseless queries do have headroom. The private bank is not proved useless.

## Development-only distribution-aware signal diagnostic

After the above test, a separately frozen [post-hoc diagnostic](2026-10-08_residual_signal_diagnostic_protocol.md) used ONLY the old development/private/public roles—no more reserve features, noise simulator or protected mechanism evaluation.

At fixed public B, compute public class-conditional gradient means gamma_k. Each client computes its own empirical class fractions p_ik internally and removes the predicted mixture:

`r_i = g_i - sum_k p_ik*gamma_k`.

With public class prior pi and fixed weights1/8:

`g_public + mean_i r_i = mean_i g_i + sum_k (pi_k - mean_i p_ik)*gamma_k`.

In these all-IN federations both priors are exactly0.25each, so aggregate restoration holds before clipping. The subtracted term changes with each client's class distribution, unlike a single parameter reference B. The whole-client residual can be clipped as one vector with a public bound in a future construction; observed norms alone cannot calibrate a privacy certificate.

| Seed | Raw mean client norm | Class-centered mean norm | Reduction of mean norm | IN identity error | Pooled gain vs frozen public offset (development) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.332195 | 0.041384 | 87.54% | 6.94e-18 | 0.006341 |
| 43 | 0.351934 | 0.048875 | 86.11% | 6.94e-18 | 0.006119 |
| 44 | 0.347000 | 0.055400 | 84.03% | 6.94e-18 | 0.000699 |

Across individual clients, centered/raw norm ratios range0.0496–0.4025. This is a promising removal of label-mixture nuisance, not evidence that low noise protects useful learning. Public/private conditional shift can make the subtracted reference misaligned and increase norms; clipping can destroy restoration. OUT changes the pooled prior and the contribution being learned: the public gradient remains while the absent residual must be0. Both worlds need explicit utility comparisons in a future mechanism.

Unprotected centrally pooled public+private training has some development headroom (table), but uses raw private pooling/additional optimization and has no held-out or privacy status. A stronger public-gradient-only step reaches development CE0.479194. At their selected steps, raw/private gradient development gains over that public step are only0.001115/0.000147/0.000372; there is no three-seed0.001screen. Centered-only correction is inconsistent. Retain this counter-evidence alongside the norm reductions; do not promote the residual as a proven defense.

## Concrete next construction question

Promote **public class-conditional control variates combined with each client's internal empirical distribution** to the next bounded feasibility question, rather than fine-tuned model subtraction or another noise density. First assess conditional-shift and prior-mismatch stress and the real noiseless/noise-aware task gain. Then define a single bounded complete-client query, explicit empty residual, public calibration and exact observer law; compare raw gradient, balanced gradient, public-only optimization and identical-query central controls. Keep internally used priors/covariance separate from public metadata or privately selected noise shapes. Any individualized noise law still needs complete-law accounting and an estimation-cost/novelty test.

The promising knowledge is that class-mixture effects can dominate individual query norms while canceling in the aggregate. That offers a concrete distribution-dependent construction target. Utility at near-chance whole-client protection remains unresolved; changing cohort size or data perclient requires an explicit new setting and independent clients, not duplicates or a smaller privacy unit. No broad Flower/CNN sweep is justified yet.

## Verification and artifact boundary

The independent development-only signal auditor verifies874numeric values, maxerror6.66e-16, independently retraining all54pooled choices and recomputing105direction/step scores,72gradient vectors, norms and identities without accessing reserve images. [Signal receipt](../../results/client_specific_noise/public_residual_signal_artifact_audit.json). Compared with the stronger public-gradient step, pooled oracle headroom is0.006080/0.005858/0.000437, again not a three-seed0.001screen. Ratio-of-mean norms gives84–88%reduction; averaging individual clients’ percentage reductions gives82–86%, a different statistic.

Six meaningful tests pass: public Hessian finite differences; encoder/inverse geometry; whole-vector public bound and Gaussian AUC ceiling; empty fine/residual semantics; all-IN offset identity/OUT imputation; explicit centered-query empty mask. Initial missing-module failures preceded implementation; a failing empty-mask test preceded the safeguard. Preliminary development was stopped with exit130 before artifacts/reserve, then the unchanged numerical grid/streams restarted with dependency/split/empty checks. Preliminary output is excluded, not another replication.

Independent contract and Jensen reviews completed. The separate [artifact auditor](../../results/client_specific_noise/public_residual_artifact_audit.json) imports no experiment/statistics and reconstructs public teachers/Hessian,48fine teacher arrays, all configurations, selections and complete score arithmetic: **260,338numeric checks**, maxerror1.14e-14,432Gaussian expectations,16supplementary public models and54supplementary contrasts, exact role/slice exclusions. Default suite **293passed,5deselected**. The unselected d51/gain1public offset is analytically zero but contains cancellation roundoff; its argmax accuracy is numerically unstable. It was not selected or used in conclusions; the auditor scores actual saved floats after verifying model reconstruction. CE-based choices and the selected d3control are unaffected.

Raw/private teacher/calibration/operator artifacts are not modeled peer transcripts. Their tuning/publication, benchmark-dependent selection and finite-precision deployment are unaccounted; no end-to-end DP release, novelty, historical metric-privacy superiority or overall experiment completion follows. [Reproduction/artifact guide](../../results/client_specific_noise/README.md). No jobs remain after final verification.
