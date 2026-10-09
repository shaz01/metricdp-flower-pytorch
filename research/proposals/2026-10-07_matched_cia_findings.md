# Matched CIA strength: radial law rejected as the current improvement

2026-10-07; active branch `feature/client-specific-noise`. Owner: “lets move with the next step dont stop until you result on something”. This note documents the verified bounded comparison and a direction-changing negative result. It does not declare the overall experiment finished, establish a deployed mechanism or authorize merging.

## Result and implication

**Radial L2-Laplace does not provide a consistent improvement over Gaussian once the voting query and CIA strength are matched.** Near chance (target AUC0.55), radial voting loses to the strongest Gaussian non-voting controls in every seed and its CE is worse than uniform prediction. At the more permissive0.80target, voting beats those controls, but Gaussian on the same query matches the radial result within the tested noise intervals. The useful effect remains query/representation construction at substantial allowed leakage; it is not a superior radial noise density.

Stop promoting the fixed-epsilon radial/Gaussian difference as a protection improvement. Earlier radial noise had lower CIA AUC partly because it added more effective noise in the48-dimensional query. The present matched-query, matched-AUC and matched-variance controls remove that optimistic interpretation. This rejects the tested law/constructor combination as our near-chance candidate; it is not a universal impossibility theorem for non-Gaussian mechanisms.

## How this comparison works

For each local teacher representation, jointly clip its weighted descriptor and compute M, the largest norm over the eight saved clients. Hold this value, noise parameters and learner settings fixed in both worlds. A target's IN query is replaced with zero in OUT; its slot and noise share remain. A passive peer knows only its own upload/coins. The strong known-alternative attacker also knows all underlying target/background datasets as auxiliary information. Its aggregate score removes the peer's upload and uses the exact Gaussian or radial likelihood ranking of the remaining seven-share channel.

For Gaussian, residual std `M/(sqrt(2)*Phi^-1(q))` gives optimal AUC at most q for every saved shift norm<=M. For radial noise, calibrate a normalized unit-shift scale by16,384independent IN/OUT draws per dimension/target, then verify with fresh draws on ALL8saved target alternatives. Radial matching is estimated, not a proven AUC bound. Rotational invariance transfers the unit-shift calibration to each query; real smaller shift ratios are measured explicitly. Noise scales are never recomputed from the OUT data.

Private M and development choices are **offline, unaccounted research calibration**, not a privately constructed deployable noise distribution. The risk target is a conditional saved-federation ceiling; it is not a common epsilon budget or a guarantee for arbitrary future datasets. The radius/full sensitivity bound can exceed M. Choosing private clipping/shape/dimension, publishing operator artifacts and finite-precision fixed-seed sampling remain outside any end-to-end guarantee.

## Frozen development and fresh confirmation

The [protocol](2026-10-07_matched_cia_protocol.md) preceded matched-law utility results. For each seed42/43/44, target0.55/0.65/0.80and lawGaussian/radial, searched all972transform/configuration combinations (raw, prior-corrected, balanced-loss; all five descriptor families, original radius/prototype/ridge/step grids),32paired development draws. **17,496settings** over18rows, selected per-family minima and strongest non-vote controls, were saved before extracting fresh utility features. Thus the comparison does not reuse the old epsilon8-selected configurations as supposedly optimal near-chance controls.

Confirmation consumed **1,024previously unused reserve images**,256perclass, disjoint from every previous role and both earlier utility slices. **968reserve images remain unused.** The private bank/public support and three partition seeds are reused, so this is fresh utility/noise confirmation, not independent client populations. Each of117arms was attacked on8target alternatives: **936rows**,4,096fresh releases per world. Independent utility streams supply256fresh draws per world. Slots0–3are the primary class-covering set; slots4–7verify calibration. Peer7 is used except target7, whose peer is6.

For each radial-selected vote query, add Gaussian calibrated at the same risk, Gaussian matching its residual coordinate variance `(d+1)s²`, and central radial on the IDENTICAL query/decoder/residual law. These settings are frozen from development, not test-selected. Central has7/8of client aggregate marginal variance because its coins are entirely unknown to the peer. The variance-matched Gaussian is not automatically AUC-matched; its theoretical/measured AUC is reported separately.

## Primary utility and leakage summary

CE/accuracy equally average IN and OUT and the four primary targets, then three saved seeds. AUC is a descriptive average of conditional target-wise values, not pooled client-population AUC or a Pareto frontier. Gaussian uses a ceiling over all saved clients, so actual target values may be smaller. Radial AUC is Monte Carlo calibrated; all-target voting ranges are0.539787–0.561123at0.55,0.637373–0.661864at0.65and0.786795–0.807692at0.80, within the predeclared0.02compatibility tolerance.

| Target AUC ceiling | Arm | CE | Accuracy | Mean measured AUC |
| ---: | --- | ---: | ---: | ---: |
| 0.55 | gaussian / votes | 1.431849 | 30.58% | 0.550814 |
| 0.55 | radial / votes | 1.430741 | 30.94% | 0.552916 |
| 0.55 | radial / gaussian_same_query | 1.423729 | 30.83% | 0.551835 |
| 0.55 | radial / gaussian_variance_match | 1.432867 | 30.66% | 0.550917 |
| 0.55 | radial / central_same_query | 1.416809 | 31.27% | 0.552459 |
| 0.55 | gaussian / strongest_nonvote | 1.354335 | 34.01% | 0.552539 |
| 0.65 | gaussian / votes | 1.216644 | 47.05% | 0.650903 |
| 0.65 | radial / votes | 1.217153 | 47.16% | 0.652232 |
| 0.65 | radial / gaussian_same_query | 1.216644 | 47.05% | 0.650903 |
| 0.65 | radial / gaussian_variance_match | 1.217635 | 46.99% | 0.650338 |
| 0.65 | radial / central_same_query | 1.201611 | 48.13% | 0.651746 |
| 0.65 | gaussian / strongest_nonvote | 1.189673 | 47.87% | 0.650757 |
| 0.80 | gaussian / votes | 0.969432 | 62.36% | 0.799495 |
| 0.80 | radial / votes | 0.970266 | 62.32% | 0.801310 |
| 0.80 | radial / gaussian_same_query | 0.969432 | 62.36% | 0.799495 |
| 0.80 | radial / gaussian_variance_match | 0.969890 | 62.33% | 0.798849 |
| 0.80 | radial / central_same_query | 0.958949 | 62.97% | 0.800043 |
| 0.80 | gaussian / strongest_nonvote | 1.094254 | 55.99% | 0.770483 |

The uniform control has CE1.386294 and accuracy25%. On the same fresh slice the teacher trained on512LABELLED public examples has CE0.984211 and accuracy72.75%. Its auxiliary labels exceed those used by the core unlabelled-prototype constructor; it is a separate strong reference, not an equal-information comparator. Private voting does not establish superior accuracy over this public-labelled learner.

## Paired fixed-stratum comparisons

Gain means reference CE minus radial voting CE; positive favors radial. Each mean is over four primary targets × two worlds. Its standard error averages the eight fixed strata's paired noise variances, rather than treating target-mean differences as random population variation. Intervals are conditional noise intervals; they do not adjust for all searched configurations or prove population effects.

| Target | Seed | Radial vote gain vs same-query Gaussian [95% interval] | Gain vs strongest Gaussian non-vote [95% interval] |
| ---: | ---: | --- | --- |
| 0.55 | 42 | 0.001154 [-0.000358, 0.002666] | -0.070604 [-0.075411, -0.065798] |
| 0.55 | 43 | 0.000231 [-0.001396, 0.001858] | -0.066205 [-0.070781, -0.061628] |
| 0.55 | 44 | -0.022418 [-0.028837, -0.016000] | -0.092407 [-0.103283, -0.081530] |
| 0.65 | 42 | -0.000340 [-0.002001, 0.001321] | -0.038497 [-0.040431, -0.036563] |
| 0.65 | 43 | 0.000073 [-0.002852, 0.002998] | -0.014129 [-0.018611, -0.009647] |
| 0.65 | 44 | -0.001261 [-0.004174, 0.001652] | -0.029814 [-0.034305, -0.025323] |
| 0.80 | 42 | -0.001434 [-0.003330, 0.000462] | 0.167365 [0.160997, 0.173734] |
| 0.80 | 43 | -0.001422 [-0.003284, 0.000439] | 0.106994 [0.103456, 0.110532] |
| 0.80 | 44 | 0.000352 [-0.001563, 0.002268] | 0.097605 [0.094063, 0.101147] |

The primary0.55gate required radial votes to beat BOTH same-query Gaussian and strongest Gaussian non-vote in all three seeds, lower interval endpoint>0.001. It fails clearly. Strong Gaussian logits achieve CE1.345380/1.346923/1.370702 and accuracy34.23/33.83/33.98%; radial voting CE1.415985/1.413128/1.463109 is worse even than uniform. Low leakage can preserve a little signal in smooth queries, but the tested voting route loses useful learning.

At0.65, balanced-loss probability controls beat radial votes by0.014129–0.038497CE. At0.80, radial voting improves0.097605–0.167365CE over the selected Gaussian non-voting controls, with all lower noise intervals>0.001. Yet all three radial versus same-query Gaussian intervals cross zero. No three-seed positive gate survives the variance-matched Gaussian comparison either. Central same-query radial consistently improves CE. Neither client placement nor this density supplies the demonstrated high-leakage representation gain.

## What we learned about construction

The nonlinear voting query is not the best representation at every privacy/attack level. Lower-leakage development selects smooth logits/probabilities as the strong controls, whereas voting becomes useful at0.80. Two of the three Gaussian0.55control choices use4prototypes (12contrasts), while the third uses16; compression alone is not yet a proven general remedy. Teacher balancing remains useful, but privacy requires examining the complete representation and observable law rather than hoping a different marginal noise identity will recover signal.

The immediate research question should therefore move to **task-aware compression or public-reference residual descriptors**, tested against the smooth-query controls under fixed public calibration. A reference residual would describe what the client adds beyond an explicitly public/protected baseline, with an empty client adding zero; whether that residual is useful and cheaper to protect is an untested hypothesis. Any use of public labels requires a declared auxiliary-information contract and a strong public-only learner—otherwise one could incorrectly credit private data for utility supplied by the reference.

Cohort size is a separate necessary feasibility control: eight fixed contributors may have insufficient aggregate signal at near-chance whole-client protection. More contributors or different per-client data size change the learning/adjacency setting; do not manufacture a gain by duplicating clients or quietly shrinking the privacy unit. A large Flower/CNN sweep or another loosely motivated continuous noise law is premature. Before the next constructor, use development-only information diagnostics to specify the useful residual/subspace and its public sensitivity bound, review nearest one-shot/private-distillation/public-reference work, then freeze a new bounded held-out test.

No new noise density, private-estimation solution or novelty is claimed. This evidence narrows where a useful client-specific distribution constructor could matter: preserving task-relevant information before noise, with honest accounting of the estimation/reference and full observer law.

## Verification and reproduction

Five meaningful new tests pass: Gaussian risk mapping, whole-vector radial moments and Gaussian variance control, norm/noise equivariance, efficient rank statistic agreement with all-pairs AUC, retained dummy noise, peer residual versus central coins. The initial variance test pooled unequal signal means across coordinates; it was corrected to estimate per-coordinate variance across draws, with no sampler change. The full default suite passes **287tests,5deselected**.

An independent agent checked26,424numeric values with zero discrepancy, reconstructed clipping/M/law scales for all17,496configurations, verified all936frozen rows and the new/remaining reserve partition, and independently recomputed the primary fixed-stratum gains. A separate saved-artifact auditor imports no simulator and verifies **33,174numeric comparisons**, max error8.673617379884035e-19,504Gaussian theoretical AUC expectations (max3.681standard errors),36stratified contrast rows and artifact/source hashes. [Audit receipt](../../results/client_specific_noise/matched_cia_artifact_audit.json), [contrasts](../../results/client_specific_noise/matched_cia_contrasts.json), [reproduction commands](../../results/client_specific_noise/README.md).

Confirmation was deterministically repeated to verify the final pipeline including the predeclared uniform/public-labelled references; this is not counted as an additional independent replication. No jobs remain running. Overall experiment remains active on its branch; this note records a research chunk and recommendation, not the owner's completion decision.
