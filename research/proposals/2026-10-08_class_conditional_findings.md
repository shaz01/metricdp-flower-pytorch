# Class-conditional residual: verified bounded findings and research steering

2026-10-08, `feature/client-specific-noise`. Owner: “lets move to that dont stop”. This records a bounded comparison within the active experiment, not overall completion, a root-level final report, or merge authorization.

The primary near-chance utility gate **fails**. Client empirical-prior centering still reduces gradient norms, but useful protected correction does not reliably beat strong public-only or raw/balanced controls. The apparent refreshed-mask gain is explained by a public gradient step; adding the protected client residual makes that matched public model worse in all3seeds. This changes the roadmap from another density variant to measuring genuinely useful private signal first.

## Reconstructable evidence

[Primary frozen protocol](2026-10-08_class_conditional_protocol.md), [plain-language constructor](class_conditional_mechanism_explained.md), [post-hoc attribution protocol](2026-10-08_class_conditional_attribution_protocol.md). Calculators: `research/calculations/class_conditional_probe.py` and `class_conditional_attribution.py`. Primary artifacts: `results/client_specific_noise/class_conditional_development.json`/`.npz`, `class_conditional_evaluation.json`/`.npz`, `class_conditional_artifact_audit.json`; supplementary `class_conditional_attribution.json` and `class_conditional_attribution_audit.json`. No new literature searches or frozen31-family corpus amendments.

Development8640settings,32draws×8strata each,36frozen mode choices,380public-only candidates,18query/stress routes. Three saved label-stress federations; public512, development1024; eight clients with256fit examples each in baseline. Public caps(.003,.01,.03,.1,.3), six positive step sizes, four dimensions and two metrics. Primaryq=.55, secondaryq=.65baseline only. Fresh320images(80/class),136unused remain;1176target cells,2048fresh exact-LR draws/world and128independent utility draws/world. Primary summaries average targets0–3 and equal OUT/IN strata; all8targets saved/audited. Shift conditions reuse images/clients, not independent replications.

Six private constructions: raw; exact target-balanced; fixed public-mean centering; empirical-prior class-centering; same residual without public restoration; target-balanced centering. Same-query central control included. Before reserve, independent development audit150,722checks/max8.88e-16. Full primary audit6,235,039numeric checks/max8.88e-16 independently reconstructs380public models,18queries,2352attack-score vectors and2352weighted/unweighted utility vectors, all selections/splits/hashes,1176Gaussian ceilings/ROC summaries. Largest empirical AUC departure2.843standard errors; finite empirical AUC can exceed a theoreticalq ceiling. Independent post-hoc attribution audit35,108numeric checks/max1.11e-16 verifies147matched offsets/gains/norms and1176noiseless strata, with unchanged320/136indices and repository-relative hash paths. Six meaningful kernel tests plus full default suite299passed/5deselected.

Preliminary development was interrupted(exit130) before artifacts/reserve to add a bank-hash guard, then same grid/streams restarted. This is not a replication. Supplementary attribution was frozen AFTER seeing primary results, uses only the SAME320images, and changes no choices or original gate. A repository-relative hash-key correction regenerated that diagnostic without numerical changes or new images.

## Primary baseline result

Strong frozen public control: full public-gradient step eta3, freshCE0.5754380234,accuracy82.8125%. Three-seed means below are fixed-cohort/equal-stratum summaries, not population estimates.

| Arm | Mean CE q.55 | Accuracy % | Mean optimal empirical ROC AUC |
|---|---:|---:|---:|
| raw | 0.575242417 | 82.72563 | 0.547571 |
| target_balanced | 0.575198410 | 82.69368 | 0.549806 |
| fixed_center | 0.575487479 | 82.79429 | 0.547416 |
| class_center | 0.575425770 | 82.81860 | 0.549648 |
| center_only | 0.578784963 | 82.67008 | 0.544064 |
| target_center | 0.575471359 | 82.79663 | 0.549732 |
| central_class_center | 0.575420737 | 82.81982 | 0.548246 |

| Seed | Class_center CE | Gain over public | Fixed-noise 95% interval | Gain over development-selected strongest nonprimary |
|---|---:|---:|---|---:|
| 42 | 0.575318939 | +0.000119084 | [+0.000090782,+0.000147387] | -0.000197727 (target_balanced) |
| 43 | 0.575509934 | -0.000071910 | [-0.000089883,-0.000053938] | -0.000249297 (raw) |
| 44 | 0.575448438 | -0.000010415 | [-0.000027906,+0.000007077] | -0.000241414 (raw) |

The retained threshold is gain>.001 in all3seeds versus both public-only and strong private controls. It fails in magnitude and consistency. Class_center baseline cap selects .003 in all3; seed42 selectsd3/Fisher,43d51/Euclidean,44d12/Euclidean; all eta3. The lower cap boundary means this rejects the tested grids/selected arms, not every possible cap/geometry. Central noise makes only a very small improvement and cannot rescue this gate.

Secondaryq.65 class_center gains over strong public are +.000556920/−.000175830/+.000016950(seeds42/43/44); again no3seedgate. Target-balanced has private gain over its weaker matched offset in seeds42/44, but not43; this does not validate primary residual or noise-density novelty.

## Distribution and public-reference stress

Prior stress genuinely thins each private dataset, with fixed1/8 slot weights. Pooled-record class0 prior≈.403785, mean-client class0 prior≈.318244; external evaluation prior=.4, all separately recorded. No duplicate examples/artificial new clients. Known noninvertible mask removes four pooled image features from every private/evaluation/public control; bias stays. Private configurations transfer without retuning; mask references are separately stale or public-refreshed.

Gains below are positive when protected class_center reduces CE. Public winner means the old-development-selected control; matched offset means B−eta*its exact public restoration with NO private correction/no noise.

| Scenario/reference | Seed | Protected CE | Gain over frozen public winner | Gain over matched public offset |
|---|---:|---:|---:|---:|
| prior/stale | 42 | 0.590343502 | -0.011407365 | +0.000235464 |
| prior/stale | 43 | 0.590599795 | -0.011663658 | -0.000020829 |
| prior/stale | 44 | 0.590634092 | -0.011697955 | -0.000055126 |
| mask/stale | 42 | 0.595219113 | +0.000028833 | +0.000639517 |
| mask/stale | 43 | 0.595491316 | -0.000243370 | +0.000367314 |
| mask/stale | 44 | 0.595523646 | -0.000275700 | +0.000334984 |
| mask/refreshed | 42 | 0.590494826 | +0.004753120 | -0.000141080 |
| mask/refreshed | 43 | 0.590490605 | +0.004757341 | -0.000136859 |
| mask/refreshed | 44 | 0.590450737 | +0.004797209 | -0.000096991 |
| both/stale | 42 | 0.595983591 | -0.000939750 | +0.000410046 |
| both/stale | 43 | 0.596260104 | -0.001216263 | +0.000133533 |
| both/stale | 44 | 0.596226784 | -0.001182943 | +0.000166853 |
| both/refreshed | 42 | 0.595175794 | -0.000131953 | +0.000020241 |
| both/refreshed | 43 | 0.595277611 | -0.000233771 | -0.000081576 |
| both/refreshed | 44 | 0.595309370 | -0.000265530 | -0.000113335 |

Refreshed mask seems to beat the frozen public refit by .004753–.004797CE. Its exact public offset already reaches CE0.5903537459,accuracy81.5625%; adding protected residual worsens CE by .000097–.000141. That offset was already a pre-reserve public candidate; its old-development rank did not select it. This is public adaptation/control-selection generalization, not a useful private-query gain. CE and accuracy differ: frozen refit accuracy82.8125% is higher, so this is not a universal utility win.

Stale mask corrections are directionally useful versus their own stale offset (+.000335–.000640CE), but below.001 and worse than refreshed public-only learning. Refreshing gamma markedly lowers norms yet turns private contribution harmful in all3seeds. Prior stress loses about.0114–.0117CE versus its frozen public control, despite comparable/higher accuracy; posterior/prior correction and step transfer affect the learning objective. Combined shift does not establish robustness.

## What is now known

1. Removing expected label-mix gradients can greatly shrink uploads. It does not prove those remaining gradients are useful, accurately estimated or aligned with held-out loss.
2. Public restoration carries real learning information. Under OUT it imputes a public contribution; under prior mismatch it changes the objective. Credit its utility to public information, not client noise construction.
3. A refreshed public reference can reduce residual norms and improve the whole model while the actual private correction worsens it. This is the concrete confound the follow-up resolves.
4. The public-cap conditional CIA channel meets its fixed-settings AUC bound; utility usefulness and privacy accounting are separate. Full student parameters permit recovery of the affine released descriptor. Whole-client contribution/dummy, retained noise, fixedslots/weights and peer-conditioned coins remain the contract. No arbitrary replacement/physical-connection secrecy/individually exposed-upload guarantee, historical Flower/CNN metric superiority, end-to-end private tuning, finite precision, population or novelty claim.

## Active research roadmap

Do not promote this tested one-step empirical-prior residual as the proposed defense, and do not spend another reserve slice on a density change for these selected affine queries. First establish **private-signal headroom** on development-only data under a transparently varied public reference budget/feature representation. Compare public-only learning to a strong unprotected private oracle, then separate sampling error, discarded directions, clipping bias and noise penalty. Require useful private correction across genuinely independent client cohorts before another protected confirmation.

A concrete next bounded question: with public reference budgets32/128/512 (disjoint, fixed subsets) and unchanged evaluation objective, where does class-conditional private correction improve the strongest matched public learner by>.001 consistently? Retain the512-example benchmark and strongest control as the anchor. Smaller budgets define a new public-resource setting. Reference richness must be a declared setting, not weakened after results to manufacture a win. If no full-information headroom remains, move to a richer task/representation or genuinely shifted client signal. Only after headroom passes should a distribution estimator or non-Gaussian law be built and compared at the same attacker contract. This next diagnostic is proposed, not run or certified.

136unused reserve images remain; preserve them until a stronger developmental case exists. Overall client-specific privacy research remains active on the feature branch.
