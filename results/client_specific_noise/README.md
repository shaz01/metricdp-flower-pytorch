# Client-specific noise construction calculations

`analytical_heterogeneity.json` contains deterministic two-dimensional reconstruction-risk calculations, not trained-model results or CIA scores. No FL experiment is declared finished by these artifacts.

Regenerate with `uv run python research/calculations/heterogeneous_profile_risk.py` from the repository root. Assumptions, boundaries and interpretation are in [the analytical findings note](../../research/proposals/2026-10-04_heterogeneity_findings.md). The saved grid search does not certify continuous global optimality.

`protected_history_two_round.json` contains two-round stationary-update risk calculations, complete controls and fixed-seed sampler/formula validation. Regenerate with `uv run python research/calculations/protected_history_two_round.py`. Read [the findings and limits](../../research/proposals/2026-10-04_protected_history_findings.md) before interpreting its conditional improvements.

`dynamic_quadratic_probe.json` and `dynamic_quadratic_trial_losses.npz` record a bounded changing-gradient learning spike: development-selected settings and fresh paired trial losses. Regenerate with `uv run python research/calculations/dynamic_quadratic_probe.py`. Read [the held-out findings](../../research/proposals/2026-10-04_dynamic_quadratic_findings.md); no CIA evaluation or experiment-completion declaration is implied.

`rotated_residual_probe.json` and `rotated_residual_trial_losses.npz` record tonight’s rotated-bank comparison, with conditional and sampled utility endpoints. Regenerate with `uv run python research/calculations/rotated_residual_probe.py`. Read [the findings and stopping instruction](../../research/proposals/2026-10-04_rotated_residual_findings.md).

`client_geometry_audit.json` and `client_geometry_audit_matrices.npz` record the offline Fashion-MNIST fixed-feature geometry audit and raw-information two-bias-coordinate oracle comparison. Regenerate with `uv run python research/calculations/client_geometry_audit.py` using the local Arrow cache (`--cache-dir` overrides it); source hashes are in JSON. The archive stores matrices, model heads, weights and dataset/split indices, not images. Read [the findings and limits](../../research/proposals/2026-10-05_client_geometry_findings.md): persistent stress-regime covariance differences but no 0.001 CE gate pass; epsilon labels do not certify privacy, and no CIA or experiment-completion claim is implied.

`broader_geometry_oracle.json` and `broader_geometry_oracle_trials.npz` contain the development-tuned 3/51-dimensional raw-information oracle, 36 records/108 comparisons, fresh test indices/bases and paired trial losses. Run `uv run python -m research.calculations.broader_geometry_oracle` with the same offline Arrow cache. Read [frozen protocol](../../research/proposals/2026-10-05_broader_oracle_protocol.md) and [findings](../../research/proposals/2026-10-05_broader_oracle_findings.md): marginal three-bias headroom in label stress, no full-head gate pass, no DP/CIA claim.

`public_slot_profile_probe.json` and `public_slot_profile_trials.npz` preserve24 roster/seed/checkpoint/regime records,72 noise-label comparisons and2048paired losses per four-arm comparison. Frozen prior profile/radius/shrink are tested on a third disjoint test slice and a cyclic roster shift. Run `uv run python -m research.calculations.public_slot_profile_probe` using the saved geometry audit and offline Arrow cache. Read [protocol](../../research/proposals/2026-10-05_public_slot_protocol.md) and [findings](../../research/proposals/2026-10-05_public_slot_findings.md): exact aligned oracle match, fresh gains below gate, harmful slot reassignment; no DP/CIA claim.

`local_selector_cost.json` records exact development-quadratic means/covariances and private-selector split curves for18records/54cells. Run `uv run python -m research.calculations.local_selector_cost` using the saved audit and offline training Arrow. No test data, noise simulation or training is used. Read [protocol](../../research/proposals/2026-10-05_local_selector_protocol.md) and [conditional law/cost findings](../../research/proposals/2026-10-05_local_selector_findings.md): permutation-invariant local rule, small partial-accounting gains belowgate, no certified training/CIA result.

`hidden_mixture_audit.json` preserves stable tail-density witnesses, feasible label-gradient limiting witnesses and54 fair contribution/dummy-adjacency development cost cases. Run `uv run python -m research.calculations.hidden_mixture_audit` with the saved geometry audit and offline training Arrow. Read [protocol](../../research/proposals/2026-10-05_hidden_mixture_protocol.md) and [analytic findings](../../research/proposals/2026-10-05_hidden_mixture_findings.md): hiding the selector gives no uniform saving for the present law; dummy calibration changes all controls fairly; no heldout/privacy-training/CIA claim.

`protected_real_history_probe.json` and `protected_real_history_trials.npz` preserve18 saved-update replay records/54 budget cases, three arms×16384 fresh conditional probe values plus an exactly integrated one-release reference. Run `uv run python -m research.calculations.protected_real_history_probe` with the saved geometry archive and cached training Arrow; `--self-check` runs bounded mathematical checks. Read [protocol](../../research/proposals/2026-10-05_protected_real_history_protocol.md) and [findings](../../research/proposals/2026-10-05_protected_real_history_findings.md): six small label-stress E8 wins against one release, zero0.001 gates; same-development raw-anchor replay, no heldoutCE/private-training/CIA claim.

`divisible_certificate_audit.json` contains deterministic sensitivity-one continuous-MSDLap variance/certificate arithmetic, four scalar settings and27 conservative equal-coordinate/round allocation cases. Run `uv run python -m research.calculations.divisible_certificate_audit`. No random draws, data/model training or CIA outputs. [Methods and limitations](../../research/literature_review/divisible_noise_methods.md); flexible smoothing split is our shift-proof extension, not an exact finite-computer sampler certificate.

`client_energy_filter_probe.json`/`.npz` contain27 coupled noisy reduced-model training cells,100 development configurations/cell and nine selected arms with128 fresh paired evaluation trajectories. Fixed and carry-forward public schedules, greedy/equal-remaining filters and two history policies share a public accounting law. No matched history policy reaches0.001 CE improvement. `client_energy_one_release.json`/`.npz` add the schedule-changing local-model release reference; `client_energy_one_release_compute_control.json`/`.npz` add post-hoc20-local-step controls. `client_energy_filter_diagnostic.json`/`.npz` preserve development-only spending and pooled-descent diagnostics. See [protocol](../../research/proposals/2026-10-06_client_energy_filter_protocol.md) and [findings/direction update](../../research/proposals/2026-10-06_client_energy_filter_findings.md). Each fixed law has a conditional whole-client peer/dummy certificate; raw-data-dependent tuning and diagnostic publication are unaccounted. No CIA or full-experiment completion claim.

Regenerate in order with the local Fashion-MNIST Arrow cache:

```sh
uv run python -m research.calculations.client_energy_filter_probe --check
uv run python -m research.calculations.client_energy_filter_probe
uv run python -m research.calculations.client_energy_one_release
uv run python -m research.calculations.client_energy_one_release --compute-control
uv run python -m research.calculations.client_energy_filter_diagnostic
```

The pilot archive contains held-out indices, paired CE/accuracy and private spending traces; the diagnostic archive adds proposed/signal norms and diagnostic alignment/descent, not images. Neither archive is the modeled peer transcript. Tests: projected gradient finite difference, cap/gauge/dummy checks and default repository suite277passed/5deselected.


## One-time descriptors and prior/balanced constructor feasibility (2026-10-06–07)

See [verified findings and research handoff](../../research/proposals/2026-10-07_private_descriptor_findings.md), [initial protocol](../../research/proposals/2026-10-06_private_descriptor_protocol.md), [constructor protocol](../../research/proposals/2026-10-06_prior_constructor_protocol.md) and [math review](../../research/proposals/2026-10-06_private_descriptor_math_review.md). These are bounded research diagnostics; the owner has not declared the overall experiment finished.

`private_descriptor_probe`, `private_descriptor_analytic_gaussian` and `private_descriptor_radial_laplace` each have a JSON and NPZ: development search/choices, paired CE/accuracy/calibration and disjoint role indices for27cells/405arms per law. The original zCDP JSON uses its legacy calibration schema (`rho`); the newer laws carry explicit law/dimension metadata. `private_prior_constructor_development` contains17,496development configurations, raw/prior/balanced teachers, counts and vote diagnostics; `private_prior_constructor_evaluation` contains frozen choices evaluated on2048fresh reserve examples with512new paired draws,18cells/810arms. All NPZ files are intentionally versioned despite the default ignore rule; none contains image arrays.

The archives include operator-only private raw teachers/counts and diagnostic objects. Their publication/tuning is unaccounted; these are NOT the modeled peer transcript or an end-to-end DP dataset release. Fixed-seed finite-precision samplers are research implementations. Whole-client dummy adjacency, individual-upload privacy and central controls are distinct contracts. No CIA score, new density, metric-privacy superiority or deployed privacy guarantee is claimed.

Reproduce using the same local Fashion-MNIST Arrow cache and earlier geometry archive, in this order (evaluation reads saved development selections):

```sh
uv run python -m research.calculations.private_descriptor_probe --check
uv run python -m research.calculations.private_descriptor_probe
uv run python -m research.calculations.private_descriptor_probe --law analytic_gaussian
uv run python -m research.calculations.private_descriptor_probe --law radial_laplace
uv run python -m research.calculations.private_prior_constructor_probe --check
uv run python -m research.calculations.private_prior_constructor_probe --stage development
uv run python -m research.calculations.private_prior_constructor_probe --stage evaluation
uv run python -m research.calculations.audit_private_descriptor_artifacts
```

`private_descriptor_artifact_audit.json` records the separate auditor's68,532numeric comparisons, max error4.440892098500626e-16, source SHA verification, old/new/role disjointness,2048confirmation examples and1992remaining reserve examples. The auditor imports neither simulator nor simulator statistics and independently reconstructs all six primary arms from cached images and saved teachers. Both kernel checks and the default repository suite277passed/5deselected on2026-10-07. No jobs running.


## Fixed-slot conditional descriptor CIA (2026-10-07)

[Protocol](../../research/proposals/2026-10-07_descriptor_cia_protocol.md) and [findings/handoff](../../research/proposals/2026-10-07_descriptor_cia_findings.md) explain156cells, corrected target-class coverage, fresh-noise confirmation, the adapted metric scale channel and raw descriptor/student/individual views. `descriptor_cia_probe.json` contains per-cell configurations, validation-selected attacks, operator calibration, ROC-AUC/uncertainty/low-FPR points and diagnostic IN/OUT utility. Its NPZ holds evaluation scores, validation scores, utility arrays and utility indices; no image arrays. SHA values bind the dataset, development choices, corrected protocol and calculator.

The conditional auxiliary attacker knows target/background datasets; this is not independently sampled client-population evaluation. Utility reuses earlier2048confirmation images, with128fresh draws per world;1992reserve images remain unused. The metric student is identical to its noisy identifiable model, so `student_exact_llr` equals descriptor LR; the finite LDA/loss bank must not be substituted for that stronger available attack. Individual B descriptor auditing sees raw individually sanitized objects, not merely nonlinear Q_i. Saturated AUC1/SE0 is finite-sample ranking separation, not certainty about population error.

```sh
uv run pytest research/calculations/tests/test_descriptor_cia_probe.py
uv run python -m research.calculations.descriptor_cia_probe
uv run python -m research.calculations.audit_descriptor_cia_artifacts
```

`descriptor_cia_artifact_audit.json` independently recomputes score statistics, all attainable low-FPR thresholds, frozen defense/validation selections, variance calibration,48Gaussian theoretical AUC expectations and utility index consistency:4,188numeric comparisons,max error8.67e-19. The default suite passes282tests,5deselected. Source calculation now retains all ROC thresholds; an initial pruning error was corrected by recomputing summaries from unchanged saved scores. Operator traces/publication/tuning and fixed-seed finite-precision samplers are unaccounted. No end-to-end private release, historical metric/CNN replication or overall experiment completion is claimed.

## Matched conditional CIA strength (2026-10-07)

[Protocol](../../research/proposals/2026-10-07_matched_cia_protocol.md) and [findings/handoff](../../research/proposals/2026-10-07_matched_cia_findings.md) document the negative radial-law result and proposed representation direction. `matched_cia_development.json`/`.npz` save17,496configuration results and independent radial calibration streams. `matched_cia_evaluation.json`/`.npz` save117arms/936target rows, fresh attack scores and utility arrays,1,024new utility indices and968remaining reserve indices. Both NPZ files are explicitly versioned, contain no image arrays, and preserve operator diagnostics outside the modeled peer transcript. `matched_cia_contrasts.json` contains36fixed-target/equal-world stratified paired noise intervals; `matched_cia_artifact_audit.json` records33,174independent arithmetic checks plus hashes/choices/splits and504Gaussian expectation checks.

```sh
uv run pytest research/calculations/tests/test_matched_cia_probe.py
uv run python -m research.calculations.matched_cia_probe --stage development
uv run python -m research.calculations.matched_cia_probe --stage evaluation
uv run python -m research.calculations.audit_matched_cia_artifacts
```

Development must precede evaluation. Reproduction requires the same local Fashion-MNIST cache and preceding descriptor archives. Calibration, attack and utility streams are distinct; common random innovations across controls support paired contrasts. Gaussian risk is conditional on saved shift norms; radial matching is Monte Carlo estimated. Offline private maximum-norm calibration, tuning/publication and fixed-seed finite precision are unaccounted. This is fresh utility/noise confirmation on saved clients, not new client populations or a common-epsilon deployable comparison. Default suite287passed/5deselected; no jobs running and no overall completion decision.

## Public-reference residuals and class-conditional signal (2026-10-08)

[Findings/handoff](../../research/proposals/2026-10-08_public_residual_findings.md), [primary protocol](../../research/proposals/2026-10-08_public_residual_protocol.md), [pre-reserve public-control addendum](../../research/proposals/2026-10-08_public_residual_control_addendum.md) and [post-hoc development-only signal protocol](../../research/proposals/2026-10-08_residual_signal_diagnostic_protocol.md). Public-only decoder regularization explains the apparent projected-model gain. The distribution-aware gradient-centering lead remains untested as a protected mechanism.

`public_residual_development.json`/`.npz`:16,758configurations/eight-stratum CE arrays,18publicreference choices,48frozenfamily minima, public B/Hessian, fine teachers/selectedboundedqueries and hashes. `public_residual_evaluation.json`/`.npz`:432target rows/54arms, rawattack scores and utility draws/noiselessdiagnostics,512freshindices and456unusedreserve. NPZ files are explicitly versioned and contain no image arrays. `public_residual_contrasts.json` contains54fixed-stratum paired comparisons. `public_residual_artifact_audit.json` records260,338checks and independent teacher/Hessian/source/split/selection reconstruction.

`public_residual_public_control_development.json` contains16public-only decoder-offset records (withduplicates), selectedBEFOREreserveaccess; `public_residual_public_control_evaluation.json` evaluatesonlythe frozenwinner and54supplementalcomparisons ontheSAME512images. This addendum was audit-directed afterdevelopmentstarted; it was not originally predeclared or a new replication. The unselected analytic zero d51offset has roundoff-sensitive argmaxaccuracy; selected d3offset and CEselection are unaffected.

`public_residual_signal_diagnostic.json` savespost-hoc OLDdevelopment-only gradient norms and public class-gradient references,54pooledoracle scores and105gradientdirection/step scores, identity checks and hashes. `public_residual_signal_artifact_audit.json` independently verifies874numeric values without readingreserveindices/images. Mean-norm reductions and pooled unprotected headroom are not noise/privacy evidence. Observed maxima must notbeused aspublicsensitivity.

Reproduce in this order using existing descriptor archives and the same local Fashion-MNIST cache:

```sh
uv run pytest research/calculations/tests/test_public_residual_probe.py
uv run python -m research.calculations.public_residual_probe --stage development
uv run python -m research.calculations.audit_public_residual_artifacts --development-only
uv run python -m research.calculations.public_residual_public_control --stage development
uv run python -m research.calculations.public_residual_probe --stage evaluation
uv run python -m research.calculations.public_residual_public_control --stage evaluation
uv run python -m research.calculations.audit_public_residual_artifacts --require-public-control
uv run python -m research.calculations.public_residual_signal_diagnostic
uv run python -m research.calculations.audit_public_residual_signal
```

Whole-client contribution/dummy cap is public for fixedparameters; researchselection/reference tuning and diagnostic publication are unaccounted. All methods now receive512task-labelledpublicexamples; no comparison credits those labels to privacy. Strong peer-knowncoins/conditionaltargetauxiliary, individual-uploadprivacy andcentral aredifferentcontracts. Freshslice/noise uses savedclients, not newpopulationreplication. Defaultsuite293passed/5deselected. No jobs running or overallcompletiondecision.
