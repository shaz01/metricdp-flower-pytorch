# Fixed-slot descriptor CIA: attack boundary and research direction

2026-10-07; active branch `feature/client-specific-noise`. Owner: “lets move to the next step”. This bounded audit advances the [utility feasibility result](2026-10-07_private_descriptor_findings.md). It does not declare the overall experiment complete or establish general superiority over metric privacy.

## Concrete finding

The epsilon8 voting constructor is **useful but clearly distinguishable** under the conditional known-alternative attacker. Across three saved label-stress seeds and four dominant-class target slots, analytic-Gaussian aggregate votes have descriptor AUC0.870226–0.893721 (mean0.879624). Radial L2-Laplace votes have AUC0.778572–0.806821 (mean0.792654), with a higher utility cost. Neither is near chance. The earlier CE gain at a common epsilon budget must not be promoted to a better CIA frontier: the Gaussian non-voting control also leaks less here, at meanAUC0.785588.

A second concrete finding concerns private calibration: the reduced metric-inspired mechanism has IN/OUT noise-standard-deviation ratios2.103454–2.370879. A two-variance likelihood-ratio attacker achieves empirical AUC0.999998–1 across the multiplier grid, including large noise. Increasing both standard deviations together does not remove their ratio. In this adaptation the student decoder is identity, so this same likelihood-ratio attack is directly available from the released student. The weaker finite LDA/loss attack bank misses much of this scale signal. This illustrates why an attack audit must consider changes to the entire released law, including private calibration, not just mean differences.

## What was tested

The [protocol](2026-10-07_descriptor_cia_protocol.md) fixes registered slots, peer7, targets0/1/2/3, public weights1/8, empty target signal in OUT and continuing noise shares. Eight deterministic local teachers are those saved in the prior development phase. Background identities/data and learner settings are fixed in both worlds. IN and OUT random releases are independent; the peer never receives counterfactual pairs, hidden scales or other clients' coins. All selectors use previously saved defense choices and attack training/validation only.

There are96fixed-budget channel cells (three seeds × four targets × two laws × four arms) and60adapted metric cells (five multipliers). Each uses256IN+256OUT training releases,256+256validation releases and1024+1024fresh evaluation releases. The first128evaluation releases per world supply diagnostic CE/accuracy on the previously used2048confirmation examples. No images from the remaining1992reserve examples were used. This is fresh-noise evaluation, not fresh utility-data or new client-population confirmation; seeds reuse the private bank.

The strong descriptor attacker knows the whole target and background datasets as auxiliary information. For aggregate A, it subtracts only its own upload, leaving seven unknown noise shares. Gaussian score uses the known mean-shift projection; radial score compares distances to the two means. These are exact likelihood-ratio rankings for the stated residual channels. Central noise is entirely server-owned and cannot be subtracted by the peer. For individual B, this audit observes the raw individually sanitized object; it is a stronger view than publishing only its nonlinear distribution Q_i. Do not transfer B's measured leakage to a Q_i-only deployment.

The student bank uses released student parameters and own noisy upload where applicable, plus target-loss attacks and training-standardized shrinkage linear discriminants. Validation selects the attack and loss orientation; evaluation scores never select a defense or classifier. It is a limited attack bank, not a proof of optimal student privacy. It uses known-alternative shadow releases under fixed datasets, not independently sampled shadow client populations. The student-only view is narrower than the aggregate-descriptor release guaranteed by A.

Ordinary ROC-AUC compares every IN score with every OUT score, with tie halves. It is oriented on validation and never folded paired concordance. Per-cell intervals use the two-sample DeLong placement variance; low-FPR points include all attainable empirical thresholds. Saturated empirical AUC1 can produce SE0/interval[1,1]; this means finite-sample ranking separation, not perfect population inference or an error-free privacy claim.

## Diagnostic summary

Each entry averages12target/seed cells. These are descriptive averages of conditional AUCs, not a pooled population AUC, statistical frontier, or simultaneous confidence estimate. Utility averages the IN and OUT means equally.

| Law / arm | Descriptor LR AUC | Student finite-bank AUC | Equal-world CE | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| analytic_gaussian / aggregate_control | 0.785588 | 0.695044 | 1.093238 | 57.45% |
| analytic_gaussian / aggregate_votes | 0.879624 | 0.791387 | 0.955695 | 63.64% |
| analytic_gaussian / central_votes | 0.879295 | 0.824509 | 0.949454 | 64.02% |
| analytic_gaussian / individual_votes | 0.880846 | 0.640195 | 1.223456 | 47.32% |
| radial_laplace / aggregate_control | 0.864593 | 0.826155 | 1.108074 | 58.05% |
| radial_laplace / aggregate_votes | 0.792654 | 0.686972 | 1.003976 | 60.74% |
| radial_laplace / central_votes | 0.841703 | 0.789844 | 0.991796 | 60.33% |
| radial_laplace / individual_votes | 0.945798 | 0.700901 | 1.171335 | 50.96% |
| metric_adaptation / metric_0.1 | 1.000000 | 1.000000 | 1.068256 | 72.60% |
| metric_adaptation / metric_0.3 | 1.000000 | 0.999925 | 1.073274 | 69.85% |
| metric_adaptation / metric_1.0 | 1.000000 | 0.856079 | 1.129769 | 56.62% |
| metric_adaptation / metric_10.0 | 1.000000 | 0.745820 | 3.950826 | 29.39% |
| metric_adaptation / metric_3.0 | 1.000000 | 0.636044 | 1.540548 | 39.17% |

For ALL metric rows, the strongest available student LR AUC equals the descriptor LR column. The student finite-bank column is shown to expose its failure to detect variance leakage, not as the metric mechanism's achieved protection. All three selected metric model-direct configurations have steps80,radius3.2,ridge0. The public denoiser is identity, so latent noisy model and released student are the same identifiable parameters.

Central voting is independently development-selected within the same family, not uniformly an identical-query placement experiment. Five of the six seed/law settings match aggregate voting. Seed44 radial central uses20steps,radius0.4,ridge0,K4; aggregate uses80steps,radius0.8,ridge0.01,K16. This explains the heterogeneous central radial leakage and prevents attributing its averaged difference solely to placement. Analytic Gaussian's matched central arm has essentially the same exact descriptor AUC, with slightly better utility; client placement is not the demonstrated contribution.

## Metric adaptation and interpretation

The repository computes maximum mean layer-wise L2 distance over raw client models, then server Gaussian std `noise_multiplier*clipping_norm/(num_clients*distance)`. This calculator uses the corresponding weight/bias-layer distances on the reduced four-class model, a fixed denominator8, explicit zero dummy, frozen model-direct transform/radius/ridge and multipliers0.1/0.3/1/3/10. Private distances are operator diagnostics only, never attacker features. The exact Gaussian likelihood ratio uses the alternative means AND variances; it can infer scale without being told the realized private distance.

The raw zero dummy can enlarge the OUT maximum distance; inverse calibration then reduces OUT noise relative to IN. A multiplier increase scales both laws but preserves this difference. In51identifiable dimensions, noisy-model norm is informative about that scale. A mean-only linear attack can appear weak while a quadratic likelihood attack succeeds. This is an empirical conditional side channel of this adapted whole-dataset game, not a theorem about all metric calibration or the historical renormalized-removal/CNN setting. It does not assign an epsilon guarantee to the metric multiplier.

The [Gaussian MIP primary paper](https://proceedings.nips.cc/paper_files/paper/2023/file/e9df36b21ff4ee211a8b71ee8b7e9f57-Paper-Conference.pdf), Section3, frames record membership as a hypothesis-testing problem. Its sampled-record game differs from our fixed whole-client dummy alternatives; its guarantees are not borrowed here. The likelihood-ratio calculations used here follow directly from the explicit Gaussian/radial channel densities. This scoped read is separate from the frozen systematic corpus and is not a replication of its gradient attack.

## Corrections, checks and limitations

The first implementation incorrectly selected targets0/2/4/6, covering only two dominant classes. A failing coverage test caught the mismatch; that run is excluded. Targets were corrected to0/1/2/3, and all training/validation/evaluation streams moved to fresh50/60/70million namespaces. The finite attack bank and defense choices stayed fixed. Preliminary outputs also prompted an algebraic student-view audit; independent source review verified the metric identity decoder. This is transparently a subsequent fresh-noise confirmation after preliminary inspection, not external preregistration.

A separate failing regression test showed that a pruned ROC curve could omit attainable low-FPR operating points. Statistics were corrected to retain every empirical threshold and recomputed from unchanged saved scores. No classifier, defense choice or random draw changed.

Independent contract review checked known-peer conditioning, dummy shares, Gaussian/radial scores and selection separation; it required explicit metric-student LR reporting, central-configuration distinctions, individual-object scope and saturated-AUC caveats. These corrections are included. The [independent arithmetic receipt](../../results/client_specific_noise/descriptor_cia_artifact_audit.json) checks156rows and4,188numeric values, max discrepancy8.67e-19, frozen defense choices, validation-selected attacks, utility indices and48Gaussian expected-AUC checks (maximum deviation1.921standard errors). The auditor imports no CIA simulator. Five targeted tests and the full default suite pass:282passed/5deselected. Raw scores and utility arrays are in [results](../../results/client_specific_noise/README.md).

The fixed-query privacy certificates remain conditional on observer/adjacency and ideal laws. Offline raw tuning, publication of operator diagnostics and NumPy finite-precision fixed-seed implementation are not an end-to-end private deployment. High AUC at epsilon8 does not violate that certificate: epsilon8 permits substantial distinguishability. Conversely weak finite-bank student attacks do not prove privacy.

## Next research direction

Keep the useful balanced-vote query, but calibrate and compare complete channels at **matched attack strength**, before claiming a protection improvement. The Gaussian voting gain currently trades lower CE for higher descriptor leakage against its Gaussian non-voting control. Radial voting reduces attack AUC but adds noise/utility cost; this may be ordinary variance rather than a special density advantage.

The next bounded protocol should compare Gaussian, radial and variance-matched Gaussian voting at development-selected near-chance targets, with independent evaluation noise and strongest query controls. Use exact Gaussian channel calculations and known-alternative radial attacks as sanity checks; do not choose an epsilon or multiplier from test AUC. Keep student-only and descriptor-visible views separate. Add independent client-population shadows and realistic target-domain auxiliary knowledge before treating this conditional attack frontier as a paper finding.

Any client-specific noise-shape proposal must account for how neighboring datasets change shape/scale. The metric variance example makes that obligation concrete. A new sampler or density is useful only if it improves the protected complete-law attack–utility frontier after estimation cost, not merely a marginal variance or an optimistically weak attack score. No broad CNN/Flower sweep, final mechanism claim or branch merge is justified by this bounded result alone.
