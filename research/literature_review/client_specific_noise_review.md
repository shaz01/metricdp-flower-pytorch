# Integrative review: constructing client-specific noise for participation privacy

Version 1, 2026-10-01. This is the detailed conceptual synthesis accompanying a bounded systematic search. It integrates inspected primary methods with the repository's verified evidence. Source-specific claims and equations live in linked technique cards; the design implications here are our deductions. No new protection technique has been implemented or demonstrated superior.

## What the project evidence actually asks us to solve

The original paper uses a trusted server and a curious participating client, which sees global models and has target-distribution shadow data. It changes server noise using inter-client model distance. The implementation uses inverse distance, and its data-dependent calibration is a heuristic rather than an established metric-DP theorem. The original paper's pooled ROC-AUC and this repository's paired score are different estimands. See the [evidence audit](../project_evidence_audit.md) for exact equations, protocol differences, source artifacts and numerical tables.

The [frontier HTML](../../experiments/contest_at_scale/reports/auc_frontier.html) is a valuable **exploratory utility–leakage map**. All 110 recorded stage entries matched their underlying artifacts. Of ten landed curves, nine have three-seed mean folded paired scores above 0.55; fresh confirmation seeds often leak more than the selecting seed. A landing or a weak clean-shadow score therefore cannot establish protection. The HTML does establish where utility survives, where calibration collapses, and which cases merit investigation. Its checkpoints are correlated and its score folds direction using the evaluation data. Keep it as the historical baseline while building an independent attack evaluation, rather than use 0.55 as a certified privacy threshold.

Our research question is consequently precise: **can clients construct private, statistically stable noise laws that improve whole-client participation protection at matched useful learning performance, against a defense-aware observer of global training trajectories?** “Better” must mean a measured improvement under that game, and any formal guarantee must use that game's adjacency and visibility.

## Six distinct decisions hidden inside “each client has a distribution”

| Decision | Alternatives | Why it changes the research problem |
|---|---|---|
| Secret | One record; whole-client dataset; participation bit; label proportions | A record-level accountant does not automatically hide an entire client. |
| Observer | Global-model peer; individual-upload server; secure-sum server | Local computation is not itself local DP. Participation metadata may reveal the answer directly. |
| Law | Isotropic; diagonal; low-rank plus floor; full covariance; learned non-Gaussian | Each requires different estimation, support and sampling analysis. |
| Construction signal | Public data; privacy preference; private gradients; privatized history; simulated population | Preferences are not private statistical estimates; private calibration needs analysis. |
| Coordination | Independent shares; secure aggregated shares; correlated shares | Privacy depends on randomness unknown to the observer, not marginal noise alone. |
| Time | Fixed; scheduled; history adaptive; temporally correlated | Repeated release and conditional accounting govern cumulative leakage. |

The [technical primer](privacy_and_noise_primer.md) explains these choices with weighted aggregation and two-dimensional examples. It is the best first read for understanding techniques without reading every paper.

## What distribution construction already exists?

**Choose a scalar law from preferences or a schedule.** LDP-Fed locally randomizes numerical updates with a metric-based exponential law; NbAFL perturbs uploads with Gaussians; time-adaptive spending personalizes clipping and privacy schedules. These establish that client placement and personalized budgets are existing ideas. Time-adaptive spending is particularly close to our trusted-server/curious-peer setup, but its client multipliers cancel with clipping, giving common absolute covariance within a round. See [client source cards CN1–CN3](client_noise_sources.md).

**Estimate a private calibration statistic.** Private adaptive clipping estimates a clipped-norm quantile with an accounted noisy statistic. It provides a concrete template for adaptation whose cost is explicit. Layer-history or feature-priority rules may be computationally simple, but the retrieved ALDP/priority methods have unresolved end-to-end sensitivity/adaptation questions. Treat them as construction precedents, not automatically valid certificates. See CN4–CN5 and the [priority-based follow-up](review_covariance_checks.md).

**Estimate output covariance under a population model.** PAC Privacy simulates algorithm outputs and uses their variation to construct noise; PAC-Private Algorithms uses a public basis and projected variances. These answer much of the distribution-construction question already, under distribution-dependent information guarantees. A bootstrap of one client's records is not automatically a distribution over client participation or over the attacker's uncertainty. Our adaptation would need an explicit client-population and side-information model. See [COV-01–COV-02](covariance_sources.md).

**Choose geometry from sensitivity and utility.** Matrix-variate Gaussian noise uses row/column covariance, and explicitly recognizes a privacy cost for privately deriving directions. IMGM provides an important negative result: IID noise is optimal for its full Frobenius sensitivity-ball/noise-overhead formulation. Thus “correlated gradients imply better anisotropic DP” is unjustified. A gain requires a different feasible shift geometry or utility objective, or a distributional rather than worst-case privacy target. See COV-04–COV-05.

**Learn a richer distribution against inference.** Residual-PAC optimizes a perturbation family and a reconstruction decoder. This is a close predecessor to learning the sampler itself. A fitted decoder's error is not automatically a bound against all inference: decoder cross entropy can overstate true conditional entropy. A learned CIA-specific sampler would need held-out adaptive attacks and a clear statement of what is formal and what is empirical. See COV-06.

**Construct aggregate noise through coordination.** Distributed discrete Gaussian and Skellam supply integer samplers compatible with secure aggregation. CAPE and CorN exploit inter-agent correlations. Their visibility/collusion assumptions are central; they do not provide private per-client parameter-space covariance estimation. See [distributed source cards](distributed_sources.md).

These are distinct construction families, not interchangeable recipes. The systematic search has not identified a fully verified drop-in mechanism combining all our desired properties. That is a search-bounded observation, not a novelty claim.

## Why moving the noise alone cannot explain a gain

For a clipped linear aggregate with fixed weights `a_i` and conditionally independent noise `ξ_i ~ N(0,Σ_i)`, the aggregate covariance is `Σ_agg=Σ_i a_i²Σ_i`. If the corresponding server mechanism adds Gaussian noise with that covariance and the same conditional mean/update process, the model-only observer sees the same conditional law. Repeating the equality at every history yields the same transcript law. This is our mathematical control, not an empirical result.

A peer knows its own update and noise. After conditioning on those, only unknown honest contributions protect the residual. A server may see individual uploads and need a different guarantee. Local noise before clipping or local optimization changes the learning path and is not covered by simple relocation equivalence. Correlated shares require cross-covariance and conditional covariance calculations. Secure aggregation can change visibility, but does not itself privatize the released model.

The first comparison must therefore match the **observable conditional aggregate law**, weights, clipping and optimization. Otherwise a claimed client-side advantage may merely reflect more total noise, altered aggregation or a weaker attacker.

## Four matrices that must not be confused

1. **Natural gradient covariance:** stochastic variability across examples/minibatches. GMIP uses it in a likelihood-based membership analysis; that is not a learned defense covariance.
2. **Sensitivity geometry:** feasible changes in an output when an entire client is removed or replaced. Worst-case DP constrains these changes.
3. **Utility curvature:** how noise directions damage loss. A local approximation uses `0.5 trace(HΣ)`; it is not a leakage bound.
4. **Attack information:** score/transcript directions that distinguish client IN from OUT. Estimating these requires a defined attack population and can overfit.

Their alignment is a plausible empirical hypothesis, not an identity. A client with one gradient history cannot directly observe its counterfactual global OUT trajectory. Public shadow federation simulations, privately estimated statistics, or a clearly specified distributional assumption are possible bridges. Each changes the guarantee and cost. The [GMIP tutorial](local_gaussian_analysis.md) and [accounting/attack foundations](foundations_and_attacks.md) explain the boundaries.

## The major proof and estimation obstacle

A released vector with `N(0,Σ(D))` noise has a data-dependent law even if Σ itself is never transmitted. Neighboring means **and covariances** can differ. The output can therefore leak the calibration choice. Add/remove clients also changes aggregate weights, the number of shares and potentially the estimator's distribution. A positive covariance floor avoids deterministic unnoised directions, but does not prove privacy.

Possible research routes are: public fixed geometry; geometry adapted only from an already protected transcript; privately estimated geometry with explicit composition; or a distributional guarantee under a client-population model. These should be separate hypotheses. For worst-case whole-client DP, privatizing an estimate from one client's records at record level is not enough by itself. For a distributional route, attacker conditioning and population misspecification must be made explicit.

Statistically, full covariance in a deep model is often unrealistic. Diagonal, per-layer and public-basis low-rank estimates with shrinkage are tractable starting points. The number of independent gradient samples, estimator drift, rank, eigenspace stability and client-specific overhead must be measured. Historical samples are correlated; treating them as independent can make an apparently precise sampler unstable. Small client/shadow datasets in the frontier make this especially relevant.

## Nearest prior work and novelty risks

| Prior family | Overlap with proposed idea | Difference still needing investigation |
|---|---|---|
| Time-adaptive spending | Local Gaussian shares, client budgets, curious peers | Common absolute noise within round; no learned per-client covariance |
| PAC / PAC-Private | Data-distribution-based directional noise construction | Population-information target rather than worst-case whole-client removal |
| MVG / IMGM | Gaussian geometry and calibration | Private geometry cost; no universal anisotropic superiority |
| Residual-PAC | Learned perturbation distribution | Approximate decoder versus certified leakage; client/transcript transfer |
| ALDP / priority-based adaptation | Client-history/feature-driven perturbation | End-to-end sensitivity and private adaptation concerns |
| CAPE / CorN | Client-generated correlated Gaussian laws | Cross-agent covariance, observer/collusion constraints |
| FACP (2026, pending methods) | Fisher-guided clipping and anisotropic FL noise | Category inference differs from participation; full proof/estimator unread |
| FedFR-ADP (pending methods) | Client heterogeneity and feedback-driven Gaussian calibration | Exact estimator, privacy unit, and noise law still unread |

Read the pending closest comparators before committing to a novelty statement. A credible contribution could be **CIA-aware construction and privacy treatment of a client-specific law**, backed by independent whole-client evaluation and estimator ablations. It cannot simply be “add adaptive noise at clients.”

## Evidence appraisal and confidence

The retrieval supports the existence of construction precedents, the distinction between privacy units, and mathematical aggregation controls. It does **not** establish which construction wins our frontier, validate all authors' proofs, or demonstrate that our proposed combination is new. Selected theory papers provide usable principles only under their assumptions. Empirical adaptation papers with unclear sensitivity/accounting remain useful adverse/heuristic comparators, but weak foundations for a formal claim.

Cross-paper utility and attack numbers were deliberately not pooled. Different architectures, client counts, adjacency, privacy budgets, observers and reconstruction/membership metrics prevent a meaningful common effect size. Code availability and detailed confidence intervals were not uniformly audited; absence of an assessment is not evidence that artifacts or uncertainty reporting are absent.

The [search report](systematic_search_report.md) states captured-record counts, exclusions, read scopes, protocol limitations and PRISMA mapping. This review is AI-assisted and methods-focused; it is not an externally registered or human dual-screened exhaustive review.

## Research decision supported by the review

Proceed first with an explicit CIA threat specification and observable-noise control, then a tractable geometry construction whose estimator and guarantee can be analyzed. Keep worst-case client DP and distributional/empirical CIA protection as separate tracks. Use the [provisional research plan](research_plan.md) to define milestones and stopping criteria. No mechanism is selected for implementation in this review.
