# Next-step handoff: comparators, threat game and construction

2026-10-01, `feature/client-specific-noise`. This is a supplement to review version 1, rather than a new claim of exhaustive literature coverage. The owner authorized committing the existing staged results and proceeding with the next research step. All 141 staged files were archive-checked, committed as `1081ff7`, and pushed to this branch. No new experiment was launched.

## Owner steering: broaden the law family

During this step, the owner asked why the project should use only Gaussian noise and proposed a potentially unique mechanism/distribution. The [expanded mechanism direction](../non_gaussian_mechanism_research.md) makes Gaussian a control and adds non-Gaussian geometry, optimized densities and learned laws. The original search counts remain a historical version-1 record; new focused source-family queries and methods cards are separately logged in [non-Gaussian foundations](non_gaussian_foundations.md) and [optimized noise](optimized_noise_followup.md).

## What this step delivered

- [FACP partial-method assessment](facp_followup.md): recovered selected primary methods, with source/theorem limitations kept explicit. Complete artifact and algorithm still pending.
- [FedFR-ADP assessment](fedfr_followup.md): documented preview evidence and full-method access gaps; no algorithm was invented from the abstract.
- [Threat specification](../threat_specification_review.md): the candidate client-participation game, actual code behavior, metadata contract, weighting/sensitivity and curious-peer conditioning.
- [Mathematical construction starting point](../noise_construction_proof_obligations.md): public ellipsoidal clipping and Gaussian sampling, weighted residual covariance, a sensitivity constraint and private-estimation proof obligations.
- [Independent mathematical review](../noise_construction_math_review.md): calculations checked and wording corrections incorporated; not complete proof certification.

## Changes to our understanding

The existing attack benchmark evaluates a narrower observation than all information available to the simulated client application. A future claim must state whether it protects global-model information, the complete curious-peer view, or data contribution among fixed registered slots. Physical absence cannot be hidden if counts or identities disclose it.

FedAvg's sample-count weights and renormalized OUT denominator change the sensitivity question. For a common norm bound C and target weight a, a one-round removal shift can be bounded by `2aC` under the specified shared-history assumptions. Fixed-denominator zero contribution instead has bound `aC`. They are different protocols. Client count alone cannot establish the appropriate calibration.

The current removal mapping preserves background data but changes active IDs used for training seeds. Same-seed IN/OUT pairs therefore do not preserve every background random innovation. This is important for paired influence diagnostics; it does not by itself invalidate comparisons of independently sampled marginal training laws.

A peer can condition on its own update/noise. Match unknown honest residual noise, not merely total aggregate variance, when testing placement. The threat specification supplies a hybrid server control that preserves the peer's own coins and view in both arms.

Public fixed geometry offers a tractable reference mechanism. Privately estimating individual geometry remains a separate research problem, requiring statistical stability, complete-law privacy treatment and a defined absent-client behavior. Noise covariance can be public; privacy must rely on unknown random draws and a valid law, not keeping public calibration secret.

These are research deductions and verified implementation observations. None demonstrates that shaped noise improves CIA protection.

## Evidence accounting

The original version-1 search counts stay fixed: 51 logged queries, 106 captured ledger entries, 31 included source families and 19 pending candidate families. Both comparator families remain pending complete methods, so partial access is not counted as full-text inclusion.

This supplement records **48 further query executions** across [root](followup_root_log.json), [FACP](followup_facp_log.json) and [FedFR](followup_fedfr_log.json) logs (12 + 25 + 11). These include overlapping discovery, section and access checks. They are not 48 new studies or necessarily 48 distinct query strings. Retrieval endpoints/outcomes are recorded separately. The initial ledger describes version-1 screening; the follow-up logs describe subsequent read-status developments. The [source inventory](source_inventory.md) remains the canonical family list with the supplement linked below it.

The non-Gaussian extension adds [four root discovery queries](followup_non_gaussian_root_log.json) and [three foundation queries](followup_non_gaussian_foundations_log.json); [optimized-noise retrieval](followup_optimized_noise_log.json) followed the root discovery without additional agent search queries. These seven executions are separate from the 48 comparator-query executions above. Four new primary method sources are included in the supplement; the initial 31-family count is unchanged.

## Next research decision

Resolve the hidden event and metadata contract first: physical participation versus whole-client learning contribution among fixed candidate slots. Then specify two candidate laws for comparison: a public/protected-history geometry reference, and a private-estimation or modeled-population constructor. Each needs an explicit sampler, estimator cost, clipping/sensitivity, absence rule, conditional accounting and attack-development/test split.

FACP/FedFR full-method acquisition remains necessary for precise replication/novelty comparison, but does not block these independent threat and mathematics tasks. Do not implement a named comparator from fragments. Do not launch a large sweep before the game, complete observable law and small independent pilot protocol are concrete.
