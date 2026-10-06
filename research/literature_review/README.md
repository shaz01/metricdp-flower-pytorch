# Client-specific noise literature review — start here

Version 1, 2026-10-01. The owner requested both systematic and integrative review and confirmed the protocol. This package provides a bounded, documented search, methods-level technique explanations and a provisional research plan. It is not an exhaustive novelty certificate or a claim that any candidate already beats metric-inspired server noise.

## Follow-up after version 1

**Observer contract, 2026-10-06:** [review and construction recommendation](../proposals/2026-10-06_observer_contract_review.md) separates model/peer privacy from individual-upload privacy, includes the peer's known randomness, and checks a conditional divisible-Laplace reference. It can improve over the stronger local requirement but does not beat matched server noise by placement alone. [New focused literature handoff](infinitely_divisible_followup.md) separately tracks Arete and FORC2025 generalized/multi-scale Laplace; full methods/proof feasibility is the next proposed step. No implementation or training launch; version-1 counts unchanged.

**Paid protected-history replay, 2026-10-05:** [findings and next design question](../proposals/2026-10-05_protected_real_history_findings.md) retain probe-selection correlation and match static/full-budget controls. Six of 54 cases improve over one release, only in label stress at budget 8, by 0.000109–0.000247 development quadratic loss; zero 0.001 gates. Raw anchored replay and unaccounted calibration do not certify private training or CIA protection. Next proposed work: review the observer-specific protection contract before another construction/sweep.

**Hidden-mixture and adjacency audit, 2026-10-05:** [analytic argument and fair cost results](../proposals/2026-10-05_hidden_mixture_findings.md) show generic tails approaching the additive bound for the current hidden-selector law. Contribution/dummy adjacency permits smaller noise independently of hiding the category; applying that change equally to controls leaves no0.001 development gate pass. Protected-history routing on measured label skew remains the next proposed design/cost question.

**Client-local construction/cost review, 2026-10-05:** [law and findings](../proposals/2026-10-05_local_selector_findings.md) define slot-invariant local routing, categorical RR and conditional L1 upload accounting. Exact development costs leave small label-stress gains at 8/16 but no 0.001 gate pass; raw history/calibration costs remain excluded. Deprioritize this paid selector. Protected-history or hidden-mixture analysis remains a proposed question, not a selected defense.

**Frozen slot control, 2026-10-05:** [findings](../proposals/2026-10-05_public_slot_findings.md) show the prior fully frozen profile assignment exactly matching the oracle in aligned primary cases, with fresh gains below0.001. Reassigning the same clients to different slots makes the frozen pattern harmful. The prior signal does not establish that private distribution estimation is needed. Next proposed work is a permutation-equivariant local construction/accounting design review before more implementation.

**Broader oracle, 2026-10-05:** [findings and next decision](../proposals/2026-10-05_broader_oracle_findings.md) test all three bias contrasts and the full 51-dimensional identifiable reduced head on fresh test examples. Only bias calibration in label stress at round 20/label 8 clears the 0.001 CE point gate across three seeds; two noise-only intervals clear it. Full-head personalization does not. The opportunity concerns clipped aggregate signal, not demonstrated covariance-noise benefit. Next proposed work is a narrow construction/accounting design review.

**Real-data geometry audit, 2026-10-05:** [findings and construction decision](../proposals/2026-10-05_client_geometry_findings.md) confirm persistent centered covariance differences in explicit label stress, but the best raw-information oracle gain is only 0.000363 CE, below the 0.001 feasibility gate. The two-bias-coordinate restriction prevents a full-model conclusion. Broader oracle design review precedes a private estimator. No CIA claim follows.

**Oct-4 stopping point:** [rotated/residual comparison](../proposals/2026-10-04_rotated_residual_findings.md) adds rotated public geometry and refreshed selectors. Residual selection beats tuned static geometry only once, by ~0.39% at a large privacy budget; one-release controls win throughout. The owner requested stopping after this comparison.

**Changing-update test:** [held-out quadratic findings](../proposals/2026-10-04_dynamic_quadratic_findings.md) show the selected adaptive bank reverting to identical static profiles in all six cells. One-release public controls perform better, even after fair public shrinkage. This limits the preceding stationary result; no CIA claim follows.

**Protected-history follow-up:** [two-round findings](../proposals/2026-10-04_protected_history_findings.md) include 15 analytical cases and the selection-noise correlation. Adaptation sometimes improves the tested fixed schedule; a public one-release control beats or matches it in every case. Next: changing-update quadratic learning, rather than a deep-model sweep.

**Next-step calculation:** [heterogeneity findings and construction decision](../proposals/2026-10-04_heterogeneity_findings.md) compare 30 bounded analytical cases with optimized public geometry and diamond-noise controls. Separately paid RR selection showed no gain; protected-history construction is the next candidate. Reproduction code and arithmetic artifact are linked in the note. These are not CIA results.

**Concrete direction, 2026-10-04:** the owner confirmed dataset-contribution secrecy. Read the [joint clipping/noise proposal](../proposals/2026-10-04_mechanism_proposal.md), [independent mathematical review](../proposals/2026-10-04_candidate_math_review.md), [analytical feasibility check](../proposals/2026-10-04_analytic_feasibility.md), and [proposed pilot protocol](../proposals/2026-10-04_pilot_protocol.md). Private construction cost is explicit; the first analytic comparison is unfavorable at budgets 4 and 8 and slightly favorable at 16 against only the listed fixed controls; optimizing the public clipping radius erases that gain. No CIA improvement or deep-model feasibility is established.

The [FACP partial-method assessment](facp_followup.md) and [FedFR-ADP access assessment](fedfr_followup.md) update the nearest-comparator evidence. FACP remains pending a complete artifact, but its auxiliary-statistics accounting exclusion is now confirmed from primary methods. FedFR-ADP remains preview-only. The [threat specification](../threat_specification_review.md) and [construction/proof obligations](../noise_construction_proof_obligations.md) develop the next research step. See [follow-up handoff](next_step_handoff.md) for new findings and preserved version-1 search counts.

The owner also opened the direction to **new non-Gaussian mechanisms/distributions**. See [expanded mechanism research](../non_gaussian_mechanism_research.md), [non-Gaussian foundations](non_gaussian_foundations.md) and [optimized noise](optimized_noise_followup.md). Gaussian remains a control, not a design restriction.

## Reading order for the project owner

1. [Detailed integrative review](client_specific_noise_review.md) — the main document: what the literature means for our question and why the construction problem matters.
2. [Technique primer](privacy_and_noise_primer.md) — intuitive examples and equations for privacy units, noise distributions, weighted aggregation, geometry and estimation.
3. Source explanations by family: [client perturbation](client_noise_sources.md), [covariance and learned distributions](covariance_sources.md), [distributed noise and trust](distributed_sources.md), [accounting and attacks](foundations_and_attacks.md), and [detailed GMIP tutorial](local_gaussian_analysis.md). The [priority-based adaptive method](review_covariance_checks.md) is an additional construction predecessor.
4. [Provisional research plan](research_plan.md) — milestones, comparison controls, proof obligations and decision criteria.

## Agent handoff — read before proposing a mechanism

Read [project evidence audit](../project_evidence_audit.md), the integrative review, primer and [evaluation requirements](evaluation_requirements.md). Use [comparison/appraisal matrix](comparison_matrix.md) to locate methods and [source inventory](source_inventory.md) for primary URLs, inspected versions and exact section locators. Use [systematic search report](systematic_search_report.md), [confirmed protocol](protocol.md) and [machine-readable screening ledger](screening_ledger.json) to understand scope and missing evidence.

Carry these constraints into every new proposal:

- CIA here means **client participation inference**, not category inference or source attribution.
- Protecting records is different from removing an entire client. Local noise placement is different from local DP.
- The frontier is an exploratory map; its folded paired score is not ordinary independent-trial ROC-AUC. Nine of ten landed curves exceed 0.55 in their three-seed means.
- Matching the conditional aggregate Gaussian law can make client/server noise equivalent for model observers. A curious peer knows its own contribution and noise.
- Private estimation can leak through the noise law even if the estimate is never explicitly released. Analyze the complete estimator/sampler across neighboring inputs and rounds.
- Natural gradient covariance, sensitivity geometry, utility curvature and attack information are different quantities.
- Client placement, personalized budgets, covariance shaping and learned samplers already have precedents. Read FACP/FedFR-ADP full methods before novelty claims.
- No experiments were launched, no winning mechanism was selected and no privacy proof was certified by this review.

## What is established and what remains open

The package identifies concrete construction families and explains their assumptions. It contains independent AI cross-checks in [covariance review](review_covariance_checks.md), [evaluation review](review_evaluation_checks.md) and [threat review](review_threat_checks.md); important qualifications were incorporated into the source cards. This is not independent human dual screening.

Full-method assessment of pending comparators, supplement/code checks, exact client-level accounting and independent defense-aware CIA pilots remain next work. The research plan proposes separate worst-case-DP and distributional/empirical tracks so those claims stay clear. Keep the old frontier as evidence and evaluate future defenses with an improved protocol.

## Updating this package

Append exact queries and explicit decisions to the appropriate family log, preserving read scopes and unfavorable findings. Update the combined ledger/counts, inventory, matrix and synthesis together. Do not count alternate URLs as new studies or reinterpret uninspected papers from abstracts. Disclose protocol amendments and maintain the 2026-10-01 cutoff for version 1. Source cards distinguish author claims from our deductions; preserve that distinction.

Temporary downloaded PDFs were not added to Git. Stable primary links and locators are recorded. The owner's supplied GMIP PDF remains a pre-existing untracked file; its tutorial also has a stable [arXiv source](https://arxiv.org/abs/2306.07273) for other machines.
