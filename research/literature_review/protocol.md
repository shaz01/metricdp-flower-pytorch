# Review protocol: client-specific noise for client inference protection

Protocol version: 1.0, 2026-10-01. Status: confirmed by the owner on 2026-10-01 (reply: "Confirm this protocol"), before the systematic source-family extraction; exploratory discovery had already begun. Review forms: systematic search and screening, followed by integrative technical synthesis, explicitly requested by the owner. No external registration has been made; OSF registration is recommended if the review is developed into a standalone systematic-review publication. This repository record is not an external registration.

## Research question and scope

How can federated clients construct and sample individual noise distributions that protect whole-client participation against an observer of global training models, while preserving more utility than fixed server-side Gaussian noise or the project's server-side metric-inspired calibration?

Secondary questions concern the source of the distribution estimate, its privacy cost, geometry and temporal dependence, adversary knowledge, aggregation scaling, and whether empirical attack evidence corresponds to the desired privacy unit.

PICOS adapted to computational privacy research:

- Population: federated/distributed learning systems and their participating clients; general private optimization mechanisms are included when their noise construction is directly transferable.
- Intervention: client-side additive perturbation, client-specific/adaptive/anisotropic distributions, distributed noise, and mechanisms for estimating private noise parameters.
- Comparator: fixed isotropic Gaussian noise, server-side client-level DP, metric-inspired server calibration, or an explicit published baseline.
- Outcomes: attack leakage and attacker access; record/client/attribute privacy definition; formal assumptions and accounting; learning utility; computational/communication cost; estimation requirements.
- Study types: theoretical mechanisms, empirical ML/privacy studies, and foundational definitions/accounting. Reviews may identify primary sources but do not replace them as evidence for algorithm details.

## Eligibility fixed before retrieval

Include English-language primary research available by 2026-10-01, without a lower date cutoff. Both peer-reviewed and preprint sources are eligible and must be labeled. Include methods that explain distribution construction, local/distributed placement, client participation privacy, or directly relevant attack evaluation. Include the owner's two local PDFs and backward/forward citation candidates under the same relevance criteria.

Exclude marketing/blog/tutorial summaries as mechanism evidence; duplicate preprint/publication records; work whose only connection is generic FL optimization; exclusively cryptographic protocols without relevant trust or noise implications; and inaccessible or abstract-only sources from detailed method-level conclusions. Such candidates remain in the ledger with exclusion or awaiting-full-text reasons. Do not silently omit adverse findings.

## Retrieval plan

Search arXiv and peer-reviewed repositories/publisher pages (PMLR, OpenReview, NeurIPS Proceedings, ACM/IEEE or author-hosted accepted papers). The web search interface may index these sources; distinguish domain-restricted discovery from a direct database export. Record exact executed queries, date, source family, candidates inspected, and selection decisions. Search engines do not expose exhaustive database hit counts; report captured records only, without fabricating total hits.

Pre-specified query families (combine the listed synonyms rather than require literal matches):

1. Federated learning AND (client inference OR participation inference OR client-level differential privacy OR user-level differential privacy).
2. Federated learning AND (local differential privacy OR client-side noise OR personalized privacy OR adaptive noise OR heterogeneous noise).
3. (Gradient OR model update) AND (anisotropic OR covariance OR Fisher OR optimal noise OR matrix Gaussian OR learned perturbation) AND privacy.
4. Federated learning AND (distributed noise OR distributed differential privacy OR secure aggregation OR correlated noise).
5. (Adaptive clipping OR private covariance OR data-dependent noise OR privacy filter) AND differential privacy.
6. (Membership inference OR reconstruction OR property inference) AND (gradient perturbation OR Gaussian privacy OR defense evaluation).

Repeat families across arXiv and proceedings sources; expand through references of the nearest methods. Search late-period 2024-2026 work separately to reduce dependence on older foundational methods. Stop after the planned query families and targeted citation expansion are documented; do not call that stopping rule database saturation.

## Screening and extraction

Use two passes: title/abstract relevance, then the accessible methods/theory/experiments sections. Deduplicate by normalized title plus arXiv/DOI/publication-family identifiers. Independently audit the extracted core methods using a second AI agent where feasible; this is not independent human dual screening.

Record: citation and stable URL, year/version/status, discovery query, screening decision and reason, read scope and section/page locators, privacy unit/adjacency, attacker observations, mechanism location, exact distribution/parameters, parameter estimator and its privacy treatment, clipping/sensitivity, composition/subsampling assumptions, reported evidence, applicability to CIA, and unresolved details.

Each technique explanation should provide the problem, an intuitive example, a step-by-step mechanism, the defining mathematics, source of its noise distribution, what the guarantee protects, limitations, and a concrete transfer assessment. Mark reviewer deductions separately from author claims. An abstract-only source cannot support a detailed implementation recipe.

## Critical appraisal and synthesis

Clinical RoB 2/ROBINS-I instruments are inapplicable to privacy proofs and simulation benchmarks. Use an explicitly adapted computational rubric: proof adjacency and adversary match; validity of private adaptation/accounting; attack adaptation and evaluation independence; matched utility/privacy comparisons; seeds/targets/uncertainty; code/artifact availability; and applicability to this repository. Record not-assessed rather than invent a score. No numerical aggregate quality score.

Do not pool accuracy or attack AUC across papers with different privacy units, tasks, adversaries, or experimental protocols. Use structured integrative synthesis and comparison tables. No clinical GRADE rating or meta-analysis is planned; if a comparable quantitative subset emerges, amend the protocol before pooling it.

Pre-specified comparisons: record versus client privacy; peer versus server adversary; isotropic versus shaped covariance; public versus private distribution construction; independent versus correlated client noise; fixed versus adaptive schedules; formal guarantees versus empirical protection; peer-reviewed versus preprint evidence. Review contradictory/negative evidence explicitly.

## Reporting and integrity

Deliver a navigable agent handoff, technical primer, technique cards, source/claim ledger, search and screening log, comparison matrix, integrative synthesis, and a provisional research roadmap. All claimed results must come from inspected sources or clearly labeled deductions. A research gap is a search-bounded hypothesis, not a certified absence. Source retrieval/AI reading is not a human-read declaration.

Report limitations: restricted repositories/search interface; English/open-access selection; no external registration; AI-assisted rather than human independent screening; and any missing full texts. Map PRISMA items honestly and do not claim complete PRISMA compliance from search documentation alone. Preserve the project's distinction between exploratory frontier scores and independent ROC-AUC trials. Do not launch experiments or declare one finished during this review.

## Protocol challenge before retrieval

- Scope is intentionally broader than CIA-only terminology because participation protection may appear under user/client-level DP. Control scope expansion by requiring an explicit transfer reason per source.
- Record-level methods are not interchangeable with client-participation guarantees. Extraction must preserve this distinction.
- Secure aggregation protects upload visibility but not automatically the released-model participation signal. Include it as a trust/placement mechanism, with appropriate qualification.
- Client-specific covariance may itself reveal private information. Inspect its construction and composition rather than assuming additive noise is enough.
- A bounded web-index search cannot establish that no relevant work exists. Retain the search limits alongside any novelty statement.
