# Systematic search and screening report

Version 1, search/read date 2026-10-01. This accompanies an integrative technical review. It documents a **bounded systematic search**, not a completed exhaustive database review or full PRISMA-compliant publication. The author's confirmed [protocol](protocol.md) fixes the scope. Primary methods rather than secondary summaries underpin the technique explanations.

## Search execution and captured-record flow

Queries cover participation/user privacy; local/personalized/adaptive perturbation; covariance/Fisher/PAC/learned samplers; distributed/correlated noise; private clipping; and membership/property/white-box attacks. Late-period 2025–2026 candidates and targeted citation expansion were included. Discovery used web-index searches of arXiv and proceedings/publisher/author repositories, not complete database exports. All family logs are retained with exact strings, date, decisions, read scope and locators.

| Logged quantity | Count | Interpretation |
|---|---:|---|
| Exact query executions / distinct strings | 51 / 51 | Includes source-identity and cutoff checks; not 51 independent databases |
| Captured ledger entries | 106 | Explicit candidate or locator decisions only; not all engine-returned hits |
| Include decisions before family reconciliation | 35 | Four overlap discoveries were delegated and recorded twice |
| Pending decisions before family reconciliation | 21 | Methods unassessed; may be accessible; two overlap candidates |
| Exclude decisions | 50 | Includes mirrors, indexes, companion locators and out-of-scope material |
| Included canonical source families | 31 | Includes two local papers, foundations and a correction; not 31 evaluated defenses |
| Pending canonical candidate families | 19 | One previously pending priority paper was subsequently read and included |

These numbers come directly from [screening_ledger.json](screening_ledger.json). Explicit aliases resolve cross-agent overlaps; locator exclusions remain recorded for traceability. The denominator of all search-engine hits is **unknown**. A PRISMA diagram claiming that 106 were all retrieved records or that 50 were all excluded studies would be misleading. No such diagram or total is asserted.

Family evidence: [foundations log](foundations_search_log.json), [client log](client_noise_search_log.json), [covariance log](covariance_search_log.json), [distributed log](distributed_search_log.json), [follow-up log](snowball_search_log.json). An exact query may produce material recorded in another family's extraction. Queries with ineffective `site.arxiv.org` syntax are preserved as actually executed, rather than silently rewritten as valid domain restrictions.

## Screening and read scope

The owner confirmed: **“Confirm this protocol.”** Exploratory source discovery and the project audit preceded confirmation; systematic source-family extraction and synthesis followed it. No external preregistration occurred. Methods-read inclusion requires a primary full text and a concrete role in construction, accounting, trust or attack evaluation. This is not a claim that every included PDF was read cover to cover. Proofs, supplements and source code were not comprehensively audited. Exact sections/theorems/pages are in the inventory and cards.

Three parallel AI agents extracted separate source families. They then cross-checked selected mathematical/privacy claims and source sections, with root integration. This provides a second AI audit of core ideas, **not independent human dual screening of every record**. Revisions addressed bounded client weights, CorN privacy conversion, RNS aggregate-visibility claims, privacy-unit distinctions and formatting. See the three retained review notes.

## Exclusions, pending evidence and adverse findings

Excluded entries include duplicate URLs/mirrors, author/index pages, secondary teaching summaries, patents and work outside the selected construction/evaluation scope. The ledger states individual reasons. Some exclusions are alternate locators of included papers and must not be treated as rejected mechanisms. A poisoning-focused paper was deferred on relevance, not because its result was unfavorable.

Pending candidates have only abstracts/search records or incomplete methods assessment. FACP and FedFR-ADP are especially close and may materially affect novelty. Their method details, private-statistic treatment and guarantees are not inferred from abstracts. Other queues include direction-aware local noise, dynamic privacy, preconditioning, reconstruction and source attacks. FedDriftGuard's publisher date was checked against the cutoff (2026-05-02; within scope), but its methods remain unassessed.

Adverse/contradictory evidence was retained: no universal anisotropic advantage under IMGM's full-ball objective; Geyer's unprotected adaptive median; NbAFL/priority sensitivity concerns; ALDP truncation/adaptation obligations; approximate decoder issues; observer-known seeds; and restrictions/questions in source shuffling. These are source statements or labeled reviewer deductions, not wholesale rejection of the papers.

## Critical appraisal and synthesis method

Use the [comparison/appraisal matrix](comparison_matrix.md). Dimensions: privacy unit/observer fit; estimator and adaptation validity; conditional accounting; attack adaptation and independence; matched utility/privacy comparisons; artifact/read completeness; and transfer to this repo. Clinical bias instruments and GRADE are inappropriate here. No numerical quality score was assigned. Unassessed code, seed or uncertainty details remain unassessed.

The [integrative synthesis](client_specific_noise_review.md) compares construction families, threat targets and proof obligations. No accuracy/AUC meta-analysis was performed because experiments differ in task, architecture, adjacency, privacy budget and observer. The frontier audit is a separate verified local evidence stream, not another compatible effect size to pool.

## PRISMA transparency map

| Reporting area | This package | Limitation |
|---|---|---|
| Objectives/eligibility | Protocol research question, adapted PICOS and eligibility | Broad computational scope, English-only |
| Information sources/search strings | Family JSON logs, primary URLs, cutoff/date | Indexed discovery; no exhaustive database export or hit universe |
| Selection/data extraction | Decisions, read scopes and section locators | AI extraction; no human dual screening of all records |
| Flow/exclusions | Captured counts and explicit reasons | Cannot supply complete retrieval-flow counts |
| Risk/appraisal | Adapted matrix and independent AI checks | No comprehensive proof/code audit or universal empirical assessment |
| Synthesis/results | Technique cards and integrative review | No quantitative pooling or common effect size |
| Registration/protocol amendments | Owner-confirmed repository protocol | Not externally registered; preliminary searches already existed |
| Support/automation | AI agents and web/PDF extraction disclosed | No human-read certification; no publication-level compliance audit |
| Availability | Git-tracked docs and machine-readable logs | Temporary PDFs not distributed; stable source links may later change |

## Reproducibility and next search

A new agent can rerun the exact queries, inspect canonical primary sources and reproduce selection judgments. Results may differ as indexes change. Extend the search by retrieving the closest pending full texts first, then adding direct ACM/IEEE and other database exports, forward citation checks and complete result screening if preparing a publication-grade systematic review. Record amendments and dates instead of representing an expanded search as part of the original version.

Version 1 answers the owner's immediate need for technique understanding and a grounded roadmap. It does not certify literature saturation or an unoccupied novelty gap.

## Subsequent follow-up

The [next-step supplement](next_step_handoff.md) documents comparator access/read updates and owner-directed non-Gaussian scope clarification. The 51-query version-1 flow above is preserved; later query executions and new source cards are separate records, not silently counted as original retrieval.
