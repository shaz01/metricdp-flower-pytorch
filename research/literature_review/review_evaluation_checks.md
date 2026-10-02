# Cross-review: covariance cards and attack/accounting foundations

Reviewer: evaluation/distributed-noise agent. Date: 2026-10-01. Reviewed `covariance_sources.md`, `foundations_and_attacks.md`, and `covariance_search_log.json`. This is a consistency/method-scope review, not independent proof certification or full double screening.

## Assessment

No blocking mathematical or adjacency error found in the inspected cards. They correctly separate record membership, whole-client participation, reconstruction, and category inference; distinguish distributional PAC/Fisher targets from worst-case DP; and flag private estimator/accounting obligations. Preserve those qualifications when condensing the root synthesis.

Primary-source spot checks: PAC Privacy PDF Theorems 3–4; PAC-Private Algorithms PDF Theorem 1/Algorithm 1; Residual-PAC proceedings PDF Proposition 5/Algorithm 2; LiRA primary HTML Section III/IV/VI headings and threat/evaluation framing; Nasr primary PDF attack taxonomy; IMGM local primary PDF experiment-scope locator. Full proofs and empirical claims were not rechecked.

## Changes/qualifications recommended for root synthesis

1. **Systematic-review completeness must be scoped.** The covariance ledger explicitly states that unselected search hits are not enumerated. This can support a documented focused source-family review, but its screened-record totals cannot be described as all retrieved hits from those queries. Report the denominator as explicitly screened candidates; separate that from returned pages and included source families. Avoid claiming exhaustive coverage, a complete PRISMA flow, independent dual screening, or saturation. The distributed ledger also includes excluded alternate pages; these are locator counts, not independent study counts.

2. **Keep theoretical calibration separate from finite-sample construction.** PAC Privacy's displayed covariance theorem assumes a deterministic M, an input-population law, and independent Gaussian perturbation with the specified covariance. PAC-Private Algorithms' directional formula is a true-variance theorem, while its practical variance convergence criterion is empirical. Root text should explicitly state deterministic M or account for internal algorithm randomness; neither covariance theorem certifies arbitrary training-data-dependent covariance reuse, conditional side information, or repeated adaptive FL releases.

3. **Learned-decoder loss is not a lower entropy certificate.** The Residual-PAC card correctly warns that decoder cross entropy upper-bounds true conditional entropy. Preserve this: a high empirical reconstruction loss could mean the decoder is weak rather than that all attackers have low information. Proposition 5 concerns an optimal Stackelberg solution with a rich decoder family; practical alternating neural training does not inherit that conclusion without the specified approximation/generalization analysis. The proceedings PDF points to the extended version for those details. An independent strong evaluation attacker remains necessary.

4. **Do not simplify IMGM into “anisotropic noise cannot help.”** Its conclusion is qualified by a full Frobenius sensitivity ball, fixed covariance, and noise-overhead objective. The card already states this correctly. A restricted sensitivity set, weighted utility objective, or modeled population can give a different optimization problem. Likewise MVG's sufficient condition does not establish that every chosen covariance is better than an analytically calibrated IID baseline.

5. **Whole-client LiRA is a proposed adaptation.** F04 labels the extension appropriately. Preserve independent attack-development/test trajectories and observer limits. Public global checkpoints are correlated attack features, rather than independent labeled examples; do not reuse the frontier's counterfactual paired score as ordinary trial-level ROC-AUC. Low-FPR recommendations need enough independent OUT realizations and uncertainty reporting; no arbitrary checkpoint/seed count guarantees this.

6. **Conditional randomness matters to peers.** F05 correctly calls for conditioning on the curious participant's own contribution/noise. The final plan must combine this with distributed-source trust assumptions: count only randomness unknown to the designated observer. A globally shared correlated-noise seed provides no hidden perturbation against a peer that knows it. Source-attribution aggregation defenses and noise placement alone do not establish participation privacy.

7. **Metadata and scope should remain visible.** F03's distinction between repository `global-dp` and Gaussian Differential Privacy is correct. COV-A01 is abstract-only and category inference, so it belongs in a pending-comparator list, not a proof/implementation comparison. Its claimed user-level guarantee is unassessed. PAC publication metadata traced through an author overview should be identified as such unless independently confirmed against proceedings; retain the inspected version when distinguishing publication year from preprint year.

## Useful synthesis statement

Prior literature already addresses whole-user add/remove privacy and construction of geometry-aware or learned noise. The unresolved project question is therefore narrower: whether a particular locally constructed distribution, under an explicit visibility/adjacency model and complete estimation/composition analysis, improves this project's measured client-participation privacy–utility trade-off. Existing cards support investigating that question; they do not establish a novelty claim or select the winning mechanism.
