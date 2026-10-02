# Cross-review: observer conditioning, units, and proof boundaries

Date: 2026-10-01. Reviewed `distributed_sources.md` and `local_gaussian_analysis.md`, with targeted checks against locally cached primary PDFs/text. This is a limited technical review, not independent certification of complete proofs.

## Findings requiring qualification before synthesis

### D08: distinguish paper claims from what its information argument establishes

The source's Appendix A, pp.19–20, establishes an equivalence between a shuffled unary encoding and its **per-residue bit count/sum**, then asserts chance source inference because no separate client models remain to compare. Hiding separate uploads does not generally imply chance posterior source inference given aggregate models, auxiliary client distributions, and arbitrary priors. The card already marks Theorem 1 as an assertion under the paper's formulation; retain this scope and do not upgrade it to universal source secrecy.

There is a separate potential gap in “only the aggregate model is revealed.” **Mathematical counterexample to that broad description:** two original integer client inputs `(0,6)` and `(3,3)` both sum to six. Modulo five, their residue sums are respectively one and six. A shuffled concatenation of unary residue encodings exposes this raw residue sum through its number of one bits. Thus raw per-residue sums can distinguish decompositions with identical original total. Decoding through CRT may yield the same original sum while the observed bit counts contain extra information. This is a proof/encoding verification question; it should remain visible instead of treating sum-only disclosure as independently verified.

Algorithm 1 on PDF p.7 prints a product-of-moduli inequality below a possible sum bound. Unique CRT recovery of an integer over a known range generally requires that range to be smaller than the product, with additional care for signs/shifts. Do not implement the printed bound without checking intended conventions. The card's CRT explanation can remain conceptual if this implementation issue is noted.

### D07: check the privacy-budget domain of the printed SDP constraint

The displayed inverse-covariance constraint matches the paper's Eq.(9). A standard Gaussian/RDP conversion gives, with $v=\max_i(R^{-1})_{ii}$ and $L=\log(1/\delta)$,

$$
\rho=2C^2Tv,\qquad \varepsilon_{\mathrm{bound}}=\rho+2\sqrt{\rho L}.
$$

Substituting the card's $v\le\varepsilon^2/(16C^2TL)$ yields

$$
\varepsilon_{\mathrm{bound}}
\le\varepsilon^2/(8L)+\varepsilon/\sqrt2.
$$

This is at most the requested $\varepsilon$ only under a suitable range, specifically $\varepsilon\le8L(1-1/\sqrt2)$ for this conservative substitution. The source's Proposition 1 asserts the final inequality without an explicit range in the inspected paragraph. Preserve the equation as the authors' formulation; before using it as a certificate, evaluate the exact converted bound or derive a sufficient constraint for the chosen budget. This does not invalidate the general covariance-design idea.

### D01: make weight and denominator assumptions explicit

The source, pp.3–5, uses bounded weights $w_k\in[0,1]$ and distinguishes fixed and bounded-denominator estimators. Its sensitivity proof needs the weighted contribution to be bounded; clipping an unweighted update suffices because weights are at most one. The displayed fixed-denominator formula should state that $qW$ is treated as fixed/public under the analysis. It is not automatically applicable to private sample-count weights that change with a removed client. The card's noise units are correct: the stated $\sigma$ is standard deviation of noise on the average.

## Checks with no blocking issue found

- **D03:** discrete Gaussian scale $\sigma/\gamma$ correctly separates real-coordinate noise scale and grid units. Sums of discrete Gaussians are correctly distinguished from exact continuous Gaussian closure.
- **D04:** Skellam $\mu$ is variance; each Poisson mean is $\mu/2$. The card does not confuse variance with standard deviation.
- **D05:** secure aggregation is correctly described as a visibility primitive, with no automatic DP or participation-metadata guarantee.
- **D06:** equal-site CAPE residual variance $\tau^2/S$ is consistent. Averaging $S$ residuals gives final variance $\tau^2/S^2$. Zero-sum noise cancellation and record-level adjacency are clearly distinguished from whole-client removal. Its joint-observation caveat is essential.
- **D07 observer conditioning:** the shared-seed base protocol and honest-but-curious extension are properly distinguished. Keep privacy tied to noise unknown to the coalition; a participating adversary with the common seed can compute and subtract that base protocol's randomness.
- **Transfer table:** correctly treats aggregate-law equality as conditional on matched clipping, optimizer path, covariance, sampling, and side information. Equal marginal model covariance alone is insufficient against a peer who knows its own noise.
- **GMIP source analysis:** accurately separates record membership, the population-dependent game, and whole-client CIA; separates gradient population covariance from isotropic defense noise; states mean-versus-sum units; qualifies CLT/asymptotic composition and missing supplements. Printed clipping and pseudocode issues are framed as verification tasks, rather than global rejection of the paper.
- **GMIP formula scope:** the effective-batch-size and approximate privacy formulas are source statements, not generally valid rules for arbitrary client covariance. Retain their displayed assumptions and the supplement queue. No new federated privacy certificate follows from them.

## Integration recommendation

Carry D08 and D07 as **included with theorem/implementation questions**, rather than proof-certified baselines. Distinguish source assertions, explicit mathematical checks, and experiments actually reproduced. The remaining cards and local GMIP note supply a sound conceptual handoff under their stated evidence limits.
