# Infinitely divisible non-Gaussian noise: focused follow-up

2026-10-06. Purpose: identify construction predecessors for the [observer-specific contract](../proposals/2026-10-06_observer_contract_review.md). This is a focused integrative update, not a rerun of version-1 systematic screening. Its 31 included primary families remain frozen; the two new families below are tracked separately. Neither is a proven client-specific FL defense in our project.

## Why this family matters

Infinite divisibility means one desired noise law can be decomposed into independent shares. Clients generate these shares locally; aggregation recovers the desired law. The protected output may be the aggregate even when each share is too weak for upload privacy. The attacker's own coins must be subtracted in the proof. A lower bound on remaining unknown shares controls dropout/collusion safety.

Distribution decomposition is established. Potential research value must lie in constructing a useful client-dependent aggregate law with complete accounting, rather than reintroducing this property as novelty.

## Pagh and Stausholm: Arete

**Verified metadata:** ALT 2022, PMLR167:881–909. [Proceedings](https://proceedings.mlr.press/v167/pagh22a.html); [official PDF](https://proceedings.mlr.press/v167/pagh22a/pagh22a.pdf); [arXiv v3](https://arxiv.org/abs/2110.06559v3). Read introduction, Definition1/9, Lemmas2–3 and Corollary5, plus Appendix A overview in arXiv. Proofs not independently reconstructed.

**Source-derived explanation:** Arete draws X1,X2 independently from Gamma(alpha,theta) and adds independent Laplace(lambda): Z=X1−X2+Laplace(lambda). The Gamma difference concentrates mass; Laplace smooths the density. Both components admit independent-share decomposition. The paper proves a scalar pure-DP mechanism with exponentially decreasing error in a large-epsilon regime. Its normalized sensitivity-one parameter certificate requires epsilon≥20. Smaller-epsilon plots are illustrative and do not inherit that certificate. Read scopes: official PDF pp.1–4, Definition9 in §3.2; arXiv Appendix A pp.21–22. Independent coordinate releases require vector privacy accounting.

**Project inference:** budget labels4/8/16 from earlier diagnostics are below that normalized formal range even before distributing budget across coordinates/rounds. Do not copy attractive plotted samples and declare them certified. Arete is a predecessor, not an unexplored new density. Scalar noise optimization is different from privately estimating each client's geometry.

## Harrison and Manurangsi: generalized and multi-scale Laplace

**Verified metadata:** FORC2025, LIPIcs329, article12. [Official record](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.FORC.2025.12); [official PDF](https://drops.dagstuhl.de/storage/00lipics/lipics-vol329-forc2025/LIPIcs.FORC.2025.12/LIPIcs.FORC.2025.12.pdf); [arXiv v1](https://arxiv.org/abs/2504.05202v1). Read arXiv abstract/introduction and sampler overview around Proposition24/Algorithms1–3; complete mechanism proofs, continuous transformation and finite-parameter constants remain pending.

**Source-derived explanation:** The generalized discrete Laplace mechanism uses differences of negative-binomial shares; the multi-scale construction combines discrete Laplace components across scales. Independent shares recover aggregate noise through infinite divisibility. The paper establishes high-epsilon scalar MSE bounds and a discrete-to-continuous transformation improving the Arete bound, plus an exact sampling method. The near-optimal scalar error result concerns query-independent additive noise, not a private distribution estimator or whole-client FL trajectory. Its stated regime includes epsilon≥1, making it a higher-priority methods comparator than Arete for our earlier budget labels.

**Project inference:** begin with its full parameter constraints and sensitivity model, not the asymptotic error bound. Derive the residual noise after a peer subtracts its shares. More honest shares can be treated as independent convolution only when they are shares of the same certified law and their parameters do not vary with unaccounted private input. Evaluate integer quantization, rounding sensitivity, modular arithmetic and numerical sampling before claiming a practical continuous-vector mechanism. This assessment deliberately does not select parameters or claim a certified epsilon4/8/16 implementation.

## Established distributed Gaussian/Skellam comparators

[Distributed discrete Gaussian, ICML2021](https://proceedings.mlr.press/v139/kairouz21a.html) and [Skellam, NeurIPS2021](https://papers.neurips.cc/paper_files/paper/2021/hash/285baacbdf8fda1de94b19282acd23e2-Abstract.html) remain required controls. The former implements discretized updates and modular secure summation; the latter uses differences of Poisson variables with closure under summation. Their secure-aggregation/finite-precision setting is stronger than our proposed trusted-server access boundary. Existing [cards](distributed_sources.md) preserve inspected theorem scopes. No dominance over these methods follows from the 2025 scalar asymptotic result.

## Agent pickup checklist

1. Reconstruct the exact generalized/multi-scale distributions, finite-parameter DP certificate, share sampler and continuous transformation from full primary methods.
2. Derive coalition/dropout residual laws and weight handling for fixed-slot dataset-contribution adjacency.
3. Choose a defensible vector accounting route; scalar epsilon cannot be reused independently in every coordinate or round.
4. Compare strong central implementations of the same law and Gaussian/Skellam references before proposing personalization.
5. Identify what a client-specific estimated statistic changes, and account for construction as well as perturbation. Keep raw-data oracle calibration labeled as such.

No pooled utility comparison, new experimental run or novelty certificate is produced here.

## Focused search provenance and limitations

Search date2026-10-06. Queries: `distributed differential privacy Laplace noise gamma infinitely divisible shares Goryczka Xiong 2015`; `distributed discrete Gaussian secure aggregation Kairouz Liu Steinke 2021 honest clients collusion Skellam mechanism`; `site.proceedings.mlr.press Pagh Stausholm Infinitely Divisible Noise Low Privacy Regime ALT 2022`; `Harrison Manurangsi 2025 infinitely divisible noise FORC drops dagstuhl 2025`. Search engines are discovery only; claims above use primary author/proceedings artifacts. Arete and the FORC paper are two families, not separate arXiv/proceedings inclusions. Goryczka–Xiong PMC full-text access returned a browser check; it is not method-appraised here. No claim of exhaustive search or absence of newer competitors. Literature claims and our transfer deductions are separated explicitly.
