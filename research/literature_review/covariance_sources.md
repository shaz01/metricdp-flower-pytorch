# Covariance and optimized-noise source family

Date: 2026-10-01. These six technique cards support the systematic-plus-integrative review. Search/screening decisions are in [covariance_search_log.json](covariance_search_log.json). Methods were read from primary full-text PDFs; proofs were not exhaustively audited and no mechanisms were reproduced. Exact section/theorem locators below support agent follow-up. New client-specific mechanisms and novelty remain open questions.

## COV-01: PAC Privacy - construct noise from output variation

**Xiao and Devadas, PAC Privacy: Automatic Privacy Measurement and Control of Data Processing** (arXiv 2210.03458; 2023 CRYPTO version identified by the authors' subsequent publication).

Repeatedly sample input datasets X from a specified population, run the mechanism M, and estimate covariance of its outputs. Add independent Gaussian noise B with covariance V. The bound is `I(X; M(X)+B) <= 0.5 log det(I + Cov(M(X)) V^-1)`. Eigenvectors identify output directions; their variances determine noise allocation. Algorithm 1 includes estimation safety margins and an isotropic fallback when eigenvector estimation is unreliable. Its confidence theorem requires uniformly bounded outputs and adequate independent simulations.

**Plain meaning:** hide the variation that reveals the unknown input, spending noise according to measured output geometry. This controls distribution-dependent inference/reconstruction through mutual information, not worst-case neighboring-dataset DP. Monte Carlo calibration is not an automatic DP release of a covariance learned from one private dataset: the assumed population/sampling model and attacker side information must be specified. For CIA, X must include the client-participation secret and relevant background uncertainty; local record bootstrap variance alone is not that model.

**Read:** Section 5.1, Theorems 3-4, Eq. (5), Algorithm 1, pp. 10-11; primary [full text](https://arxiv.org/pdf/2210.03458).

## COV-02: PAC-Private Algorithms - cheaper directional construction

**Sridhar, Xiao, and Devadas, PAC-Private Algorithms** (primary 2024/718 preprint; IEEE S&P 2025 publication identified in the authors' later overview).

Choose a public orthonormal basis A. Simulate outputs and estimate variance s_j along each basis direction, avoiding a full covariance decomposition. Sample Gaussian noise with directional variance `v_j = sqrt(s_j) sum_k sqrt(s_k) / (2 beta)`, where beta is the mutual-information budget. High-variance directions receive more noise; zero variance requires careful treatment of support and estimation. Theorem 1 proves the bound using true variances. Algorithm 1 uses an empirical convergence rule, which must be distinguished from a finite-sample confidence certificate.

**Plain meaning:** a client could potentially simulate its algorithm under a defined input distribution and build a diagonal sampler cheaply. The paper demonstrates black-box algorithms including clustering, SVM, PCA, and random forests, and stabilizes superficial output variation before adding noise. The basis is public in the theorem; a basis learned privately needs separate treatment. Reusing simulations on an actual private client's data does not automatically establish user-level DP. Membership-specific modifications appear in Appendix A, but whole-client participation with heterogeneous local datasets still needs its own input/side-information model.

**Read:** Sections 4.1, 5; Algorithm 1 and Theorem 1; [primary PDF](https://eprint.iacr.org/2024/718.pdf).

## COV-03: Fisher information loss - measure which records leak

**Hannun, Guo, and van der Maaten, Measuring Data Leakage in Machine-Learning Models with Fisher Information** (UAI 2021, PMLR161:760–770; [proceedings metadata](https://proceedings.mlr.press/v161/hannun21a.html)).

Release a fitted model `h=f(D)+N(0,sigma^2 I)`. For this output perturbation, the data Fisher-information matrix is `I_h(D)=J_f^T J_f / sigma^2`, where J_f differentiates fitted parameters with respect to data. FIL uses the largest singular value: `eta=||J_f||_2/sigma`. Smaller FIL raises a Cramer-Rao lower bound on **unbiased** reconstruction error under regularity assumptions. This is Fisher information about the data, not merely the usual parameter-importance Fisher matrix.

**Plain meaning:** identify vulnerable examples or attributes by how strongly changing them changes the fitted model relative to noise. IRFIL iteratively reduces high-leakage examples' loss weights; the defense described here still uses isotropic noise, rather than constructing a client covariance. Derivatives/Hessian inverses are tractable for regularized generalized linear models, with costs that limit direct transfer to deep FL. Continuous local reconstruction sensitivity does not directly bound a discrete whole-client IN/OUT decision, especially for biased attackers. Private data-derived weights/statistics cannot be released or plugged into a DP accountant without analyzing the complete mechanism.

**Read:** Sections 3-5, Eqs. (16)-(18), Algorithm 1; [primary full text](https://arxiv.org/pdf/2102.11673).

## COV-04: Matrix-variate Gaussian mechanism - design two covariances

**Chanyaswad, Dytso, Poor, and Mittal, MVG Mechanism: Differential Privacy under Matrix-Valued Query** (CCS 2018).

For a matrix query F(D), add matrix-normal Z with row covariance Sigma and column covariance Psi. In vector form its covariance is `Psi tensor Sigma`; a sampler is `Z=L_row G L_col^T` with standard-normal G and covariance square roots L. Theorem 3 supplies a sufficient DP constraint on inverse-covariance singular values using Frobenius sensitivity and a bound on query norm. Directional precision allocation spends less noise along useful directions while retaining noise everywhere required by the guarantee.

**Plain meaning:** layer-shaped client updates can have correlated row/column perturbations instead of independent entries. Directions may come from public domain knowledge or DP SVD/PCA; Section 5.4.2 explicitly spends privacy budget on privately deriving directions. Thus learning geometry is already recognized as a separate privacy operation. The protected unit is the query's neighboring record; whole-client protection requires redefining adjacency and bounding the complete client's update. This construction is not automatically superior to isotropic Gaussian noise: its sufficient constraint is conservative, and the next paper changes the conclusion under a full sensitivity-ball model.

**Read:** Definitions 2-4, Theorem 3, Sections 5.3-6.3; [author-hosted published PDF](https://swh.princeton.edu/~pmittal/publications/mvg-ccs18.pdf).

## COV-05: Improved Matrix Gaussian Mechanism - counterexample to universal anisotropy

**Yang, Xiang, Li, Liu, and Wang, Improved Matrix Gaussian Mechanism for Differential Privacy** (arXiv 2104.14808, 2021 preprint).

For fixed Gaussian covariance, a neighboring mean shift Delta must have bounded whitened norm: `||U1^-1 Delta U2^-T||_F <= B`, with covariance factors U1,U2 and B solved from the analytic Gaussian privacy equation. The paper considers every shift in a Frobenius sensitivity ball. Its worst direction constrains the smallest covariance-factor singular values. Setting all singular values to their minimum permitted level minimizes noise overhead, yielding IID Gaussian noise with standard deviation `s2(F)/B`.

**Plain meaning:** rotating noise cannot evade a worst-case shift that may point anywhere. Anisotropic noise is not universally better merely because client gradients have correlations. A benefit needs a restricted sensitivity geometry, a weighted utility objective, or a distributional privacy target. The inspected preprint's Type-I gradient experiments are explicitly still in progress; do not claim completed empirical FL validation from its abstract. No private covariance estimate is needed by its IID baseline. For CIA, a user-level sensitivity bound and correctly observed transcript can transfer the Gaussian principle, but record sensitivity does not itself protect a whole client.

**Read:** Section 3.1, Lemma 3, Theorems 1-3, Algorithm 1, Section 4.1, Section 5.2; [primary PDF](https://arxiv.org/pdf/2104.14808).

## COV-06: Residual-PAC - learn a distribution beyond Gaussian covariance

**Zhang and Vorobeychik, Residual-PAC Privacy: Automatic Privacy Control Beyond the Gaussian Barrier** (USENIX Security 2026 prepublication).

The authors distinguish a Gaussian covariance upper bound from true leakage of non-Gaussian outputs. Their SR-PAC mechanism selects a noise distribution Q to minimize expected utility loss while requiring sufficient conditional entropy `H(X|M(X)+B)`, with B drawn from Q. A decoder attempts to reconstruct X; alternating Monte Carlo updates train the decoder and perturbation rule. Unlike a variance-only rule, this optimizes a distribution family against inference.

**Plain meaning:** a client could learn a sampler rather than choose one Gaussian scale. However, the theoretical optimal-decoder problem and a trained neural approximation differ. Decoder cross entropy upper-bounds true conditional entropy; a weak decoder can therefore overstate residual privacy. A constraint on its empirical loss is not automatically a certificate against all attackers. The paper discusses finite-sample/optimization analysis in an extended appendix, not fully available in this proceedings file. Noise-family restrictions, population access, private calibration, and temporal composition must be resolved before whole-client transfer. Its distribution-learning idea is already prior work; novelty cannot be claimed simply for a learned sampler.

**Read:** Sections 3-4.2, Eq. (13), Algorithm 2, Proposition 5, Section 5 assumptions; [primary proceedings PDF](https://www.usenix.org/system/files/conference/usenixsecurity26/sec26_prepub_zhang-tao.pdf).

## Immediate comparator requiring full-text acquisition

**Fisher-driven privacy preservation against category inference attacks in federated learning**, High-Confidence Computing, available online 13 May 2026, DOI [10.1016/j.hcc.2026.100399](https://doi.org/10.1016/j.hcc.2026.100399).

Publisher abstract describes FACP: historical update statistics plus global Fisher importance for adaptive clipping, and anisotropic Gaussian allocation inversely related to parameter importance, claiming user-level DP. **Version-1 read scope: publisher abstract only; full publisher access returned HTTP 403.** Subsequent [follow-up](facp_followup.md) recovered selected primary method sections, including the explicit exclusion of uploaded Fisher statistics from DP accounting. Complete artifact and Algorithm 1 remain pending. The complete covariance construction, privatization, accountant and adjacency still need verification; the follow-up distinguishes recovered passages from unresolved details. Category inference recovers label distribution; it is not client participation inference despite sharing the abbreviation CIA. This is a close mechanism comparator that must be read before novelty claims.

## Integration for our research question

The literature already contains several distinct answers to distribution construction: output-covariance estimation (PAC), fixed-basis projected variances (PAC-Private Algorithms), matrix covariance factors with privatized direction discovery (MVG), and adversarially optimized distribution families (Residual-PAC). These do not prove that one answers our whole-client problem. Conversely, IMGM shows why anisotropy alone cannot guarantee a gain.

A useful candidate taxonomy is: (a) fixed/public geometry; (b) geometry estimated from already privatized updates; (c) geometry learned from private local data with explicit accounting; (d) distributional protection under a modeled client population. Comparing these avoids treating all data-dependent noise as one method.

For every candidate, specify the secret (record, whole-client dataset, participation bit, or label distribution), observation (individual upload versus aggregate/global trajectory), estimator/sampler, covariance rank and regularization, privacy target, and composition. A client upload may expose its presence through participation metadata even if the numeric update is privatized; our global-model observer and a server observer require distinct games. These are integration judgments, not claims established by any single paper.
