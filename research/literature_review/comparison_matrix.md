# Mechanism comparison and critical appraisal

2026-10-01. This adapted computational rubric replaces unsuitable clinical risk-of-bias scoring. No aggregate quality score is used. “Methods read” means selected algorithm/theory/experiment sections were inspected, not all proofs independently validated. Source-card links provide evidence and exact locators. Experiment reproducibility, code completeness and uncertainty reporting were not uniformly audited; those dimensions remain not assessed unless explicitly noted.

## Construction and guarantee matrix

| Sources | Distribution construction | Privacy unit / observation | Private calibration and accounting issue | Evidence applicable to our CIA |
|---|---|---|---|---|
| LOCAL-01 original paper | Inverse inter-client-distance server Gaussian scale | Target participation / curious peer global models | Raw data-dependent scale; heuristic, not established metric-DP certificate | Direct attack motivation; pooled ROC differs from repo score; [audit](../project_evidence_audit.md) |
| LOCAL-02 GMIP | Isotropic gradient noise; natural covariance in attacker analysis | Population record membership / model knowledge | IID population and asymptotic assumptions; local PDF supplements missing | No whole-client theorem; [tutorial](local_gaussian_analysis.md) |
| CN1 LDP-Fed | Discretized distance-weighted exponential output law | Local numerical/metric privacy; anonymous selection | High-dimensional composition and visibility conditions matter | Useful individualized non-Gaussian comparator; [card](client_noise_sources.md) |
| CN2 time-adaptive | Gaussian shares, privacy schedules and client clipping | Client level / trusted server, curious peers | Actual client covariance equal within round; schedule/accountant assumptions | Closest verified threat/placement precedent; no frontier test |
| CN3 NbAFL; CN3C comment | Gaussian uplink plus optional downlink noise | Record replacement / channel observers | Unsupported general final-model `2C/m` sensitivity shortcut; published convergence correction | Historical placement baseline with caveats, not reliable generic deep-training certificate |
| CN4 ALDP-FL | Private layer history and truncated Gaussian | Authors assert privacy; iDLG reconstruction evidence | Support-tail delta and private calibration require analysis | Reconstruction is not participation; no end-to-end proof certified |
| CN5 adaptive clipping | Noisy quantile statistic calibrates clipping | User-level aggregate DP | Explicit estimation cost plus model accounting | Useful private-adaptation template; not individual upload privacy |
| CN6 DP-SGD | Per-example clipping and minibatch Gaussian | Record adjacency / protected training release | Record accountant does not cover whole-client removal automatically | Fundamental baseline principles, not direct CIA proof |
| CN7 privacy-aligned personalized | Fixed coefficient-space Gaussian, compact representation | Central record-level DP / trusted server | Geometry from architecture/optimization; context release accounted per source | Recent dimension-reduction comparator; different protected unit |
| COV-01 PAC; COV-02 PAC-Private | Output covariance or public-basis projected variances | Population information / modeled input uncertainty | Population/simulation access, finite-sample confidence, true versus empirical variance | Direct distribution-construction precedent; client secret/conditioning must be redefined |
| COV-03 FIL/IRFIL | Isotropic release; data-Jacobian leakage, reweighting | Continuous data reconstruction / unbiased regular estimators | Private weight/statistic treatment needs complete mechanism analysis | Diagnostic, not universal discrete participation bound |
| COV-04 MVG | Matrix-normal row/column covariance | Neighboring query / stated DP assumptions | Private directions require budget; source sufficient bound conservative | Geometry principle transferable only with client sensitivity |
| COV-05 IMGM | IID optimum under full Frobenius sensitivity ball | Fixed Gaussian query DP | Optimality tied to full-ball and noise-overhead objective | Negative control against universal anisotropic claims; incomplete gradient experiments in read version |
| COV-06 Residual-PAC | Optimized noise family and decoder | Distributional conditional-entropy target | Weak neural decoder can overstate privacy; full extended appendix unread | Strong learned-sampler precedent; no whole-client transcript certificate |
| COV-S01 priority adaptation | Private feature priority, additional coordinate perturbation | Record change / upload perturbation | Inherits sensitivity shortcut; private importance/noise law unaccounted in read methods | Construction precedent with MNIST utility, not adapted CIA evidence |
| D01 McMahan; D02 Geyer | Client clipping, central Gaussian and sampling | Whole-user/client contribution / released model | D01 sensitivity requires bounded weights/denominator; D02 admits private median issue | Fundamental client-privacy baselines; cannot assume small-client utility transfers |
| D03 discrete Gaussian; D04 Skellam | Client integer noise under secure sum | Whole-user adjacency / secure aggregate | Rounding/modulus, honest residual shares, dropout/collusion; different convolution laws | Formal aggregate-noise precedents; not learned client geometry or measured CIA frontier |
| D05 secure aggregation | Cryptographic masks, no DP noise by itself | Individual input visibility / defined aggregate adversary | Protocol thresholds and sums still disclose information | Enables distributed calibration; not model participation privacy |
| D06 CAPE | Cross-site zero-sum noise plus independent residual | Single-entry record adjacency / joint HBC observations | Collusion restrictions; not removal of a site | Correlation precedent only |
| D07 CorN/Whisper family | Cross-agent covariance from topology SDP | Agent dataset change / specified observer | Common seed exposes noise to peers; HBC extension needed; exact epsilon conversion check | Coordinated-noise precedent, not per-client private parameter covariance |
| D08 2026 source protection | RNS/unary encoding plus shuffling | Source attribution / HBC server and honest shuffler | Restricted chance argument; residue counts can exceed original-sum information; CRT bound check | Important boundary/negative inference, not a CIA DP defense |
| F01 RDP; F02 analytic Gaussian; F03 Gaussian DP | Formal calibration and composition | Specified adjacency / entire release law | Conditional uniform bounds and fixed-covariance assumptions | Foundations for a proof, not empirical superiority |
| F04 LiRA; F05 Melis; F06 Nasr | Attacks, not noise constructors | Record membership/properties / different observations | Shadow calibration, own-update subtraction, stronger features and active probes | Evaluation precedents; client/transcript extension proposed here |

## Appraisal conclusions

**Threat/adjacency fit:** D01/D02 and CN2 explicitly address client/user participation, while other families provide transferable construction tools or boundary attacks. Their privacy definitions cannot be pooled or relabeled. Client-level adjacency alone is also insufficient if the observer sees participation identities.

**Private adaptation:** CN5 and MVG explicitly treat estimation/direction acquisition as privacy-sensitive operations. Geyer acknowledges an unprotected private median; NbAFL/priority adaptation have sensitivity concerns; ALDP's truncated support needs special treatment. These concerns are evidence to retain rather than exclusions chosen to favor our proposal.

**Attack evidence and independence:** no reviewed source was reproduced against this repository's frontier. Record reconstruction and source/category inference do not constitute CIA evidence. Our verified frontier remains exploratory and requires independent trajectory/target units. Do not borrow another paper's accuracy improvement as our expected effect size.

**Utility matching and external validity:** source tasks, architectures, client populations and privacy units differ. Reported utility cannot establish dominance here. Many-user language modeling and small cross-silo learning have distinct noise/utility regimes. The protocol therefore forbids pooling their numbers.

**Artifacts and proof certification:** selected full methods and targeted equations were inspected. Download hashes exist for some agent sources; no universal code audit or formal proof verification was performed. Missing supplements, empirical covariance convergence, decoder approximations and observer-known seeds are concrete unresolved risks. All otherwise unassessed dimensions stay unassessed.

## Pending comparators that can change the roadmap

FACP and FedFR-ADP are highest priority because their abstracts describe closely related anisotropic or heterogeneity-driven client noise. Other pending candidates, including direction-aware LDP, dynamic privacy, reconstruction bounds and data-free preconditioning, are listed in the [inventory](source_inventory.md). Pending means methods not assessed; it does not imply irrelevance or nonexistence. Novelty remains provisional until this queue is resolved and broader database coverage is obtained if publication claims require it.
