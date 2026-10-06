# Included-source inventory

Generated from the family ledgers on 2026-10-01. Inclusion means methods relevance, not proof certification. Read-scope details and appraisal are in source cards/logs. Supplemental corrections count as sources, not separate mechanisms.

| ID | Primary source | Status | Methods locator |
|---|---|---|---|
| CN1 | [LDP-Fed: Federated Learning with Local Differential Privacy](https://arxiv.org/html/2006.03637v1) | EdgeSys 2020 | Sections 2.3-3; Section 4 |
| CN2 | [Differentially Private Federated Learning With Time-Adaptive Privacy Spending](https://arxiv.org/abs/2502.18706) | ICLR 2025 | See client_noise_sources.md CN2 |
| CN3 | [Federated Learning with Differential Privacy: Algorithms and Performance Analysis](https://arxiv.org/pdf/1911.00222v2) | 2019 preprint; later journal publication | Section II.B and III.A, Algorithm1, Eqs4-10; PDF pp3-5 |
| CN3C | [Comments on Federated Learning with Differential Privacy: Algorithms and Performance Analysis](https://arxiv.org/pdf/2406.05858v1) | preprint status; publication not verified | PDF pp1-2, Eqs2-10 |
| CN4 | [ALDP-FL for adaptive local differential privacy in federated learning](https://www.nature.com/articles/s41598-025-12575-6) | Scientific Reports 15:26679 | Client-side algorithm, bounded perturbation, privacy-security analysis; PDF pp6-10, Eqs3-21 |
| CN5 | [Differentially Private Learning with Adaptive Clipping](https://papers.neurips.cc/paper_files/paper/2021/file/91cff01af640a24e7f9f7a5ab407889f-Paper.pdf) | NeurIPS 2021 | Sections2-2.1, Algorithm1 and accounting; PDF pp3-6 |
| CN6 | [Deep Learning with Differential Privacy](https://arxiv.org/pdf/1607.00133) | CCS 2016 | Section3, Algorithm1, moments accountant; PDF pp3-4 |
| CN7 | [Privacy-Aligned Personalized Federated Learning with Compact Adaptation and Variable-Length Gaussian Communication](https://arxiv.org/abs/2609.15950) | September 2026 preprint | See client_noise_sources.md CN7 |
| COV-01 | [PAC Privacy: Automatic Privacy Measurement and Control of Data Processing](https://arxiv.org/abs/2210.03458) | CRYPTO 2023; arXiv original 2022, inspected primary arXiv PDF | Section 5.1; Theorems 3 and 4; Equation 5; Algorithm 1, pages 10-11 |
| COV-02 | [PAC-Private Algorithms](https://eprint.iacr.org/2024/718.pdf) | IEEE S&P 2025 identified in author overview; inspected eprint 2024/718 PDF | Section 4.1; Algorithm 1; Theorem 1; Section 5 |
| COV-03 | [Measuring Data Leakage in Machine-Learning Models with Fisher Information](https://proceedings.mlr.press/v161/hannun21a.html) | UAI 2021 PMLR161 | See covariance_sources.md COV-03 |
| COV-04 | [MVG Mechanism: Differential Privacy under Matrix-Valued Query](https://swh.princeton.edu/~pmittal/publications/mvg-ccs18.pdf) | CCS 2018 published paper, author-hosted PDF | Definitions 2-4; Theorem 3; Sections 5.3-6.3 |
| COV-05 | [Improved Matrix Gaussian Mechanism for Differential Privacy](https://arxiv.org/abs/2104.14808) | Primary arXiv preprint; inspected version has gradient experiments in progress | Lemma 3; Theorems 1-3; Algorithm 1; Sections 4.1 and 5.2 |
| COV-06 | [Residual-PAC Privacy: Automatic Privacy Control Beyond the Gaussian Barrier](https://www.usenix.org/system/files/conference/usenixsecurity26/sec26_prepub_zhang-tao.pdf) | USENIX Security 2026 prepublication PDF | Section 4.2; Equation 13; Algorithm 2; Proposition 5; Assumption 1 |
| D01 | [Learning Differentially Private Recurrent Language Models](https://arxiv.org/abs/1710.06963) | ICLR 2018 | Sections 1-3; Definition 2; Algorithm 1; Lemma 1; Theorem 1; Figures 2-4 |
| D02 | [Differentially Private Federated Learning: A Client Level Perspective](https://arxiv.org/abs/1712.07557v2) | NIPS 2017 workshop; arXiv v2 March 2018 | Sections 2-5; Section 3 median-clipping caveat; Section 3 averaging formula |
| D03 | [The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation](https://proceedings.mlr.press/v139/kairouz21a.html) | ICML 2021 | Sections 2-6; Algorithms 1-2; Theorems 5,8,9; Section 4 Distributed DP |
| D04 | [The Skellam Mechanism for Differentially Private Federated Learning](https://arxiv.org/abs/2110.04995) | NeurIPS 2021 | Sections 2-5; Definitions 2.1,3.1,3.2; Theorem 3.5; Corollary 3.6 |
| D05 | [Practical Secure Aggregation for Privacy-Preserving Machine Learning](https://acmccs.github.io/papers/p1175-bonawitzA.pdf) | ACM CCS 2017 | Sections 1-2,4-6,8; Figure 4; Lemma 6.1; Theorem 6.2 |
| D06 | [Distributed Differentially Private Computation of Functions with Correlated Noise](https://arxiv.org/abs/1904.10059v3) | arXiv v3 February 2021; partially subsumed by 1910.12913 | Section 2; Sections 3.1-3.4; Algorithms 1-2; Theorem 1; Lemma 1 |
| D07 | [Optimizing Privacy-Utility Trade-off in Decentralized Learning with Generalized Correlated Noise](https://arxiv.org/abs/2501.14644v2) | Accepted IEEE ITW 2025 per primary metadata | Sections II-VI; Theorems 1-2; Algorithm 1; Equations 8-10; Section V-D |
| D08 | [Protection against Source Inference Attacks in Federated Learning](https://arxiv.org/abs/2603.02017) | ICLR 2026 | See distributed_sources.md D08 |
| F01 | [Rényi Differential Privacy](https://arxiv.org/abs/1702.07476) | IEEE CSF 2017 | Sections III–V, Propositions 1/3/7, Corollary 3 |
| F02 | [Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising](https://proceedings.mlr.press/v80/balle18a.html) | ICML 2018 | Section 3, Theorem 8, Algorithm 1; Section 4 |
| F03 | [Gaussian Differential Privacy](https://arxiv.org/abs/1905.02383) | Inspected arXiv preprint | Section 2.3 Definition 2.6 Theorem 2.7; Section 3 |
| F04 | [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570) | IEEE S&P 2022; first preprint 2021 | Sections III, IV-A/IV-C Algorithm 1, VI-B |
| F05 | [Exploiting Unintended Feature Leakage in Collaborative Learning](https://arxiv.org/abs/1805.04049) | IEEE S&P 2019 | Sections 2.2, 4.4–4.5, 8.4, 9 |
| F06 | [Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning](https://arxiv.org/abs/1812.00910) | IEEE S&P 2019 | Section II attack methods; Sections III–IV |
| LOCAL-01 | [Impact of the noise calibration on the client inference attack in federated learning](https://doi.org/10.1016/j.knosys.2026.115993) | Knowledge-Based Systems 343:115993 | Initial-Paper.pdf equations and Algorithm 1; Table 13; see project_evidence_audit.md |
| LOCAL-02 | [Gaussian Membership Inference Privacy](https://arxiv.org/abs/2306.07273) | NeurIPS 2023; owner local PDF | Local PDF; see local_gaussian_analysis.md |
| COV-S01 | [Adaptive Differential Privacy in Federated Learning: A Priority-Based Approach](https://arxiv.org/abs/2401.02453) | arXiv v1 preprint | Section III.A-D; Equations 4-6; Section IV |

## Pending closest-first queue

| ID | Candidate | Missing assessment |
|---|---|---|
| CNP1 | [Direction-aware local differential privacy for federated learning](https://www.sciencedirect.com/science/article/pii/S2214212626001924) | Directly relevant direction-aware mechanism; handoff to covariance agent/root. |
| CNP2 | [LDP-Fed+: A robust and privacy-preserving federated learning based classification framework enabled by local differential privacy](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.7429) | Adjacent local framework; avoid inventing mechanism from title. |
| CNP3 | [Local Differential Privacy Based Membership-Privacy-Preserving Federated Learning for Deep-Learning-Driven Remote Sensing](https://www.mdpi.com/2072-4292/15/20/5050) | Membership-focused local mechanism potentially relevant; different LDP-Fed usage must disambiguate. |
| CNP4 | [Aldp-fl: an adaptive local differential privacy-based federated learning mechanism for IoT](https://dblp.org/rec/journals/ijisec/LiLZW25) | Different ALDP-FL paper from Cui/Wu; cannot merge identities. |
| CNP5 | [A federated learning scheme meets dynamic differential privacy](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.12187) | Dynamic scalar privacy/clipping potential baseline; search excerpt insufficient for guarantee assessment. |
| CNP6 | [Efficient adaptive defense scheme for differential privacy in federated learning](https://www.sciencedirect.com/science/article/abs/pii/S2214212625000304) | Noise attenuation and pruning nearby baseline; no full methods inspected. |
| CNP7 | [An Effective Federated Object Detection Framework with Dynamic Differential Privacy](https://www.mdpi.com/2227-7390/12/14/2150) | Task-specific dynamic calibration candidate; full proof and source methods not assessed. |
| CNP8 | [FedDriftGuard adaptive federated learning with differential privacy for concept drift in edge environments](https://www.nature.com/articles/s41598-026-51535-6) | Published before cutoff; drift-based scalar calibration possible adjacent estimator. Full method not assessed; PMC URL access presented recaptcha, publisher date verified instead. |
| CNP9 | [Privacy-enhanced federated learning via asynchronous aggregation and local differential perturbation](https://arxiv.org/abs/2609.15885) | Recent dynamic local-noise context; reported abstract numbers not used as verified evidence. |
| COV-S02 | [FedFR-ADP: Adaptive differential privacy with feedback regulation for robust model performance in federated learning](https://doi.org/10.1016/j.inffus.2024.102796) | Close distribution-driven client Gaussian calibration predecessor; full mathematical methods not acquired |
| CNE3 | [Taming Noise-Induced Prototype Degradation for Privacy-Preserving ... supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_Taming_Noise-Induced_Prototype_CVPR_2026_supplemental.pdf) | Prototype-specific privacy adjacent; title and reference snippet alone insufficient. |
| COV-A01 | [Fisher-driven privacy preservation against category inference attacks in federated learning](https://doi.org/10.1016/j.hcc.2026.100399) | Very close clipping/anisotropic-noise user-level FL comparator, but category inference differs from client participation; publisher full-text fetch returned 403 |
| D09 | [Differentially Private Distributed Learning for Language Modeling Tasks](https://arxiv.org/abs/1712.07473) | Related distributed language-model work; not method-screened beyond search snippet, lower priority than directly user-adjacent baseline. |
| D10 | [A Correlated Noise-assisted Decentralized Differentially Private Estimation Protocol, and its application to fMRI Source Separation](https://pmc.ncbi.nlm.nih.gov/articles/PMC9232162/) | CAPE related/subsuming family; resolve 1910.12913 version and avoid independent-evidence double count. |
| ROOT-A01 | [Bounding Training Data Reconstruction in Private (Deep) Learning](https://proceedings.mlr.press/v162/guo22c.html) | Potential reconstruction/DP bridge; full methods not yet inspected |
| ROOT-A02 | [Source Inference Attacks in Federated Learning](https://arxiv.org/abs/2109.05659) | Relevant source-attribution boundary; full methods not inspected |
| ROOT-A03 | [Enhanced Source Inference Attacks in Federated Learning](https://www.ijcai.org/proceedings/2025/0536.pdf) | Related boundary attack; methods not inspected |
| ROOT-A04 | [DP-KFC: Data-Free Preconditioning for Differentially Private Deep Learning](https://proceedings.mlr.press/v306/van-den-bosch26a.html) | Potential public geometry baseline; methods not inspected |
| ROOT-A05 | [CoSIFL](https://arxiv.org/abs/2509.23190) | Related client/source inference candidate, precise methods and title to resolve |

## Subsequent read-status updates

See [FACP partial methods](facp_followup.md) and [FedFR-ADP access assessment](fedfr_followup.md). Both remain pending complete methods; version-1 screening counts are unchanged.

## Owner-directed non-Gaussian extension

Four additional primary sources were assessed after version 1: [Geng–Viswanath staircase](https://arxiv.org/abs/1212.1186), [Hardt–Talwar geometry](https://arxiv.org/abs/0907.3754), [Joseph–Yu constructions](https://proceedings.mlr.press/v247/joseph24a.html) and [Gilani et al. optimized noise](https://proceedings.mlr.press/v267/gilani25a.html). Selected methods/read scopes and appraisal are in [non-Gaussian foundations](non_gaussian_foundations.md) and [optimized-noise assessment](optimized_noise_followup.md). These are supplemental inclusions, not part of the frozen 31-family version-1 count.

## Separately tracked observer-contract follow-up, 2026-10-06

These entries are outside frozen version-1 systematic inclusion counts; discovery/read boundaries are in [focused handoff](infinitely_divisible_followup.md).

| Follow-up ID | Primary family | Read boundary and remaining gap |
|---|---|---|
| ID-F01 | [Pagh–Stausholm, ALT2022 Arete](https://proceedings.mlr.press/v167/pagh22a.html), arXiv2110.06559v3 | Introduction/definition/certificate and application overview; full proofs not reconstructed; normalized certificate epsilon≥20. |
| ID-F02 | [Harrison–Manurangsi, FORC2025](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.FORC.2025.12), arXiv2504.05202v1 | Abstract/introduction/sampler overview; full finite-parameter mechanisms, continuous transformation and proof assessment pending. |

### Sustained methods/trajectory/construction follow-up, 2026-10-06

The following cards are separately tracked, with overlap to existing foundational families; their card count is not a new unique-family/PRISMA count.

| Follow-up | Primary methods and assessment |
|---|---|
| ID-F02 methods continuation | [GDL/MSDLap detailed reconstruction](divisible_noise_methods.md): official§2–5, scalar certificates/sampler proofs, finite arithmetic, independent transfer review. |
| T-F01–03 | [Temporal noise cards](temporal_noise_followup.md): ICML2023 multi-epoch MF, EMNLP2024 BLT, ICML2025 DMM; full-participation/causal/cryptographic bounds and own reviewed Gaussian limit. |
| S-F01–08 | [Private surrogate cards](private_surrogate_followup.md): eight scoped primary-source assessments; distinguish whole-distribution sampling, sanitized objects and record-private synthetic data. Includes old foundations; no claim of eight new families. |
| F-F01 | [Feldman–Zrnic NeurIPS2021 filter](client_influence_filter.md): arXivv4 Definitions2.5/2.6, Thm3.1/4.3/4.5, Cor4.7, Alg7/Prop5.2; independently reviewed whole-client conditional transfer. |
