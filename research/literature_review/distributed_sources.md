# Distributed noise and trust-boundary source cards

Search/read date: 2026-10-01. Eight selected primary sources. These are method-level reading notes, not a claim of exhaustive coverage or proof verification. Full PDFs were accessed; selected sections listed below were read. Appendices/proofs were not exhaustively verified. Exact queries and screened candidates are in [distributed_search_log.json](distributed_search_log.json). For evaluation implications, see [evaluation_requirements.md](evaluation_requirements.md).

## D01 — McMahan et al., Learning Differentially Private Recurrent Language Models

**Metadata:** ICLR 2018; [primary PDF](https://arxiv.org/pdf/1710.06963), [conference PDF](https://openreview.net/pdf?id=BJ0hF1Z0b). Conference PDF download returned 403; arXiv PDF contains the ICLR 2018 version. **Read:** Sections 1–2, Definition 2, Algorithm 1, Lemma 1, Theorem 1, Section 3/Figures 2–4. Proof appendices not audited.

**Source explanation (approximately 180 words).** The privacy unit is a complete user history: neighbors add or remove all examples belonging to one user. This directly addresses whether that user's data helped train the released model. Each selected user performs local optimization, and its complete update is norm-clipped before averaging. The algorithm independently samples users with probability q, uses an averaging estimator with bounded sensitivity, adds Gaussian noise to the average, and composes privacy spending across rounds with a moments accountant. Flat clipping constrains the entire update; per-layer clipping allocates the norm budget across layers. For the fixed-denominator estimator, Algorithm 1 gives

\[
\theta_{t+1}=\theta_t+\frac{\sum_{k\in C_t}w_k\Delta_k}{qW}+Z_t,\quad Z_t\sim\mathcal N(0,\sigma^2 I),\quad \sigma=\frac{zS}{qW}.
\]

Weights satisfy `0<=w_k<=1`, and W is a fixed/public lower-bound normalization parameter. Clipping enforces `||Delta_k|| <= S`; the bounded denominator prevents privacy sensitivity from exploding when the random sampled cohort is small. The protected release is the noised learned model, not each clear client upload. The language-model experiments benefit from very many users and large cohorts; their utility conclusions do not establish performance in small cross-silo federations. This is a fundamental client-participation privacy baseline, despite server-side noise placement.

## D02 — Geyer, Klein, Nabi, Differentially Private Federated Learning: A Client Level Perspective

**Metadata:** NIPS 2017 workshop; inspected [arXiv v2, 1 March 2018](https://arxiv.org/abs/1712.07557v2), [PDF](https://arxiv.org/pdf/1712.07557v2). **Read:** Sections 2–3, averaging formula, median-clipping caveat, between-client variance definition, Sections 4–5. No formal proof certification.

**Source explanation (approximately 180 words).** This work explicitly seeks to hide whether an entire client participated. Its “client-sided differential privacy” terminology describes the protected unit; the mechanism in Section 3 actually perturbs a sum at the central curator. The curator samples clients, receives their local updates, clips each update's norm at S, adds Gaussian noise to the sum, and divides by the cohort size. A moments accountant tracks successive releases. Written with the summation parentheses made explicit, the Section 3 mechanism is

\[
w_{t+1}=w_t+\frac1{m_t}\left[\sum_{k\in Z_t}\frac{\Delta w_k}{\max(1,\|\Delta w_k\|_2/S)}+\mathcal N(0,\sigma^2S^2I)\right].
\]

The authors propose dynamically adjusting privacy/optimization settings as client updates evolve, using between-client update variance as an indication of heterogeneity. Crucially, they compute the clipping bound from the unrandomized median of private update norms and explicitly acknowledge that this “strictly speaking” violates privacy. Consequently, its empirical adaptive variant is not a ready-to-use end-to-end DP certificate. The study motivates client-level protection and dynamic calibration but does not establish that individually learned noise distributions protect uploads or whole-client removal.

## D03 — Kairouz, Liu, Steinke, The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation

**Metadata:** ICML 2021, PMLR 139:5201–5212; [primary proceedings page](https://proceedings.mlr.press/v139/kairouz21a.html), [PDF](https://proceedings.mlr.press/v139/kairouz21a/kairouz21a.pdf). **Read:** Sections 2–5, Algorithms 1–2, Theorems 5/8/9, distributed-DP trust discussion, Section 6 experimental setup/results. Supplementary proof details not exhaustively audited.

**Source explanation (approximately 180 words).** Clients clip their update vectors, flatten coordinates through a public random rotation, conditionally round to a discrete grid, add independent discrete Gaussian noise, and transmit integers through modular secure aggregation. The server sees only the modular noisy sum, then decodes the sum and reverses the rotation. The protected privacy unit is addition/removal of one user's records. Per-client perturbations need not satisfy the desired local-DP guarantee independently; privacy is obtained from their combined noise under the assumed secure-sum visibility. Algorithm 1 draws each integer noise coordinate from

\[
P(Y=k)\propto\exp[-k^2/(2(\sigma/\gamma)^2)],\qquad z_i=(\widetilde x_i+Y_i)\bmod m.
\]

Here gamma is the quantization granularity. Unlike continuous Gaussians, sums of discrete Gaussians are not exactly discrete Gaussian; the paper bounds the convolution discrepancy and incorporates it into privacy analysis. Quantization, clipping, and modular wraparound also require analysis. Experiments demonstrate settings close to central-Gaussian utility with restricted communication precision. The secure aggregation implementation and detailed trust assumptions are treated as an external black box; guarantees do not automatically cover a server inspecting individual messages or arbitrarily many colluding clients.

## D04 — Agarwal, Kairouz, Liu, The Skellam Mechanism for Differentially Private Federated Learning

**Metadata:** NeurIPS 2021, 34:5052–5064; [proceedings](https://proceedings.nips.cc/paper_files/paper/2021/hash/285baacbdf8fda1de94b19282acd23e2-Abstract.html), [primary PDF](https://papers.neurips.cc/paper/2021/file/285baacbdf8fda1de94b19282acd23e2-Paper.pdf). **Read:** Sections 2–4, Definitions 2.1/3.1/3.2, Theorem 3.5, Corollary 3.6, distributed application, Section 5 experimental setup/results. Bessel-function proof details not audited.

**Source explanation (approximately 175 words).** A symmetric Skellam variable is a difference of two independent Poisson variables. With total variance parameter mu,

\[
Y=P_1-P_2,\quad P_1,P_2\overset{ind}{\sim}\operatorname{Poisson}(\mu/2),\qquad E[Y]=0,\quad\operatorname{Var}(Y)=\mu.
\]

Clients add integer-valued Skellam noise to discretized updates, then use secure aggregation. Independent Skellam contributions add exactly: the aggregate remains Skellam with variance equal to the sum of local variances. This simplifies distributed sampling and privacy accounting compared with discrete Gaussian convolution. The paper supplies multidimensional Rényi-DP bounds in terms of L1/L2 sensitivity and applies subsampling/composition for learning. Its user-level adjacency adds or removes all records belonging to one user. The discrete pipeline still needs clipping, norm-controlled rounding, a finite modulus, and communication-accuracy trade-offs. The experiments compare private learning utility with continuous and discrete Gaussian alternatives; this is not a direct CIA attack evaluation. The benefit is a convenient distributed noise law and a formal aggregate mechanism, rather than a rule for learning an individualized distribution from each client's private geometry. Inspecting local uploads requires a different privacy calibration from observing only their secure sum.

## D05 — Bonawitz et al., Practical Secure Aggregation for Privacy-Preserving Machine Learning

**Metadata:** ACM CCS 2017:1175–1191, DOI [10.1145/3133956.3133982](https://doi.org/10.1145/3133956.3133982); [conference PDF](https://acmccs.github.io/papers/p1175-bonawitzA.pdf). The 2016 workshop precursor is a separate bibliographic snapshot. **Read:** Sections 1–2, protocol overview/Sections 4–5, Figure 4, security statements in Section 6, deployment discussion Section 8. Cryptographic proof hybrids not exhaustively audited.

**Source explanation (approximately 170 words).** Secure aggregation is a cryptographic visibility primitive: the server should learn the sum of clients' vectors while learning no additional information about their individual inputs. Clients establish pairwise secret masks and add them to their vectors; each pair's masks cancel when the vectors are summed. A simplified algebraic view is

\[
y_i=x_i+\sum_{j\ne i}p_{ij}\pmod R,\qquad p_{ij}=-p_{ji},\qquad\sum_i y_i=\sum_i x_i\pmod R.
\]

The practical protocol adds private masks, secret sharing, unmasking, and consistency checks to remain secure when clients drop out and to address server/client collusion within specified thresholds. Honest-but-curious and active-adversary variants have different assumptions and security arguments. Its proof is simulation-based: what an adversary learns is limited by the prescribed aggregate and its own available inputs. It does not perturb that aggregate to make neighboring datasets indistinguishable. The server therefore can still learn information inherent in the sum or successive released models. Secure aggregation enables distributed-DP noise to protect an aggregate without exposing under-noised individual contributions; it does not itself supply DP or hide network participation identities.

## D06 — Imtiaz, Mohammadi, Sarwate, Distributed Differentially Private Computation of Functions with Correlated Noise (CAPE)

**Metadata:** initial 2019 preprint; inspected [arXiv v3, 23 February 2021](https://arxiv.org/abs/1904.10059v3), [PDF](https://arxiv.org/pdf/1904.10059v3). Metadata says partially subsumed by arXiv:1910.12913; do not count these as independent evidence without family reconciliation. **Read:** Section 2, Sections 3.1–3.4, Algorithms 1–2, Theorem 1/Lemma 1, functional-mechanism application overview/experimental overview. Proof calculations not independently checked.

**Source explanation (approximately 180 words).** CAPE reduces excess noise when sites jointly estimate a statistic. Each site draws Gaussian randomness, secure-aggregates these draws, then subtracts their shared mean to create zero-sum correlated noise. Independent residual noise is added separately. In the equal-site symmetric setting,

\[
e_s=\widehat e_s-S^{-1}\sum_j\widehat e_j,\quad g_s\sim\mathcal N(0,\tau^2/S),\quad\widehat a_s=f(D_s)+e_s+g_s.
\]

The e terms cancel in the final average, while g terms remain; local releases have substantial perturbation and the pooled estimate has smaller variance. The privacy analysis uses the adversary's joint observations, not just marginal local variances. Its model assumes honest-but-curious sites and aggregator, with a bounded number of colluding sites; the displayed symmetric theorem uses at most `ceil(S/3)-1` colluding sites. The formal adjacency in Section 2 changes **one database entry**, so the guarantee protects records at a site rather than removing the entire site. Gaussian scale depends on a query's sensitivity/sample size, not a learned client update distribution. Functional-mechanism examples perturb objective coefficients for regression/neural-network optimization. The paper explains conditions for matching pooled-data noise, not a generic whole-client CIA defense under arbitrary collusion or dropout.

## D07 — Rodio, Chen, Larsson, Optimizing Privacy-Utility Trade-off in Decentralized Learning with Generalized Correlated Noise (CorN-DSGD)

**Metadata:** arXiv:2501.14644v2, 23 July 2025; metadata states accepted IEEE ITW 2025. [Primary v2](https://arxiv.org/abs/2501.14644v2), [PDF](https://arxiv.org/pdf/2501.14644v2). The January v1 title/algorithm is **Whisper D-SGD**; this is the same source family. **Read:** Sections II–V, Theorems 1–2, Algorithm 1, Equations 8–10, Section V-D HBC extension, Section VI setup/results. Proof sketches read, not independently proved.

**Source explanation (approximately 180 words).** This method designs covariance **across agents**, rather than across parameters within one client's update. Agents perform clipped stochastic-gradient steps and neighbor mixing through matrix W. They jointly sample Gaussian noise with covariance R. Utility noise is governed by `Tr(W R W^T)`, while the privacy bound depends on diagonal entries of the inverse covariance. The method solves a semidefinite program:

\[
\min_{R\succ0}\operatorname{Tr}(WRW^T),\qquad (R^{-1})_{ii}\le\frac{\varepsilon^2}{16C^2T\log(1/\delta)}.
\]

An independent component prevents total noise cancellation; a correlated component exploits mixing topology. Its adjacency varies one agent's dataset, with clipped gradients bounding per-step change. The base algorithm shares a seed among all agents, appropriate only when its randomness remains hidden from the adversary. Section V-D adds honest-but-curious coalition constraints using seed-sharing subsets and covariance unknown to each coalition. Experiments study utility under privacy budgets and graph connectivity. The mechanism uses public topology/privacy parameters to construct covariance, not private client-data covariance estimation. Adapting it to a participating-client CIA therefore requires specifying what seeds and messages that client knows; correlated marginal variance alone is insufficient.

**Independent appraisal of the displayed SDP:** do not treat its printed constraint as a universal certificate. With `rho=2 C² T max_i (R^-1)_ii`, the usual optimized RDP conversion gives `epsilon_actual <= rho+2 sqrt(rho log(1/delta))`. Substitution requires an epsilon range or a tighter calibration check. See [cross-review](review_threat_checks.md). This is our mathematical qualification, not a reproduced privacy guarantee.

## D08 — Athanasiou, Jung, Palamidessi, Protection against Source Inference Attacks in Federated Learning

**Metadata:** ICLR 2026; inspected [arXiv:2603.02017v1, 2 March 2026](https://arxiv.org/abs/2603.02017), [primary PDF](https://arxiv.org/pdf/2603.02017v1). Parent-agent discovery query: `Source Inference Attacks Federated Learning arxiv`. **Read:** Sections 3–6, Algorithm 1/Theorem 1, Section 7 overview, Appendix A proposition/proof statement scope. Detailed attack experiments/proofs not independently reproduced.

**Source explanation (approximately 180 words).** Source inference asks which client owns a known training record; it differs from asking whether a candidate client participated. This paper's attacker is an honest-but-curious server that observes shuffled messages and has target-distribution shadow data for reversing ordinary shuffling. It shows why merely permuting whole models, layers, or parameters can allow remapping. Its defense discretizes parameters, encodes them with the residue number system and unary bit encodings, and shuffles at that granular level. The residue representation is

\[
x\mapsto(x\bmod m_1,\ldots,x\bmod m_u),\quad\gcd(m_i,m_j)=1;
\]

the Chinese remainder theorem reconstructs the summed parameter. The authors intend aggregate-only disclosure and preservation of aggregation accuracy up to discretization; that visibility claim was not independently proved here. Theorem 1 asserts random-guess SIA accuracy under the paper's attack/visibility formulation. The shuffler must not collude with the attacker; a MixNet implementation distributes trust while retaining an honest shuffle entity. This is an important boundary source because hiding individual uploads can defeat their attribution while leaving the global model observable. Its noise-free aggregate disclosure is not a differential-privacy proof for whole-client removal, and its guarantee should not be transferred to CIA simply because both are client-related attacks.

**Independent proof concern:** observed unary one-counts disclose per-modulus residue sums, which need not be determined by the original total: `(0,6)` and `(3,3)` both sum to six, but their modulo-five residue sums are one and six. This challenges a generic original-sum-only interpretation. The appendix's chance-SIA step also relies on its restricted no-distinct-model comparison formulation. Its printed modulus-range inequality needs checking against unique CRT reconstruction. See [review details](review_threat_checks.md); these are unresolved review deductions, not demonstrated practical attacks.

## Transfer deductions for this project — not claims made by these papers

| Design question | Deduction / practical consequence |
|---|---|
| Does adding noise on devices inherently outperform server noise? | No such dominance follows. When independent client noises induce the same aggregate law, the model observer can see the same noise distribution. Placement primarily changes who can inspect clear updates and whose honesty is required. Different clipping, sampling, conditioning, optimizer paths, or covariance can then change the resulting utility/leakage. |
| Can an individualized distribution be calibrated only from its realized variance? | That is insufficient. The law of estimation and release across neighboring client datasets must be analyzed, including covariance changes, exposed distribution metadata, and repeated rounds. |
| What does secure aggregation buy? | It can move the relevant privacy calibration from individual upload to combined honest noise. This requires a lower bound on honest residual contributions and visibility assumptions; it does not erase the observer's own noise or unknown-dropout issues. |
| Could covariance itself reveal the missing client? | Yes as a mechanism-design risk: deleting a client can change the aggregate covariance, weights, covariance estimation, or temporal structure. A distribution-based attack may exploit these changes even if loss-based CIA is weakened. Whether this happens empirically is untested here. |
| Are CAPE/CorN solutions for the proposed idea? | They supply coordinated randomness precedents and trust/proof questions. CAPE protects records; CorN primarily constructs inter-agent covariance from topology. Neither establishes novelty or validity of privately learned per-client parameter-space noise distributions. |
| Does protecting source attribution protect participation? | No automatic implication. The model observer can still distinguish aggregate training distributions even when individual uploads cannot be attributed. |

Future reading must connect these placement/trust baselines to private local geometry estimation and to defense-aware, independent-trial CIA evaluation. No mechanism or novelty conclusion is selected in this document.
