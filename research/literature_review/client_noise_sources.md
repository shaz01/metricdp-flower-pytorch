# Client perturbation: verified source cards

Search/read date: 2026-10-01. These seven cards summarize inspected primary-source methods, not a claim that the search is exhaustive. PDF locators refer to physical PDF pages of the linked versions. Source-based explanations are deliberately concise; the linked [primer](privacy_and_noise_primer.md) supplies mathematical tutorials. Author privacy claims are distinguished from our verification of the theorem assumptions. Training and code were not reproduced.

## CN1 — LDP-Fed / CLDP-Fed (Truex et al., 2020)

**Primary source:** [arXiv:2006.03637v1](https://arxiv.org/html/2006.03637v1); EdgeSys 2020, DOI [10.1145/3378679.3394533](https://doi.org/10.1145/3378679.3394533). **Read:** §§2.3–3, mechanism/accounting and client/server workflow; §4 experimental description. **Classification:** individualized privacy preferences; condensed metric-based perturbation; selective parameter communication, not learned Gaussian covariance.

**Technique explained.** Clients locally compute gradients, perturb them using an individual privacy setting, and send updates through anonymous random client selection. The server averages accepted updates. Its demonstrated CLDP variant discretizes/clips numerical values and randomizes them using a distance-weighted exponential mechanism. For a finite value universe,

$$
P(y\mid v)\propto \exp[-\alpha d(v,y)/2].
$$

Nearby numerical values are more likely outputs than distant ones. If a quantized gradient is 4, outputs near 4 receive higher probability than faraway outputs. Selection/filtering and perturbation cycles reduce the amount of high-dimensional information communicated. Each client can choose its privacy setting; this is not a covariance estimated from its data. The paper discusses sequential accounting and sampling. Its local value/metric guarantee must not be equated automatically with whole-client removal privacy for this project's model transcript. Anonymous selection is a substantive assumption; privacy amplification cannot be transferred unchanged to an observer who knows the selected identities. No CIA frontier reproduction was performed here.

**Transfer deduction:** an important non-Gaussian control and precedent for individualized client-side randomization. It also shows that a distance-based client mechanism can remain metric privacy; placement alone does not define an opposing privacy concept.

## CN2 — Time-Adaptive Privacy Spending (Kiani et al., ICLR 2025)

**Primary source:** [arXiv:2502.18706v1](https://arxiv.org/pdf/2502.18706v1), [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d45bb0586164aa7c2289b9e1abee4405-Abstract-Conference.html). **Read:** §3, Algorithms 1–2, §4.1; PDF pp.4–6. **Classification:** client-level individualized privacy schedules/clipping; isotropic Gaussian shares; not estimated covariance geometry.

**Technique explained.** Before training, assign each client a privacy budget, a low early sampling probability, and a transition to later spending rounds. Recompute noise multipliers from remaining budget, compose Rényi privacy spending, and set client clipping norms through a harmonic-mean relation. Early restrained clients participate less; their saved budget permits less restrictive later training. Clients locally clip final updates and add Gaussian shares. The server supplies noise for nonsampled clients and uses a fixed normalization. The threat discussion targets honest-but-curious peers with a trusted server, acknowledging weaker protection against that server.

Algorithm 1 sets
$$
c_n^t=c\sigma^t/\sigma_n^t,
\quad \operatorname{Cov}(\eta_n^t)=(c_n^t\sigma_n^t)^2I/N
=(c\sigma^t)^2I/N.
$$

Thus client-specific multipliers cancel against clipping: absolute noise covariance is equal across clients within a round. Parameters depend on privacy preferences/schedules, not private covariance estimation. This is already close to the proposed client-side direction, but not its distribution-construction question.

**Transfer deduction:** use as a strong schedule/preference baseline. The server's noise for absent clients deserves explicit study: removal of a contribution should not inadvertently disclose itself through a changed aggregate noise variance.

## CN3 — NbAFL (Wei et al., 2019 preprint / 2020 journal work)

**Primary source:** [arXiv:1911.00222v2](https://arxiv.org/pdf/1911.00222v2). **Read:** §II.B and §III.A, Algorithm 1, Eqs.(4)–(10), PDF pp.3–5. **Classification:** Gaussian noise before upload plus optional server noise; record replacement adjacency; scalar calibration.

**Technique explained.** Train a local model, clip its parameters, add Gaussian noise before upload, aggregate weighted models, then optionally add extra server noise before broadcasting. The threat model uses an honest server and external observers of uplink/downlink channels, allowing fewer uplink exposures than broadcasts. Neighboring client datasets have equal size and differ in one sample. Uplink scale depends on a norm bound, minimum dataset size, privacy budget, and exposure count. Additional downlink noise depends on the number of broadcasts relative to uplink exposures and clients. The method does not learn an individual covariance from client gradients. Its main relevance is the distinction between noise protecting an upload and total noise protecting a broadcast. Its participation/scheduling assumptions differ from the current removal protocol.

**Published caution:** [Talaei & Izadi, arXiv:2406.05858v1](https://arxiv.org/pdf/2406.05858v1), pp.1–2, dispute the original Theorem 2 convergence bound, identifying a Polyak–Łojasiewicz inequality used in the wrong direction and deriving a replacement. This comment was inspected; it addresses convergence, not every privacy claim.

**Mathematical concern, separate from that comment:** Eq.(4) writes an optimizer of an averaged loss as an average of single-example optimizers. This identity does not hold generally. Therefore its ensuing (2C/|D_i|$ sensitivity must not be imported as a bound for arbitrary deep local training without additional assumptions. A whole-vector norm bound alone yields a replacement difference up to (2C$, not automatically (2C/|D_i|$.

## CN4 — ALDP-FL (Cui & Wu, Scientific Reports 2025)

**Primary source:** [publisher full text](https://www.nature.com/articles/s41598-025-12575-6), DOI 10.1038/s41598-025-12575-6; [PDF mirror](https://d-nb.info/1377063569/34). **Read:** client-side algorithm, bounded perturbation, privacy-security analysis and experiment sections; PDF pp.6–10, Eqs.(3)–(21). **Classification:** local/layer-specific historical norm calibration; truncated Gaussian, not full covariance estimation.

**Technique explained.** Each client trains locally, clips layer updates using a moving window of previous update norms, and perturbs each layer before returning its final update. Noise begins with a Gaussian scale and is truncated to a bounded interval, with density renormalized inside that interval. The intent is to reduce unusually large noise excursions while adapting clipping to changing layer magnitudes. Think of one client's early large updates and later small updates: its history adjusts the bound instead of retaining one fixed threshold. The paper reports image-classification utility and iDLG reconstruction experiments. Those attacks assess reconstruction, not whole-client participation inference. Its privacy-security section analyzes Gaussian densities, while the implemented bounded perturbation has truncated support. The article's asserted privacy guarantee has not been certified by this review.

**Mathematical concern:** truncating additive noise shifts bounded supports with the input. Output regions possible in one neighboring world and impossible in the other contribute probability mass that must fit within the approximate-DP
$\delta$ allowance. Support mismatch does **not** automatically disprove approximate DP, but the ordinary untruncated Gaussian scale formula does not automatically transfer. Historical private calibration and repeated local steps also require accounting. Include as nearby construction precedent and a cautionary proof case, not an established CIA defense.

## CN5 — Private Adaptive Clipping (Andrew et al., NeurIPS 2021)

**Primary source:** [NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2021/file/91cff01af640a24e7f9f7a5ab407889f-Paper.pdf). **Read:** §2, §2.1, Algorithm 1 and accounting discussion, PDF pp.3–6. **Classification:** private online global clipping estimation; server Gaussian aggregate perturbation; user-level target.

**Technique explained.** Every sampled client supplies a clipped update and a bit indicating whether its raw update norm is below the current threshold. The server noises the bit count, obtaining a private estimate of the fraction below threshold. It adjusts the next bound geometrically:

$$
C^{t+1}=C^t\exp[-\eta_C(\tilde b^t-\gamma)].
$$

If fewer than the target fraction
$\gamma$ lie below the threshold, the threshold increases. If too many do, it decreases. Setting
$\gamma=0.5$ tracks the median norm. Gaussian perturbation also protects the aggregate update, and the algorithm allocates noise/accounting to both queries. The construction responds to the empirical distribution of client norms without treating the observed raw quantile as free public information. This is not a separate covariance for each client; clients share the current clipping threshold. Its guarantee concerns the released server mechanism, rather than independently private unprotected uploads.

**Transfer deduction:** this is a foundational answer to “how can private statistics inform calibration?” Adaptation should carry its own privacy treatment. A local covariance extension needs a proof that covers the additional estimator and transcript, rather than simply copying the quantile rule.

## CN6 — DP-SGD (Abadi et al., CCS 2016)

**Primary source:** [arXiv:1607.00133](https://arxiv.org/pdf/1607.00133). **Read:** §3, Algorithm 1 and moments accountant, PDF pp.3–4. **Classification:** example-level gradient perturbation; Gaussian scalar scale; composition baseline, not federated client geometry.

**Technique explained.** Sample a training lot; compute a gradient for each example; clip each gradient separately; add isotropic Gaussian noise to the sum; normalize; update the model. Repeat while an accountant tracks the composition. In compact notation,

$$
\tilde g_t=\frac1L\left[\sum_{x\in B_t}
g_t(x)\min(1,C/\|g_t(x)\|_2)
+\mathcal N(0,\sigma^2C^2I)\right].
$$

Clipping bounds one example's influence before averaging, which is why it differs from clipping one final client update. The moments accountant tracks privacy-loss moments and incorporates sampling, improving repeated-step accounting over a naive per-step budget sum. Noise on every training step changes optimization; it is not equivalent to adding the same nominal noise only once to an upload. Running this algorithm locally on a hospital dataset can protect record changes under the accountant's assumptions; its original privacy unit does not automatically become the whole hospital. The paper is retained as a construction/accounting foundation, rather than evidence of superior CIA protection.

**Transfer deduction:** compare local DP-SGD with final-update client noise as distinct mechanisms. Per-example gradients may offer inputs for covariance estimation, but their existence does not privatize a learned covariance for free.

## CN7 — Privacy-Aligned Personalized FL (Xu et al., September 2026 preprint)

**Primary source:** [arXiv:2609.15950v1](https://arxiv.org/pdf/2609.15950v1). **Status:** recent preprint; peer-review acceptance not established. **Read:** §2.1–2.3, Eqs.(1)–(9), Algorithm 1, PDF pp.1–3. **Classification:** central record-level replace-one DP; one-shot private context and fixed low-dimensional coefficient-space releases.

**Technique explained.** Form bounded first/second-order input summaries and release a client's noisy mean context once. Cache that context. A generator uses it to produce coefficients in a public, fixed orthonormal model subspace. Repeated training releases clipped per-record gradients in coefficient space with isotropic Gaussian noise. A factorized generator transforms these noisy gradients into client-dependent optimization geometry. The paper also describes variable-length communication implementing the Gaussian channel through quantization. Privacy accounting includes the initial context release and repeated gradients; a future client needs a one-shot context release for generated personalization.

This is not client participation privacy: the stated setting is trusted-server central record-level DP. Nor does it estimate a client's noise covariance from private gradients. Its effective geometry emerges from the factorized generator applied to coefficient-space noise. The distinction between perturbation geometry and optimization geometry matters when assessing similarity to our proposal.

**Transfer deduction:** study compact representations and one-shot privatized client descriptions as design alternatives, but retain whole-client adjacency and model-only peer observations for CIA evaluation. Dimension reduction can reduce perturbation energy while changing the learning problem; comparison must account for architecture/personalization changes.

## Synthesis for the next agent

The closest verified precedent in threat target is CN2: it directly discusses client-level protection from curious peers and locally generated noise. Its actual covariance cancellation is easy to miss. CN1 already establishes locally individualized randomization using metric-based perturbation. CN3 establishes noising before aggregation as an older idea. Therefore novelty cannot rest on client placement, individual privacy preferences, or different scalar budgets alone.

CN4 is close in private-history-driven calibration but raises proof questions. CN5 supplies a cleaner template for accounting for adaptive estimation. CN6 clarifies record-vs-client protection and perturbation during local optimization. CN7 supplies recent compact-geometry context while explicitly using record-level privacy. These are technique categories to compare, not evidence that any one defeats the repository's CIA protocol.

Next extraction should separate: **distribution estimated from local private information**, **privacy preference selected by a user**, **common covariance with individualized sensitivity**, and **geometry induced by optimization after a fixed-noise release**. These can look similar in prose and yield different proofs and experiments.
