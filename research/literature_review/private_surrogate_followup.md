# One-time private client surrogates: literature and feasibility follow-up

Date: 2026-10-06. Target: hide an entire client's dataset contribution versus a registered zero-data slot. This is a focused primary-source follow-up, not an exhaustive systematic review or a completed experiment. No models or experiments were run for this follow-up. Sections explicitly labelled **deduction/proposal** are our reasoning, not reported results from the cited papers.

## Assessment

This is a defensible next construction question because it changes where the information bottleneck sits: privatize a reusable description of a client distribution once, then derive every subsequent update from that description. It removes repeated raw-data query accounting. It does **not** remove the information loss needed to hide a whole dataset, the cost of high-dimensional summaries, or the variance disadvantage of independently privatizing a small number of clients.

The underlying post-processing principle, memoization, private distribution sampling, and training on DP synthetic data already have predecessors. A paper cannot claim these principles as new. The possible contribution is a task-aware, bounded, whole-client surrogate constructor with an explicit empty-slot law, useful approximation guarantees, and a measured advantage over strong one-time server and local controls. Current evidence does not establish that advantage.

## Primary sources and what their techniques actually do

### S1 — Dwork and Roth, *The Algorithmic Foundations of Differential Privacy* (2014)

Read scope: targeted passages in §2, Proposition 2.1 and its proof; §3.3 Laplace mechanism. [Primary monograph](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf), printed pp. 18–19 and 31–34 (PDF pagination differs).

**Source explanation.** A randomized transformation of an already DP output retains the same privacy parameters when the transformation does not query the original data. Proposition 2.1 allows arbitrary randomized post-processing. The Laplace mechanism perturbs a vector query using noise calibrated to its global L1 sensitivity. These results concern the chosen neighboring-input relation; they do not change a record-level guarantee into a whole-client guarantee. Iterative optimization may therefore reuse a private summary without an additional data-access charge, provided its inputs are restricted to that summary and information independent of the protected dataset.

**Transfer deduction.** This provides the proof skeleton below. The substantive research work is constructing a useful summary under whole-client sensitivity, not proving that ordinary post-processing is free.

### S2 — Erlingsson, Pihur and Korolova, RAPPOR (CCS 2014)

Read scope: §§2–3, especially algorithm steps 1–4, §3.1 Theorem 1 and longitudinal-change discussion. [Primary full text](https://arxiv.org/pdf/1407.6981).

**Source explanation.** A client hashes a categorical value into a Bloom filter, randomizes its bits once, and memoizes that permanent randomized representation. Later reports randomize the memoized representation rather than the original value. The permanent channel limits long-term leakage even if repeated observations eventually reveal the memoized value. Its bound is ε∞ = 2h log((1−f/2)/(f/2)), for h hash positions and permanent randomization probability f. The paper cautions that changing, correlated underlying values do not inherit the same fixed-value longitudinal guarantee. Population recovery uses many clients' noisy reports; a permanently randomized report is not a faithful copy of each client's private input.

**Transfer deduction.** Memoize the entire sanitized surrogate, not just a random seed reused to perturb changing raw gradients. Reconstructing a new surrogate after dataset drift is a new raw-data release requiring accounting.

### S3 — Husain, Balle, Cranko and Nock, *Local Differential Privacy for Sampling* (AISTATS 2020)

Read scope: targeted passages in §§2–3, Definition 1, Lemma 2, Eqs. (1), (5), (7), and Algorithm 1; §5 inspected for experiment context, not transferred as an FL result. [Primary full text](https://proceedings.mlr.press/v108/husain20a/husain20a.pdf).

**Source explanation.** The private input is an entire distribution P, possibly the empirical distribution of a dataset. The mechanism outputs a sample, with output likelihood ratios bounded for any pair of input distributions. A mollifier is a family of distributions whose pointwise density ratios are uniformly bounded. Projecting P onto a mollifier, then releasing one sample from the projection, yields a private sampler. A reference-distribution construction constrains each fitted density within exp(±ε/2) of a public Q0. The paper minimizes KL divergence within that family and gives finite-domain and boosting constructions. Privacy is for the released sample, not automatic publication of the learned density or parameters.

**Transfer deduction.** This is directly adjacent distribution-construction literature. Repeated independent draws from the raw-data-dependent fitted law generally compose; merely keeping the same fitted law does not make unlimited draws one-time DP. A separately randomized, DP parameter vector is needed for the reusable-surrogate argument.

### S4 — Yang et al., PrivateFL (USENIX Security 2023)

Read scope: §§2.1–2.2 and 3.1–3.4, Algorithms 1–3; no reproduction of its empirical results. [Primary full text](https://www.usenix.org/system/files/usenixsecurity23-yang-yuchen.pdf), printed pp. 1596–1600.

**Source explanation.** PrivateFL treats a personalized transformation as an initial layer and learns it together with a client's model. It retains the client's previous transformation, combines it with the new global model, and accesses local training examples again each round. The transformation remains local. Its LDP variant applies per-example gradient clipping and noise while training the extended model; distributed and central variants instead use their respective model/aggregate mechanisms. The preliminary DP definition differs in a single sample. It is therefore neither a once-sanitized dataset nor evidence that DP-SGD at a client automatically protects the entire client's contribution. Its useful intuition is that privacy perturbations and clipping can increase effective heterogeneity and that transformations must preserve predictive features.

**Transfer deduction.** PrivateFL is a transformation/utility comparator, not a proof of the proposed one-time channel. A surrogate that subsequently transforms raw examples violates the post-processing restriction.

### S5 — Chen and Vikalo, FedDPMS (CVPR FedVision Workshop 2023; earlier arXiv 2022)

Read scope: §§3.1–3.3, Algorithms 1–2, Eqs. (1)–(4), and the noisy-mean filtering footnote. [Primary workshop paper](https://openaccess.thecvf.com/content/CVPR2023W/FedVision/papers/Chen_Federated_Learning_in_Non-IID_Settings_Aided_by_Differentially_Private_Synthetic_CVPRW_2023_paper.pdf). Browser PDF retrieval failed; primary PDF downloaded and extracted locally for these sections.

**Source explanation.** Clients use VAEs and share noisy class-conditional latent means through a trusted server. The server matches clients with complementary class distributions, allowing synthesis of missing-class examples. Each participating client shares latent information at most once, but subsequently trains on the union of original and synthetic examples. The privacy definition differs by one sample; its sensitivity reasoning addresses bounded latent means and individual latent records. The construction also uses local learned encoders, abundant-class identities and a local classifier to filter noisy means. The paper's empirical check of retained noise variance does not by itself establish a DP guarantee for such filtering.

**Transfer deduction.** One-time means sharing is a close architectural precedent. It does not yield a whole-transcript whole-client guarantee because original records remain training inputs; learned encoders, class identities, multiple draws and filtering require separate analysis before reuse here.

### S6 — Mendieta et al., FedDiff (WACV 2025)

Read scope: §2 DP definition, §3 method, §4 privacy/setup and §5.3 Fourier Magnitude Filtering. [Primary full text](https://openaccess.thecvf.com/content/WACV2025/papers/Mendieta_Navigating_Heterogeneity_and_Privacy_in_One-Shot_Federated_Learning_with_Diffusion_WACV_2025_paper.pdf). Browser retrieval failed; downloaded primary PDF inspected as extracted text.

**Source explanation.** Each client trains a class-conditioned diffusion model; the server collects client generators once, samples synthetic examples, and trains a classifier on them. DP-SGD uses per-sample gradient clipping and noise, with privacy accounting through Opacus. The paper defines neighboring datasets as differing in at most one record. Its Fourier filtering improvement uses additional client magnitude summaries, privatizes those summaries, and includes their expenditure in the overall budget. This illustrates that downstream sampling from a privatized generator can be reused, while a quality filter consulting original data needs protection too.

**Transfer deduction.** This is an existing private-generator-to-synthetic-training design, but its record budget cannot be assigned to whole-client adjacency. Training a large generator is also unnecessary for the first low-dimensional feasibility check.

### S7 — Hoefler, Mueller and Samek, FedKT-CSD (arXiv v1, 8 July 2026)

Read scope: §§3–4, Algorithm 1, Eq. (11), Proposition 4.1 and Appendices D/F/G; §6 inspected only to establish attack unit. Publication status verified from [primary metadata](https://arxiv.org/abs/2607.07565); [primary full text](https://arxiv.org/html/2607.07565v1). Treat as a preprint, not a confirmed peer-reviewed venue result.

**Source explanation.** Frozen public encoders produce clipped latent features. Clients send class-conditional sums, second moments and counts through secure aggregation; the server adds Gaussian noise and generates synthetic data through a public decoder. Downstream training is post-processing of statistics released once. The main adjacency replaces one feature record within a public class, leaving class counts fixed; Appendix F extends to add/remove records with noisy counts. Appendix G explicitly trusts the server to add noise after recovering exact aggregate sums. Neither mechanism protects an entire client dataset locally. Its reported membership evaluation concerns individual images, not registered-client contribution.

**Transfer deduction.** This is a particularly close sufficient-statistic/surrogate predecessor. The proposed novelty cannot be “frozen public encoder + one-time moments + unlimited synthetic training.” Client-level normalized contributions, local randomization and small-federation utility would need distinctive evidence.

### S8 — Acharya, Liu and Sun, *Discrete Distribution Estimation under User-level Local Differential Privacy* (AISTATS 2023)

Read scope: §1.1 Definitions 1.1–1.2, §2 overview/Table 1 and threshold intuition, targeted low-budget estimator/error passages and §4 overview; not all construction details or appendix lower-bound proofs. [Primary full text](https://proceedings.mlr.press/v206/acharya23a/acharya23a.pdf).

**Source explanation.** Each user holds m observations and its message must protect replacement of all m simultaneously. The estimator reconstructs a shared unknown discrete population distribution from n private user messages. More observations per user can improve estimation under stated regimes, with phase transitions in sample count and budget. The primary model assumes observations from the same distribution across users; it discusses only limited heterogeneity and leaves general heterogeneous distributions open. Local protection concerns the user's complete message, not just record perturbations. Population accuracy and privacy are distinct: a large local sample count does not supply n independent users.

**Transfer deduction.** Avoid claiming that whole-client private estimation is universally impossible or that sensitivity must always improve as 1/m. Its guarantees cannot be directly transferred to reconstructing a distinct distribution at each of eight heterogeneous clients.

## Whole-transcript proof skeleton — deduction, not a new theorem

Let each registered client construct Z_i = M_i(D_i) once. Require M_i to be (ε_i,δ_i)-DP for the actual entire-dataset-versus-empty adjacency, including the mechanism's law on the empty input. Thereafter the transcript has the form

\[
\mathcal T = F(Z_1,\ldots,Z_N; U,\text{public inputs}),
\]

where all future data access, synthetic sampling, gradient calculation, stopping and visible diagnostics use only these inputs. Under independent client construction coins, fixed other-client datasets and fixed observer-side inputs, changing D_T changes only Z_T. Its guarantee passes to the whole transcript, even with adaptive server models and peer actions that depend on that transcript. This includes repeated samples from a distribution constructed **from Z_T**.

There is no sum over rounds, nor a sum over other clients' budgets when only target T changes. Joint release of all summaries is protected per changed client; construction of several raw-data summaries at one client still composes. Privacy coins of the target cannot be revealed; peer-owned coins can be part of the observer view. Side information is held fixed in the neighboring-game argument; arbitrary conditioning on correlated auxiliary data is not an automatic stronger DP claim.

Forbidden raw-data dependencies after construction include updating the encoder, picking prototypes, recomputing normalization/counts, evaluating raw-data validation loss for stopping, filtering synthetic examples against raw examples, refreshing the surrogate, and changing traffic according to whether the slot is empty. Reusing identical noise with changing raw gradients also fails: subtracting two replies may cancel the noise. Public roster, fixed weights, fixed message dimensions and an explicit dummy protocol are retained.

## A minimal candidate that can be analysed before a sampler — proposal

Freeze a public feature map and a public dictionary of K labelled feature prototypes or histogram cells. These must cover the task jointly; labels and missing-class indicators are private here. Map every nonempty client dataset to a normalized histogram h(D)≥0 with L1 norm 1; define h(∅)=0. For the **star adjacency** D↔∅, the global L1 sensitivity is exactly 1. Release

\[
Z=h(D)+W,\qquad W_k\stackrel{ind}{\sim}\mathrm{Laplace}(0,1/\varepsilon).
\]

The standard density-ratio calculation bounds both directions by exp(ε). This mechanism also bounds arbitrary nonempty replacement by exp(2ε); do not label it ε replacement-LDP. If replacement protection is required, calibrate scale 2/ε and apply that choice equally to controls. Large datasets improve histogram estimation, **not** this worst-case whole-client sensitivity.

Project Z onto the public subprobability simplex, then assign any remaining mass to a fixed public anchor distribution. The projection, the resulting normalized surrogate distribution Q_i, a fixed number of generated examples, and all later training are post-processing. The empty slot gets the same noise mechanism and fabrication procedure: it has no raw data contribution but need not have a deterministically zero sanitized update. Do not publish its absence through a count or skip flag. The projection creates bias; repeated synthetic sampling reduces Monte Carlo error but cannot remove the original sanitization error.

An alternative is a bounded joint moment vector q(D)=mean ψ(x,y), q(∅)=0, with a public bound ||ψ||≤B. Calibrate the **joint** vector sensitivity: dummy-edge L2 sensitivity ≤B, replacement ≤2B; do not apply record sensitivity B/m. Means, class masses and second moments must be bounded/accounted together. Public PSD projection and regularization can repair noisy covariance/normal equations. Separate classwise means with private denominators and absent classes are a less safe initial parameterization.

For frozen-feature squared-loss regression, first/second moments determine the complete quadratic objective, making this a tractable exact-surrogate benchmark. For softmax classification, a few moments generally do not determine the objective; histogram resolution, prototype approximation and representation error must be measured. The word “sufficient” applies only to the declared model/objective.

## Why one-time construction may still fail — deductions

For the histogram mechanism above, before projection, each coordinate of the averaged N-client histogram has variance 2/(Nε²). More fake examples do not reduce it. At N=8 and ε=8 its coordinate standard deviation is 0.0625, equal to a uniform K=16 cell mass; K=64 cells have only 0.015625 uniform mass. These are arithmetic diagnostics, not accuracy predictions or a recommended choice of ε. A whole-client ε=8 likelihood-ratio bound does not certify attack AUC near 0.5. Small federations and fine dictionaries make the one-time bottleneck severe.

A trusted server can noise the averaged histogram once with sensitivity 1/N and variance 2/(N²ε²) per coordinate for the same dummy-edge game. Independent local noise is therefore N times noisier in variance before processing. This comparison remains relevant to our trusted-server/curious-peer contract. Client placement alone cannot establish a utility advantage; matching the observer's conditional law and the same adjacency is essential. Secure aggregation or hidden summaries may permit a less conservative peer-only law, but that is another proof, not free amplification.

The one-time method can outperform a badly composed multi-round baseline without outperforming a well-tuned one-release server method. Likewise, repeated distributed optimization of fixed surrogates is not automatically necessary: compare training directly on their pooled mixture. Exact quadratic moments admit a centralized closed-form control. Novelty needs benefits beyond avoiding needless rounds or using an informative public representation.

In contrast with fresh zero-mean update noise, the induced gradient error of Q_i is correlated across rounds and generally biased. That persistent correlation is safe only because the entire surrogate was sanitized first; it can also retain an unfavorable learning direction indefinitely. A richer surrogate spends dimensional capacity and increases estimation noise. A coarser surrogate loses client-specific structure. Useful geometry must survive this tradeoff after construction costs, not just under an oracle raw-data histogram.

## Smallest useful next research test — proposal; no launch here

Use the existing Fashion-MNIST public 4×4 average-pooling representation, not a newly privately trained encoder. Declare a compact labelled dictionary without fitting it on evaluation clients; an independently designated public development corpus is another possible basis and must be labelled as additional information. Start with exact squared-loss moments as the implementation sanity control, then a compact classification histogram. Use balanced/quantity-skew and explicit label-stress partitions; the first two are not evidence of arbitrary client-specific distributions.

Compare raw-data utility ceiling, public-anchor-only prediction, one-time server histogram/moments, one-time independently local summaries, direct pooled-surrogate training, and repeated surrogate-only FL. Keep privacy unit, dummy law, representation, weights and budgets identical. Separate representation error, sanitization error, projection bias and finite synthetic-sampling error; exact weighted histogram gradients avoid unnecessary sampling noise during the first check. Retain the previous 0.001 held-out CE improvement gate only as a practical project feasibility criterion, with independent construction repetitions and fresh evaluation; crossing it is not a privacy threshold.

The first stop/go question is whether a compact **whole-client** sanitized surrogate preserves useful predictive structure at small N and defeats the strongest relevant one-time control. If not, larger generators and CNN/CIA sweeps are unlikely to fix the fundamental bottleneck. If yes, investigate task-aware public dictionaries, analytic approximation bounds and client-specific induced update/noise laws. Ordinary independent-trial client-contribution ROC-AUC comes after utility/proof gates, using the locked peer view; the historical frontier remains a motivating paired diagnostic, not this candidate's validation metric.

## Search scope and unresolved candidates

Exact web queries issued on 2026-10-06:

1. `PrivateFL USENIX 2023 personalized data transformations differential privacy`
2. `federated learning locally differentially private synthetic data client distribution one time`
3. `RAPPOR permanent randomized response memoization privacy longitudinal 2014 paper`
4. `user level local differential privacy distribution estimation multiple samples per user paper`
5. `federated synthetic data one shot differential privacy sufficient statistics public encoder histogram client level`
6. `"Federated Learning in Non-IID Settings Aided" Chen 2023`
7. `"federated" "synthetic data" "one-shot" "differential privacy" paper`
8. `"locally private" "synthetic data" "one-time"`

This is a focused snowball/search follow-up with eight assessed source families above, not an enumeration of all returned hits and not a new PRISMA count. Search-discovered candidates requiring further primary-methods assessment include Ma/Jia/Yang ICML2024 *Better Locally Private Sparse Estimation Given Multiple Samples Per User* ([primary landing page](https://proceedings.mlr.press/v235/ma24c.html); PDF request failed, abstract only), Pla/Richard/Vono *Distribution-Aware Mean Estimation under User-level Local Differential Privacy* ([primary preprint](https://arxiv.org/abs/2410.09506); search metadata only), and the 2026 Gaussian-Head OFL family ([primary proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0da2c78fed88ad2d368994970a430c96-Abstract-Conference.html); search metadata only). No guarantee or implementation assessment is assigned to those unread methods. Generic OSFL distillation and record-LDP microdata synthesis hits are not evidence of whole-client surrogate privacy.
