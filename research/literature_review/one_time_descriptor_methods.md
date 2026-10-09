# One-time descriptor/distillation methods: focused follow-up

2026-10-06. This methods refresh supports the owner-authorized private-descriptor feasibility phase. It is not a new exhaustive search or amendment to the frozen31-family systematic review. Families overlap earlier cards; do not sum them as new studies. Read scopes below are targeted methods, not full-paper empirical replication. The earlier [private-surrogate card](private_surrogate_followup.md) remains the broad source handoff.

## What constructing a reusable client distribution means here

A public input dictionary A={x_1,…,x_K} is fixed independently of protected client datasets. A client privately trains a teacher, encodes its model/logits/probabilities on A, and sanitizes that *complete* bounded object once. From the protected object alone, it constructs conditional label probabilities Q_i(y|x_k). A joint distribution can be defined as Q_i(x_k,y)=Q_i(y|x_k)/K. All subsequent draws and student training use only Q_i and public information.

This is a distribution of labels on PUBLIC inputs, not reconstruction of the client's private image distribution and not a newly invented additive-noise density. The induced learning error is client-specific and reused across training. It is safe because the parameter/descriptor channel is protected, not because reusing arbitrary raw-data-fitted density parameters is automatically free.

Two contracts matter: linear aggregate first then one Q, or separately protected descriptors then individual Q_i. The former can use unknown aggregate shares; the latter needs a valid individual-object channel. A nonlinear map applied separately to each low-noise upload cannot simply inherit a translated-aggregate proof. See [protocol](../proposals/2026-10-06_private_descriptor_protocol.md) and [independent transfer review](../proposals/2026-10-06_private_descriptor_math_review.md).

## Closest distillation and reuse predecessors

### FedMD — Li and Wang, 2019

[Primary text](https://arxiv.org/pdf/1910.03581). Read: §§2.1–2.2 and Algorithm1, printedpp2–3. Public labelled-data pretraining precedes local private-data training. Clients evaluate common public inputs, average class scores, digest consensus and revisit private data. This explains how public-input predictions align different local models. Its inspected method does not supply the proposed whole-client noise certificate or a one-time privatized reusable descriptor. We did not assess its experimental results or other sections.

**Project implication:** public-input score exchange and distillation are established. Our contribution cannot be their introduction. Repeated private-data revisits need additional accounting; all later access must be to the sanitized descriptor for the one-time argument.

### FedDF — Lin et al., NeurIPS2020

[Official primary text](https://proceedings.neurips.cc/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Paper.pdf). Read: §3, Algorithm1, AVG_LOGITS equation and privacy-extension paragraph, printedpp3–4. The server averages model initialization, then distills averaged teacher logits on public, other-domain or generated unlabeled inputs. The inspected privacy paragraph recognizes model-exchange leakage and leaves added DP/decentralization to future exploration. AppendixB's heterogeneous implementation and experiments were not audited.

**Project implication:** strong controls include model averaging and distillation on the same public support. Replacing one with the other changes the learning algorithm; an observed gain is not automatically a noise-density gain.

### Scalable PATE — Papernot et al., ICLR2018

[Primary text](https://arxiv.org/pdf/1802.08908). Read: §§3.1–3.4 and4.1–4.3, Algorithms1–2, printedpp4–8; AppendixA Proposition8 proof, Proposition7 proof and exact Theorem6 statement. Gaussian-noisy teacher votes train a student on public inputs. Confidence thresholding has a separately accounted noisy channel. Original adjacency changes one training example; one teacher vote changes two histogram counts, giving the GNMax per-query boundalpha/sigma². Further queries compose; later student reuse is postprocessing. Full Theorem6 proof and smooth-sensitivity appendix were not audited.

**Our transfer deduction:** assigning one independent teacher to an entire client and fixing its dummy vote permits a separately analysed whole-client vote channel. That is not the paper's stated privacy unit. Our pilot instead jointly bounds a fixed bank of centered votes and perturbs the complete vector once; it does not reproduce PATE's data-dependent accountant or noisy threshold. The raw teachers remain hidden.

### FedMD-NFDP — Sun et al., 2020 preprint

[Primary text](https://arxiv.org/pdf/2009.05537). Read: §§3–4, Algorithm1, Theorems1–2 and initial Appendix without-replacement proof, printedpp2–3 and Appendixp8. Private records are sampled once; subsequent private revisits use the same subset, so subsequent distillation is postprocessing of that initial channel. Section4's guarantee is record-level with delta of orderk/n. Full with-replacement proof and experiments were not audited.

**Project implication:** a once-sampled raw subset is not a whole-client-versus-empty private object. Changing the entire client can replace the whole subset. A new whole-client sanitization channel is necessary; “one-time” alone does not establish it.

### Local Differential Privacy for Sampling — Husain et al., AISTATS2020

[Official primary text](https://proceedings.mlr.press/v108/husain20a/husain20a.pdf). Refreshed: §2 Definitions1–2, Eq1/5/7, Lemma2, §3 Algorithm1 and surrounding density construction. The private input is a distribution. A mollifier restricts pointwise density ratios so ONE sampled output has bounded privacy loss. The fitted deterministic density parameters are not automatically a DP reusable release; unlimited fresh draws from a raw-data-dependent fitted density generally compose. Approximation proofs and empirical claims were not refreshed.

**Project implication:** distinguish a private SAMPLE from a privately released PARAMETER vector. Our Q_i must derive from the latter if unlimited subsequent postprocessing is claimed.

## Noise-law controls for a complete descriptor

### Analytic Gaussian — Balle and Wang, ICML2018

[Official primary text](https://proceedings.mlr.press/v80/balle18a/balle18a.pdf). Read: §3 Lemma7/Theorem8 Eq6 and Algorithm1/calibration discussion, printedpp3–4; no empirical/denoising assessment. For joint L2 sensitivityS, the exact Gaussian condition uses

δ=Φ(S/(2σ)−epsilon*σ/S)−exp(epsilon)Φ(−S/(2σ)−epsilon*σ/S).

This is a stronger one-shot Gaussian control than converting a zCDP bound. Independent80-step bisection atdelta1e-5 gives σ²/S²=1.168910944858024,.3602749391128144,.1184581022863902 for epsilon4,8,16. Privacy concerns the entire clipped descriptor. Splitting coordinates or using record-count sensitivity would change the guarantee.

### Radial L2-Laplace — existing comparator, not a new density

[Koufogiannis–Han–Pappas primary text](https://arxiv.org/pdf/1504.00065), inspected §IV.C, Theorem8, Eq14 and radial sampling paragraph, printedp5/PDFpage4. The paper supplies density proportional toexp(−epsilon||v||_2), total squared-norm momentd(d+1)/epsilon² and Gamma-radius/uniform-direction sampling. Its optimality statement is within the stated Lipschitz-privacy/oblivious identity-query class; it is not universal optimality for our bounded-client learning problem. The dual optimality proof and multiple-user theorem were not fully audited in this refresh.

**Our independent transfer/sampler deduction:** for joint sensitivityS use s=S/epsilon. Triangle inequality supplies a pure-epsilon dummy-edge bound. A Gamma-normal variance mixture gives the same radial density, enabling divisible vector shares. Exact mixture, residual-noise proof and cautions are in the independent review. At dimension12/epsilon8 it has coordinatevariance0.203125S², versus exact analytic Gaussian0.3602749391S². These calculations motivated a frozen non-Gaussian control; variance is not a learned-model utility result.

## Scope and query log

Queries refreshed2026-10-06:

1. `FedMD Heterogenous Federated Learning via Model Distillation 1910.03581`
2. `FedDF ensemble distillation robust model fusion federated learning 2006.07242`
3. `Scalable Private Learning with PATE 1802.08908`
4. `Koufogiannis Han Pappas optimal noise differential privacy l2 Laplace spherical density 2015`
5. `multivariate exponential power distribution gamma normal mixture spherical Laplace (n+1)/2`

NFDP was discovered in the first search results and directly retrieved. Husain came from the existing card; Balle–Wang was retrieved directly by both independent reviewers; only primary methods support the claims above. The final mixture-search results were discovery context, not appraised evidence. Failed/unread candidate papers from the earlier follow-up remain pending. No systematic counts, pooled effects or novelty certificate change.
