# Accounting and attack foundations

Read date: 2026-10-01. Selected primary methods were inspected; proofs were not independently certified and code was not reproduced. These concise source cards explain the foundations needed for our mechanism, rather than pool incomparable experimental numbers. `F01`–`F06` map to the root search ledger.

## F01 — Rényi Differential Privacy (Mironov, IEEE CSF 2017)

[Primary paper](https://arxiv.org/abs/1702.07476). Read: Sections III–V, Propositions 1/3/7 and Corollary 3.

RDP bounds a divergence between the output laws for neighboring inputs. For sensitivity Δ and fixed isotropic Gaussian noise of standard deviation σ, the order-α bound is `ρ=αΔ²/(2σ²)`. Repeated adaptive releases compose by adding their conditional RDP bounds. Conversion gives `ε=ρ+log(1/δ)/(α−1)`; optimize over α. For example, ten rounds with valid conditional bound ρ per round cost at most 10ρ at that order before conversion. Subsampling requires its own theorem; simply multiplying by the sampling rate is not justified.

**Transfer deduction:** use this accounting structure for an adaptive local mechanism. Each step needs a bound valid for every neighboring input conditional on the previous transcript. Choosing a covariance from raw private gradients and plugging its realized σ into the fixed-noise formula does not prove that bound. Estimation releases and noise-parameter changes must be analyzed.

## F02 — Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising (Balle & Wang, ICML 2018)

[Proceedings and full text](https://proceedings.mlr.press/v80/balle18a.html). Read: Section 3/Theorem 8/Algorithm 1; Section 4.

Instead of the classical sufficient Gaussian scale formula, solve the exact condition for a query with L2 sensitivity Δ and fixed isotropic noise:

`Φ(Δ/(2σ)−εσ/Δ) − exp(ε) Φ(−Δ/(2σ)−εσ/Δ) ≤ δ`.

Here Φ is the standard-normal CDF. Numeric root finding finds the least σ satisfying the condition. Postprocessing can denoise the already private output without extra privacy cost. This constructs a calibrated scalar Gaussian distribution; it does not learn private covariance.

**Transfer deduction:** include a properly calibrated isotropic baseline before claiming superiority to “Gaussian DP.” For a fixed positive-definite covariance, whitening transforms mean shifts into Mahalanobis distance; that mathematical reduction requires a fixed or otherwise correctly analyzed covariance. Changing it across adjacent inputs introduces a different hypothesis test.

## F03 — Gaussian Differential Privacy (Dong, Roth & Su, 2019 preprint)

[Primary paper](https://arxiv.org/abs/1905.02383). Read: Section 2.3, Definition 2.6/Theorem 2.7; Section 3.

Privacy is expressed through the trade-off between type-I and type-II errors of every test distinguishing neighboring outputs. The Gaussian trade-off is `Gμ(α)=Φ(Φ⁻¹(1−α)−μ)`. A sensitivity-Δ Gaussian mechanism has μ=Δ/σ. Smaller μ means harder distinguishing. Composition is expressed using trade-off functions, with Gaussian asymptotics under stated conditions.

**Transfer deduction:** this connects the privacy definition to our IN/OUT participation test, provided neighbors differ by an entire client and the observer is specified. A weak attack near chance is insufficient to establish this universal test bound. **Naming:** the repository's `global-dp` is a server-noise implementation label; it is not automatically the Gaussian Differential Privacy formalism, even though both are sometimes abbreviated GDP.

## F04 — Membership Inference Attacks From First Principles (Carlini et al., IEEE S&P 2022)

[Primary full text](https://arxiv.org/html/2112.03570). Read: Section III metrics, Sections IV-A/IV-C and Algorithm 1, Section VI-B variance estimation.

LiRA trains shadow models with a candidate record included and excluded. It transforms true-label confidence and fits conditional Gaussian score distributions, then uses the likelihood ratio of IN versus OUT. The Gaussian is an **attacker's score model**, not defense noise. Comparing low-false-positive behavior reveals failures hidden by averaged attack accuracy. Estimating separate variances for each record is unreliable with few shadows; the paper studies variance sharing.

**Transfer deduction:** build a client analogue using whole-client IN/OUT shadow trajectories, trained under the deployed defense. Fix attack direction and fit parameters on attack-training data, then evaluate independent runs. This extension is proposed here, not an established result of LiRA or an existing repository implementation. Treat an entire transcript as one observed example; its checkpoints are correlated features.

## F05 — Exploiting Unintended Feature Leakage in Collaborative Learning (Melis et al., IEEE S&P 2019)

[Primary full text](https://arxiv.org/html/1805.04049). Read: Sections 2.2, 4.4–4.5, 8.4 and 9.

A participant can remove its known contribution from a shared aggregate and classify the residual using auxiliary data with known properties. Proper aggregation weights matter. Gradient features expose properties unnecessary for the intended task; active participants can amplify leakage using additional objectives. The paper distinguishes example membership and group properties and observes serious utility difficulties in its participant-DP experiment.

**Transfer deduction:** evaluate privacy conditional on the curious participant's update, sampled noise and local data. Count only randomness that remains unknown to it. A record-level guarantee does not automatically protect a group property or the removal of a whole client. The reported utility difficulty is specific evidence, not a theorem that client-level DP can never work.

## F06 — Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning (Nasr, Shokri & Houmansadr, IEEE S&P 2019)

[Primary paper](https://arxiv.org/abs/1812.00910). Read: Section II attack architectures and federated observer cases; Sections III–IV experiment setup/results.

White-box attacks combine gradients, activations, loss and temporal information rather than rely on a single loss threshold. The federated server and a participant have different observations. Active gradient-ascent probes exploit how training responds to a target record. The methods assess **record membership**, not whole-client participation.

**Transfer deduction:** our current clean-shadow loss scorer covers only a restricted attacker. An adaptive defense study should include time-series and gradient-derived features whenever available to the designated observer, and keep active attacks as a separately specified threat extension. Success against an unadapted passive scorer is not evidence of protection against these broader adversaries.

## How to use these foundations

Start from the adjacency and observer; derive sensitivity and the complete conditional release law; calibrate and compose; then test defense-aware inference on independent examples. This order prevents empirical AUC from being mistaken for a formal guarantee, and prevents record-level accounting from being relabeled as client-level privacy. The [primer](privacy_and_noise_primer.md) develops the associated mathematical examples.
