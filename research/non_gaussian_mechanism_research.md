# Research direction: learn the noise law, not only Gaussian covariance

2026-10-01. Owner steering: “why we use gaussian noise only maybe we can generate a unique noise mechanism / distribution”. **Gaussian-only design is not a project requirement.** This document expands the research space; it does not select a final mechanism or claim a new distribution has already been discovered.

## Why Gaussian was a starting point

The original server mechanisms use Gaussian perturbation, and the repository has a Gaussian utility/leakage frontier. Fixed Gaussian laws also have simple covariance aggregation and calibration. This makes them a useful control for discovering whether a gain comes from placement, geometry or the shape of the distribution. Those conveniences do not establish optimality for our whole-client threat, utility objective or composition regime.

The existing review already includes Skellam/discrete noise, metric exponential randomization and learned perturbations. The follow-up now explicitly extends to [non-Gaussian geometric/optimal mechanisms](literature_review/non_gaussian_foundations.md) and [optimized noise distributions](literature_review/optimized_noise_followup.md). Gaussian comparisons remain mandatory; they do not constrain what we may construct.

## The stronger research question

**How can each client construct a noise law from available information so that the complete released training transcript hides whole-client contribution at a smaller task-utility cost than fixed Gaussian and metric-inspired calibration?**

Distribution family, parameter estimator, sampler, clipping, aggregation, absence behavior and multi-round release are parts of the mechanism. A paper contribution could be the construction rule and its client-participation analysis even when the resulting distribution is from a known family. Conversely, inventing a new named density without better protection, a valid guarantee or an effective estimator is not sufficient.

Two meanings of “unique” should be separated:

- A new **mechanism** may learn a familiar distribution using a new, well-analyzed construction rule.
- A new **distribution family** changes the density/shape itself; it needs sampling, aggregation and privacy analysis.

Clients could use a common family with different safely constructed parameters. Giving every client an arbitrary distinct family can make it easier to fingerprint that client's inclusion. Privacy must hold when the attacker knows the mechanism and all public family/parameter information. Hidden realized randomness is distinct from a secret mechanism design.

## Candidate law families to investigate

| Family | Construction idea | What must be checked for our setting |
|---|---|---|
| Isotropic/shaped Gaussian | Fixed or estimated covariance | Baseline; private geometry, conditional sensitivity and observer-known shares |
| Laplace / geometric K-norm | Density follows a sensitivity body rather than a covariance | Correct client-level sensitivity geometry, dimensional cost and sampling |
| Staircase / optimized piecewise density | Optimize the probability allocated to regions at a fixed privacy/cost target | Scalar or restricted-dimensional assumptions; whole-client vector extension |
| Discrete/Skellam | Integer-law noise compatible with secure aggregation | Quantization, weighted aggregation, honest residual noise and modular support |
| Mixture / richer parametric law | Allocate mass to multiple scales or shapes | Privacy of component/weight selection; tails, support and transcript accounting |
| Learned density or sampler | Fit a law against utility and constrained leakage | Optimization approximation, private calibration and attacks beyond the training discriminator |

The [independent assessment](literature_review/optimized_noise_followup.md) also emphasizes that a local quadratic utility approximation depends only on covariance. To detect a shape advantage beyond covariance, measure nonquadratic/higher-order learning effects and the actual privacy law.

These are hypothesis families, not a claim that all are superior or suitable. Heavy tails can harm optimization; bounded support can create revealing events; high-dimensional pure-DP laws can need substantial noise. Test those trade-offs rather than infer performance from the name of a family.

## A general formulation beyond covariance

Let client i compute a clipped update v_i and a construction statistic s_i. Define a probability law `Q_i(theta_i)` with `theta_i=G(s_i, public_information, protected_history)`, then sample `xi_i~Q_i(theta_i)` and release the prescribed perturbed message.

Possible utility objectives include expected task loss, a validated local approximation, clipping distortion and sampler/communication cost. Possible privacy constraints include whole-client DP/RDP for the complete observer, or an explicitly distributional participation guarantee. A weaker empirical CIA objective is a separate claim. The choice of constraint must precede optimization.

**Own mathematical formulation, fixed data-independent density:** for one additive release `Y=f(D)+xi`, with density q and sensitivity shifts h, pure DP can be enforced through

`q(z) <= exp(epsilon) q(z+h)` for every relevant z and h,

in both neighboring directions. Approximate DP allows a bounded excess-mass term, requiring an integral/tail argument rather than a pointwise rule alone. Coordinate factorization needs appropriate vector sensitivity/composition; a scalar density proof cannot simply be repeated over millions of weights without accounting.

When q depends on D, compare **different laws**:

`q_D(y-f(D))` versus `q_Dprime(y-f(Dprime))`.

A fixed-density shift proof no longer answers that question. This applies to Gaussian covariance, mixture weights, learned histograms and neural samplers alike. Keeping the estimator local does not erase its influence on sampled messages.

## Client noise versus the law seen after aggregation

For shares independent conditional on the relevant common history/construction information, and fixed linear weights, the observable noise law is the convolution of the scaled client laws, not the average of their probability densities. Its characteristic function is

`phi_aggregate(t)=product_i phi_i(a_i t)`

(with an additional server factor if present). Against a peer, condition on its known shares before forming the remaining law. Gaussian covariance closure is a special case; for general laws, equal means and variances do not imply equal distinguishability.

Therefore:

- Measure/derive the **aggregate and peer-residual law**, not just local histograms.
- Do not transfer the Gaussian `sqrt(n)` noise-scale rule to arbitrary families as an equality of distributions.
- If designing a preferred aggregate density first, establish whether independent clients can generate shares that realize it. Arbitrary densities need not have the desired convolution factorization.
- Repeated aggregation can make distinct local laws look more Gaussian under appropriate central-limit conditions. Whether local shape survives in the regime of interest is an empirical/statistical question.
- Target removal changes the convolution unless absence/dummy shares and weights are explicitly handled.

A useful novel construction may optimize the observable law jointly with the local sharing rules. That is broader than finding an attractive standalone client noise histogram.

## Practical construction ladder

1. **Public fixed law:** compare isotropic Gaussian, a sensitivity-matched non-Gaussian law and an optimized public template. This separates shape from private estimation cost.
2. **Protected-history adaptation:** adapt template parameters from an already privatized transcript, with conditional validity across histories.
3. **Private client-specific construction:** estimate local information under a valid whole-client treatment or analyze the complete adaptive channel directly.
4. **Distributional/learned route:** model the client population and observer's uncertainty, optimize a sampler and evaluate model misspecification and independent adaptive attacks.

Keep law shape and geometry as separate experimental factors. Start with low-dimensional controlled releases where density/sampling/accounting can be verified, then projected/layer-shaped updates before full deep-model distribution learning. This is a research proposal, not authorization to implement or run those experiments.

## What would count as a credible contribution?

A specified construction must explain why its signal is available to clients, why it targets participation rather than merely optimization variance, and how estimating it affects privacy. It must demonstrate a utility/leakage gain against defense-aware independent attacks and matched mechanism controls, with usable computation and sampler stability. Formal guarantees must cover the stated complete observer; empirical suppression cannot substitute for them.

Nearest precedents already include optimal staircase noise, sensitivity-body mechanisms, RDP-optimized noise and learned distributional privatization. The novelty question is their missing **whole-client construction/aggregation/transcript treatment**, if any, rather than whether non-Gaussian private noise exists. The supplemental source notes establish read scope and remaining verification tasks.
