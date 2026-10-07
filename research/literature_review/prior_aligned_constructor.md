# Client prior alignment before one-time sanitization

2026-10-06 focused methods follow-up. The owner-authorized descriptor pilot exposed a raw label-stress voting tie, prompting this development-only constructor test and a separate untouched-reserve confirmation. This is not a new systematic review or a claim that log-prior adjustment is novel. Keep frozen31-family inclusion counts unchanged; family overlaps are not summed. [Follow-up protocol](../proposals/2026-10-06_prior_constructor_protocol.md).

## Why this matters for distribution construction

A teacher trained on a client with80% of one class can predict that class on every public prototype. Two clients per dominant class then yield[2,2,2,2] votes everywhere: averaging preserves no input-dependent class information. Adding less noise cannot recover information absent from the query.

A client can use its own empirical label distribution to construct a better conditional prediction BEFORE sanitizing the whole object. Under label shift, class-conditionalsP_i(x|y) are shared while priorsπ_i(y) differ. If the teacher approximatesP_i(y|x), a desired public posterior is proportional toP_i(y|x)π_public(y)/π_i(y). Taking logs gives the score correction−logπ_i+logπ_public. With a uniform public target, the latter is a common offset that disappears from identifiable logits.

This is a hypothesis about predictive distributions, not a noise-density invention. Model calibration, missing classes and conditional feature shift can break its utility. The whole-client privacy proof does not require label shift: any deterministic corrected descriptor remains protected after the complete joint cap and certified channel.

## What each source actually says

### Menon et al., *Long-tail Learning via Logit Adjustment*, ICLR2021

[Official primary PDF](https://openreview.net/pdf?id=37nvvqkCo5), [primary preprint](https://arxiv.org/pdf/2007.07314). Read: §3 Eq7–8; §4.1 Eq9; §5.1–5.3 Eq10–11, Theorem1 statement and double-adjustment discussion, printedpp5–7. Post-hoc class scores subtractτlogπ; τ1 recovers the balanced Bayes decision when scores estimate the original posterior. The training variant instead ADDSτlogπ inside cross-entropy and predicts the learned raw scores. Its §5.3 warns against applying both adjustments indiscriminately. Appendix proofs and experiments were not audited.

**Project transfer:** our prior-aligned arm is the post-hoc score adjustment of a conventionally trained teacher, before joint descriptor clipping/noise. The paper supplies an established imbalance control, not a DP mechanism. Our class-balanced-loss arm is inverse-prior reweighting, not a replication of this adjusted-logit training objective.

### Zhang et al., FedLC, ICML2022

[Official primary text](https://proceedings.mlr.press/v162/zhang22p/zhang22p.pdf). Read: §4 learning objective/calibrated loss, Eq1–7 and Theorem2/3 statements, printedpp4–5. The method is explicitly an FL/label-skew predecessor. It motivates prior-normalized class posteriors, then uses count-dependent pairwise marginsτ(n_y^(-1/4)−n_j^(-1/4)), equivalently count-based calibration of training logits. That formula is different from subtractinglogπ after training. Theorem proofs, experiments and AppendixAlgorithm1 were not assessed.

**Project transfer:** client label-distribution calibration already has close FL prior art. A gain cannot establish novelty for “using each client's distribution”. Do not call our constructor FedLC, or assign its claims/privacy unit to our pilot.

## Privacy and evaluation transfer — our deduction

Estimate π_i,c=(n_i,c+1)/(n_i+4) with fixed public smoothing1. Compute the adjustment internally from private data, then cap the COMPLETE weighted descriptor and sanitize once. No separately released prior estimator needs to be charged: it is part of the protected query map. This does not license publishing priors, changing the noise scale with them, applying raw priors to an already sanitized object, or refreshing them after construction. Config selection and research publication remain unaccounted in these pilot runs.

Include corrected MODEL and LOGIT controls, since prior adjustment is a teacher change. Include class-balanced training at the same20/80 steps, public learning rate and descriptor grids. Its example weights use inverse smoothed priors normalized to mean1; do not subtract its prior again. Public target prior must be declared; our balanced-class benchmark does not validate other deployment priors. Empty slots explicitly use zero centered descriptors even for hard-vote encoders, retaining normal noise and traffic.

The next potential contribution would be a useful whole-client constructor and empirically validated CIA/utility behavior beyond these established calibration, teacher-aggregation and one-time-sanitization controls. No novelty or metric-privacy superiority follows from an improved classifier alone.

## Search/read record

Queries2026-10-06:

1. `Menon Long-tail Learning via Logit Adjustment 2007.07314 ICLR2021`
2. `FedLC Federated Learning Label Distribution Skew Logits Calibration ICML2022 paper`

Only the scoped primary passages above support source claims. This follow-up does not reopen version1 screening, alter PRISMA counts or claim exhaustive novelty coverage.
