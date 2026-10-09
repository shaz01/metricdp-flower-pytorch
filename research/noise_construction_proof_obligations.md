# Client noise construction: a concrete mathematical starting point

2026-10-01, `feature/client-specific-noise`. This document develops the next research step from the [literature review](literature_review/README.md). It contains our mathematical deductions and candidate definitions, not a new established mechanism, novelty claim, implementation plan or experiment result. The [threat specification](threat_specification_review.md) defines the proposed observer and identifies current-code differences.

**Scope update:** the owner has explicitly opened the design to new non-Gaussian laws. The calculations below are a Gaussian reference, not a family restriction; see [general mechanism research](non_gaussian_mechanism_research.md).

## 1. Start with the actual observable release

Let a fixed candidate roster have public weights `a_i`, and let `v_i` be clipped local updates at a particular history. For independent Gaussian shares,

`Y = sum_i a_i v_i + sum_i a_i xi_i + xi_server`,
`xi_i ~ N(0,Sigma_i)`,
`Sigma_aggregate = sum_i a_i² Sigma_i + Sigma_server`.

A curious peer knows its own share, so the covariance of residual random noise after conditioning on that share is

`B = sum_{i not known to attacker} a_i² Sigma_i + Sigma_server_unknown`.

These equations assume additive noise after clipping, linear aggregation, fixed weights and conditional independence. For correlated shares, use the conditional law given everything the peer knows; subtracting marginal variances is insufficient. Noise before local optimization or clipping changes the release rule.

**Control:** compare a server mechanism whose noise has covariance B and whose sampled noise is unknown to the peer against a client mechanism with the same residual covariance B, conditional history, clipping and update law. Equal marginal total covariance alone is an inadequate peer comparison. Matching these complete conditional laws supplies a null hypothesis: placement alone should not change the observed transcript distribution.

## 2. Fix weights and absence behavior before claiming sensitivity

With one common Euclidean norm bound C, fixed denominator and target contribution removed, mean shift is bounded by `a_target C`. Whole-client replacement gives `2 a_target C`. These are per-history contribution bounds; sampling, private weights and other released metadata require their own analysis.

With normalized sample-count weighting, write `U_IN=a_target v_target+(1-a_target) U_OUT`. For `0<=a_target<1` and a nonempty surviving cohort, if both clipped vectors and U_OUT have norm at most C, removal changes the mean by at most `2 a_target C`. It is not generally `C/n`. This identity assumes the surviving contributions at the compared history are coupled consistently; it does not represent the whole accumulated IN/OUT training trajectory.

For equal-weight averages with denominators n and n-1 the same reasoning gives `2C/n`. A fixed public denominator with zero contribution for an absent slot gives a different mechanism and a smaller direct bound. Choose one explicitly. Do not reuse the old client-count noise scale as a sensitivity theorem.

Removing a target can also remove a noise share. A fixed roster with trusted-server dummy shares is one candidate way to hold the aggregate law constant for a model-only peer. Its target presence remains known to the trusted server, and it therefore does not solve server-observed participation. Existing code does not implement this proposed dummy-share mechanism. If the absent share's covariance was learned from the missing client's private data, dummy generation is itself an unresolved construction problem; a public fixed law avoids that specific obstacle.

## 3. A tractable public-geometry reference mechanism

For a public positive-definite matrix A_i, define ellipsoidal clipping:

`v_i = u_i / max(1, ||A_i^(-1/2) u_i||_2 / C_i)`.

Thus the entire client contribution lies in `v_i^T A_i^-1 v_i <= C_i²`. Sample a local Gaussian with a public covariance, for example

`xi_i = tau_i A_i^(1/2) g_i`, with `g_i ~ N(0,I)`.

The client constructs the law from a public basis/shape, a norm bound and a calibrated scale. This is a reference route separating the benefits of geometry from the privacy cost of estimating it. It is not novel simply because each public A_i differs.

Let B be the fixed positive-definite residual covariance. The matrix B and all public calibration parameters may be known to the peer; its protecting random draws must remain unknown. For fixed-roster add/remove adjacency, a sufficient bound on the target's whitened mean shift is

`sup_delta delta^T B^-1 delta <= a_target² C_target² lambda_max(A_target^(1/2) B^-1 A_target^(1/2))`.

Replace C_target by `2 C_target` for replacement adjacency. The derivation maximizes a quadratic form over the unit Euclidean ball after writing `delta=a_target C_target A_target^(1/2) z`, `||z||<=1`. This gives an explicit sensitivity/geometry constraint, rather than treating gradient correlation as sensitivity.

To use standard fixed-Gaussian calibration, require this bound to be at most `kappa²`, where kappa is the allowable whitened separation for the chosen privacy target. RDP at order alpha is at most `alpha kappa²/2`; conditional round bounds compose if they are uniform over adjacent inputs/history. Alternatively solve the analytic Gaussian condition with whitened sensitivity. The [accounting foundations](literature_review/foundations_and_attacks.md) explain the source theorems. This is only a conditional reference calculation; a full proof must also cover local-training randomness, sampling, weights, absence, all auxiliary releases and the complete observer.

A diagonal or public-basis low-rank shape plus an isotropic floor is practical. Public geometry may reduce clipping distortion in useful directions, but accuracy and privacy gains are untested. An isotropic public shape is the mandatory control.

## 4. Privately learned geometry is the actual research obstacle

A local estimator could use per-layer update statistics, projected minibatch gradients, or a population simulation model. For each choice, specify all of:

- The random variable whose distribution is estimated: records, minibatches, datasets, entire clients or client-participation worlds.
- The source and number of independent samples, versus correlated histories/bootstrap samples.
- Whether the basis and covariance are public, already protected, newly privatized, or retained only locally.
- The complete joint release: noisy update, covariance clues, Fisher/history summaries, hyperparameters, weights and diagnostic statistics.
- A whole-client treatment of estimation and subsequent releases; record-level estimation privacy is not enough by itself.
- Regularization, minimum eigenvalue, rank, drift, update frequency, sampler factors and computational cost.

A mechanism with `Sigma_i(D_i)` changes its law across neighbors even when its mean is unchanged. Not transmitting the matrix does not hide that effect. One clean DP route is a correctly client-level-private estimator followed by an update mechanism uniformly private for every possible estimator output, with composition and specified public/absent-slot behavior. It may be too costly when only one client's dataset supplies the estimate. Another route adapts from an already protected history. A third analyzes the full data-dependent law directly. None is automatic.

For distributional protection, define a population of client datasets and the attacker's conditional background knowledge, then estimate variation under that model. PAC/GMIP ideas motivate this route but do not establish whole-client DP. A record bootstrap inside one client is not a substitute for a client IN/OUT population. Evaluate population misspecification and heterogeneous/outlier clients explicitly.

## 5. Even equal means can disclose participation through variance

As a simple counterexample to mean-only reasoning, suppose `Y_IN~N(0,s_in²)` and `Y_OUT~N(0,s_out²)`. The log likelihood ratio is

`log(p_IN(y)/p_OUT(y)) = log(s_out/s_in) + y² (1/s_out² - 1/s_in²)/2`.

An attacker tests magnitude, despite identical means. Thus a loss-based attack may miss variance leakage caused by dropping a client share or privately adapting a covariance. Our future evaluation needs a covariance/magnitude-aware discriminator alongside model-loss and trajectory attacks. This example is illustrative; it does not prove that the current frontier contains such leakage.

## 6. Decision-ready next deliverables

1. Resolve or explicitly retain FACP/FedFR-ADP method gaps; they are close prior construction comparators.
2. Finalize the candidate roster, weights, observer conditioning and adjacency in the threat specification.
3. Write two fully specified candidate laws: a public/protected-history reference and a privately estimated or population-based law. Derive the complete release law and its proof obligations before implementation.
4. Design estimator-stability and independent CIA pilots with matched unknown aggregate noise, calibrated attack direction and held-out whole-client trials.

The useful paper question is whether a specified construction improves the privacy–utility frontier **after** estimation costs, weighting, absence and stronger attacks are controlled. This document supplies a concrete mathematical baseline against which that claim can be tested.
