# Two-round protected-history construction: finding and limits

2026-10-04. Deterministic analytical optimization with an independent sampler sanity check. This is not neural training, a CIA attack evaluation, or an experiment-completion declaration.

## Main result

Constructing the second noise profile from the first privatized upload can improve the tested **fixed two-round schedule** without a separate randomized-response charge. In the stationary toy below, the improvement survives averaging across independent clients. However, **a public one-release mechanism using the entire budget beats or matches the adaptive two-round mechanism in every calculated case**. The result supports a conditional adaptation hypothesis, not overall superiority, acceptable privacy strength or novelty.

The precise next question is whether protected-history construction helps when multiple learning rounds are needed because updates change with the model. Do not launch an image-model sweep on this toy's positive percentage alone.

## Explicit mechanism and privacy contract

Each client has a private input producing update u_i equal to 0.8e_1 or 0.8e_2 with equal probability. For this diagnostic the update remains the same in both rounds. The released estimate is M_i=(Y_i,0+Y_i,1)/2, and the federation estimate is the mean of M_i over N independent clients. The target is the realized mean of their unnoised updates, not the known expected population mean.

First release: public diamond clipping body of equal semiaxes a_0, 0<a_0<=0.8, and budget eta_0. The clipped center is a_0 times the true update axis. Add independent Laplace coordinates with scale 2a_0/eta_0 to form Y_i,0. This is the two-dimensional diamond K-norm reference.

Choose K_i by the larger absolute coordinate of Y_i,0. The public, fixed bank contains diamonds with swapped semiaxes (L,S) and (S,L), with 0<=S<=L<=0.8. Clip the new raw update inside the selected body and add fresh matching noise at eta_1. Each conditional body has density proportional to exp[-eta_1 ||z||_B/2]. In the interior, covariance is 8 diag(L²,S²)/eta_1², swapped with the profile. Degenerate zero-output boundary controls are specified separately; do not apply a full-dimensional density formula to them.

Analyze the hypothetical complete upload transcript. Y_i,0 is protected at eta_0; K_i is deterministic post-processing of Y_i,0. Conditional on any first upload, the second clipping/noise kernel must protect every private input at eta_1. Thus the reference total is eta_0+eta_1=E. No raw statistic enters K_i, no additional noise seed or private diagnostic is released, and no extra RR charge is needed. The first upload's privacy cost is still paid. This is an existing post-processing/composition construction principle, not a new theorem or novel mechanism claim.

The whole-input uniform bound also permits an empty-input client with zero raw update and the same selector/noise procedures. Empty contribution is not included as an attack trial in the utility calculation below. Physical connection/enrollment secrecy is not claimed. A real implementation must preserve the earlier registered-slot metadata contract.

## Selection-noise correlation: why the two-round calculation differs

With positive a_0 and eta_0, let delta=eta_0/2. The exact wrong-profile probability is

    q=0.5(1+delta) exp(-delta),  p=1-q.

The active-coordinate first noise Z_0 is correlated with choosing the wrong profile:

    E[Z_0,active * 1_wrong] = -a_0 q/2.

Both identities come from integrating the independent Laplace densities. The first identity does not hold for an exactly zero probe, where a fixed tie rule provides no private orientation information.

For one client, the full squared error is

    4 R_local = 16 a_0²/eta_0²
                +p(L+a_0-1.6)²+q(S+a_0-1.6)²
                +8(L²+S²)/eta_1²+a_0 q(L-S).

The final term accounts for selection correlation. Omitting it falsely improves the adaptive method. Reusing the earlier one-round optimal profiles is also incorrect: the two uploads' biases combine before squaring.

For N independently sampled types and independent protecting randomness, define

    beta=(a_0+pL+qS-1.6)/2.
    R_global=R_local/N+(N-1) beta²/(2N).

The second term is aggregate bias. It does not vanish by adding more clients. Geometry is reoptimized for each N; a locally optimal profile need not optimize the aggregate objective. This formula assumes stationary independent client messages, not shared-history gradients in actual FL.

## Controls and bounded optimization

- **Same-probe static control:** exactly the same initial release, budget split and probe radius; optimize a public isotropic diamond for the second release. It receives the same information and step count but fixes the second profile in advance.
- **Optimized static two-round control:** optimize both public radii and the budget split for the same arithmetic-average schedule.
- **One-release public control:** spend all E in one public diamond release and estimate u directly. It changes the schedule and is labeled separately, but is a necessary scientific comparison for repeated stationary updates.
- **Known population mean:** publicly output (0.4,0.4), using zero privacy budget. Its error against the realized mean is 0.32/N. The calculation assumes the population is known for design, so this simple control should not be hidden.

For each positive budget split, the risk is a convex quadratic in the radii. Enumerate box faces and the L=S constraint to find its geometry optimum. Search eta_0 on grids with 512 and 1024 intervals, with public predetermined zero-output second-round boundaries treated explicitly. These grids do not certify continuous global optimality or optimality over arbitrary mechanisms.

Check E in {1,2,4,8,16} and N in {1,8,48}: 15 cases. All cases, parameters, matched controls and boundary labels are saved, including cases without adaptation.

## Selected real calculation outputs

| Total epsilon | Clients | Adaptive risk | Optimized static two-round | Relative gain | Public one-release |
|---|---:|---:|---:|---:|---:|
| 4 | 1 | 0.320000 | 0.320000 | 0% | 0.320000 |
| 4 | 8 | 0.103234 | 0.104837 | 1.53% | 0.065455 |
| 4 | 48 | 0.024271 | 0.024602 | 1.35% | 0.012810 |
| 8 | 1 | 0.189396 | 0.198937 | 4.80% | 0.128000 |
| 8 | 8 | 0.033832 | 0.035862 | 5.66% | 0.018947 |
| 8 | 48 | 0.006260 | 0.006532 | 4.17% | 0.003300 |
| 16 | 1 | 0.059753 | 0.070768 | 15.57% | 0.037647 |
| 16 | 8 | 0.008147 | 0.009727 | 16.25% | 0.004932 |
| 16 | 48 | 0.001397 | 0.001658 | 15.75% | 0.000831 |

Example E=8,N=8: the optimized adaptive probe uses eta_0=4.6640625, a_0=0.8, followed by profile semiaxes about (0.696778,0.325417). The same-probe static control has risk 0.036503; separately optimizing the static budget/radii improves it to 0.035862. Adaptation achieves 0.033832, but the public one-release risk is 0.018947. Large-epsilon improvements must not be described as strong protection merely because they have a pure-DP certificate.

The largest adaptive-risk change from doubling budget-grid resolution was 5.127e-7. An independent 400,000-draw fixed-seed Laplace sanity check measured wrong-profile probability 0.203413 versus formula 0.203003, and cross moment -0.050842 versus -0.050751. Both fall within the prespecified six-standard-error validation tolerance. Sampling validates the formulas numerically; it does not certify privacy.

## Decision and next bounded task

Keep protected-history construction as a research candidate, with a **conditional positive result against a fixed two-round schedule**. Do not label it a better defense than metric privacy, a new distribution, or a general utility improvement. The all-budget one-release control demonstrates that repeated stationary contributions are an unsuitable sole justification for this mechanism.

Next build a small changing-update problem: local quadratic losses F_i(w)=0.5||w-theta_i||², shared model w_t, and raw local update theta_i-w_t. Use fixed registered slots; define the empty target's update as zero while keeping its profile/noise traffic. Let the first release both change w_1 and inform the next profile. Compare adaptive and static mechanisms under identical total budget and public optimization, and retain one-release and public-population controls.

Evaluate the actual global quadratic objective against the realized federation optimum. Shared history couples clients and changes clipping; the stationary analytic independence formula cannot simply be reused. Development simulations may tune public hyperparameters, with fresh held-out simulations for assessment. An independent contribution-inference pilot comes after learnability and utility are established; no CIA conclusion follows from this calculation. Image models remain deferred.

## Reproduce and handoff

Run `uv run python research/calculations/protected_history_two_round.py`.

- [Calculator](../calculations/protected_history_two_round.py)
- [Saved arithmetic and validation artifact](../../results/client_specific_noise/protected_history_two_round.json)
- [Independent mathematical review](2026-10-04_two_round_math_review.md)
- [Prior heterogeneity findings](2026-10-04_heterogeneity_findings.md)

The sampler/privacy building blocks remain existing [K-norm geometry](https://arxiv.org/pdf/0907.3754). The risk integrals, aggregate objective and optimization here are our own derivations and bounded calculations. No systematic-review inclusion count changes are implied.
