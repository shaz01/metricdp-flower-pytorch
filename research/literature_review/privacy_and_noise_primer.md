# Understanding client-specific noise for client inference protection

Date: 2026-10-01. Audience: project owner and agents continuing the literature review.

This primer supplies a common vocabulary, mathematical controls, and extraction questions. It is **not a completed literature review, a novelty claim, or a certified mechanism**. No external literature search was used to construct this file. Statements marked **repository evidence** follow inspected code or the existing [project evidence audit](../project_evidence_audit.md). Statements marked **mathematical deduction** follow the assumptions written beside their equations. Suggested designs are **research hypotheses**.

## 1. What exactly are we hiding?

**Repository evidence.** The project asks whether a target client participated in training. Its IN world includes that client; its OUT world removes it without reallocating the other clients' datasets. An attacker scores global checkpoints using loss on a shadow dataset representing the target. The original paper assumes a trusted server and semi-honest participating clients; see the evidence audit's original-paper discussion and [CIA implementation overview](../../experiments/cia/README.md).

Two privacy questions must remain distinct:

| Privacy unit | Neighboring worlds | Typical question |
|---|---|---|
| Record | One person's/example's record is added, removed, or replaced inside a client dataset | Was this example used? |
| Whole client | The complete contribution of one client is added, removed, or replaced | Did this client contribute? |

A hospital may hold thousands of records. A guarantee for changing one record is not automatically a useful guarantee for removing that hospital. Applying a record guarantee repeatedly to every record can yield a very weak whole-client bound. The exact bound depends on the privacy definition and adjacency convention.

**Mathematical deduction.** A differential privacy claim needs a stated adjacency relation and a released transcript. For neighboring inputs $D,D'$, a mechanism must satisfy, for every measurable output event $S$,

$$
\Pr[M(D)\in S]\le e^\varepsilon\Pr[M(D')\in S]+\delta.
$$

The transcript includes every item the attacker can observe, not just the final model. A small measured CIA score for one shadow-loss attack does not establish this inequality for all attackers.

### Participation metadata can make the question trivial

If a server directly sees that client 17 uploaded a message, upload noise cannot hide the fact that client 17 uploaded it. Whole-client participation secrecy **against that server** needs a protocol that also handles identities, message existence, scheduling, counts, or dummy traffic. Noise may still protect the client's data conditional on an upload, which is a different target.

The peer/global-model observer in the paper does not automatically have the server's membership ledger. Future work must explicitly say whether the peer knows participant identities, active-client counts, weights, client-specific noise settings, and its own update/noise. Revealing an active roster containing the target defeats participation secrecy before model inference begins.

## 2. Client-side computation is not a privacy definition

| Arrangement | Who samples noise? | What must be protected? | Key trust boundary |
|---|---|---|---|
| Central/server perturbation | Server | Released aggregate/model | Server sees raw updates and must be trusted within the claim |
| Locally randomized uploads | Clients | Each upload as seen by an observer | A privacy claim needs a bound for that observer and adjacency |
| Distributed implementation of central privacy | Clients jointly | Aggregate revealed after protected aggregation | Individual updates/noise shares must remain hidden under the stated collusion/dropout model |

**Mathematical deduction.** Computing noise on a client does not establish local differential privacy (LDP). To call an upload locally private, its randomized distribution must meet the claimed inequality under the specified local input changes. If the local input is the client's whole dataset, replacing that input is much stronger than replacing one example. Even such a bound conditional on receiving an upload does not hide whether an upload exists.

Distributed central privacy uses client noise shares whose sum has adequate noise, while an aggregation protocol hides individual shares/updates. Small shares can be adequate for an aggregate yet inadequate for individual upload privacy. Conversely, individually private uploads can produce an aggregate with more noise than a trusted central mechanism needs.

**Research implication.** “Move noise to clients” changes who sees unperturbed uploads. “Construct a better distribution” changes the perturbation geometry or law. These should be tested separately.

## 3. A concrete notation for a client's distribution

Let $u_{i,t}\in\mathbb R^p$ be client $i$'s update at round $t$. Choose a clipping rule and form

$$
\bar u_{i,t}=u_{i,t}\min(1,C_i/\|u_{i,t}\|_2),\qquad
\tilde u_{i,t}=\bar u_{i,t}+\eta_{i,t},\qquad
\eta_{i,t}\sim\mathcal N(0,\Sigma_{i,t}).
$$

At a zero update the clipping scale is defined as one. Here $C_i$ is a norm bound;

- Covariance $\Sigma$ is a matrix of variances and cross-coordinate relationships.
- Standard deviations are square roots of diagonal variances.
- Isotropic noise has
  $\Sigma_{i,t}=\sigma_{i,t}^2 I$.
- Diagonal noise allocates distinct variance to each coordinate.
- Full covariance rotates the noise ellipsoid and allows correlated coordinates.
- Low-rank-plus-isotropic noise uses
  $\Sigma_{i,t}=U_{i,t}A_{i,t}U_{i,t}^{\top}+\lambda_{i,t}I$, with positive semidefinite $A_{i,t}$ and a positive floor
  $\lambda_{i,t}>0$.

The floor supplies randomness outside the estimated subspace. Pure low-rank noise has zero variance elsewhere and can disclose a differing component exactly if neighboring outputs differ in that direction.

“Client-specific” could mean different scalar scales, different covariance geometry, different non-Gaussian families, or different samplers. A paper should specify which. Clients may also use a common distribution estimated jointly; that is adaptive distributed noise, but does not necessarily provide individual geometries.

**Research hypothesis.** Each client could estimate directions from minibatch gradients, update histories, a public reference dataset, or an approximation to loss curvature. These are distinct constructions. Statistical spread of gradients is not automatically sensitivity, and curvature is not automatically participation leakage.

## 4. What survives averaging? The essential control experiment

**Mathematical deduction, fixed weights and conditional independence.** For linear aggregation with fixed nonnegative weights
$\alpha_i$,
$\sum_i\alpha_i=1$, and independent zero-mean Gaussian noises,

$$
\tilde U=\sum_i\alpha_i\bar u_i+\sum_i\alpha_i\eta_i,
\qquad
\operatorname{Cov}\left(\sum_i\alpha_i\eta_i\right)
=\sum_i\alpha_i^2\Sigma_i.
$$

The squared weights are essential. Averaging variances with unsquared weights is wrong. If noise shares correlate across clients, additional terms
$\sum_{i\ne j}\alpha_i\alpha_j\operatorname{Cov}(\eta_i,\eta_j)$ appear. This equation addresses linear aggregation; median aggregation and perturbation before clipping generally do not obey it.

For $n$ equally weighted clients with common local covariance
$\Sigma_L$, the aggregate covariance is
$\Sigma_L/n$. To emulate server Gaussian covariance
$\Sigma_S$, choose
$\Sigma_L=n\Sigma_S$. Thus local **standard deviation** must be
$\sqrt n$ times server standard deviation, not $n$ times and not equal to it.

Example: four clients, server per-coordinate standard deviation 0.1. Each client samples standard deviation 0.2. The averaged variance is
$4(1/4)^2(0.2)^2=0.01$, giving aggregate standard deviation 0.1. Using local standard deviation 0.1 would give aggregate standard deviation 0.05.

With unequal weights and a common local scalar standard deviation,
$\sigma_L=\tau/\sqrt{\sum_i\alpha_i^2}$ achieves aggregate standard deviation
$\tau$. For weights (0.7, 0.1, 0.1, 0.1), the sum of squares is 0.52. Local standard deviation about 0.1387 matches server standard deviation 0.1; using 0.2 gives aggregate standard deviation about 0.1442. The experiment must inspect actual aggregation weights, especially quantity-skewed partitions; a client count alone does not determine weighted covariance.

### Why relocating matched Gaussian noise alone cannot improve model-only leakage

If the same clipped updates, weights, and total covariance are used, the linear aggregate has exactly the same conditional distribution as server-side Gaussian perturbation. A global-model-only attacker therefore receives the same output law in that round. If this equality holds conditional on every training history, and all other operations are the same, the entire model transcript has the same distribution by induction over rounds.

This is stronger than saying “same noise power”: the **whole covariance**, mean, release timing, conditional noise law, clipping order, and observer's side information must match. Equal trace alone does not imply equal protection. A peer who knows its own sampled noise can subtract it; equivalence must be checked for the joint distribution of the transcript and its side information, not just the marginal model distribution.

A client-side benefit could therefore come from a different observer/trust boundary, better geometry, altered local training, or a different temporal law. A simple client-side isotropic implementation with matched aggregate covariance is a necessary control, not evidence of a new defense by itself.

### Clipping and weighting are part of the mechanism

Adding noise after clipping yields a Gaussian translation conditional on the update. Adding noise before clipping truncates/transforms that distribution. Server clipping of already noised uploads changes the aggregate noise law again. Adding noise to every local gradient step changes optimization and is not equivalent to perturbing the final upload.

For a fixed-weight sum, replacing a vector with norm at most $C_i$ changes the mean by at most
$2\alpha_i C_i$. Replacing it by a zero contribution changes it by at most
$\alpha_i C_i$, **if the denominator and all other weights stay fixed**. Removing a client and renormalizing weights also changes other contributions. Formal sensitivity must include this effect. The repository's IN/OUT removal should not silently be analyzed as fixed-denominator zero padding.

## 5. A two-dimensional explanation of directional noise

**Mathematical deduction, Gaussian toy model.** Suppose the attacker observes a two-coordinate release. The OUT mean is (0, 0), the IN mean is (1, 0), and both have the same covariance. Consider:

| Covariance | Standard deviations | Total variance / expected squared noise norm |
|---|---|---:|
| Isotropic $\operatorname{diag}(2.125,2.125)$ | About (1.458, 1.458) | 4.25 |
| Directional $\operatorname{diag}(4,0.25)$ | (2, 0.5) | 4.25 |

Both spend the same total variance. The directional mechanism places more noise in the coordinate where the participation means differ, so that particular target is harder to distinguish. It adds less noise in the other coordinate, potentially preserving useful learning there.

With mean difference
$\Delta$ and common positive definite covariance
$\Sigma$, define
$d^2=\Delta^{\top}\Sigma^{-1}\Delta$. Projecting onto the likelihood-ratio direction gives, for the optimal Gaussian discriminator over independent IN and OUT releases,

$$
\operatorname{ROC\!\text{-}AUC}=\Phi(d/\sqrt2).
$$

For the example, isotropic $d^2=1/2.125$, while directional $d^2=1/4$. The latter is harder to attack along (1,0). But for a different target whose participation changes (0,1), directional $d^2=1/0.25=4$: substantially easier to attack. Protection of one estimated direction can sacrifice another target or another attack.

This analytic AUC describes a specified Gaussian one-release model and its optimal likelihood-ratio score. It is **not** the project's folded paired checkpoint statistic and is **not** a prediction of measured FL attack scores.

The utility analogue is local: if task loss is twice differentiable with positive semidefinite local Hessian $H$, then zero-mean noise has approximate expected second-order loss increase
$\frac12\operatorname{tr}(H\Sigma)$. This suggests studying noise along low-curvature directions, but only as a local approximation. Privacy-sensitive directions may also be high-curvature directions; the trade-off need not disappear.

## 6. The central proof obstacle: data-dependent distributions

If a client sets
$\Sigma_i=\Sigma_i(D_i)$, neighboring worlds can differ in **both mean and covariance**. Hiding the covariance matrix itself does not eliminate this dependence: the distribution of sampled outputs can reveal it.

**Mathematical deduction.** For Gaussian worlds with positive definite covariances,

$$
\log\frac{p_D(y)}{p_{D'}(y)}
=\frac12\log\frac{\det\Sigma_{D'}}{\det\Sigma_D}
-\frac12(y-\mu_D)^\top\Sigma_D^{-1}(y-\mu_D)
+\frac12(y-\mu_{D'})^\top\Sigma_{D'}^{-1}(y-\mu_{D'}).
$$

The determinant and quadratic terms show why plugging an observed variance into a fixed-covariance accountant is insufficient. Even equal means can leak through unequal variances. In client removal, both covariance contributions and aggregation weights can change, even if remaining clients' construction rules stay fixed.

Potential proof routes to investigate in literature include a public/fixed geometry, a privately estimated geometry with composed accounting, bounded changes of the calibration statistic, or a direct privacy analysis of the complete adaptive output law. These are search directions, not guarantees established here. A positive eigenvalue floor avoids zero-noise directions but does not by itself supply a privacy bound. A realized maximum or average variance does not certify the whole mechanism.

## 7. Estimating a useful distribution is a statistical research question

For millions of model parameters, storing full covariance costs $O(p^2)$. A covariance estimated from $m$ independent vectors has rank at most $m-1$ after centering. Small clients will need diagonal, layer-wise, low-rank, or shrinkage approximations. Correlated minibatches and dependent update histories further reduce effective information.

For each proposed estimator, record:

1. **Input:** examples, per-example gradients, minibatch gradients, whole updates, public data, or previous releases.
2. **Meaning:** variability of optimization, sensitivity bound, task curvature, or a proxy for participation signal. These are different objects.
3. **Cost:** extra passes/backpropagation, memory, upload overhead, and dependence on client dataset size.
4. **Stability:** behavior for tiny clients, rare labels, non-IID/domain-shift clients, near-zero gradients, drift, and poorly conditioned covariance.
5. **Observability/accounting:** whether statistics, bases, scales, sample counts, or diagnostic logs are released; how private estimation and repeated releases are analyzed.

A client cannot generally observe its exact IN-versus-OUT global-model difference. An estimator built from local gradients is a proxy whose alignment with participation leakage must be measured. Optimizing against one shadow-loss attack risks suppressing that score while leaving another discriminator effective.

## 8. Noise across time is a separate mechanism choice

Independent fresh noise has a block-diagonal conditional covariance across releases. Reusing one noise vector introduces off-diagonal temporal covariance. In the simple release
$y_t=\mu_t+\eta$, subtracting two rounds removes the reused noise entirely:
$y_t-y_{t-1}=\mu_t-\mu_{t-1}$. In real FL, past noise also changes future training, so exact cancellation of underlying training effects does not follow; the toy example still explains why release-by-release variance is insufficient.

For a Gaussian transcript with fixed means and block covariance
$K$, the analogous discrimination quantity is
$\Delta_{1:T}^{\top}K^{-1}\Delta_{1:T}$ when $K$ is positive definite. Two defenses with identical per-round marginal variance can have different transcript leakage. Deliberate temporal correlations require a transcript analysis, not an automatic claim of either improvement or failure.

Repeated uploads also require privacy composition or a direct adaptive-transcript proof under the stated adjacency. Client dropout and collusion can change the unknown noise remaining in an aggregate. Subtracting known noise shares leaves only unknown shares; design guarantees must be based on the assumed minimum honest contributions, not merely the scheduled client count.

## 9. What the existing frontier can and cannot establish

**Repository evidence.** [score_stage.py](../../results/contest_at_scale/auc_target_sweep/score_stage.py) uses
$q=K^{-1}\sum_t[1(s_t^{IN}>s_t^{OUT})+\tfrac12 1(s_t^{IN}=s_t^{OUT})]$ and reports
$\max(q,1-q)$, with
$s=-\text{clean shadow loss}$. Utility is the average final server accuracy across the IN and OUT runs. This differs from ordinary ROC-AUC on independent attack trials. The fold selects direction on the scored data; checkpoints are dependent. See the [evidence audit](../project_evidence_audit.md) for verified values and constraints.

The server metric calibration in [metricdp_strategy.py](../../metricdp_pytorch/metricdp_strategy.py) uses inverse maximum pairwise mean-layer model distance. The audit records the paper/code standard deviation
$zC/(nd_t)$ and the paper's contradictory explanatory prose. The implementation explicitly cautions that empirical distance calibration alone does not supply a formal DP guarantee.

For future comparison, retain the frontier as an exploratory reference while adding independent participant trials, held-out attack/direction calibration, separate parameter search and final seeds/targets, adapted attacks, confidence intervals at the trajectory/target level, and more than overall accuracy for imbalanced tasks. Equal noise multiplier or equal accuracy does not imply equal leakage.

## 10. Reusable extraction card for each paper

An agent should populate every field or write **not reported / not verified**:

| Field | Required extraction |
|---|---|
| Source and status | Title, stable primary-source URL/DOI, version/date, sections and equations actually read |
| Intended privacy target | Record membership, whole-client participation, attributes, reconstruction, or other |
| Adjacency | Add/remove vs replace; fixed denominator vs renormalization; client/data sampling |
| Observer | Peer, server, external model user; honest/semi-honest/malicious; collusion |
| Observable transcript | Uploads, all checkpoints, final model, counts/identities, calibration statistics |
| Noise placement | Training examples, gradients, final updates, aggregate, or model release |
| Distribution | Family, covariance structure, zero/nonzero mean, coordinate/client/time correlations |
| Construction recipe | Estimator, data used, mathematical rule, pseudocode in plain language |
| Guarantee | Exact theorem and assumptions; privacy unit; composition; private calibration accounting |
| Secure aggregation | Required or optional; minimum honest clients; dropout treatment |
| Evaluation | Datasets, target clients, attack knowledge/adaptation, independent repeats, search/test split |
| Comparison | Matched aggregate covariance baseline; matched utility and matched attack success |
| Practical limits | Compute, memory, small clients, covariance conditioning, rare groups |
| Transfer to this project | Which mechanism component could apply and what claim remains unproved |
| Explanation | A concrete toy example and why the method might preserve utility/protect participation |
| Evidence boundaries | Author claim vs reproduced fact vs reviewer's deduction vs proposed extension |

For each candidate, the handoff should contain a plain-language explanation, a minimal formula, an implementation recipe, an explicit threat model, and one failure case. This lets the owner understand a technique without reading the entire paper while preserving enough source anchors for later verification.

## 11. Provisional mechanism comparison ladder

This ladder defines controls for a future plan; it does not choose the winning technique:

1. Existing server isotropic perturbation and existing distance calibration.
2. Client isotropic perturbation calibrated to the same conditional aggregate covariance.
3. Client-specific scalar perturbation, controlling total aggregate covariance and observer side information.
4. Client-specific geometry at matched total noise power, followed by matched empirical leakage/utility comparisons.
5. Private/stable geometry estimation with accounting and adapted transcript attacks.

If step 2 differs substantially for a model-only observer under truly matched laws, inspect clipping, weights, release timing, and implementation before attributing a research advantage. A credible advance must identify which part of construction produces the benefit and which privacy target it supports.
