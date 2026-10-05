# Independent mathematics: broader class-contrast oracle diagnostic

2026-10-05. Review of the proposed broader raw-information audit. This is not an implemented private estimator, CIA result or novelty assessment. The intervention starts from saved unprotected Fashion-MNIST softmax checkpoints and gradients; all epsilon values remain fixed-profile counterfactual labels, not full-protocol privacy budgets.

## Contrast spaces preserve a clear parameter interpretation

Let $B\in\mathbb R^{4\times3}$ satisfy $B^\top B=I$ and $B^\top\mathbf1=0$. For class-major flattening of a $4\times17$ head, the full contrast map is $P=B\otimes I_{17}\in\mathbb R^{68\times51}$. Projection/reconstruction are $z=P^\top u$ and $u_{\mathrm{contrast}}=Pz$; projected curvature is $P^\top HP$. In matrix form a $3\times17$ contrast update reconstructs as $B Z$.

The bias-only contrast space uses $B$ in the four class-bias coordinates, leaving all feature coefficients frozen. It has three dimensions and includes every class-bias contrast, unlike the prior two-bias probe. The full-head space has 51 dimensions and omits only common-logit directions: adding the same linear feature function to all four class logits does not change softmax probabilities or cross entropy. Softmax gradients sum to zero across classes, so they already lie in this contrast subspace, up to numerical roundoff. This is an exact head-gauge reduction, not evidence that arbitrary private complements elsewhere in a model may be released unprotected.

Verify class-major flattening and bias placement explicitly. A public fixed Helmert convention is appropriate; rotating/fitting the contrast basis from private geometry would change the diagnostic and construction information. Retaining one of three Helmert groups is a fixed structural bank choice, not optimization over all directions or covariance bodies.

## Clipping, covariance and quadratic objective

For each public profile, let $r_j>0$ be its coordinate semiaxes and use radial clipping into the weighted $\ell_1$ body:

$$
v=\frac{u}{\max(1,\sum_j|u_j|/r_j)}.
$$

Equal semiaxes form one profile. The other three preserve one contrast group at common radius $C$ and shrink the other groups to $0.25C$. A fixed-profile replacement-calibrated coordinate noise draw is independent Laplace with scale $2r_j/\epsilon$, giving covariance $8\operatorname{diag}(r_j^2)/\epsilon^2$. This remains true in both 3 and 51 dimensions. Dimension dependence appears in the sum of coordinate noise costs and the clipping gauge; one cannot compare the same $C$ across dimensions as though it defined the same Euclidean sensitivity/utility constraint.

For selected profiles and fixed public comparison weights $\alpha_i$, put

$$
m=\sum_i\alpha_i v_i,\qquad
V=\frac8{\epsilon^2}\sum_i\alpha_i^2\operatorname{diag}(r_i^2).
$$

Here $m$ is the actual proposed update relative to the current checkpoint. With projected global development gradient $g$ and PSD Hessian $H$, define

$$
\ell=g^\top m,\qquad Q=\tfrac12m^\top Hm,\qquad N=\tfrac12\operatorname{tr}(HV).
$$

A postprocessed intervention $t(m+Z)$ has development quadratic change

$$
f(t)=\ell t+(Q+N)t^2,\qquad0\le t\le1.
$$

For $D=Q+N>0$, the exact optimum is $t^*=\operatorname{clip}[-\ell/(2D),0,1]$. If $D=0$, choose $t=0$ for $\ell\ge0$ and $t=1$ for $\ell<0$. Negative $D$ would indicate a different indefinite-curvature problem or a numerical error, and cannot be handled as this convex optimum without checking endpoints. Exact projected softmax cross-entropy curvature is PSD, so the present construction should have $D\ge0$ within numerical tolerance.

**Critical evaluation identity:** scaling must apply to both the mean update and sampled noise. The evaluated increment is $t(m+Z)$, and its covariance is $t^2V$. Scaling only $m$ while retaining the original noise would not evaluate the optimized objective. Fixed public postprocessing preserves a fixed-profile privacy certificate; it does not justify an unaccounted private selection of $t$.

No-op is included by $t=0$ and has zero intervention change. This does not make the saved raw checkpoint itself private. The selection criterion must include the linear gradient term; optimizing only noise trace or mean squared update norm would answer another question.

## Oracle and shared comparison scope

For every common radius and epsilon label, enumerate all $4^8$ profile assignments and optimize $t$ separately for each. The shared arm must get the same radius grid, same global development gradient/Hessian, same analytic postprocessing optimization and all four common-profile assignments. Then each shared configuration is in the oracle's candidate family, so the oracle's best **development quadratic score** cannot exceed the shared best. That inclusion is a useful correctness assertion.

The shared arm is a stronger finite-family control than the earlier unshrunk profile comparison. It is still raw-information calibration, not a public or deployable private mechanism. The oracle uses client raw gradients and globally combined assignments; it is not an individually implementable client estimator, and it has no unaccounted privacy guarantee. Nor is it a universal upper bound over all methods, directions, radii, objectives or future samples.

The global objective correctly combines all clipped means before computing curvature cost. If one instead expands about a previously updated unnoised model, clipping distortion $b=\sum_i\alpha_i(v_i-u_i)$ and its linear/quadratic effects are required. In this design, expansion about the current checkpoint directly uses $m$, so clipping effects are already represented in the actual intervention mean; do not add a second duplicate bias penalty. Retain no-intervention and noiseless raw/clipped step values as interpretable controls.

An exact oracle development advantage may reverse under nonlinear evaluation or curvature-estimation noise. Analytic minimization over $t$ plus exact assignment enumeration establishes only the optimum inside the finite radius/bank and development approximation. It does not certify global optimality for held-out cross entropy or jointly optimized arbitrary distributions.

## Evaluation and decision limits

Use the specified second official-test slice, per-class indices 256–511, disjoint from the previous per-class first-256 slice. Verify the recorded original indices rather than relying on the word “fresh.” The same saved training checkpoints and development calibration are allowed; hold out the new test examples from selecting profiles/radius/postprocessing. Score actual cross entropy by reconstructing contrast perturbations correctly and adding $t(m+Z)$ to the head or bias block.

Use 1024 paired independent noise innovations within each shared/oracle comparison. Sampling uncertainty describes perturbation noise conditional on the fixed checkpoints, clients and test slice, not independent client populations or a broad data uncertainty interval. Repeated corpus/partition/checkpoint/budget comparisons remain exploratory; the $0.001$ material-headroom threshold is a specified bounded decision gate, not a universal significance or novelty criterion. Passing it on the direct CE point estimate is distinct from a paired interval wholly beyond it.

A positive broader-oracle result would establish headroom only inside this unaccounted intervention family. A negative result would rule out material tested headroom, not every full-model distribution constructor. Either should inform the next research decision before spending compute on a private estimator or CIA sweep. Label-stress heterogeneity remains an explicitly manufactured partition stressor on real images, rather than natural population prevalence.

No code or result files were changed by this mathematical design review. An actual implementation review, once the calculator exists, should specifically verify contrast ordering, grouping, profile containment, weighted covariance, analytic endpoint handling, mean-and-noise scaling, shared inclusion, and development/evaluation separation.

## Actual calculator review

Independently inspected `research/calculations/broader_geometry_oracle.py` on 2026-10-05 after it became available. No material code or mathematical error was found. The implementing agent reports a successful self-check; this independent review inspected that check and the calculation without launching another full audit.

The Helmert basis is orthonormal and perpendicular to the common class vector. Bias positions 16/33/50/67 and the full `kron(B,I17)` mapping match the saved class-major head ordering. The 51-coordinate group blocks correspond to each contrast's 17 coefficients. Projection of mean gradients, development Hessians and evaluated perturbations is consistent. The prior full-head mean gradients are multiplied by the fixed learning-rate factor and then projected; no client-specific basis is silently learned.

Clipping uses the full weighted L1 gauge for each body, and the local noise is independent coordinate Laplace with the matching scales. The assignment evaluator correctly forms weighted linear contributions, every cross-client bilinear contribution to $Q$, and the squared-weight diagonal covariance cost. Shared candidates are identified inside the same exact assignment set and receive the same radius/shrink optimization. Explicit assertions cover oracle inclusion and the no-intervention development score.

The evaluation multiplies both the clipped mean and aggregate noise by the fitted $t$, records the corresponding $t^2$ covariance, and reconstructs every sampled perturbation through the correct contrast map. Direct CE uses stable shifted logits. It records noiseless raw, noiseless clipped unshrunk/shrunk, noise-curvature and aggregate-noise-trace diagnostics, so clipping/shrinkage effects can be distinguished from perturbation cost.

Selections use global development derivatives only. The test Hessian is used solely to log a curvature-based noise penalty, not choose the configuration. The new test slice is constructed from per-class indices 256–511 and its disjointness from the prior audit slice is explicitly checked. Dataset hashes are checked against the saved trajectory source. Paired innovations are generated after development settings are fixed; their use within shared/oracle comparisons preserves the intended marginal laws.

The shrink helper uses a $10^{-20}$ cutoff when treating quadratic cost as zero. This is a numerical endpoint convention rather than exact analytic minimization for every arbitrarily tiny positive cost. The objective and inclusion assertions are otherwise consistent with the stated finite calculation; do not promote the floating-point routine to an arbitrary-precision optimizer theorem. The projected finite-difference, containment, score reconstruction and zero-perturbation CE self-checks are appropriate bounded verification.

Full output was not yet available at this implementation review, so no outcome or feasibility-gate count is asserted here. The raw-information/privacy, finite-family, repeated-corpus and conditional-noise uncertainty limits above continue to apply. No code, results or running jobs were modified by this review.

## Independent final artifact check and recommendation

The subsequently saved JSON contains 36 partition/seed/checkpoint/space rows and 108 epsilon-label comparisons. Reading the NPZ and recomputing every saved paired mean/standard error confirmed all 108 JSON contrasts exactly; no result data were changed and no new experiment was run.

Exactly three point-estimate gates pass. They are the three seeds of the same narrow setting: bias contrasts, label-stress partition, checkpoint 20, epsilon label 8. Oracle CE reductions versus shared are respectively $0.0010714412$, $0.0011261642$ and $0.0010362398$. Two of the three noise-only normal intervals lie wholly beyond the $0.001$ threshold; the third reaches a reduction of only $0.0009947746$ at its less-favorable endpoint. These are conditional perturbation-noise intervals, not independent population replications. The full-head space passes no gate; its strongest reduction is $0.0002364715$.

All three positive cells choose common radius $0.3$, shrink factor one, and oracle assignments `[0,2,0,1,0,2,0,1]`. The shared configuration uses profile 2 for every client, at the same radius and shrink factor. Their difference is therefore not explained by a different total radius or postprocessing allocation.

### The advantage is in the aggregate mean, not reduced noise

For these cells, noiseless oracle-minus-shared CE changes are approximately $-0.001247$, $-0.001313$ and $-0.001254$. The oracle's quadratic noise penalty rises to approximately $0.000354$ from the shared penalty of $0.000179$. Aggregate noise trace rises from $0.00158203125$ to $0.002900390625$ in every positive cell. The final observed utility advantage survives this *larger* perturbation penalty because the chosen clipped aggregate mean yields a more favorable loss change.

The oracle also has a larger aggregate Euclidean deviation from the raw update, about $0.027$ versus about $0.010$–$0.011$. Thus the gain cannot accurately be described as merely preserving the raw update better. It is a favorable global mean/clipping reconfiguration under this objective. A future constructor based only on minimum Euclidean reconstruction distortion need not recover it. A claim that client-specific covariance reduced noise is contradicted by these logs.

### Stronger control and next-step scope

The stable slot assignment suggests a relevant next control: a **fixed publicly specified heterogeneous profile roster**, frozen from development information, alongside the best shared profile. If the same assignments/radius/shrink are prescribed publicly, they produce exactly the same counterfactual output law as the oracle in these cells, without a private per-client selection procedure. That configuration has not yet been established as out-of-sample by observing stability on the same examined tests; freeze its rule and assess independently before claiming such a result. A public roster inferred from private development data also needs its information provenance stated.

This does not make the current raw checkpoints/trajectory private, nor prove that public heterogeneous allocation is generally best. It shows that the present positive headroom does not yet require an individualized private estimator. Any stronger interpretation must distinguish task-specific fixed profile allocation from privately estimating a new client distribution.

The appropriate advancement is a bounded **construction-design review**: identify what changes the beneficial aggregate mean, whether a development-frozen public heterogeneous roster accounts for the apparent gain, and what information an accounted constructor would actually need. The marginal, three-coordinate, disclosed-stress-only gain does not by itself justify implementing a private estimator or launching a CIA/neural-model sweep. Construction cost could easily exceed the roughly $0.001$ CE headroom. If an accounted selection route is later proposed, specify its full transcript guarantee before implementation.

The three positive point estimates are worth retaining rather than rounding away, but they do not establish broad real-client benefit, whole-input privacy, novelty or a full-head mechanism advantage. This final review changed only this note and launched no additional jobs.
