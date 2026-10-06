# Independent robust temporal-Gaussian geometry review

2026-10-06. Read `research/literature_review/temporal_noise_followup.md`, including its stronger robust workload extension. This review verifies the mathematical deduction, rather than independently rereading its referenced primary papers. No experiment or mechanism implementation.

**Verdict:** the SDP-duality proof is correct under the stated full row-ball, spatially isotropic Gaussian, fixed public linear-workload assumptions. Optimized independent temporal noise attains the optimal workload noise error when every workload column is nonzero. Temporal correlation supplies no improvement for this specific robust problem. The result is valuable directional knowledge because it strengthens the earlier final-prefix argument to arbitrary fixed linear workloads; its restrictions must accompany it.

## Exact robust constraint

Take c>0, rho>0, temporal covariance K positive definite, and dimension d≥T. For a fixed nonadaptive Gaussian mean shift V with rows satisfying ||v_t||≤c, its zCDP parameter is

    .5 tr(K^(-1) V Vᵀ).

The normalized Gram matrix Q=VVᵀ/c² has Q positive semidefinite and diagonal≤1. Conversely every such Q is realizable by T vectors in R^d when d≥T, using a square-root factor embedded in d columns. Thus the robust constraint over every admissible row sequence is exactly

    max_{Q≽0, diag(Q)≤1} tr(K^(-1) Q) ≤ b,
    b=2rho/c².

The dimension condition is essential for this exact identification: if d<T the Gram matrices also have rank≤d, and dropping that restriction produces a potentially conservative SDP relaxation. The conclusion for the relaxed certificate is still valid, but would not establish optimality over the actual smaller-dimensional privacy-feasible set.

## Duality and optimum

The primal feasible set is compact: its diagonal bounds imply trace≤T, and positive semidefiniteness bounds the remaining entries. A strictly feasible point is Q=.5I, so Slater's condition holds. Its SDP dual is

    min sum_t lambda_t
    subject to diag(lambda)≽K^(-1), lambda≥0.

Consequently every privacy-feasible K admits an optimal dual lambda with sum lambda≤b. In fact each lambda_t≥(K^(-1))_tt>0. This matters because reciprocals are well-defined even if some workload diagonal entries vanish.

Inversion reverses positive-definite matrix order, giving

    K≽diag(1/lambda_t).

For W=AᵀA≽0, taking its positive-semidefinite trace contraction and then applying Cauchy–Schwarz gives

    tr(WK) ≥ sum_t W_tt/lambda_t
      ≥ (sum_t sqrt(W_tt))²/(sum_t lambda_t)
      ≥ (sum_t sqrt(W_tt))²/b.

If all W_tt>0, set S=sum_t sqrt(W_tt), lambda_t=b*sqrt(W_tt)/S and K=diag(1/lambda_t). The robust privacy maximum equals sum lambda=b, since it only reads Q's diagonal and is attained by diagonal entries one. This covariance therefore meets privacy and attains the lower bound. This proves global optimality among all SPD temporal K, not merely a particular triangular encoder family.

If some W_tt=0 but W is nonzero, positive semidefiniteness also makes those workload rows/columns zero. The formula remains an infimum: allocate vanishing positive lambda to ignored rounds, with their noise variance tending to infinity, and allocate the remaining budget in the proportions above. A finite positive-definite covariance need not attain that infimum. Do not implement lambda=0 as a finite zero/infinite-variance sampler without specifying the reduced observable workload/privacy problem. If W=0 every workload noise error is zero, regardless of feasible covariance.

For two-round prefix sums, W has diagonal(2,1), so the per-coordinate optimum is (sqrt(2)+1)²/b. The extension's arithmetic and its comparison against equal-allocation6/b are correct.

## Normalization, adaptive use and observer limits

With covariance K⊗I_d, tr(WK) is the noise MSE **per spatial coordinate**. The total expected squared Frobenius error is d*tr(WK), with optimal value d*S²/b. This common dimension factor does not alter the optimizer or no-correlation conclusion. Any reported total-vector MSE should retain it. Nonidentity spatial utility weights or general spatiotemporal covariance are a different optimization problem.

The fixed-query Gaussian divergence calculation alone is not a proof for arbitrary adaptive training. A causal Gaussian factorization can furnish a uniform adaptive certificate when the same-history target differences satisfy the complete row-ball bound along every history, with fresh innovations and appropriate martingale/adaptive-stream accounting. Under that certificate, this robust optimization limit still applies: it optimizes a sufficient privacy constraint and a fixed linear noise workload. A claim of exact feasibility for a specific adaptive training algorithm needs its own theorem or audited transfer; it is not established by treating its random transcript as one Gaussian with a fixed mean.

The row-ball domain deliberately protects every sequence within its envelope. Actual whole-client updates across rounds may obey a narrower reachable set. The theorem does not exclude a correlation gain after proving a tighter set, but empirical trajectory similarity alone cannot authorize that restriction. Full participation and a horizon T also matter: sparse/minimum-separation contributions have different privacy sets, so published gains from those schedules cannot contradict or automatically transfer to this bound.

In the peer game K denotes the coalition-conditioned *unknown* transcript covariance after removing known noise shares. Shapes, public weights, clipping bounds, covariance allocation and permitted coalition/dropout set must obey the same fixed/protected-history contract. A full broadcast's utility covariance may be larger than K if it includes known coalition noise; the displayed optimum is not automatically the optimum of that additional distributed allocation problem. Nor does it protect a server that accesses individual uploads.

Finally A must be fixed public and the objective must be a linear noise workload. Actual trained-model loss depends on data, clipping, optimizer dynamics and nonlinear interaction with previous noise. Public noise placement, private learned geometry, non-Gaussian noise and dimension-restricted sensitivity all lie outside this result. The justified research implication is to deprioritize generic temporal correlation based only on cancellation, compare against optimally allocated independent Gaussian controls, and require a proved additional structure before claiming a temporal utility advantage.
