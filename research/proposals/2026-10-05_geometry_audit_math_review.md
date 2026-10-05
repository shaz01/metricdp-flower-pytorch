# Independent mathematical design review: realistic client-geometry audit

2026-10-05. Design review only. This note adds no experiments, data-derived findings, privacy certificate or novelty claim. The purpose is to determine whether real client geometry warrants another accounted constructor after the bounded toy failures, rather than choose a synthetic population that makes adaptation win.

**Recommendation:** first audit persistent client signal geometry at common model states, then test whether it improves a distortion-inclusive *global* utility comparison against optimized shared geometry. A noise-only covariance comparison under common Euclidean clipping is not a promising first diagnostic: the known PSD-domination result already constrains that construction. Client-specific gradient second moments alone are insufficient justification for client-specific privacy noise.

## 1. Keep three mathematical objects separate

At fixed model state $w$, define per-example gradient $g_{ij}(w)$ and actual local contribution/update $u_i(w)$ under the specified training recipe. These may be different random variables: per-example gradients, minibatch averages and multistep client updates have different scaling and geometry.

1. **Gradient second moment:** $M_i=\mathbb E[g_{ij}g_{ij}^\top]$. Its centered covariance is $M_i-\mu_i\mu_i^\top$, where $\mu_i=\mathbb E g_{ij}$. Uncentered moments can appear anisotropic simply because a strong mean gradient points in one direction. Record both and name which is used. Neither is automatically a utility Hessian or sensitivity set.
2. **Utility curvature:** a local Hessian $H_i=\nabla^2J_i(w)$ and a global evaluation Hessian $H_G=\nabla^2J_G(w)$ answer different questions. Noise in a *shared aggregate model* is scored by $H_G$, not by assigning client $i$ its own $H_i$ merely because its noise was generated there. Indefinite full-network Hessians do not support a PSD quadratic-noise dominance interpretation without an explicit restricted/PSD approximation and locality check.
3. **Sensitivity geometry:** a body containing every allowed neighboring contribution difference. This is a worst-case property of the declared mechanism/adjacency, not an empirical cloud of observed gradients. Quantiles, principal components and observed extrema need a separate privacy/sensitivity argument before calibrating a formal release.

For an exact last-layer example, hold features $x$ fixed, let logits be $Wx$, and use softmax cross entropy. The gradient is $g=(p-e_y)x^\top$. In a consistent vectorization its Hessian is the Kronecker product of $xx^\top$ and $\operatorname{diag}(p)-pp^\top$. The observed-label gradient outer product instead contains $(p-e_y)(p-e_y)^\top$: it depends on label prediction error and is not generally that Hessian. Averaging over a label sampled from the model distribution yields the latter class factor, but empirical labels need not follow that model. Thus squared-gradient/Fisher-like statistics must not be renamed “curvature” without stating the exact estimator and assumption. This is an algebraic distinction, not an empirical conclusion about this dataset.

## 2. Why common global utility often favors a shared noise design

Let the unnoised next shared model be $w_*=w+\sum_i\alpha_i u_i$, with fixed public weights. A joint clipping/noise profile changes it by

$$
\Delta=b+\sum_i\alpha_iZ_i,
\qquad b=\sum_i\alpha_i(v_i-u_i).
$$

For fixed profiles and updates, assume independent centered noise with covariances $\Sigma_i$. The conditional second-order global loss change is

$$
\mathbb E[J_G(w_*+\Delta)-J_G(w_*)]
\approx \nabla J_G(w_*)^\top b
+\tfrac12 b^\top H_G b
+\tfrac12\sum_i\alpha_i^2\operatorname{tr}(H_G\Sigma_i).
$$

Do not omit the linear bias term unless stationarity at $w_*$ is established. Clipping biases combine before squaring; they can cancel or reinforce. A sum of local reconstruction costs or local $H_i$ penalties is not this global objective.

With equal weights, budgets and a common feasible covariance/sensitivity family, every client faces the same noise-only minimization of $\operatorname{tr}(H_G\Sigma_i)$. At least one shared minimizer exists whenever the family has a minimizer; distinct client-optimal covariances are not required by differing local gradient moments. Uniqueness is not assumed. Unequal fixed weights multiply this same objective by positive constants and do not change an unconstrained per-client argmin unless budgets/constraints are jointly coupled. Different valid sensitivity bodies, clipping distortion, estimator restrictions or time/history effects can change the conclusion; they must be demonstrated, not inferred from $M_i$.

In particular, the previously reviewed ellipsoidal law enlarged to contain a common Euclidean replacement ball has covariance proportional to $A_i/\lambda_{\min}(A_i)\succeq I$. It cannot improve any PSD quadratic noise cost over the isotropic reference at the same update budget. Measure something beyond that already constrained noise-only effect before constructing another such profile.

The meaningful geometry hypothesis is that different clients have useful updates that survive **different** valid clipped bodies, leading to a better *global clipping-plus-noise tradeoff* after construction cost. Compare actual held-out global loss where feasible, with the quadratic expansion used as a diagnostic whose accuracy is checked over the chosen perturbation scales.

## 3. A bounded real-data audit that can answer this hypothesis

Use cached data or the existing documented dataset/partition pipeline; report availability and provenance transparently. Keep the original partition rule, labels and task. Do not assign artificial client curvature matrices or rotate their data specifically to produce a desired adaptation effect.

At several common model checkpoints, compute a declared last-layer/low-dimensional representation of gradients or updates for each client. All clients must be compared at the same model state and with the same coordinate map; client-specific training endpoints confound geometry with model drift. A fixed public projection or architecture-defined block is cleanest. A basis fitted from pooled private gradients is itself a data-dependent diagnostic/constructor and must be labeled accordingly.

Report mean gradients, centered and uncentered moments, scale-normalized orientation as well as raw norms, sample counts, and estimation uncertainty. Distinguish client differences from finite-sample noise by splitting each client's observations for estimation and assessment. Evaluate persistence at future common checkpoints and across disjoint local samples; resampling the same small sample alone does not supply independent client evidence. Near-degenerate eigenspaces make single principal-vector angles unstable, so report eigengaps and subspace/projector overlap rather than claim that every changing eigenvector reflects true drift. Regularization/floors must be fixed and disclosed.

Compare geometry inferred from local gradients with independently estimated global held-out curvature and the actual current client update. Persistent $M_i$ differences are only the first gate. The next gate is whether they predict out-of-sample clipping distortion or global held-out utility at the next checkpoint better than shared public geometry. Client-local curvature can be reported as contextual evidence, but the shared-model endpoint must still use global loss.

For a finite public bank, test joint clipping/noise profiles against optimized shared profiles with the same bank, radius range, budgets and information available for tuning. Include isotropic, strongest shared shaped/body-family controls justified by the low-dimensional setup, and a known-data-independent baseline if the task admits one. The noise-only component may be integrated analytically via its covariance, while the nonlinear held-out loss and global clipping bias are assessed directly. Freeze choices on development data and assess on separate samples/checkpoints; do not choose a geometry on the same evaluation loss later reported as its benefit.

## 4. Raw-data/oracle geometry is an explicitly unaccounted diagnostic

A client profile chosen using raw local statistics can show whether there is any useful geometry signal before paying to estimate it. It does **not** supply a whole-client privacy certificate: although every fixed profile law may be valid, the selected law/body changes with private input. Protected estimation, protected history or a direct data-dependent-law proof is still needed for deployment.

An oracle chosen to optimize a specified empirical bank score is a best-in-that-bank diagnostic on that score. It is not a universal upper bound on all privacy mechanisms, future checkpoints, alternative objectives or unknown populations. An oracle tuned on final held-out loss is additionally an in-sample information advantage and cannot be reported as out-of-sample performance. A negative result may deprioritize that bank/statistic under that protocol; it does not rule out all client-specific laws.

Similarly, a positive raw-oracle gain merely creates headroom. The eventual estimator/construction cost, stale-profile behavior and complete observer law can erase it. Quantify gain against optimized shared geometry before spending compute on an accounted constructor. If only local $M_i$ differences persist but global utility favors a shared profile, prefer the shared law for this mechanism/task rather than forcing a personalization conclusion.

## 5. Low-dimensional scope and stop criteria

A last-layer/low-dimensional audit is a justified way to limit cost, but its formal release scope must remain explicit. Protecting a projected vector does not protect an additional unnoised private complement or other layers. A mechanism that releases only that block and leaves other coordinates public changes the learning estimator; include its distortion/task effect. A diagnostic projection used solely for local analysis carries no claimed full-model guarantee.

Proceed toward another constructor only if useful geometric differences persist out of sample and create global utility headroom over strong shared clipping/noise controls. If the signal is unstable, driven only by a mean-gradient/radius effect, or irrelevant after global curvature and clipping bias are accounted, revise or stop that route. These are decision gates to evaluate, not findings already established by the repo's toy results.

The audit is about whether real client geometry merits further mechanism research. It does not itself establish CIA protection, privacy-accounted distribution estimation, paper novelty or readiness for a neural-network privacy sweep.

## Concrete Fashion-MNIST audit implementation review

Reviewed `research/calculations/client_geometry_audit.py` on 2026-10-05. No substantive mathematical implementation error was found in the inspected calculations. This is an offline four-class, pooled-pixel softmax audit using real cached data, not an image-model privacy experiment or a continuation of the synthetic curvature construction.

The 68-parameter gradient ordering is class-major with 17 feature/bias entries per class. The `aibj` Hessian tensor matches that ordering, and the two projected class-bias coordinates are indices 16 and 33. Their projected gradient/Hessian formulas agree with the corresponding full softmax entries. The code's finite-difference and PSD checks are appropriate numerical checks of these identities. The intervention adds the aggregate two-coordinate update to classes 0/1's bias logits, with all other parameters frozen, so direct cross-entropy evaluation measures the declared projected update rather than a differently reconstructed vector.

For a fixed radius, bank clipping applies the correctly rotated diamond gauge and retains the original vector's radial direction. Let $D_k=R_k\operatorname{diag}(a_k^2)R_k^\top$. The selected local noise covariance is $8D_k/\epsilon^2$. With the audit's fixed comparison weights $\alpha_i$, the development quadratic change around the **current** checkpoint is correctly

$$
g^\top m+\tfrac12m^\top Hm
+\frac4{\epsilon^2}\sum_i\alpha_i^2\operatorname{tr}(HD_{k_i}),
\qquad m=\sum_i\alpha_i v_{i,k_i}.
$$

This is a model intervention relative to no intervention at the current checkpoint. It differs from expanding around an already updated unnoised model and using only a clipping-bias vector. The linear global gradient term is included correctly. Independent rotated local Laplace draws and direct held-out cross-entropy use the corresponding aggregate mean/covariance. Paired shared/oracle innovations are valid variance-reduction aids for conditional utility comparisons, not protecting randomness exposed to an adversary.

The $4^8$ enumeration contains every shared-profile assignment; the assertion that the development oracle optimum is no greater than the shared development optimum is therefore valid within the declared radius/bank and quadratic score. It does not imply the same ordering for nonlinear held-out cross-entropy, unseen populations or all possible noise mechanisms. Radius choice is common across clients in each assignment; this is not optimization over arbitrary client-specific radii or learned bodies.

Shared and oracle configurations are selected using local fit gradients and global development curvature. Official test loss is used to evaluate, not select those configurations. The shared diagnostic nevertheless uses **raw private information** and an unprotected checkpoint, just like the oracle: “shared” must not be equated with an already deployable public/private mechanism. Epsilon values label fixed-profile noise calibrations only; they do not certify the raw model trajectory, data-dependent assignments, weighting metadata or complete observer transcript. The recorded scope correctly states that limitation.

### Evidence limits still requiring explicit interpretation

- Current top-two overlap summaries concern **uncentered gradient second moments**. Client mean-gradient directions can explain their heterogeneity. Centered covariance, eigengaps, normalization effects and comparison with saved local Hessians remain necessary before attributing the pattern to persistent utility-relevant covariance/curvature.
- Local fit/check splits are disjoint and stratified. Persistence compares the same client at different common model snapshots, which is the right initial comparison. It still does not supply an independent-population uncertainty interval; seeds/partitions reuse the corpus, and all rows reuse the official test subset.
- The 2048-noise-draw standard errors and paired intervals measure **conditional perturbation-noise uncertainty** for fixed updates and fixed test examples. They do not include client sampling, data split, curvature-estimation, checkpoint-selection or future-data uncertainty. A tiny Monte Carlo interval cannot establish a general real-client effect.
- The $0.001$ held-out CE gate is an explicitly bounded headroom criterion, not a universal effect size or a family-wide significance test. Passing the point-estimate gate is different from having a noise-only interval wholly beyond the threshold. Multiple checkpoint/partition/budget looks remain exploratory; repeated use of the test subset must not become unreported tuning.
- The 80% dominant-label partition is an explicitly disclosed stress regime, not an estimate of natural client prevalence. It may reveal a mechanism opportunity under controlled label skew, but must be compared with the repository's balanced/quantity regimes and cannot alone establish realistic deployment heterogeneity.
- No intervention and the noiseless aggregate update are relevant interpretive controls. If both noisy diagnostics are worse than doing nothing, a relative oracle gain is headroom inside a harmful family rather than evidence of useful learning.

No code or result files were changed by this independent review. The audit can identify whether a more careful construction is worth investigating; its raw-information utility comparisons cannot establish CIA mitigation, an accounted estimator or novelty.

## Final metric extension and bounded result check

The subsequent code extension addresses the earlier uncentered-only scope. It now computes centered covariance $M_i-\mu_i\mu_i^\top$, the mean-gradient energy fraction $\|\mu_i\|^2/\operatorname{tr}(M_i)$, top-two covariance overlap on disjoint local splits, adjacent-snapshot persistence, and normalized Hessian Frobenius cosine. These formulas are correct. Covariance here uses the empirical-population $1/n$ convention consistently with the second moment, not the unbiased $1/(n-1)$ estimator; that is suitable for the stated geometry diagnostic.

The weighted pooled covariance is explicitly the **mean within-client covariance**, $\sum_i\alpha_i\operatorname{Cov}_i(g)$. It is not the covariance of the combined client mixture, which additionally contains between-client mean variation. Its key is named accordingly. Matrix Frobenius cosine is a valid scale-free comparison and avoids choosing Hessian eigenvectors in nearly degenerate eigenspaces, but high cosine does not establish identical utility-relevant spectra/directions. The eigengap ratio $(\lambda_2-\lambda_3)/\max(\lambda_2,10^{-12})$ is correctly computed for **uncentered moments**; it does not yet diagnose centered-covariance eigengaps. Interpret the corresponding covariance subspace stability with that distinction.

One concrete metric-label discrepancy was identified and reported: the initially inspected `pooled_h_cosine` expression used `fit_stats`, while its key called it `hessian_vs_client_holdout_cosine`. **Resolution recorded by the implementing agent:** the operand was changed to `hold_stats`, the complete audit was rerun successfully, and the committed calculation uses the held-out comparison. The covariance and moment pooled-to-heldout comparisons also use held-out statistics. The independent reviewer identified the initial issue; the implementing agent performed and checked the correction.

Reading the saved artifact confirmed 27 partition/seed/checkpoint rows and 81 intervention comparisons. At checkpoint 20, centered-covariance mean overlaps were approximately $0.996$ within and $0.996$ between clients for balanced partitions, versus $0.976$ within and $0.633$ between clients for the disclosed label-stress regime. Stress-regime adjacent-snapshot covariance overlap was approximately $0.984$. These support persistent geometry differences in that stress regime within this observed setup, rather than a generic assertion that all real clients differ substantially.

The strongest direct held-out oracle-minus-shared CE difference was $-0.0003631329$; none of the 81 comparisons passed the specified $-0.001$ feasibility threshold. This is a bounded absence of material headroom in the tested two-bias-coordinate family, not a proof that all full-model constructors fail. The saved noiseless raw projected step, noiseless clipped step and quadratic noise penalty help distinguish learning-signal/clipping behavior from added perturbation cost. The same noise-only Monte Carlo uncertainty and shared-corpus/test limitations still apply.

Persistent covariance heterogeneity and insufficient utility headroom are compatible findings. Geometry differences do not by themselves justify another private construction: this audit has not shown a large enough global utility gain over the shared diagnostic in its tested family. Both choices and the underlying checkpoints remain raw-information diagnostics, and epsilon labels remain fixed-profile counterfactual calibration parameters rather than full-transcript privacy budgets. No new jobs, code changes or result edits were made by this final review.
