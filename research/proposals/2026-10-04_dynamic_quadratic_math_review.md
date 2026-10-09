# Independent review: changing-update quadratic learning spike

2026-10-04. Mathematical design review only. No code edits, training, attack measurements or novelty claims. This is a deliberately specified finite-dimensional population model, not a theorem that the proposed method improves general FL.

## Objective and actual updates

Take $N$ IID registered clients with $\theta_i=0.8e_1$ or $0.8e_2$, equiprobably, and

$$
F_i(w)=\tfrac12\|w-\theta_i\|_2^2,
\qquad\bar\theta=\tfrac1N\sum_i\theta_i.
$$

The empirical global objective has minimizer $\bar\theta$ and exact excess

$$
\tfrac1N\sum_iF_i(w)-\min_v\tfrac1N\sum_iF_i(v)
=\tfrac12\|w-\bar\theta\|_2^2.
$$

With unit gradient step and $w_0=0$, the first raw update is $u_{i0}=\theta_i$. For public isotropic diamond semiaxis $0<a_0\le0.8$, clip it to $a_0e_{t_i}$ and release $Y_{i0}$ with independent coordinate Laplace noise of scale $2a_0/\eta_0$. Set $w_1=N^{-1}\sum_iY_{i0}$. At the next round the raw gradient step is the **entire vector** $u_{i1}=\theta_i-w_1$. It generally has both coordinates nonzero, even though each original $\theta_i$ lies on an axis.

For the profile selected from $\arg\max_j|Y_{i0,j}|$, define positive public semiaxes $(a_{i1},a_{i2})=(L,S)$ or $(S,L)$. Correct radial diamond clipping is

$$
v_i=\frac{u_{i1}}{\max\{1,\ |u_{i1,1}|/a_{i1}+|u_{i1,2}|/a_{i2}\}}.
$$

It is neither separate coordinate clipping nor clipping only the original signal axis. Draw fresh independent Laplace noise with scales $2a_{ij}/\eta_1$ and release $Y_{i1}=v_i+Z_{i1}$. The model update is $w_2=w_1+N^{-1}\sum_iY_{i1}$, **not** the stationary toy's average of two estimates of a fixed $u$.

## Conditional risk identity and shared-history dependence

Condition on the realized federation and all first-round uploads. Then $w_1$, chosen profiles and clipped vectors $v_i$ are fixed. Fresh second-round noise is independent across clients with covariance $8\operatorname{diag}(a_{i1}^2,a_{i2}^2)/\eta_1^2$. Therefore

$$
\mathbb E\left[\tfrac12\|w_2-\bar\theta\|^2\mid\theta,Y_0\right]
=\tfrac12\left\|w_1+\tfrac1N\sum_i v_i-\bar\theta\right\|^2
+\frac4{N^2\eta_1^2}\sum_i(a_{i1}^2+a_{i2}^2).
$$

For a swapped $(L,S)$ bank the last term is $4(L^2+S^2)/(N\eta_1^2)$. For the public static isotropic diamond with radius $r_1$, it is $8r_1^2/(N\eta_1^2)$. This exact identity is a useful implementation/moment check and permits integrating out later noise when estimating expected objective excess. It must still average over fresh federations and probe noise; it does not eliminate the probe's statistical uncertainty or shared-history clipping bias.

An unnoised, unclipped round 1 would satisfy $w_1+\operatorname{mean}(\theta_i-w_1)=\bar\theta$ exactly. This cancellation is a useful limiting check: the second step can correct the noisy shared model. Clipping disrupts it, and the profile choice depends on the very probe noise that produced $w_1$. Errors are consequently coupled across clients and rounds. The previous IID stationary formula $R_{\mathrm{local}}/N+(N-1)\beta^2/(2N)$ cannot simply be substituted here. A fresh second-round noise term has zero conditional cross moment, but neither the existing model error nor the clipping correction is independent of the probe.

The earlier closed-form profile misclassification probability still describes the first probe's axis label when $a_0>0$, but it does not alone determine dynamic utility. Profile geometry favoring the initial axis need not favor the residual gradient after $w_1$. Do not reuse the stationary semiaxis optimizer or assert its convexity for this changing-update hyperparameter problem.

## Whole-input interactive privacy

The first upload is $\eta_0$-DP against arbitrary replacement of the entire client input after clipping into the public diamond. Each client's profile is a public deterministic function of its already protected upload, so there is no separate RR charge. At every fixed shared history and fixed selected profile, the second upload clips an arbitrary current raw update into that public body and adds the matching replacement-calibrated noise. This is uniformly $\eta_1$-DP, including when local training or profile selection is correlated with past private randomness.

Thus, for a target slot, the conservative interactive transcript containing its uploads, public global models and profile information is bounded by $\eta_0+\eta_1=E$, provided all histories/profile outcomes use valid conditional kernels and no unprotected target-data access enters other messages. Other clients' honest responses to the protected global model do not create an independent raw target-data channel. The public models are downstream functions of protected uploads. Across the federation this is a per-target replacement guarantee, not an $NE$ cost for changing one target input; changing all clients is a different adjacency.

All registered slots retain public equal weights and message traffic in both contribution worlds. For an empty private input, define the raw update to be zero at **every** model history, rather than $-w$ (the latter would be the gradient of an artificial nonempty client with $\theta=0$). It still executes the protected selector, clipping and fresh noise protocol. The utility population used here has only nonempty quadratic clients; that simulation does not itself evaluate the IN/empty game or CIA. Timing, metadata, random seeds and diagnostics remain part of the eventual observer contract. Positive budgets and positive semiaxes are required for the displayed full-dimensional laws; deterministic-zero boundary messages must be defined separately.

## Fair controls, sampling and limits

- A **same-probe static** arm must receive exactly the same $Y_0$ and $w_1$, budgets and learning steps; only later profile construction differs. Give it an optimized public second radius, rather than an arbitrary fixed radius.
- A **tuned static** arm may optimize its public probe radius, later radius and budget split on separate development simulations. Keep this comparison distinct from the exactly matched probe arm, and report bounded search scope rather than global optimization.
- Retain a public **one-release** all-budget mechanism as a different-schedule diagnostic. It can beat a mandated two-step method without being a matched trajectory arm.
- The public population mean is $\mu=(0.4,0.4)$. Under this known IID population, the data-independent estimator $w=\mu$ has expected objective excess $0.16/N$, because $\operatorname{tr}\operatorname{Cov}(\theta_i)=0.32$. This is a zero-private-data baseline, not an estimator that knows the realized federation mean. Its validity relies on the declared public population knowledge.

Tune on development federations and freeze choices before evaluating fresh held-out federation/noise trials. Shared standard noise innovations across comparison arms can reduce variance of paired contrasts, while each individual arm must still have the declared independent-noise marginal law. Those evaluation couplings and seeds are computational comparison aids, not noise randomness disclosed to an attacker. Report uncertainty over independent federation trials, not over clients treated as independent despite a shared model. Aggregate signed paired differences and predefined objective endpoints; avoid choosing the winning arm on the confirmation sample.

This spike tests whether protected-history construction helps genuine residual updates under a shared quadratic objective. It can refute or motivate a candidate within the declared model. It cannot establish CIA protection, a broad research contribution, a useful dense-model noise scale or success outside this population. Formal whole-input privacy follows from the uniform mechanism construction, not from any observed utility or failed attack score.

## Actual calculator and artifact check

Reviewed `research/calculations/dynamic_quadratic_probe.py` and the saved JSON/NPZ on 2026-10-04. No substantive mathematical or evaluation-leakage error was found. The code uses the full residual vector and the sum-of-absolute-whitened-coordinates diamond gauge, fresh second-round noise, correct global objective and the declared protected-history selector. Development configurations, matched-probe static choices and shrinkage coefficients are all fitted on development draws; evaluation uses distinct seeds and 8192 fresh trials per cell. Common innovations across arms preserve each arm's correct noise marginal law and support paired contrasts.

The conditional-noise validation matches the derived identity: expected excess $0.0149624699$, sampled mean $0.0150152030$, Monte Carlo standard error $0.0000520998$, using 100000 fresh later-noise draws. This is a moment sanity check, not sampler privacy certification.

All six selected adaptive and static configurations have equal second-round axes and identical public settings. Reading the saved NPZ confirmed bitwise-identical adaptive/static trial-loss vectors in every cell, including after fitted shrinkage. Hence the selected mechanisms have no effective profile adaptation. All saved paired 95% normal intervals for adaptive minus one-release loss, both primary and separately shrunk contrasts, are positive. This supports the stated bounded negative utility conclusion; it does not prove that every adaptive configuration or every protected-history constructor fails.

Radii above $0.8$ are permitted in this spike: the first signal saturates at its original magnitude while noise still scales with the declared radius. The code handles this correctly by actual clipping; it does not incorrectly reuse the stationary $\delta=\eta_0/2$ classifier formula for those settings. The actual sampler remains full-dimensional with positive radii and budgets.

The shrinkage control is fitted after the primary configuration is selected, so it is not joint optimization over mechanism and postprocessing; that limitation is explicitly recorded. Report the finite declared search, conditional per-selected-configuration Monte Carlo uncertainty and absence of CIA measurements. No code or result files were changed by this independent check.
