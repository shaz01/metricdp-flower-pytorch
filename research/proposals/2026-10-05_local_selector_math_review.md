# Independent mathematics: permutation-equivariant local route and selector cost

2026-10-05. Conditional reference lemma and development-only cost review. Source checkpoints, projected updates, sample weights and calibration derivatives remain raw/unprotected. No calculation described here certifies the actual head-training transcript or measures CIA. This is not a new confirmation test or estimator implementation.

## 1. Local rule and conditional privacy lemma

Freeze a class-count rule: choose the dominant class when purity is at least $0.6$, otherwise preferred profile 0. Resolve class ties with a deterministic public class-index rule. The frozen dominant-class map is $[0,2,0,1]$. Because the same function acts on each client's own counts rather than slot ID, permuting clients, associated weights and histories permutes their preferred profiles. That is permutation equivariance, not a privacy guarantee or invariance to relabeling classes.

Use profile alphabet $\mathcal K=\{0,1,2\}$. For preferred profile $a(D)$, k-ary randomized response at budget $\xi\ge0$ gives

$$
p_D(k)=\begin{cases}
e^\xi/(e^\xi+2),&k=a(D),\\
1/(e^\xi+2),&k\ne a(D).
\end{cases}
$$

Every categorical probability ratio is at most $e^\xi$, regardless of sensitivity or discontinuity of the raw purity/count statistic. The counts and preferred label must not be additionally released unprotected. On empty input set preferred profile 0 and raw update zero, but still apply the **same RR channel**, clipping and fresh noise procedure. A deterministic override of the selected output profile for empty input would not be this channel.

Condition on fixed public model history, clipping bank, radius, postprocessing and weights. For every selected profile $k$, clip the entire three-coordinate raw bias-contrast update into its public weighted-L1 body and independently add coordinate Laplace noise of scales $2r_{kj}/\eta$, where $\eta=E-\xi>0$. Two arbitrary clipped centers differ by at most two in that body norm, so the conditional upload density ratio is at most $e^\eta$.

The hypothetical joint selected-label/upload release satisfies

$$
\frac{p_D(k)q_k(y-v_k(D))}{p_{D'}(k)q_k(y-v_k(D'))}
\le e^\xi e^\eta=e^E.
$$

Hiding the selected label is postprocessing. Fixed common scalar shrinkage also preserves the bound if it scales the whole upload, mean and noise. Stochastic training/selection is permissible under a uniform clipped-support conditional certificate, not an unsupported independence assumption about raw training state. Fresh protecting noise must retain its declared conditional kernel.

This proves a conditional whole-input replacement reference, including empty input. To claim a complete iterative protocol, prior model history must itself have accounted target privacy, and every future conditional release and metadata channel must be covered. An unprotected other-parameter complement cannot be sent. A radius/shrink/weight chosen from current raw private data is not made public merely by placing it in a configuration dictionary. The current cost audit assumes those parameters for counterfactual comparisons and does not supply their calibration privacy.

## 2. Exact moments of independent profile choices

Fix the saved raw contributions $u_i$, weights $\alpha_i$, bank and global development derivatives. Write $v_{ik}$ for the full-vector clipped contribution under profile $k$, and let clients select independently with probabilities $p_{ik}$. Define

$$
\mu_i=\sum_kp_{ik}v_{ik},\qquad
S_i=\sum_kp_{ik}v_{ik}v_{ik}^\top-\mu_i\mu_i^\top.
$$

For zero-mean conditional upload noise, $\operatorname{Cov}(Y_i)=S_i+\mathbb E_k\operatorname{Cov}(Z_i\mid k)$. Thus

$$
\mu=\sum_i\alpha_i\mu_i,\quad
V_{\mathrm{select}}=\sum_i\alpha_i^2S_i,\quad
V_{\mathrm{upload}}=\frac8{(E-\xi)^2}
\sum_i\alpha_i^2\sum_kp_{ik}\operatorname{diag}(r_k^2).
$$

The selector changes both the aggregate mean and its random dispersion. Omitting $S_i$, or treating the expected selected clipped vector as deterministic, understates cost. Conditional noise/mean covariance is zero because each profile's fresh noise is centered. These formulas require independent selector coins across clients and fresh independent conditional noise; they do not cover arbitrary correlated selector or noise mechanisms.

For a common postprocessing factor $t$, expected global quadratic intervention change is exactly

$$
g^\top\mu\,t+
\left[\tfrac12\mu^\top H\mu
+\tfrac12\operatorname{tr}(HV_{\mathrm{select}})
+\tfrac12\operatorname{tr}(HV_{\mathrm{upload}})\right]t^2.
$$

The same clipped-mean covariance logic applies to any independently sampled public mixture. An unaccounted deterministic local route instead has $S_i=0$, its own deterministic mean, and upload budget $E$; that is a diagnostic reference, not a valid way to avoid the private-selection cost.

For PSD $H$ the squared coefficient is nonnegative, and the common shrink optimum is the existing clipped scalar quadratic minimizer. Optimize the common radius and $t$ at each prescribed allocation $\xi/E\in\{0,0.05,0.125,0.25,0.5\}$, with all configuration selection limited to development information. This produces bounded proxy costs, not direct nonlinear expected CE or held-out confirmation evidence.

## 3. Public mixture and shared controls

At $\xi=0$, each of the first three profiles has probability $1/3$ for every input. The preferred label has no effect, so the mechanism is a public IID profile mixture with the full $E$ upload budget. The public comparison family must explicitly include $(1/3,1/3,1/3,0)$: it is **not on a step-0.1 simplex grid**. Adding this special point ensures the public best cannot be worse than the zero-selection-budget route with identical radius/shrink candidates. Deterministic vertices must likewise include every shared fixed profile.

For a general public common mixture $\pi$ over all four profiles, substitute $p_{ik}=\pi_k$ in the same moments. Even though the mixture is data-independent, random profile choice still produces dispersion in clipped means. Its private input is protected by the uniform fixed-profile upload bound; profile disclosure adds no input privacy cost. In the actual development audit, choosing $\pi$, radius or shrink from raw local/global derivatives is still unaccounted calibration, so the numerical arm must not be called an already public deployed mechanism.

The public simplex grid is a bounded control family, not an asserted continuous optimum. The global quadratic objective as a function of a shared mixture can include cross-client terms; convexity of the overall mixture-parameter problem should not be assumed merely because the loss Hessian is PSD. Keep shared deterministic, public-mixture and local-rule comparisons distinct, and give all the same derivative conventions and radius/shrink controls.

## 4. Interpretation and next decision

The dominant-label mapping is frozen from the previous stress-related construction. It supplies a concrete equivariant hypothesis, but does not establish that labels are the right statistic for natural client geometry, or that independent local routes reproduce the globally coupled assignment oracle. Purity thresholding can be brittle or uninformative outside the constructed stress partition. Reproducing the old label-slot association through a data-based label rule is a useful diagnostic, not a novel private distribution estimator.

Use all three seeds/regimes and two checkpoints, yielding 18 rows and 54 epsilon cells. This is development-only reuse: do not treat another optimized score on these same derivatives as fresh success at the prior $0.001$ held-out gate. Separate favorable mean effects from selector dispersion, increased upload noise and the effect of public shrinkage. Exact moment integration avoids sampling error for the quadratic proxy, but not approximation error, data uncertainty or finite-family selection bias.

An accounted conditional construction can be mathematically valid while losing utility to shared or data-independent mixture controls. Advance to implementation only if the remaining headroom after these explicit costs supports a separately specified, full observer protocol; do not use the lemma alone as evidence that the real raw-head pipeline is protected. No code/results/jobs were modified by this design review.

## Actual calculator review

Independently inspected `research/calculations/local_selector_cost.py` after it appeared. No material implementation error was found. The three-profile RR channel, empty/low-purity preferred route zero, deterministic label ties and frozen map match the design. Its fourth profile probability is zero for the local selector, while all four profiles are retained for stronger public controls.

The analytic contractions correctly compute weighted clipped means, full-Hessian mean curvature, selector dispersion and expected upload-noise cost. All covariance terms use squared aggregation weights, and shrinkage scales every quadratic term. Public deterministic vertices have zero selector dispersion, while random public mixtures include dispersion. The 287-mixture family contains all 286 step-0.1 simplex points plus the explicit first-three-profile uniform point; control-inclusion assertions therefore legitimately cover shared choices and the $\xi=0$ route.

The optimizer distinguishes an optimistically development-calibrated common radius/shrink from a fixed $C=0.3,t=1$ diagnostic. It records separate linear, mean-quadratic, selection-variance and upload-noise contributions after shrinkage. The selected allocation is chosen from a grid that includes zero: the output field `best_paid_fraction` may therefore equal zero and should be described as the **best selector-grid allocation**, not necessarily a positive paid selector. This naming issue does not affect the calculation.

The self-check reconstructs the expected objective by explicit $3^8$ independent selector enumeration with nonuniform client weights and agrees with the analytic expression. It also checks permutation of updates/weights/probabilities, RR categorical likelihood ratios and the empty/tie route. Full-run assertions additionally permute updates, weights and class counts together. These support the exact finite moment calculation, not full raw-training privacy.

Source loading uses only the offline training cache, saved client counts/means, checkpoints and global development examples. No test examples are used in this cost calculation. The global derivative and calibrated bank/budget choices remain unprotected, which is correctly stated in its limitations. The implementing agent reports a successful self-check and a running full calculation; results were not yet assessed at this code review. No calculator/results/jobs were changed by the independent reviewer.

## Independent completed cost-artifact check

The saved development artifact contains 18 rows and 54 epsilon cells. Independently read all cells and verified that every logged score equals the sum of its post-shrink linear, mean-quadratic, selector-variance and upload-noise terms. No new simulation, training or test evaluation was run.

The optimized public-mixture score equals the optimized shared score in all 54 cells. The selector-grid best beats those controls in 12 label-stress cells at labels 8/16 and loses in the other 42 cells; there are no ties. Its largest development-quadratic gain is $0.0008498411$, and zero cells pass the stated $0.001$ proxy headroom gate. This gate uses the reused development approximation, not the earlier direct held-out CE endpoint.

Every stress cell selects fraction $\xi/E=0.25$ on the prescribed grid, including the label-4 stress cells that still lose. At label 8 this means selector cost two, upload cost six, correct-route probability $0.7869860$ and fixed-profile noise inflation $16/9\approx1.7778$. At label 16 the correct-route probability rises to $0.9646632$ with selector cost four and upload cost twelve, while the same fixed-fraction noise inflation remains.

For seed 43, label stress, checkpoint 20 and total label 8, the shared predicted change is $-0.0007150051$. The optimistically calibrated selector predicts $-0.0010447670$, a gain of $0.0003297620$, versus the unaccounted deterministic route's $-0.0017716478$. The paid selector score includes selector-dispersion penalty $0.0000425708$ and upload-noise penalty $0.0005978186$. Radius $0.3$ and shrink one are selected, so the fixed-radius/shrink diagnostic coincides with the optimistic arm in that cell. This directly shows why the free route's apparent headroom should not be carried over as a paid-construction benefit.

The result preserves two distinct conclusions. The conditional reference lemma is valid under its specified public-history/configuration assumptions, and the frozen rule is permutation equivariant. Yet the accounted selector/upload costs in this exact development proxy leave only sub-threshold advantage in part of the disclosed stress regime. This is evidence about a bounded rule/bank/cost family, not proof that every private estimator fails or that this one protects the real raw model trajectory.

Choosing radius, shrink or selector budget from the same raw development derivatives remains unaccounted. A reported total label $E$ in this artifact must not be presented as a certified privacy budget for those computations, private sample weights, underlying checkpoints or unnoised model complements. The deterministic-local route is an unaccounted diagnostic; the label-0 selector-grid point is a public mixture, not positive-cost personalization.

The completed task supports documenting the conditional construction and its narrow cost limits, rather than claiming a successful CIA defense or proceeding directly to an estimator implementation. Any further mechanism must specify an accounted whole-history protocol and demonstrate surviving headroom against strong public controls under a separately declared assessment. This review edited only this note and launched no new jobs.
