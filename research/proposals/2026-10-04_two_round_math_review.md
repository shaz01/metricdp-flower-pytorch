# Independent mathematics: two-round protected-history construction

2026-10-04. Own closed-form derivation, not FL/CIA evidence or implementation. The toy uses a **constant** true update across two rounds; it does not describe evolving local-training updates. Public population parameters are assumed known for analytical controls.

## 1. Probe law, profile decision and domains

Let $u_i=s e_i$, $i\in\{1,2\}$, equiprobably, with $s=0.8$. Take a public probe diamond with equal semiaxes $0<a_0\le s$ and budget $0<\eta_0<E$. Radial clipping gives $v_0=a_0e_i$. The replacement-calibrated diamond K-norm noise has independent coordinates

$$
Z_{0j}\sim\operatorname{Laplace}(0,b),\qquad b=2a_0/\eta_0.
$$

Thus $Y_0=a_0e_i+Z_0$. Choose the profile whose long axis is along the largest absolute coordinate of $Y_0$. Ties have probability zero for this continuous positive-scale law; fix a public tie rule regardless. Put $\delta=a_0/b=\eta_0/2$.

For type 1 the wrong-selection event is $W=\{|a_0+Z_{01}|<|Z_{02}|\}$. Since $\Pr(|Z_{02}|>x)=e^{-x/b}$ for $x\ge0$,

$$
q=\Pr(W)
=\frac12\int_{\mathbb R}e^{-|t|-|\delta+t|}\,dt
=\frac12(1+\delta)e^{-\delta},\qquad p=1-q.
$$

This decision is postprocessing of a protected probe, **not binary RR**. Its $p,q$ may be used in this utility population calculation; they are not its general privacy certificate.

Boundary care is necessary. At exactly $a_0=0$ the probe is deterministic zero and a public tie rule yields an input-independent profile, so the displayed $q(\eta_0)$ is not its decision error. The limit $a_0\to0^+$ retains informative *normalized* measurements and must not be equated with that zero-radius protocol. At $\eta_0=0$ the positive-radius Laplace scale is not finite. The later budget $\eta_1=E-\eta_0$ must also be positive. Optimize within the positive domain, or separately define and assess boundary mechanisms. Arbitrarily tiny probes can additionally create numerical/finite-precision problems not represented in this ideal real-valued derivation.

## 2. Exact selection-dependent cross moment

The active probe noise is correlated with the selected later clipping bias. The needed moment is

$$
m=\mathbb E[Z_{01}\mathbf1_W]
=\frac b2\int_{\mathbb R}t e^{-|t|-|\delta+t|}\,dt
=-\frac{b\delta(1+\delta)e^{-\delta}}4
=-\frac{a_0q}{2}.
$$

To check the integral, split at $t=-\delta$ and $t=0$. The three integrals before multiplication by $b/2$ are

$$
e^{-\delta}(-\delta/2-1/4),\quad
-e^{-\delta}\delta^2/2,\quad e^{-\delta}/4.
$$

The inactive coordinate has $\mathbb E[Z_{02}\mathbf1_W]=0$ by sign symmetry. Also $\mathbb E[Z_{01}\mathbf1_{W^c}]=a_0q/2$ because the unconditional probe noise is centered. These are conditional-selection effects, even though the probe's unconditional noise mean is zero.

## 3. Exact final averaged-update risk

At round 1, the chosen public diamond has semiaxes $L,S>0$ along the selected and other coordinates, respectively. Initially allow either ordering; if the protocol requires a “long” axis, impose $L\ge S$ explicitly. For $L,S\le s$, type 1 has active round-1 clipping bias $L-s$ after correct selection and $S-s$ after wrong selection. Fresh later noise has covariance $8\operatorname{diag}(L^2,S^2)/\eta_1^2$, with coordinates swapped for the other profile.

Write $e_0=Y_0-u$ and $e_1=Y_1-u$, and consider the output $(Y_0+Y_1)/2$. Define

$$
R_0=(a_0-s)^2+16a_0^2/\eta_0^2,
$$

$$
R_1=p(s-L)^2+q(s-S)^2+8(L^2+S^2)/\eta_1^2.
$$

The exact cross term is

$$
\mathbb E[e_0^\top e_1]
=(a_0-s)[p(L-s)+q(S-s)]+(S-L)m
=(a_0-s)[p(L-s)+q(S-s)]+\frac{a_0q(L-S)}2.
$$

Fresh round-1 noise has zero cross moment with the entire past, conditional on that past; it is the **round-1 clipping bias** that produces the additional selection correlation. Therefore

$$
R_{\mathrm{adapt}}=\frac14\left[R_0+R_1+2\mathbb E[e_0^\top e_1]\right],
$$

or, equivalently,

$$
4R_{\mathrm{adapt}}
=\frac{16a_0^2}{\eta_0^2}
+p(L+a_0-2s)^2+q(S+a_0-2s)^2
+\frac8{\eta_1^2}(L^2+S^2)+a_0q(L-S).
$$

For $L\ge S$, the last selection-correlation contribution is nonnegative. Dropping it makes this adaptive risk look artificially favorable. The two constant-direction types have the same risk by symmetry; the formulas already average over them.

## 4. Geometry optimization is not the single-round optimizer

For fixed $\eta_0,a_0$, put $t=8/\eta_1^2$. Without an ordering constraint, the unconstrained minima of the averaged-update risk are

$$
L_0=\frac{p(2s-a_0)-a_0q/2}{p+t},\qquad
S_0=\frac{q(2s-a_0/2)}{q+t}.
$$

These differ from optimizing the round-1 error alone. Both are positive, but may exceed $s$. A semiaxis above $s$ leaves the corresponding clipped toy update equal to $s$ and increases noise, so it never improves this risk. Under independent bounds $0<L,S\le s$, cap each optimizer at $s$.

The uncapped optimizers can have $L_0<S_0$: probe-noise correlation can favor stronger correction in the nominally wrong-selected direction. If the mechanism insists on $L\ge S$, solve that convex constrained problem rather than relabeling the axes after optimization. When the unconstrained solution violates order, the optimum lies on the pooled boundary

$$
L=S=\min\left(s,\frac{2s-a_0}{1+2t}\right).
$$

In that case the two profile laws coincide; no adaptive geometry remains.

For joint optimization of $a_0,L,S$ at fixed budgets, the expanded quadratic has half-Hessian

$$
M=\begin{pmatrix}
1+16/\eta_0^2&(1+p)/2&q/2\\
(1+p)/2&p+t&0\\
q/2&0&q+t
\end{pmatrix},\qquad
Mx=\begin{pmatrix}2s\\2sp\\2sq\end{pmatrix}
$$

for an interior stationary point $x=(a_0,L,S)^\top$. It is a convex quadratic because it is the expected squared error of affine scaled probe/noise variables, with strictly positive noise penalties. Account for the box constraints and any ordering constraint via their active faces; an unconstrained solve alone is not a global constrained solution. In particular, a boundary value $a_0=0$ cannot silently inherit the positive-probe classification law. Optimizing $\eta_0$ is a further one-dimensional task; a bounded grid is not a continuous global-optimality certificate.

## 5. Same-probe public comparator

A valid fixed-profile baseline receives exactly the same $Y_0$, with the same $a_0,\eta_0$, and uses a public isotropic diamond of semiaxis $a$ at round 1 with budget $\eta_1$. Its final averaged-update risk is

$$
R_{\mathrm{pub}}(a)
=\frac14(a_0+a-2s)^2
+\frac{4a_0^2}{\eta_0^2}+\frac{4a^2}{\eta_1^2}.
$$

At fixed probe, its optimum over $0<a\le s$ is

$$
a^*=\min\left(s,\frac{2s-a_0}{1+16/\eta_1^2}\right).
$$

This is exactly the adaptive pooled-profile optimum. Optimizing the probe jointly for the adaptive mechanism while freezing an arbitrary probe for the public control is insufficient: report both same-probe comparisons and optimized public allocation/geometry comparisons under the same two-round protocol. A public control could also optimize how to combine the two uploads or compare a different density/body family; those are stronger competitors, not covered by “best public” here.

## 6. Privacy and limits

The probe is $\eta_0$-DP for every whole-client input after clipping into its public diamond. Profile selection $K=f(Y_0)$ uses no additional raw private information. Conditional on every realized $Y_0$ and profile, the round-1 mechanism clips an arbitrary new raw update into the specified public body and adds fresh noise with a uniform replacement certificate $\eta_1$. Thus the hypothetical transcript $(Y_0,Y_1)$ is $(\eta_0+\eta_1)=E$-DP. The profile and final average are postprocessing of this transcript, so no additional RR cost is charged. The probe cost remains fully included.

This guarantee requires every conditional law to be valid for every input/history and forbids an unprotected private bank, raw summary, seed or diagnostic in the observer's view. It does not rely on $q$ being identical across arbitrary client datasets. If future training randomness depends on the selection history, the pointwise uniform clipped-support argument still applies with fresh protecting noise. Formal composition is separate from the constant-update utility model used above.

For actual FL, two uploads are not generally averaged around the same true update: model histories, learning rates, objectives and local updates change. A positive result under this proxy would therefore motivate further research, not establish model accuracy or CIA mitigation. Match the complete observer, initialization, weights and noise randomness before transferring any comparison to the repository.

## 7. IID-client aggregation proxy

Suppose there are $N$ independent clients drawn from this equiprobable two-type population, with independent protecting noise and the same public configuration. For each client define $e=(Y_0+Y_1)/2-u$ and let $R_{\mathrm{local}}=\mathbb E\|e\|^2$ be the exact risk above. Its mean active error conditional on either type is

$$
\beta=\frac{a_0+pL+qS-2s}{2}.
$$

Thus $\mathbb E[e\mid i]=\beta e_i$, $\mathbb E e=(\beta/2,\beta/2)$, and $\|\mathbb E e\|^2=\beta^2/2$. For error of the average relative to the average true client update,

$$
R_{\mathrm{global}}
=\mathbb E\left\|\frac1N\sum_i e_i\right\|^2
=\frac{R_{\mathrm{local}}}{N}
+\frac{N-1}{N}\frac{\beta^2}{2}.
$$

For the public isotropic second-round profile substitute $pL+qS=a$. This formula is correct for IID client types and independent per-client errors. It is not the expression for a predetermined exactly balanced roster without adjustment, nor for correlated clients/updates under a shared adaptive learning process. Specify the population and averaging target when reporting it.

The bias term survives as $N$ grows; a geometry minimizing local risk need not minimize this global proxy. At fixed budgets and probe, $R_{\mathrm{global}}$ remains a convex quadratic in the semiaxes: the local risk is convex and the added squared affine bias has a nonnegative coefficient. Reoptimize for each $N$ and enforce the same box/order constraints, instead of transferring the local optimum. Public global controls must receive the same probe and also get optimized geometry/allocation.

A gain against a mandated two-round fixed-average protocol does not imply a gain against the strongest public use of the total budget. For example, a one-release optimized public isotropic diamond at budget $E$ has local risk $16s^2/(E^2+16)$; with $s=0.8,E=8$, this is $0.128$. This has a different release schedule, so retain it as an explicit diagnostic rather than a supposedly matched two-round comparison. Its own IID global risk follows the same aggregation formula using bias $a-s$ and its one-release local risk. In particular, a proposed two-round local risk near $0.189$ cannot be advertised as better public total-budget reconstruction when this allowed one-release control is about $0.128$.

No novel privacy primitive follows from reusing a protected probe. The research question is whether accounted history-dependent construction improves a specified learning protocol after public schedule/body/density controls and surviving aggregate clipping bias are considered.

## 8. Review of the actual calculator and its explicit boundary

Reviewed `research/calculations/protected_history_two_round.py` on 2026-10-04. Its local quadratic coefficients, global bias addition, same-probe optimization, one-release diagnostic and ordered box minimization agree with the derivations above. The order constraint is handled correctly: if the box optimum violates $L\ge S$, the convex constrained optimum lies on the $L=S$ face, which the reduced solve enumerates. No code was modified by this review.

The calculator's $\eta_0=E,\eta_1=0$ endpoint is explicitly **nonadaptive**: the second upload is deterministic zero, not a zero-budget positive-radius Laplace mechanism. For a positive first radius $a\le s$, this averaged estimator has IID federation risk

$$
R_{\mathrm{zero\ second}}(a)
=w(a/2-s)^2+\frac{4a^2}{NE^2},\qquad
w=\frac{N+1}{2N},
$$

and optimum

$$
a^*=\min\left(s,\frac{2ws}{w+16/(NE^2)}\right).
$$

These match the code. The deterministic second message is input-independent and costs zero privacy; the first upload costs $E$. It still participates in the fixed average, so this boundary differs from directly using the single positive upload as the estimator. That distinction explains its larger clipping bias. No $q$ is assigned at this nonadaptive endpoint. The symmetric zero-first/positive-second public-isotropic boundary gives the same proxy risk under stationary updates; it supplies no protected-history selector.

The interior budget grid excludes both zero budgets. Geometry enumeration technically includes $a_0=0$ as a box face, but the code rejects a newly winning zero-probe optimizer by assertion; it does not report that invalid classifier case as an interior adaptive result. Preserve this check or define such cases as separate protocols. Any exactly zero later-body axis would likewise need an explicit lower-dimensional kernel/support definition rather than evaluating a full-dimensional density with zero volume.

The Monte Carlo check separately samples the positive-scale probe and compares both $q$ and the cross moment to their closed forms, with stated sampling standard errors. It is a useful probability/moment check, not a privacy certificate. The direct quadratic checks also cover the IID aggregate bias formula and recovery of the public law when the two profile radii coincide. Budget search remains bounded; do not call its best row a certified continuous global optimum or a strongest-public-mechanism result.
