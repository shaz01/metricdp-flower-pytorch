# Independent check: heterogeneous two-type quadratic-risk calculation

2026-10-04. Own analytic review only; no implementation or experiments. Population parameters used to construct or optimize controls are assumed public. Optimizing them from private confirmation data would require a separate treatment.

## 1. Formulas and domain

The proposed formulas are correct for equiprobable types $u_1=s e_1$, $u_2=s e_2$, $s>0$, with $H_1=\operatorname{diag}(1,h)$, $H_2=\operatorname{diag}(h,1)$ and $0\le h\le1$. The objective is the client-type-specific proxy $\mathbb E[(Y-u_i)^\top H_i(Y-u_i)]$. It is not the loss of an aggregated global FL update.

A centered public ellipsoid with semiaxes $a,b>0$ uses radial clipping into that body and replacement-calibrated K-norm noise. In two dimensions, its covariance is $12\operatorname{diag}(a^2,b^2)/\epsilon^2$. Write $(x)_+=\max(x,0)$. Averaging clipping and noise gives

$$
R_{\mathrm{pub}}(a,b)=\tfrac12[(s-a)_+^2+(s-b)_+^2]
+\frac{6(1+h)}{\epsilon^2}(a^2+b^2).
$$

The objective separates across axes. A semiaxis above $s$ never improves bias and increases noise, so the global positive-semiaxis optimum is

$$
a^*=b^*=\frac{s\epsilon^2}{\epsilon^2+12(1+h)},\qquad
R_{\mathrm{pub}}^*=\frac{s^2\,12(1+h)}{\epsilon^2+12(1+h)}.
$$

This is global optimality **within this centered ellipsoid/radial-clipping/K-norm family**, not across every public mechanism.

For binary RR, let $0\le\xi<\epsilon$, $\eta=\epsilon-\xi>0$, $p=e^\xi/(1+e^\xi)$, $q=1-p$, and $c=12/\eta^2$. A preferred profile has semiaxes $(L,S)$ along the useful and other axes respectively; the other profile swaps them. The stated private risk is

$$
R_{\mathrm{priv}}(L,S)=p(s-L)_+^2+q(s-S)_+^2
+c[(p+qh)L^2+(ph+q)S^2].
$$

The finite-budget optimum is interior and is exactly

$$
L^*=\frac{ps}{p+c(p+qh)},\qquad
S^*=\frac{qs}{q+c(ph+q)}.
$$

Both lie strictly between zero and $s$; $L^*\ge S^*$ for $\xi\ge0,h\ge0$. A useful closed form for the optimized risk is

$$
R_{\mathrm{priv}}^*=s^2\left[
1-\frac{p^2}{p+c(p+qh)}
-\frac{q^2}{q+c(ph+q)}\right].
$$

Equivalently it is $s^2\{pc(p+qh)/[p+c(p+qh)]+qc(ph+q)/[q+c(ph+q)]\}$, which avoids subtraction of nearly equal quantities at low risk. These optimize $L,S$ at a **fixed** selection allocation; optimization over $\xi$ remains a distinct one-dimensional comparison.

## 2. Certificate and interpretation

Each public profile defines a symmetric body $B_k$; clipping confines every arbitrary client input's output to $B_k$. Its replacement-sensitive noise density is proportional to $\exp[-\eta\|z\|_{B_k}/2]$. Every profile is $\eta$-DP for every input, including nonpreferred selections. Binary RR applied to an arbitrary private preferred label is $\xi$-DP without a sensitivity calculation for the raw utility statistic. The hypothetical joint label/update release is $(\xi+\eta)$-DP; hiding the label preserves that bound. See the full uniform conditional argument in [candidate_math_review.md, §8](2026-10-04_candidate_math_review.md).

The two types are a **utility population**, not a restriction of adjacency to those two points: the certificate still needs arbitrary whole-client inputs, including empty input. A private $H_i$ may be used internally to compute a label, but must not be released unaccounted. Optimizing the public bank from this toy's known population is an oracle public control; empirical tuning must specify its information source.

At $\xi=0$, $p=q=1/2$, $\eta=\epsilon$ and $L^*=S^*=s\epsilon^2/[\epsilon^2+12(1+h)]$: the private construction reduces exactly to the optimized public isotropic law.

At $h=0$, both profiles optimize to $L^*=S^*=s\eta^2/(\eta^2+12)$, independently of $p$. The risk is $12s^2/(\eta^2+12)$, so any positive selection expenditure strictly loses against the optimized public risk $12s^2/(\epsilon^2+12)$. With no penalty for error on the other axis, there is no advantage from orienting the noise profile under this proxy.

At $h=1$, the two curvature matrices are identical, but the optimum can still be asymmetric when $\xi>0$. The selector matches geometry to differing **signal directions**. Hence any improvement in this case must not be attributed to heterogeneous curvature. It is also incorrect to rule out improvement for every budget: for example, along $\epsilon\to\infty$ with $\xi=3\log\epsilon$, $q=O(\epsilon^{-3})$ and $\eta/\epsilon\to1$, the optimized private ellipsoid risk is asymptotically $12s^2/\epsilon^2$, versus the optimized public ellipsoid's $24s^2/\epsilon^2$. This is a mathematical weak-privacy asymptotic, not a useful strong-protection result or an FL prediction.

For aggregation, client-local $H_i$ weighted errors do not automatically sum to the global model's error. A fixed global curvature $H$ would act on the average noise covariance, while clipping biases can cancel or reinforce; independent noise variances also receive squared aggregation weights. Thus this proxy diagnoses accounted local adaptation, but does not demonstrate an improved CIA/accuracy frontier.

## 3. Rotated public ellipsoids do not improve this particular control

Rotation is an important control in general, but it cannot improve the optimum here. Write a centered ellipsoid as $B=\{x:x^\top Mx\le1\}$ with $M\succ0$. Under radial clipping, its effective semiaxes for the two axis inputs are $t_1=1/\sqrt{M_{11}}$ and $t_2=1/\sqrt{M_{22}}$. The mean squared clipping biases depend only on those two diagonal entries. Its noise covariance is $12M^{-1}/\epsilon^2$.

Since $\mathbb E H_i=(1+h)I/2$, its average noise penalty depends on $\operatorname{tr}(M^{-1})$. For a two-dimensional symmetric matrix,

$$
\operatorname{tr}(M^{-1})
=\frac{M_{11}+M_{22}}{M_{11}M_{22}-M_{12}^2}
\ge\frac1{M_{11}}+\frac1{M_{22}}.
$$

Replacing $M$ by its diagonal preserves clipping for both toy inputs and weakly reduces average noise. Therefore the axis-aligned optimum already covers **all centered rotated ellipsoids with this radial clipping/noise protocol**. This proof uses the symmetric population and its averaged curvature; it does not justify omitting rotated controls in a general client population. Off-center bodies, other clipping maps and other density families are outside it.

## 4. A stronger public law control: the diamond

There is a simple nonellipsoidal control that materially improves the comparison. Use the centered weighted $\ell_1$ body

$$
B_{a,b}^{\diamond}=\{z:|z_1|/a+|z_2|/b\le1\}.
$$

Radial clipping gives exactly the same clipped outputs for the toy axis inputs as the corresponding ellipse. A uniform point in this diamond has covariance $\operatorname{diag}(a^2,b^2)/6$. Multiplying it by an independent $\operatorname{Gamma}(3,2/\epsilon)$ radius gives replacement-calibrated K-norm covariance $8\operatorname{diag}(a^2,b^2)/\epsilon^2$, rather than the ellipse's coefficient $12$. The generic convex-body certificate already covers this law; the smaller covariance does not evade the clipping/sensitivity argument.

The diamond public optimum is consequently

$$
a^*=b^*=\frac{s\epsilon^2}{\epsilon^2+8(1+h)},\qquad
R_{\mathrm{pub},\diamond}^*=\frac{s^2\,8(1+h)}{\epsilon^2+8(1+h)},
$$

strictly below the ellipsoid public optimum for finite positive budgets. Thus “beats optimized public geometry” must specify the law/body family: defeating the public ellipsoid control does not establish a gain over this equally certified diamond control. Private diamond profiles are also a relevant control: on these toy inputs the same risk/optimizer formulas apply with coefficient $12$ replaced by $8$. They retain the whole-input certificate and selection cost. This is existing K-norm body design, not a novel private mechanism.

The uniform-diamond moment can be checked directly in one quadrant: its area is $ab/2$, and averaging $x_1^2$ over the triangle gives $a^2/6$; symmetry removes cross moments. The Gamma second moment is $3\cdot4\cdot(2/\epsilon)^2=48/\epsilon^2$, yielding coefficient eight. No simulation or implementation is needed for this comparison.

## Bounded conclusion

Accept all proposed ellipsoid optimizer and risk formulas. The analysis exposes when paid private selection has a possible local benefit, but neither curvature heterogeneity nor FL improvement follows automatically. Add the optimized public diamond and, if pursuing positive results, its private counterpart as stronger body-family controls. For the current symmetric toy, rotated centered ellipsoids have been analytically ruled out as improvements, so numerical rotation sweeps add no evidence. Keep total selection/update budget, clipping distortion, public tuning assumptions and the whole-input observer certificate explicit.

## Amplitude benchmark scope correction

The separate equiprobable population (0,e1) uses circles or equal-axis diamonds in the tested profile bank, so its stated public optimum is only over isotropic radii. Public projection onto e1 followed by scalar clipping at a and Laplace noise of scale 2a/E has risk .5(1-a)^2+8a²/E². Minimizing gives a=E²/(E²+16), risk8/(E²+16). Releasing a public deterministic zero in coordinate2 preserves privacy for arbitrary inputs but discards that signal; the toy population does not penalize this discard. This stronger public control reinforces the negative result.
