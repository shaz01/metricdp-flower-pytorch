# Independent mathematical review of the two proposed noise constructions

Date: 2026-10-04. Scope: review of the parent's proposed public profile bank and randomized-response private profile selector. This is a mathematical research note, not an implementation, experiment result or novelty assessment. All derivations below are our own unless a source is identified.

**Finding:** both constructions admit the stated pure-DP certificate with explicit uniformity and transcript conditions. However, the proposed ellipsoidal bank is dominated by its isotropic member for every positive-semidefinite quadratic noise cost when every profile must contain the same full Euclidean sensitivity ball. Paying for private selection compounds that disadvantage. Treat this bank as a correctness/control construction; revise the utility mechanism before presenting it as the leading improvement hypothesis.

## 1. Input, adjacency and conditional release

Fix a public enrolled slot, dimension $d$, history $h$, Euclidean clipping radius $C>0$, finite public bank size $m\ge2$, and positive privacy budgets. Here $m$ denotes the number of profiles; $K_k$ denotes a convex body. Let the entire private client input be $D$, with an arbitrary local-training map $u(D;h)$ satisfying $\|u(D;h)\|_2\le C$ after clipping. Whole-client replacement allows any two private inputs in that same enrolled slot. Consequently $\|u(D;h)-u(D';h)\|_2\le2C$ without a record-sensitivity calculation. This is a strong conditional local privacy guarantee against the observer of that upload.

The fixed slot must report in both worlds, including an empty-dataset world if emptiness is in the input domain. A missing transport message, private sample count, raw selector statistic or timing branch is outside the displayed release law and can defeat the claimed full-transcript guarantee. This certificate protects the hidden private contribution in an enrolled slot; it does not hide public enrollment or observable participation.

An empty input mapped to zero has a direct nonempty-to-empty difference bound $C$. Using that smaller bound globally is invalid if arbitrary nonempty-to-nonempty replacement is also adjacency. Use $2C$ for the proposed all-pairs whole-input guarantee. Local-training randomness is allowable: the fixed-shift certificate below holds uniformly for every pair of clipped outputs, so integrating their randomized laws preserves the bound. Fresh protecting randomness must remain unknown to the observer.

## 2. Candidate A: density, normalization, sampler and certificate

For each public positive-definite shape matrix $A_k$, put

$$
R_k=\frac{2C}{\sqrt{\lambda_{\min}(A_k)}},\qquad
K_k=R_k A_k^{1/2}B_2^d,\qquad
\|z\|_{K_k}=\frac{\sqrt{z^\top A_k^{-1}z}}{R_k}.
$$

Every Euclidean shift with norm at most $2C$ lies in $K_k$. With $\eta=\epsilon_{\mathrm{update}}$, the normalized density is

$$
q_k(z)=\frac{\eta^d}{\Gamma(d+1)\operatorname{Vol}(K_k)}
\exp[-\eta\|z\|_{K_k}],\qquad
\operatorname{Vol}(K_k)=R_k^d\sqrt{\det A_k}\operatorname{Vol}(B_2^d).
$$

$\Gamma(d+1)=d!$; the parent's normalization is correct. Independently draw $r\sim\operatorname{Gamma}(d+1,\text{scale}=1/\eta)$ and $v\sim\operatorname{Unif}(B_2^d)$, then set $Z=rR_kA_k^{1/2}v$ and $Y=u(D;h)+Z$. The Gamma shape is **$d+1$**, not $d$, for this uniform-body representation. A different representation using a uniform sphere direction has Gamma shape $d$ for the gauge radius. Mixing those representations miscalibrates the law.

For $\delta=u(D;h)-u(D';h)$, the triangle inequality gives

$$
\frac{q_k(y-u(D;h))}{q_k(y-u(D';h))}
\le\exp[\eta\|\delta\|_{K_k}]\le e^\eta.
$$

Integrating proves pure $\eta$-DP. The bank must be public, and the chosen profile must be independent of current raw input for Candidate A, or selected only as a function of already released protected history. If it uses protected history, earlier costs remain in the full trajectory ledger. A uniform bound is required for every compared history; considering just the histories likely under one dataset is insufficient. Public adaptive selection introduces no separate new privacy cost, but does not erase prior release costs.

These are a specialization of existing K-norm constructions, not a new mechanism theorem: [Hardt–Talwar, §4, Definition 4.1/Remark 4.2/Theorem 4.3](https://arxiv.org/abs/0907.3754), with the general-domain extension in §9. [Joseph–Yu, §2.1, Lemmas 7–10](https://proceedings.mlr.press/v247/joseph24a.html) explicitly uses sensitivity sets and the same Gamma/uniform-body sampler.

## 3. Candidate B: arbitrary private statistic to categorical RR

Let $s(D;h)\in\{1,\ldots,m\}$ be any preferred-profile label computed from raw block-gradient energies, with a public tie/empty-input rule. Define

$$
p_D(k)=\begin{cases}
a=e^{\xi}/(e^{\xi}+m-1),&k=s(D;h),\\
b=1/(e^{\xi}+m-1),&k\ne s(D;h),
\end{cases}\qquad \xi=\epsilon_{\mathrm{select}}.
$$

Every categorical probability ratio is at most $e^\xi$. Therefore no sensitivity bound on the raw continuous energy statistic is required for this *categorical channel*. The statement would change if raw energy values, a sensitive bank or unprotected label were released too.

Draw $L\sim p_D$, then $Y=u(D;h)+Z_L$ with fresh conditionally independent $Z_L\sim q_L$. The hypothetical joint release has density

$$
f_D(k,y)=p_D(k)q_k(y-u(D;h)),\qquad
\frac{f_D(k,y)}{f_{D'}(k,y)}\le e^{\xi+\eta}.
$$

Thus $(L,Y)$ is $(\xi+\eta)$-DP, and hidden-label $Y$ is too by marginalization. Hiding $L$ may yield a better exact guarantee, but the additive upper bound does not assume that improvement. All profiles must satisfy the same update certificate for **every** private input, not only for inputs preferring that profile. Otherwise the proof fails precisely on the RR outcomes that choose a nonpreferred law. Correlation between label and raw training randomness can be accommodated by a uniform conditional certificate, not by assuming the label is independent of the private update.

For $T$ adaptive uploads with conditional budgets $(\xi_t,\eta_t)$, the basic pure-DP trajectory bound is $\sum_t(\xi_t+\eta_t)$. Public histories and subsequent server models are postprocessing only if their construction accesses no additional unprotected target data. Private weights, diagnostics and sample counts require separate coverage. There is no automatic amplification just because other clients participate.

## 4. Decisive quadratic-utility objection

Uniform-ball symmetry gives $\operatorname{Cov}(v)=I/(d+2)$, while $\mathbb E[r^2]=(d+1)(d+2)/\eta^2$. Hence

$$
\Sigma_k=\operatorname{Cov}(Z_k)
=\frac{(d+1)R_k^2}{\eta^2}A_k
=\frac{4C^2(d+1)}{\eta^2}\frac{A_k}{\lambda_{\min}(A_k)},
$$

$$
\mathbb E\|Z_k\|_2^2
=\frac{4C^2(d+1)}{\eta^2}
\frac{\operatorname{tr}A_k}{\lambda_{\min}(A_k)}
\ge\frac{4C^2d(d+1)}{\eta^2}.
$$

More strongly, $\Sigma_k\succeq\Sigma_0$ with $\Sigma_0=4C^2(d+1)I/\eta^2$, the isotropic law at the same update budget. Thus for any fixed $H\succeq0$, $\operatorname{tr}(H\Sigma_k)\ge\operatorname{tr}(H\Sigma_0)$. This also applies to the local second-order task-loss approximation $\operatorname{tr}(H\Sigma)/2$. Rotation of the ellipse does not rescue that objective: every directional variance is already no smaller than the isotropic one. This conclusion concerns the proposed common-ball-calibrated ellipsoidal family, not all non-Gaussian or local-DP mechanisms.

For Candidate B, $\mathbb E[Z_L\mid D]=0$ and $\operatorname{Cov}(Z_L\mid D)=\sum_kp_D(k)\Sigma_k\succeq\Sigma_0$. With fixed total budget $E=\xi+\eta$, the isotropic control can instead spend all $E$ on its update. Even an all-isotropic bank pays an energy inflation factor $(E/\eta)^2$; a genuinely anisotropic bank increases it further. A nonlinear task or differently clipped estimator could still behave differently, but that needs a specific justification rather than an expected-quadratic-noise argument.

**Two-dimensional check:** set $C=1$, $\eta=1$, $A=\operatorname{diag}(4,1)$. Then $\Sigma=\operatorname{diag}(48,12)$ and energy is $60$, while the isotropic control has $\Sigma_0=12I$ and energy $24$. At total budget $E=2$, splitting one unit to RR and one to the update makes the isotropic update-only control's energy $6$, before any shape inflation in Candidate B.

This bank also has severe dimension scaling: isotropic energy is $O(C^2d^2/\eta^2)$ and its exact expected norm is $2Cd/\eta$. At $d=2$, expected norm is $4C/\eta$. For millions of coordinates this is an obvious practicality obstacle for moderate per-upload budgets. This is the cost of this pure-DP additive construction under the stated whole-client contract, not a universal lower bound for all privatizers. A spherical sampler is cheap, but cheap generation does not make its statistical error tolerable.

RR adaptation is also weak with a large bank and small selector budget. The preferred profile probability is $e^\xi/(e^\xi+m-1)$: at $\xi=1$ it is about $0.731$ for two profiles, $0.280$ for eight. Publishing a raw energy summary to improve that decision would forfeit the proved construction.

## 5. What can reasonably be advanced

1. **Keep the validated law as a reference.** A two-dimensional analytic comparison can verify density ratios and energy formulas without launching FL. It will establish correctness and the stated limitation, not prove an accuracy improvement.
2. **Couple public geometry with clipping.** If profile $k$ defines an ellipsoidal clipping map whose output is in a smaller task-aligned public body $B_k$, replacement sensitivity lies in $2B_k$. A matching K-norm law can then have smaller variance in some directions. Compare clipping distortion together with noise cost: the original mean estimator has changed. For Candidate B, all labels must select a clipping/noise pair uniformly private on every raw input, including nonpreferred labels. A protected selector can choose such a pair, but the selection cost still competes with the gain.
3. **Use a public low-dimensional estimator if justified.** Projection can reduce the dimension exposed to noise, at the cost of discarding update components. The reconstruction must be postprocessing of that release. Sending omitted raw components separately invalidates the reduction.
4. **Explore full-law design under an explicit approximate/RDP contract.** This is a separate certificate family, not a relabeling of the current pure-DP proof. It needs its own whole-client vector divergence bound and complete trajectory accounting.

No current result establishes that block energy predicts CIA leakage, or that either candidate improves the frontier. Private selection can be proved without that predictive claim, but its utility cannot.

## 6. RDP-optimized scalar densities: nearby avenue, not a vector certificate

[Gilani et al., ICML 2025](https://proceedings.mlr.press/v267/gilani25a.html) optimize scalar additive continuous/discrete laws under a cost constraint using Rényi privacy. The existing methods-read explanation is [optimized_noise_followup.md](../literature_review/optimized_noise_followup.md), particularly §§2–4, Algorithms 1–3 and Appendix D of that paper. The proceedings identity was refreshed on 2026-10-04; this review does not newly audit their software or all proofs. Their scalar construction does not directly certify a dense client vector or private label selector.

For independent coordinate densities $q_{k,j}$ and a fixed shift $\delta$, the vector divergence is a sum of coordinate divergences. The necessary uniform certificate is

$$
\rho_k(\alpha)\ge\sup_{\|\delta\|_2\le2C}
\sum_jD_\alpha(q_{k,j}\|T_{\delta_j}q_{k,j}).
$$

The optimization is over the **joint shift ball**. Giving each coordinate a scalar bound and omitting the sum is wrong. A finite grid of shift tests is an empirical check, not a proof of the continuous supremum. Public tails, normalization, support and numerical approximation error matter; bounded-support translated densities can have infinite divergence.

If every profile instead has a valid conditional vector RDP bound $\rho_k(\alpha)\le\rho_{\mathrm{update}}(\alpha)$, composition with a profile selector can be analyzed in RDP. For RR with different preferred labels the exact order-$\alpha$ divergence is

$$
\rho_{\mathrm{RR}}(\alpha)
=\frac{1}{\alpha-1}\log
\frac{e^{\alpha\xi}+e^{(1-\alpha)\xi}+m-2}{e^\xi+m-1}.
$$

If preferred labels agree the divergence is zero. The worst-case joint release is bounded by $\rho_{\mathrm{RR}}(\alpha)+\rho_{\mathrm{update}}(\alpha)$, with the latter uniform across labels, histories and inputs. Account across rounds and convert to $(\epsilon,\delta)$ only after fixing the entire schedule. This own derivation offers a path to fair comparisons, not an assertion that optimized scalar laws already outperform whole-client vector controls.

## 7. Actionable verdict

Accept the proposed normalization, Gamma/uniform sampler, whole-input RR selector and additive privacy certificate under the conditions above. Reject an anticipated quadratic-utility advantage of anisotropic bank selection while all profiles are enlarged to cover the same Euclidean ball. The next concrete proposal should either change clipping/sensitivity geometry with explicit distortion accounting, justify a nonquadratic objective, or develop a vector-certified density optimization under an explicit approximate privacy target. Hold FL implementation until this distinction is resolved.

The nearest source relationships are existing K-norm geometry and scalar RDP law optimization; neither the public bank nor its RR composition alone establishes novelty. The target should remain contribution privacy at a fixed enrolled slot, with empirical CIA behavior evaluated as a separate consequence under a declared observer. Source links were refreshed on 2026-10-04. No new literature-review inclusion counts, experiments, status files or commits were changed by this review.

## 8. Independent check of the revised geometry-and-clipping reference

**Revision reviewed 2026-10-04:** the bank now changes the permitted clipped estimator, rather than enlarging every noise body to contain the original common Euclidean shift ball. This removes the specific PSD-domination objection in §4. It does not establish learning superiority or novelty.

Let $A_k\succ0$ be public, and define

$$
B_k=C A_k^{1/2}B_2^d,\qquad
\|u\|_{B_k}=\frac{\sqrt{u^\top A_k^{-1}u}}{C},\qquad
v_k(u)=\frac{u}{\max(1,\|u\|_{B_k})}.
$$

The clipped vector is always in $B_k$, regardless of the original training output's norm. Two arbitrary whole-client inputs therefore give $\|v_k(D)-v_k(D')\|_{B_k}\le2$. For update budget $\eta$, use

$$
q_k(z)=\frac{\eta^d}{2^d\Gamma(d+1)\operatorname{Vol}(B_k)}
\exp\left[-\frac\eta2\|z\|_{B_k}\right].
$$

The proposed sampler is correct: independently draw $r\sim\operatorname{Gamma}(d+1,\text{scale}=2/\eta)$ and $w\sim\operatorname{Unif}(B_k)$, then take $Z=rw$. It is precisely the K-norm law for sensitivity body $2B_k$. The ratio of any two translates whose centers are in $B_k$ is at most $e^\eta$. These statements require finite positive $C$, positive $\eta$ and nonsingular public shapes; setting an axis exactly to zero changes the support and needs a separate fixed-subspace analysis.

Its noise is zero-mean with

$$
\operatorname{Cov}(Z_k)=\frac{4(d+1)C^2}{\eta^2}A_k,\qquad
\mathbb E\|Z_k\|_2^2=\frac{4(d+1)C^2}{\eta^2}\operatorname{tr}A_k.
$$

The earlier $\lambda_{\min}(A_k)$ denominator disappears because the *estimator's sensitivity body changed*. It must not be reintroduced without also returning to the former common-ball contract.

### Two-dimensional profiles and the actual utility tradeoff

Choose $0<r<1$ and public shapes $A_1=\operatorname{diag}(1,r^2)$ and $A_2=\operatorname{diag}(r^2,1)$. Both have determinant $r^2$ and body area $\pi C^2r$. They therefore have identical normalization constants and total noise energy,

$$
\mathbb E\|Z_k\|_2^2=\frac{12C^2}{\eta^2}(1+r^2),
$$

versus $24C^2/\eta^2$ for the isotropic body $CB_2^2$. Both revised covariances are no larger than that isotropic covariance, with a strict reduction in one direction. Their difference from one another is indefinite, allowing a task to prefer one orientation. Equality of determinants alone is neither the utility argument nor the privacy proof; the clipping/support bounds provide the certificate.

For deterministic reference update $u=(C,0)$, profile 1 retains it exactly, while profile 2 clips it to $(rC,0)$ and incurs squared distortion $C^2(1-r)^2$. For $u=(0,C)$ the roles reverse. More generally, conditional on a raw update and selected profile, zero-mean independent noise gives the exact squared-error decomposition

$$
\mathbb E[\|v_k(u)+Z_k-u\|_2^2\mid u,k]
=\|v_k(u)-u\|_2^2+\operatorname{tr}\operatorname{Cov}(Z_k).
$$

This illustrates a genuine noise-versus-distortion hypothesis. Matching profile to a private energy statistic may reduce distortion relative to the wrong orientation, but the selector's privacy budget competes with that gain. In particular, a smaller energy than an isotropic law at the *same update budget* is not automatically smaller total error than an isotropic control spending the *entire* budget on its update. Compare both total-budget controls and clipping distortion, plus eventual task behavior. The energy-statistic preference rule still needs justification as a predictor of useful learning or CIA leakage.

### One-time private selection and correlated local training randomness

Use the categorical RR rule from §3 once, with selector budget $\xi$, to draw a profile $L$. Future uploads retain this profile. For every fixed profile $k$, public history $h$ and raw client input, each later clipped center must lie in the same public $B_k$. The protecting noise must be a fresh draw from the specified public conditional law; it must not reuse randomness entangled with raw training state.

Even if selection and local training share private randomness, the update certificate survives: condition on $L=k$ and the observed history. The clipped center can have different arbitrary conditional distributions in the two input worlds, but both are supported in $B_k$. For every two points $v,v'\in B_k$ and every output $y$,

$$
q_k(y-v)\le e^{\eta_t}q_k(y-v').
$$

Integrating over both conditional center laws proves the same bound for their noisy-output mixtures. No independence between the selected label and raw clipped center is required. The needed independence is the specified fresh noise kernel conditional on center/profile/history, or an equivalent proved conditional kernel. Positivity of RR probabilities ensures conditioning on each profile is legitimate in both worlds.

By adaptive composition, the hypothetical release $(L,Y_1,\ldots,Y_T)$ satisfies

$$
\epsilon_{\mathrm{total}}\le\xi+\sum_{t=1}^T\eta_t,
$$

provided every conditional update certificate holds uniformly over **all** histories, profiles and inputs, and the schedule has that worst-case bound. Data-dependent unaccounted stopping/budgets cannot be justified by adding only the realized path's expenditures. Hiding the profile is postprocessing of this joint release. Repeated observations may eventually reveal the selected profile accurately; this does not require paying RR repeatedly, because the one-time profile release was already covered. The remaining uploads still incur their update costs.

The selector depends on private data, so one cannot subsequently argue that the future views are independent of that data after conditioning on the selected label. They are not; each conditioned noisy-update kernel must separately satisfy the uniform certificate just established. Rebuilding the bank privately, changing the raw preference label and resampling it later, or leaking private diagnostic summaries changes the mechanism and requires new accounting. Deterministic public history-dependent changes to the clipping/noise pair are permissible only with an explicit uniform bound at every history.

**Revised verdict:** this is a valid and potentially useful reference hypothesis for geometry-specific clipping paired with non-Gaussian noise. It is sufficiently specified for a bounded two-dimensional analytic design comparison, but provides no authorization or evidence for an FL experiment. Keep the full input/observer contract, one-time selector accounting and distortion-inclusive controls explicit.
