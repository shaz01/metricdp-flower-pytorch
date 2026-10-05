# Independent mathematics: hidden-selector tails and adjacency

2026-10-05. Analytic audit only. No code/results changes or additional experiments. This note distinguishes a conditional reference mechanism from the repo's raw, unprotected source trajectories. It does not supply a CIA certificate or a new mechanism.

## 1. Mechanism and assumptions

Use three public weighted-L1 bodies with semiaxes

$$
r_0=C(1,1,1),\quad r_1=C(1,1/4,1/4),\quad
r_2=C(1/4,1,1/4),\qquad C>0.
$$

Let $v_k(u)$ be radial clipping of the full three-coordinate update into body $k$. A preferred label $s\in\{0,1,2\}$ selects $A$ using RR probabilities $a=e^\xi/(e^\xi+2)$ for $A=s$ and $b=1/(e^\xi+2)$ otherwise. Conditional on $A=k$, independent Laplace coordinate noise has scales $2r_{kj}/\eta$, with $\eta>0$. Its density is

$$
q_{\eta,k}(z)=\frac{\eta^3}{64\prod_jr_{kj}}
\exp\left[-\frac\eta2\sum_j\frac{|z_j|}{r_{kj}}\right].
$$

The hidden upload density is $f_D(y)=\sum_kp_{s(D)}(k)q_{\eta,k}(y-v_k(u(D)))$. All configuration/history/weight assumptions in the conditional lemma remain necessary. Empty input has preferred label zero and update zero, **but still uses the same RR/noise channel**. Hiding $A$ does not remove its data dependence from $f_D$.

## 2. Replacement upper bound and tail sharpness

For arbitrary whole-input replacement, every two clipped centers differ by at most two in the relevant body gauge. Each conditional upload ratio is at most $e^\eta$, while RR probability ratios are at most $e^\xi$. Summing pointwise joint bounds gives the hidden mixture upper bound $e^{\xi+\eta}$.

To assess whether hiding $A$ can uniformly improve it, take $y=T(1,1,1)$ as $T\to\infty$. Profile 0 has exponential slope $3\eta/(2C)$, whereas profiles 1 and 2 have slope $9\eta/(2C)$. Their finite normalization factors and bounded clipped centers cannot compensate for that difference. Therefore profile 0 alone dominates both mixture tails, and

$$
\lim_{T\to\infty}\frac{f_D(T\mathbf1)}{f_{D'}(T\mathbf1)}
=\frac{p_{s(D)}(0)}{p_{s(D')}(0)}
\exp\left[\frac\eta{2C}\mathbf1^\top(v_0(u(D))-v_0(u(D')))\right].
$$

In the abstract domain permitting arbitrary preferred-label/update pairs, choose $s(D)=0$, $v_0(u(D))=Ce_0$, and $s(D')\ne0$, $v_0(u(D'))=-Ce_0$. The limit is $e^{\xi+\eta}$. The universal additive replacement bound is thus sharp for that domain, even with the label hidden. This tail proof is stronger than an unsuccessful finite output-grid search: likelihood ratios can attain their supremum only in distant tails.

Fixed scalar postprocessing $t>0$ preserves the likelihood-ratio supremum because it is an invertible rescaling of the upload. At $t=0$ the hidden upload is constant and has zero input privacy cost; that degenerate case is excluded from the sharpness claim. If the label itself is released, its RR cost can remain even at zero upload. Different noise families, restricted domains or extra observation channels require separate analyses.

## 3. Coupling of the actual label rule and bias gradient

The abstract extremal pair does not automatically belong to the actual dominant-label/gradient mechanism. Its preferred label and update are coupled. For bias contrasts, a possible limiting construction is a common public model with probabilities approaching $(1/2,1/2,0,0)$ and pure class-0 versus pure class-1 clients. The frozen label map gives preferred profiles 0 and 2. In the first Helmert contrast their raw gradient-step updates approach opposite axis directions, while the other contrast components vanish.

With the specified step size $0.5$, these raw axis magnitudes approach $0.5/\sqrt2\approx0.353553$. They saturate to $\pm Ce_0$ when $C$ is small enough, including the studied $C=0.3$. Finite softmax probabilities cannot be exactly zero; this is an approximation along public model histories. It supports a sharp **uniform-over-allowed-histories** argument only when those histories/input pairs are in the declared domain. It does not automatically establish equality of the bound at a particular trained checkpoint or for every clipping radius in the bank, especially radii 1 or 3. A larger radius or more restrictive dataset/training domain needs its own reachability/sensitivity analysis.

Thus a tighter actual-data/history certificate is logically possible, but must be derived for the declared coupled domain and maintained uniformly along the actual interactive protocol. It cannot be inferred just from keeping the label secret, a few observed client updates, or a bounded density-ratio plot.

## 4. Dataset-versus-empty is a different adjacency

For a fixed public enrolled slot with empty-input update zero, the relevant direct edge is $D\leftrightarrow\varnothing$. Every nonempty clipped center has body gauge at most one relative to the empty zero center. Hence the joint and hidden-upload upper bound on these edges is

$$
\epsilon_{\mathrm{dummy}}\le\xi+\eta/2.
$$

If $s(D)=0$, RR weights agree with the empty channel and that edge has the smaller bound $\eta/2$. The worst case must still include nonzero preferred profiles if the nonempty input domain permits them.

Sharpness in the abstract dummy-edge domain follows in the **reverse** direction: compare empty input, preferred profile 0 and center zero, against nonempty input with preferred profile nonzero and $v_0(u)=-Ce_0$. The same positive diagonal tail is dominated by profile 0, giving

$$
\lim_{T\to\infty}\frac{f_{\varnothing}(T\mathbf1)}{f_D(T\mathbf1)}
=e^{\xi+\eta/2}.
$$

DP requires both directions, so reverse-tail sharpness is enough. The actual coupled-input/history qualifications from the previous section still apply. In particular, hiding the selector does not supply a universal reduction beyond the dummy-specific additive bound for this family/domain.

The dummy-edge graph is weaker than arbitrary all-pairs replacement. A dummy-edge budget $E_D$ implies at most $2E_D$ between two arbitrary nonempty datasets by chaining through empty input; it must not be described as an all-pairs $E_D$ certificate. Contribution secrecy, replacement privacy and visible physical participation remain distinct targets. The slot must send comparable traffic/noise when empty, and its private weight/sample count or other metadata cannot expose the hidden event.

## 5. Fair calibration of dummy-only controls

For a desired dummy-edge budget $E_D$ and selector cost $0\le\xi<E_D$, set

$$
\eta=2(E_D-\xi),\qquad
\operatorname{LaplaceScale}_{kj}=\frac{r_{kj}}{E_D-\xi}.
$$

A fixed public profile or input-independent public mixture can spend all $E_D$ on its upload: use $\eta=2E_D$, giving scale $r_{kj}/E_D$. Both achieve the factor-two noise-scale reduction versus their replacement calibration. Leaving a baseline at scale $2r/E_D$ and improving only the hidden-mixture arm would create an artificial advantage caused by changing adjacency, not by mixture accounting.

For existing replacement-labeled configurations with $E=\xi+\eta$, the dummy-edge upper bound is $(E+\xi)/2$. A fixed profile at the same replacement label has dummy bound $E/2$. Thus equal replacement labels do not give equal dummy budgets when selector allocations differ. Repeat the accounting/calibration comparison explicitly before reinterpreting earlier utility scores under the owner’s contribution target. Under matched dummy budget the selector still increases a fixed-shape upload variance by $[E_D/(E_D-\xi)]^2$ relative to an all-budget fixed profile.

## 6. History, observer and alternative construction

All bounds above are conditional on the same observed public history and fixed public bank/radius/shrink/weights. A complete contribution protocol must account for target effects on prior model releases and all later conditional kernels. Raw source checkpoints, data-derived global derivatives, private sample weights or complementary unnoised updates in the audit are not protected by this isolated density calculation. Marginal hidden-upload accounting may be less conservative for a restricted observer, but cannot ignore information the actual observer receives.

Constructing a profile from an already protected history can avoid a *new* RR cost because profile choice is postprocessing. It does not erase the history’s earlier privacy/utility cost or the dependence of current conditioned updates on private input. Under replacement, later conditional uploads still need their replacement bound; under dummy-only adjacency they need the appropriate zero-contribution conditional bound. The probe, empty-world traffic, weighting and all histories must follow the same explicit contract. This alternative is a protocol design question, not free estimation or a demonstrated utility gain.

The useful finding is precise: the current three-profile family has a tail obstruction to universally discounting selector cost merely because its label is hidden. The contribution-versus-empty target admits a smaller direct shift bound, but that advantage belongs to equally recalibrated public controls too. No experiment or implementation is warranted by treating those two facts as a new privacy gain.

## 7. Actual implementation and saved-result check

Independently read `research/calculations/hidden_mixture_audit.py`, its protocol, and the reused `response`, `moments`, `configurations`, `basis` and `clipped_bank` functions. The log density uses the correct coordinate-Laplace normalizer and stable log-sum-exp over the three positive-probability profiles. Both vector centers are clipped separately in each body's full weighted L1 gauge. The abstract witnesses use opposite first-coordinate updates and route0 versus route2; the coupled witnesses derive the bias update from the stated class-constant softmax probabilities and the actual Helmert basis, at fixed C=.3. They are consistent with the preceding analytic argument. Finite tail points and a small positive class probability remain numerical illustrations, rather than a proof of each saved checkpoint's supremum. The density calculations precede shrink: their sharpness interpretation requires fixed positive shrink, with the zero-shrink exception above.

The cost routine correctly passes kernel parameter `2*E` to shared profiles, public mixtures and the explicitly unaccounted deterministic-route reference. Every RR candidate uses `eta=2*(E-xi)`. The reused moment routine includes independent selector dispersion, weighted aggregation and fresh upload-noise covariance; its noise term is half the global-Hessian contraction of covariance, and shrink multiplies its penalty by t². Thus the comparison does not leave public controls at replacement noise calibration. Radius, shrink and selector-budget tuning remain developmental and unaccounted. The script reads the cached training Arrow source and developmental indices, not a new test slice.

Read the saved `results/client_specific_noise/hidden_mixture_audit.json` without rerunning the calculation. It contains 18 rows and 54 dummy-budget cells. Recomputed the winning-grid-minus-public-mixture differences and checked every logged dummy-budget identity. Shared and best public-mixture scores agree in all 54 cells. The paid selector improves the score in 12 cells and worsens it in 42, with zero 0.001 feasibility flags. All improvements are label-stress cells at E=8 or E=16: score gains span 0.000460306–0.000654549 at E=8 and 0.000418444–0.000623879 at E=16. At E=4 the stress selector loses 0.000114422–0.000151665. Maximum gain is 0.0006545493196, below the developmental threshold; these are quadratic proxy changes, not directly evaluated test cross-entropy gains.

The last abstract tail witness, xi=4 and eta=12, logs replacement ratio 16 and reverse dummy ratio 10. The last coupled witness, small probability 1e-8, C=.3, xi=2 and eta=6, logs 7.999999763456458 and 4.999999763456458. These values support the analytic limits under the explicitly constructed admissible history. No material mathematical implementation discrepancy was found. This check changes neither the lack of a complete trained-pipeline privacy certificate nor the decision against claiming free savings from a hidden selector; a paid protected-history routing design remains a separate question.
