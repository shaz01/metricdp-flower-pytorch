# Independent review: rotated diamonds and protected residual proxies

2026-10-04. Mathematical review of the proposed bounded quadratic spike. No implementation, experiments, CIA measurements or novelty claims are supplied by this note. The population/objective remains the declared dynamic quadratic model; this is not a general FL optimality argument.

## 1. Rotated body, clipping and actual noise law

Fix positive public semiaxes $L\ge S>0$ and the public angles $\phi\in\{0,\pi/4,\pi/2,3\pi/4\}$. Use the standard rotation

$$
R_\phi=\begin{pmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{pmatrix},
\qquad B_\phi=R_\phi\{x:|x_1|/L+|x_2|/S\le1\}.
$$

For a full residual update $u=\theta_i-w_1$, transform it into body coordinates $x=R_\phi^\top u$, radially clip using $|x_1|/L+|x_2|/S$, and rotate back:

$$
v_\phi(u)=R_\phi\frac{x}{\max(1,|x_1|/L+|x_2|/S)}.
$$

Draw independent $\zeta_1\sim\operatorname{Laplace}(0,2L/\eta_1)$ and $\zeta_2\sim\operatorname{Laplace}(0,2S/\eta_1)$, then use $Z=R_\phi\zeta$. The normalized density is

$$
q_\phi(z)=\frac{\eta_1^2}{16LS}
\exp\left[-\frac{\eta_1}{2}
\left(\frac{|(R_\phi^\top z)_1|}{L}
+\frac{|(R_\phi^\top z)_2|}{S}\right)\right].
$$

Rotation has unit Jacobian, so the normalization is unchanged. Its covariance and trace are

$$
\Sigma_\phi=\frac8{\eta_1^2}R_\phi\operatorname{diag}(L^2,S^2)R_\phi^\top,
\qquad\operatorname{tr}\Sigma_\phi=\frac8{\eta_1^2}(L^2+S^2).
$$

Every two clipped outputs differ by at most two in the $B_\phi$ gauge, so the density ratio for a replacement shift is at most $e^{\eta_1}$. The angle is held fixed for this conditional comparison. This is ordinary K-norm geometry with a public rotated body, not a new probability-family theorem.

**Matrix convention check:** for row-vector arrays, body coordinates are `u @ R`, clipping returns physical coordinates via `clipped @ R.T`, and local-coordinate noise is rotated by `noise @ R.T`. For column vectors the corresponding operations are $R^\top u$ and $Rx$. Mixing conventions can silently assign the wrong angle. At $\phi=\pi/4$, the column vector $e_1$ maps to body coordinates $(1/\sqrt2,-1/\sqrt2)$.

## 2. Equal axes do not eliminate diamond rotation

If $L=S=a$, covariance is $8a^2I/\eta_1^2$ for all angles. The **body and density still change** under a $45^\circ$ rotation: the gauge is a rotated $\ell_1$ norm, not a Euclidean norm. For an axis vector of magnitude $s$, the unrotated body clips to magnitude $\min(s,a)$, while the $45^\circ$ body clips to $\min(s,a/\sqrt2)$. Thus equal-axis profile selection may genuinely change clipping and the non-Gaussian law even though second moments agree.

At equal axes, the bank contains two distinct bodies: $0^\circ$ and $90^\circ$ coincide; $45^\circ$ and $135^\circ$ coincide. With $L>S$ the four specified orientations generally differ. A $180^\circ$ shift always gives the same centrally symmetric body. Use a fixed public tie rule; do not interpret duplicated profiles as distinct adaptive utility effects. This is why the earlier equal-axis *swapped-coordinate* identity check cannot be reused as an identity test across all rotated angles.

## 3. Protected constructors and interactive composition

For either proxy, choose the public-bank angle minimizing the Euclidean clipping distortion

$$
\phi_i=\operatorname*{arg\min}_{\phi}
\|x_i-v_\phi(x_i)\|_2^2,
\qquad
x_i=Y_{i0}\quad\text{or}\quad x_i=Y_{i0}-w_1.
$$

Both are deterministic functions of the client's already protected upload and the public shared model. No raw current update, raw gradient summary or unprotected training state may enter this constructor. The subsequent clipping map *does* act on the raw full residual; its conditional noise certificate protects that access.

Because the bank has common semiaxes and the Euclidean noise trace is angle-invariant, adding that public one-upload quadratic noise cost to the angle score would not change the argmin. This does not mean the score minimizes global learning risk: average clipping biases can interact, the proxy is noisy, and covariance direction can matter for another objective.

The probe $Y_{i0}$ is $\eta_0$-DP against arbitrary whole-client input replacement after public clipping. The model $w_1$ is an honest function of all protected uploads. In a conservative transcript containing all relevant first uploads, $w_1$ adds no separate raw target-data disclosure. For every fixed realized public history and selected angle, the later profile law is uniformly $\eta_1$-DP for every client input. Hence the interactive target guarantee is $\eta_0+\eta_1=E$; angle selection adds no separate RR charge. Revealing the selected angle is also covered as postprocessing of the protected history.

This reasoning does not require the angle probability to remain unchanged across inputs. It requires the *conditioned* update law to be valid for every angle/history and fresh protecting noise to have its declared conditional distribution. Other honest clients' updates depend on the target only through the protected model history; an additional unprotected target-data channel would invalidate this argument. A peer's own coins may be known, but the target's protecting coins remain unknown. Fixed enrolled slots, public weights and empty-input behavior from the preceding design remain necessary to the eventual contribution game.

The residual proxy has the identity

$$
Y_{i0}-w_1=(1-1/N)Y_{i0}-\tfrac1N\sum_{j\ne i}Y_{j0}.
$$

It is correlated with both the true residual and probe noise. At $N=1$ it is exactly zero and the public tie rule determines its profile; it is then uninformative, despite the true residual generally being nonzero. For general $N$, it may be more useful than stale $Y_{i0}$, but this is a hypothesis to compare, not an independence or consistency theorem.

## 4. Conditional objective check

Keep $F_i(w)=\tfrac12\|w-\theta_i\|^2$, $w_1=\operatorname{mean}Y_0$, full residual updates $\theta_i-w_1$, and $w_2=w_1+\operatorname{mean}Y_1$. Conditional on the realized federation and all probe uploads, the clipped vectors and angles are fixed. Fresh later-noise independence gives the exact excess-risk identity

$$
\mathbb E\left[\tfrac12\|w_2-\bar\theta\|^2\mid\theta,Y_0\right]
=\tfrac12\left\|w_1+\operatorname{mean}_i v_{\phi_i}(\theta_i-w_1)-\bar\theta\right\|^2
+\frac{4(L^2+S^2)}{N\eta_1^2}.
$$

The second term is independent of the angles, while the first is not. The same formula applies to a public fixed or independent-random angle, using its own chosen public semiaxes. It also supplies a conditional Monte Carlo check and permits integrating out later noise in risk estimation. Do not substitute the stationary IID-client bias formula or treat clients as independent after their shared model update.

## 5. Comparisons and bounded validation

Compare stale and residual proxy constructors separately. Give a tuned fixed-angle static arm the same four-angle bank, semiaxis family, total budget and development information; retain an exactly same-probe fixed-angle control as well. A public random-angle diagnostic must select independently of private data/probe and get fresh target noise; its angle distribution and randomization frequency should be specified. All its fixed-angle kernels remain valid even if the angle is revealed, so input-independent angle mixing adds no privacy cost.

Each arm's numerical law must match its rotated body: rotate **both** clipping and noise, check the resulting gauge is at most one, and verify the covariance's off-diagonal entries as well as its trace for unequal axes. Useful limiting checks include $0^\circ$ recovering the old diamond, $90^\circ$ swapping axes, round-trip orthogonality, equal-axis $90^\circ$ body identity, and equal-axis $45^\circ$ nonidentity. The no-clipping/no-noise residual correction should still reach $\bar\theta$ in one unit step.

Tune on development simulations and evaluate fixed choices on fresh federation trials. Common innovations across arms are valid paired-comparison aids if every arm's marginal noise remains correct; they are not protecting randomness shared with an attacker. Normal intervals remain exploratory Monte Carlo summaries, especially across multiple arms/configurations. Preserve one-release, public-population and fitted-on-development postprocessing diagnostics from the preceding spike; a rotated-bank win within a forced two-round schedule does not imply a strongest-public-mechanism win.

This extension tests body orientation and a history-corrected proxy, not a novel privacy primitive or an established CIA defense. Its results can support a bounded next research decision in this synthetic population. Formal privacy follows from the uniform conditional law, while learnability, contribution inference and eventual neural-model relevance require separate evidence.

## Actual implementation check

Reviewed `research/calculations/rotated_residual_probe.py` and its saved JSON on 2026-10-04. No substantive mathematical/code error was found. Body coordinates use row vectors multiplied by $R$, and the selected scalar radial factor multiplies the original full-vector update; this is equivalent to clipping in body coordinates and rotating back. Independent local-coordinate Laplace noise is transformed by $R$ in the column-vector convention. Its conditional variance addition is exactly $4(L^2+S^2)/(N\eta_1^2)$.

Maximizing the proxy radial factor correctly minimizes Euclidean radial-clipping distortion because the proxy's Euclidean norm is the same across all angles. The corrected residual proxy `max(1,0.8/a0)*Y0-w1` uses the declared public common input magnitude to reverse the probe's deterministic clipping attenuation. It adds no raw private input to profile construction, but relies on that known toy population. Its interpretation should not be generalized to an unknown heterogeneous input norm without a new construction argument.

All configuration tuning and matched-probe static selection use development federations. Fresh held-out federations are evaluated under frozen choices. Common noise innovations across arms preserve each rotated law's marginal distribution; later innovations do not enter profile selection. Integrating out later noise for the primary endpoint is valid, including paired comparisons against the sampled one-release arm: both estimate the declared expected-loss contrast, with different conditional variance. The saved secondary endpoint also samples the entire two-round output.

The 100000-draw conditional check reports expected loss $0.0537605428$, sampled mean $0.0538043114$ and standard error $0.0001197589$, consistent with the identity. This checks a moment numerically, not finite-precision privacy. Equal semiaxes in selected configurations still allow different diamond orientations, so they must not be described as bitwise-equivalent to the fixed-angle arm merely because their covariances agree.

The stored corrected-residual paired intervals show a small improvement over the tuned static arm in the $(E,N)=(16,8)$ cell, but positive corrected-residual-minus-one-release intervals in all six cells. Interpret these as finite-grid exploratory utility comparisons with the recorded population and unadjusted intervals, not a broad constructor success. The one-release comparator retains the listed isotropic initial-diamond family; no strongest-public-distribution claim follows. No code or results were changed, and no further diagnostics were launched by this review.
