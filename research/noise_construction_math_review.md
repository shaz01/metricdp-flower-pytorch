# Independent mathematical review of the construction starting point

Date: 2026-10-01. Reviewed `noise_construction_proof_obligations.md` against its stated assumptions. This review checks the displayed calculations and observer interpretation; it does not certify a complete private federated-learning protocol.

## Required clarification

**Residual randomness versus hidden covariance.** The covariance matrix $B$ need not be secret. It is the covariance of random noise whose **realized draws** remain unknown to the peer after conditioning on its observations. The peer may know every public shape, scale and accounting parameter. Replace phrases such as “server mechanism with unknown covariance B” with “server mechanism whose noise unknown to the peer has covariance B.” This avoids suggesting that formal privacy relies on an attacker not knowing the covariance matrix. For data-dependent covariances, whether the matrix is disclosed or inferred is a separate issue already identified in the document.

## Calculations checked

1. **Independent weighted shares.** With fixed weights and conditional independence, aggregate covariance is $\sum_i a_i^2\Sigma_i+\Sigma_{\mathrm{server}}$. Conditioning on known independent draws removes those draws' contributions from the residual covariance. With correlated noise, the conditional covariance generally differs from subtracting marginal variances. The document correctly separates these cases.

2. **Fixed-weight add/remove and replacement.** A clipped target with norm at most $C$ has a fixed-denominator zero-padding mean shift at most $a_{\mathrm{target}}C$. Comparing two target vectors in the same ball gives $2a_{\mathrm{target}}C$. These are correct same-history bounds.

3. **Renormalized removal.** For nonnegative normalized weights and a nonempty surviving cohort, $U_{\mathrm{IN}}=a v+(1-a)U_{\mathrm{OUT}}$, hence $\|U_{\mathrm{IN}}-U_{\mathrm{OUT}}\|=a\|v-U_{\mathrm{OUT}}\|\le2aC$. For equal weights this is $2C/n$, provided $n\ge2$. Explicitly state $0\le a<1$ and that the remaining average exists; the formula does not address removing the only client. Different ellipsoidal bounds require a fresh renormalization analysis rather than automatically reusing the common Euclidean $C$.

4. **Ellipsoidal clipping.** For public positive-definite $A$, dividing by $\max(1,\|A^{-1/2}u\|/C)$ ensures $v^TA^{-1}v\le C^2$. Zero updates cause no division by zero in this formulation.

5. **Whitened separation.** Write an add/remove shift as $\Delta=aCA^{1/2}z$, $\|z\|\le1$. Maximizing the quadratic form yields $\sup\Delta^TB^{-1}\Delta=a^2C^2\lambda_{\max}(A^{1/2}B^{-1}A^{1/2})$. The displayed bound is correct for fixed positive-definite $B$. Whole-client replacement multiplies the worst-case separation by two and its squared bound by four if both target inputs use the same public ellipsoid.

6. **Gaussian accounting.** For equal-covariance Gaussians, order-$\alpha$ Rényi divergence is $\alpha\Delta^TB^{-1}\Delta/2$, so a uniform squared-separation bound $\kappa^2$ gives $\alpha\kappa^2/2$. This requires the same covariance in the compared conditional worlds. Adaptive rounds need a bound at every possible public history; private covariance variation cannot be passed through this formula unchanged.

7. **Variance-only discriminator.** The one-dimensional log likelihood ratio has the displayed sign and scale: $\log(s_{\mathrm{out}}/s_{\mathrm{in}})+y^2(1/s_{\mathrm{out}}^2-1/s_{\mathrm{in}}^2)/2$. Identical means therefore do not eliminate distinguishability.

## Estimation and observer caveats retained

The proposed composition route is legitimate as a proof strategy: a whole-client-private estimator followed by a uniformly private update mechanism for each possible estimator output. Count the estimator's privacy cost even when its value is kept local; do not merely condition on a raw private estimate and declare it fixed. The subsequent mechanism must be private for a fixed estimator output across neighboring inputs, including the absent-slot world. An estimator output never observed externally may be treated as an internal hypothetical release for a conservative composition argument, but this is not free estimation.

The null comparison needs the complete conditional joint law, including the peer's own update/noise, identities/counts, diagnostics and accessible state. Equal marginal aggregate covariance alone is insufficient. Trusted-server dummy noise preserves a specified model-release law but does not conceal actual uploads from that server. The document correctly marks this limitation and the missing dummy construction for privately learned covariance.

## Conclusion

No algebraic error found in the displayed bounds under their stated assumptions. Apply the residual-randomness wording correction and nonempty-cohort qualification. Keep the document's conditional-reference status: covariance construction, private estimation, sampling, metadata and transcript accounting remain necessary proof obligations before any formal certificate or superior-CIA claim.
