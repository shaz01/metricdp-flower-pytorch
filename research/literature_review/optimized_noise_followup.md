# Optimizing the noise law itself: Gilani et al. (ICML 2025)

Read date: 2026-10-01. This follow-up addresses the owner's question about developing a distribution beyond Gaussian noise. It analyzes a construction precedent, not a new mechanism or novelty claim.

**Source:** Atefeh Gilani, Juan Felipe Gomez, Shahab Asoodeh, Flavio Calmon, Oliver Kosut, and Lalitha Sankar. *Optimizing Noise Distributions for Differential Privacy.* ICML 2025, PMLR 267:19505–19522. [Proceedings](https://proceedings.mlr.press/v267/gilani25a.html), [publisher-linked PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/gilani25a/gilani25a.pdf), [author software](https://github.com/SankarLab/Renyi-DP-Mechanism-Design).

**Read scope:** complete construction sections §§2–4, Theorem 3.1 and Theorem 3.6 statements/proof sketches, Algorithms 1–3, §5 numerical experiments, and Appendix D's query/normalization protocol. Physical PDF pp.3–9 and 18. Supporting proofs in Appendices A–C were not independently certified. Software repository existence was checked; code was neither audited nor executed.

## Source explanation — under 200 words

The authors optimize an additive scalar noise distribution directly rather than selecting a Gaussian variance. At fixed Rényi order, they minimize worst-case divergence from a translated version of that same distribution:

$$
\min_Q\max_{|h|\le s}D_\alpha(Q\|T_hQ),
\qquad \mathbb E_Q[c(Z)]\le C.
$$

Symmetry and convexity permit a finite representation: probabilities of central bins/integers, plus nonzero geometric tails. Continuous bins have constant density; the discrete case assigns integer probabilities. For fixed family parameters and order, optimization is convex in those probabilities. Preconditioned gradient steps preserve normalization and cost; the algorithm also tunes the Rényi order for a target failure parameter and composition count. The construction therefore changes the complete density, not only covariance. Its experiments compare privacy curves and scalar mean-query MSE, especially moderate composition. They do not demonstrate federated client participation protection. The query experiments omit the common private normalization-quantile cost from comparative accounting. Family restriction, numerical convergence, tail handling and that accounting scope remain relevant when transferring the method. [§§3–5, Eqs.(11)–(25), Algorithms 1–3, Appendix D.](https://raw.githubusercontent.com/mlresearch/v267/main/assets/gilani25a/gilani25a.pdf)

## What this resolves for our direction

Gaussian noise is a useful reference law because its privacy calculations and aggregation are tractable. It is not the only possible law. This source establishes an existing distribution-design program: choose the probability density itself by trading distinguishability against a cost. Consequently, novelty cannot rest on “learn a custom non-Gaussian sampler” alone.

“Unique” should mean a construction appropriate to the project's specified client population, observations and utility task. It should not imply that an optimization problem has one mathematically unique solution. Convexity by itself does not establish strict convexity or optimizer uniqueness. Nor do a finite distribution family and an alternating numerical procedure establish global optimality over all multivariate distributions and all composition settings.

The owner’s research question remains substantial: what information does a client use to construct its distribution; how does that information identify participation leakage; how does the construction preserve learning; and what can be proved or measured about the complete transcript? Replacing the noise family without answering those questions would be a narrower experiment.

## A mathematical interpretation for agents

The following deductions explain how a full-law design would transfer; they are not claims already proved for this FL project.

### Privacy depends on density ratios, not simply noise power

For a scalar query $f(D)$ and input-independent additive noise density $q$, two neighboring outputs have the same noise law but means separated by $h=f(D)-f(D')$. Their Rényi divergence is

$$
D_\alpha(P_D\|P_{D'})
=\frac1{\alpha-1}\log\int q(z)^\alpha q(z+h)^{1-\alpha}\,dz.
$$

A variance constraint controls average squared displacement; the integral controls how easy translated distributions are to distinguish. Two laws with identical variance can give different density ratios, tails and privacy. This is the direct reason a distribution optimizer can improve on a fixed-family scale rule.

For $T$ mechanisms with uniform conditional RDP bounds $\gamma_t(\alpha)$, a standard conversion gives

$$
\varepsilon\le\inf_{\alpha>1}
\left[\sum_{t=1}^T\gamma_t(\alpha)
+\frac{\log(1/\delta)}{\alpha-1}\right].
$$

Thus a preferred noise shape can depend on how often it is released. A law optimized for one upload is not automatically preferred over a complete training trajectory. Adaptive training needs bounds valid for every relevant history, rather than treating rounds as independent observations for statistical evaluation.

### Useful learning requires a task cost

Squared error of a scalar statistic is one possible objective. For a model update, costs could instead describe expected task-loss increase, distortion in a public representation, or degradation of a particular learning signal. For a local quadratic approximation with public positive-semidefinite curvature matrix $H$, a zero-mean perturbation with covariance $\Sigma$ has second-order expected loss increase about $\operatorname{tr}(H\Sigma)/2$. This approximation depends only on covariance: to distinguish non-Gaussian laws beyond covariance, the task objective must capture higher-order effects or a nonquadratic loss.

Even then, optimizing a task cost alone can favor a distinctive law that reveals private client characteristics. Construction from private gradients, private curvature or raw history needs its own accounting or a complete data-dependent-law proof. Keeping the sampler parameters local does not make that dependence disappear.

### High dimension changes the optimization problem

If coordinates receive independent scalar noise, $q(z)=\prod_j q_j(z_j)$, Rényi divergence for a fixed mean shift vector $\Delta$ decomposes as

$$
D_\alpha(Q\|T_\Delta Q)
=\sum_jD_\alpha(Q_j\|T_{\Delta_j}Q_j).
$$

But the sensitivity constraint concerns a **set of joint shift vectors**, such as an ellipsoid or Euclidean ball. Bounding each coordinate separately and then forgetting the sum is not a multivariate privacy proof. Millions of model coordinates can make a naive per-coordinate construction inefficient or badly calibrated.

Candidate routes include a public low-dimensional basis, a jointly designed vector law, or groups/layers with explicitly allocated sensitivity. A fixed invertible linear map transforms both sensitivity and noise density. A purely low-rank perturbation leaves zero-randomness directions; it cannot hide arbitrary neighbor differences outside the protected subspace. These are proof/design choices, not supplied automatically by a scalar optimizer.

### Client laws must be analyzed after aggregation and peer conditioning

For conditionally independent client noises with characteristic functions $\phi_i$, weights $a_i$, and an independent server noise,

$$
\phi_{\mathrm{aggregate}}(\omega)
=\phi_{\mathrm{server}}(\omega)\prod_i\phi_i(a_i\omega).
$$

The observed aggregate law is a convolution of scaled client laws. It generally is not any one client's optimized law. Summation may smooth distinctive local shapes toward a Gaussian under appropriate conditions, or preserve tail/discrete features important to privacy. Optimizing one local upload does not automatically optimize this convolution.

A curious peer can subtract its own realized noise and update. Its residual law excludes known draws, and correlated shares require the full conditional law. Client absence can also change the convolution, even when the mean is unchanged. Distribution design must explicitly specify fixed slots, weights, dummy noise, observer knowledge and the IN/OUT comparison.

## Fidelity checks before using the author optimizer

- Keep infinite geometric tails represented in the actual sampler; deleting them changes both support and privacy. For continuous noise, random placement within a selected bin is part of the law.
- Check probability normalization, nonnegativity, strict positivity where neighboring shifts require it, and tail probability/moment calculations.
- Record bin width, tail onset, decay factor and numerical optimization tolerances. Finite-family convexity is not an error bound for an approximate numerical solution.
- Recompute privacy against all admissible shifts; do not validate only one selected direction or a few test points.
- Separate the guarantee for a specific fixed distribution from a curve made by selecting a different optimized distribution for each target budget.
- Account for private preprocessing and estimator releases in any claimed total budget. A fair comparison that omits a common cost is not a complete deployment accountant.
- Audit finite-precision sampling, integer query compatibility and round composition before transferring published scalar guarantees into training.

No numerical solver or sampler was implemented or certified in this follow-up.

## Relationship to other reviewed distribution work

[Residual-PAC, COV-06](covariance_sources.md) already provides a different route: learn a perturbation family using distributional reconstruction/entropy objectives. That broader population-dependent target should remain distinct from the present worst-case translated-density objective. The previous review also covers geometric Gaussian constructions. Together they show three choices to compare: **noise scale/geometry**, **complete law under a worst-case constraint**, and **learned law under a specified population-inference game**.

For our project, the next step is to define the whole-client game and a client-accessible construction objective before choosing among these routes. A carefully specified combination could be a research contribution; the existing methods do not establish its novelty or effectiveness against CIA.
