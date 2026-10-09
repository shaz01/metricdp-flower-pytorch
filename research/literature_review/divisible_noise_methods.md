# Generalized and multi-scale Laplace: methods reconstruction

2026-10-06. Owner-approved follow-up, extended by “move without stopping ... until you find ... a very promising technique or ... knowledge ... shape our research direction.” Primary: Harrison–Manurangsi, FORC2025, [official article/PDF](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.FORC.2025.12), [full-version arXiv](https://arxiv.org/abs/2504.05202v1). Inspected official §2–5, proofs of Theorems13/15/16/18/21 and sampler propositions; §6 shuffle application is not transferred. See [independent review](../proposals/2026-10-06_divisible_methods_math_review.md). This is a reconstruction and project feasibility analysis, not a certified implementation.

## Understand the mechanisms first

Ordinary discrete Laplace has a peak at zero and exponentially decreasing integer tails. Its sampler is the difference of two geometric counts. Generalized discrete Laplace replaces those counts with negative-binomial counts of fractional shape. Small shape concentrates mass at zero; the scale and shape jointly determine privacy. Multi-scale Laplace instead adds several independently noisy integer quantities, multiplied by different integer scales. Each possible neighboring shift is protected by the corresponding scale component. A continuous version adds smoothing noise so noninteger shifts are protected too.

These mechanisms already solve scalar, query-independent noise design and independent share generation. They do not estimate a client's distribution from private data. Their nearly optimal scalar MSE is not a claim about neural accuracy, CIA or repeated whole-client training.

## Exact definitions and finite certificates

Use NB(r,p) to count failures before r successes, extended to positive real r. Its PMF is Gamma(k+r)/(Gamma(r)Gamma(k+1)) * p^r(1-p)^k. Its variance is r(1-p)/p². Here p is success probability; software conventions must be checked.

For beta>0,a>0, GDL(beta,a)=NB(beta,1-exp(-a)) minus an independent identically distributed count. Variance is beta/(cosh(a)-1). Shapes add under independent summation, so shares with shapes beta*q_i recover aggregate shape beta*sum_i q_i.

For integer sensitivity m, beta≥1 has exact worst-case privacy loss a*m. For 0<beta<1 it is log(f(0)/f(m)); the explicit PMF is hypergeometric and needs careful stable evaluation. A convenient sufficient bound is a*m+log(m/beta). Theorem15 selects beta=m*exp(2-epsilon), a=2/m when epsilon>2+log(m). These finite prerequisites matter; an asymptotic error order is not a calibration recipe.

Basic MSDLap(epsilon,m) is X=sum_{l=1..m} l*DLap_l(epsilon), with independent components. It protects every integer shift of magnitude≤m at epsilon: for shift l, retain the l-scaled component and regard the rest as independent convolution. Its exact variance is m(m+1)(2m+1)/(6*(cosh(epsilon)-1)). Theorem16 is valid for every epsilon>0; the high-epsilon MSE simplification has a narrower regime.

Theorem18 adds an integer coarse scale r and a discrete smoother. For r≥1, use r*MSDLap(epsilon-1,floor(m/r))+DLap(1/r). Split a nonnegative shift into quotient and remainder, then use symmetry for negative shifts. The two shift bounds add. r=0 refers to the basic mechanism, not a division-by-zero sampler. The extra tuning is already in the paper; it is not client-specific adaptation.

## Continuous law and an independently reconstructed extension

Theorem21 normalizes scalar sensitivity to one and uses m=ceil(exp(epsilon/3)), lattice spacing h=1/m, discrete privacy epsilon_d=epsilon-1, and

    Z=h*MSDLap(epsilon_d,m)+Lap(h/2).

Scale Z by sensitivity S for a general scalar query. A shift of magnitude≤S can be rounded to a lattice shift plus a residual of magnitude≤S*h/2. Discrete noise pays epsilon_d and the Laplace smoother pays1. Independence and the triangle inequality give total epsilon. Exact variance is

    S²/m² * [m(m+1)(2m+1)/(6*(cosh(epsilon-1)-1)) + 1/2].

This expression, rather than big-O notation, permits fair finite-budget comparison. The theorem advertises epsilon≥2. The analogous GDL substitution must satisfy the **actual integer** condition epsilon-1>2+log(ceil(exp(epsilon/3))); “epsilon>4.5” alone is insufficient because of the ceiling. For example epsilon4.6 gives m5 and fails the actual inequality.

**Our extension of the displayed shift proof:** choose any epsilon_d>0 and epsilon_s>0 with sum epsilon, and any integer m≥1. Then

    Z=(S/m)*MSDLap(epsilon_d,m)+Lap(S/(2*m*epsilon_s))

has the same scalar epsilon shift certificate. Exact variance is

    S²/m² * [m(m+1)(2m+1)/(6*(cosh(epsilon_d)-1))
             +1/(2*epsilon_s²)].

This follows from Theorem16 and the remainder proof; it is not a claim of optimality or a new noise family. It avoids falsely declaring all epsilon<2 impossible. Low-budget utility must still be checked against ordinary Laplace/Gaussian controls. No sampler has been implemented here.

## Client-side shares and the peer view

For each MSDLap scale l, client i uses a difference of independent NB(q_i,1-exp(-epsilon_d)) counts, then multiplies by l. For the smoother it uses independent Gamma(shape q_i, scale S/(2*m*epsilon_s)) differences. Weighted contribution is this complete share; an unweighted upload divides by its public aggregation weight a_i. Every client must retain every required component, including the smoother.

After a curious peer/coalition subtracts its known shares, let Q_H=sum of remaining public q_i. Require Q_H≥1 in every permitted dropout/coalition configuration. At equality the unknown law is the certified base mechanism. Above one it is the base law convolved with independent residual shares, preserving the certificate. Total released noise variance is Q_all times the base variance, whereas protection uses Q_H. For eight fixed slots and one curious peer, q_i=1/7 gives the same8/7 utility overhead identified in the [observer review](../proposals/2026-10-06_observer_contract_review.md).

For GDL, the residual beta*Q_H also permits direct scalar privacy evaluation, including cases below the conservative share threshold. This does not automatically certify the complete smoothed multi-scale law after dropout; do not discard its other components or extrapolate one component's formula.

Weights, public/protected parameters, fresh hidden randomness, same dummy noise and target outside coalition remain required. Repartitioning fixed public q_i with unchanged Q_all leaves the same aggregate law and cannot itself improve model utility. Raw-data-dependent scales/shapes change that law and require separate accounting.

## Vector and round accounting: the feasibility issue

For independent coordinate channels j at round t, a sufficient whole-client bound is sum_{t,j} epsilon_tj, provided each coordinate's complete target-versus-dummy shift is bounded by its calibrated sensitivity S_tj, uniformly conditional on preceding transcript. Coordinate bounds must include weights/clipping; they are not empirical standard deviations.

A total label8 on three coordinates in one release gives equal coordinate budgets8/3, not8. On20 visible rounds it gives8/60. For51 coordinates on20 rounds it gives8/1020. These are examples of a conservative independent-coordinate composition, not impossibility results for joint vector laws, better accountants or different release schedules. Reusing scalar epsilon8 in every coordinate/round would instead consume480 or8160 respectively.

The high-epsilon scalar error advantage can therefore disappear in the current high-dimensional, repeatedly observed task. The theorem's chosen large-budget parameterization is not a shortcut around composition. Joint sensitivity geometry and joint temporal distributions are potentially more consequential than replacing Gaussian marginals with a new scalar density. A final-model-only iteration-amplification theorem cannot be applied to a peer receiving intermediate models.

## Sampling details that matter

Naive MSDLap generates2m NB variables per share. The paper's sparse sampler first draws their total, then uses the conditional Dirichlet-multinomial law to allocate counts among components. Most coordinates are zero at high epsilon, so a sparse map saves work. Its exact Word-RAM argument assumes rational shape/input parameters and exact uniform/Bernoulli/geometric primitives. Library floating-point draws do not inherit that implementation proof.

Apparent published notation/pseudocode problems require correction before porting: max divergence lacks a log in its definition despite additive use; Theorem13/Corollary14 print epsilon in the wrong inequality direction; Algorithm2's strict boundary omits one uniform urn outcome; Algorithm3 lists integer shape although fractional rational shape is required. The rejection proof also prints a reversed power of p before subsequently using the correct acceptance-probability direction. These are independently checked implementation warnings, not claims that the distributional mechanisms fail. Details and exact page locators are in the mathematical review.

Integer noise on an unrounded continuous query exposes its fractional part. A quantized implementation must transform the query and bound rounding sensitivity, weights and decoding. A continuous smoothing proof does not automatically supply a modular finite-precision protocol. Overflow/wraparound, random seeds, abort behavior and released diagnostics belong in the actual transcript.

## Research decision

Keep these distributions as strong scalar non-Gaussian controls. Do not select them as the main client-specific defense from scalar asymptotic error alone. The original question remains distribution **construction** and the complete observable learning law. Continue with temporal mechanism predecessors and one-time private client surrogate construction: either may address repeated use of the same client's dataset more directly than a marginal-noise replacement. These are investigations, not established utility improvements or novelty claims.
