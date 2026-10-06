# Independent divisible-noise methods and transfer review

2026-10-06. Read the [official Harrison–Manurangsi FORC2025 PDF](https://drops.dagstuhl.de/storage/00lipics/lipics-vol329-forc2025/LIPIcs.FORC.2025.12/LIPIcs.FORC.2025.12.pdf), §§2–5 and Appendix A.1–A.5; visually checked the published Theorem13 inequality. No sampler or experiment was run. Verdict: the scalar methods give a concrete proof-feasible aggregate-noise family, subject to the vector, coalition and observer qualifications below. They do not establish a client-specific learned geometry mechanism.

## Primary-method calibration anchors

Notation: D=integer sensitivity; a>0; F is Gauss hypergeometric. GDL is NB(beta,1−exp(−a)) minus an independent copy (Definition9, p12:6). Its exact minimal privacy parameter (Theorem13, pp12:6–7) is

    L(beta,a,D)=aD                         if beta≥1;
    L(beta,a,D)=aD+log{F(beta,beta;1;e^(−2a))
      /F(beta,beta+D;1+D;e^(−2a))*Gamma(D+1)Gamma(beta)/Gamma(beta+D)}
                                            if 0<beta<1.

Corollary14 supplies L≤aD+log(D/beta). Theorem15 takes beta=D*exp(2−epsilon), a=2/D, requiring epsilon>2+log D.

MSDLap=sum_i=1..D i*DLap(epsilon) (Theorem16, p12:8). Theorem18 (pp12:8–9) adds discrete smoothing after coarsening. Number20 is a remark, not a theorem. Continuous Theorem21 (pp12:9–10), for epsilon≥2 and sensitivity Delta>0, sets

    m=ceil(exp(epsilon/3)); a=epsilon−1;
    Z=(Delta/m)*sum_i=1..m i*DLap(a)+Lap(Delta/(2m)).

The proof splits each displacement into its nearest lattice shift and a remainder; their costs are epsilon−1 and1. Appendix A.1/A.2 justify GDL ratio bounds; A.4 proves distributional convergence to Arete, not a new finite-parameter DP guarantee.

## Independently derived finite utility and certificate

Let S2(m)=m(m+1)(2m+1)/6. Direct independence and the second moments give, for the explicit continuous construction above,

    Var(Z)=Delta²/m² * {S2(m)/(cosh(epsilon−1)−1)+1/2}.

This is a finite exact variance for this parameter choice, rather than substituting an unspecified constant from its asymptotic order. The nearest-grid proof covers all real displacements with absolute value≤Delta. For a shifted vector, it is insufficient to quote the scalar theorem at one global sensitivity and apply the same epsilon independently to every coordinate.

A conservative vector transfer fixes coordinate bounds Delta_j≥|a_T u_Tj| and budgets epsilon_j with sum_j epsilon_j≤E. Independently instantiate the scalar law at each (Delta_j,epsilon_j). Multiplying coordinate likelihood ratios yields E for the target's contribution-versus-zero-dummy edge. A product law needs its own dimension-dependent utility accounting; continuous Theorem21 requires epsilon_j≥2, so a many-coordinate equal budget split can leave the theorem's stated regime. The existing three-dimensional bias representation is a more bounded feasibility question than dense FL vectors. A proven weighted-L1 privacy-loss profile could improve this conservative allocation, but is not furnished automatically by these scalar high-epsilon constructions.

The scalar displacement bound must be derived for the weighted query, including fixed positive public target weight a_T. Dummy adjacency allows a_T u_T rather than two arbitrary client updates. The same certificate under nonempty replacement generally needs twice the sensitivity or a different budget. Private sample weights, raw shared anchors, estimator choices and released calibration are outside this conditional scalar-noise argument. Independent coordinate laws also do not directly certify correlated matrix-shaped noise or random private rotations.

## Fractional shares and coalition transfer

Define a public fractional convolution parameter gamma_i>0 for each contributor. A continuous-MSDLap share can be represented as all m scaled GDL(gamma_i,a) components plus the smoothing Gamma(gamma_i,b) difference, where b=Delta/(2m). The scaled multicomponent terms and smoothing term use independent coins. Their characteristic functions form a convolution semigroup in gamma, because their logarithms scale linearly with gamma.

For an observer's unknown set H, gamma_H=sum_i∈H gamma_i. If gamma_H≥1, the unknown aggregate equals one fully certified continuous-MSDLap law plus an independent residual semigroup law with parameter gamma_H−1. Convolution preserves the base certificate. Every allowed coalition/dropout/history must retain this condition. The target must be outside the coalition, and the coalition's own contributions/noise must be subtracted before identifying H. Individual fractional shares are not required to satisfy local upload privacy; the observer must not see the server's individual uploads or remaining clients' seeds.

This semigroup proof is an independent transfer deduction, not a claim that the paper proves this repo's particular peer protocol. Sharing only the discrete component and adding no correctly shared continuous smoothing noise breaks the proof for real updates. Using arbitrary client-specific scales across corresponding components also breaks this common-parameter semigroup calculation. Public unequal shapes can allocate contribution across clients, but do not by themselves learn different geometries or give a client-dependent covariance utility theorem. If gamma_H<1, the certified-base convolution is unavailable: a separate exact privacy analysis is needed. The discrete GDL formula can evaluate a changed aggregate shape; it cannot be transplanted unchanged to the continuous transformed law with fractional smoothing.

All-aggregate noise variance scales by the full sum of fractional parameters. Overallocating enough for a worst allowed coalition may make the released aggregate noisier than an equivalent trusted-server law. Client placement therefore offers distributed generation under an observer/access contract, not better aggregate utility when the output law is identical. Previous raw anchor/configuration issues remain, and round-by-round histories require uniform conditional composition.

## Published notation and implementation traps

The visual PDF prints epsilon≤L in Theorem13. Its DP definition and likelihood-ratio argument require **L≤epsilon**. Corollary14 and Appendix Corollary32 repeat this apparent direction error. Likewise the max-divergence display omits a logarithm although later proofs use additive log-divergence. Interpret calibration through the actual likelihood ratio, not those literal signs.

Theorem18's floor decomposition should first use symmetry to restrict to nonnegative shifts; otherwise its stated integer bound need not hold for negative floor rounding. The continuous nearest-integer decomposition is separately valid because m is rounded upward. For transformed GDL, the finite prerequisite is epsilon−1>2+log(ceil(exp(epsilon/3))); an asymptotic shortcut epsilon>4.5 is not a sufficient finite condition. For example epsilon=4.6 gives m=5 and fails that prerequisite. The explicit MSDLap construction avoids this GDL restriction.

The printed Algorithm2 samples a one-based uniform integer but tests U<initialsize, leaving the boundary incorrectly in the reinforcement branch; the Pólya-urn argument requires U≤initialsize. Algorithm3 labels its shape integer despite rational shapes in the surrounding sampler theorem. These are traps for literal transcription, not objections to the abstract distribution/semigroup construction. Future implementation requires a separately reviewed exact sampler, including indexing, parameter convention and finite-computer concerns. Appendix weak convergence alone must not be used as a pure-DP guarantee for an approximate or truncated sampler.

An independent normalization check also reverses the printed rejection-proof power on p12:13: multiplying the proposed NB(ceil(r),p) probability by acceptance (r)_w/(ceil(r))_w yields p^(ceil(r)−r) times the desired NB(r,p) probability. This gives a valid acceptance probability≤1. The opposite printed exponent exceeds one for noninteger r. The later stated geometric trial probability has the correct direction; this is a proof transcription issue, not a failure of the rejection construction.

The next justified work is a frozen conditional observer/query contract and finite noise-cost comparison with a directly calibrated trusted-server control. Neither the scalar asymptotic theorem nor fractional divisibility alone establishes a practical gain over metric privacy, general CIA protection, or a novel local-distribution estimator.
