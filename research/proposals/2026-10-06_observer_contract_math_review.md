# Independent aggregate-observer contract mathematics review

2026-10-06. Verdict: the stated Gamma-share construction supplies a valid sufficient **conditional aggregate-view, dataset-versus-zero-dummy** privacy certificate. The observer boundary is essential: it does not protect the server's individual-upload view, and it provides no noise-placement utility advantage over another construction with the same aggregate law. This review is mathematical only; no sampler, calculation run, CIA test or complete training certificate is claimed.

## Mechanism and unknown-noise law

Use fixed registered public weights a_i>0. Write each client's aggregate contribution as

    a_i u_i + W_i,

where W_i is already the **weighted noise share**. Do not multiply this share by a_i again. For each coordinate j, take W_ij=G_ij,+−G_ij,−, with independent Gamma(shape k_i, scale b_j) variables. Coordinates and clients use independent fresh random variables. Public shapes and scales must not be raw functions of protected client data. Shapes need be positive for ordinary Gamma draws; a zero residual shape below means the deterministic zero law, not a Gamma(shape0) sampler.

A peer/coalition can subtract its own weighted queries and noise because it knows its own inputs and random coins. Let H be the remaining contributors whose noise coins are unknown to that observer, including the honest target. The residual coordinate-noise characteristic function is

    product_i∈H (1+b_j² t²)^(-k_i)
      = (1+b_j² t²)^(-K_H),   K_H=sum_i∈H k_i.

If K_H≥1, this is the characteristic function of Laplace(scale b_j) plus an independent Gamma-difference noise with shape K_H−1 and scale b_j. For K_H=1 the residual is exactly Laplace. Equality of characteristic functions establishes the distributional identity; the clients need not designate an actual Laplace component in their individual draws.

The requirement K_H≥1 must hold for every allowed unknown-noise set, public history and dropout pattern, not only for the full registered roster before coalition subtraction. Unknown *noise coins* are what matter: side information about another honest client's dataset does not remove its independent noise, whereas leaked seeds or shares can. A separate common Laplace core per coordinate makes convolution with the extra residual noise a privacy-preserving postprocessing operation.

## Shift certificate and adjacency

Fix a public history and an allowed observer's initial private inputs/coins. For a target T outside the coalition, compare its nonempty query u_T with the dummy query zero. Other conditional query kernels must have no additional direct access to the target's private input. The conditional aggregate mean shift is a_T u_T. Independent coordinate Laplace noise has density ratio bounded by

    exp(sum_j |a_T u_Tj| / b_j).

Thus requiring this sum≤epsilon for every admissible target query gives an epsilon dummy-edge certificate for the aggregate residual. Adding the independent Gamma-difference residual preserves the certificate; convolution cannot increase this shift likelihood-ratio bound. The same argument holds in the reverse direction. It is uniform over other unknown contributors' data-dependent mean shifts when those are fixed by the same conditional public history and their own inputs.

This is not an epsilon arbitrary-replacement certificate. If both possible nonempty queries have weighted gauge≤epsilon, the triangle inequality gives a replacement bound≤2epsilon, or chaining through the zero dummy gives the same result. An all-pairs epsilon claim would need a smaller query body or larger noise scale. Dummy input must retain the same registered weight, random noise generation, message schedule and visible traffic. Dropping noise when OUT invalidates this argument and can directly expose contribution state.

The observer may retain its own coins, own updates and causal state. Giving it target-independent initial randomness and its own fixed dataset does not cost privacy when the likelihood-ratio bound is uniform conditional on those variables. Its subsequent state may depend on prior protected aggregates through causal postprocessing. This does not authorize additional direct reads of target uploads, noise seeds, raw statistics, or an unprotected target-dependent global model. A malicious peer's additional actions would require a precisely stated protocol guaranteeing the same conditional response bound; the passive aggregate-view argument alone is not a general active-adversary proof.

## Composition and access boundary

Per-round conditional bounds compose along the same observed DP public history, provided the bank, weights, shapes, scales, permitted H and clipping constraints are fixed public information or accounted-for protected choices. A bank selected from already protected history can be postprocessing, but the previous releases retain their budget. Raw data-derived geometry, allocation or calibration is not free. A change of public b or k between neighboring inputs could alter the complete noise law; the above argument would no longer be a simple common-noise translation proof.

Here the server is trusted for access control and sees individual uploads. No secure-aggregation protocol is assumed or established. The result protects peers/coalitions restricted to released aggregates plus their own coins; it does not protect against a server that reveals individual uploads or joins the coalition. This boundary must be enforced by the actual communication/logging/access design. Multiple distinct releases, subset aggregates or rounds require their joint/composed accounting; each individual marginal certificate does not alone certify their combination.

Dropout/participation patterns must be conditioned on externally determined public events with the same law in both target worlds. A target-dependent dropout event is itself an observable privacy issue. The allocation must remain sufficient after every allowed externally determined dropout; if a roster shrinks below the stipulated threshold, the mechanism needs a predeclared safe response rather than silently using an insufficient aggregate. No-dropout assumptions must be stated explicitly where they are used.

## Eight-client arithmetic and correct controls

For N=8, fixed a_i=1/8, one curious peer, no dropout and k_i=1/7, the seven unknown shares have K_H=1. Suppose all clients' unweighted updates lie in the common L1 ball of radius C and the scales are equal. Set b=C/(8 epsilon). The target's weighted L1 displacement is at most C/8, so the aggregate-view dummy certificate is epsilon. Per-coordinate variance of a Gamma-difference with shape k is 2 k b².

The released full eight-client aggregate therefore has noise variance

    2 b²*(8/7) = 2 C²/(56 epsilon²).

After the curious peer subtracts its own share, the unknown noise variance is 2b²=2C²/(64epsilon²). This is the distribution used for the observer's privacy argument; it is not the full aggregate's utility variance.

By comparison, imposing independent epsilon dummy privacy on each **individual unweighted upload** with local Laplace scale C/epsilon produces weighted aggregate variance

    8*(1/8)²*2C²/epsilon² = 2C²/(8epsilon²).

The Gamma aggregate has seven times less variance than that stronger local-upload requirement. The advantage comes from the weaker observer contract and the composition of unknown noise shares, not merely client-side placement or a proof of improved utility in trained FL. A trusted-server Laplace aggregate calibrated directly to the same target sensitivity C/8 has variance2C²/(64epsilon²), so the Gamma full aggregate is 8/7 times noisier. A server can sample the same full Gamma aggregate law; placing that identical law at clients does not improve its aggregate utility. The comparison must specify the trusted server's release and colluder side information rather than infer equivalence between threat models.

## Individual-upload caveat

For 0<k≤1/2, the one-dimensional Gamma-difference density is unbounded at zero (a power singularity for k<1/2 and logarithmic divergence at k=1/2). Relative to a nonzero location shift, the density ratio is unbounded in a neighborhood of the unshifted center. Consequently a translated individual share law at k=1/7 cannot provide any finite pure-DP individual-upload shift guarantee for nonzero changes. This is a positive-measure likelihood-ratio obstruction, not merely assigning a special density value at a point.

Do not extrapolate this singularity statement to every k<1: the stated argument concerns k≤1/2. Likewise K_H≥1 is a sufficient aggregate certificate via an explicit Laplace convolution, not a proved necessary characterization of every smaller-shape law or every restricted query set. The mechanism's relevant certified object is the unknown aggregate residual, while exposed per-client uploads remain outside the protection contract.

The design is mathematically coherent under these explicit conditions. Its immediate contribution is an observer-specific sufficient law and transparent noise-cost comparison. It supplies neither a client-specific raw-distribution estimator nor a demonstrated advantage over trusted-server metric privacy, and it does not settle the practical coalition/dropout/access assumptions without a protocol review.
