# Client-specific joint clipping and noise construction: first concrete proposal

2026-10-04 · `feature/client-specific-noise` · Research proposal, not an implemented defense or a claim of improvement.

## Direction and owner decision

The owner selected **“Hide dataset contribution (recommended)”**: protect whether a registered client's private dataset contributed to learning. Public enrollment and physical connection are not the hidden event. The earlier frontier motivates improving useful learning at a given attack risk; it does not establish that any proposed defense succeeds.

Our research question is: **can clients privately construct a joint clipping/noise profile that preserves useful update directions while protecting their entire contribution, and outperform fixed profiles and metric-inspired server noise under a declared CIA observer?**

Start with a finite, public family of valid non-Gaussian mechanisms and a private client-specific constructor. This makes the privacy cost of construction explicit. It is a tractable reference for investigating richer learned distributions, not a claim that a finite bank is the final novel mechanism.

A client measures how different profiles would distort its local update, privately selects a profile, clips inside that profile's allowed region, and samples fresh noise from its associated distribution. Changing the allowed region is essential: merely stretching noise around the same clipped update can make noise worse without creating a utility benefit.

## 1. Reference game and complete view

Use N public, permanently enrolled slots with fixed public weights 1/N. In both worlds every slot sends the same-shaped message on the same public schedule. In IN the target uses its private dataset. In OUT it has empty input and an unnoised update of zero, but executes the selector and noise sampler and sends a message. This is a new contribution-privacy protocol, distinct from removing a client and renormalizing legacy FedAvg sample-count weights.

The main attacker is a curious enrolled peer: global model trajectory, public configuration, roster and weights, its own input/update/noise/selection randomness, and any released diagnostics. No target raw update, raw profile scores, seed or private sample count is revealed. Timing, message presence and errors must follow the same specified public contract; a proof for the vector alone does not cover other outputs. Treat malicious-server behavior and collusion as separate extensions. The mathematical reference protects an individual randomized upload even if exposed, a stronger observation than the primary peer-model experiment.

The formal reference uses arbitrary whole-input replacement in a fixed slot, including empty input. It therefore also covers contribution versus empty input, but is stronger than that binary game. All conditional guarantees must hold at every shared public history. Fresh target noise must remain unknown to the attacker. Use canonical slot identifiers in seeds; do not remap IDs in OUT.

## 2. Public profiles: explicit clipping, density and sampler

Let the full released update have dimension d. Fix public C > 0 and positive-definite matrices A_k. Define a symmetric clipping body

    B_k = C A_k^(1/2) B_2^d,
    ||u||_{B_k} = sqrt(u^T A_k^(-1) u) / C,
    v_k(u) = u / max(1, ||u||_{B_k}).

Set v_k(0)=0. Every clipped output lies in B_k, so any two outputs differ by at most 2 in its body norm, regardless of how local training depends on private data.

For a per-round upload budget eta, use the non-Gaussian vector density

    q_k(z) = (eta/2)^d / [d! Vol(B_k)] * exp[-eta ||z||_{B_k}/2].

Sample independently r ~ Gamma(shape=d+1, scale=2/eta) and b ~ Uniform(B_k); set Z=rb and upload Y=v_k(u)+Z. Uniform ellipsoid sampling uses a uniform Euclidean-ball point followed by C A_k^(1/2); numerical approximations require their own implementation validation. This Gamma/uniform-body representation is based on the existing [K-norm mechanism, §4](https://arxiv.org/pdf/0907.3754), adapted here to our bounded whole-input query. It is not a newly invented distribution.

Triangle inequality gives, for every fixed profile and pair of client inputs,

    q_k(y-v_k(D)) / q_k(y-v_k(D')) <= exp(eta).

This is a pointwise whole-input pure-DP certificate for the reference release. Random local training is allowed because the bound is uniform over all clipped outputs, including distributions conditional on selecting a particular profile.

**Candidate A — public constructor.** Choose one profile using only public information, fix it throughout training, clip and sample as above. Include isotropic A=I and individually fixed shaped profiles. Candidate A establishes what geometry itself provides without private estimation.

A concrete two-dimensional bank is A_1=diag(1,rho²), A_2=diag(rho²,1), with 0<rho<1. Each shrinks one allowed update direction and its noise together. Start with rho=1/2 as a proposed diagnostic, not a tuned optimum. For large vectors, a public block-diagonal family can use the same definition, but its dimensional utility must be assessed before deep-model use. No unnoised complement of a private vector may be released.

## 3. Candidate B — private client-specific construction

At public initialization, compute one local probe update u_i under a fixed public training recipe. For each public profile compute a local preference score, for example

    score_k = ||u_i - v_k(u_i)||_2² + trace(Cov(Z_k; eta_1)).

The second term is public; the first is private. This is an update-reconstruction proxy, not an estimate of attack information or guaranteed downstream loss. A curvature-weighted proxy is a later extension requiring a declared estimator. On empty input u_i=0; resolve ties using a fixed public rule. Do not upload u_i or scores.

Let k* minimize the score. Privatize that categorical preference using m-ary randomized response with one-time budget xi:

    P(K=k* | k*) = exp(xi)/(exp(xi)+m-1),
    P(K=j | k*) = 1/(exp(xi)+m-1), j != k*.

This is the standard [k-ary randomized-response channel, §4.1, Eq.4](https://proceedings.mlr.press/v48/kairouz16.pdf), applied to a whole-client preference. The preferred profile need not have a bounded score sensitivity: the channel protects any two categorical preferences. Every profile can be selected for every input, so **every profile must protect every possible input**, including nonpreferred choices.

Keep K locally for the reference run. At each round compute the current local update, clip inside B_K and add a fresh q_K draw. The resulting distribution is client-specific through a privatized choice rather than an unaccounted raw covariance estimate. Even if K is exposed, the conservative joint-transcript bound is

    epsilon_total <= xi + sum_t eta_t.

With Candidate A the corresponding bound is sum_t eta_t. Reusing K pays xi once but can make a stale profile; refitting incurs another selection cost. Hiding K is post-processing of the conservative joint release and does not remove its cost. The statement requires conditional per-history upload guarantees and no extra private diagnostics. Intermediate global models and honest clients' responses must be accounted through the interactive protocol; they cannot introduce unprotected target-data access.

The finite mixture over profiles can express different directions and clipping scales. It does not yet learn an unrestricted new density, and RR can often select the wrong profile when xi is small. That is a research tradeoff to measure rather than conceal.

## 4. Why joint construction is necessary

The independent [mathematical review](2026-10-04_candidate_math_review.md) examined an initial alternative: keep Euclidean clipping radius C for all clients and enlarge each ellipsoidal noise body to contain the same 2C sensitivity ball. Its covariance dominates the isotropic reference in positive-semidefinite order. It cannot improve any positive-semidefinite quadratic noise cost at the same upload budget; spending additional privacy budget on selection worsens that comparison. Retain this as a negative control and do not present it as the main improvement mechanism.

For the revised joint profiles, our own moment calculation gives

    Cov(Z_k) = 4(d+1) C² A_k / eta².

Reducing an eigenvalue now reduces noise in that direction **and restricts allowed signal there**. Improvement can occur only if the useful signal survives that restriction. Total proxy error includes clipping distortion, selection errors and the noise increase caused by allocating part of the total privacy budget to selection. A method that wins only when those costs are omitted has not established a benefit.

Local whole-client privacy is demanding. For A=I, total noise energy is 4d(d+1)C²/eta²; it can be prohibitive for dense neural updates and many rounds. Begin with low-dimensional complete updates and analytical tradeoffs. Do not extrapolate feasibility to the repository's deep models.

## 5. Hypotheses and comparisons

1. Different clients favor different public joint profiles when their useful updates occupy different directions. Test this first using analytical reconstruction risk and clipping diagnostics.
2. Private selection can improve reconstruction or predictive utility over the best single public profile at the **same total privacy budget**, after selection errors and clipping distortion. An unprivate oracle selector is only a diagnostic upper bound.
3. Any surviving utility benefit can improve the independent CIA/accuracy tradeoff. A lower attack score caused by training collapse is failure, not success.
4. A matched conditional aggregate law has the same model-observer behavior regardless of whether honest noise is generated on clients or by a trusted server. Use this control to separate distribution construction from placement.

Controls: vanilla; historical metric-inspired server calibration as an empirical comparator; public isotropic/shaped non-Gaussian profiles; unprivate oracle; RR-private profiles; and a server/peer hybrid with the same unknown honest-noise convolution. Preserve the peer's own update and coins in that hybrid. Do not compare only equal variance when laws differ. Gaussian controls are allowed but do not define the proposal.

A metric heuristic without a comparable formal guarantee cannot be described as losing an equal-DP-budget comparison. Report empirical CIA/utility comparisons separately from formal whole-client budget comparisons.

## 6. Route to a genuinely new distribution

The immediate contribution to investigate is the **accounted distribution-construction process**, not renaming an existing sampler. Extend a public law bank only after each law has a uniform vector privacy certificate. Possible follow-on research optimizes body shape and a full radial/directional density against clipping-plus-utility cost, with a privatized construction statistic or an explicitly modeled population criterion.

[Gilani et al., ICML 2025](https://proceedings.mlr.press/v267/gilani25a.html) already optimize scalar noise under Rényi privacy; [K-norm geometry](https://arxiv.org/pdf/0907.3754) already adapts noise to sensitivity bodies. These are construction precedents, not proofs for a private whole-client FL vector law. Novelty requires comparison against these and adaptive client-noise work, including complete FACP/FedFR-ADP methods where access remains incomplete. A scalar guarantee, matching covariance, or a trained decoder failing to infer contribution is insufficient.

## 7. Next work and handoff

Read this proposal with the [independent math review](2026-10-04_candidate_math_review.md) and [pilot protocol](2026-10-04_pilot_protocol.md). The [first closed-form check](2026-10-04_analytic_feasibility.md) shows no advantage over an optimized public clipping radius in its two-axis toy at budgets 4, 8 or 16. Next, test richer heterogeneous utility geometries analytically at a fixed total budget, including one-time selector cost and optimized public controls. If no regime survives the controls, reject or revise this candidate before implementing FL. If one survives, implement a small complete-vector pilot with explicit transcript, independent attack-development/test trials, and the protocol's collapse criteria. A large sweep is not the next step.

No training or attack experiments were launched for this proposal. The owner confirmed the hidden event, not these numerical pilot defaults or a completed mechanism choice. No new review inclusion counts or certified privacy/novelty claims are implied.
