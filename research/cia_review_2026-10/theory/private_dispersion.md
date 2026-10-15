# Private median-distance metric privacy (D4 item 1): mechanism and privacy proof

WP3, Agent C role, 2026-10-15. Implementation: `metricdp_pytorch/private_metric_strategy.py`
(`PrivateMedianMetricPrivacyServerSideFixedClipping`); tests: `tests/test_private_metric_strategy.py`; numerical checks:
`private_dispersion_checks.py` beside this note.

## 1. Why the paper's mechanism needs a repair

The paper's metric privacy adds noise with standard deviation `sigma_t = z C / (n d_t)` to the average of the clipped
updates, where `d_t` is the maximum pairwise distance between the client models of round `t`. Two problems:

1. `d_t` is computed from the participants and used without privatization. The noise level is itself a release that
   depends on who participates; WP1 measured it (gate G1(a): the blind noise-level attack reaches AUC 0.838 for the
   largest client in 3-8 client federations, `results/cia_frontier/side_channel_audit/FINDINGS.md`).
2. `d_t` has no upper bound, so `sigma_t` has no lower bound and the mechanism has no privacy guarantee at all.

Repair: release a robust dispersion statistic privately, compute the noise only from that private release, and bound
the noise from below by a public floor.

## 2. Choice of statistic and private estimator

The roadmap allows a privately released median through smooth sensitivity, propose-test-release or private quantiles.
The paper's dispersion is defined on **pairs** of clients: one client takes part in `n - 1` of the `n(n-1)/2` pairwise
distances, so adding or removing it moves a pairwise median by up to `n - 1` ranks. Rank-based private medians
(exponential mechanism, smooth sensitivity of the median) then need a rank error of order `(n - 1) / epsilon` out of
`n(n-1)/2` ranks, i.e. a constant fraction of the range for any useful `epsilon` at 3-48 clients.

This mechanism therefore uses a **per-client** distance, `s_i = dist(theta_i, theta_t)`: the paper's distance (mean
over layers of the per-layer Euclidean norm) between client `i`'s returned model and the current global model. Each
client influences only its own `s_i`. The median of the `s_i` is tracked with the private quantile estimator of
Andrew et al. (2021, "Differentially private learning with adaptive clipping"): each client contributes one bit, so the
count has sensitivity 1. Evidence E3 (`research_directions.md` Section 2) supports a median over a maximum: the median
moves at most 1.97% when one client is removed, the maximum up to 12.3%.

## 3. Mechanism

Public parameters: slot count `n` (the canonical number of clients; absent clients count as empty slots), clipping
norm `C`, noise multiplier `z > 0`, quantile bounds `0 < q_min <= q_0 <= q_max`, target quantile `gamma` (0.5), step
`eta` (0.2), bit-noise standard deviation `sigma_b > 0`, number of rounds `T`.

Data: `D = (D_1, ..., D_n)` with `D_i` the dataset of the client in slot `i`, or `D_i = ⊥` if the slot is empty.
**Adjacency:** `D` and `D'` are adjacent if they differ in exactly one slot, which holds a client in one of them and is
empty in the other (add/remove one whole client; the participant count is not public).

Round `t = 1..T`, given the public state `(theta_t, q_t)` (both functions of the public initial values and of the
releases of rounds `< t`):

1. Each present client runs any (possibly randomized) local algorithm on `(D_i, theta_t)` and returns `theta_i`.
2. Bits and updates: `b_i = 1[s_i <= q_t]`, `u_i = clip_C(theta_i - theta_t)` (scaled to norm at most `C`). For an
   empty slot `b_i = 0` and `u_i = 0`.
3. Release `R_t = (B_t, Theta_t)` with independent noises
   - `B_t = sum_i b_i + N(0, sigma_b^2)`,
   - `Theta_t = theta_t + (1/n) sum_i u_i + N(0, sigma_t^2 I_d)`, `sigma_t = z C / (n q_t)`.
4. Public updates (post-processing): `theta_{t+1} = Theta_t`;
   `q_{t+1} = clamp(q_t exp(-eta (B_t / n - gamma)), q_min, q_max)`.

The implementation matches this: the wrapped FedAvg must weight clients equally, its mean over the `m` replies is
rescaled by `m / n` to the slot denominator, the noise std uses the quantile held before the round
(`_round_quantile`), and the quantile is updated after aggregation.

## 4. Privacy

Notation follows Dong, Roth and Su (2022, "Gaussian differential privacy", JRSS-B): a mechanism is `mu`-GDP if for every
adjacent pair its trade-off function is at least `G_mu(alpha) = Phi(Phi^{-1}(1 - alpha) - mu)`.

**Lemma 1 (Gaussian mechanism).** If `f` has L2 sensitivity `Delta` under the adjacency, `f(D) + N(0, s^2 I)` is
`(Delta / s)`-GDP. (Dong et al., Theorem 2.7.)

**Lemma 2 (noise floor).** Let `s >= s_min > 0`. Then `f(D) + N(0, s^2 I)` is `(Delta / s_min)`-GDP.

*Proof.* `N(0, s^2 I)` has the law of `N(0, s_min^2 I) + N(0, (s^2 - s_min^2) I)` with independent terms, so the
mechanism is the `s_min` Gaussian mechanism followed by adding independent noise that does not depend on the data,
a post-processing. GDP is preserved by post-processing (Dong et al., Proposition 2.8), and Lemma 1 gives
`Delta / s_min` for the `s_min` mechanism. ∎

**Lemma 3 (one round).** Fix the public state `(theta_t, q_t)`. The round-`t` release `R_t` is `mu_round`-GDP with

    mu_round = sqrt(mu_u^2 + mu_b^2),   mu_u = q_max / z,   mu_b = 1 / sigma_b.

*Proof.* With the state fixed, every present client's `theta_i` is a function of its own data, `theta_t` and its own
randomness, so adding or removing the client in one slot changes only that slot's `(b_i, u_i)`; all other clients'
contributions have the same law in both datasets. Hence `sum_i b_i` changes by at most 1 (the slot's bit is 0 or 1 and
is 0 when empty), and `(1/n) sum_i u_i` changes by `u_i / n` with `||u_i|| <= C`, i.e. by at most `C / n` in L2. By
Lemma 1, `B_t` is `(1 / sigma_b)`-GDP. Because `q_t <= q_max`, `sigma_t = z C / (n q_t) >= z C / (n q_max)`, and
`sigma_t` is fixed by the state, so by Lemma 2 `Theta_t` is `(C/n) / (z C / (n q_max)) = q_max / z`-GDP. The two noises are
independent, so `R_t` is their composition, `G_{mu_u} ⊗ G_{mu_b} = G_{sqrt(mu_u^2 + mu_b^2)}` (Dong et al.,
Corollary 3.3). The randomness of the local algorithms is part of the mechanism and has the same law under both
datasets. ∎

**Theorem 1.** The `T`-round mechanism is `sqrt(T) mu_round`-GDP, i.e.

    mu_total = sqrt(T (q_max^2 / z^2 + 1 / sigma_b^2)).

*Proof.* Round `t` chooses its parameters (`theta_t`, `q_t`, hence `sigma_t`) from the public state, a function of
earlier releases. By Lemma 3 every round is `mu_round`-GDP **for every** value of that state, with the same `mu_round`.
Adaptive composition of `T` such mechanisms is `G_{mu_round}^{⊗T} = G_{sqrt(T) mu_round}` (Dong et al., Theorem 3.2 and
Corollary 3.3). ∎

**Corollary 1 (share of the median release).** The median release takes the share
`mu_b^2 / (mu_u^2 + mu_b^2)` of the composed `mu^2` (GDP composes by adding `mu^2`). With `sigma_b = 3 z / q_max`
(`bit_noise_for_share(z, q_max, 0.1)`, the default) the share is exactly 10%, for any `T`. For reporting,
`mu`-GDP converts to `(epsilon, delta(epsilon))` with `delta = Phi(-epsilon/mu + mu/2) - e^epsilon Phi(-epsilon/mu - mu/2)`
(Dong et al., Corollary 2.13; `gaussian_epsilon` in the implementation).

**Corollary 2 (no noise-level side channel beyond the accounted release).** `sigma_t` is a deterministic function of
`q_t`, which is a post-processing of `(q_0, B_1, ..., B_{t-1})`. Any attack on the noise level, including the WP1
blind noise-level attack, is a post-processing of the release and is covered by Theorem 1. The same attack on the
paper's mechanism reads `d_t`, which no accounting covers.

## 5. What the guarantee means for the baseline

Lemma 2 is where the guarantee comes from, and it has a consequence: the repaired mechanism is global DP at the floor
`sigma_min = z C / (n q_max)`, plus extra noise in rounds where the private median is below `q_max`, plus the 10% of
the budget spent on the median. **At the same formal guarantee it never adds less noise than global DP with
`sigma_min`, and it adds more whenever `q_t < q_max`.** Metric calibration can therefore not buy utility over global DP
at a matched formal guarantee; its only possible advantage is empirical (lower measured leakage at matched noise, e.g.
in rounds with small dispersion). This agrees with the repository's earlier finding that metric privacy behaves like
global DP once leakage is matched (`research/cia_review_2026-10/project_state.md` Section 4). The fair comparisons for
WP4 are: (i) matched measured leakage, as the roadmap prescribes, and (ii) matched formal `mu_total`, where global DP
gets `sigma = sigma_min` and the private-median mechanism gets the floor that gives the same `mu_total`
(`q_max / z` scaled by `sqrt(0.9)`).

Choosing `q_max`: a value close to the typical median distance keeps the extra noise small; a large `q_max` makes the
floor (and hence the guarantee) weak. `q_max` must be fixed from public information or from a separate, accounted
calibration run, not from the evaluation runs.

## 6. Assumptions and limits

- Trusted server, server-side clipping (as in the repository's DP wrappers); client models are seen by the server.
- Equal public weights and the public slot count `n` as denominator (owner decisions, roadmap Section 1). Under
  size-weighted aggregation the update sensitivity is `2 a_target C`, not `C / n`, and Theorem 1 does not apply.
- The bit uses the unclipped model distance; this is allowed because each bit depends on one client's data only.
- Floating-point and pseudo-random-generator attacks on Gaussian sampling (Mironov 2012) are out of scope; the
  implementation uses numpy generators like the repository's other DP wrappers.
- The tracker's accuracy (how close `q_t` stays to the true median under the bit noise) is a utility question, not a
  privacy one; it is measured in `private_dispersion_checks.py` on committed logs.
