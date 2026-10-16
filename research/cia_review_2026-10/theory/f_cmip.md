# Client-level membership inference with partial knowledge (D3): bounds against the GDP bound

WP3, Agent C role, 2026-10-15. Numerical checks: `f_cmip_checks.py` beside this note (outputs in `data/f_cmip_*.csv`,
figure `figures/f_cmip_regimes.png`). Direction: `research_directions.md` D3. Notation follows Dong, Roth and Su
(2022): `G_mu(alpha) = Phi(Phi^{-1}(1 - alpha) - mu)` is the trade-off function (type II error at type I error
`alpha`) of distinguishing `N(0, 1)` from `N(mu, 1)`.

## 1. Setting

One round releases `R = theta + (1/n) sum_{j in S} u_j + Z`, `Z ~ N(0, sigma^2 I_d)`, where `u_j` is client `j`'s
update clipped to norm `C`, `n` is the public slot count (equal weights, fixed denominator; roadmap Section 1) and `S`
the set of participants. **Client inference game (removal):** a target `i*` is in `S` (IN) or not (OUT); every other
participant is the same in both worlds.

Three attacker knowledge levels, from strong to weak:

- **K_inf (GDP worst case):** knows every other client's update and the target's update. Its trade-off is at least
  `G_{mu_GDP}`, `mu_GDP = C / (n sigma)`, with equality when `||u_i*|| = C`.
- **K_crowd:** knows the target's data (hence `u_i*`) and the population of clients, but not which clients the other
  `n - 1` slots hold. The sum of the other updates is random to it; Gaussian model (central limit over `n - 1` clients
  drawn from the population): `B = sum_{j != i*} u_j ~ N(m_B, Sigma_B)` with `m_B` known and `Sigma_B = (n - 1) Sigma_pop`.
- **K_m (partial knowledge, the CIA threat model):** knows the population and a shadow sample of size `m` from the
  target's distribution, not the target's data.

## 2. Results

**Proposition 1 (crowd bound, full release).** Under the Gaussian crowd model, K_crowd's trade-off for one round is
exactly `G_{mu_c}` with

    mu_c = ||S^{-1/2} u_i*|| / n,   S = sigma^2 I + Sigma_B / n^2,   and   mu_c <= ||u_i*|| / (n sigma) <= mu_GDP.

*Proof.* Given the public `theta`, `R` is `N(theta + m_B/n, S)` under OUT and `N(theta + m_B/n + u_i*/n, S)` under IN.
Whitening by `S^{-1/2}` maps the pair to `N(0, I)` and `N(S^{-1/2} u_i*/n, I)`, whose trade-off is `G` of the norm of
the mean difference (the Neyman-Pearson test projects on that direction). `S ⪰ sigma^2 I` gives
`S^{-1} ⪯ sigma^{-2} I`, hence the inequality; `||u_i*|| <= C` after clipping. ∎

**Proposition 2 (`k`-dimensional statistic).** If the attacker uses only `T = W^T R` for a fixed `W in R^{d x k}`, its
trade-off is `G_{mu_W}` with `mu_W = ||(W^T S W)^{-1/2} W^T u_i*|| / n <= mu_c`. For `k = 1` and direction `w`:

    mu_w = |w^T u_i*| / (n sqrt(sigma^2 ||w||^2 + w^T Sigma_B w / n^2)) <= rho C / (n sigma),
    rho = sigma ||w|| / sqrt(sigma^2 ||w||^2 + w^T Sigma_B w / n^2) <= 1.

*Proof.* `T` is Gaussian with covariance `W^T S W` and mean shift `W^T u_i*/n`; the first claim is Proposition 1 in
dimension `k`, and `mu_W <= mu_c` because `T` is a post-processing of `R`. For `k = 1`, `|w^T u_i*| <= ||w|| C`. ∎

The repository's loss attacker is a `k = 1` statistic: to first order the shadow-loss change of a round is
`delta = g^T (R - theta)` with `g` the shadow-loss gradient, so `w = g`. Then `sigma^2 ||g||^2` is the DP-noise
variance of the increment and `w^T Sigma_B w / n^2` its crowd variance: `rho^2` is the share of the increment's variance
that the DP noise explains. This is what Section 3 measures.

**Proposition 3 (rounds).** Stack the per-round statistics `delta_t = w_t^T (R_t - theta_t)`, with DP-noise variances
`D = diag(sigma^2 ||w_t||^2)`, crowd covariance `Gamma` across rounds and IN mean shifts `s_t = w_t^T u_t / n`. The
`T`-round trade-off is `G_{mu_T}`, `mu_T = sqrt(s^T (D + Gamma)^{-1} s)`, and

1. if the crowd terms are independent across rounds (`Gamma` diagonal), `mu_T^2 = sum_t mu_t^2`; with the worst-case
   signals `|s_t| = ||w_t|| C / n` this gives `mu_T / mu_GDP,T = sqrt(mean_t rho_t^2)`;
2. `mu_T <= sqrt(s^T D^{-1} s)` (the GDP value for the same statistics) always, and this is attained for some crowd
   covariance with the same per-round variances: a rank-one `Gamma = v v^T` with `v_t = ± sqrt(Gamma_tt)` chosen so that
   `sum_t v_t s_t / D_t = 0` (Sherman-Morrison leaves `s^T D^{-1} s` unchanged);
3. if the crowd term is constant across rounds (`Gamma = tau^2 1 1^T`, equal `s_t = s`, `D_t = D`),
   `mu_T^2 = T s^2 / (D + T tau^2) -> s^2 / tau^2`: the leakage stays bounded as `T` grows.

*Proof.* Joint Gaussianity with a common covariance gives `mu_T` as in Proposition 1; (1) is the diagonal case; for
(2), `D + Gamma ⪰ D` gives the bound and Sherman-Morrison gives
`s^T (D + v v^T)^{-1} s = s^T D^{-1} s - (s^T D^{-1} v)^2 / (1 + v^T D^{-1} v)`, equal to `s^T D^{-1} s` when
`s^T D^{-1} v = 0`; (3) follows from the same formula with `v = tau 1`. ∎

Consequence: a multi-round crowd bound needs an assumption on how the crowd's contribution is correlated over rounds.
Without one, the worst case is the GDP bound. `f_cmip_temporal.csv` illustrates this for 50 rounds with crowd variance
equal to the noise variance: the ratio to GDP is 0.707 for independent rounds, 0.140 for a constant crowd and 1.000 for
the adversarial rank-one crowd.

**Proposition 4 (shadow sample of size `m`).** Any test of K_m is also available to K_crowd (who knows the target's
distribution and can draw the shadow sample itself), so K_m's trade-off is at least that of Proposition 2 with the
realized `u_i*`, for every `m`. In the Gaussian model with `u_i* = u_bar + xi` (predictable part `u_bar`, idiosyncratic
part `xi ~ N(0, Sigma_xi)` unknown to K_m) and the shadow-gradient direction `w = g_m` aligned with `u_bar` up to a
factor `kappa(m) <= 1`, K_m's statistic is `N(kappa(m) ||w|| ||u_bar|| / n, s_0^2 + w^T Sigma_xi w / n^2)` under IN and
`N(0, s_0^2)` under OUT (`s_0^2` the crowd-plus-noise variance of Proposition 2), a binormal pair with unequal
variances; its exact trade-off is computed numerically. `f_cmip_partial_knowledge.csv` shows it for an illustrative
alignment model `kappa(m) = (1 + 32/m)^{-1/2}`: the GDP-equivalent parameter at 5% FPR rises from 0.67 (`m = 1`) to
1.35 (`m = 32`) and 1.74 (`m -> inf`), below the crowd bound 1.77 and the GDP value 2.5. The alignment model and the
idiosyncratic share are assumptions, not measurements: no committed log records the target's update direction.

**Proposition 5 (first-order participation gap; D1, P2).** One round from `theta`; client `j` takes `K_j` local
gradient steps of size `eta` on its loss `L_j`, and `eta K_j` is small enough that its gradient `g_j = grad L_j(theta)`
is constant over the steps (`L_j` smooth; error `O((eta K_j)^2)` in the update), so `Delta_j = -eta K_j g_j`. The
target's loss gap `Gamma_i = L_i(theta + Delta_OUT) - L_i(theta + Delta_IN)` is, to first order,

- **renormalized weights** (aggregation over the participants, weights `w_j` summing to 1, renormalized when `i` is
  removed): `Delta_IN - Delta_OUT = w_i (Delta_i - Delta_bar_{-i})`, so
  `Gamma_i ≈ w_i eta < g_i, K_i g_i - (K g)_bar_{-i} >`, which for equal `K` is `w_i eta K < g_i, g_i - g_bar_{-i} >`;
  a client whose gradient equals the crowd's has no gap;
- **fixed public denominator** (the DP setting above, absent slots contribute zero): `Delta_IN - Delta_OUT = Delta_i / n`,
  so `Gamma_i ≈ eta K_i ||g_i||^2 / n`: the consensus part does **not** cancel; removing any client shrinks the step;
- **centered (residual) aggregation** `c_t + (1/n) sum_j (u_j - c_t)` with `c_t` the previous released update (D4
  item 2): `Delta_IN - Delta_OUT = (Delta_i - c_t) / n`, so with `c_t ≈ -eta K g_bar_prev`,
  `Gamma_i ≈ (eta K / n) < g_i, g_i - g_bar_prev >`: the cancellation of D1's argument returns under a fixed
  denominator.

*Proof.* `L_i(theta + x) = L_i(theta) + g_i^T x + O(||x||^2)`, so `Gamma_i ≈ -g_i^T (Delta_IN - Delta_OUT)`; substitute
the aggregate differences. ∎

The per-round signal-to-noise ratio of the loss statistic is `Gamma_i / (sigma ||g_i||)`, at most
`w_i eta K ||g_i - g_bar_{-i}|| / sigma` (renormalized) and `eta K ||g_i|| / (n sigma)` (fixed denominator).
`f_cmip_participation_gap.csv` checks the formula on a quadratic toy federation (8 clients, 20 dimensions, 5 local
steps): the relative error falls from 13.5% to 0.28% (renormalized) and from 24.5% to 0.79% (fixed denominator) as
`eta K` goes from 0.5 to 0.015. In the EuroSAT runs `eta K` is far from this regime (5 local epochs of Adam, up to 145
steps), which agrees with WP2's observation that the leave-one-out residual norm did not track leakage
(`results/cia_frontier/signature_suppression/FINDINGS.md`).

## 3. Runnable regimes (gate G3, D3 part)

Rule fixed in `STATE.md` before computing. For each regime and round, `rho_r^2 = mean(sigma^2 g_r^2) / V_r`, where `g_r`
is the shadow-gradient norm at the round's start model and `V_r` the variance of the shadow-loss increment across
federations without the target (pooled over targets around each target's mean over seeds 42-45). In WP1 the runs
contain the DP noise, so `V_r` includes it. In WP2 (vanilla, 48 clients) the frontier noise is added analytically:
`V_r` = crowd variance + `sigma^2 g_r^2`. Composed ratio `sqrt(mean_r rho_r^2)` (independent rounds, Proposition 3.1).

| Regime (EuroSAT unless stated, 50 rounds) | sigma | Composed ratio | GDP `mu_total` | Noise multiple for `mu_total = 1` | Ratio there (extrapolated) |
|---|---|---|---|---|---|
| WP1 3 clients, global DP | 0.00773 | 0.562 | 1524.6 | 1525 | 1.000 |
| WP1 5 clients, global DP | 0.00773 | 0.563 | 914.8 | 915 | 1.000 |
| **WP1 8 clients, global DP** | 0.00773 | **0.546** | 571.7 | 572 | 1.000 |
| WP1 Alzheimer 3 / 5 / 8 clients | 0.01667 | 0.790 / 0.759 / 0.619 | 707 / 424 / 265 | same | 1.000 |
| **WP2 48 clients, frontier noise** | 0.00773 | **0.619** | 95.3 | 95 | 1.000 |
| WP2 48 clients, 2x frontier noise | 0.01546 | 0.817 | 47.6 | 48 | 1.000 |
| WP2 48 clients, 4x frontier noise | 0.03092 | 0.924 | 23.8 | 24 | 1.000 |

![rho_r per round](figures/f_cmip_regimes.png)

**Gate G3 (D3 part): passes as written.** In the runnable regimes (EuroSAT, 8 and 48 clients) the partial-knowledge
bound for the loss-statistic attacker is 45% and 38% tighter than the GDP bound. **But in both regimes both bounds are
vacuous:** the composed GDP parameter is 572 and 95. The crowd term helps only while the DP noise is small against the
crowd's own variation. Doubling the noise already brings the 48-client ratio to 0.817, and at the noise that a useful
guarantee (`mu_total = 1`) would require, 95 to 1,525 times the frontier noise, the extrapolated ratio is 1.000 (crowd
variance held at its measured value; with large noise the trajectories differ more and the crowd variance would grow,
which this extrapolation does not capture).

Reading: D3's formal statement for partial-knowledge attackers is correct and measurable, and it explains why the
realistic attacker is far weaker than the GDP bound at the noise levels this repository uses. It does not give a useful
formal guarantee at practical noise, which matches the earlier GMIP calculation (`data/gmip_regimes.csv`: crowd privacy
needs `n` comparable to `d` for the full model).

## 4. Assumptions and limits

- Gaussian crowd model (central limit over the other clients); the population, and hence `m_B` and `Sigma_B`, known
  to the attacker. Non-exchangeable targets (the largest clients, evidence E4) break the population premise; for them
  the bound must be taken over the subpopulation the attacker can rule in, which this note does not compute.
- Linearization of the loss statistic (Proposition 2 is exact for the linear statistic; the loss increment is linear
  only to first order).
- `V_r` is estimated from 4 seeds per target, pooled over 5-8 targets (15-24 degrees of freedom); in WP1 the seed also
  changes the data layout (population variation), in WP2 it does not (training randomness only, an underestimate of the
  crowd variance and hence conservative for the ratio). The WP2 regime adds noise to vanilla trajectories, whose
  gradients differ from those of a noisy run.
- Composition assumes independent crowd terms across rounds; Proposition 3.2 shows that without an assumption the
  worst case is GDP.
- CIFAR-10 has no committed run with increment diagnostics; it is not evaluated.
