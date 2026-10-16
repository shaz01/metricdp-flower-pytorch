# Residual clipping around the previous release (D4 item 2): privacy proof and noise-feedback analysis

WP6, 2026-10-15, written before any protocol run. Implementation: `metricdp_pytorch/residual_clipping_strategy.py`;
tests: `tests/test_residual_clipping_strategy.py`. Notation and lemmas as in
`research/cia_review_2026-10/theory/private_dispersion.md` (Dong, Roth and Su 2022).

## 1. Mechanism

Public: slot count `n`, noise multiplier `z`, centre rule (`previous` or `zero`), and either a fixed `rho` or the
private-quantile parameters (target quantile `gamma`, step `eta`, bounds, bit noise `sigma_b`, initial `rho_1`).
Adjacency: add or remove one client in one slot (an empty slot contributes nothing).

Round `t`, given the public state `(theta_t, c_t, rho_t)` (`c_1 = 0`):

1. Each present client returns `theta_i`; `u_i = theta_i - theta_t`, residual `r_i = u_i - c_t`.
2. Release `Theta_t = theta_t + c_t + (1/n) sum_i clip(r_i, rho_t) + N(0, sigma_t^2 I)`, `sigma_t = z rho_t / n`.
3. Private quantile (if used): `B_t = sum_i 1[||r_i|| <= rho_t] + N(0, sigma_b^2)`,
   `rho_{t+1} = clamp(rho_t exp(-eta (B_t / n - gamma)))`.
4. Public updates: `theta_{t+1} = Theta_t`, `c_{t+1} = Theta_t - theta_t` (centre `previous`) or `0` (centre `zero`).

With centre `zero` and fixed `rho = C`, this is plain clipping with the fixed slot denominator: the baseline.

## 2. Privacy

**Lemma 1 (one round).** For a fixed public state, `Theta_t` is `(1/z)`-GDP and `B_t` is `(1/sigma_b)`-GDP, so the round
is `sqrt(1/z^2 + 1/sigma_b^2)`-GDP (`1/z` with a fixed `rho`).

*Proof.* `c_t` and `rho_t` are functions of earlier releases, hence fixed by the state. Adding or removing the client in
one slot changes only that slot's term `clip(r_i, rho_t)`, whose norm is at most `rho_t`: by the triangle inequality the
sum changes by at most `rho_t` in L2, and the average by `rho_t / n`. Gaussian noise with `sigma_t = z rho_t / n` gives
`(rho_t / n) / sigma_t = 1/z` (Gaussian mechanism, exactly, for every `rho_t`; no noise floor is needed because the
noise scales with the state-determined `rho_t`). The bit count changes by at most 1, so `B_t` is `(1/sigma_b)`-GDP. The
noises are independent; GDP composition adds `mu^2`. ∎

**Theorem 1.** `T` rounds are `sqrt(T (1/z^2 + 1/sigma_b^2))`-GDP (adaptive composition, Dong et al. Theorem 3.2: the
round bound holds for every state).

**Same guarantee as plain clipping.** Plain clipping with `z_plain` is `sqrt(T)/z_plain`-GDP. Residual clipping with a
private `rho` and the quantile release at a share `s` of `mu^2` matches it with `z = z_plain / sqrt(1 - s)` and
`sigma_b = z_plain / sqrt(s)` (WP6 uses `s = 0.1`, as WP3). Its noise std relative to plain is then
`sigma_t / sigma_plain = rho_t / (C sqrt(1 - s))`: **residual clipping adds less noise only while `rho_t < 0.949 C`.**

## 3. Noise feedback through the centre (prediction, stated before the runs)

`c_t` is the previous release, so it contains the previous round's noise `Z_{t-1}`, with
`||Z_{t-1}|| ≈ sqrt(d) sigma_{t-1} = kappa rho_{t-1}`, `kappa = sqrt(d) z / n`. Every client's residual contains
`-Z_{t-1}`. Ignoring cross terms (independent noise in high dimension):

    ||r_i||^2 ≈ s_i^2 + D_t^2 + kappa^2 rho_{t-1}^2,

with `s_i` the client's spread around the round's mean update and `D_t` the change of the mean update between rounds. A
quantile-tracked `rho` settles where `rho^2 ≈ s~^2 + D^2 + kappa^2 rho^2`, i.e.

    rho ≈ sqrt(s~^2 + D^2) / sqrt(1 - kappa^2)   if kappa < 1;  otherwise rho grows to its upper bound.

`kappa < 1` needs `z < n / sqrt(d)`, i.e. a per-round GDP parameter `1/z > sqrt(d)/n`: **the centre's noise can be
absorbed only at vacuous privacy levels** (`sqrt(d)/n` = 11.2 for EuroSAT with `d` = 289,194 and `n` = 48). In this
screen, `z` = 0.0782 gives `kappa` = 0.876 and an amplification `1/sqrt(1 - kappa^2)` = 2.07. The committed frontier
logs give a median weighted spread of 3.31-3.42 (`pre_analysis.csv`), so the prediction is `rho` ≈ 7 for the
median-tracked variant (above `C` = 5) and noise about 1.5 times that of plain clipping. The fixed-`rho` variant keeps
plain clipping's noise but clips most residuals (predicted residual norm about 5.4 > 5, part of it the inherited noise).
**Predicted G6 outcome: fail (no sensitivity gain; extra noise or extra clipping bias), unless the measured residuals
are much smaller than these lower-bound estimates.**

The fix suggested by the analysis, a denoised centre (an average of past releases), trades noise in `c_t` for lag
behind the current update. It is outside the roadmap's design and is not part of this screen.
