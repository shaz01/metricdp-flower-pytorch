package: WP3
status: passed
gate: G3 = pass. D3: composed partial-knowledge/GDP ratio 0.546 (EuroSAT 8 clients) and 0.619 (48 clients) at the frontier noise, where both bounds are vacuous (ratio 0.817 at 2x, 0.924 at 4x, 1.000 extrapolated to a mu_total = 1 noise level). D4.1: proof complete (private_dispersion.md Theorem 1), uv run pytest 624 passed / 2 skipped, median share 10.0%
protocol: agent_roadmap.md WP3 and G3 (theory package; no protocol runs, only smoke runs)
machine: compute server (the owner assigned WP3 to this agent in the Agent C role on 2026-10-15; the roadmap places WP3 on the owner PC, which this agent cannot reach; WP3 trains nothing beyond smoke runs)
cost so far: 0 t-h (no training budget; smoke runs only)
next: owner reads FINDINGS.md and decides on the merge; WP6 (residual clipping) can start
needs from others: none (WP0-WP2 are merged into master)

## Plan (fixed before any result is computed)

- D4.1 statistic: the median over participating clients of each client's distance to the current global model (the
  paper's mean layer-wise Euclidean distance), tracked privately with the geometric quantile update of Andrew et al.
  (2021): each client contributes one bit `1[s_i <= q_t]`, the server adds Gaussian noise to the bit count with a public
  slot count as denominator. Reason: a pairwise-distance median gives one client influence on `n - 1` of `n(n-1)/2`
  values, which makes smooth-sensitivity and PTR releases expensive at 3-48 clients; a per-client statistic has count
  sensitivity 1. The noise for round `t` uses `q_t` (a function of earlier private outputs) clamped to public bounds.
- G3 (D4.1) "release of the median costs at most 10% of the total privacy budget": measured as the share of the
  composed GDP parameter, `mu_median^2 / (mu_median^2 + mu_update^2)` (GDP composes by adding `mu^2`).
- G3 (D3) "at least 20% tighter": the GDP-equivalent parameter of the partial-knowledge bound is at most 0.8 times the
  GDP parameter `C / (n sigma)` of the same release, in at least one runnable regime (EuroSAT or CIFAR-10, 8-48 clients),
  with the regime parameters taken from committed logs only.

## G3 (D3) evaluation, fixed before any regime number is computed (2026-10-15)

- Attacker class: the one-dimensional shadow-loss statistic per round (`k = 1`, the repository's loss attacker and
  WP1's increment attacker), with the target's update known to the attacker (the crowd-only relaxation, which bounds
  every shadow-sample size `m`). Bound per round: `mu_PK,r <= rho_r C / (n sigma)` with
  `rho_r^2 = sigma^2 g_r^2 / V_r`, `g_r` the shadow-gradient norm, `V_r` the variance of the increment across
  federations without the target. GDP bound: `C / (n sigma)`. Composition: crowd terms independent across rounds,
  giving the composed ratio `sqrt(mean_r rho_r^2)` (the worst case over cross-round crowd correlation is GDP itself;
  stated in the note).
- Estimation: per cell and round, `V_r` pooled over the cell's OUT trajectories of all targets around each target's
  seed mean (WP1 seeds 42-45; WP2 vanilla seeds 42-45), `sigma^2 g_r^2` averaged over the same trajectories. Regimes:
  WP1 EuroSAT global DP with 3, 5 and 8 clients (8 is the runnable one; Alzheimer and 3-5 clients reported, not used for
  the gate), and EuroSAT 48 clients from WP2 vanilla trajectories with the frontier noise `sigma = 0.001546 x 5` added
  analytically (no DP run with increment diagnostics exists at 48 clients). CIFAR-10 has no committed run with the
  needed diagnostics.
- G3 (D3) passes if the composed ratio is at most 0.8 in at least one runnable regime (EuroSAT 8 or 48 clients).

## Log

| When | What |
|---|---|
| 2026-10-15 | Owner merged WP1 and WP2 into master and assigned WP3 to this agent (Agent C role; WP6, which needs WP3, follows). Branch `feature/private-dispersion` from master 5a9a2201. |
| 2026-10-15 | D4.1 delivered: `metricdp_pytorch/private_metric_strategy.py` (private median client distance via Andrew et al. quantile tracking, public slot denominator, public quantile bounds, default bit noise for a 10% share), `tests/test_private_metric_strategy.py` (10 tests), proof `private_dispersion.md` (Theorem 1: `mu_total = sqrt(T (q_max^2/z^2 + 1/sigma_b^2))`; median share exactly 10% by construction), checks `private_dispersion_checks.py`: one-round Monte Carlo matches `G_mu` within 0.003 in type II error; private-median tracking on logged update norms has a median relative error of 12% (48 clients, mu_total 1), 3.5% (48 clients, mu_total 4), 92% (8 clients, mu_total 1), 13% (8 clients, mu_total 4). |
| 2026-10-15 | Smoke (compute server, CPU, scratch output): `smoke_private_median.py`, 3 Alzheimer clients, 3 rounds, capped data, real Flower path with the strategy swapped in at the server; completed, noise std `z C / (n q_t)` = 0.0167 in round 1, the quantile moved from 1.0 towards the true median distance (0.25), median share 10%. |
| 2026-10-15 | D3 delivered: `f_cmip.md` (Propositions 1-5 with proofs) and `f_cmip_checks.py` (Monte Carlo checks, temporal cases, shadow-size illustration, participation-gap toy check, regime evaluation from committed WP1/WP2 logs by the rule fixed above). G3 evaluated: pass on both parts. `FINDINGS.md` written. Full suite on the branch: 624 passed, 2 skipped, 5 deselected. |
