package: WP3
status: planning
gate: G3 = not yet evaluated
protocol: agent_roadmap.md WP3 and G3 (theory package; no protocol runs, only smoke runs)
machine: compute server (the owner assigned WP3 to this agent in the Agent C role on 2026-10-15; the roadmap places WP3 on the owner PC, which this agent cannot reach; WP3 trains nothing beyond smoke runs)
cost so far: 0 t-h (no training budget; smoke runs only)
next: private median-distance strategy and tests, privacy proof, f-CMIP note, numerical checks, smoke run
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

## Log

| When | What |
|---|---|
| 2026-10-15 | Owner merged WP1 and WP2 into master and assigned WP3 to this agent (Agent C role; WP6, which needs WP3, follows). Branch `feature/private-dispersion` from master 5a9a2201. |
