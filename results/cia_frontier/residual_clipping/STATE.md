package: WP6
status: planning
gate: G6 = not yet measured
protocol: results/cia_frontier/residual_clipping/PROTOCOL.md (committed with this STATE update, before any run)
machine: compute server (owner assigned WP6 to this agent on 2026-10-15, after WP3 passed G3)
cost so far: 0 t-h of 60
next: smoke tests, then the 24 protocol trajectories
needs from others: none (feature/private-dispersion merged for the private quantile)

## Pre-analysis from committed logs (no training; `pre_analysis.py`, `pre_analysis.csv`)

Residual clipping releases `c_t + (1/n) sum_i clip(u_i - c_t, rho) + noise` with `c_t` the previous released update. It
lowers the sensitivity from `C` to `rho` only if residuals are much smaller than updates. Lower bound on the residual
norm (round mean spread plus the previous round's noise; the round-to-round change of the mean is not logged):

| Runs | Median update norm | Weighted spread around the mean update | Previous noise norm | Residual lower bound | / update | / C |
|---|---|---|---|---|---|---|
| Frontier, 48 clients, global DP, seeds 42-44 | 3.07-3.14 | 3.31-3.42 | 4.16 | 5.31-5.39 | 1.72-1.73 | 1.06-1.08 |
| WP1, 8 clients, global DP, seeds 42-45 | 5.00 (clipped) | 4.34-4.61 | 4.16 | 6.01-6.21 | 1.20-1.24 | 1.20-1.24 |

Reading: in label-skewed EuroSAT (alpha 0.3) client updates are not concentrated around the consensus (spread larger than
the update norm itself), and the noise inherited through `c_t` (`sqrt(d) sigma` = 4.16 at the frontier noise) is of the
order of the clipping norm. With a median-tracked `rho`, `rho` would settle above `C`.

## Log

| When | What |
|---|---|
| 2026-10-15 | Owner asked to start WP6. Branch `feature/residual-clipping` from master 5a9a2201 with `feature/private-dispersion` (22bf9d17, WP3 passed) merged. Pre-analysis above. |
| 2026-10-15 | Owner decision: run the screen as specified despite the pre-analysis. Strategy `metricdp_pytorch/residual_clipping_strategy.py` (+ 7 tests; runner/server plumbing `--privacy residual-clipping`), `THEORY.md` (proof: `sqrt(T (1/z^2 + 1/sigma_b^2))`-GDP; noise-feedback prediction: fail), `PROTOCOL.md`, `runner.py` (24 trajectories, 8.9 t-h planned), `analyze.py`. |
