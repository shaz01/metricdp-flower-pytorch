package: WP6
status: passed
gate: G6 = pass as written (residual-median: top-decile score reduction 0.044 [0.012, 0.080], accuracy +0.09 pp; residual-q90: 0.092 [0.032, 0.152], -0.41 pp), but through 1.41x / 1.70x more noise (private rho 6.69 / 8.09 > C = 5), not through lower sensitivity; centring alone 0.020 [-0.016, 0.056]
protocol: results/cia_frontier/residual_clipping/PROTOCOL.md (committed with this STATE update, before any run)
machine: compute server (owner assigned WP6 to this agent on 2026-10-15, after WP3 passed G3)
cost so far: 7.08 h of trajectory wall time of the 60 t-h cap
next: owner reads FINDINGS.md; optional control (plain clipping with C = measured rho at the same z) needs an owner decision
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
| 2026-10-15 22:14-22:30 | Smoke (compute server, GPU 0, 2 rounds, scratch output, claim released): every mechanism's IN trajectory plus `plain` and `residual-median` end to end (14 trajectories, no errors); run JSONs record the residual settings; `analyze.py` ran on the pair. Round 2 already shows the predicted feedback: centre norm 4.24-4.46 (the previous release, mostly noise), median residual 5.85-6.02 > C, 98-100% of residuals clipped in the `previous`-centre mechanisms; plain clipping 21% clipped, noise std 0.00773 as planned. `uv run pytest` on the branch: 631 passed, 2 skipped. Smoke numbers are not results. |
| 2026-10-15 22:30-22:34 | Protocol run launched on GPU 0 at 22:30, then stopped about two minutes later (by PID, own processes only) because another user's job had started on GPU 0 during the smoke tests (claims convention: busy GPUs are off-limits); both GPUs were busy. Owner decision: share GPU 0 with that job. Relaunched 22:34: two workers on GPU 0 (8 parallel client actors each); partial trajectories from the first launch are retrained from scratch by the resumable runner. |
| 2026-10-16 02:17 | All 24 trajectories complete (two workers on GPU 0, shared with another user's job by owner decision; no stalls or errors); claim released. |
| 2026-10-16 | Analysis (`analysis.json`): G6 pass as written; mechanism diagnostics show `rho` above C and noise 1.41x / 1.70x plain, matching the pre-run prediction for the mechanism; the predicted gate fail did not happen because the extra noise cost no accuracy at this vacuous noise level. Descriptive decomposition added to `analyze.py` (each mechanism against plain, with noise and `rho` ratios). `FINDINGS.md`, figures and tables written. |
