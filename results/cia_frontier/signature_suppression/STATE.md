package: WP2
status: running
gate: G2 = inconclusive at seed 42 (no candidate holds both parts; fixed-steps open: score reduction +0.024 [-0.076, +0.124], threshold 0.05); extension running
protocol: results/cia_frontier/signature_suppression/PROTOCOL.md @ bb9bfd44
machine: compute server (smoke tests on the compute server too, at most 5 rounds, scratch output; the owner assigned WP2 to this agent on 2026-10-14)
cost so far: 15.5 h of trajectory wall time (60 seed-42 trajectories) of the 30 t-h cap; extension about 6.2 h more
next: finish the seed-43 extension (24 trajectories, PROTOCOL.md Amendment 1), run analyze.py --extension, apply G2, write FINDINGS.md
needs from others: none (feature/cia-infra merged, its STATE.md says ready)

## Log

| When | What |
|---|---|
| 2026-10-14 | Owner chose to start WP2 after WP1 finished; the agent that ran WP0/WP1 takes the Agent B role for WP2. Branch `feature/signature-suppression` from `master` (c2808d50), `feature/cia-infra` (dc409354) merged (fast-forward) for the WP0 diagnostics. |
| 2026-10-14 | Code (default-off runner options `--proximal-mu`, `--proximal-normalization`, `--consensus-kl-beta`, `--local-steps`; strategy factory `proximal_mu`), WP2 runner, `analyze.py`, synthetic tests (16 in the package); `uv run pytest`: 589 passed, 2 skipped, 5 deselected (before the analysis tests were added). PROTOCOL.md committed at bb9bfd44 before any run. |
| 2026-10-14 | Smoke (compute server, 2 rounds, scratch output, GPUs 0 and 1, released after): every configuration's IN trajectory plus `vanilla` and `fedprox-mu-1` end to end (20 trajectories, all complete, no errors). Each run JSON records its options (`proximal_mu`/`proximal_normalization`, `consensus_kl_beta`, `local_steps`, `weight_decay`, `local_epochs`, weighting); round-1 training losses and round-2 accuracies differ from vanilla for every option, so each reaches the clients. Wall time per 2-round trajectory: 44-62 s training (process start-up included), 22 s evaluation for 5 targets. `analyze.py` ran on the `vanilla`/`fedprox-mu-1` pair (5 clusters; leave-one-out residuals and per-client accuracies read). Smoke numbers are not results. |
| 2026-10-14 00:50 | Protocol run launched on the compute server: three workers of the resumable runner (two on GPU 0 from 00:50 and 00:55, one on GPU 1 from 00:50), 8 parallel client actors each, a worker loop that reruns the runner until the plan reports 60/60. About 11-12 GB of host memory per worker; a fourth worker was not started to leave memory for the server's other users. |
| 2026-10-14 01:10 | Cost check after the first three trajectories (`vanilla`, three workers sharing two GPUs): IN 697 s training + 380 s evaluation (5 targets) = 0.30 h; OUT-23 767 + 92 s, OUT-7 798 + 90 s = 0.24-0.25 h. Projection for 60 trajectories: 10 x (0.30 + 5 x 0.245) = about 15.3 h of trajectory wall time against 22.2 t-h planned and the 30 t-h cap; no escalation needed. |
| 2026-10-14 06:20 | All 60 trajectories complete (workers exited by themselves at 06:10-06:20; no stalls, no errors). Measured cost: 15.54 h (training + evaluation in every complete.json); per configuration 1.50-1.73 h, `local-epochs-1` 0.82 h. GPU claims released. |
| 2026-10-14 17:20 | Seed-42 analysis (`analysis.json`, 5 clusters). Vanilla (equal weights): clean score 0.816, AUC 0.898, accuracy 0.816. Score reduction vs vanilla `D_S` [95%]: fedprox-mu-0.01 -0.104 [-0.172, -0.036] (fails), fedprox-mu-0.1 +0.036 (accuracy -2.7 pp, fails), fedprox-mu-1 +0.124 (accuracy -10.2 pp, fails), consensus-kl-1 +0.012 (accuracy -6.6 pp, fails), consensus-kl-0.3 **+0.148 [+0.048, +0.224]** with AUC +0.170 [+0.065, +0.238] and part 2 holding, but accuracy 0.796 is 1.95 pp below vanilla (fails the 1 pp condition), fixed-steps +0.024 [-0.076, +0.124] at accuracy 0.814 (**open**). G2 inconclusive -> declared extension (Amendment 1). |
