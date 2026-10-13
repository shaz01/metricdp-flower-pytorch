package: WP2
status: running
gate: G2 = not yet measured
protocol: results/cia_frontier/signature_suppression/PROTOCOL.md @ bb9bfd44
machine: compute server (smoke tests on the compute server too, at most 5 rounds, scratch output; the owner assigned WP2 to this agent on 2026-10-14)
cost so far: 0 t-h of 30
next: launch the 60 protocol trajectories on the compute server; record the first wall times and re-estimate the cost
needs from others: none (feature/cia-infra merged, its STATE.md says ready)

## Log

| When | What |
|---|---|
| 2026-10-14 | Owner chose to start WP2 after WP1 finished; the agent that ran WP0/WP1 takes the Agent B role for WP2. Branch `feature/signature-suppression` from `master` (c2808d50), `feature/cia-infra` (dc409354) merged (fast-forward) for the WP0 diagnostics. |
| 2026-10-14 | Code (default-off runner options `--proximal-mu`, `--proximal-normalization`, `--consensus-kl-beta`, `--local-steps`; strategy factory `proximal_mu`), WP2 runner, `analyze.py`, synthetic tests (16 in the package); `uv run pytest`: 589 passed, 2 skipped, 5 deselected (before the analysis tests were added). PROTOCOL.md committed at bb9bfd44 before any run. |
| 2026-10-14 | Smoke (compute server, 2 rounds, scratch output, GPUs 0 and 1, released after): every configuration's IN trajectory plus `vanilla` and `fedprox-mu-1` end to end (20 trajectories, all complete, no errors). Each run JSON records its options (`proximal_mu`/`proximal_normalization`, `consensus_kl_beta`, `local_steps`, `weight_decay`, `local_epochs`, weighting); round-1 training losses and round-2 accuracies differ from vanilla for every option, so each reaches the clients. Wall time per 2-round trajectory: 44-62 s training (process start-up included), 22 s evaluation for 5 targets. `analyze.py` ran on the `vanilla`/`fedprox-mu-1` pair (5 clusters; leave-one-out residuals and per-client accuracies read). Smoke numbers are not results. |
