package: D1 follow-up, consensus KL (owner request 2026-10-16; not a roadmap work package)
status: running
gate: none (step 1 recomputes WP2's G2 rule for consensus-kl-0.3 over seeds 42-45; WP2's verdict stands)
protocol: results/cia_frontier/consensus_kl/PROTOCOL.md (step 1 fixed before its runs)
machine: compute server
cost so far: 0 t-h
next: step 1 runs (18 trajectories) and analysis; step 3 (36 trajectories under global DP, protocol fixed) queued after step 1; then step 2
needs from others: none

## Log

| When | What |
|---|---|
| 2026-10-16 | Owner asked to look deeper at consensus KL. Exploratory look at the committed seed-42 WP2 runs (no training): at matched accuracy consensus KL 0.3 leaks less than vanilla and weight decay (accuracy bin 0.75-0.78: mean gap +0.003 vs +0.040 and +0.014; bin 0.78-0.82: -0.012 vs +0.035 and +0.007); its accuracy cost concentrates on a few clients (lowest client 0.545 vs 0.611, client 3 0.619 vs 0.746). Owner chose: step 1 more seeds, then step 3 under DP, then step 2 smaller beta. Branch `feature/consensus-kl` from master 5a9a2201. |
| 2026-10-16 09:55 | Step 1 launched: 18 trajectories, two workers on GPU 1 (idle, no other user's job), worker loop `~/cia-server/ckl-worker.sh`. Step 3 protocol and runner (`runner_dp.py`, 36 trajectories, 13.3 t-h) fixed before its runs. |
