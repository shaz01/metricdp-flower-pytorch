package: D1 follow-up, consensus KL (owner request 2026-10-16; not a roadmap work package)
status: closed (owner decision 2026-10-16; step 2 not run)
gate: none (step 1 recomputes WP2's G2 rule for consensus-kl-0.3 over seeds 42-45; WP2's verdict stands)
protocol: results/cia_frontier/consensus_kl/PROTOCOL.md (step 1 fixed before its runs)
machine: compute server
cost so far: 16.51 t-h (step 1 5.60, step 3 10.91)
next: none; merge into master is the owner's call
needs from others: none

## Log

| When | What |
|---|---|
| 2026-10-16 | Owner asked to look deeper at consensus KL. Exploratory look at the committed seed-42 WP2 runs (no training): at matched accuracy consensus KL 0.3 leaks less than vanilla and weight decay (accuracy bin 0.75-0.78: mean gap +0.003 vs +0.040 and +0.014; bin 0.78-0.82: -0.012 vs +0.035 and +0.007); its accuracy cost concentrates on a few clients (lowest client 0.545 vs 0.611, client 3 0.619 vs 0.746). Owner chose: step 1 more seeds, then step 3 under DP, then step 2 smaller beta. Branch `feature/consensus-kl` from master 5a9a2201. |
| 2026-10-16 09:55 | Step 1 launched: 18 trajectories, two workers on GPU 1 (idle, no other user's job), worker loop `~/cia-server/ckl-worker.sh`. Step 3 protocol and runner (`runner_dp.py`, 36 trajectories, 13.3 t-h) fixed before its runs. |
| 2026-10-16 | Step 3 smoke (GPU 1, 2 rounds, IN only, ratio 4x, scratch output): both mechanisms complete; noise std 0.03092 = 4 x 0.00773, consensus KL beta 0.3 recorded and active (round-2 training loss 1.407 vs 1.105), equal weights. `analyze_dp.py` checked on synthetic cells. A chain launcher starts step 3 on GPU 1 when step 1 ends, if no other user's job is on the GPU. |
| 2026-10-16 12:55 | Step 1 done: 18 trajectories, 5.60 h, no stalls or errors (commit 3af4d2d2). Chain launcher started step 3 on GPU 1 (no other user's job). |
| 2026-10-16 19:00 | Step 3 done: 36 trajectories, 10.91 h, no stalls or errors (commit 1e0569d5). GPU 1 claim released. |
| 2026-10-16 | Step 1 result (`analysis_step1.json`, seeds 42-45, 20 clusters): the seed-42 reduction does not replicate. Clean-view score consensus-kl-0.3 0.795 [0.712, 0.869] vs vanilla 0.762 [0.659, 0.854]: reduction -0.033 [-0.112, +0.048] (AUC -0.049 [-0.142, +0.038]); per seed 42/43/44/45 consensus KL 0.668/0.848/0.844/0.820 vs vanilla 0.816/0.808/0.632/0.792. Accuracy 0.8018 vs 0.8276 (-2.58 pp, fails the 1 pp condition). Against weight decay -0.032 [-0.111, +0.048], against local-epochs-1 -0.017 [-0.088, +0.057]. Exploratory matched accuracy (`explore.py`, seeds 42-45): mean gap in IN-accuracy bins 0.75-0.78 / 0.78-0.82 is 0.020 / 0.023 (consensus KL) vs 0.032 / 0.030 (vanilla), 0.027 / 0.028 (weight decay), 0.026 / 0.022 (fixed steps): differences of about 0.01, far from the seed-42 picture. Lowest-client IN accuracy per seed 0.545/0.576/0.619/0.639 vs vanilla 0.611/0.624/0.639/0.636. |
| 2026-10-16 | Step 3 result (`analysis_step3.json`). The pre-declared rule returns **"supports consensus KL"** (at S* = 0.792, the gdp 4x score: Delta +0.094 [+0.008, +0.098]), but the rule's premise fails: gdp's clean-view score does not fall with noise over 1-4x (0.788, 0.808, 0.792), so matching on score pairs consensus KL at about 2x noise (accuracy 0.748) with gdp at 4x (0.657). At S* = 0.788 (gdp 1x) the same rule gives Delta -0.041 [-0.138, -0.029], an interval below zero. Matched noise (secondary): consensus KL costs accuracy at every level (-2.46, -1.11, -0.37 pp) and changes the score by -0.088 [-0.164, -0.012] (more leakage) at 1x, +0.032 [-0.028, +0.092] at 2x, +0.028 [-0.072, +0.088] at 4x. The rule's reading is recorded as returned; no extension (the rule calls for one only when inconclusive). Substantive reading, for the owner: no evidence that consensus KL helps under global DP in this range. |
| 2026-10-16 | Owner decision: skip step 2 and close the consensus-KL line. `FINDINGS.md` written. |
