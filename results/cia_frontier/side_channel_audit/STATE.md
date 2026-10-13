package: WP1
status: running: G1(b) extension (owner chose Escalation 3 option 1: seed 45, all 12 cells; budget 101 h, escalate above 126 h)
gate: G1(a) = 0.838 [0.779, 0.902] (pass; attributed to the metric-privacy noise level); G1(b) = +0.074 [-0.060, +0.116] (inconclusive), seeds 42-44
protocol: results/cia_frontier/side_channel_audit/PROTOCOL.md @ f7076a58
machine: compute server (owner decision 2026-10-11; the roadmap planned the owner PC)
cost: 228 of 228 trajectories complete, 75.5 h of trajectory wall time (sum of training + evaluation seconds in their complete.json; work lost to restarts not counted): above the 71 h budget, below the 89 h escalation line. Mean per trajectory: EuroSAT 1,881 / 1,725 / 1,478 s and Alzheimer 658 / 729 / 790 s at 3 / 5 / 8 clients
next: run the 76 seed-45 trajectories; recompute G1(b) at seeds 42-45; FINDINGS.md
needs from others: none

## Owner decisions (2026-10-11)

- WP1 runs on the compute server, not the owner PC (measured on the owner PC: 67 s per EuroSAT round at 3 or 8
  clients, so the specified design is about 137 h there).
- Every non-homogeneous partition is label-Dirichlet; the 3-client Alzheimer federation uses Dirichlet alpha 0.3.
- Budget: the owner confirmed 50.6 t-h (3.4 times the 15 t-h cap) as written in PROTOCOL.md; recorded in
  ESCALATION.md, Decision section.
- GPU: the owner assigned GPU 1 of the compute server, shared with another user's running job; WP1 must not
  interrupt it or exceed the GPU's memory, so parallelism is sized to the free memory.

## Compute-server execution log

| When (2026-10-11 unless dated, +03:00) | Event |
|---|---|
| 17:30 | Worktree `feature/cia-audit` (sparse), shared main venv; `uv run pytest`: 596 passed, 2 skipped (MPS only), 5 deselected; plan: 228 trajectories, 50.62 t-h |
| 17:38 | Worker 0 on GPU 1 (`CUDA_VISIBLE_DEVICES=1`, `--max-parallel-clients 8`); GPU 1 shared with another user's job (12.4 GB) |
| 17:46-17:56 | Worker 1 on GPU 1 (owner asked for more parallelism; two workers is the most GPU 1's free memory allows) |
| 18:14 | First trajectory complete (EuroSAT, 3 clients, global DP, seed 42, IN): training 1,475.0 s, evaluation 696.2 s = 0.603 h vs 0.37 h planned. First re-estimate counted every target in every trajectory (109 h); workers stopped |
| 18:22 | Worker 1's trajectory complete (OUT-0, seed 42): training 1,461.6 s, evaluation 143.7 s (OUT scores only its own target). Corrected re-estimate 65.0-70.6 h > 63.3 h: Escalation 2 raised, worker 1 stopped |
| 18:30 | Owner chose option 1 (budget 71 h). Resumed: workers 0 and 1 on GPU 1, each under a rerun loop in tmux (`wp1-w0`, `wp1-w1`): every cycle runs the 3- and 5-client cells with `--max-parallel-clients 8`, then the 8-client cells with `--max-parallel-clients 4`, until the plan reports 228/228; the interrupted OUT-1 trajectory is retrained from scratch |
| 19:33 | Another user started a second job on GPU 1 (8.2 GB at start; its twin on GPU 0 uses 14 GB), leaving too little headroom for two WP1 workers without risking the other user's jobs. Worker 1 stopped at a trajectory boundary (nothing lost); one worker continues. 5 of 228 complete |
| 20:15 | The other user's second job grew to 14.0 GB (GPU 1: 26.4 GB used by other users, 32.7 GB total). Worker 0 restarted at a trajectory boundary with `--max-parallel-clients 3` for every cell (about 2.6 GB for WP1; execution setting only, results unchanged); the limit is now read before every pass from a config file on the server. 7 of 228 complete |
| 21:48 | Owner assigned GPU 0 as well (one of the other user's jobs there had ended; 16.3 GB free). Worker 1 started on GPU 0 under the same loop (tmux `wp1-w1`, at most 3 parallel clients, about 2.6 GB EuroSAT / about 3.5 GB Alzheimer estimated). Two workers: GPU 1 (worker 0) and GPU 0 (worker 1). 9 of 228 complete |
| 22:32 | The other user's MOTR job on GPU 1 grew to 16.0 GB, leaving 1.3 GB free beside worker 0. Worker 0 stopped immediately (its in-progress trajectory restarts from scratch) and moved to GPU 0 (14.3 GB free there). Both workers now on GPU 0, nothing of WP1 on GPU 1. 12 of 228 complete |
| 2026-10-12 19:10 | 132 of 228 complete: every 3- and 5-client cell of both datasets, 13 of 27 EuroSAT 8-client global-DP trajectories. Alzheimer cost check (after the first Alzheimer trajectories): 3-client 565 s training + 95 s evaluation, 5-client 645 s + 83 s, i.e. about 2.5 times the plan's 0.2 relative cost, but EuroSAT runs at about 1.2 times its plan, so the total projects to 78 h, inside the 89 h line. Both GPUs now free of other users' jobs |
| 2026-10-12 19:11 | Owner asked for one more worker. Worker 2 started on GPU 1 (idle: no other user's compute job) under the same loop (tmux `wp1-w2`, at most 3 parallel clients). Three workers: two on GPU 0, one on GPU 1 |
| 2026-10-13 04:56 | 228 of 228 complete (last trajectory finished 04:39; the three worker loops exited on their own at 04:39-04:56). GPU claims released, worker sessions closed. No trajectory failed permanently; the rerun loop retrained the ones interrupted by worker moves |
| 2026-10-13 11:27 | `analyze.py` finished (70 min, one core): G1(a) pass, attributed to the metric-privacy noise level; G1(b) inconclusive. Figures and tables built by `findings.py` from `analysis.json` and the committed trajectories. Escalation 3 raised: the declared G1(b) extension (76 trajectories, about 25 h) would bring WP1 to about 101 h |
| 2026-10-13 11:30 | Owner chose Escalation 3 option 1. Extension launched: `--seeds 45`, 76 trajectories, four workers in tmux (`wp1-x0`..`wp1-x3`), two per GPU (both GPUs free of other users' jobs), at most 3 parallel clients each |

Measured footprint per worker (3 clients): GPU about 0.43 GB server + 0.5-0.73 GB per client actor; RAM about 1.45 GB
per client actor. The number of parallel client actors only limits concurrent Ray actors and does not change what is computed
(aggregation order is deterministic); it is set to fit the GPU memory other users leave free (see the log below).

## Ready for the server

| Piece | Where | Check |
|---|---|---|
| Runner (plan by default, shards, lock-and-skip workers, stall guard, completion count) | `runner.py` | `tests/test_runner.py` |
| Dirichlet Alzheimer module (min 125 records per client) | `data.py` | every client and shadow loader of seeds 42-44 built (smallest shadow sample 10 images) |
| Attackers and ROC tools (reusable for WP4) | `experiments/cia/increment_attack.py` | `experiments/cia/tests/test_increment_attack.py` (hand computations, Gaussian optimum, brute-force AUC) |
| Analysis and gate G1 (with the pre-run attribution rule) | `analyze.py` | `tests/test_analyze.py` (synthetic separable and null trees) |
| Frontier executor: dataset-agnostic, `releases.json` | `results/cia_frontier/eurosat_frontier/runner.py` | frontier tests |
| Suite | `uv run pytest` | 598 passed |

## Smoke tests (owner PC, at most 5 rounds, scratch output, not results)

- Timing, 3 rounds: EuroSAT 3 and 8 clients 67 and 66 s per round; Alzheimer 3 and 8 clients 12-15 s per round.
- End to end: Alzheimer, 3 clients, both mechanisms, seeds 42-43, 3 rounds: 16 trajectories, analysis ran.
- Found and fixed on the way: the frontier data module rejected `dirichlet_alpha` (runner unusable on master since
  2026-10-02); `run_one_combo` rejected checkpoint round 0 (WP0 diagnostics path); an empty Alzheimer shadow sample
  under the 10-record minimum; Ray start-up deadlocks (3 of about 17 trajectories on the owner PC: a forked child
  blocks in a Python at-fork hook and never execs) now caught by the stall guard, which kills the whole process tree
  and lets the next pass retrain the trajectory (verified: one stall killed, retrained in the second pass).
- The smoke showed the blind noise-level score also separates under global DP (update-norm channel); handled by the
  pre-run revision in PROTOCOL.md before any protocol run.
