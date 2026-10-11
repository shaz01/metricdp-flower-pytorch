package: WP1
status: blocked (measured cost projects to 65-71 h vs the 63.3 h escalation threshold; see ESCALATION.md, Escalation 2)
gate: G1 = not evaluated (2 of 228 trajectories)
protocol: results/cia_frontier/side_channel_audit/PROTOCOL.md @ f7076a58
machine: compute server (owner decision 2026-10-11; the roadmap planned the owner PC)
cost so far: 2 of 228 trajectories complete (about 1.2 h of trajectory wall time). Plan 50.6 t-h; owner budget 50.6 t-h; re-estimate from the first two trajectories 65.0-70.6 h
next: wait for the owner's decision on ESCALATION.md Escalation 2, then resume with the same command (completed trajectories are skipped)
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

| When (2026-10-11, +03:00) | Event |
|---|---|
| 17:30 | Worktree `feature/cia-audit` (sparse), shared main venv; `uv run pytest`: 596 passed, 2 skipped (MPS only), 5 deselected; plan: 228 trajectories, 50.62 t-h |
| 17:38 | Worker 0 on GPU 1 (`CUDA_VISIBLE_DEVICES=1`, `--max-parallel-clients 8`); GPU 1 shared with another user's job (12.4 GB) |
| 17:46-17:56 | Worker 1 on GPU 1 (owner asked for more parallelism; two workers is the most GPU 1's free memory allows) |
| 18:14 | First trajectory complete (EuroSAT, 3 clients, global DP, seed 42, IN): training 1,475.0 s, evaluation 696.2 s = 0.603 h vs 0.37 h planned. First re-estimate counted every target in every trajectory (109 h); workers stopped |
| 18:22 | Worker 1's trajectory complete (OUT-0, seed 42): training 1,461.6 s, evaluation 143.7 s (OUT scores only its own target). Corrected re-estimate 65.0-70.6 h > 63.3 h: Escalation 2 raised, worker 1 stopped |

Measured footprint per worker (3 clients): GPU about 0.43 GB server + 0.5-0.73 GB per client actor; RAM about 1.45 GB
per client actor. For resumption on the shared GPU 1: 8-client cells with `--max-parallel-clients 4` per worker so two
workers hold at most 8 client actors (about 9 GB); this only limits concurrent Ray actors and does not change what is
computed (aggregation order is deterministic).

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
