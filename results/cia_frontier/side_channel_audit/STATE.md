package: WP1
status: planning (budget confirmed 2026-10-11; launching on the compute server)
gate: G1 = not evaluated (no protocol run yet)
protocol: results/cia_frontier/side_channel_audit/PROTOCOL.md @ f7076a58
machine: compute server (owner decision 2026-10-11; the roadmap planned the owner PC)
cost so far: 0 t-h of protocol runs; smoke tests only (owner PC, scratch output). Projection 50.6 t-h; budget 50.6 t-h (owner-confirmed, cap raised from 15 t-h)
next: launch one worker, time the first trajectory, re-estimate the cost (escalate if > 63.3 t-h = 50.6 x 1.25), then add workers
needs from others: none

## Owner decisions (2026-10-11)

- WP1 runs on the compute server, not the owner PC (measured on the owner PC: 67 s per EuroSAT round at 3 or 8
  clients, so the specified design is about 137 h there).
- Every non-homogeneous partition is label-Dirichlet; the 3-client Alzheimer federation uses Dirichlet alpha 0.3.
- Budget: the owner confirmed 50.6 t-h (3.4 times the 15 t-h cap) as written in PROTOCOL.md; recorded in
  ESCALATION.md, Decision section.
- GPU: the owner assigned GPU 1 of the compute server, shared with another user's running job; WP1 must not
  interrupt it or exceed the GPU's memory, so parallelism is sized to the free memory.

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
