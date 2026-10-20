package: B4 confirmatory run (owner decision 11)
status: done; FINDINGS.md written; the owner decides whether it is finished
gate: confirmation rule = confirmed at 20 and at 50 public images (13 of 13 settings)
protocol: results/cia_v2/formal_regime/confirm/PROTOCOL.md (fixed 2026-10-20, before any run)
machine: compute server (CPU only)
cost so far: 0.675 CPU-h (480 runs), no GPU
next: none until the owner decides
needs from others: none (B4's Part A tuning selections and the feature cache are reused unchanged)

Branch `feature/cia-v2-formal-confirm` from master 2bce957e. B4 itself (Agent C) is closed; this sub-package re-runs
its scarce-public exploration result on a fresh layout. Owner decision 11 assigned it to the agent that ran Agent A's
packages, since no other agent session is running.

## Log

| When | What |
|---|---|
| 2026-10-20 | Branch and sparse worktree opened (with `results/cia_v2/formal_regime/results/` for the tuning records). PROTOCOL.md and this STATE written before any code. |
| 2026-10-20 10:20 | Code: `confirm_public` stage in `experiments/frozen_head/runner.py`, `confirm.py`, `baseline_check.fits` parameterized by layout and sizes (defaults unchanged); 3 new tests, 25 frozen-head tests pass. 480 runs on 8 CPU workers (10:20-10:25). |
| 2026-10-20 10:40 | Analysis: **confirmed** at 20 (7 of 7) and 50 (6 of 6) public images against best_legit; gains within about 2.5 pp of the original. FINDINGS.md written. |
