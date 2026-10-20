package: B4 confirmatory run (owner decision 11)
status: protocol fixed; code next
gate: confirmation rule (PROTOCOL.md Section 4) = not evaluated yet
protocol: results/cia_v2/formal_regime/confirm/PROTOCOL.md (fixed 2026-10-20, before any run)
machine: compute server (CPU only)
cost so far: 0 CPU-h (estimate under 1 CPU-hour)
next: `confirm_public` stage and `confirm.py` with tests; run; analysis
needs from others: none (B4's Part A tuning selections and the feature cache are reused unchanged)

Branch `feature/cia-v2-formal-confirm` from master 2bce957e. B4 itself (Agent C) is closed; this sub-package re-runs
its scarce-public exploration result on a fresh layout. Owner decision 11 assigned it to the agent that ran Agent A's
packages, since no other agent session is running.

## Log

| When | What |
|---|---|
| 2026-10-20 | Branch and sparse worktree opened (with `results/cia_v2/formal_regime/results/` for the tuning records). PROTOCOL.md and this STATE written before any code. |
