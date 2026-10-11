package: WP0
status: planning
gate: G0 = not evaluated yet
protocol: none (WP0 is infrastructure, no runs; specification is agent_roadmap.md Section 2, WP0 items 1-4)
machine: owner PC
cost so far: 0 t-h of 0 (no training allowed)
next: implement items 1-4 with default-off flags and synthetic tests, then run `uv run pytest` for G0
needs from others: none

## Scope (agent_roadmap.md WP0)

1. Fixed noise standard deviation across removal for every runner the programme uses. The EuroSAT frontier runner
   already sets `noise_multiplier = ratio * active_clients` (sigma = ratio * C for global DP, ratio * C / d_t for metric
   privacy). The plan suite (`results/contest_at_scale/plan_suite/planned_runs.py`) keeps `z = 0.01` for 3 and 2 active
   clients; add a default-off `--fixed-noise-stdv` option that scales `z` by active / canonical clients.
2. Server-side per-round, per-client norm of the client update minus the leave-one-out mean update, before clipping:
   generic default-off strategy wrapper in `metricdp_pytorch/dp_diagnostics.py`, so it also works for vanilla runs
   (WP2 needs it without DP).
3. Attack-side per-round scalars for each target shadow sample: loss before and after each round's update (the
   increment) and the shadow-gradient norm at the round's start model (`experiments/cia/cia.py`, frontier runner).
4. Default-off flags, synthetic unit tests, `uv run pytest` green.

## Gate G0 (copied from agent_roadmap.md Section 3.2)

| Gate | Pass | On pass | On fail |
|---|---|---|---|
| G0 | `uv run pytest` green; synthetic tests show the fixed-noise option gives identical noise IN/OUT and the logged scalars match hand computations | Push; set `STATE.md` to `ready` | Fix; do not start WP1 |

## Notes for other agents

- This file lives at `results/cia_frontier/cia_infra/STATE.md` on `feature/cia-infra` (the roadmap names no WP0 directory).
  Read it with `git fetch origin && git show origin/feature/cia-infra:results/cia_frontier/cia_infra/STATE.md`.
- WP0 makes small additive, default-off edits to two files no agent owns: `experiments/reproduce/runner.py`
  (accept checkpoint round 0, new `--log-loo-residuals` flag), `experiments/reproduce/server.py` (apply the wrapper when
  the flag is set). Agent B will plumb its own flags through the same files; expect trivial merge conflicts there.
