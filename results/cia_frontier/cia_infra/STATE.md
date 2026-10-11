package: WP0
status: ready
gate: G0 = pass (`uv run pytest`: 572 passed, 5 deselected; fixed-noise IN/OUT and hand-computed scalar tests below)
protocol: none (WP0 is infrastructure, no runs; specification is agent_roadmap.md Section 2, WP0 items 1-4)
machine: owner PC
cost so far: 0 t-h of 0 (no training; one 1-round `--smoke` integration check of the Flower path, 3 clients, capped data)
next: Agent A starts WP1 on feature/cia-audit only after the owner answers ESCALATION.md (MPS data-copy bug blocks owner-PC training)
needs from others: owner decision in ESCALATION.md (fix touches Agent B's `experiments/reproduce/paper_training.py`)

## What WP0 delivers (all default-off; with the flags off, run names, manifests and result keys are unchanged)

| Item | How to use it | Code | Tests |
|---|---|---|---|
| 1. Fixed noise std across removal | EuroSAT frontier runner: already fixed (`noise_multiplier = ratio * active clients`, so sigma = ratio * C for global DP and ratio * C / d_t for metric privacy). Plan suite: `--fixed-noise-stdv` scales z by active / 3 clients and writes to `results/contest_at_scale/plan_suite/results/fixed_noise_stdv/` | `results/contest_at_scale/plan_suite/planned_runs.py` (`fixed_stdv_noise_multiplier`, `cia_groups`) | `tests/test_fixed_noise_stdv.py`: the real global-DP and metric-privacy wrappers draw bit-identical noise IN and OUT from the same RNG state (plan suite 3 vs 2 clients; frontier runner 3, 5, 8, 48 clients); the default keeps the 3/2 std ratio |
| 2. Leave-one-out residual norm, server side, before clipping | `experiments.reproduce.runner --log-loo-residuals`, or `build_combos(..., diagnostics=True)` / frontier runner `--diagnostics`. Train metrics per round gain `loo-client-ids` and `loo-residual-norms` = norm of `u_i - mean_{j != i} u_j` (equal-weight mean, raw updates) | `metricdp_pytorch/dp_diagnostics.py` (`leave_one_out_residual_diagnostics`, `LeaveOneOutResidualLogging`, a wrapper outside any DP wrapper, so it also works for vanilla); `experiments/reproduce/server.py`, `runner.py` | `tests/test_dp_diagnostics.py` (3-client hand computation, errors skipped, measured before in-place clipping), `experiments/reproduce/tests/` |
| 3. Attack-side per-round scalars | Frontier runner `--diagnostics`: checkpoints round 0 too; each `measurements.json` row gains `{clean,noisy}_loss_before`, `{clean,noisy}_increment` (loss after minus before the round's update) and `{clean,noisy}_grad_norm_start` (norm of the mean shadow-loss gradient at the round's start model); manifest gains `"diagnostics": true` | `experiments/cia/cia.py` (`shadow_loss_and_gradient_norm`, `eval_model_with_gradients`); `results/cia_frontier/eurosat_frontier/runner.py` (`round_rows`) | `experiments/cia/tests/test_round_scalars.py` (softmax-regression hand computation, batching invariance), `results/cia_frontier/eurosat_frontier/tests/test_frontier.py` (increments and start-model gradient norms from synthetic losses) |
| 4. Defaults unchanged, suite green | `checkpoint-rounds` now also accepts 0 (the initial model) | `experiments/reproduce/runner.py` | 572 passed |

Integration check (owner PC, scratch output, not committed): a 1-round `--smoke` metric-privacy run, 3 Alzheimer
clients, equal weights, `--log-loo-residuals --checkpoint-rounds 0 1`, finished; the result JSON carried
`loo-residual-norms` next to the existing diagnostics and both checkpoints were written. Scoring the two checkpoints
exposed the MPS bug in ESCALATION.md; after the fix in `experiments/cia/cia.py` the MPS round-0 shadow loss and
gradient norm equal the CPU values (1.440792, 0.967849).

## Gate G0 (copied from agent_roadmap.md Section 3.2)

| Gate | Pass | On pass | On fail |
|---|---|---|---|
| G0 | `uv run pytest` green; synthetic tests show the fixed-noise option gives identical noise IN/OUT and the logged scalars match hand computations | Push; set `STATE.md` to `ready` | Fix; do not start WP1 |

## Notes for other agents

- This file lives at `results/cia_frontier/cia_infra/STATE.md` on `feature/cia-infra` (the roadmap names no WP0 directory).
  Read it with `git fetch origin && git show origin/feature/cia-infra:results/cia_frontier/cia_infra/STATE.md`.
- WP0 made small additive, default-off edits to two files no agent owns: `experiments/reproduce/runner.py`
  (`--log-loo-residuals`, checkpoint round 0) and `experiments/reproduce/server.py` (applies the wrapper). Agent B will
  plumb its own flags through the same files; expect trivial merge conflicts there.
- Agent B: if you run anything on an Apple MPS device, read ESCALATION.md first.
