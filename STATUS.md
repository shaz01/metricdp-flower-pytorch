# Project Status

**Branch:** `feature/cifar10-scaling`
**Last updated:** 2026-08-11, Mac laptop (`cifar10_homogeneous` n=4/n=8 Colab runs collected)
— see `git log` for anything more recent

This file is a short, git-tracked pickup point for any Claude Code session — this machine or
another — starting work on this repo. It reflects the branch it's committed on; check out the
branch you're working on before trusting it. Treat it as a pointer, not the source of truth: for
full narrative detail see `reports/`, for raw run data see `results/`, for chronological detail
use `git log`. Update it whenever a branch merges into `master` or `master`-level state otherwise
changes — keep it short, don't turn it into a changelog. Exception: the Active work
section (including the Currently running table) updates more often, at "worth a commit"
granularity — see `AGENTS.md`'s "Working across machines" section.

## Active work

On `feature/cifar10-scaling`: the `cifar10_homogeneous` accuracy-only sweep now has real data at
`n=4` and `n=8` (`results/client_scaling/cifar10_homogeneous/clients-{4,8}/`), 5/5 combos each, 0
failures, run on Colab A100s 2026-08-11. Final server accuracy:

| n | vanilla | global-dp nm=0.0182 | global-dp nm=0.05 | metric-privacy nm=0.0182 | metric-privacy nm=0.05 |
|---|---------|---------------------|-------------------|--------------------------|------------------------|
| 4 | 0.7090  | 0.6366              | 0.1828            | 0.7194                   | 0.6992                 |
| 8 | 0.7150  | 0.6988              | 0.4958            | 0.7106                   | 0.6968                 |

Metric-privacy beats global-DP at both client counts and both multipliers, with the gap widening
sharply at `nm=0.05` (+51.6pp at `n=4`, +20.1pp at `n=8`) — global-DP at `n=4`/`nm=0.05` collapses
to near-random (0.1828). No invalid distances or collapsed aggregations in any run. Not yet
interpreted into a report — that's the owner's call.

Operational note from this session: running two controllers in parallel raced the `colab` CLI's
own session store (`~/.config/colab-cli/sessions.json`) — when the first run finished and removed
its entry, the file was left as `{}`, orphaning the second run's still-live VM (alias unresolvable,
`status` failing with "Session not found"). The run itself was fine; only local tracking was lost.
Recovered by rebuilding the `SessionState` from the server-side assignment endpoint/token, then
`collect --session ...` as usual. Worth staggering parallel launches or checking `sessions.json`
after each completion.

`feature/scale-controlled-redo` (Phase 1 items 1 and 2 of `docs/RESEARCH_ROADMAP.md`) merged into
`master` 2026-08-06 and was deleted — see "What's established" below for what it left behind. Still
not started: `fedyogi` at `n=48`, and Phase 1's remaining item (NaN/failure-mode logging).

### Currently running

| Machine | Work | Status |
|---------|------|--------|
| Colab A100 (via Mac laptop) | `cifar10_homogeneous`, `--clients 4` | done 2026-08-11, pushed `40168d2` |
| Colab A100 (via Mac laptop) | `cifar10_homogeneous`, `--clients 8` | done 2026-08-11, pushed `5aa78ef` |

Update this table whenever a machine picks up new work: add a row, edit the Status column
in place (e.g. `running` -> `done`), and leave a finished row for one update cycle before removing
it, so machine-to-results provenance isn't lost; see `AGENTS.md`'s "Working across machines"
section.

## What's established on `master`

- The metric-privacy mechanism reproduces the source paper at 4 clients — the effect is barely
  visible at the paper's `noise_multiplier=0.01` (`reports/paper_reproduction.md`).
- A genuine, previously unpublished effect exists at 8 clients: metric-privacy beats global-DP by
  +6.9pp (homogeneous) / +12.2pp (non-IID) at `noise_multiplier=0.05`
  (`reports/client_count_scaling.md`).
- `MetricPrivacyServerSideFixedClipping.aggregate_train` (`metricdp_pytorch/metricdp_strategy.py`)
  no longer aborts a whole run on a non-finite/non-positive client-model distance or a
  `ZeroDivisionError` from Flower's own clipping code on zero-norm updates — both fall back
  gracefully (last-valid distance, or skip-and-keep-previous-round respectively) so a run's full
  round-by-round history survives instead of being discarded on one bad round. Richer per-round
  diagnostics also landed: full pairwise client-model distance distribution, per-pair client IDs,
  min/median/mean/count, not just the single max used for calibration.
  `LoggedGlobalDPServerSideFixedClipping` (`metricdp_pytorch/globaldp_strategy.py`) has the same
  zero-norm-update guard as of the scale-controlled redo — it never had one before, and 12/96 runs
  in `results/noise_by_clients/` hit exactly this crash before the fix landed.
- **A genuine client-reply-ordering non-determinism was found and fixed**: Flower's own weighted
  aggregation summed replies in network-arrival order, not deterministically — floating-point
  addition isn't associative, so this compounded into real numeric drift over rounds. Fixed via
  `DeterministicReplyOrderMixin` (`metricdp_pytorch/strategy_factory.py`, covers every aggregation
  method, not just metric-privacy) and a matching fix in `metricdp_pytorch/metrics.py`.
- **Phase 1 items 1 and 2 of `docs/RESEARCH_ROADMAP.md` are done**, redone on CUDA hardware (moved
  off this project's original Mac MPS backend, whose own non-determinism made every earlier result
  on these questions untrustworthy — see `reports/archive/constant_compute_scaling_mps_v1v2.md`).
  - **Constant-compute client-count scaling** (`reports/constant_compute_scaling.md`/`.tex`):
    `fedavg` at `n=4/8/48` + `fedyogi` at `n=4/8` — 40/40 combinations, 0 failures, 0
    invalid-distance/collapsed-aggregation rounds anywhere. The metric-privacy-vs-global-dp
    advantage shrinks from `n=4` to `n=48` but converges toward parity (`fedavg`: -2.5pp to
    +0.6pp at `n=48`), not the large reversal the MPS-era attempt found (which never survived its
    own noise-floor check). `fedyogi` at `n=48` not yet run.
  - **Noise-multiplier x client-count sweep** (`reports/noise_by_clients.md`): `n ∈ {8,16,32,48}` x
    6 noise multipliers, `fedavg` only — 96/96, 0 failures. The noise ceiling genuinely shifts up
    with client count (a fixed `noise_multiplier` is relatively less noisy at higher `n`, since
    `compute_stdv` divides by `num_sampled_clients`) — `nm=0.1` collapses training at `n=8` but
    stays healthy through `n=48`. Separately, metric-privacy's own calibrated noise goes unstable
    right at each client count's collapse boundary and underperforms global-dp there by as much as
    -18pp, worse at higher `n` — a real, more precisely localized version of the original
    `results/48client_scaling` scaling concern, and a lead for Phase 2's mechanism redesign.
- `reports/first_round_cia.md` is stale — says "no result data yet," but `results/cia_client_scaling/`
  has real trained models and partial attack scores. Needs a rewrite, not done yet. The Flower-1.32
  port-equivalence check (`reports/port_equivalence.md`) still has no committed result data.

## Where to look

- `docs/RESEARCH_ROADMAP.md` — canonical multi-session research plan (gitignored — not on every
  machine by default; copy it manually if a fresh checkout is missing it).
- `reports/*.md`, `reports/*.tex` — narrative writeups; source of truth over this file for
  anything beyond a one-line summary.
- `results/<name>/` — raw run data; `results/archive/` — superseded data kept for comparison.
- `AGENTS.md` — repo conventions, including the branch-per-experiment workflow that explains why
  most in-progress work isn't here yet.
- `git branch -a` — see which experiment branches are currently active.
