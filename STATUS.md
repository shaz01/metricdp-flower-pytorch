# Project Status

**Branch:** `feature/eurosat-scaling`
**Last updated:** 2026-08-12, CUDA workstation (EuroSAT accuracy sweep finished 6/6; EuroSAT CIA
attack code built and tested, both trajectory groups launched on GPU 0) — see `git log` for
anything more recent

This file is a short, git-tracked pickup point for any Claude Code session — this machine or
another — starting work on this repo. It reflects the branch it's committed on; check out the
branch you're working on before trusting it. Treat it as a pointer, not the source of truth: for
full narrative detail see `reports/`, for raw run data see `results/`, for chronological detail
use `git log`. Update it whenever a branch merges into `master` or `master`-level state otherwise
changes — keep it short, don't turn it into a changelog. Exception: the Active work
section (including the Currently running table) updates more often, at "worth a commit"
granularity — see `AGENTS.md`'s "Working across machines" section.

## Active work

`feature/eurosat-scaling` (branched from `master`) adds an accuracy sweep on EuroSAT (10-class
satellite land-use imagery), a comparison point for `feature/cifar100-scaling`'s own sweep on a
genuinely simpler, different-domain dataset, plus a CIA (Client Inference Attack) experiment
against that sweep's trained models. The accuracy sweep has finished (6/6, 0 failed); the CIA
attack code is built and tested, and both trajectory groups are now running (see below).

- Data plugin (`experiments/reproduce/dataset/eurosat.py`) and model
  (`experiments/reproduce/eurosat_cnn.py`, 289,194 params, no batchnorm/buffers) are done and
  tested (9 + 4 tests).
- Sweep script (`experiments/eurosat_scaling/sweep_eurosat_scaling.py`, 7 tests) targets
  `n=48` clients only (not CIFAR-100's `n=100` — EuroSAT's 21,600 train images would starve
  per-client data further at 100 clients), `homogeneous`/`non-iid` partitions x
  `vanilla`/`global-dp`/`metric-privacy` x `fedavg`, 100 rounds, `noise_multiplier =
  0.03710712210729851`. That value was calibrated empirically from a measured post-clip/
  post-aggregation signal-update norm of `2.0786466966688715` (round 3, homogeneous/global-dp/
  n=48), following the same target-noise-to-signal-ratio-of-1 formula as CIFAR-100's sweep.
- A real 20-round verification run (homogeneous/global-dp/n=48) confirmed healthy convergence with
  the calibrated value — loss 2.275→0.703, accuracy 19.5%→74.4%, no freeze — which is why the
  100-round budget above was adopted (see the sweep script's own docstring/comments for the full
  derivation and the noted noise-to-signal drift over rounds, an expected consequence of
  single-early-round calibration, not a bug).
- Before the real sweep launched, an earlier 3-round smoke combo
  (`eurosatscale__homogeneous__global-dp__fedavg__n48__r3.{json,evaluation.json}`) was run purely
  as a pipeline sanity check (confirms data plugin + model + sweep script + `runner.py` +
  `detailed_evaluation.py` all work end-to-end for EuroSAT) and to produce a real
  `.evaluation.json` to check its size for the gitignore decision below. It's kept in
  `results/eurosat_scaling/` for provenance, alongside `sweep_progress.log`, but was never part of
  the real 100-round sweep and isn't a finished/representative result.
- `.evaluation.json` gitignore check (Task 4): the smoke combo's file is 18,245,132 bytes
  (~17.4 MiB), well under the 90MB threshold, so **no `.gitignore` rule was added** — EuroSAT's
  10-class evaluation JSONs stay far under CIFAR-100's 100-class blowup risk, as expected.
- The real 6-combination accuracy sweep has since finished: 6/6 attempted, 0 failed (see
  `results/eurosat_scaling/sweep_progress.log`). All 6 `r100` result sets
  (`eurosatscale__{homogeneous,non-iid}__{vanilla,global-dp,metric-privacy}__fedavg__n48__r100.{json,evaluation.json,predictions.npz}`)
  are present in `results/eurosat_scaling/`, alongside the earlier `r3` smoke combo (kept for
  provenance, not part of the real sweep).

`experiments/cia/scripts/eurosat_scaling.py` (built + tested this session, 9 tests in
`experiments/cia/tests/test_eurosat_scaling.py`, all passing) runs a multi-round CIA (IN vs OUT)
attack against that finished sweep: 6 combos x 3 seeds (42/43/44) = 18 trajectories per group,
checkpointed at round 1 and every 10th round through 100, reusing
`experiments.cia.attack_runner.run_attack` unmodified. Launch command (one process per group,
`--group in` and `--group out` are safe to run concurrently since each writes its own report
file, `cia_in.json`/`cia_out.json`):

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.cia.scripts.eurosat_scaling --group out --max-parallel-clients 6 > results/cia_eurosat_scaling/stdout_out.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.cia.scripts.eurosat_scaling --group in --max-parallel-clients 6 > results/cia_eurosat_scaling/stdout_in.log 2>&1 &
```

Target GPU: GPU 0 (confirmed free before launch — see Currently running table below). Both groups
were launched this session; see the Currently running table below for live status.

### Currently running

| What | Where | Status |
| --- | --- | --- |
| EuroSAT CIA, `--group out` (`experiments/cia/scripts/eurosat_scaling.py`) | CUDA workstation, GPU 0 | running |
| EuroSAT CIA, `--group in` (`experiments/cia/scripts/eurosat_scaling.py`) | CUDA workstation, GPU 0 | running |

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
