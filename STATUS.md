# Project Status

**Branch:** `runs/new-auc-frontier-eurosat`
**Last updated:** 2026-09-24 02:15 CEST, macOS laptop (deadline-directed training; queue revised to preserve original runs)

This file is a short, git-tracked pickup point for any Claude Code session — this machine or
another — starting work on this repo. It reflects the branch it's committed on; check out the
branch you're working on before trusting it. Treat it as a pointer, not the source of truth: for
full narrative detail see `reports/`, for raw run data see `results/`, for chronological detail
use `git log`. Update it whenever a branch merges into `master` or `master`-level state otherwise
changes — keep it short, don't turn it into a changelog. Exception: the Active work
section (including the Currently running table) updates more often, at "worth a commit"
granularity — see `AGENTS.md`'s "Working across machines" section.

## Active work

### Current handoff (2026-09-24 02:15 CEST)

Owner authorizes useful training until **2026-09-24 15:00 CEST**, not a declaration
that either experiment is finished. Alpha **.3**, noise ratio **.001546**, target
panel **0–9**, rounds **1–100** recorded; final-round analysis primary.

- **Preserve original results:** `frontier/` has 54 verified trajectories: 44 form
  complete seeds 42/43 × global-dp/metric-privacy; 10 form partial seed 44 (IN +
  OUT 0–3 each). Missing exact library versions is a reproducibility limitation,
  NOT grounds for discarding these results. Old seed-42/43 paired concordance:
  global-dp .70/.50, metric-privacy .60/.30; target-stratified cross-seed AUC
  .575/.45. Two seeds only, no precision/privacy-certification claims.
- Repeated round-1 SIGFPE in Colab torch 2.11 stopped training. A pinned torch
  2.10/CUDA12.8 canary succeeded; setup now verifies dependencies and records
  runtime/pip-freeze (`af3e6e3`). This does NOT establish that successful old VMs
  changed versions during the original experiment. New results remain separately
  identified in `frontier_torch210/`; 15 verified as of the 01:55 snapshot.
- **Owner rejected blanket reruns.** Scheduler now skips already-completed old
  defense runs; already-running overlap runs finish. Priority: six missing vanilla
  seed-42 OUTs (4–9), then twelve missing seed-44 defense OUTs (4–9 both modes)
  under `frontier_torch210_s44_mixedenv/`, then vanilla seed 43 only if its whole
  panel fits. Mixed-environment seed 44 must be labeled, not silently pooled.
  New/old seed-42 overlap shows some paired signs change on small loss margins;
  it is a sensitivity diagnostic, not an environment-equivalence test.
- **Capacity:** lab2 currently has two usable A100 slots for the main queue.
  Owner says default exhausted compute units. lab3 auth repaired but A100
  allocations fail. lab4's two slots are reserved for the separate influence pilot.
  No purchases or account changes by agents. Proxy-token refreshers preserve
  >1h sessions; controller double-collect/setup hangs need monitoring.
- **Influence pilot:** separate `feature/influence-noise-pilot` worktree/branch.
  Matched-energy f=.5 IN accuracy 70.81% vs f=0 86.00%, severe class-recall losses.
  Owner authorized early stopping: no new f=.5 OUTs, already-running jobs finish.
  Parent selected f=.05 as next arm; pre-outcome utility screen gates its OUTs.
  f=0 reference continues. No alternative scientific decisions delegated to agent.
- Both queues have detached schedulers; recheck `.colab`/remote state before any
  launch to avoid duplicates. Execution agents monitor and collect, parent chooses
  scientific allocation. No narrative final report yet. Deadline handoff needs
  actual artifacts/coverage/failed attempts/runtime labels and VM shutdown audit.

### Historical snapshot (superseded by current handoff above)

**`runs/new-auc-frontier-eurosat` (owner chose alpha .3; seed-42 target panel complete).**
First wave (4 trainings, source `b815aba`): global-dp and metric-privacy at
noise ratio .001546, seed 42, one shared IN (fixed panel targets 0–9 evaluated at
every round 1–100) plus one OUT dropping target 0 each, in disjoint shards
`results/new_auc_frontier_eurosat/frontier/<mechanism>-r0.001546-seed-42-{in,out-0}/`.
The runner gained `--adjacency in|out|both` and `--out-targets` (panel/IN manifest
unchanged); analysis now finds nested shards and rejects the partial wave as
unbalanced. All four collected (commits `5cf897c`, `6f697d1`, `ab0d575`, `d158e3f`):
exit 0, completion markers, A100-SXM4-40GB, identical partitions, rounds 1–100 for
all 10 targets on each IN and target 0 on each OUT. Round-100 accuracy: global-dp
IN 88.0% / OUT-0 88.4%; metric-privacy IN 84.3% / OUT-0 85.6%. Wall time per IN:
~40 min training + ~33 min evaluation (CPU-bound, GPU idle); per OUT: ~40 + 3 min.
One IN/OUT pair per mechanism is not an attack estimate. Known controller fault:
Colab CLI runtime-proxy tokens expire after 3600 s; the CLI then gets 404, prunes the
session and kills its keep-alive while the VM keeps running. Both IN sessions (~78 min)
hit this; they were re-registered from the live assignment list and then collected.
Sessions over 60 min need a controller fix or manual re-register. Owner then authorized
only the remaining OUT targets 1–9 for both mechanisms (18 trajectories, ≤4 A100s
on default/lab2), shards `frontier/<mechanism>-r0.001546-seed-42-out-<t>/`.
All 18 are collected and pushed (targets 1–9, each mechanism; commits in branch history
through `47d62c1`); no retries/duplicates, no token expiry, no active sessions.
Verification across all 22 trajectories: A100, exit 0, markers, all round/target rows,
finite clean/noisy/aggregate losses, consistent partitions. Analyzer succeeds for seed 42.
Round-100 clean-loss IN−OUT signs: global-dp IN lower on 7/10 targets, mean difference
−0.03635; metric-privacy 6/10, mean −0.03290. Noisy-loss IN−OUT means −1.8538 and
−1.1058 respectively (IN lower on 9/10 targets in both modes). Final IN accuracy:
88.00% global-dp, 84.30% metric-privacy. Mean OUT accuracy over targets 0–9:
87.71% (range 86.74–88.44) global-dp; 85.35% (84.22–86.59) metric-privacy.
The per-target raw clean/noisy differences are in the measurement artifacts; one seed
is descriptive only: no CI, privacy claim, or independent-sample multiplication.
OUT cost averaged ~39.7–40.4 min training plus 3.3 min evaluation; targets 1–9 used
~13.35 GPU-hours total. Nothing beyond these 22 trajectories is authorized. Earlier
pilot state follows.
New isolated experiment under `experiments/auc_frontier/`; protocol is in its
`README.md`. EuroSAT, label-Dirichlet non-IID, 48 canonical clients, 100 rounds.
All 12 vanilla IN pilot trajectories (alpha=.1,.3,1,10 × seeds 42–44) were
collected from default/lab2 A100 Colab accounts, each with successful exit and
completion marker at `results/new_auc_frontier_eurosat/alpha_pilot/alpha-<value>-seed-<seed>/`.
Mean round-100 accuracy: .1=88.0%, .3=88.9%, 1=90.0%, 10=90.6%; across the
three partitions per alpha, mean dominant-class client share: .671, .459,
.298, .165 respectively. Accuracy improved 3.9–6.9pp from round 50 to 100
on completed pilot curves, so no round-50 plateau. The owner must choose
scientifically meaningful alpha and lock/revise the provisional ratio grid
before the defended discovery sweep; do not choose by privacy outcome.
An initial launch failed before training because Colab snapshots lack `.git`;
source revision is now passed via worker environment (`ed4eb9a`). A .3/43
session lost its CLI handle/VM late in training; a fresh session retrained
and collected it. Failed pre-training artifacts were removed from results.
Recovery exposed a detached-waiter recursion, fixed in `578cb0f` (50 relevant
tests passed). No active Colab sessions; the lost local state is terminal.
Per-trajectory manifests, partition histograms, provenance, local locks and atomic
outputs support independent seed/ratio/mechanism shards and resume. Partial attack
trajectories retrain; complete trajectories skip. No target-list sharding of IN.
Analysis distinguishes paired concordance from target-stratified ROC AUC and
resamples whole seed blocks (no CI below five seeds; no privacy-certification claim).
Tests: `uv run pytest experiments/auc_frontier/ experiments/cia/tests -q` — 140 passed,
synthetic/mocked tests only, no dataset download/training or real-data evaluation.
Next: obtain owner's alpha choice and lock/revise ratios; then run defended
discovery and selected confirmation seeds. Budget 768 trainings without vanilla
attack reference; 873 including it (12 pilot runs included). Pilot results exist;
no attack results or experiment-finished report yet.


**`feature/colab-multi-account` (in progress, 2026-09-22, macOS laptop).** Reworks
`scripts/colab/run_experiment.py` so several Google accounts can drive 8+ Colab GPUs at once:
per-account `HOME` isolation (`~/.colab-accounts/<name>/`, since `colab_cli` hardcodes its token
path), `--account auto` slot selection under a per-account cap, detached-by-default controllers,
a `sweep` command that probes every session and auto-collects runs whose controller died, and a
`.colab/commit.lock` around Git commits only — collection stays lock-free so a vanishing VM never
blocks a download. Pushing is no longer automatic. Covered by `tests/test_colab_controller.py`
and a new `tests/test_colab_end_to_end.py` that drives the real remote helpers against a fake
`colab` CLI. Exercised on default and lab2 during the EuroSAT alpha pilot.

Unrelated pre-existing breakage seen while verifying: the full `uv run pytest` run aborts inside
`experiments/reproduce/tests/test_paper_loss.py` (torch/MPS `Fatal Python error: Aborted`). It
reproduces on a clean `master` worktree and passes when that file runs alone, so it is not caused
by this branch.

---

**`feature/auc-targeted-noise-sweep` is complete (2026-09-01) and merged into `master`.** See
`reports/auc_targeted_noise_sweep.md` for the full writeup and "What's established" below for the
summary — not repeated here to avoid drifting out of sync.

---

Nothing else currently running. `reports/accuracy_vs_roc_auc.html` (refreshed 2026-08-15) was sent to
the project supervisor for review; his feedback asked for a step back from the numbers-heavy
format toward two plain-language, plot-supported claims: (1) more clients → lower CIA attack AUC,
especially vanilla, and (2) DP noise lowers attack AUC at a heavy accuracy cost. **New follow-up
report, 2026-08-16: `reports/cia_takeaways.html`** (generator: `reports/build_cia_takeaways.py`,
independent of `build_accuracy_vs_roc_auc.py` — recomputes everything from source rather than
reusing the old script's embedded numbers) builds one plot per claim, one plot per dataset where
data exists, and checked both claims against the actual numbers rather than assuming them true:
- **Claim 1 (client count) holds only partially, and only on CIFAR-10** — the only dataset ever
  run at more than one client count. Round-matched attack AUC does drop net (95%→55% from 8→100
  clients) but isn't monotonic (bumps back to 100% at 48), and Global-DP/Metric-privacy show the
  *same* shape as vanilla, not a weaker one — so the "especially vanilla" framing isn't supported.
  Flagged as an open question in the report, not silently smoothed over.
- **Claim 2 (noise vs. accuracy) holds directionally on every dataset checked** (CIFAR-10 full
  sweep, EuroSAT, CIFAR-100, Alzheimer, Fashion-MNIST — 6 checks, 6/6 show DP attack AUC ≤
  vanilla's) but "heavy cost" is dataset-dependent, not universal: real on CIFAR-100 (~8pp) and
  CIFAR-10 under high noise (~20pp for global-dp), negligible on EuroSAT/Fashion-MNIST (<2pp).
- Old `reports/accuracy_vs_roc_auc.html` was left as-is (not overwritten) per explicit instruction
  to generate a new file instead — both now exist; `cia_takeaways.html` is the one to send back to
  the supervisor.
- Housekeeping: 4 remote branches confirmed fully merged into `origin/master` (0 unique commits
  each) and deleted, both locally-absent and on `origin` — `feature/cia-client-scaling`,
  `feature/cifar10-scaling` (already reflected below), `improve-logging`, `new_experiments`. `git
  branch -a` now shows only `master`.

Separately, both `feature/eurosat-scaling` (the EuroSAT accuracy sweep and its CIA attack) and
`feature/cifar100-scaling` (the CIFAR-100 accuracy sweep and its CIA attack) are complete, merged
into `master`, and deleted (locally and on `origin`) — see "What's established" below and
`reports/eurosat_accuracy_sweep.md`/`reports/eurosat_cia.md`/
`reports/cifar-100_and_eurosat_results.tex` for the full writeups.

**CIFAR-100 CIA multi-seed retry was dropped.** The seed-42 single-seed CIA run (6/6 combos, 0
failures) was complete and reportable. A follow-on rerun to add seeds 43/44 — for a properly
3-seed-pooled analysis matching the Alzheimer/CIFAR-10/Fashion-MNIST/EuroSAT CIA protocols —
launched 2026-08-10 20:08 but never finished: the large 100-client/4.6M-parameter model combos hit
persistent, severe GPU VRAM contention on this shared machine. Four separate retry attempts (each
recovering 1-3 of the failing combos, with diminishing returns, despite reducing
`--max-parallel-clients` 16→6→10 and fixing a real `dp_diagnostics.py` crash-on-client-error bug
along the way) never reached a complete 3-seed run. Decision: drop seeds 43/44 entirely and keep
only the complete seed-42 data. `results/cia_cifar100_scaling/cia_in.json`/`cia_out.json`/
`cia_analysis.json` and the 12 per-trajectory result JSONs now hold seed-42-only data for all 6
combos; the 22 seed-43/44 per-trajectory JSONs were deleted.
`reports/cifar-100_and_eurosat_results.tex`'s CIFAR-100 CIA table reflects this (a single
seed-42-only table, all 6 combos scored, replacing the earlier 3-seed-pooled table that had two
"in progress" `global-dp` rows).

**Model history** (this repo went through several CIFAR-100 architectures before settling):
v1-v3 was a plain 3-block CNN (v2 briefly added a 4th conv block, reverted after its natural
per-round update magnitude exceeded `clipping_norm=5.0` and froze every clipping privacy mode).
v4 replaced it with a DenseNet+SELU architecture (553,220 params, concatenative skip connections,
GroupNorm(8), SELU with LeCun-normal init and AlphaDropout), built and verified for robustness to
that clipping-related freeze — and a second, independent freeze mode was found and fixed in that
generation too (`vanilla`, no DP at all, froze at n=128/homogeneous from many highly-correlated
client updates reinforcing rather than averaging out). **Per project-owner direction, v4 was itself
replaced** with the current model (v5): an adaptation of the project supervisor's own `CNNCIFAR100`
reference architecture (3 blocks of 2x[Conv3x3-GroupNorm-ReLU], channels 128/256/512,
global-average-pooled classifier, 4,631,268 params, 0 buffers — see
`experiments/cifar100_scaling/sweep_cifar100_scaling.py`'s docstring for the full model-history
record and `experiments/reproduce/cifar100_cnn.py`'s docstring for the adaptation details). This
consolidation also removed the separate `feature/cifar100-scaling-supervisor` branch/worktree that
had briefly held this model in isolation (merged into this branch, then deleted) and cleared every
prior CIFAR-100 result (v1-v4 sweep data, the supervisor model's own earlier narrower grid) —
none were kept, since all described a model, grid, or directory layout no longer in use.

GroupNorm has no running-stats buffers — unlike BatchNorm — so this still needs no
`metricdp_pytorch/metricdp_strategy.py` changes, same as v1's "no normalization at all" workaround.
Training supports weight decay (`WEIGHT_DECAY = 5e-4`) and an opt-in cosine LR schedule; the
schedule was tried and dropped — a decaying LR eventually drops client updates below
`clipping_norm`, so clipping stops binding late in each run and the DP noise-to-signal ratio drifts
back up against `noise_multiplier`, fighting the whole point of tuning it — so the sweep uses a
fixed LR instead. `noise_multiplier=0.0182`, calibrated specifically for this model at n=100 (the
only client count this sweep runs — see the `NOISE_MULTIPLIER` comment in
`experiments/cifar100_scaling/sweep_cifar100_scaling.py` for the full derivation), confirmed via a
verification run: noise-to-signal ratio 1.001 at n=100.

Full sweep and CIA results (tables, protocol, discussion) are in "What's established" below and
`reports/cifar-100_and_eurosat_results.tex` — not repeated here to avoid drifting out of sync with
those. Raw data: sweep JSONs at `results/cifar100_scaling/*.json`; CIA raw/analysis JSONs at
`results/cia_cifar100_scaling/`. Both experiments' `.evaluation.json`/`.predictions.npz`
artifacts stay local-only, gitignored (see `.gitignore` comment) — they blow past GitHub's 100MB
push limit.

Separately, on `master`: nothing currently running. `feature/scale-controlled-redo` (Phase 1 items
1 and 2 of `docs/RESEARCH_ROADMAP.md`) merged into `master` 2026-08-06 and was deleted — see
"What's established" below for what it left behind. The natural next steps there, not yet started:
`fedyogi` at `n=48` (the redo's matrix only covers `n=4/8` for `fedyogi`), and Phase 1's remaining
item (NaN/failure-mode logging in `runner.py`, motivated directly by the zero-norm-update crashes
found during the redo). After that, Phase 2 (mechanism redesign) is the next major phase.

### Currently running

Update this table whenever a machine picks up new work: add a row, edit the Status column
in place (e.g. `running` -> `done`), and leave a finished row for one update cycle before removing
it, so machine-to-results provenance isn't lost; see `AGENTS.md`'s "Working across machines"
section.

| Command | What | Status |
| --- | --- | --- |
| Main frontier queue | Colab A100 account-role lab2, ≤2 usable slots; vanilla seed 42 then missing seed-44 OUTs, pinned runtime. | Running; inspect live scheduler before launching |
| Influence pilot queue | Colab A100 account-role lab4, ≤2 slots; f=0 reference and screened f=.05; separate feature branch. | Running; f=.5 early-stopped, in-flight jobs finish |
| Additional capacity | default units exhausted; lab3 rejects A100 allocation despite repaired login. | Blocked, no purchase authorized |

## What's established on `master`

- **AUC-targeted noise sweep** (`reports/auc_targeted_noise_sweep.md`, `reports/auc_frontier.html`):
  4 datasets (EuroSAT n=48, Alzheimer n=48, Fashion-MNIST n=48, CIFAR-10 n=100) x 2 partition modes
  x 2 privacy modes = 16 curves, each an autonomous search for the noise multiplier that pushes CIA
  round-matched attack AUC to ~0.5 (10 landed, 4 collapsed-before-target, 2 anchor-not-found — see
  the report for the full per-curve table). Noise-to-neutralize-attack cost is dataset/partition/
  mechanism-dependent, not a single number: EuroSAT reaches the target cheaply in all 4
  combinations; CIFAR-10 reaches it in all 4 but the accuracy cost swings from negligible
  (non-iid/global-dp) to severe (homogeneous/global-dp, pushed to near-random accuracy); Alzheimer
  and Fashion-MNIST/homogeneous show the sweep's clearest negative result — several combinations
  break the model before the attack is ever actually neutralized, and Fashion-MNIST/homogeneous
  (both privacy modes) never found a usable low-noise anchor at all in the range searched.
  Metric-privacy's cost advantage over global-DP shows up clearly on CIFAR-10/homogeneous but
  reverses on CIFAR-10/non-iid — no blanket "metric-privacy is cheaper" claim survives this data.
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
- **CIFAR-100 accuracy sweep + CIA** (`reports/cifar-100_and_eurosat_results.tex`): `n=100`
  clients, 250 rounds, `fedavg`, `noise_multiplier=0.0182`. Accuracy sweep: 6/6 combos, 0 failed,
  21.4–29.1% accuracy (100-class task, ~1% random-baseline) — metric-privacy roughly ties
  global-dp (within ~0.6pp either way), both DP modes ~7–8pp below vanilla, partition mode barely
  moves the numbers. CIA: seed-42-only (a multi-seed 43/44 rerun was attempted for proper 3-seed
  pooling but dropped after repeated GPU VRAM contention on this large 4.6M-parameter model
  prevented it from ever completing — see `git log` on the deleted `feature/cifar100-scaling` for
  the retry history), 26 round-matched pairs per combo, all 6/6 combos scored. Leakage is highest
  for `homogeneous/vanilla` (0.846) and lowest for `non-iid/vanilla` (0.500, the no-leakage line);
  `global-dp` shows more leakage than `metric-privacy` in every partition/shadow combination on
  this seed. Every 95% CI includes 0.5 (single-seed, underpowered), so read this as a directional
  pattern, not a statistically confirmed ranking.
- **EuroSAT accuracy sweep + CIA** (`reports/eurosat_accuracy_sweep.md`, `reports/eurosat_cia.md`):
  a comparison point on satellite land-use imagery (10-class, genuinely different domain from
  CIFAR-10/CIFAR-100/Fashion-MNIST/Alzheimer), `n=48`. Accuracy sweep: 6/6 combos, 0 failed,
  87.5–90.7% accuracy across all combos; `non-iid` partitioning slightly *outperformed*
  `homogeneous` in every privacy mode, and DP mechanisms cost only 0.4–2.4pp versus vanilla. CIA:
  36/36 trajectories (18 IN + 18 OUT), 0 failed, 3 seeds from the start (CIFAR-100's CIA needed a
  post-hoc multi-seed redo after an underpowered single-seed pilot — this one skipped that
  mistake). Round-matched AUC shows the clearest leak at `homogeneous/vanilla` (0.727), both DP
  mechanisms suppress it there (to 0.606), and `non-iid` leaks much less across every privacy mode
  (one combo's noisy-shadow AUC drops to 0.212, below chance) — but confidence intervals are wide
  (~0.30–0.35 AUC units, all overlapping 0.5), so read this as a directional pattern, not a
  statistically confirmed ranking. Along the way: found and fixed a real bug in `server.py`
  (`_require_trained_arrays`) where a run whose every single round failed to aggregate crashed
  with a confusing "Missing key(s) in state_dict" error instead of a clear one — Flower's
  `Strategy.start()` only assigns `result.arrays` on a successful round, with no fallback to the
  initial model.

## Where to look

- `docs/RESEARCH_ROADMAP.md` — canonical multi-session research plan (gitignored — not on every
  machine by default; copy it manually if a fresh checkout is missing it).
- `reports/*.md`, `reports/*.tex` — narrative writeups; source of truth over this file for
  anything beyond a one-line summary.
- `results/<name>/` — raw run data; `results/archive/` — superseded data kept for comparison.
- `AGENTS.md` — repo conventions, including the branch-per-experiment workflow that explains why
  most in-progress work isn't here yet.
- `git branch -a` — see which experiment branches are currently active.
