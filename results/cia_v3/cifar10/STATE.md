Package: P2, CIFAR-10 replication of A-3 and D-4 (CIA roadmap v3, Agent P; direction F2)
status: stage b complete (312 runs); G-P2b computed, provisional until G-P3(b); owner reconsiders decision 1
gate: G-P2a passed; G-P2b: A-3 replicates (equivalent at a2, a4 not testable); D-4 does not replicate under either attacker
protocol: results/cia_v3/cifar10/PROTOCOL.md (fixed before any run)
machine: compute server
cost: about 246 GPU-h by summed run time (stage a and pilot about 28, first half 99, second half 119; failed tries not counted); cap about 290 GPU-h
next: owner's decision on the ROADMAP Section 4 row (D-4 does not replicate); P3, whose interference check G-P3(b) the P2b numbers depend on; FINDINGS on the owner's go-ahead
needs from others: none

Roadmap: `research/cia_roadmap_v3/ROADMAP.md` Section 3 (P2), Sections 1 (decisions 2, 8), 2 and 4; decision 5 in
Section 10. Branch `feature/cia-v3-cifar10` from `feature/cia-v3-infra` a8e52039 (P0's CIFAR-10 profile; G-P0 passed
2026-10-31). Sparse worktree `../metricdp-pytorch-feature-cia-v3-cifar10`.

## Hypothesis

The two central EuroSAT results replicate on a second label-skewed dataset with real leakage:
- **A-3:** at matched accuracy, metric privacy leaks like global DP (G-A1: equivalent or no detectable difference);
- **D-4:** the composite (client subsampling `q = 0.25` with logit adjustment `tau = 1`) leaks at least 0.05 AUC less
  than metric privacy at matched accuracy, under the standard and the sampling-aware attackers.
First CIFAR-10 must leak at all under the measurement standard (G-P2a), which Alzheimer did not (A-6).

## Design

- **Regime: M2's on CIFAR-10** (P0's `cifar10` profile): ten-class CIFAR-10, label-Dirichlet alpha 0.3, 48 slots, layout
  seed 42, 32 candidates (top decile 26, 27, 36, 12, 45 plus 27 by size quantile), 16 IN per replicate, 16 always in,
  32 active; 50 rounds, 5 local epochs, clipping 5, equal weights; `R = 32`, `design_seed = 2`; 1,024 trials per
  configuration.
- **Stage a:** vanilla, replicates 0-7 of the `R = 32` design (four complementary pairs, each candidate IN in 4).
  These runs are reused as vanilla's first 8 replicates in stage b.
- **Accuracy pilot** (only after G-P2a passes; `design_seed = 3`, replicates 0-1, accuracy only): global DP and
  metric privacy at 1x, 2x, 4x, 8x of `r = 0.001546`; the composite at `r` = 0.0001, 0.000773, 0.003092, 0.006184,
  0.012368 (B1's and C1's levels). 26 runs.
- **Stage b:** vanilla (all 32 replicates), global DP at 1x, 2x and 4x, metric privacy at 3 levels and the composite at
  3 levels chosen from the pilot by the rule in PROTOCOL.md, `R = 32` each, in two halves with A1's replacement rule
  between them. The composite is scored at every round (standard v3 item 4).
- **Targets:** global DP's stage-b mean final accuracy at its 2x and 4x levels (as on EuroSAT, where they were 0.740
  and 0.660).
- **No private randomness is shared between worlds:** every replicate draws its own `client_seed` and `noise_seed`
  from `(design_seed, replicate)`, the inclusion of the 32 candidates is randomized by the balanced design, and the
  composite's client sampling uses the run's own seeds. The same replicate of two configurations shares its seeds and
  inclusion, which is allowed across configurations and makes the matched-accuracy differences paired. Public
  randomness (layout, initial model, shadow samples, test split) is shared.

## Attacker and metrics

- **Gated attacker:** v2's calibrated likelihood-ratio attacker on the 11-checkpoint shadow-loss trajectory (rounds 0,
  5, ..., 50), all candidates; for the composite also the **sampling-aware attacker** of
  `research/cia_roadmap_v3/reanalysis/reanalyze.py` (`sampling_aware`, centred window-drop statistic).
- **Matched-accuracy statistic:** A1's (curves linear in mean final accuracy; pair jackknife at 97.5%, the same pairs
  in every configuration; two-way bootstrap alongside).
- Reported, not gated: top decile; TPR at 5% and 1% FPR; the composite's 51-checkpoint attacker; accuracy and the
  lowest-client accuracy interpolated at the targets (standard v3 item 6).
- **Power** (standard v3 item 2): the gates are paired matched-accuracy differences, detectable at about 0.03-0.04
  (all candidates) and 0.09-0.10 (top decile) at `R = 32`.
- **Reproduction first:** the P2 analysis reproduces A1's published metric privacy against global DP at 0.740,
  -0.000 [-0.019, 0.018], to 3 decimals before reporting any CIFAR-10 number.

## Gate

**G-P2a** (ROADMAP v3, verbatim): "Vanilla all-candidate AUC at least 0.65 at `R = 8`. Otherwise stop, write
`ESCALATION.md` and propose FEMNIST (decision 2)." Applied to the calibrated attacker's point estimate on stage a's
8 replicates; the pair-jackknife interval (4 pairs) is reported.

**G-P2b** (ROADMAP v3, verbatim):
- "G-A1 for metric privacy against global DP, reported."
- "G-C1 for the composite against metric privacy, under the standard and the sampling-aware attacker."

G-A1: better (interval excludes 0, negative), equivalent (interval within ±0.03), or worse or inconclusive; at each
target. G-C1 (decision 9): the composite's all-candidate AUC at least 0.05 below metric privacy's, with the 97.5%
pair-jackknife interval of the difference below 0, at each target; stated per attacker.

**Untestable targets** (standard v3 item 5): a target that a mechanism's curve does not reach after the replacement
rule is "not testable" for every comparison involving that mechanism and does not count as a pass. Outcomes:
- D-4 **replicates**: G-C1 passes at both targets (stated per attacker);
- **partial**: passes at one target, the other not testable;
- **does not replicate**: fails at a testable target, or no target is testable ("not testable").
A-3 replicates if G-A1 is "equivalent" or "no detectable difference" at every testable target.

## Cost

P0 cost runs (one worker; GPU shared with another user's job): CIFAR-10 vanilla 0.542 run-hours, composite 0.360 at
11 checkpoints. With every-round scoring the composite's evaluation grows from 248 s to about 1,150 s (EuroSAT ratio),
about 0.61 run-hours. Estimate: stage a 4.3, pilot 12.4, stage b vanilla 13.0, global DP 53.8, metric privacy 53.8,
composite 58.6; **about 196 run-hours**.

Note: ESCALATION.md (P0) converted run-hours to GPU-h with C1's two-worker ratio (0.68). Under ROADMAP decision 4 only
one of our workers fits on a shared GPU (peak 3.6 GB), so occupancy equals run-hours. At that rate P2 projects above
its 150 GPU-h escalation point; this is raised with the owner before stage b (stage a and the pilot cost about 17).

## Files

| File | Change |
|---|---|
| `results/cia_v3/cifar10/` | STATE, PROTOCOL, `analyze.py` with `tests/`, `pilot/`, `results/`, `analysis.json`, FINDINGS, figures, tables |
| `STATUS.md` | Agent P rows only |

The runner and data module are P0's, unchanged.

## Log

| When | What |
|---|---|
| 2026-10-31 12:43 | Stage a started on GPU 1, shared with another user's job (4.3 GB free), 4 parallel clients. |
| 2026-10-31 12:44 | **Out of memory in round 1:** the other user's job on GPU 1 grew from 6.5 to 8.1 GB. One client failed, and `--require-all-clients` stopped the run before anything was saved. Nothing partial is kept (no `complete.json`), and the replicate retrains from scratch with the same seeds. Stopped (own session only). |
| 2026-10-31 12:46 | Restarted through a retrying queue (`~/cia-server/v3-queue.sh`): 2 parallel clients (about 1.5 GB; parallelism changes scheduling, not the trajectories), and a shared GPU qualifies only with at least 3.5 GB free (our 1.5 GB plus the other job's observed 1.6 GB swing). Measured 1.4-1.5 GB. |
| 2026-10-31 13:15 | **Stopped by host RAM pressure, not by the GPU.** Other users' jobs filled the machine's RAM to 96.5% (243 of 251 GB; 13 GB available), and Ray kills workers above 95% node memory, so every pass failed within about 26 s (`low on memory` in the logs). No partial run was kept. Ray's limit is not raised, because the kernel would then kill processes, possibly other users'. Restarted 13:23 under `~/cia-server/v3-queue.sh`, which now also waits for at least 25 GB of available RAM, retries up to 20 times, and skips completed runs. |
| 2026-10-31 13:29 | Stage a restarted on GPU 1 under the queue (RAM back to 33 GB available). A first attempt at 13:28 had picked GPU 0, together with P1, because both queues polled in the same second. It was stopped within a minute, and the queue now takes a lock while choosing and claiming a GPU. |
| 2026-10-31 14:00 | One of the other user's processes on GPU 1 ended (9.4 GB free), so a second stage-a worker was added on GPU 1 (tmux `p2-w2`, 2 parallel clients; runs are locked per replicate). Our VRAM on GPU 1 is 2.9 GB for both workers. Pace about 1.1 min per round, about 58 min per run. |
| 2026-10-31 14:58 | **Out of memory on GPU 1, replicates 2 and 3 lost at round 26-27 and round 1.** Another user started a second job of about 8 GB on GPU 1, which left 2 MiB free; 6 of 32 clients failed, and `--require-all-clients` stopped both workers before anything was saved. Replicates 0 and 1 are complete. Worker 2 (`p2-w2`) is not restarted. Since 15:00 the queue has waited: GPU 0 is P1's, and GPU 1 has about 3.2 GB free, below the 3.5 GB threshold. |
| 2026-11-01 03:01 | Stage a complete, 8/8 (resumed 2026-10-31 20:55 on GPU 1 when 10.9 GB was free; one worker). |
| 2026-11-01 16:04 | `analyze.py --stage a` (71e35c74, 4 tests). **Reproduction check:** A1's metric privacy minus global DP at 0.740 from the committed runs, -0.000 [-0.019, 0.018], equal to the published value to 3 decimals. Stage a needs its own loader (`load_first_replicates`), since v2's `load_audit` requires all 32 replicates of the design; it checks inclusion against the design's first 8 columns. |
| 2026-11-01 16:05 | **G-P2a passed.** Vanilla, replicates 0-7 (256 trials), calibrated attacker on 11 checkpoints: all candidates 0.998 [0.995, 1.001] (pair jackknife, 4 pairs, 97.5%; liberal at R = 8), two-way bootstrap [0.991, 1.000]; top decile 1.000; TPR 1.000 at 5% FPR and 0.922 at 1%; final round only 0.998; uncalibrated threshold 0.927. Mean final accuracy 0.633 [0.631, 0.634] (P0 cost run 0.630), lowest client 0.457. Leakage is far above EuroSAT's (M2 vanilla 0.723), so CIFAR-10 has the headroom A-6 lacked on Alzheimer. `analysis_stage_a.json`. |
| 2026-11-01 16:06 | Accuracy pilot queued (`~/cia-server/p2-pilot.sh`, tmux `p2-pilot`, 2 parallel clients; runs taken under per-run locks so a second worker can join). Both GPUs have under 2 GB free (other users' jobs and P1's extension on GPU 0). |
| 2026-11-02 08:51 | Pilot complete, 26/26 (one worker on GPU 1; a second worker joined 03:32-08:51 under per-run locks). |
| 2026-11-02 10:40 | **Stage-b levels (PROTOCOL.md Section 3; `analyze.py --stage pilot`, `pilot_levels.json`), recorded before any stage-b run.** Pilot mean final accuracy (2 runs each): global DP 1x 0.616, 2x 0.554, 4x 0.326, 8x 0.100; metric privacy 1x 0.626, 2x 0.609, 4x 0.512, 8x 0.310; composite r 0.0001 0.614, 0.000773 0.606, 0.003092 0.537, 0.006184 0.327, 0.012368 0.100. Targets `t2` 0.5536, `t4` 0.3263; safeguard passes (`t4` >= 0.30, gap 0.227). **Metric privacy: 2x, 4x, 8x** (r 0.003092, 0.006184, 0.012368; qualifying window). **Composite: r 0.000773, 0.003092, 0.006184** (no window qualifies; smallest shortfall 0.0004, its top level 0.3267 against `t4` 0.3263; the replacement rule after the first half applies if `a4` is not bracketed). Global DP 1x, 2x, 4x as fixed. Note: CIFAR-10 accuracy collapses to chance (0.10) at 8x global DP and r 0.012368, so the 4x target sits on a steep part of every curve. |
| 2026-11-02 10:37 | **Owner approved stage b in full** (ROADMAP decision 6, on `feature/cia-v3-infra` 52311328): projected about 254 run-hours for stage b at the measured shared-GPU pace (about 55 min per run, about 35 min per composite run with every-round scoring), about 280 GPU-h for P2; cap about 290. Stage b first half queued (`~/cia-server/p2-stage-b.sh`, two workers under `v3-direct.sh`, tmux `p2-b1` with 4 parallel clients and `p2-b2` with 2, both on GPU 1, per-run locks). Both wait for 25 GB of available host RAM (18 GB at launch, other users' jobs). GPU 0 is not used: another user's job there has run P1 out of memory three times. |
| 2026-11-02 21:30 | **Stage b, first 9.5 hours mostly lost to host RAM.** Both workers started 11:57-11:59 when 25 GB was available, but the launcher checked RAM only at start. Other users' jobs then filled the node, Ray's memory monitor (95% of 251 GB) killed runs within about 30 s, and each worker moved straight to the next run; the runner exits 0 even when a run fails, so the failures did not stop the passes. 9 runs completed (comp-r0.006184 reps 10-14, gdp-r4x rep 13, vanilla reps 8-10); no partial run is used (the runner retrains a run from scratch). Stopped at 21:28 (own sessions, no leftover processes), claims removed. `p2-stage-b.sh` rewritten: before every run it waits for 22 GB of available RAM (worker 2: 28, so both cannot start at the edge) and 3 GB free on the GPU; success is read from `complete.json`; at most 6 tries per run. Restarted 21:30 with two workers on GPU 1, 2 parallel clients each. |
| 2026-11-03 00:47 | 13/152 of the first half complete. Measured since the restart: about 52 min per run with 2 parallel clients (global DP 1x: 51-52 min); two runs were killed by host-memory pressure at 22:12 and retried. Both GPUs had cleared (other users' jobs ended; GPU 0 idle, 58 GB RAM available), so a third worker was added on GPU 0 (tmux `p2-b3`, 4 parallel clients, own claim line; it requires 34 GB of available RAM before each run, so it backs off first). |
| 2026-11-03 10:56 | 58/152 of the first half complete; 45 runs since 00:47 with no failure and no wait (about 4.4 runs per hour). Both GPUs nearly idle apart from our workers (28 GB free each), 53 GB RAM available, load 21 on 48 cores: a fourth worker added on GPU 1 (tmux `p2-b4`, 4 parallel clients; requires 40 GB of available RAM before each run, so it pauses first). |
| 2026-11-04 01:56 | Stage b first half complete, 152/152 (one run failed once and was retried). All four workers exited cleanly and removed their claims. |
| 2026-11-04 19:30 | **Replacement check (PROTOCOL.md Section 4; accuracy only, no leakage statistic computed):** first-half mean final accuracy over replicates 0-15: vanilla 0.631, global DP 1x 0.609, 2x 0.546, 4x 0.319; metric privacy 2x 0.598, 4x 0.514, 8x 0.298; composite r 0.000773 0.607, 0.003092 0.534, 0.006184 0.317. Half-stage targets `a2` 0.5457, `a4` 0.3192. Metric privacy brackets both (0.598 >= a2, 0.298 <= a4); the composite brackets both (0.607 >= a2, 0.317 <= a4, a margin of 0.0025 at the low end). **No replacement.** Second half not started: the owner's adaptive-matching F0.7 chain (owner's order: F0.7 first) is about to start a shard worker on each GPU; the second half waits for the owner's call. |
| 2026-11-04 19:31 | **Second half started** at the owner's word (160 runs, replicates 16-31): four workers, `p2-b1`/`p2-b2` on GPU 1 (2 parallel clients; 22 and 28 GB RAM required before each run) and `p2-b3`/`p2-b4` on GPU 0 (4 parallel clients; 34 and 40 GB). Both GPUs idle at start; the owner's F0.7 chain will add a shard worker on each GPU when its data copy finishes. |
| 2026-11-05 11:51 | Second half 55/160. Throughput: 16 runs 19:31-23:00, 5 runs 23:00-05:00, 34 runs 05:00-11:51. 26 failed tries, all 23:00-05:00, almost all Ray host-memory kills mid-run (node at 95%; a few CUDA out-of-memory on GPU 1, where another user's TrackFormer job returned with 15 GB at about 22:48; the owner's F0.7 shard workers hold about 3 GB on each GPU). No partial run is used. Retry counters of runs at 3-4 tries were reset to 0 (the 6-try cap guards against deterministic failures; these were memory kills), so a bad night cannot exhaust them. |
| 2026-11-06 06:28 | Stage b second half complete, 160/160 (failed tries 2026-11-04 23:00 to 2026-11-05 05:00, none after). No replacement was needed. |
| 2026-11-06 09:20 | **G-P2b** (`analyze.py --stage b`, `analysis.json`; code committed first, b8de666d; reproduction check A1 -0.000 [-0.019, 0.018]). Targets `a2` 0.547, `a4` 0.319. **G-A1, metric privacy minus global DP:** a2 -0.002 [-0.008, 0.005], equivalent; a4 -0.004, **not testable** (global DP's 4x level is the curve's lowest point and defines `a4`, so leave-one-pair-out curves do not reach it; two-way [-0.041, 0.026], 984 of 2,000 resamples bracketed). **A-3 replicates** by the rule (every testable target). **G-C1, composite minus metric privacy, standard attacker:** a2 -0.017 [-0.027, -0.007], fail (below 0, but less than 0.05); a4 -0.055 [-0.086, -0.025], pass. **Sampling-aware attacker:** a2 -0.007 [-0.013, -0.001], fail; a4 -0.040 [-0.076, -0.003], fail. **D-4 does not replicate under either attacker.** Reported: 51-checkpoint attacker a2 -0.011 [-0.019, -0.004], a4 -0.051 [-0.075, -0.026]. Lowest-client accuracy at a2 / a4: global DP 0.355 / 0.160, metric privacy 0.344 / 0.155, composite 0.353 / 0.163. AUCs at a2 are 0.967-0.985, near the ceiling, so a fixed 0.05 reduction is demanding there; the rule was fixed before any run and is applied as written. |
| 2026-11-06 09:30 | **Order deviation, Agent P's:** ROADMAP's G-P3(b) says to escalate an IN-count interference problem "before P2b's analysis", and the calendar has P3 alongside stage b. P2b's analysis was run before P3 started. So G-P2b is provisional: if G-P3(b) fails, measurement standard v3 needs an IN-count correction and this analysis is redone. P3's design and gate were fixed in the ROADMAP before these numbers existed, so they cannot have been tuned to them. |
