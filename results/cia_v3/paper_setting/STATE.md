Package: P1, the paper's 3-client setting under measurement standard v3 (CIA roadmap v3, Agent P; direction F3)
status: main runs complete (120/120); extension (z 0.024, 80 runs) running on GPU 0 since 2026-11-01 15:59
gate: G-P1 on the main runs: "the published protection claim fails in its own setting" (extension pending)
protocol: results/cia_v3/paper_setting/PROTOCOL.md (fixed before any run)
machine: compute server
cost: 0 GPU-h; cap 40 GPU-h (escalate at 50); projected about 9 GPU-h with the declared extension
next: extension runs; then analysis.json with the extension, FINDINGS (after the owner's go-ahead)
needs from others: none

Roadmap: `research/cia_roadmap_v3/ROADMAP.md` Section 3 (P1), Sections 2 and 4. Branch `feature/cia-v3-paper-setting`
from `feature/cia-v3-infra` a8e52039 (P0's runner; G-P0 passed 2026-10-31). Sparse worktree
`../metricdp-pytorch-feature-cia-v3-paper-setting`.

## Hypothesis

The paper (Sáinz-Pardo Díaz et al. 2026, Table 13) reads AUCs of 0.397 (global DP) and 0.493 (metric privacy), with
intervals overlapping 0.5, as protection. They come from one IN and one OUT trajectory, pooling 20 dependent checkpoint
scores. In the same setting, with independent private randomness between worlds and 20 pairs:
1. the paper's per-pair statistic varies widely across pairs, so one draw can land below 0.5 by chance;
2. a calibrated attacker over trajectories separates IN from OUT under both DP mechanisms at the paper's noise
   multiplier `z = 0.01` (lower 95% bound above 0.5), because at that noise neither mechanism changes accuracy much
   (P0 cost pair: 0.95 client-test accuracy for global DP, IN world).

## Design

- **Setting (the paper's, Section 7.4.2):** Alzheimer MRI, 3 clients by Table 9 (target = client 3, partition 2;
  attacker client 1), FedAvg weighted by examples, 20 rounds, 5 local epochs, batch 32, lr 0.001, clipping 5. Noise std
  `z C / n_t` on the aggregate (`n_t` = 3 IN, 2 OUT), divided by the round's maximum client distance for metric
  privacy. Shadow sample: 10% of the target's training records with Gaussian pixel noise of 20% of the maximum.
- **Main:** vanilla, global DP `z = 0.01`, metric privacy `z = 0.01`; 20 pairs each, `design_seed = 1`: 120 runs.
- **Declared extension (the one allowed):** the paper's reported accuracy is not reproduced at `z = 0.01` (paper:
  global DP 0.637 IN; ours 0.95). Global DP and metric privacy are also run at `z_acc`, the multiplier at which our
  global DP's IN-world accuracy matches 0.637, from an accuracy pilot fixed in PROTOCOL.md: 20 pairs each, 80 runs.
- **Worlds and replicates:** one pair = an IN world (3 clients) and an OUT world (target removed, 2 clients). **No
  private randomness is shared between worlds:** each world draws its own `client_seed` and `noise_seed` from
  `SeedSequence([design_seed, pair, world])`, so the 40 worlds of a mechanism use 40 client seeds and 40 noise seeds,
  all distinct. Mechanisms share the seeds of the same (pair, world), which is allowed across configurations. Public
  randomness (data split, initial model, shadow samples) is shared by every run.

## Attacker and metrics

1. **The paper's statistic, per pair:** pooled ROC-AUC of `s_r = -loss(W_r, noisy shadow)` over rounds 1-20, 20 IN
   scores against 20 OUT scores. Reported: its distribution over the 20 pairs, the median and the 5th-95th percentile
   range.
2. **Calibrated attacker (primary):** v2's Gaussian likelihood-ratio attacker (`experiments.cia.audit_attacks.
   calibrated_scores`) on each trajectory's noisy-shadow losses at rounds 1-20, with one candidate (the target) and the
   40 worlds as replicates. Its balanced references leave out the scored world and its pair partner, i.e.
   leave-one-pair-out. AUC over the 40 trajectories with a pair-jackknife 95% interval (20 pairs, t with 19 df).
3. Reported alongside: the same attacker on the clean shadow sample; the uncalibrated final-round loss threshold; TPR
   at 5% FPR (one OUT trial at 20 OUT trials, so descriptive only); Table 13's accuracy (aggregated client-test
   accuracy, mean of the last 5 rounds, IN and OUT) and the lowest client accuracy.
4. **Reproduction first (standard v3 item 7):** the analysis code reproduces M2's published vanilla all-candidate AUC
   (0.723) to 3 decimals from the committed runs before it reports anything new.

## Gate

**G-P1 (report; no kill)** (ROADMAP v3 Section 3, verbatim):
- If the calibrated attacker's lower bound is above 0.5 for both DP mechanisms: "the published protection claim fails
  in its own setting".
- If the paper's statistic spans 0.5 across pairs: "Table 13 is a single draw from a wide distribution".
- Otherwise: "consistent with the paper", and the audit rests on A-3 alone.

Operational definitions are fixed in PROTOCOL.md: "lower bound" = the pair-jackknife 95% interval's lower end;
"spans 0.5" = the 5th percentile of the per-pair statistic below 0.5 and the 95th above it, for at least one DP
mechanism. Applied to the main runs (`z = 0.01`); the extension is reported under the same rule, separately.
**Untestable targets:** P1 has no matched-accuracy targets, so standard v3 item 5 does not apply. If a mechanism
has fewer than 20 complete pairs after three attempts, its statistics use the complete pairs and say so. If fewer
than 15 pairs complete, its clause of G-P1 is "not testable".

## Files

| File | Change |
|---|---|
| `results/cia_v3/paper_setting/` | STATE, PROTOCOL, `analyze.py` with `tests/`, pilot and run outputs (`pilot/`, `results/`), `analysis.json`, FINDINGS, figure |
| `STATUS.md` | Agent P rows only |

The runner is P0's `results/cia_v3/infra/paper_runner.py`, unchanged.

## Log

| When | What |
|---|---|
| 2026-10-31 12:41 | Pilot (12 runs) and main runs (120) started on GPU 0, shared with another user's job (5.9 GB free; ROADMAP decision 4), one worker, 3 parallel clients (`~/cia-server/p1-run.sh`, tmux `p1`); peak VRAM about 3.6 GB. |
| 2026-10-31 12:44 | `analyze.py` and tests committed (92a7fe8b): the reproduction check gives M2's vanilla all-candidate AUC 0.723 from the committed runs with P1's attacker code. |
| 2026-10-31 13:15 | **Stopped by host RAM pressure, not by the GPU.** Other users' jobs filled the machine's RAM to 96.5% (243 of 251 GB; 13 GB available), and Ray kills workers above 95% node memory, so every pass failed within about 26 s (`low on memory` in the logs). Pilot runs complete before the stop: z 0.03 4/4, z 0.06 3/4. No partial run was kept. Ray's limit is not raised, because the kernel would then kill processes, possibly other users'. Restarted 13:23 under `~/cia-server/v3-queue.sh`, which now also waits for at least 25 GB of available RAM, retries up to 20 times, and skips completed runs. |
| 2026-10-31 13:49 | **Accuracy pilot (12 runs) complete; 0.637 not bracketed.** Mean IN-world client-test accuracy over rounds 16-20 (two IN runs each): z 0.03 0.501 (0.500, 0.501), z 0.06 0.447 (0.446, 0.447), z 0.12 0.501 (0.501, 0.501). All three are below 0.637; P0's cost pair gave 0.95 at z 0.01. Accuracy falls off a cliff between 0.01 and 0.03; about 0.50 looks like a collapsed model predicting the majority class. By PROTOCOL.md Section 2 the grid is extended by halving below 0.03, at most two points: z 0.015 and, only if 0.637 is still not bracketed, z 0.0075. Both are queued together to save a round trip; 0.0075 enters the interpolation only if 0.015 does not bracket. They run after the main runs (`~/cia-server/p1-pilot2.sh`). |
| 2026-10-31 16:31 | Extension pilot (z 0.015, 0.0075) started on GPU 1 with 10.9 GB free, and failed within 3 minutes: another user's 8 GB job started on GPU 1 at the same moment (CUDA out of memory). No run was kept; the queue retries. Main runs: vanilla 40/40 complete at 16:14. |
| 2026-10-31 19:22 | **Extension pilot complete; `z_acc = 0.024`** (recorded before any extension run). Mean IN-world client-test accuracy over rounds 16-20 (two IN runs each): z 0.0075 0.961 (0.963, 0.960), z 0.015 0.893 (0.880, 0.906), z 0.03 0.501. 0.637 is bracketed by 0.015 and 0.03; linear interpolation in log z gives 0.02358, rounded to two significant digits 0.024 (PROTOCOL.md Section 2). z 0.0075 did not enter the interpolation. Caveat for FINDINGS: accuracy falls from 0.89 to 0.50 within one doubling of z, so the interpolated z may not give 0.637; the extension reports its measured accuracy. Global DP z 0.01 main runs 40/40 complete at 18:59. |
| 2026-10-31 21:50 | Main runs complete, 120/120. |
| 2026-11-01 15:59 | **Extension started 18 h late.** It was set to wait for the main queue (tmux `p1`) to end, but `tmux has-session -t p1` matches by prefix and also matched its own session `p1-ext`, so it never started. Restarted directly on GPU 0 (12.5 GB free, shared). |
| 2026-11-01 16:01 | **G-P1 on the main runs** (`analyze.py --main vanilla gdp-z0.01 mp-z0.01`; reproduction check M2 vanilla 0.7235 = 0.723): calibrated attacker, noisy shadow, global DP 1.000 [1.000, 1.000], metric privacy 1.000 [1.000, 1.000]; both lower bounds above 0.5, so "the published protection claim fails in its own setting". The paper's per-pair statistic: global DP median 0.853 (p05 0.732, p95 0.988), metric privacy 0.887 (0.822, 0.955), no pair below 0.5, so the "single draw" clause does not hold. Vanilla: calibrated 0.638 [0.474, 0.801] on the noisy shadow, 1.000 on the clean shadow. Accuracy (IN, rounds 16-20): vanilla 0.970, global DP 0.947, metric privacy 0.957. Diagnostic (not gated): on the clean shadow the IN model's final loss is 0.008-0.043 against 1.08-1.65 OUT under all three mechanisms, i.e. the target's records are memorized and z 0.01 does not prevent it; vanilla scores lower on the noisy shadow only because its noisy-shadow losses vary widely (final IN 9.4 +/- 15.5). Output kept outside the repo until the extension is in. |
