Package: P3, every-round attacker and interference check (CIA roadmap v3, Agent P; direction F4)
status: all runs complete (192 plus 32 follow-up); G-P3(a) FAILS (the composite leaks more than metric privacy under the every-round attacker); G-P3(b) follow-up inconclusive
gate: G-P3(a) fail at 0.740 (+0.097 [0.042, 0.151]), 0.660 not testable; G-P3(b) fail (+0.040 [0.009, 0.072]), follow-up D_int +0.012 [-0.025, 0.050] inconclusive
protocol: results/cia_v3/robustness/PROTOCOL.md (fixed before any run)
machine: compute server
cost: 0 GPU-h; cap 35 GPU-h (ROADMAP decision 5), escalate at 43.75
next: R = 64 extension of the 24-active follow-up (64 runs, one worker per GPU; PROTOCOL.md Section 6); then the owner decides on an IN-count correction
needs from others: none

Roadmap: `research/cia_roadmap_v3/ROADMAP.md` Section 3 (P3), Sections 1 and 2; decision 5 in Section 10. Branch
`feature/cia-v3-robustness` from `feature/cia-v3-infra` 52311328. Sparse worktree
`../metricdp-pytorch-feature-cia-v3-robustness`.

## Hypothesis

1. A sampling-aware attacker that sees every round, not 5-round windows, recovers more of the composite's leakage. If
   it does, D-4 on EuroSAT weakens below G-C1's rule at 0.740. (P2's sampling-aware attacker already cut the composite's
   gain on CIFAR-10.)
2. The AUC does not depend on how many candidates are IN, so the 16-of-32 design of every v2 and v3 result does not bias
   it. Known risk: 8 IN has 24 active clients and 24 IN has 40, so a target's share of the average differs (1/24 against
   1/40); the gate does not separate that from interference.

## Design

Six configurations, `R = 32`, `design_seed = 2`, EuroSAT in M2's regime, 192 runs (`PROTOCOL.md` Section 1):
- composite (`q = 0.25`, `tau = 1`) at `r` = 0.000773 and 0.003092, every-round scoring;
- metric privacy at `r` = 0.001546 and 0.003092, every-round scoring;
- global DP at `r` = 0.003092 with 8 and with 24 candidates IN.
**No private randomness is shared between worlds:** each replicate draws its own seeds from `(design_seed, replicate)`,
as in M2. The same replicate of two configurations shares them, which is allowed across configurations.

## Attacker and metrics

- **G-P3(a):** calibrated attacker on the 11 standard checkpoints plus the every-round schedule statistic `z`, for the
  composite; the standard attacker for metric privacy; A1's matched-accuracy difference.
- **G-P3(b):** the standard attacker; `AUC(8 IN) - AUC(24 IN)`; block jackknife at 97.5%.
- Reported: `PROTOCOL.md` Section 2, item 6. Power: paired differences detectable at about 0.03-0.04 (all candidates).
- **Reproduction first:** C1's -0.095 [-0.137, -0.053] at 0.740, to 3 decimals, before any new number.

## Gate

**G-P3** (ROADMAP v3, verbatim):
- "(a) The every-round sampling-aware attacker: G-C1's rule for the composite against metric privacy at 0.740 and
  0.660, where bracketed."
- "(b) |AUC(8 IN) - AUC(24 IN)| at most 0.03, by the pair jackknife at 97.5%. If not, escalate before P2b's analysis:
  measurement standard v3 then needs an IN-count correction."

**Untestable targets** (standard v3 item 5): a target outside either curve is "not testable" and does not count as a
pass. With these levels 0.660 is expected to be not testable, so (a) rests on 0.740. If 0.740 is not testable too, (a)
is "not testable". (b) has no target. Operational reading of (b): the whole interval inside +-0.03.

## Files

| File | Change |
|---|---|
| `results/cia_v3/robustness/` | STATE, PROTOCOL, `analyze.py` with `tests/`, `results/`, `analysis.json`, FINDINGS, figures |
| `STATUS.md` | Agent P rows only |

The runner is P0's, unchanged.

## Log

| When | What |
|---|---|
| 2026-11-06 | Protocol fixed before any run. Order note: P2b's analysis ran before P3 (`results/cia_v3/cifar10/STATE.md`, 2026-11-06 09:30), so G-P2b is provisional until G-P3(b). This protocol's gate is the ROADMAP's, fixed before those numbers existed; its operational readings (block jackknife, whole interval inside +-0.03) were written after them, but they depend on no P2 number. |
| 2026-11-06 21:40 | `analyze.py` and 7 tests committed (6f7bda98) before any run. Reproduction check: C1's composite minus metric privacy at 0.740, -0.095 [-0.137, -0.053], equal to the published value to 3 decimals. |
| 2026-11-06 21:43 | Runs started: `~/cia-server/p3-worker.sh` under `v3-direct.sh`, three workers with 4 parallel clients (`p3-w1`, `p3-w3` on GPU 0, `p3-w2` on GPU 1), per-run locks, RAM and GPU checks before each run (22, 28 and 34 GB of RAM; 4 GB of GPU memory). GPUs are shared with the owner's F0.7 shard workers and another user's job (decision 4). |
| 2026-11-07 14:25 | All 64 IN-count runs complete (no failures). **G-P3(b) fails:** standard attacker, all candidates, AUC 8 IN 0.694 [0.647, 0.742] (24 active, accuracy 0.731), 24 IN 0.654 [0.624, 0.684] (40 active, accuracy 0.749); difference +0.040 [0.009, 0.072], block jackknife at 97.5%. Reproduction check first (C1 -0.095 [-0.137, -0.053]). Escalated in `ESCALATION.md`; the design cannot separate interference from the active count. 173/192 runs complete; the other runs continue (they serve G-P3(a), which the escalation does not affect). |
| 2026-11-07 14:45 | **Owner chose ESCALATION.md option 2** (ROADMAP decision 7). As proposed (8 IN with 24 always-in) it does not fit EuroSAT's 16 non-candidate slots, so the active count is held at 24 instead (PROTOCOL.md Section 5, fixed before the runs): the existing 8-IN runs against 32 new runs with 24 IN and no always-in client. Agent P told the owner of the change. |
| 2026-11-07 14:37 | Runner option `--always-in-count` and the follow-up analysis committed (117e88f2; 55 tests pass; 48 committed P3 and M2 manifests rebuild unchanged). Follow-up runs started: `~/cia-server/p3b-worker.sh`, two workers on GPU 1 (`p3b-w1`, `p3b-w2`, 4 parallel clients; 40 and 46 GB of RAM required before each run). The P3 workers continue with the last composite runs (176/192). |
| 2026-11-07 18:08 | All runs complete: P3 192 (16:11) and follow-up 32 (18:08). 51 tries failed with CUDA out of memory between 14:00 and 16:00, when Agent P had five workers on the two shared GPUs; all were retried and completed. |
| 2026-11-08 13:30 | **G-P3(a) fails** (`analysis.json`; reproduction check C1 -0.095 [-0.137, -0.053]). Composite minus metric privacy at 0.740, every-round sampling-aware attacker: **+0.097 [0.042, 0.151]** (composite 0.798 against metric privacy 0.702; two-way [0.035, 0.161]); 0.660 not testable (the levels stop at 0.724 and 0.678). Per level, the every-round attacker scores the composite 0.820 (r 0.000773) and 0.788 (r 0.003092), against 0.582 and 0.578 for the standard attacker; the schedule statistic alone gives 0.807 and 0.779. Reported, same runs at 0.740: standard attacker -0.122 [-0.158, -0.086]; 11-checkpoint sampling-aware -0.065 [-0.120, -0.009]; 51 checkpoints -0.132 [-0.186, -0.079]; 51 checkpoints plus z +0.082 [0.024, 0.140]. Lowest-client accuracy at 0.740: composite 0.452, metric privacy 0.497. **Caveat for the reading:** the attacker scores IN trials with the target's real sampling schedule and OUT trials with a simulated one (`rz.simulated_schedules`, standard v3 item 3), i.e. it assumes the attacker knows when the target would be sampled; the statistic alone carries most of the signal. |
| 2026-11-08 13:30 | **Follow-up (PROTOCOL.md Section 5; `analysis_follow_up.json`):** AUC 8 IN / 24 active 0.694 [0.647, 0.742]; 24 IN / 24 active 0.682 [0.637, 0.727] (accuracy 0.761); 24 IN / 40 active 0.654 [0.624, 0.684]. `D_int` = +0.012 [-0.025, 0.050]: **inconclusive at this power** (the interval crosses +0.03). `D_act` = +0.028 [-0.008, 0.064]. By point estimates, about 0.028 of G-P3(b)'s 0.040 gap is the active count and 0.012 the IN count, but neither interval excludes 0. |
| 2026-11-08 13:30 | **Owner decisions (ROADMAP decision 8):** the every-round attacker is in scope, so D-4 does not hold on EuroSAT for it; the IN-count follow-up is extended to `R = 64` once, then the owner decides. PROTOCOL.md Section 6 fixed before the extension's runs. |
