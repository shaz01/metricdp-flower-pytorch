Package: P3, every-round attacker and interference check (CIA roadmap v3, Agent P; direction F4)
status: 192 runs running from 2026-11-06 21:43; IN-count runs first
gate: G-P3, not yet evaluated
protocol: results/cia_v3/robustness/PROTOCOL.md (fixed before any run)
machine: compute server
cost: 0 GPU-h; cap 35 GPU-h (ROADMAP decision 5), escalate at 43.75
next: runs; G-P3(b) as soon as the 64 IN-count runs are in; then the rest and G-P3(a); G-P2b stays provisional until G-P3(b)
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
