# Escalation: G-P3(b) fails (IN-count interference check), 2026-11-07

ROADMAP v3, G-P3(b): "|AUC(8 IN) - AUC(24 IN)| at most 0.03, by the pair jackknife at 97.5%. If not, escalate before
P2b's analysis: measurement standard v3 then needs an IN-count correction." P2b's analysis has already run (an order
deviation, `results/cia_v3/cifar10/STATE.md`, 2026-11-06 09:30), so G-P2b is provisional until the owner decides.

## Result

Global DP at `r = 0.003092` on EuroSAT (M2's regime), `R = 32`, `design_seed = 2`, standard attacker, all candidates.
Interval: leave-one-block-out jackknife (8 blocks of 4 replicates, t with 7 df, 97.5%; `PROTOCOL.md` Section 2). The
analysis first reproduced C1's -0.095 [-0.137, -0.053].

| Design | Active clients | AUC [97.5%] | Top decile | Mean accuracy | Lowest client |
|---|---|---|---|---|---|
| 8 IN | 24 | 0.694 [0.647, 0.742] | 0.807 | 0.731 | 0.473 |
| 24 IN | 40 | 0.654 [0.624, 0.684] | 0.802 | 0.749 | 0.527 |
| *16 IN (M2, for reference)* | *32* | *0.702 [0.643, 0.761]* | *0.746* | *0.740* | *0.512* |

**AUC(8 IN) - AUC(24 IN) = +0.040 [0.009, 0.072]:** the interval lies outside +-0.03 on one side and excludes 0. G-P3(b)
fails.

## What it can and cannot tell

- The two designs differ in two things at once: how many candidates are IN (8 or 24) and how many clients are active
  (24 or 40), because 16 slots always take part. A target's share of the average is 1/24 in one and 1/40 in the other,
  so part or all of the gap may be dilution, not interference between candidates. This design cannot separate them.
- Plain dilution would put the 16-IN design between the two. M2's 0.702 is not between them (slightly above 8 IN),
  but it was scored with twice the references per world (balanced references: about 15 against 7-8), so it is not a
  like-for-like point, and unpaired differences are detectable only at about 0.09.
- The top decile does not move (0.807 against 0.802).
- Every v2 and v3 matched-accuracy comparison uses the same 16-of-32 design for both mechanisms. The effect shifts
  absolute AUC levels; whether it changes differences between mechanisms is not measured. One reason it might: the
  composite trains about 8 sampled clients per round, against 32 for full participation.

## Options for the owner

1. **Treat it as design dependence, not a correction.** State every AUC as conditional on the 16-of-32 design, keep
   matched-accuracy differences (within one design) as they are, lift G-P2b's provisional status, and list this in the
   limits. No runs.
2. **Separate interference from dilution first.** Hold the active count at 32: 8 IN with 24 always-in slots against
   24 IN with 8 always-in, global DP 2x, `R = 32` each (64 runs, about 25 run-hours at the current shared-GPU pace).
   Needs a runner option for the always-in count (Agent P owns the runner; default-off, as P0's options). If the gap
   persists at a fixed active count, it is interference and option 3 follows; if it vanishes, option 1 holds.
3. **Define an IN-count correction now** and redo P2b's analysis with it. There is no candidate correction yet that
   does not presuppose option 2's answer.

**Recommendation:** option 2 if the paper reports absolute AUC levels or compares across designs; otherwise option 1.
G-P3(a) is unaffected and finishes today.
