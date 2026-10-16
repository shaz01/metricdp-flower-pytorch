# Consensus-KL follow-up (D1, after WP2)

Branch `feature/consensus-kl`. Owner decision 2026-10-16, after WP2 stopped D1 (G2 inconclusive): look deeper at the
one clear lead of the WP2 screen, consensus KL with beta 0.3 (local loss `CE + beta KL(p_global || p_local)` on the
client's own batches). Three steps in the owner's order: **1** more seeds, **3** under DP, **2** smaller beta. This file
fixes step 1 now, before its runs; steps 3 and 2 get their own dated sections, committed before their runs. Nothing
here changes WP2's gate or verdict; this is a follow-up the owner asked for.

## Step 1: seeds 43-45 for consensus KL 0.3 (fixed 2026-10-16, before the runs)

**Question.** Does consensus KL 0.3 keep its seed-42 leakage reduction over four seeds, and does it still beat weight
decay and one local epoch at matched accuracy?

**Runs.** `consensus-kl-0.3` with WP2's runner and settings unchanged (EuroSAT, alpha 0.3, 48 clients, 50 rounds, no DP,
equal weights, targets 23, 7, 1, 0, 3), seeds 43, 44, 45 with the seed-42 layout (`partition_seed = 42`, the WP2
extension convention): 3 x 6 = 18 trajectories, about 5 t-h. Output `results/seeds43-45/`. Reused without rerunning
(identical code path and configuration, committed on master): WP2's `vanilla`, `weight-decay-5e-4`, `local-epochs-1`
and `fixed-steps` at seeds 42-45, and `consensus-kl-0.3` at seed 42.

**Statistics.** Exactly WP2's (`results/cia_frontier/signature_suppression/PROTOCOL.md` Sections 4-5, `analyze.py`):
clean-view score and midpoint AUC pooled over rounds and targets, 2,000-resample cluster bootstrap over the 20
target-seed pairs (seed 1729), paired across configurations; accuracy = mean round-50 accuracy over a configuration's
trajectories; the G2 rule applied to `consensus-kl-0.3` (reduction of at least 0.05 vs vanilla at no more than 1 pp
accuracy loss, and beating both controls at matched accuracy). Reported as **the G2 rule recomputed for this one
candidate over four seeds**, not as a new gate. Secondary (exploratory, as in the 2026-10-16 look at seed 42):
leakage at matched accuracy along training (rounds binned by IN accuracy at that round), per-target gaps, per-client
accuracy (lowest, 10th percentile) of the IN trajectories. Code: `analyze.py` and `explore.py` in this directory.

**No extension.** Whatever the four-seed result, step 1 ends here; the owner decides on steps 3 and 2 as planned.

## Step 3 (under DP) and step 2 (smaller beta)

To be fixed in dated sections before their runs.

## Amendments

None.
