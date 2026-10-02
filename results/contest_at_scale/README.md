# Contest at scale (Aug 4–21, 2026, Olcay and Ata)

Do the paper's metric-privacy claims hold with more clients and on new datasets? The plan is in
[`plan_suite/PLAN.md`](plan_suite/PLAN.md). Every run here came after the Aug 4 determinism fixes;
earlier runs are frozen (see [`FREEZE.md`](../FREEZE.md)).

| Experiment | Who | What it found |
|---|---|---|
| [`plan_suite/`](plan_suite/) | Olcay | The paper reproduces, but its defense numbers don't; the attack transfers to new datasets |
| [`cifar10/`](cifar10/) | Olcay | A fixed multiplier made noise do nothing; with a fixed ratio, metric-privacy keeps more accuracy, but neither defense stops the attack |
| [`cifar100/`](cifar100/) | Ata | Both defenses cost 7–8 points; attack results too noisy to compare |
| [`eurosat/`](eurosat/) | Ata | Defenses almost free, but the attack is weak even without them |
| [`auc_frontier/`](auc_frontier/) | Ata | The "landings" at AUC ≈ 0.5 didn't hold across seeds |
| [`dirichlet_comparison/`](dirichlet_comparison/) | Olcay | The old "non-IID" split wasn't label-skewed; mild real skew gives the same results, strong skew makes the defenses useless |

## Caveats

- **Every attack score here is for one target client.** One client's score swings a lot, so
  differences between modes are mostly not established. [`cia_frontier`](../cia_frontier/)
  rescores across ten clients.
- **"Non-IID" means size skew only** everywhere except `plan_suite`'s 3-client split and
  `dirichlet_comparison`.
- `auc_frontier` and the cross-experiment reports in [`reports/`](reports/) fold the score
  with max(AUC, 1−AUC), which pushes it up.
