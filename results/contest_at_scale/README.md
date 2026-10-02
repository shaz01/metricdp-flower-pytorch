# Contest at scale (Aug 4–21, 2026)

**Goal.** Test the paper's metric-privacy claims beyond its own setting (see `PLAN.md`, kept
unchanged for history). The owner and teammate shared this goal. Steps:

1. Reproduce the paper on Alzheimer with 4 clients.
2. Run the paper's 3-client IN/OUT-remove client inference attack (CIA).
3. Transfer the attack to new datasets.
4. Scale to more clients.

Noise is set as a *noise ratio* (noise multiplier / number of clients). Every run here used the
Aug 4 determinism fixes (sorted replies, per-round seeds, seeded init). Runs from before those
fixes are frozen; see `/FREEZE.md`.

**Who ran what.** The owner ran `planned_runs`, `cia`, `client_scaling` and `dirichlet`. The
teammate ran `cifar100_scaling`, `cia_cifar100_scaling`, `eurosat_scaling`,
`cia_eurosat_scaling` and `auc_target_sweep`. The `dirichlet` runs are dated Aug 20 but were
merged in PR #2 on Oct 2.

**Code layout.** The code that produced these results stays in its original packages:
`experiments/cia`, `experiments/client_scaling/scripts`, `experiments/cifar100_scaling`,
`experiments/eurosat_scaling` and `experiments/reproduce`. Only results and reports live here.
The report builders in `reports/` resolve `results/` relative to this folder.

| `results/…` | Producing code | Report (`reports/…`) | Verdict |
|---|---|---|---|
| `planned_runs/reproduction` | `experiments/cia/scripts/planned_runs.py` | `results/planned_runs/reproduction/reproduction_table.tex` | Alzheimer, 4 clients, FedAvg, 3 seeds: accuracy 0.951 / 0.942 / 0.945 (vanilla / GDP / MP), 4–6 pt above the paper's Table 6 |
| `planned_runs/{alzheimer,fashion,cifar}` | `planned_runs.py`, `cifar_chunks.py`; builders in `experiments/cia/reports/` | PDFs/TeX inside each folder | 3-client multi-round CIA on Alzheimer, with transfer to Fashion-MNIST and CIFAR; CIFAR also at 3/8/16 clients, plus a noise sweep and validation at 48 clients |
| `cia/cifar10_remove`, `cia/cifar10_remove_ratio_sweep` | `experiments/cia/scripts/cifar10_remove.py`, `experiments/client_scaling/scripts/cifar10_ratio_sweep.py` | `cifar10_scaling_presentation.html`, `cifar10_ratio_sweep_presentation.html`, `cia_takeaways.html`, `accuracy_vs_roc_auc.html` | CIFAR-10, 8→100 clients: round-matched AUC falls 95%→55% overall but not monotonically (back to 100% at 48). DP modes follow the same shape as vanilla |
| `cia/check_determinism` | `experiments/cia/scripts/check_determinism.py` | — | Colab determinism check across saved artifacts |
| `client_scaling` | `experiments/client_scaling/scripts/{cifar10_homogeneous,noise_scaling_diagnostics}.py` | — | Accuracy-only homogeneous CIFAR-10 at 4/8 clients, plus the noise actually injected per configuration |
| `dirichlet` | `experiments/cia/scripts/cifar10_dirichlet.py` | `cifar10_dirichlet_presentation.html` | CIFAR-10, 8 clients, α ∈ {0.1, 1.5, 3}, seed 42. MP accuracy stays near vanilla as the ratio grows; GDP drops (α=1.5, ratio 0.00625: 0.690 vs 0.487) |
| `cifar100_scaling`, `cia_cifar100_scaling` | `experiments/cifar100_scaling/`, `experiments/cia/scripts/cifar100_scaling*.py` | `cifar-100_and_eurosat_results.tex` | 100 clients: GDP ≈ MP, both 7–8 pt below vanilla. CIA uses seed 42 only; every CI includes 0.5 |
| `eurosat_scaling`, `cia_eurosat_scaling` | `experiments/eurosat_scaling/`, `experiments/cia/scripts/eurosat_scaling*.py` | `eurosat_accuracy_sweep.md`, `eurosat_cia.md` | 48 clients: DP costs 0.4–2.4 pt. Homogeneous/vanilla leaks most (0.727); DP brings it to 0.606. CIs overlap 0.5 |
| `auc_target_sweep` | `experiments/cia/scripts/{auc_target_search,score_stage}.py` | `auc_targeted_noise_sweep.md`, `auc_frontier.html`, `eurosat_auc_frontier.html` | Noise search toward AUC ≈ 0.5 on 16 curves: 10 landed, 4 collapsed, 2 found no anchor. No blanket "MP is cheaper" result; 9/10 landings still score above 0.55 |

`cia_threat_model.tex` describes the attack's threat model.

**Known flaw.** Every attack score in this group is computed for a single target client
(partition 0). `auc_target_sweep` additionally folds the score with max(AUC, 1−AUC), which
pushes the null above 0.5. Per-client rescoring without folding is done later in
`results/cia_frontier`.

`research/project_evidence_audit.md` has a fuller audit of `auc_target_sweep`.
