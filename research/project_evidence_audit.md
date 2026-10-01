# Evidence baseline for client-specific noise research

Date: 2026-10-01. Repository inspected at `92845be8ea7aef9e62c249e45db08e4ecab7bd6f`.

This is a project catch-up and evidence audit, not a new experiment report or a completed literature review. The owner has directed the next research effort toward client-side noise, with a distribution constructed separately for each client, to improve protection against client inference attacks (CIA). Distribution construction is an open research question. The owner authorized retiring the old local `docs/RESEARCH_ROADMAP.md`; it was removed because its server-side scaling-repair objective no longer describes the intended direction.

## 1. Current project state

- The latest local commit merges the AUC-targeted noise sweep into `master`. `STATUS.md` records that sweep as complete; this audit does not make new experiment-completion decisions.
- Existing work spans paper reproduction, client-count/noise sweeps, CUDA constant-compute controls, first- and multi-round CIA, and transfer to Fashion-MNIST, CIFAR-10, CIFAR-100, and EuroSAT.
- The CUDA constant-compute report records 40 configurations, with the FedAvg accuracy advantage approaching parity at 48 clients (differences from global-DP range from -2.50 to +0.62 percentage points across the two designs/partitions). These are single-seed controls; they do not establish matched privacy. FedYogi at 48 clients remains an old uncompleted item, not an automatic priority for the new direction.
- The 96-configuration noise-by-client sweep records a client-count-dependent noise ceiling and a sharper metric-calibration failure near collapse. It does not establish the causal explanation. Earlier MPS results have reproducibility limitations; the archived large reversal must not be promoted to a confirmed scaling effect.
- CIFAR-100 CIA retains seed 42 only after unsuccessful multi-seed retries. EuroSAT's earlier CIA has three seeds. Neither should be silently pooled with the newer frontier, which changes the reporting convention and noise search.
- `STATUS.md` records no running experiments. Remote machines were not contacted in this audit, so that is recorded status, not a live remote-process check.

Sources: [status](../STATUS.md), [constant-compute report](../reports/constant_compute_scaling.md), [noise sweep](../reports/noise_by_clients.md), [scaling report](../reports/client_count_scaling.md), [EuroSAT CIA](../reports/eurosat_cia.md), [CIFAR-100/EuroSAT writeup](../reports/cifar-100_and_eurosat_results.tex), and the corresponding `results/` directories.

### Documentation that needs historical context

- `PLAN.md` records earlier removal/transfer experiments and resolved blockers; it is not the new research agenda.
- `reports/first_round_cia.md` incorrectly says no CIA result data exists and points at removed entry points. Current data and runners exist; use `experiments/cia/README.md` with the actual scripts/results for discovery.
- `reports/constant_compute_scaling.md` still says the branch is unmerged; current history and `STATUS.md` supersede that statement.
- `reports/port_equivalence.md` actually contains a five-seed numerical comparison table, with maximum output discrepancy 5.6e-17, despite no dedicated results directory. Absence of JSON artifacts is not absence of reported evidence. This audit did not rerun the isolated legacy environment.
- Some older CIA reports put confidence intervals for pooled ROC-AUC beside round-matched point estimates. The interval and point estimate must be kept attached to their respective estimands.
- The original roadmap incorrectly described itself as committed; `docs/` is ignored. Deleting its local copy will not remove copies on other machines.

## 2. Original paper: mechanism and scope

Sáinz-Pardo Díaz, J., Athanasiou, A., Jung, K., Palamidessi, C., & López García, Á. (2026). *Metric-privacy-inspired noise calibration in federated learning: Improving convergence and preventing client inference attacks*. Knowledge-Based Systems, 343, 115993. DOI: [10.1016/j.knosys.2026.115993](https://doi.org/10.1016/j.knosys.2026.115993). Local source: [Initial-Paper.pdf](../papers/Initial-Paper.pdf).

The paper assumes a trusted server and semi-honest participating clients. The attacker receives the global model and has a shadow dataset representing the target client. CIA asks whether the target client participated. This differs from asking whether a particular record was used in training. The principal utility experiments use four clients; the CIA experiment uses three with a distinctive target distribution. The strongest shadow-data setting overlaps 10% of the target training data.

Section 6 and Algorithm 1 define the maximum pairwise mean-layer model distance, then inject Gaussian noise after server-side clipping and aggregation:

\[
d_t=\max_{i\ne j}\frac{1}{L}\sum_{\ell=1}^L\|w_{i,t}^{(\ell)}-w_{j,t}^{(\ell)}\|_F,
\qquad \sigma_t=\frac{zC}{n_t d_t}.
\]

Here sigma is the standard deviation, not the variance. The repository implementation follows this inverse-distance formula; see `pairwise_model_distances` and `_add_noise_to_aggregated_arrays` in [metricdp_strategy.py](../metricdp_pytorch/metricdp_strategy.py).

**Verified inconsistency in the paper:** page 6 says that distance greater than one adds more noise, and distance below one adds less noise. Its displayed equation and Algorithm 1 imply the opposite at fixed z, C, and n. Visual inspection of the PDF page confirmed this is not a text-extraction error. Small distance increases the implemented noise scale. Any explanation of the mechanism must follow the equation and code, while explicitly acknowledging the contradictory prose.

The paper explicitly calls its method metric-privacy-inspired calibration, not a formal metric-privacy mechanism. Section 6.1 discusses accounting but supplies no numerical privacy certificate for the experiments. A future guarantee for a data-dependent noise distribution must establish the distribution's behavior across neighboring inputs; observing a realized noise scale is not itself such an analysis. This is a proof obligation for future work, not a proof supplied by this audit.

The original Table 13 attack computes pooled ROC-AUC over IN and OUT round scores. Its wide intervals overlapping 0.5 do not establish equivalence to chance. The repository frontier uses a different paired score, described next. The paper itself lists larger federations, adaptive/colluding attackers, alternative metrics, and a broader utility/privacy frontier as future work. These are motivation, not proof of novelty for any particular extension.

## 3. What auc_frontier.html actually contains

Sources: [frontier](../reports/auc_frontier.html), [generator](../reports/build_auc_frontier.py), [sweep report](../reports/auc_targeted_noise_sweep.md), [raw states](../results/auc_target_sweep/), [search controller](../experiments/cia/scripts/auc_target_search.py), [scoring implementation](../experiments/cia/scripts/score_stage.py).

The HTML is a static, self-contained SVG scatter presentation with navigation and a point table. It shows eight panels: four datasets by two partitions, each containing global-DP, metric-calibration, and a vanilla reference. Lower horizontal position and higher vertical position are preferable. The title uses "frontier", but the generator plots search/confirmation points; it does not calculate a statistically validated Pareto envelope.

- EuroSAT, Alzheimer, and four-class Fashion-MNIST: 48 canonical clients, 100 rounds, 11 attack checkpoints (1, 10, ..., 100).
- Ten-class CIFAR-10: 100 canonical clients, 20 rounds, all 20 checkpoints.
- Target partition ID is 0. OUT removes that client without repartitioning the others. Utility is the mean final-round server-test accuracy of the IN and OUT trajectories, not IN-only deployment accuracy.
- The sweep sets the base multiplier separately as noise ratio times active client count, preserving the global-DP per-coordinate standard deviation across IN/OUT removal. Metric-calibration retains its additional distance dependence.
- Scalable "non-iid" partitions use `quantity_skewed_partitions`: random assignment with unequal quantities, rather than a controlled Dirichlet label-skew or domain-shift protocol. Random finite-sample label differences can still occur.
- Vanilla references average seeds 42, 43, 44. Search uses seed 42. A landing adds seeds 43 and 44; non-landed curves do not receive this confirmation.

### Exact horizontal-axis statistic

For shadow-loss score s = -loss, the implementation computes

\[
q=\frac{1}{K}\sum_{t=1}^K\left[\mathbf{1}(s_t^{IN}>s_t^{OUT})+\tfrac12\mathbf{1}(s_t^{IN}=s_t^{OUT})\right],
\qquad q^*=\max(q,1-q).
\]

This is folded paired concordance across corresponding checkpoints. It is called round-matched AUC in the repository, but it is not the ordinary all-pairs empirical ROC-AUC. A paired design can be informative; the two estimands must not be interchanged. Averages across seeds are averages of per-seed folded scores, not a pooled ROC-AUC.

The fold chooses direction using the same observations being scored. It prevents treating a reversed attack as safe, but also raises the finite-sample null expectation above 0.5 whenever q has nonzero variability. The next evaluation needs independent direction calibration or an explicitly calibrated folded-statistic null. Checkpoints within a trajectory are dependent and are not independent client-participation trials.

### Landing values recomputed from raw state files

These means include the seed used to select the noise level. "Fresh-seed score" uses only 43 and 44 and is still a two-seed exploratory estimate. Accuracy is in percent; attack columns use the repository's folded paired statistic.

| Dataset | Partition | Mechanism | Mean accuracy | Mean score, 42-44 | Fresh-seed score, 43-44 |
|---|---|---|---:|---:|---:|
| Alzheimer | homogeneous | global-DP | 86.90 | 0.758 | 0.864 |
| CIFAR-10 | homogeneous | global-DP | 17.20 | 0.583 | 0.625 |
| CIFAR-10 | homogeneous | metric-calibration | 45.96 | 0.717 | 0.800 |
| CIFAR-10 | non-iid | global-DP | 56.61 | 0.700 | 0.775 |
| CIFAR-10 | non-iid | metric-calibration | 27.61 | 0.550 | 0.550 |
| EuroSAT | homogeneous | global-DP | 89.56 | 0.697 | 0.773 |
| EuroSAT | homogeneous | metric-calibration | 84.51 | 0.576 | 0.591 |
| EuroSAT | non-iid | global-DP | 89.41 | 0.636 | 0.682 |
| EuroSAT | non-iid | metric-calibration | 86.63 | 0.576 | 0.591 |
| Fashion-MNIST | non-iid | global-DP | 94.73 | 0.606 | 0.636 |

The 16 terminal states are ten landed, four collapsed-before-target, and two anchor-not-found. Nine of ten landing means exceed 0.55. Only CIFAR-10/non-iid/metric-calibration remains at 0.55 for all three seeds, with mean accuracy 27.61% against vanilla 56.54%. Even this is not a statistical certificate of near-chance attack performance.

### Findings to carry forward

1. **Strong motivation:** the observed utility/leakage trade-off varies substantially with dataset and partition; no universal advantage is established for metric-calibration.
2. **Unequal leakage must remain visible:** CIFAR-10/homogeneous metric-calibration has much higher accuracy than global-DP at the selected points, but also a higher confirmed attack score (0.717 versus 0.583). These are not matched-protection endpoints. Conversely, CIFAR-10/non-iid/global-DP preserves utility but has score 0.700, versus 0.550 for metric-calibration. Neither comparison alone establishes dominance at equal leakage.
3. **The strongest headline needs qualification:** statements in the existing narrative that EuroSAT's attack "goes away" or that every landed curve neutralizes CIA overstate the multi-seed evidence. EuroSAT/homogeneous/global-DP even has a larger mean folded score than its vanilla reference (0.697 versus 0.606); with this small sample this is not proof that noise increases leakage.
4. **Search failure is bounded evidence:** Fashion-MNIST/homogeneous failed a low-noise anchor search. The controller then stopped; this is not evidence that no higher-noise solution exists. The four collapse outcomes show failure along the explored geometric search path, not impossibility for all intermediate noise levels or all mechanisms.
5. **Collapse detection is coarse:** the guard uses 1/number-of-classes plus 0.02. On imbalanced Alzheimer data, this does not catch every majority-class or otherwise uninformative model. Future utility checks should include balanced accuracy, macro-F1, and class-wise behavior.
6. **Estimation limits:** the Alzheimer shadow sets contain only 8 or 11 records in inspected frontier stages. There are only three seeds at landed points, one target ID per seed, and no confidence region in the HTML. These limits are material to the new roadmap.

## 4. Evidence verification performed

- Rebuilt the HTML to a temporary path using the existing generator and current JSON files. Output was byte-identical to `reports/auc_frontier.html`; the existing report was not overwritten.
- Recomputed all 110 state entries, including repeated anchor/search entries, from their raw `runs/cia.json` losses and per-trajectory final-round accuracy JSONs. All matched the recorded state values within 1e-12.
- The non-vanilla raw directories used for this check contain 95 distinct stage runs: 31 Alzheimer, 21 CIFAR-10, 21 EuroSAT, and 22 Fashion-MNIST. IN and OUT checkpoint indices matched for every inspected stage.
- Inspected the search state machine, scoring code, removal-adjacency runners, partition functions, mechanism code, original-paper methods/threat-model/attack sections, and rendered PDF page 6 to verify its equation/prose discrepancy.
- This verifies arithmetic, provenance, and the inspected protocol. It does not rerun training, establish causal mechanism explanations, or validate a new privacy guarantee.

## 5. Questions for the forthcoming literature review

At the time of this audit, the review form was awaiting the owner's reply. The owner subsequently requested both systematic and integrative review and confirmed the protocol; version 1 is now available at [the literature-review handoff](literature_review/README.md). No new mechanism has been selected and no novelty claim has been established.

The owner's research direction suggests these extraction questions for each relevant source:

- Does it protect whole-client participation, individual records, client attributes, or reconstruction? Who is the attacker, and what exactly do they observe?
- Is noise sampled at the client, and is protection claimed for the individual upload or only its aggregate? What trust, secure-aggregation, and collusion assumptions are required?
- What does "client-specific distribution" mean: a scalar variance, diagonal/full covariance, low-rank geometry, a mixture, or a learned sampler? Which statistics construct it, and how much client data/computation does estimation require?
- Does the construction depend on private data or prior updates? How does the work account for that dependence, observable metadata, and composition over training rounds?
- Are its attacks retrained or adapted to the defense? Are test clients/trajectories and parameter-search trials separated? Does evaluation compare accuracy at matched attack success and vice versa?
- What would distinguish a new proposal from existing adaptive noise calibration, local/distributed privacy, and data-dependent perturbation work? This remains a question to investigate, not an asserted gap.

The subsequent roadmap should take the frontier as a reproducible exploratory baseline and define stronger measurement before making claims of superior CIA protection. Moving noise to clients and improving its distribution are separate design choices that should be distinguishable in future comparisons.

## Subsequent threat/evaluation audit

The [threat specification review](threat_specification_review.md) records additional verified findings: sample-count weighted FedAvg versus client-count noise scaling, application-visible world-dependent runtime configuration, and active-ID remapping of surviving clients' training seeds. These refine sensitivity and paired-run interpretations without changing the verified frontier arithmetic. The [construction obligations](noise_construction_proof_obligations.md) derive candidate controls; no new mechanism has been implemented.
