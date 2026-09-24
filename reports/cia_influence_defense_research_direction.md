# Server-side CIA defense: research direction and agent handoff

**Last updated:** 2026-09-24

**Branch:** `feature/cia-influence-defense`

**Starting commit on master:** `92845be` (`Merge branch 'feature/auc-targeted-noise-sweep'`)

**State:** scalar diagnostic complete; opt-in geometry probe implemented; first CUDA pilot complete.

**Primary empirical reference:** [auc_frontier.html](auc_frontier.html).

This is a living research and continuity report requested by the project owner. It is **not
a completed-experiment report**. It records the discussion, evidence, candidates, questions,
and next steps so work can continue in another chat or on another machine. The 2026-09-24
diagnostic in section 3.6 reanalyzes existing runs, and the first geometry pilot now has a
training result. No defense mechanism or CIA comparison result exists yet.

The owner selected **server-side changes only** and accepted **noise shaped by client influence**
as the first direction to explore. The other candidates below remain alternatives or controls.
Selecting a direction does not mean its formula, experimental budget, or success criteria have
already been settled. The owner alone decides when the experiment is finished.

## 1. Start here in a new session

1. Read [AGENTS.md](../AGENTS.md), [STATUS.md](../STATUS.md), this report, and recent git history.
2. Work on `feature/cia-influence-defense`; keep its work and eventual results on this branch
   until the owner declares the experiment finished. Do not merge early.
3. Use [auc_frontier.html](auc_frontier.html) as the common reference for the research discussion.
   Read its raw points and the qualifications in section 3, rather than equating a `landed`
   status with confirmed protection.
4. Read the section 3.6 pilot observations, then resume with the design questions in section 6.
   Specify and review the first defense prototype before a larger experiment.
5. Update this report's decision log and `STATUS.md` as meaningful work progresses. Commit and
   push at those milestones. Record machine roles, never hostnames, IP addresses, or usernames.

The local `docs/RESEARCH_ROADMAP.md` contains older candidate tracks and literature pointers,
but is gitignored and may be absent elsewhere. This committed report is sufficient to recover
the direction selected in this discussion. Older roadmap novelty statements and unverified
bibliography entries must not be treated as established facts.

## 2. Research objective and threat model

**Objective:** improve useful model accuracy at comparable client-participation inference risk,
or reduce that risk at comparable accuracy, relative to global-DP and the paper's metric-aware
calibration. Evidence should hold across independent seeds and targets, rather than only at a
selected seed's favorable search point.

The source paper is Sáinz-Pardo Díaz et al., *Metric-privacy-inspired noise calibration in
federated learning: Improving convergence and preventing client inference attacks*,
Knowledge-Based Systems 343 (2026), 115993. See the [local PDF](../papers/Initial-Paper.pdf)
and [published article](https://doi.org/10.1016/j.knosys.2026.115993).

The starting threat model follows the existing experiments:

- A trusted server receives client updates and releases aggregated models.
- An honest-but-curious participant observes released models and holds shadow data representing
  another client's distribution. The existing clean-shadow experiments use a subset of the
  target's training data, a strong knowledge assumption.
- The attacker tries to determine whether that target participated. The frontier uses paired
  IN-remove and OUT-remove trajectories, preserving the other clients' canonical partitions.
- The defense may change server-side clipping, aggregation, and noise. Changes to local training
  objectives or client update construction are outside the selected initial scope.
- Start with FedAvg, matching the frontier. Extensions to other aggregators, active malicious
  clients, collusion, or secure aggregation have not been selected.
- The defense cannot assume access to the attacker's shadow dataset or knowledge of which client
  will be targeted. Using target shadow loss belongs to evaluation, not deployed calibration.

The paper's calibration is empirical and does not establish a formal metric-DP guarantee.
This research likewise has no established formal guarantee. A claim about resistance to a tested
CIA must remain separate from a claim about differential privacy.

## 3. Existing evidence and qualifications

### 3.1 How the current mechanism works

For each round, the implementation calculates the maximum over client pairs of the mean
layer-wise Euclidean/Frobenius model distance, `d_t`. It measures this **before clipping**, clips
client updates, aggregates, and adds isotropic Gaussian noise with per-coordinate standard
deviation

\[
\sigma_t = \frac{m C}{n d_t}.
\]

Here `m` is the nominal noise multiplier, `C` is the clipping norm, and `n` is the configured
sampled-client count. Global-DP uses `m C / n` without the distance divisor.
See [metricdp_strategy.py](../metricdp_pytorch/metricdp_strategy.py) and
[globaldp_strategy.py](../metricdp_pytorch/globaldp_strategy.py).

Smaller `d_t` therefore means **more noise**, while larger `d_t` means **less noise**. Some
explanatory passages in the paper describe the opposite direction; its formula, Algorithm 1,
and this implementation agree on division by distance. Appendix G's description of measuring
distance after clipping also differs from Algorithm 1 and the current code. Use the actual
implemented order when interpreting the results.

### 3.2 Project history that matters

- The project reproduces and extends the source mechanism using Flower and PyTorch. The paper
  reproduction has aggregator-specific successes and failures; do not describe every published
  result as reproduced. See [paper_reproduction.md](paper_reproduction.md).
- The CUDA constant-compute controls show metric privacy's utility advantage shrinking toward
  parity as client count increases. Older dramatic reversals were partly affected by MPS
  nondeterminism and aggregation reply ordering. See
  [constant_compute_scaling.md](constant_compute_scaling.md).
- The noise-by-client-count sweep identifies fragility near the noise ceiling. Its exact
  boundary estimates are single-seed findings. See [noise_by_clients.md](noise_by_clients.md).
- CIFAR-100 and EuroSAT provide additional context, with different models and protocols.
  CIFAR-100's retained CIA results are seed-42-only. See
  [cifar-100_and_eurosat_results.tex](cifar-100_and_eurosat_results.tex) and
  [eurosat_cia.md](eurosat_cia.md).
- The AUC-targeted sweep is complete and merged into master. This new defense experiment is
  separate and has not started training.

### 3.3 What the frontier actually measures

The frontier contains four datasets, two partition modes, and two swept privacy modes: **16
curves**, with statuses **10 landed, 4 collapsed-before-target, 2 anchor-not-found**. It uses
FedAvg, 48 clients for EuroSAT/Alzheimer/Fashion-MNIST, and 100 for CIFAR-10. Fashion-MNIST is the
repository's four-class subset. The first three datasets use 100 training rounds with 11 attack
checkpoints; CIFAR-10 uses 20 rounds with 20 attack checkpoints.

The x-axis is a direction-reversed **paired round-matched score**. For one seed, it averages
whether negative shadow loss is larger in IN than OUT at matching checkpoint rounds (ties count
as one half), then reports `max(q, 1-q)`. This is not ordinary pooled ROC AUC over all IN/OUT
score combinations. Rounds on the same trajectory are correlated. Selecting the direction on
the evaluated data also affects its sampling behavior. Preserve this metric for historical
comparability, but do not equate it with a validated general-purpose attack or treat rounds as
independent training repetitions. See [score_stage.py](../experiments/cia/scripts/score_stage.py).

The y-axis averages **final-round server accuracy across IN and OUT**. Landing summaries below
then average those values across seeds 42, 43, and 44. The attack-score means similarly average
the already direction-reversed per-seed scores; they are not scores from pooling all losses.
Search stages use seed 42; only selected landing points receive two further seeds.

The scalable `non-iid` partition is quantity-skewed, with random class composition, rather than
a dedicated Dirichlet label-skew benchmark. In the sweep, the target canonical partition ID is
always 0, although its samples change with seed. See
[split_data.py](../metricdp_pytorch/utils/split_data.py) and the dataset remove scripts.

### 3.4 Selected endpoint evidence

These are observed endpoints, **not comparisons at equal achieved protection**. Values in the
mechanism columns are three-seed mean **accuracy / attack score** for landed curves.

| Dataset / partition | Vanilla accuracy | Global-DP | Metric privacy |
|---|---:|---:|---:|
| EuroSAT / homogeneous | 90.0% | 89.6% / 0.697 | 84.5% / 0.576 |
| EuroSAT / non-IID | 90.9% | 89.4% / 0.636 | 86.6% / 0.576 |
| Alzheimer / homogeneous | 91.3% | 86.9% / 0.758 | Collapsed before target |
| Alzheimer / non-IID | 90.7% | Collapsed before target | Collapsed before target |
| Fashion-MNIST / homogeneous | 95.4% | Anchor not found | Anchor not found |
| Fashion-MNIST / non-IID | 95.5% | 94.7% / 0.606 | Collapsed before target |
| CIFAR-10 / homogeneous | 55.5% | 17.2% / 0.583 | 46.0% / 0.717 |
| CIFAR-10 / non-IID | 56.5% | 56.6% / 0.700 | 27.6% / 0.550 |

Sources: [auc_target_sweep](../results/auc_target_sweep/),
[sweep report](auc_targeted_noise_sweep.md), and
[frontier generator](build_auc_frontier.py).

### 3.5 Findings from the preparation audit on 2026-09-24

The agent recomputed all **119 saved stages (238 training trajectories)**: 24 vanilla stages,
47 global-DP stages, and 48 metric-privacy stages. These contained 1,552 matched checkpoint pairs.
All clean-shadow losses were finite and IN/OUT checkpoint round lists matched. All 110 stage
entries in the state files, including repeated anchor/search entries, matched recomputed scores
and accuracies. Regenerated dataset sections and point-table text matched the existing HTML
without rewriting it. These were read-only artifact checks, not training or a test-suite run.

The resulting interpretation is more cautious than parts of the older report narrative:

1. **Only 1 of 10 landed curves has a three-seed mean within the target band.** That is
   CIFAR-10/non-IID/metric-privacy, at 0.550 with 27.6% accuracy. A `landed` status records a
   successful seed-42 search, not confirmation that attack resistance generalized. For example,
   Alzheimer/homogeneous/global-DP has per-seed scores 0.545, 0.727, and 1.000.
2. **Fashion-MNIST homogeneous anchor rejection compares different seed summaries.** The seed-42
   vanilla score is already 1.000, identical to all five low-noise attempts for either mechanism.
   The anchor search instead compares against the three-seed vanilla mean, 0.757576, with a
   tolerance of 0.10. Vanilla's other two effective scores are 0.727273 and 0.545455. Thus the
   rejection can be explained by the reference mismatch; it does not establish the absence of a
   usable low-noise anchor. The step-up sweep never ran for these two curves.
3. **Actual calibration failures are visible in diagnostics.** At Alzheimer/homogeneous/
   metric-privacy's final searched ratio, approximately 0.00498142, seed 42's IN trajectory
   records 68 skipped aggregation rounds out of 100, and OUT records 70. The IN noise standard
   deviation reaches approximately 6.39. These are observations of collapse and noise spikes;
   a complete causal diagnosis still requires examining their temporal ordering and controls.
4. **Better endpoint accuracy does not establish a better frontier.** CIFAR-10 homogeneous
   metric privacy preserves more accuracy than global-DP, but its mean attack score is also
   higher (0.717 versus 0.583). EuroSAT's retained utility does not demonstrate reliable attack
   neutralization: every confirmed mean is above 0.55.
5. **The search grid is incomplete evidence about feasibility.** Doubling noise can miss useful
   intermediate values, and collapse labels concern the searched path. They are not proofs that
   no successful setting exists. Alzheimer's 25% uniform-guessing collapse threshold also does
   not detect every degenerate model: majority-class prediction can achieve about 49.5%.

These qualifications apply when reading the existing reports and STATUS historical summaries;
those historical files have not been rewritten as part of this handoff.

### 3.6 First diagnostic: what existing scalar logs can tell us

**Question:** do the currently logged distance, clipping, and noise measures distinguish
trajectories with different CIA scores? This was a read-only comparison of selected existing
`search_state.json`, `cia.json`, and per-trajectory run JSONs on 2026-09-24. Each row below uses
the seed-42 IN trajectory's `train_metrics` for the server diagnostics; attack score and accuracy
come from the saved IN/OUT stage. `d+` is the median of strictly positive finite raw
`metric-dp-distance` values. The noise/signal ratio is the median of finite logged per-round
`dp-noise-to-signal-ratio` values. Skips count `metric-dp-aggregation-collapsed` rounds out of
the trajectory's round budget. These medians summarize different parts of the same training
history; none is an independent seed-level uncertainty estimate.

| Dataset / partition, metric privacy | Noise ratio | Seed | Accuracy | Attack score | Median `d+` | Median noise/signal | Skipped rounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Alzheimer / homogeneous, low noise | 0.00007783 | 42 | 91.3% | 0.909 | 0.489 | 3.13 | 0/100 |
| Alzheimer / homogeneous, collapsed endpoint | 0.00498142 | 42 | 8.4% | 0.818 | 0.422 | 117.13 | 68/100 |
| Fashion-MNIST / non-IID, low noise | 0.00066911 | 42 | 94.9% | 1.000 | 0.667 | 2.98 | 0/100 |
| Fashion-MNIST / non-IID, collapsed endpoint | 0.042823 | 42 | 25.0% | 0.545 | 0.645 | 124.13 | 96/100 |
| CIFAR-10 / homogeneous, same landing ratio | 0.005 | 42 | 48.5% | 0.550 | 1.026 | 13.58 | 0/20 |
| CIFAR-10 / homogeneous, same landing ratio | 0.005 | 43 | 44.9% | 0.900 | 1.029 | 13.75 | 0/20 |
| CIFAR-10 / homogeneous, same landing ratio | 0.005 | 44 | 44.4% | 0.700 | 1.022 | 14.12 | 0/20 |

Three observations guide the next probe:

1. **The scalar distance and median noise ratio do not determine the measured attack score.**
   CIFAR-10's three confirmation seeds have closely similar median distance (1.022–1.029) and
   noise/signal ratio (13.58–14.12) at the same noise setting, while the attack score spans
   0.550–0.900. Their target partitions, learned trajectories, and noise draws differ, so this
   is a diagnostic contrast, not proof that distance has zero predictive value. The median
   matched shadow-loss gap (`OUT loss - IN loss`) also varies across those seeds: approximately
   0.0075, 0.0634, and 0.0245.
2. **Collapse is observable in the per-round diagnostics.** The Alzheimer endpoint skips 68
   IN rounds and 70 OUT rounds; the Fashion-MNIST endpoint skips 96 rounds in both. A positive
   distance median conceals zero-distance or unusable rounds because `d+` excludes zero. Attack
   scores from these collapsed states cannot establish that a working model was protected.
3. **The existing logs cannot test the selected directional hypothesis.** `dp-update-norms-before-clipping`
   records magnitudes, and metric privacy records pairwise **pre-clipping** distances. It does not
   retain clipped update vectors, their inner products, or the resulting per-client removal
   vectors `v_i`. Checkpoints are deleted after attack evaluation. No offline reconstruction of
   the clipped influence directions from these scalar artifacts is justified.

The comparison uses one historically selected target (canonical ID 0) and a seed-42 search
path, so it should guide measurement design rather than rank new defenses. It also compares
noise/signal ratios measured against aggregate update magnitude, not a privacy parameter.

**Smallest next server measurement:** during FedAvg aggregation, form each already-clipped
`u_i`, its existing example-count weight `a_i`, and the round aggregate `u_bar` in memory.
Compute the Gram matrix `G_ij = <v_i, v_j>` for all client-removal effects `v_i`, plus
`<v_i, u_bar>`, `||u_bar||`, client weights, and a record of clipping/skip status. These
summaries permit a small-matrix spectrum and individual influence/aggregate-alignment analysis
without retaining millions of model coordinates. Sort by canonical client ID for determinism.
Emit compact per-round summaries for an initial, bounded pilot; do not add full updates or a
new noise rule to the first geometry diagnostic. A Gram matrix alone cannot prove that a
direction reveals participation to the attacker. That requires subsequent shadow-loss or
stronger held-out attack evaluation, and possibly separately retained vectors or evaluation-only
projections at selected checkpoints.

**Concrete first-pilot proposal for review:** run one 10-round, 48-client Fashion-MNIST/non-IID
FedAvg trajectory at seed 42 using the already studied low noise ratio `0.0006691085733778867`.
This uses the smallest model in the frontier (168,676 parameters) and a healthy, strongly leaky
historical setting. It is a geometry feasibility run, so its 10-round output cannot be compared
as an attack-AUC result against the existing 100-round frontier. At each round, retain the
client IDs and example counts, `G`, aggregate-alignment inner products, and singular values
derived from `G`; record whether clipping or aggregation failed. Cap the diagnostic to the
chosen run so this quadratic client calculation does not change all future experiments.
Inspect runtime and disk use before selecting any second case. A later contrast could use the
same dataset's high-noise collapse setting, but that has not been scheduled.

**Implementation point checked against installed Flower 1.32.1:**
`DifferentialPrivacyServerSideFixedClipping.aggregate_train` clips each successful reply's
model arrays in place before delegating to FedAvg. `FedAvg.aggregate_train` then weights those
arrays by the MetricRecord's `num-examples` field. A diagnostic wrapper can read the clipped
replies after the base aggregation succeeds and calculate the weighted `u_i`/`v_i` values
without altering aggregation. Preserve the current deterministic client-ID ordering and omit
the geometry for skipped/failed rounds. The first implementation should expose an explicit
opt-in switch rather than adding an `n²` diagnostic to every production round. The switch and
output schema now follow.

**Instrumentation now on this branch (2026-09-24):** pass
`--record-influence-geometry` to `experiments.reproduce.runner`. It is restricted to
`metric-privacy` with `fedavg` and at most 64 clients; other runs keep the default off. After
Flower clips the client models and aggregates, the server reconstructs each clipped update
relative to the pre-round global model, uses Flower's `num-examples` weights, and records the
client-removal effect `v_i` defined in section 5.1. The round's `train_metrics` contains:

| Field suffix after `metric-dp-influence-` | Meaning |
|---|---|
| `recorded` | 1 when a successful aggregate has geometry, 0 for a collapsed aggregate; absent if Flower returns early on client errors |
| `client-ids`, `example-counts`, `weights` | Same client order, sorted by logical ID; weights sum to one |
| `client-count`, `gram-flat` | `n`, then all `n²` values of `<v_i,v_j>` in row-major order |
| `align-with-aggregate`, `aggregate-norm` | `<v_i,u_bar>` for each client and `||u_bar||` |
| `singular-values` | Descending singular values of the client-removal-effect matrix, obtained from the Gram eigenvalues |

Existing `dp-client-clipped`, `dp-update-norms-before-clipping`, and
`metric-dp-aggregation-collapsed` fields provide clipping/skip context. This is a private
server diagnostic with a quadratic number of logged values; it does not alter the output noise
rule or make a privacy claim. It depends on the installed Flower 1.32.1 wrapper mutating the
replies to their clipped values before returning from aggregation. A future Flower upgrade
needs that behavior rechecked.

The exact **IN-only, geometry-only** pilot command from the repository root is:

```bash
uv run python -m experiments.reproduce.runner \
  --partition non-iid --privacy metric-privacy --aggregation fedavg \
  --num-clients 48 --rounds 10 --local-epochs 5 --batch-size 32 \
  --learning-rate 0.001 --seed 42 \
  --noise-multiplier 0.03211721152213856 --clipping-norm 5 \
  --initialization-epochs 20 \
  --data-module experiments.cia.scripts.fashion_mnist_remove:create_in_remove \
  --model-module experiments.reproduce.fashion_mnist_cnn:create_model \
  --max-parallel-clients 6 --record-influence-geometry \
  --output-dir results/cia_influence_defense/pilot \
  --run-name fashion-noniid-in-seed42-geometry-10r
```

The multiplier above is 48 times the historical ratio `0.0006691085733778867`, matching
the existing 48-client IN trajectory. The expected result is
`results/cia_influence_defense/pilot/fashion-noniid-in-seed42-geometry-10r.json`, with its
ordinary evaluation artifacts beside it. This command does not run OUT adjacency or the CIA
scorer and cannot yield a 10-round AUC comparable with `auc_frontier.html`. First inspect that
all ten `train_metrics` rounds have `metric-dp-influence-recorded=1`, that stored weights sum
to one, that the Gram matrix is symmetric and positive semidefinite to numerical tolerance,
and that the timing/disk cost is acceptable. Then analyze spectral concentration, alignment
with the aggregate, and per-client influence magnitude before choosing a noise rule. The branch
revision containing the instrumentation is `9a973bb`. The pilot ran locally from approximately
17:31 to 17:35 on 2026-09-24 (local time) on GPU 0 in detached `tmux` session
`cia-influence-pilot`; the session ended normally. The operational log is
`/tmp/cia-influence-pilot-20260924.log` (local and untracked). The GPU devices were available
to the process outside the filesystem sandbox; this explains the earlier false conclusion
from sandboxed `nvidia-smi`. The earlier remote SSH attempt failed before checkout or GPU
inspection, but remote access was not needed for this pilot.

**Observed pilot output.** The locally stored [run JSON](../results/cia_influence_defense/pilot/fashion-noniid-in-seed42-geometry-10r.json)
and [evaluation JSON](../results/cia_influence_defense/pilot/fashion-noniid-in-seed42-geometry-10r.evaluation.json)
contain the real values below. These raw files remain local because automatic approval review
rejected pushing per-client training diagnostics to the remote GitHub repository; a remote agent
must rerun the command or obtain an approved artifact transfer to inspect the full Gram matrices.
All ten rounds have 48 sorted client IDs and
`metric-dp-influence-recorded=1`; their weights sum to one, each Gram matrix is symmetric,
finite, and positive semidefinite to numerical tolerance (smallest eigenvalue across rounds
at least `-2.7e-18`). The evaluation's postprocessed accuracy agrees with the recorded final
server accuracy of **94.15%**. The predictions NPZ was produced locally but is excluded by the
repository's `results/**/*.npz` ignore rule; the geometry and aggregate metrics are in the
local JSON artifacts.

| Round | Clients clipped | Top 1 influence energy | Top 5 influence energy | Target ID 0 norm rank / 48 | Target norm / median | Server accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22 | 9.0% | 30.7% | 16 | 1.32 | 84.20% |
| 2 | 0 | 9.4% | 33.8% | 11 | 1.47 | 89.20% |
| 3 | 0 | 7.0% | 30.2% | 12 | 1.44 | 90.95% |
| 4 | 0 | 6.5% | 28.4% | 11 | 1.50 | 91.90% |
| 5 | 0 | 6.3% | 28.0% | 14 | 1.38 | 92.60% |
| 6 | 0 | 6.2% | 28.4% | 11 | 1.47 | 93.10% |
| 7 | 0 | 6.2% | 28.0% | 16 | 1.36 | 93.25% |
| 8 | 0 | 5.9% | 27.3% | 16 | 1.34 | 93.60% |
| 9 | 0 | 6.1% | 27.6% | 12 | 1.37 | 93.75% |
| 10 | 0 | 6.0% | 27.9% | 13 | 1.35 | 94.15% |

Here "energy" means the sum of the largest eigenvalues of `G` divided by `trace(G)`,
equivalently the squared singular-value share of the matrix of `v_i` vectors. Rank 1 means
largest removal-effect norm. In this one healthy trajectory, no single direction dominates:
the top five explain only 27.3–33.8% of influence energy, although the span is structurally
at most 47-dimensional. Target ID 0 has an above-median removal effect, but its rank stays
between 11 and 16 rather than standing out as the largest. These facts constrain the simplest
"noise only in the top few directions" proposal: a small basis would discard most measured
influence energy. They do **not** show whether the discarded directions carry CIA signal,
whether the full influence-shaped covariance protects against CIA, or how any candidate trades
privacy against accuracy. The pilot has no OUT adjacency and no attack score. Next analyze
alignment and target-loss sensitivity before selecting a covariance rule, and use independent
seeds and held-out attacks for any defense claim.

## 4. Candidates discussed and current selection

| Candidate | Proposed server change | Motivation | Main uncertainty | Decision |
|---|---|---|---|---|
| Noise shaped by client influence | Allocate noise using directions and magnitudes of individual client-removal effects | Obscure participation with less damage than isotropic perturbation | Influence directions may also contain essential learning signal; geometry itself is private | **Selected first research direction** |
| Limit individual influence before aggregation | Reduce contributions that disproportionately change the aggregate | Reduce participation signal before adding noise | May suppress useful unusual/minority clients; recomputing weights or the center changes sensitivity | Retain as alternative |
| Stabilize existing calibration | Bound/smooth the inverse-distance scale, possibly normalize its statistic | Address observed noise blow-ups | Improved stability may leave leakage intact; floors/caps change the mechanism | Proposed control, not yet implemented |

Adaptive clipping is also an important established comparison candidate. Neither adaptive
clipping nor directional/anisotropic noise alone should be presented as a novel contribution.

The working hypothesis is that **maximum pairwise model distance is an inadequate proxy for
the detectable effect of an individual client's participation**. It describes client spread,
but does not directly account for aggregation weights, the target's removal effect, or how
parameter changes affect the attacker's loss. This hypothesis is not yet validated by the data.

## 5. Selected direction: noise shaped by client influence

### 5.1 A concrete quantity available to the server

Let `u_i` be client `i`'s update after the existing norm clipping, and `a_i` its normalized
FedAvg weight, with `sum_i a_i = 1` and `a_i < 1`. Define

\[
\bar u = \sum_i a_i u_i, \qquad
\bar u_{-i} = \frac{\bar u-a_i u_i}{1-a_i}, \qquad
v_i = \bar u-\bar u_{-i} = \frac{a_i}{1-a_i}(u_i-\bar u).
\]

This is the exact change in this round's weighted average if client `i` is omitted and the
remaining weights are renormalized, **holding their submitted updates fixed**. Compute it for
every client, not just the evaluation target. No attacker shadow data is needed.

This identity is specific to that averaging operation. It is not the full IN/OUT trajectory
difference: removing a client changes future global models and therefore future local updates.
If a proposed defense changes weights, the clipping rule, or its center after removal, its
actual output difference must include those changes; the identity alone no longer describes it.

For a differentiable shadow loss and sufficiently small perturbations, a local first-order
approximation is `loss(w + v_i) - loss(w) ≈ grad(loss(w))^T v_i`. This explains why direction
could matter. The server does not know the attacker's loss gradient, and this approximation is
not a guarantee or a replacement for full-trajectory evaluation.

### 5.2 Design sketch to discuss, not an approved mechanism

One possible family is Gaussian noise with an isotropic component plus extra variance along
selected influence directions. For an appropriately scaled matrix `V` whose columns represent
influence vectors, an illustrative covariance is

\[
\Sigma_t = \sigma_{0,t}^2 I + \tau_t^2 V_t V_t^\top.
\]

It could be sampled as `sigma_0 * z + tau * V * b`, with independent standard Gaussian vectors
`z` and `b`, without materializing a parameter-by-parameter covariance matrix. The column scaling,
whether to retain magnitudes, use of a reduced basis, and both noise scales remain undecided.
This family was added here to make the selected direction concrete for discussion; the owner
has approved the direction, not this particular covariance formula.

The following distinctions are essential:

- **Low rank alone is automatic.** For weighted FedAvg, `sum_i (1-a_i) v_i = 0`, so the influence
  matrix has rank at most `n-1`. With dozens of clients and millions of parameters this already
  gives a small span. The empirical question is whether there is further spectral concentration
  and a useful separation between participation information and task utility.
- **Important learning directions may coincide with revealing directions.** Adding more noise
  there could be especially damaging. Conversely, a low-variance direction may identify a rare
  client; discarding it based on explained variance could miss leakage.
- **Private geometry can leak.** Covariance and subspace selection depend on private updates.
  An isotropic floor avoids a completely unnoised complement, but does not by itself establish
  DP, correct composition, or resistance to covariance-based attacks.
- **Magnitude and schedule matter.** A covariance rule that simply tracks shrinking updates
  could reduce noise while participation evidence accumulates. There is no reason to assume a
  per-round heuristic protects the whole released trajectory.
- **Equal noise energy is a diagnostic control, not equal privacy.** A directional mechanism
  must be compared at matched achieved attack strength as well as across noise budgets.

## 6. Questions to resolve next

### Mechanism and diagnosis

1. Does the spectrum of clipped client-removal effects concentrate beyond the automatic rank
   bound? Does that pattern persist across rounds, seeds, targets, and partitions?
2. Which influence statistics predict the observed IN/OUT shadow-loss separation better than
   the existing maximum pairwise distance?
3. How much do revealing directions overlap with useful aggregate updates? Is there room to
   perturb participation information without suppressing rare but useful client contributions?
4. Should geometry be computed for the full model or per layer? How should layers and client
   aggregation weights be normalized?
5. Should the first prototype retain all influence directions or a reduced basis? What protects
   low-variance directions associated with unusual clients?
6. How should the isotropic component, directional strength, and round schedule be chosen?
   Would smoothing help stability while preserving meaningful changes in influence?
7. Is a public/server validation dataset needed for utility-aware shaping? Its availability and
   role have not been agreed, and final-test data must not become a tuning signal.

### Evaluation and claims

8. Which attack will be primary beyond the historical paired-round diagnostic? How will score
   direction and thresholds be fitted on calibration data and evaluated on held-out trajectories?
9. How many independent seeds and target clients are needed? Checkpoint rounds are not substitutes
   for independent repetitions, and three seeds were insufficient for stable conclusions here.
10. Can an attacker exploit covariance, update norms, or combinations of rounds even when the
    current shadow-loss statistic approaches 0.5?
11. What utility floor, leakage target, uncertainty criterion, and collapse definition should
    determine success? How will minority-class performance be included for Alzheimer?
12. How will IN/OUT randomness and mechanism parameters be coupled? Existing sweep runners match
    `noise_multiplier / active_clients`, adjusting the multiplier when the target is removed.
    This equalizes global-DP's nominal standard deviation across the pair, not its multiplier.
    Specify the convention explicitly for the new defense and consider independent-randomness
    evaluation in addition to paired diagnostics.
13. What is the contribution relative to existing influence estimation, adaptive clipping,
    directional noise, and inference-defense literature? Novelty is still unverified.

## 7. Proposed next steps and decision points

This is the sequence after the bounded pilot above. A larger experiment matrix and budget have
not been set.

1. **Use the completed first diagnostic.** Section 3.6 compares selected existing scalar logs,
   attack gaps, and confirmation seeds. It identifies what the logs can and cannot answer.
2. **Use the completed compact geometry pilot.** Section 3.6 records completeness, Gram
   consistency, spectral concentration, clipping status, and its limited interpretation. The
   raw clipped vectors are not stored. Do not infer protection from the geometry alone.
3. **Choose the simplest prototype that answers the hypothesis.** Keep client training fixed.
   Start from a clearly specified FedAvg server operation and include all data-dependent parts
   in the counterfactual analysis.
4. **Design controls before a larger sweep.** Candidate comparisons: vanilla, global-DP,
   original metric privacy, stabilized scalar calibration, and the influence-based candidate.
   Consider adaptive clipping. Useful ablations include isotropic noise at matched expected
   squared norm, random directions with a matched spectrum, and removing the directional term.
5. **Set evaluation and budget.** Decide independent repetitions, target selection, tuning versus
   held-out evaluation, attack methods, and stopping rules. Search points and confirmation points
   must remain distinguishable. Do not stop at one favorable seed.
6. **Evaluate whether the idea survives.** Support requires a repeatable utility/leakage improvement
   against appropriate controls. Reject or revise the hypothesis if the apparent advantage is
   explained by weaker attacks, greater unmeasured leakage, reduced learning, or unstable seeds.
   If directions do not help, revisit influence-limiting aggregation rather than silently changing
   the selected research story.

The geometry measurement is now implemented as an opt-in probe. The 10-round pilot is complete.
A defense mechanism, full attack comparison, and larger
experiment budget still require a separate design decision; do not silently treat this
diagnostic as evidence that the proposed noise covariance works.

## 8. Available artifacts and implementation map

| Resource | What it provides / limitation |
|---|---|
| [auc_frontier.html](auc_frontier.html) | Shared reference; points from the existing sweep, not new defense results |
| [results/auc_target_sweep/](../results/auc_target_sweep/) | Search state, vanilla references, per-round attack losses, and training metrics |
| [metricdp_strategy.py](../metricdp_pytorch/metricdp_strategy.py) | Current distance calibration and per-pair diagnostics; full pair lists only at client counts up to 64 |
| [dp_diagnostics.py](../metricdp_pytorch/dp_diagnostics.py) | Update norms, clipping, aggregate signal norm, expected noise magnitude; no directional vectors |
| [strategy_factory.py](../metricdp_pytorch/strategy_factory.py) | Privacy-mode/aggregator construction and deterministic reply ordering |
| [score_stage.py](../experiments/cia/scripts/score_stage.py) | Historical paired-round attack score and IN/OUT final accuracy |
| [auc_target_search.py](../experiments/cia/scripts/auc_target_search.py) | Existing seed-42 search and seeds-43/44 confirmation; reference mismatch described above |
| [partitions.py](../experiments/cia/datasets/partitions.py) | Canonical IN/OUT removal and replacement partition views |
| [attack_runner.py](../experiments/cia/attack_runner.py) | Checkpoint evaluation; checkpoints are deleted after scoring |
| [cia_threat_model.tex](cia_threat_model.tex) | Existing trusted-server, client-participation threat model |

For the eventual experiment, use `experiments/cia_influence_defense/` and
`results/cia_influence_defense/` as proposed locations, unless the implementation plan establishes
a better fit with the existing CIA modules. These directories do not exist as new deliverables
yet. Keep the current frontier and original result artifacts intact for comparison.

Use `uv run` for Python commands. CUDA is the reference experimental platform given the documented
MPS reproducibility problems. The CUDA pilot is the first training run exercising this
instrumentation; its artifact checks are described in section 3.6. No test suite has been run
for it.

## 9. Initial literature pointers and novelty limits

The preparation read the source paper and made a small web search of related directions.
The following primary publication pages were located; this is **not a completed literature
review or novelty assessment**. Apart from the source paper, these need fuller methodological
reading before comparing a proposed algorithm or making a novelty claim.

- [Source mechanism and CIA paper](https://doi.org/10.1016/j.knosys.2026.115993): read locally;
  empirical inverse-distance Gaussian calibration and the initial CIA threat model.
- [Differentially Private Learning with Adaptive Clipping](https://proceedings.neurips.cc/paper/2021/hash/91cff01af640a24e7f9f7a5ab407889f-Abstract.html):
  established adaptive-clipping baseline; publication abstract checked.
- [Toward Understanding the Influence of Individual Clients in Federated Learning](https://ojs.aaai.org/index.php/AAAI/article/view/17263):
  client-influence estimation is existing work; publication page checked. Its full method has
  not been compared to the proposed per-round removal geometry.
- [Understanding Clipping for Federated Learning: Convergence and Client-Level Differential Privacy](https://proceedings.mlr.press/v162/zhang22b.html):
  relevant clipping/heterogeneity analysis; surfaced in the primary proceedings search.
- [Fisher-driven privacy preservation against category inference attacks in federated learning](https://doi.org/10.1016/j.hcc.2026.100399):
  primary publisher search result describes Fisher-guided anisotropic Gaussian noise. **Category
  inference is not the same target as client-participation inference**, despite the potentially
  confusing acronym. Full-text comparison remains pending.

A possible contribution is aligning server perturbation with client-removal effects and showing
that this improves participation-inference resistance at useful accuracy. Neither that novelty
nor the proposed benefit has been established.

## 10. Reproducing the main audit conclusions

This read-only command derives the endpoint means and Fashion-MNIST seed-42 reference directly
from committed JSONs. Run from the repository root; it does not train or overwrite reports.

```bash
uv run python - <<'PY'
import json
import statistics
from pathlib import Path

root = Path('results/auc_target_sweep')
inside = 0
landed = 0
for path in sorted(root.glob('*/*/*/search_state.json')):
    state = json.loads(path.read_text())
    if state['status'] != 'landed':
        continue
    points = [p for p in state['search_stages']
              if p['noise_ratio'] == state['landing_ratio']]
    points += state['confirmation_stages']
    attack = statistics.fmean(p['auc'] for p in points)
    accuracy = statistics.fmean(p['accuracy'] for p in points)
    landed += 1
    inside += 0.45 <= attack <= 0.55 + 1e-12
    print(path.parent.relative_to(root), accuracy, attack)
print('Three-seed means in target band:', inside, 'of', landed)

base = root / 'fashion-mnist/homogeneous'
print('Stored vanilla mean:', json.loads((base / 'vanilla_reference.json').read_text()))
rows = json.loads((base / 'vanilla/nm-vanilla-seed-42/runs/cia.json').read_text())
left = sorted([r for r in rows if '-in-remove' in r['run_name']],
              key=lambda r: r['server_round'])
right = sorted([r for r in rows if '-out-remove' in r['run_name']],
               key=lambda r: r['server_round'])
assert [r['server_round'] for r in left] == [r['server_round'] for r in right]
q = statistics.fmean(
    1.0 if i['target_clean_shadow_loss'] < o['target_clean_shadow_loss'] else
    0.5 if i['target_clean_shadow_loss'] == o['target_clean_shadow_loss'] else 0.0
    for i, o in zip(left, right, strict=True)
)
print('Seed-42 vanilla effective score:', max(q, 1-q))
PY
```

For the collapse example, inspect both trajectory JSONs under
`results/auc_target_sweep/alzheimer/homogeneous/metric-privacy/nm-0p00498142-seed-42/runs/`.
Sum `metric-dp-aggregation-collapsed` in `train_metrics` and inspect
`metric-dp-distance`, `metric-dp-noise-stdv`, and `dp-noise-to-signal-ratio` across rounds.

## 11. Decision log and current handoff

| Date | Decision / observation | State |
|---|---|---|
| 2026-09-24 | Owner requested project/paper/result review before brainstorming, centered on `auc_frontier.html` | Review performed; key qualifications recorded above |
| 2026-09-24 | Owner selected server-side-only scope | Agreed |
| 2026-09-24 | Discussed influence-shaped noise, influence-limiting aggregation, and stabilization of existing calibration | All retained in this report |
| 2026-09-24 | Owner accepted influence-shaped noise as the first direction to try | Direction selected; exact mechanism open |
| 2026-09-24 | Owner requested a separate branch and a report for future agents | This branch and report created |
| 2026-09-24 | Owner asked to start; read-only scalar diagnostic compared selected existing stages and identified missing clipped-update geometry | Findings and next measurement recorded in section 3.6; no new run |
| 2026-09-24 | Implemented opt-in post-clipping influence geometry and specified a 10-round IN-only pilot | CUDA pilot pending; no defense or new result claimed |
| 2026-09-24 | Pushed probe revision `9a973bb`; configured remote GPU SSH rejected available public key | No checkout/GPU inspection or pilot launch occurred |
| 2026-09-24 | User clarified local CUDA availability; outside-sandbox checks found two RTX 5000 Ada GPUs and PyTorch CUDA access; started 10-round IN pilot on idle GPU 0 | Running in `cia-influence-pilot`; results pending |
| 2026-09-24 | First 10-round IN-only geometry pilot finished on local CUDA GPU 0; all ten Gram diagnostics present and final server accuracy 94.15% | Pilot artifacts and cautious interpretation recorded above; no CIA score or defense comparison |
| 2026-09-24 | Automatic approval review rejected a push of the raw pilot JSON files because they contain per-client metrics and the remote destination was unverified | Raw artifacts kept local; report summary can be pushed separately |

**Next interaction:** use the pilot geometry and section 6 to choose and scrutinize the first
noise rule and its controls. Add an attack-linked measurement before treating a low-dimensional
noise basis as promising.
Do not claim that a defense, formal guarantee, novelty assessment, or new defense finding is
already complete.

This report was drafted with AI assistance from the project owner's discussion, repository
artifacts, source paper, and the explicitly limited literature search described above.
