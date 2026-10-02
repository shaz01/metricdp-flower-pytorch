# New EuroSAT AUC frontier — implementation ready, NOT executed

Branch: `runs/new-auc-frontier-eurosat`. Only EuroSAT, 48 canonical clients,
FedAvg, label-Dirichlet non-IID. The previous `non-iid` EuroSAT plugin was
**quantity skew**, not label Dirichlet. Old results are context, not interchangeable
observations. Existing experiment scripts and results are unchanged.

## Protocol

- Preserve the old EuroSAT model/hyperparameters: 100 rounds, 5 local epochs,
  20 initialization epochs, learning rate .001, batch size 32, clipping norm 5.
- One full-federation IN training per (alpha, mechanism, noise ratio, seed).
  For each selected canonical target ID, train one OUT model with **only that
  client removed**, leaving all other partitions unchanged. Default targets 0–19;
  these IDs are specified before viewing losses, never selected by attack success.
- One seed controls partition generation and training, as in the existing setup.
  Thus client ID 0 across seeds is an exchangeable partition slot, NOT identical
  records across seeds. IN/OUT within a seed have identical canonical partitions.
- Per-class Dirichlet(alpha) allocation with multinomial counts, deterministic
  bounded rejection until every client has >=10 records. This induces quantity
  variation too; it is not a fixed-size label-skew partitioner. Rejection and the
  minimum are part of the protocol. No fallback to quantity skew.
- Preserve existing 10% clean shadow **data** and 20%-std noisy shadow data
  construction; no ensemble of released models is given to the attacker. These
  are datasets for loss evaluation, not newly trained shadow models.
- Preserve old ratio convention: multiplier = noise_ratio × active clients
  (48 IN, 47 OUT). Fix this convention before running; changing it is a new protocol.
- Save clean/noisy target losses, aggregate test loss, shadow size at **every
  round 1–100**. Training JSON retains every-round model accuracy. Shared IN
  checkpoints are evaluated on ALL chosen targets before deletion. Checkpoint
  weights are deleted after measurements, not retained for new attacks later.
- Analysis uses only round 100. Lower clean-shadow loss means more IN-like,
  fixed before evaluation; no `max(AUC, 1-AUC)` and no picking rounds post hoc.

## Statistics: what we can and cannot claim

The old same-seed IN-vs-OUT win fraction is **paired concordance**, not ordinary
ROC AUC or classification accuracy. We report it honestly under that name.
Also report target-stratified ROC AUC: for each canonical target slot compare
all final IN seed scores with all final OUT seed scores, then average targets.
This avoids ranking different clients' intrinsically different loss scales against
each other. Repeated seeds estimate performance; the attacker sees one model per
trial, not the ensemble used by the evaluator.

Neither S*K paired comparisons nor S*S*K AUC comparisons are independent samples.
Uncertainty resamples **whole seed blocks**, retaining IN/OUT pairing and every
target, conditional on the chosen target IDs. Two seeds are discovery only;
intervals are deliberately omitted below five. Five-seed bootstrap intervals are
still exploratory and can be unstable/degenerate; they are NOT certification.
Inspect per-seed scores and increase independent seeds before proliferating OUT
clients when between-seed spread dominates. No automatic privacy success label,
no first-in-band stopping rule, and no claim of formal DP from attack failure.
AUC below .5 can indicate reversed leakage; do not call it private automatically.

The earlier conversation's sample-size/precision guarantees were optimistic.
A .45–.55 equivalence claim would need a separately powered, held-out experiment.
Do not treat OUT_a versus OUT_b as a free exact null: their training populations
actually differ. More checkpoints also do not create independent trials.

## Handoff: commands below PLAN ONLY unless `--execute` is added

Run from the repository root with `uv sync` installed. Do not launch any training
until the owner/assigned execution agent explicitly authorizes it. Planning does
not fetch data or initialize models. Synthetic unit tests need no dataset/GPU.

### 1. Alpha pilot (12 vanilla training runs, no attack)

Suggested alpha grid is provisional: .1, .3, 1, 10. Run three seeds per alpha.
Example for one alpha:

```bash
uv run python -m experiments.auc_frontier.runner --alpha .3 --seeds 42 43 44 \
  --privacy vanilla --alpha-pilot --output results/new_auc_frontier_eurosat/alpha_pilot
```

A pilot trains only full IN models; no target shadow evaluation is needed. It
retains per-round accuracy in the training JSON. Before selecting alpha, inspect
both learning curves and client sizes/class proportions (`partitions.json`
is saved for every trajectory). Choose a scientifically
meaningful heterogeneity level, not whichever alpha makes privacy look best.
Record the owner's selected alpha before starting the noise sweep. Do not reuse
quantity-skew results as a Dirichlet vanilla reference.

### 2. Discovery: selected alpha, two mechanisms, six ratios, two seeds

Example ratios span the earlier EuroSAT transition; they are provisional because
Dirichlet changes the problem. Lock or revise them after the alpha pilot, and
record the revision. Run this once per privacy mode:

```bash
uv run python -m experiments.auc_frontier.runner --alpha .3 --seeds 42 43 \
  --privacy global-dp --ratios .00019325 .0003865 .000773 .001546 .003092 .006184 \
  --output results/new_auc_frontier_eurosat/frontier
```

The `.3` here is an EXAMPLE, not a selected alpha. Repeat with `metric-privacy`.
Cost = 2 mechanisms × 6 ratios × 2 seeds × (1 IN + 20 OUT) = **504 trainings**.
Run the full chosen grid; do not stop when a single estimate hits .55.

### 3. Confirmation: add independent seeds at selected candidate levels

After reviewing discovery accuracy and seed variation, explicitly choose two
levels per mechanism. Supply their EXACT original ratio literals, same target
list and alpha, with seeds 44 45 46. Cost = 2 × 2 × 3 × 21 = **252 additional**
trainings. Analyze confirmation-only seeds separately when reporting evidence
for a level selected using discovery data; combined estimates are descriptive.
Neither analysis claims a selection-adjusted equivalence result.

Run a separate vanilla attack reference with five seeds and the same targets:

```bash
uv run python -m experiments.auc_frontier.runner --alpha .3 --seeds 42 43 44 45 46 \
  --privacy vanilla --output results/new_auc_frontier_eurosat/frontier
```

That is **105** additional trainings, not 60. Total planned budget with the
12-run alpha pilot is **873**, before any optional expansion/retries. Without
attack vanilla the pilot + defended sweep costs **768**. Every-round evaluation
adds substantial inference/checkpoint I/O even though training counts are unchanged.

### Execution, sharding and recovery

Assigned agents add `--execute` to the reviewed commands. Split work by mechanism,
ratio or seed. Do NOT split the target list within the same shared IN run. Each
trajectory has its own directory, immutable manifest, atomic measurement writes,
and completion marker. Local process locks reject concurrent writes to the same
trajectory. Across machines assign disjoint trajectories; filesystem locks do not
coordinate separate machines. Copy whole completed directories when collecting.

To shard one setting's IN and OUT trajectories across GPUs without changing
the panel, keep `--targets` fixed and add `--adjacency in` (shared IN only,
still evaluating every panel target) or `--adjacency out --out-targets 0`
(only the listed OUT trajectories; each must belong to `--targets`). Default
`--adjacency both` trains all 1+K trajectories. Give each shard its own
`--output` directory, e.g. `frontier/<mechanism>-r<ratio>-seed-<seed>-in` and
`...-out-<target>`; analysis searches manifests recursively. `complete.json`
records wall-clock `training_seconds` and `evaluation_seconds`.

Reissuing the same command skips complete measurements; incomplete trajectories
retrain in full, because some checkpoints may already have been deleted. Existing
complete runs with a different target list are rejected; expanding K needs a new
output root/re-evaluation plan. Increasing seeds does not have that limitation.
Failures propagate instead of silently counting partial trajectories as successes.
`provenance.json` records commit, Python version and device automatically; record
other environment details with each agent's execution handoff. Compare only runs
under the same protocol and backend policy.

### Analysis (no training)

```bash
uv run python -m experiments.auc_frontier.analyze results/new_auc_frontier_eurosat/frontier
uv run pytest experiments/auc_frontier/
```

Analysis prints JSON to stdout. It refuses incomplete/unbalanced designs and
includes final IN/OUT accuracies plus fixed-direction metrics and seed-cluster
intervals where supported. Store summaries alongside real results only after
runs exist. No experiment-finished report should be written without owner approval.
