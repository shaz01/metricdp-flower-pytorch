# Client Inference Attacks (CIA)

Implements both CIA protocols from Sáinz-Pardo Díaz et al. (2026): the
first-round relative-loss attack in Section 7.4.1 (Tables 9-12) and the paired
multi-round attack in Section 7.4.2 (Table 13).

## What this does

1. Partitions the Alzheimer MRI train set into 3 clients using the paper's
   Table 9 distribution (client 1 = attacker, client 2 = bystander, client 3
   = target). Table 9 as published has a typo in Client 2's row (1591/180);
   the corrected values (1491/80) are used here -- see
   `experiments/cia/datasets/paper.py`'s module docstring for the arithmetic
   proof (row sum, column sums against Table 1, and the grand total all
   reconcile only with the corrected values).
2. For each of the 18 `(privacy_mode, aggregation)` combinations, runs one
   real 1-round, 3-client Flower simulation with `local-epochs=20` (the
   paper's CIA-specific value), reusing `experiments.reproduce.runner` via
   `experiments.cia.datasets.paper:create_paper_shadow_data_module`.
3. Evaluates the resulting model's loss on the global held-out test set and
   on a stratified 10% shadow sample of the target's train indices (which
   overlaps with, not excluded from, what the target trains on -- matching
   the paper's stated "strong adversarial assumption").
4. Reports `(target_loss - aggregated_loss) / target_loss * 100` per
   combination, matching Tables 10-12's structure.

## Data-module structure

`experiments/cia/datasets/shadow.py:ShadowDataModule` is a dataset-independent
decorator: it delegates normal client/server loading to any federated data
module and derives the shadow set from the target client's actual training
loader. `experiments/cia/datasets/paper.py:PaperShadowDataModule` composes
that decorator with the corrected Table 9 partition. Scalable experiments
can instead wrap the standard Alzheimer module, or another future data
module, without duplicating shadow-split logic.

## Running the first-round attack

```bash
uv run python -m experiments.cia.scripts.runner --output-dir results/cia/first_round
```

This takes a while: 18 real training runs, each downloading/reusing the
cached Alzheimer MRI dataset and training 3 clients for 20 local epochs.
Results are written to `<output-dir>/first_round_cia.json` and printed to
stdout.

## Multi-round CIA (Table 13)

The multi-round runner trains a matched pair of 20-round FedAvg trajectories
for each privacy mode:

- **IN:** source clients 1, 2, and target client 3 participate.
- **OUT:** source clients 1 and 2 participate; client 3 is removed without
  repartitioning either remaining client's data.

At every round, the checkpoint is evaluated on the same deterministic,
stratified 10% shadow split of client 3's training data. Gaussian noise with
standard deviation 20% of each image's maximum pixel value is applied only to
the shadow view, never to federated training or utility evaluation. The
per-round score is `-mean_shadow_loss`. The reported AUC ranks the 20 IN scores
against the 20 OUT scores, and its 95% interval is a stratified percentile
bootstrap. Accuracy and weighted F1 are the aggregated client-test metrics,
reported as mean and population standard deviation over rounds 16-20.

```bash
uv run python -m experiments.cia.scripts.multi_round
```

Outputs go to `results/cia/multi_round/`. The runner caches evaluated
trajectories, atomically resumes completed privacy modes, and deletes the 20
large model checkpoints after extracting each trajectory.

## 48-client checkpoint comparison

The scalable CIA experiment now lives in `experiments/cia/scripts/client_scaling.py`.
It trains each 48-client trajectory once for 20 rounds with the concluded
client-scaling settings (`local-epochs=5`, `noise-multiplier=0.05`) and
retains checkpoints at rounds 1 and 20. Both attacks therefore evaluate the
same model trajectory rather than independently training a one-round model.

```bash
uv run python -m experiments.cia.scripts.client_scaling \
  --output-dir results/cia_client_scaling
```

Its matrix covers homogeneous/non-IID partitions, all three privacy modes,
and FedAvg/FedYogi. Results include the server round, client count, partition,
noise multiplier, shadow fraction, and realized shadow size, and are written to
`results/cia_client_scaling/cia_client_scaling.json`.

## Comparing against the paper

The paper's Table 10 (FedAvg) and Table 11 (FedYogi) report, per privacy
mode: aggregated test loss, target shadow loss, and the relative
difference. Table 12 reports test loss only, across all six aggregation
strategies. Compare the corresponding rows in
`experiments/cia/results/first_round_cia.json` against those tables. Exact
numeric parity is not expected (different hardware/library versions, and
the paper doesn't specify all stochastic-seed details), but the qualitative
pattern -- metric-privacy loss lower than global-DP loss, with comparable
CIA protection -- should hold.

## Colab CLI workflow

`scripts/colab/run_experiment.py` runs a Python experiment module on a named
Colab session, prints a live `nvidia-smi` snapshot while it trains, downloads
the designated result directory, commits only that directory, pushes the
current branch, and finally releases the VM. Git credentials never leave the
local machine.

The determinism check is a copy of the contest configuration with only seed
42. It executes the three privacy configurations twice under distinct
`check-determinism-run-1` and `check-determinism-run-2` names, then requires
exact equality after removing only the deliberately different run name:

```bash
uv run python scripts/colab/run_experiment.py run \
  --session cia-determinism \
  --gpu L4 \
  --module experiments.cia.scripts.check_determinism \
  --results results/cia/check_determinism \
  --commit-message "results(cia): add Colab determinism check"
```

The command stays attached so collection and pushing cannot be skipped. From
another terminal, inspect the training log and GPU utilization at any time:

```bash
uv run python scripts/colab/run_experiment.py status --session cia-determinism
```

If local monitoring is interrupted, the remote supervisor keeps training.
Resume the automatic collect/push/stop finalizer with:

```bash
uv run python scripts/colab/run_experiment.py wait --session cia-determinism
```

Use `collect` for a job that has already finished or `stop` to explicitly
release an abandoned VM. Colab source archives exclude notebooks and are
scanned for common GitHub-token and private-key formats before upload.

Colab CLI 0.6.0 currently declares an unpinned `jupyter-kernel-client`
dependency even though it uses an API from Google's fork. If `colab exec`
fails with `jupyter_kernel_client` missing `KernelClient`, repair the tool
environment once with:

```bash
uv pip install \
  --python ~/.local/share/uv/tools/google-colab-cli/bin/python \
  --reinstall \
  "jupyter-kernel-client @ git+https://github.com/googlecolab/jupyter-kernel-client.git"
```
