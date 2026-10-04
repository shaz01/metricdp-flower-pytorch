# Runs

Question: is the CIFAR-10 gap caused by the dataset or by the model?

- Same settings everywhere: FedAvg, 48 clients, lr 0.001, batch 32, 5 local epochs, clip 5.
- All images 32×32. EuroSAT gets downsized (bilinear).
- No training-time augmentation for either dataset (CIFAR-10 never had it; EuroSAT's crop+flip is off).

## Stage A: dataset or model?
CIFAR-10 and EuroSAT
eurosat_cnn and cifar10_cnn
Homogeneous
10 rounds
Seed 42 (add 43, 44 only if two cells come out within ~30%)
Metric only, noise ratio 0.0001 (near-vanilla: no noise feedback, but distances still logged)
Accuracy only
The noise at ratio 0.0025 is computed from the measured distances, not run.

If CIFAR-10 is far apart with both models, it's the dataset. Go to Stage B.
If it changes with the model, it's the model. Keep one model, decide which datasets Stage B uses,
then go to Stage B. We still need the homogeneous vs non-IID data.

# Runs held off
After Stage A.

## Stage B: accuracy gap
CIFAR-10 and EuroSAT
One model for both
Homogeneous and Dirichlet α = 0.3
3 seeds
Vanilla, Global, Metric
Accuracy only

Shows how the dataset changes the gap, and how label skew changes it.

## Stage C: attack
Same runs as Stage B, with the attack (IN/OUT).

Shows how label skew changes the attack score. First time we compare homogeneous and real
non-IID at the same settings.

# Code
- `runner.py` plans Stage A (`--execute` trains; `--cells` shards across machines).
- `analyze.py` prints per-cell update norm, pairwise distance, metric-privacy noise, accuracy.
- EuroSAT data module gained `resize_to` and `augment` options (defaults unchanged).

Launch (one cell per account):
```
uv run python scripts/colab/run_experiment.py run --account lab2 --session dvm-a-<cell> --gpu A100 \
  --module results.cia_frontier.dataset_vs_model.runner \
  --results results/cia_frontier/dataset_vs_model/results/stage_a \
  --commit-message "results(dataset_vs_model): stage A <cell>" \
  -- --execute --cells <cell> --output-dir results/cia_frontier/dataset_vs_model/results/stage_a
```
