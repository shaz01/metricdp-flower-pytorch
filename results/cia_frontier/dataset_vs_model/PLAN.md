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

## Stage A result
It's the combination: CIFAR-10 clients drift apart only with cifar10_cnn (0.76 vs 0.41–0.42
elsewhere). Model branch. See README.

## Stage B: accuracy and IN measurements
CIFAR-10 and EuroSAT (32×32), each with 21,600 training images. CIFAR-10 uses the same fixed
stratified subset before partitioning. At 48 clients, that is 450 images per client on average.
Model: cifar10_cnn. Use homogeneous and Dirichlet α = 0.3 partitions, 20 rounds, and seed 42.
Every run is an IN run for targets 0–9. It evaluates clean and noisy shadow losses each round.
The clean shadow is a deterministic stratified 10% subset of the target's training records. The
IN model trains on the full target training split, including those shadow records.

Vanilla, Global, Metric
Noise ratios 0.001, 0.0025, 0.004, 0.00625 (contest_at_scale/cifar10's three, plus 0.001 below:
at 450 images/client the same ratio is heavier, and metric-privacy adds 1.3–2.9× global-DP's
noise here, so each curve needs a point before collapse)
The plan is 2 × 2 × (1 + 2 × 4) = 36 IN runs. Build OUT runs later with `--out-targets`;
they exclude one target and use its identical shadow records. Compare mechanisms at equal
accuracy, not equal ratio.

## Stage C: attack
Reuse the Stage B IN losses and add OUT runs later. Score each target by the fraction of matched
rounds where its IN clean-shadow loss is lower than its OUT loss, then average targets.

Per-client influence diagnostics are saved in each training round for all privacy modes.

# Code
- `runner.py` plans Stage A (`--execute` trains; `--cells` shards across machines).
- `analyze.py` prints per-cell update norm, pairwise distance, metric-privacy noise, accuracy.
- `stage_b.py` / `analyze_stage_b.py`: same for Stage B (cells = dataset+partition; `--cells`,
  `--privacy`, `--ratios`, `--seeds` shard).
- EuroSAT data module gained `resize_to` and `augment` options (defaults unchanged).
- CIFAR-10 data module gained `train_subsample` / `subsample_seed` (default off).

Launch (one cell per account):
```
uv run python scripts/colab/run_experiment.py run --account lab2 --session dvm-a-<cell> --gpu A100 \
  --module results.cia_frontier.dataset_vs_model.runner \
  --results results/cia_frontier/dataset_vs_model/results/stage_a/<cell> \
  --commit-message "results(dataset_vs_model): stage A <cell>" \
  -- --execute --cells <cell> --output-dir results/cia_frontier/dataset_vs_model/results/stage_a/<cell>
```
