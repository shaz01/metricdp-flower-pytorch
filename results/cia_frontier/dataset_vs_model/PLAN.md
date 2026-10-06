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
Every run is an IN run: all 48 clients train. After each round it scores targets 0–9 on their
shadow sets (clean and noisy loss), like eurosat_frontier.
A target's shadow set is a fixed 10% of its own training records. So the IN model trains on the
shadow records. The noisy copy adds 20% noise to the same records.
One seed sets both the data split and training, as in eurosat_frontier.

Vanilla, Global, Metric
Noise ratios 0.001, 0.0025, 0.004, 0.00625 (contest_at_scale/cifar10's three, plus 0.001 below:
at 450 images/client the same ratio is heavier, and metric-privacy adds 1.3–2.9× global-DP's
noise here, so each curve needs a point before collapse)
That is 2 × 2 × (1 + 2 × 4) = 36 IN runs. OUT runs come later (`--out-targets`).
An OUT run drops one target (47 clients) and scores the same shadow records.
`shadows.json` stores a hash of them, and the scoring script checks IN and OUT match.
Compare mechanisms at equal accuracy, not equal ratio.

## Plateau check (sets R)
Two vanilla Dirichlet runs, 70 rounds, seed 42, one per dataset (`results/plateau/`).
The rule was fixed before the runs. Plateau = mean accuracy over rounds 60-70.
R_d = first round whose 3-round moving average is within 2 points of the plateau.
R = max R_d, rounded up to a multiple of 5, capped at 50.
Result: CIFAR-10s plateau 0.583, EuroSAT32 plateau 0.773, both R_d = 51. So R = 50 (the cap).
Accuracy still rises slowly after round 50 (about 2 points by round 70).

## Stage C: attack
Dirichlet a=0.3 only. The old homogeneous Stage B runs are near-IID and serve as a rough
homogeneous reference.
Datasets cifar10s and eurosat32. Arms: vanilla, plus global-dp and metric-privacy at
ratios 0.0025, 0.004, 0.00625. That is 14 settings. Seed 42, R = 50 rounds.
Per setting: 1 IN run that scores targets 0-6 every round, and 7 OUT runs (one per target
removed). 14 IN + 98 OUT = 112 runs. Stage B IN runs are not reused: they ran 20 rounds.
We use 7 targets, down from 10, to keep the OUT cost at 7 runs per setting.
Score each target by the fraction of matched rounds where its IN clean-shadow loss is lower
than its OUT loss, then average targets.

Script: `python -m results.cia_frontier.per_client_score <root>`. It also works on
eurosat_frontier's folders. `frontier.py` draws one accuracy-vs-attack-score PNG per dataset.

Shards (one per setting) run two streams side by side on one VM with `parallel.py`:
```
... --module results.cia_frontier.dataset_vs_model.parallel -- --execute --seeds 42 \
  --output-dir <out> --stage c --cells <ds>+dirichlet --privacy <p> --ratios <r> --rounds 50 \
  --targets 0 1 2 3 4 5 6 ::: --out-targets 0 1 2 --with-in ::: --out-targets 3 4 5 6
```

## Influence log
Every round, every run (vanilla too) saves per-client lists in `train_metrics`, keyed by
`influence-client-ids` (canonical client IDs, so OUT runs skip the dropped target):
update norm before clipping, clipped or not, distance and cosine to the weighted average of
clipped updates, leave-one-out influence norm, aggregation weight, num examples.
It only reads the updates. A test checks aggregation stays bit-identical with it on.

# Code
- `runner.py` plans Stage A (`--execute` trains; `--cells` shards across machines).
- `analyze.py` prints per-cell update norm, pairwise distance, metric-privacy noise, accuracy.
- `stage_b.py`: Stage B IN runs and later OUT runs. Shard with `--cells`, `--privacy`,
  `--ratios`, `--seeds`. Each run gets its own folder with `measurements.json`.
- `plateau.py`: the plateau rule above; writes `results/plateau/plateau.json`.
- `parallel.py`: runs several `stage_b` streams at once on one VM.
- `frontier.py`: Stage C frontier PNGs and `frontier.json`.
- `analyze_stage_b.py`: accuracy and DP diagnostics from the IN runs.
- `experiments/cia/trajectories.py`: the train-then-score loop, shared with eurosat_frontier.
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
