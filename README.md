# Metric-aware DP with Flower and PyTorch

A Flower 1.32/PyTorch reproduction of the paper's server-side metric-aware noise calibration on the Alzheimer MRI dataset.

Each metric-private round:

1. Computes the maximum pairwise client-model distance (maximum over client pairs of the mean layer-wise Euclidean/Frobenius distance).
2. Clips every client update with Flower's server-side fixed-clipping mechanism.
3. Aggregates the updates.
4. Adds Gaussian noise using `noise_multiplier / distance` for that round.

> **Privacy note:** As stated in the paper, distance-calibrated noise is empirical and does not by itself provide a formal metric-DP guarantee. Formal privacy claims require an appropriate accountant and threat-model analysis.

## Dataset

The default data module uses `Falah/Alzheimer_MRI`, the paper CNN, and the published four-client distributions. Every local client partition uses a deterministic stratified 80/20 train/test split. Scalable balanced and quantity-skewed partitions are available for custom client counts.

## Install and test

The package supports Python 3.11–3.13.

```bash
uv sync
uv run pytest
```

## Alzheimer MRI paper reproduction

The reproduction runner connects the paper-specific ServerApp and ClientApp and works both locally and on a Lightning.ai pod. Start with the tested smoke run:

```bash
uv run python -m experiments.reproduce.runner --smoke
```

Run one full paper configuration with:

```bash
uv run python -m experiments.reproduce.runner \
  --partition homogeneous \
  --privacy metric-privacy \
  --aggregation fedavg \
  --rounds 20 --local-epochs 5
```

Use `--dry-run` to inspect the resolved configuration. The four-client `auto` profile reproduces the exact published client tables; other client counts automatically use scalable partitions. Results are written under `results/reproduce/` by default.

Every run evaluates the final global model directly from memory. It writes `<run-name>.evaluation.json` and `<run-name>.predictions.npz` without creating a checkpoint. The artifacts contain raw labels/probabilities/predictions, confusion matrices, per-class and macro/micro/weighted precision, recall and F1, and one-vs-rest ROC/AUC for the server final-test split and every client's held-out split.

Run the default 36-configuration partition × privacy × aggregation matrix with:

```bash
uv run python -m experiments.reproduce.matrix_runner \
  --output-dir results-reproduce-paper/matrix \
  --client-gpus 0.25 \
  --max-parallel-clients 4
```

The matrix runner validates artifacts, skips completed configurations, retries failures once, and writes `matrix-manifest.json`. Its default is sequential experiment execution; use `--parallel-experiments` only when the machine has enough independent resources for multiple Flower simulations.

A previously saved checkpoint can still be postprocessed with `python -m experiments.reproduce.detailed_evaluation`; pass `--help` for its artifact paths and optional deletion flag.

The data layer is pluggable. Supply a factory implementing `FederatedDataModule` with:

```bash
uv run python -m experiments.reproduce.runner \
  --data-module my_package.my_dataset:create_data_module
```

The factory receives the run configuration and returns an object with `client_loaders(...)` and `server_loaders(...)`. Generic record-image adapters, indexed loaders, stratified splits, exact class-count profiles, balanced partitions, and quantity-skewed partitions are available under `metricdp_pytorch.utils`.

## Aggregator mapping

The strategy factory uses the paper's listed settings:

- FedAvg
- FedAvgM: server learning rate 0.1, momentum 0.5
- FedMedian
- FedProx: proximal μ = 0.5 (including the client-side proximal loss)
- FedOpt: Flower FedAdam with β₁ = 0, β₂ = 0, τ = 1e-9
- FedYogi: β₁ = 0.9, β₂ = 0.99, τ = 1e-3

All can be used directly, wrapped in Flower's global server-side fixed-clipping DP, or wrapped in the metric-aware variant.

## Manual Flower run

The app remains directly runnable with its `pyproject.toml` defaults:

```bash
uv run flwr run . --stream
```

The defaults use four clients, 20 rounds, noise multiplier `0.01`, and clipping norm `5.0`.

## Port-equivalence test

A separate reproducibility test compares a byte-identical snapshot of the authors' Flower 1.16 mechanism with this Flower 1.32 strategy port under controlled synthetic inputs:

```bash
uv run pytest -m reproducibility experiments/port_equivalence/test_equivalence.py
```

See [`experiments/port_equivalence/README.md`](experiments/port_equivalence/README.md) for protocol details and limitations.
