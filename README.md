# Metric-aware DP with Flower and PyTorch

A Flower 1.32 app that ports the paper's server-side metric-aware noise calibration to Flower's modern, message-based `ServerApp` strategy API.

The app trains a small PyTorch CNN on federated CIFAR-10 partitions. Each round, the server:

1. Computes the maximum pairwise client-model distance, defined as the maximum over client pairs of the mean layer-wise Euclidean/Frobenius distance.
2. Clips each client update with Flower's server-side fixed-clipping mechanism.
3. Aggregates the clipped updates with `FedAvg`.
4. Adds Gaussian noise using `noise_multiplier / distance` for that round.

> **Privacy note:** As stated in the source paper, using model distance to calibrate noise is empirical and does not by itself provide a formal metric-DP guarantee. Formal privacy claims require an appropriate accountant and threat-model analysis.

## Structure

```text
metricdp-pytorch/
├── metricdp_pytorch/
│   ├── client_app.py
│   ├── metricdp_strategy.py
│   ├── server_app.py
│   └── task.py
├── tests/
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.11–3.13
- Flower 1.32.1 or newer within the 1.x series
- PyTorch 2.10

The Flower CLI is already installed at `~/.local/bin/flwr` on this machine.

## Install

Using `uv`:

```bash
cd metricdp-pytorch
uv sync
```

Or with pip in a virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Test

```bash
uv run pytest
```

## Compare with the original implementation

A controlled synthetic experiment runs a byte-identical snapshot of the authors' Flower 1.16 source and this Flower 1.32 port with identical client models, clipping/noise settings, and five NumPy seeds. It does not depend on the separate Flower fork:

```bash
uv run pytest -m reproducibility experiments/port_equivalence/test_equivalence.py
```

The equivalent human-readable report remains available through `uv run python experiments/port_equivalence/compare.py`. Reproducibility tests are marked and excluded from ordinary `uv run pytest` runs because they create an isolated legacy environment.

The legacy worker is isolated with Flower 1.16.0, NumPy 1.26.4, and Python 3.12. The comparison checks distance, noise standard deviation, and every aggregated model value to an absolute tolerance of `1e-15`. See [`experiments/port_equivalence/README.md`](experiments/port_equivalence/README.md) for the concise protocol, results, limitations, and source fingerprints.

## Run

The local simulation is explicitly configured for ten SuperNodes, matching `num-clients`:

```bash
uv run flwr run . --stream
```

For a shorter run:

```bash
uv run flwr run . --run-config "num-server-rounds=1" --stream
```

Relevant configuration in `pyproject.toml`:

```toml
[tool.flwr.app.config]
num-clients = 10
noise-multiplier = 1.0
clipping-norm = 5.0
```

`noise-multiplier` is the base multiplier. The effective multiplier in each round is `noise-multiplier / metric-dp-distance`. The distance and resulting standard deviation are included in the aggregated training metrics.

The strategy raises an error when fewer than two models are received or when the distance is zero/non-finite, because distance-based calibration is undefined in those cases.
