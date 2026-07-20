"""Run the original Flower 1.16 strategy on deterministic synthetic updates."""

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg


def load_original_strategy(source: Path):
    """Load the authors' strategy directly from their source file."""
    spec = importlib.util.spec_from_file_location("authors_metricdp", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MetricDifferentialPrivacyServerSideFixedClipping


def arrays(values):
    """Create independent float64 model arrays."""
    return [np.asarray(value, dtype=np.float64) for value in values]


def main() -> None:
    """Aggregate one controlled legacy round and emit JSON."""
    strategy_class = load_original_strategy(Path(sys.argv[1]).resolve())
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260721
    global_arrays = arrays([[[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0, 0.0]])
    client_models = [
        arrays([[[3.0, 4.0], [0.0, 0.0]], [1.0, 2.0, 0.0]]),
        arrays([[[0.0, 0.0], [6.0, 8.0]], [2.0, 0.0, 2.0]]),
        arrays([[[-3.0, 0.0], [0.0, 4.0]], [0.0, -1.0, 3.0]]),
    ]

    strategy = strategy_class(
        strategy=FedAvg(),
        noise_multiplier=0.7,
        clipping_norm=2.5,
        num_sampled_clients=3,
    )
    strategy.current_round_params = global_arrays
    results = [
        (
            None,
            FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters(model),
                num_examples=1,
                metrics={},
            ),
        )
        for model in client_models
    ]

    # The original implementation uses NumPy's global RNG.
    np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        distance = strategy.distance_metric(results)
        aggregated, _ = strategy.aggregate_fit(1, results, [])
    if aggregated is None:
        raise RuntimeError("Legacy strategy unexpectedly skipped aggregation")

    output = {
        "distance": distance,
        "stdv": (0.7 / distance) * 2.5 / 3,
        "arrays": [array.tolist() for array in parameters_to_ndarrays(aggregated)],
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
