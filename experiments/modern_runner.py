"""Run the Flower 1.32 port on deterministic synthetic updates."""

import json
import sys

import numpy as np
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from metricdp_pytorch.metricdp_strategy import MetricPrivacyServerSideFixedClipping


def record(values) -> ArrayRecord:
    """Create a named float64 model record."""
    return ArrayRecord(
        {
            f"layer-{index}": Array(np.asarray(value, dtype=np.float64))
            for index, value in enumerate(values)
        }
    )


def main() -> None:
    """Aggregate one controlled modern round and emit JSON."""
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260721
    global_record = record([[[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0, 0.0]])
    client_models = [
        record([[[3.0, 4.0], [0.0, 0.0]], [1.0, 2.0, 0.0]]),
        record([[[0.0, 0.0], [6.0, 8.0]], [2.0, 0.0, 2.0]]),
        record([[[-3.0, 0.0], [0.0, 4.0]], [0.0, -1.0, 3.0]]),
    ]

    strategy = MetricPrivacyServerSideFixedClipping(
        strategy=FedAvg(),
        noise_multiplier=0.7,
        clipping_norm=2.5,
        num_sampled_clients=3,
    )
    strategy.current_arrays = global_record

    replies = []
    for node_id, model in enumerate(client_models, start=1):
        request = Message(
            content=RecordDict(), message_type="train", dst_node_id=node_id
        )
        replies.append(
            Message(
                content=RecordDict(
                    {
                        "arrays": model,
                        "metrics": MetricRecord({"num-examples": 1}),
                    }
                ),
                reply_to=request,
            )
        )

    np.random.seed(seed)
    aggregated, metrics = strategy.aggregate_train(1, replies)
    if aggregated is None or metrics is None:
        raise RuntimeError("Modern strategy unexpectedly skipped aggregation")

    output = {
        "distance": metrics["metric-dp-distance"],
        "stdv": metrics["metric-dp-noise-stdv"],
        "arrays": [array.tolist() for array in aggregated.to_numpy_ndarrays()],
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
