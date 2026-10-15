"""Smoke run of the private-median strategy on the real Flower path (WP3; at most 5 rounds, capped data).

Builds the reproduction runner's configuration, replaces the server's strategy constructor with
``PrivateMedianMetricPrivacyServerSideFixedClipping`` (no edit to ``strategy_factory.py``, which Agent B owns),
and runs the simulation in-process. Output goes to the directory given on the command line (scratch, not results).

    CUDA_VISIBLE_DEVICES= uv run --no-sync python research/cia_review_2026-10/theory/smoke_private_median.py <out-dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROUNDS = 3
CLIENTS = 3


def main(out_dir: Path) -> None:
    import experiments.reproduce.server as server
    from experiments.reproduce.runner import _parser, _run_worker, build_run_config
    from metricdp_pytorch.private_metric_strategy import PrivateMedianMetricPrivacyServerSideFixedClipping
    from metricdp_pytorch.strategy_factory import make_base_strategy

    args = _parser().parse_args([
        "--seed", "42", "--noise-multiplier", "0.01", "--clipping-norm", "5", "--rounds", str(ROUNDS),
        "--local-epochs", "1", "--num-clients", str(CLIENTS), "--privacy", "metric-privacy",
        "--aggregation-weighting", "equal", "--model-module", "experiments.reproduce.paper_cnn:create_model",
        "--max-client-samples", "64", "--max-test-samples", "128", "--initialization-epochs", "1",
        "--output-dir", str(out_dir), "--run-name", "smoke-private-median",
    ])
    config = build_run_config(args)
    built = []

    def private_median_strategy(**kwargs):
        base = make_base_strategy(kwargs["aggregation"], num_clients=kwargs["num_clients"],
                                  fraction_evaluate=kwargs["fraction_evaluate"], weighting=kwargs["weighting"])
        strategy = PrivateMedianMetricPrivacyServerSideFixedClipping(
            base, noise_multiplier=kwargs["noise_multiplier"], clipping_norm=kwargs["clipping_norm"],
            num_slots=kwargs["num_clients"], initial_quantile=1.0, minimum_quantile=0.01, maximum_quantile=10.0,
            seed=42)
        built.append(strategy)
        return strategy

    server.create_paper_strategy = private_median_strategy
    _run_worker(config, num_supernodes=CLIENTS, max_parallel_clients=CLIENTS, client_cpus=1.0, client_gpus=0.0,
                verbose=False)
    result = json.loads((out_dir / "smoke-private-median.json").read_text())
    rows = []
    for round_number, metrics in sorted(result["train_metrics"].items(), key=lambda item: int(item[0])):
        rows.append({key: metrics[key] for key in metrics if key.startswith("private-median")})
        rows[-1]["round"] = int(round_number)
    summary = {"strategy_built": len(built), "rounds": rows,
               "final_accuracy": result["server_evaluate_metrics"][str(ROUNDS)]["accuracy"],
               "account": vars(built[0].account(ROUNDS)) | {"mu_total": built[0].account(ROUNDS).mu_total}}
    (out_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
