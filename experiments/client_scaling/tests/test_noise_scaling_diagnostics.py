"""Tests for the noise-ratio scaling diagnostics."""

from __future__ import annotations

import json

import pytest

from experiments.client_scaling.scripts import noise_scaling_diagnostics as diagnostics


def _run_document(
    *,
    privacy: str,
    num_clients: int,
    noise_multiplier: float,
    distance: float | None,
    clipping_norm: float = 5.0,
) -> dict:
    stdv = noise_multiplier * clipping_norm / num_clients
    train_metrics = {}
    for round_number in range(1, 4):
        metrics: dict[str, float] = {"dp-noise-to-signal-ratio": 1.0}
        if distance is None:
            metrics["global-dp-noise-stdv"] = stdv
        else:
            metrics["metric-dp-distance"] = distance
            metrics["metric-dp-noise-stdv"] = stdv / distance
        train_metrics[str(round_number)] = metrics
    return {
        "metadata": {
            "privacy": privacy,
            "partition_mode": "non-iid",
            "num_clients": num_clients,
            "noise_multiplier": noise_multiplier,
            "clipping_norm": clipping_norm,
        },
        "train_metrics": train_metrics,
    }


def _write(tmp_path, name: str, document: dict):
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_global_dp_stdv_is_constant_across_client_counts_at_one_ratio(tmp_path) -> None:
    """The premise of the ratio system: nm = ratio * n pins global-DP's noise."""
    ratio = 0.00625
    stdvs = set()
    for clients in (8, 48, 100):
        path = _write(
            tmp_path,
            f"g{clients}.json",
            _run_document(
                privacy="global-dp",
                num_clients=clients,
                noise_multiplier=ratio * clients,
                distance=None,
            ),
        )
        entry = diagnostics.summarize_run(path)
        assert entry["ratio"] == pytest.approx(ratio)
        stdvs.add(round(entry["global_dp_stdv_at_this_ratio"], 12))
    assert len(stdvs) == 1


def test_metric_privacy_effective_ratio_divides_by_the_distance(tmp_path) -> None:
    ratio, clients, distance = 0.00625, 48, 0.871
    path = _write(
        tmp_path,
        "m.json",
        _run_document(
            privacy="metric-privacy",
            num_clients=clients,
            noise_multiplier=ratio * clients,
            distance=distance,
        ),
    )

    entry = diagnostics.summarize_run(path)

    assert entry["mean_max_pairwise_distance"] == pytest.approx(distance)
    assert entry["effective_ratio"] == pytest.approx(ratio / distance)


def test_distance_below_one_makes_metric_privacy_noisier_than_global_dp(tmp_path) -> None:
    """d < 1 inverts the mechanism: it injects more noise, not less."""
    path = _write(
        tmp_path,
        "m.json",
        _run_document(
            privacy="metric-privacy",
            num_clients=48,
            noise_multiplier=0.3,
            distance=0.871,
        ),
    )

    entry = diagnostics.summarize_run(path)

    assert entry["observed_noise_stdv"] > entry["global_dp_stdv_at_this_ratio"]
    assert entry["effective_ratio"] > entry["ratio"]


def test_vanilla_runs_report_no_ratio(tmp_path) -> None:
    """Vanilla's recorded multiplier is a naming artifact; it injects nothing."""
    path = _write(
        tmp_path,
        "v.json",
        _run_document(
            privacy="vanilla", num_clients=8, noise_multiplier=0.01, distance=None
        ),
    )

    entry = diagnostics.summarize_run(path)

    assert entry["ratio"] is None
    assert entry["effective_ratio"] is None


def test_sidecar_artifacts_are_not_mistaken_for_run_jsons(tmp_path) -> None:
    document = _run_document(
        privacy="global-dp", num_clients=8, noise_multiplier=0.05, distance=None
    )
    _write(tmp_path, "run.json", document)
    _write(tmp_path, "cia.json", [{"run_name": "x"}])
    _write(tmp_path, "colab_run.json", {"state": "complete"})
    _write(tmp_path, "run.evaluation.json", document)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    found = diagnostics.find_run_jsons(tmp_path)

    assert [path.name for path in found] == ["run.json"]


def test_aggregate_by_config_averages_repeats(tmp_path) -> None:
    """IN/OUT views and seeds of one config collapse into a single row."""
    entries = [
        diagnostics.summarize_run(
            _write(
                tmp_path,
                f"m{index}.json",
                _run_document(
                    privacy="metric-privacy",
                    num_clients=8,
                    noise_multiplier=0.05,
                    distance=distance,
                ),
            )
        )
        for index, distance in enumerate((1.4, 1.6))
    ]

    aggregated = diagnostics.aggregate_by_config(entries)

    assert len(aggregated) == 1
    assert aggregated[0]["runs"] == 2
    assert aggregated[0]["mean_max_pairwise_distance"] == pytest.approx(1.5)
