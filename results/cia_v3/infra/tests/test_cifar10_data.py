"""Label-Dirichlet CIFAR-10 (v3 P0 item 3): the partition is deterministic in its seed."""

from __future__ import annotations

import numpy as np
import pytest

from results.cia_frontier.eurosat_frontier.data import dirichlet_partitions
from results.cia_v3.infra import cifar10_data


def synthetic_labels():
    return tuple(np.repeat(np.arange(10), 500).tolist())


def test_partition_is_deterministic_in_its_seed():
    labels = synthetic_labels()
    first = dirichlet_partitions(labels, 48, 0.3, 42, cifar10_data.MIN_RECORDS)
    dirichlet_partitions.cache_clear()
    assert dirichlet_partitions(labels, 48, 0.3, 42, cifar10_data.MIN_RECORDS) == first
    assert dirichlet_partitions(labels, 48, 0.3, 43, cifar10_data.MIN_RECORDS) != first
    assert sorted(i for part in first for i in part) == list(range(len(labels)))
    assert min(map(len, first)) >= cifar10_data.MIN_RECORDS


def test_summary_counts_every_client_and_class():
    labels = synthetic_labels()
    parts = dirichlet_partitions(labels, 48, 0.3, 42, cifar10_data.MIN_RECORDS)
    summary = cifar10_data.summarize(labels, parts, alpha=0.3, seed=42)
    assert summary["classes"] == list(range(10)) and len(summary["clients"]) == 48
    assert sum(sum(c["class_counts"]) for c in summary["clients"]) == len(labels)
    assert all(sum(c["class_counts"]) == c["records"] for c in summary["clients"])


def test_module_rejects_other_partition_modes():
    module = cifar10_data.DirichletCIFAR10(0.3, partition_seed=42)
    with pytest.raises(ValueError):
        module.client_loaders(0, num_partitions=48, partition_mode="homogeneous", batch_size=32, seed=1)


def test_committed_summary_matches_the_dataset():
    import json
    from pathlib import Path

    try:
        summary = cifar10_data.partition_summary(0.3, 48, 42)
    except Exception:
        pytest.skip("CIFAR-10 is not available locally")
    committed = Path(__file__).resolve().parents[1] / "tables" / "cifar10_partition_summary.json"
    assert json.loads(committed.read_text()) == summary
