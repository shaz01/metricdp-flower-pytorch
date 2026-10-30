"""Training data for one randomized-inclusion run: the frontier label-Dirichlet EuroSAT (or Alzheimer MRI, owner
decision 8; or CIFAR-10, roadmap v3 decision 2), restricted to the active clients.

The partition profile names the canonical federation (``alpha``, ``clients``, ``partition_seed``,
the layout seed) and the canonical IDs of the clients that take part (``active``). Every client
keeps its canonical data, so a candidate holds the same records in every run.
"""

from __future__ import annotations

import json

from experiments.cia.datasets.partitions import PartitionViewDataModule
from results.cia_frontier.eurosat_frontier.data import DirichletEuroSAT, partition_summary
from results.cia_frontier.side_channel_audit.data import DirichletAlzheimer

__all__ = ["base_module", "create_data_module", "partition_summary"]

ALZHEIMER_MIN_RECORDS = 10  # the EuroSAT frontier's rule; WP1's 125 is infeasible at 48 clients


def base_module(alpha, partition_seed=None, dataset="eurosat", cache_dir=None):
    """Canonical (all-client) module of ``dataset``."""
    if dataset == "alzheimer":
        return DirichletAlzheimer(alpha, partition_seed=partition_seed, min_records=ALZHEIMER_MIN_RECORDS,
                                  cache_dir=cache_dir)
    if dataset == "cifar10":  # roadmap v3 decision 2
        from results.cia_v3.infra.cifar10_data import base_module as cifar10

        return cifar10(alpha, partition_seed=partition_seed, cache_dir=cache_dir)
    if dataset != "eurosat":
        raise ValueError(f"unknown dataset {dataset!r}")
    return DirichletEuroSAT(alpha, partition_seed=partition_seed, cache_dir=cache_dir)


def create_data_module(config):
    settings = json.loads(config["partition-profile"])
    module = base_module(settings["alpha"], partition_seed=settings["partition_seed"],
                         dataset=settings.get("dataset", "eurosat"), cache_dir=config.get("data-cache-dir") or None)
    return PartitionViewDataModule(module, canonical_num_partitions=settings["clients"],
                                   active_partition_ids=settings["active"])
