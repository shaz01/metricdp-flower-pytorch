"""Training data for one randomized-inclusion run: the frontier label-Dirichlet EuroSAT, restricted to the active clients.

The partition profile names the canonical federation (``alpha``, ``clients``, ``partition_seed``,
the layout seed) and the canonical IDs of the clients that take part (``active``). Every client
keeps its canonical data, so a candidate holds the same records in every run.
"""

from __future__ import annotations

import json

from experiments.cia.datasets.partitions import PartitionViewDataModule
from results.cia_frontier.eurosat_frontier.data import DirichletEuroSAT, base_module, partition_summary

__all__ = ["base_module", "create_data_module", "partition_summary"]


def create_data_module(config):
    settings = json.loads(config["partition-profile"])
    module = DirichletEuroSAT(settings["alpha"], partition_seed=settings["partition_seed"],
                              cache_dir=config.get("data-cache-dir") or None)
    return PartitionViewDataModule(module, canonical_num_partitions=settings["clients"],
                                   active_partition_ids=settings["active"])
