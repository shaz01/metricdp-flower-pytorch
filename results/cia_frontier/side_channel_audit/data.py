"""Label-Dirichlet Alzheimer MRI with the EuroSAT frontier's partition rule (WP1).

Same construction as ``results.cia_frontier.eurosat_frontier.data``: per-class
Dirichlet(alpha) proportions, multinomial counts, deterministic redraw until every
client holds >= ``MIN_RECORDS`` records; removal views over the canonical federation, so IN and
OUT share identical partitions. The package interface (``create_data_module``,
``base_module``, ``partition_summary``) is what the frontier runner expects.
"""
from __future__ import annotations

import json

import numpy as np

from experiments.cia.datasets.partitions import in_remove, out_remove
from experiments.reproduce.dataset.alzheimer import AlzheimerDataModule, AlzheimerMRIDataset
from metricdp_pytorch.utils.data import labels_from_records, make_client_loaders
from results.cia_frontier.eurosat_frontier.data import dirichlet_partitions

# Redraw until every client holds >= MIN_RECORDS records (the frontier's rule uses 10). Fixed
# before any run: the target shadow sample is about 8% of a client's records (10% of its 80%
# training split), so 125 records give a shadow sample of at least 10 images; with 10 records
# some Alzheimer clients got an empty or 3-6 image shadow sample.
MIN_RECORDS = 125


class DirichletAlzheimer(AlzheimerDataModule):
    """Label-Dirichlet Alzheimer MRI; ``partition_seed`` as in ``DirichletEuroSAT``."""

    def __init__(self, alpha, partition_seed=None, min_records=MIN_RECORDS, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.partition_seed = partition_seed
        self.min_records = min_records

    def _layout_seed(self, seed):
        return seed if self.partition_seed is None else self.partition_seed

    def client_loaders(self, partition_id, *, num_partitions, partition_mode,
                       batch_size, seed, partition_profile="auto", client_weights=None,
                       dirichlet_alpha=None, max_samples=0):
        del dirichlet_alpha  # alpha comes from the partition profile (self.alpha)
        if partition_mode != "non-iid" or client_weights is not None:
            raise ValueError("This experiment requires label-Dirichlet non-IID without weights")
        layout = self._layout_seed(seed)
        split = self.dataset["train"]
        labels = labels_from_records(split)
        parts = dirichlet_partitions(tuple(labels), num_partitions, self.alpha, layout, self.min_records)
        return make_client_loaders(
            AlzheimerMRIDataset(split), labels, parts[partition_id], batch_size=batch_size,
            seed=layout + partition_id, train_fraction=self.train_fraction, max_samples=max_samples,
        )

    def server_loaders(self, *, batch_size, seed, max_samples=0):
        return super().server_loaders(
            batch_size=batch_size, seed=self._layout_seed(seed), max_samples=max_samples,
        )


def base_module(alpha, partition_seed=None, min_records=MIN_RECORDS):
    return DirichletAlzheimer(alpha, partition_seed=partition_seed, min_records=min_records)


def partition_summary(alpha, clients, seed, min_records=MIN_RECORDS):
    labels = tuple(labels_from_records(DirichletAlzheimer(alpha).dataset["train"]))
    parts = dirichlet_partitions(labels, clients, alpha, seed, min_records)
    y = np.asarray(labels)
    classes = np.unique(y)
    return {"alpha": alpha, "seed": seed, "classes": classes.tolist(),
            "min_records": min(map(len, parts)), "max_records": max(map(len, parts)),
            "clients": [{"target": t, "records": len(part),
                         "class_counts": [int(np.sum(y[list(part)] == label)) for label in classes]}
                        for t, part in enumerate(parts)]}


def create_data_module(config):
    settings = json.loads(config["partition-profile"])
    module = DirichletAlzheimer(settings["alpha"], partition_seed=settings.get("partition_seed"),
                                cache_dir=config.get("data-cache-dir") or None)
    target = settings["out_target"]
    factory = in_remove if target is None else out_remove
    return factory(module, canonical_num_partitions=settings["clients"],
                   target_partition_id=0 if target is None else target)
