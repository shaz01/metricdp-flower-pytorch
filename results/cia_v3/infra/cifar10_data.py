"""Label-Dirichlet CIFAR-10 for the randomized-inclusion runner (CIA roadmap v3, P0 item 3; decision 2).

Same construction as ``results.cia_frontier.side_channel_audit.data`` (Alzheimer): per-class Dirichlet(alpha)
proportions over the 50,000 ten-class training images, multinomial counts, deterministic redraw until every client
holds at least ``MIN_RECORDS`` records (the EuroSAT frontier's rule). ``partition_seed`` fixes the layout (partitions,
client train/test splits, server evaluation split); the per-round client seed reseeds batch order in the client app.
"""
from __future__ import annotations

import numpy as np

from experiments.reproduce.dataset.cifar10 import Cifar10DataModule, Cifar10Dataset
from metricdp_pytorch.utils.data import labels_from_records, make_client_loaders
from results.cia_frontier.eurosat_frontier.data import dirichlet_partitions

MIN_RECORDS = 10


class DirichletCIFAR10(Cifar10DataModule):
    """Label-Dirichlet ten-class CIFAR-10; ``partition_seed`` as in ``DirichletEuroSAT``."""

    def __init__(self, alpha, partition_seed=None, min_records=MIN_RECORDS, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.partition_seed = partition_seed
        self.min_records = min_records

    def _layout_seed(self, seed):
        return seed if self.partition_seed is None else self.partition_seed

    def partitions(self, num_partitions, seed):
        labels = labels_from_records(self.dataset["train"])
        return dirichlet_partitions(tuple(labels), num_partitions, self.alpha, self._layout_seed(seed),
                                    self.min_records)

    def client_loaders(self, partition_id, *, num_partitions, partition_mode,
                       batch_size, seed, partition_profile="auto", client_weights=None,
                       dirichlet_alpha=None, max_samples=0):
        del dirichlet_alpha  # alpha comes from the partition profile (self.alpha)
        if partition_mode != "non-iid" or client_weights is not None:
            raise ValueError("This experiment requires label-Dirichlet non-IID without weights")
        layout = self._layout_seed(seed)
        split = self.dataset["train"]
        return make_client_loaders(
            Cifar10Dataset(split), labels_from_records(split), self.partitions(num_partitions, seed)[partition_id],
            batch_size=batch_size, seed=layout + partition_id, train_fraction=self.train_fraction,
            max_samples=max_samples,
        )

    def server_loaders(self, *, batch_size, seed, max_samples=0):
        return super().server_loaders(batch_size=batch_size, seed=self._layout_seed(seed), max_samples=max_samples)


def base_module(alpha, partition_seed=None, min_records=MIN_RECORDS, cache_dir=None):
    return DirichletCIFAR10(alpha, partition_seed=partition_seed, min_records=min_records, cache_dir=cache_dir)


def summarize(labels, parts, *, alpha, seed):
    """The committed partition summary (same keys as EuroSAT's and Alzheimer's)."""
    y = np.asarray(labels)
    classes = np.unique(y)
    return {"alpha": alpha, "seed": seed, "classes": classes.tolist(),
            "min_records": min(map(len, parts)), "max_records": max(map(len, parts)),
            "clients": [{"target": t, "records": len(part),
                         "class_counts": [int(np.sum(y[list(part)] == label)) for label in classes]}
                        for t, part in enumerate(parts)]}


def partition_summary(alpha, clients, seed, min_records=MIN_RECORDS):
    labels = tuple(labels_from_records(DirichletCIFAR10(alpha).dataset["train"]))
    return summarize(labels, dirichlet_partitions(labels, clients, alpha, seed, min_records), alpha=alpha, seed=seed)
