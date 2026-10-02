"""Experiment-local label-Dirichlet EuroSAT; existing partitions are unchanged."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from experiments.reproduce.dataset.eurosat import EurosatDataModule, EurosatDataset
from experiments.cia.datasets.partitions import in_remove, out_remove
from metricdp_pytorch.utils.data import cap_indices, labels_from_records, make_indexed_loader
from metricdp_pytorch.utils.split_data import split_stratified


@lru_cache(maxsize=128)
def dirichlet_partitions(labels: tuple[int, ...], clients: int, alpha: float, seed: int):
    """Allocate each class by Dirichlet proportions; reject partitions <10 records.

    Bounded rejection is explicit: very small alpha may be infeasible. No silent
    redistribution or fallback to the old quantity-skew partitioner.
    """
    if clients < 2 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Require clients >= 2 and finite alpha > 0")
    y = np.asarray(labels)
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        parts = [[] for _ in range(clients)]
        for label in np.unique(y):
            indices = np.flatnonzero(y == label)
            rng.shuffle(indices)
            counts = rng.multinomial(len(indices), rng.dirichlet(np.full(clients, alpha)))
            for part, chunk in zip(parts, np.split(indices, np.cumsum(counts)[:-1]), strict=True):
                part.extend(chunk.tolist())
        if min(map(len, parts)) >= 10:
            return tuple(tuple(p) for p in parts)
    raise ValueError("No Dirichlet allocation with >=10 records/client after 1000 attempts")


class DirichletEuroSAT(EurosatDataModule):
    """Label-Dirichlet EuroSAT.

    ``partition_seed`` fixes the data layout (client partitions, each client's
    train/test split, server evaluation split) independently of the training
    seed. The training seed passed per call then controls only training
    randomness (batch order here; model init and noise elsewhere). ``None``
    keeps the original behaviour, where one seed controls both.
    """

    def __init__(self, alpha, partition_seed=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.partition_seed = partition_seed

    def _layout_seed(self, seed):
        return seed if self.partition_seed is None else self.partition_seed

    def client_loaders(self, partition_id, *, num_partitions, partition_mode,
                       batch_size, seed, partition_profile="auto", client_weights=None,
                       max_samples=0):
        if partition_mode != "non-iid" or client_weights is not None:
            raise ValueError("This experiment requires label-Dirichlet non-IID without weights")
        layout = self._layout_seed(seed)
        split = self.dataset["train"]
        labels = labels_from_records(split, label_column="label")
        parts = dirichlet_partitions(tuple(labels), num_partitions, self.alpha, layout)
        selected = cap_indices(parts[partition_id], max_samples)
        train, test = split_stratified(labels, selected, self.train_fraction, seed=layout + partition_id)
        return tuple(make_indexed_loader(
            EurosatDataset(split, augment=augment), indices, batch_size=batch_size,
            shuffle=augment, seed=seed + partition_id,
        ) for indices, augment in ((train, True), (test, False)))

    def server_loaders(self, *, batch_size, seed, max_samples=0):
        return super().server_loaders(
            batch_size=batch_size, seed=self._layout_seed(seed), max_samples=max_samples,
        )


def partition_summary(alpha, clients, seed):
    module = DirichletEuroSAT(alpha)
    labels = tuple(labels_from_records(module.dataset["train"], label_column="label"))
    parts = dirichlet_partitions(labels, clients, alpha, seed)
    y = np.asarray(labels)
    classes = np.unique(y)
    return {"alpha": alpha, "seed": seed, "classes": classes.tolist(),
            "min_records": min(map(len, parts)), "max_records": max(map(len, parts)),
            "clients": [{"target": t, "records": len(part),
                         "class_counts": [int(np.sum(y[list(part)] == label)) for label in classes]}
                        for t, part in enumerate(parts)]}


def create_data_module(config):
    settings = json.loads(config["partition-profile"])
    module = DirichletEuroSAT(settings["alpha"], partition_seed=settings.get("partition_seed"),
                              cache_dir=config.get("data-cache-dir") or None)
    target = settings["out_target"]
    factory = in_remove if target is None else out_remove
    return factory(module, canonical_num_partitions=settings["clients"],
                   target_partition_id=0 if target is None else target)
