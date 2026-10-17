"""Public data layouts for the frozen-head federation, identical to the EuroSAT frontier module's.

Client ``t`` of an ``n``-client layout holds the Dirichlet partition ``t`` of
``results/cia_frontier/eurosat_frontier/data.py`` (``dirichlet_partitions``, alpha 0.3 by default),
split 80/20 into local train/test with ``split_stratified(seed=layout_seed + t)``, exactly as
``DirichletEuroSAT.client_loaders`` does. The server halves the EuroSAT test split with
``split_stratified(seed=layout_seed)`` as ``EurosatDataModule.server_loaders`` does: the first half
is the public set (the frontier's validation loader), the second the server test set.

Larger federations (``split_factor > 1``) split every client of the ``n / split_factor``-client
layout uniformly at random into ``split_factor`` pseudo-clients (``default_rng([layout_seed, t])``),
so the data and the label skew stay those of the base layout. Pseudo-client ``t * split_factor + k``
is then split 80/20 with ``seed=layout_seed + t * split_factor + k``. (A direct 480-client Dirichlet
layout at alpha 0.3 is infeasible under the partitioner's 10-record minimum.)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from metricdp_pytorch.utils.split_data import split_stratified
from results.cia_frontier.eurosat_frontier.data import dirichlet_partitions

DEFAULT_ALPHA = 0.3
CLIENT_TRAIN_FRACTION = 0.8
SERVER_PUBLIC_FRACTION = 0.5


@dataclass(frozen=True)
class Layout:
    """Index sets into the EuroSAT ``train`` (clients) and ``test`` (server) splits."""

    num_clients: int
    alpha: float
    layout_seed: int
    client_train: tuple[np.ndarray, ...]
    client_test: tuple[np.ndarray, ...]
    public: np.ndarray
    server_test: np.ndarray

    @property
    def client_sizes(self) -> np.ndarray:
        """Records per client (train plus local test), the size used to rank targets."""
        return np.array([len(a) + len(b) for a, b in zip(self.client_train, self.client_test, strict=True)])

    def top_decile(self) -> np.ndarray:
        """The ``ceil(n / 10)`` largest clients by size, largest first (ties by index)."""
        count = -(-self.num_clients // 10)
        return np.lexsort((np.arange(self.num_clients), -self.client_sizes))[:count]


@lru_cache(maxsize=8)
def build_layout(train_labels: tuple[int, ...], test_labels: tuple[int, ...], *, num_clients: int,
                 layout_seed: int, alpha: float = DEFAULT_ALPHA, split_factor: int = 1) -> Layout:
    if split_factor < 1 or num_clients % split_factor:
        raise ValueError("num_clients must be a positive multiple of split_factor.")
    parts = dirichlet_partitions(train_labels, num_clients // split_factor, alpha, layout_seed)
    if split_factor > 1:
        parts = [chunk.tolist() for t, part in enumerate(parts)
                 for chunk in np.array_split(np.random.default_rng([layout_seed, t]).permutation(part), split_factor)]
    train, test = zip(*(split_stratified(train_labels, part, CLIENT_TRAIN_FRACTION, seed=layout_seed + t)
                        for t, part in enumerate(parts)), strict=True)
    public, server_test = split_stratified(test_labels, list(range(len(test_labels))), SERVER_PUBLIC_FRACTION,
                                           seed=layout_seed)
    as_array = lambda indices: np.asarray(sorted(indices), dtype=np.int64)  # noqa: E731
    return Layout(num_clients=num_clients, alpha=alpha, layout_seed=layout_seed,
                  client_train=tuple(map(as_array, train)), client_test=tuple(map(as_array, test)),
                  public=as_array(public), server_test=as_array(server_test))
