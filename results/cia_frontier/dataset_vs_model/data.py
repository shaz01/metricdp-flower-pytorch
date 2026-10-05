"""Stage A data modules: CIFAR-10 and EuroSAT with identical pipelines.

Both are RGB 32x32, no training-time augmentation, homogeneous partitions.
EuroSAT is downsized from 64x64 (bilinear) so the 32x32-only ``cifar10_cnn``
can run on it, and its usual random crop + flip is switched off because the
CIFAR-10 pipeline never had augmentation.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from experiments.reproduce.dataset.cifar10 import Cifar10DataModule
from experiments.reproduce.dataset.eurosat import EurosatDataModule

IMAGE_SIZE = (32, 32)
EUROSAT_TRAIN_SIZE = 21_600  # EuroSAT's training split; CIFAR-10 is cut to this for Stage B


def _cache_dir(config: Mapping[str, Any]) -> str | None:
    return str(config.get("data-cache-dir", "")).strip() or None


def create_cifar10(config: Mapping[str, Any]) -> Cifar10DataModule:
    """Full 10-class CIFAR-10, native 32x32, no augmentation (unchanged plugin)."""
    return Cifar10DataModule(cache_dir=_cache_dir(config))


def create_eurosat32(config: Mapping[str, Any]) -> EurosatDataModule:
    """EuroSAT downsized to 32x32 with augmentation off, to match CIFAR-10."""
    return EurosatDataModule(cache_dir=_cache_dir(config), augment=False, resize_to=IMAGE_SIZE)


def create_cifar10_small(config: Mapping[str, Any]) -> Cifar10DataModule:
    """CIFAR-10 cut to a fixed stratified 21,600-image pool before partitioning.

    Matches EuroSAT's pool size so both datasets give clients the same amount of data
    under any partition, including Dirichlet. Pool is fixed across training seeds.
    """
    return Cifar10DataModule(cache_dir=_cache_dir(config), train_subsample=EUROSAT_TRAIN_SIZE,
                             subsample_seed=0)


DATASETS = {
    "cifar10": f"{__name__}:create_cifar10",
    "eurosat32": f"{__name__}:create_eurosat32",
}
STAGE_B_DATASETS = {
    "cifar10s": f"{__name__}:create_cifar10_small",
    "eurosat32": f"{__name__}:create_eurosat32",
}


# --- Stage B/C IN and OUT views -------------------------------------------------------
# A trajectory's ``partition-profile`` run-config value is JSON:
# {"dataset": "cifar10s"|"eurosat32", "clients": 48, "out_target": null|int}.
# IN (out_target null) trains all canonical clients, OUT drops one. Every other client
# keeps exactly its canonical records because partitions are always built for ``clients``.
# Optional "train_subsample" shrinks the CIFAR-10 pool; it is for local smoke runs only.

_BASES = {"cifar10s": create_cifar10_small, "eurosat32": create_eurosat32}


class AutoProfile:
    """Forward to a base module, replacing the JSON profile with the plugin's 'auto'."""

    def __init__(self, base):
        self.base = base

    @property
    def class_names(self):
        return getattr(self.base, "class_names", ())

    def client_loaders(self, partition_id, **kwargs):
        return self.base.client_loaders(partition_id, **{**kwargs, "partition_profile": "auto"})

    def server_loaders(self, **kwargs):
        return self.base.server_loaders(**kwargs)


def stage_b_profile(dataset: str, clients: int, out_target: int | None,
                    train_subsample: int | None = None) -> str:
    if dataset not in _BASES:
        raise ValueError(f"Unknown Stage B dataset {dataset!r}")
    profile = {"dataset": dataset, "clients": clients, "out_target": out_target}
    if train_subsample is not None:
        if dataset != "cifar10s":
            raise ValueError("train_subsample applies to cifar10s only")
        profile["train_subsample"] = train_subsample
    return json.dumps(profile, sort_keys=True)


def stage_b_base(settings: Mapping[str, Any], config: Mapping[str, Any]):
    """Canonical (all-clients) data module for one profile; shared by training and shadows."""
    module = _BASES[settings["dataset"]](config)
    if "train_subsample" in settings:
        module.train_subsample = int(settings["train_subsample"])
    return module


def create_stage_b_view(config: Mapping[str, Any]):
    from experiments.cia.datasets.partitions import in_remove, out_remove
    settings = json.loads(config["partition-profile"])
    target = settings["out_target"]
    factory = in_remove if target is None else out_remove
    return factory(AutoProfile(stage_b_base(settings, config)),
                   canonical_num_partitions=settings["clients"],
                   target_partition_id=0 if target is None else target)


STAGE_B_VIEW = f"{__name__}:create_stage_b_view"
