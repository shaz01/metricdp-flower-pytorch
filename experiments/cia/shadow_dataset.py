from experiments.cia.datasets.partitions import PartitionViewDataModule
from experiments.cia.datasets.shadow import ShadowDataModule
from experiments.reproduce.matrix import Combo
from metricdp_pytorch.data_module import FederatedDataModule, load_data_module
from metricdp_pytorch.utils.noisy_dataset import NoisyDataModule


def _configured_data_module(
    combo: Combo,
    *,
    target_partition_id: int,
    shadow_fraction: float,
) -> FederatedDataModule:
    """Load the same configured data-module factory used for training."""
    return load_data_module(
        combo.data_module,
        {
            "num-clients": combo.num_clients,
            "partition-mode": combo.partition,
            "target-partition-id": target_partition_id,
            "shadow-fraction": shadow_fraction,
            "seed": combo.seed,
        },
    )


def _canonical_data_module(
    data_module: FederatedDataModule,
    *,
    active_num_partitions: int,
) -> tuple[FederatedDataModule, int]:
    if isinstance(data_module, PartitionViewDataModule):
        return data_module.data_module, data_module.canonical_num_partitions
    return data_module, active_num_partitions


def _shadow_dataset(
    combo: Combo,
    data_module: FederatedDataModule,
    *,
    target_partition_id: int,
    shadow_fraction: float,
) -> ShadowDataModule:
    canonical_module, canonical_count = _canonical_data_module(
        data_module, active_num_partitions=combo.num_clients
    )
    return ShadowDataModule(
        canonical_module,
        num_clients=canonical_count,
        target_partition_id=target_partition_id,
        shadow_fraction=shadow_fraction,
        partition_mode=combo.partition,
        partition_profile="auto",
    )


def clean_shadow_dataset(
    combo: Combo,
    *,
    target_partition_id: int,
    shadow_fraction: float,
) -> ShadowDataModule:
    return _shadow_dataset(
        combo,
        _configured_data_module(
            combo,
            target_partition_id=target_partition_id,
            shadow_fraction=shadow_fraction,
        ),
        target_partition_id=target_partition_id,
        shadow_fraction=shadow_fraction,
    )


def inject_noise(
    data_module: FederatedDataModule,
    *,
    std_fraction: float = 0.1,
) -> FederatedDataModule:
    return NoisyDataModule(data_module, std_fraction=std_fraction)


def noisy_shadow_dataset(
    combo: Combo,
    *,
    target_partition_id: int,
    shadow_fraction: float,
    std_fraction: float = 0.1,
) -> ShadowDataModule:
    configured_module = _configured_data_module(
        combo,
        target_partition_id=target_partition_id,
        shadow_fraction=shadow_fraction,
    )
    canonical_module, canonical_count = _canonical_data_module(
        configured_module, active_num_partitions=combo.num_clients
    )
    return ShadowDataModule(
        inject_noise(canonical_module, std_fraction=std_fraction),
        num_clients=canonical_count,
        target_partition_id=target_partition_id,
        shadow_fraction=shadow_fraction,
        partition_mode=combo.partition,
        partition_profile="auto",
    )
