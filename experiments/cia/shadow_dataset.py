from experiments.cia.datasets.shadow import ShadowDataModule
from experiments.reproduce.dataset.alzheimer import AlzheimerDataModule
from experiments.reproduce.matrix import Combo
from metricdp_pytorch.data_module import FederatedDataModule
from metricdp_pytorch.utils.noisy_dataset import NoisyDataModule


def _shadow_dataset(
    combo: Combo,
    data_module: FederatedDataModule,
    *,
    target_partition_id: int,
    shadow_fraction: float,
) -> ShadowDataModule:
    return ShadowDataModule(
        data_module,
        num_clients=combo.num_clients,
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
        AlzheimerDataModule(),
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
    return _shadow_dataset(
        combo,
        inject_noise(AlzheimerDataModule(), std_fraction=std_fraction),
        target_partition_id=target_partition_id,
        shadow_fraction=shadow_fraction,
    )
