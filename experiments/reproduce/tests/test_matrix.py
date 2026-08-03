"""Tests for reproduction matrix expansion."""

from experiments.reproduce.matrix import Hyperparams, Matrix


def test_list_combos_only_uses_first_noise_multiplier_for_vanilla() -> None:
    matrix = Matrix(
        partitions=("homogeneous",),
        privacy_modes=("vanilla", "global-dp", "metric-privacy"),
        aggregations=("fedavg",),
        seeds=(42,),
        noise_multipliers=(0.01, 0.1, 1.0),
        hyperparams=Hyperparams(
            clipping_norm=5.0,
            rounds=20,
            local_epochs=5,
            batch_size=32,
            learning_rate=0.001,
            initialization_epochs=20,
        ),
        data_module="example.data:create_data_module",
        model_module="example.model:create_model",
    )

    combos = matrix.list_combos(name_prefix="test", num_clients=8)
    multipliers_by_privacy = {
        privacy: [
            combo.noise_multiplier for combo in combos if combo.privacy == privacy
        ]
        for privacy in matrix.privacy_modes
    }

    assert multipliers_by_privacy == {
        "vanilla": [0.01],
        "global-dp": [0.01, 0.1, 1.0],
        "metric-privacy": [0.01, 0.1, 1.0],
    }
