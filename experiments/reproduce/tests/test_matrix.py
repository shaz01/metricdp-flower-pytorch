"""Tests for reproduction matrix expansion."""

from experiments.reproduce.matrix import Hyperparams, Matrix


def test_list_combos_only_uses_first_noise_multiplier_for_vanilla() -> None:
    matrix = Matrix(
        partitions=("homogeneous",),
        privacy_modes=("vanilla", "global-dp", "metric-privacy"),
        aggregations=("fedavg",),
        seeds=(42,),
        noise_multipliers=(0.01, 0.1, 1.0),
        dirichlet_alphas=(0.5,),
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


def test_matrix_propagates_dirichlet_alpha_into_names_and_runner_args(tmp_path) -> None:
    matrix = Matrix(
        partitions=("dirichlet",),
        privacy_modes=("vanilla",),
        aggregations=("fedavg",),
        seeds=(42,),
        noise_multipliers=(0.01,),
        hyperparams=Hyperparams(
            clipping_norm=5.0,
            rounds=2,
            local_epochs=1,
            batch_size=32,
            learning_rate=0.001,
            initialization_epochs=1,
        ),
        data_module="example.data:create_data_module",
        model_module="example.model:create_model",
        dirichlet_alphas=(0.1,),
    )

    combo = matrix.list_combos(name_prefix="test", num_clients=8)[0]
    args = combo.runner_args(
        output_dir=tmp_path, max_parallel_clients=2, client_cpus=1.0
    )

    assert combo.dirichlet_alpha == 0.1
    assert "__dirichlet__alpha-0p1__" in combo.run_name()
    assert args[args.index("--dirichlet-alpha") + 1] == "0.1"


def test_data_tag_overrides_only_the_run_name_token() -> None:
    from dataclasses import replace

    from experiments.reproduce.matrix import Combo

    combo = Combo(
        name_prefix="x", num_clients=4, partition="homogeneous", privacy="vanilla",
        aggregation="fedavg", seed=42, noise_multiplier=0.01,
        hyperparams=Hyperparams(clipping_norm=5.0, rounds=2, local_epochs=1, batch_size=8,
                                learning_rate=0.001, initialization_epochs=0),
        data_module="pkg.new_name:create", model_module="pkg.model:create_model",
    )
    assert "__new_name__model" in combo.run_name()
    tagged = replace(combo, data_tag="old_name")
    assert tagged.run_name() == combo.run_name().replace("__new_name__", "__old_name__")
    kwargs = dict(output_dir=None, max_parallel_clients=1, client_cpus=1)
    assert [a for a in tagged.runner_args(**kwargs) if a != tagged.run_name()] == [
        a for a in combo.runner_args(**kwargs) if a != combo.run_name()
    ]
