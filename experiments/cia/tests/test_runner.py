"""Fast config-plumbing tests for the CIA runner (no real training)."""

from __future__ import annotations

from experiments.cia.runner import (
    CIA_CLIPPING_NORM,
    CIA_LOCAL_EPOCHS,
    CIA_MATRIX,
    CIA_NOISE_MULTIPLIER,
    CIA_SEED,
    build_cia_combos,
)


def test_cia_matrix_contains_paper_privacy_aggregation_grid() -> None:
    combos = build_cia_combos()

    assert len(combos) == len(CIA_MATRIX.privacy_modes) * len(CIA_MATRIX.aggregations)
    assert {combo.privacy for combo in combos} == set(CIA_MATRIX.privacy_modes)
    assert {combo.aggregation for combo in combos} == set(CIA_MATRIX.aggregations)


def test_cia_combos_use_paper_settings() -> None:
    combo = build_cia_combos(
        privacy_modes=("global-dp",), aggregations=("fedprox",)
    )[0]

    assert combo.num_clients == 3
    assert combo.partition == "homogeneous"
    assert combo.privacy == "global-dp"
    assert combo.aggregation == "fedprox"
    assert combo.seed == CIA_SEED
    assert combo.noise_multiplier == CIA_NOISE_MULTIPLIER
    assert combo.hyperparams.clipping_norm == CIA_CLIPPING_NORM
    assert combo.hyperparams.rounds == 1
    assert combo.hyperparams.local_epochs == CIA_LOCAL_EPOCHS
    assert combo.data_module == (
        "experiments.cia.datasets.paper:create_paper_shadow_data_module"
    )
    assert combo.run_name().endswith("__epochs-20__paper")


def test_cia_combo_runner_args_are_matrix_api_compatible(tmp_path) -> None:
    combo = build_cia_combos(
        privacy_modes=("vanilla",), aggregations=("fedavg",)
    )[0]
    joined = " ".join(
        combo.runner_args(
            output_dir=tmp_path,
            max_parallel_clients=2,
            client_cpus=1.0,
            save_model=True,
        )
    )

    assert "--data-module experiments.cia.datasets.paper:create_paper_shadow_data_module" in joined
    assert "--num-clients 3" in joined
    assert "--rounds 1" in joined
    assert "--local-epochs 20" in joined
    assert "--privacy vanilla" in joined
    assert "--aggregation fedavg" in joined
    assert "--save-model" in joined
