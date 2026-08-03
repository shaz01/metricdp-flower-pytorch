"""Tests for CIA combo checkpoint iteration."""

from __future__ import annotations

from experiments.cia import iter_combos as iter_module
from experiments.reproduce.matrix import Combo, Hyperparams


def _combo() -> Combo:
    return Combo(
        name_prefix="cia",
        num_clients=3,
        partition="homogeneous",
        privacy="vanilla",
        aggregation="fedavg",
        seed=42,
        noise_multiplier=0.01,
        hyperparams=Hyperparams(
            clipping_norm=5.0,
            rounds=1,
            local_epochs=20,
            batch_size=32,
            learning_rate=0.001,
            initialization_epochs=20,
        ),
        data_module="experiments.cia.datasets.paper:create_paper_shadow_data_module",
        model_module="experiments.reproduce.paper_cnn:create_model",
    )


def test_iter_combos_returns_every_requested_checkpoint_path(
    tmp_path, monkeypatch
) -> None:
    combo = _combo()

    monkeypatch.setattr(iter_module, "run_one_combo", lambda *_args, **_kwargs: True)

    results = list(
        iter_module.iter_combos(
            [combo],
            output_dir=tmp_path,
            max_parallel_clients=2,
            log=lambda _message: None,
            checkpoint_rounds=(1, 3),
        )
    )

    assert results == [
        (
            combo,
            True,
            (
                tmp_path / f"{combo.run_name()}.round-1.pt",
                tmp_path / f"{combo.run_name()}.round-3.pt",
            ),
        )
    ]


def test_iter_combos_returns_empty_paths_without_checkpoint_rounds(
    tmp_path, monkeypatch
) -> None:
    combo = _combo()
    monkeypatch.setattr(iter_module, "run_one_combo", lambda *_args, **_kwargs: True)

    [(returned_combo, success, paths)] = iter_module.iter_combos(
        [combo],
        output_dir=tmp_path,
        max_parallel_clients=2,
        log=lambda _message: None,
    )

    assert returned_combo == combo
    assert success is True
    assert paths == ()
