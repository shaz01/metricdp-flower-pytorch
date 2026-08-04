"""Tests for CIA combo checkpoint iteration."""

from __future__ import annotations

from experiments.cia import iter_combos as iter_module
from experiments.cia.runner import build_cia_combos


def test_iter_combos_returns_every_requested_checkpoint_path(
    tmp_path, monkeypatch
) -> None:
    combo = build_cia_combos(
        privacy_modes=("vanilla",), aggregations=("fedavg",)
    )[0]

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
    combo = build_cia_combos(
        privacy_modes=("vanilla",), aggregations=("fedavg",)
    )[0]
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
