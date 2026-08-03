"""Tests for resumable CIA attack execution."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from experiments.cia import attack_runner
from experiments.cia.result import make_cia_result
from experiments.reproduce.matrix import Combo, Hyperparams


def _combo(seed: int) -> Combo:
    return Combo(
        name_prefix="cia",
        num_clients=3,
        partition="homogeneous",
        privacy="vanilla",
        aggregation="fedavg",
        seed=seed,
        noise_multiplier=0.01,
        hyperparams=Hyperparams(
            clipping_norm=5.0,
            rounds=1,
            local_epochs=1,
            batch_size=32,
            learning_rate=0.001,
            initialization_epochs=1,
        ),
        data_module="experiments.cia.datasets.paper:create_paper_shadow_data_module",
        model_module="experiments.reproduce.paper_cnn:create_model",
    )


def test_force_reruns_requested_combo_without_removing_other_results(
    tmp_path, monkeypatch
) -> None:
    previous_combo = _combo(seed=1)
    requested_combo = _combo(seed=2)
    previous_result = make_cia_result(
        combo=previous_combo,
        server_round=1,
        aggregated_test_loss=1.0,
        target_clean_shadow_loss=2.0,
        target_noisy_shadow_loss=2.5,
        shadow_fraction=0.1,
        shadow_size=10,
    )
    report_path = tmp_path / "cia.json"
    report_path.write_text(
        json.dumps([previous_result.__dict__]), encoding="utf-8"
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()

    def fake_iter_combos(combos, **_kwargs):
        assert list(combos) == [requested_combo]
        yield requested_combo, True, (checkpoint_path,)

    monkeypatch.setattr(attack_runner, "iter_combos", fake_iter_combos)
    monkeypatch.setattr(
        attack_runner.cia,
        "eval_model",
        lambda *_args, **_kwargs: (1.0, 2.0, 2.5, 10),
    )

    results = attack_runner.run_attack(
        [requested_combo],
        output_dir=tmp_path,
        log_path=tmp_path / "attack.log",
        max_parallel_clients=1,
        force=True,
        start_message="start",
        clean_data_module_factory=lambda _combo: SimpleNamespace(
            shadow_fraction=0.1
        ),
        noisy_data_module_factory=lambda _combo: SimpleNamespace(
            shadow_fraction=0.1
        ),
        device=torch.device("cpu"),
        checkpoint_rounds=(1,),
    )

    assert {(result.seed, result.server_round) for result in results} == {(1, 1), (2, 1)}
    saved_results = json.loads(report_path.read_text(encoding="utf-8"))
    assert {(result["seed"], result["server_round"]) for result in saved_results} == {
        (1, 1),
        (2, 1),
    }
