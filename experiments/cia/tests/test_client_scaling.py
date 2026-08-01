"""Tests for the unified 48-client CIA checkpoint script."""

from __future__ import annotations

from experiments.cia import client_scaling


def test_scaling_matrix_trains_one_trajectory_per_configuration() -> None:
    combos = client_scaling.build_combos()

    assert len(combos) == 2 * 3 * 2
    assert {combo.hyperparams.rounds for combo in combos} == {20}
    assert {combo.hyperparams.local_epochs for combo in combos} == {5}
    assert {combo.noise_multiplier for combo in combos} == {0.05}
    assert {combo.num_clients for combo in combos} == {48}
    assert client_scaling.CHECKPOINT_ROUNDS == (1, 20)


def test_scaling_uses_generic_shadow_module_for_each_partition() -> None:
    combos = client_scaling.build_combos()
    modes = {}
    for combo in combos:
        modes[combo.partition] = client_scaling._data_module_for(combo).partition_mode

    assert modes == {"homogeneous": "homogeneous", "non-iid": "non-iid"}
    assert all(
        combo.data_module
        == "experiments.cia.datasets.shadow:create_shadow_data_module"
        for combo in combos
    )


def test_scaling_passes_both_rounds_to_attack_runner(tmp_path, monkeypatch) -> None:
    captured = {}

    def fake_run_attack(combos, **kwargs):
        captured["combos"] = combos
        captured.update(kwargs)
        return []

    monkeypatch.setattr(client_scaling, "run_attack", fake_run_attack)

    results = client_scaling.run_client_scaling_cia(output_dir=tmp_path)

    assert results == []
    assert len(captured["combos"]) == 12
    assert captured["checkpoint_rounds"] == (1, 20)
    assert captured["report_name"] == "cia_client_scaling.json"
