"""Planning, data-pipeline and analysis tests; no dataset download or training."""
import json

import pytest
import torch
from PIL import Image

from experiments.reproduce.dataset.eurosat import EurosatDataModule, RGBImageTransform
from experiments.reproduce.runner import _parser, build_run_config
from results.cia_frontier.dataset_vs_model import analyze, data, runner


def test_plan_is_2x2_cells_times_seeds():
    combos = runner.build_combos(seeds=[42, 43, 44])
    assert len(combos) == 12 and len({c.run_name() for c in combos}) == 12
    assert {c.partition for c in combos} == {"homogeneous"}
    assert {c.privacy for c in combos} == {"metric-privacy"}
    assert {c.hyperparams.rounds for c in combos} == {10}
    assert {c.num_clients for c in combos} == {48}
    assert all(c.noise_multiplier == pytest.approx(0.0025 * 48) for c in combos)
    assert {(c.data_tag, c.model_module.rsplit(".", 1)[-1].split(":")[0]) for c in combos} == {
        ("cifar10", "cifar10_cnn"), ("cifar10", "eurosat_cnn"),
        ("eurosat32", "cifar10_cnn"), ("eurosat32", "eurosat_cnn")}


def test_run_names_distinguish_dataset_and_model():
    names = [c.run_name() for c in runner.build_combos(seeds=[42])]
    assert any("__cifar10__cifar10_cnn" in n for n in names)
    assert any("__eurosat32__eurosat_cnn" in n for n in names)
    assert any("__eurosat32__cifar10_cnn" in n for n in names)


def test_sharding_and_validation():
    (one,) = runner.build_combos(seeds=[42], cells=["eurosat32+cifar10_cnn"])
    assert one.data_module.endswith(":create_eurosat32")
    assert one.model_module.endswith("cifar10_cnn:create_model")
    with pytest.raises(ValueError):
        runner.build_combos(seeds=[42], cells=["nope"])
    with pytest.raises(ValueError):
        runner.build_combos(seeds=[42, 42])
    vanilla = runner.build_combos(seeds=[42], privacy="vanilla")
    assert {c.noise_multiplier for c in vanilla} == {0.0}


def test_runner_args_reach_server_config(tmp_path):
    (combo,) = runner.build_combos(seeds=[42], cells=["cifar10+eurosat_cnn"])
    args = combo.runner_args(output_dir=tmp_path, max_parallel_clients=6, client_cpus=1.0)
    config = build_run_config(_parser().parse_args(list(args)))
    assert config["privacy"] == "metric-privacy" and config["partition-mode"] == "homogeneous"
    assert config["num-clients"] == 48 and config["num-server-rounds"] == 10
    assert config["data-module"] == data.DATASETS["cifar10"]
    assert config["max-client-samples"] == 450


def test_every_cell_caps_clients_to_eurosat_size():
    combos = runner.build_combos(seeds=[42])
    assert {c.max_client_samples for c in combos} == {450}
    assert all("-cap450-" in c.run_name() for c in combos)


def test_eurosat32_transform_resizes_and_skips_augmentation():
    image = Image.new("RGB", (64, 64), color=(10, 20, 30))
    plain = RGBImageTransform((64, 64), augment=False, resize_to=(32, 32))(image)
    assert plain.shape == (3, 32, 32)
    aug = RGBImageTransform((64, 64), augment=True, resize_to=(32, 32))(image)
    assert aug.shape == (3, 32, 32)
    native = RGBImageTransform((64, 64), augment=False)(image)
    assert native.shape == (3, 64, 64)
    with pytest.raises(ValueError):
        RGBImageTransform((64, 64))(Image.new("RGB", (32, 32)))


def test_eurosat32_module_views_are_32px_and_unaugmented():
    module = data.create_eurosat32({})
    assert module.resize_to == (32, 32) and module.augment is False
    view = module._view([], train=True)
    assert view.transform.resize_to == (32, 32) and view.transform.augment is False
    default = EurosatDataModule()
    assert default.resize_to is None and default.augment is True
    assert default._view([], train=True).transform.augment is True
    with pytest.raises(ValueError):
        EurosatDataModule(resize_to=(0, 32))


def test_eurosat_cnn_accepts_32px_and_cifar10_cnn_requires_it():
    from experiments.reproduce.cifar10_cnn import create_model as cifar
    from experiments.reproduce.eurosat_cnn import create_model as eurosat
    x = torch.zeros(2, 3, 32, 32)
    assert eurosat()(x).shape == (2, 10) and cifar()(x).shape == (2, 10)
    with pytest.raises(RuntimeError):
        cifar()(torch.zeros(2, 3, 64, 64))


def _fake_run(seed, norms, dists, acc):
    return {"metadata": {"seed": seed},
            "train_metrics": {str(r): {"dp-update-norms-before-clipping": norms,
                                       "metric-dp-pairwise-distances": dists,
                                       "metric-dp-noise-stdv": 0.01} for r in range(1, 4)},
            "server_evaluate_metrics": {"0": {"accuracy": 0.1}, "3": {"accuracy": acc}}}


def test_analyze_groups_by_cell(tmp_path):
    for combo in runner.build_combos(seeds=[42, 43], cells=["cifar10+cifar10_cnn"]):
        combo.result_path(tmp_path).write_text(json.dumps(
            _fake_run(combo.seed, [6.0, 7.0, 8.0], [1.0, 1.5, 2.0], 0.5 + combo.seed / 1000)))
    (tmp_path / "unrelated.json").write_text("{}")
    first = runner.build_combos(seeds=[42], cells=["cifar10+cifar10_cnn"])[0].run_name()
    (tmp_path / f"{first}.evaluation.json").write_text(json.dumps({"metadata": {}}))
    summary = analyze.summarize(tmp_path)
    assert list(summary) == ["cifar10+cifar10_cnn"]
    cell = summary["cifar10+cifar10_cnn"]
    assert cell["seeds"] == 2 and cell["update_norm"] == 7.0
    assert cell["max_pairwise_distance"] == 2.0 and cell["median_pairwise_distance"] == 1.5
    assert cell["final_accuracy"] == pytest.approx(0.5425)
    assert analyze.summarize(tmp_path, skip_rounds=3)["cifar10+cifar10_cnn"]["update_norm"] is None


def test_analyze_reads_per_cell_subdirs(tmp_path):
    for cell in ("cifar10+cifar10_cnn", "eurosat32+eurosat_cnn"):
        sub = tmp_path / cell
        sub.mkdir()
        combo = runner.build_combos(seeds=[42], cells=[cell])[0]
        combo.result_path(sub).write_text(json.dumps(_fake_run(42, [6.0], [1.0], 0.5)))
    assert sorted(analyze.summarize(tmp_path)) == ["cifar10+cifar10_cnn", "eurosat32+eurosat_cnn"]
