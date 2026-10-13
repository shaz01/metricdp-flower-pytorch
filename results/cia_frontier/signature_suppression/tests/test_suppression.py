"""Synthetic tests for the WP2 local-training options (no datasets, no GPU)."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import experiments.reproduce.paper_training as paper_training
from experiments.reproduce.paper_loss import sparse_categorical_cross_entropy
from experiments.reproduce.client import _local_training_options
from experiments.reproduce.runner import _parser, build_run_config
from metricdp_pytorch.strategy_factory import FEDPROX_MU, make_strategy
from results.cia_frontier.signature_suppression import runner

CPU = torch.device("cpu")


def _loader(batches=3, batch_size=2):
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(batches * batch_size, 2, generator=generator)
    labels = torch.tensor([0, 1] * batches)
    return DataLoader(TensorDataset(inputs, labels), batch_size=batch_size, shuffle=False)


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 2), nn.Softmax(dim=1))


class _CountingSGD(torch.optim.SGD):
    steps = 0

    def step(self, closure=None):
        type(self).steps += 1
        return super().step(closure)


def _count_steps(monkeypatch, **kwargs):
    _CountingSGD.steps = 0
    monkeypatch.setattr(paper_training.torch.optim, "Adam",
                        lambda params, lr, weight_decay: _CountingSGD(params, lr=lr))
    losses = paper_training.train_with_adam(_model(), _loader(), learning_rate=0.01, device=CPU, **kwargs)
    return _CountingSGD.steps, losses


def test_consensus_kl_matches_hand_computation():
    local = torch.tensor([[0.7, 0.3], [0.5, 0.5]])
    consensus = torch.tensor([[0.4, 0.6], [0.5, 0.5]])
    expected = (0.4 * torch.log(torch.tensor(0.4 / 0.7)) + 0.6 * torch.log(torch.tensor(0.6 / 0.3))) / 2
    assert paper_training.consensus_kl_term(local, consensus).item() == pytest.approx(expected.item(), rel=1e-6)
    assert paper_training.consensus_kl_term(consensus, consensus).item() == pytest.approx(0.0, abs=1e-7)


def test_epochs_unchanged_by_default(monkeypatch):
    steps, losses = _count_steps(monkeypatch, epochs=2)
    assert steps == 6 and len(losses) == 2


@pytest.mark.parametrize("local_steps, passes", [(2, 1), (3, 1), (7, 3)])
def test_fixed_local_steps_cycle_the_loader(monkeypatch, local_steps, passes):
    steps, losses = _count_steps(monkeypatch, epochs=5, local_steps=local_steps)
    assert steps == local_steps and len(losses) == passes


def test_consensus_term_is_zero_on_the_first_batch():
    # The consensus model is the entry state, so the first batch's objective is plain CE.
    loader = _loader(batches=1)
    model = _model()
    images, labels = next(iter(loader))
    with torch.no_grad():
        plain = sparse_categorical_cross_entropy(model(images), labels).item()
    losses = paper_training.train_with_adam(copy.deepcopy(model), loader, epochs=1, learning_rate=0.01,
                                            device=CPU, consensus_beta=1.0)
    assert losses[0] == pytest.approx(plain, rel=1e-6)


def test_consensus_term_changes_training():
    plain, regularized = _model(), _model()
    for model, beta in ((plain, 0.0), (regularized, 1.0)):
        paper_training.train_with_adam(model, _loader(), epochs=3, learning_rate=0.1, device=CPU,
                                       consensus_beta=beta)
    assert not torch.equal(plain[0].weight, regularized[0].weight)


def test_proximal_sum_convention_is_mean_times_parameter_count():
    """One SGD step: the proximal gradient at the entry point is zero, so compare two steps."""
    results = {}
    for normalization, mu in (("mean", 6.0), ("sum", 1.0)):  # 6 parameters: same penalty
        model = _model()
        torch.manual_seed(1)
        paper_training.train_with_adam(model, _loader(batches=2), epochs=1, learning_rate=0.1, device=CPU,
                                       proximal_mu=mu, proximal_normalization=normalization)
        results[normalization] = [p.detach().clone() for p in model.parameters()]
    assert sum(p.numel() for p in _model().parameters()) == 6
    for mean, total in zip(results["mean"], results["sum"], strict=True):
        assert torch.allclose(mean, total, atol=1e-6)


def test_invalid_options_are_rejected():
    for kwargs in ({"consensus_beta": -1.0}, {"local_steps": -1}, {"proximal_normalization": "max"}):
        with pytest.raises(ValueError):
            paper_training.train_with_adam(_model(), _loader(), epochs=1, device=CPU, **kwargs)


def test_strategy_factory_proximal_mu():
    common = dict(privacy="vanilla", num_clients=3, fraction_evaluate=1.0, noise_multiplier=0.0,
                  clipping_norm=5.0)
    assert make_strategy("fedprox", **common).proximal_mu == FEDPROX_MU == 0.5
    assert make_strategy("fedprox", **common, proximal_mu=0.01).proximal_mu == 0.01
    with pytest.raises(ValueError):
        make_strategy("fedavg", **common, proximal_mu=0.01)


def _args(*arguments):
    return _parser().parse_args(["--seed", "42", "--noise-multiplier", "0", "--clipping-norm", "5",
                                 "--rounds", "2", "--local-epochs", "5",
                                 "--model-module", "experiments.reproduce.paper_cnn:create_model", *arguments])


def test_runner_config_keys_only_when_set(tmp_path):
    default = build_run_config(_args("--output-dir", str(tmp_path)))
    assert not {"proximal-mu", "proximal-normalization", "consensus-kl-beta", "local-steps"} & set(default)
    config = build_run_config(_args("--output-dir", str(tmp_path), "--aggregation", "fedprox",
                                    "--proximal-mu", "0.1", "--proximal-normalization", "sum",
                                    "--consensus-kl-beta", "0.3", "--local-steps", "58"))
    assert (config["proximal-mu"], config["proximal-normalization"]) == (0.1, "sum")
    assert (config["consensus-kl-beta"], config["local-steps"]) == (0.3, 58)
    with pytest.raises(ValueError):
        build_run_config(_args("--output-dir", str(tmp_path), "--proximal-mu", "0.1"))
    with pytest.raises(ValueError):
        build_run_config(_args("--output-dir", str(tmp_path), "--local-steps", "-1"))


def test_wp2_plan_matches_protocol():
    cells = runner.plan(list(runner.CONFIGS))
    combos = [combo for _, cell in cells for combo in cell]
    assert len(cells) == 10 and len(combos) == 60
    assert runner.TARGETS == (23, 7, 1, 0, 3)
    for combo in combos:
        assert combo.privacy == "vanilla" and combo.noise_multiplier == 0
        assert combo.canonical_clients == 48 and combo.hyperparams.rounds == 50 and combo.diagnostics
        assert combo.num_clients == (48 if combo.out_target is None else 47)
        expected_weighting = "num-examples" if combo.config == "vanilla-size-weighted" else "equal"
        assert combo.aggregation_weighting == expected_weighting
    names = [combo.run_name() for combo in combos]
    assert len(set(names)) == 60
    by_config = {combo.config: combo for combo in combos}
    args = lambda c: by_config[c].runner_args(output_dir=Path("x"), max_parallel_clients=8, client_cpus=1.0)
    assert args("fedprox-mu-1")[args("fedprox-mu-1").index("--proximal-mu") + 1] == "1.0"
    assert "sum" in args("fedprox-mu-1") and by_config["fedprox-mu-1"].aggregation == "fedprox"
    assert args("fixed-steps")[-1] == "58" and "--consensus-kl-beta" in args("consensus-kl-0.3")
    assert by_config["weight-decay-5e-4"].hyperparams.weight_decay == 5e-4
    assert by_config["local-epochs-1"].hyperparams.local_epochs == 1
    vanilla = args("vanilla")
    assert not {"--proximal-mu", "--consensus-kl-beta", "--local-steps"} & set(vanilla)
    assert vanilla[vanilla.index("--weight-decay") + 1] == "0.0"


def test_fixed_step_budget_is_the_vanilla_mean():
    # Per-client 5-epoch step counts at seed 42 sum to 2,800 over 48 clients (PROTOCOL.md Section 3).
    assert runner.FIXED_LOCAL_STEPS == round(2800 / 48)


def test_client_passes_options_only_when_set():
    assert _local_training_options({"local-epochs": 5}) == {}
    assert _local_training_options({"proximal-normalization": "sum", "consensus-kl-beta": 0.3,
                                    "local-steps": 58}) == {
        "proximal_normalization": "sum", "consensus_beta": 0.3, "local_steps": 58}


def test_auc_and_tpr_match_brute_force():
    from results.cia_frontier.signature_suppression.analyze import auc, tpr_at_fpr

    rng = torch.Generator().manual_seed(3)
    positive = (torch.randint(0, 6, (40,), generator=rng) / 2).tolist()
    negative = (torch.randint(0, 5, (30,), generator=rng) / 2).tolist()
    brute = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (40 * 30)
    assert auc(positive, negative) == pytest.approx(brute)
    level = 0.1
    best = 0.0
    for threshold in set(positive) | set(negative):
        fpr = sum(n >= threshold for n in negative) / 30
        if fpr <= level:
            best = max(best, sum(p >= threshold for p in positive) / 40)
    assert tpr_at_fpr(positive, negative, level) == pytest.approx(best)


def _synthetic(shift, accuracy, targets=(23, 7, 1, 0, 3)):
    """Gaps L_OUT - L_IN per target over 50 rounds: larger shift means more leakage."""
    import numpy as np

    rng = np.random.default_rng(int(shift * 1000) + 7)
    cells = {(42, t): {view: rng.normal(shift * (1 + i / 4), 1.0, 50) for view in ("clean", "noisy")}
             for i, t in enumerate(targets)}
    return dict(cells=cells, accuracy=[accuracy] * 6, in_accuracy=[accuracy],
                client_accuracy=[{t: accuracy for t in targets}], loo_median=[{t: 1.0 for t in targets}])


def test_gate_pass_and_fail_on_synthetic_gaps():
    from results.cia_frontier.signature_suppression import analyze

    data = {"vanilla": _synthetic(1.5, 0.80), "weight-decay-5e-4": _synthetic(1.4, 0.80),
            "local-epochs-1": _synthetic(1.0, 0.70), "consensus-kl-1": _synthetic(0.0, 0.80),
            "fedprox-mu-1": _synthetic(0.0, 0.75)}
    # The remaining candidates leak exactly like vanilla (identical gaps): they fail part 1.
    for name in ("fedprox-mu-0.01", "fedprox-mu-0.1", "fixed-steps", "consensus-kl-0.3"):
        data[name] = _synthetic(1.5, 0.80)
    partial = analyze.evaluate({k: data[k] for k in ("vanilla", "consensus-kl-1")}, resamples=50)
    assert partial["gate"]["G2"].startswith("incomplete")
    result = analyze.evaluate(data, resamples=200)
    gate = result["gate"]
    # Leak removed at equal accuracy: holds both parts. Same leak removal at 5 pp accuracy loss: fails part 1.
    assert gate["candidates"]["consensus-kl-1"]["part1"] == "holds"
    assert gate["candidates"]["consensus-kl-1"]["part2"] == "holds"
    assert gate["candidates"]["fedprox-mu-1"]["part1"] == "fails"
    assert gate["candidates"]["fixed-steps"]["part1"] == "fails"
    assert gate["G2"] == "pass" and gate["passing"] == ["consensus-kl-1"]
    data["consensus-kl-1"] = _synthetic(1.5, 0.80)
    assert analyze.evaluate(data, resamples=200)["gate"]["G2"] == "fail"
