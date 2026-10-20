"""CIA v2 B5: contribution caps and the logit-adjusted local loss (default off)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.reproduce import client, runner
from experiments.reproduce.paper_training import train_with_adam
from metricdp_pytorch.utils.data import make_indexed_loader
from results.cia_v2.randomized_inclusion import runner as inclusion


def _loader(n=40, classes=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n, 5, generator=generator)
    labels = torch.tensor([0] * (n - 4) + [1, 1, 2, 2])
    return make_indexed_loader(TensorDataset(features, labels), list(range(n)), batch_size=8, shuffle=True, seed=seed)


def _model(classes=3):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(5, classes), nn.Softmax(dim=1))


def test_capped_loader_draws_a_fresh_subset_per_seed():
    loader = _loader()
    capped = client._capped_loader(loader, 10, seed=1)
    assert len(capped.dataset) == 10 and capped.batch_size == loader.batch_size
    again = client._capped_loader(loader, 10, seed=1)
    other = client._capped_loader(loader, 10, seed=2)
    assert capped.dataset.indices == again.dataset.indices != other.dataset.indices
    assert client._capped_loader(loader, 0, seed=1) is loader
    assert client._capped_loader(loader, 40, seed=1) is loader


def test_log_label_prior_is_add_one_smoothed():
    prior = client._log_label_prior(_loader().dataset, 4).exp()
    assert prior.tolist() == pytest.approx([37 / 44, 3 / 44, 3 / 44, 1 / 44])
    assert client._num_classes(_model(3)) == 3


def test_logit_adjusted_loss_matches_its_definition():
    model = _model()
    images, labels = next(iter(DataLoader(_loader().dataset, batch_size=40)))
    log_prior = torch.log(torch.tensor([0.8, 0.15, 0.05]))
    with torch.no_grad():
        expected = nn.functional.cross_entropy(model(images).log() + log_prior, labels).item()
    one_step = DataLoader(TensorDataset(images, labels), batch_size=40)
    losses = train_with_adam(model, one_step, epochs=1, learning_rate=1e-12, log_prior=log_prior, logit_tau=1.0,
                             device=torch.device("cpu"))
    assert losses[0] == pytest.approx(expected, rel=1e-5)


def test_logit_adjustment_learns_a_less_skewed_output():
    images, labels = next(iter(DataLoader(_loader(n=200).dataset, batch_size=200)))
    log_prior = client._log_label_prior(TensorDataset(images, labels), 3)
    loader = DataLoader(TensorDataset(images, labels), batch_size=32)
    plain, adjusted = _model(), _model()
    train_with_adam(plain, loader, epochs=20, learning_rate=0.05, device=torch.device("cpu"))
    train_with_adam(adjusted, loader, epochs=20, learning_rate=0.05, log_prior=log_prior, logit_tau=1.0,
                    device=torch.device("cpu"))
    with torch.no_grad():
        assert adjusted(images)[:, 0].mean() < plain(images)[:, 0].mean()


def test_logit_adjustment_needs_a_prior():
    with pytest.raises(ValueError):
        train_with_adam(_model(), _loader(), epochs=1, logit_tau=1.0, device=torch.device("cpu"))


BASE = ["--seed", "42", "--noise-multiplier", "0.5", "--clipping-norm", "5.0", "--num-clients", "4",
        "--model-module", "experiments.reproduce.eurosat_cnn:create_model", "--rounds", "2", "--local-epochs", "1"]


def test_runner_passes_the_options_only_when_set():
    config = runner.build_run_config(runner._parser().parse_args(
        [*BASE, "--contribution-cap", "326", "--logit-adjustment-tau", "1.0"]))
    assert config["contribution-cap"] == 326 and config["logit-adjustment-tau"] == 1.0
    plain = runner.build_run_config(runner._parser().parse_args(BASE))
    assert "contribution-cap" not in plain and "logit-adjustment-tau" not in plain


def test_inclusion_runs_carry_the_options_into_args_and_manifest():
    run = inclusion.build_runs(config="b5", privacy="global-dp", noise_ratio=0.003092, replicates=2, design_seed=2,
                               contribution_cap=326, logit_tau=1.0)[0]
    args = run.runner_args(output_dir=Path("out"), max_parallel_clients=4, client_cpus=1.0,
                           checkpoint_rounds=inclusion.CHECKPOINT_ROUNDS)
    flags = dict(zip(args[::1], args[1::1]))
    assert flags["--contribution-cap"] == "326" and flags["--logit-adjustment-tau"] == "1.0"
    assert inclusion.manifest(run, surrogates=False)["local_training"] == {"contribution_cap": 326, "logit_tau": 1.0}
    plain = inclusion.build_runs(config="b5", privacy="global-dp", noise_ratio=0.003092, replicates=2,
                                 design_seed=2)[0]
    assert "local_training" not in inclusion.manifest(plain, surrogates=False)
    assert "--contribution-cap" not in plain.runner_args(output_dir=Path("out"), max_parallel_clients=4,
                                                         client_cpus=1.0, checkpoint_rounds=(0,))
