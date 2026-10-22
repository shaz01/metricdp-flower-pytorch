"""A failed or missing client stops the run instead of being aggregated around (2026-10-22, CIA v2)."""

from __future__ import annotations

import pytest

from experiments.reproduce import runner
from experiments.reproduce.server import make_run_strategy, require_all_clients


class Reply:
    def __init__(self, error=False):
        self.error = error

    def has_error(self):
        return self.error


class Strategy:
    def aggregate_train(self, server_round, replies):
        return "aggregated", len(list(replies))


def test_all_clients_pass_through():
    strategy = require_all_clients(Strategy(), 3)
    assert strategy.aggregate_train(1, [Reply(), Reply(), Reply()]) == ("aggregated", 3)


@pytest.mark.parametrize("replies", [[Reply(), Reply(), Reply(error=True)], [Reply(), Reply()]])
def test_a_failed_or_missing_client_stops_the_run(replies):
    strategy = require_all_clients(Strategy(), 3)
    with pytest.raises(RuntimeError, match="of 3 clients trained"):
        strategy.aggregate_train(4, replies)


BASE = ["--seed", "42", "--noise-multiplier", "0.5", "--clipping-norm", "5.0", "--num-clients", "4",
        "--model-module", "experiments.reproduce.eurosat_cnn:create_model", "--rounds", "2", "--local-epochs", "1"]


def test_runner_flag_turns_it_on_and_it_is_off_by_default():
    on = runner.build_run_config(runner._parser().parse_args([*BASE, "--require-all-clients"]))
    assert on["require-all-clients"] is True
    assert "require-all-clients" not in runner.build_run_config(runner._parser().parse_args(BASE))
    config = {"aggregation": "fedavg", "privacy": "global-dp", "num-clients": 4, "noise-multiplier": 0.5,
              "clipping-norm": 5.0, "aggregation-weighting": "equal", "seed": 42, "require-all-clients": True}
    strategy = make_run_strategy(config)
    with pytest.raises(RuntimeError):
        strategy.aggregate_train(1, [Reply(), Reply(), Reply()])
