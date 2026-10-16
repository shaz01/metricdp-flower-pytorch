"""Tests for residual clipping around the previous released update (WP6). Synthetic, no training."""

import math

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from metricdp_pytorch.residual_clipping_strategy import ResidualClippingServerSide, residual_round_account
from metricdp_pytorch.strategy_factory import UNIT_WEIGHT_KEY

TINY = 1e-12  # noise multiplier small enough that the noise is negligible


def model(*arrays):
    return ArrayRecord({f"layer-{i}": Array(np.asarray(a, dtype=np.float64)) for i, a in enumerate(arrays)})


def replies(models):
    out = []
    for node_id, record in enumerate(models, start=1):
        request = Message(content=RecordDict(), message_type="train", dst_node_id=node_id)
        content = RecordDict({"arrays": record, "metrics": MetricRecord(
            {"num-examples": 5 * node_id, UNIT_WEIGHT_KEY: 1, "client-id": node_id - 1})})
        out.append(Message(content=content, reply_to=request))
    return out


def make(num_slots=4, **kwargs):
    settings = dict(noise_multiplier=TINY, num_slots=num_slots, seed=0)
    settings.update(kwargs)
    if "rho" not in settings and "target_quantile" not in settings:
        settings["rho"] = 1.0
    return ResidualClippingServerSide(FedAvg(weighted_by_key=UNIT_WEIGHT_KEY), **settings)


def values(arrays):
    return np.concatenate([a.ravel() for a in arrays.to_numpy_ndarrays()])


def test_center_zero_is_plain_clipping_with_slot_denominator():
    s = make(num_slots=4, center="zero", rho=5.0)
    s.current_arrays = model([1.0, 1.0])
    arrays, metrics = s.aggregate_train(1, replies([model([4.0, 5.0]), model([1.0, 3.0]), model([11.0, 1.0])]))
    # updates (3, 4) norm 5, (0, 2) norm 2, (10, 0) clipped to (5, 0); sum (8, 6) over 4 slots.
    assert values(arrays) == pytest.approx([1.0 + 2.0, 1.0 + 1.5], abs=1e-9)
    assert metrics["residual-norms"] == pytest.approx([5.0, 2.0, 10.0])
    assert metrics["residual-fraction-clipped"] == pytest.approx(1 / 3)


def test_previous_center_uses_the_released_update():
    s = make(num_slots=2, center="previous", rho=100.0)
    s.current_arrays = model([0.0, 0.0])
    first, _ = s.aggregate_train(1, replies([model([2.0, 0.0]), model([0.0, 2.0])]))
    assert values(first) == pytest.approx([1.0, 1.0])
    s.current_arrays = first
    second, metrics = s.aggregate_train(2, replies([model([4.0, 1.0]), model([1.0, 4.0])]))
    # c_2 = (1, 1); updates from (1, 1): (3, 0) and (0, 3); residuals (2, -1) and (-1, 2).
    assert metrics["residual-center-norm"] == pytest.approx(math.sqrt(2))
    assert metrics["residual-norms"] == pytest.approx([math.sqrt(5), math.sqrt(5)])
    assert values(second) == pytest.approx([1 + 1 + 0.5, 1 + 1 + 0.5])


def test_add_remove_sensitivity_is_rho_over_slots():
    rng = np.random.default_rng(3)
    others = [model(rng.normal(size=3), rng.normal(size=2)) for _ in range(3)]
    target = model(rng.normal(size=3) * 50, rng.normal(size=2) * 50)
    for center in ("zero", "previous"):
        released = []
        for present in (others + [target], others):
            s = make(num_slots=4, center=center, rho=0.7, noise_multiplier=0.3, seed=11)
            s.current_arrays = model(np.zeros(3), np.zeros(2))
            first, _ = s.aggregate_train(1, replies(others))  # same history in both worlds
            s.current_arrays = first
            arrays, _ = s.aggregate_train(2, replies(present))
            released.append(values(arrays))
        # Same noise draws (same seed, same order); the target's residual is clipped to rho = 0.7.
        assert np.linalg.norm(released[0] - released[1]) == pytest.approx(0.7 / 4, rel=1e-9)


def test_noise_scale_and_accounting():
    s = make(num_slots=48, rho=2.0, noise_multiplier=0.08, seed=5)
    s.current_arrays = model(np.zeros(200_000))
    arrays, metrics = s.aggregate_train(1, replies([model(np.zeros(200_000))]))
    assert metrics["residual-noise-stdv"] == pytest.approx(0.08 * 2.0 / 48)
    assert np.std(values(arrays)) == pytest.approx(0.08 * 2.0 / 48, rel=0.01)
    assert metrics["residual-mu-per-round"] == pytest.approx(1 / 0.08)
    assert residual_round_account(0.08, 0.5) == pytest.approx(math.hypot(12.5, 2.0))


def test_private_rho_tracks_the_target_quantile():
    norms = [0.5, 1.0, 2.0, 3.0, 4.0]
    s = make(num_slots=5, center="zero", rho=None, target_quantile=0.5, initial_rho=0.1, minimum_rho=0.01,
             maximum_rho=100.0, bit_noise_std=1e-9, learning_rate=0.3)
    for round_number in range(1, 200):
        s.current_arrays = model([0.0])
        s.aggregate_train(round_number, replies([model([n]) for n in norms]))
    assert 1.0 < s.rho <= 3.0


def test_argument_checks():
    with pytest.raises(ValueError):
        ResidualClippingServerSide(FedAvg(), noise_multiplier=1.0, num_slots=3, rho=1.0)
    with pytest.raises(ValueError):
        make(rho=1.0, target_quantile=0.5)
    with pytest.raises(ValueError):
        make(rho=None, target_quantile=0.5)
    with pytest.raises(ValueError):
        make(center="mean")


def _runner_args(*extra):
    from experiments.reproduce.runner import _parser

    return _parser().parse_args(["--seed", "42", "--noise-multiplier", "0.08", "--clipping-norm", "5", "--rounds",
                                 "2", "--local-epochs", "1", "--num-clients", "3", "--model-module",
                                 "experiments.reproduce.paper_cnn:create_model", *extra])


def test_runner_and_server_build_the_residual_strategy(tmp_path):
    from experiments.reproduce.runner import build_run_config
    from experiments.reproduce.server import residual_clipping_strategy

    default = build_run_config(_runner_args("--output-dir", str(tmp_path)))
    assert not {"num-slots", "residual-center", "residual-rho"} & set(default)
    config = build_run_config(_runner_args("--output-dir", str(tmp_path), "--privacy", "residual-clipping",
                                           "--aggregation-weighting", "equal", "--num-slots", "4",
                                           "--residual-quantile", "0.5", "--residual-rho-min", "0.05",
                                           "--residual-rho-max", "20", "--residual-bit-noise-std", "0.2"))
    strategy = residual_clipping_strategy(config)
    assert (strategy.num_slots, strategy.center, strategy.rho, strategy.adaptive) == (4, "previous", 5.0, True)
    fixed = residual_clipping_strategy(build_run_config(_runner_args(
        "--output-dir", str(tmp_path), "--privacy", "residual-clipping", "--aggregation-weighting", "equal",
        "--residual-center", "zero", "--residual-rho", "5")))
    assert (fixed.num_slots, fixed.center, fixed.rho, fixed.adaptive) == (3, "zero", 5.0, False)
    for bad in (["--privacy", "residual-clipping", "--residual-rho", "5"],  # not equal weights
                ["--residual-rho", "5"],  # flag without the mode
                ["--privacy", "residual-clipping", "--aggregation-weighting", "equal"]):  # no rho
        with pytest.raises(ValueError):
            build_run_config(_runner_args("--output-dir", str(tmp_path), *bad))
