"""CNN determinism and public-bundle construction."""

import numpy as np
import pytest

from experiments.stacked_head import bundle as bundle_module
from experiments.stacked_head.cnn import FEATURES, Net, embed, train_public
from experiments.stacked_head.tests.synthetic import labels_for, make_images
from metricdp_pytorch.stacked_head_strategy import head_scores


def test_state_dict_exposes_the_head_by_name():
    keys = set(Net().state_dict())
    assert {"head.weight", "head.bias"} <= keys and any(k.startswith("body.") for k in keys)


def test_training_is_deterministic_and_embedding_has_a_bias_column():
    labels = labels_for(8)
    images = make_images(labels, 0)
    a = train_public(images, labels, 10, 3e-3, 5)
    b = train_public(images, labels, 10, 3e-3, 5)
    assert all(np.array_equal(x.detach().numpy(), y.detach().numpy()) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    features = embed(a, images)
    assert features.shape == (32, FEATURES + 1) and np.all(features[:, -1] == 1)


@pytest.fixture(scope="module")
def prepared():
    public_labels, validation_labels, evaluation_labels = labels_for(8), labels_for(16), labels_for(24)
    return bundle_module.prepare_from_arrays(
        make_images(public_labels, 1), public_labels, make_images(validation_labels, 2), validation_labels,
        make_images(evaluation_labels, 3), evaluation_labels, seed_offset=32,
    )


def test_bundle_selects_the_best_validation_candidate_and_reports_its_control(prepared):
    candidates = prepared.meta["candidates"]
    assert len(candidates) == 6
    assert prepared.meta["base"]["validation_ce"] == min(c["validation_ce"] for c in candidates)
    ce, accuracy = head_scores(prepared.evaluation_features, prepared.evaluation_labels, prepared.theta0[None])
    assert prepared.meta["control"]["ce"] == pytest.approx(float(ce[0]))
    assert prepared.meta["control"]["accuracy"] == pytest.approx(float(accuracy[0]))


def test_bundle_geometry_is_public_orthonormal_and_shaped(prepared):
    d = prepared.theta0.shape[1]
    assert prepared.theta0.shape == (3, FEATURES + 1) and prepared.class_gradients.shape == (4, 3, d)
    assert prepared.basis.shape == (3 * d, bundle_module.MAX_DIMENSION)
    assert np.allclose(prepared.basis.T @ prepared.basis, np.eye(bundle_module.MAX_DIMENSION), atol=1e-8)


def test_bundle_round_trips_through_disk(prepared, tmp_path):
    path = tmp_path / "bundle.npz"
    bundle_module.save_bundle(prepared, path)
    loaded = bundle_module.load_bundle(path)
    assert loaded.meta == prepared.meta and set(loaded.state) == set(prepared.state)
    for name in ("theta0", "class_gradients", "basis", "validation_features", "evaluation_features"):
        assert np.array_equal(getattr(loaded, name), getattr(prepared, name))
    assert all(np.array_equal(loaded.state[k], prepared.state[k]) for k in prepared.state)
