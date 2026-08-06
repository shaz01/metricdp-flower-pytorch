"""Configuration tests for the unblocked PLAN.md experiment runner."""

from collections import Counter

from experiments.cia.datasets.paper import PAPER_CIA_CLIENT_COUNTS
from experiments.cia.scripts import planned_runs


def test_planned_matrix_matches_unblocked_plan() -> None:
    assert planned_runs.SEEDS == (42, 43, 44)
    assert planned_runs.CHECKPOINT_ROUNDS == tuple(range(1, 21))
    assert planned_runs.SHADOW_FRACTION == 0.10
    assert planned_runs.NOISE_STD_FRACTION == 0.20
    assert planned_runs.NOISE_MULTIPLIER == 0.01
    assert len(planned_runs.REPRODUCTION_MATRIX.list_combos(
        name_prefix="test", num_clients=4
    )) == 9

    names = [name for name, _clients, _matrix in planned_runs.CIA_GROUPS]
    assert names == [
        "alzheimer-in-remove",
        "alzheimer-out-remove",
        "fashion-in-remove",
        "fashion-out-remove",
        "cifar-in-remove",
        "cifar-out-remove",
    ]
    assert all("replace" not in name for name in names)


def test_suites_split_dataset_adjacency_groups() -> None:
    groups = [name for name, _clients, _matrix in planned_runs.CIA_GROUPS]
    assert [name for name in groups if name.startswith("alzheimer-")] == [
        "alzheimer-in-remove",
        "alzheimer-out-remove",
    ]
    assert [name for name in groups if name.startswith("fashion-")] == [
        "fashion-in-remove",
        "fashion-out-remove",
    ]
    assert [name for name in groups if name.startswith("cifar-")] == [
        "cifar-in-remove",
        "cifar-out-remove",
    ]


def test_parser_accepts_one_adjacency_group() -> None:
    args = planned_runs._parser().parse_args(
        ["--group", "fashion-in-remove"]
    )

    assert args.group == "fashion-in-remove"


def test_fashion_transfer_reuses_exact_table9_counts(monkeypatch) -> None:
    labels = [label for label in range(4) for _ in range(3_000)]
    captured = {}

    class FakeSplit:
        def __len__(self):
            return len(labels)

    module = planned_runs.FashionTable9DataModule()
    module._dataset = {"train": FakeSplit()}
    monkeypatch.setattr(planned_runs, "labels_from_records", lambda _split: labels)

    def fake_partition(_labels, counts, *, seed):
        captured.update(counts=counts, seed=seed)
        return [[0], [1], [2]]

    monkeypatch.setattr(planned_runs, "partition_by_class_counts", fake_partition)
    monkeypatch.setattr(
        planned_runs,
        "make_client_loaders",
        lambda *args, **kwargs: (object(), object()),
    )
    monkeypatch.setattr(planned_runs, "FashionMNISTDataset", lambda split: split)

    module.client_loaders(
        0,
        num_partitions=3,
        partition_mode="non-iid",
        batch_size=32,
        seed=43,
    )

    assert captured == {"counts": PAPER_CIA_CLIENT_COUNTS, "seed": 43}


def test_table9_counts_select_a_deterministic_fashion_subset() -> None:
    labels = [label for label in range(4) for _ in range(6_000)]

    seed_42_a = planned_runs.partition_by_class_counts(
        labels, PAPER_CIA_CLIENT_COUNTS, seed=42
    )
    seed_42_b = planned_runs.partition_by_class_counts(
        labels, PAPER_CIA_CLIENT_COUNTS, seed=42
    )
    seed_43 = planned_runs.partition_by_class_counts(
        labels, PAPER_CIA_CLIENT_COUNTS, seed=43
    )

    assert seed_42_a == seed_42_b
    assert seed_42_a != seed_43
    assert sum(map(len, seed_42_a)) == 5_120
    for partition, expected_counts in zip(
        seed_42_a, PAPER_CIA_CLIENT_COUNTS, strict=True
    ):
        assert tuple(
            Counter(labels[index] for index in partition)[label]
            for label in range(4)
        ) == expected_counts


def test_in_out_factories_share_canonical_target_partition() -> None:
    for in_factory, out_factory in (
        (planned_runs.create_alzheimer_in, planned_runs.create_alzheimer_out),
        (planned_runs.create_fashion_in, planned_runs.create_fashion_out),
        (planned_runs.create_cifar_in, planned_runs.create_cifar_out),
    ):
        in_view = in_factory({})
        out_view = out_factory({})
        assert in_view.canonical_num_partitions == 3
        assert out_view.canonical_num_partitions == 3
        assert in_view.active_partition_ids == (0, 1, 2)
        assert out_view.active_partition_ids == (0, 1)
