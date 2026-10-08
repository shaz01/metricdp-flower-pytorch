from results.cia_frontier.dataset_vs_model.frontier import window_accuracy


def test_window_mean_min_max_and_final():
    metrics = {str(r): {"accuracy": r / 100} for r in range(1, 21)}
    acc = window_accuracy(metrics, 20, window=10)
    assert acc["accuracy_window"] == 10
    assert abs(acc["accuracy"] - 0.155) < 1e-12
    assert (acc["accuracy_min"], acc["accuracy_max"], acc["accuracy_final"]) == (0.11, 0.20, 0.20)


def test_window_truncates_at_round_one():
    metrics = {str(r): {"accuracy": 0.5} for r in range(1, 4)}
    assert window_accuracy(metrics, 3, window=10)["accuracy_window"] == 3
