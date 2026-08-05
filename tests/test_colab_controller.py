"""Tests for local Colab controller argument handling."""

from scripts.colab.run_experiment import _forwarded_module_args


def test_module_argument_delimiter_is_not_forwarded() -> None:
    assert _forwarded_module_args(["--", "--suite", "fashion"]) == [
        "--suite",
        "fashion",
    ]


def test_module_arguments_without_delimiter_are_unchanged() -> None:
    assert _forwarded_module_args(["value"]) == ["value"]
