"""Shared fixtures for CIA-attack tests."""

from __future__ import annotations

import pytest
from datasets import DatasetDict

from experiments.reproduce.dataset.alzheimer import load_alzheimer_dataset


@pytest.fixture(scope="session")
def alzheimer_dataset() -> DatasetDict:
    """Download once per test session, then reuse the Hugging Face cache."""
    return load_alzheimer_dataset()
