"""Pluggable model factory used by Flower clients and servers."""

from __future__ import annotations

import importlib
from torch import nn


def load_model(factory_path: str) -> nn.Module:
    """Construct a model from a ``package.module:factory`` path."""
    module_name, separator, factory_name = factory_path.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("model-module must use the format 'package.module:factory'.")
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise TypeError(f"Model factory {factory_path!r} is not callable.")
    model = factory()
    if not isinstance(model, nn.Module):
        raise TypeError(f"Factory {factory_path!r} did not return a torch.nn.Module.")
    return model
