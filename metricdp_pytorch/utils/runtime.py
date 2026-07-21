"""Runtime configuration shared by Flower CLI and direct harness simulations."""

from __future__ import annotations

import json
import os
from typing import Any

from flwr.app import Context

RUN_CONFIG_ENV = "METRICDP_RUN_CONFIG"


def runtime_config(context: Context) -> dict[str, Any]:
    """Return Flower run config overlaid by the harness's isolated-run config."""
    config = dict(context.run_config)
    encoded = os.environ.get(RUN_CONFIG_ENV)
    if encoded:
        config.update(json.loads(encoded))
    return config
