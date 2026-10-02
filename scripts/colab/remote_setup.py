"""Set up a credential-free metricdp-pytorch source snapshot on Colab.

The Colab image's preinstalled torch is not used: from 2026-09-23 ~16:50 UTC
every run on the image's torch 2.11.0+cu130 died in round 1 with SIGFPE inside
the CUDA ``bernoulli_``/``dropout2d`` launch, while the identical workload ran
cleanly on the lock's torch 2.10.0 (+cu128 wheel) on the same VM type. The
runtime is therefore pinned explicitly, verified after install, and recorded
next to every run (``metricdp-colab-runtime.json`` / ``-pip-freeze.txt``).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

CONTENT = Path(os.environ.get("METRICDP_COLAB_CONTENT", "/content"))
ARCHIVE = CONTENT / "metricdp-source.tar.gz"
PROJECT_ROOT = CONTENT / "metricdp-pytorch"
RUNTIME_PATH = CONTENT / "metricdp-colab-runtime.json"
FREEZE_PATH = CONTENT / "metricdp-colab-pip-freeze.txt"

# torch/torchvision: uv.lock versions. The others: the versions every
# successful 2026-09-22/23 Colab attack run resolved to (flwr 1.38.0 was the
# newest release satisfying pyproject's range for all of them).
PINNED = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "flwr": "1.38.0",
    "ray": "2.55.1",
    "numpy": "2.1.3",
    "datasets": "4.8.5",
    "scikit-learn": "1.9.1",
}
_EXTRAS = {"flwr": "[simulation]"}

_PROBE = r"""
import json, importlib.metadata as md, torch
info = {name: md.version(name) for name in NAMES}
info["torch_cuda"] = torch.version.cuda
info["cudnn"] = torch.backends.cudnn.version()
info["cuda_available"] = torch.cuda.is_available()
info["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print(json.dumps(info))
"""


def requirements() -> list[str]:
    return [f"{name}{_EXTRAS.get(name, '')}=={version}" for name, version in PINNED.items()]


def mismatches(installed: dict[str, str]) -> dict[str, tuple[str, str | None]]:
    """Pinned packages whose installed version (local tag ignored) differs."""
    out = {}
    for name, wanted in PINNED.items():
        found = installed.get(name)
        if found is None or found.split("+")[0] != wanted:
            out[name] = (wanted, found)
    return out


def _driver() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def main() -> None:
    if PROJECT_ROOT.exists():
        shutil.rmtree(PROJECT_ROOT)
    PROJECT_ROOT.mkdir(parents=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        archive.extractall(PROJECT_ROOT, filter="data")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *requirements()], check=True
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    probe = _PROBE.replace("NAMES", repr(list(PINNED)))
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    info = json.loads(result.stdout.strip().splitlines()[-1])
    wrong = mismatches(info)
    if wrong:
        raise SystemExit(f"Pinned runtime not installed: {wrong}")
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    FREEZE_PATH.write_text(freeze, encoding="utf-8")
    info |= {
        "pinned": PINNED,
        "driver": _driver(),
        "python": sys.version,
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
    }
    RUNTIME_PATH.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Colab runtime: {json.dumps(info)}")
    print(f"Colab environment ready at {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
