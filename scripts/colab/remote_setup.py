"""Set up a credential-free metricdp-pytorch source snapshot on Colab."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

ARCHIVE = Path("/content/metricdp-source.tar.gz")
PROJECT_ROOT = Path("/content/metricdp-pytorch")


def main() -> None:
    if PROJECT_ROOT.exists():
        shutil.rmtree(PROJECT_ROOT)
    PROJECT_ROOT.mkdir(parents=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        archive.extractall(PROJECT_ROOT, filter="data")

    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    dependencies = [
        dependency
        for dependency in project["project"]["dependencies"]
        if not dependency.lower().startswith(("torch", "ipykernel"))
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *dependencies],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run([sys.executable, "-c", "import torch, flwr, datasets"], check=True)
    print(f"Colab environment ready at {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
