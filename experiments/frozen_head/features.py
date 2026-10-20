"""Frozen public features: ResNet-18 (ImageNet-1k weights) on EuroSAT, extracted once and cached.

The extractor is public: torchvision's ``IMAGENET1K_V1`` weights, frozen, in eval mode. Images
are upsampled from 64x64 to 224x224 (bilinear) and normalized with the ImageNet statistics; no
augmentation. The 512-dimensional global-average-pool output is the feature. The cache lives
outside the repository (``$FROZEN_HEAD_CACHE``, default ``~/.cache/metricdp_frozen_head``) and is
keyed by split, so feature row ``i`` is record ``i`` of that Hugging Face split.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from experiments.reproduce.dataset.eurosat import load_eurosat_dataset

FEATURE_DIM = 512
INPUT_SIZE = 224
EXTRACTOR = "resnet18-imagenet1k-v1"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
PREPROCESS = Compose([
    Resize((INPUT_SIZE, INPUT_SIZE), interpolation=Image.Resampling.BILINEAR),
    ToTensor(),
    Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


def cache_dir() -> Path:
    return Path(os.environ.get("FROZEN_HEAD_CACHE", Path.home() / ".cache" / "metricdp_frozen_head"))


def build_extractor() -> torch.nn.Module:
    """ResNet-18 with the classifier removed; frozen and in eval mode."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def extract(images: list[Image.Image], model: torch.nn.Module, *, batch_size: int = 256,
            device: str = "cpu") -> np.ndarray:
    """Features of PIL images as float32 ``(len(images), 512)``."""
    model = model.to(device)
    out = np.empty((len(images), FEATURE_DIM), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.stack([PREPROCESS(image.convert("RGB")) for image in images[start:start + batch_size]])
            out[start:start + len(batch)] = model(batch.to(device)).cpu().numpy()
    return out


def load_split_features(split: str, *, device: str = "cpu", batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """``(features, labels)`` of a EuroSAT split, from the cache or extracted and cached."""
    path = cache_dir() / f"eurosat_{split}_{EXTRACTOR}.npz"
    if path.exists():
        with np.load(path) as stored:
            return stored["features"], stored["labels"]
    records = load_eurosat_dataset()[split]
    labels = np.asarray(records["label"], dtype=np.int64)
    features = extract(list(records["image"]), build_extractor(), batch_size=batch_size, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, features=features, labels=labels)
    temporary.replace(path)
    return features, labels


def fingerprint(features: np.ndarray) -> str:
    """Short content hash recorded in result files, so a changed cache is detectable."""
    return hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()[:16]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    for name in ("train", "test"):
        feats, labs = load_split_features(name, device=args.device)
        print(name, feats.shape, np.bincount(labs).tolist(), fingerprint(feats))
