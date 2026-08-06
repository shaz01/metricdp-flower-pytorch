from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.cia.datasets.shadow import ShadowDataModule
from experiments.reproduce.matrix import Combo
from experiments.reproduce.paper_loss import sparse_categorical_cross_entropy
from metricdp_pytorch.model_module import load_model


def _calculate_loss(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
) -> float:
    """Return the mean sparse categorical cross-entropy over a loader."""
    model.to(device)
    model.eval()
    total_loss = 0.0
    examples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            probabilities = model(images)
            total_loss += sparse_categorical_cross_entropy(
                probabilities, labels, reduction="sum"
            ).item()
            examples += len(labels)
    if examples == 0:
        raise ValueError("The loss loader must not be empty.")
    return total_loss / examples


def eval_model(
    model_path: Path,
    *,
    clean_data_module: ShadowDataModule,
    noisy_data_module: ShadowDataModule,
    device: torch.device,
    combo: Combo,
) -> tuple[float, float, float, int]:
    """Return global, clean-shadow, and noisy-shadow losses for a checkpoint."""
    model = load_model(combo.model_module)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )

    _validation_loader, test_loader = clean_data_module.server_loaders(
        batch_size=combo.hyperparams.batch_size, seed=combo.seed
    )
    clean_shadow_loader = clean_data_module.target_shadow_loader(
        batch_size=combo.hyperparams.batch_size, seed=combo.seed
    )
    noisy_shadow_loader = noisy_data_module.target_shadow_loader(
        batch_size=combo.hyperparams.batch_size, seed=combo.seed
    )
    clean_shadow_size = len(clean_shadow_loader.dataset)
    noisy_shadow_size = len(noisy_shadow_loader.dataset)
    if clean_shadow_size != noisy_shadow_size:
        raise ValueError(
            "Clean and noisy shadow datasets must contain the same examples."
        )

    return (
        _calculate_loss(model, test_loader, device),
        _calculate_loss(model, clean_shadow_loader, device),
        _calculate_loss(model, noisy_shadow_loader, device),
        clean_shadow_size,
    )
