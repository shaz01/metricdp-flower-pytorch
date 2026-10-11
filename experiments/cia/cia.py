from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.cia.datasets.shadow import ShadowDataModule
from experiments.reproduce.matrix import Combo
from experiments.reproduce.paper_loss import sparse_categorical_cross_entropy
from metricdp_pytorch.model_module import load_model


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Copy a batch to ``device``; asynchronous copies only on CUDA.

    With torch 2.10, asynchronous host-to-MPS copies of DataLoader batches gave
    wrong, run-to-run varying losses (round-0 Alzheimer shadow loss 0.70-0.92
    instead of the CPU's 1.4408); blocking copies matched the CPU exactly.
    """
    return tensor.to(device, non_blocking=torch.device(device).type == "cuda")


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
            images = _to_device(images, device)
            labels = _to_device(labels, device)
            probabilities = model(images)
            total_loss += sparse_categorical_cross_entropy(
                probabilities, labels, reduction="sum"
            ).item()
            examples += len(labels)
    if examples == 0:
        raise ValueError("The loss loader must not be empty.")
    return total_loss / examples


def shadow_loss_and_gradient_norm(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
) -> tuple[float, float]:
    """Return the mean loss over a loader and the L2 norm of its parameter gradient.

    The model stays in eval mode, as in ``_calculate_loss``, so the loss matches
    the attack's loss and the gradient is that of the same deterministic function.
    """
    model.to(device)
    model.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model.zero_grad(set_to_none=True)
    total_loss = 0.0
    examples = 0
    for images, labels in loader:
        images = _to_device(images, device)
        labels = _to_device(labels, device)
        loss = sparse_categorical_cross_entropy(model(images), labels, reduction="sum")
        loss.backward()
        total_loss += loss.item()
        examples += len(labels)
    if examples == 0:
        raise ValueError("The loss loader must not be empty.")
    squared = sum(
        # float64 on the CPU: MPS has no float64.
        float(torch.sum(parameter.grad.detach().cpu().double() ** 2))
        for parameter in parameters
        if parameter.grad is not None
    )
    model.zero_grad(set_to_none=True)
    return total_loss / examples, squared ** 0.5 / examples


def _load_checkpoint(model_path: Path, combo: Combo) -> torch.nn.Module:
    model = load_model(combo.model_module)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    return model


def _evaluation_loaders(
        clean_data_module: ShadowDataModule,
        noisy_data_module: ShadowDataModule,
        combo: Combo,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    _validation_loader, test_loader = clean_data_module.server_loaders(
        batch_size=combo.hyperparams.batch_size, seed=combo.seed
    )
    clean_shadow_loader = clean_data_module.target_shadow_loader(
        batch_size=combo.hyperparams.batch_size, seed=combo.seed
    )
    noisy_shadow_loader = noisy_data_module.target_shadow_loader(
        batch_size=combo.hyperparams.batch_size, seed=combo.seed
    )
    if len(clean_shadow_loader.dataset) != len(noisy_shadow_loader.dataset):
        raise ValueError(
            "Clean and noisy shadow datasets must contain the same examples."
        )
    return test_loader, clean_shadow_loader, noisy_shadow_loader


def eval_model_with_gradients(
    model_path: Path,
    *,
    clean_data_module: ShadowDataModule,
    noisy_data_module: ShadowDataModule,
    device: torch.device,
    combo: Combo,
) -> dict[str, float | int]:
    """``eval_model``'s losses plus the shadow-loss gradient norm of each view."""
    model = _load_checkpoint(model_path, combo)
    test_loader, clean_loader, noisy_loader = _evaluation_loaders(
        clean_data_module, noisy_data_module, combo
    )
    clean_loss, clean_grad_norm = shadow_loss_and_gradient_norm(model, clean_loader, device)
    noisy_loss, noisy_grad_norm = shadow_loss_and_gradient_norm(model, noisy_loader, device)
    return {
        "aggregate_loss": _calculate_loss(model, test_loader, device),
        "clean_loss": clean_loss,
        "noisy_loss": noisy_loss,
        "shadow_size": len(clean_loader.dataset),
        "clean_grad_norm": clean_grad_norm,
        "noisy_grad_norm": noisy_grad_norm,
    }


def eval_model(
    model_path: Path,
    *,
    clean_data_module: ShadowDataModule,
    noisy_data_module: ShadowDataModule,
    device: torch.device,
    combo: Combo,
) -> tuple[float, float, float, int]:
    """Return global, clean-shadow, and noisy-shadow losses for a checkpoint."""
    model = _load_checkpoint(model_path, combo)
    test_loader, clean_shadow_loader, noisy_shadow_loader = _evaluation_loaders(
        clean_data_module, noisy_data_module, combo
    )
    return (
        _calculate_loss(model, test_loader, device),
        _calculate_loss(model, clean_shadow_loader, device),
        _calculate_loss(model, noisy_shadow_loader, device),
        len(clean_shadow_loader.dataset),
    )
