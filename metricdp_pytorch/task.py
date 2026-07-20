"""PyTorch model, CIFAR-10 partitioning, training, and evaluation."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Small CNN adapted from the PyTorch CIFAR-10 tutorial."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


DATA_ROOT = Path("data")
TRANSFORM = Compose(
    [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def _partition_indices(
    dataset_size: int, partition_id: int, num_partitions: int
) -> list[int]:
    """Return one deterministic IID partition of shuffled dataset indices."""
    if not 0 <= partition_id < num_partitions:
        raise ValueError("partition_id must be in [0, num_partitions).")
    generator = torch.Generator().manual_seed(42)
    shuffled = torch.randperm(dataset_size, generator=generator).tolist()
    return shuffled[partition_id::num_partitions]


def load_data(
    partition_id: int, num_partitions: int, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    """Load one deterministic IID CIFAR-10 partition and split it 80/20."""
    dataset = CIFAR10(root=DATA_ROOT, train=True, download=True, transform=TRANSFORM)
    indices = _partition_indices(len(dataset), partition_id, num_partitions)
    split = int(0.8 * len(indices))
    trainset = Subset(dataset, indices[:split])
    valset = Subset(dataset, indices[split:])
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True),
        DataLoader(valset, batch_size=batch_size),
    )


def load_centralized_dataset() -> DataLoader:
    """Load the full CIFAR-10 test set."""
    testset = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=TRANSFORM)
    return DataLoader(testset, batch_size=128)


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    """Train the model and return average minibatch loss."""
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return running_loss / (epochs * len(trainloader))


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Evaluate the model and return average loss and accuracy."""
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    return loss / len(testloader), correct / len(testloader.dataset)
