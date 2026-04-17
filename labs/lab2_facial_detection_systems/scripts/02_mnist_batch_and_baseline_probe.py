"""Lab 2 MNIST probe: inspect batches and the fully connected baseline.

The official lab trains a fully connected MNIST classifier before moving to a
CNN. I am keeping this script one step earlier than training: it loads one batch,
checks the tensor shapes, defines the baseline model, and computes one forward
pass plus cross-entropy loss.

Examples:

    python labs/lab2_facial_detection_systems/scripts/02_mnist_batch_and_baseline_probe.py
    python labs/lab2_facial_detection_systems/scripts/02_mnist_batch_and_baseline_probe.py --source mnist --download

The default synthetic mode keeps the script runnable without downloading data.
Use `--source mnist --download` on the Ubuntu box when I am ready to inspect the
real dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    from torchvision import datasets, transforms
except ImportError:  # pragma: no cover - helpful when dependencies are missing.
    datasets = None
    transforms = None


class FullyConnectedMNISTBaseline(nn.Module):
    """Two-layer dense baseline from the first MNIST section of Lab 2."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.flatten(images)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


def build_synthetic_dataset(num_samples: int) -> TensorDataset:
    """Create MNIST-shaped fake data for offline shape checks."""

    generator = torch.Generator().manual_seed(17)
    images = torch.rand(num_samples, 1, 28, 28, generator=generator)
    labels = torch.randint(0, 10, (num_samples,), generator=generator)
    return TensorDataset(images, labels)


def build_mnist_dataset(data_dir: Path, download: bool) -> Dataset:
    """Load the real MNIST training set through torchvision."""

    if datasets is None or transforms is None:
        raise RuntimeError(
            "torchvision is required for --source mnist. "
            "Install the repo requirements first."
        )

    transform = transforms.Compose([transforms.ToTensor()])
    return datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=download,
        transform=transform,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def describe_batch(images: torch.Tensor, labels: torch.Tensor) -> None:
    print("Batch")
    print(f"  images shape: {tuple(images.shape)}")
    print(f"  images dtype: {images.dtype}")
    print(f"  image range : [{images.min().item():.4f}, {images.max().item():.4f}]")
    print(f"  labels shape: {tuple(labels.shape)}")
    print(f"  labels dtype: {labels.dtype}")
    print(f"  label counts: {torch.bincount(labels, minlength=10).tolist()}")


def inspect_forward_pass(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> None:
    loss_function = nn.CrossEntropyLoss()

    with torch.no_grad():
        logits = model(images)
        loss = loss_function(logits, labels)
        predicted_digits = torch.argmax(logits, dim=1)

    print("\nBaseline forward pass")
    print(f"  model: {model.__class__.__name__}")
    print(f"  trainable parameters: {count_trainable_parameters(model):,}")
    print(f"  logits shape: {tuple(logits.shape)}")
    print(f"  loss on this untrained batch: {loss.item():.4f}")
    print(f"  first 12 predictions: {predicted_digits[:12].tolist()}")
    print("\nNo backward pass or optimizer step was run.")


def parse_args() -> argparse.Namespace:
    lab_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Inspect MNIST batch mechanics and a dense baseline forward pass."
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "mnist"],
        default="synthetic",
        help="Use offline MNIST-shaped fake data or the real torchvision MNIST dataset.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow torchvision to download MNIST when --source mnist is selected.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=lab_dir / "data",
        help="Directory for torchvision MNIST files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used for the single inspected DataLoader batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)

    if args.source == "mnist":
        dataset = build_mnist_dataset(args.data_dir, args.download)
    else:
        dataset = build_synthetic_dataset(num_samples=max(args.batch_size, 64))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    images, labels = next(iter(loader))

    describe_batch(images, labels)
    model = FullyConnectedMNISTBaseline()
    inspect_forward_pass(model, images, labels)

    print(
        "\nNext real Lab 2 step: train this baseline for a few epochs, then compare "
        "it against the CNN from the shape probe."
    )


if __name__ == "__main__":
    main()
