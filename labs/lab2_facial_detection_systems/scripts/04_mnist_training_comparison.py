"""Lab 2 Part 1: train and compare dense vs convolutional MNIST models.

This is the point where my Lab 2 work moves beyond shape/evaluation probes. The
official PyTorch lab trains a two-layer fully connected baseline, then trains a
small CNN and compares held-out accuracy. This script follows that flow while
keeping an offline synthetic default so the repo is still runnable without
downloading MNIST.

Examples:

    python labs/lab2_facial_detection_systems/scripts/04_mnist_training_comparison.py
    python labs/lab2_facial_detection_systems/scripts/04_mnist_training_comparison.py --epochs 3
    python labs/lab2_facial_detection_systems/scripts/04_mnist_training_comparison.py --source mnist --download
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

try:
    from torchvision import datasets, transforms
except Exception:  # pragma: no cover - synthetic mode should still work.
    datasets = None
    transforms = None


NUM_CLASSES = 10

SEGMENTS_BY_DIGIT = {
    0: ("top", "upper_left", "upper_right", "lower_left", "lower_right", "bottom"),
    1: ("upper_right", "lower_right"),
    2: ("top", "upper_right", "middle", "lower_left", "bottom"),
    3: ("top", "upper_right", "middle", "lower_right", "bottom"),
    4: ("upper_left", "upper_right", "middle", "lower_right"),
    5: ("top", "upper_left", "middle", "lower_right", "bottom"),
    6: ("top", "upper_left", "middle", "lower_left", "lower_right", "bottom"),
    7: ("top", "upper_right", "lower_right"),
    8: (
        "top",
        "upper_left",
        "upper_right",
        "middle",
        "lower_left",
        "lower_right",
        "bottom",
    ),
    9: ("top", "upper_left", "upper_right", "middle", "lower_right", "bottom"),
}


class FullyConnectedMNISTBaseline(nn.Module):
    """Two-layer dense baseline from the first MNIST section of Lab 2."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.flatten(images)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


class MNISTConvNet(nn.Module):
    """Small CNN matching the Lab 2 architecture at a reviewable scale."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(36 * 5 * 5, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    average_loss: float
    accuracy: float


@dataclass(frozen=True)
class EvaluationResult:
    average_loss: float
    accuracy: float
    confusion: torch.Tensor


def draw_segment_digit(label: int, generator: torch.Generator) -> torch.Tensor:
    """Create a small seven-segment MNIST-shaped image for offline training."""

    image = torch.zeros(1, 28, 28)

    segment_slices = {
        "top": (slice(4, 7), slice(8, 21)),
        "upper_left": (slice(6, 15), slice(6, 9)),
        "upper_right": (slice(6, 15), slice(19, 22)),
        "middle": (slice(13, 16), slice(8, 21)),
        "lower_left": (slice(15, 24), slice(6, 9)),
        "lower_right": (slice(15, 24), slice(19, 22)),
        "bottom": (slice(22, 25), slice(8, 21)),
    }

    for segment in SEGMENTS_BY_DIGIT[label]:
        rows, cols = segment_slices[segment]
        image[:, rows, cols] = 1.0

    shift_y = int(torch.randint(-2, 3, (1,), generator=generator).item())
    shift_x = int(torch.randint(-2, 3, (1,), generator=generator).item())
    shifted = torch.zeros_like(image)

    src_y_start = max(0, -shift_y)
    src_y_end = min(28, 28 - shift_y)
    dst_y_start = max(0, shift_y)
    dst_y_end = min(28, 28 + shift_y)

    src_x_start = max(0, -shift_x)
    src_x_end = min(28, 28 - shift_x)
    dst_x_start = max(0, shift_x)
    dst_x_end = min(28, 28 + shift_x)

    shifted[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image[
        :, src_y_start:src_y_end, src_x_start:src_x_end
    ]

    noise = 0.10 * torch.randn((1, 28, 28), generator=generator)
    background = 0.03 * torch.rand((1, 28, 28), generator=generator)
    return torch.clamp(shifted + noise + background, min=0.0, max=1.0)


def build_synthetic_digit_dataset(num_samples: int, seed: int) -> TensorDataset:
    """Build a learnable MNIST-shaped dataset without network access."""

    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(num_samples) % NUM_CLASSES
    labels = labels[torch.randperm(num_samples, generator=generator)]
    images = torch.stack(
        [draw_segment_digit(int(label.item()), generator) for label in labels],
        dim=0,
    )
    return TensorDataset(images, labels.long())


def build_mnist_dataset(
    data_dir: Path,
    download: bool,
    train: bool,
    max_examples: int,
    seed: int,
) -> Dataset:
    """Load MNIST through torchvision and optionally keep a deterministic subset."""

    if datasets is None or transforms is None:
        raise RuntimeError(
            "torchvision is required for --source mnist. "
            "Install the repo requirements before using the real dataset."
        )

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(
        root=str(data_dir),
        train=train,
        download=download,
        transform=transform,
    )

    if max_examples >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_examples]
    return Subset(dataset, indices.tolist())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def update_confusion_matrix(
    confusion: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    labels = labels.detach().cpu()
    predictions = predictions.detach().cpu()
    flat_indices = labels * NUM_CLASSES + predictions
    confusion += torch.bincount(
        flat_indices,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_function(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total += batch_size

    return EpochMetrics(
        epoch=epoch,
        average_loss=total_loss / total,
        accuracy=correct / total,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> EvaluationResult:
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_function(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size
            update_confusion_matrix(confusion, labels, predictions)

    return EvaluationResult(
        average_loss=total_loss / total,
        accuracy=correct / total,
        confusion=confusion,
    )


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    if args.source == "mnist":
        train_dataset = build_mnist_dataset(
            data_dir=args.data_dir,
            download=args.download,
            train=True,
            max_examples=args.train_size,
            seed=args.seed,
        )
        test_dataset = build_mnist_dataset(
            data_dir=args.data_dir,
            download=args.download,
            train=False,
            max_examples=args.test_size,
            seed=args.seed + 1,
        )
    else:
        train_dataset = build_synthetic_digit_dataset(args.train_size, args.seed)
        test_dataset = build_synthetic_digit_dataset(args.test_size, args.seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def parse_args() -> argparse.Namespace:
    lab_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Train the Lab 2 MNIST dense baseline and CNN."
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "mnist"],
        default="synthetic",
        help="Use offline seven-segment data or real torchvision MNIST.",
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
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to use. CPU is the default so smoke checks are portable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    train_loader, test_loader = make_loaders(args)
    loss_function = nn.CrossEntropyLoss()

    print("Lab 2 Part 1 MNIST training comparison")
    print(f"source: {args.source}")
    print(f"device: {device}")
    print(f"train examples: {len(train_loader.dataset)}")
    print(f"test examples : {len(test_loader.dataset)}")
    print(f"epochs: {args.epochs}")

    model_factories = [
        ("fully connected baseline", FullyConnectedMNISTBaseline),
        ("cnn", MNISTConvNet),
    ]

    for model_label, model_factory in model_factories:
        model = model_factory().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        print(f"\n{model_label}")
        print(f"  trainable parameters: {count_trainable_parameters(model):,}")

        for epoch in range(1, args.epochs + 1):
            metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                loss_function=loss_function,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
            )
            print(
                f"  epoch {metrics.epoch}: "
                f"train loss={metrics.average_loss:.4f}, "
                f"train accuracy={metrics.accuracy:.4f}"
            )

        result = evaluate(model, test_loader, loss_function, device)
        print(
            "  test: "
            f"loss={result.average_loss:.4f}, "
            f"accuracy={result.accuracy:.4f}"
        )
        print("  confusion matrix rows=true, columns=predicted:")
        print(result.confusion)

    if args.source == "synthetic":
        print(
            "\nSynthetic mode is a local training smoke test. "
            "Use --source mnist --download for the real Lab 2 MNIST dataset."
        )


if __name__ == "__main__":
    main()
