"""Lab 2 MNIST probe: evaluation plumbing before real training.

The official lab asks for training a fully connected baseline, evaluating it on
held-out MNIST data, and then comparing it against a CNN. I am stopping one step
before that here: this script checks that the evaluation loop, loss, accuracy,
and confusion-matrix logic work before any real optimization.

Examples:

    python labs/lab2_facial_detection_systems/scripts/03_mnist_evaluation_probe.py
    python labs/lab2_facial_detection_systems/scripts/03_mnist_evaluation_probe.py --source mnist --download

The default synthetic mode keeps this runnable offline. The MNIST mode is for a
quick sanity check once the dataset is available locally.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    from torchvision import datasets, transforms
except Exception:  # pragma: no cover - keeps synthetic mode usable offline.
    datasets = None
    transforms = None


NUM_CLASSES = 10


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
    """Small CNN matching the first Lab 2 architecture at a reviewable scale."""

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
class EvaluationResult:
    model_name: str
    examples: int
    average_loss: float
    accuracy: float
    confusion: torch.Tensor


def build_synthetic_dataset(num_samples: int) -> TensorDataset:
    """Create MNIST-shaped fake data for offline metric checks."""

    generator = torch.Generator().manual_seed(23)
    images = torch.rand(num_samples, 1, 28, 28, generator=generator)
    labels = torch.randint(0, NUM_CLASSES, (num_samples,), generator=generator)
    return TensorDataset(images, labels)


def build_mnist_dataset(data_dir: Path, download: bool, train: bool) -> Dataset:
    """Load the real MNIST split through torchvision."""

    if datasets is None or transforms is None:
        raise RuntimeError(
            "torchvision is required for --source mnist. "
            "Install the repo requirements first."
        )

    transform = transforms.Compose([transforms.ToTensor()])
    return datasets.MNIST(
        root=str(data_dir),
        train=train,
        download=download,
        transform=transform,
    )


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
    """Accumulate rows=true labels and columns=predicted labels."""

    labels = labels.detach().cpu()
    predictions = predictions.detach().cpu()
    flat_indices = labels * NUM_CLASSES + predictions
    confusion += torch.bincount(
        flat_indices,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def evaluate_batches(
    model: nn.Module,
    loader: DataLoader,
    max_batches: int,
) -> EvaluationResult:
    """Run a short evaluation pass with no gradient tracking."""

    loss_function = nn.CrossEntropyLoss(reduction="sum")
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    total_loss = 0.0
    correct = 0
    total = 0

    model.eval()
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            if batch_index >= max_batches:
                break

            logits = model(images)
            loss = loss_function(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item()
            correct += (predictions == labels).sum().item()
            total += batch_size
            update_confusion_matrix(confusion, labels, predictions)

    if total == 0:
        raise RuntimeError("No examples were evaluated. Increase --max-batches.")

    return EvaluationResult(
        model_name=model.__class__.__name__,
        examples=total,
        average_loss=total_loss / total,
        accuracy=correct / total,
        confusion=confusion,
    )


def print_result(result: EvaluationResult, parameter_count: int) -> None:
    print(f"\n{result.model_name}")
    print(f"  trainable parameters: {parameter_count:,}")
    print(f"  examples evaluated   : {result.examples}")
    print(f"  average loss         : {result.average_loss:.4f}")
    print(f"  accuracy             : {result.accuracy:.4f}")
    print("  confusion matrix rows=true, columns=predicted:")
    print(result.confusion)


def parse_args() -> argparse.Namespace:
    lab_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Check MNIST evaluation metrics before running real training."
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "mnist"],
        default="synthetic",
        help="Use offline MNIST-shaped fake data or real torchvision MNIST.",
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
        "--split",
        choices=["train", "test"],
        default="test",
        help="MNIST split to inspect when --source mnist is selected.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used by the evaluation DataLoader.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=3,
        help="Maximum number of batches to evaluate for each model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)

    if args.source == "mnist":
        dataset = build_mnist_dataset(
            data_dir=args.data_dir,
            download=args.download,
            train=args.split == "train",
        )
    else:
        dataset = build_synthetic_dataset(
            num_samples=max(args.batch_size * args.max_batches, args.batch_size)
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    models = [
        FullyConnectedMNISTBaseline(),
        MNISTConvNet(),
    ]

    print(
        "Evaluation-only probe. Models are randomly initialized; "
        "these metrics are plumbing checks, not learned performance."
    )
    print(f"source: {args.source}")
    if args.source == "mnist":
        print(f"split: {args.split}")
    print(f"batch size: {args.batch_size}")
    print(f"max batches per model: {args.max_batches}")

    for model in models:
        result = evaluate_batches(model, loader, max_batches=args.max_batches)
        print_result(result, count_trainable_parameters(model))

    print(
        "\nNext real Lab 2 step: train the fully connected baseline, then re-run "
        "evaluation on the held-out test split before comparing with the CNN."
    )


if __name__ == "__main__":
    main()
