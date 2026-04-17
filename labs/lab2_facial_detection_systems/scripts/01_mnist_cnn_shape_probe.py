"""Lab 2 starter: inspect CNN tensor shapes before training on real images.

The official lab begins with MNIST digit classification. Before downloading data
or running a long training loop, I want a small script that makes the tensor
shape story explicit:

- MNIST-like images are batches of 1 x 28 x 28 grayscale tensors.
- Convolution layers turn local spatial patterns into learned feature maps.
- Pooling reduces height and width.
- The classifier sees a flattened version of the final feature map.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


def print_section(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def describe_tensor(name: str, tensor: torch.Tensor) -> None:
    print(f"{name}:")
    print(f"  shape = {tuple(tensor.shape)}")
    print(f"  dtype = {tensor.dtype}")
    print(f"  min   = {tensor.min().item():.4f}")
    print(f"  max   = {tensor.max().item():.4f}")


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
        self.fc2 = nn.Linear(128, 10)

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> OrderedDict[str, torch.Tensor]:
        """Return intermediate tensors so the dimension changes are visible."""

        activations: OrderedDict[str, torch.Tensor] = OrderedDict()

        activations["input"] = x
        x = self.conv1(x)
        activations["after conv1"] = x
        x = self.relu(x)
        activations["after relu1"] = x
        x = self.pool1(x)
        activations["after pool1"] = x

        x = self.conv2(x)
        activations["after conv2"] = x
        x = self.relu(x)
        activations["after relu2"] = x
        x = self.pool2(x)
        activations["after pool2"] = x

        x = self.flatten(x)
        activations["after flatten"] = x
        x = self.fc1(x)
        activations["after fc1"] = x
        x = self.relu(x)
        activations["after relu3"] = x
        x = self.fc2(x)
        activations["logits"] = x

        return activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["logits"]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def run_one_synthetic_training_step(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Verify that logits, cross entropy, backprop, and optimizer step connect."""

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    logits = model(images)
    loss = loss_function(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach()


def main() -> None:
    torch.manual_seed(7)

    print_section("Synthetic MNIST-like Batch")
    batch_size = 8
    images = torch.rand(batch_size, 1, 28, 28)
    labels = torch.randint(low=0, high=10, size=(batch_size,))
    describe_tensor("images", images)
    describe_tensor("labels", labels)

    print(
        "\nShape reminder:"
        "\n- batch dimension: 8 images"
        "\n- channel dimension: 1 grayscale channel"
        "\n- spatial dimensions: 28 x 28 pixels"
    )

    print_section("CNN Forward Shapes")
    model = MNISTConvNet()
    activations = model.forward_features(images)
    for name, tensor in activations.items():
        print(f"{name:>14}: {tuple(tensor.shape)}")

    print_section("Parameters and Logits")
    parameter_count = count_trainable_parameters(model)
    logits = activations["logits"]
    predicted_digits = torch.argmax(logits, dim=1)

    print(f"trainable parameters: {parameter_count:,}")
    describe_tensor("logits", logits)
    print(f"predicted_digits shape: {tuple(predicted_digits.shape)}")
    print(f"predicted_digits: {predicted_digits.tolist()}")

    print(
        "\nLoss reminder:"
        "\n- `nn.CrossEntropyLoss` consumes raw logits with shape (batch, classes)."
        "\n- The labels are integer class ids with shape (batch,)."
        "\n- I should not add softmax before this loss; PyTorch handles that internally."
    )

    print_section("One Synthetic Training Step")
    loss = run_one_synthetic_training_step(model, images, labels)
    print(f"loss after one synthetic batch: {loss.item():.4f}")
    print("This only checks the mechanics; it is not meaningful model training yet.")


if __name__ == "__main__":
    main()
