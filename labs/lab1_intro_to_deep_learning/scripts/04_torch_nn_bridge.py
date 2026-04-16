"""Lab 1 notes in code: map the manual perceptron onto torch.nn after the math is clear."""

import torch
import torch.nn as nn


def manual_sigmoid(z: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-z))


def manual_perceptron_forward(
    x: torch.Tensor, weight_matrix: torch.Tensor, bias_vector: torch.Tensor
) -> torch.Tensor:
    z = torch.matmul(x, weight_matrix) + bias_vector
    return manual_sigmoid(z)


class TorchPerceptron(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


def main() -> None:
    x = torch.tensor(
        [
            [1.0, 0.5, -1.0],
            [0.0, 2.0, 1.0],
        ]
    )
    weight_matrix = torch.tensor(
        [
            [0.2, -0.4],
            [0.7, 0.1],
            [-0.5, 0.3],
        ]
    )
    bias_vector = torch.tensor([0.05, -0.10])

    manual_output = manual_perceptron_forward(x, weight_matrix, bias_vector)

    model = TorchPerceptron(num_inputs=3, num_outputs=2)
    with torch.no_grad():
        # One detail I do not want to forget:
        # nn.Linear stores weights as (out_features, in_features), which is the
        # transpose of the manual convention I used earlier.
        model.linear.weight.copy_(weight_matrix.transpose(0, 1))
        model.linear.bias.copy_(bias_vector)

    torch_nn_output = model(x)

    print("Manual perceptron output:\n", manual_output)
    print("\nTorch nn output:\n", torch_nn_output)

    if not torch.allclose(manual_output, torch_nn_output, atol=1e-7):
        raise RuntimeError("torch.nn result does not match the manual perceptron result.")


if __name__ == "__main__":
    main()
