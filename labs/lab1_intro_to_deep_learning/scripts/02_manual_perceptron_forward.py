"""Lab 1 notes in code: manual perceptron forward pass with matmul and sigmoid."""

import torch


def sigmoid(z: torch.Tensor) -> torch.Tensor:
    # Writing sigmoid out directly keeps the math visible instead of hiding it in a
    # layer or helper.
    return 1.0 / (1.0 + torch.exp(-z))


def manual_perceptron_forward(
    x: torch.Tensor, weight_matrix: torch.Tensor, bias_vector: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError(f"x must be rank 2, got shape {tuple(x.shape)}")
    if weight_matrix.ndim != 2:
        raise ValueError(
            f"weight_matrix must be rank 2, got shape {tuple(weight_matrix.shape)}"
        )
    if bias_vector.ndim != 1:
        raise ValueError(
            f"bias_vector must be rank 1, got shape {tuple(bias_vector.shape)}"
        )
    if x.shape[1] != weight_matrix.shape[0]:
        raise ValueError(
            "Input feature count does not match the weight matrix rows: "
            f"x.shape={tuple(x.shape)}, weight_matrix.shape={tuple(weight_matrix.shape)}"
        )
    if weight_matrix.shape[1] != bias_vector.shape[0]:
        raise ValueError(
            "Output feature count does not match the bias width: "
            f"weight_matrix.shape={tuple(weight_matrix.shape)}, "
            f"bias_vector.shape={tuple(bias_vector.shape)}"
        )

    # This is the main shape check I care about:
    # (batch_size, input_features) @ (input_features, output_features).
    linear_output = torch.matmul(x, weight_matrix)

    # The bias stays rank 1 here, and broadcasting handles copying it across the batch.
    shifted_output = linear_output + bias_vector

    # Without the activation this would still just be linear, so this is the point
    # where the perceptron actually becomes nonlinear.
    activated_output = sigmoid(shifted_output)
    return shifted_output, activated_output


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

    shifted_output, activated_output = manual_perceptron_forward(
        x=x,
        weight_matrix=weight_matrix,
        bias_vector=bias_vector,
    )

    print("x shape:", tuple(x.shape))
    print("weight_matrix shape:", tuple(weight_matrix.shape))
    print("bias_vector shape:", tuple(bias_vector.shape))
    print("\nPre-activation values z = xW + b:\n", shifted_output)
    print("\nPost-activation values y_hat = sigmoid(z):\n", activated_output)


if __name__ == "__main__":
    main()
