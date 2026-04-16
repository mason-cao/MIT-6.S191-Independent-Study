"""Lab 1 notes in code: derive perceptron gradients by hand and compare to autograd."""

import torch


def sigmoid(z: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-z))


def main() -> None:
    x = torch.tensor(
        [
            [1.0, -1.0, 2.0],
            [0.5, 2.0, -0.5],
        ]
    )
    target = torch.tensor(
        [
            [1.0],
            [0.0],
        ]
    )

    weight_matrix = torch.tensor(
        [
            [0.20],
            [-0.30],
            [0.10],
        ],
        requires_grad=True,
    )
    bias_vector = torch.tensor([0.05], requires_grad=True)

    # Forward pass first: z = xW + b, then y_hat = sigmoid(z).
    z = torch.matmul(x, weight_matrix) + bias_vector
    prediction = sigmoid(z)

    # I used 0.5 * sum((prediction - target)^2) because it makes the derivative cleaner:
    # d/dprediction of the loss is just (prediction - target).
    loss = 0.5 * torch.sum((prediction - target) ** 2)

    # This is the backprop chain written out piece by piece so I can see where each
    # factor comes from instead of treating autograd like a black box.
    d_loss_d_prediction = prediction - target
    d_prediction_d_z = prediction * (1.0 - prediction)
    d_loss_d_z = d_loss_d_prediction * d_prediction_d_z

    # Another shape check:
    # x^T is (input_features, batch_size) and dL/dz is (batch_size, output_features),
    # so their product lands back in the shape of W.
    manual_grad_w = torch.matmul(x.transpose(0, 1), d_loss_d_z)

    # The bias affects every example in the batch, so summing over the batch collects
    # the full gradient for b.
    manual_grad_b = torch.sum(d_loss_d_z, dim=0)

    loss.backward()

    print("loss:", loss.item())
    print("\nManual gradient for W:\n", manual_grad_w)
    print("\nAutograd gradient for W:\n", weight_matrix.grad)
    print("\nManual gradient for b:\n", manual_grad_b)
    print("\nAutograd gradient for b:\n", bias_vector.grad)

    w_match = torch.allclose(manual_grad_w, weight_matrix.grad, atol=1e-7)
    b_match = torch.allclose(manual_grad_b, bias_vector.grad, atol=1e-7)

    print(f"\nGradient agreement for W: {w_match}")
    print(f"Gradient agreement for b: {b_match}")

    if not (w_match and b_match):
        raise RuntimeError("Manual gradients do not match autograd.")


if __name__ == "__main__":
    main()
