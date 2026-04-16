"""Lab 1 notes in code: nn.Module, nn.Sequential, and autograd basics."""

import torch
import torch.nn as nn


class OurDenseLayer(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_inputs, num_outputs))
        self.bias = nn.Parameter(torch.randn(num_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same perceptron as before, just written inside an nn.Module.
        z = torch.matmul(x, self.W) + self.bias
        y = torch.sigmoid(z)
        return y


class LinearButSometimesIdentity(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, inputs: torch.Tensor, isidentity: bool = False) -> torch.Tensor:
        # This part from the notebook is useful because it shows that subclassing
        # nn.Module gives control over behavior, not just parameter storage.
        if isidentity:
            return inputs
        return self.linear(inputs)


def run_gradient_descent_demo() -> None:
    x = torch.randn(1)
    print(f"\nInitializing x={x.item():.4f}")

    learning_rate = 1e-2
    x_f = 4.0
    history = []

    for _ in range(500):
        x = torch.tensor([x.item()], requires_grad=True)

        # This is the same scalar optimization idea as the notebook:
        # minimize the squared error between x and a target value x_f.
        loss = (x - x_f) ** 2
        loss.backward()

        x = torch.tensor([x.item() - learning_rate * x.grad.item()])
        history.append(x.item())

    print(f"Final x after gradient descent: {history[-1]:.4f}")
    print(f"Target x_f: {x_f:.4f}")

    if abs(history[-1] - x_f) > 1e-1:
        raise RuntimeError("Gradient descent demo did not get close enough to the target.")


def main() -> None:
    torch.manual_seed(7)

    print("Custom dense layer:")
    layer = OurDenseLayer(num_inputs=2, num_outputs=3)
    x_input = torch.tensor([[1.0, 2.0]])
    y = layer(x_input)
    print("input shape:", tuple(x_input.shape))
    print("output shape:", tuple(y.shape))
    print("output:", y)

    print("\nSequential API:")
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.Sigmoid(),
    )
    model_output = model(x_input)
    print("model output shape:", tuple(model_output.shape))
    print("model output:", model_output)

    print("\nSubclassing nn.Module for custom behavior:")
    identity_model = LinearButSometimesIdentity(num_inputs=2, num_outputs=3)
    out_with_linear = identity_model(x_input)
    out_with_identity = identity_model(x_input, isidentity=True)
    print("linear output:", out_with_linear)
    print("identity output:", out_with_identity)

    if not torch.equal(out_with_identity, x_input):
        raise RuntimeError("Identity path should return the input unchanged.")

    print("\nAutograd on y = x^2:")
    x = torch.tensor(3.0, requires_grad=True)
    y = x**2
    y.backward()
    dy_dx = x.grad
    print("dy/dx at x=3:", dy_dx)

    if dy_dx.item() != 6.0:
        raise RuntimeError("Expected dy/dx to equal 6.0 when x=3.0.")

    run_gradient_descent_demo()


if __name__ == "__main__":
    main()
