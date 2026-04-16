"""Lab 1 notes in code: simple computation graphs with PyTorch tensors."""

import torch


def func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # This mirrors the style of the computation graph from the notebook:
    # build intermediate values, then combine them into a final output.
    c = torch.add(a, b)
    d = torch.multiply(a, b)
    e = torch.subtract(d, c)
    return e


def main() -> None:
    print("Simple add node:")
    a = torch.tensor(15)
    b = torch.tensor(61)

    c1 = torch.add(a, b)
    c2 = a + b

    print("c1:", c1)
    print("c2:", c2)

    print("\nSlightly larger computation graph:")
    a = torch.tensor(1.5)
    b = torch.tensor(2.5)
    e_out = func(a, b)

    print("a:", a)
    print("b:", b)
    print("e_out:", e_out)
    print("e_out shape:", tuple(e_out.shape))

    # Good reminder: a scalar tensor has rank 0, so the shape really is empty here.
    if e_out.ndim != 0:
        raise RuntimeError("Expected e_out to be a scalar tensor.")


if __name__ == "__main__":
    main()
