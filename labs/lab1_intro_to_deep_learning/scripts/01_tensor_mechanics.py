"""Lab 1 notes in code: tensor rank, shape, slicing, and broadcasting in PyTorch."""

import torch


def print_section(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def describe_tensor(name: str, tensor: torch.Tensor) -> None:
    print(f"{name}:")
    print(f"  value =\n{tensor}")
    print(f"  shape = {tuple(tensor.shape)}")
    print(f"  rank  = {tensor.ndim}")
    print(f"  dtype = {tensor.dtype}")


def main() -> None:
    print_section("Rank and Shape")

    # Easy thing to mix up: a scalar in PyTorch is rank 0, not rank 1.
    scalar = torch.tensor(7.0)

    # I still want a rank-1 example with one value because it makes the scalar vs.
    # vector distinction much easier to see.
    one_value_vector = torch.tensor([7.0])

    vector = torch.tensor([2.0, 4.0, 6.0, 8.0])

    matrix = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    # For image batches I want to keep the standard ordering straight:
    # (batch_size, channels, height, width).
    image_batch = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)

    describe_tensor("scalar", scalar)
    describe_tensor("one_value_vector", one_value_vector)
    describe_tensor("vector", vector)
    describe_tensor("matrix", matrix)
    describe_tensor("image_batch", image_batch)

    print_section("Slicing")

    row_vector = matrix[1]
    column_vector = matrix[:, 1]
    matrix_entry = matrix[0, 2]

    # Pulling out one image is a good reminder that indexing the batch dimension
    # drops the rank from 4 to 3.
    first_image = image_batch[0]

    describe_tensor("row_vector", row_vector)
    describe_tensor("column_vector", column_vector)
    describe_tensor("matrix_entry", matrix_entry)
    describe_tensor("first_image", first_image)

    print_section("Broadcasting")

    feature_batch = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    bias = torch.tensor([0.1, 0.2, 0.3])

    # Broadcasting matters because the bias gets reused across rows without me having
    # to manually build a full 3 x 3 copy.
    shifted_features = feature_batch + bias

    describe_tensor("feature_batch", feature_batch)
    describe_tensor("bias", bias)
    describe_tensor("shifted_features", shifted_features)

    print(
        "\nBroadcasting explanation:"
        "\n- `feature_batch` has shape (3, 3)."
        "\n- `bias` has shape (3,)."
        "\n- PyTorch aligns the trailing dimensions, treats `bias` as if it were copied"
        "\n  across the batch dimension, and performs the addition efficiently."
    )

    channel_scale = torch.tensor([1.0, 0.5, 2.0]).view(1, 3, 1, 1)

    # Same idea here: one scale per channel, then PyTorch handles the expansion across
    # the batch and spatial dimensions.
    scaled_images = image_batch * channel_scale

    describe_tensor("channel_scale", channel_scale)
    describe_tensor("scaled_images", scaled_images)


if __name__ == "__main__":
    main()
