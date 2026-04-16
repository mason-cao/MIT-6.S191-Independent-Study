"""Lab 1 notes in code: inspect the ABC dataset and build RNN training batches."""

from music_generation_utils import (
    build_vocabulary,
    get_batch,
    load_training_songs,
    vectorize_string,
)


def main() -> None:
    songs = load_training_songs()
    songs_joined = "\n\n".join(songs)

    print(f"Found {len(songs)} songs in the training data.")
    print("First song snippet:\n")
    print(songs[0][:400])

    vocab, char2idx, idx2char = build_vocabulary(songs_joined)
    vectorized_songs = vectorize_string(songs_joined, char2idx)

    print("\nVocabulary size:", len(vocab))
    print("First 20 character mappings:")
    print("{")
    for char in vocab[:20]:
        print(f"  {repr(char):>4}: {char2idx[char]:3d},")
    print("  ...")
    print("}")

    print(
        "\nFirst 10 characters mapped to integers:\n"
        f"{repr(songs_joined[:10])} -> {vectorized_songs[:10].tolist()}"
    )

    x_batch, y_batch = get_batch(vectorized_songs, seq_length=10, batch_size=2)
    print("\nBatch shapes:")
    print("x_batch:", tuple(x_batch.shape))
    print("y_batch:", tuple(y_batch.shape))

    # This is the key next-step relationship from the notebook.
    if not (x_batch[:, 1:] == y_batch[:, :-1]).all():
        raise RuntimeError("Expected targets to be inputs shifted one character ahead.")

    print("\nOne sequence pair step-by-step:")
    for step, (input_idx, target_idx) in enumerate(zip(x_batch[0], y_batch[0])):
        print(f"step {step:2d}")
        print(f"  input:           {input_idx.item():3d} ({repr(idx2char[input_idx.item()])})")
        print(
            f"  expected output: {target_idx.item():3d} "
            f"({repr(idx2char[target_idx.item()])})"
        )


if __name__ == "__main__":
    main()
