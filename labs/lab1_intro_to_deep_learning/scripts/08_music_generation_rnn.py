"""Lab 1 notes in code: train an LSTM on ABC music and generate a new sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from music_generation_utils import (
    OUTPUTS_DIR,
    LSTMModel,
    build_vocabulary,
    extract_song_snippets,
    generate_text,
    get_batch,
    load_training_songs,
    render_song_to_wav,
    save_song_to_abc,
    save_text,
    train_step,
    vectorize_string,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--seq-length", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-training-iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--start-string", type=str, default="X")
    parser.add_argument("--generation-length", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    songs = load_training_songs(args.data_path)
    songs_joined = "\n\n".join(songs)
    vocab, char2idx, idx2char = build_vocabulary(songs_joined)
    vectorized_songs = vectorize_string(songs_joined, char2idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path is not None
        else OUTPUTS_DIR / "lab1_music_generation_lstm.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        print("Training model...")
        for iteration in range(args.num_training_iterations):
            x_batch, y_batch = get_batch(
                vectorized_songs,
                seq_length=args.seq_length,
                batch_size=args.batch_size,
            )
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            loss = train_step(model, optimizer, x_batch, y_batch)

            if iteration % args.log_every == 0:
                print(
                    f"iteration {iteration:4d} "
                    f"loss={loss.item():.4f}"
                )

        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    else:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"--skip-train was passed but no checkpoint exists at {checkpoint_path}"
            )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")

    generated_text = generate_text(
        model=model,
        start_string=args.start_string,
        char2idx=char2idx,
        idx2char=idx2char,
        generation_length=args.generation_length,
        temperature=args.temperature,
        device=device,
    )

    generated_text_path = save_text(
        generated_text,
        OUTPUTS_DIR / "generated_abc_text.txt",
    )
    print(f"Saved generated text to {generated_text_path}")

    generated_songs = extract_song_snippets(generated_text)
    print(f"Found {len(generated_songs)} complete song snippets in generated text.")

    if not generated_songs:
        print("No complete ABC song was generated yet. Try training longer.")
        return

    first_song = generated_songs[0]
    abc_path = save_song_to_abc(first_song, OUTPUTS_DIR / "generated_song_0")
    print(f"Saved first generated song to {abc_path}")

    wav_path = render_song_to_wav(first_song, OUTPUTS_DIR / "generated_song_0")
    if wav_path is None:
        print(
            "Did not render audio because `abc2midi` and/or `timidity` are not installed."
        )
        print("On Ubuntu, installing those tools is enough to turn ABC into a WAV file.")
    else:
        print(f"Rendered audio to {wav_path}")


if __name__ == "__main__":
    main()
