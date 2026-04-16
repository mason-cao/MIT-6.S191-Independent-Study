"""Utilities for the Lab 1 ABC music-generation work."""

from __future__ import annotations

import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


LAB_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = LAB_DIR / "data"
OUTPUTS_DIR = LAB_DIR / "outputs"
OFFICIAL_IRISH_ABC_URL = (
    "https://raw.githubusercontent.com/MITDeepLearning/introtodeeplearning/master/"
    "mitdeeplearning/data/irish.abc"
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_training_data(data_path: str | Path | None = None) -> Path:
    if data_path is not None:
        resolved = Path(data_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Could not find data file at {resolved}")
        return resolved

    ensure_dir(DATA_DIR)
    default_path = DATA_DIR / "irish.abc"
    if default_path.exists():
        return default_path

    # Using the official course dataset keeps the local script aligned with the lab.
    urllib.request.urlretrieve(OFFICIAL_IRISH_ABC_URL, default_path)
    return default_path


def extract_song_snippets(text: str) -> list[str]:
    # The official helper code searches for songs separated by blank lines.
    pattern = r"(?:^|\n\n)(.*?)(?=\n\n|$)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    songs = [song.strip() for song in matches if song.strip().startswith("X:")]
    return songs


def load_training_songs(data_path: str | Path | None = None) -> list[str]:
    resolved_path = ensure_training_data(data_path)
    text = resolved_path.read_text(encoding="utf-8")
    songs = extract_song_snippets(text)
    if not songs:
        raise RuntimeError("No songs were found in the ABC dataset.")
    return songs


def build_vocabulary(text: str) -> tuple[list[str], dict[str, int], list[str]]:
    vocab = sorted(set(text))
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = list(vocab)
    return vocab, char2idx, idx2char


def vectorize_string(text: str, char2idx: dict[str, int]) -> torch.Tensor:
    # This is the same idea as the notebook's vectorization step:
    # replace each character with its vocabulary index.
    return torch.tensor([char2idx[char] for char in text], dtype=torch.long)


def get_batch(
    vectorized_songs: torch.Tensor, seq_length: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    n = vectorized_songs.shape[0] - 1
    if n <= seq_length:
        raise ValueError(
            f"Sequence length {seq_length} is too large for dataset length {n + 1}."
        )

    start_indices = torch.randint(0, n - seq_length, (batch_size,))

    # Each training example is a window of characters, and the label is the same
    # window shifted one step to the right.
    input_batch = torch.stack(
        [vectorized_songs[idx : idx + seq_length] for idx in start_indices]
    )
    output_batch = torch.stack(
        [vectorized_songs[idx + 1 : idx + seq_length + 1] for idx in start_indices]
    )
    return input_batch, output_batch


class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, batch_size, self.hidden_size, device=device),
            torch.zeros(1, batch_size, self.hidden_size, device=device),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding(x)
        if state is None:
            state = self.init_hidden(x.size(0), x.device)
        out, state = self.lstm(x, state)
        out = self.fc(out)
        return out if not return_state else (out, state)


def compute_loss(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    batch_size, seq_length, vocab_size = logits.shape
    # Flattening here is important because cross_entropy expects a 2-d tensor of
    # logits and a 1-d tensor of class indices.
    logits = logits.reshape(batch_size * seq_length, vocab_size)
    labels = labels.reshape(batch_size * seq_length)
    return F.cross_entropy(logits, labels)


def train_step(
    model: LSTMModel,
    optimizer: torch.optim.Optimizer,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
) -> torch.Tensor:
    model.train()
    optimizer.zero_grad()
    logits = model(x_batch)
    loss = compute_loss(y_batch, logits)
    loss.backward()
    optimizer.step()
    return loss.detach()


def generate_text(
    model: LSTMModel,
    start_string: str,
    char2idx: dict[str, int],
    idx2char: list[str],
    generation_length: int,
    temperature: float,
    device: torch.device,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not start_string:
        raise ValueError("start_string must not be empty")

    model.eval()
    input_idx = torch.tensor(
        [[char2idx[char] for char in start_string]],
        dtype=torch.long,
        device=device,
    )

    generated = [char for char in start_string]

    with torch.no_grad():
        predictions, state = model(input_idx, return_state=True)
        logits = predictions[:, -1, :] / temperature

        for _ in range(generation_length):
            probabilities = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probabilities, num_samples=1)
            next_char = idx2char[next_idx.item()]

            generated.append(next_char)
            predictions, state = model(next_idx, state=state, return_state=True)
            logits = predictions[:, -1, :] / temperature

    return "".join(generated)


def save_text(text: str, output_path: Path) -> Path:
    ensure_dir(output_path.parent)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def save_song_to_abc(song: str, output_stem: Path) -> Path:
    ensure_dir(output_stem.parent)
    abc_path = output_stem.with_suffix(".abc")
    abc_path.write_text(song, encoding="utf-8")
    return abc_path


def render_song_to_wav(song: str, output_stem: Path) -> Path | None:
    abc2midi = shutil.which("abc2midi")
    timidity = shutil.which("timidity")
    if abc2midi is None or timidity is None:
        return None

    abc_path = save_song_to_abc(song, output_stem)
    midi_path = output_stem.with_suffix(".mid")
    wav_path = output_stem.with_suffix(".wav")

    subprocess.run([abc2midi, str(abc_path), "-o", str(midi_path)], check=True)
    subprocess.run([timidity, str(midi_path), "-Ow", "-o", str(wav_path)], check=True)
    return wav_path
