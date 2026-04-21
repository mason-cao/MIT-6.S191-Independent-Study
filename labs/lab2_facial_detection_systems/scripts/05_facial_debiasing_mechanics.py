"""Lab 2 Part 2: facial detection and DB-VAE mechanics.

The official second half of Lab 2 uses CelebA, ImageNet negatives, and a
balanced face test set to study facial detection and algorithmic bias. Those
datasets are large and are not checked into this repo, so this script focuses on
the parts I can verify locally:

- a CNN binary face/not-face classifier with logits and BCEWithLogitsLoss
- grouped evaluation in the style of the four Lab 2 test demographics
- VAE loss, reparameterization, decoder reconstruction, and DB-VAE forward pass
- adaptive resampling over learned latent variables

The default data is synthetic and should not be interpreted as a fairness result.
It is a runnable mechanics check before moving the same code ideas to the
official Colab/GPU dataset path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


IMAGE_SIZE = 64
GROUP_KEYS = ["Light Female", "Light Male", "Dark Female", "Dark Male"]


class ConvBlock(nn.Module):
    """Convolution, ReLU, and batch norm block used by the face classifier."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        return self.batch_norm(x)


class StandardFaceClassifier(nn.Module):
    """Reduced local version of the Lab 2 CNN face detector."""

    def __init__(self, n_outputs: int = 1, n_filters: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, n_filters, kernel_size=5, stride=2, padding=2),
            ConvBlock(n_filters, 2 * n_filters, kernel_size=5, stride=2, padding=2),
            ConvBlock(2 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1),
            ConvBlock(4 * n_filters, 6 * n_filters, kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 6 * n_filters, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_outputs),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        return self.classifier(x)


class FaceDecoder(nn.Module):
    """Decoder that maps latent variables back to 64x64 RGB images."""

    def __init__(self, latent_dim: int, n_filters: int = 8) -> None:
        super().__init__()
        self.n_filters = n_filters
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 6 * n_filters),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                6 * n_filters,
                4 * n_filters,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                4 * n_filters,
                2 * n_filters,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                2 * n_filters,
                n_filters,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                n_filters,
                3,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.linear(z)
        x = x.view(-1, 6 * self.n_filters, 4, 4)
        return torch.sigmoid(self.deconv(x))


class DBVAE(nn.Module):
    """Debiasing variational autoencoder for binary facial detection."""

    def __init__(self, latent_dim: int = 16, n_filters: int = 8) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = StandardFaceClassifier(
            n_outputs=2 * latent_dim + 1,
            n_filters=n_filters,
        )
        self.decoder = FaceDecoder(latent_dim=latent_dim, n_filters=n_filters)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_output = self.encoder(x)
        y_logit = encoder_output[:, :1]
        z_mean = encoder_output[:, 1 : self.latent_dim + 1]
        z_logsigma = encoder_output[:, self.latent_dim + 1 :]
        return y_logit, z_mean, z_logsigma

    def reparameterize(
        self,
        z_mean: torch.Tensor,
        z_logsigma: torch.Tensor,
    ) -> torch.Tensor:
        return sample_latent_vector(z_mean, z_logsigma)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y_logit, z_mean, z_logsigma = self.encode(x)
        z = self.reparameterize(z_mean, z_logsigma)
        reconstruction = self.decode(z)
        return y_logit, z_mean, z_logsigma, reconstruction

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        y_logit, _, _ = self.encode(x)
        return y_logit


@dataclass(frozen=True)
class BinaryMetrics:
    average_loss: float
    accuracy: float


def make_coordinate_grid() -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(-1.0, 1.0, IMAGE_SIZE)
    return torch.meshgrid(coords, coords, indexing="ij")


def draw_synthetic_face(group: int, generator: torch.Generator) -> torch.Tensor:
    """Draw a simple face-like tensor with skin tone as the only group signal."""

    yy, xx = make_coordinate_grid()
    background = torch.tensor([0.08, 0.10, 0.12]).view(3, 1, 1)
    image = background.repeat(1, IMAGE_SIZE, IMAGE_SIZE)

    light_skin = torch.tensor([0.78, 0.58, 0.43])
    dark_skin = torch.tensor([0.34, 0.22, 0.15])
    skin = light_skin if group in (0, 1) else dark_skin
    skin = torch.clamp(
        skin + 0.04 * torch.randn((3,), generator=generator),
        min=0.05,
        max=0.95,
    )

    face_mask = (xx / 0.58) ** 2 + ((yy + 0.02) / 0.74) ** 2 <= 1.0
    image[:, face_mask] = skin[:, None]

    hair_mask = ((xx / 0.64) ** 2 + ((yy + 0.42) / 0.36) ** 2 <= 1.0) & (yy < -0.25)
    image[:, hair_mask] = torch.tensor([0.06, 0.045, 0.035]).view(3, 1)

    left_eye = ((xx + 0.22) / 0.07) ** 2 + ((yy + 0.13) / 0.04) ** 2 <= 1.0
    right_eye = ((xx - 0.22) / 0.07) ** 2 + ((yy + 0.13) / 0.04) ** 2 <= 1.0
    mouth = ((xx / 0.25) ** 2 + ((yy - 0.32) / 0.06) ** 2 <= 1.0) & (yy > 0.30)
    image[:, left_eye | right_eye | mouth] = torch.tensor([0.02, 0.02, 0.02]).view(3, 1)

    noise = 0.025 * torch.randn((3, IMAGE_SIZE, IMAGE_SIZE), generator=generator)
    return torch.clamp(image + noise, min=0.0, max=1.0)


def draw_synthetic_nonface(generator: torch.Generator) -> torch.Tensor:
    """Create an image with simple geometric texture but no face structure."""

    image = 0.08 + 0.35 * torch.rand((3, IMAGE_SIZE, IMAGE_SIZE), generator=generator)

    for _ in range(6):
        y0 = int(torch.randint(0, IMAGE_SIZE - 10, (1,), generator=generator).item())
        x0 = int(torch.randint(0, IMAGE_SIZE - 10, (1,), generator=generator).item())
        height = int(torch.randint(5, 18, (1,), generator=generator).item())
        width = int(torch.randint(5, 18, (1,), generator=generator).item())
        color = torch.rand((3, 1, 1), generator=generator)
        image[:, y0 : y0 + height, x0 : x0 + width] = color

    return torch.clamp(image, min=0.0, max=1.0)


def biased_group_counts(num_faces: int) -> list[int]:
    """Match the lab idea of overrepresented and underrepresented face groups."""

    counts = [
        int(num_faces * 0.70),
        int(num_faces * 0.20),
        int(num_faces * 0.07),
    ]
    counts.append(num_faces - sum(counts))
    return counts


def build_face_tensors(
    num_faces: int,
    seed: int,
    balanced: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    if balanced:
        base = num_faces // len(GROUP_KEYS)
        counts = [base] * len(GROUP_KEYS)
        counts[-1] += num_faces - sum(counts)
    else:
        counts = biased_group_counts(num_faces)

    images: list[torch.Tensor] = []
    groups: list[int] = []
    for group, count in enumerate(counts):
        for _ in range(count):
            images.append(draw_synthetic_face(group, generator))
            groups.append(group)

    order = torch.randperm(len(images), generator=generator)
    image_tensor = torch.stack(images, dim=0)[order]
    group_tensor = torch.tensor(groups, dtype=torch.long)[order]
    return image_tensor, group_tensor


def build_nonface_tensors(num_examples: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.stack(
        [draw_synthetic_nonface(generator) for _ in range(num_examples)],
        dim=0,
    )


def make_standard_loader(
    face_images: torch.Tensor,
    nonface_images: torch.Tensor,
    batch_size: int,
    seed: int,
) -> DataLoader:
    images = torch.cat([face_images, nonface_images], dim=0)
    labels = torch.cat(
        [
            torch.ones((len(face_images), 1)),
            torch.zeros((len(nonface_images), 1)),
        ],
        dim=0,
    )
    groups = torch.cat(
        [
            torch.zeros((len(face_images),), dtype=torch.long),
            -torch.ones((len(nonface_images),), dtype=torch.long),
        ],
        dim=0,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(images, labels, groups),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def sample_latent_vector(
    z_mean: torch.Tensor,
    z_logsigma: torch.Tensor,
) -> torch.Tensor:
    """VAE reparameterization trick."""

    epsilon = torch.randn_like(z_mean)
    return z_mean + torch.exp(0.5 * z_logsigma) * epsilon


def vae_loss_function(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    z_mean: torch.Tensor,
    z_logsigma: torch.Tensor,
    kl_weight: float = 5e-4,
) -> torch.Tensor:
    """Return per-example VAE loss: reconstruction plus KL regularization."""

    latent_loss = 0.5 * torch.sum(
        torch.exp(z_logsigma) + z_mean.pow(2) - 1.0 - z_logsigma,
        dim=1,
    )
    reconstruction_loss = torch.mean(torch.abs(x - x_recon), dim=(1, 2, 3))
    return reconstruction_loss + kl_weight * latent_loss


def debiasing_loss_function(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    y: torch.Tensor,
    y_logit: torch.Tensor,
    z_mean: torch.Tensor,
    z_logsigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DB-VAE loss: classify every sample, reconstruct only face samples."""

    vae_loss = vae_loss_function(x, x_recon, z_mean, z_logsigma)
    classification_loss = F.binary_cross_entropy_with_logits(
        y_logit,
        y,
        reduction="none",
    ).squeeze(1)
    face_indicator = (y.squeeze(1) == 1.0).float()
    total_loss = torch.mean(classification_loss + face_indicator * vae_loss)
    return total_loss, torch.mean(classification_loss)


def train_standard_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> BinaryMetrics:
    loss_function = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    model.train()
    for images, labels, _groups in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_function(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = (torch.sigmoid(logits) >= 0.5).float()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size

    return BinaryMetrics(average_loss=total_loss / total, accuracy=correct / total)


def get_latent_mu(
    images: torch.Tensor,
    dbvae: DBVAE,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    dbvae.eval()
    chunks: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            _y_logit, z_mean, _z_logsigma = dbvae.encode(batch)
            chunks.append(z_mean.cpu())

    return torch.cat(chunks, dim=0).numpy()


def training_sample_probabilities_from_mu(
    mu: np.ndarray,
    bins: int = 10,
    smoothing_fac: float = 0.001,
) -> np.ndarray:
    sample_probabilities = np.zeros(mu.shape[0], dtype=np.float64)

    for latent_index in range(mu.shape[1]):
        latent_distribution = mu[:, latent_index]
        hist_density, bin_edges = np.histogram(
            latent_distribution,
            density=True,
            bins=bins,
        )
        hist_smoothed_density = hist_density + smoothing_fac
        hist_smoothed_density /= np.sum(hist_smoothed_density)

        bin_indices = np.digitize(latent_distribution, bin_edges[1:-1])
        probabilities = 1.0 / hist_smoothed_density[bin_indices]
        probabilities /= np.sum(probabilities)
        sample_probabilities = np.maximum(sample_probabilities, probabilities)

    sample_probabilities /= np.sum(sample_probabilities)
    return sample_probabilities


def get_training_sample_probabilities(
    face_images: torch.Tensor,
    dbvae: DBVAE,
    device: torch.device,
    latent_batch_size: int,
    bins: int = 10,
    smoothing_fac: float = 0.001,
) -> np.ndarray:
    mu = get_latent_mu(face_images, dbvae, device, latent_batch_size)
    return training_sample_probabilities_from_mu(mu, bins, smoothing_fac)


def train_dbvae_epoch(
    dbvae: DBVAE,
    face_images: torch.Tensor,
    nonface_images: torch.Tensor,
    face_sample_probabilities: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    generator: torch.Generator,
) -> BinaryMetrics:
    total_loss = 0.0
    correct = 0
    total = 0
    steps = max(1, (len(face_images) + len(nonface_images)) // batch_size)
    probability_tensor = torch.from_numpy(face_sample_probabilities).float()

    dbvae.train()
    for _ in range(steps):
        positive_count = batch_size // 2
        negative_count = batch_size - positive_count
        positive_indices = torch.multinomial(
            probability_tensor,
            num_samples=positive_count,
            replacement=True,
            generator=generator,
        )
        negative_indices = torch.randint(
            len(nonface_images),
            (negative_count,),
            generator=generator,
        )

        positive_images = face_images[positive_indices]
        negative_images = nonface_images[negative_indices]
        images = torch.cat([positive_images, negative_images], dim=0)
        labels = torch.cat(
            [
                torch.ones((positive_count, 1)),
                torch.zeros((negative_count, 1)),
            ],
            dim=0,
        )

        order = torch.randperm(batch_size, generator=generator)
        images = images[order].to(device)
        labels = labels[order].to(device)

        y_logit, z_mean, z_logsigma, reconstruction = dbvae(images)
        loss, class_loss = debiasing_loss_function(
            images,
            reconstruction,
            labels,
            y_logit,
            z_mean,
            z_logsigma,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = (torch.sigmoid(y_logit) >= 0.5).float()
        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size

    return BinaryMetrics(average_loss=total_loss / total, accuracy=correct / total)


def evaluate_group_face_probabilities(
    model: nn.Module,
    face_images: torch.Tensor,
    face_groups: torch.Tensor,
    device: torch.device,
    is_dbvae: bool,
) -> dict[str, float]:
    model.eval()
    results: dict[str, float] = {}

    with torch.inference_mode():
        for group_index, group_name in enumerate(GROUP_KEYS):
            group_images = face_images[face_groups == group_index].to(device)
            if is_dbvae:
                logits = model.predict(group_images)  # type: ignore[attr-defined]
            else:
                logits = model(group_images)
            probabilities = torch.sigmoid(logits).squeeze(1)
            results[group_name] = probabilities.mean().item()

    return results


def toy_latents_from_groups(groups: torch.Tensor) -> np.ndarray:
    """Known latent structure used only to sanity-check inverse-density sampling."""

    tone = torch.where(groups < 2, torch.ones_like(groups), -torch.ones_like(groups))
    subgroup = torch.where(groups % 2 == 0, torch.ones_like(groups), -torch.ones_like(groups))
    return torch.stack([tone.float(), subgroup.float()], dim=1).numpy()


def summarize_by_group(values: np.ndarray, groups: torch.Tensor) -> dict[str, float]:
    return {
        group_name: float(values[groups.numpy() == group_index].mean())
        for group_index, group_name in enumerate(GROUP_KEYS)
    }


def print_group_table(title: str, values: dict[str, float]) -> None:
    print(f"\n{title}")
    for group_name in GROUP_KEYS:
        print(f"  {group_name:13}: {values[group_name]:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local facial detection and DB-VAE mechanics checks."
    )
    parser.add_argument("--face-count", type=int, default=160)
    parser.add_argument("--nonface-count", type=int, default=160)
    parser.add_argument("--test-per-group", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--standard-epochs", type=int, default=1)
    parser.add_argument("--dbvae-epochs", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    face_images, face_groups = build_face_tensors(
        num_faces=args.face_count,
        seed=args.seed,
        balanced=False,
    )
    nonface_images = build_nonface_tensors(args.nonface_count, seed=args.seed + 1)
    test_faces, test_groups = build_face_tensors(
        num_faces=args.test_per_group * len(GROUP_KEYS),
        seed=args.seed + 2,
        balanced=True,
    )

    print("Lab 2 Part 2 facial detection and DB-VAE mechanics")
    print(f"device: {device}")
    print(f"training faces: {len(face_images)}")
    print(f"training non-faces: {len(nonface_images)}")
    print("training face group counts:")
    for group_index, group_name in enumerate(GROUP_KEYS):
        print(f"  {group_name:13}: {(face_groups == group_index).sum().item()}")

    toy_probabilities = training_sample_probabilities_from_mu(
        toy_latents_from_groups(face_groups),
        bins=4,
        smoothing_fac=0.001,
    )
    print_group_table(
        "Known-latent resampling sanity check, mean sample probability",
        summarize_by_group(toy_probabilities, face_groups),
    )

    standard_loader = make_standard_loader(
        face_images=face_images,
        nonface_images=nonface_images,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    standard_classifier = StandardFaceClassifier().to(device)
    standard_optimizer = torch.optim.Adam(
        standard_classifier.parameters(),
        lr=args.learning_rate,
    )

    for epoch in range(1, args.standard_epochs + 1):
        metrics = train_standard_epoch(
            model=standard_classifier,
            loader=standard_loader,
            optimizer=standard_optimizer,
            device=device,
        )
        print(
            f"\nstandard CNN epoch {epoch}: "
            f"loss={metrics.average_loss:.4f}, accuracy={metrics.accuracy:.4f}"
        )

    standard_group_probs = evaluate_group_face_probabilities(
        standard_classifier,
        test_faces,
        test_groups,
        device,
        is_dbvae=False,
    )
    print_group_table("Standard CNN mean face probability on balanced test faces", standard_group_probs)

    dbvae = DBVAE(latent_dim=args.latent_dim).to(device)
    dbvae_optimizer = torch.optim.Adam(dbvae.parameters(), lr=args.learning_rate)
    train_generator = torch.Generator().manual_seed(args.seed + 3)

    for epoch in range(1, args.dbvae_epochs + 1):
        face_probabilities = get_training_sample_probabilities(
            face_images,
            dbvae,
            device,
            latent_batch_size=args.batch_size,
            bins=8,
            smoothing_fac=0.001,
        )
        print_group_table(
            f"DB-VAE epoch {epoch} learned-latent sample probability by group",
            summarize_by_group(face_probabilities, face_groups),
        )

        metrics = train_dbvae_epoch(
            dbvae=dbvae,
            face_images=face_images,
            nonface_images=nonface_images,
            face_sample_probabilities=face_probabilities,
            optimizer=dbvae_optimizer,
            device=device,
            batch_size=args.batch_size,
            generator=train_generator,
        )
        print(
            f"DB-VAE epoch {epoch}: "
            f"loss={metrics.average_loss:.4f}, classification accuracy={metrics.accuracy:.4f}"
        )

    dbvae_group_probs = evaluate_group_face_probabilities(
        dbvae,
        test_faces,
        test_groups,
        device,
        is_dbvae=True,
    )
    print_group_table("DB-VAE mean face probability on balanced test faces", dbvae_group_probs)

    print(
        "\nThis is a synthetic mechanics run. It verifies that the Lab 2 Part 2 "
        "model, losses, grouped evaluation, and adaptive resampling code connect; "
        "it does not replace the official CelebA/ImageNet/PPB experiment."
    )


if __name__ == "__main__":
    main()
