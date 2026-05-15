# Lab 2: Computer Vision and Facial Detection Systems

This folder tracks my completed local pass through MIT 6.S191 Software Lab 2.

The official lab starts with MNIST digit classification and then moves into
facial detection, algorithmic bias, and debiasing with a variational
autoencoder. I finished the lab in the style of this repo: small PyTorch scripts
that make the mechanics explicit, plus notes that explain the lecture/lab ideas
instead of only copying notebook cells.

## Current Status

- Set up the Lab 2 folder structure.
- Added a first PyTorch script that checks the shape mechanics of a simple CNN on
  MNIST-like image tensors.
- Added a second script that inspects one MNIST-shaped batch and the fully
  connected baseline's forward pass without training it.
- Added a third script that checks evaluation metrics for randomly initialized
  dense and convolutional MNIST models without training either model.
- Added a fourth script that actually trains and compares the dense MNIST
  baseline against the CNN.
- Added a fifth script that implements the local facial detection and DB-VAE
  mechanics: binary CNN logits, BCE loss, grouped face-probability evaluation,
  VAE reconstruction/KL loss, reparameterization, and adaptive latent resampling.
- Expanded the main course notes with detailed Lecture 3, Lecture 4, Lab 2
  Part 1, and Lab 2 Part 2 notes.

## What Counts As Finished Here

For this independent-study repo, Lab 2 is now finished as a local implementation
pass.

The MNIST section can run on either the offline synthetic digit data or the real
`torchvision` MNIST dataset. The facial detection section uses synthetic images
by default because the official CelebA/ImageNet/PPB datasets are large and are
not checked into this repo. That means the facial script is a mechanics check,
not a claim about real demographic fairness performance.

If I want official lab/competition numbers later, the next step is to run the
official PyTorch notebook on a GPU-backed environment with the real datasets and
copy the final grouped evaluation into this repo as an experiment note.

## Study Notes

The important split in this lab is between average accuracy and subgroup
behavior. MNIST is a clean place to practice image tensors, logits, cross
entropy, training loops, and CNN inductive bias. The facial detection section
then raises the harder question: whether the detector works evenly across
different parts of the data distribution.

The DB-VAE section is the key bridge to Lecture 4. The model learns a supervised
face/not-face output and an unsupervised latent representation of face images.
The latent representation is used to resample underrepresented regions during
training. That is different from balancing only the face/non-face labels.

## Scripts

- `01_mnist_cnn_shape_probe.py`: synthetic MNIST-like batch through a small CNN,
  focused on tensor shapes and one synthetic optimizer step.
- `02_mnist_batch_and_baseline_probe.py`: synthetic by default, with an optional
  real MNIST mode; focused on DataLoader batch shape, a two-layer dense baseline,
  logits, and one forward-pass loss.
- `03_mnist_evaluation_probe.py`: synthetic by default, with an optional real
  MNIST mode; focused on `eval()` mode, no-gradient evaluation, loss, accuracy,
  and confusion-matrix plumbing for untrained dense and CNN models.
- `04_mnist_training_comparison.py`: trains the dense baseline and CNN, then
  reports train metrics, held-out accuracy, and confusion matrices.
- `05_facial_debiasing_mechanics.py`: runs a local synthetic facial-detection
  and DB-VAE debiasing mechanics pass.

## Useful Commands

Synthetic/offline MNIST training comparison:

```bash
python labs/lab2_facial_detection_systems/scripts/04_mnist_training_comparison.py
```

Real MNIST training comparison, after installing `torchvision`:

```bash
python labs/lab2_facial_detection_systems/scripts/04_mnist_training_comparison.py --source mnist --download
```

Synthetic/offline facial detection and DB-VAE mechanics:

```bash
python labs/lab2_facial_detection_systems/scripts/05_facial_debiasing_mechanics.py
```
