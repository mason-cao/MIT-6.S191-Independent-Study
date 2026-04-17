# Lab 2: Computer Vision and Facial Detection Systems

This folder tracks my start on MIT 6.S191 Software Lab 2.

The official lab starts with MNIST digit classification and then moves into facial
detection, algorithmic bias, and debiasing with a variational autoencoder. I am
starting slowly here because I want the repo to be useful for review later, not
just a pile of copied notebook cells.

## Current Status

- Set up the Lab 2 folder structure.
- Added a first PyTorch script that checks the shape mechanics of a simple CNN on
  MNIST-like image tensors.
- Added a second script that inspects one MNIST-shaped batch and the fully
  connected baseline's forward pass without training it.
- Expanded the main course notes with Lecture 3 computer vision notes.
- Intentionally did not start the full dataset training or debiasing sections yet.

## What I Want To Understand First

- Why image tensors are usually represented as `(batch, channels, height, width)`.
- How convolution changes the channel count while preserving or changing spatial
  dimensions depending on kernel size, stride, and padding.
- How max pooling reduces spatial resolution.
- Why the final convolutional feature map has to be flattened before the classifier.
- What the fully connected MNIST baseline throws away when it flattens pixels.
- How to compare baseline and CNN results without mixing up train accuracy,
  test accuracy, and loss.
- Why `CrossEntropyLoss` expects raw logits instead of probabilities.

## Next Small Step

Run the new batch probe against the real MNIST dataset with `torchvision`, then
train the fully connected baseline for a small number of epochs. After that, I
should train the CNN and compare test accuracy, loss, and the effect of keeping
spatial structure before flattening.

I am stopping before that training step for now so this repo does not pretend
Lab 2 is done.

## Scripts

- `01_mnist_cnn_shape_probe.py`: synthetic MNIST-like batch through a small CNN,
  focused on tensor shapes and one synthetic optimizer step.
- `02_mnist_batch_and_baseline_probe.py`: synthetic by default, with an optional
  real MNIST mode; focused on DataLoader batch shape, a two-layer dense baseline,
  logits, and one forward-pass loss.
