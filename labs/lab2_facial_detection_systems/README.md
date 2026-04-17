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
- Intentionally did not start the full dataset training or debiasing sections yet.

## What I Want To Understand First

- Why image tensors are usually represented as `(batch, channels, height, width)`.
- How convolution changes the channel count while preserving or changing spatial
  dimensions depending on kernel size, stride, and padding.
- How max pooling reduces spatial resolution.
- Why the final convolutional feature map has to be flattened before the classifier.
- Why `CrossEntropyLoss` expects raw logits instead of probabilities.

## Next Small Step

Load the real MNIST dataset with `torchvision`, train a fully connected baseline,
and compare it against the CNN. The key comparison should be test accuracy and
how much spatial structure the CNN gets to use that the fully connected baseline
throws away during flattening.
