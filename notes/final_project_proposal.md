# Final Project Proposal: Worst-Group Face Detection With Latent Resampling

## Question

Can DB-VAE-style latent resampling improve worst-group face-detection
performance without hiding tradeoffs behind average accuracy?

## Motivation

Lab 2 showed the problem: a face detector can look strong on average while
performing unevenly across subgroups. Lecture 4 explains why latent variables
can capture hidden structure in the data. Lecture 6 explains why average
accuracy is not enough for robustness or fairness.

This project would turn those ideas into a small evaluation study.

## Proposed Experiment

Train and compare:

1. A standard CNN face detector.
2. A DB-VAE face detector with latent-space resampling.

Evaluate both models on:

- overall accuracy
- subgroup accuracy
- false positive rate by subgroup
- false negative rate by subgroup
- worst-group accuracy
- confidence on correct vs incorrect predictions
- a small set of representative failure examples

## Data Plan

Start with the existing synthetic data path to verify the code. Then run the
official Lab 2 dataset path with:

- CelebA faces for positive examples
- ImageNet non-face examples for negatives
- the balanced face test set used in the official lab

## Success Criteria

The project is successful if it answers the comparison honestly. A good result
does not have to show that DB-VAE wins everywhere. It should show:

- whether worst-group performance improves
- whether average performance changes
- which subgroup metrics move
- whether confidence becomes more or less reliable
- what failure cases remain

## Why This Fits 6.S191

This project uses the course ideas directly:

- Lecture 1: training loop, loss, gradients
- Lecture 3: CNN image features
- Lecture 4: VAE latent variables
- Lab 2: facial detection and DB-VAE debiasing
- Lecture 6: robustness, bias, uncertainty, and deployment caution

The main thing I want to practice is not only training another model. I want to
write an evaluation that makes the limitations visible.
