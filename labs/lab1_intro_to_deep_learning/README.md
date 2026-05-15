# Lab 1: Deep Learning in Python and Music Generation

This folder tracks my local pass through MIT 6.S191 Software Lab 1. The official
2026 lab is split into two ideas: first, learn the PyTorch mechanics behind deep
learning; second, use sequence modeling to generate music in ABC notation.

## Current Status

- Built a tensor-mechanics script for rank, shape, slicing, image-batch layout,
  and broadcasting.
- Wrote a manual batched perceptron forward pass with explicit shape checks.
- Derived a sigmoid-perceptron gradient by hand and compared it to PyTorch
  autograd.
- Bridged the manual perceptron to `torch.nn.Linear`, including the weight
  transpose convention.
- Added small computation-graph and `nn.Module` examples.
- Added a scalar gradient-descent demo so optimization is visible without a
  full model around it.
- Built the ABC music data pipeline: load songs, build a character vocabulary,
  vectorize text, and create next-character training batches.
- Added an LSTM music-generation script that trains on ABC notation, samples new
  text, saves generated ABC, and optionally renders audio if `abc2midi` and
  `timidity` are installed.

## What Counts As Finished Here

For this independent-study repo, Lab 1 is complete as a local mechanics pass.
The scripts split the official notebook flow into small checks that I can rerun
when I forget exactly what PyTorch is doing.

The main ideas I need to retain are:

- tensors are not only arrays; rank, shape, dtype, and broadcasting determine
  the model computation
- a dense layer is a batch of dot products plus a broadcasted bias
- nonlinear activations are what make stacked layers more than one big linear
  map
- autograd is chain rule bookkeeping over a computation graph
- sequence modeling turns text or music into next-token prediction
- the target batch for character modeling is the input batch shifted one
  character to the right
- LSTM generation repeats inference one character at a time, feeding sampled
  output back into the model

## Scripts

- `01_tensor_mechanics.py`: tensor rank, shape, slicing, image-batch layout, and
  broadcasting.
- `02_manual_perceptron_forward.py`: manual `XW + b` and sigmoid forward pass.
- `03_manual_gradient_vs_autograd.py`: manual gradient derivation checked
  against autograd.
- `04_torch_nn_bridge.py`: maps the manual perceptron to `torch.nn.Linear`.
- `05_tensor_computation_graphs.py`: simple PyTorch computation graph examples.
- `06_models_and_autograd.py`: custom `nn.Module`, `nn.Sequential`, autograd,
  and scalar gradient descent.
- `07_music_data_pipeline.py`: ABC song loading, vocabulary, vectorization, and
  next-character batches.
- `08_music_generation_rnn.py`: LSTM training, text sampling, ABC export, and
  optional WAV rendering.

## Useful Commands

Run the foundations scripts:

```bash
python labs/lab1_intro_to_deep_learning/scripts/01_tensor_mechanics.py
python labs/lab1_intro_to_deep_learning/scripts/02_manual_perceptron_forward.py
python labs/lab1_intro_to_deep_learning/scripts/03_manual_gradient_vs_autograd.py
python labs/lab1_intro_to_deep_learning/scripts/04_torch_nn_bridge.py
python labs/lab1_intro_to_deep_learning/scripts/05_tensor_computation_graphs.py
python labs/lab1_intro_to_deep_learning/scripts/06_models_and_autograd.py
```

Inspect the music data pipeline:

```bash
python labs/lab1_intro_to_deep_learning/scripts/07_music_data_pipeline.py
```

Train a shorter local LSTM smoke run:

```bash
python labs/lab1_intro_to_deep_learning/scripts/08_music_generation_rnn.py --num-training-iterations 50 --log-every 10 --generation-length 300
```

Run generation from an existing checkpoint:

```bash
python labs/lab1_intro_to_deep_learning/scripts/08_music_generation_rnn.py --skip-train
```
