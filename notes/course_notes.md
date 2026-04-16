# Course Notes

## Lecture 1: Intro to Deep Learning

- this lecture really sets up the whole class
- deep learning is presented as: choose a model, define a loss, optimize parameters
- the perceptron is the basic building block

### Core Setup

- input: `x`
- prediction: `y_hat = f(x; W)`
- target: `y`
- parameters: usually weights `W` and bias `b`
- training objective: make `y_hat` close to `y` over the dataset

Most important equation from the lecture:

- `y_hat = g(Wx + b)`

Need to remember what each part is doing:

- `Wx + b` = linear part
- `g(...)` = nonlinearity
- without the nonlinearity, stacking layers does not buy much because the whole thing is still just linear

### Perceptron

- single neuron = weighted sum of inputs + activation
- in the 2-feature example, the perceptron really is just learning a line in 2D
- weights change slope / orientation of the decision boundary
- bias shifts the boundary

Helpful way to think about it:

- inputs are coordinates
- weights decide which directions matter more
- bias moves the cutoff

If I ever get confused:

- check whether I can write the model as a dot product plus bias
- check whether the output is scalar or vector
- check whether the activation is part of the model or part of the loss setup

### From One Perceptron to a Network

- the slides move from one perceptron to multi-output perceptrons, then to feed-forward networks
- a dense layer is really just a matrix version of the single-perceptron idea
- one output neuron -> one set of weights
- multiple output neurons -> stack those weight vectors into a matrix

Important shape idea:

- if `x` has shape `(m,)` and the layer outputs `n` values, then the weight object has to carry `m x n` relationships
- this is why matrix multiplication becomes the natural language of neural nets so quickly

### Loss

- predictions alone are not enough; need a number that says how wrong the model is
- lecture introduces empirical loss / objective over the whole dataset
- the form is basically:
  `J(W) = (1/n) * sum L(f(x^(i); W), y^(i))`

What this means:

- `L(...)` measures error on one example
- `J(W)` averages that across the training set
- training means changing `W` to reduce `J(W)`

Loss examples to remember:

- mean squared error makes sense for continuous-valued outputs
- classification losses are different because the model is trying to choose among classes, not predict a real number

### Optimization

- gradient descent is the core picture
- update rule is: move parameters in the direction that decreases loss
- generic step:
  `W <- W - eta * grad J(W)`

Things that matter here:

- gradient tells direction of steepest increase, so subtracting it moves downhill
- learning rate `eta` is step size
- too large = unstable / overshoot
- too small = painfully slow

### Backpropagation

- backprop is just the chain rule applied efficiently through the network
- the lecture treats the network as a computation graph, which is the right way to think about it
- each node contributes part of the derivative
- gradients flow backward from loss to earlier layers

Need to keep straight:

- forward pass computes activations / predictions
- backward pass computes derivatives
- same graph, opposite direction

### Why This Lecture Matters for the Lab

- Lab 1 Part 1 is basically this lecture turned into code
- tensors = how data and parameters are stored
- matrix multiplication = how dense layers are computed
- autograd = practical backprop
- `torch.nn` = packaged version of the same math

## Lecture 2: Deep Sequence Modeling

- lecture 2 changes the data type completely
- now the input is not just one vector; it is an ordered sequence
- the key issue is dependency across time

### What Changes in Sequence Modeling

- order matters now
- length can vary
- current prediction may depend on something many steps earlier

Examples from the lecture:

- predict the next word
- sentiment classification from a sequence of words
- later in the lab: generate music one token / character at a time

This is the main mental shift:

- feed-forward setup: one input -> one output
- sequence setup: inputs and outputs are indexed by time

### RNN Idea

- RNN introduces a hidden state that gets updated every time step
- that hidden state is supposed to summarize useful past context
- same parameters are reused at every step

Useful way to think about the recurrence:

- input at time `t`: `x_t`
- hidden state from previous step: `h_(t-1)`
- new hidden state: `h_t`
- output: `y_hat_t`

The exact formula can vary, but the pattern is:

- combine current input with previous hidden state
- update hidden state
- optionally produce an output

This is why RNNs are different from plain feed-forward nets:

- they have memory, at least in principle
- the same cell is reused across time

### Unrolling Across Time

- one of the most useful slides is the computational graph unrolled across time
- that makes the recurrence much easier to reason about
- an RNN is not magic; it is just a repeated module

Important consequence:

- the hidden state creates a path from earlier inputs to later outputs
- when training, gradients have to move backward through all those time steps

### BPTT

- BPTT = backpropagation through time
- basically standard backprop, but on the unrolled sequence graph
- the longer the sequence, the longer the gradient path

What to remember:

- errors at later time steps influence earlier states
- parameter sharing means the same weights collect gradient contributions from many time steps
- sequence length affects both memory usage and optimization difficulty

### Long-Term Dependencies / Vanishing Gradients

- this is one of the main problems from the lecture
- repeated multiplication by small derivatives makes gradients shrink
- if the signal shrinks too much, the model stops learning long-range structure

What that leads to:

- model learns short-term patterns more easily than long-term ones
- important early information can get lost
- plain RNNs struggle with long memory

The lecture also points out broader limitations of vanilla RNN-style recurrence:

- encoding bottleneck
- no real parallelization across time
- weak long-memory behavior

### Sequence Tasks to Keep Straight

- next-token / next-word prediction:
  given previous context, predict what comes next
- sequence classification:
  map a whole sequence to one label, like sentiment
- generation:
  repeatedly feed predictions forward to create new sequence content

### Where the Lecture Is Going

- later parts of the lecture push beyond plain recurrence
- attention is introduced as a way to get long-range access without relying only on recurrent state
- the desired capabilities are basically:
  continuous stream, parallelization, and long memory

That is a useful progression to remember:

- feed-forward networks for fixed inputs
- RNNs for ordered sequences
- attention / transformer ideas for better long-range modeling and parallelism

## Lab 1

Lab 1 makes a lot more sense after Lecture 1 and Lecture 2.

Part 1: Intro to Deep Learning in Python

- basically the coding version of Lecture 1
- tensors, tensor operations, simple computation graphs, perceptrons, `torch.nn`, and autograd
- this is where the math has to turn into exact tensor shapes and exact operations

Things to check carefully while doing Part 1:

- rank / shape of every tensor
- what is batch dimension vs. feature dimension
- when bias is being broadcast
- when a vector should actually be a matrix because batching is involved
- what requires gradients and what does not

Part 2: Music Generation with RNNs

- coding version of Lecture 2
- sequence modeling becomes concrete through character-level music generation in ABC notation
- this is where recurrence stops being a diagram and becomes an actual training loop + sampling procedure

## Connections Between the Two Lectures and Lab 1

- Lecture 1 gives the feed-forward foundation
- Lecture 2 explains why sequential data needs different machinery
- Lab 1 is split the same way:
  Part 1 = feed-forward / PyTorch basics
  Part 2 = sequence modeling / RNN music generation

Nice thing about the course design:

- the lectures give the concepts
- the lab makes the concepts concrete almost immediately

## Lab 1 Coverage in This Repo

- Part 1 is now covered by:
  tensor mechanics, manual perceptron forward pass, manual gradients,
  computation-graph examples, `nn.Module`, `nn.Sequential`, autograd,
  and the scalar gradient-descent demo
- Part 2 is now covered by:
  ABC data loading, vocabulary building, vectorization, batching,
  LSTM training, text generation, and exporting generated songs

What still matters when I actually run it on the Ubuntu box:

- train the LSTM long enough to get usable ABC output
- check whether the generated text contains complete song snippets
- render one generated song to audio with `abc2midi` + `timidity`
