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

## Lecture 3: Deep Computer Vision

Lecture 3 shifts from sequences to images. The big framing question is: how can a
model look at raw pixels and recover useful structure about the world?

The lecture's simplest definition of vision is close to "what is where":

- what objects, people, lanes, signs, tumors, or digits are present
- where they are located in the image
- how they relate spatially to each other
- sometimes, what is likely to happen next

That last part matters for examples like driving. A visual system is not only
labeling pixels; it may need to predict motion, risk, or a future action.

### What Computers See

Images are numerical arrays.

- grayscale image: one value per pixel, so shape is basically `(height, width)`
- RGB image: three values per pixel, so shape is `(height, width, channels)`
- PyTorch convention for batches: `(batch, channels, height, width)`
- MNIST example: `(batch, 1, 28, 28)` because every digit is a 28x28 grayscale image

Important reminder:

- the network does not receive the idea of a "digit" or "face"
- it receives pixel intensities
- all semantic structure has to be learned from patterns in those numbers

This is why computer vision is a good test case for deep learning. The raw input
is low-level and high-dimensional, but the desired output can be very semantic:
digit identity, face/non-face, object box, steering angle, disease label, etc.

### Why Fully Connected Layers Are Not Enough For Images

I can technically flatten an image and feed it into a dense network:

- MNIST: `28 * 28 = 784` input features
- first dense layer with 128 hidden units: `784 * 128` weights before counting bias

That works for a small dataset like MNIST, but it is not the right inductive bias
for general images.

Problems with flattening:

- it destroys the explicit 2D neighborhood structure
- nearby pixels are treated the same as far-away pixels
- the same local feature in two different places needs to be relearned separately
- parameter count grows quickly as image size increases

The lecture's answer is convolution: learn local feature detectors and reuse the
same detector across the whole image.

### From Hand-Engineered Features To Learned Features

Older computer vision pipelines depended heavily on manual feature extraction:

- decide which patterns matter
- compute those features from the image
- pass the features into a classifier

Deep learning changes this pipeline. Instead of hand-writing edge, corner, or
texture detectors, a CNN learns useful feature detectors directly from data.

The hierarchy idea is important:

- early layers tend to detect simple local patterns like edges or corners
- middle layers can combine these into textures or parts
- deeper layers can represent more task-specific object structure

This connects back to Lecture 1: training is still "choose model, define loss,
optimize parameters." The difference is that the model architecture is now built
to respect the spatial structure of images.

### Convolution Operation

A convolution layer applies learned filters over local neighborhoods.

Useful terms:

- kernel/filter: small learned weight matrix, often `3x3` or `5x5`
- receptive field: the local input region that affects one output value
- feature map: the output produced by one learned filter across the image
- channels: stacked feature maps
- stride: how far the filter moves each step
- padding: extra border pixels added so spatial size can be controlled

Shape formula for one spatial dimension:

`output_size = floor((input_size + 2 * padding - kernel_size) / stride) + 1`

Applying this to the starter Lab 2 CNN:

- input width/height is `28`
- `3x3` convolution, stride `1`, padding `0`
- output size is `floor((28 + 0 - 3) / 1) + 1 = 26`

That matches the shape path I checked in the code: `(batch, 1, 28, 28)` becomes
`(batch, 24, 26, 26)` after the first convolution.

### Why Convolutions Help

The main advantages are:

- local connectivity: the model starts by looking at nearby pixels together
- parameter sharing: the same filter is reused at every image location
- translation handling: a feature can be detected even if it appears somewhere else
- compositionality: deeper layers combine lower-level features into richer ones

This is a better fit for images than treating every pixel as unrelated input.

### Nonlinearity And Pooling

A typical CNN block is:

`convolution -> nonlinearity -> pooling`

ReLU:

- keeps positive activations
- zeros out negative activations
- lets the network compose nonlinear functions instead of only linear filters

Pooling:

- reduces spatial size
- keeps a summary of local neighborhoods
- lowers compute and memory cost
- gives some robustness to small shifts in the input

Max pooling keeps the strongest activation in a local window. For image features,
that often means "did this feature show up somewhere in this small region?"

### CNN Architecture Pattern

The common pattern from the lecture:

- feature extractor: repeated convolution / activation / pooling layers
- classifier or task head: fully connected layers, softmax/logits, regression head, etc.

For MNIST:

- the feature extractor learns visual patterns useful for recognizing digits
- the head maps the extracted features to 10 digit logits

For other tasks:

- classification head: one label for the whole image
- regression head: continuous value, like steering angle
- detection head: object class plus bounding box coordinates
- segmentation head: label per pixel or region

The important mental model is modularity. CNNs are not only "image classifiers";
the convolutional backbone extracts spatial features, and the head determines the
task.

### Object Detection

Classification answers: "what is in this image?"

Detection has to answer more:

- what objects are present
- where each object is located
- possibly how confident the model is for each object

That means the output cannot just be one class vector. It needs localization
information, usually bounding boxes, plus class predictions.

This is the point where the loss function becomes more complex:

- classification loss for object identity
- regression-style loss for box coordinates
- sometimes additional terms for objectness / confidence

The lecture's broader point: the same deep learning pipeline can support many
vision tasks, but the output representation and loss have to match the task.

### Applications To Keep In Mind

The lecture uses computer vision examples to show why this is not just a toy
problem:

- medical imaging: detecting disease patterns from scans
- facial detection/recognition: finding and identifying faces
- autonomous driving: mapping visual input to driving decisions
- mobile photography: visual processing embedded into everyday devices

The uncomfortable part, which connects directly to Lab 2, is that high accuracy
is not enough. If the training data is biased or under-represents some groups,
the model can perform unevenly across those groups even if the average metric
looks good.

### How Lecture 3 Sets Up Lab 2

Lab 2 starts with MNIST because it is a controlled way to practice image
classification:

- image tensors
- flattening vs convolution
- logits over classes
- cross entropy
- train/test accuracy

Then it moves to facial detection and bias:

- the task changes from 10-way digit classification to face/non-face detection
- the social stakes are higher
- evaluation has to care about subgroup performance, not only average accuracy
- debiasing requires thinking about data distribution, not only model architecture

My current stopping point should be before the full training run:

- understand shapes
- inspect a real or synthetic batch
- define the baseline model carefully
- do not claim Lab 2 is complete until I train and evaluate both the MNIST models
  and the facial detection/debiasing section

### Lecture 3 Follow-Up: Details I Should Not Skip

After going back through the current Lecture 3 material and the beginning of
the official Lab 2 notebook, the most important thing is that the lecture is not
only saying "CNNs work better for images." It is giving a reason why the
architecture matches the structure of the data.

Images have two kinds of structure at the same time:

- local structure: neighboring pixels usually matter together
- global structure: the whole object or scene is built from many local patterns

A fully connected layer ignores the local structure when it flattens the image.
Flattening is not "wrong" mathematically, but it removes the explicit 2D
arrangement before the model has learned any visual features. A CNN delays that
flattening step. It first learns filters that slide over the image, so a useful
local detector can be reused at many positions.

That makes the convolution layer a stronger prior for images:

- each filter has a small receptive field
- the same filter weights are shared across spatial locations
- multiple filters create multiple feature maps
- early filters can respond to simple features
- deeper layers combine earlier features into more semantic representations

The parameter-sharing point is especially important. If a vertical edge matters
for a digit, the model should not need a totally separate detector for "vertical
edge in the top left" and "vertical edge in the bottom right." Convolution gives
the model a way to learn the detector once and apply it everywhere.

### Convolution Shapes And Parameter Counts

For PyTorch, a 2D convolution weight tensor has the conceptual shape:

`(out_channels, in_channels, kernel_height, kernel_width)`

So a layer with `in_channels=1`, `out_channels=24`, and a `3x3` kernel has:

- `24 * 1 * 3 * 3 = 216` kernel weights
- plus `24` bias terms if bias is enabled
- `240` trainable parameters total

That is much smaller than connecting every input pixel to every output location
with independent weights. The spatial output is still large, but the learned
filter parameters are shared.

The spatial size formula is still:

`output_size = floor((input_size + 2 * padding - kernel_size) / stride) + 1`

The design choices mean:

- larger kernels see more context but add parameters
- stride greater than 1 skips positions and shrinks the output
- padding can preserve edge information and control output size
- more output channels means more learned feature types

I should treat shape math as part of model design, not as bookkeeping after the
fact. If I cannot predict the shape before running the model, I probably do not
understand the architecture yet.

### Pooling Is A Tradeoff

Max pooling is not just "make the tensor smaller." The lecture motivation is
closer to: keep the strongest local evidence that a feature appeared somewhere
nearby.

Benefits:

- reduces spatial resolution
- lowers compute in later layers
- gives some tolerance to small shifts
- makes the next layer work with more compact feature maps

Cost:

- discards exact location information inside the pooling window
- can remove weak but meaningful evidence
- makes dense prediction tasks harder if used too aggressively

For MNIST classification, this tradeoff is fine because I only need one digit
label for the whole image. For segmentation or precise localization, I would
need to be more careful because the output needs spatial detail.

### Classification, Detection, And Segmentation

Lecture 3 separates several computer vision tasks that are easy to confuse:

- classification: assign one label or class distribution to the whole image
- regression: predict a continuous value from the image
- object detection: predict both object classes and object locations
- segmentation: assign labels at the pixel or region level

This matters because the network head and loss function must match the task.

For MNIST classification:

- input: image tensor
- output: 10 logits
- label: integer digit class
- loss: multiclass cross entropy

For face detection:

- input: image tensor
- output: face / not-face score
- label: binary class
- loss: binary classification loss, often logits plus sigmoid-aware loss

For object detection:

- output has to include class information and bounding-box information
- loss usually combines classification terms and coordinate-regression terms

For segmentation:

- output has to preserve spatial layout
- loss is often computed across pixels or regions

This is one of the clearer lecture-to-lab bridges: changing the task changes the
meaning of the final layer. The convolutional backbone extracts visual features,
but the head determines what question the model is answering.

### Lab 2 Part 1: What The Official Notebook Is Asking Me To Learn

The official PyTorch MNIST section uses the MNIST dataset as the first controlled
vision task:

- `60,000` training images
- `10,000` test images
- grayscale `28x28` handwritten digits
- classes `0` through `9`

The sequence of ideas is:

1. Load image-label pairs as a dataset.
2. Use a `DataLoader` to create batches.
3. Train a fully connected baseline.
4. Evaluate on held-out test data.
5. Build a CNN with convolution and pooling.
6. Compare the CNN against the dense baseline.

My repo is still before the real training step. The scripts so far only check
shape mechanics, model definitions, and forward/evaluation plumbing.

Implementation reminders:

- `transforms.ToTensor()` converts PIL images to tensors and scales pixel values
  to `[0, 1]`
- `nn.Flatten()` changes `(batch, 1, 28, 28)` into `(batch, 784)`
- `nn.Linear(28 * 28, 128)` is the first dense baseline layer
- the final MNIST layer should output 10 logits
- `nn.CrossEntropyLoss()` expects raw logits, not softmax probabilities
- use `model.train()` when optimizing and `model.eval()` when evaluating
- use `torch.no_grad()` or `torch.inference_mode()` during evaluation

Important distinction:

- logits are the model's raw class scores
- probabilities are useful for display after applying softmax
- the loss should receive logits directly

If I add softmax before `CrossEntropyLoss`, I am making optimization worse and
also mixing up model output with presentation output.

### Lab 2 Part 2: Reading Ahead Without Implementing Yet

The facial detection part raises a different issue from MNIST. MNIST asks:
"Can the model classify digits?" The face detection section asks a more applied
question: "Does the model work evenly across different groups?"

The official lab frames the data this way:

- positive training examples: face images, from CelebA
- negative training examples: non-face images, from ImageNet
- test set: face examples arranged for subgroup evaluation
- subgroup axes include skin tone and gender labels

The main concept is latent structure. Some important factors in a face image are
not necessarily explicit labels in the training loop:

- skin tone
- pose
- illumination
- occlusion
- background
- camera quality

A classifier can look strong on average while still failing for underrepresented
parts of the data distribution. That is why average accuracy is not enough for
this part of the lab.

The debiasing idea I need to understand before coding it:

- train a model that also learns a latent representation of the face data
- estimate which regions of latent space are underrepresented
- resample training examples so rare regions are seen more often
- evaluate subgroup performance, not just overall accuracy

I am not implementing DB-VAE yet. For now, I only want the MNIST evaluation
plumbing to be clean enough that later comparisons between baseline, CNN, and
debiased facial models are not confused by bad metric code.

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

## Lab 2

Lab 2 moves from sequence modeling into computer vision.

The official PyTorch path is split into:

- Part 1: MNIST digit classification
- Part 2: facial detection and debiasing

I am starting with Part 1, but only the mechanics for now. The main thing I want
to understand before training is how image-shaped tensors move through a CNN.

### First Concepts To Keep Straight

- images are represented as `(batch, channels, height, width)` in PyTorch
- MNIST images are grayscale, so the channel count starts at `1`
- a convolution learns local pattern detectors and outputs multiple feature maps
- max pooling reduces spatial size, which makes later layers cheaper
- the convolutional feature map has to be flattened before the fully connected classifier
- the final layer should output 10 logits, one for each digit class
- `CrossEntropyLoss` wants logits, not softmax probabilities

### Shape Path In The Starter CNN

For a synthetic MNIST-like batch shaped `(8, 1, 28, 28)`:

- `conv1`, 1 -> 24 channels with a 3x3 kernel: `(8, 24, 26, 26)`
- `pool1`, 2x2 max pooling: `(8, 24, 13, 13)`
- `conv2`, 24 -> 36 channels with a 3x3 kernel: `(8, 36, 11, 11)`
- `pool2`, 2x2 max pooling: `(8, 36, 5, 5)`
- flatten: `(8, 900)`
- classifier head: `(8, 10)`

This is enough for today. Next time, I should load real MNIST with
`torchvision`, run the evaluation plumbing on a small held-out batch, train the
fully connected baseline, and then train the CNN so I can compare how much
convolution helps.
