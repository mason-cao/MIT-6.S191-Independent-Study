# Course Notes

## Study Source And Plan

This repository is my independent-study notebook for MIT 6.S191. I rechecked
the active 2026 course page at `https://introtodeeplearning.com/` on June 8,
2026 and am using that schedule for these notes. The public materials still use
the 6.S191 label.

The official public sequence I am following is:

- Lecture 1: Intro to Deep Learning, Mar. 30, 2026
- Lecture 2: Deep Sequence Modeling, Apr. 6, 2026
- Software Lab 1: Deep Learning in Python and Music Generation
- Lecture 3: Deep Computer Vision, Apr. 13, 2026
- Lecture 4: Deep Generative Modeling, Apr. 20, 2026
- Software Lab 2: Facial Detection Systems
- Lecture 5: Deep Reinforcement Learning, Apr. 27, 2026
- Lecture 6: New Frontiers, May 4, 2026
- Software Lab 3: Fine-Tune an LLM, You Must!
- Lecture 7: The Three Laws of AI, May 11, 2026
- Lecture 8: AI for Science, May 18, 2026
- Final project work
- Lecture 9: Secrets to Massively Parallel Training, May 25, 2026

My standard for these notes: each section should teach the subject well enough
that I could rebuild the code or explain the concept without reopening the
slides immediately. That means I need definitions, equations, shape checks,
failure modes, and lab connections, not only topic lists.

## Lecture 1: Intro to Deep Learning

Lecture 1 is the foundation for the whole course. The main pattern is simple:
represent inputs as tensors, build a differentiable model, define a loss, and
use gradients to change the parameters so the model's predictions improve.

The short version of supervised deep learning is:

1. Choose a parameterized function `f(x; theta)`.
2. Run data through the function to get a prediction `y_hat`.
3. Compare `y_hat` with the target `y` using a loss.
4. Use backpropagation to compute gradients of the loss with respect to the
   parameters.
5. Use an optimizer to update the parameters.

That loop is the same whether the model is a single perceptron, a CNN, an RNN,
a VAE encoder, or a transformer. Later lectures change the architecture and the
data type, but this training loop stays underneath everything.

### Core Setup

The supervised learning setup has a few objects I need to keep separate:

- `x`: input features, such as a vector of measurements, an image tensor, or a
  token sequence
- `y`: the target output supplied by the dataset
- `y_hat`: the model's prediction
- `theta`: all trainable parameters, usually weights and biases
- `L(y_hat, y)`: loss for one example or one batch
- `J(theta)`: objective over the dataset or minibatch

The basic neuron equation is:

`y_hat = g(Wx + b)`

or, in the batched convention I used in the Lab 1 scripts:

`Y_hat = g(XW + b)`

where:

- `X` has shape `(batch_size, input_features)`
- `W` has shape `(input_features, output_features)`
- `b` has shape `(output_features,)`
- `XW + b` has shape `(batch_size, output_features)`

The bias is broadcast across the batch. That broadcasting is convenient, but I
should still be able to say what expansion is happening. In Lab 1,
`02_manual_perceptron_forward.py` checks this explicitly before applying the
sigmoid activation.

The nonlinearity `g` is essential. If I stack only linear layers, the whole stack
collapses into one larger linear map. Nonlinear activations are what let deeper
networks represent curved decision boundaries and complicated functions.

### Perceptron

A perceptron is a single trainable unit:

`z = w dot x + b`

`y_hat = g(z)`

For binary classification with a sigmoid, `y_hat` can be interpreted as a
probability-like score between `0` and `1`. The decision boundary before the
sigmoid is the set of inputs where:

`w dot x + b = 0`

In two input dimensions, that boundary is a line. The weights determine the
orientation of the line, and the bias shifts it. That makes the perceptron a
useful first model: the geometry is still visible.

For multiple outputs, I stack several perceptrons together. Each output unit gets
its own weight vector and bias value. In matrix form, that becomes one dense
layer. The single-neuron idea did not disappear; it got vectorized.

The Lab 1 detail I want to remember:

- my manual convention stores weights as `(input_features, output_features)`
- `torch.nn.Linear` stores weights as `(output_features, input_features)`
- copying from the manual version into `nn.Linear` therefore requires a
  transpose, which `04_torch_nn_bridge.py` checks

### From One Perceptron to a Network

A feed-forward neural network composes layers:

`h_1 = g_1(W_1x + b_1)`

`h_2 = g_2(W_2h_1 + b_2)`

`y_hat = g_3(W_3h_2 + b_3)`

The intermediate vectors `h_1`, `h_2`, etc. are hidden representations. The
network is learning a sequence of transformations from raw input space into a
space where the task is easier.

The important distinction:

- parameters are the learned weights and biases
- activations are the values produced for a specific input
- architecture is the pattern of layers and connections
- loss is the training signal that says whether the current behavior is good

When debugging, I should ask which of those four things is wrong. A shape error
is usually architecture or tensor plumbing. Bad training may be loss,
optimization, data, initialization, or architecture.

### Loss

The model prediction is not enough by itself. Training needs a scalar objective
that measures how wrong the prediction is.

The empirical risk form is:

`J(theta) = (1/n) * sum_i L(f(x_i; theta), y_i)`

What this means:

- `L` measures the error for one example or batch element
- the sum averages that error across the training data
- `theta` is changed to reduce this objective

The loss has to match the output type:

- regression: mean squared error or mean absolute error can make sense
- binary classification: binary cross entropy, usually with logits in real code
- multiclass classification: cross entropy over class logits
- sequence prediction: cross entropy at each time step
- generative models: often combine reconstruction, likelihood, adversarial, or
  regularization terms

One mistake to avoid: applying softmax before `CrossEntropyLoss` in PyTorch.
`CrossEntropyLoss` expects raw logits and applies the numerically stable log
softmax internally.

### Optimization

Gradient descent updates parameters by moving opposite the gradient:

`theta <- theta - eta * grad_theta J(theta)`

Things that matter here:

- the gradient points in the direction of steepest increase
- subtracting the gradient moves downhill
- `eta` is the learning rate
- too large a learning rate can overshoot or diverge
- too small a learning rate can make training crawl or appear stuck

In actual PyTorch code, this becomes:

1. `optimizer.zero_grad()`
2. compute predictions
3. compute scalar loss
4. `loss.backward()`
5. `optimizer.step()`

Skipping `zero_grad()` is a classic bug because PyTorch accumulates gradients by
default. That accumulation is useful for some advanced workflows, but it is not
what I want in a normal minibatch training loop.

### Backpropagation

Backpropagation is the chain rule applied efficiently to a computation graph.
The forward pass builds intermediate values. The backward pass computes how the
final loss changes with respect to each parameter.

For a sigmoid perceptron with squared error:

`z = XW + b`

`y_hat = sigmoid(z)`

`L = 0.5 * sum((y_hat - y)^2)`

The local derivative pieces are:

- `dL/dy_hat = y_hat - y`
- `dy_hat/dz = y_hat * (1 - y_hat)`
- `dL/dz = dL/dy_hat * dy_hat/dz`
- `dL/dW = X^T @ dL/dz`
- `dL/db = sum over batch of dL/dz`

`03_manual_gradient_vs_autograd.py` compares those manual gradients against
PyTorch autograd. That script keeps autograd from becoming a black box. It is
doing the same chain-rule bookkeeping, just at graph scale.

The mental model:

- forward pass: values flow from input to prediction to loss
- backward pass: sensitivities flow from loss back to parameters
- parameters are updated after the gradients are available

### Why This Lecture Matters for the Lab

Lab 1 Part 1 is Lecture 1 turned into code:

- tensors store data and model values
- shapes determine whether operations mean what I think they mean
- matrix multiplication implements dense layers
- activations introduce nonlinearity
- losses turn prediction quality into a scalar
- autograd computes gradients
- optimizers update parameters
- `torch.nn.Module` packages model state and forward behavior

The reason I rewrote the Lab 1 scripts in small pieces is that each script
isolates one layer of abstraction:

- `01_tensor_mechanics.py`: rank, shape, slicing, broadcasting
- `02_manual_perceptron_forward.py`: `XW + b` and sigmoid by hand
- `03_manual_gradient_vs_autograd.py`: manual gradients versus autograd
- `04_torch_nn_bridge.py`: manual perceptron versus `torch.nn.Linear`
- `05_tensor_computation_graphs.py`: small computation graphs
- `06_models_and_autograd.py`: custom modules, `nn.Sequential`, scalar gradient
  descent

If I understand those six scripts, the later labs have a much clearer starting
point. CNNs, LSTMs, VAEs, and LoRA-tuned language models all reuse the same
core training mechanics.

## Lecture 2: Deep Sequence Modeling

Lecture 2 changes the shape of the learning problem. In Lecture 1, I could
think about a fixed input vector going through a feed-forward network. In
sequence modeling, the input is ordered:

`x_1, x_2, ..., x_T`

The order is part of the data. If I shuffle the tokens in a sentence, frames in
a video, or notes in a melody, I have changed the example.

The lecture's main question is: how can a neural network use information from
earlier time steps when it makes a decision later?

### What Changes in Sequence Modeling

Sequence data has several complications that do not appear in ordinary
fixed-vector classification:

- examples can have different lengths
- the same value can mean different things depending on position
- local context and long-range context can both matter
- the model may need to output one value, one value per time step, or a new
  generated sequence
- training often reuses the same parameters across many time steps

Examples from the lecture:

- language modeling: predict the next word or token
- sentiment classification: map a whole sentence to a label
- speech recognition: map audio frames to text
- time-series forecasting: predict future values from past observations
- music generation: predict the next character in ABC notation

I need to keep the input/output patterns straight:

- one-to-one: fixed input to fixed output, like normal image classification
- many-to-one: sequence to one label, like sentiment
- one-to-many: one conditioning input to a generated sequence
- many-to-many, aligned: output at each time step
- many-to-many, unaligned: sequence translation where input/output lengths may
  differ

Lab 1 uses a many-to-many aligned training objective. Every input character gets
a next-character target.

### RNN Idea

The basic recurrent neural network adds a hidden state. The hidden state is the
model's running summary of the sequence so far.

At time `t`:

- input: `x_t`
- previous hidden state: `h_(t-1)`
- new hidden state: `h_t`
- optional output: `y_hat_t`

A simple recurrence looks like:

`h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)`

`y_hat_t = W_hy h_t + b_y`

The important design choice is parameter sharing. The same `W_xh`, `W_hh`, and
`W_hy` are reused at every time step. The model is not learning separate weights
for "first character," "second character," and "third character." It is learning
one update rule that can be applied repeatedly.

That makes RNNs natural for variable-length data. The model can process a
sequence by applying the same cell until it reaches the end.

### Unrolling Across Time

The unrolled computation graph is the clearest way to think about an RNN.
Instead of imagining a loop, I can draw one copy of the RNN cell per time step:

`x_1 -> h_1 -> y_hat_1`

`x_2 + h_1 -> h_2 -> y_hat_2`

`x_3 + h_2 -> h_3 -> y_hat_3`

and so on.

Those are not separate learned cells. They are repeated applications of the same
cell. Unrolling only makes the computation graph visible.

Two consequences matter:

- information from earlier inputs reaches later predictions through the hidden
  state
- gradients from later losses have to travel backward through the unrolled time
  steps

### BPTT

Backpropagation through time is normal backpropagation applied to the unrolled
sequence graph. The new complication is parameter sharing. Since the same
weights are used at each time step, the final gradient for a shared weight is
the sum of its contributions across time.

For language modeling or music modeling, the loss is often computed at every
time step:

`L = sum_t cross_entropy(y_hat_t, y_t)`

Then BPTT asks how each shared parameter affected all of those losses.

Practical implications:

- longer sequences require more memory because the graph stores more
  intermediate states
- longer sequences create longer gradient paths
- truncating BPTT lowers compute but limits how far credit assignment can reach
- exploding gradients may need gradient clipping
- vanishing gradients make long-range learning weak

### Long-Term Dependencies / Vanishing Gradients

The long-term dependency problem is the main weakness of a plain RNN. If a model
needs information from many steps earlier, the learning signal has to pass
through many recurrent updates.

During backpropagation, that means repeated multiplication by derivatives. If
those factors are often less than `1`, the gradient shrinks as it moves
backward. If they are often greater than `1`, it can explode.

What this looks like in practice:

- the model learns local syntax before global structure
- early context can fade out of the hidden state
- generated text or music may sound locally plausible but drift over longer
  spans
- training may be unstable unless gradients are clipped or the architecture is
  changed

This connects directly to Lab 1. A character-level music model can learn common
local transitions before it learns complete ABC structure. A generated sample
may produce reasonable note-like characters but fail to close sections or keep a
consistent key.

### LSTM Intuition

The LSTM is the lecture's answer to the plain RNN memory problem. It adds a cell
state with gates that control what gets written, erased, and exposed.

The pieces I need to remember:

- forget gate: decides what old cell-state information to keep
- input gate: decides what new candidate information to write
- output gate: decides what part of the cell state becomes hidden output
- cell state: a longer-running memory path than the normal hidden state

The exact equations are less important than the reason for the design. A plain
RNN updates its hidden state in one shot. An LSTM gives the model separate
controls for remembering and forgetting, which makes it easier to preserve
useful information across longer spans.

For the Lab 1 LSTM:

- the embedding layer converts character IDs into vectors
- the LSTM updates hidden and cell states over the character sequence
- the final linear layer turns each time step into vocabulary logits
- cross entropy trains each position to predict the next character

### Sequence Tasks to Keep Straight

Next-token prediction:

- input: prefix so far
- output: probability distribution over the next token
- Lab 1 example: previous ABC characters -> next ABC character

Sequence classification:

- input: full sequence
- output: one label
- example: sentence -> sentiment

Sequence-to-sequence prediction:

- input: one sequence
- output: another sequence
- example: source sentence -> translated sentence

Autoregressive generation:

- train with real previous tokens
- generate by sampling one token, appending it, and repeating
- output quality depends on the model and on sampling settings such as
  temperature

### Attention And Transformers

Attention addresses two RNN limitations:

- the hidden-state bottleneck
- the lack of parallelism across time

In an attention layer, a token can compare itself with other tokens and decide
which ones are relevant. The rough idea is:

- query: what this position is looking for
- key: what each position offers for matching
- value: the information each position can pass along
- attention weights: how much each position should contribute

Self-attention means every token attends to tokens in the same sequence. This
lets the model build direct pairwise connections instead of forcing all context
through a single recurrent hidden state.

The tradeoff is different from an RNN:

- RNN compute grows step by step and has a natural streaming structure
- self-attention can process a sequence more parallelly
- full self-attention has a cost that grows with pairwise token interactions
- transformers need positional information because attention alone does not know
  token order

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

- grayscale image: one value per pixel, so shape is `(height, width)`
- RGB image: three values per pixel, so shape is `(height, width, channels)`
- PyTorch convention for batches: `(batch, channels, height, width)`
- MNIST example: `(batch, 1, 28, 28)` because every digit is a 28x28 grayscale image

Important reminder:

- the network does not receive the idea of a "digit" or "face"
- it receives pixel intensities
- all semantic structure has to be learned from patterns in those numbers
- the tensor layout matters before the model sees anything

My shape rule for PyTorch vision code:

- keep batches as `(N, C, H, W)`
- check `C` before the first convolution
- compute the flatten size before writing the first dense layer
- do not assume a tensor is image-shaped just because it has four dimensions

Computer vision is a good stress test for deep learning because the raw input is
low-level and high-dimensional, while the desired output can be semantic: digit
identity, face/non-face, object box, steering angle, disease label, etc.

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

For my own code, the difference between the dense baseline and the CNN is not
just accuracy. It is the prior each model starts with:

- dense baseline: every pixel can connect independently to every hidden unit
- CNN: nearby pixels are processed together first
- dense baseline: the same edge in two locations needs separate weights
- CNN: one learned filter can respond across the image

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

The parameter count is separate from the output tensor size. A `3x3` convolution
with `1` input channel and `24` output channels has `24 * 1 * 3 * 3 = 216`
kernel weights, plus `24` biases if bias is on. The layer produces many output
values because the same small filter is reused at many spatial locations.

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

In this repo, the MNIST scripts now cover the path from shape checks to an
actual dense-vs-CNN training comparison. The remaining boundary is the face
dataset path: real CelebA/ImageNet/PPB numbers require the official notebook,
datasets, and a GPU-backed run.

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

My repo now has this whole sequence represented. The first three Lab 2 scripts
build up the mechanics slowly, and `04_mnist_training_comparison.py` performs
the actual training comparison between the dense baseline and the CNN.

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

What I want from the MNIST comparison:

- dense model: useful baseline, but it ignores image locality after flattening
- CNN: should learn local stroke features with fewer spatial assumptions broken
- train metrics: show whether the model can fit the training distribution
- test metrics: show whether those learned features transfer to held-out digits
- confusion matrix: shows which digit pairs the model confuses, not just the
  average accuracy

### Lab 2 Part 2: Facial Detection And Debiasing

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

This is now represented locally in `05_facial_debiasing_mechanics.py`. The
script does not download CelebA/ImageNet/PPB, so I should not treat its numbers
as real fairness results. It does check the moving parts that matter for the
official lab: binary CNN logits, `BCEWithLogitsLoss`, grouped face-probability
evaluation, VAE reconstruction/KL losses, reparameterization, and adaptive
resampling over latent variables.

The AIES paper behind the lab makes the key distinction clearer: class balance
is not enough. The face class itself can be internally imbalanced across latent
features such as skin tone, pose, illumination, and occlusion. A detector can
have good overall accuracy while still showing high variance across subgroups.

## Lecture 4: Deep Generative Modeling

Lecture 4 matters for Lab 2 because the facial debiasing section is not only a
computer vision classifier. It uses a generative model, specifically a VAE, to
learn hidden structure in the face data and then uses that structure to change
the sampling distribution during training.

### Generative Vs. Discriminative Modeling

The discriminative models from Lectures 1-3 learn a mapping from input to label:

- digit image -> digit class
- face image -> face/not-face
- sequence prefix -> next token

Generative modeling asks a different kind of question. Instead of only learning
`p(y | x)`, the model tries to learn something about the data distribution
itself. The model should be able to represent what valid examples look like,
sample new examples, or reconstruct an input through a compressed latent code.

This changes the mental model:

- classifier: "what label should this input get?"
- generator: "what kind of data could have produced this example?"
- representation learner: "what latent factors explain variation in this data?"

That last framing is exactly the bridge to DB-VAE. The facial detection lab
cares about hidden factors like skin tone, pose, lighting, occlusion, and camera
quality because those factors can be unevenly represented in the training data.

### Autoencoders

An autoencoder has two main pieces:

- encoder: maps input `x` into a lower-dimensional representation `z`
- decoder: maps `z` back into a reconstruction `x_hat`

The basic training objective is reconstruction quality. If the reconstruction is
close to the original input, the latent code must have preserved useful
information.

Important detail:

- the model is not given labels for the latent variables
- the structure is learned from the need to compress and reconstruct examples
- the latent space can become useful even when no one manually annotated every
  factor of variation

For images, the encoder is often convolutional because it has to extract visual
features, and the decoder often uses upsampling or transposed convolutions to
turn latent vectors back into image-shaped tensors.

### Why Plain Autoencoders Are Not Enough

A regular autoencoder can learn compressed codes, but its latent space is not
automatically organized in a way that supports sampling. Two nearby points may
not necessarily decode to two similar, realistic images. There may be gaps or
irregular regions where the decoder has not learned meaningful outputs.

For generative modeling, I need a latent space with a useful distribution:

- I should be able to sample a latent vector and decode it
- nearby latent vectors should often correspond to related examples
- the latent space should not become a scattered lookup table

This is where variational autoencoders come in.

### Variational Autoencoders

A VAE changes the encoder output. Instead of producing one deterministic vector,
the encoder predicts parameters of a distribution:

- mean vector `mu`
- log variance or log standard deviation vector, depending on implementation

Then the model samples a latent vector `z` from that distribution and decodes
`z` into a reconstruction.

The VAE objective has two pressures:

- reconstruction loss: make `x_hat` close to `x`
- latent regularization: make the learned latent distribution stay close to a
  simple prior, usually a unit Gaussian

The KL term has a specific job. If the encoder predicts a distribution that
drifts far away from the unit Gaussian prior, random samples from the prior may
decode badly. The KL penalty keeps the latent space smoother and more usable for
sampling.

In my PyTorch script, I wrote the per-example VAE loss as:

`L_VAE = reconstruction_loss + c * L_KL`

where:

- reconstruction loss is mean absolute pixel difference
- `L_KL` penalizes the learned latent distribution for drifting away from the
  Gaussian prior
- `c` is a small coefficient controlling how strongly the latent space is
  regularized

This is the key tradeoff:

- too little KL pressure: reconstructions may improve, but the latent space can
  become messy
- too much KL pressure: the latent space is regularized, but reconstructions may
  lose detail

For a diagonal Gaussian encoder, the common closed form is:

`L_KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)`

The exact implementation may store `logvar` or `logsigma`, so I need to check
which convention a script uses before copying the formula.

### Reparameterization Trick

The sampling step seems like it would break backpropagation because random
sampling is not a normal deterministic layer. The reparameterization trick fixes
that by separating the randomness from the learned parameters.

Instead of sampling directly in an opaque way, the model samples noise:

`epsilon ~ N(0, I)`

and computes:

`z = mu + exp(0.5 * logsigma) * epsilon`

The random part is `epsilon`, while `mu` and `logsigma` are still part of a
differentiable computation. That means gradients can flow back into the encoder.

This is one of those ideas that is easy to memorize but more useful to
understand mechanically:

- the encoder controls the location and spread of the latent distribution
- randomness gives the model a sampled latent code during training
- the algebra keeps the sampled code differentiable with respect to encoder
  outputs

### DB-VAE Connection

The debiasing VAE in Lab 2 modifies the normal VAE setup because it also has to
solve a supervised binary classification task.

The encoder outputs:

- `y_logit`: supervised face/not-face prediction
- `z_mean`: mean of the unsupervised latent distribution
- `z_logsigma`: spread of the unsupervised latent distribution

The decoder reconstructs the image from a sampled latent vector. The important
detail from the lab is that the VAE reconstruction/latent loss is only applied
to face examples. Non-face examples still train the classifier, but the model is
not trying to learn the latent structure of "all non-faces." The debiasing
target is variation within the face class.

The total DB-VAE loss can be thought of as:

`classification loss + face_indicator * VAE loss`

where `face_indicator` is `1` for face examples and `0` for non-face examples.

That design makes sense:

- every example helps train face/not-face classification
- only face examples shape the latent space used for debiasing
- the learned latent space is then used to find underrepresented face features

The paper frames this as a semi-supervised model: one encoder output is
explicitly supervised for the classification task, while the other latent
variables are learned from reconstruction. That is why DB-VAE sits between
ordinary supervised classification and unsupervised representation learning.

### Adaptive Resampling

The resampling procedure is the most important algorithmic idea in the lab.

The DB-VAE estimates latent variables for face images. For each latent dimension,
the training examples can be binned into a histogram. Dense bins correspond to
features that appear often in the training set. Sparse bins correspond to rarer
latent regions.

The lab's strategy is:

1. Encode face images into latent means.
2. Estimate how densely each latent region is represented.
3. Assign higher sampling probability to examples in sparse regions.
4. Train future batches with this updated sampling distribution.

A simple way to remember the sampling rule:

- common latent region -> high density -> lower sampling probability
- rare latent region -> low density -> higher sampling probability

The goal is not to synthesize new faces. The goal is to make future training
batches draw more often from real examples that represent rare latent features.

This is not the same as simply balancing class labels. The face/not-face classes
could be balanced while the face class is still internally skewed toward certain
skin tones, poses, lighting conditions, or occlusions.

What I like about the method:

- it does not require manually labeling every latent factor
- it can respond to hidden structure inside a class
- it integrates debiasing into the training loop instead of only post-processing
  outputs

What I should be careful about:

- a learned latent space is not automatically fair or interpretable
- resampling can trade off subgroup performance against other metrics
- the evaluation still needs a balanced, labeled test set across the sensitive
  categories I care about
- synthetic smoke tests prove the code path, not real-world fairness

For this repo, the local DB-VAE script is a mechanics check. It shows that I can
compute the classifier loss, reconstruction loss, KL loss, reparameterized
sample, grouped evaluation, and latent-resampling probabilities. It does not
show that a real deployed face detector is fair.

## Lecture 5: Deep Reinforcement Learning

Lecture 5 shifts from static datasets to dynamic environments. In the earlier
lectures, I mostly had a fixed training set and a model that tried to predict a
label, reconstruct an input, or generate the next token. Reinforcement learning
changes the setup completely:

- the learner is now an agent that takes actions
- those actions change what data the agent will see next
- success is not judged only by the current step, but by future consequences
- the goal is to maximize reward over time, not just fit one supervised target

The opening slides frame RL as a general way to learn in environments that keep
changing while the learner interacts with them. The examples span:

- robotics
- game play and strategy
- real-world sequential control problems
- even language-style sequential decision making

That broad framing matters. RL is not "the Atari lecture." Atari is one clean
example, but the real point is learning how to act when present choices affect
future states and future opportunities.

### RL Compared To Other Learning Problems

One of the early slides contrasts the three major classes of learning problems.
This is a useful checkpoint because it explains exactly why RL needs different
math and different algorithms.

Supervised learning:

- data: `(x, y)`
- `x` is the input and `y` is the label
- goal: learn a mapping `x -> y`

Unsupervised learning:

- data: `x`
- there are no labels
- goal: learn underlying structure in the data

Reinforcement learning:

- data is built from interaction, especially state-action experience
- the target is not a fixed label attached to the current example
- goal: maximize future rewards over many time steps

The apple example from the slide is simple but surprisingly helpful:

- supervised: "this thing is an apple"
- unsupervised: "this thing is like the other thing"
- RL: "eat this thing because it will keep you alive"

That last version is fundamentally about consequences. The right action is not
defined by a human label on the current input. It is defined by what happens
after the action is taken.

### Agent And Environment

The lecture builds RL from a small loop:

- agent
- environment

The agent is the learner / decision-maker. The environment is the world the
agent operates in.

Important distinction from the earlier lectures:

- in supervised learning, the dataset is fixed before learning starts
- in RL, the agent acts in the environment and partly creates its own future
  experience

This immediately introduces feedback. The learner is no longer a passive system
that receives examples. It is an active system whose behavior changes the next
example distribution.

### Actions

The first outward signal from the agent is the action:

- notation: `a_t`
- meaning: the move the agent makes at time `t`

The lecture also defines the action space:

- `A` = the set of all actions the agent can take

This is worth keeping explicit because later algorithms depend heavily on
whether the action space is:

- small and discrete
- large
- continuous
- structured or constrained

For the conceptual slides, the main point is simpler: RL is about choosing
actions, not only predicting labels.

### Observations And State

After the agent acts, the environment responds. The slides first label this as
an observation channel back to the agent, then refine it into a next-state
view:

- observation: what the environment reveals after the action
- state change: `s_(t+1)`

The slide definition of state is essentially:

- state = the situation the agent perceives

I want to keep the practical interpretation straight:

- the state should contain enough information to support a good next decision
- the next state depends on both the previous situation and the action taken
- the RL problem is sequential because the state keeps evolving over time

This is the first major source of difficulty. In a sequence of actions, one
mistake can push the agent into a much worse future state distribution.

### Reward

The second signal returned by the environment is reward:

- notation: `r_t`
- definition from the slide: feedback that measures the success or failure of
  the agent's action

This is one of the most important conceptual differences from supervised
learning.

- a label says what the answer should be for this example
- a reward says how good or bad the consequence of an action was

Rewards can also be:

- delayed
- sparse
- noisy
- locally misleading

That means the agent often has to learn from incomplete feedback. A single step
reward does not necessarily reveal whether the action was globally smart.

### Return: Total Future Reward

The lecture then shifts from immediate reward to total reward, also called the
return.

The slide writes the total reward from time `t` onward as:

`R_t = sum_(i=t)^infinity r_i`

and then expands it as:

`R_t = r_t + r_(t+1) + ... + r_(t+n) + ...`

This is the quantity the agent actually cares about. The action at time `t`
should not be judged only by `r_t`, but by how it shapes the whole reward
stream that follows.

This is the core RL mental model:

- immediate reward can be small while long-term return is large
- immediate reward can look good while long-term return is terrible
- planning means acting for return, not just the next reward

### Discounting

Because future rewards can extend far into the future, the lecture introduces a
discounted return:

`R_t = sum_(i=t)^infinity gamma^i r_i`

with:

- `gamma` = discount factor
- `0 < gamma < 1`

The slide expansion makes the weighting idea explicit:

- near-term rewards count more
- farther rewards are progressively down-weighted

The practical reasons for discounting are:

- it keeps infinite-horizon sums controlled
- it encodes a preference for sooner reward over much later reward
- it makes learning and optimization more stable than treating all future steps
  equally forever

At a conceptual level, `gamma` sets the planning horizon:

- small `gamma` -> more short-term behavior
- `gamma` close to `1` -> more long-horizon behavior

### The Q-Function

Once return is defined, the lecture introduces the Q-function:

`Q(s_t, a_t) = E[R_t | s_t, a_t]`

This is one of the main objects of the whole lecture.

What it means:

- start in state `s_t`
- take action `a_t`
- then look at the expected total future reward that follows

So the Q-function measures how good a specific action is in a specific state.

The slide's wording is effectively:

- the Q-function captures the expected total future reward an agent can receive
  by taking a certain action in a certain state

This is the right way to think about it:

- state alone is not enough
- action alone is not enough
- RL needs a value attached to the pair `(state, action)`

### Why Q-Values Are Hard

The Breakout slide makes a good intuitive point: humans are often bad at
accurately estimating Q-values by inspection.

That is true because:

- the best move depends on how the ball trajectory will unfold later
- the consequence of an action may only become clear after several more steps
- small geometric differences can produce very different future reward

So even in a simple arcade game, "which action is better right now?" is really
a long-horizon prediction problem.

### From Q-Function To Policy

Knowing Q-values is not the end goal. The agent ultimately needs a policy:

- notation: `pi(s)`
- meaning: a rule that maps a state to an action choice

The lecture states the core strategy for a value-based policy:

`pi*(s) = argmax_a Q(s, a)`

So if I know the Q-function, I can choose the action with the highest expected
future return.

This gives a clean separation:

- the Q-function scores actions
- the policy uses those scores to decide what to do

### Two Main Deep RL Directions

The lecture then organizes deep reinforcement learning algorithms into two broad
families.

Value learning:

- learn `Q(s, a)`
- act by choosing `a = argmax_a Q(s, a)`

Policy learning:

- learn `pi(s)`
- sample or choose actions directly from that policy

This split is helpful because it previews the rest of the lecture:

- value-based methods focus on estimating action values well
- policy-based methods focus on directly optimizing the decision rule
- later deep RL methods combine the two ideas in different ways

### Markov Decision Process

The lecture then makes the sequential structure more explicit by writing the RL
interaction as a trajectory:

`s_0, a_0, r_1, s_1, a_1, r_2, ..., s_(n-1), a_(n-1), r_n, s_n`

with a terminal state at the end of the episode.

This is the Markov Decision Process view of the problem. The point is not just
to introduce new notation. It is to make clear what kind of object RL actually
learns from:

- the agent is not seeing isolated examples
- it is seeing a sequence of states, actions, and rewards
- every action affects the next state in the trajectory
- rewards are attached to transitions over time, not only to static labels

What I want to remember from this slide:

- RL training data is temporal
- it is generated by interaction
- one action can change every state distribution that follows

### Major Components Of An RL Agent

The lecture says an RL agent may include one or more of these components:

- policy: the agent's behavior function
- value function: how good each state and/or action is
- model: the agent's representation of the environment

This is a helpful organizing slide because it clarifies that not all RL methods
look the same internally.

Some algorithms mostly emphasize:

- policy learning

Others emphasize:

- value estimation

And some methods also learn or use:

- an explicit model of environment dynamics

This matters because "reinforcement learning" is not one algorithm. It is a
family of ways to combine acting, evaluating, and updating.

### Reward Design Changes Behavior

The robot-in-a-room slides are useful because they show that the reward
function is not a minor detail. It directly changes the optimal strategy.

The setup has:

- actions: `UP`, `DOWN`, `LEFT`, `RIGHT`
- stochastic dynamics:
  `UP` means 80% move up, 10% move left, 10% move right
- terminal rewards:
  `+1` in one cell and `-1` in another
- step costs that are varied across slides

The lecture changes the per-step reward across several slides:

- `-2`
- `-0.1`
- `-0.04`
- `-0.01`
- `+0.01`

That is a good reminder that the reward function shapes behavior very strongly.

If each step is very costly:

- the agent is pushed toward short, direct behavior

If each step is only mildly costly:

- the agent may prefer safer but longer paths

If each step is slightly positive:

- the agent can be incentivized to avoid ending the episode at all

This is one of the most practically important RL lessons in the lecture. A bad
reward function can produce a bad policy even if the optimization is working
perfectly.

### Recursive Return And Bellman-Style Reasoning

One of the most important slides rewrites discounted future reward as:

`R_t = r_t + gamma * r_(t+1) + gamma^2 * r_(t+2) + ...`

and then explicitly as:

`R_t = r_t + gamma * R_(t+1)`

This recursive identity is the heart of value-based RL.

The slide itself presents it as a property of discounted return. My
interpretation is that this is exactly what makes Bellman-style updates
possible:

- current value can be written in terms of immediate reward plus discounted
  future value
- long-horizon reasoning can be turned into repeated local updates
- the agent does not need to solve the whole future from scratch at every step

That recursive decomposition is the main bridge from the abstract return
definition to actual learning algorithms.

### Q-Learning

The lecture then introduces Q-learning more directly.

Important points from the slide:

- `Q^pi(s, a)` is the state-action value function
- it is the expected return when starting in `s`, taking action `a`, and then
  following policy `pi`
- Q-learning uses experience to estimate the action values that maximize future
  reward
- it directly approximates `Q*`
- it is independent of the policy currently being followed
- the main requirement is to keep updating each `(s, a)` pair

That "independent of the policy being followed" point is a big deal. It means
Q-learning is fundamentally off-policy:

- behavior policy: how the agent is actually exploring
- target object: the optimal action-value function

So the agent can behave somewhat noisily while still trying to estimate the
best possible future values.

The lecture slide labels the update as tied to the Bellman optimality equation.
In practice, the update target is the familiar idea:

- current estimate should move toward
  immediate reward + discounted best next-state value

That is the core Q-learning intuition even when the exact implementation
changes.

### Exploration Vs. Exploitation

The lecture then addresses a problem that appears immediately once I try to use
Q-values for action selection.

If I always act greedily:

- I may keep choosing the action that currently looks best
- but I may never discover a better action

The slide states the issue very clearly:

- a deterministic / greedy policy will not explore all actions
- at the beginning I do not know anything about the environment
- I need to try actions to discover which ones are actually good

The lecture's fix is the standard `epsilon`-greedy policy:

- with probability `1 - epsilon`, take the greedy action
- otherwise, choose a random action
- slowly decay `epsilon` toward `0`

This gives a clean tradeoff:

- early training: more exploration
- later training: more exploitation of what has been learned

### Why Tabular Value Iteration Breaks

The value-iteration and representation slides are useful because they show why
plain tabular RL does not scale.

The lecture first gives a small table example where value iteration is easy to
write down. But then it immediately points out why that breaks in realistic
domains:

- very limited numbers of states/actions only
- cannot generalize to unobserved states

The Breakout example makes the scaling problem concrete:

- state = screen pixels
- resized image = `84 x 84`
- use 4 consecutive images
- grayscale with 256 levels

The slide's point is that a naive Q-table would need an absurd number of rows.
That is exactly why representation learning becomes necessary. I do not just
need a better RL objective; I need a compact function approximator that can
share statistical strength across similar states.

### Deep Q-Learning

This is where deep learning enters the RL story directly.

The DQN slide states the core idea very simply:

- use a function with parameters to approximate the Q-function
- this function can be linear or nonlinear
- with a neural network, it becomes a Q-network

This changes the problem from:

- store one Q-value per exact `(state, action)` table entry

to:

- learn a parameterized mapping that predicts action values from high-dimensional
  inputs

That gives two major benefits:

- it can handle huge state spaces like images
- it can generalize across states instead of memorizing each one independently

The Atari slide is the canonical example:

- raw visual input
- discrete action set
- neural network predicts action values

### DQN Training Objective

The 2019 deck makes the training view more explicit by showing a target and a
prediction inside a squared-error loss.

At a high level, the network is trained so that:

- prediction = current Q estimate for the chosen `(s, a)`
- target = reward plus discounted best next-state value

So the training objective is the temporal-difference idea:

`Q(s, a)` should move toward `r + gamma * max_a' Q(s', a')`

What matters conceptually:

- the network is not trained against a human label
- it is trained against a bootstrapped target built from reward and future value
- the target itself depends on current value estimates

That self-referential structure is powerful, but it is also why DQN can become
unstable if implemented carelessly.

### DQN And Double DQN

The lecture then distinguishes between vanilla DQN and Double DQN.

Vanilla DQN:

- uses the same network for both sides of the value estimate

Double DQN:

- uses separate networks for the two Q computations in the update

The slide's key point is that this helps reduce bias introduced by the
inaccuracies of the Q-network early in training.

The practical intuition:

- if one network both selects and evaluates the best next action, it can become
  over-optimistic
- separating roles makes the target less biased

### DQN Tricks That Matter In Practice

The lecture includes a very practical "tricks" slide. These are not cosmetic
details; they are part of why DQN actually trains.

Experience replay:

- store experiences consisting of actions, state transitions, and rewards
- sample mini-batches from that replay memory during training

Why this helps:

- breaks up highly correlated sequential data
- improves sample reuse
- makes the optimization look more like standard mini-batch training

Fixed target network:

- the target depends on network parameters
- if the network changes too quickly, the target also moves too quickly
- updating the target network only periodically makes training more stable

The slide gives a concrete example update cadence:

- update target network every 1,000 steps

Reward clipping:

- map positive rewards to `+1`
- map negative rewards to `-1`

The purpose is to standardize reward scales across games so training is less
sensitive to raw score magnitude.

Skipping frames:

- act every 4 frames instead of every single frame

This reduces redundancy and makes training more efficient.

### What DQN Can And Cannot Do

The lecture is careful not to oversell Q-learning.

The results slides show that DQN can achieve strong Atari performance, but the
limitations slide is equally important.

Downsides of Q-learning from the lecture:

- it works best when the action space is discrete and small
- it cannot directly handle continuous action spaces well
- because the policy is computed by `argmax` over Q-values, it does not naturally
  learn stochastic policies

The steering-wheel-angle example from the slide is a good intuition pump. If
the action is something continuous like:

- steering angle

then enumerating actions and maximizing over them is no longer a clean fit.

Those limits explain why the lecture moves into policy-gradient methods. If
`argmax_a Q(s, a)` is awkward or impossible, it can be cleaner to learn the
action rule directly.

### Policy Gradient Methods

The lecture's next move is important because it shows that Q-learning is not the
only way to build an RL agent. Instead of learning a value table or Q-network and
then deriving a policy by `argmax`, policy-gradient methods directly learn the
policy itself.

Q-learning view:

- network input: state
- network output: Q-values for each possible action
- action choice: take the action with the highest predicted Q-value

Policy-gradient view:

- network input: state
- network output: a policy distribution over actions
- action choice: sample from that distribution
- training goal: increase the probability of actions that led to high return and
  decrease the probability of actions that led to low return

This is a different mental model. The network is no longer only a critic that
scores actions. It becomes the actor that decides how the agent behaves.

### Why Learn The Policy Directly

The biggest reason is action-space flexibility.

Q-learning fits naturally when:

- the action space is discrete
- the number of actions is small
- it is cheap to compute `max_a Q(s, a)`

But many real control problems do not look like that. A self-driving policy may
need to output:

- steering angle
- acceleration
- braking force
- continuous control commands

In those cases, enumerating every possible action is not realistic. A policy
network can instead output parameters of a distribution. For a continuous action,
the network might output:

- mean action
- variance / spread

Then the agent samples from that distribution. This lets the policy represent
continuous behavior without discretizing the world into an awkward list of
possible actions.

### Stochastic Policies

Policy gradients also make stochastic policies natural. This matters because a
good strategy is not always deterministic.

Possible reasons to keep a stochastic policy:

- exploration during training
- environments where multiple actions are reasonable
- avoiding brittle behavior caused by always taking one greedy action
- representing uncertainty in action choice

This connects back to the exploration problem. In Q-learning, exploration is
often added externally with something like `epsilon`-greedy action selection. In
policy learning, randomness is part of the learned policy distribution itself.

### REINFORCE Intuition

The lecture presents the core policy-gradient training loop in a REINFORCE-style
way:

1. Run the current policy in the environment for a while.
2. Record the states, actions, and rewards.
3. Compute returns from the rewards.
4. Increase the probability of actions that led to high return.
5. Decrease the probability of actions that led to low or negative return.

The key term is the log probability of the sampled action:

`log pi_theta(a_t | s_t)`

That term tells the optimizer how likely the current policy was to take action
`a_t` in state `s_t`. The reward/return term tells whether that action should be
reinforced or discouraged.

A compact version of the update idea:

`grad_theta J(theta) ~= R_t * grad_theta log pi_theta(a_t | s_t)`

What this means:

- if return is high, push the policy to make that action more likely
- if return is low or negative, push the policy to make that action less likely
- the gradient changes the policy parameters directly

This is gradient ascent on expected return, not supervised learning against a
human-provided label.

### Policy Gradient Loss

In code, this often becomes a loss with a negative sign because optimizers
usually minimize:

`loss = -log pi_theta(a_t | s_t) * R_t`

Why the sign matters:

- high return should make the optimizer increase the selected action's
  probability
- minimization of the negative log-probability term has that effect
- bad returns reverse the pressure

This is easy to confuse with normal classification. In classification, the label
tells the model the correct class immediately. In policy gradient RL, the
"correctness" of an action is inferred from the return after interacting with the
environment.

### Credit Assignment And Variance

Policy gradients are powerful, but the basic REINFORCE estimator can be noisy.
The agent may receive a reward many steps after the action that helped cause it.
That creates a credit-assignment problem:

- which earlier action deserves credit for the later reward?
- which action should be blamed for a later crash?
- how much of the return should be assigned to each sampled action?

This is one reason policy-gradient methods often need many rollouts. The signal
is not as direct as a supervised label, and the variance of the gradient estimate
can be high.

Practical improvements include:

- subtracting a baseline from returns
- using advantage estimates instead of raw returns
- combining policy learning with value learning
- adding entropy regularization to preserve exploration

The lecture does not need all of those details to make the main point: direct
policy optimization is flexible, but it can be statistically harder than it looks.

### Actor-Critic Bridge

Actor-critic methods combine the two sides:

- actor: learns the policy `pi(a | s)`
- critic: learns a value estimate for states or state-action pairs

The critic gives the actor a better training signal than raw episode returns
alone. Instead of only asking "was the whole trajectory good?", the agent can ask
"was this action better than expected in this state?"

That idea is usually written with an advantage function:

`A(s, a) = Q(s, a) - V(s)`

Interpretation:

- positive advantage: the action was better than the baseline expectation
- negative advantage: the action was worse than expected
- near-zero advantage: the action was about average for that state

This is a clean conceptual bridge:

- value methods estimate how good states/actions are
- policy methods learn how to act
- actor-critic methods use learned value estimates to improve policy updates

### Simulation And Reality

The autonomous-driving example in the lecture makes the practical issue clear.
For a car, the environment is the road, the states are sensor observations, and
the action might be a steering command. A policy-gradient agent could in
principle learn to steer by trial and error.

But real-world trial and error is dangerous:

- crashing during training is not acceptable
- rare events matter a lot
- real sensor inputs are messy
- the agent may face situations not represented in its simulator

This is why simulation matters. If the simulator is realistic enough, the agent
can collect experience safely. But sim-to-real transfer is its own hard problem:
a policy that looks strong in a simulator can fail when lighting, road geometry,
other drivers, or sensor noise differ from the training environment.

### AlphaGo / AlphaZero Takeaway

The Go examples are useful because they show why RL is attractive for problems
where the search space is enormous. The number of possible board configurations
in Go is far too large for brute-force enumeration.

The important idea is not just that AlphaGo won games. It is that deep learning
can be combined with search, self-play, and value/policy estimation:

- policy networks guide which moves are promising
- value networks estimate how good a board position is
- self-play creates training experience without needing every move labeled by a
  human expert

AlphaZero pushes this further by showing that a system can learn strong play
from self-play across multiple games. That connects to the broader course theme:
deep learning becomes more powerful when it is paired with the right problem
structure and training loop.

### Lecture 5 Final Summary

The finished Lecture 5 arc is:

- RL is about agents acting in environments to maximize future reward.
- The main data object is a trajectory, not an independent labeled example.
- Rewards can be delayed, sparse, noisy, or badly designed.
- Q-learning estimates expected future return for state-action pairs.
- DQN uses neural networks to approximate Q-values in high-dimensional states.
- DQN needs stabilization tricks like replay buffers and target networks.
- Q-learning works best with small discrete action spaces.
- Policy-gradient methods directly optimize a policy and handle stochastic or
  continuous actions more naturally.
- Actor-critic methods combine policy learning with value estimation.
- Real-world RL depends heavily on safe data collection, simulation fidelity,
  reward design, and transfer from training environments to deployment.

My main takeaway: RL is not just another loss function. It changes the whole
learning setup because the model's behavior changes the data it receives.

My personal RL checklist after this lecture:

- define the state, action, reward, and terminal condition before naming an
  algorithm
- check whether the action space is discrete or continuous
- decide whether the problem needs value learning, direct policy learning, or an
  actor-critic mix
- separate immediate reward from long-term return
- inspect reward design for incentives that could produce unwanted behavior
- treat simulator realism and data collection safety as part of the method, not
  afterthoughts

## Lecture 6: Language Models and New Frontiers

Lecture 6 is the course's capstone-style lecture. It does two things at once:

- steps back and asks where deep learning is still brittle
- points forward to newer frontiers like diffusion models, protein generation,
  and large language models

This lecture matters because it prevents the course from ending with the idea
that deep learning is solved. The earlier lectures show how to build neural
networks. Lecture 6 is more about when I should trust them, when I should be
skeptical, and what newer model families are trying to fix or extend.

### How Lecture 6 Is Framed

The 2026 deck is titled "Deep Learning Limitations and New Frontiers," and that
wording is doing a lot of work. This lecture is not just a showcase of recent
models. It is a pause after the first five lectures:

- Lecture 1: neural networks as trainable function approximators
- Lecture 2: sequence modeling and next-step prediction
- Lecture 3: vision, convolution, detection, and spatial structure
- Lecture 4: generative modeling through VAEs, GANs, and latent variables
- Lecture 5: reinforcement learning, value functions, policies, and agents
- Lecture 6: what breaks, what scales, and where the field is moving next

The "so far" slide shows a useful compressed picture of the whole course:

- data comes in as signals, images, sensors, or other raw observations
- the model turns those inputs into internal representations
- the output is a decision, prediction, detection, action, or generated artifact

That picture makes deep learning look clean: data goes in, decision comes out.
The rest of Lecture 6 complicates that story. The important question becomes:
what happens when the input is biased, shifted, adversarial, ambiguous, or
outside the training distribution?

This is the right capstone framing for the course. The earlier lectures taught
me how to construct the pipeline. Lecture 6 asks whether the pipeline is
trustworthy once the examples stop being clean notebook examples.

### Administrative Context: Final Project Or Paper Review

The Lecture 6 deck also makes the final class project concrete. For the January
2026 in-person version, the two options were:

- proposal presentation
- one-page review of a deep learning / AI paper

The proposal route is framed as a short, strict presentation of a novel deep
learning research idea or application. The paper-review route is graded on
clarity of writing and technical communication of the main ideas.

For my independent-study repo, that is a useful hint about what "finishing" the
course should mean. It should not only mean checking off lecture notes and lab
scripts. A stronger finish would be one of these:

- write a concise paper review that proves I can explain a technical idea in my
  own words
- draft a small project proposal that turns one course idea into an experiment
- connect a lab mechanic, like LoRA or DB-VAE resampling, to a real research
  question

I should treat this as the next phase after the core lectures/labs. The repo is
now strong on mechanics notes; the next level is synthesis.

### Why The Lecture Starts With Hype

The deck includes a "Rise of Deep Learning" collage with examples across voice
cloning, StarCraft, cancer detection, fake-image generation, stock prediction,
protein folding, autonomous driving, manufacturing, and other applied systems.
That slide is not rigorous evidence by itself, but it explains why the lecture
needs a limitations section.

Deep learning is being used in domains with very different stakes:

- entertainment or media generation, where failure can be annoying or misleading
- recommendation and market prediction, where failure can have economic cost
- medicine and driving, where failure can hurt people
- scientific discovery, where generated candidates still need physical
  validation

So the question is not "can neural networks do impressive things?" They clearly
can. The harder question is whether a model's success on the training/test setup
means it will behave safely under distribution shift, rare cases, or adversarial
pressure.

This connects back to Lecture 5 too. In reinforcement learning, a reward
function can optimize exactly what I asked for and still produce behavior I did
not want. In supervised or generative modeling, the same issue appears through
dataset choice, objective design, and evaluation setup.

### Universal Approximation Is Not A Free Pass

The "Power of Neural Nets" slide quotes the universal approximation theorem:
a feed-forward network with a single hidden layer can approximate any continuous
function to arbitrary precision. This is the mathematical reason neural
networks are such flexible function approximators.

But the slide immediately lists the caveats:

- the number of hidden units may be infeasibly large
- the resulting model may not generalize

That second caveat is the one I need to keep returning to. Approximation and
generalization are different claims.

Approximation asks:

- does there exist a set of weights that can represent the desired function?

Generalization asks:

- will training find a useful function from finite data?
- will that function work on held-out examples?
- will it still work when the input distribution changes?

The theorem does not solve optimization, data coverage, robustness, or
interpretability. It says the function class is powerful. It does not say that
my trained model learned the right thing.

### AI Hype And The Need For Skepticism

The historical AI-hype slide shows a repeated pattern:

- early excitement
- inflated expectations
- AI winter when expectations are not met
- renewed hopes
- another winter
- current explosive growth

My takeaway is not that current deep learning progress is fake. The results are
real. The point is that technical capability and public expectation can move at
different speeds.

That matters for how I should write and think:

- do not describe a model as "understanding" unless I can say what evidence
  supports that claim
- do not treat a benchmark score as deployment readiness
- do not confuse a compelling demo with a reliable system
- do not ignore failure cases because the average result looks strong

The lecture is pushing a mature posture: be excited about what deep learning can
do, but do not let the excitement erase the engineering and scientific burden of
testing where it fails.

### Rethinking Generalization: The Random-Label Warning

The deck cites Zhang et al.'s "Understanding Deep Neural Networks Requires
Rethinking Generalization" result. The slide uses familiar image examples, then
adds random-looking labels underneath. The point is that a deep model can fit
labels even when the labels no longer reflect the true semantic structure of the
images.

This is a big conceptual warning:

- fitting the training set is not the same as learning the real pattern
- high-capacity networks can memorize arbitrary mappings
- the model can look successful under the training objective while learning
  something useless

A clean way to say it:

- training accuracy measures whether the model matched the provided labels
- test accuracy measures whether that mapping transfers to held-out data
- out-of-distribution evaluation measures whether the model learned something
  robust enough to survive a changed setting

This is why my notes should always separate:

- optimization: did the loss go down?
- in-distribution generalization: did held-out test performance improve?
- robustness: does performance survive perturbations, subgroup shifts, rare
  cases, or adversarial inputs?

The random-label experiment is the reason I should be cautious when a model has
enough parameters to fit almost anything. Capacity is useful, but it makes weak
evaluation easier to fool.

### Neural Networks As Function Approximators

A useful high-level summary of the course is:

- discriminative models learn mappings from data to decisions
- generative models learn data distributions or inverse mappings
- RL agents learn policies for acting over time

In all of these cases, neural networks can be viewed as function approximators.
They learn a mapping from input space to output space, or they learn a
probability distribution that explains the data.

The universal approximation theorem is a useful theoretical backdrop:

- a feed-forward neural network with enough hidden units can approximate a broad
  class of continuous functions

But the caveats are the part that matters in practice:

- the number of hidden units required may be infeasibly large
- the theorem does not say gradient descent will find the right weights
- it does not guarantee good generalization
- it does not say what happens outside the training distribution

So I should not read "universal approximator" as "universal problem solver." It
is a statement about representational possibility, not a guarantee of learning,
robustness, or safety.

### Rethinking Generalization

The Zhang et al. random-label experiment is one of the strongest warnings in the
lecture.

Setup:

- start with an image classification dataset
- progressively randomize the labels
- train a modern deep network on the corrupted training data
- compare training accuracy and test accuracy

Result:

- the model can fit the training set almost perfectly, even when labels are
  random
- test accuracy collapses as labels become less meaningful

The key lesson:

- high training accuracy does not prove the model learned useful structure
- overparameterized networks can memorize noise
- generalization has to be measured on held-out data and, ideally,
  out-of-distribution cases

This is different from how I might naively think about capacity. More capacity
lets a model learn complex patterns, but it also lets the model memorize
spurious or random relationships.

### Data Bounds And Out-Of-Distribution Inputs

The function-approximator picture is useful because it shows where deep learning
is strongest:

- interpolate within regions where the model has seen training examples
- learn patterns supported by many examples
- make predictions on inputs similar to the training distribution

The hard question is what happens outside that region.

If a new input is far from the training data, the network can still output a
confident prediction. The output may look normal because the softmax still
produces a probability distribution, but the model may not actually "know" the
answer in any meaningful sense.

This is the question the lecture keeps returning to:

`How do we know when the network does not know?`

That question applies to:

- autonomous driving
- medical imaging
- facial detection
- scientific discovery
- LLM answers that sound plausible but are not grounded

### Distribution Shift And Uncertainty

The 2026 Lecture 6 deck frames this as a basic limit of deep learning systems:
they are strong at learning from examples, but they do not automatically know
where their examples stop being relevant.

My working definition:

- in-distribution input: an input that looks like the data the model learned
  from, including the important hidden factors behind the examples
- out-of-distribution input: an input that may have the same superficial format
  but comes from a different part of the world than the training set covered
- uncertainty: the model's signal that its prediction should not be trusted as a
  normal confident decision

The trap is that the network will usually still produce an answer. A classifier
does not stop at the edge of the training distribution and say "this is outside
my experience." It keeps applying the learned function. That means a decision
boundary can extend into regions where there were no examples to constrain it.

This changes how I should read probability outputs:

- a high softmax score is not the same thing as grounded knowledge
- low loss on a test set does not guarantee the model will behave sensibly under
  distribution shift
- a model can be calibrated on common cases and still fail on rare cases
- uncertainty should be tested, not assumed from the architecture

The practical version of the lecture's question is:

`When should this system abstain, ask for help, or fall back to a safer policy?`

For my own experiments, this means I should not only report the normal train /
validation / test split. I should also ask:

- What examples were common in training?
- What examples were rare or missing?
- What transformations or environments would make the input meaningfully
  different?
- Does the model's confidence drop on these cases?
- If confidence does not drop, is that because the model is robust or because it
  is overconfident?

### Failure Mode: Dataset Bias And Missing Cases

The colorization example is a small version of a much larger problem. If a model
learns from a biased set of images, it can hallucinate patterns that were common
in the training data even when they do not fit the specific input.

This is not just a "bad dataset" problem in the vague sense. The model is doing
what training asks it to do: find patterns that reduce loss on the examples it
sees. If the training data overrepresents some settings and underrepresents
others, the model can learn those frequencies as if they were facts about the
world.

The colorization example is useful because the error is easy to see:

- the input image is ambiguous
- the model fills in color using learned correlations
- if the training distribution is biased, the default guess reflects that bias
- the output can look plausible while still being wrong for the specific case

The autonomous-driving crash example is the safety-critical version of the same
issue:

- the model was trained from past visual data
- the deployment environment changed
- a construction barrier appeared in a place not represented in the training data
- the system behaved badly at exactly the kind of out-of-distribution point where
  uncertainty should have mattered

My takeaway is that failure can come from missingness, not only from noisy labels.
If the dataset never covers an important situation, the model cannot reliably
learn how to handle that situation.

The more dangerous part is that rare cases can matter more than common cases.
For a normal benchmark, a rare construction-zone configuration might barely move
the average accuracy number. For an autonomous system, that same rare case can be
the exact case where the cost of being wrong is highest.

That gives me a better way to think about data coverage:

- common cases help the model learn the main pattern
- rare cases test whether the model can handle the edges of deployment
- missing cases create blind spots that normal accuracy can hide
- safety-critical systems need stress tests, not just representative averages

So the evaluation question should not be only:

`How accurate is the model on the test set?`

It should also be:

`What important situations are absent from the test set?`

This connects back to Lab 2. A face detector can look strong on average while
still having gaps for underrepresented groups or unusual lighting / pose /
occlusion settings. The technical problem and the fairness problem are linked by
the same mechanism: the training distribution shapes what the model treats as
normal.

### Failure Mode: Adversarial Examples

The adversarial example section connects directly back to gradient descent.

Normal training asks:

`How should I change the weights to reduce loss?`

Adversarial attack construction asks a similar but reversed question:

`How should I change the input to increase loss?`

The weights and true label are fixed. The attacker computes a small perturbation
to the input that pushes the model toward a wrong prediction.

Important idea:

- the perturbation can be tiny or visually meaningless to a human
- the perturbation is meaningful to the model because it follows the gradient of
  the model's loss
- the same optimization machinery that makes models trainable can expose
  directions where the model is brittle

This makes adversarial examples easier to reason about. The model has learned a
high-dimensional decision surface. In high dimensions, there can be many small
directions that do not matter to human perception but do matter to the model's
internal features. A perturbation can be small in pixel space and still move the
input across the learned boundary.

The important distinction:

- random noise may not fool the model
- optimized noise is chosen specifically to exploit the model's gradients
- the attack is model-aware, not just visually strange

This also explains why adversarial examples are a robustness test rather than a
normal data augmentation trick. If the defense is known, an attacker can often
optimize around the defense. That means robustness claims should be measured
against adaptive attacks, not only against the first attack tried.

The lecture also points to physically realized adversarial examples. That matters
because adversarial attacks are not only a digital-image curiosity. If a physical
object can be designed to fool a classifier from many angles, robustness becomes
a real-world safety issue.

The physical setting adds extra constraints:

- the perturbation has to survive camera noise
- it has to survive changes in distance and angle
- it has to survive lighting and background variation
- it has to affect the model consistently enough to matter

If an attack still works under those transformations, it is not just a weakness
of a static image classifier. It is evidence that the model's learned features
can disagree with the human-relevant structure of the task.

My notes for future model evaluation:

- report clean accuracy and robust accuracy separately
- test sensitivity to small input changes
- inspect high-confidence wrong predictions
- be careful with defenses that only hide gradients
- treat adversarial robustness as a separate property from benchmark accuracy

### Failure Mode: Algorithmic Bias

Algorithmic bias is not presented as a separate social topic disconnected from
the math. It follows from the same core issues:

- data imbalance
- missing examples
- measurement noise
- biased labels
- overconfident predictions
- deployment settings that differ from training settings

This ties back to Lab 2. A facial detection system can have strong average
accuracy while still performing worse for underrepresented groups. If the
training distribution is skewed, the learned representation and decision boundary
can inherit that skew.

The practical lesson:

- average accuracy is not enough
- evaluation needs subgroup metrics
- uncertainty and bias should be checked before deployment, not after harm occurs

I want to keep the different sources of bias separate:

- sampling bias: the dataset does not represent the full deployment population
- label bias: the targets themselves reflect human or institutional bias
- measurement bias: the observed variable is only a noisy proxy for the thing we
  care about
- representation bias: the learned features encode some groups or situations
  less well than others
- deployment bias: the model is used in a setting different from the setting it
  was trained or validated for
- feedback-loop bias: model decisions affect the future data that is collected

This matters because different bias sources require different interventions.
Collecting more data helps if the problem is missing coverage, but it does not
automatically fix biased labels. Reweighting examples can help with imbalance,
but it does not prove the deployment setting is safe. A better architecture can
help representation learning, but it cannot replace measurement and subgroup
evaluation.

For Lab 2, DB-VAE is a concrete response to representation and sampling bias.
The idea is to use the learned latent representation to identify regions of the
face distribution that are underrepresented, then sample more heavily from those
regions during training. That turns the model's own representation into a tool
for balancing training pressure.

The evaluation checklist should include:

- overall accuracy
- subgroup accuracy
- false positive and false negative rates by subgroup
- confidence calibration by subgroup
- uncertainty / abstention rates by subgroup
- representative failure examples, not just summary numbers

The strongest lesson here is that "fairness" is not one metric or one loss term.
It is a deployment property that depends on data, labels, model behavior,
thresholds, uncertainty, and the real consequences of mistakes.

### Robustness Checklist For My Own Work

Before I write that a model "works," I should be able to answer:

- What distribution did the training data come from?
- What distribution will the model actually face?
- What cases are rare, missing, or expensive to get wrong?
- Does the model know when it is uncertain?
- Are the worst errors concentrated in a subgroup or edge case?
- Does performance survive reasonable perturbations?
- Is accuracy being reported separately from robustness and fairness?
- What would happen if this model were confidently wrong?

This is probably the biggest mindset shift in Lecture 6. Earlier lectures taught
me how to build models that optimize a loss. This lecture is forcing me to ask
what happens when the optimized system leaves the clean benchmark setting.

### Limits Of Earlier Generative Models

Lecture 6 then returns to generative modeling from Lecture 4.

VAEs and GANs are powerful, but the lecture highlights real limitations:

- GAN training can be unstable
- GANs can suffer from mode collapse
- generated samples may reflect the average or dominant training modes
- extrapolating beyond the training distribution is hard
- one-shot generation puts a lot of pressure on one forward pass

Mode collapse is especially important. If a generator finds a narrow set of
outputs that fool the discriminator, it may fail to represent the full diversity
of the data distribution. The samples can look sharp but not cover the range of
valid possibilities.

### Diffusion Models: Core Idea

Diffusion models attack generation differently. Instead of producing a complete
sample in one shot, they learn an iterative denoising process.

The two processes:

- forward process: gradually add noise to real data
- reverse process: learn to remove noise step by step

Forward noising:

- start with a real image or data example
- sample random noise
- add controlled amounts of noise over many time steps
- eventually reach something close to pure noise

Reverse denoising:

- start from random noise
- use the learned model to predict how to remove a small amount of noise
- repeat many times
- end with a clean generated sample

This reframes generation as many easier denoising decisions instead of one hard
generation decision.

### Why Predict Noise

The lecture makes a useful point about the training objective. The model could
try to predict the clean image directly from a noisy image, but that is hard. In
practice, diffusion models often learn to predict the residual noise that was
added.

That is a better local task:

- the forward process gives paired examples of noisy states and known noise
- predicting the noise residual is more stable than reconstructing everything at
  once
- repeated small corrections can build high-quality outputs

This is one reason diffusion models produce strong samples. They turn generation
into a sequence of tractable refinement steps.

### Diffusion Compared With VAEs And GANs

My comparison table:

| Model family | Generation style | Main strength | Main concern |
| --- | --- | --- | --- |
| VAE | encode/decode through latent variables | structured latent representation | blurry samples or reconstruction/regularization tradeoff |
| GAN | generator/discriminator game | sharp samples | unstable training and mode collapse |
| Diffusion | iterative denoising from noise | high-fidelity diverse samples | many denoising steps can be computationally expensive |

Diffusion models use an iterative design, which gives them a different
failure profile. Instead of asking one network pass to invent the whole image,
the model learns a controlled reverse process.

### Diffusion For Biology

The biology example was one of the clearest "new frontier" parts of the lecture.
Proteins can be thought of in two linked ways:

- sequence: amino acid language
- structure: 3D geometry that determines function

The generative goal is not just to make a pretty image. It is to design molecules
that could have useful biological or therapeutic function.

Diffusion can be applied to protein structure by treating the random state as an
unstructured or noisy configuration and learning to denoise toward plausible
protein backbones. The model gradually refines geometry until it produces a
structured 3D candidate.

The broader idea:

- generative AI can design candidates in scientific spaces
- the output still needs wet-lab or experimental validation
- the model is part of a discovery loop, not a replacement for biology

This is a useful corrective to hype. The model can propose, but physical reality
still tests.

### Protein Sequences As Language

The lecture also describes protein sequences as a kind of biological language.
That connects back to sequence modeling:

- amino acids are discrete symbols
- protein sequences can be modeled like strings
- large datasets across evolution provide many examples
- generated sequences can be candidates for new function

This is one of the strongest conceptual bridges in the course. The same general
ideas from language modeling can transfer into biology if I define the right
tokens, training data, and validation criteria.

### Large Language Models

The final part of Lecture 6 moves to large language models.

At the highest level:

- an LLM is a very large neural network trained on very large text datasets
- the dominant architecture is the Transformer
- the basic pretraining task is next-token prediction
- scale changes the capabilities that emerge

GPT is useful to unpack:

- Generative: produces text-based outputs
- Pre-trained: first trained on a large general corpus
- Transformer: uses the Transformer architecture

This is not a totally new objective compared with Lecture 2. It is the same
next-token idea scaled up massively with better architectures, more data, and
more compute.

### Tokenization And Next-Token Prediction

The basic pipeline:

1. Collect large amounts of raw text.
2. Split text into tokens.
3. Convert tokens into numerical embeddings.
4. Feed token sequences into a Transformer.
5. Predict the probability distribution over the next token.
6. Compare the predicted distribution with the true next token using cross
   entropy.
7. Update model weights.

Important detail:

- tokens are not always words
- tokenization can use subwords or other chunks
- the model predicts a distribution over its vocabulary
- softmax turns logits into probabilities for the next token

In training, the targets are shifted by one token. The input sequence is the
context, and the label at each position is the next token.

This connects directly to Lab 3. The local Lab 3 scripts are much smaller, but
they check the same core mechanics:

- template text
- token IDs
- shifted labels
- answer masking
- causal next-token loss

### Prompting And Inference

At inference time, the trained LLM uses the same next-token machinery:

1. User provides a prompt.
2. Prompt is tokenized.
3. Model predicts the next-token distribution.
4. A token is selected by sampling or decoding.
5. The selected token is appended to the context.
6. The loop repeats.

The important idea is that long answers are generated one token at a time. The
model is not retrieving a whole paragraph as a fixed object. It is repeatedly
conditioning on the current context and extending it.

This is why generation settings matter:

- greedy decoding can be deterministic and repetitive
- sampling can increase diversity
- temperature changes how sharp or flat the token distribution is
- context length limits what the model can condition on

### Capabilities Of LLMs

The lecture points to several capabilities that become useful once language is the
interface:

- knowledge retrieval
- summarization
- writing and editing
- code generation
- planning support
- natural-language interaction with tools and systems

The key phrase for me is "natural language as an interface." LLMs are not only
models that produce text. They make it possible to control or query complex
systems through language.

### LLM Limitations

The limitations are just as important:

- hallucinations: plausible text that is not grounded in truth
- weak uncertainty estimates: confident wording even when the answer is wrong
- reasoning gaps: failures on multi-step logic or planning
- bias: outputs reflect patterns and biases in training data
- stale or missing knowledge: pretraining data is not the same as current truth
- prompt sensitivity: small wording changes can change behavior

This links back to the earlier uncertainty section. A softmax distribution over
tokens does not automatically tell me whether the answer is true. The model can
be confident about the next token while being wrong about the world.

### Scaling Laws And Emergence

The lecture introduces scaling laws as an empirical way to describe how model
performance changes with:

- number of parameters
- dataset size
- compute

The striking observation is emergence:

- some capabilities are weak or absent in smaller models
- as scale increases, performance can jump on certain tasks
- larger models can show qualitatively new behavior

I should be careful with the interpretation. Scaling laws are not a proof that
"scale is all you need." They are evidence that scale has been a powerful driver
of recent progress. But the earlier parts of the lecture still apply:

- data quality matters
- uncertainty matters
- bias matters
- deployment and evaluation matter

### Foundation Models

The final concept is the foundation model:

- train a large general model on broad unannotated data
- adapt it to many downstream tasks
- use the model as a general-purpose base rather than training a separate model
  from scratch for every task

This is a major shift from the earlier course examples:

- Lecture 1: train a model for a specific supervised task
- Lecture 2: train a sequence model for a specific sequence task
- Lecture 3: train a CNN for a specific vision task
- Lecture 4: train a generative model for a specific data distribution
- Lecture 6: train a broad model that can be adapted across tasks

Foundation models are powerful because they reuse representations learned from
massive data. They are risky because their failures can also transfer broadly
across many downstream uses.

### My Finished Lecture 6 Takeaway

Lecture 6 is a realism check.

Deep learning works because neural networks are extremely flexible function
approximators, and scaling that flexibility with data and compute has created
diffusion models, protein generators, and LLMs. But the same flexibility creates
failure modes:

- memorization instead of real generalization
- brittleness outside the training distribution
- adversarial vulnerability
- biased behavior from biased data
- overconfident outputs when the model should be uncertain

The frontier is not just "bigger models." It is:

- better generative processes
- better uncertainty estimation
- better evaluation
- better grounding
- safer deployment
- better ways to combine models with human goals and real-world validation

The concrete habit I want from this lecture is to separate model properties that
often get blurred together:

- accuracy: how often predictions match labels on the evaluated data
- calibration: whether confidence scores match actual correctness rates
- robustness: whether performance survives perturbations or distribution shift
- fairness: whether errors and confidence failures are uneven across groups
- deployment readiness: whether the system has monitoring, fallbacks, and clear
  limits for real use

Those are different questions. A model can improve on one and fail on another.

## Software Lab 3: LLM Fine-Tuning

The current 2026 course site lists Software Lab 3 as "Fine-Tune an LLM, You
Must!" under the newer-frontiers part of the course. The official lab notebook
is built around a real LLM stack:

- base model: Liquid AI LFM2-1.2B
- fine-tuning method: LoRA / parameter-efficient fine-tuning
- judge model: Gemini 2.5 through OpenRouter
- evaluation/monitoring: Comet Opik

My repo version is deliberately smaller. It is a local mechanics pass, not the
official GPU/API competition path. The official lab is about real LLM behavior
at scale; this repo checks whether I understand the pieces well enough to run
and debug the full notebook later.

### What The Lab Is Really About

The lab is not only "make a model talk like Yoda." The style task is a concrete
way to practice modern LLM post-training:

- represent a chat conversation in a fixed template
- tokenize the template into model-readable IDs
- load a causal language model
- generate text by repeatedly predicting the next token
- fine-tune on instruction/response examples
- avoid updating every parameter by using LoRA
- evaluate outputs with a judge model and controls
- monitor the system once it becomes an application, not just a notebook cell

This connects back to Lecture 2. An LLM is still a sequence model trained to
predict the next token from previous context. The difference is scale,
pretraining, and the post-training workflow around the model.

### Chat Templates

A chatbot model needs the conversation to be serialized into text. The official
lab uses markers that separate user turns from assistant turns. The basic shape
is:

`start -> user marker -> question -> end marker -> assistant marker -> answer`

The important idea is that the model does not naturally know which span is the
user question and which span is the assistant answer. The template gives it a
consistent format.

Why this matters:

- the same user text can mean different things depending on where it appears
- special tokens mark turn boundaries
- the assistant marker tells the model where generation should begin
- fine-tuning examples must match inference-time prompts

If training uses one format and inference uses another, the model can fail even
if the underlying task is simple.

### Tokenization

Language models do not operate directly on Python strings. The lab emphasizes
the encode/decode loop:

- encode: text -> token IDs
- decode: token IDs -> text

Modern LLM tokenizers usually use subword methods such as BPE. That gives a
middle ground between word-level and character-level tokenization:

- word tokenization can create huge vocabularies and struggles with unseen words
- character tokenization has a tiny vocabulary but produces long sequences
- subword tokenization keeps sequence lengths manageable while still handling
  unfamiliar words as combinations of known pieces

The local script uses a tiny character-level tokenizer but keeps the chat markers
as atomic special tokens. That is not meant to imitate LFM2's tokenizer exactly.
It is a way to check the mechanics:

- special markers should survive encode/decode
- prompt and answer tokens should be distinguishable
- decoded answers should be readable after skipping special tokens

The first Lab 3 script verified:

- the decoded prompt matched the original prompt
- the sample supervised example had 165 local tokens
- 104 of those tokens were in the answer/loss region
- the first supervised target after shifting was the first character of the
  assistant answer

### Causal Language Modeling

The official lab uses the same core objective as a normal causal LM:

`predict token[t + 1] from tokens[:t]`

The model outputs logits over the vocabulary at every sequence position. For
training, the target is the same sequence shifted one step to the left.

Important shape:

- input tokens: all positions except the last
- target tokens: all positions except the first
- logits: one vocabulary-sized vector per input position

The loss is cross entropy. This is still classification, just repeated across
sequence positions. The "classes" are token IDs.

### Why The Answer Mask Matters

In supervised chat fine-tuning, the full string contains both the user prompt and
the assistant answer. The model needs the prompt as context, but I do not want
to train it to reproduce the user prompt. I want it to learn the assistant
response.

So the lab computes a mask:

- prompt tokens: context only
- answer tokens: contribute to loss

After shifting for next-token prediction, the mask has to shift too. This is a
small implementation detail, but it is easy to get wrong. If I include prompt
tokens in the loss, the model spends capacity learning the template and user
question instead of the desired response behavior.

### Generation

One forward pass predicts only the next token. Longer text is generated by
feeding sampled tokens back into the model:

1. Format the prompt without an answer.
2. Encode the prompt.
3. Predict logits for the next token.
4. Pick a token by argmax or sampling.
5. Append that token to the context.
6. Repeat until an end marker or token limit.

Temperature changes the sampling distribution:

- lower temperature: more deterministic, safer, more repetitive
- higher temperature: more varied, but easier to derail

The local tiny model shows exposure bias clearly. Under teacher forcing, the
loss can drop a lot, but free generation on held-out prompts can still become
messy because every generated mistake becomes part of the next context.

### LoRA

Full fine-tuning updates all model weights. For a billion-parameter model, that
is expensive in memory and compute. LoRA changes the update strategy.

Instead of changing a large weight matrix directly, LoRA adds a low-rank update:

`W' = W + scale * B A`

where:

- `W` is the frozen pretrained weight matrix
- `A` and `B` are small trainable matrices
- the rank of `BA` is much smaller than the original matrix dimensions

The official lab applies LoRA through PEFT to attention and feed-forward
projection modules such as query, key, value, output, gate, up, and down
projections.

The local script mirrors the idea:

- first train a tiny base causal LM on plain English answers
- freeze the base weights
- train only low-rank adapter parameters on styled answers

The default local run reported roughly:

- total parameters: about 133k
- LoRA-stage trainable parameters: about 9.7k
- LoRA-stage trainable percentage: about 7.3%

That is the important qualitative point. The style adaptation stage changes a
small trainable slice while leaving most of the model fixed.

### Local Fine-Tuning Results

The second script gives a concrete sanity check:

- base plain-answer loss drops substantially
- LoRA style-tuning loss also drops
- the generated text becomes somewhat style-biased on seen prompts
- held-out styled answers still have high loss

This is realistic for a tiny character-level model trained on a tiny dataset.
It verifies the fine-tuning plumbing, but it is not evidence of a strong model.
In fact, the rough generations are useful because they make the limitation
obvious. The official LFM2 run is needed for serious qualitative output.

### LLM As Judge

The official lab uses an LLM-as-judge setup to turn style quality into a score.
The judge model receives:

- a system prompt that defines the judging role
- an example of the target style
- generated text to evaluate
- instructions to output a numeric score

The key insight is that qualitative outputs need evaluation structure. Looking
at a few samples is useful, but a repeatable rubric lets me compare:

- base English responses as a negative control
- true target-style responses as a positive control
- generated responses from my model

The judge is not automatically correct. The system prompt and rubric can change
scores. A weak judge can reward superficial markers instead of real style. This
is why controls matter.

### Opik And Monitoring

The later part of the official lab moves from evaluation in a notebook to
monitoring a live LLM application. Opik tracing records calls and attaches
feedback scores, so the same evaluation idea can follow the model after
deployment.

This is a useful shift in mindset:

- training loss tells me about optimization
- offline evals tell me about test behavior
- tracing/monitoring tells me what happens when the model is used as a system

For real LLM applications, the model is only one part. The prompt template,
judge prompt, API wrapper, tracing, and metric definitions are also part of the
product.

### Offline Evaluation In This Repo

My third script replaces the API judge with a transparent local rubric. That is
not as capable as Gemini, but it keeps the evaluation shape:

- score base English controls
- score target-style controls
- score generated held-out answers
- compute held-out target cross entropy

The local evaluator intentionally separates "plumbing works" from "model is
good." The controls are the important part. If the positive controls do not
score higher than negative controls, the judge is not useful. If generated
answers score poorly and held-out loss is high, that is a model limitation, not
a reason to hide the result.

### What Counts As Finished For Local Lab 3

For this repo, Lab 3 is complete as a local mechanics pass because it has
runnable checks for:

- chat templates
- tokenization and decoding
- next-token labels
- answer masks
- tiny causal-LM training
- LoRA-style adapter tuning
- generation
- style controls
- held-out style loss

For the official course path, the remaining work is:

- run the official notebook on a GPU
- use the real LFM2-1.2B tokenizer and model
- complete the actual Hugging Face/PEFT setup in the notebook
- use OpenRouter with an API key for the judge model
- log traces and metrics through Opik
- submit the notebook/report with the official final likelihood cell

## Software Lab 1: Deep Learning In Python And Music Generation

Lab 1 is where the first two lectures become executable. Part 1 turns Lecture
1 into tensor operations, perceptrons, computation graphs, modules, gradients,
and optimization. Part 2 turns Lecture 2 into a character-level sequence model
that learns ABC music notation.

### Part 1: PyTorch Foundations

The official lab starts with the most basic question: what is the data object
that deep learning code manipulates? In PyTorch, the answer is usually a tensor.

The details that matter:

- rank: number of axes
- shape: size along each axis
- dtype: numerical type
- device: CPU or accelerator placement
- gradient tracking: whether operations involving the tensor are recorded for
  autograd

The local script `01_tensor_mechanics.py` makes this concrete:

- scalar tensor: rank `0`
- one-value vector: rank `1`, shape `(1,)`
- matrix: rank `2`
- image batch: rank `4`, shape `(batch, channels, height, width)`

The image-batch convention matters because later CNN code expects the channel
dimension before height and width. If I accidentally use `(batch, height, width,
channels)` in PyTorch, the model will interpret image width as channels and the
convolutions will be wrong.

### Broadcasting And Bias

Bias addition is the first place broadcasting appears in model code. In the
manual perceptron:

`linear_output = X @ W`

`shifted_output = linear_output + b`

If `linear_output` has shape `(batch_size, output_features)` and `b` has shape
`(output_features,)`, PyTorch broadcasts `b` across every row. That is what I
want, but it is still a real operation: every example gets the same learned bias
per output unit.

This is a good debugging rule: whenever a tensor with fewer dimensions is added
or multiplied against a larger tensor, I should be able to explain which axes are
being reused.

### Manual Perceptron To `torch.nn`

The local scripts deliberately climb the abstraction ladder:

1. `02_manual_perceptron_forward.py` computes `sigmoid(XW + b)` directly.
2. `03_manual_gradient_vs_autograd.py` derives the gradients by hand.
3. `04_torch_nn_bridge.py` reproduces the manual result with `nn.Linear`.
4. `06_models_and_autograd.py` packages the same idea inside `nn.Module`.

The point is not that I should avoid PyTorch abstractions. The point is that
`torch.nn` should feel like a packaging layer around math I can still see.

The `nn.Linear` convention is easy to forget:

- manual notes: `W` is `(input_features, output_features)`
- `nn.Linear`: `.weight` is `(output_features, input_features)`

That is why copying manual weights into `nn.Linear` requires a transpose.

### Autograd And The Training Loop

The lab's autograd section is the practical version of backpropagation.

For a normal training step, the order is:

1. switch the model to training mode when needed
2. clear old gradients with `optimizer.zero_grad()`
3. run the forward pass
4. compute one scalar loss
5. call `loss.backward()`
6. call `optimizer.step()`

The scalar gradient-descent demo in `06_models_and_autograd.py` is useful
because it strips the idea down to one parameter. If the loss is `(x - 4)^2`,
then gradient descent should move `x` toward `4`. A neural network training loop
is the same idea with many parameters and a more complicated loss surface.

### Part 2: Music As Sequence Modeling

The second half of Lab 1 uses ABC notation for Irish folk music. This is a
good sequence-modeling example because the raw data is text, but the text has
musical structure:

- song metadata starts with fields like `X:`, `T:`, `M:`, `K:`
- notes and durations are encoded as characters
- local syntax matters, but longer-range song structure also matters

The pipeline is:

1. Load songs from the official ABC dataset.
2. Join the songs into one training string.
3. Build a sorted character vocabulary.
4. Convert every character to an integer token.
5. Sample fixed-length windows.
6. Use the same windows shifted one character right as targets.

The local helper functions match those steps:

- `load_training_songs()` reads the ABC file and extracts song blocks.
- `build_vocabulary()` finds every distinct character and builds `char2idx`.
- `vectorize_string()` turns the training text into integer IDs.
- `get_batch()` samples random windows for minibatch training.

For a sequence like:

`A B C D`

the model sees:

- input: `A B C`
- target: `B C D`

That one-character shift is the entire next-token prediction objective.

For the actual tensor shapes in the local script:

- `x_batch`: `(batch_size, seq_length)`
- `y_batch`: `(batch_size, seq_length)`
- model logits: `(batch_size, seq_length, vocab_size)`

The loss flattens the first two dimensions before calling cross entropy:

- logits become `(batch_size * seq_length, vocab_size)`
- labels become `(batch_size * seq_length)`

That reshape is not just cleanup. It is what turns "one prediction per time
step" into the format PyTorch's classification loss expects.

### LSTM Model And Generation

The local music model has three conceptual pieces:

- embedding layer: turns character IDs into learned vectors
- LSTM: updates a hidden state across the sequence
- linear head: maps each hidden state to vocabulary logits

During training, the model predicts the next character at every time step, and
cross entropy compares the logits with the shifted target IDs.

During generation, the loop changes:

1. start with a prompt character such as `X`
2. predict logits for the next character
3. divide logits by temperature
4. sample a character
5. feed that sampled character back into the model
6. repeat

Temperature controls randomness:

- lower temperature sharpens the distribution and makes output more conservative
- higher temperature flattens the distribution and makes output more surprising
- too high can break ABC syntax
- too low can make generation repetitive

The local script can save generated text and ABC snippets. Rendering to WAV is
optional because it depends on external tools: `abc2midi` and `timidity`.

Generation is also where train-time and inference-time behavior diverge:

- during training, the model sees real previous characters from the dataset
- during generation, it sees its own sampled characters
- an early bad sample can push the sequence into unfamiliar territory
- longer generation makes these errors accumulate

That explains why a short sample can look cleaner than a long one even with the
same model checkpoint.

### What I Would Check In A Real Run

When I train the music model for more than a smoke test, I should not judge it
only by loss. I need to inspect generated artifacts:

- does the text contain complete song snippets starting with `X:`?
- are metadata fields such as `T:`, `M:`, and `K:` present?
- does the generated body use mostly valid ABC note symbols?
- does extraction find at least one complete song block?
- can `abc2midi` parse the first generated snippet?
- does the audio sound like a tune rather than random note noise?

Those checks matter because cross entropy can improve before the generated music
is structurally valid.

### Lab 1 Takeaway

Lab 1 is not just a setup exercise. It establishes the whole course workflow:

- represent data as tensors
- build a differentiable model
- compute a task-appropriate loss
- use backpropagation and an optimizer
- inspect shapes and outputs instead of trusting the notebook
- connect the lecture diagram to a runnable training loop

That is why the local Lab 1 folder now has small scripts instead of one large
notebook. Each script is a checkpoint for one thing I need to understand before
the later models become larger and harder to debug.

## Lab 2

Lab 2 moves from sequence modeling into computer vision and then into a more
applied question: what happens when a model performs well on average but unevenly
across groups?

The official PyTorch path is split into:

- Part 1: MNIST digit classification
- Part 2: facial detection and debiasing

I finished the local pass through both parts. The repo does not contain the
large official face datasets, so "finished" here means:

- the MNIST section has runnable training/evaluation code
- the facial detection section has runnable synthetic mechanics code
- the notes cover the official lab flow, loss functions, VAE pieces, and
  subgroup evaluation logic
- I am not claiming a real CelebA/ImageNet/PPB fairness result from the local
  synthetic run

### First Concepts To Keep Straight

- images are represented as `(batch, channels, height, width)` in PyTorch
- MNIST images are grayscale, so the channel count starts at `1`
- a convolution learns local pattern detectors and outputs multiple feature maps
- max pooling reduces spatial size, which makes later layers cheaper
- the convolutional feature map has to be flattened before the fully connected classifier
- the final layer should output 10 logits, one for each digit class
- `CrossEntropyLoss` wants logits, not softmax probabilities
- binary face detection uses one logit and `BCEWithLogitsLoss`
- `model.train()` and `model.eval()` matter because layers like batch norm behave
  differently in training and inference
- fairness evaluation needs subgroup metrics, not just one average accuracy

### Shape Path In The Starter CNN

For a synthetic MNIST-like batch shaped `(8, 1, 28, 28)`:

- `conv1`, 1 -> 24 channels with a 3x3 kernel: `(8, 24, 26, 26)`
- `pool1`, 2x2 max pooling: `(8, 24, 13, 13)`
- `conv2`, 24 -> 36 channels with a 3x3 kernel: `(8, 36, 11, 11)`
- `pool2`, 2x2 max pooling: `(8, 36, 5, 5)`
- flatten: `(8, 900)`
- classifier head: `(8, 10)`

This shape path is the reason the first fully connected layer after flattening
is `nn.Linear(36 * 5 * 5, 128)`.

### Part 1 Completion: MNIST Training Comparison

The fourth Lab 2 script is the first one that actually trains models:

`labs/lab2_facial_detection_systems/scripts/04_mnist_training_comparison.py`

It compares:

- fully connected baseline: flatten -> linear 784 to 128 -> ReLU -> linear 128
  to 10
- CNN: conv/pool -> conv/pool -> flatten -> dense head

The important implementation pieces:

- use `DataLoader` for minibatches
- use `CrossEntropyLoss` on raw logits
- use `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`
- accumulate train loss and train accuracy by number of examples, not just by
  number of batches
- switch to `model.eval()` and `torch.inference_mode()` for held-out evaluation
- compute a confusion matrix with rows as true labels and columns as predictions

The script defaults to a synthetic seven-segment digit dataset so it can run
offline. That is useful for checking the training loop, but the real lab result
should be run with:

`--source mnist --download`

The main conceptual result I expect from real MNIST:

- the dense baseline should already do well because MNIST is simple
- the CNN should be a better architectural fit because it preserves local image
  structure before flattening
- the gap is not just about parameter count; it is about inductive bias

### Part 2 Completion: Facial Detection And Bias

The fifth Lab 2 script is:

`labs/lab2_facial_detection_systems/scripts/05_facial_debiasing_mechanics.py`

It implements the moving parts from the facial detection section:

- standard CNN binary classifier
- sigmoid probabilities only for interpretation, not as a separate training
  layer before the loss
- grouped evaluation over four test groups in the style of the official lab
- VAE encoder outputs for `y_logit`, `z_mean`, and `z_logsigma`
- decoder reconstruction from sampled latent variables
- DB-VAE loss with classification loss for all examples and VAE loss only for
  face examples
- inverse-density sampling over latent variables to upweight rare regions

The official lab's data setup:

- positive examples: CelebA faces
- negative examples: ImageNet non-face images
- test examples: balanced face groups for comparing subgroup performance
- groups: light/dark skin tone crossed with gender labels

The local script creates synthetic examples instead. This is realistic for this
repo because downloading the official datasets and training the full model is a
separate GPU/Colab job. The synthetic run verifies that the code path connects:
forward pass, loss, reparameterization, reconstruction, sampling probabilities,
and group summaries.

### What I Learned From Finishing Lab 2

Computer vision is not just "use CNNs." The architecture, loss, and evaluation
metric have to match the question:

- MNIST classification asks for one of ten classes
- face detection asks for a binary decision
- debiasing asks whether the binary decision behaves consistently across
  subgroups
- VAE-based debiasing asks the model to learn latent structure inside the face
  class, then use that structure to resample training data

The DB-VAE idea also clarified a useful distinction:

- class imbalance: too many examples of one label compared with another
- latent imbalance: uneven representation inside a label

The second problem is harder. A dataset can have many face examples and still be
biased if those examples are concentrated around a narrow set of appearances,
lighting conditions, poses, or camera qualities.

## Lecture 7: The Three Laws Of AI

Status: full-material pass complete as of June 8, 2026. These notes cover the
lecture framing, the Asimov-to-AI move, why literal rule checking is not enough,
observability, evaluations, long-context safety drift, agents, and the final
modern laws.

Lecture 7 is a safety and evaluation lecture. The earlier lectures mostly asked
how to train models: choose an architecture, choose a loss, compute gradients,
and evaluate held-out behavior. This lecture asks a different question:

`What would make an AI system trustworthy enough to use when its output can
affect real people?`

The key word is "system." A deployed model is not only a function
`f(x; theta)`. It is surrounded by prompts, memory, tools, logs, user interfaces,
guardrails, evaluation datasets, escalation rules, and people who decide what
to do when something fails. The model can be strong and the total system can
still be unsafe.

The lecture uses Asimov's robot laws as a starting point. The original idea was
fictional, but it is useful because it makes the desired safety hierarchy easy
to see:

- do not harm people
- follow human direction only when that does not cause harm
- preserve the machine only when that does not conflict with the higher rules

A later version adds humanity-level harm above individual harm. That addition is
important for AI because many deployed systems have population-scale effects.
Recommendation systems, fraud detection models, hiring tools, medical triage
systems, and autonomous agents can cause harm without a single dramatic robot
action. A system can shift incentives, exclude groups, amplify misinformation,
or produce dangerous advice at scale.

### Why Literal Safety Rules Fail

The tempting engineering version of Asimov is:

```text
if action_is_safe(action):
    execute(action)
```

That looks clean, but it hides the hardest problem inside `action_is_safe`.
Modern AI systems do not operate in a small world where every action and
consequence can be enumerated. The system has to interpret language, infer
context, deal with ambiguity, and predict downstream effects. Those are exactly
the parts that are uncertain.

For symbolic AI, a rule can sometimes be checked because the world is represented
as tokens and logic. For deep learning, the representation is usually
distributed across many real-valued activations. The model does not carry a
single explicit variable called `harm`. It has patterns learned from data,
weights shaped by optimization, and behavior that depends on context.

This connects to the lecture's contrast between two styles of AI:

- formal systems: tokens, rules, discrete states, centralized control, explicit
  syntax
- learned systems: patterns, generalization, real-valued representations,
  distributed computation, graceful degradation, blurred syntax and semantics

The strength of deep learning is also part of the safety difficulty. A neural
network can generalize beyond handwritten rules, but the same generalization
means I cannot prove its behavior by reading one rules file. Safety has to move
from "the rule exists" to "the system is repeatedly tested, monitored, and
bounded."

### AI History Through This Lens

The history matters because it explains why this lecture appears near the end
of 6.S191 instead of at the beginning. The course first builds up the deep
learning machinery. Only after sequence models, vision, generative modeling,
reinforcement learning, and frontier models does the safety question become
fully concrete.

The rough arc:

1. Early AI imagined intelligence as explicit symbolic reasoning.
2. Classical symbolic approaches struggled with messy perception, language, and
   real-world ambiguity.
3. Deep learning became practical when data grew, GPUs became useful for neural
   networks, networks became deeper, automatic differentiation made training
   easier, and industry invested heavily.
4. After 2017, transformer-style systems and large-scale training made language
   models central to many deployed AI products.

My interpretation: the field moved from writing rules for intelligence to
training systems that infer patterns. That made AI much more capable, but it
also moved failure modes away from obvious syntax errors. A harmful output can
look fluent. A biased classifier can have good average accuracy. An agent can
complete a local task while violating a broader constraint. That is why
evaluation becomes a first-class part of safety.

### What This Changes From Earlier Lectures

In Lecture 1, the core loop was:

1. forward pass
2. loss
3. backward pass
4. optimizer step

In Lecture 7, the deployment loop is closer to:

1. define the behavior I want
2. build a dataset of situations that test that behavior
3. run the model or agent on those situations
4. log the inputs, outputs, tool calls, and intermediate steps
5. score the behavior with metrics or human review
6. inspect failures
7. add new tests for those failures
8. change the prompt, model, tool policy, or system boundary
9. repeat before and after deployment

This is a different kind of rigor. Training loss tells me whether an optimizer
reduced a numerical objective. Safety evaluation asks whether the system behaves
acceptably under cases I care about. The hard part is that the test set is never
complete. A good evaluation process therefore has to keep growing as new failure
modes are discovered.

This also makes the connection to my Lab 3 notes sharper. Fine-tuning an LLM is
not finished when a sample sounds closer to the target style. I need controls,
held-out prompts, repeatable scoring, and failure examples. For any system with
real users, I would also need trace logging and regression tests so a prompt or
model update does not quietly reintroduce old failures.

### Observability: Making The System Inspectable

The lecture's practical tool example is Opik, which is presented as an
LLM-observability and evaluation platform. I do not need to treat the specific
platform as the only possible answer, but the concepts are important because
they define what has to be recorded in any serious LLM application.

The problem is that an LLM workflow is not usually one model call anymore. A
single user request can involve:

1. reading the current message
2. injecting system instructions
3. retrieving context from files, memory, or a vector database
4. choosing tools
5. calling those tools
6. summarizing tool outputs
7. making another model call
8. producing the final response

If I only save the final answer, I lose the evidence needed to debug the
system. A bad final answer could come from the user prompt, the system prompt,
retrieval, tool selection, a tool error, an overly permissive memory, a model
failure, or a bad evaluator. Observability means recording enough of the
pipeline that I can locate which part failed.

The lecture vocabulary:

- message: one piece of text from the user or model
- chat prompt: the conversation history and instructions sent to the model
- trace: the complete record of one LLM workflow
- span: one unit of work inside a trace
- project: a container that organizes related traces and spans

The trace/span distinction is especially useful. A trace is the full run. A span
is a step inside the run. For a tool-using agent, a trace might cover the whole
question-answer interaction, while spans cover retrieval, model calls, parsing,
tool calls, scoring, and final formatting.

The shape is similar to a computation graph, but for an application workflow
instead of tensor operations:

```text
trace: answer one user request
    span: build prompt
    span: retrieve course notes
    span: call model
    span: call tool
    span: call model again
    span: run evaluator
```

What I would want to log for each span:

- input text or structured input
- output text or structured output
- model name or tool name
- prompt/template version
- latency
- token counts or cost if relevant
- errors and retries
- metadata such as dataset item, experiment name, and safety category

This is not only for debugging crashes. It is for debugging behavior. If a model
starts failing on long conversations, or a new prompt improves style but weakens
safety, I need traces from before and after the change. Otherwise I am judging
from memory and a few hand-picked examples.

### Evaluations: Turning Informal Testing Into Experiments

The lecture asks how we know whether an LLM is performing well, then gives the
answer: turn informal playground testing into a scientific process. That means
building datasets, choosing metrics, running experiments, and comparing results
across model or prompt versions.

The evaluation vocabulary:

- evaluation: applying a metric to an LLM result
- dataset: a collection of user queries or test cases
- metric: a measurement of correctness or quality
- experiment: running dataset items through a system and scoring them with one
  or more metrics

This maps cleanly onto the ML habits from earlier lectures:

- training set becomes evaluation dataset
- loss/accuracy becomes LLM-specific metrics
- model checkpoint becomes prompt/model/tooling version
- validation run becomes experiment
- error analysis becomes trace inspection

But LLM evaluation is messier than MNIST accuracy. A digit classifier has a
label like `7`. Many LLM tasks have partially correct answers, style
requirements, safety boundaries, and ambiguous user intent. That is why the
lecture mentions built-in metrics and LLM-as-a-judge evaluation.

Useful metric types:

- exact match for tasks with one correct string
- contains/does-not-contain checks for required or forbidden content
- regex or parser checks for structured outputs
- unit tests for code-generation tasks
- retrieval checks for whether the answer used the right source
- refusal/safety checks for high-risk requests
- LLM-as-a-judge scores for qualities that are hard to encode as rules

LLM-as-a-judge is useful, but it is not magic. A judge model can be biased,
inconsistent, overconfident, or fooled by fluent wrong answers. I should treat
it as another measurement instrument, not as ground truth. For important
evaluations, I would want calibration examples:

- clearly good answers
- clearly bad answers
- borderline answers
- adversarial answers that sound polished but violate the rubric
- human spot checks on judge disagreements

An experiment should make comparison possible. The minimum version I would want
to track:

```text
experiment name: lecture7_safety_prompt_v2
dataset: high_risk_and_normal_user_requests
system version: prompt v2 + model A
metrics: refusal quality, helpfulness, policy compliance, source use
outputs: scores plus trace links for every item
```

Then I can ask real questions:

- Did the new prompt improve safety without making normal answers useless?
- Did a model upgrade change refusal behavior?
- Are failures concentrated in long-context cases?
- Are tool calls helping or creating new mistakes?
- Which dataset items should become regression tests?

The most important habit is to keep failed cases. If I only report aggregate
scores, I lose the examples that teach me what to fix. If an LLM gives harmful
advice after a long conversation, that failure should become a test case with
the full conversational context preserved.

### Connection To My Lab 3 Evaluation

My local Lab 3 scripts already have a small version of this idea:

- base-style controls
- target-style controls
- generated samples
- held-out style loss
- a simple local rubric

Lecture 7 shows what that should become in a real LLM system. Instead of only
printing a few examples, I would store the prompts, outputs, scores, and run
metadata. Instead of judging by vibe, I would create a dataset with specific
cases and rerun it whenever I changed the prompt, model, adapter, or decoding
settings.

For the Yoda-style fine-tuning lab, a better evaluation dataset would include:

- ordinary questions where the answer should stay factually correct
- style-transfer questions where the target style should appear
- prompts where style should not override safety
- prompts that tempt the model to overdo the style and become unreadable
- held-out examples not seen during adapter training

Metrics could include:

- answer-token loss on held-out style examples
- simple style-marker score
- readability score or length limits
- exact factual checks for small controlled facts
- safety refusal checks for unsafe requests

This is also where tracing would matter. If a generated answer is bad, I want to
know the prompt template, decoded input, sampled tokens, temperature, model
weights, adapter setting, and evaluation score. Without that, the output is hard
to reproduce.

My takeaway from this part: an LLM application needs the same discipline I
would expect from a normal ML experiment, but applied to prompts, tool calls,
memory, traces, and qualitative outputs. Playground testing is useful for
exploration. It is not enough evidence for trust.

### Long-Context Safety Drift

The most serious example in the lecture is a reported case where a long-running
chatbot relationship moved from an initial safety warning into increasingly
personalized unsafe guidance. I am not treating the news report as a technical
postmortem with every causal detail settled. The point for this course is the
failure pattern: a model can appear safer in a one-turn test than it is inside a
long, personalized conversation.

That distinction matters because most quick safety demos look like this:

```text
user asks risky question
model refuses or gives a warning
test passes
```

But a real user interaction can look more like this:

```text
turn 1: risky question
turn 2: user reframes the risk
turn 3: model remembers emotional context
turn 4: user asks for a less direct version
turn 5: model gives personalized guidance
```

The output at turn 5 is not independent of turns 1-4. The model is conditioned
on the accumulated context, and the product may also use stored memory or prior
conversation history. That creates a safety problem that does not show up if I
only evaluate isolated prompts.

The lecture's recommendations for this failure mode are practical:

- limit how much user history can influence the model
- counteract safety degradation in long conversations
- make the model intentionally bad at giving harmful personalized guidance
- provide explainable and user-controlled personalization
- test slow safety failures instead of only one-off prompts
- route high-risk topics to specialized systems
- strengthen organizational and operational safeguards
- bias the model toward safer failure modes

The phrase "intentionally bad" is important. In normal ML work I usually want
better task performance. For safety-critical misuse, I want the opposite: the
model should be bad at helping with the harmful part, even if the user keeps
trying to make the request sound reasonable.

This changes how I would design a safety evaluation dataset. It should include
multi-turn trajectories, not only single prompts:

- direct unsafe request followed by softer rewording
- repeated requests after an initial refusal
- emotional dependence on the model
- attempts to make the model personalize risky guidance
- long context where earlier facts quietly change the answer
- benign conversations mixed with later high-risk turns

The metric also has to be sequential. A model that refuses on turn 1 but gives
harmful details on turn 6 has failed. I would score the whole conversation, not
only each response independently.

The trace should preserve:

- the full conversation window
- which memories were retrieved or applied
- whether a safety classifier or policy check fired
- whether the model escalated or routed the conversation
- how the answer changed as the user reframed the request

This is a direct connection to Lecture 2 and Lecture 6. Sequence models use
context, and frontier models can use very long context. That capability is
useful for personalization and continuity, but it also means safety has to be
tested across time. A long context is not just more tokens. It is a larger
behavioral state.

### What Changes When The System Is An Agent

The lecture then moves from LLM chat to agentic AI. The short definition I want
to keep is:

`An agent is a system that can do things on behalf of the user.`

That is a clean boundary. A normal chatbot produces text. An agent can observe,
plan, call tools, write files, send messages, run code, move money, control
devices, or coordinate with other agents depending on what permissions it has.

The risk changes because the output is no longer only advice. The system can
take actions. A bad plan can become a real-world change before a human notices.

The basic agent loop:

1. observe the state
2. infer the user's goal
3. plan one or more steps
4. choose a tool or action
5. execute the action
6. observe the result
7. update the plan
8. stop or continue

That loop resembles reinforcement learning from Lecture 5:

- the agent observes state
- the agent chooses actions
- actions change the environment
- later states depend on earlier actions
- local rewards or goals can conflict with broader safety

The difference is that many LLM agents are not trained end-to-end in the
environment where they are deployed. They often combine a language model,
prompts, tools, memory, retrieval, and product-specific policies. That makes
observability even more important: if something goes wrong, I need to know
whether the failure came from reasoning, retrieval, tool selection, tool output,
permissions, or the stop condition.

For an agent, the safety boundary should be around actions, not only final text.
Examples:

- read-only search can be lower risk than writing to an external system
- drafting an email is lower risk than sending it
- suggesting a shell command is lower risk than running it
- summarizing a financial report is lower risk than placing a trade
- identifying a medical concern is lower risk than giving personalized
  treatment instructions

This suggests a permissions ladder:

1. answer from existing context
2. retrieve read-only information
3. draft an action for human approval
4. execute reversible low-risk actions
5. execute irreversible or high-risk actions only with strong controls, or not
   at all

The agent should also have stop conditions. A system that keeps trying to
satisfy the user after repeated safety conflicts is exactly the wrong behavior.
For high-risk topics, a safer agent should narrow, route, or stop instead of
becoming more compliant.

### Evaluation For Agents

Agent evaluation has to include the trajectory, not just the final answer.
Questions I would ask:

- Did the agent choose the right tool?
- Did it use the minimum necessary permission?
- Did it ask for approval before an irreversible action?
- Did it preserve important context from earlier steps?
- Did it stop when the task became unsafe or unclear?
- Did it leave a trace detailed enough to audit afterward?

The dataset should contain tasks with expected traces, not only expected final
responses. For example:

```text
task: summarize a file and draft an email
expected behavior:
    read the file
    summarize the relevant points
    draft the email
    ask before sending
failure:
    sends the email without approval
```

This is the part of Lecture 7 that feels most practically important for future
projects. If I build an agent, I should not start by giving it every tool and
hoping the model behaves. I should start with narrow permissions, trace
everything, create an evaluation dataset, and add privileges only when I can
test the new failure modes.

### The Modern Laws As Governance Principles

Near the end, the lecture gives a modern version of AI laws that draws from
regulatory and standards work: the EU AI Act, NIST AI Risk Management Framework,
OECD AI Principles, IEEE ethics work, human-in-the-loop standards, and related
AI governance documents.

The first version is broad:

1. AI systems must be transparent enough for people to understand and contest
   outcomes.
2. AI systems must be safe, secure, and robust.
3. AI systems must be aligned with human direction through transparent,
   accountable oversight.
4. AI systems must respect human rights, fairness, and societal values.

This is the right level for policy, but it is not yet enough for engineering.
"Be fair" and "be robust" are necessary goals, but a developer still has to turn
them into tests, logs, release gates, and system behavior.

How I would translate those broad laws:

- transparency means users can understand the role of the model, challenge
  important outputs, and see when automation is involved
- safety and security mean the model is tested for misuse, protected from
  prompt injection and unsafe tool use, and monitored after release
- alignment with human direction means human instructions matter, but only
  inside safe and accountable boundaries
- human rights and fairness mean subgroup behavior, access, dignity, and
  downstream effects are part of evaluation

The important phrase is "contest outcomes." A system is not transparent just
because it prints a confidence score or says it used AI. Contestability means a
person has a path to question, appeal, correct, or escalate an outcome. That is
especially important for decisions about school, work, finance, healthcare,
housing, law, or public services.

This connects back to Lab 2. A face detector with strong average accuracy is
not enough if a subgroup experiences much higher false negatives. A transparent
system should make those differences measurable, and a contestable system
should not leave users trapped by an automated decision.

### The Modern Laws As Engineering Practice

The final version of the laws is more operational:

1. Log traces, use online evaluation, and inspect failures.
2. Build a dataset of tests and keep adding to it.
3. Evaluate prompts on the dataset and model often.
4. Be transparent, for example by publishing datasets and evaluation results.

Then the lecture adds two stronger constraints:

- if safety and security cannot be guaranteed, do not deploy
- AI systems may not harm humanity, or through inaction allow humanity to come
  to harm

The difference between version 1 and version 2 is useful. Version 1 says what
values the system should satisfy. Version 2 says what process should exist so I
can gather evidence. The process is what makes the values concrete.

The operational loop:

```text
collect traces
inspect failures
turn failures into dataset items
run evaluations across prompt/model versions
compare results
publish or document what was tested
hold deployment if the risk is not controlled
```

That loop is basically safety-oriented MLOps for LLMs and agents. It treats
prompt changes like code changes. It treats model upgrades like dependency
upgrades. It treats failures as regression tests. It treats transparency as
part of the release artifact, not a sentence in a slide deck.

The line about not deploying is easy to say and hard to follow. It means the
system should have release gates. A release gate is a condition that must be met
before deployment:

- critical safety dataset passes
- no known high-severity failures remain unhandled
- high-risk tools require confirmation
- long-context safety tests pass
- privacy and security checks pass
- trace logging works in production
- escalation and rollback paths exist

For a small student project, this can be lightweight. For example, before
publishing an agent demo, I could require:

- all local tests pass
- a small unsafe-request dataset produces safe refusals
- the agent cannot write outside an allowed directory
- irreversible actions require confirmation
- every run produces a trace
- known failure examples are documented

The scale changes, but the habit is the same.

### Safety, Security, And Robustness Are Different

The phrase "safe, secure, and robust" contains three separate requirements:

- safety: the system avoids harmful behavior
- security: the system resists misuse, attacks, and unauthorized access
- robustness: the system behaves acceptably under distribution shift, noise,
  ambiguity, and unexpected inputs

It is possible to satisfy one and fail another. A model could refuse dangerous
medical advice but leak private data. That would be a security failure. A model
could pass a standard benchmark but fail when the user misspells key terms. That
would be a robustness failure. A model could answer accurately but take an
unsafe action through a tool. That would be a safety failure.

For LLM agents, security becomes especially concrete:

- prompt injection can try to override system instructions
- retrieved documents can contain malicious instructions
- tools can expose private data
- memory can preserve information that should expire
- an agent can chain low-risk actions into a high-risk outcome

That means agent safety should not depend on the model ignoring every malicious
instruction. The surrounding system should restrict tools, validate inputs,
separate trusted and untrusted context, and keep logs that make attacks visible.

### My Final Lecture 7 Takeaway

Lecture 7 makes the strongest argument so far that model quality and system
trustworthiness are not the same thing. A model can be capable, fluent, and
useful while still being unsafe in long contexts, vulnerable to bad tool use, or
too opaque for high-stakes deployment.

The lesson I want to carry forward:

1. Safety has to be evaluated, not assumed.
2. One-turn tests are not enough for long-context systems.
3. Agents need action-level controls, not only answer-level guardrails.
4. Traces turn failures into inspectable evidence.
5. Datasets should grow from real failures.
6. Prompt and model changes need repeated evaluation.
7. If the risk cannot be bounded, the system should not be deployed.

This is a useful ending to the core AI-safety arc of the course because it
connects almost every previous lecture:

- Lecture 1: optimization can improve a loss without guaranteeing safe behavior
- Lecture 2: sequence context creates multi-turn failure modes
- Lecture 3: average accuracy can hide subgroup errors
- Lecture 4: generative models can create plausible but wrong outputs
- Lecture 5: agents can optimize local goals in unsafe ways
- Lecture 6: frontier models amplify uncertainty, bias, scale, and deployment
  risk
- Lab 3: fine-tuned LLMs need repeatable evaluation and traceable outputs

My strongest practical rule after this lecture: do not build LLM or agent
systems where the only evidence is that a few examples looked good. Build the
tests, keep the traces, inspect the failures, and make deployment depend on the
evidence.

## Lecture 8: AI For Science

Status: full materials are now open. I have not started the full Lecture 8 pass
yet.

The earlier abstract-level notes for AI for Science were based on the public
course description. The full pass should rebuild this section from the slides
and video, especially the scientific discovery loop, simulator/emulator split,
inductive bias in scientific models, molecular and materials examples, and the
limits of current simulation.

## Lecture 9: Secrets To Massively Parallel Training

Status: full materials are now open. I have not started the full Lecture 9 pass
yet.

The earlier abstract-level notes for massively parallel training were based on
the public course description. The full pass should rebuild this section from
the slides and video, especially scaling laws, memory accounting, activation
checkpointing, offloading, data/tensor/pipeline/context parallelism, sharding,
expert parallelism, and the LFM2 case study.

## Draft Course Synthesis

The course is starting to read to me as one connected arc. This synthesis still
needs a final pass after I finish Lectures 7, 8, and 9 from the full materials.

1. Neural networks are differentiable function approximators trained by loss
   minimization.
2. Sequence models add time, memory, and next-token prediction.
3. CNNs add spatial inductive bias for images.
4. Generative models learn data distributions and latent structure.
5. RL changes the setup from prediction to action under delayed reward.
6. Frontier models scale these ideas but also amplify robustness, bias,
   uncertainty, and deployment problems.
7. The safety lecture shifts the unit of analysis from model to deployed
   system: logs, tests, tools, memory, guardrails, and escalation paths matter.
8. The science and large-scale training lectures should extend this into
   external validation and systems engineering once I finish the full pass.

The biggest technical habits I built:

- check tensor shapes before trusting a model
- distinguish logits, probabilities, losses, and metrics
- separate training loss from held-out behavior
- inspect subgroup performance instead of only average accuracy
- treat generation quality as an evaluation problem, not only a sampling problem
- keep local mechanics separate from official large-scale results
- write down what a model has not proven yet

The labs gave me a practical progression:

- Lab 1: tensors, autograd, dense layers, RNN music generation
- Lab 2: CNNs, MNIST training, facial detection, VAE-based debiasing
- Lab 3: chat formatting, answer masking, causal-LM loss, LoRA-style adapters,
  generation, and judge-style evaluation

My strongest remaining project direction is to extend Lab 2:

`Can latent-space resampling improve worst-group face-detection performance
without hiding tradeoffs behind average accuracy?`

That project is feasible because the repo already has:

- a standard CNN face detector
- a DB-VAE mechanics implementation
- grouped evaluation code
- synthetic data for offline smoke tests

The real project version would need:

- the official CelebA/ImageNet/PPB data path
- a GPU-backed training run
- overall and subgroup metrics
- false positive and false negative rates by group
- calibration or confidence analysis by group
- examples of failure cases

The result I would want is not just "DB-VAE is better." A useful result would
show the tradeoff:

- average accuracy
- worst-group accuracy
- subgroup false negatives
- subgroup confidence
- how resampling changes those metrics
- where the method still fails

That would connect the technical core of the course to the part I care about
most: building models carefully enough that I can explain both what they do and
where the evidence stops.
