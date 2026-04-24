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

My original stopping point was before the full training run:

- understand shapes
- inspect a real or synthetic batch
- define the baseline model carefully
- hold off on marking Lab 2 complete until I train/evaluate the MNIST models
  and implement the facial detection/debiasing mechanics

That stopping point has now been cleared for the local repo. The only remaining
caveat is that real CelebA/ImageNet/PPB numbers require the official dataset
path and a GPU-backed run.

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

That is the natural stopping point for this commit. The next part of the
lecture moves into policy-gradient methods as a way around some of these
limitations.

## Software Lab 3: LLM Fine-Tuning

The current 2026 course site lists Software Lab 3 as "Fine-Tune an LLM, You
Must!" under the newer-frontiers part of the course. The official lab notebook
is built around a real LLM stack:

- base model: Liquid AI LFM2-1.2B
- fine-tuning method: LoRA / parameter-efficient fine-tuning
- judge model: Gemini 2.5 through OpenRouter
- evaluation/monitoring: Comet Opik

My repo version is deliberately smaller. I finished a local mechanics pass, not
the official GPU/API competition path. That distinction matters because the
official lab is about real LLM behavior at scale, while this repo is checking
whether I understand the pieces well enough to run and debug the full notebook
later.

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

My local script uses a tiny character-level tokenizer but keeps the chat markers
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

My local script mirrors the idea:

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

### What Counts As Finished For My Local Lab 3

I am marking Lab 3 complete in this repo because I now have runnable local
checks for:

- chat templates
- tokenization and decoding
- next-token labels
- answer masks
- tiny causal-LM training
- LoRA-style adapter tuning
- generation
- style controls
- held-out style loss

What remains for a real course/competition submission:

- run the official notebook on a GPU
- use the real LFM2-1.2B tokenizer and model
- fill the TODOs against the actual Hugging Face/PEFT APIs
- use OpenRouter with an API key for the judge model
- log traces and metrics through Opik
- submit the notebook/report with the official final likelihood cell

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

### Two Manual Commit Points For This Progress

I did not commit automatically. If I commit this work manually, the two realistic
commit points are:

1. `Finish Lab 2 MNIST training comparison`
   - Adds the real training/evaluation loop for dense vs CNN MNIST models.
   - Moves Lab 2 Part 1 from probes to a complete local training comparison.

2. `Finish Lab 2 facial debiasing mechanics and notes`
   - Adds the local DB-VAE/facial-detection mechanics script.
   - Expands Lecture 4 and Lab 2 notes enough that future me can explain the
     method without reopening the notebook immediately.
