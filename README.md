# MIT 6.S191 Independent Study

## Why I'm doing this

I'm a high school junior in the Class of 2027, and this repository is my working log for independently studying [MIT 6.S191: Introduction to Deep Learning](https://introtodeeplearning.com/).

I'm building this project because I want to move past surface-level ML tutorials and get comfortable with the actual mechanics: implementing models, understanding the math well enough to debug them, and developing the kind of technical habits that transfer to future research. I care about the theory, but I want to learn it through code, experiments, and iteration.

## Learning Goals

- Master PyTorch well enough to build, train, and inspect models without relying on black-box abstractions.
- Understand backpropagation, gradient descent, and optimization from both the mathematical and implementation side.
- Explore generative models through hands-on lab work, especially sequence models and modern deep learning workflows.
- Bridge the gap between beginner projects and academically rigorous machine learning by reproducing ideas carefully instead of just following notebooks.

## Lab Progress

This is my tracker for the official MIT 6.S191 software labs.

| Lab | Status |
| --- | --- |
| [x] Software Lab 1: Deep Learning in Python + Music Generation | Complete |
| [x] Software Lab 2: Facial Detection Systems | Complete local pass: MNIST dense-vs-CNN training comparison, facial detection/DB-VAE mechanics, grouped bias evaluation notes, and lecture/lab write-up |
| [x] Software Lab 3: Fine-Tune an LLM, You Must! | Complete local pass: chat templates, tokenization, answer masking, tiny causal-LM training, LoRA-style adapter tuning, offline style evaluation, held-out style-loss proxy, and detailed notes |

I plan to keep updating this as I finish each official lab, re-implement sections in PyTorch where useful, and write up the parts that are mathematically interesting or practically non-obvious.

## Lecture Progress

| Lecture | Status |
| --- | --- |
| [x] Lecture 1: Intro to Deep Learning | Complete notes |
| [x] Lecture 2: Deep Sequence Modeling | Complete notes |
| [x] Lecture 3: Deep Computer Vision | Complete notes plus Lab 2 bridge |
| [x] Lecture 4: Deep Generative Modeling | Complete notes plus DB-VAE bridge |
| [x] Lecture 5: Deep Reinforcement Learning | Complete notes through DQN, policy gradients, actor-critic, simulation, and AlphaGo/AlphaZero |
| [x] Lecture 6: Language Models and New Frontiers | Complete archived-deck notes on limitations, generalization, adversarial examples, diffusion models, protein generation, LLMs, scaling, and foundation models |

## Current Manual Commit Points

I am not auto-committing study progress. The next realistic split is:

1. `Finish Lecture 5 policy-gradient notes`
2. `Add Lecture 6 limitations and robustness notes`
3. `Finish Lecture 6 new-frontiers notes and tracker`

## Hardware / Setup

All of this is running on a local Ubuntu server. Part of the point of this repo is not just learning deep learning models, but also getting comfortable with the systems side: environment setup, dependency management, remote workflows, and the kind of lightweight edge-compute/admin work that makes experiments reproducible.

## Credits

- Official course site: [MIT 6.S191: Introduction to Deep Learning](https://introtodeeplearning.com/)
- Official labs/materials: [MITDeepLearning/introtodeeplearning](https://github.com/MITDeepLearning/introtodeeplearning)
