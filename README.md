# MIT 6.S191 Independent Study

## Why I'm doing this

I'm a high school junior in the Class of 2027, and this repository is my working log for independently studying [MIT 6.S191: Introduction to Deep Learning](https://introtodeeplearning.com/).

I'm building this project because I want to move past surface-level ML tutorials and get comfortable with the actual mechanics: implementing models, understanding the math well enough to debug them, and developing the kind of technical habits that transfer to future research. I care about the theory, but I want to learn it through code, experiments, and iteration.

## Learning Goals

- Master PyTorch well enough to build, train, and inspect models without relying on black-box abstractions.
- Understand backpropagation, gradient descent, and optimization from both the mathematical and implementation side.
- Explore generative models through hands-on lab work, especially sequence models and modern deep learning workflows.
- Bridge the gap between beginner projects and academically rigorous machine learning by reproducing ideas carefully instead of just following notebooks.

## Official Course Snapshot

Source check: I rechecked the active 2026 MIT 6.S191 course page on June 8,
2026. The public materials still use the 6.S191 course label. This is the
schedule I am using for the study log:

| Course item | Official date | My local status |
| --- | --- | --- |
| Lecture 1: Intro to Deep Learning | Mar. 30, 2026 | Foundations notes rewritten |
| Lecture 2: Deep Sequence Modeling | Apr. 6, 2026 | Sequence-modeling notes rewritten |
| Software Lab 1: Deep Learning in Python + Music Generation | After Lecture 2 | Complete local PyTorch mechanics pass |
| Lecture 3: Deep Computer Vision | Apr. 13, 2026 | Vision notes and Lab 2 bridge rewritten |
| Lecture 4: Deep Generative Modeling | Apr. 20, 2026 | Generative modeling and DB-VAE notes rewritten |
| Software Lab 2: Facial Detection Systems | After Lecture 4 | Complete local mechanics pass |
| Lecture 5: Deep Reinforcement Learning | Apr. 27, 2026 | RL notes rewritten through DQN, policy gradients, and actor-critic |
| Lecture 6: New Frontiers | May 4, 2026 | Frontier-model notes rewritten |
| Software Lab 3: Fine-Tune an LLM, You Must! | After Lecture 6 | Local LLM fine-tuning mechanics pass complete |
| Lecture 7: The Three Laws of AI | May 11, 2026 | Full-material notes started: safety framing and Asimov-to-AI history |
| Lecture 8: AI for Science | May 18, 2026 | Full materials open; next lecture to start |
| Lecture 9: Secrets to Massively Parallel Training | May 25, 2026 | Full materials open; notes pending |

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
| [x] Lecture 1: Intro to Deep Learning | Foundations notes rewritten |
| [x] Lecture 2: Deep Sequence Modeling | Sequence-modeling notes rewritten |
| [x] Lecture 3: Deep Computer Vision | Vision notes and Lab 2 bridge rewritten |
| [x] Lecture 4: Deep Generative Modeling | Generative modeling and DB-VAE notes rewritten |
| [x] Lecture 5: Deep Reinforcement Learning | RL notes rewritten through DQN, policy gradients, actor-critic, simulation, and AlphaGo/AlphaZero |
| [x] Lecture 6: Language Models and New Frontiers | Frontier notes rewritten across limitations, diffusion, protein generation, LLMs, scaling, and foundation models |
| [ ] Lecture 7: The Three Laws of AI | In progress: safety framing and Asimov-to-AI history notes added |
| [ ] Lecture 8: AI for Science | Full materials open; next lecture to start |
| [ ] Lecture 9: Secrets to Massively Parallel Training | Full materials open; notes pending |

## Hardware / Setup

All of this is running on a local Ubuntu server. Part of the point of this repo is not just learning deep learning models, but also getting comfortable with the systems side: environment setup, dependency management, remote workflows, and the kind of lightweight edge-compute/admin work that makes experiments reproducible.

## Credits

- Official course site: [MIT 6.S191: Introduction to Deep Learning](https://introtodeeplearning.com/)
- Official labs/materials: [MITDeepLearning/introtodeeplearning](https://github.com/MITDeepLearning/introtodeeplearning)
