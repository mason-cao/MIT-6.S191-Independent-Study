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

Source check: I looked up the active 2026 MIT 6.S191 course page on May 15,
2026. This is the schedule I am using for the study log:

| Course item | Official date | My local status |
| --- | --- | --- |
| Lecture 1: Intro to Deep Learning | Mar. 30, 2026 | Foundations notes rewritten |
| Lecture 2: Deep Sequence Modeling | Apr. 6, 2026 | Sequence-modeling notes rewritten |
| Software Lab 1: Deep Learning in Python + Music Generation | After Lecture 2 | Complete local PyTorch mechanics pass |
| Lecture 3: Deep Computer Vision | Apr. 13, 2026 | Complete notes plus Lab 2 bridge |
| Lecture 4: Deep Generative Modeling | Apr. 20, 2026 | Complete notes plus DB-VAE bridge |
| Software Lab 2: Facial Detection Systems | After Lecture 4 | Complete local mechanics pass |
| Lecture 5: Deep Reinforcement Learning | Apr. 27, 2026 | Complete notes through DQN and policy gradients |
| Lecture 6: New Frontiers | May 4, 2026 | Complete core notes, needs final polish |
| Software Lab 3: Fine-Tune an LLM, You Must! | After Lecture 6 | Complete local mechanics pass |
| Lecture 7: AI for Science | May 11, 2026 | Official page still lists public materials as coming soon |
| Lecture 8: Secrets to Massively Parallel Training | May 18, 2026 | Future relative to this May 15 study pass |
| Lecture 9: The Three Laws of AI | May 25, 2026 | Future relative to this May 15 study pass |

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
| [x] Lecture 3: Deep Computer Vision | Complete notes plus Lab 2 bridge |
| [x] Lecture 4: Deep Generative Modeling | Complete notes plus DB-VAE bridge |
| [x] Lecture 5: Deep Reinforcement Learning | Complete notes through DQN, policy gradients, actor-critic, simulation, and AlphaGo/AlphaZero |
| [x] Lecture 6: Language Models and New Frontiers | Complete core notes on limitations, generalization, adversarial examples, diffusion models, protein generation, LLMs, scaling, and foundation models; expansion in progress |
| [ ] Lecture 7: AI for Science | Waiting for public 2026 materials |
| [ ] Lecture 8: Secrets to Massively Parallel Training | Future official lecture date |
| [ ] Lecture 9: The Three Laws of AI | Future official lecture date |

## Hardware / Setup

All of this is running on a local Ubuntu server. Part of the point of this repo is not just learning deep learning models, but also getting comfortable with the systems side: environment setup, dependency management, remote workflows, and the kind of lightweight edge-compute/admin work that makes experiments reproducible.

## Credits

- Official course site: [MIT 6.S191: Introduction to Deep Learning](https://introtodeeplearning.com/)
- Official labs/materials: [MITDeepLearning/introtodeeplearning](https://github.com/MITDeepLearning/introtodeeplearning)
