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

Lab 1 includes both the PyTorch intro section and the RNN music-generation section, so they are tracked together here.
Lab 2 is complete as an independent-study implementation pass. The repo now has runnable local scripts for the MNIST training comparison and for the facial detection/DB-VAE debiasing mechanics. The official CelebA/ImageNet/PPB experiment is too large to check into this repo, so the facial section uses a synthetic offline mechanics check plus detailed notes from the official lab and paper. If I want competition-quality numbers later, I should rerun the official notebook path on a GPU with the real datasets.
Lab 3 is also complete as a local mechanics pass, not as an official competition submission. The repo version checks templates, tokenization, answer masking, tiny causal-LM training, LoRA-style tuning, and evaluation controls locally. The full official run still requires the LFM2-1.2B notebook path, GPU runtime, OpenRouter judge access, and Opik tracing.

I plan to keep updating this as I finish each official lab, re-implement sections in PyTorch where useful, and write up the parts that are mathematically interesting or practically non-obvious.

## Current Manual Commit Points

I have not auto-committed these. If I split the current Lab 3 progress into
three realistic commits, I would use:

1. `Start Lab 3 LLM templates and tokenization notes`
   - Adds the Lab 3 folder, source caveats, shared local utilities, and the
     template/tokenization/masking probe.

2. `Add Lab 3 LoRA fine-tuning mechanics`
   - Adds the tiny causal transformer training path and LoRA-only style
     fine-tuning check.

3. `Finish Lab 3 evaluation notes and progress tracking`
   - Adds offline style-judge evaluation, held-out style loss, detailed Lab 3
     notes, and README progress updates.

## Hardware / Setup

All of this is running on a local Ubuntu server. Part of the point of this repo is not just learning deep learning models, but also getting comfortable with the systems side: environment setup, dependency management, remote workflows, and the kind of lightweight edge-compute/admin work that makes experiments reproducible.

## Credits

- Official course site: [MIT 6.S191: Introduction to Deep Learning](https://introtodeeplearning.com/)
- Official labs/materials: [MITDeepLearning/introtodeeplearning](https://github.com/MITDeepLearning/introtodeeplearning)
