# Lab 3: Fine-Tune an LLM, You Must!

This folder tracks my local independent-study pass through MIT 6.S191 Software
Lab 3.

The official lab fine-tunes Liquid AI's LFM2-1.2B model, uses Gemini 2.5 as an
LLM-as-judge evaluator through OpenRouter, and logs/monitors evaluations with
Comet Opik. I am not checking a multi-billion-parameter model, API keys, or
cloud traces into this repo. Instead, this folder contains a small offline
PyTorch version of the same mechanics so I can verify the ideas locally before
running the official notebook path.

## Current Status

- Added the Lab 3 folder structure.
- Added a shared utility module for chat templates, tiny character-level
  tokenization, style examples, masked causal-LM loss, a tiny causal
  transformer, LoRA-style adapters, generation, and simple offline style
  scoring.
- Added a first script that checks the prompt template, encode/decode path,
  answer-token masking, and next-token labels.
- Added a second script that trains a tiny base causal LM, freezes it, and
  fine-tunes only low-rank adapter weights on Yoda-style answers.
- Added a third script that evaluates base-style controls, target-style
  controls, generated samples, and held-out style cross entropy with a
  transparent local rubric.

## What Counts As Finished Here

For this repo, Lab 3 is now finished as a local mechanics pass. The scripts
cover the official pipeline at a mechanics level:

- format question/answer pairs with a chat template
- tokenize and decode text
- train with next-token prediction
- compute loss only on answer tokens
- adapt a model with a LoRA-style low-rank update while freezing base weights
- evaluate generated text against base and target-style controls
- compute a held-out style likelihood

That is not the same as completing the official competition run. The real
competition result still requires the official notebook, a GPU-backed runtime,
the LFM2-1.2B checkpoint, OpenRouter judge access, and Opik tracing.

The local model is intentionally tiny and sometimes generates broken text,
especially on held-out prompts. That is a feature of the study setup, not a
hidden result: the point here is to understand the training/evaluation plumbing
well enough to debug the official notebook later.

## Manual Commit Points

No commits have been created automatically. The planned realistic split is:

1. `Start Lab 3 LLM templates and tokenization notes`
   - Lab 3 README and local source caveats
   - shared template/tokenizer/data utilities
   - first prompt/tokenization/masking probe

2. `Add Lab 3 LoRA fine-tuning mechanics`
   - tiny causal transformer
   - base pretraining followed by LoRA-only style tuning
   - parameter-count and generation checks

3. `Finish Lab 3 evaluation notes and progress tracking`
   - offline judge-style evaluation
   - held-out style likelihood
   - README, course notes, and top-level progress updates

## Scripts

- `01_template_tokenization_probe.py`: checks the chat template, encode/decode
  path, answer mask, and shifted next-token targets.
- `02_lora_style_finetuning.py`: trains a tiny causal LM on plain answers, then
  switches to LoRA-only Yoda-style tuning and reports parameter counts, losses,
  generations, and held-out style loss.
- `03_style_judge_and_eval.py`: compares base-style controls, target-style
  controls, generated samples, and held-out target losses with a local scoring
  rubric.

## Useful Commands

```bash
python labs/lab3_llm_finetuning/scripts/01_template_tokenization_probe.py
python labs/lab3_llm_finetuning/scripts/02_lora_style_finetuning.py
python labs/lab3_llm_finetuning/scripts/03_style_judge_and_eval.py
```
