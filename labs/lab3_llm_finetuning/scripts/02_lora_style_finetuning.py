"""Lab 3 Part 2: local LoRA-style fine-tuning mechanics.

The official lab uses PEFT LoRA adapters on a pretrained LFM2-1.2B causal LM.
This script rehearses the same training structure at toy scale:

1. Train a tiny causal transformer on plain English answers.
2. Freeze the base model weights.
3. Train only low-rank adapter weights on Yoda-style answers.
4. Compare trainable parameter counts and generated answers.

The generated text is not meant to be impressive. The point is to make the
fine-tuning mechanics inspectable on CPU without a model download.
"""

from __future__ import annotations

import argparse

import torch

from lab3_utils import (
    generate_answer,
    get_heldout_examples,
    run_local_lora_finetuning,
    sequence_loss,
    yoda_style_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="yoda")
    parser.add_argument("--base-steps", type=int, default=300)
    parser.add_argument("--lora-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def summarize_loss(name: str, losses: list[float]) -> None:
    first = torch.tensor(losses[: min(10, len(losses))]).mean().item()
    last = torch.tensor(losses[-min(10, len(losses)) :]).mean().item()
    print(f"{name}: first-window={first:.3f} last-window={last:.3f}")


def main() -> None:
    args = parse_args()
    result = run_local_lora_finetuning(
        style=args.style,
        base_steps=args.base_steps,
        lora_steps=args.lora_steps,
        seed=args.seed,
    )

    print("=== Parameter counts ===")
    print(f"total parameters: {result.total_parameters:,}")
    print(f"base-training trainable parameters: {result.base_trainable_parameters:,}")
    print(f"LoRA-stage trainable parameters: {result.lora_trainable_parameters:,}")
    pct = result.lora_trainable_parameters / result.total_parameters * 100
    print(f"LoRA-stage trainable percentage: {pct:.2f}%")

    print("=== Training losses ===")
    summarize_loss("base plain-answer training", result.base_losses)
    summarize_loss("LoRA style fine-tuning", result.lora_losses)

    print("=== Generation checks ===")
    probe_questions = [
        "Why use LoRA instead of updating every parameter?",
        "Why mask the prompt tokens during supervised fine-tuning?",
    ]
    for question in probe_questions:
        answer = generate_answer(
            result.model,
            result.tokenizer,
            question,
            temperature=args.temperature,
        )
        score = yoda_style_score(answer)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"offline style score: {score:.2f}")

    held_out = get_heldout_examples(args.style)[0]
    held_out_loss = sequence_loss(
        result.model,
        result.tokenizer,
        held_out.instruction,
        held_out.response_style,
    )
    print("=== Held-out style likelihood proxy ===")
    print(f"masked cross entropy on held-out styled answer: {held_out_loss:.3f}")


if __name__ == "__main__":
    main()
