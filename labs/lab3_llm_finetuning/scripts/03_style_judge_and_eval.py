"""Lab 3 Part 3: offline style evaluation and held-out likelihood.

The official lab evaluates a Yoda-style model with an LLM-as-judge metric,
negative controls, positive controls, generated samples, Opik tracing, and a
final held-out Yoda-style likelihood. This script keeps the same evaluation
shape but replaces the API judge with a transparent local rubric.
"""

from __future__ import annotations

import argparse

from lab3_utils import (
    generate_answer,
    get_heldout_examples,
    run_local_lora_finetuning,
    sequence_loss,
    summarize_scores,
    yoda_style_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="yoda")
    parser.add_argument("--base-steps", type=int, default=300)
    parser.add_argument("--lora-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def print_score_summary(label: str, scores: list[float]) -> None:
    mean, std = summarize_scores(scores)
    print(f"{label:18s} mean={mean:.2f} std={std:.2f} n={len(scores)}")


def main() -> None:
    args = parse_args()
    result = run_local_lora_finetuning(
        style=args.style,
        base_steps=args.base_steps,
        lora_steps=args.lora_steps,
        seed=args.seed,
    )
    heldout_examples = get_heldout_examples(args.style)

    base_control_scores = [
        yoda_style_score(example.response)
        for example in result.examples + heldout_examples
    ]
    style_control_scores = [
        yoda_style_score(example.response_style)
        for example in result.examples + heldout_examples
    ]

    generated_answers = []
    for example in heldout_examples:
        generated_answers.append(
            generate_answer(
                result.model,
                result.tokenizer,
                example.instruction,
                temperature=0.0,
            )
        )
    generated_scores = [yoda_style_score(answer) for answer in generated_answers]
    heldout_losses = [
        sequence_loss(
            result.model,
            result.tokenizer,
            example.instruction,
            example.response_style,
        )
        for example in heldout_examples
    ]

    print("=== Offline judge controls ===")
    print_score_summary("base English", base_control_scores)
    print_score_summary("target style", style_control_scores)
    print_score_summary("generated", generated_scores)

    print("=== Held-out style likelihood proxy ===")
    loss_mean, loss_std = summarize_scores(heldout_losses)
    print(f"masked cross entropy mean={loss_mean:.3f} std={loss_std:.3f}")

    print("=== Held-out generations ===")
    for example, answer, score, loss in zip(
        heldout_examples,
        generated_answers,
        generated_scores,
        heldout_losses,
    ):
        print(f"Q: {example.instruction}")
        print(f"Generated: {answer}")
        print(f"Target:    {example.response_style}")
        print(f"score={score:.2f} target_loss={loss:.3f}")


if __name__ == "__main__":
    main()

