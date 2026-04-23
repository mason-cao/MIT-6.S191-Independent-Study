"""Lab 3 Part 1: chat templates, tokenization, and answer masking.

The official LLM fine-tuning lab starts with formatting conversations, encoding
text as tokens, decoding tokens back into text, and marking which tokens belong
to the assistant answer. This script checks those mechanics locally with a tiny
tokenizer before moving on to model training.
"""

from __future__ import annotations

from lab3_utils import (
    IM_END,
    build_tokenizer,
    encode_supervised_example,
    format_prompt,
    get_style_examples,
)


def main() -> None:
    examples = get_style_examples("yoda")
    sample = examples[0]
    tokenizer = build_tokenizer("yoda")

    prompt = format_prompt(sample.instruction)
    full_ids, answer_mask, full_text = encode_supervised_example(
        tokenizer,
        sample.instruction,
        sample.response_style,
    )

    prompt_ids = tokenizer.encode(prompt)
    decoded_prompt = tokenizer.decode(prompt_ids)
    decoded_answer = tokenizer.decode(
        [
            int(token_id)
            for token_id, is_answer in zip(full_ids.tolist(), answer_mask.tolist())
            if is_answer
        ],
        skip_special_tokens=True,
    )

    print("=== Chat prompt ===")
    print(prompt)
    print("=== Encoded prompt IDs ===")
    print(prompt_ids)
    print("=== Decoded prompt matches ===")
    print(decoded_prompt == prompt)
    print("=== Full supervised example ===")
    print(full_text)
    print("=== Answer-mask summary ===")
    print(f"total tokens: {len(full_ids)}")
    print(f"answer/loss tokens: {int(answer_mask.sum().item())}")
    print("=== Decoded answer-region text ===")
    print(decoded_answer)

    shifted_inputs = full_ids[:-1]
    shifted_targets = full_ids[1:]
    shifted_mask = answer_mask[1:]
    first_loss_position = int(shifted_mask.nonzero()[0].item())

    print("=== Next-token training check ===")
    print(
        "first supervised input token:",
        tokenizer.decode([int(shifted_inputs[first_loss_position])]),
    )
    print(
        "first supervised target token:",
        tokenizer.decode([int(shifted_targets[first_loss_position])]),
    )
    print("end token id:", tokenizer.token_to_id[IM_END])


if __name__ == "__main__":
    main()

