"""Shared local mechanics for MIT 6.S191 Lab 3.

The official lab fine-tunes a much larger pretrained LLM with Hugging Face,
PEFT/LoRA, OpenRouter, and Opik. This module keeps the same conceptual pieces
small enough to run offline: chat templates, tokenization, answer masking,
causal next-token loss, low-rank adapters, generation, and simple style scores.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


LAB3_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = LAB3_DIR / "outputs"

START_OF_TEXT = "<|startoftext|>"
USER_START = "<|im_start|>user\n"
ASSISTANT_START = "<|im_start|>assistant\n"
IM_END = "<|im_end|>\n"

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    START_OF_TEXT,
    USER_START,
    ASSISTANT_START,
    IM_END,
]

TEMPLATE_WITHOUT_ANSWER = (
    f"{START_OF_TEXT}{USER_START}" + "{question}" + f"{IM_END}{ASSISTANT_START}"
)
TEMPLATE_WITH_ANSWER = TEMPLATE_WITHOUT_ANSWER + "{answer}" + IM_END


@dataclass(frozen=True)
class StyleExample:
    instruction: str
    response: str
    response_style: str


@dataclass
class LocalFinetuneResult:
    tokenizer: "TinyChatTokenizer"
    model: "TinyCausalTransformer"
    examples: list[StyleExample]
    base_losses: list[float]
    lora_losses: list[float]
    base_trainable_parameters: int
    lora_trainable_parameters: int
    total_parameters: int


BASE_QUESTIONS = [
    StyleExample(
        instruction="How should I debug a neural network that is not learning?",
        response=(
            "Start with the data, then inspect the loss and gradients. "
            "Use a tiny batch and make sure the model can overfit it."
        ),
        response_style=(
            "With the data, begin. The loss and gradients, inspect them. "
            "Overfit a tiny batch first, the model must."
        ),
    ),
    StyleExample(
        instruction="What is the role of a tokenizer in an LLM?",
        response=(
            "A tokenizer converts text into numerical token IDs and converts "
            "generated token IDs back into readable text."
        ),
        response_style=(
            "Text into token IDs, a tokenizer turns. Back into readable text, "
            "generated IDs it converts."
        ),
    ),
    StyleExample(
        instruction="Why does a chat template matter?",
        response=(
            "A chat template marks who is speaking and where each turn starts "
            "and ends, so the model sees a consistent conversation format."
        ),
        response_style=(
            "Who speaks and where turns end, the template shows. A consistent "
            "conversation format, the model needs."
        ),
    ),
    StyleExample(
        instruction="What does next-token prediction train a model to do?",
        response=(
            "It trains the model to predict the following token from the "
            "previous context, one position at a time."
        ),
        response_style=(
            "From previous context, the following token it predicts. One "
            "position at a time, the training proceeds."
        ),
    ),
    StyleExample(
        instruction="Why use LoRA instead of updating every parameter?",
        response=(
            "LoRA trains small low-rank adapter matrices while the large base "
            "weights stay frozen, which makes fine-tuning cheaper."
        ),
        response_style=(
            "Small low-rank adapters, LoRA trains. Frozen, the large base "
            "weights remain. Cheaper, fine-tuning becomes."
        ),
    ),
    StyleExample(
        instruction="How do I know whether a style-tuned model improved?",
        response=(
            "Compare generated answers with negative and positive controls, "
            "then score them with a clear evaluation rubric."
        ),
        response_style=(
            "Generated answers with controls, compare. With a clear rubric, "
            "score them you must."
        ),
    ),
    StyleExample(
        instruction="Why mask the prompt tokens during supervised fine-tuning?",
        response=(
            "The prompt is context, but the answer is what we want the model to "
            "learn to produce. The loss should focus on answer tokens."
        ),
        response_style=(
            "Context, the prompt is. Produce the answer, the model must learn. "
            "On answer tokens, the loss should focus."
        ),
    ),
    StyleExample(
        instruction="What is a useful first check before trusting LLM results?",
        response=(
            "Run small probes that you can understand by hand, then scale up "
            "only after the mechanics are correct."
        ),
        response_style=(
            "Small probes you understand by hand, run first. Scale up only "
            "after correct, the mechanics are."
        ),
    ),
]

LEPRECHAUN_EXAMPLES = [
    StyleExample(
        instruction="What is a good habit for machine learning experiments?",
        response="Keep notes, fix random seeds, and compare one change at a time.",
        response_style=(
            "Keep notes, fix the seeds, and compare one wee change at a time, "
            "for that keeps the experiment honest."
        ),
    ),
    StyleExample(
        instruction="Why should I inspect generated samples?",
        response="Metrics help, but samples reveal formatting errors and strange failures.",
        response_style=(
            "Metrics help indeed, but the samples show the odd little failures "
            "hiding at the end of the rainbow."
        ),
    ),
]


HELDOUT_YODA_EXAMPLES = [
    StyleExample(
        instruction="What should I check before I trust a fine-tuned chatbot?",
        response=(
            "Check prompts, generated samples, losses, and evaluation scores "
            "before trusting the model."
        ),
        response_style=(
            "Before trust, checks you need. Prompts, samples, losses, and "
            "evals, compare them all."
        ),
    ),
    StyleExample(
        instruction="How can a judge model be useful?",
        response=(
            "A judge model can turn a qualitative style question into a "
            "repeatable score, as long as the rubric is clear."
        ),
        response_style=(
            "A qualitative style question into a repeatable score, a judge "
            "model can turn. Clear, the rubric must be."
        ),
    ),
]


def get_style_examples(style: str = "yoda") -> list[StyleExample]:
    if style == "yoda":
        return list(BASE_QUESTIONS)
    if style == "leprechaun":
        return list(LEPRECHAUN_EXAMPLES)
    raise ValueError(f"Unknown style: {style}")


def get_heldout_examples(style: str = "yoda") -> list[StyleExample]:
    if style == "yoda":
        return list(HELDOUT_YODA_EXAMPLES)
    return []


def format_prompt(question: str) -> str:
    return TEMPLATE_WITHOUT_ANSWER.format(question=question)


def format_example(question: str, answer: str) -> str:
    return TEMPLATE_WITH_ANSWER.format(question=question, answer=answer)


class TinyChatTokenizer:
    """Character tokenizer that keeps chat-template markers as atomic tokens."""

    def __init__(self, vocab: list[str]) -> None:
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id[PAD_TOKEN]
        self.unk_token_id = self.token_to_id[UNK_TOKEN]
        self.special_tokens = SPECIAL_TOKENS

    @classmethod
    def from_texts(cls, texts: list[str]) -> "TinyChatTokenizer":
        characters = sorted({character for text in texts for character in text})
        vocab = list(SPECIAL_TOKENS)
        for character in characters:
            if character not in vocab:
                vocab.append(character)
        return cls(vocab)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        index = 0
        special_by_length = sorted(self.special_tokens, key=len, reverse=True)
        while index < len(text):
            matched = False
            for special_token in special_by_length:
                if text.startswith(special_token, index):
                    token_ids.append(self.token_to_id[special_token])
                    index += len(special_token)
                    matched = True
                    break
            if matched:
                continue
            character = text[index]
            token_ids.append(self.token_to_id.get(character, self.unk_token_id))
            index += 1
        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        tokens: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token[int(token_id)]
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return "".join(tokens)


def build_tokenizer(style: str = "yoda", extra_texts: list[str] | None = None) -> TinyChatTokenizer:
    examples = get_style_examples(style)
    heldout_examples = get_heldout_examples(style)
    texts = []
    for example in examples + heldout_examples:
        texts.append(format_example(example.instruction, example.response))
        texts.append(format_example(example.instruction, example.response_style))
    texts.extend(
        [
            format_prompt("What is the capital of France? Use one word."),
            format_prompt("How should I study transformers?"),
            "Yoda test likelihood probe.",
        ]
    )
    if extra_texts:
        texts.extend(extra_texts)
    return TinyChatTokenizer.from_texts(texts)


def encode_supervised_example(
    tokenizer: TinyChatTokenizer,
    question: str,
    answer: str,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    prompt = format_prompt(question)
    full_text = format_example(question, answer)
    token_ids = tokenizer.encode(full_text)
    prompt_length = len(tokenizer.encode(prompt))
    mask = [False] * len(token_ids)
    for index in range(prompt_length, len(token_ids)):
        mask[index] = True
    return (
        torch.tensor(token_ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.bool),
        full_text,
    )


def pad_sequences(
    sequences: list[torch.Tensor],
    pad_value: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    max_length = max(sequence.numel() for sequence in sequences)
    batch = torch.full((len(sequences), max_length), pad_value, dtype=dtype)
    for index, sequence in enumerate(sequences):
        batch[index, : sequence.numel()] = sequence.to(dtype=dtype)
    return batch


def make_supervised_batch(
    tokenizer: TinyChatTokenizer,
    examples: list[StyleExample],
    use_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids_list: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for example in examples:
        answer = example.response_style if use_style else example.response
        token_ids, mask, _ = encode_supervised_example(
            tokenizer,
            example.instruction,
            answer,
        )
        ids_list.append(token_ids)
        masks.append(mask)
    ids = pad_sequences(ids_list, tokenizer.pad_token_id, torch.long)
    mask = pad_sequences(masks, 0, torch.bool)
    return ids, mask


class LoRALinear(nn.Module):
    """Linear layer with optional low-rank trainable adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 0,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.rank = rank
        self.scaling = alpha / rank if rank > 0 else 0.0
        if rank > 0:
            self.lora_a = nn.Parameter(torch.empty(rank, in_features))
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if self.rank > 0:
            nn.init.normal_(self.lora_a, mean=0.0, std=0.02)
            nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x, self.weight, self.bias)
        if self.rank > 0:
            adapter = F.linear(F.linear(x, self.lora_a), self.lora_b)
            output = output + self.scaling * adapter
        return output

    def set_adapter_trainable(self, trainable: bool) -> None:
        if self.rank == 0:
            return
        self.lora_a.requires_grad = trainable
        self.lora_b.requires_grad = trainable

    def set_base_trainable(self, trainable: bool) -> None:
        self.weight.requires_grad = trainable
        if self.bias is not None:
            self.bias.requires_grad = trainable


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, lora_rank: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = LoRALinear(d_model, d_model, bias=False, rank=lora_rank)
        self.k_proj = LoRALinear(d_model, d_model, bias=False, rank=lora_rank)
        self.v_proj = LoRALinear(d_model, d_model, bias=False, rank=lora_rank)
        self.o_proj = LoRALinear(d_model, d_model, bias=False, rank=lora_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                seq_len,
                self.n_heads,
                self.head_dim,
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        context = attention @ v
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, lora_rank: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, lora_rank)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            LoRALinear(d_model, 4 * d_model, rank=lora_rank),
            nn.GELU(),
            LoRALinear(4 * d_model, d_model, rank=lora_rank),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class TinyCausalTransformer(nn.Module):
    """Small causal LM used to rehearse Lab 3 mechanics offline."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_length: int = 256,
        lora_rank: int = 4,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(d_model, n_heads, lora_rank)
                for _ in range(n_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = LoRALinear(d_model, vocab_size, bias=False, rank=lora_rank)
        self.max_length = max_length

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, -self.max_length :]
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)[None, :, :]
        x = self.blocks(x)
        x = self.ln_final(x)
        return self.lm_head(x)

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"base", "lora"}:
            raise ValueError("stage must be 'base' or 'lora'")
        train_base = stage == "base"
        for parameter in self.parameters():
            parameter.requires_grad = train_base
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.set_base_trainable(train_base)
                module.set_adapter_trainable(not train_base)
        if stage == "lora":
            self.token_embedding.weight.requires_grad = False
            self.position_embedding.weight.requires_grad = False
            for module in self.modules():
                if isinstance(module, nn.LayerNorm):
                    for parameter in module.parameters():
                        parameter.requires_grad = False


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def masked_causal_lm_loss(
    model: TinyCausalTransformer,
    input_ids: torch.Tensor,
    answer_mask: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids[:, :-1])
    targets = input_ids[:, 1:]
    shifted_mask = answer_mask[:, 1:]
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    )
    active = shifted_mask.reshape(-1)
    return losses[active].mean()


def train_for_steps(
    model: TinyCausalTransformer,
    input_ids: torch.Tensor,
    answer_mask: torch.Tensor,
    steps: int,
    learning_rate: float,
    seed: int,
) -> list[float]:
    generator = random.Random(seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        index = generator.randrange(input_ids.size(0))
        batch_ids = input_ids[index : index + 1]
        batch_mask = answer_mask[index : index + 1]
        loss = masked_causal_lm_loss(model, batch_ids, batch_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


@torch.inference_mode()
def generate_answer(
    model: TinyCausalTransformer,
    tokenizer: TinyChatTokenizer,
    question: str,
    max_new_tokens: int = 90,
    temperature: float = 0.8,
) -> str:
    model.eval()
    end_id = tokenizer.token_to_id[IM_END]
    token_ids = tokenizer.encode(format_prompt(question))
    generated: list[int] = []
    for _ in range(max_new_tokens):
        current = torch.tensor([token_ids], dtype=torch.long)
        logits = model(current)[0, -1]
        if temperature <= 0:
            next_id = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        token_ids.append(next_id)
        if next_id == end_id:
            break
        generated.append(next_id)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.inference_mode()
def sequence_loss(
    model: TinyCausalTransformer,
    tokenizer: TinyChatTokenizer,
    question: str,
    answer: str,
) -> float:
    input_ids, answer_mask, _ = encode_supervised_example(tokenizer, question, answer)
    loss = masked_causal_lm_loss(model, input_ids[None, :], answer_mask[None, :])
    return float(loss.detach().cpu())


def yoda_style_score(text: str) -> float:
    """Small offline rubric standing in for the lab's LLM-as-judge API call."""

    stripped = text.strip()
    lower = stripped.lower()
    score = 0.0

    first_clause = stripped.split(",", 1)[0]
    first_clause_words = first_clause.split()
    if "," in stripped and 1 <= len(first_clause_words) <= 7:
        score += 0.25
    if lower.startswith(
        (
            "with ",
            "text ",
            "who ",
            "from ",
            "small ",
            "generated ",
            "context",
            "before ",
            "a qualitative ",
        )
    ):
        score += 0.20
    if " must" in lower or " you need" in lower:
        score += 0.20
    if any(marker in lower for marker in [" turns", " shows", " needs", " proceeds", " becomes", " converts", " remains"]):
        score += 0.15
    if any(marker in lower for marker in ["begin", "inspect", "train", "learn", "compare"]):
        score += 0.15
    if stripped.count(".") >= 2:
        score += 0.10
    if stripped.endswith("."):
        score += 0.10
    if lower.startswith(("a ", "the ", "it ", "lora ", "compare ", "run ", "check ")):
        score -= 0.15
    return max(0.0, min(score, 1.0))


def summarize_scores(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float)
    return float(tensor.mean()), float(tensor.std(unbiased=False))


def run_local_lora_finetuning(
    style: str = "yoda",
    base_steps: int = 300,
    lora_steps: int = 500,
    seed: int = 7,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    lora_rank: int = 4,
    base_learning_rate: float = 2e-3,
    lora_learning_rate: float = 5e-3,
) -> LocalFinetuneResult:
    """Run a tiny base-training plus LoRA-only style-tuning pass."""

    torch.manual_seed(seed)
    random.seed(seed)

    examples = get_style_examples(style)
    tokenizer = build_tokenizer(style)
    model = TinyCausalTransformer(
        vocab_size=len(tokenizer),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_length=256,
        lora_rank=lora_rank,
    )
    total_parameters = count_parameters(model)

    base_ids, base_mask = make_supervised_batch(
        tokenizer,
        examples,
        use_style=False,
    )
    model.set_training_stage("base")
    base_trainable_parameters = count_parameters(model, trainable_only=True)
    base_losses = train_for_steps(
        model,
        base_ids,
        base_mask,
        steps=base_steps,
        learning_rate=base_learning_rate,
        seed=seed,
    )

    style_ids, style_mask = make_supervised_batch(
        tokenizer,
        examples,
        use_style=True,
    )
    model.set_training_stage("lora")
    lora_trainable_parameters = count_parameters(model, trainable_only=True)
    lora_losses = train_for_steps(
        model,
        style_ids,
        style_mask,
        steps=lora_steps,
        learning_rate=lora_learning_rate,
        seed=seed + 1,
    )

    return LocalFinetuneResult(
        tokenizer=tokenizer,
        model=model,
        examples=examples,
        base_losses=base_losses,
        lora_losses=lora_losses,
        base_trainable_parameters=base_trainable_parameters,
        lora_trainable_parameters=lora_trainable_parameters,
        total_parameters=total_parameters,
    )
