"""Diagnose instruction encoding, labels, and a forward pass for Exp009."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import ModelConfig
from model.model import DecoderLanguageModel
from tokenizer.tokenizer import SimpleBPETokenizer


CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/exp009_general_chat/best_model.pt"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer_exp003"
DATASET_PATH = PROJECT_ROOT / "data/processed/exp009_general_chat/instruction_train.jsonl"
USER_PREFIX = "User: "
ASSISTANT_PREFIX = "\nAssistant: "


@dataclass
class EncodedExample:
    input_ids: list[int]
    target_ids: list[int]
    prefix_length: int
    response_ids: list[int]


def load_examples(path: Path, count: int = 5) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Dataset record is not an object.")
            prompt = record.get("prompt")
            response = record.get("response")
            if not isinstance(prompt, str) or not isinstance(response, str):
                raise ValueError("Dataset prompt and response must be strings.")
            examples.append({"prompt": prompt, "response": response})
            if len(examples) == count:
                break
    if len(examples) < count:
        raise ValueError(f"Expected at least {count} training examples, found {len(examples)}.")
    return examples


def encode_example(
    example: dict[str, str],
    tokenizer: SimpleBPETokenizer,
    context_length: int,
) -> EncodedExample:
    prefix = f"{USER_PREFIX}{example['prompt']}{ASSISTANT_PREFIX}"
    prefix_ids = tokenizer.encode(prefix, add_bos=True)
    response_ids = tokenizer.encode(example["response"], add_eos=True)
    all_ids = prefix_ids + response_ids
    response_mask = [False] * len(prefix_ids) + [True] * len(response_ids)

    max_sequence_length = context_length + 1
    if len(all_ids) > max_sequence_length:
        all_ids = all_ids[-max_sequence_length:]
        response_mask = response_mask[-max_sequence_length:]

    input_ids = all_ids[:-1]
    target_ids = [
        token_id if is_response else -100
        for token_id, is_response in zip(all_ids[1:], response_mask[1:])
    ]
    if not any(target_id != -100 for target_id in target_ids):
        raise ValueError("The response has no target tokens after truncation.")
    return EncodedExample(input_ids, target_ids, len(prefix_ids), response_ids)


def decode_ids(tokenizer: SimpleBPETokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids)


def make_batch(
    encoded_examples: list[EncodedExample],
    tokenizer: SimpleBPETokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_length = max(len(example.input_ids) for example in encoded_examples)
    input_ids = torch.full(
        (len(encoded_examples), sequence_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )
    target_ids = torch.full(
        (len(encoded_examples), sequence_length),
        -100,
        dtype=torch.long,
    )
    for index, example in enumerate(encoded_examples):
        length = len(example.input_ids)
        input_ids[index, :length] = torch.tensor(example.input_ids)
        target_ids[index, :length] = torch.tensor(example.target_ids)
    return input_ids.to(device), target_ids.to(device)


def print_example(
    index: int,
    example: dict[str, str],
    encoded: EncodedExample,
    tokenizer: SimpleBPETokenizer,
) -> None:
    valid_target_positions = [
        position for position, token_id in enumerate(encoded.target_ids) if token_id != -100
    ]
    masked_positions = [
        position for position, token_id in enumerate(encoded.target_ids) if token_id == -100
    ]
    decoded_target_ids = [
        token_id for token_id in encoded.target_ids if token_id != -100
    ]

    print(f"\n{'=' * 80}\nEXAMPLE {index}\n{'=' * 80}")
    print("Original prompt:")
    print(example["prompt"])
    print("\nOriginal response:")
    print(example["response"])
    print("\nExact formatted training sequence:")
    print(f"User: {example['prompt']}")
    print(f"Assistant: {example['response']}")
    print("\nEncoded input token IDs:")
    print(encoded.input_ids)
    print("Decoded encoded input:")
    print(repr(decode_ids(tokenizer, encoded.input_ids)))
    print("\nTarget token IDs (-100 means masked):")
    print(encoded.target_ids)
    print(f"Masked target positions (-100): {masked_positions}")
    print(f"Valid target positions: {valid_target_positions}")
    print("Decoded target tokens (unmasked only):")
    print(repr(decode_ids(tokenizer, decoded_target_ids)))
    print("Target token details:")
    for position, token_id in enumerate(encoded.target_ids):
        if token_id == -100:
            print(f"  position {position:3}: -100 [MASKED]")
        else:
            print(f"  position {position:3}: {token_id:5} {repr(decode_ids(tokenizer, [token_id]))}")

    response_start = max(0, encoded.prefix_length - 1)
    prompt_labels_masked = all(
        token_id == -100 for token_id in encoded.target_ids[:response_start]
    )
    response_labels_unmasked = all(
        token_id != -100 for token_id in encoded.target_ids[response_start:]
    )
    expected_eos = encoded.response_ids[-1] == tokenizer.eos_token_id
    eos_is_target = tokenizer.eos_token_id in encoded.target_ids[response_start:]
    print("\nVerification:")
    print(f"  Prompt target positions masked: {prompt_labels_masked}")
    print(f"  Response target positions unmasked: {response_labels_unmasked}")
    print(f"  EOS added by pipeline: {expected_eos}")
    print(f"  EOS present in response target: {eos_is_target}")
    if not (prompt_labels_masked and response_labels_unmasked and expected_eos and eos_is_target):
        raise AssertionError("Instruction encoding verification failed.")


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not TOKENIZER_DIR.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {TOKENIZER_DIR}")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    print("Loading Exp009 checkpoint and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    checkpoint_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Checkpoint does not contain a model_config dictionary.")
    config = ModelConfig(**checkpoint_config)
    tokenizer = SimpleBPETokenizer.load(TOKENIZER_DIR)
    model = DecoderLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Device: {device}")
    print(f"Context length: {config.context_length}")
    print(f"Tokenizer vocabulary size: {len(tokenizer.vocab)}")

    examples = load_examples(DATASET_PATH)
    encoded_examples = [encode_example(example, tokenizer, config.context_length) for example in examples]
    for index, (example, encoded) in enumerate(zip(examples, encoded_examples), start=1):
        print_example(index, example, encoded, tokenizer)

    input_ids, target_ids = make_batch(encoded_examples, tokenizer, device)
    with torch.inference_mode():
        _, loss = model(input_ids, target_ids)
    if loss is None:
        raise RuntimeError("Model returned no loss for the diagnostic targets.")
    valid_target_tokens = int((target_ids != -100).sum().item())
    print(f"\n{'=' * 80}\nFORWARD PASS\n{'=' * 80}")
    print(f"Input shape: {tuple(input_ids.shape)}")
    print(f"Target shape: {tuple(target_ids.shape)}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Number of valid target tokens: {valid_target_tokens}")
    print("Diagnostic completed successfully. No training was performed.")


if __name__ == "__main__":
    main()