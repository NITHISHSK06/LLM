"""Diagnostic experiment for memorizing 32 instruction-response examples."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INFERENCE_DIR = PROJECT_ROOT / "experiments" / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from generate import generate_response
from model.model import DecoderLanguageModel
from tokenizer.tokenizer import SimpleBPETokenizer
from training.instruct_train import (
    EncodedExample,
    InstructionExample,
    encode_example,
    make_batch,
)


BASE_CHECKPOINT = PROJECT_ROOT / "checkpoints/exp006_mixed_30k/best_model.pt"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer_exp003"
DATASET_PATH = PROJECT_ROOT / "data/processed/exp009_general_chat/instruction_train.jsonl"
OUTPUT_CHECKPOINT = PROJECT_ROOT / "checkpoints/diagnostic_memorization/final_model.pt"
SEED = 42
EXAMPLE_COUNT = 32
BATCH_SIZE = 8
MAX_STEPS = 2_000
LEARNING_RATE = 2e-5


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_first_single_turn_examples(path: Path) -> list[InstructionExample]:
    examples: list[InstructionExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            prompt = record.get("prompt")
            response = record.get("response")
            if not isinstance(prompt, str) or not isinstance(response, str):
                continue
            if "\n" in prompt or "\r" in prompt:
                continue
            examples.append(InstructionExample(prompt=prompt, response=response))
            if len(examples) == EXAMPLE_COUNT:
                break
    if len(examples) != EXAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXAMPLE_COUNT} single-turn examples, found {len(examples)} "
            f"while reading {path}."
        )
    return examples


def load_base_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DecoderLanguageModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Base checkpoint does not contain model_config.")
    from model.config import ModelConfig

    model = DecoderLanguageModel(ModelConfig(**checkpoint_config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def save_checkpoint(
    path: Path,
    model: DecoderLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "train_loss": train_loss,
            "validation_loss": None,
            "best_validation_loss": None,
            "model_config": asdict(model.config),
            "diagnostic": "32-example instruction memorization experiment",
            "seed": SEED,
        },
        path,
    )


def main() -> None:
    print("DIAGNOSTIC EXPERIMENT: instruction memorization")
    print(f"Training exactly the first {EXAMPLE_COUNT} single-turn examples")
    print(f"Steps: {MAX_STEPS}; batch size: {BATCH_SIZE}; learning rate: {LEARNING_RATE}")
    set_deterministic_seed(SEED)

    for path in (BASE_CHECKPOINT, TOKENIZER_DIR, DATASET_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleBPETokenizer.load(TOKENIZER_DIR)
    examples = load_first_single_turn_examples(DATASET_PATH)
    model, _ = load_base_model(BASE_CHECKPOINT, device)
    model.train()
    config = model.config
    encoded_examples: list[EncodedExample] = [
        encode_example(example, tokenizer, config.context_length)
        for example in examples
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=config.weight_decay)
    rng = random.Random(SEED)
    final_loss = float("nan")

    print(f"Device: {device}")
    print(f"Encoded examples: {len(encoded_examples)}")
    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = make_batch(
            encoded_examples,
            BATCH_SIZE,
            tokenizer.pad_token_id,
            rng,
            device,
        )
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss}")
        final_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step:4d}/{MAX_STEPS}: training loss = {final_loss:.6f}")

    save_checkpoint(OUTPUT_CHECKPOINT, model, optimizer, MAX_STEPS, final_loss)
    print(f"Final training loss: {final_loss:.6f}")
    print(f"Saved diagnostic checkpoint: {OUTPUT_CHECKPOINT.relative_to(PROJECT_ROOT)}")

    model.eval()
    print("\nGREEDY DECODING RESULTS")
    for index, example in enumerate(examples, start=1):
        generated = generate_response(
            model,
            tokenizer,
            example.prompt,
            max_new_tokens=100,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            device=device,
            greedy=True,
        )
        print(f"\nExample {index}")
        print(f"Prompt: {example.prompt}")
        print(f"Expected response: {example.response}")
        print(f"Generated response: {generated}")

    print("\nDiagnostic experiment completed successfully. No normal Exp009 checkpoint or tokenizer was modified.")


if __name__ == "__main__":
    main()