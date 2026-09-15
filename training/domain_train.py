"""Fine-tune the existing instruction model on coding data."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import ModelConfig
from model.model import DecoderLanguageModel
from tokenizer.tokenizer import SimpleBPETokenizer
from training.instruct_train import (
    EncodedExample,
    encode_examples,
    evaluate_loss,
    load_split_examples,
    make_batch,
    select_device,
    set_seed,
)

DEFAULT_CODING_TRAIN = Path("data/processed/coding_train.jsonl")
DEFAULT_CODING_VAL = Path("data/processed/coding_val.jsonl")
DEFAULT_GENERAL_TRAIN = Path("data/processed/instruction_train.jsonl")
DEFAULT_GENERAL_VAL = Path("data/processed/instruction_val.jsonl")
DEFAULT_OUTPUT = Path("checkpoints/coding")


def choose_base_checkpoint(requested: Path | None) -> Path:
    if requested is not None:
        return requested
    v2 = Path("checkpoints/instruct_v2/best_model.pt")
    v1 = Path("checkpoints/instruct/best_model.pt")
    if v2.exists():
        return v2
    if v1.exists():
        return v1
    raise FileNotFoundError(
        "No instruction-tuned checkpoint found. Expected checkpoints/instruct_v2/best_model.pt "
        "or checkpoints/instruct/best_model.pt."
    )


def load_model(checkpoint_path: Path, device: torch.device) -> DecoderLanguageModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_data = checkpoint.get("model_config")
    if not isinstance(config_data, dict):
        raise ValueError("Checkpoint does not contain model_config.")
    model = DecoderLanguageModel(ModelConfig(**config_data)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def mixed_batch(
    coding: list[EncodedExample],
    general: list[EncodedExample],
    batch_size: int,
    pad_id: int,
    general_ratio: float,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not coding:
        raise ValueError("Coding training data is empty.")
    selected: list[EncodedExample] = []
    for _ in range(batch_size):
        use_general = bool(general) and rng.random() < general_ratio
        selected.append(rng.choice(general if use_general else coding))
    return make_batch(selected, len(selected), pad_id, rng, device)


@torch.inference_mode()
def validation_loss(
    model: DecoderLanguageModel,
    examples: list[EncodedExample],
    config: ModelConfig,
    tokenizer: SimpleBPETokenizer,
    device: torch.device,
    seed: int,
) -> float:
    return evaluate_loss(model, examples, config, tokenizer, device, random.Random(seed))


def save_checkpoint(path: Path, model: DecoderLanguageModel, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, step: int, train_loss: float, val_loss: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    domain_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "perplexity": math.exp(min(val_loss, 20.0)),
        "model_config": model.config.__dict__,
        "domain_config": domain_config,
    }, path)


def train(args: argparse.Namespace) -> None:
    if not 0.0 <= args.general_data_ratio <= 1.0:
        raise ValueError("general_data_ratio must be between 0 and 1.")
    set_seed(args.seed)
    device = select_device()
    checkpoint_path = choose_base_checkpoint(args.checkpoint)
    model = load_model(checkpoint_path, device)
    tokenizer = SimpleBPETokenizer.load(args.tokenizer_dir)
    coding_train = encode_examples(load_split_examples(args.train_data), tokenizer, model.config.context_length)
    coding_val = encode_examples(load_split_examples(args.val_data), tokenizer, model.config.context_length)
    general_train: list[EncodedExample] = []
    if args.general_data_ratio > 0:
        if args.general_train.exists():
            general_train = encode_examples(load_split_examples(args.general_train), tokenizer, model.config.context_length)
        else:
            print(f"Warning: general data not found, using coding-only batches: {args.general_train}")
            args.general_data_ratio = 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    resume_step = 0
    best_val = math.inf
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scaler_state_dict" in state:
            scaler.load_state_dict(state["scaler_state_dict"])
        resume_step = int(state.get("step", 0))
        best_val = float(state.get("validation_loss", math.inf))
        print(f"Resumed from {args.resume} at step {resume_step}")

    steps_per_epoch = max(1, math.ceil(len(coding_train) / args.batch_size))
    total_steps = args.max_iterations or args.epochs * steps_per_epoch
    rng = random.Random(args.seed)
    model.train()
    best_path = args.output_dir / "best_model.pt"
    last_val = math.inf
    last_train = math.inf
    no_improvement = 0
    for step in range(resume_step + 1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = mixed_batch(coding_train, general_train, args.batch_size, tokenizer.pad_id, args.general_data_ratio, rng, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            _, loss = model(inputs, targets)
        assert loss is not None
        last_train = loss.item()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        if step == 1:
            print(f"Input shape: {tuple(inputs.shape)}")
            print(f"Target shape: {tuple(targets.shape)}")
            print(f"Initial loss: {last_train:.4f}")
            print(f"Parameters: {model.parameter_count():,}")
        if step % args.eval_interval == 0 or step == total_steps:
            last_val = validation_loss(model, coding_val, model.config, tokenizer, device, args.seed + step)
            perplexity = math.exp(min(last_val, 20.0))
            print(f"Step: {step} | Train Loss: {last_train:.4f} | Validation Loss: {last_val:.4f} | Perplexity: {perplexity:.2f}")
            if last_val < best_val:
                best_val = last_val
                no_improvement = 0
                save_checkpoint(best_path, model, optimizer, scaler, step, last_train, last_val, args)
                print(f"Saved best model: {best_path}")
            else:
                no_improvement += 1
                if no_improvement >= args.patience:
                    print(f"Early stopping after {no_improvement} evaluations without improvement.")
                    break
        if step % args.checkpoint_interval == 0 or step == total_steps:
            save_checkpoint(args.output_dir / f"checkpoint_{step:06d}.pt", model, optimizer, scaler, step, last_train, last_val, args)
            save_checkpoint(args.output_dir / "latest.pt", model, optimizer, scaler, step, last_train, last_val, args)

    metadata_path = Path("evaluation/experiments.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    except json.JSONDecodeError:
        metadata = {}
    metadata["phase10_domain_specialization"] = {
        "experiment_name": "coding_specialization",
        "base_checkpoint": str(checkpoint_path),
        "parameter_count": model.parameter_count(),
        "tokenizer_vocabulary_size": len(tokenizer.vocab),
        "coding_training_examples": len(coding_train),
        "coding_validation_examples": len(coding_val),
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "general_data_ratio": args.general_data_ratio,
        "training_loss": last_train,
        "validation_loss": last_val,
        "perplexity": math.exp(min(last_val, 20.0)) if math.isfinite(last_val) else None,
        "output_checkpoint": str(best_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Specialize the instruction model on coding data.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train_data", "--train-data", type=Path, default=DEFAULT_CODING_TRAIN)
    parser.add_argument("--val_data", "--val-data", type=Path, default=DEFAULT_CODING_VAL)
    parser.add_argument("--general_train", type=Path, default=DEFAULT_GENERAL_TRAIN)
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizer"))
    parser.add_argument("--output_dir", "--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_iterations", "--max-iterations", type=int, default=None)
    parser.add_argument("--lr", "--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=4)
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval_interval", "--eval-interval", type=int, default=50)
    parser.add_argument("--checkpoint_interval", "--checkpoint-interval", type=int, default=250)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gradient_clip", "--gradient-clip", type=float, default=1.0)
    parser.add_argument("--general_data_ratio", "--general-data-ratio", type=float, default=0.2)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
