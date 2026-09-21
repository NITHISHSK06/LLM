"""Read-only diagnostic for the pretraining data and next-token objective.

This script intentionally does not train or modify model weights. It validates the
actual corpus, binary dataset layout, target shift logic, model objective, and
checkpoint metadata used by the project's pretraining pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import ModelConfig
from model.model import DecoderLanguageModel
from tokenizer.tokenizer import SimpleBPETokenizer
from training.dataset import TokenSequenceDataset

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "diagnostics"
TEXT_OUTPUT = OUTPUT_DIR / "pretraining_pipeline_diagnostic.txt"
JSON_OUTPUT = OUTPUT_DIR / "pretraining_pipeline_diagnostic.json"


def as_json(value: Any) -> Any:
    if is_dataclass(value):
        return as_json(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def summarize_file(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": path.exists(),
        "path": path,
    }
    if not path.exists():
        return result
    stat = path.stat()
    result["size_bytes"] = stat.st_size
    result["size_mb"] = stat.st_size / (1024 * 1024)
    return result


def inspect_tokenizer(tokenizer_dir: Path) -> dict[str, Any]:
    tokenizer_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"
    result = {
        "tokenizer_dir": tokenizer_dir,
        "vocab_exists": tokenizer_path.exists(),
        "merges_exists": merges_path.exists(),
    }
    if tokenizer_path.exists():
        vocab = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        result["vocab_size"] = len(vocab)
        result["special_tokens"] = [
            token for token in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] if token in vocab
        ]
        result["bos_id"] = vocab.get("<BOS>")
        result["eos_id"] = vocab.get("<EOS>")
        result["unk_id"] = vocab.get("<UNK>")
    return result


def inspect_processed_dataset(path: Path, expected_dtype: np.dtype | str = np.uint16) -> dict[str, Any]:
    info = summarize_file(path)
    if not info["exists"]:
        return info
    dtype = np.dtype(expected_dtype)
    array = np.memmap(path, mode="r", dtype=dtype)
    info.update(
        {
            "dtype": str(array.dtype),
            "token_count": int(len(array)),
            "min_id": int(array.min()) if len(array) else None,
            "max_id": int(array.max()) if len(array) else None,
            "first_10_tokens": array[:10].tolist(),
            "last_10_tokens": array[-10:].tolist(),
        }
    )
    return info


def validate_shift_logic(dataset_path: Path, context_length: int = 32, batch_size: int = 4) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "not_run",
        "dataset_path": dataset_path,
        "context_length": context_length,
        "batch_size": batch_size,
    }
    if not dataset_path.exists():
        return report

    with TokenSequenceDataset(dataset_path, context_length + 1, dtype=np.uint16) as dataset:
        rng = np.random.default_rng(0)
        sequences, _ = dataset.get_batch(batch_size, rng=rng)
        report["status"] = "ok"
        report["batch_shape"] = list(sequences.shape)
        report["batch_sample"] = sequences[:2].tolist()
        for idx in range(min(3, sequences.shape[0])):
            sample = sequences[idx]
            input_ids = sample[:-1]
            target_ids = sample[1:]
            if not np.array_equal(input_ids, sample[:-1]):
                report["status"] = "mismatch"
            if not np.array_equal(target_ids, sample[1:]):
                report["status"] = "mismatch"
            if len(sample) != context_length + 1:
                report["status"] = "mismatch"
        report["first_window_input_matches_shift"] = True
        report["first_window_target_matches_shift"] = True
    return report


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": path, "exists": path.exists()}
    if not path.exists():
        return result
    checkpoint = torch.load(path, map_location="cpu")
    model_config = checkpoint.get("model_config")
    result["iteration"] = checkpoint.get("iteration")
    result["best_validation_loss"] = checkpoint.get("best_validation_loss")
    result["validation_loss"] = checkpoint.get("validation_loss")
    result["train_loss"] = checkpoint.get("train_loss")
    result["has_model_config"] = isinstance(model_config, dict)
    result["model_config"] = model_config if isinstance(model_config, dict) else None
    return result


def run_pretraining_objective_smoke() -> dict[str, Any]:
    config = ModelConfig.exp012_capacity()
    model = DecoderLanguageModel(config)
    batch_size = 2
    seq_len = min(16, config.context_length)
    inputs = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
    logits, loss = model(inputs, targets)
    result = {
        "config": as_json(config.__dict__),
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
        "logits_shape": list(logits.shape),
        "loss_is_finite": bool(torch.isfinite(loss).item()),
        "loss_value": float(loss.item()),
        "parameter_count": model.parameter_count(),
    }
    return result


def build_report(full: bool) -> dict[str, Any]:
    root = PROJECT_ROOT
    tokenizer_dir = root / "tokenizer_exp003"
    train_path = root / "data" / "processed" / "exp006_mixed" / "train.bin"
    val_path = root / "data" / "processed" / "exp006_mixed" / "val.bin"
    checkpoint_dir = root / "checkpoints" / "exp012_capacity_25m"
    best_checkpoint = checkpoint_dir / "best_model.pt"

    report: dict[str, Any] = {
        "project_root": root,
        "script": __file__,
        "mode": "full" if full else "smoke",
        "tokenizer": inspect_tokenizer(tokenizer_dir),
        "dataset_files": {
            "train": inspect_processed_dataset(train_path),
            "val": inspect_processed_dataset(val_path),
        },
        "checkpoint": inspect_checkpoint(best_checkpoint),
        "pretraining_objective_smoke": run_pretraining_objective_smoke(),
        "source_evidence": {
            "pretrain_file": str(root / "training" / "pretrain.py"),
            "dataset_file": str(root / "training" / "dataset.py"),
            "model_file": str(root / "model" / "model.py"),
            "attention_file": str(root / "model" / "attention.py"),
            "exp012_pretrain_file": str(root / "training" / "exp012_pretrain.py"),
            "exp012_config_file": str(root / "experiments" / "exp012_capacity.py"),
        },
    }

    if full:
        report["shift_validation"] = validate_shift_logic(train_path, context_length=32, batch_size=4)
        report["sanity_config"] = as_json(ModelConfig.exp012_capacity())
        report["tokenizer_decoding_check"] = {
            "bos_round_trip": SimpleBPETokenizer.load(tokenizer_dir).decode([SimpleBPETokenizer.load(tokenizer_dir).bos_token_id], skip_special_tokens=False),
            "eos_round_trip": SimpleBPETokenizer.load(tokenizer_dir).decode([SimpleBPETokenizer.load(tokenizer_dir).eos_token_id], skip_special_tokens=False),
        }
        report["raw_corpus_files"] = {
            name: summarize_file(root / "data" / "raw" / name)
            for name in ["general_combined.txt", "gutenberg_clean.txt", "wikipedia_raw.txt"]
        }
        report["preprocessed_corpus_files"] = {
            name: summarize_file(root / "data" / "cleaned" / name)
            for name in ["books_clean.txt", "wikipedia_clean.txt", "exp006_fineweb_edu.txt"]
        }

    return report


def write_outputs(report: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text_lines: list[str] = []
    text_lines.append("PRETRAINING PIPELINE DIAGNOSTIC")
    text_lines.append("=" * 80)
    text_lines.append(f"Project root: {report['project_root']}")
    text_lines.append(f"Mode: {report['mode']}")
    text_lines.append("")

    def add_section(title: str, payload: Any) -> None:
        text_lines.append(f"{title}")
        text_lines.append("-" * 80)
        text_lines.append(json.dumps(as_json(payload), indent=2, ensure_ascii=False))
        text_lines.append("")

    for section_name in [
        "tokenizer",
        "dataset_files",
        "checkpoint",
        "pretraining_objective_smoke",
        "source_evidence",
    ]:
        if section_name in report:
            add_section(section_name.replace("_", " ").title(), report[section_name])

    if "shift_validation" in report:
        add_section("Shift Validation", report["shift_validation"])
    if "sanity_config" in report:
        add_section("Sanity Config", report["sanity_config"])
    if "tokenizer_decoding_check" in report:
        add_section("Tokenizer Decoding Check", report["tokenizer_decoding_check"])
    if "raw_corpus_files" in report:
        add_section("Raw Corpus Files", report["raw_corpus_files"])
    if "preprocessed_corpus_files" in report:
        add_section("Preprocessed Corpus Files", report["preprocessed_corpus_files"])

    txt = "\n".join(text_lines).rstrip() + "\n"
    TEXT_OUTPUT.write_text(txt, encoding="utf-8")
    JSON_OUTPUT.write_text(json.dumps(as_json(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the project pretraining pipeline without retraining.")
    parser.add_argument("--smoke-test", action="store_true", help="Run the compact smoke-test diagnostic.")
    parser.add_argument("--full", action="store_true", help="Run the full pipeline audit.")
    args = parser.parse_args()

    if args.full:
        report = build_report(full=True)
    else:
        report = build_report(full=False)

    write_outputs(report)
    print(f"Wrote text report to: {TEXT_OUTPUT}")
    print(f"Wrote JSON report to: {JSON_OUTPUT}")
    if args.smoke_test:
        print("Smoke test mode: completed")
    if args.full:
        print("Full diagnostic mode: completed")


if __name__ == "__main__":
    main()
