"""Clean and split coding instruction data without touching the raw file."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT = Path("data/raw/coding_instruction.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed")


def clean_record(record: Any, context_length: int) -> dict[str, Any] | None:
    if not isinstance(record, dict) or not isinstance(record.get("prompt"), str) or not isinstance(record.get("response"), str):
        return None
    prompt = re.sub(r"\s+", " ", record["prompt"]).strip()
    response = re.sub(r"\s+", " ", record["response"]).strip()
    if not prompt or not response or len(response.split()) < 3:
        return None
    if len(prompt.split()) + len(response.split()) > context_length * 2:
        return None
    cleaned: dict[str, Any] = {"prompt": prompt, "response": response}
    if isinstance(record.get("category"), str) and record["category"].strip():
        cleaned["category"] = record["category"].strip()
    return cleaned


def prepare(path: Path, context_length: int, validation_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Coding dataset not found: {path}")
    stats = Counter()
    seen: set[tuple[str, str]] = set()
    cleaned: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        stats["input_examples"] += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            stats["invalid_examples"] += 1
            continue
        item = clean_record(record, context_length)
        if item is None:
            stats["rejected_examples"] += 1
            continue
        key = (item["prompt"].casefold(), item["response"].casefold())
        if key in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(key)
        cleaned.append(item)
    if len(cleaned) < 2:
        raise ValueError("At least two clean coding examples are required for a non-leaking split.")
    shuffled = cleaned.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = min(max(1, int(len(shuffled) * validation_fraction)), len(shuffled) - 1)
    stats["cleaned_examples"] = len(cleaned)
    stats["training_examples"] = len(shuffled) - validation_count
    stats["validation_examples"] = validation_count
    return shuffled[:-validation_count], shuffled[-validation_count:], dict(stats)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare coding train and validation JSONL files.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_records, validation_records, result = prepare(args.input, args.context_length, args.validation_fraction, args.seed)
    train_path = args.output_dir / "coding_train.jsonl"
    validation_path = args.output_dir / "coding_val.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(validation_path, validation_records)
    print(json.dumps(result, indent=2))
    print(f"Saved: {train_path}")
    print(f"Saved: {validation_path}")


if __name__ == "__main__":
    main()
