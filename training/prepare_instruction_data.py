"""Clean and split instruction JSONL without overwriting the raw source."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT = Path("data/raw/instruction.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def infer_category(prompt: str) -> str | None:
    """Only add metadata when the category is reasonably obvious."""
    lowered = prompt.casefold()
    if any(word in lowered for word in ("summarize", "summary")):
        return "summarization"
    if any(word in lowered for word in ("write", "poem", "story", "paragraph")):
        return "creative_writing"
    if any(word in lowered for word in ("explain", "what is", "why is")):
        return "explanation"
    if any(word in lowered for word in ("hello", "how are you", "help me")):
        return "conversation"
    return None


def load_clean_examples(path: Path, context_length: int, min_response_words: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Instruction dataset not found: {path}")
    stats = Counter()
    seen: set[tuple[str, str]] = set()
    examples: list[dict[str, Any]] = []
    max_characters = context_length * 8
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            stats["empty_lines"] += 1
            continue
        stats["input_lines"] += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            stats["invalid_examples"] += 1
            continue
        if not isinstance(record, dict) or not isinstance(record.get("prompt"), str) or not isinstance(record.get("response"), str):
            stats["invalid_examples"] += 1
            continue
        prompt = normalize_text(record["prompt"])
        response = normalize_text(record["response"])
        if not prompt or not response:
            stats["empty_examples"] += 1
            continue
        if len(response.split()) < min_response_words:
            stats["short_responses"] += 1
            continue
        key = (prompt.casefold(), response.casefold())
        if key in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(key)
        if len(prompt) + len(response) > max_characters:
            stats["long_examples_removed"] += 1
            continue
        cleaned: dict[str, Any] = {"prompt": prompt, "response": response}
        if isinstance(record.get("category"), str) and record["category"].strip():
            cleaned["category"] = record["category"].strip()
        else:
            inferred = infer_category(prompt)
            if inferred is not None:
                cleaned["category"] = inferred
        examples.append(cleaned)
    stats["cleaned_examples"] = len(examples)
    return examples, dict(stats)


def split_examples(examples: list[dict[str, Any]], validation_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(examples) < 2:
        raise ValueError("At least two cleaned examples are required to split the dataset.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    import random
    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = min(max(1, int(len(shuffled) * validation_fraction)), len(shuffled) - 1)
    return shuffled[:-validation_count], shuffled[-validation_count:]


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(example, ensure_ascii=False) + "\n" for example in examples),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean instruction train/validation JSONL files.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--min-response-words", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    examples, stats = load_clean_examples(args.input, args.context_length, args.min_response_words)
    train_examples, validation_examples = split_examples(examples, args.validation_fraction, args.seed)
    train_path = args.output_dir / "instruction_train.jsonl"
    validation_path = args.output_dir / "instruction_val.jsonl"
    write_jsonl(train_path, train_examples)
    write_jsonl(validation_path, validation_examples)
    categories = Counter(example["category"] for example in examples if "category" in example)
    print(json.dumps({"stats": stats, "category_distribution": dict(categories)}, indent=2))
    print(f"Saved training data: {train_path}")
    print(f"Saved validation data: {validation_path}")


if __name__ == "__main__":
    main()
