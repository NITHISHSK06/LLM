"""Validate the manually authored coding instruction dataset."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT = Path("data/raw/coding_instruction.jsonl")
DEFAULT_OUTPUT = Path("evaluation/coding_data_quality_report.json")


def inspect(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Coding dataset not found: {path}")
    records: list[dict[str, Any]] = []
    invalid = 0
    duplicate_prompts = 0
    duplicate_pairs = 0
    seen_prompts: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    categories: Counter[str] = Counter()
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    issues: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
            prompt, response = record.get("prompt"), record.get("response")
            if not isinstance(prompt, str) or not isinstance(response, str):
                raise ValueError("missing prompt or response")
            prompt, response = prompt.strip(), response.strip()
            if not prompt or not response:
                raise ValueError("empty prompt or response")
        except (json.JSONDecodeError, ValueError) as error:
            invalid += 1
            issues.append({"line": line_number, "issue": str(error)})
            continue
        prompt_words, response_words = len(prompt.split()), len(response.split())
        pair = (prompt.casefold(), response.casefold())
        if prompt.casefold() in seen_prompts:
            duplicate_prompts += 1
        if pair in seen_pairs:
            duplicate_pairs += 1
        seen_prompts.add(prompt.casefold())
        seen_pairs.add(pair)
        category = record.get("category", "uncategorized")
        categories[str(category)] += 1
        prompt_lengths.append(prompt_words)
        response_lengths.append(response_words)
        flags = {
            "very_short_response": response_words < 3,
            "extremely_long_response": response_words > 1_000,
            "excessive_whitespace": bool(re.search(r"\s{4,}", prompt + response)),
            "invalid_unicode": "�" in prompt + response or any(unicodedata.category(c) == "Cs" for c in prompt + response),
            "repeated_characters": bool(re.search(r"(.)\1{5,}", prompt + response)),
            "repeated_sentences": len(re.findall(r"[^.!?]+", response)) != len(set(part.strip().casefold() for part in re.findall(r"[^.!?]+", response))),
            "malformed_code_block": response.count("```") % 2 != 0,
        }
        if any(flags.values()):
            issues.append({"line": line_number, "issues": [name for name, value in flags.items() if value]})
        records.append({"line": line_number, "prompt_words": prompt_words, "response_words": response_words, **flags})
    count = len(records)
    return {
        "path": str(path),
        "total_examples": count + invalid,
        "valid_examples": count,
        "invalid_examples": invalid,
        "duplicate_prompts": duplicate_prompts,
        "duplicate_prompt_response_pairs": duplicate_pairs,
        "average_prompt_length_words": sum(prompt_lengths) / count if count else 0.0,
        "average_response_length_words": sum(response_lengths) / count if count else 0.0,
        "prompt_length_words": {"minimum": min(prompt_lengths, default=0), "maximum": max(prompt_lengths, default=0)},
        "response_length_words": {"minimum": min(response_lengths, default=0), "maximum": max(response_lengths, default=0)},
        "category_distribution": dict(categories),
        "issues": issues,
        "notes": ["Code-block checks are structural heuristics and do not prove syntax correctness.", "No external model or API was used."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit coding instruction data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = inspect(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
