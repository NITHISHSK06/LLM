"""Audit raw corpus and instruction data without changing either file."""

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

DEFAULT_GENERAL = Path("data/raw/general.txt")
DEFAULT_INSTRUCTIONS = Path("data/raw/instruction.jsonl")
DEFAULT_OUTPUT = Path("evaluation/data_quality_report.json")
DEFAULT_RESPONSE_OUTPUT = Path("evaluation/response_quality_report.json")


def approximate_token_count(path: Path, tokenizer_path: Path | None) -> int | None:
    """Count tokens when a project tokenizer is available, without requiring it."""
    if tokenizer_path is None or not path.exists():
        return None
    try:
        from tokenizer.tokenizer import SimpleBPETokenizer
        tokenizer = SimpleBPETokenizer.load(tokenizer_path)
        return len(tokenizer.encode(path.read_text(encoding="utf-8", errors="replace")))
    except (FileNotFoundError, KeyError, ValueError):
        return None


def inspect_general(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "characters": 0,
        "lines": 0,
        "empty_lines": 0,
        "very_short_lines": 0,
        "extremely_long_lines": 0,
        "excessive_whitespace_lines": 0,
        "invalid_unicode_lines": 0,
        "duplicate_lines": 0,
    }
    if not path.exists():
        return report
    text = path.read_text(encoding="utf-8", errors="replace")
    report["characters"] = len(text)
    lines = text.splitlines()
    report["lines"] = len(lines)
    seen: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if stripped and stripped.casefold() in seen:
            report["duplicate_lines"] += 1
        seen.add(stripped.casefold())
        report["empty_lines"] += int(not stripped)
        report["very_short_lines"] += int(0 < len(stripped) < 10)
        report["extremely_long_lines"] += int(len(line) > 20_000)
        report["excessive_whitespace_lines"] += int(bool(re.search(r"\s{4,}", line)))
        report["invalid_unicode_lines"] += int("�" in line or any(
            unicodedata.category(character) == "Cs" for character in line
        ))
    return report


def response_quality(response: str) -> dict[str, Any]:
    words = re.findall(r"[\w']+", response, flags=re.UNICODE)
    sentences = [part.strip() for part in re.split(r"[.!?]+", response) if part.strip()]
    repeated_words = sum(
        words[index].lower() == words[index + 1].lower()
        for index in range(len(words) - 1)
    )
    repeated_sentences = len(sentences) - len({sentence.casefold() for sentence in sentences})
    excessive_punctuation = bool(re.search(r"([!?.,])\1{3,}", response))
    unusual_character_repetition = bool(re.search(r"(.)\1{5,}", response))
    complete_sentence = bool(re.search(r"[.!?]\s*$", response)) if response else False
    return {
        "empty": not bool(response.strip()),
        "word_count": len(words),
        "very_short": 0 < len(words) < 3,
        "extremely_long": len(words) > 1_000,
        "complete_sentence_ending": complete_sentence,
        "repeated_adjacent_words": repeated_words,
        "repeated_sentences": repeated_sentences,
        "excessive_punctuation": excessive_punctuation,
        "unusual_character_repetition": unusual_character_repetition,
    }


def inspect_instructions(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "total_nonempty_lines": 0,
        "valid_examples": 0,
        "broken_json": 0,
        "missing_prompt": 0,
        "missing_response": 0,
        "empty_prompts": 0,
        "empty_responses": 0,
        "duplicate_examples": 0,
        "repeated_responses": 0,
        "very_short_prompts": 0,
        "very_short_responses": 0,
        "extremely_long_examples": 0,
        "excessive_whitespace_examples": 0,
        "invalid_unicode_examples": 0,
        "category_distribution": {},
    }
    quality_records: list[dict[str, Any]] = []
    if not path.exists():
        return report, quality_records

    examples: list[tuple[str, str]] = []
    response_counts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        report["total_nonempty_lines"] += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            report["broken_json"] += 1
            continue
        if not isinstance(record, dict):
            report["broken_json"] += 1
            continue
        prompt = record.get("prompt")
        response = record.get("response")
        if not isinstance(prompt, str):
            report["missing_prompt"] += 1
        if not isinstance(response, str):
            report["missing_response"] += 1
        if not isinstance(prompt, str) or not isinstance(response, str):
            continue
        prompt = prompt.strip()
        response = response.strip()
        report["valid_examples"] += 1
        report["empty_prompts"] += int(not prompt)
        report["empty_responses"] += int(not response)
        report["very_short_prompts"] += int(0 < len(prompt.split()) < 3)
        report["very_short_responses"] += int(0 < len(response.split()) < 3)
        report["extremely_long_examples"] += int(len(prompt) + len(response) > 20_000)
        report["excessive_whitespace_examples"] += int(bool(re.search(r"\s{4,}", prompt + response)))
        report["invalid_unicode_examples"] += int("�" in prompt + response)
        examples.append((prompt, response))
        response_counts[response.casefold()] += 1
        category = record.get("category")
        if isinstance(category, str) and category.strip():
            categories[category.strip()] += 1
        quality = response_quality(response)
        quality["line_number"] = line_number
        quality["text"] = f"{prompt} {response}"
        quality_records.append(quality)

    report["duplicate_examples"] = len(examples) - len(set(examples))
    report["repeated_responses"] = sum(count - 1 for count in response_counts.values() if count > 1)
    report["category_distribution"] = dict(categories)
    return report, quality_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw LLM datasets.")
    parser.add_argument("--general", type=Path, default=DEFAULT_GENERAL)
    parser.add_argument("--instructions", type=Path, default=DEFAULT_INSTRUCTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--response-output", type=Path, default=DEFAULT_RESPONSE_OUTPUT)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    args = parser.parse_args()
    general_report = inspect_general(args.general)
    general_report["approximate_token_count"] = approximate_token_count(args.general, args.tokenizer_dir)
    instruction_report, quality_records = inspect_instructions(args.instructions)
    if args.tokenizer_dir is not None and args.instructions.exists():
        try:
            from tokenizer.tokenizer import SimpleBPETokenizer
            tokenizer = SimpleBPETokenizer.load(args.tokenizer_dir)
            instruction_report["approximate_token_count"] = sum(
                len(tokenizer.encode(item["text"])) for item in quality_records
            )
        except (FileNotFoundError, KeyError, ValueError):
            instruction_report["approximate_token_count"] = None
    quality_summary = {
        "examples_checked": len(quality_records),
        "empty_responses": sum(item["empty"] for item in quality_records),
        "very_short_responses": sum(item["very_short"] for item in quality_records),
        "repeated_adjacent_words": sum(item["repeated_adjacent_words"] > 0 for item in quality_records),
        "repeated_sentences": sum(item["repeated_sentences"] > 0 for item in quality_records),
        "excessive_punctuation": sum(item["excessive_punctuation"] for item in quality_records),
        "unusual_character_repetition": sum(item["unusual_character_repetition"] for item in quality_records),
        "records": quality_records,
    }
    report = {
        "general_text": general_report,
        "instruction_jsonl": instruction_report,
        "response_quality": quality_summary,
        "notes": [
            "Category distribution reports only category metadata already present in the source JSONL.",
            "Rule-based checks are data-quality signals, not a complete grammar evaluator.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.response_output.parent.mkdir(parents=True, exist_ok=True)
    args.response_output.write_text(
        json.dumps(quality_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("DATA QUALITY REPORT")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")
    print(f"Saved: {args.response_output}")


if __name__ == "__main__":
    main()
