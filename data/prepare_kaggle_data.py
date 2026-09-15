"""Convert local Kaggle exports into clean training files.

This module only reads and transforms data; it never trains or downloads a model.
Run from the project root, for example:
    python data/prepare_kaggle_data.py --input-dir data/kaggle --category dailydialog
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_INPUT_DIR = Path("data/kaggle")
DEFAULT_INSTRUCTION_OUTPUT = Path("data/raw/instruction.jsonl")
DEFAULT_GENERAL_OUTPUT = Path("data/raw/general.txt")
TEXT_KEYS = ("text", "content", "article", "document", "body", "sentence")
PROMPT_KEYS = ("prompt", "question", "instruction", "input", "source")
RESPONSE_KEYS = ("response", "answer", "output", "target", "corrected")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\x00", " ")).strip()


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".text"}:
        yield {"text": path.read_text(encoding="utf-8", errors="replace")}
        return
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            yield from csv.DictReader(handle)
        return
    if suffix in {".json", ".jsonl"}:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".jsonl":
            values = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            values = json.loads(raw)
        if isinstance(values, dict):
            values = values.get("data", values.get("records", [values]))
        if not isinstance(values, list):
            raise ValueError(f"Expected a JSON list or JSONL records in {path}")
        for value in values:
            if isinstance(value, dict):
                yield value
        return
    raise ValueError(f"Unsupported dataset file type: {path}")


def values(record: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            result.append(normalize_text(value))
    return result


def process_qa(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    examples: list[dict[str, str]] = []
    stats = {"records": 0, "invalid": 0}
    for record in records:
        stats["records"] += 1
        prompts, responses = values(record, PROMPT_KEYS), values(record, RESPONSE_KEYS)
        if not prompts or not responses:
            stats["invalid"] += 1
            continue
        examples.append({"prompt": prompts[0], "response": responses[0], "category": "qa"})
    return examples, stats


def process_grammar(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    examples: list[dict[str, str]] = []
    stats = {"records": 0, "invalid": 0}
    for record in records:
        stats["records"] += 1
        source = values(record, ("original", "incorrect", "input", "sentence", "source"))
        target = values(record, ("corrected", "correct", "target", "output"))
        if not source or not target:
            stats["invalid"] += 1
            continue
        examples.append({"prompt": f"Correct this sentence: {source[0]}", "response": target[0], "category": "grammar"})
    return examples, stats


def process_dailydialog(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    examples: list[dict[str, str]] = []
    stats = {"records": 0, "invalid": 0}
    for record in records:
        stats["records"] += 1
        turns = record.get("dialogue", record.get("conversation", record.get("utterances")))
        if isinstance(turns, str):
            turns = re.split(r"\s*__eou__\s*|\s*\n\s*", turns)
        if not isinstance(turns, list):
            stats["invalid"] += 1
            continue
        clean_turns = [normalize_text(turn) for turn in turns if isinstance(turn, str) and normalize_text(turn)]
        for index in range(1, len(clean_turns)):
            context = "\n".join(clean_turns[max(0, index - 3):index])
            examples.append({"prompt": context, "response": clean_turns[index], "category": "conversation"})
    return examples, stats


def process_wikipedia(records: Iterable[dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    documents: list[str] = []
    stats = {"records": 0, "invalid": 0}
    for record in records:
        stats["records"] += 1
        texts = values(record, TEXT_KEYS)
        if not texts:
            stats["invalid"] += 1
            continue
        documents.extend(texts)
    return documents, stats


def discover_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {input_dir}")
    files = sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".txt", ".text", ".csv", ".json", ".jsonl"})
    if not files:
        raise FileNotFoundError(f"No supported dataset files found under {input_dir}")
    return files


def write_instruction(path: Path, examples: Iterable[dict[str, str]]) -> dict[str, int]:
    seen: set[tuple[str, str]] = set()
    kept, duplicates, empty = 0, 0, 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            prompt, response = normalize_text(example.get("prompt", "")), normalize_text(example.get("response", ""))
            if not prompt or not response:
                empty += 1
                continue
            key = (prompt.casefold(), response.casefold())
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            record = {"prompt": prompt, "response": response}
            if example.get("category"):
                record["category"] = example["category"]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    return {"kept": kept, "duplicates_removed": duplicates, "empty_removed": empty}


def write_general(path: Path, documents: Iterable[str]) -> dict[str, int]:
    seen: set[str] = set()
    kept, duplicates, empty = 0, 0, 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            text = normalize_text(document)
            if not text:
                empty += 1
                continue
            key = text.casefold()
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            handle.write(text + "\n\n")
            kept += 1
    return {"kept": kept, "duplicates_removed": duplicates, "empty_removed": empty}


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean local Kaggle exports into LLM training data.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--category", choices=("dailydialog", "qa", "grammar", "wikipedia"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    files = discover_files(args.input_dir)
    records = (record for path in files for record in iter_records(path))
    if args.category == "dailydialog":
        result, processing = process_dailydialog(records)
        output = args.output or DEFAULT_INSTRUCTION_OUTPUT
        writing = write_instruction(output, result)
    elif args.category == "qa":
        result, processing = process_qa(records)
        output = args.output or DEFAULT_INSTRUCTION_OUTPUT
        writing = write_instruction(output, result)
    elif args.category == "grammar":
        result, processing = process_grammar(records)
        output = args.output or DEFAULT_INSTRUCTION_OUTPUT
        writing = write_instruction(output, result)
    else:
        result, processing = process_wikipedia(records)
        output = args.output or DEFAULT_GENERAL_OUTPUT
        writing = write_general(output, result)
    print(json.dumps({"category": args.category, "files": [str(path) for path in files], "processing": processing, "writing": writing, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
