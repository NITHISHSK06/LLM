"""Prepare a deterministic single-turn general conversation dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_INPUT = PROJECT_ROOT / "data/processed/exp008/instruction_train.jsonl"
VAL_INPUT = PROJECT_ROOT / "data/processed/exp008/instruction_val.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/exp009_general_chat"


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load records and return records plus filtering counts."""
    records: list[dict[str, Any]] = []
    stats = {
        "original": 0,
        "single_turn": 0,
        "removed_empty_malformed": 0,
        "removed_multiturn": 0,
        "removed_duplicates": 0,
    }

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            stats["original"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["removed_empty_malformed"] += 1
                continue

            if not isinstance(record, dict):
                stats["removed_empty_malformed"] += 1
                continue
            prompt = record.get("prompt")
            response = record.get("response")
            if not isinstance(prompt, str) or not isinstance(response, str):
                stats["removed_empty_malformed"] += 1
                continue
            if "\n" in prompt or "\r" in prompt:
                stats["removed_multiturn"] += 1
                continue

            stats["single_turn"] += 1
            prompt = prompt.strip()
            response = response.strip()
            if not prompt or not response:
                stats["removed_empty_malformed"] += 1
                continue

            records.append({"prompt": prompt, "response": response})

    return records, stats


def deduplicate_records(
    records: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    stats: dict[str, int],
) -> list[dict[str, Any]]:
    unique_records: list[dict[str, Any]] = []
    for record in records:
        pair = (record["prompt"], record["response"])
        if pair in seen:
            stats["removed_duplicates"] += 1
            continue
        seen.add(pair)
        unique_records.append(record)
    return unique_records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def length_stats(records: list[dict[str, Any]], field: str) -> tuple[int, int, float]:
    lengths = [len(record[field]) for record in records]
    if not lengths:
        return 0, 0, 0.0
    return min(lengths), max(lengths), sum(lengths) / len(lengths)


def print_report(
    train_stats: dict[str, int],
    val_stats: dict[str, int],
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
) -> None:
    all_records = train_records + val_records
    prompt_min, prompt_max, prompt_average = length_stats(all_records, "prompt")
    response_min, response_max, response_average = length_stats(all_records, "response")

    print("Exp009 General Chat Dataset Report")
    print(f"Original train count: {train_stats['original']}")
    print(f"Original validation count: {val_stats['original']}")
    print(f"Single-turn train count: {train_stats['single_turn']}")
    print(f"Single-turn validation count: {val_stats['single_turn']}")
    print(
        "Removed empty/malformed examples: "
        f"{train_stats['removed_empty_malformed'] + val_stats['removed_empty_malformed']} "
        f"(train={train_stats['removed_empty_malformed']}, validation={val_stats['removed_empty_malformed']})"
    )
    print(
        "Removed multi-turn examples: "
        f"{train_stats['removed_multiturn'] + val_stats['removed_multiturn']} "
        f"(train={train_stats['removed_multiturn']}, validation={val_stats['removed_multiturn']})"
    )
    print(
        "Removed duplicate examples: "
        f"{train_stats['removed_duplicates'] + val_stats['removed_duplicates']} "
        f"(train={train_stats['removed_duplicates']}, validation={val_stats['removed_duplicates']})"
    )
    print(f"Final train count: {len(train_records)}")
    print(f"Final validation count: {len(val_records)}")
    print(f"Prompt character length: min={prompt_min}, max={prompt_max}, average={prompt_average:.2f}")
    print(f"Response character length: min={response_min}, max={response_max}, average={response_average:.2f}")


def main() -> None:
    train_records, train_stats = load_jsonl(TRAIN_INPUT)
    val_records, val_stats = load_jsonl(VAL_INPUT)
    seen_pairs: set[tuple[str, str]] = set()
    train_records = deduplicate_records(train_records, seen_pairs, train_stats)
    val_records = deduplicate_records(val_records, seen_pairs, val_stats)

    train_output = OUTPUT_DIR / "instruction_train.jsonl"
    val_output = OUTPUT_DIR / "instruction_val.jsonl"
    write_jsonl(train_output, train_records)
    write_jsonl(val_output, val_records)

    print_report(train_stats, val_stats, train_records, val_records)
    print(f"Saved train dataset: {train_output.relative_to(PROJECT_ROOT)}")
    print(f"Saved validation dataset: {val_output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()