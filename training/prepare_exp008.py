"""Prepare the Exp008 mixed instruction-tuning dataset."""

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

EXPERIMENT_NAME = "exp008_mixed_instruction"
RAW_DIR = Path("data/raw/exp008")
OUTPUT_DIR = Path("data/processed/exp008")
SEED = 42

CATEGORY_TARGETS = {
    "general_qa": {"train": 9000, "val": 1000},
    "explanations": {"train": 6750, "val": 750},
    "factual_qa": {"train": 6750, "val": 750},
    "reasoning": {"train": 4500, "val": 500},
    "writing": {"train": 4500, "val": 500},
    "conversation": {"train": 6750, "val": 750},
    "coding": {"train": 4500, "val": 500},
    "instruction_following": {"train": 2250, "val": 250},
}

FILTERING_RULES = [
    "All records must be JSON objects with string prompt and response fields.",
    "Whitespace is normalized to single spaces.",
    "Empty prompts or responses are rejected.",
    "Malformed records are rejected.",
    "Exact duplicate prompt/response pairs are removed.",
    "Obvious URL and HTML spam patterns are removed.",
    "Severely repetitive responses are removed.",
    "Each category is split independently with a fixed seed of 42.",
]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def has_obvious_spam(value: str) -> bool:
    lowered = value.casefold()
    if "http://" in lowered or "https://" in lowered:
        return True
    if "<script" in lowered or "</script" in lowered:
        return True
    if "<html" in lowered or "</html" in lowered:
        return True
    if "click here" in lowered or "buy now" in lowered:
        return True
    if "href=" in lowered:
        return True
    if "\n\n" in lowered and ("http" in lowered or "www." in lowered):
        return True
    return False


def has_severe_repetition(text: str) -> bool:
    compact = normalize_text(text)
    if not compact:
        return True
    words = compact.split()
    if not words:
        return True

    word_counts = Counter(words)
    if len(words) >= 20 and max(word_counts.values()) / len(words) >= 0.5:
        return True

    for size in (2, 3, 4):
        n_grams = [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]
        if len(n_grams) >= size * 3:
            gram_counts = Counter(n_grams)
            if max(gram_counts.values()) >= max(3, len(n_grams) // 3):
                return True

    if re.search(r"(.)\1{8,}", compact):
        return True

    return False


def build_category_manifest() -> dict[str, dict[str, int]]:
    manifest: dict[str, dict[str, int]] = {}
    for category, targets in CATEGORY_TARGETS.items():
        manifest[category] = {"train": targets["train"], "val": targets["val"], "total": targets["train"] + targets["val"]}
    return manifest


def read_category_file(category: str, path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw dataset for {category}: {path}")

    cleaned: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    category_stats = {
        "input_records": 0,
        "invalid_records": 0,
        "empty_or_missing_fields": 0,
        "duplicates_removed": 0,
        "url_or_html_spam_removed": 0,
        "repetitive_response_removed": 0,
        "accepted": 0,
    }

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for original_index, line in enumerate(handle):
            if not line.strip():
                continue
            category_stats["input_records"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                category_stats["invalid_records"] += 1
                continue

            if not isinstance(record, dict):
                category_stats["invalid_records"] += 1
                continue

            prompt = record.get("prompt")
            response = record.get("response")
            if not isinstance(prompt, str) or not isinstance(response, str):
                category_stats["empty_or_missing_fields"] += 1
                continue

            prompt = normalize_text(prompt)
            response = normalize_text(response)
            if not prompt or not response:
                category_stats["empty_or_missing_fields"] += 1
                continue

            if has_obvious_spam(prompt) or has_obvious_spam(response):
                category_stats["url_or_html_spam_removed"] += 1
                continue

            if has_severe_repetition(response):
                category_stats["repetitive_response_removed"] += 1
                continue

            pair = (prompt.casefold(), response.casefold())
            if pair in seen_pairs:
                category_stats["duplicates_removed"] += 1
                continue
            seen_pairs.add(pair)

            cleaned.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "category": category,
                    "source": path.name,
                    "original_index": original_index,
                }
            )
            category_stats["accepted"] += 1

    return cleaned, category_stats


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_stats(records: list[dict[str, Any]], duplicate_count: int, rejected_counts: dict[str, int]) -> dict[str, Any]:
    prompts = [len(record["prompt"]) for record in records]
    responses = [len(record["response"]) for record in records]

    def summarize(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"min": 0, "max": 0, "mean": 0.0}
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    category_distribution = Counter(record["category"] for record in records)
    return {
        "number_of_examples": len(records),
        "prompt_character_statistics": summarize(prompts),
        "response_character_statistics": summarize(responses),
        "category_distribution": dict(sorted(category_distribution.items())),
        "duplicate_counts": {"overall": duplicate_count, "by_category": {k: v.get("duplicate_count", 0) for k, v in {}}},
        "rejected_counts": rejected_counts,
    }


def ensure_supported_files(raw_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    for category in CATEGORY_TARGETS:
        candidates = [
            raw_dir / f"{category}.jsonl",
            raw_dir / f"{category}.JSONL",
            raw_dir / f"{category}.json",
            raw_dir / f"{category}.txt",
        ]
        matched = next((path for path in candidates if path.exists()), None)
        if matched is None:
            raise FileNotFoundError(f"Missing raw JSONL for category '{category}' in {raw_dir}")
        files[category] = matched
    return files


def split_category_examples(examples: list[dict[str, Any]], train_count: int, val_count: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(examples) < train_count + val_count:
        raise ValueError(
            f"Not enough examples in category after filtering: need {train_count + val_count}, found {len(examples)}."
        )

    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)
    return shuffled[:train_count], shuffled[train_count : train_count + val_count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Exp008 mixed instruction-tuning dataset.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    raw_dir = args.raw_dir
    output_dir = args.output_dir
    seed = args.seed

    category_files = ensure_supported_files(raw_dir)
    category_manifest = build_category_manifest()
    all_cleaned: list[dict[str, Any]] = []
    per_category_rejected: dict[str, int] = {}
    per_category_seen: dict[str, int] = {}
    total_before = 0

    for category, path in category_files.items():
        cleaned, stats = read_category_file(category, path)
        all_cleaned.extend(cleaned)
        per_category_seen[category] = len(cleaned)
        total_before += stats["input_records"]
        per_category_rejected[category] = (
            stats["invalid_records"]
            + stats["empty_or_missing_fields"]
            + stats["duplicates_removed"]
            + stats["url_or_html_spam_removed"]
            + stats["repetitive_response_removed"]
        )

        required_total = category_manifest[category]["total"]
        if len(cleaned) < required_total:
            raise ValueError(
                f"Category '{category}' has only {len(cleaned)} examples after filtering, but needs at least {required_total}."
            )

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    category_totals: dict[str, dict[str, int]] = {category: {"train": 0, "val": 0, "total": 0} for category in CATEGORY_TARGETS}

    for category in CATEGORY_TARGETS:
        path = category_files[category]
        cleaned, _ = read_category_file(category, path)
        train_count = CATEGORY_TARGETS[category]["train"]
        val_count = CATEGORY_TARGETS[category]["val"]
        cat_train, cat_val = split_category_examples(cleaned, train_count, val_count, seed)
        train_records.extend(cat_train)
        val_records.extend(cat_val)
        category_totals[category] = {
            "train": len(cat_train),
            "val": len(cat_val),
            "total": len(cat_train) + len(cat_val),
        }

    if len(train_records) != 45_000 or len(val_records) != 5_000:
        raise ValueError(
            f"Dataset split size mismatch: train={len(train_records)}, validation={len(val_records)}; expected 45000 and 5000."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "instruction_train.jsonl"
    val_path = output_dir / "instruction_val.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)

    manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "seed": seed,
        "total_examples_before_filtering": total_before,
        "total_examples_after_filtering": len(all_cleaned),
        "train_examples": len(train_records),
        "validation_examples": len(val_records),
        "per_category_counts": category_totals,
        "per_category_rejected_counts": per_category_rejected,
        "source_filenames": {category: path.name for category, path in category_files.items()},
        "filtering_rules": FILTERING_RULES,
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    duplicate_count = sum(
        1 for record in all_cleaned if any(
            other["prompt"].casefold() == record["prompt"].casefold() and other["response"].casefold() == record["response"].casefold()
            for other in all_cleaned
        )
    )
    duplicate_count = duplicate_count // 2

    stats = {
        "number_of_examples": len(all_cleaned),
        "prompt_character_statistics": {
            "min": min(len(record["prompt"]) for record in all_cleaned) if all_cleaned else 0,
            "max": max(len(record["prompt"]) for record in all_cleaned) if all_cleaned else 0,
            "mean": sum(len(record["prompt"]) for record in all_cleaned) / len(all_cleaned) if all_cleaned else 0.0,
        },
        "response_character_statistics": {
            "min": min(len(record["response"]) for record in all_cleaned) if all_cleaned else 0,
            "max": max(len(record["response"]) for record in all_cleaned) if all_cleaned else 0,
            "mean": sum(len(record["response"]) for record in all_cleaned) / len(all_cleaned) if all_cleaned else 0.0,
        },
        "category_distribution": dict(sorted(Counter(record["category"] for record in all_cleaned).items())),
        "duplicate_counts": {"overall": duplicate_count, "by_category": {category: 0 for category in CATEGORY_TARGETS}},
        "rejected_counts": per_category_rejected,
    }
    (output_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Exp008 dataset preparation complete.")
    print(f"Train examples: {len(train_records)}")
    print(f"Validation examples: {len(val_records)}")
    print(f"Output directory: {output_dir}")
    print(f"Manifest: {output_dir / 'dataset_manifest.json'}")
    print(f"Stats: {output_dir / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
