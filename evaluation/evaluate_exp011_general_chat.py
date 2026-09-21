"""Compare Exp006, Exp010, and Exp011 on the Exp011 held-out test set."""

from __future__ import annotations

import csv
import json
import random
import sys
import argparse
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_exp010_general_chat import (
    BASE_CHECKPOINT,
    TOKENIZER_DIR,
    generate_outputs,
    load_model,
    load_test_data,
    metric_summary,
    category_summary,
    comparison,
    repetition_metrics,
    relevance,
)
from tokenizer.tokenizer import SimpleBPETokenizer

EXP010_CHECKPOINT = PROJECT_ROOT / "checkpoints/exp010_general_chat/best_model.pt"
EXP011_CHECKPOINT = PROJECT_ROOT / "checkpoints/exp011_general_chat/best_model.pt"
EXP011_TEST_DATA = PROJECT_ROOT / "data/processed/exp011_general_chat/instruction_test.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evaluation/results"
JSONL_PATH = RESULTS_DIR / "exp011_comparison.jsonl"
CSV_PATH = RESULTS_DIR / "exp011_comparison.csv"
METRICS_PATH = RESULTS_DIR / "exp011_metrics.json"
SEED = 42

CSV_FIELDS = [
    "index", "category", "prompt", "expected_response",
    "exp006_response", "exp010_response", "exp011_response",
]


def load_exp011_test_data(test_data: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with test_data.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Exp011 test record is not an object.")
                for field in ("category", "prompt", "response"):
                    if not isinstance(record.get(field), str):
                        raise ValueError(f"Exp011 test record has invalid {field}: {record}")
                records.append(record)
    if len(records) != 1000:
        raise ValueError(f"Expected 1000 Exp011 test examples, found {len(records)}.")
    return records


def generate_records(
    records: list[dict[str, Any]],
    models: dict[str, torch.nn.Module],
    tokenizer: SimpleBPETokenizer,
    device: torch.device,
    temperature: float,
    top_k: int,
    top_p: float,
) -> dict[str, list[str]]:
    outputs: dict[str, list[str]] = {}
    for name, model in models.items():
        # Reset before each model so sampling is reproducible and model-order independent.
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        outputs[name] = generate_outputs(
            model,
            tokenizer,
            records,
            device,
            temperature,
            top_k,
            top_p,
        )
    return outputs


def metric_summary_exp011(records: list[dict[str, Any]], outputs: list[str]) -> dict[str, Any]:
    converted = [dict(record, response=record["response"]) for record in records]
    original = metric_summary(converted, outputs)
    return {
        "exact_match": original["exact_match_rate"],
        "nonempty_rate": original["non_empty_rate"],
        "repeated_unigram_ratio": original["repeated_unigram_ratio"],
        "repeated_bigram_ratio": original["repeated_bigram_ratio"],
        "repetition_rate": original["repetition_rate"],
        "average_chars": original["average_generated_character_length"],
        "median_chars": original["median_generated_character_length"],
        "min_chars": original["minimum_generated_character_length"],
        "max_chars": original["maximum_generated_character_length"],
        "average_tokens": original["average_generated_token_length"],
        "heuristic_relevance": original["heuristic_relevance_rate"],
    }


def category_metrics(records: list[dict[str, Any]], outputs: list[str]) -> dict[str, Any]:
    converted = [dict(record, response=record["response"]) for record in records]
    result = {}
    for category in sorted({record["category"] for record in records}):
        selected = [(record, output) for record, output in zip(converted, outputs) if record["category"] == category]
        result[category] = metric_summary_exp011(
            [item[0] for item in selected], [item[1] for item in selected]
        )
    return result


def write_comparison(records: list[dict[str, Any]], outputs: dict[str, list[str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records):
            item = {
                "index": index + 1,
                "category": record["category"],
                "prompt": record["prompt"],
                "expected_response": record["response"],
                "exp006_response": outputs["exp006"][index],
                "exp010_response": outputs["exp010"][index],
                "exp011_response": outputs["exp011"][index],
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow({
                "index": index + 1,
                "category": record["category"],
                "prompt": record["prompt"],
                "expected_response": record["response"],
                "exp006_response": outputs["exp006"][index],
                "exp010_response": outputs["exp010"][index],
                "exp011_response": outputs["exp011"][index],
            })


def generalization_subset(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    if any("generalization" in record for record in records):
        subset = [record for record in records if record.get("generalization") is True]
        if subset:
            return subset, "Explicitly marked independent subset in test data."
    return [], "No independent generalization marker exists in the Exp011 test records; no subset was inferred from text."


def build_metrics(
    records: list[dict[str, Any]],
    greedy_outputs: dict[str, list[str]],
    sample_outputs: dict[str, list[str]],
    exp006_checkpoint: Path,
    exp010_checkpoint: Path,
    exp011_checkpoint: Path,
    tokenizer_dir: Path,
) -> dict[str, Any]:
    greedy_metrics = {name: metric_summary_exp011(records, values) for name, values in greedy_outputs.items()}
    sample_metrics = {name: metric_summary_exp011(records, values) for name, values in sample_outputs.items()}
    category = {
        "exp006": category_metrics(records, greedy_outputs["exp006"]),
        "exp010": category_metrics(records, greedy_outputs["exp010"]),
        "exp011": category_metrics(records, greedy_outputs["exp011"]),
    }
    qualitative = {}
    for category_name in sorted({record["category"] for record in records}):
        index = next(i for i, record in enumerate(records) if record["category"] == category_name)
        qualitative[category_name] = {
            "prompt": records[index]["prompt"],
            "expected_response": records[index]["response"],
            "exp006_response": greedy_outputs["exp006"][index],
            "exp010_response": greedy_outputs["exp010"][index],
            "exp011_response": greedy_outputs["exp011"][index],
        }
    return {
        "experiment": "Exp011 general conversation comparison",
        "seed": SEED,
        "test_set_size": len(records),
        "checkpoints": {
            "exp006": str(exp006_checkpoint),
            "exp010": str(exp010_checkpoint),
            "exp011": str(exp011_checkpoint),
        },
        "tokenizer": str(tokenizer_dir),
        "primary_decoding": {"greedy": True, "temperature": 0.0, "top_k": 0, "top_p": 1.0, "max_new_tokens": 60},
        "secondary_decoding": {"greedy": False, "temperature": 0.7, "top_k": 40, "top_p": 0.9, "max_new_tokens": 60},
        "primary_greedy": {
            "aggregate": greedy_metrics,
            "exp006_vs_exp010": comparison(greedy_metrics["exp006"], greedy_metrics["exp010"]),
            "exp006_vs_exp011": comparison(greedy_metrics["exp006"], greedy_metrics["exp011"]),
            "exp010_vs_exp011": comparison(greedy_metrics["exp010"], greedy_metrics["exp011"]),
        },
        "secondary_sampling": {"aggregate": sample_metrics},
        "per_category_greedy": category,
        "qualitative_representative_by_category": qualitative,
        "generalization_subset": {
            "available": False,
            "size": 0,
            "reason": "No independent generalization marker exists in the Exp011 test records; no subset was inferred from text.",
        },
    }


def print_summary(metrics: dict[str, Any]) -> None:
    print("\nPRIMARY GREEDY AGGREGATE METRICS")
    for model_name, values in metrics["primary_greedy"]["aggregate"].items():
        print(f"\n{model_name}:")
        for key, value in values.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.6f}")
    print("\nPER-CATEGORY RELEVANCE, REPETITION, AND AVERAGE LENGTH")
    for category in sorted(metrics["per_category_greedy"]["exp011"]):
        print(f"{category}:")
        for model_name in ("exp006", "exp010", "exp011"):
            values = metrics["per_category_greedy"][model_name][category]
            print(
                f"  {model_name}: relevance={values['heuristic_relevance']:.4f}, "
                f"repetition={values['repetition_rate']:.4f}, "
                f"avg_length={values['average_chars']:.2f}"
            )
    print("\nOUTPUT FILES")
    print(JSONL_PATH)
    print(CSV_PATH)
    print(METRICS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Exp006, Exp010, and Exp011 on the Exp011 held-out test set.")
    parser.add_argument("--exp006-checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--exp010-checkpoint", type=Path, default=EXP010_CHECKPOINT)
    parser.add_argument("--exp011-checkpoint", type=Path, default=EXP011_CHECKPOINT)
    parser.add_argument("--test-data", type=Path, default=EXP011_TEST_DATA)
    parser.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DIR)
    args = parser.parse_args()

    required = (
        args.exp006_checkpoint,
        args.exp010_checkpoint,
        args.exp011_checkpoint,
        args.tokenizer_dir,
        args.test_data,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required evaluation path not found: {path}")

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    records = load_exp011_test_data(args.test_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exp006 checkpoint: {args.exp006_checkpoint.resolve()}")
    print(f"Exp010 checkpoint: {args.exp010_checkpoint.resolve()}")
    print(f"Exp011 checkpoint: {args.exp011_checkpoint.resolve()}")
    tokenizer = SimpleBPETokenizer.load(args.tokenizer_dir)
    models = {
        "exp006": load_model(args.exp006_checkpoint, tokenizer, device),
        "exp010": load_model(args.exp010_checkpoint, tokenizer, device),
        "exp011": load_model(args.exp011_checkpoint, tokenizer, device),
    }
    print("Exp011 comparison evaluation; no training will be performed.")
    print(f"Device: {device}; test examples: {len(records)}")

    print("Running 5-example smoke test for all three models...")
    smoke_outputs = generate_records(records[:5], models, tokenizer, device, 0.0, 0, 1.0)
    if any(len(values) != 5 or not all(isinstance(value, str) for value in values) for values in smoke_outputs.values()):
        raise RuntimeError("Smoke test did not produce five responses per model.")
    print("Smoke test passed.")

    print("Running primary greedy evaluation...")
    greedy_outputs: dict[str, list[str]] = {name: [] for name in models}
    for start in range(0, len(records), 50):
        batch_outputs = generate_records(records[start:start + 50], models, tokenizer, device, 0.0, 0, 1.0)
        for name in models:
            greedy_outputs[name].extend(batch_outputs[name])
        print(f"Processed greedy examples: {min(start + 50, len(records))}/{len(records)}")
    if any(len(values) != 1000 for values in greedy_outputs.values()):
        raise RuntimeError("Full greedy evaluation did not produce exactly 1,000 outputs per model.")

    print("Running secondary sampling evaluation...")
    sample_outputs: dict[str, list[str]] = {name: [] for name in models}
    for start in range(0, len(records), 50):
        batch_outputs = generate_records(records[start:start + 50], models, tokenizer, device, 0.7, 40, 0.9)
        for name in models:
            sample_outputs[name].extend(batch_outputs[name])
        print(f"Processed sampled examples: {min(start + 50, len(records))}/{len(records)}")
    if any(len(values) != 1000 for values in sample_outputs.values()):
        raise RuntimeError("Full sampling evaluation did not produce exactly 1,000 outputs per model.")

    write_comparison(records, greedy_outputs)
    metrics = build_metrics(
        records,
        greedy_outputs,
        sample_outputs,
        args.exp006_checkpoint,
        args.exp010_checkpoint,
        args.exp011_checkpoint,
        args.tokenizer_dir,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_summary(metrics)
    print("Evaluation completed successfully. No model, tokenizer, training code, or checkpoint was modified.")


if __name__ == "__main__":
    main()