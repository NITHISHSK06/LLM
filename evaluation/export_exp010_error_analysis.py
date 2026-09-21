"""Export Exp006 and Exp010 test generations for qualitative error analysis."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_exp010_general_chat import (
    BASE_CHECKPOINT,
    TEST_DATA,
    TOKENIZER_DIR,
    TUNED_CHECKPOINT,
    generate_response,
    load_model,
    load_test_data,
    relevance,
    repetition_metrics,
)
from tokenizer.tokenizer import SimpleBPETokenizer


OUTPUT_DIR = PROJECT_ROOT / "evaluation/results"
JSONL_PATH = OUTPUT_DIR / "exp010_error_analysis.jsonl"
CSV_PATH = OUTPUT_DIR / "exp010_error_analysis.csv"
MAX_NEW_TOKENS = 60
PROGRESS_INTERVAL = 25

CSV_FIELDS = [
    "index",
    "category",
    "prompt",
    "expected_response",
    "exp006_response",
    "exp010_response",
    "exp006_repetitive",
    "exp010_repetitive",
    "exp006_response_length",
    "exp010_response_length",
    "exp010_likely_relevant",
]


def response_record(
    index: int,
    source: dict[str, Any],
    exp006_response: str,
    exp010_response: str,
) -> dict[str, Any]:
    exp006_repetition = repetition_metrics(exp006_response)["repetition_flag"]
    exp010_repetition = repetition_metrics(exp010_response)["repetition_flag"]
    return {
        "index": index,
        "category": source["category"],
        "prompt": source["prompt"],
        "expected_response": source["response"],
        "exp006_response": exp006_response,
        "exp010_response": exp010_response,
        "exp006_empty": not exp006_response.strip(),
        "exp010_empty": not exp010_response.strip(),
        "exp006_repetitive": exp006_repetition,
        "exp010_repetitive": exp010_repetition,
        "exp006_response_length": len(exp006_response),
        "exp010_response_length": len(exp010_response),
        "exp010_likely_relevant": relevance(exp010_response, str(source["category"])),
    }


def generate_one(
    model: torch.nn.Module,
    tokenizer: SimpleBPETokenizer,
    prompt: str,
    device: torch.device,
) -> str:
    with torch.inference_mode():
        return generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            device=device,
            greedy=True,
        )


def main() -> None:
    print("Exp010 error-analysis export: deterministic greedy generation")
    for required in (TEST_DATA, BASE_CHECKPOINT, TUNED_CHECKPOINT, TOKENIZER_DIR):
        if not required.exists():
            raise FileNotFoundError(f"Required path not found: {required}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_test_data()
    if len(records) != 500:
        raise ValueError(f"Expected 500 test examples, found {len(records)}.")
    tokenizer = SimpleBPETokenizer.load(TOKENIZER_DIR)
    exp006_model = load_model(BASE_CHECKPOINT, tokenizer, device)
    exp010_model = load_model(TUNED_CHECKPOINT, tokenizer, device)

    print("Running 5-example smoke test...")
    for source in records[:5]:
        base_response = generate_one(exp006_model, tokenizer, source["prompt"], device)
        tuned_response = generate_one(exp010_model, tokenizer, source["prompt"], device)
        if not isinstance(base_response, str) or not isinstance(tuned_response, str):
            raise RuntimeError("Smoke test generation did not return strings.")
    print("Smoke test passed: both models generated responses for 5 examples.")

    exported: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, source in enumerate(records, start=1):
        try:
            base_response = generate_one(exp006_model, tokenizer, source["prompt"], device)
            tuned_response = generate_one(exp010_model, tokenizer, source["prompt"], device)
            exported.append(response_record(index, source, base_response, tuned_response))
        except Exception as error:
            errors.append(f"index {index}: {error}")
        if index % PROGRESS_INTERVAL == 0:
            print(f"Processed {index}/{len(records)} examples")

    if errors:
        raise RuntimeError("Generation errors occurred:\n" + "\n".join(errors))
    if len(exported) != len(records):
        raise RuntimeError(f"Exported {len(exported)} records instead of {len(records)}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for item in exported:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in exported:
            writer.writerow({field: item[field] for field in CSV_FIELDS})

    print(f"Exported examples: {len(exported)}")
    print(f"JSONL: {JSONL_PATH}")
    print(f"CSV: {CSV_PATH}")
    print(f"Errors: {len(errors)}")
    print("First 5 exported examples:")
    for item in exported[:5]:
        print(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()