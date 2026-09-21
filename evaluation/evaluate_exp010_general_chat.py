"""Evaluate Exp006 and Exp010 on the held-out Exp010 test set."""

from __future__ import annotations

import json
import platform
import random
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
INFERENCE_DIR = PROJECT_ROOT / "experiments" / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from generate import generate_response
from model.config import ModelConfig
from model.model import DecoderLanguageModel
from tokenizer.tokenizer import SimpleBPETokenizer

SEED = 42
BASE_CHECKPOINT = PROJECT_ROOT / "checkpoints/exp006_mixed_30k/best_model.pt"
TUNED_CHECKPOINT = PROJECT_ROOT / "checkpoints/exp010_general_chat/best_model.pt"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer_exp003"
TEST_DATA = PROJECT_ROOT / "data/processed/exp010_general_chat/instruction_test.jsonl"
REPORT_PATH = PROJECT_ROOT / "evaluation/exp010_evaluation_report.txt"
METRICS_PATH = PROJECT_ROOT / "evaluation/exp010_metrics.json"
HUMAN_PATH = PROJECT_ROOT / "evaluation/exp010_human_eval.jsonl"

RELEVANCE_CUES = {
    "greetings": ("hi", "hello", "hey", "nice to see", "good to see"),
    "introductions": ("meet", "welcome", "name", "glad you are here"),
    "wellbeing": ("how", "well", "okay", "listen", "doing"),
    "small_talk": ("sounds", "day", "weather", "conversation", "enjoy"),
    "thanks": ("welcome", "pleasure", "glad", "problem", "anytime"),
    "apologies": ("sorry", "apolog", "mistake", "forgive", "make it right"),
    "asking_for_help": ("help", "sure", "of course", "what do you need", "assist"),
    "offering_help": ("help", "hand", "useful", "happy", "available"),
    "positive_emotions": ("great", "wonderful", "happy", "glad", "lovely"),
    "negative_emotions": ("sorry", "difficult", "support", "listen", "hard"),
    "encouragement": ("believe", "keep", "progress", "step", "rooting"),
    "agreement": ("agree", "right", "same", "makes sense", "view"),
    "disagreement": ("different", "disagree", "another", "though", "consider"),
    "acknowledgement": ("got it", "understood", "noted", "thanks", "see"),
    "farewell": ("goodbye", "bye", "later", "take care", "see you"),
    "good_morning": ("morning", "day", "slept", "start"),
    "good_night": ("night", "sleep", "rest", "tomorrow", "peaceful"),
    "casual_questions": ("what", "enjoy", "depends", "favorite", "tell you"),
    "daily_activities": ("usually", "schedule", "routine", "habit", "most days"),
    "conversational_followup": ("what", "how", "more", "next", "tell"),
}


def normalize(text: str) -> str:
    for old, new in {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text.casefold()).strip()


def load_test_data() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with TEST_DATA.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(record.get("prompt"), str) or not isinstance(record.get("response"), str):
                    raise ValueError("Malformed test record.")
                records.append(record)
    if not records:
        raise ValueError("Exp010 test set is empty.")
    return records


def load_model(checkpoint_path: Path, tokenizer: SimpleBPETokenizer, device: torch.device) -> DecoderLanguageModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_data = checkpoint.get("model_config")
    if not isinstance(config_data, dict):
        raise ValueError(f"Checkpoint lacks model_config: {checkpoint_path}")
    model = DecoderLanguageModel(ModelConfig(**config_data)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def repetition_metrics(text: str) -> dict[str, Any]:
    words = re.findall(r"[\w']+", normalize(text))
    bigrams = list(zip(words, words[1:]))
    repeated_words = len(words) - len(set(words))
    repeated_bigrams = len(bigrams) - len(set(bigrams))
    short_phrase_flag = any(words[index:index + 3] == words[index + 3:index + 6] for index in range(max(0, len(words) - 5)))
    unigram_ratio = repeated_words / len(words) if words else 0.0
    bigram_ratio = repeated_bigrams / len(bigrams) if bigrams else 0.0
    return {
        "repeated_unigram_ratio": unigram_ratio,
        "repeated_bigram_ratio": bigram_ratio,
        "repetition_flag": bool(short_phrase_flag or unigram_ratio > 0.35 or bigram_ratio > 0.30),
    }


def relevance(text: str, category: str) -> bool:
    normalized = normalize(text)
    return any(cue in normalized for cue in RELEVANCE_CUES.get(category, ()))


def generate_outputs(
    model: DecoderLanguageModel,
    tokenizer: SimpleBPETokenizer,
    records: list[dict[str, Any]],
    device: torch.device,
    temperature: float,
    top_k: int,
    top_p: float,
) -> list[str]:
    outputs: list[str] = []
    for record in records:
        outputs.append(generate_response(
            model, tokenizer, record["prompt"], 60, temperature, top_k, top_p, device,
            greedy=temperature == 0.0,
        ))
    return outputs


def metric_summary(records: list[dict[str, Any]], outputs: list[str]) -> dict[str, Any]:
    exact = [normalize(output) == normalize(record["response"]) for record, output in zip(records, outputs)]
    non_empty = [bool(normalize(output)) for output in outputs]
    repetitions = [repetition_metrics(output) for output in outputs]
    lengths_chars = [len(output) for output in outputs]
    lengths_tokens = [len(re.findall(r"\S+", output)) for output in outputs]
    relevance_values = [relevance(output, str(record["category"])) for record, output in zip(records, outputs)]
    return {
        "test_size": len(records),
        "exact_match_rate": mean(exact),
        "non_empty_rate": mean(non_empty),
        "repeated_unigram_ratio": mean(item["repeated_unigram_ratio"] for item in repetitions),
        "repeated_bigram_ratio": mean(item["repeated_bigram_ratio"] for item in repetitions),
        "repetition_rate": mean(item["repetition_flag"] for item in repetitions),
        "average_generated_character_length": mean(lengths_chars),
        "median_generated_character_length": median(lengths_chars),
        "average_generated_token_length": mean(lengths_tokens),
        "minimum_generated_character_length": min(lengths_chars),
        "maximum_generated_character_length": max(lengths_chars),
        "heuristic_relevance_rate": mean(relevance_values),
    }


def category_summary(records: list[dict[str, Any]], outputs: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category in sorted({str(record["category"]) for record in records}):
        selected = [(record, output) for record, output in zip(records, outputs) if record["category"] == category]
        subset_records = [item[0] for item in selected]
        subset_outputs = [item[1] for item in selected]
        summary = metric_summary(subset_records, subset_outputs)
        result[category] = {
            "test_examples": len(selected),
            "non_empty_responses": sum(bool(normalize(output)) for output in subset_outputs),
            "average_response_length": mean(len(output) for output in subset_outputs),
            "exact_match_rate": summary["exact_match_rate"],
            "repetition_rate": summary["repetition_rate"],
            "heuristic_relevance_rate": summary["heuristic_relevance_rate"],
        }
    return result


def comparison(base: dict[str, Any], tuned: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in tuned.items():
        if isinstance(value, (int, float)) and isinstance(base.get(key), (int, float)):
            result[key] = {"exp006": base[key], "exp010": value, "absolute_difference": value - base[key]}
    return result


def write_human_export(records: list[dict[str, Any]], base_outputs: list[str], tuned_outputs: list[str]) -> None:
    with HUMAN_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for record, base, tuned in zip(records, base_outputs, tuned_outputs):
            item = {
                "id": record.get("id"),
                "category": record["category"],
                "prompt": record["prompt"],
                "expected_response": record["response"],
                "base_generated": base,
                "exp010_generated": tuned,
                "base_repetition_flag": repetition_metrics(base)["repetition_flag"],
                "exp010_repetition_flag": repetition_metrics(tuned)["repetition_flag"],
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def representative_rows(records: list[dict[str, Any]], base_outputs: list[str], tuned_outputs: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successful: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for index, (record, base, tuned) in enumerate(zip(records, base_outputs, tuned_outputs)):
        base_rel = relevance(base, str(record["category"]))
        tuned_rel = relevance(tuned, str(record["category"]))
        base_rep = repetition_metrics(base)["repetition_flag"]
        tuned_rep = repetition_metrics(tuned)["repetition_flag"]
        row = {"index": index + 1, "prompt": record["prompt"], "expected": record["response"], "base": base, "exp010": tuned}
        all_rows.append(row)
        if (tuned_rel and not base_rel) or (normalize(tuned) == normalize(record["response"]) and normalize(base) != normalize(record["response"])):
            successful.append(row)
        if not normalize(tuned) or tuned_rep or (not tuned_rel and base_rel):
            failures.append(row)
    for row in all_rows:
        if len(successful) >= 10:
            break
        if row not in successful:
            successful.append(row)
    for row in all_rows:
        if len(failures) >= 10:
            break
        if row not in failures:
            failures.append(row)
    return successful[:10], failures[:10]


def report_text(records: list[dict[str, Any]], base: dict[str, Any], tuned: dict[str, Any], sample_base: dict[str, Any], sample_tuned: dict[str, Any], base_categories: dict[str, Any], tuned_categories: dict[str, Any], unseen_base: dict[str, Any], unseen_tuned: dict[str, Any], successes: list[dict[str, Any]], failures: list[dict[str, Any]], device: torch.device) -> str:
    lines = [
        "Exp010 General Conversation Evaluation Report",
        "===============================================",
        "Experiment: Exp010 general conversation",
        'Research question: Can a small explicitly designed general-chat dataset improve general conversation behavior over DailyDialog-only tuning?',
        "Scientific caution: this is one held-out test run; no statistical significance is claimed.",
        f"Base checkpoint: {BASE_CHECKPOINT}",
        f"Tuned checkpoint: {TUNED_CHECKPOINT}",
        f"Tokenizer: {TOKENIZER_DIR}",
        f"Test set size: {len(records)}",
        "Primary decoding: greedy=True, temperature=0.0, top_k=0, top_p=1.0, max_new_tokens=60",
        "Secondary decoding: temperature=0.7, top_k=40, top_p=0.9, max_new_tokens=60",
        f"Python: {platform.python_version()} | PyTorch: {torch.__version__} | Device: {device}",
        "",
        "PRIMARY METRICS: ALL TEST EXAMPLES",
    ]
    for key in tuned:
        if key in base and isinstance(tuned[key], (int, float)):
            lines.append(f"{key}: Exp006={base[key]:.6f} | Exp010={tuned[key]:.6f} | absolute_difference={tuned[key] - base[key]:+.6f}")
    lines.append("\nSECONDARY SAMPLING METRICS")
    for key in sample_tuned:
        if key in sample_base and isinstance(sample_tuned[key], (int, float)):
            lines.append(f"{key}: Exp006={sample_base[key]:.6f} | Exp010={sample_tuned[key]:.6f} | absolute_difference={sample_tuned[key] - sample_base[key]:+.6f}")
    lines.append("\nPER-CATEGORY GREEDY METRICS: EXP006")
    lines.append(json.dumps(base_categories, indent=2, ensure_ascii=False))
    lines.append("PER-CATEGORY GREEDY METRICS: EXP010")
    lines.append(json.dumps(tuned_categories, indent=2, ensure_ascii=False))
    lines.append("\nGENERALIZATION SUBSET: HELD-OUT WORDING VARIATIONS")
    lines.append(json.dumps({"Exp006": unseen_base, "Exp010": unseen_tuned, "comparison": comparison(unseen_base, unseen_tuned)}, indent=2, ensure_ascii=False))
    lines.append("\nDETERMINISTIC HUMAN-REVIEW SAMPLE (50 examples, seed 42)")
    sample_indices = random.Random(SEED).sample(range(len(records)), min(50, len(records)))
    for number, index in enumerate(sample_indices, start=1):
        lines.extend([f"\n[{number}] {records[index]['category']}", f"Prompt: {records[index]['prompt']}", f"Expected: {records[index]['response']}", f"Exp006: {base_outputs_global[index]}", f"Exp010: {tuned_outputs_global[index]}"])
    lines.append("\nREPRESENTATIVE SUCCESSFUL/IMPROVED EXAMPLES")
    lines.extend(format_rows(successes))
    lines.append("\nREPRESENTATIVE FAILURE EXAMPLES")
    lines.extend(format_rows(failures))
    return "\n".join(lines) + "\n"


def format_rows(rows: list[dict[str, Any]]) -> list[str]:
    output: list[str] = []
    for row in rows:
        output.extend([f"\nExample {row['index']}", f"Prompt: {row['prompt']}", f"Expected: {row['expected']}", f"Exp006: {row['base']}", f"Exp010: {row['exp010']}"])
    if not rows:
        output.append("(No examples met the deterministic selection criterion.)")
    return output


base_outputs_global: list[str] = []
tuned_outputs_global: list[str] = []


def main() -> None:
    global base_outputs_global, tuned_outputs_global
    for required in (BASE_CHECKPOINT, TUNED_CHECKPOINT, TOKENIZER_DIR, TEST_DATA):
        if not required.exists():
            raise FileNotFoundError(f"Required evaluation input is missing: {required}")
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    records = load_test_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleBPETokenizer.load(TOKENIZER_DIR)
    base_model = load_model(BASE_CHECKPOINT, tokenizer, device)
    tuned_model = load_model(TUNED_CHECKPOINT, tokenizer, device)
    base_outputs_global = generate_outputs(base_model, tokenizer, records, device, 0.0, 0, 1.0)
    tuned_outputs_global = generate_outputs(tuned_model, tokenizer, records, device, 0.0, 0, 1.0)
    base_metrics = metric_summary(records, base_outputs_global)
    tuned_metrics = metric_summary(records, tuned_outputs_global)
    base_categories = category_summary(records, base_outputs_global)
    tuned_categories = category_summary(records, tuned_outputs_global)

    torch.manual_seed(SEED)
    sample_base_outputs = generate_outputs(base_model, tokenizer, records, device, 0.7, 40, 0.9)
    torch.manual_seed(SEED)
    sample_tuned_outputs = generate_outputs(tuned_model, tokenizer, records, device, 0.7, 40, 0.9)
    sample_base_metrics = metric_summary(records, sample_base_outputs)
    sample_tuned_metrics = metric_summary(records, sample_tuned_outputs)

    unseen_records = [record for record in records if record.get("generalization") is True]
    unseen_indices = [index for index, record in enumerate(records) if record.get("generalization") is True]
    unseen_base_metrics = metric_summary(unseen_records, [base_outputs_global[index] for index in unseen_indices])
    unseen_tuned_metrics = metric_summary(unseen_records, [tuned_outputs_global[index] for index in unseen_indices])
    write_human_export(records, base_outputs_global, tuned_outputs_global)
    successes, failures = representative_rows(records, base_outputs_global, tuned_outputs_global)

    payload = {
        "experiment": "Exp010 general conversation",
        "seed": SEED,
        "test_set_size": len(records),
        "base_checkpoint": str(BASE_CHECKPOINT),
        "tuned_checkpoint": str(TUNED_CHECKPOINT),
        "tokenizer": str(TOKENIZER_DIR),
        "decoding": {"primary": {"greedy": True, "temperature": 0.0, "top_k": 0, "top_p": 1.0, "max_new_tokens": 60}, "secondary": {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "max_new_tokens": 60}},
        "primary": {"exp006": base_metrics, "exp010": tuned_metrics, "comparison": comparison(base_metrics, tuned_metrics)},
        "secondary_sampling": {"exp006": sample_base_metrics, "exp010": sample_tuned_metrics, "comparison": comparison(sample_base_metrics, sample_tuned_metrics)},
        "per_category": {"exp006": base_categories, "exp010": tuned_categories},
        "generalization_subset": {"size": len(unseen_records), "exp006": unseen_base_metrics, "exp010": unseen_tuned_metrics, "comparison": comparison(unseen_base_metrics, unseen_tuned_metrics)},
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(report_text(records, base_metrics, tuned_metrics, sample_base_metrics, sample_tuned_metrics, base_categories, tuned_categories, unseen_base_metrics, unseen_tuned_metrics, successes, failures, device), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {METRICS_PATH}")
    print(f"Wrote {HUMAN_PATH}")
    print("Evaluation completed without training.")


if __name__ == "__main__":
    main()