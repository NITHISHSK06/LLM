"""Analyze completed Exp011 comparison responses without loading models."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "evaluation/results/exp011_comparison.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "evaluation/results"
JSON_PATH = OUTPUT_DIR / "exp011_qualitative_analysis.json"
REPORT_PATH = OUTPUT_DIR / "exp011_qualitative_report.txt"
EXPECTED_COUNT = 1000
MAX_REPRESENTATIVES = 3

MODEL_FIELDS = {
    "exp006": "exp006_response",
    "exp010": "exp010_response",
    "exp011": "exp011_response",
}

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

STOP_WORDS = {
    "a", "an", "and", "are", "be", "but", "by", "for", "from", "have", "i",
    "if", "in", "is", "it", "me", "my", "of", "on", "or", "that", "the", "to",
    "we", "with", "you", "your",
}


def normalize(text: str) -> str:
    text = text.casefold().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.casefold())


def repetition_metrics(text: str) -> dict[str, Any]:
    tokens = words(text)
    bigrams = list(zip(tokens, tokens[1:]))
    repeated_words = len(tokens) - len(set(tokens))
    repeated_bigrams = len(bigrams) - len(set(bigrams))
    short_phrase = any(
        tokens[index:index + 3] == tokens[index + 3:index + 6]
        for index in range(max(0, len(tokens) - 5))
    )
    unigram_ratio = repeated_words / len(tokens) if tokens else 0.0
    bigram_ratio = repeated_bigrams / len(bigrams) if bigrams else 0.0
    return {
        "repeated_unigram_ratio": unigram_ratio,
        "repeated_bigram_ratio": bigram_ratio,
        "repetition_flag": bool(short_phrase or unigram_ratio > 0.35 or bigram_ratio > 0.30),
    }


def relevance_details(record: dict[str, Any], response: str) -> dict[str, Any]:
    normalized = normalize(response)
    category = str(record["category"])
    cues = [cue for cue in RELEVANCE_CUES.get(category, ()) if cue in normalized]
    expected_tokens = {token for token in words(record["expected_response"]) if token not in STOP_WORDS}
    response_tokens = set(words(response))
    overlap = len(expected_tokens & response_tokens) / len(expected_tokens) if expected_tokens else 0.0
    if len(cues) >= 2 or (cues and overlap >= 0.08):
        classification = "clearly_relevant"
    elif cues or overlap >= 0.08:
        classification = "partially_relevant"
    else:
        classification = "irrelevant"
    return {
        "classification": classification,
        "heuristic_relevance": bool(cues),
        "matched_cues": cues,
        "expected_token_overlap": overlap,
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    required = {"index", "category", "prompt", "expected_response", *MODEL_FIELDS.values()}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not required.issubset(record):
                raise ValueError(f"Malformed comparison record on line {line_number}.")
            records.append(record)
    if len(records) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} comparison records, found {len(records)}.")
    return records


def example_view(record: dict[str, Any], model: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": record["index"],
        "category": record["category"],
        "prompt": record["prompt"],
        "expected_response": record["expected_response"],
        "response": record[MODEL_FIELDS[model]],
        "classification": details.get("classification"),
        "heuristic_relevance": details.get("heuristic_relevance"),
        "repetition": details.get("repetition", False),
        "response_length": len(record[MODEL_FIELDS[model]]),
    }


def representative_examples(
    records: list[dict[str, Any]],
    analyses: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model in MODEL_FIELDS:
        result[model] = {}
        model_records = list(zip(records, analyses[model]))
        for label in ("clearly_relevant", "partially_relevant", "irrelevant"):
            selected = [(record, item) for record, item in model_records if item["classification"] == label]
            selected.sort(key=lambda pair: (pair[1]["heuristic_relevance"], pair[1]["expected_token_overlap"]), reverse=label != "irrelevant")
            result[model][label] = [example_view(record, model, item) for record, item in selected[:MAX_REPRESENTATIVES]]
        repetitive = [pair for pair in model_records if pair[1]["repetition_metrics"]["repetition_flag"]]
        repetitive.sort(key=lambda pair: pair[1]["repetition_metrics"]["repeated_bigram_ratio"], reverse=True)
        result[model]["repetitive"] = [example_view(record, model, {"repetition": True}) for record, _ in repetitive[:MAX_REPRESENTATIVES]]
        shortest = sorted(model_records, key=lambda pair: len(pair[0][MODEL_FIELDS[model]]))
        result[model]["overly_short"] = [example_view(record, model, {}) for record, _ in shortest[:MAX_REPRESENTATIVES]]
        longest = sorted(model_records, key=lambda pair: len(pair[0][MODEL_FIELDS[model]]), reverse=True)
        result[model]["unusually_long"] = [example_view(record, model, {}) for record, _ in longest[:MAX_REPRESENTATIVES]]
    return result


def duplicate_summary(records: list[dict[str, Any]], model: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        response = record[MODEL_FIELDS[model]]
        groups[normalize(response)].append({"index": record["index"], "category": record["category"]})
    duplicate_groups = {text: items for text, items in groups.items() if text and len(items) > 1}
    repeated = sorted(duplicate_groups.items(), key=lambda pair: (-len(pair[1]), pair[0]))
    return {
        "unique_normalized_responses": len(groups),
        "normalized_duplicate_response_count": len(duplicate_groups),
        "examples_with_a_duplicate_response": sum(len(items) for items in duplicate_groups.values()),
        "maximum_duplicate_frequency": max((len(items) for items in groups.values()), default=0),
        "most_frequent_repeated_normalized_responses": [
            {"normalized_response": text, "count": len(items), "examples": items[:20]}
            for text, items in repeated[:10]
        ],
    }


def category_statistics(records: list[dict[str, Any]], analyses: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    categories = sorted({str(record["category"]) for record in records})
    for category in categories:
        indices = [index for index, record in enumerate(records) if record["category"] == category]
        result[category] = {"number_of_examples": len(indices), "models": {}}
        for model in MODEL_FIELDS:
            values = [analyses[model][index] for index in indices]
            result[category]["models"][model] = {
                "average_response_length": mean(value["response_length"] for value in values),
                "repetition_rate": mean(value["repetition_metrics"]["repetition_flag"] for value in values),
                "heuristic_relevance": mean(value["heuristic_relevance"] for value in values),
            }
    return result


def category_representatives(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        category = str(record["category"])
        result.setdefault(category, {
            "index": record["index"],
            "prompt": record["prompt"],
            "expected_response": record["expected_response"],
            "exp006_response": record["exp006_response"],
            "exp010_response": record["exp010_response"],
            "exp011_response": record["exp011_response"],
        })
    return result


def analyze(records: list[dict[str, Any]], input_path: Path) -> dict[str, Any]:
    analyses: dict[str, list[dict[str, Any]]] = {}
    for model, field in MODEL_FIELDS.items():
        analyses[model] = []
        for record in records:
            response = record[field]
            repetition = repetition_metrics(response)
            details = relevance_details(record, response)
            details.update({
                "response_length": len(response),
                "repetition_metrics": repetition,
            })
            analyses[model].append(details)
    return {
        "input": str(input_path),
        "example_count": len(records),
        "methodology": {
            "relevance": "Clearly relevant requires at least two category cues or one cue plus expected-response token overlap; partially relevant requires one cue or overlap; otherwise irrelevant.",
            "repetition": "Matches the existing evaluator: repeated 3-word phrase, repeated unigram ratio above 0.35, or repeated bigram ratio above 0.30.",
            "overly_short": "The three shortest responses per model.",
            "unusually_long": "The three longest responses per model.",
            "duplicate_normalization": "Casefolded alphanumeric text with punctuation and whitespace normalized.",
        },
        "representative_examples": representative_examples(records, analyses),
        "category_representatives": category_representatives(records),
        "duplicate_summary": {model: duplicate_summary(records, model) for model in MODEL_FIELDS},
        "template_repeated_response_behavior": {
            model: {
                "duplicate_group_count": duplicate_summary(records, model)["normalized_duplicate_response_count"],
                "duplicate_example_count": duplicate_summary(records, model)["examples_with_a_duplicate_response"],
                "dominant_response_frequency": duplicate_summary(records, model)["maximum_duplicate_frequency"],
                "possible_template_behavior": duplicate_summary(records, model)["maximum_duplicate_frequency"] >= 5,
            }
            for model in MODEL_FIELDS
        },
        "category_statistics": category_statistics(records, analyses),
    }


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "Exp011 qualitative comparison analysis",
        f"Examples analyzed: {result['example_count']}",
        "",
        "Duplicate and possible template behavior",
    ]
    for model, summary in result["duplicate_summary"].items():
        lines.append(
            f"{model}: {summary['normalized_duplicate_response_count']} duplicate groups, "
            f"{summary['examples_with_a_duplicate_response']} examples in duplicate groups, "
            f"most frequent count={summary['maximum_duplicate_frequency']}"
        )
    lines.extend(["", "Category statistics"])
    for category, category_data in result["category_statistics"].items():
        lines.append(f"{category} ({category_data['number_of_examples']} examples)")
        for model, values in category_data["models"].items():
            lines.append(
                f"  {model}: avg_length={values['average_response_length']:.2f}, "
                f"repetition_rate={values['repetition_rate']:.3f}, "
                f"heuristic_relevance={values['heuristic_relevance']:.3f}"
            )
    lines.extend(["", "Representative examples are available in the JSON artifact."])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze completed Exp011 comparison results.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    args = parser.parse_args()
    records = load_records(args.input)
    result = analyze(records, args.input)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(result)
    print(f"Analyzed {len(records)} examples from {args.input}")
    for model, summary in result["duplicate_summary"].items():
        print(
            f"{model}: {summary['normalized_duplicate_response_count']} normalized duplicate groups; "
            f"most frequent response count={summary['maximum_duplicate_frequency']}"
        )
    print(f"JSON: {JSON_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()