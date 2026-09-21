"""Analyze semantic and category confusion in completed Exp011 responses."""

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
JSON_PATH = OUTPUT_DIR / "exp011_category_confusion.json"
REPORT_PATH = OUTPUT_DIR / "exp011_category_confusion.txt"
EXPECTED_COUNT = 1000

CATEGORIES = [
    "greetings", "introductions", "wellbeing", "small_talk", "thanks", "apologies",
    "asking_for_help", "offering_help", "positive_emotions", "negative_emotions",
    "encouragement", "agreement", "disagreement", "acknowledgement", "farewell",
    "good_morning", "good_night", "casual_questions", "daily_activities",
    "conversational_followup",
]

MODEL_FIELDS = {"exp010": "exp010_response", "exp011": "exp011_response"}

# These are intentionally specific phrases. Single generic words such as "well"
# or "glad" are too ambiguous to support a reliable confusion-matrix label.
RESPONSE_PATTERNS: dict[str, tuple[str, ...]] = {
    "greetings": ("hello", "hi", "hey", "nice to see you", "good to see you"),
    "introductions": ("nice to meet", "pleasure to meet", "glad to meet", "my name is", "welcome"),
    "wellbeing": ("how are you", "how have you", "how are things", "feel better", "doing well"),
    "small_talk": ("the weather", "sounds interesting", "enjoy the day", "nice conversation", "how was your day"),
    "thanks": ("thank you", "thanks", "you are welcome", "my pleasure", "i appreciate"),
    "apologies": ("i am sorry", "i'm sorry", "my apologies", "i apologize", "forgive me"),
    "asking_for_help": ("how can i help", "what do you need", "i can help", "figure it out together", "assist you"),
    "offering_help": ("offer to help", "happy to help", "lend a hand", "extra hand", "here to help"),
    "positive_emotions": ("i am glad", "i'm glad", "i am happy", "i'm happy", "that is wonderful", "that is lovely"),
    "negative_emotions": ("that sounds difficult", "that sounds hard", "i am sorry to hear", "i'm sorry to hear", "feel supported"),
    "encouragement": ("keep going", "believe in yourself", "you can do it", "next step", "keep making progress"),
    "agreement": ("i agree", "you are right", "you are absolutely right", "makes sense", "i feel the same"),
    "disagreement": ("i disagree", "a different view", "another perspective", "see it differently", "not necessarily"),
    "acknowledgement": ("got it", "i understand", "understood", "noted", "i hear you"),
    "farewell": ("goodbye", "good bye", "see you later", "take care", "talk to you later"),
    "good_morning": ("good morning", "this morning", "start the day", "begin the day"),
    "good_night": ("good night", "sleep well", "have a good rest", "rest well", "tomorrow morning"),
    "casual_questions": ("what do you enjoy", "what is your favorite", "what's your favorite", "tell me about", "what do you like"),
    "daily_activities": ("usually", "daily routine", "my schedule", "most days", "daily activities"),
    "conversational_followup": ("tell me more", "what part", "what happened next", "how about you", "i would like to hear more"),
}

EXPLICIT_RULES = {
    "good_morning_to_good_night": ("good_morning", ("good night", "sleep well", "rest well")),
    "good_night_to_good_morning": ("good_night", ("good morning", "start the day", "begin the day")),
    "greetings_to_farewell": ("greetings", RESPONSE_PATTERNS["farewell"]),
    "introductions_to_wellbeing": ("introductions", RESPONSE_PATTERNS["wellbeing"]),
    "asking_for_help_to_emotional": ("asking_for_help", RESPONSE_PATTERNS["positive_emotions"] + RESPONSE_PATTERNS["negative_emotions"]),
    "thanks_to_unrelated": ("thanks", RESPONSE_PATTERNS["farewell"] + RESPONSE_PATTERNS["apologies"] + RESPONSE_PATTERNS["good_night"]),
    "apologies_to_unrelated": ("apologies", RESPONSE_PATTERNS["farewell"] + RESPONSE_PATTERNS["good_morning"] + RESPONSE_PATTERNS["good_night"]),
}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    required = {"index", "category", "prompt", "expected_response", *MODEL_FIELDS.values(), "exp006_response"}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not required.issubset(record):
                raise ValueError(f"Malformed comparison record on line {line_number}.")
            if record["category"] not in CATEGORIES:
                raise ValueError(f"Unknown category on line {line_number}: {record['category']}")
            records.append(record)
    if len(records) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} comparison records, found {len(records)}.")
    return records


def pattern_scores(response: str) -> dict[str, int]:
    text = normalize(response)
    return {
        category: sum(2 if len(pattern.split()) > 1 else 1 for pattern in patterns if pattern in text)
        for category, patterns in RESPONSE_PATTERNS.items()
    }


def predict_category(response: str) -> tuple[str | None, dict[str, int]]:
    scores = pattern_scores(response)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] < 2:
        return None, scores
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None, scores
    return ranked[0][0], scores


def explicit_confusions(expected: str, response: str) -> list[str]:
    text = normalize(response)
    found: list[str] = []
    for name, (required_category, patterns) in EXPLICIT_RULES.items():
        if expected == required_category and any(pattern in text for pattern in patterns):
            found.append(name)
    return found


def analyze_response(record: dict[str, Any], response: str) -> dict[str, Any]:
    predicted, scores = predict_category(response)
    explicit = explicit_confusions(str(record["category"]), response)
    return {
        "predicted_category": predicted,
        "category_scores": scores,
        "clearly_aligned": predicted == record["category"],
        "clearly_mismatched": predicted is not None and predicted != record["category"],
        "explicit_confusions": explicit,
        "response_length": len(response),
    }


def empty_matrix() -> dict[str, dict[str, int]]:
    return {expected: {predicted: 0 for predicted in CATEGORIES} for expected in CATEGORIES}


def representative(record: dict[str, Any], response: str, analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": record["index"],
        "expected_category": record["category"],
        "prompt": record["prompt"],
        "expected_response": record["expected_response"],
        "response": response,
        "predicted_category": analysis["predicted_category"],
        "explicit_confusions": analysis["explicit_confusions"],
    }


def category_analysis(
    records: list[dict[str, Any]],
    model: str,
    analyses: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, int]], dict[str, list[dict[str, Any]]]]:
    matrix = empty_matrix()
    by_category: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_category[str(record["category"])].append(index)
        predicted = analyses[index]["predicted_category"]
        if predicted is not None:
            matrix[str(record["category"])][predicted] += 1

    summary: dict[str, Any] = {}
    confusion_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for category in CATEGORIES:
        indices = by_category[category]
        incorrect_patterns: Counter[str] = Counter()
        explicit_counts: Counter[str] = Counter()
        for index in indices:
            item = analyses[index]
            predicted = item["predicted_category"]
            if predicted is not None and predicted != category:
                incorrect_patterns[predicted] += 1
                confusion_examples[f"{category}->{predicted}"].append(
                    representative(records[index], records[index][MODEL_FIELDS[model]], item)
                )
            explicit_counts.update(item["explicit_confusions"])
        summary[category] = {
            "total_examples": len(indices),
            "clearly_aligned_responses": sum(analyses[index]["clearly_aligned"] for index in indices),
            "clearly_mismatched_responses": sum(analyses[index]["clearly_mismatched"] for index in indices),
            "unclassified_responses": sum(analyses[index]["predicted_category"] is None for index in indices),
            "obvious_category_confusion_counts": dict(explicit_counts),
            "most_common_incorrect_response_pattern": incorrect_patterns.most_common(1)[0][0] if incorrect_patterns else None,
            "most_common_incorrect_response_pattern_count": incorrect_patterns.most_common(1)[0][1] if incorrect_patterns else 0,
        }
    for key in confusion_examples:
        confusion_examples[key] = confusion_examples[key][:3]
    return summary, matrix, confusion_examples


def generic_patterns(records: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    leading: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        response = record[MODEL_FIELDS[model]]
        normalized = normalize(response)
        exact[normalized].append({"index": record["index"], "category": record["category"]})
        prefix = " ".join(normalized.split()[:6])
        if prefix:
            leading[prefix].append({"index": record["index"], "category": record["category"]})
    patterns = []
    for source, groups in (("exact", exact), ("leading_six_words", leading)):
        for pattern, examples in groups.items():
            categories = sorted({item["category"] for item in examples})
            if len(examples) >= 3 and len(categories) >= 2:
                patterns.append({"type": source, "pattern": pattern, "count": len(examples), "category_count": len(categories), "categories": categories, "examples": examples[:20]})
    return sorted(patterns, key=lambda item: (-item["count"], item["type"], item["pattern"]))[:20]


def compare_models(records: list[dict[str, Any]], analyses_by_model: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category in CATEGORIES:
        indices = [index for index, record in enumerate(records) if record["category"] == category]
        result[category] = {}
        for model, analyses in analyses_by_model.items():
            result[category][model] = {
                "clearly_aligned": sum(analyses[index]["clearly_aligned"] for index in indices),
                "clearly_mismatched": sum(analyses[index]["clearly_mismatched"] for index in indices),
                "explicit_confusion_count": sum(bool(analyses[index]["explicit_confusions"]) for index in indices),
            }
        result[category]["same_predicted_category_count"] = sum(
            analyses_by_model["exp010"][index]["predicted_category"] == analyses_by_model["exp011"][index]["predicted_category"]
            and analyses_by_model["exp010"][index]["predicted_category"] is not None
            for index in indices
        )
    return result


def build_result(records: list[dict[str, Any]], input_path: Path) -> dict[str, Any]:
    analyses_by_model = {
        model: [analyze_response(record, record[field]) for record in records]
        for model, field in MODEL_FIELDS.items()
    }
    model_results: dict[str, Any] = {}
    confusion_examples: dict[str, Any] = {}
    for model, analyses in analyses_by_model.items():
        summary, matrix, examples = category_analysis(records, model, analyses)
        model_results[model] = {
            "confusion_matrix_expected_rows_predicted_columns": matrix,
            "category_summary": summary,
            "generic_response_patterns": generic_patterns(records, model),
        }
        confusion_examples[model] = examples
    return {
        "input": str(input_path),
        "example_count": len(records),
        "categories": CATEGORIES,
        "methodology": {
            "reliable_prediction": "A response is assigned only when one category has a unique score of at least 2 from specific phrase patterns; ambiguous or weakly signaled responses are unclassified.",
            "explicit_confusions": "Contradiction rules cover morning/night, greeting/farewell, introduction/wellbeing, help/emotional, thanks, and apology mismatches.",
            "matrix": "Rows are expected categories and columns are reliable predicted response categories; unclassified responses are omitted from matrix cells.",
        },
        "models": model_results,
        "representative_confusion_examples": confusion_examples,
        "exp010_vs_exp011": compare_models(records, analyses_by_model),
    }


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "Exp011 category-confusion analysis",
        f"Examples analyzed: {result['example_count']}",
        "",
        "Reliable predictions use specific phrase heuristics; ambiguous responses are unclassified.",
    ]
    for model, model_data in result["models"].items():
        lines.extend(["", f"{model} category summary"])
        for category, summary in model_data["category_summary"].items():
            confusion = ", ".join(f"{key}={value}" for key, value in summary["obvious_category_confusion_counts"].items()) or "none"
            lines.append(
                f"{category}: total={summary['total_examples']}, aligned={summary['clearly_aligned_responses']}, "
                f"mismatched={summary['clearly_mismatched_responses']}, obvious={confusion}, "
                f"common_incorrect={summary['most_common_incorrect_response_pattern'] or 'none'}"
            )
        lines.extend(["", f"{model} generic response patterns across categories"])
        for pattern in model_data["generic_response_patterns"][:10]:
            lines.append(f"{pattern['type']}: count={pattern['count']}, categories={', '.join(pattern['categories'])}, pattern={pattern['pattern']}")
    lines.extend(["", "Representative strongest confusion patterns"])
    for model, groups in result["representative_confusion_examples"].items():
        lines.append(model)
        for confusion, examples in sorted(groups.items(), key=lambda item: item[0]):
            lines.append(f"  {confusion}: {len(examples)} representative example(s)")
            for example in examples:
                lines.append(f"    index {example['index']}: {example['response']}")
    lines.extend(["", "Exp010 versus Exp011 comparison is available in the JSON artifact; no overall score or ranking is produced."])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp011 semantic/category confusion.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    args = parser.parse_args()
    records = load_records(args.input)
    result = build_result(records, args.input)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(result)
    print(f"Analyzed {len(records)} examples from {args.input}")
    for model, model_data in result["models"].items():
        summary = model_data["category_summary"]
        aligned = sum(item["clearly_aligned_responses"] for item in summary.values())
        mismatched = sum(item["clearly_mismatched_responses"] for item in summary.values())
        print(f"{model}: clearly aligned={aligned}; clearly mismatched={mismatched}")
    print(f"JSON: {JSON_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()