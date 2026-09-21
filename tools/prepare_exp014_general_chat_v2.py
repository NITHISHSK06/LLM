"""Build and validate the research-quality EXP014 v2 conversational dataset.

The primary source is DailyDialog. Exp011 is used only when a row passes the
quality filters because its older synthetic rows are not uniformly natural.
This script writes only to data/processed/exp014_general_chat_v2 and never
loads model, tokenizer, or checkpoint files.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed" / "exp014_general_chat_v2"
SEED = 42
MAX_CONTEXT_CHARS = 1800
MAX_RESPONSE_CHARS = 700
RESPONSE_CAP = 18
ALLOWED_CATEGORIES = (
    "greetings", "introductions", "wellbeing", "small_talk", "thanks", "apologies",
    "asking_for_help", "offering_help", "positive_emotions", "negative_emotions",
    "encouragement", "agreement", "disagreement", "acknowledgement", "farewell",
    "good_morning", "good_night", "casual_questions", "daily_activities",
    "conversational_followup",
)

SEED_EXAMPLES = [
    ("greetings", "hi", "Hey! How are you doing?"),
    ("greetings", "hi", "Hi! What are you up to today?"),
    ("greetings", "hi", "Hey, good to hear from you. How is everything?"),
    ("greetings", "hi", "Hello! How has your day been?"),
    ("greetings", "hi", "Hey there! What is new?"),
    ("greetings", "hey", "Hi there! What is up?"),
    ("greetings", "hey", "Hey! How have you been?"),
    ("greetings", "hey", "Hi! How is your day going?"),
    ("greetings", "hey", "Hello! Nice to hear from you."),
    ("greetings", "hello", "Hello! How is your day going?"),
    ("greetings", "hello", "Hi there! What are you up to?"),
    ("greetings", "hello", "Hey! How are things?"),
    ("greetings", "hello", "Hello! It is good to hear from you."),
    ("greetings", "hi there", "Hey! Good to hear from you."),
    ("greetings", "hey there", "Hi! What have you been up to?"),
    ("greetings", "hello there", "Hello! Nice to hear from you."),
    ("good_morning", "good morning", "Good morning! I hope your day gets off to a good start."),
    ("good_morning", "morning", "Morning! How is the day looking for you?"),
    ("greetings", "good afternoon", "Good afternoon! How has your day been so far?"),
    ("good_night", "good night", "Good night. Rest well and take care."),
    ("wellbeing", "how are you?", "I am doing pretty well, thanks. How about you?"),
    ("wellbeing", "how's it going?", "It is going well so far. What about you?"),
    ("wellbeing", "how have you been?", "I have been keeping busy, but things are going okay."),
    ("greetings", "nice to see you", "Nice to see you too! What is new?"),
    ("introductions", "nice to meet you", "Nice to meet you too. What brings you here?"),
    ("thanks", "thanks for helping me", "You are welcome! I am glad I could help."),
    ("thanks", "thank you for listening", "Of course. I am glad you felt comfortable sharing that."),
    ("apologies", "I am sorry I was late", "That is okay. Thanks for letting me know."),
    ("asking_for_help", "Can you help me with this?", "Sure! Tell me what you need help with."),
    ("asking_for_help", "I am stuck on this problem", "Let us look at it together. Which part is causing trouble?"),
    ("offering_help", "Do you need a hand with those boxes?", "That would be helpful, thank you."),
    ("positive_emotions", "I had a really good day", "That is great to hear. What made it a good day?"),
    ("negative_emotions", "I had a really bad day", "I am sorry to hear that. Do you want to talk about what happened?"),
    ("encouragement", "I am nervous about tomorrow", "That makes sense. Take it one step at a time; you can handle it."),
    ("agreement", "I think that is a good idea", "I agree. It sounds like a practical way forward."),
    ("disagreement", "I do not think that plan will work", "I see your concern. What change would make it more workable?"),
    ("acknowledgement", "Okay, I understand", "Thanks for confirming. We are on the same page."),
    ("farewell", "I have to go now", "No problem. Take care, and talk to you later."),
    ("casual_questions", "What are you doing today?", "I am just here chatting with you. What about you?"),
    ("daily_activities", "What do you usually do after work?", "I usually unwind for a while, then make dinner or take a walk."),
    ("small_talk", "The weather is nice today", "It really is. It makes being outside much more appealing."),
    ("conversational_followup", "User: I started learning guitar.\nAssistant: That sounds fun!\nUser: It is harder than I expected.", "The first steps can be frustrating. What part has been hardest so far?"),
    ("conversational_followup", "User: I went to a new cafe.\nAssistant: Did you enjoy it?\nUser: Yes, the pastries were excellent.", "That sounds worth returning to. Which pastry did you like best?"),
]


def normalize(value: str) -> str:
    value = value.casefold().replace("’", "'")
    value = re.sub(r"\s+", " ", value.strip())
    value = re.sub(r"\s+([,.!?;:])", r"\1", value)
    value = re.sub(r"([,.!?;:])\s*", r"\1 ", value)
    return value.strip()


def words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", normalize(value))


def token_count(value: str) -> int:
    return len(words(value))


def looks_broken(prompt: str, response: str) -> bool:
    text = f"{prompt} {response}"
    if not prompt.strip() or not response.strip():
        return True
    if len(prompt) > MAX_CONTEXT_CHARS or len(response) > MAX_RESPONSE_CHARS:
        return True
    if re.search(r"\[(?:variant|sample|example)[^\]]*\]", text, re.I):
        return True
    if re.search(r"\b(?:example|sample)[_-]?\d{2,}\b", text, re.I):
        return True
    if "when the day has been full" in normalize(text) or "when i need a change of pace" in normalize(text):
        return True
    if re.search(r"(.)\1{5,}", text):
        return True
    if text.count("\n") > 18:
        return True
    return False


def classify(prompt: str, response: str) -> str | None:
    text = normalize(f"{prompt} {response}")
    prompt_norm = normalize(prompt)
    if prompt_norm in {"hi", "hey", "hello", "hi there", "hey there", "hello there", "nice to see you"}:
        return "greetings"
    if re.search(r"\b(good morning|morning)\b", prompt_norm):
        return "good_morning"
    if re.search(r"\b(good night|goodnight)\b", prompt_norm):
        return "good_night"
    rules = [
        ("introductions", r"\b(my name is|i am called|nice to meet|introduce myself)\b"),
        ("thanks", r"\b(thank|thanks|appreciate)\b"),
        ("apologies", r"\b(sorry|apolog|forgive me)\b"),
        ("asking_for_help", r"\b(can you help|could you help|need help|help me|stuck on)\b"),
        ("offering_help", r"\b(can i help|shall i help|need a hand|want me to help|let me help)\b"),
        ("negative_emotions", r"\b(bad day|sad|upset|stressed|overwhelmed|worried|lonely|frustrated|angry)\b"),
        ("positive_emotions", r"\b(great day|good day|happy|excited|glad|wonderful|fantastic)\b"),
        ("encouragement", r"\b(nervous|discouraged|believe in me|encourage|can i do|worried about)\b"),
        ("farewell", r"\b(goodbye|bye|see you|have to go|take care)\b"),
        ("acknowledgement", r"\b(got it|understood|i understand|okay,? i see|makes sense)\b"),
        ("disagreement", r"\b(do not agree|don't agree|disagree|not sure|not right|different view)\b"),
        ("agreement", r"\b(i agree|good idea|you are right|you're right|makes sense)\b"),
        ("wellbeing", r"\b(how are you|how's it going|how have you been|feeling all right|doing today)\b"),
        ("daily_activities", r"\b(usually do|what have you been doing|day look like|after work|daily routine)\b"),
        ("casual_questions", r"\b(what do you like|favorite|what are you doing|what is your)\b"),
    ]
    for category, pattern in rules:
        if re.search(pattern, text):
            return category
    if "?" in prompt:
        return "small_talk"
    if "\n" in prompt:
        return "conversational_followup"
    return "small_talk"


def to_labeled_context(raw_prompt: str) -> str:
    turns = [part.strip() for part in raw_prompt.splitlines() if part.strip()]
    if not turns:
        return ""
    labels = []
    for index, turn in enumerate(turns):
        labels.append(f"{'User' if index % 2 == 0 else 'Assistant'}: {turn}")
    return "\n".join(labels)


def load_jsonl(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_examples() -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    source_files: list[str] = []
    daily_path = ROOT / "data" / "raw" / "dailydialog_train.jsonl"
    source_files.append(str(daily_path.relative_to(ROOT)))
    for item in load_jsonl(daily_path):
        prompt = to_labeled_context(str(item.get("prompt", "")))
        response = str(item.get("response", "")).strip()
        category = classify(prompt, response)
        if category and not looks_broken(prompt, response):
            rows.append({"category": category, "prompt": prompt, "response": response, "source": "dailydialog_train"})

    for split in ("train", "val"):
        path = ROOT / "data" / "processed" / "exp011_general_chat" / f"instruction_{split}.jsonl"
        source_files.append(str(path.relative_to(ROOT)))
        for item in load_jsonl(path):
            prompt = str(item.get("prompt", "")).strip()
            response = str(item.get("response", "")).strip()
            category = str(item.get("category", ""))
            if category not in ALLOWED_CATEGORIES or looks_broken(prompt, response):
                continue
            rows.append({"category": category, "prompt": prompt, "response": response, "source": f"exp011_{split}"})
    return rows, source_files


def add_seed_examples(rows: list[dict[str, str]]) -> None:
    for category, prompt, response in SEED_EXAMPLES:
        rows.append({"category": category, "prompt": prompt, "response": response, "source": "curated_seed"})


def near_duplicate(prompt: str, response: str, buckets: dict[tuple[str, ...], list[tuple[str, str]]]) -> bool:
    current_words = words(f"{prompt} {response}")
    key = tuple(current_words[:6])
    for old_prompt, old_response in buckets[key]:
        old_words = words(f"{old_prompt} {old_response}")
        if SequenceMatcher(None, current_words, old_words).ratio() >= 0.94:
            return True
    buckets[key].append((prompt, response))
    return False


def deduplicate(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    unique: list[dict[str, str]] = []
    prompts: set[str] = set()
    pairs: set[str] = set()
    responses: Counter[str] = Counter()
    buckets: dict[tuple[str, ...], list[tuple[str, str]]] = defaultdict(list)
    stats = Counter(raw_examples=len(rows))
    greeting_prompts = {"hi", "hey", "hello"}
    for row in rows:
        prompt_norm = normalize(row["prompt"])
        response_norm = normalize(row["response"])
        pair_norm = f"{prompt_norm}\n{response_norm}"
        if prompt_norm in prompts and prompt_norm not in greeting_prompts:
            stats["duplicate_prompts_removed"] += 1
            continue
        if pair_norm in pairs:
            stats["duplicate_pairs_removed"] += 1
            continue
        if responses[response_norm] >= RESPONSE_CAP:
            stats["response_cap_removed"] += 1
            continue
        if prompt_norm not in greeting_prompts and near_duplicate(row["prompt"], row["response"], buckets):
            stats["near_duplicate_removed"] += 1
            continue
        if prompt_norm in greeting_prompts:
            bucket_key = ("__greeting__", prompt_norm)
            buckets[bucket_key].append((row["prompt"], row["response"]))
        prompts.add(prompt_norm)
        pairs.add(pair_norm)
        responses[response_norm] += 1
        unique.append(row)
    stats["final_examples"] = len(unique)
    return unique, dict(stats)


def balance_categories(rows: list[dict[str, str]], max_per_category: int = 2_000) -> list[dict[str, str]]:
    """Keep source-rich categories from overwhelming the conversational mix."""
    rng = random.Random(SEED)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    balanced: list[dict[str, str]] = []
    for category in ALLOWED_CATEGORIES:
        category_rows = grouped.get(category, [])
        rng.shuffle(category_rows)
        balanced.extend(category_rows[:max_per_category])
    return balanced


def split_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(SEED)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[normalize(row["prompt"])].append(row)
    groups = list(grouped.values())
    rng.shuffle(groups)
    rng.shuffle(rows)
    target_test = round(len(rows) * 0.10)
    target_val = round(len(rows) * 0.10)
    result = {"train": [], "val": [], "test": []}
    for group in groups:
        if len(result["test"]) + len(group) <= target_test:
            result["test"].extend(group)
        elif len(result["val"]) + len(group) <= target_val:
            result["val"].extend(group)
        else:
            result["train"].extend(group)
    return result


def metrics(rows: list[dict[str, str]]) -> dict[str, object]:
    prompts = [normalize(row["prompt"]) for row in rows]
    responses = [normalize(row["response"]) for row in rows]
    pairs = [f"{prompt}\n{response}" for prompt, response in zip(prompts, responses)]
    top = Counter(responses).most_common(20)
    return {
        "total_examples": len(rows),
        "unique_prompts": len(set(prompts)),
        "unique_responses": len(set(responses)),
        "unique_pairs": len(set(pairs)),
        "unique_normalized_response_ratio": len(set(responses)) / len(rows) if rows else 0,
        "average_prompt_characters": round(sum(map(len, prompts)) / len(rows), 2) if rows else 0,
        "average_response_characters": round(sum(map(len, responses)) / len(rows), 2) if rows else 0,
        "average_prompt_tokens": round(sum(map(token_count, prompts)) / len(rows), 2) if rows else 0,
        "average_response_tokens": round(sum(map(token_count, responses)) / len(rows), 2) if rows else 0,
        "minimum_response_characters": min(map(len, responses), default=0),
        "maximum_response_characters": max(map(len, responses), default=0),
        "top_response_frequency_ratio": round(top[0][1] / len(rows), 5) if top else 0,
        "top_20_normalized_responses": top,
    }


def category_response_metrics(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(normalize(row["response"]))
    result: dict[str, dict[str, object]] = {}
    for category, responses in sorted(grouped.items()):
        counts = Counter(responses)
        result[category] = {
            "examples": len(responses),
            "unique_normalized_responses": len(counts),
            "top_response_frequency_ratio": round(counts.most_common(1)[0][1] / len(responses), 5),
            "dominated": counts.most_common(1)[0][1] / len(responses) > 0.10,
        }
    return result


def greeting_stats(rows: list[dict[str, str]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for greeting in ("hi", "hey", "hello"):
        matches = [row for row in rows if normalize(row["prompt"]) == greeting]
        result[greeting] = {
            "count": len(matches),
            "unique_responses": len({normalize(row["response"]) for row in matches}),
            "examples": [{"prompt": row["prompt"], "response": row["response"]} for row in matches[:10]],
        }
    return result


def validate(rows: list[dict[str, str]], splits: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    errors: list[str] = []
    seen_prompts: dict[str, str] = {}
    seen_pairs: dict[str, str] = {}
    for split_name, split in splits.items():
        for row in split:
            prompt = normalize(row["prompt"])
            response = normalize(row["response"])
            pair = f"{prompt}\n{response}"
            if not prompt or not response:
                errors.append("empty prompt or response")
            if re.search(r"\[(?:variant|sample|example)[^\]]*\]|\b(?:example|sample)[_-]?\d{2,}\b", f"{prompt} {response}", re.I):
                errors.append("artificial identifier detected")
            if row["category"] not in ALLOWED_CATEGORIES:
                errors.append(f"invalid category: {row['category']}")
            if prompt in seen_prompts and seen_prompts[prompt] != split_name:
                errors.append("prompt leakage")
            if pair in seen_pairs and seen_pairs[pair] != split_name:
                errors.append("pair leakage")
            seen_prompts[prompt] = split_name
            seen_pairs[pair] = split_name
    return {"passed": not errors, "error_count": len(errors), "errors": sorted(set(errors))}


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"category": row["category"], "prompt": row["prompt"], "response": row["response"]}, ensure_ascii=False) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, source_files = source_examples()
    add_seed_examples(rows)
    final_rows, duplicate_stats = deduplicate(rows)
    final_rows = balance_categories(final_rows)
    duplicate_stats["final_examples_after_category_balance"] = len(final_rows)
    final_rows.sort(key=lambda row: (row["category"], normalize(row["prompt"])))
    splits = split_rows(final_rows)
    quality = validate(final_rows, splits)
    if not quality["passed"]:
        raise RuntimeError(f"EXP014 v2 quality validation failed: {quality}")

    for split_name, split in splits.items():
        write_jsonl(OUTPUT_DIR / f"instruction_{split_name}.jsonl", split)

    category_counts = {name: dict(Counter(row["category"] for row in split)) for name, split in splits.items()}
    all_metrics = metrics(final_rows)
    category_response_stats = category_response_metrics(final_rows)
    greetings = greeting_stats(final_rows)
    metadata = {
        "dataset_name": "exp014_general_chat_v2",
        "version": "2.0",
        "seed": SEED,
        "source_files": source_files,
        "category_counts": category_counts,
        "split_counts": {name: len(split) for name, split in splits.items()},
        "total_examples": len(final_rows),
        "unique_prompts": all_metrics["unique_prompts"],
        "unique_responses": all_metrics["unique_responses"],
        "unique_pairs": all_metrics["unique_pairs"],
        "duplicate_statistics": duplicate_stats,
        "response_diversity_statistics": all_metrics,
        "category_response_diversity": category_response_stats,
        "greeting_statistics": greetings,
        "quality_validation": quality,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = [
        "EXP014 GENERAL CHAT V2 DATASET REPORT",
        "======================================",
        f"Final examples: {len(final_rows)}",
        f"Splits: {metadata['split_counts']}",
        f"Sources: {', '.join(source_files)}",
        "",
        "Category distribution:",
    ]
    for split_name, counts in category_counts.items():
        report.append(f"{split_name}: {counts}")
    report.extend(["", "Response diversity:"])
    for key in ("unique_prompts", "unique_responses", "unique_pairs", "unique_normalized_response_ratio", "average_prompt_characters", "average_response_characters", "average_prompt_tokens", "average_response_tokens", "minimum_response_characters", "maximum_response_characters", "top_response_frequency_ratio"):
        report.append(f"{key}: {all_metrics[key]}")
    report.extend(["", "Category response diversity:", json.dumps(category_response_stats, indent=2), "", "Duplicate statistics:", json.dumps(duplicate_stats, indent=2), "", "Greeting statistics:", json.dumps(greetings, indent=2, ensure_ascii=False), "", "Top 20 normalized responses:", json.dumps(all_metrics["top_20_normalized_responses"], indent=2, ensure_ascii=False), "", f"Quality validation: {quality}"])
    (OUTPUT_DIR / "dataset_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({"metadata": metadata, "report": str(OUTPUT_DIR / "dataset_report.txt")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()