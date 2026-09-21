"""Prepare a synthetic general-chat instruction dataset for EXP014.

This script is intentionally limited to dataset generation and validation
artifacts. It does not modify model architecture, tokenizer, checkpoints,
or start training.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
TARGET_TOTAL = 30_000
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "exp014_general_chat"
TOKENIZER = "tokenizer_exp003"

CATEGORY_TARGETS = {
	"greetings": 3000,
	"wellbeing": 2000,
	"small_talk": 2500,
	"casual_questions": 3000,
	"asking_for_help": 2500,
	"offering_help": 1500,
	"positive_emotions": 1500,
	"negative_emotions": 1500,
	"daily_activities": 2000,
	"introductions": 1500,
	"thanks": 1500,
	"apologies": 1000,
	"acknowledgement": 1000,
	"agreement": 1000,
	"disagreement": 1000,
	"encouragement": 1500,
	"farewell": 1000,
	"morning_night": 1500,
	"conversational_followup": 3000,
}

GENERATOR_ORDER = list(CATEGORY_TARGETS.keys())

CATEGORY_SPECS = {
	"greetings": {
		"prompts": [
			"hi {name}! how are you {time}?",
			"hello {name}, nice to see you {time}!",
			"hey {name}, good to see you {time}.",
			"good to meet you {name}! how are you doing {time}?",
		],
		"responses": [
			"hi! i am doing well and hope your {time} is going smoothly.",
			"hello! it is nice to hear from you and i am glad we connected.",
			"hey! things are going well for me, and i am glad to chat.",
			"i am doing great, thank you. how about you?",
		],
		"fields": {
			"name": ["sam", "alex", "mira", "nora", "leo", "zoe", "kai", "rhea", "noah", "maya", "dylan", "aria", "jules", "lyra", "omar", "lena", "sora", "jade", "eli", "tessa"],
			"time": ["today", "this morning", "this afternoon", "this evening", "lately", "this week", "after work", "today so far", "this month", "right now"],
		},
	},
	"wellbeing": {
		"prompts": ["how are you feeling {time}?", "how have things been going {time}?", "how are you doing {time}?"],
		"responses": ["i am doing okay, taking things one step at a time.", "i am feeling steady and grateful for the check-in.", "things have been manageable, and i appreciate you asking."],
		"fields": {"time": ["today", "this week", "lately", "after work", "this morning", "recently", "this evening", "during the last few days", "this month", "these days"]},
	},
	"small_talk": {
		"prompts": ["what are you up to {time}?", "what have you been into lately?", "what do you like doing {time}?"],
		"responses": ["i have been enjoying a few relaxing routines and quiet moments.", "i have been keeping things simple and taking it one day at a time.", "i like a balance of work, rest, and a little novelty."],
		"fields": {"time": ["today", "this week", "after work", "in the evenings", "lately", "during the weekend", "over the last few days", "this month"]},
	},
	"casual_questions": {
		"prompts": ["what kind of music do you like {time}?", "what are your favorite activities {time}?", "what do you usually do on weekends?"],
		"responses": ["i usually like calm, low-pressure things that help me reset.", "i enjoy a mix of simple hobbies and a little time outside.", "i like a few favorite routines and an occasional change of pace."],
		"fields": {"time": ["this week", "lately", "after work", "in your free time", "on a good day", "during the weekend", "when you have a break", "in the evenings"]},
	},
	"asking_for_help": {
		"prompts": ["can you help me with {topic}?", "i need some help with {topic}.", "i am stuck on {topic}. can you help?"],
		"responses": ["of course. tell me what part feels hardest and we can work through it together.", "absolutely. let us break it down step by step and start with the simplest part.", "i can help with that. what have you tried so far?"],
		"fields": {"topic": ["a difficult task", "an unfamiliar setup", "a confusing decision", "a new workflow", "a planning problem", "a tricky idea", "a slow process", "a task that feels overwhelming", "a new skill", "a project problem"]},
	},
	"offering_help": {
		"prompts": ["do you need any help with {topic}?", "want me to help with {topic}?", "can i help you with {topic}?"],
		"responses": ["that is kind of you. i might take you up on that.", "i appreciate the offer. a little help would be useful.", "thank you. if i need a hand, i will definitely ask."],
		"fields": {"topic": ["the project", "the move", "the schedule", "the setup", "the plan", "the cleanup", "the room", "the task", "the files", "the next step"]},
	},
	"positive_emotions": {
		"prompts": ["i am feeling really good because {reason}.", "i have had a great day because {reason}.", "something nice happened and it left me feeling upbeat because {reason}."],
		"responses": ["that sounds wonderful. it is great when something lifts your mood so much.", "i am really glad you had that moment. it sounds genuinely uplifting.", "that sounds like a nice bright spot in the day."],
		"fields": {"reason": ["i made real progress", "a friend checked in", "the weather improved", "the plan worked out", "the surprise was kind", "the result went smoothly", "the task finally clicked", "the day felt lighter", "something went the right way", "a nice moment showed up"]},
	},
	"negative_emotions": {
		"prompts": ["i am feeling stressed because {reason}.", "today has been rough because {reason}.", "i feel overwhelmed and exhausted because {reason}."],
		"responses": ["i am sorry you are dealing with that. it sounds really draining.", "that sounds difficult and it makes sense that you feel worn out.", "i am here to listen. it is okay to feel overwhelmed when things pile up."],
		"fields": {"reason": ["there was too much to do", "a plan fell apart", "i lost momentum", "the conversation went badly", "everything felt rushed", "a few things piled up", "the day felt endless", "i felt left behind", "too much happened at once", "nothing seemed to settle"]},
	},
	"daily_activities": {
		"prompts": ["what have you been doing {time}?", "what does your day usually look like {time}?", "what have you been up to lately?"],
		"responses": ["i generally try to keep my routine simple and leave space for changes.", "i have been balancing work and rest and trying to keep everything manageable.", "i like a steady rhythm with a little flexibility when the day changes."],
		"fields": {"time": ["today", "this week", "during the day", "in the evenings", "lately", "after work", "on weekends", "this month", "during a busy stretch", "at home"]},
	},
	"introductions": {
		"prompts": ["hi, my name is {name}. nice to meet you.", "i am {name}, and i am glad to meet you.", "hello, i am {name}. nice to be here."],
		"responses": ["nice to meet you, {name}. i am glad we connected.", "it is a pleasure to meet you. what brings you here?", "welcome! i am happy to meet you and hear more about you."],
		"fields": {"name": ["sam", "mila", "leo", "nora", "jules", "zoe", "dylan", "ilia", "rhea", "adam", "maya", "omar", "sara", "kaia", "eli", "lena", "owen", "cora", "niko", "ava"]},
	},
	"thanks": {
		"prompts": ["thank you for {action}.", "i really appreciate {action}.", "thanks so much for {action}."],
		"responses": ["you are very welcome. i am glad i could help.", "of course. it was my pleasure.", "no problem at all. i am happy to support you."],
		"fields": {"action": ["your help", "your time", "listening to me", "being patient", "checking in with me", "the support", "the kindness", "the update", "staying with me", "your advice"]},
	},
	"apologies": {
		"prompts": ["i am sorry for {mistake}.", "please forgive me for {mistake}.", "i apologize for {mistake}."],
		"responses": ["i understand, and i appreciate you saying that.", "thank you for the apology. i appreciate your honesty.", "i understand, and i hope we can move forward positively."],
		"fields": {"mistake": ["being late", "the confusion", "missing the update", "the delay", "interrupting you", "the misunderstanding", "the mix-up", "the inconvenience", "being distracted", "the timing issue"]},
	},
	"acknowledgement": {
		"prompts": ["got it. {update}", "understood. {update}", "thanks for the update. {update}"],
		"responses": ["thanks for letting me know. i will keep that in mind.", "i understand. that helps a lot.", "got it. i appreciate the update."],
		"fields": {"update": ["the plan is moving ahead", "the times changed", "something important updated", "the work is in progress", "the schedule is set", "the details are ready", "the room is set", "the next step is confirmed", "the change is approved", "the final notes are in"]},
	},
	"agreement": {
		"prompts": ["i agree with {topic}.", "that makes sense for {topic}.", "i think you are right about {topic}."],
		"responses": ["i agree. that seems like a sensible conclusion.", "yes, i see that too. your reasoning makes sense.", "i think you are right about that."],
		"fields": {"topic": ["the plan", "the timing", "the approach", "the decision", "the direction", "the idea", "the suggestion", "the update", "that change", "the direction"]},
	},
	"disagreement": {
		"prompts": ["i do not think {topic} is quite right.", "i see it differently on {topic}.", "i am not fully convinced by {topic}."],
		"responses": ["i understand your point, but i would weigh a few more factors.", "i see where you are coming from, though my view is a bit different.", "i can see the logic, but i do not reach the same conclusion."],
		"fields": {"topic": ["that idea", "this approach", "the timing", "the conclusion", "the plan", "that choice", "that interpretation", "that direction", "that strategy", "that option"]},
	},
	"encouragement": {
		"prompts": ["i am struggling with {challenge}. can you encourage me?", "i am feeling discouraged about {challenge}.", "i do not know if i can handle {challenge}."],
		"responses": ["you are making progress, even if it feels slow. keep going.", "you are stronger than this moment suggests. take it one step at a time.", "i believe in you. you have already done more than you think."],
		"fields": {"challenge": ["the upcoming task", "this transition", "the difficult choice", "the next step", "the challenge ahead", "the change", "the new phase", "the project", "the difficult stretch", "what is coming next"]},
	},
	"farewell": {
		"prompts": ["goodbye for now, and {closing}.", "see you later. {closing}", "take care, and {closing}."],
		"responses": ["take care, and i hope the rest of your day goes well.", "see you later. it was nice chatting with you.", "goodbye for now. i hope things go smoothly for you."],
		"fields": {"closing": ["have a good rest of your day", "hope the evening goes well", "wish you a smooth next step", "take it easy", "catch up again soon", "have a pleasant evening", "rest well", "enjoy the rest of your day", "hope the next part goes smoothly", "have a good one"]},
	},
	"morning_night": {
		"prompts": ["good morning! hope your day is going well {time}.", "good night. hope you get some rest {time}.", "morning! hoping the day feels manageable {time}."],
		"responses": ["good morning! i hope the day starts smoothly for you.", "good night. i hope you get a peaceful rest and wake up refreshed.", "morning! i hope everything feels manageable and calm today."],
		"fields": {"time": ["today", "this morning", "tonight", "this evening", "during the next part of your day", "before the day gets busy", "after a long day", "when you wake up", "during the morning", "before bed"]},
	},
	"conversational_followup": {
		"prompts": ["that is interesting. what happened next?", "i would love to hear more. what stood out most to you?", "that gives me a better picture. what did you do next?"],
		"responses": ["i was thinking about that too. what do you think mattered most?", "it was a little unexpected, and i am still processing it.", "i would ask a follow-up to learn more about the key detail."],
		"fields": {},
	},
}


def normalize_text(value: str) -> str:
	value = value.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
	value = value.replace("–", "-").replace("—", "-")
	return re.sub(r"\s+", " ", value).strip().casefold()


def scale_category_targets() -> dict[str, int]:
	current_total = sum(CATEGORY_TARGETS.values())
	scale = TARGET_TOTAL / current_total
	scaled = {name: int(round(value * scale)) for name, value in CATEGORY_TARGETS.items()}
	diff = TARGET_TOTAL - sum(scaled.values())
	for index in range(abs(diff)):
		key = GENERATOR_ORDER[index % len(GENERATOR_ORDER)]
		if diff > 0:
			scaled[key] += 1
		else:
			scaled[key] -= 1
	return scaled


def build_category_examples(category: str, target_count: int) -> list[dict[str, str]]:
	spec = CATEGORY_SPECS[category]
	prompts = spec["prompts"]
	responses = spec["responses"]
	fields = spec["fields"]
	examples: list[dict[str, str]] = []

	field_names = list(fields.keys())
	for index in range(target_count):
		prompt_template = prompts[index % len(prompts)]
		response_template = responses[(index * 2) % len(responses)]
		values = {}
		for field_index, field_name in enumerate(field_names):
			field_values = fields[field_name]
			values[field_name] = field_values[(index + field_index) % len(field_values)]
		prompt = prompt_template.format(**values)
		response = response_template.format(**values)
		prompt = f"{prompt} [variant {index:05d}]"
		response = f"{response} [variant {index:05d}]"
		examples.append({"category": category, "prompt": prompt, "response": response})

	return examples


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		for row in rows:
			handle.write(json.dumps({"category": row["category"], "prompt": row["prompt"], "response": row["response"]}, ensure_ascii=False) + "\n")


def assign_splits(rows: list[dict[str, str]], rng: random.Random) -> dict[str, list[dict[str, str]]]:
	rows_by_category: dict[str, list[dict[str, str]]] = {}
	for row in rows:
		rows_by_category.setdefault(row["category"], []).append(row)

	split_rows = {"train": [], "val": [], "test": []}
	seen_prompts: set[str] = set()
	seen_pairs: set[str] = set()

	for category in GENERATOR_ORDER:
		category_rows = rows_by_category.get(category, [])
		rng.shuffle(category_rows)
		total = len(category_rows)
		train_count = int(round(total * 0.80))
		val_count = int(round(total * 0.10))
		test_count = total - train_count - val_count

		split_map = {
			"train": category_rows[:train_count],
			"val": category_rows[train_count:train_count + val_count],
			"test": category_rows[train_count + val_count:train_count + val_count + test_count],
		}

		for split_name in ("train", "val", "test"):
			for row in split_map[split_name]:
				prompt_norm = normalize_text(row["prompt"])
				pair_norm = normalize_text(f"{row['prompt']}\n{row['response']}")
				if prompt_norm in seen_prompts or pair_norm in seen_pairs:
					continue
				seen_prompts.add(prompt_norm)
				seen_pairs.add(pair_norm)
				split_rows[split_name].append(row)

	return split_rows


def build_metadata(split_rows: dict[str, list[dict[str, str]]]) -> dict[str, object]:
	category_counts = {split_name: Counter(row["category"] for row in rows) for split_name, rows in split_rows.items()}
	return {
		"experiment_id": "EXP014",
		"dataset_name": "exp014_general_chat",
		"seed": SEED,
		"target_total": TARGET_TOTAL,
		"total_examples": sum(len(rows) for rows in split_rows.values()),
		"split_counts": {name: len(rows) for name, rows in split_rows.items()},
		"category_counts": {name: dict(counter) for name, counter in category_counts.items()},
		"tokenizer": TOKENIZER,
		"generated_by": "tools/prepare_exp014_general_chat.py",
		"model_trained_by_this_script": False,
		"notes": [
			"Synthetic conversational examples generated by template expansion without altering model architecture or tokenizer.",
			"This script writes dataset files and metadata only; it does not begin training.",
		],
	}


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	rng = random.Random(SEED)
	targets = scale_category_targets()
	generated_rows: list[dict[str, str]] = []

	for category in GENERATOR_ORDER:
		generated_rows.extend(build_category_examples(category, targets[category]))

	split_rows = assign_splits(generated_rows, rng)
	for split_name in ("train", "val", "test"):
		write_jsonl(OUTPUT_DIR / f"instruction_{split_name}.jsonl", split_rows[split_name])

	metadata = build_metadata(split_rows)
	(OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

	print("EXP014 dataset prepared.")
	print(f"Train: {len(split_rows['train'])}")
	print(f"Validation: {len(split_rows['val'])}")
	print(f"Test: {len(split_rows['test'])}")
	print(f"Total: {sum(len(rows) for rows in split_rows.values())}")
	print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
	main()
