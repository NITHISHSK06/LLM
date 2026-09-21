"""Validate the generated EXP014 dataset and emit a summary report."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "exp014_general_chat"


def normalize_text(value: str) -> str:
	value = value.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
	value = value.replace("–", "-").replace("—", "-")
	return re.sub(r"\s+", " ", value).strip().casefold()


def load_jsonl(path: Path) -> list[dict[str, str]]:
	rows: list[dict[str, str]] = []
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			if line.strip():
				rows.append(json.loads(line))
	return rows


def main() -> None:
	splits = {name: load_jsonl(DATA_DIR / f"instruction_{name}.jsonl") for name in ("train", "val", "test")}
	all_rows = [row for rows in splits.values() for row in rows]

	prompt_lookup: dict[str, str] = {}
	pair_lookup: dict[str, str] = {}
	for split_name, rows in splits.items():
		for row in rows:
			prompt_norm = normalize_text(str(row["prompt"]))
			pair_norm = normalize_text(f"{row['prompt']}\n{row['response']}")
			if prompt_norm in prompt_lookup and prompt_lookup[prompt_norm] != split_name:
				raise ValueError(f"Prompt overlap across splits: {prompt_norm}")
			if pair_norm in pair_lookup and pair_lookup[pair_norm] != split_name:
				raise ValueError(f"Prompt-response overlap across splits: {pair_norm}")
			prompt_lookup[prompt_norm] = split_name
			pair_lookup[pair_norm] = split_name

	category_counts = {split_name: Counter(row["category"] for row in rows) for split_name, rows in splits.items()}
	response_counts = Counter(normalize_text(str(row["response"])) for row in all_rows)

	report_lines = [
		"EXP014 Dataset Validation Report",
		"==============================",
		f"Total rows: {len(all_rows)}",
		f"Train rows: {len(splits['train'])}",
		f"Validation rows: {len(splits['val'])}",
		f"Test rows: {len(splits['test'])}",
		"",
		"Category counts:",
	]
	for split_name in ("train", "val", "test"):
		report_lines.append(f"{split_name}: {dict(sorted(category_counts[split_name].items()))}")

	report_lines.extend(["", "Most frequent normalized responses:", json.dumps(response_counts.most_common(10), ensure_ascii=False)])
	report_path = DATA_DIR / "dataset_report.txt"
	report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

	print("EXP014 validation complete.")
	print(f"Train: {len(splits['train'])}")
	print(f"Validation: {len(splits['val'])}")
	print(f"Test: {len(splits['test'])}")
	print(f"Report: {report_path}")


if __name__ == "__main__":
	main()
