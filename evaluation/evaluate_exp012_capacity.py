"""Compare Exp011 and Exp012 on the unchanged Exp011 held-out test set."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

INFERENCE_DIR = PROJECT_ROOT / "experiments" / "inference"
if str(INFERENCE_DIR) not in sys.path:
	sys.path.insert(0, str(INFERENCE_DIR))

from evaluation.evaluate_exp010_general_chat import (
	RELEVANCE_CUES,
	generate_outputs,
	load_model,
	normalize,
	repetition_metrics,
)
from experiments.exp012_capacity import TOKENIZER_DIR
from model.model import DecoderLanguageModel
from tokenizer.tokenizer import SimpleBPETokenizer
from training.instruct_train import encode_examples, evaluate_loss, load_split_examples

BASELINE_CHECKPOINT = PROJECT_ROOT / "checkpoints/exp011_general_chat/best_model.pt"
EXP012_CHECKPOINT = PROJECT_ROOT / "checkpoints/exp012_capacity_25m_sft/best_model.pt"
TEST_DATA = PROJECT_ROOT / "data/processed/exp011_general_chat/instruction_test.jsonl"
VALIDATION_DATA = PROJECT_ROOT / "data/processed/exp011_general_chat/instruction_val.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evaluation/results"
METRICS_PATH = RESULTS_DIR / "exp012_capacity_comparison.json"
REPORT_PATH = RESULTS_DIR / "exp012_capacity_comparison.txt"
QUALITATIVE_PATH = RESULTS_DIR / "exp012_capacity_qualitative.jsonl"
SEED = 42


def load_records(path: Path) -> list[dict[str, Any]]:
	records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
	if not records or any(not isinstance(record, dict) for record in records):
		raise ValueError(f"Invalid or empty evaluation data: {path}")
	for record in records:
		for field in ("category", "prompt", "response"):
			if not isinstance(record.get(field), str):
				raise ValueError(f"Evaluation record is missing string field {field}: {record}")
	return records


def predicted_category(text: str, categories: list[str]) -> str:
	normalized = normalize(text)
	scores = {
		category: sum(cue in normalized for cue in RELEVANCE_CUES.get(category, ()))
		for category in categories
	}
	best_category, best_score = max(scores.items(), key=lambda item: item[1])
	return best_category if best_score else "unknown"


def diagnostic_metrics(records: list[dict[str, Any]], outputs: list[str]) -> dict[str, Any]:
	categories = sorted({record["category"] for record in records})
	repetitions = [repetition_metrics(output) for output in outputs]
	normalized_outputs = [normalize(output) for output in outputs]
	lengths = [len(output) for output in outputs]
	token_lengths = [len(re.findall(r"\S+", output)) for output in outputs]
	alignment = [
		any(cue in normalize(output) for cue in RELEVANCE_CUES.get(record["category"], ()))
		for record, output in zip(records, outputs)
	]
	predicted = [predicted_category(output, categories) for output in outputs]
	confusion_labels = categories + ["unknown"]
	confusion = {actual: {label: 0 for label in confusion_labels} for actual in categories}
	for record, predicted_label in zip(records, predicted):
		confusion[record["category"]][predicted_label] += 1
	counts = Counter(value for value in normalized_outputs if value)
	generic_templates = {
		"i understand",
		"i am here to help",
		"i am happy to help",
		"that is a good question",
		"i am sorry to hear that",
		"thank you for sharing",
	}
	template_hits = sum(value in generic_templates for value in normalized_outputs)
	duplicate_response_count = sum(count for count in counts.values() if count > 1)
	return {
		"test_size": len(records),
		"response_repetition_rate": mean(item["repetition_flag"] for item in repetitions),
		"repeated_unigram_ratio": mean(item["repeated_unigram_ratio"] for item in repetitions),
		"repeated_bigram_ratio": mean(item["repeated_bigram_ratio"] for item in repetitions),
		"response_length": {
			"average_characters": mean(lengths),
			"average_tokens": mean(token_lengths),
			"minimum_characters": min(lengths),
			"maximum_characters": max(lengths),
		},
		"heuristic_category_alignment_proxy": mean(alignment),
		"heuristic_category_mismatch_proxy": 1.0 - mean(alignment),
		"diagnostic_note": "Heuristic category alignment is a diagnostic proxy, not semantic accuracy.",
		"generic_response_template_frequency": template_hits / len(outputs),
		"duplicate_normalized_response_frequency": duplicate_response_count / len(outputs),
		"unique_normalized_response_ratio": len(set(normalized_outputs)) / len(outputs),
		"category_confusion_matrix": confusion,
	}


def validation_metrics(
	model: DecoderLanguageModel,
	validation_path: Path,
	tokenizer: SimpleBPETokenizer,
	device: torch.device,
) -> dict[str, float]:
	config = model.config
	examples = encode_examples(load_split_examples(validation_path), tokenizer, config.context_length)
	loss = evaluate_loss(model, examples, config, tokenizer, device, random.Random(SEED))
	return {"validation_loss": loss, "perplexity": math.exp(min(loss, 20.0))}


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate Exp011 versus Exp012 on the same Exp011 test set.")
	parser.add_argument("--baseline-checkpoint", type=Path, default=BASELINE_CHECKPOINT)
	parser.add_argument("--exp012-checkpoint", type=Path, default=EXP012_CHECKPOINT)
	parser.add_argument("--test-data", type=Path, default=TEST_DATA)
	parser.add_argument("--validation-data", type=Path, default=VALIDATION_DATA)
	parser.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DIR)
	args = parser.parse_args()
	for path in (args.baseline_checkpoint, args.exp012_checkpoint, args.test_data, args.validation_data, args.tokenizer_dir):
		if not path.exists():
			raise FileNotFoundError(f"Required evaluation path not found: {path}")

	records = load_records(args.test_data)
	if len(records) != 1000:
		raise ValueError(f"Expected the unchanged Exp011 test set of 1,000 records, found {len(records)}.")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	tokenizer = SimpleBPETokenizer.load(args.tokenizer_dir)
	models = {
		"exp011_baseline": load_model(args.baseline_checkpoint, tokenizer, device),
		"exp012_capacity": load_model(args.exp012_checkpoint, tokenizer, device),
	}
	outputs: dict[str, list[str]] = {}
	for name, model in models.items():
		torch.manual_seed(SEED)
		outputs[name] = generate_outputs(model, tokenizer, records, device, 0.0, 0, 1.0)

	metrics: dict[str, Any] = {
		"experiment": "Exp012 model capacity scaling",
		"research_question": "Does increasing model capacity from approximately 10M to approximately 25M parameters improve general conversational instruction-following?",
		"hypothesis": "Increasing model capacity may improve semantic instruction-following and reduce generic or off-topic generation, while all other experimental conditions remain controlled.",
		"seed": SEED,
		"test_set": str(args.test_data),
		"tokenizer": str(args.tokenizer_dir),
		"diagnostic_note": "Heuristic category alignment and mismatch are diagnostic proxies, not semantic accuracy.",
		"models": {},
	}
	for name, model in models.items():
		model_metrics = diagnostic_metrics(records, outputs[name])
		model_metrics["total_parameter_count"] = model.parameter_count()
		model_metrics.update(validation_metrics(model, args.validation_data, tokenizer, device))
		metrics["models"][name] = model_metrics

	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	with QUALITATIVE_PATH.open("w", encoding="utf-8") as handle:
		for index, record in enumerate(records):
			if index < 10 or index % max(1, len(records) // 20) == 0:
				handle.write(json.dumps({
					"index": index + 1,
					"category": record["category"],
					"prompt": record["prompt"],
					"expected_response": record["response"],
					"exp011_baseline": outputs["exp011_baseline"][index],
					"exp012_capacity": outputs["exp012_capacity"][index],
				}, ensure_ascii=False) + "\n")

	lines = [
		"Exp012 Capacity Scaling Evaluation",
		"==================================",
		metrics["research_question"],
		metrics["hypothesis"],
		"Heuristic category alignment/mismatch are diagnostic proxies, NOT semantic accuracy.",
		f"Test set: {args.test_data} (unchanged Exp011 test set; {len(records)} records)",
		"",
	]
	for name, values in metrics["models"].items():
		lines.extend([
			name,
			f"  total parameters: {values['total_parameter_count']:,}",
			f"  validation loss: {values['validation_loss']:.6f}",
			f"  perplexity: {values['perplexity']:.6f}",
			f"  response repetition: {values['response_repetition_rate']:.6f}",
			f"  repeated unigram ratio: {values['repeated_unigram_ratio']:.6f}",
			f"  repeated bigram ratio: {values['repeated_bigram_ratio']:.6f}",
			f"  response length: {values['response_length']}",
			f"  heuristic category alignment proxy: {values['heuristic_category_alignment_proxy']:.6f}",
			f"  heuristic category mismatch proxy: {values['heuristic_category_mismatch_proxy']:.6f}",
			f"  generic response/template frequency: {values['generic_response_template_frequency']:.6f}",
			f"  unique normalized response ratio: {values['unique_normalized_response_ratio']:.6f}",
			"  category confusion matrix:",
			json.dumps(values["category_confusion_matrix"], indent=2),
			"",
		])
	lines.extend([
		"Representative qualitative outputs are in exp012_capacity_qualitative.jsonl.",
		f"Machine-readable metrics: {METRICS_PATH}",
	])
	REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
	print(f"Wrote {REPORT_PATH}")
	print(f"Wrote {METRICS_PATH}")
	print(f"Wrote {QUALITATIVE_PATH}")
	print("Evaluation completed without training or modifying either checkpoint.")


if __name__ == "__main__":
	main()