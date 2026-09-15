"""Evaluate the instruction-tuned model without retraining or external APIs."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch

# Make direct execution from the project root resolve sibling packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from inference.generate import (
	format_instruction_prompt,
	generate_tokens,
	load_model_and_tokenizer,
	select_device,
)
from training.instruct_train import (
	DEFAULT_DATASET,
	encode_examples,
	evaluate_loss,
	load_instruction_examples,
	split_examples,
)


DEFAULT_CHECKPOINT = Path("checkpoints/instruct/best_model.pt")
DEFAULT_PROMPTS = Path("evaluation/test_prompts.json")
DEFAULT_RESULTS = Path("evaluation/results.json")
DEFAULT_BEFORE_AFTER = Path("evaluation/before_after.json")
DEFAULT_REPORT = Path("evaluation/report.txt")
DEFAULT_HUMAN_EVAL = Path("evaluation/human_eval.md")


def load_prompts(path: Path) -> list[dict[str, str]]:
	if not path.exists():
		raise FileNotFoundError(f"Evaluation prompts not found: {path}")
	records = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(records, list) or not records:
		raise ValueError("Evaluation prompt file must contain a non-empty JSON list.")
	prompts: list[dict[str, str]] = []
	for index, record in enumerate(records, 1):
		if not isinstance(record, dict) or not isinstance(record.get("category"), str) or not isinstance(record.get("prompt"), str):
			raise ValueError(f"Invalid evaluation prompt at index {index}.")
		prompts.append({"category": record["category"], "prompt": record["prompt"]})
	return prompts


def repeated_pattern(text: str) -> bool:
	"""Detect adjacent repeated words or repeated short phrases."""
	words = re.findall(r"[A-Za-z0-9']+", text.lower())
	if len(words) < 2:
		return False
	if any(words[index] == words[index + 1] for index in range(len(words) - 1)):
		return True
	for phrase_length in (2, 3):
		for index in range(len(words) - (phrase_length * 2) + 1):
			first = words[index : index + phrase_length]
			second = words[index + phrase_length : index + (phrase_length * 2)]
			if first == second:
				return True
	return bool(re.search(r"(.{8,}?)(?:\s+\1){1,}", text, re.IGNORECASE))


def generate_results(
	model: torch.nn.Module,
	tokenizer: Any,
	device: torch.device,
	prompts: list[dict[str, str]],
	max_new_tokens: int,
	temperature: float,
	top_k: int,
	top_p: float,
) -> tuple[list[dict[str, str]], dict[str, float]]:
	results: list[dict[str, str]] = []
	lengths: list[int] = []
	repetition_count = 0
	empty_count = 0
	max_limit_count = 0
	for record in prompts:
		formatted_prompt = format_instruction_prompt(record["prompt"])
		prompt_ids = tokenizer.encode(formatted_prompt, add_bos=True)
		all_ids = generate_tokens(
			model,
			tokenizer,
			formatted_prompt,
			max_new_tokens,
			temperature,
			top_k,
			top_p,
			device,
			greedy=temperature == 0.0,
		)
		new_ids = all_ids[len(prompt_ids) :]
		response = tokenizer.decode(new_ids).strip()
		word_count = len(re.findall(r"\S+", response))
		lengths.append(word_count)
		empty_count += int(not response)
		repetition_count += int(repeated_pattern(response))
		reached_eos = bool(new_ids) and new_ids[-1] == tokenizer.eos_id
		max_limit_count += int(len(new_ids) >= max_new_tokens and not reached_eos)
		results.append(
			{
				"category": record["category"],
				"prompt": record["prompt"],
				"response": response,
			}
		)
	count = len(results)
	metrics = {
		"average_response_length_words": sum(lengths) / count,
		"repetition_rate": repetition_count / count,
		"empty_response_percentage": 100.0 * empty_count / count,
		"max_token_limit_percentage": 100.0 * max_limit_count / count,
	}
	return results, metrics


def calculate_validation_loss(
	model: torch.nn.Module,
	checkpoint_path: Path,
	tokenizer: Any,
	device: torch.device,
	seed: int,
) -> float | None:
	"""Recompute instruction validation loss when the source data is available."""
	if not DEFAULT_DATASET.exists():
		return None
	try:
		examples = load_instruction_examples(DEFAULT_DATASET)
		_, validation_raw = split_examples(examples, 0.1, seed)
		validation_examples = encode_examples(
			validation_raw, tokenizer, model.config.context_length
		)
		loss = evaluate_loss(
			model, validation_examples, model.config, tokenizer, device, random.Random(seed)
		)
		model.eval()
		return loss
	except (ValueError, KeyError):
		checkpoint = torch.load(checkpoint_path, map_location="cpu")
		stored_loss = checkpoint.get("validation_loss")
		return float(stored_loss) if stored_loss is not None and math.isfinite(float(stored_loss)) else None


def write_human_eval(results: list[dict[str, str]], path: Path) -> None:
	rows = [
		"# Human Evaluation",
		"",
		"Scores are intentionally blank. Review each response manually using a 1-5 scale.",
		"",
		"| Prompt | Response | Grammar (1-5) | Relevance (1-5) | Coherence (1-5) | Instruction Following (1-5) | Overall Quality (1-5) |",
		"|---|---|---:|---:|---:|---:|---:|",
	]
	for result in results:
		prompt = result["prompt"].replace("|", "\\|").replace("\n", " ")
		response = result["response"].replace("|", "\\|").replace("\n", " ")
		rows.append(f"| {prompt} | {response} |  |  |  |  |  |")
	rows.extend(
		[
			"",
			"Automatic metrics do not replace human judgment and do not establish grammar, intelligence, factuality, or overall quality.",
		]
	)
	path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_report(
	checkpoint_path: Path,
	parameter_count: int,
	prompts: list[dict[str, str]],
	results: list[dict[str, str]],
	metrics: dict[str, float],
	validation_loss: float | None,
	path: Path,
) -> None:
	perplexity = math.exp(min(validation_loss, 20.0)) if validation_loss is not None else None
	repetition_warning = metrics["repetition_rate"] > 0.2
	lines = [
		"MODEL EVALUATION REPORT",
		"=======================",
		f"Model: 10M parameter instruction-tuned Transformer",
		f"Checkpoint: {checkpoint_path}",
		f"Parameters: {parameter_count:,}",
		f"Number of test prompts: {len(prompts)}",
		f"Average response length (words): {metrics['average_response_length_words']:.2f}",
		f"Validation loss: {validation_loss:.4f}" if validation_loss is not None else "Validation loss: unavailable",
		f"Perplexity: {perplexity:.4f}" if perplexity is not None else "Perplexity: unavailable",
		f"Repetition rate: {metrics['repetition_rate']:.2%}",
		f"Empty responses: {metrics['empty_response_percentage']:.2f}%",
		f"Reached max token limit: {metrics['max_token_limit_percentage']:.2f}%",
		"",
		"AUTOMATIC SUMMARY",
		"-----------------",
		"Prompt following, grammar, coherence, relevance, and factuality require human review in human_eval.md.",
		"The automatic metrics describe output behavior but do not measure intelligence or guarantee grammatical correctness.",
		"Repetition warning: inspect responses for repeated words or phrases." if repetition_warning else "No excessive repetition threshold was detected by the simple heuristic.",
		"Known limitations: the model is small, may be incomplete or factually incorrect, and was evaluated on a small manually written prompt set.",
		"",
		"GENERATED RESPONSES",
		"-------------------",
	]
	for result in results:
		lines.extend([f"[{result['category']}] {result['prompt']}", result["response"], ""])
	path.write_text("\n".join(lines), encoding="utf-8")


def run_before_after(
	base_checkpoint: Path,
	tuned_results: list[dict[str, str]],
	tokenizer_dir: Path,
	device: torch.device,
	max_new_tokens: int,
	temperature: float,
	top_k: int,
	top_p: float,
	path: Path,
) -> None:
	if not base_checkpoint.exists():
		path.write_text("[]\n", encoding="utf-8")
		print(f"Skipped before/after comparison; checkpoint not found: {base_checkpoint}")
		return
	base_model, tokenizer, _ = load_model_and_tokenizer(base_checkpoint, tokenizer_dir, device)
	comparison: list[dict[str, str]] = []
	for result in tuned_results:
		formatted_prompt = format_instruction_prompt(result["prompt"])
		prompt_ids = tokenizer.encode(formatted_prompt, add_bos=True)
		all_ids = generate_tokens(
			base_model, tokenizer, formatted_prompt, max_new_tokens,
			temperature, top_k, top_p, device, greedy=temperature == 0.0,
		)
		comparison.append(
			{
				"prompt": result["prompt"],
				"pretrained_response": tokenizer.decode(all_ids[len(prompt_ids) :]).strip(),
				"instruction_tuned_response": result["response"],
			}
		)
	path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate the trained Transformer without retraining.")
	parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
	parser.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizer"))
	parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
	parser.add_argument("--max-new-tokens", type=int, default=100)
	parser.add_argument("--temperature", type=float, default=0.0)
	parser.add_argument("--top-k", type=int, default=0)
	parser.add_argument("--top-p", type=float, default=1.0)
	parser.add_argument("--seed", type=int, default=42)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	prompts = load_prompts(args.prompts)
	device = select_device()
	model, tokenizer, device = load_model_and_tokenizer(args.checkpoint, args.tokenizer_dir, device)
	results, metrics = generate_results(
		model, tokenizer, device, prompts, args.max_new_tokens,
		args.temperature, args.top_k, args.top_p,
	)
	validation_loss = calculate_validation_loss(model, args.checkpoint, tokenizer, device, args.seed)
	perplexity = math.exp(min(validation_loss, 20.0)) if validation_loss is not None else float("nan")
	args.results_path = DEFAULT_RESULTS
	args.results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	write_human_eval(results, DEFAULT_HUMAN_EVAL)
	write_report(
		args.checkpoint, model.parameter_count(), prompts, results,
		metrics, validation_loss, DEFAULT_REPORT,
	)
	run_before_after(
		Path("checkpoints/best_model.pt"), results, args.tokenizer_dir, device,
		args.max_new_tokens, args.temperature, args.top_k, args.top_p,
		DEFAULT_BEFORE_AFTER,
	)
	print("# ========================================")
	print("MODEL EVALUATION")
	print("# ========================================")
	print(f"Parameters: ~{model.parameter_count() / 1_000_000:.2f}M")
	print(f"Test prompts: {len(prompts)}")
	print(f"Validation Loss: {validation_loss:.4f}" if validation_loss is not None else "Validation Loss: unavailable")
	print(f"Perplexity: {perplexity:.4f}" if math.isfinite(perplexity) else "Perplexity: unavailable")
	print(f"Average Response Length: {metrics['average_response_length_words']:.2f} words")
	print(f"Repetition Rate: {metrics['repetition_rate']:.2%}")
	if metrics["repetition_rate"] > 0.2:
		print("Warning: excessive repetition detected by the simple heuristic.")
	print("\n# ========================================")
	print("SAMPLE RESPONSES")
	print("# ========================================")
	for result in results[:2]:
		print(f"Prompt:\n{result['prompt']}\n\nResponse:\n{result['response']}\n")
	print("Saved evaluation/results.json, evaluation/before_after.json, evaluation/report.txt, and evaluation/human_eval.md")


if __name__ == "__main__":
	main()
