"""Evaluate coding specialization and general capability retention."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.generate import format_instruction_prompt, generate_tokens, load_model_and_tokenizer, select_device

CODING_PROMPTS = Path("evaluation/coding_test_prompts.json")
RETENTION_PROMPTS = Path("evaluation/general_retention_prompts.json")


def records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return value


def metrics(responses: list[str], max_new_tokens: int, tokenizer: Any) -> dict[str, float]:
    if not responses:
        return {"average_response_length_words": 0.0, "empty_response_percentage": 100.0, "repetition_rate": 0.0, "max_token_limit_percentage": 0.0, "malformed_output_rate": 0.0, "code_block_frequency": 0.0}
    repetition = 0
    maxed = 0
    malformed = 0
    code_blocks = 0
    lengths = []
    for response in responses:
        words = re.findall(r"\w+", response.casefold())
        lengths.append(len(words))
        repetition += int(bool(re.search(r"\b(\w+)(?:\s+\1){1,}\b", response, re.IGNORECASE)))
        repetition += int(bool(re.search(r"(.{8,}?)(?:\s+\1)", response, re.IGNORECASE)))
        maxed += int(len(tokenizer.encode(response)) >= max_new_tokens)
        code_blocks += int("```" in response)
        malformed += int(response.count("```") % 2 != 0)
    return {
        "average_response_length_words": sum(lengths) / len(lengths),
        "empty_response_percentage": 100.0 * sum(not response.strip() for response in responses) / len(responses),
        "repetition_rate": min(1.0, repetition / len(responses)),
        "max_token_limit_percentage": 100.0 * maxed / len(responses),
        "malformed_output_rate": 100.0 * malformed / len(responses),
        "code_block_frequency": 100.0 * code_blocks / len(responses),
    }


def run_model(path: Path, prompts: list[dict[str, Any]], tokenizer_dir: Path, device: torch.device, max_new_tokens: int) -> tuple[list[str], dict[str, float] | None, int | None]:
    if not path.exists():
        return [""] * len(prompts), None, None
    model, tokenizer, _ = load_model_and_tokenizer(path, tokenizer_dir, device)
    responses: list[str] = []
    for item in prompts:
        formatted = format_instruction_prompt(item["prompt"])
        prompt_ids = tokenizer.encode(formatted, add_bos=True)
        generated = generate_tokens(model, tokenizer, formatted, max_new_tokens, 0.0, 0, 1.0, device, greedy=True)
        responses.append(tokenizer.decode(generated[len(prompt_ids):]).strip())
    return responses, metrics(responses, max_new_tokens, tokenizer), model.parameter_count()


def checkpoint_loss(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"validation_loss": None, "perplexity": None}
    state = torch.load(path, map_location="cpu")
    loss = state.get("validation_loss")
    loss = float(loss) if loss is not None else None
    return {"validation_loss": loss, "perplexity": math.exp(min(loss, 20.0)) if loss is not None else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate coding specialization and retention.")
    parser.add_argument("--base", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--v1", type=Path, default=Path("checkpoints/instruct_v2/best_model.pt"))
    parser.add_argument("--coding", type=Path, default=Path("checkpoints/coding/best_model.pt"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizer"))
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--coding-output", type=Path, default=Path("evaluation/coding_model_comparison.json"))
    parser.add_argument("--retention-output", type=Path, default=Path("evaluation/general_retention_report.json"))
    args = parser.parse_args()
    coding_prompts = records(CODING_PROMPTS)
    retention_prompts = records(RETENTION_PROMPTS)
    device = select_device()
    paths = {"base": args.base, "instruction_v1": args.v1, "coding_specialized": args.coding}
    coding_runs: dict[str, tuple[list[str], dict[str, float] | None, int | None]] = {}
    retention_runs: dict[str, tuple[list[str], dict[str, float] | None, int | None]] = {}
    for label, path in paths.items():
        print(f"Evaluating {label}: {path}")
        coding_runs[label] = run_model(path, coding_prompts, args.tokenizer_dir, device, args.max_new_tokens)
        if label in {"instruction_v1", "coding_specialized"}:
            retention_runs[label] = run_model(path, retention_prompts, args.tokenizer_dir, device, args.max_new_tokens)

    comparison = []
    for index, prompt in enumerate(coding_prompts):
        comparison.append({
            "id": prompt["id"], "category": prompt["category"], "prompt": prompt["prompt"],
            "expected_behavior": prompt["expected_behavior"],
            "base_response": coding_runs["base"][0][index],
            "instruction_v1_response": coding_runs["instruction_v1"][0][index],
            "coding_specialized_response": coding_runs["coding_specialized"][0][index],
        })
    args.coding_output.parent.mkdir(parents=True, exist_ok=True)
    args.coding_output.write_text(json.dumps({"results": comparison, "metrics": {label: run[1] for label, run in coding_runs.items()}, "validation": {label: checkpoint_loss(path) for label, path in paths.items()}}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    retention = []
    for index, prompt in enumerate(retention_prompts):
        retention.append({
            "category": prompt["category"], "prompt": prompt["prompt"],
            "instruction_v1_response": retention_runs["instruction_v1"][0][index],
            "coding_specialized_response": retention_runs["coding_specialized"][0][index],
        })
    args.retention_output.parent.mkdir(parents=True, exist_ok=True)
    args.retention_output.write_text(json.dumps({"results": retention, "metrics": {label: run[1] for label, run in retention_runs.items()}, "purpose": "Compare general response behavior before and after coding specialization; human review is required."}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {args.coding_output}")
    print(f"Saved: {args.retention_output}")
    for label, run in coding_runs.items():
        print(f"{label}: {run[1] if run[1] is not None else 'checkpoint unavailable'}")


if __name__ == "__main__":
    main()
