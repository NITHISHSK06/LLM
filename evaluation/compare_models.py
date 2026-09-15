"""Compare base, original instruction-tuned, and improved checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.generate import format_instruction_prompt, generate_tokens, load_model_and_tokenizer, select_device

DEFAULT_PROMPTS = Path("evaluation/test_prompts.json")
DEFAULT_OUTPUT = Path("evaluation/model_comparison.json")
DEFAULT_REPORT = Path("evaluation/improvement_report.md")


def generate_response(model: Any, tokenizer: Any, device: torch.device, prompt: str, max_new_tokens: int) -> str:
    formatted = format_instruction_prompt(prompt)
    prompt_ids = tokenizer.encode(formatted, add_bos=True)
    all_ids = generate_tokens(
        model, tokenizer, formatted, max_new_tokens, 0.0, 0, 1.0, device, greedy=True
    )
    return tokenizer.decode(all_ids[len(prompt_ids):]).strip()


def load_prompts(path: Path) -> list[str]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [record["prompt"] for record in records]


def checkpoint_stats(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"validation_loss": None, "perplexity": None}
    checkpoint = torch.load(path, map_location="cpu")
    loss = checkpoint.get("validation_loss")
    loss = float(loss) if loss is not None else None
    return {
        "validation_loss": loss,
        "perplexity": math.exp(min(loss, 20.0)) if loss is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare three local model checkpoints.")
    parser.add_argument("--base", type=Path, default=Path("checkpoints/base/best_model.pt"))
    parser.add_argument("--v1", type=Path, default=Path("checkpoints/instruct/best_model.pt"))
    parser.add_argument("--v2", type=Path, default=Path("checkpoints/instruct_v2/best_model.pt"))
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizer"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()
    prompts = load_prompts(args.prompts)
    device = select_device()
    checkpoint_paths = {"base": args.base, "instruction_v1": args.v1, "instruction_v2": args.v2}
    loaded: dict[str, tuple[Any, Any]] = {}
    for label, path in checkpoint_paths.items():
        if path.exists():
            model, tokenizer, _ = load_model_and_tokenizer(path, args.tokenizer_dir, device)
            loaded[label] = (model, tokenizer)
        else:
            print(f"Checkpoint unavailable: {path}")

    comparison: list[dict[str, str]] = []
    for prompt in prompts:
        record = {
            "prompt": prompt,
            "base_response": "",
            "instruction_v1_response": "",
            "instruction_v2_response": "",
        }
        for label, (model, tokenizer) in loaded.items():
            record[f"{label}_response"] = generate_response(model, tokenizer, device, prompt, args.max_new_tokens)
        comparison.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    stats = {label: checkpoint_stats(path) for label, path in checkpoint_paths.items()}
    lines = [
        "# Model Improvement Report",
        "",
        "This report compares the same evaluation prompts across local checkpoints.",
        "Automatic metrics do not establish that one model is more intelligent or grammatical.",
        "",
        "## Dataset",
        "",
        "See `evaluation/data_quality_report.json` and `evaluation/response_quality_report.json` for rule-based data checks.",
        "",
        "## Checkpoint Metrics",
        "",
        "| Model | Checkpoint | Validation loss | Perplexity |",
        "|---|---|---:|---:|",
    ]
    for label, path in checkpoint_paths.items():
        item = stats[label]
        loss = "unavailable" if item["validation_loss"] is None else f"{item['validation_loss']:.4f}"
        perplexity = "unavailable" if item["perplexity"] is None else f"{item['perplexity']:.4f}"
        lines.append(f"| {label} | `{path}` | {loss} | {perplexity} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Review `model_comparison.json` and complete `evaluation/human_eval.md` before claiming improvement.",
        "Compare relevance, instruction following, grammar, coherence, repetition, and response length, not validation loss alone.",
        "",
        "## Sample Responses",
        "",
    ])
    for record in comparison[:8]:
        lines.extend([
            f"### {record['prompt']}",
            "",
            f"**Base:** {record['base_response']}",
            "",
            f"**Instruction V1:** {record['instruction_v1_response']}",
            "",
            f"**Instruction V2:** {record['instruction_v2_response']}",
            "",
        ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved comparison: {args.output}")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
