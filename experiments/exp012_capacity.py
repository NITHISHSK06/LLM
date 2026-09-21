"""Shared Exp012 configuration and parameter-report helpers."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	 sys.path.insert(0, str(PROJECT_ROOT))

from model.config import ModelConfig
from model.model import DecoderLanguageModel

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints/exp012_capacity_25m"
SFT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints/exp012_capacity_25m_sft"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer_exp003"
PRETRAIN_TRAIN_PATH = PROJECT_ROOT / "data/processed/exp006_mixed/train.bin"
PRETRAIN_VAL_PATH = PROJECT_ROOT / "data/processed/exp006_mixed/val.bin"
INSTRUCTION_TRAIN_PATH = PROJECT_ROOT / "data/processed/exp011_general_chat/instruction_train.jsonl"
INSTRUCTION_VAL_PATH = PROJECT_ROOT / "data/processed/exp011_general_chat/instruction_val.jsonl"


def make_config() -> ModelConfig:
	return ModelConfig.exp012_capacity()


def parameter_report(model: DecoderLanguageModel) -> dict[str, object]:
	breakdown = model.parameter_breakdown()
	total = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
	return {
		"experiment": "Exp012 capacity scaling",
		"model_config": asdict(model.config),
		"total_trainable_parameters": total,
		"parameter_count_millions": total / 1_000_000,
		"components": breakdown,
		"weight_tying": False,
	}


def write_model_metadata(output_dir: Path, model: DecoderLanguageModel) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	report = parameter_report(model)
	(output_dir / "model_config.json").write_text(
		json.dumps(report["model_config"], indent=2) + "\n", encoding="utf-8"
	)
	(output_dir / "parameter_count_report.json").write_text(
		json.dumps(report, indent=2) + "\n", encoding="utf-8"
	)


def print_parameter_report(model: DecoderLanguageModel) -> None:
	report = parameter_report(model)
	print("Exp012 parameter verification")
	print(f"Total trainable parameters: {report['total_trainable_parameters']:,}")
	print(f"Parameter count (millions): {report['parameter_count_millions']:.6f}")
	for name, count in report["components"].items():
		print(f"{name}: {count:,}")