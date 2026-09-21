"""Instruction-tune Exp012 using exactly the Exp011 prepared splits."""

from __future__ import annotations

import sys
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp012_capacity import (
	INSTRUCTION_TRAIN_PATH,
	INSTRUCTION_VAL_PATH,
	SFT_CHECKPOINT_DIR,
	TOKENIZER_DIR,
	make_config,
	write_model_metadata,
)
from model.model import DecoderLanguageModel
from training.instruct_train import train


class Tee:
	def __init__(self, *streams) -> None:
		self.streams = streams

	def write(self, text: str) -> int:
		for stream in self.streams:
			stream.write(text)
			stream.flush()
		return len(text)

	def flush(self) -> None:
		for stream in self.streams:
			stream.flush()


def main() -> None:
	config = make_config()
	model = DecoderLanguageModel(config)
	write_model_metadata(SFT_CHECKPOINT_DIR, model)
	SFT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
	log_path = SFT_CHECKPOINT_DIR / "training.log"
	base_checkpoint = PROJECT_ROOT / "checkpoints/exp012_capacity_25m/best_model.pt"
	if not base_checkpoint.exists():
		raise FileNotFoundError(f"Exp012 pretraining checkpoint not found: {base_checkpoint}")

	args = type("Exp012InstructionArgs", (), {
		"seed": 42,
		"base_checkpoint": base_checkpoint,
		"tokenizer_dir": TOKENIZER_DIR,
		"train_data": INSTRUCTION_TRAIN_PATH,
		"val_data": INSTRUCTION_VAL_PATH,
		"dataset": INSTRUCTION_TRAIN_PATH,
		"learning_rate": 2e-5,
		"weight_decay": 0.01,
		"batch_size": 8,
		"max_iterations": 1_000,
		"eval_interval": 100,
		"checkpoint_interval": 500,
		"eval_iterations": 10,
		"gradient_clip": 1.0,
		"patience": 7,
		"validation_fraction": 0.1,
		"output_dir": SFT_CHECKPOINT_DIR,
		"experiment_output": PROJECT_ROOT / "evaluation/exp012_instruction_training.json",
		"resume": None,
	})()

	with log_path.open("a", encoding="utf-8") as log_handle:
		tee = Tee(sys.stdout, log_handle)
		with redirect_stdout(tee):
			print("Exp012 instruction tuning: exact Exp011 train/validation files.")
			print(f"Train data: {INSTRUCTION_TRAIN_PATH}")
			print(f"Validation data: {INSTRUCTION_VAL_PATH}")
			print(f"Tokenizer: {TOKENIZER_DIR}")
			train(args)


if __name__ == "__main__":
	main()