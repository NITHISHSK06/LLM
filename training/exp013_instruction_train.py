"""Exp013: DailyDialog SFT on the 25.52M-parameter Exp012 pretrained model."""

from __future__ import annotations

import argparse
import sys
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp012_capacity import make_config, write_model_metadata
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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train Exp013 SFT on the Exp012 pretrained model.")
	parser.add_argument("--base-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "exp012_capacity_25m" / "best_model.pt")
	parser.add_argument("--train_data", type=Path, default=PROJECT_ROOT / "data" / "raw" / "dailydialog_train.jsonl")
	parser.add_argument("--val_data", type=Path, default=PROJECT_ROOT / "data" / "raw" / "dailydialog_val.jsonl")
	parser.add_argument("--tokenizer-dir", type=Path, default=PROJECT_ROOT / "tokenizer_exp003")
	parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "exp013_dailydialog_25m")
	parser.add_argument("--max-iterations", type=int, default=5000)
	parser.add_argument("--eval-interval", type=int, default=100)
	parser.add_argument("--checkpoint-interval", type=int, default=500)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--learning-rate", type=float, default=2e-5)
	parser.add_argument("--weight-decay", type=float, default=0.01)
	parser.add_argument("--patience", type=int, default=7)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--eval-iterations", type=int, default=10)
	parser.add_argument("--gradient-clip", type=float, default=1.0)
	parser.add_argument("--validation-fraction", type=float, default=0.1)
	parser.add_argument("--resume", type=Path, default=None)
	return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
	missing = []
	for name, path in {
		"base checkpoint": args.base_checkpoint,
		"train data": args.train_data,
		"validation data": args.val_data,
		"tokenizer directory": args.tokenizer_dir,
	}.items():
		if not path.exists():
			missing.append(f"{name}: {path}")
	if missing:
		raise FileNotFoundError("Missing required Exp013 paths:\n" + "\n".join(missing))


def print_config(args: argparse.Namespace) -> None:
	print("Exp013 configuration")
	print(f"Base checkpoint: {args.base_checkpoint}")
	print(f"Train data: {args.train_data}")
	print(f"Validation data: {args.val_data}")
	print(f"Tokenizer: {args.tokenizer_dir}")
	print(f"Output directory: {args.output_dir}")
	print(f"Max iterations: {args.max_iterations}")
	print(f"Evaluation interval: {args.eval_interval}")
	print(f"Checkpoint interval: {args.checkpoint_interval}")
	print(f"Batch size: {args.batch_size}")
	print(f"Learning rate: {args.learning_rate}")
	print(f"Weight decay: {args.weight_decay}")
	print(f"Patience: {args.patience}")
	print(f"Seed: {args.seed}")
	print(f"Eval iterations: {args.eval_iterations}")
	print(f"Gradient clip: {args.gradient_clip}")
	print(f"Validation fraction: {args.validation_fraction}")
	print(f"Resume: {args.resume}")


def main() -> None:
	args = parse_args()
	args.output_dir.mkdir(parents=True, exist_ok=True)
	validate_paths(args)

	config = make_config()
	model = DecoderLanguageModel(config)
	write_model_metadata(args.output_dir, model)

	log_path = args.output_dir / "training.log"
	training_args = argparse.Namespace(
		seed=args.seed,
		base_checkpoint=args.base_checkpoint,
		tokenizer_dir=args.tokenizer_dir,
		train_data=args.train_data,
		val_data=args.val_data,
		dataset=args.train_data,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		batch_size=args.batch_size,
		max_iterations=args.max_iterations,
		eval_interval=args.eval_interval,
		checkpoint_interval=args.checkpoint_interval,
		eval_iterations=args.eval_iterations,
		gradient_clip=args.gradient_clip,
		patience=args.patience,
		validation_fraction=args.validation_fraction,
		output_dir=args.output_dir,
		experiment_output=args.output_dir / "experiment.json",
		resume=args.resume,
	)

	with log_path.open("a", encoding="utf-8") as log_handle:
		tee = Tee(sys.stdout, log_handle)
		with redirect_stdout(tee):
			print_config(args)
			print("Starting Exp013 instruction tuning.")
			train(training_args)


if __name__ == "__main__":
	main()
