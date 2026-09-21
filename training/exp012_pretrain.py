"""Run Exp012 pretraining from random initialization on the Exp006 corpus."""

from __future__ import annotations

import argparse
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
	sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp012_capacity import (
	CHECKPOINT_DIR,
	PRETRAIN_TRAIN_PATH,
	PRETRAIN_VAL_PATH,
	make_config,
	write_model_metadata,
)
from model.model import DecoderLanguageModel
from training.pretrain import train


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
	parser = argparse.ArgumentParser(description="Pretrain Exp012 from random initialization.")
	parser.add_argument("--train-path", type=Path, default=PRETRAIN_TRAIN_PATH)
	parser.add_argument("--val-path", type=Path, default=PRETRAIN_VAL_PATH)
	parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
	parser.add_argument("--resume", type=Path, default=None)
	return parser.parse_args()


def resolve_exp006_paths(args: argparse.Namespace) -> None:
	"""Accept the old exp006 alias while preserving explicit custom paths."""
	if args.train_path.exists() and args.val_path.exists():
		return
	if args.train_path.parent.name == "exp006" and args.val_path.parent.name == "exp006":
		fallback_train = PROJECT_ROOT / "data/processed/exp006_mixed/train.bin"
		fallback_val = PROJECT_ROOT / "data/processed/exp006_mixed/val.bin"
		if fallback_train.exists() and fallback_val.exists():
			print(
				f"Exp006 alias directory not found; using existing mixed corpus: "
				f"{fallback_train.parent}"
			)
			args.train_path = fallback_train
			args.val_path = fallback_val


def main() -> None:
	args = parse_args()
	resolve_exp006_paths(args)
	if args.resume is not None and args.resume.parent.resolve() != args.checkpoint_dir.resolve():
		raise ValueError("Exp012 may only resume from its own checkpoint directory; old checkpoints are not compatible.")
	config = make_config()
	model = DecoderLanguageModel(config)
	write_model_metadata(args.checkpoint_dir, model)
	args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
	log_path = args.checkpoint_dir / "training.log"
	with log_path.open("a", encoding="utf-8") as log_handle:
		tee = Tee(sys.stdout, log_handle)
		with redirect_stdout(tee):
			print("Exp012 pretraining: random initialization; no Exp006 checkpoint loaded.")
			print(f"Train data: {args.train_path}")
			print(f"Validation data: {args.val_path}")
			print(f"Configuration: {config}")
			train(
				config,
				train_path=args.train_path,
				validation_path=args.val_path,
				checkpoint_dir=args.checkpoint_dir,
				resume_path=args.resume,
			)
		latest_path = args.checkpoint_dir / "latest.pt"
		if latest_path.exists():
			shutil.copy2(latest_path, args.checkpoint_dir / "latest_model.pt")
			print(f"Saved: {args.checkpoint_dir / 'latest_model.pt'}")


if __name__ == "__main__":
	main()