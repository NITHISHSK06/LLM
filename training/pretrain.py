"""Pretrain the decoder-only Transformer with next-token prediction.

Run from the project root:

    python training/pretrain.py

This module intentionally uses the project's own model and binary dataset. It
does not download or load pretrained weights.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# Make direct execution from the project root resolve sibling packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from model.config import ModelConfig
from model.model import DecoderLanguageModel
from training.dataset import TokenSequenceDataset
from training.experiments import record_experiment


DEFAULT_TRAIN_PATH = Path("data/processed/train.bin")
DEFAULT_VAL_PATH = Path("data/processed/val.bin")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/exp002_general_10m")


def set_seed(seed: int) -> None:
	"""Seed Python, NumPy, and PyTorch for repeatable experiments."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
	if torch.cuda.is_available():
		device = torch.device("cuda")
		print("Device: CUDA")
		print(f"GPU: {torch.cuda.get_device_name(device)}")
		return device
	print("Device: CPU")
	print("Warning: CPU training will be much slower than CUDA training.")
	return torch.device("cpu")


def vocab_size_dtype(vocab_size: int) -> np.dtype:
	"""Use a binary dtype that can represent every configured token ID."""
	return np.dtype(np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32)


def move_batch(
	dataset: TokenSequenceDataset,
	batch_size: int,
	context_length: int,
	device: torch.device,
	rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Load context+1 tokens, then split them into inputs and next-token targets."""
	sequences, _ = dataset.get_batch(batch_size, rng=rng)
	sequences = torch.from_numpy(sequences).to(device=device, dtype=torch.long)
	return sequences[:, :-1], sequences[:, 1:]


@torch.no_grad()
def estimate_validation_loss(
	model: DecoderLanguageModel,
	dataset: TokenSequenceDataset,
	config: ModelConfig,
	device: torch.device,
	rng: np.random.Generator,
	use_amp: bool,
) -> float:
	"""Average loss over fresh validation batches without creating gradients."""
	model.eval()
	losses: list[float] = []
	for _ in range(config.eval_iterations):
		inputs, targets = move_batch(dataset, config.batch_size, config.context_length, device, rng)
		with torch.amp.autocast("cuda", enabled=use_amp):
			_, loss = model(inputs, targets)
		assert loss is not None
		if not torch.isfinite(loss):
			raise FloatingPointError(f"Non-finite validation loss: {loss.item()}")
		losses.append(loss.item())
	model.train()
	return sum(losses) / len(losses)


def save_checkpoint(
	path: Path,
	model: DecoderLanguageModel,
	optimizer: torch.optim.Optimizer,
	scheduler: torch.optim.lr_scheduler.LRScheduler,
	scaler: torch.amp.GradScaler,
	iteration: int,
	train_loss: float,
	validation_loss: float,
	best_validation_loss: float,
) -> None:
	"""Save all state needed to continue training."""
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"scheduler_state_dict": scheduler.state_dict(),
			"scaler_state_dict": scaler.state_dict(),
			"iteration": iteration,
			"train_loss": train_loss,
			"validation_loss": validation_loss,
			"best_validation_loss": best_validation_loss,
			"model_config": asdict(model.config),
		},
		path,
	)


def load_checkpoint(
	path: Path,
	model: DecoderLanguageModel,
	optimizer: torch.optim.Optimizer,
	scheduler: torch.optim.lr_scheduler.LRScheduler,
	scaler: torch.amp.GradScaler,
	device: torch.device,
) -> tuple[int, float, float]:
	if not path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {path}")
	checkpoint = torch.load(path, map_location=device)
	model.load_state_dict(checkpoint["model_state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	if "scheduler_state_dict" in checkpoint:
		scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
	if "scaler_state_dict" in checkpoint:
		scaler.load_state_dict(checkpoint["scaler_state_dict"])
	print(f"Resumed from: {path} at step {checkpoint['iteration']}")
	return (
		int(checkpoint["iteration"]),
		float(checkpoint.get("best_validation_loss", math.inf)),
		float(checkpoint.get("validation_loss", math.inf)),
	)


@torch.no_grad()
def run_sanity_test(
	model: DecoderLanguageModel,
	dataset: TokenSequenceDataset,
	config: ModelConfig,
	device: torch.device,
	rng: np.random.Generator,
	use_amp: bool,
) -> float:
	"""Verify one batch can execute without modifying model weights."""
	model.eval()

	inputs, targets = move_batch(
		dataset,
		config.batch_size,
		config.context_length,
		device,
		rng,
	)

	with torch.amp.autocast("cuda", enabled=use_amp):
		logits, loss = model(inputs, targets)

	assert loss is not None

	if not torch.isfinite(loss):
		raise FloatingPointError(
			f"Non-finite sanity-test loss: {loss.item()}"
		)

	print(f"Input shape:  {tuple(inputs.shape)}")
	print(f"Target shape: {tuple(targets.shape)}")
	print(f"Logits shape: {tuple(logits.shape)}")
	print(f"Initial loss:  {loss.item():.4f}")
	print(f"Parameter count: {model.parameter_count():,}")
	print(f"Device: {device.type.upper()}")
	print("Sanity test passed. No optimizer update performed.")

	model.train()
	return loss.item()


def train(
	config: ModelConfig,
	train_path: Path,
	validation_path: Path,
	checkpoint_dir: Path,
	resume_path: Path | None = None,
) -> None:
	for path in (train_path, validation_path):
		if not path.exists() or path.stat().st_size == 0:
			raise ValueError(
				f"Processed dataset is missing or empty: {path}. "
				"Run Phase 3 preprocessing first."
			)

	set_seed(config.seed)
	device = select_device()
	use_amp = device.type == "cuda"
	if use_amp:
		print("Mixed precision: enabled")

	model = DecoderLanguageModel(config).to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=config.learning_rate,
		weight_decay=config.weight_decay,
	)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer,
		T_max=config.max_iterations,
	)
	scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
	start_iteration = 0
	best_validation_loss = math.inf
	last_validation_loss = math.inf
	last_train_loss = math.inf

	with TokenSequenceDataset(train_path, config.context_length + 1, vocab_size_dtype(config.vocab_size)) as train_dataset, TokenSequenceDataset(
		validation_path, config.context_length + 1, vocab_size_dtype(config.vocab_size)
	) as validation_dataset:
		rng = np.random.default_rng(config.seed)
		if resume_path is None:
			run_sanity_test(model, train_dataset, config, device, rng, use_amp)
		else:
			start_iteration, best_validation_loss, last_validation_loss = load_checkpoint(
				resume_path, model, optimizer, scheduler, scaler, device
			)

		for iteration in range(start_iteration + 1, config.max_iterations + 1):
			optimizer.zero_grad(set_to_none=True)
			inputs, targets = move_batch(train_dataset, config.batch_size, config.context_length, device, rng)
			with torch.amp.autocast("cuda", enabled=use_amp):
				_, loss = model(inputs, targets)
			assert loss is not None
			if not torch.isfinite(loss):
				raise FloatingPointError(f"Non-finite training loss at step {iteration}: {loss.item()}")
			last_train_loss = loss.item()
			if use_amp:
				scaler.scale(loss).backward()
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
				scaler.step(optimizer)
				scaler.update()
			else:
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
				optimizer.step()
			scheduler.step()

			if iteration % config.eval_interval == 0 or iteration == config.max_iterations:
				last_validation_loss = estimate_validation_loss(
					model, validation_dataset, config, device, rng, use_amp
				)
				perplexity = math.exp(min(last_validation_loss, 20.0))
				print(
					f"Step: {iteration} | Train Loss: {loss.item():.4f} | "
					f"Validation Loss: {last_validation_loss:.4f} | "
					f"Perplexity: {perplexity:.2f}"
				)
				if last_validation_loss < best_validation_loss:
					best_validation_loss = last_validation_loss
					save_checkpoint(
						checkpoint_dir / "best_model.pt",
						model,
						optimizer,
						scheduler,
						scaler,
						iteration,
						loss.item(),
						last_validation_loss,
						best_validation_loss,
					)
					print(f"Saved best model: {checkpoint_dir / 'best_model.pt'}")

			if iteration % config.checkpoint_interval == 0 or iteration == config.max_iterations:
				save_checkpoint(
					checkpoint_dir / f"checkpoint_{iteration:06d}.pt",
					model,
					optimizer,
					scheduler,
					scaler,
					iteration,
					loss.item(),
					last_validation_loss,
					best_validation_loss,
				)
				save_checkpoint(
					checkpoint_dir / "latest.pt",
					model,
					optimizer,
					scheduler,
					scaler,
					iteration,
					loss.item(),
					last_validation_loss,
					best_validation_loss,
				)
				print(f"Saved checkpoint: {checkpoint_dir / f'checkpoint_{iteration:06d}.pt'}")

	record_experiment(
		{
			"dataset_name": str(train_path),
			"dataset_size": train_path.stat().st_size,
			"approximate_token_count": train_path.stat().st_size // np.dtype(vocab_size_dtype(config.vocab_size)).itemsize,
			"model_parameter_count": model.parameter_count(),
			"vocabulary_size": config.vocab_size,
			"context_length": config.context_length,
			"num_layers": config.num_layers,
			"num_heads": config.num_heads,
			"embedding_dim": config.embedding_dim,
			"ffn_dim": config.ffn_dim,
			"batch_size": config.batch_size,
			"learning_rate": config.learning_rate,
			"steps": config.max_iterations,
			"training_loss": float(last_train_loss),
			"validation_loss": last_validation_loss,
			"perplexity": math.exp(min(last_validation_loss, 20.0)) if math.isfinite(last_validation_loss) else None,
			"checkpoint_path": str(checkpoint_dir / "best_model.pt"),
			"notes": "decoder-only pretraining",
		}
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Pretrain the educational decoder-only Transformer.")
	parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
	parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
	parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
	parser.add_argument("--resume", type=Path, default=None)
	parser.add_argument("--batch-size", type=int, default=None)
	parser.add_argument("--learning-rate", type=float, default=None)
	parser.add_argument("--max-iterations", type=int, default=None)
	parser.add_argument("--eval-interval", type=int, default=None)
	parser.add_argument("--checkpoint-interval", type=int, default=None)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	config = ModelConfig()
	for name in (
		"batch_size",
		"learning_rate",
		"max_iterations",
		"eval_interval",
		"checkpoint_interval",
	):
		value = getattr(args, name, None)
		if value is not None:
			setattr(config, name, value)
	config.__post_init__()
	train(
		config,
		train_path=args.train_path,
		validation_path=args.val_path,
		checkpoint_dir=args.checkpoint_dir,
		resume_path=args.resume,
	)


if __name__ == "__main__":
	main()
