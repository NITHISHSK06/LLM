"""Supervised instruction fine-tuning for the project's pretrained model."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

# Make direct execution from the project root resolve sibling packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = PROJECT_ROOT / "experiments" / "inference"

if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from generate import generate_response
from model.config import ModelConfig
from model.model import DecoderLanguageModel
from tokenizer.tokenizer import SimpleBPETokenizer
from training.experiments import record_experiment


DEFAULT_DATASET = Path("data/raw/instruction.jsonl")
DEFAULT_TRAIN_DATA = Path("data/processed/instruction_train.jsonl")
DEFAULT_VAL_DATA = Path("data/processed/instruction_val.jsonl")
DEFAULT_BASE_CHECKPOINT = Path("checkpoints/base/best_model.pt")
DEFAULT_OUTPUT_DIR = Path("checkpoints/instruct")
USER_PREFIX = "User: "
ASSISTANT_PREFIX = "\nAssistant: "
EVALUATION_PROMPTS = [
	"What is artificial intelligence?",
	"Explain machine learning in simple words.",
	"What is Python?",
	"Why is education important?",
	"What is deep learning?",
	"Explain neural networks simply.",
	"What is a computer?",
	"Write a short paragraph about technology.",
]


@dataclass(frozen=True)
class InstructionExample:
	prompt: str
	response: str


@dataclass
class EncodedExample:
	input_ids: list[int]
	target_ids: list[int]


def set_seed(seed: int) -> None:
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
	print("Warning: CPU instruction tuning will be much slower than CUDA training.")
	return torch.device("cpu")


def load_instruction_examples(path: Path, minimum_examples: int = 2) -> list[InstructionExample]:
	if not path.exists():
		raise FileNotFoundError(f"Instruction dataset not found: {path}")
	valid: list[InstructionExample] = []
	invalid_count = 0
	for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
		if not line.strip():
			continue
		try:
			record = json.loads(line)
			prompt = record.get("prompt") if isinstance(record, dict) else None
			response = record.get("response") if isinstance(record, dict) else None
			if not isinstance(prompt, str) or not isinstance(response, str):
				raise ValueError("prompt and response must be strings")
			prompt, response = prompt.strip(), response.strip()
			if not prompt or not response:
				raise ValueError("prompt and response must be non-empty")
			valid.append(InstructionExample(prompt, response))
		except (json.JSONDecodeError, ValueError, AttributeError) as error:
			invalid_count += 1
			print(f"Skipping invalid instruction line {line_number}: {error}")
	if len(valid) < minimum_examples:
		raise ValueError(f"At least {minimum_examples} valid instruction example(s) are required.")
	print(f"Loaded {len(valid)} valid examples; skipped {invalid_count} invalid examples.")
	return valid


def load_split_examples(path: Path) -> list[InstructionExample]:
	"""Load a prepared split without mixing it with the other split."""
	examples = load_instruction_examples(path, minimum_examples=1)
	if not examples:
		raise ValueError(f"Prepared split is empty: {path}")
	return examples


def split_examples(
	examples: list[InstructionExample],
	validation_fraction: float,
	seed: int,
) -> tuple[list[InstructionExample], list[InstructionExample]]:
	if not 0.0 < validation_fraction < 1.0:
		raise ValueError("validation_fraction must be between 0 and 1.")
	shuffled = examples.copy()
	random.Random(seed).shuffle(shuffled)
	validation_count = max(1, int(len(shuffled) * validation_fraction))
	validation_count = min(validation_count, len(shuffled) - 1)
	return shuffled[:-validation_count], shuffled[-validation_count:]


def encode_example(
	example: InstructionExample,
	tokenizer: SimpleBPETokenizer,
	context_length: int,
) -> EncodedExample:
	"""Create labels only for response tokens; prompt labels use -100."""
	prefix = f"{USER_PREFIX}{example.prompt}{ASSISTANT_PREFIX}"
	prefix_ids = tokenizer.encode(prefix, add_bos=True)
	response_ids = tokenizer.encode(example.response, add_eos=True)
	all_ids = prefix_ids + response_ids
	response_mask = [False] * len(prefix_ids) + [True] * len(response_ids)
	max_sequence_length = context_length + 1
	if len(all_ids) > max_sequence_length:
		all_ids = all_ids[-max_sequence_length:]
		response_mask = response_mask[-max_sequence_length:]
	input_ids = all_ids[:-1]
	target_ids = [
		token_id if is_response else -100
		for token_id, is_response in zip(all_ids[1:], response_mask[1:])
	]
	if not any(target_id != -100 for target_id in target_ids):
		raise ValueError("The response has no target tokens after truncation.")
	return EncodedExample(input_ids, target_ids)


def encode_examples(
	examples: list[InstructionExample],
	tokenizer: SimpleBPETokenizer,
	context_length: int,
) -> list[EncodedExample]:
	encoded: list[EncodedExample] = []
	for example in examples:
		try:
			encoded.append(encode_example(example, tokenizer, context_length))
		except ValueError as error:
			print(f"Skipping example during tokenization: {error}")
	if not encoded:
		raise ValueError("No instruction examples could be tokenized.")
	return encoded


def make_batch(
	examples: list[EncodedExample],
	batch_size: int,
	pad_id: int,
	rng: random.Random,
	device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
	chosen = [rng.choice(examples) for _ in range(batch_size)]
	sequence_length = max(len(example.input_ids) for example in chosen)
	input_batch = torch.full((batch_size, sequence_length), pad_id, dtype=torch.long)
	target_batch = torch.full((batch_size, sequence_length), -100, dtype=torch.long)
	for index, example in enumerate(chosen):
		length = len(example.input_ids)
		input_batch[index, :length] = torch.tensor(example.input_ids)
		target_batch[index, :length] = torch.tensor(example.target_ids)
	return input_batch.to(device), target_batch.to(device)


def evaluate_loss(
	model: DecoderLanguageModel,
	examples: list[EncodedExample],
	config: ModelConfig,
	tokenizer: SimpleBPETokenizer,
	device: torch.device,
	rng: random.Random,
) -> float:
	model.eval()
	losses: list[float] = []
	with torch.inference_mode():
		for _ in range(config.eval_iterations):
			inputs, targets = make_batch(examples, config.batch_size, tokenizer.pad_token_id, rng, device)
			_, loss = model(inputs, targets)
			assert loss is not None
			losses.append(loss.item())
	model.train()
	return sum(losses) / len(losses)


def save_checkpoint(
	path: Path,
	model: DecoderLanguageModel,
	optimizer: torch.optim.Optimizer,
	step: int,
	train_loss: float,
	validation_loss: float,
	best_validation_loss: float,
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"step": step,
			"train_loss": train_loss,
			"validation_loss": validation_loss,
			"best_validation_loss": best_validation_loss,
			"model_config": asdict(model.config),
		},
		path,
	)


def evaluate_checkpoints(
	base_checkpoint: Path,
	tuned_checkpoint: Path,
	tokenizer_dir: Path,
	max_new_tokens: int,
	device: torch.device,
) -> None:
	"""Generate the same small prompt set before and after fine-tuning."""
	for label, checkpoint_path in (
		("Before instruction tuning", base_checkpoint),
		("After instruction tuning", tuned_checkpoint),
	):
		model, tokenizer, _ = load_inference_model(checkpoint_path, tokenizer_dir, device)
		print(f"\n=== {label} ===")
		for prompt in EVALUATION_PROMPTS:
			response = generate_response(
				model, tokenizer, prompt, max_new_tokens,
				temperature=0.0, top_k=0, top_p=1.0,
				device=device, greedy=True,
			)
			print(f"Prompt: {prompt}\nResponse: {response}\n")


def load_inference_model(
	checkpoint_path: Path,
	tokenizer_dir: Path,
	device: torch.device,
) -> tuple[DecoderLanguageModel, SimpleBPETokenizer, torch.device]:
	"""Load a checkpoint for the optional before/after generation report."""
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint_path}")
	checkpoint = torch.load(checkpoint_path, map_location=device)
	config = ModelConfig(**checkpoint["model_config"])
	model = DecoderLanguageModel(config).to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()
	return model, SimpleBPETokenizer.load(tokenizer_dir), device


def load_pretrained_model(
	checkpoint_path: Path,
	device: torch.device,
) -> tuple[DecoderLanguageModel, dict]:
	if not checkpoint_path.exists():
		for candidate in (Path("checkpoints/best_model.pt"), Path("checkpoints/base/best_model.pt")):
			if candidate.exists():
				checkpoint_path = candidate
				break
	if not checkpoint_path.exists():
		raise FileNotFoundError(
			f"Pretrained checkpoint not found: {checkpoint_path}. "
			"Complete Phase 5 pretraining first."
		)
	checkpoint = torch.load(checkpoint_path, map_location=device)
	checkpoint_config = checkpoint.get("model_config")
	if not isinstance(checkpoint_config, dict):
		raise ValueError("The pretrained checkpoint does not contain model_config.")
	config = ModelConfig(**checkpoint_config)
	model = DecoderLanguageModel(config).to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	return model, checkpoint


def run_sanity_test(
	model: DecoderLanguageModel,
	train_examples: list[EncodedExample],
	validation_examples: list[EncodedExample],
	config: ModelConfig,
	tokenizer: SimpleBPETokenizer,
	device: torch.device,
	rng: random.Random,
) -> float:
	was_training = model.training
	model.eval()
	with torch.no_grad():
		inputs, targets = make_batch(train_examples, config.batch_size, tokenizer.pad_token_id, rng, device)
		logits, loss = model(inputs, targets)
	model.train(was_training)
	assert loss is not None
	print(f"Training examples:   {len(train_examples)}")
	print(f"Validation examples: {len(validation_examples)}")
	print(f"Input shape:         {tuple(inputs.shape)}")
	print(f"Target shape:        {tuple(targets.shape)}")
	print(f"Initial loss:        {loss.item():.4f}")
	print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
	return loss.item()


def train(args: argparse.Namespace) -> None:
	set_seed(args.seed)
	device = select_device()
	model, _ = load_pretrained_model(args.base_checkpoint, device)
	config = model.config
	tokenizer = SimpleBPETokenizer.load(args.tokenizer_dir)
	learning_rate = (
		config.instruction_learning_rate
		if args.learning_rate is None
		else args.learning_rate
	)
	if args.train_data is not None or args.val_data is not None:
		if args.train_data is None or args.val_data is None:
			raise ValueError("--train_data and --val_data must be provided together.")
		train_raw = load_split_examples(args.train_data)
		validation_raw = load_split_examples(args.val_data)
	else:
		examples = load_instruction_examples(args.dataset)
		train_raw, validation_raw = split_examples(examples, args.validation_fraction, args.seed)
	train_examples = encode_examples(train_raw, tokenizer, config.context_length)
	validation_examples = encode_examples(validation_raw, tokenizer, config.context_length)
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=learning_rate,
		weight_decay=args.weight_decay,
	)
	start_step = 0
	best_validation_loss = math.inf
	last_validation_loss = math.inf
	steps_without_improvement = 0
	last_train_loss = math.inf
	rng = random.Random(args.seed)
	if args.resume:
		checkpoint = torch.load(args.resume, map_location=device)
		model.load_state_dict(checkpoint["model_state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		start_step = int(checkpoint["step"])
		best_validation_loss = float(checkpoint.get("best_validation_loss", math.inf))
		last_validation_loss = float(checkpoint.get("validation_loss", math.inf))
		print(f"Resumed from: {args.resume} at step {start_step}")
	else:
		run_sanity_test(
			model, train_examples, validation_examples,
			config, tokenizer, device, rng,
		)

	for step in range(start_step + 1, args.max_iterations + 1):
		optimizer.zero_grad(set_to_none=True)
		inputs, targets = make_batch(train_examples, args.batch_size, tokenizer.pad_token_id, rng, device)
		_, loss = model(inputs, targets)
		assert loss is not None
		if not torch.isfinite(loss):
			raise FloatingPointError(f"Non-finite training loss at step {step}: {loss.item()}")
		last_train_loss = loss.item()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
		optimizer.step()

		if step % args.eval_interval == 0 or step == args.max_iterations:
			last_validation_loss = evaluate_loss(
				model, validation_examples, config, tokenizer, device, rng,
			)
			perplexity = math.exp(min(last_validation_loss, 20.0))
			print(
				f"Step: {step} | Train Loss: {loss.item():.4f} | "
				f"Validation Loss: {last_validation_loss:.4f} | Perplexity: {perplexity:.2f}"
			)
			if last_validation_loss < best_validation_loss:
				best_validation_loss = last_validation_loss
				steps_without_improvement = 0
				save_checkpoint(
					args.output_dir / "best_model.pt", model, optimizer, step,
					loss.item(), last_validation_loss, best_validation_loss,
				)
				print(f"Saved best model: {args.output_dir / 'best_model.pt'}")
			else:
				steps_without_improvement += 1
				if steps_without_improvement >= args.patience:
					print(f"Early stopping at step {step}: validation loss did not improve for {args.patience} evaluations.")
					break

		if step % args.checkpoint_interval == 0 or step == args.max_iterations:
			save_checkpoint(
				args.output_dir / f"checkpoint_{step:06d}.pt", model, optimizer, step,
				loss.item(), last_validation_loss, best_validation_loss,
			)
			save_checkpoint(
				args.output_dir / "latest.pt", model, optimizer, step,
				loss.item(), last_validation_loss, best_validation_loss,
			)

	args.experiment_output.parent.mkdir(parents=True, exist_ok=True)
	experiment = {
		"checkpoint": str(args.base_checkpoint),
		"dataset_version": "prepared_instruction_data_v1",
		"train_data": str(args.train_data) if args.train_data else str(args.dataset),
		"validation_data": str(args.val_data) if args.val_data else "seeded split from dataset",
		"output_dir": str(args.output_dir),
		"learning_rate": learning_rate,
		"batch_size": args.batch_size,
		"max_iterations": args.max_iterations,
		"eval_interval": args.eval_interval,
		"patience": args.patience,
		"seed": args.seed,
		"best_validation_loss": best_validation_loss,
		"final_step": step,
	}
	args.experiment_output.write_text(json.dumps(experiment, indent=2) + "\n", encoding="utf-8")
	record_experiment(
		{
			"dataset_name": str(args.train_data or args.dataset),
			"dataset_size": sum(path.stat().st_size for path in (args.train_data, args.val_data) if path is not None and path.exists()),
			"model_parameter_count": model.parameter_count(),
			"vocabulary_size": config.vocab_size,
			"context_length": config.context_length,
			"num_layers": config.num_layers,
			"num_heads": config.num_heads,
			"embedding_dim": config.embedding_dim,
			"ffn_dim": config.ffn_dim,
			"batch_size": args.batch_size,
			"learning_rate": learning_rate,
			"steps": args.max_iterations,
			"training_loss": last_train_loss,
			"validation_loss": last_validation_loss,
			"perplexity": math.exp(min(last_validation_loss, 20.0)) if math.isfinite(last_validation_loss) else None,
			"checkpoint_path": str(args.output_dir / "best_model.pt"),
			"notes": "response-only instruction tuning",
		}
	)

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Instruction-tune the pretrained Transformer.")
	parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
	parser.add_argument("--checkpoint", "--base-checkpoint", dest="base_checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
	parser.add_argument("--train_data", "--train-data", dest="train_data", type=Path, default=DEFAULT_TRAIN_DATA)
	parser.add_argument("--val_data", "--val-data", dest="val_data", type=Path, default=DEFAULT_VAL_DATA)
	parser.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizer"))
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--experiment-output", type=Path, default=Path("evaluation/experiments.json"))
	parser.add_argument("--resume", type=Path, default=None)
	parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=8)
	parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=None)
	parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=0.01)
	parser.add_argument("--max_iterations", "--max-iterations", dest="max_iterations", type=int, default=1_000)
	parser.add_argument("--eval_interval", "--eval-interval", dest="eval_interval", type=int, default=100)
	parser.add_argument("--checkpoint_interval", "--checkpoint-interval", dest="checkpoint_interval", type=int, default=500)
	parser.add_argument("--eval_iterations", "--eval-iterations", dest="eval_iterations", type=int, default=10)
	parser.add_argument("--gradient_clip", "--gradient-clip", dest="gradient_clip", type=float, default=1.0)
	parser.add_argument("--validation_fraction", "--validation-fraction", dest="validation_fraction", type=float, default=0.1)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--evaluate", action="store_true")
	parser.add_argument("--evaluation-max-new-tokens", type=int, default=60)
	parser.add_argument("--patience", type=int, default=5)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.evaluate:
		print("Evaluation-only mode: no training will be performed.")
		device = select_device()
		evaluate_checkpoints(
			args.base_checkpoint,
			args.output_dir / "best_model.pt",
			args.tokenizer_dir,
			max_new_tokens=args.evaluation_max_new_tokens,
			device=device,
		)
		return
	train(args)


if __name__ == "__main__":
	main()
