"""Generate text from the project's trained decoder-only Transformer.

Examples from the project root:

    python inference/generate.py --prompt "Artificial intelligence is"
    python inference/generate.py --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make direct execution resolve project-level packages.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import ModelConfig
from model.model import DecoderLanguageModel
from tokenizer.tokenizer import SimpleBPETokenizer


DEFAULT_TOKENIZER_DIR = Path("tokenizer_exp003")
DEFAULT_CHECKPOINT = Path(
    "checkpoints/exp003_books_wikipedia_10m/best_model.pt")
TEST_PROMPTS = [
	"Artificial intelligence is",
	"Machine learning is",
	"The future of technology",
	"Python is a programming language",
]
USER_PREFIX = "User: "
ASSISTANT_PREFIX = "\nAssistant: "


def format_instruction_prompt(prompt: str) -> str:
	return f"{USER_PREFIX}{prompt}{ASSISTANT_PREFIX}"


def select_device() -> torch.device:
	if torch.cuda.is_available():
		device = torch.device("cuda")
		print("Device: CUDA")
		print(f"GPU: {torch.cuda.get_device_name(device)}")
		return device
	print("Device: CPU")
	return torch.device("cpu")


def load_model_and_tokenizer(
	checkpoint_path: Path = DEFAULT_CHECKPOINT,
	tokenizer_dir: Path = DEFAULT_TOKENIZER_DIR,
	device: torch.device | None = None,
) -> tuple[DecoderLanguageModel, SimpleBPETokenizer, torch.device]:
	"""Load model dimensions from the checkpoint instead of guessing them."""
	if not checkpoint_path.exists():
		legacy_candidates = [
			Path("checkpoints/best_model.pt"),
			Path("checkpoints/instruct/best_model.pt"),
			Path("checkpoints/instruct_v2/best_model.pt"),
		]
		for candidate in legacy_candidates:
			if candidate.exists():
				checkpoint_path = candidate
				break
	if not checkpoint_path.exists():
		available = sorted(str(path) for path in Path("checkpoints").rglob("*.pt"))
		available_text = "\n".join(available) if available else "  (no .pt checkpoint files found)"
		raise FileNotFoundError(
			f"Best checkpoint not found: {checkpoint_path}\n"
			f"Available checkpoints in {checkpoint_path.parent}:\n{available_text}\n"
			"Train the model first or pass --checkpoint with a valid .pt file."
		)
	if not tokenizer_dir.exists():
		raise FileNotFoundError(
			f"Tokenizer directory not found: {tokenizer_dir}. "
			"Build the tokenizer during Phase 3 first."
		)

	device = device or select_device()
	checkpoint = torch.load(checkpoint_path, map_location=device)
	checkpoint_config = checkpoint.get("model_config")
	if not isinstance(checkpoint_config, dict):
		raise ValueError(f"Checkpoint {checkpoint_path} does not contain model_config.")
	config = ModelConfig(**checkpoint_config)
	model = DecoderLanguageModel(config).to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()
	tokenizer = SimpleBPETokenizer.load(tokenizer_dir)
	return model, tokenizer, device


def apply_top_k_top_p(
	logits: torch.Tensor,
	top_k: int,
	top_p: float,
) -> torch.Tensor:
	"""Remove unlikely tokens before sampling."""
	if top_k > 0:
		top_k = min(top_k, logits.shape[-1])
		threshold = torch.topk(logits, top_k).values[..., -1, None]
		logits = logits.masked_fill(logits < threshold, float("-inf"))

	if top_p < 1.0:
		sorted_logits, sorted_indices = torch.sort(logits, descending=True)
		sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
		cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
		remove = cumulative_probabilities > top_p
		remove[..., 1:] = remove[..., :-1].clone()
		remove[..., 0] = False
		indices_to_remove = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
		logits = logits.masked_fill(indices_to_remove, float("-inf"))
	return logits


@torch.inference_mode()
def generate_tokens(
	model: DecoderLanguageModel,
	tokenizer: SimpleBPETokenizer,
	prompt: str,
	max_new_tokens: int,
	temperature: float,
	top_k: int,
	top_p: float,
	device: torch.device,
	greedy: bool = False,
) -> list[int]:
	"""Generate token IDs while retaining only the model's context window."""
	if max_new_tokens < 0:
		raise ValueError("max_new_tokens must be non-negative.")
	if temperature < 0.0:
		raise ValueError("temperature must be non-negative.")
	if top_k < 0:
		raise ValueError("top_k must be non-negative.")
	if not 0.0 < top_p <= 1.0:
		raise ValueError("top_p must be greater than 0 and at most 1.")

	token_ids = tokenizer.encode(prompt, add_bos=True)
	if not token_ids:
		token_ids = [tokenizer.bos_token_id]

	for _ in range(max_new_tokens):
		context_ids = token_ids[-model.config.context_length :]
		input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
		logits, _ = model(input_ids)
		next_token_logits = logits[:, -1, :]

		if greedy or temperature == 0.0:
			next_token = torch.argmax(next_token_logits, dim=-1)
		else:
			next_token_logits = next_token_logits / temperature
			next_token_logits = apply_top_k_top_p(next_token_logits, top_k, top_p)
			probabilities = torch.softmax(next_token_logits, dim=-1)
			next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)

		next_token_id = int(next_token.item())
		token_ids.append(next_token_id)
		if next_token_id == tokenizer.eos_token_id:
			break
	return token_ids


def generate_text(
	model: DecoderLanguageModel,
	tokenizer: SimpleBPETokenizer,
	prompt: str,
	max_new_tokens: int,
	temperature: float,
	top_k: int,
	top_p: float,
	device: torch.device,
	greedy: bool = False,
) -> str:
	token_ids = generate_tokens(
		model,
		tokenizer,
		prompt,
		max_new_tokens,
		temperature,
		top_k,
		top_p,
		device,
		greedy,
	)
	return tokenizer.decode(token_ids)


def generate_response(
	model: DecoderLanguageModel,
	tokenizer: SimpleBPETokenizer,
	prompt: str,
	max_new_tokens: int,
	temperature: float,
	top_k: int,
	top_p: float,
	device: torch.device,
	greedy: bool = False,
) -> str:
	"""Generate only the assistant continuation for instruction-formatted input."""
	formatted_prompt = format_instruction_prompt(prompt)
	prompt_ids = tokenizer.encode(formatted_prompt, add_bos=True)
	all_token_ids = generate_tokens(
		model,
		tokenizer,
		formatted_prompt,
		max_new_tokens,
		temperature,
		top_k,
		top_p,
		device,
		greedy,
	)
	return tokenizer.decode(all_token_ids[len(prompt_ids) :])


def print_generation(
	model: DecoderLanguageModel,
	tokenizer: SimpleBPETokenizer,
	prompt: str,
	args: argparse.Namespace,
	device: torch.device,
) -> None:
	generated = generate_text(
		model,
		tokenizer,
		prompt,
		args.max_new_tokens,
		args.temperature,
		args.top_k,
		args.top_p,
		device,
		args.greedy,
	)
	print("Prompt:")
	print(prompt)
	print("\nGenerated:")
	print(generated)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate text with the trained Transformer.")
	parser.add_argument("--prompt", type=str, default=None)
	parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
	parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
	parser.add_argument("--max-new-tokens", type=int, default=100)
	parser.add_argument("--temperature", type=float, default=0.8)
	parser.add_argument("--top-k", type=int, default=50)
	parser.add_argument("--top-p", type=float, default=0.9)
	parser.add_argument("--greedy", action="store_true")
	parser.add_argument("--interactive", action="store_true")
	parser.add_argument("--test-prompts", action="store_true")
	parser.add_argument("--instruction-mode", action="store_true")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if not args.interactive and not args.test_prompts and args.prompt is None:
		raise SystemExit("Provide --prompt, --interactive, or --test-prompts.")

	model, tokenizer, device = load_model_and_tokenizer(args.checkpoint, args.tokenizer_dir)
	instruction_mode = args.instruction_mode or args.checkpoint.parent.name in {
		"instruct",
		"instruct_v2",
		"coding",
	}
	if args.interactive:
		print('Interactive mode. Type "exit" to stop.')
		while True:
			prompt = input("You: ").strip()
			if prompt.lower() in {"exit", "quit"}:
				break
			if prompt:
				if instruction_mode:
					response = generate_response(
						model, tokenizer, prompt, args.max_new_tokens,
						args.temperature, args.top_k, args.top_p, device, args.greedy,
					)
				else:
					response = generate_text(
						model, tokenizer, prompt, args.max_new_tokens,
						args.temperature, args.top_k, args.top_p, device, args.greedy,
					)
				print(f"Model: {response}")
		return

	if args.test_prompts:
		for prompt in TEST_PROMPTS:
			print_generation(model, tokenizer, prompt, args, device)
	else:
		if instruction_mode:
			generated = generate_response(
				model, tokenizer, args.prompt, args.max_new_tokens,
				args.temperature, args.top_k, args.top_p, device, args.greedy,
			)
			print("Prompt:")
			print(args.prompt)
			print("\nGenerated:")
			print(generated)
		else:
			print_generation(model, tokenizer, args.prompt, args, device)


if __name__ == "__main__":
	main()
