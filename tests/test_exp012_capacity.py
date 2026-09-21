"""Smoke test for the Exp012 capacity-scaled model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp012_capacity import make_config, print_parameter_report
from model.model import DecoderLanguageModel


def main() -> None:
	torch.manual_seed(42)
	config = make_config()
	model = DecoderLanguageModel(config)
	input_ids = torch.randint(0, config.vocab_size, (2, 16))
	target_ids = torch.randint(0, config.vocab_size, (2, 16))
	logits, loss = model(input_ids, target_ids)

	assert logits.shape == (2, 16, 10_000), logits.shape
	assert loss is not None and torch.isfinite(loss), loss
	assert torch.isfinite(logits).all(), "Logits contain NaN or Inf."

	print(f"Input shape:  {tuple(input_ids.shape)}")
	print(f"Logits shape: {tuple(logits.shape)}")
	print(f"Loss:         {loss.item():.4f}")
	print_parameter_report(model)
	print("Exp012 smoke test passed.")


if __name__ == "__main__":
	main()