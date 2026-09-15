"""Complete decoder-only Transformer language model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .transformer import TransformerBlock


class DecoderLanguageModel(nn.Module):
	"""Token/position embeddings followed by causal Transformer blocks."""

	def __init__(self, config: ModelConfig | None = None) -> None:
		super().__init__()
		self.config = config or ModelConfig()
		self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
		self.position_embedding = nn.Embedding(self.config.context_length, self.config.embedding_dim)
		self.blocks = nn.ModuleList(
			[TransformerBlock(self.config) for _ in range(self.config.num_layers)]
		)
		self.final_norm = nn.LayerNorm(self.config.embedding_dim)
		self.language_model_head = nn.Linear(
			self.config.embedding_dim,
			self.config.vocab_size,
			bias=False,
		)
		print(f"Total parameters: {self.parameter_count() / 1_000_000:.2f} million")

	def parameter_count(self) -> int:
		return sum(parameter.numel() for parameter in self.parameters())

	def forward(
		self,
		input_ids: torch.Tensor,
		targets: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor | None]:
		if input_ids.ndim != 2:
			raise ValueError("input_ids must have shape [batch_size, sequence_length].")
		batch_size, sequence_length = input_ids.shape
		if sequence_length > self.config.context_length:
			raise ValueError(
				f"Sequence length {sequence_length} exceeds context length "
				f"{self.config.context_length}."
			)
		if targets is not None and targets.shape != input_ids.shape:
			raise ValueError("targets must have the same shape as input_ids.")

		positions = torch.arange(sequence_length, device=input_ids.device)
		hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
		for block in self.blocks:
			hidden_states = block(hidden_states)
		logits = self.language_model_head(self.final_norm(hidden_states))

		loss = None
		if targets is not None:
			loss = F.cross_entropy(
				logits.reshape(batch_size * sequence_length, self.config.vocab_size),
				targets.reshape(batch_size * sequence_length),
			)
		return logits, loss


def main() -> None:
	"""Run a small architecture and forward-pass smoke test."""
	torch.manual_seed(42)
	model = DecoderLanguageModel()
	print(model)

	batch_size, sequence_length = 2, 16
	input_ids = torch.randint(0, model.config.vocab_size, (batch_size, sequence_length))
	target_ids = torch.randint(0, model.config.vocab_size, (batch_size, sequence_length))
	logits, loss = model(input_ids, target_ids)
	expected_shape = (batch_size, sequence_length, model.config.vocab_size)
	assert logits.shape == expected_shape, (logits.shape, expected_shape)
	assert loss is not None and torch.isfinite(loss)
	print(f"Input shape:  {tuple(input_ids.shape)}")
	print(f"Logits shape: {tuple(logits.shape)}")
	print(f"Loss:         {loss.item():.4f}")
	print("Forward-pass test passed.")


if __name__ == "__main__":
	main()
