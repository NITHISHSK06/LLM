"""Configuration for the educational decoder-only Transformer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
	"""Model dimensions are centralized so scaling only needs config changes."""

	vocab_size: int = 10_000
	context_length: int = 256
	embedding_dim: int = 256
	num_layers: int = 6
	num_heads: int = 8
	ffn_dim: int = 1_024
	dropout: float = 0.1
	batch_size: int = 8
	learning_rate: float = 3e-4
	instruction_learning_rate: float = 2e-5
	weight_decay: float = 0.1
	max_iterations: int = 5_000
	eval_interval: int = 500
	eval_iterations: int = 20
	gradient_clip: float = 1.0
	checkpoint_interval: int = 1_000
	seed: int = 42

	@classmethod
	def exp012_capacity(cls) -> ModelConfig:
		"""Return the isolated Exp012 approximately 25.5M model configuration."""
		return cls(
			vocab_size=10_000,
			context_length=256,
			embedding_dim=384,
			num_layers=10,
			num_heads=8,
			ffn_dim=1_536,
			dropout=0.1,
			batch_size=8,
			learning_rate=3e-4,
			instruction_learning_rate=2e-5,
			weight_decay=0.1,
			max_iterations=30_000,
			eval_interval=500,
			eval_iterations=20,
			gradient_clip=1.0,
			checkpoint_interval=1_000,
			seed=42,
		)

	def validate(self) -> None:
		"""Validate dimensions and training values before constructing a model."""
		self.__post_init__()

	def __post_init__(self) -> None:
		if self.embedding_dim % self.num_heads != 0:
			raise ValueError("embedding_dim must be divisible by num_heads.")
		if min(
			self.vocab_size,
			self.context_length,
			self.embedding_dim,
			self.num_layers,
			self.num_heads,
			self.ffn_dim,
		) <= 0:
			raise ValueError("Model dimensions must be positive.")
		if not 0.0 <= self.dropout < 1.0:
			raise ValueError("dropout must be in the range [0, 1).")
		if self.batch_size <= 0 or self.max_iterations <= 0:
			raise ValueError("batch_size and max_iterations must be positive.")
		if self.learning_rate <= 0 or self.weight_decay < 0:
			raise ValueError("learning_rate must be positive and weight_decay non-negative.")
		if self.instruction_learning_rate <= 0:
			raise ValueError("instruction_learning_rate must be positive.")
		if min(self.eval_interval, self.eval_iterations, self.checkpoint_interval) <= 0:
			raise ValueError("Evaluation and checkpoint intervals must be positive.")
		if self.gradient_clip <= 0:
			raise ValueError("gradient_clip must be positive.")
