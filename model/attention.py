"""Causal multi-head self-attention implemented with basic PyTorch layers."""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
	"""Multi-head self-attention that masks every future position."""

	def __init__(self, config: ModelConfig) -> None:
		super().__init__()
		if config.embedding_dim % config.num_heads != 0:
			raise ValueError("embedding_dim must be divisible by num_heads.")

		self.num_heads = config.num_heads
		self.head_dim = config.embedding_dim // config.num_heads
		self.context_length = config.context_length
		self.query_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
		self.key_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
		self.value_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
		self.output_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
		self.attention_dropout = nn.Dropout(config.dropout)
		self.output_dropout = nn.Dropout(config.dropout)

		# True entries are legal attention connections; future positions are false.
		causal_mask = torch.tril(
			torch.ones(config.context_length, config.context_length, dtype=torch.bool)
		)
		self.register_buffer("causal_mask", causal_mask.view(1, 1, config.context_length, config.context_length))

	def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
		batch_size, sequence_length, _ = tensor.shape
		tensor = tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim)
		return tensor.transpose(1, 2)

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		batch_size, sequence_length, embedding_dim = hidden_states.shape
		if sequence_length > self.context_length:
			raise ValueError(
				f"Sequence length {sequence_length} exceeds context length "
				f"{self.context_length}."
			)

		query = self._split_heads(self.query_projection(hidden_states))
		key = self._split_heads(self.key_projection(hidden_states))
		value = self._split_heads(self.value_projection(hidden_states))

		# Attention scores are scaled to keep softmax gradients well behaved.
		scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
		mask = self.causal_mask[:, :, :sequence_length, :sequence_length]
		scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
		weights = torch.softmax(scores, dim=-1)
		weights = self.attention_dropout(weights)
		attended = torch.matmul(weights, value)

		attended = attended.transpose(1, 2).contiguous()
		attended = attended.view(batch_size, sequence_length, embedding_dim)
		return self.output_dropout(self.output_projection(attended))
