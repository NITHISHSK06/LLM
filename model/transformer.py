"""Pre-layer-normalized decoder blocks for the language model."""

from __future__ import annotations

import torch
from torch import nn

from .attention import CausalSelfAttention
from .config import ModelConfig


class FeedForward(nn.Module):
	"""Position-wise MLP used after the attention sub-layer."""

	def __init__(self, config: ModelConfig) -> None:
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(config.embedding_dim, config.ffn_dim),
			nn.GELU(),
			nn.Linear(config.ffn_dim, config.embedding_dim),
			nn.Dropout(config.dropout),
		)

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		return self.network(hidden_states)


class TransformerBlock(nn.Module):
	"""One decoder block with pre-normalization and residual connections."""

	def __init__(self, config: ModelConfig) -> None:
		super().__init__()
		self.attention_norm = nn.LayerNorm(config.embedding_dim)
		self.attention = CausalSelfAttention(config)
		self.feed_forward_norm = nn.LayerNorm(config.embedding_dim)
		self.feed_forward = FeedForward(config)

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		hidden_states = hidden_states + self.attention(self.attention_norm(hidden_states))
		hidden_states = hidden_states + self.feed_forward(self.feed_forward_norm(hidden_states))
		return hidden_states
