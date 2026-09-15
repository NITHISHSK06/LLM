import torch

from model.attention import CausalSelfAttention
from model.config import ModelConfig


def tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=32, context_length=8, embedding_dim=16, num_layers=1, num_heads=4, ffn_dim=32, dropout=0.0)


def test_attention_shape_and_heads():
    attention = CausalSelfAttention(tiny_config()).eval()
    output = attention(torch.randn(2, 5, 16))
    assert output.shape == (2, 5, 16)
    assert attention.head_dim == 4
    assert attention.causal_mask.shape == (1, 1, 8, 8)


def test_attention_does_not_leak_future_tokens():
    attention = CausalSelfAttention(tiny_config()).eval()
    prefix = torch.randn(1, 3, 16)
    first = attention(torch.cat([prefix, torch.zeros(1, 2, 16)], dim=1))[:, :3]
    second = attention(torch.cat([prefix, torch.ones(1, 2, 16)], dim=1))[:, :3]
    torch.testing.assert_close(first, second)
    assert torch.all(torch.triu(attention.causal_mask[0, 0], diagonal=1) == 0)
