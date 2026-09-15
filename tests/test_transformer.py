import torch

from model.config import ModelConfig
from model.transformer import TransformerBlock


def test_transformer_forward_and_layer_norm():
    config = ModelConfig(vocab_size=32, context_length=8, embedding_dim=16, num_layers=1, num_heads=4, ffn_dim=32, dropout=0.0)
    block = TransformerBlock(config).eval()
    hidden = torch.randn(2, 5, 16)
    normalized = block.attention_norm(hidden)
    assert normalized.shape == hidden.shape
    assert torch.allclose(normalized.mean(dim=-1), torch.zeros(2, 5), atol=1e-5)
    assert block(hidden).shape == hidden.shape


def test_residual_connections_preserve_input_when_sublayers_are_zero():
    config = ModelConfig(vocab_size=32, context_length=8, embedding_dim=16, num_layers=1, num_heads=4, ffn_dim=32, dropout=0.0)
    block = TransformerBlock(config).eval()
    for parameter in block.attention.parameters():
        parameter.data.zero_()
    for parameter in block.feed_forward.parameters():
        parameter.data.zero_()
    hidden = torch.randn(1, 4, 16)
    torch.testing.assert_close(block(hidden), hidden)
