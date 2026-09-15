import torch

from model.config import ModelConfig
from model.model import DecoderLanguageModel


def tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=40, context_length=8, embedding_dim=16, num_layers=2, num_heads=4, ffn_dim=32, dropout=0.0)


def test_logits_loss_and_backpropagation():
    model = DecoderLanguageModel(tiny_config())
    inputs = torch.randint(0, 40, (2, 6))
    targets = torch.randint(0, 40, (2, 6))
    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 6, 40)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_default_parameter_count_is_reported_at_target_scale():
    model = DecoderLanguageModel(ModelConfig())
    assert model.parameter_count() == 9_924_608
