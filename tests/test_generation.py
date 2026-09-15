import pytest
import torch

from inference.generate import generate_tokens, load_model_and_tokenizer
from model.config import ModelConfig
from tokenizer.tokenizer import train_tokenizer


class ToyModel(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, eos_id: int | None = None):
        super().__init__()
        self.config = ModelConfig(vocab_size=vocab_size, context_length=context_length, embedding_dim=8, num_layers=1, num_heads=2, ffn_dim=16, dropout=0.0)
        self.eos_id = eos_id

    def forward(self, input_ids):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.config.vocab_size)
        token = self.eos_id if self.eos_id is not None else 1
        logits[:, -1, token] = 10.0
        return logits, None


def test_generation_respects_max_tokens_and_eos():
    tokenizer = train_tokenizer("hello world", vocab_size=32)
    model = ToyModel(len(tokenizer.vocab), 4, tokenizer.eos_id)
    ids = generate_tokens(model, tokenizer, "hello", 5, 0.0, 0, 1.0, torch.device("cpu"), greedy=True)
    assert ids[-1] == tokenizer.eos_id
    assert len(ids) <= len(tokenizer.encode("hello", add_bos=True)) + 1


def test_generation_context_is_truncated():
    tokenizer = train_tokenizer("hello world", vocab_size=32)
    model = ToyModel(len(tokenizer.vocab), 3)
    ids = generate_tokens(model, tokenizer, "hello world hello world", 2, 0.0, 0, 1.0, torch.device("cpu"), greedy=True)
    assert len(ids) == len(tokenizer.encode("hello world hello world", add_bos=True)) + 2


def test_invalid_checkpoint_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        load_model_and_tokenizer(tmp_path / "missing.pt", tmp_path / "tokenizer", torch.device("cpu"))
