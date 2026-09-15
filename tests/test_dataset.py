import numpy as np

from training.dataset import TokenSequenceDataset, encode_split
from tokenizer.tokenizer import train_tokenizer


def test_dataset_alignment_and_shapes(tmp_path):
    path = tmp_path / "tokens.bin"
    np.asarray([0, 1, 2, 3, 4, 5], dtype=np.uint16).tofile(path)
    with TokenSequenceDataset(path, context_length=3) as dataset:
        inputs, targets = dataset.get_batch(1, rng=np.random.default_rng(1))
        assert inputs.shape == (1, 3)
        assert targets.shape == (1, 3)
        assert np.array_equal(targets, inputs + 1)


def test_encode_split_adds_sequence_markers():
    tokenizer = train_tokenizer("one\ntwo", vocab_size=32)
    tokens = encode_split(tokenizer, "one\ntwo")
    assert tokens[0] == tokenizer.bos_id
    assert tokenizer.eos_id in tokens
