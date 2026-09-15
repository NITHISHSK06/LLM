from tokenizer.tokenizer import SimpleBPETokenizer, _prepare_training_text, train_tokenizer


def test_prepare_training_text_samples_large_inputs():
    long_text = ("hello world " * 50_000) + "final tokens"
    sampled = _prepare_training_text(long_text, max_chars=2000)
    assert len(sampled) <= 2000
    assert sampled.endswith("final tokens")


def test_tokenizer_round_trip_and_special_tokens(tmp_path):
    tokenizer = train_tokenizer("Hello, world! Hello again.", vocab_size=64)
    ids = tokenizer.encode("Hello, world!", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "Hello, world!"
    tokenizer.save(tmp_path)
    loaded = SimpleBPETokenizer.load(tmp_path)
    assert loaded.decode(loaded.encode("Hello, world!")) == "Hello, world!"


def test_unknown_text_does_not_crash():
    tokenizer = train_tokenizer("abc", vocab_size=16)
    ids = tokenizer.encode("🙂")
    assert isinstance(ids, list)
    assert tokenizer.decode(ids, skip_special_tokens=False) == "<UNK>"
