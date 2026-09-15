"""Prepare a plain-text corpus for next-token language-model training.

The tokenizer must be trained separately before preprocessing.

Example:

    python tokenizer/tokenizer.py build \
        --input data/raw/general_combined.txt \
        --output tokenizer_new \
        --vocab-size 10000

Then preprocess:

    python training/dataset.py preprocess \
        --input data/raw/general_combined.txt \
        --output-dir data/processed \
        --tokenizer-dir tokenizer_new \
        --vocab-size 10000 \
        --validation-fraction 0.10

The output files contain flat streams of token IDs.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from tokenizer.tokenizer import SimpleBPETokenizer


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT = Path("data/raw/general_combined.txt")
DEFAULT_OUTPUT = Path("data/processed")
DEFAULT_TOKENIZER = Path("tokenizer_new")

DEFAULT_VOCAB_SIZE = 10000
DEFAULT_VALIDATION_FRACTION = 0.10


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace and remove clearly corrupted lines."""

    cleaned_lines: list[str] = []

    for raw_line in text.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        # Unicode replacement character usually means corrupted text.
        replacement_count = line.count("�")

        # Remove control characters except tab.
        control_count = sum(
            1
            for character in line
            if ord(character) < 32 and character not in "\t"
        )

        # Check how much of the line is printable.
        printable_count = sum(
            character.isprintable()
            for character in line
        )

        if replacement_count > 0:
            continue

        if control_count > 0:
            continue

        if len(line) > 0:
            printable_ratio = printable_count / len(line)

            if printable_ratio < 0.85:
                continue

        # Normalize whitespace.
        line = re.sub(r"\s+", " ", line)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def split_text(
    text: str,
    validation_fraction: float,
) -> tuple[str, str]:
    """Split text into training and validation portions."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        raise ValueError(
            "The cleaned corpus needs at least two non-empty lines."
        )

    validation_count = max(
        1,
        int(len(lines) * validation_fraction),
    )

    if validation_count >= len(lines):
        validation_count = 1

    split_index = len(lines) - validation_count

    train_text = "\n".join(
        lines[:split_index]
    )

    validation_text = "\n".join(
        lines[split_index:]
    )

    return train_text, validation_text


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def encode_split(
    tokenizer: SimpleBPETokenizer,
    text: str,
    split_name: str = "dataset",
) -> np.ndarray:
    """Encode a text split using an already-trained tokenizer."""

    token_ids: list[int] = []

    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]

    total_lines = len(lines)

    if total_lines == 0:
        raise ValueError(
            f"{split_name} split contains no usable lines."
        )

    print(
        f"\nEncoding {split_name}: "
        f"{total_lines:,} lines"
    )

    start_time = time.time()

    for index, line in enumerate(lines, start=1):

        ids = tokenizer.encode(
            line,
            add_bos=True,
            add_eos=True,
        )

        token_ids.extend(ids)

        # Progress every 1000 lines.
        if index % 1000 == 0 or index == total_lines:

            elapsed = time.time() - start_time

            print(
                f"  {index:,}/{total_lines:,} lines | "
                f"tokens: {len(token_ids):,} | "
                f"time: {elapsed:.1f}s",
                end="\r",
                flush=True,
            )

    print()

    if not token_ids:
        raise ValueError(
            f"{split_name} split produced no tokens."
        )

    # uint16 is enough for vocab <= 65,535.
    if len(tokenizer.vocab) <= np.iinfo(
        np.uint16
    ).max:

        dtype = np.uint16

    else:

        dtype = np.uint32

    array = np.asarray(
        token_ids,
        dtype=dtype,
    )

    # Safety check.
    min_id = int(array.min())
    max_id = int(array.max())

    if min_id < 0:
        raise ValueError(
            f"{split_name} contains a negative token ID: {min_id}"
        )

    if max_id >= len(tokenizer.vocab):
        raise ValueError(
            f"{split_name} contains invalid token ID "
            f"{max_id}, but tokenizer vocabulary size is "
            f"{len(tokenizer.vocab)}."
        )

    print(
        f"{split_name} encoding complete: "
        f"{len(array):,} tokens"
    )

    return array


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess_text(
    text: str,
    output_dir: Path = DEFAULT_OUTPUT,
    tokenizer_dir: Path = DEFAULT_TOKENIZER,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> tuple[Path, Path]:
    """Clean, split, tokenize, and save the corpus."""

    # -----------------------------------------------------------------------
    # Load tokenizer
    # -----------------------------------------------------------------------

    print("=" * 70)
    print("LLM DATASET PREPROCESSING")
    print("=" * 70)

    print(
        f"\nLoading tokenizer from:\n"
        f"  {tokenizer_dir}"
    )

    if not tokenizer_dir.exists():
        raise FileNotFoundError(
            f"Tokenizer directory not found: {tokenizer_dir}\n"
            f"Build the tokenizer first."
        )

    tokenizer = SimpleBPETokenizer.load(
        tokenizer_dir
    )

    actual_vocab_size = len(tokenizer.vocab)

    print(
        f"Tokenizer vocabulary: "
        f"{actual_vocab_size:,}"
    )

    # -----------------------------------------------------------------------
    # Validate vocabulary
    # -----------------------------------------------------------------------

    if actual_vocab_size != vocab_size:

        raise ValueError(
            "\nTokenizer/model vocabulary mismatch!\n"
            f"Expected: {vocab_size:,}\n"
            f"Found:    {actual_vocab_size:,}\n\n"
            "Use the correct --tokenizer-dir or "
            "rebuild the tokenizer."
        )

    print("Vocabulary check: PASS")

    # -----------------------------------------------------------------------
    # Clean text
    # -----------------------------------------------------------------------

    print("\nCleaning corpus...")

    clean_start = time.time()

    cleaned = clean_text(text)

    clean_time = time.time() - clean_start

    if not cleaned:
        raise ValueError(
            "The input corpus is empty after cleaning."
        )

    print(
        f"Cleaned characters: "
        f"{len(cleaned):,}"
    )

    print(
        f"Cleaning time: "
        f"{clean_time:.2f}s"
    )

    # -----------------------------------------------------------------------
    # Split
    # -----------------------------------------------------------------------

    print(
        f"\nSplitting corpus "
        f"(validation = {validation_fraction:.1%})..."
    )

    train_text, validation_text = split_text(
        cleaned,
        validation_fraction,
    )

    print(
        f"Training characters:   "
        f"{len(train_text):,}"
    )

    print(
        f"Validation characters: "
        f"{len(validation_text):,}"
    )

    # -----------------------------------------------------------------------
    # Encode
    # -----------------------------------------------------------------------

    train_ids = encode_split(
        tokenizer,
        train_text,
        split_name="training",
    )

    validation_ids = encode_split(
        tokenizer,
        validation_text,
        split_name="validation",
    )

    # -----------------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = output_dir / "train.bin"
    validation_path = output_dir / "val.bin"

    # -----------------------------------------------------------------------
    # Save binary files
    # -----------------------------------------------------------------------

    print("\nSaving binary datasets...")

    train_ids.tofile(train_path)

    validation_ids.tofile(
        validation_path
    )

    # -----------------------------------------------------------------------
    # Final statistics
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Cleaned characters: {len(cleaned):,}"
    )

    print(
        f"Training tokens:    {len(train_ids):,}"
    )

    print(
        f"Validation tokens:  {len(validation_ids):,}"
    )

    print(
        f"Total tokens:       "
        f"{len(train_ids) + len(validation_ids):,}"
    )

    print(
        f"Tokenizer size:     "
        f"{actual_vocab_size:,}"
    )

    print(
        f"Training dtype:     "
        f"{train_ids.dtype}"
    )

    print(
        f"Training file:      "
        f"{train_path}"
    )

    print(
        f"Validation file:    "
        f"{validation_path}"
    )

    print("=" * 70)

    return train_path, validation_path


# ---------------------------------------------------------------------------
# File preprocessing
# ---------------------------------------------------------------------------

def preprocess_file(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    tokenizer_dir: Path = DEFAULT_TOKENIZER,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> tuple[Path, Path]:

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input corpus not found: {input_path}"
        )

    print(
        f"Input corpus:\n"
        f"  {input_path}"
    )

    text = input_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    print(
        f"Input characters: "
        f"{len(text):,}"
    )

    return preprocess_text(
        text,
        output_dir=output_dir,
        tokenizer_dir=tokenizer_dir,
        vocab_size=vocab_size,
        validation_fraction=validation_fraction,
    )


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class TokenSequenceDataset:
    """Read a flat token stream and return random input/target windows."""

    def __init__(
        self,
        path: Path,
        context_length: int = 256,
        dtype: np.dtype | type = np.uint16,
    ) -> None:

        if not path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {path}"
            )

        if context_length < 1:
            raise ValueError(
                "context_length must be positive."
            )

        self.tokens = np.memmap(
            path,
            mode="r",
            dtype=dtype,
        )

        if len(self.tokens) <= context_length:
            raise ValueError(
                f"{path} needs more than "
                f"{context_length} tokens."
            )

        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def close(self) -> None:
        """Release memory map."""

        memory_map = getattr(
            self.tokens,
            "_mmap",
            None,
        )

        if memory_map is not None:
            memory_map.close()

    def __enter__(
        self,
    ) -> "TokenSequenceDataset":

        return self

    def __exit__(
        self,
        _exception_type: object,
        _exception: object,
        _traceback: object,
    ) -> None:

        self.close()

    def get_batch(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return input/target batches for next-token prediction."""

        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive."
            )

        rng = (
            rng
            if rng is not None
            else np.random.default_rng()
        )

        starts = rng.integers(
            0,
            len(self),
            size=batch_size,
        )

        inputs = np.stack(
            [
                self.tokens[
                    start:
                    start + self.context_length
                ]
                for start in starts
            ]
        )

        targets = np.stack(
            [
                self.tokens[
                    start + 1:
                    start + self.context_length + 1
                ]
                for start in starts
            ]
        )

        return inputs, targets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess a text corpus for "
            "language-model training."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Load tokenizer and build binary datasets.",
    )

    preprocess_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    preprocess_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    preprocess_parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER,
    )

    preprocess_parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
    )

    preprocess_parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )

    args = parser.parse_args()

    if args.command == "preprocess":

        preprocess_file(
            input_path=args.input,
            output_dir=args.output_dir,
            tokenizer_dir=args.tokenizer_dir,
            vocab_size=args.vocab_size,
            validation_fraction=args.validation_fraction,
        )


if __name__ == "__main__":
    main()