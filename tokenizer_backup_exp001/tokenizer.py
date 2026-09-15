# python -m pytest
# python training/dataset.py preprocess --vocab-size 10000
# python training/pretrain.py
# python training/instruct_train.py --base-checkpoint checkpoints/base/best_model.pt
# python inference/generate.py --interactive

"""
A small byte-free, character-initialized BPE tokenizer.

This tokenizer is intentionally educational. It learns BPE merge rules
directly from the user's corpus and stores only JSON/text files.

The training implementation uses incremental pair-frequency updates instead
of rescanning the complete corpus for every merge.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

# Unicode-friendly pre-tokenization.
#
# The important difference from the old pattern is that Unicode words
# are preserved instead of only matching A-Z/a-z/0-9.
TOKEN_PATTERN = re.compile(
    r"\s+|[\w]+|[^\w\s]",
    re.UNICODE,
)


class SimpleBPETokenizer:
    """A compact BPE tokenizer with a reusable on-disk vocabulary."""

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
    ) -> None:

        self.vocab = vocab

        self.id_to_token = {
            token_id: token
            for token, token_id in vocab.items()
        }

        self.merges = merges

        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(merges)
        }

        self.pad_id = vocab["<PAD>"]
        self.unk_id = vocab["<UNK>"]
        self.bos_id = vocab["<BOS>"]
        self.eos_id = vocab["<EOS>"]

    @staticmethod
    def _pretokens(text: str) -> list[str]:
        """
        Split into words, punctuation and whitespace
        without losing spaces.
        """
        return TOKEN_PATTERN.findall(text)

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        """
        Apply learned BPE merges in merge-rank order.
        """

        while len(symbols) > 1:

            candidates = []

            for index in range(len(symbols) - 1):

                pair = (
                    symbols[index],
                    symbols[index + 1],
                )

                rank = self.merge_ranks.get(pair)

                if rank is not None:
                    candidates.append((rank, index))

            if not candidates:
                break

            _, index = min(candidates)

            symbols[index:index + 2] = [
                symbols[index] + symbols[index + 1]
            ]

        return symbols

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Convert text into token IDs."""

        token_ids: list[int] = []

        if add_bos:
            token_ids.append(self.bos_id)

        for pretoken in self._pretokens(text):

            symbols = self._apply_merges(
                list(pretoken)
            )

            token_ids.extend(
                self.vocab.get(
                    symbol,
                    self.unk_id,
                )
                for symbol in symbols
            )

        if add_eos:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Convert token IDs back into text."""

        pieces: list[str] = []

        for token_id in token_ids:

            token = self.id_to_token.get(
                int(token_id),
                "<UNK>",
            )

            if (
                skip_special_tokens
                and token in SPECIAL_TOKENS
            ):
                continue

            pieces.append(token)

        return "".join(pieces)

    def save(self, directory: str | Path) -> None:

        directory = Path(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            directory / "vocab.json"
        ).write_text(
            json.dumps(
                self.vocab,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        with (
            directory / "merges.txt"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:

            for first, second in self.merges:

                file.write(
                    json.dumps(
                        [first, second],
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        (
            directory / "tokenizer_config.json"
        ).write_text(
            json.dumps(
                {
                    "type": "character_bpe",
                    "special_tokens": SPECIAL_TOKENS,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "SimpleBPETokenizer":

        directory = Path(directory)

        vocab_path = directory / "vocab.json"
        merges_path = directory / "merges.txt"

        if (
            not vocab_path.exists()
            or not merges_path.exists()
        ):
            raise FileNotFoundError(
                f"Tokenizer files are missing in {directory}. "
                "Run the build command first."
            )

        vocab = json.loads(
            vocab_path.read_text(
                encoding="utf-8"
            )
        )

        merges = [
            tuple(json.loads(line))
            for line in merges_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        return cls(
            vocab,
            merges,
        )


# ---------------------------------------------------------------------
# OPTIMIZED BPE TRAINER
# ---------------------------------------------------------------------


def train_tokenizer(
    text: str,
    vocab_size: int = 8192,
) -> SimpleBPETokenizer:
    """
    Learn frequent adjacent symbol pairs using incremental updates.

    The old implementation recalculated pair frequencies by scanning the
    complete corpus after every merge.

    This implementation:

    1. Builds pair frequencies once.
    2. Selects the most frequent pair.
    3. Updates only the pairs affected by that merge.
    4. Uses a max-heap to efficiently find the next pair.

    This is substantially faster for larger corpora.
    """

    if not text.strip():

        raise ValueError(
            "The training corpus is empty."
        )

    if vocab_size <= len(SPECIAL_TOKENS):

        raise ValueError(
            "vocab_size must be larger than "
            "the number of special tokens."
        )

    # --------------------------------------------------------------
    # Step 1: Build initial character sequences
    # --------------------------------------------------------------

    word_sequences = [
        list(piece)
        for piece in SimpleBPETokenizer._pretokens(text)
        if piece
    ]

    # --------------------------------------------------------------
    # Step 2: Initial vocabulary
    # --------------------------------------------------------------

    symbol_counts = Counter(
        symbol
        for sequence in word_sequences
        for symbol in sequence
    )

    symbols = [
        symbol
        for symbol, _ in symbol_counts.most_common()
    ]

    available = (
        vocab_size
        - len(SPECIAL_TOKENS)
    )

    # If the corpus itself contains fewer symbols than requested,
    # we can still continue creating BPE merges.
    merges: list[tuple[str, str]] = []

    # --------------------------------------------------------------
    # Step 3: Build pair frequencies ONCE
    # --------------------------------------------------------------

    pair_counts: Counter[
        tuple[str, str]
    ] = Counter()

    for sequence in word_sequences:

        for pair in zip(
            sequence,
            sequence[1:],
        ):

            pair_counts[pair] += 1

    # --------------------------------------------------------------
    # Step 4: Efficient heap
    # --------------------------------------------------------------

    import heapq

    # Python heapq is a min-heap, so use negative frequency.
    #
    # Format:
    #
    # (-count, pair)
    #
    # We also use the pair itself as a deterministic tie-breaker.

    heap = [
        (-count, pair)
        for pair, count in pair_counts.items()
    ]

    heapq.heapify(heap)

    # --------------------------------------------------------------
    # Step 5: Repeatedly merge the best pair
    # --------------------------------------------------------------

    merge_number = 0

    while len(symbols) < available:

        # Find a valid pair from the heap.
        best_pair = None

        while heap:

            negative_count, pair = heapq.heappop(heap)

            count = -negative_count

            current_count = pair_counts.get(
                pair,
                0,
            )

            # Ignore stale heap entries.
            if count != current_count:
                continue

            if count < 2:
                best_pair = None
                break

            best_pair = pair
            break

        if best_pair is None:
            break

        pair = best_pair

        merged = (
            pair[0]
            + pair[1]
        )

        # ----------------------------------------------------------
        # Add new vocabulary symbol
        # ----------------------------------------------------------

        if merged not in symbols:

            symbols.append(merged)

        merges.append(pair)

        merge_number += 1

        # ----------------------------------------------------------
        # Find sequences containing the selected pair
        # ----------------------------------------------------------

        affected_sequences: list[int] = []

        for index, sequence in enumerate(
            word_sequences
        ):

            for position in range(
                len(sequence) - 1
            ):

                if (
                    sequence[position],
                    sequence[position + 1],
                ) == pair:

                    affected_sequences.append(
                        index
                    )

                    break

        # ----------------------------------------------------------
        # Update affected sequences
        # ----------------------------------------------------------

        for index in affected_sequences:

            sequence = word_sequences[index]

            # Remove old adjacent-pair counts
            old_pairs = list(
                zip(
                    sequence,
                    sequence[1:],
                )
            )

            for old_pair in old_pairs:

                pair_counts[old_pair] -= 1

                if pair_counts[old_pair] <= 0:

                    del pair_counts[old_pair]

            # Apply merge
            updated: list[str] = []

            position = 0

            while position < len(sequence):

                if (
                    position + 1 < len(sequence)
                    and (
                        sequence[position],
                        sequence[position + 1],
                    ) == pair
                ):

                    updated.append(
                        merged
                    )

                    position += 2

                else:

                    updated.append(
                        sequence[position]
                    )

                    position += 1

            word_sequences[index] = updated

            # Add new adjacent-pair counts
            new_pairs = list(
                zip(
                    updated,
                    updated[1:],
                )
            )

            for new_pair in new_pairs:

                pair_counts[new_pair] += 1

                heapq.heappush(
                    heap,
                    (
                        -pair_counts[new_pair],
                        new_pair,
                    ),
                )

        # ----------------------------------------------------------
        # Progress information
        # ----------------------------------------------------------

        if (
            merge_number % 500 == 0
            or len(symbols) >= available
        ):

            print(
                f"BPE merge {merge_number:,} | "
                f"vocab symbols: {len(symbols):,} | "
                f"best pair count: "
                f"{pair_counts.get(pair, 0):,}"
            )

    # --------------------------------------------------------------
    # Step 6: Construct vocabulary
    # --------------------------------------------------------------

    vocab_tokens = (
        SPECIAL_TOKENS
        + symbols[:available]
    )

    vocab = {
        token: token_id
        for token_id, token
        in enumerate(vocab_tokens)
    }

    print(
        f"\nRequested vocabulary: {vocab_size:,}"
    )

    print(
        f"Actual vocabulary:    {len(vocab):,}"
    )

    print(
        f"Learned merges:        {len(merges):,}"
    )

    if len(vocab) < vocab_size:

        print(
            "\nWARNING: tokenizer could not reach "
            f"{vocab_size:,} tokens."
        )

    return SimpleBPETokenizer(
        vocab,
        merges,
    )


def build_from_file(
    input_path: Path,
    output_dir: Path,
    vocab_size: int,
) -> None:

    if not input_path.exists():

        raise FileNotFoundError(
            f"Training corpus not found: "
            f"{input_path}"
        )

    print(
        f"Reading training corpus: {input_path}"
    )

    text = input_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    print(
        f"Corpus characters: {len(text):,}"
    )

    tokenizer = train_tokenizer(
        text,
        vocab_size,
    )

    tokenizer.save(
        output_dir
    )

    print(
        f"\nSaved tokenizer with "
        f"{len(tokenizer.vocab):,} tokens "
        f"to {output_dir}"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train or test the educational "
            "character-initialized BPE tokenizer."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # --------------------------------------------------------------
    # BUILD
    # --------------------------------------------------------------

    build_parser = subparsers.add_parser(
        "build",
        help="Build tokenizer from a text corpus",
    )

    build_parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/raw/general.txt"
        ),
    )

    build_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "tokenizer"
        ),
    )

    build_parser.add_argument(
        "--vocab-size",
        type=int,
        default=8192,
    )

    # --------------------------------------------------------------
    # TEST
    # --------------------------------------------------------------

    test_parser = subparsers.add_parser(
        "test",
        help="Encode and decode sample text",
    )

    test_parser.add_argument(
        "--text",
        default="Hello, how are you?",
    )

    test_parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path(
            "tokenizer"
        ),
    )

    # --------------------------------------------------------------
    # ARGUMENTS
    # --------------------------------------------------------------

    args = parser.parse_args()

    if args.command == "build":

        build_from_file(
            args.input,
            args.output,
            args.vocab_size,
        )

    else:

        tokenizer = (
            SimpleBPETokenizer.load(
                args.tokenizer_dir
            )
        )

        token_ids = tokenizer.encode(
            args.text,
            add_bos=True,
            add_eos=True,
        )

        print(
            f"Input:   {args.text}"
        )

        print(
            f"IDs:     {token_ids}"
        )

        print(
            f"Tokens:  "
            f"{[tokenizer.id_to_token[i] for i in token_ids]}"
        )

        print(
            f"Decoded: "
            f"{tokenizer.decode(token_ids)}"
        )


if __name__ == "__main__":
    main()