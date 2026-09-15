"""
Efficient character-level BPE tokenizer.

Compatible with the existing project:
    tokenizer/
        tokenizer.py
        vocab.json
        merges.txt

The tokenizer:
- Uses the same regex pre-tokenization strategy.
- Starts from characters.
- Learns BPE merges.
- Saves vocab.json and merges.txt.
- Supports encode/decode/save/load.
- Uses frequency-weighted unique pre-token sequences during training
  for much better training performance than the original implementation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

# Keep this compatible with the original tokenizer.
TOKEN_PATTERN = re.compile(
    r"\s+|[A-Za-z0-9_]+|[^\w\s]",
    re.UNICODE,
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class SimpleBPETokenizer:
    """
    Character-initialized BPE tokenizer.

    The vocabulary contains:
        special tokens
        initial character symbols
        merged BPE symbols

    merges.txt contains:
        symbol1 symbol2

    Merge order determines priority during encoding.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        special_tokens: List[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges

        self.special_tokens = (
            special_tokens
            if special_tokens is not None
            else SPECIAL_TOKENS.copy()
        )

        self.token_to_id = self.vocab
        self.id_to_token = {
            idx: token for token, idx in self.vocab.items()
        }

        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]

    # -----------------------------------------------------------------------
    # Pre-tokenization
    # -----------------------------------------------------------------------

    @staticmethod
    def _pretokens(text: str) -> List[str]:
        """
        Split text using the same regex used by the original tokenizer.
        """
        return TOKEN_PATTERN.findall(text)

    # -----------------------------------------------------------------------
    # BPE application
    # -----------------------------------------------------------------------

    def _apply_merges(self, symbols: List[str]) -> List[str]:
        """
        Apply learned BPE merges to one pre-token.

        At every stage, choose the available adjacent pair with the
        highest-priority learned merge.
        """

        if len(symbols) < 2:
            return symbols

        symbols = list(symbols)

        while len(symbols) > 1:
            best_rank = None
            best_index = None

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])

                rank = self.merge_ranks.get(pair)

                if rank is None:
                    continue

                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = i

            if best_index is None:
                break

            merged = symbols[best_index] + symbols[best_index + 1]

            symbols = (
                symbols[:best_index]
                + [merged]
                + symbols[best_index + 2:]
            )

        return symbols

    # -----------------------------------------------------------------------
    # Encoding
    # -----------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode text into token IDs.
        """

        token_ids: List[int] = []

        if add_bos:
            token_ids.append(self.bos_token_id)

        pretokens = self._pretokens(text)

        for piece in pretokens:
            symbols = list(piece)
            symbols = self._apply_merges(symbols)

            for symbol in symbols:
                token_id = self.vocab.get(
                    symbol,
                    self.unk_token_id,
                )
                token_ids.append(token_id)

        if add_eos:
            token_ids.append(self.eos_token_id)

        return token_ids

    # -----------------------------------------------------------------------
    # Decoding
    # -----------------------------------------------------------------------

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back into text.
        """

        pieces: List[str] = []

        for token_id in token_ids:
            token = self.id_to_token.get(
                int(token_id),
                self.unk_token,
            )

            if skip_special_tokens and token in self.special_tokens:
                continue

            pieces.append(token)

        return "".join(pieces)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """
        Save tokenizer files.
        """

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        vocab_path = directory / "vocab.json"
        merges_path = directory / "merges.txt"

        vocab_path.write_text(
            json.dumps(
                self.vocab,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        with merges_path.open("w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")

            for left, right in self.merges:
                f.write(f"{left} {right}\n")

        print(f"Saved tokenizer to: {directory}")
        print(f"Vocabulary size: {len(self.vocab):,}")
        print(f"Number of merges: {len(self.merges):,}")

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "SimpleBPETokenizer":
        """
        Load tokenizer from vocab.json and merges.txt.
        """

        directory = Path(directory)

        vocab_path = directory / "vocab.json"
        merges_path = directory / "merges.txt"

        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Missing vocabulary file: {vocab_path}"
            )

        if not merges_path.exists():
            raise FileNotFoundError(
                f"Missing merges file: {merges_path}"
            )

        vocab = json.loads(
            vocab_path.read_text(encoding="utf-8")
        )

        merges: List[Tuple[str, str]] = []

        with merges_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line.startswith("#"):
                    continue

                parts = line.split(" ", 1)

                if len(parts) != 2:
                    continue

                merges.append(
                    (parts[0], parts[1])
                )

        return cls(
            vocab=vocab,
            merges=merges,
        )


# ---------------------------------------------------------------------------
# Efficient BPE training
# ---------------------------------------------------------------------------

def train_tokenizer(
    text: str,
    vocab_size: int = 10_000,
) -> SimpleBPETokenizer:
    """
    Train an efficient character-level BPE tokenizer.

    Major optimization compared with the original implementation:

    Original:
        Every occurrence of every pre-token is represented separately.

    This implementation:
        1. Counts unique pre-token strings.
        2. Stores each unique sequence only once.
        3. Uses its frequency as a weight.
        4. Maintains an index from pair -> affected word types.
        5. Only updates sequences containing the selected pair.

    This dramatically reduces the amount of work on large corpora.
    """

    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError(
            f"vocab_size must be greater than "
            f"{len(SPECIAL_TOKENS)}"
        )

    print("=" * 70)
    print("EFFICIENT BPE TOKENIZER TRAINING")
    print("=" * 70)

    print(f"Corpus characters: {len(text):,}")
    print(f"Target vocabulary: {vocab_size:,}")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Pre-tokenize and count unique pieces
    # -----------------------------------------------------------------------

    print("Step 1/6: Pre-tokenizing corpus...")

    pretokens = SimpleBPETokenizer._pretokens(text)

    print(f"Total pre-token occurrences: {len(pretokens):,}")

    word_counts = Counter(pretokens)

    print(
        f"Unique pre-token types: {len(word_counts):,}"
    )

    # Release the huge occurrence list.
    del pretokens

    # -----------------------------------------------------------------------
    # Step 2: Build unique character sequences
    # -----------------------------------------------------------------------

    print()
    print("Step 2/6: Building unique character sequences...")

    # Each unique pre-token gets an integer ID.
    words: List[List[str]] = []
    frequencies: List[int] = []

    for word, frequency in word_counts.items():
        words.append(list(word))
        frequencies.append(frequency)

    del word_counts

    print(f"Unique sequences: {len(words):,}")

    # -----------------------------------------------------------------------
    # Step 3: Build initial vocabulary
    # -----------------------------------------------------------------------

    print()
    print("Step 3/6: Building initial vocabulary...")

    symbol_set = set()

    for sequence in words:
        symbol_set.update(sequence)

    initial_symbols = sorted(symbol_set)

    print(
        f"Initial character symbols: {len(initial_symbols):,}"
    )

    available_merges = (
        vocab_size
        - len(SPECIAL_TOKENS)
        - len(initial_symbols)
    )

    if available_merges <= 0:
        raise ValueError(
            "Vocabulary size is too small for the number "
            "of initial symbols."
        )

    print(
        f"Maximum BPE merges: {available_merges:,}"
    )

    # -----------------------------------------------------------------------
    # Step 4: Initial pair counts + pair index
    # -----------------------------------------------------------------------

    print()
    print("Step 4/6: Building pair statistics...")

    pair_counts: Counter[Tuple[str, str]] = Counter()

    # pair_to_words[pair] = set(word IDs containing that pair)
    pair_to_words: Dict[
        Tuple[str, str],
        set[int]
    ] = defaultdict(set)

    for word_id, sequence in enumerate(words):
        frequency = frequencies[word_id]

        if len(sequence) < 2:
            continue

        seen_pairs = set(
            zip(
                sequence[:-1],
                sequence[1:],
            )
        )

        # Pair frequency must count every occurrence,
        # weighted by word frequency.
        occurrence_counts = Counter(
            zip(
                sequence[:-1],
                sequence[1:],
            )
        )

        for pair, count in occurrence_counts.items():
            pair_counts[pair] += count * frequency
            pair_to_words[pair].add(word_id)

    print(
        f"Initial unique pairs: {len(pair_counts):,}"
    )

    # -----------------------------------------------------------------------
    # Step 5: Heap-based BPE merging
    # -----------------------------------------------------------------------

    print()
    print("Step 5/6: Learning BPE merges...")

    import heapq

    # Python heap is a min-heap.
    #
    # Store:
    #     (-frequency, pair)
    #
    # Most frequent pair comes first.
    heap = [
        (-count, pair)
        for pair, count in pair_counts.items()
    ]

    heapq.heapify(heap)

    merges: List[Tuple[str, str]] = []

    # For progress reporting.
    progress_interval = 100

    while len(merges) < available_merges:

        # ---------------------------------------------------------------
        # Find current best pair.
        # ---------------------------------------------------------------

        selected_pair = None
        selected_count = None

        while heap:

            negative_count, pair = heapq.heappop(heap)

            current_count = pair_counts.get(
                pair,
                0,
            )

            heap_count = -negative_count

            # Ignore stale heap entries.
            if heap_count != current_count:
                continue

            if current_count <= 0:
                continue

            selected_pair = pair
            selected_count = current_count
            break

        if selected_pair is None:
            print(
                "No more valid pairs available."
            )
            break

        left, right = selected_pair

        # ---------------------------------------------------------------
        # Record merge.
        # ---------------------------------------------------------------

        merges.append(selected_pair)

        merged_symbol = left + right

        # ---------------------------------------------------------------
        # Only update word types containing the selected pair.
        # ---------------------------------------------------------------

        affected_word_ids = list(
            pair_to_words.get(
                selected_pair,
                set(),
            )
        )

        # Remove selected pair from index.
        pair_to_words.pop(
            selected_pair,
            None,
        )

        # ---------------------------------------------------------------
        # Update every affected unique word.
        # ---------------------------------------------------------------

        for word_id in affected_word_ids:

            old_sequence = words[word_id]
            frequency = frequencies[word_id]

            if len(old_sequence) < 2:
                continue

            # Verify pair still exists.
            contains_pair = False

            for i in range(len(old_sequence) - 1):
                if (
                    old_sequence[i] == left
                    and old_sequence[i + 1] == right
                ):
                    contains_pair = True
                    break

            if not contains_pair:
                continue

            # -----------------------------------------------------------
            # Remove all old pair contributions for this word.
            # -----------------------------------------------------------

            old_pairs = Counter(
                zip(
                    old_sequence[:-1],
                    old_sequence[1:],
                )
            )

            for old_pair, occurrence_count in old_pairs.items():

                pair_counts[old_pair] -= (
                    occurrence_count * frequency
                )

                if pair_counts[old_pair] <= 0:
                    pair_counts.pop(
                        old_pair,
                        None,
                    )

                word_set = pair_to_words.get(
                    old_pair
                )

                if word_set is not None:
                    word_set.discard(word_id)

                    if not word_set:
                        pair_to_words.pop(
                            old_pair,
                            None,
                        )

            # -----------------------------------------------------------
            # Merge all non-overlapping occurrences.
            # -----------------------------------------------------------

            new_sequence: List[str] = []

            i = 0

            while i < len(old_sequence):

                if (
                    i < len(old_sequence) - 1
                    and old_sequence[i] == left
                    and old_sequence[i + 1] == right
                ):
                    new_sequence.append(
                        merged_symbol
                    )
                    i += 2
                else:
                    new_sequence.append(
                        old_sequence[i]
                    )
                    i += 1

            words[word_id] = new_sequence

            # -----------------------------------------------------------
            # Add new pair contributions.
            # -----------------------------------------------------------

            new_pairs = Counter(
                zip(
                    new_sequence[:-1],
                    new_sequence[1:],
                )
            )

            for new_pair, occurrence_count in new_pairs.items():

                pair_counts[new_pair] += (
                    occurrence_count * frequency
                )

                pair_to_words[
                    new_pair
                ].add(word_id)

                # Push a fresh heap entry.
                heapq.heappush(
                    heap,
                    (
                        -pair_counts[new_pair],
                        new_pair,
                    ),
                )

        # ---------------------------------------------------------------
        # Add the newly created symbol to vocabulary later.
        # ---------------------------------------------------------------

        if (
            len(merges) % progress_interval == 0
            or len(merges) == 1
        ):
            print(
                f"Merge {len(merges):5d}/{available_merges:,} "
                f"| pair=({left!r}, {right!r}) "
                f"| frequency={selected_count:,} "
                f"| new_token={merged_symbol!r}"
            )

    # -----------------------------------------------------------------------
    # Step 6: Construct vocabulary
    # -----------------------------------------------------------------------

    print()
    print("Step 6/6: Constructing vocabulary...")

    vocab: Dict[str, int] = {}

    # Special tokens first.
    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)

    # Initial characters.
    for symbol in initial_symbols:
        if symbol not in vocab:
            vocab[symbol] = len(vocab)

    # Merged symbols.
    for left, right in merges:
        merged_symbol = left + right

        if merged_symbol not in vocab:
            vocab[merged_symbol] = len(vocab)

    # If the exact target size wasn't reached, that's okay.
    # This can happen if there are no more useful pairs.
    print(
        f"Final vocabulary size: {len(vocab):,}"
    )

    print(
        f"Final merge count: {len(merges):,}"
    )

    print("=" * 70)
    print("TOKENIZER TRAINING COMPLETE")
    print("=" * 70)

    return SimpleBPETokenizer(
        vocab=vocab,
        merges=merges,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_tokenizer(
    input_path: str | Path,
    output_dir: str | Path,
    vocab_size: int,
) -> None:
    """
    Load text, train tokenizer, and save it.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    print(f"Loading corpus: {input_path}")

    text = input_path.read_text(
        encoding="utf-8"
    )

    print(
        f"Loaded {len(text):,} characters."
    )

    tokenizer = train_tokenizer(
        text=text,
        vocab_size=vocab_size,
    )

    tokenizer.save(output_dir)

    # -----------------------------------------------------------------------
    # Quick sanity test
    # -----------------------------------------------------------------------

    print()
    print("Running tokenizer sanity check...")

    test_texts = [
        "Artificial intelligence is transforming technology.",
        "Machine learning is a branch of artificial intelligence.",
        "The quick brown fox jumps over the lazy dog.",
        "Hello, world!",
    ]

    for test_text in test_texts:
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)

        print()
        print(f"Input:   {test_text}")
        print(f"Tokens:  {len(encoded)}")
        print(f"IDs:     {encoded[:20]}")
        print(f"Decoded: {decoded}")

    print()
    print("Tokenizer sanity check complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Train the project's BPE tokenizer."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # -----------------------------------------------------------------------
    # build
    # -----------------------------------------------------------------------

    build_parser = subparsers.add_parser(
        "build",
        help="Build a tokenizer from a text corpus.",
    )

    build_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input text file.",
    )

    build_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output tokenizer directory.",
    )

    build_parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
        help="Target vocabulary size.",
    )

    # -----------------------------------------------------------------------
    # test
    # -----------------------------------------------------------------------

    test_parser = subparsers.add_parser(
        "test",
        help="Test an existing tokenizer.",
    )

    test_parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Tokenizer directory.",
    )

    test_parser.add_argument(
        "--text",
        type=str,
        default="Artificial intelligence is amazing!",
        help="Text to encode/decode.",
    )

    args = parser.parse_args()

    if args.command == "build":

        build_tokenizer(
            input_path=args.input,
            output_dir=args.output,
            vocab_size=args.vocab_size,
        )

    elif args.command == "test":

        tokenizer = SimpleBPETokenizer.load(
            args.tokenizer
        )

        encoded = tokenizer.encode(
            args.text
        )

        decoded = tokenizer.decode(
            encoded
        )

        print(f"Input:   {args.text}")
        print(f"Tokens:  {encoded}")
        print(f"Count:   {len(encoded)}")
        print(f"Decoded: {decoded}")


if __name__ == "__main__":
    main()