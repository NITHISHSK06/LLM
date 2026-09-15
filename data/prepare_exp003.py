from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BOOKS_FILE = PROJECT_ROOT / "data" / "cleaned" / "books_clean.txt"
WIKI_FILE = PROJECT_ROOT / "data" / "cleaned" / "wikipedia_clean.txt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "exp003_corpus.txt"

# Target ratio: 70% Wikipedia / 30% Books
# Books contain ~4.9M tokens, so target Wiki is ~11.44M tokens.
#
# We use character proportion as a deterministic approximation here,
# then verify the REAL token ratio with the tokenizer afterward.

BOOK_RATIO = 0.30
WIKI_RATIO = 0.70


def read_text(path: Path) -> str:
    print(f"Reading: {path}")
    return path.read_text(encoding="utf-8")


def main():
    books = read_text(BOOKS_FILE)
    wiki = read_text(WIKI_FILE)

    print()
    print("=" * 60)
    print("Experiment 003 corpus preparation")
    print("=" * 60)

    print(f"Books characters: {len(books):,}")
    print(f"Wiki characters:  {len(wiki):,}")

    # Use the complete book corpus.
    # Select enough Wikipedia text to approximately produce 70/30.
    #
    # Because books/wiki have different characters-per-token ratios,
    # this is only an initial selection. Actual token ratio will be
    # measured after tokenization.

    target_wiki_chars = int(
        len(books) * (WIKI_RATIO / BOOK_RATIO)
    )

    target_wiki_chars = min(target_wiki_chars, len(wiki))

    wiki_selected = wiki[:target_wiki_chars]

    output = (
        "===== EXPERIMENT 003: BOOKS =====\n\n"
        + books
        + "\n\n"
        + "===== EXPERIMENT 003: WIKIPEDIA =====\n\n"
        + wiki_selected
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(output, encoding="utf-8")

    print()
    print(f"Selected Wiki characters: {len(wiki_selected):,}")
    print(f"Output characters:         {len(output):,}")
    print()
    print(f"Saved to:")
    print(OUTPUT_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()