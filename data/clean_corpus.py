from pathlib import Path
import re
import hashlib


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BOOKS_DIR = PROJECT_ROOT / "data" / "raw" / "books"
WIKIPEDIA_FILE = PROJECT_ROOT / "data" / "raw" / "wikipedia" / "wikipedia_raw.txt"

OUTPUT_DIR = PROJECT_ROOT / "data" / "cleaned"
BOOKS_OUTPUT = OUTPUT_DIR / "books_clean.txt"
WIKIPEDIA_OUTPUT = OUTPUT_DIR / "wikipedia_clean.txt"


# ============================================================
# Basic text cleaning
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize text while preserving useful punctuation and paragraphs."""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null bytes and other control characters,
    # but preserve newline and tab.
    text = "".join(
        char for char in text
        if char == "\n" or char == "\t" or not ord(char) < 32
    )

    # Normalize tabs
    text = text.replace("\t", " ")

    # Remove excessive spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)

    # Remove spaces at the beginning/end of lines
    text = "\n".join(line.strip() for line in text.splitlines())

    return text.strip()


# ============================================================
# Gutenberg cleaning
# ============================================================

def remove_gutenberg_boilerplate(text: str) -> str:
    """
    Remove Project Gutenberg header/footer when detectable.

    We keep the actual book content.
    """

    start_patterns = [
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*",
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .*?\*\*\*",
    ]

    end_patterns = [
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*",
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .*?\*\*\*",
    ]

    # Find the beginning of the actual book
    start_match = None

    for pattern in start_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            start_match = match
            break

    if start_match:
        text = text[start_match.end():]

    # Find the end of the actual book
    end_match = None

    for pattern in end_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            end_match = match
            break

    if end_match:
        text = text[:end_match.start()]

    return text


# ============================================================
# Gutenberg fallback cleaning
# ============================================================

def remove_gutenberg_fallback(text: str) -> str:
    """
    Remove common Gutenberg metadata if the official markers
    weren't detected.
    """

    lines = text.splitlines()

    # Locate common beginning markers
    start_index = 0

    for i, line in enumerate(lines[:150]):
        normalized = line.lower()

        if (
            "start of the project gutenberg ebook" in normalized
            or "start of this project gutenberg ebook" in normalized
        ):
            start_index = i + 1
            break

    # Locate common ending markers
    end_index = len(lines)

    for i in range(max(0, len(lines) - 200), len(lines)):
        normalized = lines[i].lower()

        if (
            "end of the project gutenberg ebook" in normalized
            or "end of this project gutenberg ebook" in normalized
        ):
            end_index = i
            break

    return "\n".join(lines[start_index:end_index])


# ============================================================
# Clean one book
# ============================================================

def clean_book(path: Path) -> str:
    print(f"Cleaning book: {path.name}")

    text = path.read_text(encoding="utf-8", errors="replace")

    original_length = len(text)

    text = remove_gutenberg_boilerplate(text)

    # If markers weren't found, use fallback
    if len(text) >= original_length * 0.98:
        text = remove_gutenberg_fallback(text)

    text = normalize_text(text)

    return text


# ============================================================
# Clean books
# ============================================================

def clean_books() -> tuple[str, int]:

    book_files = sorted(
        path for path in BOOKS_DIR.glob("*.txt")
        if path.is_file()
    )

    if not book_files:
        raise FileNotFoundError(
            f"No .txt books found in {BOOKS_DIR}"
        )

    print()
    print("=" * 60)
    print("BOOK CLEANING")
    print("=" * 60)
    print(f"Books found: {len(book_files)}")
    print()

    cleaned_books = []
    seen_hashes = set()

    duplicate_count = 0

    for path in book_files:
        text = clean_book(path)

        if not text:
            print(f"WARNING: Empty after cleaning: {path.name}")
            continue

        # Exact-document deduplication
        text_hash = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        if text_hash in seen_hashes:
            print(f"Duplicate skipped: {path.name}")
            duplicate_count += 1
            continue

        seen_hashes.add(text_hash)
        cleaned_books.append(
            f"\n\n===== BOOK: {path.stem} =====\n\n{text}"
        )

    combined = "\n".join(cleaned_books).strip() + "\n"

    print()
    print(f"Unique books: {len(cleaned_books)}")
    print(f"Duplicate books skipped: {duplicate_count}")
    print(f"Clean book characters: {len(combined):,}")

    return combined, len(cleaned_books)


# ============================================================
# Wikipedia cleaning
# ============================================================

def clean_wikipedia() -> str:

    if not WIKIPEDIA_FILE.exists():
        raise FileNotFoundError(
            f"Wikipedia file not found: {WIKIPEDIA_FILE}"
        )

    print()
    print("=" * 60)
    print("WIKIPEDIA CLEANING")
    print("=" * 60)

    print(f"Input: {WIKIPEDIA_FILE}")

    # Read line-by-line so we don't unnecessarily duplicate
    # the entire 93 MB string in memory multiple times.
    paragraphs = []
    seen_hashes = set()

    current_article = []

    def save_article(lines):
        if not lines:
            return

        article = "\n".join(lines).strip()

        if len(article) < 200:
            return

        article = normalize_text(article)

        if not article:
            return

        article_hash = hashlib.sha256(
            article.encode("utf-8")
        ).hexdigest()

        if article_hash in seen_hashes:
            return

        seen_hashes.add(article_hash)
        paragraphs.append(article)

    with WIKIPEDIA_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        for line in f:
            line = line.strip()

            # Blank lines generally separate articles/paragraphs
            if not line:
                if current_article:
                    save_article(current_article)
                    current_article = []
                continue

            current_article.append(line)

        if current_article:
            save_article(current_article)

    cleaned = "\n\n".join(paragraphs).strip() + "\n"

    print(f"Unique Wikipedia sections: {len(paragraphs):,}")
    print(f"Clean Wikipedia characters: {len(cleaned):,}")

    return cleaned


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CORPUS CLEANING")
    print("=" * 60)

    # --------------------------------------------------------
    # Books
    # --------------------------------------------------------

    books_text, book_count = clean_books()

    BOOKS_OUTPUT.write_text(
        books_text,
        encoding="utf-8"
    )

    print(f"Saved books: {BOOKS_OUTPUT}")

    # --------------------------------------------------------
    # Wikipedia
    # --------------------------------------------------------

    wikipedia_text = clean_wikipedia()

    WIKIPEDIA_OUTPUT.write_text(
        wikipedia_text,
        encoding="utf-8"
    )

    print(f"Saved Wikipedia: {WIKIPEDIA_OUTPUT}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)

    print(f"Books:       {book_count:,}")
    print(f"Book chars:  {len(books_text):,}")
    print(f"Wiki chars:  {len(wikipedia_text):,}")
    print(
        f"Total chars: {len(books_text) + len(wikipedia_text):,}"
    )

    print()
    print("Output files:")
    print(f"  {BOOKS_OUTPUT}")
    print(f"  {WIKIPEDIA_OUTPUT}")


if __name__ == "__main__":
    main()