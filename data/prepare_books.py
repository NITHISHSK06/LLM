"""Clean local book text into the general-language corpus."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def clean_book_text(text: str) -> str:
    """Clean a single book/document."""

    text = str(text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove Gutenberg START/END markers if present
    text = re.sub(
        r"\*\*\*\s*START OF (?:THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(
        r"\*\*\*\s*END OF (?:THE )?PROJECT GUTENBERG EBOOK.*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Normalize whitespace inside lines
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    paragraphs: list[str] = []
    seen: set[str] = set()

    for raw in re.split(r"\n\s*\n+", text):
        paragraph = raw.strip()

        # Ignore tiny fragments
        if not paragraph or len(paragraph) < 20:
            continue

        # Remove common page/chapter noise
        if re.fullmatch(
            r"(?:chapter|contents|page)\s*[\w\d .:-]*",
            paragraph,
            re.IGNORECASE,
        ):
            continue

        # Remove exact duplicate paragraphs
        key = paragraph.casefold()

        if key in seen:
            continue

        seen.add(key)
        paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)


def find_text_column(df: pd.DataFrame) -> str:
    """Automatically find the column containing the book text."""

    preferred = [
        "text",
        "content",
        "book",
        "novel",
        "body",
        "full_text",
        "description",
    ]

    columns_lower = {
        str(column).lower(): column
        for column in df.columns
    }

    for name in preferred:
        if name in columns_lower:
            return columns_lower[name]

    # Fallback: choose the string column with the most text
    candidates = []

    for column in df.columns:
        if df[column].dtype == "object":
            total_chars = (
                df[column]
                .fillna("")
                .astype(str)
                .str.len()
                .sum()
            )

            candidates.append((total_chars, column))

    if not candidates:
        raise ValueError(
            "Could not find a text column in the Gutenberg dataset."
        )

    candidates.sort(reverse=True)

    return candidates[0][1]


def process_gutenberg_csv(
    input_file: Path,
    output_file: Path,
) -> None:
    """Convert Gutenberg CSV into cleaned text."""

    print(f"Reading Gutenberg dataset:")
    print(f"  {input_file}")

    df = pd.read_csv(input_file)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    text_column = find_text_column(df)

    print(f"\nSelected text column: {text_column}")

    documents: list[str] = []

    for value in df[text_column].dropna():
        cleaned = clean_book_text(value)

        if len(cleaned) >= 500:
            documents.append(cleaned)

    print(f"Usable documents: {len(documents):,}")

    # Remove duplicate books/documents
    documents = list(dict.fromkeys(documents))

    print(f"After deduplication: {len(documents):,}")

    output = "\n\n".join(documents)

    if not output:
        raise ValueError(
            "No usable text remained after cleaning."
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        output + "\n",
        encoding="utf-8",
    )

    total_chars = len(output)

    print("\nGutenberg processing complete!")
    print(f"Output: {output_file}")
    print(f"Characters: {total_chars:,}")
    print(
        f"Approx tokens: {total_chars / 4:,.0f}"
    )


def process_local_books(
    input_dir: Path,
    output_file: Path,
) -> None:
    """Process locally stored .txt books."""

    files = sorted(input_dir.rglob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"No .txt books found under {input_dir}"
        )

    documents = []

    for path in files:
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        cleaned = clean_book_text(text)

        if cleaned:
            documents.append(cleaned)

    output = "\n\n".join(documents)

    if not output:
        raise ValueError(
            "No usable text remained after cleaning."
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        output + "\n",
        encoding="utf-8",
    )

    print(
        f"Processed {len(files):,} local books"
    )

    print(
        f"Saved to: {output_file}"
    )

    print(
        f"Characters: {len(output):,}"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Prepare book data for LLM pretraining."
    )

    parser.add_argument(
        "--gutenberg",
        type=Path,
        default=Path(
            "data/kaggle/gutenberg/"
            "gutenberg_novels_dataset.csv"
        ),
        help="Gutenberg CSV dataset",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/raw/gutenberg_clean.txt"
        ),
        help="Output cleaned corpus",
    )

    parser.add_argument(
        "--local-books",
        type=Path,
        default=None,
        help="Optional local books directory",
    )

    args = parser.parse_args()

    if args.local_books is not None:

        process_local_books(
            args.local_books,
            args.output,
        )

    else:

        if not args.gutenberg.exists():
            raise FileNotFoundError(
                f"Gutenberg dataset not found: "
                f"{args.gutenberg}"
            )

        process_gutenberg_csv(
            args.gutenberg,
            args.output,
        )


if __name__ == "__main__":
    main()