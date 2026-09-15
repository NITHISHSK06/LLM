from pathlib import Path


GENERAL_FILE = Path("data/raw/general.txt")
GUTENBERG_FILE = Path("data/raw/gutenberg_clean.txt")
OUTPUT_FILE = Path("data/raw/general_combined.txt")


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()


def main():
    general = read_text(GENERAL_FILE)
    gutenberg = read_text(GUTENBERG_FILE)

    combined = (
        general
        + "\n\n"
        + gutenberg
        + "\n"
    )

    OUTPUT_FILE.write_text(
        combined,
        encoding="utf-8",
    )

    print("Corpus merge complete!")
    print(f"Original general.txt: {len(general):,} characters")
    print(f"Gutenberg corpus:     {len(gutenberg):,} characters")
    print(f"Combined corpus:      {len(combined):,} characters")
    print(f"Approx tokens:        {len(combined) / 4:,.0f}")
    print(f"Saved to:             {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
    