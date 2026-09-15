from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("data/cleaned/exp006_fineweb_edu.txt")

# Approximate target.
# Your tokenizer averages ~2.4 characters/token,
# so 50M characters ≈ 20M tokens.
TARGET_CHARS = 50_000_000

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = []
    previous_blank = False

    for line in text.split("\n"):
        line = " ".join(line.split())

        if not line:
            if not previous_blank:
                lines.append("")
            previous_blank = True
        else:
            lines.append(line)
            previous_blank = False

    return "\n".join(lines).strip()


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print("Loading FineWeb-Edu in streaming mode...")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )

    # Shuffle the streaming dataset so we don't simply take
    # the beginning of the corpus.
    dataset = dataset.shuffle(
        seed=42,
        buffer_size=10_000,
    )

    total_chars = 0
    documents = 0

    with OUTPUT.open("w", encoding="utf-8") as f:
        for example in dataset:
            text = clean_text(example["text"])

            if len(text) < 200:
                continue

            f.write(text)
            f.write("\n\n")

            total_chars += len(text)
            documents += 1

            if documents % 1000 == 0:
                print(
                    f"Documents: {documents:,} | "
                    f"Characters: {total_chars:,} / {TARGET_CHARS:,}"
                )

            if total_chars >= TARGET_CHARS:
                break

    print("\nFinished.")
    print(f"Documents: {documents:,}")
    print(f"Characters: {total_chars:,}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()