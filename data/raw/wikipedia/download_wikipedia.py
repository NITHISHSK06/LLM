from datasets import load_dataset
from pathlib import Path

OUTPUT = Path("data/raw/wikipedia/wikipedia_raw.txt")
MAX_ARTICLES = 10000

print("Loading Wikipedia dataset...")

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True,
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

count = 0

with OUTPUT.open("w", encoding="utf-8") as f:
    for item in dataset:
        text = item.get("text", "").strip()

        if not text:
            continue

        f.write(text)
        f.write("\n\n")

        count += 1

        if count % 1000 == 0:
            print(f"Processed {count} articles")

        if count >= MAX_ARTICLES:
            break

print(f"Finished: {count} articles")
print(f"Saved to: {OUTPUT}")