import json
from pathlib import Path
from collections import Counter

TRAIN_FILE = Path("data/raw/dailydialog_train.jsonl")
VAL_FILE = Path("data/raw/dailydialog_val.jsonl")


def analyze(file_path):
    examples = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    prompt_chars = [len(x["prompt"]) for x in examples]
    response_chars = [len(x["response"]) for x in examples]

    # Rough English estimate:
    # ~4 characters ≈ 1 token
    prompt_tokens = [x / 4 for x in prompt_chars]
    response_tokens = [x / 4 for x in response_chars]

    print("\n" + "=" * 50)
    print(file_path)
    print("=" * 50)

    print(f"Examples: {len(examples):,}")

    print("\nPROMPT")
    print(f"Average characters : {sum(prompt_chars)/len(prompt_chars):.1f}")
    print(f"Maximum characters : {max(prompt_chars):,}")
    print(f"Average tokens     : {sum(prompt_tokens)/len(prompt_tokens):.1f}")
    print(f"Maximum tokens     : {max(prompt_tokens):.1f}")

    print("\nRESPONSE")
    print(f"Average characters : {sum(response_chars)/len(response_chars):.1f}")
    print(f"Maximum characters : {max(response_chars):,}")
    print(f"Average tokens     : {sum(response_tokens)/len(response_tokens):.1f}")
    print(f"Maximum tokens     : {max(response_tokens):.1f}")

    print("\n256-TOKEN CONTEXT CHECK")

    long_prompts = sum(
        1 for tokens in prompt_tokens
        if tokens > 256
    )

    long_total = sum(
        1 for tokens in prompt_tokens
        if tokens > 256 - 50
    )

    print(f"Prompt > 256 tokens     : {long_prompts:,}")
    print(f"Prompt > ~206 tokens    : {long_total:,}")

    print("\n")


def main():
    analyze(TRAIN_FILE)
    analyze(VAL_FILE)


if __name__ == "__main__":
    main()