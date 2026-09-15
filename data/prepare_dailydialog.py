import json
import re
from pathlib import Path

import pandas as pd


INPUT_DIR = Path("data/kaggle/dailydialog")
OUTPUT_DIR = Path("data/raw")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    """Clean an individual utterance."""

    text = str(text)

    # Remove surrounding quotes
    text = text.strip().strip("'").strip('"')

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


def parse_dialog(dialog):
    """
    Parse DailyDialog's unusual CSV dialog representation.

    Example:
    ['Hello'
     'How are you?'
     'I am fine.']
    """

    if pd.isna(dialog):
        return []

    text = str(dialog).strip()

    # Remove [ and ]
    if text.startswith("["):
        text = text[1:]

    if text.endswith("]"):
        text = text[:-1]

    # DailyDialog CSV uses quoted utterances separated
    # by newline/whitespace rather than normal commas.
    pattern = r"""['"]\s*(.*?)\s*['"](?=\s*['"]|\s*$)"""

    matches = re.findall(pattern, text, flags=re.DOTALL)

    utterances = []

    for match in matches:
        cleaned = clean_text(match)

        if cleaned:
            utterances.append(cleaned)

    return utterances


def convert_file(input_file, output_file):

    df = pd.read_csv(input_file)

    print(f"Reading: {input_file}")
    print(f"Rows: {len(df)}")

    examples = []

    for dialog in df["dialog"]:

        utterances = parse_dialog(dialog)

        if len(utterances) < 2:
            continue

        # Create context -> response examples
        for i in range(1, len(utterances)):

            context = "\n".join(utterances[:i])
            response = utterances[i]

            if not context or not response:
                continue

            examples.append({
                "prompt": context,
                "response": response
            })

    # Remove duplicates
    unique_examples = []
    seen = set()

    for example in examples:

        key = (
            example["prompt"],
            example["response"]
        )

        if key in seen:
            continue

        seen.add(key)
        unique_examples.append(example)

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for example in unique_examples:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(f"Examples created: {len(unique_examples)}")
    print(f"Saved: {output_file}")
    print()


def main():

    convert_file(
        INPUT_DIR / "train.csv",
        OUTPUT_DIR / "dailydialog_train.jsonl"
    )

    convert_file(
        INPUT_DIR / "validation.csv",
        OUTPUT_DIR / "dailydialog_val.jsonl"
    )

    convert_file(
        INPUT_DIR / "test.csv",
        OUTPUT_DIR / "dailydialog_test.jsonl"
    )


if __name__ == "__main__":
    main()