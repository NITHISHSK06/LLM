"""Build an augmented Exp012 supervised fine-tuning dataset."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/exp011_general_chat/instruction_train.jsonl"
VALIDATION_SOURCE = ROOT / "data/processed/exp011_general_chat/instruction_val.jsonl"
OUTPUT = ROOT / "data/processed/exp012_sft_quality"

QUALITY_EXAMPLES = [
    ("Explain machine learning simply.", "Machine learning is a way for computers to learn patterns from examples. Instead of programming every rule, we give the computer data and it uses those patterns to make predictions or decisions."),
    ("What is artificial intelligence?", "Artificial intelligence is the field of building computer systems that can perform tasks that usually require human intelligence, such as understanding language, recognizing images, and making decisions."),
    ("Explain deep learning simply.", "Deep learning is a type of machine learning that uses neural networks with many layers. The layers learn increasingly useful patterns from data."),
    ("What is a neural network?", "A neural network is a machine-learning model made of connected layers of artificial neurons. It adjusts its numerical weights while training so its output becomes closer to the desired answer."),
    ("What is the difference between training and testing a model?", "Training uses examples to adjust a model's weights. Testing uses separate examples to measure how well the trained model works on data it has not seen before."),
    ("What is overfitting?", "Overfitting happens when a model memorizes its training examples instead of learning general patterns. It performs well on training data but poorly on new data."),
    ("What is a dataset?", "A dataset is an organized collection of examples used for analysis or machine learning. Each example may contain inputs, and sometimes a label that represents the desired output."),
    ("What is Python?", "Python is a general-purpose programming language known for readable syntax and a large ecosystem of libraries. It is commonly used for web development, automation, data analysis, and machine learning."),
    ("Write a Python function that adds two numbers.", "```python\ndef add(first, second):\n    return first + second\n```\n\nThe function returns the sum of its two arguments."),
    ("How do you define a function in Python?", "Use the `def` keyword followed by a name and parentheses. For example, `def greet(name):` starts a function definition, and an indented `return` statement can provide its result."),
    ("What is an algorithm?", "An algorithm is a finite sequence of clear steps for solving a problem or completing a task. A good algorithm is precise and eventually produces a result."),
    ("What is the purpose of a validation set?", "A validation set is held-out data used during development to compare settings and detect overfitting. It is separate from the final test set used for the last evaluation."),
    ("Explain tokens in language models.", "A token is a small unit of text, such as a character, word fragment, or word. A language model converts text into token IDs and learns to predict the next token."),
    ("What is a language model?", "A language model learns patterns in text and assigns probabilities to possible next tokens. A decoder-only language model generates text by repeatedly predicting and appending one token."),
    ("How does gradient descent train a model?", "Gradient descent calculates how each weight affects the error, then changes the weights a small amount in the direction that reduces that error. Repeating this over batches improves the model."),
]


def load_jsonl(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def main() -> None:
    dialogue = load_jsonl(SOURCE)
    validation = load_jsonl(VALIDATION_SOURCE)
    quality = [{"prompt": prompt, "response": response} for prompt, response in QUALITY_EXAMPLES]
    # Repetition gives the small curated knowledge set enough sampling weight
    # without altering the original project datasets.
    train = dialogue + quality * 100
    write_jsonl(OUTPUT / "instruction_train.jsonl", train)
    write_jsonl(OUTPUT / "instruction_val.jsonl", validation + quality)
    print(f"Wrote {len(train)} training and {len(validation) + len(quality)} validation examples to {OUTPUT}")


if __name__ == "__main__":
    main()
