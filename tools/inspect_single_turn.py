import json

path = "data/processed/exp008/instruction_train.jsonl"

single_turn = []

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        if "\n" not in item["prompt"]:
            single_turn.append(item)

print(f"Total single-turn examples: {len(single_turn)}")
print("\nFIRST 30 SINGLE-TURN EXAMPLES:\n")

for i, item in enumerate(single_turn[:30], start=1):
    print(f"[{i}] USER: {item['prompt']}")
    print(f"    ASSISTANT: {item['response']}")
    print()
