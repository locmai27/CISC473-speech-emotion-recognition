from pathlib import Path
import csv
import json

DATA_DIR = Path("../data")

# Load clip information
clips = {}

with open(DATA_DIR / "clip_table.csv", "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        clips[row["id"]] = {
            "person": row["person"],
            "label": row["label"]
        }

# Load LOSO splits
with open(DATA_DIR / "splits.json", "r") as f:
    splits = json.load(f)

# Look at the first split
split = splits[0]

print("Split:", split["name"])
print("Held-out test speaker:", split["test_person"])
print("Training clips:", len(split["train_ids"]))
print("Testing clips:", len(split["test_ids"]))

print("\nExample training clip:")
example_train = split["train_ids"][0]
print(example_train, clips[example_train])

print("\nExample testing clip:")
example_test = split["test_ids"][0]
print(example_test, clips[example_test])
