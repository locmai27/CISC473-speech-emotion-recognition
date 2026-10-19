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

# First LOSO split
split = splits[0]

# Speakers available for training
training_speakers = sorted({
    clips[clip_id]["person"]
    for clip_id in split["train_ids"]
})

# Use first 3 training speakers for validation
validation_speakers = training_speakers[:3]

train_ids = [
    clip_id
    for clip_id in split["train_ids"]
    if clips[clip_id]["person"] not in validation_speakers
]

val_ids = [
    clip_id
    for clip_id in split["train_ids"]
    if clips[clip_id]["person"] in validation_speakers
]

test_ids = split["test_ids"]

print("Test speaker:", split["test_person"])
print("Validation speakers:", validation_speakers)

print("Training clips:", len(train_ids))
print("Validation clips:", len(val_ids))
print("Testing clips:", len(test_ids))

