from pathlib import Path
import csv
import json
import librosa
import numpy as np

DATA_DIR = Path("../data")
RAVDESS_DIR = DATA_DIR / "RAVDESS"

# Load clip information
clips = {}

with open(DATA_DIR / "clip_table.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        clips[row["id"]] = {
            "person": row["person"],
            "label": row["label"]
        }

# Load splits
with open(DATA_DIR / "splits.json", "r") as f:
    splits = json.load(f)

# First LOSO fold
split = splits[0]

training_speakers = sorted({
    clips[clip_id]["person"]
    for clip_id in split["train_ids"]
})

validation_speakers = training_speakers[:3]

train_ids = [
    clip_id for clip_id in split["train_ids"]
    if clips[clip_id]["person"] not in validation_speakers
]

val_ids = [
    clip_id for clip_id in split["train_ids"]
    if clips[clip_id]["person"] in validation_speakers
]

test_ids = split["test_ids"]


def extract_mfcc(clip_id):
    matches = list(RAVDESS_DIR.rglob(f"{clip_id}.wav"))

    if not matches:
        raise FileNotFoundError(f"Could not find {clip_id}.wav")

    audio, sr = librosa.load(matches[0], sr=16000)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    # Put time first: (time frames, 40 features)
    return mfcc.T


def process_set(ids, name):
    features = []
    labels = []

    for i, clip_id in enumerate(ids):
        mfcc = extract_mfcc(clip_id)

        features.append(mfcc)
        labels.append(clips[clip_id]["label"])

        if (i + 1) % 100 == 0:
            print(f"{name}: processed {i + 1}/{len(ids)}")

    return features, labels


print("Processing training clips...")
X_train, y_train = process_set(train_ids, "Train")

print("\nProcessing validation clips...")
X_val, y_val = process_set(val_ids, "Validation")

print("\nProcessing test clips...")
X_test, y_test = process_set(test_ids, "Test")

print("\nDONE")
print("Training examples:", len(X_train))
print("Validation examples:", len(X_val))
print("Test examples:", len(X_test))

print("\nExample MFCC shape:", X_train[0].shape)
print("Example emotion:", y_train[0])
