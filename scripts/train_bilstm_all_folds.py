from pathlib import Path
import csv
import json
import random
import copy

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader


# =========================================================
# SETTINGS
# =========================================================

SEED = 999
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

DATA_DIR = Path("../data")
RAVDESS_DIR = DATA_DIR / "RAVDESS"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print("Using device:", device)


# =========================================================
# LOAD CLIP TABLE
# =========================================================

clips = {}

with open(DATA_DIR / "clip_table.csv", "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        clips[row["id"]] = {
            "person": row["person"],
            "label": row["label"]
        }


# =========================================================
# LOAD LOSO SPLITS
# =========================================================

with open(DATA_DIR / "splits.json", "r") as f:
    splits = json.load(f)

all_people = sorted({
    info["person"]
    for info in clips.values()
})


# =========================================================
# LABELS
# =========================================================

emotions = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

label_to_index = {
    label: i for i, label in enumerate(emotions)
}

index_to_label = {
    i: label for label, i in label_to_index.items()
}


# =========================================================
# EXTRACT ALL MFCCs ONCE
# =========================================================

def extract_mfcc(clip_id):

    matches = list(RAVDESS_DIR.rglob(f"{clip_id}.wav"))

    if not matches:
        raise FileNotFoundError(f"Could not find {clip_id}.wav")

    audio, sr = librosa.load(
        matches[0],
        sr=16000
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    return mfcc.T.astype(np.float32)


print("\nExtracting MFCCs for all clips...")

feature_cache = {}

all_ids = sorted(clips.keys())

for i, clip_id in enumerate(all_ids):

    feature_cache[clip_id] = extract_mfcc(clip_id)

    if (i + 1) % 200 == 0:
        print(f"Processed {i + 1}/{len(all_ids)}")

print("MFCC extraction complete.\n")


# =========================================================
# DATASET
# =========================================================

class EmotionDataset(Dataset):

    def __init__(self, ids, mean, std):
        self.ids = ids
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        clip_id = self.ids[index]

        x = feature_cache[clip_id]

        # Normalize using TRAINING statistics only
        x = (x - self.mean) / self.std

        y = label_to_index[
            clips[clip_id]["label"]
        ]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            clip_id
        )


def collate_batch(batch):

    sequences, labels, ids = zip(*batch)

    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long
    )

    padded = pad_sequence(
        sequences,
        batch_first=True
    )

    labels = torch.stack(labels)

    return padded, lengths, labels, ids


# =========================================================
# MODEL
# =========================================================

class BiLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 8)
        )

    def forward(self, x, lengths):

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1
        )

        return self.classifier(combined)


# =========================================================
# EVALUATION
# =========================================================

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for x, lengths, y, _ in loader:

            x = x.to(device)
            y = y.to(device)

            output = model(x, lengths)

            loss = criterion(output, y)

            total_loss += loss.item() * len(y)

            predictions = output.argmax(dim=1)

            correct += (
                predictions == y
            ).sum().item()

            total += len(y)

    return (
        total_loss / total,
        correct / total
    )


# =========================================================
# RUN ALL 24 FOLDS
# =========================================================

all_predictions = []
fold_results = []

for fold_number, split in enumerate(splits, start=1):

    print("\n" + "=" * 60)
    print(
        f"FOLD {fold_number}/24 "
        f"- TEST SPEAKER {split['test_person']}"
    )
    print("=" * 60)

    test_person = split["test_person"]

    # -----------------------------------------------------
    # Choose 3 validation speakers cyclically
    # -----------------------------------------------------

    test_index = all_people.index(test_person)

    validation_speakers = [
        all_people[
            (test_index + offset) % len(all_people)
        ]
        for offset in range(1, 4)
    ]

    train_ids = [
        clip_id
        for clip_id in split["train_ids"]
        if clips[clip_id]["person"]
        not in validation_speakers
    ]

    val_ids = [
        clip_id
        for clip_id in split["train_ids"]
        if clips[clip_id]["person"]
        in validation_speakers
    ]

    test_ids = split["test_ids"]

    print("Validation speakers:", validation_speakers)
    print("Train clips:", len(train_ids))
    print("Validation clips:", len(val_ids))
    print("Test clips:", len(test_ids))

    # -----------------------------------------------------
    # Normalization statistics from TRAINING ONLY
    # -----------------------------------------------------

    training_frames = np.concatenate(
        [
            feature_cache[clip_id]
            for clip_id in train_ids
        ],
        axis=0
    )

    mean = training_frames.mean(axis=0)

    std = (
        training_frames.std(axis=0)
        + 1e-8
    )

    # -----------------------------------------------------
    # Data loaders
    # -----------------------------------------------------

    train_dataset = EmotionDataset(
        train_ids,
        mean,
        std
    )

    val_dataset = EmotionDataset(
        val_ids,
        mean,
        std
    )

    test_dataset = EmotionDataset(
        test_ids,
        mean,
        std
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch
    )

    # -----------------------------------------------------
    # Reset random seed before each fold
    # -----------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = BiLSTM().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = None

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for x, lengths, y, _ in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(
                x,
                lengths
            )

            loss = criterion(
                output,
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * len(y)
            )

            predictions = output.argmax(
                dim=1
            )

            correct += (
                predictions == y
            ).sum().item()

            total += len(y)

        train_loss = total_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train acc {train_accuracy:.3f} | "
            f"Val acc {val_accuracy:.3f} | "
            f"Val loss {val_loss:.3f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

    # -----------------------------------------------------
    # Load best validation model
    # -----------------------------------------------------

    model.load_state_dict(
        best_model_state
    )

    # -----------------------------------------------------
    # Test held-out speaker ONCE
    # -----------------------------------------------------

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion
    )

    print(
        f"\nBest epoch: {best_epoch}"
    )

    print(
        f"Test accuracy: {test_accuracy:.3f}"
    )

    fold_results.append({
        "split": split["name"],
        "test_person": test_person,
        "validation_speakers":
            "-".join(validation_speakers),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_accuracy": test_accuracy
    })

    # -----------------------------------------------------
    # Save predictions in memory
    # -----------------------------------------------------

    model.eval()

    with torch.no_grad():

        for x, lengths, y, ids in test_loader:

            x = x.to(device)

            output = model(
                x,
                lengths
            )

            predictions = output.argmax(
                dim=1
            ).cpu()

            for clip_id, true_index, pred_index in zip(
                ids,
                y,
                predictions
            ):

                all_predictions.append({
                    "id": clip_id,
                    "true":
                        index_to_label[
                            true_index.item()
                        ],
                    "prediction":
                        index_to_label[
                            pred_index.item()
                        ],
                    "split":
                        split["name"],
                    "method":
                        "mfcc_bilstm",
                    "seed":
                        SEED
                })


# =========================================================
# SAVE COMBINED PREDICTIONS
# =========================================================

prediction_file = (
    DATA_DIR /
    "predictions_mfcc_bilstm_all_folds.csv"
)

with open(
    prediction_file,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "id",
            "true",
            "prediction",
            "split",
            "method",
            "seed"
        ]
    )

    writer.writeheader()
    writer.writerows(all_predictions)


# =========================================================
# SAVE FOLD SUMMARY
# =========================================================

results_file = (
    DATA_DIR /
    "mfcc_bilstm_fold_results.csv"
)

with open(
    results_file,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split",
            "test_person",
            "validation_speakers",
            "best_epoch",
            "best_val_loss",
            "test_accuracy"
        ]
    )

    writer.writeheader()
    writer.writerows(fold_results)


print("\n" + "=" * 60)
print("ALL 24 FOLDS COMPLETE")
print("=" * 60)

print(
    "Total test predictions:",
    len(all_predictions)
)

print(
    "Predictions saved to:",
    prediction_file
)

print(
    "Fold results saved to:",
    results_file
)
