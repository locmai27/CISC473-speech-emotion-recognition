from pathlib import Path
import csv
import json
import random

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Settings
# -------------------------

SEED = 42
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path("../data")
RAVDESS_DIR = DATA_DIR / "RAVDESS"

device = torch.device("cpu")
print("Using device:", device)


# -------------------------
# Load clip information
# -------------------------

clips = {}

with open(DATA_DIR / "clip_table.csv", "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        clips[row["id"]] = {
            "person": row["person"],
            "label": row["label"]
        }


# -------------------------
# Load LOSO split
# -------------------------

with open(DATA_DIR / "splits.json", "r") as f:
    splits = json.load(f)

split = splits[0]

training_speakers = sorted({
    clips[clip_id]["person"]
    for clip_id in split["train_ids"]
})

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
print("Train:", len(train_ids))
print("Validation:", len(val_ids))
print("Test:", len(test_ids))


# -------------------------
# Emotion labels
# -------------------------

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


# -------------------------
# MFCC extraction
# -------------------------

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

    return mfcc.T.astype(np.float32)


print("\nExtracting MFCCs...")


def prepare_features(ids):
    features = []
    labels = []

    for i, clip_id in enumerate(ids):
        features.append(extract_mfcc(clip_id))
        labels.append(label_to_index[clips[clip_id]["label"]])

        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(ids)}")

    return features, labels


X_train, y_train = prepare_features(train_ids)
X_val, y_val = prepare_features(val_ids)
X_test, y_test = prepare_features(test_ids)


# -------------------------
# Normalize using TRAINING data only
# -------------------------

all_train_frames = np.concatenate(X_train, axis=0)

mean = all_train_frames.mean(axis=0)
std = all_train_frames.std(axis=0) + 1e-8

X_train = [(x - mean) / std for x in X_train]
X_val = [(x - mean) / std for x in X_val]
X_test = [(x - mean) / std for x in X_test]


# -------------------------
# Dataset
# -------------------------

class EmotionDataset(Dataset):

    def __init__(self, features, labels, ids):
        self.features = features
        self.labels = labels
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = torch.tensor(self.features[index])
        y = torch.tensor(self.labels[index], dtype=torch.long)

        return x, y, self.ids[index]


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


train_dataset = EmotionDataset(X_train, y_train, train_ids)
val_dataset = EmotionDataset(X_val, y_val, val_ids)
test_dataset = EmotionDataset(X_test, y_test, test_ids)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch
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


# -------------------------
# BiLSTM model
# -------------------------

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


model = BiLSTM().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -------------------------
# Evaluation
# -------------------------

def evaluate(loader):

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

            correct += (predictions == y).sum().item()
            total += len(y)

    return total_loss / total, correct / total


# -------------------------
# Training
# -------------------------

best_val_loss = float("inf")
best_model_path = DATA_DIR / "best_bilstm.pt"

print("\nTraining...\n")

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x, lengths, y, _ in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x, lengths)

        loss = criterion(output, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(y)

        predictions = output.argmax(dim=1)

        correct += (predictions == y).sum().item()
        total += len(y)

    train_loss = total_loss / total
    train_accuracy = correct / total

    val_loss, val_accuracy = evaluate(val_loader)

    print(
        f"Epoch {epoch:02d} | "
        f"Train loss: {train_loss:.3f} | "
        f"Train acc: {train_accuracy:.3f} | "
        f"Val loss: {val_loss:.3f} | "
        f"Val acc: {val_accuracy:.3f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            best_model_path
        )


# -------------------------
# Test ONCE after training
# -------------------------

model.load_state_dict(
    torch.load(best_model_path)
)

test_loss, test_accuracy = evaluate(test_loader)

print("\nTraining complete.")
print("Best model:", best_model_path)
print(f"Test accuracy: {test_accuracy:.3f}")


# -------------------------
# Save test predictions
# -------------------------

prediction_file = DATA_DIR / "predictions_fold_01.csv"

model.eval()

with open(prediction_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "id",
        "true",
        "prediction",
        "split",
        "method",
        "seed"
    ])

    with torch.no_grad():

        for x, lengths, y, ids in test_loader:

            x = x.to(device)

            output = model(x, lengths)

            predictions = output.argmax(dim=1).cpu()

            for clip_id, true_index, pred_index in zip(
                ids,
                y,
                predictions
            ):

                writer.writerow([
                    clip_id,
                    index_to_label[true_index.item()],
                    index_to_label[pred_index.item()],
                    split["name"],
                    "mfcc_bilstm",
                    SEED
                ])

print("Predictions saved to:", prediction_file)
