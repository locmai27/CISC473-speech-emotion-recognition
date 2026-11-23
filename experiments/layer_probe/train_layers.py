import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
FEATURES = HERE / "hubert_layers.npz"
MANIFEST = HERE.parent / "ssl/data/validation_manifest.json"
RESULTS = HERE / "results"

LAYERS = [3, 6, 9, 12]
SEEDS = [42, 43, 44]
LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]


class EmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 8),
        )

    def forward(self, x):
        return self.network(x)


def evaluate(model, x, y):
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y).item()
        predictions = logits.argmax(dim=1).cpu().numpy()
        truth = y.cpu().numpy()

    recalls = [
        np.mean(predictions[truth == label] == label)
        for label in range(8)
        if np.any(truth == label)
    ]

    return loss, float(np.mean(recalls)), float(
        np.mean(predictions == truth)
    ), predictions


def train_fold(layer, seed, fold, features, ids, labels, device,
               epochs, patience, batch_size):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    index = {clip_id: i for i, clip_id in enumerate(ids)}

    train_idx = [index[i] for i in fold["train_ids"]]
    val_idx = [index[i] for i in fold["validation_ids"]]
    test_idx = [index[i] for i in fold["test_ids"]]

    # Fit normalization on training speakers only.
    mean = features[train_idx].mean(axis=0)
    std = features[train_idx].std(axis=0)
    std = np.maximum(std, 1e-6)

    normalized = (features - mean) / std

    x = torch.tensor(normalized, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)

    model = EmotionClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    generator = torch.Generator().manual_seed(seed)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0

    start = time.monotonic()

    for epoch in range(1, epochs + 1):
        model.train()

        order = torch.randperm(
            len(train_idx), generator=generator
        ).tolist()

        for offset in range(0, len(order), batch_size):
            batch_indices = [
                train_idx[i]
                for i in order[offset:offset + batch_size]
            ]

            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(x[batch_indices]), y[batch_indices]
            )
            loss.backward()
            optimizer.step()

        val_loss, _, _, _ = evaluate(
            model, x[val_idx], y[val_idx]
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            stale += 1

        if stale >= patience:
            break

    model.load_state_dict(best_state)

    _, uar, accuracy, predictions = evaluate(
        model, x[test_idx], y[test_idx]
    )

    elapsed = time.monotonic() - start

    result = {
        "layer": layer,
        "seed": seed,
        "fold": fold["name"],
        "test_speaker": fold["test_person"],
        "uar": uar,
        "accuracy": accuracy,
        "best_epoch": best_epoch,
        "elapsed_seconds": round(elapsed, 2),
    }

    prediction_rows = [
        {
            "layer": layer,
            "seed": seed,
            "fold": fold["name"],
            "id": clip_id,
            "true": LABELS[labels[index[clip_id]]],
            "prediction": LABELS[int(pred)],
        }
        for clip_id, pred in zip(fold["test_ids"], predictions)
    ]

    return result, prediction_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-folds", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    with np.load(FEATURES) as data:
        ids = data["ids"].tolist()
        labels = np.array(
            [LABELS.index(label) for label in data["labels"]]
        )
        layer_features = {
            layer: data[f"layer_{layer}"].copy()
            for layer in args.layers
        }

    with open(MANIFEST) as f:
        folds = json.load(f)["folds"][:args.max_folds]

    assert len(ids) == 1440
    assert len(set(ids)) == 1440

    results = []
    predictions = []

    total = len(args.layers) * len(args.seeds) * len(folds)
    completed = 0

    print(f"Device: {device}", flush=True)
    print(f"Planned training runs: {total}", flush=True)

    for layer in args.layers:
        for seed in args.seeds:
            for fold in folds:
                result, rows = train_fold(
                    layer, seed, fold,
                    layer_features[layer],
                    ids, labels, device,
                    args.epochs, args.patience,
                    args.batch_size,
                )

                results.append(result)
                predictions.extend(rows)
                completed += 1

                print(
                    f"[{completed}/{total}] "
                    f"Layer {layer} | Seed {seed} | "
                    f"{fold['name']} | "
                    f"UAR {result['uar']:.3f} | "
                    f"Accuracy {result['accuracy']:.3f} | "
                    f"{result['elapsed_seconds']}s",
                    flush=True,
                )

                # Save progress after every fold.
                with open(RESULTS / "fold_results.json", "w") as f:
                    json.dump(results, f, indent=2)

                with open(
                    RESULTS / "predictions.csv", "w", newline=""
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "layer", "seed", "fold",
                            "id", "true", "prediction"
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(predictions)

    print("SUCCESS: Training completed!", flush=True)


if __name__ == "__main__":
    main()
