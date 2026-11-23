from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

SSL_DIR = Path(__file__).resolve().parents[1] / "ssl"
sys.path.insert(0, str(SSL_DIR))

from extract_features import read_clip_table, load_audio

AUDIO_DIR = Path.home() / "Desktop/CISC473-project/data/RAVDESS"
CLIP_TABLE = SSL_DIR / "data/clip_table.csv"
OUTPUT = Path(__file__).resolve().parent / "hubert_layers.npz"

MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--facebook--hubert-base-ls960"
    / "snapshots/dba3bb02fda4248b6e082697eee756de8fe8aa8a"
)

LAYERS = [3, 6, 9, 12]


def main():
    rows = read_clip_table(CLIP_TABLE, AUDIO_DIR)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}", flush=True)
    print(f"Recordings: {len(rows)}", flush=True)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    model = HubertModel.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.config.apply_spec_augment = False
    model.eval().to(device)

    features = {
        layer: np.empty((len(rows), 768), dtype=np.float32)
        for layer in LAYERS
    }
    ids = []
    labels = []
    speakers = []

    start = time.monotonic()

    for index, row in enumerate(rows):
        audio, _, _ = load_audio(row["path"])

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        with torch.inference_mode():
            outputs = model(
                input_values=inputs.input_values.to(device),
                output_hidden_states=True,
            )

            for layer in LAYERS:
                pooled = outputs.hidden_states[layer].mean(dim=1)
                features[layer][index] = (
                    pooled[0].float().cpu().numpy()
                )

        ids.append(row["id"])
        labels.append(row["label"])
        speakers.append(row["person"])

        if (index + 1) % 25 == 0:
            elapsed = time.monotonic() - start
            print(
                f"{index + 1}/{len(rows)} recordings "
                f"| {elapsed:.1f}s elapsed",
                flush=True,
            )

    np.savez_compressed(
        OUTPUT,
        ids=np.array(ids),
        labels=np.array(labels),
        speakers=np.array(speakers),
        **{f"layer_{layer}": features[layer] for layer in LAYERS},
    )

    print(f"SUCCESS: Saved features to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
