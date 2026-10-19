from pathlib import Path
import csv

RAVDESS_DIR = Path("../data/RAVDESS")
OUTPUT_FILE = Path("../data/clip_table.csv")

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

rows = []

for wav_file in sorted(RAVDESS_DIR.rglob("*.wav")):
    clip_id = wav_file.stem
    parts = clip_id.split("-")

    emotion_code = parts[2]
    actor_id = parts[6]

    rows.append({
        "id": clip_id,
        "person": actor_id,
        "label": emotion_map[emotion_code],
    })

with OUTPUT_FILE.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "person", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Created {OUTPUT_FILE}")
print(f"Found {len(rows)} audio clips")
