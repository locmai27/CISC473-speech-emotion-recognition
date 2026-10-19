# Frozen speech encoders: reproducible SSL experiments

This folder publishes Vu's HuBERT and wav2vec 2.0 experiments. The browser splitter
and scorer remain separate from training. These results adapt a published Dense
head to final-layer, eight-class LOSO evaluation; they do not reproduce a paper's
original numerical result or establish attention pooling as a new architecture.

## Final matched results

| Configuration | UAR (%) | WAR / accuracy (%) |
|---|---:|---:|
| HuBERT + Dense mean pooling | 67.12 +/- 0.45 | 67.73 +/- 0.53 |
| HuBERT + Dense attention pooling | 64.87 +/- 1.30 | 65.51 +/- 1.04 |
| wav2vec 2.0 + Dense mean pooling | 49.67 +/- 0.65 | 50.58 +/- 0.70 |
| wav2vec 2.0 + Dense attention pooling | 50.26 +/- 1.05 | 50.74 +/- 0.91 |

Each seed pools 1,440 out-of-fold predictions across all 24 held-out speakers.
Values are mean and **sample** standard deviation across seeds 42,43,44.
There are 288 fits, 17,280 test predictions, and 51,840 replayed validation
predictions. Validation speakers recur across folds: those predictions are
model-selection data, not an independent held-out test benchmark.

For test speaker `s`, the next three IDs cyclically from 01 through 24 validate;
the remaining 20 train. Fold sizes are 1,200/180/60 clips. The original shared
test sets are unchanged. This matches the speaker rule confirmed for the teammate
MFCC implementation; that implementation was not independently audited here.
Identical numeric seeds across different architectures are not required. The
teammate report uses different seeds, which should be reported explicitly.

Attention minus mean UAR is -2.257 points for HuBERT (negative for all three seeds)
and +0.586 points for wav2vec 2.0 (positive for two seeds, negative for one).
No statistical significance is claimed.

## What is in Git and what is in the release

| Location | Contents |
|---|---|
| Python files in this folder | Portable extraction, training, evaluation and plotting interfaces |
| `data/` | Original clip table, unchanged outer LOSO splits, confirmed validation manifest, dataset audit |
| `results/matched_loso/` | Final summaries, predictions, 288 histories, validation exports, audits, plots, exact original source snapshot |
| `results/historical/` | Earlier independent-validation and pooled-linear experiments, clearly separated |
| `artifact_manifest.json` | Release URLs, archive sizes, SHA-256 checksums and artifact IDs |
| GitHub release assets | Full frame features, all selected SSL classifier weights, historical fitted artifacts and pooled features |

The frame caches contain 1,440 variable-length float32 arrays per encoder, each
`frames x 768`. They retain clip IDs, emotion labels, speaker IDs, original audio
SHA-256 values, and extraction fingerprints. **They are the full frame sequences
used in the Dense experiments**, not the older time-averaged linear features.

The release is
[ssl-midterm-matched-loso-2026-10-19](https://github.com/locmai27/CISC473-speech-emotion-recognition/releases/tag/ssl-midterm-matched-loso-2026-10-19).
Large arrays and checkpoints are release downloads rather than Git blobs. Raw
audio and pretrained encoder weights are downloaded from their original sources.
See [ATTRIBUTION.md](ATTRIBUTION.md) for dataset terms and references.

## Environment

Commands below run from the repository root. Python 3.11 was used. Create a
virtual environment, activate it using your shell, and install one PyTorch build:

```text
python -m venv .venv
```

Windows PowerShell activation: `.\.venv\Scripts\Activate.ps1`.
Linux/macOS activation: `source .venv/bin/activate`.

For NVIDIA CUDA 12.6:

```text
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r experiments/ssl/requirements.txt
```

For CPU, use `https://download.pytorch.org/whl/cpu` instead. `--device cpu` works
for extraction and training. Final reported fits used an RTX 4070 and
PyTorch 2.7.1+cu126; the released fixed features were extracted on CPU. Versions
are recorded in the result and feature metadata.

## Download data and inspect existing results

```text
python experiments/ssl/download_audio.py
python experiments/ssl/download_artifacts.py --assets hubert_features wav2vec2_features matched_checkpoints
```

The audio downloader verifies the original RAVDESS archive checksum and clip
IDs/labels. The artifact downloader verifies every archive's SHA-256 and rejects
unsafe extraction paths or different existing outputs. Downloads unpack under
`experiments/ssl/artifacts/`, which is ignored by Git. Roughly 1.6 GiB of release
downloads are needed for both feature caches and the selected matched weights;
allow additional space for unpacked arrays, original audio and your environment.

Replay the released checkpoints, verify training-only normalization and minimum
validation-loss selection, and regenerate validation histories/results:

```text
python experiments/ssl/evaluate_checkpoints.py
```

This verifies 288 selected models, reproduces 17,280 test predictions, and exports
`validation_predictions.csv`, `validation_metrics.csv`, and `training_history.csv`.
The default result location is `artifacts/runs/matched_loso/`, populated by the
checkpoint archive. The same lightweight results are committed in
`results/matched_loso/`. Exact cross-device floating-point agreement is not
guaranteed; the published replay audit uses the same CUDA training backend.

## Train the same configurations from the cached features

Use a **fresh output directory** rather than trying to resume the original
checkpoints with the portable source's different code fingerprint:

```text
python experiments/ssl/frame_experiment.py --phase train --device cuda --results-dir experiments/ssl/artifacts/runs/reproduction
python experiments/ssl/evaluate_checkpoints.py --results-dir experiments/ssl/artifacts/runs/reproduction
```

The training algorithm and head operations are preserved from the measured run.
Portable interface changes have new source hashes; the exact original code and
config hashes remain in `results/matched_loso/source_snapshot/` and `config.json`.
Do not edit or relabel those original fingerprints. Interrupted portable runs
resume only when their configuration fingerprint matches.

Both heads apply two frame-wise Linear(128), ReLU, Dropout(0.2) stages, masked
mean or learned 128-to-1 attention pooling, and eight output logits. Attention
scores start at zero, and common head weights start identically within each
encoder/seed/fold pair. Only classifier weights are trained. Adam uses learning
rate 0.001, zero weight decay, batch size 32, at most 20 epochs, and cross-entropy.
Early stopping monitors validation loss with patience four and selects the
minimum-validation-loss checkpoint. Every fold refits frame normalization on its
20 training speakers. Padding is excluded from pooling. There is no augmentation,
truncation, encoder fine-tuning or layer search in these completed runs.

## Regenerate features and speaker partitions

```text
python experiments/ssl/prepare_splits.py
python experiments/ssl/frame_experiment.py --phase extract --device cpu --features-dir experiments/ssl/artifacts/reextracted_frames
```

The published manifest is already included in `data/validation_manifest.json`.
The splitter command writes an independently verifiable cyclic manifest under
`artifacts/`; its speaker membership matches the published one.

Extraction downloads pinned `facebook/hubert-base-ls960` and
`facebook/wav2vec2-base` revisions, converts recordings to mono at 16 kHz, applies
the checkpoint waveform normalization, freezes the encoder in evaluation mode,
and retains Transformer layer 12. It reads complete recordings without padding,
augmentation or truncation. New feature outputs have their own provenance
fingerprints. To train on them, pass the same `--features-dir` to training and
evaluation. The original release caches are not overwritten.

`--phase extract --limit 1` checks one recording per encoder. A training smoke
check can use `--epochs 2 --max-fold-pairs 1` with a fresh results directory;
partial runs are not included in a complete LOSO summary.

## Generate confusion matrices for every available model

The plotter accepts the shared scorer's exact prediction header:
`id,true,prediction,split,method,seed`. It keeps methods separate, detects duplicate
test predictions, checks comparable seed coverage, and reports seed-level scores.

```text
python experiments/ssl/plot_confusions.py --predictions experiments/ssl/results/matched_loso/predictions.csv --output-dir experiments/ssl/artifacts/plots
```

Each method gets raw-count and row-normalized PNG/PDF figures, labelled CSV
matrices, and a metrics JSON. Labels use the same alphabetical order in every
new plot. Three-seed test plots aggregate 4,320 predictions over the same 1,440
recordings, not 4,320 independent recordings.

Additional scorer-format prediction files can be passed to `--predictions`
to generate multiple methods in one command. Model names must be distinct.
Use `--partition validation` for validation CSVs; those legitimately repeat a clip
in several folds and must not be interpreted as held-out LOSO test results.

```text
python experiments/ssl/plot_training_curves.py --results-dir experiments/ssl/results/matched_loso
python -m unittest discover -s experiments/ssl/tests -v
```

The learning-curve example is fixed to test speaker 01 and seed 42. It was selected
by ID rather than performance. Tests cover speaker leakage, cyclic wraparound,
padding invariance, initially uniform attention, separate-method scoring,
duplicate-prediction rejection, aggregate-count loading, and safe artifact extraction.

## Historical experiments and scope

`results/historical/frame_results/` preserves the earlier four-configuration run
with independently chosen validation speakers, including excluded CPU pilots.
`results/historical/results_provisional/` contains earlier pooled-feature linear
baselines with different normalization and model selection. Their scores must
not be mixed with the matched final results or treated as an isolated head ablation.
Full historical weights and pooled caches are available using:

```text
python experiments/ssl/download_artifacts.py --assets historical
```

HuBERT layer probing and pitch/time/noise augmentation remain future work.
Nothing here demonstrates an exact published-score reproduction. The checkpoint
replay audit covers the SSL models. The merged repository also includes the
teammate's MFCC source and predictions under the root `scripts/` and `results/`
folders; their inclusion does not extend that checkpoint audit to the BiLSTM.
