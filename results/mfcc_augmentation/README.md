# MFCC + BiLSTM Acoustic Data Augmentation

## Experimental Setup

- Dataset: RAVDESS (1,440 recordings, 24 speakers, 8 emotions)
- Evaluation: 24-fold leave-one-speaker-out cross-validation
- Random seed: 42
- Features: 40 MFCCs, 16 kHz sampling rate
- Model: Two-layer bidirectional LSTM
- Epochs: 20
- Batch size: 32
- Learning rate: 0.001

## Augmentation

Each original training recording produces three additional samples:

- Pitch shifting: +/- 2 semitones
- Time stretching: 0.9x or 1.1x
- Gaussian noise: 20-30 dB SNR

Training samples increase from 1,200 to 4,800 per fold.
Validation and test recordings remain unchanged.
Normalization statistics are computed from training data only.

## Results

| Metric | Baseline | Augmented | Change |
|---|---:|---:|---:|
| Accuracy | 31.67% | 35.90% | +4.23 pp |
| UAR | 30.01% | 34.44% | +4.43 pp |

## Findings

Augmentation improved recall for five of eight emotions.
The largest improvements were observed for happy,
fearful, and disgust. Recall decreased for calm,
angry, and surprised.

## Limitations

- Only one random seed was evaluated.
- Individual augmentation effects were not isolated.
- Augmentation increases training updates per epoch.
- Statistical significance has not been established.

## Reproduction

From the repository root:

SER_DATA_DIR=/path/to/data python scripts/train_bilstm_augmented.py

## Three-seed results (seeds 42, 123, 999)

Seeds 123 and 999 were added with the same script (SER_SEED), protocol, and settings as seed 42. They ran on a Colab T4 GPU (SER_DEVICE=cuda); seed 42 ran on CPU. Baselines are the published MFCC + BiLSTM runs with the same seeds.

| Seed | Baseline UAR | Augmented UAR | UAR gain | Baseline WAR | Augmented WAR |
|---:|---:|---:|---:|---:|---:|
| 42 | 30.01% | 34.44% | +4.43 pp | 31.67% | 35.90% |
| 123 | 28.45% | 36.78% | +8.33 pp | 30.00% | 38.82% |
| 999 | 29.43% | 39.84% | +10.42 pp | 31.25% | 41.25% |
| Mean ± SD | 29.30 ± 0.79% | 37.02 ± 2.71% | 7.73 ± 3.04 pp | 30.97 ± 0.87% | 38.66 ± 2.68% |

Per-seed predictions: predictions_mfcc_bilstm_augmented_seed{42,123,999}.csv; summary with per-emotion recall: summary_3seeds.csv. Augmented training still uses 4x training examples (and optimizer steps) per epoch.
