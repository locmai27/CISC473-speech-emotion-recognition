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

