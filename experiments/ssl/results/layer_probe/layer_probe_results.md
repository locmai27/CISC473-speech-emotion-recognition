# HuBERT selected-layer probing and augmentation

Completed fits: 432.

| Layer | Training | UAR (%) mean ± SD | WAR (%) mean ± SD | Seeds |
|---:|---|---:|---:|---:|
| 6 | aug | 72.94 ± 0.90 | 73.40 ± 0.84 | 3 |
| 12 | aug | 64.02 ± 0.49 | 64.49 ± 0.38 | 3 |
| 3 | clean | 68.27 ± 0.51 | 69.19 ± 0.49 | 3 |
| 6 | clean | 74.22 ± 0.73 | 74.95 ± 0.86 | 3 |
| 9 | clean | 68.03 ± 0.85 | 68.06 ± 0.85 | 3 |
| 12 | clean | 66.64 ± 0.10 | 67.38 ± 0.31 | 3 |

## Augmentation gain (augmented − clean UAR, percentage points)

| Layer | Gain mean ± SD | Per seed |
|---:|---:|---|
| 6 | -1.28 ± 0.67 | -1.11, -0.72, -2.02 |
| 12 | -2.63 ± 0.58 | -2.99, -2.93, -1.95 |

Interaction, gain(L6) - gain(L12), UAR, paired by seed: +1.35 pp (per seed +1.89, +2.21, -0.07; same sign for every seed: False). No significance test is claimed.

Layer-12 control: rerun UAR 66.64% vs published 67.12%; test-prediction agreement 95.4%.

## Protocol

Frozen facebook/hubert-base-ls960 (pinned revision); hidden_states[L] from the Transformer, full recordings, no padding.
Same Dense(128)x2 + masked mean pooling head and training procedure for every layer; a fresh classifier per layer, seed and fold.
24 LOSO folds (20 train / 3 cyclic-next validation / 1 test speaker), seeds 42, 43, 44, training-only normalization,
minimum-validation-loss checkpoint (patience 4, max 20 epochs). Test speakers scored only after selection.
Augmentation: pitch ±2 semitones, time stretch 0.9/1.1, Gaussian noise 20–30 dB SNR, one copy each per clip,
applied to the waveform before HuBERT; training speakers only; validation/test audio clean. Exposure matched to clean
training (1,200 samples per epoch, version drawn uniformly from original + 3 copies).
Layers and policy were fixed in advance (not chosen on test data). This is a selected-layer comparison (4 of 12 layers).
SDs are across three training seeds on one dataset, not independent replications.
