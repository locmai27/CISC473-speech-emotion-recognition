# CISC 473: Speech emotion recognition

Compare MFCC + BiLSTM with frozen HuBERT and wav2vec 2.0 on eight-class RAVDESS
using speaker-independent leave-one-speaker-out (LOSO) evaluation.

- [MFCC + BiLSTM scripts](scripts/README.md): Anastasia's preprocessing and training code.
- [MFCC + BiLSTM results](results/): per-seed predictions, fold results, and confusion matrix.
- [Shared dataset labels and outer splits](data/): clip IDs, speaker IDs, emotion labels,
  and the 24 LOSO test partitions.
- [Published SSL experiments](experiments/ssl/README.md): extraction, classifier code,
  matched speaker partitions, validation/test results, confusion plots, checkpoints,
  and downloadable feature caches.
- [Proposal](docs/CISC_473_proj_proposal.pdf).

`main` combines the `vu-mai` and `anastasia-tumanov` branches. Vu owns the shared
evaluation infrastructure and SSL experiments; Anastasia owns the MFCC + BiLSTM pipeline.
Each pipeline keeps its original paths and outputs.

The shared outer split holds out one test speaker. The complete training scripts
reserve the next three speaker IDs cyclically for validation and use the other
20 speakers for training. SSL seeds are 42, 43, and 44; published BiLSTM seeds are
42, 123, and 999. These seed sets are reported separately.

The root `data/` clip table and splits match the copies distributed with the SSL
experiments. SSL feature caches and classifier weights remain available in the
[matched LOSO release](https://github.com/locmai27/CISC473-speech-emotion-recognition/releases/tag/ssl-midterm-matched-loso-2026-10-19).

Run checks manually from the repository root after installing the SSL dependencies:

```text
python -m compileall -q scripts experiments/ssl
python -m unittest discover -s experiments/ssl/tests -v
```

These checks do not launch model training.

## Reproducing the reported results

Install dependencies once from the repository root: `pip install -r requirements.txt`
(see [experiments/ssl/README.md](experiments/ssl/README.md) for the CUDA build of PyTorch).
Each experiment then runs with one command. Seeds are fixed and selectable with `SER_SEED`.

| Result | Command (from the repository root unless noted) | Outputs |
|---|---|---|
| RAVDESS audio | `python experiments/ssl/download_audio.py` (Zenodo download, checksum-verified) | `experiments/ssl/artifacts/audio/` |
| MFCC + BiLSTM, seeds 42/123/999 | `cd scripts && SER_SEED=42 python train_bilstm_all_folds.py` (repeat for 123, 999) | `data/`; published copies in `results/` |
| MFCC + BiLSTM + augmentation, seeds 42/123/999 | `SER_DATA_DIR=data SER_SEED=42 python scripts/train_bilstm_augmented.py` (repeat for 123, 999) | `data/`; published copies in `results/mfcc_augmentation/` |
| HuBERT / wav2vec 2.0, mean vs attention pooling | see [experiments/ssl/README.md](experiments/ssl/README.md) (`frame_experiment.py`) | `experiments/ssl/results/matched_loso/` |
| HuBERT layer probing + augmentation | open [the Colab notebook](experiments/ssl/notebooks/hubert_layer_probe_colab.ipynb) and run all cells, or run the `layer_probe.py` subcommands it lists | `experiments/ssl/results/layer_probe/` |

The MFCC scripts expect the RAVDESS `.wav` files under `data/RAVDESS/` (any subfolder layout);
linking `data/RAVDESS` to `experiments/ssl/artifacts/audio` after the download works.
Trained SSL and layer-probing classifier weights are in the GitHub releases
[`ssl-midterm-matched-loso-2026-10-19`](https://github.com/locmai27/CISC473-speech-emotion-recognition/releases/tag/ssl-midterm-matched-loso-2026-10-19)
and [`ssl-layer-probe-2026-11-25`](https://github.com/locmai27/CISC473-speech-emotion-recognition/releases/tag/ssl-layer-probe-2026-11-25).
MFCC + BiLSTM models are small and retrain from scratch with the commands above.
