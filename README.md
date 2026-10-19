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
