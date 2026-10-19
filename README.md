# CISC 473: Speech emotion recognition

Compare MFCC + BiLSTM with frozen HuBERT and wav2vec 2.0 on eight-class RAVDESS
using speaker-independent leave-one-speaker-out (LOSO) evaluation.

- [Shared splitter](splitter/index.html): turn `id,person,label` into fixed LOSO test sets.
- [Shared scorer](scorer/index.html): score `id,true,prediction,split,method,seed` predictions.
- [Published SSL experiments](experiments/ssl/README.md): extraction, classifier code,
  matched speaker partitions, validation/test results, confusion plots, checkpoints,
  and downloadable feature caches.
- [Proposal](docs/CISC_473_proj_proposal.pdf).

The SSL publication lives on the `vu-mai` branch and contains only Vu's
HuBERT and wav2vec 2.0 code and artifacts. Anastasia owns the MFCC + BiLSTM
implementation and its results; those artifacts are not included here.
