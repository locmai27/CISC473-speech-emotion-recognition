# MFCC + BiLSTM pipeline

Anastasia's uploaded source files are retained at their original paths.
`train_bilstm_all_folds.py` is the full 24-fold LOSO training entry point.
It reserves the next three speaker IDs cyclically for validation, normalizes
MFCCs using training data only, and selects the lowest-validation-loss model.

The scripts use paths relative to the working directory. Run them from `scripts/`,
with recordings under `data/RAVDESS/` and labels/splits under `data/`:

```text
cd scripts
python train_bilstm_all_folds.py
```

Running this command launches training. It requires PyTorch, NumPy, and librosa.
The uploaded script currently sets `SEED = 999`; the published results in
`../results/` cover seeds 42, 123, and 999. The script writes its predictions and
fold results to `../data/` with unversioned filenames. Preserve each run separately
before changing the seed; the published per-seed CSVs remain in `../results/`.

The other preprocessing and single-fold scripts are exploratory helpers.
Some select the first three non-test speakers instead of using the complete
training script's cyclic rule. Use `train_bilstm_all_folds.py` and its published
fold-results CSVs for the final LOSO protocol.

The uploaded branch does not include classifier checkpoints or a pinned MFCC
dependency environment. The merge preserves the supplied source and results;
it does not claim to have retrained the BiLSTM models.
