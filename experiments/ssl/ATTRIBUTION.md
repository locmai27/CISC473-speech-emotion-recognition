# Data, pretrained representations, and source provenance

## RAVDESS

Steven R. Livingstone and Frank A. Russo (2018), *The Ryerson Audio-Visual
Database of Emotional Speech and Song (RAVDESS)*.

- Dataset: https://zenodo.org/records/1188976
- Paper: https://doi.org/10.1371/journal.pone.0196391
- Official dataset terms: Creative Commons Attribution-NonCommercial-ShareAlike
  4.0 International: https://creativecommons.org/licenses/by-nc-sa/4.0/

The release includes derived speech-feature arrays, labels, and fitted classifier
artifacts, not the original audio. Dataset-derived feature archives are distributed
with the dataset attribution and CC BY-NC-SA 4.0 terms. See the original record
for commercial licensing. This notice does not change the dataset's terms or
grant additional rights in third-party material.

Changes from the source recordings: arithmetic-mean mono conversion, resampling
from 48 kHz to 16 kHz, checkpoint waveform normalization, and extraction of frozen
final-layer representations. The clips' supplied emotion labels and speaker IDs
were retained. Fold-fitted normalization is stored with each classifier checkpoint.

## Pretrained encoders and classifier inspiration

- wav2vec 2.0: Baevski et al. (2020), https://arxiv.org/abs/2006.11477
- HuBERT: Hsu et al. (2021), https://doi.org/10.1109/TASLP.2021.3122291
- SER Dense head inspiration: Pepino et al. (2021), https://arxiv.org/abs/2104.03502
- Checkpoints: https://huggingface.co/facebook/hubert-base-ls960 and
  https://huggingface.co/facebook/wav2vec2-base

Encoder revisions are pinned in `extract_features.py` and the cached metadata.
Pretrained encoder weights are not redistributed here; download them from the
linked model repositories and consult their terms. Attention pooling is established
prior art. Our code adapts a Dense classifier and tests pooling under our protocol;
it is not a claim of a novel attention architecture.

## Measured code and portable interfaces

The exact measured-run scripts are retained byte-for-byte in
`results/matched_loso/source_snapshot/`, and in the selected-checkpoint archive.
Their historic local path defaults explain the recorded fingerprints. Use the
portable entry points at this folder's top level for a fresh checkout; their new
source hashes must not be substituted into the original recorded run configs.

No new blanket open-source license is imposed on the repository by this
publication. Dataset terms above apply to the dataset-derived feature artifacts;
third-party licenses and any later project-code license remain separate.
