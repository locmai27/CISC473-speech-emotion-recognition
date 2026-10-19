# Agents

This repo is the CISC 473 speech emotion recognition project. The proposal is `docs/CISC_473_proj_proposal.pdf`.

Vu owns the shared splitter and scorer, plus the HuBERT and wav2vec paths. Anastasia owns the clip table, audio preparation, augmentation, and the MFCC + BiLSTM path. Vu's published SSL experiment code, results, and artifact manifest live in `experiments/ssl/`. Keep the SSL training pipeline separate from Anastasia's MFCC pipeline. Do not add a shared training UI or put training into the browser splitter or scorer.

## Splitter

`splitter/index.html` reads one CSV and writes `splits.json`.

The header must be exactly `id,person,label`. Reject any other header. Leave one person out, one split per person, sorted so the same table always produces the same file. A person must not appear in both train and test. Do not add a column mapper.

## Scorer

`scorer/index.html` reads predictions and writes `scores.json`. It does not train and it does not read audio.

The header must be exactly `id,true,prediction,split,method,seed`. Score each `method` only on its own rows. One method shows that method's UAR, WAR, and confusion table. Several methods show a comparison, still with separate scores.

UAR is the mean recall of each true label. WAR is recall weighted by how many clips have that label. Several seeds are the mean across seeds, with the sample standard deviation.

## Do not

- Reshuffle splits between runs.
- Score the test clips before training for that split has finished.
- Mix two methods into one UAR.
- Put layer picking, early stopping, or augmentation into the splitter or scorer.
