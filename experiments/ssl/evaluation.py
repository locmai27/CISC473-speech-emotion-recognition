"""Shared metric and speaker-partition validation for the SSL experiments."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime

import extract_features as shared

HERE = Path(__file__).resolve().parent
LOG = logging.getLogger('ser.train')
LABELS = shared.LABELS
PREDICTION_HEADER = ['id', 'true', 'prediction', 'split', 'method', 'seed']


def write_csv(path, rows, fields):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def metrics(true, prediction):
    import numpy as np
    true, prediction = np.asarray(true, dtype=int), np.asarray(prediction, dtype=int)
    if true.shape != prediction.shape or true.ndim != 1 or true.size == 0:
        raise ValueError('Metric inputs must be nonempty vectors of matching size.')
    if np.any(true < 0) or np.any(true >= 8) or np.any(prediction < 0) or np.any(prediction >= 8):
        raise ValueError('Emotion index outside 0-7.')
    cm = np.zeros((8, 8), dtype=np.int64)
    np.add.at(cm, (true, prediction), 1)
    support = cm.sum(axis=1)
    if np.any(support == 0):
        raise ValueError('All eight classes must occur in this evaluation partition.')
    recalls = np.diag(cm) / support
    return {'uar': float(recalls.mean()), 'war': float(np.trace(cm) / cm.sum()),
            'recalls': recalls.tolist(), 'confusion': cm.tolist(), 'n': int(cm.sum())}


def validation_manifest(splits, by_id, path, val_seed=42):
    """Create or verify fixed validation speakers without changing LOSO tests."""
    ids = set(by_id)
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
    else:
        folds = []
        for split in splits:
            pool_people = sorted({by_id[x]['person'] for x in split['train_ids']})
            ranked = sorted(pool_people, key=lambda p: hashlib.sha256(
                f'{val_seed}|{split["name"]}|{p}'.encode()).hexdigest())
            val_people = sorted(ranked[:3])
            folds.append({
                'name': split['name'], 'test_person': split['test_person'],
                'validation_persons': val_people,
                'training_persons': sorted(set(pool_people) - set(val_people)),
                'train_ids': [x for x in split['train_ids'] if by_id[x]['person'] not in val_people],
                'validation_ids': [x for x in split['train_ids'] if by_id[x]['person'] in val_people],
                'test_ids': split['test_ids'],
            })
        manifest = {
            'policy': 'Rank non-test speakers by SHA256 of val_seed|fold_name|speaker; first three are validation.',
            'validation_seed': val_seed,
            'matches_anastasia_validation_confirmed': False,
            'note': 'Original LOSO test sets are preserved. These validation speaker identities were generated independently; match with MFCC validation before claiming an exactly matched model comparison.',
            'folds': folds,
        }
        shared.atomic_json(path, manifest)
    folds = manifest['folds']
    if len(folds) != len(splits) or len({f['name'] for f in folds}) != len(splits):
        raise ValueError('Validation manifest does not cover each LOSO fold exactly once.')
    by_name = {f['name']: f for f in folds}
    all_test_ids = []
    for split in splits:
        fold = by_name[split['name']]
        train, val, test = map(set, (fold['train_ids'], fold['validation_ids'], fold['test_ids']))
        if test != set(split['test_ids']) or fold['test_person'] != split['test_person']:
            raise ValueError('Validation manifest changed an original LOSO test set.')
        if train | val != set(split['train_ids']) or train | val | test != ids:
            raise ValueError('Training/validation partitions do not cover the supplied training pool.')
        if train & val or train & test or val & test:
            raise ValueError('Clip overlap in validation manifest.')
        groups = [{by_id[x]['person'] for x in part} for part in (train, val, test)]
        if [len(x) for x in groups] != [20, 3, 1] or groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError('Expected 20/3/1 disjoint training/validation/test speakers.')
        if groups[2] != {split['test_person']} or groups[1] != set(fold['validation_persons']) or groups[0] != set(fold['training_persons']):
            raise ValueError('Manifest speaker lists and recording lists disagree.')
        if [len(train), len(val), len(test)] != [1200, 180, 60]:
            raise ValueError('Unexpected RAVDESS partition sizes.')
        if any(len(fold[k]) != len(set(fold[k])) for k in ['train_ids','validation_ids','test_ids']):
            raise ValueError('Duplicate recording in a manifest partition.')
        all_test_ids.extend(fold['test_ids'])
    if len(all_test_ids) != len(ids) or set(all_test_ids) != ids:
        raise ValueError('LOSO tests must cover every recording exactly once.')
    return manifest
