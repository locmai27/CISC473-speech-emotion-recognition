"""Train frozen-feature linear SER classifiers using the supplied LOSO folds."""
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


def load_embeddings(path, by_id):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        ids = data['ids'].tolist()
        if len(ids) != len(set(ids)) or set(ids) != set(by_id):
            raise ValueError(f'Full-dataset extraction must finish before training: {path}')
        if data['features'].shape != (len(ids), 1, 768) or data['layer_indices'].tolist() != [12]:
            raise ValueError('Initial classifiers require final-layer features with shape (1440,1,768).')
        for i, clip_id in enumerate(ids):
            if data['labels'][i] != by_id[clip_id]['label'] or data['persons'][i] != by_id[clip_id]['person']:
                raise ValueError(f'Cached label or speaker mismatch for {clip_id}')
        features = data['features'][:, 0, :].astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError('Non-finite embeddings.')
        fingerprint = str(data['fingerprint'])
    return features, ids, fingerprint


def fit_standardizer(train):
    import numpy as np
    # Statistics are fitted only to training clips, never validation/test clips.
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


def train_run(method, seed, fold, features, id_to_index, by_id, args, device, fingerprint):
    import numpy as np
    import torch
    from torch import nn
    run_dir = args.results_dir / 'runs' / method / f'seed_{seed}' / fold['name']
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path, prediction_path = run_dir / 'result.json', run_dir / 'predictions.csv'
    if result_path.exists() and prediction_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result['fingerprint'] != fingerprint:
            raise ValueError(f'Training settings changed: {run_dir}. Choose a fresh --results-dir.')
        if shared.sha256(prediction_path) != result['predictions_sha256']:
            raise ValueError(f'Cached predictions changed: {prediction_path}')
        with prediction_path.open(encoding='utf-8', newline='') as handle:
            predictions = list(csv.DictReader(handle))
        if len(predictions) != 60 or {p['id'] for p in predictions} != set(fold['test_ids']):
            raise ValueError('Cached predictions do not match this held-out speaker.')
        LOG.info('[RESUME] %s seed=%d %s | UAR=%.4f WAR=%.4f', method, seed, fold['name'], result['test']['uar'], result['test']['war'])
        return result, predictions
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    arrays = {}
    for key in ['train_ids', 'validation_ids', 'test_ids']:
        indices = [id_to_index[x] for x in fold[key]]
        arrays[key] = features[indices]
    mean, std = fit_standardizer(arrays['train_ids'])
    x = {key: torch.from_numpy((values - mean) / std).to(device) for key, values in arrays.items()}
    y = {key: torch.tensor([LABELS.index(by_id[clip_id]['label']) for clip_id in fold[key]], dtype=torch.long, device=device) for key in arrays}
    model = nn.Linear(768, 8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_uar, best_loss, best_epoch = -1.0, float('inf'), 0
    history = []
    started = time.monotonic()
    LOG.info('[TRAIN] %s seed=%d %s | train=%d val=%d test=%d | validation speakers=%s',
             method, seed, fold['name'], len(fold['train_ids']), len(fold['validation_ids']), len(fold['test_ids']), fold['validation_persons'])
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(fold['train_ids']), generator=generator)
        loss_total = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x['train_ids'][indices]), y['train_ids'][indices])
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.inference_mode():
            logits = model(x['validation_ids'])
            val_loss = float(criterion(logits, y['validation_ids']))
            val_scores = metrics(y['validation_ids'].cpu().numpy(), logits.argmax(dim=1).cpu().numpy())
        item = {'epoch': epoch, 'train_loss': loss_total / len(order), 'validation_loss': val_loss,
                'validation_uar': val_scores['uar'], 'validation_war': val_scores['war']}
        history.append(item)
        if val_scores['uar'] > best_uar + 1e-12 or (abs(val_scores['uar'] - best_uar) <= 1e-12 and val_loss < best_loss):
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_uar, best_loss, best_epoch = val_scores['uar'], val_loss, epoch
        LOG.info('[EPOCH] %s seed=%d %s %02d/%d | loss=%.4f | val UAR=%.4f WAR=%.4f | best epoch=%d',
                 method, seed, fold['name'], epoch, args.epochs, item['train_loss'], val_scores['uar'], val_scores['war'], best_epoch)
    # First use of held-out-speaker features for prediction occurs only after training/model selection.
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        prediction = model(x['test_ids']).argmax(dim=1).cpu().numpy()
    test_scores = metrics(y['test_ids'].cpu().numpy(), prediction)
    predictions = [{'id': clip_id, 'true': by_id[clip_id]['label'], 'prediction': LABELS[int(pred)],
                    'split': fold['name'], 'method': method, 'seed': seed}
                   for clip_id, pred in zip(fold['test_ids'], prediction)]
    write_csv(prediction_path, predictions, PREDICTION_HEADER)
    checkpoint = {'model_state_dict': best_state, 'standardization_mean': torch.from_numpy(mean),
                  'standardization_std': torch.from_numpy(std), 'labels': LABELS,
                  'input_size': 768, 'output_size': 8, 'best_epoch': best_epoch,
                  'seed': seed, 'split': fold['name'], 'fingerprint': fingerprint}
    checkpoint_temp = run_dir / 'classifier.pt.tmp'
    torch.save(checkpoint, checkpoint_temp)
    checkpoint_temp.replace(run_dir / 'classifier.pt')
    result = {'fingerprint': fingerprint, 'method': method, 'seed': seed, 'split': fold['name'],
              'test_person': fold['test_person'], 'validation_persons': fold['validation_persons'],
              'best_epoch': best_epoch, 'best_validation_uar': best_uar, 'history': history,
              'test': test_scores, 'elapsed_seconds': time.monotonic() - started,
              'predictions_sha256': shared.sha256(prediction_path)}
    shared.atomic_json(result_path, result)
    LOG.info('[TEST] %s seed=%d %s | selected epoch=%d | UAR=%.4f WAR=%.4f | %.1fs',
             method, seed, fold['name'], best_epoch, test_scores['uar'], test_scores['war'], result['elapsed_seconds'])
    return result, predictions


def aggregate(predictions, seeds):
    import numpy as np
    per_seed = []
    for seed in seeds:
        rows = [p for p in predictions if int(p['seed']) == seed]
        if len(rows) != 1440 or len({p['id'] for p in rows}) != 1440:
            raise ValueError('Each seed must have exactly 1440 out-of-fold predictions.')
        scores = metrics([LABELS.index(p['true']) for p in rows], [LABELS.index(p['prediction']) for p in rows])
        per_seed.append({'seed': seed, **scores})
    pooled = metrics([LABELS.index(p['true']) for p in predictions], [LABELS.index(p['prediction']) for p in predictions])
    return {'uar_mean': float(np.mean([p['uar'] for p in per_seed])),
            'uar_std': float(np.std([p['uar'] for p in per_seed], ddof=1)),
            'war_mean': float(np.mean([p['war'] for p in per_seed])),
            'war_std': float(np.std([p['war'] for p in per_seed], ddof=1)),
            'seeds': seeds, 'folds_per_seed': 24, 'clips_per_seed': 1440,
            'per_seed': per_seed, 'pooled_confusion': pooled['confusion']}


def confusion_plot(method, matrix, output_dir, seed_count):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cm = np.asarray(matrix)
    normalized = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
    for row in range(8):
        for col in range(8):
            ax.text(col, row, f'{normalized[row,col]:.2f}', ha='center', va='center',
                    color='white' if normalized[row,col] > 0.55 else '#172b4d', fontsize=9)
    ax.set(xticks=range(8), yticks=range(8), xticklabels=LABELS, yticklabels=LABELS,
           xlabel='Predicted emotion', ylabel='True emotion',
           title=f'{method}: held-out-speaker confusion\n24 LOSO folds, {seed_count} seeds; rows normalized')
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
    fig.colorbar(im, ax=ax, label='Fraction within true emotion')
    fig.text(0.5, 0.015, f'Counts pool {seed_count} predictions per recording; original dataset has 1,440 unique clips.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_dir / f'{method}_confusion.png', dpi=180)
    plt.close(fig)
    write_csv(output_dir / f'{method}_confusion_counts.csv',
              [{'true': LABELS[i], **{label: int(cm[i,j]) for j,label in enumerate(LABELS)}} for i in range(8)], ['true', *LABELS])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', type=Path, default=HERE / 'features')
    parser.add_argument('--clip-table', type=Path, default=HERE.parent / 'ravdess/clip_table.csv')
    parser.add_argument('--splits', type=Path, default=HERE.parent / 'ravdess/splits.json')
    parser.add_argument('--results-dir', type=Path, default=HERE / 'results')
    parser.add_argument('--validation-manifest', type=Path)
    parser.add_argument('--models', choices=['both','hubert','wav2vec2'], default='both')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42,43,44])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 3:
        parser.error('Supply at least three distinct training seeds.')
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error('Epochs, batch size, and learning rate must be positive.')
    args.results_dir.mkdir(parents=True, exist_ok=True)
    shared.setup_logging(HERE / 'logs')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device cpu or install GPU PyTorch.')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    with args.clip_table.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['id','person','label']:
            raise ValueError('Unexpected clip table header.')
        table = list(reader)
    by_id = {r['id']: r for r in table}
    if len(table) != len(by_id) or len(table) != 1440:
        raise ValueError('Expected 1440 unique RAVDESS clips.')
    splits = json.loads(args.splits.read_text(encoding='utf-8-sig'))
    if len(splits) != 24:
        raise ValueError('Expected the supplied 24 LOSO folds.')
    manifest_path = args.validation_manifest or args.results_dir / 'validation_manifest.json'
    manifest = validation_manifest(splits, by_id, manifest_path)
    if not manifest.get('matches_anastasia_validation_confirmed', False):
        LOG.warning('LOSO tests match supplied splits. Validation speakers are independently fixed; exact MFCC validation matching is NOT confirmed. Manifest: %s', manifest_path)
    LOG.info('TRAINING CONFIG | Linear(768,8), Adam lr=%g, batch=%d, epochs=%d, seeds=%s, device=%s', args.learning_rate, args.batch_size, args.epochs, args.seeds, device)
    LOG.info('Select highest validation UAR; ties use lower validation loss. Test predictions are produced after selection only.')
    names = list(shared.MODELS) if args.models == 'both' else [args.models]
    all_predictions, all_runs, method_summaries = [], [], {}
    for name in names:
        method = f'{name}_final_linear'
        feature_path = args.features_dir / name / 'last/embeddings.npz'
        features, ids, feature_fingerprint = load_embeddings(feature_path, by_id)
        config = {'model': name, 'feature_fingerprint': feature_fingerprint,
                  'features_sha256': shared.sha256(feature_path), 'script_sha256': shared.sha256(__file__),
                  'splits_sha256': shared.sha256(args.splits), 'manifest_sha256': shared.sha256(manifest_path),
                  'clip_table_sha256': shared.sha256(args.clip_table), 'seeds': args.seeds,
                  'epochs': args.epochs, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
                  'head': 'Linear(768,8)', 'optimizer': 'Adam', 'weight_decay': 0.0,
                  'loss': 'unweighted_cross_entropy', 'standardization': 'training_mean_and_population_std',
                  'selection': 'highest_validation_UAR_then_lowest_validation_cross_entropy',
                  'torch': torch.__version__, 'device': str(device)}
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        settings_path = args.results_dir / f'{method}_settings.json'
        if settings_path.exists() and json.loads(settings_path.read_text()) != config:
            raise ValueError('Existing training settings differ; choose a fresh --results-dir.')
        shared.atomic_json(settings_path, config)
        id_to_index = {clip_id: i for i,clip_id in enumerate(ids)}
        method_predictions = []
        for seed in args.seeds:
            for fold in manifest['folds']:
                result, predictions = train_run(method, seed, fold, features, id_to_index, by_id, args, device, fingerprint)
                all_runs.append({'method': method, 'seed': seed, 'split': fold['name'],
                                 'uar': result['test']['uar'], 'war': result['test']['war'],
                                 'best_epoch': result['best_epoch'], 'validation_uar': result['best_validation_uar']})
                method_predictions.extend(predictions)
        summary = aggregate(method_predictions, args.seeds)
        method_summaries[method] = summary
        all_predictions.extend(method_predictions)
        confusion_plot(method, summary['pooled_confusion'], args.results_dir, len(args.seeds))
        LOG.info('[MODEL DONE] %s | UAR %.4f +/- %.4f | WAR %.4f +/- %.4f', method,
                 summary['uar_mean'], summary['uar_std'], summary['war_mean'], summary['war_std'])
    write_csv(args.results_dir / 'predictions.csv', all_predictions, PREDICTION_HEADER)
    write_csv(args.results_dir / 'fold_metrics.csv', all_runs, ['method','seed','split','uar','war','best_epoch','validation_uar'])
    write_csv(args.results_dir / 'summary.csv', [{'method': method, **{k:value[k] for k in ['uar_mean','uar_std','war_mean','war_std']}, 'seeds': len(args.seeds), 'folds': 24} for method,value in method_summaries.items()],
              ['method','uar_mean','uar_std','war_mean','war_std','seeds','folds'])
    shared.atomic_json(args.results_dir / 'summary.json', {'methods': method_summaries,
                       'validation_matches_mfcc_confirmed': manifest.get('matches_anastasia_validation_confirmed', False),
                       'validation_manifest': str(manifest_path), 'created_at': datetime.now().astimezone().isoformat()})
    report = ['# Preliminary frozen-encoder results', '',
              'RAVDESS, eight emotions, supplied 24 LOSO test folds, three validation speakers per fold.', '',
              'Validation selection is independently fixed and has not been confirmed to match Anastasia\'s experiment.' if not manifest.get('matches_anastasia_validation_confirmed', False) else 'Validation speakers are confirmed to match the MFCC experiment.', '',
              '| Model | UAR (mean +/- sample SD) | WAR (mean +/- sample SD) |', '| --- | --- | --- |']
    for method, result in method_summaries.items():
        report.append(f'| {method} | {result["uar_mean"]:.4f} +/- {result["uar_std"]:.4f} | {result["war_mean"]:.4f} +/- {result["war_std"]:.4f} |')
    report.extend(['', 'Each seed contributes 1,440 out-of-fold predictions. Means and sample standard deviations summarize independent training seeds, not individual speakers.', '',
                   'Architecture: frozen pretrained encoder, mean pooling of Transformer layer 12, training-fold feature standardization, Linear(768,8) classifier. Adam, learning rate 0.001, batch size 32, 20 epochs, unweighted cross-entropy; no augmentation or encoder fine-tuning. Select the checkpoint with highest validation UAR, breaking ties by lower validation loss. Test scores do not influence model selection.', '',
                   'The standardization mean and standard deviation are fitted to the 1,200 training clips in each fold. The 180 validation and 60 test clips are transformed using those training statistics.', ''])
    (args.results_dir / 'midterm_results.md').write_text('\n'.join(report), encoding='utf-8')
    LOG.info('SUCCESS: all training complete. Results and scorer-compatible predictions: %s', args.results_dir)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOG.warning('Interrupted. Completed fold/seed runs are saved; rerun the same command to resume.')
        sys.exit(130)
    except Exception:
        LOG.exception('TRAINING FAILED. Review the error and rerun to resume completed runs.')
        sys.exit(1)
