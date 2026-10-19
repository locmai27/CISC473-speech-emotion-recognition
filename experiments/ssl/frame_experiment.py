"""Four frozen-encoder SER configurations: Dense mean versus attention pooling.

Adapts Pepino's 128/128 frame Dense head to eight-class, 24-fold LOSO.
Final Transformer layer only; all frames retained. This is not a reproduction
of the paper's learned layer mixture or original evaluation protocol.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
import extract_features as shared
import evaluation
from models import FrameDenseHead

HERE = Path(__file__).resolve().parent
LOG = logging.getLogger('ser.frames')


def local_snapshot(name, cache_dir):
    spec = shared.MODELS[name]
    path = cache_dir / ('models--' + spec['repo'].replace('/', '--')) / 'snapshots' / spec['revision']
    if not (path / 'pytorch_model.bin').is_file():
        raise FileNotFoundError(f'Download the encoder first: {path}')
    return path


def extract_frames(name, rows, root, device, cache_dir):
    from transformers import HubertModel, Wav2Vec2Model, Wav2Vec2FeatureExtractor
    destination = root / name
    destination.mkdir(parents=True, exist_ok=True)
    settings = {
        'model': shared.MODELS[name], 'layer': 12, 'dtype': 'float32',
        'sample_rate': 16000, 'channels': 'arithmetic_mean',
        'resampling': 'scipy.signal.resample_poly', 'waveform_normalization': 'checkpoint_per_clip',
        'pooling': None, 'padding': False, 'truncation': False, 'augmentation': False,
        'frozen': True, 'eval': True, 'device': str(device),
        'script_sha256': shared.sha256(__file__),
        'packages': {k: importlib.metadata.version(k) for k in ['torch','transformers','numpy','scipy','soundfile']},
    }
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    settings['fingerprint'] = fingerprint
    path = destination / 'settings.json'
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError(f'Feature settings changed: {path}; use a fresh feature directory.')
    shared.atomic_json(path, settings)
    pending = []
    for row in rows:
        target = destination / (row['id'] + '.npz')
        if not target.exists():
            pending.append(row)
            continue
        with np.load(target, allow_pickle=False) as cached:
            if str(cached['fingerprint']) != fingerprint or str(cached['audio_sha256']) != shared.sha256(row['path']):
                raise ValueError(f'Feature provenance mismatch: {target}')
            if cached['features'].ndim != 2 or cached['features'].shape[1] != 768 or not np.isfinite(cached['features']).all():
                raise ValueError(f'Invalid cached frames: {target}')
    if pending:
        snapshot = local_snapshot(name, cache_dir)
        cls = HubertModel if name == 'hubert' else Wav2Vec2Model
        LOG.info('Loading frozen %s on %s; %d uncached clips.', name, device, len(pending))
        processor = Wav2Vec2FeatureExtractor.from_pretrained(snapshot, local_files_only=True)
        encoder, info = cls.from_pretrained(snapshot, local_files_only=True, output_loading_info=True, attn_implementation='eager')
        if any(info.get(key) for key in ['missing_keys','mismatched_keys','error_msgs']):
            raise RuntimeError(f'Incomplete encoder weights: {info}')
        encoder.requires_grad_(False).eval().to(device)
        encoder.config.apply_spec_augment = False
        started = time.monotonic()
        for i, row in enumerate(pending, 1):
            audio, source_sr, channels = shared.load_audio(row['path'])
            values = processor(audio, sampling_rate=16000, return_tensors='pt')['input_values'].to(device)
            with torch.inference_mode():
                frames = encoder(input_values=values).last_hidden_state[0].float().cpu().numpy()
            if not np.isfinite(frames).all():
                raise ValueError('Non-finite encoder output.')
            shared.atomic_npz(destination / (row['id'] + '.npz'), features=frames,
                              fingerprint=np.asarray(fingerprint), audio_sha256=np.asarray(shared.sha256(row['path'])),
                              clip_id=np.asarray(row['id']), label=np.asarray(row['label']), person=np.asarray(row['person']))
            if i == 1 or i % 30 == 0 or i == len(pending):
                LOG.info('FRAMES %s %d/%d new clips; elapsed %.1f min', name, i, len(pending), (time.monotonic()-started)/60)
        del encoder, processor, values, frames
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    LOG.info('Frame cache complete: %s, %d recordings.', name, len(rows))
    return fingerprint


def load_frames(name, rows, root):
    settings = json.loads((root / name / 'settings.json').read_text())
    if settings['model'] != shared.MODELS[name] or settings['layer'] != 12 or settings['sample_rate'] != 16000:
        raise ValueError('Frame cache does not use the pinned final-layer encoder configuration.')
    result = {}
    for row in rows:
        with np.load(root / name / (row['id'] + '.npz'), allow_pickle=False) as data:
            if (str(data['fingerprint']) != settings['fingerprint'] or str(data['clip_id']) != row['id']
                    or str(data['label']) != row['label'] or str(data['person']) != row['person']
                    or str(data['audio_sha256']) != shared.sha256(row['path'])):
                raise ValueError('Cached feature provenance mismatch: ' + row['id'])
            values = data['features']
            if values.ndim != 2 or values.shape[1] != 768 or not np.isfinite(values).all():
                raise ValueError('Invalid cached frame sequence: ' + row['id'])
            result[row['id']] = values
    return result


def standardizer(frames, ids):
    total = np.zeros(768, np.float64)
    squares = np.zeros(768, np.float64)
    count = 0
    for clip_id in ids:
        x = frames[clip_id].astype(np.float64)
        total += x.sum(0)
        squares += np.square(x).sum(0)
        count += len(x)
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(squares/count - mean**2, 0)), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def batch(ids, normalized, by_id, device):
    length = max(len(normalized[i]) for i in ids)
    values = np.zeros((len(ids), length, 768), dtype=np.float32)
    mask = np.zeros((len(ids), length), dtype=bool)
    for j, clip_id in enumerate(ids):
        n = len(normalized[clip_id])
        values[j, :n] = normalized[clip_id]
        mask[j, :n] = True
    labels = [shared.LABELS.index(by_id[i]['label']) for i in ids]
    return torch.from_numpy(values).to(device), torch.from_numpy(mask).to(device), torch.tensor(labels, device=device)


def predict(model, ids, normalized, by_id, device, batch_size):
    model.eval()
    truth, predictions = [], []
    loss_sum = 0.
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            x, mask, y = batch(ids[start:start+batch_size], normalized, by_id, device)
            logits = model(x, mask)
            loss_sum += nn.functional.cross_entropy(logits, y, reduction='sum').item()
            truth.extend(y.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return loss_sum/len(ids), evaluation.metrics(truth, predictions), predictions


def fit(name, pooling, seed, fold, frames, by_id, args, device, fingerprint):
    method = f'{name}_dense_{pooling}'
    directory = args.results_dir / 'runs' / method / f'seed_{seed}' / fold['name']
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['fingerprint'] != fingerprint or result['predictions_sha256'] != shared.sha256(directory / 'predictions.csv'):
            raise ValueError(f'Run settings or predictions changed: {directory}')
        import csv
        with (directory / 'predictions.csv').open(newline='') as f:
            predictions = list(csv.DictReader(f))
        LOG.info('RESUME %s seed=%d %s', method, seed, fold['name'])
        return result, predictions
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    # Initialize common weights identically; zero attention scores initially give mean pooling.
    baseline = FrameDenseHead('mean')
    common = copy.deepcopy(baseline.state_dict())
    model = baseline if pooling == 'mean' else FrameDenseHead('attention')
    if pooling != 'mean':
        model.load_state_dict(common, strict=False)
        nn.init.zeros_(model.attention.weight)
    model.to(device)
    torch.manual_seed(seed + 10000)
    generator = torch.Generator().manual_seed(seed)
    mean, std = standardizer(frames, fold['train_ids'])
    # Transform validation/test using training statistics only; test is evaluated after selection.
    normalized = {i: (frames[i]-mean)/std for i in by_id}
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    best_loss, best_epoch, best_state, stale = float('inf'), 0, None, 0
    history = []
    started = time.monotonic()
    LOG.info('TRAIN %s seed=%d %s device=%s train/val/test=1200/180/60', method, seed, fold['name'], device)
    for epoch in range(1, args.epochs+1):
        model.train()
        order = torch.randperm(len(fold['train_ids']), generator=generator).tolist()
        train_loss = 0.
        for start in range(0, len(order), args.batch_size):
            ids = [fold['train_ids'][i] for i in order[start:start+args.batch_size]]
            x, mask, y = batch(ids, normalized, by_id, device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x, mask), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*len(ids)
        val_loss, scores, _ = predict(model, fold['validation_ids'], normalized, by_id, device, args.batch_size)
        history.append({'epoch': epoch, 'training_loss': train_loss/1200, 'validation_loss': val_loss,
                        'validation_uar': scores['uar'], 'validation_war': scores['war']})
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        LOG.info('EPOCH %s seed=%d %s %d/%d train loss=%.4f val loss=%.4f val UAR=%.4f best=%d',
                 method, seed, fold['name'], epoch, args.epochs, train_loss/1200, val_loss, scores['uar'], best_epoch)
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    _, scores, prediction = predict(model, fold['test_ids'], normalized, by_id, device, args.batch_size)
    predictions = [{'id': i, 'true': by_id[i]['label'], 'prediction': shared.LABELS[p], 'split': fold['name'], 'method': method, 'seed': seed}
                   for i,p in zip(fold['test_ids'], prediction)]
    evaluation.write_csv(directory / 'predictions.csv', predictions, evaluation.PREDICTION_HEADER)
    torch.save({'model_state_dict': best_state, 'mean': torch.from_numpy(mean), 'std': torch.from_numpy(std),
                'pooling': pooling, 'labels': shared.LABELS, 'seed': seed, 'split': fold['name'],
                'best_epoch': best_epoch, 'fingerprint': fingerprint}, directory / 'classifier.pt')
    result = {'fingerprint': fingerprint, 'method': method, 'seed': seed, 'split': fold['name'],
              'test_person': fold['test_person'], 'validation_persons': fold['validation_persons'],
              'best_epoch': best_epoch, 'history': history, 'test': scores, 'device': str(device),
              'torch_version': torch.__version__, 'elapsed_seconds': time.monotonic()-started,
              'predictions_sha256': shared.sha256(directory / 'predictions.csv')}
    shared.atomic_json(result_path, result)
    LOG.info('DONE %s seed=%d %s UAR=%.4f WAR=%.4f %.1fs', method, seed, fold['name'], scores['uar'], scores['war'], result['elapsed_seconds'])
    return result, predictions


def reports(results, predictions, root):
    # Existing scoring/export implementation is shared with the verified linear baselines.
    evaluation.write_csv(root / 'predictions.csv', predictions, evaluation.PREDICTION_HEADER)
    summary = []
    for method in sorted({p['method'] for p in predictions}):
        seed_scores = []
        for seed in [42,43,44]:
            rows = [p for p in predictions if p['method']==method and int(p['seed'])==seed]
            if len(rows) != 1440 or len({p['id'] for p in rows}) != 1440:
                continue
            scores = evaluation.metrics([shared.LABELS.index(p['true']) for p in rows], [shared.LABELS.index(p['prediction']) for p in rows])
            seed_scores.append({'seed': seed, **scores})
        if len(seed_scores) == 3:
            summary.append({'method': method,
                            'uar_mean': float(np.mean([s['uar'] for s in seed_scores])),
                            'uar_std': float(np.std([s['uar'] for s in seed_scores],ddof=1)),
                            'war_mean': float(np.mean([s['war'] for s in seed_scores])),
                            'war_std': float(np.std([s['war'] for s in seed_scores],ddof=1)),
                            'seeds': seed_scores})
    shared.atomic_json(root / 'summary.json', {'complete': len(results)==288, 'completed_fits': len(results), 'planned_fits':288,'methods':summary})
    fields = ['method','uar_mean','uar_std','war_mean','war_std']
    evaluation.write_csv(root / 'summary.csv', [{k:r[k] for k in fields} for r in summary], fields)
    lines = ['# Midterm frozen-encoder classifier comparison', '', f'Completed fits: {len(results)}/288.', '',
             '| Encoder + classifier | UAR (%) mean ± SD | Accuracy (%) mean ± SD |',
             '|---|---:|---:|']
    for row in summary:
        lines.append(f"| {row['method']} | {100*row['uar_mean']:.2f} ± {100*row['uar_std']:.2f} | {100*row['war_mean']:.2f} ± {100*row['war_std']:.2f} |")
    manifest = json.loads((root / 'validation_manifest.json').read_text(encoding='utf-8'))
    partition_note = ('Validation speakers follow Anastasia’s confirmed next-three-speaker cyclic rule; all original shared test sets are preserved.'
                      if manifest.get('policy_id') == 'cyclic_next_3' else
                      'Validation speakers were selected independently; agreement with the MFCC validation choices is not established.')
    lines += ['', 'Three seeds (42, 43, 44), 24 speaker-disjoint outer LOSO folds per seed, eight emotions.',
              'Each fold: 20 training speakers, 3 validation speakers, 1 test speaker. ' + partition_note,
              '', 'Both heads: frozen final encoder layer; training-frame normalization; two frame-wise Dense(128), ReLU and dropout(0.2) stages; eight output logits and cross-entropy.',
              'Mean variant uses masked time averaging; attention variant replaces only pooling with a learned 128→1 score. Shared weights start identically and attention starts uniform.',
              'Adam learning rate 0.001, batch size 32, maximum 20 epochs; validation-loss early stopping, patience 4. Keep the checkpoint with minimum validation loss. All recording frames are retained.',
              '', 'This adapts Pepino’s Dense classifier to our eight-class LOSO protocol and final-layer features. It does not reproduce learned layer mixing or the paper’s evaluation splits.',
              'Attention pooling is established prior art. This experiment measures its effect under the matched protocol; it does not establish architecture novelty.',
              '', 'Standard deviations reflect three training seeds on the same dataset, not independent dataset replications. Test results were computed only after validation checkpoint selection.', '']
    (root / 'midterm_results.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--phase', choices=['all','extract','train'], default='all')
    parser.add_argument('--features-dir', type=Path, default=HERE/'artifacts'/'frame_features')
    parser.add_argument('--results-dir', type=Path, default=HERE/'artifacts'/'runs'/'matched_loso')
    parser.add_argument('--validation-manifest', type=Path, default=HERE/'data'/'validation_manifest.json')
    parser.add_argument('--log-dir', type=Path, default=HERE/'artifacts'/'logs')
    parser.add_argument('--clip-table', type=Path, default=HERE/'data'/'clip_table.csv')
    parser.add_argument('--splits', type=Path, default=HERE/'data'/'splits.json')
    parser.add_argument('--audio-dir', type=Path, default=HERE/'artifacts'/'audio')
    parser.add_argument('--cache-dir', type=Path, default=HERE/'artifacts'/'hf-cache')
    parser.add_argument('--limit', type=int, default=0, help='Extraction smoke limit only; zero means all recordings.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-fold-pairs', type=int, default=0, help='Smoke/pilot limit; zero runs all configurations.')
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    shared.setup_logging(args.log_dir)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable.')
    rows = shared.read_clip_table(args.clip_table, args.audio_dir)
    by_id = {r['id']:r for r in rows}
    splits = json.loads(args.splits.read_text())
    # Supplied JSON is a document with splits, rather than necessarily a bare list.
    if isinstance(splits,dict):
        splits = splits['splits']
    manifest_source = args.validation_manifest
    if not manifest_source.is_file():
        raise FileNotFoundError(f'Prepare the explicit validation manifest first: {manifest_source}')
    manifest = evaluation.validation_manifest(splits, by_id, manifest_source)
    destination_manifest=args.results_dir/'validation_manifest.json'
    if destination_manifest.exists() and json.loads(destination_manifest.read_text()) != manifest:
        raise ValueError('Existing result directory uses different speaker partitions.')
    shared.atomic_json(destination_manifest, manifest)
    args.cache_dir.mkdir(parents=True,exist_ok=True)
    if args.phase != 'train':
        for name in ['hubert','wav2vec2']:
            shared.download_model(name, args.cache_dir)
            extract_frames(name, rows[:args.limit] if args.limit else rows, args.features_dir, device, args.cache_dir)
    if args.phase == 'extract':
        return
    if args.limit:
        raise ValueError('--limit is only valid with --phase extract.')
    config = {'epochs':args.epochs,'patience':args.patience,'batch_size':args.batch_size,
              'learning_rate':.001,'seeds':[42,43,44], 'script_sha256':shared.sha256(__file__),
              'head_script_sha256':shared.sha256(HERE/'models.py'),
              'validation_manifest_sha256':shared.sha256(manifest_source),
              'clip_table_sha256':shared.sha256(args.clip_table),
              'shared_splits_sha256':shared.sha256(args.splits),
              'features':{n:json.loads((args.features_dir/n/'settings.json').read_text())['fingerprint'] for n in ['hubert','wav2vec2']}}
    fingerprint = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    config_path = args.results_dir/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Training config changed; choose a fresh results directory.')
    shared.atomic_json(config_path, config)
    results, predictions, pairs = [], [], 0
    for name in ['hubert','wav2vec2']:
        frames = load_frames(name, rows, args.features_dir)
        for seed in [42,43,44]:
            for fold in manifest['folds']:
                for pooling in ['mean','attention']:
                    result, rows_out = fit(name,pooling,seed,fold,frames,by_id,args,device,fingerprint)
                    results.append(result)
                    predictions.extend(rows_out)
                    reports(results,predictions,args.results_dir)
                pairs += 1
                if args.max_fold_pairs and pairs >= args.max_fold_pairs:
                    LOG.info('PILOT LIMIT reached: partial results only.')
                    return
        del frames
        gc.collect()
    summary = reports(results,predictions,args.results_dir)
    plot_confusions(summary,args.results_dir)
    LOG.info('COMPLETE: all 288 fits; report %s',args.results_dir/'midterm_results.md')


def plot_confusions(summary, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for row in summary:
        counts = sum(np.asarray(seed['confusion']) for seed in row['seeds'])
        normalized = counts/counts.sum(1,keepdims=True)
        fig, ax = plt.subplots(figsize=(8,7))
        image = ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
        for i in range(8):
            for j in range(8):
                ax.text(j,i,f'{normalized[i,j]:.2f}',ha='center',va='center',color='white' if normalized[i,j]>.5 else 'black',fontsize=8)
        ax.set(xticks=range(8),yticks=range(8),xticklabels=shared.LABELS,yticklabels=shared.LABELS,
               xlabel='Predicted emotion',ylabel='True emotion',title=row['method']+'\nPooled LOSO predictions across three seeds')
        plt.setp(ax.get_xticklabels(),rotation=40,ha='right')
        fig.colorbar(image,ax=ax,label='Recall fraction')
        fig.tight_layout()
        fig.savefig(root/(row['method']+'_confusion.png'),dpi=160)
        plt.close(fig)
        np.savetxt(root/(row['method']+'_confusion_counts.csv'),counts,delimiter=',',fmt='%d',header=','.join(shared.LABELS),comments='')


if __name__ == '__main__':
    main()
