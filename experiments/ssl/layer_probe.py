"""HuBERT selected-layer probing and layer x augmentation experiment.

Frozen HuBERT, Dense(128)x2 + masked mean pooling, 24-fold LOSO, seeds 42/43/44.
The training loop mirrors frame_experiment.fit() exactly (same seeding order,
training-only normalization, Adam 1e-3, batch 32, <=20 epochs, validation-loss
early stopping with patience 4, minimum-validation-loss checkpoint).

Conditions
  clean : training on clean audio, layers 3, 6, 9, 12 (configurable)
  aug   : training on clean + 3 augmented copies per clip (pitch +/-2 st,
          stretch 0.9/1.1, Gaussian noise 20-30 dB SNR, matching the MFCC
          augmentation policy). Exposure is matched to clean: each epoch every
          training clip is used once, as a randomly chosen version (original or
          one of its 3 copies), so both conditions take the same optimizer steps.
          Validation and test audio stay clean.

Subcommands: extract, verify-extraction, import-released, check-released,
             train, report, package.  Run `python layer_probe.py <cmd> --help`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn

import extract_features as shared
import evaluation
from models import FrameDenseHead

HERE = Path(__file__).resolve().parent
LABELS = shared.LABELS
SEEDS = [42, 43, 44]
AUG_METHODS = ['pitch', 'stretch', 'noise']   # versions 1, 2, 3; version 0 is the original
LOG = logging.getLogger('ser.layers')


# --------------------------------------------------------------------------- utils
def log_setup(log_dir, name):
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s')
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for h in [logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / f'{name}.log', encoding='utf-8')]:
        h.setFormatter(fmt)
        root.addHandler(h)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_rows(clip_table, audio_dir=None):
    """Clip table rows; audio paths are checked only when audio_dir is given."""
    if audio_dir is not None:
        return shared.read_clip_table(clip_table, audio_dir)
    with Path(clip_table).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['id', 'person', 'label']:
            raise ValueError('Clip table header must be exactly id,person,label.')
        return list(reader)


def load_manifest(args, by_id):
    splits = json.loads(args.splits.read_text())
    if isinstance(splits, dict):
        splits = splits['splits']
    manifest = evaluation.validation_manifest(splits, by_id, args.validation_manifest)
    if manifest.get('policy_id') != 'cyclic_next_3':
        raise ValueError('Expected the published cyclic_next_3 validation manifest.')
    return manifest


# ------------------------------------------------------------------ feature store
class FeatureSet:
    """One (condition, layer) cache: float32 frames in one raw file + an index.

    Directory layout: <root>/<condition>/layer_XX/{frames.f32, index.json, settings.json, COMPLETE}
    index['items'] maps "clip_id|version" -> [offset_frames, n_frames].
    Opened with np.memmap so parallel workers share the OS page cache.
    """

    def __init__(self, directory):
        self.dir = Path(directory)
        if not (self.dir / 'COMPLETE').exists():
            raise FileNotFoundError(f'Feature cache is incomplete or missing: {self.dir}')
        self.settings = json.loads((self.dir / 'settings.json').read_text())
        index = json.loads((self.dir / 'index.json').read_text())
        self.items = {k: tuple(v) for k, v in index['items'].items()}
        total = index['total_frames']
        self.data = np.memmap(self.dir / 'frames.f32', dtype=np.float32, mode='r', shape=(total, 768))
        self.fingerprint = self.settings['fingerprint']

    def get(self, clip_id, version=0):
        start, n = self.items[f'{clip_id}|{version}']
        return self.data[start:start + n]

    def versions(self, clip_id):
        return sorted(int(k.split('|')[1]) for k in self.items if k.startswith(clip_id + '|'))


class FeatureWriter:
    def __init__(self, directory, settings):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        for name in ['COMPLETE', 'frames.f32', 'index.json']:
            (self.dir / name).unlink(missing_ok=True)
        settings = dict(settings)
        settings['fingerprint'] = json_hash(settings)
        self.settings = settings
        shared.atomic_json(self.dir / 'settings.json', settings)
        self.handle = (self.dir / 'frames.f32').open('wb')
        self.items, self.offset, self.extra = {}, 0, {}

    def add(self, clip_id, version, frames, extra=None):
        frames = np.ascontiguousarray(frames, dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != 768 or not np.isfinite(frames).all():
            raise ValueError(f'Invalid frames for {clip_id}|{version}: {frames.shape}')
        self.handle.write(frames.tobytes())
        self.items[f'{clip_id}|{version}'] = [self.offset, len(frames)]
        if extra:
            self.extra[f'{clip_id}|{version}'] = extra
        self.offset += len(frames)

    def close(self):
        self.handle.close()
        shared.atomic_json(self.dir / 'index.json', {'total_frames': self.offset, 'items': self.items,
                                                     'augmentation_parameters': self.extra})
        (self.dir / 'COMPLETE').write_text(self.settings['fingerprint'] + '\n')


# ------------------------------------------------------------------ augmentation
def augment(audio, clip_id, method, aug_seed, sr=16000):
    """Deterministic per-clip augmentation; parameters follow the MFCC policy."""
    import librosa
    digest = hashlib.sha256(f'{aug_seed}|{clip_id}|{method}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    if method == 'pitch':
        steps = float(rng.choice([-2.0, 2.0]))
        out = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
        params = {'n_steps': steps}
    elif method == 'stretch':
        rate = float(rng.choice([0.9, 1.1]))
        out = librosa.effects.time_stretch(audio, rate=rate)
        params = {'rate': rate}
    elif method == 'noise':
        snr_db = float(rng.uniform(20, 30))
        noise_power = np.mean(audio.astype(np.float64) ** 2) / (10 ** (snr_db / 10))
        out = audio + rng.normal(0, np.sqrt(noise_power), size=audio.shape)
        params = {'snr_db': snr_db}
    else:
        raise ValueError(method)
    out = np.asarray(out, dtype=np.float32)
    if not np.isfinite(out).all() or out.size < 400:
        raise ValueError(f'Bad augmented audio: {clip_id} {method}')
    return out, params


def _augmented_job(task):
    path, clip_id, aug_seed = task
    audio, _, _ = shared.load_audio(path)
    return clip_id, [augment(audio, clip_id, m, aug_seed) for m in AUG_METHODS]


# -------------------------------------------------------------------- extraction
def cmd_extract(args):
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    from concurrent.futures import ProcessPoolExecutor
    log_setup(args.log_dir, f'extract-{args.condition}')
    device = torch.device(args.device)
    # Keep GPU extraction in full float32 (no TF32) so it stays close to the released CPU cache.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = read_rows(args.clip_table, args.audio_dir)
    if args.limit:
        rows = rows[:args.limit]
    layers = sorted(set(args.layers))
    snapshot = shared.download_model('hubert', args.cache_dir)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(snapshot, local_files_only=True)
    encoder, info = HubertModel.from_pretrained(snapshot, local_files_only=True, output_loading_info=True,
                                                attn_implementation='eager')
    if any(info.get(k) for k in ['missing_keys', 'mismatched_keys', 'error_msgs']):
        raise RuntimeError(f'Incomplete encoder weights: {info}')
    encoder.config.apply_spec_augment = False
    encoder.requires_grad_(False).eval().to(device)
    base = {
        'model': shared.MODELS['hubert'], 'dtype': 'float32', 'sample_rate': 16000,
        'channels': 'arithmetic_mean', 'resampling': 'scipy.signal.resample_poly',
        'waveform_normalization': 'checkpoint_per_clip', 'pooling': None, 'padding': False,
        'truncation': False, 'frozen': True, 'eval': True, 'device': str(device),
        'hidden_state_index': 'transformers hidden_states[layer] (0 = CNN/projection output, 12 = last_hidden_state)',
        'condition': args.condition, 'limit': args.limit, 'clips': len(rows),
        'script_sha256': shared.sha256(__file__),
        'packages': {k: importlib.metadata.version(k) for k in ['torch', 'transformers', 'numpy', 'scipy', 'soundfile', 'librosa']},
    }
    if args.condition == 'aug':
        base['augmentation'] = {'versions': {str(i + 1): m for i, m in enumerate(AUG_METHODS)},
                                'pitch_semitones': [-2.0, 2.0], 'stretch_rates': [0.9, 1.1],
                                'noise_snr_db_uniform': [20, 30], 'aug_seed': args.aug_seed,
                                'applied_to': 'raw 16 kHz mono waveform before HuBERT preprocessing',
                                'stored_versions': 'augmented copies only (1..3); originals come from the clean cache'}
    else:
        base['augmentation'] = False
    writers = {L: FeatureWriter(args.features_dir / args.condition / f'layer_{L:02d}', {**base, 'layer': L})
               for L in layers}

    def encode(audio):
        values = processor(audio, sampling_rate=16000, return_tensors='pt')['input_values'].to(device)
        with torch.inference_mode():
            out = encoder(input_values=values, output_hidden_states=True)
        hs = out.hidden_states
        if len(hs) != 13:
            raise RuntimeError(f'Expected 13 hidden states, got {len(hs)}')
        if not torch.equal(hs[12], out.last_hidden_state):
            raise RuntimeError('hidden_states[12] differs from last_hidden_state.')
        return {L: hs[L][0].float().cpu().numpy() for L in layers}

    started = time.monotonic()
    if args.condition == 'clean':
        for i, row in enumerate(rows, 1):
            audio, _, _ = shared.load_audio(row['path'])
            for L, frames in encode(audio).items():
                writers[L].add(row['id'], 0, frames)
            if i == 1 or i % 100 == 0 or i == len(rows):
                LOG.info('clean %d/%d clips, %.1f min', i, len(rows), (time.monotonic() - started) / 60)
    else:
        tasks = [(str(r['path']), r['id'], args.aug_seed) for r in rows]
        with ProcessPoolExecutor(max_workers=args.cpu_workers) as pool:
            for i, (clip_id, outputs) in enumerate(pool.map(_augmented_job, tasks, chunksize=4), 1):
                for version, (audio, params) in enumerate(outputs, 1):
                    for L, frames in encode(audio).items():
                        writers[L].add(clip_id, version, frames, {'method': AUG_METHODS[version - 1], **params})
                if i == 1 or i % 100 == 0 or i == len(rows):
                    LOG.info('aug %d/%d clips (x3 copies), %.1f min', i, len(rows), (time.monotonic() - started) / 60)
    for w in writers.values():
        w.close()
    LOG.info('Extraction complete: %s layers %s -> %s', args.condition, layers, args.features_dir / args.condition)


def cmd_import_released(args):
    """Convert the released layer-12 .npz cache into a FeatureSet (condition 'released')."""
    log_setup(args.log_dir, 'import-released')
    rows = read_rows(args.clip_table)
    src = args.released_dir
    settings = json.loads((src / 'settings.json').read_text())
    w = FeatureWriter(args.features_dir / 'released' / 'layer_12',
                      {'source': str(src), 'released_fingerprint': settings['fingerprint'], 'layer': 12,
                       'condition': 'released'})
    for row in rows:
        with np.load(src / f"{row['id']}.npz", allow_pickle=False) as d:
            if str(d['fingerprint']) != settings['fingerprint'] or str(d['clip_id']) != row['id']:
                raise ValueError('Released cache provenance mismatch: ' + row['id'])
            w.add(row['id'], 0, d['features'])
    w.close()
    LOG.info('Imported %d released layer-12 sequences.', len(rows))


def cmd_verify_extraction(args):
    """Compare new clean layer-12 frames with the released layer-12 cache."""
    log_setup(args.log_dir, 'verify-extraction')
    new = FeatureSet(args.features_dir / 'clean' / 'layer_12')
    old = FeatureSet(args.features_dir / 'released' / 'layer_12')
    rows = read_rows(args.clip_table)
    max_abs, rel, shape_mismatch, compared = 0.0, [], 0, 0
    for row in rows:
        if f"{row['id']}|0" not in new.items:
            continue
        a, b = np.asarray(new.get(row['id'])), np.asarray(old.get(row['id']))
        compared += 1
        if a.shape != b.shape:
            shape_mismatch += 1
            continue
        max_abs = max(max_abs, float(np.abs(a - b).max()))
        rel.append(float(np.linalg.norm(a - b) / np.linalg.norm(b)))
    report = {'compared_clips': compared, 'frame_count_mismatches': shape_mismatch,
              'max_abs_difference': max_abs, 'max_relative_l2_difference': max(rel) if rel else None,
              'mean_relative_l2_difference': float(np.mean(rel)) if rel else None,
              'tolerance_relative_l2': args.tolerance}
    report['passed'] = bool(compared > 0 and shape_mismatch == 0 and rel and max(rel) <= args.tolerance)
    shared.atomic_json(args.results_dir / 'extraction_verification.json', report)
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit('Layer-12 verification FAILED: new extraction does not match the released cache.')


# ---------------------------------------------------------------------- training
def standardizer(arrays):
    """Identical arithmetic to frame_experiment.standardizer (float64 accumulation)."""
    total = np.zeros(768, np.float64)
    squares = np.zeros(768, np.float64)
    count = 0
    for x in arrays:
        x = np.asarray(x, dtype=np.float64)
        total += x.sum(0)
        squares += np.square(x).sum(0)
        count += len(x)
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(squares / count - mean ** 2, 0)), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def make_batch(arrays, labels, mean, std, device):
    """Same padding/mask construction as frame_experiment.batch; normalization per clip."""
    length = max(len(a) for a in arrays)
    values = np.zeros((len(arrays), length, 768), dtype=np.float32)
    mask = np.zeros((len(arrays), length), dtype=bool)
    for j, a in enumerate(arrays):
        values[j, :len(a)] = (a - mean) / std
        mask[j, :len(a)] = True
    return (torch.from_numpy(values).to(device), torch.from_numpy(mask).to(device),
            torch.tensor(labels, device=device))


def evaluate(model, ids, clean, by_id, mean, std, device, batch_size):
    model.eval()
    truth, preds, loss_sum = [], [], 0.
    with torch.inference_mode():
        for s in range(0, len(ids), batch_size):
            chunk = ids[s:s + batch_size]
            x, m, y = make_batch([clean.get(i) for i in chunk], [LABELS.index(by_id[i]['label']) for i in chunk],
                                 mean, std, device)
            logits = model(x, m)
            loss_sum += nn.functional.cross_entropy(logits, y, reduction='sum').item()
            truth.extend(y.cpu().tolist())
            preds.extend(logits.argmax(1).cpu().tolist())
    return loss_sum / len(ids), evaluation.metrics(truth, preds), preds


def method_name(layer, condition):
    return f'hubert_L{layer:02d}_dense_mean_{condition}'


def fit(job, sets, by_id, folds, args, device, fingerprint):
    layer, condition, seed, fold_name = job['layer'], job['condition'], job['seed'], job['fold']
    fold = folds[fold_name]
    method = method_name(layer, condition)
    directory = args.results_dir / 'runs' / method / f'seed_{seed}' / fold_name
    result_path = directory / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['fingerprint'] == fingerprint and result['predictions_sha256'] == shared.sha256(directory / 'predictions.csv'):
            return 'resumed'
        raise ValueError(f'Existing run has different settings: {directory}. Use a fresh results directory.')
    directory.mkdir(parents=True, exist_ok=True)
    clean = sets[('released' if condition == 'released' else 'clean', layer)]
    augset = sets.get(('aug', layer)) if condition == 'aug' else None

    # ---- seeding order identical to frame_experiment.fit for pooling='mean'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = FrameDenseHead('mean').to(device)
    torch.manual_seed(seed + 10000)
    generator = torch.Generator().manual_seed(seed)

    train_ids = fold['train_ids']
    if augset is None:
        mean, std = standardizer(clean.get(i) for i in train_ids)
    else:
        # Training data = originals + their augmented copies (training speakers only).
        mean, std = standardizer([clean.get(i) for i in train_ids] +
                                 [augset.get(i, v) for i in train_ids for v in (1, 2, 3)])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    best_loss, best_epoch, best_state, stale, history = float('inf'), 0, None, 0, []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_ids), generator=generator).tolist()
        if augset is not None:
            # Matched exposure: one version per clip per epoch, uniformly among {0,1,2,3}.
            vrng = np.random.default_rng([seed, int(fold['test_person']), epoch, 7])
            version_of = vrng.integers(0, 4, size=len(train_ids))
        train_loss = 0.
        for s in range(0, len(order), args.batch_size):
            idx = order[s:s + args.batch_size]
            if augset is None:
                arrays = [clean.get(train_ids[i]) for i in idx]
            else:
                arrays = [clean.get(train_ids[i]) if version_of[i] == 0 else augset.get(train_ids[i], int(version_of[i]))
                          for i in idx]
            x, m, y = make_batch(arrays, [LABELS.index(by_id[train_ids[i]]['label']) for i in idx], mean, std, device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x, m), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(idx)
        val_loss, scores, _ = evaluate(model, fold['validation_ids'], clean, by_id, mean, std, device, args.batch_size)
        history.append({'epoch': epoch, 'training_loss': train_loss / len(train_ids), 'validation_loss': val_loss,
                        'validation_uar': scores['uar'], 'validation_war': scores['war']})
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    # Test speaker scored only after checkpoint selection.
    _, scores, prediction = evaluate(model, fold['test_ids'], clean, by_id, mean, std, device, args.batch_size)
    predictions = [{'id': i, 'true': by_id[i]['label'], 'prediction': LABELS[p], 'split': fold_name,
                    'method': method, 'seed': seed} for i, p in zip(fold['test_ids'], prediction)]
    evaluation.write_csv(directory / 'predictions.csv', predictions, evaluation.PREDICTION_HEADER)
    torch.save({'model_state_dict': best_state, 'mean': torch.from_numpy(mean), 'std': torch.from_numpy(std),
                'pooling': 'mean', 'layer': layer, 'condition': condition, 'labels': LABELS, 'seed': seed,
                'split': fold_name, 'best_epoch': best_epoch, 'fingerprint': fingerprint}, directory / 'classifier.pt')
    result = {'fingerprint': fingerprint, 'method': method, 'layer': layer, 'condition': condition, 'seed': seed,
              'split': fold_name, 'test_person': fold['test_person'], 'validation_persons': fold['validation_persons'],
              'best_epoch': best_epoch, 'best_validation_loss': best_loss, 'history': history, 'test': scores,
              'device': str(device), 'torch_version': torch.__version__,
              'elapsed_seconds': time.monotonic() - started,
              'predictions_sha256': shared.sha256(directory / 'predictions.csv')}
    shared.atomic_json(result_path, result)
    return f"UAR={scores['uar']:.4f} best_epoch={best_epoch} {result['elapsed_seconds']:.1f}s"


def plan(args, manifest):
    jobs = []
    for L in args.clean_layers:
        jobs += [{'layer': L, 'condition': 'clean'}]
    for L in args.aug_layers:
        jobs += [{'layer': L, 'condition': 'aug'}]
    for L in args.released_layers:
        jobs += [{'layer': L, 'condition': 'released'}]
    full = [{**j, 'seed': s, 'fold': f['name']} for j in jobs for s in SEEDS for f in manifest['folds']]
    if args.max_folds:
        keep = {f['name'] for f in manifest['folds'][:args.max_folds]}
        full = [j for j in full if j['fold'] in keep]
    return full


def training_config(args, sets):
    return {'epochs': args.epochs, 'patience': args.patience, 'batch_size': args.batch_size, 'learning_rate': .001,
            'weight_decay': 0., 'seeds': SEEDS, 'pooling': 'mean', 'head_script_sha256': shared.sha256(HERE / 'models.py'),
            'training_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'validation_manifest_sha256': shared.sha256(args.validation_manifest),
            'clip_table_sha256': shared.sha256(args.clip_table), 'splits_sha256': shared.sha256(args.splits),
            'aug_exposure': 'matched: one uniformly chosen version (original or 1 of 3 copies) per training clip per epoch',
            'aug_normalization': 'mean/std over originals and copies of training-speaker clips',
            'features': {f'{c}/layer_{L:02d}': s.fingerprint for (c, L), s in sorted(sets.items())}}


def cmd_train(args):
    log_setup(args.log_dir, f'train-worker{args.worker_id}')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    rows = read_rows(args.clip_table)
    by_id = {r['id']: r for r in rows}
    manifest = load_manifest(args, by_id)
    folds = {f['name']: f for f in manifest['folds']}
    sets = {}
    for L in args.clean_layers:
        sets[('clean', L)] = FeatureSet(args.features_dir / 'clean' / f'layer_{L:02d}')
    for L in args.aug_layers:
        sets[('clean', L)] = FeatureSet(args.features_dir / 'clean' / f'layer_{L:02d}')
        sets[('aug', L)] = FeatureSet(args.features_dir / 'aug' / f'layer_{L:02d}')
    for L in args.released_layers:
        sets[('released', L)] = FeatureSet(args.features_dir / 'released' / f'layer_{L:02d}')
    config = training_config(args, sets)
    fingerprint = json_hash(config)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    cpath = args.results_dir / 'config.json'
    if cpath.exists() and json.loads(cpath.read_text()) != config:
        # Workers launched together all compute the same config; anything else is a real change.
        raise ValueError('Training config differs from the one already in this results directory.')
    if not cpath.exists():
        tmp = cpath.with_suffix(f'.{os.getpid()}.tmp')   # per-process temp: workers may start together
        tmp.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        tmp.replace(cpath)
    jobs = plan(args, manifest)
    # Interleave so each worker gets a similar mix of clean and augmented fits.
    mine = jobs[args.worker_id::args.num_workers]
    LOG.info('worker %d/%d: %d of %d fits on %s', args.worker_id, args.num_workers, len(mine), len(jobs), device)
    for k, job in enumerate(mine, 1):
        status = fit(job, sets, by_id, folds, args, device, fingerprint)
        LOG.info('[w%d %d/%d] %s seed=%d %s: %s', args.worker_id, k, len(mine),
                 method_name(job['layer'], job['condition']), job['seed'], job['fold'], status)
    LOG.info('worker %d finished.', args.worker_id)


# ------------------------------------------------------------------ released check
def cmd_check_released(args):
    """Replay released hubert_dense_mean checkpoints on released layer-12 features."""
    log_setup(args.log_dir, 'check-released')
    device = torch.device(args.device)
    rows = read_rows(args.clip_table)
    by_id = {r['id']: r for r in rows}
    manifest = load_manifest(args, by_id)
    feats = FeatureSet(args.features_dir / 'released' / 'layer_12')
    with (args.released_runs / 'predictions.csv').open(newline='') as f:
        published = {(r['id'], int(r['seed'])): r['prediction'] for r in csv.DictReader(f) if r['method'] == 'hubert_dense_mean'}
    agree = total = 0
    truth, preds = [], []
    for fold in manifest['folds']:
        ckpt = torch.load(args.released_runs / 'runs' / 'hubert_dense_mean' / f'seed_{args.seed}' / fold['name'] / 'classifier.pt',
                          map_location='cpu', weights_only=False)
        model = FrameDenseHead('mean')
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        mean, std = ckpt['mean'].numpy(), ckpt['std'].numpy()
        _, _, p = evaluate(model, fold['test_ids'], feats, by_id, mean, std, device, 32)
        for i, k in zip(fold['test_ids'], p):
            agree += LABELS[k] == published[(i, args.seed)]
            total += 1
            truth.append(LABELS.index(by_id[i]['label']))
            preds.append(k)
    m = evaluation.metrics(truth, preds)
    out = {'seed': args.seed, 'checkpoints': len(manifest['folds']), 'test_predictions': total,
           'agreement_with_published_predictions': agree / total, 'replayed_uar': m['uar'], 'replayed_war': m['war']}
    print(json.dumps(out, indent=2))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    shared.atomic_json(args.results_dir / 'released_checkpoint_check.json', out)


# ------------------------------------------------------------------------ report
def seed_metrics(rows):
    return evaluation.metrics([LABELS.index(r['true']) for r in rows], [LABELS.index(r['prediction']) for r in rows])


def cmd_report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    log_setup(args.log_dir, 'report')
    root = args.results_dir
    preds, results = [], []
    for rp in sorted((root / 'runs').glob('*/seed_*/*/result.json')):
        results.append(json.loads(rp.read_text()))
        with (rp.parent / 'predictions.csv').open(newline='') as f:
            preds.extend(csv.DictReader(f))
    if not preds:
        raise SystemExit('No finished fits found.')
    keys = [(p['method'], p['seed'], p['id']) for p in preds]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate test predictions.')
    evaluation.write_csv(root / 'predictions.csv', preds, evaluation.PREDICTION_HEADER)
    meta = {r['method']: (r['layer'], r['condition']) for r in results}
    summary, per_seed = [], {}
    for method in sorted(meta):
        seeds = []
        for seed in SEEDS:
            rows = [p for p in preds if p['method'] == method and int(p['seed']) == seed]
            if len(rows) == 1440 and len({r['id'] for r in rows}) == 1440:
                seeds.append({'seed': seed, **seed_metrics(rows)})
        n_fits = sum(r['method'] == method for r in results)
        layer, condition = meta[method]
        row = {'method': method, 'layer': layer, 'condition': condition, 'fits': n_fits, 'complete_seeds': len(seeds)}
        if seeds:
            per_seed[method] = {s['seed']: s for s in seeds}
            u, w = [s['uar'] for s in seeds], [s['war'] for s in seeds]
            row.update({'uar_mean': float(np.mean(u)), 'uar_std': float(np.std(u, ddof=1)) if len(u) > 1 else float('nan'),
                        'war_mean': float(np.mean(w)), 'war_std': float(np.std(w, ddof=1)) if len(w) > 1 else float('nan')})
            rec = np.mean([s['recalls'] for s in seeds], axis=0)
            row.update({f'recall_{lab}': float(r) for lab, r in zip(LABELS, rec)})
            row['mean_best_epoch'] = float(np.mean([r['best_epoch'] for r in results if r['method'] == method]))
        summary.append(row)
    shared.atomic_json(root / 'summary.json', {'methods': summary, 'per_seed': {
        m: {str(k): {x: v[x] for x in ['uar', 'war', 'recalls', 'confusion']} for k, v in d.items()} for m, d in per_seed.items()}})
    fields = ['method', 'layer', 'condition', 'fits', 'complete_seeds', 'uar_mean', 'uar_std', 'war_mean', 'war_std',
              'mean_best_epoch'] + [f'recall_{l}' for l in LABELS]
    evaluation.write_csv(root / 'summary.csv', [{k: r.get(k, '') for k in fields} for r in summary], fields)

    # Augmentation gain and layer x augmentation interaction (paired by seed).
    gains = []
    for L in sorted({l for l, c in meta.values() if c == 'aug'}):
        c, a = per_seed.get(method_name(L, 'clean'), {}), per_seed.get(method_name(L, 'aug'), {})
        for seed in sorted(set(c) & set(a)):
            gains.append({'layer': L, 'seed': seed, 'clean_uar': c[seed]['uar'], 'aug_uar': a[seed]['uar'],
                          'uar_gain': a[seed]['uar'] - c[seed]['uar'], 'war_gain': a[seed]['war'] - c[seed]['war']})
    if gains:
        evaluation.write_csv(root / 'augmentation_gain.csv', gains, list(gains[0]))
    interaction = None
    gl = {(g['layer'], g['seed']): g['uar_gain'] for g in gains}
    aug_layers = sorted({g['layer'] for g in gains})
    if len(aug_layers) == 2:
        lo, hi = aug_layers
        diffs = [gl[(lo, s)] - gl[(hi, s)] for s in SEEDS if (lo, s) in gl and (hi, s) in gl]
        if diffs:
            interaction = {'definition': f'gain(L{lo}) - gain(L{hi}), UAR, paired by seed', 'per_seed': diffs,
                           'mean': float(np.mean(diffs)), 'std': float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None,
                           'same_sign_all_seeds': bool(all(d > 0 for d in diffs) or all(d < 0 for d in diffs))}
            shared.atomic_json(root / 'interaction.json', interaction)

    # Control: layer-12 clean rerun vs published hubert_dense_mean.
    control = None
    if args.published_predictions and args.published_predictions.exists() and method_name(12, 'clean') in per_seed:
        with args.published_predictions.open(newline='') as f:
            pub = [r for r in csv.DictReader(f) if r['method'] == 'hubert_dense_mean']
        pub_map = {(r['id'], int(r['seed'])): r['prediction'] for r in pub}
        new = [p for p in preds if p['method'] == method_name(12, 'clean')]
        agree = np.mean([pub_map[(p['id'], int(p['seed']))] == p['prediction'] for p in new])
        pu = [seed_metrics([r for r in pub if int(r['seed']) == s])['uar'] for s in SEEDS]
        nu = [per_seed[method_name(12, 'clean')][s]['uar'] for s in SEEDS if s in per_seed[method_name(12, 'clean')]]
        control = {'published_uar_mean': float(np.mean(pu)), 'rerun_uar_mean': float(np.mean(nu)),
                   'uar_difference': float(np.mean(nu) - np.mean(pu)), 'prediction_agreement': float(agree)}
        shared.atomic_json(root / 'layer12_control_check.json', control)

    # Figure: UAR by layer.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for condition, style, label in [('clean', 'o-', 'Clean training'), ('aug', 's--', 'Augmented training'),
                                    ('released', 'x', 'Released L12 features')]:
        rows = sorted([r for r in summary if r['condition'] == condition and 'uar_mean' in r], key=lambda r: r['layer'])
        if rows:
            ax.errorbar([r['layer'] for r in rows], [100 * r['uar_mean'] for r in rows],
                        yerr=[100 * (r['uar_std'] if np.isfinite(r['uar_std']) else 0) for r in rows],
                        fmt=style, capsize=4, label=label)
    ax.set(xlabel='HuBERT Transformer layer', ylabel='UAR (%)', xticks=sorted({r['layer'] for r in summary}),
           title='HuBERT selected-layer probing (Dense + mean pooling)\nmean ± sample SD over seeds 42-44, 24-fold LOSO')
    ax.axhline(12.5, color='grey', lw=.8, ls=':')
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'layer_uar.png', dpi=180)
    fig.savefig(root / 'layer_uar.pdf')
    plt.close(fig)

    # Figure: per-emotion recall by layer (clean).
    clean_rows = sorted([r for r in summary if r['condition'] == 'clean' and 'uar_mean' in r], key=lambda r: r['layer'])
    if clean_rows:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        width = .8 / len(clean_rows)
        for k, r in enumerate(clean_rows):
            ax.bar(np.arange(8) + k * width, [100 * r[f'recall_{l}'] for l in LABELS], width, label=f"Layer {r['layer']}")
        ax.set(xticks=np.arange(8) + .4 - width / 2, ylabel='Recall (%)', title='Per-emotion recall by layer (clean, mean over seeds)')
        ax.set_xticklabels(LABELS, rotation=30, ha='right')
        ax.legend(ncol=len(clean_rows))
        fig.tight_layout()
        fig.savefig(root / 'per_emotion_recall_by_layer.png', dpi=180)
        plt.close(fig)

    # Markdown summary.
    lines = ['# HuBERT selected-layer probing and augmentation', '',
             f'Completed fits: {len(results)}.', '',
             '| Layer | Training | UAR (%) mean ± SD | WAR (%) mean ± SD | Seeds |', '|---:|---|---:|---:|---:|']
    for r in sorted(summary, key=lambda r: (r['condition'], r['layer'])):
        if 'uar_mean' in r:
            lines.append(f"| {r['layer']} | {r['condition']} | {100*r['uar_mean']:.2f} ± {100*r['uar_std']:.2f} | "
                         f"{100*r['war_mean']:.2f} ± {100*r['war_std']:.2f} | {r['complete_seeds']} |")
    if gains:
        lines += ['', '## Augmentation gain (augmented − clean UAR, percentage points)', '',
                  '| Layer | Gain mean ± SD | Per seed |', '|---:|---:|---|']
        for L in aug_layers:
            g = [x['uar_gain'] for x in gains if x['layer'] == L]
            sd = np.std(g, ddof=1) if len(g) > 1 else float('nan')
            lines.append(f"| {L} | {100*np.mean(g):+.2f} ± {100*sd:.2f} | {', '.join(f'{100*v:+.2f}' for v in g)} |")
    if interaction:
        lines += ['', f"Interaction, {interaction['definition']}: {100*interaction['mean']:+.2f} pp "
                      f"(per seed {', '.join(f'{100*v:+.2f}' for v in interaction['per_seed'])}; "
                      f"same sign for every seed: {interaction['same_sign_all_seeds']}). No significance test is claimed."]
    if control:
        lines += ['', f"Layer-12 control: rerun UAR {100*control['rerun_uar_mean']:.2f}% vs published "
                      f"{100*control['published_uar_mean']:.2f}%; test-prediction agreement {100*control['prediction_agreement']:.1f}%."]
    lines += ['', '## Protocol', '',
              'Frozen facebook/hubert-base-ls960 (pinned revision); hidden_states[L] from the Transformer, full recordings, no padding.',
              'Same Dense(128)x2 + masked mean pooling head and training procedure for every layer; a fresh classifier per layer, seed and fold.',
              '24 LOSO folds (20 train / 3 cyclic-next validation / 1 test speaker), seeds 42, 43, 44, training-only normalization,',
              'minimum-validation-loss checkpoint (patience 4, max 20 epochs). Test speakers scored only after selection.',
              'Augmentation: pitch ±2 semitones, time stretch 0.9/1.1, Gaussian noise 20–30 dB SNR, one copy each per clip,',
              'applied to the waveform before HuBERT; training speakers only; validation/test audio clean. Exposure matched to clean',
              'training (1,200 samples per epoch, version drawn uniformly from original + 3 copies).',
              'Layers and policy were fixed in advance (not chosen on test data). This is a selected-layer comparison (4 of 12 layers).',
              'SDs are across three training seeds on one dataset, not independent replications.']
    (root / 'layer_probe_results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


# ----------------------------------------------------------------------- package
def cmd_package(args):
    root = args.results_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    ckpt_zip = out / f'{args.prefix}-checkpoints.zip'
    res_zip = out / f'{args.prefix}-results.zip'
    n = 0
    with zipfile.ZipFile(ckpt_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted((root / 'runs').rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(root).as_posix())
                n += p.name == 'classifier.pt'
        z.write(root / 'config.json', 'config.json')
    with zipfile.ZipFile(res_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.iterdir()):
            if p.is_file():
                z.write(p, p.name)
    assets = []
    for p in [ckpt_zip, res_zip]:
        assets.append({'name': p.name, 'bytes': p.stat().st_size, 'sha256': shared.sha256(p)})
    manifest = {'classifiers': n, 'assets': assets, 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    shared.atomic_json(out / f'{args.prefix}-manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


# -------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp):
        sp.add_argument('--features-dir', type=Path, default=HERE / 'artifacts' / 'layer_features')
        sp.add_argument('--results-dir', type=Path, default=HERE / 'artifacts' / 'runs' / 'layer_probe')
        sp.add_argument('--log-dir', type=Path, default=HERE / 'artifacts' / 'logs')
        sp.add_argument('--clip-table', type=Path, default=HERE / 'data' / 'clip_table.csv')
        sp.add_argument('--splits', type=Path, default=HERE / 'data' / 'splits.json')
        sp.add_argument('--validation-manifest', type=Path, default=HERE / 'data' / 'validation_manifest.json')
        sp.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')

    sp = sub.add_parser('extract'); common(sp)
    sp.add_argument('--condition', choices=['clean', 'aug'], required=True)
    sp.add_argument('--layers', type=int, nargs='+', required=True)
    sp.add_argument('--audio-dir', type=Path, default=HERE / 'artifacts' / 'audio')
    sp.add_argument('--cache-dir', type=Path, default=HERE / 'artifacts' / 'hf-cache')
    sp.add_argument('--aug-seed', type=int, default=473)
    sp.add_argument('--cpu-workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    sp.add_argument('--limit', type=int, default=0)
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser('import-released'); common(sp)
    sp.add_argument('--released-dir', type=Path, default=HERE / 'artifacts' / 'frame_features' / 'hubert')
    sp.set_defaults(func=cmd_import_released)

    sp = sub.add_parser('verify-extraction'); common(sp)
    sp.add_argument('--tolerance', type=float, default=1e-3, help='Max relative L2 difference per clip.')
    sp.set_defaults(func=cmd_verify_extraction)

    sp = sub.add_parser('check-released'); common(sp)
    sp.add_argument('--released-runs', type=Path, default=HERE / 'artifacts' / 'runs' / 'matched_loso')
    sp.add_argument('--seed', type=int, default=42)
    sp.set_defaults(func=cmd_check_released)

    sp = sub.add_parser('train'); common(sp)
    sp.add_argument('--clean-layers', type=int, nargs='*', default=[3, 6, 9, 12])
    sp.add_argument('--aug-layers', type=int, nargs='*', default=[6, 12])
    sp.add_argument('--released-layers', type=int, nargs='*', default=[])
    sp.add_argument('--worker-id', type=int, default=0)
    sp.add_argument('--num-workers', type=int, default=1)
    sp.add_argument('--threads', type=int, default=2)
    sp.add_argument('--epochs', type=int, default=20)
    sp.add_argument('--patience', type=int, default=4)
    sp.add_argument('--batch-size', type=int, default=32)
    sp.add_argument('--max-folds', type=int, default=0, help='Pilot only: first N folds.')
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser('report'); common(sp)
    sp.add_argument('--published-predictions', type=Path, default=HERE / 'results' / 'matched_loso' / 'predictions.csv')
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser('package'); common(sp)
    sp.add_argument('--output-dir', type=Path, default=HERE / 'artifacts' / 'release_upload')
    sp.add_argument('--prefix', default='ssl-layer-probe')
    sp.set_defaults(func=cmd_package)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
