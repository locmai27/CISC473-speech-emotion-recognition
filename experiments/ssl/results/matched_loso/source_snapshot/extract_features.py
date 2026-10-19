"""Download pinned frozen encoders and cache RAVDESS utterance features.

No classifier training, label fitting, augmentation, or fold selection happens here.
Run `python extract_features.py --help` for options.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import sys
import threading
import time
from datetime import datetime


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
MODELS = {
    'hubert': {
        'repo': 'facebook/hubert-base-ls960',
        'revision': 'dba3bb02fda4248b6e082697eee756de8fe8aa8a',
    },
    'wav2vec2': {
        'repo': 'facebook/wav2vec2-base',
        'revision': '0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8',
    },
}
LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
LOG = logging.getLogger('ser.extract')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def setup_logging(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime('%Y%m%d-%H%M%S-%f')
    path = log_dir / f'extract-{stamp}.log'
    formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s')
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for handler in [logging.StreamHandler(sys.stdout),
                    logging.FileHandler(path, encoding='utf-8'),
                    logging.FileHandler(log_dir / 'latest.log', mode='w', encoding='utf-8')]:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    logging.captureWarnings(True)
    LOG.info('Log file: %s', path)
    return path


@contextlib.contextmanager
def heartbeat(phase, interval=20):
    """Keep the log informative during long downloads/model loading."""
    stop = threading.Event()
    started = time.monotonic()
    def report():
        while not stop.wait(interval):
            LOG.info('%s is still running (%ds elapsed).', phase, time.monotonic() - started)
    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def read_clip_table(path, audio_dir):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['id', 'person', 'label']:
            raise ValueError('Clip table header must be exactly id,person,label.')
        rows = list(reader)
    seen = set()
    for row in rows:
        clip_id = row['id']
        if not re.fullmatch(r'03-01-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', clip_id):
            raise ValueError(f'Invalid speech recording ID: {clip_id!r}')
        parts = clip_id.split('-')
        if clip_id in seen:
            raise ValueError(f'Duplicate clip ID: {clip_id}')
        seen.add(clip_id)
        if row['person'] != parts[-1] or not 1 <= int(parts[-1]) <= 24:
            raise ValueError(f'Speaker mismatch: {clip_id}')
        if not 1 <= int(parts[2]) <= 8 or row['label'] != LABELS[int(parts[2]) - 1]:
            raise ValueError(f'Emotion mismatch: {clip_id}')
        row['path'] = Path(audio_dir) / f'Actor_{row["person"]}' / f'{clip_id}.wav'
        if not row['path'].is_file():
            raise FileNotFoundError(f'Audio is missing: {row["path"]}')
    if not rows:
        raise ValueError('Clip table is empty.')
    return rows


def load_audio(path, target_sr=16000):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    values, source_sr = sf.read(str(path), dtype='float32', always_2d=True)
    channels = values.shape[1]
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError(f'Empty or non-finite audio: {path}')
    # The original WAV files remain unchanged. Average channels before resampling.
    mono = values.mean(axis=1, dtype=np.float32)
    if source_sr != target_sr:
        divisor = math.gcd(source_sr, target_sr)
        mono = resample_poly(mono, target_sr // divisor, source_sr // divisor).astype(np.float32)
    if mono.size < 400:
        raise ValueError(f'Recording too short for the encoders: {path}')
    return mono, source_sr, channels


def download_model(name, cache_dir):
    from huggingface_hub import snapshot_download
    spec = MODELS[name]
    LOG.info('[DOWNLOAD] %s @ %s', spec['repo'], spec['revision'])
    LOG.info('Weights are several hundred MB per encoder; terminal progress bars show transfer progress.')
    with heartbeat(f'{name} download/cache check'):
        snapshot = Path(snapshot_download(
            repo_id=spec['repo'], revision=spec['revision'], cache_dir=str(cache_dir),
            allow_patterns=['config.json', 'preprocessor_config.json', 'pytorch_model.bin'],
            max_workers=1,
        ))
    for filename in ['config.json', 'preprocessor_config.json', 'pytorch_model.bin']:
        path = snapshot / filename
        if not path.is_file():
            raise FileNotFoundError(f'Model download incomplete: {path}')
        LOG.info('[READY] %s: %.1f MiB', filename, path.stat().st_size / 1024**2)
    LOG.info('Model cache: %s', snapshot)
    return snapshot


def check_cache(path, fingerprint, audio_hash, expected_shape):
    import numpy as np
    with np.load(path, allow_pickle=False) as cached:
        if str(cached['fingerprint']) != fingerprint:
            raise ValueError(f'Cache settings changed: {path}. Use a fresh --output-dir.')
        if str(cached['audio_sha256']) != audio_hash:
            raise ValueError(f'Audio changed since extraction: {path}. Use a fresh --output-dir.')
        if cached['features'].shape != expected_shape or not np.isfinite(cached['features']).all():
            raise ValueError(f'Invalid feature cache: {path}. Move that file aside and rerun.')


def extract_model(name, snapshot, rows, args, device, packages):
    import numpy as np
    import torch
    from transformers import HubertModel, Wav2Vec2Model, Wav2Vec2FeatureExtractor
    spec = MODELS[name]
    LOG.info('[LOAD] Loading %s on %s in float32.', name, device)
    with heartbeat(f'{name} model loading'):
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(snapshot), local_files_only=True)
        model_cls = HubertModel if name == 'hubert' else Wav2Vec2Model
        model, loading_info = model_cls.from_pretrained(
            str(snapshot), local_files_only=True, output_loading_info=True,
            attn_implementation='eager',
        )
        if loading_info.get('missing_keys') or loading_info.get('mismatched_keys') or loading_info.get('error_msgs'):
            raise RuntimeError(f'Encoder did not load all required pretrained weights: {loading_info}')
        if loading_info.get('unexpected_keys'):
            LOG.info('Unused pretraining-only weights: %s', loading_info['unexpected_keys'])
        model.requires_grad_(False)
        model.eval()
        model.config.apply_spec_augment = False
        model.to(device)
    if extractor.sampling_rate != 16000:
        raise ValueError(f'Unexpected checkpoint sample rate: {extractor.sampling_rate}')
    layers = list(range(1, model.config.num_hidden_layers + 1)) if name == 'hubert' and args.hubert_layers == 'all' else [model.config.num_hidden_layers]
    mode = 'all' if len(layers) > 1 else 'last'
    destination = args.output_dir / name / mode
    clips_dir = destination / 'clips'
    clips_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        'schema_version': 1, 'model': spec['repo'], 'revision': spec['revision'],
        'script_sha256': sha256(__file__), 'sample_rate': 16000,
        'channel_policy': 'arithmetic_mean', 'resampling': 'scipy.signal.resample_poly',
        'waveform_normalization': extractor.to_dict(), 'pooling': 'mean_over_all_unpadded_frames',
        'layer_indices': layers, 'hidden_size': model.config.hidden_size,
        'dtype': 'float32', 'attention_implementation': 'eager', 'clips_per_forward': 1,
        'encoder_frozen': True, 'encoder_training_mode': False, 'augmentation': False,
        'packages': packages,
    }
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    settings['fingerprint'] = fingerprint
    settings_path = destination / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text(encoding='utf-8')) != settings:
        raise ValueError(f'Extraction settings differ from {settings_path}. Use a fresh --output-dir.')
    atomic_json(settings_path, settings)
    LOG.info('[EXTRACT] %s | clips=%d | Transformer layers=%s | hidden size=%d', name, len(rows), layers, model.config.hidden_size)
    LOG.info('Frozen=True | eval=True | gradients=False | augmentation=False | no padding/truncation')
    LOG.info('Checkpoint waveform normalization is per recording; no dataset statistics are fitted.')
    LOG.info('Saving one pooled vector per layer per clip to %s', clips_dir)
    started = time.monotonic()
    created = resumed = 0
    output_files = []
    for index, row in enumerate(rows, start=1):
        path = clips_dir / f'{row["id"]}.npz'
        source_hash = sha256(row['path'])
        expected_shape = (len(layers), model.config.hidden_size)
        if path.exists():
            check_cache(path, fingerprint, source_hash, expected_shape)
            resumed += 1
            action = 'cached'
        else:
            clip_started = time.monotonic()
            mono, source_sr, channels = load_audio(row['path'])
            if channels > 1:
                LOG.info('[AUDIO] %s: %d channels -> mono.', row['id'], channels)
            inputs = extractor(mono, sampling_rate=16000, padding=False, return_tensors='pt')
            input_values = inputs['input_values'].to(device)
            with torch.inference_mode():
                # Each forward uses one complete recording, so pooling includes no padding.
                outputs = model(input_values=input_values, output_hidden_states=len(layers) > 1, return_dict=True)
                if len(layers) > 1:
                    pooled = torch.stack([outputs.hidden_states[layer][0].mean(dim=0) for layer in layers])
                else:
                    pooled = outputs.last_hidden_state[0].mean(dim=0).unsqueeze(0)
                features = pooled.float().cpu().numpy()
                frame_count = outputs.last_hidden_state.shape[1]
            if features.shape != expected_shape or not np.isfinite(features).all():
                raise ValueError(f'Invalid encoder features for {row["id"]}: {features.shape}')
            atomic_npz(path, features=features, layer_indices=np.asarray(layers),
                       clip_id=np.asarray(row['id']), label=np.asarray(row['label']),
                       person=np.asarray(row['person']), fingerprint=np.asarray(fingerprint),
                       audio_sha256=np.asarray(source_hash), frame_count=np.asarray(frame_count),
                       original_sample_rate=np.asarray(source_sr), original_channels=np.asarray(channels),
                       resampled_samples=np.asarray(mono.size))
            if created == 0:
                LOG.info('[EXAMPLE] %s | %d Hz -> 16000 Hz | %.2fs | frames=%d | saved shape=%s | %.2fs compute',
                         row['id'], source_sr, mono.size / 16000, frame_count, features.shape, time.monotonic() - clip_started)
            del outputs, pooled, input_values
            created += 1
            action = 'saved'
        output_files.append(path)
        if index == 1 or index % args.log_every == 0 or index == len(rows):
            elapsed = time.monotonic() - started
            LOG.info('[%s] %4d/%d (%5.1f%%) | %s %s | new=%d cached=%d | elapsed=%.1fmin | rough remaining=%.1fmin',
                     name, index, len(rows), 100 * index / len(rows), action, row['id'],
                     created, resumed, elapsed / 60, elapsed / index * (len(rows) - index) / 60)
    vectors = []
    for path in output_files:
        with np.load(path, allow_pickle=False) as cached:
            vectors.append(cached['features'])
    combined_path = destination / 'embeddings.npz'
    atomic_npz(combined_path, features=np.stack(vectors), ids=np.asarray([r['id'] for r in rows]),
               labels=np.asarray([r['label'] for r in rows]), persons=np.asarray([r['person'] for r in rows]),
               layer_indices=np.asarray(layers), fingerprint=np.asarray(fingerprint))
    summary = {
        'model': spec['repo'], 'revision': spec['revision'], 'clips': len(rows),
        'new_clips': created, 'resumed_clips': resumed, 'layer_indices': layers,
        'array_shape': [len(rows), len(layers), model.config.hidden_size],
        'elapsed_seconds': time.monotonic() - started, 'device': str(device),
        'clip_table_sha256': sha256(args.clip_table), 'fingerprint': fingerprint,
        'complete_table': len(rows) == args.full_table_size,
        'created_at': datetime.now().astimezone().isoformat(),
    }
    atomic_json(destination / 'summary.json', summary)
    LOG.info('[DONE] %s | combined shape=%s | %s', name, summary['array_shape'], combined_path)
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--audio-dir', type=Path, default=HERE.parent / 'ravdess' / 'audio')
    parser.add_argument('--clip-table', type=Path, default=HERE.parent / 'ravdess' / 'clip_table.csv')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'features')
    parser.add_argument('--cache-dir', type=Path, default=PROJECT / 'work' / 'hf-cache')
    parser.add_argument('--log-dir', type=Path, default=HERE / 'logs')
    parser.add_argument('--models', choices=['both', 'hubert', 'wav2vec2'], default='both')
    parser.add_argument('--hubert-layers', choices=['last', 'all'], default='last')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--limit', type=int, default=0, help='First N clips for a smoke test; 0 means the full table.')
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--cpu-threads', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--download-only', action='store_true')
    return parser


def main():
    args = build_parser().parse_args()
    setup_logging(args.log_dir)
    if args.limit < 0 or args.log_every < 1 or args.cpu_threads < 1:
        raise ValueError('--limit must be nonnegative; --log-every and --cpu-threads must be positive.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    LOG.info('Python %s | %s | script=%s', platform.python_version(), platform.platform(), __file__)
    LOG.info('Importing PyTorch and Transformers...')
    try:
        import torch
        import numpy as np
        import transformers
        import soundfile
    except (ImportError, OSError) as error:
        raise RuntimeError('Dependencies could not be loaded. Run run.ps1 without -SkipSetup.') from error
    packages = {name: importlib.metadata.version(name) for name in ['torch','transformers','huggingface-hub','numpy','scipy','soundfile','tokenizers','safetensors']}
    LOG.info('Packages: %s', packages)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Use run.ps1 -Device cuda to install GPU PyTorch, or select cpu.')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
    LOG.info('Device: %s | GPU: %s | CPU threads: %d', device,
             torch.cuda.get_device_name(0) if device.type == 'cuda' else 'unused', args.cpu_threads)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = list(MODELS) if args.models == 'both' else [args.models]
    if args.download_only:
        for name in names:
            download_model(name, args.cache_dir)
        LOG.info('SUCCESS: checkpoints downloaded. No features extracted.')
        return
    LOG.info('[VALIDATE] clip table: %s | audio: %s', args.clip_table, args.audio_dir)
    rows = read_clip_table(args.clip_table, args.audio_dir)
    args.full_table_size = len(rows)
    LOG.info('Validated %d recordings, %d speakers, %d emotion labels.', len(rows), len({r['person'] for r in rows}), len({r['label'] for r in rows}))
    if args.limit:
        rows = rows[:args.limit]
        LOG.info('SMOKE TEST: extracting only %d/%d recordings. This is not a complete dataset.', len(rows), args.full_table_size)
    LOG.info('No classifier is trained here. Fixed frozen features can be reused in every LOSO fold.')
    results = []
    for name in names:
        snapshot = download_model(name, args.cache_dir)
        results.append(extract_model(name, snapshot, rows, args, device, packages))
    LOG.info('SUCCESS: %d encoder(s), %d clips each. Results: %s', len(results), len(rows), args.output_dir)
    LOG.info('Next stage: train emotion classifiers using the shared LOSO folds and agreed validation speakers.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOG.warning('Interrupted. Completed clip caches are safe. Rerun the same command to resume.')
        sys.exit(130)
    except Exception:
        LOG.exception('FAILED. Review this error and the log, then rerun to resume completed clips.')
        sys.exit(1)
