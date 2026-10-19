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
PROJECT = HERE.parents[1]
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


