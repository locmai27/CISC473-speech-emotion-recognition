"""Time two frozen-feature heads; synthetic timing inputs, not SER results.

DenseMean follows Pepino's frame-wise 128/128 Dense head. DenseAttention
changes only the final time-pooling operation. Input normalization and any
encoder-layer mixture belong upstream and are not included in this benchmark.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch import nn


class FrameDenseHead(nn.Module):
    def __init__(self, pooling='mean', input_size=768, hidden_size=128, classes=8):
        super().__init__()
        self.frame_network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.2),
        )
        self.attention = nn.Linear(hidden_size, 1, bias=False) if pooling == 'attention' else None
        self.classifier = nn.Linear(hidden_size, classes)

    def forward(self, features, mask):
        hidden = self.frame_network(features)
        if self.attention is None:
            weights = mask.to(hidden.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True)
        else:
            scores = self.attention(hidden).squeeze(-1).masked_fill(~mask, float('-inf'))
            weights = scores.softmax(dim=1)
        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(pooled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--frames', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'frame_head_timing.json')
    args = parser.parse_args()
    if args.steps < 5:
        parser.error('Use at least five measured steps.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU PyTorch is not installed; use --device cpu.')
    x = torch.randn(args.batch_size, args.frames, 768, device=device)
    mask = torch.ones(args.batch_size, args.frames, dtype=torch.bool, device=device)
    target = torch.randint(0, 8, (args.batch_size,), device=device)
    criterion = nn.CrossEntropyLoss()
    results = {}
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    print('SYNTHETIC COMPUTE BENCHMARK ONLY: no emotion-recognition scores are measured.', flush=True)
    print(f'Device={device}, threads={args.threads}, batch={args.batch_size}, frames={args.frames}', flush=True)
    for pooling in ['mean', 'attention']:
        model = FrameDenseHead(pooling=pooling).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        def train_step():
            optimizer.zero_grad(set_to_none=True)
            criterion(model(x, mask), target).backward()
            optimizer.step()
        model.train()
        for _ in range(5):
            train_step()
        timings = []
        for _ in range(args.steps):
            synchronize()
            start = time.perf_counter()
            train_step()
            synchronize()
            timings.append(time.perf_counter() - start)
        model.eval()
        eval_times = []
        with torch.inference_mode():
            for _ in range(10):
                synchronize()
                start = time.perf_counter()
                model(x, mask)
                synchronize()
                eval_times.append(time.perf_counter() - start)
        train_time, eval_time = statistics.median(timings), statistics.median(eval_times)
        train_steps = math.ceil(1200 / args.batch_size) * 20
        eval_steps = math.ceil(180 / args.batch_size) * 20 + math.ceil(60 / args.batch_size)
        one_fit = train_steps * train_time + eval_steps * eval_time
        # Per head: two encoders, 24 folds, three seeds.
        total_fits = 2 * 24 * 3
        results[pooling] = {
            'parameters': sum(p.numel() for p in model.parameters()),
            'median_training_step_seconds': train_time,
            'median_evaluation_step_seconds': eval_time,
            'estimated_20_epoch_fit_compute_seconds': one_fit,
            'fits_per_head': total_fits,
            'estimated_compute_minutes_all_fits': one_fit * total_fits / 60,
        }
        print(pooling, json.dumps(results[pooling]), flush=True)
    total = sum(row['estimated_compute_minutes_all_fits'] for row in results.values())
    report = {
        'benchmark_only': True, 'real_training_not_run': True,
        'torch': torch.__version__, 'device': str(device), 'threads': args.threads,
        'batch_size': args.batch_size, 'frames_per_batch': args.frames,
        'measured_steps_per_head': args.steps, 'models': results,
        'total_classifier_fits_in_projection': 288,
        'estimated_total_compute_minutes': total,
        'excludes': ['encoder feature extraction','disk I/O','data normalization','batch preparation','checkpoint saving','hyperparameter search'],
        'warning': 'A timing projection is not a guaranteed end-to-end runtime. Real one-fold timing is needed before a full run.',
    }
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Projected compute for both heads and both encoders: {total:.1f} minutes, excluding pipeline overhead.',flush=True)
    print(f'Saved: {args.output}',flush=True)


if __name__ == '__main__':
    main()
