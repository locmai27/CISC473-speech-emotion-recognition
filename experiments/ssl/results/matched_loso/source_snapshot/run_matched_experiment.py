"""Run and audit the matched LOSO classifier comparison without overwriting earlier results."""
from __future__ import annotations
import csv
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent.parent
RESULTS=HERE/'frame_results_matched'


def run(command):
    print('RUN: '+' '.join(map(str,command)),flush=True)
    subprocess.run(list(map(str,command)),check=True)


def finalize(started):
    from extract_features import atomic_json, sha256
    summary=json.loads((RESULTS/'summary.json').read_text())
    assert summary['complete'] and summary['completed_fits']==288 and len(summary['methods'])==4
    assert json.loads((RESULTS/'checkpoint_audit.json').read_text())['passed']
    assert json.loads((RESULTS/'shared_scorer_verification.json').read_text())['matched']
    assert json.loads((RESULTS/'split_audit.json').read_text())['passed']
    rows=[]
    for path in sorted((RESULTS/'runs').glob('*/*/*/result.json')):
        result=json.loads(path.read_text())
        assert result['device']=='cuda'
        rows.append({'method':result['method'],'seed':result['seed'],'split':result['split'],
                     'test_person':result['test_person'],'validation_persons':';'.join(result['validation_persons']),
                     'best_epoch':result['best_epoch'],'epochs_run':len(result['history']),
                     'uar':result['test']['uar'],'war':result['test']['war'],
                     'elapsed_seconds':result['elapsed_seconds'],'device':result['device'],
                     'torch_version':result['torch_version']})
    assert len(rows)==288
    with (RESULTS/'fold_metrics.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    titles={'hubert_dense_mean':'HuBERT + mean pooling','hubert_dense_attention':'HuBERT + attention pooling',
            'wav2vec2_dense_mean':'wav2vec 2.0 + mean pooling','wav2vec2_dense_attention':'wav2vec 2.0 + attention pooling'}
    latex=[r'\begin{center}',r'\begin{tabular}{lcc}',r'\toprule',
           r'\textbf{Configuration} & \textbf{UAR (\%)} & \textbf{WAR (\%)} \\',r'\midrule']
    for item in summary['methods']:
        latex.append(titles[item['method']]+f" & ${100*item['uar_mean']:.2f} \\pm {100*item['uar_std']:.2f}$ & ${100*item['war_mean']:.2f} \\pm {100*item['war_std']:.2f}$" + r' \\')
    latex += [r'\bottomrule',r'\end{tabular}',
              r'\captionof{table}{Matched-split SSL results on eight-class RAVDESS. Each seed pools predictions across all 24 LOSO folds; values are mean $\pm$ sample standard deviation over seeds 42, 43, and 44. The next three speaker IDs cyclically provide validation, and the other 20 speakers train.}',
              r'\label{tab:matched_ssl}',r'\end{center}']
    (RESULTS/'results_table.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')
    methods={r['method']:r for r in summary['methods']}
    lines=['', '## Results by training seed', '', '| Configuration | Seed | UAR (%) | WAR (%) |', '|---|---:|---:|---:|']
    for item in summary['methods']:
        for seed in item['seeds']:
            lines.append(f"| {titles[item['method']]} | {seed['seed']} | {100*seed['uar']:.3f} | {100*seed['war']:.3f} |")
    lines += ['', '## Pooling comparison', '']
    for encoder in ['hubert','wav2vec2']:
        baseline=methods[encoder+'_dense_mean']; attention=methods[encoder+'_dense_attention']
        delta=100*(attention['uar_mean']-baseline['uar_mean'])
        deltas=[100*(a['uar']-b['uar']) for a,b in zip(attention['seeds'],baseline['seeds'])]
        lines.append(f"- {encoder}: attention minus mean UAR = {delta:+.3f} percentage points; seed-wise differences: {', '.join(f'{d:+.3f}' for d in deltas)}. No significance claim.")
    lines += ['', '## Verified evaluation and remaining report work', '',
              '- Speaker partitions match the next-three cyclic rule confirmed by the user. See speaker_splits.csv and validation_manifest.json. The teammate implementation itself was not independently inspected.',
              '- The supplied clip table and original test sets were verified; all 2880 cached feature files match their original audio hashes, labels, speaker IDs, and encoder fingerprints.',
              '- All 288 fits used CUDA. All selected checkpoints were replayed and all 17,280 predictions reproduced. The shared JavaScript scorer agrees with the Python summary.',
              '- Normalization is fitted on each new training partition, not on validation/test speakers. Checkpoints minimize validation loss.',
              '- Seed values 42,43,44 are retained by user instruction; identical teammate seeds are not assumed.',
              '- Replace old provisional SSL numbers and pooling interpretations throughout the report. Use results_table.tex and the new confusion matrices/learning_curves.png.',
              '- MFCC-BiLSTM results and final settings must come from Anastasia. A comparison with published scores under their original protocol has not been reproduced by this run.',
              '- These final-layer, unaugmented runs do not include HuBERT layer probing or acoustic augmentation.', '']
    with (RESULTS/'midterm_results.md').open('a',encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    atomic_json(RESULTS/'runtime_summary.json',{'devices':{'cuda':288},
        'total_fit_compute_seconds':sum(r['elapsed_seconds'] for r in rows),
        'orchestrator_elapsed_seconds':time.monotonic()-started})
    complete={'complete':True,'fits':288,'configurations':4,'seeds':[42,43,44],'folds_per_seed':24,
              'predictions':17280,'cyclic_speaker_rule_verified':True,'checkpoint_replay_passed':True,
              'shared_scorer_matched':True,'feature_cache_verified':True,
              'validation_manifest_sha256':sha256(RESULTS/'validation_manifest.json'),
              'completed_at_utc':datetime.now(timezone.utc).isoformat(),'report':'midterm_results.md'}
    atomic_json(RESULTS/'COMPLETE.json',complete)
    print(json.dumps(complete,indent=2),flush=True)


def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    kernel=ctypes.windll.kernel32
    kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint]
    kernel.SetThreadExecutionState.restype=ctypes.c_uint
    if not kernel.SetThreadExecutionState(0x80000001):
        raise ctypes.WinError()
    try:
        run([sys.executable,'-u',HERE/'prepare_matched_splits.py'])
        snapshot=RESULTS/'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for filename in ['frame_experiment.py','benchmark_frame_heads.py','extract_features.py',
                         'train_classifiers.py','prepare_matched_splits.py','run_matched_experiment.py']:
            shutil.copy2(HERE/filename,snapshot/filename)
        run([sys.executable,'-u',HERE/'frame_experiment.py','--phase','train','--device','cuda',
             '--results-dir',RESULTS,'--validation-manifest',RESULTS/'validation_manifest.json',
             '--log-dir',HERE/'logs'/'frames_matched'])
        run([sys.executable,'-u',PROJECT/'work'/'audit_frame_results.py','--results-dir',RESULTS])
        run(['C:/Program Files/nodejs/node.exe',PROJECT/'work'/'verify_frame_scorer.cjs',RESULTS])
        run([sys.executable,'-u',HERE/'plot_training_curves.py','--results-dir',RESULTS])
        finalize(started)
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        print('Temporary idle sleep prevention released.',flush=True)


if __name__=='__main__':
    main()
