"""Audit all frame-head checkpoints against data partitions and saved predictions."""
import csv
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
import frame_experiment as experiment
from models import FrameDenseHead

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=HERE/'artifacts'/'runs'/'matched_loso')
    parser.add_argument('--features-dir', type=Path, default=HERE/'artifacts'/'frame_features')
    parser.add_argument('--clip-table', type=Path, default=HERE/'data'/'clip_table.csv')
    parser.add_argument('--audio-dir', type=Path, default=HERE/'artifacts'/'audio')
    parser.add_argument('--max-checkpoints', type=int, default=0, help='Smoke-test limit; no complete audit marker is written.')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = args.results_dir
    manifest = json.loads((root/'validation_manifest.json').read_text())
    folds = {f['name']:f for f in manifest['folds']}
    rows = experiment.shared.read_clip_table(args.clip_table,args.audio_dir)
    by_id = {r['id']:r for r in rows}
    report = {'passed':False,'checkpoints':0,'predictions_reproduced':0,'device':str(device),'devices_in_training':{},'normalization_training_only':True,'selection_uses_minimum_validation_loss':True}
    normalization = {}
    validation_predictions=[]
    validation_metrics=[]
    histories=[]
    for name in ['hubert','wav2vec2']:
        frames = experiment.load_frames(name,rows,args.features_dir)
        for pooling in ['mean','attention']:
            paths = sorted((root/'runs'/f'{name}_dense_{pooling}').glob('seed_*/*/classifier.pt'))
            assert len(paths)==72
            for path in paths:
                checkpoint = torch.load(path,map_location='cpu',weights_only=True)
                result = json.loads((path.parent/'result.json').read_text())
                # Replay using the fit's original device to avoid cross-device rounding differences.
                replay_device = torch.device(result['device'])
                fold = folds[checkpoint['split']]
                key = (name,fold['name'])
                if key not in normalization:
                    normalization[key] = experiment.standardizer(frames,fold['train_ids'])
                mean,std = normalization[key]
                np.testing.assert_array_equal(checkpoint['mean'].numpy(),mean)
                np.testing.assert_array_equal(checkpoint['std'].numpy(),std)
                chosen = min(result['history'],key=lambda r:r['validation_loss'])['epoch']
                assert chosen == result['best_epoch'] == checkpoint['best_epoch']
                assert result['predictions_sha256']==experiment.shared.sha256(path.parent/'predictions.csv')
                assert result['validation_persons'] == fold['validation_persons']
                assert checkpoint['fingerprint'] == result['fingerprint']
                model = FrameDenseHead(pooling).to(replay_device)
                model.load_state_dict(checkpoint['model_state_dict'])
                normalized = {i:(frames[i]-mean)/std for i in fold['test_ids']+fold['validation_ids']}
                _,metrics,predictions = experiment.predict(model,fold['test_ids'],normalized,by_id,replay_device,32)
                with (path.parent/'predictions.csv').open(newline='') as f:
                    saved = list(csv.DictReader(f))
                assert len(saved)==60
                for i,p,r in zip(fold['test_ids'],predictions,saved):
                    assert r['id']==i and r['true']==by_id[i]['label']
                    assert r['prediction']==experiment.shared.LABELS[p]
                assert metrics==result['test']
                val_loss,val_scores,val_predictions=experiment.predict(model,fold['validation_ids'],normalized,by_id,replay_device,32)
                selected=next(h for h in result['history'] if h['epoch']==result['best_epoch'])
                np.testing.assert_allclose([val_loss,val_scores['uar'],val_scores['war']],
                                           [selected['validation_loss'],selected['validation_uar'],selected['validation_war']],rtol=1e-6,atol=1e-7)
                common={'method':result['method'],'seed':result['seed'],'split':result['split']}
                validation_predictions.extend({'id':i,'true':by_id[i]['label'],
                    'prediction':experiment.shared.LABELS[p],**common} for i,p in zip(fold['validation_ids'],val_predictions))
                validation_metrics.append({**common,'best_epoch':result['best_epoch'],
                    'validation_loss':val_loss,'validation_uar':val_scores['uar'],'validation_war':val_scores['war'],'n':val_scores['n']})
                histories.extend({**common,**h,'selected_checkpoint':h['epoch']==result['best_epoch']} for h in result['history'])
                report['checkpoints']+=1
                report['predictions_reproduced']+=60
                if args.max_checkpoints and report['checkpoints']>=args.max_checkpoints:
                    print('Smoke audit passed; partial output is not a complete experiment audit.',flush=True)
                    return
                source_device=result['device']
                report['devices_in_training'][source_device]=report['devices_in_training'].get(source_device,0)+1
            print(f'Audited {name} {pooling}: {report["checkpoints"]}/288 fits.',flush=True)
    assert report['checkpoints']==288 and report['predictions_reproduced']==17280
    summary=json.loads((root/'summary.json').read_text())
    assert summary['complete'] and summary['completed_fits']==288 and len(summary['methods'])==4
    assert len(validation_predictions)==51840
    experiment.evaluation.write_csv(root/'validation_predictions.csv',validation_predictions,experiment.evaluation.PREDICTION_HEADER)
    experiment.evaluation.write_csv(root/'validation_metrics.csv',validation_metrics,list(validation_metrics[0]))
    experiment.evaluation.write_csv(root/'training_history.csv',histories,list(histories[0]))
    report['validation_predictions_exported']=len(validation_predictions)
    report['selected_validation_metrics_match_recorded_history']=True
    report['passed']=True
    (root/'checkpoint_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    main()
