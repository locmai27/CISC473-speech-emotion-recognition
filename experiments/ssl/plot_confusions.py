"""Generate count and row-normalized confusion matrices, separately for every method."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import evaluation

LABELS=sorted(evaluation.LABELS)
HERE=Path(__file__).resolve().parent


def prediction_matrices(paths,partition='test'):
    groups={}
    seen=set()
    for path in paths:
        with path.open(encoding='utf-8-sig',newline='') as handle:
            reader=csv.DictReader(handle)
            if reader.fieldnames!=evaluation.PREDICTION_HEADER:
                raise ValueError(f'{path}: expected header {evaluation.PREDICTION_HEADER}')
            for row in reader:
                if row['true'] not in LABELS or row['prediction'] not in LABELS:
                    raise ValueError('Unknown RAVDESS label.')
                # Test clips must occur once per method and seed. Validation clips
                # legitimately recur in different folds, so include the fold there.
                key=(row['method'],row['seed'],row['id'])+((row['split'],) if partition=='validation' else ())
                if key in seen: raise ValueError(f'Duplicate prediction: {key}')
                seen.add(key)
                groups.setdefault(row['method'],{}).setdefault(row['seed'],[]).append(row)
    result={}
    for method,seeds in groups.items():
        counts=np.zeros((8,8),dtype=np.int64)
        scores=[]
        coverage=None
        for seed,rows in sorted(seeds.items()):
            ids={r['id'] for r in rows}
            if coverage is not None and ids!=coverage: raise ValueError('Seed recording coverage differs.')
            coverage=ids
            true=[LABELS.index(r['true']) for r in rows]
            predicted=[LABELS.index(r['prediction']) for r in rows]
            score=evaluation.metrics(true,predicted)
            counts+=np.asarray(score['confusion'])
            scores.append({'seed':seed,'uar':score['uar'],'war':score['war'],'predictions':len(rows)})
        result[method]=(counts,{'partition':partition,'seed_scores':scores,
                               'uar_mean':float(np.mean([s['uar'] for s in scores])),
                               'war_mean':float(np.mean([s['war'] for s in scores])),
                               'uar_std':float(np.std([s['uar'] for s in scores],ddof=1)) if len(scores)>1 else None,
                               'war_std':float(np.std([s['war'] for s in scores],ddof=1)) if len(scores)>1 else None})
    if not result: raise ValueError('No predictions.')
    return result


def load_counts(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.reader(handle); header=next(reader)
        if header!=['true_label']+LABELS: raise ValueError('Expected true_label followed by alphabetically ordered emotion columns.')
        rows=list(reader)
    if [r[0] for r in rows]!=LABELS: raise ValueError('Unexpected row labels.')
    counts=np.asarray([[int(x) for x in r[1:]] for r in rows],dtype=np.int64)
    if counts.shape!=(8,8) or np.any(counts<0) or np.any(counts.sum(1)==0): raise ValueError('Invalid confusion counts.')
    return counts


def plot(method,counts,metadata,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not method or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in method):
        raise ValueError('Method names must contain only letters, digits, underscores and hyphens.')
    normalized=counts/counts.sum(1,keepdims=True)
    for kind,values in [('counts',counts),('row_normalized',normalized)]:
        fig,ax=plt.subplots(figsize=(9,8))
        maximum=1. if kind=='row_normalized' else max(1,int(counts.max()))
        heat=ax.imshow(values,cmap='Blues',vmin=0,vmax=maximum)
        for i in range(8):
            for j in range(8):
                text=f'{values[i,j]:.2f}' if kind=='row_normalized' else str(values[i,j])
                ax.text(j,i,text,ha='center',va='center',fontsize=9,color='white' if values[i,j]>maximum/2 else 'black')
        ax.set(xticks=range(8),yticks=range(8),xticklabels=LABELS,yticklabels=LABELS,
               xlabel='Predicted emotion',ylabel='True emotion',
               title=method.replace('_',' ')+'\n'+metadata.get('partition','test')+' predictions; '+str(int(counts.sum()))+' total')
        plt.setp(ax.get_xticklabels(),rotation=40,ha='right')
        fig.colorbar(heat,ax=ax,label='Fraction of true class' if kind=='row_normalized' else 'Prediction count')
        fig.tight_layout()
        for extension in ['png','pdf']: fig.savefig(output/(method+'_'+kind+'.'+extension),dpi=180)
        plt.close(fig)
        with (output/(method+'_'+kind+'.csv')).open('w',newline='',encoding='utf-8') as handle:
            writer=csv.writer(handle); writer.writerow(['true_label']+LABELS)
            writer.writerows([[label]+row.tolist() for label,row in zip(LABELS,values)])
    metadata={**metadata,'labels':LABELS,'total_predictions':int(counts.sum()),
              'pooled_uar':float(np.diag(normalized).mean()),'pooled_war':float(np.trace(counts)/counts.sum())}
    (output/(method+'_metrics.json')).write_text(json.dumps(metadata,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,nargs='+')
    parser.add_argument('--counts-csv',type=Path)
    parser.add_argument('--method',help='Required with --counts-csv.')
    parser.add_argument('--partition',choices=['test','validation'],default='test')
    parser.add_argument('--output-dir',type=Path,default=HERE/'artifacts/plots')
    args=parser.parse_args()
    if bool(args.predictions)==bool(args.counts_csv): parser.error('Provide exactly one of --predictions and --counts-csv.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.counts_csv:
        if not args.method: parser.error('--counts-csv requires --method.')
        matrices={args.method:(load_counts(args.counts_csv),{'partition':args.partition,
                   'source':'supplied aggregate counts; seed-level predictions unavailable','seed_scores':None})}
    else:
        matrices=prediction_matrices(args.predictions,args.partition)
    for method,(counts,metadata) in matrices.items():
        plot(method,counts,metadata,args.output_dir)
        print(f'{method}: {counts.sum()} predictions; saved count/normalized PNG, PDF, CSV and metrics.',flush=True)


if __name__=='__main__': main()
