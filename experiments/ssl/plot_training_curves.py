"""Plot a fixed example chosen by fold/seed ID, rather than test performance."""
import csv
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parent/'results'/'matched_loso'
METHODS=[('hubert_dense_mean','HuBERT + Dense mean pooling'),
         ('hubert_dense_attention','HuBERT + Dense attention pooling'),
         ('wav2vec2_dense_mean','wav2vec 2.0 + Dense mean pooling'),
         ('wav2vec2_dense_attention','wav2vec 2.0 + Dense attention pooling')]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=ROOT)
    args=parser.parse_args()
    root=args.results_dir
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True)
    rows=[]
    for ax,(method,title) in zip(axes.flat,METHODS):
        result=json.loads((root/'runs'/method/'seed_42'/'left_out_01'/'result.json').read_text())
        history=result['history']
        epochs=[h['epoch'] for h in history]
        ax.plot(epochs,[h['training_loss'] for h in history],label='Training',color='#2877b8',linewidth=1.8)
        ax.plot(epochs,[h['validation_loss'] for h in history],label='Validation',color='#d35d36',linewidth=1.8)
        best=history[result['best_epoch']-1]
        ax.scatter([best['epoch']],[best['validation_loss']],color='#d35d36',edgecolor='black',s=55,zorder=3,label='Selected checkpoint')
        ax.axvline(best['epoch'],color='gray',linestyle=':',alpha=.7)
        ax.set(title=title,xlabel='Epoch',ylabel='Cross-entropy loss',ylim=(0,None))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        for h in history:
            rows.append({'method':method,**h,'selected_checkpoint':h['epoch']==result['best_epoch']})
    fig.suptitle('Learning curves: held-out speaker 01, seed 42\nFixed illustrative fold; checkpoint selected by minimum validation loss',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92))
    fig.savefig(root/'learning_curves.png',dpi=180)
    plt.close(fig)
    with (root/'learning_curves.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved learning_curves.png and learning_curves.csv; example selected by fold/seed ID rather than performance.')

if __name__=='__main__':
    main()
