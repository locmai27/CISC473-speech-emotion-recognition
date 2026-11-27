"""Create and verify cyclic next-three-speaker validation partitions."""
import argparse
import csv
import json
from pathlib import Path
import evaluation
import extract_features as shared

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip-table',type=Path,default=HERE/'data/clip_table.csv')
    parser.add_argument('--splits',type=Path,default=HERE/'data/splits.json')
    parser.add_argument('--output',type=Path,default=HERE/'artifacts/validation_manifest.json')
    args=parser.parse_args()
    with args.clip_table.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames != ['id','person','label']: raise ValueError('Expected id,person,label header.')
        rows=list(reader)
    by_id={r['id']:r for r in rows}
    if len(by_id)!=len(rows): raise ValueError('Duplicate clip ID.')
    document=json.loads(args.splits.read_text(encoding='utf-8-sig'))
    splits=document['splits'] if isinstance(document,dict) else document
    folds=[]
    for split in splits:
        test=int(split['test_person'])
        val=[f'{((test-1+i)%24)+1:02d}' for i in (1,2,3)]
        pool=split['train_ids']
        folds.append({'name':split['name'],'test_person':split['test_person'],
                      'validation_persons':val,'training_persons':sorted({by_id[i]['person'] for i in pool}-set(val)),
                      'train_ids':[i for i in pool if by_id[i]['person'] not in val],
                      'validation_ids':[i for i in pool if by_id[i]['person'] in val],
                      'test_ids':split['test_ids']})
    manifest={'policy_id':'cyclic_next_3','policy':'Next three speaker IDs cyclically from 01 through 24 validate; the other 20 train.',
              'clip_table_sha256':shared.sha256(args.clip_table),'shared_splits_sha256':shared.sha256(args.splits),'folds':folds}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists() and json.loads(args.output.read_text())!=manifest:
        raise ValueError('Refusing to replace a different manifest; choose a new --output.')
    shared.atomic_json(args.output,manifest)
    evaluation.validation_manifest(splits,by_id,args.output)
    print(f'Verified 24 folds, disjoint 20/3/1 speakers, 1200/180/60 clips: {args.output}')


if __name__=='__main__': main()
