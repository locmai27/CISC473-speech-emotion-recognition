"""Verify the supplied clips, cyclic speaker partitions, and reusable frame caches."""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
import extract_features as shared
import train_classifiers as evaluation

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'frame_results_matched'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    clip_path = HERE.parent/'ravdess'/'clip_table.csv'
    split_path = HERE.parent/'ravdess'/'splits.json'
    rows = shared.read_clip_table(clip_path, HERE.parent/'ravdess'/'audio')
    by_id = {r['id']:r for r in rows}
    assert len(rows) == len(by_id) == 1440
    assert Counter(r['person'] for r in rows) == {f'{i:02d}':60 for i in range(1,25)}
    supplied_table = Path('C:/Users/locsf/Downloads/clip_table (1).csv')
    supplied_splits = Path('C:/Users/locsf/Downloads/splits (1).json')
    assert sorted(read_csv(clip_path), key=lambda r:r['id']) == sorted(read_csv(supplied_table), key=lambda r:r['id'])
    document = json.loads(split_path.read_text(encoding='utf-8-sig'))
    assert document == json.loads(supplied_splits.read_text(encoding='utf-8-sig'))
    splits = document['splits'] if isinstance(document,dict) else document
    assert len(splits) == 24
    folds = []
    for split in splits:
        speaker = int(split['test_person'])
        validation = [f'{((speaker-1+offset)%24)+1:02d}' for offset in (1,2,3)]
        training = sorted({by_id[i]['person'] for i in split['train_ids']} - set(validation))
        folds.append({'name':split['name'], 'test_person':split['test_person'],
                      'validation_persons':validation, 'training_persons':training,
                      'train_ids':[i for i in split['train_ids'] if by_id[i]['person'] not in validation],
                      'validation_ids':[i for i in split['train_ids'] if by_id[i]['person'] in validation],
                      'test_ids':split['test_ids']})
    manifest = {'policy_id':'cyclic_next_3',
                'policy':'For test speaker s, validation speakers are the next three IDs cyclically from 01 through 24; the other 20 speakers train.',
                'confirmation_source':'Anastasia rule reported directly by user in this conversation.',
                'matches_anastasia_validation_confirmed':True,
                'note':'Matches the confirmed speaker-selection rule; independent access to the teammate training artifacts was not provided. Training seeds are 42,43,44 and need not match teammate seeds.',
                'clip_table_sha256':shared.sha256(clip_path),
                'shared_splits_sha256':shared.sha256(split_path), 'folds':folds}
    path = ROOT/'validation_manifest.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == manifest, 'Existing matched manifest differs.'
    else:
        shared.atomic_json(path,manifest)
    evaluation.validation_manifest(splits,by_id,path)
    by_test = {f['test_person']:f for f in folds}
    for speaker,expected in [('01',['02','03','04']),('22',['23','24','01']),
                             ('23',['24','01','02']),('24',['01','02','03'])]:
        assert by_test[speaker]['validation_persons'] == expected
    evaluation.write_csv(ROOT/'speaker_splits.csv',[
        {'fold':f['name'],'test_speaker':f['test_person'],
         'validation_speakers':';'.join(f['validation_persons']),
         'training_speakers':';'.join(f['training_persons']),
         'train_clips':len(f['train_ids']),'validation_clips':len(f['validation_ids']),
         'test_clips':len(f['test_ids'])} for f in folds],
        ['fold','test_speaker','validation_speakers','training_speakers','train_clips','validation_clips','test_clips'])
    print('PASS: 24 cyclic folds, 1200/180/60 clips, disjoint speakers, unchanged original test sets, and supplied clip table match.',flush=True)
    audio_hashes = {r['id']:shared.sha256(r['path']) for r in rows}
    checked = {}
    for name in ['hubert','wav2vec2']:
        folder = HERE/'frame_features'/name
        settings = json.loads((folder/'settings.json').read_text())
        assert settings['layer']==12 and settings['frozen'] and not settings['augmentation']
        assert settings['sample_rate']==16000 and settings['model']==shared.MODELS[name]
        assert len(list(folder.glob('*.npz'))) == 1440
        frames_total = 0
        for index,row in enumerate(rows,1):
            with np.load(folder/(row['id']+'.npz'),allow_pickle=False) as cached:
                assert str(cached['fingerprint']) == settings['fingerprint']
                assert str(cached['audio_sha256']) == audio_hashes[row['id']]
                assert str(cached['clip_id'])==row['id'] and str(cached['person'])==row['person'] and str(cached['label'])==row['label']
                features = cached['features']
                assert features.ndim==2 and features.shape[1]==768 and np.isfinite(features).all()
                frames_total += len(features)
            if index%240==0:
                print(f'CACHE VERIFIED {name}: {index}/1440 clips',flush=True)
        checked[name]={'clips':1440,'frames':frames_total,'feature_fingerprint':settings['fingerprint'],
                       'audio_hashes_match':True,'labels_and_speakers_match':True,'finite_full_sequences':True}
    shared.atomic_json(ROOT/'split_audit.json',{
        'passed':True,'folds':24,'clips':1440,'matches_supplied_clip_table':True,
        'matches_supplied_splits':True,'matches_confirmed_cyclic_rule':True,
        'disjoint_train_validation_test_speakers':True,'all_original_test_sets_preserved':True,
        'test_coverage_exactly_once':True,'partition_sizes':[1200,180,60],
        'validation_manifest_sha256':shared.sha256(path),'cached_features':checked})
    print('PASS: all 2880 feature files retain their original audio, label, speaker and encoder provenance.',flush=True)


if __name__=='__main__':
    main()
