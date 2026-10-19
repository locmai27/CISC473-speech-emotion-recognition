import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import evaluation
import plot_confusions
from models import FrameDenseHead
from download_artifacts import safe_extract


class ExperimentContracts(unittest.TestCase):
    def test_cyclic_wraparound_and_leakage_rejection(self):
        with (HERE/'data/clip_table.csv').open(newline='') as handle: rows=list(csv.DictReader(handle))
        by_id={r['id']:r for r in rows}
        source=json.loads((HERE/'data/splits.json').read_text())
        splits=source['splits'] if isinstance(source,dict) else source
        manifest=evaluation.validation_manifest(splits,by_id,HERE/'data/validation_manifest.json')
        folds={f['test_person']:f for f in manifest['folds']}
        for test,val in [('01',['02','03','04']),('22',['23','24','01']),('23',['24','01','02']),('24',['01','02','03'])]:
            self.assertEqual(folds[test]['validation_persons'],val)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'manifest.json'
            manifest['folds'][0]['train_ids'].append(manifest['folds'][0]['test_ids'][0])
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): evaluation.validation_manifest(splits,by_id,path)

    def test_padding_cannot_change_predictions(self):
        torch.manual_seed(42)
        values=torch.randn(2,6,768)
        mask=torch.tensor([[True]*3+[False]*3,[True]*6])
        changed=values.clone(); changed[0,3:]=10000.
        for pooling in ['mean','attention']:
            model=FrameDenseHead(pooling).eval()
            with torch.inference_mode():
                torch.testing.assert_close(model(values,mask),model(changed,mask))

    def test_uniform_attention_matches_mean(self):
        torch.manual_seed(43)
        mean=FrameDenseHead('mean').eval(); attention=FrameDenseHead('attention').eval()
        attention.load_state_dict(mean.state_dict(),strict=False)
        torch.nn.init.zeros_(attention.attention.weight)
        values=torch.randn(2,5,768); mask=torch.tensor([[True]*2+[False]*3,[True]*5])
        with torch.inference_mode(): torch.testing.assert_close(mean(values,mask),attention(values,mask))

    def test_methods_are_scored_separately_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'predictions.csv'
            rows=[]
            for method in ['perfect','uninformed']:
                for seed in ['42','43','44']:
                    for index,label in enumerate(plot_confusions.LABELS):
                        rows.append({'id':str(index),'true':label,'prediction':label if method=='perfect' else 'calm',
                                     'split':'fold','method':method,'seed':seed})
            evaluation.write_csv(path,rows,evaluation.PREDICTION_HEADER)
            matrices=plot_confusions.prediction_matrices([path])
            self.assertAlmostEqual(matrices['perfect'][1]['uar_mean'],1.)
            self.assertAlmostEqual(matrices['uninformed'][1]['uar_mean'],.125)
            evaluation.write_csv(path,rows+[rows[0]],evaluation.PREDICTION_HEADER)
            with self.assertRaises(ValueError): plot_confusions.prediction_matrices([path])

    def test_aggregate_count_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'counts.csv'
            with path.open('w',newline='') as handle:
                writer=csv.writer(handle)
                writer.writerow(['true_label',*plot_confusions.LABELS])
                for index,label in enumerate(plot_confusions.LABELS):
                    writer.writerow([label,*[3 if column==index else 0 for column in range(8)]])
            counts=plot_confusions.load_counts(path)
            np.testing.assert_array_equal(counts,np.eye(8,dtype=int)*3)

    def test_archive_cannot_escape_or_replace_different_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); archive=root/'test.zip'; output=root/'output'
            with zipfile.ZipFile(archive,'w') as zipped: zipped.writestr('../outside.txt','bad')
            with self.assertRaises(ValueError): safe_extract(archive,output)
            self.assertFalse((root/'outside.txt').exists())
            with zipfile.ZipFile(archive,'w') as zipped: zipped.writestr('inside.txt','original')
            safe_extract(archive,output)
            (output/'inside.txt').write_text('changed')
            with self.assertRaises(FileExistsError): safe_extract(archive,output)
            self.assertEqual((output/'inside.txt').read_text(),'changed')


if __name__=='__main__': unittest.main()
