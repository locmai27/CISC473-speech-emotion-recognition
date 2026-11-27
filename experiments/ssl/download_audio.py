"""Download the original RAVDESS speech archive and verify it against the clip table."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
import extract_features as shared

HERE=Path(__file__).resolve().parent
URL='https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1'
ARCHIVE_MD5='bc696df654c87fed845eb13823edef8a'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir',type=Path,default=HERE/'artifacts/audio')
    parser.add_argument('--archive',type=Path,default=HERE/'artifacts/Audio_Speech_Actors_01-24.zip')
    parser.add_argument('--clip-table',type=Path,default=HERE/'data/clip_table.csv')
    args=parser.parse_args()
    args.archive.parent.mkdir(parents=True,exist_ok=True)
    if not args.archive.exists():
        temporary=args.archive.with_suffix('.zip.part')
        request=urllib.request.Request(URL,headers={'User-Agent':'CISC473-SSL/1.0'})
        with urllib.request.urlopen(request,timeout=60) as response,temporary.open('wb') as handle:
            count=0
            while block:=response.read(1024*1024):
                handle.write(block); count+=len(block)
                if count%(20*1024*1024)<len(block): print(f'Downloaded {count/2**20:.0f} MiB',flush=True)
        temporary.replace(args.archive)
    digest=hashlib.md5()
    with args.archive.open('rb') as handle:
        while block:=handle.read(1024*1024): digest.update(block)
    if digest.hexdigest()!=ARCHIVE_MD5: raise ValueError('Official archive checksum mismatch.')
    with args.clip_table.open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    expected={r['id'] for r in rows}
    with zipfile.ZipFile(args.archive) as archive:
        members=[m for m in archive.infolist() if m.filename.lower().endswith('.wav') and not m.filename.startswith('__MACOSX/')]
        if len(members)!=1440 or {Path(m.filename).stem for m in members}!=expected:
            raise ValueError('Archive differs from the shared clip table.')
        for member in members:
            clip_id=Path(member.filename).stem
            target=args.audio_dir/('Actor_'+clip_id.split('-')[-1])/(clip_id+'.wav')
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(member) as source,target.open('wb') as output:
                shutil.copyfileobj(source,output)
    checked=shared.read_clip_table(args.clip_table,args.audio_dir)
    print(f'Checksum, IDs, speakers and labels verified for {len(checked)} recordings: {args.audio_dir}')
    print('Dataset attribution and terms: https://zenodo.org/records/1188976 (CC BY-NC-SA 4.0).')


if __name__=='__main__': main()
