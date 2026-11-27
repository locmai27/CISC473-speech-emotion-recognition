"""Download selected release artifacts, verify SHA-256, and safely extract them."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import urllib.request
import zipfile
import zlib

HERE=Path(__file__).resolve().parent


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        while block:=handle.read(1024*1024): digest.update(block)
    return digest.hexdigest()


def safe_extract(path,output):
    output=output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        members=[]
        for item in archive.infolist():
            relative=PurePosixPath(item.filename)
            if relative.is_absolute() or '..' in relative.parts or '\\' in item.filename or ':' in item.filename:
                raise ValueError('Unsafe archive entry: '+item.filename)
            if stat.S_ISLNK(item.external_attr>>16): raise ValueError('Archive symlinks are not accepted.')
            target=(output/item.filename).resolve()
            if not target.is_relative_to(output): raise ValueError('Archive entry leaves the output directory.')
            if target.exists() and not item.is_dir():
                crc=0
                with target.open('rb') as handle:
                    while block:=handle.read(1024*1024): crc=zlib.crc32(block,crc)
                if target.stat().st_size!=item.file_size or (crc&0xffffffff)!=item.CRC:
                    raise FileExistsError('Existing artifact differs; choose a fresh --output-dir: '+str(target))
            members.append((item,target))
        for item,target in members:
            if item.is_dir(): target.mkdir(parents=True,exist_ok=True); continue
            if target.exists(): continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(item) as source,target.open('wb') as handle:
                shutil.copyfileobj(source,handle)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',nargs='+',required=True,
                        choices=['hubert_features','wav2vec2_features','matched_checkpoints','historical'])
    parser.add_argument('--manifest',type=Path,default=HERE/'artifact_manifest.json')
    parser.add_argument('--output-dir',type=Path,default=HERE/'artifacts')
    parser.add_argument('--no-extract',action='store_true')
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    selected={item['id']:item for item in manifest['assets']}
    downloads=args.output_dir/'downloads'; downloads.mkdir(parents=True,exist_ok=True)
    for key in args.assets:
        item=selected[key]; target=downloads/item['name']
        if target.exists():
            if target.stat().st_size!=item['bytes'] or sha256(target)!=item['sha256']:
                raise ValueError('Existing archive checksum mismatch: '+str(target))
        else:
            temporary=target.with_suffix('.zip.part')
            request=urllib.request.Request(item['url'],headers={'User-Agent':'CISC473-SSL/1.0'})
            with urllib.request.urlopen(request,timeout=120) as response,temporary.open('wb') as handle:
                total=0
                while block:=response.read(1024*1024):
                    handle.write(block); total+=len(block)
                    if total%(50*1024*1024)<len(block): print(f'{item["name"]}: {total/2**20:.0f} MiB',flush=True)
            if temporary.stat().st_size!=item['bytes'] or sha256(temporary)!=item['sha256']:
                raise ValueError('Downloaded archive checksum mismatch: '+item['name'])
            temporary.replace(target)
        if not args.no_extract: safe_extract(target,args.output_dir)
        print('Verified '+item['name'],flush=True)


if __name__=='__main__': main()
