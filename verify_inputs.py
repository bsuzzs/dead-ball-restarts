"""Verify required raw inputs against the recorded SHA-256 manifest."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(os.environ.get('RESTART_DATA',ROOT/'data/raw')))
    args=parser.parse_args();manifest=json.loads((ROOT/'data/source_sha256.json').read_text());errors=[]
    for name,expected in manifest.items():
        path=args.data/name
        if not path.is_file():errors.append(f'{name}: missing');continue
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected:errors.append(f'{name}: SHA-256 mismatch')
    print(json.dumps({'passed':not errors,'expected_files':len(manifest),'errors':errors},indent=2));return 1 if errors else 0
if __name__=='__main__':sys.exit(main())
