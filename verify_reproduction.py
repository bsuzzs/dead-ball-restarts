"""Compare newly computed outputs with frozen aggregate reference results."""
import argparse,json,math,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def compare_json(actual,expected,path,errors):
    if isinstance(expected,dict):
        if not isinstance(actual,dict):errors.append(f'{path}: expected object');return
        for k,v in expected.items():
            if k not in actual:errors.append(f'{path}.{k}: missing')
            else:compare_json(actual[k],v,f'{path}.{k}',errors)
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):errors.append(f'{path}: list shape differs');return
        for i,(a,b) in enumerate(zip(actual,expected)):compare_json(a,b,f'{path}[{i}]',errors)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        if not isinstance(actual,(int,float)) or not (math.isnan(expected) and math.isnan(actual) or math.isclose(actual,expected,rel_tol=1e-6,abs_tol=1e-8)):
            errors.append(f'{path}: {actual!r} != {expected!r}')
    elif actual!=expected:errors.append(f'{path}: value differs')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(os.environ.get('RESTART_OUT',ROOT/'outputs')))
    args=parser.parse_args();errors=[];count=0
    refs=sorted(p for p in (ROOT/'reference_results').iterdir() if p.suffix in ('.csv','.json') and not p.name.startswith('.'))
    for reference in refs:
        actual=args.output/reference.name
        if not actual.is_file():errors.append(f'{reference.name}: missing output');continue
        try:
            if reference.suffix=='.csv':
                pd.testing.assert_frame_equal(pd.read_csv(actual),pd.read_csv(reference),check_exact=False,check_dtype=False,rtol=1e-6,atol=1e-8)
            else:compare_json(json.loads(actual.read_text()),json.loads(reference.read_text()),reference.name,errors)
            count+=1
        except (AssertionError,ValueError,OSError) as exc:errors.append(f'{reference.name}: {str(exc)[:300]}')
    for name in ['Figure'+str(i) for i in range(1,8)]+['FigureS'+str(i) for i in range(1,6)]:
        for ext in ['png','pdf','svg','tiff']:
            p=args.output/'figures'/f'{name}.{ext}'
            if not p.is_file() or not p.stat().st_size:errors.append(f'{p.name}: missing/empty figure')
    result={'passed':not errors,'reference_files_compared':count,'expected_reference_files':len(refs),'rtol':1e-6,'atol':1e-8,'errors':errors}
    print(json.dumps(result,indent=2,ensure_ascii=False));return 0 if result['passed'] else 1
if __name__=='__main__':sys.exit(main())
