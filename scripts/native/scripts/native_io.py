from pathlib import Path
import os,subprocess,json
BASE=Path(__file__).resolve().parents[3]/'results/figure1/native'
CODE=Path(__file__).resolve().parent
RUNTIME=CODE.parent
def environment(python):
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',MPLCONFIGDIR=str(BASE/'.cache/matplotlib'),NUMBA_CACHE_DIR=str(BASE/'.cache/numba'))
    env['LD_LIBRARY_PATH']=str(Path(python).resolve().parents[1]/'lib')+':'+env.get('LD_LIBRARY_PATH','')
    env.pop('PYTHONPATH',None)
    return env

def run_command(cmd,out,env):
    out.mkdir(parents=True,exist_ok=True)
    (out/'command.json').write_text(json.dumps(list(map(str,cmd)),indent=2)+'\n')
    with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        result=subprocess.run(list(map(str,cmd)),cwd=BASE,env=env,stdout=stdout,stderr=stderr)
    if result.returncode:raise RuntimeError(f'Native command failed: {result.returncode}; see {out}')
