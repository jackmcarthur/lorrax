"""Record native compiler calls by the existing cache identity and Python caller."""
import json,os,time,traceback
from pathlib import Path
from jax._src import compiler
from common import jax_compile_cache as cache
original=getattr(compiler,cache._COMPILE_ENTRY_POINT)

def compile_one(*args,**kwargs):
 module=args[1] if len(args)>1 else kwargs['module']
 name,digest,_=cache._compile_module_identity(module)
 stack=[dict(file=f.filename,line=f.lineno,function=f.name) for f in traceback.extract_stack()
        if '/src/' in f.filename and '/jax/' not in f.filename]
 start=time.perf_counter()
 try:
  return original(*args,**kwargs)
 finally:
  seconds=time.perf_counter()-start
  if int(os.environ.get('SLURM_PROCID','0'))==0:
   with Path('compile_modules.jsonl').open('a') as stream:
    stream.write(json.dumps(dict(name=name,digest=digest,seconds=seconds,stack=stack))+'\n')
setattr(compiler,cache._COMPILE_ENTRY_POINT,compile_one)
