import compile_receipts
import time, json, functools
import jax
original_jit = jax.jit

def jit(fun=None, *args, **kwargs):
    result = original_jit(fun, *args, **kwargs)
    if getattr(fun, '__name__', '') != 'finish':
        return result
    @functools.wraps(result)
    def measured(value):
        jax.block_until_ready(value)
        start = time.perf_counter()
        out = result(value)
        jax.block_until_ready(out)
        if jax.process_index() == 0:
            with open('finish_units.jsonl', 'a') as f:
                f.write(json.dumps({'ms': 1000*(time.perf_counter()-start)})+'\n')
        return out
    return measured
jax.jit = jit
from gw import gw_jax
from runtime import run_main_and_finalize
run_main_and_finalize(gw_jax.main)
