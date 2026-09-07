import compile_receipts
import time, json, functools
import jax
from common import jax_compile_cache
from gw import photon_sigma, gw_jax
from runtime import run_main_and_finalize
name = '_make_photon_static_block_kernel'
original = getattr(photon_sigma, name)

def factory(*args, **kwargs):
    kernel = original(*args, **kwargs)
    carriers = [getattr(x, "green_parent", x) for x in args[3:5]]
    shapes = [list(x.psi_nmu.shape) for x in carriers]
    @functools.wraps(kernel)
    def measured(*a, **kw):
        before = jax_compile_cache.compile_cache_stats()
        start = time.perf_counter()
        value = kernel(*a, **kw)
        jax.block_until_ready(value)
        elapsed = 1000*(time.perf_counter()-start)
        after = jax_compile_cache.compile_cache_stats()
        if jax.process_index() == 0:
            with open('boundary.jsonl', 'a') as f:
                f.write(json.dumps(dict(label=name, options={"shapes": shapes}, host_ms=elapsed,
                    compiles=after['compiles']-before['compiles'],
                    compile_s=after['compile_secs']-before['compile_secs']))+'\n')
        return value
    return measured
setattr(photon_sigma, name, factory)
run_main_and_finalize(gw_jax.main)
