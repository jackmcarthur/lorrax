"""Capture the real coupled-current tile and time repeated charge fit units."""
import ctypes
import json
from pathlib import Path
import time
from gw import gw_jax, isdf_fitting
import jax
from common import jax_compile_cache
from runtime import run_main_and_finalize

rows = []
captured = False
original_fit = isdf_fitting.fit_one_rchunk
coupled_name = ('_z_q_face_parent' if hasattr(isdf_fitting, '_z_q_face_parent')
                else '_z_q_face_coupled_mu123')
original_coupled = getattr(isdf_fitting, coupled_name)


def timed(label, call):
    """Receipt synchronized host duration and compiler work of one real unit."""
    state = jax_compile_cache._STATE
    before = state.compiles, state.compile_secs
    start = time.perf_counter()
    with jax.profiler.TraceAnnotation(label):
        result = call()
        jax.block_until_ready(result)
    row = dict(label=label, seconds=time.perf_counter()-start,
               compiles=state.compiles-before[0],
               compile_seconds=state.compile_secs-before[1], shape=list(result.shape))
    rows.append(row)
    print('[prof-zw-zeta] '+json.dumps(row), flush=True)
    return result


def coupled(*args, **kwargs):
    """Repeat the same coupled tile without altering the fit's returned values."""
    global captured
    if not kwargs.get('coupled_mu123', coupled_name.endswith('coupled_mu123')):
        return original_coupled(*args, **kwargs)
    result = timed('zeta_T_tile_first', lambda: original_coupled(*args, **kwargs))
    cudart = ctypes.CDLL('libcudart.so') if not captured and jax.process_index() == 0 else None
    if cudart and cudart.cudaProfilerStart() != 0:
        raise RuntimeError('cudaProfilerStart failed')
    try:
        repeat = timed('zeta_T_tile_warm', lambda: original_coupled(*args, **kwargs))
    finally:
        jax.effects_barrier()
        if cudart and cudart.cudaProfilerStop() != 0:
            raise RuntimeError('cudaProfilerStop failed')
    captured = True
    if float(jax.numpy.max(jax.numpy.abs(result-repeat))) != 0:
        raise AssertionError('Repeated coupled tile differs')
    return result


def fit(**kwargs):
    """Repeat charge fit units, retaining the original solve result."""
    if kwargs.get('vertex_mu_L', 0):
        return original_fit(**kwargs)
    result = timed('zeta_C_fit_first', lambda: original_fit(**kwargs))
    repeat = timed('zeta_C_fit_warm', lambda: original_fit(**kwargs))
    if float(jax.numpy.max(jax.numpy.abs(result-repeat))) != 0:
        raise AssertionError('Repeated charge fit differs')
    return result


isdf_fitting.fit_one_rchunk = fit
setattr(isdf_fitting, coupled_name, coupled)


def main():
    """Persist unit receipts before the intentional post-fit exit."""
    try:
        return gw_jax.main()
    finally:
        Path(f'unit_timings.rank{jax.process_index()}.json').write_text(json.dumps(rows, indent=2)+'\n')
        jax_compile_cache._report()


run_main_and_finalize(main)
