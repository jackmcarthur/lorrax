"""Measure real screening calls and capture one repeated warm TT block."""
import ctypes
import hashlib
import json
from pathlib import Path
import time

from gw import gw_jax, w_isdf
import jax
import numpy as np
from common import jax_compile_cache
from runtime import run_main_and_finalize

original_chi = w_isdf.compute_no_pair_dirac_current_block
original_restore = getattr(w_isdf, 'photon_blocks_full_q', None)
records = []


def timed(label, function):
    """Block a real public unit and receipt its process-local compiler work."""
    state = jax_compile_cache._STATE
    before = state.compiles, state.compile_secs
    start = time.perf_counter()
    with jax.profiler.TraceAnnotation(label):
        result = function()
        jax.block_until_ready(result)
    row = dict(label=label, seconds=time.perf_counter()-start,
               compiles=state.compiles-before[0],
               compile_seconds=state.compile_secs-before[1])
    records.append(row)
    print('[prof-zw] ' + json.dumps(row), flush=True)
    return result


def chi(*args, **kwargs):
    """Repeat the same block arguments without changing the production result."""
    pair = kwargs['vertex_left'], kwargs['vertex_right']
    quad = args[2]
    metadata = dict(pair=pair, nodes=len(quad.tau),
                    node_digest=hashlib.sha256(np.asarray(quad.tau).tobytes()+np.asarray(quad.alpha).tobytes()).hexdigest())
    for name, wfns in zip(('left', 'right'), args[:2]):
        parent = getattr(wfns, 'green_parent', None)
        face = parent.psi_mun if parent is not None else wfns.psi_mun
        metadata[name+'_shape'] = list(face.shape)
        metadata[name+'_plan_id'] = id(parent.plan) if parent is not None else None
    print('[prof-zw-metadata] ' + json.dumps(metadata), flush=True)
    result = timed(f'chi_{pair}_first', lambda: original_chi(*args, **kwargs))
    cudart = ctypes.CDLL('libcudart.so') if pair == (1, 1) and jax.process_index() == 0 else None
    if cudart and cudart.cudaProfilerStart() != 0:
        raise RuntimeError('cudaProfilerStart failed')
    try:
        warm = timed(f'chi_{pair}_warm', lambda: original_chi(*args, **kwargs))
    finally:
        jax.effects_barrier()
        if cudart and cudart.cudaProfilerStop() != 0:
            raise RuntimeError('cudaProfilerStop failed')
    delta = jax.numpy.max(jax.numpy.abs(result-warm))
    if float(delta) != 0:
        raise AssertionError(f'Repeated chi differs: {delta}')
    return result


def restore(response, keys, *, term='W'):
    """Time each consumed output block without changing its one-block lifetime."""
    for key in keys:
        iterator = original_restore(response, (key,), term=term)
        value = timed(f'restore_{term}_{key}', lambda: next(iterator)[1])
        yield key, value
        del value
        iterator.close()



original_solve = w_isdf.solve_w


def solve(V, chi0, meta, mesh, **kwargs):
    """Repeat Dyson with a private donated chi copy and the same public owner."""
    saved = jax.numpy.array(chi0, copy=True)
    saved.block_until_ready()
    print('[prof-zw-dyson-shape] '+str(chi0.shape), flush=True)
    result = timed('dyson_first', lambda: original_solve(V, chi0, meta, mesh, **kwargs))
    repeat = timed('dyson_warm', lambda: original_solve(V, saved, meta, mesh, **kwargs))
    if float(jax.numpy.max(jax.numpy.abs(result-repeat))) != 0:
        raise AssertionError('Repeated Dyson differs')
    return result


from gw import head_correction
original_completion = head_correction.complete_static_slab_photon_q0


def completion(*args, **kwargs):
    """Time the real Gamma completion without repeating its donated operands."""
    return timed('gamma_completion', lambda: original_completion(*args, **kwargs))


from common.centroid_basis import PackedCentroidBasis


def seam(original, name):
    """Count only public host seams, excluding traced calls inside other units."""
    def call(self, value, *args, **kwargs):
        if isinstance(value, jax.core.Tracer):
            return original(self, value, *args, **kwargs)
        label = f'{name}_mu{self.n_logical}_packed{self.n_packed}_shape{value.shape}'
        return timed(label, lambda: original(self, value, *args, **kwargs))
    return call


for name in ('pack_axis', 'unpack_axis', 'pack_operator', 'unpack_operator'):
    original = getattr(PackedCentroidBasis, name)
    setattr(PackedCentroidBasis, name, seam(original, name))
w_isdf.solve_w = solve
head_correction.complete_static_slab_photon_q0 = completion

w_isdf.compute_no_pair_dirac_current_block = chi
if original_restore is not None:
    w_isdf.photon_blocks_full_q = restore


def main():
    """Write per-rank unit receipts even if the science driver refuses."""
    try:
        return gw_jax.main()
    finally:
        Path(f'unit_timings.rank{jax.process_index()}.json').write_text(json.dumps(records, indent=2)+'\n')
        jax_compile_cache._report()


run_main_and_finalize(main)
