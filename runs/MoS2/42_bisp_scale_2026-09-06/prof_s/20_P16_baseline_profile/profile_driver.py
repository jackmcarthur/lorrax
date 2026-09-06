"""Measure existing callable boundaries without changing their arithmetic."""
import ctypes
import functools
import json
import os
import time
from pathlib import Path
from gw import gw_jax, photon_sigma, w_isdf
import jax
from common import jax_compile_cache
from runtime import run_main_and_finalize


def snapshot():
    state = jax_compile_cache.compile_cache_stats()
    return state['compiles'], state['compile_secs']


def receipt(label, start, before, metadata):
    count, seconds = snapshot()
    row = dict(label=label, host_ms=1000*(time.perf_counter()-start),
               compiles=count-before[0], compile_s=seconds-before[1], **metadata)
    if int(os.environ['SLURM_PROCID']) == 0:
        with Path('boundary.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print('[boundary] '+json.dumps(row), flush=True)


kernel_calls = 0

def measured(function, label, metadata):
    @functools.wraps(function)
    def call(*args, **kwargs):
        global kernel_calls
        selected = label == 'sigma_block' and kernel_calls == 21
        if label == 'sigma_block':
            kernel_calls += 1
        if selected:
            assert ctypes.CDLL('libcudart.so').cudaProfilerStart() == 0
        before, start = snapshot(), time.perf_counter()
        with jax.profiler.TraceAnnotation(label):
            result = function(*args, **kwargs)
            jax.block_until_ready(result)
        if selected:
            jax.effects_barrier()
            assert ctypes.CDLL('libcudart.so').cudaProfilerStop() == 0
        receipt(label, start, before, metadata)
        return result
    return call


original_factory = photon_sigma._make_photon_static_block_kernel

def factory(*args, **kwargs):
    before, start = snapshot(), time.perf_counter()
    kernel = original_factory(*args, **kwargs)
    receipt('sigma_factory', start, before, dict(options=kwargs))
    return measured(kernel, 'sigma_block', dict(options=kwargs))

photon_sigma._make_photon_static_block_kernel = factory
original_restore = getattr(w_isdf, 'photon_blocks_full_q', None)
if original_restore is not None:
    def restored(*args, **kwargs):
        iterator = original_restore(*args, **kwargs)
        while True:
            before, start = snapshot(), time.perf_counter()
            try:
                with jax.profiler.TraceAnnotation('sigma_restore'):
                    key, value = next(iterator)
                    value.block_until_ready()
            except StopIteration:
                return
            receipt('sigma_restore', start, before, dict(key=key, term=kwargs['term']))
            yield key, value
    w_isdf.photon_blocks_full_q = restored
photon_sigma.compute_static_photon_sigma = measured(
    photon_sigma.compute_static_photon_sigma, 'sigma_static_stage', {})


def captured_main():
    return gw_jax.main()

run_main_and_finalize(captured_main)
