"""The native ELPA thread request respects both OpenMP and process affinity."""
import ctypes.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/ffi/cpp/elpa/eigh_ffi.cc'


def _thread_function():
    """Compile the production metadata owner, without a GPU/ELPA stand-in."""
    text = SOURCE.read_text()
    start = text.index('static int affinity_threads()')
    opened = text.index('{', start)
    depth = 1
    end = opened + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def test_elpa_thread_request_uses_actual_openmp_runtime(tmp_path):
    compiler = shutil.which('g++')
    if compiler is None or ctypes.util.find_library('gomp') is None:
        pytest.skip('the real GNU OpenMP runtime and C++ compiler are required')
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 4:
        pytest.skip('four allowed CPUs are required to separate runtime and affinity')
    header = ROOT / 'src/ffi/cpp/common/mkl_thread_pin.h'
    source = tmp_path / 'thread_cap.cc'
    source.write_text('#include <sched.h>\n#include <algorithm>\n'
                      f'#include "{header}"\n' + _thread_function() +
                      '\nextern "C" int requested_threads() { return affinity_threads(); }\n')
    library = tmp_path / 'thread_cap.so'
    subprocess.run([compiler, '-std=c++17', '-shared', '-fPIC', str(source),
                    '-Wl,--no-as-needed', '-lgomp', '-Wl,--as-needed', '-ldl',
                    '-o', str(library)], check=True, capture_output=True, text=True)
    # A fresh process loads a locally scoped library whose actual dependency
    # supplies OpenMP. Neither a fake vendor function nor an environment parser
    # substitutes for the production resolver/runtime call.
    child = '''import ctypes,json,os,sys
os.sched_setaffinity(0, json.loads(sys.argv[2]))
lib=ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_LOCAL)
lib.requested_threads.restype=ctypes.c_int
print(json.dumps({'requested':lib.requested_threads(),'affinity':len(os.sched_getaffinity(0))}))
'''
    for omp, cpus in ((2, allowed[:4]), (8, allowed[:2]), (6, allowed[:4]), (1, allowed[:4])):
        env = {**os.environ, 'OMP_NUM_THREADS': str(omp)}
        value = json.loads(subprocess.check_output(
            [sys.executable, '-c', child, str(library), json.dumps(cpus)], env=env, text=True))
        assert value == {'requested': min(omp, len(cpus)), 'affinity': len(cpus)}, (omp, value)
