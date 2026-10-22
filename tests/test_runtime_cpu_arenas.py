"""CPU allocator startup policy, including its actual process-wide witness.

Run on compute ranks with the normal CPU MPI prelude. The decision tests run
before worker startup; the integration check then inspects glibc malloc_info
after the real JAX/MPI runtime has created its worker pools.
"""
import ctypes
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

from runtime.blas_env import tune_cpu_malloc_arenas


class ArenaPolicy(unittest.TestCase):
    def setUp(self):
        # This policy reads backend state without initializing it. Keep the
        # decision tests independent of a test runner's already-live backend.
        scope=patch('runtime.blas_env.sys',SimpleNamespace(modules={}))
        scope.start()
        self.addCleanup(scope.stop)
        self.env={}
        scope=patch('runtime.blas_env.os',SimpleNamespace(environ=self.env))
        scope.start()
        self.addCleanup(scope.stop)
        scope=patch('runtime.blas_env.env_falsy',return_value=False)
        scope.start()
        self.addCleanup(scope.stop)

    def test_accelerator_does_not_open_libc(self):
        with patch('ctypes.CDLL', side_effect=AssertionError('unexpected libc open')):
            self.assertFalse(tune_cpu_malloc_arenas('cuda,cpu')['applied'])
            self.assertFalse(tune_cpu_malloc_arenas('rocm')['applied'])

    def test_existing_accelerator_client_wins_over_environment(self):
        fake=SimpleNamespace(modules={'jax._src.xla_bridge':SimpleNamespace(_default_backend='cuda')})
        with patch('runtime.blas_env.sys',fake), \
                patch('ctypes.CDLL',side_effect=AssertionError('unexpected libc open')):
            self.assertFalse(tune_cpu_malloc_arenas('cpu')['applied'])

    def test_caller_arena_choices_win(self):
        for values in ({'MALLOC_ARENA_MAX':'7'}, {'MALLOC_ARENA_MAX':''},
                       {'GLIBC_TUNABLES':'glibc.malloc.arena_max=5'}):
            self.env.clear()
            self.env.update(values)
            with self.subTest(values=values), \
                    patch('ctypes.CDLL',side_effect=AssertionError('unexpected libc open')):
                self.assertFalse(tune_cpu_malloc_arenas('cpu')['applied'])

    def test_existing_malloc_optout(self):
        with patch('runtime.blas_env.env_falsy',return_value=True), \
                patch('ctypes.CDLL',side_effect=AssertionError('unexpected libc open')):
            self.assertFalse(tune_cpu_malloc_arenas('cpu')['applied'])

    def test_missing_or_rejecting_glibc(self):
        with patch('ctypes.CDLL',return_value=object()):
            self.assertEqual(tune_cpu_malloc_arenas('cpu')['reason'],'not glibc')
        libc=Mock(gnu_get_libc_version=Mock(),mallopt=Mock(return_value=0))
        with patch('ctypes.CDLL',return_value=libc):
            self.assertFalse(tune_cpu_malloc_arenas('cpu')['applied'])
            libc.mallopt.assert_called_once_with(-8,2)


def actual_cpu_startup():
    import runtime
    stack=runtime.initialize_communicator_stack()
    import jax
    if jax.local_devices()[0].platform!='cpu':
        raise RuntimeError('this integration check requires CPU ranks')
    policy=stack.facts['malloc_tune']['cpu_arenas']
    if 'MALLOC_ARENA_MAX' in os.environ or any(
            v.startswith('glibc.malloc.arena_max=')
            for v in os.environ.get('GLIBC_TUNABLES','').split(':')):
        assert not policy['applied'],policy
    else:
        assert policy['applied'] and policy['maximum']==2,policy
        libc=ctypes.CDLL(None)
        libc.fopen.argtypes=[ctypes.c_char_p,ctypes.c_char_p]
        libc.fopen.restype=ctypes.c_void_p
        libc.fclose.argtypes=[ctypes.c_void_p]
        libc.malloc_info.argtypes=[ctypes.c_int,ctypes.c_void_p]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'malloc.xml'
            stream=libc.fopen(str(p).encode(),b'w')
            assert stream and libc.malloc_info(0,stream)==0
            libc.fclose(stream)
            arenas=len(ET.parse(p).getroot().findall('heap'))
        assert arenas<=2,arenas
        print(f'CPU_ARENAS PASS rank={jax.process_index()} actual_arenas={arenas}',flush=True)
    runtime.finalize_process(0)


if __name__=='__main__':
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(ArenaPolicy))
    if not result.wasSuccessful(): raise SystemExit(1)
    actual_cpu_startup()
