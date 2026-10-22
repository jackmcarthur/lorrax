"""CPU compiler heap lifecycle, with injected readers and no environment edits."""
import sys
import types
import unittest
from unittest.mock import Mock, patch

from common import jax_compile_cache as cache

MiB = 1 << 20
CPU = types.SimpleNamespace(platform="cpu")


class CpuCompileHeapTests(unittest.TestCase):
    def test_cumulative_growth_natural_fall_and_tiny_followup(self):
        state = cache._CacheState()
        runtime = types.SimpleNamespace(_MALLOC_TUNE=dict(applied=True, trim_mb=128))
        resident = [100, 140, 140, 200, 200, 240, 110, 110, 111, 20, 160, 30]
        info = Mock(return_value=types.SimpleNamespace(fordblks=256 * MiB))
        trim = Mock(return_value=1)
        with patch.object(cache, "_STATE", state), patch.dict(sys.modules, runtime=runtime), \
             patch.object(cache, "_cpu_compile_rss", side_effect=[x * MiB for x in resident]), \
             patch.object(cache, "_cpu_malloc_functions", return_value=(info, trim)):
            # Several individually subthreshold compiles accumulate past128MiB.
            for _ in range(3):
                cache._cpu_compile_heap(CPU)
                cache._cpu_compile_heap(CPU, after=True)
            self.assertEqual(state.cpu_heap_trims, 1)
            self.assertEqual(state.cpu_heap_rss, 110 * MiB)
            # A subsequent tiny compile does not scan or trim the free heap.
            cache._cpu_compile_heap(CPU)
            cache._cpu_compile_heap(CPU, after=True)
            self.assertEqual(info.call_count, 1)
            # A natural RSS fall lowers the watermark before a later compile.
            cache._cpu_compile_heap(CPU)
            self.assertEqual(state.cpu_heap_rss, 20 * MiB)
            cache._cpu_compile_heap(CPU, after=True)
        self.assertEqual(trim.call_count, 2)
        self.assertEqual(state.cpu_heap_trim_bytes, 260 * MiB)

    def test_live_heap_and_other_libc_are_not_reclaimed(self):
        state = cache._CacheState()
        runtime = types.SimpleNamespace(_MALLOC_TUNE=dict(applied=True, trim_mb=128))
        trim = Mock()
        info = Mock(return_value=types.SimpleNamespace(fordblks=64 * MiB))
        with patch.object(cache, "_STATE", state), patch.dict(sys.modules, runtime=runtime), \
             patch.object(cache, "_cpu_compile_rss", side_effect=[10 * MiB, 200 * MiB]), \
             patch.object(cache, "_cpu_malloc_functions", return_value=(info, trim)):
            cache._cpu_compile_heap(CPU)
            cache._cpu_compile_heap(CPU, after=True)
        trim.assert_not_called()
        with patch.object(cache, "_STATE", state), patch.dict(sys.modules, runtime=runtime), \
             patch.object(cache, "_cpu_compile_rss", return_value=200 * MiB), \
             patch.object(cache, "_cpu_malloc_functions", return_value=None):
            cache._cpu_compile_heap(CPU, after=True)
        self.assertEqual(state.cpu_heap_trims, 0)

    def test_gpu_and_allocator_opt_out_do_not_read_host_heap(self):
        runtime = types.SimpleNamespace(_MALLOC_TUNE=dict(applied=False))
        with patch.dict(sys.modules, runtime=runtime), \
             patch.object(cache, "_cpu_compile_rss", side_effect=AssertionError("unexpected RSS read")), \
             patch.object(cache, "_cpu_malloc_functions", side_effect=AssertionError("unexpected libc call")):
            for after in (False, True):
                cache._cpu_compile_heap(types.SimpleNamespace(platform="cuda"), after=after)
                cache._cpu_compile_heap(CPU, after=after)
            runtime._MALLOC_TUNE = dict(applied=True, trim_mb=-1)
            cache._cpu_compile_heap(CPU)
            cache._cpu_compile_heap(CPU, after=True)

    def test_persistent_hit_skips_cleanup_and_real_compile_keeps_result(self):
        compiler = types.ModuleType("jax._src.compiler")
        compiler.backend_compile_and_load = lambda *args, **kwargs: "live-executable"
        def request(backend, module):
            if module == "cache-hit":
                return "cached-executable"
            return compiler.backend_compile_and_load(backend, module, (), None, ())
        compiler.compile_or_get_cached = request
        jax = types.ModuleType("jax")
        src = types.ModuleType("jax._src")
        src.compiler = compiler
        jax._src = src
        state = cache._CacheState()
        with patch.object(cache, "_STATE", state), \
             patch.dict(sys.modules, {"jax": jax, "jax._src": src, "jax._src.compiler": compiler}), \
             patch.object(cache, "_cpu_compile_heap") as cleanup:
            cache._install_compile_counter()
            self.assertEqual(compiler.compile_or_get_cached(CPU, "cache-hit"), "cached-executable")
            cleanup.assert_not_called()
            self.assertEqual(compiler.compile_or_get_cached(CPU, "cold-module"), "live-executable")
            self.assertEqual(cleanup.call_args_list[0].args, (CPU,))
            self.assertEqual(cleanup.call_args_list[1].kwargs, {"after": True})
            self.assertEqual(state.compiles, 1)


if __name__ == "__main__":
    unittest.main()
