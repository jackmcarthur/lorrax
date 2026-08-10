"""THE RED TWINS of the cache contract, as three tiny four-rank programs.

NEVER AUTO-COLLECTED — the leading underscore keeps it out of ``test_*.py``
globbing, exactly like ``tests/_env_leak_twin.py``.  It is driven by
``tests/test_jax_cache_contract.py``, which launches it at P=4 through
``tests/mesh_launch.py`` and asserts on the resulting key dumps.

WHY THE TWINS ARE PROGRAMS AND NOT MOCKS
-----------------------------------------
The contract's two arms fail for two unrelated reasons, and a gate that
cannot tell them apart is a gate that will one day report the wrong one:

``veto``        carries a ``jax.debug.callback`` inside the jit.  JAX
                REFUSES to write a persistent entry for any module with a
                host callback (``jax/_src/compiler.py::_cache_write``), so
                the module is never on disk, every rank misses it, and
                every rank recompiles.  This is class A of the campaign's
                two cache-PITA classes (FIX_warmcache.md).  Its key set is
                the SAME on every rank, so it must take ARM 2 red and leave
                ARM 3 green — that discrimination is the point.

``rankstatic``  passes ``jax.process_index()`` as a STATIC argument, so each
                rank compiles a different program and holds a different key.
                This is class B (FIX_multislice_cachekey.md).  It must take
                ARM 3 red.

``clean``       the control.  Same shape, same number of jits, nothing
                rank-dependent and no callback.  If the control is not green
                the harness is broken and neither twin means anything.

Both twins are taken against JAX's OWN behaviour — a real persistent cache
directory, a real four-process agreement — and not against a proxy for it,
so they also fail if jax changes the rules out from under this contract.

Usage (the launcher supplies the four processes)::

    ISDF_JAX_CACHE_DIR=<dir> LORRAX_JAX_CACHE_KEYDUMP=<dir> \\
        python tests/_cache_contract_probe.py {clean|veto|rankstatic}
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

KINDS = ("clean", "veto", "rankstatic")


def main(argv) -> int:
    kind = (argv[1] if len(argv) > 1 else "clean").strip()
    if kind not in KINDS:
        print(f"unknown probe kind {kind!r}; want one of {KINDS}",
              file=sys.stderr)
        return 2

    from runtime import init_jax_distributed
    init_jax_distributed()

    import jax
    import jax.numpy as jnp
    from common.jax_compile_cache import (compile_cache_stats,
                                          ensure_jax_compile_cache)

    ensure_jax_compile_cache()

    idx = jax.process_index()
    n = jax.process_count()
    x = jnp.arange(16.0).reshape(4, 4)

    # -- the shared payload: two jits every arm compiles identically ------
    @jax.jit
    def _sq(a):
        return a * a + 1.0

    @jax.jit
    def _row(a):
        return jnp.sin(a).sum(axis=0)

    out = float(_sq(x).sum()) + float(_row(x).sum())

    # -- the arm-specific third jit ---------------------------------------
    if kind == "veto":
        # CLASS A.  The callback makes the module unpersistable; the
        # program is identical on every rank, so only ARM 2 may go red.
        @jax.jit
        def _with_callback(a):
            s = a.sum()
            jax.debug.callback(lambda v: None, s)
            return s * 2.0

        out += float(_with_callback(x))
    elif kind == "rankstatic":
        # CLASS B.  ``idx`` is STATIC, so rank r compiles a program no
        # other rank compiles: four ranks, four keys, one shared cache
        # directory that only process 0 may write to.
        @partial(jax.jit, static_argnums=(1,))
        def _rank_static(a, r):
            return a.sum() * (r + 1.0)

        out += float(_rank_static(x, int(idx)))
    else:
        @jax.jit
        def _third(a):
            return jnp.cos(a).sum()

        out += float(_third(x))

    s = compile_cache_stats()
    print(f"[probe {kind}] rank {idx}/{n} out={out:.6f} "
          f"compiles={s['compiles']} probes={s['probes']} hits={s['hits']} "
          f"vetoed={s['vetoed']} keys={len(s['keys'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
