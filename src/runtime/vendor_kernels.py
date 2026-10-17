"""The leg's vendor-library first-use policy: persistent kernel stores and the BLAS warm-up.

Two vendor costs of a ROCm process sit outside XLA and outside JAX's compile
cache (``docs/architecture/compilation.md`` section 3).  :data:`lxkit.LEGS`
declares them per leg and this module implements the declarations; a leg that
declares none (CUDA, the host) gets no-ops, so its behaviour is unchanged.

* ``PlatformLeg.runtime_kernel_stores``: rocFFT compiles each new FFT kernel
  with hipRTC.  Its kernels are keyed by their own content (name, arch, HIP
  version, generator checksum), like LORRAX's kconv image store
  (:func:`ffi.fft.cubin_cache_dir`), so they live in
  ``lxkit.user_cache_dir(name)/<ffi-tag>`` (the enforced native closure
  identity) and outlive any compile-cache namespace.  The
  library reads one snapshot (``reader_env``) and writes new kernels to one
  file per process (``writer_env``; several writers on one SQLite file on
  Lustre spin without progress).  At exit each process folds its file into a
  copy of the snapshot and renames the copy into place, so a reader never sees
  a torn file; two processes folding at once can drop each other's rows, which
  the next run rebuilds.
* ``PlatformLeg.blas_host_warmup``: rocBLAS loads its Tensile tables on a
  process's first GEMM (2.84 s on MI250X).  One 8x8 GEMM per declared dtype is
  compiled here, on the main thread so every rank compiles the same programs
  in the same order (the compile agreement), and run on a helper thread, so
  the load overlaps the driver's imports.  The programs hold no collective.

Nothing here raises: a store that cannot be placed or folded, or a warm-up
that fails, is named once and the library does what it does without it.
"""
from __future__ import annotations

import atexit
import os
import shutil
import socket
import sqlite3
import threading
import time
from pathlib import Path

__all__ = ["arm_runtime_kernel_stores", "fold_into_snapshot", "start_blas_host_warmup",
           "receipt"]

_STORES: list = []        # one record per armed store
_WARMUP: dict = {}        # the BLAS warm-up record (empty: none started)


def _process_leg():
    import jax
    from lxkit import device_vendor, leg_for_vendor
    return leg_for_vendor(device_vendor(jax.local_devices()[0]))


def arm_runtime_kernel_stores(*, proc_idx: int = 0, say=print) -> list:
    """Point the leg's vendor kernel stores at ``lxkit.user_cache_dir``; the armed records.

    Runs after the backend exists and before the first FFT, which is when
    rocFFT reads its variables.  A caller that exported either variable owns
    the library's cache, and the store is not armed.
    """
    try:
        leg = _process_leg()
    except Exception:                                      # noqa: BLE001
        return []
    from lxkit import user_cache_dir
    try:
        from common.jax_compile_cache import native_cache_tag
        tag = native_cache_tag()
    except Exception as exc:                               # noqa: BLE001
        if proc_idx == 0 and leg is not None and leg.runtime_kernel_stores:
            say(f"  [vendor-kernels] no native closure identity ({type(exc).__name__}: "
                f"{exc}); the {leg.key} kernel stores are not armed.")
        return []
    armed = []
    for writer_env, reader_env, name in (leg.runtime_kernel_stores if leg is not None else ()):
        if os.environ.get(writer_env) or os.environ.get(reader_env):
            if proc_idx == 0:
                say(f"  [vendor-kernels] {name}: {writer_env} or {reader_env} exported by "
                    f"the caller; the {leg.key} kernel store is not armed.")
            continue
        # One store per native closure: kernels compiled by one vendor library
        # set are never read by another (lxkit.native_provider).
        root = user_cache_dir(name) / tag
        snapshot = root / f"{name}_kernel_cache.db"
        writer = root / "writers" / f"{socket.gethostname()}.{os.getpid()}.db"
        try:
            writer.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            if proc_idx == 0:
                say(f"  [vendor-kernels] {name}: cannot create {writer.parent} ({exc}); "
                    f"its kernels compile again in every process.")
            continue
        present = snapshot.is_file()
        if present:
            os.environ[reader_env] = str(snapshot)
        os.environ[writer_env] = str(writer)
        rec = {"name": name, "snapshot": str(snapshot), "writer": str(writer),
               "snapshot_present": present, "folded": None, "proc_idx": proc_idx}
        _STORES.append(rec)
        armed.append(rec)
    if armed:
        atexit.register(_fold_at_exit, say)
    return armed


def fold_into_snapshot(writer: Path, snapshot: Path) -> int:
    """Fold ``writer``'s rows into ``snapshot`` (copy, insert-or-ignore, rename); rows added.

    Generic over the tables the writer holds: the vendor's schema is read from
    the writer itself.  The writer is removed afterwards.  An empty or absent
    writer (every kernel came from the snapshot) adds nothing and writes nothing.
    """
    writer, snapshot = Path(writer), Path(snapshot)
    if not writer.is_file():
        return 0
    src = sqlite3.connect(f"file:{writer}?mode=ro", uri=True)
    try:
        tables = src.execute(
            "select name, sql from sqlite_master where type = 'table'").fetchall()
        rows = sum(src.execute(f'select count(*) from "{t}"').fetchone()[0] for t, _ in tables)
    finally:
        src.close()
    if rows == 0:
        writer.unlink(missing_ok=True)
        return 0
    tmp = snapshot.with_name(f"{snapshot.name}.tmp.{socket.gethostname()}.{os.getpid()}")
    if snapshot.is_file():
        shutil.copyfile(snapshot, tmp)
    db = sqlite3.connect(f"file:{tmp}", uri=True)
    try:
        db.execute("attach database ? as w", (f"file:{writer}?mode=ro",))
        before = db.total_changes
        for t, sql in tables:
            db.execute(sql.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1))
            db.execute(f'insert or ignore into main."{t}" select * from w."{t}"')
        added = db.total_changes - before
        db.commit()
        db.execute("detach database w")
    finally:
        db.close()
    with open(tmp, "rb+") as handle:          # durable before it becomes the snapshot
        os.fsync(handle.fileno())
    os.replace(tmp, snapshot)
    writer.unlink(missing_ok=True)
    return added


def _fold_at_exit(say) -> None:
    for rec in _STORES:
        try:
            rec["folded"] = fold_into_snapshot(Path(rec["writer"]), Path(rec["snapshot"]))
        except Exception as exc:                           # noqa: BLE001
            rec["folded"] = f"{type(exc).__name__}: {exc}"
            if rec["proc_idx"] == 0:
                say(f"  [vendor-kernels] {rec['name']}: new kernels not folded into "
                    f"{rec['snapshot']} ({rec['folded']}); the next run compiles them again.")
            continue
        if rec["proc_idx"] == 0 and rec["folded"]:
            say(f"  [vendor-kernels] {rec['name']}: {rec['folded']} new kernels folded "
                f"into {rec['snapshot']}")


def start_blas_host_warmup(*, proc_idx: int = 0, say=print) -> dict:
    """Compile one 8x8 GEMM per ``blas_host_warmup`` dtype here; run them on a helper thread."""
    try:
        leg = _process_leg()
    except Exception:                                      # noqa: BLE001
        return {}
    dtypes = tuple(leg.blas_host_warmup) if leg is not None else ()
    if not dtypes or _WARMUP:
        return dict(_WARMUP)
    _WARMUP.update(leg=leg.key, dtypes=dtypes, compile_s=None, run_s=None, error=None)
    try:
        import numpy as np
        import jax
        import jax.numpy as jnp
        dev = jax.local_devices()[0]
        t0 = time.perf_counter()
        progs = []
        for name in dtypes:
            a = jax.device_put(np.ones((8, 8), dtype=name), dev)
            progs.append((jax.jit(jnp.matmul).lower(a, a).compile(), a))
        _WARMUP["compile_s"] = time.perf_counter() - t0
    except Exception as exc:                               # noqa: BLE001
        _WARMUP["error"] = f"{type(exc).__name__}: {exc}"
        if proc_idx == 0:
            say(f"  [vendor-kernels] BLAS warm-up not started ({_WARMUP['error']}); "
                f"the first GEMM pays the library's host initialisation.")
        return dict(_WARMUP)

    def _run():
        t = time.perf_counter()
        try:
            for compiled, a in progs:
                compiled(a, a).block_until_ready()
            _WARMUP["run_s"] = time.perf_counter() - t
        except Exception as exc:                           # noqa: BLE001
            _WARMUP["error"] = f"{type(exc).__name__}: {exc}"

    th = threading.Thread(target=_run, name="lorrax-blas-warmup", daemon=True)
    _WARMUP["thread"] = th
    th.start()
    return dict(_WARMUP)


def receipt() -> dict:
    """What this process armed and warmed (for the startup facts and a run's stats)."""
    w = {k: v for k, v in _WARMUP.items() if k != "thread"}
    th = _WARMUP.get("thread")
    if th is not None:
        w["done"] = not th.is_alive()
    return {"stores": [dict(r) for r in _STORES], "blas_warmup": w}
