"""Algebra parity: ``isdf.core.z_q_from_psi_sm(layout='legacy')`` vs
``layout='face'`` on the SAME synthetic ψ data — the r-chunk half of the
zeta-fit face-psi contract (``feat/zeta-fit-rchunk-face-psi-2026-08-22``).
The template this file follows is ``tests/test_isdf_cq_face_parity.py``
(the CCT/STEP-2 half of the same contract); see
``docs/architecture/zeta_fit_face_psi_cct.md``'s r-chunk section for the
full derivation.

Unlike the CCT half, ``layout='face'`` here does NOT need
:func:`distrib_la.gemm_plan` (no SUMMA GEMM — see ``isdf.core._z_q_face``'s
own docstring for why): its ONLY collectives are ``jax.lax.all_to_all``,
``jax.lax.all_gather`` and ``jax.lax.psum``, all of which run correctly on
an EMULATED multi-device CPU mesh, unlike cuBLASMp's real-multi-process
GUARD 4.  So this file's default check runs in a CPU subprocess
(``--xla_force_host_platform_device_count``, no GPU, no ``lx run``needed —
mirrors ``tests/test_zeta_mesh_invariance.py``'s own worker convention),
and additionally exposes a ``__main__`` CLI for a real-CUDA confirmation
run, matching ``test_isdf_cq_face_parity.py``'s own shape:

    lx run -N 1 -G 4 -n 4 bash <wrapper.sh> \\
        tests/test_isdf_zq_face_parity.py --mesh 2x2

Cases (mirroring the CCT parity test's coverage, task-specified: "including
non-mesh-divisible band edges and ns=2"):

* ``ns1_asym`` — ns=1, L/R band windows whose edges are NOT multiples of
  the mesh (px·py=4) or of psi_mun's own 'y'-shard width, and whose L
  window (``[0, 21)``) straddles that shard boundary too — exactly the
  case ``_z_q_face``'s per-position masked-``psum`` gather exists to
  handle without a resident single-axis copy or a breakpoint-aligned band
  chunking.
* ``ns2_spinor`` — ns=2 (the GEMM-seam / einsum-order concern for a
  bispinor-shaped input; charge channel only, gamma_L=gamma_R=None).
* ``ns1_lower_asym`` — the L window's LOWER edge also differs from
  ``band_range_full``'s own origin (``[5, 30)``), the case a
  psi_mun-offset bug (``bc.lo - _bfs`` vs some other origin) would show up
  in first.

Each case ALSO forces a band chunk (``(0, 24)``, width 24) to straddle
psi_mun's own 'y'-shard boundary (shard width ``nb_full/p_y = 36/2 = 18``)
— the multi-shard-span case ``_z_q_face``'s masked-gather-then-``psum``
design handles generally (no alignment requirement), unlike the
breakpoint-insertion approach considered and rejected in the design note.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_TOL = 1.0e-9


def _crand(rng, *shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)).astype("complex128")


#: (ns, l_range, r_range, seed) -- nb_full=36, n_rmu=4, kgrid=(2,2,1),
#: band_chunk_ranges=((0,24),(24,36)) (widths 24/12, both p_xy=4-divisible;
#: psi_mun's own p_y=2-shard width is 18, so bc0=[0,24) straddles it) held
#: fixed across all three cases -- see module docstring.
_CASES = (
    ("ns1_asym",       dict(ns=1, l_range=(0, 21), r_range=(9, 36), seed=1)),
    ("ns2_spinor",     dict(ns=2, l_range=(0, 21), r_range=(9, 36), seed=2)),
    ("ns1_lower_asym", dict(ns=1, l_range=(5, 30), r_range=(0, 36), seed=3)),
)


def _worker(case_name: str) -> int:
    """Runs in a fresh subprocess (JAX_PLATFORMS=cpu,
    --xla_force_host_platform_device_count=4 set by the caller) so the CPU
    client's device count is fixed before jax ever imports.  Prints one
    JSON line: {"max_rel": ...} or {"skip": "..."}."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    import isdf.core as _core
    from isdf.core import z_q_from_psi_sm
    if os.environ.get("LORRAX_ZQ_PARITY_DEBUG"):
        print(f"[isdf.core] {_core.__file__}", file=sys.stderr)
        print(f"[has layout=]{'layout' in z_q_from_psi_sm.__code__.co_varnames}",
              file=sys.stderr)

    ns, l_range, r_range, seed = (
        _CASES_BY_NAME[case_name]["ns"], _CASES_BY_NAME[case_name]["l_range"],
        _CASES_BY_NAME[case_name]["r_range"], _CASES_BY_NAME[case_name]["seed"])

    devs = jax.devices()
    PX = PY = 2
    if len(devs) < PX * PY:
        print(json.dumps({"skip": f"only {len(devs)} devices (<{PX*PY})"}))
        return 0
    mesh = Mesh(np.asarray(devs[: PX * PY]).reshape(PX, PY), ("x", "y"))

    rng = np.random.default_rng(seed)
    kgrid = (2, 2, 1); nk = 4
    fft_grid = (4, 4, 4); n_rtot = 4 * 4 * 4
    ngkmax = n_rtot // 2 + 1
    n_rmu = 4                       # divides p_x=2
    n_zchunk = 12                   # divides p_y=2
    nb_full = 36                    # divides p_y=2 (psi_mun's own shard axis)
    band_chunk_ranges = ((0, 24), (24, 36))   # widths 24, 12: both p_xy=4-divisible
    l0, l1 = l_range
    r0, r1 = r_range
    if (min(l0, r0), max(l1, r1)) != (0, nb_full):
        raise ValueError("test setup: full range must equal (0, nb_full)")

    # ---- shared Y-side synthetic data (psi_G_store mock) --------------
    # Multi-device mock: shards bands over P=px*py devices in row-major
    # ('x','y') order, each rank's tile padded to bpd_max -- the exact
    # layout the real store + band all_to_all/all_gather assume.
    # Mirrors tests/test_zeta_mesh_invariance.py's own ``_MeshStore``.
    g_index = np.full((nk, *fft_grid), ngkmax, dtype=np.int32)
    for k in range(nk):
        cells = sorted(rng.choice(n_rtot, ngkmax, replace=False))
        for idx, cell in enumerate(cells):
            g_index[k, cell // 16, (cell // 4) % 4, cell % 4] = idx
    psi_G = _crand(rng, nk, nb_full, ns, ngkmax)
    kvecs = rng.uniform(-0.5, 0.5, size=(nk, 3)).astype(np.float64)

    class _MeshStore:
        def __init__(self, mesh):
            P_tot = mesh.size
            self._py = mesh.shape['y']
            self.band_chunk_ranges = band_chunk_ranges
            offs = [0]
            for lo, hi in band_chunk_ranges:
                offs.append(offs[-1] + (hi - lo))
            self._offs = offs
            self._bpd_per_bc = tuple(
                (hi - lo) // P_tot for lo, hi in band_chunk_ranges)
            self._bpd_max = max(self._bpd_per_bc)
            self._per_rank_shape = (nk, nb_full, ns, ngkmax)

            class _M:
                pass
            self.meta = _M()
            self.meta.fft_grid = fft_grid
            self.meta.nk_tot = nk
            self.meta.nspinor = ns
            self._g = jax.device_put(
                jnp.asarray(g_index),
                NamedSharding(mesh, P(None, None, None, None)))
            self._k = jax.device_put(
                jnp.asarray(kvecs), NamedSharding(mesh, P(None, None)))

        def _slice_local_tile_bc(self, x_idx, y_idx, bc_idx):
            r = int(x_idx) * self._py + int(y_idx)
            bc = int(bc_idx)
            b_lo = self._offs[bc]
            bpd = self._bpd_per_bc[bc]
            out = np.zeros((nk, self._bpd_max, ns, ngkmax), dtype=np.complex128)
            out[:, :bpd, :, :] = psi_G[:, b_lo + r * bpd: b_lo + (r + 1) * bpd, :, :]
            return out

        @property
        def g_index(self):
            return self._g

        @property
        def kvecs_frac(self):
            return self._k

    store = _MeshStore(mesh)
    # Production always supplies a hoisted psi_r_cache (built ONCE before
    # the r-chunk loop, gw.isdf_fitting.fit_zeta_to_h5 STEP 5) -- exercise
    # the REAL cache path, not just the io_callback compatibility
    # fallback, using the REAL build_psi_r_cache_sm against this mock
    # store (it only reads the same four attributes _MeshStore already
    # implements).
    from isdf.core import build_psi_r_cache_sm
    psi_r_cache = jax.block_until_ready(
        build_psi_r_cache_sm(store, mesh_xy=mesh))

    # ---- X-side: ONE random source, spanning the FULL [0, nb_full)
    # window, playing the "conjugated psi*" role the legacy X-form
    # carries -- SAME convention the CCT parity test uses for its own
    # psi_rmuT_X_loaded.
    X_full = _crand(rng, nk, n_rmu, nb_full, ns)     # (k, mu, n, s), conjugated role
    psi_l_X_np = X_full[:, :, l0:l1, :]
    psi_r_X_np = X_full[:, :, r0:r1, :]
    # psi_mun: un-conjugated ψ, (k, s, mu, n) -- EXACTLY
    # gw.wavefunction_bundle.build_wavefunctions_face's own formula
    # (conj(psi_rmuT_X).transpose(0,3,1,2)), with X_full standing in for
    # the (unsliced) psi_rmuT_X.
    psi_mun_np = np.conj(X_full).transpose(0, 3, 1, 2)

    rep = NamedSharding(mesh, P())
    x1_4 = NamedSharding(mesh, P(None, "x", None, None))
    mun_spec = NamedSharding(mesh, P(None, None, "x", "y"))

    # ---- legacy ---------------------------------------------------------
    Z_legacy = jax.block_until_ready(z_q_from_psi_sm(
        jax.device_put(jnp.asarray(psi_l_X_np), x1_4),
        jax.device_put(jnp.asarray(psi_r_X_np), x1_4),
        store, psi_r_cache,
        band_chunk_ranges=band_chunk_ranges,
        band_range_left=l_range, band_range_right=r_range,
        r_start_dyn=0, r_chunk_size=n_zchunk,
        gamma_L=None, gamma_R=None, kgrid=kgrid, mesh_xy=mesh,
        layout="legacy"))

    # ---- face -------------------------------------------------------
    idx = np.arange(nb_full)
    w_l = np.where((idx >= l0) & (idx < l1), 1.0, 0.0)
    w_r = np.where((idx >= r0) & (idx < r1), 1.0, 0.0)
    Z_face = jax.block_until_ready(z_q_from_psi_sm(
        psi_G_store=store, psi_r_cache=psi_r_cache,
        band_chunk_ranges=band_chunk_ranges,
        r_start_dyn=0, r_chunk_size=n_zchunk,
        kgrid=kgrid, mesh_xy=mesh,
        layout="face",
        psi_mun=jax.device_put(jnp.asarray(psi_mun_np), mun_spec),
        weight_l=jnp.asarray(w_l), weight_r=jnp.asarray(w_r)))

    # process_allgather, NOT device_get: both are genuinely multi-process
    # sharded (P(None,'x','y')) under a REAL multi-process mesh (the
    # `lx run -N 1 -G 4 -n 4` CLI path -- 4 processes, one device each),
    # where device_get on a process-non-addressable array raises.  Under
    # the CPU-emulated single-process worker path this is a harmless
    # equivalent read.  Mirrors test_isdf_cq_face_parity.py's own fix for
    # the identical issue.
    from jax.experimental import multihost_utils as _mhu
    Zl = np.asarray(_mhu.process_allgather(Z_legacy, tiled=True))
    Zf = np.asarray(_mhu.process_allgather(Z_face, tiled=True))
    if os.environ.get("LORRAX_ZQ_PARITY_DEBUG"):
        print(f"[shapes] Zl={Zl.shape} Zf={Zf.shape} "
              f"nan_l={int(np.isnan(Zl).sum())} nan_f={int(np.isnan(Zf).sum())} "
              f"finite_l={int(np.isfinite(Zl).sum())} "
              f"finite_f={int(np.isfinite(Zf).sum())}",
              file=sys.stderr)
    if Zl.shape != Zf.shape:
        print(json.dumps({
            "error": f"shape mismatch: legacy={Zl.shape} face={Zf.shape}"}))
        return 1
    absdiff = np.abs(Zl - Zf)
    ref_scale = float(np.abs(Zl).max())
    max_abs = float(absdiff.max())
    max_rel = max_abs / max(ref_scale, 1e-300)
    print(json.dumps({
        "max_abs": max_abs, "ref_scale": ref_scale, "max_rel": max_rel,
        "case": case_name, "ns": ns, "l_range": list(l_range),
        "r_range": list(r_range),
    }))
    return 0


_CASES_BY_NAME = {name: kwargs for name, kwargs in _CASES}


def _run_worker(case_name: str, ndev: int = 4, timeout: int = 300):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={ndev}").strip()
    # Pin PYTHONPATH to THIS checkout's src/ -- prepended (not appended) so
    # it wins over anything a wrapping harness (lx test's venv, a stray
    # editable install) already put on the child's sys.path.  Mirrors
    # tests/harness.py's own ``run_gw_subprocess`` convention.
    _repo_src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env["PYTHONPATH"] = _repo_src + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    res = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "worker", case_name],
        env=env, capture_output=True, text=True, timeout=timeout)
    assert res.returncode == 0, (
        f"worker {case_name} failed rc={res.returncode}\nSTDOUT:\n"
        f"{res.stdout}\nSTDERR:\n{res.stderr}")
    lines = [ln for ln in res.stdout.splitlines() if ln.strip().startswith("{")]
    assert lines, f"no JSON from worker.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return json.loads(lines[-1])


@pytest.mark.parametrize("name,kwargs", _CASES, ids=[c[0] for c in _CASES])
def test_zq_face_layout_matches_legacy(name, kwargs):
    out = _run_worker(name)
    if "skip" in out:
        pytest.skip(f"z_q face-layout parity gate: {out['skip']}")
    assert "error" not in out, out.get("error")
    assert out["max_rel"] < _TOL, (
        f"z_q_from_psi_sm layout='face' vs 'legacy' parity FAILED: "
        f"max relative diff {out['max_rel']:.3e} (case {name})")


# ---------------------------------------------------------------------------
# Real-CUDA CLI (matches test_isdf_cq_face_parity.py's shape; not required
# for the check above, since this path needs no gemm_plan/real-process
# CUDA, but the task asks for a real-hardware confirmation too).
# ---------------------------------------------------------------------------

def _cli_main():
    import argparse
    import numpy as np
    import jax

    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="2x2", help="PxQ process mesh")
    args = ap.parse_args()
    px, py = (int(v) for v in args.mesh.lower().split("x"))
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    p0(f"backend={jax.default_backend()} mesh={args.mesh} "
       f"processes={jax.process_count()} devices={jax.device_count()}")
    if jax.device_count() != px * py:
        p0(f"REFUSE: need exactly {px * py} devices for a {args.mesh} mesh; "
           f"got {jax.device_count()}")
        return 1
    if (px, py) != (2, 2):
        p0("REFUSE: this file's synthetic shapes are fixed for a 2x2 mesh "
           "(n_rmu=4, nb_full=36, band_chunk widths 24/12) -- pass "
           "--mesh 2x2.")
        return 1

    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "")
    failures = 0
    for name, _kwargs in _CASES:
        try:
            rc = _worker_inline(name)
        except Exception as exc:  # noqa: BLE001 -- report, don't crash the sweep
            failures += 1
            p0(f"FAIL {name}: {exc}")
            continue
        if rc.get("skip"):
            p0(f"SKIP {name}: {rc['skip']}")
            continue
        if "error" in rc:
            failures += 1
            p0(f"FAIL {name}: {rc['error']}")
            continue
        ok = rc["max_rel"] < _TOL
        if not ok:
            failures += 1
        p0(f"{'PASS' if ok else 'FAIL'} {name}: max|diff|={rc['max_abs']:.3e} "
           f"(ref scale {rc['ref_scale']:.3e}) max|rel diff|={rc['max_rel']:.3e}")
    p0(f"done: {len(_CASES) - failures}/{len(_CASES)} cases passed")
    return 1 if failures else 0


def _worker_inline(case_name: str) -> dict:
    """Runs ``_worker``'s BODY in-process (this IS the CUDA process
    already), capturing its one JSON print instead of spawning a
    subprocess."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _worker(case_name)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"worker {case_name} printed no JSON (rc={rc})")
    return json.loads(lines[-1])


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "worker":
        # CPU-emulated subprocess path (spawned by _run_worker): a SINGLE
        # process pretending to have --xla_force_host_platform_device_count
        # devices needs no multi-process bootstrap at all.
        sys.exit(_worker(sys.argv[2]))
    # Real-CUDA CLI path (`lx run -N 1 -G 4 -n 4 ... --mesh 2x2`): mirrors
    # tests/test_isdf_cq_face_parity.py's own bootstrap EXACTLY -- init
    # BEFORE importing jax/distrib_la, guarded by __main__, so plain pytest
    # collection of this file never pays or triggers this.  Missing this
    # was the difference between "processes=1, devices=1" (each srun task
    # standalone) and a genuine multi-process mesh -- measured directly:
    # the first real-CUDA attempt without it printed exactly that and
    # refused every case (see this branch's own commit history).
    _TESTS = os.path.dirname(os.path.abspath(__file__))
    _REPO = os.path.dirname(_TESTS)
    for _svc in ("lxkit", "distrib_la"):
        _src = os.path.join(_REPO, "services", _svc, "src")
        if os.path.isdir(_src) and _src not in sys.path:
            sys.path.insert(0, _src)
    from lxkit.gate import platform_from_env
    from runtime import initialize_communicator_stack
    _plat = platform_from_env()
    _RUNTIME = initialize_communicator_stack(
        platform="gpu" if _plat == "CUDA" else "cpu")
    sys.exit(_cli_main())
