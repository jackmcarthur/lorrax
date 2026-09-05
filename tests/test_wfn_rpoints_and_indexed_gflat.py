"""Arbitrary-flat-r-point twins of the contiguous slab helpers.

A parent-k ζ-fit kernel processes real-space *tiles* whose grid points are
a symmetry orbit, so they are scattered through the flat r index rather
than contiguous in it.  Three helpers grew arbitrary-point variants for
that consumer, and this file is their gate:

* :func:`common.wfn_transforms.to_rpoints_inner` — the twin of
  ``to_rchunk_inner``.  On a contiguous index list it must agree with the
  slab version BIT FOR BIT (it is the same math with a gather in place of
  a ``dynamic_slice``); on a permuted list it must agree with a numpy
  gather of the full-box reference.
* :func:`common.wfn_transforms.accumulate_rchunk_to_gflat` with
  ``r_indices=`` — the scatter twin of its ``r0=`` path, including the
  per-q Bloch phase, which must be looked up at the tile's own cells and
  not at ``r0 + j``.
* :class:`common.psi_G_store.PsiGStore` with ``k_domain='ibz'`` — the
  store over the WFN file's own parent k rows.

Everything runs in a CPU subprocess with four emulated devices
(``--xla_force_host_platform_device_count=4``), which is the only way to
fix the device count before jax imports; the pattern is
``tests/test_isdf_zq_face_parity.py``'s ``_worker`` / ``_run_worker``.
The mesh cases need a real 2x2 ('x', 'y') mesh because
``accumulate_rchunk_to_gflat`` shards μ over both axes and ``PsiGStore``
shards bands over both.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

#: Scatter order can differ between the r0 and r_indices paths, so the
#: permuted comparisons are allclose rather than exact.
_TOL = 1.0e-13


def _case_rpoints() -> dict:
    """``to_rpoints_inner`` vs ``to_rchunk_inner`` on the same synthetic ψ."""
    import numpy as np
    import jax.numpy as jnp

    from common.wfn_transforms import to_rchunk_inner, to_rpoints_inner

    rng = np.random.default_rng(20260905)
    fft_grid = (4, 4, 4)
    n_rtot = 64
    nk, nb, ns, ngkmax = 2, 3, 1, 9
    r0, r_len = 16, 8

    psi = jnp.asarray(
        rng.standard_normal((nk, nb, ns, ngkmax))
        + 1j * rng.standard_normal((nk, nb, ns, ngkmax)),
        dtype=jnp.complex128)
    # Sentinel ``ngkmax`` in most cells: only a handful of box cells hold a
    # G coefficient, exactly as a real G-sphere does.
    g_index = np.full((nk,) + fft_grid, ngkmax, dtype=np.int32)
    for k in range(nk):
        cells = rng.choice(n_rtot, size=ngkmax, replace=False)
        flat = g_index[k].reshape(n_rtot)
        flat[cells] = np.arange(ngkmax, dtype=np.int32)
        g_index[k] = flat.reshape(fft_grid)
    g_index_j = jnp.asarray(g_index)
    # Non-trivial k vectors: a zero phase would hide a wrong phase lookup.
    kvecs = jnp.asarray([[0.25, -0.5, 0.125], [-0.375, 0.25, 0.5]],
                        dtype=jnp.float64)

    slab = np.asarray(to_rchunk_inner(
        psi, g_index_j, fft_grid, r0, r_len, kvecs_frac=kvecs))
    contiguous = np.asarray(to_rpoints_inner(
        psi, g_index_j, fft_grid,
        jnp.arange(r0, r0 + r_len, dtype=jnp.int32), kvecs_frac=kvecs))

    # Full-box reference, gathered in numpy — the independent check that
    # an arbitrary order picks the cells it names.
    full = np.asarray(to_rchunk_inner(
        psi, g_index_j, fft_grid, 0, n_rtot, kvecs_frac=kvecs))
    perm = rng.permutation(n_rtot)[:r_len].astype(np.int32)
    permuted = np.asarray(to_rpoints_inner(
        psi, g_index_j, fft_grid, jnp.asarray(perm), kvecs_frac=kvecs))

    # Out-of-range cells are pad slots: the gathers stay in bounds (so the
    # values are finite, not NaN), and the CALLER's mask zeroes them.
    oob = np.array([r0, n_rtot, r0 + 1, n_rtot + 6, -3, r0 + 2],
                   dtype=np.int32)
    padded = np.asarray(to_rpoints_inner(
        psi, g_index_j, fft_grid, jnp.asarray(oob), kvecs_frac=kvecs))
    keep = ((oob >= 0) & (oob < n_rtot))
    masked = padded * keep.astype(np.complex128)

    return {
        "contiguous_bit_identical": bool(np.array_equal(contiguous, slab)),
        "permuted_max_abs": float(
            np.max(np.abs(permuted - full[..., perm]))),
        "padded_all_finite": bool(np.all(np.isfinite(padded))),
        "masked_pad_all_zero": bool(np.all(masked[..., ~keep] == 0)),
        "masked_kept_matches": bool(np.array_equal(
            masked[..., keep], full[..., oob[keep]])),
    }


def _case_gflat() -> dict:
    """``accumulate_rchunk_to_gflat(r_indices=)`` vs its ``r0=`` path."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from common.wfn_transforms import accumulate_rchunk_to_gflat

    devs = jax.devices()
    if len(devs) < 4:
        return {"skip": f"only {len(devs)} devices (<4)"}
    mesh = Mesh(np.asarray(devs[:4]).reshape(2, 2), ("x", "y"))

    rng = np.random.default_rng(20260906)
    fft_grid = (4, 4, 4)
    n_rtot = 64
    n_q, n_rmu, ngkmax = 2, 4, 5           # n_rmu divisible by mesh.size = 4
    r0, r_len = 16, 8

    sphere_idx = np.stack([
        rng.choice(n_rtot, size=ngkmax, replace=False).astype(np.int32)
        for _ in range(n_q)])
    # Non-trivial q so the pre-FFT Bloch phase is actually exercised.
    qvec = np.array([[0.0, 0.0, 0.0], [0.5, -0.25, 0.125]], dtype=np.float64)

    rch = (rng.standard_normal((n_q, n_rmu, r_len))
           + 1j * rng.standard_normal((n_q, n_rmu, r_len)))
    mu_spec = NamedSharding(mesh, P(None, ("x", "y"), None))

    def _run(slab_np, *, r0=None, r_indices=None):
        slab = jax.device_put(
            jnp.asarray(slab_np, dtype=jnp.complex128), mu_spec)
        acc = jax.device_put(
            jnp.zeros((n_q, n_rmu, ngkmax), dtype=jnp.complex128), mu_spec)
        out = accumulate_rchunk_to_gflat(
            slab, acc, mesh=mesh, fft_grid=fft_grid, r0=r0,
            sphere_idx=sphere_idx, qvec_frac=qvec, norm="backward",
            r_indices=(None if r_indices is None
                       else jnp.asarray(r_indices, dtype=jnp.int32)))
        return np.asarray(out)

    base = _run(rch, r0=r0)
    contiguous = _run(rch, r_indices=np.arange(r0, r0 + r_len, dtype=np.int32))

    # A permuted index list with the correspondingly permuted slab columns
    # writes the same value into the same box cell, so it must reproduce
    # the contiguous answer.
    sigma = rng.permutation(r_len)
    permuted = _run(rch[..., sigma],
                    r_indices=(r0 + sigma).astype(np.int32))

    # Pad slots: DISTINCT out-of-range sentinels (the kernel promises unique
    # indices) with exactly-zero slab columns.
    idx_pad = np.concatenate(
        [np.arange(r0, r0 + r_len, dtype=np.int32),
         n_rtot + np.arange(3, dtype=np.int32)])
    rch_pad = np.concatenate(
        [rch, np.zeros((n_q, n_rmu, 3), dtype=rch.dtype)], axis=-1)
    padded = _run(rch_pad, r_indices=idx_pad)

    both = neither = ""
    try:
        _run(rch, r0=r0, r_indices=np.arange(r0, r0 + r_len, dtype=np.int32))
    except ValueError as exc:
        both = str(exc)
    try:
        _run(rch)
    except ValueError as exc:
        neither = str(exc)

    return {
        "contiguous_bit_identical": bool(np.array_equal(contiguous, base)),
        "base_nonzero": float(np.max(np.abs(base))),
        "permuted_max_abs": float(np.max(np.abs(permuted - base))),
        "padded_max_abs": float(np.max(np.abs(padded - base))),
        "both_error": both,
        "neither_error": neither,
    }


class _FakeLoader:
    """The narrow slice of ``WfnLoader`` that ``PsiGStore`` population uses.

    Coefficients are a deterministic function of (k, band, spinor, G) so a
    row from the ibz domain is distinguishable from a full-BZ row.
    """

    def __init__(self, *, mesh, nk_ibz, nk_full, nbands, nspinor, ngkmax):
        import numpy as np
        self._mesh = mesh
        self._np = np
        self.nkpts = int(nk_ibz)
        self.nk_full = int(nk_full)
        self.nbands = int(nbands)
        self.nspinor = int(nspinor)
        self.ngkmax = int(ngkmax)
        self.load_k = []

    def _nk(self, k):
        return self.nkpts if k == "ibz" else self.nk_full

    def load(self, *, bands, k, sharding, bispinor=False,
             bispinor_lift="raw"):
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding
        self.load_k.append(str(k))
        np = self._np
        b_lo, b_hi = int(bands[0]), int(bands[1])
        nk = self._nk(k)
        kk = np.arange(nk)[:, None, None, None]
        bb = np.arange(b_lo, b_hi)[None, :, None, None]
        ss = np.arange(self.nspinor)[None, None, :, None]
        gg = np.arange(self.ngkmax)[None, None, None, :]
        arr = (100 * kk + 10 * bb + ss + 0.01 * gg).astype(np.complex128)
        return jax.device_put(jnp.asarray(arr),
                              NamedSharding(self._mesh, sharding))

    def box_index_dev(self, *, k, mesh):
        import jax.numpy as jnp
        return jnp.zeros((self._nk(k), 2, 2, 2), dtype=jnp.int32)

    def kvecs(self, *, k):
        return self._np.zeros((self._nk(k), 3), dtype=self._np.float64)


def _case_store_ibz() -> dict:
    """``PsiGStore(k_domain='ibz')`` holds parent rows, not unfolded ones."""
    import numpy as np
    import jax
    from types import SimpleNamespace
    from jax.sharding import Mesh

    from common.psi_G_store import build_psi_G_store

    devs = jax.devices()
    if len(devs) < 4:
        return {"skip": f"only {len(devs)} devices (<4)"}
    mesh = Mesh(np.asarray(devs[:4]).reshape(2, 2), ("x", "y"))

    nk_ibz, nk_full, nbands = 3, 8, 4
    meta = SimpleNamespace(nk_tot=nk_full, nspinor=1, fft_grid=(2, 2, 2),
                           n_rtot=8, b_id_4_user=nbands, b_id_4=nbands)
    out = {}
    for domain in ("full_bz", "ibz"):
        loader = _FakeLoader(mesh=mesh, nk_ibz=nk_ibz, nk_full=nk_full,
                             nbands=nbands, nspinor=1, ngkmax=5)
        with build_psi_G_store(
                wfn=loader, mesh_xy=mesh, meta=meta,
                band_chunk_ranges=((0, nbands),),
                k_domain=domain) as store:
            out[domain] = {
                "k_domain": store.k_domain,
                "tile_nk": int(store._per_rank_shape[0]),
                "g_index_nk": int(store.g_index.shape[0]),
                "kvecs_nk": int(store.kvecs_frac.shape[0]),
                "load_k": list(loader.load_k),
            }

    bad = ""
    try:
        build_psi_G_store(
            wfn=_FakeLoader(mesh=mesh, nk_ibz=nk_ibz, nk_full=nk_full,
                            nbands=nbands, nspinor=1, ngkmax=5),
            mesh_xy=mesh, meta=meta, band_chunk_ranges=((0, nbands),),
            k_domain="half_bz")
    except ValueError as exc:
        bad = str(exc)
    out["bad_error"] = bad
    return out


_WORKERS = {
    "rpoints": _case_rpoints,
    "gflat": _case_gflat,
    "store_ibz": _case_store_ibz,
}


def _worker(case_name: str) -> int:
    """Runs in a fresh subprocess (JAX_PLATFORMS=cpu,
    --xla_force_host_platform_device_count=4 set by the caller) so the CPU
    client's device count is fixed before jax ever imports.  Prints one
    JSON line."""
    print(json.dumps(_WORKERS[case_name]()))
    return 0


def _run_worker(case_name: str, ndev: int = 4, timeout: int = 300):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    # The Perlmutter runtime exports the CUDA-only pair-convolution
    # accelerator as ``on``; inheriting that dial makes the CPU child refuse
    # before it reaches the comparison.
    env["LORRAX_CONV_KPAIR_FFI"] = "off"
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={ndev}").strip()
    # Pin PYTHONPATH to THIS checkout's src/ -- prepended so it wins over
    # anything a wrapping harness already put on the child's sys.path.
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


def test_to_rpoints_inner_matches_the_contiguous_slab():
    out = _run_worker("rpoints")
    assert out["contiguous_bit_identical"], (
        "to_rpoints_inner on r0 + arange must reproduce to_rchunk_inner "
        "bit for bit")
    assert out["permuted_max_abs"] < _TOL, (
        "a permuted index list must gather the cells it names: max abs diff "
        f"{out['permuted_max_abs']:.3e}")
    assert out["padded_all_finite"], (
        "out-of-range indices must clip into bounds, not produce NaN/inf")
    assert out["masked_pad_all_zero"]
    assert out["masked_kept_matches"], (
        "the in-range cells of a padded index list must be untouched by "
        "the presence of pad slots")


def test_indexed_gflat_accumulate_matches_the_r0_path():
    out = _run_worker("gflat")
    if "skip" in out:
        pytest.skip(f"indexed accumulate gate: {out['skip']}")
    assert out["base_nonzero"] > 0.0, "test setup produced an all-zero result"
    assert out["contiguous_bit_identical"], (
        "r_indices = r0 + arange must reproduce the r0 path exactly")
    assert out["permuted_max_abs"] < _TOL, (
        "a permuted tile with permuted columns must reproduce the r0 path: "
        f"max abs diff {out['permuted_max_abs']:.3e}")
    assert out["padded_max_abs"] < _TOL, (
        "sentinel pad slots with zero slab columns must change nothing: "
        f"max abs diff {out['padded_max_abs']:.3e}")
    assert "exactly one of r0 / r_indices" in out["both_error"]
    assert "exactly one of r0 / r_indices" in out["neither_error"]


def test_psi_g_store_ibz_domain_holds_parent_rows():
    out = _run_worker("store_ibz")
    if "skip" in out:
        pytest.skip(f"PsiGStore k_domain gate: {out['skip']}")
    full, ibz = out["full_bz"], out["ibz"]
    assert full["k_domain"] == "full_bz" and ibz["k_domain"] == "ibz"
    assert full["load_k"] == ["full_bz"] and ibz["load_k"] == ["ibz"]
    for field in ("tile_nk", "g_index_nk", "kvecs_nk"):
        assert full[field] == 8, f"full_bz {field} = {full[field]}, want 8"
        assert ibz[field] == 3, f"ibz {field} = {ibz[field]}, want 3"
    assert "k_domain must be 'full_bz' or 'ibz'" in out["bad_error"]


if __name__ == "__main__":
    sys.exit(_worker(sys.argv[2]))
