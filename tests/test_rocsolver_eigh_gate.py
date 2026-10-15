"""The ROCM batched local eigh (``lorrax_rocsolver_batched_eigh``) against ``jnp.linalg.eigh``
on the same device.

``distrib_la.local_eigh.BATCHED_EIGH_ROUTE`` sends a stack of complex128 / float64 Hermitian
matrices to one rocSOLVER strided-batched heevd/syevd call (``src/ffi/cpp/rocsolver``) instead
of jaxlib's per-matrix loop.  Gate (``ffi.common.families`` row ``rocsolver``, tol 1e-12), per
case, on every device of the mesh (a batch-sharded stack, each device solving its local block
inside ``shard_map``, the ``batch_reshard`` geometry of line_select and the constructor):

* residual ``max|A V - V diag(w)| / max|A| <= TOL`` and orthogonality ``max|V^H V - I| <= TOL``;
* eigenvalues ``max|w - w_ref| / max|A| <= TOL`` against the XLA route on the same device,
  ``jnp.linalg.eigh``, for float64 and generic complex stacks.  On structured / degenerate complex
  stacks the device XLA route (the reflector over jaxlib's zheevd loop) is itself defective: it
  returns NaN vectors with info = 0 for some matrices (eigh_probe4), so the reference there is
  host LAPACK (``np.linalg.eigh``), both routes' failed matrices are counted and printed, and
  the new route passes only if it fails no more matrices than the old one and every failed
  matrix is NaN in both w and v (the route's explicit failure);
* eigenspaces: per eigenvalue cluster (gap > 1e-8 max|A|) the projector difference
  ``max|V_c V_c^H - R_c R_c^H| <= PROJ_TOL`` -- vectors are compared as subspaces, never
  entrywise (degenerate spectra and the gauge are free).

Cases: random, exactly degenerate (multiplicity 3 and 8 clusters, ``Q diag(l) Q^H``),
real-valued complex, diagonal with repeats, zero rows, block-diagonal, an all-zero and a
scalar slot; UPLO L/U without symmetrization; the Fe line_select shape c128[16, 312, 312].
Plus the row: a stack outside it (batch 1) is ``jnp.linalg.eigh`` bit for bit.

ROCm: ``python -m tests.test_rocsolver_eigh_gate`` on any mesh (one GCD or the 2x2 P=4 mesh).
Under pytest the device gate skips; the row's CPU checks run.
"""
import sys

import numpy as np
import pytest

TOL = 1e-12
PROJ_TOL = 1e-9


def _herm(rng, b, n, dtype):
    x = rng.standard_normal((b, n, n))
    if np.dtype(dtype).kind == "c":
        x = x + 1j * rng.standard_normal((b, n, n))
    return ((x + np.conj(np.swapaxes(x, -1, -2))) / 2).astype(dtype)


def _degenerate(rng, b, n, dtype, mult):
    out = []
    for _ in range(b):
        q, _r = np.linalg.qr(_herm(rng, 1, n, dtype)[0] + 3 * np.eye(n))
        lam = np.repeat(rng.standard_normal(-(-n // mult)), mult)[:n]
        out.append((q * lam) @ np.conj(q.T))
    a = np.stack(out)
    return ((a + np.conj(np.swapaxes(a, -1, -2))) / 2).astype(dtype)


def _case(kind, rng, b, n, dtype):
    if kind.startswith("degenerate"):
        return _degenerate(rng, b, n, dtype, int(kind[10:]))
    a = _herm(rng, b, n, dtype)
    if kind == "real_valued": a = a.real.astype(dtype)
    if kind == "diagonal":    a = np.stack([np.diag(np.round(np.diag(m), 1)) for m in a]).astype(dtype)
    if kind == "zero_row":    a[:, -3:, :] = 0; a[:, :, -3:] = 0
    if kind == "block":       h = n // 2; a[:, :h, h:] = 0; a[:, h:, :h] = 0
    if kind == "slots":       a[0] = 0; a[-1] = 2.5 * np.eye(n)
    return a


def compare(a, w, v, w_ref, v_ref):
    """The gate's numbers for one local stack (numpy arrays)."""
    scale = max(1.0, float(np.abs(a).max()))
    n = a.shape[-1]
    vh = np.conj(np.swapaxes(v, -1, -2))
    res = float(np.abs(a @ v - v * w[..., None, :]).max()) / scale
    orth = float(np.abs(vh @ v - np.eye(n)).max())
    dw = float(np.abs(w - w_ref).max()) / scale
    proj = 0.0
    for i in range(a.shape[0]):
        cuts = np.flatnonzero(np.diff(w_ref[i]) > 1e-8 * scale) + 1
        for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, n]):
            p = v[i][:, lo:hi] @ np.conj(v[i][:, lo:hi].T)
            r = v_ref[i][:, lo:hi] @ np.conj(v_ref[i][:, lo:hi].T)
            proj = max(proj, float(np.abs(p - r).max()))
    finite = bool(np.all(np.isfinite(w)) and np.all(np.isfinite(v)))
    return dict(res=res, orth=orth, dw=dw, proj=proj, finite=finite)


def _full_np(a, uplo, sym):
    """numpy twin of distrib_la.local_eigh._full: the full Hermitian matrix eigh solves."""
    ah = lambda t: np.conj(np.swapaxes(t, -1, -2))
    if sym:
        return (a + ah(a)) / 2
    if uplo == "U":
        return np.triu(a) + ah(np.triu(a, 1))
    return np.tril(a) + ah(np.tril(a, -1))


def passes(m):
    return (m["finite"] and m["res"] <= TOL and m["orth"] <= TOL and m["dw"] <= TOL
            and m["proj"] <= PROJ_TOL)


CASES = (
    # (kind, b per device, n, dtype, uplo, symmetrize_input)
    ("random", 16, 312, "complex128", None, True),          # the Fe line_select stack
    ("random", 8, 64, "complex128", None, True),
    ("random", 4, 600, "complex128", "U", False),
    ("degenerate3", 4, 120, "complex128", None, True),
    ("degenerate8", 4, 200, "complex128", "L", False),
    ("degenerate16", 16, 312, "complex128", None, True),   # eigh_probe4's failing class
    ("real_valued", 4, 312, "complex128", None, True),
    ("diagonal", 4, 312, "complex128", None, True),
    ("zero_row", 4, 128, "complex128", None, True),
    ("block", 4, 312, "complex128", None, True),
    ("slots", 4, 312, "complex128", None, True),
    ("random", 16, 312, "float64", None, True),
    ("random", 4, 900, "float64", "L", False),
    ("degenerate3", 4, 150, "float64", None, True),
    ("diagonal", 4, 128, "float64", None, True),
    ("zero_row", 4, 128, "float64", "U", False),
    ("block", 4, 312, "float64", None, True),
    ("slots", 4, 200, "float64", None, True),
)
STRUCTURED = ("real_valued", "diagonal", "zero_row", "block", "slots", "degenerate3", "degenerate8",
              "degenerate16")


def gate_cases(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import device_put_process_local
    from distrib_la import local_eigh as le

    names = tuple(mesh.axis_names)
    ndev = int(np.prod(mesh.devices.shape))
    spec = P(names)                                   # batch over every mesh axis
    rng = np.random.default_rng(20261015)            # the same global stacks on every process
    out = {}
    for kind, b, n, dt, uplo, sym in CASES:
        glob = _case(kind, rng, b * ndev, n, dt)        # b matrices per device
        x = device_put_process_local(glob, NamedSharding(mesh, spec))
        row = le.batched_eigh_row((b, n, n), np.dtype(dt))
        assert row is not None, f"stack {(b, n, n)} {dt} is outside the ROCM row {le.BATCHED_EIGH_ROUTE}"
        new = lambda t: le.eigh_local(t, UPLO=uplo, symmetrize_input=sym)
        host_ref = dt == "complex128" and kind in STRUCTURED
        if host_ref:        # the leg's previous route (reflector + jaxlib's zheevd loop) ...
            ref = lambda t: le._reflected_eigh(t, uplo, sym, solve=jnp.linalg.eigh)
        else:               # XLA itself
            ref = lambda t: jnp.linalg.eigh(t, UPLO=uplo, symmetrize_input=sym)
        sm = lambda f: jax.jit(jax.shard_map(lambda t: tuple(f(t)), mesh=mesh, in_specs=spec,
                                             out_specs=(spec, spec)))
        w, v = sm(new)(x)
        w0, v0 = sm(ref)(x)
        # The matrix both solve, in numpy (eigh_local's _full on the host copy), sliced like
        # this process's shards; each device's block is compared on its own.
        a_full = np.asarray(_full_np(glob, uplo, sym))
        shards = lambda t: sorted(t.addressable_shards, key=lambda s: s.index[0].start or 0)
        ms, ref_nan, new_nan, flagged = [], 0, 0, True
        for sw, sv, sw0, sv0 in zip(shards(w), shards(v), shards(w0), shards(v0)):
            a_loc, w_n, v_n = a_full[sw.index[0]], np.asarray(sw.data), np.asarray(sv.data)
            w_r, v_r = np.asarray(sw0.data), np.asarray(sv0.data)
            if host_ref:    # ... fails silently on these stacks (NaN vectors, info = 0; pool
                # log eigh_probe4), so it is no reference there: LAPACK on the host is.
                ref_nan += int(sum(not np.isfinite(x).all() for x in v_r))
                w_r, v_r = np.linalg.eigh(a_loc)
            fin = np.array([np.isfinite(v_n[i]).all() for i in range(len(v_n))])
            new_nan += int((~fin).sum())
            flagged &= bool(np.isnan(w_n[~fin]).all() and np.isnan(v_n[~fin]).all())
            if fin.any():
                ms.append(compare(a_loc[fin], w_n[fin], v_n[fin], w_r[fin], v_r[fin]))
        m = {k: (all(d[k] for d in ms) if k == "finite" else max(d[k] for d in ms)) for k in ms[0]}
        # A failed matrix must be NaN in w and v; the route may fail only where rocSOLVER's
        # known silent zheevd failure also fails the old route (never more matrices than it).
        m["finite"] = m["finite"] and flagged and (new_nan == 0 or (host_ref and new_nan <= ref_nan))
        m["ref_nan"], m["new_nan"] = ref_nan, new_nan
        out[f"{kind:12s} {dt:10s} b={b:3d} n={n:4d} uplo={uplo} sym={sym}"] = m
    # Outside the row (float64, one matrix): jnp.linalg.eigh unchanged.
    assert le.batched_eigh_row((64, 64), np.float64) is None
    a1 = jnp.asarray(_herm(rng, 1, 64, "float64")[0])
    w1, v1 = le.eigh_local(a1)
    w2, v2 = jnp.linalg.eigh(a1)
    out["batch-1 float64 is jnp.linalg.eigh bit for bit"] = dict(
        finite=True, res=0.0, orth=0.0, proj=0.0,
        dw=0.0 if (np.array_equal(np.asarray(w1), np.asarray(w2))
                   and np.array_equal(np.asarray(v1), np.asarray(v2))) else 1.0)
    return out


def main() -> int:
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    jax.config.update("jax_enable_x64", True)
    from ffi.gate import mesh_ffi_platform
    mesh = stack.mesh
    if mesh_ffi_platform(mesh) != "ROCM":
        print(f"SKIP: needs a ROCm mesh, got {mesh_ffi_platform(mesh)}", flush=True)
        runtime.finalize_process(0)
        return 0
    out = gate_cases(mesh)
    bad = {k: m for k, m in out.items() if not passes(m)}
    for k, m in out.items():
        print(f"r{jax.process_index()} {'ok  ' if not k in bad else 'FAIL'} {k}  res {m['res']:.1e} "
              f"orth {m['orth']:.1e} dw {m['dw']:.1e} proj {m['proj']:.1e}"
              + (f"  [failed matrices: old route {m['ref_nan']}, new route {m['new_nan']}]"
                 if m.get("ref_nan") or m.get("new_nan") else ""), flush=True)
    nbad = len(bad)
    print(f"r{jax.process_index()} ROCSOLVER_EIGH_GATE {'PASS' if not bad else 'FAIL'} on "
          f"{tuple(mesh.devices.shape)}: {len(out) - len(bad)}/{len(out)} (tol {TOL:g}, proj {PROJ_TOL:g})",
          flush=True)
    rc = 1 if nbad else 0
    runtime.finalize_process(rc)
    return rc


# ---- pytest: the row's logic (CPU) ------------------------------------------------------

def test_the_row_is_rocm_only_and_bounded():
    from distrib_la.local_eigh import BATCHED_EIGH_ROUTE, batched_eigh_row
    assert set(BATCHED_EIGH_ROUTE) == {"ROCM"}
    row = BATCHED_EIGH_ROUTE["ROCM"]
    assert row.target == "lorrax_rocsolver_batched_eigh" and set(row.windows) == {"complex128", "float64"}
    assert row.windows["complex128"].driver == "heevd" and row.windows["float64"].driver == "syevd"
    for dt, w in row.windows.items():
        assert batched_eigh_row((16, 312, 312), dt, "ROCM") is row        # the Fe line_select stack
        assert batched_eigh_row((w.min_batch, w.min_n, w.min_n), dt, "ROCM") is row
        assert batched_eigh_row((2, w.max_n + 1, w.max_n + 1), dt, "ROCM") is None
        if w.min_n > 1:
            assert batched_eigh_row((64, w.min_n - 1, w.min_n - 1), dt, "ROCM") is None
        if w.min_batch > 1:
            assert batched_eigh_row((w.min_n, w.min_n), dt, "ROCM") is None
        for leg in ("CUDA", "cpu"):
            assert batched_eigh_row((16, 312, 312), dt, leg) is None
    assert batched_eigh_row((16, 312, 312), np.complex64, "ROCM") is None


def test_the_target_is_in_both_rocm_tables_and_the_family_row():
    pytest.importorskip("jax")
    from ffi.common import ffi_loader, families
    from distrib_la import loader
    t = "lorrax_rocsolver_batched_eigh"
    assert loader._PLATFORMS["ROCM"]["targets"][t] == ffi_loader._PLATFORMS["ROCM"]["targets"][t]
    assert families.family_of(t) == "rocsolver" and families.state("rocsolver", "ROCM")[0] == "built"
    assert families.FAMILIES["rocsolver"].tol == TOL
    for leg in ("CUDA", "cpu"):
        assert t not in loader._PLATFORMS[leg]["targets"] and t not in ffi_loader._PLATFORMS[leg]["targets"]


def test_compare_is_gauge_and_degeneracy_blind():
    rng = np.random.default_rng(1)
    a = _degenerate(rng, 2, 30, "complex128", 3)
    w, v = np.linalg.eigh(a)
    # A random unitary inside every 3-fold cluster: a different basis of the same eigenspaces.
    v2 = v.copy()
    for i in range(2):
        for lo in range(0, 30, 3):
            q, _r = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
            v2[i][:, lo:lo + 3] = v[i][:, lo:lo + 3] @ q
    assert passes(compare(a, w, v2, w, v))
    v3 = v2.copy(); v3[0][:, [0, 3]] = v3[0][:, [3, 0]]     # swapped across clusters: caught
    assert not passes(compare(a, w, v3, w, v))


def test_device_gate_needs_rocm():
    pytest.skip("run as a script on a ROCm mesh: python -m tests.test_rocsolver_eigh_gate")


if __name__ == "__main__":
    sys.exit(main())
