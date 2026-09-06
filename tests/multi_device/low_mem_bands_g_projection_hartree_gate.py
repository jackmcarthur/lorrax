"""Real 4-rank CUDA gate: algebra parity for the face-layout G builder and
band projector against BOTH the legacy code path AND a NumPy reference.

Guide: reports/gwjax_low_mem_bands_audit_2026-08-22/report.md, census rows
2/3/4 and sections 3/4/5 — the task this file certifies.

WHY REAL CUDA, NOT EMULATED.  Every face-layout numeric path added by this
task routes through ``distrib_la.gemm_plan`` (build_G and the two-GEMM
projector) — a cuBLASMp-only surface that refuses on
any non-CUDA mesh by construction (``gemm_plan``'s own docstring, and the
sibling ``feat/distrib-la-planned-gemm-2026-08-22`` branch this builds on).
So there is no emulated-CPU leg for these checks to fall back to; this file
IS the certification.  What CAN and DOES run on an emulated CPU mesh —
refusal-path coverage and the (s,mu) GEMM-seam merge/split reshape's own
correctness-and-no-collective claim — lives in
``tests/test_windowed_exp_iEt.py``, ``tests/test_contract_bands.py``,
``tests/test_wavefunction_bundle_face_carrier.py`` and
``tests/test_cohsex_sigma_face.py``.

SAME ψ, EVERY REPRESENTATION.  Every check below builds ONE host NumPy ψ
array and derives every legacy/face copy from it by pure reshape/
transpose — never independent random data per copy — because that is
exactly how the real bundle builders work (``wavefunction_bundle.
build_wavefunctions``/``build_wavefunctions_face``, and the carrier's own
``test_face_matches_legacy_same_psi``): psi_xr/psi_yr/psi_nmu are the SAME
un-conjugated ψ values, only differently sharded, and psi_xn/psi_yn/
psi_mun are the same values with the band axis moved last.  A NumPy
reference computed straight from the physics formula is the ground truth
every check compares BOTH layouts against — not just each other, since two
implementations can agree and both be wrong.

Checks:
  1. G, identity weight (ns=1 and ns=2) — build_G(phases=None).
  2. G, diagonal band weight (ns=1 and ns=2) — build_G(phases=w).
  3. G, HOSTILE logical pad — a val/cond-style window boundary that does
     NOT divide the mesh (report obstacle #3's bring-up path: legacy
     physically slices the window, face carries the FULL extent and a
     band-identity mask that is zero outside it) — build_G_tau.
  4. G, dense Gij REFUSES by name under layout='face', even with a real
     gemm plan in hand (obstacle #4's named escape hatch).
  5. Projection — contract_bands_block_reshard(layout='face') against
     layout='legacy' and a NumPy einsum, ns=1 and ns=2.
Run:
    lx run -G 4 -n 4 env PYTHONPATH=... python3 -u \\
        tests/multi_device/low_mem_bands_g_projection_hartree_gate.py \\
        --mesh 2x2
"""
from __future__ import annotations

import argparse
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_TESTS)
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

if __name__ == "__main__":
    from runtime import initialize_communicator_stack
    _RUNTIME = initialize_communicator_stack(platform="gpu")

import numpy as np                                              # noqa: E402

RTOL = 1e-10


# ---------------------------------------------------------------------------
# Shared operand / measurement helpers — same idiom as
# services/distrib_la/tests/test_distrib_la_multiproc.py.
# ---------------------------------------------------------------------------

def _rng_mat(rng, shape, dtype=np.complex128):
    a = rng.standard_normal(shape)
    if np.dtype(dtype).kind == "c":
        a = a + 1j * rng.standard_normal(shape)
    return a.astype(dtype)


def _put(np_arr, mesh, spec):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.device_put(np.asarray(np_arr), NamedSharding(mesh, P(*spec)))


def _gather(x):
    import jax
    if jax.process_count() == 1:
        return np.asarray(x)
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max()) / max(float(np.abs(b).max()), 1e-300)


class _raises:
    def __init__(self, exc, match=""):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(
                f"expected {self.exc} matching {self.match!r}, nothing raised")
        if not issubclass(et, self.exc):
            return False
        if self.match and self.match not in str(ev):
            raise AssertionError(
                f"{et.__name__}({ev}) did not mention {self.match!r}")
        return True


# ---------------------------------------------------------------------------
# 1/2. G — identity and diagonal-band-weight parity.
# ---------------------------------------------------------------------------

def check_g_weighted(mesh, dtype="complex128", *, ns=2, mu=8, nb=6, nk=2,
                     weighted=False):
    from gw.greens_function_kernel import build_G
    from distrib_la import gemm_plan

    rng = np.random.default_rng(2026082201 + ns)
    psi_np = _rng_mat(rng, (nk, nb, ns, mu), dtype)   # (nk, n, s, mu)

    # legacy: psi_xn = psi_np transposed to (nk,s,mu,n); psi_yr = psi_np as-is
    psi_xn_np = psi_np.transpose(0, 2, 3, 1)
    psi_xn = _put(psi_xn_np, mesh, (None, None, "x", None))
    psi_yr = _put(psi_np, mesh, (None, None, None, "y"))

    # face: psi_mun = same transpose as psi_xn; psi_nmu = same as psi_yr
    psi_mun = _put(psi_xn_np, mesh, (None, None, "x", "y"))
    psi_nmu = _put(psi_np, mesh, (None, "x", None, "y"))

    w_np = _rng_mat(rng, (nk, nb), dtype) if weighted else None
    w = _put(w_np, mesh, (None, None)) if weighted else None

    plan = gemm_plan(mesh, m=mu * ns, k=nb, n=mu * ns, nq=nk, dtype=dtype)

    G_legacy = build_G(psi_xn, psi_yr, phases=w, layout="legacy")
    G_face = build_G(psi_mun, psi_nmu, phases=w, layout="face", gemm=plan)

    if weighted:
        want = np.einsum("ksxn,kn,knty->ksxty", psi_xn_np, w_np,
                         np.conj(psi_np), optimize=True)
    else:
        want = np.einsum("ksxn,knty->ksxty", psi_xn_np, np.conj(psi_np),
                         optimize=True)

    r_legacy = _rel(_gather(G_legacy), want)
    r_face = _rel(_gather(G_face), want)
    assert r_legacy < RTOL, f"legacy G rel err {r_legacy:.3e}"
    assert r_face < RTOL, f"face G rel err {r_face:.3e}"
    return {"legacy": r_legacy, "face": r_face}


# ---------------------------------------------------------------------------
# 3. G — hostile logical pad (val/cond-style window not mesh-divisible).
# ---------------------------------------------------------------------------

def check_g_hostile_pad(mesh, dtype="complex128", *, ns=2, mu=8, nb_full=8,
                        nk=2, lo=0, hi=3):
    """``hi - lo = 3`` on a 2x2 mesh: a physically-legal legacy slice, but
    NOT the shape a face GEMM could ever tile (report obstacle #3's whole
    reason for existing).  ``nb_full=8`` IS mesh-divisible — the GEMM's
    contracted axis stays at the padded full extent; only the LOGICAL
    window is hostile, expressed as a band-identity mask (obstacle #3's
    bring-up path)."""
    from gw.greens_function_kernel import build_G_tau
    from distrib_la import gemm_plan

    rng = np.random.default_rng(2026082203)
    psi_np = _rng_mat(rng, (nk, nb_full, ns, mu), dtype)
    enk_np = np.sort(rng.standard_normal((nk, nb_full)), axis=1)
    psi_xn_np = psi_np.transpose(0, 2, 3, 1)

    t = 0.37 + 0.0j
    enk_win = enk_np[:, lo:hi]
    e_ref = float(np.max(enk_win))

    psi_xn_win = _put(psi_xn_np[:, :, :, lo:hi], mesh, (None, None, "x", None))
    psi_yr_win = _put(psi_np[:, lo:hi], mesh, (None, None, None, "y"))
    enk_win_dev = _put(enk_win, mesh, (None, None))
    G_legacy = build_G_tau(psi_xn_win, psi_yr_win, enk_win_dev, t,
                           e_ref=e_ref, layout="legacy")

    psi_mun = _put(psi_xn_np, mesh, (None, None, "x", "y"))
    psi_nmu = _put(psi_np, mesh, (None, "x", None, "y"))
    enk_full = _put(enk_np, mesh, (None, None))
    mask_np = np.zeros((nk, nb_full), dtype=bool)
    mask_np[:, lo:hi] = True
    mask = _put(mask_np, mesh, (None, None))
    plan = gemm_plan(mesh, m=mu * ns, k=nb_full, n=mu * ns, nq=nk,
                     dtype=dtype)
    G_face = build_G_tau(psi_mun, psi_nmu, enk_full, t, e_ref=e_ref,
                         mask=mask, layout="face", gemm=plan)

    phase_full = np.exp(-t * (enk_np - e_ref))
    phase_win = np.where(mask_np, phase_full, 0.0)
    want = np.einsum("ksxn,kn,knty->ksxty", psi_xn_np, phase_win,
                     np.conj(psi_np), optimize=True)

    r_legacy = _rel(_gather(G_legacy), want)
    r_face = _rel(_gather(G_face), want)
    assert r_legacy < RTOL, f"legacy hostile-pad rel err {r_legacy:.3e}"
    assert r_face < RTOL, f"face hostile-pad rel err {r_face:.3e}"
    return {"legacy": r_legacy, "face": r_face, "window": f"[{lo},{hi})"}


# ---------------------------------------------------------------------------
# 4. G — dense Gij refuses by name under face, even with a real plan.
# ---------------------------------------------------------------------------

def check_g_dense_gij_refuses(mesh, dtype="complex128", *, ns=1, mu=8,
                              nb=4, nk=2):
    from gw.greens_function_kernel import build_G
    from distrib_la import gemm_plan

    plan = gemm_plan(mesh, m=mu * ns, k=nb, n=mu * ns, nq=nk, dtype=dtype)
    psi_mun = _put(np.zeros((nk, ns, mu, nb), dtype), mesh,
                   (None, None, "x", "y"))
    psi_nmu = _put(np.zeros((nk, nb, ns, mu), dtype), mesh,
                   (None, "x", None, "y"))
    Gij = _put(np.eye(nb, dtype=dtype)[None].repeat(nk, axis=0), mesh,
              (None, None, None))
    with _raises(NotImplementedError, "Gij"):
        build_G(psi_mun, psi_nmu, Gij=Gij, layout="face", gemm=plan)
    return True


def check_g_parent_unfold(mesh, dtype="complex128", *, ns=2, mu=8, nb=8,
                          nk_parent=2, nk_full=6):
    """Real distributed GEMM on parent k, then authenticated local unfold."""
    from types import SimpleNamespace

    import jax.numpy as jnp

    from distrib_la import gemm_plan
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw.greens_function_kernel import build_G

    identity = np.eye(3, dtype=np.int32)
    irr = np.arange(nk_full, dtype=np.int32) % nk_parent

    def spinor_action(rows, *, nspinor):
        rows = np.asarray(rows)
        return np.broadcast_to(
            np.eye(nspinor, dtype=np.complex128),
            rows.shape + (nspinor, nspinor)).copy()

    sym = SimpleNamespace(
        sym_matrices=identity[None],
        translations=np.zeros((1, 3), dtype=np.float64),
        irr_idx_k=irr,
        sym_idx_k=np.zeros((nk_full,), dtype=np.int32),
        unfolded_kpts=np.zeros((nk_full, 3), dtype=np.float64),
        kirr_fullids=np.arange(nk_parent, dtype=np.int32),
        spinor_action=spinor_action,
    )
    centroids = np.stack((
        np.arange(mu, dtype=np.int32),
        np.zeros(mu, dtype=np.int32),
        np.zeros(mu, dtype=np.int32)), axis=1)
    plan = build_centroid_k_unfold_plan(
        sym, centroids, (mu, 1, 1), mesh, nspinor=ns,
        parent_k_frac=np.zeros((nk_parent, 3), dtype=np.float64))

    rng = np.random.default_rng(2026090101 + ns)
    parent_np = _rng_mat(rng, (nk_parent, nb, ns, mu), dtype)
    full_np = parent_np[irr]
    # Every face is in the run's packed centroid order (the loader samples
    # the packed table; the host packer stands in for it here).
    pack = plan.layout.axis.pack_host
    parent_np = pack(parent_np, axis=3)
    full_np = pack(full_np, axis=3)
    mu = int(plan.n_centroid_packed)
    full_mun = _put(
        full_np.transpose(0, 2, 3, 1), mesh,
        (None, None, "x", "y"))
    full_nmu = _put(full_np, mesh, (None, "x", None, "y"))
    parent_mun = _put(
        parent_np.transpose(0, 2, 3, 1), mesh,
        (None, None, "x", "y"))
    parent_nmu = _put(parent_np, mesh, (None, "x", None, "y"))

    full_gemm = gemm_plan(
        mesh, m=mu * ns, k=nb, n=mu * ns, nq=nk_full, dtype=dtype)
    parent_gemm = gemm_plan(
        mesh, m=mu * ns, k=nb, n=mu * ns, nq=nk_parent, dtype=dtype)
    got_full = build_G(
        full_mun, full_nmu, layout="face", gemm=full_gemm)
    got_parent = build_G(
        parent_mun, parent_nmu, layout="face", gemm=parent_gemm,
        k_unfold_plan=plan)
    jnp.asarray(got_parent).block_until_ready()

    err = _rel(_gather(got_parent), _gather(got_full))
    assert err < RTOL, f"parent-k unfolded G rel err {err:.3e}"
    return {"rel": err, "nk": f"{nk_parent}->{nk_full}"}


# ---------------------------------------------------------------------------
# 5. Projection.
# ---------------------------------------------------------------------------

def check_projection(mesh, dtype="complex128", *, ns=2, mu=8, nb=4, nk=2):
    from common.contract_bands import contract_bands_block_reshard

    rng = np.random.default_rng(2026082205 + ns)
    O_np = _rng_mat(rng, (nk, ns, mu, ns, mu), dtype)
    psi_l_np = _rng_mat(rng, (nk, nb, ns, mu), dtype)   # (nk, m, s, mu)
    psi_r_np = _rng_mat(rng, (nk, ns, mu, nb), dtype)   # (nk, s', nu, n)

    O = _put(O_np, mesh, (None, None, "x", None, "y"))
    psi_l = _put(psi_l_np, mesh, (None, None, None, "x"))
    psi_r = _put(psi_r_np, mesh, (None, None, "y", None))
    proj_legacy = contract_bands_block_reshard(mesh)
    got_legacy = proj_legacy(psi_l, O, psi_r)

    # face: psi_nmu(nk,n,s,mu) == psi_l_np's OWN shape already; psi_mun
    # (nk,s,mu,n) == psi_r_np's OWN shape already -- same arrays, same
    # physical psi, only the mesh sharding differs (see module docstring).
    psi_nmu = _put(psi_l_np, mesh, (None, "x", None, "y"))
    psi_mun = _put(psi_r_np, mesh, (None, None, "x", "y"))
    proj_face = contract_bands_block_reshard(
        mesh, layout="face", face_shape=(nk, nb, mu, ns))
    got_face = proj_face(psi_nmu, O, psi_mun)

    want = np.einsum("kmsx,ksxty,ktyn->kmn", np.conj(psi_l_np), O_np,
                     psi_r_np, optimize=True)

    r_legacy = _rel(_gather(got_legacy), want)
    r_face = _rel(_gather(got_face), want)
    assert r_legacy < RTOL, f"legacy projection rel err {r_legacy:.3e}"
    assert r_face < RTOL, f"face projection rel err {r_face:.3e}"
    return {"legacy": r_legacy, "face": r_face}


# ---------------------------------------------------------------------------
# 6. Hartree.
# ---------------------------------------------------------------------------

_CLI_CELLS = [
    ("g_identity_ns1", lambda mesh, dt: check_g_weighted(mesh, dt, ns=1)),
    ("g_identity_ns2", lambda mesh, dt: check_g_weighted(mesh, dt, ns=2)),
    ("g_diag_weight_ns1",
     lambda mesh, dt: check_g_weighted(mesh, dt, ns=1, weighted=True)),
    ("g_diag_weight_ns2",
     lambda mesh, dt: check_g_weighted(mesh, dt, ns=2, weighted=True)),
    ("g_hostile_pad_ns2", lambda mesh, dt: check_g_hostile_pad(mesh, dt, ns=2)),
    ("g_hostile_pad_ns1", lambda mesh, dt: check_g_hostile_pad(mesh, dt, ns=1)),
    ("g_dense_gij_refuses",
     lambda mesh, dt: check_g_dense_gij_refuses(mesh, dt)),
    ("g_parent_unfold_ns2",
     lambda mesh, dt: check_g_parent_unfold(mesh, dt, ns=2)),
    ("projection_ns1", lambda mesh, dt: check_projection(mesh, dt, ns=1)),
    ("projection_ns2", lambda mesh, dt: check_projection(mesh, dt, ns=2)),
]


def _mesh_from_arg(spec):
    import jax
    from jax.sharding import Mesh
    px, py = (int(v) for v in spec.lower().split("x"))
    return Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))


def _cli_main():
    import jax

    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--dtype", default="complex128")
    args = ap.parse_args()
    mesh = _mesh_from_arg(args.mesh)
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    p0(f"backend={jax.default_backend()} mesh={args.mesh} "
       f"processes={jax.process_count()} devices={jax.device_count()}",
       flush=True)

    failures, ran = 0, 0
    for name, fn in _CLI_CELLS:
        if args.only and args.only not in name:
            continue
        tag = f"{name}[{args.mesh},{args.dtype}]"
        try:
            out = fn(mesh, args.dtype)
            ran += 1
            p0(f"PASS {tag} {out if out is not True else ''}", flush=True)
        except AssertionError as exc:
            failures += 1
            p0(f"FAIL {tag}: {exc}", flush=True)
        except Exception as exc:                                # noqa: BLE001
            failures += 1
            p0(f"ERROR {tag}: {type(exc).__name__}: "
               f"{' '.join(str(exc).split())[:600]}", flush=True)
    p0(f"done: {ran} cells ran, {failures} failures", flush=True)
    return 1 if (failures or ran == 0) else 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(_cli_main)
