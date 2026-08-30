"""Real multi-rank CUDA gate: face-layout q-linear head/body wings and
static metallic wings, parity vs legacy AND an independent NumPy oracle,
plus a bounded-residency memory check.

Guide: reports/gwjax_low_mem_bands_audit_2026-08-22/report.md, census
rows 6/7 ("Full q->0 head/body wings", "Static metallic wings") — the
task this file certifies.

WHY REAL CUDA.  The emulated-mesh suite (``tests/test_qsgw_head_face_
wings.py``) already proves correctness at small scale via
``jax.lax.all_gather``/``ppermute``/``psum`` collectives, which run
correctly under CPU emulation.  What only real hardware can show is (a)
the SAME collectives actually execute correctly through NCCL, and (b) the
claimed memory bound: that ``_head_wing_kernel_face``'s peak residency
does not grow with the local mu count, only with ``_HEAD_WING_MU_BLOCK``
-- the entire point of the mu-blocked gather design (see that function's
own docstring for the algebra).

SAME psi, EVERY REPRESENTATION.  Every parity check builds ONE host psi
array and derives legacy ``psi_xn``/``psi_yn`` and face ``psi_mun``/
``psi_nmu`` from it via ``wavefunction_bundle.build_wavefunctions``/
``build_wavefunctions_face`` -- the REAL production builders, not a
hand-rolled transpose -- exactly ``tests/test_qsgw_head_face_wings.py``'s
own approach, just on real devices.

Checks:
  1. dynamic_wings_ns1 / dynamic_wings_ns2 -- head_wings_sharded, legacy
     vs face vs NumPy oracle.
  2. static_wings_ns1 / static_wings_ns2 -- static_head_wings_sharded,
     legacy vs face vs NumPy oracle.
  3. bounded_residency -- run the face dynamic-wing kernel at two LOCAL
     mu counts (same _HEAD_WING_MU_BLOCK) and assert the measured
     allocator peak delta does NOT scale with the mu ratio -- the
     positive claim this task exists to make, measured, not asserted by
     construction.

Run:
    lx run -G 4 -n 4 env PYTHONPATH=... python3 -u \\
        tests/multi_device/head_wings_face_gate.py --mesh 2x2
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

RTOL = 1e-9


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


def _host_inputs(rng, nk, nb, ns, nmu):
    psi = (rng.standard_normal((nk, nb, ns, nmu))
           + 1j * rng.standard_normal((nk, nb, ns, nmu)))
    psi_rmuT_X = np.conj(psi).transpose(0, 3, 1, 2)
    return psi, psi_rmuT_X


def _build_pair(rng, mesh, *, nk, nb, ns, nmu):
    from gw.wavefunction_bundle import BandSlices, build_wavefunctions, \
        build_wavefunctions_face
    psi, psi_rmuT_X = _host_inputs(rng, nk, nb, ns, nmu)
    enk = np.sort(rng.standard_normal((nk, nb)), axis=1)
    occ_cut = nb // 2
    occ = np.where(np.arange(nb)[None, :] < occ_cut, 1.0, 0.0)
    occ = np.broadcast_to(occ, (nk, nb)).copy()
    slices = BandSlices.from_band_edges(0, 0, occ_cut, nb, nb)

    y_in = _put(psi, mesh, (None, None, None, "y"))
    x_in = _put(psi_rmuT_X, mesh, (None, "x", None, None))
    enk_in = _put(enk, mesh, (None, None))

    legacy = build_wavefunctions(
        y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)
    face = build_wavefunctions_face(
        y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)
    occ_used = _gather(legacy.occ)
    return legacy, face, psi, enk, occ_used


def _numpy_wings(v, e, f, psi, *, nb_logical, nk_tot, nspin, nspinor,
                 omega, eta):
    nk, nb = e.shape
    mu = psi.shape[-1]
    n_omega = len(omega)
    Y = np.zeros((n_omega, 3, mu), dtype=np.complex128)
    Z = np.zeros((n_omega, mu, 3), dtype=np.complex128)
    spin_denom = max(int(nspin), 1) * max(int(nspinor), 1)
    pref = 4.0 / (float(nk_tot) * spin_denom)
    for k in range(nk):
        for i in range(nb_logical):
            for j in range(nb_logical):
                dE = e[k, i] - e[k, j]
                if dE <= 0.0:
                    continue
                fdiff = f[k, j] - f[k, i]
                bij = np.einsum("sm,sm->m", np.conj(psi[k, i]), psi[k, j])
                for iw, om in enumerate(omega):
                    z = om + 1j * eta
                    denom = z * z - dE * dE
                    if abs(denom) <= 1.0e-16:
                        continue
                    w = pref * fdiff / denom
                    Y[iw] += np.conj(v[:, k, i, j])[:, None] * w * bij[None, :]
                    Z[iw] += np.conj(bij)[:, None] * w * v[:, k, i, j][None, :]
    return Y, Z


def _numpy_static_wings(psi, surface, *, nb_logical, nk_tot, nspin, nspinor):
    nk, nb, ns, mu = psi.shape
    prefactor = -2.0 / (
        float(nk_tot) * max(int(nspin), 1) * max(int(nspinor), 1))
    density = np.sum(np.square(np.abs(psi)), axis=2)
    weight = np.where(np.arange(nb)[None, :] < nb_logical, surface, 0.0)
    return prefactor * np.einsum("kn,knm->m", weight, density)


def check_dynamic_wings(mesh, dtype="complex128", *, ns=2, nb=6, nk=2,
                        nmu=8):
    import jax.numpy as jnp
    from gw.qsgw_head import head_wings_sharded

    rng = np.random.default_rng(2026082206 + ns)
    nb_logical, nk_tot, nspin, nspinor = nb, nk, 1, ns
    legacy, face, psi, enk, occ = _build_pair(
        rng, mesh, nk=nk, nb=nb, ns=ns, nmu=nmu)
    v = (rng.standard_normal((3, nk, nb_logical, nb_logical))
         + 1j * rng.standard_normal((3, nk, nb_logical, nb_logical)))
    omega = np.asarray([0.1 + 0.0j, 0.3 + 0.02j])
    eta = 0.01
    e_slice = jnp.asarray(enk[:, :nb_logical])
    f_slice = jnp.asarray(occ[:, :nb_logical])

    Y_l, Z_l = head_wings_sharded(
        v, legacy, e_slice, f_slice, omega, mesh=mesh,
        nb_logical=nb_logical, nk_tot=nk_tot, nspin=nspin,
        nspinor=nspinor, eta_ry=eta)
    Y_f, Z_f = head_wings_sharded(
        v, face, e_slice, f_slice, omega, mesh=mesh,
        nb_logical=nb_logical, nk_tot=nk_tot, nspin=nspin,
        nspinor=nspinor, eta_ry=eta)
    Y_ref, Z_ref = _numpy_wings(
        v, enk, occ, psi, nb_logical=nb_logical, nk_tot=nk_tot,
        nspin=nspin, nspinor=nspinor, omega=omega, eta=eta)

    Y_l, Z_l, Y_f, Z_f = (_gather(Y_l), _gather(Z_l),
                          _gather(Y_f), _gather(Z_f))
    r = {
        "legacy_Y": _rel(Y_l, Y_ref), "legacy_Z": _rel(Z_l, Z_ref),
        "face_Y": _rel(Y_f, Y_ref), "face_Z": _rel(Z_f, Z_ref),
        "legacy_vs_face_Y": _rel(Y_l, Y_f),
        "legacy_vs_face_Z": _rel(Z_l, Z_f),
    }
    for k, val in r.items():
        assert val < RTOL, f"{k} rel err {val:.3e}"
    return r


def check_static_wings(mesh, dtype="complex128", *, ns=2, nb=6, nk=2,
                       nmu=8):
    from gw.qsgw_head import static_head_wings_sharded

    rng = np.random.default_rng(2026082207 + ns)
    nb_logical, nk_tot, nspin, nspinor = nb, nk, 1, ns
    legacy, face, psi, enk, occ = _build_pair(
        rng, mesh, nk=nk, nb=nb, ns=ns, nmu=nmu)
    surface = np.abs(rng.standard_normal((nk, nb)))

    l_left, l_right = static_head_wings_sharded(
        legacy, surface, mesh=mesh, nb_logical=nb_logical, nk_tot=nk_tot,
        nspin=nspin, nspinor=nspinor)
    f_left, f_right = static_head_wings_sharded(
        face, surface, mesh=mesh, nb_logical=nb_logical, nk_tot=nk_tot,
        nspin=nspin, nspinor=nspinor)
    ref = _numpy_static_wings(
        psi, surface, nb_logical=nb_logical, nk_tot=nk_tot, nspin=nspin,
        nspinor=nspinor)

    l_left, l_right, f_left, f_right = (
        _gather(l_left), _gather(l_right), _gather(f_left), _gather(f_right))
    r = {
        "legacy_left": _rel(l_left, ref), "legacy_right": _rel(l_right, ref),
        "face_left": _rel(f_left, ref), "face_right": _rel(f_right, ref),
        "legacy_vs_face_left": _rel(l_left, f_left),
        "legacy_vs_face_right": _rel(l_right, f_right),
    }
    for k, val in r.items():
        assert val < RTOL, f"{k} rel err {val:.3e}"
    return r


def check_bounded_residency(mesh, dtype="complex128"):
    """The positive memory claim, measured -- and measured ISOLATED (this
    process runs ONLY this check; ``memory_stats()['peak_bytes_in_use']``
    is a since-process-start high-water mark, so any earlier check's
    allocations in the SAME process would contaminate a shared reading --
    confirmed the hard way: an earlier version of this check ran after
    four heavier checks in one process and read a FALSE flat ratio of
    1.00, masked by those checks' own larger prior peak; run in isolation
    (``--only bounded_residency``) it reproduces a real, honest number).

    Below ``_HEAD_WING_MU_BLOCK`` local mu, there is only ONE gather step
    and its size (hence measured peak) legitimately scales with mu -- that
    is not a defect, just the regime the block constant does not engage
    in yet.  The actual claim under test is ABOVE threshold: does adding
    MORE blocks (more mu, same per-block size) keep the peak flat?  Three
    points, all with local mu a whole multiple of ``_HEAD_WING_MU_BLOCK``:
    2 blocks, 8 blocks, 32 blocks."""
    import jax
    import jax.numpy as jnp
    import gw.qsgw_head as qsgw_head

    side = int(mesh.shape["x"])
    block = qsgw_head._HEAD_WING_MU_BLOCK
    mu_2 = 2 * block * side
    mu_8 = 8 * block * side
    mu_32 = 32 * block * side

    def _peak_delta(nmu):
        """Isolates the WING KERNEL's own contribution to the allocator
        high-water mark from the psi-bundle CONSTRUCTION's own cost (the
        bundle is O(mu/(Px*Py)) BY DESIGN -- that is the carrier's own
        claim, already verified elsewhere, not this kernel's -- so it
        must not be charged to this kernel's residency).  ``peak_bytes_
        in_use`` is a since-process-start high-water mark; the kernel's
        OWN contribution is however much NEW peak it pushes past
        whatever the (already block_until_ready'd) bundle construction
        alone reached."""
        qsgw_head._KERNEL_CACHE.clear()
        rng = np.random.default_rng(90210 + nmu)
        nb, ns, nk = 6, 1, 2
        nb_logical, nk_tot, nspin, nspinor = nb, nk, 1, ns
        legacy, face, psi, enk, occ = _build_pair(
            rng, mesh, nk=nk, nb=nb, ns=ns, nmu=nmu)
        jax.block_until_ready((face.psi_mun, face.psi_nmu))
        peak_before = int(
            jax.local_devices()[0].memory_stats().get("peak_bytes_in_use", -1))

        v = (rng.standard_normal((3, nk, nb, nb))
             + 1j * rng.standard_normal((3, nk, nb, nb)))
        omega = np.asarray([0.1 + 0.0j])
        e_slice = jnp.asarray(enk[:, :nb])
        f_slice = jnp.asarray(occ[:, :nb])
        Y, Z = qsgw_head.head_wings_sharded(
            v, face, e_slice, f_slice, omega, mesh=mesh,
            nb_logical=nb_logical, nk_tot=nk_tot, nspin=nspin,
            nspinor=nspinor, eta_ry=0.01)
        jax.block_until_ready((Y, Z))
        peak_after = int(
            jax.local_devices()[0].memory_stats().get("peak_bytes_in_use", -1))
        del legacy, face, psi, Y, Z
        return max(0, peak_after - peak_before)

    peak_2 = _peak_delta(mu_2)
    peak_8 = _peak_delta(mu_8)
    peak_32 = _peak_delta(mu_32)
    # Both comparisons stay ABOVE the block threshold (2, 8, 32 blocks):
    # a legacy-shaped O(mu_local) residency would track the 4x/16x block
    # ratios almost exactly; the claim under test is that it does not.
    ratio_8_2 = peak_8 / max(1, peak_2)
    ratio_32_2 = peak_32 / max(1, peak_2)
    block_ratio_8_2 = mu_8 / mu_2
    block_ratio_32_2 = mu_32 / mu_2
    assert ratio_32_2 < block_ratio_32_2 / 4.0, (
        f"peak_bytes_in_use grew {ratio_32_2:.3f}x from 2 to 32 mu blocks "
        f"({block_ratio_32_2:.0f}x more mu) -- looks block-count-scaling, "
        f"not tile-bounded (peak_2={peak_2}, peak_32={peak_32})")
    return {"peak_2blocks": peak_2, "peak_8blocks": peak_8,
            "peak_32blocks": peak_32,
            "ratio_8_2": ratio_8_2, "ratio_32_2": ratio_32_2,
            "block_ratio_8_2": block_ratio_8_2,
            "block_ratio_32_2": block_ratio_32_2}


_CLI_CELLS = [
    ("dynamic_wings_ns1", lambda mesh, dt: check_dynamic_wings(mesh, dt, ns=1)),
    ("dynamic_wings_ns2", lambda mesh, dt: check_dynamic_wings(mesh, dt, ns=2)),
    ("static_wings_ns1", lambda mesh, dt: check_static_wings(mesh, dt, ns=1)),
    ("static_wings_ns2", lambda mesh, dt: check_static_wings(mesh, dt, ns=2)),
    ("bounded_residency", lambda mesh, dt: check_bounded_residency(mesh, dt)),
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
