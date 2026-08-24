"""Deterministic P4 algebra gate for transverse photon direct Hartree.

The oracle is the dense four-spinor formula, written with explicit 4x4
alpha matrices rather than LORRAX's monomial-gamma implementation.  The gate
also pins the sign and sole 1/Nk prefactor, the exact large-component-only
zero, quadratic kinetic-balance scaling, and the shared scalar-Hartree owner.

Run on one four-GPU node::

    lx run -G 4 -n 4 env PYTHONPATH=... python3 -u \
      tests/multi_device/photon_hartree_direct_gate.py --mesh 2x2
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

_TESTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_TESTS))
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

if __name__ == "__main__":
    from runtime import initialize_communicator_stack
    _RUNTIME = initialize_communicator_stack(platform="gpu")

import numpy as np  # noqa: E402


_ALPHA = np.asarray([
    [[0, 0, 0, 1], [0, 0, 1, 0],
     [0, 1, 0, 0], [1, 0, 0, 0]],
    [[0, 0, 0, -1j], [0, 0, 1j, 0],
     [0, -1j, 0, 0], [1j, 0, 0, 0]],
    [[0, 0, 1, 0], [0, 0, 0, -1],
     [1, 0, 0, 0], [0, -1, 0, 0]],
], dtype=np.complex128)


def _put(array, mesh, spec):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.device_put(
        np.asarray(array), NamedSharding(mesh, P(*spec)))


def _gather(array):
    import jax
    if jax.process_count() == 1:
        return np.asarray(array)
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(array, tiled=True))


def _relative_error(got, want):
    scale = max(float(np.max(np.abs(want))), 1e-300)
    return float(np.max(np.abs(got - want))) / scale


def _psi(small_scale: float, *, nk=2, nb=8, nmu_logical=5,
         nmu_padded=8):
    """Fixed analytic bispinor; no PRNG and exact-zero padding."""
    k, n, s, mu = np.indices((nk, nb, 4, nmu_logical))
    real = (0.13 * (k + 1) + 0.071 * (n + 1)
            + 0.037 * (s + 1) + 0.019 * (mu + 1))
    imag = (0.023 * (k + 1) - 0.041 * (n + 1)
            + 0.029 * (s + 1) - 0.017 * (mu + 1))
    logical = (real + 1j * imag) / (n + 2.0)
    logical[:, :, 2:, :] *= float(small_scale)
    out = np.zeros((nk, nb, 4, nmu_padded), dtype=np.complex128)
    out[..., :nmu_logical] = logical
    return out


def _dense_tt_hartree(psi, occupations, V_tt, *, nb_sigma):
    nk = psi.shape[0]
    psi_eval = psi[:, :nb_sigma]
    rho = np.empty((3, psi.shape[-1]), dtype=np.float64)
    for B in range(3):
        rho[B] = np.real(np.einsum(
            "knsx,st,kntx,kn->x", np.conj(psi_eval), _ALPHA[B],
            psi_eval, occupations, optimize=True)) / nk

    field = np.zeros((3, psi.shape[-1]), dtype=np.complex128)
    for A in range(3):
        for B in range(3):
            field[A] += V_tt[A, B] @ rho[B]

    result = np.zeros(
        (nk, nb_sigma, nb_sigma), dtype=np.complex128)
    for A in range(3):
        result += np.einsum(
            "kmsx,st,kntx,x->kmn", np.conj(psi_eval), _ALPHA[A],
            psi_eval, field[A], optimize=True)
    return result


def _make_face_bundle(psi, occupations, mesh, slices):
    from gw.wavefunction_bundle import (
        PSI_MUN_SPEC, PSI_NMU_SPEC, Wavefunctions,
    )
    nk, nb = occupations.shape[0], psi.shape[1]
    enk = np.arange(nk * nb, dtype=np.float64).reshape(nk, nb) / 19.0
    return Wavefunctions(
        psi_nmu=_put(psi, mesh, PSI_NMU_SPEC),
        psi_mun=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_MUN_SPEC),
        enk=_put(enk, mesh, (None, None)),
        occ=_put(np.zeros_like(enk), mesh, (None, None)),
        slices=slices,
        layout="face",
    )


def check(mesh):
    from gw.cohsex_sigma import _make_cohsex_kernels
    from gw.photon_layout import PhotonBasisLayout, pack_photon_operator
    from gw.photon_sigma import compute_static_photon_hartree
    from gw.wavefunction_bundle import BandSlices

    nk, nb, nb_sigma = 2, 8, 5
    n_c_logical, n_t_logical = 3, 5
    layout = PhotonBasisLayout.from_centroid_extents(
        n_c_logical, n_t_logical, mesh)
    n_t = layout.padded_extent(1)
    slices = BandSlices.from_band_edges(0, 0, 2, nb_sigma, nb)

    occupations = np.asarray(
        [[1.0, 0.82, 0.57, 0.31, 0.09],
         [0.93, 0.76, 0.49, 0.22, 0.04]], dtype=np.float64)
    Gij_np = np.zeros((nk, nb_sigma, nb_sigma), dtype=np.complex128)
    diag = np.arange(nb_sigma)
    Gij_np[:, diag, diag] = occupations
    Gij = _put(Gij_np, mesh, (None, None, None))

    u = np.arange(1, n_t_logical + 1, dtype=np.float64)
    K_logical = np.diag(0.7 + 0.11 * u) + 0.013 * np.outer(u, u)
    K = np.zeros((n_t, n_t), dtype=np.complex128)
    K[:n_t_logical, :n_t_logical] = K_logical
    channel_kernel = np.asarray(
        [[1.00, 0.17, -0.08],
         [0.17, 0.73, 0.11],
         [-0.08, 0.11, 1.21]], dtype=np.float64)
    V_tt = np.empty((3, 3, n_t, n_t), dtype=np.complex128)
    for A in range(3):
        for B in range(3):
            V_tt[A, B] = channel_kernel[A, B] * K

    zero_blocks: dict[tuple[int, int], np.ndarray] = {}

    def get_block(A, B):
        shape = layout.block_shape(nk, A, B)
        block = zero_blocks.setdefault(
            (shape[1], shape[2]), np.zeros(shape, dtype=np.complex128))
        if A == 0 or B == 0:
            host = block
        else:
            host = block.copy()
            host[0] = V_tt[A - 1, B - 1]
        return _put(host, mesh, (None, "x", "y"))

    V_packed = pack_photon_operator(get_block, nk, layout, mesh)
    meta = SimpleNamespace(nk_tot=nk)

    def production(small_scale):
        psi = _psi(
            small_scale, nk=nk, nb=nb,
            nmu_logical=n_t_logical, nmu_padded=n_t)
        wfns = _make_face_bundle(psi, occupations, mesh, slices)
        got = compute_static_photon_hartree(
            wfns_transverse=wfns,
            Gij=Gij,
            V_packed=V_packed,
            photon_layout=layout,
            meta=meta,
            mesh_xy=mesh,
            verbose=False,
        )
        return psi, wfns, _gather(got)

    psi1, wfns1, got1 = production(1.0)
    want1 = _dense_tt_hartree(
        psi1, occupations, V_tt, nb_sigma=nb_sigma)
    tt_error = _relative_error(got1, want1)
    assert tt_error < 2e-12, f"TT direct dense error {tt_error:.3e}"

    hermiticity = _relative_error(
        got1, np.conj(np.swapaxes(got1, -1, -2)))
    assert hermiticity < 2e-12, (
        f"TT direct result is not Hermitian: {hermiticity:.3e}")

    _, _, got0 = production(0.0)
    nonrel_zero = float(np.max(np.abs(got0)))
    assert nonrel_zero == 0.0, (
        f"large-component-only TT direct is not exact zero: {nonrel_zero}")

    _, _, got2 = production(2.0)
    scale_error = _relative_error(got2, 4.0 * got1)
    assert scale_error < 2e-12, (
        f"small-component x2 did not give TT Hartree x4: {scale_error:.3e}")

    # The refactor's scalar identity channel remains the same density and
    # local-potential owner, with the same positive Hartree sign and 1/Nk.
    V00 = np.zeros((nk, n_t, n_t), dtype=np.complex128)
    V00[0] = K
    V00_dev = _put(V00, mesh, (None, "x", "y"))
    _, _, scalar_hartree = _make_cohsex_kernels(
        mesh, (nk, 1, 1), nk, layout="face",
        face_shape=(nk, nb, n_t, 4))
    got_scalar = _gather(scalar_hartree(wfns1, Gij, V00_dev))[
        :, :nb_sigma, :nb_sigma]
    rho0_ref = np.einsum(
        "knsx,kn->x", np.abs(psi1[:, :nb_sigma]) ** 2,
        occupations, optimize=True) / nk
    phi0_ref = K @ rho0_ref
    want_scalar = np.einsum(
        "kmsx,x,knsx->kmn", np.conj(psi1[:, :nb_sigma]), phi0_ref,
        psi1[:, :nb_sigma], optimize=True)
    scalar_error = _relative_error(got_scalar, want_scalar)
    assert scalar_error < 2e-12, (
        f"shared scalar Hartree error {scalar_error:.3e}")

    # A sign flip is not a convention available to this stored-V formula:
    # the plus-sign oracle must be much closer than its negative red twin.
    wrong_sign_error = _relative_error(got1, -want1)
    assert wrong_sign_error > 1.0, (
        f"TT Hartree sign gate is non-discriminating: {wrong_sign_error:.3e}")
    return {
        "tt_dense_rel": tt_error,
        "hermiticity_rel": hermiticity,
        "nonrel_max": nonrel_zero,
        "small_component_x2_rel_to_x4": scale_error,
        "scalar_dense_rel": scalar_error,
        "wrong_sign_rel": wrong_sign_error,
    }


def _main():
    import jax
    from jax.sharding import Mesh

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    args = parser.parse_args()
    px, py = (int(v) for v in args.mesh.lower().split("x"))
    mesh = Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))
    result = check(mesh)
    if jax.process_index() == 0:
        print(
            f"PASS photon_hartree_direct backend={jax.default_backend()} "
            f"mesh={args.mesh} {result}", flush=True)
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(_main)
