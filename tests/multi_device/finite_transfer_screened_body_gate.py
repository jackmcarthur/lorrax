"""Real P4 gate for finite-q current/contact and the private TT oracle.

The deterministic single-device cells in ``test_dft_gauge_vertices.py`` own
the independent numerical oracles.  This cell owns the production layout:
one 2x2 mesh, non-divisible logical band/centroid extents, WfnLoader-paired
k/G/box labels, a nonzero q-IBZ row after a same-shape q=0 call, the incumbent
distributed face Green builder, and the public row's fail-closed identity
gate.  The same all-P band sweep contracts the exact q,-q kinetic+VNL
contact, preserving the receipt/fingerprint identity and exact-zero band
tail.  Endpoint, contact and target faces carry one immutable basis receipt;
production publication remains blocked on the separate C/Z, completion,
rectangular-IBZ-action, and artifact-provenance work.

Run only through the Perlmutter compute harness::

    lx run -N 1 -G 4 -n 4 python3 -u \
      tests/multi_device/finite_transfer_screened_body_gate.py --mesh 2x2
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

_TESTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_TESTS))
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

if __name__ == "__main__":
    from runtime import initialize_communicator_stack
    _RUNTIME = initialize_communicator_stack(platform="gpu")

import numpy as np  # noqa: E402


def _gather(value):
    import jax
    if jax.process_count() == 1:
        return np.asarray(value)
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(value, tiled=True))


def _put(value, mesh, spec):
    import jax
    from jax.sharding import NamedSharding
    return jax.device_put(np.asarray(value), NamedSharding(mesh, spec))


def check_finite_transfer_screened_body(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from common import mtxel_sweep
    from common.bispinor_init import lift_to_4spinor
    from common.parallel_transport import wfn_fingerprint
    from common.wfn_layout import band_sphere_spec
    from common.wfn_transforms import gflat_to_rmu
    from file_io.wfn_basis import WavefunctionBasisReceipt
    from gw import w_isdf
    from gw.wavefunction_bundle import (
        AuthenticatedWavefunctions, BandSlices, PSI_MUN_SPEC, PSI_NMU_SPEC,
        Wavefunctions)
    from psp import dft_operators
    from runtime.padding import pad_axis, padded_mu_extent
    from tests.test_dft_gauge_vertices import _setup

    if tuple(mesh.devices.shape) != (2, 2):
        raise AssertionError(f"gate requires a 2x2 mesh; got {mesh.devices.shape}")
    setup = replace(
        _setup(curved=True, third_derivatives=True),
        uniform_gauge_fingerprint="sha256:" + "5" * 64)
    nk, nb_logical, ngkmax, ns = 4, 5, 4, 4
    fft_grid = (4, 1, 1)
    G = np.asarray([
        [-1, 0, 0], [0, 0, 0], [1, 0, 0], [-2, 0, 0],
    ], dtype=np.int32)
    gvecs = np.broadcast_to(G, (nk, ngkmax, 3)).copy()
    ngk = np.full(nk, 3, dtype=np.int32)
    mask = (np.arange(ngkmax)[None, :] < ngk[:, None]).astype(np.float64)
    kvecs = np.zeros((nk, 3), dtype=np.float64)
    kvecs[:, 0] = np.arange(nk) / nk

    rng = np.random.default_rng(20260825)
    psi_L = (rng.standard_normal((nk, nb_logical, 2, ngkmax))
             + 1j * rng.standard_normal((nk, nb_logical, 2, ngkmax)))
    psi_L *= mask[:, None, None, :]
    psi_4 = lift_to_4spinor(
        jnp.asarray(psi_L), jnp.asarray(gvecs), jnp.asarray(kvecs),
        jnp.asarray(setup.B))

    geom = mtxel_sweep.SweepGeometry(
        mesh=mesh, fft_grid=fft_grid, ngkmax=ngkmax,
        nb=nb_logical, ns=ns, nk=nk, cell_volume=1.0)
    if geom.nb != 8 or geom.p_prod != 4:
        raise AssertionError(
            f"gate expected band pad 5->8 at P4; got nb={geom.nb}, "
            f"p_prod={geom.p_prod}")
    psi_4 = pad_axis(psi_4, geom.p_prod, axis=1).array
    psi_4 = _put(psi_4, mesh, band_sphere_spec())

    box_index = np.full((nk, 4, 1, 1), ngkmax, dtype=np.int32)
    for ik in range(nk):
        for ig, g in enumerate(G[:3]):
            box_index[ik, int(g[0]) % 4, 0, 0] = ig
    box_index_dev = _put(box_index, mesh, P(None, None, None, None))
    r_mu = np.asarray([
        [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [1, 0, 0],
    ], dtype=np.int32)

    q_int = np.zeros((nk, 3), dtype=np.int32)
    q_int[:, 0] = np.arange(nk)
    kqfull = np.asarray([
        [(ik - iq) % nk for iq in range(nk)] for ik in range(nk)
    ], dtype=np.int32)
    sym = SimpleNamespace(
        q_irr_kgrid_int=q_int,
        q_irr_full_idx=np.arange(nk, dtype=np.int32),
        kqfull_map=kqfull)
    wfn = SimpleNamespace(
        nbands=nb_logical, nspinor=2, nelec=2,
        energies=np.zeros((1, nk, nb_logical), dtype=np.float64),
        kpoints=kvecs, kgrid=np.asarray([4, 1, 1]),
        fft_grid=fft_grid, ngkmax=ngkmax,
        symmetry=lambda: sym,
        gvecs=lambda *, k: gvecs,
        ngk_valid=lambda *, k: ngk,
        kvecs=lambda *, k: kvecs,
        box_index_dev=lambda *, k, mesh: box_index_dev)
    # The production door receives a real WfnLoader.  This deterministic
    # in-memory fixture supplies the same loader protocol without an HDF5
    # file, so route only this gate's process through the existing owner.
    dft_operators._as_loader = lambda value: value
    basis_wfn_fingerprint = wfn_fingerprint(wfn)
    endpoint_kwargs = dict(
        wfn=wfn, band_start=0, band_stop=nb_logical, geom=geom,
        vnl_setup=setup, r_mu=r_mu,
        basis_receipt=WavefunctionBasisReceipt.from_source(
            wfn=wfn, wfn_fingerprint_value=basis_wfn_fingerprint,
            role='transverse', bispinor=True,
            band_interval=(0, nb_logical), fft_grid=fft_grid,
            centroid_fft_idx=r_mu, n_rmu_logical=len(r_mu),
            n_rmu_padded=padded_mu_extent(len(r_mu), mesh)),
        projector_row_chunk=1, g_chunk=2,
        include_transfer_q2_identity=True)

    # The second call must hit the same endpoint executable while taking q,
    # target rows and wraps as runtime operands.
    endpoint_q0 = mtxel_sweep.finite_transfer_current_to_centroids(
        psi_4, iq_irr=0, **endpoint_kwargs)
    endpoint = mtxel_sweep.finite_transfer_current_to_centroids(
        psi_4, iq_irr=1, **endpoint_kwargs)
    endpoint_minus_q = mtxel_sweep.finite_transfer_current_to_centroids(
        psi_4, iq_irr=3, **endpoint_kwargs)
    endpoint.current_nmu.block_until_ready()
    del endpoint_q0

    contact_kwargs = dict(
        wfn=wfn, band_start=0, band_stop=nb_logical, geom=geom,
        vnl_setup=setup, basis_receipt=endpoint_kwargs["basis_receipt"],
        wfn_fingerprint_value=basis_wfn_fingerprint,
        projector_row_chunk=1, g_chunk=2,
        include_transfer_q2_identity=True)
    contact = mtxel_sweep.sweep_finite_transfer_contact_matrix_elements(
        psi_4, iq_irr=1, **contact_kwargs)
    contact_minus_q = (
        mtxel_sweep.sweep_finite_transfer_contact_matrix_elements(
            psi_4, iq_irr=3, **contact_kwargs))
    contact.lambda_raw.block_until_ready()

    if endpoint.current_nmu.sharding.spec != P(None, "x", None, None, "y"):
        raise AssertionError(
            f"wrong endpoint nmu sharding {endpoint.current_nmu.sharding.spec}")
    if endpoint.current_mun.sharding.spec != P(None, None, None, "x", "y"):
        raise AssertionError(
            f"wrong endpoint mun sharding {endpoint.current_mun.sharding.spec}")
    if tuple(endpoint.current_nmu.shape) != (nk, 8, 3, ns, 8):
        raise AssertionError(f"wrong padded endpoint shape {endpoint.current_nmu.shape}")
    np.testing.assert_array_equal(endpoint.kminq_idx, kqfull[:, 1])
    np.testing.assert_array_equal(endpoint.q_irr_kgrid_int, [1, 0, 0])
    np.testing.assert_array_equal(endpoint.q_crys, [0.25, 0.0, 0.0])
    endpoint_host = _gather(endpoint.current_nmu)
    band_tail = float(np.max(np.abs(endpoint_host[:, nb_logical:])))
    centroid_tail = float(np.max(np.abs(endpoint_host[..., len(r_mu):])))
    if band_tail != 0.0 or centroid_tail != 0.0:
        raise AssertionError(
            f"endpoint pad is not exact zero: band={band_tail}, "
            f"centroid={centroid_tail}")

    contact_spec = P(None, None, None, "x", "y")
    if contact.lambda_raw.sharding.spec != contact_spec:
        raise AssertionError(
            f"wrong contact sharding {contact.lambda_raw.sharding.spec}")
    if tuple(contact.lambda_raw.shape) != (nk, 3, 3, 8, 8):
        raise AssertionError(
            f"wrong padded contact shape {contact.lambda_raw.shape}")
    if (contact.hamiltonian_config_operator_fingerprint
            != endpoint.hamiltonian_config_operator_fingerprint):
        raise AssertionError("current/contact Hamiltonian identities differ")
    if contact.basis_receipt is not endpoint.basis_receipt:
        raise AssertionError("current/contact basis receipt object split")
    if (not contact.vnl_contact_ward_certified
            or contact.downfolded_complement is not None):
        raise AssertionError(
            "contact Ward/complement state does not remain fail-closed")
    contact_host = _gather(contact.lambda_raw)
    contact_minus_host = _gather(contact_minus_q.lambda_raw)
    contact_tail = max(
        float(np.max(np.abs(contact_host[..., nb_logical:, :]))),
        float(np.max(np.abs(contact_host[..., :, nb_logical:]))))
    if contact_tail != 0.0:
        raise AssertionError(
            f"contact band pad is not exact zero: {contact_tail}")
    np.testing.assert_allclose(
        contact_host, contact_minus_host, rtol=8.0e-12, atol=8.0e-12)

    raw_rmu = gflat_to_rmu(
        psi_4, box_index_dev, r_mu, mesh=mesh, fft_grid=fft_grid,
        kvecs_frac=kvecs, norm="ortho")
    raw_rmu = pad_axis(
        raw_rmu, padded_mu_extent(len(r_mu), mesh), axis=-1).array
    to_faces = jax.jit(
        lambda value: (value, value.transpose(0, 2, 3, 1)),
        out_shardings=(NamedSharding(mesh, PSI_NMU_SPEC),
                       NamedSharding(mesh, PSI_MUN_SPEC)))
    psi_nmu, psi_mun = to_faces(raw_rmu)
    energies = np.full((nk, geom.nb), 100.0, dtype=np.float64)
    energies[:, :nb_logical] = np.asarray(
        [-0.8, -0.3, 0.4, 0.9, 1.4])[None, :]
    slices = BandSlices.from_band_edges(
        0, 0, 2, 5, 8, b4_chi=5, b4_sigma=8)
    wfns = Wavefunctions(
        enk=_put(energies, mesh, P(None, None)),
        occ=_put(np.zeros_like(energies), mesh, P(None, None)),
        slices=slices, psi_nmu=psi_nmu, psi_mun=psi_mun, layout="face",
    )
    quad = SimpleNamespace(
        tau=np.asarray([0.0]), alpha=np.asarray([1.0]))
    meta = SimpleNamespace(nkx=4, nky=1, nkz=1, nk_tot=nk)
    try:
        w_isdf.compute_finite_transfer_current_block_row(
            endpoint, wfns, quad, meta, mesh,
            vertex_left=1, vertex_right=2)
    except NotImplementedError as exc:
        if "FULL remains unavailable" not in str(exc):
            raise
    else:
        raise AssertionError(
            "finite-transfer public body row bypassed the remaining FULL "
            "refusal")
    wrong_receipt = replace(
        endpoint.basis_receipt, centroid_table_md5="0" * 32)
    try:
        w_isdf._compute_finite_transfer_current_block_row_unverified(
            endpoint, endpoint_minus_q,
            AuthenticatedWavefunctions(wfns, wrong_receipt),
            quad, meta, mesh,
            vertex_left=1, vertex_right=2)
    except ValueError as exc:
        if "before Green contraction" not in str(exc):
            raise
    else:
        raise AssertionError(
            "finite-transfer Green oracle accepted mismatched target faces")
    chi = w_isdf._compute_finite_transfer_current_block_row_unverified(
        endpoint, endpoint_minus_q,
        AuthenticatedWavefunctions(wfns, endpoint.basis_receipt),
        quad, meta, mesh,
        vertex_left=1, vertex_right=2)
    chi.block_until_ready()
    if chi.sharding.spec != P("x", "y"):
        raise AssertionError(f"wrong fixed-q chi sharding {chi.sharding.spec}")
    chi_host = _gather(chi)
    chi_pad = max(float(np.max(np.abs(chi_host[len(r_mu):]))),
                  float(np.max(np.abs(chi_host[:, len(r_mu):]))))
    if chi_pad != 0.0 or not np.all(np.isfinite(chi_host)):
        raise AssertionError(
            f"fixed-q body pad/finite check failed: pad={chi_pad}")
    ward_abs = float(np.max(_gather(endpoint.vnl_ward_residual_abs)))
    return {
        "endpoint_shape": tuple(int(v) for v in endpoint.current_nmu.shape),
        "band_pad_max": band_tail,
        "centroid_pad_max": centroid_tail,
        "contact_shape": tuple(int(v) for v in contact.lambda_raw.shape),
        "contact_band_pad_max": contact_tail,
        "chi_pad_max": chi_pad,
        "ward_abs_max": ward_abs,
    }


def _mesh_from_arg(spec):
    import jax
    from jax.sharding import Mesh
    px, py = (int(v) for v in spec.lower().split("x"))
    return Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))


def _main():
    import jax
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    args = parser.parse_args()
    mesh = _mesh_from_arg(args.mesh)
    result = check_finite_transfer_screened_body(mesh)
    if jax.process_index() == 0:
        print(f"PASS finite_transfer_screened_body[{args.mesh}] {result}",
              flush=True)
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(_main)
