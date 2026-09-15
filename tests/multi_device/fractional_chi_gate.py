"""P=4 dense Kubo gate for the finite-occupation chi0 contour kernel."""

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

import json
import os
from pathlib import Path
import sys

from types import SimpleNamespace

import jax
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import process_count, process_rank, resolve_mesh
from gw.w_isdf import compute_chi0, compute_chi0_contour_fractional
from gw.wavefunction_bundle import (
    BandSlices,
    PSI_MUN_SPEC,
    PSI_NMU_SPEC,
    Wavefunctions,
)


def _put(value, mesh, spec):
    sharding = NamedSharding(mesh, spec)
    return jax.make_array_from_callback(
        value.shape, sharding, lambda index: value[index])


def _dense(psi, enk, occ, time, weights, z):
    nk, nb, _, nmu = psi.shape
    out = np.zeros((z.size, nk, nmu, nmu), np.complex128)
    projection = weights * np.exp(1j * z[:, None] * time[None, :])
    for q in range(nk):
        for k in range(nk):
            kmq = (k - q) % nk
            for a in range(nb):
                for b in range(nb):
                    delta = enk[kmq, b] - enk[k, a]
                    fdiff = occ[k, a] - occ[kmq, b]
                    M = np.einsum(
                        "sm,sm->m", np.conj(psi[kmq, b]), psi[k, a])
                    time_sum = np.sum(
                        -1j * projection
                        * np.exp(-1j * delta * time)[None, :],
                        axis=1,
                    )
                    out[:, q] += (
                        time_sum[:, None, None]
                        * fdiff
                        * np.outer(M, np.conj(M))[None, :, :]
                    )
    return out / np.sqrt(float(nk))


def _direct(psi, enk, occ, surface, z):
    """Tiny independent ordered-pair resolvent, including the static limit."""
    nk, nb, _, nmu = psi.shape
    out = np.zeros((len(z), nk, nmu, nmu), np.complex128)
    for q in range(nk):
        for k in range(nk):
            kmq = (k-q) % nk
            for a in range(nb):
                for b in range(nb):
                    de = enk[k,a]-enk[kmq,b]
                    fdiff = occ[k,a]-occ[kmq,b]
                    M = np.einsum("sm,sm->m",psi[k,a],np.conj(psi[kmq,b]))
                    for iz, point in enumerate(z):
                        if point == 0 and abs(de) <= 64*np.finfo(float).eps*max(1.,abs(enk[k,a]),abs(enk[kmq,b])):
                            weight = -0.5*(surface[k,a]+surface[kmq,b])
                        else:
                            weight = fdiff/(point+de)
                        out[iz,q] += weight*np.outer(M,np.conj(M))
    return out/np.sqrt(float(nk))


def main():
    rank = process_rank()
    if process_count() != 4:
        raise RuntimeError("fractional chi gate requires exactly four processes")
    mesh = resolve_mesh()
    if tuple(int(n) for n in mesh.devices.shape) != (2, 2):
        raise RuntimeError("fractional chi gate requires a 2x2 process mesh")

    rng = np.random.default_rng(20260814)
    nk, nb, ns, nmu = 3, 4, 2, 4
    psi = (
        rng.normal(size=(nk, nb, ns, nmu))
        + 1j * rng.normal(size=(nk, nb, ns, nmu))
    )
    enk = np.array([
        [-1.3, -0.4, 0.2, 1.1],
        [-1.1, -0.2, 0.5, 1.4],
        [-1.4, -0.1, 0.7, 1.2],
    ])
    occ = np.array([
        [1.0, 0.82, 0.10, 0.0],
        [1.0, 0.61, 0.25, 0.0],
        [1.0, 0.74, -0.01, 0.0],
    ])
    slices = BandSlices.from_band_edges(0, 0, 2, nb, nb)
    wfns = Wavefunctions(
        psi_mun=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_MUN_SPEC),
        psi_nmu=_put(psi, mesh, PSI_NMU_SPEC),
        enk=_put(enk, mesh, P(None, None)),
        occ=_put(occ, mesh, P(None, None)),
        slices=slices,
        layout="face",
    )
    time = np.array([0.13, 0.41, 0.79])
    z = np.array([0.32 + 0.18j, 0.77 + 0.24j])
    weights = np.array([
        [0.19, 0.31, 0.17],
        [0.23, 0.27, 0.11],
    ])
    values = compute_chi0_contour_fractional(
        wfns,
        time,
        weights,
        z,
        SimpleNamespace(nkx=3, nky=1, nkz=1),
        mesh,
    )
    got = np.stack([
        np.asarray(multihost_utils.process_allgather(value, tiled=True))
        for value in values
    ])
    want = _dense(psi, enk, occ, time, weights, z)
    error = float(np.max(np.abs(got - want)))
    scale = max(float(np.max(np.abs(want))), 1.0e-300)
    relative = error / scale
    if rank == 0:
        print(
            "[fractional-chi] world=4 mesh=2x2 "
            "max_abs={:.3e} max_rel={:.3e}".format(error, relative),
            flush=True,
        )
    if relative > 5.0e-12:
        raise AssertionError("fractional chi dense Kubo mismatch")

    # --- W1: the finite-q static divided-difference row -------------------
    from gw import efermi
    from gw.w_isdf import compute_chi0_static_fractional

    mu, width = 0.15, 0.08
    occ_mp1 = np.asarray(jax.device_get(
        efermi.mp1_occupations(enk, mu, width)))
    surface = np.asarray(jax.device_get(
        efermi.mp1_negative_derivative(enk, mu, width)))
    wfns_mp1 = Wavefunctions(
        psi_mun=wfns.psi_mun,
        psi_nmu=wfns.psi_nmu,
        enk=wfns.enk,
        occ=_put(occ_mp1, mesh, P(None, None)),
        slices=slices,
        layout="face",
    )
    state = SimpleNamespace(
        f_kn=occ_mp1, mu_ry=mu, smearing_family="mp1",
        smearing_width_ry=width)
    kminq = np.stack([[(k - q) % nk for k in range(nk)] for q in range(nk)])
    got_static = np.asarray(multihost_utils.process_allgather(
        compute_chi0_static_fractional(
            wfns_mp1, SimpleNamespace(nk_tot=nk, n_rmu=nmu), mesh,
            occupation_state=state, kminq_rows=kminq),
        tiled=True))

    want_static = _direct(psi,enk,occ_mp1,surface,np.asarray([0j]))[0]
    err_s = float(np.max(np.abs(got_static - want_static)))
    rel_s = err_s / max(float(np.max(np.abs(want_static))), 1.0e-300)

    # W1.a-2 consistency: the exact-static value stored in the shifted
    # origin slot differs from chi(i*varpi_1) by O((varpi_1/gap)^2) at
    # finite q.  Resolvent oracle, finite-q row q=1.
    varpi1 = 2.0e-5

    def _resolvent(z, q):
        return _direct(psi,enk,occ_mp1,surface,np.asarray([z]))[0,q]

    shift_rel = (
        np.max(np.abs(_resolvent(1j * varpi1, 1) - _resolvent(0.0, 1)))
        / np.max(np.abs(_resolvent(0.0, 1))))
    if rank == 0:
        print(
            "[fractional-chi] static finite-q max_rel={:.3e}; "
            "origin-shift consistency (q=1) rel={:.3e}".format(
                rel_s, shift_rel),
            flush=True,
        )
    if rel_s > 5.0e-12:
        raise AssertionError("finite-q static divided-difference mismatch")

    # Family controls include q=0 exact degeneracies and literal nonzero z.
    # The FD oracle uses cosh, independently of the sigmoid helper under test.
    from gw.w_isdf import compute_chi0_direct_fractional
    occ_fd = 0.5*(1.0-np.tanh((enk-mu)/(2*width)))
    surface_fd = 1.0/(4*width*np.cosh((enk-mu)/(2*width))**2)
    direct_z = np.asarray([0j,0.32+0.18j,0.77+0.24j])
    family_checks = {}
    face_wfns = Wavefunctions(
        psi_mun=_put(psi.transpose(0,2,3,1),mesh,PSI_MUN_SPEC),
        psi_nmu=_put(psi,mesh,PSI_NMU_SPEC),
        enk=wfns.enk,occ=wfns.occ,slices=slices,layout="face")
    meta_direct = SimpleNamespace(nk_tot=nk,n_rmu=nmu,nspinor=ns)
    for family, occupations, derivative in (("mp1",occ_mp1,surface),("fd",occ_fd,surface_fd)):
        family_state = SimpleNamespace(f_kn=occupations,mu_ry=mu,
            smearing_family=family,smearing_width_ry=width)
        expected = _direct(psi,enk,occupations,derivative,direct_z)
        for layout, carrier in (("legacy",wfns),("face",face_wfns)):
            actual = np.asarray(multihost_utils.process_allgather(
                compute_chi0_direct_fractional(carrier,direct_z,meta_direct,mesh,
                    occupation_state=family_state,kminq_rows=kminq),tiled=True))
            errors = [float(np.max(np.abs(actual[i]-expected[i]))
                      /np.max(np.abs(expected[i]))) for i in range(len(direct_z))]
            assert max(errors) < 5e-12,(family,layout,errors)
            family_checks[family+"_"+layout] = errors
    actual_derivative = np.asarray(efermi.fd_negative_derivative(enk,mu,width))
    fd_error = float(np.max(np.abs(actual_derivative-surface_fd))/np.max(surface_fd))
    at_mu = float(efermi.fd_negative_derivative(np.asarray(mu),mu,width))
    assert fd_error < 5e-14 and abs(at_mu-1/(4*width)) < 1e-14
    try:
        compute_chi0_direct_fractional(wfns,direct_z,meta_direct,mesh,
            occupation_state=SimpleNamespace(smearing_family="unsupported"),kminq_rows=kminq)
    except ValueError as exc:
        assert "GATE static_fractional_needs_mp1" in str(exc)
        assert "unsupported" in str(exc) and "'fd'" in str(exc)
    else:
        raise AssertionError("unsupported occupation family accepted")
    if rank == 0:
        print("[fractional-chi families] "+json.dumps(family_checks),flush=True)

    # --- Integer insulator: minimax orientation completion -----------------
    # Flat unit gaps make the one-node (tau=0, alpha=1) inverse exact.  The
    # oracle below literally sums every ordered Adler-Wiser pair, including
    # both v->c and c->v.  Complex random spinors deliberately break the
    # special real/TR condition under which replacing the reverse
    # orientation by a second copy of the forward orientation would work.
    nv = 2
    enk_i = np.concatenate((
        -0.5 * np.ones((nk, nv)),
        +0.5 * np.ones((nk, nb - nv)),
    ), axis=1)
    occ_i = np.concatenate((
        np.ones((nk, nv)), np.zeros((nk, nb - nv))), axis=1)
    wfns_i = Wavefunctions(
        psi_mun=wfns.psi_mun,
        psi_nmu=wfns.psi_nmu,
        enk=_put(enk_i, mesh, P(None, None)),
        occ=_put(occ_i, mesh, P(None, None)),
        slices=slices,
        layout="face",
    )
    got_i = np.asarray(multihost_utils.process_allgather(
        compute_chi0(
            wfns_i,
            SimpleNamespace(tau=np.asarray([0.0]), alpha=np.asarray([1.0])),
            SimpleNamespace(nkx=nk, nky=1, nkz=1, nk_tot=nk),
            mesh,
        ),
        tiled=True,
    ))

    want_i = np.zeros((nk, nmu, nmu), np.complex128)
    for q in range(nk):
        for k in range(nk):
            kmq = (k - q) % nk
            for a in range(nb):
                for b in range(nb):
                    de = enk_i[k, a] - enk_i[kmq, b]
                    fdiff = occ_i[k, a] - occ_i[kmq, b]
                    if fdiff == 0.0:
                        continue
                    M = np.einsum(
                        "sm,sm->m", psi[k, a], np.conj(psi[kmq, b]))
                    want_i[q] += (fdiff / de) * np.outer(M, np.conj(M))
    want_i /= np.sqrt(float(nk))

    err_i = float(np.max(np.abs(got_i - want_i)))
    rel_i = err_i / max(float(np.max(np.abs(want_i))), 1.0e-300)
    neg_q = np.asarray([(-q) % nk for q in range(nk)])
    recip_i = (
        float(np.max(np.abs(got_i - np.conj(got_i[neg_q]))))
        / max(float(np.max(np.abs(got_i))), 1.0e-300)
    )
    if rank == 0:
        print(
            "[integer-static-chi] direct_AW_max_abs={:.3e} "
            "direct_AW_max_rel={:.3e} q_recip={:.3e}".format(
                err_i, rel_i, recip_i),
            flush=True,
        )
    if rel_i > 5.0e-12:
        raise AssertionError("integer static chi direct Adler-Wiser mismatch")
    if recip_i > 5.0e-12:
        raise AssertionError("integer static chi q reciprocity mismatch")
    if rank == 0 and len(sys.argv) > 1:
        (Path(sys.argv[1])/"receipt.json").write_text(json.dumps(dict(
            status="PASS",job=os.getenv("SLURM_JOB_ID"),step=os.getenv("SLURM_STEP_ID"),
            contour_relative=relative,static_mp1_relative=rel_s,
            direct_family_relative=family_checks,fd_derivative_relative=fd_error,
            fd_derivative_at_mu=at_mu,unsupported_family="REFUSED",
            integer_relative=rel_i),indent=2))
    multihost_utils.sync_global_devices("fractional_chi_gate_pass")


if __name__ == "__main__":
    try:
        main()
    finally:
        finalize_process()
