"""P=4 dense Kubo gate for the finite-occupation chi0 contour kernel."""

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import process_count, process_rank, resolve_mesh
from gw.w_isdf import (
    compute_chi0_contour_fractional,
    intraband_chi1,
    intraband_pair_block,
)
from gw.wavefunction_bundle import (
    BandSlices,
    PSI_XN_SPEC,
    PSI_XR_SPEC,
    PSI_YN_SPEC,
    PSI_YR_SPEC,
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


def _dense_crossing_block(psi, enk, occ, kminq_rows, q_row, z):
    """Independent exact-K oracle for full/selected/complement chi rows."""
    nk, nb, _, nmu = psi.shape
    kmq = np.asarray(kminq_rows[q_row], np.int64)
    support = (
        np.any(occ != 0.0, axis=0) & np.any(occ != 1.0, axis=0)
    )
    same_delta = enk[kmq, :] - enk
    lambda_q = (
        float(np.max(np.abs(same_delta[:, support])))
        if np.any(support) else 0.0
    )
    full = np.zeros((nmu, nmu), np.complex128)
    selected = np.zeros_like(full)
    complement = np.zeros_like(full)
    n_selected = 0
    n_cross_band = 0
    for k in range(nk):
        for a in range(nb):
            for b in range(nb):
                delta = enk[kmq[k], b] - enk[k, a]
                delta_f = occ[k, a] - occ[kmq[k], b]
                vertex = np.einsum(
                    "sm,sm->m", np.conj(psi[k, a]), psi[kmq[k], b])
                coefficient = (
                    delta_f * (-2.0 * delta / (delta * delta - z * z))
                    / (2.0 * np.sqrt(float(nk)))
                )
                term = coefficient * np.outer(vertex, np.conj(vertex))
                full += term
                in_block = delta_f != 0.0 and abs(delta) <= lambda_q
                if in_block:
                    selected += term
                    n_selected += 1
                    n_cross_band += int(a != b)
                else:
                    complement += term
    return full, selected, complement, n_selected, n_cross_band


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
        psi_xn=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_XN_SPEC),
        psi_xr=_put(psi, mesh, PSI_XR_SPEC),
        psi_yr=_put(psi, mesh, PSI_YR_SPEC),
        psi_yn=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_YN_SPEC),
        enk=_put(enk, mesh, P(None, None)),
        occ=_put(occ, mesh, P(None, None)),
        slices=slices,
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
        psi_xn=wfns.psi_xn, psi_xr=wfns.psi_xr, psi_yr=wfns.psi_yr,
        psi_yn=wfns.psi_yn, enk=wfns.enk,
        occ=_put(occ_mp1, mesh, P(None, None)), slices=slices)
    state = SimpleNamespace(
        f_kn=occ_mp1, mu_ry=mu, smearing_family="mp1",
        smearing_width_ry=width)
    kminq = np.stack([[(k - q) % nk for k in range(nk)] for q in range(nk)])
    got_static = np.asarray(multihost_utils.process_allgather(
        compute_chi0_static_fractional(
            wfns_mp1, SimpleNamespace(nk_tot=nk, n_rmu=nmu), mesh,
            occupation_state=state, kminq_rows=kminq),
        tiled=True))

    want_static = np.zeros((nk, nmu, nmu), np.complex128)
    for q in range(nk):
        for k in range(nk):
            kmq = (k - q) % nk
            for a in range(nb):
                for b in range(nb):
                    de = enk[k, a] - enk[kmq, b]
                    scale = max(1.0, abs(enk[k, a]), abs(enk[kmq, b]))
                    if abs(de) > 64.0 * np.finfo(np.float64).eps * scale:
                        divided = (occ_mp1[k, a] - occ_mp1[kmq, b]) / de
                    else:
                        divided = -0.5 * (surface[k, a] + surface[kmq, b])
                    M = np.einsum(
                        "sm,sm->m", psi[k, a], np.conj(psi[kmq, b]))
                    want_static[q] += divided * np.outer(M, np.conj(M))
    want_static /= np.sqrt(float(nk))
    err_s = float(np.max(np.abs(got_static - want_static)))
    rel_s = err_s / max(float(np.max(np.abs(want_static))), 1.0e-300)

    # W1.a-2 consistency: the exact-static value stored in the shifted
    # origin slot differs from chi(i*varpi_1) by O((varpi_1/gap)^2) at
    # finite q.  Resolvent oracle, finite-q row q=1.
    varpi1 = 2.0e-5

    def _resolvent(z, q):
        out = np.zeros((nmu, nmu), np.complex128)
        for k in range(nk):
            kmq = (k - q) % nk
            for a in range(nb):
                for b in range(nb):
                    delta = enk[kmq, b] - enk[k, a]
                    fdiff = occ_mp1[k, a] - occ_mp1[kmq, b]
                    if fdiff == 0.0 and delta == 0.0:
                        continue
                    M = np.einsum(
                        "sm,sm->m", psi[k, a], np.conj(psi[kmq, b]))
                    out += (fdiff / (z - delta)) * np.outer(M, np.conj(M))
        return out / np.sqrt(float(nk))

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

    # --- WP1: exact crossing block, all wedge rows and three z ----------
    # One near-static point, one strip point and one far-line point.  The
    # oracle independently repeats the literal S(q) definition; it does not
    # call either production selector/evaluator.
    z_cross = np.array([2.0e-5j, 0.77 + 0.24j, 2.0j])
    worst_selected = 0.0
    worst_complement = 0.0
    blocks = []
    for q_row in range(nk):
        block = intraband_pair_block(
            wfns_mp1,
            SimpleNamespace(nk_tot=nk, n_rmu=nmu),
            state,
            kminq,
            q_row,
        )
        blocks.append(block)
        if block[2][0].sharding.spec != P(None, "x"):
            raise AssertionError("crossing P_x lost its pair/x sharding")
        if block[2][1].sharding.spec != P(None, "y"):
            raise AssertionError("crossing P_y lost its pair/y sharding")
        if q_row == 0 and int(block[0].size) != 0:
            raise AssertionError("Gamma crossing block is not empty")
        for z_value in z_cross:
            chi1_row = intraband_chi1(block, z_value)
            if chi1_row.sharding.spec != P("x", "y"):
                raise AssertionError("chi1 row lost its x/y sharding")
            got_block = np.asarray(multihost_utils.process_allgather(
                chi1_row, tiled=True))
            (want_full, want_block, want_complement,
             n_selected, n_cross_band) = _dense_crossing_block(
                psi, enk, occ_mp1, kminq, q_row, z_value)
            scale_b = max(float(np.max(np.abs(want_block))), 1.0e-300)
            rel_b = float(np.max(np.abs(got_block - want_block))) / scale_b
            scale_c = max(
                float(np.max(np.abs(want_complement))), 1.0e-300)
            rel_c = float(np.max(np.abs(
                (want_full - got_block) - want_complement))) / scale_c
            worst_selected = max(worst_selected, rel_b)
            worst_complement = max(worst_complement, rel_c)
            if q_row == 0:
                if n_selected != 0 or np.any(got_block != 0.0):
                    raise AssertionError(
                        "Gamma S(q) or chi1 is not identically empty")
            elif n_selected == 0 or n_cross_band == 0:
                raise AssertionError(
                    "finite-q fixture did not exercise the selected "
                    "near-degenerate cross-band channel")
    if rank == 0:
        print(
            "[fractional-chi] crossing block all-q/three-z "
            "selected max_rel={:.3e} complement max_rel={:.3e}".format(
                worst_selected, worst_complement),
            flush=True,
        )
    if worst_selected > 1.0e-12 or worst_complement > 1.0e-12:
        raise AssertionError("crossing-block dense/complement mismatch")
    chi1_wedge_dist = jnp.stack([
        intraband_chi1(block, z_cross[1]) for block in blocks
    ])
    if chi1_wedge_dist.sharding.spec != P(None, "x", "y"):
        raise AssertionError("stacked chi1 wedge lost its q/x/y sharding")
    chi1_wedge = np.asarray(multihost_utils.process_allgather(
        chi1_wedge_dist, tiled=True))
    if chi1_wedge.shape != (nk, nmu, nmu):
        raise AssertionError(
            "stacked crossing wedge has wrong global shape: "
            + str(chi1_wedge.shape))
    multihost_utils.sync_global_devices("fractional_chi_gate_pass")


if __name__ == "__main__":
    main()
    finalize_process()
