"""The G-split matrix-element sweep against an independent per-k reference.

``common.mtxel_sweep.sweep_matrix_elements`` moves ψ (and, for an FFT
operator, O ψ) from its band layout to a G-split layout with one
all-to-all, contracts each rank's G slab, and reduce-scatters the
``(nb, nb)`` partial into the ``P(None, 'x', 'y')`` block.  Diagonal
operators act on the slab; V_NL and its K-derivative are separable, their
slab projections psummed.  Every arm here is compared on a 2x2 mesh with a
NumPy matrix that shares no layout, collective or contraction code with
the sweep: host box scatter and ``numpy.fft`` for the local potential,
host ``|k+G|²`` and ``2(k+G)``, and host ``c† E c`` / ``dc† E c + c† E dc``
on projectors built for the WHOLE G list (the sweep builds them per slab).

The fixture is built so every padded path runs: 10 bands on a mesh whose
band carrier is 12, a G table of 41 (not divisible by 4, so the all-to-all
carrier is 44) with 4 pad columns, and 6 k-points tiled as 1, 2 and 3.
Parity target 1e-13 relative to the largest element (the sweep only
reassociates the G sum).  The red twins — an unconjugated bra and a
transposed block — must miss the same reference by orders of magnitude,
which is what makes a pass evidence.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp                                        # noqa: E402
from jax.sharding import Mesh, NamedSharding                   # noqa: E402

from common import mtxel_sweep                                 # noqa: E402
from common.mtxel_sweep import (                               # noqa: E402
    SweepGeometry, dipole_operator, four_current_potential_operator,
    kinetic_operator, local_potential_operator, sum_operators,
    sweep_matrix_elements, vnl_operator)
from common.wfn_layout import band_sphere_spec                 # noqa: E402

RTOL = 1.0e-13
NK, NB, NGK, NGKMAX = 6, 10, 37, 41
GRID = (6, 6, 8)
VOLUME = 31.0
NATOMS, NBETA = 2, 2
TOTAL_R = NATOMS * NBETA


def _mesh() -> Mesh:
    devices = jax.devices()
    side = 2 if len(devices) >= 4 else 1
    return Mesh(np.asarray(devices[:side * side]).reshape(side, side),
                ("x", "y"))


def _fixture(ns, seed=20260924):
    rng = np.random.default_rng(seed + ns)
    nx, ny, nz = GRID
    gv = np.zeros((NK, NGKMAX, 3), dtype=np.int32)
    gv[:, NGK:] = (nx // 2, ny // 2, nz // 2)          # the pad sentinel
    # The per-k sphere index (common.gvec_fft_box.build_sphere_box_index):
    # the flat C-order box cell of each slot, n_rtot + g on a pad slot.
    n_rtot = nx * ny * nz
    bidx = np.tile(n_rtot + np.arange(NGKMAX, dtype=np.int32), (NK, 1))
    for ik in range(NK):
        cells = rng.choice(n_rtot, size=NGK, replace=False)
        xyz = np.column_stack(np.unravel_index(cells, GRID))
        gv[ik, :NGK] = xyz
        bidx[ik, :NGK] = cells
    gmask = np.zeros((NK, NGKMAX))
    gmask[:, :NGK] = 1.0
    psi = (rng.standard_normal((NK, NB, ns, NGKMAX))
           + 1j * rng.standard_normal((NK, NB, ns, NGKMAX)))
    # Pad columns are NOT zeroed: the sweep's own mask must remove them.
    kvecs = rng.standard_normal((NK, 3)) * 0.25
    return psi.astype(np.complex128), gv, gmask, bidx, kvecs


def _vnl_setup(ns, seed=7):
    """Synthetic ``VNLSetup``: one l = 0 channel, two betas, two sites."""
    from psp import vnl_ops

    rng = np.random.default_rng(seed)
    n_q, dq = 96, 0.05
    tau = rng.standard_normal((NATOMS, 3))
    ch = vnl_ops.ChannelMeta(
        l=0, nbeta=NBETA, msize=1, R=NBETA, tau=tau,
        E=np.zeros((2, 2, NBETA, NBETA)), beta_table_start=0,
        natoms=NATOMS)
    q = np.arange(n_q) * dq
    G_table = np.stack([np.exp(-0.5 * (q - 0.4 * (b + 1)) ** 2)
                        for b in range(NBETA)])
    Gp_table = np.stack([-(q - 0.4 * (b + 1)) * G_table[b]
                         for b in range(NBETA)])
    E = (rng.standard_normal((ns, ns, TOTAL_R, TOTAL_R))
         + 1j * rng.standard_normal((ns, ns, TOTAL_R, TOTAL_R)))
    return vnl_ops.VNLSetup(
        channels=[ch], dq=dq, n_q=n_q, q_max=n_q * dq,
        G_table=jnp.asarray(G_table), Gp_table=jnp.asarray(Gp_table),
        prefactor=1.3, B=np.eye(3) * 1.17, cell_volume=VOLUME,
        total_R=TOTAL_R, nspinor=ns,
        E_super=jnp.asarray(E, dtype=jnp.complex128), l_max=0,
        row_beta_idx=jnp.asarray(np.tile(np.arange(NBETA), NATOMS),
                                 dtype=jnp.int32),
        row_l=jnp.zeros(TOTAL_R, dtype=jnp.int32),
        row_m=jnp.zeros(TOTAL_R, dtype=jnp.int32),
        row_tau=jnp.asarray(np.repeat(tau, NBETA, axis=0)))


# ---------------------------------------------------------------------------
# The independent reference: one k, whole G, NumPy contractions
# ---------------------------------------------------------------------------

def _ref_local(psi_m, gv, mask, V_r):
    from psp.get_DFT_mtxels import local_potential_scalars
    sc = local_potential_scalars(VOLUME, int(np.prod(GRID)))
    nb, ns, _ = psi_m.shape
    box = np.zeros((nb, ns, *GRID), dtype=np.complex128)
    phys = mask > 0
    g = gv[phys]
    box[:, :, g[:, 0], g[:, 1], g[:, 2]] = psi_m[:, :, phys]
    scale, dv, fn, post = (float(sc.scale), float(sc.deltaV),
                           float(sc.fft_norm), float(sc.post))
    psi_r = np.fft.ifftn(box, axes=(-3, -2, -1), norm="ortho") * scale
    phi = np.fft.fftn(psi_r * V_r, axes=(-3, -2, -1), norm="ortho") \
        * (dv * fn)
    ket = phi[..., gv[:, 0], gv[:, 1], gv[:, 2]] * mask
    return np.einsum("msg,nsg->mn", np.conj(psi_m), ket) * post


def _ref_kinetic(psi_m, gv, kvec, bdot):
    K = gv + kvec[None]
    T = np.einsum("gi,ij,gj->g", K, bdot, K)
    return np.einsum("msg,g,nsg->mn", np.conj(psi_m), T, psi_m)


def _ref_projections(psi_m, gv, kvec, setup, dZ=False):
    from psp import vnl_ops
    kd = vnl_ops.build_vnl_kdata_traced(
        jnp.asarray(kvec), jnp.asarray(gv), setup, compute_dZ=dZ)
    ns_e = int(setup.E_super.shape[0])
    Z = np.asarray(kd.Z)
    c = np.einsum("Rg,nsg->Rsn", np.conj(Z), psi_m[:, :ns_e])
    if not dZ:
        return c, None
    dc = np.einsum("aRg,nsg->aRsn", np.conj(np.asarray(kd.dZ)),
                   psi_m[:, :ns_e])
    return c, dc


def _couple(E, c):
    return np.einsum("stRQ,Qtn->Rsn", E, c)


def _ref_vnl(psi_m, gv, kvec, setup):
    E = np.asarray(setup.E_super)
    c, _ = _ref_projections(psi_m, gv, kvec, setup)
    return np.einsum("Rsm,Rsn->mn", np.conj(c), _couple(E, c))


def _ref_dipole(psi_m, gv, kvec, B, setup, sign):
    Kc = (gv + kvec[None]) @ B
    p = 2.0 * np.einsum("msg,gj,nsg->jmn", np.conj(psi_m), Kc, psi_m)
    if setup is None:
        return p
    E = np.asarray(setup.E_super)
    c, dc = _ref_projections(psi_m, gv, kvec, setup, dZ=True)
    D = _couple(E, c)
    dD = np.stack([_couple(E, dc[j]) for j in range(3)])
    v_nl = (np.einsum("jRsm,Rsn->jmn", np.conj(dc), D)
            + np.einsum("Rsm,jRsn->jmn", np.conj(c), dD))
    return p + sign * v_nl


def _reference(case, psi, gv, gmask, bidx, kvecs, extra):
    out = []
    for ik in range(NK):
        psi_m = psi[ik] * gmask[ik][None, None, :]
        args = (psi_m, gv[ik].astype(float), kvecs[ik])
        if case == "vh":
            out.append(_ref_local(psi_m, gv[ik], gmask[ik], extra["V_r"]))
        elif case == "kinetic":
            out.append(_ref_kinetic(*args, extra["bdot"]))
        elif case == "vnl":
            out.append(_ref_vnl(psi_m, gv[ik], kvecs[ik], extra["setup"]))
        elif case == "kin_ion":
            out.append(_ref_kinetic(*args, extra["bdot"])
                       + _ref_local(psi_m, gv[ik], gmask[ik], extra["V_r"])
                       + _ref_vnl(psi_m, gv[ik], kvecs[ik], extra["setup"]))
        elif case in ("dipole", "dipole_legacy", "dipole_p"):
            out.append(_ref_dipole(
                psi_m, gv[ik].astype(float), kvecs[ik], extra["B"],
                None if case == "dipole_p" else extra["setup"],
                -1.0 if case == "dipole_legacy" else 1.0))
        elif case == "dirac_current":
            from common.bispinor_init import apply_dirac_velocity_to_ket
            ket = np.asarray(apply_dirac_velocity_to_ket(jnp.asarray(psi_m)))
            out.append(np.einsum("msg,cnsg->cmn", np.conj(psi_m), ket))
        elif case in ("four_current", "uniform_current"):
            # The band-layout operator's OWN ket at one k on one device:
            # this arm isolates the sweep's layout and collectives for a
            # component-carrying FFT ket, not the operator physics.
            op = extra["op"]
            ket = np.asarray(op.apply(
                jnp.asarray(psi[ik])[None], jnp.asarray(gv[ik]),
                jnp.asarray(gmask[ik]), jnp.asarray(bidx[ik])[None],
                jnp.asarray(kvecs[ik]), *op.consts))[0]
            out.append(np.einsum("msg,nsgc->cmn", np.conj(psi_m), ket)
                       * op.post)
        else:
            raise ValueError(case)
    return np.stack(out)


def _operator(case, geom, extra):
    if case == "vh":
        return local_potential_operator(geom, extra["V_r"])
    if case == "kinetic":
        return kinetic_operator(geom, extra["bdot"])
    if case == "vnl":
        return vnl_operator(geom, extra["setup"])
    if case == "kin_ion":
        return sum_operators(kinetic_operator(geom, extra["bdot"]),
                             local_potential_operator(geom, extra["V_r"]),
                             vnl_operator(geom, extra["setup"]))
    if case in ("dipole", "dipole_legacy", "dipole_p"):
        return dipole_operator(
            geom, bvec=extra["B"], blat=1.0,
            vnl_setup=None if case == "dipole_p" else extra["setup"],
            vnl_velocity_sign=-1.0 if case == "dipole_legacy" else 1.0)
    if case == "dirac_current":
        return mtxel_sweep.dirac_current_operator(geom)
    if case in ("four_current", "uniform_current"):
        return extra["op"]
    raise ValueError(case)


def _run(case, ns, k_tile=None, use_scan=True, monkeypatch=None):
    mesh = _mesh()
    psi, gv, gmask, bidx, kvecs = _fixture(ns)
    rng = np.random.default_rng(3)
    extra = dict(V_r=rng.standard_normal(GRID),
                 bdot=np.eye(3) * 1.31 + 0.07,
                 B=np.eye(3) * 1.17 + 0.05 * rng.standard_normal((3, 3)))
    with mesh:
        geom = SweepGeometry(mesh=mesh, fft_grid=GRID, ngkmax=NGKMAX, nb=NB,
                             ns=ns, nk=NK, cell_volume=VOLUME)
        if case in ("vnl", "kin_ion", "dipole", "dipole_legacy"):
            extra["setup"] = _vnl_setup(ns if ns == 2 else 2)
        if case == "uniform_current":
            # The uniform-gauge kernel refuses a setup without coupled-row
            # provenance; reuse the gauge-vertex test's canonical builder.
            from tests.test_dft_gauge_vertices import _setup as gauge_setup
            setup = gauge_setup(curved=False, natoms=2)
            extra["op"] = mtxel_sweep.uniform_gauge_operator(
                geom, bvec=np.asarray(setup.B), blat=1.0, vnl_setup=setup,
                include_contact=False)
        if case == "four_current":
            extra["op"] = four_current_potential_operator(
                geom, extra["V_r"], rng.standard_normal((3, *GRID)),
                charge_nspinor=2)
        if k_tile is not None:
            real = mtxel_sweep.plan_sweep
            monkeypatch.setattr(
                mtxel_sweep, "plan_sweep",
                lambda g, o: real(g, o)._replace(k_tile=k_tile))
        psi_pad = np.pad(psi, ((0, 0), (0, geom.nb - NB), (0, 0), (0, 0)))
        psi_j = jax.make_array_from_callback(
            psi_pad.shape, NamedSharding(mesh, band_sphere_spec()),
            lambda idx: psi_pad[idx])
        H = sweep_matrix_elements(
            psi_j, geom=geom, operator=_operator(case, geom, extra),
            gvecs=gv, gmask=gmask, box_index=bidx, kvecs=kvecs,
            use_scan=use_scan)
        got = np.asarray(H)
        spec = H.sharding.spec
    ref = _reference(case, psi, gv, gmask, bidx, kvecs, extra)
    return got, ref, spec, geom


def _rel(got, ref):
    return float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))


CASES = [("vh", 2), ("kinetic", 2), ("vnl", 2), ("kin_ion", 2),
         ("dipole", 2), ("dipole_legacy", 2), ("dipole_p", 2),
         ("vnl", 4), ("four_current", 4), ("uniform_current", 4),
         ("dirac_current", 4)]


@pytest.mark.mesh(4)
@pytest.mark.parametrize("case,ns", CASES)
def test_sweep_matches_independent_reference(case, ns):
    got, ref, spec, geom = _run(case, ns)
    assert tuple(spec)[-2:] == ("x", "y"), spec
    assert geom.nb == 12 and NGKMAX % 4 != 0      # both pads really run
    lead = got.shape[:-2]
    # Pad rows and columns of the block are exact zeros.
    assert not np.any(got[..., NB:, :]) and not np.any(got[..., :, NB:])
    logical = got[..., :NB, :NB]
    if ref.ndim == 4:                               # (nk, c, nb, nb)
        assert lead == (NK, ref.shape[1])
    rel = _rel(logical, ref)
    assert rel <= RTOL, f"{case} ns={ns}: rel {rel:.3e}"
    # RED TWINS on the same reference: an unconjugated bra, and the block
    # transposed (m and n, i.e. 'x' and 'y', swapped).  Each must miss by
    # far more than the tolerance, or the pass above proves nothing.
    assert _rel(np.conj(logical), ref) > 1e-3
    assert _rel(np.swapaxes(logical, -1, -2), ref) > 1e-3


@pytest.mark.mesh(4)
@pytest.mark.parametrize("k_tile", [1, 2, 3])
@pytest.mark.parametrize("case", ["vh", "dipole"])
def test_k_tiles_agree(case, k_tile, monkeypatch):
    """Every k tile computes the same matrix: tiling only batches."""
    got, ref, _, _ = _run(case, 2, k_tile=k_tile, monkeypatch=monkeypatch)
    assert _rel(got[..., :NB, :NB], ref) <= RTOL


@pytest.mark.mesh(4)
def test_python_loop_matches_scan():
    got_scan, ref, _, _ = _run("kin_ion", 2)
    got_loop, _, _, _ = _run("kin_ion", 2, use_scan=False)
    assert _rel(got_loop[..., :NB, :NB], ref) <= RTOL
    assert _rel(got_scan, got_loop) <= RTOL


@pytest.mark.mesh(4)
def test_red_twin_conjugated_bra_sweep_fails(monkeypatch):
    """A sweep with the bra conjugation removed must fail the gate.

    Patches the one contraction to drop ``conj`` and runs the real sweep:
    the reference comparison that passes above must then report a large
    error, so it is sensitive to the defect it guards.
    """
    from common.wfn_transforms import _KERNEL_CACHE

    def drop_sweeps():
        for key in [k for k in _KERNEL_CACHE
                    if k[0] == "sweep_matrix_elements"]:
            del _KERNEL_CACHE[key]

    drop_sweeps()                      # force a trace with the defect in
    monkeypatch.setattr(mtxel_sweep.jnp, "conj", lambda x: x)
    try:
        got, _, _, _ = _run("vh", 2)
    finally:
        monkeypatch.undo()
        drop_sweeps()                  # never leave the defect cached
    ref = _run("vh", 2)[1]
    assert _rel(got[..., :NB, :NB], ref) > 1e-3


@pytest.mark.mesh(4)
def test_operator_tuple_is_bitwise_per_operator():
    """One sweep, several operators (the kin_ion + velocity fusion, B13).

    The tuple shares ψ's all-to-all and G slabs; every operator keeps its
    own ket, contraction and reduction, so each block must be BIT-identical
    to that operator's own sweep — not merely within tolerance.
    """
    mesh = _mesh()
    psi, gv, gmask, bidx, kvecs = _fixture(2)
    rng = np.random.default_rng(3)
    extra = dict(V_r=rng.standard_normal(GRID),
                 bdot=np.eye(3) * 1.31 + 0.07,
                 B=np.eye(3) * 1.17 + 0.05 * rng.standard_normal((3, 3)),
                 setup=_vnl_setup(2))
    with mesh:
        geom = SweepGeometry(mesh=mesh, fft_grid=GRID, ngkmax=NGKMAX, nb=NB,
                             ns=2, nk=NK, cell_volume=VOLUME)
        ops = (_operator("kin_ion", geom, extra),
               _operator("dipole", geom, extra))
        psi_pad = np.pad(psi, ((0, 0), (0, geom.nb - NB), (0, 0), (0, 0)))
        psi_j = jax.make_array_from_callback(
            psi_pad.shape, NamedSharding(mesh, band_sphere_spec()),
            lambda idx: psi_pad[idx])
        kw = dict(geom=geom, gvecs=gv, gmask=gmask, box_index=bidx,
                  kvecs=kvecs)
        both = sweep_matrix_elements(psi_j, operator=ops, **kw)
        alone = [sweep_matrix_elements(psi_j, operator=o, **kw) for o in ops]
        assert isinstance(both, tuple) and len(both) == 2
        for b, a in zip(both, alone):
            assert b.sharding.spec == a.sharding.spec
            assert np.array_equal(np.asarray(b), np.asarray(a))
