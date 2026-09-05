"""The nonlocal term enters the velocity as ``p + dV_NL/dK``, and only so.

HISTORY, IN THREE LINES.  ``common.mtxel_sweep.dipole_operator`` once
assembled ``p`` MINUS the nonlocal commutator term.  Measured against
BerkeleyGW's q -> 0 head on silicon (the table in that function's
docstring), the minus arm is 31 % high in eps00(0) while the plus arm
agrees to 1e-5; the owner ruled for plus on 2026-08-09, and on
2026-09-05 the ``vnl_velocity_sign`` knob that kept the minus arm
reachable, the ``--vnl-mode numeric`` finite-difference cross-check and
their CLI/deck plumbing were retired.  There is one velocity now.

WHAT THIS FILE GATES.  (1) The operator IS ``p + dV_NL/dK``, bit for
bit against the two kernels it is built from.  (2) The nonlocal term is
live -- the operator differs from bare ``p`` by exactly that term, so a
tree in which the projector term stopped reaching the assembly fails
here rather than in an absorption spectrum.  (3) Every ``dipole.h5`` is
stamped ``analytic`` / ``+1``, and the provenance check refuses a file
stamped with the retired arm.  (4) The retired deck key refuses by name
instead of being parsed, stored and never read -- this project's named
failure mode.

Everything here is synthetic and device-free: the V_NL setup is a
handful of random projector rows, because the quantity under test is a
sign and not a pseudopotential.  The per-(psi, G-list) projector
contraction is NOT touched -- it reproduces Quantum ESPRESSO to ~10
significant figures and the standing project rule protects it.
``tests/test_bse_oscillator_strengths.py`` imports the fixtures below.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
import pytest

from jax.sharding import Mesh

from common.mtxel_sweep import (VNL_VELOCITY_SIGN, SweepGeometry,
                                _operator_key, dipole_operator)

NK, NB, NS, NGK, NGKMAX = 2, 4, 1, 10, 12
GRID = (4, 4, 4)
NATOMS, NBETA, MSIZE = 2, 2, 1          # one l = 0 channel on two sites
TOTAL_R = NATOMS * NBETA * MSIZE


def _mesh():
    """A 1x1 ('x', 'y') mesh, so ``band_sphere_spec`` divides by one.

    The sign is not a distributed quantity and this file does not
    pretend to gate the sharding; ``tests/multi_device/mtxel_sweep_gate``
    owns that.  One device keeps the file runnable wherever pytest runs.
    """
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))


def _vnl_setup(seed=11):
    """A synthetic ``VNLSetup`` with Z, dZ and E_super all nonzero.

    One l = 0 channel, two radial betas, two sites -- enough that the
    ``compute_dZ=True`` branch runs its real per-channel JVP and returns
    a full-rank dZ, and no more, because the quantity under test is a
    SIGN and not a pseudopotential.  The row metadata is laid out
    atom-major to match the order ``_build_vnl_kdata_core`` reshapes its
    dZ blocks into.

    E_super is deliberately random and not the zeros
    ``tests/test_psp_padded_gvectors`` uses: a null E_super makes
    ``v_nl`` identically zero and this whole file becomes a tautology
    wearing a measurement's clothes.
    """
    from psp import vnl_ops

    rng = np.random.default_rng(seed)
    n_q = 96
    dq = 0.02
    tau = rng.standard_normal((NATOMS, 3))
    ch = vnl_ops.ChannelMeta(
        l=0, nbeta=NBETA, msize=MSIZE, R=NBETA * MSIZE, tau=tau,
        E=np.zeros((2, 2, NBETA * MSIZE, NBETA * MSIZE)),
        beta_table_start=0, natoms=NATOMS)
    # Smooth radial tables and their exact derivative, so the interpolant
    # and the JVP see a form factor rather than noise.
    qgrid = np.arange(n_q) * dq
    G_table = np.stack([np.exp(-0.5 * (qgrid - 0.3 * (b + 1)) ** 2)
                        for b in range(NBETA)])
    Gp_table = np.stack([-(qgrid - 0.3 * (b + 1)) * G_table[b]
                         for b in range(NBETA)])
    row_beta = np.tile(np.arange(NBETA, dtype=np.int32), NATOMS)
    row_tau = np.repeat(tau, NBETA * MSIZE, axis=0)
    return vnl_ops.VNLSetup(
        channels=[ch], dq=dq, n_q=n_q, q_max=n_q * dq,
        G_table=jnp.asarray(G_table, dtype=jnp.float64),
        Gp_table=jnp.asarray(Gp_table, dtype=jnp.float64),
        prefactor=1.3, B=np.eye(3) * 1.17, cell_volume=1.0,
        total_R=TOTAL_R, nspinor=NS,
        E_super=jnp.asarray(
            rng.standard_normal((NS, NS, TOTAL_R, TOTAL_R))
            + 1j * rng.standard_normal((NS, NS, TOTAL_R, TOTAL_R)),
            dtype=jnp.complex128),
        l_max=0,
        row_beta_idx=jnp.asarray(row_beta, dtype=jnp.int32),
        row_l=jnp.zeros(TOTAL_R, dtype=jnp.int32),
        row_m=jnp.zeros(TOTAL_R, dtype=jnp.int32),
        row_tau=jnp.asarray(row_tau, dtype=jnp.float64),
    )


def _fixture(seed=3):
    """psi, the D10 G table and its pad mask, the k list, and B."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = GRID
    gv = np.zeros((NK, NGKMAX, 3), dtype=np.int32)
    bidx = np.zeros((NK,) + GRID, dtype=np.int32)
    for ik in range(NK):
        cells = rng.choice(nx * ny * nz, size=NGK, replace=False)
        for i, c in enumerate(cells):
            gv[ik, i] = [c // (ny * nz), (c // nz) % ny, c % nz]
            bidx[ik, gv[ik, i, 0], gv[ik, i, 1], gv[ik, i, 2]] = i
    gmask = np.zeros((NK, NGKMAX), dtype=np.float64)
    gmask[:, :NGK] = 1.0
    psi = (rng.standard_normal((NK, NB, NS, NGKMAX))
           + 1j * rng.standard_normal((NK, NB, NS, NGKMAX))
           ).astype(np.complex128)
    psi[..., NGK:] = 0.0
    kvecs = rng.standard_normal((NK, 3)) * 0.25
    bvec, blat = np.eye(3) * 1.17, 1.0
    return psi, gv, gmask, bidx, kvecs, bvec, blat


def _geom(mesh):
    return SweepGeometry(mesh=mesh, fft_grid=GRID, ngkmax=NGKMAX, nb=NB,
                         ns=NS, nk=NK, cell_volume=1.0)


def _apply_one_k(op, psi, gv, gmask, bidx, kvec):
    """The operator's own output at one k, as the ket it returns."""
    return np.asarray(jax.device_get(
        op.apply(jnp.asarray(psi)[None], jnp.asarray(gv),
                 jnp.asarray(gmask), jnp.asarray(bidx),
                 jnp.asarray(kvec), *op.consts)))


def _reference_ket(psi, gv, gmask, kvec, bvec, blat, setup, sign=+1.0):
    """``p ψ + sign · (dV_NL/dK) ψ`` written out from the two kernels.

    Built from the kernels rather than from a saved array on purpose: a
    frozen array would prove the code matches its own past self on one
    fixture, while this reproduces the literal line ``v = v + pad`` and
    therefore fails if the operator stops being that line.  ``sign`` is
    here only so the red twin can build the retired arm.
    """
    from psp.dft_operators import apply_kinetic_velocity_to_ket
    from psp import vnl_ops

    B = jnp.asarray(np.asarray(bvec, dtype=np.float64) * float(blat),
                    dtype=jnp.float64)
    ket = jnp.asarray(psi) * jnp.asarray(gmask)[None, None, :]
    v = apply_kinetic_velocity_to_ket(ket, jnp.asarray(gv),
                                      jnp.asarray(kvec), B)
    kdata = vnl_ops.build_vnl_kdata_traced(jnp.asarray(kvec),
                                           jnp.asarray(gv), setup,
                                           compute_dZ=True)
    ns_e = int(kdata.E_super.shape[0])
    v_nl = vnl_ops.apply_vnl_velocity_to_ket(ket[:, :ns_e], kdata.Z,
                                             kdata.dZ, kdata.E_super)
    v = v + v_nl if sign > 0.0 else v - v_nl
    return np.asarray(jax.device_get(jnp.moveaxis(v, 0, -1)[None]))


def _mtxel(ket_out, psi, gmask):
    """<m|v|n> from an operator's ket, so the moving quantity is the one
    that lands in ``dipole.h5`` and not an intermediate."""
    bra = np.conj(psi * gmask[None, None, :])
    return np.einsum("msG,nsGa->amn", bra, ket_out[0], optimize=True)


# ---------------------------------------------------------------------------
# 1. The operator is p + dV_NL/dK, bit for bit
# ---------------------------------------------------------------------------

def test_the_constant_is_plus_one():
    assert VNL_VELOCITY_SIGN == +1.0


def test_operator_is_p_plus_dvnl_dk_bit_for_bit():
    mesh = _mesh()
    psi, gv, gmask, bidx, kvecs, bvec, blat = _fixture()
    setup = _vnl_setup()
    with mesh:
        geom = _geom(mesh)
        op = dipole_operator(geom, bvec=bvec, blat=blat, vnl_setup=setup)
        for ik in range(NK):
            got = _apply_one_k(op, psi[ik], gv[ik], gmask[ik], bidx[ik],
                               kvecs[ik])
            ref = _reference_ket(psi[ik], gv[ik], gmask[ik], kvecs[ik],
                                 bvec, blat, setup)
            assert np.array_equal(got, ref), (
                f"k={ik}: the operator moved off p + dV_NL/dK by "
                f"{np.max(np.abs(got - ref)):.3e}.  Bit identity, not a "
                f"tolerance: the reference is the literal expression the "
                f"operator is documented to be.")


def test_the_reference_can_fail():
    """The retired minus arm must NOT match, or the cell above compares
    two spellings of one number and proves nothing."""
    mesh = _mesh()
    psi, gv, gmask, bidx, kvecs, bvec, blat = _fixture()
    setup = _vnl_setup()
    with mesh:
        geom = _geom(mesh)
        op = dipole_operator(geom, bvec=bvec, blat=blat, vnl_setup=setup)
        got = _apply_one_k(op, psi[0], gv[0], gmask[0], bidx[0], kvecs[0])
        retired = _reference_ket(psi[0], gv[0], gmask[0], kvecs[0], bvec,
                                 blat, setup, sign=-1.0)
        assert not np.array_equal(got, retired)
        assert np.max(np.abs(got - retired)) > 1e-6 * np.max(np.abs(got))


# ---------------------------------------------------------------------------
# 2. The nonlocal term is live, and it is the whole difference from p
# ---------------------------------------------------------------------------

def test_the_nonlocal_term_is_live_and_is_the_whole_difference_from_p():
    mesh = _mesh()
    psi, gv, gmask, bidx, kvecs, bvec, blat = _fixture()
    setup = _vnl_setup()
    with mesh:
        geom = _geom(mesh)
        full = dipole_operator(geom, bvec=bvec, blat=blat, vnl_setup=setup)
        p_only = dipole_operator(geom, bvec=bvec, blat=blat, vnl_setup=None)
        for ik in range(NK):
            args = (psi[ik], gv[ik], gmask[ik], bidx[ik], kvecs[ik])
            m_full = _mtxel(_apply_one_k(full, *args), psi[ik], gmask[ik])
            m_p = _mtxel(_apply_one_k(p_only, *args), psi[ik], gmask[ik])
            scale = np.max(np.abs(m_full))
            assert np.max(np.abs(m_full - m_p)) > 1e-3 * scale, (
                f"k={ik}: the projector term is not reaching the assembled "
                f"velocity (or the fixture produced a null one)")
            # ... and what it adds is dV_NL/dK and nothing else.
            v_nl_only = (_reference_ket(*args[:3], kvecs[ik], bvec, blat, setup)
                         - _reference_ket(*args[:3], kvecs[ik], bvec, blat,
                                          setup, sign=-1.0)) / 2.0
            assert np.allclose(m_full - m_p, _mtxel(v_nl_only, psi[ik], gmask[ik]),
                               rtol=1e-12, atol=1e-12 * scale)


def test_operator_key_separates_p_only_from_p_plus_vnl():
    """The sweep's jit cache is keyed on ``_operator_key``; the two
    operators must not share a compiled program."""
    mesh = _mesh()
    *_, bvec, blat = _fixture()
    setup = _vnl_setup()
    with mesh:
        geom = _geom(mesh)
        k_full = _operator_key(dipole_operator(geom, bvec=bvec, blat=blat,
                                               vnl_setup=setup))
        k_p = _operator_key(dipole_operator(geom, bvec=bvec, blat=blat,
                                            vnl_setup=None))
    assert k_full != k_p


# ---------------------------------------------------------------------------
# 3. The stamp, and the refusal of the retired arm
# ---------------------------------------------------------------------------

class _FakeWfn:
    def __init__(self, nbands=8, nelec=4, nspinor=1, nk=3):
        rng = np.random.default_rng(0)
        self.energies = rng.standard_normal((1, nk, nbands))
        self.kpoints = rng.standard_normal((nk, 3))
        self.nelec, self.nspinor, self.nbands = nelec, nspinor, nbands


def _stamp(path, **kw):
    import h5py

    from psp.get_dipole_mtxels import stamp_dipole_provenance

    with h5py.File(str(path), "w") as h5:
        h5.create_dataset("dipole_cart", data=np.zeros((3, 1, 1, 1)))
        stamp_dipole_provenance(h5, wfn=_FakeWfn(), wfn_path="WFN.h5",
                                nval=2, ncond=3, nband=8, nb_written=4,
                                bispinor=False, skip_vnl=False, **kw)


def test_every_stamp_says_analytic_plus_one(tmp_path):
    import h5py

    p = tmp_path / "dipole.h5"
    _stamp(p)
    with h5py.File(str(p), "r") as h5:
        assert float(h5.attrs["prov_vnl_velocity_sign"]) == +1.0
        assert h5.attrs["prov_vnl_mode"] == "analytic"


@pytest.mark.parametrize("stale", [
    {"prov_vnl_velocity_sign": -1.0},          # the retired arm
    {"prov_vnl_mode": "numeric"},              # the retired cross-check
])
def test_the_provenance_check_refuses_the_retired_arm(tmp_path, monkeypatch,
                                                      stale):
    import h5py

    from common import sanity
    from psp.get_dipole_mtxels import check_dipole_provenance

    monkeypatch.setattr(sanity, "sanity_strict", lambda: False)
    wfn = _FakeWfn()
    p = tmp_path / "dipole.h5"
    _stamp(p)
    lines = []
    assert check_dipole_provenance(p, wfn=wfn, nval=2, ncond=3, nband=8,
                                   print_fn=lines.append) is True
    with h5py.File(str(p), "r+") as h5:
        for key, value in stale.items():
            h5.attrs[key] = value
    lines = []
    assert check_dipole_provenance(p, wfn=wfn, nval=2, ncond=3, nband=8,
                                   print_fn=lines.append) is False
    (key,) = stale
    assert any(key in line for line in lines)


def test_a_stamp_without_the_sign_is_refused(tmp_path, monkeypatch):
    """A stamped file that cannot say its arm predates 2026-08-09 and was
    built with the retired minus; it is uncheckable, not legacy-accepted."""
    import h5py

    from common import sanity
    from psp.get_dipole_mtxels import check_dipole_provenance

    monkeypatch.setattr(sanity, "sanity_strict", lambda: False)
    p = tmp_path / "dipole.h5"
    _stamp(p)
    with h5py.File(str(p), "r+") as h5:
        del h5.attrs["prov_vnl_velocity_sign"]
    assert check_dipole_provenance(p, wfn=_FakeWfn(), nval=2, ncond=3,
                                   nband=8, print_fn=lambda *a: None) is False


# ---------------------------------------------------------------------------
# 4. The retired deck key and CLI flags refuse by name
# ---------------------------------------------------------------------------

def _deck(tmp_path, body):
    p = tmp_path / "deck.in"
    p.write_text("[cohsex]\nwfn_file = WFN.h5\n" + body)
    return str(p)


@pytest.mark.parametrize("written", ["+1", "-1", "flipped", ""])
def test_the_deck_key_is_retired(tmp_path, written):
    from gw.gw_config import read_lorrax_input

    with pytest.raises(ValueError, match="vnl_velocity_sign.*retired"):
        read_lorrax_input(_deck(tmp_path, f"vnl_velocity_sign = {written}\n"))


def test_a_deck_without_the_key_reads(tmp_path):
    from gw.gw_config import read_lorrax_input

    params = read_lorrax_input(_deck(tmp_path, ""))
    assert "vnl_velocity_sign" not in params


def test_the_producer_has_no_numeric_arm_and_no_sign_flag():
    """Source-text scan of the producer: the CLI surface is gone, not
    merely ignored.  Read as text so the scan needs no FFI host library."""
    import pathlib

    from common import mtxel_sweep

    # ``psp`` is a namespace package; locate the tree from a real module.
    src = (pathlib.Path(mtxel_sweep.__file__).parents[1] / "psp"
           / "get_dipole_mtxels.py").read_text()
    for gone in ("--vnl-mode", "--vnl-h", "--vnl-num-scheme",
                 "--vnl-velocity-sign", "resolve_vnl_velocity_sign",
                 "compute_vnl_matrix_from_setup"):
        assert gone not in src, gone
