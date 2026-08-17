"""The relative sign of i[r, V_NL] in the assembled velocity, as a knob.

WHAT WAS IN DISPUTE, AND HOW IT WAS SETTLED.
``common.mtxel_sweep.dipole_operator`` used to assemble the velocity as
``p`` MINUS the nonlocal commutator term.  Measured against BerkeleyGW's
own q -> 0 head at all 265 contour-deformation frequencies on the
si_bigcond_prep mean field (nval 8 / ncond 92 / nband 100), that arm is
31.38 % high in eps00(0) and 17.45 % high in omega_p, while the other
agrees to 1.0e-5.  ``gw.mpa.head_dipole.head_fsum_from_transitions``
carries the whole four-arm table.  **The owner ruled on 2026-08-09 and
the default is now the PLUS arm.**  This file no longer gates "the
default is the legacy sign"; it gates that both arms remain reachable
from a keyword, a CLI flag and a deck key without anybody patching a
source file, that the default really is the new one at the operator's
cache key, and that the LEGACY arm is still bit-identical to the
pre-knob expression -- which is what keeps every ``dipole.h5`` built
before the flip reproducible.

THE FALSE CASE, WHICH IS THE WHOLE POINT.  This project's named failure
mode is a key that is parsed, stored and never read -- ``x_only`` was a
value of ``compute_mode`` while 62 decks wrote ``x_only = false``
believing it a switch, and ``screening_method = ctsp`` reached a typed
field no reader touched and silently ran minimax.  Grepping for
references does not prove a knob is live.  So every assertion below is
paired: the legacy arm must be BIT-IDENTICAL to the expression the code
had before the knob existed, and the two arms must DIFFER by a margin
no rounding could produce.  A knob that passed the first and failed the
second would be exactly the defect.

THE CACHE COLLISION IS A REAL HAZARD HERE, not a hypothetical one.  The
sweep's jit cache is keyed on ``_operator_key``, and a sign closed over
by the operator's ``apply`` closure without entering that tuple would
hash identically for the two arms -- so an A/B in one process would
serve the FIRST arm's compiled program to the second and the two arms
would agree, which is the shape of the defect that once let both halves
of an A/B import the same tree and agree at "128 passed" while testing
neither branch.  ``test_sweep_does_not_serve_one_arm_for_the_other``
runs the two sweeps in one process, in both orders, and is the check
that would fail if the sign left the key.

Everything here is synthetic and device-free: the V_NL setup is a
handful of random projector rows, because the quantity under test is a
sign and not a pseudopotential.  The per-(psi, G-list) projector
contraction is NOT touched -- it reproduces Quantum ESPRESSO to ~10
significant figures and the standing project rule protects it.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
import pytest

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.mtxel_sweep import (VNL_VELOCITY_SIGN_FLIPPED,
                                VNL_VELOCITY_SIGN_SHIPPED, SweepGeometry,
                                _operator_key, blocks_to_host,
                                dipole_operator, sweep_matrix_elements)

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
    dZ blocks into; a mismatch there would pair a projector with the
    wrong derivative and both arms would move together, which is why the
    straddle assertion in the perturbation test is worth having.

    E_super is deliberately random and not the zeros
    ``tests/test_psp_padded_gvectors`` uses: a null E_super makes
    ``v_nl`` identically zero, both arms agree, and this whole file
    becomes a tautology wearing a measurement's clothes.
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


def _pre_knob_ket(psi, gv, gmask, kvec, bvec, blat, setup, sign):
    """The expression the code had BEFORE the knob, written out here.

    This is the reference the default arm must match bit for bit, and it
    is written from the kernels rather than from a saved array on
    purpose: a frozen array would prove the code matches its own past
    self on one fixture, while this reproduces the literal line
    ``v = v - _pad_spinor(v_nl, ...)`` and therefore fails if the
    default stops being that line.
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
    v = v - v_nl if sign < 0.0 else v + v_nl
    return np.asarray(jax.device_get(jnp.moveaxis(v, 0, -1)[None]))


def _mtxel(ket_out, psi, gmask):
    """<m|v|n> from an operator's ket, so the moving quantity is the one
    that lands in ``dipole.h5`` and not an intermediate."""
    bra = np.conj(psi * gmask[None, None, :])
    return np.einsum("msG,nsGa->amn", bra, ket_out[0], optimize=True)


# ---------------------------------------------------------------------------
# 1. The default arm is the pre-knob expression, bit for bit
# ---------------------------------------------------------------------------

def test_default_arm_is_bit_identical_to_the_pre_knob_assembly():
    """The DEFAULT is the flipped arm; the LEGACY arm is still exact.

    Two assertions that used to be one.  Before 2026-08-09 the default
    was the legacy sign and this cell proved the knob had not disturbed
    it.  The default has since moved -- so what needs proving is (a)
    that it really moved, at the operator's cache key and not merely in
    a docstring, and (b) that the legacy arm, explicitly requested, is
    STILL the pre-knob expression bit for bit.  (b) is the promise that
    keeps every ``dipole.h5`` committed before the flip reproducible,
    and it is the one that would quietly rot if nobody checked it.
    """
    mesh = _mesh()
    psi, gv, gmask, bidx, kvecs, bvec, blat = _fixture()
    setup = _vnl_setup()
    with mesh:
        geom = _geom(mesh)
        op = dipole_operator(geom, bvec=bvec, blat=blat, vnl_setup=setup)
        assert op.key[-1] == VNL_VELOCITY_SIGN_FLIPPED, (
            "the default must BE the flipped sign, not merely behave "
            "like it: a default that resolved elsewhere and happened to "
            "agree on this fixture would pass every assertion below")

        legacy = dipole_operator(geom, bvec=bvec, blat=blat,
                                 vnl_setup=setup,
                                 vnl_velocity_sign=VNL_VELOCITY_SIGN_SHIPPED)
        for ik in range(NK):
            got = _apply_one_k(legacy, psi[ik], gv[ik], gmask[ik],
                               bidx[ik], kvecs[ik])
            ref = _pre_knob_ket(psi[ik], gv[ik], gmask[ik], kvecs[ik],
                                bvec, blat, setup,
                                VNL_VELOCITY_SIGN_SHIPPED)
            assert np.array_equal(got, ref), (
                f"k={ik}: the legacy arm moved off the pre-knob "
                f"expression by {np.max(np.abs(got - ref)):.3e}.  This "
                f"assertion is bit-identity and not a tolerance, because "
                f"the promise it keeps is that every dipole.h5 built "
                f"before the flip is still reproducible.")


def test_the_bit_identity_reference_can_fail():
    """The FALSE case for the reference itself.

    ``test_default_arm_is_bit_identical_...`` is only evidence if its
    reference can disagree.  Feed the same reference builder the other
    sign and it must not match -- otherwise the comparison is between
    two spellings of one number and proves nothing about either arm.

    It follows the primary cell onto the LEGACY arm: since the default
    moved to ``+1`` this must compare the legacy operator against the
    flipped reference, or it would be handing the builder the very sign
    the operator now uses and asserting they differ, which is a cell
    that fails for the wrong reason.
    """
    mesh = _mesh()
    psi, gv, gmask, bidx, kvecs, bvec, blat = _fixture()
    setup = _vnl_setup()
    with mesh:
        geom = _geom(mesh)
        op = dipole_operator(geom, bvec=bvec, blat=blat, vnl_setup=setup,
                             vnl_velocity_sign=VNL_VELOCITY_SIGN_SHIPPED)
        got = _apply_one_k(op, psi[0], gv[0], gmask[0], bidx[0], kvecs[0])
        wrong = _pre_knob_ket(psi[0], gv[0], gmask[0], kvecs[0], bvec, blat,
                              setup, VNL_VELOCITY_SIGN_FLIPPED)
        assert not np.array_equal(got, wrong)
        assert np.max(np.abs(got - wrong)) > 1e-6 * np.max(np.abs(got))


# ---------------------------------------------------------------------------
# 2. The flipped arm MOVES the physics -- the perturbation test
# ---------------------------------------------------------------------------

def test_flipped_arm_moves_the_matrix_elements():
    """Perturb it and watch the physics move.

    The margin asserted is not "not equal": the two arms must differ by
    exactly twice the nonlocal term, which is a statement about WHAT
    moved and not merely that something did.  A knob wired to the wrong
    operand would fail this while passing a bare inequality.
    """
    mesh = _mesh()
    psi, gv, gmask, bidx, kvecs, bvec, blat = _fixture()
    setup = _vnl_setup()
    with mesh:
        geom = _geom(mesh)
        shipped = dipole_operator(geom, bvec=bvec, blat=blat,
                                  vnl_setup=setup,
                                  vnl_velocity_sign=VNL_VELOCITY_SIGN_SHIPPED)
        flipped = dipole_operator(geom, bvec=bvec, blat=blat,
                                  vnl_setup=setup,
                                  vnl_velocity_sign=VNL_VELOCITY_SIGN_FLIPPED)
        p_only = dipole_operator(geom, bvec=bvec, blat=blat, vnl_setup=None)
        for ik in range(NK):
            args = (psi[ik], gv[ik], gmask[ik], bidx[ik], kvecs[ik])
            m_s = _mtxel(_apply_one_k(shipped, *args), psi[ik], gmask[ik])
            m_f = _mtxel(_apply_one_k(flipped, *args), psi[ik], gmask[ik])
            m_p = _mtxel(_apply_one_k(p_only, *args), psi[ik], gmask[ik])
            scale = np.max(np.abs(m_s))
            moved = np.max(np.abs(m_f - m_s))
            assert moved > 1e-3 * scale, (
                f"k={ik}: flipping the sign moved the matrix elements by "
                f"{moved:.3e} against a scale of {scale:.3e}.  A knob that "
                f"is parsed, stored and never read is this project's named "
                f"failure mode; this is the assertion that catches it.")
            # The two arms straddle p exactly: (v_+ + v_-)/2 == p.
            assert np.allclose(0.5 * (m_f + m_s), m_p, rtol=1e-12,
                               atol=1e-12 * scale), (
                f"k={ik}: the two arms do not straddle the momentum-only "
                f"operator, so what the knob moved is not the nonlocal "
                f"term.")
            assert np.max(np.abs(m_s - m_p)) > 1e-3 * scale, (
                "fixture produced a null nonlocal term, which would make "
                "every assertion in this file vacuous")


# ---------------------------------------------------------------------------
# 3. The cache key -- and the sweep that would collide without it
# ---------------------------------------------------------------------------

def test_operator_key_separates_the_two_arms_and_nothing_else():
    mesh = _mesh()
    _psi, _gv, _gmask, _bidx, _kv, bvec, blat = _fixture()
    setup = _vnl_setup()
    with mesh:
        geom = _geom(mesh)
        k_s = _operator_key(dipole_operator(
            geom, bvec=bvec, blat=blat, vnl_setup=setup,
            vnl_velocity_sign=VNL_VELOCITY_SIGN_SHIPPED))
        k_f = _operator_key(dipole_operator(
            geom, bvec=bvec, blat=blat, vnl_setup=setup,
            vnl_velocity_sign=VNL_VELOCITY_SIGN_FLIPPED))
    assert k_s != k_f, (
        "the two arms hash the same, so the sweep's jit cache would serve "
        "one arm's compiled program for the other")
    # ... and the sign is the ONLY thing that separates them, which is
    # what makes the inequality above evidence about the sign rather than
    # about some incidental difference between two factory calls.
    assert k_s[:-1] == k_f[:-1]
    assert (k_s[-1], k_f[-1]) == (VNL_VELOCITY_SIGN_SHIPPED,
                                  VNL_VELOCITY_SIGN_FLIPPED)


@pytest.mark.parametrize("order", [(-1.0, 1.0), (1.0, -1.0)])
def test_sweep_does_not_serve_one_arm_for_the_other(order):
    """Both arms, both orders, one process -- the end-to-end cache check.

    Run in both orders on purpose: a cache that collides returns the
    FIRST arm's answer for both, so a single order would be passed by an
    implementation that always returns the flipped arm.
    """
    mesh = _mesh()
    psi, gv, gmask, bidx, kvecs, bvec, blat = _fixture()
    setup = _vnl_setup()
    with mesh:
        geom = _geom(mesh)
        sharding = NamedSharding(mesh, P(None, ('x', 'y'), None, None))
        psi_j = jax.device_put(jnp.asarray(psi), sharding)
        out = {}
        for sign in order:
            op = dipole_operator(geom, bvec=bvec, blat=blat,
                                 vnl_setup=setup, vnl_velocity_sign=sign)
            blk = sweep_matrix_elements(
                psi_j, geom=geom, operator=op, gvecs=jnp.asarray(gv),
                gmask=jnp.asarray(gmask), box_index=jnp.asarray(bidx),
                kvecs=jnp.asarray(kvecs))
            out[sign] = np.asarray(blocks_to_host(blk, nb=NB))
    scale = np.max(np.abs(out[-1.0]))
    moved = np.max(np.abs(out[1.0] - out[-1.0]))
    assert moved > 1e-3 * scale, (
        f"the two arms came back {moved:.3e} apart on a scale of "
        f"{scale:.3e} after two sweeps in one process: the second sweep "
        f"was served the first's compiled program, which is what the "
        f"sign in _operator_key exists to prevent.")


# ---------------------------------------------------------------------------
# 4. The refusal, and the producer's resolution order
# ---------------------------------------------------------------------------

def test_finite_q_producer_requires_explicit_velocity_sign():
    import inspect
    from psp.get_dipole_mtxels import compute_finite_q_mtxels

    sign = inspect.signature(compute_finite_q_mtxels).parameters["vnl_velocity_sign"]
    assert sign.default is inspect.Parameter.empty


@pytest.mark.parametrize("bad", [0.0, 2.0, -0.5, "sideways"])
def test_operator_refuses_anything_that_is_not_a_sign(bad):
    mesh = _mesh()
    *_, bvec, blat = _fixture()
    with mesh:
        geom = _geom(mesh)
        with pytest.raises((ValueError, TypeError)):
            dipole_operator(geom, bvec=bvec, blat=blat, vnl_setup=None,
                            vnl_velocity_sign=bad)


@pytest.mark.parametrize("cli, deck, want", [
    (None, "", VNL_VELOCITY_SIGN_FLIPPED),          # neither: the default
    (None, None, VNL_VELOCITY_SIGN_FLIPPED),        # absent key
    (None, "-1", VNL_VELOCITY_SIGN_SHIPPED),
    (None, "+1", VNL_VELOCITY_SIGN_FLIPPED),
    (None, "flipped", VNL_VELOCITY_SIGN_FLIPPED),
    (None, "shipped", VNL_VELOCITY_SIGN_SHIPPED),
    (1.0, "", VNL_VELOCITY_SIGN_FLIPPED),           # CLI with no deck key
    (1.0, "-1", VNL_VELOCITY_SIGN_FLIPPED),         # CLI beats the deck
    (-1.0, "+1", VNL_VELOCITY_SIGN_SHIPPED),        # ... in both directions
])
def test_producer_resolution_order(cli, deck, want):
    from psp.get_dipole_mtxels import resolve_vnl_velocity_sign

    assert resolve_vnl_velocity_sign(cli, deck) == want


@pytest.mark.parametrize("deck", ["0", "2", "yes", "-1.5"])
def test_producer_refuses_a_deck_key_that_is_not_a_sign(deck):
    from psp.get_dipole_mtxels import resolve_vnl_velocity_sign

    with pytest.raises(ValueError, match="vnl_velocity_sign"):
        resolve_vnl_velocity_sign(None, deck)


def _deck(tmp_path, body):
    p = tmp_path / "deck.in"
    p.write_text("[cohsex]\nwfn_file = WFN.h5\n" + body)
    return str(p)


@pytest.mark.parametrize("written, want", [("+1", "+1"), ("-1", "-1"),
                                           ("flipped", "flipped")])
def test_the_deck_key_survives_the_deck_reader(tmp_path, written, want):
    """The FALSE case a grep would have passed, and did.

    ``read_lorrax_input`` builds its params dict from ``_DEFAULTS``
    ALONE, so a key absent from that table is parsed, reported as
    unrecognized and dropped -- and the producer then reads its own
    default and runs the OTHER ARM while the deck says otherwise.  That
    is not hypothetical: the first flipped-arm ``dipole.h5`` of this
    campaign came back stamped ``-1.0``, with "1 unrecognized deck
    key(s)" in the log, because the key had been plumbed everywhere
    except into ``_DEFAULTS``.  Reading the value back verbatim -- not
    folded to a bool, not coerced -- is what says the whole chain from
    the deck line to the resolver is connected.
    """
    from gw.gw_config import read_lorrax_input

    params = read_lorrax_input(
        _deck(tmp_path, f"vnl_velocity_sign = {written}\n"))
    assert params["vnl_velocity_sign"] == want


def test_a_deck_that_omits_the_key_reads_as_not_declared(tmp_path):
    """Empty is UNSET and must stay distinguishable from an explicit
    ``-1``: the two produce the same operator and the same numbers, but
    only one of them is a deck that made a choice."""
    from gw.gw_config import read_lorrax_input

    params = read_lorrax_input(_deck(tmp_path, ""))
    assert params["vnl_velocity_sign"] == ""


def test_strict_keys_accepts_a_deck_that_names_the_sign(tmp_path):
    """``strict_keys = true`` REFUSES unknown keys by name, so a deck
    that declares the sign under it is the sharpest witness that the key
    is registered rather than merely tolerated."""
    from gw.gw_config import read_lorrax_input

    params = read_lorrax_input(_deck(
        tmp_path, "vnl_velocity_sign = +1\nstrict_keys = true\n"))
    assert params["vnl_velocity_sign"] == "+1"


def test_the_whole_chain_from_deck_line_to_resolved_sign(tmp_path):
    """Deck text in, arm out -- the two halves joined.

    Each half is gated above; this is the cell that fails if they are
    gated against different spellings of the key.
    """
    from gw.gw_config import read_lorrax_input
    from psp.get_dipole_mtxels import resolve_vnl_velocity_sign

    for written, want in (("+1", VNL_VELOCITY_SIGN_FLIPPED),
                          ("-1", VNL_VELOCITY_SIGN_SHIPPED),
                          ("flipped", VNL_VELOCITY_SIGN_FLIPPED),
                          ("", VNL_VELOCITY_SIGN_FLIPPED)):
        line = f"vnl_velocity_sign = {written}\n" if written else ""
        params = read_lorrax_input(_deck(tmp_path, line))
        got = resolve_vnl_velocity_sign(None, params["vnl_velocity_sign"])
        assert got == want, (written, got, want)


# ---------------------------------------------------------------------------
# 5. The stamp -- a dipole.h5 that cannot say which arm built it
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
                                bispinor=False, skip_vnl=False,
                                vnl_mode="analytic", **kw)


@pytest.mark.parametrize("sign", [VNL_VELOCITY_SIGN_SHIPPED,
                                  VNL_VELOCITY_SIGN_FLIPPED])
def test_the_stamp_records_the_arm_that_ran(tmp_path, sign):
    import h5py

    p = tmp_path / "dipole.h5"
    _stamp(p, vnl_velocity_sign=sign)
    with h5py.File(str(p), "r") as h5:
        assert float(h5.attrs["prov_vnl_velocity_sign"]) == sign


def test_an_unstamped_file_stays_unstamped(tmp_path):
    """``None`` is "written before the knob existed", which is a real
    state on disk and must not be forged into a claim about the arm."""
    import h5py

    p = tmp_path / "dipole.h5"
    _stamp(p)
    with h5py.File(str(p), "r") as h5:
        assert "prov_vnl_velocity_sign" not in h5.attrs
