"""Gates for the downfold: the algebraic identity, its red twins, the error bar.

EVERY GATE HERE HAS A RED TWIN.  A gate that has never been seen to fail is
not a gate, it is a hope with an assert in front of it — and this suite is
the foundation of an initiative whose whole premise is that a compression can
look right and be wrong.  So each cell that asserts something states, right
beside it, the perturbation that must break it, and asserts that too.

Deck-free and CPU-only: random psi at tiny shapes, seconds per cell.  The
real-deck observable-drift gate (downfold si_bse_debug, run the BSE driver on
the small bundle UNCHANGED, compare exciton eigenvalues) needs a GPU and a
finished GW run, so it lives in the campaign report rather than here.

The multi-device arm at the foot of the file follows this repo's established
shape: one process, ``--xla_force_host_platform_device_count``, the test file
re-executing itself as its own worker.  That is a REAL 2x2 mesh with real
collectives and no MPI.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import jax                                                    # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                       # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P    # noqa: E402

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from common.collectives import resolve_mesh                   # noqa: E402
from gw import downfold                                       # noqa: E402
from gw.downfold import BandWindow                            # noqa: E402


@pytest.fixture(autouse=True)
def _bands_gemm_on_the_xla_lowering(monkeypatch):
    """Route the congruence's GEMM through XLA rather than the vendor dial.

    ``contract_bands_block_reshard`` REQUIRES the MKL batched-GEMM handler on
    a CPU mesh (decisions.md 2026-08-01) and refuses when the host ``.so`` is
    not built.  These cells are ALGEBRA gates — they assert that T = I and
    that a congruence reproduces its operand — and the dial is a BACKEND
    concern gated by the vendor-GEMM suite, not by this one.  Pinning the XLA
    lowering here also makes the CPU cells exercise the SAME code path a
    production GPU run takes: the dial resolves to None on a non-CPU mesh
    (XLA:GPU's dot lowering already hits cuBLAS), so on Perlmutter the
    congruence is the XLA plan regardless.
    """
    monkeypatch.setenv("LORRAX_BANDS_GEMM_FFI", "0")


NK = 4
NB = 6
NS = 1
MU = 8
KGRID = (NK, 1, 1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mu_pad(n):
    """The device-legal μ extent ``run_downfold`` itself pads to.

    A CELL MUST BE ABLE TO EXPRESS ITSELF ON THE MESH IT FINDS.  Measured
    2026-08-10 at four real GPUs (`owedlegs_0810/_logs/p4gates_gpu4.log`):
    three cells in this initiative's two files were written at literal
    extents — ``m = n = mu_S = 3``, an ``(nq, 4, 9)`` transfer — that do not
    divide a 2×2 mesh, so ``contract_bands_block_reshard`` and ``pjit``
    refused them BY NAME while the product they gate was healthy.  A 1×1
    shape is not a small case of a P>1 shape; it is a different shape, and a
    gate that only has one cannot certify the geometry production runs in.

    The repair is the tree's own recipe rather than a rounder literal:
    ``runtime.padding.padded_mu_extent`` is what ``downfold_run`` calls at
    ``:348`` to size ``mu_S_pad``, and the zeta writer's own comment records
    that "the transfer carries the device-legal μ pads on both axes ... and
    the pad rows are exactly zero, so the slice is a restriction and not a
    choice".  These cells now build that same padded operand, zero-filled,
    and slice the logical block back off the answer.  At one device the
    extent is unchanged, so the CPU numbers these cells have always reported
    are untouched.
    """
    from runtime.padding import padded_mu_extent
    return int(padded_mu_extent(int(n), int(jax.device_count())))


def _psi(mesh, *, seed=0, nk=NK, nb=NB, ns=NS, mu=MU):
    """Random psi-at-centroids in the two layouts the Gram kernel wants."""
    rng = np.random.default_rng(seed)
    y = (rng.normal(size=(nk, nb, ns, mu))
         + 1j * rng.normal(size=(nk, nb, ns, mu)))
    psi_Y = jnp.asarray(y)
    psi_X = jnp.conj(psi_Y).transpose(0, 3, 1, 2)
    psi_Y = jax.lax.with_sharding_constraint(
        psi_Y, NamedSharding(mesh, P(None, None, None, "y")))
    psi_X = jax.lax.with_sharding_constraint(
        psi_X, NamedSharding(mesh, P(None, "x", None, None)))
    return psi_X, psi_Y


def _dense_gram(psi_y_host, window, *, mu_rows=None):
    """The Gram, built the slow honest way, straight off the definition.

    ``S(q)[mu,nu] = sum_{m in L, n in R, k} X[mu,a] conj(X[nu,a])`` with
    ``X[mu,a] = conj(psi_m(k, mu)) psi_n(k+q, mu)``, factorised through the
    Khatri-Rao identity the kernel exploits.  Written out with every index
    visible, because this reference is the only thing in the suite that
    pins the CONJUGATION CONVENTION — and the campaign has been bitten twice
    by conjugation conventions hidden behind time-reversal symmetry at q=0.

    **THIS REFERENCE CARRIED THE SAME q-SIGN ERROR AS THE IMPLEMENTATION IT
    CHECKS, 2026-08-09 to 2026-08-11.**  Its prose said "m at ``k-q``, n at
    ``k``" and its code put the LEFT (m) window at ``k+q`` and the RIGHT (n)
    window at ``k`` — the ``q -> -q`` relabelling — so the three cells below
    agreed with a Gram built at ``-q`` and reported a green.  A reference
    written from the same hand as the code is not a reference; the cell that
    could not be written that way is
    :func:`test_the_gram_is_labelled_by_PLUS_q_analytically`, which has a
    closed form.  See ``tests/known_failures/2026-08-11-downfold-gram-q-sign.md``.
    """
    (bl0, bl1), (br0, br1) = window.left, window.right
    nk, nb, ns, mu = psi_y_host.shape
    assert ns == 1, "the dense reference is written for ns=1"
    psi = psi_y_host[:, :, 0, :]                     # (nk, nb, mu)
    rows = mu if mu_rows is None else mu_rows
    out = np.zeros((nk, rows, mu), dtype=np.complex128)
    for q in range(nk):
        acc = np.zeros((rows, mu), dtype=np.complex128)
        for k in range(nk):
            kq = (k + q) % nk
            # sum over the RIGHT (n) window at k+q
            right = np.einsum("nm,nv->mv", psi[kq, br0:br1, :rows],
                              np.conj(psi[kq, br0:br1, :]))
            # sum over the LEFT (m) window at k
            left = np.einsum("nm,nv->mv", np.conj(psi[k, bl0:bl1, :rows]),
                             psi[k, bl0:bl1, :])
            acc += right * left
        out[q] = acc
    return out


def _S_all(mesh, psi_X, psi_Y, window):
    return np.asarray(jax.device_get(downfold.pair_density_gram(
        psi_X, psi_Y, window, kgrid=KGRID, mesh_xy=mesh)))


def _transfer_dense(S_SS, S_cross, rcond):
    """T = S_SS^+ S_cross, host-side, for the red twins to perturb."""
    out = np.zeros_like(S_cross)
    for q in range(S_SS.shape[0]):
        w, V = np.linalg.eigh(0.5 * (S_SS[q] + S_SS[q].conj().T))
        keep = w > rcond * w.max()
        winv = np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0)
        out[q] = V @ (winv[:, None] * (V.conj().T @ S_cross[q]))
    return out


# ---------------------------------------------------------------------------
# 1.  The Gram: the conjugation convention, pinned against the definition
# ---------------------------------------------------------------------------

def _plane_wave_psi(mesh, R):
    """One band, ``psi(k, r_mu) = exp(2 pi i k R_mu)`` on the KGRID k-axis.

    Chosen because the Gram then has a CLOSED FORM (see the cell below), so
    the reference is an identity rather than a second transcription of the
    code under test.
    """
    k = np.arange(NK) / NK
    y = np.exp(2j * np.pi * np.outer(k, np.asarray(R, dtype=float)))
    y = y.reshape(NK, 1, 1, len(R))                       # (nk, nb=1, ns=1, mu)
    psi_Y = jax.lax.with_sharding_constraint(
        jnp.asarray(y), NamedSharding(mesh, P(None, None, None, "y")))
    psi_X = jax.lax.with_sharding_constraint(
        jnp.conj(jnp.asarray(y)).transpose(0, 3, 1, 2),
        NamedSharding(mesh, P(None, "x", None, None)))
    return psi_X, psi_Y


def test_the_gram_is_labelled_by_PLUS_q_analytically():
    """THE SIGN OF q, against a CLOSED FORM — the cell the suite was missing.

    With one band and ``psi(k, r_mu) = exp(2 pi i k R_mu)`` the pair density
    at momentum transfer q is ``X[mu, k] = conj(psi(k, r_mu)) psi(k+q, r_mu)
    = exp(2 pi i q R_mu)``, so straight from the definition

        S(q)[mu, nu] = sum_k X[mu,k] conj(X[nu,k])
                     = nk * exp(2 pi i q (R_mu - R_nu)).

    That is NOT invariant under ``q -> -q``, which is the whole point: every
    other Gram cell in this file compares against a hand-written double loop,
    and from 2026-08-09 to 2026-08-11 that loop and the kernel call carried
    the SAME ``q -> -q`` error, so they agreed and the suite was green while
    every downfolded child was built from the transfer belonging to ``-q``.
    A closed form cannot be written from the implementation's hand.

    Measured consequence on ``si_bse_debug``: wedge-composition covariance
    1.170e+00 before, 3.7e-08 after (``tests/known_failures/
    2026-08-11-downfold-gram-q-sign.md``).
    """
    mesh = resolve_mesh()
    R = np.arange(1, MU + 1, dtype=float)
    psi_X, psi_Y = _plane_wave_psi(mesh, R)
    win = BandWindow(left=(0, 1), right=(0, 1))
    got = _S_all(mesh, psi_X, psi_Y, win)
    q = np.arange(NK) / NK
    want = NK * np.exp(2j * np.pi * q[:, None, None]
                       * (R[None, :, None] - R[None, None, :]))
    assert got.shape == want.shape
    assert np.max(np.abs(got - want)) < 1e-10 * NK


def test_RED_TWIN_the_raw_kernel_labels_the_gram_by_MINUS_q():
    """The shipped defect, held as a twin: the kernel's q axis runs backwards.

    This is verbatim what ``pair_density_gram`` returned until 2026-08-11 —
    the kernel's output with NO q-axis relabel — and it must MISS the closed
    form by order one at every q with ``-q != q``.  If it ever agreed, the
    cell above would be testing arithmetic and not the labelling of q.

    It also pins WHY the error hid for two days, and the pin is the shape of
    the failure rather than its size: at the q with ``-q == q`` (here q=0 and
    q=1/2) the two labellings agree EXACTLY.  On ``si_bse_debug`` those were
    8 of 64 q — the three stars ``q_irr`` (0,0,0), (0,0,1/2), (0,1/2,1/2) —
    and they were the only blocks of the shipped child that the real-deck
    probe found intact.  A missing unfold phase would have tracked the
    umklapp wrap count instead; it does not.
    """
    from isdf.core import c_q_from_psi_sm

    mesh = resolve_mesh()
    R = np.arange(1, MU + 1, dtype=float)
    psi_X, psi_Y = _plane_wave_psi(mesh, R)
    raw = np.asarray(jax.device_get(c_q_from_psi_sm(
        psi_X[:, :, 0:1, :], psi_Y[:, 0:1, :, :],
        psi_X[:, :, 0:1, :], psi_Y[:, 0:1, :, :],
        kgrid=KGRID, mesh_xy=mesh)))
    q = np.arange(NK) / NK
    want = NK * np.exp(2j * np.pi * q[:, None, None]
                       * (R[None, :, None] - R[None, None, :]))
    self_inv = np.array([qi for qi in range(NK) if (-qi) % NK == qi])
    other = np.array([qi for qi in range(NK) if (-qi) % NK != qi])
    assert self_inv.size and other.size
    assert np.max(np.abs(raw[self_inv] - want[self_inv])) < 1e-10 * NK
    assert np.max(np.abs(raw[other] - want[other])) > 0.5 * NK
    # and it is EXACTLY want relabelled by q -> -q, which is the mechanism.
    neg = downfold.negate_q_index(KGRID)
    assert np.max(np.abs(raw - want[neg])) < 1e-10 * NK


def test_the_q_relabel_is_an_involution_and_fixes_only_self_inverse_q():
    """``negate_q_index`` is the relabel, and it is checked as one.

    Cheap, and it is the only piece of the repair that is index arithmetic
    rather than a measurement: applying it twice must be the identity, and
    its fixed points must be exactly the q with ``-q == q`` on that grid.
    """
    for grid in ((4, 1, 1), (4, 4, 4), (2, 3, 1), (1, 1, 1), (6, 2, 3)):
        idx = downfold.negate_q_index(grid)
        n = grid[0] * grid[1] * grid[2]
        assert idx.shape == (n,)
        assert np.array_equal(idx[idx], np.arange(n))
        n1, n2, n3 = grid
        want_fixed = {
            (i * n2 + j) * n3 + k
            for i in range(n1) for j in range(n2) for k in range(n3)
            if (-i) % n1 == i and (-j) % n2 == j and (-k) % n3 == k}
        assert set(np.flatnonzero(idx == np.arange(n)).tolist()) == want_fixed


def test_gram_matches_the_definition():
    mesh = resolve_mesh()
    psi_X, psi_Y = _psi(mesh, seed=1)
    win = BandWindow(left=(0, NB), right=(0, NB))
    got = _S_all(mesh, psi_X, psi_Y, win)
    want = _dense_gram(np.asarray(jax.device_get(psi_Y)), win)
    assert np.max(np.abs(got - want)) < 1e-10 * np.max(np.abs(want))


def test_gram_matches_the_definition_on_an_asymmetric_window():
    """The leg mapping is only observable when the two windows differ."""
    mesh = resolve_mesh()
    psi_X, psi_Y = _psi(mesh, seed=2)
    win = BandWindow(left=(0, 2), right=(2, NB))
    got = _S_all(mesh, psi_X, psi_Y, win)
    want = _dense_gram(np.asarray(jax.device_get(psi_Y)), win)
    assert np.max(np.abs(got - want)) < 1e-10 * np.max(np.abs(want))


def test_gram_definition_check_can_fail():
    """RED TWIN: swap the two band windows and the asymmetric check breaks.

    If this passed, the previous cell would be testing arithmetic rather
    than the conjugation convention it exists to pin.
    """
    mesh = resolve_mesh()
    psi_X, psi_Y = _psi(mesh, seed=2)
    win = BandWindow(left=(0, 2), right=(2, NB))
    swapped = BandWindow(left=(2, NB), right=(0, 2))
    got = _S_all(mesh, psi_X, psi_Y, swapped)
    want = _dense_gram(np.asarray(jax.device_get(psi_Y)), win)
    assert np.max(np.abs(got - want)) > 1e-3 * np.max(np.abs(want))


def test_cross_gram_by_call_equals_the_slice():
    """S_cross built by a CALL is S_LL sliced by ``keep``.

    The design builds the cross Gram by a second kernel call rather than by
    slicing an 'x'-sharded axis, because an arbitrary-index gather on a
    sharded axis is something GSPMD plans badly.  That is a performance
    choice, so the two routes must agree exactly, and this is the cell that
    says so.
    """
    mesh = resolve_mesh()
    psi_X, psi_Y = _psi(mesh, seed=3)
    win = BandWindow(left=(0, NB), right=(0, NB))
    keep = np.array([0, 2, 3, 6], dtype=np.int64)
    S_LL = _S_all(mesh, psi_X, psi_Y, win)
    psi_S_X, _ = downfold.slice_psi_to_centroids(
        psi_X, psi_Y, keep, len(keep), mesh)
    S_cross = _S_all(mesh, psi_S_X, psi_Y, win)
    assert np.max(np.abs(S_cross - S_LL[:, keep, :])) < 1e-11 * np.max(
        np.abs(S_LL))


# ---------------------------------------------------------------------------
# 2.  GATE 1 — the algebraic identity, and its four red twins
# ---------------------------------------------------------------------------

def _identity_setup(seed=7, rcond=1e-10):
    mesh = resolve_mesh()
    psi_X, psi_Y = _psi(mesh, seed=seed)
    win = BandWindow(left=(0, NB), right=(0, NB))
    S = _S_all(mesh, psi_X, psi_Y, win)
    return mesh, psi_X, psi_Y, win, S, rcond


def test_identity_gate_transfer_is_exactly_the_identity():
    """mu_S = mu_L, keep = everything, window = every band  ==>  T = I.

    Not "to within tolerance": when the small set IS the large set,
    ``S_SS = S_cross = S_LL`` and T is the same matrix inverted against
    itself.  That makes the exact-reproduction test an ALGEBRAIC IDENTITY
    rather than a hopeful comparison, which is the single strongest reason
    the design chose subset selection over a fresh fit.
    """
    _mesh, _psi_X, _psi_Y, _win, S, rcond = _identity_setup()
    T = _transfer_dense(S, S, rcond)
    eye = np.eye(S.shape[-1])[None].repeat(S.shape[0], axis=0)
    assert np.max(np.abs(T - eye)) < 1e-12


def test_identity_gate_congruence_reproduces_W():
    """T = I  ==>  W_S = T W_L T-dagger = W_L to machine precision."""
    mesh, psi_X, psi_Y, win, S, rcond = _identity_setup()
    mu = S.shape[-1]
    rng = np.random.default_rng(11)
    a = rng.normal(size=(NK, mu, mu)) + 1j * rng.normal(size=(NK, mu, mu))
    W_L = jnp.asarray(a + np.conj(np.swapaxes(a, 1, 2)))
    W_L = jax.lax.with_sharding_constraint(
        W_L, NamedSharding(mesh, P(None, "x", "y")))
    S_j = jax.lax.with_sharding_constraint(
        jnp.asarray(S), NamedSharding(mesh, P(None, "x", "y")))
    T_x, T_y, _ = downfold.build_transfer(
        S_j, S_j, mesh, rcond=rcond, announce=False)
    W_S = downfold.congruence(mesh, T_x, T_y)(W_L)
    got = np.asarray(jax.device_get(W_S))
    want = np.asarray(jax.device_get(W_L))
    scale = np.max(np.abs(want))
    assert np.max(np.abs(got - want)) < 1e-11 * scale


def test_identity_gate_red_twin_permuted_keep():
    """RED TWIN: permute ``keep`` and T must stop being the identity.

    If a permutation still gave T = I, the gate would not be reading
    ``keep_idx`` at all — it would be asserting that a matrix equals itself.
    """
    _mesh, _psi_X, _psi_Y, _win, S, rcond = _identity_setup()
    perm = np.array([3, 1, 0, 2, 5, 4, 7, 6])
    S_SS = S[:, perm, :][:, :, perm]
    S_cross = S[:, perm, :]
    T = _transfer_dense(S_SS, S_cross, rcond)
    eye = np.eye(S.shape[-1])[None].repeat(S.shape[0], axis=0)
    assert np.max(np.abs(T - eye)) > 0.5


def test_the_pair_density_gram_is_hermitian_at_every_q():
    """MEASURED, and it is why the naive conjugation twin is VACUOUS.

    ``S(q)[mu,nu] = sum_k [sum_n psi_n(k,mu) conj(psi_n(k,nu))]
    [sum_m conj(psi_m(k+q,mu)) psi_m(k+q,nu)]`` is Hermitian at EVERY q, not
    only at q=0 — each bracket is a Gram in its own right and the product of
    two Hermitian matrices under a Hadamard product is Hermitian.  So
    ``S_cross -> S_cross-dagger`` on the identity setup changes nothing and a
    red twin phrased that way would pass while testing nothing.

    That is the same hazard the campaign met twice (the BSE exchange
    conjugation and the ``wq_resolvent`` convention, both hidden by
    time-reversal symmetry at q=0), one level up: here the symmetry that
    hides it is Hermiticity of the Gram itself, at every q.  The twin that
    DOES bite is the next cell — flip which psi leg carries the conjugation
    when the Gram is BUILT.
    """
    _mesh, _psi_X, _psi_Y, _win, S, _rcond = _identity_setup()
    assert np.max(np.abs(S - np.conj(np.swapaxes(S, 1, 2)))) < 1e-10 * np.max(
        np.abs(S))


def test_identity_gate_red_twin_conjugation_of_the_psi_leg():
    """RED TWIN: build the Gram's row legs from psi instead of conj(psi).

    The pair density is ``rho_mn = conj(psi_m) psi_n``; which leg carries the
    conjugation is a convention that the Gram's Hermiticity CANNOT protect,
    because getting it wrong builds a different matrix rather than the same
    one transposed.
    """
    mesh, _psi_X, psi_Y, win, S, rcond = _identity_setup()
    psi_X_unconj = jax.lax.with_sharding_constraint(
        psi_Y.transpose(0, 3, 1, 2),
        NamedSharding(mesh, P(None, "x", None, None)))
    S_wrong = _S_all(mesh, psi_X_unconj, psi_Y, win)
    T = _transfer_dense(S, S_wrong, rcond)
    eye = np.eye(S.shape[-1])[None].repeat(S.shape[0], axis=0)
    assert np.max(np.abs(T - eye)) > 1e-3, (
        "un-conjugating the psi row leg left T = I, so the identity gate is "
        "blind to the conjugation convention")


def test_identity_gate_red_twin_wrong_band_window():
    """RED TWIN: build the Grams on a shifted window and T must break.

    This is ``isdf_basis_adequacy_at_large_nband.md`` in miniature — a basis
    fitted against one window and used on another.  Done by accident at
    nband=1024 it produced a QP gap of 0.36 eV where the answer is ~3.1-3.7
    eV, with every gate in the suite green.
    """
    mesh, psi_X, psi_Y, _win, S, rcond = _identity_setup()
    shifted = BandWindow(left=(2, NB), right=(2, NB))
    S_shift = _S_all(mesh, psi_X, psi_Y, shifted)
    T = _transfer_dense(S_shift, S, rcond)
    eye = np.eye(S.shape[-1])[None].repeat(S.shape[0], axis=0)
    assert np.max(np.abs(T - eye)) > 1e-3


def test_identity_gate_red_twin_perturbed_transfer():
    """RED TWIN (the FALSE arm of the identity gate): perturb T, W must move.

    The congruence is what carries the physics; a gate on T alone would not
    notice a congruence that ignored its operands.
    """
    mesh, _psi_X, _psi_Y, _win, S, rcond = _identity_setup()
    mu = S.shape[-1]
    rng = np.random.default_rng(13)
    a = rng.normal(size=(NK, mu, mu)) + 1j * rng.normal(size=(NK, mu, mu))
    W_L = jnp.asarray(a + np.conj(np.swapaxes(a, 1, 2)))
    W_L = jax.lax.with_sharding_constraint(
        W_L, NamedSharding(mesh, P(None, "x", "y")))
    S_j = jax.lax.with_sharding_constraint(
        jnp.asarray(S), NamedSharding(mesh, P(None, "x", "y")))
    T_x, T_y, _ = downfold.build_transfer(
        S_j, S_j, mesh, rcond=rcond, announce=False)
    bad = T_x.at[:, 0, 1].add(0.05)
    W_S = downfold.congruence(mesh, bad, T_y)(W_L)
    got = np.asarray(jax.device_get(W_S))
    want = np.asarray(jax.device_get(W_L))
    assert np.max(np.abs(got - want)) > 1e-4 * np.max(np.abs(want))


def test_the_congruence_is_T_W_Tdagger_and_not_its_conjugate():
    """Pin the congruence's own conjugation with a COMPLEX, non-Hermitian T.

    The identity gate cannot see this: at T = I, ``T W T-dagger`` and
    ``conj(T) W conj(T)-dagger`` are the same matrix, and so are they for
    any real perturbation of it.  Only a complex T separates them, and the
    whole downfold hangs off which one the primitive computes.
    """
    mesh = resolve_mesh()
    rng = np.random.default_rng(5)
    nq, m_s, m_l = 3, 4, 8
    T = (rng.normal(size=(nq, m_s, m_l))
         + 1j * rng.normal(size=(nq, m_s, m_l)))
    a = rng.normal(size=(nq, m_l, m_l)) + 1j * rng.normal(size=(nq, m_l, m_l))
    W = a + np.conj(np.swapaxes(a, 1, 2))
    T_x = jax.lax.with_sharding_constraint(
        jnp.asarray(T), NamedSharding(mesh, P(None, None, "x")))
    T_y = jax.lax.with_sharding_constraint(
        jnp.asarray(T), NamedSharding(mesh, P(None, None, "y")))
    W_j = jax.lax.with_sharding_constraint(
        jnp.asarray(W), NamedSharding(mesh, P(None, "x", "y")))
    got = np.asarray(jax.device_get(
        downfold.congruence(mesh, T_x, T_y)(W_j)))
    want = np.einsum("qim,qmn,qjn->qij", T, W, np.conj(T))
    wrong = np.einsum("qim,qmn,qjn->qij", np.conj(T), W, T)
    scale = np.max(np.abs(want))
    assert np.max(np.abs(got - want)) < 1e-12 * scale
    assert np.max(np.abs(got - wrong)) > 0.1 * scale      # the red twin


def test_head_vector_transforms_with_the_conjugate_transfer():
    """``g0_S = conj(T) g0`` — and ``T g0`` must FAIL.

    The q->0 Coulomb head is injected downstream of the bundle as the rank-1
    ``s * conj(g0_mu) * g0_nu`` (``gw.head_correction.apply_q0_head_rank1``),
    so the ONLY correct map for g0 is the one under which the congruence of
    that rank-1 matrix is the same rank-1 form in the small basis.

    THIS CELL EXISTS BECAUSE THE WRONG ONE SHIPPED AND WAS MEASURED.  Using
    ``T g0`` gives a vector of the right size and the wrong phases; the error
    bar never sees it (the head is not in the bundle's V), every shape check
    passes, and on si_bse_debug the lowest exciton came out at 0.211 eV
    instead of 2.347 eV.

    SHAPED ON THE MESH, not on 1×1 — see :func:`_mu_pad`.  ``m_s = 3`` is
    the interesting case (μ_S generically does NOT divide the mesh: the real
    deck selects 185 at 2×2), so it is kept and carried on the padded extent
    the driver itself uses, with the pad rows exactly zero.  Zero rows of T
    contribute zero rows to the congruence and zero entries to ``g0_S``, so
    slicing the logical block off the answer restores the unpadded identity
    exactly rather than approximately.
    """
    mesh = resolve_mesh()
    rng = np.random.default_rng(101)
    nq, m_s, m_l = 2, 3, 6
    m_s_pad, m_l_pad = _mu_pad(m_s), _mu_pad(m_l)
    T = np.zeros((nq, m_s_pad, m_l_pad), dtype=np.complex128)
    T[:, :m_s, :m_l] = (rng.normal(size=(nq, m_s, m_l))
                        + 1j * rng.normal(size=(nq, m_s, m_l)))
    g0 = np.zeros(m_l_pad, dtype=np.complex128)
    g0[:m_l] = rng.normal(size=m_l) + 1j * rng.normal(size=m_l)
    head = np.conj(g0)[:, None] * g0[None, :]
    T_x = jax.lax.with_sharding_constraint(
        jnp.asarray(T), NamedSharding(mesh, P(None, None, "x")))
    T_y = jax.lax.with_sharding_constraint(
        jnp.asarray(T), NamedSharding(mesh, P(None, None, "y")))
    head_L = jax.lax.with_sharding_constraint(
        jnp.asarray(np.broadcast_to(head, (nq, m_l_pad, m_l_pad)).copy()),
        NamedSharding(mesh, P(None, "x", "y")))
    head_S = np.asarray(jax.device_get(
        downfold.congruence(mesh, T_x, T_y)(head_L)))[0][:m_s, :m_s]

    g0_S = np.asarray(jax.device_get(downfold.transform_head_vector(
        jnp.asarray(g0), T_x, 0, mesh)))[:m_s]
    rebuilt = np.conj(g0_S)[:, None] * g0_S[None, :]
    scale = np.max(np.abs(head_S))
    assert np.max(np.abs(rebuilt - head_S)) < 1e-12 * scale

    wrong = np.einsum("mn,n->m", T[0], g0)[:m_s]
    rebuilt_wrong = np.conj(wrong)[:, None] * wrong[None, :]
    assert np.max(np.abs(rebuilt_wrong - head_S)) > 0.1 * scale, (
        "T g0 also reproduced the congruenced head, so this gate is blind "
        "to the conjugation that decides the whole head treatment")


# ---------------------------------------------------------------------------
# 3.  The error bar, and the ridge that must break it
# ---------------------------------------------------------------------------

def _dense_X(psi_y_host, window, q=0):
    """``X[mu, a]`` at one q, a = (m, n, k), built index by index.

    ``X[mu, (m,n,k)] = conj(psi_m(k+q, mu)) psi_n(k, mu)`` — the ISDF
    coefficient matrix itself, in the kernel's own q convention (see
    :func:`_dense_gram`).  At these shapes N = nk*|L|*|R| is a couple of
    hundred, so the N x N observable IS formable and the projector identity
    can be checked the honest way rather than through the trace shortcut it
    exists to justify.
    """
    (bl0, bl1), (br0, br1) = window.left, window.right
    nk, _nb, ns, mu = psi_y_host.shape
    assert ns == 1
    psi = psi_y_host[:, :, 0, :]
    cols = []
    for k in range(nk):
        kq = (k + q) % nk
        for m in range(bl0, bl1):
            for n in range(br0, br1):
                cols.append(np.conj(psi[kq, m, :]) * psi[k, n, :])
    return np.stack(cols, axis=1)               # (mu, N)


def test_pythagorean_error_bar_matches_the_trace_identity():
    """eps_W from mu x mu traces == the projector computation, exactly.

    This certifies the diagnostic itself: the trace form never forms the
    N x N observable, so if it were wrong nothing downstream would notice.
    """
    mesh = resolve_mesh()
    psi_X, psi_Y = _psi(mesh, seed=17)
    win = BandWindow(left=(0, NB), right=(0, NB))
    S_LL = _S_all(mesh, psi_X, psi_Y, win)
    keep = np.array([0, 1, 4, 5], dtype=np.int64)
    rcond = 1e-10
    mu = S_LL.shape[-1]
    rng = np.random.default_rng(19)
    a = rng.normal(size=(NK, mu, mu)) + 1j * rng.normal(size=(NK, mu, mu))
    W_L = a + np.conj(np.swapaxes(a, 1, 2))

    psi_S_X, psi_S_Y = downfold.slice_psi_to_centroids(
        psi_X, psi_Y, keep, len(keep), mesh)
    S_SS = _S_all(mesh, psi_S_X, psi_S_Y, win)
    S_cross = _S_all(mesh, psi_S_X, psi_Y, win)
    T = _transfer_dense(S_SS, S_cross, rcond)
    W_S = np.einsum("qij,qjk,qlk->qil", T, W_L, np.conj(T))

    def _pin(x, spec=P(None, "x", "y")):
        return jax.lax.with_sharding_constraint(
            jnp.asarray(x), NamedSharding(mesh, spec))

    eps, n_L, n_S = downfold.epsilon_w(
        _pin(W_L), _pin(S_LL), _pin(W_S), _pin(S_SS), mesh)

    for q in range(NK):
        want_L = np.trace(
            W_L[q] @ S_LL[q] @ W_L[q].conj().T @ S_LL[q]).real
        want_S = np.trace(
            W_S[q] @ S_SS[q] @ W_S[q].conj().T @ S_SS[q]).real
        assert abs(n_L[q] - want_L) < 1e-8 * abs(want_L)
        assert abs(n_S[q] - want_S) < 1e-8 * abs(want_S)
        assert abs(eps[q] - np.sqrt(max(0.0, 1.0 - want_S / want_L))) < 1e-10
    # A genuine compression 8 -> 4 loses something; a zero error bar here
    # would mean the diagnostic is not measuring the projection at all.
    assert np.all(eps > 1e-6)


def _pythagoras_terms(mesh, ridge_rel):
    """``(||W||^2, ||W_S||^2, ||W - W_S||^2)`` on the DENSE N x N observable.

    Nothing here uses the trace shortcut: the observable is formed, the
    residual is subtracted, and the three Frobenius norms are read straight
    off.  ``ridge`` perturbs S_SS before the solve.
    """
    psi_X, psi_Y = _psi(mesh, seed=23)
    win = BandWindow(left=(0, NB), right=(0, NB))
    S_LL = np.asarray(_S_all(mesh, psi_X, psi_Y, win))[0]
    keep = np.array([0, 1, 4, 5], dtype=np.int64)
    X = _dense_X(np.asarray(jax.device_get(psi_Y)), win, q=0)
    mu = S_LL.shape[-1]
    rng = np.random.default_rng(29)
    a = rng.normal(size=(mu, mu)) + 1j * rng.normal(size=(mu, mu))
    W_L = a + np.conj(a.T)

    S_kk = S_LL[np.ix_(keep, keep)]
    S_SS = S_kk + ridge_rel * float(np.mean(np.real(np.diag(S_kk)))) * np.eye(
        len(keep))
    S_cross = S_LL[keep, :]
    T = _transfer_dense(S_SS[None], S_cross[None], 1e-10)[0]
    W_S = T @ W_L @ T.conj().T

    obs = X.conj().T @ W_L @ X
    obs_S = X[keep].conj().T @ W_S @ X[keep]
    return (float(np.linalg.norm(obs) ** 2),
            float(np.linalg.norm(obs_S) ** 2),
            float(np.linalg.norm(obs - obs_S) ** 2))


def test_pythagoras_holds_on_the_dense_observable():
    """``||W||^2 - ||W_S||^2 == ||W - W_S||^2`` — the projector identity.

    This is what makes the trace-form error bar an EXACT statement rather
    than an estimate, and it is checked here on the N x N observable that
    the trace form exists to avoid ever forming.
    """
    n_L, n_S, n_R = _pythagoras_terms(resolve_mesh(), 0.0)
    assert abs((n_L - n_S) - n_R) < 1e-8 * n_L
    assert n_R > 1e-6 * n_L, "an 8 -> 4 compression lost nothing at all"


def test_pythagoras_red_twin_a_ridge_breaks_it():
    """RED TWIN: ridge S_SS and Pythagoras must FAIL.

    The identity holds because the fit is an ORTHOGONAL projection.  A ridge
    destroys that and takes the identity with it — which is the executable
    form of the argument for why no ridge is applied anywhere on this path.
    A silently ridged solve would leave the driver printing a plausible
    ``eps_W`` that means nothing.
    """
    n_L, n_S, n_R = _pythagoras_terms(resolve_mesh(), 0.1)
    assert abs((n_L - n_S) - n_R) > 1e-4 * n_L, (
        "a ridged fit still satisfied Pythagoras, so the identity is not "
        "testing the projection")


# ---------------------------------------------------------------------------
# 4.  THE KNOB TRAP — mu_S is validated against the EIGENVALUE rank
# ---------------------------------------------------------------------------

def test_mu_small_above_the_eigenvalue_rank_is_refused():
    """The rank refusal fires, and the message carries the measured ceiling.

    Built on a deliberately rank-deficient pool: the psi at four of the
    eight centroids is a copy of the psi at the other four, so the window
    holds at most four independent pair-density directions no matter how
    many points are offered.  Asking for eight must refuse, and must say
    four.
    """
    mesh = resolve_mesh()
    rng = np.random.default_rng(31)
    half = (rng.normal(size=(NK, NB, NS, 4))
            + 1j * rng.normal(size=(NK, NB, NS, 4)))
    y = np.concatenate([half, half], axis=-1)
    psi_Y = jax.lax.with_sharding_constraint(
        jnp.asarray(y), NamedSharding(mesh, P(None, None, None, "y")))
    psi_X = jax.lax.with_sharding_constraint(
        jnp.conj(psi_Y).transpose(0, 3, 1, 2),
        NamedSharding(mesh, P(None, "x", None, None)))
    win = BandWindow(left=(0, NB), right=(0, NB))
    S = downfold.pair_density_gram(psi_X, psi_Y, win, kgrid=KGRID,
                                   mesh_xy=mesh)
    S_q0 = jax.lax.with_sharding_constraint(
        S[0], NamedSharding(mesh, P("x", "y")))
    with pytest.raises(ValueError) as exc:
        downfold.select_cur_centroids(
            S_q0, 8, rcond=1e-8, select_tol=None, mesh_xy=mesh,
            mu_large_logical=MU, print_fn=lambda *a: None)
    msg = str(exc.value)
    assert "only 4" in msg or "holds only 4" in msg, msg
    assert "mu_small = auto" in msg
    assert "R19" in msg


def test_mu_small_auto_lands_on_the_certified_rank():
    """RED TWIN of the refusal: at ``auto`` the same pool does NOT refuse."""
    mesh = resolve_mesh()
    rng = np.random.default_rng(31)
    half = (rng.normal(size=(NK, NB, NS, 4))
            + 1j * rng.normal(size=(NK, NB, NS, 4)))
    y = np.concatenate([half, half], axis=-1)
    psi_Y = jax.lax.with_sharding_constraint(
        jnp.asarray(y), NamedSharding(mesh, P(None, None, None, "y")))
    psi_X = jax.lax.with_sharding_constraint(
        jnp.conj(psi_Y).transpose(0, 3, 1, 2),
        NamedSharding(mesh, P(None, "x", None, None)))
    win = BandWindow(left=(0, NB), right=(0, NB))
    S = downfold.pair_density_gram(psi_X, psi_Y, win, kgrid=KGRID,
                                   mesh_xy=mesh)
    S_q0 = jax.lax.with_sharding_constraint(
        S[0], NamedSharding(mesh, P("x", "y")))
    keep, rep = downfold.select_cur_centroids(
        S_q0, "auto", rcond=1e-8, select_tol=None, mesh_xy=mesh,
        mu_large_logical=MU, print_fn=lambda *a: None)
    assert rep.mu_small == 4 == len(keep)
    assert rep.requested_auto


def _rank_deficient_selection(mu_small, printed):
    """The rank-deficient pool of the two cells above, with prints captured.

    Eight centroids carrying four independent directions, so ``auto`` lands
    on 4 and an explicit 4 is legal — the two arms differ ONLY in how μ_S
    was spelled, which is the whole point of the pair below.
    """
    mesh = resolve_mesh()
    rng = np.random.default_rng(31)
    half = (rng.normal(size=(NK, NB, NS, 4))
            + 1j * rng.normal(size=(NK, NB, NS, 4)))
    y = np.concatenate([half, half], axis=-1)
    psi_Y = jax.lax.with_sharding_constraint(
        jnp.asarray(y), NamedSharding(mesh, P(None, None, None, "y")))
    psi_X = jax.lax.with_sharding_constraint(
        jnp.conj(psi_Y).transpose(0, 3, 1, 2),
        NamedSharding(mesh, P(None, "x", None, None)))
    win = BandWindow(left=(0, NB), right=(0, NB))
    S = downfold.pair_density_gram(psi_X, psi_Y, win, kgrid=KGRID,
                                   mesh_xy=mesh)
    S_q0 = jax.lax.with_sharding_constraint(
        S[0], NamedSharding(mesh, P("x", "y")))
    return downfold.select_cur_centroids(
        S_q0, mu_small, rcond=1e-8, select_tol=None, mesh_xy=mesh,
        mu_large_logical=MU, print_fn=lambda *a: printed.append(" ".join(
            str(x) for x in a)))


def test_auto_prints_the_loud_accuracy_warning():
    """``mu_small = auto`` must SAY that it sized by rank, not by accuracy.

    The measured reason this cell exists (``PIPELINE_HEALTH.md``,
    2026-08-10): following this tree's own guidance end to end, `auto` on a
    936-centroid si parent produced a bundle whose lowest BSE eigenvalue was
    2.087 eV wrong — 2.3449 eV down to 0.2579 eV — with ``eps_W`` reading
    1.33e-2 and NOTHING anywhere refusing.  The driver still runs, because
    ``auto`` is an explicit user choice; what it may not do is stay quiet.
    """
    printed = []
    _keep, rep = _rank_deficient_selection("auto", printed)
    assert rep.requested_auto
    blob = "\n".join(printed)
    assert downfold.AUTO_HAZARD in blob, blob
    assert "WARNING" in blob and "NOT BY ACCURACY" in blob, blob
    # The measurement travels with the warning, or a reader has to take it
    # on faith.
    for needle in ("PIPELINE_HEALTH.md", "2.3449", "0.2579", "1.33e-2",
                   "624", "target-accuracy"):
        assert needle in blob, f"{needle!r} missing from the auto warning"


def test_auto_warning_red_twin_an_explicit_mu_small_says_nothing():
    """RED TWIN: the same pool, μ_S spelled as an integer, prints no warning.

    A warning that fires on every run is a warning nobody reads.  The user
    who typed a number has already made the sizing decision the block exists
    to interrupt, so the block must not fire — and the two cells differ in
    exactly one argument.
    """
    printed = []
    _keep, rep = _rank_deficient_selection(4, printed)
    assert not rep.requested_auto
    assert rep.mu_small == 4
    blob = "\n".join(printed)
    assert downfold.AUTO_HAZARD not in blob, blob
    assert "WARNING" not in blob, blob
    assert "!!!!" not in blob, blob


def test_the_two_tolerances_are_different_knobs():
    """The selection certificate and the eigenvalue rank must DISAGREE.

    ``DOWNFOLD_RANK_PROBE.md`` §7: pivoted Cholesky stops on a residual
    Schur diagonal and the truncation stops on an eigenvalue, and the
    residual decays much more slowly than the spectrum — measured, about 3x
    the rank at the same nominal number.  If this cell ever passed by the
    two agreeing, the driver's side-by-side report would be theatre.
    """
    rng = np.random.default_rng(37)
    n = 64
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    U, _, Vh = np.linalg.svd(A)
    lam = 10.0 ** (-np.arange(n) / 4.0)            # smooth, no knee
    G = (U * lam) @ U.conj().T
    G = 0.5 * (G + G.conj().T)
    del Vh
    tol = 1e-6
    eig_rank = downfold.rank_criterion.select_rank(
        np.linalg.eigvalsh(G), tol)
    from centroid.pivoted_cholesky import pivoted_cholesky_select
    _piv, _L, pc_rank, *_ = pivoted_cholesky_select(
        jnp.asarray(G), n, tol_rel=tol)
    assert int(pc_rank) > int(eig_rank), (
        f"selection rank {int(pc_rank)} did not exceed the eigenvalue rank "
        f"{int(eig_rank)} at the same nominal tolerance {tol}")


# ---------------------------------------------------------------------------
# 5.  The input file: refusals that must fire
# ---------------------------------------------------------------------------

def _write_input(tmp_path, body):
    p = tmp_path / "downfold.in"
    p.write_text("[downfold]\n" + body)
    return str(p)


_MINIMAL = ("source_restart = parent\noutput_restart = small\n"
            "n_val = 4\nn_cond = 4\nmu_small = 16\n")


def test_input_file_minimal_parses():
    import tempfile
    from gw.downfold_config import DownfoldConfig
    with tempfile.TemporaryDirectory() as d:
        import pathlib
        cfg = DownfoldConfig.from_input_file(
            _write_input(pathlib.Path(d), _MINIMAL))
    assert cfg.band_range_left == (0, 4)
    assert cfg.band_range_right == (4, 8)
    assert cfg.mu_small == 16
    assert cfg.downfold_rcond == downfold.DEFAULT_RCOND
    assert cfg.mode == "cur"


@pytest.mark.parametrize("body,needle", [
    (_MINIMAL + "downfold_rcnod = 1e-6\n", "does not have"),
    (_MINIMAL + "mode = refit\n", "not implemented in stage 1"),
    (_MINIMAL + "plan = distributed\n", "later stage"),
    (_MINIMAL + "band_range_left = 0:8\nband_range_right = 0:8\n",
     "given TWICE"),
    ("source_restart = a\noutput_restart = b\nmu_small = 4\n",
     "no retained band window"),
    ("source_restart = a\noutput_restart = b\nn_val = 4\nn_cond = 4\n",
     "'mu_small' has no default"),
    ("source_restart = a\noutput_restart = a\nn_val = 4\nn_cond = 4\n"
     "mu_small = 4\n", "same path"),
    (_MINIMAL + "downfold_rcond = 12.0\n", "must lie in (0, 1)"),
])
def test_input_file_refusals(tmp_path, body, needle):
    from gw.downfold_config import DownfoldConfig
    with pytest.raises(ValueError) as exc:
        DownfoldConfig.from_input_file(_write_input(tmp_path, body))
    assert needle in str(exc.value), str(exc.value)


def test_input_file_refuses_a_gw_deck(tmp_path):
    from gw.downfold_config import DownfoldConfig
    p = tmp_path / "cohsex.in"
    p.write_text("[cohsex]\nnval = 8\nncond = 52\n")
    with pytest.raises(ValueError) as exc:
        DownfoldConfig.from_input_file(str(p))
    assert "not a downfold input file" in str(exc.value)


# ---------------------------------------------------------------------------
# 6.  §q_irr — the downfold on the WEDGE, and whether its child unfolds
# ---------------------------------------------------------------------------
#
# THE CLAIM UNDER TEST, stated so a reader knows what a green here buys.
# ``gw.downfold``'s D2b block derives that the wedge downfold's child is
# storable on the wedge iff the kept centroid set is orbit-closed.  These
# cells test the two halves separately and then together:
#
#   (a) q-DIAGONALITY — the wedge transfer is the full-BZ transfer's own row,
#       BIT-FOR-BIT.  No symmetry needed; it is why a wedge run is a
#       restriction rather than an approximation, and it is what licenses
#       downfold_run transporting a q-IBZ ζ against a full-BZ T.
#   (b) COVARIANCE — congruence commutes with the tree's OWN unfold kernel
#       when keep is orbit-closed, and the child's tables are the parent's
#       restricted.  Run through ``symmetry_maps.unfold_isdf_operator``, not
#       a reimplementation of it, so a green is about the shipping unfold.
#   (c) THE RED TWIN the whole section exists for: a selection that is NOT
#       orbit-closed must be CAUGHT, and — separately — must actually be
#       WRONG if it were let through.  A refusal nobody has seen guard a real
#       disagreement is decoration.
#
# The parent here is SYNTHETIC AND EXACTLY COVARIANT BY CONSTRUCTION: the
# full-BZ operands are DEFINED as the unfold of a wedge block.  That is the
# honest reference, because the property under test is the commutation of two
# maps, not the physics that makes a pair-density Gram covariant in the first
# place (which is the parent's own wedge-storage precondition, measured at
# write time by ``verify_centroid_orbit_closure`` and not re-litigated here).

QIRR_MU = 12
QIRR_ORB = 4                       # 3 orbits of 4 under a cyclic group
QIRR_NQ_FULL = 4
QIRR_IRR_IDX = np.array([0, 1, 1, 1], dtype=np.int64)
QIRR_SYM_IDX = np.array([0, 0, 1, 2], dtype=np.int64)
QIRR_Q_IRR_FRAC = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
#: keep two WHOLE orbits — the covariance condition, satisfied.
QIRR_KEEP_CLOSED = np.array([0, 1, 2, 3, 8, 9, 10, 11], dtype=np.int64)
#: the same size, but cutting through both orbits — the red twin.
QIRR_KEEP_BROKEN = np.array([0, 1, 2, 4, 8, 9, 10, 11], dtype=np.int64)


def _qirr_tables(seed=5):
    """``(sym_perm, L_table)`` for a cyclic group with row 0 the IDENTITY.

    Row 0 must be the identity op with zero wrap, because that is what makes
    the ``sym_idx == 0`` full-BZ q's the wedge blocks themselves and turns
    gate (a) into a bit-for-bit statement instead of a tolerance.
    """
    n_orb = QIRR_MU // QIRR_ORB
    perm = np.empty((QIRR_ORB, QIRR_MU), dtype=np.int64)
    for g in range(QIRR_ORB):
        for o in range(n_orb):
            base = o * QIRR_ORB
            for j in range(QIRR_ORB):
                perm[g, base + j] = base + (j + g) % QIRR_ORB
    rng = np.random.default_rng(seed)
    L = rng.integers(-1, 2, size=(QIRR_ORB, QIRR_MU, 3)).astype(np.int64)
    L[0] = 0
    assert np.array_equal(perm[0], np.arange(QIRR_MU))
    return perm, L


def _qirr_unfold(A_ibz, mesh, sym_perm, L_table):
    """The tree's own unfold, on this section's tables."""
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import unfold_isdf_operator
    return unfold_isdf_operator(
        A_ibz, irr_idx=jnp.asarray(QIRR_IRR_IDX),
        sym_idx=jnp.asarray(QIRR_SYM_IDX), sym_perm=jnp.asarray(sym_perm),
        L_table=jnp.asarray(L_table),
        q_irr_frac=jnp.asarray(QIRR_Q_IRR_FRAC), mesh_xy=mesh,
        n_sym_spatial=QIRR_ORB)


def _qirr_parent(mesh, seed=3):
    """A wedge parent ``(S_ibz, W_ibz)`` and its unfold, both on device."""
    rng = np.random.default_rng(seed)
    n_ibz = QIRR_Q_IRR_FRAC.shape[0]
    a = (rng.normal(size=(n_ibz, QIRR_MU, QIRR_MU))
         + 1j * rng.normal(size=(n_ibz, QIRR_MU, QIRR_MU)))
    # S: Hermitian POSITIVE DEFINITE (it is a Gram) with a well-separated
    # spectrum, so the rank truncation retains the same count on both routes
    # and the comparison is of numbers rather than of two different ranks.
    S_ibz = np.einsum("qij,qkj->qik", a, a.conj()) + QIRR_MU * np.eye(QIRR_MU)
    b = (rng.normal(size=(n_ibz, QIRR_MU, QIRR_MU))
         + 1j * rng.normal(size=(n_ibz, QIRR_MU, QIRR_MU)))
    W_ibz = b + np.conj(np.swapaxes(b, 1, 2))
    munu = NamedSharding(mesh, P(None, "x", "y"))
    S_j = jax.lax.with_sharding_constraint(jnp.asarray(S_ibz), munu)
    W_j = jax.lax.with_sharding_constraint(jnp.asarray(W_ibz), munu)
    return S_j, W_j


def _qirr_downfold(S_all, W_all, keep, mesh, rcond=1e-10):
    """One downfold, on whatever q set its operands carry.

    Deliberately the PRODUCTION primitives — ``build_transfer`` and
    ``congruence`` — so that both routes below differ in their q set and in
    nothing else.
    """
    munu = NamedSharding(mesh, P(None, "x", "y"))
    S_host = np.asarray(jax.device_get(S_all))
    S_SS = jax.lax.with_sharding_constraint(
        jnp.asarray(S_host[:, keep, :][:, :, keep]), munu)
    S_cross = jax.lax.with_sharding_constraint(
        jnp.asarray(S_host[:, keep, :]), munu)
    T_x, T_y, reports = downfold.build_transfer(
        S_SS, S_cross, mesh, rcond=rcond, announce=False)
    W_S = downfold.congruence(mesh, T_x, T_y)(W_all)
    return T_x, W_S, [int(r.rank_criterion) for r in reports]


def test_qirr_the_wedge_transfer_is_the_full_bz_transfer_bit_for_bit():
    """(a) q-DIAGONALITY.  T[q] uses no q but its own.

    At the full-BZ q whose ``sym_idx`` is 0 the unfold is the identity map
    (identity permutation, zero wrap, unit phase), so the full-BZ route's
    operands at that q ARE the wedge block.  Same inputs, same eigh, same
    bits — asserted as equality and not as a tolerance, because anything
    weaker would also pass if the transfer had picked up a dependence on the
    surrounding q axis.

    THIS is the licence for ``downfold_run.write_downfolded_zeta`` to
    transport a q-IBZ ζ against a transfer built on the full BZ: the row it
    reads is the matrix a wedge-native run would have computed.
    """
    mesh = resolve_mesh()
    sym_perm, L_table = _qirr_tables()
    S_ibz, W_ibz = _qirr_parent(mesh)
    S_full = _qirr_unfold(S_ibz, mesh, sym_perm, L_table)
    W_full = _qirr_unfold(W_ibz, mesh, sym_perm, L_table)
    keep = QIRR_KEEP_CLOSED

    T_ibz, _W_S_ibz, rank_ibz = _qirr_downfold(S_ibz, W_ibz, keep, mesh)
    T_full, _W_S_full, rank_full = _qirr_downfold(S_full, W_full, keep, mesh)

    t_i = np.asarray(jax.device_get(T_ibz))
    t_f = np.asarray(jax.device_get(T_full))
    # the full-BZ q that carries each wedge block untouched
    for i in range(QIRR_Q_IRR_FRAC.shape[0]):
        q = int(np.flatnonzero((QIRR_IRR_IDX == i) & (QIRR_SYM_IDX == 0))[0])
        assert np.array_equal(t_i[i], t_f[q]), (
            f"wedge transfer i={i} differs from the full-BZ transfer at its "
            f"own q={q}: the downfold has acquired a dependence on the q set "
            f"it was handed, and a wedge run is then not a restriction of a "
            f"full-BZ one")
        assert rank_ibz[i] == rank_full[q]


def test_qirr_covariance_the_wedge_child_unfolds_to_the_full_bz_child():
    """(b) THE DEFINITIVE GATE.  Congruence commutes with the unfold.

    Downfold the wedge, unfold the CHILD with the child's own tables, and
    compare against downfolding the already-unfolded parent.  Both routes run
    the shipping ``build_transfer`` / ``congruence`` / ``unfold_isdf_operator``.

    NOT ASSERTED BIT-FOR-BIT, and the reason is stated rather than absorbed:
    the two routes apply the same unitary at different points of the same
    chain (before the eigh on one side, after the congruence on the other),
    so they differ by floating-point REASSOCIATION — plus, at every q whose
    op is not the identity, a multiply by a unit-modulus phase and later by
    its conjugate, which is exact only in exact arithmetic.  The band below
    is that floor, measured; a DIFFERENT ALGEBRA would miss it by order one,
    which is what the red twin two cells down demonstrates.
    """
    mesh = resolve_mesh()
    sym_perm, L_table = _qirr_tables()
    S_ibz, W_ibz = _qirr_parent(mesh)
    S_full = _qirr_unfold(S_ibz, mesh, sym_perm, L_table)
    W_full = _qirr_unfold(W_ibz, mesh, sym_perm, L_table)
    keep = QIRR_KEEP_CLOSED

    stab = downfold.star_stability(keep, sym_perm)
    assert stab.closed, stab.describe()
    child_perm, child_L = downfold.child_unfold_tables(
        keep, sym_perm, L_table)

    _T_i, W_S_ibz, _r = _qirr_downfold(S_ibz, W_ibz, keep, mesh)
    _T_f, W_S_full, _r2 = _qirr_downfold(S_full, W_full, keep, mesh)
    W_S_unfolded = _qirr_unfold(W_S_ibz, mesh, child_perm, child_L)

    got = np.asarray(jax.device_get(W_S_unfolded))
    want = np.asarray(jax.device_get(W_S_full))
    assert got.shape == want.shape == (QIRR_NQ_FULL, keep.size, keep.size)
    rel = np.max(np.abs(got - want)) / np.max(np.abs(want))
    assert rel < 1e-12, (
        f"the q_irr-downfolded child does not unfold to the full-BZ "
        f"downfolded child (max rel {rel:.3e}).  Congruence and unfolding "
        f"have stopped commuting — check the child tables before the algebra")


def test_qirr_red_twin_a_symmetry_broken_selection_is_refused():
    """(c1) RED TWIN.  A selection that cuts an orbit must be CAUGHT.

    ``QIRR_KEEP_BROKEN`` is the same SIZE as the closed set and takes 3 of
    orbit 0's 4 members plus one of orbit 1's, so nothing about a shape or a
    count distinguishes it.  ``child_unfold_tables`` must refuse it by name.
    """
    sym_perm, L_table = _qirr_tables()
    stab = downfold.star_stability(QIRR_KEEP_BROKEN, sym_perm)
    assert not stab.closed
    assert stab.violating_ops, "a broken selection with no violating op"
    with pytest.raises(ValueError) as exc:
        downfold.child_unfold_tables(QIRR_KEEP_BROKEN, sym_perm, L_table)
    assert "orbit-closed" in str(exc.value)


def test_qirr_red_twin_a_symmetry_broken_selection_really_is_wrong():
    """(c2) THE REFUSAL IS LOAD-BEARING, not decoration.

    Force the broken selection through with the closure check bypassed — by
    handing the child unfold a table built the way an unguarded
    implementation would build it, ``pos[]`` of the images that DO land
    inside the kept set and an arbitrary slot for the ones that do not.  The
    result must miss the full-BZ answer by order one.  A refusal that only
    ever guarded agreeing numbers would be worth deleting.
    """
    mesh = resolve_mesh()
    sym_perm, L_table = _qirr_tables()
    S_ibz, W_ibz = _qirr_parent(mesh)
    S_full = _qirr_unfold(S_ibz, mesh, sym_perm, L_table)
    W_full = _qirr_unfold(W_ibz, mesh, sym_perm, L_table)
    keep = QIRR_KEEP_BROKEN

    pos = np.full(QIRR_MU, 0, dtype=np.int64)     # the arbitrary slot
    pos[keep] = np.arange(keep.size, dtype=np.int64)
    child_perm = pos[sym_perm[:, keep]]
    child_L = L_table[:, keep]

    _T_i, W_S_ibz, _r = _qirr_downfold(S_ibz, W_ibz, keep, mesh)
    _T_f, W_S_full, _r2 = _qirr_downfold(S_full, W_full, keep, mesh)
    got = np.asarray(jax.device_get(
        _qirr_unfold(W_S_ibz, mesh, child_perm, child_L)))
    want = np.asarray(jax.device_get(W_S_full))
    rel = np.max(np.abs(got - want)) / np.max(np.abs(want))
    assert rel > 0.1, (
        f"a selection that breaks star symmetry unfolded to within "
        f"{rel:.3e} of the right answer — if that is real, the closure "
        f"refusal is guarding nothing and should be reconsidered, not kept")


def test_qirr_the_cur_selection_is_not_star_stable_by_default():
    """THE MEASUREMENT the fix rests on: CUR breaks the star, generically.

    On a Gram that commutes with the whole group — which is what the q = 0
    selection Gram IS, since q = 0 is invariant under every op and the unfold
    phases are unity there — the pivot order fills orbits GREEDILY but stops
    at exactly ``mu_S``.  So closure holds only when ``mu_S`` lands on an
    orbit boundary, and a user-chosen ``mu_S`` generally does not.

    Pinned here because the whole q_irr storage decision turns on it, and
    because "the selection is symmetry-stable" is exactly the kind of thing
    that gets assumed once and inherited forever.
    """
    mesh = resolve_mesh()
    sym_perm, _L = _qirr_tables()
    rng = np.random.default_rng(19)
    A = (rng.normal(size=(QIRR_MU, QIRR_MU))
         + 1j * rng.normal(size=(QIRR_MU, QIRR_MU)))
    M = A @ A.conj().T
    G = sum(M[np.ix_(p, p)] for p in sym_perm)
    G = 0.5 * (G + G.conj().T)
    for p in sym_perm:                     # the symmetry we are selecting on
        assert np.max(np.abs(G[np.ix_(p, p)] - G)) < 1e-10
    G_j = jax.lax.with_sharding_constraint(
        jnp.asarray(G), NamedSharding(mesh, P("x", "y")))

    closed_at, broken_at = [], []
    for mu_S in range(2, QIRR_MU):
        keep, _rep = downfold.select_cur_centroids(
            G_j, mu_S, rcond=1e-10, select_tol=1e-12, mesh_xy=mesh,
            mu_large_logical=QIRR_MU, print_fn=lambda *a, **k: None)
        (closed_at if downfold.star_stability(keep, sym_perm).closed
         else broken_at).append(mu_S)
    assert broken_at, (
        "every mu_S came back orbit-closed — if that is reproducible on real "
        "decks the covariance condition is free and this section's repair is "
        "unnecessary; do not silently drop the check, measure it and say so")
    assert closed_at, "no mu_S was orbit-closed; the construction is wrong"
    # On THIS synthetic, closure lands on orbit boundaries — q=0 gives every
    # member of an orbit the same Schur diagonal and the tie-break is index
    # order, so the pivot order fills orbits in turn.  That is a property of
    # the construction and NOT of production Grams: on si_bse_debug the
    # closed sizes would be the seven multiples of 24 and the real pivot
    # order lands on none of them (0 of 185).
    assert all(m % QIRR_ORB == 0 for m in closed_at), closed_at


def _qirr_selection_gram(sym_perm, seed=19):
    """A q=0 selection Gram that genuinely commutes with the whole group.

    Which is what the real one IS — q = 0 is invariant under every op and the
    unfold phases are unity there — and it is why every member of an orbit
    presents pivoted Cholesky with the same residual diagonal.
    """
    rng = np.random.default_rng(seed)
    A = (rng.normal(size=(QIRR_MU, 3 * QIRR_MU))
         + 1j * rng.normal(size=(QIRR_MU, 3 * QIRR_MU)))
    M = A @ A.conj().T
    G = sum(M[np.ix_(p, p)] for p in sym_perm)
    G = 0.5 * (G + G.conj().T)
    for p in sym_perm:
        assert np.max(np.abs(G[np.ix_(p, p)] - G)) < 1e-9
    return G


@pytest.mark.parametrize("mu_S_request", [7, 9, 11])
def test_qirr_THE_COMPOSITION_an_orbit_floored_selection_gives_a_storable_child(
        mu_S_request):
    """(d) THE COMPOSITION.  The covariance gate, on the SHIPPING selection.

    Every cell above this one hands the covariance route a keep set chosen by
    hand (``QIRR_KEEP_CLOSED``).  That measured whether congruence and
    unfolding commute; it could not measure whether the selection this driver
    actually makes produces a keep set they commute ON, and the answer used to
    be no — 0 of 185 admissible mu_S orbit-closed on ``si_bse_debug``.

    This cell closes that gap: the keep set comes from
    ``select_cur_centroids`` in orbit mode, at a REQUEST that falls between
    rungs, and the wedge child unfolded with the child's own tables must
    reproduce the full-BZ child.  A green here is the statement "wedge
    children are storable at the selection the driver makes", which is the
    composition the orbit floor buys.
    """
    mesh = resolve_mesh()
    sym_perm, L_table = _qirr_tables()
    G = _qirr_selection_gram(sym_perm)
    G_j = jax.lax.with_sharding_constraint(
        jnp.asarray(G), NamedSharding(mesh, P("x", "y")))

    keep, rep = downfold.select_cur_centroids(
        G_j, mu_S_request, rcond=1e-10, select_tol=1e-12, mesh_xy=mesh,
        mu_large_logical=QIRR_MU, print_fn=lambda *a, **k: None,
        sym_perm=sym_perm)
    assert keep.size <= mu_S_request, "the floor overran the point budget"
    assert keep.size < mu_S_request or mu_S_request % QIRR_ORB == 0
    assert rep.star is not None and rep.star.closed, (
        "the orbit-mode selection is not orbit-closed, so there is no "
        "composition to test — the kernel's orbit_id mode is not doing what "
        "its docstring promises")

    child_perm, child_L = downfold.child_unfold_tables(keep, sym_perm, L_table)
    S_ibz, W_ibz = _qirr_parent(mesh)
    S_full = _qirr_unfold(S_ibz, mesh, sym_perm, L_table)
    W_full = _qirr_unfold(W_ibz, mesh, sym_perm, L_table)
    _T_i, W_S_ibz, _r = _qirr_downfold(S_ibz, W_ibz, keep, mesh)
    _T_f, W_S_full, _r2 = _qirr_downfold(S_full, W_full, keep, mesh)
    got = np.asarray(jax.device_get(
        _qirr_unfold(W_S_ibz, mesh, child_perm, child_L)))
    want = np.asarray(jax.device_get(W_S_full))
    assert got.shape == want.shape == (QIRR_NQ_FULL, keep.size, keep.size)
    rel = np.max(np.abs(got - want)) / np.max(np.abs(want))
    assert rel < 1e-12, (
        f"the child of an ORBIT-FLOORED selection (request {mu_S_request} "
        f"points -> realized {keep.size}) does not unfold to the full-BZ "
        f"child (max rel {rel:.3e}).  The floor is supposed to make this "
        f"structural; if it does not, the wedge child is not storable and "
        f"the composition claim is false")


def test_qirr_RED_TWIN_the_point_granular_selection_still_cuts_orbits():
    """The floor is load-bearing: without it, the SAME mu_S is not closed.

    A cell that only ever ran the orbit path would show the composition
    working and say nothing about what the floor bought.  This is the control
    arm: the same Gram, the same mu_S, ``sym_perm=None``, and the delivered
    set must break closure — so ``child_unfold_tables`` refuses it and the
    child of a point-granular selection has no tables at all.
    """
    mesh = resolve_mesh()
    sym_perm, L_table = _qirr_tables()
    G = _qirr_selection_gram(sym_perm)
    G_j = jax.lax.with_sharding_constraint(
        jnp.asarray(G), NamedSharding(mesh, P("x", "y")))
    broken = []
    for mu_S in (7, 9, 11):
        keep, rep = downfold.select_cur_centroids(
            G_j, mu_S, rcond=1e-10, select_tol=1e-12, mesh_xy=mesh,
            mu_large_logical=QIRR_MU, print_fn=lambda *a, **k: None)
        assert keep.size == mu_S
        assert rep.star is None, (
            "closure was reported without a sym_perm — an absence must not "
            "be dressed as a measurement")
        if not downfold.star_stability(keep, sym_perm).closed:
            broken.append(mu_S)
            with pytest.raises(ValueError, match="orbit-closed"):
                downfold.child_unfold_tables(keep, sym_perm, L_table)
    assert broken, (
        "no point-granular selection broke closure on this construction, so "
        "the orbit floor bought nothing here and this section's claim is "
        "untested — measure it on a real deck before believing it")


def test_qirr_orbit_completion_restores_closure_at_a_bounded_cost():
    """The repair, and its price.  Completion ADDS; it never drops.

    Dropping the offending centroids instead would shrink the retained
    subspace below what the selection certified, which is the failure the
    rank refusal exists to prevent — so the repair rounds ``mu_S`` UP, and
    the caller must treat the returned length as the authority.
    """
    sym_perm, L_table = _qirr_tables()
    for mu_S in range(2, QIRR_MU):
        keep = np.arange(mu_S, dtype=np.int64)          # a cutting selection
        closed = downfold.orbit_complete_keep(keep, sym_perm)
        assert set(keep.tolist()) <= set(closed.tolist()), "completion dropped"
        assert downfold.star_stability(closed, sym_perm).closed
        # A PROPERTY OF THIS SYNTHETIC, NOT A GENERAL BOUND.  `keep` here is
        # a prefix of index order against a synthetic whose orbits are index
        # blocks, so completion cannot reach past the block mu_S stopped in.
        # The generalisation this assert used to state in its message — "on a
        # greedy pivot order the cost is the tail of one orbit" — is REFUTED
        # on a production centroid set (si_bse_debug: 0 of 185 admissible
        # mu_S closed, completion 185 -> 480, the whole parent basis;
        # tests/known_failures/2026-08-10-downfold-qirr-star-stability.md).
        # Kept as a cheap structural check on `orbit_complete_keep`, with the
        # claim it is evidence for narrowed to what it actually covers.
        assert closed.size - keep.size < QIRR_ORB, (
            "completion cost a whole group order on the SYNTHETIC, whose "
            "orbits are contiguous index blocks — so an index-order prefix "
            "cannot need more than the block it stopped inside.  This says "
            "nothing about a real pivot order; see the amendment.")
        # and the tables it makes available are the parent's, restricted
        cperm, cL = downfold.child_unfold_tables(closed, sym_perm, L_table)
        assert cperm.shape == (QIRR_ORB, closed.size)
        assert np.array_equal(cL, L_table[:, closed])
        for s in range(QIRR_ORB):
            assert np.array_equal(closed[cperm[s]], sym_perm[s, closed]), (
                "child permutation does not satisfy keep[alpha_S(j)] = "
                "alpha(keep[j]) — the child's unfold would gather the wrong "
                "rows")


# ---------------------------------------------------------------------------
# 6b. §κ — the child-covariance gate's TOLERANCE, and that it still bites
# ---------------------------------------------------------------------------
#
# OWNER RULING, 2026-08-11: *"make it scale if you think that is the most
# likely thing to be more robust to say 100x more atoms and centroids."*
#
# WHAT WENT WRONG WITH THE CONSTANT, because the repair only makes sense
# against it.  ``CHILD_COVARIANCE_TOL = 1e-9`` was chosen "in the empty
# decades between the synthetic floor 1.7e-15 and the red twin 8.6e-01" — on
# the synthetic in section 6, whose transfer solve runs at a condition number
# of order 20.  The quantity the gate measures is a PSEUDO-INVERSE
# perturbation, so its floor is a function of the solve's conditioning, and
# the production deck runs at ``kappa_eff = 8.9e+05``.  Consequence, filed in
# ``tests/known_failures/2026-08-11-downfold-gram-q-sign.md``: after the
# q-sign defect was fixed the deck achieved **3.729e-08**, the gate compared
# that against 1e-9, and printed **REFUTED on a correct result**.
#
# The cells below hold the repair to three things, in order of how easy each
# is to get wrong:
#
#   (i)   the EXPONENT is the one the filed measurements support (0.41), not
#         the one the same row's prose reasons to (2, "second order in the
#         condition number").  A tolerance whose exponent came from theory
#         while the data said otherwise would be a fit to nothing.
#   (ii)  the FLOOR still holds at small kappa, so section 6's own synthetic
#         is gated no more loosely than it was before this ruling.
#   (iii) THE GATE STILL BITES AT PRODUCTION CONDITIONING.  This is the one
#         that matters and it gets a live red twin, not an arithmetic one:
#         the shipped q-sign defect, reproduced at a kappa_eff comparable to
#         the production deck's, run through the DRIVER'S OWN gate function,
#         must still print REFUTED — and by millions, not by a whisker.
#
# WHAT THESE CELLS CANNOT SHOW, stated so nobody reads more into the green
# than is there: the synthetic parent of section 6 is covariant to machine
# precision BY CONSTRUCTION (the full BZ is defined as the unfold of the
# wedge), so its correct arm sits near 1e-11 even at kappa_eff = 7e5 and
# would pass at 1e-9 too.  The scaling's NECESSITY is a property of a real
# pair-density Gram, which carries a 3.1e-10 covariance residual of its own
# for the pinv to amplify; that evidence is the production deck's, it is
# filed, and the cell that reads it here reads it as filed numbers.

#: The production deck's own numbers, quoted from
#: ``tests/known_failures/2026-08-11-downfold-gram-q-sign.md`` (deck
#: ``si_bse_debug``, mu_L 480 -> mu_S 168, ``downfold_rcond`` 1.1e-6, four
#: real GPUs, workspace ``/pscratch/sd/j/jackm/wedgechild_0811/``).
PROD_KAPPA_EFF = 8.945e5
PROD_WORST_REL_FIXED = {"V_qmunu": 3.729e-08, "W0_qmunu": 3.004e-08}
PROD_WORST_REL_BROKEN = {"V_qmunu": 1.170e+00, "W0_qmunu": 1.241e+00}
#: The three-point rcond sweep the exponent is fitted to, same row.
PROD_KAPPA_SWEEP = ((2.0e1, 3.9e-10), (9.9e1, 1.2e-09), (8.9e5, 3.7e-08))


def test_child_covariance_tol_covers_the_filed_kappa_sweep():
    """(i) THE FIT, against the measurements it claims to come from.

    Every point of the filed sweep must sit UNDER the tolerance the formula
    gives at its own kappa — that is what "calibrated to the data" means
    here — and none may sit absurdly far under, because a tolerance three
    decades above every measurement is not a fit, it is an abdication.
    """
    from gw import downfold_run as dr
    for kap, achieved in PROD_KAPPA_SWEEP:
        tol = dr.child_covariance_tol(kap)
        assert achieved < tol, (
            f"the filed floor {achieved:.3e} at kappa_eff {kap:.3e} is ABOVE "
            f"the tolerance {tol:.3e} the formula gives there, so the gate "
            f"still refutes a measurement it was fitted to")
        assert tol / achieved < 1.0e3, (
            f"tol/achieved = {tol / achieved:.3e} at kappa_eff {kap:.3e} — "
            f"the formula has stopped tracking the data it was fitted to")


def test_child_covariance_tol_uses_the_MEASURED_exponent_not_the_theory_one():
    """(i) The exponent is 0.41 from the fit, and NOT 2 from the prose.

    ``tests/known_failures/2026-08-11-downfold-gram-q-sign.md`` reasons that
    "the pinv's perturbation is second order in the condition number" and
    then reports a sweep whose log-log slope is 0.409.  The ruling's design
    constraint is that the exponent comes from the measurements, so this cell
    pins that it did: a decade of kappa must move the tolerance by about
    10**0.41, and emphatically not by 10**2.

    It also pins the DIRECTION of the extrapolation error, which is the part
    the owner's "100x more atoms and centroids" turns on.  The measured local
    slope FALLS with kappa (0.70 across the first decade of the sweep, 0.38
    across the last four), so a single global 0.41 opens the gate slightly
    faster than the data does at large kappa — the safe direction.
    """
    from gw import downfold_run as dr
    lo, hi = dr.child_covariance_tol(1.0e4), dr.child_covariance_tol(1.0e5)
    slope = np.log10(hi / lo)
    assert abs(slope - dr.CHILD_COVARIANCE_KAPPA_EXPONENT) < 1e-9
    assert 0.35 < slope < 0.50, (
        f"a decade of kappa moves the tolerance by 10**{slope:.3f}; the "
        f"filed sweep's least-squares slope is 0.409 and the pairwise "
        f"slopes span 0.38..0.70")
    assert slope < 1.0, (
        "the exponent is at or above 1 — that is the theory sentence, not "
        "the measurement, and at production kappa it would cost the gate "
        "the discrimination the red twin below demands")
    # monotone, so a worse-conditioned run never gets a TIGHTER gate
    kaps = np.geomspace(1.0, 1e12, 40)
    tols = np.array([dr.child_covariance_tol(k) for k in kaps])
    assert np.all(np.diff(tols) >= 0.0)


def test_child_covariance_tol_floors_at_the_old_absolute_value():
    """(ii) THE FLOOR, so a tiny-kappa synthetic is not gated vacuously tight.

    And the fallback, which is the other half of "kappa is a measurement, not
    a knob": handed no kappa, or a NaN, or a nonsense one, the tolerance does
    NOT guess — it returns the floor, and the gate's own verdict line says
    the tolerance was not scaled.  A gate that invented a kappa in order to
    open itself would be the loosening this whole row exists to refuse.
    """
    from gw import downfold_run as dr
    assert dr.CHILD_COVARIANCE_TOL == 1e-9
    for bad in (None, float("nan"), 0.0, -1.0, "not a number"):
        assert dr.child_covariance_tol(bad) == 1e-9, bad
    assert dr.child_covariance_tol(1e-30) == 1e-9
    assert dr.child_covariance_tol(1.0) == 1e-9
    # and it is a FLOOR, not a clamp: it only ever raises
    for k in (1e2, 1e4, 1e6, 1e9):
        assert dr.child_covariance_tol(k) >= 1e-9


def test_the_FIXED_PRODUCTION_DECK_now_passes_and_its_defect_still_does_not():
    """(iii-a) THE RULING'S OWN CASE, on the deck's filed numbers.

    Both halves, because either alone is worthless.  The FIXED deck —
    3.729e-08 and 3.004e-08 at kappa_eff 8.945e+05 — must now read PASS,
    which is the defect this ruling repairs.  The SAME deck's pre-fix
    numbers — 1.170e+00 and 1.241e+00, the child built from the transfer
    belonging to -q — must still read REFUTED at the SAME kappa, because the
    tolerance moved and the deck's conditioning did not.

    That pair is the whole claim: the gate stopped reporting kappa_eff and
    did not stop reporting breakage.
    """
    from gw import downfold_run as dr
    tol = dr.child_covariance_tol(PROD_KAPPA_EFF)
    for name, worst in PROD_WORST_REL_FIXED.items():
        assert worst < tol, (
            f"the FIXED production deck's {name} covariance {worst:.3e} "
            f"still refutes against tol {tol:.3e} at kappa_eff "
            f"{PROD_KAPPA_EFF:.3e} — the ruling is not implemented")
    for name, worst in PROD_WORST_REL_BROKEN.items():
        assert worst > tol, (
            f"the BROKEN production deck's {name} covariance {worst:.3e} "
            f"passes against tol {tol:.3e} — the scaling has eaten the gate")
    # and the margin on each side is not a whisker
    assert tol / max(PROD_WORST_REL_FIXED.values()) > 2.0
    assert min(PROD_WORST_REL_BROKEN.values()) / tol > 1.0e6, (
        "the gate has under 1e6 of discrimination against the real defect "
        "at production conditioning")


def _qirr_ill_conditioned_parent(mesh, kappa_design=3.0e8, seed=3):
    """A wedge parent like ``_qirr_parent`` but with a DESIGNED spectrum.

    ``_qirr_parent`` builds ``a a^H + mu I``, which lands at kappa_eff ~ 11 —
    the regime the 1e-9 constant was calibrated in, and the regime that
    cannot say anything about a production deck.  This builds the same object
    (Hermitian positive definite, one block per wedge q) with a geometric
    eigenvalue ladder instead, so the transfer solve below runs at a
    kappa_eff the caller MEASURES and asserts rather than assumes.

    ``W`` is left exactly as ``_qirr_parent`` builds it: the operand under
    congruence is not what is being conditioned.
    """
    rng = np.random.default_rng(seed)
    n_ibz = QIRR_Q_IRR_FRAC.shape[0]
    S_ibz = np.empty((n_ibz, QIRR_MU, QIRR_MU), dtype=np.complex128)
    for q in range(n_ibz):
        a = (rng.normal(size=(QIRR_MU, QIRR_MU))
             + 1j * rng.normal(size=(QIRR_MU, QIRR_MU)))
        U, _ = np.linalg.qr(a)
        lam = np.geomspace(1.0, 1.0 / kappa_design, QIRR_MU)
        M = U @ np.diag(lam) @ U.conj().T
        S_ibz[q] = 0.5 * (M + M.conj().T)
    b = (rng.normal(size=(n_ibz, QIRR_MU, QIRR_MU))
         + 1j * rng.normal(size=(n_ibz, QIRR_MU, QIRR_MU)))
    W_ibz = b + np.conj(np.swapaxes(b, 1, 2))
    munu = NamedSharding(mesh, P(None, "x", "y"))
    return (jax.lax.with_sharding_constraint(jnp.asarray(S_ibz), munu),
            jax.lax.with_sharding_constraint(jnp.asarray(W_ibz), munu))


def _qirr_child_through_the_shipping_gate(mesh, *, q_sign_break):
    """Build a full-BZ child at production conditioning and GATE it.

    Returns ``(verdict_lines, worst_rel, kappa_max, tol)``, where the verdict
    comes from ``downfold_run._gate_child_wedge_storability`` — the driver's
    OWN function, not a reimplementation of it, so a green here is about the
    gate that runs in production and not about a second transcription of it.

    ``q_sign_break=True`` reproduces THE SHIPPED DEFECT of 2026-08-09..11: the
    transfer is solved from the Gram belonging to ``-q`` and then applied to
    the tensor at ``+q``.  That is exactly what ``pair_density_gram`` did
    before the ``negate_q_index`` relabel — the "bare conj(T) construction" —
    and it is a GENUINE covariance break rather than a scaled-up round-off,
    which is what makes it the right red twin for a tolerance change.
    """
    from gw import downfold_run as dr
    from types import SimpleNamespace

    sym_perm, L_table = _qirr_tables()
    S_ibz, W_ibz = _qirr_ill_conditioned_parent(mesh)
    S_full = np.asarray(jax.device_get(
        _qirr_unfold(S_ibz, mesh, sym_perm, L_table)))
    W_full = _qirr_unfold(W_ibz, mesh, sym_perm, L_table)
    keep = QIRR_KEEP_CLOSED

    S_SS = S_full[:, keep, :][:, :, keep]
    S_cross = S_full[:, keep, :]
    if q_sign_break:
        # the q axis run backwards, which is what the kernel returned
        neg = downfold.negate_q_index((QIRR_NQ_FULL, 1, 1))
        assert np.array_equal(neg[neg], np.arange(QIRR_NQ_FULL))
        assert 0 < int(np.sum(neg != np.arange(QIRR_NQ_FULL))) < QIRR_NQ_FULL,\
            "the relabel must move SOME q and fix others, as it does on a deck"
        S_SS, S_cross = S_SS[neg], S_cross[neg]

    munu = NamedSharding(mesh, P(None, "x", "y"))
    # rcond well below the designed conditioning, so nothing is truncated and
    # kappa_eff is the Gram's own condition number rather than the cap.
    T_x, T_y, reports = downfold.build_transfer(
        jax.lax.with_sharding_constraint(jnp.asarray(S_SS), munu),
        jax.lax.with_sharding_constraint(jnp.asarray(S_cross), munu),
        mesh, rcond=1e-14, announce=False)
    W_S = downfold.congruence(mesh, T_x, T_y)(W_full)

    tables = SimpleNamespace(
        sym_perm=sym_perm, L_table=L_table, irr_idx_q=QIRR_IRR_IDX,
        sym_idx_q=QIRR_SYM_IDX, q_irr_frac=QIRR_Q_IRR_FRAC,
        n_sym_spatial=QIRR_ORB)
    kappa_per_q = np.array([r.kappa_eff for r in reports], dtype=np.float64)
    rank_per_q = np.array([r.rank_criterion for r in reports], dtype=np.int64)
    out = []
    _cp, _cl, _slots, worst_rel, kap_max, tol = \
        dr._gate_child_wedge_storability(
            {"W0_qmunu": W_S}, keep, tables, mesh, keep.size, keep.size,
            rank_per_q=rank_per_q, kappa_per_q=kappa_per_q,
            parent_tensor=W_full, mu_L=QIRR_MU, print_fn=out.append)
    return "\n".join(out), worst_rel, kap_max, tol


def test_qirr_the_gate_PASSES_a_correct_child_at_PRODUCTION_kappa():
    """(iii-b) The scaled gate is not vacuous: a correct child at high kappa.

    The conditioning is MEASURED, not assumed — the cell asserts the solve
    actually reached a kappa_eff within an order of magnitude of the
    production deck's 8.9e+05 before it reads anything into the verdict.  A
    "high-kappa" cell that quietly ran at kappa 11 would be the shape of
    failure this whole section is about.
    """
    mesh = resolve_mesh()
    log, worst_rel, kap, tol = _qirr_child_through_the_shipping_gate(
        mesh, q_sign_break=False)
    assert kap > 1.0e5, (
        f"the ill-conditioned synthetic only reached kappa_eff {kap:.3e}; it "
        f"is not testing production conditioning and its verdict says "
        f"nothing about the ruling")
    assert "VERDICT: PASS" in log, log
    assert worst_rel < tol
    # The CONTROL ran and is at the unscaled floor: the harness is exonerated
    # independently of the tolerance under test.
    assert "NOT TAKEN" not in log
    assert "THE CONTROL FAILED" not in log


def test_qirr_RED_TWIN_a_q_sign_break_at_PRODUCTION_kappa_is_still_REFUTED():
    """(iii-c) THE RED TWIN THE RULING DEMANDS, and it is a real defect.

    Same deck-scale conditioning, same tables, same shipping gate — and the
    transfer built at ``-q`` and applied at ``+q``, which is the defect that
    poisoned every downfolded child between 2026-08-09 and 2026-08-11.  It
    must still print REFUTED, and the margin is asserted rather than admired:
    a scaled tolerance that merely happened to stay below an order-one number
    would be one deck away from not doing so.

    Note the shape as well as the size, which is what named the mechanism on
    the real deck: the relabel is an involution that fixes some q and moves
    others, so this twin is a partial break — the honest hard case — and not
    a uniformly corrupted tensor.
    """
    mesh = resolve_mesh()
    log, worst_rel, kap, tol = _qirr_child_through_the_shipping_gate(
        mesh, q_sign_break=True)
    assert kap > 1.0e5, f"red twin ran at kappa_eff {kap:.3e}, not production"
    assert "VERDICT: REFUTED" in log, log
    assert worst_rel > 0.1, (
        f"the q-sign break only moved the covariance to {worst_rel:.3e} on "
        f"this synthetic — if that is real it is not the order-one defect "
        f"the production deck showed, and this twin is not testing it")
    assert worst_rel / tol > 1.0e6, (
        f"the red twin clears the scaled tolerance by only "
        f"{worst_rel / tol:.3e}x at kappa_eff {kap:.3e}.  The gate is "
        f"supposed to keep enormous discrimination against real breakage "
        f"after this ruling, not squeak past it")
    # and the log says so LOUDLY, with the mechanism, not just a number
    assert "THE CHILD IS NOT WEDGE-STORABLE" in log
    assert "x the tolerance THIS run's own conditioning allows" in log


def test_the_gate_does_NOT_scale_without_a_measured_kappa():
    """kappa is a measurement or it is nothing — handed none, the gate says so.

    Constraint from the ruling: ``kappa_eff`` is the number recorded in this
    run's provenance and is never estimated inside the gate.  So the caller
    that fails to supply it gets the OLD absolute floor and a verdict line
    that names the absence, rather than a silently scaled tolerance built on
    a guess.
    """
    mesh = resolve_mesh()
    from gw import downfold_run as dr
    from types import SimpleNamespace
    sym_perm, L_table = _qirr_tables()
    S_ibz, W_ibz = _qirr_parent(mesh)
    W_full = _qirr_unfold(W_ibz, mesh, sym_perm, L_table)
    S_full = np.asarray(jax.device_get(
        _qirr_unfold(S_ibz, mesh, sym_perm, L_table)))
    keep = QIRR_KEEP_CLOSED
    munu = NamedSharding(mesh, P(None, "x", "y"))
    T_x, T_y, _r = downfold.build_transfer(
        jax.lax.with_sharding_constraint(
            jnp.asarray(S_full[:, keep, :][:, :, keep]), munu),
        jax.lax.with_sharding_constraint(jnp.asarray(S_full[:, keep, :]),
                                         munu),
        mesh, rcond=1e-10, announce=False)
    W_S = downfold.congruence(mesh, T_x, T_y)(W_full)
    tables = SimpleNamespace(
        sym_perm=sym_perm, L_table=L_table, irr_idx_q=QIRR_IRR_IDX,
        sym_idx_q=QIRR_SYM_IDX, q_irr_frac=QIRR_Q_IRR_FRAC,
        n_sym_spatial=QIRR_ORB)
    out = []
    *_rest, _worst, kap_max, tol = dr._gate_child_wedge_storability(
        {"W0_qmunu": W_S}, keep, tables, mesh, keep.size, keep.size,
        rank_per_q=np.zeros(QIRR_NQ_FULL, dtype=np.int64), kappa_per_q=None,
        parent_tensor=W_full, mu_L=QIRR_MU, print_fn=out.append)
    log = "\n".join(out)
    assert np.isnan(kap_max)
    assert tol == dr.CHILD_COVARIANCE_TOL
    assert "kappa_eff was NOT SUPPLIED" in log, log
    assert "is NOT scaled" in log


# ---------------------------------------------------------------------------
# 7.  Multi-device: the whole chain, P=1 against P>1
# ---------------------------------------------------------------------------

def _mesh_invariance_payload():
    """Run the chain on whatever mesh this process has; return the digest."""
    mesh = resolve_mesh()
    psi_X, psi_Y = _psi(mesh, seed=41, mu=16)
    win = BandWindow(left=(0, NB), right=(0, NB))
    S_LL = downfold.pair_density_gram(psi_X, psi_Y, win, kgrid=KGRID,
                                      mesh_xy=mesh)
    S_q0 = jax.lax.with_sharding_constraint(
        S_LL[0], NamedSharding(mesh, P("x", "y")))
    keep, rep = downfold.select_cur_centroids(
        S_q0, 8, rcond=1e-10, select_tol=None, mesh_xy=mesh,
        mu_large_logical=16, print_fn=lambda *a: None)
    psi_S_X, psi_S_Y = downfold.slice_psi_to_centroids(
        psi_X, psi_Y, keep, 8, mesh)
    S_cross = downfold.pair_density_gram(psi_S_X, psi_Y, win, kgrid=KGRID,
                                         mesh_xy=mesh)
    S_SS = downfold.pair_density_gram(psi_S_X, psi_S_Y, win, kgrid=KGRID,
                                      mesh_xy=mesh)
    T_x, T_y, reports = downfold.build_transfer(
        S_SS, S_cross, mesh, rcond=1e-10, announce=False)
    rng = np.random.default_rng(43)
    a = rng.normal(size=(NK, 16, 16)) + 1j * rng.normal(size=(NK, 16, 16))
    W_L = jax.lax.with_sharding_constraint(
        jnp.asarray(a + np.conj(np.swapaxes(a, 1, 2))),
        NamedSharding(mesh, P(None, "x", "y")))
    W_S = downfold.congruence(mesh, T_x, T_y)(W_L)
    eps, _n_L, _n_S = downfold.epsilon_w(W_L, S_LL, W_S, S_SS, mesh)
    return {
        "devices": int(jax.device_count()),
        "mesh": [int(v) for v in mesh.shape.values()],
        "keep": [int(v) for v in keep],
        "rank": [int(r.rank_criterion) for r in reports],
        "W_S_re": np.asarray(jax.device_get(W_S)).real.ravel().tolist(),
        "W_S_im": np.asarray(jax.device_get(W_S)).imag.ravel().tolist(),
        "eps": [float(v) for v in eps],
    }


def _run_worker(ndev, timeout=900):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={ndev}").strip()
    res = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "worker"],
        env=env, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise AssertionError(
            f"worker at ndev={ndev} failed rc={res.returncode}\n"
            f"{res.stdout[-4000:]}\n{res.stderr[-4000:]}")
    line = [ln for ln in res.stdout.splitlines() if ln.startswith("{")][-1]
    return json.loads(line)


@pytest.mark.parametrize("ndev", [4])
def test_downfold_is_mesh_invariant(ndev):
    """The whole chain at P>1 must reproduce P=1.

    The local/GSPMD plan emits no block-cyclic factorisation anywhere, so
    agreement here is expected to be TIGHT, not merely within a gauge
    tolerance.  The Gram and the congruence do run real collectives at P>1
    (psum_scatter, all-to-all), so their reduction ORDER changes and exact
    bit-identity is not claimed; the band asserted is 1e-12 relative, which
    is a floating-point reassociation, not a different algorithm.
    """
    one = _run_worker(1)
    many = _run_worker(ndev)
    assert one["devices"] == 1 and many["devices"] == ndev
    assert one["keep"] == many["keep"], (
        "the selected centroid set changed with the device count — the "
        "retained subspace must not depend on the mesh")
    assert one["rank"] == many["rank"]
    a = np.array(one["W_S_re"]) + 1j * np.array(one["W_S_im"])
    b = np.array(many["W_S_re"]) + 1j * np.array(many["W_S_im"])
    assert np.max(np.abs(a - b)) < 1e-12 * np.max(np.abs(a))
    assert np.allclose(one["eps"], many["eps"], rtol=1e-10, atol=1e-14)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        print(json.dumps(_mesh_invariance_payload()))
