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
    ``X[mu,a] = conj(psi_m(k-q, mu)) psi_n(k, mu)``, factorised through the
    Khatri-Rao identity the kernel exploits.  Written out with every index
    visible, because this reference is the only thing in the suite that
    pins the CONJUGATION CONVENTION — and the campaign has been bitten twice
    by conjugation conventions hidden behind time-reversal symmetry at q=0.
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
            # sum over the RIGHT (n) window at k
            right = np.einsum("nm,nv->mv", psi[k, br0:br1, :rows],
                              np.conj(psi[k, br0:br1, :]))
            # sum over the LEFT (m) window at k(+)q
            left = np.einsum("nm,nv->mv", np.conj(psi[kq, bl0:bl1, :rows]),
                             psi[kq, bl0:bl1, :])
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
    """
    mesh = resolve_mesh()
    rng = np.random.default_rng(101)
    nq, m_s, m_l = 2, 3, 6
    T = (rng.normal(size=(nq, m_s, m_l))
         + 1j * rng.normal(size=(nq, m_s, m_l)))
    g0 = rng.normal(size=m_l) + 1j * rng.normal(size=m_l)
    head = np.conj(g0)[:, None] * g0[None, :]
    T_x = jax.lax.with_sharding_constraint(
        jnp.asarray(T), NamedSharding(mesh, P(None, None, "x")))
    T_y = jax.lax.with_sharding_constraint(
        jnp.asarray(T), NamedSharding(mesh, P(None, None, "y")))
    head_L = jax.lax.with_sharding_constraint(
        jnp.asarray(np.broadcast_to(head, (nq, m_l, m_l)).copy()),
        NamedSharding(mesh, P(None, "x", "y")))
    head_S = np.asarray(jax.device_get(
        downfold.congruence(mesh, T_x, T_y)(head_L)))[0]

    g0_S = np.asarray(jax.device_get(downfold.transform_head_vector(
        jnp.asarray(g0), T_x, 0, mesh)))
    rebuilt = np.conj(g0_S)[:, None] * g0_S[None, :]
    scale = np.max(np.abs(head_S))
    assert np.max(np.abs(rebuilt - head_S)) < 1e-12 * scale

    wrong = np.einsum("mn,n->m", T[0], g0)
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
    # the greedy-fill structure: closure lands on orbit boundaries
    assert all(m % QIRR_ORB == 0 for m in closed_at), closed_at


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
        assert closed.size - keep.size < QIRR_ORB, (
            "completion cost a whole group order; on a greedy pivot order it "
            "should be the tail of the one orbit mu_S stopped inside")
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
