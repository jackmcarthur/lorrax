"""The q=0 screened head of the BSE direct term uses the exact overlaps.

At q = 0, G = 0 the direct-kernel pair densities are band overlaps,
M_nn'(k,k;G=0) = <nk|n'k> = delta_nn', so for a normalised exciton the head
content <A|head|A> / (whead/(Omega N_k)) is exactly sum|A|^2 = 1.  The ISDF
rank-1 head ``conj(g0) g0`` on the W tile evaluates the overlaps as
``sum_mu g0_mu psi*_n(r_mu) psi_n'(r_mu)``, which is not orthonormal; that
path over-counted by 0.8 % on MoS2 12x12 (N_mu 1198), -4.3 meV.

This fixture has deliberately non-orthonormal centroid values, so the ISDF
content is far from 1.  The exact form is opt-in (``q0_head="diagonal"``):
there the loader's head (``bse_head._inject_q0_head``) plus the operator
diagonal (``bse_head.head_shifted_eps_c``) must give 1 to 1e-12, while the
default ``"rank1"`` must still be the ISDF head on W, so consumers that never
read ``q0_head_ry`` (TD-aGW, screening, ladder) are unchanged.  The W-term
contraction is the one ``bse_stack_matvec.make_w_term`` documents.  CPU only,
no mesh, no FFI.
"""
import inspect

import numpy as np
import jax.numpy as jnp
import pytest

from bse.bse_head import _inject_q0_head, head_shifted_eps_c, q0_head_diagonal
from bse.bse_loading import load_bse_data_from_restart_sharded
from gw.head_correction import apply_q0_head_rank1_sharded

NK, NC, NV, NS, NMU = 4, 3, 2, 2, 24
WHEAD, VHEAD, OMEGA = 4171.918465, 9304.681621, 702.20


def _fixture(seed=7):
    rng = np.random.default_rng(seed)
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    psi_c = cplx(NK, NC, NS, NMU) / np.sqrt(NS * NMU)        # (k, c, s, mu)
    psi_v = cplx(NK, NV, NS, NMU) / np.sqrt(NS * NMU)
    g0 = 1.0 + 0.3 * rng.standard_normal(NMU)                # zeta(0, mu, G=0)
    A = cplx(NC, NV, NK)
    A /= np.linalg.norm(A)                                   # normalised exciton
    return psi_c, psi_v, g0, A


def _w_term_expectation(A, psi_c, psi_v, W0):
    """<A|W A> for the q=0 tile W0 alone (k' = k), make_w_term's formula."""
    T = np.einsum("kctm,kvsn,cvk->ktmsn", psi_c, psi_v.conj(), A)
    WA = np.einsum("kctm,kvsn,mn,ktmsn->cvk", psi_c.conj(), psi_v, W0, T) / NK
    return complex(np.vdot(A, WA))


def test_q0_head_content_is_exact_and_the_isdf_head_is_not():
    psi_c, psi_v, g0, A = _fixture()
    ref = WHEAD / (OMEGA * NK)                     # Ry, the exact head weight
    V0 = jnp.zeros((NMU, NMU), jnp.complex128)
    W_q = jnp.zeros((NMU, NMU, 2, 2, 1), jnp.complex128)
    g0j = jnp.asarray(g0, jnp.complex128)
    whead = jnp.asarray([WHEAD], jnp.complex128)

    # ISDF rank-1 head (the former whead injection): content != 1.
    _, W_isdf = apply_q0_head_rank1_sharded(V0, W_q, g0j, g0j, None, whead, OMEGA)
    c_isdf = _w_term_expectation(A, psi_c, psi_v, np.asarray(W_isdf[:, :, 0, 0, 0])) / ref
    O_c = np.einsum("m,kctm,kdtm->kcd", g0, psi_c.conj(), psi_c)   # ISDF <c|c'>
    O_v = np.einsum("m,kvsm,kwsm->kvw", g0, psi_v.conj(), psi_v)
    c_oo = np.einsum("cvk,kcd,kvw,dwk->", A.conj(), O_c, O_v.conj(), A)
    assert abs(c_isdf - c_oo) < 1e-12 * abs(c_oo)
    assert abs(c_isdf - 1.0) > 1e-2, c_isdf

    # Default "rank1": the former injection, bit for bit, and no diagonal.
    V_r, W_r, q0_r, _ = _inject_q0_head(
        V0, W_q, g0j, g0j, VHEAD, whead, OMEGA, w0_ready=True, nk=NK)
    assert q0_r == 0.0
    np.testing.assert_array_equal(np.asarray(W_r), np.asarray(W_isdf))
    rank1 = {"q0_head": "rank1", "q0_head_ry": q0_r}
    assert inspect.signature(load_bse_data_from_restart_sharded).parameters[
        "q0_head"].default == "rank1"

    # "diagonal": V gets vhead, W stays head-free, whead -> diagonal.
    V_out, W_out, q0, _ = _inject_q0_head(
        V0, W_q, g0j, g0j, VHEAD, whead, OMEGA, w0_ready=True, nk=NK,
        q0_head="diagonal")
    assert W_out is W_q
    np.testing.assert_allclose(
        np.asarray(V_out), VHEAD / OMEGA * np.outer(g0, g0), rtol=1e-14)
    np.testing.assert_array_equal(np.asarray(V_out), np.asarray(V_r))
    assert abs(q0 - ref) <= 1e-15 * ref
    assert q0_head_diagonal(WHEAD, OMEGA, NK) == q0
    diag = {"q0_head": "diagonal", "q0_head_ry": q0}

    # The operator: D = eps_c - eps_v with the operand conduction leg;
    # H = D + V - W, so the head's W-sign content is <A|W0|A> - <A|dD|A>.
    rng = np.random.default_rng(3)
    eps_c, eps_v = rng.random((NK, NC)) + 0.5, -rng.random((NK, NV))
    D = lambda ec: ec.T[:, None, :] - eps_v.T[None, :, :]     # (c, v, k)

    def content(W_tile, bundle):
        dD = D(np.asarray(head_shifted_eps_c(eps_c, bundle))) - D(eps_c)
        return (_w_term_expectation(A, psi_c, psi_v, np.asarray(
            W_tile[:, :, 0, 0, 0] - W_q[:, :, 0, 0, 0]))
            - np.vdot(A, dD * A)) / ref

    assert abs(content(W_out, diag) - 1.0) <= 1e-12
    assert abs(content(W_r, rank1) - c_isdf) <= 1e-12 * abs(c_isdf)
    assert head_shifted_eps_c(eps_c, rank1) is eps_c
    assert head_shifted_eps_c(eps_c, {}) is eps_c

    # Deferred (densifying) and unscreened loads carry no diagonal.
    for kw in (dict(w0_ready=True, defer_whead=True), dict(w0_ready=False)):
        assert _inject_q0_head(V0, W_q, g0j, g0j, VHEAD, whead, OMEGA,
                               nk=NK, q0_head="diagonal", **kw)[2] == 0.0

    # Guards: a scalar on a rank-1 bundle would count the head twice; a
    # diagonal bundle without its scalar, or an unknown mode, refuses.
    with pytest.raises(ValueError, match="q0_head_mode"):
        head_shifted_eps_c(eps_c, {"q0_head": "rank1", "q0_head_ry": q0})
    with pytest.raises(KeyError):
        head_shifted_eps_c(eps_c, {"q0_head": "diagonal"})
    with pytest.raises(ValueError, match="q0_head must be"):
        head_shifted_eps_c(eps_c, {"q0_head": "exact"})


# The BSE operator drivers audited to build every operand through
# head_shifted_eps_c.  A new opt-in must be audited and added here.
_DIAGONAL_DRIVERS = {"bse_jax.py", "absorption_haydock.py", "exciton_bands.py",
                     "bse_feast.py", "bse_kpm.py"}


def test_only_audited_drivers_request_the_diagonal_head():
    import pathlib
    import bse
    root = pathlib.Path(bse.__file__).parent
    found = {f.name for f in root.glob("*.py")
             if 'q0_head="diagonal"' in f.read_text()}
    assert found == _DIAGONAL_DRIVERS, found
