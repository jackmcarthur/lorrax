"""The dipole-route exchange head: the contraction, and the twin that must fail.

``K^head_{t,t'} = (1/N_k) conj(d_a(t)) M_ab d_b(t')`` is rank three over the
transition index and never enters the μ basis (``LT_HEAD_PROBLEM.md`` §6).  The
matvec adds it as the SAME contraction as the exchange term with ``(M_Y, V_q0,
M_X)`` replaced by ``(D_head, M_head, D_head)``, which means the conjugation
convention has to match the exchange term's exactly — and the exchange term's
conjugation is the one the tree has already had to fix once (``K^x = M V M†``:
conjugated vertex on the encode leg).

These gates are on the CONTRACTION, spelled out against a dense reference
built by hand, so they need no FFI, no fixture, no GPU and no mesh — a
deliberate choice, because the matvec module itself cannot be imported without
the host FFI library.  What they cannot see is the plumbing; the bit-identity
of the OFF arm is gated separately in ``tests/test_exciton_bands.py``.

CENSUS-CLASS: crossed-convention gates.  They should carry the ``census``
pytest marker as soon as it exists — it does not exist on ``main`` as of
2026-08-09 (a parallel lane is introducing it).
"""
import numpy as np
import pytest

RNG = np.random.default_rng(6)
NK, NC, NV, NB = 4, 3, 2, 5


def _operands():
    """A dipole block, a PSD cell moment, and a trial vector stack."""
    d = (RNG.normal(size=(NK, NC, NV, 3))
         + 1j * RNG.normal(size=(NK, NC, NV, 3)))       # d_a(t)
    # M = <v q q> is real symmetric POSITIVE SEMI-DEFINITE by construction:
    # it is an average of v*q q^T with v >= 0.
    A = RNG.normal(size=(3, 3))
    M = A @ A.T
    X = (RNG.normal(size=(NB, NC, NV, NK))
         + 1j * RNG.normal(size=(NB, NC, NV, NK)))
    return d, M, X


def _dense_head(d, M):
    """K^head_{t,t'} = conj(d_a(t)) M_ab d_b(t'), t = (c, v, k) flattened."""
    dt = np.transpose(d, (1, 2, 0, 3)).reshape(-1, 3)    # (nt, 3), t=(c,v,k)
    return np.conj(dt) @ M.astype(complex) @ dt.T


def _matvec_head(D_head, M_head, X):
    """The production spelling, lifted out of bse_stack_matvec._matvec."""
    sqrt_nk = np.sqrt(NK)
    Sh = np.einsum("kcva,bcvk->ba", np.conj(D_head), X) / sqrt_nk
    Uh = Sh @ M_head.astype(Sh.dtype).T
    HX = np.einsum("kcva,ba->bcvk", D_head, Uh)
    return HX / sqrt_nk


def test_the_matvec_head_is_the_dense_head_matrix():
    """Applying the term must equal multiplying by K^head/N_k."""
    d, M, X = _operands()
    D_head = np.conj(d)                                  # what the driver hands in
    got = _matvec_head(D_head, M, X)

    K = _dense_head(d, M) / NK
    Xf = X.reshape(NB, -1)
    want = (Xf @ K.T).reshape(NB, NC, NV, NK)
    assert np.allclose(got, want, atol=1e-11), (
        f"matvec head != dense K^head/N_k, max|Δ| = "
        f"{np.max(np.abs(got - want)):.3e}")


def test_the_head_is_hermitian_and_positive_semidefinite():
    """Both follow from M real symmetric PSD, and both are physics.

    Hermiticity is required of any kernel the BSE Hamiltonian carries.  PSD is
    the statement that the head is a self-interaction of the transition
    density through a repulsive kernel — ``v >= 0`` sample by sample, so
    ``M`` is PSD, so ``K^head`` is.  Its eigenvalue count is at most 3, which
    is what makes exactly one eigenvector of an N-fold degenerate bright
    multiplet pick up the shift and N-1 not.
    """
    d, M, _ = _operands()
    K = _dense_head(d, M)
    assert np.allclose(K, K.conj().T, atol=1e-11), "K^head is not Hermitian"
    w = np.linalg.eigvalsh(K)
    assert w.min() > -1e-9 * max(abs(w.max()), 1.0), f"not PSD: {w.min():.3e}"
    assert int(np.sum(w > 1e-9 * w.max())) <= 3, (
        f"rank {int(np.sum(w > 1e-9 * w.max()))} > 3 — the head is a "
        f"three-vector outer product and cannot have more")


@pytest.mark.parametrize("twin", ["transposed_encode", "unconjugated_encode",
                                  "conjugated_decode", "transposed_moment"])
def test_red_twin_a_crossed_contraction_must_fail(twin):
    """Deliberately crossed conventions, and what each one breaks.

    The exchange term's conjugation has been wrong in this tree before, and
    the head term copies its shape, so it can be wrong the same way.  Each
    twin below is a spelling somebody could plausibly write; the gate is that
    every one of them is DISTINGUISHABLE — either it stops reproducing the
    dense head, or it stops being Hermitian.
    """
    d, M, X = _operands()
    D_head = np.conj(d)
    sqrt_nk = np.sqrt(NK)

    if twin == "transposed_encode":
        # decode vertex on the encode leg and vice versa
        Sh = np.einsum("kcva,bcvk->ba", D_head, X) / sqrt_nk
        Uh = Sh @ M.astype(complex).T
        got = np.einsum("kcva,ba->bcvk", np.conj(D_head), Uh) / sqrt_nk
    elif twin == "unconjugated_encode":
        Sh = np.einsum("kcva,bcvk->ba", D_head, X) / sqrt_nk
        Uh = Sh @ M.astype(complex).T
        got = np.einsum("kcva,ba->bcvk", D_head, Uh) / sqrt_nk
    elif twin == "conjugated_decode":
        Sh = np.einsum("kcva,bcvk->ba", np.conj(D_head), X) / sqrt_nk
        Uh = Sh @ M.astype(complex).T
        got = np.einsum("kcva,ba->bcvk", np.conj(D_head), Uh) / sqrt_nk
    else:  # transposed_moment — invisible for symmetric M, so use a twin M
        Mx = M + np.array([[0.0, 0.7, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        Sh = np.einsum("kcva,bcvk->ba", np.conj(D_head), X) / sqrt_nk
        got = np.einsum("kcva,ba->bcvk", D_head,
                        Sh @ Mx.astype(complex)) / sqrt_nk
        want = (X.reshape(NB, -1) @ (_dense_head(d, Mx) / NK).T).reshape(X.shape)
        assert not np.allclose(got, want, atol=1e-9), (
            "the moment transpose was invisible; a non-symmetric M must be "
            "distinguishable so the gate covers the index order too")
        # and the correct spelling still agrees on that same non-symmetric M
        ok = _matvec_head(D_head, Mx, X)
        assert np.allclose(ok, want, atol=1e-11)
        return

    K = _dense_head(d, M) / NK
    want = (X.reshape(NB, -1) @ K.T).reshape(X.shape)
    assert not np.allclose(got, want, atol=1e-9), (
        f"the {twin} twin reproduced the correct head — this gate cannot see "
        f"the crossed convention it exists for")


def test_the_off_arm_is_not_emitted_at_all():
    """Default OFF must trace a program with no head contraction in it.

    Not a zero-valued one — the same discipline ``W_q0=None`` already carries
    in ``bse_davidson_helpers``.  Multiplying by a zero tensor would be a
    different HLO and could move the last ulp of a run that never asked for
    the feature; the DEFAULT finite-Q exchange path is proven exact
    (``LT_HEAD_PROBLEM.md`` §1.3) and must stay bit-identical.

    Fixture-free because ``bse_stack_matvec`` cannot be imported without the
    host FFI library.  The numeric half of this gate — the same deck run with
    the key unset on both trees — is a deck A/B, reported rather than
    committed.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "bse"
    mv = ast.parse((root / "bse_stack_matvec.py").read_text(encoding="utf-8"))

    builder = next(n for n in ast.walk(mv)
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "build_bse_stack_matvec")
    kw = {a.arg: d for a, d in zip(builder.args.kwonlyargs,
                                   builder.args.kw_defaults)}
    assert "head_tensor" in kw and kw["head_tensor"].value is False, (
        "build_bse_stack_matvec must default head_tensor=False")

    inner = next(n for n in ast.walk(builder)
                 if isinstance(n, ast.FunctionDef) and n.name == "_matvec")
    guarded = [n for n in inner.body
               if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
               and n.test.id == "head_tensor"]
    assert guarded, (
        "the head term is not inside an `if head_tensor:` block — with the "
        "feature off it would still be traced")
    src = (root / "bse_stack_matvec.py").read_text(encoding="utf-8")
    body = ast.get_source_segment(src, guarded[0])
    assert "D_head" in body and "M_head" in body, (
        "the guarded block does not contain the head contraction")

    xb = ast.parse((root / "exciton_bands.py").read_text(encoding="utf-8"))
    ps = next(n for n in ast.walk(xb)
              if isinstance(n, ast.FunctionDef) and n.name == "build_path_solver")
    kw2 = {a.arg: d for a, d in zip(ps.args.kwonlyargs, ps.args.kw_defaults)}
    assert "head_tensor" in kw2 and kw2["head_tensor"].value is False


def test_red_twin_a_transposed_head_matrix_breaks_hermiticity():
    """The gate the solver itself would trip on: α-Hermiticity."""
    d, M, _ = _operands()
    dt = np.transpose(d, (1, 2, 0, 3)).reshape(-1, 3)
    good = np.conj(dt) @ M.astype(complex) @ dt.T
    bad = dt @ M.astype(complex) @ dt.T            # both vertices unconjugated
    assert np.allclose(good, good.conj().T, atol=1e-11)
    dev = np.max(np.abs(bad - bad.conj().T))
    assert dev > 1e-6 * np.max(np.abs(bad)), (
        "the unconjugated head stayed Hermitian, so the solver's own "
        "alpha-Hermiticity report would not catch it")
