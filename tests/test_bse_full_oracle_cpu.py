"""The dense oracle for the full BSE (``tests/bse_full_oracle.py``) against independent references.

A solver test is only as good as its reference, so the reference is tested first, on the
synthetic BSE-structured pencils the oracle generates (positive-definite ``K``: dominant,
near-singular, exactly degenerate, clustered; and one slightly indefinite pencil with a single
imaginary pair), with energies in Ry:

1. Assembly: ``assemble_dense_ab`` recovers dense ``A`` and ``B`` from identity columns of
   ``pair(X, s) = A X + s B conj(X)`` (a real-linear, not complex-linear, applier) to round-off
   on a multi-axis window, and ``structure_errors`` flags a non-Hermitian ``A`` and a
   non-symmetric ``B`` (red twins).
2. The dense structure-preserving solve (Cholesky of ``K``) equals the brute-force
   ``numpy.linalg.eig`` of ``[[A, B], [-B*, -A*]]``: roots to 1e-9 Ry, residuals of
   ``H z = omega z``, ``X^H X - Y^H Y = 1`` per root and Sigma-orthonormality of the whole
   set, and the eigenspace of every degenerate or clustered group by Sigma-projector.
3. The roots equal the analytic or independently computed values the generator records
   (``sqrt(a^2 - |b|^2)`` for the degenerate and clustered pencils, ``eigvalsh(P^(1/2) M P^(1/2))``
   with ``P = A + B``, ``M = A - B`` for the near-singular and indefinite ones).
4. The indefinite pencil: ``reference_roots`` reports ``definite = False`` and the one
   imaginary pair at its analytic value, returns the ``N - 1`` real roots, and the
   structure-preserving route refuses it.
5. The production dense solve (``bse_nontda.solve_nontda_definite_pencil``) gives the oracle's
   roots on the definite pencils and refuses the indefinite one (the oracle's cross-check
   against ``src/``; the oracle module itself imports no ``src/``).

``JAX_PLATFORMS=cpu python -m pytest tests/test_bse_full_oracle_cpu.py``; NumPy only except
test 5, one process, a few seconds.
"""
import numpy as np
import pytest

from tests import bse_full_oracle as orc

DEFINITE = ("dominant", "near_singular", "degenerate", "clustered")
OMEGA_TOL = 1e-9          # Ry; routes agree to round-off times the condition number of K
RESID_TOL = 1e-10         # |H z - omega z| / (|z| |H|_2)
NORM_TOL = 1e-9           # X^H X - Y^H Y = 1 and the Sigma Gram matrix of the full set
SPAN_TOL = 1e-6           # Sigma-projector distance of a degenerate or clustered group


@pytest.fixture(scope="module")
def pencils():
    return {p["name"]: p for p in orc.standard_pencils()}


def test_generators_have_bse_structure(pencils):
    """Every generated pencil has ``A = A^H``, ``B = B^T`` (1e-14) and the recorded definiteness."""
    assert set(pencils) == {*DEFINITE, "indefinite"}
    for name, p in pencils.items():
        orc.assert_bse_structure(p["A"], p["B"], tol=1e-14)
        assert (p["k_min"] > 0) == p["definite"], (name, p["k_min"])
        assert np.iscomplexobj(p["A"]) and np.linalg.norm(p["A"].imag) > 1e-3 * np.linalg.norm(p["A"])
    assert pencils["near_singular"]["k_min"] < 1e-3                 # K itself is nearly singular
    assert pencils["indefinite"]["n_imag_pairs"] == 1


def test_assembly_from_a_real_linear_applier_and_structure_red_twins(pencils):
    """Identity columns of ``pair(X, s)`` return dense ``A`` and ``B``; broken structure is flagged."""
    p = pencils["dominant"]
    A, B = p["A"], p["B"]
    n = A.shape[0]
    window = (3, 4, n // 12)                                        # a (c, v, k)-like window
    pair = orc.pair_from_dense(A, B)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))
    # the applier is real-linear in X: i X is not i times the output
    out = pair(X.reshape((2,) + window), 1.0).reshape(2, n)
    assert np.allclose(out, X @ A.T + X.conj() @ B.T, atol=1e-14)
    assert not np.allclose(pair((1j * X).reshape((2,) + window), 1.0).reshape(2, n), 1j * out, atol=1e-6)
    for chunk in (1, 7, 64):
        A2, B2 = orc.assemble_dense_ab(pair, window, chunk=chunk)
        assert np.max(np.abs(A2 - A)) <= 1e-14 * np.max(np.abs(A))
        assert np.max(np.abs(B2 - B)) <= 1e-14 * np.max(np.abs(B))
    # red twins: a non-Hermitian A and a non-symmetric B are measured, not hidden
    herm, sym = orc.structure_errors(A + 1e-3 * np.triu(np.ones((n, n))), B)
    assert herm > 1e-4 and sym <= 1e-14
    herm, sym = orc.structure_errors(A, B + 1e-3 * np.triu(np.ones((n, n)), 1))
    assert sym > 1e-4 and herm <= 1e-14
    with pytest.raises(AssertionError, match="not complex symmetric"):
        orc.assert_bse_structure(A, B + 1e-3 * np.triu(np.ones((n, n)), 1))
    with pytest.raises(ValueError, match="not BSE-structured"):
        orc.reference_roots(A, B + 1e-3 * np.triu(np.ones((n, n)), 1))
    # red twin: the assembly reads the applier, so an applier without its B term assembles B = 0
    no_b = lambda X, s: pair(X, 0.0)  # noqa: E731
    Aw, Bw = orc.assemble_dense_ab(no_b, window)
    assert np.linalg.norm(Bw) <= 1e-14 and np.linalg.norm(Aw - A) <= 1e-14 * np.linalg.norm(A)


@pytest.mark.parametrize("name", DEFINITE)
def test_structure_preserving_equals_brute_force_eig(pencils, name):
    """Cholesky structure-preserving solve vs ``eig(H)``: roots, residuals, norms, eigenspaces."""
    p = pencils[name]
    A, B, n = p["A"], p["B"], p["n"]
    sp = orc.solve_structure_preserving(A, B)
    bf = orc.solve_brute_force(A, B)
    assert bf["imag"].size == 0 and bf["omega"].size == n and bf["n_real_negative"] == n
    assert np.max(np.abs(sp["omega"] - bf["omega"])) <= OMEGA_TOL
    # eig returns non-Sigma-orthogonal vectors for roots 1e-7 apart: orthogonal to ~eps/gap = 2e-9
    # (measured), so its Gram matrix is held to 1e-8; the Cholesky route to round-off.
    for out, gram_tol in ((sp, NORM_TOL), (bf, 1e-8)):
        assert np.all(np.diff(out["omega"]) >= 0)
        assert np.max(orc.eigpair_residuals(A, B, out["omega"], out["X"], out["Y"])) <= RESID_TOL
        assert np.max(np.abs(orc.sigma_norms(out["X"], out["Y"]) - 1.0)) <= NORM_TOL
        gram = orc.sigma_gram(out["X"], out["Y"])
        assert np.max(np.abs(gram - np.eye(n))) <= gram_tol           # Sigma-orthonormal, degenerate sets too
    # eigenspace by group (the vectors of a degenerate or clustered group are basis dependent)
    for g in orc.degenerate_groups(sp["omega"], rtol=1e-5):
        d = orc.sigma_projector_distance(sp["X"][:, g], sp["Y"][:, g], bf["X"][:, g], bf["Y"][:, g])
        assert d <= SPAN_TOL, (name, g, d)
    # the partner root: [Y*; X*] solves H z = -omega z
    H = orc.bse_H(A, B)
    Zp = np.concatenate([sp["Y"].conj(), sp["X"].conj()], axis=0)
    assert np.max(np.linalg.norm(H @ Zp + Zp * sp["omega"][None, :], axis=0)) <= 1e-10 * np.linalg.norm(H, 2)
    # truncation
    ref = orc.reference_roots(A, B, n_eig=5)
    assert ref["definite"] and ref["route"] == "structure_preserving"
    assert ref["omega"].shape == (5,) and ref["X"].shape == (n, 5) and ref["Y"].shape == (n, 5)
    assert np.array_equal(ref["omega"], sp["omega"][:5])


@pytest.mark.parametrize("name", ("near_singular", "degenerate", "clustered"))
def test_roots_equal_the_generators_independent_values(pencils, name):
    """The recorded analytic or symmetric-eigenproblem roots, not another BSE solver."""
    p = pencils[name]
    ref = orc.reference_roots(p["A"], p["B"])
    assert np.max(np.abs(ref["omega"] - p["omega_exact"])) <= 1e-10, name
    if name == "degenerate":
        groups = orc.degenerate_groups(ref["omega"])
        assert sorted(len(g) for g in groups) == [3] * 6
    if name == "near_singular":
        assert ref["omega"][0] < 0.05 and p["k_min"] < 1e-3          # a genuinely soft mode


def test_dominant_pencil_is_a_perturbed_tamm_dancoff_spectrum(pencils):
    """Small coupling: the full roots sit within ``|B|_2`` + ``|A - diag|_2`` of the diagonal energies."""
    p = pencils["dominant"]
    A, B = p["A"], p["B"]
    ref = orc.reference_roots(A, B)
    e = np.sort(np.real(np.diag(A)))
    bound = np.linalg.norm(B, 2) + np.linalg.norm(A - np.diag(np.diag(A)), 2)
    assert np.max(np.abs(ref["omega"] - e)) <= bound
    tda = np.linalg.eigvalsh(A)
    assert np.max(np.abs(ref["omega"] - tda)) <= np.linalg.norm(B, 2) * 1.01   # Weyl: coupling shifts <= |B|


def test_indefinite_pencil_reports_its_imaginary_pair_and_real_roots(pencils):
    """One imaginary pair (``K`` not positive definite): detected, not hidden; the rest is real."""
    p = pencils["indefinite"]
    A, B, n = p["A"], p["B"], p["n"]
    with pytest.raises(ValueError, match="not positive definite"):
        orc.solve_structure_preserving(A, B)
    ref = orc.reference_roots(A, B)
    assert ref["definite"] is False and ref["route"] == "brute_force"
    assert ref["imag"].shape == (1,)
    assert abs(ref["imag"][0] - p["imag_exact"][0]) <= 1e-9 * max(1.0, p["imag_exact"][0])
    assert ref["omega"].shape == (n - 1,)
    assert np.max(np.abs(ref["omega"] - p["omega_exact"])) <= 1e-9
    assert np.max(orc.eigpair_residuals(A, B, ref["omega"], ref["X"], ref["Y"])) <= RESID_TOL
    assert np.max(np.abs(orc.sigma_norms(ref["X"], ref["Y"]) - 1.0)) <= NORM_TOL
    assert np.max(np.abs(orc.sigma_gram(ref["X"], ref["Y"]) - np.eye(n - 1))) <= NORM_TOL
    # its imaginary pair is a pair: lambda = +-i gamma with gamma ~ sqrt(delta * mean(P)) small
    assert 0 < ref["imag"][0] < 0.1


def test_projector_distance_red_twin(pencils):
    """``sigma_projector_distance`` separates different eigenspaces (it is not vacuous)."""
    p = pencils["degenerate"]
    ref = orc.reference_roots(p["A"], p["B"])
    g0, g1 = orc.degenerate_groups(ref["omega"])[:2]
    d_same = orc.sigma_projector_distance(ref["X"][:, g0], ref["Y"][:, g0], ref["X"][:, g0], ref["Y"][:, g0])
    d_other = orc.sigma_projector_distance(ref["X"][:, g0], ref["Y"][:, g0], ref["X"][:, g1], ref["Y"][:, g1])
    assert d_same <= 1e-12
    assert d_other >= np.sqrt(2 * len(g0)) * 0.99                    # orthogonal Sigma-spans of rank 3
    # a rotation inside the group does not change the span
    rng = np.random.default_rng(3)
    Q = orc.random_unitary(len(g0), rng)
    d_rot = orc.sigma_projector_distance(ref["X"][:, g0], ref["Y"][:, g0],
                                         ref["X"][:, g0] @ Q, ref["Y"][:, g0] @ Q)
    assert d_rot <= 1e-12


def test_production_dense_pencil_solve_agrees_with_the_oracle(pencils):
    """``bse_nontda.solve_nontda_definite_pencil`` equals the oracle; it refuses the indefinite pencil."""
    pytest.importorskip("jax")
    from bse.bse_nontda import solve_nontda_definite_pencil
    for name in DEFINITE:
        p = pencils[name]
        ref = orc.reference_roots(p["A"], p["B"])
        n_eig = 12
        omega, Z = solve_nontda_definite_pencil(p["A"], p["B"], n_eig)
        assert np.max(np.abs(omega - ref["omega"][:n_eig])) <= OMEGA_TOL, name
        X, Y = Z[:p["n"]], Z[p["n"]:]
        assert np.max(orc.eigpair_residuals(p["A"], p["B"], omega, X, Y)) <= 1e-9, name
        assert np.max(np.abs(orc.sigma_norms(X, Y) - 1.0)) <= NORM_TOL, name
        for g in orc.degenerate_groups(omega, rtol=1e-5):
            # a group cut by n_eig is a partial eigenspace: compare whole groups only
            full = [i for i in orc.degenerate_groups(ref["omega"], rtol=1e-5)
                    if i[0] == g[0]][0]
            if len(full) != len(g):
                continue
            d = orc.sigma_projector_distance(X[:, g], Y[:, g], ref["X"][:, g], ref["Y"][:, g])
            assert d <= SPAN_TOL, (name, d)
    p = pencils["indefinite"]
    with pytest.raises(ValueError, match="NOT positive definite"):
        solve_nontda_definite_pencil(p["A"], p["B"], 5)


@pytest.mark.parametrize("k_min", (1e-8, 1e-9))
def test_production_dense_solve_keeps_its_digits_near_a_zero_mode(k_min):
    """A soft mode (``lambda_min(K) = k_min`` Ry) must not cost the other roots their digits.

    The roots are compared with the generator's independent values
    (``eigvalsh(P^(1/2) M P^(1/2))``).  The Cholesky form ``L^-1 Sigma L^-H`` holds every root to
    1e-10 Ry and the eigenpair residual to 1e-10; the ``K^(-1/2)`` form it replaced lost them on
    all 39 other roots (4.3e-10 Ry at ``k_min = 1e-8``, 5.0e-9 Ry at 1e-9; residual 1.4e-9 and
    1.3e-8), which is the red twin of this test.
    """
    pytest.importorskip("jax")
    from bse.bse_nontda import solve_nontda_definite_pencil
    p = orc.near_singular_pencil(eps=k_min)
    A, B, n = p["A"], p["B"], p["n"]
    assert abs(p["k_min"] - k_min) <= 1e-3 * k_min and p["omega_exact"][0] < 1e-3
    omega, Z = solve_nontda_definite_pencil(A, B, n)
    X, Y = Z[:n], Z[n:]
    err = np.abs(omega - p["omega_exact"])
    resid = orc.eigpair_residuals(A, B, omega, X, Y)
    print(f"MEASURED dense reference, k_min = {k_min:g} Ry: soft root error {err[0]:.2e} Ry, "
          f"other roots {err[1:].max():.2e} Ry, residual {resid.max():.2e}", flush=True)
    assert err.max() <= 1e-10, err.max()
    assert resid.max() <= 1e-10, resid.max()
    assert np.max(np.abs(orc.sigma_norms(X, Y) - 1.0)) <= 1e-12
    assert np.max(np.abs(orc.sigma_gram(X, Y) - np.eye(n))) <= NORM_TOL
