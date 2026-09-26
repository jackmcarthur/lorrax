"""Which conjugation the exciton dipole projection carries, decided by measurement.

THE DEFECT THIS FILE CLOSES.  ``absorption_eigvecs`` contracted
``Σ_t A^S_t d_t`` while ``davidson_absorption`` contracted
``Σ_t conj(A^S_t) d_t``.  Those are not two spellings of one number: for
complex ``A`` they differ in MODULUS, by up to 6.8x per element on the
committed dipole fixtures, so the two drivers reported different
oscillator strengths from the same eigenvectors
(``KNOWN_FAILURES.md``, "THE TWO ABSORPTION DRIVERS DISAGREE ON A
CONJUGATION").

WHY THE ANSWER IS A MEASUREMENT AND NOT A CONVENTION.  ``|A|^2`` and
``|d|^2`` are both blind to conjugation, so no existing gate could see
this, and an appeal to BerkeleyGW's spelling only moves the question to
whether the two codes store ``A`` the same way.  The identity that
settles it needs neither: for ANY Hermitian ``H`` and ANY seed ``d``,

    ⟨d|(z − H)^-1|d⟩  ==  Σ_S |⟨S|d⟩|^2 / (z − E_S)                (*)

exactly, with no convention anywhere in it.  The left side is what the
Haydock route evaluates (``absorption_haydock``, which never forms an
eigenvector and so cannot get this wrong); the right side is what the
sum-over-states routes build.  Only one contraction makes them equal,
and this file runs both arms on a Hermitian ``H`` with genuinely complex
eigenvectors and shows which.

  TRUE arm  : ``⟨0|r̂_α|S⟩ = Σ_t A^S_t conj(d^α_t)``  — satisfies (*).
  FALSE arm : ``⟨0|r̂_α|S⟩ = Σ_t A^S_t d^α_t``        — the pre-fix
              ``absorption_eigvecs`` spelling; it does NOT satisfy (*),
              and ``test_the_unconjugated_contraction_breaks_the_resolvent``
              is the cell that says so rather than leaving it asserted.

Fixture-free and import-light on purpose: no deck, no GPU, no FFI, no
``dipole.h5``.  The physics content is the identity, and a synthetic
Hermitian block carries it as well as a real one while running
everywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bse.absorption_common import exciton_dipole_projections, load_eigenvectors_h5


#: Transition block.  Small, complex, and NOT square in (c, v) — a square
#: block lets an accidental transpose pass, which is the neighbouring
#: defect in the same class.
N_COND, N_VAL, N_K = 3, 2, 4
N_TRANS = N_COND * N_VAL * N_K

SEED = 20260809


def _hermitian_block(rng):
    """A Hermitian ``H`` over the transition block, with complex eigenvectors.

    Built as ``M + M†`` from a complex Gaussian: Hermitian to the last
    bit, and generic enough that no eigenvector is accidentally real (a
    real eigenvector basis makes both arms agree and the whole file a
    tautology — ``test_the_arms_really_do_differ`` refuses that).
    """
    M = rng.standard_normal((N_TRANS, N_TRANS)) + 1j * rng.standard_normal(
        (N_TRANS, N_TRANS))
    return M + M.conj().T


def _case():
    """(H, eigenvalues, A in driver layout, d in driver layout)."""
    rng = np.random.default_rng(SEED)
    H = _hermitian_block(rng)
    E, V = np.linalg.eigh(H)                       # columns are states
    A = V.T.reshape(N_TRANS, N_K, N_COND, N_VAL)   # (N, nk, nc, nv)
    d_flat = (rng.standard_normal((3, N_TRANS))
              + 1j * rng.standard_normal((3, N_TRANS)))
    d = d_flat.reshape(3, N_K, N_COND, N_VAL)      # (3, nk, nc, nv)
    return H, E, A, d, d_flat


def _resolvent(H, d_flat, z):
    """``⟨d|(z − H)^-1|d⟩`` per polarisation, by direct solve."""
    n = H.shape[0]
    X = np.linalg.solve(z * np.eye(n) - H, d_flat.T)          # (T, 3)
    return np.einsum("aT,Ta->a", d_flat.conj(), X)


def _sum_over_states(E, proj, z):
    """``Σ_S |proj_S|^2 / (z − E_S)`` per polarisation."""
    return np.einsum("Sa,S->a", np.abs(proj) ** 2, 1.0 / (z - E))


# ---------------------------------------------------------------------------
# 1. The adjudication
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("z", [2.0 + 0.30j, -1.5 + 0.05j])
def test_the_conjugated_dipole_contraction_reproduces_the_resolvent(z):
    """TRUE arm.  ``Σ_t A_t conj(d_t)`` satisfies the spectral identity.

    This is the derivation executed: the sum-over-states spectrum built
    from ``exciton_dipole_projections`` IS the resolvent the Haydock
    route continues-fraction its way to, at two unrelated complex
    frequencies (one well inside the spectrum, one below it), to machine
    precision.
    """
    H, E, A, d, d_flat = _case()
    proj = exciton_dipole_projections(A, d)
    np.testing.assert_allclose(
        _sum_over_states(E, proj, z), _resolvent(H, d_flat, z),
        rtol=1e-11, atol=1e-11,
        err_msg="the conjugated contraction did not reproduce ⟨d|(z−H)⁻¹|d⟩")


@pytest.mark.parametrize("z", [2.0 + 0.30j, -1.5 + 0.05j])
def test_the_unconjugated_contraction_breaks_the_resolvent(z):
    """CROSSED RED TWIN — the FALSE arm, run rather than described.

    ``Σ_t A_t d_t`` is what ``absorption_eigvecs`` shipped until
    2026-08-09.  It is a linear functional of ``A`` just like the true
    one, it is invariant under the same degenerate-subspace rotations,
    and it obeys the same f-sum over a COMPLETE set — which is why
    nothing cheaper than this identity catches it.  It fails (*), and by
    a wide margin, not a tolerance.
    """
    H, E, A, d, d_flat = _case()
    wrong = np.einsum("Nkcv,akcv->Na", A, d, optimize=True)
    got, want = _sum_over_states(E, wrong, z), _resolvent(H, d_flat, z)
    rel = float(np.max(np.abs(got - want) / np.abs(want)))
    assert rel > 1e-2, (
        f"the UNCONJUGATED contraction reproduced the resolvent to "
        f"{rel:.3e}, which cannot happen for a complex eigenbasis — the "
        f"synthetic H has lost its imaginary part and every cell in this "
        f"file is a tautology")


def test_the_arms_really_do_differ():
    """RED TWIN for the twin: the two arms are distinct on this fixture.

    If the synthetic block ever became real-symmetric, ``conj(d)`` and
    ``d`` would coincide, both arms would satisfy (*), and the cells
    above would pass while measuring nothing.
    """
    _, _, A, d, _ = _case()
    true_arm = exciton_dipole_projections(A, d)
    false_arm = np.einsum("Nkcv,akcv->Na", A, d, optimize=True)
    ratio = float(np.median(np.abs(true_arm) / np.abs(false_arm)))
    assert not np.allclose(np.abs(true_arm), np.abs(false_arm), rtol=1e-6), (
        f"the two contractions agreed in modulus (median ratio {ratio:.6f}); "
        f"the probe has no phase content")


# ---------------------------------------------------------------------------
# 2. One contraction, two drivers
# ---------------------------------------------------------------------------


def test_a_transposed_block_is_refused_not_broadcast():
    """RED TWIN.  A silent transpose is the neighbouring defect.

    ``(nk, nc, nv)`` and ``(nc, nv, nk)`` have the same element count, so
    a reshape-based contraction would happily produce a plausible wrong
    number from mismatched layouts.  The shape check is what stops that.
    """
    _, _, A, d, _ = _case()
    with pytest.raises(ValueError, match="transition axes"):
        exciton_dipole_projections(A, np.transpose(d, (0, 2, 3, 1)))


# ---------------------------------------------------------------------------
# 3. The non-TDA file, which has a different contraction and no driver
# ---------------------------------------------------------------------------

def _write_evec_h5(path, *, use_tda, coupling):
    import h5py
    with h5py.File(str(path), "w") as f:
        p = f.create_group("exciton_header/params")
        p.create_dataset("spin_kernel", data=3)
        p.create_dataset("nc", data=N_COND)
        p.create_dataset("nv", data=N_VAL)
        p.create_dataset("ns", data=1)
        p.create_dataset("use_tda", data=1 if use_tda else 0)
        kp = f.create_group("exciton_header/kpoints")
        kp.create_dataset("nk", data=N_K)
        kp.create_dataset("kpts", data=np.zeros((3, N_K)))
        shape = (1, 2, N_K, N_COND, N_VAL, 1, 2)
        d = f.create_group("exciton_data")
        d.create_dataset("eigenvalues", data=np.array([1.0, 2.0]))
        d.create_dataset("eigenvectors", data=np.zeros(shape))
        if coupling:
            d.create_dataset("eigenvectors_coupling", data=np.zeros(shape))


@pytest.mark.parametrize("use_tda,coupling", [(False, True), (True, True),
                                              (False, False)])
def test_non_tda_eigenvectors_are_refused(tmp_path, use_tda, coupling):
    """A non-TDA file used to yield the TDA answer, silently.

    ``bse_io.write_eigenvectors_stream`` persists the resonant X to
    ``eigenvectors`` and the coupling Y to ``eigenvectors_coupling``;
    this reader only ever returned X, so a full-BSE solve post-processed
    for absorption dropped the whole anti-resonant channel without a
    word.  Either witness — the ``use_tda`` flag or the presence of the
    coupling block — now refuses the file and names the contraction it
    would need.
    """
    p = tmp_path / "eigenvectors.h5"
    _write_evec_h5(p, use_tda=use_tda, coupling=coupling)
    with pytest.raises(NotImplementedError, match="NON-TDA"):
        load_eigenvectors_h5(str(p))


def test_a_tda_file_still_loads(tmp_path):
    """RED TWIN for the guard: it must not refuse the TDA files it exists for."""
    p = tmp_path / "eigenvectors.h5"
    _write_evec_h5(p, use_tda=True, coupling=False)
    eigvals, A, params = load_eigenvectors_h5(str(p))
    assert A.shape == (2, N_K, N_COND, N_VAL)
    assert params["nc"] == N_COND and params["nv"] == N_VAL
