"""THE RED TWINS for the screening-content declaration.

The 2026-08-09 n_p = 1 bridge gate came back at Sigma_c = -130.651 eV
against a Godby-Needs +0.6754 eV on the same two samples, and the cause
was that the W(omega) store had been filled with the FULL screened
interaction where the multipole method fits the correlation part
W_c = W - v (MPA_THEORY 1.1/1.2).  Measured on that store, |v| is
104-119 % of |W| at the probe frequency.

WHAT MAKES IT A DECLARATION AND NOT A CHECK.  Nothing on the consuming
side can see it.  The fit reproduces whatever it is handed -- the
production store's backward error is 4.0e-16 -- the tensor is Hermitian
either way, the pole positions stay in the fourth quadrant, and the
k-star relation SURVIVES, because v_q is symmetry-covariant: the
full-BZ Sigma cube from the defective field carries exactly the 8
distinct band-8 values this mesh admits, with the same k partition as
the Godby-Needs arm beside it.  So every twin below is about a REFUSAL
firing, and the last one is about the refusal firing on the shape of
silence -- an undeclared store -- because that is what the defective
production stores actually are.
"""
import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("symmetry_maps.qirr_store")

from file_io import mpa_store as MS                       # noqa: E402


def _fit_store(path, content, *, n_q=1, n_mu=2, n_p=1, unit="Ry"):
    MS.allocate_fit_store(path, n_q=n_q, n_mu=n_mu, n_p=n_p,
                          energy_unit=unit, screening_content=content)
    rng = np.random.default_rng(3)
    Om = (rng.uniform(0.5, 2.0, size=(n_p, n_mu, n_mu))
          - 0.1j * rng.uniform(0.0, 1.0, size=(n_p, n_mu, n_mu)))
    B = rng.normal(size=(n_p, n_mu, n_mu)) + 0j
    diag = {"condition": np.full((n_mu, n_mu), 2.0),
            "backward_error": np.full((n_mu, n_mu), 1e-15)}
    for q in range(n_q):
        MS.write_fit_block(path, q, list(range(n_mu)), Om, B, diag)
    MS.finalize_fit_store(path)
    return path


def test_a_store_that_holds_W_is_refused_BY_NAME(tmp_path):
    """The twin for the defect itself: 'W' must not reach a Sigma.

    Not "must not be preferred" -- must not be READABLE by the consumer
    gate at all.  The message has to name the object, because the file is
    otherwise a perfectly good fit store and the reader who sees this is
    holding 23 GB of poles that took eight minutes to make.
    """
    p = _fit_store(str(tmp_path / "w.h5"), "W")
    led = MS.fit_completion_ledger(p)
    assert led["screening_content"] == "W"
    with pytest.raises(ValueError) as exc:
        MS.require_correlation_part(led["screening_content"],
                                    where="test", source=p)
    msg = str(exc.value)
    assert "W_c = W - v" in msg
    assert "-130.651" in msg or "130.651" in msg
    # And it says what to DO, not merely that it is unhappy.
    assert "Wc0_q = W0_q - V_q" in msg


def test_an_UNDECLARED_store_is_refused_and_not_assumed(tmp_path):
    """THE SILENT-FALLBACK TWIN, and the one that matters most.

    Every fit store written before 2026-08-09 -- including both
    production ones -- carries no declaration at all.  A reader that
    treated 'no attr' as 'must be the correlation part, everyone knows
    that' would pass exactly the two files that hold W.  So absence is a
    refusal with its own message, and the test asserts the message names
    the migration call rather than just failing.
    """
    p = str(tmp_path / "legacy.h5")
    _fit_store(p, "W_c")
    import h5py
    with h5py.File(p, "a") as f:
        del f.attrs[MS.FIT_SCREENING_CONTENT_ATTR]
    led = MS.fit_completion_ledger(p)
    assert led["screening_content"] is None
    with pytest.raises(ValueError) as exc:
        MS.require_correlation_part(led["screening_content"],
                                    where="test", source=p)
    msg = str(exc.value)
    assert "does not declare" in msg
    assert "declare_fit_screening_content" in msg
    # A declared W_c store passes the same call, or the twin proves nothing.
    assert MS.require_correlation_part(
        MS.fit_completion_ledger(
            _fit_store(str(tmp_path / "ok.h5"), "W_c"))["screening_content"],
        where="test") == "W_c"


def test_the_allocator_has_no_default_and_no_guessed_spelling(tmp_path):
    """No default, and an unknown spelling refuses instead of coercing.

    The sibling of the energy-unit declaration and for the same reason: a
    defaulted declaration is not a declaration, it is the old behaviour
    with a new attr beside it.
    """
    with pytest.raises(ValueError) as exc:
        MS.allocate_fit_store(str(tmp_path / "x.h5"), n_q=1, n_mu=2, n_p=1,
                              energy_unit="Ry")
    assert "no default" in str(exc.value)
    with pytest.raises(ValueError, match="not one of"):
        MS.allocate_fit_store(str(tmp_path / "y.h5"), n_q=1, n_mu=2, n_p=1,
                              energy_unit="Ry", screening_content="screened")
    with pytest.raises(ValueError, match="not one of"):
        MS.canonical_screening_content("W minus v", where="test")
    # Case-insensitive IN, canonical OUT -- a script's 'w_c' is 'W_c'.
    assert MS.canonical_screening_content("w_c", where="t") == "W_c"
    assert MS.canonical_screening_content("  w ", where="t") == "W"


def test_a_declaration_is_never_replaced_by_a_different_one(tmp_path):
    """Re-declaring the SAME value is a no-op; a different one refuses.

    The bytes did not change, so at most one of two declarations is
    true.  Without this a setup script could 'fix' a store by relabelling
    it, and the relabelled file would be indistinguishable from a correct
    one.
    """
    p = _fit_store(str(tmp_path / "d.h5"), "W")
    assert MS.declare_fit_screening_content(p, "W") == "W"
    with pytest.raises(ValueError, match="already declares"):
        MS.declare_fit_screening_content(p, "W_c")
