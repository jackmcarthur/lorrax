"""First-principles contract for htransform's compact QP state overlay."""
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


_DEFAULT_KPOINTS = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])


def test_selected_htransform_source_is_this_worktree():
    import bandstructure.htransform as htransform

    source_root = Path(__file__).resolve().parents[1]
    assert Path(htransform.__file__).resolve().is_relative_to(source_root)


def _source_wfn(*, energies=None):
    if energies is None:
        energies = np.linspace(-1.0, 1.0, 64, dtype=np.float64).reshape(
            1, 2, 32)
    return SimpleNamespace(
        energies=np.asarray(energies, dtype=np.float64),
        kpoints=_DEFAULT_KPOINTS.copy(),
        nelec=8,
        nspinor=2,
        nbands=32,
        path=None,
    )


def _write_rotations(path, U, E_ry, band_range, *, kgrid=(2, 1, 1),
                     kpoints=None, source_wfn=True):
    from file_io.qp_wfn import write_qp_rotations_h5

    if kpoints is None:
        kpoints = _DEFAULT_KPOINTS
    write_qp_rotations_h5(
        str(path), U, np.asarray(E_ry) / 2.0,
        int(band_range[0]), int(band_range[1]), np.asarray(kpoints),
        *kgrid, k_storage="full",
        source_wfn=(_source_wfn() if source_wfn is True else source_wfn))


def _state(ctilde, enk, *, band_range=(10, 15), kgrid=(2, 1, 1),
           kpoints=None, wfn=None):
    import jax.numpy as jnp

    if kpoints is None:
        kpoints = _DEFAULT_KPOINTS
    if wfn is None:
        wfn = _source_wfn()
    return dict(
        basis=SimpleNamespace(
            ctilde=jnp.asarray(ctilde), band_range=band_range),
        enk_sigma=jnp.asarray(enk),
        sym=SimpleNamespace(unfolded_kpts=np.asarray(kpoints)),
        meta=SimpleNamespace(kgrid=kgrid),
        wfn=wfn,
        wfn_path="unstamped-mean-field-WFN.h5",
        log_fn=lambda *_a, **_k: None,
    )


def _unitaries(rng, nk, nb):
    result = []
    for _ in range(nk):
        raw = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
        q, r = np.linalg.qr(raw)
        phase = np.diag(r)
        phase = np.where(np.abs(phase) > 0, phase / np.abs(phase), 1.0)
        result.append(q * np.conj(phase)[None, :])
    return np.asarray(result)


def test_first_principles_u_f_u_dagger_identity():
    """The row rotation used by the implementation is the required operator."""
    rng = np.random.default_rng(9417)
    nk, nb, rank = 2, 4, 6
    C = rng.normal(size=(nk, nb, rank)) \
        + 1j * rng.normal(size=(nk, nb, rank))
    U = _unitaries(rng, nk, nb)
    fE = rng.normal(size=(nk, nb))

    C_qp = np.einsum("kmn,kma->kna", U, C, optimize=True)
    from_rotated_states = np.einsum(
        "kn,kna,knb->kab", fE, C_qp, np.conj(C_qp), optimize=True)
    in_dft_labels = np.einsum(
        "kmn,kn,kln->kml", U, fE, np.conj(U), optimize=True)
    direct = np.einsum(
        "kml,kma,klb->kab", in_dft_labels, C, np.conj(C), optimize=True)
    np.testing.assert_allclose(
        from_rotated_states, direct, rtol=3e-13, atol=3e-13)


def test_compact_rotation_is_u_f_u_dagger_and_keeps_dft_guards(tmp_path):
    """C_QP=U.T C makes the existing diagonal fH builder equal f(H_QP)."""
    pytest.importorskip("jax")
    from bandstructure.htransform import resolve_qp_hamiltonian_state

    rng = np.random.default_rng(9417)
    nk, nb_fit, nb_qp, rank = 2, 5, 3, 4
    fit_range, qp_range = (10, 15), (11, 14)
    C = rng.normal(size=(nk, nb_fit, rank)) \
        + 1j * rng.normal(size=(nk, nb_fit, rank))
    enk = rng.normal(size=(nb_fit, nk))
    U = _unitaries(rng, nk, nb_qp)
    E = rng.normal(size=(nk, nb_qp))
    path = tmp_path / "qp_wfn_rotations.h5"
    _write_rotations(path, U, E, qp_range)

    receipts = []
    C_qp_dev, enk_qp_dev, authenticated_range = resolve_qp_hamiltonian_state(
        **_state(C, enk, band_range=fit_range),
        qp_rotations_file=str(path), receipt_fn=receipts.append)
    C_qp = np.asarray(C_qp_dev)
    enk_qp = np.asarray(enk_qp_dev)
    assert authenticated_range == qp_range
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.artifact_path == str(path.resolve())
    assert receipt.band_range == qp_range
    assert receipt.kgrid == (2, 1, 1)
    assert receipt.source_wfn_fingerprint_scheme
    assert len(receipt.source_wfn_fingerprint) == 64

    # The canonical QP-WFN convention has no conjugation on U here.
    C_block = C[:, 1:4]
    np.testing.assert_allclose(
        C_qp[:, 1:4], np.einsum("kmn,kma->kna", U, C_block),
        rtol=2e-13, atol=2e-13)

    fE = rng.normal(size=(nk, nb_qp))
    fH_from_rotated_states = np.einsum(
        "kn,kna,knb->kab", fE, C_qp[:, 1:4],
        np.conj(C_qp[:, 1:4]), optimize=True)
    fH_in_dft_labels = np.einsum(
        "kmn,kn, kln->kml", U, fE, np.conj(U), optimize=True)
    fH_direct = np.einsum(
        "kml,kma,klb->kab", fH_in_dft_labels, C_block,
        np.conj(C_block), optimize=True)
    np.testing.assert_allclose(
        fH_from_rotated_states, fH_direct, rtol=3e-13, atol=3e-13)

    # The fully containing htransform window is a block-identity extension.
    np.testing.assert_array_equal(C_qp[:, 0], C[:, 0])
    np.testing.assert_array_equal(C_qp[:, 4], C[:, 4])
    np.testing.assert_array_equal(enk_qp[0], enk[0])
    np.testing.assert_array_equal(enk_qp[4], enk[4])
    np.testing.assert_allclose(enk_qp[1:4], E.T, rtol=0.0, atol=0.0)


def test_identity_rotation_is_exact_identity(tmp_path):
    pytest.importorskip("jax")
    from bandstructure.htransform import resolve_qp_hamiltonian_state

    rng = np.random.default_rng(19)
    C = rng.normal(size=(2, 3, 4)) + 1j * rng.normal(size=(2, 3, 4))
    enk = rng.normal(size=(3, 2))
    U = np.broadcast_to(np.eye(3, dtype=np.complex128), (2, 3, 3)).copy()
    E = enk.T.copy()
    path = tmp_path / "identity.h5"
    _write_rotations(path, U, E, (4, 7))

    got_C, got_E, authenticated_range = resolve_qp_hamiltonian_state(
        **_state(C, enk, band_range=(4, 7)),
        qp_rotations_file=str(path))
    np.testing.assert_array_equal(np.asarray(got_C), C)
    np.testing.assert_array_equal(np.asarray(got_E), enk)
    assert authenticated_range == (4, 7)


def test_qp_block_may_not_cut_through_the_fit_window(tmp_path):
    pytest.importorskip("jax")
    from bandstructure.htransform import resolve_qp_hamiltonian_state

    U = np.broadcast_to(np.eye(3, dtype=np.complex128), (2, 3, 3)).copy()
    E = np.zeros((2, 3))
    path = tmp_path / "cut.h5"
    _write_rotations(path, U, E, (9, 12))
    state = _state(np.ones((2, 5, 2)), np.zeros((5, 2)))
    with pytest.raises(ValueError, match="partial overlap/cut-through"):
        resolve_qp_hamiltonian_state(
            **state, qp_rotations_file=str(path))


@pytest.mark.parametrize("mismatch", ["kgrid", "kpoint"])
def test_qp_artifact_must_match_the_wfn_k_set(tmp_path, mismatch):
    pytest.importorskip("jax")
    from bandstructure.htransform import resolve_qp_hamiltonian_state

    U = np.broadcast_to(np.eye(2, dtype=np.complex128), (2, 2, 2)).copy()
    E = np.zeros((2, 2))
    path = tmp_path / f"bad-{mismatch}.h5"
    kwargs = {}
    if mismatch == "kgrid":
        kwargs["kgrid"] = (1, 2, 1)
    else:
        kwargs["kpoints"] = np.asarray(
            [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    _write_rotations(path, U, E, (11, 13), **kwargs)
    state = _state(np.ones((2, 5, 2)), np.zeros((5, 2)))
    match = "kgrid" if mismatch == "kgrid" else "full-BZ k-point"
    with pytest.raises(ValueError, match=match):
        resolve_qp_hamiltonian_state(
            **state, qp_rotations_file=str(path))


def test_qp_artifact_must_authenticate_the_source_wfn(tmp_path):
    pytest.importorskip("jax")
    from bandstructure.htransform import resolve_qp_hamiltonian_state

    U = np.broadcast_to(np.eye(2, dtype=np.complex128), (2, 2, 2)).copy()
    E = np.zeros((2, 2))
    state = _state(np.ones((2, 5, 2)), np.zeros((5, 2)))

    legacy = tmp_path / "legacy-unstamped.h5"
    _write_rotations(legacy, U, E, (11, 13))
    import h5py
    from file_io.qp_wfn import (
        QP_ROT_WFN_FINGERPRINT_ATTR,
        QP_ROT_WFN_FINGERPRINT_SCHEME_ATTR,
    )
    with h5py.File(legacy, "r+") as h5:
        del h5.attrs[QP_ROT_WFN_FINGERPRINT_ATTR]
        del h5.attrs[QP_ROT_WFN_FINGERPRINT_SCHEME_ATTR]
    with pytest.raises(ValueError, match="no authenticated source-WFN"):
        resolve_qp_hamiltonian_state(
            **state, qp_rotations_file=str(legacy))

    mismatched = tmp_path / "mismatched-wfn.h5"
    changed = _source_wfn()
    changed.energies = np.asarray(changed.energies).copy()
    changed.energies[0, 0, 0] += 1.0e-3
    _write_rotations(mismatched, U, E, (11, 13), source_wfn=changed)
    with pytest.raises(ValueError, match="different mean-field WFN"):
        resolve_qp_hamiltonian_state(
            **state, qp_rotations_file=str(mismatched))


def test_qp_state_sources_cannot_be_stacked(tmp_path):
    pytest.importorskip("jax")
    from file_io.qp_wfn import (
        QP_WFN_ATTR,
        QP_WFN_SCHEME,
        refuse_conflicting_qp_state_sources,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        refuse_conflicting_qp_state_sources(
            wfn_path="unstamped.h5", eqp_file="eqp1.dat",
            qp_rotations_file="qp_wfn_rotations.h5")

    qp_wfn = tmp_path / "WFN_qp.h5"
    with h5py.File(qp_wfn, "w") as h5:
        h5.attrs[QP_WFN_ATTR] = QP_WFN_SCHEME
        h5.attrs["qp_wfn_band_start"] = 2
        h5.attrs["qp_wfn_band_stop"] = 5
    for eqp, rotations in (("eqp1.dat", None),
                           (None, "qp_wfn_rotations.h5")):
        with pytest.raises(ValueError, match="already a LORRAX QP WFN"):
            refuse_conflicting_qp_state_sources(
                wfn_path=str(qp_wfn), eqp_file=eqp,
                qp_rotations_file=rotations)


def test_bse_eqp_frontends_reach_the_shared_stamp_owner():
    """The deepest shared correction seam covers every public BSE frontend."""
    pytest.importorskip("jax")
    import inspect
    from bse import bse_window

    adapter = inspect.getsource(bse_window.refuse_eqp_on_a_qp_wfn)
    correction = inspect.getsource(bse_window.apply_eqp_corrections)
    assert "refuse_conflicting_qp_state_sources" in adapter
    assert "refuse_conflicting_qp_state_sources" in correction
    source_root = Path(__file__).resolve().parents[1]
    for path in (
            "src/bse/bse_jax.py", "src/bse/absorption_haydock.py",
            "src/bse/davidson_absorption.py", "src/bse/bse_loading.py"):
        text = (source_root / path).read_text(encoding="utf8")
        assert ("apply_eqp_and_reslice_bands" in text
                or "apply_eqp_corrections" in text), path
