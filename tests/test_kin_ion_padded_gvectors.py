"""Owner decision D10 — ngkmax-padded G loading must agree with the ragged
path to 1e-12, and the pad mask must be load-bearing.

The three operator kernels ``gw.kin_ion_io`` drives per k
(``compute_kinetic_k``, ``compute_local_V_k``, ``vnl_ops.vnl_matrix``
through ``vnl_matrix_from_kdata``) each take a ``g_mask`` hook.  Feeding
them the loader's fixed-shape ``(ngkmax, 3)`` G table plus that mask is
what collapses "one JIT lowering per DISTINCT ngk" to one, and it is what
makes the k loop uniform enough to pipeline behind a single readback.

WHAT EACH TEST PROVES, and how it can fail
------------------------------------------
Every agreement assertion here is paired with a NEGATIVE CONTROL that
runs the same code with the mask removed.  That is deliberate: a pad row
is a *valid* FFT-box index (it has to be — a gather at a pad slot must
not run off the box), so a masking bug does not raise, it silently adds
``ngkmax − ngk`` extra copies of one component.  A green
"padded == ragged" on its own would also be green if both sides were
equally wrong; the RED control is what rules that out.

**The pad value changed on 2026-08-08 and this file was re-derived, not
re-run.**  ``WfnLoader.gvecs()`` used to pad with ``(0, 0, 0)`` — Miller
Γ, a physical component of every G-sphere, which is what made the leak
invisible.  It now pads with the FFT-box **pad sentinel**
(``common.gvec_fft_box.fft_box_pad_sentinel``): the Nyquist corner
``(-nx/2, -ny/2, -nz/2)``, a cell no physical G may occupy — an invariant
``pad_gvecs_to_sentinel`` ENFORCES rather than assumes.  Three
consequences, each pinned below:

* The controls are no longer "wrong by a wide margin" assertions with a
  hand-picked threshold.  Both kernels contract per-G after building
  their G-independent factor, so the leak has a CLOSED FORM: it is
  exactly ``npad ×`` the same kernel evaluated on the one-row G-list
  ``[sentinel]``.  The controls assert that identity to ``RTOL_D10``.
  A leak of any other size — a different pad value, a pad row landing
  somewhere else, a mask silently applied — falsifies it.
* The leak got LOUDER, which is the point of the corner: the sentinel
  carries the largest ``|k+G|²`` in the box.  MEASURED on this fixture,
  the unmasked kinetic block moves by 3.157e-01 of its own scale, versus
  7.762e-04 under the old Γ pad — 407× more visible.  ``V_loc`` moves by
  1.796e+00 vs 1.454e+00.
* ``gw.kin_ion_io`` can now REFUSE an unmasked padded list on the pad
  marker itself, catching even a single pad row; the old guard needed
  ``> 1`` all-zero row.

``kin_ion`` matrix elements sit inside H₀'s ~500 eV cancellation
(⟨T+V_ion+V_NL⟩ ≈ −502 eV, ⟨V_H⟩ ≈ +461 eV, sum ≈ −42 eV), so the
tolerance is stated on the *absolute* matrix element, in the same Ry the
kernels return.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from common.gvec_fft_box import fft_box_pad_sentinel
from psp.dft_operators import (
    PaddedGVectors,
    gather_psi_G_from_crys,
    padded_gvectors,
)
from psp.get_DFT_mtxels import compute_kinetic_k, compute_local_V_k


def _write_kin_ion_provenance_twin(path, *, bispinor):
    import h5py

    with h5py.File(path, "w") as h5:
        ds = h5.create_dataset(
            "kin_ion", data=np.zeros((1, 2, 2), dtype=np.complex128))
        if bispinor is not None:
            ds.attrs["bispinor"] = bool(bispinor)


def test_kin_ion_provenance_refuses_wrong_or_missing_bispinor_before_read(
        tmp_path, monkeypatch):
    """Same-shaped two- and four-spinor mean fields are not interchangeable."""
    import h5py
    from file_io import kin_ion as owner

    true_path = tmp_path / "kin_ion_true.h5"
    false_path = tmp_path / "kin_ion_false.h5"
    legacy_path = tmp_path / "kin_ion_unstamped.h5"
    _write_kin_ion_provenance_twin(true_path, bispinor=True)
    _write_kin_ion_provenance_twin(false_path, bispinor=False)
    _write_kin_ion_provenance_twin(legacy_path, bispinor=None)

    reads = []
    original_getitem = h5py.Dataset.__getitem__

    def record_payload_read(dataset, key):
        if dataset.name == "/kin_ion":
            reads.append(key)
        return original_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", record_payload_read)
    import pytest
    with pytest.raises(ValueError, match="requires a resolved Hartree source"):
        owner.validate_kin_ion_against_run(
            true_path, expected_bispinor=True,
            selected_hartree_source="auto",
            print_fn=lambda *_args: None)
    assert owner.validate_kin_ion_against_run(
        true_path, expected_bispinor=True,
        selected_hartree_source="isdf",
        print_fn=lambda *_args: None)["bispinor"]

    with pytest.raises(ValueError, match="bispinor=False.*bispinor=True"):
        owner.validate_kin_ion_against_run(
            false_path, expected_bispinor=True,
            selected_hartree_source="isdf",
            print_fn=lambda *_args: None)
    with pytest.raises(ValueError, match="no bispinor provenance stamp"):
        owner.validate_kin_ion_against_run(
            legacy_path, expected_bispinor=True,
            selected_hartree_source="isdf",
            print_fn=lambda *_args: None)
    assert reads == [], "provenance gate read a kin_ion matrix payload"


def test_pauli_reference_hartree_requires_explicit_two_plus_four_receipts(
        tmp_path):
    import h5py
    import pytest
    from common.four_current_model import (
        PAULI_REFERENCE_BARE_TRANSVERSE_MODEL,
        PAULI_TWO_SPINOR_CHARGE_REPRESENTATION,
        RAW_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION,
    )
    from file_io import kin_ion as owner

    path = tmp_path / "kin_ion_pauli_reference.h5"
    with h5py.File(path, "w") as h5:
        ds = h5.create_dataset(
            "kin_ion", data=np.zeros((1, 2, 2), dtype=np.complex128))
        ds.attrs["bispinor"] = True
        ds.attrs["bispinor_gw_mode"] = (
            PAULI_REFERENCE_BARE_TRANSVERSE_MODEL)
        ds.attrs["charge_representation"] = (
            PAULI_TWO_SPINOR_CHARGE_REPRESENTATION)
        ds.attrs["spatial_current_representation"] = (
            RAW_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION)
        vh = h5.create_dataset(
            owner.HARTREE_DATASET,
            data=np.zeros((1, 2, 2), dtype=np.complex128))
        vh.attrs["matrix_nspinor"] = 2
        vh.attrs["charge_representation"] = (
            PAULI_TWO_SPINOR_CHARGE_REPRESENTATION)

    owner.validate_kin_ion_against_run(
        path, expected_bispinor=True,
        expected_bispinor_gw_mode=PAULI_REFERENCE_BARE_TRANSVERSE_MODEL,
        selected_hartree_source="stored", print_fn=lambda *_args: None)
    with h5py.File(path, "r+") as h5:
        h5[owner.HARTREE_DATASET].attrs["matrix_nspinor"] = 4
    with pytest.raises(ValueError, match="requires matrix_nspinor=2"):
        owner.validate_kin_ion_against_run(
            path, expected_bispinor=True,
            expected_bispinor_gw_mode=(
                PAULI_REFERENCE_BARE_TRANSVERSE_MODEL),
            selected_hartree_source="stored", print_fn=lambda *_args: None)


def test_fractional_hartree_receipt_checks_support_and_wfn_both_directions(
        tmp_path, monkeypatch):
    import h5py
    import pytest
    from types import SimpleNamespace
    from common import parallel_transport
    from file_io import kin_ion as owner

    path = tmp_path / "kin_ion_fractional.h5"
    current_fingerprint = "a" * 64
    monkeypatch.setattr(
        parallel_transport, "wfn_fingerprint",
        lambda _wfn: current_fingerprint)
    with h5py.File(path, "w") as h5:
        ds = h5.create_dataset(
            "kin_ion", data=np.zeros((1, 2, 2), dtype=np.complex128))
        ds.attrs["bispinor"] = False
        ds.attrs["wfn_fingerprint_scheme"] = (
            parallel_transport.WFN_FINGERPRINT_SCHEME)
        ds.attrs["wfn_fingerprint"] = current_fingerprint
        ds.attrs["exact_hartree_occupation_policy"] = (
            owner.HARTREE_OCCUPATION_POLICY)
        ds.attrs["exact_hartree_expected_electrons"] = 1.25
        ds.attrs["exact_hartree_density_band_stop"] = 4
        vh = h5.create_dataset(
            owner.HARTREE_DATASET,
            data=np.zeros((1, 2, 2), dtype=np.complex128))
        vh.attrs["density_band_stop"] = 4

    wfn = SimpleNamespace(
        occupations_are_exact_integer=False,
        num_electrons=1.25,
        physical_density_band_stop=4)
    owner.validate_kin_ion_against_run(
        path, expected_bispinor=False, selected_hartree_source="stored",
        wfn=wfn, print_fn=lambda *_args: None)

    with h5py.File(path, "r+") as h5:
        del h5["kin_ion"].attrs["exact_hartree_expected_electrons"]
    with pytest.raises(ValueError, match="atomically"):
        owner.validate_kin_ion_against_run(
            path, expected_bispinor=False,
            selected_hartree_source="stored", wfn=wfn,
            print_fn=lambda *_args: None)

    with h5py.File(path, "r+") as h5:
        h5["kin_ion"].attrs["exact_hartree_expected_electrons"] = 1.25
        h5[owner.HARTREE_DATASET].attrs["density_band_stop"] = 3
    with pytest.raises(ValueError, match="density support"):
        owner.validate_kin_ion_against_run(
            path, expected_bispinor=False,
            selected_hartree_source="stored", wfn=wfn,
            print_fn=lambda *_args: None)

    with h5py.File(path, "r+") as h5:
        h5[owner.HARTREE_DATASET].attrs["density_band_stop"] = 4
        h5["kin_ion"].attrs["wfn_fingerprint"] = "b" * 64
    wfn.occupations_are_exact_integer = True
    with pytest.raises(ValueError, match="different WFN fingerprint"):
        owner.validate_kin_ion_against_run(
            path, expected_bispinor=False,
            selected_hartree_source="stored", wfn=wfn,
            print_fn=lambda *_args: None)

    with h5py.File(path, "r+") as h5:
        for name in ("occupation_policy", "expected_electrons",
                     "density_band_stop"):
            del h5["kin_ion"].attrs[f"exact_hartree_{name}"]
    with pytest.raises(ValueError, match="different WFN fingerprint"):
        owner.validate_kin_ion_against_run(
            path, expected_bispinor=False,
            selected_hartree_source="isdf", wfn=wfn,
            print_fn=lambda *_args: None)


def test_live_hartree_receipt_reuses_exact_loaded_wfn_binding(
        tmp_path, monkeypatch):
    """The live GW gate must not rescan a WFN already bound by gw_init."""
    import h5py
    from types import SimpleNamespace
    from common import parallel_transport
    from file_io import kin_ion as owner

    path = tmp_path / "kin_ion_bound_wfn.h5"
    fingerprint = "c" * 64
    scans = []
    monkeypatch.setattr(
        parallel_transport, "wfn_fingerprint",
        lambda source: (scans.append(source), fingerprint)[1])
    wfn = SimpleNamespace(
        occupations_are_exact_integer=False,
        num_electrons=1.25,
        physical_density_band_stop=4)
    binding = parallel_transport.bind_wfn_fingerprint(wfn)
    assert scans == [wfn]

    with h5py.File(path, "w") as h5:
        ds = h5.create_dataset(
            "kin_ion", data=np.zeros((1, 2, 2), dtype=np.complex128))
        ds.attrs["bispinor"] = False
        ds.attrs["wfn_fingerprint_scheme"] = (
            parallel_transport.WFN_FINGERPRINT_SCHEME)
        ds.attrs["wfn_fingerprint"] = fingerprint
        ds.attrs["exact_hartree_occupation_policy"] = (
            owner.HARTREE_OCCUPATION_POLICY)
        ds.attrs["exact_hartree_expected_electrons"] = 1.25
        ds.attrs["exact_hartree_density_band_stop"] = 4
        vh = h5.create_dataset(
            owner.HARTREE_DATASET,
            data=np.zeros((1, 2, 2), dtype=np.complex128))
        vh.attrs["density_band_stop"] = 4

    def refuse_rescan(_source):
        raise AssertionError("live Hartree validation rescanned the WFN")

    monkeypatch.setattr(
        parallel_transport, "wfn_fingerprint", refuse_rescan)
    owner.validate_kin_ion_against_run(
        path, expected_bispinor=False, selected_hartree_source="stored",
        wfn=wfn, wfn_fingerprint_binding=binding,
        print_fn=lambda *_args: None)
    assert scans == [wfn]


# D10's gate.  Bit-exactness is explicitly NOT required: appending exact
# zeros cannot change a sum in IEEE-754, but a shape change does move
# XLA's choice of reduction BLOCKING, so the two routes may associate the
# same nonzero terms differently.
RTOL_D10 = 1e-12

_GRID = (6, 6, 8)
# (-3, -3, -4) on this grid; box cell (3, 3, 4).
_SENTINEL, _SENTINEL_FLAT = fft_box_pad_sentinel(_GRID)


# ---------------------------------------------------------------------------
# Fixtures: a small but non-degenerate plane-wave problem
# ---------------------------------------------------------------------------

def _fixture(nb=4, nspinor=2, grid=_GRID, ngk=37, ngkmax=48, seed=7):
    """(ψ box, ragged G, padded G, mask, k, bdot, V_r, cell_volume).

    The G list is drawn WITHOUT replacement from the box's index set and
    deliberately includes ``(0,0,0)`` at a non-zero position: Γ is the
    row the OLD pad aliased, so keeping it here means the ragged
    reference still exercises the component the previous contract got
    wrong.

    The sentinel's flat slot is EXCLUDED from the draw.  That is not
    fixture convenience — it is the invariant
    ``common.gvec_fft_box.pad_gvecs_to_sentinel`` enforces on every real
    table ("no physical G in the pad sentinel's box cell"), and the
    refusal in ``gw.kin_ion_io`` is only sound because of it.  A fixture
    that drew the corner anyway would be testing a table the loader
    refuses to build.
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = grid

    pool = np.setdiff1d(np.arange(nx * ny * nz),
                        np.asarray([_SENTINEL_FLAT]))
    flat = rng.choice(pool, size=ngk, replace=False)
    G = np.stack(np.unravel_index(flat, (nx, ny, nz)), axis=-1).astype(np.int32)
    G[3] = np.zeros(3, dtype=np.int32)                    # force a Γ row
    # De-duplicate against the forced Γ row so the ragged list stays a set.
    keep = [0] * ngk
    seen = set()
    for i, row in enumerate(G):
        t = tuple(int(v) for v in row)
        keep[i] = t not in seen
        seen.add(t)
    G = G[np.asarray(keep, dtype=bool)]
    ngk = int(G.shape[0])
    assert not np.any(np.all(G % np.asarray(grid) == np.asarray(
        [nx // 2, ny // 2, nz // 2]), axis=1)), \
        "fixture drew a physical G on the sentinel cell"

    pad = ngkmax - ngk
    assert pad > 0, "fixture must actually pad"
    G_pad = np.concatenate(
        [G, np.broadcast_to(_SENTINEL, (pad, 3)).astype(np.int32)], axis=0)
    mask = np.concatenate([np.ones(ngk), np.zeros(pad)]).astype(np.float64)

    psi = (rng.standard_normal((nb, nspinor, nx, ny, nz))
           + 1j * rng.standard_normal((nb, nspinor, nx, ny, nz)))
    psi_box = jnp.asarray(psi, dtype=jnp.complex128)

    A = rng.standard_normal((3, 3))
    bdot = A @ A.T + 3.0 * np.eye(3)                      # SPD metric
    kvec = np.asarray([0.125, -0.25, 0.0])
    V_r = jnp.asarray(rng.standard_normal((nx, ny, nz)), dtype=jnp.float64)
    return psi_box, G, G_pad, mask, kvec, bdot, V_r, 137.5


def _dev(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _scale(a) -> float:
    return float(np.max(np.abs(np.asarray(a))))


def _one_sentinel_row() -> np.ndarray:
    """The one-row G-list ``[sentinel]`` — the unit of the pad leak."""
    return np.asarray(_SENTINEL, dtype=np.int32)[None, :]


# ---------------------------------------------------------------------------
# 1. The kernels: padded+masked == ragged, and the mask is load-bearing
# ---------------------------------------------------------------------------

def test_kinetic_padded_matches_ragged_and_mask_is_load_bearing():
    psi, G, G_pad, mask, kvec, bdot, _V, _vol = _fixture()
    npad = int(G_pad.shape[0] - G.shape[0])

    ragged = compute_kinetic_k(psi, G, kvec, bdot)
    padded = compute_kinetic_k(psi, G_pad, kvec, bdot, g_mask=mask)
    unmasked = compute_kinetic_k(psi, G_pad, kvec, bdot)      # NEGATIVE CONTROL

    scale = _scale(ragged)
    assert scale > 1.0, "fixture produced a degenerate T"
    assert _dev(padded, ragged) <= RTOL_D10 * scale

    # RED, and red for a DERIVED reason.  ``_compute_kinetic_k_jit``
    # builds T_G from the G-list, gathers ψ at each G's box cell and
    # contracts ``einsum('msg,nsg->mn')`` — a sum over independent g.  So
    # dropping the mask adds exactly ``npad`` copies of the same kernel
    # run on the single-row list ``[sentinel]``.  Asserting THAT (rather
    # than "moved by more than 1e-6") is what makes this control
    # falsifiable: change the pad value, or let a mask leak in, and the
    # predicted leak stops matching.
    leak = np.asarray(compute_kinetic_k(psi, _one_sentinel_row(), kvec, bdot))
    assert _dev(unmasked, np.asarray(ragged) + npad * leak) <= RTOL_D10 * scale
    # …and the leak is not a rounding-scale nuisance: the sentinel sits at
    # the box corner, the largest |k+G|² there is.  MEASURED 3.157e-01.
    assert _scale(npad * leak) > 0.1 * scale


def test_local_V_padded_matches_ragged_and_mask_is_load_bearing():
    psi, G, G_pad, mask, _k, _bdot, V_r, vol = _fixture()
    npad = int(G_pad.shape[0] - G.shape[0])

    ragged = compute_local_V_k(psi, G, V_r, vol)
    padded = compute_local_V_k(psi, G_pad, V_r, vol, g_mask=mask)
    unmasked = compute_local_V_k(psi, G_pad, V_r, vol)        # NEGATIVE CONTROL

    scale = _scale(ragged)
    assert scale > 0.0
    assert _dev(padded, ragged) <= RTOL_D10 * scale

    # Same closed form.  ``_compute_local_V_k_jit`` builds φ_G from the
    # WHOLE box (the G-list plays no part) and only then contracts
    # ``einsum('bsg,nsg->bn')``, so the per-pad-row contribution is again
    # the one-row kernel.  MEASURED leak 1.796e+00 of scale.
    leak = np.asarray(compute_local_V_k(psi, _one_sentinel_row(), V_r, vol))
    assert _dev(unmasked, np.asarray(ragged) + npad * leak) <= RTOL_D10 * scale
    assert _scale(npad * leak) > 0.1 * scale


def test_gather_psi_G_padded_zeroes_pad_columns():
    """The V_NL route masks ψ_G, not Z — check the ψ side directly.

    ``vnl_ops.vnl_matrix`` contracts ψ_G against Z twice, so zeroing ψ_G
    at the pad columns is sufficient to make Z's (finite, evaluated at
    K = kvec) pad values inert.  This pins that ψ-side contract.
    """
    psi, G, G_pad, mask, *_ = _fixture()

    ragged = np.asarray(gather_psi_G_from_crys(psi, G))
    padded = np.asarray(gather_psi_G_from_crys(psi, G_pad, mask))

    ngk = int(G.shape[0])
    assert np.array_equal(padded[..., :ngk], ragged)
    assert np.all(padded[..., ngk:] == 0)

    # NEGATIVE CONTROL, stated as the MECHANISM rather than "nonzero":
    # an unmasked pad column is ψ read at the sentinel's box cell.  Under
    # the old Γ pad this was ψ(0,0,0) — a physical coefficient of the
    # list, which is precisely why the leak was invisible.
    cell = tuple(int(v) % int(n) for v, n in zip(_SENTINEL, _GRID))
    unmasked = np.asarray(gather_psi_G_from_crys(psi, G_pad))
    expected = np.asarray(psi)[..., cell[0], cell[1], cell[2]]
    assert np.array_equal(
        unmasked[..., ngk:],
        np.broadcast_to(expected[..., None], unmasked[..., ngk:].shape))
    assert np.max(np.abs(unmasked[..., ngk:])) > 0.0


# ---------------------------------------------------------------------------
# 2. The sum-of-terms: the whole ⟨m|T+V|n⟩ block, which is what H₀ uses
# ---------------------------------------------------------------------------

def test_summed_operator_block_agrees_at_1e12():
    """T + V_loc together, the combination ``get_kin_ion_k`` returns.

    Tested as a sum because that is the quantity D10 gates: the pieces
    could each drift within tolerance and still cancel differently.
    """
    psi, G, G_pad, mask, kvec, bdot, V_r, vol = _fixture(nb=6, ngk=41)

    ragged = (compute_kinetic_k(psi, G, kvec, bdot)
              + compute_local_V_k(psi, G, V_r, vol))
    padded = (compute_kinetic_k(psi, G_pad, kvec, bdot, g_mask=mask)
              + compute_local_V_k(psi, G_pad, V_r, vol, g_mask=mask))

    scale = _scale(ragged)
    assert _dev(padded, ragged) <= RTOL_D10 * scale


def _wfn_stub(bdot, fft_grid=None):
    """Minimal ``wfn`` for ``get_kin_ion_k`` — bdot / cell_volume / grid.

    ``fft_grid`` is optional on purpose: it is what selects which arm of
    the guard is available, and both arms are exercised below.
    """
    class _W:
        cell_volume = 137.5
    w = _W()
    w.bdot = bdot
    if fft_grid is not None:
        w.fft_grid = tuple(int(v) for v in fft_grid)
    return w


def test_get_kin_ion_k_refuses_a_padded_list_without_its_mask():
    """The mask contract is ENFORCED, not merely documented.

    A pad row is a valid box index, so an unmasked padded list returns a
    plausible-looking matrix that is wrong by ``ngkmax − ngk`` extra
    copies of the sentinel-cell component.  ``_refuse_padded_gvecs_
    without_mask`` has TWO arms and this exercises both, plus the case
    each one fails to see.

    Arm (b), the pad-value-agnostic backstop: a physical G-sphere is a
    SET (and the full-BZ unfold ``G ↦ S·G − G_umklapp`` is injective, so
    it stays one), therefore ANY duplicate row means padding.  Available
    with no ``fft_grid``.
    """
    import pytest
    from gw.kin_ion_io import get_kin_ion_k

    psi, G, G_pad, mask, kvec, bdot, V_r, vol = _fixture()
    w = _wfn_stub(bdot)                              # NO fft_grid → arm (b)

    with pytest.raises(ValueError, match="duplicate row"):
        get_kin_ion_k(psi, G_pad, kvec, V_r, None, w)

    # The RAGGED list must still be accepted without a mask — otherwise
    # the guard is just breaking the old path.  It is a set (the fixture
    # de-duplicates) and carries no sentinel row.
    out = get_kin_ion_k(psi, G, kvec, V_r, None, w)
    assert np.isfinite(np.asarray(out)).all()
    # ...and the padded list WITH its mask agrees with it at 1e-12.
    out_pad = get_kin_ion_k(psi, G_pad, kvec, V_r, None, w, g_mask=mask)
    assert _dev(out_pad, out) <= RTOL_D10 * _scale(out)


def test_the_duplicate_arm_is_blind_to_a_single_pad_row():
    """Construct the case where arm (b) returns FALSE.

    One pad row is not a duplicate of anything, so the set test cannot
    see it — exactly the blind spot the pre-2026-08 ``> 1 all-zero row``
    guard also had.  This is the gap arm (a) exists to close.  Asserting
    the blindness HERE is what stops the next test from being a check
    that passes for the wrong reason, and the numeric half shows the
    blind spot is not harmless: the one leaked row moves the T+V block
    by 2.9e-2 of its own scale (MEASURED).

    Tested on ``refuse_padded_gvecs_without_mask`` directly rather than
    through ``get_kin_ion_k``: importing ``gw.kin_ion_io`` runs
    ``initialize_communicator_stack``, which refuses without a built FFI
    ``.so``.  The integration path is covered by
    ``test_get_kin_ion_k_refuses_a_padded_list_without_its_mask`` above.
    """
    from common.gvec_fft_box import refuse_padded_gvecs_without_mask

    psi, G, _Gp, _m, kvec, bdot, V_r, vol = _fixture()
    G_pad1 = np.concatenate(
        [G, np.asarray(_SENTINEL, dtype=np.int32)[None, :]], axis=0)

    # FALSE: no fft_grid, one pad row, nothing duplicated → accepted.
    refuse_padded_gvecs_without_mask(G_pad1, None)

    # …and it really would have been wrong, by exactly the one-row leak.
    one = _one_sentinel_row()
    ragged = (np.asarray(compute_kinetic_k(psi, G, kvec, bdot))
              + np.asarray(compute_local_V_k(psi, G, V_r, vol)))
    leaked = (np.asarray(compute_kinetic_k(psi, G_pad1, kvec, bdot))
              + np.asarray(compute_local_V_k(psi, G_pad1, V_r, vol)))
    leak = (np.asarray(compute_kinetic_k(psi, one, kvec, bdot))
            + np.asarray(compute_local_V_k(psi, one, V_r, vol)))
    scale = _scale(ragged)
    assert _dev(leaked, ragged + leak) <= RTOL_D10 * scale
    assert _dev(leaked, ragged) > 1e-3 * scale


def test_the_sentinel_arm_catches_a_single_pad_row():
    """Arm (a): with ``fft_grid`` known, ONE pad row is enough.

    Sound in the direction that matters — every pad row IS the sentinel,
    so a padded list always fires.  The other direction (a ragged list
    that legitimately holds the corner would be REFUSED) is a real false
    positive, not excluded by construction; it is excluded by
    MEASUREMENT: no WFN available — the five fixtures under ``tests/``,
    nor the production MoS₂ decks (36×36×135 / 80 Ry, 24×24×80) — has a
    physical G on the corner cell in any k, margin ``(9,9,32)`` vs
    ``(18,18,67)``.  The fixture's exclusion of the corner from its draw
    mirrors that measured fact rather than assuming it away.
    """
    import pytest
    from common.gvec_fft_box import refuse_padded_gvecs_without_mask

    _psi, G, _Gp, _m, *_ = _fixture()
    G_pad1 = np.concatenate(
        [G, np.asarray(_SENTINEL, dtype=np.int32)[None, :]], axis=0)

    with pytest.raises(ValueError, match="pad sentinel cell"):
        refuse_padded_gvecs_without_mask(G_pad1, _GRID)

    # Ragged still accepted with the grid present — the guard must not
    # simply be refusing everything.
    refuse_padded_gvecs_without_mask(G, _GRID)

    # A row at ``+nx/2`` is a DIFFERENT Miller index but the SAME box
    # cell, and must still be refused: the box cell is what the ψ gather
    # reads, so an unmasked pad there is indistinguishable from the
    # sentinel.  Miller-triple equality would MISS this.
    nx, ny, nz = _GRID
    aliased = np.concatenate(
        [G, np.asarray([[nx // 2, ny // 2, nz // 2]], dtype=np.int32)], axis=0)
    assert not np.array_equal(aliased[-1], np.asarray(_SENTINEL))
    with pytest.raises(ValueError, match="pad sentinel cell"):
        refuse_padded_gvecs_without_mask(aliased, _GRID)


# ---------------------------------------------------------------------------
# 3. PaddedGVectors itself
# ---------------------------------------------------------------------------

class _FakeLoader:
    """Minimal stand-in for ``WfnLoader``'s paired k/G accessors.

    ``padded_gvectors`` reaches the loader through ``_as_loader``, which
    accepts a real ``WfnLoader`` or falls back to reopening ``_filename``.
    Duck-typing the three methods keeps this test free of a WFN fixture;
    the full-deck agreement is measured by the sbatch harness, not here.
    """

    def __init__(self, gvecs, ngk):
        self._g, self._n = gvecs, ngk
        self._k = np.arange(gvecs.shape[0] * 3, dtype=np.float64).reshape(-1, 3)

    def gvecs(self, *, k="full_bz"):
        return self._g

    def ngk_valid(self, *, k="full_bz"):
        return self._n

    def kvecs(self, *, k="full_bz"):
        return self._k


def test_padded_gvectors_mask_matches_ngk_valid(monkeypatch):
    import psp.dft_operators as dop

    nk, ngkmax = 4, 9
    ngk = np.asarray([9, 7, 5, 8], dtype=np.int32)
    rng = np.random.default_rng(1)
    g = rng.integers(0, 5, size=(nk, ngkmax, 3)).astype(np.int32)
    for j in range(nk):
        g[j, int(ngk[j]):] = _SENTINEL                    # loader's pad rows

    fake = _FakeLoader(g, ngk)
    monkeypatch.setattr(dop, "_as_loader", lambda w: fake)

    tab = padded_gvectors(object())
    assert isinstance(tab, PaddedGVectors)
    assert tab.ngkmax == ngkmax and tab.n_k == nk
    assert np.array_equal(tab.kvecs, fake._k)
    assert tab.mask.sum(axis=1).tolist() == ngk.tolist()
    for j in range(nk):
        G_j, m_j = tab.at(j)
        assert np.array_equal(m_j[: int(ngk[j])], np.ones(int(ngk[j])))
        assert np.all(m_j[int(ngk[j]):] == 0.0)
        # Pad rows carry the sentinel — still a VALID box index (which is
        # why the mask is load-bearing), just a detectable one.  The mask
        # comes from ``ngk_valid`` and never from the pad value: that is
        # what this asserts, by handing over a table whose pad rows the
        # fake loader chose.
        assert np.array_equal(
            G_j[int(ngk[j]):],
            np.broadcast_to(_SENTINEL, (ngkmax - int(ngk[j]), 3)))


def test_padded_gvectors_refuses_unpaired_k_rows(monkeypatch):
    import pytest
    import psp.dft_operators as dop

    fake = _FakeLoader(
        np.zeros((2, 4, 3), dtype=np.int32),
        np.asarray([4, 4], dtype=np.int32))
    fake._k = np.zeros((1, 3), dtype=np.float64)
    monkeypatch.setattr(dop, "_as_loader", lambda w: fake)

    with pytest.raises(ValueError, match="coordinate half of the G table"):
        padded_gvectors(object())


def test_padded_gvectors_refuses_invalid_logical_extents(monkeypatch):
    import pytest
    import psp.dft_operators as dop

    fake = _FakeLoader(
        np.zeros((2, 4, 3), dtype=np.int32),
        np.asarray([4, 5], dtype=np.int32))
    monkeypatch.setattr(dop, "_as_loader", lambda w: fake)
    with pytest.raises(ValueError, match="rows must lie"):
        padded_gvectors(object())

    fake._n = np.asarray([4], dtype=np.int32)
    with pytest.raises(ValueError, match="one logical extent per G row"):
        padded_gvectors(object())


def test_sweep_consumers_take_kvecs_from_the_same_gtab():
    """Every migrated sweep pairs ``gtab.gvecs`` with ``gtab.kvecs``."""
    root = Path(__file__).resolve().parents[1]
    for relpath, expected in (
            ("src/gw/kin_ion_io.py", 2),
            ("src/gw/sc_iteration.py", 1),
    ):
        tree = ast.parse((root / relpath).read_text(encoding="utf-8"))
        paired = 0
        for call in (node for node in ast.walk(tree)
                     if isinstance(node, ast.Call)):
            keywords = {kw.arg: kw.value for kw in call.keywords
                        if kw.arg is not None}
            if ast.unparse(keywords.get("gvecs")) != "gtab.gvecs":
                continue
            paired += 1
            assert ast.unparse(keywords.get("kvecs")) == "gtab.kvecs", (
                f"{relpath}:{call.lineno} separates the physical k "
                "representative from the PaddedGVectors G rows")
        assert paired == expected, (
            f"{relpath}: expected {expected} gtab-backed sweeps, found {paired}")


# ---------------------------------------------------------------------------
# 4. sweep_local_k — the one-readback contract
# ---------------------------------------------------------------------------

def test_sweep_local_k_places_blocks_by_global_index():
    from common.collectives import sweep_local_k

    rng = np.random.default_rng(3)
    ref = {ik: jnp.asarray(rng.standard_normal((2, 2))
                           + 1j * rng.standard_normal((2, 2)))
           for ik in (1, 4, 7)}

    calls: list[int] = []

    def per_k(ik):
        calls.append(ik)
        return ref[ik]

    vals, idx = sweep_local_k([1, 4, 7], 5, (2, 2), per_k,
                              print_fn=lambda *a: None)

    assert calls == [1, 4, 7]
    assert vals.shape == (5, 2, 2)
    assert idx.tolist() == [1, 4, 7, -1, -1]
    for j, ik in enumerate((1, 4, 7)):
        assert np.array_equal(vals[j], np.asarray(ref[ik]))
    # Unused slots stay zero — ``_gather_indexed_blocks`` skips them by
    # index, but a nonzero pad would corrupt any consumer that does not.
    assert not np.any(vals[3:])


def test_sweep_local_k_empty_rank_keeps_the_collective_shape():
    """world > nk leaves ranks with NO k — they still join the gather.

    ``_gather_indexed_blocks`` ends in ``process_allgather``, a
    collective: a rank that returned a differently-shaped placeholder
    would mis-assemble the result on every rank, not just its own.  This
    is the exact failure P=64 on a 16-k deck would hit.
    """
    from common.collectives import sweep_local_k

    def per_k(ik):                      # never called
        raise AssertionError("empty rank must not dispatch")

    vals, idx = sweep_local_k([], 1, (6, 6), per_k, print_fn=lambda *a: None)
    assert vals.shape == (1, 6, 6)
    assert vals.dtype == np.complex128
    assert not np.any(vals)
    assert idx.tolist() == [-1]


def test_sweep_local_k_refuses_a_shape_mismatch():
    """The declared item_shape is enforced, not trusted."""
    from common.collectives import sweep_local_k
    import pytest

    with pytest.raises(ValueError, match="item_shape"):
        sweep_local_k([0], 1, (4, 4), lambda ik: jnp.zeros((3, 3)),
                      print_fn=lambda *a: None)


def test_sweep_local_k_lookahead_does_not_change_values():
    """Pipelining depth is a scheduling knob, never a numerical one."""
    from common.collectives import sweep_local_k

    psi, G, G_pad, mask, kvec, bdot, V_r, vol = _fixture(nb=3)

    def per_k(ik):
        return compute_kinetic_k(psi * (ik + 1.0), G_pad, kvec, bdot,
                                 g_mask=mask)

    a, _ = sweep_local_k(range(4), 4, (3, 3), per_k, lookahead=1,
                         print_fn=lambda *x: None)
    b, _ = sweep_local_k(range(4), 4, (3, 3), per_k, lookahead=8,
                         print_fn=lambda *x: None)
    assert np.array_equal(a, b)
