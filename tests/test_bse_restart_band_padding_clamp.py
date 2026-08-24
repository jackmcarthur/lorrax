"""The BSE restart loader must clamp n_val/n_cond against the LOGICAL band
count, not the padded ``enk_full``/``psi_full_y`` array shape.

THE DEFECT (KNOWN_LORRAX_ISSUES.md, ``src/bse/bse_loading.py`` padding-clamp
row, 2026-08-23).  ``enk_full`` / ``psi_full_y`` are written at
``round_up(nband, world_size)`` bands (``file_io.tagged_arrays``), so a run
whose ``nband`` does not divide the WRITING process count leaves phantom
bands on disk with no marker distinguishing them from real ones.  Both BSE
restart readers used to read ``n_cond_available`` off the padded array's OWN
shape (``enk_full.shape[1]`` / ``n_bands_total = enk_full.shape[1]``), so a
window request near the true edge silently resolved past it into the
padding.  Measured consequences, all from the SAME root cause: a
``head_delta_mismatch`` crash, a TRS-gauge crash, and (this file's own
concern) a SILENT pair-basis-window over-grant that no shape check catches
because the padded array is genuinely that wide.

THE FIRST FIX ATTEMPT READ ``band_window``'s ``b4`` AND WAS WRONG.  ``b4``
looked like a natural "logical count" stamp (``assert_restart_window_matches``
reads it for a related, but different, purpose), but
``file_io.wavefunction_bundle.BandSlices`` documents and ENFORCES that
``b4`` IS the padded top ("``b4`` is ``max(b4_chi, b4_sigma)`` PADDED to the
world size"), for every writer, ``low_mem_bands`` or not.  On a real P=16
Si_scalar restart (nband=20) ``band_window`` reads ``[0,0,4,14,32]`` -- the
SAME 32 as ``enk_full.shape[1]``.  That first fix was a no-op; it passed a
P=4 cell only because ``round_up(20,4)==20`` made the padding a no-op there
too.  This fixture therefore does NOT stamp ``band_window`` with the
logical count at all -- a fixture that did would certify a fix that fails
on a real restart file, which is exactly what happened once
(KNOWN_LORRAX_ISSUES.md carries the measured correction).

THE SIGNAL THE FIX ACTUALLY USES, measured on that same real restart: a
padding band's ``psi_full_y`` row is EXACT ZERO (the writer never copies a
band the deck did not ask for), while ``enk_full``'s padding rows are NOT
zero -- ordinary-looking energies, because the underlying WFN often has
more DFT bands than the deck requested and ``enk_full`` is written
un-clipped.  So this fixture's padding rows are zero in ``psi_full_y`` and
an out-of-place sentinel in ``enk_full`` (999.0) -- deliberately NOT
correlated, so a fix that (wrongly) inferred the count from ``enk_full``
instead of ``psi_full_y`` would show it immediately.

WHY THIS FIXTURE NEEDS NO GPU AND NO MULTI-RANK MESH.  The padding this bug
reads incorrectly is baked into the ARRAY at *write* time by whatever
process count wrote it -- it has nothing to do with the mesh the file is
*read* back on.  So "a P that does not divide nband" is reproduced by hand:
build a restart whose ``enk_full``/``psi_full_y`` are padded to
``round_up(nband_logical, P_write)`` for a ``P_write`` that does not divide
``nband_logical``, and read it back through the real P=1 loader.

Both readers get their own cell (module docstring: "Both publish the
resolved window under the same four names ... take the band window from
the same guard" -- so a fix to one without the other would be exactly the
shadow-accounting drift QUALITY_PATTERNS #3 warns about).
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from bse import bse_loading

N_MU = 4
NKX, NKY, NKZ = 1, 1, 1
NSPINOR = 1

# n_occ=2 valence, 4 REAL conduction bands (indices 2..5) -> logical nband=6.
N_OCC = 2
N_COND_LOGICAL = 4
NBAND_LOGICAL = N_OCC + N_COND_LOGICAL          # 6

# 16 does NOT divide 6: round_up(6, 16) = 16, ten phantom bands appended.
P_WRITE = 16
NBAND_PADDED = 16
assert NBAND_LOGICAL % P_WRITE != 0
assert NBAND_PADDED > NBAND_LOGICAL

_SEED = 20260823


def _energies():
    """Well-separated real energies, then an obviously-distinct sentinel on
    every phantom band -- NOT the signal the fix reads (psi_full_y is), so
    a fix that accidentally keyed off enk_full would still see this and a
    correct fix must NOT let it leak into eps_c either way."""
    real = np.array([-2.0, -1.0, 1.0, 2.0, 3.0, 4.0])       # NBAND_LOGICAL
    assert real.shape[0] == NBAND_LOGICAL
    phantom = np.full(NBAND_PADDED - NBAND_LOGICAL, 999.0)
    return np.concatenate([real, phantom])[None, :]          # (nk=1, nb)


def _write_restart(path, *, padded: bool):
    """``padded=True``: the P=16-write fixture (the hazard) -- ``psi_full_y``
    zero on every phantom band, exactly what the real writer produces.
    ``padded=False``: the red twin's clean sibling -- band count already
    logical, nothing to detect, so a fix that broke the ordinary path would
    show up here instead of only on the padded arm."""
    rng = np.random.default_rng(_SEED)

    def _cplx(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

    nb = NBAND_PADDED if padded else NBAND_LOGICAL
    enk = _energies() if padded else _energies()[:, :NBAND_LOGICAL]
    psi = _cplx((1, nb, NSPINOR, N_MU))
    if padded:
        # THE ACTUAL SIGNAL: a padding band was never written, so it is
        # exactly zero -- not "small", not "noisy", the literal np.zeros a
        # real writer's pre-allocated buffer leaves untouched.
        psi[:, NBAND_LOGICAL:, :, :] = 0.0
    V = _cplx((1, 1, 1, NKX, NKY, NKZ, N_MU, N_MU))
    W0 = _cplx((1, 1, 1, NKX, NKY, NKZ, N_MU, N_MU))
    g0 = _cplx((N_MU,))

    with h5py.File(path, "w") as f:
        dv = f.create_dataset("V_qmunu", data=V)
        dv.attrs["V_ready"] = True
        dw = f.create_dataset("W0_qmunu", data=W0)
        dw.attrs["W0_ready"] = True
        f.create_dataset("psi_full_y", data=psi)
        f.create_dataset("enk_full", data=enk)
        f.create_dataset("G0_mu_nu", data=g0)
        f.create_dataset("vhead", data=1.5)
        f.create_dataset("whead", data=np.array([1.5], dtype=np.complex128))
        f.create_dataset("kgrid", data=np.array([NKX, NKY, NKZ]))
        # A REAL restart carries this too (BandSlices' padded b4, NOT the
        # logical count -- see module docstring).  Stamped here at its
        # REAL (padded) value so a fix that still reads it by mistake is
        # caught rather than accidentally passing on a fixture nicer than
        # reality.
        f.create_dataset(
            "band_window",
            data=np.array([0, 0, N_OCC, nb, nb], dtype=np.int64))


@pytest.fixture()
def padded_restart(tmp_path):
    path = tmp_path / "padded.h5"
    _write_restart(path, padded=True)
    return str(path)


@pytest.fixture()
def clean_restart(tmp_path):
    path = tmp_path / "clean.h5"
    _write_restart(path, padded=False)
    return str(path)


def _mesh_1x1():
    import jax
    from jax.sharding import Mesh
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
               axis_names=("x", "y"))


# ---------------------------------------------------------------------------
# 1. THE RED TWIN: the pre-fix formula, exercised directly and shown wrong.
# ---------------------------------------------------------------------------
def test_the_shape_only_formula_reads_the_padding_as_real_bands():
    """Not a claim about the fix -- a claim about the HAZARD it closes.

    ``enk_full.shape[1]`` on the padded fixture is 16; the true conduction
    count under it is 16-2=14, ten of them padding.  If this stopped being
    true the fixture would not be exercising the bug at all and every green
    cell below would be vacuous (QUALITY_PATTERNS: a check that cannot fail
    is not evidence)."""
    padded_shape1 = NBAND_PADDED
    naive_n_cond_available = padded_shape1 - N_OCC
    assert naive_n_cond_available == 14
    assert naive_n_cond_available != N_COND_LOGICAL


def test_the_band_window_formula_ALSO_reads_the_padding_as_real_bands():
    """The FIRST (wrong) fix attempt, exercised directly and shown wrong.

    ``band_window``'s ``b4`` on the padded fixture is ALSO 16 (the padded
    top, by ``BandSlices`` construction -- see module docstring), so a fix
    keyed off it resolves to the SAME wrong 14, not the logical 4."""
    band_window_b4 = NBAND_PADDED
    naive_n_cond_available = band_window_b4 - N_OCC
    assert naive_n_cond_available == 14
    assert naive_n_cond_available != N_COND_LOGICAL


# ---------------------------------------------------------------------------
# 2. THE FIX, through the real P=1 sharded loader.
# ---------------------------------------------------------------------------
def test_sharded_loader_clamps_to_the_logical_count_not_the_padded_one(
        padded_restart):
    data = bse_loading.load_bse_data_from_restart_sharded(
        padded_restart, n_val=100, n_cond=100, mesh_xy=_mesh_1x1(),
        n_occ=N_OCC, cell_volume=270.0)
    assert data["n_cond"] == N_COND_LOGICAL, (
        f"resolved n_cond={data['n_cond']}, expected the LOGICAL "
        f"{N_COND_LOGICAL} -- {data['n_cond']} bands were read means the "
        f"padded extent leaked through")
    assert data["n_val"] == N_OCC
    eps_c = np.asarray(data["eps_c"])
    # No sentinel value anywhere in what was actually read as conduction.
    assert not np.any(np.isclose(eps_c, 999.0)), (
        "a phantom (padding) band's sentinel energy reached eps_c")


def test_sharded_loader_clean_sibling_is_unaffected(clean_restart):
    """The red twin's clean pair: an unpadded file (write P divided nband)
    resolves the SAME way with or without the fix, so this closes only the
    padded case and does not silently narrow the ordinary one."""
    data = bse_loading.load_bse_data_from_restart_sharded(
        clean_restart, n_val=100, n_cond=100, mesh_xy=_mesh_1x1(),
        n_occ=N_OCC, cell_volume=270.0)
    assert data["n_cond"] == N_COND_LOGICAL
    assert data["n_val"] == N_OCC


# ---------------------------------------------------------------------------
# 3. THE SAME FIX, through the single-device ring-subset loader — the
#    module docstring's "same guard" promise, checked rather than assumed.
# ---------------------------------------------------------------------------
def test_ring_subset_loader_clamps_to_the_logical_count_not_the_padded_one(
        padded_restart):
    data = bse_loading._load_ring_subset(
        padded_restart, n_val=100, n_cond=100, px=1, py=1, n_occ=N_OCC)
    assert data["n_cond"] == N_COND_LOGICAL, (
        f"resolved n_cond={data['n_cond']}, expected the LOGICAL "
        f"{N_COND_LOGICAL}")
    eps_c = np.asarray(data["eps_c"])
    assert not np.any(np.isclose(eps_c, 999.0))


# ---------------------------------------------------------------------------
# 4. Degenerate back-compat: a restart whose psi_full_y is ALL zero at k=0
#    (nothing loaded at all -- a different, upstream defect) falls back to
#    the array's own shape rather than resolving to 0 and obscuring it.
# ---------------------------------------------------------------------------
def test_an_all_zero_psi_file_falls_back_to_the_array_shape(tmp_path):
    path = tmp_path / "allzero.h5"
    _write_restart(path, padded=True)
    with h5py.File(path, "r+") as f:
        f["psi_full_y"][...] = 0.0
    data = bse_loading.load_bse_data_from_restart_sharded(
        str(path), n_val=100, n_cond=100, mesh_xy=_mesh_1x1(),
        n_occ=N_OCC, cell_volume=270.0)
    assert data["n_cond"] == NBAND_PADDED - N_OCC
