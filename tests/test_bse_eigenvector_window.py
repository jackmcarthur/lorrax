"""Which band window ``eigenvectors.h5`` describes, and who is allowed to say.

THE DEFECT THIS FILE CLOSES.  ``--n-val``/``--n-cond`` are a REQUEST.  The
loader clamps it to what the restart file holds and then, under
``--band-degeneracy snap`` (the default from ``824032b7`` until the owner
flipped it to ``strict`` on 2026-08-10, and an explicit opt-in since), widens
it outward past any cut multiplet — on the flagship Si deck ``n_cond 4``
becomes ``8``.  The flip narrows how often the gap between request and
resolved window opens; it does not close it, because ``snap`` still exists and
the clamp is unconditional, so every cell below still stands.
``bse_jax._preview_lanczos`` nonetheless handed ``write_eigenvectors_stream``
the *request*, so the HDF5 dataset was created at the requested ``nc`` and the
write of the real component died with ``TypeError: Can't broadcast``, leaving a
truncated ``eigenvectors.h5`` behind.  ``bse.absorption_eigvecs`` therefore
could not be run end to end on any snapping deck, which is how a wrong
conjugation survived in it for as long as it did
(``FIX_absorption_conjugation.md`` §8; ``SMALL_ISSUES.md`` row 32).

THREE NUMBERS, AND ONLY ONE OF THEM IS THE ANSWER.  For a window that snapped
3 → 4 conduction bands on a 2×2 mesh there are three counts in flight:

    the CLI request              3   — stale the moment the guard fires
    the loader's resolved count  4   — the bands actually solved  ← the answer
    the mesh-rounded pad extent  4   — what the ARRAY is shaped by, not bands

``nc``/``nv`` in the file are what ``absorption_eigvecs`` slices ``dipole.h5``
with, against ``n_occ``, so they have to name real bands: the resolved count.
The pad is dropped BY COUNT, the same spelling as ``pad_zone_mask_np`` — and
because a stale request and a mesh pad both make the incoming array *wider*
than the declared window and cannot be told apart by shape, the trim is checked
by VALUE: the pad block is decoupled by construction and exactly zero, so a
discarded block carrying weight is not pad and is refused.  Trading the old
loud crash for a silent truncation would be the worse bug.

Every cell carries its red twin: the clean window that must round-trip in
silence next to the stale one that must refuse.  Fixture-free — no deck, no
GPU, no FFI; the contract is about counts, and a synthetic block carries it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import h5py
import numpy as np
import pytest

from bse.bse_io import write_eigenvectors_stream

SRC = Path(__file__).resolve().parent.parent / "src" / "bse"

#: The snapped window: 3 conduction bands requested, 4 resolved.  NOT square
#: in (c, v) — a square block lets an accidental transpose pass, which is the
#: neighbouring defect in the same class.
N_COND_REQ = 3
N_COND, N_VAL = 4, 2
NKX, NKY, NKZ = 2, 1, 1
NK = NKX * NKY * NKZ
N_WRITE = 3

SEED = 20260809


def _eigen_block(n_cond_pad=N_COND, n_val_pad=N_VAL, seed=SEED):
    """``(N_WRITE, 1, nc_pad, nv_pad, NK)`` TDA eigenvectors, pad exactly zero.

    Complex and non-symmetric on purpose: a real or symmetric block cannot
    distinguish the valence flip from its own inverse.
    """
    rng = np.random.default_rng(seed)
    shape = (N_WRITE, 1, n_cond_pad, n_val_pad, NK)
    vecs = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    # The pad block is EXACT zero — ψ pad = 0 and every kernel term is
    # bilinear in ψ, so the pad decouples and the solver never puts weight
    # there.  This file relies on that fact, so it states it.
    vecs[:, :, N_COND:, :, :] = 0.0
    vecs[:, :, :, N_VAL:, :] = 0.0
    return vecs


def _eigenvalues():
    return np.linspace(0.20, 0.26, N_WRITE)


def _write(tmp_path, vecs, n_val, n_cond, name="eigenvectors.h5", **kw):
    out = tmp_path / name
    write_eigenvectors_stream(
        str(out), _eigenvalues(), vecs, n_val, n_cond,
        NKX, NKY, NKZ, N_WRITE, **kw)
    return out


# ---------------------------------------------------------------------------
# 1.  the resolved window reaches the file
# ---------------------------------------------------------------------------
def test_the_post_snap_window_is_what_the_file_reports_p1(tmp_path):
    """Declared 4v-window ⇒ ``nc``/``nv`` and the dataset shape both say 4×2."""
    out = _write(tmp_path, _eigen_block(), N_VAL, N_COND)
    with h5py.File(out, "r") as f:
        params = f["exciton_header/params"]
        assert int(params["nc"][()]) == N_COND
        assert int(params["nv"][()]) == N_VAL
        assert int(params["bse_hamiltonian_size"][()]) == NK * N_COND * N_VAL
        assert f["exciton_data/eigenvectors"].shape == (
            1, N_WRITE, NK, N_COND, N_VAL, 1, 2)


def test_the_pre_snap_request_is_refused_not_written_p1(tmp_path):
    """RED TWIN of the cell above — and of the crash the row reported.

    Declaring the stale request (3) against a solve that carried 4 discards a
    band with real weight.  It must refuse, by name, and leave nothing behind:
    the pre-fix failure wrote a truncated file before dying.
    """
    out = tmp_path / "eigenvectors.h5"
    with pytest.raises(ValueError, match=r"DISCARD"):
        write_eigenvectors_stream(
            str(out), _eigenvalues(), _eigen_block(), N_VAL, N_COND_REQ,
            NKX, NKY, NKZ, N_WRITE)
    assert not out.exists(), (
        "a refused write must not leave a truncated eigenvectors.h5 — that is "
        "the half of the original defect that was worse than the crash")


def test_a_window_wider_than_the_solve_is_refused_p1(tmp_path):
    """The other direction: declaring more bands than were solved."""
    with pytest.raises(ValueError, match=r"does not fit"):
        write_eigenvectors_stream(
            str(tmp_path / "e.h5"), _eigenvalues(), _eigen_block(),
            N_VAL, N_COND + 1, NKX, NKY, NKZ, N_WRITE)


# ---------------------------------------------------------------------------
# 2.  the mesh pad is dropped by count, and only the pad
# ---------------------------------------------------------------------------
def test_the_mesh_pad_is_trimmed_and_the_values_are_untouched_p1(tmp_path):
    """A 4×2 window padded to 6×4 writes the SAME file as the unpadded one.

    This is the case a writer that trusted the array's own shape would get
    wrong — it would export two zero conduction bands under real band labels.
    """
    plain = _eigen_block()
    padded = np.zeros((N_WRITE, 1, 6, 4, NK), dtype=plain.dtype)
    padded[:, :, :N_COND, :N_VAL, :] = plain[:, :, :N_COND, :N_VAL, :]

    a = _write(tmp_path, plain, N_VAL, N_COND, name="plain.h5")
    b = _write(tmp_path, padded, N_VAL, N_COND, name="padded.h5")
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        va = fa["exciton_data/eigenvectors"][:]
        vb = fb["exciton_data/eigenvectors"][:]
        assert va.shape == vb.shape == (1, N_WRITE, NK, N_COND, N_VAL, 1, 2)
        assert np.array_equal(va, vb)


def test_weight_in_the_dropped_block_is_refused_p1(tmp_path):
    """RED TWIN of the trim: pad that is not pad does not get silently dropped.

    Same shapes as the cell above, one non-zero element outside the declared
    window.  The trim's licence is that the discarded block is exactly zero,
    so the cell that would notice it stopped being zero has to exist.
    """
    padded = np.zeros((N_WRITE, 1, 6, 4, NK), dtype=np.complex128)
    plain = _eigen_block()
    padded[:, :, :N_COND, :N_VAL, :] = plain[:, :, :N_COND, :N_VAL, :]
    padded[0, 0, N_COND, 0, 0] = 1.0e-3
    with pytest.raises(ValueError, match=r"DISCARD"):
        _write(tmp_path, padded, N_VAL, N_COND)


# ---------------------------------------------------------------------------
# 3.  the layout the trim must not disturb
# ---------------------------------------------------------------------------
def test_the_valence_axis_is_still_flipped_on_write_p1(tmp_path):
    """Trim-then-flip must land the same values BGW's convention wants.

    Our internal slice puts ``v=0`` at the DEEPEST valence; BGW's file puts it
    at the HIGHEST (``BSE/input_fi.f90:407``).  Flipping before trimming would
    slide the pad columns to the front and keep them as the "highest valence"
    bands, so the order of the two operations is a fact worth pinning.
    """
    vecs = _eigen_block()
    out = _write(tmp_path, vecs, N_VAL, N_COND)
    with h5py.File(out, "r") as f:
        got = f["exciton_data/eigenvectors"][0]        # (nS, nk, nc, nv, ns, 2)
    for s in range(N_WRITE):
        want = np.transpose(vecs[s, 0, :N_COND, :N_VAL, :], (2, 0, 1))[:, :, ::-1]
        assert np.allclose(got[s, :, :, :, 0, 0], want.real, atol=0, rtol=0)
        assert np.allclose(got[s, :, :, :, 0, 1], want.imag, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 4.  the callers, at the source — the cells that would have caught it
# ---------------------------------------------------------------------------
def _call_kwargs_and_args(path: Path, callee: str):
    """Every call to ``callee`` in ``path``, as (positional ids, keyword ids)."""
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name != callee:
            continue
        pos = [a.id if isinstance(a, ast.Name) else None for a in node.args]
        kw = {k.arg: (k.value.id if isinstance(k.value, ast.Name) else None)
              for k in node.keywords}
        found.append((pos, kw))
    return found


@pytest.mark.parametrize("driver,callee", [
    ("bse_jax.py", "write_eigenvectors_stream"),
    ("absorption_haydock.py", "slice_dipole_to_bse_window"),
])
def test_no_driver_feeds_a_band_consumer_its_raw_request_p1(driver, callee):
    """``n_val``/``n_cond`` as bare names are the CLI request. They are stale.

    Both drivers resolve the window through a loader that may widen it, then
    hand the counts to something that slices bands — the writer in one case,
    the dipole table in the other.  Passing the bare parameter name is exactly
    the defect, in both, and it is invisible on a non-snapping deck, so the
    gate has to read the source rather than wait for a deck that snaps.
    """
    calls = _call_kwargs_and_args(SRC / driver, callee)
    assert calls, f"no call to {callee} found in {driver} — did it move?"
    for pos, kw in calls:
        names = set(n for n in pos if n) | set(v for v in kw.values() if v)
        assert not (names & {"n_val", "n_cond"}), (
            f"{driver} passes the raw request to {callee}: {sorted(names)}.  "
            f"Use the loader's resolved counts (data['n_val'] / "
            f"data['n_cond']) — --band-degeneracy may have widened them.")


def test_the_loaders_agree_on_the_names_the_resolved_window_travels_under_p1():
    """Both loaders must publish the window under the same four keys.

    ``_load_ring_subset`` (1 device) used not to return any of them, so its
    caller re-derived the counts from ``psi_c.shape[1]`` — the PADDED number —
    or kept using its own request.  A single-device escape hatch that reports
    its window differently from the sharded loader is how the two routes come
    to disagree about what a given request means.
    """
    tree = ast.parse((SRC / "bse_loading.py").read_text())
    wanted = {"n_val", "n_cond", "n_val_pad", "n_cond_pad"}
    loaders = {"_load_ring_subset", "load_bse_data_from_restart_sharded"}
    seen = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        if fn.name not in loaders:
            continue
        seen.add(fn.name)
        keys = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Dict):
                keys |= {k.value for k in node.keys
                         if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        assert wanted <= keys, (
            f"{fn.name} does not publish {sorted(wanted - keys)} in its bundle")
    # Both loaders must have BEEN THERE to be measured.  Without this the cell
    # passes vacuously the day either one is renamed or moved to another file —
    # which is exactly what the 2026-08-10 bse_io split would have done to it.
    assert seen == loaders, (
        f"only {sorted(seen)} found in bse/bse_loading.py; this cell measures "
        f"nothing about {sorted(loaders - seen)} and has gone stale")
