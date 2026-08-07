"""Layer L-c: four REAL processes, a real 2x2 mesh, the collective read.

This is the tier the other two cannot reach.  ``_auto_pick_backend``
returns ``eager`` whenever ``jax.process_count() <= 1``, so an emulated
2x2 (layer L-b) exercises the eager backend and NOTHING else, and every
claim about the phdf5 collective MPI-IO read on a mesh bigger than 1x1
has to come from here::

    lx run --cpu -N 1 -n 4 python3 \\
        services/wfn_loader/tests/test_wfn_loader_multiproc.py --mesh 2x2
    lx run -N 1 -G 4 -n 4 python3 \\
        services/wfn_loader/tests/test_wfn_loader_multiproc.py --mesh 2x2

ONE SET OF CHECK BODIES, THREE CALLERS — the ``_CLI_CELLS`` pattern taken
from ``services/distrib_la/tests/test_distrib_la_multiproc.py`` rather
than reinvented.  Every ``check_*(mesh, ...)`` below is called by a pytest
cell (on whatever mesh this process can build, which is 1x1), by
``_cli_main`` under ``srun``, AND by
``tests/bench/wfn_loader_backend_parity_test.py``, which is now a thin
argv wrapper over these same functions.  Duplicating the logic across the
three would mean the multi-rank leg testing something slightly different
from the thing the suite pins, which is how a matrix leg drifts out of
agreement with its own reference.

WHAT THIS FILE COVERS THAT NOTHING ELSE DOES

* THE BAND-PAD CLAMP CLASS.  ``22049c3`` shipped a per-rank band clamp
  that used the FILE's ``mnband`` where it needed the REQUEST's ``b_hi``.
  The two agree on every divisible geometry, so the defect lived for
  months behind a green suite whose only multi-rank parity harness
  defaulted to ``--bands 0,4`` on a 4-rank mesh.  ``check_band_pad_clamp_
  parity`` is that harness with the defaults that would have caught it,
  and ``check_band_bound_negative_control`` is the FALSIFICATION: it
  reintroduces the perturbation in a test-local twin of the counts table
  and asserts the twin DIFFERS on the tail rank here and MATCHES on a
  divisible control.  A negative control that cannot go red is decoration.

* THE SENTINEL/ZERO CONJUNCTION ON A SHARDED MULTI-RANK LOAD (survey
  §7.4).  "Mask detectable ≠ mask optional": ψ zero AND ``gvecs`` at the
  FFT-box pad sentinel for every slot past ``ngk_valid``, both backends,
  both k modes.  Covered at P=1 (L-a) and on an emulated 2x2 (L-b); this
  is the first time it is asserted on ψ that a COLLECTIVE READ produced.

* THE FIRST CUDA-PLATFORM EXECUTION of the promoted ``SlabIO.read_slabs``
  door end to end (commits 7b32c2b + 22b37df).  The step-0 fold
  measurement was CPU-milan; the union read's device-staging tail
  (``cudaMemcpyAsync`` H2D rather than a host memcpy) is a different code
  path in the same handler and had never run under this loader.

* PER-RANK DIFFERENT WINDOWS.  ``load_process_local`` exists so rank *r*
  can ask for ``k=[7]`` while rank *s* asks for ``k=[9]`` with no
  collective, no barrier and no cross-rank shape agreement
  (``gw.kin_ion_io``'s ρ sweep).  A single-process cell cannot tell that
  from a lucky ordering; four ranks asking for four different shapes
  either completes or hangs.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pytest

# Path first: this file runs as a bare script under srun, where nothing has
# put services/*/src anywhere (`lx` rewrites the container PYTHONPATH to
# exactly <checkout>/src).
_TESTS = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))
_REPO = os.path.dirname(_SERVICES)
for _svc in ("lxkit", "wfn_loader"):
    _src = os.path.join(_SERVICES, _svc, "src")
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)
_LORRAX_SRC = os.path.join(_REPO, "src")
if os.path.isdir(_LORRAX_SRC) and _LORRAX_SRC not in sys.path:
    sys.path.append(_LORRAX_SRC)

# CLI multi-rank mode: jax.distributed.initialize must run before ANY
# XLA-backend touch, so it happens at import time when this module is the
# entry point of a multi-task launch.  Same order as distrib_la's CLI mode.
# ONE GPU PER PROCESS (``local_device_ids=[0]``) when CUDA_VISIBLE_DEVICES
# names exactly one device, which is what `lx run -G 4 -n 4` arranges: the
# phdf5 collective read is one MPI rank per process and JAX's
# one-process-per-GPU model is what the handler was built against.
if __name__ == "__main__":
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        import jax as _jax_boot
        _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if _cvd and "," not in _cvd:
            _jax_boot.distributed.initialize(local_device_ids=[0])
        else:
            _jax_boot.distributed.initialize()

from wfn_loader import WfnLoader                              # noqa: E402

#: The checked-in hostile deck (survey §6.4): nrk 9, mnband 82, nspinor 2,
#: ngkmax 1963, ngk 1917..1963, ntran 2.  82 % 4 = 2 and min(ngk) < ngkmax,
#: so a 2x2 leg on it exercises BOTH pad axes on one file.  Byte-size
#: identical to the ``/pscratch`` MoS2 3x3 deck the parity harness used to
#: name, whose machine is gone.
DECK_REL = "tests/regression/gnppm_debug/WFN.h5"

#: The hostile band window.  ``(0, 10)`` on a 4-rank mesh gives per-rank
#: clamped band counts [3, 3, 3, 1] summing to 10, with nb_padded = 12 —
#: i.e. the tail rank's clamp is LOAD-BEARING, which is the exact
#: condition ``--bands 0,4`` never produced.
DEFAULT_BANDS = (0, 10)


def deck_path(name: str = DECK_REL) -> str | None:
    """Absolute path of the checked-in deck, or ``None`` if absent.

    A plain function: this file is also a bare ``__main__`` under
    ``srun``, where the pytest fixture machinery does not exist.  The
    pytest cells still take the ``gnppm_wfn`` fixture (which skips with
    the named reason), and
    :func:`test_the_bare_script_and_the_fixture_name_the_same_file`
    asserts the two resolutions agree, so the duplication cannot drift.
    """
    p = os.path.join(_REPO, name)
    return p if os.path.exists(p) else None


# ---------------------------------------------------------------------------
# Small helpers — host numpy, no collectives
# ---------------------------------------------------------------------------

def _band_spec():
    from jax.sharding import PartitionSpec as P
    return P(None, ("x", "y"), None, None)


def _world(mesh) -> int:
    return int(mesh.shape["x"]) * int(mesh.shape["y"])


def _local(x):
    """This PROCESS's shard of a global array, as host numpy.

    No gather.  Bit-identity between two backends is a per-rank claim —
    each rank owns a band block and compares its own bytes — and routing
    it through ``process_allgather`` would add a collective whose own
    reduction could mask a difference, besides costing P x the array.
    """
    shards = x.addressable_shards
    assert len(shards) == 1, (
        f"this process addresses {len(shards)} shards; these checks assume "
        f"one device per process (the phdf5 backend's own model)")
    return np.asarray(shards[0].data), shards[0].index


def _sentinel(loader):
    from common.gvec_fft_box import fft_box_pad_sentinel
    return fft_box_pad_sentinel(tuple(int(s) for s in loader.fft_grid))


class _raises:
    """``pytest.raises`` that also works in the pytest-free CLI mode."""

    def __init__(self, exc, match=""):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(
                f"expected {self.exc} matching {self.match!r}, nothing raised")
        if not issubclass(et, self.exc if isinstance(self.exc, tuple)
                          else (self.exc,)):
            return False
        if self.match and self.match not in str(ev):
            raise AssertionError(
                f"{et.__name__} raised but {self.match!r} not in {str(ev)!r}")
        return True


# ---------------------------------------------------------------------------
# The counts table, reimplemented HERE — the negative control's instrument
# ---------------------------------------------------------------------------

def band_counts_table(*, nb_padded: int, world: int, b_lo: int,
                      band_bound: int) -> list:
    """Per-rank REAL band count for a band-sharded collective read.

    A TEST-LOCAL twin of the band dimension of
    ``file_io._slab_io_ffi._derive_window_counts`` — deliberately a second
    implementation, because its whole purpose is to be perturbed.  Rank
    ``r`` owns ``[r*bpr, (r+1)*bpr)`` of the padded band axis and reads
    ``min(bpr, band_bound - b_lo - r*bpr)`` real rows, floored at 0; the
    rest of its block is pad and must come back exactly zero.

    ``band_bound`` is THE PARAMETER THE 22049c3 DEFECT GOT WRONG.  The
    correct value is the REQUEST's ``b_hi``; the defect used the FILE's
    ``mnband``.  On any window whose length divides the world the two
    agree exactly, which is why a suite defaulting to ``--bands 0,4`` on
    four ranks reported green for months.

    :func:`check_band_bound_negative_control` calls this twice — once with
    each bound — and :func:`test_the_twin_agrees_with_the_door` pins it
    against the real door function so the twin cannot quietly drift into
    agreeing with itself.
    """
    bpr = int(nb_padded) // int(world)
    return [max(0, min(bpr, int(band_bound) - int(b_lo) - r * bpr))
            for r in range(int(world))]


# ---------------------------------------------------------------------------
# Check bodies — shared by the pytest cells, the CLI matrix and the bench
# harness.  Each raises AssertionError with the measured numbers in it.
# ---------------------------------------------------------------------------

def check_band_pad_clamp_parity(mesh, deck, *, bands=DEFAULT_BANDS,
                                k="ibz", backends=("eager", "phdf5")):
    """THE band-pad clamp class: eager vs phdf5, BIT-IDENTICAL, per rank.

    ``atol`` has no seat at this table.  Both backends read the same f64
    bytes off the same file and assemble them with the same unfold; a
    difference of one ulp is a difference in WHICH BYTES, not in rounding,
    and a tolerance would hide exactly the clamp defect this check exists
    for.  ``np.array_equal`` on this process's own shard.

    Anti-tautology, asserted BEFORE anything is loaded:
      * ``(b_hi - b_lo) % world != 0`` — a divisible window makes the
        tail rank's clamp inert and the whole check vacuous.  This is the
        assertion whose absence let ``--bands 0,4`` pass for months.
      * ``min(ngk) < ngkmax`` — a rectangular deck has no G pad.

    Then, on this rank's shard:
      * shapes agree and the band axis is ``round_up(nb, world)``;
      * PAD ROWS EXIST and are exactly zero in BOTH backends;
      * the rank's REAL band rows are nonzero, so "pad is zero" is
        discriminating rather than a claim about a dead read;
      * the observed real-row count matches ``band_counts_table`` for
        this rank, and the table sums to ``nb_logical``.
    """
    from runtime.padding import round_up
    b_lo, b_hi = int(bands[0]), int(bands[1])
    nb_logical = b_hi - b_lo
    world = _world(mesh)
    spec = _band_spec()

    assert world >= 2, (
        f"check_band_pad_clamp_parity on a {world}-device mesh: every "
        f"band window divides a world of 1, so the clamp is inert and "
        f"this check would assert nothing.  It is a >= 2x2 check.")
    assert nb_logical % world != 0, (
        f"ANTI-TAUTOLOGY: bands={bands} gives nb_logical={nb_logical}, "
        f"which DIVIDES world={world}.  Every rank then reads a full "
        f"block, the tail clamp never fires, and this check cannot see "
        f"the 22049c3 class at all.  Pick a non-dividing window "
        f"(the default is {DEFAULT_BANDS}).")

    counts = band_counts_table(nb_padded=int(round_up(nb_logical, world)),
                               world=world, b_lo=b_lo, band_bound=b_hi)
    assert sum(counts) == nb_logical, (
        f"the per-rank band counts {counts} sum to {sum(counts)}, not "
        f"nb_logical={nb_logical}: the split loses or duplicates bands")

    out = {"counts": counts, "world": world, "bands": (b_lo, b_hi), "k": k}
    loaders, shards = {}, {}
    try:
        for name in backends:
            loaders[name] = WfnLoader(deck, mesh=mesh, backend=name)
        first = loaders[backends[0]]
        ngk = np.asarray(first.ngk)
        assert int(ngk.min()) < int(first.ngkmax), (
            f"ANTI-TAUTOLOGY: this deck is rectangular (ngk == ngkmax == "
            f"{int(first.ngkmax)}), so there is no G pad to check")
        assert b_hi <= int(first.nbands)
        out["ngk"] = (int(ngk.min()), int(first.ngkmax))

        for name, loader in loaders.items():
            psi = loader.load(bands=bands, k=k, sharding=spec)
            shards[name] = _local(psi)
            assert psi.shape[1] == int(round_up(nb_logical, world)), (
                f"{name}: band axis {psi.shape[1]}, want "
                f"{int(round_up(nb_logical, world))}")

        ref_name = backends[0]
        ref, ref_idx = shards[ref_name]
        out["shard_shape"] = tuple(int(s) for s in ref.shape)

        # This rank's real-vs-pad row split, measured off the DATA.
        import jax
        r = int(jax.process_index()) if jax.process_count() > 1 else 0
        want_real = counts[r] if r < len(counts) else 0
        nonzero = [b for b in range(ref.shape[1]) if ref[:, b].any()]
        assert nonzero == list(range(want_real)), (
            f"rank {r}: real band rows are {nonzero}, but the counts table "
            f"says this rank owns {want_real} real rows ({counts}).  Either "
            f"the clamp is wrong or a real band read as zero.")
        assert ref.shape[1] > want_real, (
            f"rank {r}: block of {ref.shape[1]} rows is entirely real, so "
            f"the pad-zero assertion below is vacuous on this rank")
        out["rank_real_rows"] = want_real

        for name, (arr, idx) in shards.items():
            assert arr.shape == ref.shape, f"{name}: {arr.shape} vs {ref.shape}"
            assert idx == ref_idx, (
                f"{name} owns global slice {idx}, {ref_name} owns {ref_idx}; "
                f"the two backends did not shard the band axis the same way "
                f"and the comparison below would be between different bands")
            pad = arr[:, want_real:]
            assert not pad.any(), (
                f"{name} rank {r}: {int(np.count_nonzero(pad))} nonzero pad "
                f"entries past real row {want_real} (max |x| "
                f"{np.abs(pad).max():.3e})")

        for name, (arr, _idx) in shards.items():
            if name == ref_name:
                continue
            same = bool(np.array_equal(arr, ref))
            if not same:
                d = np.abs(arr - ref)
                bad = np.unravel_index(int(np.argmax(d)), d.shape)
                raise AssertionError(
                    f"{ref_name} vs {name} are NOT bit-identical on rank "
                    f"{r}'s shard (bands={bands} k={k!r} world={world} "
                    f"counts={counts}): {int(np.count_nonzero(d))} of "
                    f"{d.size} entries differ, max|Δ| {float(d.max()):.6e} "
                    f"at {bad} ({ref_name}={ref[bad]!r} {name}={arr[bad]!r})")
            out[f"{ref_name}_vs_{name}"] = "bit-identical"
    finally:
        for loader in loaders.values():
            loader.close()
    return out


def check_band_bound_negative_control(mesh, deck, *, bands=DEFAULT_BANDS):
    """THE FALSIFICATION CELL.  22049c3's perturbation, reintroduced.

    The defect used the FILE's ``mnband`` where the per-rank band clamp
    needed the REQUEST's ``b_hi``.  This check does not need an ``.so`` or
    four ranks to state the claim, because the claim is about the TABLE:

      1. at THIS geometry the perturbed table DIFFERS from the correct
         one, and it differs on the TAIL rank specifically — which is
         what makes a non-dividing window the geometry that exposes it;
      2. at a DIVISIBLE control geometry the two tables MATCH exactly —
         which is why the defect survived every green suite that ran
         ``--bands 0,4`` on four ranks;
      3. the perturbed table's rows sum to MORE than ``nb_logical``, so
         the failure mode is a tail rank reading FILE DATA into slots the
         request said were pad.  Not a crash: a wrong number.

    If any of the three stops holding, this check goes red — which is
    what "the negative control must be able to fail" means.  The twin it
    perturbs is pinned against the real door in
    :func:`test_the_twin_agrees_with_the_door`.
    """
    from runtime.padding import round_up
    b_lo, b_hi = int(bands[0]), int(bands[1])
    nb_logical = b_hi - b_lo
    world = _world(mesh)
    assert world >= 2, (
        f"the negative control needs a tail rank; world={world} has none")
    assert nb_logical % world != 0, (
        f"ANTI-TAUTOLOGY: bands={bands} divides world={world}, so this IS "
        f"the control geometry and claim 1 cannot hold")

    with WfnLoader(deck) as loader:
        mnband = int(loader.nbands)
    assert mnband > b_hi, (
        f"this deck's mnband={mnband} is not past the requested b_hi="
        f"{b_hi}, so substituting one for the other changes nothing and "
        f"the perturbation is not a perturbation")

    nb_padded = int(round_up(nb_logical, world))
    good = band_counts_table(nb_padded=nb_padded, world=world, b_lo=b_lo,
                             band_bound=b_hi)
    bad = band_counts_table(nb_padded=nb_padded, world=world, b_lo=b_lo,
                            band_bound=mnband)

    # 1. differs, and on the TAIL rank
    assert good != bad, (
        f"the 22049c3 perturbation (band_bound {b_hi} -> mnband {mnband}) "
        f"produced the SAME table {good} at bands={bands} world={world}.  "
        f"This geometry does not expose the defect and the check above is "
        f"measuring nothing.")
    diff = [r for r in range(world) if good[r] != bad[r]]
    assert diff and diff[-1] == world - 1, (
        f"the perturbation differs on ranks {diff}, not on the TAIL rank "
        f"{world - 1}: good={good} bad={bad}")

    # 2. the divisible control geometry MATCHES
    ctrl_nb = nb_padded                      # divisible by construction
    ctrl_good = band_counts_table(nb_padded=ctrl_nb, world=world, b_lo=b_lo,
                                  band_bound=b_lo + ctrl_nb)
    ctrl_bad = band_counts_table(nb_padded=ctrl_nb, world=world, b_lo=b_lo,
                                 band_bound=mnband)
    assert ctrl_good == ctrl_bad, (
        f"the CONTROL geometry (bands=({b_lo},{b_lo + ctrl_nb}), divisible "
        f"by world={world}) also differs: good={ctrl_good} bad={ctrl_bad}.  "
        f"Then the defect was never geometry-dependent and the story this "
        f"check tells about why it survived is wrong.")

    # 3. the failure mode is EXTRA rows, not missing ones
    assert sum(bad) > sum(good) == nb_logical, (
        f"perturbed rows sum to {sum(bad)}, correct to {sum(good)}, "
        f"nb_logical={nb_logical}")
    return {"good": good, "perturbed": bad, "tail_rank": world - 1,
            "control_good": ctrl_good, "control_perturbed": ctrl_bad,
            "mnband": mnband}


def check_sentinel_mask_conjunction(mesh, deck, *, bands=DEFAULT_BANDS,
                                    backends=("eager", "phdf5"),
                                    k_modes=("ibz", "full_bz")):
    """THE contract's named cell (survey §7.4), on a sharded load.

    For EVERY ``(kk, j)`` with ``j >= ngk_valid[kk]``:

        ψ[kk, :, :, j] == 0   AND   gvecs[kk, j] == fft_box_pad_sentinel

    Both halves, as a CONJUNCTION.  The zero makes the slot inert in any
    contraction; the sentinel makes an UNMASKED slot detectable rather
    than silently aliased onto Γ, which is a physical component of every
    G-sphere.  Asserting one half only is the hole that let a consumer
    add ``ngkmax − ngk`` extra copies of ψ(Γ) with no symptom at all.

    Non-vacuously: the pad-slot count is asserted > 0 first.  Both
    backends and both k modes, because the full-BZ path rebuilds the G
    table through the symmetry unfold and could lose the sentinel there
    and nowhere else, and the collective read is a different producer of
    the ψ half from the host read.
    """
    spec = _band_spec()
    out = {}
    for name in backends:
        with WfnLoader(deck, mesh=mesh, backend=name) as loader:
            sentinel, flat = _sentinel(loader)
            grid = tuple(int(s) for s in loader.fft_grid)
            assert 0 <= int(flat) < grid[0] * grid[1] * grid[2]
            for k in k_modes:
                psi, _idx = _local(
                    loader.load(bands=bands, k=k, sharding=spec))
                gvecs = loader.gvecs(k=k)
                nv = np.asarray(loader.ngk_valid(k=k))
                n_k, ngkmax = psi.shape[0], psi.shape[3]
                assert gvecs.shape == (n_k, ngkmax, 3), (
                    f"{name}/{k}: gvecs {gvecs.shape} does not match ψ "
                    f"{psi.shape} — the two halves describe different "
                    f"k-sets and the conjunction is meaningless")
                pad_slots = int(n_k * ngkmax - int(nv.sum()))
                assert pad_slots > 0, (
                    f"{name}/{k}: no pad slots (ngkmax={ngkmax}, ngk="
                    f"{sorted(set(map(int, nv)))}), so the conjunction "
                    f"below is vacuous at this geometry")
                bad_psi, bad_g = [], []
                for kk in range(n_k):
                    n = int(nv[kk])
                    if psi[kk, :, :, n:].any():
                        bad_psi.append((kk, n,
                                        int(np.count_nonzero(
                                            psi[kk, :, :, n:]))))
                    tail = gvecs[kk, n:]
                    if not np.array_equal(
                            tail, np.broadcast_to(sentinel, tail.shape)):
                        bad_g.append((kk, n, tail[
                            int(np.argmax((tail != sentinel).any(axis=1)))]))
                assert not bad_psi, (
                    f"{name}/{k}: ψ is NONZERO past ngk_valid at "
                    f"(kk, ngk, n_nonzero) = {bad_psi[:6]}")
                assert not bad_g, (
                    f"{name}/{k}: gvecs pad rows are not the sentinel "
                    f"{tuple(int(v) for v in sentinel)} at (kk, ngk, row) = "
                    f"{bad_g[:6]}")
                out[f"{name}/{k}"] = dict(
                    pad_slots=pad_slots, shard=tuple(int(s)
                                                     for s in psi.shape))
    return out


def check_load_process_local_per_rank_windows(mesh, deck, *, nb=3):
    """Per-rank DIFFERENT windows, against a serial h5py twin.

    ``load_process_local`` is the second primitive precisely so rank *r*
    may ask for a window nobody else asked for (``gw.kin_ion_io``'s ρ
    sweep is partitioned over k).  Two claims:

      1. THE VALUES.  Rank *r*'s ``(bands=(r, r+nb), k='ibz')`` window is
         re-read here with plain ``h5py`` — a genuinely independent path:
         no jax, no sharding, no unfold — and must match BIT FOR BIT.
         ``k='ibz'`` on purpose: the raw IBZ slab is the one whose h5py
         twin is a hyperslab and a re/im pack, so the reference is a
         reference rather than a second copy of the unfold.
      2. THE SHAPE CONTRACT.  Exactly ``nb`` bands (no mesh padding), one
         addressable shard, committed to this process's own device.

    Running this at all under ``srun -n 4`` is itself the third claim:
    four ranks requesting four DIFFERENT band windows either completes or
    hangs, and nothing in ``load_process_local`` may enter a collective.
    There is no barrier in this function for that reason.
    """
    import h5py
    import jax
    r = int(jax.process_index())
    b_lo, b_hi = r, r + int(nb)
    out = {"rank": r, "window": (b_lo, b_hi)}

    with WfnLoader(deck, mesh=mesh, backend="eager") as loader:
        assert b_hi <= int(loader.nbands)
        psi = loader.load_process_local(bands=(b_lo, b_hi), k="ibz")
        assert psi.shape[1] == int(nb), (
            f"rank {r}: load_process_local padded the band axis to "
            f"{psi.shape[1]}, want exactly {nb}")
        assert len(psi.addressable_shards) == 1
        assert psi.sharding.num_devices == 1
        arr = np.asarray(psi)
        ngk = np.asarray(loader.ngk)
        starts = np.asarray(loader._kpt_starts)
        ngkmax = int(loader.ngkmax)
        nrk = int(loader.nkpts)

    # The SERIAL TWIN: plain h5py, no jax anywhere.
    with h5py.File(deck, "r") as f:
        ds = f["wfns/coeffs"]
        ref = np.zeros(arr.shape, dtype=np.complex128)
        for ik in range(nrk):
            s = int(starts[ik])
            n = int(ngk[ik])
            raw = ds[b_lo:b_hi, :, s:s + n, :]
            ref[ik, :, :, :n] = raw[..., 0] + 1j * raw[..., 1]
    assert ref.shape == (nrk, int(nb), arr.shape[2], ngkmax)
    assert np.array_equal(arr, ref), (
        f"rank {r}: load_process_local(bands=({b_lo},{b_hi}), k='ibz') "
        f"differs from the serial h5py twin; "
        f"{int(np.count_nonzero(arr != ref))} of {arr.size} entries, "
        f"max|Δ| {float(np.abs(arr - ref).max()):.6e}")
    # ...and the twin is not trivially zero, which would make the
    # comparison pass on a loader that returned nothing.
    assert ref.any() and np.count_nonzero(ref) > ref.size // 100
    out["nonzero"] = int(np.count_nonzero(ref))
    return out


# ---------------------------------------------------------------------------
# pytest entry points.  Single process, so the mesh is 1x1; the 2x2
# answers come from the CLI matrix below.
# ---------------------------------------------------------------------------

def _mesh_1x1(platform=None):
    import jax
    from jax.sharding import Mesh
    devs = jax.devices() if platform is None else jax.devices(platform)
    return Mesh(np.asarray(devs[:1]).reshape(1, 1), ("x", "y"))


def _needs_real_mesh(world_min=2):
    """SKIP unless this process is one of >= ``world_min`` REAL ranks.

    Device count is not process count: an emulated 2x2 cannot drive the
    collective read (``_auto_pick_backend`` returns ``eager`` below P=2
    before it probes for anything).  The skip names the leg that runs the
    cell, because a skip nobody can point at a covering run for is lost
    coverage.
    """
    import jax
    from jax.sharding import Mesh
    n = int(jax.process_count())
    if n < world_min:
        pytest.skip(
            f"real multi-process leg: this is a single-process pytest "
            f"(process_count={n}), and the collective read needs >= "
            f"{world_min} PROCESSES — device count does not substitute.  "
            f"Covered by leg L-c: `lx run --cpu -N 1 -n 4 python3 "
            f"services/wfn_loader/tests/test_wfn_loader_multiproc.py "
            f"--mesh 2x2`")
    devs = jax.devices()
    return Mesh(np.asarray(devs[:n]).reshape(2, n // 2), ("x", "y"))


def _needs_phdf5():
    """SKIP unless a library on some platform can serve the collective read.

    ABSENT is an honest skip; the probe's own three-way reason (unknown
    target / library would not load / loaded but does not export the
    symbol) is quoted into it, because those three have three different
    fixes.  BUILT-AND-BROKEN is the skip-honesty gate's business, not
    this helper's.
    """
    from file_io.slab_io import probe_read_availability
    reasons = []
    for plat in ("CUDA", "cpu"):
        ok, why = probe_read_availability(plat)
        if ok:
            return plat
        reasons.append(f"{plat}: {why}")
    pytest.skip(
        "not built on this machine — no FFI library serves the collective "
        "WFN read here (" + "; ".join(reasons)[:400] + ").  Covered by leg "
        "L-c with the BUILD_NOTES .so pins.")


def test_the_bare_script_and_the_fixture_name_the_same_file(gnppm_wfn):
    """The CLI's ``deck_path`` and the suite's ``gnppm_wfn`` fixture agree.

    This file resolves the deck itself because it runs as a bare script
    under ``srun`` where no fixture exists; the conftest resolves it for
    the pytest cells.  Two resolutions of one path is exactly the kind of
    duplication that drifts silently — the cluster leg would read a
    different (or absent) file and report ``0 cells ran`` — so it is
    asserted rather than hoped.
    """
    assert deck_path() is not None
    assert os.path.realpath(deck_path()) == os.path.realpath(gnppm_wfn)


def test_the_twin_agrees_with_the_door(gnppm_wfn):
    """``band_counts_table`` vs the REAL clip, on the loader's own request.

    The negative control perturbs a test-local twin.  A twin that had
    drifted from the door would let the control pass while describing a
    table nothing computes — so it is pinned here against
    ``file_io._slab_io_ffi._derive_window_counts``, driven with exactly
    the ``(shape, offsets, valid_shapes)`` ``_phdf5_build`` would hand it
    for the hostile window at world=4.

    Needs no ``.so``: the counts derivation is pure arithmetic in Python,
    which is the point of it living next to ``_derive_valid_shape``
    instead of in the handler.
    """
    from file_io._slab_io_ffi import _derive_window_counts
    from runtime.padding import round_up
    b_lo, b_hi = DEFAULT_BANDS
    world, nb_logical = 4, b_hi - b_lo
    nb_padded = int(round_up(nb_logical, world))
    with WfnLoader(gnppm_wfn) as loader:
        ns, ngkmax = int(loader.nspinor), int(loader.ngkmax)
        ngk = np.asarray(loader.ngk)
        starts = np.asarray(loader._kpt_starts)
        band_extent = min(b_hi, int(loader.nbands))

    bpr = nb_padded // world
    per_rank_shape = (bpr, ns, ngkmax, 2)
    rank_offsets = np.array([[r * bpr, 0, 0, 0] for r in range(world)],
                            dtype=np.int64)
    valid_shapes = np.stack(
        [[band_extent - b_lo, ns, int(ngk[ib]), 2]
         for ib in range(len(ngk))], axis=0).astype(np.int64)
    assert starts.shape[0] == len(ngk)

    counts = _derive_window_counts(per_rank_shape=per_rank_shape,
                                   rank_offsets=rank_offsets,
                                   valid_shapes=valid_shapes)
    n_win = valid_shapes.shape[0]
    door_band = [int(counts.reshape(world, n_win, 4)[r, 0, 0])
                 for r in range(world)]
    twin = band_counts_table(nb_padded=nb_padded, world=world, b_lo=b_lo,
                             band_bound=b_hi)
    assert door_band == twin == [3, 3, 3, 1], (
        f"door={door_band} twin={twin}; the step-0 measurement recorded "
        f"[3, 3, 3, 1] for bands={DEFAULT_BANDS} at world=4")
    # ...and the window axis really is the one that varies, so reading
    # window 0's band count above is reading the right cell.
    assert {int(counts.reshape(world, n_win, 4)[0, w, 2])
            for w in range(n_win)} == {int(v) for v in ngk}


def test_the_negative_control_runs_without_a_cluster(gnppm_wfn):
    """``check_band_bound_negative_control`` at world=4, in one process.

    The control is about the TABLE, so it needs neither an ``.so`` nor
    four ranks — only a mesh shape to compute against.  Running it here
    means the falsification arm is exercised on every laptop, and the
    cluster leg re-runs it beside the parity check it protects.
    """
    class _FakeMesh:
        shape = {"x": 2, "y": 2}
    got = check_band_bound_negative_control(_FakeMesh(), gnppm_wfn)
    assert got["good"] == [3, 3, 3, 1]
    assert got["perturbed"] == [3, 3, 3, 3]
    assert got["tail_rank"] == 3
    assert got["control_good"] == got["control_perturbed"] == [3, 3, 3, 3]


def test_the_negative_control_can_fail(gnppm_wfn):
    """RED TWIN for the control: a DIVISIBLE window must make it refuse.

    ``bands=(0, 12)`` divides a 4-rank world, so the perturbation is
    invisible and the control has nothing to detect — it must say so
    rather than report a pass.  Without this cell a control that had
    silently become a tautology would look identical to a working one.
    """
    class _FakeMesh:
        shape = {"x": 2, "y": 2}
    with pytest.raises(AssertionError, match="ANTI-TAUTOLOGY"):
        check_band_bound_negative_control(_FakeMesh(), gnppm_wfn,
                                          bands=(0, 12))


def test_the_parity_check_refuses_a_divisible_window(gnppm_wfn):
    """RED TWIN for the parity check's own anti-tautology guard.

    Needs no library: the guard fires before a loader is opened, which
    is deliberate — a check that discovered its geometry was vacuous
    AFTER a 15 GB collective read would have wasted the leg it was
    supposed to protect.
    """
    class _FakeMesh:
        shape = {"x": 2, "y": 2}
    with pytest.raises(AssertionError, match="ANTI-TAUTOLOGY"):
        check_band_pad_clamp_parity(_FakeMesh(), gnppm_wfn, bands=(0, 12))
    with pytest.raises(AssertionError, match="1-device mesh"):
        check_band_pad_clamp_parity(_mesh_1x1(), gnppm_wfn)


def test_sentinel_mask_conjunction_eager_arm_at_1x1(gnppm_wfn):
    """The conjunction's EAGER arm, single process.

    The phdf5 arm and the sharded band axis are leg L-c; what runs here
    is the half that needs no library, so a break in the sentinel/zero
    pairing turns red on a laptop instead of only on four ranks.
    """
    got = check_sentinel_mask_conjunction(
        _mesh_1x1(), gnppm_wfn, backends=("eager",))
    assert set(got) == {"eager/ibz", "eager/full_bz"}
    assert all(v["pad_slots"] > 0 for v in got.values())


def test_load_process_local_per_rank_windows_at_1x1(gnppm_wfn):
    """The h5py twin arm, single process (rank 0's window).

    The per-rank-DIFFERENT-window claim needs four ranks and is leg L-c;
    the VALUE claim — that ψ is what a plain h5py hyperslab says it is —
    does not, and it is the one that would go wrong quietly.
    """
    got = check_load_process_local_per_rank_windows(_mesh_1x1(), gnppm_wfn)
    assert got["rank"] == 0 and got["window"] == (0, 3)
    assert got["nonzero"] > 0


def test_band_pad_clamp_parity_needs_the_real_leg(gnppm_wfn):
    """The parity check itself, on whatever this process can build.

    Skips on a single-process pytest (naming leg L-c) and again if no
    library can serve the collective read.  On a machine that HAS both,
    this runs the same body the cluster leg runs.
    """
    mesh = _needs_real_mesh()
    _needs_phdf5()
    check_band_pad_clamp_parity(mesh, gnppm_wfn)


def test_the_cli_cells_are_all_reachable():
    """Every ``_CLI_CELLS`` row names a function that exists, and every
    check body is in the table.

    Cheap, and it is the failure the CLI mode cannot report: a typo'd or
    dropped row makes the multi-rank leg quietly run a smaller matrix and
    print ``done: 0 failures``.
    """
    names = {name for name, _, _ in _CLI_CELLS}
    assert len(names) == len(_CLI_CELLS), "duplicate _CLI_CELLS name"
    called = set()
    for _name, _plat, fn in _CLI_CELLS:
        # Read the GLOBALS each row's lambda actually references, not its
        # label: matching on the label is the version of this check that
        # silently rots.
        mine = {g for g in fn.__code__.co_names if g.startswith("check_")}
        assert mine, f"{_name}: calls no check body"
        called |= mine
    bodies = {k for k in globals() if k.startswith("check_")}
    missing = bodies - called
    assert not missing, (
        f"check bodies with no _CLI_CELLS row (the 2x2 leg would never run "
        f"them): {sorted(missing)}")
    unknown = called - bodies
    assert not unknown, f"_CLI_CELLS names bodies that do not exist: {unknown}"


# ---------------------------------------------------------------------------
# CLI mode — the real multi-rank matrix.
# ---------------------------------------------------------------------------

_CLI_CELLS = [
    # (name, platform: 'cpu' | 'CUDA' | '', fn(mesh, deck, bands))
    ("band_pad_clamp_parity_ibz", "",
     lambda mesh, deck, bands: check_band_pad_clamp_parity(
         mesh, deck, bands=bands, k="ibz")),
    ("band_pad_clamp_parity_full_bz", "",
     lambda mesh, deck, bands: check_band_pad_clamp_parity(
         mesh, deck, bands=bands, k="full_bz")),
    ("band_bound_negative_control", "",
     lambda mesh, deck, bands: check_band_bound_negative_control(
         mesh, deck, bands=bands)),
    ("sentinel_mask_conjunction", "",
     lambda mesh, deck, bands: check_sentinel_mask_conjunction(
         mesh, deck, bands=bands)),
    ("load_process_local_per_rank_windows", "",
     lambda mesh, deck, bands: check_load_process_local_per_rank_windows(
         mesh, deck)),
]


def _mesh_from_arg(spec):
    import jax
    from jax.sharding import Mesh
    px, py = (int(v) for v in spec.lower().replace("×", "x").split("x"))
    return Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))


def _cli_main(argv=None):
    import jax

    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--mesh", required=True, help="PxQ process mesh")
    ap.add_argument("--wfn", default="", help="deck (default: the in-repo "
                                              "gnppm hostile fixture)")
    ap.add_argument("--bands", default=",".join(map(str, DEFAULT_BANDS)),
                    help="b_lo,b_hi (default NON-DIVISIBLE, see --help)")
    ap.add_argument("--only", default="", help="substring filter")
    args = ap.parse_args(argv)

    deck = args.wfn or deck_path()
    if not deck or not os.path.exists(deck):
        print(f"deck not found: {deck!r}", flush=True)
        return 1
    bands = tuple(int(v) for v in args.bands.split(","))
    mesh = _mesh_from_arg(args.mesh)
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    p0(f"backend={jax.default_backend()} mesh={args.mesh} "
       f"processes={jax.process_count()} devices={jax.device_count()} "
       f"deck={deck} bands={bands}", flush=True)
    try:
        from file_io.slab_io import probe_read_availability
        for plat in ("CUDA", "cpu"):
            p0(f"  probe_read_availability({plat}) = "
               f"{probe_read_availability(plat)}", flush=True)
    except Exception as exc:                                   # noqa: BLE001
        p0(f"  probe_read_availability unavailable: {exc}", flush=True)

    is_cpu = jax.default_backend() == "cpu"
    failures, ran = 0, 0
    for name, platform, fn in _CLI_CELLS:
        if args.only and args.only not in name:
            continue
        if platform == "cpu" and not is_cpu:
            p0(f"SKIP {name}[{args.mesh}] (host-only)", flush=True)
            continue
        if platform == "CUDA" and is_cpu:
            p0(f"SKIP {name}[{args.mesh}] (CUDA-only)", flush=True)
            continue
        tag = f"{name}[{args.mesh},bands={bands[0]},{bands[1]}]"
        try:
            out = fn(mesh, deck, bands)
            ran += 1
            p0(f"PASS {tag} {out if out is not True else ''}", flush=True)
        except AssertionError as exc:
            failures += 1
            p0(f"FAIL {tag}: {exc}", flush=True)
        except Exception as exc:                               # noqa: BLE001
            failures += 1
            p0(f"ERROR {tag}: {type(exc).__name__}: "
               f"{' '.join(str(exc).split())[:600]}", flush=True)
    # RAN, not just failures.  "0 failures" out of 0 cells is the shape of
    # every artifact-free green in this tree's history.
    p0(f"done: {ran} cells ran, {failures} failures", flush=True)
    return 1 if (failures or ran == 0) else 0


if __name__ == "__main__":
    sys.exit(_cli_main())
