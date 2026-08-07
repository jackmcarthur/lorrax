"""Layer L-b: four emulated devices, a real 2x2 mesh, real padded ψ.

L-a asserts the loader's refusals and its single-device values; this tier
is the first one where the PADDING CONTRACT has anything to pad.  A 1x1
mesh has band divisor 1, so every band-pad assertion below is vacuous
there — ``nb_padded == nb_logical`` and the "pad rows are zero" claim is a
claim about an empty slice.  Four host devices in ONE process, an
``('x','y')`` mesh over them, and the eager backend running for real is
what makes it non-vacuous, and every cell here says so with an explicit
anti-tautology assertion rather than trusting the arithmetic.

WHY ONLY THE EAGER BACKEND LIVES HERE.  ``_auto_pick_backend`` returns
``eager`` whenever ``jax.process_count() <= 1``, whatever the DEVICE count
— the phdf5 backend carries a per-PROCESS MPI-IO context, so one process
pretending to be four devices cannot drive it.  That is the contract, not
a limitation of this file, and
:func:`test_four_devices_in_one_process_is_still_the_eager_backend`
observes it rather than asserting it in a comment.  The phdf5 backend on a
2x2 is layer L-c (``test_wfn_loader_multiproc.py``, four real ranks).

WHY THE CELLS SKIP RATHER THAN ASSERT below four devices: device count is
a property of how the leg was LAUNCHED, not of the code under test.
``tests/KNOWN_FAILURES.md`` lists eleven cells that failed a 1-device leg
purely for writing ``assert n_dev >= 4``.  ``conftest.py`` sets
``XLA_FLAGS`` when it can and explains exactly when it cannot (the full
monorepo run imports jax before this suite's conftest loads, and the flag
is read once); the skip reason names the leg that does run these.

THE DECK.  ``tests/regression/gnppm_debug/WFN.h5`` — nrk 9, mnband 82,
ngkmax 1963, ngk 1917..1963.  82 % 4 = 2 and min(ngk) < ngkmax, so BOTH
pad axes are live on the same file.  Cells skip with a named reason when
the monorepo fixture tree is absent (a standalone install has none), and
``test_wfn_loader_skip_honesty.py`` turns that skip into a FAILURE on a
machine whose profile promises the fixtures.
"""
from __future__ import annotations

import numpy as np
import pytest
from lxkit.testing import hostile_extents, require_devices

from wfn_loader import WfnLoader


def _mesh(px, py):
    """A ``px x py`` mesh of HOST devices.

    ``"cpu"`` explicitly, not ``jax.devices()``.  The emulation knob is
    ``--xla_force_host_platform_device_count``, which creates HOST
    devices; on any box with a GPU the default backend is ``cuda`` and
    ``jax.device_count()`` answers 1, so asking the default backend makes
    these cells skip on the machine where the flag worked (measured, WSL,
    one CudaDevice).  Asking for CPU devices also keeps this tier off the
    SHARED GPUs on Perlmutter, which is where it belongs: nothing here
    needs one.
    """
    import jax
    from jax.sharding import Mesh
    require_devices(px * py, "cpu")
    return Mesh(np.asarray(jax.devices("cpu")[:px * py]).reshape(px, py),
                ("x", "y"))


def _band_spec():
    from jax.sharding import PartitionSpec as P
    return P(None, ("x", "y"), None, None)


def _divisor(mesh, spec):
    """The band-axis pad factor the loader will apply for ``spec``.

    Read from ``runtime.padding.spec_divisor`` — the SAME call
    ``_default_sharding`` makes, and the same one ``common.mtxel_sweep``
    makes so a sweep can consume this ψ without re-padding.  Recomputing
    it here as ``px*py`` would be a second copy of exactly the arithmetic
    whose drift this suite exists to catch.
    """
    from runtime.padding import spec_divisor
    return int(spec_divisor(mesh, spec, 1))


#: Band-window lengths that do NOT divide a 2x2 mesh's band axis.
#:
#: The LOGICAL extents come from ``lxkit.testing.hostile_extents`` (the
#: five families measured on Perlmutter job 56389339, generalized off the
#: mesh shape).  Both components of each family are candidates, because a
#: band window is one axis and lxkit hands back two.
#:
#: FILTERED against 4, not against 2: lxkit's families are hostile to a
#: PER-AXIS extent, and the band axis here is sharded by BOTH mesh axes,
#: so its divisor is ``2*2``.  ``hostile_extents((2,2))`` offers 8, which
#: divides 4 and would make the cell's own anti-tautology assertion fire.
#: The filter is here and the assertion is in the cell, so a divisor that
#: ever stops being 4 turns red with a reason instead of quietly padding
#: nothing.  At (2,2) this is (1, 7, 11) -> padded (4, 8, 12).
_HOSTILE_NB = tuple(sorted(
    n for n in ({int(c.logical[0]) for c in hostile_extents((2, 2))} |
                {int(c.logical[1]) for c in hostile_extents((2, 2))})
    if n % 4))


# ---------------------------------------------------------------------------
# The band axis: a non-dividing window, padded up, pad rows exactly zero
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nb_logical", _HOSTILE_NB)
def test_a_nondividing_band_window_pads_up_and_the_pad_rows_are_zero(
        gnppm_wfn, nb_logical):
    """The band half of the padding contract, with its own falsifier.

    Five claims, and they fail differently:

      0. ANTI-TAUTOLOGY, first, so a vacuous geometry cannot reach the
         rest: the band divisor is > 1, this window does not divide it,
         and the padded length is therefore strictly greater.  Without
         this the whole cell passes on a 1x1 mesh while measuring nothing
         — which is what the 22049c3 class looked like from the outside.
      1. the returned shape is ``round_up(nb_logical, divisor)``;
      2. the LOGICAL block is bit-identical to the mesh-less load of the
         same window — padding must not perturb the answer;
      3. the pad rows are EXACT zeros (not small, not NaN);
      4. the array really is sharded over four devices on the band axis,
         so claims 1-3 are about a distributed array rather than a
         replicated one that happens to have the right shape.
    """
    from runtime.padding import round_up
    mesh = _mesh(2, 2)
    spec = _band_spec()
    div = _divisor(mesh, spec)

    # 0. ANTI-TAUTOLOGY
    assert div > 1, (
        f"the band divisor for {spec} on this mesh is {div}; every "
        f"assertion below is then about an unpadded load and this cell "
        f"measures nothing")
    assert nb_logical % div != 0, (
        f"nb_logical={nb_logical} DIVIDES the band divisor {div}, so this "
        f"row is not hostile — the parametrization drifted")
    nb_padded = int(round_up(nb_logical, div))
    assert nb_padded > nb_logical

    with WfnLoader(gnppm_wfn, mesh=mesh, backend="eager") as loader:
        assert nb_logical <= int(loader.nbands)
        psi = loader.load(bands=(0, nb_logical), k="ibz", sharding=spec)
        assert psi.shape[1] == nb_padded, (
            f"nb_logical={nb_logical} div={div}: got band axis "
            f"{psi.shape[1]}, want {nb_padded}")
        # 4. it is really distributed, and on the band axis
        assert len(psi.addressable_shards) == 4
        assert psi.addressable_shards[0].data.shape[1] == nb_padded // div
        assert psi.sharding.spec == spec

        a = np.asarray(psi)
        # 3. pad rows are EXACT zeros
        pad = a[:, nb_logical:, :, :]
        assert pad.size > 0 and not pad.any(), (
            f"nb_logical={nb_logical}: {int(np.count_nonzero(pad))} of "
            f"{pad.size} pad-row entries are nonzero (max |x| "
            f"{np.abs(pad).max():.3e})")
        # 2. the logical block is the unpadded answer, bit for bit.
        #
        # The unpadded reference is an EXPLICIT fully-replicated spec, not
        # ``sharding=None``.  MEASURED writing this cell: on a loader that
        # HAS a multi-device mesh, ``sharding=None`` does not mean
        # "replicated" — ``_default_sharding`` substitutes the production
        # band spec ``P(None,('x','y'),None,None)``, so the "reference"
        # came back padded to 12 and claim 2 compared a padded array with
        # itself.  Worth writing down: every caller who reaches for
        # ``sharding=None`` expecting the serial layout gets the sharded
        # one the moment their loader acquires a mesh.
        from jax.sharding import PartitionSpec as P
        ref = np.asarray(loader.load(bands=(0, nb_logical), k="ibz",
                                     sharding=P(None, None, None, None)))
        assert ref.shape[1] == nb_logical, (
            "the replicated reference padded too, so claim 2 compares a "
            "padded array with itself")
        assert np.array_equal(a[:, :nb_logical], ref), (
            f"nb_logical={nb_logical}: the padded load's logical block "
            f"differs from the mesh-less load; max|Δ| "
            f"{np.abs(a[:, :nb_logical] - ref).max():.3e}")


def test_the_band_pad_zero_check_can_fail(gnppm_wfn):
    """RED TWIN for the cell above.

    The "pad rows are exactly zero" assertion is only worth anything if
    the REAL rows are not also zero — on a ψ that happened to be all
    zeros (a mis-plumbed read, an empty hyperslab) the check above passes
    while covering nothing.  So: the same slab, sliced at the LOGICAL
    boundary instead of the padded one, must be nonzero, and every real
    band row must be nonzero individually.  A read that returned zeros
    for one band would otherwise be invisible to this whole file.
    """
    mesh = _mesh(2, 2)
    nb_logical = 10
    with WfnLoader(gnppm_wfn, mesh=mesh, backend="eager") as loader:
        a = np.asarray(loader.load(bands=(0, nb_logical), k="ibz",
                                   sharding=_band_spec()))
    real = a[:, :nb_logical]
    assert real.any(), "the whole logical block is zero; the read is dead"
    dead = [b for b in range(nb_logical) if not real[:, b].any()]
    assert dead == [], (
        f"bands {dead} came back all-zero and would be indistinguishable "
        f"from pad rows, so the pad-zero assertion is not discriminating")
    # ...and the pad slice this cell's twin asserts on is genuinely there.
    assert a.shape[1] > nb_logical


# ---------------------------------------------------------------------------
# The G axis: zero coefficient AND sentinel Miller index, on a real shard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", ["ibz", "full_bz"])
def test_the_g_pad_is_zero_coefficient_and_sentinel_gvec_on_a_2x2(
        gnppm_wfn, k):
    """The CONJUNCTION, on a sharded array (survey §7.4).

    "Mask detectable ≠ mask optional": for every ``(kk, j)`` past
    ``ngk_valid[kk]`` the coefficient must be zero — which makes the slot
    inert in any contraction — AND the matching ``gvecs`` row must be the
    FFT-box pad sentinel, which is what makes an unmasked slot detectable
    instead of silently aliased onto Γ (zeros are the Miller index of Γ, a
    physical component of every G-sphere).

    Asserting only one half is the hole: a consumer that dropped
    ``ngk_valid`` used to add ``ngkmax − ngk`` extra copies of ψ(Γ) with
    no symptom at all.  The non-vacuity check counts the pad slots first,
    because on a rectangular deck there would be none and the loop would
    pass by iterating zero times.

    Both k modes, because the full-BZ path rebuilds the G table through
    the symmetry unfold and could lose the sentinel there and nowhere else.
    """
    from common.gvec_fft_box import fft_box_pad_sentinel
    mesh = _mesh(2, 2)
    with WfnLoader(gnppm_wfn, mesh=mesh, backend="eager") as loader:
        psi = np.asarray(loader.load(bands=(0, 10), k=k,
                                     sharding=_band_spec()))
        gvecs = loader.gvecs(k=k)
        nv = loader.ngk_valid(k=k)
        fft_grid = tuple(int(s) for s in loader.fft_grid)
        sentinel, flat = fft_box_pad_sentinel(fft_grid)

    n_k, ngkmax = psi.shape[0], psi.shape[3]
    pad_slots = int(n_k * ngkmax - int(nv.sum()))
    assert pad_slots > 0, (
        f"k={k}: ngk is rectangular at this geometry (ngkmax={ngkmax}, "
        f"ngk={sorted(set(map(int, nv)))}), so there are NO pad slots and "
        f"the conjunction below is vacuous")
    assert gvecs.shape == (n_k, ngkmax, 3)

    for kk in range(n_k):
        n = int(nv[kk])
        tail_psi = psi[kk, :, :, n:]
        assert not tail_psi.any(), (
            f"k={k} kk={kk}: {int(np.count_nonzero(tail_psi))} nonzero ψ "
            f"entries past ngk_valid={n}")
        tail_g = gvecs[kk, n:]
        assert np.array_equal(
            tail_g, np.broadcast_to(sentinel, tail_g.shape)), (
            f"k={k} kk={kk}: gvecs pad rows are not the sentinel "
            f"{tuple(int(v) for v in sentinel)}; first bad row "
            f"{tail_g[np.argmax((tail_g != sentinel).any(axis=1))]}")
    # The sentinel's FLAT slot is in bounds for this box.  That is what
    # makes a gather at a pad slot safe even before the coefficients are
    # zeroed, and it is a different claim from the Miller index matching:
    # a sentinel outside the box would satisfy every assertion above and
    # segfault the first consumer that trusted the pair.
    n_cells = fft_grid[0] * fft_grid[1] * fft_grid[2]
    assert 0 <= int(flat) < n_cells, (
        f"the pad sentinel's flat slot {int(flat)} is outside the "
        f"{fft_grid} FFT box ({n_cells} cells)")


def test_the_sentinel_conjunction_can_fail(gnppm_wfn):
    """RED TWIN.  Each half of the conjunction, broken on purpose.

    Two constructions, because the two halves fail for different reasons
    and a twin that only broke one would leave the other decorative:

      * ψ with a nonzero coefficient planted in a pad slot must trip the
        zero half;
      * ``gvecs`` with a pad row overwritten by Γ (all zeros — the exact
        aliasing the sentinel exists to make detectable) must trip the
        sentinel half.
    """
    from common.gvec_fft_box import fft_box_pad_sentinel
    mesh = _mesh(2, 2)
    with WfnLoader(gnppm_wfn, mesh=mesh, backend="eager") as loader:
        psi = np.asarray(loader.load(bands=(0, 10), k="ibz",
                                     sharding=_band_spec()))
        gvecs = np.array(loader.gvecs(k="ibz"), copy=True)
        nv = loader.ngk_valid(k="ibz")
        sentinel, _ = fft_box_pad_sentinel(
            tuple(int(s) for s in loader.fft_grid))

    kk = int(np.argmin(nv))
    n = int(nv[kk])
    assert n < psi.shape[3], "no pad slot at the raggedest k; twin is inert"

    bad_psi = psi.copy()
    bad_psi[kk, 0, 0, n] = 1e-30            # smaller than any tolerance
    assert bad_psi[kk, :, :, n:].any(), (
        "a planted nonzero in a pad slot did not register, so the ψ half "
        "of the conjunction cannot detect one")

    bad_g = gvecs.copy()
    bad_g[kk, n] = 0                        # Γ: the aliasing failure mode
    assert not np.array_equal(
        bad_g[kk, n:], np.broadcast_to(sentinel, bad_g[kk, n:].shape)), (
        "a pad row overwritten with Γ still matched the sentinel, so the "
        "gvecs half of the conjunction cannot detect the aliasing it "
        "exists to detect")


# ---------------------------------------------------------------------------
# The two primitives, side by side, where the difference is visible
# ---------------------------------------------------------------------------

def test_load_process_local_does_not_pad_where_load_does(gnppm_wfn):
    """THE anti-tautology half L-a could not supply.

    On a 1x1 mesh both primitives return ``nb == b_hi - b_lo`` and the
    "load_process_local does not pad" claim is unfalsifiable.  Here the
    same window through ``load`` comes back rounded up to the band
    divisor and through ``load_process_local`` comes back exact, on the
    SAME loader — which is the whole reason ``gw.kin_ion_io`` has a
    second primitive to call.
    """
    from runtime.padding import round_up
    mesh = _mesh(2, 2)
    spec = _band_spec()
    div = _divisor(mesh, spec)
    nb_logical = 10
    assert nb_logical % div != 0 and div > 1     # the geometry is hostile

    with WfnLoader(gnppm_wfn, mesh=mesh, backend="eager") as loader:
        glob = loader.load(bands=(0, nb_logical), k="ibz", sharding=spec)
        local = loader.load_process_local(bands=(0, nb_logical), k="ibz")
        assert glob.shape[1] == int(round_up(nb_logical, div)) == 12
        assert local.shape[1] == nb_logical == 10, (
            f"load_process_local padded to {local.shape[1]}; its contract "
            f"is that nothing about its array is global")
        assert len(local.addressable_shards) == 1
        assert len(glob.addressable_shards) == 4
        # ...and the values agree on the logical block, so the difference
        # is padding and not a second read path.
        assert np.array_equal(np.asarray(glob)[:, :nb_logical],
                              np.asarray(local))


@pytest.mark.parametrize("spec_axes,want_div", [
    ((None, ("x", "y"), None, None), 4),
    ((None, "x", None, None), 2),
    ((None, "y", None, None), 2),
    ((None, None, None, None), 1),
])
def test_the_band_divisor_follows_the_partition_spec(gnppm_wfn, spec_axes,
                                                     want_div):
    """The pad factor is derived from the SPEC, not from the mesh size.

    A loader that used ``mesh.devices.size`` would pad to 4 for every one
    of these, and the only place that shows is a spec that shards the
    band axis over ONE mesh axis — which is what a k-parallel driver
    passes.  Four rows: both axes, each axis alone, and fully replicated.
    """
    from jax.sharding import PartitionSpec as P
    from runtime.padding import round_up
    mesh = _mesh(2, 2)
    spec = P(*spec_axes)
    assert _divisor(mesh, spec) == want_div, (
        f"{spec}: band divisor {_divisor(mesh, spec)}, want {want_div}")
    nb_logical = 10
    with WfnLoader(gnppm_wfn, mesh=mesh, backend="eager") as loader:
        psi = loader.load(bands=(0, nb_logical), k="ibz", sharding=spec)
    assert psi.shape[1] == int(round_up(nb_logical, want_div))
    assert not np.asarray(psi)[:, nb_logical:].any()


def test_four_devices_in_one_process_is_still_the_eager_backend(gnppm_wfn):
    """DEVICE count is not PROCESS count, observed rather than assumed.

    This is WHY this file tests only the eager backend, so it is the one
    thing here that must be checked rather than written in the docstring.
    ``_auto_pick_backend`` returns ``eager`` at ``process_count() <= 1``
    BEFORE it ever probes for a library, so an emulated 2x2 cannot reach
    the collective read however many devices it shows — and a refactor
    that reordered those two checks would send this tier looking for an
    ``.so`` on every laptop.
    """
    import jax
    mesh = _mesh(2, 2)
    assert jax.process_count() == 1 < mesh.devices.size
    with WfnLoader(gnppm_wfn, mesh=mesh, backend="auto") as loader:
        assert loader.backend == "eager"
    # ...and asking for the collective backend by name is NOT refused at
    # construction — it is a legitimate request that needs four PROCESSES,
    # which is leg L-c.  The refusal that does fire here is the mesh one.
    with WfnLoader(gnppm_wfn, mesh=mesh, backend="phdf5") as loader:
        assert loader.backend == "phdf5"
    with pytest.raises(ValueError, match="requires a Mesh"):
        WfnLoader(gnppm_wfn, backend="phdf5")


def test_box_index_dev_is_one_buffer_per_k_and_mesh(gnppm_wfn):
    """The replicated-leak fix (agent_h §3 Finding 3), on a real mesh.

    Every ``psi_G_store._populate_from_loader`` used to ``device_put`` a
    fresh ``(nk, nx, ny, nz) int32`` replicated buffer — 0.16 GB/rank each,
    ~1.3 GB/rank after four bispinor channels.  The cache means every
    caller for the same ``(k, mesh)`` gets the SAME ``jax.Array``, and
    ``is`` is the assertion: equal contents would pass on the leak.
    A 2x2 is where the replication is real.
    """
    mesh = _mesh(2, 2)
    with WfnLoader(gnppm_wfn, mesh=mesh, backend="eager") as loader:
        a = loader.box_index_dev(k="ibz")
        b = loader.box_index_dev(k="ibz")
        assert a is b, "box_index_dev returned a second device buffer"
        c = loader.box_index_dev(k="full_bz")
        assert c is not a, (
            "the ibz and full_bz tables are the same object; the cache key "
            "has lost the k-set and one k-mode is serving the other")
        assert a.sharding.is_fully_replicated
        assert np.array_equal(np.asarray(a), loader.box_index(k="ibz"))
