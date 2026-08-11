"""The looped-with-partials route: its checkpoint, its resume, its refusals.

WHAT THE ROUTE IS.  ``compute_mpa_sigma_c_omega_grid`` already walks the
whole pole axis in one process with one pole's slab resident — the farm's
only structural addition is a CHECKPOINT between the terms of that sum.
``pass_partial_dir`` (deck key ``mpa_pass_partial_dir``) adds the
checkpoint without the farm: a per-pole accumulator, gathered and stripped
at the end of each ``for p`` iteration, written in the single-pole partial
format a farm leg writes, then folded into a running host-space total.

WHAT THESE CELLS COVER AND WHAT THEY DO NOT.  Everything here runs on the
host with no store and no devices: the naming the writer and the resume
check must agree on, the stamp checks the resume door owes, and the FOLD
ARITHMETIC — that summing per-pole cubes the way the loop does is the same
number, to the bit, as :func:`combine_pass_partials` summing the same
files.  That last one is the CPU half of the bit-identity claim; the other
half is the loop actually producing those cubes, which needs the real
integrator and therefore a multi-rank mesh leg, and is carried by the
``-G 4 -n 4`` driver run in this lane's report rather than by pytest.  A
four-DEVICE pytest process cannot stand in for it: the site's flat-k FFI
handler aborts on a multi-device mesh inside one process (measured at
every mu from 8 to 512, 2026-08-10; see ``tests/test_mpa_pass_p4.py``).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from gw.mpa import sigma_pass


RYD = 13.605693122994


def _rec(p, *, groups=("g0",)):
    return sigma_pass.PassRecord(
        pole_index=int(p), n_legacy_modes=1 + int(p), n_mpa_modes=2,
        legacy_b_mass=0.5 * (p + 1), mpa_b_mass=1.5, n_tau_nodes=7 + int(p),
        groups=[str(g) for g in groups],
        re_omega_min_ev=1.0 * p, re_omega_max_ev=2.0 * p,
        gamma_min_ev=0.1, gamma_max_ev=0.2)


def _cube(seed, shape=(3, 2, 4, 4)):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)).astype(np.complex128)


def _write(path, cube, p, *, n_p=2, om=(0.0, 0.05, 0.11), store="fit.h5",
           **kw):
    return sigma_pass.write_pass_partial(
        path, cube, [_rec(p)], n_p=n_p, poles=[int(p)],
        omega_grid_ry=np.asarray(om, dtype=np.float64), fit_src=store,
        print_fn=lambda *a, **k: None, **kw)


# ---------------------------------------------------------------------------
#  The name.  The writer and the resume check are the only two readers of
#  it and they must agree, so it is a function rather than an f-string in
#  two places.
# ---------------------------------------------------------------------------

def test_the_partial_name_is_one_function_and_sorts_in_the_pinned_order():
    names = [os.path.basename(sigma_pass.pole_partial_path("/d", p))
             for p in (0, 1, 2, 9, 10, 11)]
    assert names == [f"partial_p{p:04d}.h5" for p in (0, 1, 2, 9, 10, 11)]
    # THE PADDING IS THE POINT.  A directory listing of these is the
    # pinned ascending pole order, which is the order every other reader
    # in this pipeline — the globber behind ``mpa_pass_partial_in``, the
    # manifest tooling — already walks a partial directory in.
    assert names == sorted(names)


def test_the_partial_name_lives_under_the_directory_it_is_given():
    got = sigma_pass.pole_partial_path("/scratch/run/parts", 3)
    assert got == "/scratch/run/parts/partial_p0003.h5"


# ---------------------------------------------------------------------------
#  The resume door: a round trip, then every stamp it is required to check.
# ---------------------------------------------------------------------------

def test_a_checkpointed_pole_reads_back_as_its_cube_and_its_record(tmp_path):
    cube = _cube(1)
    path = sigma_pass.pole_partial_path(tmp_path, 1)
    _write(path, cube, 1)
    got, rec = sigma_pass.read_pole_partial(
        path, pole=1, n_p=2, omega_grid_ry=[0.0, 0.05, 0.11],
        fit_src="fit.h5")
    # BIT-IDENTICAL, not close.  A resumed pole contributes the same
    # complex128 bytes the integrating iteration would have folded, or the
    # resumed total is a different floating-point number from the fresh
    # one and neither run can be reproduced from the other.
    assert got.dtype == np.complex128
    assert got.tobytes() == cube.tobytes()
    want = _rec(1)
    assert rec.pole_index == 1
    assert rec.n_legacy_modes == want.n_legacy_modes
    assert rec.n_mpa_modes == want.n_mpa_modes
    assert rec.n_tau_nodes == want.n_tau_nodes
    assert rec.legacy_b_mass == want.legacy_b_mass
    assert rec.groups == want.groups
    assert rec.re_omega_max_ev == want.re_omega_max_ev
    assert rec.gamma_max_ev == want.gamma_max_ev


def test_red_twin_a_partial_from_another_store_is_refused(tmp_path):
    path = sigma_pass.pole_partial_path(tmp_path, 0)
    _write(path, _cube(2), 0, store="somebody_elses_fit.h5")
    with pytest.raises(ValueError, match="fit store"):
        sigma_pass.read_pole_partial(
            path, pole=0, n_p=2, omega_grid_ry=[0.0, 0.05, 0.11],
            fit_src="fit.h5")


def test_red_twin_a_partial_on_another_omega_grid_is_refused(tmp_path):
    path = sigma_pass.pole_partial_path(tmp_path, 0)
    _write(path, _cube(3), 0, om=(0.0, 0.05, 0.12))
    with pytest.raises(ValueError, match="different Σ ω grid"):
        sigma_pass.read_pole_partial(
            path, pole=0, n_p=2, omega_grid_ry=[0.0, 0.05, 0.11],
            fit_src="fit.h5")


def test_red_twin_a_partial_from_a_store_with_another_n_p_is_refused(tmp_path):
    path = sigma_pass.pole_partial_path(tmp_path, 0)
    _write(path, _cube(4), 0, n_p=8)
    with pytest.raises(ValueError, match="n_p="):
        sigma_pass.read_pole_partial(
            path, pole=0, n_p=2, omega_grid_ry=[0.0, 0.05, 0.11],
            fit_src="fit.h5")


def test_red_twin_a_window_farmed_fragment_is_not_a_pole(tmp_path):
    """THE ONE THE COMBINER'S POLE CHECK CANNOT SEE.

    A group-farmed leg's cube carries pole 0 and holds a RUN of pole 0's
    window groups.  Its pole list says ``[0]``, so a resume that only
    looked at the pole list would fold it as the whole pole and silently
    drop every group the fragment does not hold — and the result is a
    smooth, finite, plausible Σ.  The group stamp is the tell and it is
    checked.
    """
    path = sigma_pass.pole_partial_path(tmp_path, 0)
    _write(path, _cube(5), 0, group_spec="0.pos_cond:0-4/16")
    with pytest.raises(ValueError, match="WINDOW-FARMED fragment"):
        sigma_pass.read_pole_partial(
            path, pole=0, n_p=2, omega_grid_ry=[0.0, 0.05, 0.11],
            fit_src="fit.h5")


def test_red_twin_a_file_under_one_poles_name_holding_another_is_refused(
        tmp_path):
    """The name says pole 1; the manifest inside says pole 0.  Folding it
    as pole 1 adds pole 0 twice and never adds pole 1, which is two of the
    combiner's four coverage failures at once and neither is visible in
    the array.
    """
    path = sigma_pass.pole_partial_path(tmp_path, 1)
    _write(path, _cube(6), 0)
    with pytest.raises(ValueError, match="carries poles"):
        sigma_pass.read_pole_partial(
            path, pole=1, n_p=2, omega_grid_ry=[0.0, 0.05, 0.11],
            fit_src="fit.h5")


def test_red_twin_a_multi_pole_cube_is_not_a_resume_unit(tmp_path):
    """A whole-walk leg's cube (poles [0, 1]) in a resume directory."""
    path = sigma_pass.pole_partial_path(tmp_path, 0)
    sigma_pass.write_pass_partial(
        path, _cube(7), [_rec(0), _rec(1)], n_p=2, poles=[0, 1],
        omega_grid_ry=np.array([0.0, 0.05, 0.11]), fit_src="fit.h5",
        print_fn=lambda *a, **k: None)
    with pytest.raises(ValueError, match="carries poles"):
        sigma_pass.read_pole_partial(
            path, pole=0, n_p=2, omega_grid_ry=[0.0, 0.05, 0.11],
            fit_src="fit.h5")


# ---------------------------------------------------------------------------
#  The fold.  The CPU half of the bit-identity claim.
# ---------------------------------------------------------------------------

def test_the_looped_fold_is_bit_identical_to_the_combiner_on_the_same_files(
        tmp_path):
    """THE ARITHMETIC THE ROUTE RESTS ON.

    The loop folds ``sigma_total = zeros; sigma_total + cube_p`` in the
    pinned ascending walk order.  :func:`combine_pass_partials` folds
    ``total = zeros_like; total + e["cube"]`` over the files sorted into
    the same order.  They are the same expression on the same operands, so
    they are the same bits — which is what lets a looped run and a
    two-leg farm be compared at the bit rather than at a tolerance, and
    what makes a run half-resumed from disk equal to a fresh one.
    """
    c0, c1 = _cube(11), _cube(12)
    p0 = sigma_pass.pole_partial_path(tmp_path, 0)
    p1 = sigma_pass.pole_partial_path(tmp_path, 1)
    _write(p0, c0, 0)
    _write(p1, c1, 1)

    # The loop's fold, spelled exactly as the loop spells it.
    loop_total = None
    for p in sigma_pass.resolve_pass_poles(2, None, None):
        cube, _rec_p = sigma_pass.read_pole_partial(
            sigma_pass.pole_partial_path(tmp_path, p), pole=p, n_p=2,
            omega_grid_ry=[0.0, 0.05, 0.11], fit_src="fit.h5")
        if loop_total is None:
            loop_total = np.zeros_like(cube)
        loop_total = loop_total + cube

    combined, poles, _audit = sigma_pass.combine_pass_partials(
        [p0, p1], n_p=2, omega_grid_ry=[0.0, 0.05, 0.11], fit_src="fit.h5",
        print_fn=lambda *a, **k: None)
    assert poles == (0, 1)
    assert loop_total.tobytes() == combined.tobytes()
    # And it is a genuine sum, not two aliases of one cube.
    assert loop_total.tobytes() != c0.tobytes()


def test_a_half_resumed_walk_is_the_same_number_as_a_whole_one(tmp_path):
    """Pole 0 off disk, pole 1 freshly integrated, in the pinned order.

    The resume changes WHERE a term comes from and never where it lands in
    the sum, so the mixed walk has to equal the all-fresh one bit for bit;
    if it did not, a job that died and restarted would produce a Σ that no
    rerun could reproduce.
    """
    c0, c1 = _cube(21), _cube(22)
    p0 = sigma_pass.pole_partial_path(tmp_path, 0)
    _write(p0, c0, 0)

    fresh = np.zeros_like(c0)
    fresh = fresh + c0
    fresh = fresh + c1

    mixed = None
    from_disk, _r = sigma_pass.read_pole_partial(
        p0, pole=0, n_p=2, omega_grid_ry=[0.0, 0.05, 0.11], fit_src="fit.h5")
    for cube in (from_disk, c1):
        if mixed is None:
            mixed = np.zeros_like(cube)
        mixed = mixed + cube
    assert mixed.tobytes() == fresh.tobytes()


# ---------------------------------------------------------------------------
#  The resume decision at more than one rank.  The symptom of getting this
#  wrong is a HANG, so it is the one branch that gets a red twin standing
#  in for four processes.
# ---------------------------------------------------------------------------

def test_at_one_process_the_resume_is_just_the_file_being_there(tmp_path):
    p = sigma_pass.pole_partial_path(tmp_path, 0)
    assert sigma_pass.agree_on_resume(p, pole=0) is False
    _write(p, _cube(31), 0)
    assert sigma_pass.agree_on_resume(p, pole=0) is True


def _four_ranks(monkeypatch, votes):
    """``process_count() == 4`` with a chosen allgather outcome.

    Verbatim ``process_allgather(tiled=False)`` semantics on a fully
    addressable operand: the per-rank value comes back stacked on a new
    leading axis of length four.  pytest is ONE process, so the axis has
    to be installed rather than allocated -- the same technique
    ``tests/test_mpa_pass_p4.py`` uses for C1's red twin, and for the same
    reason: no in-suite cell can produce a process axis natively.
    """
    import jax
    from gw import ppm_windows as PW

    monkeypatch.setattr(jax, "process_count", lambda: 4)
    monkeypatch.setattr(
        PW, "_to_host_np",
        lambda a, dtype=np.complex128, *, tiled=False:
            np.asarray(votes, dtype=dtype).reshape(4, 1))


def test_four_ranks_that_all_see_the_checkpoint_all_skip(monkeypatch,
                                                         tmp_path):
    _four_ranks(monkeypatch, [1, 1, 1, 1])
    assert sigma_pass.agree_on_resume(
        sigma_pass.pole_partial_path(tmp_path, 2), pole=2) is True


def test_four_ranks_that_all_miss_the_checkpoint_all_integrate(monkeypatch,
                                                               tmp_path):
    _four_ranks(monkeypatch, [0, 0, 0, 0])
    assert sigma_pass.agree_on_resume(
        sigma_pass.pole_partial_path(tmp_path, 2), pole=2) is False


def test_red_twin_ranks_that_disagree_refuse_instead_of_deadlocking(
        monkeypatch, tmp_path):
    """THE ONE THAT HAS NO ERROR MESSAGE IF IT IS NOT CAUGHT HERE.

    Two ranks folding pole 2 from disk and two integrating it do not
    return different numbers -- they meet in the next tau dispatch's
    collective and stop, with no traceback, which on a batch queue is
    indistinguishable from a slow leg.
    """
    _four_ranks(monkeypatch, [1, 1, 0, 1])
    with pytest.raises(RuntimeError, match="do not agree"):
        sigma_pass.agree_on_resume(
            sigma_pass.pole_partial_path(tmp_path, 2), pole=2)


# ---------------------------------------------------------------------------
#  The refusals, asked before the store is opened.
# ---------------------------------------------------------------------------

def _refusal_call(tmp_path, **kw):
    """``compute_mpa_sigma_c_omega_grid`` up to its first refusal.

    ``wfns``/``meta``/``mesh_xy`` are ``None`` and ``fit_src`` names a file
    that does not exist ON PURPOSE: the refusal under test is required to
    fire before any of them is touched, so a cell that had to build a
    store to reach it would be testing a later refusal than the one that
    exists.  If one of these ever raises AttributeError or OSError instead
    of ValueError, the guard has drifted below the store read.
    """
    return sigma_pass.compute_mpa_sigma_c_omega_grid(
        None, str(tmp_path / "no_such_store.h5"), None, None,
        ppm_cfg=SimpleNamespace(window_edge_factor=1.5,
                                regularization_ev=0.5,
                                fermi_reference="midgap"),
        quad=None, omega_grid_ry=[0.0, 0.05],
        pass_partial_dir=str(tmp_path / "parts"),
        print_fn=lambda *a, **k: None, **kw)


@pytest.mark.parametrize("kw, key", [
    ({"pole_subset": (0,)}, "mpa_pole_subset"),
    ({"group_subset": {(0, "pos_cond"): (0, 4, 16)}}, "mpa_group_subset"),
    ({"census_out": "census.json"}, "mpa_pass_census_out"),
])
def test_the_looped_route_refuses_the_keys_that_redefine_its_cubes(
        tmp_path, kw, key):
    with pytest.raises(ValueError, match=key):
        _refusal_call(tmp_path, **kw)


def test_the_refusal_names_every_clashing_key_not_just_the_first(tmp_path):
    with pytest.raises(ValueError) as exc:
        _refusal_call(tmp_path, pole_subset=(0,), census_out="c.json")
    assert "mpa_pole_subset" in str(exc.value)
    assert "mpa_pass_census_out" in str(exc.value)


def test_the_directory_is_created_by_the_route_that_writes_into_it(tmp_path):
    """A deck naming a directory that does not exist yet is not an error;
    a run that got as far as pole 0 and then failed to write its
    checkpoint would have paid the whole integration for nothing.
    """
    parts = tmp_path / "deep" / "parts"
    with pytest.raises(Exception):
        # Fails later, in the store read -- the point is only that it got
        # past the guard, and that the guard made the directory on its way.
        _refusal_call(tmp_path)
    assert os.path.isdir(tmp_path / "parts")
    assert not parts.exists()          # nothing else was created


# ---------------------------------------------------------------------------
#  The deck key, at the config layer.
# ---------------------------------------------------------------------------

def test_the_deck_key_exists_and_defaults_to_the_unlooped_walk():
    from gw import gw_config

    assert gw_config._DEFAULTS["mpa_pass_partial_dir"] == ""
    assert "mpa_pass_partial_dir" in gw_config.LorraxConfig.__annotations__
