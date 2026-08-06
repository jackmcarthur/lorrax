"""The SC eigh is a layout choice, not a side effect of a physics knob.

Until 2026-08-05 ``gw_iteration_map`` reached ``distributed_eigh_bands``
— the only eigh on this surface that keeps no whole ``(nb, nb)`` tile on
one rank — solely through
``if bool(getattr(inputs.config, "density_self_consistent", False)):``.
That flag rebuilds ρ from the QP orbitals and defaults to False, so on
the default configuration every SC iteration ≥ 1 put a full ``(nb, nb)``
on one device: 1.6 GB at nb=1e4, which is the residency the scaling
target forbids.

``config.sc.eigh`` (deck key ``sc_eigh``) is the decision now.  This file
pins the resolution table, both refusals, and — where an FFI distributed
eigh is reachable — that the two implementations agree.

The resolver reads only the mesh's geometry, so the table is exercised
against a duck-typed mesh: a real 8×8 ``jax.sharding.Mesh`` needs 64
devices and this suite runs on one.  The agreement test uses a real 1×1
mesh and skips when no distributed backend resolves.

Requires jax (``sc_iteration`` imports it at module scope), so this runs
in the container, not on a login node.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

jax = pytest.importorskip("jax")

import jax.numpy as jnp                                        # noqa: E402
from jax.sharding import Mesh                                  # noqa: E402

from gw import sc_iteration                                    # noqa: E402
from gw.sc_iteration import (                                  # noqa: E402
    _SC_EIGH_TILE_BUDGET_FRACTION, _resolve_sc_eigh)


class _FakeMesh:
    """The three members ``_resolve_sc_eigh`` reads off a ``Mesh``."""

    def __init__(self, px, py):
        self.axis_names = ("x", "y")
        self.shape = {"x": px, "y": py}
        self.size = px * py


class _SC:
    def __init__(self, eigh):
        self.eigh = eigh


class _Mem:
    def __init__(self, per_device_gb):
        self.per_device_gb = per_device_gb


class _Config:
    def __init__(self, eigh="auto", per_device_gb=40.0):
        self.sc = _SC(eigh)
        self.memory = _Mem(per_device_gb)


def _resolver_src():
    """Body of ``_resolve_sc_eigh`` — sliced to the NEXT top-level def.

    Not to ``def _midgap_efermi(``: that is defined EARLIER in the file
    (line ~283 vs ~347), so slicing to it silently yields an empty string
    and every assertion on the body passes vacuously.
    """
    src = pathlib.Path(sc_iteration.__file__).read_text()
    at = src.index("def _resolve_sc_eigh(")
    nxt = src.index("\ndef ", at)
    return src[at:nxt]


def _resolve(nb, px, py, lines=None, **kw):
    return _resolve_sc_eigh(nb, _FakeMesh(px, py), _Config(**kw),
                            print_fn=(lines.append if lines is not None
                                      else (lambda *_: None)))


# ---------------------------------------------------------------------------
# The decoupling itself
# ---------------------------------------------------------------------------

def test_the_eigh_choice_does_not_read_density_self_consistent():
    """The defect, stated as a test: the resolver takes no physics knob.

    Its only inputs are ``nb``, the mesh and ``config.sc.eigh`` /
    ``config.memory``, so a config with no ``density_self_consistent``
    attribute at all must resolve.
    """
    cfg = _Config()
    assert not hasattr(cfg, "density_self_consistent")
    got = _resolve_sc_eigh(128, _FakeMesh(2, 2), cfg,
                           print_fn=lambda *_: None)
    assert got in ("native", "distributed")


def test_iteration_map_resolves_the_eigh_separately_from_the_efermi_rule():
    """Source gate: the two decisions must not share a branch again.

    ``density_self_consistent`` may still select the E_F rule — that is a
    physics question — but it must not appear in the eigh dispatch.
    """
    src = pathlib.Path(sc_iteration.__file__).read_text()
    body = src[src.index("def gw_iteration_map("):
               src.index("def _scissor_E_qp_for_outofrange(")]
    eigh_at = body.index("eigh_kind = _resolve_sc_eigh(")
    call_at = body.index("E_qp_ry, U_qp = distributed_eigh_bands(")
    dsc_at = body.index('"density_self_consistent"', eigh_at)
    assert eigh_at < call_at < dsc_at, (
        "the eigh dispatch must be resolved before, and independently of, "
        "the density_self_consistent branch")


# ---------------------------------------------------------------------------
# The resolution table
# ---------------------------------------------------------------------------

def test_explicit_native_is_honoured_whatever_the_size():
    assert _resolve(100_000, 8, 8, eigh="native") == "native"


def test_small_tiles_stay_on_the_native_batch():
    """The measured decks must not move: nb=128 is 0.25 MiB, nb=512 is 4 MiB."""
    assert _resolve(128, 2, 2) == "native"
    assert _resolve(512, 8, 8) == "native"


def test_one_device_stays_native():
    """On a 1×1 mesh 'distributed' is the same tile with an FFI call on it."""
    assert _resolve(100_000, 1, 1) == "native"


def test_an_indivisible_band_window_stays_native_under_auto_for_SIZE():
    """nb = 46 stays native, but the REASON changed on 2026-08-06.

    It used to be divisibility: ``distributed_eigh_bands`` padded and did
    not unpad, so ``auto`` refused to walk into a silent shape change.
    That callee now pads with a sentinel and slices back by count, so
    divisibility is not a condition any more — nb = 46 stays native
    purely because one (46, 46) tile is 33 kB, far under
    ``_SC_EIGH_TILE_BUDGET_FRACTION`` of the 40 GB/device budget.

    nb = 46 is the repo's own gnppm fixture (nval + ncond = 26 + 20);
    the band divisor of an 8×8 mesh is 64 and 46 % 64 = 46.
    """
    assert 46 % 64 != 0
    lines = []
    assert _resolve(46, 8, 8, lines=lines) == "native"
    assert not lines, ("a small tile must resolve on size alone, without "
                       "probing a backend")
    # ...and it is the SIZE holding it there, not divisibility: shrink the
    # budget and the SAME indivisible window now reaches the backend probe.
    # (It may still land on native when no distributed backend resolves on
    # this node -- that is the documented degrade, and it PRINTS, which is
    # what distinguishes "tried and degraded" from "short-circuited".)
    lines = []
    _resolve(46, 8, 8, per_device_gb=1e-6, lines=lines)
    assert any("auto wanted the distributed eigh" in L for L in lines), (
        "an indivisible nb must now reach the backend probe; if it never "
        "does, the lifted refusal has been reinstated by the auto branch")


def test_an_indivisible_band_window_is_SERVED_when_asked_explicitly():
    """The 2026-08-06 contract change: this used to raise ValueError.

    The old refusal was a SHAPE objection — the callee returned
    ``(nk, nb_pad)`` / ``(nk, nb_pad, nb_pad)`` which the carry and every
    band-indexed operand beside it would not match.  It now returns the
    logical extent at any nb, so there is nothing left to refuse.
    """
    assert _resolve(46, 8, 8, eigh="distributed") == "distributed"
    assert _resolve(10, 2, 2, eigh="distributed") == "distributed"


def test_the_backend_probe_asks_about_the_PADDED_extent():
    """``auto`` must probe the extent the eigh runs at, not ``nb``.

    ``ffi.linalg.resolve`` has its own divisibility guard.  Probing the
    logical ``nb`` would trip it for every indivisible window and degrade
    to native — reinstating the lifted refusal by accident, silently.
    """
    body = _resolver_src()
    assert "n=nb_solve" in body, (
        "the resolve_backend probe must be asked about round_up(nb, "
        "pad_div), not nb")
    assert "round_up(nb, pad_div)" in body


def test_the_divisor_is_the_band_divisor_not_the_two_axes_separately():
    """nb = 10 on a 2×2 mesh divides both axes and is STILL padded to 12.

    The divisor is ``spec_divisor(mesh, band_sphere_spec(), 1)`` — px·py
    on the default psi layout — not the two axes separately.  That still
    matters after the refusal was lifted, because it is the extent the
    backend probe and the pad both use; it is simply no longer a reason
    to refuse.
    """
    from runtime.padding import round_up
    assert 10 % 2 == 0                       # divides both axes...
    assert round_up(10, 2 * 2) == 12         # ...and is still padded to 12
    # and the resolver takes that divisor from the spec, not from px, py
    assert "spec_divisor(mesh_xy, band_sphere_spec(), 1)" in _resolver_src()


# ---------------------------------------------------------------------------
# The pad contract the lifted refusal now depends on
# ---------------------------------------------------------------------------

def test_the_pad_is_a_sentinel_and_the_result_is_sliced_back_by_count():
    """Source gate on ``distributed_eigh_bands``.

    The refusal above was lifted ONLY because this callee stopped
    returning padded shapes.  If the slice-back is ever removed, the
    resolver silently starts handing the carry ``(nk, nb_pad)`` again —
    so pin both halves here, next to the test that depends on them.
    """
    from gw import qsgw_density
    src = pathlib.Path(qsgw_density.__file__).read_text()
    body = src[src.index("def distributed_eigh_bands("):]
    body = body[:body.index("\ndef ")]
    assert "_EIGH_PAD_SENTINEL_RY" in body, (
        "the band pad must carry the large diagonal sentinel; a zero pad "
        "injects exact-0.0 eigenvalues mid-spectrum and moves band order, "
        "_midgap_efermi and the occupations")
    assert "E = E[:, :nb]" in body and "U = U[:, :nb, :nb]" in body, (
        "the eigh result must be sliced back to the LOGICAL band extent "
        "by COUNT — that is what makes the sentinel correct rather than "
        "merely convenient, and what the lifted refusal relies on")


def test_the_sentinel_is_orders_clear_of_any_physical_eigenvalue():
    """1e10 Ry vs O(1) Ry QP eigenvalues.

    Not a style check: the sentinel is only safe if pad eigenvalues
    cannot interleave with the physical spectrum at ANY deck.  An
    identity pad (1.0 Ry = 13.6 eV) fails exactly this — it is above most
    states of interest but inside a wide QP window.
    """
    from gw.qsgw_density import _EIGH_PAD_SENTINEL_RY as S
    assert S >= 1e8
    assert S / 13.6057 > 1e6          # eV, vs any conceivable QP window


def test_a_sentinel_pad_block_decouples_exactly():
    """The property the whole scheme rests on, checked numerically.

    ``[H 0; 0 sI]`` is block diagonal with zero coupling, so the pad
    eigenvalues are EXACTLY s, they occupy the last slots of an ascending
    spectrum, and the physical eigenvectors carry zero weight on pad rows.
    """
    from gw.qsgw_density import _EIGH_PAD_SENTINEL_RY as S
    rng = np.random.default_rng(3)
    nb, nb_pad = 10, 16
    A = rng.normal(size=(nb, nb))
    H = 0.5 * (A + A.T)
    Hp = np.zeros((nb_pad, nb_pad))
    Hp[:nb, :nb] = H
    Hp[np.arange(nb, nb_pad), np.arange(nb, nb_pad)] = S
    w, V = np.linalg.eigh(Hp)
    assert np.all(w[nb:] == S)                       # exactly, not approx
    assert np.abs(w[:nb] - np.linalg.eigh(H)[0]).max() < 1e-10
    assert np.abs(V[nb:, :nb]).max() == 0.0          # no leakage into pad rows


def test_a_large_tile_leaves_the_native_batch():
    """nb = 16384 on a 4×4 mesh: one tile is 4 GiB against a 40 GB budget.

    ``auto`` still lands on ``native`` when no distributed eigh backend
    resolves on this node — a deliberate fallback that prints its reason
    — so the size test is read off either outcome.
    """
    lines = []
    got = _resolve(16384, 4, 4, lines=lines)
    assert got == "distributed" or any(
        "auto wanted the distributed eigh" in ln for ln in lines)


def test_the_budget_fraction_is_a_fraction():
    assert 0.0 < _SC_EIGH_TILE_BUDGET_FRACTION < 0.5


def test_the_threshold_moves_with_the_device_budget():
    """Derived from bytes and the budget, not from a band count.

    Same nb (one tile = 1 GiB) on the same mesh: a 4 GB/device budget
    must try the distributed path, a 4000 GB one must not.
    """
    small, big = [], []
    _resolve(8192, 2, 2, lines=small, per_device_gb=4.0)
    got_big = _resolve(8192, 2, 2, lines=big, per_device_gb=4000.0)
    assert got_big == "native" and big == []
    assert any("auto wanted the distributed eigh" in ln for ln in small) or \
        _resolve(8192, 2, 2, per_device_gb=4.0) == "distributed"


def test_a_bad_sc_eigh_value_is_refused_at_config_parse():
    from gw.gw_config import SCConfig
    with pytest.raises(ValueError, match="sc_eigh must be"):
        SCConfig(max_iter=3, tol_ev=1e-4, accelerator="rcrop",
                 history_depth=5, mixing=1.0, dump_dir=None, eigh="nope")


def test_sc_eigh_defaults_to_auto():
    from gw.gw_config import SCConfig
    sc = SCConfig(max_iter=3, tol_ev=1e-4, accelerator="rcrop",
                  history_depth=5, mixing=1.0, dump_dir=None)
    assert sc.eigh == "auto"


# ---------------------------------------------------------------------------
# The eigh the dispatch selects satisfies the contract
# ---------------------------------------------------------------------------
#
# THE DISTRIBUTED SIDE IS NOT EXERCISED HERE, DELIBERATELY.  Calling
# ``distributed_eigh_bands`` in a bare pytest process resolves — the FFI
# host library is loaded and every mesh guard passes on a 1x1 mesh — and
# then ScaLAPACK/BLACS ABORTS THE INTERPRETER, because there is no MPI
# context.  A ``resolve_backend`` probe does not predict that: resolve
# succeeded and the process still died.  Measured on job 7890040, where
# it killed the run at the last test of the file and would have taken the
# whole suite with it.  (Under ``srun --mpi=pmi2`` the same test passes —
# same job, step 3 — which is exactly why a probe-and-skip is not enough:
# whether it aborts depends on the launcher, not on anything the process
# can ask.)
#
# The two paths are compared END TO END instead, on a real deck, which is
# a stronger statement than a unit test: job 7890020 ran the mos2_4x4
# 3-iteration rCROP/IBZ arm twice on the same tree, once at the default
# (native) and once at ``sc_eigh = distributed``, and eqp0 agreed to
# max|dE_QP| = 7.000001e-09 eV, rms 1.694661e-09 eV (eqp1: 7.000001e-09 /
# 1.627162e-09).  ``multi_device/batched_eigh_dispatch_gate.py`` item 7
# gates ``distributed_eigh_bands`` itself.
#
# What runs here is the branch the default path takes, against a host
# reference — no FFI, no MPI, no abort.

def _random_hermitian(rng, nk, nb):
    A = (rng.normal(size=(nk, nb, nb))
         + 1j * rng.normal(size=(nk, nb, nb))).astype(np.complex128)
    return 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))


def test_the_native_eigh_matches_a_host_reference():
    """The contract both branches owe, checked on the one that always runs.

    Eigenvalues at 1e-12 relative; ``U`` only through ``U diag(E) Uᴴ``,
    because a degenerate eigenvalue leaves an arbitrary unitary inside
    its subspace and no implementation promises a representative.  That
    reconstruction is also exactly what the SC loop consumes ``U`` for —
    psi is rotated with U and Sigma is rotated back with the same U — so
    it is the comparable quantity, not a weakened one.

    The transposed-U negative control matters: the transpose of a
    unitary is also unitary, so the reconstruction alone would pass on an
    implementation that returned ROWS instead of columns.
    """
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    rng = np.random.default_rng(2)
    H = _random_hermitian(rng, 3, 8)

    eigh_kshard, _ = sc_iteration._kshard_eigh_kernels(
        mesh, sc_iteration._band_rotation_spec())
    E_n, U_n = (np.asarray(a) for a in eigh_kshard(jnp.asarray(H)))
    E_h, U_h = np.linalg.eigh(H)

    scale = float(np.abs(E_h).max())
    assert E_n.shape == E_h.shape
    assert float(np.abs(E_n - E_h).max()) <= 1e-12 * scale

    def rebuild(E, U):
        return U @ (E[:, :, None] * np.conj(np.swapaxes(U, -1, -2)))

    assert np.allclose(rebuild(E_n, U_n), rebuild(E_h, U_h),
                       atol=1e-11 * scale, rtol=0.0)
    # Columns, not rows: A U == U diag(E).
    assert np.allclose(H @ U_n, U_n * E_n[:, None, :],
                       atol=1e-11 * scale, rtol=0.0)
    U_bad = np.conj(np.swapaxes(U_n, -1, -2))
    assert not np.allclose(H @ U_bad, U_bad * E_n[:, None, :],
                           atol=1e-11 * scale, rtol=0.0)
