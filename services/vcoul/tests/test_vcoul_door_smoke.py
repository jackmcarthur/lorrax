"""The door, exercised: does what comes through it compute?

MINIMAL BY DESIGN, and the design says so out loud.  This is the step-1
bootstrap tier — enough to prove the package is wired, imports on its own
and answers with numbers of the right shape and sign.  The real suite is
step 2 (SERVICE_FORM tiers L-a..L-c: closed-form ``v(q+G)`` against an
independently-written metric evaluation, head-slot injection ordering,
knob liveness, the ``wrap_points_to_voronoi`` membership property,
constructible refusals).  Nothing here is a substitute for those, and a
cell here that started asserting physics would be the wrong file.

What IS pinned here, because it is what the extraction claims:

* the door's public surface is complete and its arity is what the
  layering rule's ``door_names`` reads;
* ``get_kernel`` dispatches on all three dimensionalities and refuses a
  fourth BY NAME;
* the ``v_qG_table`` driver runs on synthetic arrays through the door,
  with the head-slot injection and the cutoff mask both live;
* ``CoulombGeometry.from_wfn`` takes a duck-typed stand-in — the whole
  reason the service can accept a lorrax loader without importing one.
"""

from __future__ import annotations

import numpy as np
import pytest

import vcoul as V


# ---------------------------------------------------------------------------
# Fixtures: a cubic cell and a non-cubic (hexagonal, MoS2-class) one.  The
# non-cubic row is not decoration — a cubic-only smoke test is exactly what
# let a transposed draw ship for three months (tests/test_vcoul_minibz_
# head_draw.py has the proof that silicon cannot see it).
# ---------------------------------------------------------------------------
CUBIC = 2.0 * np.pi * np.eye(3)
_A_HEX = 3.16 / 0.529177
_C_HEX = 12.3 / 0.529177
HEX = 2.0 * np.pi * np.array([
    [1.0 / _A_HEX, 1.0 / (np.sqrt(3.0) * _A_HEX), 0.0],
    [0.0, 2.0 / (np.sqrt(3.0) * _A_HEX), 0.0],
    [0.0, 0.0, 1.0 / _C_HEX]])


def _celvol(bvec):
    return abs(8.0 * np.pi ** 3 / np.linalg.det(bvec))


def _geom(bvec):
    return V.CoulombGeometry(bvec=bvec, cell_volume=_celvol(bvec))


def _gvecs():
    """``(1, 3, nG)`` Miller table with G=(0,0,0) at slot 0."""
    g = [(0, 0, 0)]
    g += [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
          if (i, j, k) != (0, 0, 0)]
    return np.asarray(g, dtype=np.float64).T[None, :, :]


# ---------------------------------------------------------------------------
# The door itself
# ---------------------------------------------------------------------------

def test_every_name_in_all_is_actually_there():
    """``__all__`` is what ``tests/test_layering.py`` reads to decide which
    ``from vcoul import X`` spellings are door traffic rather than a reach
    past it, so a name listed but not bound would silently widen that rule.
    """
    missing = [n for n in V.__all__ if not hasattr(V, n)]
    assert not missing, missing
    assert len(V.__all__) == len(set(V.__all__)), "duplicate in __all__"


def test_the_door_does_not_export_what_stayed_in_lorrax():
    """The boundary, from the other side.

    ``compute_all_V_q`` (the ζ-layout dispatcher), ``build_bgw_v_grid_fn``
    (config + path reading) and ``compute_q0_averages`` (deck-facing) are
    named in DESIGN_vcoul.md as STAYING in gw.  If one of them turned up
    here later, the service would have grown a deck edge and this cell is
    where that is noticed.
    """
    for name in ("compute_all_V_q", "build_bgw_v_grid_fn",
                 "compute_q0_averages", "Meta"):
        assert not hasattr(V, name), (
            f"vcoul exports {name!r}, which DESIGN_vcoul.md keeps in gw")


# ---------------------------------------------------------------------------
# get_kernel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sys_dim,cls", [(3, "Bulk3D"), (2, "Slab2D"),
                                         (0, "Box0D")])
def test_get_kernel_dispatches(sys_dim, cls):
    k = V.get_kernel(sys_dim)
    assert type(k).__name__ == cls
    assert int(k.sys_dim) == sys_dim


def test_get_kernel_defaults_to_bulk_3d():
    assert type(V.get_kernel(None)).__name__ == "Bulk3D"


def test_get_kernel_refuses_an_unknown_dimensionality():
    """RED TWIN for the dispatch: the refusal must fire and must SAY the
    vocabulary, or a deck typo becomes a silent 3D run."""
    with pytest.raises(ValueError, match="expected 0 .box., 2 .slab."):
        V.get_kernel(1)


# ---------------------------------------------------------------------------
# v_qG_table through the door
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bvec", [CUBIC, HEX], ids=["cubic", "hex"])
@pytest.mark.parametrize("sys_dim", [3, 2])
def test_v_qG_table_runs_on_synthetic_arrays(bvec, sys_dim):
    geom = _geom(bvec)
    gvec = _gvecs()
    q = np.array([[0.25, 0.0, 0.0]])
    v = V.v_qG_table(V.get_kernel(sys_dim), q, gvec, geometry=geom)

    assert v.shape == (1, gvec.shape[2])
    assert v.dtype == np.float64
    assert np.all(np.isfinite(v))
    # q+G is never zero at q=(0.25,0,0), so nothing is in the zeroed slot
    # and every entry is a positive Coulomb value.
    assert np.all(v > 0.0), v


def test_q_plus_G_equals_zero_is_zeroed_not_infinite():
    """The ``denom < 1e-12`` guard, at the one q where it fires."""
    v = V.v_qG_table(V.get_kernel(3), np.zeros((1, 3)), _gvecs(),
                     geometry=_geom(CUBIC))
    assert v[0, 0] == 0.0, "G=(0,0,0) at q=0 must be zeroed, not 8pi/0"
    assert np.all(np.isfinite(v))
    assert np.all(v[0, 1:] > 0.0)


def test_the_head_slot_injection_lands_at_miller_zero():
    """``v_head_miniBZ`` replaces the Miller-(0,0,0) slot and only it."""
    geom = _geom(CUBIC)
    gvec = _gvecs()
    q = np.array([[0.25, 0.0, 0.0]])
    kgrid = (4, 4, 4)
    head = np.zeros(kgrid, dtype=np.float64)
    head[1, 0, 0] = 12345.0                      # q_frac (0.25,0,0) -> (1,0,0)

    plain = V.v_qG_table(V.get_kernel(3), q, gvec, geometry=geom)
    with_head = V.v_qG_table(V.get_kernel(3), q, gvec, geometry=geom,
                             v_head_miniBZ=head)
    assert with_head[0, 0] == 12345.0
    np.testing.assert_array_equal(with_head[0, 1:], plain[0, 1:])


def test_the_cutoff_mask_zeroes_past_the_radius_and_the_head_with_it():
    """Order is load-bearing: the head goes in BEFORE the cutoff, so a head
    slot outside the bare-Coulomb cutoff is zeroed like any other G."""
    geom = _geom(CUBIC)
    gvec = _gvecs()
    q = np.array([[0.25, 0.0, 0.0]])
    head = np.zeros((4, 4, 4), dtype=np.float64)
    head[1, 0, 0] = 12345.0

    # |q+G|^2 at G=0, q=(0.25,0,0) on the cubic cell is (0.25*2pi)^2 ~ 2.47.
    cut = V.v_qG_table(V.get_kernel(3), q, gvec, geometry=geom,
                       v_head_miniBZ=head, vcoul_cutoff_ry=1.0)
    assert cut[0, 0] == 0.0, (
        "the head slot survived a cutoff it sits outside of — the "
        "injection must happen before the mask")
    loose = V.v_qG_table(V.get_kernel(3), q, gvec, geometry=geom,
                         v_head_miniBZ=head, vcoul_cutoff_ry=1.0e6)
    assert loose[0, 0] == 12345.0


def test_v_qG_table_refuses_a_malformed_components_table():
    with pytest.raises(ValueError, match=r"must be \(n_q, 3, ngkmax\)"):
        V.v_qG_table(V.get_kernel(3), np.zeros((1, 3)), np.zeros((1, 4, 5)),
                     geometry=_geom(CUBIC))


def test_v_qG_table_refuses_a_malformed_head_table():
    with pytest.raises(ValueError, match=r"must be \(nkx, nky, nkz\)"):
        V.v_qG_table(V.get_kernel(3), np.zeros((1, 3)), _gvecs(),
                     geometry=_geom(CUBIC), v_head_miniBZ=np.zeros((4, 4)))


# ---------------------------------------------------------------------------
# CoulombGeometry.from_wfn — the duck type
# ---------------------------------------------------------------------------

class _FakeWfn:
    """Everything ``from_wfn`` is allowed to know about a loader.

    Deliberately NOT a lorrax class: the service declares no dependency on
    lorrax, so it cannot name ``file_io.WfnLoader`` — and it must not need
    to.  Anything carrying these attributes works.
    """
    def __init__(self, bvec, blat=2.0, with_box=True):
        self.blat = blat
        self.bvec = np.asarray(bvec, dtype=np.float64) / blat
        self.cell_volume = _celvol(np.asarray(bvec, dtype=np.float64))
        if with_box:
            self.bdot = np.asarray(bvec) @ np.asarray(bvec).T
            self.fft_grid = np.asarray((6, 6, 6), dtype=int)


def test_from_wfn_takes_the_product_once():
    """``blat * bvec`` — the multiplication five call sites used to write."""
    w = _FakeWfn(HEX, blat=2.0)
    g = V.CoulombGeometry.from_wfn(w)
    np.testing.assert_allclose(g.bvec, HEX, rtol=0, atol=1e-12)
    assert g.cell_volume == pytest.approx(_celvol(HEX))
    assert g.bvec.dtype == np.float64


def test_from_wfn_without_the_box_extras_is_fine():
    """``bdot`` / ``fft_grid`` are read only by the 0-D kernel; a 3-D
    loader that has neither must still build a geometry."""
    g = V.CoulombGeometry.from_wfn(_FakeWfn(CUBIC, with_box=False))
    assert g.bdot is None and g.fft_grid is None


def test_the_geometry_refuses_a_bvec_that_is_not_three_by_three():
    """RED TWIN for the shape guard, and the reason it exists: the failure
    it prevents (passing ``wfn.bvec`` without ``blat``) has no shape error
    of its own, so the constructor is the only place to catch a wrong-shape
    cousin of it."""
    with pytest.raises(ValueError, match="must be .3, 3."):
        V.CoulombGeometry(bvec=np.eye(2), cell_volume=1.0)


def test_the_geometry_is_frozen():
    g = _geom(CUBIC)
    with pytest.raises(Exception):
        g.cell_volume = 2.0


# ---------------------------------------------------------------------------
# The one path that needs the box extras
# ---------------------------------------------------------------------------

def test_box_0d_needs_bdot_and_fft_grid_and_says_so():
    """A geometry without the box extras must REFUSE rather than compute
    a Wigner-Seitz FFT out of nothing."""
    with pytest.raises(ValueError, match="bdot and fft_grid"):
        V.v_qG_table(V.get_kernel(0), np.zeros((1, 3)), _gvecs(),
                     geometry=_geom(CUBIC))


def test_box_0d_serves_q_equals_zero_and_refuses_finite_q():
    geom = V.CoulombGeometry.from_wfn(_FakeWfn(CUBIC))
    v = V.v_qG_table(V.get_kernel(0), np.zeros((1, 3)), _gvecs(),
                     geometry=geom)
    assert v.shape == (1, _gvecs().shape[2])
    assert np.all(np.isfinite(v))
    # G=0 is NOT zeroed for a box: the truncated v(G=0) is finite.
    assert v[0, 0] != 0.0

    with pytest.raises(NotImplementedError, match="q=0 routine"):
        V.v_qG_table(V.get_kernel(0), np.array([[0.25, 0.0, 0.0]]), _gvecs(),
                     geometry=geom)
