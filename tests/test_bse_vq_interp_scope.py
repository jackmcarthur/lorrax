"""``vq_interp``'s slab scope (``slab_scope_violations`` / ``assert_slab_scope``).

FIXTURE-FREE, for the same reason ``test_bse_vq_interp_criterion.py`` is:
``tests/test_bse_vq_interp.py`` is gated behind an out-of-repo Perlmutter
fixture and skips everywhere else, and everything asserted here is a
statement about CELL GEOMETRY plus one string off the restart's
Coulomb-policy stamp, so it runs wherever pytest runs, on numpy alone.

WHAT THIS PINS, and why it exists.  ``PIPELINE_HEALTH.md`` punch row 23
recorded ``bse.exciton_bands --vq-mode interp`` failing two gates on
``si_bse_debug`` -- ``makeVq_vs_disk`` at 3.218e-01 and ``slab_axes_offdiag``
at 1.000e+00 -- and read them as a defect of the off-grid exchange.  They are
not.  Both are the module's slab assumption meeting a 3-D fcc cell: the
kernel is the Ismail-Beigi 2-D truncation with no ``sys_dim`` anywhere in the
file, and the long-range model fits an in-plane polynomial per ``|G_z|``
channel, which is exact only where ``K_z`` is constant within a channel.  The
cells below are the two real decks of that measurement -- Si bulk (refused)
and the MoS2 reference slab (served) -- so a future change that quietly
widened or narrowed the scope has to walk past them.
"""
import numpy as np
import pytest

from bse import vq_interp as V

# Same geometries, in bohr, as tests/test_bse_vq_interp_criterion.py.  Si is
# si_bse_debug's cell (fcc primitive, a = 10.262); MoS2 is the reference slab.
A_SI, A_MOS2, C_MOS2 = 10.262, 5.9715, 22.677


def _fcc_bvec(a):
    A = 0.5 * a * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    return 2 * np.pi * np.linalg.inv(A).T


def _hex_slab_bvec(a, c):
    A = np.array([[a, 0., 0.], [-0.5 * a, np.sqrt(3) / 2 * a, 0.],
                  [0., 0., c]])
    return 2 * np.pi * np.linalg.inv(A).T


def _planar_q(n=4):
    """A q_z = 0 coarse grid — what a slab deck has."""
    t = (np.arange(n) - n // 2) / n
    return np.array([[x, y, 0.0] for x in t for y in t])


def _bgw_wrapped_q(kgrid=(4, 4, 4)):
    """si_bse_debug's own 4x4x4 q list in the BGW wrap, as load_zeta_coarse
    reconstructs it: wrap to (-k/2, k/2], then divide."""
    kg = np.asarray(kgrid, dtype=np.float64)
    idx = np.stack(np.meshgrid(*[np.arange(int(s)) for s in kgrid],
                               indexing="ij"), axis=-1).reshape(-1, 3)
    idx = idx.astype(np.float64)
    return np.where(idx > kg[None, :] / 2.0, idx - kg[None, :], idx) / kg[None, :]


SLAB_POLICY = {"sys_dim": "2", "mc_average_vcoul_body": "false",
               "bare_coulomb_cutoff": "30.0"}
BULK_POLICY = {"sys_dim": "3", "mc_average_vcoul_body": "true",
               "bare_coulomb_cutoff": "25.0"}


# --------------------------------------------------------------------------
# the served deck: nothing about a real slab may be refused
# --------------------------------------------------------------------------
def test_the_reference_slab_is_in_scope():
    """The MoS2 reference deck — the cell every production number in this
    module was measured on — must pass every condition, stamp included."""
    assert V.slab_scope_violations(_hex_slab_bvec(A_MOS2, C_MOS2),
                                   qfr=_planar_q(), policy=SLAB_POLICY) == []
    V.assert_slab_scope(_hex_slab_bvec(A_MOS2, C_MOS2), qfr=_planar_q(),
                        policy=SLAB_POLICY)


def test_an_unstamped_restart_is_judged_on_geometry_alone():
    """``policy=None`` is a restart written before the Coulomb stamp.  It must
    not become a refusal by itself — the geometry conditions need no stamp,
    and they are the ones that cannot be worked around."""
    assert V.slab_scope_violations(_hex_slab_bvec(A_MOS2, C_MOS2),
                                   qfr=_planar_q(), policy=None) == []


def test_the_q_grid_is_optional_so_early_callers_still_get_the_axis_check():
    """``qfr=None`` skips only the q-grid condition; the axis one still runs,
    so a caller holding geometry but no q list is not silently unguarded."""
    assert V.slab_scope_violations(_hex_slab_bvec(A_MOS2, C_MOS2)) == []
    why = V.slab_scope_violations(_fcc_bvec(A_SI))
    assert len(why) == 1 and "slab_axes_offdiag" in why[0]


# --------------------------------------------------------------------------
# the refused deck: si_bse_debug, and each condition separately
# --------------------------------------------------------------------------
def test_si_bse_debug_is_refused_for_all_three_reasons():
    """Row 23's deck.  fcc axes, a q_z != 0 grid, and a sys_dim=3 stamp — the
    three are independent, and all three are true of it at once."""
    why = V.slab_scope_violations(_fcc_bvec(A_SI), qfr=_bgw_wrapped_q(),
                                  policy=BULK_POLICY)
    assert len(why) == 3
    assert any("sys_dim=3" in w for w in why)
    assert any("slab_axes_offdiag" in w for w in why)
    assert any("q_z" in w for w in why)


def test_the_fcc_axis_ratio_is_exactly_one_not_a_tightenable_tolerance():
    """``slab_axes_offdiag`` = 1.000e+00 on si_bse_debug is not a small
    number that a looser gate could absorb: on a cubic reciprocal lattice
    ``b3``'s in-plane components equal its own z-component by construction.
    This is the cell that makes 'relax the tolerance' unavailable."""
    b = _fcc_bvec(A_SI)
    ratio = max(np.max(np.abs(b[2, :2])), np.max(np.abs(b[:2, 2]))) \
        / abs(b[2, 2])
    assert ratio == pytest.approx(1.0, abs=1e-14)


def test_a_bulk_stamp_alone_refuses_even_on_slab_separable_axes():
    """The kernel condition is INDEPENDENT of the geometry one.  A tetragonal
    cell with b3 || z and a planar grid still stored its V_qmunu under the
    bulk kernel, and this module has no bulk kernel to match it with — the
    3.218e-01 -> 4.593e-02 half of the row-23 measurement."""
    b = np.diag([0.9, 0.9, 0.3])
    assert V.slab_scope_violations(b, qfr=_planar_q(), policy=SLAB_POLICY) == []
    why = V.slab_scope_violations(b, qfr=_planar_q(), policy=BULK_POLICY)
    assert len(why) == 1 and "sys_dim=3" in why[0]


def test_a_nonplanar_q_grid_alone_refuses_on_slab_separable_axes():
    """And so is the q-grid condition: b3 || z fixes K_z per channel only
    when q_z = 0 as well."""
    b = np.diag([0.9, 0.9, 0.3])
    why = V.slab_scope_violations(b, qfr=_bgw_wrapped_q(), policy=SLAB_POLICY)
    assert len(why) == 1 and "q_z" in why[0]


# --------------------------------------------------------------------------
# the refusal itself
# --------------------------------------------------------------------------
def test_the_refusal_names_the_reasons_and_the_modes_that_do_serve():
    """A refusal that does not say what to do instead is a dead end.  Both
    working modes have to be in the message, and so does the punch row that
    is this refusal's measured history."""
    with pytest.raises(ValueError) as e:
        V.assert_slab_scope(_fcc_bvec(A_SI), qfr=_bgw_wrapped_q(),
                            policy=BULK_POLICY, source="isdf_tensors_936.h5")
    msg = str(e.value)
    assert "isdf_tensors_936.h5" in msg
    assert "--vq-mode ongrid" in msg and "--vq-mode refit" in msg
    assert "row 23" in msg
    for frag in ("sys_dim=3", "slab_axes_offdiag", "q_z"):
        assert frag in msg


def test_an_in_scope_deck_raises_nothing():
    V.assert_slab_scope(_hex_slab_bvec(A_MOS2, C_MOS2), qfr=_planar_q(),
                        policy=SLAB_POLICY, source="ok.h5")


def test_the_scope_check_is_wired_into_the_loader():
    """One site, and it is the loader's — not a thing each caller remembers.
    An AST check rather than a run, because running it needs the fixture this
    file exists to avoid."""
    import ast
    import inspect
    src = inspect.getsource(V.load_zeta_coarse)
    tree = ast.parse(src.lstrip())
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "assert_slab_scope" in called, (
        "load_zeta_coarse no longer applies the slab-scope refusal; every "
        "vq_interp entry point reaches the model through it")


def test_a_cell_with_no_z_axis_at_all_is_named_not_divided_by():
    """``b3_z`` is the denominator of both ``slab_axes_offdiag`` and the
    truncation length ``z_c = pi/b3_z``.  A cell whose b3 lies in the plane has
    neither, and must come back as a sentence rather than as a division."""
    b = np.array([[0.9, 0., 0.], [0., 0.9, 0.], [0., 0.3, 0.]])
    why = V.slab_scope_violations(b, qfr=_planar_q(), policy=SLAB_POLICY)
    assert len(why) == 1 and "no z-projection" in why[0]
