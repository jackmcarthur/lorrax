"""The vcoul compat shims re-export the door's OBJECTS, not copies.

The extraction's promise is that every old import path answers with the
same object the door answers with — `is`-identity, not equality — so a
consumer on an old path and a consumer on the door can never diverge.
These cells pin that wiring for every shim, which is what lets the rest
of the suite exercise old paths as standing shim coverage without each
file re-proving the identity.  (Cross-branch ruling: the shims stay for
the whole wave, so this pin has a whole-wave lifetime.)

The kernel classes are the one deliberate non-identity: the old-path
classes are (wfn, meta)-facing adapter SUBCLASSES of the service kernels.
For those the pin is the subclass relationship plus the arithmetic method
being inherited unmodified — the adapter may translate, it may not
re-implement.
"""
# Bootstrap first, door second — the same order rule the audit arm proved
# load-bearing at gw/v_q_g_flat.py and scripts/checks (D1/D2): nothing
# guarantees another test module has put services/*/src on sys.path yet.
from ffi import _services
_services.ensure_on_path()
import vcoul  # noqa: E402


def test_function_reexports_are_the_door_objects():
    import importlib

    import gw.compute_vcoul as gcv
    import gw.compute_vcoul_0d as gcv0
    import gw.vcoul as gv
    import gw.coulomb.base as gcb
    # file_io/__init__ re-exports the FUNCTION read_bgw_vcoul, shadowing
    # the submodule attribute of the same name — go through importlib.
    frb = importlib.import_module("file_io.read_bgw_vcoul")

    assert gcv.build_v_head_miniBZ_fn_3d is vcoul.build_v_head_miniBZ_fn_3d
    assert gcv.build_miniBZ_dq_cart is vcoul.build_miniBZ_dq_cart
    assert gv.wrap_points_to_voronoi is vcoul.wrap_points_to_voronoi
    assert gcb.minibz_voronoi_batches is vcoul.minibz_voronoi_batches
    assert gcb.minibz_inscribed_sphere_r2 is vcoul.minibz_inscribed_sphere_r2
    assert gcb.minibz_average is vcoul.minibz_average
    assert gcb._minibz_kernel_bare is vcoul._minibz_kernel_bare
    assert gcv0.compute_vcoul_box is vcoul.compute_vcoul_box
    assert gcv0._round_up_fft_size is vcoul._round_up_fft_size
    assert frb.read_bgw_vcoul is vcoul.read_bgw_vcoul
    assert frb.fill_v_grid_for_q is vcoul.fill_v_grid_for_q
    assert frb.fill_v_sphere_for_q is vcoul.fill_v_sphere_for_q
    assert frb.BGWVcoulTable is vcoul.BGWVcoulTable
    assert gcb.SysDim is vcoul.SysDim


def test_kernel_adapters_subclass_and_inherit_the_arithmetic():
    """Adapters translate signatures; they must not re-implement physics."""
    import gw.coulomb as gc

    for old, new in ((gc.Bulk3D, vcoul.Bulk3D),
                     (gc.Slab2D, vcoul.Slab2D),
                     (gc.Box0D, vcoul.Box0D)):
        assert issubclass(old, new)
        # The arithmetic method is inherited, not overridden — a shim that
        # re-implements _v_bare_per_q is a second physics copy, the exact
        # failure mode the extraction exists to end.
        assert old._v_bare_per_q is new._v_bare_per_q


def test_identity_pin_can_fail():
    """Red twin: a genuinely different function object is not `is`-equal,
    so the identity assertions above are falsifiable, not tautological."""
    def imposter(*a, **k):
        raise AssertionError("unreachable")
    assert imposter is not vcoul.build_v_head_miniBZ_fn_3d
