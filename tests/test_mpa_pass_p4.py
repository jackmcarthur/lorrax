"""The MPA Σ pass on more than one process, and on more than one device.

WHAT BROKE (C1, found 2026-08-10 by the sixteen-GPU baseline lane running
the four-GPU rule).  ``scripts/perf_mpa16/go_pass_mesh.sh 6 <tag> 4`` —
pole 6 of the Si production deck, ``-G=4 -n=4``, a 2×2 mesh over four
PROCESSES — died on all four ranks 59 s in, in the first τ dispatch:

    TypeError: cannot reshape array of shape (4, 64, 100) (size 25600)
               into shape (64, 100) (size 6400)
      gw/greens_function_kernel.py:61   mask = jnp.reshape(mask, enk.shape)
      gw/mpa/sigma_pass.py:1218         run_pass_branch

``(64, 100)`` is ``(n_k, n_band)`` and the leading ``4`` is
``jax.process_count()``.  The pass loop gathers its A-side operands to host
at their source shape (``sigma_pass._host_at_source_shape``) and then used
to hand them back DOWN to ``ppm_windows._build_windows_for_branch`` as
``jnp.asarray(host_array)`` — a process-local, fully addressable device
array — whose own ``process_allgather(tiled=False)`` re-invented the axis
at its true length.  At one process that length is 1 and ``build_G_tau``'s
reshape absorbed it silently; at four it is 4 and the reshape raises.  A
one-by-one mesh had been hiding a process-axis bug by making it look like a
spin axis, which is exactly what the comment at the reshape used to claim.

THE TWO CELLS BELOW ARE NOT INTERCHANGEABLE, and the difference is the
structural gap this defect exposes in the ≥4-GPU rule:

  * :func:`test_the_pass_loop_survives_a_four_process_gather` is the cell
    that WOULD HAVE CAUGHT IT.  The axis is a function of
    ``jax.process_count()``, and pytest is ONE process however many devices
    it can see — so no in-suite cell can produce the axis natively.  This
    one installs the four-process ``process_allgather`` semantics verbatim
    and drives the production planner and the production consumer through
    them.  It needs no devices and runs in the ordinary census.

  * :func:`test_mpa_pass_branch_on_a_2x2_mesh_matches_one_device` is the
    real 2×2 mesh pass leg, asserting the pass's own answer against the
    single-shard one.  It carries the ``mesh`` marker so it RUNS on a
    ≥4-device allocation instead of skipping.  It could not have caught
    C1 by itself — four devices in one process is not four processes —
    and saying so is the point: a four-DEVICE gate is not a four-PROCESS
    gate, and only a multi-rank driver leg exercises the gather natively.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw import ppm_windows as PW
from gw.mpa import sigma_pass as SP


RYD = 13.605693122994

# The four-process marker.  ``mesh`` is the suite marker the conftest lane
# is adding on fix/conftest-mesh-cells-2026-08-10 so that sharded cells run
# under the suite on a ≥4-device allocation rather than skipping — the two
# sharded cells in tests/test_ppm_crossing_completion.py are its acceptance
# case and this one joins them.  The device-count skipif below is kept so
# the cell is honest on a one-GPU box either way; when that conftest lands
# it is the marker that gets the cell four devices to find.
needs_mesh = pytest.mark.skipif(
    jax.device_count() < 4,
    reason="needs >=4 devices: XLA_FLAGS=--xla_force_host_platform_device_count=4")


# ---------------------------------------------------------------------------
#  A small pass problem, at production SHAPES rather than production size.
# ---------------------------------------------------------------------------

NKX, NKY, NKZ = 4, 4, 4
NK = NKX * NKY * NKZ          # = n_q as well: the pass integrates over all k
# THE K-GRID IS THE PRODUCTION ONE ON PURPOSE.  Everything else here is
# shrunk to suite size, but the flat-k FFT goes through the cuFFT strided
# FFI handler -- which is the REQUIRED layer (`LORRAX_FFT_FFI=0` is a
# refusal: there is nothing to opt out to) and which fails to plan a toy
# transform: measured CUFFT_EXEC_FAILED on an A100 at 2x2x1 and at 2x2x2,
# in both the fused and the decomposed modes, and the failure is a process
# ABORT rather than an exception a cell could catch.  4x4x4 is the deck's
# own grid and is served.  Shrink the centroids and the bands, never the
# k-grid.
NS = 1                        # spinor axis, replicated on the mesh
NMU = 8                       # ISDF centroids — divisible by both mesh axes
NB_FULL = 6                   # the A-space band extent
NB_PROJ = 4                   # the Σ window — m_pad / n_pad, mesh-divisible


def _operands(seed=11, gamma_scale=0.02):
    """E_A / masks / pole field / ψ for a whole pass branch, host side.

    ``gamma_scale`` sets the pole widths against the smearing ``xi`` the
    planner is given: the default puts every pole BELOW it, so the whole
    field routes through the legacy group -- which is the group that
    re-enters ``_build_windows_for_branch`` and therefore the one that
    carried C1.  A larger scale straddles ``xi`` and produces the MPA
    families beside it.
    """
    rng = np.random.default_rng(seed)

    # E_A ≥ 0 (energy above the Fermi level) with a genuine spread, so the
    # planner's core/stripe/slab split has something to split on.
    E_A = np.abs(rng.random((NK, NB_FULL))) * 1.5 + 0.05
    mask_A = np.ones((NK, NB_FULL), dtype=bool)
    mask_A[:, -1] = False              # one dead band, so the mask is not trivial

    # The pole field: Re Ω spread across the crossing threshold, Γ small
    # enough that some poles route legacy and some route MPA.
    a = np.abs(rng.random((NK, NMU, NMU))) * 2.0 + 0.05
    g = np.abs(rng.random((NK, NMU, NMU))) * gamma_scale + 1.0e-4
    live = np.ones(a.shape, dtype=bool)
    B = (rng.random(a.shape) + 1j * rng.random(a.shape)).astype(np.complex128)

    omega_ry = np.array([0.0, 0.05, 0.11], dtype=np.float64)

    psi_coh_xn = (rng.random((NK, NS, NMU, NB_FULL))
                  + 1j * rng.random((NK, NS, NMU, NB_FULL))).astype(np.complex128)
    psi_coh_yr = (rng.random((NK, NB_FULL, NS, NMU))
                  + 1j * rng.random((NK, NB_FULL, NS, NMU))).astype(np.complex128)
    psi_proj_xr = (rng.random((NK, NB_PROJ, NS, NMU))
                   + 1j * rng.random((NK, NB_PROJ, NS, NMU))).astype(np.complex128)
    psi_proj_yn = (rng.random((NK, NS, NMU, NB_PROJ))
                   + 1j * rng.random((NK, NS, NMU, NB_PROJ))).astype(np.complex128)

    return dict(E_A=E_A, mask_A=mask_A, a=a, g=g, live=live, B=B,
                omega_ry=omega_ry,
                psi=(psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn))


def _plan(op, *, space="cond", neg=False):
    """The production planner, on the small problem."""
    return SP.plan_branch_groups(
        a_ry=op["a"], gamma_ry=op["g"], live_mask=op["live"],
        E_A_host=op["E_A"], base_mask_A_host=op["mask_A"],
        omega_nonneg_ry=op["omega_ry"], space=space, neg_omega_half=neg,
        xi_ry=0.5 / RYD, edge_factor=1.5, rel_tol=1.0e-8,
        target_error=1.0e-6, laplace_max_nodes=32,
        crossing_eps_q=1.0e-10, crossing_max_nodes=200,
        use_shipped_minimax_tables=True,
        log_tag="", print_fn=lambda *a, **k: None)


# ---------------------------------------------------------------------------
#  THE RED TWIN — the four-process gather, installed rather than allocated.
# ---------------------------------------------------------------------------

def _four_process_allgather(a, dtype=np.complex128, *, tiled=False):
    """``ppm_windows._to_host_np`` as it behaves at ``process_count() == 4``.

    Verbatim semantics of ``jax.experimental.multihost_utils.
    process_allgather``: a FULLY ADDRESSABLE operand is stacked across the
    processes, so a ``tiled=False`` gather comes back with a leading axis of
    length four; a globally-sharded (non-fully-addressable) operand cannot be
    gathered that way at all, so ``_to_host_np`` forces ``tiled=True`` and
    gets the reconstructed global array at its own shape.  In this process
    every array is addressable, which is the whole reason the axis has to be
    installed rather than allocated.
    """
    if not tiled and getattr(a, "is_fully_addressable", True):
        return np.stack([np.asarray(a, dtype=dtype)] * 4, axis=0)
    return np.asarray(a, dtype=dtype)


def test_the_pass_loop_survives_a_four_process_gather(monkeypatch):
    """No window the pass builds may carry the process axis.  C1's red twin.

    Pre-fix this cell reproduces the production failure exactly: the legacy
    group's windows come back with ``mask_A`` of shape ``(4, n_k, n_b)`` and
    the real consumer raises the production ``TypeError``.  Post-fix the
    A-side operands never go back onto a device, so nothing gathers them a
    second time and every window's selector has ``E_A``'s own shape.
    """
    from gw.greens_function_kernel import build_G_tau

    monkeypatch.setattr(PW, "_to_host_np", _four_process_allgather)

    # Two fields, so BOTH window families are scored.  The narrow one routes
    # entirely through the legacy group -- the group that re-enters
    # _build_windows_for_branch, and the one that broke.  The wide one is
    # planned on the sign-definite branch ("val" on the +omega half), which
    # is where the MPA families build their own windows from
    # base_mask_A_host directly.
    narrow = _operands()
    wide = _operands(gamma_scale=1.0)
    plans = [("narrow/cond", narrow, _plan(narrow)[0]),
             ("wide/val", wide, _plan(wide, space="val")[0])]

    names = {tag: [g.name for g in gs] for tag, _op, gs in plans}
    assert any(n == "legacy" for n in names["narrow/cond"]), names
    assert any(n.startswith("mpa[") for n in names["wide/val"]), names

    for tag, op, groups in plans:
        assert groups, f"{tag}: the planner produced no groups to check"
        want = op["E_A"].shape
        for grp in groups:
            for win in grp.windows:
                assert np.shape(win.mask_A) == want, (
                    f"{tag} {grp.name}:{win.name} mask_A is "
                    f"{np.shape(win.mask_A)}, not {want} — a gather has put "
                    f"the process axis back on it")

        # And the production consumer, which is where it actually died.
        xn, yr, _xr, _yn = op["psi"]
        enk = jnp.asarray(op["E_A"])
        for grp in groups:
            for win in grp.windows:
                G = build_G_tau(jnp.asarray(xn), jnp.asarray(yr), enk,
                                1j * 0.25, e_ref=float(win.E_ref_A),
                                mask=jnp.asarray(win.mask_A))
                assert G.shape == (NK, NS, NMU, NS, NMU)


def test_the_installed_gather_really_does_prepend_the_process_axis():
    """The red twin's own instrument, scored — else it proves nothing.

    If ``_four_process_allgather`` did not actually reproduce the axis, the
    cell above would pass at any commit and mean nothing.  This is the
    FALSE case: on a device array it prepends four, and on a host-side
    array that never went back to a device there is nothing to gather.
    """
    src = jnp.asarray(np.zeros((3, 5), dtype=np.float64))
    assert _four_process_allgather(src, np.float64, tiled=False).shape == (4, 3, 5)
    assert _four_process_allgather(src, np.float64, tiled=True).shape == (3, 5)
    assert PW._already_on_host(np.zeros((3, 5), dtype=bool), bool).shape == (3, 5)


# ---------------------------------------------------------------------------
#  THE MESH LEG — the pass's own answer, 2×2 against one shard.
# ---------------------------------------------------------------------------

def _run_branch(op, groups, mesh):
    """``run_pass_branch`` on ``mesh``, assembled back to the global cube."""
    from gw.ppm_tau_kernel import _get_sigma_tau_kernel

    xn, yr, xr, yn = op["psi"]
    kgrid = (NKX, NKY, NKZ)
    with mesh:
        psi = (jax.device_put(xn, NamedSharding(mesh, P(None, None, 'x', None))),
               jax.device_put(yr, NamedSharding(mesh, P(None, None, None, 'y'))),
               jax.device_put(xr, NamedSharding(mesh, P(None, None, None, 'x'))),
               jax.device_put(yn, NamedSharding(mesh, P(None, None, 'y', None))),
               NK, NB_PROJ)
        kernels = (_get_sigma_tau_kernel(mesh_xy=mesh, kgrid=kgrid),
                   _get_sigma_tau_kernel(mesh_xy=mesh, kgrid=kgrid,
                                         merged_x=True))
        tiles = SP.run_pass_branch(
            groups=groups, omega_nonneg_ry=op["omega_ry"],
            E_A=jnp.asarray(op["E_A"]),
            B_p=jnp.asarray(op["B"], dtype=jnp.complex128),
            psi=psi, tau_kernels=kernels, mesh_xy=mesh,
            log_tag="", print_fn=lambda *a, **k: None)
    assert tiles is not None, "the pass branch produced no tiles"
    full = np.zeros((op["omega_ry"].size, NK, NB_PROJ, NB_PROJ),
                    dtype=np.complex128)
    for t, ix in zip(tiles.tiles, tiles.tile_index):
        full[ix] = t
    return full


@pytest.mark.mesh
@needs_mesh
def test_mpa_pass_branch_on_a_2x2_mesh_matches_one_device():
    """The pass's answer must not depend on the mesh it was computed on.

    The sink is declared ``P(None, None, 'x', 'y')``, so the band axes shard
    the moment the pass is given a mesh and the ψ-projection tail becomes a
    reduce-scatter over four shards instead of one local dot.  That is a
    re-association of the same sum, so the two answers agree to floating
    point and not bit-for-bit; anything larger than that is a real finding
    about the sharded path and not a rounding difference.

    NOTE ON WHAT THIS CELL DOES NOT COVER.  pytest runs in ONE process, so
    a 2×2 mesh here spans four devices of one process and
    ``jax.process_count()`` is 1.  C1 was a process-count defect and this
    cell would not have caught it — see
    :func:`test_the_pass_loop_survives_a_four_process_gather`, which does.
    """
    devs = jax.devices()[:4]
    mesh4 = Mesh(np.asarray(devs).reshape(2, 2), ('x', 'y'))
    mesh1 = Mesh(np.asarray(devs[:1]).reshape(1, 1), ('x', 'y'))

    op = _operands()
    groups, _ = _plan(op)

    got4 = _run_branch(op, groups, mesh4)
    got1 = _run_branch(op, groups, mesh1)

    scale = float(np.max(np.abs(got1))) or 1.0
    delta = float(np.max(np.abs(got4 - got1)))
    assert delta / scale < 1.0e-12, (
        f"2x2 mesh and one device disagree by {delta:.3e} "
        f"({delta / scale:.3e} relative) — larger than the reduce-scatter "
        f"re-association this comparison allows")


@pytest.mark.mesh
@needs_mesh
def test_the_pass_sink_shards_the_band_axes_it_says_it_does():
    """The premise of the cell above: on 2×2 there really are four shards.

    If the pass silently ran replicated on a mesh, the comparison would be
    trivially green and would score nothing.  Four devices, four tiles, and
    each tile a quarter of the band block.
    """
    devs = jax.devices()[:4]
    mesh4 = Mesh(np.asarray(devs).reshape(2, 2), ('x', 'y'))
    op = _operands()
    groups, _ = _plan(op)

    from gw.ppm_tau_kernel import _get_sigma_tau_kernel
    xn, yr, xr, yn = op["psi"]
    with mesh4:
        psi = (jax.device_put(xn, NamedSharding(mesh4, P(None, None, 'x', None))),
               jax.device_put(yr, NamedSharding(mesh4, P(None, None, None, 'y'))),
               jax.device_put(xr, NamedSharding(mesh4, P(None, None, None, 'x'))),
               jax.device_put(yn, NamedSharding(mesh4, P(None, None, 'y', None))),
               NK, NB_PROJ)
        kernels = (_get_sigma_tau_kernel(mesh_xy=mesh4, kgrid=(NKX, NKY, NKZ)),
                   _get_sigma_tau_kernel(mesh_xy=mesh4, kgrid=(NKX, NKY, NKZ),
                                         merged_x=True))
        tiles = SP.run_pass_branch(
            groups=groups, omega_nonneg_ry=op["omega_ry"],
            E_A=jnp.asarray(op["E_A"]),
            B_p=jnp.asarray(op["B"], dtype=jnp.complex128),
            psi=psi, tau_kernels=kernels, mesh_xy=mesh4,
            log_tag="", print_fn=lambda *a, **k: None)
    assert len(tiles.tiles) == 4, [np.shape(t) for t in tiles.tiles]
    for t in tiles.tiles:
        assert np.shape(t)[-2:] == (NB_PROJ // 2, NB_PROJ // 2), np.shape(t)
