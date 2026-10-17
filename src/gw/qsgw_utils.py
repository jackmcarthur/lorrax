"""QSGW Σ_xc build, the one-shot update_H seam, and SC-COHSEX diagnostics.

The post-self-energy plumbing in ``gw_jax`` accepts replicated or
band-sharded ``(nk, nb, nb)`` arrays.  The dynamic correlation's on-shell
pair ``(2, k, m_X, n_Y)`` and any matrix derived from it retain their band
tiles; diagonal-only operations extract and return only the bounded
``(nk, nb)`` diagonal.  Everything in this module is structured around that
seam:

- :func:`solve_qp` is the one-shot pass-through ``Σ_xc + V_H``; no QP
  equation is solved anywhere (Σ is read at the present energies).
- :func:`build_qsgw_sigma_xc` is a JIT'd JAX kernel that takes the
  on-device sharded on-shell pair ``Σ_mn(E_m)``, ``Σ_mn(E_n)``
  (``ppm_sigma.OnShellSigma``) and returns the Hermitised QSGW Σ_xc in the
  selected replicated or two-axis band-sharded layout.
"""

from __future__ import annotations


import numpy as np
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.units import RYD_TO_EV


def omega_coverage(omega_grid: np.ndarray,
                   eval_kn: np.ndarray) -> tuple[np.ndarray, int, float]:
    """``(covered_mask, n_uncovered, fraction_uncovered)`` for an ω request.

    ``covered_mask[k, n]`` is True where ``eval_kn[k, n]`` lies inside
    ``[ω_min, ω_max]``, the sampled range Σ(E) is read in rather than
    replaced by Σ(ω = 0).  Interior
    holes in a patched grid are a different question and are owned by
    :func:`assert_omega_grid_covers`; this answers the outer one.
    """
    omega = np.asarray(omega_grid, dtype=np.float64)
    arr = np.asarray(eval_kn, dtype=np.float64)
    covered = (arr >= float(omega[0])) & (arr <= float(omega[-1]))
    n_out = int(np.count_nonzero(~covered))
    frac = float(n_out) / float(arr.size) if arr.size else 0.0
    return covered, n_out, frac


def sigma_eval_omega(omega_grid: np.ndarray, eval_kn: np.ndarray,
                     policy: str) -> tuple[np.ndarray, np.ndarray]:
    """``(omega_read_kn, covered_kn)``: where each QSGW Sigma(E) evaluation
    reads, under ``sigma_out_of_grid`` (docs/self_consistency.md §4).

    The coverage is :func:`omega_coverage`'s, the one classification the Sigma
    build, the SC grid growth and the sum-band tail mask all read.  A covered
    energy reads itself.  An uncovered one reads the nearest grid edge under
    ``clamp`` and omega = 0 under ``static``; under ``cover`` the SC growth has
    already covered every protected identity, so what remains uncovered is a
    frozen-core band (decoupled from H) and reads omega = 0.
    """
    covered = omega_coverage(omega_grid, eval_kn)[0]
    omega = np.asarray(omega_grid, dtype=np.float64)
    e = np.asarray(eval_kn, dtype=np.float64)
    if policy == "clamp":
        return np.clip(e, float(omega[0]), float(omega[-1])), covered
    if policy in ("cover", "static"):
        return np.where(covered, e, 0.0), covered
    raise ValueError(f"sigma_out_of_grid must be cover, clamp or static; got {policy!r}")




def sigma_grid_edge_ambiguity(
    at_zero_kn_ev: np.ndarray,
    at_edge_kn_ev: np.ndarray,
    omega_range_ev,
    e_kn_ev: np.ndarray,
    growth_window_ev: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """States whose QSGW diagonal fixed point is not unique at a grid edge.

    The out-of-grid rule above (and :func:`build_qsgw_sigma_xc`) makes the
    QSGW map DISCONTINUOUS at each sampled-grid edge ω_e: a state inside the
    grid takes Σ_c,nn(E_n), a state outside takes Σ_c,nn(0), so its diagonal
    jumps by

        Δ_n = Re Σ_c,nn(0) − Re Σ_c,nn(ω_e)      (ω_e = the nearer edge).

    In the diagonal model E = A + Σ(E), when the jump carries a state OUTWARD
    (Δ_n > 0 at the top edge, Δ_n < 0 at the bottom) a state within |Δ_n| of
    the edge, on either side, has a self-consistent partner on the other
    side: two fixed points, and the one the loop reaches depends on its path.
    That is what is flagged.  An inward jump leaves no fixed point within
    |Δ_n| outside the edge (such a state maps back inside), so a converged
    state there cannot occur and an inside state is unique.  Measured: Fe 4^3 bispinor H-point states at
    E-μ = +9.8 (inside, Σ(E)) and +11.7 eV (outside, Σ(0)) around a +10 eV
    edge, Δ = 1.87 eV, one branch per accelerator (sandbox claim 2688).

    No threshold: the band is the jump itself.  Parameters are the diagonal
    Σ_c,nn(0) and Σ_c,nn(ω_e) ``(nk, nb)`` (eV, complex or real; the on-shell
    ``zero`` and ``edge`` tables, ``ppm_sigma.on_shell_request``), the sampled
    range ``(ω_min, ω_max)`` and the evaluation energies ``(nk, nb)``, all on
    the same E_F-relative reference.  Returns ``(ambiguous_kn, jump_kn_ev)``.

    ``growth_window_ev`` is the SC loop's padded window (``scissor.
    sc_padded_window_ev``): a state inside it GROWS the grid instead of
    leaving it, so the Σ(E)/Σ(0) switch sits at the outer of the grid edge
    and the window edge.  Where the window reaches past the grid, Σ there is
    unsampled and the sampled edge value stands in for it (an estimate).
    Fe 4^3 at +28 eV: without this, 12 states near the +28 grid edge were
    flagged at map 0 although the grid would have grown under them.
    """
    e = np.asarray(e_kn_ev, dtype=np.float64)
    at_zero, at_edge = (np.real(np.asarray(a)) for a in (at_zero_kn_ev, at_edge_kn_ev))
    if at_zero.shape != e.shape or at_edge.shape != e.shape:
        raise ValueError(
            f"sigma_grid_edge_ambiguity: Sigma diagonals {at_zero.shape}/{at_edge.shape} "
            f"do not match the energies {e.shape}")
    lo, hi = float(omega_range_ev[0]), float(omega_range_ev[-1])
    if growth_window_ev is not None:
        lo, hi = min(lo, float(growth_window_ev[0])), max(hi, float(growth_window_ev[1]))
    use_top = np.abs(e - hi) <= np.abs(e - lo)
    edge = np.where(use_top, hi, lo)
    jump = at_zero - at_edge
    outward = np.where(use_top, jump > 0.0, jump < 0.0)
    ambiguous = outward & (np.abs(e - edge) < np.abs(jump))
    return ambiguous, jump


# ---------------------------------------------------------------------------
# ω-grid hole refusal for solved QP energies (host NumPy)
# ---------------------------------------------------------------------------

def assert_omega_grid_covers(E_kn_ry, in_grid_mask, omega_grid_ry, *,
                             context):
    """Refuse solved QP energies inside a hole of a patched ω grid.

    A patched grid (``sigma_omega_patches_ev``) has interior gaps by
    design — that is what makes the MPA crossing rule's cost independent
    of the dynamic range.  The Σ(ω)→E piecewise-linear interpolation is
    silent about a hole: an energy in the gap would be interpolated
    across it and come back plausible-looking and wrong.  So a hole is
    detected from the grid itself (a step above 3× the median step) and
    an in-grid-classified energy strictly inside one — more than one
    median step from both hole edges — is a refusal that names the
    energy and the fix (widen or add a patch).  Contiguous grids have
    no holes and return immediately.
    """
    omega = np.asarray(omega_grid_ry, dtype=np.float64)
    if omega.size < 2:
        return
    steps = np.diff(omega)
    ref = float(np.median(steps))
    holes = np.nonzero(steps > 3.0 * ref)[0]
    if holes.size == 0:
        return
    E = np.asarray(E_kn_ry, dtype=np.float64)
    mask = np.asarray(in_grid_mask, dtype=bool)
    for i in holes:
        lo, hi = float(omega[i] + ref), float(omega[i + 1] - ref)
        inside = mask & (E > lo) & (E < hi)
        if np.any(inside):
            worst = float(E[inside].flat[0])
            raise ValueError(
                f"GATE omega_grid_hole ({context}): "
                f"{int(np.sum(inside))} in-grid QP energies sit inside "
                f"the ω-grid hole ({lo * RYD_TO_EV:.2f}, "
                f"{hi * RYD_TO_EV:.2f}) eV — e.g. "
                f"{worst * RYD_TO_EV:.3f} eV — where Σ(ω) would be "
                "interpolated across the gap.  FALSE case: every solved "
                "QP energy lies on a grid patch.  Widen the nearest "
                "sigma_omega_patches_ev patch (QP energies drift between "
                "QSGW iterations; leave headroom).")


# ---------------------------------------------------------------------------
# Static-operator diagonal extraction from a band-sharded (k, m_X, n_Y)
# ---------------------------------------------------------------------------

def _is_band_sharded_static(a) -> bool:
    """True when a static ``(nk,nb,nb)`` operator partitions a band axis."""
    spec = getattr(getattr(a, "sharding", None), "spec", None)
    if spec is None or getattr(a, "ndim", 0) != 3:
        return False
    spec = tuple(spec) + (None,) * (3 - len(tuple(spec)))
    return spec[1] is not None or spec[2] is not None


# Module-scope kernel cache, one per mesh (NOT a closure inside the caller),
# so SC iterations hit the pjit cache instead of retracing a fresh function.
_EXTRACT_DIAG_SHARDED_KERNEL_CACHE: dict[int, object] = {}


def _extract_diag_sharded_kernel(mesh_xy: Mesh):
    """Diagonal of a P(None,None,'x','y')-sharded ``(lead, k, nb, nb)`` array
    WITHOUT materializing it on any device.

    Structural (inside shard_map, where the partitioner cannot hoist a
    gather — QUALITY_PATTERNS #4): each shard extracts the global-diagonal
    elements it owns (the (m, n) intersection with the diagonal), everything
    else contributes exact zeros, and ONE psum over both mesh axes
    replicates the (lead, nk, nb) diagonal (nb² → nb vs the full gather).  Every diagonal element is
    owned by exactly one shard, so the psum adds a value to exact zeros —
    movement-only up to IEEE ``x + 0.0`` (bit-exact for every x ≠ -0.0).

    Requires a SQUARE global band extent with both axes divisible by their
    mesh axis (m/p_x, n/p_y integral) — guaranteed by the sharded layout's
    resolve-time divisibility refusal (no mesh pad reaches this path).
    """
    key = id(mesh_xy)
    fn = _EXTRACT_DIAG_SHARDED_KERNEL_CACHE.get(key)
    if fn is None:
        from functools import partial
        from jax import shard_map

        p_x = int(mesh_xy.shape['x'])

        @jax.jit
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=P(None, None, 'x', 'y'),
                 out_specs=P(None, None, None),
                 check_vma=False)
        def _diag_sharded(tile):
            ix = jax.lax.axis_index('x')
            iy = jax.lax.axis_index('y')
            mb = tile.shape[2]
            nbl = tile.shape[3]
            nb = mb * p_x                       # square global extent
            i = jnp.arange(nb)
            a = i - ix * mb
            b = i - iy * nbl
            own = (a >= 0) & (a < mb) & (b >= 0) & (b < nbl)
            a_c = jnp.clip(a, 0, mb - 1)
            b_c = jnp.clip(b, 0, nbl - 1)
            vals = tile[:, :, a_c, b_c]         # (nω, nk, nb) local gather
            vals = jnp.where(own[None, None, :], vals,
                             jnp.zeros((), dtype=tile.dtype))
            return jax.lax.psum(vals, axis_name=('x', 'y'))  # exact-reduce: one owner per diagonal element

        fn = _diag_sharded
        _EXTRACT_DIAG_SHARDED_KERNEL_CACHE[key] = fn
    return fn


def static_sigma_diag_to_host(sigma_knn, mesh_xy: Mesh) -> np.ndarray:
    """Return only a static operator's bounded ``(nk,nb)`` diagonal.

    A canonical ``P(None,'x','y')`` input stays tiled: it is promoted by a
    length-one leading axis and uses the sharded diagonal psum.  A
    replicated/local input retains the historical host diagonal.  Any other
    matrix-axis partition refuses instead of causing an implicit all-gather.
    """
    from common.collectives import gather_to_host

    arr = jnp.asarray(sigma_knn)
    if arr.ndim != 3 or int(arr.shape[1]) != int(arr.shape[2]):
        raise ValueError(
            "static Sigma diagonal extraction requires (nk,nb,nb); got "
            f"{arr.shape}")
    spec = getattr(getattr(arr, "sharding", None), "spec", None)
    if spec is not None:
        spec = tuple(spec) + (None,) * (3 - len(tuple(spec)))
        if spec[1] is not None or spec[2] is not None:
            if spec != (None, "x", "y"):
                raise ValueError(
                    "static Sigma diagonal extraction requires canonical "
                    "P(None,'x','y') band sharding; got "
                    f"PartitionSpec{spec}")
            arr4 = jax.lax.with_sharding_constraint(
                jnp.expand_dims(arr, axis=0),
                NamedSharding(mesh_xy, P(None, None, "x", "y")))
            return np.asarray(gather_to_host(
                _extract_diag_sharded_kernel(mesh_xy)(arr4)))[0]
    host = np.asarray(gather_to_host(arr))
    return np.diagonal(host, axis1=1, axis2=2).copy()


# ---------------------------------------------------------------------------
# QSGW Σ_xc build — the sharded on-shell pair → selected layout
# ---------------------------------------------------------------------------

def check_on_shell_reads(table_ev, want_ev, *, what):
    """``GATE on_shell_reads``: an on-shell table holds the energies a reader asks for.

    The on-shell Sigma was evaluated at energies fixed before its sweep
    (``ppm_sigma.on_shell_request``); a reader that now asks for others would
    get Sigma at the wrong energy with nothing to interpolate.  The tables
    round-trip eV -> Ry -> eV, hence the 1e-9 eV tolerance.
    """
    table, want = np.asarray(table_ev, np.float64), np.asarray(want_ev, np.float64)
    if table.shape != want.shape or not np.allclose(table, want, rtol=0.0, atol=1.0e-9):
        drift = (np.inf if table.shape != want.shape
                 else float(np.max(np.abs(table - want), initial=0.0)))
        raise ValueError(
            f"GATE on_shell_reads ({what}): the on-shell Sigma was evaluated at other "
            f"energies (max |dE| = {drift:.3e} eV, shapes {table.shape} vs {want.shape}); "
            "the read energies must be the ones given to the Sigma sweep")


@lru_cache(maxsize=8)
def _qsgw_on_shell_kernel(mesh_xy: Mesh, replicated_output: bool):
    out_3d = NamedSharding(
        mesh_xy, P(None, None, None) if replicated_output else P(None, "x", "y"))

    @jax.jit
    def _kernel(pair, sig_x):
        # The QSGW ansatz with the reads made by the sweep itself:
        # pair[0] = Sigma_mn(E_m), pair[1] = Sigma_mn(E_n).
        M = jax.lax.with_sharding_constraint(0.5 * (pair[0] + pair[1]) + sig_x, out_3d)
        Mh = 0.5 * (M + jnp.conj(jnp.swapaxes(M, -1, -2)))
        return jax.lax.with_sharding_constraint(Mh, out_3d)
    return _kernel


def build_qsgw_sigma_xc(
    sigma_c,
    sigma_x_kij_ry: jax.Array,
    omega_ev: np.ndarray,
    e_qp_kn_ev: np.ndarray,
    mesh_xy: Mesh,
    *,
    replicated_output: bool = True,
    band_axis=None,
    out_of_grid: str = "cover",
) -> tuple[jax.Array, dict[str, float]]:
    """Build the static Hermitian QSGW Σ_xc[k, m, n] from an on-shell Sigma.

    Implements the standard QSGW ansatz

        Σ_xc^QSGW_ij(k) = ½[ Σ_xc_ij(k, E_i(k)) + Σ_xc_ij(k, E_j(k)) ]ʰ

    where ``[·]ʰ`` denotes the Hermitian part.  Σ_xc = Σ_c + Σ_x; Σ_x is
    ω-independent so it is added once.  ``sigma_c`` is a
    ``ppm_sigma.OnShellSigma``: its pair holds Σ_mn at the read energies
    :func:`sigma_eval_omega` gives for ``e_qp_kn_ev`` on the sampled range
    ``omega_ev`` (both E_F-relative) under ``out_of_grid``, which is checked
    (``GATE on_shell_reads``) rather than assumed.  No interpolation.

    The result is replicated by default; ``replicated_output=False`` returns
    ``P(None,'x','y')`` and performs the Hermitian transpose collectively
    (the fixed-Sigma consumers that feed a distributed eigensolver).

    Returns ``(sigma_xc_qsgw_kij_ry, diagnostics)``; ``diagnostics`` has
    ``n_clipped`` (count of ``E_kn`` outside ``[ω_min, ω_max]``) and
    ``omega_min/max_ev``.
    """
    omega = np.asarray(omega_ev, dtype=np.float64)
    E = np.asarray(e_qp_kn_ev, dtype=np.float64)
    if sigma_c.pair is None:
        raise ValueError("build_qsgw_sigma_xc: a diagonal-only on-shell Sigma has no pair")
    read, inside = sigma_eval_omega(omega, E, out_of_grid)
    check_on_shell_reads(sigma_c.request.table_ev("pair"), read, what="QSGW pair")
    n_clipped = int(np.count_nonzero(~inside))
    if band_axis is not None:
        from runtime.padding import authenticate_axis, pad_square
        authenticate_axis(sigma_c.pair, band_axis, axis=-1, where="QSGW on-shell Sigma_c")
        sigma_x_kij_ry = pad_square(sigma_x_kij_ry, band_axis)
    out = _qsgw_on_shell_kernel(mesh_xy, bool(replicated_output))(sigma_c.pair, sigma_x_kij_ry)
    from common.timing import stage_wait
    stage_wait(out)   # a timing-only wait (LORRAX_STAGE_SYNC)
    if band_axis is not None and replicated_output:
        from runtime.padding import strip_axis
        out = strip_axis(strip_axis(out, band_axis, axis=-2), band_axis, axis=-1)
    nk, nb = E.shape
    return out, {
        "n_clipped": float(n_clipped),
        "frac_clipped": float(n_clipped) / float(nk * nb) if nk * nb else 0.0,
        "omega_min_ev": float(omega[0]), "omega_max_ev": float(omega[-1])}


# ---------------------------------------------------------------------------
# The opt-in QSGW cube write — one implementation, two seams
# ---------------------------------------------------------------------------

def write_qsgw_sigma_cube(
    sigma_omega_h5_path,
    sigma_xc_qsgw_kij_ry,
    *,
    config,
    star_already_selected: bool = False,
    print_fn=print,
) -> bool:
    """Append ``sigma_xc_qsgw_kij_ev`` to a ``sigma_mnk.h5`` just written.

    No-op unless the deck sets ``write_qsgw_datasets`` (see
    ``gw_config._DEFAULTS`` for what the appendix is and why it is off by
    default).  Returns whether anything was written.

    TWO CALL SITES, AND THEY ARE NOT INTERCHANGEABLE.  Each hands over
    a QSGW build — Σ_x + Σ_c^QSGW, already Hermitised by
    :func:`build_qsgw_sigma_xc`'s closing ``½(M + M†)`` — at the one
    moment its own path has the build that goes into H, in the basis the
    file's other cubes are in:

    * ``gw.sigma_dispatch.compute_sigma_xc``, immediately after the QSGW
      build, on the one-shot path where ``finalize_dynamic_sigma``
      created the file a few statements earlier.  Gated there on the same
      ``write_sigma_omega_h5`` flag that decided whether the file was
      written at all, so it cannot fire during an SC iteration (which
      passes False and writes nothing).
    * ``gw.sc_iteration.dump_sigma_omega_h5_final``, right after the
      converged single write, and BEFORE ``run_sc_driver`` rotates the
      matrix back to the DFT basis.

    A replicated cube is appended by rank 0.  A band-sharded cube is not
    gathered for this optional appendix; the omission is announced and all
    ranks meet the same barrier.
    """
    if not bool(getattr(config, "write_qsgw_datasets", False)):
        return False
    if not sigma_omega_h5_path or sigma_xc_qsgw_kij_ry is None:
        return False
    from common.collectives import barrier
    from file_io import append_qsgw_datasets_h5

    if _is_band_sharded_static(sigma_xc_qsgw_kij_ry):
        print_fn(
            "  QSGW cube appendix omitted: the live operator is "
            "P(None,'x','y')-sharded and this optional rank-0 writer does "
            "not gather it")
        barrier("qsgw_sigma_cube_append_sharded_skip")
        return False

    if jax.process_index() == 0:
        append_qsgw_datasets_h5(
            sigma_omega_h5_path,
            {"sigma_xc_qsgw_kij_ev":
                RYD_TO_EV * np.asarray(sigma_xc_qsgw_kij_ry)},
            star_already_selected=star_already_selected,
            print_fn=print_fn)
    barrier("qsgw_sigma_cube_append")
    return True


# ---------------------------------------------------------------------------
# update_H — the one-shot qp_solver seam
# ---------------------------------------------------------------------------

def solve_qp(
    qp_solver,
    sigma_result,
    kin_ion: jax.Array,
    *,
    config,
    mesh_xy: Mesh,
    print_fn=print,
) -> jax.Array:
    """``update_H[Σ; one_shot_dft]`` — turn a :class:`~gw.sigma_dispatch.SigmaResult`
    into ``sigma_total = Σ_xc + V_H`` (Ry) whose eigh yields the QP
    eigenstates.

    The full-matrix QSGW-Hermitianised Σ_xc was already evaluated at E_DFT
    inside ``compute_sigma_xc`` (the same call the SC iteration map makes),
    so this is a pass-through: no QP equation is solved.  Its ``eigh``
    result is distinct from the fixed-DFT-state diagonal G0W0 result written
    to ``eqp0.dat``.  Static modes (X_ONLY / COHSEX) land here too:
    ``sigma_xc_kij_ry`` is the mode's total Σ_xc by construction.
    ``self_consistent`` is NOT handled here — the SC driver owns its own
    loop, which evaluates Σ at each map's own energies (``sc_iteration``).
    """
    from .gw_config import QPSolver

    del kin_ion, mesh_xy
    if qp_solver is not QPSolver.ONE_SHOT_DFT:
        raise ValueError(
            f"solve_qp: qp_solver = {qp_solver.value} is not a one-shot "
            "solver; self_consistent runs through run_sc_driver.")
    return sigma_result.sigma_xc_kij_ry + sigma_result.v_h_kij_ry


# ---------------------------------------------------------------------------
# QP-energy comparison plot
# ---------------------------------------------------------------------------

def plot_qp_energy_comparison(
    output_png: str,
    e_ref_kn_ev: np.ndarray,
    e_static_kn_ev: np.ndarray,
    e_dyn0_kn_ev: np.ndarray,
    e_diag_sc_kn_ev: np.ndarray,
) -> str:
    """Scatter + k=0 trend of QP energies for the three approximations."""
    import matplotlib.pyplot as plt

    x = np.asarray(e_ref_kn_ev, dtype=np.float64).reshape(-1)
    y_static = np.asarray(e_static_kn_ev, dtype=np.float64).reshape(-1)
    y_dyn0 = np.asarray(e_dyn0_kn_ev, dtype=np.float64).reshape(-1)
    y_diag = np.asarray(e_diag_sc_kn_ev, dtype=np.float64).reshape(-1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(x, y_static, s=10, alpha=0.6, label="Static COHSEX")
    axes[0].scatter(x, y_dyn0, s=10, alpha=0.6, label="Bare X + Σ_c(0)")
    axes[0].scatter(x, y_diag, s=10, alpha=0.6, label="Diagonal SC Σ(E)")
    mn = float(min(np.min(x), np.min(y_static), np.min(y_dyn0), np.min(y_diag)))
    mx = float(max(np.max(x), np.max(y_static), np.max(y_dyn0), np.max(y_diag)))
    axes[0].plot([mn, mx], [mn, mx], "k--", lw=1)
    axes[0].set_xlabel("Reference energy (eV)")
    axes[0].set_ylabel("QP energy (eV)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("All (k, n)")

    b = np.arange(e_ref_kn_ev.shape[1])
    axes[1].plot(b, e_static_kn_ev[0], "-o", ms=3, label="Static COHSEX")
    axes[1].plot(b, e_dyn0_kn_ev[0], "-o", ms=3, label="Bare X + Σ_c(0)")
    axes[1].plot(b, e_diag_sc_kn_ev[0], "-o", ms=3, label="Diagonal SC Σ(E)")
    axes[1].set_xlabel("Band index")
    axes[1].set_ylabel("Energy at k = 0 (eV)")
    axes[1].set_title("k = 0")
    axes[1].legend(fontsize=8)

    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    return output_png


# ---------------------------------------------------------------------------
# Managed-file cleanup — shared by the SC loop's disk-bounded artifacts
# ---------------------------------------------------------------------------

def remove_managed(dir_path, pattern, *, keep=(), barrier_tag, print_fn=print):
    """Rank-0 scan-and-unlink of managed entries, then a collective barrier.

    Removes every entry of ``dir_path`` whose NAME fullmatches ``pattern``
    and whose full path is not in ``keep``, on rank 0 only; all ranks then
    meet at ``barrier_tag`` so the directory looks the same everywhere on
    return.  A matching directory (a per-map scratch generation) is removed
    with its contents.  Returns the removed names (empty off rank 0).  Shared
    by ``sc_iteration._clear_sc_eqp_snapshots``,
    ``gw.mpa.model.retain_iteration_artifacts`` and the shared-pole exports and
    scratch; each caller owns its own regex, keep-set and report line.
    """
    import os
    import re
    import shutil
    from common.collectives import barrier, process_rank

    managed = re.compile(pattern)
    keep = set(keep)
    root = os.path.abspath(os.fspath(dir_path))
    removed = []
    if process_rank() == 0 and os.path.isdir(root):
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if managed.fullmatch(name) and path not in keep:
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                removed.append(name)
    barrier(barrier_tag, print_fn=print_fn)
    return removed


__all__ = [
    "build_qsgw_sigma_xc",
    "check_on_shell_reads",
    "static_sigma_diag_to_host",
    "omega_coverage",
    "plot_qp_energy_comparison",
    "remove_managed",
    "sigma_grid_edge_ambiguity",
    "sigma_eval_omega",
    "write_qsgw_sigma_cube",
]
