"""Coarse k-grid to fine: the exact interpolation, and the one densifier.

AUTHORITY RULE — coarse to fine is a ZERO-PAD IN REAL SPACE.  Zero-padding the
R-lattice evaluates the exact band-limited trigonometric polynomial defined by
the coarse samples on ANY finer uniform grid.  When the grids nest it passes
through every coarse sample; for a non-divisor target (for example 8→12) it
passes through the subset of points the two grids share.  When the requested
grid equals the coarse one the bundle is returned untouched, byte for byte.
There is exactly one sharded implementation of that densification
(``make_w_densifier``), the (μ, ν) sharding survives it end to end, and no step
materialises a replicated N_μ²-class array.

WHAT THE INTERPOLANT IS ALLOWED TO SEE.  Only the smooth BODY.  W's Γ head is a
Kronecker delta, and the trigonometric interpolant of a delta is a Dirichlet
kernel — it smears a fraction of the head's ~10³ meV prefactor onto fine q that
should carry none of it, and it cannot produce either the bulk 1/q² rise or
the slab 1/|q| cusp inside the coarse Γ cell at all.  So the loader DEFERS the
rank-1 whead injection whenever a
densification is pending, this module densifies the head-excluded body, and
``build_w_head_channel`` re-attaches an analytic per-fine-q head from the one
ratified integrand (``gw.head_densify``).  ``w_head_densify = legacy`` restores
the interpolated-delta path and exists only so the A/B that prices the repair
has a control arm that is the shipped code rather than a reconstruction of it.

What is grid-dependent and what is not is the module's other rule.  ψ and the
quasiparticle energies are re-interpolated through one htransform; ``W``'s
direct term is densified; the q=0 exchange BODY is built from the centroids and
the G-sphere alone and is therefore k-grid INVARIANT, so it is carried through
unchanged.  Only the rank-1 head scalar depends on the grid, and whether to
rebuild the EXCHANGE head is the deck's ``head_minibz_average`` key — read here
the same way every other consumer of that key in the tree reads it, default
off.  This path once forced that rebuild on regardless, which quietly replaced
the deck's disk body with a model reconstruction of itself.
"""
from __future__ import annotations

import os
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.fft_helpers import make_sharded_fftn_3d, make_sharded_ifftn_3d

from .bse_serial import compute_pair_amplitude
from .bse_window import PAD_EPS_GUARD_RY


def _zeropad_R_axis(x: jax.Array, axis: int, n_fine: int) -> jax.Array:
    """Spectral zero-pad one real-space (R) axis from a coarse to a fine k-lattice.

    ``x`` is a real-space tensor whose ``axis`` is the R-lattice DUAL to a
    COARSE BZ axis of length ``n_c = x.shape[axis]`` (i.e. an ``ifft`` over that
    BZ axis already happened).  We embed the ``n_c`` coarse Fourier coefficients
    into ``n_fine`` slots so that an ``fft`` back to ``n_fine`` BZ points is the
    exact band-limited trigonometric interpolant of the coarse ``W(k)``.  The
    coefficient embedding is defined for every ``n_fine >= n_c``; if the two
    uniform grids nest it passes through all coarse samples, and otherwise it
    passes through the points common to both grids.

    Per-element R-lattice index map (the crux).  numpy/JAX FFT order maps array
    index ``i∈[0,N)`` to the physical lattice-vector rep ``n = fftfreq(N)*N``:

        n = i        for 0 ≤ i < ⌈N/2⌉     (the non-negative reps 0,1,…)
        n = i − N    for ⌈N/2⌉ ≤ i < N      (the negative reps …,−2,−1, wrapped)

    For even ``N`` the single Nyquist rep ``n = −N/2`` sits at ``i = N/2`` (it is
    in the negative branch here — kept single-sided, NOT split; sample agreement
    is exact regardless, and single-sided = "coarse WS cell, zero outside",
    which is precisely the requested zero-pad).  Embedding coarse→fine with
    ``s = ⌈n_c/2⌉ = (n_c+1)//2`` non-negative reps:

        coarse i_c ∈ [0, s)      (reps n = 0 … s−1)      → fine i_f = i_c
        coarse i_c ∈ [s, n_c)    (reps n = −(n_c−s) … −1) → fine i_f = i_c + (n_fine − n_c)

    i.e. the low block keeps its indices, the high (negative-freq) block slides
    to the TOP of the fine axis, and the gap ``i_f ∈ [s, n_fine − (n_c−s))`` is
    ZERO (the high-|R| coefficients absent from the coarse cell — incl. the
    fine-only Nyquist and the absent coarse ``+n_c/2``).  Realised as
    ``concat([low, zeros, high])`` — the textbook fft zero-insertion, exact for
    any parity of ``n_c``.

    NOTE: scale is applied ONCE by the caller (``pad_W_R_to_grid``), not here.
    """
    n_c = x.shape[axis]
    if n_fine == n_c:
        return x
    if n_fine < n_c:
        raise ValueError(
            f"_zeropad_R_axis: fine length {n_fine} must be at least the "
            f"coarse length {n_c}; spectral zero-padding cannot coarsen an "
            f"axis.")
    s = (n_c + 1) // 2
    lo = jax.lax.slice_in_dim(x, 0, s, axis=axis)
    hi = jax.lax.slice_in_dim(x, s, n_c, axis=axis)
    zshape = list(x.shape)
    zshape[axis] = n_fine - n_c
    zeros = jnp.zeros(zshape, dtype=x.dtype)
    return jnp.concatenate([lo, zeros, hi], axis=axis)


def pad_W_R_to_grid(W_R_coarse: jax.Array,
                    fine_grid: tuple[int, int, int]) -> jax.Array:
    """Zero-pad a coarse-k-grid real-space ``W_R`` onto a finer k-lattice.

    Enables a CHEAP coarse-grid W (e.g. GW/W on 6×6) to drive the BSE direct
    term on an arbitrarily FINE interpolated exciton sampling (e.g. 8×8 W on a
    12×12 exciton grid): zero-padding in R is EXACT trigonometric interpolation
    in k.  It agrees with the coarse ``W(k)`` at every k-point common to the two
    grids (all coarse points when the grids nest).

    Parameters
    ----------
    W_R_coarse : (..., ncx, ncy, ncz)
        Screened interaction on the real-space (R) lattice DUAL to a COARSE BZ
        grid, i.e. ``ifftn(W_q_coarse, axes=last-3, norm='ortho')``.  Leading
        axes (μ, ν, spin, …) are carried through untouched; only the trailing
        three (kx, ky, kz) R-axes are re-embedded.
    fine_grid : (nfx, nfy, nfz)
        Target BZ grid.  Each ``nf`` must be at least the matching coarse
        length; integer nesting is not required.

    Returns
    -------
    W_R_fine : (..., nfx, nfy, nfz)
        Such that ``fftn(W_R_fine, norm='ortho')`` sampled at the fine BZ points
        coinciding with the coarse grid equals ``W_q_coarse`` bit-close, i.e.
        ``W_R_fine == ifftn(W_q_fine_interpolant, norm='ortho')`` — a drop-in for
        the matvec's ``W_R`` on the FINE grid.

    Scale.  ortho FFTs carry ``1/√N``, and ``N`` grows coarse→fine.  For the
    fine ``fft`` of the embedded coefficients to reproduce the coarse samples,
    the padded tensor is multiplied by ``√(∏nf / ∏nc)`` (= 2 for 6×6×1→12×12×1).

    Degenerate ``fine_grid == coarse shape``: returns ``W_R_coarse`` UNCHANGED
    (exact byte-identical no-op — scale is 1 and no axis is padded).
    """
    if W_R_coarse.ndim < 3:
        raise ValueError(
            f"pad_W_R_to_grid: expected trailing (kx,ky,kz) axes, got ndim="
            f"{W_R_coarse.ndim}")
    coarse_grid = tuple(int(s) for s in W_R_coarse.shape[-3:])
    fine_grid = tuple(int(s) for s in fine_grid)
    if fine_grid == coarse_grid:
        return W_R_coarse                         # exact no-op (fast path)
    out = W_R_coarse
    for ax, nf in zip((-3, -2, -1), fine_grid):
        out = _zeropad_R_axis(out, ax, nf)
    n_c_tot = coarse_grid[0] * coarse_grid[1] * coarse_grid[2]
    n_f_tot = fine_grid[0] * fine_grid[1] * fine_grid[2]
    scale = jnp.sqrt(jnp.asarray(n_f_tot / n_c_tot, dtype=out.real.dtype))
    return out * scale


def make_w_densifier(
    mesh_xy: Mesh,
    w_spec: P,
    fine_grid: tuple[int, int, int],
    *,
    output: str = "k",
):
    """THE coarse→fine W densifier — the ONE sharded implementation.

    Returns a jitted ``fn(W_q_coarse) -> W_fine`` that takes a coarse-grid
    ``W_q`` of shape ``(μ, ν, ncx, ncy, ncz)`` sharded as ``w_spec`` (the k
    axes must be replicated in ``w_spec``; only μ/ν may be sharded) and
    produces the fine-grid W via  ifftn(k→R) → zero-pad R
    (:func:`pad_W_R_to_grid`, exact trig interpolation) → optionally
    fftn(R→k):

      * ``output='R'`` → ``W_R_fine`` — what the exciton_bands
        ``--w-coarse-grid`` path feeds the matvec directly;
      * ``output='k'`` → ``W_q_fine`` — what the ``bse_k_grid`` bundle
        densification stores, so each solver's own ``ifftn(W_q)``
        reproduces the padded ``W_R``.

    SHARDING / SCALING ENVELOPE.  Both FFTs are the shard_map interior
    kernels (:func:`make_sharded_ifftn_3d` / :func:`make_sharded_fftn_3d`)
    and the R-axis zero-pad is traced INSIDE one ``jax.jit`` whose
    ``out_shardings`` pins ``w_spec``, so the (μ,ν) sharding survives end to
    end: per-rank peak stays at the local ``(μ_loc, ν_loc, nk_fine)`` tile.
    No step materializes a replicated N_μ²-class array — the eager
    ``local_ifftn3``/``local_fftn3`` + ``device_put`` form this replaces
    all-gathered the full tensor per rank (audit P0-4/P2-7) and is what the
    ``tests/test_fft_shardmap_context.py`` gate now bans.
    """
    if output not in ("k", "R"):
        raise ValueError(f"make_w_densifier: output must be 'k' or 'R', got {output!r}")
    fine_grid = tuple(int(s) for s in fine_grid)
    w_sh = NamedSharding(mesh_xy, w_spec)
    _ifftn = make_sharded_ifftn_3d(mesh_xy, w_spec, w_spec,
                                   axes=(2, 3, 4), norm="ortho")
    _fftn = make_sharded_fftn_3d(mesh_xy, w_spec, w_spec,
                                 axes=(2, 3, 4), norm="ortho")

    @partial(jax.jit, out_shardings=w_sh)
    def _densify(W_q_coarse):
        W_R_fine = pad_W_R_to_grid(_ifftn(W_q_coarse), fine_grid)
        if output == "R":
            return W_R_fine
        return _fftn(W_R_fine)

    return _densify


def decimate_W_q_to_subgrid(W_q: jax.Array,
                            coarse_grid: tuple[int, int, int]) -> jax.Array:
    """Sub-sample a fine-grid ``W_q`` onto a coarse BZ sub-grid (every m-th q).

    ``W_q`` is ``(μ, ν, nfx, nfy, nfz)`` on a FINE BZ grid; returns the tiles at
    the coarse sub-grid BZ points ``q_j = j·(nf/nc)`` (which coincide with a
    coarse ``nc``-point grid since ``nf`` is a multiple of ``nc``).  This is the
    honest "what a coarse-grid W sampling looks like, in the SAME ISDF μ-basis"
    — used to drive ``pad_W_R_to_grid`` from a single fine restart (the q=0 tile,
    incl. its rank-1 head, is preserved because q=0 is on both grids).
    """
    nf = tuple(int(s) for s in W_q.shape[-3:])
    coarse_grid = tuple(int(s) for s in coarse_grid)
    for a, (f, c) in enumerate(zip(nf, coarse_grid)):
        if c <= 0 or f % c != 0:
            raise ValueError(
                f"decimate_W_q_to_subgrid: fine axis {a} length {f} must be a "
                f"positive multiple of coarse {c}.")
    rx, ry, rz = (nf[0] // coarse_grid[0], nf[1] // coarse_grid[1],
                  nf[2] // coarse_grid[2])
    return W_q[:, :, ::rx, ::ry, ::rz]


def _parse_grid_spec(spec) -> Optional[tuple[int, int, int]]:
    """Parse a ``bse_k_grid`` value → ``(nx, ny, nz)`` or ``None`` (unset).

    Accepts ``"NX NY NZ"``, ``"NX,NY,NZ"``, a 3-tuple/list, or an empty
    string / None (→ ``None``).  Single source for both the cohsex.in key and
    any driver flag.
    """
    if spec is None:
        return None
    if isinstance(spec, (tuple, list)):
        parts = list(spec)
    else:
        s = str(spec).strip()
        if not s:
            return None
        parts = s.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"bse_k_grid must have 3 integers, got {spec!r}")
    return tuple(int(x) for x in parts)


def _read_lorrax_input_quietly(input_file: Optional[str]) -> dict:
    """The deck as a dict, or ``{}`` — a config read must never crash a load.

    Same tolerance as :func:`_resolve_bse_k_grid`'s own read: the loader has
    to work on a restart with no deck beside it, and a malformed key is a
    reason to fall back to defaults loudly, not to lose the tensors.
    """
    if input_file is None or not os.path.isfile(input_file):
        return {}
    try:
        from gw.gw_config import read_lorrax_input
        return read_lorrax_input(input_file) or {}
    except Exception as exc:
        print(f"BSE: deck read failed ({exc}); using defaults")
        return {}


def _resolve_bse_k_grid(bse_k_grid, input_file: Optional[str]):
    """Resolve the fine grid: explicit ``bse_k_grid`` arg wins; else read the
    ``bse_k_grid`` key from ``input_file`` (cohsex.in).  Returns a 3-tuple or
    ``None`` (feature off)."""
    fine = _parse_grid_spec(bse_k_grid)
    if fine is not None:
        return fine
    if input_file is not None and os.path.isfile(input_file):
        try:
            from gw.gw_config import read_lorrax_input
            return _parse_grid_spec(read_lorrax_input(input_file).get("bse_k_grid"))
        except Exception as exc:               # config parse must never crash load
            print(f"BSE: bse_k_grid config read failed ({exc}); feature off")
    return None


#: Deck / CLI values for ``w_head_densify``.  ``c1`` is the default and is the
#: repaired path; ``legacy`` is the pre-C1 behaviour, kept ONLY so the A/B that
#: prices the repair can run both arms through the shipped driver.  It is not a
#: supported production choice and says so when it fires.
W_HEAD_DENSIFY_MODES = ("c1", "legacy")


def resolve_w_head_densify(mode, params=None) -> str:
    """Resolve the coarse→fine W head treatment: ``'c1'`` (default) or
    ``'legacy'``.

    Explicit argument wins; else the deck's ``w_head_densify`` key; else
    ``'c1'``.  ``'legacy'`` re-enables trigonometric interpolation of the
    Kronecker-delta head — the defect ``gw.head_densify`` documents — and
    exists so the densified-versus-native A/B has a control arm that is the
    SHIPPED code path rather than a reconstruction of it.
    """
    val = mode
    if val is None and params is not None:
        val = params.get("w_head_densify")
    if val is None:
        return "c1"
    val = str(val).strip().lower()
    if val not in W_HEAD_DENSIFY_MODES:
        raise ValueError(
            f"w_head_densify = {mode!r} is not one of {W_HEAD_DENSIFY_MODES}. "
            f"'c1' splits the Γ head off before the densifier and re-attaches "
            f"it per fine q; 'legacy' trigonometrically interpolates it, which "
            f"is the documented defect and is kept only as an A/B control.")
    return val


def build_w_head_channel(wfn, sym, meta, params, *, coarse_grid, fine_grid,
                         whead, ref_grid, input_file, restart_file,
                         gamma_cell: str = "fine", log_fn=print):
    """The C1 head channel ``S_fine(q)`` for a coarse→fine W, end to end.

    ONE composer for BOTH shipping densification paths — the ``bse_k_grid``
    bundle densification (:func:`_interpolate_bse_data_to_grid`) and
    ``exciton_bands --w-coarse-grid``.  They differ only in which grid the
    reference head was measured on (``ref_grid``), and that difference is an
    argument rather than two code paths.

    Resolves ``S_cart`` (restart dataset, else a ``dipole.h5`` rebuild),
    evaluates the integrand's own cell average on ``ref_grid`` so the caller's
    ``whead`` can be checked against it, and hands the anchored per-q scalars
    back.  Everything it prints is meant to be read: the provenance ratio is
    the one number that says whether the head being re-attached describes the
    same screening the run solved with.

    Parameters
    ----------
    wfn, sym, meta : loader / symmetry table / system parameters
        ``wfn`` supplies the Coulomb geometry (``blat·bvec`` rows and Ω) via
        :meth:`vcoul.CoulombGeometry.from_wfn`; ``meta.sys_dim`` selects the
        bulk-3D or slab-2D ``vcoul`` kernel and is REQUIRED — a Meta without it
        is refused, not assumed bulk (see the body); ``sym`` is read only by the
        ``dipole.h5`` fallback route for ``S_cart`` (a restart that carries
        ``S_cart_head`` never touches it).  Both densification call sites
        already hold all three from their htransform leg, which is also where
        the ``sys_dim`` stamp is put on.
    params : dict
        Deck keys.  ``head_minibz_average`` must match the GW run — it selects
        the Baldereschi-Tosatti analytic-sphere branch of the estimator.
    coarse_grid, fine_grid, ref_grid : tuple[int, int, int]
    whead : float
        The head scalar the loader injects, Ry·bohr³, real.
    gamma_cell : {"fine", "coarse"}
        ``"coarse"`` is the red twin; see
        :func:`gw.head_densify.build_fine_head_scalars`.

    Returns
    -------
    numpy.ndarray, shape ``fine_grid``, float64
    """
    from gw import head_densify
    from gw.head_correction import resolve_head_S_cart
    from ffi import _services
    _services.ensure_on_path()
    from vcoul import CoulombGeometry

    params = params or {}

    # ── DIMENSIONALITY FIRST, before the S tensor and before the Γ-cell
    # integral: an unsupported/unstamped deck must hear that, not "your
    # dipole.h5 is missing".
    #
    # ``Meta`` has no ``sys_dim`` field — it is STAMPED on the two Metas the
    # tree builds, by ``gw.gw_jax.main`` and by
    # ``bandstructure.htransform.initialize_wfns`` (which is the one BOTH
    # shipping densification paths hold: the ``bse_k_grid`` bundle leg below
    # and ``bse.exciton_bands``' ``--w-coarse-grid`` leg).  Until 2026-08-17
    # this read ``getattr(meta, "sys_dim", None)`` into a guard that RETURNED
    # on ``None``, and htransform did not stamp: so the bulk-3D refusal could
    # not fire on either path, and a ``sys_dim = 2`` deck re-attached
    # ``8π/|q|²`` — the untruncated 3D pole — where the true 2D head goes as
    # ``8π·z_c/|q_∥|``.  That error has no fixed size: it GROWS as the fine
    # grid densifies, which is the one thing a densification is for.
    sys_dim = getattr(meta, "sys_dim", None)
    if sys_dim is None:
        raise ValueError(
            "w_head_densify = c1: the Meta handed to "
            "bse.bse_densify.build_w_head_channel carries no ``sys_dim``, so "
            "gw.head_densify cannot safely select a bulk or slab kernel.  "
            "``Meta`` has no sys_dim field; it is "
            "stamped, by gw.gw_jax.main on the GW driver's Meta and by "
            "bandstructure.htransform.initialize_wfns on this caller's.  A "
            "Meta arriving here without it was built by a third site, and "
            "that site is the fix — do not default it to 3 here.")
    sd = head_densify._validated_sys_dim(sys_dim)

    S_cart, prov = resolve_head_S_cart(
        restart_file, input_file=input_file, wfn=wfn, sym=sym, meta=meta,
        params=params, print_fn=log_fn)
    if S_cart is None:
        raise ValueError(
            f"w_head_densify = c1 needs the head's S tensor and could not get "
            f"one: {prov}.  C1 re-attaches W's Γ head per fine q from the "
            f"integrand ⟨v/(1 − v qᵀSq)⟩, and without S there is no integrand "
            f"— only the coarse cell AVERAGE, which is the number the "
            f"trigonometric interpolant already mishandles.  Fix by rerunning "
            f"GW so the restart carries S_cart_head, or by putting the run's "
            f"dipole.h5 beside the deck.  To proceed anyway with the "
            f"documented defect, set w_head_densify = legacy and read the "
            f"result knowing the Γ head is interpolated.")

    geom = CoulombGeometry.from_wfn(wfn)
    analytic_sphere = bool(params.get("head_minibz_average", False))
    gamma_ref = head_densify.gamma_cell_head_scalar(
        geom, ref_grid, S_cart, sys_dim=sd,
        analytic_sphere=analytic_sphere)
    ratio = float(whead) / gamma_ref

    S_fine = head_densify.build_fine_head_scalars(
        geom, coarse_grid, fine_grid, S_cart,
        head_ref=float(whead), gamma_ref=gamma_ref, ref_grid=ref_grid,
        sys_dim=sd,
        analytic_sphere=analytic_sphere, gamma_cell=gamma_cell)

    # THE SUM RULE, reported on the deck.  The head channel's zone average is
    # a property of the material and the cell, and the grid-independent number
    # it converges to is (1/N_c)·⟨S⟩ over the COARSE cell — so the target has
    # to be expressed on the coarse cell even when the reference head was
    # measured on a different one (which is exactly the ``--w-coarse-grid``
    # case, where a natively fine restart is decimated).  Getting this wrong
    # does not change a single number that ships; it changes what the log
    # claims, which is worse, because a diagnostic nobody can trust is a
    # diagnostic nobody reads.
    cg = tuple(coarse_grid)
    gamma_coarse = (gamma_ref if tuple(ref_grid) == cg
                    else head_densify.gamma_cell_head_scalar(
                        geom, cg, S_cart, sys_dim=sd,
                        analytic_sphere=analytic_sphere))
    n_c = cg[0] * cg[1] * cg[2]
    target = float(whead) * (gamma_coarse / gamma_ref) / n_c
    weight = head_densify.coarse_gamma_cell_weights(
        head_densify.fine_q_cart(geom.bvec, fine_grid), cg, fine_grid,
        bvec=geom.bvec)
    zone = head_densify.head_channel_zone_average(S_fine)
    log_fn(
        f"[w-head-c1] S_cart: {prov}; provenance ratio whead/⟨v/(1−vqSq)⟩ on "
        f"{tuple(ref_grid)} = {ratio:.6f} (1.0 means the restart's head and "
        f"this S are the same screening)")
    log_fn(
        f"[w-head-c1] head channel re-attached at "
        f"{int(np.count_nonzero(S_fine))} fine q carrying total weight "
        f"{float(np.sum(weight)):.6f} (= N_f/N_c = "
        f"{(fine_grid[0]*fine_grid[1]*fine_grid[2])/n_c:.6f}; more points than "
        f"weight means boundary/common-cell sharing), of "
        f"{fine_grid[0]*fine_grid[1]*fine_grid[2]} fine q total")
    log_fn(
        f"[w-head-c1] S(Γ_fine) = {S_fine[0, 0, 0]:.4f} vs injected "
        f"{float(whead):.4f} Ry·bohr³ (×{S_fine[0, 0, 0]/float(whead):.3f} "
        f"for a {(fine_grid[0]*fine_grid[1]*fine_grid[2])/n_c:.0f}× finer "
        f"cell); SUM RULE: zone average {zone:.6f} vs the grid-independent "
        f"{target:.6f} ({100.0*abs(zone/target - 1.0):.1f}% — a midpoint "
        f"quadrature error that shrinks under refinement, 0 when fine == "
        f"coarse) [gamma_cell={gamma_cell}]")
    if abs(ratio - 1.0) > 0.05:
        import warnings
        msg = (f"w_head_densify = c1: the injected head and the S tensor "
               f"disagree by {100*abs(ratio-1.0):.1f}% on {tuple(ref_grid)} "
               f"(whead={float(whead):.4f}, integrand={gamma_ref:.4f} "
               f"Ry·bohr³).  The re-attached head keeps the INJECTED scale, "
               f"so nothing silently changes magnitude, but its q-dependence "
               f"comes from an S that does not reproduce it — most often a "
               f"deck vhead/whead_0freq override pinned to an external value, "
               f"or a head_minibz_average / wcoul0_source that differs from "
               f"the GW run's.")
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        log_fn(f"[w-head-c1] [WARN] {msg}")
    return S_fine


def _interpolate_bse_data_to_grid(
    data: dict,
    fine_grid: tuple[int, int, int],
    restart_file: str,
    input_file: str,
    mesh_xy: Mesh,
    *,
    head_channel: dict | None = None,
    distrib_la_batched_route: str | None = None,
    htransform_a_band: int | None = None,
    log_fn=print,
) -> dict:
    """Interpolate the WHOLE coarse BSE ``data`` bundle onto ``fine_grid``.

    The single coarse→fine densification the ``bse_k_grid`` knob drives, living
    in the GENERAL init so EVERY solver consumes a fine-grid bundle unchanged.
    Each piece goes through the shared builder its exciton_bands sibling uses
    (until 2026-07-31 the W leg was a SECOND, eager copy — ``local_ifftn3``/
    ``local_fftn3`` outside any shard_map, all-gathering the (μ,ν)-sharded W
    per rank; audit P0-4):

      * ψ_{v,c}(k) and QP ε_{v,c}(k) on the fine mesh ← ONE htransform fH
        (``bandstructure.htransform.initialize_wfns`` +
        ``bandstructure.bse_setup.compute_wfns_fi`` with ``kgrid_fi=fine_grid``);
        the fH is built over the full loaded band window and only the BSE
        sub-window [nval−n_val, nval+n_cond) is returned (interior, guarded).
      * V_Q exchange q=0 tile ← CARRIED THROUGH unchanged by default.  A Q=0
        exciton's exchange is the single q=0 tile (dense in k,k'), and that
        tile's BODY is built from the centroids and the G-sphere alone, so it
        is k-grid-INVARIANT: densifying it is unnecessary.  Only the rank-1
        head scalar ``<v>_mBZ`` is grid-dependent (the cell is BZ/N_k).  The
        deck's ``head_minibz_average`` key (default off) chooses: OFF keeps the
        coarse bundle's tile exactly, deck ``vhead`` and all; ON rebuilds it
        through ``bse.vq_interp.build_vq_evaluator`` (the SAME builder
        exciton_bands calls) at Q=0 with the FINE mini-BZ head
        (``minibz_head_vlr(..., kgrid=fine_grid)``), replacing the disk body
        and the deck head.  Until 2026-08-09 this path forced the ON branch
        unconditionally, ignoring the key — see the block comment below.
      * W direct ← ``make_w_densifier(output='k')`` — the ONE sharded
        coarse→fine densifier (shard_map FFTs + jitted R zero-pad, (μ,ν)
        sharding preserved end to end); the exciton_bands ``--w-coarse-grid``
        flag runs the SAME factory with ``output='R'``.  ``bse_k_grid`` drives
        it automatically (the banner below is the "W-pad fired" proof).
        Its operand is the head-EXCLUDED body and the Γ head is re-attached
        analytically afterwards — see ``head_channel`` below.

    Band dims (n_val/n_cond and their mesh-pads), n_rmu(_pad), and the q=0 head
    projectors g0 are grid-INVARIANT and carried through untouched; only the k
    axis (and W's k-axes) change.  Returns a new bundle with the SAME keys.

    Parameters
    ----------
    head_channel : dict or None
        C1's split-and-reattach hand-off from the loader, ``None`` on the
        ``legacy`` arm.  When present it carries ``whead`` (the head scalar the
        loader would have injected at q=0, Ry·bohr³), ``cell_volume`` (Ω), and
        optionally ``gamma_cell`` (``"fine"`` = C1, ``"coarse"`` = the red
        twin).  Its presence is also the SIGNAL that the loader deferred the
        injection, so the W tile in ``data`` is the pre-injection body; the two
        must not disagree, which is why one dict carries both the decision and
        the numbers rather than a flag beside a value.
    """
    from bandstructure import htransform as ht
    from bandstructure.bse_setup import compute_wfns_fi
    from gw.gw_config import (
        read_lorrax_input,
        resolve_distrib_la_batched_route,
    )

    from . import vq_interp

    coarse_grid = (int(data["nkx"]), int(data["nky"]), int(data["nkz"]))
    fine_grid = tuple(int(s) for s in fine_grid)
    for a, (f, c) in enumerate(zip(fine_grid, coarse_grid)):
        if c <= 0 or f < c:
            raise ValueError(
                f"bse_k_grid axis {a}: fine {f} must be at least the coarse "
                f"restart extent {c}; trigonometric densification cannot "
                f"coarsen an axis.")
    n_val = int(data["n_val"]); n_cond = int(data["n_cond"])
    nv_pad = int(data["n_val_pad"]); nc_pad = int(data["n_cond_pad"])
    n_rmu = int(data["n_rmu"]); n_rmu_pad = int(data["n_rmu_pad"])
    nk_f = fine_grid[0] * fine_grid[1] * fine_grid[2]
    log_fn(f"[bse_k_grid] coarse {coarse_grid[0]}x{coarse_grid[1]}x{coarse_grid[2]}"
           f" → fine {fine_grid[0]}x{fine_grid[1]}x{fine_grid[2]} "
           f"({nk_f} k-pts); interpolating ψ/ε (htransform), V_Q (vq_interp), "
           f"W (zero-pad in R)")

    params = read_lorrax_input(input_file)
    _distrib_la_batched_route = resolve_distrib_la_batched_route(
        params, override=distrib_la_batched_route)

    # ── WHICH ISDF BASIS THE htransform FITS IN ───────────────────────────
    # On a NATIVE bundle: the deck's own table, and ``keep`` is None — this
    # block is then a no-op and everything below is the program it always was.
    #
    # On a DOWNFOLDED bundle the two questions come apart, and the densifier
    # has to answer them through ``exciton_bands.resolve_isdf_basis``, the ONE
    # owner of this contract.  The sole whole-state QRCP fit applies the
    # ordered ``keep`` before evaluating its basis and is born at μ_S.
    # Without that route, the pad below is asked for
    # (μ_S_pad − μ_L) columns — NEGATIVE — and the run dies in ``jnp.pad``
    # with an index error that names neither the downfold nor the basis.
    # Same defect class as PIPELINE_HEALTH row 4, which closed this for
    # exciton_bands and never reached the ``bse_k_grid`` path.
    from .exciton_bands import resolve_isdf_basis
    centroids_path, keep_idx = resolve_isdf_basis(
        restart_file, params, input_file, n_rmu_bundle=n_rmu, log=log_fn)
    params["centroids_file"] = centroids_path
    # HOST numpy closed over the trace, not a device_put: an eager device_put
    # of a host array is the hidden-all-gather site; a numpy array becomes an
    # HLO constant and is process-local by construction.
    keep = None if keep_idx is None else np.asarray(keep_idx, dtype=np.int32)

    # ── ψ_{v,c}(k_fine), ε_{v,c}(k_fine): ONE htransform fH ───────────────
    _fit_subset = keep
    _output_keep = None
    (wfn, sym, meta, _mesh, basis,
     enk_sigma) = ht.initialize_wfns(
         input_file, params, log_fn, mesh_xy=mesh_xy,
         centroid_subset_idx=_fit_subset)
    ctilde, B_at_mu = basis.ctilde, basis.basis_at_nodes
    nb_window = int(ctilde.shape[1])
    nval_in = int(params["nval"])          # window-relative CBM index
    b_min, b_max = nval_in - n_val, nval_in + n_cond
    if b_min < 0 or b_max > nb_window:
        raise ValueError(
            f"bse_k_grid BSE window [{b_min},{b_max}) escapes the htransform fH "
            f"window [0,{nb_window}); the input's nval/nband must load "
            f">= {n_val} valence and >= {n_cond} conduction guard bands.")
    # SPLASH RADIUS OF THE f-SHOULDER (``2026-08-11-fifth-wall-is-the-f-\
    # transform-shoulder.md`` §7, audited here).  The check above refuses
    # ``b_max > nb_window`` and so PERMITS ``b_max == nb_window`` — a deck
    # whose ``nband`` equals ``nval + ncond`` has no conduction guard at all
    # and lands exactly where the refit did: f(eps) is identically zero at and
    # above ``max_k eps`` of the window's own top band, so the top of the BSE
    # conduction selection comes back from eigh as a null-space direction.
    # ``compute_wfns_fi``'s f-shoulder gate refuses; this says it first, in
    # the deck's own vocabulary.
    _n_guard = nb_window - b_max
    if _n_guard < 4:
        log_fn(f"  [warn] bse_k_grid: only {_n_guard} conduction guard "
               f"band(s) above the BSE window [{b_min},{b_max}) inside the "
               f"{nb_window}-band htransform window.  The f-transform's shift "
               f"is max_k eps of THAT window's top band, so f == 0 there and "
               f"the shoulder below it carries ~1% of fH's weight; the top of "
               f"the returned window may be arbitrary.  Raise nband/ncond.")
    bundle = compute_wfns_fi(
        ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
        kgrid_co=coarse_grid, kgrid_fi=fine_grid,
        band_window_fi=(b_min, b_max), mesh_xy=mesh_xy, log_fn=log_fn,
        a_band_index=htransform_a_band,
        batch_size=int(params.get("wfn_fi_q_chunk", 0)),
        centroid_keep_idx=_output_keep,
        distrib_la_batched_route=_distrib_la_batched_route)

    x4 = NamedSharding(mesh_xy, P(None, None, None, "x"))
    y4 = NamedSharding(mesh_xy, P(None, None, None, "y"))
    rep = NamedSharding(mesh_xy, P())

    @partial(jax.jit, out_shardings=(x4, y4, x4, y4, rep, rep))
    def _split_pad(psi, enk):
        # psi (nk_f, n_val+n_cond, ns, n_mu); enk (nk_f, n_val+n_cond).
        # A downfold parent→child column slice has already run INSIDE
        # compute_wfns_fi as each q chunk entered its retained cache, before
        # the global concatenate.  This jit therefore never receives a
        # parent-width psi operand.
        psi_v = psi[:, :n_val]
        psi_c = psi[:, n_val:n_val + n_cond]
        psi_v = jnp.pad(psi_v, ((0, 0), (0, nv_pad - n_val), (0, 0),
                                (0, n_rmu_pad - psi.shape[3])))
        psi_c = jnp.pad(psi_c, ((0, 0), (0, nc_pad - n_cond), (0, 0),
                                (0, n_rmu_pad - psi.shape[3])))
        # ε pad = SIGNED SENTINEL, not zero (module note): ΔE = ε_c − ε_v is
        # the diagonal of the operator the BSE drivers diagonalise.
        eps_v = jnp.pad(enk[:, :n_val], ((0, 0), (0, nv_pad - n_val)),
                        constant_values=-PAD_EPS_GUARD_RY)
        eps_c = jnp.pad(enk[:, n_val:n_val + n_cond],
                        ((0, 0), (0, nc_pad - n_cond)),
                        constant_values=PAD_EPS_GUARD_RY)
        return (jax.lax.with_sharding_constraint(psi_v, x4),
                jax.lax.with_sharding_constraint(psi_v, y4),
                jax.lax.with_sharding_constraint(psi_c, x4),
                jax.lax.with_sharding_constraint(psi_c, y4),
                eps_v, eps_c)

    psi_v_X, psi_v_Y, psi_c_X, psi_c_Y, eps_v, eps_c = _split_pad(
        bundle.psi_rmu_Y, bundle.enk_full)
    M_X = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(psi_c_X, psi_v_X), x4)
    M_Y = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(psi_c_Y, psi_v_Y), y4)

    # ── V_Q exchange q=0 tile on the fine grid ────────────────────────────
    # A Q=0 exciton's exchange kernel is DENSE in (k,k') through the ONE q=0
    # tile (bse_serial.apply_bse_hamiltonian_single_device); so the fine
    # k-grid's exchange q-set is just q=0.
    #
    # WHAT IS AND IS NOT GRID-DEPENDENT.  The q=0 exchange BODY
    # V_{μν} = Σ_{G≠0} conj(ζ̃_{0,μ}(G)) v(G) ζ̃_{0,ν}(G) is built from the
    # centroids and the G-sphere alone — it never sees the k-grid, so it needs
    # no densification and the coarse bundle's tile is already the fine grid's
    # answer.  The ONE k-grid-dependent piece is the rank-1 HEAD scalar, whose
    # mini-BZ cell average <v>_mBZ is taken over BZ/N_k and therefore shrinks
    # coarse→fine.  That rescale is what the original rebuild (964c682b,
    # 2026-07-20) was after, and its comment said so.
    #
    # It bought the rescale by forcing ``head_minibz_average=True`` and
    # REPLACING the whole tile, which cost two things the deck never agreed to:
    # (i) the opt-in mini-BZ head average — documented in KNOWN_FAILURES as
    # averaging the WRONG MOMENT (a scalar <v> where the nonanalytic head needs
    # the 3×3 second moment <v q_a q_b>) — reached every coarse→fine user who
    # had not set the key; and (ii) the deck's exact disk body and its
    # loader-injected ``vhead`` rank-1 head (``bse_head._inject_q0_head``)
    # were discarded and substituted by an LR-only model reconstruction.
    #
    # Both now follow the deck's ``head_minibz_average`` key, default off, like
    # every other reader of that key in the tree (``exciton_bands.main``'s
    # ``head_mbz``, ``head_correction.HeadResolver``'s ``analytic_sphere``,
    # ``vq_interp.build_vq_evaluator``).  What the OFF arm gives
    # up is only the head RESCALE, and at Q=0 the head is annihilated by
    # ⟨u_ck|u_vk⟩ = 0 up to the ISDF orthogonality residual
    # (LT_HEAD_PROBLEM.md §2.1) — so a stale head SCALE is inert to that
    # residual, while a replaced BODY is not.
    head_mbz = bool(params.get("head_minibz_average", False))
    if head_mbz:
        # Opt-in: rebuild the tile with the FINE mini-BZ head via the SAME
        # vq_interp builder exciton_bands uses; the body is the b26p/stencil
        # model reproduced at the q=0 training point (run_nulls certifies the
        # reproduction).  This arm is bit-for-bit the pre-2026-08-09 behaviour.
        vqm = vq_interp.build_vq_evaluator(
            restart_file, mesh_xy, n_rmu_pad, head_minibz_average=True,
            distrib_la_batched_route=_distrib_la_batched_route,
            log_fn=log_fn)
        gstar, head_val = vq_interp.minibz_head_vlr(
            vqm.zx, vqm.prep, np.zeros(3), kgrid=fine_grid)
        V_q0 = vqm.eval_vq(jnp.zeros(3), vqm.prep["V_SRc"], vqm.pinvF,
                           vqm.coeffs_packed,
                           jnp.asarray(head_val, dtype=jnp.float64),
                           jnp.asarray(gstar, dtype=jnp.int32))
        V_q0 = 0.5 * (V_q0 + jnp.conj(V_q0).T)   # Hermitize (stencil residue)
        V_q0 = jax.device_put(V_q0, NamedSharding(mesh_xy, P("x", "y")))
        log_fn(f"[bse_k_grid] V_q0 exchange tile REBUILT via vq_interp "
               f"eval_vq(Q=0), fine mini-BZ head <v_LR>={head_val:.4f} "
               f"(gstar={gstar}) — head_minibz_average=true (deck opt-in).  "
               f"The deck's disk body and vhead rank-1 head are REPLACED; that "
               f"average is the scalar <v>, not the moment tensor the head "
               f"needs (tests/KNOWN_FAILURES.md).")
    else:
        V_q0 = data["V_q0"]
        log_fn("[bse_k_grid] V_q0 exchange tile CARRIED THROUGH from the "
               "coarse bundle unchanged (head_minibz_average off = the "
               "default): the q=0 body is k-grid-independent and the deck's "
               "loader-injected vhead rank-1 head is preserved exactly.  The "
               "head SCALE carried is the coarse mini-BZ <v>; at Q=0 the head "
               "is annihilated by ISDF orthogonality, so the stale scale is "
               "inert to the orthogonality residual (LT_HEAD_PROBLEM.md §2.1). "
               "Set head_minibz_average=true to rebuild with the fine mini-BZ "
               "head instead.")

    # ── W direct: coarse W_q → ifft(R) → zero-pad R → fine → fft back, all
    # inside the ONE sharded densifier (shard_map FFTs + jitted pad with
    # out_shardings) — the (μ,ν) sharding never drops, so per-rank peak is the
    # local (μ_loc, ν_loc, nk_fine) tile, never a replicated N_μ² array.
    densify_W = make_w_densifier(
        mesh_xy, P("x", "y", None, None, None), fine_grid, output="k")
    W_q_fine = densify_W(data["W_q"])
    log_fn(f"[bse_k_grid] W zero-padded in R "
           f"{coarse_grid[0]}x{coarse_grid[1]}x{coarse_grid[2]}→"
           f"{fine_grid[0]}x{fine_grid[1]}x{fine_grid[2]} "
           f"(exact trig-interp; direct term now on the fine grid)")

    # ── C1: re-attach W's Γ head, per fine q, on the densified BODY ────────
    # The loader DEFERRED the rank-1 whead injection when it saw a pending
    # densification (``_inject_q0_head(defer_whead=True)``), so ``data["W_q"]``
    # above is the pre-injection body and what the trig interpolant just
    # handled carried no Kronecker delta.  Now the head goes back on, with a
    # per-fine-q scalar from the one ratified integrand (gw.head_densify).
    #
    # ``head_channel`` is None on the ``legacy`` arm, where the loader injected
    # as before and the delta went through the interpolant — the documented
    # defect, kept only so the A/B has a shipped control.
    if head_channel is not None:
        from gw.head_densify import attach_head_channel
        S_fine = build_w_head_channel(
            wfn, sym, meta, params,
            coarse_grid=coarse_grid, fine_grid=fine_grid,
            whead=head_channel["whead"], ref_grid=coarse_grid,
            input_file=input_file, restart_file=restart_file,
            gamma_cell=head_channel.get("gamma_cell", "fine"), log_fn=log_fn)
        W_q_fine = attach_head_channel(
            W_q_fine, data["g0_X"], data["g0_Y"], S_fine,
            head_channel["cell_volume"])
        log_fn("[bse_k_grid] W Γ head re-attached AFTER densification (C1): "
               "the interpolant saw the body only, and the head is analytic "
               "at every fine q inside the coarse Γ cell.  This also changes "
               "the Davidson/FEAST preconditioner diagonal, which reads "
               "W_q[:,:,0,0,0] directly — correctness-neutral, convergence-"
               "relevant.")

    out = dict(data)
    out.update({
        "psi_c_X": psi_c_X, "psi_c_Y": psi_c_Y,
        "psi_v_X": psi_v_X, "psi_v_Y": psi_v_Y,
        "M_X": M_X, "M_Y": M_Y,
        "eps_c": eps_c, "eps_v": eps_v,
        "W_q": W_q_fine, "V_q0": V_q0,
        "V_q_full": None,                       # finite-q resolvent not a fine use
        # The finite-q EXCHANGE tiles survive the densification, on the COARSE
        # q-grid they were stored on, and they are exact there.  V_{μν}(q) is
        # built from ζ_q and the G-sphere; it never sees the BSE k-grid, so the
        # coarse tile IS the fine solve's answer at any Q that lies on the
        # coarse grid.  What the coarse restart cannot supply is a tile at a
        # fine q that is not also a coarse q — that needs an exchange model at
        # arbitrary q (``--vq-mode interp``), not an interpolation of this
        # array, and no such tile is invented here.
        #
        # Kept under a SEPARATE key with its own grid stamp rather than left in
        # ``V_q_full``: every other consumer of ``V_q_full`` (bse_w_exact's
        # finite-q resolvent) indexes it with the bundle's own grid and would
        # silently read the wrong tile on a densified bundle.  ``None`` there is
        # the refusal those consumers already have; this key is opt-in and
        # carries the grid that indexes it.
        "V_q_coarse": data["V_q_full"],
        "V_q_coarse_grid": coarse_grid,
        "nkx": fine_grid[0], "nky": fine_grid[1], "nkz": fine_grid[2],
    })
    return out
