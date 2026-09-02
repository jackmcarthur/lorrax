"""Bispinor bare exchange self-energy: Σ^B from the transverse V_q^{i,j} tiles.

Per the Phase-1 DHF + bare-Breit design (``BISPINOR_DHFB_DESIGN.md`` §3),
the bispinor bare exchange splits into two pieces:

* **Σ^C-bare** — charge channel, uses ``V_qmunu_CC`` (== V_00).  This is
  identical to today's scalar ``compute_cohsex_sigma(..., compute_bare_x=
  True)``; the bispinor pipeline doesn't change anything here, just runs
  on 4-spinor wavefunctions instead of 2-spinor.

* **Σ^B** — transverse-only, summed over (i, j) ∈ {1, 2, 3}²:

    Σ^B_{αβ}(12) = -Σ_{i,j} γ̃^i_{αγ} G^0_{γδ}(12) γ̃^j_{δβ} D^{ij}_bare(12)

  where γ̃^μ ≡ γ^0 γ^μ (so γ̃^0 = I_4 and γ̃^i = α^i in the LORRAX
  convention), and D^{ij}_bare = V_qmunu_TT_{ij} = -v P_T^{ij}.  The
  Lorentz-metric minus belongs to the propagator builder; this Sigma
  contraction carries no compensating sign.

This module computes Σ^B by reusing the existing scalar
``sigma_sx`` kernel.  The trick:

* **γ̃^i insertion at the LEFT (G-build direct) vertex** and
* **γ̃^j insertion at the RIGHT (G-build conjugated) vertex**

are both performed by :func:`gw.wavefunction_bundle.with_lorentz_vertices`
— a REPRESENTATION-AWARE bundle operation (not this module's own code,
since 2026-08-23): it folds γ̃^i/γ̃^j into whichever pair of fields plays
the G-build's direct/conjugated role for ``wfns_transverse.layout``
(``psi_xn``/``psi_yr`` under the legacy four-copy carrier,
``psi_mun``/``psi_nmu`` under the two-face carrier — see that function's
own docstring for the field/axis table).  That is what lets this module
call ``_make_cohsex_kernels`` with the SAME layout dispatch every other
static Σ channel already uses (``gw.cohsex_sigma``), rather than only
ever working on a legacy bundle.

After the vertex rewrite, the existing kernel chain
``build_G → _convolve → project`` evaluates the formula above
verbatim — no changes to the kernel itself, in either layout.

The 9 (i, j) tiles in the sum decompose as 6 unique kernel calls
(3 diagonal + 3 upper-triangular) + 3 Hermitian-conjugate fills
delivered for free by :class:`gw.v_q_bispinor.BispinorVqReader`.

Public surface
--------------
* :func:`compute_sigma_x_bispinor` — orchestrator that opens
  ``v_q_bispinor.h5`` via ``BispinorVqReader``, loops the 9 transverse
  (i, j) pairs, and returns the summed Σ^B as a (nk, nb_sigma,
  nb_sigma) array on the same sharding as today's scalar Σ_X.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .v_q_bispinor import BispinorVqReader
from .wavefunction_bundle import (
    bundle_bytes_per_rank,
    face_kernel_kwargs,
    padded_centroid_extent,
    with_lorentz_vertices,
)


# Lorentz indices that contribute to Σ^B (transverse only).  The 6
# (0, i) / (i, 0) tiles vanish by Coulomb gauge and are skipped; the
# (0, 0) bare-CC tile is handled by the existing scalar
# compute_cohsex_sigma path.
_TRANSVERSE_INDICES = (1, 2, 3)


def compute_sigma_x_bispinor(
    *,
    wfns_transverse,
    Gij: jax.Array,
    bispinor_v_q_path: Path | str,
    meta,
    mesh_xy: Mesh,
    print_fn=print,
    verbose: bool = True,
) -> jax.Array:
    """Compute Σ^B (transverse-only bispinor bare exchange).

    Sums

        Σ^B[k, m, n] = -Σ_{i, j ∈ {1, 2, 3}} ⟨m, k| γ̃^i V^{i,j}_{q=k-k'}
                         γ̃^j |n, k⟩ ⟨m', k'| ⟨m, k|

    over the 9 transverse Lorentz pairs by reusing the existing
    ``sigma_sx`` kernel (with γ̃ folded into ψ; see module docstring).

    Parameters
    ----------
    wfns_transverse
        :class:`gw.wavefunction_bundle.Wavefunctions` sampled at the
        **transverse** centroid set ``r_{μ_T}``.  All 9 transverse
        tiles share these centroids.  See :meth:`__init__` of the
        next agent's bispinor wfns plumbing for how this is built —
        load_wfns currently only emits one wfns bundle per centroid
        set, so ``prepare_isdf_and_wavefunctions`` needs a small
        extension to emit a second bundle on ``cents_curr_idx`` when
        ``cfg.bispinor`` and ``cfg.paths.centroids_file_current`` are
        set.

    Gij
        Band-space occupation projector (nk, nb_sigma, nb_sigma) —
        same operator the scalar Σ_X uses.

    bispinor_v_q_path
        Path to ``v_q_bispinor.h5`` written by
        :func:`gw.v_q_bispinor.compute_V_q_bispinor_g_flat_to_h5`.

    meta, mesh_xy
        Standard plumbing.

    Returns
    -------
    jax.Array
        Σ^B[k, m, n] REPLICATED, windowed to ``(nk, nb_sigma, nb_sigma)`` —
        the same shape/sharding as the scalar Σ_X, in EITHER layout.
        Under ``layout='legacy'`` this is what ``sigma_sx_k`` already
        returns (its accessors window internally); under ``layout=
        'face'`` this function gathers-then-windows its own summed
        ``(nk, nb_full, nb_full)`` result itself, mirroring
        ``compute_cohsex_sigma``'s identical gather-then-window sequence
        for its OTHER static channels — so a caller can add this
        function's return value to ``sig_x`` unconditionally, without a
        per-layout shape special-case of its own.
    """
    from .cohsex_sigma import _make_cohsex_kernels
    nk_tot = int(meta.nk_tot)
    # layout dispatch: face_kernel_kwargs(wfns_transverse) is {} under
    # layout='legacy' (dead-code-eliminated at trace time, byte-identical
    # to the pre-2026-08-23 unconditional call) and {"layout": "face",
    # "face_shape": ...} under layout='face' — the SAME kwargs helper
    # every other static Sigma channel uses (gw.cohsex_sigma._face_kwargs
    # is a thin alias of this same function).  This is what makes
    # sigma_sx_k route through greens_function_kernel.build_G(layout=...)
    # / common.contract_bands.contract_bands_block_reshard(layout=...)
    # for a face-layout wfns_transverse rather than only ever working on
    # the legacy carrier.
    sigma_sx_k, _ = _make_cohsex_kernels(
        mesh_xy, meta.kgrid, nk_tot, **face_kernel_kwargs(wfns_transverse))

    # Instrumented psi inventory disclosure (report §7's "disclose the
    # selected layout"): the transverse-centroid bundle DOUBLES whichever
    # psi inventory the primary bundle already carries.  MEASURED (not
    # modeled) from wfns_transverse's own arrays — see
    # wavefunction_bundle.bundle_bytes_per_rank's docstring for why this
    # lives here rather than in gw.gflat_memory_model.
    if verbose and jax.process_index() == 0:
        _psi_bytes = bundle_bytes_per_rank(wfns_transverse)
        _per_field = ", ".join(
            f"{k}={v / 1e9:.4f} GB" for k, v in _psi_bytes.items() if k != "total")
        print_fn(
            f"  Σ^B transverse ψ inventory (layout={wfns_transverse.layout!r}): "
            f"{_psi_bytes['total'] / 1e9:.4f} GB/rank ({_per_field}) — "
            f"doubles whichever primary-bundle psi inventory this run "
            f"already carries")

    # ψ is delivered at PADDED n_rmu (load_centroids_band_chunked rounds
    # to mesh-product); V tiles on disk are at LOGICAL extent.  Pad V to
    # match ψ's μ-axis so the convolve broadcasts correctly.  Pad rows
    # of ψ are zero (Phase 3a invariant), so zero-padding V is exact.
    # The extent :func:`_pad_V_to_padded` pads the on-disk V tile up to;
    # one owner (gw.wavefunction_bundle.padded_centroid_extent) for this
    # module, the packed response and the sixteen-block Sigma.
    n_rmu_T_padded = padded_centroid_extent(wfns_transverse)

    def _pad_V_to_padded(V_logical: jax.Array) -> jax.Array:
        n_l = int(V_logical.shape[-2])
        n_r = int(V_logical.shape[-1])
        if n_l == n_rmu_T_padded and n_r == n_rmu_T_padded:
            return V_logical
        return jnp.pad(
            V_logical,
            ((0, 0), (0, n_rmu_T_padded - n_l), (0, n_rmu_T_padded - n_r)),
        )

    sig_x_b = None
    contributions: dict[tuple[int, int], float] = {}
    with BispinorVqReader(bispinor_v_q_path, mesh_xy,
) as reader:
        for i in _TRANSVERSE_INDICES:
            for j in _TRANSVERSE_INDICES:
                wfns_ij = with_lorentz_vertices(wfns_transverse, i, j)
                V_ij = _pad_V_to_padded(reader.get_tile(i, j))
                if wfns_transverse.layout == "face":
                    # Face's two-array carrier serves BOTH the G-build's
                    # internal band sum AND the outer projection bra/ket
                    # (unlike legacy's four independent fields) — the
                    # G-build must read the γ̃-inserted operands
                    # (``wfns_ij``) while projection reads the ORIGINAL,
                    # un-rotated ones (``wfns_transverse``).
                    # ``sigma_sx``'s ``wfns_g=`` parameter exists
                    # precisely for this (gw.cohsex_sigma.
                    # _make_cohsex_kernels_face's own docstring).
                    contrib = sigma_sx_k(
                        wfns_transverse, Gij, V_ij, wfns_g=wfns_ij)
                else:
                    contrib = sigma_sx_k(wfns_ij, Gij, V_ij)
                contrib.block_until_ready()
                # Per-tile diagonal trace (eV) for diagnostic comparison
                # against agent-B's MoS2 reference values (commit 69e8863).
                from common import RYD_TO_EV
                try:
                    tr = float(jnp.einsum('kmm->', contrib).real) * RYD_TO_EV
                except Exception:
                    tr = float('nan')
                contributions[(i, j)] = tr
                if verbose and jax.process_index() == 0:
                    print_fn(f"  Σ^B tile (μ_L={i}, ν_L={j}): tr Σ = {tr:+.6f} eV")
                sig_x_b = contrib if sig_x_b is None else sig_x_b + contrib

    if sig_x_b is None:  # pragma: no cover — should never happen
        raise RuntimeError("compute_sigma_x_bispinor produced no contributions")

    if wfns_transverse.layout == "face":
        # sigma_sx_k(layout='face') never windows its own output (report
        # §3: a face-sharded band axis cannot be sliced to an arbitrary
        # logical window) — sig_x_b is still (nk, nb_full, nb_full),
        # 2-D sharded.  Gather to replicated FIRST (legal at any size),
        # THEN window to nb_sigma (a plain slice of a replicated array,
        # no divisibility constraint) — the SAME sequence
        # compute_cohsex_sigma/compute_sigma_x already use for their
        # OTHER static channels, so this function's return contract
        # matches theirs regardless of which layout built it.
        rep = NamedSharding(mesh_xy, P(None, None, None))
        sig_x_b = jax.lax.with_sharding_constraint(sig_x_b, rep)
        nb_sigma = wfns_transverse.slices.nb_sigma
        sig_x_b = sig_x_b[:, :nb_sigma, :nb_sigma]
        sig_x_b = jax.lax.with_sharding_constraint(sig_x_b, rep)

    sig_x_b.block_until_ready()
    return sig_x_b
