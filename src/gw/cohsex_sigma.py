"""Static COHSEX self-energy pipeline.

Builds Σ_SX, Σ_COH, and V_H (Hartree) in the ISDF basis, using
flat-k/flat-q sharding consistent with the rest of the GW stack.

    Σ_SX(k)  = -project[ FFT[ G_occ(R) * W(R) / √Nk ] ]
    Σ_COH(k) = +project[ FFT[ G_RI(R)  * (W − V)(R) / (2√Nk) ] ]
    V_H(k)   =  project[ V(q=0) * ρ ]

The screening operand is W for the COHSEX channel and V for bare
exchange — pass V as ``W_or_V_q`` to get Σ_X out of the same kernel.

The driver entry :func:`compute_cohsex_sigma` builds all three
contributions from a wavefunction bundle and flat-q V / W and returns
them as a dict.  Static head correction (q→0 band-diagonal terms) is
optional and applied to SX/COH (and to the bare-X pass separately).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .greens_function_kernel import build_G
from .head_correction import static_head_terms_to_kij
from .wavefunction_bundle import project as _project
from .wavefunction_bundle import G_FFT7D_SPEC, V_FFT5D_SPEC


# ---------------------------------------------------------------------------
# Gij — occupation projector in band space.  Static COHSEX uses the
# band-space occupation projector (not the τ-phase form used by chi0);
# kept alongside the COHSEX kernels because it's only consumed here.
# ---------------------------------------------------------------------------

def build_Gij(meta, mesh_xy: Mesh) -> jax.Array:
    """Occupation projector G_ij = diag(1,...,1,0,...,0) for sigma bands.

    ─ NOTE TO FUTURE EDITORS — THE numpy USAGE BELOW IS INTENTIONAL ─
    (nk, nb_sigma, nb_sigma) is a tiny host-side matrix (<1 MiB in
    every realistic case).  The all-``jnp`` version fired 8 standalone
    pjits per call (zeros, eye, dynamic_slice, scatter, convert) for
    zero runtime benefit.  Commit 7781b80 (2026-04-18) converted to
    numpy; the ``device_put`` at the end places it on the mesh.
    DO NOT "fix" back to ``jnp``.
    """
    nocc = min(meta.nelec, meta.nb_sigma)
    Gij = np.zeros((meta.nk_tot, meta.nb_sigma, meta.nb_sigma), dtype=np.complex128)
    Gij[:, :nocc, :nocc] = np.eye(nocc, dtype=np.complex128)
    return jax.device_put(Gij, NamedSharding(mesh_xy, P(None, None, None)))


# ---------------------------------------------------------------------------
# Kernel factory — one cached build produces all three static kernels
# (sigma_sx, sigma_coh, hartree) for a given (mesh, kgrid).  Chi0 and
# PPM sigma use the same factory pattern.
# ---------------------------------------------------------------------------

_cohsex_kernel_cache: dict[tuple[object, ...], tuple] = {}


def _make_cohsex_kernels(mesh_xy: Mesh, kgrid: tuple[int, int, int], nk_tot: int):
    """Cached factory: returns (sigma_sx, sigma_coh, hartree) jit'd kernels.

    Keyed on (id(mesh_xy), kgrid) — same shape the chi0 / ppm_sigma
    kernel caches use.  ``nk_tot`` = prod(kgrid) and is redundant for
    cache-lookup purposes; it stays as a positional arg because the
    Hartree kernel closes over it as a compile-time constant.
    """
    cache_key = (id(mesh_xy), tuple(int(x) for x in kgrid))
    if cache_key in _cohsex_kernel_cache:
        return _cohsex_kernel_cache[cache_key]

    from common.fft_helpers import make_flat_k_fftn, make_flat_k_ifftn

    _G_ifftn = make_flat_k_ifftn(mesh_xy, kgrid, G_FFT7D_SPEC, norm='ortho')
    _G_fftn  = make_flat_k_fftn( mesh_xy, kgrid, G_FFT7D_SPEC, norm='ortho')
    _V_ifftn = make_flat_k_ifftn(mesh_xy, kgrid, V_FFT5D_SPEC, norm='ortho')
    _inv_sqrt_nk = -1.0 / jnp.sqrt(float(nk_tot))

    @jax.jit
    def _convolve(G_k, V_or_W, prefactor):
        """Σ^k-space convolution Σ = pref · FFT[ G(R) · V(R) / √Nk ]."""
        G_R = _G_ifftn(G_k)
        V_R = _V_ifftn(V_or_W)[:, None, :, None, :]
        return prefactor * _G_fftn(G_R * V_R * _inv_sqrt_nk)

    @jax.jit
    def sigma_sx(wfns, Gij, W_q):
        """Screened exchange:  Σ_SX = -project[ FFT[ G_occ(R) · W(R) / √Nk ] ].

        Pass V_q in place of W_q to get bare exchange Σ_X.
        """
        s = wfns.slices
        G_occ = build_G(wfns.xn(s.sigma), wfns.yr(s.sigma), Gij=Gij)
        return _project(wfns.xr(s.sigma), wfns.yn(s.sigma),
                        _convolve(G_occ, W_q, 1.0))

    @jax.jit
    def sigma_coh(wfns, W_q, V_q):
        """Coulomb-hole:  Σ_COH = +project[ FFT[ G_RI(R) · (W-V)(R) / (2·√Nk) ] ]."""
        s = wfns.slices
        G_ri = build_G(wfns.xn(s.full), wfns.yr(s.full))
        return _project(wfns.xr(s.sigma), wfns.yn(s.sigma),
                        _convolve(G_ri, W_q - V_q, -0.5))

    @jax.jit
    def hartree(wfns, Gij, V_q):
        """V_H(m,n,k) = <m| V(q=0, no G0) · ρ |n>.  V_q flat-k (nk,μ,μ); uses V_q[0]."""
        s = wfns.slices
        psi_yr, psi_xr = wfns.yr(s.sigma), wfns.xr(s.sigma)
        rho = jnp.real(jnp.einsum(
            'kisx,kjsx,kij->x',
            jnp.conj(psi_yr), psi_yr, Gij, optimize=True))
        Vrho = jnp.einsum(
            'xy,y->x', V_q[0],
            rho / jnp.asarray(nk_tot, dtype=jnp.float64), optimize=True)
        return jnp.einsum(
            'kmsx,x,knsx->kmn',
            jnp.conj(psi_xr), Vrho, psi_xr, optimize=True)

    _cohsex_kernel_cache[cache_key] = (sigma_sx, sigma_coh, hartree)
    return sigma_sx, sigma_coh, hartree


# ---------------------------------------------------------------------------
# Static head addition — q→0 band-diagonal head correction for COHSEX.
# ---------------------------------------------------------------------------

def _add_static_head(sig_sx, sig_coh, *, static_head_terms, meta, mesh_xy,
                     do_screened: bool):
    """Add the q→0 head correction to SX/COH (no-op if terms is None)."""
    if static_head_terms is None:
        return sig_sx, sig_coh
    sx_h, coh_h = static_head_terms_to_kij(
        static_head_terms, nk_tot=meta.nk_tot, do_screened=do_screened)
    if not do_screened:
        coh_h = jnp.zeros_like(coh_h)
    rep = NamedSharding(mesh_xy, P(None, None, None))
    return (sig_sx + jax.device_put(sx_h, rep),
            sig_coh + jax.device_put(coh_h, rep))


# ---------------------------------------------------------------------------
# Top-level driver.
# ---------------------------------------------------------------------------

def compute_cohsex_sigma(
    wfns,
    V_q: jax.Array,
    W_q: jax.Array,
    meta,
    mesh_xy: Mesh,
    *,
    Gij: jax.Array | None = None,
    do_screened: bool = True,
    static_head_terms=None,
    compute_bare_x: bool = True,
    wfns_transverse=None,
    bispinor_v_q_path=None,
    backend=None,
    use_ffi_io: bool | None = None,
) -> dict:
    """Evaluate static COHSEX self-energy components.

    Parameters
    ----------
    wfns, V_q, W_q
        Wavefunction bundle and flat-q Coulomb / screened operands
        (nq, μ, μ).  Pass V_q for W_q when ``do_screened=False`` (the
        caller is responsible for that substitution; the Gij-based
        sx/coh channels don't test it themselves).
    Gij
        Band-space occupation projector (nk, nb_sigma, nb_sigma).
        If ``None``, built via :func:`build_Gij`.  Kept as a parameter
        so the SC-COHSEX loop can iterate on it.
    static_head_terms
        Optional q→0 head correction terms.  Applied to SX/COH and
        separately to the bare-X pass.
    compute_bare_x
        Whether to also compute Σ_X (bare exchange) using V_q.

    Returns
    -------
    dict with keys:
        sig_sx   (nk, nb_sigma, nb_sigma)  screened exchange + head
        sig_coh  (nk, nb_sigma, nb_sigma)  Coulomb hole + head (if screened)
        sig_h    (nk, nb_sigma, nb_sigma)  Hartree
        sig_x    (nk, nb_sigma, nb_sigma)  bare exchange + head, or None

    All four returned arrays are pinned to **fully-replicated** sharding
    ``P(None, None, None)`` so the post-self-energy plumbing in
    ``gw_jax`` can operate on replicated H_kmn without resharding seams.
    They are small (``nk · nb_sigma² · 16 B`` ≲ tens of MB) so replication
    is essentially free; the heavy ω-grid Σ_c tensor stays sharded
    upstream in ``ppm_sigma`` and is only collapsed into a replicated
    Σ_xc^QSGW after the energy-domain contraction.
    """
    if Gij is None:
        Gij = build_Gij(meta, mesh_xy)

    kgrid = meta.kgrid
    nk_tot = int(meta.nk_tot)
    sigma_sx_k, sigma_coh_k, hartree_k = _make_cohsex_kernels(
        mesh_xy, kgrid, nk_tot)

    rep = NamedSharding(mesh_xy, P(None, None, None))

    with mesh_xy:
        sig_sx  = sigma_sx_k(wfns, Gij, W_q)
        sig_coh = sigma_coh_k(wfns, W_q, V_q)
        sig_h   = hartree_k(wfns, Gij, V_q)
        sig_sx, sig_coh = _add_static_head(
            sig_sx, sig_coh,
            static_head_terms=static_head_terms,
            meta=meta, mesh_xy=mesh_xy, do_screened=do_screened)
        sig_sx  = jax.lax.with_sharding_constraint(sig_sx,  rep)
        sig_coh = jax.lax.with_sharding_constraint(sig_coh, rep)
        sig_h   = jax.lax.with_sharding_constraint(sig_h,   rep)
        sig_sx.block_until_ready()
        sig_coh.block_until_ready()
        sig_h.block_until_ready()

    sig_x = None
    if compute_bare_x:
        with mesh_xy:
            sig_x = sigma_sx_k(wfns, Gij, V_q)
        if static_head_terms is not None:
            x_head, _ = static_head_terms_to_kij(
                static_head_terms, nk_tot=meta.nk_tot, do_screened=False)
            sig_x = sig_x + jax.device_put(x_head, rep)
        sig_x = jax.lax.with_sharding_constraint(sig_x, rep)
        sig_x.block_until_ready()

        # Bispinor bare exchange: add Σ^B (transverse-only sum over
        # (i, j) ∈ {1, 2, 3}²) to sig_x.  No-op when ``wfns_transverse``
        # or ``bispinor_v_q_path`` is missing.  See
        # ``gw.sigma_x_bispinor`` and ``BISPINOR_DHFB_DESIGN.md`` §3.
        if wfns_transverse is not None and bispinor_v_q_path is not None:
            from .sigma_x_bispinor import compute_sigma_x_bispinor
            with mesh_xy:
                sig_x_b = compute_sigma_x_bispinor(
                    wfns_transverse=wfns_transverse,
                    Gij=Gij,
                    bispinor_v_q_path=bispinor_v_q_path,
                    meta=meta, mesh_xy=mesh_xy,
                    backend=backend, use_ffi_io=use_ffi_io,
                )
            sig_x_b.block_until_ready()
            sig_x = sig_x + sig_x_b

    return {
        "sig_sx":  sig_sx,
        "sig_coh": sig_coh,
        "sig_h":   sig_h,
        "sig_x":   sig_x,
    }


def compute_v_h_sigma_x(
    wfns,
    V_q: jax.Array,
    meta,
    mesh_xy: Mesh,
    *,
    Gij: jax.Array | None = None,
    static_head_terms=None,
    wfns_transverse=None,
    bispinor_v_q_path=None,
    backend=None,
    use_ffi_io: bool | None = None,
) -> dict:
    """Two-kernel V-only path: ``sig_h`` (Hartree) + ``sig_x`` (bare exchange).

    Skips the screened SX/COH kernels entirely — used by callers that
    don't need them (X_ONLY mode, and PPM modes via the dispatcher,
    which gets its dynamic Σ_c straight from
    :mod:`gw.ppm_pipeline`).  Each kernel is the same jit'd primitive
    used by :func:`compute_cohsex_sigma`, just called from a Python
    entry that won't ever invoke ``sigma_sx_k(W_q)`` or
    ``sigma_coh_k(W_q, V_q)`` and so saves two flat-q convolutions per
    call (≈ the ``W_q`` cost on each, roughly half the cohsex_sigma
    wall on dense band manifolds).

    Returned dict mirrors :func:`compute_cohsex_sigma`'s contract with
    ``sig_sx`` / ``sig_coh`` set to zero placeholders so downstream
    ``cohsex["sig_sx"]`` accesses don't have to special-case None.

    Bispinor: identical to ``compute_cohsex_sigma``'s ``compute_bare_x``
    branch — Σ^B is added to ``sig_x`` when both ``wfns_transverse``
    and ``bispinor_v_q_path`` are supplied.
    """
    if Gij is None:
        Gij = build_Gij(meta, mesh_xy)
    sigma_sx_k, _, hartree_k = _make_cohsex_kernels(
        mesh_xy, meta.kgrid, int(meta.nk_tot))
    rep = NamedSharding(mesh_xy, P(None, None, None))

    with mesh_xy:
        sig_h = hartree_k(wfns, Gij, V_q)
        sig_x = sigma_sx_k(wfns, Gij, V_q)
        sig_h = jax.lax.with_sharding_constraint(sig_h, rep)
        sig_x = jax.lax.with_sharding_constraint(sig_x, rep)
        sig_h.block_until_ready()
        sig_x.block_until_ready()

    if static_head_terms is not None:
        x_head, _ = static_head_terms_to_kij(
            static_head_terms, nk_tot=meta.nk_tot, do_screened=False)
        sig_x = sig_x + jax.device_put(x_head, rep)
        sig_x = jax.lax.with_sharding_constraint(sig_x, rep)
        sig_x.block_until_ready()

    if wfns_transverse is not None and bispinor_v_q_path is not None:
        from .sigma_x_bispinor import compute_sigma_x_bispinor
        with mesh_xy:
            sig_x_b = compute_sigma_x_bispinor(
                wfns_transverse=wfns_transverse,
                Gij=Gij,
                bispinor_v_q_path=bispinor_v_q_path,
                meta=meta, mesh_xy=mesh_xy,
                backend=backend, use_ffi_io=use_ffi_io,
            )
        sig_x_b.block_until_ready()
        sig_x = sig_x + sig_x_b

    zero_kij = jnp.zeros_like(sig_x)
    return {
        "sig_sx":  zero_kij,
        "sig_coh": zero_kij,
        "sig_h":   sig_h,
        "sig_x":   sig_x,
    }


def get_cohsex_kernels(meta, mesh_xy: Mesh):
    """Return the three jit'd (sigma_sx, sigma_coh, hartree) kernels.

    Exposed for the SC-COHSEX fixed-point loop, which needs to call the
    kernels repeatedly with a mutated Gij inside its own jit/mixing
    harness.  New callers should use :func:`compute_cohsex_sigma`.
    """
    return _make_cohsex_kernels(mesh_xy, meta.kgrid, int(meta.nk_tot))
