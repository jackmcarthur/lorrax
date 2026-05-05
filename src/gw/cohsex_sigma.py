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
    return jax.device_put(jnp.asarray(Gij), NamedSharding(mesh_xy, P(None, None, None)))


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

    All arrays are sharded consistently with their upstream kernels.
    """
    if Gij is None:
        Gij = build_Gij(meta, mesh_xy)

    kgrid = meta.kgrid
    nk_tot = int(meta.nk_tot)
    sigma_sx_k, sigma_coh_k, hartree_k = _make_cohsex_kernels(
        mesh_xy, kgrid, nk_tot)

    with mesh_xy:
        sig_sx  = sigma_sx_k(wfns, Gij, W_q)
        sig_coh = sigma_coh_k(wfns, W_q, V_q)
        sig_h   = hartree_k(wfns, Gij, V_q)
        sig_sx, sig_coh = _add_static_head(
            sig_sx, sig_coh,
            static_head_terms=static_head_terms,
            meta=meta, mesh_xy=mesh_xy, do_screened=do_screened)
        sig_sx.block_until_ready()
        sig_coh.block_until_ready()
        sig_h.block_until_ready()

    sig_x = None
    if compute_bare_x:
        with mesh_xy:
            sig_x = sigma_sx_k(wfns, Gij, V_q)
        sig_x.block_until_ready()
        if static_head_terms is not None:
            x_head, _ = static_head_terms_to_kij(
                static_head_terms, nk_tot=meta.nk_tot, do_screened=False)
            rep = NamedSharding(mesh_xy, P(None, None, None))
            sig_x = sig_x + jax.device_put(x_head, rep)

    return {
        "sig_sx":  sig_sx,
        "sig_coh": sig_coh,
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


# ---------------------------------------------------------------------------
# Bispinor (DHF / DHFB) bare-X self-energy
# ---------------------------------------------------------------------------
#
# For 4-spinor wavefunctions the bare-X is decomposed in Lorentz channels
# by inserting γ̃^{μ_L} ≡ γ^0 γ^{μ_L} between the conjugated and direct
# ψ at each centroid:
#
#    Σ_X^B(k, m, n) = -Σ_{q, n_occ} Σ_{μ_L, ν_L, μ_X, μ_Y}
#       [ψ*_{m,k,α}(μ_X) γ̃^{μ_L}_{αβ} ψ_{n_occ,k-q,β}(μ_X)]
#     × V^{μ_L, ν_L}_q(μ_X, μ_Y)
#     × [ψ*_{n_occ,k-q,γ}(μ_Y) γ̃^{ν_L}_{γδ} ψ_{n,k,δ}(μ_Y)]
#
# Of the 16 (μ_L, ν_L) pairs, six (the (0,i) / (i,0) channels for
# i ∈ {1,2,3}) vanish by Coulomb gauge — the V_q lorentz driver
# returns 10 non-zero tiles in a dict.
#
# Mapping to the existing ``sigma_sx`` machinery:  ``build_G`` already
# emits ``G_base[k_d, β, μ_X, γ, μ_Y] = Σ_{n_occ} ψ_{n_occ}(β, μ_X)
# ψ*_{n_occ}(γ, μ_Y)`` with two FREE spinor axes (β, γ), one per side.
# The γ̃ vertices act linearly on those axes:
#
#    G_lorentz[μ_L, ν_L](k_d, α, μ_X, δ, μ_Y) =
#        γ̃^{μ_L}_{αβ} · G_base[k_d, β, μ_X, γ, μ_Y] · γ̃^{ν_L}_{γδ}
#
# Then ``_convolve(G_lorentz, V^{μ_L, ν_L}, +1)`` and ``_project`` (with
# the same outer ψ_xr / ψ_yn) gives that tile's contribution.  Sum over
# all 10 tiles and negate.
#
# Centroid-set selection per tile.  μ_L=0 was fit on the CHARGE
# centroid set (kmeans on ρ_charge), μ_L ∈ {1,2,3} on the CURRENT
# centroid set (kmeans on |j^Gordon|²).  The tile's left side uses the
# bundle matching μ_L, the right side the bundle matching ν_L:
#
#    (0,0): wfns_charge × wfns_charge   — 1 tile
#    (i,j) for i,j ∈ {1,2,3}:           — 9 tiles, BOTH sides current
#
# (No mixed-bundle tile exists because (0,i)/(i,0) are zero by gauge.)
# Each tile's ``build_G`` therefore reads ψ from a SINGLE bundle, and
# the per-(μ_X, μ_Y) contractions stay local on the ('x','y') sharding.

# Tile classification.  (0,0) uses charge bundle on both sides; the
# transverse 9 tiles use current bundle on both sides.
_BISPINOR_TILES_CHARGE: tuple[tuple[int, int], ...] = ((0, 0),)
_BISPINOR_TILES_CURRENT: tuple[tuple[int, int], ...] = (
    (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 2), (2, 3),
    (3, 1), (3, 2), (3, 3),
)


def _gamma_tilde_4x4(mu_lorentz: int) -> jax.Array:
    """Return γ̃^{μ_L} as a (4, 4) c128 jnp array.  Re-uses the
    convention from :mod:`common.gamma_matrices` (matrices stored
    are already γ^0 γ^μ; see header note in that module)."""
    from common.gamma_matrices import gamma0, gamma1, gamma2, gamma3
    return (gamma0, gamma1, gamma2, gamma3)[int(mu_lorentz)]


def _make_sigma_x_lorentz_kernel(mesh_xy: Mesh, kgrid, nk_tot: int):
    """Cached factory: bispinor bare-X kernel for ONE (μ_L, ν_L) tile.

    Returns ``f(wfns, Gij, V_block, gamma_mu_L, gamma_nu_L) → Σ contribution``
    that sums into the running total.  Shares the convolve/project
    primitives with :func:`_make_cohsex_kernels`; the only difference
    is the γ̃ multiplication on the (β, γ) axes of ``build_G``'s output
    before the convolve.
    """
    cache_key = (id(mesh_xy), tuple(int(x) for x in kgrid))
    if cache_key in _sigma_x_lorentz_cache:
        return _sigma_x_lorentz_cache[cache_key]

    from common.fft_helpers import make_flat_k_fftn, make_flat_k_ifftn

    _G_ifftn = make_flat_k_ifftn(mesh_xy, kgrid, G_FFT7D_SPEC, norm='ortho')
    _G_fftn  = make_flat_k_fftn( mesh_xy, kgrid, G_FFT7D_SPEC, norm='ortho')
    _V_ifftn = make_flat_k_ifftn(mesh_xy, kgrid, V_FFT5D_SPEC, norm='ortho')
    _inv_sqrt_nk = -1.0 / jnp.sqrt(float(nk_tot))

    @jax.jit
    def _sigma_x_lorentz_one_tile(
        psi_xn_sigma, psi_yr_sigma, psi_xr_sigma, psi_yn_sigma,
        Gij, V_block, gamma_mu, gamma_nu,
    ):
        """Single-tile contribution: build G, multiply γ̃, convolve, project.

        Inputs:
            psi_xn_sigma    (nk, s, μ_X, nb_sigma)   from bundle.xn(s.sigma)
            psi_yr_sigma    (nk, nb_sigma, s, μ_Y)   from bundle.yr(s.sigma)
            psi_xr_sigma    (nk, nb_sigma, s, μ_X)   from bundle.xr(s.sigma)
            psi_yn_sigma    (nk, s, μ_Y, nb_sigma)   from bundle.yn(s.sigma)
            Gij             (nk, nb_sigma, nb_sigma) occupation projector
            V_block         (nq, μ_X, μ_Y)           single Lorentz tile of V
            gamma_mu, gamma_nu   (4, 4) c128         γ̃^{μ_L}, γ̃^{ν_L}

        Returns Σ contribution shape (nk, m, n) — the SIGNED contribution
        for this tile (caller accumulates and applies the overall −1).
        """
        # G_base[k, β, μ_X, γ, μ_Y] = Σ_{i,j∈band} ψ(k, β, μ_X, i) Gij(k,i,j)
        #                                          ψ*(k, j, γ, μ_Y)
        G_base = build_G(psi_xn_sigma, psi_yr_sigma, Gij=Gij)

        # Insert γ̃ on the (β, γ) axes:
        # G_lorentz[k, α, μ_X, δ, μ_Y]
        #   = Σ_{β, γ} γ̃^{μ_L}_{αβ} G_base[k, β, μ_X, γ, μ_Y] γ̃^{ν_L}_{γδ}
        G_lorentz = jnp.einsum(
            'ab, kbxgy, gd -> kaxdy',
            gamma_mu, G_base, gamma_nu,
            optimize=True)

        # Convolution Σ = +FFT[ G_lorentz(R) · V_block(R) / √Nk ]
        # (positive prefactor; the overall −1 sits at the call site).
        G_R = _G_ifftn(G_lorentz)
        V_R = _V_ifftn(V_block)[:, None, :, None, :]
        sigma_k = _G_fftn(G_R * V_R * _inv_sqrt_nk)

        # _project to band basis:
        # Σ(k, m, n) = Σ_{α, μ_X, δ, μ_Y} ψ*_{m,k,α}(μ_X) σ(k,α,μ_X,δ,μ_Y)
        #                                  ψ_{n,k,δ}(μ_Y)
        return _project(psi_xr_sigma, psi_yn_sigma, sigma_k)

    _sigma_x_lorentz_cache[cache_key] = _sigma_x_lorentz_one_tile
    return _sigma_x_lorentz_one_tile


_sigma_x_lorentz_cache: dict[tuple[object, ...], object] = {}


def compute_sigma_x_lorentz(
    wfns_charge,
    wfns_current,
    V_blocks: dict,
    meta,
    mesh_xy: Mesh,
    *,
    Gij: jax.Array | None = None,
    print_fn=print,
) -> jax.Array:
    """Bispinor bare-X self-energy Σ_X^B summed over 10 (μ_L, ν_L) tiles.

    Parameters
    ----------
    wfns_charge
        Wavefunction bundle on the CHARGE centroid set (n_rmu_0); used
        for the (0,0) tile.
    wfns_current
        Wavefunction bundle on the CURRENT centroid set (n_rmu_i); used
        for the 9 transverse tiles (i,j) with i,j ∈ {1,2,3}.
    V_blocks
        ``dict[(μ_L, ν_L), Array]`` from
        ``v_q_lorentz.compute_all_V_q_lorentz_sharded`` reshaped to
        kgrid-shape ``(nkx, nky, nkz, n_rmu_L, n_rmu_R)``.
    meta, mesh_xy, Gij
        Standard COHSEX inputs.

    Returns
    -------
    sig_x   (nk, nb_sigma, nb_sigma)  bispinor bare exchange Σ_X^B.
    """
    if Gij is None:
        Gij = build_Gij(meta, mesh_xy)
    if wfns_charge is None or wfns_current is None:
        raise ValueError(
            "compute_sigma_x_lorentz requires both charge and current "
            "wavefunction bundles; call prepare_isdf_and_wavefunctions "
            "with cfg.bispinor=True so wfns_current is built.")

    kernel = _make_sigma_x_lorentz_kernel(mesh_xy, meta.kgrid, int(meta.nk_tot))

    # Pre-stage γ̃ matrices on the mesh (replicated, tiny).
    rep = NamedSharding(mesh_xy, P())
    gammas = {mu_L: jax.device_put(_gamma_tilde_4x4(mu_L), rep)
              for mu_L in (0, 1, 2, 3)}

    # Reshape every V_blocks[(μ_L, ν_L)] from kgrid-shape to flat-q for
    # the (k, μ, μ) → 5D V_q FFT helper.  V_blocks were stored kgrid-shape
    # by compute_V_q so the restart writer could consume them; the FFT
    # helper wants flat-q.
    nkx, nky, nkz = meta.kgrid
    nq = nkx * nky * nkz

    sig_x = None
    contributions: dict[tuple[int, int], float] = {}

    def _consume_tile(mu_L: int, nu_L: int, bundle):
        """Add tile (μ_L, ν_L)'s contribution to the running sum."""
        nonlocal sig_x
        block = V_blocks[(mu_L, nu_L)]
        # flatten kgrid axes back to flat-q for the V_FFT5D_SPEC helper
        block_flat = block.reshape(nq, *block.shape[-2:])
        s = bundle.slices
        contrib = kernel(
            bundle.xn(s.sigma), bundle.yr(s.sigma),
            bundle.xr(s.sigma), bundle.yn(s.sigma),
            Gij, block_flat,
            gammas[mu_L], gammas[nu_L])
        contrib.block_until_ready()
        try:
            tr = float(jnp.einsum('kmm->', contrib).real)
        except Exception:
            tr = float('nan')
        contributions[(mu_L, nu_L)] = tr
        sig_x = contrib if sig_x is None else (sig_x + contrib)

    with mesh_xy:
        for mu_L, nu_L in _BISPINOR_TILES_CHARGE:
            _consume_tile(mu_L, nu_L, wfns_charge)
        for mu_L, nu_L in _BISPINOR_TILES_CURRENT:
            if (mu_L, nu_L) not in V_blocks:
                continue
            _consume_tile(mu_L, nu_L, wfns_current)

    # The overall Σ_X = −∫ G·V sign is already baked into the per-tile
    # _convolve via ``_inv_sqrt_nk = -1/√nk`` (matches the existing
    # sigma_sx_k convention).  Don't double-negate sig_x here.

    # Diagnostic dump.
    print_fn(f"\n  Σ_X^B (bispinor) per-tile diagonal trace contributions:")
    for (m, n), tr in sorted(contributions.items()):
        print_fn(f"      ({m},{n}): tr Σ = {tr:+.6f} eV")
    return sig_x


def compute_cohsex_sigma_bispinor(
    wfns_charge,
    wfns_current,
    V_blocks: dict,
    W_q,                     # ignored when do_screened=False
    meta,
    mesh_xy: Mesh,
    *,
    Gij: jax.Array | None = None,
    do_screened: bool = False,
    static_head_terms=None,
    compute_bare_x: bool = True,
    print_fn=print,
) -> dict:
    """Bispinor counterpart of :func:`compute_cohsex_sigma`.

    For the x_only path (``do_screened=False``) we only assemble Σ_X^B
    (the 10-tile bispinor bare exchange) and the standard Hartree term
    on the charge bundle.  Σ_SX^B / Σ_COH^B require the screened W to
    also be supplied as a 10-tile dict — wiring up screened bispinor
    is the next step after this lands.  For now the screened branch
    raises if requested.
    """
    if do_screened:
        raise NotImplementedError(
            "Bispinor screened (Σ_SX, Σ_COH) requires W as a 10-tile "
            "Lorentz dict — not yet implemented.  Use do_screened=False "
            "(x_only path) for now.")
    if Gij is None:
        Gij = build_Gij(meta, mesh_xy)

    # Hartree V_H = ⟨n| V(q=0) ρ |n⟩.  In the bispinor pipeline ρ is
    # the charge density ψ̄ γ̃^0 ψ = ψ† ψ (the (0,0) Lorentz channel),
    # so V_H couples only to ``V_blocks[(0,0)]`` and the CHARGE
    # centroid bundle.  Reuse the existing scalar Hartree kernel by
    # passing it the rank-3 V_q from the (0,0) tile (kgrid-shape) flat-
    # q-reshaped — same structure compute_cohsex_sigma feeds it in the
    # non-bispinor path.
    sigma_sx_k, sigma_coh_k, hartree_k = _make_cohsex_kernels(
        mesh_xy, meta.kgrid, int(meta.nk_tot))
    nq = int(meta.nk_tot)
    V00_kgrid = V_blocks[(0, 0)]
    V00_flat = V00_kgrid.reshape(nq, *V00_kgrid.shape[-2:])
    with mesh_xy:
        sig_h = hartree_k(wfns_charge, Gij, V00_flat)
        sig_h.block_until_ready()

    sig_x = None
    if compute_bare_x:
        sig_x = compute_sigma_x_lorentz(
            wfns_charge, wfns_current, V_blocks, meta, mesh_xy,
            Gij=Gij, print_fn=print_fn)
        if static_head_terms is not None:
            x_head, _ = static_head_terms_to_kij(
                static_head_terms, nk_tot=meta.nk_tot, do_screened=False)
            rep = NamedSharding(mesh_xy, P(None, None, None))
            sig_x = sig_x + jax.device_put(x_head, rep)

    return {
        "sig_sx":  None,
        "sig_coh": None,
        "sig_h":   sig_h,
        "sig_x":   sig_x,
    }
