"""Contract static screened exchange and Coulomb hole on canonical parent wavefunctions."""
from __future__ import annotations

from functools import partial

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

def _spin_capacity(meta) -> float:
    """Electrons per occupied band, from meta's DECLARED spin structure.

    Delegates to the ONE canonical helper
    (``psp.get_DFT_mtxels.spin_degeneracy_factor`` — ``Meta`` duck-types
    the two attrs it reads, and bispinor meta.nspinor=4 yields exactly
    1.0, like nspinor=2).  The presence check exists because the helper
    getattr-defaults a missing attr to 1, which would hand a meta that
    never declared its spin structure the SCALAR factor 2 *silently* —
    the V_H it scales is a ~500 eV quantity, so refuse loudly instead.
    """
    for attr in ("nspin", "nspinor"):
        if not hasattr(meta, attr):
            raise AttributeError(
                f"cohsex_sigma: meta.{attr} is missing — the Hartree ρ and "
                f"the fixed-N window check are capacity-weighted (2 e⁻ per "
                f"band iff nspin == nspinor == 1), so meta must declare its "
                f"spin structure (got a {type(meta).__name__} without it).  "
                f"Fix: pass a common.meta.Meta, or give the stub explicit "
                f"nspin/nspinor.")
    # Computed from meta's own declared file-spin structure.  NOT delegated
    # to psp.get_DFT_mtxels.spin_degeneracy_factor: that helper is now the
    # loader's occupation-capacity door (wfn.occupation_state_capacity) and
    # takes a WfnLoader, not a Meta.  nspinor_wfnfile keeps bispinor
    # (meta.nspinor=4, file 2) at exactly 1.0.
    return 2.0 if (int(meta.nspin) == 1
                   and int(getattr(meta, "nspinor_wfnfile", meta.nspinor))
                   == 1) else 1.0


def build_Gij(meta, mesh_xy: Mesh, occupation_state=None) -> jax.Array:
    """Occupation projector G_ij = diag(1,...,1,0,...,0) for sigma bands.

    **Band coverage — the Hartree density is complete regardless of
    ``nval``.**  The Σ slice this multiplies is
    ``BandSlices.sigma = slice(0, b3 - b0)``, i.e. global bands
    ``[b0, b3) = [0, nelec + ncond)``, so the ``nocc = min(nelec,
    nb_sigma) = nelec`` rows set to 1 below are global bands
    ``[0, nelec)`` — *every* occupied band, including deep semicore, for
    any ``nval``.  (A second, unused ``Meta.band_ranges.sigma = (b1, b3)``
    convention used to exist and suggested — falsely — that a deck with
    ``nval < nelec`` drops occupied bands out of ρ.  It is deleted;
    ``BandSlices`` is the only band-window source.)

    ─ NOTE TO FUTURE EDITORS — THE numpy USAGE BELOW IS INTENTIONAL ─
    (nk, nb_sigma, nb_sigma) is a tiny host-side matrix (<1 MiB in
    every realistic case).  The all-``jnp`` version fired 8 standalone
    pjits per call (zeros, eye, dynamic_slice, scatter, convert) for
    zero runtime benefit.  Commit 7781b80 (2026-04-18) converted to
    numpy; the ``device_put`` at the end places it on the mesh.
    DO NOT "fix" back to ``jnp``.
    """
    # With an occupation state (duck-typed gw.efermi.OccupationState) the
    # projector is diag(f) — exact for insulators too: step occupations are
    # exactly {0.0, 1.0}, so diag(f) == the eye(nocc) block bit-for-bit
    # (asserted in tests/test_sigma_fermi_split.py).  Weights are never
    # clipped: an MP-overshoot band (f<0 or f>1) contributes with its sign.
    if occupation_state is not None:
        f = np.asarray(occupation_state.f_kn, dtype=np.float64)
        f = f.reshape(int(meta.nk_tot), -1)
        if f.shape[1] < int(meta.nb_sigma):
            raise ValueError(
                f"build_Gij: occupation state carries {f.shape[1]} bands but "
                f"the sigma window needs {int(meta.nb_sigma)}.")
        f_win = f[:, : int(meta.nb_sigma)]
        # The metallic form of the same V_H-silently-small hazard the
        # integer guard below prevents: electrons carried by bands outside
        # the Σ window would silently leave the Hartree density.
        # ``n_electrons`` is capacity-weighted (the efermi solvers count
        # f × spin_degeneracy_factor), so the band sum must be too: 2
        # electrons per band on a spin-restricted scalar WFN, exactly 1.0
        # for nspinor ≥ 2 (bispinor meta.nspinor=4 included).
        n_win = (_spin_capacity(meta)
                 * float(np.sum(f_win)) / float(meta.nk_tot))
        n_target = float(occupation_state.n_electrons)
        if abs(n_win - n_target) > 1.0e-8:
            raise ValueError(
                f"build_Gij: the sigma window holds {n_win:.10f} electrons "
                f"but the occupation state solved for {n_target:.10f}.  The "
                "Hartree density would be missing weight carried by bands "
                "outside the window; widen nb_sigma or re-solve.")
        Gij = np.zeros((meta.nk_tot, meta.nb_sigma, meta.nb_sigma),
                       dtype=np.complex128)
        idx = np.arange(int(meta.nb_sigma))
        Gij[:, idx, idx] = f_win.astype(np.complex128)
        from common.collectives import device_put_process_local
        return device_put_process_local(
            Gij, NamedSharding(mesh_xy, P(None, None, None)))
    # Integer path (occupation_state None) — unchanged, bit-exact.
    # The coverage claim above, enforced rather than merely asserted in
    # prose: if the Σ window were ever narrower than the occupied
    # manifold, ``min`` would silently drop occupied bands out of ρ and
    # V_H would come out systematically small with no other symptom.
    # ``nb_sigma = nelec + ncond`` makes this unreachable for ncond >= 0;
    # the guard exists so a future band-window change cannot reintroduce
    # it quietly.
    if int(meta.nb_sigma) < int(meta.nelec):
        raise ValueError(
            f"build_Gij: sigma window has {int(meta.nb_sigma)} bands but the "
            f"system has {int(meta.nelec)} occupied bands.  The Hartree "
            f"density would be missing {int(meta.nelec) - int(meta.nb_sigma)} "
            "occupied bands, which no centroid count can repair.")
    nocc = min(meta.nelec, meta.nb_sigma)
    Gij = np.zeros((meta.nk_tot, meta.nb_sigma, meta.nb_sigma), dtype=np.complex128)
    Gij[:, :nocc, :nocc] = np.eye(nocc, dtype=np.complex128)
    # Process-local placement, NOT plain ``jax.device_put``: on a
    # multi-process mesh the latter silently runs multihost
    # ``assert_equal`` — a P-linear all-gather of the operand (scorecard
    # AA.1).  Gij is a pure function of (nk, nb_sigma, nelec) —
    # bit-identical on every rank by construction (np.eye block, no
    # roundoff).  LORRAX_CHECK_REPLICA=1 restores the assertion.
    from common.collectives import device_put_process_local
    return device_put_process_local(
        Gij, NamedSharding(mesh_xy, P(None, None, None)))


def _resolve_Gij(Gij, meta, mesh_xy: Mesh, occupation_state):
    """The ONE place the static Σ entries decide their occupation projector.

    ``Gij`` supplied by the caller still wins — the SC-COHSEX loop iterates
    on its own projector.  But a caller handing in BOTH a projector and an
    occupation state is asking for two occupation models in one Σ, and the
    state is the one that would be silently dropped (the exact class of
    defect this threading exists to close).  Refuse instead; TASTE 13.
    """
    if Gij is None:
        return build_Gij(meta, mesh_xy, occupation_state)
    if occupation_state is not None:
        raise ValueError(
            "static Sigma: both an explicit Gij and an occupation_state "
            "were supplied.  The explicit projector would silently ignore "
            "the state's diag(f) weights; pass one or the other.")
    return Gij


# ---------------------------------------------------------------------------
# Kernel factory — one cached build produces all three static kernels
# (sigma_sx, sigma_coh, hartree) for a given (mesh, kgrid).  Chi0 and
# PPM sigma use the same factory pattern.
# ---------------------------------------------------------------------------

def _occ_diag_full(Gij, nb_sigma, nb_full):
    """(nk, nb_sigma, nb_sigma) diagonal occupation matrix -> (nk,
    nb_full) COMPLEX weight vector, zero-padded outside [0, nb_sigma).
    Every production Gij (integer or diag(f), :func:`build_Gij`) is
    diagonal by construction — obstacle #4's "carry the occupation vector
    as the common path".  This reads that diagonal rather than doing the
    O(nb_sigma^2) contraction the face path exists to avoid; it does not
    detect a genuinely dense (off-diagonal) Gij, which
    :func:`greens_function_kernel.build_G` already refuses by name for
    face layout before this is reached.

    Module-level (not a closure) since 2026-08-22: shared by this
    module's own static kernels (below) AND ``gw.ppm_sigma``'s
    invalid-pole static-limit term, which builds the identical face-G
    occupation weight for the SAME reason (single-source-of-truth
    microservice rule) — see ``gw.ppm_sigma._compute_invalid_static_sigma``.
    """
    if nb_sigma > nb_full:
        raise ValueError(
            f"_occ_diag_full: nb_sigma={nb_sigma} exceeds nb_full={nb_full}")
    diag = jnp.diagonal(Gij, axis1=1, axis2=2)   # (nk, nb_sigma)
    return jnp.pad(diag, ((0, 0), (0, nb_full - nb_sigma)))


def _face_kwargs(wfns) -> dict:
    """Select the canonical Sigma face shapes and typed parent transport."""
    from .wavefunction_bundle import sigma_face_kernel_kwargs
    return sigma_face_kernel_kwargs(wfns)


_cohsex_kernel_cache: dict[tuple[object, ...], tuple] = {}
_static_convolution_cache: dict[tuple[object, ...], object] = {}


def _make_static_convolution(mesh_xy: Mesh, kgrid: tuple[int, int, int],
                             nk_tot: int, *, q0_only=False, lorentz=False):
    """Own the normalized flat-k convolution for scalar and streamed Lorentz sums."""
    from ffi import ffi_dial_key
    from ffi.mklfft import fused_fft_ffi_enabled
    from common.fft_helpers import make_flat_k_gw_conv
    key = (id(mesh_xy), tuple(kgrid), ffi_dial_key(), int(nk_tot), q0_only, lorentz)
    if key in _static_convolution_cache:
        return _static_convolution_cache[key]
    scale = -1.0 / (float(nk_tot) if q0_only else np.sqrt(float(nk_tot)))
    if lorentz:
        convolve = _make_lorentz_convolution(mesh_xy, kgrid, scale, q0_only)
    elif q0_only:
        @jax.jit
        def convolve(G_k, interaction, prefactor):
            return prefactor * G_k * interaction[0][None, None, :, None, :] * scale
    elif fused_fft_ffi_enabled():
        # The fused owner bounds the exposed Green lifetime on large scalar decks.
        fused = make_flat_k_gw_conv(mesh_xy, kgrid, G_FFT7D_SPEC, V_FFT5D_SPEC,
                                    norm='ortho', mult=scale)
        @jax.jit
        def convolve(G_k, interaction, prefactor):
            return prefactor * fused(G_k, interaction)
    else:
        from common.fft_helpers import make_flat_k_fftn, make_flat_k_ifftn
        inverse_g = make_flat_k_ifftn(mesh_xy, kgrid, G_FFT7D_SPEC, norm='ortho')
        forward_g = make_flat_k_fftn(mesh_xy, kgrid, G_FFT7D_SPEC, norm='ortho')
        inverse_v = make_flat_k_ifftn(mesh_xy, kgrid, V_FFT5D_SPEC, norm='ortho')
        @jax.jit
        def convolve(G_k, interaction, prefactor):
            return prefactor * forward_g(
                inverse_g(G_k) * inverse_v(interaction)[:, None, :, None, :] * scale)
    _static_convolution_cache[key] = convolve
    return convolve


def _make_lorentz_convolution(mesh_xy, kgrid, scale, q0_only):
    """Transform one Green tensor and stream vertex-weighted interactions into one sum."""
    from common.gamma_matrices import gamma_apply
    from common.fft_helpers import make_flat_k_fftn, make_flat_k_ifftn
    if not q0_only:
        inverse_g = make_flat_k_ifftn(mesh_xy, kgrid, G_FFT7D_SPEC, norm='ortho')
        forward_g = make_flat_k_fftn(mesh_xy, kgrid, G_FFT7D_SPEC, norm='ortho')
        inverse_v = make_flat_k_ifftn(mesh_xy, kgrid, V_FFT5D_SPEC, norm='ortho')

    @jax.jit
    def convolve(G_k, interactions, prefactor, vertices):
        green = G_k if q0_only else inverse_g(G_k)

        def add(total, block):
            interaction, (left, right) = block
            value = gamma_apply(green, *left, axis=1)
            value = gamma_apply(value, right[0], jnp.conj(right[1]), axis=3)
            weight = (interaction[0][None, None, :, None, :] if q0_only else
                      inverse_v(interaction)[:, None, :, None, :])
            return total + value * weight, None

        sigma, _ = jax.lax.scan(add, jnp.zeros_like(green), (interactions, vertices), unroll=1)
        return prefactor * (sigma if q0_only else forward_g(sigma)) * scale
    return convolve


def _make_cohsex_kernels(mesh_xy: Mesh, kgrid: tuple[int, int, int],
                         nk_tot: int, *, layout: str = "face",
                         face_shape=None, k_unfold_plan=None):
    """Build static SX/COH kernels on canonical faces with optional typed parent transport."""
    if layout != "face":
        raise ValueError(
            f"_make_cohsex_kernels: layout must be 'face', "
            f"got {layout!r}")
    if face_shape is None:
        raise ValueError(
            "_make_cohsex_kernels(layout='face') requires "
            "face_shape=(nk, nb_full, n_rmu, nspinor)")
    # Both checked BEFORE any FFT/FFI setup below — a bad layout argument
    # or a missing face_shape fails fast and cleanly rather than surfacing
    # as an unrelated FFI probe error from work that was about to be
    # thrown away anyway.
    from ffi import ffi_dial_key
    cache_key = (id(mesh_xy), tuple(int(x) for x in kgrid), ffi_dial_key(),
                layout, face_shape, k_unfold_plan)
    if cache_key in _cohsex_kernel_cache:
        return _cohsex_kernel_cache[cache_key]

    _convolve = _make_static_convolution(mesh_xy, kgrid, nk_tot)
    kernels = _make_cohsex_kernels_face(
        mesh_xy, face_shape, _convolve, k_unfold_plan=k_unfold_plan)

    _cohsex_kernel_cache[cache_key] = kernels
    return kernels


def _make_cohsex_kernels_face(mesh_xy: Mesh, face_shape, _convolve,
                              k_unfold_plan=None):
    """Contract static SX/COH and project before selecting the requested output band window."""
    from distrib_la import gemm_plan
    from common.contract_bands import contract_bands_block_reshard

    nk, nb_full, n_rmu, ns = (int(v) for v in face_shape)
    g_shape = (face_shape if k_unfold_plan is None
               else (k_unfold_plan.n_parent, *face_shape[1:]))
    nk_g, nb_g, n_rmu_g, ns_g = (int(v) for v in g_shape)
    mu_s = n_rmu_g * ns_g

    g_plan = gemm_plan(mesh_xy, m=mu_s, k=nb_g, n=mu_s, nq=nk_g,
                       dtype=jnp.complex128)
    proj_fn = contract_bands_block_reshard(
        mesh_xy, layout="face", face_shape=tuple(g_shape))
    if k_unfold_plan is not None:
        from ffi import _services
        _services.ensure_on_path()
        from symmetry_maps import unfold_file_wedge_band_operator
        _k_rows = np.asarray(k_unfold_plan.parent_full_rows, dtype=np.int32)
        _sym = k_unfold_plan.sym

    def _g_operands(wfns, wfns_g=None):
        """(direct face, conjugated face, band-table owner) for the G build."""
        g = wfns_g if wfns_g is not None else wfns
        if k_unfold_plan is None:
            return g.psi_mun, g.psi_nmu, g
        if wfns_g is not None:
            raise NotImplementedError(
                "_make_cohsex_kernels_face: a separate G-build bundle "
                "(bispinor vertex trick) is not combined with the parent "
                "route.")
        c = wfns.green_parent
        return c.psi_mun, c.psi_nmu, c

    def _project_bands(wfns, sigma_k):
        if k_unfold_plan is None:
            return _project(wfns.psi_nmu, wfns.psi_mun, sigma_k,
                            layout="face", face_project_fn=proj_fn)
        c = wfns.green_parent
        parent_rows = _project(
            c.psi_nmu, c.psi_mun,
            jnp.take(sigma_k, jnp.asarray(_k_rows), axis=0),
            layout="face", face_project_fn=proj_fn)
        # Static Σ is Hermitian, so conj and transpose coincide; use the
        # operator rule the dynamic route uses (unfold_file_wedge_band_
        # operator) so the two routes share one spelling.
        return unfold_file_wedge_band_operator(
            _sym, parent_rows, trs_rule="transpose")

    @jax.jit
    def sigma_sx(wfns, Gij, W_q, *, wfns_g=None):
        """Build occupied Green functions from the selected endpoints and project with the original states."""
        s = wfns.slices
        g_mun, g_nmu, _ = _g_operands(wfns, wfns_g)
        phases = _occ_diag_full(Gij, s.nb_sigma, nb_full)
        if k_unfold_plan is not None:
            phases = k_unfold_plan.parent_rows(phases)
        G_occ = build_G(g_mun, g_nmu, phases=phases,
                        layout="face", gemm=g_plan,
                        k_unfold_plan=k_unfold_plan)
        return _project_bands(wfns, _convolve(G_occ, W_q, 1.0))

    @partial(jax.jit, static_argnames=("ri_bands",))
    def sigma_coh(wfns, W_q, V_q, *, ri_bands=None):
        s = wfns.slices
        bands = (s.sigma_sum if ri_bands is None
                 else slice(int(ri_bands[0]), int(ri_bands[1])))
        g_mun, g_nmu, owner = _g_operands(wfns)
        mask = owner.band_mask(bands).astype(jnp.complex128)
        G_ri = build_G(g_mun, g_nmu, phases=mask,
                       layout="face", gemm=g_plan,
                       k_unfold_plan=k_unfold_plan)
        return _project_bands(wfns, _convolve(G_ri, W_q - V_q, -0.5))

    return sigma_sx, sigma_coh


# ---------------------------------------------------------------------------
# Static head addition — q→0 band-diagonal head correction for COHSEX.
# ---------------------------------------------------------------------------

def _replicate_head(head_kij, mesh_xy: Mesh):
    """Replicate a q→0 head matrix (nk, nb_sigma, nb_sigma) on the mesh.

    Shared by all four head placement sites (SX/COH in
    :func:`_add_static_head`; bare-X in :func:`compute_cohsex_sigma` and
    :func:`compute_sigma_x`).  Two concerns, in order:

    1. The q→0 head is a GLOBAL correction and must be bit-identical on
       every process.  On the full-BZ fallback path (e.g. a centroid set
       whose orbit closure fails) each rank can compute it with
       roundoff-level (~1e-19) divergence, so rank 0's copy is broadcast
       first.  No-op single-process, and a value no-op when the ranks
       already agree (the IBZ cascade path).
    2. Placement uses ``device_put_process_local``, NOT a bare
       ``jax.device_put``: on a multi-process replicated sharding the
       latter silently runs multihost ``assert_equal`` — a P-linear
       all-gather of the (nk, nb_sigma, nb_sigma) complex128 operand
       (scorecard AA.1/Y.5).  Post-broadcast bit-identity is exactly
       device_put_process_local's documented precondition;
       LORRAX_CHECK_REPLICA=1 restores the assertion.  (AO-sweep
       stragglers: the bare-X sites had neither the broadcast nor the
       process-local placement, and _add_static_head paid the assert
       all-gather on top of its broadcast — consolidated here, release
       audit 2026-07-28.)
    """
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        head_kij = multihost_utils.broadcast_one_to_all(head_kij)
    from common.collectives import device_put_process_local
    return device_put_process_local(
        head_kij, NamedSharding(mesh_xy, P(None, None, None)))


def _replicate_band_sigma(sigma_kij, mesh_xy: Mesh):
    """Place an already distributed N_b-class Sigma at its output seam.

    Static Sigma producers share this exact post-contraction transition.
    The operand is ``(nk, nb, nb)``, never a centroid-class response body.
    """
    return jax.lax.with_sharding_constraint(
        sigma_kij, NamedSharding(mesh_xy, P(None, None, None)))


def _add_static_head(sig_sx, sig_coh, *, static_head_terms, meta, mesh_xy,
                     do_screened: bool):
    """Add the q→0 head correction to SX/COH (no-op if terms is None)."""
    if static_head_terms is None:
        return sig_sx, sig_coh
    sx_h, coh_h = static_head_terms_to_kij(
        static_head_terms, nk_tot=meta.nk_tot, do_screened=do_screened)
    if not do_screened:
        coh_h = jnp.zeros_like(coh_h)
    return (sig_sx + _replicate_head(sx_h, mesh_xy),
            sig_coh + _replicate_head(coh_h, mesh_xy))


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
    mu_bases=None,
    occupation_state=None,
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
    occupation_state
        The iteration's :class:`gw.efermi.OccupationState`, or ``None``.
        ``None`` is the insulating default and keeps the integer ``occ >
        0.5`` projector bit-for-bit; a state makes Σ_X / Σ_SX / V_H read
        the same ``diag(f)`` weights Σ_c already uses.  Mutually
        exclusive with an explicit ``Gij`` (see :func:`_resolve_Gij`).

    Returns
    -------
    dict with keys:
        sig_sx   (nk, nb_sigma, nb_sigma)  physical occupied/SX component:
                                              screened charge exchange + head,
                                              plus bare transverse Σ^B when
                                              the bispinor operands are present
        sig_coh  (nk, nb_sigma, nb_sigma)  Coulomb hole + head (if screened)
        sig_x    (nk, nb_sigma, nb_sigma)  bare exchange + head, or None

    All returned arrays are pinned to **fully-replicated** sharding
    ``P(None, None, None)`` so the post-self-energy plumbing in
    ``gw_jax`` can operate on replicated H_kmn without resharding seams.
    They are small (``nk · nb_sigma² · 16 B`` ≲ tens of MB) so replication
    is essentially free; the heavy ω-grid Σ_c tensor stays sharded
    upstream in ``ppm_sigma`` and is only collapsed into a replicated
    Σ_xc^QSGW after the energy-domain contraction.
    """
    Gij = _resolve_Gij(Gij, meta, mesh_xy, occupation_state)

    kgrid = meta.kgrid
    nk_tot = int(meta.nk_tot)
    sigma_sx_k, sigma_coh_k = _make_cohsex_kernels(
        mesh_xy, kgrid, nk_tot, **_face_kwargs(wfns))

    with mesh_xy:
        sig_sx  = sigma_sx_k(wfns, Gij, W_q)
        sig_coh = sigma_coh_k(wfns, W_q, V_q)
        nb_sigma = wfns.slices.nb_sigma
        sig_sx  = _replicate_band_sigma(sig_sx, mesh_xy)
        sig_coh = _replicate_band_sigma(sig_coh, mesh_xy)
        sig_sx  = sig_sx[:, :nb_sigma, :nb_sigma]
        sig_coh = sig_coh[:, :nb_sigma, :nb_sigma]
        sig_sx, sig_coh = _add_static_head(
            sig_sx, sig_coh,
            static_head_terms=static_head_terms,
            meta=meta, mesh_xy=mesh_xy, do_screened=do_screened)
        sig_sx.block_until_ready()
        sig_coh.block_until_ready()

    sig_x = None
    sig_x_b = None
    if compute_bare_x:
        with mesh_xy:
            sig_x = sigma_sx_k(wfns, Gij, V_q)
        sig_x = _replicate_band_sigma(sig_x, mesh_xy)
        sig_x = sig_x[:, : wfns.slices.nb_sigma, : wfns.slices.nb_sigma]
        if static_head_terms is not None:
            x_head, _ = static_head_terms_to_kij(
                static_head_terms, nk_tot=meta.nk_tot, do_screened=False)
            # Broadcast + process-local replication — same per-rank
            # roundoff-divergence risk and hidden-assert cost as the SX/COH
            # heads; see _replicate_head.
            sig_x = sig_x + _replicate_head(x_head, mesh_xy)
        sig_x = _replicate_band_sigma(sig_x, mesh_xy)
        sig_x.block_until_ready()

        # Bispinor bare exchange: add Σ^B (transverse-only sum over
        # (i, j) ∈ {1, 2, 3}²) to the bare-X diagnostic AND to the
        # physical occupied/SX component.  The COHSEX dispatcher builds
        # Sigma_xc = sig_sx + sig_coh, so this is the single seam that carries
        # Σ^B into Eqp, the live Hamiltonian and sigma_diag without changing
        # X_ONLY (which uses compute_sigma_x) or the packed full-photon
        # route (which replaces all three photon components).  No-op when
        # ``wfns_transverse`` or ``bispinor_v_q_path`` is missing.  See
        # ``gw.sigma_x_bispinor`` and ``BISPINOR_DHFB_DESIGN.md`` §3.
        if wfns_transverse is not None and bispinor_v_q_path is not None:
            # face-layout defensive backstop REMOVED 2026-08-23
            # (feat/transverse-zeta-face-2026-08-23): compute_sigma_x_
            # bispinor is representation-aware since feat/bispinor-
            # face-2026-08-23 (with_lorentz_vertices, face_kernel_kwargs
            # dispatch) and the low_mem_bands_bispinor_unported envelope
            # row that made this branch unreachable for face is now
            # lifted — this call is the real, gated path, not dead code.
            from .sigma_x_bispinor import compute_sigma_x_bispinor
            with mesh_xy:
                sig_x_b = compute_sigma_x_bispinor(
                    wfns_transverse=wfns_transverse,
                    Gij=Gij,
                    bispinor_v_q_path=bispinor_v_q_path, mu_bases=mu_bases,
                    meta=meta, mesh_xy=mesh_xy,
                )
            sig_x_b.block_until_ready()
            sig_x = sig_x + sig_x_b
            sig_sx = sig_sx + sig_x_b

    return {
        "sig_sx":  sig_sx,
        "sig_coh": sig_coh,
        "sig_x":   sig_x,
        "sig_x_b": sig_x_b,
    }


def compute_sigma_x(
    wfns,
    V_q: jax.Array,
    meta,
    mesh_xy: Mesh,
    *,
    Gij: jax.Array | None = None,
    static_head_terms=None,
    wfns_transverse=None,
    bispinor_v_q_path=None,
    mu_bases=None,
    occupation_state=None,
    return_transverse: bool = False,
):
    """Bare-exchange-only path for modes without static screening.

    Skips the screened SX/COH kernels entirely — used by callers that
    don't need them (X_ONLY mode, and PPM modes via the dispatcher,
    which gets its dynamic Σ_c straight from
    :mod:`gw.ppm_pipeline`).  Each kernel is the same jit'd primitive
    used by :func:`compute_cohsex_sigma`, just called from a Python
    entry that won't ever invoke ``sigma_sx_k(W_q)`` or
    ``sigma_coh_k(W_q, V_q)`` and so saves two flat-q convolutions per
    call (≈ the ``W_q`` cost on each, roughly half the cohsex_sigma
    wall on dense band manifolds).

    Bispinor: identical to ``compute_cohsex_sigma``'s ``compute_bare_x``
    branch — Σ^B is added to ``sig_x`` when both ``wfns_transverse``
    and ``bispinor_v_q_path`` are supplied.

    ``occupation_state`` carries the same contract as in
    :func:`compute_cohsex_sigma`: ``None`` is insulating and bit-exact,
    a state puts ``diag(f)`` into Σ_X.
    """
    Gij = _resolve_Gij(Gij, meta, mesh_xy, occupation_state)
    sigma_sx_k, _ = _make_cohsex_kernels(
        mesh_xy, meta.kgrid, int(meta.nk_tot), **_face_kwargs(wfns))
    with mesh_xy:
        sig_x = sigma_sx_k(wfns, Gij, V_q)
        sig_x = _replicate_band_sigma(sig_x, mesh_xy)
        nb_sigma = wfns.slices.nb_sigma
        sig_x = sig_x[:, :nb_sigma, :nb_sigma]
        sig_x.block_until_ready()

    if static_head_terms is not None:
        x_head, _ = static_head_terms_to_kij(
            static_head_terms, nk_tot=meta.nk_tot, do_screened=False)
        # Broadcast + process-local replication — same per-rank roundoff-
        # divergence risk and hidden-assert cost as the SX/COH heads; see
        # _replicate_head.  This is the X_ONLY/PPM production entry.
        sig_x = sig_x + _replicate_head(x_head, mesh_xy)
        sig_x = _replicate_band_sigma(sig_x, mesh_xy)
        sig_x.block_until_ready()

    sig_x_b = None
    if wfns_transverse is not None and bispinor_v_q_path is not None:
        # face-layout defensive backstop REMOVED 2026-08-23 — see
        # compute_cohsex_sigma's identical removal, same session/reason.
        from .sigma_x_bispinor import compute_sigma_x_bispinor
        with mesh_xy:
            sig_x_b = compute_sigma_x_bispinor(
                wfns_transverse=wfns_transverse,
                Gij=Gij,
                bispinor_v_q_path=bispinor_v_q_path, mu_bases=mu_bases,
                meta=meta, mesh_xy=mesh_xy,
            )
        sig_x_b.block_until_ready()
        sig_x = sig_x + sig_x_b

    if return_transverse:
        return sig_x, sig_x_b
    return sig_x
