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

Two-face carrier (``low_mem_bands = true``, ``wfns.layout == "face"``):
:func:`_make_cohsex_kernels` dispatches to
:func:`_make_cohsex_kernels_legacy` or :func:`_make_cohsex_kernels_face`,
whose SX/COH bodies retain their pre-existing algorithms and which build G via
``greens_function_kernel.build_G(layout="face")``, projects via
``common.contract_bands.contract_bands_block_reshard(layout="face")``
(through ``wavefunction_bundle.project``), and gives Hartree its OWN
dedicated planned GEMM (V_H is diagonal-in-μ, not a general (μ,ν)
operator).  Face kernels return the FULL nb_full×nb_full matrix — a
legal face band axis cannot be sliced to the σ window (report §3) — so
:func:`compute_cohsex_sigma`/:func:`compute_v_h_sigma_x` gather to
replicated FIRST and window to nb_sigma second, the opposite order from
legacy's own (window-then-gather) sequence; see the guide,
``reports/gwjax_low_mem_bands_audit_2026-08-22/report.md``, and
``claims/0428.md`` for the real-CUDA parity gate.
"""
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
        n_win = float(np.sum(f_win)) / float(meta.nk_tot)
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
    """``{}`` under ``layout='legacy'``; the ``layout='face'`` +
    ``face_shape`` kwargs :func:`_make_cohsex_kernels` needs under
    ``layout='face'``.  Thin alias for
    :func:`gw.wavefunction_bundle.face_kernel_kwargs`, the shared owner
    (2026-08-22) — kept under this name so this module's own call sites
    below did not need to change."""
    from .wavefunction_bundle import face_kernel_kwargs
    return face_kernel_kwargs(wfns)


_cohsex_kernel_cache: dict[tuple[object, ...], tuple] = {}
_static_convolution_cache: dict[tuple[object, ...], object] = {}
_hartree_density_cache: dict[tuple[object, ...], object] = {}
_hartree_field_cache: dict[tuple[object, ...], object] = {}
_hartree_projection_cache: dict[tuple[object, ...], object] = {}


def make_hartree_density_kernel(
    mesh_xy: Mesh, *, layout: str, face_shape=None,
):
    """Occupied one-point ``sum_kn f_kn psi^dag Gamma_B psi`` owner.

    The returned vector is the unnormalised k-sum on its native centroid
    ``P('y')`` face.  The sole ``1/Nk`` factor remains in the field matvec
    kernel, exactly where the incumbent scalar Hartree path applies it.

    ``wfns_vertex`` is produced by the canonical
    :func:`gw.wavefunction_bundle.with_lorentz_vertices`: its right/G-conjugate
    field carries ``Gamma_B``.  Scalar Hartree passes ``wfns`` itself, which
    is exactly what channel zero's identity specialization returns.
    """
    shape_key = None if face_shape is None else tuple(int(v) for v in face_shape)
    key = (id(mesh_xy), layout, shape_key)
    if key in _hartree_density_cache:
        return _hartree_density_cache[key]

    if layout not in ("legacy", "face"):
        raise ValueError(f"Hartree density: unknown layout {layout!r}")
    if layout == "face" and face_shape is None:
        raise ValueError("face Hartree density requires face_shape")
    nb_full = None if face_shape is None else int(face_shape[1])

    @jax.jit
    def density(wfns, wfns_vertex, Gij):
        s = wfns.slices
        if layout == "legacy":
            psi = wfns.yr(s.sigma)
            vertex_psi = wfns_vertex.yr(s.sigma)
            rho_sum = jnp.real(jnp.einsum(
                "kisx,kjsx,kij->x", jnp.conj(psi), vertex_psi, Gij,
                optimize=True))
            return jax.lax.with_sharding_constraint(
                rho_sum, NamedSharding(mesh_xy, P("y")))

        occ = jnp.real(_occ_diag_full(
            Gij, s.nb_sigma, nb_full)).astype(jnp.float64)
        psi = wfns.psi_nmu
        vertex_psi = wfns_vertex.psi_nmu
        rho_sum = jnp.real(jnp.sum(
            jnp.conj(psi) * vertex_psi * occ[:, :, None, None],
            axis=(0, 1, 2)))
        return jax.lax.with_sharding_constraint(
            rho_sum, NamedSharding(mesh_xy, P("y")))

    _hartree_density_cache[key] = density
    return density


def make_hartree_field_kernel(mesh_xy: Mesh, nk_tot: int):
    """Canonical ``V_AB(q=0) rho_B / Nk`` distributed matvec."""
    key = (id(mesh_xy), int(nk_tot))
    if key in _hartree_field_cache:
        return _hartree_field_cache[key]

    @jax.jit
    def field(V_AB, rho_sum):
        rho = rho_sum / jnp.asarray(nk_tot, dtype=jnp.float64)
        rho = jax.lax.with_sharding_constraint(
            rho, NamedSharding(mesh_xy, P("y")))
        Vrho = jnp.einsum(
            "xy,y->x", V_AB[0], rho.astype(V_AB.dtype), optimize=True)
        return jax.lax.with_sharding_constraint(
            Vrho, NamedSharding(mesh_xy, P("x")))

    _hartree_field_cache[key] = field
    return field


def make_hartree_projection_kernel(
    mesh_xy: Mesh, *, layout: str, face_shape=None,
):
    """Project one local ``Gamma_A phi_A`` field into band space.

    This is the canonical local-potential Hartree projection for both scalar
    charge and transverse photon fields.  It never materialises a diagonal
    centroid operator: only the centroid vector and ordinary band-matrix
    result are live.  Photon Hartree may therefore sum ``phi_A = sum_B
    V_AB rho_B`` before projecting, paying three band GEMMs rather than nine.
    """
    shape_key = None if face_shape is None else tuple(int(v) for v in face_shape)
    key = (id(mesh_xy), layout, shape_key)
    if key in _hartree_projection_cache:
        return _hartree_projection_cache[key]

    if layout not in ("legacy", "face"):
        raise ValueError(f"Hartree projection: unknown layout {layout!r}")
    hartree_plan = None
    if layout == "face":
        if face_shape is None:
            raise ValueError("face Hartree projection requires face_shape")
        from distrib_la import gemm_plan
        nk, nb_full, n_rmu, ns = (int(v) for v in face_shape)
        hartree_plan = gemm_plan(
            mesh_xy, m=nb_full, k=n_rmu * ns, n=nb_full, nq=nk,
            dtype=jnp.complex128)

    @jax.jit
    def project(wfns, wfns_vertex, field_x):
        if layout == "legacy":
            psi = wfns.xn(wfns.slices.sigma)
            vertex_psi = wfns_vertex.xn(wfns.slices.sigma)
            return jnp.einsum(
                "ksxm,x,ksxn->kmn", jnp.conj(psi), field_x, vertex_psi,
                optimize=True)

        from common.contract_bands import merge_spin_centroid

        Vrho_rep = jax.lax.with_sharding_constraint(
            field_x, NamedSharding(mesh_xy, P(None)))
        Vrho_y = jax.lax.with_sharding_constraint(
            Vrho_rep, NamedSharding(mesh_xy, P("y")))

        psi_nmu, psi_mun = wfns.psi_nmu, wfns.psi_mun
        vertex_psi_mun = wfns_vertex.psi_mun
        weighted = (jnp.conj(psi_nmu)
                    * Vrho_y.astype(psi_nmu.dtype)[None, None, None, :])
        left = merge_spin_centroid(weighted, 2, 3)
        right = merge_spin_centroid(vertex_psi_mun, 1, 2)
        return hartree_plan(left, right)

    _hartree_projection_cache[key] = project
    return project


def _make_static_convolution(mesh_xy: Mesh, kgrid: tuple[int, int, int],
                             nk_tot: int, *, q0_only: bool = False):
    """Build the one flat-k convolution shared by every static Sigma block.

    Scalar COHSEX and photon blocks differ only in their endpoint centroid
    extents.  Those are the independently sharded trailing axes of the
    canonical FFT specs, so the same convolution serves square and
    rectangular operators without another FFT service or convention.
    """
    from ffi import ffi_dial_key
    cache_key = (
        id(mesh_xy), tuple(int(x) for x in kgrid), ffi_dial_key(), int(nk_tot),
        bool(q0_only))
    if cache_key in _static_convolution_cache:
        return _static_convolution_cache[cache_key]

    if q0_only:
        # Closed form of the SAME normalized flat-k convolution when the
        # interaction is structurally zero except at q=0:
        #   -1/Nk sum_q G(k-q)V(q) = -G(k)V(0)/Nk.
        # This branch is used only for retained head factors and avoids three
        # extra full-q FFT convolutions per Lorentz block.
        @jax.jit
        def _convolve(G_k, V_or_W, prefactor):
            V0 = V_or_W[0][None, None, :, None, :]
            return prefactor * G_k * V0 * (-1.0 / float(nk_tot))
    else:
        from common.fft_helpers import make_flat_k_fftn, make_flat_k_ifftn

        _G_ifftn = make_flat_k_ifftn(
            mesh_xy, kgrid, G_FFT7D_SPEC, norm='ortho')
        _G_fftn = make_flat_k_fftn(
            mesh_xy, kgrid, G_FFT7D_SPEC, norm='ortho')
        _V_ifftn = make_flat_k_ifftn(
            mesh_xy, kgrid, V_FFT5D_SPEC, norm='ortho')
        _inv_sqrt_nk = -1.0 / jnp.sqrt(float(nk_tot))

        @jax.jit
        def _convolve(G_k, V_or_W, prefactor):
            """Sigma^k = pref * FFT[G(R) V(R) / sqrt(Nk)]."""
            G_R = _G_ifftn(G_k)
            V_R = _V_ifftn(V_or_W)[:, None, :, None, :]
            return prefactor * _G_fftn(G_R * V_R * _inv_sqrt_nk)

    _static_convolution_cache[cache_key] = _convolve
    return _convolve


def _make_cohsex_kernels(mesh_xy: Mesh, kgrid: tuple[int, int, int],
                         nk_tot: int, *, layout: str = "legacy",
                         face_shape=None):
    """Cached factory returning SX/COH kernels and one Hartree scheduler.

    Keyed on (id(mesh_xy), kgrid, ffi_dial_key(), layout, face_shape) —
    same shape the chi0 / ppm_sigma kernel caches use, extended with the
    two-face carrier's static layout tag (report §5: a static tag
    SPECIALIZES a kernel build rather than branching a compiled one).
    The ``ffi_dial_key()`` component is load-bearing: the
    ``make_flat_k_*`` factories below read ``LORRAX_FFT_FFI`` at FACTORY
    time, so without the dials in the key a mid-process flag flip would
    serve a kernel built for the stale backend (the flat-k FFT service
    contract, ``docs/dev/flat_k_fft_service.md``).  ``nk_tot`` =
    prod(kgrid) and is redundant for cache-lookup purposes; it stays as a
    positional arg because the Hartree kernel closes over it as a
    compile-time constant.

    ``layout='face'`` requires ``face_shape=(nk, nb_full, n_rmu,
    nspinor)`` — the two ``distrib_la.gemm_plan``s this branch builds
    (one for G, one dedicated to Hartree's diagonal-weight contraction,
    plus the two inside the shared projector) need their shapes fixed
    EAGERLY, here, once.  Hartree schedules its three shared jitted stages
    directly, so scalar and photon direct terms reuse those executables
    instead of compiling a redundant outer Hartree wrapper.
    """
    if layout not in ("legacy", "face"):
        raise ValueError(
            f"_make_cohsex_kernels: layout must be 'legacy' or 'face', "
            f"got {layout!r}")
    if layout == "face" and face_shape is None:
        raise ValueError(
            "_make_cohsex_kernels(layout='face') requires "
            "face_shape=(nk, nb_full, n_rmu, nspinor)")
    # Both checked BEFORE any FFT/FFI setup below — a bad layout argument
    # or a missing face_shape fails fast and cleanly rather than surfacing
    # as an unrelated FFI probe error from work that was about to be
    # thrown away anyway.
    from ffi import ffi_dial_key
    cache_key = (id(mesh_xy), tuple(int(x) for x in kgrid), ffi_dial_key(),
                layout, face_shape)
    if cache_key in _cohsex_kernel_cache:
        return _cohsex_kernel_cache[cache_key]

    _convolve = _make_static_convolution(mesh_xy, kgrid, nk_tot)
    hartree_density = make_hartree_density_kernel(
        mesh_xy, layout=layout, face_shape=face_shape)
    hartree_field = make_hartree_field_kernel(mesh_xy, nk_tot)
    hartree_project = make_hartree_projection_kernel(
        mesh_xy, layout=layout, face_shape=face_shape)

    if layout == "legacy":
        kernels = _make_cohsex_kernels_legacy(
            _convolve, hartree_density, hartree_field, hartree_project)
    else:
        kernels = _make_cohsex_kernels_face(
            mesh_xy, face_shape, _convolve,
            hartree_density, hartree_field, hartree_project)

    _cohsex_kernel_cache[cache_key] = kernels
    return kernels


def _make_cohsex_kernels_legacy(
    _convolve, hartree_density, hartree_field, hartree_project,
):
    """Legacy static kernels; SX/COH retain their pre-face bodies exactly."""

    @jax.jit
    def sigma_sx(wfns, Gij, W_q):
        """Screened exchange:  Σ_SX = -project[ FFT[ G_occ(R) · W(R) / √Nk ] ].

        Pass V_q in place of W_q to get bare exchange Σ_X.
        """
        s = wfns.slices
        G_occ = build_G(wfns.xn(s.sigma), wfns.yr(s.sigma), Gij=Gij)
        return _project(wfns.xr(s.sigma), wfns.yn(s.sigma),
                        _convolve(G_occ, W_q, 1.0))

    @partial(jax.jit, static_argnames=("ri_bands",))
    def sigma_coh(wfns, W_q, V_q, *, ri_bands=None):
        """Coulomb-hole:  Σ_COH = +project[ FFT[ G_RI(R) · (W-V)(R) / (2·√Nk) ] ].

        ``ri_bands`` RESTRICTS THE INTERMEDIATE-STATE SUM, and exists because
        this term HAS one.  ``G_RI`` is a sum over every band in
        ``s.sigma_sum`` with no occupation projector, i.e. the Coulomb hole's
        slowly convergent unoccupied tail; ``None`` (the default, and the only
        value any production caller passes) is ``s.sigma_sum`` -- the Σ band
        sum, NOT the loaded extent ``s.full`` = max(chi, sigma), which is the
        same slice on every unsplit deck.  A ``(lo, hi)`` half-open band
        range restricts it, which is what lets a caller measure how much of
        this term is band-truncation error rather than assert that it has
        none — see ``gw.ppm_sigma._invalid_static_coh_by_bracket``.  An int
        tuple rather than a ``slice`` because it is a jit static argument and
        ``slice`` is not hashable before Python 3.12.
        """
        s = wfns.slices
        # ``sigma_sum``, not ``full``: the Coulomb-hole resolution-of-identity
        # G IS the Σ band sum, and ``full`` is the LOADED extent
        # (= max(chi, sigma)).  They are the same slice on every unsplit deck.
        bands = (s.sigma_sum if ri_bands is None
                 else slice(int(ri_bands[0]), int(ri_bands[1])))
        G_ri = build_G(wfns.xn(bands), wfns.yr(bands))
        return _project(wfns.xr(s.sigma), wfns.yn(s.sigma),
                        _convolve(G_ri, W_q - V_q, -0.5))

    def hartree(wfns, Gij, V_q):
        """Schedule the three shared jitted Hartree stages."""
        rho_sum = hartree_density(wfns, wfns, Gij)
        field = hartree_field(V_q, rho_sum)
        return hartree_project(wfns, wfns, field)

    return sigma_sx, sigma_coh, hartree


def _make_cohsex_kernels_face(
    mesh_xy: Mesh, face_shape, _convolve, hartree_density, hartree_field,
    hartree_project,
):
    """Face-layout kernel bodies.

    G and Σ-projection route through the two owning modules'
    ``layout='face'`` branches (``greens_function_kernel.build_G``,
    ``common.contract_bands.contract_bands_block_reshard`` via
    ``wavefunction_bundle.project``); Hartree gets its OWN dedicated
    planned GEMM from the enclosing factory because ``V_H`` is diagonal-in-μ,
    not a general
    (μ,ν) operator — routing it through the general two-GEMM projector
    would materialise a needless dense diag(Vρ) operand.  Every
    ``distrib_la.gemm_plan`` used by any of the three kernels below is
    built by the enclosing factory, once, eagerly — see
    :func:`_make_cohsex_kernels`'s
    docstring.

    None of the three kernels windows its OWN output to the σ band count:
    ``psi_nmu``/``psi_mun`` cover the full loaded extent, and slicing a
    2-D-sharded band axis to an arbitrary (non-mesh-divisible) window is
    exactly the illegal operation report §3 names.  Each returns the FULL
    (nk, nb_full, nb_full) matrix, still ``P(None,'x','y')``; the caller
    (:func:`compute_cohsex_sigma` / :func:`compute_v_h_sigma_x`) gathers
    it to replicated FIRST (legal at any size — the same gather legacy
    already pays for its smaller ``nb_sigma``-windowed result) and only
    THEN takes the ``[:nb_sigma, :nb_sigma]`` sub-block, a plain slice of
    a replicated array with no divisibility constraint at all.  This is
    the bring-up path's named cost (report §3): every face Σ call does
    the FULL nb_full×nb_full GEMM work, not the windowed nb_sigma one.
    """
    from distrib_la import gemm_plan
    from common.contract_bands import contract_bands_block_reshard

    nk, nb_full, n_rmu, ns = (int(v) for v in face_shape)
    mu_s = n_rmu * ns

    g_plan = gemm_plan(mesh_xy, m=mu_s, k=nb_full, n=mu_s, nq=nk,
                       dtype=jnp.complex128)
    proj_fn = contract_bands_block_reshard(
        mesh_xy, layout="face", face_shape=face_shape)

    @jax.jit
    def sigma_sx(wfns, Gij, W_q, *, wfns_g=None):
        """``wfns_g``, when given, supplies the G-BUILD's two operands
        (``psi_mun``/``psi_nmu``) INSTEAD of ``wfns``; the projection
        step always reads ``wfns``'s own copies.  Defaults to ``wfns``
        (every non-bispinor caller — identical to the pre-2026-08-23
        body, since ``g = wfns_g or wfns`` then reproduces the exact
        prior computation byte-for-byte).

        THE REASON THIS PARAMETER EXISTS AT ALL, and not on the legacy
        sibling: the two-face carrier stores exactly ONE (psi_mun,
        psi_nmu) pair, reused for BOTH the G-build's internal band sum
        and the outer band-basis projection — unlike the legacy
        four-copy carrier, whose G-vertex fields (psi_xn/psi_yr) and
        projection fields (psi_xr/psi_yn) are already four INDEPENDENT
        arrays.  The bispinor Σ^B transverse-vertex trick
        (``gw.sigma_x_bispinor``) needs γ̃ folded into the G-build's
        INTERNAL band sum only — never into the OUTER projection bra/
        ket (module docstring of ``gw.sigma_x_bispinor``: "the γ̃ vertex
        sits on the build_G side of the kernel chain, not the
        projection side").  On legacy that separation is already free
        (``gw.wavefunction_bundle.with_lorentz_vertices`` only ever
        touches psi_xn/psi_yr); on face, WITHOUT this parameter, the
        SAME two arrays would have to serve both roles at once, and a
        γ̃-inserted psi_mun/psi_nmu would corrupt the projection's outer
        bra/ket along with the G-build — MEASURED: real 4-rank CUDA,
        transverse (mu_L, nu_L) != (0,0) tiles disagreed with the
        legacy reference by O(1) relative before this parameter existed
        (tests/multi_device/bispinor_transverse_vertex_face_gate.py's
        own bring-up).
        """
        s = wfns.slices
        g = wfns_g if wfns_g is not None else wfns
        phases = _occ_diag_full(Gij, s.nb_sigma, nb_full)
        G_occ = build_G(g.psi_mun, g.psi_nmu, phases=phases,
                        layout="face", gemm=g_plan)
        return _project(wfns.psi_nmu, wfns.psi_mun,
                        _convolve(G_occ, W_q, 1.0),
                        layout="face", face_project_fn=proj_fn)

    @partial(jax.jit, static_argnames=("ri_bands",))
    def sigma_coh(wfns, W_q, V_q, *, ri_bands=None):
        s = wfns.slices
        bands = (s.sigma_sum if ri_bands is None
                 else slice(int(ri_bands[0]), int(ri_bands[1])))
        mask = wfns.band_mask(bands).astype(jnp.complex128)
        G_ri = build_G(wfns.psi_mun, wfns.psi_nmu, phases=mask,
                       layout="face", gemm=g_plan)
        return _project(wfns.psi_nmu, wfns.psi_mun,
                        _convolve(G_ri, W_q - V_q, -0.5),
                        layout="face", face_project_fn=proj_fn)

    def hartree(wfns, Gij, V_q):
        """Schedule the three shared jitted Hartree stages."""
        rho_sum = hartree_density(wfns, wfns, Gij)
        field = hartree_field(V_q, rho_sum)
        return hartree_project(wfns, wfns, field)

    return sigma_sx, sigma_coh, hartree


# ---------------------------------------------------------------------------
# Static head addition — q→0 band-diagonal head correction for COHSEX.
# ---------------------------------------------------------------------------

def _replicate_head(head_kij, mesh_xy: Mesh):
    """Replicate a q→0 head matrix (nk, nb_sigma, nb_sigma) on the mesh.

    Shared by all four head placement sites (SX/COH in
    :func:`_add_static_head`; bare-X in :func:`compute_cohsex_sigma` and
    :func:`compute_v_h_sigma_x`).  Two concerns, in order:

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
    Gij = _resolve_Gij(Gij, meta, mesh_xy, occupation_state)

    kgrid = meta.kgrid
    nk_tot = int(meta.nk_tot)
    sigma_sx_k, sigma_coh_k, hartree_k = _make_cohsex_kernels(
        mesh_xy, kgrid, nk_tot, **_face_kwargs(wfns))

    with mesh_xy:
        sig_sx  = sigma_sx_k(wfns, Gij, W_q)
        sig_coh = sigma_coh_k(wfns, W_q, V_q)
        sig_h   = hartree_k(wfns, Gij, V_q)
        if wfns.layout == "legacy":
            # UNCHANGED — add the head to the kernel's own nb_sigma-sized
            # result, THEN gather.  Do not reorder this branch.
            sig_sx, sig_coh = _add_static_head(
                sig_sx, sig_coh,
                static_head_terms=static_head_terms,
                meta=meta, mesh_xy=mesh_xy, do_screened=do_screened)
            sig_sx  = _replicate_band_sigma(sig_sx, mesh_xy)
            sig_coh = _replicate_band_sigma(sig_coh, mesh_xy)
            sig_h   = _replicate_band_sigma(sig_h, mesh_xy)
        else:
            # Face kernels return the FULL nb_full x nb_full matrix, still
            # 2-D sharded (_make_cohsex_kernels_face's docstring): gather
            # to replicated FIRST (legal at any size), THEN window to
            # nb_sigma (a plain slice of a replicated array — no
            # divisibility constraint), THEN add the (nb_sigma-shaped)
            # head.  Reversing legacy's order is required here, not a
            # stylistic choice: adding an nb_sigma head to an nb_full
            # array would be a shape error.
            nb_sigma = wfns.slices.nb_sigma
            sig_sx  = _replicate_band_sigma(sig_sx, mesh_xy)
            sig_coh = _replicate_band_sigma(sig_coh, mesh_xy)
            sig_h   = _replicate_band_sigma(sig_h, mesh_xy)
            sig_sx  = sig_sx[:, :nb_sigma, :nb_sigma]
            sig_coh = sig_coh[:, :nb_sigma, :nb_sigma]
            sig_h   = sig_h[:, :nb_sigma, :nb_sigma]
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
        if wfns.layout == "face":
            # Gather + window BEFORE the (nb_sigma-shaped) head — see the
            # same reasoning in the sig_sx/sig_coh block above.
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
        # X_ONLY (which uses compute_v_h_sigma_x) or the packed full-photon
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
                    bispinor_v_q_path=bispinor_v_q_path,
                    meta=meta, mesh_xy=mesh_xy,
                )
            sig_x_b.block_until_ready()
            sig_x = sig_x + sig_x_b
            sig_sx = sig_sx + sig_x_b

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
    occupation_state=None,
    compute_bare_x: bool = True,
) -> dict:
    """V-only Hartree plus optional bare-exchange path.

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

    ``compute_bare_x=False`` retains this function as the canonical Hartree
    owner while returning an exact-zero ``sig_x`` placeholder.  No exchange
    kernel or bispinor V file is touched.  The packed full-photon path uses
    that seam because its one block loop evaluates X from the same packed V
    that enters COH; computing the historical scalar+TT X here and discarding
    it would be both wasteful and a second V source.

    ``occupation_state`` carries the same contract as in
    :func:`compute_cohsex_sigma`: ``None`` is insulating and bit-exact,
    a state puts ``diag(f)`` into Σ_X and V_H.
    """
    Gij = _resolve_Gij(Gij, meta, mesh_xy, occupation_state)
    sigma_sx_k, _, hartree_k = _make_cohsex_kernels(
        mesh_xy, meta.kgrid, int(meta.nk_tot), **_face_kwargs(wfns))
    with mesh_xy:
        sig_h = hartree_k(wfns, Gij, V_q)
        sig_h = _replicate_band_sigma(sig_h, mesh_xy)
        if wfns.layout == "face":
            # Full nb_full x nb_full -> nb_sigma window, legal now that
            # both are replicated — see compute_cohsex_sigma's identical
            # reasoning.
            nb_sigma = wfns.slices.nb_sigma
            sig_h = sig_h[:, :nb_sigma, :nb_sigma]
        sig_h.block_until_ready()

        if compute_bare_x:
            sig_x = sigma_sx_k(wfns, Gij, V_q)
            sig_x = _replicate_band_sigma(sig_x, mesh_xy)
            if wfns.layout == "face":
                sig_x = sig_x[:, :nb_sigma, :nb_sigma]
            sig_x.block_until_ready()
        else:
            sig_x = jnp.zeros_like(sig_h)

    if compute_bare_x and static_head_terms is not None:
        x_head, _ = static_head_terms_to_kij(
            static_head_terms, nk_tot=meta.nk_tot, do_screened=False)
        # Broadcast + process-local replication — same per-rank roundoff-
        # divergence risk and hidden-assert cost as the SX/COH heads; see
        # _replicate_head.  This is the X_ONLY/PPM production entry.
        sig_x = sig_x + _replicate_head(x_head, mesh_xy)
        sig_x = _replicate_band_sigma(sig_x, mesh_xy)
        sig_x.block_until_ready()

    if (compute_bare_x and wfns_transverse is not None
            and bispinor_v_q_path is not None):
        # face-layout defensive backstop REMOVED 2026-08-23 — see
        # compute_cohsex_sigma's identical removal, same session/reason.
        from .sigma_x_bispinor import compute_sigma_x_bispinor
        with mesh_xy:
            sig_x_b = compute_sigma_x_bispinor(
                wfns_transverse=wfns_transverse,
                Gij=Gij,
                bispinor_v_q_path=bispinor_v_q_path,
                meta=meta, mesh_xy=mesh_xy,
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
