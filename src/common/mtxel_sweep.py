"""One k-tile scan serving V_H, kinetic+ion and dipole — G-split contraction.

API contract: ``docs/dev/rho_vh_2d_design.md``.  Read
``docs/architecture/decisions.md`` first: D10 (fixed-shape ``ngkmax`` G
tables) and the 2026-08-04 SlabIO padding entry are both load-bearing.

WHAT THIS REPLACES
------------------
Three sweeps with one shape between them —
``gw.kin_ion_io._vh_block``, ``._kin_ion_block`` and
``psp.get_dipole_mtxels._dipole_block`` — each calling
``collectives.gather_k_blocks``, which is k-partitioned and returns an
array **identical on every rank**.  Three walls follow:

  W1  the per-k full-band FFT box: 1.77 GB at b600 bispinor, 37 GB and
      OOM at 12×12, because the local plan takes BOTH sides of the
      matrix element out of one box.
  W2  the replicated ``(nk, nb, nb)``: 829 MB at 12×12, 9.2 GB at
      nb=2000, ×3 for dipole.
  W3  the k-partitioned plan CANNOT USE MORE THAN ``nk`` RANKS. Each rank
      takes whole k, so its wall is one full-band k no matter how large P
      is (4.02 s at nb=512/P=16 and 4.87 s at nb=600/P=64, jobs 7888868,
      7888877).  Never shard k: a molecule has ``nk = 1``.

THE STRUCTURAL FACT THAT COLLAPSES THE THREE
--------------------------------------------
All four operators are ``H[m,n] = Σ_{s,G} conj(ψ_m) · (O ∘ ψ)_n`` and
differ ONLY in ``O ∘ ψ``:

    kinetic   T_G · ψ                    diagonal in G
    dipole    2(k+G) · ψ  +  ∂V_NL/∂K    diagonal in G  +  separable
    V_NL      Z E Z† ψ                   separable, R ≪ nb·ns projectors
    V_H/V_loc F[ V(r) · F⁻¹ψ ]           FFT round trip

**The m side is never transformed.**  The output is ``nb²`` per k, the
operand ``nb·ns·N_G``: the contraction axis G is the big one.

THE PLAN (rules R1/R3 of ``reports/gwjax_scaling_levers_2026-09-23``)
--------------------------------------------------------------------
ψ is resident at ``band_sphere_spec`` (whole bands, nb/P per rank).  One
``shard_map`` scans k TILES; per tile of K k-points::

    O_band ψ      ← FFT operators, per k, on this rank's whole bands
    ψ, O_band ψ   → all-to-all → (nb, ns, G/P)       the G-split layout
    ket           = O_band ψ + O_diag ψ              diagonal terms, on the slab
    part[m,n]     = Σ_{s, G∈slab} conj ψ_m · ket_n   local GEMM
    H[m_X, n_Y]   = reduce-scatter(part)             one ``(c, nb, nb)`` per k
    + Σ c*_m E c_n with c = psum_slab(Z_slab† ψ)     separable terms, R·ns·nb

Per rank and per k this moves ``(1 + c_band)·ψ_k/P`` plus ``c·nb²``, where
the band-split plan it replaces moved ``ψ_k/p_x + c·ψ_k/p_y`` through a
collective-permute and two all-gathers (VI3 12×12, P16: 17.7 MB permuted
and 2×55.7 MB received per k, ``runs/runtime/mtxel_sweep_20260923``
census_a01_old_p16.txt).  A diagonal or separable operator moves only ψ:
the dipole's three-component ket never crosses the mesh, and V_NL's
projectors are built per slab (``R × G/P``), where the band-split plan built
the whole ``(R, ngkmax)`` — and ``(3, R, ngkmax)`` for the dipole — on every
rank for every k.

Measured, VI3 12×12 window [0,120), 144 k, ns=2, P4 on one node, warm
medians, band-split → G-split (``runs/runtime/mtxel_sweep_20260923``
a10/a14): V_H 2.963 → 2.480 s, kinetic 0.675 → 0.144 s, dipole p
0.900 → 0.275 s; executables 12.78 → 12.78, 9.93 → 9.74, 10.27 → 9.88 GiB;
blocks equal to 1.4e-14 relative.  V_H gains least: its wall is the FFTs.
At P16 over OFI (A100-40GB, old and new on the same pool, legs a20/a21):
V_H 1.599 → 1.220 s, kinetic 0.860 → 0.232 s, dipole p 0.919 → 0.302 s;
executables 4.126 → 4.126, 2.895 → 2.739, 3.055 → 2.778 GiB.  Per k and
rank the HLO moves 212.5 → 35.4 MiB (V_H), 968.8 → 105.0 (kin_ion's
T+V_loc+V_NL), 616.3 → 63.5 (dipole).  In the drivers at 360 bands:
kin_ion "T + ionic matrix" 10.25 → 7.14 s, dipole "q=0 velocity" 13.97 →
8.12 s; written artifacts equal to 5.1e-13 relative.

WHERE THE CROSSOVER IS.  The one non-``1/P`` transient is the tile's
``(K, c, nb, nb)`` partial.  The G split beats the band split in both bytes
and memory while ``nb·√P < ns·N_G``; with N_G/nb ≈ 200–1000 at production
cutoffs that is P ≲ 1e5, beyond the design envelope, and the ratio does
not change with system size (both scale with the cell).

k TILES.  ``K`` comes from :func:`plan_sweep`: k is packed only until each
per-peer all-to-all block is bandwidth-sized, and never past what the
resident ψ sphere bounds; ``K = 1`` at ``nk = 1``.  At VI3 12x12 every
K > 1 measured slower than K = 1, so the rule returns K = 1 there.  The FFT
box of a band operator is one k at a time inside the tile and never
scales with K.

WHO GATHERS, AND WHERE IT IS SAID
---------------------------------
The sharded block is the RETURN VALUE.  ``blocks_to_host`` at the end of
this module is the only boundary that undoes it, and it is called by
name at the sinks that cannot take a sharded operand (two serial h5py
writes, one replicated global operand).  A consumer that can stay
sharded — ``gw.sc_iteration.rebuild_hartree_dft_basis`` — does not call
it.  There is no implicit gather anywhere in this path.

WHY ψ(G) IS RESIDENT AND THE BOX NEVER IS
-----------------------------------------
The *box* is huge; the G-sphere is not.  ``nk·nb·ns·ngkmax·16`` is 1.2 GB
globally at b600 but **≈19 MB/rank sharded at P=64**, which is what makes
a genuine ``lax.scan`` over k possible at all.  Check the number for your
deck before assuming it: at 12×12 with nb=2000 it is ~10× larger.

THE FFT IS DEVICE-LOCAL, AND THAT IS WHY THE WHOLE SWEEP IS ONE PROGRAM
----------------------------------------------------------------------
A band-layout operator runs inside the sweep's ``shard_map`` on whole
bands, so its transforms are ``fft_helpers.local_{i,}fftn3`` — the inner
kernels of ``make_sharded_{i,}fftn_3d`` — and no FFT is distributed.  The
sphere→box gather is ``wfn_transforms._box_kernel`` fed a traced per-k
index, so nothing memoises a device G-index (the ``UnexpectedTracerError``
of job 7888526).  The flat-k FFT FFI is not used: its axis order suits the
Σ τ kernel's μ² tiles, and on this box it measured 2.0× slower (0.206 s
against 0.104 s, job 7889250 arm ``fftbench``).  The whole sweep is one
``jax.jit`` cached in ``_KERNEL_CACHE``, keyed on shapes, plan, mesh and
the operator's structural identity.

At MoS2 4×4 (nk=16, nb=128, ns=2, 24×24×80) the operator's wall IS the two
transforms — 5.70 s of 5.74 s at P=1 — and they scale linearly in P
(``tests/multi_device`` jobs 7889383–7889386); the sphere→box layout copy
XLA:GPU inserts before ``fft`` is ``_box_kernel``'s, not this module's.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common import timing
# PSI_MUN_SPEC / PSI_NMU_SPEC went with the two face-sharded producers
# deleted on 2026-09-02; this module now spells only the band-sphere
# layout, still through the wfn_layout owner.
from common.wfn_layout import band_sphere_spec
from common.wfn_transforms import _box_kernel, _cached_jit, _sharding_key
from runtime.padding import pad_axis


__all__ = [
    "SweepGeometry",
    "Operator",
    "kinetic_operator",
    "local_potential_operator",
    "four_current_potential_operator",
    "axis_function_operator",
    "collapsed_position_operator",
    "axis_window_operator",
    "vnl_operator",
    "dipole_operator",
    "dirac_current_operator",
    "uniform_gauge_operator",
    "sum_operators",
    "sweep_matrix_elements",
    "sweep_uniform_current_matrix_elements",
    "UniformGaugeMatrixElements",
    "UniformGaugeCurrentMatrixElements",
    "FiniteTransferCurrentEndpoint",
    "blocks_to_host",
]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class SweepGeometry:
    """The fixed shapes every operator and the scan agree on.

    Built once per sweep.  Everything here is static at trace time, which
    is what lets the scan body lower ONCE for the whole k range (D10).
    """

    __slots__ = ("mesh", "fft_grid", "ngkmax", "nb", "nb_logical", "ns",
                 "nk", "cell_volume", "ngrid", "p_prod", "band_axis")

    def __init__(self, *, mesh: Mesh, fft_grid: Sequence[int], ngkmax: int,
                 nb: int, ns: int, nk: int, cell_volume: float):
        from runtime.padding import padded_axis

        self.mesh = mesh
        self.fft_grid = tuple(int(s) for s in fft_grid)
        self.ngkmax = int(ngkmax)
        self.ns = int(ns)
        self.nk = int(nk)
        self.cell_volume = float(cell_volume)
        self.ngrid = int(np.prod(self.fft_grid))

        # ``nb`` is the LOGICAL band count.  The divisor is derived FROM THE
        # SPEC by the shared ``spec_divisor``, not assumed to be ∏ p_a — the
        # same call ``wfn_loader.WfnLoader._default_sharding`` makes for its own
        # ``p_band``.  Both default to ``P(None, ('x','y'), None, None)``, so
        # both get px·py and ψ FROM THE LOADER IS ALREADY BAND-PADDED FOR
        # THIS SWEEP: the pad below is a no-op on the production path and
        # exists for callers that build ψ themselves.  That agreement holds
        # only while both derive it here.
        #
        # Not a nicety at production shapes: nb=600 on an 8×8 mesh is
        # 64·9.375, and JAX raises IndivisibleError when the sharded array is
        # CONSTRUCTED rather than degrading (job 7888869).
        self.band_axis = padded_axis(
            int(nb), mesh, name="matrix-element sweep band carrier",
            spec=self.spec_sphere_xy, axis=1)
        self.p_prod = self.band_axis.divisor
        self.nb_logical = self.band_axis.logical
        self.nb = self.band_axis.carrier

    # ψ's resident layout: bands over the WHOLE mesh, all of G per band —
    # the loader's ``band_sphere_spec``.  A band-layout operator runs here
    # (whole bands, local FFT, no redundancy); the sweep all-to-alls to a
    # G-split layout for everything else.
    @property
    def spec_sphere_xy(self) -> P:
        return band_sphere_spec()

    # The output block: m on 'x', n on 'y', a replicated component axis
    # (dipole: 3 Cartesian directions) leading the two band axes.
    def spec_block_for(self, ncomp: int) -> P:
        return P(None, "x", "y") if not ncomp else P(None, None, "x", "y")


# ---------------------------------------------------------------------------
# The operator protocol
# ---------------------------------------------------------------------------

class Operator(NamedTuple):
    """``O ∘ ψ`` plus the normalisation that belongs to it.

    An operator is applied in whichever of the sweep's two layouts makes
    it LOCAL, and names that by which slots it fills (any combination; the
    contributions add):

    ``apply`` — BAND layout, for an operator that needs every G of a band
    at once (the FFT round trip of a local potential).  Called per k,
    inside the sweep's ``shard_map``, on this rank's whole bands:

      psi_n   (1, nb/P, ns, ngkmax) c128 — this rank's bands, all of G
      gvec    (ngkmax, 3) i32  — this k's G table (D10 fixed shape)
      gmask   (ngkmax,)   f64  — 1 on physical G, 0 on pad columns
      bidx    (1, ngkmax) i32 — per-k sphere index (flat box cell per G slot)
      kvec    (3,) f64

    and returns ``(1, nb/P, ns, ngkmax)`` — or, when ``ncomp > 0``,
    ``(1, nb/P, ns, ngkmax, ncomp)``.  The sweep all-to-alls that ket to
    the G-split layout.  Anything transforming must use the device-local
    kernels (``fft_helpers.local_{i,}fftn3``): it already runs per rank.

    ``apply_g`` — G-SPLIT layout, for an operator diagonal in G (T, p,
    Dirac α).  Called per k on this rank's G slab of EVERY band:

      psi     (nb, ns, g_slab) c128, already masked
      gvec    (g_slab, 3) i32;  gmask (g_slab,) f64;  kvec (3,) f64

    returning ``(nb, ns, g_slab[, ncomp])``.  Only ψ crosses the mesh for
    these terms; their ket never does.

    ``coeffs`` + ``couple`` — SEPARABLE, ``O = Σ |β⟩ D ⟨β|`` with a small
    projector count (V_NL and its K-derivatives).  ``coeffs(psi, gvec,
    gmask, kvec, *consts)`` returns a tuple of this slab's partial
    projections, BAND LAST, e.g. ``c[R,s,n] = Σ_{G∈slab} Z*[R,G] ψ[n,s,G]``;
    the sweep psums them over the mesh (``R·ns·nb``, never a G axis) and
    hands ``couple(coef_m, coef_n, *consts)`` the bra block's and the ket
    block's columns, which returns ``(nb_m, nb_n)`` or ``(ncomp, nb_m,
    nb_n)``.  The projectors are built on the slab, so no rank builds a
    whole-G ``Z`` for every k.

    No slot may form a ``(nb, nb)`` over G or gather over bands; the sweep
    owns every collective.

    ``ncomp`` is 0 for a scalar operator (T, V_loc, V_H, V_NL), 3 for
    a Cartesian one (dipole), and 12 for the one packed uniform
    current/contact transaction.  It is not a shape the sweep can infer:
    the sweep has to pick its einsum and its output spec at trace time,
    before it has seen the operator's output.

    ``post`` rides WITH the operator rather than being a sweep argument.
    The factor is part of the operator's own normalisation — the local
    potential's ``sqrt(1/volume)`` closes the same chain as its FFT
    constants — so a caller that picks an operator has already picked its
    normalisation and cannot pair the two up wrongly.  A ``post_scale=``
    argument on the sweep would be one more thing to get right at every
    call site, which is the mistake the SlabIO padding ruling names.
    """
    apply: Callable | None = None
    post: float = 1.0
    ncomp: int = 0
    #: Runtime operands.  The sweep threads them through its own jit and
    #: hands them to ``apply`` after ``kvec``.  Anything CLOSED OVER instead
    #: is a jaxpr CONSTANT, which ties the compiled program to that value —
    #: see :func:`_operator_key`.
    consts: tuple = ()
    #: STRUCTURAL identity for the sweep's jit cache.  See
    #: :func:`_operator_key`; an operator that leaves it empty falls back to
    #: ``id(apply)``, i.e. one lowering per factory call.
    key: tuple = ()
    #: G-split ket (diagonal-in-G terms); see the class docstring.
    apply_g: Callable | None = None
    #: Separable projector terms: slab coefficients and their coupling.
    coeffs: Callable | None = None
    couple: Callable | None = None


def _operator_key(op: "Operator") -> tuple:
    """Structural identity of an operator, for the sweep's jit cache.

    ``id(op.apply)`` is not it.  Every factory call builds a fresh closure,
    so a caller that rebuilds its operator per iteration misses on every
    one and re-traces, re-lowers and re-COMPILES the whole sweep — and
    ``gw.sc_iteration.rebuild_hartree_dft_basis`` rebuilds
    ``local_potential_operator`` once per density-SC step.  The persistent
    compile cache does not cover it either, because V_H changes each step
    and was baked into the module as a ``c128[nx,ny,nz]`` literal (visible
    in the lowered HLO as ``%constant.465``).

    Measured at b600/P=64, three successive sweeps with a fresh operator
    and a perturbed V_H, CACHE-COLD: steady-state 2.394 s/iteration with
    one XLA compile and one permanent ``_KERNEL_CACHE`` entry each, against
    2.167 s with zero of both once the key is structural and V_H rides in
    as an operand (job 7889250, arms ``base:recompile:cold`` and
    ``pat:recompile:cold``).  The cache growth is the more important half:
    it is one live executable plus its baked potential per iteration, for
    the life of the process.

    ``id(vnl_setup)`` inside a key is deliberate and safe: the cache entry
    holds the jitted ``_run``, which holds the operator closure, which
    holds the setup, so the id cannot be recycled onto a different object
    while the entry lives.
    """
    return tuple(op.key) if op.key else (
        'id', id(op.apply), id(op.apply_g), id(op.coeffs), id(op.couple))


def kinetic_operator(geom: SweepGeometry, bdot) -> Operator:
    """``T ∘ ψ = |k+G|² ψ`` — diagonal in G, no FFT.

    The reference is ``psp.get_DFT_mtxels._compute_kinetic_k_jit``, which
    masks ``T_G`` (not ψ) and contracts.  Masking the diagonal is
    sufficient and is reproduced exactly here, so the only difference
    between the two routes is the reassociation the band sharding forces
    on the G sum — which is why the gate is 1e-12 relative and not
    bit-identity (numerical-tolerance ruling; D10's ``RTOL_D10``).

    This operator is the skeleton's isolation test: no FFT means a
    failure here is the all-to-all, the slab GEMM or the reduce-scatter,
    never the transform.  Diagonal in G, so it acts on the G slab and its
    ket never crosses the mesh.
    """
    from psp.get_DFT_mtxels import kinetic_diagonal

    def op_g(psi, gvec, gmask, kvec, bdot_j):
        T_G = kinetic_diagonal(gvec, kvec, bdot_j, g_mask=gmask)
        return psi * T_G[None, None, :].astype(psi.dtype)

    return Operator(apply_g=op_g, post=1.0,
                    consts=(jnp.asarray(np.asarray(bdot, dtype=np.float64)),),
                    key=('kinetic', geom.ngkmax, geom.ns))


def local_potential_operator(
    geom: SweepGeometry, V_r, *, dirac_vector: bool = False,
) -> Operator:
    """Local scalar ``V`` or Dirac-vector ``sum_i alpha_i A_i`` operator.

    The default is ``V ∘ ψ = F[V(r) F⁻¹ψ]``, term-for-term the normalisation of
    ``psp.get_DFT_mtxels.compute_local_V_k``, so the two agree to
    round-off and the difference is pure reassociation from the sharding.

    ``dirac_vector=True`` consumes ``V_r.shape == (3,nx,ny,nz)`` and applies
    ``F[sum_i alpha_i V_i(r) F⁻¹ψ]`` with the canonical monomial gamma
    tables.  It is the same scatter/IFFT/FFT/gather and the same
    normalisation, not a parallel band-projection implementation.

    BAND layout (``Operator.apply``): the round trip needs every G of a
    band, and the sweep's ``shard_map`` hands this rank whole bands, so the
    transforms are the device-local kernels ``fft_helpers.local_{i,}fftn3``
    — the inner kernels of ``make_sharded_{i,}fftn_3d`` — and no FFT is
    ever distributed.  Only the scatter and the gather touch G, and both
    take their index as a traced operand.

    ``V_r`` is the potential on the FFT grid, replicated.  It rides in as
    an operand (``consts``) because it does not depend on k.
    """
    from common.fft_helpers import local_fftn3, local_ifftn3

    # THE SHARED NORMALISATION.  Same function the local plan's
    # ``_compute_local_V_k_jit`` calls, so the two agree by construction
    # rather than by hand.  Evaluated ONCE here, at factory-build time and
    # outside any trace, then frozen to Python floats: the constants
    # become jaxpr literals instead of riding through the scan as
    # operands, which is one fewer thing for XLA to keep live per
    # iteration.
    from psp.get_DFT_mtxels import local_potential_scalars
    _sc = local_potential_scalars(geom.cell_volume, geom.ngrid)
    scale = float(_sc.scale)
    deltaV = float(_sc.deltaV)
    fft_norm = float(_sc.fft_norm)
    vector = bool(dirac_vector)
    V_r_j = jnp.asarray(V_r, dtype=jnp.complex128)
    if vector:
        if int(geom.ns) != 4:
            raise ValueError(
                "Dirac-vector local potential requires four-component "
                f"bispinors; geom.ns={int(geom.ns)}")
        expected = (3, *tuple(int(s) for s in geom.fft_grid))
        if tuple(int(s) for s in V_r_j.shape) != expected:
            raise ValueError(
                "Dirac-vector local potential must have shape "
                f"{expected}; got {tuple(int(s) for s in V_r_j.shape)}")
        from common.gamma_matrices import gamma_apply, gamma_perm_phase
        alpha_vertices = tuple(gamma_perm_phase(i) for i in (1, 2, 3))
    elif tuple(int(s) for s in V_r_j.shape) != tuple(geom.fft_grid):
        raise ValueError(
            "scalar local potential must have shape "
            f"{tuple(geom.fft_grid)}; got "
            f"{tuple(int(s) for s in V_r_j.shape)}")

    def op(psi_n, gvec, gmask, bidx, kvec, V_r_j):
        # sphere → box.  ``_box_kernel`` is reused verbatim: it is pure
        # jax, band sharding rides through (the gather is over the G
        # axis, no cross-rank op), and its ngkmax zero-slot makes the
        # sentinel index gather exact zero.
        box = _box_kernel(psi_n, bidx, fft_grid=geom.fft_grid)
        psi_r = local_ifftn3(box, axes=(-3, -2, -1), norm='ortho') * scale
        if vector:
            phi_r = jnp.zeros_like(psi_r)
            for i, (perm, phase) in enumerate(alpha_vertices):
                phi_r = phi_r + V_r_j[i] * gamma_apply(
                    psi_r, perm, phase, axis=2)
        else:
            phi_r = psi_r * V_r_j
        phi_G = local_fftn3(phi_r, axes=(-3, -2, -1), norm='ortho') \
            * (deltaV * fft_norm)
        # box → sphere.  Advanced indexing on the three FFT axes only.
        gx = gvec[:, 0]
        gy = gvec[:, 1]
        gz = gvec[:, 2]
        out = phi_G[..., gx, gy, gz]
        return out * gmask[None, None, None, :].astype(out.dtype)

    # V(r) rides in as an OPERAND, not as a closed-over constant.  It is the
    # only input of this operator that moves between calls, and baking it
    # ties one compiled sweep to one V_H — a full lowering per density-SC
    # step (:func:`_operator_key`).  The scalars above stay literals: they
    # are functions of the geometry and do not move.
    key = (('local_potential', 'dirac_vector', geom.fft_grid, geom.ngkmax,
            geom.ns, scale, deltaV, fft_norm,
            tuple(int(d) for d in V_r_j.shape))
           if vector else
           # Preserve the historical scalar cache key byte-for-byte: adding
           # this feature must not invalidate every Vloc/VH executable.
           ('local_potential', geom.fft_grid, geom.ngkmax, geom.ns,
            scale, deltaV, fft_norm,
            tuple(int(d) for d in V_r_j.shape)))
    return Operator(apply=op, post=float(_sc.post), consts=(V_r_j,), key=key)


def four_current_potential_operator(
    geom: SweepGeometry, V_scalar_r, V_vector_r, *, charge_nspinor: int,
) -> Operator:
    """Pack scalar ``V_H`` and ``sum_i alpha_i A_i`` into one FFT sweep.

    The returned two components are separate matrix elements, not their sum:
    component 0 is the scalar charge Hartree and component 1 is the spatial
    Dirac-current Hartree.  They share the sphere scatter, inverse FFT, ket
    all-to-all and slab contraction.  The forward FFT is batched over the two
    outputs, preserving the decomposition required by ``sigma_mnk.h5``.

    ``charge_nspinor`` applies only to component 0.  This is load-bearing for
    the Pauli-reference model: the full four-spinor carries the current, while
    the scalar charge uses its leading source-WFN components.  Zeroing the
    scalar operator ket outside that block is algebraically identical to
    slicing both bra and ket because those output spinor rows are exact zero.
    """
    from common.fft_helpers import local_fftn3, local_ifftn3
    from common.gamma_matrices import gamma_apply, gamma_perm_phase
    from psp.get_DFT_mtxels import local_potential_scalars

    if int(geom.ns) != 4:
        raise ValueError(
            "four-current local potential requires four-component "
            f"bispinors; geom.ns={int(geom.ns)}")
    charge_ns = int(charge_nspinor)
    if not 0 < charge_ns <= 4:
        raise ValueError(
            "four-current charge_nspinor must be in [1,4]; got "
            f"{charge_nspinor}")
    grid = tuple(int(s) for s in geom.fft_grid)
    V0 = jnp.asarray(V_scalar_r, dtype=jnp.complex128)
    V1 = jnp.asarray(V_vector_r, dtype=jnp.complex128)
    if tuple(int(s) for s in V0.shape) != grid:
        raise ValueError(
            f"scalar four-current potential must have shape {grid}; got "
            f"{tuple(int(s) for s in V0.shape)}")
    if tuple(int(s) for s in V1.shape) != (3, *grid):
        raise ValueError(
            "spatial four-current potential must have shape "
            f"{(3, *grid)}; got {tuple(int(s) for s in V1.shape)}")

    scalars = local_potential_scalars(geom.cell_volume, geom.ngrid)
    scale = float(scalars.scale)
    fft_scale = float(scalars.deltaV * scalars.fft_norm)
    alpha_vertices = tuple(gamma_perm_phase(i) for i in (1, 2, 3))
    charge_mask = jnp.asarray(
        np.arange(4) < charge_ns, dtype=jnp.complex128).reshape(
            1, 1, 4, 1, 1, 1)

    def op(psi_n, gvec, gmask, bidx, kvec, V0, V1):
        del kvec
        box = _box_kernel(psi_n, bidx, fft_grid=geom.fft_grid)
        psi_r = local_ifftn3(box, axes=(-3, -2, -1), norm="ortho") * scale
        phi_scalar = psi_r * charge_mask * V0
        phi_vector = jnp.zeros_like(psi_r)
        for i, (perm, phase) in enumerate(alpha_vertices):
            phi_vector = phi_vector + V1[i] * gamma_apply(
                psi_r, perm, phase, axis=2)
        # (component, k, band, spinor, x, y, z): one batched forward FFT.
        phi_G = local_fftn3(jnp.stack((phi_scalar, phi_vector), axis=0),
                            axes=(-3, -2, -1), norm="ortho") * fft_scale
        gx, gy, gz = gvec[:, 0], gvec[:, 1], gvec[:, 2]
        out = phi_G[..., gx, gy, gz]
        out = jnp.moveaxis(out, 0, -1)
        return out * gmask[None, None, None, :, None].astype(out.dtype)

    return Operator(
        apply=op, post=float(scalars.post), ncomp=2, consts=(V0, V1),
        key=("four_current_local_potential", grid, geom.ngkmax, geom.ns,
             charge_ns, scale, fft_scale))


def axis_function_operator(
    geom: SweepGeometry, *, axis: int, coefficients, key: tuple,
) -> Operator:
    """A function ``F(f_a)`` of ONE fractional coordinate, applied in G
    space by its Fourier coefficients.

    ``F(f) = sum_g K(g) e^{2 pi i g f}`` multiplies a wavefunction as
    ``(F psi)(G) = sum_g K(g) psi(G - g e_a)``: a Toeplitz product along the
    box's ``a`` axis on the UNWRAPPED integer ``G_a`` difference, with no FFT
    and no grid sample (a sampled ``F`` aliases a slowly decaying tail into
    the pair densities).  ``coefficients(g)`` maps an integer array to
    ``K(g)``; ``key`` is the operator's structural identity for the sweep's
    jit cache (:func:`_operator_key`).  ``post = 1``: ``psi(G)`` is
    normalised on the sphere, so the block is ``<m| F |n>``.

    BAND layout (``Operator.apply``): the Toeplitz acts along one FFT axis
    of this rank's whole bands inside the sweep's ``shard_map``, so it is
    rank-local by construction.
    """
    a = int(axis)
    if a not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2; got {axis}")
    n = int(geom.fft_grid[a])
    g_of_index = np.fft.fftfreq(n, 1.0 / n).astype(np.int64)     # box order
    diff = g_of_index[:, None] - g_of_index[None, :]             # G_a(i) - G_a(j)
    M_j = jnp.asarray(np.asarray(coefficients(diff), dtype=np.complex128))
    box_axis = 3 + a                                             # (1, nb, ns, x, y, z)

    def op(psi_n, gvec, gmask, bidx, kvec, M):
        del kvec
        box = _box_kernel(psi_n, bidx, fft_grid=geom.fft_grid)
        moved = jnp.moveaxis(box, box_axis, -1)
        phi = jnp.moveaxis(
            jnp.einsum("...j,ij->...i", moved, M, optimize=True),
            -1, box_axis)
        out = phi[..., gvec[:, 0], gvec[:, 1], gvec[:, 2]]
        return out * gmask[None, None, None, :].astype(out.dtype)

    return Operator(apply=op, post=1.0, consts=(M_j,),
                    key=(*tuple(key), geom.fft_grid, geom.ngkmax, geom.ns, a))


def collapsed_position_operator(
    geom: SweepGeometry, *, axis: int, center: float,
) -> Operator:
    """``zeta_a = 2 pi wrap(f_a - f_a^0)``, the position conjugate to a
    COLLAPSED reduced k axis, as an :func:`axis_function_operator`.

    The sawtooth has the series ``sum_{g != 0} i (-1)^g / g e^{2 pi i g x}``,
    so ``K(g) = i (-1)^g e^{-2 pi i g f_a^0} / g``.  Its branch cut sits half
    a cell from ``center`` ``f_a^0``, at the centre of the largest vacuum gap
    (``common.parallel_transport.collapsed_axis_center``).  The block is
    ``<m| zeta_a |n>``, dimensionless: the ``a`` component of the
    reduced-coordinate Berry connection.
    """
    c = float(center)

    def sawtooth(g):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                g != 0,
                1.0j * ((-1.0) ** g) * np.exp(-2.0j * np.pi * g * c)
                / np.where(g != 0, g, 1),
                0.0 + 0.0j)

    return axis_function_operator(
        geom, axis=axis, coefficients=sawtooth,
        key=('collapsed_position', round(c, 12)))


def axis_window_operator(
    geom: SweepGeometry, *, axis: int, center: float, width: float,
) -> Operator:
    """The indicator of ``|wrap(f_a - center)| < width / 2`` as an
    :func:`axis_function_operator`: ``<n| chi |n>`` is the fraction of band
    ``n``'s density inside that slab of the cell.  The producer's probe of
    the density at a collapsed axis's branch cut.
    """
    c, w = float(center), float(width)
    if not 0.0 < w <= 1.0:
        raise ValueError(f"window width must be in (0, 1]; got {width}")

    def window(g):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                g != 0,
                np.exp(-2.0j * np.pi * g * c) * np.sin(np.pi * g * w)
                / (np.pi * np.where(g != 0, g, 1)),
                w + 0.0j)

    return axis_function_operator(
        geom, axis=axis, coefficients=window,
        key=('axis_window', round(c, 12), round(w, 12)))


def _ket(psi_n, gmask):
    """The scan's ``(1, nb, ns, ngkmax)`` ket as the ``(nb, ns, nG)`` the
    ``psp`` apply-to-ket kernels take, masked on the D10 pad columns.

    The mask has to reach ψ for the projector operators, not just the
    diagonal: ``Z`` (and ``dZ``) are FINITE on a pad column — they are
    evaluated at ``K = kvec`` there, not at a zero form factor — so an
    unmasked ket adds ``ngkmax - ngk`` spurious projector overlaps.
    ``psp.dft_operators.vnl_matrix_from_kdata`` documents the same rule
    for the local plan and satisfies it the same way.

    The leading k axis is singleton, so dropping it moves the band axis
    from 1 to 0 without moving any data.  It runs on this rank's own bands
    inside the sweep's ``shard_map``; the sweep owns every collective.
    """
    psi = psi_n[0]
    return psi * gmask[None, None, :].astype(psi.dtype)


def _pad_spinor(x, ns: int):
    """Zero-fill a ``(..., nb, ns_op, nG)`` operator output back to ``ns``.

    ``E_super`` is built at the FILE's ``nspinor``; a bispinor ψ carries
    4 components, of which V_NL acts on the first 2.  The local plan
    handles this by slicing BOTH sides of the matrix element
    (``dft_operators.vnl_matrix_from_kdata``).  The sweep cannot: the m
    side is shared by every operator in a sum, so it is the ket that is
    padded back.  The two are identical because the pad rows are exact
    zeros — ``Σ_s conj(ψ_m,s)(Oψ)_{n,s}`` then runs over ``s < ns_op``
    either way.
    """
    have = int(x.shape[-2])
    if have == ns:
        return x
    pad = [(0, 0)] * x.ndim
    pad[-2] = (0, ns - have)
    return jnp.pad(x, tuple(pad))


def vnl_operator(geom: SweepGeometry, vnl_setup) -> Operator:
    """``⟨m|V_NL|n⟩ = Σ c*_m E c_n`` with ``c = Z† ψ`` — SEPARABLE.

    V_NL is ``Z E Z†`` with a projector count ``R`` far below ``nb·ns``, so
    it never needs a G-space ket: each rank projects its G slab of every
    band onto the projectors built ON THAT SLAB (``c`` partial, ``R·ns·nb``),
    the sweep psums the partials, and the bra and ket blocks couple through
    ``E`` (:func:`psp.vnl_ops.vnl_block_from_coefficients`).  The same
    ``Z E Z†`` the local plan's ``vnl_ops.vnl_matrix`` contracts, with the G
    sum split over ranks: gate at 1e-12 relative, not bit-identity.

    The pad columns are inert because the sweep masks ψ before calling
    ``coeffs``: ``Z`` itself is FINITE on a pad column (evaluated at
    ``K = kvec``), so it is ψ that must be zero there.

    The projector BUILD is slab-sized too: ``Z`` is ``(R, G/P)`` per rank,
    where the band-split plan this replaced built the whole ``(R, ngkmax)``
    on every rank for every k (replicated work, measured flat in P at MoS2
    4×4, job 7889392).

    A bispinor ψ carries 4 components of which V_NL acts on the first
    ``E_super``'s ``nspinor``: the projections run over those alone, which
    is identical to padding the ket with exact-zero rows.
    """
    from psp import vnl_ops

    ns_e = int(vnl_setup.E_super.shape[0])

    def coeffs(psi, gvec, gmask, kvec):
        del gmask                      # ψ arrives masked
        Z = vnl_ops.build_vnl_kdata_traced(kvec, gvec, vnl_setup).Z
        c, _ = vnl_ops.projector_coefficients(psi[:, :ns_e], Z)
        return (c,)

    def couple(bra, ket):
        return vnl_ops.vnl_block_from_coefficients(
            bra[0], ket[0], vnl_setup.E_super)

    return Operator(coeffs=coeffs, couple=couple, post=1.0,
                    key=('vnl', geom.ngkmax, geom.ns, id(vnl_setup)))


#: The shipped relative sign of the nonlocal commutator term inside
#: :func:`dipole_operator`, as a named constant so the two arms of the
#: open question can be spelled without a magic ``-1.0`` at four call
#: sites.  ``-1.0`` is the historical assembly; the default is
#: :data:`VNL_VELOCITY_SIGN_FLIPPED` (``psp.get_dipole_mtxels``); see that
#: function's SIGN section for what is actually in dispute.
VNL_VELOCITY_SIGN_SHIPPED = -1.0

#: The other arm.  It is not "the fix" — the choice is the owner's — it
#: is the second reproducible configuration, so that a measurement of
#: the difference does not require patching a source file.
VNL_VELOCITY_SIGN_FLIPPED = +1.0


_VNL_VELOCITY_SIGN_WORDS = {
    "shipped": VNL_VELOCITY_SIGN_SHIPPED,
    "minus": VNL_VELOCITY_SIGN_SHIPPED,
    "flipped": VNL_VELOCITY_SIGN_FLIPPED,
    "plus": VNL_VELOCITY_SIGN_FLIPPED,
}


def require_vnl_velocity_sign(value) -> float:
    """Return one of the two characterized nonlocal-velocity signs.

    ``value`` is the resolved ``vnl_velocity_sign`` deck/CLI value or the
    direct :func:`dipole_operator` argument.  This is the single owner of
    both the word aliases and the two-arm validation, so the producer and
    operator cannot drift on which values are meaningful.
    """
    raw = value
    if isinstance(raw, str):
        raw = raw.strip().lower()
        raw = _VNL_VELOCITY_SIGN_WORDS.get(raw, raw)
    try:
        sign = float(raw)
    except (TypeError, ValueError):
        sign = None
    if sign not in (VNL_VELOCITY_SIGN_SHIPPED,
                    VNL_VELOCITY_SIGN_FLIPPED):
        raise ValueError(
            "GATE vnl_velocity_sign: the nonlocal velocity has only two "
            "characterized sign arms.\n"
            f"  got:  vnl_velocity_sign = {value!r}\n"
            "  want: vnl_velocity_sign in {-1, +1, shipped, minus, "
            "flipped, plus}\n"
            "  why:  this is a sign, not a scale; any other multiplier "
            "would produce a velocity operator that is neither measured "
            "arm and that no BerkeleyGW comparison characterizes\n"
            "  doc:  docs/input_reference.md, vnl_velocity_sign.")
    return sign


def dirac_current_operator(geom: SweepGeometry) -> Operator:
    """Uniform paramagnetic c alpha on the actual four-component carrier.

    The existing sweep owns band sharding, masks and bra contraction. This
    operator includes neither a nonlocal potential derivative nor a QP
    correction; consumers combine those explicitly in band space.
    """
    from common.bispinor_init import apply_dirac_velocity_to_ket
    if geom.ns != 4:
        raise ValueError('Dirac current sweep requires four-component wavefunctions')

    def op_g(psi, gvec, gmask, kvec):
        # Spinor mixing only, diagonal in G: acts on the G slab.
        return jnp.moveaxis(apply_dirac_velocity_to_ket(psi), 0, -1)

    return Operator(apply_g=op_g, post=1.0, ncomp=3, consts=(),
                    key=('dirac_current', geom.ngkmax, geom.ns))


def dipole_operator(geom: SweepGeometry, *, bvec, blat,
                    vnl_setup=None,
                    vnl_velocity_sign=VNL_VELOCITY_SIGN_FLIPPED,
                    hubbard=None) -> Operator:
    """``v ∘ ψ = 2(k+G)_cart ψ ± (∂V_NL/∂K_cart) ψ [+ (∂V_U/∂K_cart) ψ]``.

    THREE components.  ``hubbard`` (a ``psp.hubbard_ops.HubbardSetup``) adds
    the DFT+U commutator ``i[r, V_U]`` for a QE ortho-atomic DFT+U mean
    field, through ``hubbard_ops.apply_hubbard_velocity_to_ket`` -- the same
    band-local projector apply as V_NL (atomic rows replicated,
    ``4·N_atwfc·ngkmax·16`` B per k: 204 MB on VI3 12x12, a sixth of V_NL's
    Z/dZ), so ψ stays band-sharded and nothing ``(nb, nb)`` is formed.
    Its sign is the physical +1 and is not the V_NL knob's.  ``None`` (every
    non-DFT+U deck) executes the pre-Hubbard operator literally.

    The velocity matrix ``psp.get_dipole_mtxels`` writes is
    ``p - i[r, V_NL]`` in the stored convention, assembled here as
    ``p_cart + v_NL_cart``.  WHICH ARM A FILE WAS BUILT WITH IS NOT A
    PROPERTY OF THIS DOCSTRING: it is stamped into every ``dipole.h5``
    as ``prov_vnl_velocity_sign``, and files written before that stamp
    existed are the ``-1`` arm.  No velocity physics is written here:
    ``p`` is ``dft_operators.apply_kinetic_velocity_to_ket`` applied on the
    G slab (diagonal in G, so only ψ crosses the mesh), and the nonlocal
    term is the SEPARABLE ``(dc)† E c + c† E dc`` of
    ``vnl_ops.vnl_velocity_block_from_coefficients`` — the bra contraction
    of ``vnl_ops.apply_vnl_velocity_to_ket`` — on projections the sweep
    reduces over the mesh (see :func:`vnl_operator`).

    ``vnl_setup=None`` reproduces ``--skip-vnl`` (p̂ only).

    The component axis of ``p``'s ket is moved to the END, where the
    sweep's contraction expects it.

    THE SIGN, WHICH WAS AN OPEN QUESTION AND IS NOW DECIDED
    -------------------------------------------------------
    ``vnl_velocity_sign`` multiplies the nonlocal term and nothing else.
    It takes exactly ``+1.0`` (the DEFAULT since 2026-08-09) or ``-1.0``
    (the arm every ``dipole.h5`` committed before that date was built
    with, kept reachable so those files stay reproducible).  The two
    signs are separate branches rather than a scalar multiply: the legacy
    arm's nonlocal block is exactly negated (IEEE negation is exact) and
    added, i.e. the subtraction it always was.

    The decision was measured, not argued.  On the si_bigcond_prep mean
    field at the band window matched to the BerkeleyGW contour-
    deformation reference (nval 8 / ncond 92 / nband 100), against
    BerkeleyGW's own stored q → 0 head at all 265 CD frequencies —
    the two percentage columns are eps00 and omega_p SEPARATELY, which
    an earlier draft of this table collapsed into one and thereby
    understated the shipped arm's eps00 error by half:

        arm                       eps00(0)   d_eps    omega_p    d_wp
        BerkeleyGW (reference)     24.2205      --   18.101 eV     --
        p only (``--skip-vnl``)    27.8686  +15.06%  19.546 eV  +7.99%
        sign −1 (legacy)           31.8204  +31.38%  21.259 eV +17.45%
        sign +1 (DEFAULT)          24.2208   +0.00%  18.101 eV  +0.00%

    The structural argument is the sharpest: dropping the term entirely
    is BETTER than including it with the legacy sign, which is the
    signature of a sign and not of a magnitude.  Four further witnesses
    agree, three of them internal to this tree — the surviving
    ``vnl_ops.vnl_velocity_matrix`` derivative owner and
    ``orbital_magnetization`` use ``+dV_NL/dK``, and ``--vnl-mode numeric``
    did too (by way of a double negation nobody had noticed).

    ``gw.mpa.head_dipole.head_fsum_from_transitions`` carries the same
    table and the f-sum saturations beside it.

    WHAT THIS KNOB IS NOT.  It is not a claim about the per-(ψ, G-list)
    projector contraction, which reproduces Quantum ESPRESSO to ~10
    significant figures and which the standing project rule protects.
    Only the sign with which the assembled term enters the velocity is
    parameterised here.

    WHAT THE DEFAULT CHANGE DID NOT DO.  It did not re-cut the committed
    fixtures.  Every ``dipole.h5`` under ``tests/regression`` was built
    with ``-1`` and still is, so until those are regenerated a bare run
    of this operator and the files in the tree are two different
    operators — which is precisely what ``prov_vnl_velocity_sign`` on
    the h5 exists to make visible, and what
    ``tests/test_bse_oscillator_strengths.py`` exists to notice.

    THE CACHE HAZARD THE KEY CLOSES.  ``_operator_key`` is the sweep's
    jit-cache identity, and two operators that hash the same share a
    compiled program — which for a closed-over sign would mean the
    second arm of an A/B silently re-running the first.  That is exactly
    the defect class this project has already paid for twice, so the
    sign joins the key.  Two sweeps at the two signs in one process is
    a supported thing to do, and it is what the test does.
    """
    from psp.dft_operators import apply_kinetic_velocity_to_ket
    from psp import vnl_ops

    sign = require_vnl_velocity_sign(vnl_velocity_sign)

    B = jnp.asarray(np.asarray(bvec, dtype=np.float64) * float(blat),
                    dtype=jnp.float64)
    flipped = sign > 0.0

    def op_g(psi, gvec, gmask, kvec, B):
        # p = 2(k+G)_cart: diagonal in G, applied on the slab.
        return jnp.moveaxis(
            apply_kinetic_velocity_to_ket(psi, gvec, kvec, B), 0, -1)

    separable = {}
    if vnl_setup is not None:
        ns_e = int(vnl_setup.E_super.shape[0])

        def coeffs(psi, gvec, gmask, kvec, B):
            del gmask, B                    # ψ arrives masked
            kdata = vnl_ops.build_vnl_kdata_traced(kvec, gvec, vnl_setup,
                                                   compute_dZ=True)
            return vnl_ops.projector_coefficients(
                psi[:, :ns_e], kdata.Z, kdata.dZ)

        def couple(bra, ket, B):
            del B
            v_nl = vnl_ops.vnl_velocity_block_from_coefficients(
                bra[0], bra[1], ket[0], ket[1], vnl_setup.E_super)
            # The sweep ADDS this block to p's; the shipped arm's literal
            # subtraction is that add of an exactly negated block.
            return v_nl if flipped else -v_nl

        separable = dict(coeffs=coeffs, couple=couple)

    hub = {}
    if hubbard is not None:
        if not flipped:
            raise ValueError(
                "GATE dftu_velocity_sign: i[r, V_U] requested with the legacy "
                "vnl_velocity_sign = -1 arm\n  got:  vnl_velocity_sign = -1 with a "
                "DFT+U Hubbard setup\n  want: vnl_velocity_sign = +1\n  why:  the -1 "
                "arm exists only to reproduce pre-2026-08-09 files, none of which "
                "carries V_U; a mixed-sign velocity is no DFT operator\n  fix:  drop "
                "vnl_velocity_sign or set it to +1")
        from psp.hubbard_ops import apply_hubbard_velocity_to_ket

        def op_u(psi_n, gvec, gmask, bidx, kvec, B):
            # i[r, V_U] on the band layout: the Loewdin rows need the whole
            # G sphere of each band, which a G slab does not hold.  Summed on
            # the two Pauli components, then one spinor pad.
            del bidx, B
            psi = _ket(psi_n, gmask)
            v_u = apply_hubbard_velocity_to_ket(
                psi[:, :2], kvec, gvec, gmask, hubbard)
            return jnp.moveaxis(_pad_spinor(v_u, int(psi.shape[1])), 0, -1)[None]

        hub = dict(apply=op_u)

    return Operator(apply_g=op_g, post=1.0, ncomp=3, consts=(B,),
                    key=('dipole', geom.ngkmax, geom.ns, float(blat),
                         None if vnl_setup is None else id(vnl_setup),
                         sign)
                    + (() if hubbard is None else (('hubbard', id(hubbard)),)),
                    **separable, **hub)


class UniformGaugeCurrentMatrixElements(NamedTuple):
    r"""Band-sharded uniform current action without unrelated response jets.

    This is a component-selection view of :func:`uniform_gauge_operator`, not
    a second current implementation.  It exists for Hall consumers, which
    need only ``Gamma_raw`` and the exact Hamiltonian/operator fingerprint;
    retaining contact and transfer jets for that terminal three-number
    reduction is prohibitive on a production band manifold.
    """

    gamma_raw: jax.Array
    hamiltonian_config_operator_fingerprint: str


class UniformGaugeMatrixElements(NamedTuple):
    r"""Band-sharded uniform gauge action and optional transfer jet.

    ``gamma_raw`` is the dimensionless no-pair vertex
    ``(alpha_FS/2) dH_Pauli_Ry/dk``. ``lambda_raw`` is its exact uniform
    derivative ``(alpha_FS/2) d2H_Pauli_Ry/dkdk``.  Their shapes are
    ``(nk,3,nb,nb)`` and ``(nk,3,3,nb,nb)`` and both retain the sweep's
    two-dimensional band sharding.  With the separately priced transfer-q2
    capability, ``dgamma_dq_raw`` and ``d2gamma_dq2_raw`` have shapes
    ``(nk,3,3,nb,nb)`` and ``(nk,3,3,3,nb,nb)``.  They are deliberately one
    transaction: response-jet and contact consumers must not reopen the WFN
    or rebuild projectors.  Hall's current-only component selection is the
    smaller sibling above and calls the same operator/sweep owners.
    """

    gamma_raw: jax.Array
    lambda_raw: jax.Array
    hamiltonian_config_operator_fingerprint: str
    dgamma_dq_raw: jax.Array | None = None
    d2gamma_dq2_raw: jax.Array | None = None


class FiniteTransferCurrentEndpoint(NamedTuple):
    r"""One exact finite-q current endpoint sampled at current centroids.

    ``current_nmu`` and ``current_mun`` are the two face orientations of
    ``Gamma_i(k,q)|Psi_nk>``.  Their shapes are ``(nk,nb,3,4,n_rmu)`` and
    ``(nk,3,4,n_rmu,nb)``; after flattening the replicated ``(cart,spin)``
    pair, they use the canonical :data:`common.wfn_layout.PSI_NMU_SPEC` and
    :data:`common.wfn_layout.PSI_MUN_SPEC`.  They are deliberately not a
    :class:`gw.wavefunction_bundle.Wavefunctions`: the endpoint depends
    jointly on ``(k,q)`` and pretending it were a q-independent wavefunction
    face would let the incumbent k-FFT silently apply the wrong operator at
    every other q.

    The two fingerprints name different facts.  The Hamiltonian identity is
    byte-identical to the uniform current/contact transaction so a future
    head/body loader can require exact equality.  The path identity also
    binds the finite-segment quadrature order and Ward tolerances; it is the
    numerical certificate for this realization, not a second body-only
    Hamiltonian identity.

    ``iq_irr`` and ``q_irr_kgrid_int`` retain the symmetry service's IBZ row
    identity; ``q_crys`` is its BGW signed fractional representative.  The
    response consumer can therefore keep a one-row block attached to its
    storage label without rebuilding a q grid.

    ``basis_receipt`` is the exact immutable object supplied by the target
    wavefunction bundle.  The producer authenticates it against the WFN,
    physical band interval, FFT grid, ordered centroid table and live padded
    extent, then propagates that same object rather than inferring provenance
    from the two face shapes.

    This NamedTuple is an orchestration record, not a compiled operand: its
    q labels and fingerprints are host strings/NumPy arrays.  The producer
    compiles only numerical inputs and constructs this record afterward;
    the private response oracle validates it and extracts its arrays before
    calling the cached Green kernel.
    """

    current_nmu: jax.Array
    current_mun: jax.Array
    n_rmu_logical: int
    iq_irr: int
    q_irr_kgrid_int: np.ndarray
    q_crys: np.ndarray
    kminq_idx: np.ndarray
    g_wrap: np.ndarray
    vnl_ward_residual_abs: jax.Array
    vnl_ward_residual_rel: jax.Array
    vnl_ward_reference_norm: jax.Array
    hamiltonian_config_operator_fingerprint: str
    vnl_path_operator_fingerprint: str
    # Appended with a default so both pre-receipt positions AND constructor
    # arity remain compatible for readers treating this as a positional row.
    basis_receipt: object = None


def uniform_gauge_operator(geom: SweepGeometry, *, bvec, blat,
                           vnl_setup, include_contact: bool = True,
                           include_transfer_q2: bool = False,
                           kinetic_balance_lift: str = "raw") -> Operator:
    r"""One apply-to-ket owner for current and exact uniform contact.

    The first three packed components are

    ``Gamma_i = alpha_i + (alpha_FS/2) dV_NL/dK_i``

    on the kinetic-balance bispinor.  Contracting the ``alpha_i`` term with
    that bispinor is identically ``(alpha_FS/2) dT/dK_i``; no second
    sigma.p spelling is introduced here.  With ``include_contact=True``
    (the default), the final nine components are

    ``Lambda_ab = (alpha_FS/2) d2(T+V_NL)/dK_a dK_b``.

    Kinetic contact comes from :mod:`psp.dft_operators`; the exact-origin,
    row/G-bounded VNL current and contact come from :mod:`psp.vnl_ops`.
    Everything is evaluated inside one :func:`sweep_matrix_elements` scan,
    so the ψ all-to-all and projector coefficient pass are not paid by
    separate current/contact drivers.

    ``kinetic_balance_lift`` names the representation already carried by
    ``psi_n``.  The historical default is raw and remains on its old exact
    operations and cache key.  ``include_transfer_q2=True`` extends the SAME
    transaction with the explicit ICL transfer derivatives.  For the raw
    representation and repository's bra ``k-q`` orientation,

    ``Q_raw[i,a] = -(alpha/2) sigma_a sigma_i
                    -(alpha/4) V_NL,ia``

    and ``Q2_raw[i,a,b] = (alpha/6) V_NL,iab``.  For the isometric
    representation the same product rule additionally contains the analytic
    first/second derivatives of the normalized bra endpoint, including their
    cross terms with the ICL path derivative.  Derivative families are
    consumed one Cartesian row at a time inside this operator; no three- or
    nine-WFN jet escapes it.  Both branches reuse
    :func:`common.bispinor_init.kinetic_balance_lift_jet` rather than spelling
    a second sigma-product or normalization derivative.

    These are explicit vertex derivatives only.  They do not include
    eigenstate, energy, occupation, or response-weight derivatives and are
    not by themselves a generalized long-wave response.
    """
    if int(geom.ns) != 4:
        raise ValueError(
            "uniform_gauge_operator requires the canonical four-component "
            f"kinetic-balance WFN carrier; geom.ns={int(geom.ns)}")
    if vnl_setup is None:
        raise ValueError(
            "uniform_gauge_operator requires the canonical VNLSetup; a "
            "kinetic-only transaction cannot certify pseudopotential current")
    if int(vnl_setup.nspinor) != 2:
        raise ValueError(
            "uniform_gauge_operator requires a two-component Pauli VNLSetup; "
            f"got nspinor={int(vnl_setup.nspinor)}")
    transfer_q2 = bool(include_transfer_q2)
    contact_enabled = bool(include_contact or transfer_q2)
    if contact_enabled and vnl_setup.Gpp_table is None:
        raise ValueError(
            "uniform_gauge_operator requires VNLSetup built with "
            "compute_contact=True")
    if transfer_q2 and vnl_setup.Gppp_table is None:
        raise ValueError(
            "uniform_gauge_operator transfer q2 requires VNLSetup built "
            "with compute_transfer_q2=True")

    from common.bispinor_init import (
        HALFALPHA,
        ISOMETRIC_KINETIC_BALANCE_LIFT,
        RAW_KINETIC_BALANCE_LIFT,
        kinetic_balance_lift_jet,
        kinetic_balance_lift_provenance,
    )
    from common.gamma_matrices import gamma_apply, gamma_perm_phase
    from psp.dft_operators import apply_kinetic_contact_to_ket
    from psp import vnl_ops

    B_host = np.asarray(bvec, dtype=np.float64) * float(blat)
    if not np.array_equal(B_host, np.asarray(vnl_setup.B, dtype=np.float64)):
        raise ValueError(
            "uniform_gauge_operator reciprocal lattice differs from the "
            "VNLSetup used to differentiate the Hamiltonian")
    alpha_vertices = tuple(gamma_perm_phase(i) for i in (1, 2, 3))
    halfalpha = jnp.asarray(HALFALPHA, dtype=jnp.float64)
    lift_mode = str(kinetic_balance_lift).strip().lower()
    lift_provenance = kinetic_balance_lift_provenance(lift_mode)
    isometric_lift = lift_mode == ISOMETRIC_KINETIC_BALANCE_LIFT
    if lift_mode not in (
            RAW_KINETIC_BALANCE_LIFT, ISOMETRIC_KINETIC_BALANCE_LIFT):
        raise AssertionError("kinetic-balance lift owner admitted a bad mode")
    B = jnp.asarray(B_host, dtype=jnp.float64)

    def op(psi_n, gvec, gmask, bidx, kvec):
        del bidx
        psi_4 = _ket(psi_n, gmask)
        psi_L = psi_4[:, :2, :]

        # The alpha matrices are the incumbent monomial gamma owner.  The
        # input was lifted by WfnLoader through bispinor_init.lift_to_4spinor,
        # so this contraction consumes (rather than reimplements) sigma.p.
        gamma_kin = jnp.stack([
            gamma_apply(psi_4, perm, phase, axis=1)
            for perm, phase in alpha_vertices
        ], axis=0)

        vnl = vnl_ops.apply_icl_vnl_transfer_jet_to_ket(
            psi_L, gvec, kvec, vnl_setup, gmask,
            include_contact=contact_enabled, include_q2=transfer_q2)
        gamma_vnl = _pad_spinor(
            halfalpha.astype(psi_4.real.dtype)
            * vnl.gamma0_cart_ket,
            int(psi_4.shape[1]))
        gamma = gamma_kin + gamma_vnl

        fields = [gamma]
        if contact_enabled:
            lambda_kin = apply_kinetic_contact_to_ket(psi_L)
            lambda_large = halfalpha.astype(psi_4.real.dtype) * (
                lambda_kin + vnl.lambda0_cart_ket)
            contact = _pad_spinor(lambda_large, int(psi_4.shape[1]))
            fields.append(contact.reshape(9, *contact.shape[2:]))
        if transfer_q2:
            if not isometric_lift:
                # Historical raw action, byte-for-byte: dPsi/dK is
                # independent of K, so the zero endpoint avoids another
                # reciprocal-lattice operand.  q2 is now named uniformly as
                # an adjoint source; Hermiticity makes its physical value
                # unchanged at the sole post-sweep orientation boundary.
                _lifted_zero, dpsi_dK = kinetic_balance_lift_jet(
                    psi_L,
                    jnp.zeros((int(psi_L.shape[-1]), 3),
                              dtype=psi_4.real.dtype))
                del _lifted_zero
                kinetic_q1_source = jnp.stack([
                    gamma_apply(dpsi_dK, perm, phase, axis=2)
                    for perm, phase in alpha_vertices
                ], axis=0)
                vnl_q1 = _pad_spinor(
                    halfalpha.astype(psi_4.real.dtype)
                    * vnl.dgamma_dq_cart_ket,
                    int(psi_4.shape[1]))
                q1_adjoint_source = kinetic_q1_source - vnl_q1
                q2_sweep_source = _pad_spinor(
                    halfalpha.astype(psi_4.real.dtype)
                    * vnl.d2gamma_dq2_cart_ket,
                    int(psi_4.shape[1]))
            else:
                # The input already carries r(K).  Build each analytic
                # endpoint family from that carrier, consume it immediately,
                # and retain only the incumbent packed vertex action.
                K_cart = (
                    gvec.astype(jnp.float64) + kvec[None, :]) @ B
                dpsi_vnl_qderivative = []
                q1_rows = []
                for a in range(3):
                    dpsi_a = kinetic_balance_lift_jet(
                        psi_L, K_cart, representation=lift_mode,
                        cartesian_K_derivative_axes=(a,))
                    vnl_a = vnl_ops.apply_icl_vnl_transfer_jet_to_ket(
                        dpsi_a[:, :2], gvec, kvec, vnl_setup, gmask,
                        include_contact=True, include_q2=False)
                    alpha_a = jnp.stack([
                        gamma_apply(dpsi_a, perm, phase, axis=1)
                        for perm, phase in alpha_vertices
                    ], axis=0)
                    endpoint_vnl_a = _pad_spinor(
                        halfalpha.astype(psi_4.real.dtype)
                        * vnl_a.gamma0_cart_ket,
                        int(psi_4.shape[1]))
                    q1_rows.append(alpha_a + endpoint_vnl_a)
                    # Only this q-derivative is needed by the symmetric q2
                    # cross terms.  Do not retain the full VNL result (or the
                    # consumed derivative WFN) across families.
                    dpsi_vnl_qderivative.append(
                        vnl_a.dgamma_dq_cart_ket)
                kinetic_q1_source = jnp.stack(q1_rows, axis=1)
                vnl_q1 = _pad_spinor(
                    halfalpha.astype(psi_4.real.dtype)
                    * vnl.dgamma_dq_cart_ket,
                    int(psi_4.shape[1]))
                q1_adjoint_source = kinetic_q1_source - vnl_q1

                q2_families = {}
                for a in range(3):
                    for b in range(a, 3):
                        d2psi_ab = kinetic_balance_lift_jet(
                            psi_L, K_cart, representation=lift_mode,
                            cartesian_K_derivative_axes=(a, b))
                        vnl_ab = (
                            vnl_ops.apply_icl_vnl_transfer_jet_to_ket(
                                d2psi_ab[:, :2], gvec, kvec, vnl_setup,
                                gmask, include_contact=False,
                                include_q2=False))
                        alpha_ab = jnp.stack([
                            gamma_apply(
                                d2psi_ab, perm, phase, axis=1)
                            for perm, phase in alpha_vertices
                        ], axis=0)
                        endpoint_vnl_ab = _pad_spinor(
                            halfalpha.astype(psi_4.real.dtype)
                            * (vnl_ab.gamma0_cart_ket
                               - dpsi_vnl_qderivative[a][:, b]
                               - dpsi_vnl_qderivative[b][:, a]
                               + vnl.d2gamma_dq2_cart_ket[:, a, b]),
                            int(psi_4.shape[1]))
                        q2_families[a, b] = alpha_ab + endpoint_vnl_ab
                q2_sweep_source = jnp.stack(tuple(
                    jnp.stack(tuple(
                        q2_families[min(a, b), max(a, b)]
                        for b in range(3)), axis=1)
                    for a in range(3)), axis=1)
                # This is an adjoint source: the physical bra-endpoint q2 is
                # its band-space adjoint after the sole m/n contraction,
                # exactly as q1 is minus the adjoint of its source.  That
                # orientation cannot be performed here without materializing
                # an nb-by-nb object inside the apply-to-ket operator.
            fields.extend((
                q1_adjoint_source.reshape(
                    9, *q1_adjoint_source.shape[2:]),
                q2_sweep_source.reshape(
                    27, *q2_sweep_source.shape[3:]),
            ))

        packed = jnp.concatenate(tuple(fields), axis=0)
        return jnp.moveaxis(packed, 0, -1)[None]

    operator_key = (
        ("uniform_gauge_current_contact" if contact_enabled
         else "uniform_gauge_current"), geom.ngkmax, geom.ns,
        float(blat), id(vnl_setup))
    if transfer_q2:
        operator_key += ("explicit_transfer_q2",)
    if isometric_lift:
        operator_key += ("kinetic_balance", lift_provenance)
    return Operator(
        apply=op, post=1.0,
        ncomp=(48 if transfer_q2 else (12 if contact_enabled else 3)),
        key=operator_key)


def _gauge_hamiltonian_operator_fingerprint(
    *, wfn, vnl_setup, band_start: int, band_stop: int,
    geom: SweepGeometry, include_transfer_q2: bool,
    kinetic_balance_lift: str = "raw",
) -> str:
    """Compose the one uniform/finite-q Hamiltonian operator identity.

    This is the exact grammar historically in the complete uniform-gauge
    sweep, moved without changing a byte so the arbitrary-transfer endpoint
    could not invent a near-duplicate body fingerprint.  (Both of those
    consumers -- ``sweep_uniform_gauge_matrix_elements`` and
    ``finite_transfer_current_to_centroids`` -- were deleted on 2026-09-02;
    the one surviving caller is
    :func:`sweep_uniform_current_matrix_elements`.)  The finite-path quadrature/tolerance identity stays a
    separate certificate owned by :mod:`psp.vnl_ops`.
    """
    start, stop = int(band_start), int(band_stop)
    vnl_fingerprint = str(
        getattr(vnl_setup, "uniform_gauge_fingerprint", "")).strip()
    if (not vnl_fingerprint.startswith("sha256:")
            or len(vnl_fingerprint) != len("sha256:") + 64
            or any(c not in "0123456789abcdef"
                   for c in vnl_fingerprint[7:])):
        raise ValueError(
            "gauge-current transaction requires the canonical VNLSetup "
            "content fingerprint; rebuild it with build_vnl_setup(..., "
            "compute_contact=True)")

    import hashlib
    from common.bispinor_init import (
        ISOMETRIC_KINETIC_BALANCE_LIFT,
        kinetic_balance_lift_provenance,
    )
    from common.parallel_transport import (
        WFN_FINGERPRINT_SCHEME, fingerprint_update_value, wfn_fingerprint)
    from psp import vnl_ops

    lift_mode = str(kinetic_balance_lift).strip().lower()
    lift_provenance = kinetic_balance_lift_provenance(lift_mode)

    digest = hashlib.sha256()
    digest.update(b"lorrax.uniform_gauge_operator/v1\0")
    for label, value in (
        ("wfn_scheme", WFN_FINGERPRINT_SCHEME),
        ("wfn", wfn_fingerprint(wfn)),
        ("vnl", vnl_fingerprint),
        ("vnl_gauge_path", vnl_ops.ICL_STRAIGHT_GAUGE_PATH),
        ("kinetic_balance", lift_provenance),
        ("band_interval", f"{start}:{stop}"),
        ("nk", str(int(geom.nk))),
        ("cell_volume", float(geom.cell_volume).hex()),
    ):
        fingerprint_update_value(digest, label, value)
    if bool(include_transfer_q2):
        fingerprint_update_value(
            digest, "transfer_jet",
            ("explicit_q2_isometric_endpoint_v1"
             if lift_mode == ISOMETRIC_KINETIC_BALANCE_LIFT
             else "explicit_q2_fixed_large_component_v1"))
    return "sha256:" + digest.hexdigest()


def _uniform_gauge_sweep_fingerprint(
    *, wfn, vnl_setup, band_start: int, band_stop: int,
    geom: SweepGeometry, include_transfer_q2: bool,
    kinetic_balance_lift: str = "raw",
) -> str:
    """Validate one uniform sweep manifold and return its sole identity."""
    start, stop = int(band_start), int(band_stop)
    if start < 0 or stop <= start or stop > int(wfn.nbands):
        raise ValueError(
            "uniform gauge band interval must satisfy "
            f"0 <= start < stop <= WFN.nbands; got [{start},{stop})")
    if stop - start != int(geom.nb_logical):
        raise ValueError(
            "uniform gauge band interval does not match SweepGeometry: "
            f"[{start},{stop}) vs nb_logical={int(geom.nb_logical)}")
    return _gauge_hamiltonian_operator_fingerprint(
        wfn=wfn, vnl_setup=vnl_setup, band_start=start, band_stop=stop,
        geom=geom, include_transfer_q2=bool(include_transfer_q2),
        kinetic_balance_lift=kinetic_balance_lift)


def sweep_uniform_current_matrix_elements(
    psi_G,
    *,
    wfn,
    band_start: int,
    band_stop: int,
    geom: SweepGeometry,
    bvec,
    blat,
    vnl_setup,
    gvecs,
    gmask,
    box_index,
    kvecs,
    use_scan: bool = True,
    kinetic_balance_lift: str = "raw",
) -> UniformGaugeCurrentMatrixElements:
    """Select only ``Gamma_raw`` from the canonical uniform-gauge sweep.

    The operator closure, band-layout VNL action and fingerprint owner
    are :func:`_uniform_gauge_operator_identity` and its siblings; this is
    now their ONLY caller (the complete sweep that produced contact and
    transfer components was deleted on 2026-09-02, unreachable from any
    production path).  A Hall-only producer does not retain contact/response
    matrices that it cannot consume.
    """
    fingerprint = _uniform_gauge_sweep_fingerprint(
        wfn=wfn, vnl_setup=vnl_setup, band_start=band_start,
        band_stop=band_stop, geom=geom, include_transfer_q2=False,
        kinetic_balance_lift=kinetic_balance_lift)
    gamma_raw = sweep_matrix_elements(
        psi_G,
        geom=geom,
        operator=uniform_gauge_operator(
            geom, bvec=bvec, blat=blat, vnl_setup=vnl_setup,
            include_contact=False, include_transfer_q2=False,
            kinetic_balance_lift=kinetic_balance_lift),
        gvecs=gvecs,
        gmask=gmask,
        box_index=box_index,
        kvecs=kvecs,
        use_scan=use_scan,
    )
    return UniformGaugeCurrentMatrixElements(
        gamma_raw=gamma_raw,
        hamiltonian_config_operator_fingerprint=fingerprint)


def sum_operators(*ops: Operator) -> Operator:
    """``(O₁ + O₂ + …) ∘ ψ`` — one sweep, one all-to-all, one reduction.

    ``⟨m|T+V_loc+V_NL|n⟩`` is ONE matrix element, so it is one sweep.  Each
    slot sums over the terms that fill it: band-layout kets add before
    their all-to-all, G-split kets add before the slab GEMM, and separable
    terms keep their own projections (concatenated into one psum) and add
    their coupled blocks.  So T+V_loc+V_NL moves ψ and V_loc ψ once and
    runs one GEMM, where three sweeps would pay all of it three times.

    Each term's ``post`` is folded into its own contribution, which is
    what lets operators with different normalisations (the local
    potential's ``sqrt(1/Ω)``, the others 1) share one contraction.
    Algebraically identical, since the contraction is linear; numerically
    one scalar multiply moves from after the G sum to before it, ~1 ulp.
    """
    if not ops:
        raise ValueError("sum_operators: at least one operator required")
    ncomp = int(ops[0].ncomp)
    bad = [i for i, o in enumerate(ops) if int(o.ncomp) != ncomp]
    if bad:
        raise ValueError(
            f"sum_operators: operands disagree on ncomp — operand 0 has "
            f"{ncomp}, operands {bad} do not.  A scalar and a Cartesian "
            f"operator do not add.")

    # Each term keeps its OWN consts; the sum concatenates them and hands
    # every term back its own span, so one operand list serves the sweep and
    # no term learns anything about its neighbours.
    spans, off = [], 0
    for o in ops:
        spans.append((off, off + len(o.consts)))
        off += len(o.consts)
    all_consts = tuple(c for o in ops for c in o.consts)

    def _scaled(o, x):
        return x if o.post == 1.0 else x * o.post

    def _ket_sum(slot):
        terms = [(span, o) for span, o in zip(spans, ops)
                 if getattr(o, slot) is not None]
        if not terms:
            return None

        def summed(*args):
            # (layout operands..., *all_consts): split at the consts.
            cut = len(args) - len(all_consts)
            head, cs = args[:cut], args[cut:]
            acc = None
            for (a, b), o in terms:
                t = _scaled(o, getattr(o, slot)(*head, *cs[a:b]))
                acc = t if acc is None else acc + t
            return acc
        return summed

    sep = [(span, o) for span, o in zip(spans, ops) if o.coeffs is not None]
    coeffs = couple = None
    if sep:
        def coeffs(psi, gvec, gmask, kvec, *cs):
            return tuple(o.coeffs(psi, gvec, gmask, kvec, *cs[a:b])
                         for (a, b), o in sep)

        def couple(bra, ket, *cs):
            acc = None
            for i, ((a, b), o) in enumerate(sep):
                t = _scaled(o, o.couple(bra[i], ket[i], *cs[a:b]))
                acc = t if acc is None else acc + t
            return acc

    return Operator(apply=_ket_sum('apply'), apply_g=_ket_sum('apply_g'),
                    coeffs=coeffs, couple=couple, post=1.0, ncomp=ncomp,
                    consts=all_consts,
                    key=('sum',) + tuple(_operator_key(o) for o in ops))


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

class SweepPlan(NamedTuple):
    """Static shapes of one sweep executable (:func:`plan_sweep`).

    k_tile
        k-points per scan step; divides ``geom.nk``.  One all-to-all, one
        slab GEMM and one reduce-scatter serve the whole tile.
    g_carrier
        ``ngkmax`` padded to a multiple of P: the extent the all-to-all
        splits into P slabs.  Pad columns are zero in ψ and in the mask.
    band_chunk
        bands per application of a band-layout operator: this rank's
        ``nb/P`` when its FFT boxes fit the budget, else the fewest chunks
        that do (:func:`plan_sweep`).
    """
    k_tile: int
    g_carrier: int
    band_chunk: int


def plan_sweep(geom: SweepGeometry, operator) -> SweepPlan:
    """The G carrier and the k tile.

    THE k TILE PACKS k ONLY WHILE THE ALL-TO-ALL IS LATENCY-BOUND.  A tile
    buys fewer collective rounds and scan trips; it costs a K-times larger
    working set, and at VI3 12x12 that cost is real: at P4 every K > 1 was
    slower than K = 1 (V_H 2.48 s at K=1, 2.57–2.64 s at K=2..16; dipole-p
    0.28 s against 0.33–0.47 s) and the sphere-sized K=24 tile raised the
    executable from 9.9 to 17.4 GiB (``runs/runtime/mtxel_sweep_20260923``,
    legs a11–a13).  So ``K`` is the SMALLEST divisor of ``nk`` whose
    per-peer all-to-all block, ``K·(nb/P)·ns·g_carrier·16 / P``, reaches
    :data:`A2A_BANDWIDTH_BLOCK_BYTES`, never more than the largest divisor
    whose step fits in this rank's resident ψ sphere (the density scan's
    memory rule), and ``K = 1`` at ``nk = 1``.  Measured at P16 over OFI
    (legs a22/a24): at the production window (1.2 MB per-peer blocks) the
    rule picks K = 1 and K = 2 would be ~4% faster for ~2% more executable
    memory; in a 16-band window (145 KB blocks) it picks K = 8, 12–18%
    faster than K = 1.

    One k of a step holds, per rank: ψ and its G-split copy, the operator's
    ket in band layout and in G-split layout (``c = max(ncomp, 1)``
    components each) and the ``(c, nb, nb)`` slab partial plus its
    block-ordered copy.  The FFT box of a band operator is NOT in the step:
    it runs one k at a time inside the tile and never scales with K.
    """
    from runtime.padding import bounded_partition_tile, padded_axis

    c128 = 16.0
    n_ranks = int(geom.p_prod)
    nb_rank = geom.nb // n_ranks
    g_carrier = padded_axis(
        geom.ngkmax, geom.mesh, name="matrix-element sweep G all-to-all",
        spec=P(None, None, None, ("x", "y")), axis=3).carrier
    ops = (operator,) if isinstance(operator, Operator) else tuple(operator)
    c = sum(max(int(o.ncomp), 1) for o in ops)
    ket_copies = sum(
        (2 * max(int(o.ncomp), 1) if o.apply is not None else 0)
        + (max(int(o.ncomp), 1) if o.apply_g is not None else 0)
        for o in ops)
    sphere = geom.nk * nb_rank * geom.ns * geom.ngkmax * c128
    step = ((2 + ket_copies) * nb_rank * geom.ns * g_carrier * c128
            + 2.0 * c * float(geom.nb) ** 2 * c128)
    k_mem = max(1, bounded_partition_tile(
        geom.nk, max(1, int(sphere // step)), 1))
    peer_block = nb_rank * geom.ns * g_carrier * c128 / n_ranks
    k_tile = next((k for k in range(1, k_mem + 1) if geom.nk % k == 0
                   and k * peer_block >= A2A_BANDWIDTH_BLOCK_BYTES), k_mem)
    return SweepPlan(k_tile, g_carrier, _band_chunk(geom, ops, nb_rank, k_tile * step))


#: The share of the stage room the band-layout FFT boxes may fill.
_BOX_ROOM_FRACTION = 0.9


def _band_chunk(geom, ops, nb_rank: int, step_bytes: float) -> int:
    """Bands per application of the band-layout operators, from the run's budget.

    A band operator holds FFT boxes of ``ns·N_r·16`` bytes per band: ψ(r), one
    product and one forward transform per component, ``2 + 2·max(ncomp, 1)``
    boxes (the four-current potential: six).  Its bands are independent, so
    the fewest equal chunks whose boxes fit beside the step's live set in the
    stage room (``common.gpu_utils.device_room_bytes``) are applied one after
    another; one chunk, this rank's ``nb/P``, when they fit.  Every process
    computes the same chunk.
    """
    copies = max((2 + 2 * max(int(o.ncomp), 1) for o in ops if o.apply is not None),
                 default=0)
    if not copies:
        return nb_rank
    from common.gpu_utils import device_budget_bytes, device_room_bytes, record_stage_price
    box = copies * float(geom.ns) * float(np.prod(geom.fft_grid)) * 16.0
    room = float(device_room_bytes())
    fit = max(1, int((_BOX_ROOM_FRACTION * room - step_bytes) // box))
    n_chunks = -(-nb_rank // min(fit, nb_rank))
    chunk = -(-nb_rank // n_chunks)
    record_stage_price("matrix-element sweep, plan_sweep",
                       device_budget_bytes() - room + step_bytes + chunk * box)
    return chunk


#: Per-peer all-to-all block (bytes) at which the collective is taken to be
#: bandwidth-bound; :func:`plan_sweep` packs k only below it.  A transport
#: property, not a knob: no deck or env var reaches it.
A2A_BANDWIDTH_BLOCK_BYTES = 1 << 20


def _sweep_body(geom: SweepGeometry, operators: tuple, spans: tuple,
                plan: SweepPlan, *, use_scan: bool):
    """The per-rank program: scan k tiles; per tile, band → G split, GEMM,
    reduce-scatter.

    Local operands (``ops`` dict): ``psi`` ``(nk, nb/P, ns, ngkmax)`` (this
    rank's bands); the replicated ``kvec`` ``(nk, 3)``, ``gvec`` ``(nk,
    ngkmax, 3)`` and ``gmask`` ``(nk, ngkmax)`` — each rank cuts its own G
    slab from them per tile, so no padded ``(nk, g_carrier)`` table ever
    exists — and, for a band-layout operator, the replicated ``bidx``.
    Returns one ``(nk, [ncomp,] nb/p_x, nb/p_y)`` block of
    ``P(None, [None,] 'x', 'y')`` per operator, in order.  Operators share
    the ψ all-to-all and the G slabs; each keeps its own ket, contraction
    and reduction, so its arithmetic is the same as in a sweep of its own.
    ``spans`` slices each operator's ``consts`` out of the flat operand list.

    ORDER CONVENTION, shared with ``band_sphere_spec`` and the density scan:
    the linear rank over ``('x','y')`` is ``x·p_y + y``.  The band
    all-to-all concatenates blocks in that order (so bands come back in
    global order), it hands slab ``r`` of G to rank ``r``, and the
    reduce-scatter delivers the ``r``-th of the ``(p_x, p_y)`` blocks to
    rank ``r`` — i.e. rows ``x`` and columns ``y``.
    """
    from common.contract_bands import (bands_to_contraction_slabs,
                                       reduce_scatter_to_band_block)

    axes = ("x", "y")
    px, py = int(geom.mesh.shape["x"]), int(geom.mesh.shape["y"])
    n_ranks = px * py
    nb, ngk = geom.nb, geom.ngkmax
    nbx, nby = nb // px, nb // py
    gc, K = plan.g_carrier, plan.k_tile
    n_tiles = geom.nk // K
    need_gvec_s = any(o.apply_g is not None or o.coeffs is not None
                      for o in operators)

    def to_g_split(a):
        """(K, nb/P, ns, ngkmax, …) band layout → (K, nb, ns, gc/P, …)."""
        return bands_to_contraction_slabs(a, band_axis=1, slab_axis=3,
                                          carrier=gc, axes=axes)

    def to_blocks(part):
        """(K, [c,] nb, nb) slab partial → this rank's summed (x, y) block
        (``K·c·nb²·16`` bytes per rank: 0.26 MB per k at nb=128)."""
        return reduce_scatter_to_band_block(part, px=px, py=py, axes=axes)

    def band_ket(operator, t, consts):
        """The band-layout operator one k at a time: its box never scales
        with K.  (K, nb/P, ns, ngkmax[, c])."""
        def one(xs):
            p, g, m, b, kv = xs
            nbr, c = int(p.shape[0]), int(plan.band_chunk)
            if c >= nbr:
                return operator.apply(p[None], g, m, b[None], kv, *consts)[0]
            # Band chunks, one after another: the boxes are (c, ns, grid).
            n = -(-nbr // c)
            chunks = jnp.pad(p, ((0, n * c - nbr),) + ((0, 0),) * (p.ndim - 1))
            _, out = jax.lax.scan(
                lambda carry, pc: (carry, operator.apply(pc[None], g, m, b[None], kv,
                                                         *consts)[0]),
                None, chunks.reshape(n, c, *p.shape[1:]), unroll=1)
            return out.reshape(n * c, *out.shape[2:])[:nbr]
        xs = (t["psi"], t["gvec"], t["gmask"], t["bidx"], t["kvec"])
        if K == 1:
            return one(jax.tree_util.tree_map(lambda a: a[0], xs))[None]
        _, out = jax.lax.scan(lambda c_, x: (c_, one(x)), None, xs,
                              unroll=1)
        return out

    def per_k(fn, consts):
        return jax.vmap(lambda *a: fn(*a, *consts))

    def my_slab(a):
        """(K, ngkmax, …) replicated table → this rank's (K, g_slab, …)."""
        pad = [(0, 0)] * a.ndim
        pad[1] = (0, gc - ngk)
        gs = gc // n_ranks
        return jax.lax.dynamic_slice_in_dim(
            jnp.pad(a, pad), jax.lax.axis_index(axes) * gs, gs, axis=1)

    def op_block(operator, t, psi_g, gv_s, gm_s, consts):
        """One operator's (K, [c,] nb/p_x, nb/p_y) block of one tile."""
        contraction = ("kmsg,knsg->kmn" if not operator.ncomp
                       else "kmsg,knsgc->kcmn")
        ket = None
        if operator.apply is not None:
            ket = to_g_split(band_ket(operator, t, consts))
        if operator.apply_g is not None:
            kg = per_k(operator.apply_g, consts)(
                psi_g, gv_s, gm_s, t["kvec"])
            ket = kg if ket is None else ket + kg
        blk = None
        if ket is not None:
            blk = to_blocks(jnp.einsum(contraction, jnp.conj(psi_g), ket,
                                       optimize=True))
        if operator.coeffs is not None:
            # Separable terms: psum the slab projections (R·ns·nb, no G
            # axis), then couple this rank's bra rows and ket columns.
            co = jax.lax.psum(per_k(operator.coeffs, consts)(
                psi_g, gv_s, gm_s, t["kvec"]), axes)
            x0 = jax.lax.axis_index("x") * nbx
            y0 = jax.lax.axis_index("y") * nby

            def cols(start, width):
                return jax.tree_util.tree_map(
                    lambda a: jax.lax.dynamic_slice_in_dim(
                        a, start, width, axis=a.ndim - 1), co)
            sb = per_k(operator.couple, consts)(cols(x0, nbx),
                                                cols(y0, nby))
            blk = sb if blk is None else blk + sb
        return blk * operator.post

    def tile(t, consts):
        gm_s = my_slab(t["gmask"])
        gv_s = my_slab(t["gvec"]) if need_gvec_s else None
        psi_g = to_g_split(t["psi"]) * gm_s[:, None, None, :].astype(
            t["psi"].dtype)
        return tuple(op_block(o, t, psi_g, gv_s, gm_s, consts[a:b])
                     for o, (a, b) in zip(operators, spans))

    def body(ops, *consts):
        tiles = {k: v.reshape(n_tiles, K, *v.shape[1:])
                 for k, v in ops.items()}
        if use_scan:
            _, Hs = jax.lax.scan(lambda c_, t: (c_, tile(t, consts)), None,
                                 tiles, unroll=1)
        else:
            per_tile = [tile({k: v[i] for k, v in tiles.items()}, consts)
                        for i in range(n_tiles)]
            Hs = tuple(jnp.stack(col) for col in zip(*per_tile))
        return tuple(H.reshape(geom.nk, *H.shape[2:]) for H in Hs)

    return body


# FORCED SYNC, AND WHAT IT COSTS.  ``watch=True`` makes the section
# ``block_until_ready`` the returned block before it stops its clock, so the
# row is the sweep's COMPUTE, not its dispatch.  Without it the row would be
# the ~ms it takes to enqueue and the nk·nb²·ngkmax of work would land on
# whichever unrelated stage blocked next — the failure
# ``collectives.sweep_local_k`` records, and the reason
# ``kin_ion_io`` already wraps this call in one section instead of nk.
#
# The sync is free at three of the four call sites (``kin_ion_io``'s two
# sweeps and ``get_dipole_mtxels``): each follows the call with
# ``blocks_to_host``, which gathers and therefore blocks on the next line
# anyway.  At the fourth (``sc_iteration.rebuild_hartree_dft_basis``) it is
# free only on the IBZ path, where ``KStarMap.select`` reads the block back
# to host immediately; on the full-BZ path it is a REAL change — H_vh would
# otherwise stay lazy across ``compute_screening``, so the sweep's device
# work could overlap the screening's host-side compile. That overlap is
# given up on purpose: it is the only way the row is the sweep's own time
# rather than χ₀'s.
@timing.timed("mtxel.sweep", watch=True)
def sweep_matrix_elements(
    psi_G,
    *,
    geom: SweepGeometry,
    operator,
    gvecs,
    gmask,
    box_index,
    kvecs,
    use_scan: bool = True,
):
    """``H[k, m, n] = Σ_{s,G} conj(ψ_mk) (O ∘ ψ)_nk`` for every k.

    Parameters
    ----------
    psi_G : (nk, nb, ns, ngkmax) c128
        The G-sphere ψ, resident on device, at ``band_sphere_spec`` (the
        loader's layout; any other is constrained to it once).
    geom, operator
        See above.  ``operator`` may also be a TUPLE of operators: one
        sweep then returns one block per operator, sharing the ψ read, the
        ψ all-to-all and the G slabs (e.g. ⟨m|T+V_loc+V_NL|n⟩ and ⟨m|v|n⟩
        from one pass), each block computed exactly as its own sweep would.
    gvecs : (nk, ngkmax, 3) i32
        The loader's own fixed-shape table (D10) — ``PaddedGVectors.gvecs``.
    gmask : (nk, ngkmax) f64
        Its pad mask.  MANDATORY, not optional.  Pad rows carry the
        FFT-box sentinel Miller index (see ``common.gvec_fft_box``), which
        is a valid box cell, so a forgotten mask does not crash — it
        silently contracts the sentinel column into every matrix element.
        The sentinel is chosen so that no physical G of a padded row maps
        to it, which makes the omission detectable rather than harmless;
        it does not make the mask optional.
    box_index : (nk, ngkmax) i32
        Sphere→box index map (``WfnLoader.box_index``).  Only consumed —
        and only moved to the device — for an operator with a band-layout
        ``apply`` (an FFT, or the DFT+U velocity ket); every other operator
        ignores it.
    kvecs : (nk, 3) f64
    use_scan : bool
        ``True`` (default) runs ``lax.scan`` over k tiles — one lowering
        for the whole sweep.  ``False`` runs the identical tile body in a
        Python loop: same arithmetic and collectives, different control
        flow, so a disagreement isolates the scan itself.

    Returns
    -------
    (nk, nb, nb) c128 sharded ``P(None, 'x', 'y')`` — or, for an operator
    with ``ncomp > 0``, ``(nk, ncomp, nb, nb)`` sharded
    ``P(None, None, 'x', 'y')``.  Band extents are the mesh-PADDED
    ``geom.nb``; :func:`blocks_to_host` is the boundary that trims back to
    logical.  The one transient that is not ``1/P`` is a tile's
    ``(K, ncomp, nb, nb)`` slab partial, priced in :func:`_sweep_body`.

    MASKING IS IMPLICIT AND UNCONDITIONAL
    -------------------------------------
    ψ is masked on its G slab before any G-split slot or the contraction
    sees it, and every band-layout operator masks its own output.  A corner
    sentinel makes the bra mask redundant only while BOTH operands' pad
    entries come from stored sphere coefficients — ``phi_G`` at the
    sentinel is not zero, and ``Z`` is finite there — which is a padding
    question no caller should have to reason about, so there is no flag.

    A BAND WINDOW NEEDS NO ARGUMENT
    -------------------------------
    ``⟨m|O|n⟩`` takes BOTH indices from one ψ, so restricting the sweep to a
    band window is a windowed ψ plus a geometry built at the window's
    ``nb``::

        psi_win = wfn.load(bands=(lo, hi), sharding=band_sphere_spec(), ...)
        sweep_matrix_elements(psi_win, geom=SweepGeometry(
            ..., nb=hi - lo), ...)      ->  (nk, hi-lo, hi-lo)

    and the block comes back at WINDOW indices.  A ``band_window=`` argument
    would be a second way to say the same thing (the 2026-08-04 SlabIO
    padding ruling), so there deliberately is none.

    READ THE WINDOW; DO NOT SLICE A RESIDENT ψ TO IT.  An eager
    ``psi_G[:, lo:hi]`` on the ``('x','y')``-sharded band axis lowers to a
    dynamic slice with a runtime start, and the partitioner resolves that by
    REPLICATING the result: the whole window on every rank.  That was
    124.52 GiB per rank on VI3 12x12 at P100 (``gw.sc_iteration.
    _dft_psi_sphere`` has the evidence).  The loader shards a window as it
    reads it, at exactly this sweep's carrier.
    """
    from common.shard_map import shard_map

    mesh = geom.mesh
    nk = geom.nk

    psi = jnp.asarray(psi_G, dtype=jnp.complex128)
    if psi.shape[1] not in (geom.nb_logical, geom.nb):
        raise ValueError(
            f"sweep_matrix_elements: psi_G band axis must be the logical "
            f"nb={geom.nb_logical} or the mesh-padded nb={geom.nb}; got "
            f"{tuple(psi.shape)}")
    if psi.shape[0] != nk or psi.shape[2] != geom.ns \
            or psi.shape[3] != geom.ngkmax:
        raise ValueError(
            f"sweep_matrix_elements: psi_G must be "
            f"(nk, nb, ns, ngkmax) = "
            f"({nk}, {geom.nb_logical}, {geom.ns}, {geom.ngkmax}), "
            f"got {tuple(psi.shape)}")
    single = isinstance(operator, Operator)
    operators = (operator,) if single else tuple(operator)
    for o in operators:
        if o.apply is None and o.apply_g is None and o.coeffs is None:
            raise ValueError(
                "sweep_matrix_elements: an operator fills no slot")
        if (o.coeffs is None) != (o.couple is None):
            raise ValueError(
                "sweep_matrix_elements: a separable operator needs both "
                "coeffs and couple")

    # THE BAND PAD, applied here so no caller states it (SlabIO ruling,
    # decisions.md 2026-08-04: padding is the infrastructure's business).
    # Pad bands are ψ = 0, so the extra rows AND columns of ⟨m|O|n⟩ are
    # exactly zero -- the product of an exact zero, not "close to zero".
    psi = pad_axis(psi, geom.p_prod, axis=1).array

    plan = plan_sweep(geom, operators)
    band = any(o.apply is not None for o in operators)
    rep = P()

    gvecs_j = jnp.asarray(gvecs, dtype=jnp.int32)
    gmask_j = jnp.asarray(gmask, dtype=jnp.float64)
    kvecs_j = jnp.asarray(kvecs, dtype=jnp.float64)
    # The box index is the largest table here (nk·N_r int32, replicated —
    # 759 MiB at VI3 12x12) and only a transforming operator reads it.
    bidx_j = jnp.asarray(box_index, dtype=jnp.int32) if band else None
    # The operator's runtime operands.  They are jit ARGUMENTS, so one
    # executable serves every value of them; anything the operator closes
    # over instead is a jaxpr constant and forces a lowering per value.
    op_consts, spans = [], []
    for o in operators:
        spans.append((len(op_consts), len(op_consts) + len(o.consts)))
        op_consts.extend(jnp.asarray(c) for c in o.consts)
    op_consts = tuple(op_consts)

    out_spec = tuple(geom.spec_block_for(int(o.ncomp)) for o in operators)
    body = _sweep_body(geom, operators, tuple(spans), plan,
                       use_scan=bool(use_scan))

    def _run(psi, gvecs_, gmask_, kvecs_, bidx_, *consts_):
        ops = {"psi": jax.lax.with_sharding_constraint(
                   psi, NamedSharding(mesh, geom.spec_sphere_xy)),
               "kvec": kvecs_, "gvec": gvecs_, "gmask": gmask_}
        specs = {"psi": geom.spec_sphere_xy, "kvec": rep, "gvec": rep,
                 "gmask": rep}
        if band:
            ops["bidx"] = bidx_
            specs["bidx"] = rep
        per_rank = shard_map(
            body, mesh=mesh,
            in_specs=(specs,) + (rep,) * len(consts_),
            out_specs=out_spec, check_vma=False)
        return per_rank(ops, *consts_)

    # NO ``donate_argnums``: ψ's lifetime belongs to the CALLER —
    # ``gw.sc_iteration._PSI_G_CACHE`` holds the SAME ψ across every
    # density-SC iteration, so a blanket donation would invalidate a buffer
    # the next iteration reads.
    fn = _cached_jit(
        'sweep_matrix_elements',
        (psi.shape, geom.ngkmax, geom.ns, nk, bool(use_scan),
         tuple((_operator_key(o), float(o.post), int(o.ncomp))
               for o in operators),
         geom.fft_grid, tuple(plan), mesh, _sharding_key(psi)[1],
         None if bidx_j is None else tuple(bidx_j.shape),
         tuple((tuple(int(d) for d in c.shape), str(c.dtype))
               for c in op_consts)),
        lambda: jax.jit(_run))
    blocks = fn(psi, gvecs_j, gmask_j, kvecs_j, bidx_j, *op_consts)
    return blocks[0] if single else blocks


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------

def _host_chunk_bytes() -> int:
    """Payload cap for one collective in :func:`blocks_to_host`.

    THE SAME KNOB AS THE ROUTINE THIS ONE MIRRORS, not a second number.
    ``LORRAX_COLLECTIVE_CHUNK_MB`` is the calibrated per-instruction
    transport cap (default 128 MB; ``docs/dev/env_vars.md``), and
    ``common.collectives._owner_gather_chunk_bytes`` is its one resolver —
    imported here rather than re-derived, so a recalibration reaches this
    path too.  A hard-coded ``1 << 28`` stood here until 2026-08-05, which
    made this the only chunked collective in the tree that ignored the dial
    it claimed to follow.

    ONE DIFFERENCE IN HOW THE CAP IS SPENT, stated because in both places it
    means PER-RANK TRANSIENT: ``gather_indexed_blocks_to_owner`` divides the
    cap by ``world`` because ``process_allgather`` stacks one copy per rank;
    the gather below is a reshard to a fully replicated spec, whose
    transient is ONE chunk however large P is.  So the chunk extent here is
    ``cap // per_k`` with no world factor, and the two are consistent, not
    divergent.
    """
    from common.collectives import _owner_gather_chunk_bytes
    return _owner_gather_chunk_bytes()


def blocks_to_host(H, *, nb: int, owner_only: bool = False):
    """Sharded ``H[k, …, m_X, n_Y]`` → a host ``numpy`` array.

    THE BOUNDARY IS EXPLICIT AND IT IS NOT FREE.  ``sweep_matrix_elements``
    returns the block SHARDED, which is the point: no rank holds a full
    ``(nb, nb)``.  Every consumer that keeps its result in that layout
    (``gw.sc_iteration.rebuild_hartree_dft_basis``) must NOT call this.
    It exists for the sinks that cannot take a sharded operand — today the
    serial ``h5py`` writes in ``gw.kin_ion_io`` and
    ``psp.get_dipole_mtxels`` — and it re-materialises the replicated
    ``(nk, nb, nb)`` on the ranks that keep it.  The live
    ``gw.sigma_dispatch`` G-space route does not cross this boundary; its
    star broadcast and basis rotation retain ``P(None,'x','y')``.  What the
    sweep removes upstream of here is the per-k full-band FFT box and the
    ``P ≤ nk`` ceiling; an artifact writer still has to hold its table.

    ``owner_only=True`` keeps it on rank 0 and returns ``None`` elsewhere
    — the same contract ``collectives.gather_k_blocks(owner_only=True)``
    offers, and for the same reason (the only consumer is the rank-0 file
    write).  The gather runs in leading-axis chunks so a peer's transient
    is one chunk rather than the whole table; the chunk count is derived
    from replicated shapes, so every rank enters the same number of
    collectives.

    ``nb`` is the LOGICAL band count.  The sweep's output carries the
    mesh-padded extent and the pad rows and columns are exact zeros
    (products of a zero band); trimming them here is the caller stating
    logical shapes only, per decisions.md 2026-08-04.

    WALL W2 SURVIVES HERE, AND WHAT IT WOULD TAKE TO REMOVE IT
    ----------------------------------------------------------
    The handoff's §6.4 is open: W1 (the per-k full-band FFT box) and W3
    (the ``P ≤ nk`` ceiling) are gone, the replicated ``(nk, nb, nb)`` is
    not — this function re-materialises it.  Scoped, because a partial
    version that gathers anyway would be worse than an honest boundary:

    * Both h5 sinks (``gw.kin_ion_io.main``, ``psp.get_dipole_mtxels.main``)
      are RAW ``h5py.File`` writes on rank 0, not SlabIO.  ``SlabIO.
      write_slab`` does take a sharded ``jax.Array`` and write it as a
      hyperslab, so converting them is the mechanism, and it would now
      keep its promise: this paragraph used to withhold the conversion
      because "the allgather backend gathers to rank 0 first", which was
      true until 233a830d deleted that backend and the ``slab_io`` router
      with it.  There is one transport left and it writes from the shards,
      so the reason recorded here is spent — what is left is the work.
    * The former third consumer is converted: the live G-space source returns
      ``P(None,'x','y')`` and rotates it without a host or replicated seam.
      W2 therefore survives only at the two serial artifact writers named
      above, not in the in-memory driver path.

    Converting those two writers is separate I/O work this module does not
    own.  Recorded rather than half-landed.
    """
    from common.collectives import gather_to_host, process_rank

    nk = int(H.shape[0])
    tail = tuple(int(s) for s in H.shape[1:])
    per_k = int(np.prod(tail)) * 16
    step = max(1, min(nk, _host_chunk_bytes() // max(per_k, 1)))
    keep = (not owner_only) or process_rank() == 0

    mesh = H.sharding.mesh
    rep = NamedSharding(mesh, P(*([None] * H.ndim)))
    out = None
    for a in range(0, nk, step):
        chunk = H if step >= nk else H[a:a + step]
        # XLA's own all-gather, not a host-side one: constraining to a
        # fully replicated spec inside a jit is the reshard the transport
        # is certified on, and it lands ``gather_to_host`` on its
        # ``is_fully_replicated`` arm (a local read, no second collective).
        fn = _cached_jit(
            'mtxel_replicate', (chunk.shape, _sharding_key(chunk)),
            lambda: jax.jit(
                lambda x: jax.lax.with_sharding_constraint(x, rep)))
        blk = gather_to_host(fn(chunk))[..., :nb, :nb]
        if keep:
            if out is None:
                out = np.empty((nk,) + blk.shape[1:], dtype=blk.dtype)
            out[a:a + blk.shape[0]] = blk
    return out
