"""One k-scan serving V_H, kinetic+ion and dipole — 2-D band-sharded.

Implements ``docs/dev/matrix_element_sweep_handoff.md``.  Read
``docs/architecture/decisions.md`` first: D10 (fixed-shape ``ngkmax`` G
tables) and the 2026-08-04 SlabIO padding entry are both load-bearing.

WHAT THIS REPLACES
------------------
Three sweeps with one shape between them —
``gw.kin_ion_io._vh_block``, ``._kin_ion_block`` and
``psp.get_dipole_mtxels._dipole_block`` — each calling
``collectives.gather_k_blocks``, which is k-partitioned and returns an
array **identical on every rank**.  Three walls follow (measured in
``docs/dev/rho_vh_2d_design.md`` §1):

  W1  the per-k full-band FFT box: 1.77 GB at b600 bispinor, 37 GB and
      OOM at 12×12, because the local plan takes BOTH sides of the
      matrix element out of one box.
  W2  the replicated ``(nk, nb, nb)``: 829 MB at 12×12, 9.2 GB at
      nb=2000, ×3 for dipole.
  W3  the k-partitioned plan CANNOT USE MORE THAN ``nk`` RANKS. Each rank
      takes whole k, so its wall is one full-band k no matter how large P
      is: adding processors past ``nk`` changes nothing. Measured — the
      before arm is 4.02 s at nb=512/P=16 and 4.87 s at nb=600/P=64,
      i.e. flat in P once scaled for nb (jobs 7888868, 7888877).

PARALLELISM OF THIS DESIGN, STATED ONCE SO IT IS NOT RE-DERIVED WRONG
--------------------------------------------------------------------
The scan processes **one k at a time** and shards **that k's bands over
every process**.  Therefore:

* ``nk`` DOES NOT AFFECT PARALLEL EFFICIENCY.  It is the scan trip count
  and nothing else — it scales total work linearly and the wall linearly,
  exactly as a serial loop bound would.  Any statement of the form "this
  wins when P > nk" is wrong and describes the plan being replaced.
* Efficiency is ``nb_logical / nb_padded`` and nothing else.  The sweep
  keeps scaling until ``P = nb``, where the k-partitioned plan stopped
  scaling at ``P = nk``.
* THE ONLY WAY A RANK GOES IDLE is ``nb_logical < P``: the band pad rounds
  ``nb`` up to a multiple of ``∏ p_a`` and the ranks holding only pad
  bands do zero work.  That is the whole of it — there is no other idle
  case at any ``P``, ``nk`` or mesh shape.

What the sweep pays for this is a per-k reshard collective, which the
k-partitioned plan does not need (a rank owning a whole k communicates
nothing).  So the trade is COMMUNICATION against SCALABILITY, not idle
ranks: at ``P = nk`` the old plan is already at its own ceiling and the
sweep is slower by the collective (measured 1.45× at nb=512/P=16); past
``P = nk`` the old plan is stuck and the sweep is not (2.05× at
nb=600/P=64).

THE STRUCTURAL FACT THAT COLLAPSES THE THREE
--------------------------------------------
Only the local-potential terms need a real-space excursion:

    kinetic   T_G · ψ                    diagonal in G
    dipole    (k+G) · ψ                  diagonal in G
    V_NL      projector sum              no FFT
    V_H/V_loc F[ V(r) · F⁻¹ψ ]           FFT round trip

All four are ``H[m,n] = Σ_{s,G} conj(ψ_m) · (O ∘ ψ)_n`` and differ ONLY
in ``O ∘ ψ``.  One skeleton, one pluggable operator.

**The m side is never transformed.**  For V_H it is the raw stored
sphere; for the rest the operator is applied on the n side alone.  That
is what removes W1 — and it is why the operator protocol below is
"sphere in, sphere out" rather than anything box-shaped.

THE PLAN
--------
::

    psi_m_X  ←  reshard ONCE, outside the loop   # shared by ALL k and ALL operators
    scan over k:
        Opsi    ←  O ∘ psi_XY[k]                 # nb/P bands per rank
        Opsi_Y  ←  reshard along 'x'             # the ONE per-k collective
        H[k]    ←  einsum('bsg,nsg->bn', conj(psi_m_X[k]), Opsi_Y)

``H`` comes out ``(nk, nb, nb)`` sharded ``P(None, 'x', 'y')`` — the
output is sharded and the CONTRACTION axis is replicated.  The
alternative (shard over G, psum the partials) needs every rank to hold a
full ``(nb, nb)`` to reduce into, which is exactly W2.  Not a
preference; forced.

WHY ψ(G) IS RESIDENT AND THE BOX NEVER IS
-----------------------------------------
The *box* is huge; the G-sphere is not.  ``nk·nb·ns·ngkmax·16`` is 1.2 GB
globally at b600 but **≈19 MB/rank sharded at P=64**, which is what makes
a genuine ``lax.scan`` over k possible at all — the reason
``collectives.sweep_local_k`` is a Python loop is that its ψ load is host
I/O, and that obstacle disappears once ψ is already on device.  Check the
number for your deck before assuming it: at 12×12 with nb=2000 it is ~10×
larger.

WHY THE FFT IS INLINED HERE RATHER THAN CALLED THROUGH wfn_transforms
---------------------------------------------------------------------
``to_rbox``/``from_rbox`` memoise a **device** G-index
(``_cached_gindex_dev``).  Inside ``lax.scan`` the per-k G table is a
dynamic slice, i.e. a TRACER, and that cache would capture it — the
``UnexpectedTracerError`` measured at job 7888526.  The handoff spec's
stated remedy is to hoist the device G-index out and pass it as an
operand, which is what this module does: it reuses
``wfn_transforms._box_kernel`` (pure jax, takes its index as an argument,
so tracers are fine) and builds the sharded transforms from the SAME
``fft_helpers`` factories those helpers use.  Nothing in ``fft_helpers``
or ``wfn_transforms`` is modified or duplicated — the transform is still
the certified shard_map'd FFI FFT, just handed a traced index.

A bare ``jnp.fft`` here would be the CrI3 6×6×1 80 Ry 121 GB OOM: on a
sharded tensor XLA's planner is free to insert an all-gather and emit a
global FFT.  See the module comment above ``wfn_transforms._local_box_fft``.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.wfn_transforms import _box_kernel
from runtime.padding import pad_axis_to


__all__ = [
    "SweepGeometry",
    "Operator",
    "kinetic_operator",
    "local_potential_operator",
    "sweep_matrix_elements",
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
                 "nk", "cell_volume", "ngrid", "p_prod")

    def __init__(self, *, mesh: Mesh, fft_grid: Sequence[int], ngkmax: int,
                 nb: int, ns: int, nk: int, cell_volume: float):
        from runtime.padding import round_up, mesh_divisor

        self.mesh = mesh
        self.fft_grid = tuple(int(s) for s in fft_grid)
        self.ngkmax = int(ngkmax)
        self.ns = int(ns)
        self.nk = int(nk)
        self.cell_volume = float(cell_volume)
        self.ngrid = int(np.prod(self.fft_grid))

        # ``nb`` is the LOGICAL band count the caller has.  The band axis is
        # sharded over the whole mesh, so it must divide ∏ p_a; the pad is
        # computed here and applied by the sweep, and the caller never states
        # it.  Same routine and same zero-band argument as the ψ loader's
        # ``wfn_transforms.gflat_to_rmu`` — ``runtime.padding.pad_axis_to``.
        #
        # This is not a nicety at production shapes: nb=600 on an 8×8 mesh is
        # 64·9.375, and JAX raises IndivisibleError when the sharded array is
        # constructed rather than degrading (job 7888869).
        self.p_prod = mesh_divisor(mesh)
        self.nb_logical = int(nb)
        self.nb = round_up(int(nb), self.p_prod)

    # Sphere-shaped operands, band-sharded over the WHOLE mesh.  Used for
    # the n side during the operator: FFT work is 2nb/P per rank with no
    # px-fold redundancy, which is the point of carrying ('x','y') here
    # rather than transforming inside the column layout.
    @property
    def spec_sphere_xy(self) -> P:
        return P(None, ("x", "y"), None, None)

    @property
    def spec_sphere_x(self) -> P:
        return P(None, "x", None, None)

    @property
    def spec_sphere_y(self) -> P:
        return P(None, "y", None, None)

    @property
    def spec_box_xy(self) -> P:
        return P(None, ("x", "y"), None, None, None, None)

    @property
    def spec_block(self) -> P:
        return P(None, "x", "y")


# ---------------------------------------------------------------------------
# The operator protocol
# ---------------------------------------------------------------------------

class Operator(NamedTuple):
    """``O ∘ ψ`` plus the normalisation that belongs to it.

    ``apply`` is called INSIDE the scan body with traced per-k operands:

      psi_n   (1, nb, ns, ngkmax) c128, sharded ``spec_sphere_xy``
      gvec    (ngkmax, 3) i32  — this k's G table (D10 fixed shape)
      gmask   (ngkmax,)   f64  — 1 on physical G, 0 on pad columns
      bidx    (1, nx, ny, nz) i32 — sphere→box index map for this k
      kvec    (3,) f64

    and must return ``(1, nb, ns, ngkmax)`` in the same layout.  It must
    NOT form anything of shape ``(nb, nb)`` and must not gather over bands.

    ``post`` rides WITH the operator rather than being a sweep argument.
    The factor is part of the operator's own normalisation — the local
    potential's ``sqrt(1/volume)`` closes the same chain as its FFT
    constants — so a caller that picks an operator has already picked its
    normalisation and cannot pair the two up wrongly.  A ``post_scale=``
    argument on the sweep would be one more thing to get right at every
    call site, which is the mistake the SlabIO padding ruling names.
    """
    apply: Callable
    post: float = 1.0


def kinetic_operator(geom: SweepGeometry, bdot) -> Operator:
    """``T ∘ ψ = |k+G|² ψ`` — diagonal in G, no FFT.

    The reference is ``psp.get_DFT_mtxels._compute_kinetic_k_jit``, which
    masks ``T_G`` (not ψ) and contracts.  Masking the diagonal is
    sufficient and is reproduced exactly here, so the only difference
    between the two routes is the reassociation the band sharding forces
    on the G sum — which is why the gate is 1e-12 relative and not
    bit-identity (numerical-tolerance ruling; D10's ``RTOL_D10``).

    This operator is the skeleton's isolation test: no FFT means a
    failure here is the scan, the reshard or the einsum, never the
    transform.
    """
    from psp.get_DFT_mtxels import kinetic_diagonal

    def op(psi_n, gvec, gmask, bidx, kvec):
        T_G = kinetic_diagonal(gvec, kvec, bdot, g_mask=gmask)
        return psi_n * T_G[None, None, None, :].astype(psi_n.dtype)

    return Operator(apply=op, post=1.0)


def local_potential_operator(geom: SweepGeometry, V_r) -> Operator:
    """``V ∘ ψ = F[ V(r) · F⁻¹ψ ]`` — the FFT round trip, for V_H and V_loc.

    Term-for-term the normalisation of
    ``psp.get_DFT_mtxels.compute_local_V_k``, so the two agree to
    round-off and the difference is pure reassociation from the sharding.

    The two transforms are built ONCE, here, outside the scan — they are
    pure functions of shape, so nothing about them is per-k.  Only the
    scatter and the gather touch G, and both take their index as a traced
    operand.

    ``V_r`` is the potential on the FFT grid, replicated.  It is closed
    over rather than scanned because it does not depend on k.
    """
    from common.fft_helpers import make_sharded_fftn_3d, make_sharded_ifftn_3d

    mesh = geom.mesh
    box_spec = geom.spec_box_xy
    ifftn = make_sharded_ifftn_3d(mesh, box_spec, box_spec,
                                  norm='ortho', axes=(-3, -2, -1))
    fftn = make_sharded_fftn_3d(mesh, box_spec, box_spec,
                                norm='ortho', axes=(-3, -2, -1))

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
    V_r_j = jnp.asarray(V_r, dtype=jnp.complex128)

    def op(psi_n, gvec, gmask, bidx, kvec):
        # sphere → box.  ``_box_kernel`` is reused verbatim: it is pure
        # jax, band sharding rides through (the gather is over the G
        # axis, no cross-rank op), and its ngkmax zero-slot makes the
        # sentinel index gather exact zero.
        box = _box_kernel(psi_n, bidx, ngkmax=geom.ngkmax)
        psi_r = ifftn(box) * scale
        phi_r = psi_r * V_r_j
        phi_G = fftn(phi_r) * (deltaV * fft_norm)
        # box → sphere.  Advanced indexing on the three replicated FFT
        # axes only, so the band sharding is untouched.
        gx = gvec[:, 0]
        gy = gvec[:, 1]
        gz = gvec[:, 2]
        out = phi_G[..., gx, gy, gz]
        return out * gmask[None, None, None, :].astype(out.dtype)

    return Operator(apply=op, post=float(_sc.post))


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def sweep_matrix_elements(
    psi_G,
    *,
    geom: SweepGeometry,
    operator: Operator,
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
        The G-sphere ψ, resident on device.  Any band sharding; it is
        constrained internally to both layouts it needs.
    geom, operator
        See above.
    gvecs : (nk, ngkmax, 3) i32
        The loader's own fixed-shape table (D10) — ``PaddedGVectors.gvecs``.
    gmask : (nk, ngkmax) f64
        Its pad mask.  MANDATORY, not optional: pad rows hold ``(0,0,0)``,
        a valid box index that ALIASES physical Γ, so a forgotten mask does
        not crash — it silently folds ψ(G=0) into every pad column, inside
        H₀'s ~500 eV cancellation.
    box_index : (nk, nx, ny, nz) i32
        Sphere→box index map (``WfnLoader.box_index``).  Only consumed by
        operators that transform; the kinetic and dipole operators ignore it.
    kvecs : (nk, 3) f64
    use_scan : bool
        ``True`` (default) runs ``lax.scan`` — one lowering for the whole
        sweep.  ``False`` runs the identical body in a Python loop, which
        is the reference the scan is gated against: same arithmetic, same
        shardings, different control flow, so a disagreement isolates the
        scan itself.

    Returns
    -------
    (nk, nb, nb) c128 sharded ``P(None, 'x', 'y')``.  No rank ever holds a
    full ``(nb, nb)`` tile.

    MASKING IS IMPLICIT AND UNCONDITIONAL, AND THAT IS THE POINT
    ------------------------------------------------------------
    Both operands are masked, always.  There is no flag.

    It is TRUE that the mask is sometimes unnecessary: measured (job
    7888534), with pad G rows at the box corner ``(nx//2, ny//2, nz//2)``,
    which cannot intersect the sphere, the unmasked result is
    bit-identical to the masked one (``0.000e+00``).  But the rule is
    narrower than it looks — a corner sentinel removes the need for a
    mask **iff both operands' pad entries come from stored sphere
    coefficients**.  ``phi_G`` at the sentinel is NOT zero (multiplying by
    V(r) spreads support over the box); what kills the term is the m side
    being exact zeros.  If either operand is gathered from the BOX,
    ``(0,0,0)`` = Γ is a real coefficient and the mask is mandatory again.

    So the condition under which the flag could be set safely depends on
    where BOTH operands came from — which is exactly the kind of
    padding question the 2026-08-04 SlabIO ruling says a caller must not
    have to reason about.  The cost of always masking is one multiply per
    operand per k against a ``(ngkmax,)`` vector.  That is not worth a
    decision, so there is no decision to make.
    """
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

    # THE BAND PAD, applied here so no caller states it (SlabIO ruling,
    # decisions.md 2026-08-04: padding is the infrastructure's business).
    # Pad bands are ψ = 0, so the extra rows AND columns of ⟨m|O|n⟩ are
    # exactly zero -- they are not "close to zero", they are the product of
    # an exact zero, so no downstream mask is needed and no tolerance is
    # spent on them.
    psi, _ = pad_axis_to(psi, geom.p_prod, axis=1)

    gvecs_j = jnp.asarray(gvecs, dtype=jnp.int32)
    gmask_j = jnp.asarray(gmask, dtype=jnp.float64)
    kvecs_j = jnp.asarray(kvecs, dtype=jnp.float64)
    bidx_j = jnp.asarray(box_index, dtype=jnp.int32)

    # THE HOISTED RESHARD.  ψ_m is never transformed, so the column
    # layout is built ONCE and reused across every k *and* every
    # operator: nk+1 all-gathers for a sweep, not 2·nk, and not 3× that
    # for three sweeps.  ≈151 MB/rank at b600/P=64.
    psi_m_X = jax.lax.with_sharding_constraint(
        psi, NamedSharding(mesh, geom.spec_sphere_x))
    psi_n_XY = jax.lax.with_sharding_constraint(
        psi, NamedSharding(mesh, geom.spec_sphere_xy))

    block_sharding = NamedSharding(mesh, geom.spec_block)

    def one_k(ik_psi_n, ik_psi_m, gvec, gm, bidx, kvec):
        """The body.  Identical under scan and under the Python loop."""
        Opsi = operator.apply(ik_psi_n, gvec, gm, bidx, kvec)

        # THE ONE PER-K COLLECTIVE, expressed as a re-shard rather than a
        # hand-rolled all-gather: XLA inserts it, and it is along 'x'
        # only, not a global collective.  Payload is the sphere-space
        # operator output (9.4 MB at b600/P=64), never the r-space box.
        Opsi_Y = jax.lax.with_sharding_constraint(
            Opsi, NamedSharding(mesh, geom.spec_sphere_y))

        m_side = ik_psi_m * gm[None, None, None, :].astype(ik_psi_m.dtype)
        m_side_X = jax.lax.with_sharding_constraint(
            m_side, NamedSharding(mesh, geom.spec_sphere_x))

        blk = jnp.einsum('kbsg,knsg->kbn',
                         jnp.conj(m_side_X), Opsi_Y, optimize=True)
        blk = blk * operator.post
        return jax.lax.with_sharding_constraint(blk, block_sharding)

    if not use_scan:
        # Reference control.  Same arithmetic and shardings, Python
        # control flow — so a scan-vs-loop disagreement is the scan.
        out = [one_k(psi_n_XY[ik:ik + 1], psi_m_X[ik:ik + 1],
                     gvecs_j[ik], gmask_j[ik], bidx_j[ik:ik + 1], kvecs_j[ik])
               for ik in range(nk)]
        return jax.lax.with_sharding_constraint(
            jnp.concatenate(out, axis=0), block_sharding)

    def body(carry, xs):
        psi_n_k, psi_m_k, gvec, gm, bidx, kvec = xs
        # scan strips the leading axis; put the singleton k axis back so
        # every shape inside the body matches the non-scan path exactly.
        blk = one_k(psi_n_k[None], psi_m_k[None], gvec, gm,
                    bidx[None], kvec)
        return carry, blk[0]

    _, H = jax.lax.scan(
        body, None,
        (psi_n_XY, psi_m_X, gvecs_j, gmask_j, bidx_j, kvecs_j))
    return jax.lax.with_sharding_constraint(H, block_sharding)
