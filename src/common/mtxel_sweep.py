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

    scan over k:
        Opsi    ←  O ∘ psi_XY[k]                 # nb/P bands per rank
        Opsi_Y  ←  reshard onto 'y'              # the per-k ket collective
        psi_m_X ←  reshard psi_XY[k] onto 'x'    # the per-k bra collective
        H[k]    ←  einsum('bsg,nsg->bn', conj(psi_m_X), Opsi_Y)

BOTH reshards are per-k.  The bra one used to be hoisted out of the scan
— one ``(nk, nb, ns, ngkmax)`` all-gather instead of ``nk`` slice-sized
ones, on the argument that fewer collectives is fewer collectives.  That
was measured and it is the wrong way round: at b600/P=64 (nk=16, nb=640,
ns=2, ngkmax=11008) hoisting costs **2.220 s against 1.985 s**, 1.12×,
and holds a 430 MiB/rank ``P(None,'x',…)`` copy of ψ live across the
whole scan on top of the 54 MiB/rank ``('x','y')`` one (jobs 7889241,
7889250, arms ``vh`` / ``vh_nohoist``).  Per k the same bytes move, but
in ``nk`` pieces that overlap with the FFT round trip instead of one
serialised block, and the mask multiply that precedes the reshard runs
on the ``nb/(p_x·p_y)`` shard rather than the ``nb/p_x`` one.

``H`` comes out ``(nk, nb, nb)`` sharded ``P(None, 'x', 'y')`` — the
output is sharded and the CONTRACTION axis is replicated.  The
alternative (shard over G, psum the partials) needs every rank to hold a
full ``(nb, nb)`` to reduce into, which is exactly W2.  Not a
preference; forced.

A Cartesian operator (dipole) appends a length-3 REPLICATED component
axis and comes back ``(nk, 3, nb, nb)`` at ``P(None, None, 'x', 'y')``.
Three components ride ONE sweep rather than three, so the bra reshard,
the bra mask and the scan are paid once instead of three times; only the
ket payload and the einsum are 3×.  The component axis is MINOR on the
ket by measurement, not by convenience: carrying it LEADING instead —
``'kbsg,kcnsg->kcbn'`` on a ``(1, 3, nb, ns, ngkmax)`` operator output,
which also removes the ``moveaxis`` — is 3.8 % SLOWER at b600/P=64
(1.327 s against 1.279 s, job 7889241, arms ``dip_first`` /
``dip_last``), and the ``moveaxis`` itself costs nothing because XLA
folds it into the copy that feeds the dot (1.288 s for a variant that
never forms it, arm ``dip_last_nomv``).

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
a genuine ``lax.scan`` over k possible at all — the reason
``collectives.sweep_local_k`` is a Python loop is that its ψ load is host
I/O, and that obstacle disappears once ψ is already on device.  Check the
number for your deck before assuming it: at 12×12 with nb=2000 it is ~10×
larger.

WHY THE FFT IS INLINED HERE, AND WHY THAT BUYS THE OUTER JIT
------------------------------------------------------------
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
``fft_helpers.make_sharded_{i,}fftn_3d``, i.e. ``shard_map`` around
XLA's own local ``jnp.fft``, just handed a traced index.  That is NOT
the flat-k FFI handler and should not be confused with it: the FFI
handler reads the FFT axes as the LEADING flattened one, this box has
them minor, and routing this shape through it measured 2.0× slower
(0.206 s against 0.104 s for the same volume, job 7889250 arm
``fftbench``).  The FFI's win in the Σ τ kernel is avoiding a μ²-tile
transpose; there is no such transpose here.

**Consequence: the whole sweep IS wrapped in one ``jax.jit``.** The
earlier prohibition ("do NOT wrap the body in an outer jit", job 7888526)
applied to a version that called ``to_rbox``/``from_rbox``; this one does
not touch them, so nothing memoises a device G-index and there is no
tracer to escape. The jit is cached in ``_KERNEL_CACHE`` — the same
``_cached_jit`` the transforms use — keyed on the shapes, the sharding
and the operator identity, so it is built once and not once per call.

The outer jit means the input constraint, the scan and the final
constraint lower as ONE program, so XLA places the per-k collectives
relative to the scan body's compute rather than seeing them as separate
eagerly-dispatched ops. Everything is one k at a time inside it; the jit
changes what XLA is allowed to see, not the algorithm.

WHAT THE SWEEP'S WALL IS ACTUALLY MADE OF
-----------------------------------------
Measured at b600/P=64 (job 7889241), worst rank, median of 3, by
substituting the operator and by running the skeleton's pieces alone:

    whole sweep, local-potential operator          2.220 s   100 %
    same skeleton with an IDENTITY operator        0.862 s    39 %
    the per-k ket reshard alone                    0.176 s     8 %

So the operator — the sphere→box gather, the FFT round trip, the V(r)
multiply and the box→sphere gather — is 61 % of the wall and the
collective is 8 %.  A statement that the per-k reshard is the sweep's
cost at ``P ≤ nk`` is wrong at this shape; the box traffic is.  Two
consequences that were tested and came out negative are recorded so they
are not re-proposed: routing the transforms through the flat-k FFT FFI
(``ffi.fft``) is **2.0× slower** here, 0.206 s against 0.104 s for the
same volume, because that handler's win is avoiding a μ²-tile transpose
and this path's box is already FFT-minor; and moving the per-k gather
onto the other mesh axis is within noise, 0.180 s against 0.176 s
(job 7889241, arms ``fftbench``, ``reshard_only`` / ``reshard_only_x``).

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

from common.wfn_transforms import _box_kernel, _cached_jit, _sharding_key
from runtime.padding import pad_axis_to


__all__ = [
    "SweepGeometry",
    "band_sphere_spec",
    "Operator",
    "kinetic_operator",
    "local_potential_operator",
    "vnl_operator",
    "dipole_operator",
    "sum_operators",
    "sweep_matrix_elements",
    "blocks_to_host",
]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def band_sphere_spec() -> P:
    """THE ψ layout: ``(n_k, nb, nspinor, ngkmax)``, bands over the mesh.

    One definition, three consumers — ``file_io.wfn_loader`` defaults to
    it, this sweep contracts in it, and ``gw.qsgw_density`` builds ρ from
    it.  A second literal of the same PartitionSpec would not raise if it
    drifted; it would silently insert a reshard between them, which is the
    same class of latent bug ``runtime.padding.spec_divisor`` removed on
    the band divisor.
    """
    return P(None, ("x", "y"), None, None)



class SweepGeometry:
    """The fixed shapes every operator and the scan agree on.

    Built once per sweep.  Everything here is static at trace time, which
    is what lets the scan body lower ONCE for the whole k range (D10).
    """

    __slots__ = ("mesh", "fft_grid", "ngkmax", "nb", "nb_logical", "ns",
                 "nk", "cell_volume", "ngrid", "p_prod")

    def __init__(self, *, mesh: Mesh, fft_grid: Sequence[int], ngkmax: int,
                 nb: int, ns: int, nk: int, cell_volume: float):
        from runtime.padding import round_up, spec_divisor

        self.mesh = mesh
        self.fft_grid = tuple(int(s) for s in fft_grid)
        self.ngkmax = int(ngkmax)
        self.ns = int(ns)
        self.nk = int(nk)
        self.cell_volume = float(cell_volume)
        self.ngrid = int(np.prod(self.fft_grid))

        # ``nb`` is the LOGICAL band count.  The divisor is derived FROM THE
        # SPEC by the shared ``spec_divisor``, not assumed to be ∏ p_a — the
        # same call ``file_io.wfn_loader._default_sharding`` makes for its own
        # ``p_band``.  Both default to ``P(None, ('x','y'), None, None)``, so
        # both get px·py and ψ FROM THE LOADER IS ALREADY BAND-PADDED FOR
        # THIS SWEEP: the pad below is a no-op on the production path and
        # exists for callers that build ψ themselves.  That agreement holds
        # only while both derive it here.
        #
        # Not a nicety at production shapes: nb=600 on an 8×8 mesh is
        # 64·9.375, and JAX raises IndivisibleError when the sharded array is
        # CONSTRUCTED rather than degrading (job 7888869).
        self.p_prod = spec_divisor(mesh, self.spec_sphere_xy, 1)
        self.nb_logical = int(nb)
        self.nb = round_up(int(nb), self.p_prod)

    # Sphere-shaped operands, band-sharded over the WHOLE mesh.  Used for
    # the n side during the operator: FFT work is 2nb/P per rank with no
    # px-fold redundancy, which is the point of carrying ('x','y') here
    # rather than transforming inside the column layout.
    @property
    def spec_sphere_xy(self) -> P:
        return band_sphere_spec()

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

    # --- the optional COMPONENT axis (dipole: 3 Cartesian directions) ---
    # It is length 3 and REPLICATED: it cannot usefully divide a mesh
    # axis, and carrying all three through one sweep is what keeps the
    # hoisted m-side reshard at ONE all-gather instead of three (the
    # handoff's §3b, which is stated for "all four operators" and holds
    # verbatim for the three Cartesian components of one operator).
    # Sphere operands carry it LAST so the band axis stays at index 1 and
    # every spec above extends by appending a single ``None``.
    @staticmethod
    def with_comp(spec: P, ncomp: int) -> P:
        return spec if not ncomp else P(*spec, None)

    def spec_block_for(self, ncomp: int) -> P:
        return P(None, "x", "y") if not ncomp else P(None, None, "x", "y")


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

    and must return ``(1, nb, ns, ngkmax)`` in the same layout — or, when
    ``ncomp > 0``, ``(1, nb, ns, ngkmax, ncomp)``: the band axis stays at
    index 1 and the component axis is appended.  It must NOT form
    anything of shape ``(nb, nb)`` and must not gather over bands.

    ``ncomp`` is 0 for a scalar operator (T, V_loc, V_H, V_NL) and 3 for
    a Cartesian one (dipole).  It is not a shape the sweep can infer:
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
    apply: Callable
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
    return tuple(op.key) if op.key else ('id', id(op.apply))


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

    def op(psi_n, gvec, gmask, bidx, kvec, bdot_j):
        T_G = kinetic_diagonal(gvec, kvec, bdot_j, g_mask=gmask)
        return psi_n * T_G[None, None, None, :].astype(psi_n.dtype)

    return Operator(apply=op, post=1.0,
                    consts=(jnp.asarray(np.asarray(bdot, dtype=np.float64)),),
                    key=('kinetic', geom.ngkmax, geom.ns))


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

    def op(psi_n, gvec, gmask, bidx, kvec, V_r_j):
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

    # V(r) rides in as an OPERAND, not as a closed-over constant.  It is the
    # only input of this operator that moves between calls, and baking it
    # ties one compiled sweep to one V_H — a full lowering per density-SC
    # step (:func:`_operator_key`).  The scalars above stay literals: they
    # are functions of the geometry and do not move.
    return Operator(apply=op, post=float(_sc.post), consts=(V_r_j,),
                    key=('local_potential', geom.fft_grid, geom.ngkmax,
                         geom.ns, scale, deltaV, fft_norm,
                         tuple(int(d) for d in V_r_j.shape)))


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
    from 1 to 0 without moving any data; the sweep re-adds it before the
    einsum.  No sharding constraint is issued here — XLA carries the band
    sharding through a squeeze, and the one per-k collective is the
    reshard in the sweep body, which this must not duplicate.
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
    """``V_NL ∘ ψ = Z E Z† ψ`` — a projector sum: no FFT, no band gather.

    Why it fits the ``Operator`` protocol at all: the G sum inside the
    projector overlap ``P[R,s,n] = Σ_G conj(Z) ψ`` runs over the
    REPLICATED G axis, so it is rank-local; the free index is the band,
    which stays sharded.  Nothing here needs a collective and nothing
    forms an ``(nb, nb)``.

    ``vnl_ops.apply_vnl`` is the kernel, unmodified — the same
    ``Z E Z†`` the local plan's ``vnl_ops.vnl_matrix`` contracts, taken
    one step earlier.  The two therefore differ only in where the bra is
    applied, which reassociates the G sum: gate at 1e-12 relative, not
    bit-identity.

    Cost per k: ``Z`` is ``(total_R, ngkmax)`` REPLICATED (32 MB at
    total_R=100, ngkmax=20000) — it does not scale with nb, so it does
    not enter the per-rank wall this sweep exists to remove.
    """
    from psp import vnl_ops

    def op(psi_n, gvec, gmask, bidx, kvec):
        psi = _ket(psi_n, gmask)
        kdata = vnl_ops.build_vnl_kdata_traced(kvec, gvec, vnl_setup)
        ns_e = int(kdata.E_super.shape[0])
        out = vnl_ops.apply_vnl(psi[:, :ns_e], kdata.Z, kdata.E_super)
        return _pad_spinor(out, int(psi.shape[1]))[None]

    return Operator(apply=op, post=1.0,
                    key=('vnl', geom.ngkmax, geom.ns, id(vnl_setup)))


def dipole_operator(geom: SweepGeometry, *, bvec, blat,
                    vnl_setup=None) -> Operator:
    """``v ∘ ψ = 2(k+G)_cart ψ − (∂V_NL/∂K_cart) ψ`` — THREE components.

    The velocity matrix ``psp.get_dipole_mtxels`` writes is
    ``p + i[r, V_NL]``, assembled there as ``p_cart - v_NL_cart`` (the
    sign flip at its ``vNL_cart = -vNL_cart`` line, which exists so the
    stored convention matches BGW's).  Both halves already have
    apply-to-ket kernels — ``dft_operators.apply_kinetic_velocity_to_ket``
    and ``vnl_ops.apply_vnl_velocity_to_ket``, each ``(3, nb, ns, nG)`` —
    so this operator is their difference and no velocity physics is
    written twice.

    ``vnl_setup=None`` reproduces ``--skip-vnl`` (p̂ only).

    The component axis is moved to the END, where the sweep's specs
    expect it; that is a transpose of the operator output, not of ψ.
    """
    from psp.dft_operators import apply_kinetic_velocity_to_ket
    from psp import vnl_ops

    B = jnp.asarray(np.asarray(bvec, dtype=np.float64) * float(blat),
                    dtype=jnp.float64)

    def op(psi_n, gvec, gmask, bidx, kvec, B):
        psi = _ket(psi_n, gmask)
        v = apply_kinetic_velocity_to_ket(psi, gvec, kvec, B)
        if vnl_setup is not None:
            kdata = vnl_ops.build_vnl_kdata_traced(kvec, gvec, vnl_setup,
                                                   compute_dZ=True)
            ns_e = int(kdata.E_super.shape[0])
            v_nl = vnl_ops.apply_vnl_velocity_to_ket(
                psi[:, :ns_e], kdata.Z, kdata.dZ, kdata.E_super)
            v = v - _pad_spinor(v_nl, int(psi.shape[1]))
        return jnp.moveaxis(v, 0, -1)[None]

    return Operator(apply=op, post=1.0, ncomp=3, consts=(B,),
                    key=('dipole', geom.ngkmax, geom.ns, float(blat),
                         None if vnl_setup is None else id(vnl_setup)))


def sum_operators(*ops: Operator) -> Operator:
    """``(O₁ + O₂ + …) ∘ ψ`` — one sweep, one per-k collective.

    ``⟨m|T+V_loc+V_NL|n⟩`` is ONE matrix element, so it is one sweep:
    summing on the KET costs one extra ``(nb/P, ns, ngkmax)`` add per
    term and leaves the reshard and the einsum — the expensive halves —
    paid once, where three sweeps would pay all three three times.

    Each term's ``post`` is folded into its own contribution before the
    sum, which is what lets operators with different normalisations
    (``local_potential_operator`` carries ``sqrt(1/Ω)``, the others 1)
    share one einsum.  Algebraically identical, since the einsum is
    linear; numerically it moves one scalar multiply from after the G
    sum to before it, i.e. ~1 ulp, inside the 1e-12 gate.
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

    def op(psi_n, gvec, gmask, bidx, kvec, *cs):
        acc = None
        for (a, b), o in zip(spans, ops):
            term = o.apply(psi_n, gvec, gmask, bidx, kvec, *cs[a:b])
            if o.post != 1.0:
                term = term * o.post
            acc = term if acc is None else acc + term
        return acc

    return Operator(apply=op, post=1.0, ncomp=ncomp, consts=all_consts,
                    key=('sum',) + tuple(_operator_key(o) for o in ops))


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
    (nk, nb, nb) c128 sharded ``P(None, 'x', 'y')`` — or, for an operator
    with ``ncomp > 0``, ``(nk, ncomp, nb, nb)`` sharded
    ``P(None, None, 'x', 'y')``.  No rank ever holds a full ``(nb, nb)``
    tile.  Band extents are the mesh-PADDED ``geom.nb``;
    :func:`blocks_to_host` is the boundary that trims back to logical.

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

    Now priced, so the question stays closed: dropping the bra-side mask
    entirely is 2.208 s against 2.220 s at b600/P=64, i.e. inside the
    noise (job 7889241, arms ``vh`` / ``vh_nomask``).  The lowered HLO
    says why — the mask is fused into the ``kLoop`` copy that already has
    to lay the bra out as ``c128[nb/p_x, ns·ngkmax]`` for the dot, so it
    costs a multiply on bytes that were being touched anyway.
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

    ncomp = int(operator.ncomp)
    block_sharding = NamedSharding(mesh, geom.spec_block_for(ncomp))
    # 'kbsg,knsg->kbn' for a scalar operator; the Cartesian one carries a
    # replicated component axis through the SAME contraction.
    contraction = 'kbsg,knsg->kbn' if not ncomp else 'kbsg,knsgc->kcbn'

    # The operator's runtime operands.  They are jit ARGUMENTS, so one
    # executable serves every value of them; anything the operator closes
    # over instead is a jaxpr constant and forces a lowering per value.
    op_consts = tuple(jnp.asarray(c) for c in operator.consts)

    def _run(psi, gvecs_, gmask_, bidx_, kvecs_, *consts_):
        # ONE resident layout.  ⟨m|O|n⟩ contracts every m with every n, so
        # the bra must reach 'x' and the ket 'y' — that split is the POINT,
        # not a cost to be optimised away (owner, 2026-08-04) — but BOTH
        # are reached per k, from this one ``('x','y')`` copy.  Hoisting the
        # bra's ``P(None,'x',…)`` copy out of the scan was measured slower
        # and 430 MiB/rank heavier at b600/P=64; see the module docstring.
        psi_n_XY = jax.lax.with_sharding_constraint(
            psi, NamedSharding(mesh, geom.spec_sphere_xy))

        def one_k(ik_psi_n, ik_psi_m, gvec, gm, bidx, kvec):
            """The body.  Identical under scan and under the Python loop."""
            Opsi = operator.apply(ik_psi_n, gvec, gm, bidx, kvec, *consts_)

            # THE PER-K KET COLLECTIVE, expressed as a re-shard rather than
            # a hand-rolled all-gather: XLA inserts it.  Payload is the
            # sphere-space operator output, never the r-space box —
            # 3.4 MiB/rank in, 26.9 MiB/rank out at b600/P=64.
            #
            # It lowers to TWO instructions, not one (HLO read at
            # b600/P=64, job 7889241): a ``collective-permute`` of the whole
            # shard that transposes the 8×8 rank grid, then an
            # ``all-gather`` over ``replica_groups=[8,8]<=[8,8]T(1,0)``,
            # i.e. the STRIDED 'x' groups.  The permute is what makes
            # ``('x','y') → 'y'`` expressible as a contiguous gather at all,
            # and it is a global exchange, not an 'x'-local one.  Both are
            # inside the scan body; the whole per-k collective measures
            # 0.176 s of the sweep's 2.220 s, and issuing it over the other
            # mesh axis instead is 0.180 s, i.e. within noise.
            Opsi_Y = jax.lax.with_sharding_constraint(
                Opsi, NamedSharding(
                    mesh, geom.with_comp(geom.spec_sphere_y, ncomp)))

            # THE PER-K BRA COLLECTIVE.  The mask runs FIRST, on the
            # ``('x','y')`` shard, so the multiply touches nb/(p_x·p_y)
            # bands and only the masked result is gathered onto 'x'.
            m_side = ik_psi_m * gm[None, None, None, :].astype(ik_psi_m.dtype)
            m_side_X = jax.lax.with_sharding_constraint(
                m_side, NamedSharding(mesh, geom.spec_sphere_x))

            blk = jnp.einsum(contraction,
                             jnp.conj(m_side_X), Opsi_Y, optimize=True)
            blk = blk * operator.post
            return jax.lax.with_sharding_constraint(blk, block_sharding)

        if not use_scan:
            # Reference control.  Same arithmetic and shardings, Python
            # control flow — so a scan-vs-loop disagreement is the scan.
            # It is ALSO 1.33× faster at b600/P=64 (1.674 s against
            # 2.220 s, job 7889250 arm ``vh_unrollfull``): the scan's while
            # body serialises the per-k collective against the compute that
            # XLA can overlap once the trip is unrolled.  Not the default,
            # because unrolling makes the module — and the compile — linear
            # in nk, which at a 12×12 deck's nk=144 is the cost
            # ``gw.v_q_g_flat`` moved to a scan to escape.  Left as the
            # caller's flag, with the number attached.
            out = [one_k(psi_n_XY[ik:ik + 1], psi_n_XY[ik:ik + 1],
                         gvecs_[ik], gmask_[ik], bidx_[ik:ik + 1],
                         kvecs_[ik])
                   for ik in range(nk)]
            return jax.lax.with_sharding_constraint(
                jnp.concatenate(out, axis=0), block_sharding)

        def body(carry, xs):
            psi_k, gvec, gm, bidx, kvec = xs
            # scan strips the leading axis; put the singleton k axis back
            # so every shape inside the body matches the non-scan path.
            # ONE ψ operand serves both sides: the bra and the ket start
            # from the same ``('x','y')`` slice and are resharded inside.
            blk = one_k(psi_k[None], psi_k[None], gvec, gm, bidx[None], kvec)
            return carry, blk[0]

        _, H = jax.lax.scan(
            body, None, (psi_n_XY, gvecs_, gmask_, bidx_, kvecs_))
        return jax.lax.with_sharding_constraint(H, block_sharding)

    fn = _cached_jit(
        'sweep_matrix_elements',
        (psi.shape, geom.ngkmax, geom.ns, nk, bool(use_scan),
         _operator_key(operator), float(operator.post), ncomp,
         geom.fft_grid, _sharding_key(psi),
         tuple((tuple(int(d) for d in c.shape), str(c.dtype))
               for c in op_consts)),
        lambda: jax.jit(_run))
    return fn(psi, gvecs_j, gmask_j, bidx_j, kvecs_j, *op_consts)


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------

#: Payload cap for one collective in :func:`blocks_to_host`.  256 MiB is
#: the same order as ``common.collectives``' owner-gather chunk and is
#: what keeps a peer's transient below the full table it is not keeping.
_HOST_CHUNK_BYTES = 1 << 28


def blocks_to_host(H, *, nb: int, owner_only: bool = False):
    """Sharded ``H[k, …, m_X, n_Y]`` → a host ``numpy`` array.

    THE BOUNDARY IS EXPLICIT AND IT IS NOT FREE.  ``sweep_matrix_elements``
    returns the block SHARDED, which is the point: no rank holds a full
    ``(nb, nb)``.  Every consumer that keeps its result in that layout
    (``gw.sc_iteration.rebuild_hartree_dft_basis``) must NOT call this.
    It exists for the sinks that cannot take a sharded operand — the
    serial ``h5py`` writes in ``gw.kin_ion_io`` and
    ``psp.get_dipole_mtxels``, and the replicated global operand
    ``gw.sigma_dispatch``'s gspace route builds — and it re-materialises
    the replicated ``(nk, nb, nb)`` on the ranks that keep it.  What the
    sweep removes upstream of here is the per-k full-band FFT box and the
    ``P ≤ nk`` ceiling; the table itself is the output and something has
    to hold it.

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
    """
    from common.collectives import gather_to_host, process_rank

    nk = int(H.shape[0])
    tail = tuple(int(s) for s in H.shape[1:])
    per_k = int(np.prod(tail)) * 16
    step = max(1, min(nk, _HOST_CHUNK_BYTES // max(per_k, 1)))
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
