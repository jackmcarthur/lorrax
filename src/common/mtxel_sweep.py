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
* The one IDLE-RANK case is ``nb_logical < P``: the band pad rounds ``nb``
  up to a multiple of ``∏ p_a`` and the ranks holding only pad bands do
  zero work.
* There is ALSO one REPLICATED term, and an earlier version of this
  paragraph denied it ("there is no other idle case at any P, nk or mesh
  shape").  That was asserted, not measured, and it is false: the V_NL and
  dipole operators call ``vnl_ops.build_vnl_kdata_traced`` inside the scan
  body, and Z is ``(total_R, ngkmax)`` REPLICATED — every rank builds the
  same Z for every k, where the k-partitioned plan built it for ``nk/P``
  k.  Now measured (``tests/multi_device/mtxel_vnl_scaling_probe.py``, job
  7889392, MoS2 4×4 deck_b300, nk=16, nb=128, ns=2, ngkmax=1964,
  total_R=62 so Z is 1.9 MiB): the build costs 0.017 s at P=1, 0.010 s at
  P=4 and 0.020 s at P=16 — FLAT, no trend — while the whole V_NL operator
  over the kinetic baseline is 0.064 / 0.031 / 0.039 s.  So the build is
  27 % of V_NL's cost at P=1 and 51 % at P=16: the share rises with P
  exactly as replicated work must.  Z+dZ for the dipole is 0.046 / 0.055 /
  0.061 s, likewise flat.
  NOT FIXED, deliberately.  The candidate fixes are to shard the G axis of
  the build and all-gather Z per k, or to hoist a per-k Z table out of the
  scan; the first pays a collective per k to save ~1 ms of arithmetic (the
  per-k reshard this sweep already issues measures 0.176 s at b600/P=64),
  and the second does not reduce the build count at all — it is already
  one per k.  Against a V_H sweep of 5.740 s at P=1 the whole term is
  ≤ 20 ms.  The claim is corrected rather than the code.

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

WHERE THE OPERATOR'S WALL GOES, AND THE FUSION ARGUMENT THAT IS WRONG
---------------------------------------------------------------------
Measured on the MoS2 4×4 density-SC shape — nk=16, nb=128, ns=2, grid
24×24×80, ngkmax=1968, i.e. the sweep
``gw.sc_iteration.rebuild_hartree_dft_basis`` issues and the one behind the
5.72 s/iteration ``mtxel.sweep`` row of job 7889362 — by running the SAME
skeleton with operators that add one stage at a time
(``tests/multi_device/mtxel_fusion_probe.py``; jobs 7889383, 7889385,
7889386; median of 3, worst rank)::

                                     P=1      P=4      P=16
      identity (scan+reshards+dot)   0.047 s  0.071 s  0.037 s
      + sphere→box, box→sphere       0.046 s  0.060 s  0.033 s
      + V(r) multiply                0.044 s  0.059 s  0.032 s
      + ifftn                        2.963 s  0.805 s  0.204 s
      + fftn   (= production)        5.740 s  1.566 s  0.359 s
      bare ifftn+fftn on the box     2.797 s  0.836 s  0.188 s

FIRST: the operator's wall IS the two transforms — 5.70 s of 5.74 s at P=1,
99.2 %.  The sphere→box gather, the V(r) multiply, the box→sphere gather,
the scan, BOTH per-k reshards and the einsum together are 0.047 s.  Nothing
outside the transforms is worth optimising at this shape, and the b600/P=64
attribution above (operator 61 %, collective 8 %) is a statement about that
shape, not this one.

SECOND: THE SHARD_MAP BOUNDARIES ARE NOT A COST.  The argument that they are
— ``ifftn`` and ``fftn`` are each a ``shard_map``, a shard_map is a hard
fusion boundary, therefore the 180 MiB box is re-materialised at each of the
four crossings — was tested by putting sphere→box, both transforms, the
multiply and box→sphere inside ONE ``shard_map`` (two crossings, not eight).
The single-region operator is SLOWER: 5.909 s against 5.740 s at P=1
(−2.9 %), −1.5 % at P=4, −2.6 % at P=16, and bit-identical (max rel delta
0.000e+00).  It cannot help.  XLA never fuses an elementwise op into an
``fft`` — it is a library call — so the box materialises between the stages
either way, and collapsing the regions removes only the
``SPMDFullToShardShape`` pairs, which the SPMD partitioner has already
elided (0 of them in the optimized HLO at every P).

THERE IS NO 30× OVERHEAD; THERE IS A 30× SHORTFALL IN CORES.  The 29.2
GFLOP of transform arithmetic is "≈0.2 s" only at ~150 GFLOP/s, which is
what 16 ranks deliver and one does not: the bare transform pair measures
10.5 GFLOP/s at P=1, 35.0 at P=4, 155.8 at P=16.  Job 7889362 ran ``-N 1
-n 1``.  The sweep itself is 5.740 s at P=1 and 0.359 s at P=16 — 16.0× on
16 ranks, linear.  A wall that scales linearly in P is work.

WHAT IS LEFT is a P-INDEPENDENT 1.9–2.1× over a bare transform pair on the
same box, and it is on the PRODUCER side.  The same two transforms fed by a
box built in-jit from ``_box_kernel`` measure 5.956 s against 2.803 s
parameter-fed, and dropping the trailing box→sphere gather does not move it
(5.869 s; job 7889385).  The optimized HLO says why: ``_box_kernel``'s
gather emits the box r-major/band-minor
(``c128[128,2,1,24,24,80]{1,0,5,4,3,2}``) and XLA:CPU's ``fft`` demands
band-major/r-minor, so the fft operand is a relayout —
``fft(%copy_bitcast_fusion)`` here against ``fft(%b.1)`` in the bare pair.
Expressing the box→sphere gather on a FLATTENED r axis, so the consumer
stops pulling the layout the other way, does NOT remove it: 5.567 s against
5.535 s at P=1 and 0.355 s against 0.368 s at P=16, values bit-identical
(job 7889386, arm ``prod_flatg``).  Closing it means changing the layout
``wfn_transforms._box_kernel`` produces, which is not this module.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common import timing
from common.wfn_layout import PSI_MUN_SPEC, PSI_NMU_SPEC, band_sphere_spec
from common.wfn_transforms import _box_kernel, _cached_jit, _sharding_key
from runtime.padding import pad_axis


__all__ = [
    "SweepGeometry",
    "Operator",
    "kinetic_operator",
    "local_potential_operator",
    "four_current_potential_operator",
    "vnl_operator",
    "dipole_operator",
    "uniform_gauge_operator",
    "sum_operators",
    "sweep_matrix_elements",
    "sweep_uniform_current_matrix_elements",
    "sweep_uniform_gauge_matrix_elements",
    "finite_transfer_current_to_centroids",
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
        box = _box_kernel(psi_n, bidx, ngkmax=geom.ngkmax)
        psi_r = ifftn(box) * scale
        if vector:
            phi_r = jnp.zeros_like(psi_r)
            for i, (perm, phase) in enumerate(alpha_vertices):
                phi_r = phi_r + V_r_j[i] * gamma_apply(
                    psi_r, perm, phase, axis=2)
        else:
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
    reshard and bra contraction.  The forward FFT is batched over the two
    outputs, preserving the decomposition required by ``sigma_mnk.h5``.

    ``charge_nspinor`` applies only to component 0.  This is load-bearing for
    the Pauli-reference model: the full four-spinor carries the current, while
    the scalar charge uses its leading source-WFN components.  Zeroing the
    scalar operator ket outside that block is algebraically identical to
    slicing both bra and ket because those output spinor rows are exact zero.
    """
    from common.fft_helpers import make_sharded_fftn_3d, make_sharded_ifftn_3d
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

    mesh = geom.mesh
    ifftn = make_sharded_ifftn_3d(
        mesh, geom.spec_box_xy, geom.spec_box_xy,
        norm="ortho", axes=(-3, -2, -1))
    comp_box_spec = P(None, None, ("x", "y"), None, None, None, None)
    fftn = make_sharded_fftn_3d(
        mesh, comp_box_spec, comp_box_spec,
        norm="ortho", axes=(-3, -2, -1))
    scalars = local_potential_scalars(geom.cell_volume, geom.ngrid)
    scale = float(scalars.scale)
    fft_scale = float(scalars.deltaV * scalars.fft_norm)
    alpha_vertices = tuple(gamma_perm_phase(i) for i in (1, 2, 3))
    charge_mask = jnp.asarray(
        np.arange(4) < charge_ns, dtype=jnp.complex128).reshape(
            1, 1, 4, 1, 1, 1)

    def op(psi_n, gvec, gmask, bidx, kvec, V0, V1):
        del kvec
        box = _box_kernel(psi_n, bidx, ngkmax=geom.ngkmax)
        psi_r = ifftn(box) * scale
        phi_scalar = psi_r * charge_mask * V0
        phi_vector = jnp.zeros_like(psi_r)
        for i, (perm, phase) in enumerate(alpha_vertices):
            phi_vector = phi_vector + V1[i] * gamma_apply(
                psi_r, perm, phase, axis=2)
        # (component, k, band, spinor, x, y, z): the component batch is
        # replicated and the band shard moves from axis 1 to axis 2.
        phi_G = fftn(jnp.stack((phi_scalar, phi_vector), axis=0)) * fft_scale
        gx, gy, gz = gvec[:, 0], gvec[:, 1], gvec[:, 2]
        out = phi_G[..., gx, gy, gz]
        out = jnp.moveaxis(out, 0, -1)
        return out * gmask[None, None, None, :, None].astype(out.dtype)

    return Operator(
        apply=op, post=float(scalars.post), ncomp=2, consts=(V0, V1),
        key=("four_current_local_potential", grid, geom.ngkmax, geom.ns,
             charge_ns, scale, fft_scale))


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

    The VNL couplings are built at the FILE's ``nspinor``; a bispinor ψ carries
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

    IT DOES, HOWEVER, ENTER THE PARALLEL EFFICIENCY, and the module
    docstring used to deny that.  Building Z is replicated work: every rank
    builds it for every k.  Measured at MoS2 4×4 deck_b300 (nk=16, nb=128,
    ngkmax=1964, total_R=62; job 7889392) the build is 0.017 / 0.010 /
    0.020 s at P = 1 / 4 / 16 — flat — and 27 % → 51 % of this operator's
    whole cost over the kinetic baseline across that range.  Read the
    parallelism section of the module docstring for why it is left alone.
    """
    from psp import vnl_ops

    def op(psi_n, gvec, gmask, bidx, kvec):
        psi = _ket(psi_n, gmask)
        kdata = vnl_ops.build_vnl_kdata_traced(kvec, gvec, vnl_setup)
        ns_e = int(kdata.couplings.E_rows.shape[0])
        out = vnl_ops.apply_vnl(
            psi[:, :ns_e], kdata.Z, kdata.couplings)
        return _pad_spinor(out, int(psi.shape[1]))[None]

    return Operator(apply=op, post=1.0,
                    key=('vnl', geom.ngkmax, geom.ns, id(vnl_setup)))


#: The shipped relative sign of the nonlocal commutator term inside
#: :func:`dipole_operator`, as a named constant so the two arms of the
#: open question can be spelled without a magic ``-1.0`` at four call
#: sites.  ``-1.0`` is what every ``dipole.h5`` in the tree was built
#: with and is the default everywhere; see that function's SIGN section
#: for what is actually in dispute.
VNL_VELOCITY_SIGN_SHIPPED = -1.0

#: The other arm.  It is not "the fix" — the choice is the owner's — it
#: is the second reproducible configuration, so that a measurement of
#: the difference does not require patching a source file.
VNL_VELOCITY_SIGN_FLIPPED = +1.0


def dipole_operator(geom: SweepGeometry, *, bvec, blat,
                    vnl_setup=None,
                    vnl_velocity_sign=VNL_VELOCITY_SIGN_FLIPPED) -> Operator:
    """``v ∘ ψ = 2(k+G)_cart ψ ± (∂V_NL/∂K_cart) ψ`` — THREE components.

    The velocity matrix ``psp.get_dipole_mtxels`` writes is
    ``p - i[r, V_NL]`` in the stored convention, assembled here as
    ``p_cart + v_NL_cart``.  WHICH ARM A FILE WAS BUILT WITH IS NOT A
    PROPERTY OF THIS DOCSTRING: it is stamped into every ``dipole.h5``
    as ``prov_vnl_velocity_sign``, and files written before that stamp
    existed are the ``-1`` arm.  Both halves already have
    apply-to-ket kernels — ``dft_operators.apply_kinetic_velocity_to_ket``
    and ``vnl_ops.apply_vnl_velocity_to_ket``, each ``(3, nb, ns, nG)`` —
    so this operator is their difference and no velocity physics is
    written twice.

    ``vnl_setup=None`` reproduces ``--skip-vnl`` (p̂ only).

    The component axis is moved to the END, where the sweep's specs
    expect it; that is a transpose of the operator output, not of ψ.

    THE SIGN, WHICH WAS AN OPEN QUESTION AND IS NOW DECIDED
    -------------------------------------------------------
    ``vnl_velocity_sign`` multiplies the nonlocal term and nothing else.
    It takes exactly ``+1.0`` (the DEFAULT since 2026-08-09) or ``-1.0``
    (the arm every ``dipole.h5`` committed before that date was built
    with, kept reachable so those files stay reproducible).  The two
    signs are separate branches rather than a scalar multiply, so the
    legacy arm still executes the literal subtraction it always did.

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

    sign = float(vnl_velocity_sign)
    if sign not in (VNL_VELOCITY_SIGN_SHIPPED, VNL_VELOCITY_SIGN_FLIPPED):
        raise ValueError(
            f"GATE vnl_velocity_sign: got {vnl_velocity_sign!r}; the only "
            f"values are {VNL_VELOCITY_SIGN_SHIPPED} (shipped) and "
            f"{VNL_VELOCITY_SIGN_FLIPPED} (flipped).  This is a SIGN, not "
            f"a scale: an arbitrary multiplier would let a run report a "
            f"velocity operator that is neither arm of the open question "
            f"and that no comparison with BerkeleyGW characterises.")

    B = jnp.asarray(np.asarray(bvec, dtype=np.float64) * float(blat),
                    dtype=jnp.float64)
    flipped = sign > 0.0

    def op(psi_n, gvec, gmask, bidx, kvec, B):
        psi = _ket(psi_n, gmask)
        v = apply_kinetic_velocity_to_ket(psi, gvec, kvec, B)
        if vnl_setup is not None:
            kdata = vnl_ops.build_vnl_kdata_traced(kvec, gvec, vnl_setup,
                                                   compute_dZ=True)
            ns_e = int(kdata.couplings.E_rows.shape[0])
            v_nl = vnl_ops.apply_vnl_velocity_to_ket(
                psi[:, :ns_e], kdata.Z, kdata.dZ, kdata.couplings)
            pad = _pad_spinor(v_nl, int(psi.shape[1]))
            v = v + pad if flipped else v - pad
        return jnp.moveaxis(v, 0, -1)[None]

    return Operator(apply=op, post=1.0, ncomp=3, consts=(B,),
                    key=('dipole', geom.ngkmax, geom.ns, float(blat),
                         None if vnl_setup is None else id(vnl_setup),
                         sign))


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
                           include_transfer_q2: bool = False) -> Operator:
    r"""One apply-to-ket owner for raw current and exact uniform contact.

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
    so the WFN bra reshard and projector coefficient pass are not paid by
    separate current/contact drivers.

    ``include_transfer_q2=True`` extends the SAME transaction with the
    explicit ICL transfer derivatives at fixed large-component coefficients.
    For the repository's bra ``k-q`` orientation,

    ``Q_raw[i,a] = -(alpha/2) sigma_a sigma_i
                    -(alpha/4) V_NL,ia``

    and ``Q2_raw[i,a,b] = (alpha/6) V_NL,iab``.  The Pauli ordering comes
    from differentiating the BRA kinetic-balance endpoint.  It is formed by
    sweeping ``alpha_i dPsi_a`` and taking its negative band-space adjoint,
    thereby reusing :func:`common.bispinor_init.kinetic_balance_lift_jet`
    instead of spelling a second sigma-product kernel.

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

    from common.bispinor_init import HALFALPHA, kinetic_balance_lift_jet
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
            # dPsi/dK is independent of K.  Evaluating the canonical lift
            # jet at zero avoids threading a second reciprocal-lattice
            # operand through this already k-resolved operator.
            _lifted_zero, dpsi_dK = kinetic_balance_lift_jet(
                psi_L,
                jnp.zeros((int(psi_L.shape[-1]), 3),
                          dtype=psi_4.real.dtype))
            del _lifted_zero
            # This is B[i,a]=<Psi|alpha_i dPsi_a>.  The physical bra k-q
            # derivative is -B^dagger, preserving sigma_a sigma_i ordering.
            kinetic_q1_source = jnp.stack([
                gamma_apply(dpsi_dK, perm, phase, axis=2)
                for perm, phase in alpha_vertices
            ], axis=0)
            vnl_q1 = _pad_spinor(
                halfalpha.astype(psi_4.real.dtype)
                * vnl.dgamma_dq_cart_ket,
                int(psi_4.shape[1]))
            q1_adjoint_source = kinetic_q1_source - vnl_q1
            q2 = _pad_spinor(
                halfalpha.astype(psi_4.real.dtype)
                * vnl.d2gamma_dq2_cart_ket,
                int(psi_4.shape[1]))
            fields.extend((
                q1_adjoint_source.reshape(
                    9, *q1_adjoint_source.shape[2:]),
                q2.reshape(27, *q2.shape[3:]),
            ))

        packed = jnp.concatenate(tuple(fields), axis=0)
        return jnp.moveaxis(packed, 0, -1)[None]

    operator_key = (
        ("uniform_gauge_current_contact" if contact_enabled
         else "uniform_gauge_current"), geom.ngkmax, geom.ns,
        float(blat), id(vnl_setup))
    if transfer_q2:
        operator_key += ("explicit_transfer_q2",)
    return Operator(
        apply=op, post=1.0,
        ncomp=(48 if transfer_q2 else (12 if contact_enabled else 3)),
        key=operator_key)


def _gauge_hamiltonian_operator_fingerprint(
    *, wfn, vnl_setup, band_start: int, band_stop: int,
    geom: SweepGeometry, include_transfer_q2: bool,
) -> str:
    """Compose the one uniform/finite-q Hamiltonian operator identity.

    This is the exact grammar historically in
    :func:`sweep_uniform_gauge_matrix_elements`, moved without changing a
    byte so the arbitrary-transfer endpoint cannot invent a near-duplicate
    body fingerprint.  The finite-path quadrature/tolerance identity stays a
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
    from common.bispinor_init import KINETIC_BALANCE_LIFT_PROVENANCE
    from common.parallel_transport import (
        WFN_FINGERPRINT_SCHEME, fingerprint_update_value, wfn_fingerprint)
    from psp import vnl_ops

    digest = hashlib.sha256()
    digest.update(b"lorrax.uniform_gauge_operator/v1\0")
    for label, value in (
        ("wfn_scheme", WFN_FINGERPRINT_SCHEME),
        ("wfn", wfn_fingerprint(wfn)),
        ("vnl", vnl_fingerprint),
        ("vnl_gauge_path", vnl_ops.ICL_STRAIGHT_GAUGE_PATH),
        ("kinetic_balance", KINETIC_BALANCE_LIFT_PROVENANCE),
        ("band_interval", f"{start}:{stop}"),
        ("nk", str(int(geom.nk))),
        ("cell_volume", float(geom.cell_volume).hex()),
    ):
        fingerprint_update_value(digest, label, value)
    if bool(include_transfer_q2):
        fingerprint_update_value(
            digest, "transfer_jet", "explicit_q2_fixed_large_component_v1")
    return "sha256:" + digest.hexdigest()


def _uniform_gauge_sweep_fingerprint(
    *, wfn, vnl_setup, band_start: int, band_stop: int,
    geom: SweepGeometry, include_transfer_q2: bool,
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
        geom=geom, include_transfer_q2=bool(include_transfer_q2))


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
) -> UniformGaugeCurrentMatrixElements:
    """Select only ``Gamma_raw`` from the canonical uniform-gauge sweep.

    The operator closure, band reshard, VNL action and fingerprint owner are
    exactly those used by :func:`sweep_uniform_gauge_matrix_elements`.
    Only its static output component set differs, so a Hall-only producer
    does not retain contact/response matrices that it cannot consume.
    """
    fingerprint = _uniform_gauge_sweep_fingerprint(
        wfn=wfn, vnl_setup=vnl_setup, band_start=band_start,
        band_stop=band_stop, geom=geom, include_transfer_q2=False)
    gamma_raw = sweep_matrix_elements(
        psi_G,
        geom=geom,
        operator=uniform_gauge_operator(
            geom, bvec=bvec, blat=blat, vnl_setup=vnl_setup,
            include_contact=False, include_transfer_q2=False),
        gvecs=gvecs,
        gmask=gmask,
        box_index=box_index,
        kvecs=kvecs,
        use_scan=use_scan,
    )
    return UniformGaugeCurrentMatrixElements(
        gamma_raw=gamma_raw,
        hamiltonian_config_operator_fingerprint=fingerprint)


def sweep_uniform_gauge_matrix_elements(
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
    include_transfer_q2: bool = False,
) -> UniformGaugeMatrixElements:
    """Run the canonical uniform current/contact transaction once.

    The returned objects are views of one packed sweep output.  No host
    gather, duplicate contraction, or new sharding convention is introduced.
    The shared fingerprint is derived here from the canonical WFN identity,
    the VNL owner's host-built content identity, the physical band interval,
    and geometry.  This is the Hamiltonian/operator identity, deliberately
    separate from the artifact-format convention stamped by its writer.
    """
    fingerprint = _uniform_gauge_sweep_fingerprint(
        wfn=wfn, vnl_setup=vnl_setup, band_start=band_start,
        band_stop=band_stop, geom=geom,
        include_transfer_q2=bool(include_transfer_q2))

    packed = sweep_matrix_elements(
        psi_G,
        geom=geom,
        operator=uniform_gauge_operator(
            geom, bvec=bvec, blat=blat, vnl_setup=vnl_setup,
            include_transfer_q2=bool(include_transfer_q2)),
        gvecs=gvecs,
        gmask=gmask,
        box_index=box_index,
        kvecs=kvecs,
        use_scan=use_scan,
    )
    q1_raw = q2_raw = None
    if bool(include_transfer_q2):
        q1_source = packed[:, 12:21].reshape(
            int(packed.shape[0]), 3, 3,
            int(packed.shape[-2]), int(packed.shape[-1]))
        q1_raw = -jnp.swapaxes(jnp.conj(q1_source), -1, -2)
        q2_raw = packed[:, 21:48].reshape(
            int(packed.shape[0]), 3, 3, 3,
            int(packed.shape[-2]), int(packed.shape[-1]))
    return UniformGaugeMatrixElements(
        gamma_raw=packed[:, :3],
        lambda_raw=packed[:, 3:12].reshape(
            int(packed.shape[0]), 3, 3,
            int(packed.shape[-2]), int(packed.shape[-1])),
        hamiltonian_config_operator_fingerprint=fingerprint,
        dgamma_dq_raw=q1_raw,
        d2gamma_dq2_raw=q2_raw,
    )


def finite_transfer_current_to_centroids(
    psi_G,
    *,
    wfn,
    band_start: int,
    band_stop: int,
    geom: SweepGeometry,
    vnl_setup,
    r_mu,
    basis_receipt,
    iq_irr: int,
    path_order: int = 12,
    path_rtol: float = 1.0e-10,
    path_atol: float = 1.0e-12,
    projector_row_chunk: int = 64,
    g_chunk: int = 1024,
    fft_chunk_size: int | None = None,
    include_transfer_q2_identity: bool = True,
) -> FiniteTransferCurrentEndpoint:
    r"""Build the exact ICL finite-q current ket at current centroids.

    This is the missing transaction between the WFN/G-sphere lifetime and
    the current-ISDF/response layers.  It consumes, rather than rebuilds,
    the loader-owned paired ``(k,G,box_index)`` gauge, the symmetry-owned
    ``k -> k-q`` map, the canonical G-wrap lookup, the bounded ICL projector
    action, the monomial alpha matrices, and
    :func:`common.wfn_transforms.gflat_to_rmu`.

    For the repository's ``bra k-q, ket k`` orientation the returned field
    is

    ``Gamma_i(k,q)|Psi_nk> = alpha_i|Psi_nk>
       + (alpha_fs/2) Gamma_i^NL,ICL(k,q)|psi_L,nk>``.

    The local alpha action is translated onto the canonical target G sphere
    before it is added to the rectangular VNL action.  The target is then
    sampled with the target representative's paired ``box_index`` and Bloch
    vector.  Bands remain sharded over all processors throughout; no band
    matrix or fewer-than-P current carrier is formed.  The centroid tail is
    zero padded to :func:`runtime.padding.padded_mu_extent` so it can be
    converted to the incumbent two-face layout without inventing a second
    padding convention.

    ``basis_receipt`` must be the receipt carried by the target
    :class:`gw.wavefunction_bundle.Wavefunctions` bundle for these same
    source inputs.  The canonical receipt owner checks it before the
    finite-transfer kernel runs and the endpoint returns it unchanged.

    This routine intentionally emits one q-resolved endpoint.  It does not
    pass it to the incumbent q-independent k-FFT C/Z or Green builders:
    doing so would reuse this q's operator at every q.  Their future caller
    must stream q rows and use this same endpoint on both the zeta and chi
    sides.  FULL remains fail-closed until that complementary contraction,
    exact contact/downfolded completion, symmetry processing, and artifact
    identity propagation exist.
    """
    from common.bispinor_init import HALFALPHA
    from common.gamma_matrices import gamma_apply, gamma_perm_phase
    from common.kq_mapping import umklapp_G_wrap
    from common.parallel_transport import build_g_wrap_lookup
    from common.wfn_transforms import gflat_to_rmu
    from psp import vnl_ops
    from psp.dft_operators import padded_gvectors
    from symmetry_maps import bgw_integer_q_to_fractional

    start, stop = int(band_start), int(band_stop)
    if start < 0 or stop <= start or stop > int(wfn.nbands):
        raise ValueError(
            "finite-transfer current band interval must satisfy "
            f"0 <= start < stop <= WFN.nbands; got [{start},{stop})")
    if stop - start != int(geom.nb_logical):
        raise ValueError(
            "finite-transfer current band interval does not match "
            f"SweepGeometry: [{start},{stop}) vs "
            f"nb_logical={int(geom.nb_logical)}")
    if int(geom.ns) != 4:
        raise ValueError(
            "finite-transfer current requires the canonical four-component "
            f"kinetic-balance carrier; geom.ns={int(geom.ns)}")
    if vnl_setup is None or int(vnl_setup.nspinor) != 2:
        raise ValueError(
            "finite-transfer current requires the canonical two-component "
            "Pauli VNLSetup")
    if bool(include_transfer_q2_identity) and vnl_setup.Gppp_table is None:
        raise ValueError(
            "finite-transfer FULL-body identity requires VNLSetup built "
            "with compute_transfer_q2=True so it exactly matches the "
            "uniform head transaction")
    mesh_shape = tuple(int(v) for v in geom.mesh.devices.shape)
    if (tuple(geom.mesh.axis_names) != ("x", "y")
            or len(mesh_shape) != 2 or mesh_shape[0] != mesh_shape[1]):
        raise ValueError(
            "finite-transfer current requires the canonical square (x,y) "
            f"processor mesh; got axes={tuple(geom.mesh.axis_names)}, "
            f"shape={mesh_shape}")

    required_loader_api = ("symmetry", "box_index_dev")
    missing_loader_api = tuple(
        name for name in required_loader_api
        if not callable(getattr(wfn, name, None)))
    if missing_loader_api:
        raise TypeError(
            "finite-transfer current requires one canonical WfnLoader; "
            f"missing callable API {missing_loader_api}")
    if tuple(int(v) for v in wfn.fft_grid) != tuple(geom.fft_grid):
        raise ValueError(
            "finite-transfer current SweepGeometry.fft_grid must match "
            f"WfnLoader.fft_grid exactly; got {tuple(geom.fft_grid)} vs "
            f"{tuple(int(v) for v in wfn.fft_grid)}")
    if int(wfn.ngkmax) != int(geom.ngkmax):
        raise ValueError(
            "finite-transfer current SweepGeometry.ngkmax must match "
            f"WfnLoader.ngkmax exactly; got {int(geom.ngkmax)} vs "
            f"{int(wfn.ngkmax)}")
    r_mu_host = np.ascontiguousarray(np.asarray(r_mu, dtype=np.int32))
    if r_mu_host.ndim != 2 or int(r_mu_host.shape[1]) != 3:
        raise ValueError(
            f"finite-transfer current r_mu must be (n_rmu,3); got "
            f"{r_mu_host.shape}")
    from file_io.wfn_basis import WavefunctionBasisReceipt
    if not isinstance(basis_receipt, WavefunctionBasisReceipt):
        raise TypeError(
            "finite-transfer current requires the canonical immutable "
            "WavefunctionBasisReceipt from its target Wavefunctions bundle; "
            f"got {type(basis_receipt)!r}")
    from runtime.padding import padded_mu_extent
    expected_mu_padded = padded_mu_extent(
        int(r_mu_host.shape[0]), geom.mesh)
    basis_receipt.assert_matches_source(
        wfn=wfn, role="transverse", bispinor=True,
        band_interval=(start, stop),
        fft_grid=geom.fft_grid,
        centroid_fft_idx=r_mu_host,
        n_rmu_logical=int(r_mu_host.shape[0]),
        n_rmu_padded=expected_mu_padded,
        where="finite-transfer current endpoint")

    # Only after the target bundle's receipt authenticates the source do any
    # finite-transfer current arrays enter construction.
    psi = jnp.asarray(psi_G, dtype=jnp.complex128)
    if int(psi.shape[1]) not in (int(geom.nb_logical), int(geom.nb)):
        raise ValueError(
            "finite-transfer current psi_G band axis must be logical or "
            f"mesh padded ({int(geom.nb_logical)} or {int(geom.nb)}); "
            f"got {tuple(psi.shape)}")
    if (int(psi.shape[0]), int(psi.shape[2]), int(psi.shape[3])) != (
            int(geom.nk), 4, int(geom.ngkmax)):
        raise ValueError(
            "finite-transfer current psi_G must have shape "
            f"(nk,nb,4,ngkmax)=({int(geom.nk)},nb,4,"
            f"{int(geom.ngkmax)}); got {tuple(psi.shape)}")
    psi = pad_axis(psi, geom.p_prod, axis=1).array
    sym = wfn.symmetry()
    gtab = padded_gvectors(wfn, k="full_bz")
    gvecs_host = np.ascontiguousarray(gtab.gvecs, dtype=np.int32)
    ngk_valid_host = np.ascontiguousarray(gtab.ngk, dtype=np.int32)
    gmask_host = np.ascontiguousarray(gtab.mask, dtype=np.float64)
    kvecs_host = np.ascontiguousarray(gtab.kvecs, dtype=np.float64)
    box_index = wfn.box_index_dev(k="full_bz", mesh=geom.mesh)
    expected_g = (int(geom.nk), int(geom.ngkmax), 3)
    if gvecs_host.shape != expected_g:
        raise ValueError(
            f"finite-transfer current paired G table must be {expected_g}; "
            f"got {gvecs_host.shape}")
    iq = int(iq_irr)
    q_rows_int = np.asarray(sym.q_irr_kgrid_int, dtype=np.int64)
    q_full_rows = np.asarray(sym.q_irr_full_idx, dtype=np.int32)
    if q_rows_int.ndim != 2 or q_rows_int.shape[1] != 3:
        raise ValueError(
            "finite-transfer current requires SymMaps.q_irr_kgrid_int "
            f"with shape (nq_irr,3); got {q_rows_int.shape}")
    if q_full_rows.shape != (int(q_rows_int.shape[0]),):
        raise ValueError(
            "finite-transfer current q-IBZ rows/full-row labels disagree: "
            f"{q_rows_int.shape} vs {q_full_rows.shape}")
    if iq < 0 or iq >= int(q_rows_int.shape[0]):
        raise ValueError(
            f"finite-transfer current iq_irr={iq} outside symmetry-owned "
            f"q-IBZ rows [0,{int(q_rows_int.shape[0])})")
    kgrid_host = np.asarray(wfn.kgrid, dtype=np.int64)
    if kgrid_host.shape != (3,) or np.any(kgrid_host <= 0):
        raise ValueError(
            "finite-transfer current requires WfnLoader.kgrid with three "
            f"positive entries; got {kgrid_host}")
    if np.any(q_rows_int < 0) or np.any(q_rows_int >= kgrid_host[None, :]):
        raise ValueError(
            "finite-transfer current SymMaps q-IBZ integer rows must lie "
            f"inside WfnLoader.kgrid={kgrid_host.tolist()}")
    q_crys = bgw_integer_q_to_fractional(q_rows_int[iq], kgrid_host)
    iq_full = int(q_full_rows[iq])
    kqfull = np.asarray(sym.kqfull_map, dtype=np.int32)
    if kqfull.shape != (int(geom.nk), int(geom.nk)):
        raise ValueError(
            "finite-transfer current requires the symmetry-owned full "
            f"k-q map with shape ({int(geom.nk)},{int(geom.nk)}); got "
            f"{kqfull.shape}")
    if iq_full < 0 or iq_full >= int(geom.nk):
        raise ValueError(
            f"finite-transfer q-IBZ full-row label {iq_full} is outside "
            f"[0,{int(geom.nk)})")
    kminq_idx = np.ascontiguousarray(
        kqfull[:, iq_full], dtype=np.int32)
    if kminq_idx.shape != (int(geom.nk),):
        raise ValueError(
            "symmetry k-q map has wrong shape for finite-transfer current: "
            f"{kminq_idx.shape} vs ({int(geom.nk)},)")
    if (np.any(kminq_idx < 0)
            or np.any(kminq_idx >= int(geom.nk))):
        raise ValueError("symmetry k-q map contains an out-of-range row")
    if not np.array_equal(
            np.sort(kminq_idx), np.arange(int(geom.nk), dtype=np.int32)):
        raise ValueError(
            "finite-transfer current requires each symmetry-owned k-q row "
            "map to be a permutation of the full BZ")
    target_k_host = kvecs_host[kminq_idx]
    g_wrap = np.ascontiguousarray(np.asarray(jax.device_get(
        umklapp_G_wrap(
            jnp.asarray(kvecs_host), jnp.asarray(target_k_host),
            jnp.asarray(q_crys))), dtype=np.int32))
    del target_k_host

    # Reverse lookup: for each TARGET G row, locate the SOURCE coefficient
    # whose plane wave becomes it after the photon transfer.  Since
    # k-q=k_target+G_wrap,
    #
    #   exp(-iq.r) exp(i(k+G_source).r)
    #       = exp(i(k_target+G_target).r)
    #
    # requires G_source=G_target-G_wrap.  This is the canonical lookup with
    # center/neighbor roles reversed and ``-G_wrap``, not another G
    # dictionary implementation.
    source_for_target = np.zeros(
        (int(geom.nk), int(geom.ngkmax)), dtype=np.int32)
    source_for_target_valid = np.zeros_like(source_for_target, dtype=bool)
    for ik in range(int(geom.nk)):
        target_row = int(kminq_idx[ik])
        idx, valid = build_g_wrap_lookup(
            gvecs_host[ik], gvecs_host[target_row], -g_wrap[ik],
            ngk_neighbor=int(ngk_valid_host[ik]),
            ngk_center=int(ngk_valid_host[target_row]))
        source_for_target[ik] = idx
        source_for_target_valid[ik] = valid

    gvecs_j = jnp.asarray(gvecs_host, dtype=jnp.int32)
    gmask_j = jnp.asarray(gmask_host, dtype=jnp.float64)
    wrap_j = jnp.asarray(g_wrap, dtype=jnp.int32)
    kminq_j = jnp.asarray(kminq_idx, dtype=jnp.int32)
    source_for_target_j = jnp.asarray(source_for_target, dtype=jnp.int32)
    source_for_target_valid_j = jnp.asarray(source_for_target_valid)
    q_j = jnp.asarray(q_crys, dtype=jnp.float64)
    alpha_vertices = tuple(gamma_perm_phase(i) for i in (1, 2, 3))
    halfalpha = jnp.asarray(HALFALPHA, dtype=jnp.float64)
    endpoint_sharding = NamedSharding(
        geom.mesh, P(None, ('x', 'y'), None, None, None))

    def _run(psi_, source_g_, source_mask_, source_k_, q_, wrap_,
             target_rows_, source_index_, source_valid_):
        psi_ = jax.lax.with_sharding_constraint(
            psi_, NamedSharding(geom.mesh, geom.spec_sphere_xy))

        def one_k(xs):
            (psi_k, G_source, mask_source, k_source, target_row, wrap_k,
             source_index, source_valid) = xs
            # Select both halves of the target (k,G) gauge from the same
            # loader tables inside the k scan.  No per-q reordered full G
            # table or independently paired k table becomes an operand.
            G_target = source_g_[target_row]
            mask_target = source_mask_[target_row]
            k_target = source_k_[target_row]
            physical = psi_k * mask_source[None, None, :].astype(psi_k.dtype)
            alpha_source = jnp.stack([
                gamma_apply(physical, perm, phase, axis=1)
                for perm, phase in alpha_vertices
            ], axis=1)  # (nb,3,4,nG_source)
            alpha_target = jnp.take(
                alpha_source, source_index, axis=-1)
            alpha_target = jnp.where(
                source_valid[None, None, None, :], alpha_target,
                jnp.zeros((), dtype=alpha_target.dtype))

            finite = vnl_ops.compute_icl_vnl_finite_transfer_to_ket(
                physical[:, :2], G_source, G_target,
                k_source, k_target, q_, wrap_k, vnl_setup,
                mask_source, mask_target,
                path_order=int(path_order), path_rtol=float(path_rtol),
                path_atol=float(path_atol),
                projector_row_chunk=int(projector_row_chunk),
                g_chunk=int(g_chunk))
            gamma_vnl = _pad_spinor(finite.gamma_cart_ket, 4)
            gamma_vnl = jnp.moveaxis(gamma_vnl, 0, 1)
            total = alpha_target + halfalpha.astype(
                alpha_target.real.dtype) * gamma_vnl
            return (total, finite.ward_residual_abs,
                    finite.ward_residual_rel, finite.ward_reference_norm,
                    finite.certified)

        _, values = jax.lax.scan(
            lambda carry, xs: (carry, one_k(xs)), None,
            (psi_, source_g_, source_mask_, source_k_, target_rows_, wrap_,
             source_index_, source_valid_),
            unroll=1)
        current_G = jax.lax.with_sharding_constraint(
            values[0], endpoint_sharding)
        return (current_G, values[1], values[2], values[3], values[4])

    kernel = _cached_jit(
        'finite_transfer_current_to_ket',
        (tuple(int(v) for v in psi.shape), id(geom.mesh), id(vnl_setup),
         int(path_order), float(path_rtol), float(path_atol),
         int(projector_row_chunk), int(g_chunk),
         _sharding_key(psi)),
        lambda: jax.jit(_run))
    current_G, ward_abs, ward_rel, ward_ref, certified = kernel(
        psi, gvecs_j, gmask_j, jnp.asarray(kvecs_host), q_j, wrap_j,
        kminq_j, source_for_target_j, source_for_target_valid_j)

    certified_host = np.asarray(jax.device_get(certified), dtype=bool)
    if not np.all(certified_host):
        abs_host = np.asarray(jax.device_get(ward_abs), dtype=np.float64)
        rel_host = np.asarray(jax.device_get(ward_rel), dtype=np.float64)
        bad = np.flatnonzero(~certified_host)
        raise RuntimeError(
            "GATE finite_transfer_current_icl_ward_uncertified: exact ICL "
            f"endpoint failed for source k rows {bad.tolist()}; "
            f"max_abs={float(np.max(abs_host[bad])):.6e}, "
            f"max_rel={float(np.max(rel_host[bad])):.6e}, "
            f"path_order={int(path_order)}, rtol={float(path_rtol):.3e}, "
            f"atol={float(path_atol):.3e}")

    nk, nb, _, ns, ngkmax = (int(v) for v in current_G.shape)
    current_flat = current_G.reshape(nk, nb, 3 * ns, ngkmax)
    current_rmu_flat = gflat_to_rmu(
        current_flat, box_index, r_mu_host, mesh=geom.mesh,
        fft_grid=geom.fft_grid, kvecs_frac=kvecs_host,
        k_row_map=kminq_idx, norm="ortho", chunk_size=fft_chunk_size)
    del current_G, current_flat
    n_rmu_logical = int(r_mu_host.shape[0])
    # ``expected_mu_padded`` is the canonical runtime-padding owner's exact
    # target (including the fixed-P pad-invariance test knob).  Using it as
    # the divisor retains :func:`pad_axis` as the sole array pad spelling.
    rmu_pad = pad_axis(current_rmu_flat, expected_mu_padded, axis=-1)
    if int(rmu_pad.logical) != n_rmu_logical:
        raise AssertionError(
            "canonical centroid padding changed the logical extent: "
            f"{int(rmu_pad.logical)} != {n_rmu_logical}")
    current_rmu_flat = rmu_pad.array
    n_rmu_padded = int(rmu_pad.padded)
    current_nmu_sharding = NamedSharding(geom.mesh, PSI_NMU_SPEC)
    current_mun_sharding = NamedSharding(geom.mesh, PSI_MUN_SPEC)
    to_faces = _cached_jit(
        'finite_transfer_current_to_faces',
        (tuple(int(v) for v in current_rmu_flat.shape), id(geom.mesh)),
        lambda: jax.jit(
            lambda value: (value, value.transpose(0, 2, 3, 1)),
            out_shardings=(current_nmu_sharding, current_mun_sharding)))
    current_nmu, current_mun = to_faces(current_rmu_flat)
    current_nmu = current_nmu.reshape(nk, nb, 3, ns, n_rmu_padded)
    current_mun = current_mun.reshape(nk, 3, ns, n_rmu_padded, nb)
    del current_rmu_flat

    fingerprint = _gauge_hamiltonian_operator_fingerprint(
        wfn=wfn, vnl_setup=vnl_setup, band_start=start, band_stop=stop,
        geom=geom,
        include_transfer_q2=bool(include_transfer_q2_identity))
    path_fingerprint = vnl_ops.icl_vnl_finite_transfer_operator_fingerprint(
        vnl_setup, path_order=int(path_order), path_rtol=float(path_rtol),
        path_atol=float(path_atol))
    return FiniteTransferCurrentEndpoint(
        current_nmu=current_nmu,
        current_mun=current_mun,
        n_rmu_logical=n_rmu_logical,
        basis_receipt=basis_receipt,
        iq_irr=iq,
        q_irr_kgrid_int=np.ascontiguousarray(q_rows_int[iq], dtype=np.int32),
        q_crys=np.ascontiguousarray(q_crys, dtype=np.float64),
        kminq_idx=kminq_idx,
        g_wrap=g_wrap,
        vnl_ward_residual_abs=ward_abs,
        vnl_ward_residual_rel=ward_rel,
        vnl_ward_reference_norm=ward_ref,
        hamiltonian_config_operator_fingerprint=fingerprint,
        vnl_path_operator_fingerprint=path_fingerprint,
    )


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
        Its pad mask.  MANDATORY, not optional.  Pad rows carry the
        FFT-box sentinel Miller index (see ``common.gvec_fft_box``), which
        is a valid box cell, so a forgotten mask does not crash — it
        silently contracts the sentinel column into every matrix element.
        The sentinel is chosen so that no physical G of a padded row maps
        to it, which makes the omission detectable rather than harmless;
        it does not make the mask optional.
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

    A BAND WINDOW NEEDS NO ARGUMENT
    -------------------------------
    ``⟨m|O|n⟩`` takes BOTH indices from one ψ, so restricting the sweep to a
    band window is a windowed ψ plus a geometry built at the window's
    ``nb``::

        sweep_matrix_elements(psi_G[:, lo:hi], geom=SweepGeometry(
            ..., nb=hi - lo), ...)      ->  (nk, hi-lo, hi-lo)

    and the block comes back at WINDOW indices.  Everything else keys off
    ``geom``: the band pad, both reshards, the einsum and the output spec.
    A ``band_window=`` argument would be a second way to say the same thing
    and one more thing for every call site to get right — the mistake the
    2026-08-04 SlabIO padding ruling names — so there deliberately is none.
    The one cost is that slicing a ``('x','y')``-sharded ψ on the SHARDED
    axis is an eager reshard; it is paid once, outside the scan, against a
    scan whose every stage is then ``nb/(hi-lo)`` times smaller.
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
    psi = pad_axis(psi, geom.p_prod, axis=1).array

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
            body, None, (psi_n_XY, gvecs_, gmask_, bidx_, kvecs_), unroll=1)
        return jax.lax.with_sharding_constraint(H, block_sharding)

    # NO ``donate_argnums``, and that is a decision rather than an omission.
    # ψ is the only operand big enough to be worth donating — 129 MB global
    # at the MoS2 4×4 shape, 8 MB/rank at P=16 — and its lifetime is a
    # property of the CALLER, not of the sweep: ``gw.kin_ion_io`` does
    # ``del H_kin_ion, psi_G`` on the next line, but
    # ``gw.sc_iteration._PSI_G_CACHE`` holds the SAME ψ across every
    # density-SC iteration, so a blanket donation would invalidate a buffer
    # the next iteration reads.  Expressing it would need a per-call-site
    # flag, and it would not reach the thing that is actually large: the
    # 180 MiB per-k box lives inside the scan body, where donation of a jit
    # argument cannot help it (job 7889383's stage table).
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
