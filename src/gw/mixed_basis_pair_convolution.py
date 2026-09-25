"""Mixed-basis pair convolution: two compact operators on plane-wave spheres → one on a response sphere.

The operation, with no prefactor::

    X_q(G, G') = Σ_R Σ_{r,r'} e^{-i(q+G)·(r+R)} Σ_{αβ} A^{αβ}(r+R, r') conj C^{αβ}(r+R, r') e^{+i(q+G')·r'}

    A(r+R, r') = N_k⁻¹ Σ_k e^{ik·R} a_k(r, r'),   a_k(r, r') = Σ_{p,p'} e^{i(k+p)·r} A_k(p α, p' β) e^{-i(k+p')·r'}

and the same for C.  r, r' run over the N_r points of the cell's FFT box, R over the
N_k cells of the Born–von Kármán supercell, k over the C-order k-grid, p, p' over
the operand's plane-wave sphere at k, and G, G' over the output sphere at q.  The
caller applies Ω/N_r², spin and occupation factors.

χ₀ in the adjoint form is the first caller: A = Gc(τ), C = Gv(τ) (the adjoint of the
second propagator, so both factors share ownership and indexing), and
X_R = Σ_αβ Gc_R ⊙ conj(Gv_R) is the product the ISDF χ₀ accumulates.  Σ = G ⊙ W is the
same pair convolution with the second operand on the response sphere and the
output on the ψ sphere (sandbox TASTE 97); the operand and output bases are
therefore separate arguments.

Schedule (one τ node; P ranks; every stage one ``shard_map`` over ``('x','y')``)::

    1  A_k̄(p, p')  2-D tile at the parents            → slab: p over all P, p' local
    2  typed transport, column half (p' → src(p'), phase, spin), on the compact tile
    3  p' → r'  (sphere → box, sign −1), × e^{-ik·r'}
    4  all-to-all: r' over all P, every p local        H_k(p, r')   [one r' chunk]
    5  per batch J of r' columns, every k local:
         typed transport, row half; p → r (sign +1); k → R; Σ_αβ A_R conj C_R; R → q
         (one fused k-convolution); keep q in the output set; × e^{-iq·r};
         r → G (box → sphere)                        T_q(G, r'_J)
    6  all-to-all: G over all P, every r' local; × e^{+iq·r'}; r' → G';
       all-to-all to the 2-D layout                  X_q(G, G')  P(None, 'x', 'y')

The typed transport (``SphereTransport``) is ``symmetry_maps.unfold_load_tables``'
pair-transpose rule in G space: the compact tiles are unfolded, ψ never is.  Its
column half acts on the compact tile before step 3 (every p' is local there) and
its row half inside the step-5 gather (every p is local there), so neither half
moves data between ranks; the product of the two halves is the whole action.

Memory law, per rank, complex128 (``describe()`` prints it)::

    resident  16·[N_k·n_s²·(M_A + M_C)·N_r/(P·n_c) + n_q·M_X·N_r/P]
    batch     16·c·N_k·J·N_r,  c ≈ 4·n_s² + 2

No N_k·N_r² object exists.  n_c (r' chunks), J (the batch), the k chunk of steps
2–4 and the q chunk of step 6 all come from the device budget; each is one when
everything fits (sandbox TASTE 96).  The r' chunking recomputes steps 2–3 per chunk
(O(N_k M N_r log N_r) each, ~M/N_r of the step-5 work) and bounds H by 1/n_c.

Backends, chosen when the plan is built:

* ``'router'`` (CUDA and cpu meshes): the box transforms are ``LocalFourierPlan``
  on the lowering platform's leg (CUDA: cuBLAS/cuBLASDx supported-axis GEMMs and
  cuFFT) and step 5's k-convolution is ``ffi.fft.make_fused_conv_kplane`` (CUDA:
  nvidia-mathdx mode 6, the Bloch phase and the [A | C] split on its load; cpu:
  the router's plan route).
* ``'xla'`` (any platform; the portable fallback): the same stages with every box
  and k transform one ``jnp.fft`` call (``common.fft_helpers``' local FFT) between
  gathers that embed and restrict the supports, and the product in XLA.

``backend='xla'`` on a CUDA mesh is the parity and benchmark arm; production
passes nothing.  n_s is 1, 2 or 4 (the bispinor width; mode 6 takes n_s ≤ 4): the
spin blocks ride as axes of every stage and meet only in the step-5 trace.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.fft_helpers import local_fftn3, local_ifftn3
from common.fourier_plan import LocalFourierPlan
from common.shard_map import shard_map
from runtime.padding import padded_axis

__all__ = ["SphereSet", "SphereTransport", "PairOperand", "MixedBasisPairConvolution",
           "PairConvChunks", "plan_pair_convolution_chunks", "alias_free_margin"]

_XY = ("x", "y")
_C16 = 16                         # bytes per complex128 element
_BACKENDS = (None, "router", "xla")


# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, eq=False)
class SphereSet:
    """Plane-wave spheres on one FFT box, one row per k (or q) point.

    ``gvecs (n, width, 3)`` Miller indices; slots at or past ``ngk[i]`` are pad
    slots, never read.  ``frac (n, 3)`` is the crystal-coordinate k of each row,
    the representative its Miller indices go with (``k + G`` is the wavevector).
    """
    gvecs: np.ndarray
    ngk: np.ndarray
    frac: np.ndarray

    def __post_init__(self):
        g = np.asarray(self.gvecs, dtype=np.int64)
        ngk = np.asarray(self.ngk, dtype=np.int64).reshape(-1)
        frac = np.asarray(self.frac, dtype=np.float64)
        if g.ndim != 3 or g.shape[2] != 3 or ngk.shape != (g.shape[0],) \
                or frac.shape != (g.shape[0], 3):
            raise ValueError(f"SphereSet: want gvecs (n, width, 3), ngk (n,), frac (n, 3); "
                             f"got {g.shape}, {ngk.shape}, {frac.shape}")
        if np.any(ngk < 1) or np.any(ngk > g.shape[1]):
            raise ValueError(f"SphereSet: ngk must lie in [1, {g.shape[1]}]")
        object.__setattr__(self, "gvecs", g)
        object.__setattr__(self, "ngk", ngk)
        object.__setattr__(self, "frac", frac)

    @property
    def n(self) -> int:
        return int(self.gvecs.shape[0])

    @property
    def width(self) -> int:
        return int(self.gvecs.shape[1])

    def live(self) -> np.ndarray:
        """``(n, width)`` bool, True on the physical slots."""
        return np.arange(self.width)[None, :] < self.ngk[:, None]

    def union_support(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per axis, the sorted Miller values any live slot takes: the bounding box of the union."""
        live = self.live()
        return tuple(np.unique(self.gvecs[..., a][live]) for a in range(3))

    def box_cells(self, support) -> np.ndarray:
        """``(n, width)`` int64: each live slot's flat cell in the ``support`` box, −1 on a pad slot."""
        pos = [np.searchsorted(s, self.gvecs[..., a]) for a, s in enumerate(support)]
        k1, k2 = len(support[1]), len(support[2])
        cell = (pos[0] * k1 + pos[1]) * k2 + pos[2]
        return np.where(self.live(), cell, -1)


@dataclasses.dataclass(frozen=True, eq=False)
class SphereTransport:
    """The typed G-space transport of a compact operator from its parent rows to the full grid::

        O_k[p α, p' β] = Σ_γδ U_k[α,γ] · mph_k(p) · S_{row(k)}[src_k(p) γ, src_k(p') δ] · nph_k(p') · conj U_k[β,δ]

    ``S`` is the parent operator, or its transposed partner on an antiunitary row
    (``anti``); ``src_k(p)`` the parent slot of child slot ``p``; with
    ``ph_k(p) = phase[k, p]`` (``e^{-2πi (k̄+G)·t}``, ``isdf.zeta_mubatch.typed_child_G_tables``),
    ``mph = ph``, ``nph = conj(ph)`` on a unitary row and the conjugates on an
    antiunitary one: ``symmetry_maps.unfold_load_tables``' pair-transpose rule on
    sphere slots.  ``row (N_k,)``, ``anti (N_k,)``, ``spin (N_k, n_s, n_s)``,
    ``src``/``phase (N_k, width)`` (pad slots never read), ``n_parent`` rows.
    """
    row: np.ndarray
    anti: np.ndarray
    spin: np.ndarray
    src: np.ndarray
    phase: np.ndarray
    n_parent: int

    def __post_init__(self):
        row = np.asarray(self.row, dtype=np.int32).reshape(-1)
        nk = int(row.size)
        anti = np.asarray(self.anti, dtype=bool).reshape(-1)
        spin = np.asarray(self.spin, dtype=np.complex128)
        src = np.asarray(self.src, dtype=np.int64)
        phase = np.asarray(self.phase, dtype=np.complex128)
        if (anti.shape != (nk,) or spin.ndim != 3 or spin.shape[0] != nk
                or spin.shape[1] != spin.shape[2] or src.ndim != 2 or src.shape[0] != nk
                or phase.shape != src.shape):
            raise ValueError("SphereTransport: want row/anti (N_k,), spin (N_k, n_s, n_s), "
                             f"src/phase (N_k, width); got {row.shape}, {anti.shape}, "
                             f"{spin.shape}, {src.shape}, {phase.shape}")
        if nk and (row.min() < 0 or row.max() >= int(self.n_parent)):
            raise ValueError(f"SphereTransport: parent rows must lie in [0, {self.n_parent})")
        for name, val in (("row", row), ("anti", anti), ("spin", spin), ("src", src),
                          ("phase", phase)):
            object.__setattr__(self, name, val)
        object.__setattr__(self, "n_parent", int(self.n_parent))

    @property
    def ns(self) -> int:
        return int(self.spin.shape[-1])

    @classmethod
    def identity(cls, sphere: SphereSet, ns: int) -> "SphereTransport":
        """Full-grid input: every row its own parent, every slot its own source."""
        nk, w = sphere.n, sphere.width
        return cls(row=np.arange(nk), anti=np.zeros(nk, bool),
                   spin=np.broadcast_to(np.eye(ns, dtype=np.complex128), (nk, ns, ns)),
                   src=np.broadcast_to(np.arange(w), (nk, w)),
                   phase=np.ones((nk, w), np.complex128), n_parent=nk)

    @classmethod
    def typed(cls, plan, *, fft_grid, parent_sphere_index, children: SphereSet) -> "SphereTransport":
        """The parents' typed transport onto ``children``.

        ``plan`` carries the symmetry tables (``irr_idx``, ``sym_idx``,
        ``spin_action_full``, ``k_parent_frac``, ``n_sym_spatial``,
        ``spatial_ops``, ``translations``; a ``gw.centroid_k_unfold`` plan does);
        ``parent_sphere_index (n_parent, width)`` is the parents' slot → flat box
        cell table (``common.gvec_fft_box.build_sphere_box_index``, ``≥ N_r`` on a
        pad slot).  The slot map and phase are ``typed_child_G_tables``'.
        """
        from isdf.zeta_mubatch import typed_child_G_tables
        pslot, phase, anti = typed_child_G_tables(
            plan, fft_grid=fft_grid, sphere_par=parent_sphere_index,
            gvec_child=children.gvecs, ngk_child=children.ngk, k_child=children.frac)
        return cls(row=np.asarray(plan.irr_idx), anti=anti, spin=np.asarray(plan.spin_action_full),
                   src=pslot, phase=phase,
                   n_parent=int(np.asarray(parent_sphere_index).shape[0]))


@dataclasses.dataclass(frozen=True, eq=False)
class PairOperand:
    """One compact operator's basis: its sphere at every full-grid k and its transport from the parents."""
    sphere: SphereSet
    transport: SphereTransport

    def __post_init__(self):
        if self.transport.src.shape != (self.sphere.n, self.sphere.width):
            raise ValueError(f"PairOperand: transport src {self.transport.src.shape} does not "
                             f"match the sphere ({self.sphere.n}, {self.sphere.width})")


# ---------------------------------------------------------------------------
# Box, aliasing and chunk planning (host)
# ---------------------------------------------------------------------------

def alias_free_margin(fft_grid, left_support, right_support, out_support) -> np.ndarray:
    """Per axis, ``N − (the smallest box that keeps the output sphere alias-free)``.

    The product ``A ⊙ conj C`` holds r-frequencies ``G = p − p'' + g₀`` with ``p``,
    ``p''`` in the operand supports and an umklapp ``g₀ ∈ {−1, 0, 1}`` (k, k' and q
    are grid representatives); a frequency outside the kept support aliases onto
    it iff the two differ by a nonzero multiple of ``N``.  The box is alias-free
    iff every margin is ≥ 1; each operand support must also fit (distinct cells).
    """
    fg = np.asarray(fft_grid, dtype=np.int64)
    out = np.empty(3, dtype=np.int64)
    for a in range(3):
        l, r, o = (np.asarray(s[a], dtype=np.int64) for s in (left_support, right_support, out_support))
        lo, hi = int(l.min() - r.max() - 1), int(l.max() - r.min() + 1)
        need = max(hi - int(o.min()), int(o.max()) - lo,
                   int(l.max() - l.min()), int(r.max() - r.min()))
        out[a] = int(fg[a]) - need
    return out


@dataclasses.dataclass(frozen=True)
class PairConvChunks:
    """The budget-derived schedule: r' chunks, batch width, k and q chunks, and the byte model."""
    n_c: int
    J: int
    kc: int
    qc: int
    n_r_carrier: int
    bytes_resident: int
    bytes_expand: int
    bytes_middle: int
    bytes_final: int
    target: int

    @property
    def hwm(self) -> int:
        return self.bytes_resident + max(self.bytes_expand, self.bytes_middle, self.bytes_final)


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def plan_pair_convolution_chunks(*, n_ranks, n_k, n_s, widths, width_out, n_q, n_r,
                                 kboxes, kbox_out, n_parent_tiles, target_bytes,
                                 j_cap=64, n_c=None, J=None, kc=None, qc=None) -> PairConvChunks:
    """The schedule for one τ node (every count one when everything fits).

    ``widths``/``kboxes``: the two operands' slot carriers and union-box cells;
    ``n_parent_tiles``: the slab copies' element count per rank (inputs held).
    ``n_c``/``J``/``kc``/``qc`` pin those counts (tests and the benchmark); the rest follow
    (``kc`` must divide ``n_k`` and ``qc`` must divide ``n_q``).
    Refuses ``GATE pairconv-capacity`` when even n_c at one batch column per rank
    per chunk and unit k and q chunks exceeds ``target_bytes``.
    """
    Pn, nk, ns = int(n_ranks), int(n_k), int(n_s)
    Mw, Mo, nq, nr = [int(w) for w in widths], int(width_out), int(n_q), int(n_r)

    def carrier(nc, j):
        return padded_axis(nr, Pn * nc * j, name="mixed-basis r' columns").carrier

    def resident(nc, j):
        cols = carrier(nc, j) // Pn
        H = nk * ns * ns * sum(Mw) * (cols // nc)
        T = nq * Mo * cols
        return _C16 * (H + T + int(n_parent_tiles) + nq * Mo * Mo // Pn)

    def expand(kc, nc, j):          # the full r' transform of one k chunk, its phased gather, the all-to-all
        cols = carrier(nc, j) // Pn // nc
        return _C16 * max(kc * ns * ns * (m // Pn) * (kb + 3 * nr) + kc * ns * ns * m * cols
                          for m, kb in zip(Mw, kboxes))

    def middle(j):                  # compact gathers, the p→r output and its workspace, F, U, Y, the r→G box
        return _C16 * (nk * j * ns * ns * sum(kboxes) + 2 * nk * j * ns * ns * 2 * nr + nk * nr
                       + nk * j * nr + 2 * nq * j * nr + 2 * nq * j * kbox_out + 2 * nq * Mo * j)

    def final(qc, nc, j):           # the all-to-all output, the phased box, its transform and gather
        return _C16 * (3 * qc * (Mo // Pn) * carrier(nc, j) + 2 * qc * (Mo // Pn) * (kbox_out + Mo))

    target = int(target_bytes)
    nc_range = [int(n_c)] if n_c is not None else range(1, nr + 1)
    for nc in nc_range:
        if resident(nc, 1) + max(expand(1, nc, 1), middle(1), final(1, nc, 1)) <= target \
                or n_c is not None:
            break
    else:
        raise RuntimeError(
            f"GATE pairconv-capacity: got {resident(nr, 1) + middle(1)} B per rank at the smallest "
            f"schedule; want at most {target} B; why: the mixed-basis pair convolution keeps "
            "H_k(p, r') for one r' chunk and T_q(G, r') resident on all P ranks; fix: more ranks "
            "or more memory per device")
    cols_chunk = padded_axis(nr, Pn * nc, name="mixed-basis r' columns").carrier // Pn // nc
    if J is None:
        J = 1
        for j in range(min(int(j_cap), cols_chunk), 0, -1):
            if resident(nc, j) + max(middle(j), expand(1, nc, j), final(1, nc, j)) <= target:
                J = j
                break
    J = int(J)
    if kc is None:
        kc = max([d for d in _divisors(nk)
                  if resident(nc, J) + expand(d, nc, J) <= target] or [1])
    if qc is None:
        qc = max([d for d in _divisors(nq)
                  if resident(nc, J) + final(d, nc, J) <= target] or [1])
    if nk % int(kc) or nq % int(qc):
        raise ValueError(f"pair-conv chunks: kc={kc} must divide n_k={nk} and qc={qc} n_q={nq}")
    return PairConvChunks(n_c=int(nc), J=J, kc=int(kc), qc=int(qc),
                          n_r_carrier=carrier(nc, J), bytes_resident=resident(nc, J),
                          bytes_expand=expand(kc, nc, J), bytes_middle=middle(J),
                          bytes_final=final(qc, nc, J), target=target)


def _budget_target(ns: int) -> int:
    """The run's per-device budget (the minimum over processes) times the BFC utilization for n_s."""
    from common.gpu_utils import (bfc_fragmentation_target_utilization, get_device_memory_gb,
                                  minimum_process_budget_gb)
    budget = minimum_process_budget_gb(get_device_memory_gb()) * 1e9
    return int(budget * bfc_fragmentation_target_utilization(int(ns)))


# ---------------------------------------------------------------------------
# Device-side pieces (rank-local, inside the stage shard_maps)
# ---------------------------------------------------------------------------

def _grid_phase(frac, sign, fft_grid, flat_idx):
    """``exp(sign·2πi Σ_a frac[:, a]·r_a)`` for the flat box cells ``flat_idx``: ``(n, *idx.shape)``.

    ``r_a = i_a / N_a`` from the C-order cell; cells past the box read any value
    (their data is zero)."""
    n1, n2, n3 = (int(v) for v in fft_grid)
    i1, i2, i3 = flat_idx // (n2 * n3), (flat_idx // n3) % n2, flat_idx % n3
    f = jnp.asarray(frac, jnp.float64)
    r = lambda i, n: (i.reshape(-1).astype(jnp.float64) / n)[None, :]
    arg = f[:, 0, None] * r(i1, n1) + f[:, 1, None] * r(i2, n2) + f[:, 2, None] * r(i3, n3)
    return jnp.exp((sign * 2j * np.pi) * arg).reshape((f.shape[0],) + tuple(flat_idx.shape))


def _column_half(src_tile, csrc, cph, spin, *, n_s):
    """The transport's column half on ``(k, rows, γ, M, δ)``: gather the parent slots of each box
    cell (``csrc``, out of range on an empty cell reads 0), the phase ``cph``, then
    ``Σ_δ · conj U_k[β, δ]``; returns ``(k, rows, γ, β, cells)``."""
    idx = jnp.broadcast_to(csrc[:, None, None, :, None],
                           src_tile.shape[:3] + (csrc.shape[1], n_s))
    v = jnp.take_along_axis(src_tile, idx, axis=3, mode="fill", fill_value=0)
    v = v * cph[:, None, None, :, None]
    uc = jnp.conj(spin)
    out = [sum(v[..., d] * uc[:, b, d][:, None, None, None] for d in range(n_s)) for b in range(n_s)]
    return jnp.stack(out, axis=3)


def _row_half(h, lsrc, lph, spin, *, n_s):
    """The transport's row half on ``(k, M, γ, β, J)``: gather the parent slot of each box cell,
    the phase, then ``Σ_γ U_k[α, γ] ·``; returns ``(k, α, J, β, cells)``."""
    idx = jnp.broadcast_to(lsrc[:, :, None, None, None], (h.shape[0], lsrc.shape[1]) + h.shape[2:])
    v = jnp.take_along_axis(h, idx, axis=1, mode="fill", fill_value=0)
    v = v * lph[:, :, None, None, None]
    out = [sum(spin[:, a, c][:, None, None, None] * v[:, :, c] for c in range(n_s)) for a in range(n_s)]
    return jnp.transpose(jnp.stack(out, axis=1), (0, 1, 4, 3, 2))


class _GatherFFT:
    """The fallback's box transform, ``LocalFourierPlan``'s contract on its three trailing axes:
    one gather per supported input axis (off-support reads 0), one ``jnp.fft`` over the
    three axes, one take per supported output axis."""

    def __init__(self, fft_grid, *, sign, norm, in_support=None, out_support=None):
        self.sign, self.norm = int(sign), norm
        self.embed, self.take = [], []
        for a, n in enumerate(fft_grid):
            ax = a - 3
            if in_support is not None:
                idx = np.asarray(in_support[a], np.int64) % n
                pos = np.full(n, idx.size, np.int32)
                pos[idx] = np.arange(idx.size, dtype=np.int32)
                self.embed.append((ax, pos))
            if out_support is not None:
                self.take.append((ax, (np.asarray(out_support[a], np.int64) % n).astype(np.int32)))

    def __call__(self, x):
        for ax, pos in self.embed:
            x = jnp.take(x, jnp.asarray(pos), axis=ax, mode="fill", fill_value=0)
        fft = local_fftn3 if self.sign < 0 else local_ifftn3
        x = fft(x, axes=(-3, -2, -1), norm=self.norm)
        for ax, idx in self.take:
            x = jnp.take(x, jnp.asarray(idx), axis=ax)
        return x


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MixedBasisPairConvolution:
    """``plan(A, C) -> X``: one τ node of the mixed-basis pair convolution (module docstring).

    ``A``, ``C``: ``(n_parent, M, n_s, M, n_s)`` complex128 at
    ``P(None, 'x', None, 'y', None)``, the ``gw.greens_function_kernel.build_G_parents``
    layout on sphere slots (``M`` = the operand's slot carrier, ``width_carrier``).
    ``A_partner``/``C_partner``: the transposed partners an antiunitary row reads;
    ``None`` reads ``conj`` of the tile (a Green of real weights).  Returns
    ``X (n_q, M_X, M_X)`` at ``P(None, 'x', 'y')`` on the output carrier (pad rows
    and columns are zero).

    ``out`` holds the output rows (q-points, any subset of the grid, e.g. the IBZ).
    ``budget_bytes`` overrides the device budget (the planner's target);
    ``chunks = (n_c, J[, kc, qc])`` pins those counts (tests, the benchmark).
    """

    def __init__(self, mesh: Mesh, *, kgrid, fft_grid, left: PairOperand, right: PairOperand,
                 out: SphereSet, backend: str | None = None, budget_bytes: int | None = None,
                 chunks: tuple[int, int] | None = None):
        if backend not in _BACKENDS:
            raise ValueError(f"MixedBasisPairConvolution: backend must be one of {_BACKENDS}, "
                             f"got {backend!r}")
        self.mesh = mesh
        self.px, self.py = int(mesh.shape["x"]), int(mesh.shape["y"])
        self.P = self.px * self.py
        self.kgrid = tuple(int(v) for v in kgrid)
        self.fft_grid = tuple(int(v) for v in fft_grid)
        self.nk = int(np.prod(self.kgrid))
        self.nr = int(np.prod(self.fft_grid))
        self.ops = (left, right)
        self.out = out
        ns = left.transport.ns
        if right.transport.ns != ns:
            raise ValueError(f"MixedBasisPairConvolution: operand spin widths {ns} and "
                             f"{right.transport.ns} differ; the spin trace pairs them")
        self.ns = ns
        for name, op in (("left", left), ("right", right)):
            if op.sphere.n != self.nk:
                raise ValueError(f"MixedBasisPairConvolution: the {name} sphere has {op.sphere.n} "
                                 f"rows, the k-grid {self.kgrid} has {self.nk}")
            kin = np.rint(op.sphere.frac * np.asarray(self.kgrid)).astype(np.int64)
            if (np.max(np.abs(op.sphere.frac * np.asarray(self.kgrid) - kin)) > 1e-8
                    or not np.array_equal(np.ravel_multi_index(
                        (kin % np.asarray(self.kgrid)).T, self.kgrid), np.arange(self.nk))):
                raise ValueError(f"MixedBasisPairConvolution: the {name} rows are not the "
                                 "C-order k-grid the k-convolution assumes")
        if not np.allclose(left.sphere.frac, right.sphere.frac, atol=1e-12):
            raise NotImplementedError(
                "MixedBasisPairConvolution: the two operands use different k representatives; "
                "one row phase serves both on the fused load (the Σ caller, lane K2)")
        qin = np.rint(out.frac * np.asarray(self.kgrid)).astype(np.int64)
        if np.max(np.abs(out.frac * np.asarray(self.kgrid) - qin), initial=0.0) > 1e-8:
            raise ValueError("MixedBasisPairConvolution: output q-points are off the k-grid")
        kg = np.asarray(self.kgrid)
        self.q_rows = np.ravel_multi_index((qin % kg).T, self.kgrid).astype(np.int32)
        self.q_neg_rows = np.ravel_multi_index(((-qin) % kg).T, self.kgrid).astype(np.int32)
        self.nq = int(out.n)

        # ---- supports, aliasing, carriers ------------------------------------
        self.sup = tuple(op.sphere.union_support() for op in self.ops)
        self.sup_out = out.union_support()
        margin = alias_free_margin(self.fft_grid, self.sup[0], self.sup[1], self.sup_out)
        if np.any(margin < 1):
            raise ValueError(
                f"GATE pairconv-alias: got box {self.fft_grid} with margins {margin.tolist()}; want "
                "every margin ≥ 1; why: product frequencies outside the output sphere would fold "
                "onto it; fix: a larger FFT box (alias_free_margin names the deficit per axis)")
        self.kbox = tuple(tuple(len(s) for s in sup) for sup in self.sup)
        self.kbox_out = tuple(len(s) for s in self.sup_out)
        self.m_axis = tuple(padded_axis(op.sphere.width, self.P, name=f"pair-conv {nm} slots")
                            for op, nm in zip(self.ops, ("left", "right")))
        self.mo_axis = padded_axis(out.width, self.P, name="pair-conv output slots")
        self.width_carrier = tuple(a.carrier for a in self.m_axis)

        # ---- host tables -----------------------------------------------------
        self._tables = [self._operand_tables(op, sup, ax.carrier)
                        for op, sup, ax in zip(self.ops, self.sup, self.m_axis)]
        cells = out.box_cells(self.sup_out)
        ocell = np.full((self.nq, self.mo_axis.carrier), int(np.prod(self.kbox_out)), np.int32)
        ocell[:, :out.width] = np.where(cells >= 0, cells, int(np.prod(self.kbox_out)))
        self._ocell = ocell

        # ---- backend ---------------------------------------------------------
        if backend is None:
            from ffi.fft import kconv_backend
            try:
                kconv_backend(mesh)
                backend = "router"
            except RuntimeError:
                backend = "xla"
        self.backend = backend

        # ---- schedule --------------------------------------------------------
        n_par = sum(op.transport.n_parent * (2 if np.any(op.transport.anti) else 1)
                    * ax.carrier * ax.carrier * ns * ns // self.P
                    for op, ax in zip(self.ops, self.m_axis))
        target = (int(budget_bytes) if budget_bytes is not None else _budget_target(ns))
        self.chunks = plan_pair_convolution_chunks(
            n_ranks=self.P, n_k=self.nk, n_s=ns, widths=self.width_carrier,
            width_out=self.mo_axis.carrier, n_q=self.nq, n_r=self.nr,
            kboxes=[int(np.prod(k)) for k in self.kbox], kbox_out=int(np.prod(self.kbox_out)),
            n_parent_tiles=n_par, target_bytes=target,
            **dict(zip(("n_c", "J", "kc", "qc"), chunks or ())))
        c = self.chunks
        self.nr_carrier = c.n_r_carrier
        self.cols_rank = self.nr_carrier // self.P
        self.cols_chunk = self.cols_rank // c.n_c
        self.n_batch = self.cols_chunk // c.J
        self._build()

    # ------------------------------------------------------------------ tables
    def _operand_tables(self, op: PairOperand, sup, m_carrier):
        """Per full-grid k and union-box cell: the parent slot (``m_carrier`` = empty), the
        row-half phase and the column-half phase."""
        tr, sph = op.transport, op.sphere
        cells = sph.box_cells(sup)
        nbox = int(np.prod([len(s) for s in sup]))
        csrc = np.full((self.nk, nbox), m_carrier, dtype=np.int32)
        mph = np.zeros((self.nk, nbox), np.complex128)
        nph = np.zeros((self.nk, nbox), np.complex128)
        for k in range(self.nk):
            live = cells[k] >= 0
            c = cells[k][live]
            ph = tr.phase[k][live]
            csrc[k, c] = tr.src[k][live]
            mph[k, c] = np.conj(ph) if tr.anti[k] else ph
            nph[k, c] = ph if tr.anti[k] else np.conj(ph)
        return dict(row=tr.row.astype(np.int32), anti=tr.anti.astype(np.int32), spin=tr.spin,
                    csrc=csrc, mph=mph, nph=nph, n_parent=tr.n_parent,
                    has_anti=bool(np.any(tr.anti)))

    def _put(self, a):
        from lxkit import device_put_process_local
        return device_put_process_local(np.asarray(a), NamedSharding(self.mesh, P()))

    # ------------------------------------------------------------------ build
    def _plan(self, *, sign, norm, in_support=None, out_support=None):
        """One box transform over the three trailing axes: ``LocalFourierPlan`` on the router
        backend, the gather + ``jnp.fft`` composition on the fallback."""
        if self.backend == "xla":
            return _GatherFFT(self.fft_grid, sign=sign, norm=norm, in_support=in_support,
                              out_support=out_support)
        axes = (-3, -2, -1)
        return LocalFourierPlan(
            self.fft_grid, axes, sign=sign, norm=norm, mesh=self.mesh,
            in_support=None if in_support is None else dict(zip(axes, in_support)),
            out_support=None if out_support is None else dict(zip(axes, out_support)))

    def _build(self):
        mesh, ns, nk, nr, P_ = self.mesh, self.ns, self.nk, self.nr, self.P
        c = self.chunks
        fg = self.fft_grid
        kfrac = self.ops[0].sphere.frac
        self._dev = [dict((key, self._put(t[key])) for key in ("row", "anti", "spin", "csrc", "mph", "nph"))
                     for t in self._tables]
        self._dev_k = self._put(kfrac)
        self._dev_q = self._put(self.out.frac)
        self._dev_ocell = self._put(self._ocell)
        self._dev_qrows = self._put(self.q_neg_rows if self.backend == "router" else self.q_rows)

        # ---- 1: 2-D tile → slab ----------------------------------------------
        def slab(a):
            return jax.lax.all_to_all(a, "y", split_axis=1, concat_axis=3, tiled=True)
        self._slab = jax.jit(shard_map(slab, mesh=mesh, in_specs=P(None, "x", None, "y", None),
                                       out_specs=P(None, _XY, None, None, None), check_vma=False))

        # ---- 2-4: column half, p' → r', Bloch column phase, r' chunk, all-to-all
        self._expand = []
        for t, sup, ax in zip(self._tables, self.sup, self.m_axis):
            self._expand.append(self._build_expand(t, sup, ax.carrier))

        # ---- 5: the streamed middle ------------------------------------------
        same_support = all(np.array_equal(a, b) for a, b in zip(*self.sup))
        plan_row = [self._plan(sign=+1, norm="forward", in_support=s) for s in self.sup]
        plan_out = self._plan(sign=-1, norm="backward", out_support=self.sup_out)
        J, nq, nb = c.J, self.nq, self.n_batch
        kbox, kbo = self.kbox, self.kbox_out
        nbo = int(np.prod(kbo))
        cols_chunk = self.cols_chunk
        backend = self.backend
        if backend == "router":
            from ffi.fft import make_fused_conv_kplane
            ident = list(range(ns))
            kconv = make_fused_conv_kplane(mesh, self.kgrid, ns, perm_l=ident, phase_l=[1] * ns,
                                           perm_r=ident, phase_r=[1] * ns)
        else:
            kinv = lambda x: local_ifftn3(x, axes=(0, 1, 2), norm="backward")
            kfwd = lambda x: local_fftn3(x, axes=(0, 1, 2), norm="backward")
        box = jnp.arange(nr, dtype=jnp.int32)

        def middle(HA, HC, T, j, kf, qf, qrows, ocell, ra, ma, sa, rc, mc, sc_):
            F = _grid_phase(kf, +1, fg, box)                              # (N_k, N_r) e^{ik·r}
            Q = _grid_phase(qf, -1, fg, box)                              # (n_q, N_r) e^{-iq·r}

            def step(T, b):
                off = b * J
                xa = _row_half(jax.lax.dynamic_slice_in_dim(HA, off, J, axis=4), ra, ma, sa, n_s=ns)
                xc = _row_half(jax.lax.dynamic_slice_in_dim(HC, off, J, axis=4), rc, mc, sc_, n_s=ns)
                if same_support:
                    x = jnp.concatenate([xa, xc], axis=2).reshape((nk, ns, 2 * J, ns) + kbox[0])
                    D = plan_row[0](x)
                else:
                    D = jnp.concatenate([plan_row[0](xa.reshape((nk, ns, J, ns) + kbox[0])),
                                         plan_row[1](xc.reshape((nk, ns, J, ns) + kbox[1]))], axis=2)
                D = D.reshape(nk, ns, 2 * J, ns, nr)                      # (k, α, [A|C] J, β, r)
                if backend == "router":
                    U = kconv(D.reshape(nk, 1, ns, 2 * J, ns, nr), F.reshape(nk, 1, nr))
                    Y = jnp.take(U, qrows, axis=0) * (1.0 / nk)           # row −q holds q
                else:
                    a = (D * F[:, None, None, None, :]).reshape(self.kgrid + (ns, 2 * J, ns, nr))
                    aR = kinv(a).reshape(nk, ns, 2 * J, ns, nr)
                    X = sum(aR[:, s1, :J, s2] * jnp.conj(aR[:, s1, J:, s2])
                            for s1 in range(ns) for s2 in range(ns))
                    Xq = kfwd(X.reshape(self.kgrid + (J, nr))).reshape(nk, J, nr)
                    Y = jnp.take(Xq, qrows, axis=0)
                Y = (Y * Q[:, None, :]).reshape((nq, J) + fg)
                Z = plan_out(Y).reshape(nq, J, nbo)
                idx = jnp.broadcast_to(ocell[:, None, :], (nq, J, ocell.shape[1]))
                Tb = jnp.take_along_axis(Z, idx, axis=2, mode="fill", fill_value=0)
                T = jax.lax.dynamic_update_slice(T, jnp.swapaxes(Tb, 1, 2), (0, 0, j * cols_chunk + off))
                return T, None

            T, _ = jax.lax.scan(step, T, jnp.arange(nb), unroll=1)
            return T

        hspec = P(None, None, None, None, _XY)
        rep = P()
        self._middle = jax.jit(shard_map(
            middle, mesh=mesh,
            in_specs=(hspec, hspec, P(None, None, _XY)) + (rep,) * 11,
            out_specs=P(None, None, _XY), check_vma=False), donate_argnums=(2,))

        # ---- 6: T → G over P, e^{iq·r'}, r' → G', 2-D layout ----------------
        plan_col_out = self._plan(sign=+1, norm="forward", out_support=self.sup_out)
        qc = c.qc
        mo_loc = self.mo_axis.carrier // P_
        n_qc = nq // qc

        def final(T, qf, ocell):
            def step(_, i):
                Tq = jax.lax.dynamic_slice_in_dim(T, i * qc, qc, axis=0)
                Tq = jax.lax.all_to_all(Tq, _XY, split_axis=1, concat_axis=2, tiled=True)
                q_i = jax.lax.dynamic_slice_in_dim(qf, i * qc, qc, axis=0)
                Tq = Tq[..., :nr] * _grid_phase(q_i, +1, fg, box)[:, None, :]
                Z = plan_col_out(Tq.reshape((qc, mo_loc) + fg)).reshape(qc, mo_loc, nbo)
                oc = jax.lax.dynamic_slice_in_dim(ocell, i * qc, qc, axis=0)
                idx = jnp.broadcast_to(oc[:, None, :], (qc, mo_loc, oc.shape[1]))
                return None, jnp.take_along_axis(Z, idx, axis=2, mode="fill", fill_value=0)

            _, X = jax.lax.scan(step, None, jnp.arange(n_qc), unroll=1)
            X = X.reshape(nq, mo_loc, -1)
            return jax.lax.all_to_all(X, "y", split_axis=2, concat_axis=1, tiled=True)

        self._final = jax.jit(shard_map(final, mesh=mesh, in_specs=(P(None, None, _XY), rep, rep),
                                        out_specs=P(None, "x", "y"), check_vma=False))

        def zeros_T():
            return jnp.zeros((nq, self.mo_axis.carrier, self.nr_carrier), jnp.complex128)
        self._zeros_T = jax.jit(zeros_T, out_shardings=NamedSharding(mesh, P(None, None, _XY)))

    def _build_expand(self, t, sup, m_carrier):
        """Steps 2–4 for one operand: ``fn(S, St, j) -> H`` for r' chunk ``j``."""
        mesh, ns, nk, nr, P_ = self.mesh, self.ns, self.nk, self.nr, self.P
        c = self.chunks
        fg = self.fft_grid
        kc, n_kc = c.kc, self.nk // c.kc
        m_loc = m_carrier // P_
        kb = tuple(len(s) for s in sup)
        plan_col = self._plan(sign=-1, norm="backward", in_support=sup)
        cols_rank, cols_chunk = self.cols_rank, self.cols_chunk
        n_par, conj_partner = t["n_parent"], True
        o = jnp.arange(cols_chunk, dtype=jnp.int32)
        rho = jnp.arange(P_, dtype=jnp.int32)

        def expand(S, St, j, row, anti, spin, csrc, nph, kf, *, partner):
            src = jnp.concatenate([S, St], axis=0) if partner else S
            # the r' columns of chunk j that each destination rank owns (≥ N_r: pad, reads 0)
            cols = rho[:, None] * cols_rank + j * cols_chunk + o[None, :]      # (P, cols_chunk)

            def step(_, i):
                k0 = i * kc
                rows = jax.lax.dynamic_slice_in_dim(row, k0, kc)
                an = jax.lax.dynamic_slice_in_dim(anti, k0, kc)
                if partner:
                    g = jnp.take(src, rows + n_par * an, axis=0)
                else:
                    g = jnp.take(src, rows, axis=0)
                    g = jnp.where((an != 0)[:, None, None, None, None], jnp.conj(g), g)
                x = _column_half(g, jax.lax.dynamic_slice_in_dim(csrc, k0, kc),
                                 jax.lax.dynamic_slice_in_dim(nph, k0, kc),
                                 jax.lax.dynamic_slice_in_dim(spin, k0, kc), n_s=ns)
                y = plan_col(x.reshape((kc, m_loc, ns, ns) + kb)).reshape(kc, m_loc, ns, ns, nr)
                y = jnp.take(y, cols, axis=4, mode="fill", fill_value=0)       # (kc, m, γ, β, P, cols)
                y = y * _grid_phase(jax.lax.dynamic_slice_in_dim(kf, k0, kc), -1, fg,
                                    cols)[:, None, None, None]
                y = jax.lax.all_to_all(y, _XY, split_axis=4, concat_axis=1, tiled=True)
                return None, y.reshape(kc, m_carrier, ns, ns, cols_chunk)

            _, H = jax.lax.scan(step, None, jnp.arange(n_kc), unroll=1)
            return H.reshape(nk, m_carrier, ns, ns, cols_chunk)

        sspec, rep = P(None, _XY, None, None, None), P()
        out = P(None, None, None, None, _XY)
        fns = {}
        for partner in (False, True):
            f = (lambda S, St, j, *tb, _p=partner: expand(S, St, j, *tb, partner=_p))
            fns[partner] = jax.jit(shard_map(f, mesh=mesh, in_specs=(sspec, sspec) + (rep,) * 7,
                                             out_specs=out, check_vma=False))
        return fns

    # ------------------------------------------------------------------ call
    def __call__(self, A, C, *, A_partner=None, C_partner=None, timings: dict | None = None):
        import time

        def mark(name, x, t0):
            """Stage walls (``timings[name]``, summed over r' chunks; ``timings[name + '_chunks']``
            per chunk) when the caller asks for them: each stage is then fenced."""
            if timings is not None:
                jax.block_until_ready(x)
                dt = time.perf_counter() - t0
                timings[name] = timings.get(name, 0.0) + dt
                timings.setdefault(name + "_chunks", []).append(dt)
            return time.perf_counter()

        for name, x, ax in (("A", A, self.m_axis[0]), ("C", C, self.m_axis[1])):
            want = (None, ax.carrier, self.ns, ax.carrier, self.ns)
            if x.ndim != 5 or tuple(x.shape[1:]) != want[1:] or x.dtype != jnp.complex128:
                raise ValueError(f"MixedBasisPairConvolution: {name} must be complex128 "
                                 f"(n_parent, {ax.carrier}, {self.ns}, {ax.carrier}, {self.ns}); "
                                 f"got {x.shape} {x.dtype}")
        t0 = time.perf_counter()
        slabs = []
        for x, xt, t in ((A, A_partner, self._tables[0]), (C, C_partner, self._tables[1])):
            s = self._slab(x)
            st = self._slab(xt) if (xt is not None and t["has_anti"]) else None
            slabs.append((s, st))
        t0 = mark("slab", slabs, t0)
        T = self._zeros_T()
        t0 = mark("alloc", T, t0)
        d = self._dev
        for j in range(self.chunks.n_c):
            jj = jnp.asarray(j, jnp.int32)
            H = []
            for (s, st), fns, dv in zip(slabs, self._expand, d):
                H.append(fns[st is not None](s, s if st is None else st, jj, dv["row"], dv["anti"],
                                             dv["spin"], dv["csrc"], dv["nph"], self._dev_k))
            t0 = mark("expand", H, t0)
            T = self._middle(H[0], H[1], T, jj, self._dev_k, self._dev_q, self._dev_qrows,
                             self._dev_ocell, d[0]["csrc"], d[0]["mph"], d[0]["spin"],
                             d[1]["csrc"], d[1]["mph"], d[1]["spin"])
            del H
            t0 = mark("middle", T, t0)
        X = self._final(T, self._dev_q, self._dev_ocell)
        mark("final", X, t0)
        return X

    # ------------------------------------------------------------------ receipts
    def collective_census(self) -> dict:
        """Collectives in each stage's compiled HLO, lowered on abstract operands:
        ``{stage: {op: count}}`` (the structural claim: the streamed middle moves nothing)."""
        import re
        ns, nk, nq = self.ns, self.nk, self.nq
        sd = lambda shape, dt, spec: jax.ShapeDtypeStruct(shape, dt, sharding=NamedSharding(self.mesh, spec))
        rep = lambda shape, dt: sd(shape, dt, P())
        i32, c128, f64 = jnp.int32, jnp.complex128, jnp.float64
        ops = {}
        for name, (t, M) in zip(("left", "right"), zip(self._tables, self.width_carrier)):
            tile = sd((t["n_parent"], M, ns, M, ns), c128, P(None, "x", None, "y", None))
            slab = sd((t["n_parent"], M, ns, M, ns), c128, P(None, _XY, None, None, None))
            nbox = t["csrc"].shape[1]
            ops[f"slab {name}"] = self._slab.lower(tile)
            ops[f"expand {name}"] = self._expand[0 if name == "left" else 1][False].lower(
                slab, slab, rep((), i32), rep((nk,), i32), rep((nk,), i32), rep((nk, ns, ns), c128),
                rep((nk, nbox), i32), rep((nk, nbox), c128), rep((nk, 3), f64))
        H = [sd((nk, M, ns, ns, self.P * self.cols_chunk), c128, P(None, None, None, None, _XY))
             for M in self.width_carrier]
        T = sd((nq, self.mo_axis.carrier, self.nr_carrier), c128, P(None, None, _XY))
        tb = [(rep((nk, t["csrc"].shape[1]), i32), rep((nk, t["csrc"].shape[1]), c128),
               rep((nk, ns, ns), c128)) for t in self._tables]
        ops["middle"] = self._middle.lower(H[0], H[1], T, rep((), i32), rep((nk, 3), f64),
                                           rep((nq, 3), f64), rep((nq,), i32),
                                           rep((nq, self.mo_axis.carrier), i32), *tb[0], *tb[1])
        ops["final"] = self._final.lower(T, rep((nq, 3), f64), rep((nq, self.mo_axis.carrier), i32))
        pat = re.compile(r"\b(all-to-all|all-gather|all-reduce|reduce-scatter|collective-permute)"
                         r"(-start)?\(")
        out = {}
        for name, low in ops.items():
            text = low.compile().as_text()
            counts = {}
            for m in pat.finditer(text):
                counts[m.group(1)] = counts.get(m.group(1), 0) + 1
            out[name] = counts
        return out

    def describe(self) -> str:
        """The plan receipt: backend, box, carriers, schedule and the memory law."""
        c = self.chunks
        gb = lambda b: f"{b / 1e9:.3f} GB"
        return (f"[pair-conv] backend {self.backend}; P={self.P}; k-grid {self.kgrid} (N_k={self.nk}); "
                f"box {self.fft_grid} (N_r={self.nr}, carrier {self.nr_carrier}); n_s={self.ns}; "
                f"slots A/C/X {self.ops[0].sphere.width}/{self.ops[1].sphere.width}/{self.out.width} "
                f"(carriers {self.width_carrier[0]}/{self.width_carrier[1]}/{self.mo_axis.carrier}); "
                f"union boxes {self.kbox[0]}/{self.kbox[1]}/{self.kbox_out}; n_q={self.nq}\n"
                f"[pair-conv] schedule: r' chunks n_c={c.n_c}, batch J={c.J} ({self.n_batch} per chunk "
                f"per rank), k chunk {c.kc}, q chunk {c.qc}\n"
                f"[pair-conv] memory law per rank: resident {gb(c.bytes_resident)} "
                f"(H {gb(_C16 * self.nk * self.ns ** 2 * sum(self.width_carrier) * self.cols_chunk)}, "
                f"T {gb(_C16 * self.nq * self.mo_axis.carrier * self.cols_rank)}); transients expand "
                f"{gb(c.bytes_expand)}, middle {gb(c.bytes_middle)}, final {gb(c.bytes_final)}; "
                f"HWM {gb(c.hwm)} against target {gb(c.target)}")

    def strip(self, X):
        """The logical output ``(n_q, width, width)`` on the host (gathers; small outputs only)."""
        from jax.experimental import multihost_utils
        x = np.asarray(multihost_utils.process_allgather(X, tiled=True)) \
            if not X.is_fully_addressable else np.asarray(X)
        w = self.out.width
        return x[:, :w, :w]
