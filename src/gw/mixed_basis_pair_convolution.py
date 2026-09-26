"""Mixed-basis pair convolution: two compact operators on plane-wave spheres → one on a third sphere set.

The operation, with no prefactor, in its two products (``product=``)::

    'trace'   X_q(G, G')        = Σ_R Σ_{r,r'} e^{-i(q+G)·(r+R)} Σ_{αβ} A^{αβ}(r+R, r') conj C^{αβ}(r+R, r') e^{+i(q+G')·r'}
    'scalar'  X_q(G α, G' β)    = Σ_R Σ_{r,r'} e^{-i(q+G)·(r+R)} A^{αβ}(r+R, r') B(r+R, r') e^{+i(q+G')·r'}

    A(r+R, r') = N_k⁻¹ Σ_k e^{ik·R} a_k(r, r'),   a_k(r, r') = Σ_{p,p'} e^{i(k+p)·r} A_k(p α, p' β) e^{-i(k+p')·r'}

and the same for C and B (B one spin channel).  r, r' run over the N_r points of the
cell's FFT box, R over the N_k cells of the Born–von Kármán supercell, k over the
C-order k-grid, p, p' over the operand's plane-wave sphere at k, and G, G' over the
output sphere at q.  The caller applies the physical prefactor.

χ₀ in the adjoint form is the first caller (``'trace'``): A = Gc(τ), C = Gv(τ) (the
adjoint of the second propagator, so both factors share ownership and indexing), and
X_R = Σ_αβ Gc_R ⊙ conj(Gv_R) is the product the ISDF χ₀ accumulates.

Σ = −G ⊙ W is the second caller (``'scalar'``, sandbox TASTE 97): A = G(τ) at the
k-parents on the ψ sphere, B = W(τ) at the q-IBZ on the χ sphere, the output on the ψ
sphere at the k-IBZ, in the ``build_G_parents`` layout.  With G from
``build_G_parents`` (unit-norm sphere coefficients) and BGW's
``W(r, r') = (N_k Ω)⁻¹ Σ_q Σ_{GG'} e^{i(q+G)·r} W_q(G, G') e^{-i(q+G')·r'}``,
Σ_k(p α, p' β; τ) = −X/(Ω·N_r²); χ₀'s X/(Ω·N_r²) is BGW's χ_q(G, G') in the same
convention (the normalization tests hold both against band sums).

The plain product is the conjugated one on B's time-reversed image
C_k[p, p'] = conj B_{−k}[p̄, p̄'] (k + G_p = −(k̄ + G_p̄)): conj C(r+R, r') = B(r+R, r'),
so ``'scalar'`` composes B's transport with time reversal
(``SphereTransport.time_reversed``, tables only) and runs the same stages and the same
k-convolution.  Its spin blocks ride as columns of a one-channel k-convolution: B is
broadcast over the n_s² blocks of A on the compact box (both operands' tables on the union of
their supports), and one p→r call writes D = [A | B].

Schedule (one τ node; P ranks; every stage one ``shard_map`` over ``('x','y')``)::

    1  A_k̄(p, p')  2-D tile at the parents            → slab: p over all P, p' local
    2  p' → r' at the parents (sphere → box, sign −1)  H̃_k̄(p, x), every x local
    3  column unfold per child k: x = α_k(r'), × e^{-2πi k̄·(x+L)}, 𝒯, conj U
    4  all-to-all: r' over all P, every p local        H_k(p, r')   [one r' chunk]
    5  per batch J of r' columns, every k local:
         typed transport, row half; p → r (sign +1); k → R; Σ_αβ A_R conj C_R; R → q
         (one fused k-convolution); keep q in the output set; × e^{-iq·r};
         r → G (box → sphere)                        T_q(G, r'_J)
    6  all-to-all: G over all P, every r' local; × e^{+iq·r'}; r' → G';
       all-to-all to the 2-D layout                  X_q(G, G')  P(None, 'x', 'y')

The typed transport (``SphereTransport``) is ``symmetry_maps.unfold_load_tables``'
pair-transpose rule in G space: the compact tiles are unfolded, ψ never is.  Its
column half acts in steps 2–3 (every p' and every r' is local there) and its row half
inside the step-5 gather (every p is local there), so neither half moves data between
ranks; the product of the two halves is the whole action.

The expand at the k-parents (steps 2–3).  A child k of parent k̄ under the operation
{mtrx | τ} (``antiunitary`` 𝒯 = conj) reads its parent at y = mtrx·(r − τ) on both
coordinates (``isdf.zeta_mubatch.typed_child_G_tables``' r-space action); on the column,
with y = x_α + L (``symmetry_maps.centroid_source_map_and_wrap``, snapped)::

    H_k(p γ, β, r') = Σ_δ conj U_k[β, δ] · 𝒯[ e^{-2πi k̄·(x_α + L)} H̃_k̄(p γ, δ, x_α) ],
    H̃_k̄(p γ, δ, x) = Σ_p' Ŝ_k̄[p γ, p' δ] e^{-2πi Ḡ_p'·x}

where Ŝ is the parent tile (conj of the transposed partner on an antiunitary row when
partners are passed, then 𝒯 conjugates it back) and the row index p stays the parent slot
for the row half.  So p' → r' runs on the n_parent parents' own spheres, not on N_k
children's rotated ones (N_k/n_parent fewer box transforms: 4.9× at Fe 4³, 8.7× at 8³),
and each child is a column gather with one phase.  H stays at every k on the columns
the middle reads: with the wedge those are N_k·N_w pairs, the same information as the
parents' H̃ on every column (n_parent·N_r); holding H̃ instead would not shrink it.

Memory law, per rank, complex128 (``describe()`` prints it; n_A, n_C, n_X the operand
and output spin widths: n_s, n_s, 1 for ``'trace'`` and n_s, 1, n_s for ``'scalar'``)::

    H         16·N_k·(n_A²·M_A + n_C²·M_C)·N_r/(P·n_c)          expand, middle
    T         16·n_q·n_X²·M_X·N_r/P                              expand … final
    batch     16·c·N_k·J·N_r,  c ≈ 4·n_A² + 2·n_X² + 2           middle

charged per stage over the objects live in it (``PairConvChunks.stage_bytes``: the slab
tiles throughout, H with the middle's T through the expand and the middle, the middle's
T with the rebuilt T through the wedge's rebuild, the final T with X through the final
stage); the HWM is the largest stage.  No N_k·N_r² object exists.  n_c (r' chunks), J (the batch), the k chunk of steps
2–4 and the q chunk of step 6 all come from the device budget; each is one when
everything fits (sandbox TASTE 96).  The r' chunking recomputes steps 2–3 per chunk
(O(n_parent M N_r log N_r) each) and bounds H by 1/n_c.

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
spin blocks ride as axes of every stage and meet only in the step-5 product.

The r'-column wedge (``wedge=ColumnWedge``; ``None`` computes every column).  For
operands covariant under a space group (every parent invariant under its little
group, as a Green of complete multiplets is), X(gx, gx') = X(x, x') on a unitary row
and conj X(x, x') on an antiunitary one (real weights).  With ``x_μ = g(x_α + L)``
(``symmetry_maps.centroid_source_map_and_wrap``) and ``q' + G'' = ±S⁻ᵀ(q + G)``
(``isdf.zeta_mubatch.typed_child_G_tables``, q the child, q' the parent)::

    T_q(G, x_μ) = e^{-2πi q·(S⁻¹L)} · 𝒯[ e^{-2πi (q'+G'')·Sτ} T_{q'}(G'', x_α) ]     𝒯 = conj on antiunitary rows

and on a spin-carrying output (``'scalar'``, n_s > 1) the row's spin action U acts on
both spin indices, T^{αβ} = Σ_γδ U_αγ (…)^{γδ} conj U_βδ, since G(gx, gx') =
U G(x, x') U† (U T for an antiunitary row) and W(gx, gx') = W(x, x') (conj); so steps 2–5 run on one representative column per grid-point orbit (N_w ≈ N_r/|G|),
with every full-grid q as the middle's output rows (N_k·N_w q-columns, no more than
n_q·N_r), and a rank-local rebuild fills T_q(G, r') on whole orbits (the orbit-packed
``common.grouped_layout`` view of r') before step 6.  H exists only on the
representatives, so the r' chunk count and the all-to-all shrink by N_r/N_w too.
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

__all__ = ["SphereSet", "SphereTransport", "PairOperand", "ColumnWedge", "MixedBasisPairConvolution",
           "PairConvChunks", "plan_pair_convolution_chunks", "alias_free_margin"]

_XY = ("x", "y")
_C16 = 16                         # bytes per complex128 element
_BACKENDS = (None, "router", "xla")
_PRODUCTS = ("trace", "scalar")   # χ₀ = Σ_αβ A ⊙ conj C;  Σ = A^{αβ} ⊙ B (module docstring)


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

    def recentred(self) -> "SphereSet":
        """The same spheres with each row's k taken in [-½, ½) per axis (``k → k − n``,
        ``G → G + n``, n = ⌊k + ½⌋): ``k + G`` and every Bloch factor are unchanged, and the
        union of the rows' spheres is as tight as one sphere allows (a k in [0, 1) widens it
        by one plane per axis: Fe 8³ at 70 Ry, 14 against 13).  The representative is
        canonical, so two sphere sets on one grid agree row by row (±½ both map to −½)."""
        n = np.floor(self.frac + 0.5 + 1e-9).astype(np.int64)
        return SphereSet(self.gvecs + n[:, None, :], self.ngk, self.frac - n)

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

    The same action in r space, which the expand uses on the column coordinate
    (module docstring, "The expand at the k-parents"): ``parent`` the parents' own
    spheres (slot order of the tiles), and each child's spatial operation, ``rot (N_k, 3, 3)``
    BGW ``mtrx`` and ``tnp (N_k, 3)`` BGW ``tnp`` (2π·τ): the child reads its parent at
    ``y = mtrx·(r − τ)`` (``symmetry_maps.centroid_source_map_and_wrap``).
    """
    row: np.ndarray
    anti: np.ndarray
    spin: np.ndarray
    src: np.ndarray
    phase: np.ndarray
    n_parent: int
    parent: SphereSet
    rot: np.ndarray
    tnp: np.ndarray

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
        rot = np.asarray(self.rot, dtype=np.int64)
        tnp = np.asarray(self.tnp, dtype=np.float64)
        if (not isinstance(self.parent, SphereSet) or self.parent.n != int(self.n_parent)
                or rot.shape != (nk, 3, 3) or tnp.shape != (nk, 3)):
            raise ValueError(f"SphereTransport: want parent a SphereSet of {self.n_parent} rows, "
                             f"rot (N_k, 3, 3), tnp (N_k, 3); got {getattr(self.parent, 'n', None)}, "
                             f"{rot.shape}, {tnp.shape}")
        for name, val in (("row", row), ("anti", anti), ("spin", spin), ("src", src),
                          ("phase", phase), ("rot", rot), ("tnp", tnp)):
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
                   phase=np.ones((nk, w), np.complex128), n_parent=nk, parent=sphere,
                   rot=np.broadcast_to(np.eye(3, dtype=np.int64), (nk, 3, 3)),
                   tnp=np.zeros((nk, 3)))

    @classmethod
    def typed(cls, plan, *, fft_grid, parent_sphere_index, children: SphereSet,
              ns: int | None = None) -> "SphereTransport":
        """The parents' typed transport onto ``children``.

        ``plan`` carries the symmetry tables (``irr_idx``, ``sym_idx``,
        ``spin_action_full``, ``k_parent_frac``, ``n_sym_spatial``,
        ``spatial_ops``, ``translations``; a ``gw.centroid_k_unfold`` plan does);
        ``parent_sphere_index (n_parent, width)`` is the parents' slot → flat box
        cell table (``common.gvec_fft_box.build_sphere_box_index``, ``≥ N_r`` on a
        pad slot).  The slot map and phase are ``typed_child_G_tables``'.  ``ns=1`` on
        a spinor plan is a spin-scalar operand (W): the trivial representation, with
        the conj rule on antiunitary rows (``unfold_load_tables(trs_rule='conj')``).
        """
        from isdf.zeta_mubatch import typed_child_G_tables
        pslot, phase, anti = typed_child_G_tables(
            plan, fft_grid=fft_grid, sphere_par=parent_sphere_index,
            gvec_child=children.gvecs, ngk_child=children.ngk, k_child=children.frac)
        spin = np.asarray(plan.spin_action_full)
        if ns is not None and int(ns) != spin.shape[-1]:
            if int(ns) != 1:
                raise ValueError(f"SphereTransport.typed: ns={ns} differs from the plan's spin "
                                 f"width {spin.shape[-1]}; only a spin-scalar operand (ns=1) may")
            spin = np.ones((spin.shape[0], 1, 1), np.complex128)
        # the parents' spheres, exactly: each parent's Miller indices from one of its children,
        # k̄ + G = ±mtrx⁻ᵀ(k + G') (typed_child_G_tables' relation), in the tiles' slot order
        cell = np.asarray(parent_sphere_index, dtype=np.int64)
        n_par, w_par = (int(v) for v in cell.shape)
        live_par = cell < int(np.prod(fft_grid))
        kp = np.asarray(plan.k_parent_frac, np.float64)
        irr = np.asarray(plan.irr_idx, dtype=np.int64)
        op = np.asarray(plan.sym_idx, dtype=np.int64) % int(plan.n_sym_spatial)
        rot = np.asarray(plan.spatial_ops, np.int64)[op]
        mill = np.zeros((n_par, w_par, 3), np.int64)
        for p in range(n_par):
            k0 = int(np.flatnonzero(irr == p)[0])
            lk = np.arange(children.width) < children.ngk[k0]
            K = (children.frac[k0][None, :] + children.gvecs[k0][lk]) * (-1.0 if anti[k0] else 1.0)
            G = np.rint(np.linalg.solve(rot[k0].T.astype(np.float64), K.T).T - kp[p]).astype(np.int64)
            mill[p, pslot[k0][lk]] = G
            if int(lk.sum()) != int(live_par[p].sum()):
                raise ValueError(f"SphereTransport.typed: child {k0} does not cover parent {p}'s sphere")
        return cls(row=irr, anti=anti, spin=spin, src=pslot, phase=phase, n_parent=n_par,
                   parent=SphereSet(mill, live_par.sum(axis=1), kp), rot=rot,
                   tnp=np.asarray(plan.translations, np.float64)[op])

    def time_reversed(self, sphere: SphereSet) -> "SphereTransport":
        """The transport of the time-reversed operator ``C_k[p, p'] = conj B_{k̄}[p̄, p̄']``,
        ``k̄ = −k`` on the grid and ``k + G_p = −(k̄ + G_p̄)``, on the same ``sphere`` rows.

        Then ``conj C(r+R, r') = B(r+R, r')``, and the conjugated pair product ``A ⊙ conj C``
        is the plain one ``A ⊙ B``.  Tables only: row ``k̄``'s parent, its antiunitary
        flag flipped, ``conj U_k̄``, and its slot map and phase read at ``p̄``.  B's
        antiunitary rows must read ``conj`` of the tile (no transposed partner).  Refuses
        a row set not closed under ``k → −k`` or a sphere not closed under ``G → −G``.
        """
        frac, g, live = sphere.frac, sphere.gvecs, sphere.live()
        s = frac[:, None, :] + frac[None, :, :]
        match = np.all(np.abs(s - np.rint(s)) < 1e-8, axis=2)
        if not np.all(match.sum(axis=1) == 1):
            raise ValueError("SphereTransport.time_reversed: the rows are not closed under k → −k")
        kb = np.argmax(match, axis=1)
        shift = np.rint(-frac - frac[kb]).astype(np.int64)                 # −k − k̄, integer
        span = int(np.abs(g[live]).max()) + int(np.abs(shift).max()) + 1
        base = 2 * span + 1
        code = lambda m: ((m[..., 0] + span) * base + (m[..., 1] + span)) * base + (m[..., 2] + span)
        pbar = np.zeros(g.shape[:2], np.int64)
        for k in range(sphere.n):
            b = int(kb[k])
            lb, lk = np.flatnonzero(live[b]), np.flatnonzero(live[k])
            cb = code(g[b, lb])
            order = np.argsort(cb)
            want = code(shift[k][None, :] - g[k, lk])
            pos = np.minimum(np.searchsorted(cb[order], want), order.size - 1)
            if not np.array_equal(cb[order][pos], want):
                raise ValueError(f"SphereTransport.time_reversed: row {k}'s sphere is not the "
                                 f"inverse of row {b}'s (−k)")
            pbar[k, lk] = lb[order[pos]]
        src = np.where(live, np.take_along_axis(self.src[kb], pbar, axis=1), self.src)
        phase = np.where(live, np.take_along_axis(self.phase[kb], pbar, axis=1), 0.0)
        return SphereTransport(row=self.row[kb], anti=~self.anti[kb], spin=np.conj(self.spin[kb]),
                               src=src, phase=phase, n_parent=self.n_parent, parent=self.parent,
                               rot=self.rot[kb], tnp=self.tnp[kb])


@dataclasses.dataclass(frozen=True, eq=False)
class PairOperand:
    """One compact operator's basis: its sphere at every full-grid k and its transport from the parents."""
    sphere: SphereSet
    transport: SphereTransport

    def __post_init__(self):
        if self.transport.src.shape != (self.sphere.n, self.sphere.width):
            raise ValueError(f"PairOperand: transport src {self.transport.src.shape} does not "
                             f"match the sphere ({self.sphere.n}, {self.sphere.width})")


@dataclasses.dataclass(frozen=True, eq=False)
class ColumnWedge:
    """The space group on the column coordinate r' (module docstring, "The r'-column wedge").

    ``sym_matrices (n_sym, 3, 3)`` BGW ``mtrx``, ``translations (n_sym, 3)`` BGW ``tnp``
    (2π·τ), ``rows`` canonical operation rows in ``[0, 2·n_sym)`` (a row ≥ n_sym is
    antiunitary with spatial part ``row − n_sym``, ``SymMaps.operation_rows``);
    one row per spatial part is kept, the unitary one when both are present.
    ``out_full``: the output sphere at every full-grid q in C order, the middle's
    rows on the wedge.  ``spin (len(rows), n_s, n_s)``, aligned with ``rows``: each
    row's spinor action (``SymMaps.spinor_action``), needed by a spin-carrying output
    (``'scalar'`` at n_s > 1); ``None`` for a spin-traced one.  Antiunitary rows need
    operands of real weights (no partners); ``from_symmaps(..., unitary_only=True)``
    is the partner path.
    """
    sym_matrices: np.ndarray
    translations: np.ndarray
    rows: np.ndarray
    out_full: SphereSet
    spin: np.ndarray | None = None

    def __post_init__(self):
        S = np.asarray(self.sym_matrices, dtype=np.int64)
        t = np.asarray(self.translations, dtype=np.float64)
        rows = np.asarray(self.rows, dtype=np.int64).reshape(-1)
        n = int(S.shape[0]) if S.ndim == 3 else -1
        if S.ndim != 3 or S.shape[1:] != (3, 3) or t.shape != (n, 3) or rows.size < 1:
            raise ValueError(f"ColumnWedge: want sym_matrices (n, 3, 3), translations (n, 3) and "
                             f"at least one row; got {S.shape}, {t.shape}, {rows.shape}")
        if rows.min() < 0 or rows.max() >= 2 * n:
            raise ValueError(f"ColumnWedge: rows must lie in [0, {2 * n})")
        spin = None if self.spin is None else np.asarray(self.spin, dtype=np.complex128)
        if spin is not None and (spin.ndim != 3 or spin.shape[0] != rows.size
                                 or spin.shape[1] != spin.shape[2]):
            raise ValueError(f"ColumnWedge: spin must be ({rows.size}, n_s, n_s) aligned with rows; "
                             f"got {spin.shape}")
        keep = {}
        for i in sorted(range(rows.size), key=lambda i: (rows[i] % n, rows[i] >= n)):
            keep.setdefault(int(rows[i] % n), i)
        idx = sorted(keep.values(), key=lambda i: rows[i])
        object.__setattr__(self, "sym_matrices", S)
        object.__setattr__(self, "translations", t)
        object.__setattr__(self, "rows", rows[idx].astype(np.int32))
        object.__setattr__(self, "spin", None if spin is None else spin[idx])

    @property
    def n_sym(self) -> int:
        return int(self.sym_matrices.shape[0])

    @property
    def spatial(self) -> np.ndarray:
        return self.rows % self.n_sym

    @property
    def anti(self) -> np.ndarray:
        return self.rows >= self.n_sym

    @classmethod
    def from_symmaps(cls, sym, out_full: SphereSet, *, unitary_only: bool = False,
                     ns: int = 1) -> "ColumnWedge":
        """The authorized rows of a ``SymMaps`` (``active_symmetry_rows``); ``unitary_only``
        keeps the unitary ones (operands with transposed partners); ``ns > 1`` carries the
        rows' spinor actions (a spin-carrying output)."""
        S = np.asarray(sym.sym_matrices)
        rows = np.asarray(sym.active_symmetry_rows)
        if unitary_only:
            rows = rows[rows < S.shape[0]]
        spin = None if int(ns) == 1 else np.asarray(sym.spinor_action(rows, nspinor=int(ns)))
        return cls(S, np.asarray(sym.translations)[:S.shape[0]], rows, out_full, spin)


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


def screened_coulomb_cutoff_cap(fft_grid, psi: SphereSet, *, bvec, q_frac) -> float:
    """The largest χ-sphere cutoff (Ry) the box keeps alias-free, measured on these spheres.

    ``alias_free_margin`` allows the output Miller window ``[hi − N + 1, lo + N − 1]``
    per axis (``lo``/``hi`` the product's reach from the recentred ψ union support,
    umklapp included); the cap is the smallest ``|q + G|²`` (``vcoul``'s metric,
    ``(q + G)·bvec``) over the recentred ``q_frac`` rows and every G outside that
    window: a sphere strictly below it keeps every margin ≥ 1.  On a density box
    (N = 4·g_ψ + 1) it sits below 4·ecutwfc, the reach of a pair product.
    """
    fg = np.asarray(fft_grid, dtype=np.int64)
    sup = psi.recentred().union_support()
    lo_ok, hi_ok = np.empty(3, np.int64), np.empty(3, np.int64)
    for a in range(3):
        s = np.asarray(sup[a], np.int64)
        lo, hi = int(s.min() - s.max() - 1), int(s.max() - s.min() + 1)
        lo_ok[a], hi_ok[a] = hi - int(fg[a]) + 1, lo + int(fg[a]) - 1
    q = np.asarray(q_frac, np.float64)
    q = q - np.rint(q)
    span = int(np.max(np.abs(np.concatenate([lo_ok, hi_ok])))) + 2
    rng = np.arange(-span, span + 1)
    G = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    outside = np.any((G < lo_ok) | (G > hi_ok), axis=1)
    b = np.asarray(bvec, np.float64)
    e = [np.min(np.sum(((qi[None, :] + G[outside]) @ b) ** 2, axis=1)) for qi in q]
    return float(min(e))


def screened_sphere_set(*, fft_grid, psi: SphereSet, bvec, q_frac, ecutwfc: float,
                        screened_coulomb_cutoff: float | None = None) -> SphereSet:
    """The χ_q(G, G') output sphere at the ``q_frac`` rows from the deck key
    ``screened_coulomb_cutoff`` (Ry; unset = ``ecutwfc``), on the WFN's FFT box.

    Refuses ``GATE screened-coulomb-cutoff`` at or above the box's measured alias cap
    (``screened_coulomb_cutoff_cap``).  The sphere is ``common.coulomb_sphere``'s
    ``|q + G|² ≤ cutoff`` in its padded layout, with the cutoff moved to the middle of
    the gap of the |q + G|² spectrum (over the ``q_frac`` rows) it falls in: a shell
    within 1e-9 of the cutoff is kept, and every row's sphere holds whole shells, so the
    rotated images of a sphere agree on membership (the wedge's G tables need that)."""
    from common.coulomb_sphere import compute_per_q_bare_coulomb_components
    from vcoul import fft_box_miller
    cut = float(ecutwfc if screened_coulomb_cutoff is None else screened_coulomb_cutoff)
    cap = screened_coulomb_cutoff_cap(fft_grid, psi, bvec=bvec, q_frac=q_frac)
    if 0.0 < cut < cap:
        _, G = fft_box_miller(tuple(int(v) for v in fft_grid))
        b = np.asarray(bvec, np.float64)
        e = np.unique(np.concatenate([np.sum(((qi[None, :] + G) @ b) ** 2, axis=1)
                                      for qi in np.asarray(q_frac, np.float64)]))
        j = int(np.searchsorted(e, cut * (1.0 + 1e-9), side="right"))
        if 0 < j < e.size:
            cut = 0.5 * (e[j - 1] + e[j])
    if not 0.0 < cut < cap:
        raise ValueError(
            f"GATE screened-coulomb-cutoff: got {cut:g} Ry ({cut / ecutwfc:.3f}·ecutwfc); want "
            f"0 < cutoff < {cap:.4f} Ry ({cap / ecutwfc:.3f}·ecutwfc), this FFT box's measured alias "
            f"cap for {tuple(int(v) for v in fft_grid)}; why: a larger χ sphere holds G onto which "
            "frequencies of the ψ-pair product fold on this box; fix: lower screened_coulomb_cutoff")
    pkg = compute_per_q_bare_coulomb_components(fft_grid=tuple(int(v) for v in fft_grid),
                                                bvec=bvec, q_irr_frac=np.asarray(q_frac),
                                                vcoul_cutoff_ry=cut, sys_dim=3)
    return SphereSet(np.asarray(pkg["gvec_components_padded"]).transpose(0, 2, 1),
                     np.asarray(pkg["ngk_per_q"]), np.asarray(q_frac))


@dataclasses.dataclass(frozen=True)
class PairConvChunks:
    """The budget-derived schedule: r' chunks, batch width, k and q chunks, and the byte model.

    The model is per stage, over the objects live in that stage (``__call__``'s order):
    the slab tiles live throughout; H (one r' chunk) and the middle's T through the expand
    and the middle; the middle's T and the rebuilt T (wedge) through the rebuild; the T the
    final stage reads and X through the final stage.  The HWM is the largest stage."""
    n_c: int
    J: int
    kc: int
    qc: int
    n_r_carrier: int
    bytes_tiles: int
    bytes_h: int
    bytes_t_mid: int
    bytes_t_out: int
    bytes_out: int
    bytes_expand: int
    bytes_middle: int
    bytes_final: int
    target: int
    bytes_rebuild: int = 0

    @property
    def bytes_resident(self) -> int:
        """What the expand and the middle hold: the tiles, H and the middle's T."""
        return self.bytes_tiles + self.bytes_h + self.bytes_t_mid

    @property
    def stage_bytes(self) -> dict:
        t_fin = self.bytes_t_out if self.bytes_t_out else self.bytes_t_mid
        out = dict(expand=self.bytes_resident + self.bytes_expand,
                   middle=self.bytes_resident + self.bytes_middle,
                   final=self.bytes_tiles + t_fin + self.bytes_final + self.bytes_out)
        if self.bytes_t_out:
            out["rebuild"] = (self.bytes_tiles + self.bytes_t_mid + self.bytes_t_out
                              + self.bytes_rebuild)
        return out

    @property
    def hwm(self) -> int:
        return max(self.stage_bytes.values())


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def plan_pair_convolution_chunks(*, n_ranks, n_k, spins, widths, width_out, n_q, n_r,
                                 kboxes, kbox_out, n_parent_tiles, target_bytes,
                                 j_cap=64, n_c=None, J=None, kc=None, qc=None,
                                 wedge=None, kboxes_parent=None, parents_per_step=None) -> PairConvChunks:
    """The schedule for one τ node (every count one when everything fits).

    ``spins = (n_A, n_C, n_X)``: the two operands' and the output's spin widths
    (``'trace'``: n_s, n_s, 1; ``'scalar'``: n_s, 1, n_s).
    ``widths``/``kboxes``: the two operands' slot carriers and union-box cells;
    ``n_parent_tiles``: the slab copies' element count per rank (inputs held).
    ``kboxes_parent``: the operands' parent union-box cells (the expand's p'→r' input);
    ``parents_per_step(kc)``: the parents one expand step transforms for ``kc`` children
    (default ``kc``, one transform per child).
    ``n_c``/``J``/``kc``/``qc`` pin those counts (tests and the benchmark); the rest follow
    (``kc`` must divide ``n_k`` and ``qc`` must divide ``n_q``).
    ``wedge`` (the r'-column wedge; ``None``: every column) is
    ``dict(n_cols, n_q_mid, width_mid, kbox_mid, t_cols, n_rows)``: the middle's columns
    (P × the most representatives one rank owns), its output rows (every full-grid q),
    their slot carrier and union-box cells, the rebuilt T's columns per rank (whole
    orbits, padded) and the operation rows the rebuild gathers.
    Refuses ``GATE pairconv-capacity`` when even n_c at one batch column per rank
    per chunk and unit k and q chunks exceeds ``target_bytes``.
    """
    Pn, nk = int(n_ranks), int(n_k)
    na, nc_, nx = (int(v) for v in spins)
    ch = (na * na, nc_ * nc_)                   # the operands' spin blocks
    cx = nx * nx                                # the output's
    cd = na * na                                # D's columns per operand per batch column (both products)
    bcast = nx > 1                              # 'scalar': B broadcast over A's blocks on the compact box
    Mw, Mo, nq, nr = [int(w) for w in widths], int(width_out), int(n_q), int(n_r)
    wd = (dict(n_cols=nr, n_q_mid=nq, width_mid=Mo, kbox_mid=kbox_out, t_cols=0, n_rows=0)
          if wedge is None else dict(wedge))
    ncol, nqm, Mm, kbm = (int(wd[k]) for k in ("n_cols", "n_q_mid", "width_mid", "kbox_mid"))
    tcol, nrow = int(wd["t_cols"]), int(wd["n_rows"])

    def carrier(nc, j):
        return padded_axis(ncol, Pn * nc * j, name="mixed-basis r' columns").carrier

    def t_total(nc, j):             # columns of the T the final stage reads, all ranks
        return carrier(nc, j) if wedge is None else Pn * tcol

    tiles = _C16 * int(n_parent_tiles)                  # the slab copies: every stage
    x_out = _C16 * (nq * Mo * Mo * cx // Pn)            # X: the final stage
    t_out = 0 if wedge is None else _C16 * nq * Mo * cx * tcol   # the rebuilt T: rebuild, final

    def h_bytes(nc, j):             # H of one r' chunk: the expand and the middle
        cols = carrier(nc, j) // Pn
        return _C16 * nk * sum(c * m for c, m in zip(ch, Mw)) * (cols // nc)

    def t_mid(nc, j):               # the middle's T: the expand, the middle and the rebuild
        return _C16 * nqm * Mm * cx * (carrier(nc, j) // Pn)

    def rebuild(nc, j):             # one q row: the gathered operation rows (+ the spin sandwich) and the rebuilt columns
        cols = carrier(nc, j) // Pn
        return 0 if wedge is None else _C16 * cx * ((3 + bcast) * nrow * Mm * cols + 2 * Mo * tcol)

    kbp = kboxes if kboxes_parent is None else kboxes_parent
    npc_of = (lambda k: k) if parents_per_step is None else parents_per_step

    def expand(kc, nc, j):          # one step: its parents' full r' transform, the children's column unfold, the all-to-all
        cols = carrier(nc, j) // Pn // nc
        npc = int(npc_of(kc))
        return _C16 * max(npc * c * (m // Pn) * (kb + 3 * nr) + 2 * kc * c * m * cols
                          for c, m, kb in zip(ch, Mw, kbp))

    def middle(j):                  # compact gathers, the p→r outputs, D and its workspace, F, U, Y, the r→G box
        gath = sum(c * kb for c, kb in zip(ch, kboxes)) + (2 * cd * kboxes[0] if bcast else 0)
        return _C16 * (nk * j * gath + nk * j * nr * 4 * cd + nk * nr
                       + nk * j * cx * nr + 2 * nqm * j * cx * (nr + kbm + Mm))

    def final(qc, nc, j):           # the all-to-all output, the phased box, its transform and gather
        return _C16 * cx * (3 * qc * (Mo // Pn) * max(t_total(nc, j), nr)
                            + 2 * qc * (Mo // Pn) * (kbox_out + Mo))

    def hwm(nc, j, kc_, qc_):       # the largest stage (PairConvChunks.stage_bytes)
        held = tiles + h_bytes(nc, j) + t_mid(nc, j)
        stages = [held + expand(kc_, nc, j), held + middle(j),
                  tiles + (t_out or t_mid(nc, j)) + final(qc_, nc, j) + x_out]
        if wedge is not None:
            stages.append(tiles + t_mid(nc, j) + t_out + rebuild(nc, j))
        return max(stages)

    target = int(target_bytes)
    nc_range = [int(n_c)] if n_c is not None else range(1, ncol + 1)
    for nc in nc_range:
        if hwm(nc, 1, 1, 1) <= target or n_c is not None:
            break
    else:
        raise RuntimeError(
            f"GATE pairconv-capacity: got {hwm(ncol, 1, 1, 1)} B per rank at the smallest "
            f"schedule; want at most {target} B; why: the mixed-basis pair convolution keeps "
            "H_k(p, r') for one r' chunk and T_q(G, r') resident on all P ranks; fix: more ranks "
            "or more memory per device")
    cols_chunk = padded_axis(ncol, Pn * nc, name="mixed-basis r' columns").carrier // Pn // nc
    if J is None:
        J = 1
        for j in range(min(int(j_cap), cols_chunk), 0, -1):
            if hwm(nc, j, 1, 1) <= target:
                J = j
                break
    J = int(J)
    if kc is None:
        kc = max([d for d in _divisors(nk) if hwm(nc, J, d, 1) <= target] or [1])
    if qc is None:
        qc = max([d for d in _divisors(nq) if hwm(nc, J, int(kc), d) <= target] or [1])
    if nk % int(kc) or nq % int(qc):
        raise ValueError(f"pair-conv chunks: kc={kc} must divide n_k={nk} and qc={qc} n_q={nq}")
    return PairConvChunks(n_c=int(nc), J=J, kc=int(kc), qc=int(qc),
                          n_r_carrier=carrier(nc, J), bytes_tiles=tiles, bytes_h=h_bytes(nc, J),
                          bytes_t_mid=t_mid(nc, J), bytes_t_out=t_out, bytes_out=x_out,
                          bytes_expand=expand(kc, nc, J), bytes_middle=middle(J),
                          bytes_final=final(qc, nc, J), target=target,
                          bytes_rebuild=rebuild(nc, J))


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
    layout on sphere slots (``M`` = the operand's slot carrier, ``width_carrier``); a
    one-channel operand may come as ``(n_parent, M, M)`` at ``P(None, 'x', 'y')``, this
    plan's own ``'trace'`` output layout (W from χ).
    ``A_partner``/``C_partner``: the transposed partners an antiunitary row reads;
    ``None`` reads ``conj`` of the tile (a Green of real weights; W's conj rule).
    Returns, on the output carrier with zero pad rows and columns,
    ``'trace'``: ``X (n_q, M_X, M_X)`` at ``P(None, 'x', 'y')``;
    ``'scalar'``: ``X (n_q, M_X, n_s, M_X, n_s)`` at ``P(None, 'x', None, 'y', None)``
    (the ``build_G_parents`` layout: Σ at the k-IBZ from G at the k-parents and W at the
    q-IBZ, where ``C`` is W and needs no partner).

    ``product``: ``'trace'`` (χ₀) or ``'scalar'`` (Σ), the module docstring's two products.
    ``out`` holds the output rows (q-points, any subset of the grid, e.g. the IBZ).
    ``wedge``: the r'-column wedge (``ColumnWedge``); ``None`` computes every column
    (the validation arm, and the only schedule for operands that are not covariant).
    ``budget_bytes`` overrides the device budget (the planner's target);
    ``chunks = (n_c, J[, kc, qc])`` pins those counts (tests, the benchmark).
    """

    def __init__(self, mesh: Mesh, *, kgrid, fft_grid, left: PairOperand, right: PairOperand,
                 out: SphereSet, product: str = "trace", backend: str | None = None,
                 budget_bytes: int | None = None, chunks: tuple[int, int] | None = None,
                 wedge: ColumnWedge | None = None):
        if backend not in _BACKENDS:
            raise ValueError(f"MixedBasisPairConvolution: backend must be one of {_BACKENDS}, "
                             f"got {backend!r}")
        if product not in _PRODUCTS:
            raise ValueError(f"MixedBasisPairConvolution: product must be one of {_PRODUCTS}, "
                             f"got {product!r}")
        # Internally every sphere is recentred (k in [-½, ½)): the union boxes, and with
        # them the box transforms' supports, are as small as the spheres allow, and every
        # operand row takes the same (canonical) representative, so one Bloch phase serves both.
        left = PairOperand(left.sphere.recentred(), left.transport)
        right = PairOperand(right.sphere.recentred(), right.transport)
        ns, ns_r = left.transport.ns, right.transport.ns
        if product == "trace" and ns_r != ns:
            raise ValueError(f"MixedBasisPairConvolution: product 'trace' pairs equal spin widths; "
                             f"got {ns} and {ns_r}")
        if product == "scalar":
            if ns_r != 1:
                raise ValueError(f"MixedBasisPairConvolution: product 'scalar' wants a one-channel "
                                 f"right operand; got spin width {ns_r}")
            # A ⊙ B = A ⊙ conj C on B's time-reversed image (module docstring)
            right = PairOperand(right.sphere, right.transport.time_reversed(right.sphere))
        out = out.recentred()
        self.product = product
        self.spins = (ns, ns_r, 1 if product == "trace" else ns)       # (n_A, n_C, n_X)
        self.wedge = wedge
        mid = out if wedge is None else wedge.out_full.recentred()   # the middle's output rows
        self.mesh = mesh
        self.px, self.py = int(mesh.shape["x"]), int(mesh.shape["y"])
        self.P = self.px * self.py
        self.kgrid = tuple(int(v) for v in kgrid)
        self.fft_grid = tuple(int(v) for v in fft_grid)
        self.nk = int(np.prod(self.kgrid))
        self.nr = int(np.prod(self.fft_grid))
        self.ops = (left, right)
        self.out = out
        self.ns = ns
        nx = self.spins[2]
        if wedge is not None and nx > 1 and (wedge.spin is None or wedge.spin.shape[-1] != nx):
            raise ValueError(
                f"GATE pairconv-wedge-spin: got a wedge with spin "
                f"{None if wedge.spin is None else wedge.spin.shape}; want each row's ({nx}, {nx}) "
                "spinor action; why: a spin-carrying output transforms as U X U† under each row; "
                f"fix: ColumnWedge.from_symmaps(..., ns={nx})")
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
        kg = np.asarray(self.kgrid)
        for name, s in (("output", out), ("wedge output", mid)):
            qin = np.rint(s.frac * kg).astype(np.int64)
            if np.max(np.abs(s.frac * kg - qin), initial=0.0) > 1e-8:
                raise ValueError(f"MixedBasisPairConvolution: {name} q-points are off the k-grid")
        if wedge is not None and not np.array_equal(np.ravel_multi_index(
                (np.rint(mid.frac * kg).astype(np.int64) % kg).T, self.kgrid), np.arange(self.nk)):
            raise ValueError("MixedBasisPairConvolution: the wedge's out_full rows are not the "
                             "C-order k-grid")
        self.nq = int(out.n)
        self.out_mid = mid
        self.nq_mid = int(mid.n)
        qin = np.rint(mid.frac * kg).astype(np.int64)
        self.q_rows = np.ravel_multi_index((qin % kg).T, self.kgrid).astype(np.int32)
        self.q_neg_rows = np.ravel_multi_index(((-qin) % kg).T, self.kgrid).astype(np.int32)

        # ---- supports, aliasing, carriers ------------------------------------
        self.sup = tuple(op.sphere.union_support() for op in self.ops)
        self.sup_out = out.union_support()
        self.sup_mid = mid.union_support()
        margin = np.minimum(alias_free_margin(self.fft_grid, self.sup[0], self.sup[1], self.sup_out),
                            alias_free_margin(self.fft_grid, self.sup[0], self.sup[1], self.sup_mid))
        if np.any(margin < 1):
            raise ValueError(
                f"GATE pairconv-alias: got box {self.fft_grid} with margins {margin.tolist()}; want "
                "every margin ≥ 1; why: product frequencies outside the output sphere would fold "
                "onto it; fix: a larger FFT box (alias_free_margin names the deficit per axis)")
        # The boxes the operands' tables and p→r transforms use: each operand's own union support,
        # or for 'scalar' the union of both, so W's one channel is broadcast over G's n_s² blocks
        # on the compact box and one p→r call writes D = [G | W] directly (no full-box concat).
        if product == "scalar":
            u = tuple(np.union1d(a, b) for a, b in zip(*self.sup))
            self.sup_tab = (u, u)
        else:
            self.sup_tab = self.sup
        self.kbox = tuple(tuple(len(s) for s in sup) for sup in self.sup_tab)
        self.kbox_out = tuple(len(s) for s in self.sup_out)
        self.kbox_mid = tuple(len(s) for s in self.sup_mid)
        self.m_axis = tuple(padded_axis(op.sphere.width, self.P, name=f"pair-conv {nm} slots")
                            for op, nm in zip(self.ops, ("left", "right")))
        self.mo_axis = padded_axis(out.width, self.P, name="pair-conv output slots")
        self.mm_axis = padded_axis(mid.width, self.P, name="pair-conv wedge output slots")
        self.width_carrier = tuple(a.carrier for a in self.m_axis)

        # ---- host tables -----------------------------------------------------
        self._tables = [self._operand_tables(op, sup, ax.carrier)
                        for op, sup, ax in zip(self.ops, self.sup_tab, self.m_axis)]
        self._ptables = [self._parent_tables(op, ax.carrier) for op, ax in zip(self.ops, self.m_axis)]
        self._ocell = self._out_cells(out, self.sup_out, self.mo_axis.carrier)
        self._ocell_mid = self._out_cells(mid, self.sup_mid, self.mm_axis.carrier)
        self._wt = None if wedge is None else self._wedge_tables(wedge, out, mid)

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
                    * ax.carrier * ax.carrier * op.transport.ns ** 2 // self.P
                    for op, ax in zip(self.ops, self.m_axis))
        target = (int(budget_bytes) if budget_bytes is not None else _budget_target(ns))
        wt = self._wt
        self.chunks = plan_pair_convolution_chunks(
            n_ranks=self.P, n_k=self.nk, spins=self.spins, widths=self.width_carrier,
            width_out=self.mo_axis.carrier, n_q=self.nq, n_r=self.nr,
            kboxes=[int(np.prod(k)) for k in self.kbox], kbox_out=int(np.prod(self.kbox_out)),
            n_parent_tiles=n_par, target_bytes=target,
            kboxes_parent=[int(np.prod(pt["kbox"])) for pt in self._ptables],
            parents_per_step=lambda kc_: max(self._steps(pt["pi"][v], kc_)["npc"]
                                            for pt in self._ptables for v in pt["pi"]),
            wedge=None if wt is None else dict(
                n_cols=self.P * wt["n_rep_rank"], n_q_mid=self.nq_mid,
                width_mid=self.mm_axis.carrier, kbox_mid=int(np.prod(self.kbox_mid)),
                t_cols=wt["layout"].shard_size, n_rows=len(wedge.rows)),
            **dict(zip(("n_c", "J", "kc", "qc"), chunks or ())))
        c = self.chunks
        self.nr_carrier = c.n_r_carrier
        self.cols_rank = self.nr_carrier // self.P
        self.cols_chunk = self.cols_rank // c.n_c
        self.n_batch = self.cols_chunk // c.J
        # the r' column each (rank, slot) carries: the box in order, or each rank's orbit
        # representatives; a value ≥ N_r is a pad column (its H reads zero)
        if wt is None:
            self._coltab = np.arange(self.nr_carrier, dtype=np.int32).reshape(self.P, self.cols_rank)
        else:
            self._coltab = np.full((self.P, self.cols_rank), self.nr, np.int32)
            for r, reps in enumerate(wt["reps"]):
                self._coltab[r, :len(reps)] = reps
        self._build()

    # ------------------------------------------------------------------ tables
    @staticmethod
    def _out_cells(s: SphereSet, sup, carrier):
        """``(n, carrier)`` int32: each output slot's cell in the ``sup`` box (pads: past the box)."""
        nbo = int(np.prod([len(v) for v in sup]))
        cells = s.box_cells(sup)
        ocell = np.full((s.n, carrier), nbo, np.int32)
        ocell[:, :s.width] = np.where(cells >= 0, cells, nbo)
        return ocell

    def _wedge_tables(self, wedge: ColumnWedge, out: SphereSet, mid: SphereSet) -> dict:
        """The r'-orbit tables, host side; the one owner of the wedge's index algebra.

        r' side: ``centroid_source_map_and_wrap`` over every box cell gives, per row j,
        ``x_μ = g_j(x_α + L)``; ``permutation_orbit_labels`` the orbits; the lowest-index
        cell is each orbit's representative; each member takes the lowest row that
        sources it from the representative.  ``build_grouped_shard_layout`` packs whole
        orbits on one rank (LPT on member counts).  G side: ``typed_child_G_tables`` with
        the output q as children of the full-grid q' = ±S⁻ᵀq under each row.
        """
        from types import SimpleNamespace
        from common.grouped_layout import build_grouped_shard_layout
        from common.gvec_fft_box import build_sphere_box_index
        from isdf.zeta_mubatch import typed_child_G_tables
        from symmetry_maps import centroid_source_map_and_wrap, permutation_orbit_labels
        nr, P_, fg = self.nr, self.P, np.asarray(self.fft_grid)
        S = wedge.sym_matrices[wedge.spatial]
        grid = np.stack(np.unravel_index(np.arange(nr), self.fft_grid), axis=-1).astype(np.int32)
        alpha, L = centroid_source_map_and_wrap(grid, S, wedge.translations[wedge.spatial], fg)
        layout = build_grouped_shard_layout(permutation_orbit_labels(alpha), P_)
        labels = np.asarray(layout.canonical_group_id, np.int64)          # the layout's orbit ids
        n_orb = int(layout.n_groups)
        rep = np.full(n_orb, nr, np.int64)
        np.minimum.at(rep, labels, np.arange(nr))
        hit = alpha == rep[labels][None, :]                               # (n_rows, N_r)
        if not np.all(hit.any(axis=0)):
            raise AssertionError("ColumnWedge: an orbit member no row sources from its representative")
        jrow = np.argmax(hit, axis=0)                                     # lowest row per member
        Lm = L[jrow, np.arange(nr)].astype(np.float64)                    # (N_r, 3)
        reps, rep_loc = [], np.empty(n_orb, np.int64)
        for r in range(P_):
            orbs = np.flatnonzero(layout.group_owner == r)
            orbs = orbs[np.argsort(layout.group_start[orbs])]
            reps.append(rep[orbs].astype(np.int32))
            rep_loc[orbs] = np.arange(orbs.size)
        # per packed column: (row, local representative) and the lattice-wrap phase
        p2c = layout.packed_to_canonical
        live = p2c >= 0
        mu = np.where(live, p2c, 0)
        Rinv = np.rint(np.linalg.inv(S.astype(np.float64))).astype(np.int64)   # (n_rows, 3, 3)
        wrap = np.einsum("mab,mb->ma", Rinv[jrow[mu]], Lm[mu])                # S⁻¹ L per column
        phl = np.where(live[None, :], np.exp(-2j * np.pi * out.frac @ wrap.T), 0.0)
        # G side: children (q_i, row j) of parents q' = ±S⁻ᵀ q_i on the full grid
        nrow, kg = len(wedge.rows), np.asarray(self.kgrid)
        sign = np.where(wedge.anti, -1.0, 1.0)
        kbar = sign[None, :, None] * np.einsum("jba,ib->ija", np.linalg.inv(S.astype(np.float64)),
                                               out.frac)                 # S⁻ᵀ q, (n_q, n_rows, 3)
        kint = np.rint(kbar * kg)
        if np.max(np.abs(kbar * kg - kint)) > 1e-6:
            raise ValueError("ColumnWedge: a row does not map the output q onto the k-grid")
        par = np.ravel_multi_index((kint.astype(np.int64) % kg).reshape(-1, 3).T, self.kgrid)
        plan = SimpleNamespace(irr_idx=par, sym_idx=np.tile(wedge.rows, self.nq),
                               spatial_ops=wedge.sym_matrices, translations=wedge.translations,
                               n_sym_spatial=wedge.n_sym, k_parent_frac=mid.frac)
        sidx = build_sphere_box_index(mid.gvecs, self.fft_grid, mid.width, ngk_valid=mid.ngk)
        pslot, gph, _ = typed_child_G_tables(
            plan, fft_grid=self.fft_grid, sphere_par=sidx,
            gvec_child=np.repeat(out.gvecs, nrow, axis=0), ngk_child=np.repeat(out.ngk, nrow),
            k_child=np.repeat(out.frac, nrow, axis=0))
        mo, mm = self.mo_axis.carrier, self.mm_axis.carrier
        ps = np.full((self.nq * nrow, mo), mm, np.int32)
        ps[:, :out.width] = np.where(pslot < mid.width, pslot, mm)
        ph = np.zeros((self.nq * nrow, mo), np.complex128)
        ph[:, :out.width] = gph
        return dict(layout=layout, reps=reps, rep_loc=rep_loc, labels=labels, jrow=jrow,
                    n_orbits=n_orb, n_rep_rank=max(len(r) for r in reps), phl=phl,
                    par=par.reshape(self.nq, nrow).astype(np.int32),
                    pslot=ps.reshape(self.nq, nrow, mo), gph=ph.reshape(self.nq, nrow, mo),
                    anti=wedge.anti.copy())

    def _operand_tables(self, op: PairOperand, sup, m_carrier):
        """Per full-grid k and union-box cell: the parent slot (``m_carrier`` = empty), the
        row-half phase and the column-half phase."""
        tr, sph = op.transport, op.sphere
        cells = sph.box_cells(sup)
        nbox = int(np.prod([len(s) for s in sup]))
        csrc = np.full((self.nk, nbox), m_carrier, dtype=np.int32)
        mph = np.zeros((self.nk, nbox), np.complex128)
        for k in range(self.nk):
            live = cells[k] >= 0
            c = cells[k][live]
            ph = tr.phase[k][live]
            csrc[k, c] = tr.src[k][live]
            mph[k, c] = np.conj(ph) if tr.anti[k] else ph
        return dict(row=tr.row.astype(np.int32), anti=tr.anti.astype(np.int32), spin=tr.spin,
                    csrc=csrc, mph=mph, n_parent=tr.n_parent,
                    has_anti=bool(np.any(tr.anti)))

    def _parent_tables(self, op: PairOperand, m_carrier):
        """The expand's host tables for one operand (module docstring, "The expand at the
        k-parents"): each parent's slot at each cell of the parents' union box, the parents'
        k̄, each child's transform-set entry ``pi`` (``row``; ``row + n_parent`` on an
        antiunitary row when partners are passed) and its operation's column map
        ``y = mtrx·(r' − τ) = x_α + L`` over every box cell
        (``symmetry_maps.centroid_source_map_and_wrap``, one row per distinct operation)."""
        from symmetry_maps import centroid_source_map_and_wrap
        tr = op.transport
        par = tr.parent.recentred()
        if par.width > m_carrier:
            raise ValueError(f"MixedBasisPairConvolution: parent spheres of width {par.width} exceed "
                             f"the operand's slot carrier {m_carrier}")
        sup = par.union_support()
        nbox = int(np.prod([len(v) for v in sup]))
        cells = par.box_cells(sup)
        pcsrc = np.full((par.n, nbox), m_carrier, np.int32)
        for p in range(par.n):
            lv = cells[p] >= 0
            pcsrc[p, cells[p][lv]] = np.flatnonzero(lv)
        key = np.concatenate([tr.rot.reshape(self.nk, 9).astype(np.float64), tr.tnp], axis=1)
        uniq, op_id = np.unique(key, axis=0, return_inverse=True)
        grid = np.stack(np.unravel_index(np.arange(self.nr), self.fft_grid), axis=-1).astype(np.int32)
        alpha, L = centroid_source_map_and_wrap(grid, uniq[:, :9].reshape(-1, 3, 3).astype(np.int64),
                                                uniq[:, 9:], np.asarray(self.fft_grid))
        row = tr.row.astype(np.int64)
        pi = {False: row}
        if np.any(tr.anti):
            pi[True] = row + par.n * tr.anti.astype(np.int64)
        return dict(sup=sup, kbox=tuple(len(v) for v in sup), pcsrc=pcsrc, kbar=par.frac,
                    op=op_id.reshape(-1).astype(np.int32), alpha=np.asarray(alpha, np.int32),
                    L=np.asarray(L, np.float64), pi=pi, n_parent=par.n)

    def _steps(self, pi, kc):
        """The expand's steps for ``kc`` children each: the children in transform-set order
        (``pi``, then k), cut into steps of ``kc``; a step transforms the ``npc`` consecutive
        entries from ``start`` (``npc`` the widest step's span, so every step has one shape)."""
        nk = self.nk
        order = np.lexsort((np.arange(nk), pi)).reshape(nk // kc, kc)
        lo, hi = pi[order[:, 0]], pi[order[:, -1]]
        npc = int(np.max(hi - lo + 1))
        n_src = int(pi.max()) + 1
        start = np.minimum(lo, max(n_src - npc, 0)).astype(np.int32)
        npc = min(npc, n_src)
        return dict(npc=npc, start=start, order=order.astype(np.int32),
                    lpar=(pi[order] - start[:, None]).astype(np.int32),
                    n_transforms=npc * (nk // kc), n_src=n_src)

    def _put(self, a, spec=P()):
        from lxkit import device_put_process_local
        return device_put_process_local(np.asarray(a), NamedSharding(self.mesh, spec))

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
        self._dev = [dict((key, self._put(t[key])) for key in ("row", "anti", "spin", "csrc", "mph"))
                     for t in self._tables]
        self._dev_k = self._put(kfrac)
        self._dev_q = self._put(self.out.frac)
        self._dev_qmid = self._put(self.out_mid.frac)
        self._dev_ocell = self._put(self._ocell)
        self._dev_ocell_mid = self._put(self._ocell_mid)
        self._dev_qrows = self._put(self.q_neg_rows if self.backend == "router" else self.q_rows)
        self._dev_cols = self._put(self._coltab)

        # ---- 1: 2-D tile → slab ----------------------------------------------
        def slab(a):
            return jax.lax.all_to_all(a, "y", split_axis=1, concat_axis=3, tiled=True)
        self._slab = jax.jit(shard_map(slab, mesh=mesh, in_specs=P(None, "x", None, "y", None),
                                       out_specs=P(None, _XY, None, None, None), check_vma=False))

        # ---- 2-4: p' → r' at the parents, the children's column unfold, all-to-all
        self._pstep = [{v: self._steps(pt["pi"][v], c.kc) for v in pt["pi"]} for pt in self._ptables]
        self._dev_exp = []
        for t, pt, ps in zip(self._tables, self._ptables, self._pstep):
            shared = (self._put(pt["pcsrc"]), self._put(pt["alpha"]), self._put(pt["L"]),
                      self._put(self._coltab))
            row, anti, spin = t["row"].astype(np.int64), t["anti"], np.asarray(t["spin"])
            self._dev_exp.append({v: tuple(self._put(a) for a in (
                st["start"], st["lpar"], st["order"], pt["op"][st["order"]],
                pt["kbar"][row[st["order"]]], anti[st["order"]].astype(np.int32),
                spin[st["order"]])) + shared for v, st in ps.items()})
        self._expand = [self._build_expand(t, pt, ps, ax.carrier)
                        for t, pt, ps, ax in zip(self._tables, self._ptables, self._pstep, self.m_axis)]

        # ---- 5: the streamed middle ------------------------------------------
        # The k-convolution pairs D's [A | C] columns and traces nsk spin blocks: 'trace' has
        # nsk = n_s and the J batch columns; 'scalar' has nsk = 1 and cw = n_s²·J columns
        # (α, j, β), C's one channel broadcast over A's blocks.
        _, ns_c, nx = self.spins
        scalar = self.product == "scalar"
        nsk = 1 if scalar else ns
        same_support = all(np.array_equal(a, b) for a, b in zip(*self.sup_tab))    # always for 'scalar'
        plan_row = [self._plan(sign=+1, norm="forward", in_support=s) for s in self.sup_tab]
        plan_out = self._plan(sign=-1, norm="backward", out_support=self.sup_mid)
        J, nq, nb = c.J, self.nq_mid, self.n_batch
        cw = nx * nx * J
        kbox, kbo = self.kbox, self.kbox_mid
        nbo = int(np.prod(kbo))
        cols_chunk = self.cols_chunk
        backend = self.backend
        if backend == "router":
            from ffi.fft import make_fused_conv_kplane
            ident = list(range(nsk))
            kconv = make_fused_conv_kplane(mesh, self.kgrid, nsk, perm_l=ident, phase_l=[1] * nsk,
                                           perm_r=ident, phase_r=[1] * nsk)
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
                xc = _row_half(jax.lax.dynamic_slice_in_dim(HC, off, J, axis=4), rc, mc, sc_, n_s=ns_c)
                if scalar:                  # W broadcast on the compact box, one p→r call
                    xc = jnp.broadcast_to(xc, xa.shape)
                    x = jnp.concatenate([xa.reshape((nk, cw, -1)), xc.reshape((nk, cw, -1))], axis=1)
                    D = plan_row[0](x.reshape((nk, 2 * cw) + kbox[0]))
                elif same_support:
                    x = jnp.concatenate([xa, xc], axis=2).reshape((nk, ns, 2 * J, ns) + kbox[0])
                    D = plan_row[0](x)
                else:
                    D = jnp.concatenate([plan_row[0](xa.reshape((nk, ns, J, ns) + kbox[0])),
                                         plan_row[1](xc.reshape((nk, ns, J, ns) + kbox[1]))], axis=2)
                D = D.reshape(nk, nsk, 2 * cw, nsk, nr)                   # (k, α, [A|C] cw, β, r)
                if backend == "router":
                    U = kconv(D.reshape(nk, 1, nsk, 2 * cw, nsk, nr), F.reshape(nk, 1, nr))
                    Y = jnp.take(U, qrows, axis=0) * (1.0 / nk)           # row −q holds q
                else:
                    a = (D * F[:, None, None, None, :]).reshape(self.kgrid + (nsk, 2 * cw, nsk, nr))
                    aR = kinv(a).reshape(nk, nsk, 2 * cw, nsk, nr)
                    X = sum(aR[:, s1, :cw, s2] * jnp.conj(aR[:, s1, cw:, s2])
                            for s1 in range(nsk) for s2 in range(nsk))
                    Xq = kfwd(X.reshape(self.kgrid + (cw, nr))).reshape(nk, cw, nr)
                    Y = jnp.take(Xq, qrows, axis=0)
                Y = (Y * Q[:, None, :]).reshape((nq, cw) + fg)
                Z = plan_out(Y).reshape(nq, cw, nbo)
                idx = jnp.broadcast_to(ocell[:, None, :], (nq, cw, ocell.shape[1]))
                Tb = jnp.take_along_axis(Z, idx, axis=2, mode="fill", fill_value=0)
                Tb = jnp.transpose(Tb.reshape(nq, nx, J, nx, -1), (0, 4, 1, 3, 2))   # (q, M, α, β, j)
                T = jax.lax.dynamic_update_slice(T, Tb, (0, 0, 0, 0, j * cols_chunk + off))
                return T, None

            T, _ = jax.lax.scan(step, T, jnp.arange(nb), unroll=1)
            return T

        hspec = P(None, None, None, None, _XY)
        rep = P()
        self._middle = jax.jit(shard_map(
            middle, mesh=mesh,
            in_specs=(hspec, hspec, hspec) + (rep,) * 11,
            out_specs=hspec, check_vma=False), donate_argnums=(2,))

        # ---- 6: T → G over P, e^{iq·r'}, r' → G', 2-D layout ----------------
        plan_col_out = self._plan(sign=+1, norm="forward", out_support=self.sup_out)
        qc = c.qc
        mo_loc = self.mo_axis.carrier // P_
        nq_o, nbo_o = self.nq, int(np.prod(self.kbox_out))      # the output rows and their box
        n_qc = nq_o // qc
        wedge = self._wt is not None

        def final(T, qf, ocell, c2p):
            def step(_, i):
                Tq = jax.lax.dynamic_slice_in_dim(T, i * qc, qc, axis=0)
                Tq = jax.lax.all_to_all(Tq, _XY, split_axis=1, concat_axis=4, tiled=True)
                q_i = jax.lax.dynamic_slice_in_dim(qf, i * qc, qc, axis=0)
                # the columns in box order: a slice, or the orbit-packed view's gather
                Tq = jnp.take(Tq, c2p, axis=4) if wedge else Tq[..., :nr]
                Tq = Tq * _grid_phase(q_i, +1, fg, box)[:, None, None, None, :]
                Z = plan_col_out(Tq.reshape((qc, mo_loc, nx, nx) + fg)).reshape(qc, mo_loc, nx, nx, nbo_o)
                oc = jax.lax.dynamic_slice_in_dim(ocell, i * qc, qc, axis=0)
                idx = jnp.broadcast_to(oc[:, None, None, None, :], (qc, mo_loc, nx, nx, oc.shape[1]))
                return None, jnp.take_along_axis(Z, idx, axis=4, mode="fill", fill_value=0)

            _, X = jax.lax.scan(step, None, jnp.arange(n_qc), unroll=1)
            X = X.reshape(nq_o, mo_loc, nx, nx, -1)
            if not scalar:                  # χ: (q, M, M)
                return jax.lax.all_to_all(X.reshape(nq_o, mo_loc, -1), "y", split_axis=2,
                                          concat_axis=1, tiled=True)
            X = jnp.transpose(X, (0, 1, 2, 4, 3))                        # Σ: (q, m, α, M, β)
            return jax.lax.all_to_all(X, "y", split_axis=3, concat_axis=1, tiled=True)

        self._out_spec = P(None, "x", "y") if not scalar else P(None, "x", None, "y", None)
        self._final = jax.jit(shard_map(final, mesh=mesh, in_specs=(hspec, rep, rep, rep),
                                        out_specs=self._out_spec, check_vma=False))
        self._dev_c2p = self._put(np.arange(nr, dtype=np.int32) if not wedge else
                                  self._wt["layout"].canonical_to_packed.astype(np.int32))

        def zeros_T():
            return jnp.zeros((self.nq_mid, self.mm_axis.carrier, nx, nx, self.nr_carrier), jnp.complex128)
        self._zeros_T = jax.jit(zeros_T, out_shardings=NamedSharding(mesh, hspec))

        # a one-channel operand in the 3-D (n, M, M) layout → the 5-D slab input
        self._as5 = jax.jit(lambda x: x[:, :, None, :, None],
                            out_shardings=NamedSharding(mesh, P(None, "x", None, "y", None)))
        if wedge:
            self._build_rebuild()

    def _build_rebuild(self):
        """The wedge's rank-local rebuild: T_w(q', G'', rep) on every full-grid q' →
        T_q(G, r') on whole orbits at the output q (module docstring)."""
        wt, mesh = self._wt, self.mesh
        nq, nrow, mo = self.nq, len(self.wedge.rows), self.mo_axis.carrier
        nx = self.spins[2]
        R = self.cols_rank
        lay = wt["layout"]
        live = lay.packed_to_canonical >= 0
        mu = np.where(live, lay.packed_to_canonical, 0)
        sel = np.where(live, wt["jrow"][mu] * R + wt["rep_loc"][wt["labels"][mu]], 0).astype(np.int32)
        spin = (np.ones((nrow, 1, 1), np.complex128) if nx == 1
                else np.asarray(self.wedge.spin, np.complex128))
        self._dev_rebuild = (self._put(wt["par"]), self._put(wt["pslot"]), self._put(wt["gph"]),
                             self._put(wt["anti"]), self._put(spin),
                             self._put(sel, P(_XY)),
                             self._put(wt["phl"], P(None, _XY)))

        def rebuild(Tm, par, pslot, gph, anti, U, sel, phl):
            def step(_, i):
                g = jnp.take(Tm, par[i], axis=0)                          # (rows, M_mid, α, β, R)
                idx = jnp.broadcast_to(pslot[i][:, :, None, None, None], (nrow, mo, nx, nx, R))
                g = (jnp.take_along_axis(g, idx, axis=1, mode="fill", fill_value=0)
                     * gph[i][:, :, None, None, None])
                g = jnp.where(anti[:, None, None, None, None], jnp.conj(g), g)
                if nx > 1:                  # U (…) U† per row, elementwise (QUALITY_PATTERNS §11)
                    u = lambda a, c: U[:, a, c][:, None, None, None]
                    g = jnp.stack([sum(u(a, c) * g[:, :, c] for c in range(nx)) for a in range(nx)], 2)
                    g = jnp.stack([sum(g[:, :, :, d] * jnp.conj(u(b, d)) for d in range(nx))
                                   for b in range(nx)], 3)
                flat = jnp.transpose(g, (1, 2, 3, 0, 4)).reshape(mo, nx, nx, nrow * R)
                return None, jnp.take(flat, sel, axis=3) * phl[i][None, None, None, :]
            _, T = jax.lax.scan(step, None, jnp.arange(nq), unroll=1)
            return T

        rep = P()
        hspec = P(None, None, None, None, _XY)
        self._rebuild = jax.jit(shard_map(
            rebuild, mesh=mesh, in_specs=(hspec, rep, rep, rep, rep, rep, P(_XY), P(None, _XY)),
            out_specs=hspec, check_vma=False))

    def _build_expand(self, t, pt, psteps, m_carrier):
        """Steps 2–4 for one operand at the k-parents: ``fns[partner](S, St, j, *tabs) -> H`` for
        r' chunk ``j`` (module docstring, "The expand at the k-parents"), with ``tabs`` its
        device tables (``self._dev_exp``)."""
        mesh, nk, nr, P_ = self.mesh, self.nk, self.nr, self.P
        ns = int(np.asarray(t["spin"]).shape[-1])            # this operand's spin width
        kc = self.chunks.kc
        fg = self.fft_grid
        m_loc = m_carrier // P_
        kbp = pt["kbox"]
        plan_par = self._plan(sign=-1, norm="backward", in_support=pt["sup"])
        cols_chunk = self.cols_chunk
        n_par = pt["n_parent"]
        nfg = jnp.asarray(fg, jnp.float64)

        def expand(S, St, j, start, lpar, kidx, cop, ckbar, canti, cspin, pcsrc, alpha, L, coltab,
                   *, partner, npc):
            # the transform set: the parent tiles, and with partners conj of the transposed
            # partners, which an antiunitary child reads before its own conjugation
            src = jnp.concatenate([S, jnp.conj(St)], axis=0) if partner else S
            pcs = jnp.concatenate([pcsrc, pcsrc], axis=0) if partner else pcsrc
            cols = jax.lax.dynamic_slice_in_dim(coltab, j * cols_chunk, cols_chunk, axis=1).reshape(-1)
            live = cols < nr                                  # a pad column (≥ N_r) reads 0
            colc = jnp.where(live, cols, 0)
            nc_all = cols.shape[0]                            # P × the chunk's columns

            def step(H, i):
                st = start[i]
                g = jax.lax.dynamic_slice_in_dim(src, st, npc, axis=0)          # (npc, m, γ, M, δ)
                cs = jax.lax.dynamic_slice_in_dim(pcs, st, npc, axis=0)         # (npc, cells)
                idx = jnp.broadcast_to(cs[:, None, None, :, None], g.shape[:3] + (cs.shape[1], ns))
                x = jnp.take_along_axis(g, idx, axis=3, mode="fill", fill_value=0)
                x = jnp.transpose(x, (1, 2, 4, 0, 3)).reshape((m_loc, ns, ns, npc) + kbp)
                h = plan_par(x).reshape(m_loc, ns, ns, npc * nr)                 # H̃_π̄(p γ, δ, x)
                # the step's children: y = x_α + L on each destination column, phase e^{-2πi k̄·y}
                op2 = cop[i][:, None] * nr + colc[None, :]                      # (kc, cols)
                a = jnp.take(alpha.reshape(-1), op2)
                Lg = jnp.take(L.reshape(-1, 3), op2, axis=0)
                ia = jnp.stack([a // (fg[1] * fg[2]), (a // fg[2]) % fg[1], a % fg[2]], axis=-1)
                y = ia.astype(jnp.float64) / nfg + Lg
                ph = jnp.exp(-2j * np.pi * jnp.sum(ckbar[i][:, None, :] * y, axis=-1))
                flat = jnp.where(live[None, :], lpar[i][:, None] * nr + a, npc * nr)
                v = jnp.take(h, flat.reshape(-1), axis=3, mode="fill", fill_value=0)
                v = v.reshape(m_loc, ns, ns, kc, nc_all) * ph[None, None, None]
                v = jnp.where((canti[i] != 0)[None, None, None, :, None], jnp.conj(v), v)
                uc = jnp.conj(cspin[i])                                         # Σ_δ · conj U[β, δ]
                v = jnp.stack([sum(v[:, :, d] * uc[:, b, d][None, None, :, None] for d in range(ns))
                               for b in range(ns)], axis=2)                     # (m, γ, β, kc, cols)
                v = jnp.transpose(v, (3, 0, 1, 2, 4)).reshape(kc, m_loc, ns, ns, P_, cols_chunk)
                v = jax.lax.all_to_all(v, _XY, split_axis=4, concat_axis=1, tiled=True)
                H = H.at[kidx[i]].set(v.reshape(kc, m_carrier, ns, ns, cols_chunk),
                                      unique_indices=True)
                return H, None

            H0 = jnp.zeros((nk, m_carrier, ns, ns, cols_chunk), jnp.complex128)
            H, _ = jax.lax.scan(step, H0, jnp.arange(start.shape[0]), unroll=1)
            return H

        sspec, rep = P(None, _XY, None, None, None), P()
        out = P(None, None, None, None, _XY)
        fns = {}
        for partner, steps in psteps.items():
            f = (lambda S, St, j, *tb, _p=partner, _n=steps["npc"]:
                 expand(S, St, j, *tb, partner=_p, npc=_n))
            fns[partner] = jax.jit(shard_map(f, mesh=mesh, in_specs=(sspec, sspec) + (rep,) * 12,
                                             out_specs=out, check_vma=False))
        return fns

    # ------------------------------------------------------------------ call
    def __call__(self, A, C, *, A_partner=None, C_partner=None, timings: dict | None = None):
        import time

        def mark(name, x, t0):
            """Stage walls (``timings[name]``, summed over r' chunks; ``timings[name + '_chunks']``
            per chunk) when the caller asks for them: each stage is then fenced.  With the walls,
            this rank's device peak so far after each stage (``timings[name + '_peak_chunks']``,
            bytes; the running maximum, so a stage's own peak is where it first rises)."""
            if timings is not None:
                jax.block_until_ready(x)
                dt = time.perf_counter() - t0
                timings[name] = timings.get(name, 0.0) + dt
                timings.setdefault(name + "_chunks", []).append(dt)
                stats = jax.local_devices()[0].memory_stats() or {}
                timings.setdefault(name + "_peak_chunks", []).append(int(stats.get("peak_bytes_in_use", 0)))
            return time.perf_counter()

        ops_in = []
        for name, x, ax, t in (("A", A, self.m_axis[0], self._tables[0]),
                               ("C", C, self.m_axis[1], self._tables[1])):
            n_s = int(np.asarray(t["spin"]).shape[-1])
            if n_s == 1 and x.ndim == 3:
                x = self._as5(x)                    # (n, M, M) at P(None, 'x', 'y'), e.g. W
            if x.ndim != 5 or tuple(x.shape[1:]) != (ax.carrier, n_s, ax.carrier, n_s) \
                    or x.dtype != jnp.complex128:
                raise ValueError(f"MixedBasisPairConvolution: {name} must be complex128 "
                                 f"(n_parent, {ax.carrier}, {n_s}, {ax.carrier}, {n_s}); "
                                 f"got {x.shape} {x.dtype}")
            ops_in.append(x)
        A, C = ops_in
        if self.product == "scalar" and C_partner is not None:
            raise ValueError(
                "GATE pairconv-scalar-partner: got a transposed partner for the one-channel operand; "
                "want W on the conj rule (no partner); why: the plain product reads W's time-reversed "
                "image through its transport, which conjugates rather than transposes; fix: pass "
                "C_partner=None")
        if self.wedge is not None and bool(np.any(self.wedge.anti)) \
                and (A_partner is not None or C_partner is not None):
            raise ValueError(
                "GATE pairconv-wedge-partner: got transposed partners with antiunitary wedge rows; "
                "want real weights (no partners) or a unitary-only wedge; why: on an antiunitary "
                "row X(gx, gx') is conj of the partners' product, not of X; fix: "
                "ColumnWedge.from_symmaps(..., unitary_only=True)")
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
            for (s, st), fns, de in zip(slabs, self._expand, self._dev_exp):
                v = st is not None
                H.append(fns[v](s, s if st is None else st, jj, *de[v]))
            t0 = mark("expand", H, t0)
            T = self._middle(H[0], H[1], T, jj, self._dev_k, self._dev_qmid, self._dev_qrows,
                             self._dev_ocell_mid, d[0]["csrc"], d[0]["mph"], d[0]["spin"],
                             d[1]["csrc"], d[1]["mph"], d[1]["spin"])
            del H
            t0 = mark("middle", T, t0)
        if self._wt is not None:
            T = self._rebuild(T, *self._dev_rebuild)
            t0 = mark("rebuild", T, t0)
        X = self._final(T, self._dev_q, self._dev_ocell, self._dev_c2p)
        mark("final", X, t0)
        return X

    # ------------------------------------------------------------------ receipts
    def collective_census(self) -> dict:
        """Collectives in each stage's compiled HLO, lowered on abstract operands:
        ``{stage: {op: count}}`` (the structural claim: the streamed middle moves nothing)."""
        import re
        pat = re.compile(r"\b(all-to-all|all-gather|all-reduce|reduce-scatter|collective-permute)"
                         r"(-start)?\(")
        out = {}
        for name, low in self._lowered_stages().items():
            text = low.compile().as_text()
            counts = {}
            for m in pat.finditer(text):
                counts[m.group(1)] = counts.get(m.group(1), 0) + 1
            out[name] = counts
        return out

    def compiled_memory(self) -> dict:
        """Each stage's compiled per-rank buffers, ``{stage: {argument, output, alias, temp}}``
        in bytes (``Compiled.memory_analysis``): the XLA-side check of the planner's transients."""
        out = {}
        for name, low in self._lowered_stages().items():
            m = low.compile().memory_analysis()
            out[name] = dict(argument=int(m.argument_size_in_bytes), output=int(m.output_size_in_bytes),
                             alias=int(m.alias_size_in_bytes), temp=int(m.temp_size_in_bytes))
        return out

    def _lowered_stages(self) -> dict:
        """Every stage lowered on abstract operands of this plan's shapes."""
        nk, nq, nx = self.nk, self.nq, self.spins[2]
        sd = lambda shape, dt, spec: jax.ShapeDtypeStruct(shape, dt, sharding=NamedSharding(self.mesh, spec))
        rep = lambda shape, dt: sd(shape, dt, P())
        i32, c128, f64 = jnp.int32, jnp.complex128, jnp.float64
        h5 = P(None, None, None, None, _XY)
        ops, H, tb = {}, [], []
        for name, (t, M) in zip(("left", "right"), zip(self._tables, self.width_carrier)):
            ns = int(np.asarray(t["spin"]).shape[-1])
            tile = sd((t["n_parent"], M, ns, M, ns), c128, P(None, "x", None, "y", None))
            slab = sd((t["n_parent"], M, ns, M, ns), c128, P(None, _XY, None, None, None))
            nbox = t["csrc"].shape[1]
            ops[f"slab {name}"] = self._slab.lower(tile)
            i = 0 if name == "left" else 1
            ops[f"expand {name}"] = self._expand[i][False].lower(
                slab, slab, rep((), i32), *self._dev_exp[i][False])
            H.append(sd((nk, M, ns, ns, self.P * self.cols_chunk), c128, h5))
            tb.append((rep((nk, nbox), i32), rep((nk, nbox), c128), rep((nk, ns, ns), c128)))
        nqm, mm = self.nq_mid, self.mm_axis.carrier
        Tm = sd((nqm, mm, nx, nx, self.nr_carrier), c128, h5)
        ops["middle"] = self._middle.lower(H[0], H[1], Tm, rep((), i32), rep((nk, 3), f64),
                                           rep((nqm, 3), f64), rep((nqm,), i32),
                                           rep((nqm, mm), i32), *tb[0], *tb[1])
        T = Tm
        if self._wt is not None:
            n_pad = self._wt["layout"].n_padded
            T = sd((nq, self.mo_axis.carrier, nx, nx, n_pad), c128, h5)
            ops["rebuild"] = self._rebuild.lower(
                Tm, *(rep(np.shape(a), a.dtype) for a in self._dev_rebuild[:5]),
                sd((n_pad,), i32, P(_XY)), sd((nq, n_pad), c128, P(None, _XY)))
        ops["final"] = self._final.lower(T, rep((nq, 3), f64), rep((nq, self.mo_axis.carrier), i32),
                                         rep((self.nr,), i32))
        return ops

    def describe(self) -> str:
        """The plan receipt: backend, box, carriers, schedule and the memory law."""
        c = self.chunks
        gb = lambda b: f"{b / 1e9:.3f} GB"
        return (f"[pair-conv] backend {self.backend}; P={self.P}; k-grid {self.kgrid} (N_k={self.nk}); "
                f"box {self.fft_grid} (N_r={self.nr}, carrier {self.nr_carrier}); product "
                f"{self.product!r}, spins A/C/X {'/'.join(map(str, self.spins))}; "
                f"slots A/C/X {self.ops[0].sphere.width}/{self.ops[1].sphere.width}/{self.out.width} "
                f"(carriers {self.width_carrier[0]}/{self.width_carrier[1]}/{self.mo_axis.carrier}); "
                f"union boxes {self.kbox[0]}/{self.kbox[1]}/{self.kbox_out}; n_q={self.nq}\n"
                f"[pair-conv] schedule: r' chunks n_c={c.n_c}, batch J={c.J} ({self.n_batch} per chunk "
                f"per rank), k chunk {c.kc}, q chunk {c.qc}; expand at the parents: "
                + "; ".join(f"{nm} {st['n_transforms']} p'→r' transforms per chunk for {self.nk} k "
                            f"({len(st['start'])} steps × {st['npc']} of {st['n_src']})"
                            for nm, ps in zip(("left", "right"), self._pstep)
                            for v, st in ps.items() if not v) + "\n"
                f"[pair-conv] memory law per rank: tiles {gb(c.bytes_tiles)}, H {gb(c.bytes_h)}, "
                f"middle T {gb(c.bytes_t_mid)}"
                f"{'' if self._wt is None else ', rebuilt T ' + gb(c.bytes_t_out)}, X {gb(c.bytes_out)}; "
                f"transients expand {gb(c.bytes_expand)}, middle {gb(c.bytes_middle)}, final "
                f"{gb(c.bytes_final)}{'' if self._wt is None else ', rebuild ' + gb(c.bytes_rebuild)}; "
                f"stages " + ", ".join(f"{k} {gb(v)}" for k, v in c.stage_bytes.items())
                + f"; HWM {gb(c.hwm)} against target {gb(c.target)}" + self._describe_wedge())

    def _describe_wedge(self) -> str:
        wt = self._wt
        if wt is None:
            return "\n[pair-conv] r'-column wedge: off (every column)"
        w = self.wedge
        return (f"\n[pair-conv] r'-column wedge: {len(w.rows)} rows ({int(w.anti.sum())} antiunitary); "
                f"{wt['n_orbits']} orbits of {self.nr} columns ({self.nr / wt['n_orbits']:.2f}x); "
                f"representatives per rank {wt['n_rep_rank']} (max), orbit members per rank "
                f"{wt['layout'].shard_size} (pad {wt['layout'].pad_fraction:.1%}); middle rows "
                f"{self.nq_mid} (every q), union box {self.kbox_mid}")

    def strip(self, X):
        """The logical output on the host, ``(n_q, width, width)`` or ``(n_q, width, n_s, width,
        n_s)`` (gathers; small outputs only)."""
        from jax.experimental import multihost_utils
        x = np.asarray(multihost_utils.process_allgather(X, tiled=True)) \
            if not X.is_fully_addressable else np.asarray(X)
        w = self.out.width
        return x[:, :w, :w] if x.ndim == 3 else x[:, :w, :, :w, :]
