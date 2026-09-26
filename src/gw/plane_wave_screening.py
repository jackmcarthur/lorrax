"""Screened interaction on the plane-wave response sphere: W_q(G, G') on the q wedge.

The equations, per q in the irreducible wedge and per complex frequency z::

    ε_q(G, G'; z) = δ_GG' − v_q(G) χ_q(G, G'; z)
    W_q(z) = ε_q(z)⁻¹ v_q,        W^c_q(z) = W_q(z) − v_q

in BerkeleyGW's convention: Rydberg, v_q(G) = 8π/|q+G|² (a slab multiplies the
Ismail-Beigi factor), no 1/Ω, and
W(r, r') = (N_k Ω)⁻¹ Σ_q Σ_GG' e^{i(q+G)·r} W_q(G, G') e^{-i(q+G')·r'}.

**χ from the pair convolution.**  ``gw.mixed_basis_pair_convolution`` returns the raw
sum X_q(G, G') of one τ node.  With A = Σ_c w_c |c⟩⟨c| and C = Σ_v w_v |v⟩⟨v| on
plane-wave coefficients normalized Σ_p |c(p)|² = 1 (A the conduction propagator,
w_c = e^{-E_c τ}; C the valence one, w_v = e^{+E_v τ})::

    χ_q(G, G'; τ) = −s · X_q(G, G') / (Ω · N_r²),     s = 2 / (n_spin · n_spinor)

(``chi_pair_sum_scale`` is the magnitude; the minus sign travels with the τ-rule
weights, as ``w_isdf._laplace_chi_args`` prefolds it).  The test suite pins it
against a BerkeleyGW M-matrix band sum.

**The Γ cell.**  At q = 0 the Dyson solve runs with v(G=0) = 0 (``vcoul`` zeroes
q+G = 0), which is the exact head-removed body: the G=0 row and column of χ cannot
enter it.  The head is the ISDF path's, owner for owner:

    S_eff(z) = S(z) + Y(z) · W_body(Γ, z) · Z(z)                   (the wing fold)
    vc0 = ⟨v⟩,  W_0(0, 0; z) = ⟨v / (1 − v qᵀS_eff q)⟩              (the Γ mini-BZ cell)

χ_q(0, G') → qᵀY(G'), χ_q(G, 0) → Z(G)q, χ_q(0, 0) → qᵀSq as q → 0 (Cartesian q,
BerkeleyGW χ units).  The W wings are zero: they are odd in q and average to zero
over the cell (BerkeleyGW's semiconductor rule; ISDF production).  Not carried, by
ruling (the ISDF path omits them too): the O(q⁰) wing and the rank-3 body term
⟨(W_body Z q)(qᵀY W_body) v/ε(q)⟩.

**Symmetry.**  W is held on the q wedge; nothing here unfolds.  At fixed τ the
antiunitary rows read conj W (no partner); at a fixed complex z they read
conj W(z̄), equivalently the transposed partner at −z̄.

Owners composed (nothing is rebuilt here): ``gw.compute_vcoul.compute_v_q_per_G``
and the ``vcoul`` kernel's ``q0_average``, ``gw.mixed_basis_pair_convolution``
(the sphere and its alias cap), ``gw.w_isdf.solve_w`` (the Dyson solve; ``linalg``
local or distributed through ``distrib_la``), ``gw.head_correction``'s
``fold_small_head_wings_sharded``, and ``gw.mpa.pade_fit.fit_mpa_poles_batched``.

Memory per rank, complex128, one stack = 16·n_q·M²/P (M the sphere carrier): the Dyson
stage holds the χ and W samples (2·n_z stacks) and one sample's workspace (local
16·5·M², distributed 16·4·n_q·M²/P); the fit stage holds the W^c samples and the poles
(n_z + 3·n_p stacks) and the fit transient of ``fit_q_batch`` q rows (about 8.5 kB per
element at n_p = 8, read from the compiled executable).  ``plan_q_chunks`` derives the
wedge-q chunk count from the device budget; ``describe()`` prints the law.
"""
from __future__ import annotations

import dataclasses
from functools import lru_cache, partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.shard_map import shard_map
from ffi import _services
from runtime.padding import PaddedAxis, padded_axis

_services.ensure_on_path()

import vcoul                                                        # noqa: E402

from .mixed_basis_pair_convolution import SphereSet, screened_sphere_set   # noqa: E402

__all__ = ["ResponseSpheres", "response_spheres", "chi_pair_sum_scale", "accumulate_chi",
           "sphere_coulomb", "SphereScreening"]

_C16 = 16
_SYS_DIMS = (2, 3)


# ---------------------------------------------------------------------------
# The response sphere, resolved once on the full grid
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, eq=False)
class ResponseSpheres:
    """The χ/W sphere at every full-grid q (``full``, C order) and at the wedge (``irr``).

    Both come from one ``screened_coulomb_cutoff`` resolution; ``irr_rows[i]`` is the
    full-grid row of wedge row ``i``.  Slot 0 is G = 0 at every row."""
    full: SphereSet
    irr: SphereSet
    irr_rows: np.ndarray


def _grid_rows(frac, kgrid) -> np.ndarray:
    kg = np.asarray(kgrid, dtype=np.int64)
    n = np.rint(np.asarray(frac, np.float64) * kg).astype(np.int64)
    if np.max(np.abs(np.asarray(frac) * kg - n), initial=0.0) > 1e-8:
        raise ValueError("response_spheres: a q-point is off the k-grid")
    return np.ravel_multi_index((n % kg).T, tuple(kg)).astype(np.int32)


def response_spheres(*, fft_grid, psi: SphereSet, bvec, kgrid, q_irr_frac, ecutwfc: float,
                     screened_coulomb_cutoff: float | None = None) -> ResponseSpheres:
    """The response sphere from the deck key ``screened_coulomb_cutoff`` (Ry; unset = ecutwfc).

    ``screened_sphere_set`` moves the cutoff to the middle of the |q+G|² gap it falls
    in over the rows it is given, so the full grid and the wedge are both built from
    it and their row sizes must agree (a symmetry maps each full row onto its wedge
    representative's spectrum); a mismatch refuses."""
    kg = tuple(int(v) for v in kgrid)
    ii = np.stack(np.meshgrid(*(np.arange(n) for n in kg), indexing="ij"), -1).reshape(-1, 3)
    q_full = ii / np.asarray(kg, np.float64)
    kw = dict(fft_grid=fft_grid, psi=psi, bvec=bvec, ecutwfc=ecutwfc,
              screened_coulomb_cutoff=screened_coulomb_cutoff)
    full = screened_sphere_set(q_frac=q_full, **kw)
    irr = screened_sphere_set(q_frac=np.asarray(q_irr_frac, np.float64), **kw)
    rows = _grid_rows(irr.frac, kg)
    if not np.array_equal(irr.ngk, full.ngk[rows]):
        raise ValueError(
            "GATE response-sphere-rows: got wedge sphere sizes "
            f"{irr.ngk.tolist()} against the full grid's {full.ngk[rows].tolist()}; want equal; "
            "why: the cutoff resolved on the wedge landed in a different shell gap than on the "
            "full grid; fix: pass the wedge rows of this k-grid")
    return ResponseSpheres(full=full, irr=irr, irr_rows=rows)


# ---------------------------------------------------------------------------
# Factors and the bare interaction
# ---------------------------------------------------------------------------

def chi_pair_sum_scale(*, cell_volume: float, n_r: int, n_spin: int = 1, n_spinor: int = 1) -> float:
    """|χ_q| / |X_q|: s/(Ω·N_r²) with s = 2/(n_spin·n_spinor) (module docstring)."""
    return 2.0 / (int(n_spin) * int(n_spinor)) / (float(cell_volume) * float(n_r) ** 2)


@lru_cache(maxsize=None)
def _accumulate(mesh):
    spec = NamedSharding(mesh, P(None, None, "x", "y"))
    return jax.jit(lambda acc, X, w: acc + w[:, None, None, None] * X[None],
                   donate_argnums=(0,), out_shardings=spec)


def accumulate_chi(acc, X, weights, *, scale: float, mesh: Mesh):
    """χ(z_j) += scale·weights[j]·X for every sample j: one τ node of a minimax rule.

    ``acc (n_z, n_q, M, M)`` at ``P(None, None, 'x', 'y')`` (donated), ``X (n_q, M, M)``
    one raw pair sum at ``P(None, 'x', 'y')`` (the pair convolution's output; both
    particle–hole orientations are separate sums, each with its own weight row),
    ``weights (n_z,)`` the rule's α_l(z_j) (``minimax_screening``'s rules, as the ISDF
    χ₀ applies them), ``scale`` −``chi_pair_sum_scale``."""
    return _accumulate(mesh)(acc, X, np.asarray(weights, np.complex128) * float(scale))


def sphere_coulomb(sphere: SphereSet, *, geometry, sys_dim: int, carrier: int,
                   v_head_fn=None) -> np.ndarray:
    """v_q(G) on the sphere's slots, BerkeleyGW units (8π/|q+G|², no 1/Ω): ``(n_q, carrier)``.

    ``gw.compute_vcoul.compute_v_q_per_G`` evaluates it, the ISDF path's call (the
    ``vcoul`` kernel, the slab truncation for ``sys_dim = 2``, the optional q≠0
    mini-BZ body head ``v_head_fn``); its tables carry 1/Ω, which is multiplied
    back.  Pad slots and q+G = 0 hold exact zeros."""
    from .compute_vcoul import compute_v_q_per_G
    sd = int(sys_dim)
    if sd not in _SYS_DIMS:
        raise ValueError(
            f"GATE response-sphere-sys-dim: got sys_dim={sd}; want one of {_SYS_DIMS}; why: the "
            "0-D box kernel has no plane-wave sphere; fix: sys_dim 3 (bulk) or 2 (slab)")
    if int(carrier) < sphere.width:
        raise ValueError(f"sphere_coulomb: carrier {carrier} < sphere width {sphere.width}")
    table = compute_v_q_per_G(np.asarray(sphere.frac, np.float64),
                              np.asarray(sphere.gvecs).transpose(0, 2, 1),
                              bvec=np.asarray(geometry.bvec), cell_volume=float(geometry.cell_volume),
                              sys_dim=sd, v_head_fn=v_head_fn)
    v = np.zeros((sphere.n, int(carrier)), np.float64)
    v[:, :sphere.width] = np.where(sphere.live(), table * float(geometry.cell_volume), 0.0)
    return v


# ---------------------------------------------------------------------------
# The screening plan
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _diag_embed(mesh):
    out = NamedSharding(mesh, P(None, "x", "y"))
    return jax.jit(lambda v: v[..., :, None] * jnp.eye(v.shape[-1], dtype=v.dtype),
                   out_shardings=out)


@lru_cache(maxsize=None)
def _zeros_like_stack(mesh):
    return jax.jit(jnp.zeros_like, out_shardings=NamedSharding(mesh, P(None, None, "x", "y")))


@lru_cache(maxsize=None)
def _set_row(mesh):
    return jax.jit(lambda acc, x, j: jax.lax.dynamic_update_index_in_dim(acc, x, j, 0),
                   donate_argnums=(0,), out_shardings=NamedSharding(mesh, P(None, None, "x", "y")))


@lru_cache(maxsize=None)
def _minus_diag(mesh):
    """W − diag(v), and the Γ head slot set to ``head`` (per leading z)."""
    spec = NamedSharding(mesh, P(None, None, "x", "y"))

    def f(W, v, ig, head):
        Wc = W - v[None, :, :, None] * jnp.eye(W.shape[-1], dtype=W.dtype)
        if ig is None:
            return Wc
        return Wc.at[:, ig, 0, 0].set(head)
    return jax.jit(f, static_argnums=(2,), out_shardings=spec)


class SphereScreening:
    """W_q(G, G') on the q wedge from χ_q(G, G'; z) (module docstring).

    ``sphere``: the wedge rows (``ResponseSpheres.irr``); ``geometry``: a
    ``vcoul.CoulombGeometry``; ``kgrid``: the Γ-centred grid of the mini-BZ cell;
    ``linalg``: ``'local'`` or ``'distributed'`` (the deck dial, resolved by the
    caller).  Arrays ``(…, n_q, M, M)`` live at ``P(…, 'x', 'y')`` on the carrier
    ``M = self.axis.carrier`` with exact-zero pad rows and columns, K1's output layout.
    """

    def __init__(self, mesh: Mesh, *, sphere: SphereSet, geometry, sys_dim: int, kgrid,
                 linalg: str = "local", v_head_fn=None):
        if linalg not in ("local", "distributed"):
            raise ValueError(f"SphereScreening: linalg must be 'local' or 'distributed', got {linalg!r}")
        self.mesh = mesh
        self.P = int(mesh.shape["x"]) * int(mesh.shape["y"])
        self.sphere = sphere
        self.geometry = geometry
        self.sys_dim = int(sys_dim)
        self.kgrid = tuple(int(v) for v in kgrid)
        self.linalg = linalg
        self.axis: PaddedAxis = padded_axis(sphere.width, self.P, name="response sphere slots")
        self.M = int(self.axis.carrier)
        g0 = np.flatnonzero(np.all(np.abs(sphere.frac - np.rint(sphere.frac)) < 1e-12, axis=1))
        if g0.size > 1:
            raise ValueError("SphereScreening: more than one Γ row in the wedge")
        self.gamma = None if g0.size == 0 else int(g0[0])
        if self.gamma is not None and np.any(sphere.gvecs[self.gamma, 0] != 0):
            raise ValueError("SphereScreening: slot 0 at Γ is not G = 0")
        self.v = sphere_coulomb(sphere, geometry=geometry, sys_dim=self.sys_dim,
                                carrier=self.M, v_head_fn=v_head_fn)
        self._V = _diag_embed(mesh)(self.v.astype(np.complex128))
        self._q0_kernel = vcoul.get_kernel(self.sys_dim)
        self._fit_temp: dict = {}

    # ------------------------------------------------------------------ Dyson
    def solve(self, chi):
        """W_q = (I − v χ_q)⁻¹ v at every wedge q, ``chi (n_q, M, M)`` BerkeleyGW units (donated)."""
        from .w_isdf import solve_w
        return solve_w(self._V, chi, None, self.mesh, dyson_solver=self.linalg,
                       pref=1.0, axis=self.axis)

    def solve_samples(self, chi_z):
        """``solve`` at each leading z of ``chi_z (n_z, n_q, M, M)``; W stacked the same way
        (written in place: χ and W stacks plus one sample's Dyson workspace are live)."""
        n_z = int(chi_z.shape[0])
        W = _zeros_like_stack(self.mesh)(chi_z)
        for j in range(n_z):
            W = _set_row(self.mesh)(W, self.solve(chi_z[j]), j)
        return W

    # ------------------------------------------------------------------ Γ head
    def gamma_head(self, W_z, S, Y, Z):
        """(vc0, wcoul0 (n_z,), S_eff (n_z, 3, 3)) from the Γ body of ``W_z`` and the small fields.

        ``S (n_z, 3, 3)``; ``Y (n_z, 3, M)`` at ``P(None, None, 'x')`` and
        ``Z (n_z, M, 3)`` at ``P(None, 'y', None)`` (the wings, zero on pad slots)."""
        from .head_correction import fold_small_head_wings_sharded
        if self.gamma is None:
            raise ValueError("SphereScreening.gamma_head: the wedge has no Γ row")
        S_eff = fold_small_head_wings_sharded(
            np.asarray(S, np.complex128), Y, W_z[:, self.gamma], Z, 1.0, mesh_xy=self.mesh)
        S_eff = np.asarray(jax.device_get(S_eff))
        kw = dict(method="auto")
        if self.sys_dim == 2:
            kw = {}
        vc0, w0 = None, []
        for s in S_eff:
            v, w = self._q0_kernel.q0_average(self.geometry, self.kgrid, S_cart=s, **kw)
            vc0 = complex(v) if vc0 is None else vc0
            w0.append(complex(w))
        return vc0, np.asarray(w0, np.complex128), S_eff

    # ------------------------------------------------------------------ W^c
    def correlation(self, W_z, *, wcoul0=None, vc0=None):
        """W^c = W − v at every wedge q; at Γ slot (0, 0) holds wcoul0 − vc0 (wings zero).

        Returns ``(Wc_z, v)``: ``v (n_q, M)`` with vc0 in the Γ head slot."""
        v = np.array(self.v, copy=True)
        if self.gamma is None:
            return _minus_diag(self.mesh)(W_z, v, None, None), v
        if wcoul0 is None or vc0 is None:
            raise ValueError("SphereScreening.correlation: the wedge holds Γ; pass wcoul0 and vc0")
        head = np.asarray(wcoul0, np.complex128) - complex(vc0)
        Wc = _minus_diag(self.mesh)(W_z, v, self.gamma, head)
        v[self.gamma, 0] = float(np.real(vc0))
        return Wc, v

    # ------------------------------------------------------------------ poles
    def fit_poles(self, Wc_z, z_samples, n_p: int, *, Wc_negative=None, rcond: float = 1.0e-13,
                  solve: str = "loewner", budget_bytes: int | None = None):
        """The MPA model of every element, ``pade_fit.fit_mpa_poles_batched`` on local tiles.

        ``Wc_z (2n_p, n_q, M, M)`` at the samples ``z_samples (2n_p,)``
        (``mpa.sampling.double_parallel_grid``); ``Wc_negative`` the same element at
        ``−z`` (the ordered fit; returns ``B_odd``).  Returns ``(Omega, B, B_odd, cond)``:
        ``(n_p, n_q, M, M)`` at ``P(None, None, 'x', 'y')`` with zeros on pad elements
        and on the Γ wings (structural zeros, not fitted); ``B_odd`` is ``None`` for
        the even fit; ``cond`` the largest pole-fit condition number over fitted elements.

        The fit's transient is per element; the local q rows run in batches of
        ``self.fit_q_batch(...)`` (one when everything fits, TASTE 96)."""
        qb = self.fit_q_batch(n_p, ordered=Wc_negative is not None, rcond=rcond, solve=solve,
                              budget_bytes=budget_bytes)
        return _fit_tiles(self.mesh, int(n_p), float(rcond), solve, Wc_negative is not None,
                          self.gamma, qb)(
            Wc_z, Wc_z if Wc_negative is None else Wc_negative, np.asarray(z_samples, np.complex128),
            np.asarray(self.sphere.ngk, np.int32))

    def fit_q_batch(self, n_p: int, *, ordered: bool = False, rcond: float = 1.0e-13,
                    solve: str = "loewner", budget_bytes: int | None = None) -> int:
        """q rows per fit batch: half the device budget over the compiled per-q-row transient."""
        key = (int(n_p), bool(ordered), float(rcond), solve)
        if key not in self._fit_temp:
            sd = jax.ShapeDtypeStruct
            spec = NamedSharding(self.mesh, P(None, None, "x", "y"))
            w = sd((2 * int(n_p), 1, self.M, self.M), jnp.complex128, sharding=spec)
            f = _fit_tiles(self.mesh, int(n_p), float(rcond), solve, bool(ordered), None, 1)
            mem = f.lower(w, w, sd((2 * int(n_p),), jnp.complex128),
                          sd((1,), jnp.int32)).compile().memory_analysis()
            self._fit_temp[key] = max(1, int(getattr(mem, "temp_size_in_bytes", 0)))
        if budget_bytes is None:
            from common.gpu_utils import get_device_memory_gb, minimum_process_budget_gb
            budget_bytes = int(minimum_process_budget_gb(get_device_memory_gb()) * 1e9)
        n_loc = self.sphere.n
        return int(max(1, min(n_loc, (int(budget_bytes) // 2) // self._fit_temp[key])))

    def plan_q_chunks(self, n_z: int, n_p: int, *, budget_bytes: int | None = None,
                      ordered: bool = False) -> int:
        """Wedge-q chunks for the caller's χ → W → poles pass: one when everything fits.

        Per rank, for n_c wedge rows: the Dyson stage holds the χ and W samples
        (2·n_z stacks) plus one sample's workspace, and the fit stage the W^c samples
        and the poles (n_z + 3·n_p stacks) plus one q row's fit transient
        (``fit_q_batch``'s compiled figure).  Past one chunk the caller reruns the χ τ
        loop per chunk (the pair convolution's cost), so the remedy there is more ranks."""
        if budget_bytes is None:
            from common.gpu_utils import get_device_memory_gb, minimum_process_budget_gb
            budget_bytes = int(minimum_process_budget_gb(get_device_memory_gb()) * 1e9)
        self.fit_q_batch(n_p, ordered=ordered, budget_bytes=budget_bytes)
        temp_q = self._fit_temp[(int(n_p), bool(ordered), 1.0e-13, "loewner")]
        n_q, M, Pn = self.sphere.n, self.M, self.P
        row = _C16 * M * M / Pn
        work = (_C16 * 5 * M * M if self.linalg == "local" else _C16 * 4 * n_q * M * M / Pn)
        for n_c in range(1, n_q + 1):
            nq_c = -(-n_q // n_c)
            need = max(2 * int(n_z) * row * nq_c + work,
                       (int(n_z) + 3 * int(n_p)) * row * nq_c + temp_q)
            if need <= budget_bytes:
                return n_c
        raise ValueError(
            f"GATE pw-screening-budget: got a per-rank budget of {budget_bytes / 1e9:.2f} GB; "
            f"want one wedge row's samples, poles and workspace to fit; why: the response "
            f"sphere carrier M={M} at P={Pn}; fix: more ranks or a smaller screened_coulomb_cutoff")

    # ------------------------------------------------------------------ receipt
    def describe(self, n_z: int | None = None, n_p: int | None = None) -> str:
        n_q, M, Pn = self.sphere.n, self.M, self.P
        gb = lambda b: f"{b / 1e9:.3f} GB"
        per = _C16 * n_q * M * M / Pn
        s = (f"[pw-screening] n_q={n_q} (Γ row {self.gamma}); sphere width {self.sphere.width} "
             f"(ngk {int(self.sphere.ngk.min())}..{int(self.sphere.ngk.max())}), carrier M={M}, "
             f"P={Pn}; sys_dim={self.sys_dim}; linalg={self.linalg}; one (n_q, M, M) stack "
             f"{gb(per)}/rank")
        if n_z is not None and n_p is not None:
            dy = (_C16 * 5 * -(-n_q // Pn) * M * M if self.linalg == "local"
                  else _C16 * 4 * n_q * M * M / Pn)
            s += (f"; law per rank: Dyson stage χ + W samples {gb(2 * n_z * per)} + workspace "
                  f"{gb(dy)}; fit stage W^c samples {gb(n_z * per)} + poles {gb(3 * n_p * per)} "
                  f"+ the fit transient (budget-derived q batch)")
        return s


@lru_cache(maxsize=None)
def _fit_tiles(mesh, n_p, rcond, solve, ordered, gamma, qb):
    """One compiled fit of every local element, ``qb`` q rows at a time (``lax.map``)."""
    from .mpa import pade_fit
    eig = "lapack" if solve == "thiele" else "jax_qr"     # mpa.fit_driver's policy
    spec = P(None, None, "x", "y")

    def _local(w, wn, z, ngk):
        n_z, n_q, mx, my = w.shape
        rows = jax.lax.axis_index("x") * mx + jnp.arange(mx)
        cols = jax.lax.axis_index("y") * my + jnp.arange(my)

        def one_q(args):
            wq, wnq, nq, iq = args                                         # (n_z, mx, my)
            live = (rows[:, None] < nq) & (cols[None, :] < nq)
            if gamma is not None:        # the Γ wings are structural zeros, not fitted
                wing = (rows[:, None] == 0) != (cols[None, :] == 0)
                live = live & ~((iq == gamma) & wing)
            tile = jnp.transpose(wq, (1, 2, 0)).reshape(-1, n_z)
            if ordered:
                neg = jnp.transpose(wnq, (1, 2, 0)).reshape(-1, n_z)
                Om, B, D, diag = pade_fit.fit_mpa_poles_batched(
                    tile, z, n_p, W_negative_tile=neg, return_odd=True, rcond=rcond, eig=eig,
                    solve=solve)
            else:
                Om, B, diag = pade_fit.fit_mpa_poles_batched(tile, z, n_p, rcond=rcond, eig=eig,
                                                             solve=solve)
                D = jnp.zeros_like(B)
            shape = lambda a: jnp.transpose(a.reshape(mx, my, n_p), (2, 0, 1))
            Om, B, D = (jnp.where(live[None], shape(a), 0.0 + 0.0j) for a in (Om, B, D))
            cond = jnp.max(jnp.where(live, diag["cond_pade"].reshape(mx, my), 0.0))
            return Om, B, D, cond

        xs = (jnp.moveaxis(w, 1, 0), jnp.moveaxis(wn, 1, 0), ngk, jnp.arange(n_q))
        Om, B, D, cond = jax.lax.map(one_q, xs, batch_size=min(int(qb), n_q))
        mv = lambda a: jnp.moveaxis(a, 0, 1)                                # (n_p, n_q, mx, my)
        return mv(Om), mv(B), mv(D), jax.lax.pmax(jnp.max(cond), ("x", "y"))

    f = shard_map(_local, mesh=mesh, in_specs=(spec, spec, P(None), P(None)),
                  out_specs=(spec, spec, spec, P()), check_vma=False)

    @jax.jit
    def run(w, wn, z, ngk):
        Om, B, D, cond = f(w, wn, z, ngk)
        return Om, B, (D if ordered else None), cond
    return run
