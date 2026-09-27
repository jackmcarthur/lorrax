"""Plane-wave (ISDF-free) GW stages: ψ(G) store → χ₀(τ) → W_q(G, G') → Σ_x, Σ_c(ω).

The stage graph is ``docs/architecture/plane_wave_gw_stages.md``.  This module is
opt-in and unwired: ``gw.gw_jax`` never imports it, and it runs only as
``python -m gw.plane_wave_pipeline``.  Every stage is an existing owner:

    ψ store     ``wfn_loader.WfnLoader.load`` at the k-parents, placed in the r_μ face
                arrays (``gw.wavefunction_bundle.ParentGreenCarrier``: ``psi_nmu``
                ``(n_par, n_X, s, μ_Y)``, ``psi_mun`` ``(n_par, s, μ_X, n_Y)``) with the
                μ axis read as the sphere slot, zero-padded to one ngkmax carrier
    G(τ)        ``gw.greens_function_kernel.build_G_parents`` on those faces (the G builder,
                unchanged: ψ(G) in, G_k(p, p') out)
    χ₀(τ)       ``gw.mixed_basis_pair_convolution`` ``'trace'`` (K1), both particle–hole
                orientations, on the rule of ``gw.minimax_screening`` per sample
    χ(iω)       ``gw.plane_wave_screening.accumulate_chi`` (K3's factor)
    W           ``gw.plane_wave_screening.SphereScreening`` (K3: v_q(G), Dyson through
                ``w_isdf.solve_w``)
    Σ_x         the same pair convolution, ``'scalar'`` (K2), with B = v_q(G) on the
                χ sphere at the q-IBZ = the k-parents
    Σ_c(ω)      ``--deck``: χ at the MPA samples through ``mpa.model._evaluate_samples``
                (this basis as its χ producer), W^c and ``SphereScreening.fit_poles``,
                then ``mpa.sigma.compute_sigma_c_mpa_omega_grid`` with a τ body that
                chains ``build_G_tau`` → the ``'scalar'`` pair convolution → the face
                projector (``PlaneWaveGW.sigma_c``)

Head policy of this cut: the Γ head slot of v is zero (q + G = 0 excluded) and the
Γ W is the body solve only (v(0) = 0); the Γ head (``SphereScreening.gamma_head``)
needs the head owner's S(z) and a G-basis wing producer, which this cut does not wire.
A deck compared against it must say ``head_correction = off`` and
``mc_average_vcoul_body = false``.

Normalizations (K2, K3; both pinned against band sums in their suites):
χ_q = −s·X/(Ω·N_r²), s = 2/(n_spin·n_spinor); Σ_k = −X/(Ω·N_r²).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from types import SimpleNamespace

import numpy as np

__all__ = ["PlaneWaveSystem", "open_plane_wave_system", "PlaneWaveGW", "main"]

_RY_EV = 13.605693122994


@dataclasses.dataclass(frozen=True, eq=False)
class PlaneWaveSystem:
    """Host tables of one WFN for the plane-wave stages (every sphere on one slot carrier)."""
    sym: object
    kgrid: tuple
    fft_grid: tuple
    geometry: object          # vcoul.CoulombGeometry
    psi_full: object          # SphereSet: ψ sphere at every full-grid k (C order)
    psi_par: object           # SphereSet: ψ sphere at the k-parents
    plan: object              # the typed-transport tables (SphereTransport.typed, build_G_parents)
    psi_nmu: object           # (n_par, nb_pad, ns, ngkmax) ψ(G) at the parents, P(None,'x',None,'y')
    width: int                # one slot width for every sphere (the ngkmax the carrier pads)
    enk: np.ndarray           # (n_par, nb) Ry
    ecut: float               # max |k+G|² of the ψ spheres (Ry)
    ns: int
    cut_gap: float            # min over k of E[nb] − E[nb−1] (Ry): a multiplet split at the band cut is ~0


@dataclasses.dataclass(frozen=True, eq=False)
class SphereKPlan:
    """The k tables the plane-wave stages read: the typed transport (``SphereTransport.typed``),
    the G builder (``build_G_parents``) and the Σ executor (``parent_rows``, ``sym``,
    ``parent_full_rows``, ``n_parent``).  ``eq=False``: identity hashing, like
    ``CentroidKUnfoldPlan`` (a compiled kernel's cache key)."""
    irr_idx: np.ndarray
    sym_idx: np.ndarray
    spin_action_full: np.ndarray
    k_parent_frac: np.ndarray
    n_sym_spatial: int
    spatial_ops: np.ndarray
    translations: np.ndarray
    mesh_xy: object
    n_full: int
    sym: object
    parent_full_rows: np.ndarray

    @property
    def n_parent(self) -> int:
        return int(self.k_parent_frac.shape[0])

    def parent_rows(self, array, *, axis: int = 0):
        from .centroid_k_unfold import parent_rows
        return parent_rows(self.irr_idx, self.n_parent, array, axis=axis)


@dataclasses.dataclass(frozen=True, eq=False)
class SphereResidues:
    """The Σ executor's residue carrier on the plane-wave path (``MemoryPoleSource(q_wedge=)``).

    The executor wraps every residue batch in it (``values``) with ``load`` = the pair
    convolution's device tables (``MixedBasisPairConvolution.tables``), so the τ body reads
    them as window-executable arguments (``ppm_tau_kernel._wedge_residues``): multi-process
    JAX refuses to close over a global array.  The residues are on the q-IBZ rows, which the
    pair convolution's W transport unfolds on its load."""
    values: object = None
    load: object = None
    trs_rule: object = None
    tables_fn: object = None

    def with_load(self, mesh_xy):
        del mesh_xy
        return dataclasses.replace(self, load=self.tables_fn())


def _register_pytrees():
    import jax
    if not getattr(_register_pytrees, "done", False):
        jax.tree_util.register_dataclass(SphereResidues, data_fields=["values", "load"],
                                         meta_fields=["trs_rule", "tables_fn"])
        _register_pytrees.done = True


def _pad_width(g, width):
    return np.pad(g, ((0, 0), (0, width - g.shape[1]), (0, 0)))


def open_plane_wave_system(wfn_path: str, mesh, *, nb: int) -> PlaneWaveSystem:
    """Read ψ(G), E and the symmetry tables once (the loader's own doors)."""
    from jax.sharding import PartitionSpec as P
    from file_io import WfnLoader
    from vcoul import CoulombGeometry
    from .mixed_basis_pair_convolution import SphereSet
    with WfnLoader(wfn_path, mesh=mesh) as w:
        sym = w.symmetry()
        kgrid = tuple(int(v) for v in w.kgrid)
        box = tuple(int(v) for v in w.fft_grid)
        geom = CoulombGeometry.from_wfn(w)
        kf = np.asarray(w.kvecs(k="full_bz"))
        gf, nf = np.asarray(w.gvecs(k="full_bz")), np.asarray(w.ngk_valid(k="full_bz"))
        kp = np.asarray(w.kvecs(k=sym.parent_k_domain))
        gp, npar = np.asarray(w.gvecs(k=sym.parent_k_domain)), np.asarray(w.ngk_valid(k=sym.parent_k_domain))
        # the sharded loader lands ψ(G) directly in the ψ_nmu face layout (zero pad slots); the
        # band window is read to a multiple of the ranks (the phdf5 build refuses an explicit
        # spec otherwise); bands past ``nb`` carry zero weight in every Green
        n_load = -(-int(nb) // int(mesh.size)) * int(mesh.size)
        if n_load > int(w.nbands):
            raise ValueError(f"plane_wave_pipeline: nb={nb} rounds to {n_load} > the WFN's {w.nbands} bands")
        psi_nmu = w.load(bands=(0, n_load), k=sym.parent_k_domain, sharding=P(None, "x", None, "y"))
        ns = int(w.nspinor)
        if sym.parent_k_domain != "ibz":
            raise ValueError(f"plane_wave_pipeline: parent_k_domain {sym.parent_k_domain!r}; want 'ibz'")
        e_all = np.asarray(w.energies, np.float64)[0]
        enk = e_all[:, :int(nb)]
        cut_gap = (float(np.min(e_all[:, int(nb)] - e_all[:, int(nb) - 1]))
                   if e_all.shape[1] > int(nb) else float("inf"))
    width = max(gf.shape[1], gp.shape[1], int(psi_nmu.shape[-1]))
    gf, gp = _pad_width(gf, width), _pad_width(gp, width)
    bvec = np.asarray(geom.bvec, np.float64)
    ecut = max(float(np.max(np.sum(((kf[i][None] + gf[i, :nf[i]]) @ bvec) ** 2, axis=1)))
               for i in range(len(nf)))
    n_sp = int(np.asarray(sym.sym_matrices).shape[0])
    sidx = np.asarray(sym.sym_idx_k, np.int32)
    spin = (np.asarray(sym.spinor_action(sidx, nspinor=2)) if ns == 2
            else np.ones((len(sidx), 1, 1), np.complex128))
    plan = SphereKPlan(irr_idx=np.asarray(sym.irr_idx_k, np.int32), sym_idx=sidx,
                       spin_action_full=spin, k_parent_frac=kp, n_sym_spatial=n_sp,
                       spatial_ops=np.asarray(sym.sym_matrices)[:n_sp],
                       translations=np.asarray(sym.translations)[:n_sp],
                       mesh_xy=mesh, n_full=len(sidx), sym=sym,
                       parent_full_rows=np.asarray(sym.kirr_fullids, np.int32))
    return PlaneWaveSystem(sym=sym, kgrid=kgrid, fft_grid=box, geometry=geom,
                           psi_full=SphereSet(gf, nf, kf), psi_par=SphereSet(gp, npar, kp),
                           plan=plan, psi_nmu=psi_nmu, width=width, enk=enk, ecut=ecut, ns=ns, cut_gap=cut_gap)


class PlaneWaveGW:
    """The one-shot plane-wave stages on one mesh (module docstring)."""

    def __init__(self, mesh, system: PlaneWaveSystem, *, nval: int,
                 screened_coulomb_cutoff: float | None = None, wedge: bool = False):
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        from common.gvec_fft_box import build_sphere_box_index
        from runtime.padding import padded_axis
        from distrib_la import gemm_plan
        from .mixed_basis_pair_convolution import (ColumnWedge, MixedBasisPairConvolution,
                                                   PairOperand, SphereSet, SphereTransport,
                                                   screened_sphere_set)
        from .plane_wave_screening import SphereScreening, chi_pair_sum_scale
        from .wavefunction_bundle import ParentGreenCarrier
        s = system
        self.mesh, self.s, self.nval = mesh, s, int(nval)
        self.P = int(mesh.shape["x"]) * int(mesh.shape["y"])
        n_par, nb_c, ns, ngk = (int(v) for v in s.psi_nmu.shape)
        nb, width = int(s.enk.shape[1]), s.width
        box, bvec = s.fft_grid, np.asarray(s.geometry.bvec, np.float64)
        self.n_r = int(np.prod(box))
        self.omega_cell = float(s.geometry.cell_volume)
        # ---- ψ(G) in the r_μ face arrays: μ is the sphere slot, one carrier for every k
        self.M = int(padded_axis(width, self.P, name="psi sphere slots").carrier)
        self.nb_c, self.n_par = nb_c, n_par
        nmu = NamedSharding(mesh, P(None, "x", None, "y"))
        mun = NamedSharding(mesh, P(None, None, "x", "y"))
        # bands past ``nb`` (the rank-multiple read) are zero rows: inert in every Green and
        # projection, so the Σ executor's band carrier may include them
        live = (np.arange(nb_c) < nb)[None, :, None, None]
        psi_nmu = jax.jit(lambda a: jnp.pad(jnp.where(live, a, 0.0),
                                            ((0, 0), (0, 0), (0, 0), (0, self.M - ngk))),
                          out_shardings=nmu)(s.psi_nmu)
        psi_mun = jax.jit(lambda a: jnp.transpose(a, (0, 2, 3, 1)), out_shardings=mun)(psi_nmu)
        # the parent carrier's roles (gw.wavefunction_bundle.parent_sigma_operands): ψ_mun is the
        # direct operand of the G builder, ψ_nmu the conjugated one and the projection face
        # pad bands carry the top band's energy (their ψ rows are zero): no window moves
        self.enk = np.repeat(s.enk[:, -1:], self.nb_c, axis=1)
        self.enk[:, :nb] = s.enk
        e = s.enk
        self.mu = 0.5 * (float(e[:, :nval].max()) + float(e[:, nval:].min()))
        self.x_min = float(e[:, nval:].min() - e[:, :nval].max())
        self.x_max = float(e[:, nval:].max() - e[:, :nval].min())
        self.carrier = ParentGreenCarrier(psi_nmu=psi_nmu, psi_mun=psi_mun,
                                          enk=jnp.asarray(self.enk), occ=jnp.asarray(
                                              (np.arange(self.nb_c) < nval)[None, :].repeat(n_par, 0)
                                              .astype(np.float64)), plan=s.plan)
        self._gemm = gemm_plan(mesh, m=self.M * ns, k=self.nb_c, n=self.M * ns, nq=n_par,
                               dtype=jnp.complex128, layout="face")
        self.nb, self.ns = nb, ns
        # ---- operands and plans
        sidx = build_sphere_box_index(s.psi_par.gvecs, box, width, ngk_valid=s.psi_par.ngk)
        tr = SphereTransport.typed(s.plan, fft_grid=box, parent_sphere_index=sidx, children=s.psi_full)
        if np.any(tr.anti):
            raise ValueError(
                "GATE pw-pipeline-antiunitary: got antiunitary k rows; want a unitary k wedge; why: "
                "this cut builds real-weight Greens and the Σ route's W partner rule is not wired; "
                "fix: a WFN whose IBZ folds with unitary rows (inversion present)")
        op = PairOperand(s.psi_full, tr)
        nkf = s.psi_full.n
        cut = s.ecut if screened_coulomb_cutoff is None else float(screened_coulomb_cutoff)
        s_all = screened_sphere_set(fft_grid=box, psi=s.psi_full, bvec=bvec,
                                    q_frac=np.concatenate([s.psi_full.frac, s.psi_par.frac]),
                                    ecutwfc=s.ecut, screened_coulomb_cutoff=cut)
        self.w_full = SphereSet(s_all.gvecs[:nkf], s_all.ngk[:nkf], s_all.frac[:nkf])
        self.w_par = SphereSet(s_all.gvecs[nkf:], s_all.ngk[nkf:], s_all.frac[nkf:])
        widx = build_sphere_box_index(self.w_par.gvecs, box, self.w_par.width, ngk_valid=self.w_par.ngk)
        w_op = PairOperand(self.w_full, SphereTransport.typed(s.plan, fft_grid=box,
                                                              parent_sphere_index=widx,
                                                              children=self.w_full, ns=1))
        wedge_chi = ColumnWedge.from_symmaps(s.sym, self.w_full) if wedge else None
        wedge_sig = ColumnWedge.from_symmaps(s.sym, s.psi_full, ns=ns) if wedge else None
        common = dict(kgrid=s.kgrid, fft_grid=box, left=op)
        t0 = time.perf_counter()
        self.chi_conv = MixedBasisPairConvolution(mesh, right=op, out=self.w_par, product="trace",
                                                  wedge=wedge_chi, **common)
        self.sig_conv = MixedBasisPairConvolution(mesh, right=w_op, out=s.psi_par, product="scalar",
                                                  wedge=wedge_sig, **common)
        self.t_plans = time.perf_counter() - t0
        self.screen = SphereScreening(mesh, sphere=self.w_par, geometry=s.geometry, sys_dim=3,
                                      kgrid=s.kgrid)
        self.chi_scale = -chi_pair_sum_scale(cell_volume=self.omega_cell, n_r=self.n_r,
                                             n_spin=1, n_spinor=ns)
        self.walls: dict = {}

    # ------------------------------------------------------------------ G(τ)
    def green(self, weights):
        """G_k(p, p') at the parents with band weights ``(n_par, nb)`` (real or complex): the G
        builder, unchanged."""
        import jax.numpy as jnp
        from .greens_function_kernel import build_G_parents
        weights = np.asarray(weights)
        w = np.zeros((self.n_par, self.nb_c), weights.dtype)
        w[:, :self.nb] = weights
        return build_G_parents(self.carrier.psi_mun, self.carrier.psi_nmu, phases=jnp.asarray(w),
                               layout="face", gemm=self._gemm, k_unfold_plan=self.s.plan,
                               real_weights=not np.iscomplexobj(w)).G

    def _timed(self, name, fn, *a):
        import jax
        t0 = time.perf_counter()
        out = fn(*a)
        jax.block_until_ready(out)
        self.walls.setdefault(name, []).append(time.perf_counter() - t0)
        return out

    # ------------------------------------------------------------------ χ₀ → χ(z)
    def chi_nodes(self, tau, alpha_rows, *, both: bool):
        """χ_q(G, G'; z_j) = scale·Σ_l alpha_rows[j, l]·X(τ_l), ``(n_z, n_par, M_χ, M_χ)``.

        Per node the conduction Green carries e^{-(E−μ)τ} and the valence Green e^{+(E−μ)τ̄}:
        the pair convolution reads its right operand conjugated, so each (v, c) pair carries
        e^{-Δτ}, the ISDF χ₀ kernel's per-pair factor about the band edges.  ``both`` adds the
        reverse particle–hole orientation (a real-τ rule, as ``w_isdf._laplace_chi_args``
        prefolds it); a contour rule takes one orientation over its ±iτ nodes."""
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        from .plane_wave_screening import accumulate_chi
        tau = np.asarray(tau, np.complex128)
        alpha_rows = np.asarray(alpha_rows, np.complex128).reshape(-1, tau.size)
        e = self.s.enk - self.mu
        cond = np.arange(self.nb) >= self.nval
        Mc = int(self.screen.M)
        acc = jax.jit(lambda: jnp.zeros((alpha_rows.shape[0], self.w_par.n, Mc, Mc), jnp.complex128),
                      out_shardings=NamedSharding(self.mesh, P(None, None, "x", "y")))()
        real = lambda w: w.real if not np.any(np.imag(tau)) else w
        for l, t in enumerate(tau):
            Gc = self._timed("green", self.green, real(np.where(cond, np.exp(-e * t), 0.0)))
            Gv = self._timed("green", self.green, real(np.where(~cond, np.exp(e * np.conj(t)), 0.0)))
            for L, R in (((Gc, Gv), (Gv, Gc)) if both else ((Gc, Gv),)):
                X = self._timed("chi_pair_conv", self.chi_conv, L, R)
                acc = accumulate_chi(acc, X, alpha_rows[:, l], scale=self.chi_scale, mesh=self.mesh)
                del X
            del Gc, Gv
        return acc

    def chi(self, omegas_ry, *, target_error: float = 1e-8):
        """χ_q(G, G'; iω_j) at the k-parent q rows, ``(n_z, n_par, M_χ, M_χ)``; the rule per
        sample is ``minimax_screening``'s (ω = 0: 1/x; else x/(x² + ω²))."""
        from .minimax_screening import (solve_laplace_minimax_imag_interval,
                                        solve_laplace_minimax_interval)
        n_z = len(omegas_ry)
        acc = None
        self.rules = []
        for j, om in enumerate(omegas_ry):
            rule = (solve_laplace_minimax_interval(self.x_min, self.x_max, target_error=target_error)
                    if om == 0.0 else solve_laplace_minimax_imag_interval(
                        self.x_min, self.x_max, float(om), target_error=target_error))
            # 1/x rule: χ(0) = 2·Σ 1/x (both orientations); x/(x²+ω²) rule the same per orientation
            self.rules.append(dict(omega_ry=float(om), n_tau=len(rule.tau),
                                   max_error=float(rule.max_error)))
            rows = np.zeros((n_z, len(rule.tau)))
            rows[j] = np.asarray(rule.alpha)
            part = self.chi_nodes(rule.tau, rows, both=True)
            acc = part if acc is None else acc + part
        return acc

    # ------------------------------------------------------------------ MPA samples
    def wavefunctions(self):
        """The Σ executor's bundle: RSK4's parent carrier with full-k energy tables.

        Band edges: occupied ``nval``, QP window and Σ sum ``nb``, carrier ``nb_c`` (the
        rank-multiple read, zero ψ rows past ``nb``)."""
        import jax.numpy as jnp
        from .wavefunction_bundle import BandSlices, Wavefunctions
        irr = np.asarray(self.s.plan.irr_idx)
        occ = (np.arange(self.nb_c) < self.nval).astype(np.float64)
        sl = BandSlices.from_band_edges(0, 0, self.nval, self.nb, self.nb_c, b4_logical=self.nb)
        return Wavefunctions(enk=jnp.asarray(self.enk[irr]),
                             occ=jnp.asarray(np.broadcast_to(occ, (irr.size, self.nb_c))),
                             slices=sl, green_parent=self.carrier, layout="face")

    def mpa_samples(self, config, wfns, *, print_fn=print):
        """χ at the MPA plan's samples through the MPA owner's route dispatch.

        ``mpa.model``'s plan (``make_mpa_plan`` on ``build_static_quadrature``: the ISDF
        run's rules and z) and ``_evaluate_samples`` with this basis as its χ producer:
        static and imaginary points on their minimax rules, line points on the damped-line
        contour (``w_isdf._chi0_contour_alpha_rows`` weights about E_gap = 0).
        Returns ``(z (n_z,), χ (n_z, n_par, M_χ, M_χ))``."""
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        from .minimax_screening import build_static_quadrature
        from .mpa import sample_plan
        from .mpa.model import _evaluate_samples, make_mpa_plan
        from .w_isdf import _chi0_contour_alpha_rows
        quad, _ = build_static_quadrature(wfns, config.minimax_config, print_fn=print_fn)
        plan = make_mpa_plan(config, quad, material_class="insulator")
        z = sample_plan.plan_z(plan)
        Mc = int(self.screen.M)
        out = jax.jit(lambda: jnp.zeros((z.size, self.w_par.n, Mc, Mc), jnp.complex128),
                      out_shardings=NamedSharding(self.mesh, P(None, None, "x", "y")))()
        put = jax.jit(lambda a, x, i: a.at[i].set(x), donate_argnums=(0,),
                      out_shardings=NamedSharding(self.mesh, P(None, None, "x", "y")))
        box = [out]

        def write_full(point, value):
            box[0] = put(box[0], value, int(point["index"]))

        def contour(tau, weights, signs, zz):
            # the ISDF weights carry −1 (their static prefold does too); this basis's real-τ
            # rules carry none and a negative scale, so the contour rows are negated here
            rows = -_chi0_contour_alpha_rows(tau, weights, signs, zz, 0.0)
            acc = self.chi_nodes(tau, rows, both=False)
            return acc[0] if len(zz) == 1 else tuple(acc[j] for j in range(len(zz)))

        producer = SimpleNamespace(
            static=lambda q: self.chi_nodes(q.tau, np.asarray(q.alpha)[None], both=True)[0],
            imag=lambda q: self.chi_nodes(q.tau, np.asarray(q.alpha)[None], both=True)[0],
            contour=contour)
        _evaluate_samples(None, sample_plan.plan_routes(plan), quad, config, None, self.mesh,
                          material_class="insulator", sym=self.s.sym, energy_reference=None,
                          occupation_state=None, write_full=write_full, write_wedge=None,
                          static_gamma_override=None, gamma_row=None, kminq_rows=None,
                          chi=producer, print_fn=print_fn)
        self.mpa_plan = dict(z_ry=z, n_tau_lines=[len(p) for _, p in sample_plan.plan_routes(plan)["lines"]])
        return z, box[0]

    def mpa_poles(self, config, z, chi_z):
        """W → W^c → the MPA fit: ``(Omega, B, cond)`` on the q wedge (= the k-parents).

        The Γ W^c head slot is 0 (head off: v(0) = 0 makes the Γ row and column of W zero)."""
        W = self._timed("dyson", self.screen.solve_samples, chi_z)
        Wc, _ = self.screen.correlation(W, wcoul0=np.zeros(z.size), vc0=0.0)
        del W
        Omega, B, _, cond = self._timed(
            "mpa_fit", lambda: self.screen.fit_poles(Wc, z, int(config.mpa.n_poles),
                                                     solve=str(config.mpa.pole_solver)))
        return Omega, B, float(cond)

    # ------------------------------------------------------------------ Σ_c(ω)
    def sigma_c(self, config, wfns, Omega, B, *, print_fn=print):
        """Σ_c,nm(k, ω) through the Σ owner (``mpa.sigma.compute_sigma_c_mpa_omega_grid``):
        its planner, windows and accumulator, with this basis's τ body.

        ``_sigma_kij``: ``build_G_tau`` on the ψ(G) faces → the ``'scalar'`` pair convolution
        against W(τ) on the response sphere → Σ_k = −X/(Ω·N_r²) → the face projector
        (``contract_bands_block_reshard``).  The ISDF body differs only in its middle."""
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        from common.contract_bands import contract_bands_block_reshard
        from distrib_la import gemm_plan
        from .greens_function_kernel import build_G_tau
        from .mpa.sigma import MemoryPoleSource, compute_sigma_c_mpa_omega_grid
        from .ppm_tau_kernel import get_shared_sigma_tau_kernel
        from .ppm_windows import sigma_regularization_for_config
        from .sigma_box_plan import resolve_sigma_box_cache_dir
        mesh, plan, ns = self.mesh, self.s.plan, self.ns
        face_shape = (self.n_par, self.nb_c, self.M, ns)
        g_plan = gemm_plan(mesh, m=self.M * ns, k=self.nb_c, n=self.M * ns, nq=self.n_par,
                           dtype=jnp.complex128, layout="face", enable_active_range=True,
                           warmup=False)
        o_spec = NamedSharding(mesh, P(None, None, "x", None, "y"))
        scale = -1.0 / (self.omega_cell * self.n_r ** 2)
        conv = self.sig_conv

        def factory(_synthesis, sigma_axis):
            project = contract_bands_block_reshard(
                mesh, channels="none", layout="face", face_shape=face_shape,
                face_band_extent=sigma_axis.carrier)

            def sigma_kij(xn, yr, pxr, pyn, E_A, mask_A, E_ref_A, t, W_t, W_pt=None, load=None):
                opts = dict(e_ref=E_ref_A, layout="face", gemm=g_plan, k_unfold_plan=plan,
                            trim_zero_bands=True, unfold=False)
                opts["mask" if mask_A.dtype == jnp.bool_ else "band_weight"] = mask_A
                G = build_G_tau(xn, yr, E_A, 1j * t, **opts)
                X = conv(G.G, W_t, tables=load)                     # (n_par, M, ns, M, ns)
                O = jax.lax.with_sharding_constraint(
                    jnp.transpose(X * scale, (0, 2, 1, 4, 3)), o_spec)   # spin-major
                return project(pxr, O, pyn)

            return get_shared_sigma_tau_kernel(
                mesh_xy=mesh, kgrid=self.s.kgrid, layout="face", face_shape=face_shape,
                face_band_extent=sigma_axis.carrier, k_unfold_plan=plan, _sigma_kij=sigma_kij)

        _register_pytrees()
        poles = MemoryPoleSource(Omega, B, None, mesh_xy=mesh, axis=self.screen.axis,
                                 q_wedge=SphereResidues(tables_fn=conv.tables),
                                 provenance={"basis": "plane-wave sphere",
                                             "screening_diagrams": None})
        xi = sigma_regularization_for_config(config)
        from common.units import RYD_TO_EV
        return compute_sigma_c_mpa_omega_grid(
            wfns, poles, SimpleNamespace(nkx=self.s.kgrid[0], nky=self.s.kgrid[1],
                                         nkz=self.s.kgrid[2], mu_basis=None), mesh,
            omega_grid_ry=config.omega_grid_ry, efermi_ry=self.mu,
            regularization_width_ry=xi.resolved_ry,
            edge_factor=float(config.sigma.window_edge_factor),
            quadrature_eps=float(config.sigma.quadrature_eps),
            quadrature_cache_dir=resolve_sigma_box_cache_dir(config.sigma.quadrature_cache_dir,
                                                             config.input_dir),
            omega_grid_step_ry=float(config.sigma.omega_step_ev) / RYD_TO_EV,
            # one pole batch: every τ node pays one pair convolution, whatever the pole count
            pole_batch_size=min(int(config.mpa.n_poles), 8), material_class="insulator",
            tau_kernel_factory=factory, print_fn=print_fn)

    # ------------------------------------------------------------------ W
    def screened(self, chi_z):
        """W_q(G, G'; iω_j) = (1 − vχ)⁻¹ v; Γ is the head-removed body (v(0) = 0)."""
        return self._timed("dyson", self.screen.solve_samples, chi_z)

    # ------------------------------------------------------------------ Σ_x
    def exchange(self):
        """Σ_x,nn(k) at the k-parents (Ry): the 'scalar' pair convolution with B = v, projected."""
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        v = self.screen.v.astype(np.complex128)                           # (n_par, M_χ), Γ head 0
        V = jax.jit(lambda a: a[:, :, None] * jnp.eye(a.shape[1], dtype=a.dtype)[None],
                    out_shardings=NamedSharding(self.mesh, P(None, "x", "y")))(jnp.asarray(v))
        Gocc = self._timed("green", self.green,
                           np.broadcast_to((np.arange(self.nb) < self.nval).astype(float),
                                           self.s.enk.shape))
        X = self._timed("sigma_pair_conv", self.sig_conv, Gocc, V)
        S = X * (-1.0 / (self.omega_cell * self.n_r ** 2))
        # diagonal band projection ⟨n k|Σ_x|n k⟩ on the same r_μ faces (ψ in its sphere slots)
        diag = jax.jit(lambda s, p: jnp.einsum("knap,kpaqb,knbq->kn", jnp.conj(p), s, p))(
            S, self.carrier.psi_nmu)
        return np.real(_host(diag, self.mesh))[:, :self.nb]


def _host(x, mesh):
    """A small device array on every process's host (replicated, then the local copy)."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    y = jax.jit(lambda a: a, out_shardings=NamedSharding(mesh, P()))(x)
    return np.asarray(y.addressable_shards[0].data)


def _one_shot(args, mesh, system, gw, say, t_read):
    """Σ_x and Σ_c(ω) at the k-parents; per-stage walls and the running per-rank peak."""
    import jax
    from .gw_config import LorraxConfig, resolve_mpa_sampling_alpha
    # the cut is insulating (a gapped minimax interval), so the MPA sampling exponent resolves
    # as gw_jax resolves it for an insulator
    config = resolve_mpa_sampling_alpha(
        LorraxConfig.from_input_file(args.deck, print_fn=lambda *a, **k: None), "insulator",
        print_fn=say)
    stages, t = {}, time.perf_counter()

    def stage(name):
        nonlocal t
        now = time.perf_counter()
        stages[name] = dict(wall_s=now - t, peak_gb=_peak_gb())
        say(f"stage {name}: {now - t:.2f} s, running peak {stages[name]['peak_gb']:.2f} GB/rank")
        t = time.perf_counter()

    sigx = gw.exchange()
    stage("sigma_x")
    wfns = gw.wavefunctions()
    z, chi_z = gw.mpa_samples(config, wfns, print_fn=say)
    stage("chi_mpa_samples")
    Omega, B, cond = gw.mpa_poles(config, z, chi_z)
    del chi_z
    stage("w_dyson_and_mpa_fit")
    body = gw.sigma_c(config, wfns, Omega, B, print_fn=say)
    stage("sigma_c_tau_sweep")
    # the diagonal at the raw parents (the full-k rows that ARE the parents)
    rows = np.asarray(system.plan.parent_full_rows)
    diag = jax.jit(lambda a: jax.numpy.diagonal(a[:, rows], axis1=-2, axis2=-1))(body.sigma_c_kij)
    sc = _host(diag, mesh)[:, :, :gw.nb]                                  # (n_ω, n_par, nb) Ry
    om = np.asarray(body.omega_ry)
    x = system.enk - gw.mu                                               # Σ_c(E) at E − μ
    at = np.array([[np.interp(x[k, n], om, sc[:, k, n].real) + 1j * np.interp(x[k, n], om, sc[:, k, n].imag)
                    for n in range(gw.nb)] for k in range(gw.n_par)])
    walls = {k: dict(n=len(v), total_s=float(np.sum(v)), warm_s=float(np.min(v[1:] if len(v) > 1 else v)))
             for k, v in gw.walls.items()}
    res = dict(kgrid=system.kgrid, fft_grid=system.fft_grid, n_r=gw.n_r, P=gw.P, nb=gw.nb, nval=gw.nval,
               psi_width=int(system.width), psi_carrier=gw.M, chi_width=int(gw.w_par.width),
               chi_carrier=int(gw.screen.M), kpar_frac=system.psi_par.frac.tolist(),
               enk_ev=(system.enk * _RY_EV).tolist(), mu_ev=gw.mu * _RY_EV,
               sigx_ev=(sigx * _RY_EV).tolist(),
               sigc_at_e_ev_re=(at.real * _RY_EV).tolist(), sigc_at_e_ev_im=(at.imag * _RY_EV).tolist(),
               omega_ev=(om * _RY_EV).tolist(),
               sigc_omega_ev_re=(sc.real * _RY_EV).tolist(), sigc_omega_ev_im=(sc.imag * _RY_EV).tolist(),
               mpa_z_ry_re=np.real(z).tolist(), mpa_z_ry_im=np.imag(z).tolist(), mpa_fit_cond=cond,
               stages=stages, walls=walls, t_read_s=t_read, t_plans_s=gw.t_plans,
               chi_law=gw.chi_conv.describe(), sigma_law=gw.sig_conv.describe(),
               screen_law=gw.screen.describe(len(z), int(config.mpa.n_poles)))
    for k in range(gw.n_par):
        say(f"k{k} Σ_c(E) re (eV): " + " ".join(f"{v:.4f}" for v in at[k].real * _RY_EV))
    say("walls:", json.dumps(walls))
    if jax.process_index() == 0:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


def _peak_gb():
    import jax
    st = jax.local_devices()[0].memory_stats() or {}
    return float(st.get("peak_bytes_in_use", 0)) / 1e9


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plane-wave (ISDF-free) GW stages, first cut (K4).")
    ap.add_argument("--wfn", required=True)
    ap.add_argument("--nval", type=int, required=True)
    ap.add_argument("--nb", type=int, required=True, help="bands in G and the χ₀ sum")
    ap.add_argument("--omega-ry", default="0.0,0.5", help="imaginary frequencies of χ/W (Ry)")
    ap.add_argument("--screened-coulomb-cutoff", type=float, default=None)
    ap.add_argument("--wedge", action="store_true")
    ap.add_argument("--deck", default=None,
                    help="a gw_jax deck (compute_mode = mpa): run Σ_x + the MPA Σ_c(ω) one-shot on its "
                         "Σ grid, η, ε and MPA plan instead of the iω χ/W diagnostic")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    import jax
    from common.collectives import prepare_mesh
    mesh = prepare_mesh(print_fn=lambda *a, **k: None)
    say = (lambda *a: print("[pw-pipeline]", *a, flush=True)) if jax.process_index() == 0 else (lambda *a: None)
    t0 = time.perf_counter()
    system = open_plane_wave_system(args.wfn, mesh, nb=args.nb)
    t_read = time.perf_counter() - t0
    gw = PlaneWaveGW(mesh, system, nval=args.nval,
                     screened_coulomb_cutoff=args.screened_coulomb_cutoff, wedge=args.wedge)
    say(f"system: kgrid {system.kgrid}, box {system.fft_grid} (N_r {gw.n_r}), n_par {gw.n_par}, "
        f"nb {gw.nb}, ns {gw.ns}, ψ width {system.width} (carrier {gw.M}), χ sphere "
        f"{gw.w_par.width} (carrier {gw.screen.M}), gap {gw.x_min * _RY_EV:.4f} eV, band-cut gap "
        f"{system.cut_gap * _RY_EV:.4f} eV; "
        f"read {t_read:.1f} s, plans {gw.t_plans:.1f} s")
    say(gw.chi_conv.describe())
    say(gw.sig_conv.describe())
    if args.deck is not None:
        return _one_shot(args, mesh, system, gw, say, t_read)
    sigx = gw.exchange()
    peak_x = _peak_gb()
    omegas = [float(v) for v in args.omega_ry.split(",")]
    chi_z = gw.chi(omegas)
    peak_chi = _peak_gb()
    W = gw.screened(chi_z)
    peak_w = _peak_gb()
    # ε⁻¹_q(0, 0; iω) = W_q(0, 0)/v_q(0) at q ≠ 0 (the Γ row is the head-removed body)
    w00 = _host(W[:, :, 0, 0], mesh)
    v0 = gw.screen.v[:, 0]
    qc = system.psi_par.frac @ np.asarray(system.geometry.bvec, np.float64)
    einv = {f"{om:g}": [None if v0[i] == 0 else float(np.real(w00[j, i] / v0[i]))
                        for i in range(len(v0))] for j, om in enumerate(omegas)}
    walls = {k: dict(n=len(v), total_s=float(np.sum(v)), warm_s=float(np.min(v[1:] if len(v) > 1 else v)))
             for k, v in gw.walls.items()}
    res = dict(kgrid=system.kgrid, fft_grid=system.fft_grid, n_r=gw.n_r, P=gw.P, nb=gw.nb, nval=gw.nval,
               psi_width=int(system.width), psi_carrier=gw.M, chi_width=int(gw.w_par.width),
               chi_carrier=int(gw.screen.M), kpar_frac=system.psi_par.frac.tolist(),
               q_cart_norm=np.linalg.norm(qc, axis=1).tolist(),
               enk_ev=(system.enk * _RY_EV).tolist(), cut_gap_ev=system.cut_gap * _RY_EV, sigx_ev=(sigx * _RY_EV).tolist(),
               eps_inv_00=einv, rules=gw.rules, walls=walls,
               peak_gb=dict(after_sigx=peak_x, after_chi=peak_chi, after_w=peak_w),
               chi_law=gw.chi_conv.describe(), sigma_law=gw.sig_conv.describe(),
               screen_law=gw.screen.describe(len(omegas), 1), t_read_s=t_read, t_plans_s=gw.t_plans)
    for i in range(sigx.shape[0]):
        say(f"k{i} {np.round(system.psi_par.frac[i], 4).tolist()}  Σ_x (eV): "
            + " ".join(f"{v:.4f}" for v in sigx[i] * _RY_EV))
    for om, row in einv.items():
        say(f"eps^-1_00(q; i{om} Ry): " + " ".join("—" if v is None else f"{v:.4f}" for v in row))
    say("walls:", json.dumps(walls))
    say(f"peak per rank (GB): Σ_x {peak_x:.2f}, χ {peak_chi:.2f}, W {peak_w:.2f}")
    if jax.process_index() == 0:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    from runtime import initialize_communicator_stack, run_main_and_finalize
    initialize_communicator_stack(platform="gpu")
    run_main_and_finalize(main)
