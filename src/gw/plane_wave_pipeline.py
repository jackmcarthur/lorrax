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
    def chi_nodes(self, tau, alpha_rows, *, both: bool, reverse_rows=None):
        """χ_q(G, G'; z_j) = scale·Σ_l alpha_rows[j, l]·X(τ_l), ``(n_z, n_par, M_χ, M_χ)``.

        Per node the conduction Green carries e^{-(E−μ)τ} and the valence Green e^{+(E−μ)τ̄}:
        the pair convolution reads its right operand conjugated, so each (v, c) pair carries
        e^{-Δτ}, the ISDF χ₀ kernel's per-pair factor about the band edges.  ``both`` adds the
        reverse particle–hole orientation (a real-τ rule, as ``w_isdf._laplace_chi_args``
        prefolds it); a contour rule takes one orientation over its ±iτ nodes.
        ``reverse_rows`` (same shape as ``alpha_rows``) weights the reverse orientation with its
        own row: a response-bank rule (``minimax.response_group_rules``), whose reverse rows fit
        1/(d + z) at conj(τ) while the forward rows fit 1/(d − z) at τ."""
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        from .plane_wave_screening import accumulate_chi
        tau = np.asarray(tau, np.complex128)
        alpha_rows = np.asarray(alpha_rows, np.complex128).reshape(-1, tau.size)
        if reverse_rows is not None:
            reverse_rows = np.asarray(reverse_rows, np.complex128).reshape(alpha_rows.shape)
        e = self.s.enk - self.mu
        cond = np.arange(self.nb) >= self.nval
        Mc = int(self.screen.M)
        acc = jax.jit(lambda: jnp.zeros((alpha_rows.shape[0], self.w_par.n, Mc, Mc), jnp.complex128),
                      out_shardings=NamedSharding(self.mesh, P(None, None, "x", "y")))()
        real = lambda w: w.real if not np.any(np.imag(tau)) else w
        for l, t in enumerate(tau):
            Gc = self._timed("green", self.green, real(np.where(cond, np.exp(-e * t), 0.0)))
            Gv = self._timed("green", self.green, real(np.where(~cond, np.exp(e * np.conj(t)), 0.0)))
            terms = [((Gc, Gv), alpha_rows[:, l])]
            if both:
                terms.append(((Gv, Gc), alpha_rows[:, l]))
            if reverse_rows is not None:
                terms.append(((Gv, Gc), reverse_rows[:, l]))
            for (L, R), rows in terms:
                X = self._timed("chi_pair_conv", self.chi_conv, L, R)
                acc = accumulate_chi(acc, X, rows, scale=self.chi_scale, mesh=self.mesh)
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

    def contour_chi(self, tau, weights, signs, zz):
        """χ(z_j) of a damped-line contour rule (the MPA owner's line route): ``(n_z, n_par, M, M)``.

        The ISDF weights carry −1 (their static prefold does too); this basis's real-τ rules
        carry none and a negative scale, so the contour rows are negated here."""
        from .w_isdf import _chi0_contour_alpha_rows
        rows = -_chi0_contour_alpha_rows(tau, weights, signs, np.asarray(zz), 0.0)
        return self.chi_nodes(tau, rows, both=False)

    def line_chi(self, z, *, rel_tol=1e-8, max_order=256):
        """χ(z_j) for points on one damped line Im z = const through ``minimax.damped_line_rule``
        (sized on max|Re z| + the top transition, as the MPA owner sizes it), ±τ nodes.
        Returns ``(χ (n_z, n_par, M, M), node count)``."""
        import minimax
        z = np.asarray(z, np.complex128)
        if np.ptp(z.imag) > 1e-14 * np.max(z.imag):
            raise ValueError("line_chi: the points must share one line height")
        rule = minimax.damped_line_rule(float(z[0].imag), self.x_max + float(np.max(np.abs(z.real))),
                                        rel_tol=rel_tol, max_order=max_order)
        t, h = np.asarray(rule["t"]), np.asarray(rule["h"])
        tau = np.concatenate((1j * t, -1j * t))
        signs = np.concatenate((np.ones(t.size, np.int8), -np.ones(t.size, np.int8)))
        weights = np.broadcast_to(np.concatenate((1j * h, -1j * h)), (z.size, 2 * t.size))
        return self.contour_chi(tau, weights, signs, z), int(tau.size)

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
            acc = self.contour_chi(tau, weights, signs, zz)
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
    @property
    def _face_shape(self):
        return (self.n_par, self.nb_c, self.M, self.ns)

    def sigma_kij(self, sigma_axis):
        """The τ body's spatial part: ``build_G_tau`` on the ψ(G) faces → the ``'scalar'`` pair
        convolution against W(τ) on the q-IBZ response sphere → Σ_k = −X/(Ω·N_r²) → the face
        projector (``contract_bands_block_reshard``).  Signature of the resident route's
        ``_sigma_kij`` (``ppm_tau_kernel``); ``load`` is the pair convolution's device tables."""
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        from common.contract_bands import contract_bands_block_reshard
        from distrib_la import gemm_plan
        from .greens_function_kernel import build_G_tau
        mesh, plan, ns = self.mesh, self.s.plan, self.ns
        g_plan = gemm_plan(mesh, m=self.M * ns, k=self.nb_c, n=self.M * ns, nq=self.n_par,
                           dtype=jnp.complex128, layout="face", enable_active_range=True,
                           warmup=False)
        o_spec = NamedSharding(mesh, P(None, None, "x", None, "y"))
        scale = -1.0 / (self.omega_cell * self.n_r ** 2)
        conv = self.sig_conv
        project = contract_bands_block_reshard(
            mesh, channels="none", layout="face", face_shape=self._face_shape,
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
        return sigma_kij

    def sigma_c(self, config, wfns, Omega, B, *, print_fn=print):
        """Σ_c,nm(k, ω) through the Σ owner (``mpa.sigma.compute_sigma_c_mpa_omega_grid``):
        its planner, windows and accumulator, with this basis's τ body.

        ``_sigma_kij``: ``build_G_tau`` on the ψ(G) faces → the ``'scalar'`` pair convolution
        against W(τ) on the response sphere → Σ_k = −X/(Ω·N_r²) → the face projector
        (``contract_bands_block_reshard``), :meth:`sigma_kij`.  The ISDF body differs only in
        its middle."""
        from .mpa.sigma import MemoryPoleSource, compute_sigma_c_mpa_omega_grid
        from .ppm_tau_kernel import get_shared_sigma_tau_kernel
        from .ppm_windows import sigma_regularization_for_config
        from .sigma_box_plan import resolve_sigma_box_cache_dir
        mesh, plan = self.mesh, self.s.plan
        conv = self.sig_conv

        def factory(_synthesis, sigma_axis):
            return get_shared_sigma_tau_kernel(
                mesh_xy=mesh, kgrid=self.s.kgrid, layout="face", face_shape=self._face_shape,
                face_band_extent=sigma_axis.carrier, k_unfold_plan=plan,
                _sigma_kij=self.sigma_kij(sigma_axis))

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

    # ------------------------------------------------------------------ shared pole
    def sphere_meta(self, port_count=None):
        """The metadata the shared-pole owners read, with the response sphere as the port axis:
        ``n_rmu`` is the sphere's logical width, ``n_rmu_padded`` its carrier (the ISDF packed
        centroid count's slots).  No centroid basis (``mu_basis = None``).

        ``port_count`` (diagnostic): the port count the recipe sizes its direction widths and
        pole budget from, in place of the sphere width (they are fractions of it), so the
        sphere model can carry the direction and pole counts an ISDF basis of that size gets.
        The operators stay on the sphere."""
        ax = self.screen.axis
        ports = int(ax.logical) if port_count is None else int(port_count)
        kx, ky, kz = (int(v) for v in self.s.kgrid)
        return SimpleNamespace(nk_tot=kx * ky * kz, nkx=kx, nky=ky, nkz=kz, nspinor=self.ns,
                               nspinor_wfnfile=self.ns, nspin=1, cell_volume=self.omega_cell,
                               n_rmu=ports, n_rmu_padded=max(ports, int(ax.carrier)),
                               b_id_4_chi_user=self.nb, mu_basis=None)

    def shared_pole_recipe(self, config, wfns, meta, *, print_fn=print):
        """The recipe owner's supports, widths and pole budget on this basis (n = sphere width):
        ``bind_shared_pole_census`` then ``resolve_shared_pole_recipe``, as gw_jax calls them."""
        from .shared_pole_recipe import bind_shared_pole_census, resolve_shared_pole_recipe
        nk = int(meta.nk_tot)
        bind_shared_pole_census(wfns, meta, occupation_state=None,
                                trs_allowed=bool(self.s.sym.trs_allowed),
                                state_capacity=2.0 / self.ns, kweights=np.full(nk, 1.0 / nk))
        meta.shared_pole_recipe = resolve_shared_pole_recipe(config, wfns, meta, mesh_xy=self.mesh,
                                                             print_fn=print_fn)
        return meta.shared_pole_recipe

    def response_rows(self, z, *, rel_tol, group_size=8):
        """The response bank's complex-time rules at ``z`` (upper half plane) as this basis's rows.

        ``minimax.response_group_rules`` on the snapped transition interval (the insulating
        support of ``response_bank.response_quadrature``: decay rate 0, reference = lo): forward
        rows fit 1/(d − z) with e^{−(d−ref)t}, reverse rows fit 1/(d + z) at conj(t), value and
        d/ds.  This basis's forward pair carries e^{−dt} and its reverse pair e^{−d·conj t}
        (:meth:`chi_nodes`), so the rows carry e^{ref·t} and e^{ref·conj t}.  Returns
        ``[(t, forward, reverse)]`` with rows ``(2 n_z, L)``: value of z_j at 2j, d/ds at 2j+1."""
        import minimax
        from .response_bank import response_groups
        from .sigma_box_plan import snap_outward
        lo, hi = snap_outward(self.x_min, 1., -1), snap_outward(self.x_max, 1., +1)
        z = np.asarray(z, np.complex128).reshape(-1)
        out = []
        for members in response_groups(z, int(group_size)):
            for rule in minimax.response_group_rules(lo, hi, z[members], rel_tol=float(rel_tol),
                                                     decay_rate=0.):
                c = int(rule["count"])
                t = np.asarray(rule["t"], np.complex128)[:c]
                ref = float(rule["reference_ry"])
                gauge = (np.exp(ref * t), np.exp(ref * np.conj(t)))
                rows = [np.zeros((2 * z.size, c), np.complex128) for _ in range(2)]
                for j, m in enumerate(rule["members"]):
                    i = int(members[m])
                    for side in (0, 1):
                        rows[side][2 * i] = np.asarray(rule["value"])[j, side, :c] * gauge[side]
                        rows[side][2 * i + 1] = np.asarray(rule["derivative"])[j, side, :c] * gauge[side]
                out.append((t, rows[0], rows[1]))
        return out

    def response_chi(self, z, *, rel_tol, group_size=8):
        """χ(z_j) and ∂_sχ(z_j) on the q-IBZ sphere, ``(2 n_z, n_par, M_χ, M_χ)`` interleaved;
        one Green pair per rule node (K1 'trace', both orientations).  Returns (χ, node count)."""
        acc, nodes = None, 0
        for t, fwd, rev in self.response_rows(z, rel_tol=rel_tol, group_size=group_size):
            part = self.chi_nodes(t, fwd, both=False, reverse_rows=rev)
            acc = part if acc is None else acc + part
            nodes += int(t.size)
        return acc, nodes

    def bare_moments(self):
        """A0, A1 of χ(z) = A0/s + A1/s² + … (s = z², Ry): the exact band sums
        −scale·Σ_orientations Σ_vc d^{1,3} ρρ† at τ = 0 through the binomials of d = e_c − e_v,
        six correlations, as ``response_bank.exact_bare_moments`` forms them.
        Returns ``(2, n_par, M_χ, M_χ)``: A0, A1."""
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        from .plane_wave_screening import accumulate_chi
        e = self.s.enk - self.mu
        cond = np.arange(self.nb) >= self.nval
        Mc = int(self.screen.M)
        acc = jax.jit(lambda: jnp.zeros((2, self.w_par.n, Mc, Mc), jnp.complex128),
                      out_shardings=NamedSharding(self.mesh, P(None, None, "x", "y")))()
        # (power of e_c, power of e_v) -> coefficient in (A0, A1)
        terms = {(1, 0): (1., 0.), (0, 1): (-1., 0.), (3, 0): (0., 1.), (2, 1): (0., -3.),
                 (1, 2): (0., 3.), (0, 3): (0., -1.)}
        for (a, b), rows in terms.items():
            Gc = self._timed("green", self.green, np.where(cond, e ** a, 0.0))
            Gv = self._timed("green", self.green, np.where(~cond, e ** b, 0.0))
            for L, R in ((Gc, Gv), (Gv, Gc)):
                X = self._timed("chi_pair_conv", self.chi_conv, L, R)
                acc = accumulate_chi(acc, X, rows, scale=-self.chi_scale, mesh=self.mesh)
                del X
            del Gc, Gv
        return acc

    def response_programs(self, config):
        """The response owner's algebra on this basis (``response_bank._response_programs``):
        ``value(H, χ) = H X (I − X)⁻¹ H`` with X = HχH (W^c), ``slope(H, W^c, ∂_sχ) = W ∂_sχ W``,
        ``moments(H, A0, A1) = (M1, M3)``; H = diag √v_q(G) on the sphere, pref 1 (χ carries
        this basis's normalization).  Returns ``(value, slope, moments, H)``."""
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        from .gw_config import linalg_resolution
        from .response_bank import _response_programs
        resolution = linalg_resolution({"linalg": config.backend.linalg})
        backend = "off" if resolution.layout == "local" else "distributed"
        value, slope, moments, _ = _response_programs(
            self.mesh, int(self.screen.M), backend, resolution.batched_route, 1.0, False, None)
        h = np.sqrt(np.asarray(self.screen.v, np.float64)).astype(np.complex128)   # Γ head slot 0
        H = jax.jit(lambda a: a[:, :, None] * jnp.eye(a.shape[1], dtype=a.dtype)[None],
                    out_shardings=NamedSharding(self.mesh, P(None, "x", "y")))(jnp.asarray(h))
        return value, slope, moments, H

    def shared_pole_model(self, config, wfns, meta, *, print_fn=print):
        """The shared-pole model of W^c on the sphere: recipe → χ, ∂_sχ at every support (the
        response rules) → W^c, ∂_sW^c and M1, M3 (the response algebra) → the constructor's
        round functions, one parent per rank.

        FOLLOW-UP (coordinator, 2026-09-26): the round sequence below repeats the control flow
        of ``shared_pole_constructor.construct_shared_poles`` (selection → infinity → tables →
        ``reduce_round`` → gates), because that owner reads its samples from the disk bank and
        its basis from ``meta.mu_basis``.  It must become
        ``construct_shared_poles(bank=<reader protocol>, basis=<axis>)`` and this method must
        call it; the branch does not land before that.

        Returns a dict: ``b`` (n_par, M_χ, K) at ``P(None, 'x', None)``, ``poles2`` host
        (n_par, K) Ry², ``counts`` (n_par,), per-parent gate rows, the recipe and timings."""
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        from file_io.shared_pole_store import face_width
        from .response_bank import bank_points
        from .shared_pole_capacity import constructor_eigenplan
        from .shared_pole_directions import (_round_kernels, _sample_point, leading_response_directions,
                                             line_panel_states, line_sample_states, port_extent,
                                             select_round_states)
        from .shared_pole_local import reduce_round, round_tables
        mesh, n = self.mesh, int(self.screen.M)
        t0 = time.perf_counter()
        recipe = self.shared_pole_recipe(config, wfns, meta, print_fn=print_fn)
        points = bank_points(recipe)                        # distinct id -> z (Ry)
        chi, nodes = self.response_chi(points, rel_tol=float(recipe["bank_rule_tolerance"]))
        t_chi = time.perf_counter() - t0
        value, slope, moments, H = self.response_programs(config)
        Wc = [value(H, chi[2 * j]) for j in range(points.size)]
        dWc = [slope(H, Wc[j], chi[2 * j + 1]) for j in range(points.size)]
        del chi
        A = self.bare_moments()
        # The exact band sums are Hermitian; the pair convolution's r'-wedge unfold leaves an
        # anti-Hermitian residue above the 1e-12 relative bar of the constructor's checked
        # eigensolver.  Measured here, then removed from the moments only (ISDF's t = 0
        # correlation is formed Hermitian in its stream kernel).
        anti = jax.jit(lambda a: jnp.sqrt(jnp.sum(jnp.abs(a - jnp.conj(jnp.swapaxes(a, -1, -2))) ** 2,
                                                  axis=(-2, -1)))
                       / (2 * jnp.sqrt(jnp.sum(jnp.abs(a) ** 2, axis=(-2, -1)))))
        herm = jax.jit(lambda a: 0.5 * (a + jnp.conj(jnp.swapaxes(a, -1, -2))),
                       out_shardings=NamedSharding(mesh, P(None, "x", "y")))
        self.hermiticity = {name: np.asarray(_host(anti(a), mesh)).tolist() for name, a in (
            ("A0", A[0]), ("A1", A[1]), ("Wc_first_imag", Wc[int(np.argmin(np.abs(points.real)))]))}
        print_fn("shared-pole anti-Hermitian residue ||a - a^H||/(2||a||) per q: "
                 + json.dumps({k: [f"{x:.2e}" for x in v] for k, v in self.hermiticity.items()}))
        M1, M3 = moments(H, herm(A[0]), herm(A[1]))
        del A
        t_samples = time.perf_counter() - t0
        fit = [int(i) for i in recipe["fit_ids"]]
        line = [i for i in fit if _sample_point(recipe, i).real != 0]
        dense = [i for i in fit if i not in line]
        ranks = int(mesh.size)
        batch = NamedSharding(mesh, P(("x", "y")))
        column_extent = port_extent(mesh)
        eig, svd = constructor_eigenplan(mesh, n, "local"), constructor_eigenplan(mesh, 2 * n, "local")
        kernels = _round_kernels(mesh, "batch")
        blocks, pole_rows, active_rows, gates = [], [], [], []
        for lo in range(0, self.n_par, ranks):
            ids = [min(q, self.n_par - 1) for q in range(lo, lo + ranks)]
            real = min(ranks, self.n_par - lo)

            def stack(fields, ids=tuple(ids)):
                return jax.jit(lambda *a: jnp.stack([f[jnp.asarray(ids)] for f in a], axis=1),
                               out_shardings=batch)(*fields)
            rows = lambda f, ids=tuple(ids): jax.jit(lambda a: a[jnp.asarray(ids)],
                                                      out_shardings=batch)(f)
            line_states = {}
            for sid in line:
                sel = line_sample_states(stack([Wc[sid]]), stack([dWc[sid]]), recipe, sid=sid,
                                         ordered=False, real=real, mesh_xy=mesh, eigh_plan=eig,
                                         svd_plan=svd, column_extent=column_extent, logical_n=n)
                st = sel["states"]
                panels = kernels.stack(st[0][1], *[a for x in st for a in x[2:]])
                line_states[sid] = line_panel_states(panels, sel["counts"], recipe, sid=sid,
                                                     ordered=False, mesh_xy=mesh)
            states, counts, _roles = select_round_states(
                dict(Wc=stack([Wc[i] for i in dense]), dWc_ds=stack([dWc[i] for i in dense])),
                recipe, sample_ids=dense, real=real, mesh_xy=mesh, eigh_plan=eig, svd_plan=svd,
                column_extent=column_extent, logical_n=n, ordered=False, line_states=line_states)
            m1, m3 = rows(M1), rows(M3)
            width = min(n, max(1, int(recipe["infinity_width"])))
            qi, values = leading_response_directions(
                m1, width, eigh_plan=eig, column_extent=column_extent,
                multiplet_tol=recipe["multiplet_relative_tolerance"], real_rows=real)
            infinity = (qi, kernels.apply(m1, qi), kernels.apply(m3, qi))
            tables = round_tables(counts, tuple(int(x[1].shape[-1]) for x in states),
                                  [x[0] for x in states], [int(v.shape[-1]) for v in values],
                                  int(qi.shape[-1]), column_extent=column_extent, ordered=False,
                                  odd_moments=False)
            side = int(tables["active"].shape[-1])
            model, _signed, vectors, diag = reduce_round(
                states, infinity, tables, real=real, mesh_xy=mesh,
                native_eigh=constructor_eigenplan(mesh, side, "local").native_fn, ordered=False,
                odd_moments=False, keep_budget=recipe.get("pole_budget"))
            reduction, zero, retained, _perm = jax.tree.map(np.asarray, diag)
            poles, active = (np.asarray(a) for a in vectors)
            for slot in range(real):
                gates.append(dict(
                    q=lo + slot, side=side, K=int(active[slot].sum()),
                    gram_valid=bool(reduction["gram_valid"][slot]),
                    gram_min_relative=float(reduction["gram_min_relative"][slot]),
                    gram_diagonal_positive=bool(reduction["gram_diagonal_positive"][slot]),
                    retained_metric_positive=bool(reduction["retained_metric_positive"][slot]),
                    zero_policy=bool(zero["zero_policy"][slot]),
                    retained_moments_max=float(max(np.max(v[slot]) for v in retained.values()))))
            blocks.append((jax.jit(lambda b: b, out_shardings=NamedSharding(mesh, P(None, "x", None)))(
                model[0]), real))
            pole_rows.append(poles[:real])
            active_rows.append(active[:real])
            del states, infinity, model, line_states
        counts = np.concatenate([a.sum(axis=-1) for a in active_rows]).astype(np.int64)
        K = int(face_width(mesh, int(counts.max())))
        poles2 = np.ones((self.n_par, K))
        # rounds may solve at different pencil sides; each parent's model is an active prefix
        rows = [(p, int(a.sum())) for P_, A_ in zip(pole_rows, active_rows) for p, a in zip(P_, A_)]
        for q, (p, c) in enumerate(rows):
            poles2[q, :c] = p[:c]
        pad = lambda b: jnp.pad(b[:, :, :K], ((0, 0), (0, 0), (0, max(0, K - b.shape[-1]))))
        b = jax.jit(lambda *bs: jnp.concatenate([pad(x)[:r] for x, r in zip(bs, [r for _, r in blocks])]),
                    out_shardings=NamedSharding(mesh, P(None, "x", None)))(*[x for x, _ in blocks])
        return dict(b=b, poles2=poles2, counts=counts, K=K, recipe=recipe, gates=gates,
                    points=points, response_nodes=nodes, dense=dense, line=line,
                    walls=dict(chi_s=t_chi, samples_s=t_samples, total_s=time.perf_counter() - t0))

    def shared_pole_w(self, model, z):
        """W^c(q, z) = b diag(1/(z² − Ω²)) b† of the model at the parents, ``(n_z, n_par, M, M)``."""
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        s = np.asarray(z, np.complex128) ** 2
        active = np.arange(model["K"])[None, :] < model["counts"][:, None]
        w = np.where(active[None], 1.0 / (s[:, None, None] - model["poles2"][None]), 0.0)
        return jax.jit(lambda b, w: jnp.einsum("qmk,zqk,qnk->zqmn", b, w, jnp.conj(b)),
                       out_shardings=NamedSharding(self.mesh, P(None, None, "x", "y")))(
            model["b"], jnp.asarray(w))

    def sigma_c_shared_pole(self, config, wfns, model, meta, *, print_fn=print):
        """Σ_c,nm(k, ω) of the shared-pole model through the Σ owner's shared-pole route
        (``compute_sigma_c_mpa_omega_grid(sigma_w_model='shared_pole')``) with the in-memory
        model (``MemorySharedPoleModel``) and this basis's ``sector_context``: W(τ) by
        ``synthesize_shared_pole_parents`` on the sphere faces, then :meth:`sigma_kij`."""
        import hashlib
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        from common.collectives import device_put_process_local
        from common.units import RYD_TO_EV
        from distrib_la import gemm_plan
        from runtime.padding import pad_to_axis
        from .mpa.sigma import (MemorySharedPoleModel, SynthesisTau, WSynthesis,
                                compute_sigma_c_mpa_omega_grid, synthesize_shared_pole_parents)
        from .mpa.sigma_windows import shared_pole_intervals
        from .ppm_windows import sigma_regularization_for_config
        from .sigma_box_plan import resolve_sigma_box_cache_dir
        from .wavefunction_bundle import parent_sigma_operands
        mesh, m, K = self.mesh, int(self.screen.M), int(model["K"])
        place = lambda spec: jax.jit(lambda b: b[:, :, None, :], out_shardings=NamedSharding(mesh, spec))
        bX = place(P(None, "x", None, "y"))(model["b"])
        bY = place(P(None, "y", None, "x"))(model["b"])
        gemm = gemm_plan(mesh, m=m, k=K, n=m, nq=self.n_par, dtype=jnp.complex128, layout="face")
        replicated = NamedSharding(mesh, P())
        p2 = device_put_process_local(np.asarray(model["poles2"]), replicated)
        tables = self.sig_conv.tables()
        # The resident factors are this caller's live stage in the recipe's ledger; the window
        # executable's workspace is admitted beside them (SynthesisTau.admit).
        ledger = meta.shared_pole_capacity
        ledger.reserve("plane_wave.sigma.factors",
                       resident_bytes_per_rank=sum(int(a.addressable_shards[0].data.nbytes)
                                                   for a in (bX, bY)),
                       workspace_bytes_per_rank=0, concurrent_with=())
        ledger.live_stages = ("plane_wave.sigma.factors",)

        def synthesis(_reader, _ledger, frequencies, _schedule):
            def w_kernel(bx, by, poles2, intervals, load, e_ref, t, hole):
                del hole                                     # TRS model: no -q partner branch
                plus, _ = synthesize_shared_pole_parents(bx, by, poles2, intervals, e_ref, t,
                                                         mesh_xy=mesh, gemm=gemm, layout="face")
                return plus, load

            def window_operands(_space, indices, bounds):
                iv = shared_pole_intervals(frequencies, np.asarray(indices), np.asarray(bounds))
                return (bX, bY, p2, device_put_process_local(iv, replicated), tables)
            return WSynthesis(w_kernel, window_operands, lambda: (bX, bY), lambda _r=None: None, 0,
                              ("plane-wave", mesh, self.n_par, m, K), ordered=False)

        def tau_kernel(synth, sigma_axis):
            _, right_yr, _, right_proj, _, _ = parent_sigma_operands(wfns)
            right_proj = pad_to_axis(right_proj, sigma_axis, axis=3)
            kij = self.sigma_kij(sigma_axis)

            def spatial(xn, yr, xr, yn, energies, weight, reference, t, interactions):
                W_t, load = interactions
                return kij(xn, yr, xr, yn, energies, weight, reference, t, W_t, None, load)
            key = ("plane-wave", mesh, self._face_shape, int(sigma_axis.carrier), id(self))
            return SynthesisTau(spatial, synth, right_yr, right_proj, 0, "sigma.plane_wave.tau",
                                meta, key, (self.s.plan,))

        census = meta.shared_pole_census
        recipe = model["recipe"]
        identity = dict(iteration_id="plane_wave_oneshot",
                        hamiltonian=hashlib.sha256(np.asarray(self.s.enk).tobytes()).hexdigest(),
                        wavefunctions="plane-wave sphere", energies=census["energy_sha256"],
                        occupations=census["occupation_sha256"],
                        centroids=f"plane-wave sphere M={m}", recipe_hash=recipe["recipe_hash"],
                        gate_hash=recipe["gate_hash"])
        src = MemorySharedPoleModel(model["poles2"], model["counts"], identity=identity,
                                    factors=(bX, bY))
        context = dict(schedule=lambda _ledger: dict(status="PASS", route="plane-wave in-memory"),
                       synthesis=synthesis, tau_kernel=tau_kernel)
        xi = sigma_regularization_for_config(config)
        return compute_sigma_c_mpa_omega_grid(
            wfns, src, meta, mesh,
            omega_grid_ry=config.omega_grid_ry, efermi_ry=self.mu,
            regularization_width_ry=xi.resolved_ry,
            edge_factor=float(config.sigma.window_edge_factor),
            quadrature_eps=float(config.sigma.quadrature_eps),
            quadrature_cache_dir=resolve_sigma_box_cache_dir(config.sigma.quadrature_cache_dir,
                                                             config.input_dir),
            omega_grid_step_ry=float(config.sigma.omega_step_ev) / RYD_TO_EV,
            material_class="insulator", sigma_w_model="shared_pole", sector_context=context,
            print_fn=print_fn)

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
    if str(config.sigma.w_model) == "shared_pole":
        return _one_shot_shared_pole(args, mesh, system, gw, say, t_read, config, wfns, sigx,
                                     stage, stages)
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


def _diag_at_energies(body, system, gw, mesh):
    """Σ_c(k, n, ω) diagonal at the raw parents and its value at E_nk − μ (Ry)."""
    import jax
    rows = np.asarray(system.plan.parent_full_rows)
    diag = jax.jit(lambda a: jax.numpy.diagonal(a[:, rows], axis1=-2, axis2=-1))(body.sigma_c_kij)
    sc = _host(diag, mesh)[:, :, :gw.nb]                                  # (n_ω, n_par, nb) Ry
    om = np.asarray(body.omega_ry)
    x = system.enk - gw.mu
    at = np.array([[np.interp(x[k, n], om, sc[:, k, n].real) + 1j * np.interp(x[k, n], om, sc[:, k, n].imag)
                    for n in range(gw.nb)] for k in range(gw.n_par)])
    return sc, om, at


def complex_z_points(eta_ev):
    """Step-2 evaluation points (eV): imaginary axis, the damped line at the Σ η, off-axis."""
    imag = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    line = [0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0]
    off = [(2.0, 1.0), (5.0, 1.0), (5.0, 2.6), (10.0, 2.6), (10.0, 5.0), (20.0, 2.6), (20.0, 5.0),
           (30.0, 5.0)]
    pts = [("imag", complex(0.0, u)) for u in imag]
    pts += [("line", complex(w, eta_ev)) for w in line]
    pts += [("off", complex(a, b)) for a, b in off]
    return pts


def _w_errors(mesh, W, ref, g0):
    """Per (z, q): ‖W − W_ref‖_F/‖W_ref‖_F and |ΔW_00|/|W_ref,00| at G = 0 (q ≠ 0; the long-
    wavelength element that ⟨ρ_nn|W|ρ_nn⟩ reduces to for intraband pairs at small q)."""
    import jax
    import jax.numpy as jnp
    fro = jax.jit(lambda a, b: (jnp.sqrt(jnp.sum(jnp.abs(a - b) ** 2, axis=(-2, -1))),
                                jnp.sqrt(jnp.sum(jnp.abs(b) ** 2, axis=(-2, -1)))))
    d, n = (np.asarray(_host(x, mesh)) for x in fro(W, ref))
    q = np.arange(len(g0))
    head = jax.jit(lambda a: a[:, q, g0, g0])
    h, h0 = np.asarray(_host(head(W), mesh)), np.asarray(_host(head(ref), mesh))
    with np.errstate(divide="ignore", invalid="ignore"):
        head_rel = np.where(np.abs(h0) > 0, np.abs(h - h0) / np.abs(h0), np.nan)
    return d / n, head_rel, h0


def _exact_chi_points(gw, zs, tol, kinds):
    """χ(z_j) for the step-2 points (value only, at 2j of the returned stack).

    Points on the damped line at the Σ η take the MPA owner's damped-line contour rule
    (``PlaneWaveGW.line_chi``, tolerance 1e-8, all of them one rule): the response rule cannot
    fit a pole within η of the transition band past Re z ≈ 10 eV.  Every other point takes
    one response rule; one that does not fit at the recipe tolerance takes the first of
    1e-7 … 1e-4 that does.  The tolerance is returned per point."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    rows, tols, nodes = [], np.full(zs.size, np.nan), 0
    line = np.flatnonzero(kinds == "line")
    for i, z in enumerate(zs):
        if i in line:
            continue
        got = None
        for t in (tol, 1e-7, 1e-6, 1e-5, 1e-4):
            try:
                got = gw.response_rows(np.asarray([z]), rel_tol=t, group_size=1)
            except ValueError:
                continue
            tols[i] = t
            break
        if got is None:
            raise ValueError(f"exact chi: no response rule fits z = {z} Ry at tolerance <= 1e-4")
        for t_nodes, fwd, rev in got:
            F = np.zeros((2 * zs.size, t_nodes.size), np.complex128)
            R = np.zeros_like(F)
            F[2 * i:2 * i + 2], R[2 * i:2 * i + 2] = fwd, rev
            rows.append((t_nodes, F, R))
            nodes += int(t_nodes.size)
    acc = None
    for t_nodes, F, R in rows:
        part = gw.chi_nodes(t_nodes, F, both=False, reverse_rows=R)
        acc = part if acc is None else acc + part
    if line.size:
        chi_line, n_line = gw.line_chi(zs[line], rel_tol=1e-8)
        idx = np.asarray(2 * line)
        acc = jax.jit(lambda a, b: a.at[idx].set(b), donate_argnums=(0,),
                      out_shardings=NamedSharding(gw.mesh, P(None, None, "x", "y")))(acc, chi_line)
        tols[line] = 1e-8
        nodes += n_line
        del chi_line
    return acc, nodes, tols


def _one_shot_shared_pole(args, mesh, system, gw, say, t_read, config, wfns, sigx, stage, stages):
    """Shared-pole W on the sphere → Σ_c(ω) through the Σ owner; the model against the exact
    Dyson W^c(z) at complex z, with the 8-pole MPA of the same samples beside it."""
    import jax
    import jax.numpy as jnp
    meta = gw.sphere_meta(args.port_count)
    model = gw.shared_pole_model(config, wfns, meta, print_fn=say)
    stage("shared_pole_model")
    for row in model["gates"]:
        say("shared-pole parent", json.dumps(row))
    say(f"shared-pole K per parent {model['counts'].tolist()} (n = {int(gw.screen.axis.logical)}, "
        f"budget {model['recipe'].get('pole_budget')}), response nodes {model['response_nodes']}")
    recipe = model["recipe"]
    tol = float(recipe["bank_rule_tolerance"])
    # ---- convention checks of the response-rule χ against RSK5's validated minimax route
    u = 2.0 / _RY_EV
    chi_r, _ = gw.response_chi(np.asarray([1j * u]), rel_tol=tol)
    chi_m = gw.chi([u])
    rel = jax.jit(lambda a, b: jnp.sqrt(jnp.sum(jnp.abs(a - b) ** 2)) / jnp.sqrt(jnp.sum(jnp.abs(b) ** 2)))
    check_imag = float(_host(rel(chi_r[0], chi_m[0]), mesh))
    z0, dz = complex(5.0, 2.6) / _RY_EV, 1e-4
    zz = np.asarray([z0, z0 * (1 + dz), z0 * (1 - dz)])
    chi_d, _ = gw.response_chi(zz, rel_tol=tol)
    fd = (chi_d[2] - chi_d[4]) / ((zz[1] ** 2) - (zz[2] ** 2))
    check_ds = float(_host(rel(chi_d[1], fd), mesh))
    del chi_r, chi_m, chi_d, fd
    say(f"check: response-rule chi(2i eV) vs minimax chi rel {check_imag:.3e}; d/ds vs central "
        f"difference (5+2.6i eV, dz/z 1e-4) rel {check_ds:.3e}")
    stage("convention_checks")
    # ---- step 2: complex z
    eta_ev = float(config.sigma.regularization_ev)
    pts = [] if args.skip_complex_z else complex_z_points(eta_ev)
    complex_z = _complex_z(args, mesh, system, gw, say, config, wfns, model, pts, tol, eta_ev, rel) \
        if pts else None
    if complex_z is not None and jax.process_index() == 0:
        with open(args.out.replace(".json", "") + "_complex_z.json", "w") as f:
            json.dump(complex_z, f, indent=1)
    stage("complex_z")
    # ---- Σ_c through the shared-pole route
    body = gw.sigma_c_shared_pole(config, wfns, model, meta, print_fn=say)
    stage("sigma_c_tau_sweep")
    return _write_shared_pole_result(args, mesh, system, gw, say, t_read, config, model, sigx, body,
                                     stages, dict(response_vs_minimax_chi_2i_ev=check_imag,
                                                  ds_vs_central_difference=check_ds), complex_z)


def _complex_z(args, mesh, system, gw, say, config, wfns, model, pts, tol, eta_ev, rel):
    """Step 2: the model W^c(q, z) against the exact Dyson W^c(q, z) at complex z, with the
    8-pole MPA of the same basis beside it."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    zs = np.asarray([p[1] for p in pts]) / _RY_EV
    kinds = np.asarray([p[0] for p in pts])
    chi_z, nodes_z, tol_z = _exact_chi_points(gw, zs, tol, kinds)
    say("exact chi(z): rule tolerance per point " + " ".join(f"{t:.0e}" for t in tol_z)
        + f"; {nodes_z} Green pairs")
    # the two exact routes at one damped-line point: the damped-line rule vs a response rule
    j5 = int(np.flatnonzero((kinds == "line") & np.isclose(zs.real * _RY_EV, 5.0))[0])
    chi_5, _ = gw.response_chi(zs[j5:j5 + 1], rel_tol=1e-6)
    check_line = float(_host(rel(chi_5[0], chi_z[2 * j5]), mesh))
    del chi_5
    say(f"check: damped-line rule vs response rule (1e-6) chi at 5+{eta_ev:g}i eV rel {check_line:.3e}")
    value, _slope, _moments, H = gw.response_programs(config)
    W_ex = jax.jit(lambda *a: jnp.stack(a), out_shardings=NamedSharding(mesh, P(None, None, "x", "y")))(
        *[value(H, chi_z[2 * j]) for j in range(zs.size)])
    del chi_z
    gv = np.asarray(gw.w_par.gvecs)
    g0 = np.asarray([int(np.flatnonzero(np.all(gv[q] == 0, axis=1))[0]) for q in range(gw.n_par)])
    fro_sp, head_sp, h0 = _w_errors(mesh, gw.shared_pole_w(model, zs), W_ex, g0)
    mpa = None
    try:
        zm, chi_m = gw.mpa_samples(config, wfns, print_fn=say)
        Omega, B, cond = gw.mpa_poles(config, zm, chi_m)
        del chi_m
        s2 = jnp.asarray(zs ** 2)
        W_mpa = jax.jit(lambda O, B: jnp.einsum("pqmn,zpqmn->zqmn", 2.0 * O * B,
                                                1.0 / (s2[:, None, None, None, None] - O[None] ** 2)),
                        out_shardings=NamedSharding(mesh, P(None, None, "x", "y")))(Omega, B)
        fro_mpa, head_mpa, _ = _w_errors(mesh, W_mpa, W_ex, g0)
        mpa = dict(n_poles=int(config.mpa.n_poles), cond=float(cond),
                   fro_rel=fro_mpa.tolist(), head_rel=head_mpa.tolist())
        del W_mpa, Omega, B
    except Exception as err:                      # the contrast is optional; the refusal is recorded
        mpa = dict(error=f"{type(err).__name__}: {err}")
    del W_ex
    for j, (kind, zev) in enumerate(pts):
        say(f"z {kind:4s} {zev.real:6.2f}{zev.imag:+6.2f}i eV: shared pole Frob max/rms over q "
            f"{np.nanmax(fro_sp[j]):.2e}/{np.sqrt(np.nanmean(fro_sp[j] ** 2)):.2e}, head max "
            f"{np.nanmax(head_sp[j]):.2e}"
            + ("" if "fro_rel" not in mpa else
               f"; MPA Frob max {np.nanmax(mpa['fro_rel'][j]):.2e}, head max {np.nanmax(mpa['head_rel'][j]):.2e}"))
    return dict(points_ev=[[k, v.real, v.imag] for k, v in pts], eta_ev=eta_ev,
                response_nodes=nodes_z, rule_tolerance=tol_z.tolist(),
                q_frac=system.psi_par.frac.tolist(),
                head_exact_ry=[[abs(x) for x in r] for r in np.asarray(h0).tolist()],
                line_rule_vs_response_rule_5_eta=check_line,
                shared_pole=dict(fro_rel=fro_sp.tolist(), head_rel=head_sp.tolist()), mpa=mpa)


def _write_shared_pole_result(args, mesh, system, gw, say, t_read, config, model, sigx, body,
                              stages, checks, complex_z):
    import jax
    recipe = model["recipe"]
    sc, om, at = _diag_at_energies(body, system, gw, mesh)
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
               w_model="shared_pole",
               shared_pole=dict(K=model["counts"].tolist(), K_carrier=model["K"],
                                pole_budget=recipe.get("pole_budget"), n=int(gw.screen.axis.logical),
                                port_count=args.port_count,
                                widths=dict(imaginary=recipe["imaginary_width"],
                                            infinity=recipe["infinity_width"],
                                            line_cap=recipe["line_direction_cap"]),
                                gates=model["gates"], response_nodes=model["response_nodes"],
                                support_z_ev=[[v.real * _RY_EV, v.imag * _RY_EV] for v in model["points"]],
                                fit_ids=[int(i) for i in recipe["fit_ids"]],
                                held_ids=[int(i) for i in recipe["held_ids"]],
                                poles_ev=[(np.sqrt(model["poles2"][q, :c]) * _RY_EV).tolist()
                                          for q, c in enumerate(model["counts"])],
                                walls=model["walls"], hermiticity=gw.hermiticity),
               checks=checks, complex_z=complex_z,
               stages=stages, walls=walls, t_read_s=t_read, t_plans_s=gw.t_plans,
               chi_law=gw.chi_conv.describe(), sigma_law=gw.sig_conv.describe())
    for k in range(gw.n_par):
        say(f"k{k} Σ_c(E) re (eV): " + " ".join(f"{v:.4f}" for v in at[k].real * _RY_EV))
    say("walls:", json.dumps(walls))
    if jax.process_index() == 0:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
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
    ap.add_argument("--port-count", type=int, default=None,
                    help="shared-pole deck, diagnostic: size the recipe's direction widths and pole "
                         "budget from this port count instead of the sphere width")
    ap.add_argument("--skip-complex-z", action="store_true",
                    help="shared-pole deck: skip the model-vs-exact W^c(z) comparison")
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
