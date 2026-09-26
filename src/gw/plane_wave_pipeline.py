"""Plane-wave (ISDF-free) GW stages, first cut: ψ(G) store → χ₀(τ) → W_q(G, G') → Σ_x.

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
    psi: np.ndarray           # (n_par, nb, ns, width) ψ(G) at the parents, unit norm
    enk: np.ndarray           # (n_par, nb) Ry
    ecut: float               # max |k+G|² of the ψ spheres (Ry)
    ns: int
    cut_gap: float            # min over k of E[nb] − E[nb−1] (Ry): a multiplet split at the band cut is ~0


def _pad_width(g, width):
    return np.pad(g, ((0, 0), (0, width - g.shape[1]), (0, 0)))


def open_plane_wave_system(wfn_path: str, mesh, *, nb: int) -> PlaneWaveSystem:
    """Read ψ(G), E and the symmetry tables once (the loader's own doors)."""
    from file_io import WfnLoader
    from vcoul import CoulombGeometry
    from .mixed_basis_pair_convolution import SphereSet
    with WfnLoader(wfn_path, backend="eager") as w:
        sym = w.symmetry()
        kgrid = tuple(int(v) for v in w.kgrid)
        box = tuple(int(v) for v in w.fft_grid)
        geom = CoulombGeometry.from_wfn(w)
        kf = np.asarray(w.kvecs(k="full_bz"))
        gf, nf = np.asarray(w.gvecs(k="full_bz")), np.asarray(w.ngk_valid(k="full_bz"))
        kp = np.asarray(w.kvecs(k=sym.parent_k_domain))
        gp, npar = np.asarray(w.gvecs(k=sym.parent_k_domain)), np.asarray(w.ngk_valid(k=sym.parent_k_domain))
        psi = np.asarray(w.load(bands=(0, int(nb)), k=sym.parent_k_domain))[:, :int(nb)]
        ns = int(w.nspinor)
        if sym.parent_k_domain != "ibz":
            raise ValueError(f"plane_wave_pipeline: parent_k_domain {sym.parent_k_domain!r}; want 'ibz'")
        e_all = np.asarray(w.energies, np.float64)[0]
        enk = e_all[:, :int(nb)]
        cut_gap = (float(np.min(e_all[:, int(nb)] - e_all[:, int(nb) - 1]))
                   if e_all.shape[1] > int(nb) else float("inf"))
    width = max(gf.shape[1], gp.shape[1], psi.shape[-1])
    gf, gp = _pad_width(gf, width), _pad_width(gp, width)
    psi = np.pad(psi, ((0, 0), (0, 0), (0, 0), (0, width - psi.shape[-1])))
    bvec = np.asarray(geom.bvec, np.float64)
    ecut = max(float(np.max(np.sum(((kf[i][None] + gf[i, :nf[i]]) @ bvec) ** 2, axis=1)))
               for i in range(len(nf)))
    n_sp = int(np.asarray(sym.sym_matrices).shape[0])
    sidx = np.asarray(sym.sym_idx_k, np.int32)
    spin = (np.asarray(sym.spinor_action(sidx, nspinor=2)) if ns == 2
            else np.ones((len(sidx), 1, 1), np.complex128))
    plan = SimpleNamespace(irr_idx=np.asarray(sym.irr_idx_k, np.int32), sym_idx=sidx,
                           spin_action_full=spin, k_parent_frac=kp, n_sym_spatial=n_sp,
                           spatial_ops=np.asarray(sym.sym_matrices)[:n_sp],
                           translations=np.asarray(sym.translations)[:n_sp],
                           mesh_xy=mesh, n_full=len(sidx))
    return PlaneWaveSystem(sym=sym, kgrid=kgrid, fft_grid=box, geometry=geom,
                           psi_full=SphereSet(gf, nf, kf), psi_par=SphereSet(gp, npar, kp),
                           plan=plan, psi=psi, enk=enk, ecut=ecut, ns=ns, cut_gap=cut_gap)


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
        n_par, nb, ns, width = s.psi.shape
        box, bvec = s.fft_grid, np.asarray(s.geometry.bvec, np.float64)
        self.n_r = int(np.prod(box))
        self.omega_cell = float(s.geometry.cell_volume)
        # ---- ψ(G) in the r_μ face arrays: μ is the sphere slot, one carrier for every k
        self.M = int(padded_axis(width, self.P, name="psi sphere slots").carrier)
        self.nb_c = int(padded_axis(nb, self.P, name="bands").carrier)
        x = np.zeros((n_par, self.nb_c, ns, self.M), np.complex128)
        x[:, :nb, :, :width] = s.psi
        nmu = NamedSharding(mesh, P(None, "x", None, "y"))
        mun = NamedSharding(mesh, P(None, None, "x", "y"))
        psi_nmu = jax.device_put(x, nmu)
        psi_mun = jax.device_put(np.ascontiguousarray(np.transpose(x, (0, 2, 3, 1))), mun)
        # the parent carrier's roles (gw.wavefunction_bundle.parent_sigma_operands): ψ_mun is the
        # direct operand of the G builder, ψ_nmu the conjugated one and the projection face
        self.enk = np.zeros((n_par, self.nb_c))
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
        """G_k(p, p') at the parents with band weights ``(n_par, nb)``: the G builder, unchanged."""
        import jax.numpy as jnp
        from .greens_function_kernel import build_G_parents
        w = np.zeros((self.s.psi.shape[0], self.nb_c))
        w[:, :self.nb] = weights
        return build_G_parents(self.carrier.psi_mun, self.carrier.psi_nmu, phases=jnp.asarray(w),
                               layout="face", gemm=self._gemm, k_unfold_plan=self.s.plan,
                               real_weights=True).G

    def _timed(self, name, fn, *a):
        import jax
        t0 = time.perf_counter()
        out = fn(*a)
        jax.block_until_ready(out)
        self.walls.setdefault(name, []).append(time.perf_counter() - t0)
        return out

    # ------------------------------------------------------------------ χ₀ → χ(iω)
    def chi(self, omegas_ry, *, target_error: float = 1e-8):
        """χ_q(G, G'; iω_j) at the k-parent q rows, ``(n_z, n_par, M_χ, M_χ)``; the rule per
        sample is ``minimax_screening``'s (ω = 0: 1/x; else x/(x² + ω²))."""
        import jax
        from jax.sharding import NamedSharding, PartitionSpec as P
        from .minimax_screening import (solve_laplace_minimax_imag_interval,
                                        solve_laplace_minimax_interval)
        from .plane_wave_screening import accumulate_chi
        e = self.s.enk - self.mu
        cond = np.arange(self.nb) >= self.nval
        n_z = len(omegas_ry)
        Mc = int(self.screen.M)
        acc = jax.device_put(np.zeros((n_z, self.w_par.n, Mc, Mc), np.complex128),
                             NamedSharding(self.mesh, P(None, None, "x", "y")))
        self.rules = []
        for j, om in enumerate(omegas_ry):
            rule = (solve_laplace_minimax_interval(self.x_min, self.x_max, target_error=target_error)
                    if om == 0.0 else solve_laplace_minimax_imag_interval(
                        self.x_min, self.x_max, float(om), target_error=target_error))
            # 1/x rule: χ(0) = 2·Σ 1/x (both orientations); x/(x²+ω²) rule the same per orientation
            self.rules.append(dict(omega_ry=float(om), n_tau=len(rule.tau),
                                   max_error=float(rule.max_error)))
            wsel = np.zeros(n_z)
            for t, a in zip(np.asarray(rule.tau), np.asarray(rule.alpha)):
                Gc = self._timed("green", self.green, np.where(cond, np.exp(-e * t), 0.0))
                Gv = self._timed("green", self.green, np.where(~cond, np.exp(e * t), 0.0))
                for L, R in ((Gc, Gv), (Gv, Gc)):
                    X = self._timed("chi_pair_conv", self.chi_conv, L, R)
                    wsel[:] = 0.0
                    wsel[j] = float(a)
                    acc = accumulate_chi(acc, X, wsel, scale=self.chi_scale, mesh=self.mesh)
                    del X
                del Gc, Gv
        return acc

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
        V = jax.device_put(v[:, :, None] * np.eye(v.shape[1])[None],
                           NamedSharding(self.mesh, P(None, "x", "y")))
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
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    import jax
    from jax.sharding import Mesh
    n = len(jax.devices())
    px = int(np.sqrt(n))
    mesh = Mesh(np.asarray(jax.devices()).reshape(px, n // px), ("x", "y"))
    say = (lambda *a: print("[pw-pipeline]", *a, flush=True)) if jax.process_index() == 0 else (lambda *a: None)
    t0 = time.perf_counter()
    system = open_plane_wave_system(args.wfn, mesh, nb=args.nb)
    t_read = time.perf_counter() - t0
    gw = PlaneWaveGW(mesh, system, nval=args.nval,
                     screened_coulomb_cutoff=args.screened_coulomb_cutoff, wedge=args.wedge)
    say(f"system: kgrid {system.kgrid}, box {system.fft_grid} (N_r {gw.n_r}), n_par {system.psi.shape[0]}, "
        f"nb {gw.nb}, ns {gw.ns}, ψ width {system.psi.shape[-1]} (carrier {gw.M}), χ sphere "
        f"{gw.w_par.width} (carrier {gw.screen.M}), gap {gw.x_min * _RY_EV:.4f} eV, band-cut gap "
        f"{system.cut_gap * _RY_EV:.4f} eV; "
        f"read {t_read:.1f} s, plans {gw.t_plans:.1f} s")
    say(gw.chi_conv.describe())
    say(gw.sig_conv.describe())
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
               psi_width=int(system.psi.shape[-1]), psi_carrier=gw.M, chi_width=int(gw.w_par.width),
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
