"""chi0 per-tau-step jit (the OOM bottleneck at Si 4x4x4 60 Ry mem=16 GB).

Mirrors the production construction in ``gw.w_isdf._get_chi_minimax_kernel``:
    Gv_k, Gc_k = build_G(...) with with_sharding_constraint
    Gv_R = ifftn(Gv_k)
    Gc_mR = fftn(Gc_k)
    chi_tau = einsum('Rambn,Rbnam->Rmn', Gc_mR, Gv_R)
    chi_R_acc = chi_R_acc + prefactor * chi_tau    (donated)

Input shapes (flat-k form):
    chi_R_acc:   (n_k, n_rmu, n_rmu)                P(None, 'y', 'x')
    psi_val_xn:  (n_k, n_s, n_rmu, n_b_v)           P(None, None, 'x', None)
    psi_val_yr:  (n_k, n_b_v, n_s, n_rmu)           P(None, None, None, 'y')
    psi_cond_yr: (n_k, n_b_c, n_s, n_rmu)           P(None, None, None, 'y')
    psi_cond_xn: (n_k, n_s, n_rmu, n_b_c)           P(None, None, 'x', None)
    enk_v:       (n_k, n_b_v)  replicated
    enk_c:       (n_k, n_b_c)  replicated
    tau, prefactor, vmax, cmin: scalars

Shard math (16 bytes/complex128):
    T_chi_acc  = 16 * n_k * n_rmu^2                        / (p_x * p_y)
    T_Gbuf     = 16 * n_k * n_s^2 * n_rmu^2                / (p_x * p_y)   # each Gv/Gc, Gv_R/Gc_mR
    T_psi_x    = 16 * n_k * n_s * n_rmu * n_b              / p_x           # psi_*_xn
    T_psi_y    = 16 * n_k * n_s * n_rmu * n_b              / p_y           # psi_*_yr

Handoff profile (nk=64, μ=2400, n_s=2, mesh=2x2) shows 4 live T_Gbuf at peak
(5.49 GiB each) plus 1 chi_R_acc (1.37 GiB); fit should recover β_Gbuf ≈ 4
and β_chi_acc ≈ 1.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..core import AotKernel, Knobs, MeshSpec, SysDims, register_kernel


# ---------------------------------------------------------------------------
# Dimensional volume primitives for this kernel.
#
# Derived from the allocation dump at
# ``runs/Si/C_probe_si444_60Ry/.../xla_dump/..._tau_step..-memory-usage-report.txt``
# (see chi0_mem16_handoff.md):
#
#   allocation 14 (preallocated-temp): 21.97 GiB = 4 × Gbuf (4 concurrent)
#   allocation 0 (chi_R_acc):           1.37 GiB = T_chi  (donated -> aliased)
#   allocation 1 (psi_cond_xn):        675 MiB   = T_psi  (one of four psi inputs)
#   allocation 2 (psi_cond_yr):        675 MiB   = T_psi
#   tail: psi_val_*  ~50 MiB each       (small; nb_v << nb_c in production)
#
# With chi_R_acc donated, fit should give β[chi] ≈ 0 and β[Gbuf] ≈ 4.
# The combined ``psi`` primitive sums the four input shards, so β[psi] ≈ 1.
# ---------------------------------------------------------------------------

_B = 16.0  # complex128 bytes per element

def _T_Gbuf(sys: SysDims, knobs: Knobs, mesh: MeshSpec) -> float:
    """Full (nk, s, μ, s, μ) Gv-family buffer on P(None,None,'x',None,'y')."""
    return _B * sys.n_k * (sys.n_s ** 2) * sys.n_rmu ** 2 / (mesh.p_x * mesh.p_y)

def _T_chi(sys: SysDims, knobs: Knobs, mesh: MeshSpec) -> float:
    """chi_R_acc (nk, μ, μ) on P(None,'y','x') — donated; expected β≈0."""
    return _B * sys.n_k * sys.n_rmu ** 2 / (mesh.p_x * mesh.p_y)

def _T_psi(sys: SysDims, knobs: Knobs, mesh: MeshSpec) -> float:
    """Sum of the four psi input-buffer shards (nb_v + nb_c, x and y copies).

    psi_*_xn are sharded P(None,None,'x',None) → 1/p_x.
    psi_*_yr are sharded P(None,None,None,'y') → 1/p_y.
    """
    nb = sys.n_b_v + sys.n_b_c
    return (_B * sys.n_k * sys.n_s * sys.n_rmu * nb
            * (1.0 / mesh.p_x + 1.0 / mesh.p_y))


@register_kernel
class Chi0TauStepKernel(AotKernel):
    name = "chi0_tau_step"
    SYSTEM_DIMS = ("n_k", "n_rmu", "n_s", "n_b_v", "n_b_c", "kgrid")
    KNOBS = ()  # no chunk knob in the current implementation
    PRIMITIVES = {
        "Gbuf": _T_Gbuf,
        "chi":  _T_chi,
        "psi":  _T_psi,
    }

    # Specs on the flat-k form; the 3D form inside make_flat_k_*fftn is
    # built by prepending (None, None, None) to these.
    _SPEC_CHI_ACC = P(None, "y", "x")
    _SPEC_PSI_XN = P(None, None, "x", None)
    _SPEC_PSI_YR = P(None, None, None, "y")
    _SPEC_ENK    = P()

    def build_specs(self, sys: SysDims, knobs: Knobs, mesh: Mesh):
        nk, mu, ns = sys.n_k, sys.n_rmu, sys.n_s
        nbv, nbc = sys.n_b_v, sys.n_b_c

        def sds(shape, spec):
            return jax.ShapeDtypeStruct(
                shape, jnp.complex128, sharding=NamedSharding(mesh, spec))

        def sdf(shape, spec):
            return jax.ShapeDtypeStruct(
                shape, jnp.float64, sharding=NamedSharding(mesh, spec))

        return (
            sds((nk, mu, mu),       self._SPEC_CHI_ACC),
            sds((nk, ns, mu, nbv),  self._SPEC_PSI_XN),
            sds((nk, nbv, ns, mu),  self._SPEC_PSI_YR),
            sds((nk, nbc, ns, mu),  self._SPEC_PSI_YR),
            sds((nk, ns, mu, nbc),  self._SPEC_PSI_XN),
            sdf((nk, nbv),          self._SPEC_ENK),
            sdf((nk, nbc),          self._SPEC_ENK),
            jax.ShapeDtypeStruct((), jnp.float64),  # tau
            jax.ShapeDtypeStruct((), jnp.complex128),  # prefactor
            jax.ShapeDtypeStruct((), jnp.float64),  # vmax
            jax.ShapeDtypeStruct((), jnp.float64),  # cmin
        )

    def build_callable(self, sys: SysDims, knobs: Knobs, mesh: Mesh):
        """Construct the _tau_step exactly as production does.

        NOTE: duplicates the construction pattern in
        ``gw.w_isdf._get_chi_minimax_kernel`` because that factory returns
        only ``_chi_kernel``, not ``_tau_step`` directly.  If production
        changes its sharding specs, this must update in lockstep — the
        validation point (gamma calibration vs runtime) will catch drift.
        """
        from common.fft_helpers import make_flat_k_ifftn, make_flat_k_fftn
        from gw.greens_function_kernel import build_G as _build_G_mm

        kgrid = sys.kgrid

        # 7-D specs on the 3-D-form array (nkx, nky, nkz, s, μ_X, s, μ_Y)
        _Gv_spec = P(None, None, None, None, "x", None, "y")
        _Gc_spec = P(None, None, None, None, "y", None, "x")
        _chi_spec_3d = P(None, None, None, "y", "x")

        # 5-D output specs on the flat-k form (nk, s, μ, s, μ) and chi (nk, μ, μ)
        _Gv_out_flatk = P(None, None, "x", None, "y")
        _Gc_out_flatk = P(None, None, "y", None, "x")
        _chi_R_spec = P(None, "y", "x")

        _Gv_ifftn       = make_flat_k_ifftn(mesh, kgrid, _Gv_spec, norm="ortho")
        _Gc_fftn        = make_flat_k_fftn (mesh, kgrid, _Gc_spec, norm="ortho")

        def _tau_step(
            chi_R_acc, psi_val_xn, psi_val_yr, psi_cond_yr, psi_cond_xn,
            enk_v, enk_c, tau_scalar, prefactor_scalar, vmax, cmin,
        ):
            phases_v = jnp.exp(-tau_scalar * (vmax - enk_v))
            phases_c = jnp.exp(-tau_scalar * (enk_c - cmin))
            Gv_k = jax.lax.with_sharding_constraint(
                _build_G_mm(psi_val_xn, psi_val_yr, phases=phases_v),
                NamedSharding(mesh, _Gv_out_flatk))
            Gc_k = jax.lax.with_sharding_constraint(
                _build_G_mm(psi_cond_xn, psi_cond_yr, phases=phases_c),
                NamedSharding(mesh, _Gc_out_flatk))
            Gv_k = jnp.conj(Gv_k)
            Gc_k = jnp.conj(Gc_k)
            Gv_R = _Gv_ifftn(Gv_k)
            Gc_mR = _Gc_fftn(Gc_k)
            chi_tau = jax.lax.with_sharding_constraint(
                jnp.einsum("Rambn,Rbnam->Rmn", Gc_mR, Gv_R, optimize=True),
                NamedSharding(mesh, _chi_R_spec))
            return chi_R_acc + prefactor_scalar * chi_tau

        # donate_argnums=(0,) matches production — AOT memory_analysis picks
        # up the aliasing and reports it as alias_size_in_bytes.
        return partial(jax.jit, donate_argnums=(0,))(_tau_step)
