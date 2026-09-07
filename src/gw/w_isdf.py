"""Screen charge and current response with typed parent transport and q-IBZ operators."""
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.experimental import compilation_cache as jax_compilation_cache
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import numpy as np

from common import Meta, jax_profile
from common.bispinor_init import NO_PAIR_DIRAC_CURRENT_MODEL
from common.collectives import device_put_process_local
from common.jax_compile_cache import ensure_jax_compile_cache
from runtime.padding import (
    authenticate_padded_axis,
    padded_mu_axis,
    solve_at_logical,
)
from .efermi import (OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
                     band_in_occupation_window, occupation_weight_floor)
from .minimax_screening import MinimaxNodes


# ============================================================================
# Cache and sharding registry
# ============================================================================

_chi_minimax_kernel_cache: dict = {}
_w_solve_cache: dict = {}

# The static metal fallback streams one ordered band-pair tile at a time.
# Width 8 left production Bi executing 529 tiny scan steps per q row.  Width
# 32 reduces that to 36 steps for its 184-band physical window; the documented
# transient is 5.8 GiB/rank on the P36 geometry, well below an 80 GiB A100.
# This remains a compile-time performance choice, not a physics/input dial.
_STATIC_FRACTIONAL_PAIR_TILE = 32


def _complete_static_vertex_orientations(forward_R, reverse_R=None):
    r"""Return both ordered Hermitian-vertex orientations in R space.

    ``forward_R`` has endpoint axes ``(mu_A,mu_B)``.  For two different
    Hermitian vertices, ``reverse_R`` is the reversed ordered contribution
    in its natural ``(mu_B,mu_A)`` orientation.  Its dagger maps it back to
    the forward endpoint order before addition::

        forward_R + reverse_R^dagger

    Charge is the same-vertex special case: its natural reverse is
    ``swapaxes(forward_R)`` and the expression reduces exactly to the
    incumbent ``forward_R + conj(forward_R)`` completion.  Replacing either
    form by ``2*forward_R`` is valid only in a real gauge and is wrong for a
    complex broken-time-reversal wavefunction.

    Keep this completion before the final R-to-q FFT.  It is transition
    algebra, not a post-hoc q symmetrization, and preserves sharding
    elementwise.
    """
    if reverse_R is None:
        # Preserve the incumbent scalar graph and arithmetic order exactly.
        return forward_R + jnp.conj(forward_R)
    return forward_R + jnp.conj(jnp.swapaxes(reverse_R, -1, -2))


# ============================================================================
# χ₀ kernel — minimax quadrature
# ============================================================================

def _get_chi_minimax_kernel(mesh_xy: Mesh, kgrid: tuple[int, int, int],
                            n_out: int = 1, *,
                            complex_contour: bool = False,
                            layout: str = "face", face_shape=None,
                            right_face_shape=None,
                            vertex_pairs=None,
                            k_unfold_plan=None):
    """Build the face or parent minimax response with vertices applied after unfold."""
    nkx, nky, nkz = kgrid
    nk = nkx * nky * nkz
    n_out = int(n_out)
    # ffi_dial_key(): the make_flat_k_fftn factories below read
    # LORRAX_FFT_FFI at FACTORY time, so the dials must be part of this
    # cache key or a mid-process flag flip serves the stale backend
    # (flat-k FFT service contract, docs/dev/flat_k_fft_service.md).
    from ffi import ffi_dial_key
    complex_contour = bool(complex_contour)
    if layout not in ("face", "axis"):
        raise ValueError(
            f"_get_chi_minimax_kernel: layout must be 'face', "
            f"got {layout!r}")
    vertex_classes = None
    if vertex_pairs is not None:
        vertex_pairs = tuple(tuple(int(v) for v in pair) for pair in vertex_pairs)
        if not vertex_pairs or any(len(pair) != 2 or any(v not in range(4) for v in pair)
                                   for pair in vertex_pairs):
            raise ValueError("vertex_pairs must name nonempty Lorentz pairs in 0..3.")
        vertex_classes = tuple(tuple(v == 0 for v in pair) for pair in vertex_pairs)
        if len(set(vertex_classes)) != 1 or n_out != 1:
            raise ValueError("Vertex outputs must share one family pair and one static rule.")
    cache_key = (id(mesh_xy), kgrid, ffi_dial_key(), n_out,
                 complex_contour, layout, face_shape, right_face_shape,
                 vertex_classes, (tuple(id(p) for p in k_unfold_plan)
                               if isinstance(k_unfold_plan, tuple) else id(k_unfold_plan)))
    if cache_key in _chi_minimax_kernel_cache:
        return _chi_minimax_kernel_cache[cache_key]

    if face_shape is None:
        raise ValueError(
            "_get_chi_minimax_kernel(layout='face') requires "
            "face_shape=(nk, nb_full, n_rmu, nspinor)")
    kernel = _get_chi_minimax_kernel_face(
        mesh_xy, kgrid, nk, n_out, complex_contour, face_shape,
        right_face_shape=right_face_shape,
        vertex_pairs=vertex_pairs,
        k_unfold_plan=k_unfold_plan, layout=layout)
    _chi_minimax_kernel_cache[cache_key] = kernel
    return kernel


def _contract_chi_vertices(Gv_R, Gc_R, operands, identities, complex_contour):
    """Contract Hermitian endpoint vertices and complete their ordered orientations."""
    from common.gamma_matrices import gamma_double_contract
    left_identity, right_identity = identities
    reverse = None
    if operands is None or (left_identity and right_identity):
        forward = jnp.einsum('Rambn,Rambn->Rmn', Gc_R, jnp.conj(Gv_R), optimize=True)
    else:
        perm_l, phase_l, perm_r, phase_r = operands
        forward = gamma_double_contract(
            jnp.conj(Gv_R), Gc_R,
            perm_L=None if left_identity else perm_l,
            phase_L=None if left_identity else phase_l,
            perm_R=None if right_identity else perm_r,
            phase_R=None if right_identity else jnp.conj(phase_r),
            spin_axes=(1, 3))
        Gv_R_ba = jnp.transpose(Gv_R, (0, 3, 4, 1, 2))
        Gc_R_ba = jnp.transpose(Gc_R, (0, 3, 4, 1, 2))
        reverse = gamma_double_contract(
            jnp.conj(Gv_R_ba), Gc_R_ba,
            perm_L=None if right_identity else perm_r,
            phase_L=None if right_identity else jnp.conj(phase_r),
            perm_R=None if left_identity else perm_l,
            phase_R=None if left_identity else phase_l,
            spin_axes=(1, 3))
    return (forward if complex_contour else
            _complete_static_vertex_orientations(forward, reverse))


def _get_chi_minimax_kernel_face(mesh_xy, kgrid, nk, n_out, complex_contour,
                                 face_shape, *, right_face_shape=None,
                                 vertex_pairs=None,
                                 k_unfold_plan=None, layout="face"):
    """Build masked valence/conduction Green functions and integrate both response orientations."""
    from common.fft_helpers import make_flat_k_fftn
    from distrib_la import gemm_plan
    from .wavefunction_bundle import (
        G_FFT7D_SPEC as _G_spec,
        G_FLATK_SPEC as _G_out_flatk,
        CHI_Q_SPEC as _chi_spec,
        CHI_R_SPEC as _chi_R_spec,
    )
    from .greens_function_kernel import build_G_tau
    from common.wfn_layout import psi_specs
    _psi_nmu_spec, _psi_mun_spec = psi_specs(layout)

    nk_shape, nb_full, n_rmu_left, ns = (int(v) for v in face_shape)
    if right_face_shape is None:
        right_face_shape = face_shape
    nk_right, nb_right, n_rmu_right, ns_right = (
        int(v) for v in right_face_shape)
    plans = (k_unfold_plan if isinstance(k_unfold_plan, tuple)
             else (k_unfold_plan, k_unfold_plan))
    left_plan, right_plan = plans
    expected_input_nk = nk if left_plan is None else left_plan.n_parent
    if (nk_shape, nk_right, nb_right, ns_right) != (
            expected_input_nk, expected_input_nk, nb_full, ns):
        raise ValueError("chi endpoints must share parent k, band and spin extents.")
    if any(p is not None and p.n_full != nk for p in plans):
        raise ValueError("chi parent plan full-k extent disagrees with kgrid.")
    if (left_plan is None) != (right_plan is None):
        raise ValueError("Both chi endpoints must carry parent plans together.")
    n_rmu_out_left, n_rmu_out_right = n_rmu_left, n_rmu_right
    paired = isinstance(k_unfold_plan, tuple)
    if paired and any(not np.array_equal(getattr(left_plan, name), getattr(right_plan, name))
                      for name in ("irr_idx", "sym_idx", "k_parent_frac", "spin_action_full")):
        raise ValueError("chi family plans disagree on the raw-parent k action.")

    _Gv_fftn        = make_flat_k_fftn(mesh_xy, kgrid, _G_spec,   norm='ortho')
    _Gc_fftn        = make_flat_k_fftn(mesh_xy, kgrid, _G_spec,   norm='ortho')
    _chi_fftn_local = make_flat_k_fftn(mesh_xy, kgrid, _chi_spec, norm='ortho')

    _rep0 = P()
    _rep1 = P(None)
    _rep2 = P(None, None)
    _G_k_shard = NamedSharding(mesh_xy, _G_out_flatk)
    _chi_R_shard = NamedSharding(mesh_xy, _chi_R_spec)
    _psi_mun_shard = NamedSharding(mesh_xy, _psi_mun_spec)
    _psi_nmu_shard = NamedSharding(mesh_xy, _psi_nmu_spec)

    # ONE planned GEMM, built here (eagerly, once) and shared by every Gv
    # and Gc build this kernel ever does — see distrib_la.gemm_plan's own
    # "hoist this call out of every per-k/per-tau loop" instruction.
    g_plan = gemm_plan(
        mesh_xy, m=n_rmu_left * ns, k=nb_full,
        n=n_rmu_right * ns, nq=nk if paired else expected_input_nk,
        dtype=jnp.complex128, layout=layout)
    if paired:
        from common.shard_map import shard_map
        def unfold_pair(psi_left, psi_right):
            return (left_plan.unfold_face(psi_left, spin_axis=1, mu_axis=2,
                                         mesh_axis="x"),
                    right_plan.unfold_face(psi_right, spin_axis=2, mu_axis=3,
                                          mesh_axis="y"))
        unfold_pair = shard_map(unfold_pair, mesh=mesh_xy,
            in_specs=(_psi_mun_spec, _psi_nmu_spec),
            out_specs=(_psi_mun_spec, _psi_nmu_spec), check_vma=False)

    @partial(jax.jit,
             in_shardings=(_psi_mun_shard, _psi_nmu_shard,
                            NamedSharding(mesh_xy, _rep2),   # mask_v
                            NamedSharding(mesh_xy, _rep2),   # mask_c
                            NamedSharding(mesh_xy, _rep2),   # enk_full
                            NamedSharding(mesh_xy, _rep0),
                            NamedSharding(mesh_xy, _rep0),
                            NamedSharding(mesh_xy, _rep0)),
             out_shardings=(_G_k_shard, _G_k_shard))
    def _build_Gv_Gc(psi_mun_left, psi_nmu_right,
                    mask_v, mask_c, enk_full,
                    tau_scalar, vmax, cmin):
        if paired:
            psi_mun_left, psi_nmu_right = unfold_pair(
                psi_mun_left, psi_nmu_right)
            rows = jnp.asarray(left_plan.irr_idx)
            mask_v, mask_c, enk_full = (jnp.take(v, rows, axis=0)
                                       for v in (mask_v, mask_c, enk_full))
        t_c = jnp.conj(tau_scalar) if complex_contour else tau_scalar
        Gv_k = jax.lax.with_sharding_constraint(
            build_G_tau(psi_mun_left, psi_nmu_right, enk_full,
                       -tau_scalar, e_ref=vmax,
                       mask=mask_v, layout=layout, gemm=g_plan,
                       k_unfold_plan=None if paired else k_unfold_plan),
            _G_k_shard)
        Gc_k = jax.lax.with_sharding_constraint(
            build_G_tau(psi_mun_left, psi_nmu_right, enk_full,
                       t_c, e_ref=cmin,
                       mask=mask_c, layout=layout, gemm=g_plan,
                       k_unfold_plan=None if paired else k_unfold_plan),
            _G_k_shard)
        return jnp.conj(Gv_k), jnp.conj(Gc_k)

    _nodes_shard = MinimaxNodes(
        t=NamedSharding(mesh_xy, _rep1),
        alpha=NamedSharding(mesh_xy, _rep1 if n_out == 1 else P()),
    )

    def _finish_chi(value):
        return _chi_fftn_local(value)

    if n_out >= 2:
        _chi_R_out = tuple(_chi_R_shard for _ in range(n_out))

        @partial(jax.jit,
                 in_shardings=(_nodes_shard, _psi_mun_shard, _psi_nmu_shard,
                                NamedSharding(mesh_xy, _rep2),
                                NamedSharding(mesh_xy, _rep2),
                                NamedSharding(mesh_xy, _rep2),
                                NamedSharding(mesh_xy, _rep0),
                                NamedSharding(mesh_xy, _rep0)),
                 out_shardings=_chi_R_out,
                 static_argnums=())
        def minimax_tau_integrate_chi_multi(
            nodes, psi_mun, psi_nmu, mask_v, mask_c, enk_full, vmax, cmin,
        ):
            zero = jax.lax.with_sharding_constraint(
                jnp.zeros((nk, n_rmu_out_left, n_rmu_out_right),
                          dtype=jnp.complex128),
                _chi_R_shard)
            acc0 = tuple(zero for _ in range(n_out))

            def _body(accs, xs):
                t_scalar, alpha_col = xs
                tau_kernel = (t_scalar if complex_contour else
                              jnp.real(t_scalar).astype(jnp.float64))
                Gv_k, Gc_k = _build_Gv_Gc(psi_mun, psi_nmu, mask_v, mask_c,
                                          enk_full, tau_kernel, vmax, cmin)
                Gv_R = _Gv_fftn(Gv_k)
                Gc_R = _Gc_fftn(Gc_k)
                chi_tau = jax.lax.with_sharding_constraint(
                    jnp.einsum('Rambn,Rambn->Rmn',
                               Gc_R, jnp.conj(Gv_R), optimize=True),
                    _chi_R_shard)
                if not complex_contour:
                    chi_tau = _complete_static_vertex_orientations(chi_tau)
                return tuple(a + alpha_col[i] * chi_tau
                             for i, a in enumerate(accs)), None

            final_R, _ = jax.lax.scan(
                _body, acc0, (nodes.t, jnp.transpose(nodes.alpha)), unroll=1)
            return tuple(_finish_chi(f) for f in final_R)

        return minimax_tau_integrate_chi_multi

    identities = ((True, True) if vertex_pairs is None else
                  tuple(v == 0 for v in vertex_pairs[0]))
    n_vertices = 1 if vertex_pairs is None else len(vertex_pairs)

    def _single_impl(
        nodes, psi_mun, psi_nmu, mask_v, mask_c, enk_full, vmax, cmin,
        vertex_operands,
    ):
        stack_shard = NamedSharding(mesh_xy, P(None, None, 'x', 'y'))
        zero = jax.lax.with_sharding_constraint(
            jnp.zeros((n_vertices, nk, n_rmu_out_left, n_rmu_out_right),
                      dtype=jnp.complex128), stack_shard)
        tables = (None if vertex_pairs is None else
                  tuple(jnp.stack(values) for values in zip(*vertex_operands)))

        def _body(accumulators, xs):
            t_scalar, alpha_scalar = xs
            tau_kernel = (t_scalar if complex_contour else
                          jnp.real(t_scalar).astype(jnp.float64))
            Gv_k, Gc_k = _build_Gv_Gc(psi_mun, psi_nmu, mask_v, mask_c,
                                      enk_full, tau_kernel, vmax, cmin)
            Gv_R, Gc_R = _Gv_fftn(Gv_k), _Gc_fftn(Gc_k)

            def vertex_step(index, acc):
                operands = None if tables is None else tuple(table[index] for table in tables)
                chi_tau = _contract_chi_vertices(
                    Gv_R, Gc_R, operands, identities, complex_contour)
                chi_tau = jax.lax.with_sharding_constraint(chi_tau, _chi_R_shard)
                previous = jax.lax.dynamic_index_in_dim(acc, index, axis=0, keepdims=False)
                value = previous + alpha_scalar * chi_tau
                return jax.lax.dynamic_update_index_in_dim(acc, value, index, axis=0)

            return jax.lax.fori_loop(0, n_vertices, vertex_step, accumulators, unroll=1), None

        final_R, _ = jax.lax.scan(
            _body, zero, (nodes.t, nodes.alpha), unroll=1)
        return tuple(_finish_chi(final_R[index]) for index in range(n_vertices))

    _base_in = (_nodes_shard, _psi_mun_shard, _psi_nmu_shard,
                NamedSharding(mesh_xy, _rep2),
                NamedSharding(mesh_xy, _rep2),
                NamedSharding(mesh_xy, _rep2),
                NamedSharding(mesh_xy, _rep0),
                NamedSharding(mesh_xy, _rep0))

    if vertex_pairs is not None:
        vertex_shard = tuple(tuple(NamedSharding(mesh_xy, P()) for _ in range(4))
                             for _ in vertex_pairs)

        @partial(jax.jit, in_shardings=(*_base_in, vertex_shard),
                 out_shardings=(_chi_R_shard,) * n_vertices)
        def minimax_tau_integrate_chi_vertex(
            nodes, psi_mun, psi_nmu, mask_v, mask_c, enk_full, vmax, cmin,
            vertex_operands,
        ):
            """Share each Green/FFT pair across the ordered vertices of one family class."""
            return _single_impl(nodes, psi_mun, psi_nmu, mask_v, mask_c,
                                enk_full, vmax, cmin, vertex_operands)
        return minimax_tau_integrate_chi_vertex

    @partial(jax.jit, in_shardings=_base_in, out_shardings=_chi_R_shard)
    def minimax_tau_integrate_chi(
        nodes, psi_mun, psi_nmu, mask_v, mask_c, enk_full, vmax, cmin,
    ):
        return _single_impl(
            nodes, psi_mun, psi_nmu, mask_v, mask_c, enk_full,
            vmax, cmin, (None,))[0]

    return minimax_tau_integrate_chi


def _get_chi_fractional_contour_kernel(
    mesh_xy: Mesh, kgrid: tuple[int, int, int], n_out: int,
    *, layout: str = "face", face_shape=None, k_unfold_plan=None,
):
    """Build the face or parent retarded response on one positive-time sweep."""
    from ffi import ffi_dial_key

    grid = tuple(int(n) for n in kgrid)
    n_out = int(n_out)
    if n_out < 1:
        raise ValueError("fractional contour chi0 requires at least one output")
    if layout not in ("face", "axis"):
        raise ValueError(
            f"_get_chi_fractional_contour_kernel: layout must be 'face' "
            f"got {layout!r}")
    cache_key = ("fractional_contour", id(mesh_xy), grid, ffi_dial_key(),
                 n_out, layout, face_shape, id(k_unfold_plan))
    if cache_key in _chi_minimax_kernel_cache:
        return _chi_minimax_kernel_cache[cache_key]

    if face_shape is None:
        raise ValueError(
            "_get_chi_fractional_contour_kernel(layout='face') requires "
            "face_shape=(nk, nb_full, n_rmu, nspinor)")
    kernel = _get_chi_fractional_contour_kernel_face(
        mesh_xy, grid, n_out, face_shape, k_unfold_plan=k_unfold_plan, layout=layout)
    _chi_minimax_kernel_cache[cache_key] = kernel
    return kernel


def _get_chi_fractional_contour_kernel_face(
    mesh_xy: Mesh, kgrid: tuple[int, int, int], n_out: int, face_shape,
    *, k_unfold_plan=None, layout="face",
):
    """Integrate the retarded response using final occupied and unoccupied band weights."""
    from common.fft_helpers import make_flat_k_fftn
    from distrib_la import gemm_plan
    from .greens_function_kernel import build_G_tau
    from .wavefunction_bundle import (
        G_FFT7D_SPEC,
        G_FLATK_SPEC,
        CHI_Q_SPEC,
        CHI_R_SPEC,
        PSI_MUN_SPEC,
        PSI_NMU_SPEC,
    )

    from common.wfn_layout import psi_specs
    PSI_NMU_SPEC, PSI_MUN_SPEC = psi_specs(layout)
    grid = tuple(int(n) for n in kgrid)
    nk = int(np.prod(grid))
    n_out = int(n_out)
    nk_shape, nb_full, n_rmu, ns = (int(v) for v in face_shape)
    # Raw-parent transport (``k_unfold_plan``): the two G's are contracted
    # on the n_parent packed parent faces and unfolded to full k in PACKED
    # centroid order before the FFT -- the minimax kernel's own seam
    # (``_get_chi_minimax_kernel_face``) -- and the finished χ0(q) is
    # restored to the canonical order at the end.
    expected_input_nk = (
        nk if k_unfold_plan is None else int(k_unfold_plan.n_parent))
    if nk_shape != expected_input_nk:
        raise ValueError(
            f"_get_chi_fractional_contour_kernel_face: face_shape "
            f"nk={nk_shape} does not match the expected input k extent "
            f"{expected_input_nk}")
    if k_unfold_plan is not None and int(k_unfold_plan.n_full) != nk:
        raise ValueError(
            "_get_chi_fractional_contour_kernel_face: plan n_full "
            f"{k_unfold_plan.n_full} != prod(kgrid)={nk}.")

    G_fftn = make_flat_k_fftn(
        mesh_xy, grid, G_FFT7D_SPEC, norm="ortho")
    chi_fftn = make_flat_k_fftn(
        mesh_xy, grid, CHI_Q_SPEC, norm="ortho")
    G_shard = NamedSharding(mesh_xy, G_FLATK_SPEC)
    chi_R_shard = NamedSharding(mesh_xy, CHI_R_SPEC)
    rep2 = NamedSharding(mesh_xy, P(None, None))
    rep1 = NamedSharding(mesh_xy, P(None))
    rep0 = NamedSharding(mesh_xy, P())
    psi_mun_shard = NamedSharding(mesh_xy, PSI_MUN_SPEC)
    psi_nmu_shard = NamedSharding(mesh_xy, PSI_NMU_SPEC)

    # ONE planned GEMM, built here (eagerly, once) and shared by every Gf
    # and Gu build this kernel ever does — mirrors
    # _get_chi_minimax_kernel_face's own g_plan.
    g_plan = gemm_plan(mesh_xy, m=n_rmu * ns, k=nb_full, n=n_rmu * ns,
                       nq=nk, dtype=jnp.complex128, layout=layout)
    def _finish(value):
        return chi_fftn(value)

    @partial(
        jax.jit,
        in_shardings=(
            rep1, rep0,
            psi_mun_shard, psi_nmu_shard,
            rep2, rep2, rep2, rep0,
        ),
        out_shardings=tuple(chi_R_shard for _ in range(n_out)),
    )
    def integrate(
        time_nodes,
        projection_rows,
        psi_mun,
        psi_nmu,
        enk_full,
        occ_f,
        occ_u,
        energy_reference,
    ):
        # The caller supplies occ and 1-occ after support masking; do not invert again.
        n_mu = psi_mun.shape[2]
        zero = jax.lax.with_sharding_constraint(
            jnp.zeros((nk, n_mu, n_mu), dtype=jnp.complex128),
            chi_R_shard,
        )
        initial = tuple(zero for _ in range(n_out))

        def body(accumulators, node):
            time, projection = node
            tau = jnp.asarray(1j, dtype=jnp.complex128) * time
            Gf_k = jax.lax.with_sharding_constraint(
                jnp.conj(build_G_tau(
                    psi_mun,
                    psi_nmu,
                    enk_full,
                    -tau,
                    e_ref=energy_reference,
                    band_weight=occ_f,
                    layout=layout,
                    gemm=g_plan,
                    k_unfold_plan=k_unfold_plan,
                )),
                G_shard,
            )
            Gu_k = jax.lax.with_sharding_constraint(
                jnp.conj(build_G_tau(
                    psi_mun,
                    psi_nmu,
                    enk_full,
                    -tau,
                    e_ref=energy_reference,
                    band_weight=occ_u,
                    layout=layout,
                    gemm=g_plan,
                    k_unfold_plan=k_unfold_plan,
                )),
                G_shard,
            )
            Gf_R = G_fftn(Gf_k)
            Gu_R = G_fftn(Gu_k)
            A_R = jax.lax.with_sharding_constraint(
                jnp.einsum(
                    "Rambn,Rambn->Rmn",
                    Gu_R,
                    jnp.conj(Gf_R),
                    optimize=True,
                ),
                chi_R_shard,
            )
            reverse_R = jnp.conj(A_R)
            updated = tuple(
                accumulators[i]
                - 1j * projection[i] * A_R
                + 1j * projection[i] * reverse_R
                for i in range(n_out)
            )
            return updated, None

        final_R, _ = jax.lax.scan(
            body,
            initial,
            (time_nodes, jnp.transpose(projection_rows)),
            unroll=1,
        )
        return tuple(_finish(value) for value in final_R)

    return integrate


# ============================================================================
# W solve — plan 1 of 2: LOCAL (q-parallel per-q dense LU)
# ============================================================================

def _get_w_solve_fn_local(mesh_xy: Mesh, nq: int, n_rmu: int,
                          n_rmu_logical: int | None = None):
    """W = (I - V χ)⁻¹ V via q-parallel shard_map.  All arrays flat-q: (nq, μ, μ).

    The LOCAL plan: q's are scattered over all devices
    (``P(('x','y'),None,None)``) and each rank runs one dense pivoted LU
    (``lu_factor``/``lu_solve``) per owned q.  LU is the right inner
    solve: A is SQUARE and generically well conditioned (it is I minus a
    term whose spectral radius is < 1 wherever the RPA screening is
    physical — an eigenvalue of Vχ₀ reaching 1 is a plasmon instability,
    not a numerical one).  One factorisation, one triangular pair of
    solves.

    ``n_rmu_logical``: when smaller than ``n_rmu`` (μ-padded inputs),
    the per-q pivoted LU is μ-SLICED to the logical extent and the W
    pad rows/cols are zero-filled after (their exact value: V pad rows
    are zero).  Load-bearing for device-count invariance — LU at the
    padded extent regroups partial sums per pad extent, and the
    resulting 1e-8-rel W wobble is amplified to eV on near-pole GN-PPM
    bands (reports/device_invariance_2026-07-08/ROOT_CAUSE.md, charge
    manifestation).  At zero pad the slice/fill are no-ops.
    """
    from common.shard_map import shard_map

    n_log = int(n_rmu_logical) if n_rmu_logical is not None else int(n_rmu)
    if n_log > int(n_rmu):
        raise ValueError(
            f"_get_w_solve_fn_local: n_rmu_logical={n_log} exceeds extent {n_rmu}")
    mu_pad = int(n_rmu) - n_log

    cache_key = ("local", id(mesh_xy), nq, n_rmu, n_log)
    if cache_key in _w_solve_cache:
        return _w_solve_cache[cache_key]

    q_shard = NamedSharding(mesh_xy, P(('x', 'y'), None, None))
    # ── W COMES OUT 2-D SHARDED: W_q(μ_X, ν_Y) ────────────────────────
    # This used to be ``rep_3d = P(None, None, None)`` — a full
    # all-gather of the whole (nq, μ, μ) stack onto every rank, and the
    # last replicated O(nq·μ²) object in the production path (scorecard
    # J.2 #3: nq·μ²·16 per rank, ×2 for the static+probe pair, break-μ
    # ≈ 4.4 k, re-paid every SC iteration).  Nothing wanted it: every
    # consumer either is layout-agnostic (sigma_dispatch, sc_iteration,
    # gw_jax) or immediately re-imposes exactly this layout —
    # ``symmetry_maps.unfold_isdf_operator`` (P(None,'x','y') in and out),
    # ``cohsex_sigma._convolve``'s 5-D V_FFT5D_SPEC = P(None,None,None,
    # 'x','y'), ``ppm_sigma.fit_ppm``'s q_shard, and
    # ``head_wing_schur`` which literally undid the replication by hand.
    # The ONLY q-index anywhere is ``screening.py``'s ``W[0]``
    # hermiticity gate, which is a two-reduction check on one tile.
    # The distributed plan has always returned P(None,'x','y'), so the
    # whole downstream chain is already proven on this layout.
    #
    # Collective-wise the final constraint changes from an ALL-GATHER
    # (every rank ends holding nq·μ²·16 B) to an ALL-TO-ALL (nq·μ²·16/P
    # per rank).  The values are untouched — the shard_map above
    # computes the same numbers either way and this is pure data
    # movement — so the change is bit-exact by construction.
    nat_3d = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    # Intermediate sharding for the reshard from P(None,'x','y') → q_shard.
    # Routing through P('x',None,'y') (x parks on nq, y stays on μ₂) lets
    # SPMD plan it as two single-axis all_to_alls instead of the
    # "Involuntary full rematerialization" it falls into when asked to
    # un-shard both x and y on μ simultaneously via a fully replicated
    # intermediate.  Measured at Si 4×4×4 60Ry (nq=64, μ=1200, 2×2 mesh):
    #   via a fully replicated P(None,None,None) intermediate:
    #                          peak 2.95 GB/dev (temp 2.21 GB) — Involuntary Remat
    #   via P('x',None,'y'): peak 1.11 GB/dev (temp 0.37 GB)  -- 62% reduction
    reshard_mid = NamedSharding(mesh_xy, P('x', None, 'y'))
    q_spec = P(('x', 'y'), None, None)

    # ``chi_flat`` is donated (position 1): the caller releases χ₀ right
    # after this call (module contract, same as the distributed plan —
    # the ``del chi0_q_solve`` inside ``screening.py``'s ``W.exec``
    # timing block).  ``V_flat`` is NOT donated — V is reused
    # by COHSEX Σ_SX, Σ_COH, Σ_X and the PPM fit's Wc = W - V step.
    @partial(jax.jit, donate_argnums=(1,))
    def _solve_w(V_flat: jax.Array, chi_flat: jax.Array, pref: jax.Array) -> jax.Array:
        """V_flat, chi_flat: (nq, μ, μ).  Returns W: (nq, μ, μ)."""
        nq_local = V_flat.shape[0]
        n = V_flat.shape[1]
        chi_scaled = pref * chi_flat

        # Pad to device count then reshard to q-parallel
        from runtime.padding import padded_axis
        nq_padded = padded_axis(
            nq_local, mesh_xy, name="local W q carrier",
            spec=q_spec, axis=0).carrier
        pad = nq_padded - nq_local
        V_padded = jnp.pad(V_flat, ((0, pad), (0, 0), (0, 0))) if pad > 0 else V_flat
        chi_padded = jnp.pad(chi_scaled, ((0, pad), (0, 0), (0, 0))) if pad > 0 else chi_scaled
        V_q = jax.lax.with_sharding_constraint(
            jax.lax.with_sharding_constraint(V_padded, reshard_mid), q_shard)
        chi_q = jax.lax.with_sharding_constraint(
            jax.lax.with_sharding_constraint(chi_padded, reshard_mid), q_shard)

        def _local_solve(V_local, chi_local):
            nq_dev = V_local.shape[0]

            def _dyson_log(V_log, chi_log):
                # Solve at the LOGICAL μ extent (see _get_w_solve_fn_local
                # docstring; slice/zero-refill via solve_at_logical).
                # V/χ pad rows are exact zeros, so the sliced system IS
                # the logical Dyson system; the W pad block is exactly
                # zero (A_pad = I, RHS_pad = 0).
                A = jnp.eye(n_log, dtype=V_log.dtype) - V_log @ chi_log
                lu, piv = jsp_linalg.lu_factor(A)
                return jsp_linalg.lu_solve((lu, piv), V_log)

            def solve_one(iq, W_acc):
                W_row = solve_at_logical(
                    _dyson_log, n_log, (V_local[iq], chi_local[iq]),
                    pad_axes=(-2, -1))
                return jax.lax.dynamic_update_slice(
                    W_acc, W_row[None, :, :], (iq, 0, 0))
            return jax.lax.fori_loop(0, nq_dev, solve_one, jnp.zeros_like(V_local))

        W_flat = shard_map(
            _local_solve, mesh=mesh_xy,
            in_specs=(q_spec, q_spec), out_specs=q_spec,
        )(V_q, chi_q)

        if pad > 0:
            W_flat = W_flat[:nq_local]
        # Land W on P(None,'x','y') through the SAME single-axis staging the
        # input reshard uses, in reverse:
        #     q-parallel [px·py,1,1] -> P('x',None,'y') [px,1,py]
        #                            -> P(None,'x','y') [1,px,py]
        # Asking SPMD for the composite in ONE step makes it
        # replicate-then-partition: MEASURED on a real 2×2 CUDA mesh and on
        # CPU, `[SPMD] Involuntary full rematerialization ... cannot go from
        # {devices=[4,1,1]} to {devices=[1,2,2]} ... op_name=
        # "jit(_solve_w)/shard_map"`, 1 per rank, where the base (replicated)
        # output produced none.  That transient is the whole nq·μ² object
        # this change exists to stop materialising, so the staging is
        # load-bearing, not cosmetic.  Each stage moves ONE mesh axis, which
        # is a single all_to_all — the same reasoning (and the same
        # `reshard_mid`) as the 62 %-peak-reduction note above.
        W_flat = jax.lax.with_sharding_constraint(W_flat, reshard_mid)
        return jax.lax.with_sharding_constraint(W_flat, nat_3d)

    _w_solve_cache[cache_key] = _solve_w
    return _solve_w


# ============================================================================
# W solve — plan 2 of 2: DISTRIBUTED (2-D-sharded stacked-GEMM backsolve)
# ============================================================================

def _get_w_solve_fn_distributed(mesh_xy: Mesh, nq: int, n_rmu: int,
                                n_rmu_logical: int,
                                distrib_la_batched_route: str = "batch_reshard"):
    """W = solve(A, V), A = (1 − pref·V·χ₀), everything 2-D sharded.

    The DISTRIBUTED plan — the scale-out route for thousands of
    low-memory processes, in the same architectural family as the
    ζ-fit's distributed rank-truncate tier
    (:func:`isdf.core._factor_c_q_distributed_rank_truncate`):

    1. **A build** — per q-block, ``A = I − V·(pref·χ)`` as a 2-D block
       GEMM inside ``shard_map``: rank (x, y) all-gathers V's row block
       along 'y' (full k for its i rows, μ·μ/Px per rank) and χ's column
       block along 'x' (full k for its j columns, μ·μ/Py per rank),
       multiplies locally, and subtracts from its identity tile.  The
       gathers are STRUCTURAL — inside shard_map the partitioner cannot
       hoist them into a full-stack gather (the per_q-tier lesson,
       quality pattern #4).  The q loop is chunked HOST-side so one
       collective instruction never exceeds ``LORRAX_COLLECTIVE_CHUNK_MB``
       (the AF transport bound; separate XLA executions cannot be
       re-combined by a compiler pass).
    2. **Factor + backsolve** — ONE resolved
       :class:`distrib_la.Plan` for ``solve_lu`` with
       ``backend='distributed'`` (ScaLAPACK ``pzgetrf``/``pzgetrs`` on a
       CPU mesh, cuSOLVERMp on CUDA — ``resolve._DISTRIBUTED_DEFAULT``),
       consuming the block-cyclic tiles where they already live.

    **No rank ever materialises a full (μ, μ) tile**: inputs, A, the LU
    factors and W all stay ``P(None,'x','y')`` (per-rank blocks of
    μ/Px × μ/Py; the largest per-rank transient is the μ·μ/min(Px,Py)
    gathered GEMM operand).  W lands natively in ``P(None,'x','y')`` —
    no relayout, unlike the local plan.

    Padding contract, and why it is exact: V and χ pad rows/cols are
    exact zeros (the bilinear-in-zero-padded-ψ contract), so at the
    PADDED extent ``A = [[A_log, 0], [0, I]]`` and ``RHS = [[V_log], [0]]``
    hold EXACTLY — the identity-embedded block-diagonal system whose
    solution is ``[[W_log], [0]]``; partial pivoting cannot mix the
    blocks (every pad column is a unit vector, every pad row is zero in
    the logical columns).  Therefore W's pad rows/cols leave the solve as
    exact zeros without a separate post-solve mask graph.  Unlike the local
    plan the LOGICAL
    block is formed/factored at the padded extent, so W here carries the
    ≤1e-8-rel pad-extent regrouping wobble — which is subsumed by the
    block-cyclic factorisation's own non-bit-identity; this plan's
    numerical contract is the Dyson residual (``LORRAX_W_RESIDUAL_CHECK``),
    not bit-identity with the local plan.

    Geometry/capability failures (host lib absent, non-square or 1-D
    mesh, n not divisible, process coverage) RAISE at resolve time with
    the resolver's own message — an explicitly requested distributed
    solve never silently downgrades to the local plan (quality pattern
    #6/#8).
    """
    n_ext = int(n_rmu)
    n_log = int(n_rmu_logical)
    if n_log > n_ext:
        raise ValueError(
            f"_get_w_solve_fn_distributed: n_rmu_logical={n_log} exceeds "
            f"extent {n_ext}")

    # The local masking closure captures the logical prefix.  Two callers can
    # legitimately share one padded extent while owning different logical
    # prefixes (notably scalar charge versus an explicit packed carrier), so
    # omitting n_log would reuse a closure with the wrong exact-zero mask.
    cache_key = ("distributed", id(mesh_xy), nq, n_ext, n_log,
                 str(distrib_la_batched_route))
    if cache_key in _w_solve_cache:
        return _w_solve_cache[cache_key]

    from common.shard_map import shard_map
    from ffi import _services
    _services.ensure_on_path()
    from distrib_la import plan as linalg_plan
    # House chunking pattern — single source (scorecard AF): one emitted
    # collective's payload is bounded by LORRAX_COLLECTIVE_CHUNK_MB.
    # TODO(release): promote _chunk_q/_chunk_log to a public home (e.g.
    # common/collectives.chunk_q/chunk_log) and import them publicly from
    # both isdf/core and here — gw physics code should not reach into
    # another package's underscore-private namespace, and _chunk_log's
    # module-global dedup set is cross-package shared mutable state with
    # no public contract (audit fix/zq 2026-07-28, _idx 29; needs an
    # isdf/core edit, outside this fix's file set).
    from isdf.core import _chunk_q, _chunk_log

    # Every guard fires HERE (vocabulary, platform, capability, process
    # coverage, mesh geometry — including the 1-D-mesh cusolvermp refusal
    # — and divisibility) — resolve.py's ladder, with its own messages.
    # ``distributed`` maps to the platform default (ScaLAPACK on cpu,
    # cuSOLVERMp on CUDA) in ONE place.  No compensating ``p.is_native``
    # re-check is needed: an explicit request that cannot be honored
    # raises at resolve time (the former silent 1-D-mesh degenerate-to-
    # native was removed; audit fix/zq 2026-07-28), so a returned plan is
    # always the distributed backend it names.
    p = linalg_plan(
        "solve_lu", mesh_xy, backend="distributed", n=n_ext,
        batched_route=distrib_la_batched_route)

    px = int(mesh_xy.shape['x'])
    py = int(mesh_xy.shape['y'])
    nat = NamedSharding(mesh_xy, P(None, 'x', 'y'))

    if jax.process_index() == 0:
        print(f"  [W solve] w_dyson_solver=distributed -> {p.describe()}",
              flush=True)

    def _logical_tile(A_loc):
        """Zero the nonphysical rows/columns of one distributed μ tile."""
        i0 = jax.lax.axis_index('x') * (n_ext // px)
        j0 = jax.lax.axis_index('y') * (n_ext // py)
        logical = jnp.logical_and(
            i0 + jnp.arange(n_ext // px)[:, None] < n_log,
            j0 + jnp.arange(n_ext // py)[None, :] < n_log)
        return jnp.where(logical[None, :, :], A_loc,
                         jnp.zeros((), dtype=A_loc.dtype))

    # The two collectives ``_a_local`` emits, per q (2-D block GEMM):
    #   all_gather('y')  V   (μ/Px, μ/Py) -> (μ/Px, μ)  = μ²/Px · 16 B
    #   all_gather('x')  χ   (μ/Px, μ/Py) -> (μ, μ/Py)  = μ²/Py · 16 B
    # The BIGGER of the two sets the q-block (see ``_chunk_q``).
    per_q_coll = max(n_ext * (n_ext // px), n_ext * (n_ext // py)) * 16

    @partial(shard_map, mesh=mesh_xy,
             in_specs=(P(None, 'x', 'y'), P(None, 'x', 'y')),
             out_specs=P(None, 'x', 'y'), check_vma=False)
    def _a_local(V_loc, chi_loc):
        # A[q,i,j] = δ_ij − Σ_k V[q,i,k]·χs[q,k,j] on my (i on 'x',
        # j on 'y') tile.  Classic 2-D block GEMM pairing — same shape
        # of communication as ``isdf.core._distributed_pinv_apply``.
        # Mask BOTH inputs here, before either gather: the public seam
        # authenticates their common product-padded extent, while this local
        # operation makes the exact-zero pad contract structural even if an
        # upstream padded buffer was poisoned.  No full μ² tile is formed.
        V_loc = _logical_tile(V_loc)
        chi_loc = _logical_tile(chi_loc)
        V_row = jax.lax.all_gather(V_loc, 'y', axis=2, tiled=True)
        chi_col = jax.lax.all_gather(chi_loc, 'x', axis=1, tiled=True)
        prod = jnp.einsum('qik,qkj->qij', V_row, chi_col)
        i0 = jax.lax.axis_index('x') * (n_ext // px)
        j0 = jax.lax.axis_index('y') * (n_ext // py)
        eye_tile = jnp.equal(
            i0 + jnp.arange(n_ext // px)[:, None],
            j0 + jnp.arange(n_ext // py)[None, :]).astype(V_loc.dtype)
        return eye_tile[None, :, :] - prod

    @partial(jax.jit, donate_argnums=(2,), out_shardings=nat)
    def _a_chunk(V_blk, chi_blk, A_acc, q0):
        return jax.lax.dynamic_update_slice(
            A_acc, _a_local(V_blk, chi_blk), (q0, 0, 0))

    # χ is donated here (module contract: the caller releases χ₀ after
    # solve_w — see screening.py's ``del chi0_q_solve``).
    _scale = jax.jit(lambda c, pref: pref * c,
                     donate_argnums=(0,), out_shardings=nat)
    _zeros_like = jax.jit(jnp.zeros_like, out_shardings=nat)
    # RHS must be a FRESH buffer, never an alias of the caller's V —
    # the FFI backsolve DONATES both operands (docs/dev/linalg_ffi.md
    # "Sharp edges") and V is still needed by Σ_SX/Σ_COH/Σ_X and the
    # PPM fit's Wc = W − V.
    @partial(shard_map, mesh=mesh_xy, in_specs=P(None, 'x', 'y'),
             out_specs=P(None, 'x', 'y'), check_vma=False)
    def _zero_pad_rhs_local(V_loc):
        return _logical_tile(V_loc)

    # This remains a fresh RHS buffer (V is not donated), but unlike a plain
    # copy it also makes the identity-embedded system's zero pad exact.
    _copy_zero_pad = jax.jit(_zero_pad_rhs_local, out_shardings=nat)

    def _solve_w_dist(V_flat: jax.Array, chi_flat: jax.Array,
                      pref: jax.Array) -> jax.Array:
        """V_flat, chi_flat: (nq, μ, μ) at P(None,'x','y').  Returns W
        (nq, μ, μ) at P(None,'x','y').  chi_flat's buffer is consumed."""
        nq_local = int(V_flat.shape[0])
        qb = _chunk_q(nq_local, per_q_coll)
        _chunk_log('W Dyson A-build (GEMM)', nq_local, qb, per_q_coll)
        chi_scaled = _scale(chi_flat, pref)
        A = _zeros_like(V_flat)
        # Host-level q-block loop: ONE XLA execution per block, so the
        # emitted all_gather payloads are bounded by construction and
        # cannot be re-combined by a compiler pass (AF note in
        # isdf/core).  At most two compiled shapes (full + remainder).
        for q0 in range(0, nq_local, qb):
            q1 = min(q0 + qb, nq_local)
            A = _a_chunk(V_flat[q0:q1], chi_scaled[q0:q1], A, q0)
        B = _copy_zero_pad(V_flat)
        # ONE plan call for the whole stack: one descriptor, one
        # workspace; A and B are donated into the FFI.
        W = p.batched(A, B)
        # House falsy vocabulary — same parse (and same rationale comment)
        # as common/collectives.py's LORRAX_CHECK_REPLICA fix (workstream
        # AT): the narrow "0"/""/"false" tuple this replaced meant
        # LORRAX_W_RESIDUAL_CHECK=off/no/False silently ENABLED the
        # diagnostic — which must be OFF when taking collective-table
        # probes (docs/dev/env_vars.md).  (audit fix/zq 2026-07-28)
        if os.environ.get("LORRAX_W_RESIDUAL_CHECK", "0").strip().lower() \
                not in ("", "0", "false", "no", "off"):
            _w_residual_report(V_flat, chi_scaled, W, n_ext)
        return W

    _w_solve_cache[cache_key] = _solve_w_dist
    return _solve_w_dist


def _w_residual_report(V_flat, chi_scaled, W, n_ext, n_check: int = 4):
    """Direct Dyson residual ‖(1−Vχ)W − V‖/‖V‖ on the first few q.

    THE strict numerical contract of the distributed plan (a
    block-cyclic LU is not bit-comparable to the local per-q LU; the
    residual is what certifies the solve — quality pattern #6, "test
    what executes").  Diagnostic-only, opt-in via
    ``LORRAX_W_RESIDUAL_CHECK=1``; never on in the traced production
    path, so the collective-table gate is taken with it OFF.
    """
    ns = min(int(V_flat.shape[0]), int(n_check))

    @jax.jit
    def _res(V_s, chi_s, W_s):
        A_s = jnp.eye(n_ext, dtype=V_s.dtype)[None, :, :] - V_s @ chi_s
        num = jnp.linalg.norm((A_s @ W_s - V_s).reshape(ns, -1), axis=1)
        den = jnp.linalg.norm(V_s.reshape(ns, -1), axis=1)
        return num / den

    r = np.asarray(jax.device_get(_res(V_flat[:ns], chi_scaled[:ns], W[:ns])))
    if jax.process_index() == 0:
        vals = "  ".join(f"q{iq}={v:.3e}" for iq, v in enumerate(r))
        print(f"  [W solve] Dyson residual |(1-Vchi)W - V|/|V| ({ns} q): "
              f"{vals}  max={r.max():.3e}", flush=True)


def _w_solve_pref_scalar(meta) -> float:
    """The physical-state prefactor in front of χ₀ in the Dyson solve.

    ``nspinor_wfnfile`` is the source-WFN state multiplicity.  In a
    kinetic-balance lift ``meta.nspinor`` becomes four only to describe the
    bispinor representation; the band and occupation axes are unchanged.
    Using that representation width here would therefore halve every
    charge/current response block.  Read the source field strictly: silently
    falling back to the representation width would reinstate that error.
    """
    nq = int(meta.nk_tot)
    nspin = max(1, int(getattr(meta, 'nspin', 1)))
    nspinor_wfnfile = max(1, int(meta.nspinor_wfnfile))
    return 2.0 / (
        float(max(1, nq)) ** 0.5
        * float(nspin)
        * float(nspinor_wfnfile))


def _resolve_w_solve_fn(meta, mesh_xy, *, n_rmu, n_rmu_logical=None,
                        dyson_solver=None,
                        distrib_la_batched_route: str = "batch_reshard"):
    """Return ``(solve_fn, pref)`` for the requested W plan.

    Single source of truth for the two-plan dispatch.  Both ``solve_w``
    and ``precompile_solve_w`` go through this helper — the dispatch
    logic exists in one place.

    ``dyson_solver`` (input key ``w_dyson_solver``) selects the plan:

    ``local`` (default; ``auto`` is an alias)
        per-q pivoted LU inside the q-parallel shard_map —
        :func:`_get_w_solve_fn_local`.
    ``distributed``
        the 2-D-sharded stacked-GEMM backsolve through the linalg plan
        facade — :func:`_get_w_solve_fn_distributed`.  Refuses loudly at
        resolve time when the mesh/build cannot run it; never silently
        downgrades.

    W comes out ``P(None,'x','y')`` on BOTH — that is the module's
    output contract, not a per-plan detail.
    """
    from .gw_config import normalize_w_dyson_solver
    dyson = normalize_w_dyson_solver(dyson_solver)
    nq = int(meta.nk_tot)
    pref_scalar = _w_solve_pref_scalar(meta)

    # Scalar charge solves take their logical prefix from ``meta``.  The
    # packed photon solve is a direct sum of independently padded channel
    # blocks and therefore has no single charge-logical prefix; its caller
    # passes the complete packed carrier extent explicitly.  Never infer the
    # latter from ``n_rmu`` merely because it is the array width -- that was
    # the historical opt-out-by-omission path for scalar padding.
    n_log = (int(getattr(meta, 'mu_solve_extent', meta.n_rmu))
             if n_rmu_logical is None else int(n_rmu_logical))

    if dyson == "distributed":
        solve_fn = _get_w_solve_fn_distributed(
            mesh_xy, nq, n_rmu, n_log, distrib_la_batched_route)
    else:
        solve_fn = _get_w_solve_fn_local(
            mesh_xy, nq, n_rmu, n_rmu_logical=n_log)
    return solve_fn, jnp.asarray(pref_scalar, dtype=jnp.complex128)


def _require_w_operand_geometry(V_q, chi0_q, meta, mesh_xy, *,
                                n_rmu_logical=None):
    """Authenticate the public Dyson carrier without owning its q set.

    The q axis may be full-BZ or an irreducible wedge; its mapping belongs to
    the screening/MPA caller.  The two centroid axes, however, must be one
    square runtime carrier shared by V and chi, owned by the packed basis
    when present or by the canonical suffix-padding receipt otherwise.
    """
    v_shape = tuple(int(n) for n in V_q.shape)
    chi_shape = tuple(int(n) for n in chi0_q.shape)
    if v_shape != chi_shape:
        raise ValueError(
            "solve_w requires V_q and chi0_q to have the same padded "
            f"(q,mu,nu) extent; got V_q.shape={v_shape} and "
            f"chi0_q.shape={chi_shape}.")
    if len(v_shape) != 3 or v_shape[0] < 1 or v_shape[1] != v_shape[2]:
        raise ValueError(
            "solve_w requires equal rank-3 square (q,mu,nu) operands; "
            f"got V_q.shape=chi0_q.shape={v_shape}.")
    n_logical = (int(getattr(meta, 'mu_solve_extent', meta.n_rmu))
                 if n_rmu_logical is None else int(n_rmu_logical))
    basis = getattr(meta, "mu_basis", None)
    mu_axis = (basis.solve_axis if basis is not None and n_rmu_logical is None
               else padded_mu_axis(n_logical, mesh_xy))
    authenticate_padded_axis(
        mu_axis.logical, v_shape[1], mu_axis,
        name="Dyson row-centroid carrier")
    authenticate_padded_axis(
        mu_axis.logical, v_shape[2], mu_axis,
        name="Dyson column-centroid carrier")
    return n_logical


def solve_w(V_q, chi0_q, meta, mesh_xy, *, dyson_solver=None,
            n_rmu_logical=None,
            distrib_la_batched_route: str = "batch_reshard"):
    """W(q) = (I − V χ₀)⁻¹ V  via a Dyson solve.  **W comes out sharded.**

    All arrays flat-q: V(nq, μ, μ), χ₀(nq, μ, μ) → W(nq, μ, μ).
    Scalar inputs use ``meta.mu_basis``'s packed runtime extent when present,
    otherwise ``padded_mu_extent(meta.n_rmu, mesh_xy)``.  Their q axis may be full-BZ
    or an irreducible wedge; q-set ownership stays with the caller.  A packed
    direct-sum caller supplies ``n_rmu_logical`` explicitly because its
    channel padding is internal rather than one trailing scalar prefix.  The
    distributed plan masks scalar trailing pad rows/columns to exact zero
    before its first contraction.

    **Output contract:** ``W`` is ``P(None, 'x', 'y')`` — 2-D sharded
    W_q(μ_X, ν_Y) — on both plans, and stays that way into its
    consumers (Σ_SX/Σ_COH's 5-D FFT spec, the PPM fit, the IBZ unfold,
    the restart writer).

    ``dyson_solver`` (input key ``w_dyson_solver``) picks one of the
    TWO plans — see :func:`_resolve_w_solve_fn`:

    - ``local`` (default): q-parallel reshard + per-q dense LU via
      shard_map.  Legal on any mesh; each rank holds whole (μ, μ)
      tiles for its q's.
    - ``distributed``: 2-D-sharded stacked-GEMM backsolve through the
      distrib_la plan door (ScaLAPACK on CPU, cuSOLVERMp on CUDA).
      No rank ever materialises a full (μ, μ) tile — the P→∞ memory
      ceiling.  Slower than ``local`` at moderate P; that is priced and
      accepted (the point is the per-rank memory ceiling, not speed).

    ``chi0_q``'s buffer is CONSUMED (donated) on both plans — the
    caller must drop its reference after this call.
    """
    n_logical = _require_w_operand_geometry(
        V_q, chi0_q, meta, mesh_xy, n_rmu_logical=n_rmu_logical)
    solve_fn, pref = _resolve_w_solve_fn(
        meta, mesh_xy, n_rmu=chi0_q.shape[1],
        n_rmu_logical=n_logical, dyson_solver=dyson_solver,
        distrib_la_batched_route=distrib_la_batched_route)
    with jax_profile.annotation("W_solve"):
        return solve_fn(V_q, chi0_q, pref)


def _chi_face_kwargs(wfns) -> dict:
    """Return the canonical face shape arguments from the wavefunction carrier."""
    from .wavefunction_bundle import face_kernel_kwargs
    return face_kernel_kwargs(wfns)


def _chi_parent_face_kwargs(wfns) -> dict:
    """Return Green-function shape and typed raw-parent transport arguments."""
    from .wavefunction_bundle import green_face_kernel_kwargs
    return green_face_kernel_kwargs(wfns)


def _chi_layout_operands(wfns, eref):
    """Select face operands and physical valence/conduction masks for the minimax response."""
    s = wfns.slices
    eref_j = jnp.asarray(eref, dtype=wfns.enk.dtype)
    carrier = wfns.green_parent
    if carrier is None:
        carrier = wfns
    mask_v = carrier.band_mask(s.val)
    mask_c = carrier.band_mask(s.cond)
    enk_full = carrier.enk - eref_j
    return (carrier.psi_mun, carrier.psi_nmu,
            mask_v, mask_c, enk_full)


def compute_chi0(wfns, quad, meta, mesh_xy, *, energy_reference=0.0):
    """Compute χ₀(q) from a wavefunction bundle and minimax quadrature.

    Returns flat-q array (nq, μ, μ).

    ``quad.tau`` and ``quad.alpha`` approximate either 1/x (static) or
    x/(x²+ωp²) (imaginary-frequency) on [x_min, x_max] where x = E_c - E_v.
    The physical static/imaginary-axis χ₀ contains both ordered
    particle-hole orientations.  In the real-space convolution used here::

        χ₀ = -Σ_ℓ α_ℓ [A_R(τ_ℓ) + conj(A_R(τ_ℓ))]

    before the final R-to-q FFT.  The conjugate term maps to
    ``conj(A_-q)`` and is distinct from ``A_q`` for complex broken-TR states.

    A uniform energy shift via ``energy_reference`` is applied to both
    valence and conduction energies before building the minimax factors.
    Because only differences enter, this is algebraically invariant; the
    knob lets callers align the global zero (e.g. midgap, VBM, CBM).
    """
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))

    s = wfns.slices
    enk_v = wfns.enk[:, s.val]
    enk_c = wfns.enk[:, s.cond]
    eref = 0.0 if energy_reference is None else float(energy_reference)
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64) - eref
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64) - eref
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    E_gap = cmin - vmax

    tau = np.asarray(quad.tau, dtype=np.float64)
    # Fold the one-orientation prefactor (-exp(-τ·E_gap)) into α.  The
    # kernel adds A_R + conj(A_R), the two ordered particle-hole
    # orientations, before this weight is applied.  ``MinimaxNodes`` carries
    # both in complex128; τ has Im=0 for the Laplace quad.
    alpha_chi = -1.0 * np.asarray(quad.alpha, dtype=np.float64) * np.exp(-tau * E_gap)
    nodes = MinimaxNodes(
        t=jnp.asarray(tau, dtype=jnp.complex128),
        alpha=jnp.asarray(alpha_chi, dtype=jnp.complex128),
    )

    kernel = _get_chi_minimax_kernel(
        mesh_xy, kgrid, **_chi_parent_face_kwargs(wfns))
    return kernel(
        nodes, *_chi_layout_operands(wfns, eref),
        jnp.asarray(vmax, dtype=jnp.float64),
        jnp.asarray(cmin, dtype=jnp.float64),
    )


def _chi0_imag_ordered_kernel_args(wfns, quad, energy_reference):
    """Nodes/operands of the ordered imaginary-axis route (see below)."""
    if getattr(quad, "alpha_odd", None) is None:
        raise ValueError(
            "GATE chi0_imag_ordered_needs_odd_kernel: the ordered response "
            "received no odd quadrature weights.\n"
            "  got:  quad.alpha_odd = None\n"
            "  want: build_imag_quadrature(..., with_odd_kernel = true)\n"
            "  why:  without odd weights the time-reversal-odd response "
            "channel is zero by construction\n"
            "  doc:  docs/dev/notes/DERIVATION_gnppm_nonhermitian.md")
    s = wfns.slices
    enk_v = wfns.enk[:, s.val]
    enk_c = wfns.enk[:, s.cond]
    eref = 0.0 if energy_reference is None else float(energy_reference)
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64) - eref
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64) - eref
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    E_gap = cmin - vmax
    tau = np.asarray(quad.tau, dtype=np.float64)
    alpha = np.asarray(quad.alpha, dtype=np.float64)
    beta = np.asarray(quad.alpha_odd, dtype=np.float64)
    if tau.shape != alpha.shape or tau.shape != beta.shape:
        raise ValueError(
            "chi0_imag_ordered: tau, alpha and alpha_odd must share one "
            f"node axis; got {tau.shape}, {alpha.shape}, {beta.shape}")
    # gamma_l = -(alpha_l - i beta_l) e^{-tau_l E_gap}: the resolvent
    # -1/(x + i omega_p) of the kernel's OWN orientation (the -Delta pole,
    # DERIVATION_gnppm_nonhermitian.md section 2).  The conjugate partner
    # receives conj(gamma) through the q-negated conjugate below.
    gamma = -(alpha - 1j * beta) * np.exp(-tau * E_gap)
    nodes = MinimaxNodes(
        t=jnp.asarray(tau, dtype=jnp.complex128),
        alpha=jnp.asarray(gamma, dtype=jnp.complex128),
    )
    return (
        nodes, *_chi_layout_operands(wfns, eref),
        jnp.asarray(vmax, dtype=jnp.float64),
        jnp.asarray(cmin, dtype=jnp.float64),
    )


def compute_chi0_imag_ordered(wfns, quad, meta, mesh_xy, *, q_neg_index,
                              energy_reference=0.0):
    """χ₀(q; iω_p) with BOTH particle-hole orientations carrying their own
    frequency weight — the route for a deck whose measured time-reversal
    verdict is false.  Returns flat-q (nq, μ, μ), ``P(None, 'x', 'y')``.

    :func:`compute_chi0` applies the EVEN kernel ``x/(x²+ωp²)`` to the
    orientation sum ``A_R + conj(A_R)``, which deletes the anti-Hermitian,
    magnetisation-odd channel ``iω(P^q − conj(P^{−q}))/(ω²+Δ²)`` of χ₀(iω)
    (lane G, measured on CrI3 run 128).  The exact object is the SAME two
    carriers with independent complex weights::

        χ₀_q(iωp) = F_q + conj(F_{−q}),
        F_q       = Σ_l γ_l e^{−τ_l E_gap} A_q(τ_l),   γ_l = −(α_l − iβ_l),

    with ``α`` the served even rule (unchanged) and ``β`` the odd rule
    ``ωp/(x²+ωp²)`` on the same nodes (``quad.alpha_odd``).  ``F_q`` is one
    sweep of the existing ``complex_contour`` kernel (real nodes, complex
    weights, no in-kernel completion) — no second response implementation —
    and the partner is the flat-q negation gather of its conjugate, which
    ``FFT_R[conj(A_R)] = conj(A_{−q})`` makes exact.  On a Θ deck
    ``conj(A_{−q}) = A_q`` and this equals :func:`compute_chi0` to roundoff;
    the caller keeps the incumbent path there so Θ decks stay bit-identical.
    Reciprocity ``χ_{−q} = conj(χ_q)`` holds by construction.

    ``q_neg_index`` is the public ``symmetry_maps.q_negation_index`` row
    permutation for ``meta.kgrid`` — passed in, never rebuilt here (TASTE 4).
    The probe roles run on the FULL BZ, which is the only grid on which the
    involution is meaningful.
    """
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    args = _chi0_imag_ordered_kernel_args(wfns, quad, energy_reference)
    kernel = _get_chi_minimax_kernel(
        mesh_xy, kgrid, n_out=1, complex_contour=True,
        **_chi_face_kwargs(wfns))
    F_q = kernel(*args)
    q_neg = np.asarray(q_neg_index)
    nq = int(F_q.shape[0])
    if (q_neg.shape != (nq,)
            or not np.array_equal(q_neg[q_neg], np.arange(nq))):
        raise ValueError(
            "compute_chi0_imag_ordered: q_neg_index must be an involution "
            f"over the full flat-q axis [0, {nq}); got shape {q_neg.shape}.")
    return _complete_imag_ordered(F_q, jnp.asarray(q_neg, dtype=jnp.int32))


@jax.jit
def _complete_imag_ordered(F_q, q_neg):
    """``F_q + conj(F_{-q})`` — a leading-axis gather, sharding-preserving."""
    return F_q + jnp.conj(jnp.take(F_q, q_neg, axis=0))


def precompile_chi0_imag_ordered(wfns, quad, meta, mesh_xy, *,
                                 energy_reference=None):
    """AOT sibling of :func:`compute_chi0_imag_ordered` (the contour
    kernel's compile; the completion gather is negligible)."""
    if len(np.asarray(quad.tau)) == 0:
        return
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    args = _chi0_imag_ordered_kernel_args(wfns, quad, energy_reference)
    kernel = _get_chi_minimax_kernel(
        mesh_xy, kgrid, n_out=1, complex_contour=True,
        **_chi_face_kwargs(wfns))
    kernel.lower(*args).compile()


def compute_no_pair_dirac_current_block(
    wfns_left, wfns_right, quad, meta, mesh_xy, *,
    vertex_left: int, vertex_right: int, energy_reference=0.0,
):
    """Return one paramagnetic no-pair block without a Ward contact."""
    return compute_no_pair_dirac_current_blocks(
        wfns_left, wfns_right, quad, meta, mesh_xy,
        vertex_pairs=((vertex_left, vertex_right),),
        energy_reference=energy_reference)[0]


def _require_current_chi_endpoints(wfns_left, wfns_right):
    """Authenticate the paired raw-parent face, spin and band extents."""
    if wfns_left.layout != wfns_right.layout:
        raise ValueError(
            "compute_no_pair_dirac_current_block endpoint layouts differ: "
            f"{wfns_left.layout!r} vs {wfns_right.layout!r}")
    if wfns_left.layout not in ("face", "axis"):
        raise ValueError(
            "compute_no_pair_dirac_current_block requires "
            "the canonical face layout")
    if wfns_right.slices != wfns_left.slices or tuple(
            wfns_right.enk.shape) != tuple(wfns_left.enk.shape):
        raise ValueError(
            "chi endpoint bundles must share band slices and energy-table "
            f"shape; got left={wfns_left.slices}/{wfns_left.enk.shape}, "
            f"right={wfns_right.slices}/{wfns_right.enk.shape}")
    left, right = wfns_left.green_parent, wfns_right.green_parent
    if left is None or right is None:
        raise ValueError("Four-current response requires two raw-parent carriers.")
    if left.psi_mun.shape[1] != 4 or right.psi_nmu.shape[2] != 4:
        raise ValueError("Four-current vertices require four spin components.")
    return left, right


def compute_no_pair_dirac_current_blocks(
    wfns_left, wfns_right, quad, meta, mesh_xy, *,
    vertex_pairs, energy_reference=0.0,
):
    """Integrate one centroid-family class with shared Greens and return q-IBZ blocks."""
    left, right = _require_current_chi_endpoints(wfns_left, wfns_right)
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))

    s = wfns_left.slices
    enk_v = wfns_left.enk[:, s.val]
    enk_c = wfns_left.enk[:, s.cond]
    eref = 0.0 if energy_reference is None else float(energy_reference)
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64) - eref
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64) - eref
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    E_gap = cmin - vmax

    tau = np.asarray(quad.tau, dtype=np.float64)
    # Fold the one-orientation prefactor (-exp(-τ·E_gap)) into alpha.  The
    # kernel adds the exact forward and reverse ordered transitions before
    # the final q FFT; retaining the historical -2 here would double count.
    alpha_chi = -1.0 * np.asarray(quad.alpha, dtype=np.float64) * np.exp(-tau * E_gap)
    nodes = MinimaxNodes(
        t=jnp.asarray(tau, dtype=jnp.complex128),
        alpha=jnp.asarray(alpha_chi, dtype=jnp.complex128),
    )

    from .wavefunction_bundle import green_face_kernel_kwargs
    left_shape = green_face_kernel_kwargs(wfns_left)["face_shape"]
    right_shape = green_face_kernel_kwargs(wfns_right)["face_shape"]
    kernel = _get_chi_minimax_kernel(
        mesh_xy, kgrid, layout=left.layout, face_shape=left_shape,
        right_face_shape=right_shape, vertex_pairs=vertex_pairs,
        k_unfold_plan=(left.plan, right.plan))
    mask_v = left.plan.parent_rows(wfns_left.band_mask(s.val))
    mask_c = left.plan.parent_rows(wfns_left.band_mask(s.cond))
    args = (nodes, left.psi_mun, right.psi_nmu, mask_v, mask_c,
            left.enk - jnp.asarray(eref, dtype=left.enk.dtype),
            jnp.asarray(vmax, dtype=jnp.float64), jnp.asarray(cmin, dtype=jnp.float64))
    from common.gamma_matrices import gamma_perm_phase
    vertices = []
    for A, B in vertex_pairs:
        perm_l, phase_l = gamma_perm_phase(A)
        perm_r, phase_r = gamma_perm_phase(B)
        vertices.append((perm_l, jnp.conj(phase_l), perm_r, jnp.conj(phase_r)))
    full = kernel(*args, tuple(vertices))
    rows = jnp.asarray(left.plan.sym.q_irr_full_idx)
    return tuple(jnp.take(block, rows, axis=0) for block in full)


_WARD_SUBTRACTED_NO_PAIR = "ward_subtracted_no_pair"

STATIC_PHOTON_NO_PAIR_MODEL = NO_PAIR_DIRAC_CURRENT_MODEL


@partial(jax.jit, donate_argnums=(0,))
def _subtract_static_tt_contact(chi_tt):
    """Subtract the Γ contact on q-IBZ before its typed star transport."""
    corrected = chi_tt - chi_tt[0:1]
    # Make the q=0 cancellation structural rather than roundoff-dependent.
    return corrected.at[0].set(jnp.zeros_like(corrected[0]))


def compute_experimental_no_pair_photon_chi0(
    wfns_charge, wfns_transverse, quad, meta, mesh_xy, layout, *,
    current_contact: str = _WARD_SUBTRACTED_NO_PAIR,
    energy_reference=0.0,
):
    """Build all sixteen no-pair blocks with an experimental TT proxy.

    One family class and the donated packed accumulator are resident at a
    time; its vertices share the same Green/FFT pair at each tau node.
    """
    from .photon_layout import pack_photon_operator

    layout.assert_mesh(mesh_xy)
    if current_contact != _WARD_SUBTRACTED_NO_PAIR:
        raise ValueError(
            "full static photon response currently requires "
            f"current_contact={_WARD_SUBTRACTED_NO_PAIR!r}; "
            f"got {current_contact!r}")
    if (wfns_charge.layout != wfns_transverse.layout
            or wfns_charge.layout not in ("face", "axis")):
        raise ValueError(
            "full four-current response requires matching face or axis layouts "
            "for charge and transverse endpoint bundles; "
            f"got {wfns_charge.layout!r}/{wfns_transverse.layout!r}")
    from .wavefunction_bundle import padded_centroid_extent
    n_c = padded_centroid_extent(wfns_charge)
    n_t = padded_centroid_extent(wfns_transverse)
    if (n_c != layout.carrier_extent(0) or
            n_t != layout.carrier_extent(1)):
        raise ValueError(
            "photon layout padded extents do not match wavefunction "
            f"bundles: layout C/T=({layout.carrier_extent(0)},"
            f"{layout.carrier_extent(1)}), wfns C/T=({n_c},{n_t})")
    nq = len(wfns_charge.green_parent.plan.sym.q_irr_full_idx)
    families = (wfns_charge, wfns_transverse,
                wfns_transverse, wfns_transverse)
    classes = tuple(tuple((A, B) for A in left for B in right)
                    for left in ((0,), (1, 2, 3)) for right in ((0,), (1, 2, 3)))
    pending_classes = iter(classes)
    blocks = {}

    def get_block(A, B):
        if not blocks:
            pairs = next(pending_classes)
            values = compute_no_pair_dirac_current_blocks(
                families[A], families[B], quad, meta, mesh_xy,
                vertex_pairs=pairs, energy_reference=energy_reference)
            blocks.update(zip(pairs, values))
        chi_ab = blocks.pop((A, B))
        if A and B:
            return _subtract_static_tt_contact(chi_ab)
        return chi_ab

    order = tuple(pair for pairs in classes for pair in pairs)
    return pack_photon_operator(get_block, nq, layout, mesh_xy, block_order=order)


#: Provenance of the unscreened-current packed body.  The bare-transverse
#: family declares NO current response model: the fifteen current blocks of
#: chi are zero, not approximated.
STATIC_PHOTON_BARE_CURRENT_MODEL = "bare_breit_no_current_response_v1"
STATIC_PHOTON_BARE_CURRENT_CONTACT = "none: current channels unscreened"


@dataclass(frozen=True)
class StaticPhotonResponse:
    """Declared no-pair full-body static photon response and provenance."""

    layout: object
    V_packed: jax.Array
    W_packed: jax.Array
    current_contact: str
    #: The approximation stamp: ``gamma_completed_*`` when the Gamma-cell
    #: completion ran, ``DEBUG_headless_*`` under ``head_correction = off``
    #: (see :func:`compute_static_photon_response`).  Required: no
    #: constructor may leave it at a default naming a mode that no longer
    #: exists.
    approximation: str
    head_completion: object | None = None
    current_model: str = STATIC_PHOTON_NO_PAIR_MODEL
    qgrid_policy: object = None
    family_plans: tuple = ()


def photon_blocks_full_q(packed, keys, *, layout, family_plans, qgrid_policy):
    """Restore each source once and apply the canonical Lorentz mixing for one class."""
    from symmetry_maps import unfold_isdf_operator, mix_lorentz_blocks
    from symmetry_maps import bgw_integer_q_to_fractional
    from .photon_layout import photon_block_view
    a, b = map(bool, keys[0])
    if any((bool(A), bool(B)) != (a, b) for A, B in keys):
        raise ValueError("A photon restore must contain one endpoint shape class.")
    left, right = family_plans[a], family_plans[b]
    sym, mesh, policy = left.sym, left.mesh_xy, qgrid_policy
    qfrac = bgw_integer_q_to_fractional(sym.q_irr_kgrid_int, policy.kgrid)
    pairs = tuple((C, D) for C in ((1, 2, 3) if a else (0,))
                  for D in ((1, 2, 3) if b else (0,)))
    parent_blocks = jnp.stack([photon_block_view(packed, layout, C, D, mesh)
                               for C, D in pairs])

    def restore(source):
        return unfold_isdf_operator(
            source, irr_idx=sym.irr_idx_q, sym_idx=policy.unfold_sym_idx,
            sym_perm=left.sym_perm, L_table=left.L_table,
            right_sym_perm=right.sym_perm, right_L_table=right.L_table,
            q_irr_frac=qfrac, mesh_xy=mesh, n_sym_spatial=policy.n_sym_spatial,
            axis_local_sym_perm=left.centroid_local_perm,
            right_axis_local_sym_perm=right.centroid_local_perm)

    restored = jax.lax.map(restore, parent_blocks)
    sources = {pair: restored[i] for i, pair in enumerate(pairs)}
    yield from mix_lorentz_blocks(sources, sym=sym, sym_idx=policy.unfold_sym_idx,
                                 mesh_xy=mesh, keys=keys).items()


def _load_static_photon_hall(
    config, meta, mesh_xy, wfn, wfn_fingerprint_binding, *,
    screen_current: bool, print_fn=print,
):
    """Load/authenticate the optional Hall artifact and gate its model.

    An unnamed ``static_gauge_hall_file`` is the declared ``sigma_H = 0``
    default.  A named path always reaches the one artifact loader, including
    the absent-path refusal.  The bare-transverse model admits an authenticated
    artifact only when its value is exactly zero: then the Hall response is
    identically absent and the packed operator is the same charge/TT block
    diagonal model as the unnamed case.  Any nonzero component still refuses.
    """
    hall_path = str(config.paths.static_gauge_hall_file).strip()
    if not hall_path:
        if jax.process_index() == 0:
            print_fn(
                "  [photon head] Hall term: sigma_H = 0 (no "
                "static_gauge_hall_file named).  For a Chern-trivial "
                "insulator the static Hall coefficient is exactly zero; "
                "supply the artifact from get_dipole_mtxels "
                "--static-gauge-hall-only to include a measured value.",
                flush=True)
        return None

    from file_io.static_gauge_head import load_static_gauge_hall_artifact
    hall = load_static_gauge_hall_artifact(
        hall_path,
        mesh_xy=mesh_xy,
        wfn=wfn,
        expected_band_start=int(meta.b_id_0),
        expected_band_stop=int(meta.b_id_4_chi_user),
        expected_nk_tot=int(meta.nk_tot),
        wfn_fingerprint_binding=wfn_fingerprint_binding,
    )
    sigma_h = np.asarray(jax.device_get(hall.sigma_H), dtype=np.float64)
    if not bool(screen_current) and np.any(sigma_h != 0.0):
        raise ValueError(
            "GATE packed_bare_transverse_hall_unavailable: a nonzero Hall "
            "artifact has no channel in the bare-transverse model.\n"
            f"  got:  static_gauge_hall_file = {hall_path}, "
            f"sigma_H = {sigma_h.tolist()} bohr^-1\n"
            "  want: an authenticated artifact with sigma_H = [0, 0, 0], "
            "an unnamed static_gauge_hall_file, or "
            "bispinor_gw = full_static_cohsex\n"
            "  why:  bare_transverse declares chi_TT = chi_CT = 0, hence "
            "W_CT = 0 at every finite q; a nonzero Gamma-only CT/TC block "
            "would not be a limit of that model\n"
            "  doc:  docs/input_reference.md, static_gauge_hall_file.")
    if jax.process_index() == 0:
        suffix = (
            " (exact zero is compatible with bare_transverse)"
            if not bool(screen_current) else "")
        print_fn(
            "  [photon head] Hall term: sigma_H = "
            f"{sigma_h.tolist()} bohr^-1 from authenticated {hall_path}"
            f"{suffix}",
            flush=True)
    return hall


def compute_static_photon_response(
    wfns_charge, wfns_transverse, quad, bispinor_v_q_path,
    meta, mesh_xy, *,
    screen_current: bool,
    mu_bases,
    W_charge=None,
    wfn=None,
    config=None,
    photon_g0_vectors=None,
    wf_binding_charge=None,
    wf_binding_transverse=None,
    wfn_fingerprint_binding=None,
    current_contact: str = _WARD_SUBTRACTED_NO_PAIR,
    energy_reference=0.0,
    dyson_solver: str = "distributed",
    distrib_la_batched_route: str = "batch_reshard",
    print_fn=print,
) -> StaticPhotonResponse:
    """Build the packed static photon body and complete its Gamma cell.

    THE SCREENING OWNER OF BOTH PACKED STATIC MODES.  ``screen_current``
    (resolved once by :func:`gw_config.packed_photon_screens_current`, never
    defaulted here) selects which:

    ``screen_current = True`` -- ``bispinor_gw = full_static_cohsex``: the
    sixteen no-pair blocks of ``chi``, one distributed Dyson solve at
    omega=0.

    ``screen_current = False`` -- the ``bare_transverse`` family: the twelve
    current blocks of ``chi`` are ZERO by declaration, so the packed Dyson
    equation is block diagonal and neither the current blocks nor the packed
    solve are built at all.  The CC block is screened by the incumbent
    scalar owner (``gw.screening.compute_screening_model`` -> :func:`solve_w`
    at ``n_C``) and arrives as ``W_charge``; this function assembles
    ``W_packed = diag(W_00, D_TT)`` with ``W_CT = 0`` through the sole
    packer.  The sixteen-block Sigma consumer then returns the screened
    charge COHSEX in CC, the bare Breit exchange ``Sigma^B`` in TT
    (``SX(D_TT) = X(D_TT)``, ``COH(D_TT - D_TT) = 0``) and zero in CT/TC --
    the incumbent ``gw.sigma_x_bispinor`` result, block for block.

    Both modes then run ONE Gamma-cell completion
    (:func:`gw.head_correction.complete_static_slab_photon_q0`) from the
    bounded response of
    :func:`gw.static_gauge_response.build_static_photon_head_response` --
    bare ``<D>`` into V, the charge ``S^{00}``/wing head into W, the Hall
    CT/TC term from ``config.paths.static_gauge_hall_file`` when that
    artifact exists (``sigma_H = 0`` otherwise, announced).  With the
    charge-only ``R(q)`` the coupled 4x4 solve returns
    ``diag(W^{00}_h(q), D_TT(q))``, so the same completion inserts the
    charge head AND the bare ``<D_TT> = -<v P^T>`` that the
    ``bispinor_tt_head_correction`` overlay writes into the TT V tiles on
    the incumbent route (that key is refused here, GATE
    ``packed_bare_transverse_tt_head_double_count``).  The Hall term needs a
    screened CT/TC channel to live in, so a nonzero Hall artifact is refused
    on the bare route; an authenticated exact-zero artifact is admitted and
    gives the same operator as the unnamed zero-Hall default.  The completion
    runs under ``head_correction = full`` (the
    default); ``off`` skips it behind a DEBUG banner and is not a production
    setting (owner ruling 2026-09-01).  The current q^2/contact/complement
    terms are omitted by model in either case.

    MEMORY.  Both modes keep the packed body resident: ``V_packed`` and
    ``W_packed`` are each ``(nq, N_packed, N_packed)`` complex128 at
    ``P(None,'x','y')`` with ``N_packed = n_C + 3 n_T``, i.e.
    ``16 nq N_packed^2 / P`` bytes per rank each.  The bare route's
    incumbent predecessor held one TT tile at a time instead, so this IS a
    new resident carrier for that route (it is the same object the screened
    mode already holds).  The figure is printed at this site below; the
    per-block streaming inside ``gw.photon_sigma`` is unchanged.

    ``print_fn`` is the driver's rank-zero printer.  In production mode the
    driver sinks ordinary component chatter, so the DEBUG banner below
    carries a WARNING token (retained in the run record's warning block)
    and the driver copies the completion / Hall status into its
    ``Photon head`` record line from the returned ``head_completion``.
    """
    from .gw_config import (
        BispinorGWMode, HeadCorrection,
        coerce_bispinor_gw_mode, packed_bare_transverse_route,
        packed_photon_screens_current)
    from .photon_layout import (
        PhotonBasisLayout, pack_photon_channel_vectors, photon_block_view,
        pack_photon_operator)
    from .v_q_bispinor import ZERO_TILES, BispinorVqReader

    if str(dyson_solver).strip().lower() != "distributed":
        raise ValueError(
            "packed static photon response requires "
            "dyson_solver='distributed'")
    if config is None:
        raise ValueError(
            "packed static photon response requires the run config: the "
            "head policy, the photon mode and the Hall artifact path are "
            "read from it, never defaulted here")
    head_policy = config.head.correction
    if head_policy not in (HeadCorrection.OFF, HeadCorrection.FULL):
        raise ValueError(
            "packed static photon response accepts only head_correction="
            f"full (the default) or the DEBUG value off; got {head_policy!r}")
    coupled_head = head_policy is HeadCorrection.FULL
    photon_mode = coerce_bispinor_gw_mode(
        getattr(config, "bispinor_gw", BispinorGWMode.BARE_TRANSVERSE))
    screen_current = bool(screen_current)
    if screen_current != packed_photon_screens_current(config):
        raise ValueError(
            "packed static photon response received screen_current="
            f"{screen_current} for bispinor_gw={photon_mode.value!r}; the "
            "selector is resolved once by "
            "gw_config.packed_photon_screens_current and must not be "
            "restated at the call site")
    if not screen_current:
        if photon_mode is not BispinorGWMode.BARE_TRANSVERSE:
            raise ValueError(
                "the unscreened-current packed route serves only "
                f"bare_transverse; got {photon_mode.value!r}")
        route_taken, route_reason = packed_bare_transverse_route(config)
        if not route_taken:
            raise ValueError(
                "packed static photon response reached with "
                f"bispinor_gw={photon_mode.value!r} outside its envelope: "
                f"{route_reason}")
        if W_charge is None:
            raise ValueError(
                "the packed bare-transverse route requires the incumbent "
                "scalar W(omega=0) on the charge block; refusing to build a "
                "second charge screening owner here")
    elif photon_mode is not BispinorGWMode.FULL_STATIC_COHSEX:
        raise ValueError(
            "packed static photon response received bispinor_gw="
            f"{photon_mode.value!r}")
    elif W_charge is not None:
        raise ValueError(
            "full_static_cohsex screens the charge block inside the packed "
            "Dyson solve; an external W_charge would be ignored")
    hall = None
    if coupled_head:
        if (wfn is None or photon_g0_vectors is None
                or wf_binding_charge is None
                or wf_binding_transverse is None
                or wfn_fingerprint_binding is None):
            raise ValueError(
                "the packed static photon completion requires the fresh "
                "WFN, four Gamma vectors, and both wavefunction-basis "
                "bindings")
        for binding, carrier, role in (
                (wf_binding_charge, wfns_charge, "charge"),
                (wf_binding_transverse, wfns_transverse, "transverse")):
            if (binding.wavefunctions is not carrier
                    or binding.receipt.role != role):
                raise ValueError(
                    f"packed static photon {role} wavefunction binding does "
                    "not name the supplied carrier")
        hall = _load_static_photon_hall(
            config, meta, mesh_xy, wfn, wfn_fingerprint_binding,
            screen_current=screen_current, print_fn=print_fn)
    elif jax.process_index() == 0:
        print_fn(
            "\n  ==========================================================\n"
            "  WARNING -- DEBUG: Gamma-cell head disabled by "
            "head_correction=off\n"
            "  The packed static photon V and W keep a ZERO q=Gamma, G=0\n"
            "  slot: no bare <D> insertion, no charge S00/wing head, no\n"
            "  Hall term.  This is a brute-force k-grid convergence /\n"
            "  debugging setting, NOT a production calculation\n"
            "  (owner ruling 2026-09-01, docs/architecture/decisions.md).\n"
            "  ==========================================================\n",
            flush=True)

    plans = (wfns_charge.green_parent.plan, wfns_transverse.green_parent.plan)
    sym = plans[0].sym
    from .qgrid_symmetry import qgrid_trs_policy_for
    policy = qgrid_trs_policy_for(
        sym=sym, irr_idx_q=sym.irr_idx_q, sym_idx_q=sym.sym_idx_q,
        kgrid=tuple(meta.kgrid), n_sym_spatial=plans[0].n_sym_spatial,
        context="packed photon response")
    nq = len(sym.q_irr_full_idx)
    from symmetry_maps import bgw_integer_q_to_fractional
    qfrac = bgw_integer_q_to_fractional(sym.q_irr_kgrid_int, meta.kgrid)
    with BispinorVqReader(bispinor_v_q_path, mesh_xy, mu_bases=mu_bases,
                         family_plans=plans) as reader:
        if reader.n_q_total != nq:
            raise ValueError("Photon V and response disagree on q-IBZ extent.")
        layout = PhotonBasisLayout.from_centroid_extents(
            plans[0].n_centroid_packed, plans[1].n_centroid_packed, mesh_xy)
        V_packed = pack_photon_operator(
            lambda A, B: None if (A, B) in ZERO_TILES else reader.get_tile(A, B),
            nq, layout, mesh_xy)

    if jax.process_index() == 0:
        # MEASURED at the site, per lane C's design note: the packed body is
        # the resident carrier of BOTH modes and is new to the bare route.
        n_packed = int(layout.packed_extent)
        n_ranks = max(int(jax.process_count()), 1)
        body_bytes = 16.0 * nq * n_packed * n_packed / n_ranks
        print_fn(
            "  [photon response] DECLARED no-pair model "
            "Psi=(Psi_L,(alpha_FS/2)*sigma.p*Psi_L), "
            "j=c*Psi^dagger*alpha*Psi; "
            + ("bubble-screened Breit; "
               f"current_contact={current_contact}; "
               if screen_current else
               "BARE Breit (chi_TT = chi_CT = 0: no current blocks, no "
               "packed Dyson solve); ")
            + f"head_correction={head_policy.value}",
            flush=True)
        print_fn(
            f"  [photon response] packed body N_packed={n_packed} "
            f"(n_C={layout.carrier_extent(0)} + 3*n_T="
            f"{layout.carrier_extent(1)}), nq={nq}: "
            f"{body_bytes / 1e9:.4f} GB/rank resident for EACH of V and W",
            flush=True)
    if screen_current:
        chi_packed = compute_experimental_no_pair_photon_chi0(
            wfns_charge, wfns_transverse, quad, meta, mesh_xy, layout,
            current_contact=current_contact,
            energy_reference=energy_reference)

        W_packed = solve_w(
            V_packed, chi_packed, meta, mesh_xy,
            dyson_solver="distributed",
            # Direct sum of already-padded C/T channel blocks: unlike the
            # scalar carrier, this space has no one logical prefix to mask.
            n_rmu_logical=int(layout.packed_extent),
            distrib_la_batched_route=distrib_la_batched_route)
        # The bare Hartree/exchange stage that follows does not depend on W
        # and could otherwise begin allocating its Green/operator workspaces
        # while the asynchronous distributed LU still owns A, RHS and donated
        # chi.  Finish the response here, inside its timing/lifetime boundary.
        W_packed.block_until_ready()
    else:
        # chi_TT = chi_CT = 0 makes the packed Dyson equation block diagonal:
        #     W_packed = diag((1 - D_00 chi_00)^-1 D_00, D_TT),  W_CT = 0.
        # Neither the fifteen current blocks of chi nor the (n_C + 3 n_T)^2
        # solve is built.  The CC block was solved by the incumbent scalar
        # owner at n_C; the rest of W is V, block for block, through the sole
        # packer -- one local write per block, no gather.
        # The screened branch checks this inside
        # compute_experimental_no_pair_photon_chi0; the bare branch builds no
        # chi, so assert it here rather than let photon_sigma fail on a
        # block/bundle shape mismatch several stages later.
        from .wavefunction_bundle import padded_centroid_extent
        n_c = padded_centroid_extent(wfns_charge)
        n_t = padded_centroid_extent(wfns_transverse)
        if (n_c != layout.carrier_extent(0) or n_t != layout.carrier_extent(1)):
            raise ValueError(
                "photon layout padded extents do not match wavefunction "
                f"bundles: layout C/T=({layout.carrier_extent(0)},"
                f"{layout.carrier_extent(1)}), wfns C/T=({n_c},{n_t})")
        W_cc = jnp.take(jnp.asarray(W_charge),
                        jnp.asarray(sym.q_irr_full_idx), axis=0)
        expected_cc = layout.block_shape(nq, 0, 0)
        if tuple(W_cc.shape) != expected_cc:
            raise ValueError(
                "the packed bare-transverse route needs the incumbent scalar "
                f"W on the CC block: expected {expected_cc} from the photon "
                f"layout, got {tuple(W_cc.shape)}.  The charge centroid "
                "padding of the scalar and packed paths must agree.")
        with mesh_xy:
            W_cc = jax.lax.with_sharding_constraint(
                W_cc.astype(V_packed.dtype),
                NamedSharding(mesh_xy, P(None, "x", "y")))

        def _bare_W_block(A, B):
            if (int(A), int(B)) == (0, 0):
                return W_cc
            if (int(A), int(B)) in ZERO_TILES:
                # Coulomb-gauge zero: V_CT = 0 and chi_CT = 0, so W_CT = 0.
                return None
            return photon_block_view(V_packed, layout, A, B, mesh_xy)

        W_packed = pack_photon_operator(
            _bare_W_block, nq, layout, mesh_xy)
        W_packed.block_until_ready()
        del W_cc
    # This packed path bypasses screening._gate_w, so apply its two valid
    # static stage invariants here through the shared sanity owner.  Full-q
    # scalar reciprocity is deliberately not borrowed: Lorentz current
    # channels transform as vectors and require their own derived relation.
    from common import sanity
    sanity.refuse_nonfinite("static packed photon W", W_packed)
    if not sanity.check_hermitian(
            "static packed photon W[q=0]", W_packed[0], rtol=1.0e-6,
            always=True):
        raise ValueError(
            "static packed photon W[q=0] failed the canonical Hermiticity "
            "gate before coupled head/body folding")

    head_completion = None
    if coupled_head:
        from vcoul import (
            CoulombGeometry, get_kernel, slab_minibz_photon_cubature)
        from .head_correction import complete_static_slab_photon_q0
        from .static_gauge_response import (
            build_static_photon_head_response)

        response = build_static_photon_head_response(
            wfns_charge,
            input_dir=config.input_dir,
            mesh=mesh_xy,
            wfn=wfn,
            meta=meta,
            config=config,
            layout=layout,
            hall_transaction=hall,
            wfn_fingerprint_binding=wfn_fingerprint_binding,
        )
        if len(photon_g0_vectors) != 4:
            raise ValueError(
                "the packed static photon completion requires four "
                "literal-Gamma vectors")
        g0_X = pack_photon_channel_vectors(
            tuple(photon_g0_vectors), layout, mesh_xy, axis_name="x")[0]
        y_sharding = NamedSharding(mesh_xy, P(None, "y"))
        g0_Y = pack_photon_channel_vectors(
            tuple(device_put_process_local(vector, y_sharding)
                  for vector in photon_g0_vectors),
            layout, mesh_xy, axis_name="y")[0]
        geometry = CoulombGeometry.from_wfn(wfn)
        cubature = slab_minibz_photon_cubature(
            get_kernel(2), geometry, tuple(int(v) for v in meta.kgrid))
        V_packed, W_packed, head_completion = (
            complete_static_slab_photon_q0(
                V_packed, W_packed, response, g0_X, g0_Y, cubature,
                mesh_xy=mesh_xy, family_plans=plans))
        jax.block_until_ready((V_packed, W_packed))
        sanity.refuse_nonfinite(
            "Gamma-completed static photon V", V_packed)
        sanity.refuse_nonfinite(
            "Gamma-completed static photon W", W_packed)
        if jax.process_index() == 0:
            print_fn(
                "  [photon head] Gamma-cell completion applied: bare <D> "
                "into V, charge S00/wing head into W; hall_source="
                f"{response.hall_source}; ward={head_completion.ward_residual:.3e}, "
                f"hermiticity={head_completion.hermiticity_residual:.3e}, "
                "dyson_forward_bound="
                f"{head_completion.max_dyson_forward_error_bound:.3e}",
                flush=True)

    if screen_current:
        return StaticPhotonResponse(
            layout=layout, V_packed=V_packed, W_packed=W_packed,
            current_contact=current_contact,
            qgrid_policy=policy, family_plans=plans,
            head_completion=head_completion,
            current_model=STATIC_PHOTON_NO_PAIR_MODEL,
            approximation=(
                "gamma_completed_no_pair_static_photon_v1"
                if coupled_head
                else "DEBUG_headless_no_pair_static_photon_v1"),
        )
    return StaticPhotonResponse(
        layout=layout, V_packed=V_packed, W_packed=W_packed,
        current_contact=STATIC_PHOTON_BARE_CURRENT_CONTACT,
        qgrid_policy=policy, family_plans=plans,
        head_completion=head_completion,
        current_model=STATIC_PHOTON_BARE_CURRENT_MODEL,
        approximation=(
            "gamma_completed_bare_transverse_photon_v1"
            if coupled_head
            else "DEBUG_headless_bare_transverse_photon_v1"),
    )


def _chi0_multi_kernel_args(wfns, tau, alpha_rows, energy_reference):
    """Shared host prep for the multi-output χ₀ paths (compute + precompile).

    ``tau``: (L,) node vector (the fused static∪extra union on the probe-
    reuse path).  ``alpha_rows``: (n_out, L) RAW quadrature weights, one
    row per output, all on ``tau``.  Row 0 is normally the static weights
    (zero-padded onto any extra nodes — zero-weight nodes add exact
    zeros); further rows are probe representations on the same nodes.
    The one-orientation prefactor ``-exp(-τ·E_gap)`` folds into every row;
    the kernel adds the reverse ordered transition through the shared
    R-space orientation combiner exactly as the single-output path does.
    """
    s = wfns.slices
    enk_v = wfns.enk[:, s.val]
    enk_c = wfns.enk[:, s.cond]
    eref = 0.0 if energy_reference is None else float(energy_reference)
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64) - eref
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64) - eref
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    E_gap = cmin - vmax
    tau = np.asarray(tau, dtype=np.float64)
    alpha_rows = np.asarray(alpha_rows, dtype=np.float64)
    if alpha_rows.ndim != 2 or alpha_rows.shape[1] != tau.shape[0]:
        raise ValueError(
            f"chi0 multi: alpha_rows shape {alpha_rows.shape} does not "
            f"match tau nodes ({tau.shape[0]},) — every row must be a "
            f"weight vector on quad.tau.")
    alpha_chi = -1.0 * alpha_rows * np.exp(-tau * E_gap)[None, :]
    nodes = MinimaxNodes(
        t=jnp.asarray(tau, dtype=jnp.complex128),
        alpha=jnp.asarray(alpha_chi, dtype=jnp.complex128),
    )
    args = (
        nodes, *_chi_layout_operands(wfns, eref),
        jnp.asarray(vmax, dtype=jnp.float64),
        jnp.asarray(cmin, dtype=jnp.float64),
    )
    return args, alpha_rows.shape[0]


def compute_chi0_multi(wfns, tau, alpha_rows, meta, mesh_xy, *,
                       energy_reference=0.0):
    """χ₀ at several weight vectors over ONE τ sweep — see
    ``_get_chi_minimax_kernel(n_out>=2)``.  Returns an ``n_out``-tuple of
    flat-q (nq, μ, μ) arrays, one per row of ``alpha_rows``."""
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    args, n_out = _chi0_multi_kernel_args(
        wfns, tau, alpha_rows, energy_reference)
    kernel = _get_chi_minimax_kernel(mesh_xy, kgrid, n_out=n_out,
                                     **_chi_parent_face_kwargs(wfns))
    return kernel(*args)


def precompile_chi0_multi(wfns, tau, alpha_rows, meta, mesh_xy, *,
                          energy_reference=None):
    """AOT lower+compile sibling of :func:`precompile_chi0` for the
    multi-output kernel."""
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    if len(np.asarray(tau)) == 0:
        return
    args, n_out = _chi0_multi_kernel_args(
        wfns, tau, alpha_rows, energy_reference)
    kernel = _get_chi_minimax_kernel(mesh_xy, kgrid, n_out=n_out,
                                     **_chi_parent_face_kwargs(wfns))
    kernel.lower(*args).compile()


def _chi0_contour_alpha_rows(tau, weight_rows, frequency_sign, z_values,
                             E_gap):
    """Complete contour weights for both independent-particle resolvents.

    ``frequency_sign=+1`` represents ``-1/(Delta-z)`` and ``-1`` represents
    ``-1/(Delta+z)``.  The device kernel evolves ``Delta-E_gap``, so this
    host-side coefficient supplies the omitted gap and requested frequency.
    """
    tau = np.asarray(tau, dtype=np.complex128)
    weight_rows = np.asarray(weight_rows, dtype=np.complex128)
    frequency_sign = np.asarray(frequency_sign)
    z_values = np.asarray(z_values, dtype=np.complex128)
    if (tau.ndim != 1 or z_values.ndim != 1 or z_values.size == 0 or
            frequency_sign.shape != tau.shape or
            weight_rows.shape != (z_values.size, tau.size)):
        raise ValueError(
            "chi0 contour requires tau/sign (L,), z (n_out,), and "
            "weight_rows (n_out,L)")
    if not np.all(np.isin(frequency_sign, (-1, 1))):
        raise ValueError("chi0 contour frequency_sign must contain only +/-1")
    exponent = -tau[None, :] * (
        float(E_gap) - frequency_sign[None, :] * z_values[:, None])
    return -weight_rows * np.exp(exponent)


def _chi0_contour_kernel_args(wfns, tau, weight_rows, frequency_sign,
                              z_values, energy_reference):
    """Prepare complex-frequency rows and the existing sharded operands."""
    s = wfns.slices
    enk_v = wfns.enk[:, s.val]
    enk_c = wfns.enk[:, s.cond]
    eref = 0.0 if energy_reference is None else float(energy_reference)
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64) - eref
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64) - eref
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    tau = np.asarray(tau, dtype=np.complex128)
    alpha_rows = _chi0_contour_alpha_rows(
        tau, weight_rows, frequency_sign, z_values, cmin - vmax)
    nodes = MinimaxNodes(
        t=jnp.asarray(tau, dtype=jnp.complex128),
        alpha=jnp.asarray(
            alpha_rows[0] if alpha_rows.shape[0] == 1 else alpha_rows,
            dtype=jnp.complex128),
    )
    args = (
        nodes, *_chi_layout_operands(wfns, eref),
        jnp.asarray(vmax, dtype=jnp.float64),
        jnp.asarray(cmin, dtype=jnp.float64),
    )
    return args, alpha_rows.shape[0]


def compute_chi0_contour(wfns, tau, weight_rows, frequency_sign, z_values,
                         meta, mesh_xy, *, energy_reference=0.0):
    """Evaluate several complex-frequency chi0 values in one node sweep.

    The scalar contour arrays select the two ``Delta +/- z`` resolvents.  All
    Green-function construction, FFTs, contraction, and sharding are the same
    operations used by :func:`compute_chi0`.
    """
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    args, n_out = _chi0_contour_kernel_args(
        wfns, tau, weight_rows, frequency_sign, z_values, energy_reference)
    kernel = _get_chi_minimax_kernel(
        mesh_xy, kgrid, n_out=n_out, complex_contour=True,
        **_chi_parent_face_kwargs(wfns))
    return kernel(*args)


def compute_chi0_contour_ordered(
    wfns,
    time,
    weights,
    z_values,
    meta,
    mesh_xy,
    *,
    q_neg_index,
    energy_reference=0.0,
    return_reflected=False,
):
    r"""Evaluate magnetic contour samples with both ordered orientations.

    For an upper-half-plane sample ``z`` the independent-particle response is

    ``chi0_q(z) = F_q(z) + conj(F_{-q}(-conj(z)))``,

    where the kernel's native orientation is
    ``F_q(z) = -P_q/(z+Delta)``.  Both ``F(z)`` and
    ``F(-conj(z))`` are outputs of ONE contour sweep through the existing
    response kernel.  The second orientation is then a flat-q negation
    gather and conjugation; no second response kernel is evaluated and no
    large intermediate is rematerialized on fewer than all processors.

    This is the complex-contour analogue of
    :func:`compute_chi0_imag_ordered`.  The incumbent
    :func:`compute_chi0_contour` applies the two scalar resolvents to the same
    transition orientation, which is valid after a time-reversal completion
    but deletes the magnetisation-odd channel when time reversal is broken.
    Callers therefore select this route only from ``SymMaps.trs_allowed``.

    Parameters
    ----------
    wfns
        Wavefunction bundle.  Its flat k axis remains sharded as in the
        ordinary contour kernel.
    time, weights
        Positive real-time quadrature nodes and weights, shape ``(L,)``, in
        reciprocal-energy and time units respectively.
    z_values
        Upper-half-plane complex frequencies, shape ``(n_z,)``, in the same
        energy unit used by ``wfns.enk``.
    meta, mesh_xy
        Runtime metadata and the two-dimensional processor mesh.
    q_neg_index
        Public ``symmetry_maps.q_negation_index`` permutation, shape
        ``(n_q,)``.  It must be an involution on the complete flat q grid.
    energy_reference
        Common energy origin subtracted from valence and conduction bands.
    return_reflected
        When true, also return the independently completed response at
        ``-conj(z)``.  Both orientations already belong to the same contour
        sweep; this option exposes the second completion without evaluating
        another response kernel.  The default preserves the incumbent return
        object exactly.

    Returns
    -------
    jax.Array or tuple[jax.Array, ...]
        One flat-q ``(n_q, n_mu, n_mu)`` response for one frequency, or an
        ``n_z`` tuple for several frequencies.  Arrays retain
        ``P(None, 'x', 'y')`` sharding.
    """
    time = np.asarray(time, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    z = np.asarray(z_values, dtype=np.complex128)
    if (time.ndim != 1 or time.size == 0 or weights.shape != time.shape
            or z.ndim != 1 or z.size == 0):
        raise ValueError(
            "compute_chi0_contour_ordered: time/weights must be nonempty "
            f"(L,) arrays and z_values a nonempty (n_z,) array; got "
            f"{time.shape}, {weights.shape}, {z.shape}.")
    if (not np.all(np.isfinite(time)) or not np.all(np.isfinite(weights))
            or not np.all(np.isfinite(z)) or np.any(time <= 0.0)
            or np.any(np.imag(z) <= 0.0)):
        raise ValueError(
            "GATE chi0_contour_ordered_domain: nodes and weights must be "
            "finite, every time node must be positive, and every z value "
            "must be finite with Im(z) > 0. FALSE case: the damped-line "
            "quadrature and upper-half-plane MPA samples satisfy all three "
            "conditions.")

    # The reflected point remains in the upper half plane.  Both rows use
    # the kernel's native ``-1/(z+Delta)`` orientation: negative imaginary
    # time, frequency sign -1, and weights -i*h.  This is the orientation
    # whose imaginary-axis limit is exactly compute_chi0_imag_ordered's
    # ``-(alpha-i*beta)`` carrier.  Their partner relation is applied only
    # after this single response-kernel invocation.
    z_sweep = np.concatenate((z, -np.conj(z)))
    tau = -1j * time
    signs = -np.ones(time.size, dtype=np.int8)
    weight_rows = np.broadcast_to(
        -1j * weights, (z_sweep.size, time.size))

    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    args, n_out = _chi0_contour_kernel_args(
        wfns, tau, weight_rows, signs, z_sweep, energy_reference)
    kernel = _get_chi_minimax_kernel(
        mesh_xy, kgrid, n_out=n_out, complex_contour=True,
        **_chi_face_kwargs(wfns))
    orientations = kernel(*args)
    if not isinstance(orientations, tuple):  # n_out == 2*n_z >= 2
        orientations = (orientations,)

    q_neg = np.asarray(q_neg_index)
    nq = int(orientations[0].shape[0])
    if (q_neg.shape != (nq,)
            or np.any(q_neg < 0) or np.any(q_neg >= nq)
            or not np.array_equal(q_neg[q_neg], np.arange(nq))):
        raise ValueError(
            "compute_chi0_contour_ordered: q_neg_index must be an "
            f"involution over the full flat-q axis [0, {nq}); got shape "
            f"{q_neg.shape}.")
    q_neg_jax = jnp.asarray(q_neg, dtype=jnp.int32)
    completed = tuple(
        _complete_contour_ordered(
            orientations[i], orientations[z.size + i], q_neg_jax)
        for i in range(z.size)
    )
    primary = completed[0] if z.size == 1 else completed
    if not return_reflected:
        return primary
    reflected = tuple(
        _complete_contour_ordered(
            orientations[z.size + i], orientations[i], q_neg_jax)
        for i in range(z.size)
    )
    reflected = reflected[0] if z.size == 1 else reflected
    return primary, reflected


@jax.jit
def _complete_contour_ordered(F_q_z, F_q_reflected, q_neg):
    """``F_q(z) + conj(F_{-q}(-conj(z)))``; sharding-preserving."""
    return F_q_z + jnp.conj(jnp.take(F_q_reflected, q_neg, axis=0))


def precompile_chi0_contour(wfns, tau, weight_rows, frequency_sign,
                            z_values, meta, mesh_xy, *,
                            energy_reference=None):
    """AOT sibling of :func:`compute_chi0_contour`."""
    if len(np.asarray(tau)) == 0:
        return
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    args, n_out = _chi0_contour_kernel_args(
        wfns, tau, weight_rows, frequency_sign, z_values, energy_reference)
    kernel = _get_chi_minimax_kernel(
        mesh_xy, kgrid, n_out=n_out, complex_contour=True,
        **_chi_parent_face_kwargs(wfns))
    kernel.lower(*args).compile()


def _occupation_support_slices(
        occupations,
        occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT):
    """Smallest contiguous f and (1-f) band supports without truncation.

    THIS IS THE ONE PLACE χ₀'s TWO GREEN'S FUNCTIONS GET THEIR BANDS, and
    unlike the Σ planner's mask it is a genuine COST cut: the returned slices
    index ``wfns.xn``/``yr``, so a band outside them is absent from the
    ``build_G_tau`` contraction rather than merely multiplied by a small
    weight.  ``occupation_support_bandwidth`` reads the same two slices to
    size the damped-line rule, so widening them also buys quadrature nodes.

    ``occupation_window_threshold`` is the OCCUPANCY at which a band leaves a
    support; the cut is on the branch WEIGHT — ``f`` on the occupied side,
    ``1 − f`` on the empty side, matching ``band_weight=occ_f`` and
    ``band_weight=1.0 - occ_u`` in the kernel — at the floor
    ``1 − threshold``, by MAGNITUDE.  Nothing is clipped: MP1 occupations
    overshoot [0, 1] and a wrong-side band's NEGATIVE weight is kept by
    ``abs`` exactly as the historical rule kept it (the argument is at
    ``gw.efermi.band_in_occupation_window``).  Partially occupied bands
    belong to both slices, as before.

    ``threshold = 1.0`` gives floor 0.0 and restores the historical exact
    rule (``occ != 0`` / ``occ != 1``) bit-for-bit; an insulating table, whose
    weights are exactly 0 or 1, gives the same two slices at EVERY threshold,
    since ``abs(1) > floor`` and ``abs(0) > floor`` are threshold-independent
    on [0.5, 1.0].
    """
    occ = np.asarray(jax.device_get(occupations), dtype=np.float64)
    if occ.ndim != 2:
        raise ValueError(
            "fractional contour occupations must have shape (nk, nb), got "
            + str(occ.shape))
    floor = occupation_weight_floor(occupation_window_threshold)
    f_support = np.any(band_in_occupation_window(occ, floor), axis=0)
    u_support = np.any(band_in_occupation_window(1.0 - occ, floor), axis=0)
    if not np.any(f_support) or not np.any(u_support):
        raise ValueError(
            "fractional contour chi0 needs at least one band clearing the "
            f"occupation window on each side (threshold "
            f"{float(occupation_window_threshold)!r} ⇒ |weight| > {floor!r}); "
            "raise occupation_window_threshold toward 1.0 to widen the "
            "supports, or check the occupation table")
    f_idx = np.flatnonzero(f_support)
    u_idx = np.flatnonzero(u_support)
    return (
        slice(int(f_idx[0]), int(f_idx[-1]) + 1),
        slice(int(u_idx[0]), int(u_idx[-1]) + 1),
    )


def _chi0_fractional_contour_args(
    wfns,
    time_nodes,
    weight_rows,
    z_values,
    occupations,
    energy_reference,
    occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
):
    """Prepare final band weights on the occupied and unoccupied support windows."""
    time_nodes = np.asarray(time_nodes, dtype=np.float64)
    z_values = np.asarray(z_values, dtype=np.complex128)
    if time_nodes.ndim != 1 or time_nodes.size == 0:
        raise ValueError(
            "fractional contour time_nodes must be a nonempty 1-D array")
    if not np.all(np.isfinite(time_nodes)) or np.any(time_nodes < 0.0):
        raise ValueError(
            "fractional contour time_nodes must be finite and nonnegative")
    if z_values.ndim != 1 or z_values.size == 0:
        raise ValueError(
            "fractional contour z_values must be a nonempty 1-D array")
    if np.any(z_values.imag <= 0.0):
        raise ValueError(
            "fractional retarded contour requires Im(z) > 0; the exact static "
            "divided-difference limit is a separate quadrature target")

    weights = np.asarray(weight_rows, dtype=np.complex128)
    if weights.ndim == 1:
        if weights.shape != time_nodes.shape:
            raise ValueError(
                "fractional contour 1-D weights must match time_nodes")
        weights = np.broadcast_to(
            weights[None, :], (z_values.size, time_nodes.size))
    if weights.shape != (z_values.size, time_nodes.size):
        raise ValueError(
            "fractional contour weights must have shape (n_z, n_time)")
    projection_rows = weights * np.exp(
        1j * z_values[:, None] * time_nodes[None, :])

    occ_full = wfns.occ if occupations is None else occupations
    if tuple(occ_full.shape) != tuple(wfns.enk.shape):
        raise ValueError(
            "fractional contour occupation shape {} does not match energies "
            "{}".format(occ_full.shape, wfns.enk.shape))
    f_slice, u_slice = _occupation_support_slices(
        occ_full, occupation_window_threshold)
    eref = 0.0 if energy_reference is None else float(energy_reference)
    # Invert occupations before masking so excluded bands retain zero weight.
    nb_full = int(wfns.slices.nb_full)
    idx = np.arange(nb_full)
    f_ind = jnp.asarray(
        (idx >= f_slice.start) & (idx < f_slice.stop), dtype=occ_full.dtype)
    u_ind = jnp.asarray(
        (idx >= u_slice.start) & (idx < u_slice.stop), dtype=occ_full.dtype)
    occ_f_face = occ_full * f_ind[None, :]
    occ_u_face = (1.0 - occ_full) * u_ind[None, :]
    carrier = wfns.green_parent
    if carrier is not None:
        # Raw parents: the packed parent faces and the parents' rows of
        # the (star-invariant) energy and weight tables; the kernel built
        # with the matching ``k_unfold_plan`` transports G to full k.
        plan = carrier.plan
        args = (
            jnp.asarray(time_nodes, dtype=jnp.float64),
            jnp.asarray(projection_rows, dtype=jnp.complex128),
            carrier.psi_mun,
            carrier.psi_nmu,
            carrier.enk,
            plan.parent_rows(occ_f_face),
            plan.parent_rows(occ_u_face),
            jnp.asarray(eref, dtype=jnp.float64),
        )
        return args, z_values.size
    args = (
        jnp.asarray(time_nodes, dtype=jnp.float64),
        jnp.asarray(projection_rows, dtype=jnp.complex128),
        wfns.psi_mun,
        wfns.psi_nmu,
        wfns.enk,
        occ_f_face,
        occ_u_face,
        jnp.asarray(eref, dtype=jnp.float64),
    )
    return args, z_values.size


def compute_chi0_contour_fractional(
    wfns,
    time_nodes,
    weight_rows,
    z_values,
    meta,
    mesh_xy,
    *,
    occupations=None,
    energy_reference=0.0,
    occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
):
    """Evaluate retarded finite-occupation chi0 at complex frequencies.

    weight_rows contains the positive real-time quadrature weights; this
    routine supplies exp(i*z*t) and both exact Keldysh terms.  It does not
    implement z=0: the gapless static limit contains the finite divided
    difference -df/dE and requires its own certified integration rule.

    ``occupation_window_threshold`` is the OCCUPANCY at which a band leaves
    one of the two Green's-function supports; it MUST be the same value the
    caller gave ``occupation_support_bandwidth``, or the damped-line rule is
    sized for transitions the band slices no longer contain.
    """
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    args, n_out = _chi0_fractional_contour_args(
        wfns,
        time_nodes,
        weight_rows,
        z_values,
        occupations,
        energy_reference,
        occupation_window_threshold,
    )
    kernel = _get_chi_fractional_contour_kernel(
        mesh_xy, kgrid, n_out, **_chi_parent_face_kwargs(wfns))
    values = kernel(*args)
    return values[0] if n_out == 1 else values


# ============================================================================
# Exact finite-occupation response — face-layout ordered-pair kernel
# ============================================================================
#
def _fractional_pair_scan_face(
    psi_mun_a, psi_nmu_a, psi_mun_b, psi_nmu_b, energy_a, energy_b,
    occ_a, occ_b, surface_a, surface_b, z_values, *,
    nb_full, nb_logical, tile, unfold_x=None, unfold_y=None, roll_b=None,
    k_unfold_plan=None,
):
    """Stream ordered band-pair tiles from canonical faces with optional typed parent transport."""
    # The k extent of the PAIR SUM is the energy table's (full BZ).  With
    # raw parents (``unfold_x``/``unfold_y`` given) the ψ operands carry
    # n_parent rows and every band tile is unfolded to full k after its
    # gather -- symmetry_maps.unfold_wavefunction_local, tables already
    # sliced to this rank's μ slab -- so the tile, not a face, is the
    # largest full-k ψ object that ever exists.  ``roll_b`` is the k−q map
    # applied to the "b" role AFTER the unfold (full-k rows).
    nk = energy_a.shape[0]
    ns = psi_mun_a.shape[1]
    nmu_x_loc = psi_mun_a.shape[2]
    nmu_y_loc = psi_nmu_a.shape[3]
    if (unfold_x is None) != (unfold_y is None):
        raise ValueError(
            "_fractional_pair_scan_face: give unfold tables for BOTH faces "
            "or neither.")

    def _children(tile_psi, tables):
        """(n_parent, s, mu_loc, t) -> (nk, s, mu_loc, t) by the typed action;
        identity when the operand already spans full k."""
        if tables is None:
            return tile_psi
        return k_unfold_plan.unfold_face(
            tile_psi, spin_axis=1, mu_axis=2, tables=tables)

    def _roll(tile_psi):
        return tile_psi if roll_b is None else jnp.take(tile_psi, roll_b, axis=0)
    shard_w_y = psi_mun_a.shape[3]   # psi_mun's own 'y'-shard width (bands)
    shard_w_x = psi_nmu_a.shape[1]   # psi_nmu's own 'x'-shard width (bands)
    y_idx = jax.lax.axis_index('y')
    x_idx = jax.lax.axis_index('x')

    def _gather_mun(psi_mun_local, g_lo):
        """(nk, s, mu_X_loc, tile) un-conjugated, present on every rank —
        masked-gather + psum('y') from psi_mun's local shard (bands on
        'y').  psi_mun's own axis order (nk, s, mu, n) already matches
        the direct endpoint (nk, s, mu_X, n) -- no reorder needed."""
        p = jnp.arange(tile, dtype=jnp.int32)
        global_band = g_lo + p
        if shard_w_y == nb_full:
            return jnp.take(psi_mun_local, jnp.clip(global_band, 0, nb_full - 1), axis=3)
        owner = global_band // shard_w_y
        owns = owner == y_idx
        local_idx = jnp.clip(
            global_band - y_idx * shard_w_y, 0, shard_w_y - 1)
        gathered = jnp.take(psi_mun_local, local_idx, axis=3)
        gathered = jnp.where(owns[None, None, None, :], gathered, 0)
        return jax.lax.psum(gathered, 'y')

    def _gather_nmu(psi_nmu_local, g_lo):
        """(nk, s, mu_Y_loc, tile) un-conjugated, present on every rank —
        masked-gather + psum('x') from psi_nmu's local shard (bands on
        'x'), then a LOCAL (no-comm, bounded-size — this tile is `tile`
        bands wide, not nb_full) axis reorder: psi_nmu stores (nk, n, s,
        mu), band axis SECOND, so the post-gather (nk, tile, s, mu_Y_loc)
        needs one transpose to match the band-last endpoint (nk, s, mu, n) order."""
        p = jnp.arange(tile, dtype=jnp.int32)
        global_band = g_lo + p
        if shard_w_x == nb_full:
            gathered = jnp.take(psi_nmu_local, jnp.clip(global_band, 0, nb_full - 1), axis=1)
            return jnp.transpose(gathered, (0, 2, 3, 1))
        owner = global_band // shard_w_x
        owns = owner == x_idx
        local_idx = jnp.clip(
            global_band - x_idx * shard_w_x, 0, shard_w_x - 1)
        gathered = jnp.take(psi_nmu_local, local_idx, axis=1)
        gathered = jnp.where(owns[None, :, None, None], gathered, 0)
        gathered = jax.lax.psum(gathered, 'x')
        return jnp.transpose(gathered, (0, 2, 3, 1))

    # ``nb_full`` is a storage extent and can include rank-padding above the
    # physical chi window.  Do not execute empty O(nb**2) tile pairs merely
    # because the carrier is wider than ``nb_logical``.  Keep enough backing
    # storage for a final partial tile, but size the scans from the logical
    # sum extent.
    from runtime.padding import padded_axis
    nb_pad = padded_axis(
        int(nb_logical), tile,
        name="face fractional-response band carrier").carrier
    pad = max(0, nb_pad - int(nb_full))
    pad2 = ((0, 0), (0, pad))
    ea_full = jnp.pad(energy_a, pad2)
    eb_full = jnp.pad(energy_b, pad2)
    fa_full = jnp.pad(occ_a, pad2)
    fb_full = jnp.pad(occ_b, pad2)
    sa_full = jnp.pad(surface_a, pad2)
    sb_full = jnp.pad(surface_b, pad2)
    z = jnp.asarray(z_values, dtype=jnp.complex128)
    ntiles = nb_pad // tile

    def _pair_contribution(pa_x, pb_x, pa_y, pb_y, ea, eb, fa, fb, sa, sb,
                           ga, gb):
        # At coincident energies, df/dE is minus the supplied surface weight.
        de = ea[:, :, None] - eb[:, None, :]
        df = fa[:, :, None] - fb[:, None, :]
        scale = jnp.maximum(
            1.0, jnp.maximum(jnp.abs(ea[:, :, None]), jnp.abs(eb[:, None, :])))
        separated = jnp.abs(de) > 64.0 * jnp.finfo(jnp.float64).eps * scale
        diagonal_limit = -0.5 * (sa[:, :, None] + sb[:, None, :])
        static_divided = jnp.where(
            separated, df / jnp.where(separated, de, 1.0), diagonal_limit)
        dynamic = df[None, :, :, :] / (de[None, :, :, :] + z[:, None, None, None])
        weights = jnp.where(
            (z == 0.0)[:, None, None, None], static_divided[None, :, :, :],
            dynamic)
        logical = (
            (ga[:, None] < int(nb_logical)) & (gb[None, :] < int(nb_logical))
        )[None, :, :]
        weights = jnp.where(logical[None, :, :, :], weights, 0.0)
        density_x = jnp.einsum(
            "ksma,ksmb->kmab", pa_x, jnp.conj(pb_x), optimize=True)
        density_y = jnp.einsum(
            "ksna,ksnb->knab", pa_y, jnp.conj(pb_y), optimize=True)
        return jnp.einsum(
            "zkab,kmab,knab->zmn", weights, density_x, jnp.conj(density_y),
            optimize=True)

    zero = jnp.zeros((z.size, nmu_x_loc, nmu_y_loc), dtype=jnp.complex128)

    def _outer(acc, ia_step):
        ia = ia_step * tile
        ga = ia + jnp.arange(tile)
        a_x = _children(_gather_mun(psi_mun_a, ia), unfold_x)
        a_y = _children(_gather_nmu(psi_nmu_a, ia), unfold_y)
        ea = jax.lax.dynamic_slice(ea_full, (0, ia), (nk, tile))
        fa = jax.lax.dynamic_slice(fa_full, (0, ia), (nk, tile))
        sa = jax.lax.dynamic_slice(sa_full, (0, ia), (nk, tile))

        def _inner(acc_inner, ib_step):
            ib = ib_step * tile
            gb = ib + jnp.arange(tile)
            b_x = _roll(_children(_gather_mun(psi_mun_b, ib), unfold_x))
            b_y = _roll(_children(_gather_nmu(psi_nmu_b, ib), unfold_y))
            eb = jax.lax.dynamic_slice(eb_full, (0, ib), (nk, tile))
            fb = jax.lax.dynamic_slice(fb_full, (0, ib), (nk, tile))
            sb = jax.lax.dynamic_slice(sb_full, (0, ib), (nk, tile))
            contribution = _pair_contribution(
                a_x, b_x, a_y, b_y, ea, eb, fa, fb, sa, sb, ga, gb)
            return acc_inner + contribution, None

        acc_inner, _ = jax.lax.scan(
            _inner, jnp.zeros_like(acc), jnp.arange(ntiles), unroll=1)
        return acc + acc_inner, None

    chi, _ = jax.lax.scan(_outer, zero, jnp.arange(ntiles), unroll=1)
    return chi / jnp.sqrt(jnp.asarray(nk, jnp.float64))


_PARENT_UNFOLD_OPERANDS = {}


def _parent_face_unfold_operands(plan, mesh_xy):
    """Place the eight typed parent-unfold tables from host data on their declared mesh axes."""
    key = (id(plan), id(mesh_xy))
    hit = _PARENT_UNFOLD_OPERANDS.get(key)
    if hit is not None:
        return hit
    t = plan.wavefunction_unfold_tables()
    from common.collectives import device_put_process_local
    rep = lambda spec: NamedSharding(mesh_xy, spec)
    ops = (
        device_put_process_local(np.asarray(t["irr_idx"], np.int32), rep(P(None))),
        device_put_process_local(np.asarray(t["sym_idx"], np.int32), rep(P(None))),
        device_put_process_local(np.asarray(t["k_irr_frac"], np.float64), rep(P(None, None))),
        device_put_process_local(np.asarray(t["spin_action_full"], np.complex128), rep(P(None, None, None))),
        device_put_process_local(np.asarray(t["local_perm"], np.int32), rep(P(None, "x"))),
        device_put_process_local(np.asarray(t["L_table"], np.float64), rep(P(None, "x", None))),
        device_put_process_local(np.asarray(t["local_perm"], np.int32), rep(P(None, "y"))),
        device_put_process_local(np.asarray(t["L_table"], np.float64), rep(P(None, "y", None))),
    )
    hit = (ops, int(t["n_sym_spatial"]))
    _PARENT_UNFOLD_OPERANDS[key] = hit
    return hit


_PARENT_UNFOLD_SPECS = (P(None), P(None), P(None, None), P(None, None, None),
                        P(None, "x"), P(None, "x", None),
                        P(None, "y"), P(None, "y", None))


_CHILD_FACE_KERNELS: dict = {}


def iter_parent_children_faces(carrier, mesh_xy, *, slices, by_parent=True):
    """Yield typed child faces in parent order, optionally as one full-k batch."""
    from types import SimpleNamespace
    from common.shard_map import shard_map
    from ffi import _services
    _services.ensure_on_path()
    from common.wfn_layout import psi_specs
    PSI_NMU_SPEC, PSI_MUN_SPEC = psi_specs(carrier.layout)

    plan = carrier.plan
    ops, n_sym_spatial = _parent_face_unfold_operands(plan, mesh_xy)
    irr, sym_rows, kfrac, U, perm_x, L_x, perm_y, L_y = ops
    irr_np = np.asarray(plan.irr_idx)
    host_tables = plan.wavefunction_unfold_tables()

    def _kernel(spec, spin_axis, mu_axis, perm_spec, L_spec, n_rows):
        key = (plan, id(mesh_xy), spec, spin_axis, mu_axis, int(n_rows))
        hit = _CHILD_FACE_KERNELS.get(key)
        if hit is None:
            def body(psi, irr_r, sym_r, kfrac_r, U_r, perm, L):
                tables = dict(irr_idx=irr_r, sym_idx=sym_r, k_irr_frac=kfrac_r,
                              spin_action_full=U_r, local_perm=perm, L_table=L,
                              n_sym_spatial=n_sym_spatial)
                return plan.unfold_face(
                    psi, spin_axis=spin_axis, mu_axis=mu_axis, tables=tables)
            hit = jax.jit(shard_map(
                body, mesh=mesh_xy,
                in_specs=(spec, P(None), P(None), P(None, None),
                          P(None, None, None), perm_spec, L_spec),
                out_specs=spec, check_vma=False))
            _CHILD_FACE_KERNELS[key] = hit
        return hit

    rep = lambda spec: NamedSharding(mesh_xy, spec)
    groups = [np.flatnonzero(irr_np == parent) for parent in range(int(plan.n_parent))]
    if not by_parent:
        groups = [np.concatenate(groups)]
    for rows in groups:
        if rows.size == 0:
            continue
        irr_r = device_put_process_local(
            np.asarray(host_tables["irr_idx"], np.int32)[rows], rep(P(None)))
        sym_r = device_put_process_local(
            np.asarray(host_tables["sym_idx"], np.int32)[rows], rep(P(None)))
        U_r = device_put_process_local(
            np.asarray(host_tables["spin_action_full"], np.complex128)[rows],
            rep(P(None, None, None)))
        mun = _kernel(PSI_MUN_SPEC, 1, 2, P(None, "x"), P(None, "x", None),
                      rows.size)(carrier.psi_mun, irr_r, sym_r, kfrac, U_r,
                                 perm_x, L_x)
        nmu = _kernel(PSI_NMU_SPEC, 2, 3, P(None, "y"), P(None, "y", None),
                      rows.size)(carrier.psi_nmu, irr_r, sym_r, kfrac, U_r,
                                 perm_y, L_y)
        yield rows, SimpleNamespace(
            layout=carrier.layout, psi_mun=mun, psi_nmu=nmu, slices=slices,
            enk=jnp.take(carrier.enk, irr_np[rows], axis=0),
            occ=jnp.take(carrier.occ, irr_np[rows], axis=0))


def _unfold_tables_from_operands(irr, sym, kfrac, U, perm_x, L_x, perm_y, L_y,
                                 n_sym_spatial):
    common = dict(irr_idx=irr, sym_idx=sym, k_irr_frac=kfrac,
                  spin_action_full=U, n_sym_spatial=n_sym_spatial)
    return (dict(common, local_perm=perm_x, L_table=L_x),
            dict(common, local_perm=perm_y, L_table=L_y))


def _get_chi_static_fractional_gamma_kernel_face(
    mesh_xy: Mesh, *, nb_full: int, nb_logical: int, pair_tile: int,
    k_unfold_plan=None, layout="face",
):
    """Build the static Gamma divided-difference kernel on canonical faces or parents."""
    from common.shard_map import shard_map
    from common.wfn_layout import psi_specs
    PSI_NMU_SPEC, PSI_MUN_SPEC = psi_specs(layout)

    tile = int(pair_tile)
    key = ("static_fractional_gamma_face", id(mesh_xy), int(nb_full),
           int(nb_logical), tile, id(k_unfold_plan), layout)
    hit = _chi_minimax_kernel_cache.get(key)
    if hit is not None:
        return hit

    if k_unfold_plan is None:
        def _local(psi_mun, psi_nmu, energies, occupations, surface_weight):
            return _fractional_pair_scan_face(
                psi_mun, psi_nmu, psi_mun, psi_nmu, energies, energies,
                occupations, occupations, surface_weight, surface_weight,
                jnp.zeros((1,), dtype=jnp.complex128),
                nb_full=nb_full, nb_logical=nb_logical, tile=tile)
        in_specs = (PSI_MUN_SPEC, PSI_NMU_SPEC, P(None, None), P(None, None),
                    P(None, None))
    else:
        # Raw parents: the packed parent faces plus the unfold tables; each
        # band tile is unfolded to full k inside the scan.  The result is in
        # PACKED centroid order on both axes; the caller restores it.
        n_sym_spatial = int(k_unfold_plan.n_sym_spatial)

        def _local(psi_mun, psi_nmu, energies, occupations, surface_weight,
                   *tables):
            unfold_x, unfold_y = _unfold_tables_from_operands(
                *tables, n_sym_spatial=n_sym_spatial)
            return _fractional_pair_scan_face(
                psi_mun, psi_nmu, psi_mun, psi_nmu, energies, energies,
                occupations, occupations, surface_weight, surface_weight,
                jnp.zeros((1,), dtype=jnp.complex128),
                nb_full=nb_full, nb_logical=nb_logical, tile=tile,
                unfold_x=unfold_x, unfold_y=unfold_y, k_unfold_plan=k_unfold_plan)
        in_specs = (PSI_MUN_SPEC, PSI_NMU_SPEC, P(None, None), P(None, None),
                    P(None, None)) + _PARENT_UNFOLD_SPECS

    kernel = jax.jit(shard_map(
        _local,
        mesh=mesh_xy,
        in_specs=in_specs,
        out_specs=P(None, "x", "y"),
        check_vma=False,
    ))
    _chi_minimax_kernel_cache[key] = kernel
    return kernel


def _get_chi_fractional_q_kernel_face(
    mesh_xy: Mesh, *, nb_full: int, nb_logical: int, pair_tile: int,
    n_z: int, k_unfold_plan=None, layout="face",
):
    """Roll the unfolded b endpoint to k−q inside the ordered-pair contraction."""
    from common.shard_map import shard_map
    from common.wfn_layout import psi_specs
    PSI_NMU_SPEC, PSI_MUN_SPEC = psi_specs(layout)

    tile = int(pair_tile)
    key = ("direct_fractional_q_face", id(mesh_xy), int(nb_full),
           int(nb_logical), tile, int(n_z), id(k_unfold_plan), layout)
    hit = _chi_minimax_kernel_cache.get(key)
    if hit is not None:
        return hit

    if k_unfold_plan is None:
        def _local(psi_mun, psi_nmu, kminq_idx, energies, occupations,
                   surface_weight, z_values):
            psi_mun_b = jnp.take(psi_mun, kminq_idx, axis=0)
            psi_nmu_b = jnp.take(psi_nmu, kminq_idx, axis=0)
            eb = jnp.take(energies, kminq_idx, axis=0)
            fb = jnp.take(occupations, kminq_idx, axis=0)
            sb = jnp.take(surface_weight, kminq_idx, axis=0)
            return _fractional_pair_scan_face(
                psi_mun, psi_nmu, psi_mun_b, psi_nmu_b, energies, eb,
                occupations, fb, surface_weight, sb, z_values,
                nb_full=nb_full, nb_logical=nb_logical, tile=tile)
        in_specs = (PSI_MUN_SPEC, PSI_NMU_SPEC, P(None), P(None, None),
                    P(None, None), P(None, None), P(None))
    else:
        # Raw parents: the k−q roll acts on FULL-k rows, so it is applied
        # to each unfolded band tile inside the scan (``roll_b``), never to
        # the parent operand.
        n_sym_spatial = int(k_unfold_plan.n_sym_spatial)

        def _local(psi_mun, psi_nmu, kminq_idx, energies, occupations,
                   surface_weight, z_values, *tables):
            unfold_x, unfold_y = _unfold_tables_from_operands(
                *tables, n_sym_spatial=n_sym_spatial)
            eb = jnp.take(energies, kminq_idx, axis=0)
            fb = jnp.take(occupations, kminq_idx, axis=0)
            sb = jnp.take(surface_weight, kminq_idx, axis=0)
            return _fractional_pair_scan_face(
                psi_mun, psi_nmu, psi_mun, psi_nmu, energies, eb,
                occupations, fb, surface_weight, sb, z_values,
                nb_full=nb_full, nb_logical=nb_logical, tile=tile,
                unfold_x=unfold_x, unfold_y=unfold_y, roll_b=kminq_idx,
                k_unfold_plan=k_unfold_plan)
        in_specs = (PSI_MUN_SPEC, PSI_NMU_SPEC, P(None), P(None, None),
                    P(None, None), P(None, None), P(None)) + _PARENT_UNFOLD_SPECS

    kernel = jax.jit(shard_map(
        _local,
        mesh=mesh_xy,
        in_specs=in_specs,
        out_specs=P(None, "x", "y"),
        check_vma=False,
    ))
    _chi_minimax_kernel_cache[key] = kernel
    return kernel


def compute_chi0_static_fractional_gamma(
    wfns,
    energies_kn_ry,
    occupations_kn,
    surface_weight_kn,
    meta,
    mesh_xy,
    *,
    nb_logical: int,
):
    r"""Return the exact static fractional-occupation chi0 at Gamma.

    The ordered-pair kernel evaluates

    ``(f_ka-f_kb)/(E_ka-E_kb)``

    and uses ``df/dE`` on the degenerate diagonal.  The supplied surface
    table owns that diagonal limit; the QSGW metal path supplies periodic
    tetrahedron weights, while off-diagonal pairs retain the carried MP1
    occupations.  The returned ``(1,n_mu,n_mu)`` array has the historical
    raw-chi normalization expected by :func:`solve_w`.

    This direct tiled implementation is the exact finite-band fallback.  A
    future certified separable divided-difference minimax target can replace
    its internals without changing this API or the Dyson/head callers.
    """
    e = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    f = jnp.asarray(occupations_kn, dtype=jnp.float64)
    surface = jnp.asarray(surface_weight_kn, dtype=jnp.float64)
    if e.ndim != 2 or f.shape != e.shape or surface.shape != e.shape:
        raise ValueError(
            "static fractional chi requires matching (nk,nb) energies, "
            f"occupations, and surface weights; got {e.shape}, {f.shape}, "
            f"{surface.shape}")
    if int(e.shape[0]) != int(meta.nk_tot):
        raise ValueError(
            f"static fractional chi has nk={e.shape[0]}, expected "
            f"meta.nk_tot={meta.nk_tot}")
    if not (0 < int(nb_logical) <= int(e.shape[1])):
        raise ValueError(
            f"static fractional chi needs 0 < nb_logical <= {e.shape[1]}; "
            f"got {nb_logical}")
    # face: the persistent carrier spans the FULL [b0,b4) window and
    # cannot be band-sliced (obstacle #3) -- pad the caller's
    # energies/occupations/surface table up to nb_full instead (any
    # padded position is >= nb_logical, so the pair kernel's own
    # nb_logical mask excludes it regardless).
    nb_full = int(wfns.slices.nb_full)
    if nb_full < int(e.shape[1]):
        raise ValueError(
            "compute_chi0_static_fractional_gamma(layout='face'): the "
            f"loaded face carrier covers only {nb_full} bands, fewer than "
            f"the {e.shape[1]} the caller's tables provide")
    bpad = nb_full - int(e.shape[1])
    pad2 = ((0, 0), (0, bpad))
    e_full = jnp.pad(e, pad2)
    f_full = jnp.pad(f, pad2)
    surface_full = jnp.pad(surface, pad2)
    carrier = wfns.green_parent
    if carrier is not None:
        # Raw parents: packed parent faces + unfold tables in, χ out in the
        # run's packed order like every other operator.
        plan = carrier.plan
        tables, _ = _parent_face_unfold_operands(plan, mesh_xy)
        return _get_chi_static_fractional_gamma_kernel_face(
            mesh_xy,
            nb_full=nb_full,
            nb_logical=int(nb_logical),
            pair_tile=_STATIC_FRACTIONAL_PAIR_TILE,
            k_unfold_plan=plan, layout=wfns.layout,
        )(carrier.psi_mun, carrier.psi_nmu, e_full, f_full, surface_full,
          *tables)
    return _get_chi_static_fractional_gamma_kernel_face(
        mesh_xy,
        nb_full=nb_full,
        nb_logical=int(nb_logical),
        pair_tile=_STATIC_FRACTIONAL_PAIR_TILE, layout=wfns.layout,
    )(wfns.psi_mun, wfns.psi_nmu, e_full, f_full, surface_full)


def occupation_support_bandwidth(
        energies_kn_ry, occupations_kn,
        occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT):
    """Largest transition energy over the occupation supports, Ry.

    ``max(E over the (1-f) support) − min(E over the f support)`` over the
    SAME two slices :func:`_occupation_support_slices` hands the χ₀ kernel,
    so the rule bandwidth and the bands it must resolve can never disagree —
    which is why the threshold is an argument here rather than a second
    default.  An MP1 overshoot band at a support edge is included, by
    magnitude.  This — not ``quad.x_max`` — sizes the damped-line rule
    bandwidth on metal plans, where the occupied and empty supports overlap.
    """
    e = np.asarray(jax.device_get(energies_kn_ry), dtype=np.float64)
    f_slice, u_slice = _occupation_support_slices(
        occupations_kn, occupation_window_threshold)
    return float(np.max(e[:, u_slice]) - np.min(e[:, f_slice]))


def compute_chi0_static_fractional(
    wfns,
    meta,
    mesh_xy,
    *,
    occupation_state,
    kminq_rows,
    nb_logical=None,
):
    """Exact static finite-occupation chi0 for every stored q row.

    The finite-q generalization of
    :func:`compute_chi0_static_fractional_gamma`: for wedge row j the b
    side of every ordered pair rides at ``k − q_j`` through the caller's
    precomputed flat map ``kminq_rows[j]`` (``common.kq_mapping``), and
    the divided difference ``(f_a(k)−f_b(k−q))/(E_a(k)−E_b(k−q))`` uses
    the analytic MP1 ``−df/dE`` midpoint limit on accidentally degenerate
    pairs.  This is the literal static member of the shared ordered-pair
    evaluator; the metal MPA shifted-origin slot instead calls
    :func:`compute_chi0_direct_fractional` at its stamped nonzero ``z``.
    Returns ``(n_q, n_mu, n_mu)``
    wedge rows in the raw-chi normalization expected by :func:`solve_w`,
    sharded ``P(None, 'x', 'y')``.
    """
    return compute_chi0_direct_fractional(
        wfns, np.asarray([0.0j], dtype=np.complex128), meta, mesh_xy,
        occupation_state=occupation_state, kminq_rows=kminq_rows,
        nb_logical=nb_logical)


def compute_chi0_direct_fractional(
    wfns,
    z_values,
    meta,
    mesh_xy,
    *,
    occupation_state,
    kminq_rows,
    nb_logical=None,
    progress_fn=None,
):
    """Exact finite-occupation chi0 at selected complex frequencies.

    This is the ordered-pair escape hatch for isolated points at which the
    damped-contour evaluator is unaffordable.  It shares the static kernel's
    band-pair scan and distributed centroid output.  A zero entry uses the
    MP1 divided-difference limit; every nonzero entry is evaluated at its
    literal complex coordinate.  With one frequency the returned shape is
    ``(n_q,n_mu,n_mu)``; otherwise it is ``(n_z,n_q,n_mu,n_mu)``.
    ``progress_fn``, when supplied, is called as
    ``progress_fn(rows_done, rows_total, elapsed_seconds)`` after each q-row
    result is device-ready.  It changes synchronization only, never values.
    """
    from gw.efermi import mp1_negative_derivative

    family = getattr(occupation_state, "smearing_family", None)
    if family != "mp1":
        raise ValueError(
            "GATE static_fractional_needs_mp1: direct fractional chi0 "
            "received an unsupported smearing family.\n"
            f"  got:  occupation_state.smearing_family = {family!r}\n"
            "  want: occupation_state.smearing_family = 'mp1'\n"
            "  why:  this path's intraband diagonal is the analytic MP1 "
            "-df/dE; a step occupation belongs to the insulating chi0 path\n"
            "  doc:  docs/theory/metallic-mpa-screening.md")
    e = jnp.asarray(wfns.enk, dtype=jnp.float64)
    f = jnp.asarray(occupation_state.f_kn, dtype=jnp.float64)
    if f.shape != e.shape:
        raise ValueError(
            f"static fractional chi occupations {f.shape} do not match "
            f"energies {e.shape}")
    if int(e.shape[0]) != int(meta.nk_tot):
        raise ValueError(
            f"static fractional chi has nk={e.shape[0]}, expected "
            f"meta.nk_tot={meta.nk_tot}")
    nb = int(e.shape[1])
    nb_log = nb if nb_logical is None else int(nb_logical)
    if not (0 < nb_log <= nb):
        raise ValueError(
            f"static fractional chi needs 0 < nb_logical <= {nb}; got "
            f"{nb_logical}")
    kmq = np.asarray(kminq_rows, dtype=np.int32)
    if kmq.ndim != 2 or kmq.shape[1] != int(e.shape[0]):
        raise ValueError(
            "static fractional chi kminq_rows must have shape (n_q, nk="
            f"{e.shape[0]}); got {kmq.shape}")
    z = np.asarray(z_values, dtype=np.complex128)
    if z.ndim != 1 or not z.size or not np.all(np.isfinite(z)):
        raise ValueError(
            "direct fractional chi z_values must be a finite nonempty vector")
    surface = mp1_negative_derivative(
        e, float(occupation_state.mu_ry),
        float(occupation_state.smearing_width_ry))
    # face: wfns.enk is already (nk, nb_full) -- e/f/surface above are
    # ALREADY at the full loaded extent for this call site (they are
    # wfns.enk/occupation_state.f_kn/its own derivative, not a caller-
    # narrowed sub-window), but pad defensively for the general case
    # rather than assume it (mirrors the Gamma wrapper's own guard).
    nb_full = int(wfns.slices.nb_full)
    if nb_full < nb:
        raise ValueError(
            "compute_chi0_direct_fractional(layout='face'): the loaded "
            f"face carrier covers only {nb_full} bands, fewer than the "
            f"{nb} the caller's energies/occupations table provides")
    bpad = nb_full - nb
    pad2 = ((0, 0), (0, bpad))
    e_full = jnp.pad(e, pad2)
    f_full = jnp.pad(f, pad2)
    surface_full = jnp.pad(surface, pad2)
    carrier = wfns.green_parent
    plan = None if carrier is None else carrier.plan
    psi_mun_in, psi_nmu_in = (
        (wfns.psi_mun, wfns.psi_nmu) if carrier is None
        else (carrier.psi_mun, carrier.psi_nmu))
    tables = () if plan is None else _parent_face_unfold_operands(plan, mesh_xy)[0]
    kernel = _get_chi_fractional_q_kernel_face(
        mesh_xy, nb_full=nb_full, nb_logical=nb_log,
        pair_tile=_STATIC_FRACTIONAL_PAIR_TILE, n_z=z.size,
        k_unfold_plan=plan, layout=wfns.layout)
    rows = []
    for q_row, row in enumerate(kmq):
        started = time.monotonic()
        value = kernel(
            psi_mun_in, psi_nmu_in, jnp.asarray(row), e_full, f_full,
            surface_full, jnp.asarray(z), *tables)
        if progress_fn is not None:
            value.block_until_ready()
            progress_fn(q_row + 1, len(kmq), time.monotonic() - started)
        rows.append(value)
    values = jnp.stack(rows, axis=1)
    return values[0] if z.size == 1 else values


def precompile_chi0_contour_fractional(
    wfns,
    time_nodes,
    weight_rows,
    z_values,
    meta,
    mesh_xy,
    *,
    occupations=None,
    energy_reference=0.0,
    occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
):
    """AOT sibling of compute_chi0_contour_fractional."""
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    args, n_out = _chi0_fractional_contour_args(
        wfns,
        time_nodes,
        weight_rows,
        z_values,
        occupations,
        energy_reference,
        occupation_window_threshold,
    )
    _get_chi_fractional_contour_kernel(
        mesh_xy, kgrid, n_out, **_chi_face_kwargs(wfns)).lower(*args).compile()


def precompile_chi0(wfns, quad, meta, mesh_xy, *, energy_reference=None):
    """AOT lower+compile of the χ₀ minimax kernel at the real input
    shapes/shardings — warms the JAX in-process cache so the first
    ``compute_chi0`` call is execution-only.  Call inside a dedicated
    ``timing.section('chi0_W.chi.compile')`` block to separate compile
    from exec in the end-of-run timing report.
    """
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    eref = 0.0 if energy_reference is None else float(energy_reference)
    s = wfns.slices
    enk_v = wfns.enk[:, s.val]
    enk_c = wfns.enk[:, s.cond]
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64) - eref
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64) - eref
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    E_gap = cmin - vmax
    tau = np.asarray(quad.tau, dtype=np.float64)
    if len(tau) == 0:
        return  # compute_chi0 falls through to a static-zeros path — nothing to compile
    alpha_chi = -1.0 * np.asarray(quad.alpha, dtype=np.float64) * np.exp(-tau * E_gap)
    nodes = MinimaxNodes(
        t=jnp.asarray(tau, dtype=jnp.complex128),
        alpha=jnp.asarray(alpha_chi, dtype=jnp.complex128),
    )

    kernel = _get_chi_minimax_kernel(
        mesh_xy, kgrid, **_chi_parent_face_kwargs(wfns))
    kernel.lower(
        nodes, *_chi_layout_operands(wfns, eref),
        jnp.asarray(vmax, dtype=jnp.float64),
        jnp.asarray(cmin, dtype=jnp.float64),
    ).compile()


def precompile_solve_w(V_q, chi0_q, meta, mesh_xy, *, dyson_solver=None,
                       n_rmu_logical=None,
                       distrib_la_batched_route: str = "batch_reshard"):
    """AOT lower+compile of the W-solve jit.  See ``precompile_chi0``.

    Goes through the same ``_resolve_w_solve_fn`` dispatch as
    :func:`solve_w` so both paths agree on which jit to compile.
    """
    ensure_jax_compile_cache()
    n_logical = _require_w_operand_geometry(
        V_q, chi0_q, meta, mesh_xy, n_rmu_logical=n_rmu_logical)
    solve_fn, pref = _resolve_w_solve_fn(
        meta, mesh_xy, n_rmu=chi0_q.shape[1],
        n_rmu_logical=n_logical, dyson_solver=dyson_solver,
        distrib_la_batched_route=distrib_la_batched_route)
    # The DISTRIBUTED plan is a plain function around chunked jits + one
    # FFI call, not a single jit, so there is nothing to lower here —
    # the first real call builds the BLACS descriptor and compiles its
    # own modules (scorecard L §5, amortised from call 2; the ζ tier
    # behaves the same way).
    if not hasattr(solve_fn, "lower"):
        return
    solve_fn.lower(V_q, chi0_q, pref).compile()
