"""Stream Lorentz blocks through parent-face static Σ contractions."""

from __future__ import annotations

from ffi import _services
_services.ensure_on_path()
from distrib_la import mesh_key as _mesh_key
from dataclasses import dataclass

from common.collectives import device_put_process_local

import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P



_TERM_X = 0
_TERM_SX = 1
_TERM_COH = 2
_HEAD_CC = 0
_HEAD_CTTC = 1
_HEAD_TT = 2
#: ``blocks`` selections of :func:`compute_static_photon_sigma`.  ``all`` is
#: the sixteen-block static COHSEX Sigma; ``current`` is the fifteen non-CC
#: blocks, used by the dynamic packed route whose CC block is owned by the
#: scalar Sigma_c machinery at every frequency.
PHOTON_BLOCKS_ALL = "all"
PHOTON_BLOCKS_CURRENT = "current"
_PHOTON_BLOCK_SELECTIONS = (PHOTON_BLOCKS_ALL, PHOTON_BLOCKS_CURRENT)
_photon_sigma_kernel_cache: dict[tuple[object, ...], object] = {}


def photon_head_alpha_operator(geom):
    r"""Return the production spatial ``gamma-tilde`` vertex sweep operator.

    The input of :func:`common.mtxel_sweep.sweep_matrix_elements` must be the
    four-component WFN produced by the run's kinetic-balance loader.  This
    operator applies the same :func:`common.gamma_matrices.gamma_apply`
    matrices as :func:`_make_lorentz_convolution`, yielding
    ``<m k|alpha_A|n k>`` for ``A=1,2,3``.  It intentionally excludes the
    ICL pseudopotential term in the uniform-gauge response ``Gamma_raw``:
    that response vertex is a different operator from the photon Sigma
    vertex used by the existing production convolution.
    """
    from common.gamma_matrices import gamma_apply, gamma_perm_phase
    from common.mtxel_sweep import Operator

    if int(geom.ns) != 4:
        raise ValueError(
            "photon head alpha vertices require kinetic-balance "
            f"four-spinors; geom.ns={int(geom.ns)}")
    vertices = tuple(gamma_perm_phase(A) for A in (1, 2, 3))

    def apply(psi_n, gvec, gmask, bidx, kvec):
        del gvec, bidx, kvec
        psi = psi_n * gmask[None, None, None, :].astype(psi_n.dtype)
        applied = jnp.stack(
            tuple(gamma_apply(psi, perm, phase, axis=2)
                  for perm, phase in vertices), axis=-1)
        return applied

    return Operator(
        apply=apply, post=1.0, ncomp=3,
        key=("photon_head_alpha", int(geom.ngkmax), int(geom.ns)))


def contract_ordered_tt_head_sigma(
    gamma_ext_m,
    f_plus_ext_m,
    f_minus_ext_m,
    occupations_km,
    *,
    nk_tot: int,
    cell_volume_bohr3: float,
    intermediate_mask_km=None,
):
    r"""Contract an ordered, cell-averaged transverse head into diagonal Sigma.

    This is the small final consumer for a full-frequency ``q=0`` photon
    head.  The inputs use the production photon-current convention

    ``gamma_ext_m[k,A,n,m] = <n k | Gamma_A | m k>``

    with ``A=1,2,3``.  These are the production ``gamma-tilde`` vertices
    projected through the run's kinetic-balance four-spinor WFN carrier;
    they are not the pseudopotential uniform-gauge ``Gamma_raw`` operator.
    ``f_plus_ext_m`` and ``f_minus_ext_m`` are the two ordered contour
    functions evaluated at ``epsilon_n-epsilon_m+i*eta``.  Their shape is
    ``(nk,n_external,n_intermediate,3,3)`` and their raw cell-moment unit is
    ``Ry*bohr^3``.

    The Green pole chooses ``F_-`` with weight ``f_m`` and ``F_+`` with
    weight ``1-f_m``.  The returned array has shape ``(nk,n_external)``.
    The completed head matrices are raw cell moments.  This consumer applies
    ``+1/(cell_volume_bohr3*nk_tot)`` exactly once.  The positive correlation
    sign follows the ordered ``iGW`` contour convention; it must not be
    copied from the static convolution's occupied-G exchange sign.

    ``F_+`` and ``F_-`` must be formed from the decaying retarded remainder
    ``H(z)=W(z)-W(infinity)``.  Any nonzero instantaneous ``W(infinity)``
    belongs in :func:`contract_instantaneous_tt_head_exchange`, rather than
    in this ordered contour contraction.
    """
    gamma = jnp.asarray(gamma_ext_m, dtype=jnp.complex128)
    f_plus = jnp.asarray(f_plus_ext_m, dtype=jnp.complex128)
    f_minus = jnp.asarray(f_minus_ext_m, dtype=jnp.complex128)
    occupations = jnp.asarray(occupations_km, dtype=jnp.float64)
    if gamma.ndim != 4 or int(gamma.shape[1]) != 3:
        raise ValueError(
            "gamma_ext_m must have shape (nk,3,n_external,n_intermediate); "
            f"got {tuple(gamma.shape)}")
    expected_f = (gamma.shape[0], gamma.shape[2], gamma.shape[3], 3, 3)
    if tuple(f_plus.shape) != expected_f or tuple(f_minus.shape) != expected_f:
        raise ValueError(
            "ordered TT functions must both have shape "
            f"{expected_f}; got {tuple(f_plus.shape)} and "
            f"{tuple(f_minus.shape)}")
    expected_occ = (gamma.shape[0], gamma.shape[3])
    if tuple(occupations.shape) != expected_occ:
        raise ValueError(
            f"occupations_km must have shape {expected_occ}; got "
            f"{tuple(occupations.shape)}")
    if int(nk_tot) <= 0:
        raise ValueError(f"nk_tot must be positive; got {nk_tot}")
    if not np.isfinite(float(cell_volume_bohr3)) or float(cell_volume_bohr3) <= 0:
        raise ValueError(
            "cell_volume_bohr3 must be finite and positive; got "
            f"{cell_volume_bohr3}")
    if intermediate_mask_km is None:
        mask = jnp.ones(expected_occ, dtype=jnp.float64)
    else:
        mask = jnp.asarray(intermediate_mask_km, dtype=jnp.float64)
        if tuple(mask.shape) != expected_occ:
            raise ValueError(
                f"intermediate_mask_km must have shape {expected_occ}; got "
                f"{tuple(mask.shape)}")

    ordered = (
        occupations[:, None, :, None, None] * f_minus
        + (jnp.float64(1.0) - occupations[:, None, :, None, None]) * f_plus
    )
    per_intermediate = jnp.einsum(
        "kanm,knmab,kbnm->knm", gamma, ordered, jnp.conj(gamma),
        optimize=True)
    return jnp.sum(per_intermediate * mask[:, None, :], axis=-1) / (
        float(cell_volume_bohr3) * float(nk_tot))


def contract_instantaneous_tt_head_exchange(
    gamma_ext_m,
    interaction_tt,
    occupations_km,
    *,
    nk_tot: int,
    cell_volume_bohr3: float,
    intermediate_mask_km=None,
):
    r"""Contract an instantaneous cell-averaged TT head with occupied states.

    ``interaction_tt`` is either one ``(3,3)`` matrix or a ``(nk,3,3)``
    raw cell moment in ``Ry*bohr^3``.  This routine is used independently for the bare TT
    exchange and for a contact-induced ``W(infinity)-D`` change, keeping the
    constant term out of the finite-pole contour.
    """
    gamma = jnp.asarray(gamma_ext_m, dtype=jnp.complex128)
    occupations = jnp.asarray(occupations_km, dtype=jnp.float64)
    interaction = jnp.asarray(interaction_tt, dtype=jnp.complex128)
    if gamma.ndim != 4 or int(gamma.shape[1]) != 3:
        raise ValueError(
            "gamma_ext_m must have shape (nk,3,n_external,n_intermediate); "
            f"got {tuple(gamma.shape)}")
    expected_occ = (gamma.shape[0], gamma.shape[3])
    if tuple(occupations.shape) != expected_occ:
        raise ValueError(
            f"occupations_km must have shape {expected_occ}; got "
            f"{tuple(occupations.shape)}")
    if interaction.ndim == 2:
        if tuple(interaction.shape) != (3, 3):
            raise ValueError(
                "interaction_tt must have shape (3,3) or (nk,3,3); "
                f"got {tuple(interaction.shape)}")
        interaction = jnp.broadcast_to(
            interaction[None, :, :], (gamma.shape[0], 3, 3))
    elif tuple(interaction.shape) != (gamma.shape[0], 3, 3):
        raise ValueError(
            "interaction_tt must have shape (3,3) or "
            f"({gamma.shape[0]},3,3); got {tuple(interaction.shape)}")
    if int(nk_tot) <= 0:
        raise ValueError(f"nk_tot must be positive; got {nk_tot}")
    if not np.isfinite(float(cell_volume_bohr3)) or float(cell_volume_bohr3) <= 0:
        raise ValueError(
            "cell_volume_bohr3 must be finite and positive; got "
            f"{cell_volume_bohr3}")
    if intermediate_mask_km is None:
        mask = jnp.ones(expected_occ, dtype=jnp.float64)
    else:
        mask = jnp.asarray(intermediate_mask_km, dtype=jnp.float64)
        if tuple(mask.shape) != expected_occ:
            raise ValueError(
                f"intermediate_mask_km must have shape {expected_occ}; got "
                f"{tuple(mask.shape)}")
    per_intermediate = jnp.einsum(
        "kanm,kab,kbnm->knm", gamma, interaction, jnp.conj(gamma),
        optimize=True)
    weights = occupations * mask
    return -jnp.sum(per_intermediate * weights[:, None, :], axis=-1) / (
        float(cell_volume_bohr3) * float(nk_tot))


@dataclass(frozen=True)
class StaticPhotonHeadSigmaDiagnostics:
    """Exact diagonal contraction of the final q=0 Lorentz-block updates.

    ``components_tskn_ry`` has term order ``(X,SX,COH)`` and sector order
    ``(CC,CT+TC,TT)``.  The sectors classify FINAL completed V/W blocks;
    they are not nonlinear counterfactual Dyson solves.
    """

    components_tskn_ry: jax.Array
    max_closure_residual_ry: float
    output_basis: str

    def __post_init__(self) -> None:
        shape = tuple(self.components_tskn_ry.shape)
        if len(shape) != 4 or shape[:2] != (3, 3):
            raise ValueError(
                "static photon head diagnostics must be (3 terms,3 sectors,"
                f"nk,nb); got {shape}")
        if self.output_basis != "dft":
            raise ValueError(
                "static photon head Sigma diagnostics must be stamped in "
                f"the DFT output basis; got {self.output_basis!r}")


@dataclass(frozen=True)
class StaticPhotonSigmaDiagnostics:
    """Lorentz-sector split of the physical static self-energy.

    ``components_skij_ry`` has sector order ``(CC, CT+TC, TT)`` and contains
    ``Sigma_SX + Sigma_COH`` for the selected blocks.  Unlike the head-only
    diagnostic above, these are band-space operators so the self-consistent
    driver can rotate them with the total Sigma before taking diagonals.
    """

    components_skij_ry: jax.Array
    max_closure_residual_ry: float

    def __post_init__(self) -> None:
        shape = tuple(self.components_skij_ry.shape)
        if len(shape) != 4 or shape[0] != 3 or shape[-2] != shape[-1]:
            raise ValueError(
                "static photon Sigma diagnostics must be "
                f"(3 sectors,nk,nb,nb); got {shape}")


def _head_sector(A: int, B: int) -> int:
    if int(A) == 0 and int(B) == 0:
        return _HEAD_CC
    if (int(A) == 0) != (int(B) == 0):
        return _HEAD_CTTC
    return _HEAD_TT


def _diagnostic_diagonal(matrix, basis_rotation, mesh_xy):
    """Exact output-basis diagonal for ``(nk,...,nb,nb)`` batches."""
    if basis_rotation is None:
        diag = jnp.diagonal(matrix, axis1=-2, axis2=-1)
    else:
        from .qsgw_density import diagonal_rotated_band_matrix
        diag = diagonal_rotated_band_matrix(
            matrix, basis_rotation, mesh=mesh_xy, to_qp=False)
    # Diagnostics are O(nk*nb), so their public carrier is deliberately
    # replicated.  No dense rotated nb^2 matrix crosses this boundary.
    return jax.lax.with_sharding_constraint(
        diag, NamedSharding(mesh_xy, P(*([None] * diag.ndim))))


def _require_packed_operator(name, packed, mesh_xy):
    expected = NamedSharding(mesh_xy, P(None, "x", "y"))
    have = packed.sharding
    if (getattr(have, "mesh", None) != expected.mesh
            or getattr(have, "spec", None) != expected.spec):
        raise ValueError(
            f"photon operator {name} must remain P(None,'x','y'); got "
            f"{packed.sharding}.  A photon body may not be gathered or "
            "placed on fewer than all ranks.")


def _make_photon_static_class_kernel(
    mesh_xy, kgrid, nk_tot, wfns_left, wfns_right, *, with_head=False,
):
    """Share unfolded endpoints, G, transforms and projection across one Lorentz class."""
    from common.shard_map import shard_map
    from ffi import ffi_dial_key
    from common.contract_bands import contract_bands_block_reshard
    from distrib_la import gemm_plan
    from .cohsex_sigma import _make_static_convolution
    from .greens_function_kernel import build_G
    left, right = wfns_left.green_parent, wfns_right.green_parent
    layout = left.layout
    plans = (left.plan, right.plan)
    shapes = tuple((p.n_parent, c.psi_nmu.shape[1], p.n_centroid_packed, p.nspinor)
                   for c, p in zip((left, right), plans))
    key = (_mesh_key(mesh_xy), tuple(kgrid), tuple(map(id, plans)), shapes, layout, ffi_dial_key(), with_head)
    if key in _photon_sigma_kernel_cache:
        return _photon_sigma_kernel_cache[key]
    plan_key = ("plans", _mesh_key(mesh_xy), tuple(kgrid), nk_tot, shapes, layout, ffi_dial_key())
    if plan_key not in _photon_sigma_kernel_cache:
        project = contract_bands_block_reshard(mesh_xy, layout=layout,
            face_shape=shapes[0], right_face_shape=shapes[1])
        g_plan = gemm_plan(mesh_xy, m=shapes[0][2]*shapes[0][3], k=shapes[0][1],
            n=shapes[1][2]*shapes[1][3], nq=shapes[0][0], dtype=jnp.complex128, layout=layout)
        _photon_sigma_kernel_cache[plan_key] = project, g_plan
    project, g_plan = _photon_sigma_kernel_cache[plan_key]
    convolve = _make_static_convolution(mesh_xy, kgrid, nk_tot, q0_only=False, lorentz=True)
    head_convolve = (_make_static_convolution(mesh_xy, kgrid, nk_tot, q0_only=True, lorentz=True)
                     if with_head else None)
    rows = np.asarray(plans[0].parent_full_rows)
    @jax.jit
    def contract_class(left, right, weights, interaction, factor, vertices, head_interaction=None):
        weights = plans[0].parent_rows(weights)
        G = build_G(left.psi_mun, right.psi_nmu, phases=jnp.real(weights),
                    layout=layout, gemm=g_plan, k_unfold_plan=plans[0],
                    right_k_unfold_plan=plans[1])
        sigma = convolve(G, interaction, factor, vertices)
        result = project(left.projection_faces()[0], jnp.take(sigma, jnp.asarray(rows), axis=0), right.projection_faces()[1])
        if with_head:
            head_sigma = head_convolve(G, head_interaction, factor, vertices)
            head = project(left.projection_faces()[0], jnp.take(head_sigma, jnp.asarray(rows), axis=0), right.projection_faces()[1])
            return result, head
        return result
    _photon_sigma_kernel_cache[key] = contract_class
    return contract_class


def _photon_head_pairs(response, term, mesh_xy):
    """Prepare covariant Gamma factors once per term using their canonical orbit owner."""
    from .head_correction import _photon_q0_factor_orbit
    factors = response.head_completion.q0_factors
    pairs = (factors.bare_pair,) if term == _TERM_X else factors.screened_pairs
    bare = (factors.bare_pair,) if term == _TERM_COH else ()
    def expand(pairs):
        return tuple((device_put_process_local(images[0][i], NamedSharding(mesh_xy, P(None, 'x'))),
                      device_put_process_local(images[1][i], NamedSharding(mesh_xy, P(None, 'y'))))
            for pair in pairs for images in (_photon_q0_factor_orbit(
                *pair, layout=response.layout, plans=factors.family_plans, mesh_xy=mesh_xy),)
            for i in range(images[0].shape[0]))
    return expand(pairs), expand(bare)


def _make_photon_class_restore(response, keys):
    """Compile one canonical full-q producer per class without caching interaction arrays."""
    from .w_isdf import photon_blocks_full_q
    from common.gamma_matrices import gamma_perm_phase
    layout, plans, policy = response.layout, response.family_plans, response.qgrid_policy
    key = ("restore", id(layout), tuple(map(id, plans)), id(policy), keys)
    if key not in _photon_sigma_kernel_cache:
        @jax.jit
        def restore(packed):
            interactions = jnp.stack([value for _, value in photon_blocks_full_q(
                packed, keys, layout=layout, family_plans=plans, qgrid_policy=policy)])
            vertices = jax.tree.map(lambda *v: jnp.stack(v),
                *((gamma_perm_phase(A), gamma_perm_phase(B)) for A, B in keys))
            return interactions, vertices
        _photon_sigma_kernel_cache[key] = restore
    return _photon_sigma_kernel_cache[key]


def contract_lorentz_blocks(blocks, *, families, term, response, Gij, meta, mesh_xy,
                            head_diagnostics=False, admit_kernel=None):
    """Yield one parent-band sum per endpoint class while retaining one resident Green."""
    from .cohsex_sigma import _occ_diag_full
    from .photon_layout import photon_q0_low_rank_block
    if tuple(f.green_parent.plan for f in families) != response.family_plans:
        raise ValueError("Photon interaction and wavefunctions use different parent plans.")
    if term not in (_TERM_X, _TERM_SX, _TERM_COH):
        raise ValueError(f"Unknown static Sigma term {term}.")
    packed = response.V_packed if term == _TERM_X else response.W_packed
    if term == _TERM_COH:
        packed = packed - response.V_packed
    with_head = head_diagnostics and response.head_completion is not None
    pairs, bare = _photon_head_pairs(response, term, mesh_xy) if with_head else ((), ())
    for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
        keys = tuple((A, B) for A, B in blocks if bool(A) == bool(a) and bool(B) == bool(b))
        if not keys:
            continue
        left, right = families[a], families[b]
        slices = left.slices
        weights = (_occ_diag_full(Gij, slices.nb_sigma, slices.nb_full)
                   if term != _TERM_COH else left.band_mask(slices.sigma_sum).astype(jnp.complex128))
        weights = jax.lax.with_sharding_constraint(
            jnp.broadcast_to(weights, (meta.nk_tot, slices.nb_full)), NamedSharding(mesh_xy, P()))
        interactions, vertices = _make_photon_class_restore(response, keys)(packed)
        head_blocks = None
        if with_head:
            head_blocks = jnp.stack([photon_q0_low_rank_block(pairs, response.layout, A, B, mesh_xy)
                - (photon_q0_low_rank_block(bare, response.layout, A, B, mesh_xy) if bare else 0)
                for A, B in keys])
        kernel = _make_photon_static_class_kernel(mesh_xy, meta.kgrid, meta.nk_tot,
                                                  left, right, with_head=with_head)
        arguments = (left.green_parent, right.green_parent, weights, interactions,
                     -0.5 if term == _TERM_COH else 1.0, vertices, head_blocks)
        if admit_kernel is not None:
            admit_kernel(kernel, arguments, keys[0])
        value = kernel(*arguments)
        result, head = value if with_head else (value, None)
        yield keys[0], result, head


def compute_static_photon_sigma(
    *, wfns_charge, wfns_transverse, Gij, response, meta, mesh_xy,
    blocks=PHOTON_BLOCKS_ALL, diagnostic_basis_rotation=None,
    diagnostic_input_basis=None, head_diagnostics=False, print_fn=print, verbose=True,
):
    """Sum X/SX/COH Lorentz sectors on parents before their band-operator unfold."""
    from symmetry_maps import unfold_file_wedge_band_operator
    from .cohsex_sigma import _replicate_band_sigma
    if blocks not in _PHOTON_BLOCK_SELECTIONS:
        raise ValueError(f"Unknown photon block selection {blocks!r}.")
    families = (wfns_charge, wfns_transverse)
    if wfns_charge.slices != wfns_transverse.slices:
        raise ValueError("Photon endpoint band windows differ.")
    if head_diagnostics and response.head_completion is not None:
        if diagnostic_input_basis not in ("dft", "qp") or (
                (diagnostic_input_basis == "qp") != (diagnostic_basis_rotation is not None)):
            raise ValueError("Photon head diagnostic basis/rotation mismatch.")
    for name, packed in (("V", response.V_packed), ("W", response.W_packed)):
        _require_packed_operator(name, packed, mesh_xy)
    keys = [(a,b) for a in range(4) for b in range(4)
            if blocks == PHOTON_BLOCKS_ALL or a or b]
    sector_values = [[None]*3 for _ in range(3)]
    heads = [[None]*3 for _ in range(3)]
    totals, head_totals = [None]*3, [None]*3
    for term in range(3):
        for key, value, head in contract_lorentz_blocks(keys, families=families,
                term=term, response=response, Gij=Gij, meta=meta, mesh_xy=mesh_xy,
                head_diagnostics=head_diagnostics):
            sector = _head_sector(*key)
            old = sector_values[term][sector]
            sector_values[term][sector] = value if old is None else old + value
            totals[term] = value if totals[term] is None else totals[term] + value
            if head is not None:
                old = heads[term][sector]
                heads[term][sector] = head if old is None else old + head
                head_totals[term] = head if head_totals[term] is None else head_totals[term] + head
            if verbose and jax.process_index() == 0:
                print_fn(f"  packed photon Sigma term {term} class {key} submitted")
    sym = wfns_charge.green_parent.plan.sym
    nb = wfns_charge.slices.nb_sigma

    @jax.jit
    def finish(value):
        parent = _replicate_band_sigma(value, mesh_xy)[:, :nb, :nb]
        return unfold_file_wedge_band_operator(sym, parent, trs_rule="transpose")
    sig_x, sig_sx, sig_coh = (finish(value) for value in totals)
    zero = jnp.zeros_like(totals[0])
    sectors = jnp.stack([finish(
        (zero if sector_values[1][s] is None else sector_values[1][s]) +
        (zero if sector_values[2][s] is None else sector_values[2][s])) for s in range(3)])
    residual = float(jnp.max(jnp.abs(jnp.sum(sectors,axis=0)-(sig_sx+sig_coh))))
    limit = 1e-13 + 1e-11*float(jnp.max(jnp.abs(sig_sx+sig_coh)))
    if residual > limit:
        raise ValueError(f"GATE photon_sigma_sector_closure: {residual} > {limit}")
    diagnostics = StaticPhotonSigmaDiagnostics(sectors, residual)
    if not head_diagnostics or response.head_completion is None:
        return sig_x, sig_sx, sig_coh, None, diagnostics
    head_components = jnp.stack([jnp.stack([_diagnostic_diagonal(
        finish(zero if value is None else value), diagnostic_basis_rotation, mesh_xy)
        for value in row]) for row in heads])
    direct = jnp.stack([_diagnostic_diagonal(finish(value), diagnostic_basis_rotation,
                                         mesh_xy) for value in head_totals])
    residual = float(jnp.max(jnp.abs(jnp.sum(head_components,axis=1)-direct)))
    limit = 1e-13 + 1e-11*float(jnp.max(jnp.abs(direct)))
    if residual > limit:
        raise ValueError(f"GATE photon_head_sigma_sector_closure: {residual} > {limit}")
    return sig_x, sig_sx, sig_coh, StaticPhotonHeadSigmaDiagnostics(
        head_components, residual, "dft"), diagnostics
