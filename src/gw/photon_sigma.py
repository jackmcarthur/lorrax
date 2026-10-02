"""Stream Lorentz blocks through parent-face static Σ contractions."""

from __future__ import annotations

from ffi import _services
_services.ensure_on_path()
from distrib_la import mesh_key as _mesh_key
from dataclasses import dataclass
from functools import lru_cache

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


def _photon_class_kernel(node):
    """``jit``: one Lorentz class's static Σ, the sector engine's node at τ = 0.

    ``node`` is the class's :func:`gw.mpa.sector_sigma.sector_node` on its one W
    branch (:func:`_class_w_tables`); the interaction is the parent pair
    ``(W, conj W)`` on the irreducible q (:func:`_class_parents`) and the weights
    the static Green's band weights at every full k (occupations, or the COH
    band mask).  ``factor`` (-1/2 or 1, an exact power of two) scales the
    projected class.  One program per node, kept for the run.
    """
    hit = _photon_sigma_kernel_cache.get(node.key)
    if hit is None:
        from gw.mpa.sector_sigma import ParentW
        spatial, plan = node.spatial, node.plans[0]

        @jax.jit
        def contract_class(xn, xr, yr, yn, weights, interaction, factor, loads):
            weights = plan.parent_rows(weights)
            zero = jnp.zeros((), jnp.float64)
            return factor * spatial(xn, yr, xr, yn, jnp.zeros(weights.shape, jnp.float64),
                                    weights, zero, zero, ParentW(*interaction, hole=False), loads)
        # The node rides along so its key (ids of plans and tables) cannot be reused.
        hit = _photon_sigma_kernel_cache[node.key] = (node, contract_class)
    return hit[1]


def _photon_head_class_kernel(mesh_xy, nk_tot, wfns_left, wfns_right):
    """``jit``: the q -> 0 head diagnostic of one Lorentz class (``sigma_freq_debug_output``).

    The head is a pointwise product on the unfolded full-k Green
    (``cohsex_sigma.make_lorentz_q0_product``), so this debug path alone builds
    the class's whole parent Green and projects every band.
    """
    from ffi import ffi_dial_key
    from common.contract_bands import contract_bands_block_reshard
    from distrib_la import gemm_plan
    from .cohsex_sigma import make_lorentz_q0_product
    from .greens_function_kernel import build_G_parents
    left, right = wfns_left.green_parent, wfns_right.green_parent
    layout = left.layout
    plans = (left.plan, right.plan)
    shapes = tuple((p.n_parent, c.psi_nmu.shape[1], p.n_centroid_packed, p.nspinor)
                   for c, p in zip((left, right), plans))
    key = ("head", _mesh_key(mesh_xy), tuple(map(id, plans)), shapes, layout, ffi_dial_key(),
           int(nk_tot))
    hit = _photon_sigma_kernel_cache.get(key)
    if hit is None:
        project = contract_bands_block_reshard(mesh_xy, layout=layout,
            face_shape=shapes[0], right_face_shape=shapes[1])
        g_plan = gemm_plan(mesh_xy, m=shapes[0][2]*shapes[0][3], k=shapes[0][1],
            n=shapes[1][2]*shapes[1][3], nq=shapes[0][0], dtype=jnp.complex128, layout=layout)
        head_product = make_lorentz_q0_product(nk_tot)
        rows = jnp.asarray(np.asarray(plans[0].parent_full_rows))

        @jax.jit
        def head_class(left, right, weights, factor, head_interaction, head_vertices):
            weights = plans[0].parent_rows(weights)
            green = build_G_parents(left.psi_mun, right.psi_nmu, phases=jnp.real(weights),
                                    layout=layout, gemm=g_plan, k_unfold_plan=plans[0])
            G = plans[0].unfold_operator(green.G, operator_transpose=green.partner(),
                                         right_plan=plans[1])
            head_sigma = head_product(G, head_interaction, factor, head_vertices)
            return project(left.projection_faces()[0], jnp.take(head_sigma, rows, axis=0),
                           right.projection_faces()[1])
        # The entry keeps the plans alive, so their ids in the key cannot be reused.
        hit = _photon_sigma_kernel_cache[key] = (plans, head_class)
    return hit[1]


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


def _policy_key(policy):
    """Content key of a q-grid TRS policy (a frozen dataclass holding arrays)."""
    import dataclasses
    import hashlib
    if policy is None:
        return None
    parts = []
    for f in dataclasses.fields(policy):
        value = getattr(policy, f.name)
        if isinstance(value, np.ndarray):
            value = (value.shape, value.dtype.str,
                     hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
        parts.append((f.name, value))
    return tuple(parts)


@lru_cache(maxsize=None)
def band_sigma_finish(mesh_xy, nb, sym):
    """Replicate a parent-band Sigma, keep ``nb`` bands and unfold the wedge to full k.

    One program per (mesh, band count, SymMaps); SC maps keep the SymMaps
    object, so every map dispatches the same executable.
    """
    from symmetry_maps import unfold_file_wedge_band_operator
    from .cohsex_sigma import _replicate_band_sigma

    @jax.jit
    def finish(value):
        parent = _replicate_band_sigma(value, mesh_xy)[:, :nb, :nb]
        return unfold_file_wedge_band_operator(sym, parent, trs_rule="transpose")
    return finish


_CLASS_TABLES = {}


def _class_w_tables(plans, policy, lefts, rights, mesh_xy):
    """One photon class's q-unfold tables for the W-parent door, cached by content.

    The full-q restore these replace (``gw.w_isdf.photon_blocks_full_q``)
    unfolds each Lorentz block by the family plans' centroid maps on the TRS
    policy's operation rows (the Hermitian ``conj`` rule) and then mixes the
    blocks by the Lorentz action; these are the same tables with the Lorentz
    action as the endpoint action.  The door reads them with the
    pair-transpose rule and the partner conj(W), which is the conj rule.
    """
    from symmetry_maps import unfold_load_tables, bgw_integer_q_to_fractional
    key = (tuple(map(id, plans)), _policy_key(policy), tuple(lefts), tuple(rights),
           _mesh_key(mesh_xy))
    hit = _CLASS_TABLES.get(key)
    if hit is None:
        left, right = plans
        sym = left.sym
        rows = np.asarray(policy.unfold_sym_idx, np.int32)
        lorentz = np.asarray(sym.lorentz_action(rows), np.complex128)
        act = lambda ids: np.ascontiguousarray(lorentz[:, list(ids)][:, :, list(ids)])
        tables = unfold_load_tables(
            irr_idx=np.asarray(sym.irr_idx_q, np.int32), sym_idx=rows, sym_perm=left.sym_perm,
            L_table=left.L_table, k_irr_frac=bgw_integer_q_to_fractional(sym.q_irr_kgrid_int, policy.kgrid),
            spin_action_full=act(lefts), n_sym_spatial=int(policy.n_sym_spatial), mesh_xy=mesh_xy,
            logical_centroid_extent=left.n_centroid_packed, right_sym_perm=right.sym_perm,
            right_L_table=right.L_table, trs_rule='pair_transpose', right_spin_action_full=act(rights),
            right_logical_centroid_extent=right.n_centroid_packed)
        hit = _CLASS_TABLES[key] = (plans, tables)
    return hit[1]


@lru_cache(maxsize=None)
def _class_parents(mesh_xy, layout, lefts, rights):
    """``jit(packed -> (W, conj W))``: one class of a packed operator as ``(nq, m, nA, n, nB)``."""
    from .photon_layout import photon_block_view
    spec = NamedSharding(mesh_xy, P(None, 'x', None, 'y', None))

    @jax.jit
    def parents(packed):
        W = jnp.stack([jnp.stack([photon_block_view(packed, layout, A, B, mesh_xy) for B in rights], axis=-1)
                       for A in lefts], axis=2)
        W = jax.lax.with_sharding_constraint(W, spec)
        return W, jnp.conj(W)
    return parents


def contract_lorentz_blocks(blocks, *, families, term, response, Gij, meta, mesh_xy,
                            head_diagnostics=False, admit_kernel=None, placed=None):
    """Yield one parent-band sum per endpoint class, each the sector engine's τ = 0 node.

    Each class of the packed operator enters as its irreducible-q pair
    ``(W, conj W)`` (:func:`_class_parents`), unfolded on the k-convolution's
    load (:func:`_class_w_tables`): no full-q class operand and no whole-tile
    Green (:func:`gw.mpa.sector_sigma.sector_node`).  The value is
    ``(n_parent, carrier, carrier)`` on the QP window's band carrier.
    ``placed``: the families' node operands from :func:`place_photon_families`
    (placed here when ``None``).
    """
    from .cohsex_sigma import _occ_diag_full
    from .photon_layout import photon_q0_low_rank_block
    from gw.mpa.sector_sigma import sector_node
    if tuple(f.green_parent.plan for f in families) != response.family_plans:
        raise ValueError("Photon interaction and wavefunctions use different parent plans.")
    if term not in (_TERM_X, _TERM_SX, _TERM_COH):
        raise ValueError(f"Unknown static Sigma term {term}.")
    band_axis, operands = placed or place_photon_families(families, mesh_xy)
    packed = response.V_packed if term == _TERM_X else response.W_packed
    if term == _TERM_COH:
        packed = packed - response.V_packed
    with_head = head_diagnostics and response.head_completion is not None
    pairs, bare = _photon_head_pairs(response, term, mesh_xy) if with_head else ((), ())
    factor = -0.5 if term == _TERM_COH else 1.0
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
        lefts = tuple(dict.fromkeys(A for A, _ in keys))
        rights = tuple(dict.fromkeys(B for _, B in keys))
        tables = _class_w_tables(response.family_plans[a:a + 1] + response.family_plans[b:b + 1],
                                 response.qgrid_policy, lefts, rights, mesh_xy)
        pair = _class_parents(mesh_xy, response.layout, lefts, rights)(packed)
        node = sector_node(left, right, keys, meta, mesh_xy, (tables,), band_axis, static=True)
        kernel = _photon_class_kernel(node)
        arguments = (*operands[a][0], *operands[b][1], weights, pair, factor, node.loads)
        if admit_kernel is not None:
            admit_kernel(kernel, arguments, keys[0])
        result = kernel(*arguments)
        del pair
        head = None
        if with_head:
            from common.gamma_matrices import gamma_perm_phase
            head_blocks = jnp.stack([photon_q0_low_rank_block(pairs, response.layout, A, B, mesh_xy)
                - (photon_q0_low_rank_block(bare, response.layout, A, B, mesh_xy) if bare else 0)
                for A, B in keys])
            vertices = jax.tree.map(lambda *v: jnp.stack(v),
                *((gamma_perm_phase(A), gamma_perm_phase(B)) for A, B in keys))
            head = _photon_head_class_kernel(mesh_xy, meta.nk_tot, left, right)(
                left.green_parent, right.green_parent, weights, factor, head_blocks, vertices)
        yield keys[0], result, head


def place_photon_families(families, mesh_xy):
    """``(band_axis, operands)``: the QP window's static band carrier and each family's
    sector-node operands, ``((xn, xr), (yr, yn))`` (placed once per Σ call)."""
    from .ppm_sigma import sigma_band_axis
    from gw.mpa.sector_sigma import sector_left_operands, sector_right_operands
    band_axis = sigma_band_axis(int(families[0].slices.nb_sigma), mesh_xy, ansatz="static")
    placed = {}
    for f in families:
        if id(f) not in placed:
            placed[id(f)] = (sector_left_operands(f, band_axis, mesh_xy),
                             sector_right_operands(f, band_axis, mesh_xy))
    return band_axis, tuple(placed[id(f)] for f in families)


def compute_static_photon_sigma(
    *, wfns_charge, wfns_transverse, Gij, response, meta, mesh_xy,
    blocks=PHOTON_BLOCKS_ALL, diagnostic_basis_rotation=None,
    diagnostic_input_basis=None, head_diagnostics=False, print_fn=print, verbose=True,
):
    """Sum X/SX/COH Lorentz sectors on parents before their band-operator unfold."""
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
    placed = place_photon_families(families, mesh_xy)
    for term in range(3):
        for key, value, head in contract_lorentz_blocks(keys, families=families,
                term=term, response=response, Gij=Gij, meta=meta, mesh_xy=mesh_xy,
                head_diagnostics=head_diagnostics, placed=placed):
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
    finish = band_sigma_finish(mesh_xy, int(wfns_charge.slices.nb_sigma),
                               wfns_charge.green_parent.plan.sym)
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
