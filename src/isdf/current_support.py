"""Actual source/body support for same-Z current specializations.

Physical Slab admission additionally requires the actual all-atom layer
geometry and point-kernel tables. Neither witness bounds the absent global
compensation tail or proves radial/angular or occupied-Sigma convergence.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _hash_arrays(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _source_support_digest(wfn, G, counts, k, q):
    from vcoul import CoulombGeometry
    geometry = CoulombGeometry.from_wfn(wfn)
    return _hash_arrays(G, counts, k, wfn.fft_grid, geometry.bvec,
        np.asarray([geometry.cell_volume]), wfn.kgrid, np.asarray([wfn.nbands]), q)


def _current_q_chart(wfn, plan, zeta):
    return _physical_q_chart(wfn, plan, zeta.fit_q_full_indices, int(zeta.store.Q))


def _physical_q_chart(wfn, plan, q_full_indices, Q):
    from symmetry_maps import bgw_integer_q_to_fractional
    q_rows = np.asarray(q_full_indices)
    integer_q = np.asarray(plan.sym.kvecs_asints)
    if (q_rows.shape != (Q,) or q_rows.dtype.kind not in 'iu'
            or np.any(q_rows < 0) or np.any(q_rows >= len(integer_q))
            or len(np.unique(q_rows)) != len(q_rows)):
        raise ValueError('same-Z support requires exact distinct physical q row indices')
    return bgw_integer_q_to_fractional(integer_q[q_rows], np.asarray(wfn.kgrid))


def _actual_atom_centers(wfn):
    fractional = np.asarray(wfn.atom_crys, np.float64)
    lattice = float(wfn.alat) * np.asarray(wfn.avec, np.float64)
    if (fractional.ndim != 2 or fractional.shape[1] != 3 or not len(fractional)
            or not np.isfinite(fractional).all() or lattice.shape != (3, 3)
            or not np.isfinite(lattice).all() or np.linalg.det(lattice) <= 0):
        raise ValueError('physical mixed current requires actual finite atom/cell geometry')
    return (fractional % 1.) @ lattice


def _slab_layer_geometry(wfn, body_head_slots, q, centers_cart, support_radius):
    """All atom/image half-period clearances for the public point Slab.

    Q0/Q2 neutrality gives n=K^4 h with compact h. DeltaT n is a
    differential polynomial applied to a Poisson kernel on the half-cell
    shifted z planes. Every pair must avoid both compact supports. The
    radial provider retains its separate disjoint-sphere and moment guards.
    """
    from vcoul import Slab2D
    from isdf.atomic_breit import SLAB_BODY_HEAD_SCHEMA
    centers = np.asarray(centers_cart, np.float64)
    actual = _actual_atom_centers(wfn)
    radius = np.asarray(support_radius)
    if (centers.shape != actual.shape or not np.isfinite(centers).all()
            or not np.allclose(centers, actual, rtol=0, atol=2e-12)
            or radius.dtype.kind not in 'fiu' or radius.shape not in ((), (len(actual),))):
        raise ValueError('physical mixed current must use actual WFN centers and support radii')
    radii = np.broadcast_to(radius.astype(np.float64), (len(actual),)).copy()
    if not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError('physical mixed current requires finite positive support radii')
    q = np.asarray(q, np.float64)
    if (body_head_slots.get('schema') != SLAB_BODY_HEAD_SCHEMA
            or q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all()
            or np.any(q[:, 2] != 0.)):
        raise ValueError('physical mixed current requires the actual qz=0 Slab chart')
    zc = Slab2D._truncation_half_height_from_bvec(body_head_slots['bvec'])
    if float(body_head_slots['slab_half_height']) != zc:
        raise ValueError('physical mixed current Slab half-height changed')
    if np.any(np.asarray(body_head_slots['gamma_head_cartesian'])):
        raise ValueError('physical mixed current requires actual point Gamma kernel')
    dz = centers[:, 2, None] - centers[None, :, 2]
    Lz = 2 * zc
    images = np.rint(dz / Lz - .5).astype(np.int64)
    separation = abs(dz - (images + .5) * Lz)
    margin = separation - radii[:, None] - radii[None, :]
    tolerance = 256 * np.finfo(float).eps * max(Lz, 2 * float(np.max(radii)), 1.)
    if np.any(margin <= tolerance):
        raise ValueError('physical mixed current atom/image supports cross Slab shifted planes')
    return dict(centers=centers.copy(), actual_centers=actual, radii=radii,
        descriptor=dict(schema='lorrax.current.point_Slab_layer_support.v1',
            atom_geometry_sha256=_hash_arrays(actual, radii),
            prepared_geometry_sha256=_hash_arrays(centers, radii),
            centers_cart=centers.tolist(), support_radii=radii.tolist(),
            truncation_half_height=zc, period_z=Lz,
            nearest_half_period_images=images.tolist(),
            all_pair_clearance_margin=margin.tolist(),
            minimum_clearance_margin=float(np.min(margin)),
            geometry_rounding_envelope=tolerance,
            qz_zero=True, kernel_policy='actual_point_Slab_tables_required',
            moment_policy='per_Cartesian_retained_harmonic_Q0_Q2',
            completion='mixed_reciprocal_plus_bulk_onsite_D',
            qualification='Provider must retain disjoint-sphere/Q0,Q2 guards; '
                'global compensation tail and occupied-Sigma bound remain separate'))


def _authenticate_point_kernel_tables(policy, pairs, v_tables, current_basis_rows, prefactor):
    from isdf.atomic_breit import _current_point_body_tile
    pairs = tuple((int(a), int(b)) for a, b in pairs)
    basis = np.asarray(current_basis_rows, np.complex128)
    scale = float(prefactor)
    Q, ng = np.asarray(policy['gvec_components']).shape[::2]
    if (not pairs or len(pairs) != len(v_tables) or len(set(pairs)) != len(pairs)
            or any(a < 0 or b < 0 or a >= 3 or b >= 3 for a, b in pairs)
            or basis.shape != (3, 3) or not np.isfinite(basis).all()
            or not np.allclose(basis @ basis.conj().T, np.eye(3), rtol=0, atol=2e-12)
            or not np.isfinite(scale) or scale == 0):
        raise ValueError('invalid physical mixed point-kernel basis/pairs/prefactor')
    tables = tuple(np.asarray(passed, np.complex128) for passed in v_tables)
    for values in tables:
        if values.shape != (Q, ng) or not np.isfinite(values).all():
            raise ValueError('physical mixed actual point-kernel table domain changed')
    for first in range(0, ng, 512):
        width = min(512, ng - first)
        expected = _current_point_body_tile(policy, np.arange(Q), first, width,
            pairs, basis, scale, sys_dim=2)
        for index, values in enumerate(tables):
            wanted = expected[:, index]
            error = abs(values[:, first:first + width] - wanted)
            envelope = 512 * np.finfo(float).eps * (1 + abs(wanted))
            if np.any(error > envelope):
                raise ValueError('physical mixed current requires actual point Slab kernel; '
                                 'finite averaged/head replacements need their own D_H completion')



def _source_body_support(wfn, plan, q_full_indices, body_head_slots):
    from vcoul import CoulombGeometry
    from common.coulomb_sphere import compute_per_q_bare_coulomb_components
    from isdf.atomic_breit import _authenticate_body_head_slots
    G, counts, k = _source_geometry(wfn, plan)
    grid = np.asarray(wfn.fft_grid)
    if grid.shape != (3,) or grid.dtype.kind not in 'iu' or np.any(grid < 2):
        raise ValueError('same-Z support requires the actual integer FFT grid')
    geometry = CoulombGeometry.from_wfn(wfn)
    B = np.asarray(geometry.bvec, np.float64)
    if not np.array_equal(B, np.asarray(body_head_slots['bvec'])):
        raise ValueError('same-Z support source and body reciprocal cells differ')
    max_K2 = 0.
    for row, n, parent_k in zip(G, counts, k):
        physical = row[:n]
        wrapped = physical.astype(np.int64) % grid
        flat = (wrapped[:, 0]*grid[1]+wrapped[:, 1])*grid[2]+wrapped[:, 2]
        if len(np.unique(flat)) != n:
            raise ValueError('same-Z source FFT map is not injective')
        K = (physical+parent_k) @ B
        max_K2 = max(max_K2, float(np.max(np.sum(K*K, axis=1))))
    cutoff = float(body_head_slots['cutoff_ry'])
    if not np.isfinite(cutoff) or cutoff <= 0 or 4*max_K2 > cutoff:
        raise ValueError('same-Z body cutoff does not cover actual source-pair support')
    q = _physical_q_chart(wfn, plan, q_full_indices, len(body_head_slots['q_frac']))
    if not np.array_equal(q, np.asarray(body_head_slots['q_frac'])):
        raise ValueError('same-Z support current q chart differs from body q chart')
    # This sufficient ellipsoid bound also excludes missing Nyquist images.
    coordinate_bound = np.sqrt(cutoff)*np.linalg.norm(np.linalg.inv(B), axis=0)
    margin = grid/2-coordinate_bound-np.max(abs(q), axis=0)
    if not np.all(margin > 0):
        raise ValueError('same-Z body sphere cannot be represented uniquely on the actual FFT')
    package = compute_per_q_bare_coulomb_components(tuple(grid), B, q, cutoff)
    expected_G = np.asarray(package['gvec_components_padded'])
    expected_n = np.asarray(package['ngk_per_q'])
    body_G = np.asarray(body_head_slots['gvec_components'])
    body_n = np.asarray(body_head_slots['ngk_per_q'])
    if not np.array_equal(expected_n, body_n):
        raise ValueError('same-Z support requires the complete public body sphere')
    for i, n in enumerate(body_n):
        if not np.array_equal(expected_G[i, :, :n], body_G[i, :, :n]):
            raise ValueError('same-Z support body reciprocal rows are incomplete or reordered')
    K = (body_G.transpose(0, 2, 1)+q[:, None]) @ B
    active = np.arange(body_G.shape[-1])[None] < body_n[:, None]
    mask = active & (np.sum(K*K, axis=-1) <= cutoff)
    _authenticate_body_head_slots(body_head_slots, K, body_n,
                                  float(geometry.cell_volume), mask)
    return dict(source=(G, counts, k), grid=grid, q=q, B=B,
                max_K2=max_K2, cutoff=cutoff, margin=margin,
                body_G=body_G, body_n=body_n)


class SmoothCurrentSupport:
    """Typed witness built only from actual WFN and public body owners."""
    __slots__ = ('_wfn', '_plan', '_zetas', '_bindings', '_factors',
                 '_source_digest', '_body_digest', '_descriptor',
                 '_body_head_slots', '_layer')

    @property
    def descriptor(self):
        return json.loads(self._descriptor)

    @property
    def physical_mixed(self):
        """Derived layer admission; callers cannot replace it with a bool."""
        return self._layer is not None

    def authenticate(self, zetas, body_head_slots=None, *, centers_cart=None,
                     support_radius=None):
        from isdf.current_coefficients import _fit_binding, _factor_leaves
        from isdf.atomic_breit import _body_head_digest
        if (self._wfn is None
                or tuple(id(z) for z in zetas) != tuple(id(z) for z in self._zetas)
                or tuple(_fit_binding(z) for z in zetas) != self._bindings
                or any(len(_factor_leaves(z)) != len(saved)
                    or any(a is not b for a, b in zip(_factor_leaves(z), saved))
                    for z, saved in zip(zetas, self._factors))):
            raise ValueError('same-Z support requires unchanged actual OWN/source/q/basis')
        policy = (self._body_head_slots if body_head_slots is None and self.physical_mixed
                  else body_head_slots)
        if policy is not None and (
                policy.get('payload_sha256') != self._body_digest
                or _body_head_digest(policy) != self._body_digest):
            raise ValueError('same-Z support body/head identity changed')
        G, counts, k = _source_geometry(self._wfn, self._plan)
        q = _current_q_chart(self._wfn, self._plan, zetas[0])
        if _source_support_digest(self._wfn, G, counts, k, q) != self._source_digest:
            raise ValueError('same-Z support actual WFN reciprocal support changed')
        if self._layer is not None:
            actual = _actual_atom_centers(self._wfn)
            saved = self._layer
            if _hash_arrays(actual, saved['radii']) != saved['descriptor']['atom_geometry_sha256']:
                raise ValueError('physical mixed current actual atom geometry changed')
            if centers_cart is not None or support_radius is not None:
                if centers_cart is None or support_radius is None:
                    raise ValueError('physical mixed current requires both geometry operands')
                layer = _slab_layer_geometry(self._wfn, self._body_head_slots, q,
                                            centers_cart, support_radius)
                if layer['descriptor'] != saved['descriptor']:
                    raise ValueError('physical mixed current prepared geometry changed')
        elif centers_cart is not None or support_radius is not None:
            raise ValueError('private hybrid witness has no physical Slab layer admission')

    def authenticate_point_kernel(self, pairs, v_tables, current_basis_rows, prefactor):
        """Check actual tables against the shared public point-body owner.

        Metadata flags do not exclude finite-q averaged heads. This check
        deliberately compares all passed physical tables, including Gamma
        and tied minimum slots. No Fourier or source coefficient scan occurs.
        """
        if not self.physical_mixed:
            raise ValueError('physical mixed current requires actual Slab layer admission')
        self.authenticate(self._zetas, self._body_head_slots)
        _authenticate_point_kernel_tables(self._body_head_slots, pairs, v_tables,
                                          current_basis_rows, prefactor)

    def close(self):
        """Drop actual factor/source owners; preserve only descriptor provenance."""
        self._wfn = self._plan = None
        self._zetas = self._factors = ()
        self._body_head_slots = None


def _source_geometry(wfn, plan):
    domain = plan.sym.parent_k_domain
    G = np.asarray(wfn.gvecs(k=domain))
    counts = np.asarray(wfn.ngk_valid(k=domain))
    k = np.asarray(wfn.kvecs(k=domain), np.float64)
    if (G.ndim != 3 or G.shape[-1] != 3 or not np.isfinite(G).all()
            or np.any(G != np.rint(G)) or counts.shape != (len(G),)
            or counts.dtype.kind not in 'iu' or np.any(counts < 1)
            or np.any(counts > G.shape[1]) or k.shape != (len(G), 3)
            or not np.isfinite(k).all()
            or not np.array_equal(k, np.asarray(plan.k_parent_frac))):
        raise ValueError('same-Z support must use the actual plan WFN parent chart')
    return G, counts, k


def preflight_physical_current_support(wfn, plan, q_full_indices, body_head_slots, *,
                                       centers_cart, support_radius, pairs, v_tables,
                                       current_basis_rows, prefactor):
    """Reject unsupported actual source/layer/kernel before RHS/C-fit allocation.

    No fake OWN factors or Zeta objects are constructed. The returned plain
    provenance cannot replace the post-fit typed live-factor witness.
    """
    from isdf.atomic_breit import _body_head_digest
    support = _source_body_support(wfn, plan, q_full_indices, body_head_slots)
    layer = _slab_layer_geometry(wfn, body_head_slots, support['q'], centers_cart, support_radius)
    _authenticate_point_kernel_tables(body_head_slots, pairs, v_tables,
                                      current_basis_rows, prefactor)
    G, counts, k = support['source']
    return dict(schema='lorrax.current.physical_Slab_preflight.v1',
        source_geometry_sha256=_source_support_digest(wfn, G, counts, k, support['q']),
        body_head_sha256=_body_head_digest(body_head_slots),
        physical_source_bands=int(wfn.nbands),
        q_full_indices=np.asarray(q_full_indices, int).tolist(),
        FFT_grid=support['grid'].tolist(), source_max_K2_Ry=support['max_K2'],
        pair_bandwidth_bound_Ry=4*support['max_K2'], body_cutoff_Ry=support['cutoff'],
        unique_FFT_margin=support['margin'].tolist(), physical_Slab_layer=layer['descriptor'],
        qualification='Actual source/layer/point tables only; post-fit typed OWN3 witness, '
                      'provider disjoint-sphere/moment and compensation-tail guards remain required')


def certify_smooth_current_support(wfn, plan, zetas, body_head_slots, *,
                                  centers_cart=None, support_radius=None):
    """Prove source-pair coverage using the same public q-aware body sphere.

    Normalized RKB and band rotations do not add G support; the fit solve
    mixes centroid rows only. Rebuilding this witness from actual operands
    therefore admits the smooth restriction identity without a user bool.
    No coefficient payload is read, and no current factor is rebuilt.
    Optional actual prepared centers/radii additionally derive the physical
    Slab layer witness; the provider must authenticate actual point tables
    and retain its disjoint-sphere and Q0/Q2 moment guards.
    """
    from isdf.current_coefficients import _fit_binding, _factor_leaves
    from isdf.atomic_breit import _body_head_digest
    if (len(zetas) != 3 or body_head_slots is None
            or any(z.mesh is not plan.mesh_xy or z.mu_basis is not zetas[0].mu_basis
                   or z.fit_vertex_mu_L != c for c, z in enumerate(zetas, 1))
            or any(z.fit_augmentation_identity != zetas[0].fit_augmentation_identity
                   for z in zetas)):
        raise ValueError('same-Z support requires one complete actual OWN3 fit family')
    support = _source_body_support(wfn, plan, zetas[0].fit_q_full_indices, body_head_slots)
    G, counts, k = support['source']
    grid, q = support['grid'], support['q']
    max_K2, cutoff, margin = support['max_K2'], support['cutoff'], support['margin']
    body_G, body_n = support['body_G'], support['body_n']
    for z in zetas:
        if (not np.array_equal(z.ngk_per_q, body_n)
                or not np.array_equal(z.fit_q_full_indices, zetas[0].fit_q_full_indices)
                or int(z.store.g_axis.logical) < int(np.max(body_n))):
            raise ValueError('same-Z support zeta q/count identity mismatch')
        for i, n in enumerate(body_n):
            if (not np.array_equal(body_G[i, :, :n], z.gvec_components[i, :, :n])):
                raise ValueError('same-Z support body or zeta reciprocal rows are incomplete or reordered')
    if (centers_cart is None) != (support_radius is None):
        raise ValueError('physical mixed current requires both geometry operands')
    layer = (None if centers_cart is None else
             _slab_layer_geometry(wfn, body_head_slots, q, centers_cart, support_radius))
    witness = SmoothCurrentSupport()
    witness._wfn, witness._plan, witness._zetas = wfn, plan, tuple(zetas)
    witness._bindings = tuple(_fit_binding(z) for z in zetas)
    witness._factors = tuple(_factor_leaves(z) for z in zetas)
    witness._source_digest = _source_support_digest(wfn, G, counts, k, q)
    witness._body_digest = _body_head_digest(body_head_slots)
    witness._body_head_slots, witness._layer = body_head_slots, layer
    witness._descriptor = json.dumps(dict(
        schema=('lorrax.current.same_Z_hybrid_support.v1' if layer is None
                else 'lorrax.current.same_Z_physical_Slab_support.v1'),
        source_geometry_sha256=witness._source_digest,
        body_head_sha256=witness._body_digest,
        field_identity=zetas[0].fit_augmentation_identity,
        physical_source_bands=int(wfn.nbands),
        q_full_indices=np.asarray(zetas[0].fit_q_full_indices, int).tolist(),
        FFT_grid=grid.tolist(), source_max_K2_Ry=max_K2,
        pair_bandwidth_bound_Ry=4*max_K2, body_cutoff_Ry=cutoff,
        unique_FFT_margin=margin.tolist(),
        physical_Slab_layer=None if layer is None else layer['descriptor'],
        qualification=('Reciprocal support for the same-Z hybrid only; '
                       'no compensation-tail or continuum-Slab convergence' if layer is None else
                       'Support and all-atom Slab layer only; actual point-kernel/provider moment guards '
                       'required; global compensation tail and occupied-Sigma bound remain separate')),
        sort_keys=True)
    witness.authenticate(zetas, body_head_slots)
    return witness
