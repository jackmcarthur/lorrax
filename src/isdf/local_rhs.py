"""Atom-local density correction in the existing ISDF normal equations.

The sampled endpoint is the reconstructed carrier; the right endpoint is
evaluated twice, on the AE and smooth atomic grids. Their rectangular pair
Grams differ by exactly the local RHS. The shared core owns band contractions,
four-component half reduction, typed parent transport and k correlation.
"""
from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np


def _authenticate_faces(faces, plan, *, name):
    if not isinstance(faces, tuple) or len(faces) != 2:
        raise ValueError(f"local_density_rhs: {name} must be (psi_nmu, psi_mun)")
    pn, pm = faces
    if pn.ndim != 4 or pm.ndim != 4:
        raise ValueError(f"local_density_rhs: {name} faces must have four axes")
    npar, nb, ns, npoint = map(int, pn.shape)
    if tuple(pm.shape) != (npar, ns, npoint, nb):
        raise ValueError(f"local_density_rhs: {name} complementary faces disagree")
    if (npar != int(plan.n_parent) or ns != int(plan.nspinor)
            or npoint != int(plan.n_centroid_packed)):
        raise ValueError(f"local_density_rhs: {name} faces disagree with their typed plan")
    if pn.dtype != pm.dtype or str(pn.dtype) != 'complex128':
        raise ValueError(f"local_density_rhs: {name} requires complex128 faces")
    return npar, nb, ns, npoint


def _authenticate_plan_pair(left_plan, right_plan, *, kgrid, mesh_xy):
    for name in ('n_parent', 'n_full', 'nspinor', 'n_sym_spatial'):
        if int(getattr(left_plan, name)) != int(getattr(right_plan, name)):
            raise ValueError(f"local_density_rhs: left/right typed {name} differ")
    for name in ('irr_idx', 'sym_idx', 'k_parent_frac', 'spin_action_full',
                 'spatial_ops', 'translations'):
        left, right = getattr(left_plan, name), getattr(right_plan, name)
        if (left is None) != (right is None) or not np.array_equal(left, right):
            raise ValueError(f"local_density_rhs: left/right typed {name} differ")
    if (int(left_plan.n_full) != int(np.prod(kgrid))
            or left_plan.mesh_xy != mesh_xy or right_plan.mesh_xy != mesh_xy):
        raise ValueError("local_density_rhs: plans disagree with kgrid or device mesh")


@lru_cache(maxsize=None)
def _rhs_gemm(mesh_xy, npar, nb, ns, nmu, nlocal):
    from distrib_la import gemm_plan
    return gemm_plan(mesh_xy, m=ns*nmu, n=ns*nlocal, k=nb, nq=npar,
                     dtype=jnp.complex128, layout='face', warmup=False)


def _selected_q_layout(q_indices, kgrid, q_neg_idx):
    """Restrict the canonical q-negation map without changing its convention."""
    if q_indices is None:
        return None, q_neg_idx, None
    requested = np.asarray(q_indices)
    nk = int(np.prod(kgrid))
    if (requested.ndim != 1 or requested.dtype.kind not in 'iu' or len(requested) < 1
            or len(np.unique(requested)) != len(requested) or np.any(requested < 0)
            or np.any(requested >= nk)):
        raise ValueError("local_density_rhs: q_indices must name unique full-q rows")
    union = requested
    restricted_neg = None
    if q_neg_idx is not None:
        from symmetry_maps import q_negation_index
        canonical_neg = np.asarray(q_negation_index(kgrid))
        if not np.array_equal(np.asarray(q_neg_idx), canonical_neg):
            raise ValueError("local_density_rhs: selected q requires canonical q negation")
        union = np.unique(np.concatenate((requested, canonical_neg[requested])))
        restricted_neg = np.searchsorted(union, canonical_neg[union]).astype(np.int32)
    # No completion uses only the requested rows, preserving their order.
    output_rows = (np.searchsorted(union, requested).astype(np.int32)
                   if restricted_neg is not None else np.arange(len(requested),dtype=np.int32))
    return union.astype(np.int32), restricted_neg, output_rows


def local_density_rhs(*, centroid_faces, atom_ae_faces, atom_ps_faces,
                      left_plan, right_plan, weight_l, weight_r, kgrid,
                      mesh_xy, q_neg_idx=None, vertex_terms=((1.0, 0, 0),),
                      return_smooth=False, q_indices=None):
    r"""Fit-source ``Delta Z_q(mu,r_a)=Z_AE-Z_PS`` on an atomic point axis.

    Computes ``sum_pairs conj(A_pair(mu)) [rho_AE(r_a)-rho_PS(r_a)]``
    without constructing a band-pair tensor. The same augmented sample
    ``A_pair(mu)`` is used for both right endpoints. It is the local piece
    of the normal equation described in ``docs/theory/isdf-zeta-vq.md``.

    Parameters
    ----------
    centroid_faces : tuple of complex128 arrays
        Reconstructed sample carrier ``(psi_nmu,psi_mun)`` with shapes
        ``(n_parent,nb,4,nmu)`` and ``(n_parent,4,nmu,nb)``. Layouts are
        ``P(None,'x',None,'y')`` and ``P(None,None,'x','y')``.
    atom_ae_faces, atom_ps_faces : tuple of complex128 arrays
        The corresponding AE and smooth atomic-grid endpoints, with the
        same parent/band/spin axes and ``nlocal`` replacing ``nmu``.
        Wavefunction units and the normalized kinetic-balance representation
        must match the centroid samples. No quadrature weights are inserted.
    left_plan, right_plan : CentroidKUnfoldPlan
        Independently authenticated packed point axes. Raw-parent k rows,
        operation rows, spin actions and space-group metadata must agree.
    weight_l, weight_r : (nb,) float64
        Shared band-window weights, zero on padded band slots; replicated.
    kgrid : tuple of three int
        Full-zone correlation grid in the canonical flat k order.
    mesh_xy : Mesh
        The common square device mesh.
    q_neg_idx : (N_k,) int, optional
        Typed q-negation involution. Supplied for ordered LR+RL completion;
        each Cartesian vertex pair completes before complex channel weights.
    vertex_terms : tuple of (complex, int, int)
        Channel decomposition ``(weight,gamma_left,gamma_right)`` in the
        existing Cartesian monomial vertices, 0 through 3. Charge is 0,0.
    return_smooth : bool
        Charge-only optional retention of the smooth atomic RHS. Both the
        difference and smooth result use identical LR+RL completion and C.
    q_indices : integer array, optional
        Requested full-q rows in output order. Every native quarter is
        gathered onto their union with canonical negative partners before
        accumulation; LR+RL completes on that union, then returns these rows.

    Returns
    -------
    delta_z : (N_k,nmu,nlocal) complex128
        Full-q local RHS at ``P(None,'x','y')``, in each plan's packed order.
        All sixteen four-spinor terms survive through the four Pauli-half
        pairs. Atomic compression and quadrature belong to the next stage.
        With ``return_smooth=True``, returns ``(delta_z, smooth_z)`` instead.
        Every finite primitive contributes to both outputs after identical
        LR+RL completion, q selection and complex weighting.
    """
    from isdf.core import (_c_q_dirac_quarters,
                           complete_ordered_pair_normal_equations)

    _authenticate_plan_pair(left_plan, right_plan, kgrid=kgrid, mesh_xy=mesh_xy)
    npar, nb, ns, nmu = _authenticate_faces(
        centroid_faces, left_plan, name='centroid')
    ae = _authenticate_faces(atom_ae_faces, right_plan, name='atomic AE')
    ps = _authenticate_faces(atom_ps_faces, right_plan, name='atomic smooth')
    if ae != ps or ae[:3] != (npar, nb, ns):
        raise ValueError("local_density_rhs: all endpoint parent/band/spin axes must agree")
    if ns != 4:
        raise ValueError("local_density_rhs: the atomic correction currently requires four spinors")
    wl, wr = jnp.asarray(weight_l, jnp.float64), jnp.asarray(weight_r, jnp.float64)
    if wl.shape != (nb,) or wr.shape != (nb,):
        raise ValueError(f"local_density_rhs: band weights must have shape {(nb,)}")
    terms = tuple((complex(w), int(i), int(j)) for w, i, j in vertex_terms)
    if not terms or any(i not in range(4) or j not in range(4) for _, i, j in terms):
        raise ValueError("local_density_rhs: vertex terms must use Cartesian vertices 0..3")
    if any(not np.isfinite(w) for w, _, _ in terms):
        raise ValueError("local_density_rhs: vertex weights must be finite")
    gemm = _rhs_gemm(mesh_xy, npar, nb, ns, nmu, ae[-1])
    selected_q, completion_neg, output_rows = _selected_q_layout(q_indices,kgrid,q_neg_idx)
    total = smooth_total = None
    for weight, left_vertex, right_vertex in terms:
        def endpoint_rhs(right_nmu):
            return _c_q_dirac_quarters(
                centroid_faces[1], right_nmu, wl, wr, plan=left_plan,
                right_plan=right_plan, kgrid=kgrid, mesh_xy=mesh_xy,
                gemm=gemm, gamma_L=left_vertex, gamma_R=right_vertex,
                **({'q_indices':selected_q} if selected_q is not None else {}))
        ae_rhs = endpoint_rhs(atom_ae_faces[0])
        ps_rhs = endpoint_rhs(atom_ps_faces[0])
        piece = ae_rhs - ps_rhs
        del ae_rhs
        if not return_smooth:
            del ps_rhs
        if completion_neg is not None:
            piece = complete_ordered_pair_normal_equations(piece, completion_neg)
        if output_rows is not None:
            piece = jnp.take(piece,jnp.asarray(output_rows),axis=0)
        if weight != 1.0:
            piece = weight * piece
        total = piece if total is None else total + piece
        if return_smooth:
            if completion_neg is not None:
                ps_rhs = complete_ordered_pair_normal_equations(ps_rhs, completion_neg)
            if output_rows is not None:
                ps_rhs = jnp.take(ps_rhs,jnp.asarray(output_rows),axis=0)
            if weight != 1.0:
                ps_rhs = weight * ps_rhs
            smooth_total = ps_rhs if smooth_total is None else smooth_total + ps_rhs
    if return_smooth:
        return total, smooth_total
    return total
