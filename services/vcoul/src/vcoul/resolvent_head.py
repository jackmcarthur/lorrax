"""Scalar macroscopic completion of a microscopic resolvent and its actions.

The response contains the q-squared Cartesian coefficients Xi(s), dXi/ds
and physical moments Xi(z)=2 M1/z**2+2 M3/z**4+..., s=z**2.
Only the omitted macroscopic Coulomb line is restored here. Geometry and
quadrature come from the existing bare-kernel sample providers.
"""
from __future__ import annotations

import numpy as np

from vcoul.minibz import iter_minibz_photon_samples, slab_minibz_photon_cubature


def _chunk_fields(q, v, weight, response):
    """Weighted sums of Wc, its exact s derivative and its first two moments."""
    tensors = np.asarray(response['Xi'], np.complex128)
    derivatives = np.asarray(response['dXi_ds'], np.complex128)
    moments = response['moments']
    if tensors.ndim != 3 or tensors.shape[1:] != (3, 3) or derivatives.shape != tensors.shape:
        raise ValueError('scalar resolvent head needs Xi/dXi_ds with matching (nz,3,3) shapes')
    qform = lambda a: np.einsum('qi,ij,qj->q', q, a, q, optimize=True)
    values, slopes = [], []
    for tensor, derivative in zip(tensors, derivatives):
        polar = qform(tensor)
        denominator = 1-v*polar
        # Correlation form avoids subtracting two large bare heads at high z.
        values.append(np.sum(weight*v*v*polar/denominator))
        slopes.append(np.sum(weight*v*v*qform(derivative)/denominator**2))
    c1, c3 = 2*qform(moments['M1']), 2*qform(moments['M3'])
    return dict(vc0=np.sum(weight*v), Wc=np.asarray(values),
        dWc_ds=np.asarray(slopes),
        M1=.5*np.sum(weight*v*v*c1),
        M3=.5*np.sum(weight*(v*v*c3+v**3*c1*c1)))


def scalar_resolvent_head(kernel, geometry, kgrid, response, *, analytic_sphere=False):
    """Complete and average a microscopic scalar-head resolvent.

    Parameters
    ----------
    kernel, geometry, kgrid
        Existing dimension-specific Coulomb kernel, authenticated geometry,
        and three integer k extents. Outputs are bare head units, without
        an additional cell-volume or k-point factor.
    response : mapping
        ``Xi``/``dXi_ds`` complex arrays (nz,3,3), and ``moments`` with
        (3,3) M1/M3 tensors in the existing CH**jB/2 convention.
    analytic_sphere : bool
        The bulk analytic-sphere extension is not represented by the sampled
        integrand and refuses; it is never silently replaced by a point rule.

    Returns
    -------
    dict
        Scalar ``vc0``, (nz,) Wc/dWc_ds, high-frequency M1/M3 and quadrature
        receipt. The inverse-frequency moment used by Sigma is different.
    """
    if analytic_sphere:
        raise ValueError('GATE scalar_resolvent_head_sphere: analytic-sphere response actions unavailable')
    dimension = int(kernel.sys_dim)
    if dimension == 2:
        # The same authenticated 16/24/32 WS/Duffy ladder as q0_average.
        receipt = slab_minibz_photon_cubature(kernel, geometry, kgrid)
        ladder = []
        for chunk in receipt.chunks:
            count = int(chunk.physical_count)
            weight = np.asarray(chunk.sample_weight[:count])
            fields = _chunk_fields(np.asarray(chunk.q_cart[:count]),
                np.asarray(chunk.D_raw[:count,0,0]), weight, response)
            measure = float(weight.sum())
            if not np.isfinite(measure) or measure <= 0:
                raise ValueError('GATE scalar_resolvent_head_measure: positive finite quadrature required')
            ladder.append({name: value/measure for name,value in fields.items()})
        # Reuse the scalar owner's mandatory convergence rule for every
        # response action, not only its bare/value component.
        for name in ('Wc','dWc_ds','M1','M3'):
            width = np.size(ladder[-1][name])
            for slot in range(width):
                rows = [(order, row['vc0'], np.ravel(row[name])[slot])
                        for order,row in zip(receipt.orders,ladder)]
                kernel._require_q0_ladder_converged(receipt, rows)
        result = ladder[-1]
        rule = dict(method=receipt.method, orders=list(receipt.orders),
                    counts=list(receipt.physical_counts))
    elif dimension == 3:
        totals, counts = {}, {}
        for rep, _start, _stop, q, raw, count, weight, _analytic in iter_minibz_photon_samples(
                kernel, geometry, kgrid, method='auto', analytic_sphere=False):
            fields = _chunk_fields(q[:count], raw[:count,0,0], weight[:count], response)
            if rep not in totals:
                totals[rep],counts[rep] = fields,count
            else:
                totals[rep] = {name: totals[rep][name]+value for name,value in fields.items()}
                counts[rep] += count
        result = {name: np.mean([row[name]/counts[rep] for rep,row in totals.items()],axis=0)
                  for name in totals[0]}
        rule = dict(method='existing bulk auto sampler; fixed rep+0 seeds', replicates=len(totals),
                    samples=[counts[rep] for rep in sorted(counts)])
    else:
        raise ValueError('GATE scalar_resolvent_head_dimension: finite-q 2D/3D head required')
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError('GATE scalar_resolvent_head_nonfinite: all response actions must be finite')
    rule.update(kgrid=list(map(int,kgrid)),cell_volume=float(geometry.cell_volume),
                reciprocal_lattice_rows=np.asarray(geometry.bvec).tolist())
    return dict(result,quadrature=rule)
