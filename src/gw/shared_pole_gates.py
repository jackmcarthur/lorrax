"""Measured gates and diagnostics of a constructed shared-pole model.

Moment identities, the zero-Ritz policy, passivity and reciprocity. The receipt
rows these feed are listed in docs/architecture/shared_pole_model.md section 9.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from distrib_la import hermitian_part
from gw.shared_pole_pencil import _adjoint


def ordered_moment_identity(signed, infinity, *, matmul):
    """Projected z-moments of the signed model against the bank's M0..M3.

    m_n(model) = sum_j c_j c_j^H mu_j^-(n+1) on the ORIGINAL Q_inf, relative
    Frobenius defect of each order against its own |Q^H 2M_n Q|. With k0, k1
    in the full Galerkin span it matches n = 0..3 exactly; after the keep and
    retention cuts only to projection accuracy (diagnostic, like the TRS
    full_m1/full_m3 rows). ``infinity`` is the five-panel tuple; returns {m0..m3: [b]}.
    """
    c, mu, retained = signed
    qi, *moments = infinity
    a = matmul(qi, c, transa="C")
    inverse = jnp.where(retained, 1 / jnp.where(retained, mu, 1), 0)
    rows = {}
    for n, m in enumerate(moments):
        exact = 2 * matmul(qi, m, transa="C")
        value = matmul(a * (inverse ** (n + 1))[:, None, :], a, transb="C")
        scale = jnp.linalg.norm(exact, axis=(-2, -1))
        rows[f"m{n}"] = jnp.linalg.norm(value - exact, axis=(-2, -1)) / jnp.where(scale > 0, scale, 1)
    return rows


def ordered_pole_bound_ry(m1, inverse_coulomb_sqrt, *, energy_span_ry, gap_ry, matmul, eigh):
    """Upper bound on the RPA poles of W from the bank's own quantities, per parent.

    The poles are +-eigenvalues of s3 M, M = M0 + K with M0 = diag(D) the bare
    transitions and K = Phi^H V Phi >= 0, so |Omega| <= ||M|| <= D_max + ||K||.
    Every term of A0 = sum_j D_j (P_j + conj P_j) is PSD with D_j >= D_min, so
    ||K|| <= ||x2|| / D_min with x2 = H A0 H, and x2 <= H^+ 2M1 H^+ (the
    difference is x1^2 >= 0). Hence Omega_max <= D_max + lambda_max(H^+ 2M1 H^+)/D_min.
    ``energy_span_ry`` >= D_max and ``gap_ry`` <= D_min come from the bank census;
    a gapless census returns +inf and the bound never fires. ``m1`` and
    ``inverse_coulomb_sqrt`` are [b,n,n] face arrays; returns [b] float64 (Ry).
    """
    if not float(gap_ry) > 0:
        return jnp.full((m1.shape[0],), jnp.inf)
    whitened = hermitian_part(matmul(inverse_coulomb_sqrt, matmul(2 * m1, inverse_coulomb_sqrt)))
    top = eigh(whitened)[0][:, -1]
    return float(energy_span_ry) + jnp.maximum(top, 0) / float(gap_ry)


def ordered_shared_pole_value(model, partner, z, *, matmul):
    """Evaluate the ordered carrier Wc_p(z) from stored positive-pole factors.

    ``model`` and ``partner`` are (b [b,n,K], poles2 [b,K], active [b,K]) for
    parent p and for the parent of -q (the same tuple at q = -q). Returns
    b diag(1/(2W(z-W))) b^H - conj(b~) diag(1/(2W~(z+W~))) b~^T. At p = -q
    this is sum Re(bb^H)/(z^2-W^2) + i Im(bb^H) z/(W(z^2-W^2)): the odd channel
    is one extra contraction of the same vector, with no extra storage.
    """
    b, poles2, active = model
    bt, poles2t, activet = partner
    w = jnp.sqrt(jnp.where(active, poles2, 1.0))
    wt = jnp.sqrt(jnp.where(activet, poles2t, 1.0))
    d = jnp.where(active, 1 / (2 * w * (z - w)), 0)
    dt = jnp.where(activet, 1 / (2 * wt * (z + wt)), 0)
    return (matmul(b * d[:, None, :], b, transb="C")
            - matmul(bt.conj() * dt[:, None, :], bt.conj(), transb="C"))


def signed_shared_pole_passivity(signed, inverse_coulomb_sqrt, *, eta_ry, matmul, eigh, gates):
    """Hermitian-part passivity of the signed particle-hole model at z = i eta.

    -Wc_r(i eta) = sum_j c_j c_j^H/(1 - i eta mu_j); its Hermitian part sums
    both signs of real frequency and lies in [0, I] after V whitening for a
    stable response. The anti-Hermitian part is the odd channel: reported.
    """
    c, mu, retained = signed
    whitened = matmul(inverse_coulomb_sqrt, c)
    weight = jnp.where(retained, 1 / (1 - 1j * eta_ry * mu), 0)
    response = matmul(whitened * weight[:, None, :], whitened, transb="C")
    return _passivity_response_checks(response, eigh=eigh, gates=gates)


def retained_moment_identity(pencil, coefficients, model, infinity_selector, *, matmul):
    """Check the two moment blocks on projected latent infinity states.

    Let E select infinity columns of X, G=X.H X, H=X.H T X, and Y be
    the active Ritz coefficient map after both Gram and zero-policy cuts.
    A=Y.H G E and B=Y A represent P_ret X_inf = X B. The independent
    original-pencil targets are B.H G B / 2 and B.H H B / 2; the model
    values are A.H A / 2 and A.H Lambda A / 2. These compare the same
    retained latent states, not the discarded original infinity components.

    ``pencil=(G,H,O)`` has face tiles [b,R,R]/[b,n,R]; coefficients is
    [b,R,Kp], model=(b,Lambda,active), and infinity_selector E [b,R,r_inf].
    All arrays are complex128 face tiles except replicated real Lambda and
    boolean active. Returns relative Frobenius defects, one per q and moment.
    """
    g, h, _ = pencil
    _, poles, active = model
    y = coefficients * active[:, None, :]
    a = matmul(y, matmul(g, infinity_selector), transa="C")
    projected = matmul(y, a)
    rows = {}
    for name, operator, weight in (("M1", g, jnp.ones_like(poles)), ("M3", h, poles)):
        exact = matmul(projected, matmul(operator, projected), transa="C") / 2
        reconstructed = matmul(a, a * weight[:, :, None], transa="C") / 2
        norm = jnp.linalg.norm(exact, axis=(-2, -1))
        defect = jnp.linalg.norm(reconstructed - exact, axis=(-2, -1))
        rows[name] = jnp.where(norm > 0, defect / jnp.where(norm > 0, norm, 1),
                               jnp.where(defect == 0, 0, jnp.inf))
    return rows


def apply_shared_pole_zero_policy(model, *, gates):
    """Drop low Ritz values only within the resolved factor-weight budget.

    ``model=(b,poles2,active)`` has shapes [b,n,Kp], [b,Kp], [b,Kp].
    The weight is sum_j ||b_j||**2 (Ry**3), excluding carrier sentinels.
    A failed predicate must refuse before export; no pole is clipped.
    """
    b, poles, active = model
    drop = active & (poles <= gates["zero_ritz_policy"]["threshold"]["lambda_cutoff_ry2"])
    keep = active & ~drop
    weights = jnp.sum(jnp.abs(b) ** 2, axis=-2)
    total = jnp.sum(jnp.where(active, weights, 0), axis=-1)
    lost = jnp.sum(jnp.where(drop, weights, 0), axis=-1)
    fraction = lost / jnp.where(total > 0, total, 1)
    count = jnp.sum(keep, axis=-1, dtype=jnp.int64)
    finite = jnp.all(jnp.isfinite(b), axis=(-2, -1)) & jnp.all(jnp.isfinite(poles), axis=-1)
    admitted = (finite & (total > 0) & (count > 0)
                & (fraction <= gates["zero_ritz_policy"]["threshold"]["max_dropped_weight_fraction"]))
    return (jnp.where(keep[:, None, :], b, 0), jnp.where(keep, poles, 1), keep), {
        "zero_policy": admitted,
        "dropped_factor_weight_fraction": fraction,
        "dropped_count": jnp.sum(drop, axis=-1, dtype=jnp.int64),
        "factor_weight": total,
        "retained_rank": count,
    }


def sort_shared_pole_columns(model, *, matrix_sharding=None):
    """Sort joint active b/Lambda columns, retaining ties and safe padding.

    Returns the same three model arrays and the [b,Kp] permutation. Active
    entries form a prefix, inactive b is exactly zero, inactive Lambda is
    1 Ry**2. Stable sorting preserves equal-pole order. The factor is taken
    along its column axis where it lives: the round program sorts each
    parent on its own rank.
    """
    b, poles, active = model
    order = jnp.argsort(jnp.where(active, poles, jnp.inf), axis=-1, stable=True)
    from gw.shared_pole_pencil import _matrix_take_columns
    b = _matrix_take_columns(b, order, matrix_sharding)
    poles = jnp.take_along_axis(poles, order, axis=-1)
    active = jnp.take_along_axis(active, order, axis=-1)
    return (jnp.where(active[:, None, :], b, 0), jnp.where(active, poles, 1), active), order


def shared_pole_treatment_mask(poles, active, *, ceiling_ry):
    """Return the complete-column mask for a numerical frequency ceiling.

    The input model already has an ascending active prefix. The ceiling is a
    treatment policy, not a spectral bound; diagnostics therefore report only
    counts and extrema and assign no accuracy to the dropped contribution.
    Cross-sector callers must call this once for their common pole census and
    apply the returned mask to both endpoint factors.
    """
    ceiling_ry = float(ceiling_ry)
    if not np.isfinite(ceiling_ry) or ceiling_ry <= 0.0:
        raise ValueError("shared-pole treatment ceiling must be positive")
    keep = active & (poles <= ceiling_ry ** 2)
    omega = jnp.sqrt(jnp.where(active, poles, 0))
    retained_omega = jnp.sqrt(jnp.where(keep, poles, 0))
    return keep, {
        "active_prefix": ~jnp.any((~keep[:, :-1]) & keep[:, 1:], axis=-1),
        "retained_nonempty": jnp.any(keep, axis=-1),
        "dropped_count": jnp.sum(active & ~keep, axis=-1, dtype=jnp.int64),
        "original_count": jnp.sum(active, axis=-1, dtype=jnp.int64),
        "original_omega_max_ry": jnp.max(omega, axis=-1),
        "retained_omega_max_ry": jnp.max(retained_omega, axis=-1),
    }


def shared_pole_passivity(model, inverse_coulomb_sqrt, *, eta_ry, matmul, eigh, gates):
    """Test raw latent-model passivity on the authenticated Coulomb support.

    The authenticated inverse square root [b,n,n] and b [b,n,Kp] are
    face-tiled complex128. Lambda/active [b,Kp] are replicated. Returns
    device scalars; all ranks must evaluate them before the host refusal.
    """
    b, poles, active = model
    whitened = matmul(inverse_coulomb_sqrt, b)
    weight = jnp.where(active, 1 / (poles + eta_ry**2), 0)
    response = matmul(whitened * weight[:, None, :], whitened, transb="C")
    return _passivity_response_checks(response, eigh=eigh, gates=gates)


def shared_pole_operator_passivity(wc, inverse_coulomb_sqrt, *, matmul, eigh, gates):
    """Test an evaluated physical Wc(i eta) against the actual V support.

    ``wc`` may include the versioned symmetry realization; no factors of its
    averaged residues need to be materialized. Both square operands retain
    their caller's all-P faces. The caller owns evaluation at imaginary eta
    and memory admission, including the two square GEMMs and eigensolve.
    """
    inverse = inverse_coulomb_sqrt
    response = -matmul(matmul(inverse, wc), inverse)
    return _passivity_response_checks(response, eigh=eigh, gates=gates)


def _passivity_response_checks(response, *, eigh, gates):
    """Common 0 <= whitened response <= I and Hermiticity gate."""
    herm = hermitian_part(response)
    norm = jnp.linalg.norm(herm, axis=(-2, -1))
    anti = jnp.linalg.norm(response - _adjoint(response), axis=(-2, -1))
    anti = anti / jnp.where(norm > 0, 2 * norm, 1)
    values, _ = eigh(herm)
    minimum, maximum = values[:, 0], values[:, -1]
    limit = gates["passivity"]["threshold"].get("antihermitian_relative_max")
    passed = ((minimum >= gates["passivity"]["threshold"]["eigenvalue_min"])
              & (maximum <= gates["passivity"]["threshold"]["eigenvalue_max"])
              & (True if limit is None else anti <= limit)
              & jnp.all(jnp.isfinite(values), axis=-1) & jnp.isfinite(anti))
    return {"passivity": passed, "passivity_min": minimum,
            "passivity_max": maximum, "passivity_antihermitian_relative": anti}


def shared_pole_reciprocity(value, reference, *, gates):
    """Check W(s).T=W(s) where held input data have this extra symmetry.

    ``value`` and ``reference`` are matching [...,n,n] complex128 response
    faces (W in Ry or dW/ds in inverse Ry). Leading axes label parents and
    samples. Returns scalar arrays on those axes; no matrix is retained.
    For a real-residue Stieltjes response, transpose symmetry off the real
    s axis is equivalent to entrywise realness on the negative s axis.
    Scalar TRS alone does not impose this symmetry at a generic fixed q.
    This is a sampled model gate, not a certificate at every frequency.
    """
    threshold = gates["model_reciprocity"]["threshold"]
    def defect(a):
        return jnp.linalg.norm(a-jnp.swapaxes(a, -1, -2), axis=(-2, -1)) / jnp.maximum(
            jnp.linalg.norm(a, axis=(-2, -1)), jnp.finfo(jnp.float64).tiny)
    exact, measured = defect(reference), defect(value)
    applicable = exact <= threshold["reference_relative_max"]
    passed = (jnp.isfinite(exact) & jnp.isfinite(measured)
              & (~applicable | (measured <= threshold["model_relative_max"])))
    return {"passed": passed, "applicable": applicable,
            "reference_relative": exact, "model_relative": measured}
