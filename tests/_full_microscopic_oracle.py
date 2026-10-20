"""Literal-index tiny oracles for the full-port coherent snapshot contract.

These deliberately do not use a production contraction, pole reduction or
quadrature. Energies share one arbitrary unit; densities are dimensionless.
"""
import numpy as np


def dagger(value):
    return np.asarray(value).conj().swapaxes(-1, -2)


def ordered_pr(psi, energy, density, partner, sites, gamma, spin_weight=1.):
    """Full spatial PR[site,mu,nu] by the explicit ordered transition sum.

    psi[k,band,spin,mu], energy[k,band], density[k,band,band]. The energy
    indices b,d are RIGHT of rho: Pg-Pl cancels its quadratic terms.
    """
    nk, nb, ns, nmu = psi.shape
    out = np.zeros((len(sites), nmu, nmu), complex)
    eye = np.eye(nb)
    for k in range(nk):
        kq = int(partner[k])
        for b in range(nb):
            for d in range(nb):
                left = np.zeros(nmu, complex)
                right = np.zeros(nmu, complex)
                for mu in range(nmu):
                    for s in range(ns):
                        right[mu] += psi[k, b, s, mu] * psi[kq, d, s, mu].conjugate()
                        for a in range(nb):
                            for c in range(nb):
                                weight = (eye[a, b] * density[kq, c, d].conjugate()
                                          - density[k, a, b] * eye[c, d])
                                left[mu] += weight * psi[k, a, s, mu] * psi[kq, c, s, mu].conjugate()
                residue = np.outer(left, right.conjugate())
                for iz, z in enumerate(sites):
                    out[iz] += residue / (z - energy[k, b] + energy[kq, d] + 1j * gamma)
    return out * spin_weight / nk


def ordered_pr_time(psi, energy, density, partner, times, spin_weight=1.):
    """Time-node PR amplitudes by explicit b,d transition residues, no FFT."""
    nk, nb, ns, nmu = psi.shape
    out = np.zeros((len(times), nmu, nmu), complex)
    eye = np.eye(nb)
    for k in range(nk):
        kq = int(partner[k])
        for b in range(nb):
            for d in range(nb):
                left, right = np.zeros(nmu, complex), np.zeros(nmu, complex)
                for mu in range(nmu):
                    for s in range(ns):
                        right[mu] += psi[k, b, s, mu] * psi[kq, d, s, mu].conjugate()
                        for a in range(nb):
                            for c in range(nb):
                                weight = (eye[a, b]*density[kq, c, d].conjugate()
                                          - density[k, a, b]*eye[c, d])
                                left[mu] += weight*psi[k, a, s, mu]*psi[kq, c, s, mu].conjugate()
                for it, t in enumerate(times):
                    out[it] += np.outer(left, right.conjugate())*np.exp(
                        -(energy[k, b]-energy[kq, d])*t)
    return out*spin_weight/nk


def density_vertices(psi, partner, external):
    """Physical spatial Sigma vertices conj(psi_external)*psi_internal."""
    nk, nb, ns, nmu = psi.shape
    out = np.zeros((nk, len(external), nb, nmu), complex)
    for k in range(nk):
        for a, band in enumerate(external):
            for n in range(nb):
                for mu in range(nmu):
                    for s in range(ns):
                        out[k, a, n, mu] += (psi[k, band, s, mu].conjugate()
                                                * psi[partner[k], n, s, mu])
    return out


def retarded_reference_pairs(vertices, energy, density, external, poles, residues, eta):
    """Literal full-rho Rp/Rh row+column shells; both are RETARDED.

    Pole branches are the accepted T=0 frequency reference: greater R at
    Re p>0 and lesser -R at Re p<0. No positivity/Hermiticity of R assumed.
    energy/density are already at the internal k-q rows. Returned rp,rh
    have [row_or_column,k,a,b]; all poles must lie below evaluation eta.
    """
    nk, na, ni, nmu = vertices.shape
    if eta <= np.max(np.asarray(poles).imag, initial=-np.inf):
        raise ValueError('oracle Laplace evaluation height does not exceed growth')
    rp = np.zeros((2, nk, na, na), complex)
    rh = np.zeros_like(rp)
    eye = np.eye(ni)
    zero_tolerance = 64*np.finfo(float).eps*max(np.max(abs(poles), initial=0.), 1.)
    for k in range(nk):
        for a in range(na):
            for b in range(na):
                for n in range(ni):
                    for m in range(ni):
                        for p, residue in zip(poles, residues):
                            for mu in range(nmu):
                                for nu in range(nmu):
                                    raw = (vertices[k, a, m, mu] * residue[mu, nu]
                                           * vertices[k, b, n, nu].conjugate())
                                    zero = abs(p.real) <= zero_tolerance
                                    hw = .5 if zero else float(p.real < 0)
                                    pw = .5 if zero else float(p.real > 0)
                                    hole = -hw*raw*density[k, m, n]
                                    particle = pw*raw*(eye[m, n]-density[k, m, n])
                                    for shell in range(2):
                                        z = external[k, a if shell == 0 else b]
                                        denominator = z - energy[k, n] - p + 1j * eta
                                        rp[shell, k, a, b] += particle / denominator
                                        rh[shell, k, a, b] += hole / denominator
    return rp, rh


def branch_coefficients(rp, rh):
    """Full complex retarded columns -> MH, Gamma, D (without shell average)."""
    total = rp[1] + rh[1]
    return dict(m_h=.5 * (total + dagger(total)),
                gamma=1j * (total - dagger(total)),
                injection=1j * (rh[1] - dagger(rh[1])))


def advanced_hole_compatibility_pair(retarded_hole):
    """Legacy val[0]^H must recover exactly the retarded HOLE COLUMN."""
    return np.stack((dagger(retarded_hole[1]), dagger(retarded_hole[0])))
