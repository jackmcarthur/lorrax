r"""Direct verification at :math:`q=\Gamma`: :math:`\chi^0` and PPM :math:`\Sigma^c`.

Everything is at :math:`\mathbf q=0` so :math:`\mathbf k-\mathbf q=\mathbf k`.

:math:`\chi^0` (sum-over-states)
--------------------------------

.. math::

    \chi^0_{\mu\nu}(\mathbf q\!=\!0,\,\omega)
      = \frac{-2}{N_k}\sum_{\mathbf k}\sum_{v}^{\rm occ}\sum_{c}^{\rm unocc}
        \frac{\Delta E_{cv,\mathbf k}}
             {\Delta E_{cv,\mathbf k}^{\,2}-\omega^2}\;
        \rho^*_{cv,\mu}(\mathbf k)\;\rho_{cv,\nu}(\mathbf k)

where :math:`\rho_{cv,\mu}(\mathbf k)=\sum_s\psi_{c\mathbf k s}(\mu)\,
\psi^*_{v\mathbf k s}(\mu)` and
:math:`\Delta E_{cv}=E_{c\mathbf k}-E_{v\mathbf k}`.
Works for real or complex :math:`\omega` (pass :math:`i\omega_p`
to test the imaginary-axis screening).

:math:`\Sigma^c` branches (single :math:`n,\mathbf k`, :math:`q\!=\!0` contribution)
-------------------------------------------------------------------------------------

Pair product:
:math:`M_{n''n,\mu}=\sum_s\psi^*_{n''\mathbf k s}(\mu)\,\psi_{n\mathbf k s}(\mu)`.

.. math::

    \Sigma^{(+)}_{nn}\big|_{q=0}
      = -\frac{1}{N_k}\sum_{v}^{\rm occ}\sum_{\mu\nu}
        M^*_{vn,\mu}\,M_{vn,\nu}\;
        \frac{B_{\mu\nu}}{\omega_{\rm rel}+h_v+\Omega_{\mu\nu}}

.. math::

    \Sigma^{(-)}_{nn}\big|_{q=0}
      = -\frac{1}{N_k}\sum_{c}^{\rm unocc}\sum_{\mu\nu}
        M^*_{cn,\mu}\,M_{cn,\nu}\;
        \frac{B_{\mu\nu}}{\omega_{\rm rel}-E_c-\Omega_{\mu\nu}}

Static branch bounds (:math:`B/\Omega=-W^c(0)/2`):

.. math::

    \Sigma^{(+)}_{\rm static}\big|_{q=0}
      = +\frac{1}{2N_k}\sum_v^{\rm occ}\sum_{\mu\nu}
        |M|^2_{\mu\nu}\,W^c_{\mu\nu}

.. math::

    \Sigma^{(-)}_{\rm static}\big|_{q=0}
      = -\frac{1}{2N_k}\sum_c^{\rm unocc}\sum_{\mu\nu}
        |M|^2_{\mu\nu}\,W^c_{\mu\nu}

Elementwise :math:`\Omega/(h+\Omega)\le1` guarantees
:math:`|\Sigma^{(\pm)}|\le|\Sigma^{(\pm)}_{\rm static}|`.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

RYD2EV = 13.6056980659


def chi0_q0_direct(
    *,
    psi: jax.Array,       # (nk, nb, nspinor, nrmu)
    enk: jax.Array,       # (nk, nb)
    occ: jax.Array,       # (nk, nb)
    omega: complex = 0.0, # any complex frequency
    print_fn=print,
) -> jnp.ndarray:
    r"""Return :math:`\chi^0_{\mu\nu}(q\!=\!0,\omega)` as ``(nrmu, nrmu)``."""

    nk, nb, nspinor, nrmu = psi.shape
    omega2 = jnp.asarray(complex(omega) ** 2, dtype=jnp.complex128)

    occ_mask = occ > 0.5                                 # (nk, nb)
    v_mask = occ_mask                                    # (nk, nb)
    c_mask = ~occ_mask                                   # (nk, nb)

    # rho_{cv,mu}(k) = sum_s psi_c(k,s,mu) conj(psi_v(k,s,mu))
    # (nk, nb_c, nb_v, nrmu) via (nk,c,s,mu) x (nk,v,s,mu)*
    rho = jnp.einsum('kcsm,kvsm->kcvm', psi, jnp.conj(psi))   # (nk,nb,nb,nrmu)

    # dE(k,c,v) = E_c(k) - E_v(k)
    dE = enk[:, :, None] - enk[:, None, :]                      # (nk, nb, nb)

    # weight = dE / (dE^2 - omega^2),  masked to c=unocc, v=occ
    denom = dE * dE - omega2
    w = jnp.where(jnp.abs(denom) > 1e-30, dE / denom, 0.0)
    mask_cv = c_mask[:, :, None] & v_mask[:, None, :]            # (nk, nb, nb)
    w = jnp.where(mask_cv, w, 0.0).astype(jnp.complex128)

    # chi0(mu,nu) = (-2/Nk) sum_{k,c,v} w * rho* x rho
    chi0 = jnp.einsum('kcv,kcvm,kcvn->mn', w, jnp.conj(rho), rho)
    chi0 = (-2.0 / nk) * chi0

    if print_fn is not None:
        print_fn(f"  [chi0_q0_direct] ω={omega}, max|χ⁰|={float(jnp.max(jnp.abs(chi0))):.6e}")
    return chi0


def sigma_c_q0_direct(
    *,
    psi_k: jax.Array,       # (nb, nspinor, nrmu)  wavefunctions at ik
    enk_k: jax.Array,       # (nb,)                eigenvalues at ik
    occ_k: jax.Array,       # (nb,)                occupations at ik
    B_q0: jax.Array,        # (nrmu, nrmu)         PPM amplitude at q=0
    Omega_q0: jax.Array,    # (nrmu, nrmu)         PPM frequency at q=0
    mask_q0: jax.Array,     # (nrmu, nrmu)         valid PPM elements
    Wc0_q0: jax.Array,      # (nrmu, nrmu)         W^c(q=0,0)
    efermi: float,
    iband: int,
    omega_rel_ry: float = 0.0,
    nk_tot: int = 1,        # for the 1/Nk prefactor
    print_fn=print,
) -> dict:
    r"""Return :math:`\Sigma^{(\pm)}_{nn}|_{q=0}` and static bounds."""

    nb, nspinor, nrmu = psi_k.shape
    ef = float(efermi)
    omega_rel = float(omega_rel_ry)

    occ_mask = occ_k > 0.5                              # (nb,)
    psi_n = psi_k[iband]                                # (nspinor, nrmu)

    # M_{n'',mu} = sum_s conj(psi_{n''}(s,mu)) psi_n(s,mu)
    M = jnp.einsum('nsm,sm->nm', jnp.conj(psi_k), psi_n)   # (nb, nrmu)

    # |M|^2_{n'',mu,nu} = conj(M_mu) M_nu
    MM = jnp.conj(M[:, :, None]) * M[:, None, :]             # (nb, nrmu, nrmu)

    # energy axes
    h_v = ef - enk_k                                    # (nb,)  >=0 for occupied
    E_c = enk_k - ef                                    # (nb,)  >=0 for unoccupied

    # --- dynamic branches ---
    # Sigma^(+): denom = omega_rel + h_v[n''] + Omega[mu,nu]
    denom_plus = omega_rel + h_v[:, None, None] + Omega_q0[None, :, :]     # (nb, nrmu, nrmu)
    safe_plus = mask_q0[None, :, :] & (jnp.abs(denom_plus) > 1e-30)
    integrand_plus = jnp.where(safe_plus, MM * B_q0[None, :, :] / denom_plus, 0.0)
    # sum over occupied n'', mu, nu
    sigma_plus_per_band = jnp.sum(integrand_plus, axis=(1, 2))             # (nb,)
    sigma_plus = jnp.sum(jnp.where(occ_mask, sigma_plus_per_band, 0.0))
    sigma_plus = (-1.0 / nk_tot) * sigma_plus

    # Sigma^(-): denom = omega_rel - E_c[n''] - Omega[mu,nu]
    denom_minus = omega_rel - E_c[:, None, None] - Omega_q0[None, :, :]    # (nb, nrmu, nrmu)
    safe_minus = mask_q0[None, :, :] & (jnp.abs(denom_minus) > 1e-30)
    integrand_minus = jnp.where(safe_minus, MM * B_q0[None, :, :] / denom_minus, 0.0)
    sigma_minus_per_band = jnp.sum(integrand_minus, axis=(1, 2))
    sigma_minus = jnp.sum(jnp.where(~occ_mask, sigma_minus_per_band, 0.0))
    sigma_minus = (-1.0 / nk_tot) * sigma_minus

    # --- static bounds:  B/Omega = -Wc0/2 ---
    Wc0_contrib = jnp.sum(MM * Wc0_q0[None, :, :], axis=(1, 2))           # (nb,)
    sigma_plus_static  = (+1.0 / (2.0 * nk_tot)) * jnp.sum(jnp.where(occ_mask, Wc0_contrib, 0.0))
    sigma_minus_static = (-1.0 / (2.0 * nk_tot)) * jnp.sum(jnp.where(~occ_mask, Wc0_contrib, 0.0))

    sigma_c = sigma_plus + sigma_minus
    sigma_c_static = sigma_plus_static + sigma_minus_static

    E_nk = float(enk_k[iband])
    is_occ = "occupied" if bool(occ_mask[iband]) else "unoccupied"

    if print_fn is not None:
        ev = RYD2EV
        sp  = complex(sigma_plus);          sm  = complex(sigma_minus)
        sps = complex(sigma_plus_static);   sms = complex(sigma_minus_static)
        sc  = complex(sigma_c);             scs = complex(sigma_c_static)
        print_fn("")
        print_fn("=" * 72)
        print_fn("  DIRECT Σ^c(q=0) VERIFICATION")
        print_fn("=" * 72)
        print_fn(f"  iband={iband}, E_nk={E_nk*ev:.4f} eV ({is_occ})")
        print_fn(f"  E_F={ef*ev:.4f} eV, ω_rel={omega_rel*ev:.4f} eV, Nk={nk_tot}")
        print_fn("")
        print_fn(f"  {'':24s}  {'Re (eV)':>12s}  {'Im (eV)':>12s}")
        print_fn(f"  {'Σ^(+) direct':24s}  {sp.real*ev:12.4f}  {sp.imag*ev:12.4f}")
        print_fn(f"  {'Σ^(+) static bound':24s}  {sps.real*ev:12.4f}  {sps.imag*ev:12.4f}")
        print_fn(f"  {'Σ^(-) direct':24s}  {sm.real*ev:12.4f}  {sm.imag*ev:12.4f}")
        print_fn(f"  {'Σ^(-) static bound':24s}  {sms.real*ev:12.4f}  {sms.imag*ev:12.4f}")
        print_fn(f"  {'Σ^c total':24s}  {sc.real*ev:12.4f}  {sc.imag*ev:12.4f}")
        print_fn(f"  {'Σ^c static total':24s}  {scs.real*ev:12.4f}  {scs.imag*ev:12.4f}")
        print_fn("")
        ok_p = abs(sp) <= abs(sps) * (1.0 + 1e-6)
        ok_m = abs(sm) <= abs(sms) * (1.0 + 1e-6)
        print_fn(f"  |Σ^(+)| ≤ |Σ^(+)_static|?  {ok_p}  "
                 f"({abs(sp)*ev:.4f} vs {abs(sps)*ev:.4f})")
        print_fn(f"  |Σ^(-)| ≤ |Σ^(-)_static|?  {ok_m}  "
                 f"({abs(sm)*ev:.4f} vs {abs(sms)*ev:.4f})")
        if abs(sps) > 1e-14:
            print_fn(f"  Σ^(+) damping: {sp.real / sps.real:.6f}")
        if abs(sms) > 1e-14:
            print_fn(f"  Σ^(-) damping: {sm.real / sms.real:.6f}")
        print_fn("=" * 72)

    return dict(
        sigma_plus=complex(sigma_plus),
        sigma_minus=complex(sigma_minus),
        sigma_plus_static=complex(sigma_plus_static),
        sigma_minus_static=complex(sigma_minus_static),
        sigma_c=complex(sigma_c),
        sigma_c_static=complex(sigma_c_static),
    )