"""Finite physical spectral-window kernels, unnormalized, in Ry units.

The caller assigns spectral intervals, not pole centers, to product windows.
EVAL owns width validation and the Lorentzian/dispersive convention. This
module supplies their finite-interval integrals, not a new Sigma transport.
"""
import numpy as np
from scipy.special import exp1

W_CERT_RY = 149.7645 / 13.605693122994


def _scaled_e1(z):
    """Return exp(z) E1(z) on its principal sheet without overflow."""
    z = np.asarray(z, complex)
    large = np.abs(z) > 80
    safe = np.where(large, 1+0j, z)
    direct = np.exp(safe)*exp1(safe)
    denominator = np.where(large, z, 1+0j)
    term = 1/denominator
    asymptotic = term.copy()
    for k in range(1, 33):
        term *= -k/denominator
        asymptotic += term
    return np.where(large, asymptotic, direct)


def _fourier_resolvent(c, t, lower, upper):
    """Integral exp(-iwt)/(w-c) dw with continuous E1 endpoint branches.

    c is off the real axis [Ry]; t can be complex [inverse Ry]. Bounds are
    finite nonnegative real energies [Ry]. Arrays broadcast without sharding.
    """
    s, c = np.broadcast_arrays(1j*np.asarray(t, complex), np.asarray(c, complex))
    zero = s == 0
    safe_s = np.where(zero, 1+0j, s)
    z0, z1 = safe_s*(lower-c), safe_s*(upper-c)
    value = (np.exp(-safe_s*lower)*_scaled_e1(z0)
             - np.exp(-safe_s*upper)*_scaled_e1(z1))
    # E1 jumps -2*pi*i when crossing the negative real axis upwards.
    # Continue along the straight image of [lower,upper], not independently
    # on the principal sheets of the endpoints.
    dy = z1.imag-z0.imag
    fraction = -z0.imag/np.where(dy == 0, 1., dy)
    crossing = ((fraction > 0) & (fraction < 1)
                & ((z0+(z1-z0)*fraction).real < 0))
    correction = np.where(crossing, -2j*np.pi*np.sign(dy), 0j)
    # Only evaluate this exponential where a crossing occurs. Its magnitude
    # is then bounded by the physical endpoint exponentials.
    value += np.exp(np.where(crossing, -safe_s*c, 0j))*correction
    mass = np.log(upper-c)-np.log(lower-c)
    return np.where(zero, mass, value)


def fourier_components(t, omega, lower=0., upper=W_CERT_RY, *,
                       allow_anticausal=False):
    """Return finite-window Fourier integrals (L, minus D).

    Parameters
    ----------
    t : complex array
        Complex time nodes [inverse Ry], broadcast against omega.
    omega : complex array
        Positive-center spectral poles [Ry]. Width is -Im(omega).
    lower, upper : float
        Spectral integration bounds [Ry], inside [0,W_CERT_RY]. No mass
        renormalization. Real atoms at the lower edge belong to the low
        interval by the carrier's explicit selection mask.
    allow_anticausal : bool
        Explicit opt-in to the EVAL signed-width diagnostic, never a bypass
        of gamma+eta<=0 refusal.

    Returns
    -------
    lorentz, minus_dispersive : complex array
        Dimensionless integrals; rho_R=H*L-A*D for adjoint-mirrored residues.
        The dispersive result is NaN for exactly real poles (PV route not
        implemented); Hermitian atoms use only lorentz.
    """
    from residue_kernel import width_diagnostic
    width_diagnostic(omega, allow_anticausal=allow_anticausal)
    if not (0 <= lower < upper <= W_CERT_RY):
        raise ValueError('Require a finite spectral interval inside W_cert')
    t, omega = np.broadcast_arrays(np.asarray(t, complex), np.asarray(omega, complex))
    if not np.all(np.isfinite(t)):
        raise ValueError('Nonfinite time')
    a, gamma = omega.real, -omega.imag
    real = gamma == 0
    g = np.where(real, 1., np.abs(gamma))
    cs = (a+1j*g, a-1j*g, -a+1j*g, -a-1j*g)
    h1, h2, h3, h4 = [_fourier_resolvent(c, t, lower, upper) for c in cs]
    lorentz = np.sign(gamma)*(h1-h2-h3+h4)/(2j*np.pi)
    # Nearly coincident mirror poles lose relative precision in the four-E1
    # subtraction. Expand the odd centered difference, not the physical
    # spectrum or its width: I(c+a)-I(c-a)=2*a*I_2(c)+2*a**3*I_4(c)+O(a**5),
    # where I_m=int exp(-i*w*t)/(w-c)**m dw. Endpoint integration by parts
    # gives (m-1)I_m=f(L)/(L-c)**(m-1)-f(U)/(U-c)**(m-1)-i*t*I_(m-1).
    small = (~real) & (a/g < 1e-4) & (a*np.abs(t) < 1e-4)
    if np.any(small):
        tt, aa, gg = t[small], a[small], g[small]
        terms = []
        for c in (1j*gg, -1j*gg):
            value = _fourier_resolvent(c, tt, lower, upper)
            odd = np.zeros_like(value)
            for order in range(2, 5):
                value = (np.exp(-1j*lower*tt)/(lower-c)**(order-1)
                         - np.exp(-1j*upper*tt)/(upper-c)**(order-1)
                         - 1j*tt*value)/(order-1)
                if order % 2 == 0:
                    odd += aa**(order-1)*value
            terms.append(odd)
        lorentz = np.array(lorentz, copy=True)
        lorentz[small] = np.sign(gamma[small])*(terms[0]-terms[1])/(1j*np.pi)
    dispersive = -(h1+h2+h3+h4)/(2*np.pi)
    atom = np.where((a > lower) & (a <= upper), np.exp(-1j*a*t), 0j)
    return np.where(real, atom, lorentz), np.where(real, np.nan+0j, dispersive)


def mass(omega, lower=0., upper=W_CERT_RY):
    """Unnormalized finite Lorentzian mass, signed for negative widths."""
    a, gamma = np.real(omega), -np.imag(omega)
    g = np.where(gamma == 0, 1., np.abs(gamma))
    def primitive(w):
        return (np.arctan((w-a)/g)-np.arctan((w+a)/g))/np.pi
    value = np.sign(gamma)*(primitive(upper)-primitive(lower))
    return np.where(gamma == 0, np.asarray((a > lower) & (a <= upper), float), value)


def stieltjes_components(z, omega, lower=0., upper=W_CERT_RY, *, xp=np):
    """Finite-window (L, minus D) Stieltjes kernels for EVAL transport.

    z includes the signed fermion eta [Ry]; omega is a stable or explicitly
    approved signed-width pole [Ry]. Host callers must first invoke EVAL's
    width_diagnostic. Supports NumPy and JAX arrays with broadcasting. The
    exactly-real dispersive PV case is deliberately NaN, as in Fourier.
    """
    z, omega = xp.broadcast_arrays(xp.asarray(z, dtype=xp.complex128),
                                 xp.asarray(omega, dtype=xp.complex128))
    a, gamma = xp.real(omega), -xp.imag(omega)
    real = gamma == 0
    g = xp.where(real, 1., xp.abs(gamma))
    def h(c):
        delta = z-c
        close = xp.abs(delta) < 1e-7*xp.abs(c)
        # Endpoint logs follow the real integration interval continuously:
        # H=[log(U-c)-log(L-c)-log(z-U)+log(z-L)]/(z-c).
        numerator = (xp.log(upper-c)-xp.log(lower-c)
                     -xp.log(z-upper)+xp.log(z-lower))
        regular = numerator/xp.where(close, 1+0j, delta)
        d1 = 1/(c-lower)-1/(c-upper)
        d2 = (-1/(c-lower)**2+1/(c-upper)**2)/2
        d3 = (1/(c-lower)**3-1/(c-upper)**3)/3
        return xp.where(close, d1+delta*d2+delta*delta*d3, regular)
    h1,h2,h3,h4 = [h(c) for c in (a+1j*g,a-1j*g,-a+1j*g,-a-1j*g)]
    lorentz = xp.sign(gamma)*(h1-h2-h3+h4)/(2j*xp.pi)
    dispersive = -(h1+h2+h3+h4)/(2*xp.pi)
    atom = xp.where((a > lower) & (a <= upper), 1/(z-a), 0j)
    return xp.where(real, atom, lorentz), xp.where(real, xp.nan+0j, dispersive)
