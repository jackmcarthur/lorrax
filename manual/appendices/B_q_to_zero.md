# Appendix B — The q → 0 head of W

At $\mathbf q \to 0$, $\mathbf G = \mathbf G' = 0$, the screened interaction's
head is the product of a $1/q^2$ divergence and a $q^2$ zero. The bare side is the
mini-BZ average $\bar v_0$ of §6.2. The polarizability side vanishes as
$\chi_{00}(\mathbf q) = q^2 s(\hat{\mathbf q}) + O(q^4)$ with the coefficient
built from transition dipoles: LORRAX assembles the s-tensor from the
`dipole.h5` matrix elements (§9.1) and angle-averages over the mini-BZ
(`wcoul0_source = s_tensor`, the default; `epshead` imports a static BGW
`epsmat` head for debugging).

Static head: $W_{00}(0) = \bar v_0 / \epsilon_{\rm head}$ with
$\epsilon_{\rm head}$ from the averaged s-tensor. Dynamic head, needed by the
plasmon-pole fit: the two head samples $w_1 = W^c_{00}(0)$ and
$w_2 = W^c_{00}(i\omega_p)$ fix a scalar Godby–Needs pole,

$$
\Omega_h^2 = \frac{-w_2\, z^2}{w_1 - w_2}, \qquad
W^c_{\rm head}(\omega) = \frac{B_h}{\omega^2 - \Omega_h^2}, \qquad
B_h = -w_1\,\Omega_h^2,
$$

(residue $R_h = B_h/2\Omega_h$; note this normalization differs from the body
fit's §7.4 convention, where $B$ is itself the residue). Because the $q\to0$
matrix elements are band-diagonal, the head enters Σ analytically per state,

$$
\Sigma^c_{n,\rm head}(\omega) = \frac{R_h}{V_{\rm cell} N_k}
\left[\frac{f_n}{\omega-\varepsilon_n+\Omega_h - i\eta}
+ \frac{1-f_n}{\omega-\varepsilon_n-\Omega_h + i\eta}\right],
$$

whose $\omega\to\varepsilon_n$ limit reproduces the static COHSEX head
$\mp W^c_{00}(0)/2V_{\rm cell}N_k$. Overrides for parity work: `vhead`,
`whead_0freq`, `whead_imfreq` (the last is currently required for bispinor
GN-PPM, §8.4), and `ppm_head_omega_h_ry` to pin $\Omega_h$ directly; under
`ppm_model = hl` the head pole is instead set analytically from
$\Omega_h^2 = \omega_p^2/(1-\epsilon_{\rm head}^{-1})$.

In 2D the truncated kernel diverges only as $1/q$ and the same construction
applies with the slab kernel's angular structure. Wing (head–body) corrections
follow the BerkeleyGW treatment.
<!-- TODO(verify): wing handling detail vs fixwings.f90 conventions before
     publication. -->
