# Hybertsen-Louie generalized plasmon pole

The Hybertsen-Louie generalized plasmon-pole model replaces each matrix
element of the inverse dielectric response by one symmetric pole pair. Its two
parameters are fixed by static screening and a frequency-moment sum rule.

## 1. Single-pole ansatz

For one matrix element, write

$$
\epsilon^{-1}_{GG'}(q,\omega)
=\delta_{GG'}+
\frac{\Omega^2_{GG'}(q)}
{\omega^2-\widetilde\omega^2_{GG'}(q)+i0^+}.
$$

The ansatz has the correct high-frequency limit and satisfies
\(\epsilon^{-1}(-\omega)=\epsilon^{-1}(\omega)^*\). For positive frequency,

$$
-\operatorname{Im}\epsilon^{-1}(\omega)
=\frac{\pi\Omega^2}{2\widetilde\omega}
\delta(\omega-\widetilde\omega).
$$

The positive-frequency moments are therefore

$$
M_1
=\frac{2}{\pi}\int_0^\infty
\omega[-\operatorname{Im}\epsilon^{-1}(\omega)]\,d\omega
=\Omega^2,
$$

$$
M_{-1}
=\frac{2}{\pi}\int_0^\infty
\frac{-\operatorname{Im}\epsilon^{-1}(\omega)}{\omega}\,d\omega
=\frac{\Omega^2}{\widetilde\omega^2}.
$$

Kramers-Kronig gives the same inverse moment from the static response:

$$
\epsilon^{-1}_{GG'}(q,0)
=\delta_{GG'}-\frac{\Omega^2_{GG'}(q)}
{\widetilde\omega^2_{GG'}(q)}
=\delta_{GG'}-M_{-1}.
$$

Thus static screening and \(M_1\) determine the model:

$$
\Omega^2=M_1^{\mathrm{sum\ rule}},
\qquad
\widetilde\omega^2_{GG'}(q)
=\frac{\Omega^2_{GG'}(q)}
{\delta_{GG'}-\epsilon^{-1}_{GG'}(q,0)}.
$$

For a dielectric matrix these relations are applied in the precise
elementwise convention of the implementation, including its Coulomb factors
and Hermiticity handling. The scalar derivation states what is matched; it
does not license treating an arbitrary off-diagonal element as a positive
spectral density.

## 2. Meaning and limitation

The model exactly reproduces:

1. the RPA static inverse dielectric matrix; and
2. the first frequency moment supplied by the f-sum rule.

Equivalently, it reproduces \(M_1\) and the \(M_{-1}\) fixed by the static
limit. It does not fit the detailed loss spectrum. Once inserted into
\(W(\omega)=\epsilon^{-1}(\omega)v\), the remaining frequency integral in
Sigma is analytic.

This distinguishes HL from the Godby-Needs model. GN fixes one pole from two
explicit samples, \(W(0)\) and \(W(i\omega_p)\); HL fixes it from one static
sample and one moment. MPA instead fits several complex poles from a
double-parallel sample set.

## References

- Hybertsen and Louie, *Phys. Rev. B* **34**, 5390 (1986).
- Hybertsen and Louie, *Phys. Rev. B* **37**, 2733 (1988).
- Soininen *et al.*, *J. Phys.: Condens. Matter* **15**, 2573 (2003).
