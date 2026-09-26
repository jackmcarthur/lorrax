# Hybertsen–Louie generalized plasmon pole

`compute_mode = hl_ppm` replaces each matrix element of the correlation part of
the screened interaction by one symmetric pole pair. Its two parameters are
fixed by the static response and by the first frequency moment, which LORRAX
reads from one real-axis sample above every transition.

## 1. Single-pole ansatz

For one matrix element,

$$
\epsilon^{-1}_{GG'}(q,\omega)
=\delta_{GG'}+
\frac{\Omega^2_{GG'}(q)}
{\omega^2-\widetilde\omega^2_{GG'}(q)+i0^+}.
$$

The ansatz decays as $\omega^{-2}$ and satisfies
$\epsilon^{-1}(-\omega)=\epsilon^{-1}(\omega)^*$. For $\omega>0$,

$$
-\operatorname{Im}\epsilon^{-1}(\omega)
=\frac{\pi\Omega^2}{2\widetilde\omega}
\delta(\omega-\widetilde\omega),
$$

so the positive-frequency moments are

$$
M_1=\frac{2}{\pi}\int_0^\infty
\omega\,[-\operatorname{Im}\epsilon^{-1}(\omega)]\,d\omega=\Omega^2,
\qquad
M_{-1}=\frac{2}{\pi}\int_0^\infty
\frac{-\operatorname{Im}\epsilon^{-1}(\omega)}{\omega}\,d\omega
=\frac{\Omega^2}{\widetilde\omega^2}.
$$

Kramers–Kronig ties $M_{-1}$ to the static response,
$\epsilon^{-1}_{GG'}(q,0)=\delta_{GG'}-M_{-1}$, so

$$
\Omega^2=M_1^{\mathrm{sum\ rule}},
\qquad
\widetilde\omega^2_{GG'}(q)
=\frac{\Omega^2_{GG'}(q)}
{\delta_{GG'}-\epsilon^{-1}_{GG'}(q,0)}.
$$

The model reproduces the static inverse dielectric matrix and the $f$-sum
moment exactly and nothing else: it does not fit the loss spectrum. Once
$W(\omega)=\epsilon^{-1}(\omega)v$ has this form, the frequency integral in
$\Sigma$ is analytic. Godby–Needs instead fixes the pole from $W(0)$ and
$W(i\omega_p)$; [MPA](THEORY_mpa_implementation.md) fits several complex poles
to a double-parallel sample set.

## 2. How LORRAX fixes the two parameters

**Body.** Write the correlation part as
$W_c(z)=2\widetilde\omega B/(z^2-\widetilde\omega^2)$. Two samples,
$W_c(0)$ and $W_c(z_p)$, fix it elementwise on the $(q,\mu,\nu)$ tensor:

$$
\widetilde\omega^2=-\frac{z_p^2\,W_c(z_p)}{W_c(0)-W_c(z_p)},
\qquad
B=-\tfrac12\,W_c(0)\,\widetilde\omega .
$$

GN-PPM and HL-PPM share this algebra
(`gw.minimax_screening.fit_gn_ppm_from_wc_pair`) and differ only in the probe:
GN takes $z_p=i\omega_p$, HL takes the real $z_p=\omega_p$ = `ppm_omega_p`
(Ry). At a real probe above the spectrum,
$\omega_p^2W_c(\omega_p)\to2\widetilde\omega B$ as $\omega_p\to\infty$, the
first-moment coefficient; the HL body therefore reads $M_1$ from the
band-summed response itself rather than from the electron density. An element
whose $\operatorname{Re}\widetilde\omega^2$ is not positive and finite takes
`ppm_invalid_mode` (`static_limit` by default: drop the pole and add the static
COHSEX term). HL keeps every fitted pole; the GN-only 0.2 % tail coarsening is
never applied.

**Head.** The $q\to0$ head uses the bulk plasma frequency,
$\omega_p^2=16\pi n_e/V_{\rm cell}$ (Ry²), and the $f$-sum form
$\widetilde\omega_h^2=\omega_p^2/[1-\epsilon^{-1}_{00}(0)]$
(`gw.head_correction.fit_head_hl_analytic`), as BerkeleyGW does.

**Cost.** Two screened interactions ($\omega=0$ and $\omega_p$), elementwise
algebra, and the shared one-pole route through
[the MPA Σ consumer](THEORY_mpa_implementation.md#mpa-sigma)
(`gw.ppm_sigma` writes the fit as a one-pole store).

## 3. What it refuses

| condition | why |
|---|---|
| `ppm_omega_p` at or below the largest transition energy | the real-axis $\chi_0$ rule (`build_real_quadrature`) needs $\Omega>x_{\max}$; below it the probe sits in the spectrum |
| metallic WFN occupations | GN/HL-PPM split bands by a 0/1 step (`gw_config.validate_material_inputs`; [input reference](../input_reference.md)) |
| `screening_diagrams = w_bse` or `w_rpa_resolvent` | a real-axis resolvent $(z-H)^{-1}$ needs a broadening policy the ladder does not have |

On a time-reversal-broken deck HL runs the single-residue fit: $W_c$ at a real
frequency outside the spectrum is Hermitian, so one sample cannot separate the
odd residue ([non-Hermitian GN-PPM](../dev/notes/DERIVATION_gnppm_nonhermitian.md)
§6).

## References

- Hybertsen and Louie, *Phys. Rev. B* **34**, 5390 (1986).
- Hybertsen and Louie, *Phys. Rev. B* **37**, 2733 (1988).
- Soininen *et al.*, *J. Phys.: Condens. Matter* **15**, 2573 (2003).
