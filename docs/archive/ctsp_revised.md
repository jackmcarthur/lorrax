# Complex-Time Quadrature for $GW$ Frequency Convolutions

---

The $GW$ self-energy is the frequency convolution
$$
\Sigma(\omega)_{rr'} = \frac{i}{2\pi}\int_{-\infty}^{+\infty} d\omega'\; G_{rr'}(\omega - \omega')\, W_{rr'}(\omega')\,.
$$
The one-particle Green's function has simple poles at the Kohn–Sham eigenvalues:
$$
\begin{aligned}
G_{rr'}(\omega)
&= \sum_n \frac{\psi_{rn}\psi^*_{r'n}}{\omega - E_n + i\eta\,\operatorname{sgn}(E_n - \mu)}\,,\\[4pt]
G_{rr'}(t)
&= -i\sum_v \psi_{rv}\psi^*_{r'v}\,e^{-iE_v t}\,\theta(t)
\;+\; i\sum_c \psi_{rc}\psi^*_{r'c}\,e^{-iE_c t}\,\theta(-t)\,.
\end{aligned}
$$
The correlation part of the screened interaction, $W_c = W - v = vXv$, inherits the pole structure of the dressed polarizability $X$. $X$ has an exact representation in terms of $N_vN_c$ plasmon poles $R_p$ at frequencies $\pm\Omega_p$:
$$
X^{\mathrm{MP}}_{rr'}(z) = \sum_{p=1}^{n_p}\frac{2\Omega_p R_p}{z^2 - \Omega_p^2}\,,\qquad
\operatorname{Re}\Omega_p > 0\,,\;\;\operatorname{Im}\Omega_p \leq 0\,.
$$
It is possible to replace these many plasmons with a multi-plasmon pole model; full-frequency accuracy typically requires $n_p \sim \mathcal{O}(10)$ complex poles per $r,r'$ element. With residue matrices $B^p_{rr'}$ and pole frequencies $\omega_p$:
$$
W_{c,rr'}(t) = -i\sum_p B^p_{rr'}\bigl[e^{-i\omega_p t}\,\theta(t) + e^{i\omega_p t}\,\theta(-t)\bigr]\,.
$$

Inserting these spectral representations into the convolution yields
$$
\Sigma(\omega)_{rr'} = \underbrace{\sum_{p,v}\frac{B^p_{rr'}\,\psi_{rv}\psi^*_{r'v}}{\omega - E_v + \omega_p + i\eta}}_{\displaystyle\Sigma^{(+)}} \;+\; \underbrace{\sum_{p,c}\frac{B^p_{rr'}\,\psi_{rc}\psi^*_{r'c}}{\omega - E_c - \omega_p - i\eta}}_{\displaystyle\Sigma^{(-)}}\,.
$$
Direct evaluation costs $\mathcal{O}(N_{\mathrm{bands}}\cdot N_{\mathrm{plasmon}}\cdot N_r^2) = \mathcal{O}(N^4)$. Since $G$ and $W_c$ each admit $\mathcal{O}(N^3)$ spectral representations, the $N^4$ bottleneck is because we need to sum over plasmon-wavefunction pairs simultaneously, as they are coupled by the energy denominator $1/(\omega - E_n \pm \omega_p + i\eta)$. We will show how an integral transform scheme replaces this denominator with a product of factors depending on $E_n$ and $\omega_p$ independently, allowing $\mathcal{O}(N^3)$ evaluation.

We can motivate this approach with the convolution theorem. For $\Sigma^{(+)}$, the step-function structure of $G$ and $W$ restricts integration to $t > 0$:
$$
\Sigma^{(+)}(\omega)_{rr'} = i\int_0^\infty dt\; e^{i\omega t}\;G_v(t)_{rr'}\;W(t)_{rr'} = i\int_0^\infty dt\; e^{i\omega t}\;\biggl(\sum_v\psi_{rv}\psi^*_{r'v}\,e^{-iE_v t}\biggr)\biggl(\sum_p B^p_{rr'}\,e^{-i\omega_p t}\biggr)\,.
$$
The product $G_v(t)W(t)$ is an $\mathcal{O}(N^2)$ Hadamard (elementwise) product at each $t$, where both $G_v(t)$ and $W(t)$ have $\mathcal{O}(N^3)$ evaluation times (given the set of wavefunctions and energies, and an existing plasmon model). The $\omega$-dependence is moved into the phase $e^{i\omega t}$. Although this is an $\mathcal{O}(N^3)$ expression, there are practical difficulties to numerical evaluation since the integrand oscillates strongly on the entire range from $t=0$ to $t=\infty$. 

Resolving the contributions from poles of all frequencies between $E^{(\mathrm{gap})}$ and $E^{(\mathrm{bandwidth})}$ would require $\mathcal{O}(E^{(\mathrm{bw})}/E^{(\mathrm{gap})})$ quadrature nodes, which for a typical semiconductor exceeds $\sim 100$ and negates the cubic-scaling advantage. The challenge is to find a quadrature of $\int_0^\infty dt\,e^{ixt}$ that requires $\mathcal{O}(N^0)$ $t$-points.

The integral $\int_0^\infty dt\,e^{ixt}$ does not converge pointwise; it is a distribution:
$$
\int_0^\infty dt\;e^{ixt} = \pi\delta(x) + i\,\operatorname{PV}\!\left(\frac{1}{x}\right).
$$
Any quadrature scheme for this integral must account for both the $\delta$-function piece and the $1/x$ principal-value piece. In our $\Sigma$ integral, we identify $x$ with $\omega - E_n \pm \omega_p$. Thus the $\delta$-function piece contributes where $\omega=E_n\mp\omega_p$. 

When all $x$ in a given spectral region share the same sign, a Wick rotation to imaginary time eliminates the oscillation entirely: the integral becomes a real, decaying Laplace transform and $1/x$ is recovered exactly. When $x$ takes both signs (energy crossings), the Wick rotation is unavailable and the oscillatory integral must be evaluated directly. Damping the integrand with a rapidly decaying envelope $h(t)$ suppresses the $\delta$-function, which requires infinite-time support to build up, while preserving $1/x$ for $|x|$ larger than the inverse envelope width.

---

Consider the contour-deformation (CD) approach, in which the $\omega'$ contour is deformed into the complex plane, avoiding all poles of $W_c$ and enclosing only the poles of $G$:
$$
\begin{aligned}
\Sigma_c(\omega)_{rr'}
&= \underbrace{-\frac{1}{2\pi}\int_{-\infty}^{\infty} d\omega'\;G_{rr'}(\omega - i\omega')\,W_{c,rr'}(i\omega')}_{\displaystyle\Sigma^{(\mathrm{imag})}:\;\text{smooth, }\mathcal{O}(10)\text{ points}}
\\[4pt]
&\quad+\underbrace{\sum_{\substack{n:\,E_n\;\text{enclosed}}} \operatorname{sgn}(E_n-\mu)\;\psi_{rn}\psi^*_{r'n}\,W_{c,rr'}(\omega - E_n)}_{\displaystyle\Sigma^{(\mathrm{res})}:\;\mathcal{O}(N)\text{ terms}}\,.
\end{aligned}
$$
The imaginary-axis integral converges rapidly because the integrand is smooth and $\sim|\omega'|^{-2}$. The residue sum is the bottleneck: $\mathcal{O}(N)$ evaluations of $W_c$, each costing $\mathcal{O}(N^3)$, give $\mathcal{O}(N^4)$ overall.

CTSP addresses both pieces with $\mathcal{O}(N^3)$ cost by working in the time domain, where the frequency-axis split of CD maps to a time-axis split. Non-crossing window pairs are integrated along the imaginary time axis ($t = i\tau/\zeta$) via Gauss–Laguerre quadrature, the time-domain analogue of $\Sigma^{(\mathrm{imag})}$. Crossing window pairs are integrated along a real-time path damped by the HGL envelope, effectively a contour near $t + i\gamma^{-1}$; these collectively evaluate contributions that in CD are split between the residue sum and the nearby portion of the imaginary-axis integral.

---

The CTSP method replaces the energy denominators $1/x$ by time integrals whose separability makes them amenable to $\mathcal{O}(N^3)$ evaluation. We treat the harder case first: energy crossings, where $x = \omega - E_n \pm \omega_p$ takes both signs.

Multiplying $e^{ixt}$ by any envelope $h(t)$ decaying sufficiently fast at $t \to \infty$ yields a smooth, odd regularization of $1/x$:
$$
F(x;\gamma) = \gamma\,\operatorname{Im}\int_0^\infty d\tau\;h(\tau)\,e^{i\gamma x\tau}\,,
$$
where $\tau = \gamma t$ is a dimensionless rescaled time. For $|x| \gg \gamma^{-1}$ the oscillations self-average on timescales shorter than the envelope's support, and $F(x;\gamma) \to 1/x$; for $|x| \lesssim \gamma^{-1}$ the envelope smooths the singularity. The parameter $\gamma^{-1}$ plays the role of an energy broadening, directly analogous to the $i\eta$ in the original denominator or the Lorentzian broadening in standard $GW$ implementations.

This regularization is useful only if the integral can be evaluated by a quadrature that converges in $\mathcal{O}(N^0)$ points, independent of the number of poles contributing. The key property enabling this is that the integrand $h(\tau)e^{i\gamma x\tau}$ is a product of a rapidly decaying weight and an oscillatory function of bounded frequency. For weight functions of the form $h(\tau) = e^{-\alpha\tau - \beta\tau^2}$, generalized Gaussian quadrature with $N$ points is exact for integrands of the form $h(\tau)p(\tau)$ where $p$ is a polynomial of degree $\leq 2N - 1$, and converges rapidly for smooth or analytic $s(\tau)$ in $\int_0^\infty h(\tau)s(\tau)\,d\tau$. When $s$ is oscillatory with frequency content bounded by $\gamma E^{(\mathrm{bw})}$, convergence is polynomial but controlled by the dimensionless bandwidth $\gamma E^{(\mathrm{bw})}$ alone, not by the number of poles.

Any envelope that decays fast enough gives $F \to 1/x$ at large $|x|$. The question is how fast. Expanding $F(x;\gamma) = 1/x + a_1/x^3 + \cdots$, the leading error coefficient $a_1$ depends on the ratio of the first moment of $h$ to its zeroth moment. The specific choice $h(\tau) = e^{-\tau - \tau^2/2}$ (the HGL weight) is the unique member of the family $e^{-\alpha\tau - \beta\tau^2}$ for which $a_1 = 0$, achieved when $\langle\tau\rangle_h / \langle 1\rangle_h = 1$ (the $1\!:\!1/2$ ratio of linear to quadratic exponent terms). The result is
$$
F(x;\gamma) = \frac{1}{x} + \mathcal{O}\!\left(\frac{1}{x^5}\right)\qquad\text{as}\;|x|\to\infty\,,
$$
two orders better than the Lorentzian ($\mathcal{O}(1/x^3)$). This controls the matching error at window boundaries where the method switches between crossing and non-crossing quadratures. In the limit $\gamma\to\infty$, $F(x;\gamma)\to 1/x$ pointwise, recovering the exact energy denominator.

The regularized integral inherits the separability of the original time-domain convolution. At each HGL quadrature node $\tau_u$, the propagators $\widetilde{G}_l$ and $\widetilde{W}_m$ are independently constructed in $\mathcal{O}(N^3)$, and the $\omega$-dependence factors out as a scalar phase. Define complex propagators at each node:
$$
\begin{aligned}
\widetilde{G}_l(\tau_u)_{rr'} &= \sum_{\{n\in\mathcal{L}_l\}} \psi_{rn}\psi^*_{r'n}\,e^{i\gamma\tau_u E_n}\,,\\[4pt]
\widetilde{W}_m(\tau_u)_{rr'} &= \sum_{\{p\in\mathcal{M}_m\}} B^p_{rr'}\,e^{i\gamma\tau_u\omega_p}\,,
\end{aligned}
$$
where $\mathcal{L}_l$ and $\mathcal{M}_m$ are the state and mode index sets in windows $l$ and $m$. The crossing contribution from window pair $(l,m)$ is
$$
\Sigma^{\mathrm{cross}}_{lm}(\omega) = \gamma\,\operatorname{Im}\sum_{u=1}^{N^{(\tau,\mathrm{HGL})}} w_u\;e^{i\gamma\tau_u\omega}\;\widetilde{G}_l(\tau_u)_{r'r}\;\widetilde{W}_m(\tau_u)_{rr'}\,.
$$
The $\omega$-dependence is entirely in the scalar phase $e^{i\gamma\tau_u\omega}$, factored out of the $\mathcal{O}(N^3)$ propagator product; all crossing frequencies are batched by a precomputed phase vector. In an implementation using the Euler identity, one stores two complex matrices per $\tau_u$, and the Hermitian product $\widetilde{W}_m^\dagger\,\widetilde{G}_l = P_+(\tau_u) - iP_\times(\tau_u)$ yields the real and imaginary tiles directly:
$$
\Sigma^{\mathrm{cross}}_{lm}(\omega) = \gamma\sum_u w_u\bigl[\cos(\gamma\tau_u\omega)\,P_\times(\tau_u) - \sin(\gamma\tau_u\omega)\,P_+(\tau_u)\bigr]\,.
$$

---

When all energy denominators in a window pair share the same sign, i.e., $\omega$ lies outside $[E^{(\mathrm{gap})}_{lm}, E^{(\mathrm{bw})}_{lm}]$, the regularization is unnecessary and one can do strictly better. The Wick rotation $t = i\tau/\zeta_{lm}$ maps the oscillatory Fourier integral to a purely decaying one:
$$
\int_0^\infty dt\;e^{ixt} \;\xrightarrow{\;t\,=\,i\tau/\zeta\;}\; \frac{i}{\zeta}\int_0^\infty d\tau\;e^{-x\tau/\zeta} \qquad(x > 0)\,.
$$
Here $\zeta_{lm}$ is the Wick-rotation rate, the ratio of physical imaginary time to the dimensionless quadrature variable $\tau$. Define the imaginary-time propagators for the window:
$$
\begin{aligned}
\bar{G}_l(\tau)_{rr'} &= \sum_{\{n\in\mathcal{L}_l\}} \psi_{rn}\psi^*_{r'n}\;e^{-\tau(E^{(\mathrm{max})}_l - E_n)/\zeta^{-1}_{lm}}\,,\\[4pt]
\bar{W}_m(\tau)_{rr'} &= \sum_{\{p\in\mathcal{M}_m\}} B^p_{rr'}\;e^{-\tau(\omega_p - \omega^{(\mathrm{min})}_m)/\zeta^{-1}_{lm}}\,.
\end{aligned}
$$
The exponents are measured from the window edges and bounded by $\tau\,\zeta_{lm} E^{(\mathrm{bw})}_{lm}$. Extracting the overall decay $e^{-\tau(\zeta_{lm}E^{(\mathrm{gap})}_{lm} - 1)}$ into the Gauss–Laguerre weight $e^{-\tau}$, the contribution from window pair $(l,m)$ is
$$
\Sigma^{\mathrm{GL}}_{lm}(\omega) = \zeta_{lm}\sum_{u=1}^{N^{(\tau,\mathrm{GL})}} w_u\;e^{-\tau_u(\zeta_{lm} E^{(\mathrm{gap})}_{lm} - 1)}\;\underbrace{e^{-\zeta_{lm}\omega\tau_u}}_{\omega\text{-phase}}\;\bar{G}_l(\zeta_{lm}\tau_u)_{r'r}\;\bar{W}_m(\zeta_{lm}\tau_u)_{rr'}\,.
$$
The integrand is real, non-negative, and monotonically decaying, ideal for Gauss–Laguerre (GL) quadrature, which converges exponentially for analytic integrands. The $\omega$-dependence is again a scalar phase per node, enabling simultaneous accumulation over all non-crossing frequencies. This Wick-rotated integral is the time-domain analogue of the imaginary-axis piece $\Sigma^{(\mathrm{imag})}$ of CD: both evaluate propagators along the imaginary axis where all poles are avoided and the response is smooth.

---

The accuracy of both quadratures depends on the ratio of bandwidth to gap in the spectral region being integrated. A single window spanning the full band structure of a typical semiconductor has $E^{(\mathrm{bw})}/E^{(\mathrm{gap})} \gtrsim 100$; the GL node count scales as $\sqrt{E^{(\mathrm{bw})}/E^{(\mathrm{gap})}}$ and the HGL count as $(\gamma E^{(\mathrm{bw})})^2$, both of which become large. Since both node counts depend on the bandwidth each quadrature must handle, the direct lever for reducing cost is to narrow the spectral range per window pair. The remedy is energy windowing: partitioning the spectra of $G$ and $W$ into $N_{n_w}$ band-state windows and $N_{p_w}$ plasmon-mode windows, so that each pair $(l,m)$ has its own reduced gap, bandwidth, and condition ratio. Defining the energy denominator $x_{np} = \omega - E_n \pm \omega_p$ for each state-mode pair:
$$
E^{(\mathrm{gap})}_{lm} = \min_{\{n\in\mathcal{L}_l,\,p\in\mathcal{M}_m\}}|x_{np}|\,,\qquad E^{(\mathrm{bw})}_{lm} = \max_{\{n\in\mathcal{L}_l,\,p\in\mathcal{M}_m\}}|x_{np}|\,,\qquad \alpha_{lm} = \sqrt{E^{(\mathrm{bw})}_{lm}/E^{(\mathrm{gap})}_{lm}}\,.
$$
The full self-energy is $\Sigma = \sum_{l,m}\Sigma_{lm}$. For a given $\omega$, most window pairs have all denominators of one sign and are treated by GL; the few pairs where $\omega \in [E^{(\mathrm{gap})}_{lm}, E^{(\mathrm{bw})}_{lm}]$ require HGL.

The GL energy-scale parameter $\zeta_{lm}$ governs how the physical imaginary time maps to the dimensionless quadrature variable; its optimal value $\zeta^{-1}_{lm} = \sqrt{E^{(\mathrm{bw})}_{lm}E^{(\mathrm{gap})}_{lm}}$ equalizes the fractional quadrature error at both edges of the window, exploiting the symmetry of the GL error on a logarithmic energy scale about $\zeta\Delta E = 1$. Window boundaries are chosen by minimizing the total operation count
$$
C^{(\mathrm{tot})}(\epsilon^{(q)}) = N_r^2\sum_{l,m} N^{(\tau)}_{lm}(\epsilon^{(q)})\bigl(L_l + L_m\bigr)\,,
$$
where $L_l = \int_{\mathrm{win}\,l}D(E)\,dE$ counts states or modes. This 1D optimization over window edges depends on $D(E)$ and the plasmon mode density $D^{(p)}(\omega)$; it is $\mathcal{O}(N^0)$ and performed once. Optimal windows are narrower where spectral density is high, wider where sparse; van Hove singularities are isolated in their own windows (a Lebesgue-type decomposition). Both $E^{(\mathrm{gap})}_{lm}$ and $E^{(\mathrm{bw})}_{lm}$ are intensive for periodic systems, so $N^{(\tau)}_{lm} = \mathcal{O}(N^0)$ and the total cost is $\mathcal{O}(N^3)$.

---

The following explicit formulas, fitted to numerical experiments on the GL and HGL quadratures, allow the node count for each window pair to be determined from the target error $\epsilon^{(q)}$ alone. The required number of GL nodes is
$$
N^{(\tau,\mathrm{GL})} = \alpha\bigl(0.4 - 0.3\ln\epsilon^{(q)}\bigr)\,,\qquad \alpha = \sqrt{E^{(\mathrm{bw})}/E^{(\mathrm{gap})}}\,,
$$
valid for $\epsilon^{(q)} < 0.135$, with error decaying as $\epsilon \sim e^{-cN/\alpha}$ ($c \approx 3.3$). The GL nodes (roots of the Laguerre polynomial $L_N$) have density $\rho(\tau) \sim (\pi\sqrt{\tau})^{-1}$ near the origin and extend to $\tau_{\max} \sim 4N$; in the physical imaginary time $t = \tau/\zeta$, this gives dense sampling at $t \sim \hbar/E^{(\mathrm{bw})}$ (fast decay components) and sparse sampling at $t \sim \hbar/E^{(\mathrm{gap})}$ (slow decay). The effective weight per node, $w_u\,e^{-(\zeta E^{(\mathrm{gap})}-1)\tau_u}$, decays rapidly at large $\tau$, concentrating the quadrature at short-to-intermediate times.

The required number of HGL nodes is
$$
N^{(\tau,\mathrm{HGL})} = c_2 x^2 + c_1 x + c_0\,,\qquad x = \gamma\,E^{(\mathrm{bw})}_{lm}\,,
$$
with coefficients
$$
c_2 = -0.0036\ln\epsilon^{(q)} + 0.11\,,\quad c_1 = -0.0043(\ln\epsilon^{(q)})^2 - 0.13\ln\epsilon^{(q)} + 0.54\,,\quad c_0 = -0.204\ln\epsilon^{(q)} - 0.29\,.
$$
The dominant $x^2$ scaling reflects the cost of resolving $\sim\gamma E^{(\mathrm{bw})}$ oscillation periods in the Fourier integrand with accumulating phase error; convergence is polynomial rather than exponential. HGL nodes (generated from the moments of $h(\tau)$ via Golub–Welsch) are quasi-uniformly distributed with $\tau_{\max} \sim \mathcal{O}(\sqrt{N})$ due to the Gaussian envelope. The broadening $\gamma$ trades precision against cost: $N^{(\tau,\mathrm{HGL})} \propto (\gamma E^{(\mathrm{bw})})^2$, while the regularization error $F(x;\gamma) - 1/x = \mathcal{O}(\gamma^{-4}/x^5)$ is negligible for $|x| \gg \gamma^{-1}$.

The entire method is controlled by a single convergence parameter $\epsilon^{(q)}$. The following table summarizes the key analytic properties.

| | GL (non-crossing) | HGL (crossing) |
|:---|:---|:---|
| Physical time path | Imaginary: $t = i\tau/\zeta$ | Real: $t = \tau/\gamma$, envelope $e^{-\gamma t - \gamma^2 t^2/2}$ |
| Integrand | $\bar{G}_l(i\tau)\bar{W}_m(i\tau)$: real, monotone | $\widetilde{G}_l(\tau)\widetilde{W}_m(\tau)e^{i\omega\tau}$: oscillatory, Gaussian-damped |
| Error vs $N^{(\tau)}$ | Exponential: $\epsilon \sim e^{-cN/\alpha}$ | Polynomial in $N$ for fixed $\gamma E^{(\mathrm{bw})}$ |
| $N^{(\tau)}$ | $\alpha(0.4 - 0.3\ln\epsilon^{(q)})$ | $c_2(\gamma E^{(\mathrm{bw})})^2 + c_1(\gamma E^{(\mathrm{bw}})) + c_0$ |
| Controlling ratio | $\alpha = \sqrt{E^{(\mathrm{bw})}/E^{(\mathrm{gap})}}$ | $x = \gamma E^{(\mathrm{bw})}$ |
| Free parameter | $\zeta_{lm} = (E^{(\mathrm{bw})}E^{(\mathrm{gap})})^{-1/2}$, per pair | $\gamma$, global |
| $\omega$-dependence | Phase $e^{-\zeta\omega\tau_u}$ per node | Phase $e^{i\gamma\omega\tau_u}$ per node |
| Node density | $\rho \sim 1/\sqrt{\tau}$; $\tau_{\max} \sim 4N$ | Quasi-uniform; $\tau_{\max} \sim \sqrt{N}$ |
| Approximation to $1/x$ | Exact (definite-sign $x$) | $1/x + \mathcal{O}(1/x^5)$; $F(0) = \gamma\sqrt{\pi/2}$ |
| CD analogue | Imaginary-axis integral | Crossing windows (residues + nearby imaginary-axis contributions) |
| System-size scaling of $N^{(\tau)}$ | $\mathcal{O}(N^0)$ | $\mathcal{O}(N^0)$ |

---

**References.** (1) M. Kim, G. J. Martyna, and S. Ismail-Beigi, Phys. Rev. B **101**, 035139 (2020). (2) D. A. Leon *et al.*, Phys. Rev. B **107**, 245132 (2023).
