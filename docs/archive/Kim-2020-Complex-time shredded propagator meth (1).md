\title{
Complex-time shredded propagator method for large-scale $\boldsymbol{G W}$ calculations
}

\author{
Minjung Kim, ${ }^{1}$ Glenn J. Martyna, ${ }^{2,3}$ and Sohrab Ismail-Beigi ${ }^{1, *}$ \\ ${ }^{1}$ Department of Applied Physics, Yale University, New Haven, Connecticut 06520, USA \\ ${ }^{2}$ IBM TJ Watson Laboratory, Yorktown Heights, 10598, New York, USA \\ ${ }^{3}$ Pimpernel Science, Software and Information Technology, Westchester, New York 10598, USA
}
(Received 23 April 2019; published 23 January 2020)

\begin{abstract}
The $G W$ method is a many-body electronic structure technique capable of generating accurate quasiparticle properties for realistic systems spanning physics, chemistry, and materials science. Despite its power, $G W$ is not routinely applied to study large complex assemblies due to the method's high computational overhead and quartic scaling with particle number. Here, the $G W$ equations are recast, exactly, as Fourier-Laplace time integrals over complex time propagators. The propagators are then "shredded" via energy partitioning and the time integrals approximated in a controlled manner using generalized Gaussian quadrature(s) while discrete variable methods are employed to represent the required propagators in real space. The resulting cubic scaling $G W$ method has a sufficiently small prefactor to outperform standard quartic scaling methods on small systems ( $\gtrsim 10$ atoms) and offers $2-3$ order of magnitude improvement in large systems ( $\approx 200-300$ atoms). It also represents a substantial improvement over other cubic methods tested for all system sizes studied. The approach can be applied to any theoretical framework containing large sums of terms with energy differences in the denominator.
\end{abstract}

DOI: 10.1103/PhysRevB.101.035139

\section*{I. INTRODUCTION}

Density functional theory (DFT) [1,2] within the local density (LDA) or generalized gradient (GGA) [3,4] approximation provides a solid workhorse capable of realistically modeling an ever increasing number and variety of systems spanning condensed matter physics, materials science, chemistry, and biology. Generally, this approach provides a highly satisfactory description of the total energy, electron density, atomic geometries, vibrational modes, etc. However, DFT is a ground-state theory for electrons and DFT band energies do not have direct physical meaning because DFT is not formally a quasiparticle theory. Therefore significant failures can arise when DFT band structure is used to predict electronic excitations [5-7].

The $G W$ approximation to the electron self-energy [8-11] is one of the most accurate fully $a b$ initio methods for the prediction of electronic excitations. Despite its power, $G W$ is not routinely applied to complex materials systems due to its unfavorable computational scaling: the cost of a standard $G W$ calculation scales as $\mathcal{O}\left(N^{4}\right)$ where $N$ is the number of atoms in the simulation cell whereas the standard input to a $G W$ study, a Kohn-Sham DFT calculation, scales as $\mathcal{O}\left(N^{3}\right)$.

Reducing the computational overhead of $G W$ calculations has been the subject of much prior research. First, $G W$ methods scaling as $\mathcal{O}\left(N^{4}\right)$ but with smaller prefactors either avoid the use of unoccupied states via iterative matrix inversion [12-18] or use sum rules or energy integration to greatly reduce the number of unoccupied states required for convergence [19-21]. Second, cubic-scaling $\mathcal{O}\left(N^{3}\right)$ methods, including both a spectral representation approach [22]

\footnotetext{
*sohrab.ismail-beigi@yale.edu
}
and a space/imaginary time method [23] utilizing analytical continuation from imaginary to real frequencies, have been proposed. Third, a linear scaling $G W$ technique [24] has recently been developed that employs stochastic approaches for the total density of electronic states with the caveat that the nondeterministic stochastic noise must be added to the list of usual convergence parameters.

Here, we present a deterministic, small prefactor, $\mathcal{O}\left(N^{3}\right)$ scaling $G W$ approach that does not require analytic continuation. The $G W$ equations are first recast exactly using FourierLaplace identities into the complex time domain where products of propagators expressed in real space using discrete variable techniques [25] are integrated over time to generate an $\mathcal{O}\left(N^{3}\right) G W$ formalism. However, the time integrals are challenging to perform numerically due to the multiple timescales inherent in the propagators. Second, the timescale challenge is met by shredding the propagators in energy space, again exactly, to allow windows of limited dynamical bandwidth to be treated via generalized Gaussian quadrature numerical integration with low overhead and high accuracy. The unique combination of a (complex) time domain formalism, bandwidth taming propagator partitioning, and discrete variable real-space forms of the propagators permits a fast $\mathcal{O}\left(N^{3}\right)$ to emerge. Last, our approach is easy to implement in standard $G W$ applications [26,27] because the formulae follow naturally from those of the standard approach(es) and much of the existing software can be refactored to utilize our reduced order technique.

The resulting $G W$ formalism is tested to ensure both its accuracy and high performance in comparison to the standard $\mathcal{O}\left(N^{4}\right)$ approach for crystalline silicon, magnesium oxide, and aluminium. The new method's accuracy and performance are compared also to that of reduced overhead quartic scaling methods as well as existing $\mathcal{O}\left(N^{3}\right)$ scaling techniques.

Importantly, we provide estimates of the speed-up over conventional $G W$ computations and the memory requirement in the application of the new method to study technologically and scientifically interesting systems consisting of $\lesssim 200-300$ atoms-the sweet spot for the approach on today's supercomputers.

\section*{II. THEORY}

\section*{A. Summary of $G W$}

The theoretical object of interest for understanding oneelectron properties such as quasiparticle bands and wave functions is the one-electron Green's function $G\left(x, t, x^{\prime}, t^{\prime}\right)$, which describes the propagation amplitude of an electron starting at $x^{\prime}$ at time $t^{\prime}$ and ending at $x$ at time $t$ [28]:
$$
i G\left(x, t, x^{\prime}, t^{\prime}\right)=\left\langle T\left\{\hat{\psi}(x, t) \hat{\psi}\left(x^{\prime}, t^{\prime}\right)^{\dagger}\right\}\right\rangle,
$$
where the electron coordinate $x=(r, \sigma)$ specifies electron position ( $r$ ) and spin ( $\sigma$ ). Here, $\hat{\psi}(x, t)$ is the electron annihilation field operator at ( $x, t$ ), $T$ is the time-ordering operator, and the average is over the statistical ensemble of interest. We focus primarily on the zero-temperature case (i.e., groundstate averaging); however, to treat systems with small gaps, the grand canonical ensemble is invoked. As is standard, henceforth atomic units are employed: $\hbar=1$ and the quantum of charge $e=1$.

The Green's function in the frequency domain obeys Dyson's equation
$$
G^{-1}(\omega)=\omega I-\left[T+V_{\text {ion }}+V_{H}+\Sigma(\omega)\right],
$$
where the $x, x^{\prime}$ indices have been suppressed; a more compact but complete notation shall be employed henceforth
$$
G(\omega)_{x, x^{\prime}}=G\left(x, x^{\prime}, \omega\right) .
$$

Above, $I$ is the identity operator, $T$ is the electron kinetic operator, $V_{\text {ion }}$ is the electron-ion interaction potential operator (or pseudopotential for valence electron only calculations), $V_{H}$ is the Hartree potential operator, and $\Sigma(\omega)$ is the self-energy operator encoding all the many-body interaction effects on the electron Green's function.

The $G W$ approximation to the self-energy is
$$
\Sigma(t)_{x, x^{\prime}}=i G(t)_{x, x^{\prime}} W\left(t^{+}\right)_{r, r^{\prime}},
$$
where $t^{+}$is infinitesimally larger than $t$ and $W(t)_{r, r^{\prime}}$ is the dynamical screened Coulomb interaction between an external test charge at $\left(r^{\prime}, 0\right)$ and $(r, t)$ :
$$
W(\omega)_{r, r^{\prime}}=\int d r^{\prime \prime} \epsilon^{-1}(\omega)_{r, r^{\prime \prime}} V_{r^{\prime \prime}, r^{\prime}}
$$

Here, $\epsilon$ is the linear response, dynamic and nonlocal microscopic dielectric screening matrix, and $V_{r, r^{\prime}}=1 /\left|r-r^{\prime}\right|$ is the bare Coulomb interaction. The $G W$ self-energy includes the effects due to dynamical and nonlocal screening on the propagation of electrons in a many-body environment. The notation introduced above (to be continued below) is that parametric functional dependencies are placed in parentheses and explicit dependencies are given as subscripts; the alternative notation wherein all variables are in parentheses with explicit dependencies given first followed by parametric dependencies
separated by a semicolon is also employed where convenient [e.g., $W\left(r, r^{\prime} ; \omega\right) \equiv W(\omega)_{r, r^{\prime}}$ ].

To provide a closed and complete set of equations, one must approximate $\epsilon$. The most common approach is the random-phase approximation (RPA): one first writes $\epsilon$ in terms of the dynamic irreducible polarizability $P$ via
$$
\begin{equation*}
\epsilon(\omega)_{r, r^{\prime}}=\delta\left(r-r^{\prime}\right)-\int d r^{\prime \prime} V_{r, r^{\prime \prime}} P(\omega)_{r^{\prime \prime}, r} \tag{1}
\end{equation*}
$$
and $P$ is related to $G$ by the RPA
$$
P(t)_{r, r^{\prime}}=-i \sum_{\sigma, \sigma^{\prime}} G(t)_{x, x^{\prime}} G(-t)_{x^{\prime}, x}
$$

In the vast majority of $G W$ calculations, including the formalism given here, the Green's function is approximated by an independent electron form (band theory) specified by a complete set of one-particle eigenstates $\psi_{n}(x)$ (compactified to $\left.\psi_{x, n}\right)$ and eigenvalues $E_{n}$
$$
\begin{equation*}
G(\omega)_{x, x^{\prime}}=\sum_{n} \frac{\psi_{x, n} \psi_{x^{\prime}, n}^{*}}{\omega-E_{n}} . \tag{2}
\end{equation*}
$$

The $\psi_{n}$ and $E_{n}$ are obtained as eigenstates of a noninteracting one-particle Hamiltonian from a first principles method such as density functional theory $[1,2]$, although one is not limited to this choice. Although not central to the analysis given here, formally $E_{n}$ has a small imaginary part that is positive for occupied states (i.e., energies below the chemical potential) and negative for unoccupied states. We have suppressed the nonessential crystal momentum index $k$ in Eq. (2) for simplicity-including it simply amounts to adding the $k$ index to the eigenstates $\psi_{x, n} \rightarrow \psi_{x, n}^{k}$ and energies $E_{n} \rightarrow E_{n}^{k}$ and averaging over the $k$ sampled in the first Brillouin zone (BZ).

For our purposes, the frequency domain representations of all quantities are useful. The Green's function $G$ in frequency space is given in Eq. (2) while the frequency dependent polarizability $P$ is
$$
\begin{align*}
P(\omega)_{r, r^{\prime}}= & \sum_{c, v, \sigma, \sigma^{\prime}} \psi_{x, c} \psi_{x, v}^{*} \psi_{x^{\prime}, c}^{*} \psi_{x^{\prime}, v}\left[f\left(E_{v}\right)-f\left(E_{c}\right)\right] \\
& \times \frac{2\left(E_{c}-E_{v}\right)}{\omega^{2}-\left(E_{c}-E_{v}\right)^{2}} \\
= & \sum_{c, v, \sigma, \sigma^{\prime}} \psi_{x, c} \psi_{x, v}^{*} \psi_{x^{\prime}, c}^{*} \psi_{x^{\prime}, v}\left[f\left(E_{v}\right)-f\left(E_{c}\right)\right] \\
& \times\left[\frac{1}{\left(\omega-\left(E_{c}-E_{v}\right)\right)}-\frac{1}{\left(\omega+\left(E_{c}-E_{v}\right)\right)}\right] \tag{3}
\end{align*}
$$

Here, $v$ labels occupied (valence) eigenstates while $c$ labels unoccupied (conduction) eigenstates. The occupancy function $f(E)$ required to handle finite temperatures for zero/small gap systems is explicitly included (see Sec. IID); for gapped systems at zero temperature $f\left(E_{v}\right)=1$ and $f\left(E_{c}\right)=0$. [The occupancy $f(E ; \beta, \mu)$ formally depends parametrically on two thermodynamic variables: the inverse temperature $\beta=$ $1 / k_{B} T$ and the chemical potential $\mu$.] We have employed a general, compact notation valid for collinear and noncollinear spin calculations. For collinear spin, nonzero contributions to $P$ only occur when the spin indices $\sigma$ and $\sigma^{\prime}$ of $x=(r, \sigma)$ and
$x^{\prime}=\left(r^{\prime}, \sigma^{\prime}\right)$ match; for the full spinor (noncollinear) case, we sum over all the spin projections $\sigma, \sigma^{\prime}$ in the usual way.

Of particular practical importance is the zero-frequency or static polarizability $P(\omega=0)$ (which we also simply denote as $P$ below)
$$
\begin{equation*}
P_{r, r^{\prime}}=-2 \sum_{c, v, \sigma, \sigma^{\prime}} \frac{\psi_{x, c} \psi_{x, v}^{*} \psi_{x^{\prime}, c}^{*} \psi_{x^{\prime}, v}\left[f\left(E_{v}\right)-f\left(E_{c}\right)\right]}{E_{c}-E_{v}} \tag{4}
\end{equation*}
$$
which is employed both as part of plasmon-pole models of the frequency dependent screening [8-11,29] as well as within the COHSEX approximation [8] (see below). Again, the crystal momentum index has been suppressed for simplicity; including it requires the replacements $P \rightarrow P^{q}$, where $q$ is the momentum transfer, $\psi_{x, v} \rightarrow \psi_{x, v}^{k}$ and $E_{v} \rightarrow E_{v}^{k}, \psi_{x, c} \rightarrow$ $\psi_{x, c}^{k+q}$ and $E_{c} \rightarrow E_{c}^{k+q}$, and averaging Eqs. (3) and (4) over $k$ (i.e., Brillouin zone sampling). We note that current numerical methods for computing $P$ based on the sum-over-states formulas, e.g., that of Eq. (4), have an $\mathcal{O}\left(N^{4}\right)$ scaling (e.g., see Ref. [26]).

Formally, the screened interaction $W$ can always be represented as a sum of "plasmon" screening modes indexed by $p$,
$$
\begin{align*}
W(\omega)_{r, r^{\prime}} & =V_{r, r^{\prime}}+\sum_{p} \frac{2 \omega_{p} B_{r, r^{\prime}}^{p}}{\omega^{2}-\omega_{p}^{2}} \\
& =V_{r, r^{\prime}}+\sum_{p} B_{r, r^{\prime}}^{p}\left(\frac{1}{\omega-\omega_{p}}-\frac{1}{\omega+\omega_{p}}\right) \tag{5}
\end{align*}
$$

Here, $B_{p}$ is the mode strength for screening mode $p$ and $\omega_{p}>0$ is its frequency. This form is directly relevant when making computationally efficient plasmon-pole models for the screened interaction [29]. The self-energy is then given by
$$
\begin{align*}
\Sigma(\omega)_{x, x^{\prime}}= & -\sum_{v} \psi_{x, v} \psi_{x^{\prime}, v}^{*} W\left(\omega-E_{v}\right)_{r, r^{\prime}} \\
& +\sum_{n, p} \frac{\psi_{x, n} B_{r, r^{\prime}}^{p} \psi_{x^{\prime}, n}^{*}}{\omega-E_{n}-\omega_{p}} \\
= & -\sum_{v} \psi_{x, v} V_{r, r^{\prime}} \psi_{x^{\prime}, v}^{*} \\
& +\sum_{v, p} \frac{\psi_{x, v} B_{r, r^{\prime}}^{p} \psi_{x^{\prime}, v}^{*}}{\omega-E_{v}+\omega_{p}}+\sum_{c, p} \frac{\psi_{x, c} B_{r, r^{\prime}}^{p} \psi_{x^{\prime}, c}^{*}}{\omega-E_{c}-\omega_{p}} \tag{6}
\end{align*}
$$
where the $n$ sum is over all bands (i.e., valence and conduction). Inclusion of crystal momentum in Eq. (6) means $\Sigma(\omega)$ carries a $k$ index, $\psi_{x, v} \rightarrow \psi_{x, v}^{k-q}$ and $E_{v} \rightarrow E_{v}^{k-q}$. All screening quantities derived from $P^{q}$ now also carry a $q$ index, $W^{q}, \omega_{p}^{q}$ and $B^{p q}$, and Eq. (6) is averaged over the $q$ sampling.

Within the COHSEX approximation, when the applicable screening frequencies, $\omega_{p}$, are much larger than the interband energies of interest, the frequency dependence of $\Sigma$ can be neglected
$$
\begin{align*}
\Sigma_{x, x^{\prime}}^{(\text {COHSEX })}= & -\sum_{v} \psi_{x, v} \psi_{x^{\prime}, v}^{*} W(0)_{r, r^{\prime}} \\
& +\frac{1}{2} \delta\left(x-x^{\prime}\right)\left[W(0)_{r, r^{\prime}}-V_{r, r^{\prime}}\right] \tag{7}
\end{align*}
$$
where the label in the supercript is placed in paranthesis to avoid possible confusion-a convention to be followed below. The numerically intensive part of the COHSEX approximation is the computation of the static polarizability, Eq. (4)once $P$ is on hand, the static $W(0)$ is completely determined by $P$ via matrix multiplication and inversion,
$$
W(0)=\epsilon^{-1}(0) V=(I-V P)^{-1} V .
$$

Equations (3), (4), (6), and (7) are of primary interest, here, as evaluating them scales as $\mathcal{O}\left(N^{4}\right)$ as written. Terms with manifestly cubic scaling terms will not be discussed further. The computation of observables such as $\epsilon_{\infty}$ and the band gap in various approximations, e.g., $E^{\text {(gap, } \mathrm{G}_{0} \mathrm{~W}_{0} \text { ) }}$ and $E^{\text {(gap,COHSEX) }}$, from the key terms, are described in Refs. [8-11]. The superscript on the band gap is employed to distinguish the gap of the input single-particle spectrum gap $E^{\text {(gap) }}$, from appropriate corrections to it which we present below to evaluate the performance of the new method. Comparison of the accuracy of different approximations to the gap is not part of this work but is fully described in the above references.

\section*{B. Complex time shredded propagator formalism}

We now describe the main ideas and merits of our new approach to cubic scaling $G W$ calculations. The resulting formalism is general and can be applied to a broad array of theoretical frameworks whose evaluation involves sums over states with energy differences in denominators.

The analytic structure of the equations central to $G W$ calculations, outlined in the prior section, necessitates the evaluation of terms of the form
$$
\begin{equation*}
\chi(\omega)_{r, r^{\prime}}=\sum_{i=1}^{N_{a}} \sum_{j=1}^{N_{b}} \frac{A_{r, r^{\prime}}^{i} B_{r, r^{\prime}}^{j}}{\omega+a_{i}-b_{j}} \tag{8}
\end{equation*}
$$
as can be discerned from Eqs. (3), (4), (6), and (7). The input energies $a_{i}$ and $b_{j}$ and the matrices $A^{i}$ and $B^{j}$ are either direct outputs of the $\mathcal{O}\left(N^{3}\right)$ ground state calculation (i.e., single particle energies and products of wave functions when $\chi=P$ ), or are obtained from $\mathcal{O}\left(N^{3}\right)$ matrix operations on the frequency dependent polarizability $P(\omega)$, or other such derived quantities.

The analytic form of $\chi$ in Eq. (8) arises because we have chosen to work in the frequency or energy representation. However, one can equally well represent such an equation in real, imaginary or complex time by changing the structure of the theory to involve time integrals over propagators. Here, we will effect the change of representation from time to frequency directly through the introduction of Fourier-Laplace identities which allows us to reduce the computational complexity of the $G W$ calculation. This imaginary time formalism has connections to prior work found in Refs. [23,30,31].

In more detail, while the frequency representation has advantages, the evaluation of Eq. (8) scales as $\mathcal{O}\left(N_{a} N_{b} N_{r}^{2}\right)$ because the numerator is separable but the energy denominator is not. This basic structure of the frequency representation leads to the familiar $\mathcal{O}\left(N^{4}\right)$ computational complexity of $G W$ as the number of states or modes ( $N_{a}, N_{b}$ ) and the number real-space points ( $N_{r}$ ) required to represent them, here by discrete variable methods, scale as the number of electrons, $N$.

For the widely used plane wave (i.e., Fourier) basis, adopted herein, a uniform grid in $r$ space that is dual to the finite $g$-space representation is indicated-fast Fourier transforms (FFTs) switch between the dual spaces, $g$ and $r$ space, both efficiently and exactly (without information loss); for other basis sets, appropriate real-space discrete variable representations (DVRs) with similar dual properties can be adopted [25,32,33].

In the following, a time domain formalism that reduces the computational complexity of Eq. (8) by $N$ to achieve $\mathcal{O}\left(\left(N_{a}+\right.\right.$ $\left.\left.N_{b}\right) N_{r}^{2}\right) \sim \mathcal{O}\left(N^{3}\right)$ scaling, in a controlled and rapidly convergent manner, is developed. This will be accomplished through the introduction of time integrals and associated propagators which we shall then shred (i.e., partition) to tame the multiple timescales inherent to the theory. Again, the resulting formulation is general: it applies to any theory with the structure of Eq. (8).

Reduced scaling is enabled by replacing the energy denominator $1 /\left(\omega+a_{i}-b_{j}\right)$ of Eq. (8) by a separable form through the introduction of the generalized Fourier-Laplace transform
$$
\begin{equation*}
F(E ; \zeta)=\int_{0}^{\infty} d \tau h(\tau ; \zeta) \exp [-\zeta E \tau] \tag{9}
\end{equation*}
$$

That is, inserting the transform, Eq. (8) becomes
$$
\begin{equation*}
\chi(\omega ; \zeta)_{r, r^{\prime}}=\sum_{i=1}^{N_{a}} \sum_{j=1}^{N_{b}} F\left(\omega+a_{i}-b_{j} ; \zeta\right) A_{r, r^{\prime}}^{i} B_{r, r^{\prime}}^{j} \tag{10}
\end{equation*}
$$

Here, $\zeta$ is a complex constant with $|\zeta|$ akin to an inverse Planck's constant that sets the energy scale, and $h(\tau ; \zeta)$ is a weight function. The desired separability arises from the exponential function in the integrand of $F(E ; \zeta)$ and allows us to reduce the computational complexity of $G W$. In the following, the $\zeta$ dependence of $\chi$ will be suppressed for reasons that will become immediately apparent.

To motivate the utility of Eq. (10), consider the case where $\forall i, j$ either $\omega+a_{i}-b_{j}>0$ or $\omega+a_{i}-b_{j}<0$ : here, $\zeta$ is chosen to be real (positive for the first case and negative for the second), and we set $h(\tau ; \zeta)=\zeta$. This corresponds to a textbook Laplace transform [34] and yields an exact expression for the energy denominator:
$$
\begin{equation*}
\lim _{h(\tau ; \zeta) \rightarrow \zeta} F\left(\omega+a_{i}-b_{j} ; \zeta\right)=\frac{1}{\omega+a_{i}-b_{j}} \tag{11}
\end{equation*}
$$

For this case, the introduction of the transform involves no approximation, and $h(\tau ; \zeta)=\zeta$ will be employed to establish and describe our formalism. It is directly applicable to the static limit of $\chi(\omega)$ where $\omega \rightarrow 0$ and $a_{i}-b_{j}>0 \forall i, j$ [i.e., gapped systems, cf. the static polarizability matrix of Eq. (4)]. The importance of the actual value of $\zeta$ will become clear below. A yet more general treatment, applicable to gapless systems and finite frequencies $\omega \neq 0$, requiring nontrivial $h(\tau ; \zeta)$, will then be given, wherein $F$ becomes an approximation to the inverse of the energy denominator within the class of regularization procedures commonly employed in standard $G W$ computations.

Inserting the generalized Fourier-Laplace identity into Eq. (8) yields
$$
\begin{align*}
\chi(0)_{r, r^{\prime}}= & \int_{0}^{\infty} d \tau h(\tau ; \zeta)\left[\sum_{i=1}^{N_{a}} A_{r, r^{\prime}}^{i} e^{-\zeta\left(a_{i}-E^{(\mathrm{off})}\right) \tau}\right] \\
& \times\left[\sum_{j=1}^{N_{b}} B_{r, r^{\prime}}^{j} e^{-\zeta\left(E^{(\mathrm{off})}-b_{j}\right) \tau}\right] \\
= & \int_{0}^{\infty} d \tau h(\tau ; \zeta) \rho_{r, r^{\prime}}^{(A)}(\zeta \tau) \bar{\rho}_{r, r^{\prime}}^{(B)}(\zeta \tau) \\
= & \int_{0}^{\infty} d \tau h(\tau ; \zeta) \tilde{\chi}(\zeta \tau ; 0)_{r, r^{\prime}} \tag{12}
\end{align*}
$$

Here, $E^{(\text {off })}$ is a convenient energy offset selected such that all the exponential functions are decaying (e.g., midgap) and
$$
\begin{align*}
& \rho^{(A)}(\zeta \tau)_{r, r^{\prime}}=\sum_{i=1}^{N_{a}} A_{r, r^{\prime}}^{i} e^{-\zeta\left(a_{i}-E^{(\mathrm{off})}\right) \tau} \\
& \bar{\rho}^{(B)}(\zeta \tau)_{r, r^{\prime}}=\sum_{j=1}^{N_{b}} B_{r, r^{\prime}}^{j} e^{-\zeta\left(E^{(\mathrm{off})}-b_{j}\right) \tau} \\
& \tilde{\chi}(\zeta \tau ; 0)_{r, r^{\prime}}=\rho^{(A)}(\zeta \tau)_{r, r^{\prime}} \bar{\rho}^{(B)}(\zeta \tau)_{r, r^{\prime}} \tag{13}
\end{align*}
$$
where the $\rho^{(A, B)}(\zeta \tau)$ are imaginary time propagators (manifestly, for $a_{i}>b_{i} \forall i, j$ but the reverse is treated by letting $\zeta \rightarrow-\zeta$ and switching the $\rho$ and $\bar{\rho}$ labels). The result is a separable form for $\tilde{\chi}(\tau \zeta ; 0)_{r, r^{\prime}}$, a product of $A$ and $B$ propagators, whose zero frequency transform over $h(\tau ; \zeta)$ yields the desired $\chi(0)_{r, r^{\prime}}$. This exact reformulation can be evaluated in $\mathcal{O}\left(N^{3}\right)$ given that an $\mathcal{O}\left(N^{0}\right)$ scaling discretization (i.e., quadrature) of the time integral can be defined.

Consider that the largest energy difference in the argument of the exponential terms defining $\tilde{\chi}(\zeta \tau ; 0)_{r, r^{\prime}}$, is the bandwidth $E^{(\text {bw })}=\max \left(a_{i}\right)-\min \left(b_{j}\right)$ while the smallest energy difference is the gap $E^{\text {(gap) }}=\min \left(a_{i}\right)-\max \left(b_{j}\right)$ which are both known from input. Both energy differences are essentially independent of system size $N$ for large $N$ (exactly so for periodically replicated arrays of atoms in a supercell). Hence the longest and shortest timescales, $\sim \hbar / E^{(\mathrm{bw})}$ and $\sim \hbar / E^{(\mathrm{gap})}$, in $\tilde{\chi}(\tau \zeta ; 0)_{r, r^{\prime}}$ are independent of $N$. Therefore, barring nonanalytic behavior in the density of states or modes, a system size independent discretization scheme can be devised to generate $\chi(0)_{r, r^{\prime}}$ from $\tilde{\chi}(\zeta \tau ; 0)_{r, r^{\prime}}$. Of course, the formulation is most useful when the discrete form rapidly approaches the continuous integral with increasing number of discretizations (i.e., quadrature points).

The development of a rapidly convergent discretization scheme is, however, challenged by the large dynamic range present in the electronic structure of most materials systems, $E^{(\text {bw })} / E^{(\text {gap })} \gtrsim 100$. Simply selecting the free parameter $|\zeta| \approx$ $1 / E^{(\text {bw })}$ to treat such large bandwidths is insufficient to allow a small number of discretizations (i.e., number of quadrature points) to represent the time integrals accurately. Hence, an efficient approach capable of taming the multiple timescale challenge presented by the large dynamic range in the integrand, $\tilde{\chi}(\zeta \tau ; 0)_{r, r^{\prime}}$, of $\chi(0)_{r, r^{\prime}}$, will be given. Once such an approach has been developed for gapped systems, the solution
![](https://cdn.mathpix.com/cropped/2025_06_13_f17928b5a3ec4669a328g-05.jpg?height=599&width=879&top_left_y=200&top_left_x=130)

FIG. 1. An example of the proposed energy windowing approach with $N_{a_{w}}=N_{b_{w}}=2$ (a) For gapped systems, the energy ranges of $\left\{a_{i}\right\}$ and $\left\{b_{j}\right\}$ do not overlap. (b) For systems with overlapping energy ranges, energy window pairs arise both with energy crossings, red arrows, and without, blue arrows.
will be generalized to treat gapless systems and response functions at finite frequencies through use of imaginary $\zeta$ and nontrivial $h(\tau ; \zeta)$.

In order to tame the multiple timescales inherent in the present time domain approach to $\chi(0)_{r, r^{\prime}}$, the propagators $\rho^{(A, B)}$ must be modified. Borrowing ideas from Feynman's path integral approach, the propagators are "shredded" (sliced into pieces) in energy space. That is, the energy range spanned by $a_{i}$ is partitioned into $N_{a_{w}}$ contiguous energy windows indexed by $l=1, \ldots, N_{a_{w}}$ and $b_{j}$ is similarly partitioned into $N_{b_{w}}$ windows indexed by $m=1, \ldots, N_{b_{w}}$; to illustrate this shredding, a $2 \times 2$ energy window decomposition for a gapped system is shown in Fig. 1(a) (i.e., $N_{a_{w}}=N_{b_{w}}=2$ ). Shredding the propagators allows $\tilde{\chi}(\tau \zeta ; 0)_{r, r^{\prime}}$ to be recast exactly as a sum over window pairs $(l, m)$,
$$
\begin{equation*}
\chi(0)_{r, r^{\prime}}=\sum_{l=1}^{N_{a_{w}}} \sum_{m=1}^{N_{b_{w}}} \int_{0}^{\infty} d \tau h\left(\tau ; \zeta_{l m}\right) \tilde{\chi}^{l m}\left(\zeta_{l m} \tau ; 0\right)_{r, r^{\prime}} \tag{14}
\end{equation*}
$$
where for each window pair $(l, m)$,
$$
\begin{align*}
\tilde{\chi}^{l m}\left(\zeta_{l m} \tau ; 0\right)_{r, r^{\prime}} & =\rho_{l m}^{(A)}\left(\zeta_{l m} \tau\right)_{r, r^{\prime}} \bar{\rho}_{l m}^{(B)}\left(\zeta_{l m} \tau\right)_{r, r^{\prime}} \\
\rho_{l m}^{(A)}\left(\zeta_{l m} \tau\right)_{r, r^{\prime}} & =\sum_{\{i \in \mathcal{L}\}} A_{r, r^{\prime}}^{i} e^{-\zeta_{l m}\left(a_{i}-E^{(\mathrm{off})}\right) \tau} \\
\bar{\rho}_{l m}^{(B)}\left(\zeta_{l m} \tau\right)_{r, r^{\prime}} & =\sum_{\{j \in \mathcal{M}\}} B_{r, r^{\prime}}^{j} e^{\left.-\zeta_{l m}^{(\mathrm{off})}-b_{j}\right) \tau} \tag{15}
\end{align*}
$$

Here, $\mathcal{L}$ and $\mathcal{M}$ represent the sets of integer indices of the single particle states that contribute to the $l$ th A-type and $m$ th B-type energy windows, respectively. The energy $E^{(\text {off })}$ is an offset chosen for convenience: e.g., choosing it to be in the gap between the smallest $a_{i}$ and largest $b_{j}$ to generate strictly decaying exponential functions. As above, treating $b_{j}>a_{i}$ only necessitates reversing the sign of the $\zeta_{l m}$ and switching the bar labels on the density matrices. The energy windows need not be equally spaced in energy; in fact, the optimal choice of windows is not equally spaced even for a uniform density of states or modes as shown in Sec. II C.

The shredded form of $\chi(0)_{r, r^{\prime}}$ given in Eq. (14) has computational complexity of $\mathcal{O}\left(N^{3}\right)$ because the operation count to evaluate it, is
$$
\begin{equation*}
N_{r}^{2} \sum_{l m}\left(L_{l}^{(A)}+L_{m}^{(B)}\right) N_{l m}^{(\tau, h)} \sim \mathcal{O}\left(N^{3}\right) \tag{16}
\end{equation*}
$$
to be compared with the operation count of the standard $G W$ method, $N_{a} N_{b} N_{r}^{2} \sim \mathcal{O}\left(N^{4}\right)$. Here, the $L_{l}^{(A)}, L_{m}^{(B)} \sim \mathcal{O}(N)$ are the number of states or modes in the $l$ th and $m$ th energy windows, respectively, and $N_{l m}^{(\tau, h)} \sim \mathcal{O}\left(N^{0}\right)$ is the number of quadrature points required for accurate integration in a specific window pair $(l, m)$ (see Sec. II C).

The shredded propagator formulation of $\chi(0)_{r, r^{\prime}}$ has four important advantages. First, every term in the double sum over window pairs $(l, m)$ has its own intrinsic bandwidth which is handled by its own $\zeta_{l m}$ while preserving the desired separability. Second, each window pair can be assigned its own quadrature optimized to treat its limited dynamic range. Third, the windows can be selected to minimize the dynamic range in the window pairs which allows small $N_{l m}^{(\tau, h)}$ (i.e., efficient quadrature) to treat all pairs with small fractional error, $\epsilon^{(q)}$. These first three advantages are sufficient to tame the multiple timescale challenge. Fourth, finite frequency expressions for gapped systems as well as gapless systems at finite temperature can be addressed utilizing simple extensions of Eq. (14) as demonstrated below.

The next theoretical issue to tackle is to show that the optimal windows can be found in $\mathcal{O}\left(N^{3}\right)$ or less computational effort given the input energies $a_{i}$ and $b_{j}$. Since the computationally intensive part of $\chi(0)_{r, r^{\prime}}$ involves its $r, r^{\prime}$ spatial dependence, it is best to choose an optimal windowing scheme in the limit $A_{r, r^{\prime}}^{i}, B_{r, r^{\prime}}^{j} \rightarrow 1$ as, within a limited energy range of a window pair, the spatial dependence of the $A^{i}$ or $B^{j}$ are to good approximation similar. (Note, the plane-wave basis approach considered here does not exploit spatial locality and full-sized $N_{r}^{2}$ matrices are employed, but other approaches may benefit considering spatial locality in window creation). If the density of states for $a_{i}$ and $b_{j}$ is taken to be locally flat, then the optimal number and placement of windows can be determined in $\mathcal{O}\left(N^{0}\right)$; if the actual density of states is taken into account, the scaling remains $\mathcal{O}\left(N^{0}\right)$ as the density of states is an input from the electronic structure computation (typically, KS-DFT). Here, optimal indicates the windows are selected to minimize the operation count, Eq. (16), required to compute Eq. (14) over the number and placement (in energy space) of the windows. In practice, as discussed in Sec. II C, we take $N_{l m}^{(\tau, h)}$ to be the number of quadrature points required to guarantee a prespecified, upper error bound, obeyed by all the time integrals of each window pair; again, each window pair $(l, m)$ has its own tuned quadrature and timescale taming parameter, $\zeta_{l m}$.

The control given by the energy windowed formulation of $\chi(0)_{r, r^{\prime}}$ in Eq. (14) is the key to extending our efficient $\mathcal{O}\left(N^{3}\right)$ method to gapless systems and to finite frequencies. For gapless systems at zero frequency, there will be some few energy windows pairs (most likely only one) for which $a_{i}=$ $b_{j}$ happens at least once. This is not problematic because, e.g., for the case of computing the polarizability matrix of Eqs. (3) and (4), the occupancy difference $f\left(E_{v}\right)-f\left(E_{c}\right)$ regularizes
the singularity of the denominator via L'Hôpital's rule applied to $\left[f\left(E_{v}\right)-f\left(E_{c}\right)\right] /\left(E_{c}-E_{v}\right)$ (the mapping from the general formalism being $\left.a_{i} \rightarrow E_{v}, b_{j} \rightarrow E_{c}\right)$. Adding the occupancy factors presents no difficulties: all that is required is to take the difference between two terms of the same form as Eq. (8) in the problematic window pair(s) with an overlapping energy range-a small added expense (see Sec. IID). However, a more general approach that can handle finite frequencies, described next, can also be adopted to handle gapless systems.

For the case of finite frequency $\omega \neq 0$, in some window pair(s) the quantity in the denominator, $e_{i j}=\omega+a_{i}-b_{j}$, can change sign [see Fig. 1(b)]. In standard $G W$ implementations, singularities (zeros of $e_{i j}$ ) that may arise in these window pairs are tamed by either dropping their contributions to the sum when $\left|e_{i j}\right|$ is small [9] or by regularizing $1 / e_{i j}$, e.g., replacing $1 / e_{i j}$ by $e_{i j} /\left(e_{i j}^{2}+|\zeta|^{-2}\right)$ [26].

Lorentzian regularization can be accommodated easily within our time domain formalism by selecting $h(\tau ; \zeta)=$ $|\zeta| \exp (-\tau)$ for the weight function in Eqs. (9) and (10) and choosing $\zeta$ to be a pure imaginary number,
$$
\begin{align*}
\frac{e_{i j}}{e_{i j}^{2}+|\zeta|^{-2}}= & \operatorname{Im}\left[\int_{0}^{\infty} d \tau|\zeta| e^{-\tau} e^{i|\zeta| e_{i j} \tau}\right] \\
= & |\zeta| \int_{0}^{\infty} d \tau e^{-\tau}\left[\sin \left(|\zeta|\left(\omega-b_{j}\right)\right) \cos \left(|\zeta| a_{i}\right)\right. \\
& \left.-\cos \left(|\zeta|\left(\omega-b_{j}\right)\right) \sin \left(|\zeta| a_{i}\right)\right] \tag{17}
\end{align*}
$$
for the small number of window pairs where $e_{i j}$ changes sign. In order to factorize the complex exponential and expose the separability of $i, j$ in the second line of the above equation, we have chosen to decompose the energy difference as $e_{i j}=\left(\omega-b_{j}\right)+\left(a_{j}\right)$, but the decomposition $e_{i j}=(\omega+$ $\left.a_{i}\right)+\left(-b_{j}\right)$ is also possible. Nonetheless, a large number of quadrature points must be taken to accurately discretize the time integral of Eq. (17), in practice.

Alternatively, as will be detailed in Sec. II E, the weight function
$$
h(\tau ; \zeta)=|\zeta| \exp \left(-\tau-\tau^{2} / 2\right)
$$
and its transform
$$
\begin{align*}
F\left(e_{i j} ; \zeta\right)= & |\zeta| \operatorname{Im}\left\{\sqrt{\frac{\pi}{2}} \exp \left(-\frac{\left(e_{i j}|\zeta|+i\right)^{2}}{2}\right)\right. \\
& \left.\times\left[1+i \operatorname{erfi}\left(\frac{e_{i j}|\zeta|+i}{\sqrt{2}}\right)\right]\right\} \tag{18}
\end{align*}
$$
form a preferable choice of regularization. Importantly, the transform, Eq. (18), approaches $1 / e_{i j}$ at large $e_{i j}$, is well behaved for all $e_{i j}$ but can be generated accurately with fewer time integration quadrature points than required by the Lorentzian. The benefits of the alternative weight function, an asymptotic analysis, and the associated rapidly convergent quadrature are presented in Sec. II E 2 and associated appendices.

Lastly, we note that the new formalism can handle problematic regions/points in the density of states that might need specialized treatment, such as van Hove singularities, by simply assigning them their own window in a Lebesgue-type approach (see Sec. II C 3) [35]. As long as the number of
![](https://cdn.mathpix.com/cropped/2025_06_13_f17928b5a3ec4669a328g-06.jpg?height=1365&width=877&top_left_y=198&top_left_x=1033)

FIG. 2. Numerical error vs computational savings for our cubic scaling formalism, CTSP-W, compared to the standard quartic $G W$ formulation for bulk Si and MgO modeled in a 16 atom supercell. The CTSP-W error decreases and computational work increases as the integration error is decreased (i.e., the number of quadrature points is increased). Computational work is measured by the ratio of operation count, Eq. (16), of the cubic method to the quartic method. (Top) Error in the macroscopic optical dielectric constant $\left[\epsilon_{\infty}(\mathrm{MgO})=6.35, \epsilon_{\infty}(\mathrm{Si})=64.85\right]$. (Bottom) Error in the COHSEX band gap $\left[E^{\text {(gap,COHSEX) }}(\mathrm{MgO})=7.56 \mathrm{eV}, E^{\text {(gap,COHSEX) }}(\mathrm{Si})=\right.$ $1.92 \mathrm{eV}]$.
special regions/points is independent of systems size, the scaling of the method remains $\mathcal{O}\left(N^{3}\right)$.

In order to convince the reader that the new formalism represents an important improvement, we provide a comparison of our $\mathcal{O}\left(N^{3}\right)$ time domain results to those of the corresponding $\mathcal{O}\left(N^{4}\right)$ direct frequency domain computation in Fig. 2 for two standard test systems, crystalline silicon and magnesium oxide. In the figure, the new method is referred to via the sobriquet complex time shredded propagator (CTSP) method where CTSP-W indicates the use of optimal windowing, and in the discussion to follow, CTSP-1 the use of one window. Even for small unit/supercells, the $\mathcal{O}\left(N^{3}\right)$ computational approach outlined above delivers a significant reduction in computational effort compared to the standard
approach (the CTSP error decreases exponentially with the number of time integration quadrature points as given in Sec. II C 2 b and logarithm-linear plots are thereby the natural way to present the data).

The detailed analysis underlying CTSP's reduced scaling with system size and high performance is presented in Secs. II C-II F and associated appendices. We also show below that (all) the new method's parameters can be reduced to one, the fractional time integration quadrature error $\epsilon^{(q)}$, which allows for the easily tunable convergence demonstrated by the results given above (see Fig. 2). The use of the simple operation count as given in Eq. (16) to represent computational work is, also, justified in the following.

\section*{C. Static polarization matrix in $\mathcal{O}\left(N^{3}\right)$ for gapped systems}

The static polarizability matrix defined in Eq. (4) reduces, for systems with large energy gaps compared to $k_{B} T$, to
$$
P_{r, r^{\prime}}=-2 \sum_{v}^{N_{v}} \sum_{c}^{N_{c}} \frac{\psi_{r, v}^{*} \psi_{r, c} \psi_{r^{\prime}, c}^{*} \psi_{r^{\prime}, v}}{E_{c}-E_{v}}
$$
as the occupation number functions for this special case are zero or one; the occupancies will be reintroduced to treat zero-gap systems in Sec. IID. Here, $N_{v}$ and $N_{c}$ are the number of valence and conduction states, respectively. Nonessential indices or quantum numbers such as spin $\sigma$ and Bloch $k$ vector have been suppressed.

\section*{1. Laplace identity and shredded propagators}

Employing the energy windowing approach of Eqs. (14) and (15), the energy range of the valence and conduction band is divided into $N_{v_{w}}$ and $N_{c_{w}}$ partitions with the valence and conduction partition indexed by $l$ and $m$ ranging from $E_{l}^{(v, \text { min) }}$ to $E_{l}^{(v, \max )}$ and $E_{m}^{(c, \min )}$ to $E_{m}^{(c, \max )}$, respectively. Thus the static polarizability can be written as
$$
\begin{equation*}
P_{r, r^{\prime}}=\sum_{l=1}^{N_{v_{w}}} \sum_{m=1}^{N_{c_{w}}} P_{r, r^{\prime}}^{l m} \tag{19}
\end{equation*}
$$
where each window pair $(l, m)$ contributes
$$
\begin{align*}
P_{r, r^{\prime}}^{l m}= & -2 \zeta_{l m} \int_{0}^{\infty} d \tau e^{-\zeta_{l m} E_{l m}^{(\mathrm{gap})} \tau} \\
& \times \rho_{m}\left(\zeta_{l m} \tau\right)_{r, r^{\prime}} \bar{\rho}_{l}\left(\zeta_{l m} \tau\right)_{r^{\prime}, r} \tag{20}
\end{align*}
$$
via the Laplace identity where the choice $h=\zeta$ generates the desired energy denominator, $1 /\left(E_{c}-E_{v}\right)$ [i.e., $F(x ; \zeta)=1 / x$ in Eq. (10)]. Each window pair $(l, m)$ has its own energy gap, $E_{l m}^{(\text {gap })}=E_{m}^{(c, \text { min })}-E_{l}^{(v, \text { max })}$, energy scale, $\zeta_{l m}$, and bandwidth, $E_{l m}^{(\mathrm{bw})}=E_{m}^{(c, \max )}-E_{l}^{(v, \min )}$. [To connect directly to the formalism of Eqs. (14) and (15), the sign of $\zeta$ has been reversed and the bar labels on the density matrices have been switched.] The imaginary time density matrices for the windows are given by
$$
\begin{align*}
\rho_{m}(\tau)_{r, r^{\prime}} & =\sum_{\{c \in \mathcal{M}\}} e^{-\tau \Delta E_{m c}} \psi_{r, c} \psi_{r^{\prime}, c}^{*}  \tag{21}\\
\bar{\rho}_{l}(\tau)_{r, r^{\prime}} & =\sum_{\{v \in \mathcal{L}\}} e^{-\tau \Delta E_{l v}} \psi_{r, v} \psi_{r^{\prime}, v}^{*} \tag{22}
\end{align*}
$$
where, again, the integer indices of the single particle states in the $m$ th conduction and $l$ th valence windows are contained in the sets, $\mathcal{M}$ and $\mathcal{L}$, respectively. Here, $\Delta E_{l v}=E_{l}^{(v, \max )}-$ $E_{v}$ and $\Delta E_{m c}=E_{c}-E_{m}^{(c, \min )}$ are defined with respect to the edges of each energy window. A good choice of windows can significantly reduce the dynamic range, i.e., the bandwidth to band gap ratio $E_{l m}^{(\text {bw })} / E_{l m}^{(\text {gap })}$, for all window pairs. This allows coarse quadrature grids to be employed to approximate the time integrals in all window pairs with controlled accuracy as given next.

\section*{2. Discrete approximation to the time integral}

The continuous imaginary time integral of Eq. (20) must be discretized in an efficient and error-controlled manner to form an effective numerical method. The natural choice is GaussLaguerre (GL) quadrature
$$
\begin{equation*}
\int_{0}^{\infty} d \tau e^{-\tau} s(\tau) \approx \sum_{u=1}^{N^{(\tau, \mathrm{GL})}} w_{u} s\left(\tau_{u}\right) \tag{23}
\end{equation*}
$$

Here, $N^{(\tau, \mathrm{GL})}$ is the number of quadrature points, the $u$ are the integer indices of the points, $s(\tau)$ is the function to be integrated over the exponential function, $\exp (-\tau)$, the $\{w\}$ and $\{\tau\}$ are the $N^{(\tau, \mathrm{GL})}$ member sets of the quadrature weights and nodes [36] whose explicit dependence on $N^{(\tau, \mathrm{GL})}$ has been suppressed for clarity. Inserting the discrete approximation, the contribution from each window pair $(l, m)$ is
$$
\begin{align*}
P_{r, r^{\prime}}^{l m}= & -2 \zeta_{l m} \sum_{u=1}^{N_{l m}^{(\tau, \mathrm{GL})}} w_{u} e^{-\tau_{u}\left(\zeta_{l m} E_{l m}^{(\mathrm{gap})}-1\right)} \\
& \times \rho_{m}\left(\zeta_{l m} \tau_{u}\right)_{r, r^{\prime}} \bar{\rho}_{l}\left(\zeta_{l m} \tau_{u}\right)_{r^{\prime}, r} \tag{24}
\end{align*}
$$
a. Optimal error-equalizing energy scale factor $\zeta_{l m}$. The energy scale factor $\zeta_{l m}$ is selected to equalize the error of all integrals in a window pair. The geometric mean, $\zeta_{l m}^{-1} \approx$ $\sqrt{E_{l m}^{(\mathrm{bw})} E_{l m}^{(\mathrm{gap})}}$, is close to the optimal error matching choice as described in Appendix A 1: the end points of the window range are treated with (nearly) equal accuracy.
b. Estimating the number of quadrature points. For any set of interband transition energies $\left\{E_{m}-E_{l}\right\}$ in window pair $(l, m)$, the largest quadrature errors occur for the largest interband transition energy $E_{l m}^{(\mathrm{bw})}$ and the smallest interband transition energy $E_{l m}^{\text {(gap) }}$. Taking $\zeta_{l m}^{-1}=\sqrt{E_{l m}^{\text {(bw) }} E_{l m}^{\text {(gap) }}}$ to balance the error across the window pair, the number of quadrature points, $N_{l m}^{(\tau, \mathrm{GL})}$, required to generate the desired fractional error level, scales as $\sim \sqrt{E_{l m}^{(\mathrm{bw})} / E_{l m}^{(\mathrm{gap})}}$ (see Appendix A). Stripping the indices for clarity, we find
$$
\begin{align*}
N^{(\tau, \mathrm{GL})}\left(\alpha ; \epsilon^{(q)}\right) & =\alpha\left(y-0.3 \ln \epsilon^{(q)}\right) \\
\alpha & =\sqrt{\frac{E^{(\mathrm{bw})}}{E^{(\mathrm{gap})}}}, \quad y=0.4 \tag{25}
\end{align*}
$$
to be a good approximation, valid for $\epsilon^{(q)}<0.135$ (see Appendix A). To extend the range to $\epsilon^{(q)}<1$, we simply set $y=1$. Importantly, the procedure ensures that $N_{l m}^{(\tau, \mathrm{GL})}$ is chosen such that time integration error for any term in a window pair has upper bound $\epsilon^{(q)}$.

\section*{3. Optimal windowing}

Given that the number of points required to generate maximal fractional quadrature error $\epsilon^{(q)}$ for a given window pair can be neatly determined, we now consider the construction of the optimal set of windows. This can be accomplished via minimization of the cost to compute the static polarizability over the number of windows, $N_{v_{w}}$ and $N_{c_{w}}$, and the associated $N_{v_{w}}$ and $N_{c_{w}}$ member sets, $\left\{E^{(v, \text { min })}, E^{(v, \text { max })}\right\}$ and $\left\{E^{(c, \min )}, E^{(c, \max )}\right\}$ of the window positions in energy space,
$$
\begin{align*}
C^{(\mathrm{GL})}\left(\epsilon^{(q)}\right)= & \sum_{l=1}^{N_{v_{w}}} \sum_{m=1}^{N_{c_{w}}} N^{(\tau, \mathrm{GL})}\left(\alpha_{l m} ; \epsilon^{(q)}\right) \\
& \times\left(\int_{E_{l}^{(v, \min )}}^{E_{l}^{(v, \max )}} D(E) d E+\int_{E_{m}^{(c, \min )}}^{E_{m}^{(c, \max )}} D(E) d E\right) \\
= & \sum_{l=1}^{N_{v_{w}}} \sum_{m=1}^{N_{c_{w}}} C_{l m}^{(\mathrm{GL})}\left(\epsilon^{(q)}\right) \tag{26}
\end{align*}
$$
which for clarity are omitted from the dependencies of $C^{(\mathrm{GL})}\left(\epsilon^{(q)}\right)$. Here, $N^{(\tau, \mathrm{GL})}\left(\alpha_{l m} ; \epsilon^{(q)}\right)$ is given in Eq. (25), and $D(E)$ is the density of states (which will be taken on additional indices when performing $k$-point sampling as given in Appendix B). The integrals over the density of states $D(E)$ are simply the number or fraction of states in the appropriate energy window.

For a density of states with problematic points, we assign windows to those regions a priori (fixed position in energy space) allowing for fast minimization over the smooth parts of $D(E)$. For example, if there is a special point in the $D(E)$ at energy $E_{\text {special }}$, a window boundary is fixed to bracket this energy $\left[E_{\text {special }}-\Delta E / 2, E_{\text {special }}+\Delta E / 2\right]$, allowing the minimization to proceed over the smoothly varying regions of the DOS integral in a Lebesgue inspired approach (i.e., the DOS is only required to be Lebesgue integrable) [35].

The cost estimator, Eq. (26), can be minimized straightforwardly, as detailed in Appendix B, once at the start of a $G W$ calculation. The computational complexity of the minimization procedure is negligible $\mathcal{O}\left(N^{0}\right)$ compared to both the $\mathcal{O}\left(N^{3}\right)$ computational complexity of both $P$ and the input band structure. We note that for the form of $N^{(\tau, \mathrm{GL})}\left(\alpha ; \epsilon^{(q)}\right)$ in Eq. (25), the optimal windowing, both the number of windows and their positions in energy, is independent of error level as $N^{(\tau, \mathrm{GL})}\left(\alpha ; \epsilon^{(q)}\right)=\alpha \cdot U\left(\epsilon^{(q)}\right)$ is separable. Importantly, all parameters of the method are now completely determined by the usual set (input band structure and a choice of energy cutoff in the conduction band) and one new parameter, $\epsilon^{(q)}$, the fractional quadrature error required to accurately transform from the time domain to the frequency domain. The quadrature error will be connected to the error in physical quantities in Sec. III.

\section*{D. Static $\boldsymbol{P}$ for gapless systems}

The standard approach employed to treat gapless systems is to introduce a smoothed step function $f(E ; \mu, \beta)$ for the electron occupation numbers as a function of energy $E$ centered on the chemical potential $\mu$ (Fermi level) with "smoothing" pa-
rameter or inverse temperature $\beta$ [37-39]. Examples include the Fermi-Dirac distribution of the grand canonical ensemble
$$
f(E)=\frac{1}{1+\exp [\beta(E-\mu)]}
$$
where formally, $\beta=1 / k_{B} T$, or the more rapidly (numerically) convergent and hence convenient
$$
f(E)=\frac{1}{2} \operatorname{erfc}(\beta(E-\mu))
$$

Typical literature values of $\beta$ correspond to temperatures above ambient conditions (e.g., $\beta^{-1}=0.1 \mathrm{eV} \approx 1000 \mathrm{~K}$ ). The static RPA irreducible polarizability matrix including the occupation functions is given in Eq. (4).

To proceed, note that the energy-dependent part of the sum in Eq. (4),
$$
\begin{equation*}
J_{c v}=\frac{f\left(E_{v}\right)-f\left(E_{c}\right)}{E_{c}-E_{v}} \tag{27}
\end{equation*}
$$
is smooth for all energies and has the finite value $-f^{\prime}(\mu)$ as $E_{v}, E_{c} \rightarrow \mu$ (note, $E_{c} \geqslant E_{v} \forall c, v$ ). Hence, for a calculation with a small but finite gap, the terms in the sum for $P$ are finite and well behaved such that windowing plus quadrature approach will work well. As before, we split $P$ into a sum over window pairs with the contributions from each window pair now given by
$$
\begin{aligned}
P_{r, r^{\prime}}^{l m}= & -2 \zeta_{l m} \sum_{u=1}^{N_{l m}^{(\mathrm{GL})}} w_{u} e^{-\tau_{u}\left(\zeta_{l m} E_{l m}^{(\mathrm{gap})}-1\right)} \\
& \times\left[S_{r^{\prime}, r}^{l m u} Q_{r, r^{\prime}}^{l m u}-T_{r^{\prime}, r}^{l m u} Z_{r, r^{\prime}}^{l m u}\right.
\end{aligned}
$$
where
$$
\begin{aligned}
S_{r, r^{\prime}}^{l m u} & =\sum_{\{v \in \mathcal{L}\}} f\left(E_{v}\right) e^{-\tau_{u} \zeta_{l m} \Delta E_{v l}} \psi_{r, v} \psi_{r^{\prime}, v}^{*} \\
Q_{r, r^{\prime}}^{l m u} & =\sum_{\{c \in \mathcal{M}\}} e^{-\tau_{u} \zeta_{l m} \Delta E_{m}} \psi_{r, c} \psi_{r^{\prime}, c}^{*} \\
T_{r, r^{\prime}}^{l m u} & =\sum_{\{v \in \mathcal{L}\}} e^{-\tau_{u} \zeta_{l m} \Delta E_{v l}} \psi_{r, v} \psi_{r^{\prime}, v}^{*} \\
Z_{r, r^{\prime}}^{l m u} & =\sum_{\{c \in \mathcal{M}\}} f\left(E_{c}\right) e^{-\tau_{u} \zeta_{l m} \Delta E_{m}} \psi_{r, c} \psi_{r^{\prime}, c}^{*}
\end{aligned}
$$

The five-index entities $S, Q, T, Z$ can be computed with $O\left(N_{v} N_{r}^{2}\right)$ or $O\left(N_{c} N_{r}^{2}\right)$ operations (i.e., cubic scaling), where $N_{r}$ is the number of $r$ grid points (see also Sec. III.C). Since $f\left(E_{c}\right)$ becomes small as a function of increasing $E_{c}$, the $T Z$ term need only be computed for the few window pairs where $\beta\left(E_{c}-\mu\right)$ is sufficiently small. Hence, the additional work required to treat gapless systems is, in fact, modest.

Direct application of the cost-optimal energy windowing method for gapped systems in Sec. II C generates infinite quadrature grids in situations where the gap is exactly zero due to degeneracy at the Fermi energy. The solution is straightforward: the key quantity that is to be represented by quadrature is $J_{c v}$ of Eq. (27). For $E_{c}-E_{v} \rightarrow 0, J_{c v} \rightarrow-f^{\prime}(\mu)$ where $-f^{\prime}(\mu)=\beta / 4$ for the Fermi-Dirac distribution and $\beta / \sqrt{2 \pi}$ for the erfc form above. Thus, the system has an effective gap of $\sim \beta^{-1}$. For energy window pairs ( $l, m$ ) that contain
degenerate states at the Fermi energy, we manually set their gap to $E_{l m}^{(\text {gap })}=1 / \beta$ via a "scissoring" operation [i.e., shifting the conduction band up by $1 /(2 \beta)$ and valence bound down by $1 /(2 \beta)]$ in the offending window pair and then applying the method of Sec. II C. Alternatively, the regularization approach of the next section can be adopted for zero-gap systems.

\section*{E. $\boldsymbol{\Sigma}(\omega)$ in cubic computational complexity}

Given poles of the screened interaction $W(\omega)_{r, r^{\prime}}, \omega_{p}$, with residues, $B_{r, r^{\prime}}^{p}$, the dynamic (frequency-dependent) part of the $G W$ self-energy can be expressed as
$$
\begin{equation*}
\Sigma(\omega)_{r, r^{\prime}}=\sum_{p, v} \frac{B_{r, r^{\prime}}^{p}\left[\psi_{r v} \psi_{r^{\prime} v}^{*}\right]}{\omega-E_{v}+\omega_{p}}+\sum_{p, c} \frac{B_{r, r^{\prime}}^{p}\left[\psi_{r c} \psi_{r^{\prime} c}^{*}\right]}{\omega-E_{c}-\omega_{p}} \tag{28}
\end{equation*}
$$
[omitting the static/bare potential term in Eq. (6) as it can be computed in $\mathcal{O}\left(N^{3}\right)$ and is, thus, not of interest here]. In the following, we develop a cubic scaling energy window-plusquadrature technique that delivers Eq. (28) directly ${ }^{1}$ for real frequencies $\omega$ in such a way that analytical continuation is not required.

\section*{1. Windowing for $\Sigma(\omega)$}

The dynamic self-energy,
$$
\begin{align*}
\Sigma(\omega)_{r, r^{\prime}} & =\Sigma^{(+)}(\omega)_{r, r^{\prime}}+\Sigma^{(-)}(\omega)_{r, r^{\prime}}, \\
\Sigma^{(+)}(\omega)_{r, r^{\prime}} & =\sum_{p, v} \frac{B_{r, r^{\prime}}^{p}\left[\psi_{r v} \psi_{r^{\prime} v}^{*}\right]}{\omega-E_{v}+\omega_{p}}, \\
\Sigma^{(-)}(\omega)_{r, r^{\prime}} & =\sum_{p, c} \frac{B_{r, r^{\prime}}^{p}\left[\psi_{r c} \psi_{r^{\prime} c}^{*}\right]}{\omega-E_{c}-\omega_{p}}, \tag{29}
\end{align*}
$$
consists of two terms, labeled ( $\pm$ ). The ( + ) term involves the valence single particle states, their shifted energies $\left(E_{v}-\omega\right)$, the plasmon residues and their modes ( $\omega_{p}$ ). The ( - ) term involves the conduction single particle states, their shifted energies ( $E_{c}-\omega$ ), the plasmon residues and their mode complement $\left(-\omega_{p}\right)$. An efficient windowed scheme requires independently decomposing the two terms as is now usual,
$$
\begin{align*}
\Sigma^{(+)}(\omega)_{r, r^{\prime}} & \left.=\sum_{m=1}^{N_{v w}^{(+)}} \sum_{l=1}^{N_{p w}^{(+)}} \Sigma^{(+)}\left(\omega ; \zeta_{l m}^{(+)}\right)_{r, r^{\prime}}^{l m}\right)  \tag{30}\\
\Sigma^{(-)}(\omega)_{r, r^{\prime}} & \left.=\sum_{m=1}^{N_{c w}^{(-)}} \sum_{l=1}^{N_{p w}^{(-)}} \Sigma^{(-)}\left(\omega ; \zeta_{l m}^{(-)}\right)_{r, r^{\prime}}^{l m}\right) \tag{31}
\end{align*}
$$
simply using the shifted single-particle energies and $\pm$ plasmon modes to define the windows. Note, $\zeta_{l m}^{(+)} \neq \zeta_{l m}^{(-)}, N_{p_{w}}^{(+)} \neq$ $N_{p_{w}}^{(-)}$and the index sets are also unique to each term, + and - . Almost all the window pairs ( $l, m$ ) in Eqs. (30) and (31) can be treated using the approach of Sec. IIC with GL quadrature

\footnotetext{
${ }^{1}$ To avoid excessive memory use, one can compute the large matrix $\Sigma(\omega)_{r, r^{\prime}}$ for a fixed $\omega$ and then compute and only store the much smaller number of desired matrix elements $\langle n| \Sigma(\omega)\left|n^{\prime}\right\rangle$ before moving to the next $\omega$ value.
}
because the denominator, $x=\omega-E_{n} \pm \omega_{p}$, is finite and does not change sign where $n=v$ for + case and $n=c$ for case. The difficulty is that, for some few window pairs, the denominator, $x$, changes sign such that the Eq. (11) does not apply. Thus a scheme to treat window pairs with energy crossings is required.

\section*{2. Specialized quadrature for energy crossings}

We treat energy window pairs $(l, m)$ with an energy crossing, where $x=\omega-E_{n} \pm \omega_{p}$ changes sign as the sum over $p$ and the generalized index, $n$, in the windows is performed, by replacing $1 / x$ by the regularized $F(x ; \zeta)$ of Eq. (9),
$$
\begin{align*}
\Sigma^{( \pm)}(\omega)_{r, r^{\prime}}^{l m}= & \sum_{\left\{p \in \mathcal{L}^{( \pm)}\right\}} \sum_{\left\{n \in \mathcal{M}^{( \pm)}\right\}} B_{r, r^{\prime}}^{p}\left[\psi_{r n} \psi_{r^{\prime} n}^{*}\right] \\
& \times F\left(\omega-E_{n} \pm \omega_{p} ; \zeta\right) \tag{32}
\end{align*}
$$
where $\zeta$ is same for all windows with a crossing. As discussed in Sec. II B, the two standard regularization strategies in the $G W$ literature are (1) to these zero contributions for small $x$ [i.e., setting $F(x ; \zeta)=0$ for small $x$ ] or (2) to use a Lorentzian smoothing function with $\zeta=-i \gamma, \gamma>0$ and $h(t ; \zeta)=\gamma e^{-\tau}$, i.e.,
$$
F(x ; \zeta)=\frac{x}{x^{2}+\gamma^{-2}}=\operatorname{Im} \int_{0}^{\infty} d \tau \gamma e^{-\tau} e^{i \tau \gamma x}
$$

Below we shall eschew $\zeta$ and work in terms of $\gamma$ which is more natural.

As detailed in Appendix C, a better choice for the weight function and resulting transform are
$$
\begin{align*}
& h(\tau ; \gamma)=\gamma \exp \left(-\tau-\tau^{2} / 2\right) \\
& F(x ; \gamma)=\gamma \operatorname{Im}\left\{\sqrt{\frac{\pi}{2}} e^{-\frac{(x \gamma+i)^{2}}{2}}\left[1+i \operatorname{erfi}\left(\frac{x \gamma+i}{\sqrt{2}}\right)\right]\right\} \tag{33}
\end{align*}
$$

The new weight has a transform that both approaches $1 / x$ faster than the Lorentzian in the large $x$ limit (see Appendix C), and is regular for all $x$. In addition, its transform can be accurately computed via time integration with fewer quadrature points than required by weight that leads to the Lorentzian (i.e., the pure exponential function).

A Gaussian-type quadrature for the new weight function can be generated following the standard procedure [40] to create a set of nodes $\{\tau\}$ and weights $\{w\}$ for a given quadrature grid size $N^{(\tau, \text { HGL })}$ (see Appendix H). The superscript HGL denotes Hermite-Gauss-Laguerre quadrature since the weight function has both linear and quadratic terms in the exponent. Inserting the result, the discrete approximation becomes
$$
\begin{align*}
F(x ; \gamma) & \approx \gamma I m \sum_{u=1}^{N^{(\tau, \mathrm{HGL})}} w_{u} e^{i \tau_{u} x \gamma} \\
& \approx \gamma \sum_{u=1}^{N^{(\tau, \mathrm{HGL})}} w_{u} \sin \left(\tau_{u} x \gamma\right) \tag{34}
\end{align*}
$$

Finally, for the window pairs $(l, m)$ with an energy crossing
$$
\begin{align*}
\Sigma^{( \pm)}(\omega)_{r, r^{\prime}}^{l m}= & \gamma \sum_{u=1}^{N_{l m}^{(\tau, \text { HGL }, \pm)}} w_{u}\left\{\left[\sum_{\left\{p \in \mathcal{L}^{( \pm)}\right\}} B_{r, r^{\prime}}^{p} \sin \left( \pm \tau_{u} \omega_{p} \gamma\right)\right]\right. \\
& \times\left[\sum_{\left\{n \in \mathcal{M}^{( \pm)}\right\}} \psi_{r n} \psi_{r^{\prime} n}^{*} \cos \left(\tau_{u}\left(\omega-\epsilon_{n}\right) \gamma\right)\right] \\
& +\left[\sum_{\{p \in \mathcal{L}\}} B_{r, r^{\prime}}^{p} \cos \left( \pm \tau_{u} \omega_{p} \gamma\right)\right] \\
& \left.\times\left[\sum_{\left\{n \in \mathcal{M}^{( \pm)}\right\}} \psi_{r n} \psi_{r^{\prime} n}^{*} \sin \left(\tau_{u}\left(\omega-\epsilon_{n}\right) \gamma\right)\right]\right\} \tag{35}
\end{align*}
$$
which is separable and can be computed in $\mathcal{O}\left(N^{3}\right)$. Again, one value of broadening parameter $\gamma$ is selected for all windows with energy crossings. The parameter $\gamma$ is a convergence parameter taken to be as small as possible without effecting results. The number of grid points will vary depending on the bandwidth in the window pair scaled by $\gamma$ and the desired fractional error.

\section*{3. Quadrature points for specified error level}

For window pairs without an energy crossing, $\omega-E_{n} \pm$ $\omega_{p}$ does not change sign, and the GL quadrature previously analyzed is utilized (the general subscript is $n$ is used to denote that either $c$ or $v$ states are possible). For window pairs with energy crossings, the HGL quadrature is required. Appendix D details the construction of $N^{(\tau, \mathrm{HGL})}\left(x ; \epsilon^{(q)}\right)$,
$$
\begin{equation*}
N^{(\tau, \mathrm{HGL})}\left(x ; \epsilon^{(q)}\right)=c_{2}\left(\epsilon^{(q)}\right) x^{2}+c_{1}\left(\epsilon^{(q)}\right) x+c_{0}\left(\epsilon^{(q)}\right), \tag{36}
\end{equation*}
$$
where $x=\gamma\left(E_{\max }-E_{\min }\right)$ is the bandwidth of the window pair with energy crossings (scaled by $\gamma$ ), and $c_{2}, c_{1}$, and $c_{0}$ are low order polynomial functions of $\ln \epsilon^{(q)}$. The values of the coefficients are given in Appendix D.

\section*{4. Optimal window choice}

We now consider the computational cost to compute $\Sigma(\omega)$ for window pairs with an energy crossing,
$$
\begin{aligned}
C_{l m}^{(\mathrm{HGL})}\left(\epsilon^{(q)}\right)= & 2 N_{l m}^{(\tau, \mathrm{HGL})}\left(x_{l m} ; \epsilon^{(q)}\right) \\
& \times\left(\int_{\omega_{m}^{(p, \text { min })}}^{\omega_{m}^{(p, \text { max })}} D^{(p)}(\omega) d \omega+\int_{E_{l}^{(n, \text { min })}}^{E_{l}^{(n, \text { max })}} D(E) d E\right)
\end{aligned}
$$

Here, the $m$ th plasmon mode window spans the energy range $\left[\omega_{m}^{(p, \min )}, \omega_{m}^{(p, \max )}\right]$, the $l$ th band energy window spans the energy range $\left[E_{l}^{(n, \text { min })}, E_{l}^{(n, \text { max })}\right]$, the density of plasmon modes is $D^{(p)}(\omega)$ and the density of band states is $D(E)$. (The explicit dependence of the cost function on the window edges is, again, suppressed.) The parameter $x_{l m}$ is $x_{l m}=\gamma E_{l m}^{(\mathrm{bw})}$ where $\Delta E_{l m}^{(\mathrm{bw})}$ is the absolute value of the maximum energy difference between the single particle and plasmon modes in the window pair. Although potentially discontinuous as the window ranges evolve during minimization, the insertion does not prevent rapid numerical convergence of the cost function
to its minimum value. Further discussion can be found in Appendix E.

\section*{F. Cubic scaling $\boldsymbol{P}(\boldsymbol{\omega})$}

The energy window plus time integral quadrature methods developed to compute the static $P$ and the dynamic $\Sigma(\omega)$ can be applied directly and without modification to the computation of the frequency-dependent polarizability $P(\omega)$ of Eq. (3) with $\mathcal{O}\left(N^{3}\right)$ computational effort. The key observation is that $P(\omega)$ can be written as the sum of two simple energy denominator poles:
$$
\begin{align*}
P(\omega)_{r, r^{\prime}}= & \sum_{c, v, \sigma, \sigma^{\prime}}\left[\psi_{x, c} \psi_{x^{\prime}, c}^{*}\right]\left[\psi_{x^{\prime}, v} \psi_{x, v}^{*}\right] \\
& \times\left(\frac{1}{\omega-\left(E_{c}-E_{v}\right)}-\frac{1}{\omega+\left(E_{c}-E_{v}\right)}\right) \tag{37}
\end{align*}
$$

Since $P(\omega)=P(-\omega)$, we need only focus on $P(\omega)$ for $\omega>0$. The second energy denominator $\omega+E_{c}-E_{v}$ is always positive definite since $E_{c}-E_{v} \geqslant 0$ and can be evaluated in $\mathcal{O}\left(N^{3}\right)$ with the same GL quadrature methodology developed for evaluating the static $P$ in Sec. II C; the presence of $\omega>0$ in the second denominator enlarges the effective energy gap and enhances convergence of our method. The first energy denominator $\omega-\left(E_{c}-E_{v}\right)$ can change sign once $\omega$ is larger than the energy gap. However, this term can be evaluated with $\mathcal{O}\left(N^{3}\right)$ effort using the energy crossing quadrature/regularization method developed for $\Sigma(\omega)$ in Sec. II E.

\section*{III. RESULTS: STANDARD BENCHMARKS}

Here, the application of the new CTSP method to standard benchmark systems is presented. Results for the optical dielectric constant and the energy band gap within the COHSEX approximation are given for crystalline silicon ( Si ) and magnesium oxide (MgO). Next, studies of the static polarization of crystalline Al, a gapless systems, are presented. Last, a $G_{0} W_{0}$ computation of the band gap of crystalline Si is given.

\section*{A. Optical dielectric constant and COHSEX band gap}

In order to evaluate the performance of the new reduced order method, CTSP, we study two standard benchmark materials: Si and MgO . We first perform plane wave pseudopotential DFT calculations for both materials to generate the DFT band structure and then employ the results in the reported $G W$ computations. Appendix G contains the details of the DFT and $G W$ calculations.

Silicon is a prototypical three-dimensional covalent crystal (diamond structure) with a moderate band gap ( 0.5 eV in DFTLDA) while rocksalt MgO is an ionic crystal with a relatively large gap ( 4.4 eV in DFT-LDA). To judge the performance of CTSP, the convergence of two basic observables are studied: the macroscopic optical dielectric constant $\epsilon_{\infty}$ and the band gap within the COHSEX approximation to the self-energy [8].

Figure 3 shows the error in $\epsilon_{\infty}$ as a function of the computational savings achieved by both CTSP-W and CTSP$1 \mathcal{O}\left(N^{3}\right)$ techniques, and the $\mathcal{O}\left(N^{3}\right)$ interpolation method described in Appendix F, relative to the standard $\mathcal{O}\left(N^{4}\right)$ method for 16 atom (periodic) supercells of MgO and Si .