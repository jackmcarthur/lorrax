\title{
Mixed Stochastic-Deterministic Approach for Many-Body Perturbation Theory Calculations
}

\author{
Aaron R. Altman®, ${ }^{1}$ Sudipta Kundu®, ${ }^{1}$ and Felipe H. da Jornada® ${ }^{1,2, *}$ \\ ${ }^{1}$ Department of Materials Science and Engineering, Stanford University, Stanford, California 94305, USA \\ ${ }^{2}$ Stanford Institute for Materials and Energy Sciences, SLAC National Accelerator Laboratory, Menlo Park, California 94025, USA
}
(Received 7 April 2023; revised 24 August 2023; accepted 5 December 2023; published 20 February 2024)

\begin{abstract}
We present an approach for $G W$ calculations of quasiparticle energies with quasiquadratic scaling by approximating high-energy contributions to the Green's function in its Lehmann representation with effective stochastic vectors. The method is easy to implement without altering the $G W$ code, converges rapidly with stochastic parameters, and treats systems of various dimensionality and screening response. Our calculations on a $5.75^{\circ}$ twisted $\mathrm{MoS}_{2}$ bilayer show how large-scale $G W$ methods include geometry relaxations and electronic correlations on an equal basis in structurally nontrivial materials.
\end{abstract}

DOI: 10.1103/PhysRevLett.132.086401

Many-body perturbation theory (MBPT) within the firstprinciples $G W$ approximation is a proven and widespread method for computing accurate quasiparticle (QP) properties of materials [1-5]. Obtaining fully converged QP energies within the $G W$ approach for small to moderately sized bulk systems, with up to a few hundred atoms in the unit cell, is a routine procedure with modern highperformance supercomputers [6-9]. However, exploring the more complex many-body physics of large systems that are relevant in electronic and technological applications is difficult due to the quartic scaling in system size of the standard $G W$ formalism [2], limiting the applicability of the method in large-scale problems, such as those involving twisted materials displaying moiré physics [10-17].

Several approaches have been developed recently to deal with these shortcomings. They fall mainly into two categories: modifying the standard reciprocal-space $G W$ formalism, wherein the electronic Green's function $G$ is still evaluated in its Lehmann representation as a sum over bands, or employing different representations of the theory that avoid the explicit sum over bands. Notable techniques in the former category include replacing high-energy orbitals with simple ansatz wave functions and using completion relations to truncate the sum over bands [1822]. Approaches in the latter category are diverse, with several achieving cubic or subcubic scaling. An important technique is transforming to bases where the evaluation of the polarizability is formally cubic scaling. This includes working in real space and imaginary time where the polarizability is separable [ 23,24 ], manipulating the spectral functions in a localized-orbital basis [25], exploiting sparse overlap integrals in a Gaussian basis [26], and using tensor hypercontraction [27] and density-fitting methods [28-30]. Independent of these cubic scaling methods, stochastic approaches [31-35] can achieve linear scaling with system size by working in the time domain.

Additionally, there are representations of the theory that still scale quarticly but exhibit lower prefactors, such as within the framework of density-functional perturbation theory [36,37].

In this Letter, we propose a simple and rigorous approach that combines the stochastic and sum-over-bands methods to achieve a quasiquadratic scaling $G W$ formalism with a small prefactor (speedups of $\sim 100$-fold on systems with tens of atoms), from given input mean-field wave functions. It offers large computational savings in both the calculation of the dielectric function and the QP self-energy. The performance gain is achieved by the stochastic compression of all mean-field Kohn-Sham states outside a small-energy region around the Fermi level [38,39], including occupied states.

Our approach is compatible with standard reciprocalspace $G W$ codes and is simple to implement. It is also straightforward to converge independent of the $G W$ code that uses it and eliminates sum-over-bands truncation parameters in the $G W$ calculation by allowing one to include all eigenstates from the mean-field Hamiltonian. These advantages allow the computation of QP properties of complex systems of hundreds of atoms with moderate computational expense, which we demonstrate for several systems. Finally, unlike purely stochastic approaches, we observe speedups with respect to a fully deterministic approach for all system sizes, not only for large systems. We highlight the applicability of our method on several systems of different dimensionality, including a large-scale problem of a $5.75^{\circ}$ twisted bilayer of $\mathrm{MoS}_{2}$.

Method.-The GW approximation in its most common non-self-consistent form is based on the noninteracting single-particle Green's function,
$$
\begin{equation*}
G(\omega) \equiv \sum_{n, \mathbf{k}} \frac{\left|\phi_{n \mathbf{k}}\right\rangle\left\langle\phi_{n \mathbf{k}}\right|}{\omega-E_{n \mathbf{k}} \mp i \eta} \equiv \sum_{\mathbf{k}} G_{\mathbf{k}}(\omega), \tag{1}
\end{equation*}
$$
where $\left|\phi_{n \mathbf{k}}\right\rangle$ are mean-field states, typically obtained from density-functional theory (DFT) calculations, with band index $n$ and wave vector $\mathbf{k} ; E_{n \mathbf{k}}$ are the corresponding eigenenergies, $\omega$ is the evaluation frequency, $\eta=0^{+}$, and where the sign is negative (positive) when $E_{n \mathbf{k}}$ is below (above) the Fermi energy. As in other stochastic approaches to $G W$ calculations [31], our method is based on the stochastic resolution of the identity operator, $\lim _{N \rightarrow \infty} N^{-1} \sum_{i=1}^{N}\left|\zeta_{i}\right\rangle\left\langle\zeta_{i}\right|=\mathbb{1}$, where off diagonals vanish with a standard deviation of $1 / \sqrt{N}$, and $\left|\zeta_{i}\right\rangle$ are random vectors [see Eq. (5)].

When computing electronic properties within MBPT, it is important to accurately capture the pole structure of $G$ close to the Fermi energy. For instance, the noninteracting polarizability matrix $\chi_{\mathbf{G}, \mathbf{G}^{\prime}}^{0}(\mathbf{q}, \omega)$ at a wave vector $\mathbf{q}$ and plane wave indices $\mathbf{G}$ and $\mathbf{G}^{\prime}$ has poles at frequencies corresponding to the energy difference between conduction ( $c$ ) and valence ( $v$ ) states, $\chi_{\mathbf{G}, \mathbf{G}^{\prime}}^{0}(\mathbf{q}, \omega) \sim \sum_{v c \mathbf{k}} A_{\mathbf{G}, \mathbf{G}^{\prime}}^{v c}(\mathbf{k}, \mathbf{q})[\omega \pm \left.\left(E_{c \mathbf{k}}-E_{v \mathbf{k}}\right)\right]^{-1}$, where $A_{\mathbf{G}, \mathbf{G}^{\prime}}^{v c}$ are matrix elements. Accurately describing the low-frequency behavior of $\chi^{0}$ is critical in MBPT calculations. This depends sensitively on the pole structure of $G$ close to the Fermi energy, but less so on the pole structure of $G$ at farther frequencies. For instance, when evaluating the electronic self-energy $\Sigma^{G W}$ within the con-tour-deformation approach [40-42], $\Sigma^{G W}$ depends on an integral of the screened Coulomb interaction $W$ along the imaginary frequency axis-for which the pole structure of $\chi^{0}$ gets smoothed out-plus residues of $W$ are typically evaluated at energies close to the Fermi energy.

This motivates us to express $G$ as one term that contains the exact contributions to the pole structure close to the Fermi energy $G_{\mathbf{k}}^{P}$ and another contribution that we write as a sum over $N_{S}$ subspaces that are farther from the Fermi energy $G_{\mathbf{k}}^{S}$,
$$
\begin{equation*}
G_{\mathbf{k}}(\omega) \approx G_{\mathbf{k}}^{P}(\omega)+\sum_{S}^{N_{S}} G_{\mathbf{k}}^{S}(\omega) . \tag{2}
\end{equation*}
$$
$G_{\mathbf{k}}^{P}$ is computed exactly within Eq. (1) for bands $n \in P$, where $P$ is a small protected subspace with $N_{P}$ bands closest to the Fermi energy. This deterministic region contains the states of interest for which QP properties are desired, though it is unnecessary when computing only the polarizability [43]. The remaining subspaces $S$ are still required for accurately expressing the self-energy $\Sigma^{G W}$, but their pole structure may be approximated. For each subspace $S$, we first approximate the near-continuum pole distribution or branch cut at $\left\{E_{n \mathbf{k}}\right\}$ for states $n \in S$ with a single pole at an average energy $\bar{E}_{S \mathbf{k}}$. Next, we identify the $\operatorname{sum} \sum_{n \in S}\left|\phi_{n \mathbf{k}}\right\rangle\left\langle\phi_{n \mathbf{k}}\right|$ as a projection onto the subspace $S$. This projection can be compressed using the stochastic resolution of the identity operator,

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/9a69bcd8-e997-4a8b-ac40-f2d5b253e4ed-2.jpg?height=400&width=860&top_left_y=163&top_left_x=1091}
\captionsetup{labelformat=empty}
\caption{FIG. 1. (a) Diagram of the method's band-partitioning scheme. $E_{\text {min }}$ is the energy of the deepest valence state. (b),(c) Comparison of the error in the QP energies for $G W$ calculations performed with a traditional deterministic approach and using stochastic pseudobands for (b) an isolated benzene molecule and (c) bulk wurtzite ZnO . Stochastic pseudobands reach converged QP energies (within $10-100 \mathrm{meV}$ ) with fewer total bands than a deterministic truncation of the Hilbert space. Pseudobands parameters are (b) $N_{P}^{c}=50$ and (c) $N_{P}^{c}=10$; both (b) and (c) used $N_{\xi}^{c}=2$ and $N_{S}^{c}=\{1,5,50,250\}$ (corresponding to $\left.\mathcal{F}^{c}=\{1.9,0.42,0.054,0.015\}\right)$. Pseudobands were not used to compress valence states.}
\end{figure}
$$
\begin{equation*}
G_{\mathbf{k}}^{S}(\omega) \approx \frac{1}{\omega-\bar{E}_{S \mathbf{k}} \mp i \eta} \sum_{i=1}^{N_{\xi}}\left|\xi_{i, \mathbf{k}}^{S}\right\rangle\left\langle\xi_{i, \mathbf{k}}^{S}\right|, \tag{3}
\end{equation*}
$$
where $\left|\xi_{i, \mathbf{k}}^{S}\right\rangle$ are vectors that stochastically project any vector onto the subspace $S$ of interest, and which we denote by "stochastic pseudobands." Note that the subspaces $S$ can run over both unoccupied and occupied states [see Fig. 1(a)].

The number of stochastic pseudobands $N_{\xi}$ is a convergence parameter and controls the stochastic error of the resolution of the identity. The number of subspaces $N_{S}$ is also a convergence parameter and controls the error of the average energy approximation. In the limit $N_{S}, N_{\xi} \rightarrow \infty$, we recover the original Green's function in Eq. (1). The partition Eq. (2) is a stochastic-deterministic approach and allows us to maintain high accuracy for important states close to the Fermi energy, while compressing states that are less relevant. Our approach is similar in spirit to other stochastic methods for $G W$ calculations [32-35], but does not require propagation in real time.

Next, we show how to partition the subspaces $\{S\}$ in Eq. (2) and construct each stochastic pseudoband $\left|\xi_{i, \mathbf{k}}^{S}\right\rangle$. A practical approach is to enforce that the error from each subspace to the Green's function or static polarizability matrix $\chi_{\mathbf{G}, \mathbf{G}^{\prime}}^{0}(\mathbf{q}, \omega=0)$ is roughly constant. This is achieved by enforcing a constant ratio
$$
\begin{equation*}
\mathcal{F} \equiv \frac{\Delta E_{S}}{\bar{E}_{S}}=\mathrm{const} \tag{4}
\end{equation*}
$$
where $\bar{E}_{S}$ is the average energy of the Kohn-Sham states in each subspace $S$ (referenced to the Fermi level) and $\Delta E_{S}$ is the energy range spanned by $S$ [43]. The ratio
$\mathcal{F}$ is inversely proportional to the number of subspaces, $\mathcal{F} \sim 1 / N_{S}$.

Finally, for each subspace $S$, we construct stochastic pseudobands by taking random linear combinations of Kohn-Sham states in $S$,
$$
\begin{equation*}
\left|\xi_{i, \mathbf{k}}^{S}\right\rangle=\frac{1}{\sqrt{N_{\xi}}} \sum_{n \in S} \alpha_{i, n \mathbf{k}}^{S}\left|\phi_{n \mathbf{k}}\right\rangle \tag{5}
\end{equation*}
$$
with random phases $\alpha=e^{2 \pi i \theta}$ for random $\theta \in[0,1)$, where $i \in\left\{1, \ldots, N_{\xi}\right\}$ are the different stochastic pseudobands that realize the projection onto $S$.

The proposed stochastic compression can be easily implemented in most MBPT codes that use a spectral representation of $G$, and we have implemented our developmental version in the BerkeleyGW code [44]. One only needs to modify the input Kohn-Sham orbitals and combine them according to Eq. (5). In particular, no modification of the $G W$ code is required: the pseudobands approach is a pre-processing step to the $G W$ calculation. We also provide a pseudocode [43] and reference implementation [61]. The method as described here focuses on compressing the Green's function for the efficient evaluation of the static dielectric function, which is the quantity of interest in calculations that use plasmon-pole models. Still, a simple extension, whereby one takes $\Delta E_{S}$ to be a constant ( $\Delta E$ ) instead of a quantity proportional to $\bar{E}_{S}$, allows the evaluation of the inverse dielectric function at arbitrary frequencies with small statistical errors and large computational savings [43]. We stress that our approach is amenable to compressing both valence and conduction states, offering especially large speedups for the computation of the dielectric function, which scales with their product. We summarize the quantities introduced in Table I below.

We note that (1) convergence testing with respect to pseudobands parameters is rarely required, as the typical values listed in Table I were sufficient to converge all systems studied and (2) $N_{P}$ only needs to be large enough to include the states of interest for computing the electronic self-energy and can be zero for computing only the

\begin{table}
\captionsetup{labelformat=empty}
\caption{TABLE I. Pseudobands parameters: conv., auto., and aux. are convergence, automatically determined, and auxiliary parameters, respectively. $N_{P}$ should be zero when evaluating the polarizability for large systems and finite to evaluate the $G W$ selfenergy of deterministic states.}
\begin{tabular}{llll}
\hline \hline Parameter & \multicolumn{1}{c}{ Description } & \begin{tabular}{c} 
Typical \\
value
\end{tabular} \\
\hline $\mathcal{F}$ & Conv. & Constant energy ratio Eq. (4) & $1 \%-2 \%$ \\
$N_{S}$ & Auto. & Number of stochastic subspaces, $N_{S} \propto(1 / \mathcal{F})$ & $10-200$ \\
$N_{\xi}$ & Conv. & Number of pseudobands per subspacé & $2-3$ \\
$N_{P}$ & Aux. & Number of protected bands & $\geq 0$ \\
\hline \hline
\end{tabular}
\end{table}
dielectric matrix for both semiconductors and metals [43]. Additionally, our approach removes the band truncation parameters employed in the sum over states in traditional $G W$ calculations. This is because, when constructing the stochastic pseudobands, we can easily consider all bands from the mean-field Hamiltonian by diagonalizing it with scalable linear algebra packages such as elpa [45]. While one can benefit from similar speedups from our pseudobands approach when generating input Kohn-Sham states with iterative solvers, directly diagonalizing the DFT Hamiltonian is typically faster and more numerically stable [43].

Results.-We benchmark our stochastic pseudobands approach on systems spanning dimensionality, electronic structure, and screening environment to numerically verify its convergence behavior. We demonstrate quasiquadratic scaling for $G W$ calculations on ZnO supercells up to 256 atoms while maintaining constant error. Finally, we perform a large-scale calculation of the $G W \mathrm{QP}$ band structure of a $5.75^{\circ}$ twisted $\mathrm{MoS}_{2}$ moiré bilayer to address questions regarding the emergent electronic structure in twisted 2D materials. Computational details are provided in the Supplemental Material [43]. Specific pseudobands convergence parameters are listed with the computations below ( $v$ and $c$ superscripts indicate pseudoband parameters used for valence and conduction states, respectively). Regardless of the system, we note that $N_{\xi} \geq 2$ should be used, as $N_{\xi}=1$ does not resolve the projection over each subspace. Additionally, as currently implemented, compressing valence states with stochastic pseudobands does not offer advantages for calculating the self-energy operator (as opposed to the dielectric matrix). This is because the bare exchange contribution to the self-energy $\Sigma^{X}$, which involves matrix elements with occupied states, is very sensitive to the character of the valence wave functions. Since the calculation of the self-energy only scales with the sum of the valence and conduction bands, compressing valence states does not provide significant acceleration for $\Sigma^{G W}$ in any case. However, stochastic pseudobands always provide speedups when compressing the conduction states for the operators studied here.

Convergence behavior.-We show systematic convergence of QP energies for two systems, an isolated benzene molecule and bulk wurtzite ZnO . Additional benchmarks on bilayer $\mathrm{MoS}_{2}$ and a metallic $\mathrm{Ag}_{54} \mathrm{Pd}$ nanoparticle are presented in the Supplemental Material [43]. Figure 1 summarizes our approach by comparing the error in QP energies for an isolated benzene molecule and for wurtzite ZnO with respect to the number of bands $N_{b}$ included in the MBPT calculations-both in the summations to evaluate the dielectric matrix and self-energy. For each value of $N_{b}$, we include either the lowest $N_{b}$ Kohn-Sham orbitals, in the deterministic case, or both a set of $N_{P}$ Kohn-Sham states in the protected region plus $N_{S} N_{\xi}$ stochastic pseudobands, such that $N_{b}=N_{P}+N_{S} N_{\xi}$. Hence, our tests assess
whether, for a fixed computational effort, stochastic pseudobands yield more accurate QP energies by approximating the high-energy part of the Hilbert space that gets truncated in deterministic calculations. Figure 1(b) shows the root mean square error $\sqrt{N^{-1} \sum_{n}^{N}\left(E_{n}^{\mathrm{QP}}-E_{n}^{\text {ref }}\right)^{2}}$ over 19 QP levels around the Fermi energy of benzene for both the deterministic calculation and pseudobands. Figure 1(c) shows the error of the band gap $\left|E_{\text {gap }}^{\mathrm{QP}}-E_{\text {gap }}^{\text {ref }}\right|$ of ZnO , again comparing both the deterministic calculations and those using stochastic pseudobands. In both cases, $E^{\text {ref }}$ is obtained from a highly converged deterministic calculation -utilizing 30000 bands for benzene and 10000 bands for ZnO .

For both materials, stochastic pseudobands outperform the deterministic results by $10-100$-fold in error for the same computational effort for all but the least converged calculations. Conversely, we find that, to achieve the same error, the deterministic calculation requires approximately 10-100 times as many bands as used in stochastic pseudobands calculations. We see rapid and systematic convergence behavior for all systems studied.

Scaling and computational cost.-In addition to the good convergence behavior, utilization of stochastic pseudobands also significantly improves the computational scaling of the $G W$ approach with system size. Traditionally, the calculation of the dielectric matrix consists of two primary computationally demanding steps: constructing the noninteracting polarizability matrix $\chi_{\mathbf{G}, \mathbf{G}^{\prime}}^{0}$, which scales as $\mathcal{O}\left(N_{\mathbf{G}}^{2} N_{c} N_{v}\right) \sim \mathcal{O}\left(N^{4}\right)$, and then inverting the RPA dielectric matrix $\epsilon_{\mathbf{G}, \mathbf{G}^{\prime}}$, which scales as $\mathcal{O}\left(N_{\mathbf{G}}^{3}\right) \sim \mathcal{O}\left(N^{3}\right)$, where $N_{v}$ and $N_{c}$ are the numbers of valence and conduction bands, and $N_{\mathbf{G}}$ and $N$ are the number of reciprocal-lattice vectors and the overall system size, respectively. With stochastic pseudobands, the cost to compute the noninteracting polarizability is $\mathcal{O}\left(N_{\mathbf{G}}^{2} N_{S}^{2} N_{\xi}^{2}\right) \sim \mathcal{O}\left(N^{2}\right)$, since one can always take $N_{P}=0$. It still takes $\mathcal{O}\left(N^{3}\right)$ to invert the dielectric matrix, although that cost can be reduced with low-rank techniques [46,62,63], making the $G W$ workflow quasiquadratic. Our results show that the computational savings are insensitive to the details of $N_{P}$ (see Fig. 2). Additionally, from Eq. (4), the total number of states when utilizing pseudobands is roughly the logarithm of the initial number of states, yielding a significant reduction in the number of states used in the MBPT calculations and a low algorithmic prefactor. Moreover, due to the high performance of dis-tributed-memory linear algebra solvers, we find that the inversion of the dielectric matrix is only a significant bottleneck for large systems, with hundreds to thousands of atoms in the unit cell. In fact, for the largest system we studied of $5.75^{\circ}$ twisted bilayer $\mathrm{MoS}_{2}$, inversion took only $14 \%$ of the total run time.

Figure 2 shows the computational scaling for calculating a well-converged dielectric matrix for ZnO , with a plane wave cutoff of 80 Ry , where we consider systematically larger

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/9a69bcd8-e997-4a8b-ac40-f2d5b253e4ed-4.jpg?height=398&width=858&top_left_y=163&top_left_x=1091}
\captionsetup{labelformat=empty}
\caption{FIG. 2. Scaling curve for the dielectric computation per $\boldsymbol{q} / \boldsymbol{k}$ point for ZnO supercells showing quasiquadratic behavior up to 256 atoms. Band gap errors are maintained at $<50 \mathrm{meV}$ for constant convergence parameters $N_{\xi}, \mathcal{F}$ [43].}
\end{figure}
supercells containing from 8 to 256 atoms [43]. To make these calculations feasible, it was critical to use our approach wherein both valence and conduction states away from the Fermi energy are compressed into stochastic pseudobands. All supercells exhibited constant error $<50 \mathrm{meV}$ when we performed subsequent self-energy calculations of the quasiparticle energies with unchanged convergence parameters $\left(N_{\xi}^{v}=4, N_{\xi}^{c}=2, \mathcal{F}^{v / c}=0.02\right) . \quad N_{P}^{v / c}$ was chosen to be 20 for the $2 \times 1 \times 1$ supercell and scales with the system size to allow for the evaluation of the self-energy within the same energy window [43]. Even with a nonzero $N_{P}$, we find that the approach displays, in practice, a quasiquadratic scaling for the evaluation of the dielectric function and self-energy.

Application to large systems.-Moiré bilayers such as twisted bilayer graphene or transition metal dichalcogenides (TMDs) have been at the research forefront for investigating correlated electronic phases in condensed matter systems [10,64-68]. Semiconducting TMD moiré bilayers have gained additional interest as hosts of different types of emergent excitons for possible applications in optoelectronic and exciton-based qubit devices [69-79]. A correct description of the QP properties is often a prerequisite to understanding these emergent phenomena, but the large system size and variations of the dielectric function [80,81] requiring fine Brillouin zone (BZ) samplings [82] has made them difficult to study with firstprinciples $G W$ calculations.

Using the pseudobands approach, we perform explicit $G W$ calculations on a $5.75^{\circ}$ twisted bilayer of $\mathrm{MoS}_{2}$ containing 546 atoms in the moiré supercell [Fig. 3(a)] and further unfold the moiré band structure to the unit cell of the high-symmetry, $0^{\circ}$ twisted bilayer structure [83,84], known as the 3 R stacking [red dots in Fig. 3(b)]. We compare such large-scale $G W$ calculations to DFT calculations performed directly on the 3R structure [blue lines in Fig. 3(b)] and find differences in the band gap, relative energy splitting, and band ordering.

We rationalize these differences through contributions from moiré and quasiparticle effects. To understand moiré effects, we perform DFT calculations on the twisted

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/9a69bcd8-e997-4a8b-ac40-f2d5b253e4ed-5.jpg?height=589&width=858&top_left_y=163&top_left_x=176}
\captionsetup{labelformat=empty}
\caption{FIG. 3. (a),(c) $G W$ (computed with stochastic pseudobands) and DFT band structure for the $5.75^{\circ}$ twisted moiré system in the moiré BZ . (b) Band structure for the untwisted, 3 R -stacked $\mathrm{MoS}_{2}$ bilayer (solid lines) and the corresponding unfolded band structure of the $5.75^{\circ}$ twisted moiré system (dots), at the DFT and $G W$ levels. The weight of the projection represents the contribution of the unit cell state to the corresponding moiré state at the same energy. Inset above (c): BZs of the twisted and untwisted structures.}
\end{figure}
structure [Fig. 3(c)] and unfold the resulting band structure onto the BZ of the 3R structure [blue dots in Fig. 3(b)]. Compared to direct DFT calculations on the 3 R structure, the DFT calculations on the $5.75^{\circ}$ twisted system display a larger energy splitting between the first two valence states at $\Gamma$. This is expected since these states originate from the interlayer chalcogen interactions, which change with stacking and twist angle. Next, to capture quasiparticle effects, we perform $G W$ calculations on the 3R structure [red lines in Fig. 3(b)]. Compared again to the DFT calculation on the 3R structure, QP effects mainly increase the band gap and reorder the conduction states at the $K$ and $\Lambda$ valleys. Our calculations highlight that moiré effects and quasiparticle corrections both play significant roles in twisted materials, need to be accounted for on the same footing [17], and are additive here to within 100 meV [43].

Conclusion.-We present a mixed stochastic-deterministic approach for $G W$ calculations. Given input mean-field states, the method displays quasiquadratic scaling for the tests performed up to 256 atoms, $\sim 100$-fold speedups for systems of tens of atoms, and smooth convergence behavior. The usage of stochastic pseudobands is compatible with systems of any dimension and nontrivial screening environments, and extends standard MBPT codes to handle systems of several hundreds of atoms with moderate computational expense. We envision that, beyond further studies on moiré systems, structurally large and technologically relevant systems such as interfaces, surfaces, and extended defects can be studied with this approach to address fundamental questions involving the interplay
between nonlocal screening environments and self-energy effects.

Supplemental datasets are openly available from the Zenodo repository [61].
A. R. A. acknowledges helpful discussions with Mauro Del Ben, Johnathan D. Georgaras, and Emma M. Simmerman. This work was primarily supported by the Center for Computational Study of Excited-State Phenomena in Energy Materials (C2SEPEM), funded by the U.S. Department of Energy (DOE), Office of Basic Energy Sciences (BES) under Award No. DE-AC0205CH11231 at the Lawrence Berkeley National Laboratory (LBL), as part of the Computational Materials Sciences Program. The study of twisted $\mathrm{MoS}_{2}$ was supported by the U.S. DOE BES Award No. DE-SC0021984. This research used resources of the National Energy Research Scientific Computing Center (NERSC), a U.S. DOE Office of Science User Facility located at LBL, operated under Contract No. DE-AC02-05CH11231 using NERSC Grant No. BES-ERCAP m3606, and from the Texas Advanced Computing Center (TACC) at The University of Texas at Austin, funded by the National Science Foundation (NSF) Grant No. 1818253, through allocation DMR21077 for the development of algorithms. Large-scale calculations used resources of the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory, which is supported by the U.S. DOE BES under Award No. DE-AC05-00OR22725.
*jornada@stanford.edu
[1] L. Hedin and S. Lundqvist, in Solid State Physics (Elsevier, New York, 1970), Vol. 23, pp. 1-181.
[2] M. S. Hybertsen and S. G. Louie, Phys. Rev. B 34, 5390 (1986).
[3] G. Onida, L. Reining, and A. Rubio, Rev. Mod. Phys. 74, 601 (2002).
[4] M. L. Cohen and S. G. Louie, Fundamentals of Condensed Matter Physics (Cambridge University Press, Cambridge, England, 2016).
[5] S. G. Louie and M. L. Cohen, Conceptual Foundations of Materials: A Standard Model for Ground- and ExcitedState Properties (Elsevier, New York, 2006).
[6] B. Sahni, Vikram, J. Kangsabanik, and A. Alam, J. Phys. Chem. Lett. 11, 6364 (2020).
[7] M. J. van Setten, M. Giantomassi, X. Gonze, G.-M. Rignanese, and G. Hautier, Phys. Rev. B 96, 155207 (2017).
[8] M. J. van Setten, F. Caruso, S. Sharifzadeh, X. Ren, M. Scheffler, F. Liu, J. Lischner, L. Lin, J. R. Deslippe, S. G. Louie et al., J. Chem. Theory Comput. 11, 5665 (2015).
[9] A. Stuke, C. Kunkel, D. Golze, M. Todorović, J. T. Margraf, K. Reuter, P. Rinke, and H. Oberhofer, Sci. Data 7, 58 (2020).
[10] D. M. Kennes, M. Claassen, L. Xian, A. Georges, A. J. Millis, J. Hone, C. R. Dean, D. N. Basov, A. N. Pasupathy, and A. Rubio, Nat. Phys. 17, 155 (2021).
[11] M. H. Naik, S. Kundu, I. Maity, and M. Jain, Phys. Rev. B 102, 075413 (2020).
[12] F. Wu, T. Lovorn, E. Tutuc, I. Martin, and A. H. MacDonald, Phys. Rev. Lett. 122, 086402 (2019).
[13] M. Angeli and A. H. MacDonald, Proc. Natl. Acad. Sci. U.S.A. 118, e2021826118 (2021).
[14] S. Carr, S. Fang, and E. Kaxiras, Nat. Rev. Mater. 5, 748 (2020).
[15] K. Tran, J. Choi, and A. Singh, 2D Mater. 8, 022002 (2020).
[16] L. Xian, M. Claassen, D. Kiese, M. M. Scherer, S. Trebst, D. M. Kennes, and A. Rubio, Nat. Commun. 12, 5644 (2021).
[17] X. Lu, X. Li, and L. Yang, Phys. Rev. B 100, 155416 (2019).
[18] G. Samsonidze, M. Jain, J. Deslippe, M. L. Cohen, and S. G. Louie, Phys. Rev. Lett. 107, 186404 (2011).
[19] W. Gao, W. Xia, X. Gao, and P. Zhang, Sci. Rep. 6, 36849 (2016).
[20] F. Bruneval and X. Gonze, Phys. Rev. B 78, 085125 (2008).
[21] J. Deslippe, G. Samsonidze, M. Jain, M. L. Cohen, and S. G. Louie, Phys. Rev. B 87, 165124 (2013).
[22] J. Berger, L. Reining, and F. Sottile, Phys. Rev. B 82, 041103(R) (2010).
[23] P. Liu, M. Kaltak, J. Klimeš, and G. Kresse, Phys. Rev. B 94, 165109 (2016).
[24] M. Kim, G. J. Martyna, and S. Ismail-Beigi, Phys. Rev. B 101, 035139 (2020).
[25] D. Foerster, P. Koval, and D. Sánchez-Portal, J. Chem. Phys. 135, 074105 (2011).
[26] J. Wilhelm, D. Golze, L. Talirz, J. Hutter, and C. A. Pignedoli, J. Phys. Chem. Lett. 9, 306 (2018).
[27] R. M. Parrish, E. G. Hohenstein, N. F. Schunck, C. D. Sherrill, and T. J. Martínez, Phys. Rev. Lett. 111, 132505 (2013).
[28] A. Förster and L. Visscher, J. Chem. Theory Comput. 16, 7381 (2020).
[29] H. Ma, L. Wang, L. Wan, J. Li, X. Qin, J. Liu, W. Hu, L. Lin, C. Yang, and J. Yang, J. Phys. Chem. A 125, 7545 (2021).
[30] I. Duchemin and X. Blase, J. Chem. Theory Comput. 17, 2383 (2021).
[31] D. Neuhauser, Y. Gao, C. Arntsen, C. Karshenas, E. Rabani, and R. Baer, Phys. Rev. Lett. 113, 076402 (2014).
[32] V. Vlček, E. Rabani, D. Neuhauser, and R. Baer, J. Chem. Theory Comput. 13, 4997 (2017).
[33] V. Vlček, W. Li, R. Baer, E. Rabani, and D. Neuhauser, Phys. Rev. B 98, 075107 (2018).
[34] M. Romanova and V. Vlček, npj Comput. Mater. 8, 11 (2022).
[35] M. Romanova and V. Vlček, J. Chem. Phys. 153, 134103 (2020).
[36] F. Giustino, M. L. Cohen, and S. G. Louie, Phys. Rev. B 81, 115105 (2010).
[37] M. Govoni and G. Galli, J. Chem. Theory Comput. 11, 2680 (2015).
[38] L. Hung, F. H. Da Jornada, J. Souto-Casares, J. R. Chelikowsky, S. G. Louie, and S. Öğüt, Phys. Rev. B 94, 085125 (2016).
[39] M. Del Ben, F. H. Da Jornada, A. Canning, N. Wichmann, K. Raman, R. Sasanka, C. Yang, S. G. Louie, and J. Deslippe, Comput. Phys. Commun. 235, 187 (2020).
[40] A. Oschlies, R. W. Godby, and R. J. Needs, Phys. Rev. B 51, 1527 (1995).
[41] S. Lebègue, B. Arnaud, M. Alouani, and P. E. Bloechl, Phys. Rev. B 67, 155208 (2003).
[42] F. Bruneval, Exchange and correlation in the electronic structure of solids, from silicon to cuprous oxide: GW approximation and beyond, Ph.D. thesis, Ecole Polytechnique Palaiseau, France, 2005.
[43] See Supplemental Material at http://link.aps.org/ supplemental/10.1103/PhysRevLett.132.086401 for details on the following: Sec. S1, computational details; Sec. S1.5, information about direct diagonalization of the mean-field Hamiltonian; Sec. S1.6, pseudocode implementation of the pseudobands approach; Sec. S2, additional convergence tests; Sec. S2.2.1, discussion on extrapolation of $G W$ corrections from unit cells to moiré cells; Sec. S3, convergence derivations; Sec. S3.2.2, discussion about the value of $N_{P}$, which includes Refs. [2,8,35,40-42,44-60].
[44] J. Deslippe, G. Samsonidze, D. A. Strubbe, M. Jain, M. L. Cohen, and S. G. Louie, Comput. Phys. Commun. 183, 1269 (2012).
[45] A. Marek, V. Blum, R. Johanni, V. Havu, B. Lang, T. Auckenthaler, A. Heinecke, H.-J. Bungartz, and H. Lederer, J. Phys. Condens. Matter 26, 213201 (2014).
[46] M. Del Ben, F. H. da Jornada, G. Antonius, T. Rangel, S. G. Louie, J. Deslippe, and A. Canning, Phys. Rev. B 99, 125128 (2019).
[47] R. Baer, D. Neuhauser, and E. Rabani, Phys. Rev. Lett. 111, 106402 (2013).
[48] P. Giannozzi, S. Baroni, N. Bonini, M. Calandra, R. Car, C. Cavazzoni, D. Ceresoli, G. L. Chiarotti, M. Cococcioni, I. Dabo et al., J. Phys. Condens. Matter 21, 395502 (2009).
[49] J. P. Perdew, K. Burke, and M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996).
[50] M. J. van Setten, M. Giantomassi, E. Bousquet, M. J. Verstraete, D. R. Hamann, X. Gonze, and G.-M. Rignanese, Comput. Phys. Commun. 226, 39 (2018).
[51] M. Schlipf and F. Gygi, Comput. Phys. Commun. 196, 36 (2015).
[52] S. Ismail-Beigi, Phys. Rev. B 73, 233103 (2006).
[53] R. T. Downs and M. Hall-Wallace, Am. Mineral. 88, 247 (2003).
[54] T. Rangel, M. Del Ben, D. Varsano, G. Antonius, F. Bruneval, F. H. da Jornada, M. J. van Setten, O. K. Orhan, D. D. O'Regan, A. Canning et al., Comput. Phys. Commun. 255, 107242 (2020).
[55] A. H. Larsen, J. J. Mortensen, J. Blomqvist, I. E. Castelli, R. Christensen, M. Dułak, J. Friis, M. N. Groves, B. Hammer, C. Hargus et al., J. Phys. Condens. Matter 29, 273002 (2017).
[56] A. P. Thompson, H. M. Aktulga, R. Berger, D. S. Bolintineanu, W. M. Brown, P. S. Crozier, P. J. in't Veld, A. Kohlmeyer, S. G. Moore, T. D. Nguyen et al., Comput. Phys. Commun. 271, 108171 (2022).
[57] M. H. Naik, I. Maity, P. K. Maiti, and M. Jain, J. Phys. Chem. C 123, 9770 (2019).
[58] J.-W. Jiang, H. S. Park, and T. Rabczuk, J. Appl. Phys. 114, 064307 (2013).
[59] F. H. da Jornada, D. Y. Qiu, and S. G. Louie, Phys. Rev. B 95, 035109 (2017).
[60] D. Y. Qiu, F. H. da Jornada, and S. G. Louie, Phys. Rev. Lett. 111, 216805 (2013).
[61] A. R. Altman, S. Kundu, and F. H. da Jornada, Supplemental datasets for "Mixed stochastic-deterministic approach for many-body perturbation theory calculations", Zenodo, 10.5281/zenodo. 8278011 (2023).
[62] H. F. Wilson, F. Gygi, and G. Galli, Phys. Rev. B 78, 113303 (2008).
[63] M. Shao, L. Lin, C. Yang, F. Liu, F. H. Da Jornada, J. Deslippe, and S. G. Louie, Sci. China Math. 59, 1593 (2016).
[64] E. Y. Andrei, D. K. Efetov, P. Jarillo-Herrero, A. H. MacDonald, K. F. Mak, T. Senthil, E. Tutuc, A. Yazdani, and A. F. Young, Nat. Rev. Mater. 6, 201 (2021).
[65] Z. Zheng, Q. Ma, Z. Bi, S. de La Barrera, M.-H. Liu, N. Mao, Y. Zhang, N. Kiper, K. Watanabe, T. Taniguchi et al., Nature (London) 588, 71 (2020).
[66] Y. Cao, V. Fatemi, S. Fang, K. Watanabe, T. Taniguchi, E. Kaxiras, and P. Jarillo-Herrero, Nature (London) 556, 43 (2018).
[67] L. Wang, E.-M. Shih, A. Ghiotto, L. Xian, D. A. Rhodes, C. Tan, M. Claassen, D. M. Kennes, Y. Bai, B. Kim, K. Watanabe, T. Taniguchi, X. Zhu, J. Hone, A. Rubio, A. N. Pasupathy, and C. R. Dean, Nat. Mater. 19, 861 (2020).
[68] Y. Xu, S. Liu, D. A. Rhodes, K. Watanabe, T. Taniguchi, J. Hone, V. Elser, K. F. Mak, and J. Shan, Nature (London) 587, 214 (2020).
[69] K. Tran, G. Moody, F. Wu, X. Lu, J. Choi, K. Kim, A. Rai, D. A. Sanchez, J. Quan, A. Singh et al., Nature (London) 567, 71 (2019).
[70] H. Yu, G.-B. Liu, J. Tang, X. Xu, and W. Yao, Sci. Adv. 3, e1701696 (2017).
[71] T. I. Andersen, G. Scuri, A. Sushko, K. De Greve, J. Sung, Y. Zhou, D. S. Wild, R. J. Gelly, H. Heo, D. Bérubé, A. Y. Joe, L. A. Jauregui, K. Watanabe, T. Taniguchi, P. Kim, H. Park, and M. D. Lukin, Nat. Mater. 20, 480 (2021).
[72] M. H. Naik, E. C. Regan, Z. Zhang, Y.-H. Chan, Z. Li, D. Wang, Y. Yoon, C. S. Ong, W. Zhao, S. Zhao, M. I. B. Utama, B. Gao, X. Wei, M. Sayyad, K. Yumigeta, K. Watanabe, T. Taniguchi, S. Tongay, F. H. da Jornada, F. Wang, and S. G. Louie, Nature (London) 609, 52 (2022).
[73] L. A. Jauregui, A. Y. Joe, K. Pistunova, D. S. Wild, A. A. High, Y. Zhou, G. Scuri, K. De Greve, A. Sushko, C.-H. Yu et al., Science 366, 870 (2019).
[74] O. Karni, E. Barré, S. C. Lau, R. Gillen, E. Y. Ma, B. Kim, K. Watanabe, T. Taniguchi, J. Maultzsch, K. Barmak, R. H. Page, and T. F. Heinz, Phys. Rev. Lett. 123, 247402 (2019).
[75] O. Karni et al., Nature (London) 603, 247 (2022).
[76] Y. Tang, J. Gu, S. Liu, K. Watanabe, T. Taniguchi, J. Hone, K. F. Mak, and J. Shan, Nat. Nanotechnol. 16, 52 (2021).
[77] K. L. Seyler, P. Rivera, H. Yu, N. P. Wilson, E. L. Ray, D. G. Mandrus, J. Yan, W. Yao, and X. Xu, Nature (London) 567, 66 (2019).
[78] Y. Shimazaki, I. Schwartz, K. Watanabe, T. Taniguchi, M. Kroner, and A. Imamoğlu, Nature (London) 580, 472 (2020).
[79] C. Jin, J. Kim, M. I. B. Utama, E. C. Regan, H. Kleemann, H. Cai, Y. Shen, M. J. Shinner, A. Sengupta, K. Watanabe et al., Science 360, 893 (2018).
[80] S. Latini, T. Olsen, and K. S. Thygesen, Phys. Rev. B 92, 245123 (2015).
[81] D. Y. Qiu, F. H. da Jornada, and S. G. Louie, Phys. Rev. B 93, 235435 (2016).
[82] F. H. da Jornada, D. Y. Qiu, and S. G. Louie, Phys. Rev. B 95, 035109 (2017).
[83] S. Kundu, T. Amit, H. R. Krishnamurthy, M. Jain, and S. Refaely-Abramson, npj Comput. Mater. 9, 186 (2023).
[84] V. Popescu and A. Zunger, Phys. Rev. B 85, 085201 (2012).