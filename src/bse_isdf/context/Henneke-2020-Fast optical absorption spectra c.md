\section*{Communications in Applied Mathematics and Computational Science}
![](https://cdn.mathpix.com/cropped/ca5af182-2f19-48ee-9bf5-df0a26ec22a9-01.jpg?height=981&width=497&top_left_y=9&top_left_x=962)

\section*{FAST OPTICAL ABSORPTION SPECTRA CALCULATIONS FOR PERIODIC SOLID STATE SYSTEMS}

Felix Henneke, Lin Lin, Christian Vorwerk, Claudia Draxl, Rupert Klein and Chao Yang
vol. 15 no. 1
2020

\title{
FAST OPTICAL ABSORPTION SPECTRA CALCULATIONS FOR PERIODIC SOLID STATE SYSTEMS
}

\author{
Felix Henneke, Lin Lin, Christian Vorwerk, Claudia Draxl, Rupert Klein and Chao Yang
}

\begin{abstract}
We present a method to construct an efficient approximation to the bare exchange and screened direct interaction kernels of the Bethe-Salpeter Hamiltonian for periodic solid state systems via the interpolative separable density fitting technique. We show that the cost of constructing the approximate Bethe-Salpeter Hamiltonian can be reduced to nearly optimal as $\mathscr{O}\left(N_{k}\right)$ with respect to the number of samples in the Brillouin zone $N_{k}$ for the first time. In addition, we show that the cost for applying the Bethe-Salpeter Hamiltonian to a vector scales as $\mathscr{O}\left(N_{k} \log N_{k}\right)$. Therefore, the optical absorption spectrum, as well as selected excitation energies, can be efficiently computed via iterative methods such as the Lanczos method. This is a significant reduction from the $O\left(N_{k}^{2}\right)$ and $\mathscr{O}\left(N_{k}^{3}\right)$ scaling associated with a brute force approach for constructing the Hamiltonian and diagonalizing the Hamiltonian, respectively. We demonstrate the efficiency and accuracy of this approach with both one-dimensional model problems and three-dimensional real materials (graphene and diamond). For the diamond system with $N_{k}=2197$, it takes 6 hours to assemble the Bethe-Salpeter Hamiltonian and 4 hours to fully diagonalize the Hamiltonian using 169 cores when the brute force approach is used. The new method takes less than 3 minutes to set up the Hamiltonian and 24 minutes to compute the absorption spectrum on a single core.
\end{abstract}

\section*{1. Introduction}

The Bethe-Salpeter equation (BSE), derived from the many-body perturbation theory (MBPT), is a widely used method for describing the optical absorption process in molecules and solids $[32 ; 33 ; 36 ; 24 ; 1 ; 25 ; 7]$. It models the behavior of an electron-hole pair, which is an excitation process with two quasiparticles. Solving the BSE requires constructing and diagonalizing a structured matrix, called the Bethe-Salpeter Hamiltonian (BSH). In the context of optical absorption, the

\footnotetext{
MSC2010: 65F15, 65Z05.
Keywords: Bethe-Salpeter equation, interpolative separable density fitting, optical absorption function.
}
eigenvalues of the BSH are the exciton energies and the corresponding eigenfunctions yield the exciton wavefunctions. The BSH consists of the so-called bare exchange and screened direct interaction kernels that depend on single particle orbitals obtained from a quasiparticle (usually at the GW level) or mean-field calculation. For isolated systems such as molecules, the construction of these kernels requires at least $\mathscr{O}\left(N_{e}^{5}\right)$ operations in a conventional approach, where $N_{e}$ is the number of electrons in the system. This is very costly for large systems that contain hundreds or more atoms. Recent efforts have actively explored methods for efficient representation of the BSH, in order to reduce the high computational cost of BSE calculations $[4 ; 3 ; 16 ; 21 ; 30 ; 27 ; 28 ; 31 ; 23]$.

In a recent work [13], two of the authors have presented an efficient way to construct the BSH for molecular systems, and to efficiently solve the BSE eigenvalue problem using an iterative scheme. This approach is based on the recently developed interpolative separable density fitting (ISDF) decomposition [19; 20]. The ISDF decomposition has been applied to accelerate a number of applications in computational chemistry and materials science, including the computation of two-electron integrals [19], correlation energy in the random phase approximation [18], density functional perturbation theory [15], and hybrid density functional calculations [12; 8]. In this scheme, a matrix consisting of products of single particle orbital pairs is efficiently approximated as a low-rank matrix product of a matrix built with a small number of auxiliary basis vectors and an expansion coefficient matrix. This decomposition allows us to construct efficient representations of the bare exchange and screened direct kernels. For isolated molecular systems, the construction of the ISDF-compressed BSH matrix only requires $\mathscr{O}\left(N_{e}^{3}\right)$ operations when the rank of the numerical auxiliary basis is kept at $\mathscr{O}\left(N_{e}\right)$. This results in considerable reduction of the cost compared to the $\mathscr{O}\left(N_{e}^{5}\right)$ complexity required in a conventional approach. By keeping the interaction kernels in a decomposed form, the matrix-vector multiplications required in the iterative diagonalization procedures of the Hamiltonian $H_{\mathrm{BSE}}$ can be performed efficiently. We can further use these efficient matrix-vector multiplications in a structure-preserving Lanczos algorithm [34] to obtain an approximate absorption spectrum without an explicit diagonalization of the approximate $H_{\mathrm{BSE}}$.

This paper generalizes the work in [13] to periodic solid state systems. According to the Bloch decomposition, each single particle orbital in a periodic system can be characterized by an orbital index $i$ and a Brillouin zone index $\boldsymbol{k}$. Compared to isolated systems, the total number of electrons $N_{e}$ is equal to the number of electrons per unit cell multiplied by the number of $\boldsymbol{k}$-points denoted by $N_{k}$. It has been observed that for many extended systems, the number of orbitals (both occupied and virtual orbitals) required for one particular $\boldsymbol{k}$ index can be relatively small, and is independent of $N_{e}$. Hence, the difficulty of optical absorption spectra
calculations for periodic systems mainly arise from the large number of $\boldsymbol{k}$-points. This is particularly the case when the excitons are delocalized in the real space, or when the Fermi-surface is not smooth (such as graphene, and other metallic systems). In such case, $N_{k}$ can often be rather large (from hundreds to hundreds of thousands; see, e.g., [29], where a $120 \times 120 \times 1 \boldsymbol{k}$-grid is used for the quasi-two-dimensional $\mathrm{MoS}_{2}$ system) in order to properly discretize and sample the Brillouin zone. The cost for constructing the bare exchange and screened direct kernels scales as $O\left(N_{k}^{2}\right)$, while the cost for diagonalizing the corresponding BSH scales as $\mathscr{O}\left(N_{k}^{3}\right)$. This is prohibitively expensive when a dense discretization of the Brillouin zone is needed.

With the help of ISDF for periodic systems [20], we reduce the computational cost for producing optical absorption spectra to a scaling almost linear in $N_{k}$. First, the complexity of the bare exchange and screened direct kernel construction for extended systems is reduced to the optimal complexity of $\mathscr{O}\left(N_{k}\right)$. A sufficiently reduced representation of the pair product orbitals is possible, thanks to the smoothness of the single particle orbitals with respect to the $\boldsymbol{k}$ index, and the fact that the Brillouin zone is a compact domain. Second, the separable structure of the decomposition makes it possible to exploit a convolutional structure in the screened direct kernel. The complexity of applying the approximated kernels to a vector with respect to $N_{k}$ is thus only $0\left(N_{k} \log N_{k}\right)$. Instead of diagonalizing the BSH directly, we use iterative methods such as the Lanczos method to evaluate the optical absorption spectrum. The same strategy can be applied to evaluate selected excitation energies.

Despite the increasingly wide adoption of the BSE theory in condensed matter physics and quantum chemistry for analyzing optical properties of materials, we could not find a precise mathematical description of how the BSH is constructed for periodic systems in the literature. Therefore, after concise review of the single particle theory and the Bethe-Salpeter equation for periodic systems in Section 2.1, we provide a relatively self-contained derivation of the BSE for periodic systems in Section 2.2 from a numerical linear algebra perspective. We hope our presentation (especially using a discretized Brillouin zone so that all matrices are of finite dimension) is useful to readers not familiar with the matter.

Then the rest of the paper is organized as follows. The interpolative separable density fitting for periodic systems is introduced in Section 3, and the application of the approximate BSH in the ISDF format to a vector in Section 4. The numerical results are presented in Section 5, followed by a conclusion in Section 6.

\section*{2. Preliminaries}
2.1. Single particle theory for periodic systems. To facilitate further discussion we briefly review Bloch-Floquet theory for periodic systems. Without loss of generality we consider a three-dimensional crystal. The Bravais lattice with lattice
vectors $\boldsymbol{a}_{1}, \boldsymbol{a}_{2}, \boldsymbol{a}_{3} \in \mathbb{R}^{3}$ is defined as
$$
\begin{equation*}
\mathbb{L}=\left\{\boldsymbol{R} \mid \boldsymbol{R}=n_{1} \boldsymbol{a}_{1}+n_{2} \boldsymbol{a}_{2}+n_{3} \boldsymbol{a}_{3}, n_{1}, n_{2}, n_{3} \in \mathbb{Z}\right\} . \tag{2-1}
\end{equation*}
$$

In single particle theories such as the Kohn-Sham density functional theory, the self-consistent effective potential $V_{\text {eff }}$ is real-valued and $\mathbb{L}$-periodic, i.e.,
$$
V_{\mathrm{eff}}(\boldsymbol{r}+\boldsymbol{R})=V_{\mathrm{eff}}(\boldsymbol{r}) \quad \text { for all } \boldsymbol{r} \in \mathbb{R}^{3} \text { and } \boldsymbol{R} \in \mathbb{L} .
$$

The unit cell is defined as
$$
\begin{equation*}
\Omega=\left\{\boldsymbol{r}=c_{1} \boldsymbol{a}_{1}+c_{2} \boldsymbol{a}_{2}+c_{3} \boldsymbol{a}_{3} \mid 0 \leq c_{1}, c_{2}, c_{3}<1\right\} . \tag{2-2}
\end{equation*}
$$

The Bravais lattice induces a reciprocal lattice $\mathbb{L}^{*}$, with its lattice vectors $\boldsymbol{b}_{1}, \boldsymbol{b}_{2}, \boldsymbol{b}_{3}$ satisfying $\boldsymbol{a}_{\alpha} \cdot \boldsymbol{b}_{\beta}=2 \pi \delta_{\alpha \beta}, \alpha, \beta \in\{1,2,3\}$. The unit cell of the reciprocal lattice is called the (first) Brillouin zone and denoted by $\Omega^{*}$, defined as
$$
\Omega^{*}=\left\{\boldsymbol{k}=k_{1} \boldsymbol{b}_{1}+k_{2} \boldsymbol{b}_{2}+k_{3} \boldsymbol{b}_{3} \left\lvert\,-\frac{1}{2} \leq k_{1}\right., k_{2}, k_{3}<\frac{1}{2}\right\} .
$$

The Brillouin zone has a number of special points related to the symmetry of the crystal. The common special point is the $\Gamma$-point, which corresponds to $\boldsymbol{k}= [0,0,0]^{\top}$.

According to the Bloch-Floquet theory, the spectrum of the Hamiltonian $\mathscr{H}= -\frac{1}{2} \nabla_{\boldsymbol{r}}^{2}+V_{\text {eff }}(\boldsymbol{r})$ can be relabeled using two indices ( $i, \boldsymbol{k}$ ), where $i \in \mathbb{N}$ is called the band index and $\boldsymbol{k} \in \Omega^{*}$ is the Brillouin zone index. Each generalized eigenfunction $\psi_{i \boldsymbol{k}}(\boldsymbol{r})$ is known as a Bloch orbital and satisfies $\mathscr{H}_{i \boldsymbol{k}}(\boldsymbol{r})=\epsilon_{i \boldsymbol{k}} \psi_{i \boldsymbol{k}}(\boldsymbol{r})$ with Bloch boundary conditions $\psi_{i k}(\boldsymbol{r}+\boldsymbol{R})=e^{\mathrm{i} \boldsymbol{k} \cdot \boldsymbol{R}} \psi_{i \boldsymbol{k}}(\boldsymbol{r})$ for any $\boldsymbol{R} \in \mathbb{L}$. Furthermore, $\psi_{i \boldsymbol{k}}$ can be decomposed using the Bloch decomposition
$$
\begin{equation*}
\psi_{i k}(\boldsymbol{r})=e^{\mathrm{i} \boldsymbol{k} \cdot \boldsymbol{r}} u_{i k}(\boldsymbol{r}) \tag{2-3}
\end{equation*}
$$
where $u_{i k}(\boldsymbol{r})$ is the periodic part of $\psi_{i k}(\boldsymbol{r})$ satisfying the periodic boundary condition on the unit cell
$$
\begin{equation*}
u_{i k}(\boldsymbol{r}+\boldsymbol{R})=u_{i k}(\boldsymbol{r}) \quad \text { for all } \boldsymbol{R} \in \mathbb{L} . \tag{2-4}
\end{equation*}
$$

It can be directly obtained by solving the eigenvalue problem
$$
\begin{equation*}
\mathscr{H}(\boldsymbol{k}) u_{i k}=\epsilon_{i k} u_{i k}(\boldsymbol{r}), \quad \boldsymbol{r} \in \Omega, \boldsymbol{k} \in \Omega^{*}, \tag{2-5}
\end{equation*}
$$
where $\mathscr{H}(\boldsymbol{k})=-\frac{1}{2}\left(\nabla_{\boldsymbol{r}}+\mathrm{i} \boldsymbol{k}\right)^{2}+V_{\text {eff }}(\boldsymbol{r})$. For each $\boldsymbol{k} \in \Omega^{*}$, the eigenvalues $\epsilon_{i \boldsymbol{k}}$ are ordered nondecreasingly. For a fixed $i,\left\{\epsilon_{i \boldsymbol{k}}\right\}$ as a function of $\boldsymbol{k}$ is called a Bloch band. The collection of all eigenvalues forms the band structure of the crystal, which characterizes the spectrum of the operator $\mathscr{H}$.

In the discussion below, we denote by $N_{v}$ the number of valence bands (i.e., occupied orbitals per unit cell in the ground state) and $N_{c}$ the number of conduction bands (i.e., unoccupied orbitals per unit cell in the ground state). We also define
$N=N_{v}+N_{c}$. We assume the systems to be insulating, in the sense that the following band isolation conditions between the valence and conduction bands are satisfied:
$$
\begin{equation*}
\inf \left|\epsilon_{i \boldsymbol{k}}-\epsilon_{i^{\prime} \boldsymbol{k}^{\prime}}\right|:=\epsilon_{g}>0, \quad \boldsymbol{k}, \boldsymbol{k}^{\prime} \in \Omega^{*}, 1 \leq i \leq N_{v}, N_{v}+1 \leq i^{\prime} \leq N . \tag{2-6}
\end{equation*}
$$

Denote by $|\Omega|$ the volume of the unit cell, and by
$$
\left|\Omega^{*}\right|=\frac{(2 \pi)^{3}}{|\Omega|}
$$
the volume of the Brillouin zone. The Bloch orbitals $\left\{\psi_{i k}\right\}$ satisfy the orthonormality condition in the distributional sense:
$$
\begin{equation*}
\int_{\mathbb{R}^{3}} \psi_{i^{\prime} k^{\prime}}^{*}(\boldsymbol{r}) \psi_{i, \boldsymbol{k}}(\boldsymbol{r}) \mathrm{d} \boldsymbol{r}=\left|\Omega^{*}\right| \delta_{i^{\prime}, i} \delta\left(\boldsymbol{k}^{\prime}-\boldsymbol{k}\right) \tag{2-7}
\end{equation*}
$$

Here $\delta_{i^{\prime}, i}$ is the Kronecker $\delta$ symbol for a discrete set, while $\delta\left(\boldsymbol{k}^{\prime}-\boldsymbol{k}\right)$ is the Dirac delta distribution. Equation (2-7) implies the normalization condition when integrated over the Brillouin zone:
$$
\begin{equation*}
\frac{1}{\left|\Omega^{*}\right|} \int_{\Omega^{*}} \int_{\mathbb{R}^{3}} \psi_{i^{\prime} \boldsymbol{k}}^{*}(\boldsymbol{r}) \psi_{i \boldsymbol{k}}(\boldsymbol{r}) \mathrm{d} \boldsymbol{r} \mathrm{~d} \boldsymbol{k}=\delta_{i^{\prime}, i} \tag{2-8}
\end{equation*}
$$

From the Bloch orbitals, the ground state electron density can be constructed as
$$
\begin{equation*}
\rho(\boldsymbol{r})=\frac{1}{\left|\Omega^{*}\right|} \int_{\Omega^{*}} \sum_{i=1}^{N_{v}}\left|\psi_{i \boldsymbol{k}}(\boldsymbol{r})\right|^{2} \mathrm{~d} \boldsymbol{k}=\frac{1}{\left|\Omega^{*}\right|} \int_{\Omega^{*}} \sum_{i=1}^{N_{v}}\left|u_{i \boldsymbol{k}}(\boldsymbol{r})\right|^{2} \mathrm{~d} \boldsymbol{k} \tag{2-9}
\end{equation*}
$$

In order to practically perform calculations for periodic systems, the integration with respect to the Brillouin zone $\Omega^{*}$ needs to be discretized using a quadrature. The most commonly used scheme is based on the Monkhorst-Pack grid [22]
$$
\begin{equation*}
\mathscr{K}_{s}^{\ell}=\left\{\left.\sum_{\alpha=1}^{3} \frac{m_{\alpha}-s_{\alpha}}{N_{\alpha}^{\ell}} \boldsymbol{b}_{\alpha} \right\rvert\, m_{\alpha}=-\frac{N_{\alpha}^{\ell}}{2}+1, \ldots, \frac{N_{\alpha}^{\ell}}{2}, 0 \leq s_{\alpha}<1, \alpha=1,2,3\right\} . \tag{2-10}
\end{equation*}
$$

It is clear that $\mathscr{K}_{\boldsymbol{s}}^{\ell} \subset \Omega^{*}$ and that it corresponds to a uniform discretization of the Brillouin zone. When the shift vector $\boldsymbol{s}=\mathbf{0}$, we denote $\mathscr{K}^{\ell}:=\mathscr{K}_{\mathbf{0}}^{\ell}$, and the calculation of periodic systems can be equivalently performed using a supercell consisting of $N_{1}^{\ell} \times N_{2}^{\ell} \times N_{3}^{\ell}$ unit cells. The supercell is denoted by $\Omega^{\ell}$, and is further equipped with a periodic boundary condition called the Born-von Karman boundary condition [2]. The calculation of a periodic crystal can thus be recovered by taking the limit $N_{\alpha}^{\ell} \rightarrow \infty$. We denote by $N_{k} \equiv N^{\ell}:=N_{1}^{\ell} N_{2}^{\ell} N_{3}^{\ell}$ the total number of unit cells, or equivalently the total number of Monkhorst-Pack grid points in the Brillouin zone.

Assuming the Brillouin zone is discretized using $\mathscr{K}^{\ell}$, the orthogonality condition (2-7) becomes
$$
\begin{equation*}
\int_{\Omega^{\ell}} \psi_{i^{\prime} k^{\prime}}^{*}(\boldsymbol{r}) \psi_{i \boldsymbol{k}}(\boldsymbol{r}) \mathrm{d} \boldsymbol{r}=\delta_{i^{\prime}, i} \delta_{\boldsymbol{k}^{\prime}, \boldsymbol{k}}, \quad \boldsymbol{k}, \boldsymbol{k}^{\prime} \in \mathscr{K}^{\ell} \tag{2-11}
\end{equation*}
$$

We also modify the Bloch decomposition as
$$
\begin{equation*}
\psi_{i \boldsymbol{k}}(\boldsymbol{r})=\frac{1}{\sqrt{N^{\ell}}} e^{i \boldsymbol{k} \cdot \boldsymbol{r}} u_{i \boldsymbol{k}}(\boldsymbol{r}), \quad \boldsymbol{k} \in \mathscr{K}^{\ell} . \tag{2-12}
\end{equation*}
$$

Here the normalization factor $1 / \sqrt{N^{\ell}}$ is introduced so that the orthogonality condition for the periodic part implies
$$
\begin{equation*}
\int_{\Omega} u_{i^{\prime} \boldsymbol{k}}^{*}(\boldsymbol{r}) u_{i \boldsymbol{k}}(\boldsymbol{r}) \mathrm{d} \boldsymbol{r}=\delta_{i^{\prime}, i}, \quad \boldsymbol{k} \in \mathscr{K}^{\ell} \tag{2-13}
\end{equation*}
$$

To facilitate the bookkeeping effort of various relevant constants in practical calculations, in the discussion below we will always assume that the Brillouin zone is discretized into $\Im^{\ell}$ with a corresponding supercell $\Omega^{\ell}$. The volume of the supercell is $\left|\Omega^{\ell}\right|=N^{\ell}|\Omega|=N_{k}|\Omega|$. The unit cell is further discretized into a uniform grid $\left\{\boldsymbol{r}_{i}\right\}_{i=1}^{N_{g}}$. Practical BSE calculations often truncate the number of conduction bands aggressively, in the sense that $N_{g} \gg N_{v}+N_{c}=: N$. Numerical results indicate that in many cases, the low-lying excitation spectrum is relatively insensitive to $N_{c}$, and one can often choose $N_{c} \approx N_{v}$. Unless otherwise clarified, we may not distinguish a continuous vector $u(\boldsymbol{r})$ and the corresponding discretized vector $\left\{u\left(\boldsymbol{r}_{i}\right)\right\}$. Similarly, when the context is clear, we do not distinguish the kernel of an operator $A\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right)$ and its discretized matrix $\left\{A\left(\boldsymbol{r}_{i}, \boldsymbol{r}_{j}\right)\right\}$.
2.2. Bethe-Salpeter equation for periodic systems. The Bethe-Salpeter equation is an eigenvalue problem of the form
$$
\begin{equation*}
H_{\mathrm{BSE}} X=E X, \tag{2-14}
\end{equation*}
$$
where $H_{\mathrm{BSE}}$ is the Bethe-Salpeter Hamiltonian (BSH), $X$ is the exciton wavefunction, and $E$ is the corresponding exciton energy. For periodic systems, the BSH has the block structure
$$
H_{\mathrm{BSE}}=\left[\begin{array}{cc}
D+2 V_{A}-W_{A} & 2 V_{B}-W_{B}  \tag{2-15}\\
-2 \bar{V}_{B}+\bar{W}_{B} & -D-2 \bar{V}_{A}+\bar{W}_{A}
\end{array}\right],
$$
where $D\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right)=\left(\epsilon_{i_{c} \boldsymbol{k}}-\epsilon_{i_{v} \boldsymbol{k}}\right) \delta_{i_{v}, j_{v}} \delta_{i_{c}, j_{c}} \delta_{\boldsymbol{k}, \boldsymbol{k}^{\prime}}$ is an $\left(N_{v} N_{c} N_{k}\right) \times\left(N_{v} N_{c} N_{k}\right)$ diagonal matrix. The quasiparticle energies $\epsilon_{i_{v} k}, \epsilon_{i_{c} k}$ are typically obtained from a GW calculation [32]. The $V_{A}$ and $V_{B}$ matrices represent the bare exchange interaction of electron-hole pairs, and the $W_{A}$ and $W_{B}$ matrices are referred to as
the screened direct interaction of electron-hole pairs. These matrices are defined as
$$
\begin{align*}
V_{A}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right) & =\int_{\Omega^{\ell} \times \Omega^{\ell}} \bar{\psi}_{i_{c} \boldsymbol{k}}(\boldsymbol{r}) \psi_{i_{v} \boldsymbol{k}}(\boldsymbol{r}) V\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) \bar{\psi}_{j_{v} \boldsymbol{k}^{\prime}}\left(\boldsymbol{r}^{\prime}\right) \psi_{j_{c} \boldsymbol{k}^{\prime}}\left(\boldsymbol{r}^{\prime}\right) \mathrm{d} \boldsymbol{r} \mathrm{~d} \boldsymbol{r}^{\prime} \\
V_{B}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right) & =\int_{\Omega^{\ell} \times \Omega^{\ell}} \bar{\psi}_{i_{c} \boldsymbol{k}}(\boldsymbol{r}) \psi_{i_{v} \boldsymbol{k}}(\boldsymbol{r}) V\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) \bar{\psi}_{j_{c} \boldsymbol{k}^{\prime}}\left(\boldsymbol{r}^{\prime}\right) \psi_{j_{v} \boldsymbol{k}^{\prime}}\left(\boldsymbol{r}^{\prime}\right) \mathrm{d} \boldsymbol{r} \mathrm{~d} \boldsymbol{r}^{\prime} \\
W_{A}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right) & =\int_{\Omega^{\ell} \times \Omega^{\ell}} \bar{\psi}_{i_{c} \boldsymbol{k}}(\boldsymbol{r}) \psi_{j_{c} \boldsymbol{k}^{\prime}}(\boldsymbol{r}) W\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) \bar{\psi}_{j_{v} \boldsymbol{k}^{\prime}}\left(\boldsymbol{r}^{\prime}\right) \psi_{i_{v} \boldsymbol{k}}\left(\boldsymbol{r}^{\prime}\right) \mathrm{d} \boldsymbol{r} \mathrm{~d} \boldsymbol{r}^{\prime}  \tag{2-16}\\
W_{B}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right) & =\int_{\Omega^{\ell} \times \Omega^{\ell}} \bar{\psi}_{i_{c} \boldsymbol{k}}(\boldsymbol{r}) \psi_{j_{v} \boldsymbol{k}^{\prime}}(\boldsymbol{r}) W\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) \bar{\psi}_{j_{c} \boldsymbol{k}^{\prime}}\left(\boldsymbol{r}^{\prime}\right) \psi_{i_{v} \boldsymbol{k}}\left(\boldsymbol{r}^{\prime}\right) \mathrm{d} \boldsymbol{r} \mathrm{~d} \boldsymbol{r}^{\prime}
\end{align*}
$$

Here $\psi_{i_{v} \boldsymbol{k}}$ and $\psi_{i_{c} \boldsymbol{k}}$ are the valence and conduction single particle orbitals typically obtained from a Kohn-Sham density functional theory (KSDFT) calculation, respectively, and $V\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right)$ and $W\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right)$ are the bare and screened Coulomb interactions. Both $V_{A}$ and $W_{A}$ are Hermitian, whereas $V_{B}$ and $W_{B}$ are complex symmetric. Within the so-called Tamm-Dancoff approximation (TDA) [25], both $V_{B}$ and $W_{B}$ are neglected in (2-15). In this case, the $H_{\mathrm{BSE}}$ becomes Hermitian and we can focus on computing the upper left block of $H_{\mathrm{BSE}}$. Both the KSDFT and GW calculations can be challenging in their own right. In this work, however, we consider their output as given and the starting point of our BSE calculation.

In the following discussion, when a single index $i$ is used, it refers to either $i_{v}$ or $i_{c}$. Using the Bloch decomposition (2-12), the matrix elements of the BSH can be written using the periodic part of the orbitals as
$$
\begin{align*}
V_{A}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right)= & \frac{1}{N_{k}^{2}} \int_{\Omega^{\ell} \times \Omega^{\ell}} \bar{u}_{i_{c} \boldsymbol{k}}(\boldsymbol{r}) u_{i_{v} \boldsymbol{k}}(\boldsymbol{r}) V\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) \bar{u}_{j_{v} \boldsymbol{k}^{\prime}}\left(\boldsymbol{r}^{\prime}\right) u_{j_{c} \boldsymbol{k}^{\prime}}\left(\boldsymbol{r}^{\prime}\right) \mathrm{d} \boldsymbol{r} \mathrm{~d} \boldsymbol{r}^{\prime} \\
V_{B}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right)= & \frac{1}{N_{k}^{2}} \int_{\Omega^{\ell} \times \Omega^{\ell}} \bar{u}_{i_{c} \boldsymbol{k}}(\boldsymbol{r}) u_{i_{v} \boldsymbol{k}}(\boldsymbol{r}) V\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) \bar{u}_{j_{c} \boldsymbol{k}^{\prime}}\left(\boldsymbol{r}^{\prime}\right) u_{j_{v} \boldsymbol{k}^{\prime}}\left(\boldsymbol{r}^{\prime}\right) \mathrm{d} \boldsymbol{r} \mathrm{~d} \boldsymbol{r}^{\prime} \\
W_{A}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right)= & \frac{1}{N_{k}^{2}} \int_{\Omega^{\ell} \times \Omega^{\ell}} e^{-\mathrm{i}\left(\boldsymbol{k}-\boldsymbol{k}^{\prime}\right) \cdot\left(\boldsymbol{r}-\boldsymbol{r}^{\prime}\right)} \bar{u}_{i_{c} \boldsymbol{k}}(\boldsymbol{r}) u_{j_{c} \boldsymbol{k}^{\prime}}(\boldsymbol{r})  \tag{2-17}\\
& \times W\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) \bar{u}_{j_{v} \boldsymbol{k}^{\prime}}\left(\boldsymbol{r}^{\prime}\right) u_{i_{v} \boldsymbol{k}}\left(\boldsymbol{r}^{\prime}\right) \mathrm{d} \boldsymbol{r} \mathrm{~d} \boldsymbol{r}^{\prime} \\
W_{B}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right)= & \frac{1}{N_{k}^{2}} \int_{\Omega^{\ell} \times \Omega^{\ell}} e^{-\mathrm{i}\left(\boldsymbol{k}-\boldsymbol{k}^{\prime}\right) \cdot\left(\boldsymbol{r}-\boldsymbol{r}^{\prime}\right)} \bar{u}_{i_{c} \boldsymbol{k}}(\boldsymbol{r}) u_{j_{v} \boldsymbol{k}^{\prime}}(\boldsymbol{r}) \\
& \times W\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) \bar{u}_{j_{c} \boldsymbol{k}^{\prime}}\left(\boldsymbol{r}^{\prime}\right) u_{i_{v} \boldsymbol{k}}\left(\boldsymbol{r}^{\prime}\right) \mathrm{d} \boldsymbol{r} \mathrm{~d} \boldsymbol{r}^{\prime}
\end{align*}
$$

Note that $V_{A}, V_{B}$ in (2-17) do not involve the phase factors, since the factor $e^{\mathrm{i} \boldsymbol{k} \cdot \boldsymbol{r}}$ cancels exactly due to the complex conjugate operation. The phase factor only appears in the $W_{A}, W_{B}$ terms.

Equation (2-17) requires the evaluation of integrals of the form
$$
\begin{equation*}
\mathcal{V}(f, g):=\frac{1}{N_{k}} \int_{\Omega^{\ell} \times \Omega^{\ell}} \bar{f}(\boldsymbol{r}) V\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) g\left(\boldsymbol{r}^{\prime}\right) \mathrm{d} \boldsymbol{r} \mathrm{~d} \boldsymbol{r}^{\prime} \tag{2-18}
\end{equation*}
$$
and
$$
\begin{equation*}
\mathcal{W}_{\boldsymbol{q}}(f, g):=\frac{1}{N_{k}} \int_{\Omega^{\ell} \times \Omega^{\ell}} e^{-\mathrm{i} \boldsymbol{q} \cdot\left(\boldsymbol{r}-\boldsymbol{r}^{\prime}\right)} \bar{f}(\boldsymbol{r}) W\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) g\left(\boldsymbol{r}^{\prime}\right) \mathrm{d} \boldsymbol{r} \mathrm{~d} \boldsymbol{r}^{\prime} \tag{2-19}
\end{equation*}
$$

Using such notation,
$$
\begin{align*}
V_{A}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right) & =\frac{1}{N_{k}} \mathscr{V}\left(\bar{u}_{i_{v} \boldsymbol{k}} u_{i_{c} \boldsymbol{k}}, \bar{u}_{j_{v} \boldsymbol{k}^{\prime}} u_{j_{c} \boldsymbol{k}^{\prime}}\right), \\
V_{B}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right) & =\frac{1}{N_{k}} \mathscr{V}\left(\bar{u}_{i_{v} \boldsymbol{k}} u_{i_{c} \boldsymbol{k}}, \bar{u}_{j_{c} \boldsymbol{k}^{\prime}} u_{j_{v} \boldsymbol{k}^{\prime}}\right),  \tag{2-20}\\
W_{A}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right) & =\frac{1}{N_{k}} W_{\boldsymbol{k}-\boldsymbol{k}^{\prime}}\left(\bar{u}_{j_{c} \boldsymbol{k}^{\prime}} u_{i_{c} \boldsymbol{k}}, \bar{u}_{j_{v} \boldsymbol{k}^{\prime}} u_{i_{v} \boldsymbol{k}}\right), \\
W_{B}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right) & =\frac{1}{N_{k}} W_{\boldsymbol{k}-\boldsymbol{k}^{\prime}}\left(\bar{u}_{j_{v} \boldsymbol{k}^{\prime}} u_{i_{c} \boldsymbol{k}}, \bar{u}_{j_{c} \boldsymbol{k}^{\prime}} u_{i_{v} \boldsymbol{k}}\right) .
\end{align*}
$$

In (2-18) and (2-19), $f, g$ are periodic functions in the unit cell, and can be represented using their Fourier representations. For instance,
$$
\begin{equation*}
f(\boldsymbol{r})=\sum_{\boldsymbol{G} \in \mathbb{L}^{*}} \hat{f}(\boldsymbol{G}) e^{\mathrm{i} \boldsymbol{G} \cdot \boldsymbol{r}}, \tag{2-21}
\end{equation*}
$$
and its Fourier coefficients can be computed as
$$
\begin{equation*}
\hat{f}(\boldsymbol{G})=\frac{1}{|\Omega|} \int_{\Omega} e^{-\mathrm{i} \boldsymbol{G} \cdot \boldsymbol{r}} f(\boldsymbol{r}) \mathrm{d} \boldsymbol{r} \tag{2-22}
\end{equation*}
$$

Hence, Parseval's identity reads
$$
\begin{equation*}
\int_{\Omega} \bar{f}(\boldsymbol{r}) g(\boldsymbol{r}) \mathrm{d} \boldsymbol{r}=|\Omega| \sum_{\boldsymbol{G} \in \mathbb{L}^{*}} \overline{\hat{f}}(\boldsymbol{G}) \hat{g}(\boldsymbol{G}) . \tag{2-23}
\end{equation*}
$$

Both of the kernels $V, W$ satisfy the translation symmetry
$$
\begin{equation*}
V\left(\boldsymbol{r}+\boldsymbol{R}, \boldsymbol{r}^{\prime}+\boldsymbol{R}\right)=V\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right), \quad W\left(\boldsymbol{r}+\boldsymbol{R}, \boldsymbol{r}^{\prime}+\boldsymbol{R}\right)=W\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) \text { for all } \boldsymbol{R} \in \mathbb{L} . \tag{2-24}
\end{equation*}
$$

Equation (2-24) also defines the values of $V, W$ for $\boldsymbol{r}, \boldsymbol{r}^{\prime}$ beyond the supercell $\Omega^{\ell}$. The Fourier representation of $V$ takes the form
$$
\begin{equation*}
V\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right)=\frac{1}{\left|\Omega^{\ell}\right|} \sum_{\boldsymbol{k} \in \mathscr{K} \ell} \sum_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} e^{\mathrm{i}(\boldsymbol{k}+\boldsymbol{G}) \cdot \boldsymbol{r}} \widehat{V}_{\boldsymbol{k}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right) e^{-\mathrm{i}\left(\boldsymbol{k}+\boldsymbol{G}^{\prime}\right) \cdot \boldsymbol{r}^{\prime}}, \tag{2-25}
\end{equation*}
$$
and the Fourier coefficients can be computed as
$$
\begin{equation*}
\widehat{V}_{\boldsymbol{k}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right)=\frac{1}{\left|\Omega^{\ell}\right|} \int_{\Omega^{\ell} \times \Omega^{\ell}} \mathrm{d} \boldsymbol{r} \mathrm{~d} \boldsymbol{r}^{\prime} e^{-\mathrm{i}(\boldsymbol{k}+\boldsymbol{G}) \cdot \boldsymbol{r}} V\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) e^{\mathrm{i}\left(\boldsymbol{k}+\boldsymbol{G}^{\prime}\right) \cdot \boldsymbol{r}^{\prime}} \tag{2-26}
\end{equation*}
$$

Similarly, the Fourier representation for $W$ can be defined.

It should be noted that the Coulomb kernel $V$ only depends on the distance between $\boldsymbol{r}$ and $\boldsymbol{r}^{\prime}$, i.e., it has the further translational symmetry property that
$$
\begin{equation*}
V\left(\boldsymbol{r}+\boldsymbol{r}^{\prime \prime}, \boldsymbol{r}^{\prime}+\boldsymbol{r}^{\prime \prime}\right)=V\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) \quad \text { for all } \boldsymbol{r}^{\prime \prime} \in \Omega^{\ell} . \tag{2-27}
\end{equation*}
$$

As a result, its Fourier transform $\widehat{V}_{\boldsymbol{k}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right)$ can be simplified into a diagonal matrix
$$
\begin{equation*}
\widehat{V}_{\boldsymbol{k}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right)=\frac{4 \pi}{|\boldsymbol{k}+\boldsymbol{G}|^{2}} \delta_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} \tag{2-28}
\end{equation*}
$$

In fact, the Coulomb kernel periodized with respect to the supercell $\Omega^{\ell}$ is defined to be the inverse Fourier transform of (2-28).

Using such notation, we have
$$
\begin{align*}
& \int_{\Omega^{\ell}} V\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) g\left(\boldsymbol{r}^{\prime}\right) \mathrm{d} \boldsymbol{r}^{\prime} \\
& =\frac{1}{\left|\Omega^{\ell}\right|} \int_{\Omega^{\ell}} \mathrm{d} \boldsymbol{r}^{\prime} \sum_{\boldsymbol{k} \in \mathscr{K}} \sum_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} e^{\mathrm{i}(\boldsymbol{k}+\boldsymbol{G}) \cdot \boldsymbol{r}} \widehat{V}_{\boldsymbol{k}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right) e^{-\mathrm{i}\left(\boldsymbol{k}+\boldsymbol{G}^{\prime}\right) \cdot \boldsymbol{r}^{\prime}} g\left(\boldsymbol{r}^{\prime}\right) \\
& =\frac{1}{\left|\Omega^{\ell}\right|} \sum_{\boldsymbol{R} \in \mathbb{L}} \int_{\Omega} \mathrm{d} \boldsymbol{r}^{\prime} \sum_{\boldsymbol{k} \in \mathscr{K} \ell} \sum_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} e^{\mathrm{i}(\boldsymbol{k}+\boldsymbol{G}) \cdot \boldsymbol{r}} \widehat{V}_{\boldsymbol{k}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right) e^{-\mathrm{i}\left(\boldsymbol{k}+\boldsymbol{G}^{\prime}\right) \cdot\left(\boldsymbol{r}^{\prime}+\boldsymbol{R}\right)} g\left(\boldsymbol{r}^{\prime}+\boldsymbol{R}\right) \\
& =\frac{1}{\left|\Omega^{\ell}\right|} \int_{\Omega} \mathrm{d} \boldsymbol{r}^{\prime} \sum_{\boldsymbol{k} \in \mathscr{K}^{\ell}} \sum_{\boldsymbol{R} \in \mathbb{L}} e^{-\mathrm{i} \boldsymbol{k} \cdot \boldsymbol{R}} \sum_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} e^{\mathrm{i}(\boldsymbol{k}+\boldsymbol{G}) \cdot \boldsymbol{r}} \widehat{V}_{\boldsymbol{k}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right) e^{-\mathrm{i}\left(\boldsymbol{k}+\boldsymbol{G}^{\prime}\right) \cdot \boldsymbol{r}^{\prime}} g\left(\boldsymbol{r}^{\prime}\right) \tag{2-29}
\end{align*}
$$

Here we have used $e^{-\mathrm{i} \boldsymbol{G}^{\prime} \cdot \boldsymbol{R}}=1$ and the fact that $g$ is periodic with respect to the unit cell $\Omega$, as well as the identity
$$
\begin{equation*}
\int_{\Omega^{\ell}} f\left(\boldsymbol{r}^{\prime}\right) \mathrm{d} \boldsymbol{r}^{\prime}=\sum_{\boldsymbol{R} \in \mathbb{L}} \int_{\Omega} f\left(\boldsymbol{r}^{\prime}+\boldsymbol{R}\right) \mathrm{d} \boldsymbol{r}^{\prime} \tag{2-30}
\end{equation*}
$$

Furthermore, from (2-22) and the identity
$$
\sum_{\boldsymbol{R} \in \mathbb{L}} e^{-\mathrm{i} \boldsymbol{k} \cdot \boldsymbol{R}}=N_{k} \delta_{\boldsymbol{k}, 0}
$$
we have
$$
\begin{align*}
\int_{\Omega^{\ell}} V\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) g\left(\boldsymbol{r}^{\prime}\right) \mathrm{d} \boldsymbol{r}^{\prime} & =\frac{1}{|\Omega|} \int_{\Omega} \mathrm{d} \boldsymbol{r}^{\prime} \sum_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} e^{\mathrm{i} \boldsymbol{G} \cdot \boldsymbol{r}} \widehat{V}_{\mathbf{0}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right) e^{-\mathrm{i} \boldsymbol{G}^{\prime} \cdot \boldsymbol{r}^{\prime}} g\left(\boldsymbol{r}^{\prime}\right) \\
& =\sum_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} e^{\mathrm{i} \boldsymbol{G} \cdot \boldsymbol{r}} \widehat{V}_{\mathbf{0}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right) \hat{g}\left(\boldsymbol{G}^{\prime}\right) \tag{2-31}
\end{align*}
$$

Compared to (2-28), the definition of $\widehat{V}_{\mathbf{0}}$ should be modified to
$$
\widehat{V}_{\mathbf{0}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right)= \begin{cases}\left(4 \pi /|\boldsymbol{G}|^{2}\right) \delta_{\boldsymbol{G}, \boldsymbol{G}^{\prime}}, & \boldsymbol{G} \neq \mathbf{0}  \tag{2-32}\\ 0, & \boldsymbol{G}=\mathbf{0}\end{cases}
$$

Another way to understand (2-32) is that it can only be applied to a mean-zero function $g(\boldsymbol{r})$, such that $\hat{g}(\mathbf{0})=0$. In other words, $g$ should be in the range of the Laplacian operator with the periodic boundary condition. This is indeed correct for BSE calculations, due to the orthogonality condition between the valence and conduction bands
$$
\int_{\Omega} \bar{u}_{i_{c} k}(\boldsymbol{r}) u_{i_{v} k}(\boldsymbol{r}) \mathrm{d} \boldsymbol{r}=0
$$

This implies
$$
\begin{align*}
\mathscr{V}(f, g) & =\frac{1}{N_{k}} \int_{\Omega^{\ell}} \bar{f}(\boldsymbol{r}) \sum_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} e^{\mathrm{i} \boldsymbol{G} \cdot \boldsymbol{r}} \widehat{V}_{\mathbf{0}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right) \hat{g}\left(\boldsymbol{G}^{\prime}\right) \\
& =\int_{\Omega} \bar{f}(\boldsymbol{r}) \sum_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} e^{\mathrm{i} \boldsymbol{G} \cdot \boldsymbol{r}} \widehat{V}_{\mathbf{0}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right) \hat{g}\left(\boldsymbol{G}^{\prime}\right) \\
& =|\Omega| \sum_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} \overline{\hat{f}}(\boldsymbol{G}) \widehat{V}_{\mathbf{0}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right) \hat{g}\left(\boldsymbol{G}^{\prime}\right) \\
& =|\Omega| \sum_{\boldsymbol{G} \neq \mathbf{0}} \frac{4 \pi}{|\boldsymbol{G}|^{2}} \overline{\hat{f}}(\boldsymbol{G}) \hat{g}(\boldsymbol{G}) \tag{2-33}
\end{align*}
$$

Similarly for the $W$ part,
$$
\begin{align*}
& \int_{\Omega^{\ell}} e^{-\mathrm{i} \boldsymbol{q} \cdot\left(\boldsymbol{r}-\boldsymbol{r}^{\prime}\right)} W\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) g\left(\boldsymbol{r}^{\prime}\right) \mathrm{d} \boldsymbol{r}^{\prime} \\
& =\frac{1}{\left|\Omega^{\ell}\right|} \int_{\Omega^{\ell}} \mathrm{d} \boldsymbol{r}^{\prime} e^{-\mathrm{i} \boldsymbol{q} \cdot\left(\boldsymbol{r}-\boldsymbol{r}^{\prime}\right)} \sum_{\boldsymbol{k} \in \mathscr{K}} \sum_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} e^{\mathrm{i}(\boldsymbol{k}+\boldsymbol{G}) \cdot \boldsymbol{r}} \widehat{W}_{\boldsymbol{k}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right) e^{-\mathrm{i}\left(\boldsymbol{k}+\boldsymbol{G}^{\prime}\right) \cdot \boldsymbol{r}^{\prime}} g\left(\boldsymbol{r}^{\prime}\right) \\
& =\frac{1}{\left|\Omega^{\ell}\right|} \int_{\Omega} \mathrm{d} \boldsymbol{r}^{\prime} e^{\mathrm{i}(\boldsymbol{k}-\boldsymbol{q}) \cdot\left(\boldsymbol{r}-\boldsymbol{r}^{\prime}\right)} \sum_{\boldsymbol{k} \in \mathscr{K}^{\ell}} \sum_{\boldsymbol{R} \in \mathbb{L}} e^{-\mathrm{i}(\boldsymbol{k}-\boldsymbol{q}) \cdot \boldsymbol{R}} \\
& \quad \times \sum_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} e^{\mathrm{i} \boldsymbol{G} \cdot \boldsymbol{r}} \widehat{W}_{\boldsymbol{k}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right) e^{-\mathrm{i} \boldsymbol{G}^{\prime} \cdot \boldsymbol{r}^{\prime}} g\left(\boldsymbol{r}^{\prime}\right) \tag{2-34}
\end{align*}
$$

In order to obtain a nonvanishing quantity in the equation above, note that the quantity $\sum_{\boldsymbol{R} \in \mathbb{L}} e^{-\mathrm{i}(\boldsymbol{k}-\boldsymbol{q}) \cdot \boldsymbol{R}}=N_{k}$ if $\boldsymbol{k}-\boldsymbol{q} \in \mathbb{L}^{*}$, and is otherwise 0 . Therefore, the summation with respect to $\boldsymbol{k}$ should be restricted to those satisfying
$$
\boldsymbol{k}-\boldsymbol{q}=\boldsymbol{G}^{\prime \prime}, \quad \boldsymbol{G}^{\prime \prime} \in \mathbb{1}^{*}
$$

Since $\boldsymbol{k}$ is restricted to the first Brillouin zone, there is a unique $\boldsymbol{G}^{\prime \prime}$ (and therefore $\boldsymbol{k}$ ) for each given $\boldsymbol{q}$ satisfying this relation. Also note that $\boldsymbol{k}-\boldsymbol{q}$ may exceed the first Brillouin zone. In other words, it is indeed possible to have $\boldsymbol{G}^{\prime \prime} \neq \mathbf{0}$. Then for a
given $\boldsymbol{q}$,
$$
\begin{align*}
& \int_{\Omega^{\ell}} e^{-\mathrm{i} \boldsymbol{q} \cdot\left(\boldsymbol{r}-\boldsymbol{r}^{\prime}\right)} W\left(\boldsymbol{r}, \boldsymbol{r}^{\prime}\right) g\left(\boldsymbol{r}^{\prime}\right) \mathrm{d} \boldsymbol{r}^{\prime} \\
&=\frac{1}{|\Omega|} \int_{\Omega} \mathrm{d} \boldsymbol{r}^{\prime} \sum_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} e^{\mathrm{i}\left(\boldsymbol{G}+\boldsymbol{G}^{\prime \prime}\right) \cdot \boldsymbol{r}} \widehat{W}_{\boldsymbol{G}^{\prime \prime}+\boldsymbol{q}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right) e^{-\mathrm{i}\left(\boldsymbol{G}^{\prime}+\boldsymbol{G}^{\prime \prime}\right) \cdot \boldsymbol{r}^{\prime}} g\left(\boldsymbol{r}^{\prime}\right) \\
&=\sum_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} e^{\mathrm{i}\left(\boldsymbol{G}+\boldsymbol{G}^{\prime \prime}\right) \cdot \boldsymbol{r}} \widehat{W}_{\boldsymbol{G}^{\prime \prime}+\boldsymbol{q}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right) \hat{g}\left(\boldsymbol{G}^{\prime}+\boldsymbol{G}^{\prime \prime}\right) \\
&=\sum_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} e^{\mathrm{i} \boldsymbol{G} \cdot \boldsymbol{r}} \widehat{W}_{\boldsymbol{G}^{\prime \prime}+\boldsymbol{q}}\left(\boldsymbol{G}-\boldsymbol{G}^{\prime \prime}, \boldsymbol{G}^{\prime}-\boldsymbol{G}^{\prime \prime}\right) \hat{g}\left(\boldsymbol{G}^{\prime}\right) \\
&=\sum_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} e^{\mathrm{i} \boldsymbol{G} \cdot \boldsymbol{r}} \widehat{W}_{\boldsymbol{q}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right) \hat{g}\left(\boldsymbol{G}^{\prime}\right) \tag{2-35}
\end{align*}
$$

In the last equality, we have used the definition of the Fourier coefficients in (2-26). We then readily have
$$
\begin{equation*}
\mathscr{W}_{\boldsymbol{q}}(f, g)=|\Omega| \sum_{\boldsymbol{G}, \boldsymbol{G}^{\prime}} \overline{\hat{f}}(\boldsymbol{G}) \widehat{W}_{\boldsymbol{q}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right) \hat{g}\left(\boldsymbol{G}^{\prime}\right) . \tag{2-36}
\end{equation*}
$$

Therefore, despite that $W_{\boldsymbol{q}}(f, g)$ is significantly more complex to define, the resulting formula in the Fourier representation is remarkably similar to the form of $\mathscr{V}(f, g)$.

\section*{3. Interpolative separable density fitting for periodic systems}

In order to reduce the computational complexity, we seek to minimize the number of integrals in (2-16). We will use the interpolative separable density fitting decomposition (ISDF) [19; 20]. For periodic systems, we first consider the general form of decomposition
$$
\begin{equation*}
Z_{i \boldsymbol{k}, j \boldsymbol{k}^{\prime}}(\boldsymbol{r}):=u_{i \boldsymbol{k}}(\boldsymbol{r}) \bar{u}_{j \boldsymbol{k}^{\prime}}(\boldsymbol{r}) \approx \sum_{\mu=1}^{N_{\mu}} \zeta_{\mu}(\boldsymbol{r}) u_{i \boldsymbol{k}}\left(\hat{\boldsymbol{r}}_{\mu}\right) \bar{u}_{j \boldsymbol{k}^{\prime}}\left(\hat{\boldsymbol{r}}_{\mu}\right) . \tag{3-1}
\end{equation*}
$$

When the unit cell is discretized into a uniform grid $\left\{\boldsymbol{r}_{n}\right\}_{n=1}^{N_{g}}, Z$ can be viewed as a matrix with its row index being $r$, and the column index being a multi-index $\left(i \boldsymbol{k}, j \boldsymbol{k}^{\prime}\right)$. The matrix size is thus $N_{g} \times N^{2} N_{k}^{2}$ (recall that $N=N_{v}+N_{c}$ ). For a given $\boldsymbol{r}$, $u_{i k}(\boldsymbol{r}) \bar{u}_{j k^{\prime}}(\boldsymbol{r})$ can be viewed as a row vector of size $N^{2} N_{k}^{2}$. The ISDF decomposition then states that all such matrix rows can be approximately expanded using a linear combination of matrix rows with respect to a selected set of interpolation points $\left\{\hat{\boldsymbol{r}}_{\mu}\right\}_{\mu=1}^{N_{\mu}} \subset\left\{\boldsymbol{r}_{i}\right\}_{i=1}^{N_{g}}$. The coefficients of such a linear combination, or interpolating vectors, are denoted by $\left\{\zeta_{\mu}(\boldsymbol{r})\right\}_{\mu=1}^{N_{\mu}}$. Here $N_{\mu}$ can be interpreted as the numerical rank of the ISDF decomposition.

The compression of the pair products $u_{i k}(\boldsymbol{r}) \bar{u}_{j k^{\prime}}(\boldsymbol{r})$ can be understood from the following two limits. First, if only the $\Gamma$-point is used to sample the Brillouin zone, we find that there are $N_{v} N_{c} \sim N^{2}$ pairs of functions. However, the number of grid points $N_{g}$ only scales linearly with respect to $N$. Hence, the numerical rank of the pair products must scale asymptotically as $\mathscr{O}(N)$. In fact, when all orbitals are smooth functions, we can expect the numerical rank $N_{\mu}$ to be much lower than $N_{g}$. This statement has been confirmed by recent analysis [17]. Second, if a large number of $\boldsymbol{k}$-points are used to discretize the Brillouin zone, $N_{v}, N_{c}$ are often relatively small, and the number of grid points in the unit cell $N_{g}$ does not increase with respect to $N_{k}$. Hence, as $N_{k}$ increases, we may also expect that the numerical rank $N_{\mu}$ will be determined by smoothness of $u$ with respect to $\boldsymbol{r}, \boldsymbol{k}$, and is asymptotically independent of $N_{k}$. This is indeed what has been observed numerically [20]. Throughout the discussion below, we will focus on the second scenario, i.e., we will explicitly write down the scaling with respect to $N_{g}, N$, and $N_{k}$, but we will primarily focus on the scaling with respect to $N_{k}$.

Assuming the interpolation points $\left\{\hat{\boldsymbol{r}}_{\mu}\right\}_{\mu=1}^{N_{\mu}}$ are already chosen, the interpolation vectors can be efficiently evaluated using a least squares method as follows [12]. Using a linear algebra notation, (3-1) can be written as
$$
\begin{equation*}
Z \approx \Theta C . \tag{3-2}
\end{equation*}
$$

Here $\Theta=\left[\zeta_{1}, \zeta_{2}, \ldots, \zeta_{N_{\mu}}\right]$ contains the interpolating vectors. Each column of $C$ indexed by ( $i \boldsymbol{k}, j \boldsymbol{k}^{\prime}$ ) is given by
$$
\left[u_{i \boldsymbol{k}}\left(\hat{\boldsymbol{r}}_{1}\right) \bar{u}_{j \boldsymbol{k}^{\prime}}\left(\hat{\boldsymbol{r}}_{1}\right), \ldots, u_{i \boldsymbol{k}}\left(\hat{\boldsymbol{r}}_{\mu}\right) \bar{u}_{j \boldsymbol{k}^{\prime}}\left(\hat{\boldsymbol{r}}_{\mu}\right), \ldots, u_{i \boldsymbol{k}}\left(\hat{\boldsymbol{r}}_{N_{\mu}}\right) \bar{u}_{j \boldsymbol{k}^{\prime}}\left(\hat{\boldsymbol{r}}_{N_{\mu}}\right)\right]^{\top} .
$$

Equation (3-2) is an over-determined linear system with respect to the interpolation vectors $\Theta$. The least squares approximation to the solution is given by
$$
\begin{equation*}
\Theta=Z C^{*}\left(C C^{*}\right)^{-1} . \tag{3-3}
\end{equation*}
$$

Due to the tensor product structure of $Z$ and $C$, the matrix-matrix multiplications $Z C^{*}$ and $C C^{*}$ can be carried out efficiently [12], with computational $\operatorname{cost} \odot\left(N_{g} N_{\mu} N N_{k}\right)$ and $\mathscr{O}\left(N_{\mu}^{2} N N_{k}\right)$, respectively. The cost of inverting the matrix $C C^{*}$ is $\mathscr{O}\left(N_{\mu}^{3}\right)$, and the overall cost of evaluating $\Theta$ is thus bounded by $\mathbb{O}\left(N_{g} N_{\mu} N N_{k}+N_{\mu}^{3}+N_{g} N_{\mu}^{2}\right)$. Hence, the cost scales cubically with respect to the number of electrons in the unit cell, and linearly with respect to the number of $\boldsymbol{k}$-points.

Equation (3-1) is the general form of ISDF. In the BSE calculations, we may further distinguish whether $i, j$ should take valence or conduction band indices only, as well as whether $\boldsymbol{k}, \boldsymbol{k}^{\prime}$ can be set to be the same. For instance, (2-17) suggests
that in order to compress $V_{A}, V_{B}$, we only need the ISDF decomposition
$$
\begin{equation*}
Z_{i_{c} i_{v} k}^{V}(\boldsymbol{r}):=u_{i_{c} k}(\boldsymbol{r}) \bar{u}_{i_{v} k}(\boldsymbol{r}) \approx \sum_{\mu=1}^{N_{\mu}^{V}} \zeta_{\mu}^{V}(\boldsymbol{r}) u_{i_{c} k}\left(\hat{\boldsymbol{r}}_{\mu}\right) \bar{u}_{i_{v} k}\left(\hat{\boldsymbol{r}}_{\mu}\right) . \tag{3-4}
\end{equation*}
$$

Note that the number of columns of the matrix $Z^{V}$ is only $N_{v} N_{c} N_{k}$, and the number of fitting functions $N_{\mu}^{V}$ can be chosen to be less than $N_{\mu}$. The computation of $W_{A}, W_{B}$ requires the general ISDF format (3-1).

The interpolation points $\left\{\hat{\boldsymbol{r}}_{\mu}\right\}_{\mu=1}^{N_{\mu}}$ can be chosen in different ways. In this work we employ a randomized variant of QR with column pivoting (QRCP) [19; 20; 9]. Another recently developed method is based on the centroidal Voronoi decomposition (CVT) [8]. We observed that in our examples it is even possible to work with coarse uniform grids as interpolation points, reducing the computational effort for finding the points to essentially zero while only slightly increasing the error. Since the computation of interpolation points is not the bottleneck in our problem, however, we stick to the previously developed techniques.

\section*{4. Fast algorithm for applying the BSH to a vector}

Once the ISDF decomposition is obtained, we may compute the matrix elements
$$
\begin{equation*}
\tilde{V}_{A, \mu \nu}=\mathscr{V}\left(\zeta_{\mu}^{V}, \zeta_{\nu}^{V}\right), \quad \tilde{V}_{B, \mu \nu}=\mathscr{V}\left(\zeta_{\mu}^{V}, \bar{\zeta}_{\nu}^{V}\right), \quad \mu, \nu=1, \ldots, N_{\mu}^{V} \tag{4-1}
\end{equation*}
$$
and similarly
$$
\begin{equation*}
\tilde{W}_{\boldsymbol{q}, \mu \nu}=\tilde{W}_{\boldsymbol{q}}\left(\zeta_{\mu}, \zeta_{\nu}\right), \quad \mu, \nu=1, \ldots, N_{\mu} . \tag{4-2}
\end{equation*}
$$

The expressions in (2-17) can then be approximated in the ISDF format as
$$
\begin{align*}
& V_{A}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right) \approx \frac{1}{N_{k}} \sum_{\mu, v=1}^{N_{\mu}^{V}} \bar{u}_{i_{c} \boldsymbol{k}}\left(\hat{\boldsymbol{r}}_{\mu}\right) u_{i_{v} \boldsymbol{k}}\left(\hat{\boldsymbol{r}}_{\mu}\right) \tilde{V}_{A, \mu v} \bar{u}_{j_{v} \boldsymbol{k}^{\prime}}\left(\hat{\boldsymbol{r}}_{v}\right) u_{j_{c} \boldsymbol{k}^{\prime}}\left(\hat{\boldsymbol{r}}_{v}\right), \\
& V_{B}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right) \approx \frac{1}{N_{k}} \sum_{\mu, v=1}^{N_{\mu}^{V}} \bar{u}_{i_{c} \boldsymbol{k}}\left(\hat{\boldsymbol{r}}_{\mu}\right) u_{i_{v} \boldsymbol{k}}\left(\hat{\boldsymbol{r}}_{\mu}\right) \widetilde{V}_{B, \mu v} \bar{u}_{j_{c} \boldsymbol{k}^{\prime}}\left(\hat{\boldsymbol{r}}_{v}\right) u_{j_{v} \boldsymbol{k}^{\prime}}\left(\hat{\boldsymbol{r}}_{v}\right),  \tag{4-3}\\
& W_{A}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right)=\frac{1}{N_{k}} \sum_{\mu, v=1}^{N_{\mu}} \bar{u}_{i_{c} \boldsymbol{k}}\left(\hat{\boldsymbol{r}}_{\mu}\right) u_{j_{c} \boldsymbol{k}^{\prime}}\left(\hat{\boldsymbol{r}}_{\mu}\right) \widetilde{W}_{\boldsymbol{k}-\boldsymbol{k}^{\prime}, \mu v} \bar{u}_{j_{v} \boldsymbol{k}^{\prime}}\left(\hat{\boldsymbol{r}}_{v}\right) u_{i_{v} \boldsymbol{k}}\left(\hat{\boldsymbol{r}}_{v}\right), \\
& W_{B}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right)=\frac{1}{N_{k}} \sum_{\mu, v=1}^{N_{\mu}} \bar{u}_{i_{c} \boldsymbol{k}}\left(\hat{\boldsymbol{r}}_{\mu}\right) u_{j_{v} \boldsymbol{k}^{\prime}}\left(\hat{\boldsymbol{r}}_{\mu}\right) \widetilde{W}_{\boldsymbol{k}-\boldsymbol{k}^{\prime}, \mu v} \bar{u}_{j_{c} \boldsymbol{k}^{\prime}}\left(\hat{\boldsymbol{r}}_{v}\right) u_{i_{v} \boldsymbol{k}}\left(\hat{\boldsymbol{r}}_{v}\right) .
\end{align*}
$$

In order to use the Fourier representation (2-33) and (2-36), we first need to perform Fourier transforms for $\left\{\zeta_{\mu}^{V}\right\}$ and $\left\{\zeta_{\mu}\right\}$. Using the fast Fourier transform (FFT), and assuming that the number of Fourier coefficients $\boldsymbol{G}$ is also $N_{g}$, the computational cost for the Fourier transform scales as $\mathscr{O}\left(N_{\mu}^{V} N_{g} \log N_{g}\right)$ and $\mathscr{O}\left(N_{\mu} N_{g} \log N_{g}\right)$, respectively. The Fourier coefficients $\widehat{V}_{\boldsymbol{k}}$ can be obtained analytically, and we assume the coefficients $\widehat{W}_{\boldsymbol{k}}$ are already provided from, e.g., a GW calculation. The cost for computing $\widetilde{V}_{A}, \widetilde{V}_{B}$ using (2-33) is then $\mathbb{O}\left(\left(N_{\mu}^{V}\right)^{2} N_{g}\right)$. Similarly the cost for computing all $\widetilde{W}_{q}$ matrices is $O\left(N_{\mu}^{2} N_{g} N_{k}\right)$. In particular, the total cost for the initial setup stage scales as $\mathscr{O}\left(N_{k}\right)$ with respect to the number of $\boldsymbol{k}$-points.

After this initial setup stage, each entry of the BSH can be computed with $\mathscr{O}\left(\left(N_{\mu}^{V}\right)^{2}+N_{\mu}^{2}\right)$ operations. If the entire BSH matrix is to be constructed, the cost will be $O\left(N_{\mu}^{2} N_{k}^{2} N_{v}^{2} N_{c}^{2}\right)$.

Below we demonstrate that if we only aim to apply the Hamiltonian $H_{\mathrm{BSE}}$ to an arbitrary vector without ever assembling the full Hamiltonian, the computational cost can be greatly reduced.

For simplicity, let us focus on the case when the Tamm-Dancoff approximation (TDA) is used. Applying the Hamiltonian $H_{\mathrm{BSE}}=D+2 V_{A}-W_{B}$ to a vector $X \in \mathbb{C}^{N_{v} N_{c} N_{k}}$ amounts to evaluating the three terms
$$
\begin{align*}
{[D X]\left(i_{v} i_{c} \boldsymbol{k}\right) } & =\left(\epsilon_{i_{c} \boldsymbol{k}}-\epsilon_{i_{v} \boldsymbol{k}^{\prime}}\right) X\left(i_{v} i_{c} \boldsymbol{k}\right) \\
{\left[V_{A} X\right]\left(i_{v} i_{c} \boldsymbol{k}\right) } & =\sum_{j_{v}, j_{c}, \boldsymbol{k}^{\prime}} V_{A}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right) X\left(j_{v} j_{c} \boldsymbol{k}^{\prime}\right)  \tag{4-4}\\
{\left[W_{A} X\right]\left(i_{v} i_{c} \boldsymbol{k}\right) } & =\sum_{j_{v}, j_{c}, \boldsymbol{k}^{\prime}} W_{A}\left(i_{v} i_{c} \boldsymbol{k}, j_{v} j_{c} \boldsymbol{k}^{\prime}\right) X\left(j_{v} j_{c} \boldsymbol{k}^{\prime}\right)
\end{align*}
$$

Computing the first term for all ( $i_{v} i_{c} \boldsymbol{k}$ ) clearly costs $\mathscr{O}\left(N_{v} N_{c} N_{k}\right)$ operations. We now show that the second and third terms can also be computed efficiently.

Using (4-3), the second term in (4-4) can be regrouped as
$$
\begin{align*}
\frac{1}{N_{k}} \sum_{\mu} \bar{u}_{i_{c} \boldsymbol{k}}\left(\hat{\boldsymbol{r}}_{\mu}\right) u_{i_{v} \boldsymbol{k}}\left(\hat{\boldsymbol{r}}_{\mu}\right) & \left\{\sum_{v} \tilde{V}_{A, \mu v}\right. \\
\times & \left.\left(\sum_{\boldsymbol{k}^{\prime}}\left(\sum_{j_{c}} u_{j_{c} \boldsymbol{k}^{\prime}}\left(\hat{\boldsymbol{r}}_{v}\right)\left(\sum_{j_{v}} \bar{u}_{j_{v} \boldsymbol{k}^{\prime}}\left(\hat{\boldsymbol{r}}_{v}\right) X\left(j_{v} j_{c} \boldsymbol{k}^{\prime}\right)\right)\right)\right)\right\} . \tag{4-5}
\end{align*}
$$

This means one can first perform contractions over $j_{v}, j_{c}$, and $\boldsymbol{k}^{\prime}$ to obtain a quantity that only depends on $\hat{\boldsymbol{r}}_{v}$. The computational complexity is $O\left(N_{\mu}^{V}\left(N_{v} N_{c} N_{k}+N_{c} N_{k}\right)\right)$. The two remaining sums can be computed with $\odot\left(\left(N_{\mu}^{V}\right)^{2}+N_{\mu}^{V} N_{v} N_{c} N_{k}\right)$ operations. The total complexity of computing $V_{A} X$ is bounded by $\mathscr{O}\left(\left(N_{\mu}^{V}\right)^{2}+N_{\mu}^{V} N_{v} N_{c} N_{k}\right)$.

For the third term in (4-4) we obtain
$$
\begin{align*}
\frac{1}{N_{k}} \sum_{v} u_{i_{v} \boldsymbol{k}} & \left(\hat{\boldsymbol{r}}_{v}\right)\left\{\sum_{\mu} \bar{u}_{i_{c} \boldsymbol{k}}\left(\hat{\boldsymbol{r}}_{\mu}\right)\right. \\
& \left.\times\left(\sum_{\boldsymbol{k}^{\prime}} \widetilde{W}_{\boldsymbol{k}-\boldsymbol{k}^{\prime}, \mu v}\left(\sum_{j_{c}} u_{j_{c} \boldsymbol{k}^{\prime}}\left(\hat{\boldsymbol{r}}_{\mu}\right)\left(\sum_{j_{v}} \bar{u}_{j_{v} \boldsymbol{k}^{\prime}}\left(\hat{\boldsymbol{r}}_{v}\right) X\left(j_{v} j_{c} \boldsymbol{k}^{\prime}\right)\right)\right)\right)\right\} . \tag{4-6}
\end{align*}
$$

Here, we exploited the separable structure of the decomposition to reorder the products in such a way that all terms depending on $\boldsymbol{k}$ and $\boldsymbol{k}^{\prime}$ are to the left and right, respectively, of $\widetilde{W}_{\boldsymbol{k}-\boldsymbol{k}^{\prime}, \mu \nu}$. The two innermost contractions over $j_{v}$ and $j_{c}$ result in a quantity that only depends on $\boldsymbol{k}, \hat{\boldsymbol{r}}_{\mu}$, and $\hat{\boldsymbol{r}}_{v}$. The cost for these two steps is $O\left(N_{\mu} N_{k} N_{v} N_{c}+N_{\mu}^{2} N_{k} N_{c}\right)$. The sum over $\boldsymbol{k}^{\prime}$ then has the structure of a discrete convolution, for each fixed $\mu \nu$ pair. Therefore, it can be computed for all $\boldsymbol{k}$ simultaneously in $\mathscr{O}\left(N_{\mu}^{2} N_{k} \log N_{k}\right)$ operations by fast convolution algorithms, e.g., by using the FFT with zero-padded vectors. The remaining summation operations over $\mu$ and $v$ are then obtained with $\mathscr{O}\left(N_{\mu}^{2} N_{c} N_{k}+N_{\mu} N_{v} N_{c} N_{k}\right)$ operations. In total the computation of $W_{A} X$ amounts to $\mathscr{O}\left(N_{\mu} N_{v} N_{c} N_{k}+N_{\mu}^{2} N_{c} N_{k}+N_{\mu}^{2} N_{k} \log N_{k}\right)$ operations.

Combining the results for the three parts of the Hamiltonian, we see that the computational complexity is given by
$$
\mathcal{O}\left(\left(N_{\mu}+N_{\mu}^{V}\right) N_{v} N_{c} N_{k}+\left(N_{\mu}^{V}\right)^{2}+N_{\mu}^{2} N_{c} N_{k}+N_{\mu}^{2} N_{k} \log N_{k}\right)
$$

In particular, the cost with respect to the number of $\boldsymbol{k}$-points only scales as $\mathbb{O}\left(N_{k} \log N_{k}\right)$. This allows us to perform BSE calculations for complex materials which require a very large number of $\boldsymbol{k}$-points.

By avoiding the explicit construction of $H_{\mathrm{BSE}}$, the new algorithm also drastically reduces the storage cost. The storage cost for $H_{\mathrm{BSE}}$ alone is $\mathscr{O}\left(\left(N_{v} N_{c} N_{k}\right)^{2}\right)$. In the new algorithm, the storage cost of $\widehat{W}_{\boldsymbol{q}}$ becomes the dominant component and scales only linearly with respect to $N_{k}$.

As an example, the matrix-free application of $H_{\mathrm{BSE}}$ can be used to compute the optical absorption spectrum, which requires the evaluation of the quantity
$$
\begin{equation*}
\varepsilon_{2}(\omega)=\operatorname{Im}\left[\frac{8 \pi}{|\Omega|} d_{r}^{*}\left((\omega-\mathrm{i} \eta) I-H_{\mathrm{BSE}}\right)^{-1} d_{l}\right] \tag{4-7}
\end{equation*}
$$

Here $d_{r}$ and $d_{l}$ are called the right and left optical transition vectors, and $\eta$ is a broadening factor used to account for the exciton lifetime. We also compute the smallest eigenvalues of $H_{\mathrm{BSE}}$, which are of interest in their own right, as they represent the transition energies of bound excitons in many semiconducting solid state materials.

To observe the absorption spectrum and identify its main peaks, it is possible to use a structure-preserving iterative method instead of explicitly computing all eigenpairs of $H_{\mathrm{BSE}}$. We refer readers to [6; 34] for details of the structure-preserving Lanczos algorithm, which has been implemented in the BSEPACK [35] library. ${ }^{1}$ When TDA is used, the structure-preserving Lanczos reduces to a standard Lanczos algorithm. For the computation of the first eigenvalue we use standard ARPACK [14] routines for Hermitian matrices.

\section*{5. Numerical examples}

To illustrate the efficiency of ISDF for BSE calculations in crystals, we apply the method to compute the excitation modes and absorption spectra of a onedimensional model problem as well as two real material systems, diamond (3D bulk) and graphene (quasi-2D). For both systems, we determine the optical absorption spectra on $\boldsymbol{k}$-grids close to those employed in previously published calculations to demonstrate that our method is suitable for state-of-the-art calculations, both for 3D and quasi-2D materials. We furthermore provide a numerical scaling analysis and a more detailed analysis of the error in the ISDF in the case of the one-dimensional model and diamond. We show that a good approximation of the spectrum can be obtained with a small number of interpolation vectors.

The method was implemented in the programming language Julia [5] and the source code is available. ${ }^{2}$ As the input to our method for the actual materials, we employ the KSDFT single particle orbitals, quasiparticle energies, and screened Coulomb potential computed by exciting [10; 37], an all-electron full-potential code with implementations of density functional theory and many-body perturbation theory. The Tamm-Dancoff approximation is used in all calculations.

All calculation for the proposed method were carried out on a single core of an Intel Core i5-8250U CPU at 1.60 GHz .
5.1. One-dimensional problems. For the one-dimensional problem, we take the single particle orbitals $\psi_{i k}(\boldsymbol{r})$ in (2-16) to be eigenfunctions of a single particle Hamiltonian $\mathscr{H}(\boldsymbol{k})$ in which the effective potential is defined as
$$
V_{\mathrm{eff}}(r)=20 \cos (4 \pi r / L)+0.2 \sin (2 \pi r / L),
$$
where the unit cell size is $|\Omega| \equiv L=1.5$.
The bare Coulomb potential used in (2-16) is chosen to be
$$
\begin{equation*}
V\left(r, r^{\prime}\right)=\frac{1}{\sqrt{\left(r-r^{\prime}\right)^{2}+0.01}}, \tag{5-1}
\end{equation*}
$$

\footnotetext{
${ }^{1}$ https://sites.google.com/a/lbl.gov/bsepack/
${ }^{2}$ https://github.com/fhenneke/BSE_k_ISDF.jl/
}

\begin{figure}
\includegraphics[max width=\textwidth]{https://cdn.mathpix.com/cropped/ca5af182-2f19-48ee-9bf5-df0a26ec22a9-18.jpg?height=393&width=993&top_left_y=131&top_left_x=231}
\captionsetup{labelformat=empty}
\caption{Figure 1. Left: the potentials $V(r, 0)$ and $W(r, 0)$. Right: band structure with coefficients of the lowest eigenfunction for $N_{k}=128$. The areas of the circles on the valence and conduction bands at position $\boldsymbol{k}$ are proportional to $\sum_{i_{c}}\left|X\left(i_{v} i_{c} \boldsymbol{k}\right)\right|^{2}$ and $\sum_{i_{v}}\left|X\left(i_{v} i_{c} \boldsymbol{k}\right)\right|^{2}$.}
\end{figure}
and the screened interaction is chosen as
$$
\begin{equation*}
W\left(r, r^{\prime}\right)=\frac{(3+\sin (2 \pi r / L))\left(3+\cos \left(4 \pi r^{\prime} / L\right)\right)}{16} e^{-\left(r-r^{\prime}\right)^{2} /\left(32 L^{2}\right)} V\left(r, r^{\prime}\right) . \tag{5-2}
\end{equation*}
$$

Compared to the smoothed-out Coulomb potential $V$, the chosen screened interaction $W$ decays exponentially and also contains lattice periodic contributions. The potentials are shown in Figure 1. Both potentials are periodically extended $N_{k}-1$ times outside of the unit cell. The particular structure of the potentials has an influence on the band structure and spectrum of the BSH, but was observed to not significantly impact the convergence behavior or the run time scaling of the ISDF method.

The Bloch functions $u_{i k}$ are sampled on $N_{g}=128$ uniformly distributed grid points within the unit cell, and the number of $\boldsymbol{k}$-points $N_{k}$ ranges from 16 to 4096 in our experiments.

For each $\boldsymbol{k}$-point, the first four eigenstates are treated as the valence states in this model, while the remaining eigenstates are considered as the conduction states, separated by an energy gap from the former. We use all $N_{v}=4$ valence bands and $N_{c}=5$ conduction bands to construct the approximate $H_{\text {BSE }}$. The number of $\boldsymbol{k}$-points was chosen to be $N_{k}=256$ in the error analysis of the ISDF approximation, and varies from 16 to 4096 in the run time analysis and the analysis of the error in the absorption spectrum. The largest resulting Hamiltonian is of size $81920 \times 81920$.

Figure 2 shows how the ISDF approximation error varies with respect to the truncation parameter $N_{\mu}^{i j}$ and how the accuracy of the approximate spectrum of $H_{\mathrm{BSE}}$ changes with respect to the ISDF approximation error.

In the left subfigure, we plot the relative error $\left\|\Theta^{\alpha \beta} C^{\alpha \beta}-Z^{\alpha \beta}\right\|_{F} /\left\|Z^{\alpha \beta}\right\|_{F}$, $\alpha, \beta \in\{v, c\}$, where $\|\cdot\|_{F}$ is the Frobenius norm, for different choices of truncation levels $N_{\mu}$ (or number of interpolation points). As expected, when $N_{\mu}$ is too small, ISDF results in relatively large error. As $N_{\mu}$ becomes slightly larger, the ISDF approximation error decays exponentially with respect to $N_{\mu}$ up to $N_{\mu}=20 \sim 30$.

\begin{figure}
\includegraphics[max width=\textwidth]{https://cdn.mathpix.com/cropped/ca5af182-2f19-48ee-9bf5-df0a26ec22a9-19.jpg?height=446&width=1060&top_left_y=134&top_left_x=199}
\captionsetup{labelformat=empty}
\caption{Figure 2. Left: ISDF approximation error $\|Z-\Theta C\|_{F} /\|Z\|_{F}$ for different choices of $N_{\mu}$. Right: resulting errors in the spectrum of $H_{\mathrm{BSE}}$ for different ISDF error tolerances.}
\end{figure}

\begin{figure}
\includegraphics[max width=\textwidth]{https://cdn.mathpix.com/cropped/ca5af182-2f19-48ee-9bf5-df0a26ec22a9-19.jpg?height=430&width=713&top_left_y=729&top_left_x=371}
\captionsetup{labelformat=empty}
\caption{Figure 3. Run times for the initial setup and individual matrix-free matrix-vector products.}
\end{figure}

At this truncation level, the error is on the order of $10^{-8}$, which is sufficiently small for obtaining a highly accurate approximation of the spectrum of $H_{\text {BSE }}$ as shown in the right subfigure. In this subfigure, we plot the relative error in the first eigenvalue and in the overall optical absorption spectrum against the ISDF error tolerance $Z_{\text {tol }}$. For each $Z_{\text {tol }}$, we choose the smallest truncation parameters $N_{\mu}$ with the resulting error in $Z^{\alpha, \beta}$ being less than or equal to $Z_{\text {tol }}$ for $\alpha, \beta \in\{v, c\}$.

In Figure 3, we plot the timing measurements for both the construction of $\widetilde{V}$ and $\widetilde{W}$ and the multiplication of the approximate $H_{\mathrm{BSE}}$ with a vector with respect to $N_{k}$. In these calculations, the ISDF truncation parameters $N_{\mu}$ are chosen so that the relative error in $Z^{\alpha \beta}$ is below $Z_{\text {tol }}=10^{-5}$. This error tolerance resulted in the choices of $N_{\mu}^{v v}=17, N_{\mu}^{c c}=23$, and $N_{\mu}^{v c}=21$.

As we can see in Figure 3, the scaling of the run time for the construction of $\widetilde{V}$ and $\widetilde{W}$ is nearly linear with respect to $N_{k}$, which is in excellent agreement with the theoretical computational complexity presented in the preceding section. The scaling of the run time for the multiplication of the approximate $H_{\mathrm{BSE}}$ with a vector also looks linear in $N_{k}$. In fact, a more detailed investigation showed that the

\begin{figure}
\includegraphics[max width=\textwidth]{https://cdn.mathpix.com/cropped/ca5af182-2f19-48ee-9bf5-df0a26ec22a9-20.jpg?height=448&width=1043&top_left_y=133&top_left_x=210}
\captionsetup{labelformat=empty}
\caption{Figure 4. Optical absorption spectrum for diamond (left) and graphene (right).}
\end{figure}

\begin{table}
\begin{tabular}{|c|cc|}
\hline parameters & diamond & graphene \\
\hline$N_{v}$ & 4 & 4 \\
$N_{c}$ & 10 & 5 \\
$N_{k}$ & $13 \times 13 \times 13$ & $42 \times 42 \times 1$ \\
$N_{r}$ & $20 \times 20 \times 20$ & $15 \times 15 \times 50$ \\
$N_{\mu}^{v v}$ & 70 & 50 \\
$N_{\mu}^{c c}$ & 220 & 180 \\
$N_{\mu}^{v c}$ & 100 & 60 \\
$N_{\text {iter }}$ & 150 & 100 \\
\hline
\end{tabular}
\captionsetup{labelformat=empty}
\caption{Table 1. Parameters used in the computation of spectra and the benchmarks.}
\end{table}
convolutions in $\boldsymbol{k}$ in the application of $W$ dominate the cost of the matrix-vector multiplications, in good agreement with the theoretical $\mathscr{O}\left(N_{k} \log N_{k}\right)$ complexity shown earlier.

For comparison, without the use of ISDF, the construction of $H_{\mathrm{BSE}}$ is estimated to take about 460000 seconds for $N_{k}=4096$. With our method it took less than 10 seconds.
5.2. Three-dimensional problems. We now compare optical absorption spectra for diamond and graphene computed from the approximate $H_{\text {BSE }}$ constructed via ISDF with corresponding reference spectra. The reference spectra are obtained from the exact $H_{\mathrm{BSE}}$ from the exciting code [10; 37]. The comparison is shown in Figure 4. The reference spectrum for diamond is constructed on a $13 \times 13 \times 13 \boldsymbol{k}$-grid using all 4 valence and 10 conduction states. Fourier components $\widehat{W}_{\boldsymbol{q}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right)$ in (2-35) are calculated up to a cutoff $|\boldsymbol{G}+\boldsymbol{q}| \leq 2.5 \mathrm{a}_{0}^{-1}$, where $\mathrm{a}_{0}$ is the Bohr radius. The screened Coulomb interaction is calculated within the random-phase approximation (RPA) including 100 conduction states. For graphene, the reference spectrum is obtained on a $42 \times 42 \times 1 \boldsymbol{k}$-grid using all 4 valence and 5 conduction states. Fourier

\begin{figure}
\includegraphics[max width=\textwidth]{https://cdn.mathpix.com/cropped/ca5af182-2f19-48ee-9bf5-df0a26ec22a9-21.jpg?height=442&width=1039&top_left_y=136&top_left_x=211}
\captionsetup{labelformat=empty}
\caption{Figure 5. Left: optical absorption spectrum for diamond with differently accurate ISDF approximations. Right: estimated errors in ISDF approximation with different numbers of interpolation points.}
\end{figure}

\begin{table}
\begin{tabular}{|lcrr|}
\hline \multicolumn{4}{|c|}{ error in } \\
$Z_{\text {tol }}$ & absorption function & \multicolumn{2}{c|}{ first eigenvalue } \\
\hline 0.5 & 0.199 & 0.0038 & $(20.7 \mathrm{meV})$ \\
0.1 & 0.056 & 0.0011 & $(6.2 \mathrm{meV})$ \\
0.05 & 0.040 & 0.0006 & $(3.3 \mathrm{meV})$ \\
\hline
\end{tabular}
\captionsetup{labelformat=empty}
\caption{Table 2. Relative (and absolute) errors in the spectrum of $H_{\mathrm{BSE}}$ for different ISDF error tolerances.}
\end{table}
components $\widehat{W}_{\boldsymbol{q}}\left(\boldsymbol{G}, \boldsymbol{G}^{\prime}\right)$ in (2-35) are calculated up to a cutoff $|\boldsymbol{G}+\boldsymbol{q}| \leq 2.0 \mathrm{a}_{0}^{-1}$, and 80 conduction states are included in the RPA calculations for the screened Coulomb potential. The numerical parameters of the reference and approximate calculations are shown in Table 1. The number of interpolation vectors was chosen such that the relative ISDF error was around 0.1 .

We can clearly see that for both diamond and graphene, the approximate optical absorption spectrum matches well with the reference spectrum. In particular, the positions and heights of all major peaks are in good agreement. We should note that, in the case of diamond, the absorption spectrum produced by a $13 \times 13 \times 13 \boldsymbol{k}$-grid is in good agreement with measurements [26] and previous BSE calculations [11]. In the case of graphene, however, larger $\boldsymbol{k}$-grids have been reported for BSE calculations [38] to produce an optical absorption spectrum in good agreement with the experimental result.

Figure 5 shows that the ISDF approximation error can be systematically reduced as we increase the number interpolating vectors $N_{\mu}$. However, Figure 4 shows that the approximate absorption spectrum is already in good agreement with the reference spectrum, when the relative ISDF approximation error is at 0.1 . Thus, it seems unnecessary to use a larger number of interpolation vectors in these cases. This observation is corroborated by the relative difference between the first eigenvalue

\begin{figure}
\includegraphics[max width=\textwidth]{https://cdn.mathpix.com/cropped/ca5af182-2f19-48ee-9bf5-df0a26ec22a9-22.jpg?height=432&width=714&top_left_y=141&top_left_x=371}
\captionsetup{labelformat=empty}
\caption{Figure 6. Run times for the initial setup and individual matrix-free matrix-vector products.}
\end{figure}
of the approximate $H_{\mathrm{BSE}}$ computed using ARPACK and that of reference $H_{\mathrm{BSE}}$ constructed in exciting shown in Table 2. With a relative ISDF approximation error of $Z_{\text {tol }}=0.1$, the error in the first BSE eigenvalue is below 10 meV in both examples shown here.

To illustrate the run time scaling of the method in the 3D examples, we measure the time it takes to construct the approximate $H_{\mathrm{BSE}} \mathrm{via}$ ISDF as well as the time it takes to multiply the resulting $H_{\mathrm{BSE}}$ with vectors for the diamond example. We use $\boldsymbol{k}$-grids of sizes $N_{k}=n_{k} \times n_{k} \times n_{k}$ for $n_{k} \in\{2,3,4,5,7,9,13\}$. The resulting timing measurements are plotted in Figure 6. It can be seen that the run time for constructing the approximate $H_{\mathrm{BSE}}$ scales linearly with the number of $\boldsymbol{k}$-points. The multiplication of $H_{\mathrm{BSE}}$ with vectors scales as $\mathscr{O}\left(N_{k} \log N_{k}\right)$ for sufficiently large $N_{k}$. As in the model problem, the convolutions in $\boldsymbol{k}$ in the application of $W$ dominate the cost of the matrix-vector multiplications. For comparison, computing the ISDF decomposition of the Hamiltonian for the case $N_{k}=13^{3}$ took 147 seconds, whereas the full assembly of the Hamiltonian took about 6 hours in exciting on 13 compute nodes with 13 cores each. The optical absorption function was obtained by running about 150 Lanczos steps, which amounts to about 24 minutes for each fixed direction $(x, y$, and $z$ ), compared to almost 4 hours required in the exciting code for the full diagonalization on 13 compute nodes.

\section*{6. Conclusion}

In this paper, we examined the possibility of using the ISDF technique to reduce the computational complexity of BSH construction and the subsequent iterative approximation of the optical absorption spectrum and excitation energies of electronhole (exciton) pairs for solids. For periodic systems, a fine $\boldsymbol{k}$-point sampling in the Brillouin zone is often required to produce accurate results, whereas the number of bands per $\boldsymbol{k}$-point required to construct the bare exchange and screened direct kernels of the BSH is relatively small. We showed that the complexity of the ISDF
procedure scales linearly with respect to the number of $\boldsymbol{k}$-points ( $N_{k}$ ) when the ranks of the approximate bare exchange and screened direct kernels produced by the ISDF procedure are chosen to be independent of $N_{k}$. By keeping the bare exchange and screened direct kernels in the low-rank decomposed form produced by the ISDF procedure, an iterative method used to obtain the optical absorption spectrum and selected excitation energies (eigenvalues of the BSH) can be implemented with cost scaling as $\mathscr{O}\left(N_{k} \log N_{k}\right)$. Our numerical experiments, which were performed on a 1D model as well as two different types of actual materials (diamond and graphene), confirm our complexity analysis. They demonstrate that the ISDF technique can indeed significantly reduce the cost of BSE calculation for solids while maintaining the same accuracy provided by a standard BSE calculation implemented in the software exciting. Our current implementation of the ISDF technique is done using the Julia programming language for a single node. A distributed parallel implementation is needed to accommodate a much finer $\boldsymbol{k}$-point sampling which is required in the case of the graphene example to produce a computed absorption spectrum that matches with experimental results.

\section*{Acknowledgments}

This work was partially supported by the U.S. Department of Energy (DOE) under grant DE-SC0017867 (Lin), by the Center for Computational Study of Excited-State Phenomena in Energy Materials (C2SEPEM) at the Lawrence Berkeley National Laboratory, which is funded by the DOE, Office of Science, Basic Energy Sciences, Materials Sciences and Engineering Division, under contract number DE-AC0205CH11231 (Yang), by the Scientific Discovery Through Advanced Computing (SciDAC) program, and by the CAMERA program (Lin and Yang). Within a framework of cooperation between the University of California, Berkeley and the Freie Universität Berlin, the latter sponsored an extended visit of Henneke and Klein in Berkeley. We thank Wei Hu, Meiyue Shao, and Kyle Thicke for helpful discussions. Draxl and Klein thank the Institute for Pure and Applied Mathematics at the University of California, Los Angeles for its support during the 2013 fall program on "Materials for a sustainable energy future" and for creating the inspiring scientific atmosphere that initiated their collaboration.

\section*{References}
[1] S. Albrecht, G. Onida, and L. Reining, Ab initio calculation of the quasiparticle spectrum and excitonic effects in $\mathrm{Li}_{2} \mathrm{O}$, Phys. Rev. B 55 (1997), no. 16, 10278-10281.
[2] N. W. Ashcroft and N. D. Mermin, Solid state physics, Harcourt, New York, 1976.
[3] P. Benner, V. Khoromskaia, and B. N. Khoromskij, A reduced basis approach for calculation of the Bethe-Salpeter excitation energies by using low-rank tensor factorisations, Mol. Phys. 114 (2016), no. 7-8, 1148-1161.
[4] P. Benner, S. Dolgov, V. Khoromskaia, and B. N. Khoromskij, Fast iterative solution of the Bethe-Salpeter eigenvalue problem using low-rank and QTT tensor approximation, J. Comput. Phys. 334 (2017), 221-239. MR Zbl
[5] J. Bezanson, A. Edelman, S. Karpinski, and V. B. Shah, Julia: a fresh approach to numerical computing, SIAM Rev. 59 (2017), no. 1, 65-98. MR Zbl
[6] J. Brabec, L. Lin, M. Shao, N. Govind, C. Yang, Y. Saad, and E. G. Ng, Efficient algorithms for estimating the absorption spectrum within linear response TDDFT, J. Chem. Theory Comput. 11 (2015), no. 11, 5197-5208.
[7] J. Deslippe, G. Samsonidze, D. A. Strubbe, M. Jain, M. L. Cohen, and S. G. Louie, BerkeleyGW: a massively parallel computer package for the calculation of the quasiparticle and optical properties of materials and nanostructures, Comput. Phys. Commun. 183 (2012), no. 6, 12691289.
[8] K. Dong, W. Hu, and L. Lin, Interpolative separable density fitting through centroidal Voronoi tessellation with applications to hybrid functional electronic structure calculations, J. Chem. Theory Comput. 14 (2018), no. 3, 1311-1320.
[9] G. H. Golub and C. F. Van Loan, Matrix computations, 4th ed., Johns Hopkins University, Baltimore, MD, 2013. MR Zbl
[10] A. Gulans, S. Kontur, C. Meisenbichler, D. Nabok, P. Pavone, S. Rigamonti, S. Sagmeister, U. Werner, and C. Drax1, exciting: a full-potential all-electron package implementing densityfunctional theory and many-body perturbation theory, J. Phys. Condens. Mat. 26 (2014), no. 36, art. id. 363202.
[11] P. H. Hahn, K. Seino, W. G. Schmidt, J. Furthmüller, and F. Bechstedt, Quasiparticle and excitonic effects in the optical spectra of diamond, SiC, Si, GaP, GaAs, InP, and AlN, Phys. Status Solidi B 242 (2005), no. 13, 2720-2728.
[12] W. Hu, L. Lin, and C. Yang, Interpolative separable density fitting decomposition for accelerating hybrid density functional calculations with applications to defects in silicon, J. Chem. Theory Comput. 13 (2017), no. 11, 5420-5431.
[13] W. Hu, M. Shao, A. Cepellotti, F. H. da Jornada, L. Lin, K. Thicke, C. Yang, and S. G. Louie, Accelerating optical absorption spectra and exciton energy computation via interpolative separable density fitting, ICCS 2018, II (Y. Shi, H. Fu, Y. Tian, V. V. Krzhizhanovskaya, M. H. Lees, J. Dongarra, and P. M. A. Sloot, eds.), Lecture Notes in Comput. Sci., no. 10861, Springer, 2018, pp. 604-617. MR
[14] R. B. Lehoucq, D. C. Sorensen, and C. Yang, ARPACK users' guide: solution of large-scale eigenvalue problems with implicitly restarted arnoldi methods, Software, Environments, and Tools, no. 6, Society for Industrial and Applied Mathematics, Philadelphia, PA, 1998. MR Zbl
[15] L. Lin, Z. Xu, and L. Ying, Adaptively compressed polarizability operator for accelerating large scale ab initio phonon calculations, Multiscale Model. Simul. 15 (2017), no. 1, 29-55. MR Zbl
[16] M. P. Ljungberg, P. Koval, F. Ferrari, D. Foerster, and D. Sánchez-Portal, Cubic-scaling iterative solution of the Bethe-Salpeter equation for finite systems, Phys. Rev. B 92 (2015), no. 7, art. id. 075422.
[17] J. Lu, C. D. Sogge, and S. Steinerberger, Approximating pointwise products of Laplacian eigenfunctions, J. Funct. Anal. 277 (2019), no. 9, 3271-3282. MR Zbl
[18] J. Lu and K. Thicke, Cubic scaling algorithms for RPA correlation using interpolative separable density fitting, J. Comput. Phys. 351 (2017), 187-202. MR Zbl
[19] J. Lu and L. Ying, Compression of the electron repulsion integral tensor in tensor hypercontraction format with cubic scaling cost, J. Comput. Phys. 302 (2015), 329-335. MR Zbl
[20] $\_\_\_\_$ , Fast algorithm for periodic density fitting for Bloch waves, Ann. Math. Sci. Appl. 1 (2016), no. 2, 321-339. MR Zbl
[21] M. Marsili, F. Mosconi, Edoardo De Angelis, and P. Umari, Large-scale GW-BSE calculations with $N^{3}$ scaling: excitonic effects in dye-sensitized solar cells, Phys. Rev. B 95 (2017), no. 7, art. id. 075415.
[22] H. J. Monkhorst and J. D. Pack, Special points for Brillouin-zone integrations, Phys. Rev. B 13 (1976), no. 12, 5188-5192. MR
[23] N. L. Nguyen, H. Ma, M. Govoni, F. Gygi, and G. Galli, Finite-field approach to solving the Bethe-Salpeter equation, Phys. Rev. Lett. 122 (2019), no. 23, art. id. 237402. MR
[24] G. Onida, L. Reining, R. W. Godby, R. Del Sole, and W. Andreoni, Ab initio calculations of the quasiparticle and absorption spectra of clusters: the sodium tetramer, Phys. Rev. Lett. 75 (1995), no. 5, 818-821.
[25] G. Onida, L. Reining, and A. Rubio, Electronic excitations: density-functional versus many-body Green's-function approaches, Rev. Mod. Phys. 74 (2002), no. 2, 601-659.
[26] H. R. Phillip and E. A. Taft, Kramers-Kronig analysis of reflectance data for diamond, Phys. Rev. 136 (1964), no. 5A, A1445-A1448.
[27] Y. Ping, D. Rocca, and G. Galli, Electronic excitations in light absorbers for photoelectrochemical energy conversion: first principles calculations based on many body perturbation theory, Chem. Soc. Rev. 42 (2013), 2437-2469.
[28] Y. Ping, D. Rocca, D. Lu, and G. Galli, Ab initio calculations of absorption spectra of semiconducting nanowires within many-body perturbation theory, Phys. Rev. B 85 (2012), no. 3, art. id. 035316.
[29] D. Y. Qiu, F. H. da Jornada, and S. G. Louie, Optical spectrum of $\mathrm{MoS}_{2}$ : many-body effects and diversity of exciton states, Phys. Rev. Lett. 111 (2013), no. 21, art. id. 216805.
[30] D. Rocca, D. Lu, and G. Galli, Ab initio calculations of optical absorption spectra: solution of the Bethe-Salpeter equation within density matrix perturbation theory, J. Chem. Phys. 133 (2010), no. 16, art. id. 164109.
[31] D. Rocca, Y. Ping, R. Gebauer, and G. Galli, Solution of the Bethe-Salpeter equation without empty electronic states: application to the absorption spectra of bulk systems, Phys. Rev. B 85 (2012), no. 4, art. id. 045116.
[32] M. Rohlfing and S. G. Louie, Electron-hole excitations and optical spectra from first principles, Phys. Rev. B 62 (2000), no. 8, 4927-4944.
[33] E. E. Salpeter and H. A. Bethe, A relativistic equation for bound-state problems, Phys. Rev. 84 (1951), 1232-1242. MR Zbl
[34] M. Shao, F. H. da Jornada, L. Lin, C. Yang, J. Deslippe, and S. G. Louie, A structure preserving Lanczos algorithm for computing the optical absorption spectrum, SIAM J. Matrix Anal. Appl. 39 (2018), no. 2, 683-711. MR Zbl
[35] M. Shao and C. Yang, BSEPACK user's guide, user manual, 2016. arXiv
[36] G. Strinati, Application of the Green's functions method to the study of the optical properties of semiconductors, Riv. Nuovo Cimento 11 (1988), no. 12, 1-86.
[37] C. Vorwerk, B. Aurich, C. Cocchi, and C. Drax1, Bethe-Salpeter equation for absorption and scattering spectroscopy: implementation in the exciting code, Electron. Struct. 1 (2019), no. 3, art. id. 037001.
[38] L. Yang, J. Deslippe, C.-H. Park, M. L. Cohen, and S. G. Louie, Excitonic effects on the optical response of graphene and bilayer graphene, Phys. Rev. Lett. 103 (2009), no. 18, art. id. 186802.

Received December 10, 2019.
Felix HenneKe: felix.henneke@fu-berlin.de
Institut für Mathematik, Freie Universität Berlin, Berlin, Germany
LIN LIN: linlin@math.berkeley.edu
Department of Mathematics, University of California, Berkeley, Berkeley, CA, United States
and
Computational Research Division, Lawrence Berkeley National Laboratory, Berkeley, CA, United States

CHRISTIAN VORWERK: vorwerk@physik.hu-berlin.de
Institut für Physik, IRIS Adlershof, Humboldt-Universität zu Berlin, Berlin, Germany
ClAUDIA DRAXL: claudia.draxl@physik.hu-berlin.de
Institut für Physik, IRIS Adlershof, Humboldt-Universität zu Berlin, Berlin, Germany
RUPERT KLEIN: rupert.klein@fu-berlin.de
Institut für Mathematik, Freie Universität Berlin, Berlin, Germany
CHAO YANG: cyang@lbl.gov
Computational Research Division, Lawrence Berkeley National Laboratory, Berkeley, CA, United States

\section*{Communications in Applied Mathematics and Computational Science}
msp.org/camcos

\section*{EDITORS}

\section*{Managing Editor}

John B. Bell
Lawrence Berkeley National Laboratory, USA
jbbell@lbl.gov

\section*{Board of Editors}

\begin{tabular}{|l|l|l|l|}
\hline Marsha Berger & New York University berger@cs.nyu.edu & Ahmed Ghoniem & Massachusetts Inst. of Technology, USA ghoniem@mit.edu \\
\hline Alexandre Chorin & University of California, Berkeley, USA chorin@math.berkeley.edu & Raz Kupferman & The Hebrew University, Israel raz@math.huji.ac.il \\
\hline Phil Colella & Lawrence Berkeley Nat. Lab., USA pcolella@lbl.gov & Randall J. LeVeque & University of Washington, USA rj1@amath.washington.edu \\
\hline Peter Constantin & University of Chicago, USA const@cs.uchicago.edu & Mitchell Luskin & University of Minnesota, USA luskin@umn.edu \\
\hline Maksymilian Dryja & Warsaw University, Poland maksymilian.dryja@acn.waw.pl & Yvon Maday & Université Pierre et Marie Curie, France maday@ann.jussieu.fr \\
\hline M. Gregory Forest & University of North Carolina, USA forest@amath.unc.edu & James Sethian & University of California, Berkeley, USA sethian@math.berkeley.edu \\
\hline Leslie Greengard & New York University, USA greengard@cims.nyu.edu & Juan Luis Vázquez & Universidad Autónoma de Madrid, Spain juanluis.vazquez@uam.es \\
\hline Rupert Klein & Freie Universität Berlin, Germany rupert.klein@pik-potsdam.de & Alfio Quarteroni & Politecnico di Milano, Italy alfio.quarteroni@polimi.it \\
\hline Nigel Goldenfeld & University of Illinois, USA nigel@uiuc.edu & Eitan Tadmor & University of Maryland, USA etadmor@cscamm.umd.edu \\
\hline & & Denis Talay & INRIA, France denis.talay@inria.fr \\
\hline
\end{tabular}

\section*{PRODUCTION}
production@msp.org
Silvio Levy, Scientific Editor

\begin{abstract}
See inside back cover or msp.org/camcos for submission instructions.
The subscription price for 2020 is US $\$ 110$ /year for the electronic version, and $\$ 165 /$ year ( $+\$ 15$, if shipping outside the US) for print and electronic. Subscriptions, requests for back issues from the last three years and changes of subscriber address should be sent to MSP.
Communications in Applied Mathematics and Computational Science (ISSN 2157-5452 electronic, 1559-3940 printed) at Mathematical Sciences Publishers, 798 Evans Hall \#3840, c/o University of California, Berkeley, CA 94720-3840, is published continuously online. Periodical rate postage paid at Berkeley, CA 94704, and additional mailing offices.
\end{abstract}

\section*{PUBLISHED BY}

\section*{mathematical sciences publishers}

\section*{nonprofit scientific publishing}
http://msp.org/
© 2020 Mathematical Sciences Publishers

\section*{Communications in Applied Mathematics and Computational Science}
vol. 15
no. I
2020

Investigation of finite-volume methods to capture shocks and turbulence spectra in compressible flows

Emmanuel Motheau and John Wakefield
A stochastic version of Stein variational gradient descent for efficient 37 sampling

Lei Li, Yingzhou Li, Jian-Guo Liu, Zibu Liu and Jianfeng Lu
A third-order multirate Runge-Kutta scheme for finite volume solution of 3D time-dependent Maxwell's equations

Marina Kotovshchikova, Dmitry K. Firsov and Shiu Hong Lui

Fast optical absorption spectra calculations for periodic solid state systems Felix Henneke, Lin Lin, Christian Vorwerk, Claudia Draxl, Rupert Klein and Chao Yang