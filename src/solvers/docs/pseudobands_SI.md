\title{
Supplementary Material for Mixed Stochastic-Deterministic Approach for Many-Body Perturbation Theory Calculations
}
Aaron R. Altman ${ }^{1}$, Sudipta Kundu ${ }^{1}$, and Felipe H. da Jornada ${ }^{* 1,2}$
${ }^{1}$ Department of Materials Science and Engineering, Stanford University, Stanford, California 94305, USA
${ }^{2}$ Stanford Institute for Materials and Energy Sciences, SLAC National Accelerator Laboratory, Menlo Park, California 94025, USA
Contents
1 Computational Details ..... S1
1.1 Benzene ..... S2
1.2 ZnO ..... S2
1.3 $\mathrm{MoS}_{2}$ ..... S3
1.4 $\quad \mathrm{Ag}_{54} \mathrm{Pd}$ ..... S3
1.5 Obtaining Unoccupied Mean-Field States ..... S4
1.6 Pseudocode and Python Scripts for Pseudobands ..... S5
2 Convergence Tests for Additional Systems ..... S7
2.1 ZnO Supercells ..... S7
2.2 $\mathrm{MoS}_{2}$ Bilayer Unit Cell ..... S8
2.2.1 Extrapolation of Quasiparticle Corrections for Moiré Systems ..... S9
2.3 $\mathrm{Ag}_{54} \mathrm{Pd}$ Nanoparticle ..... S9
3 Convergence Proofs for GW Quantities with Pseudobands ..... S12
3.1 Convergence of the Green's Function $G(\omega)$ ..... S12
3.2 Convergence of the Non-interacting Polarizability $\chi^{0}$ ..... S13
3.2.1 Partition of Subspaces ..... S18
3.2.2 Extension to Full-Frequency Calculations ..... S18
3.3 Convergence of the GW Self-Energy $\Sigma^{G W}$ ..... S20

\section*{1 Computational Details}

Subsections below include material-specific details such as convergence parameters. See Ref. [1] for the atomic structures. For all materials, density-functional theory (DFT) calculations were performed within the Quantum ESPRESSO package [2] with the Perdew-Burke-Ernzerhof (PBE) exchange-correlation functional [3] to obtain mean-field energies and wavefunctions. All calculations used the plane-wave basis and scalar-relativistic norm-conserving pseudopotentials [4,5]. GW calculations were performed in the BerkeleyGW code [6] in a single-shot approach. Reference GW

\footnotetext{
*E-mail: jornada@stanford.edu
}
calculations used for comparison with stochastic pseudobands GW calculations utilized enough unoccupied states in the sum-over-bands to be effectively fully converged with respect to the other cutoffs used, as we detail below for each system.

For details about strong and weak scaling of the BerkeleyGW code, we refer the reader to Ref [6].
In section S1.5 we describe our approach to obtaining the large numbers of unoccupied meanfield states required for GW calculations. In section S1.6 we provide pseudocode and reference to standalone Python scripts for our implementation of pseudobands. The scripts are agnostic to the input mean-field wavefunctions and the GW code which utilizes the output wavefunctions, except for file format.

\subsection*{1.1 Benzene}

Benzene atomic structure is taken from the GW100 benchmark [7]. The wavefunction plane-wave cutoff used was 50 Ry , a unit cell that encloses eight times the volume containing $95 \%$ of the meanfield charge density was used, and 30,000 mean-field states were generated. We chose a modest unit cell and energy cutoffs to allow us to perform fully converged deterministic calculations without relying on extrapolations of convergence parameters. The GW calculations employed a 12 Ry dielectric cutoff, the Hybertsen-Louie Generalized Plasmon-Pole (GPP) model for the frequency dependence of the dielectric function [8], and a box-truncation scheme of the Coulomb interaction to avoid spurious effects from periodic images [9]. The self-energy was evaluated for the 10 highest occupied and 9 lowest unoccupied molecular orbitals. For the reference GW calculation, all 30,000 Kohn-Sham states were used (i.e., all the Kohn-Sham states available given the simulation unit cell and wavefunction cutoff). Pseudobands compressed the same 30,000 states into varying numbers of stochastic bands. Root-mean-square (RMS) quasiparticle (QP) errors when using pseudobands for GW were computed over these 19 states.

The stochastic pseudobands parameters used were $N_{P}^{c}=50, N_{\xi}^{c}=2, N_{S}^{c}=\{1,5,50,250\} . N_{P}=50$ is enough to capture all relevant band reordering. We did not employ stochastic pseudobands to compress valence states since a single benzene molecule has a small number of occupied orbitals.

\subsection*{1.2 ZnO}

ZnO crystal structure is taken from experiment, obtained from the American Mineralogist Crystal Structure Database (_database_code_amcsd 0005203) [10]. A wavefunction cutoff of 200 Ry was used with a $4 \times 4 \times 4 \mathrm{k}$-grid, and 10,000 mean-field states were generated. For the GW calculation, a dielectric cutoff of 80 Ry was used with frequency dependence from the GPP model, and the selfenergy was evaluated at the $\Gamma$ point to compute the valence band maximum (VBM) and conduction band minimum (CBM). With these parameters we obtain a band gap of 2.69 eV - within 100 meV of a recently reported value [11].

Pseudobands parameters used were $N_{P}^{c}=10, N_{\xi}^{c}=2, N_{S}^{c}=\{1,5,50,250\}$. Valence pseudobands were not used.

Supercell structures for the scaling test were generated with the Python module ase [12]. Deterministic GW calculations were performed for the $2 \times 1 \times 1,2 \times 2 \times 1$, and $2 \times 2 \times 2$ supercells to verify stability of the QP gap with respect to supercell size - the deviation from the unit cell GW gap was less than 1 meV for different supercell sizes (variations in the GW gap are related to the numerical treatment of the $q \rightarrow 0$ limit of the screened Coulomb interaction). Thus, all errors from utilizing stochastic pseudobands were measured with respect to the unit-cell QP values.

We chose the number of protected bands to scale with the supercell size to account for the effects of band-folding and avoid an increase in the error in the QP energies. We picked $N_{P}^{v / c}$ being 10
times the number of ZnO unit cells in each supercell calculation. The other parameters were fixed at $N_{\xi}^{v}=4, N_{\xi}^{c}=2, N_{S}^{v}=5$, and $N_{S}^{c}=185 \pm 10$. There were slight variations of $N_{S}^{c}$ due to differences in how the bands fell into the allocated slices depending on the supercell size. All runs used the slice ratio $\mathcal{F}=0.02$.

\section*{1.3 $\mathrm{MoS}_{2}$}

The AA stacked bilayer $\mathrm{MoS}_{2}$ structure was obtained with relaxation in the LAMMPS code [13] with the Kolmogorov-Crespi interlayer potential [14] and Stillinger-Weber intralayer potential [15]. The $5.75^{\circ}$ twisted $\mathrm{MoS}_{2}$ bilayer was generated with a homemade moire-structure code and relaxed in LAMMPS with the same potentials.
For the AA bilayer, we used a wavefunction cutoff of 35 Ry , the unit cell extended $20 \AA$ in the nonperiodic direction, and a $6 \times 6 \times 1 \mathrm{k}$-grid. Convergence tests of all relevant direct and indirect QP valence-conduction gaps (e.g. $K-\Gamma, \Gamma-K, K-K, M-K$, etc.) with respect to the wavefunction cutoff were performed, and the 35 Ry cutoff was sufficient to converge these gaps to within 30 meV with of the QP gaps from a 70 Ry wavefunction cutoff calculation. To properly converge the dielectric function we employed the nonuniform neck subsampling (NNS) technique [16] for q-points near $\Gamma$. We included 4,055 mean-field states for the GW calculations [17], truncated the Coulomb interaction along the non-periodic direction, and employed a cutoff of 35 Ry for the dielectric matrix with frequency dependence from the GPP model. The self-energy was evaluated for states VBM-4 through CBM +4 with a 15 Ry cutoff for the screened Coulomb interaction, enough to converge relative QP energies at the K and $\Lambda_{\text {min }}$ points of the lowest-energy conduction bands to within about 100 meV .

Extensive convergence testing for pseudobands was not performed for this system. Instead, we chose the following reasonable parameters and obtained excellent results for the QP energies relative to the deterministic calculation: $N_{P}^{c}=10, N_{\xi}^{c}=3, N_{S}^{c}=151$ (the corresponding slice fraction $\mathcal{F}=0.02$ ). Valence pseudobands were not used due to the small number of occupied states.

For the $5.75^{\circ}$ twisted bilayer, we could only perform the calculation with stochastic pseudobands. We employed a $3 \times 3 \times 1 \mathrm{k}$-grid with a 35 Ry wavefunction cutoff at the mean-field level, and a $20 \AA$ cell in the non-periodic direction. For the dielectric function we used a 25 Ry cutoff, and the GPP model was used for the frequency dependence. NNS was not employed as the $3 \times 3 \times 1$ q-grid is fine enough to sample the sharp peak in the dielectric function due to the large cell size. The self-energy was computed with a 15 Ry cutoff for the screened Coulomb interaction for 27 valence and 13 conduction states, and linear interpolation in the moiré BZ gave the QP band structure. Pseudobands compressed all states up to the wavefunction cutoff ( 380,000 input KohnSham orbitals were used to create all stochastic pseudobands). The parameters used were $N_{P}^{v / c}= 1000, N_{\xi}^{v / c}=3, N_{S}^{v}=36, N_{S}^{c}=211$, corresponding to $\mathcal{F}^{v / c}=0.02$. This value of $N_{P}$ corresponds to the AA-stacked bilayer scaled by the supercell size ( $\sim 10 \times 10$ ).

\section*{1.4 $\quad \mathbf{A g}_{54} \mathbf{P d}$}

An $\mathrm{Ag}_{55}$ icosahedral nanoparticle structure was generated with the ase Python package [12]. One of the Ag atoms on an edge of the icosahedron was then replaced with a Pd atom, without relaxation, to make $\mathrm{Ag}_{54} \mathrm{Pd}$. A $25 \AA^{3}$ cubic unit cell was used, with a somewhat small wavefunction cutoff of 23 Ry to demonstrate the method. We generated 10,000 mean-field states. Rather than using the GPP model, the dielectric function was computed explicitly in frequency space from 2.5 eV to 4.5 eV with a 100 meV grid spacing and a 200 meV broadening. We employed the static subspace approximation to speed up the full-frequency calculation with a basis size of 2500 eigenstates [18]. The self-energy was not computed; convergence of pseudobands was measured with respect to the
macroscopic dielectric function $\epsilon^{M}(\omega)=1 / \epsilon_{00}^{-1}(\omega)$.
To make the stochastic pseudobands approach amenable to evaluating the dielectric matrix at arbitrary frequencies, we slightly modify the way energy subspaces are chosen (see section S3.2.2 below). In short, we include a low-energy region up to $\omega_{\text {max }}$ where the energy spanned by each subspace $\Delta E_{S}=\delta \omega$ is constant, after which exponential slices begin as usual. The following pseudobands parameters were used: $N_{P}^{v / c}=10, N_{\xi}^{v / c}=\{1,5\}, N_{S}^{v}=\{2,10,20,50\}, N_{S}^{c}=\{2,10,100,500\}$ (corresponding to $\mathcal{F}^{v}=\{0.51,0.11,0.11,0.0079\}$ and $\left.\mathcal{F}^{c}=\{0.67,0.14,0.014,0.0027\}\right)$. Here the $N_{S}$ parameters only describe the exponential portion of the slices. $\omega_{\text {max }}$ and $\delta \omega$ were determined by the energy range and spectral resolution of the desired calculation of the dielectric matrix; we set $\omega_{\max }=10.2 \mathrm{eV}$ and $\delta \omega=100 \mathrm{meV}$. In principle $\omega_{\max }$ can be made smaller, and convergence testing should be performed for real calculations.

\subsection*{1.5 Obtaining Unoccupied Mean-Field States}

In this subsection, we briefly clarify our approach to generating the large number of unoccupied states required for the GW calculations, which is independent of the pseudobands approach. We emphasize that while the approach described below is very efficient, it is not required to apply the pseudobands method, which is agnostic to the procedure used to generate the mean-field states.

As described in the main text, we perform a direct diagonalization of the mean-field Hamiltonian $H^{\mathrm{MF}}$ with a highly optimized linear algebra solver such as ELPA [19] to obtain all mean-field states up to the chosen plane-wave cutoff. When the number of bands is a significant fraction ( $>10 \%$ ) of the size of the Hamiltonian, iterative diagonalizers become inefficient and/or numerically unstable. Therefore, there are at least three clear advantages in diagonalizing the full Kohn-Sham Hamiltonian as opposed to using an iterative solver:
1. Well-converged wavefunctions: iterative diagonalizers in DFT codes are optimized for the lowenergy part of the energy spectrum. When solving for a large fraction of the eigenspace, they are inefficient and often struggle to obtain well-converged eigenvectors, especially when one requests thousands of Kohn-Sham states. Poorly converged states can lead to an unphysically large screening response, and is easily avoided with a direct diagonalizer.
2. Easier convergence: a direct matrix diagonalization algorithm yields the whole unoccupied manifold described up to the wavefunction cutoff, facilitating convergence testing. There is no concern about having to recompute unoccupied Kohn-Sham states if the previous amount was insufficient. Additionally, when coupled with the pseudobands approach, a direct diagonalization of the DFT Hamiltonian further eliminates the convergence parameters of how many bands to include: they are all included within our stochastic formalism.
3. Significant speedup: a major benefit of directly diagonalizing the Kohn-Sham Hamiltonian is that, in practice, it is significantly faster than using iterative diagonalizers when one is interested in more than about $5 \%$ of the spectrum. For reference, below we include previous benchmarks we performed on Edison at the National Energy Research Scientific Computing Center (NERSC), comparing a direct diagonalizer, Parabands, bundled with the BerkeleyGW package [6] and using the ELPA library, as well as reference values of the Quantum ESPRESSO [2] Davidson algorithm.

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table S1: Benchmarks for Direct vs. Iterative Diagonalization of $H^{\mathrm{MF}}$}
\begin{tabular}{|l|l|l|l|}
\hline System & $\mathrm{N}_{2}$ molecule & TTF molecule & $\mathrm{MoS}_{2}$ monolayer \\
\hline Size of Hamiltonian $H^{\mathrm{MF}}$ & 36 k & 137 k & 77 k \\
\hline Number of bands generated & 6.5k (18\% of $H^{\mathrm{MF}}$ ) & 27 k ( $20 \%$ of $H^{\mathrm{MF}}$ ) & 5.1k ( $7 \%$ of $H^{\mathrm{MF}}$ ) \\
\hline Wall time: Quantum ESPRESSO (Davidson) & 3.5 h (256 tasks × 4 threads) & >4h (never finished) (512 tasks × 1 thread) & 3.5h (64 tasks × 1 thread) \\
\hline Wall time: Parabands (ELPA 1-stage) & 4 mins (256 tasks × 1 thread) & 26 mins (512 tasks × 1 thread) & 11 mins (512 tasks $\times 4$ threads) \\
\hline
\end{tabular}
\end{table}

We also briefly describe the implementation of the exact diagonalization of $H^{\mathrm{MF}}$ within the Parabands code in BerkeleyGW: Parabands reads as input the Kohn-Sham potential and wavefunctions obtained from a self-consistent-field calculation of a DFT code. Then, it uses the non-local parts of the pseudopotentials in Kleinman-Bylander form to construct $H^{\mathrm{MF}}$ on-the-fly in the plane-wave basis. Finally, a direct diagonalizer such as ELPA is called, and the resulting wavefunctions and eigenenergies are written to file.

Although the exact diagonalization approach is very efficient for getting the large numbers of unoccupied states required to converge GW sums-over-bands, the pseudobands approach will provide the same benefits regardless of how these states are generated, as long as there are enough to otherwise converge the GW calculations.

Finally, we note that, for all systems studied, the diagonalization of the mean-field Hamiltonian took less time than the evaluation of the dielectric matrix when utilizing the pseudobands approach. Even for the largest ZnO supercell studied, the diagonalization of the Kohn-Sham Hamiltonian took only $38 \%$ of the computational resources necessary to evaluate the dielectric matrix, and for the $\mathrm{MoS}_{2}$ moiré system, the diagonalization took $88 \%$ of the resources of the dielectric calculation.

\subsection*{1.6 Pseudocode and Python Scripts for Pseudobands}

We present a pseudocode for our implementation of pseudobands, and provide a reference to a standalone Python implementation [1]. The Python scripts are agnostic to the mean-field code that generates the input mean-field states and to the GW code which uses the output. The only assumption is that the input wavefunctions are in BerkeleyGW format ${ }^{1}$. The pseudocode follows:

\footnotetext{
${ }^{1}$ http://manual.berkeleygw.org/3.0/wfn_h5_spec/
}
```
Algorithm 1 Stochastic Pseudobands
input : float en [ $\mathrm{n} \_\mathrm{k}, \mathrm{n} \_\mathrm{b}$ ] \# mean-field energies (kpoints, bands)
        complex wfn [n_k, n_b, n_basis] \# mean-field wavefunctions
        int np_v, np_c $\geq 0$ \# protected states (valence/conduction)
        int $\mathrm{n} \xi \_\mathrm{v}, \mathrm{n} \xi \_\mathrm{c}>0 \quad \#$ stochastic states per subspace
        float $2 \gtrsim \mathcal{F} \_\mathrm{v}, \mathcal{F} \_\mathrm{c}>0 \quad$ \# constant energy fraction
output: float en_out [n_k, n_spb ] \# output energies (kpoints, pseudobands)
        complex wfn_out [n_k, n_spb, n_basis] \# output wavefunctions
begin
    set Fermi level to zero
    partition en into valence and conduction states en_v, en_c
    \# construct slices
    slices_c $=$ construct_slices $\left(\right.$ en_c, np_c, $\left.\mathcal{F} \_\mathrm{c}\right) \quad$ \# See Alg. 2
    slices_v = construct_slices (-en_v, np_v, $\mathcal{F} \_$v)
    \# initialize outputs
    initialize output wavefunction wfn_out and energies en_out
    copy protected states and energies to wfn_out and en_out
    \# construct pseudobands
    construct_pseudobands (wfn_out, en_out, slices_c, en_c, wfn) \# See Alg. 3
    construct_pseudobands(wfn_out, en_out, slices_v, en_v, wfn)
    return wfn_out, en_out
end
```

```
Algorithm 2 Construction of Slices
input : en, np, $\mathcal{F}$
output: slices
begin
    initialize array slices and first $=$ en [ $\mathrm{np}+1$ ]
    while first $<\min ($ en $[:,-1])$ do
        last $=$ first $*(1+\mathcal{F})$
        first_idx = find_band_idx (first) \# find index of band with this energy
        last_idx = find_band_idx (last)
        append (slices, [first_idx, last_idx])
        first $=$ en [last_idx + 1]
    end
    return slices
end
```

```
Algorithm 3 Construction of Pseudobands
input : slices, en, wfn
output: wfn_out, en_out
begin
    for slice $\in$ slices do
        set n_b_slice = slice [1] - slice [0]
        compute index of next pseudoband shift
        for $i k \in$ range $\left(n \_k\right)$ do
            for $\mathrm{i} \xi \in \operatorname{range}\left(\mathrm{n} \xi \_\mathrm{c}\right)$ do
                generate array of random phases phases [n_b_slice]
                phases /= $\operatorname{sqrt}\left(\mathrm{n} \xi \_\mathrm{c}\right)$ \# normalization
                \# matvec to obtain pseudobands
                wfn_out [ik, shift, :] = matvec(wfn [ik, slice, :], phases)
                \# output energy is mean energy of slice
                en_out [ik, shift] = mean(en [ik, slice])
            end
        end
    end
    return wfn_out, en_out
end
```


\section*{2 Convergence Tests for Additional Systems}

\subsection*{2.1 ZnO Supercells}

Here we show the band gap error for the calculations of bulk wurtzite ZnO supercells discussed in the main text. In Figure $\mathrm{S} 1(\mathrm{a})$ below, we reproduce the scaling curve from the main text for ZnO supercells, and in Figure S1(b) we show the band gap error from each supercell calculation resulting from the use of stochastic pseudobands, in reference to the highly converged deterministic unit cell calculation involving 10,000 Kohn-Sham states. As can be seen, except for a serendipitously low error for the $2 \times 2 \times 1$ supercell, the error is maintained at a constant below 50 meV , without improving the pseudobands parameters $N_{\xi}$ and $\mathcal{F}$.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f0c30553-77d0-403a-a265-b56065452e10-08.jpg?height=917&width=912&top_left_y=326&top_left_x=571}
\captionsetup{labelformat=empty}
\caption{Figure S1: (a) Scaling curve for the dielectric computation per $\mathrm{q} / \mathrm{k}$-point for ZnO supercells showing quasi-quadratic behavior up to 256 atoms. (b) QP band gap errors for each calculation with pseudobands; errors are maintained at $<50 \mathrm{meV}$ for the same convergence parameters $N_{\xi}, \mathcal{F}$.}
\end{figure}

\section*{2.2 $\mathrm{MoS}_{2}$ Bilayer Unit Cell}

To test convergence in 2D materials, we calculate the band structure of an AA-stacked MoS ${ }_{2}$ bilayer. This also serves as preparation for the calculation on a moiré superlattice of bilayer $\mathrm{MoS}_{2}$, as described above. Thorough parameter testing was not performed as with the systems in the main text; instead, we chose reasonable parameters based on those systems and performed a single pseudobands calculation, along with the full deterministic calculation (see section S1.3). The results are shown in Figure S2 - with a 20 meV RMS error in the QP energy levels, the two calculations are nearly indistinguishable by eye. Moreover, the pseudobands GW calculation was about 10 times faster, even for this small system.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f0c30553-77d0-403a-a265-b56065452e10-09.jpg?height=695&width=989&top_left_y=313&top_left_x=567}
\captionsetup{labelformat=empty}
\caption{Figure S2: QP band structure of the AA-stacked MoS ${ }_{2}$ bilayer. The agreement between the calculation employing stochastic pseudobands and the deterministic approach is nearly perfect, with an RMS error of 20 meV .}
\end{figure}

\subsection*{2.2.1 Extrapolation of Quasiparticle Corrections for Moiré Systems}

We briefly address a common approach taken to evaluate quasiparticle corrections of moiré bilayers [20] before the explicit GW calculations allowed by pseudobands approach. For moderate twist angles (so as to not form strong structural inhomogeneity), one expects that the polarizability will not change significantly from that of the untwisted bilayer. For these systems, one may approximate the GW QP energies $E_{\text {moiré }}^{\mathrm{GW}}$ evaluated on the moiré cell as $E_{\text {moiré }}^{\mathrm{GW}} \approx E_{\text {moiré }}^{\mathrm{DFT}}+\delta E_{\text {prim }}^{\mathrm{GW}-\mathrm{DFT}}$, where $E_{\text {moiré }}^{\mathrm{DFT}}$ are the DFT Kohn-Sham eigenvalues obtained on the moiré cell, and $E_{\text {prim }}^{\mathrm{GW}-\mathrm{DFT}}$ is the GW self-energy correction obtained on the high-symmetry primitive unit cell, relative to the DFT calculation on the equivalent structure.

For the GW calculation of the $5.75^{\circ}$ twisted $\mathrm{MoS}_{2}$ bilayer presented in the main text, we find the structurally-driven moiré effects and electron correlation-driven quasiparticle effects are additive to within 100 meV ; i.e. we find that the difference between the additive approach and explicit calculation of $E_{\text {moiré }}^{\mathrm{GW}}$ varies from 10 meV to 94 meV for selected quasiparticle energies between relevant high-symmetry states close to the Fermi energy. In particular, the splitting $E_{\text {VBM }}(\Gamma)- E_{\mathrm{VBM}-1}(\Gamma)$ is accurately captured by the additive approach, with an error of only 10 meV . However, for the splittings $E_{\mathrm{VBM}}(\Gamma)-E_{\mathrm{VBM}}(K)$ and $E_{\mathrm{CBM}}(K)-E_{\mathrm{CBM}}(\Lambda)$, the error is 94 meV and 42 meV , respectively (see Fig. 3 in the main text), showing that the additive approach cannot be applied everywhere in the moiré cell with uniform accuracy. Furthermore, one expects that, for other large-scale systems wherein the dielectric function is not uniform within the cell, such as in moiré systems displaying varying degree of hybridization between valence and conduction bands (e.g., bilayer graphene) or when there is charge doping localized within the moiré cell, such an additive procedure may show further inaccuracies.

\section*{2.3 $\mathbf{A g}_{54} \mathbf{P d}$ Nanoparticle}

Here we test the convergence of the imaginary part of the macroscopic dielectric function of an $\mathrm{Ag}_{54} \mathrm{Pd}$ nanoparticle in the full-frequency scheme (see section S 3.2 .2 ) with respect to $N_{S}$ and $N_{\xi}$.

In Figure S3a we show several calculations of the macroscopic dielectric function with varying pseudobands parameters listed in the legend, along with the reference calculation. Figure S3b shows the same data as Figure S3a but as a percent error, demonstrating convergence over the whole frequency grid. In Table S2 we summarize the results from Figure S3, showing the pseudobands parameters, corresponding speedups, and root mean square percentage error (RMSPE). RMSPE is defined as
$$
\begin{equation*}
\mathrm{RMSPE}=\sqrt{\frac{1}{N_{\omega}} \sum_{\omega}\left(\frac{\epsilon_{2}^{M}(\omega)[\mathrm{SPB}]-\epsilon_{2}^{M}(\omega)}{\epsilon_{2}^{M}(\omega)}\right)^{2}} \times 100 \%, \tag{S1}
\end{equation*}
$$
where for this calculation the sum over $\omega$ ranges from $\omega=2.5 \mathrm{eV}$ to $\omega=4.5 \mathrm{eV}$, with $N_{\omega}=21$. Untabulated parameters are the same for all runs, and are written in section S1.4. As can be seen, smooth convergence is achieved as the pseudobands parameters become better, demonstrating the ability to handle full-frequency calculations. Further convergence can be achieved by reducing $\delta \omega$, which was 100 meV for these calculations. We emphasize that while the $N_{\xi}=1$ calculations do converge, this is primarily due to the large values of $N_{S}$ used and the averaging of the matrix elements discussed in section S3.2 below; the $N_{\xi}=5$ calculations which actually make use of stochastic averages converge much faster. We also note that in general, the full-frequency dielectric calculation is not as rapidly converging as the dielectric calculation with the GPP model, or the self-energy calculation, so a higher value of $N_{\xi}$ is recommended. See section S3.2.2 for details.

\begin{table}
\begin{tabular}{|l|l|l|l|l|}
\hline $N_{\xi}^{v, c}$ & $N_{S}^{v}$ & $N_{S}^{c}$ & Speedup Factor & RMSPE (\%) \\
\hline 1 & 2 & 2 & 84 & 22.2 \\
\hline 1 & 10 & 10 & 86 & 17.2 \\
\hline 1 & 20 & 100 & 80 & 14.6 \\
\hline 1 & 50 & 500 & 80 & 13.5 \\
\hline 5 & 2 & 2 & 65 & 11.2 \\
\hline 5 & 10 & 10 & 65 & 1.2 \\
\hline 5 & 20 & 100 & 52 & 3.0 \\
\hline 5 & 50 & 500 & 26 & 2.9 \\
\hline
\end{tabular}
\captionsetup{labelformat=empty}
\caption{Table S2: (rightmost column) Root mean square percent error (RMSPE) of the frequency-dependent imaginary part of the macroscopic dielectric function $\epsilon_{2}^{M}(\omega)$ for an $\mathrm{Ag}_{54} \mathrm{Pd}$ nanoparticle, showing convergence over $N_{S}$ and $N_{\xi}$. Other columns show pseudobands parameters used for each calculation and the corresponding speedup factor observed, excluding I/O. Error and speedups are measured against a highly converged deterministic calculation. Small variations are expected in the error and speedups due to different random coefficients $\alpha$ and different node configurations, respectively. Corresponding $\mathcal{F}$ values to the $N_{S}$ in the table are $\mathcal{F}^{v}=\{0.51,0.11,0.11,0.0079\}$ and $\mathcal{F}^{c}=\{0.67,0.14,0.014,0.0027\}$.}
\end{table}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f0c30553-77d0-403a-a265-b56065452e10-11.jpg?height=1653&width=1208&top_left_y=320&top_left_x=466}
\captionsetup{labelformat=empty}
\caption{Figure S3: (a) Pseudobands convergence of the imaginary part of the macroscopic dielectric function for an $\mathrm{Ag}_{54} \mathrm{Pd}$ nanoparticle. All calculations used $N_{P}^{v / c}=10$ and other parameters are listed in the legend. Speedups of 10-100 times are observed relative to the deterministic calculation. (b) The same data as (a) but as a percent difference. $\epsilon_{2}^{M}[$ Full $]$ denotes the deterministic calculation.}
\end{figure}

\section*{3 Convergence Proofs for GW Quantities with Pseudobands}

In this section, we rigorously prove the intuitive convergence of our method described in the main text. We do so by showing explicitly that the error introduced to the electronic Green's function $G(\omega)$ and the static non-interacting polarizability $\chi_{\mathbf{G}, \mathbf{G}^{\prime}}^{0}(\mathbf{q}, \omega \approx 0)$ goes to 0 upon the usage of stochastic pseudobands. Additionally, we qualitatively discuss the convergence of the self-energy $\Sigma^{G W}$ when we utilize stochastic pseudobands.

We first recall the equations defining pseudobands from the main text:
$$
\begin{align*}
G(\omega) & \equiv \sum_{n, \mathbf{k}} \frac{\left|\phi_{n \mathbf{k}}\right\rangle\left\langle\phi_{n \mathbf{k}}\right|}{\omega-E_{n \mathbf{k}} \mp i \eta} \equiv \sum_{\mathbf{k}} G_{\mathbf{k}}(\omega)  \tag{S2}\\
G_{\mathbf{k}}(\omega) & \approx G_{\mathbf{k}}^{P}(\omega)+\sum_{S}^{N_{S}} G_{\mathbf{k}}^{S}(\omega)  \tag{S3}\\
G_{\mathbf{k}}^{P}(\omega) & =\sum_{n \in P}^{N_{P}} \frac{\left|\phi_{n \mathbf{k}}\right\rangle\left\langle\phi_{n \mathbf{k}}\right|}{\omega-E_{n \mathbf{k}} \mp i \eta}  \tag{S4}\\
G_{\mathbf{k}}^{S}(\omega) & =\frac{1}{\omega-\bar{E}_{S \mathbf{k}} \mp i \eta} \sum_{i=1}^{N_{\xi}}\left|\xi_{i, \mathbf{k}}^{S}\right\rangle\left\langle\xi_{i, \mathbf{k}}^{S}\right|  \tag{S5}\\
\left|\xi_{i, \mathbf{k}}^{S}\right\rangle & =\frac{1}{\sqrt{N_{\xi}}} \sum_{n \in S} \alpha_{i, n \mathbf{k}}^{S}\left|\phi_{n \mathbf{k}}\right\rangle \tag{S6}
\end{align*}
$$

Here $\left|\phi_{n \mathbf{k}}\right\rangle$ are Kohn-Sham states, $E_{n \mathbf{k}}$ are the corresponding mean-field eigenenergies, $\eta=0^{+}, P$ is the protected subspace, $S$ is a stochastic subspace of which there are $N_{S}, \bar{E}_{S \mathbf{k}}$ is the average energy of the Kohn-Sham states in subspace $S,\left|\xi_{i, \mathbf{k}}^{S}\right\rangle$ is a stochastic vector in subspace $S$ of which we take $N_{\xi}$ to resolve the projection onto subspace $S$, and $\alpha_{i, n \mathbf{k}}^{S}$ are uniformly distributed random phases with which we construct the pseudobands $\left|\xi_{i, \mathbf{k}}^{S}\right\rangle$. The stochastic subspaces $S$ can run over unoccupied and occupied states, and we treat the general case here. We do not make assumptions about the distribution of subspaces $S$ in terms of how many states each subspace holds at this point. Rather, we will derive a partition that allows rapid convergence of the quantities of interest (see section 3.2.1).
In the following subsections, we show convergence of $G$ and $\chi^{0}(\omega \approx 0)$, discuss convergence of $\Sigma^{G W}$, derive a heuristic for partitioning subspaces, and extend the approach to finite $\omega \neq 0$.

\subsection*{3.1 Convergence of the Green's Function $G(\omega)$}

Below, we denote the Green's function evaluated with stochastic pseudobands (SPB) as $G(\omega)$ [SPB]. The convergence of $G(\omega)$ [SPB] is not smooth in the sense that, if we define the error of stochastic pseudobands for $G$ as
$$
\begin{equation*}
\operatorname{Err}[G(\omega)[\mathrm{SPB}]] \equiv G(\omega)[\mathrm{SPB}]-G(\omega) \tag{S7}
\end{equation*}
$$
and take the limit $\eta \rightarrow 0$, we always have poles in $\operatorname{Err}[G(\omega)[\mathrm{SPB}]]$ at every energy $E_{n \mathbf{k}}$ of the original mean-field states. This is because we approximate a number of densely-packed poles of $G$ with a single pole of their combined weight in a given subspace $S$, we always have poles in our error coming from the true $G$.

However, if we assume $\omega$ is not close to a pole of either $G(\omega)[\mathrm{SPB}]$ or $G(\omega)$, then we can obtain expressions for the expectation and variance of $\operatorname{Err}[G(\omega)[\mathrm{SPB}]]$ and show that both go to 0 .

First, for a given subspace $S$, we define $G[S]$ as the restriction of $G[\mathrm{SPB}]$ to $S$, and find
$$
\begin{equation*}
\mathbb{E}\left[\operatorname{Err}\left[G_{\mathbf{k}}(\omega)[S]\right]\right]=\sum_{n \in S}\left|\phi_{n \mathbf{k}}\right\rangle\left\langle\phi_{n \mathbf{k}}\right|\left[\frac{1}{\omega-\bar{E}_{S, \mathbf{k}} \mp i \eta}-\frac{1}{\omega-E_{n \mathbf{k}} \mp i \eta}\right] \tag{S8}
\end{equation*}
$$
since the stochastic off-diagonals have an expectation of 0 . Thus, the main effect of the constant energy denominator approximation is to slightly change the expected value of the Green's function coming from subspace $S$. Since $\bar{E}_{S, \mathbf{k}}-E_{n, \mathbf{k}} \lesssim \Delta E_{S}$, the energy range spanned by $S$, this error is roughly proportional to $\Delta E_{S} /\left(\omega-\bar{E}_{S, \mathbf{k}}\right)^{2}$, and decreases as the number of slices increases, $N_{S} \rightarrow \infty$.
Next, we move on to the variance of the error of $G[S]$, to which only the off-diagonal terms in the Kohn-Sham basis contribute. We obtain, for a given matrix element $n \neq n^{\prime} \in S$ for a given subspace $S$,
$$
\begin{align*}
\operatorname{Var}\left[\operatorname{Err}\left[G_{n, n^{\prime}, \mathbf{k}}(\omega)[S]\right]\right] & =\frac{1}{\left(\omega-\bar{E}_{S, \mathbf{k}}\right)^{2}} \mathbb{E}\left[\left|\frac{1}{N_{\xi}} \sum_{i=1}^{N_{\xi}} \alpha_{n, i}^{S}\left(\alpha_{n^{\prime}, i}^{S}\right)^{*}\right|^{2}\right]  \tag{S9}\\
& =\frac{1}{\left(\omega-\bar{E}_{S, \mathbf{k}}\right)^{2}} \cdot \frac{1}{N_{\xi}} \tag{S10}
\end{align*}
$$
which approaches 0 with increasing stochastic pseudobands, $N_{\xi} \rightarrow \infty$. Thus, with both the expectation and variance of the error converging to 0 as $N_{S} \rightarrow \infty$ and $N_{\xi} \rightarrow \infty$, we have convergence of our proposed approach when evaluating the electronic Green's function away from a pole in any subspace.

In practice however, we take $\eta>0$ to be a small finite number on the order of 100 meV which broadens and smooths out the poles by shifting them away from the real axis, so convergence can be achieved at any $\omega$. This is the standard practice, for instance, when evaluating the frequencydependent dielectric function. With this finite broadening, the convergence of $G(\omega)$ is much smoother than in the formal $\eta \rightarrow 0$ limit, and the usage of stochastic pseudobands is well behaved at any frequency.

Additionally, we see empirically that as few as $N_{\xi}=2$ or 3 pseudobands are typically enough to get $<100 \mathrm{meV}$ error in the QP energies, which is a surprisingly low parameter given the $1 / N_{\xi}$ convergence of the Green's function itself. As discussed in the following section, the polarizability $\chi^{0}(\omega \approx 0)$ [SPB] tends to converge much faster than $G[\mathrm{SPB}]$, partially due to the rapidly oscillating nature of the matrix elements involving Kohn-Sham states used in the evaluation of the polarizability.

\subsection*{3.2 Convergence of the Non-interacting Polarizability $\chi^{0}$}

Here we compute, in detail, the error of $\chi^{0}(\omega \approx 0)[\mathrm{SPB}]$, the static non-interacting polarizability evaluated with stochastic pseudobands. While the following is done in a plane-wave basis, a similar analysis should hold in other bases. We start from the Adler-Wiser formula for the non-interacting polarizability matrix [8],
$$
\begin{equation*}
\chi_{\mathbf{G}, \mathbf{G}^{\prime}}^{0}(\mathbf{q}, \omega=0)=\sum_{v c \mathbf{k}} \frac{\left\langle\phi_{c \mathbf{k}}\right| e^{-i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}\left|\phi_{v \mathbf{k}+\mathbf{q}}\right\rangle\left\langle\phi_{v \mathbf{k}+\mathbf{q}}\right| e^{i\left(\mathbf{q}+\mathbf{G}^{\prime}\right) \cdot \mathbf{r}}\left|\phi_{c \mathbf{k}}\right\rangle}{E_{v \mathbf{k}+\mathbf{q}}-E_{c \mathbf{k}}} \equiv \sum_{\mathbf{k}} \chi_{\mathbf{G}, \mathbf{G}^{\prime}, \mathbf{k}}^{0}(\mathbf{q}, 0) \tag{S11}
\end{equation*}
$$
where we focus for now on $\omega=0$ as used in the GPP, where $\mathbf{G}, \mathbf{G}^{\prime}$ are reciprocal lattice vectors, $v$ and $c$ denote valence and conduction bands, respectively, and where we have again partitioned over $\mathbf{k}$-points. For convenience, we define $M_{v c \mathbf{k}}\left(\mathbf{q}, \mathbf{G}^{\prime}\right)=\left\langle\phi_{v \mathbf{k}+\mathbf{q}}\right| e^{i\left(\mathbf{q}+\mathbf{G}^{\prime}\right) \cdot \mathbf{r}}\left|\phi_{c \mathbf{k}}\right\rangle$.

With the usage of stochastic pseudobands, we can partition Eq. (S11) into four components depending on whether $v$ or $c$ is deterministic or stochastic,
$$
\begin{align*}
\chi_{\mathbf{G}, \mathbf{G}^{\prime}, \mathbf{k}}^{0}(\mathbf{q}, 0)[\mathrm{SPB}] & =\chi_{\mathbf{G}, \mathbf{G}^{\prime}}^{0}(\mathbf{q}, 0)\left[v \leq N_{P}^{v} ; c \leq N_{P}^{c}\right] \\
& +\chi_{\mathbf{G}, \mathbf{G}^{\prime}}^{0}(\mathbf{q}, 0)\left[v \leq N_{P}^{v} ; c \in S^{c}\right]+\chi_{\mathbf{G}, \mathbf{G}^{\prime}}^{0}(\mathbf{q}, 0)\left[v \in S^{v} ; c \leq N_{P}^{c}\right]  \tag{S12}\\
& +\chi_{\mathbf{G}, \mathbf{G}^{\prime}}^{0}(\mathbf{q}, 0)\left[v \in S^{v} ; c \in S^{c}\right]
\end{align*}
$$
where we take the indexing convention that we count up from the Fermi level for both valence and conduction states. Then, we can define the error of the polarizability with pseudobands as
$$
\begin{equation*}
\operatorname{Err}\left[\chi_{\mathbf{G}, \mathbf{G}^{\prime}, \mathbf{k}}^{0}(\mathbf{q}, 0)[\mathrm{SPB}]\right] \equiv \chi_{\mathbf{G}, \mathbf{G}^{\prime}, \mathbf{k}}^{0}(\mathbf{q}, 0)[\mathrm{SPB}]-\chi_{\mathbf{G}, \mathbf{G}^{\prime}, \mathbf{k}}^{0}(\mathbf{q}, 0) \tag{S13}
\end{equation*}
$$

As for $G$, to prove convergence we compute the expectation and variance of the error Eq. (S13) and show both tend to 0 as $N_{S}, N_{\xi} \rightarrow \infty$.

Clearly, the first term in Eq. (S12) contributes no error. So, we focus on the last three terms, starting with $\chi_{\mathbf{G}, \mathbf{G}^{\prime}}^{0}(\mathbf{q}, 0)\left[v \leq N_{P}^{v} ; c \in S^{c}\right]$. Explicitly, we have for the matrix elements with pseudobands:
$$
\begin{equation*}
\left\langle\phi_{v \mathbf{k}+\mathbf{q}}\right| e^{i\left(\mathbf{q}+\mathbf{G}^{\prime}\right) \cdot \mathbf{r}}\left|\xi_{i \mathbf{k}}^{S^{c}}\right\rangle=\frac{1}{\sqrt{N_{\xi}}} \sum_{n \in S^{c}} \alpha_{i, n \mathbf{k}}^{S^{c}}\left\langle\phi_{v \mathbf{k}+\mathbf{q}}\right| e^{i\left(\mathbf{q}+\mathbf{G}^{\prime}\right) \cdot \mathbf{r}}\left|\phi_{n \mathbf{k}}\right\rangle \tag{S14}
\end{equation*}
$$

Restricting to a single valence band $v$ and k -point $\mathbf{k}$, we have for the error:
$$
\begin{align*}
\operatorname{Err}\left[\chi_{\mathbf{G}, \mathbf{G}^{\prime}, \mathbf{k}}^{0}(\mathbf{q}, 0)\left[v ; c \in S^{c}\right]\right]= & \frac{1}{N_{\xi}} \sum_{i=1}^{N_{\xi}} \sum_{n, n^{\prime} \in S^{c}}\left(\alpha_{i, n \mathbf{k}}^{S^{c}}\right)^{*} \alpha_{i, n^{\prime} \mathbf{k}}^{S^{c}}\left\langle\phi_{n \mathbf{k}}\right| e^{-i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}\left|\phi_{v \mathbf{k}+\mathbf{q}}\right\rangle \\
& \times\left\langle\phi_{v \mathbf{k}+\mathbf{q}}\right| e^{i\left(\mathbf{q}+\mathbf{G}^{\prime}\right) \cdot \mathbf{r}}\left|\phi_{n^{\prime} \mathbf{k}}\right\rangle \times \frac{1}{E_{v \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}}  \tag{S15}\\
- & \sum_{n \in S^{c}}\left\langle\phi_{n \mathbf{k}}\right| e^{-i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}\left|\phi_{v \mathbf{k}+\mathbf{q}}\right\rangle \\
& \times\left\langle\phi_{v \mathbf{k}+\mathbf{q}}\right| e^{i\left(\mathbf{q}+\mathbf{G}^{\prime}\right) \cdot \mathbf{r}}\left|\phi_{n \mathbf{k}}\right\rangle \times \frac{1}{E_{v \mathbf{k}+\mathbf{q}}-E_{n \mathbf{k}}}
\end{align*}
$$

We can partition this into diagonal and off-diagonal terms so that we can treat the mean-energy error and stochastic error independently:
$$
\begin{equation*}
\operatorname{Err}\left[\chi_{\mathbf{G}, \mathbf{G}^{\prime}, \mathbf{k}}^{0}(\mathbf{q}, 0)\left[v ; c \in S^{c}\right]\right]=\left[n=n^{\prime}\right]+\left[n \neq n^{\prime}\right] \tag{S16}
\end{equation*}
$$

The first term of Eq. (S16) has no stochastic coefficients, and corresponds only to the mean-energy error:
$$
\begin{gather*}
{\left[n=n^{\prime}\right]=\sum_{n \in S^{c}}\left\langle\phi_{n \mathbf{k}}\right| e^{-i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}\left|\phi_{v \mathbf{k}+\mathbf{q}}\right\rangle\left\langle\phi_{v \mathbf{k}+\mathbf{q}}\right| e^{i\left(\mathbf{q}+\mathbf{G}^{\prime}\right) \cdot \mathbf{r}}\left|\phi_{n \mathbf{k}}\right\rangle} \\
\times\left[\frac{1}{E_{v \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}}-\frac{1}{E_{v \mathbf{k}+\mathbf{q}}-E_{n \mathbf{k}}}\right] . \tag{S17}
\end{gather*}
$$

We define $\delta E_{n, \mathbf{k}} \equiv E_{n \mathbf{k}}-\bar{E}_{S^{c} \mathbf{k}}$ to write:
$$
\begin{align*}
{\left[n=n^{\prime}\right] } & =\sum_{n \in S^{c}} M_{v n \mathbf{k}}(\mathbf{q}, \mathbf{G})^{*} M_{v n \mathbf{k}}\left(\mathbf{q}, \mathbf{G}^{\prime}\right) \times\left[\frac{1}{E_{v \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}}-\frac{1}{E_{v \mathbf{k}+\mathbf{q}}-\delta E_{n, \mathbf{k}}-\bar{E}_{S^{c} \mathbf{k}}}\right]  \tag{S18}\\
& =\sum_{n \in S^{c}} M_{v n \mathbf{k}}(\mathbf{q}, \mathbf{G})^{*} M_{v n \mathbf{k}}\left(\mathbf{q}, \mathbf{G}^{\prime}\right) \\
& \quad \times\left[\frac{1}{E_{v \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}}-\frac{1}{E_{v \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}} \cdot \frac{1}{1-\frac{\delta E_{n, \mathbf{k}}}{E_{v \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}}}\right]  \tag{S19}\\
& \approx \frac{-1}{\left(E_{v \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}\right)^{2}} \sum_{n \in S^{c}} M_{v n \mathbf{k}}(\mathbf{q}, \mathbf{G})^{*} M_{v n \mathbf{k}}\left(\mathbf{q}, \mathbf{G}^{\prime}\right) \times \delta E_{n, \mathbf{k}} \tag{S20}
\end{align*}
$$
where in the last line we have taken a first-order expansion around $\delta E_{n, \mathbf{k}}=0$. This expansion is always valid as the energy denominator $\left|E_{v \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}\right|>0$. This $\left[n=n^{\prime}\right]$ term contributes only to the expectation of the error, and vanishes as $N_{S} \rightarrow \infty$.
Now we turn to the off-diagonal contribution $\left[n \neq n^{\prime}\right]$ of Eq. (S16):
$$
\begin{align*}
{\left[n \neq n^{\prime}\right]=} & \frac{1}{N_{\xi}} \sum_{i=1}^{N_{\xi}} \sum_{n \neq n^{\prime} \in S^{c}}\left(\alpha_{i, n \mathbf{k}}^{S^{c}}\right)^{*} \alpha_{i, n^{\prime} \mathbf{k}}^{S^{c}}\left\langle\phi_{n \mathbf{k}}\right| e^{-i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}\left|\phi_{v \mathbf{k}+\mathbf{q}}\right\rangle  \tag{S21}\\
& \times\left\langle\phi_{v \mathbf{k}+\mathbf{q}}\right| e^{i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}\left|\phi_{n^{\prime} \mathbf{k}}\right\rangle \times \frac{1}{E_{v \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}}
\end{align*}
$$

It is easy to show that the distribution of the random variable $\left(\alpha_{i, n \mathbf{k}}^{S^{c}}\right)^{*} \alpha_{i, n^{\prime} \mathbf{k}}^{S^{c}}$ is uniform, the same as its factors. Thus, the expectation is simply 0 :
$$
\begin{equation*}
\mathbb{E}\left[n \neq n^{\prime}\right]=0 \tag{S22}
\end{equation*}
$$

By analogy, we obtain the following expressions for $\chi_{\mathbf{G}, \mathbf{G}^{\prime}}^{0}(\mathbf{q}, 0)\left[v \in S^{v} ; c\right]$ :
$$
\begin{equation*}
\left[n=n^{\prime}\right] \approx \frac{1}{\left(\bar{E}_{S^{v} \mathbf{k}+\mathbf{q}}-E_{c \mathbf{k}}\right)^{2}} \sum_{n \in S^{v}} M_{n c \mathbf{k}}(\mathbf{q}, \mathbf{G})^{*} M_{n c \mathbf{k}}\left(\mathbf{q}, \mathbf{G}^{\prime}\right) \times \delta E_{n, \mathbf{k}+\mathbf{q}} \tag{S23}
\end{equation*}
$$
and
$$
\begin{equation*}
\mathbb{E}\left[n \neq n^{\prime}\right]=0 \tag{S24}
\end{equation*}
$$

For the fully stochastic term $\chi_{\mathbf{G}, \mathbf{G}^{\prime}}^{0}(\mathbf{q}, 0)\left[v \in S^{v} ; c \in S^{c}\right]$ we have
$$
\begin{align*}
\operatorname{Err}\left[\chi _ { \mathbf { G } , \mathbf { G } ^ { \prime } , \mathbf { k } } ^ { 0 } ( \mathbf { q } , 0 ) \left[v \in S^{v} ; c \in\right.\right. & \left.\left.S^{c}\right]\right]=\left[\frac{1}{N_{\xi}^{2}} \sum_{i, j=1}^{N_{\xi}} \sum_{\substack{v, v^{\prime} \in S^{v} \\
c, c^{\prime} \in S^{c}}} \alpha_{i, v \mathbf{k}+\mathbf{q}}^{S^{v}}\left(\alpha_{i, v^{\prime} \mathbf{k}+\mathbf{q}}^{S^{v}}\right)^{*}\left(\alpha_{j, c \mathbf{k}}^{S^{c}}\right)^{*} \alpha_{j, c^{\prime} \mathbf{k}}^{S^{c}}\right. \\
& \left.\times\left\langle\phi_{c \mathbf{k}}\right| e^{-i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}\left|\phi_{v \mathbf{k}+\mathbf{q}}\right\rangle\left\langle\phi_{v^{\prime} \mathbf{k}+\mathbf{q}}\right| e^{i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}\left|\phi_{c^{\prime} \mathbf{k}}\right\rangle \times \frac{1}{\bar{E}_{S^{v} \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}}\right] \\
& -\left[\sum_{\substack{v \in S^{v} \\
c \in S^{c}}}\left\langle\phi_{c \mathbf{k}}\right| e^{-i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}\left|\phi_{v \mathbf{k}+\mathbf{q}}\right\rangle\left\langle\phi_{v \mathbf{k}+\mathbf{q}}\right| e^{i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}\left|\phi_{c \mathbf{k}}\right\rangle \times \frac{1}{E_{v \mathbf{k}+\mathbf{q}}-E_{c \mathbf{k}}}\right] . \tag{S25}
\end{align*}
$$

Again for any term not on the diagonal we have
$$
\begin{equation*}
\mathbb{E}[\text { off-diagonal }]=0 \tag{S26}
\end{equation*}
$$

For the diagonal we have
$$
\begin{align*}
{\left[v=v^{\prime} ; c=c^{\prime}\right] } & =\sum_{\substack{v \in S^{v} \\
c \in S^{c}}} M_{v c \mathbf{k}}(\mathbf{q}, \mathbf{G})^{*} M_{v c \mathbf{k}}\left(\mathbf{q}, \mathbf{G}^{\prime}\right) \times\left[\frac{1}{\bar{E}_{S^{v} \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}}-\frac{1}{\bar{E}_{S^{v} \mathbf{k}+\mathbf{q}}+\delta E_{v, \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}-\delta E_{c, \mathbf{k}}}\right]  \tag{S27}\\
& \approx \frac{1}{\left(\bar{E}_{S^{v} \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}\right)^{2}} \sum_{\substack{v \in S^{v} \\
c \in S^{c}}} M_{v c \mathbf{k}}(\mathbf{q}, \mathbf{G})^{*} M_{v c \mathbf{k}}\left(\mathbf{q}, \mathbf{G}^{\prime}\right) \times\left(\delta E_{v, \mathbf{k}+\mathbf{q}}-\delta E_{c, \mathbf{k}}\right) \tag{S28}
\end{align*}
$$

From expressions (S20), (S23), and (S28) we see that convergence of the mean-energy error is achieved as $N_{S} \rightarrow \infty$. In particular, the sums in these expressions are always bounded from above and below by a value proportional to $\pm \mathcal{M}^{2} \cdot \Delta E_{S} \cdot \operatorname{dim}(S)^{2}$. Here $\mathcal{M}$ is the maximum magnitude of all matrix elements, $\Delta E_{S}$ is the energy range spanned by $S$, and $\operatorname{dim}(S)$ is the dimension of subspace $S$. As $N_{S} \rightarrow \infty, \Delta E_{S} \rightarrow 0$ and $\operatorname{dim}(S) \rightarrow 0$, and convergence of the expectation of the polarizability evaluated with pseudobands is achieved.

Now, we turn to the variance of the error. This is a bit more tedious because, e.g., the products $\alpha_{i, v \mathbf{k}+\mathbf{q}}^{S^{v}}\left(\alpha_{i, v^{\prime} \mathbf{k}+\mathbf{q}}^{S^{v}}\right)^{*}\left(\alpha_{j, c \mathbf{k}}^{S^{c}}\right)^{*} \alpha_{j, c^{\prime} \mathbf{k}}^{S^{c}}$ and $\alpha_{i, v \mathbf{k}+\mathbf{q}}^{S^{v}}\left(\alpha_{i, v^{\prime} \mathbf{k}+\mathbf{q}}^{S^{v}}\right)^{*}$ are not necessarily independent random variables. So, we calculate
$$
\begin{equation*}
\operatorname{Var}\left[\operatorname{Err}\left[\chi_{\mathbf{G}, \mathbf{G}^{\prime}, \mathbf{k}}^{0}(\mathbf{q}, 0)[\mathrm{SPB}]\right]\right]=\mathbb{E}\left[\left|\operatorname{Err}_{\text {off-diagonal }}\left[\chi_{\mathbf{G}, \mathbf{G}^{\prime}, \mathbf{k}}^{0}(\mathbf{q}, 0)[\mathrm{SPB}]\right]\right|^{2}\right] \tag{S29}
\end{equation*}
$$
where this is the standard definition $\operatorname{Var}[Z]=\mathbb{E}\left[|Z-\mathbb{E}[Z]|^{2}\right]$, and we have subtracted the previously computed expectations to obtain Err $_{\text {off-diagonal }}$, which consists only of off-diagonal pseudobands contributions to $\chi_{\mathbf{G}, \mathbf{G}^{\prime}, \mathbf{k}}^{0}(\mathbf{q}, 0)[\mathrm{SPB}]$.
Explicitly, we have (all terms below are implied to have only off-diagonal contributions over the band indices in their sums):
$$
\begin{align*}
& \operatorname{Var}\left[\operatorname{Err}\left[\chi_{\mathbf{G}, \mathbf{G}^{\prime}, \mathbf{k}}^{0}(\mathbf{q}, 0)[\mathrm{SPB}]\right]\right] \\
= & \mathbb{E}\left[\left[v \leq N_{P}^{v} ; c \in S^{c}\right]^{2}+\left[v \in S^{v} ; c \leq N_{P}^{c}\right]^{2}+\left[v \in S^{v} ; c \in S^{c}\right]^{2}\right]  \tag{S30a}\\
+ & \mathbb{E}\left[\left[v \leq N_{P}^{v} ; c \in S^{c}\right]\left(\left[v \in S^{v} ; c \leq N_{P}^{c}\right]\right)^{*}+\left[v \in S^{v} ; c \leq N_{P}^{c}\right]\left(\left[v \leq N_{P}^{v} ; c \in S^{c}\right]\right)^{*}\right]  \tag{S30b}\\
+ & \mathbb{E}\left[\left[v \leq N_{P}^{v} ; c \in S^{c}\right]\left(\left[v \in S^{v} ; c \in S^{c}\right]\right)^{*}+\left[v \in S^{v} ; c \leq N_{P}^{c}\right]\left(\left[v \in S^{v} ; c \in S^{c}\right]\right)^{*}\right]  \tag{S30c}\\
+ & \mathbb{E}\left[\left[v \in S^{v} ; c \in S^{c}\right]\left(\left[v \leq N_{P}^{v} ; c \in S^{c}\right]\right)^{*}+\left[v \in S^{v} ; c \in S^{c}\right]\left(\left[v \in S^{v} ; c \leq N_{P}^{c}\right]\right)^{*}\right] . \tag{S30d}
\end{align*}
$$

Above, Eq. (S30a) contains the variances of each summand in Eq. (S12), Eq. (S30b) contains covariances of terms that are independent, and Eqs. (S30c, S30d) contain non-independent covariances.

First we deal with Eq. (S30a) by computing the following covariance:
$$
\begin{align*}
\operatorname{Cov}\left[\left(\left(\alpha_{i, n \mathbf{k}}^{S}\right)^{*} \alpha_{i, n^{\prime} \mathbf{k}}^{S}\right),\left(\left(\alpha_{i, n^{\prime \prime} \mathbf{k}}^{S}\right)^{*} \alpha_{i, n^{\prime \prime \prime} \mathbf{k}}^{S}\right)\right] & =\mathbb{E}\left[\left(\alpha_{i, n \mathbf{k}}^{S}\right)^{*} \alpha_{i, n^{\prime} \mathbf{k}}^{S} \alpha_{i, n^{\prime \prime} \mathbf{k}}^{S}\left(\alpha_{i, n^{\prime \prime} \mathbf{k}}^{S}\right)^{*}\right] \\
& = \begin{cases}1 & n=n^{\prime \prime} ; n^{\prime}=n^{\prime \prime \prime} \\
0 & \text { otherwise }\end{cases} \tag{S31}
\end{align*}
$$

Analogously,
$$
\begin{align*}
& \operatorname{Cov}\left[\left(\alpha_{i, v \mathbf{k}+\mathbf{q}}^{S^{v}}\left(\alpha_{i, v^{\prime} \mathbf{k}+\mathbf{q}}^{S^{v}}\right)^{*}\left(\alpha_{j, c \mathbf{k}}^{S^{c}}\right)^{*} \alpha_{j, c^{\prime} \mathbf{k}}^{S^{c}}\right),\left(\alpha_{i, v^{\prime \prime} \mathbf{k}+\mathbf{q}}^{S^{v}}\left(\alpha_{i, v^{\prime \prime \prime} \mathbf{k}+\mathbf{q}}^{S^{v}}\right)^{*}\left(\alpha_{j, c^{\prime \prime} \mathbf{k}}^{S^{c}}\right)^{*} \alpha_{j, c^{\prime \prime \prime} \mathbf{k}}^{S^{c}}\right)\right] \\
& =\mathbb{E}\left[\alpha _ { i , v \mathbf { k } + \mathbf { q } } ^ { S ^ { v } } ( \alpha _ { i , v ^ { \prime } \mathbf { k } + \mathbf { q } } ^ { S ^ { v } } ) ^ { * } ( \alpha _ { j , c \mathbf { k } } ^ { S ^ { c } } ) ^ { * } \alpha _ { j , c ^ { \prime } \mathbf { k } } ^ { S ^ { c } } \left(\alpha_{i, v^{\prime \prime} \mathbf{k}+\mathbf{q}}^{S^{v}} \alpha_{i, v^{\prime \prime \prime} \mathbf{k}+\mathbf{q}}^{\left.S_{j, c^{\prime \prime} \mathbf{k}}^{S^{c}}\left(\alpha_{j, c^{\prime \prime} \mathbf{k}}^{S^{c}}\right)^{*}\right]}\right.\right.  \tag{S32}\\
& = \begin{cases}1 & v=v^{\prime \prime} ; v^{\prime}=v^{\prime \prime \prime} ; c=c^{\prime \prime} ; c^{\prime}=c^{\prime \prime \prime} \\
0 & \text { otherwise }\end{cases}
\end{align*}
$$

Thus, the variance of the terms in Eq. (S30a) can be computed as the sum of variances of their summands over $n$ and $n^{\prime}$ :
$$
\begin{align*}
\mathbb{E}\left[\left[v \leq N_{P}^{v} ; c \in S^{c}\right]^{2}\right] & \left.\left.=\frac{1}{N_{\xi}} \sum_{S^{c}} \sum_{\substack{v \leq N_{P}^{v} \\
c \neq c^{\prime} \in S^{c}}}\left|\left\langle\phi_{c \mathbf{k}}\right| e^{-i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}\right| \phi_{v \mathbf{k}+\mathbf{q}}\right\rangle\left.\right|^{2}\left|\left\langle\phi_{v \mathbf{k}+\mathbf{q}}\right| e^{i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}\right| \phi_{c^{\prime} \mathbf{k}}\right\rangle\left.\right|^{2} \\
& \times \frac{1}{\left(E_{v \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}\right)^{2}}  \tag{S33a}\\
\mathbb{E}\left[\left[v \in S^{v} ; c \leq N_{P}^{c}\right]^{2}\right] & \left.\left.=\frac{1}{N_{\xi}} \sum_{S^{v}} \sum_{\substack{c \leq N_{P}^{c} \\
v \neq v^{\prime} \in S^{v}}}\left|\left\langle\phi_{c \mathbf{k}}\right| e^{-i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}\right| \phi_{v \mathbf{k}+\mathbf{q}}\right\rangle\left.\right|^{2}\left|\left\langle\phi_{v^{\prime} \mathbf{k}+\mathbf{q}}\right| e^{i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}\right| \phi_{c \mathbf{k}}\right\rangle\left.\right|^{2} \\
& \times \frac{1}{\left(\bar{E}_{S^{v} \mathbf{k}+\mathbf{q}}-E_{c \mathbf{k}}\right)^{2}}  \tag{S33b}\\
\mathbb{E}\left[\left[v \in S^{v} ; c \in S^{c}\right]^{2}\right] & \left.\left.=\frac{1}{N_{\xi}^{2}} \sum_{S^{v}, S^{c}} \sum_{v \neq v^{\prime} \in S^{v}}\left|\left\langle\phi_{c \mathbf{k}}\right| e^{-i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}\right| \phi_{v \mathbf{k}+\mathbf{q}}\right\rangle\left.\right|^{2}\left|\left\langle\phi_{v^{\prime} \mathbf{k}+\mathbf{q}}\right| e^{i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}\right| \phi_{c^{\prime} \mathbf{k}}\right\rangle\left.\right|^{2} \\
& \times \frac{1}{\left(\bar{E}_{S^{v} \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}\right)^{2}} \tag{S33c}
\end{align*}
$$

We can see that all these terms go as $1 / N_{\xi}$ or smaller, following the same convergence trend as the stochastic resolution of the identity. Next, we compute the remaining covariances. As noted before, Eq. (S30b) contains independent random variables, so these terms are 0 ( $\alpha_{v}$ cannot cancel out $\alpha_{c}$ and vice versa). For Eqs. (S30c, S30d) we have terms of the following form:
$$
\begin{equation*}
\mathbb{E}\left[\left(\alpha_{j, c \mathbf{k}}^{S^{c}}\right)^{*} \alpha_{j, c^{\prime} \mathbf{k}}^{S^{c}}\left(\alpha_{i, v \mathbf{k}+\mathbf{q}}^{S^{v}}\right)^{*} \alpha_{i, v^{\prime} \mathbf{k}+\mathbf{q}}^{S^{v}} \alpha_{j, c^{\prime \prime} \mathbf{k}}^{S^{c}}\left(\alpha_{j, c^{\prime \prime \prime} \mathbf{k}}^{S^{c}}\right)^{*}\right]=0, \tag{S34}
\end{equation*}
$$
as there is always a pair of phases ( $v, v^{\prime}$ in this case) which cannot be cancelled out. So, despite the explicit dependence of some of these products, they still have zero covariance. With this, we conclude that the variance of the polarizability is actually given by the sum of Eqs. (S33). Thus, we obtain that
$$
\begin{align*}
\operatorname{Var}\left[\operatorname{Err}\left[\chi_{\mathbf{G}, \mathbf{G}^{\prime}, \mathbf{k}}^{0}(\mathbf{q}, 0)[\mathrm{SPB}]\right]\right] & \propto \frac{C_{1}}{N_{\xi}}+\frac{C_{2}}{N_{\xi}^{2}}  \tag{S35a}\\
\Longrightarrow \quad \lim _{N_{\xi} \rightarrow \infty} \operatorname{Var}\left[\operatorname{Err}\left[\chi_{\mathbf{G}, \mathbf{G}^{\prime}, \mathbf{k}}^{0}(\mathbf{q}, 0)[\mathrm{SPB}]\right]\right] & =0 \tag{S35b}
\end{align*}
$$
for some constants $C_{1}, C_{2}$ that can depend on $N_{S}$ but are always finite. Thus, Eq. (S35b) holds for any value of $N_{S}$. With the expectation of the error also going to 0 as $N_{S} \rightarrow \infty$ as shown before, this completes the proof of convergence of the static polarizability with pseudobands.

We note that, in practice, due to the rapidly oscillating nature of matrix elements $M_{v c \mathbf{k}}\left(\mathbf{q}, \mathbf{G}^{\prime}\right)$ with respect to $v$ and $c$, there is a large amount of averaging out in expressions (S20), (S23), and (S28). This enables convergence of the expectations derived even with modest values of $N_{S}$ and $N_{\xi}$, i.e., the average case is much better than the worst case.

In section 3.2.2 we provide a framework for extending the usage of stochastic pseudobands to $\omega \neq 0$, which we verify numerically in section S2.3. Note that the above derivations can be extended to small $\omega$ up to roughly $\min \left(E_{\mathrm{CBM}}-E_{v=N_{P}^{v}}, E_{c=N_{P}^{c}}-E_{\mathrm{VBM}}\right)$, as $\omega$ bounded by this value do not produce any divergences in the above expressions.

\subsection*{3.2.1 Partition of Subspaces}

Here we derive a physically-motivated heuristic for how to partition the mean-field states $\left|\phi_{n \mathbf{k}}\right\rangle$ into the stochastic subspaces $\{S\}$, using the expressions derived in the previous section. While optimizing this partition over the error of $\chi^{0}$, for example, would be very difficult, it is also not guaranteed that this optimum would minimize the error of $\Sigma^{G W}$ anyways. Instead, we simply demand that each stochastic subspace should contribute a roughly constant error to $\chi^{0}(\omega=0)$. We saw before that $N_{S}$ is primarily responsible for controlling the expectation value of $\chi^{0}[\mathrm{SPB}]$, while $N_{\xi}$ is responsible for controlling the variance. Therefore, we only focus on the expectation value for this problem. Additionally, we ignore the contribution of Eq. (S28) due to its larger energy denominator. Therefore, we focus on Eq. (S20) (Eq. (S23) behaves similarly), and derive the following bounds:
$$
\begin{align*}
\left|\mathbb{E}\left[\operatorname{Err}\left[\chi_{\mathbf{G}, \mathbf{G}^{\prime}, \mathbf{k}}^{0}(\mathbf{q}, 0)\left[v ; c \in S^{c}\right]\right]\right]\right| & \approx \frac{1}{\left(E_{v \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}\right)^{2}}\left|\sum_{n \in S^{c}} M_{v n \mathbf{k}}(\mathbf{q}, \mathbf{G})^{*} M_{v n \mathbf{k}}\left(\mathbf{q}, \mathbf{G}^{\prime}\right) \times \delta E_{n, \mathbf{k}}\right| \\
& \leq \frac{\mathcal{M}^{2}}{\left(E_{v \mathbf{k}+\mathbf{q}}-\bar{E}_{S^{c} \mathbf{k}}\right)^{2}} \sum_{n \in S^{c}}\left|\delta E_{n, \mathbf{k}}\right| \\
& \leq \frac{\mathcal{M}^{2}}{\left(\bar{E}_{S^{c} \mathbf{k}}\right)^{2}} \cdot \Delta E_{S^{c} \mathbf{k}} \cdot \operatorname{dim}\left(S^{c}\right) \tag{S36}
\end{align*}
$$
where in the last line we used that the Fermi level is our reference energy $E_{F}=0$. In 3D, $\operatorname{dim}\left(S^{c}\right)= \int g(E) d E \sim \bar{E}_{S}^{1 / 2} \Delta E_{S}$ where $g$ is the density of states, so we want $\Delta E_{S}^{2} / \bar{E}_{S}^{3 / 2}$ to be a constant to accrue constant error in each slice. To simplify this condition and be dimension-independent, we instead take $\Delta E_{S} / \bar{E}_{S}$ to be a constant as a practical prescription.

With these approximations we have a simple and exponential partition of the total energy range into stochastic subspaces defined by the convergence parameter $\mathcal{F}$,
$$
\begin{equation*}
\frac{\Delta E_{S}}{\bar{E}_{S}} \equiv \mathcal{F}=\text { const. } \tag{S37}
\end{equation*}
$$

For a fixed plane-wave cutoff, this also gives the relationship $\mathcal{F} \sim 1 / N_{S}$ where $N_{S}$ is the number of subspaces in the partition.

\subsection*{3.2.2 Extension to Full-Frequency Calculations}

As noted at the end of section 3.2, the convergence of $\chi^{0}$ was only proved for small $\omega$ roughly bounded by $\min \left(E_{\mathrm{CBM}}-E_{v=N_{P}^{v}}, E_{c=N_{P}^{c}}-E_{\mathrm{VBM}}\right)$. To extend this to larger $\omega$ we must treat the full Adler-Wiser formula with both its retarded and advanced components:
$$
\begin{equation*}
\chi_{\mathbf{G}, \mathbf{G}^{\prime}}^{0, r / a}(\mathbf{q}, \omega)=\sum_{v c \mathbf{k}} M_{v c \mathbf{k}}(\mathbf{q}, \mathbf{G})^{*} M_{v c \mathbf{k}}\left(\mathbf{q}, \mathbf{G}^{\prime}\right) \times \frac{1}{2}\left[\frac{1}{E_{v \mathbf{k}+\mathbf{q}}-E_{c \mathbf{k}}-\omega \mp i \delta}+\frac{1}{E_{v \mathbf{k}+\mathbf{q}}-E_{c \mathbf{k}}+\omega \pm i \delta}\right], \tag{S38}
\end{equation*}
$$
where the upper (lower) signs are for the retarded (advanced) function, and $\delta$ is a broadening used in practice to account for finite k-point sampling. Eq. (S38) has symmetric poles at both positive and negative $\omega$, and both terms individually take the same form as the $\chi^{0}(\omega=0)$ in Eq. (S11). Thus, for finite $\omega \neq 0$, we can see that the same derivations from before go through, as long as we don't produce any divergent terms. Due to the broadening $i \delta$, we never encounter true divergences, so convergence is still achieved in a similar manner as for the Green's function in section S3.1.

While convergence is achieved, it is not as rapid as for the GPP due to $\omega$ being in the regime of poles which are approximated with pseudobands. To make convergence more rapid, we employ the following scheme to modify the subspace partition described in the previous section: Given a maximum frequency of interest $\omega_{\max }$ and a frequency-grid spacing $\delta \omega$ over which the dielectric function is to be sampled, we partition subspaces so that they span a uniform energy range $\Delta E_{S}=$ const. up to $\omega_{\max }$. Beyond $\omega_{\max }$, we begin the exponential slices as usual. We typically take $\Delta E_{S} \sim \delta \omega$ for the uniform slices, though the energy range of each uniform slice is an additional convergence parameter that only appears for these full-frequency calculations. The full-frequency partitioning scheme is depicted diagrammatically in Figure S4 below.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f0c30553-77d0-403a-a265-b56065452e10-19.jpg?height=341&width=1458&top_left_y=1007&top_left_x=339}
\captionsetup{labelformat=empty}
\caption{Figure S4: Diagram of the slice partition for pseudobands that can handle full-frequency dielectric calculations. Only the partition for the unoccupied states is shown, but the same is true for the occupied states.}
\end{figure}

We note that $\omega=0$ is no longer a special frequency in fully frequency-dependent calculations that do not rely on plasmon-pole models, hence the uniform slices to control the error induced from being in the vicinity of poles. We can extend this argument to deduce that, in either the fully frequencydependent or plasmon-pole-based calculations, we may set $N_{P}=0$ and still achieve convergence with no protected states. As $N_{S}$ and $N_{\xi}$ increase, the error of the polarizability of the slice at $E_{F}$ still tends to 0 . This allows a fully quadratic formalism for the evaluation of the polarizability, though the drawback is that one must use better convergence parameters than if $N_{P}$ was finite. In practice, we used finite values for $N_{P}$ to simplify the workflow, since it allows us to use a similar set of wavefunction files for the calculation of the polarizability and self-energy, though this was not required.

Moreover, this argument for setting $N_{P}=0$ extends to metals. In particular, the reason for changing the way we choose slices in the explicit frequency scheme is because we evaluate the polarizability at energies $\omega$ that are close to occupied-to-unoccupied transitions that appear as poles in the polarizability. For metals, either within plasmon-pole models or an explicit frequency-dependent calculation, one runs into the same issue of evaluating the polarizability near a pole, except this pole occurs at $\omega=0$. However, as argued above, with the use of a finite broadening $\delta$, the pseudobands approach always converges for the evaluation of any frequency. Combined with the fact that one can always use a finite (convergable to 0 ) broadening $\delta$ either in plasmon-pole or explicit frequency calculations, we conclude that $N_{P}=0$ is applicable to metals as well as semiconductors. In fact, a broadening not smaller than $\delta \sim v_{F} \Delta \mathbf{k}$ is desirable anyway (where $v_{F}$ is the Fermi velocity and $\Delta \mathbf{k}$ is the $\mathbf{k}$-grid
spacing), since using a smaller broadening artificially discretizes the density of states and does not capture the continuum of transitions which characterizes the dielectric response of metals.

\subsection*{3.3 Convergence of the GW Self-Energy $\Sigma^{G W}$}

The self-energy is somewhat more complicated to evaluate by hand, and we only discuss convergence qualitatively. However, having shown convergence for $G[\mathrm{SPB}]$ and $\chi^{0}[\mathrm{SPB}]$, convergence of $\Sigma^{G W}$ follows.

Here, we review the expressions used in the contour-deformation approach for computing $\Sigma^{G W}$ that is commonly employed in practice [21-23]. We analyze the forms of these expressions and make some important points for applying pseudobands to $\Sigma^{G W}$. The expressions evaluated for $\Sigma^{G W}$ in the contour-deformation approach are
$$
\begin{align*}
\Sigma^{G W}\left(\mathbf{r}, \mathbf{r}^{\prime}, \omega\right) & \equiv \Sigma^{X}\left(\mathbf{r}, \mathbf{r}^{\prime}\right)+\Sigma^{\operatorname{Cor}}\left(\mathbf{r}, \mathbf{r}^{\prime}, \omega\right)  \tag{S39}\\
\Sigma^{\operatorname{Cor}}\left(\mathbf{r}, \mathbf{r}^{\prime}, \omega\right) & \equiv \Sigma^{\operatorname{Int}}\left(\mathbf{r}, \mathbf{r}^{\prime}, \omega\right)+\Sigma^{\operatorname{Res}}\left(\mathbf{r}, \mathbf{r}^{\prime}, \omega\right)  \tag{S40}\\
\Sigma^{X}\left(\mathbf{r}, \mathbf{r}^{\prime}\right) & =-\sum_{v}^{\text {occ }} \phi_{v}(\mathbf{r}) \phi_{v}^{*}\left(\mathbf{r}^{\prime}\right) v\left(\mathbf{r}, \mathbf{r}^{\prime}\right)  \tag{S41}\\
\Sigma^{\operatorname{Res}}\left(\mathbf{r}, \mathbf{r}^{\prime}, \omega\right) & =-\sum_{n} \phi_{n}(\mathbf{r}) \phi_{n}^{*}\left(\mathbf{r}^{\prime}\right) W^{\operatorname{Cor}}\left(\mathbf{r}, \mathbf{r}^{\prime}, E_{n}-\omega\right) \times\left[f_{n} \theta\left(E_{n}-\omega\right)-\left(1-f_{n}\right) \theta\left(\omega-E_{n}\right)\right]  \tag{S42}\\
\Sigma^{\operatorname{Int}}\left(\mathbf{r}, \mathbf{r}^{\prime}, \omega\right) & =-\frac{1}{\pi} \sum_{n} \phi_{n}(\mathbf{r}) \phi_{n}^{*}\left(\mathbf{r}^{\prime}\right) \int_{0}^{\infty} d \omega^{\prime} \frac{\omega-E_{n}}{\left(\omega-E_{n}\right)^{2}+\left(\omega^{\prime}\right)^{2}} \times W^{\operatorname{Cor}}\left(\mathbf{r}, \mathbf{r}^{\prime}, i \omega^{\prime}\right) \tag{S43}
\end{align*}
$$
where $W^{\text {Cor }} \equiv W\left(\mathbf{r}, \mathbf{r}^{\prime}, \omega\right)-v\left(\mathbf{r}, \mathbf{r}^{\prime}\right), v$ is the bare Coulomb interaction, $W=\epsilon^{-1} v$ is the screened Coulomb interaction, $f_{n}$ is the Fermi-Dirac occupation factor, and $\theta(\omega)$ is the Heaviside step function.
We note that $\Sigma^{\mathrm{Int}}$ involves an integral of the screened Coulomb interaction along the imaginary axis, so it does not pick up poles of $W$ or $G$ [23], and one can directly utilize the pseudobands approach. On the other hand, when evaluating the self-energy for states close to the Fermi energy, $\Sigma^{\operatorname{Res}}(\omega)$ involves sums over final states $n$ having energy $E_{n}$ closer to the Fermi energy than the energy $\omega$ at which one evaluates the self-energy [23]. Such final states depend sensitively on the energy at which one is approximating them, which one can account for by systematically using smaller broadening parameters and stochastic subspaces, as in the case of metals, or by simply employing a nonzero protection window $N_{P}$. Because self-energy calculations are relatively inexpensive, and because we eventually wish to compute matrix elements $\langle n \mathbf{k}| \Sigma^{G W}|n \mathbf{k}\rangle$ evaluated on deterministic states $|n \mathbf{k}\rangle$, we find it easier to simply employ a finite $N_{P}$. Finally, we note that the exchange interaction $\Sigma^{X}$ is static but involves sums over all occupied states. While we can in principle benefit from the stochastic pseudobands approach, we notice that the speed-up for the systems studied here is not advantageous given the extra stochastic error. So, the pseudobands approach is also valid for the evaluation of the quasiparticle self-energy, although we recommend one to use a finite value of $N_{P}$ and only conduction pseudobands. While this approach may be optimized in the future, the evaluation of the self-energy for the VBM and CBM scales as $O\left(N^{3}\right)$, and hence does not represent the computational bottleneck in typical GW calculations.

This concludes the convergence derivations of GW quantities within the pseudobands approach.

\section*{References}
[1] Altman, A. R.; Kundu, S.; da Jornada, F. H. Supplemental Datasets for Manuscript: Mixed Stochastic-Deterministic Approach for Many-Body Perturbation Theory Calculations. 2023; https://doi.org/10.5281/zenodo. 8278011.
[2] Giannozzi, P.; Baroni, S.; Bonini, N.; Calandra, M.; Car, R.; Cavazzoni, C.; Ceresoli, D.; Chiarotti, G. L.; Cococcioni, M.; Dabo, I., et al. QUANTUM ESPRESSO: a modular and opensource software project for quantum simulations of materials. Journal of physics: Condensed matter 2009, 21, 395502.
[3] Perdew, J. P.; Burke, K.; Ernzerhof, M. Generalized gradient approximation made simple. Physical review letters 1996, 77, 3865.
[4] van Setten, M. J.; Giantomassi, M.; Bousquet, E.; Verstraete, M. J.; Hamann, D. R.; Gonze, X.; Rignanese, G.-M. The PseudoDojo: Training and grading a 85 element optimized normconserving pseudopotential table. Computer Physics Communications 2018, 226, 39-54.
[5] Schlipf, M.; Gygi, F. Optimization algorithm for the generation of ONCV pseudopotentials. Computer Physics Communications 2015, 196, 36-44.
[6] Deslippe, J.; Samsonidze, G.; Strubbe, D. A.; Jain, M.; Cohen, M. L.; Louie, S. G. BerkeleyGW: A massively parallel computer package for the calculation of the quasiparticle and optical properties of materials and nanostructures. Computer Physics Communications 2012, 183, 1269-1289.
[7] van Setten, M. J.; Caruso, F.; Sharifzadeh, S.; Ren, X.; Scheffler, M.; Liu, F.; Lischner, J.; Lin, L.; Deslippe, J. R.; Louie, S. G., et al. GW 100: Benchmarking G 0 W 0 for molecular systems. Journal of chemical theory and computation 2015, 11, 5665-5687.
[8] Hybertsen, M. S.; Louie, S. G. Electron correlation in semiconductors and insulators: Band gaps and quasiparticle energies. Physical Review B 1986, 34, 5390.
[9] Ismail-Beigi, S. Truncation of periodic image interactions for confined systems. Physical Review B 2006, 73, 233103.
[10] Downs, R. T.; Hall-Wallace, M. The American Mineralogist crystal structure database. American Mineralogist 2003, 88, 247-250.
[11] Rangel, T.; Del Ben, M.; Varsano, D.; Antonius, G.; Bruneval, F.; da Jornada, F. H.; van Setten, M. J.; Orhan, O. K.; O'Regan, D. D.; Canning, A., et al. Reproducibility in G0W0 calculations for solids. Computer Physics Communications 2020, 255, 107242.
[12] Larsen, A. H.; Mortensen, J. J.; Blomqvist, J.; Castelli, I. E.; Christensen, R.; Dułak, M.; Friis, J.; Groves, M. N.; Hammer, B.; Hargus, C., et al. The atomic simulation environment-a Python library for working with atoms. Journal of Physics: Condensed Matter 2017, 29, 273002.
[13] Thompson, A. P.; Aktulga, H. M.; Berger, R.; Bolintineanu, D. S.; Brown, W. M.; Crozier, P. S.; in't Veld, P. J.; Kohlmeyer, A.; Moore, S. G.; Nguyen, T. D., et al. LAMMPS-a flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales. Computer Physics Communications 2022, 271, 108171.
[14] Naik, M. H.; Maity, I.; Maiti, P. K.; Jain, M. Kolmogorov-Crespi potential for multilayer transition-metal dichalcogenides: capturing structural transformations in moiré superlattices. The Journal of Physical Chemistry C 2019, 123, 9770-9778.
[15] Jiang, J.-W.; Park, H. S.; Rabczuk, T. Molecular dynamics simulations of single-layer molybdenum disulphide (MoS2): Stillinger-Weber parametrization, mechanical properties, and thermal conductivity. Journal of Applied Physics 2013, 114, 064307.
[16] da Jornada, F. H.; Qiu, D. Y.; Louie, S. G. Nonuniform sampling schemes of the Brillouin zone for many-electron perturbation-theory calculations in reduced dimensionality. Physical Review B 2017, 95, 035109.
[17] Qiu, D. Y.; da Jornada, F. H.; Louie, S. G. Optical spectrum of MoS 2: many-body effects and diversity of exciton states. Physical review letters 2013, 111, 216805.
[18] Del Ben, M.; da Jornada, F. H.; Antonius, G.; Rangel, T.; Louie, S. G.; Deslippe, J.; Canning, A. Static subspace approximation for the evaluation of G 0 W 0 quasiparticle energies within a sum-over-bands approach. Physical Review B 2019, 99, 125128.
[19] Marek, A.; Blum, V.; Johanni, R.; Havu, V.; Lang, B.; Auckenthaler, T.; Heinecke, A.; Bungartz, H.-J.; Lederer, H. The ELPA library: scalable parallel eigenvalue solutions for electronic structure theory and computational science. Journal of Physics: Condensed Matter 2014, 26, 213201.
[20] Lu, X.; Li, X.; Yang, L. Modulated interlayer exciton properties in a two-dimensional moiré crystal. Physical Review B 2019, 100, 155416.
[21] Oschlies, A.; Godby, R.; Needs, R. GW self-energy calculations of carrier-induced band-gap narrowing in n-type silicon. Physical Review B 1995, 51, 1527.
[22] Lebègue, S.; Arnaud, B.; Alouani, M.; Bloechl, P. Implementation of an all-electron GW approximation based on the projector augmented wave method without plasmon pole approximation: Application to Si, SiC, AlAs, InAs, NaH, and KH. Physical Review B 2003, 67, 155208.
[23] Bruneval, F. Exchange and Correlation in the Electronic Structure of Solids, from Silicon to Cuprous Oxide: GW approximation and beyond. PhD Thesis 2005,