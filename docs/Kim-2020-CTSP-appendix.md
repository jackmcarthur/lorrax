dices. In Appendix A, the Gauss-Laguerre quadrature for window pairs without energy crossing is discussed. In Appendix B, the determination of the optimal windowing by minimization of the cost function for quantities without energy crossings is described. Appendices C and D discuss the weight function and quadrature employed to treated window pairs with energy crossings, respectively, while Appendix E describes minimization of the cost function for quantities whose evaluation involves treating energy window pairs with energy crossings. In Appendix F an alternative $\mathcal{O}\left(N^{3}\right)$ method based on interpolation is given, and in Appendix G computational details related to the results presented in the main text are described. Last, matlab code to generate the weights and nodes of the Hermite-Gauss-Laguerre quadrature is presented.

\section*{Appendix A: Gauss-Laguerre quadrature optimization}

We provide the optimizations required to evaluate energy denominators by discrete approximation to time domain integrals for a set of energies in a window pair.

\section*{1. Optimal error matching choice for energy scale $\zeta$}

First, we describe the optimal error matching choice of $\zeta_{l m}$ for the Gauss-Laguerre (GL) quadrature of Eq. (24). We suppress the energy window index $l m$ and describe why $\zeta^{-1} \approx \sqrt{E^{\text {(bw) }} E^{\text {(gap) }}}$ is a good choice for the energy scale $\zeta$ : it equalizes the error of the GL quadrature across a given window pair.

We seek to optimally approximate the continuous time integral yielding the desired energy denominaor via numerical quadrature,
$$
\frac{1}{\Delta}=\zeta \int_{0}^{\infty} e^{-\zeta \Delta \tau} d \tau \approx \zeta \sum_{u=1}^{N^{(\tau, \mathrm{GL})}} w_{u} e^{-\tau_{u}(\zeta \Delta-1)}
$$
for $\Delta=E_{c}-E_{v}>0$. That is, defining the dimensionless quantity, $x=\zeta \Delta$, we wish to minimize the error
$$
\begin{equation*}
\frac{\epsilon^{(q)}(x)}{x}=\frac{1}{x}-\sum_{u=1}^{N^{(\tau, \mathrm{GL})}} w_{u} \exp \left(-\tau_{u}(x-1)\right) . \tag{A1}
\end{equation*}
$$
for $x$ spanning the scaled range of a given window pair. We first note that the error is exactly zero at $x=1$ since GL quadrature is exact when integrating $e^{-\tau}$. Figure 12 shows a plot of the error versus $x$. The the error curve is symmetric around $\ln x=0$, especially when smaller error values are of interest, which is the case herein. That is, the integration error, to a good approximation, is even in $\ln x$ about $\ln x=0$.

Second, the interband energies $\Delta$ range from $E^{(\text {gap })}$ to $E^{(\mathrm{bw})}$. Examining Fig. 12, the lowest errors are sampled

\begin{figure}
\includegraphics[max width=\textwidth]{https://cdn.mathpix.com/cropped/59dcea53-65fc-493a-9d14-75ff36f4a405-19.jpg?height=461&width=558&top_left_y=178&top_left_x=1228}
\captionsetup{labelformat=empty}
\caption{FIG. 12. Gauss-Laguerre (GL) quadrature error in the integration of $e^{-x \tau}$ with 12 quadrature points as a function of $\log _{10} x$, solid blue curve (see Eq. (A.1)). The dashed black horizontal line shows that for $-0.75 \lesssim \log _{10} x \lesssim 0.75$ equal error is generated for $x$ and $1 / x$}
\end{figure}
as $x$ ranges from its lowest value of $\zeta E^{\text {(gap) }}$ to its highest value of $\zeta E^{\text {(bw) }}$. Therefore, it is reasonable to choose $\zeta$ such that $x=\zeta E^{(\mathrm{gap})}<1$ and $x=\zeta E^{(\mathrm{bw})}>1$ straddle $x=1$ and have the same error, i.e., optimal error equalization. For a symmetric error function about $\ln x=0$, this requires $-\ln \left(\zeta E^{\text {(gap) }}\right)=\ln \left(\zeta E^{\text {(bw) }}\right)$ which yields the geometric mean $\zeta^{-1}=\sqrt{E^{(\text {bw })} E^{(\text {gap })}}$. The geometric mean becomes exactly optimal as $N^{(\tau, \mathrm{GL})}$ is increased as well as when $E^{\text {(bw) }} / E^{\text {(gap) }}$ is close to unity (the many windows limit).

\section*{2. Number of Gauss-Laguerre quadarature points for bounded error}

When we fix $\zeta^{-1}=\sqrt{E^{\text {(bw) }} E^{\text {(gap) }}}$, the maximum fractional error of Eq. (A1), $\epsilon^{(q)}$, occurs at the largest energy transition (i.e., the error in computing the inverse energy $1 / E^{\text {(bw) }}$ via quadrature). For $N^{(\tau, \mathrm{GL})}$ quadrature points, we have
$$
\begin{equation*}
\epsilon^{(q)}(\alpha)=1-\alpha \sum_{u=1}^{N^{(\tau, \mathrm{GL})}} w_{u} \exp \left[(1-\alpha) \tau_{u}\right] . \tag{A2}
\end{equation*}
$$
where $\alpha=\sqrt{E^{(\mathrm{bw})} / E^{(\mathrm{gap})}}$. The analogous equation for the fractional error in the computation of $1 / E^{\text {(gap) }}$ has $\alpha$ replaced by $1 / \alpha$, and is equal to the error of Eq. (A2) due to optimal error-matching choice of $\zeta$. Figure 13 displays a contour plot of the fractional quadrature error, $\epsilon^{(q)}$ of Eq. (A2). The plot demonstrates that $N^{(\tau, \mathrm{GL})}$ is essentially linear in $\alpha$ for any fixed choice of fractional error. Analysis of the contour plot shows that an accurate and compact explicit relation between the variables is
$$
\begin{align*}
N^{(\tau, \mathrm{GL})}\left(\alpha ; \epsilon^{(q)}\right) & =\alpha\left(y-0.3 \ln \epsilon^{(q)}\right)  \tag{A3}\\
y & =0.4
\end{align*}
$$

This equation fits the data well for $\epsilon^{(q)} \leq 0.135$ but the range can be extended to unit $\epsilon^{(q)}$ by taking $y=1.0$.

\begin{figure}
\includegraphics[max width=\textwidth]{https://cdn.mathpix.com/cropped/59dcea53-65fc-493a-9d14-75ff36f4a405-20.jpg?height=465&width=572&top_left_y=184&top_left_x=309}
\captionsetup{labelformat=empty}
\caption{FIG. 13. The fractional error, $\epsilon^{(q)}$, of Gauss-Lauguerre quadrature (for $x=1 / E^{(\mathrm{bw})}$ ) as a function of $\alpha$ and $N^{(\tau, \mathrm{GL})}$. Here, $\alpha$ is defined to be $\sqrt{E^{(\mathrm{bw})} / E^{(\mathrm{gap})}}$. Each contour is labeled by $\epsilon^{(q)}$. For a fixed fractional error, $N^{(\tau, \mathrm{GL})}$ is linear in $\alpha$.}
\end{figure}

Note, our choice bounds the integration error: the end points of the energy windows are worst cases and all other transitions are computed more accurately. Hence, the number of quadrature points needed to compute the interband transitions within an energy window pair can be simply estimated so as to ensure a maximal a priori fractional error bound, $\epsilon^{(q)}$.

\section*{Appendix B: Optimal sets of energy windows}

We describe our prescription to determine the optimal number and placement of energy windows in the range of $E_{c}$ and $E_{v}$. This is accomplished by minimizing the computational cost function $C^{(\mathrm{GL})}\left(\epsilon^{(q)}\right)$ of Eq. (26). In this appendix, we omit the fractional error level $\epsilon^{(q)}$ as it does not affect the optimal set of energy windows. To motivate the discussion, consider a $2 \times 2$ window scheme where the two free parameters are the dividing energy values $E_{v}^{*}$ and $E_{c}^{*}$ in the valence and conduction bands, respectively, that determine the boundaries of the windows. These are converted to dimensionless quantities, $E_{c}^{\text {(ratio) }}=\left(E_{c}^{*}-E_{c}^{\text {(min) }}\right) /\left(E_{c}^{\text {(max) }}-E_{c}^{*}\right)$ and $E_{v}^{\text {(ratio) }}=\left(E_{v}^{*}-E_{v}^{\text {(min) }}\right) /\left(E_{v}^{\text {(max) }}-E_{v}^{*}\right)$. Figure 14 shows the dependence of the cost $C^{(\mathrm{GL})}$ on two ratios for the case of flat densities of states. The function, $C^{(\mathrm{GL})}$, is a smooth function of the window boundaries and we find that this smoothness is not confined to $2 \times 2$ windowing but carries over to larger number of windows. Note, the position of the minimum in Fig. 14 is nontrivial, occurring at the point, $\left(E_{v}^{(\text {ratio })}, E_{c}^{(\text {ratio })}\right)=(1.25,0.29)$.

Since $C^{(\mathrm{GL})}$ is a smooth function of the energy window partitions, for a given number of windows $\left(N_{v_{w}}, N_{c_{w}}\right)$ and some starting set of window partitions (e.g. all equal), we can employ a simple gradient descent algorithm to minimize $C^{(\mathrm{GL})}$ over the positions of the energy window boundaries and to find the minimum value of $C^{(\mathrm{GL})}\left(N_{v_{w}}, N_{c_{w}}\right)$. By varying the number of windows

\begin{figure}
\includegraphics[max width=\textwidth]{https://cdn.mathpix.com/cropped/59dcea53-65fc-493a-9d14-75ff36f4a405-20.jpg?height=439&width=583&top_left_y=208&top_left_x=1209}
\captionsetup{labelformat=empty}
\caption{FIG. 14. Computational cost to compute the static $P$ using a $N_{v_{w}}=2 \times N_{c_{w}}=2$ window scheme as a function of energy window size for a 16 -atom Si system with 399 states ( 32 occupied and 367 unoccupied states) and one $k$-point (no $k$-point sampling and hence $q=0$ strictly). The position of the minimum does not occur at the equipartition point, $(1,1)$, but rather at $\left(E_{v}^{\text {(ratio) }}, E_{c}^{\text {(ratio) }}\right)=(1.25,0.29)$.}
\end{figure}
$N_{v_{w}}, N_{c_{w}}$ over a reasonable range and tabulating the minimized cost function $C^{(\mathrm{GL})}\left(N_{v_{w}}, N_{c_{w}}\right)$, the global minimum and the hence the optimal choice of windowing, i.e., the number of windows pairs $\left\{N_{v_{w}}, N_{c_{w}}\right\}$ and their partitioning of the energy ranges, can be found. In practice, varying the number of window from 1 to 9 is sufficient to determine the best windowing choice for all the systems we have considered; so that, 81 small minimization procedures are performed in total. Note, the process is simplified because of the separable nature of $N^{(\tau, \mathrm{GL})}$ of Eq. A3: the partitioning results do not depend on the desired fractional error, $\epsilon^{(q)}$.

Figure 15 illustrates the minimal value the cost function at several $\left\{N_{v_{w}}, N_{c_{w}}\right\}$ for a bulk Si crystal described by 16 -atom supercell and 32 valence and 367 conduction band states. Here, the minimal computational load occurs for at the point, $\left(N_{v_{w}}=1, N_{c_{w}}=5\right)$. For $k$-point sampling over the first BZ under the CTSP-W method, the computation of $P^{q}$ at momentum transfer $q$ is optimized by applying the windowing with cost function minimization procedure to each $k, k+q$ pair. That is, the densities of states acquire band indices, $D^{k}(E), D^{k+q}(E)$, and the number of windows $\left\{N_{v_{w}}^{k}, N_{c_{w}}^{k+q}\right\}$ and their partition, the sets $\left\{E_{k}^{(v, \min )}, E_{k}^{(v, \max )}\right\},\left\{E_{k+q}^{(c, \min )}, E_{k+q}^{(c, \max )}\right\}$, are optimized for each $(k, k+q)$ pair in the BZ .

\section*{Appendix C: Weight function for window pairs with energy crossings}

We develop a weight function and associated quadrature for the case when $F(x ; \zeta)$ must be evaluated for energy differences $x$ that are both positive and negative within a window pair, i.e., energy crossings occur. A standard choice in the GW literature is to employ a Lorentzian broadening parameter $\gamma>0$ to regularize the

\begin{figure}
\includegraphics[max width=\textwidth]{https://cdn.mathpix.com/cropped/59dcea53-65fc-493a-9d14-75ff36f4a405-21.jpg?height=433&width=602&top_left_y=214&top_left_x=305}
\captionsetup{labelformat=empty}
\caption{FIG. 15. Minimized computational cost, $C^{(\mathrm{GL})}$, to compute the static $P$ of a 16 -atom Si system for window pairs $\left\{N_{v_{w}}, N_{c_{w}}\right\}$ spanning ( $N_{v_{w}}=9 \times N_{c_{w}}=9$ ) (81 total pairs). The total number of bands in the system was taken to be 399 ( 32 occupied and 367 unoccupied states) and one $k$-point (no $k$-point sampling and hence $q=0$ strictly). The position of the minimum is at the point, ( $N_{v_{w}}=1, N_{c_{w}}=5$ ).}
\end{figure}
singularity of $1 / x$ by replacing it with
$$
\begin{equation*}
F(x)=\operatorname{Im} \frac{\gamma}{1-i x \gamma}=\frac{x}{x^{2}+\gamma^{-2}} . \tag{C1}
\end{equation*}
$$
in the spirit of the additional scattering that typically ameliorates resonances in real materials. This odd function in $x$ is continuous, approximates $1 / x$ when $\gamma|x| \gg 1$, and has a separable form as a Fourier integral
$$
\begin{equation*}
F(x)=\gamma \operatorname{Im} \int_{0}^{\infty} d \tau e^{-\tau} e^{i \tau x \gamma} \tag{C2}
\end{equation*}
$$

The exponential weight function implies that the most appropriate quadrature method for approximating the integral is the simply Gauss-Laguerre quadrature. Hence, this $F(x)$ can be used to separate the sums over $n$ and $p$ when computing $\Sigma(\omega)$.

The difficulties with this choice are practical. First, the quadrature grids needed for reasonable errors can become large. Second, the function approaches $1 / x$ only when $|x| \gg \gamma^{-1}$ such that if $\gamma^{-1}$ is not small compared to the width of the energy windows being employed, there will be sizable errors across window boundaries when we switch from $F(x)$ to $1 / x$. On the other hand, if we make $\gamma^{-1}$ small to avoid this matching error, the steepness of $F(x)$ near the origin, which is directly related to the rapid oscillations versus $\tau$ of $e^{-i \gamma x \tau}$ with large $\gamma$ in the integral form of $F$ in Eq. (C2), requires a large quadrature grid to describe accurately.

We alleviate the above difficulties by taking advantage of the freedom afforded in choosing the functional form of $F(x ; \zeta)$ in Eq. (9). Instead of employing the weight function $h(\tau ; \zeta)=|\zeta| e^{-\tau}$ with $\zeta=i \gamma$, we propose to use
$$
h(\tau ; \zeta)=|\zeta| \exp \left(-\tau-\tau^{2} / 2\right)
$$
which falls off much faster for large $\tau$ and will thus generate a much smoother $F(x)$ for small $x$. However, since

\begin{figure}
\includegraphics[max width=\textwidth]{https://cdn.mathpix.com/cropped/59dcea53-65fc-493a-9d14-75ff36f4a405-21.jpg?height=538&width=676&top_left_y=212&top_left_x=1170}
\captionsetup{labelformat=empty}
\caption{FIG. 16. Left: comparison of the two weight functions described in the text for $\gamma=1$. The blue dashed curve is the exponential weight $\exp (-\tau)$ associated with Lorentzian broadening; the solid red curve is the new weight function associated with Eq. (C.3). Right: Fourier transforms of the weight functions. The transform of the exponential weight $e^{-\tau}$ (dashed blue) is $x /\left(1+x^{2}\right)$ while the transform of the weight $h(\tau)=\exp \left(-\tau-\tau^{2} / 2\right)$ is given by Eq. (C.3) (solid red). For comparison, the target function $1 / x$ is shown as well (short dashed green). Equation (C.3) is smoother for small $x$ and approaches $1 / x$ more rapidly at large $x$ than $x /\left(1+x^{2}\right)$.}
\end{figure}
its behavior for small $\tau$ is the same as the $e^{-\tau}$, the associated $F(x)$ will also approach $1 / x$ asymptotically at large $x$. In addition, choosing the ratio of exactly $1 / 2$ between the prefactors of the linear and quadratic parts of the exponential defining $h$ is not arbitrary: this choice of ratio guarantees that $F(x ; \zeta)=1 / x+O\left(1 / x^{5}\right)$ for large $x$ while any other choice $F(x ; \zeta)=1 / x+O\left(1 / x^{3}\right)$. We also note the tranform $F(x)$ can be written, in closed form, in terms of the generalized error function
$$
\begin{equation*}
F(x)=\zeta \operatorname{Im}\left\{\sqrt{\frac{\pi}{2}} e^{-\frac{(x \zeta+i)^{2}}{2}}\left[1+i \operatorname{erfi}\left(\frac{x \zeta+i}{\sqrt{2}}\right)\right]\right\} . \tag{C3}
\end{equation*}
$$

Figure 16 shows a comparison of the two weight functions and their computed Fourier transforms $F(x)$. The weights and nodes for a Gaussian-type quadrature for the weight function, $\exp \left(-\tau-\tau^{2} / 2\right)$, which we term Hermite-Gauss-Laguerre (HGL) quadrature, can be generate using the procedures embodied in the matlab code provided in Appendix H.

It is useful to compare the accuracy of with which the two choices of weight function can be numerically integrated. Table III shows the number of quadrature points required to generate a specified error when using the Lorentzian generating weight $e^{-\tau}$ and improved weight $\exp \left(-\tau-\tau^{2} / 2\right)$ for an energy window of unit width. To generate this table, we specify a maximum percentage error and then find $\gamma$ such that $F(x)$ differs from $1 / x$ by less than the specified error when $x=1$. We then find the size of a quadrature grid $N^{(\tau, \mathrm{HGL})}$ such that the difference between the quadrature approximation of

\begin{table}
\begin{tabular}{c|c|c}
$\%$ error & $N^{(\tau, \mathrm{GL})}\left(w=e^{-\tau}\right)$ & $N^{(\tau, \mathrm{HGL})}\left(w=e^{-\tau-\tau^{2} / 2}\right)$ \\
\hline 5 & 6 & 1 \\
1 & 24 & 1 \\
0.1 & 124 & 5 \\
0.01 & 547 & 15 \\
0.001 & 2216 & 36
\end{tabular}
\captionsetup{labelformat=empty}
\caption{TABLE III. Number of quadrature grid points required to meet the maximum specified percent error for the for the integration of $\exp (i \tau)$ over the two weight functions discussed in this section - the energy window has unit width, i.e., $\gamma x=1$.}
\end{table}

Eq. (35) and the true $F(x)$ is below the same error level for all $x$ in the window (i.e., $0 \leq x \leq 1$ ). It is clear that the new weight function and associated quadrature is at least an order of magnitude more efficient in generating its transfrom than the standard choice $e^{-\tau}$.

\section*{Appendix D: Hermite-Gauss-Laguerre quadrature grid size at fixed error}

A necessary input to the cost function, whose minimization determines optimal window placement, is the number of grid points required to generate a desired error level, $\epsilon^{(q)}$, in the time integrals. Figure 1(b) shows a $2 \times 2$ windowing example containing window pairs with an energy crossing. That is, the sign of the denominator changes for the window pairs $\left\{E_{a, 1}, E_{b, 1}\right\}$ and $\left\{E_{a, 2}, E_{b, 1}\right\}$. In order to treat such pairs, we employ the weight function $h(\tau ; \zeta)=|\zeta| \exp \left(-\tau-\tau^{2} / 2\right)$ and Hermite-Gauss-Laguerre quadrature to discretize the $\tau$ integrals. For all window pairs without energy crossing, the time integrals are discretized using Gauss-Laguerre quadrature and the methodology developed for static $P$ computations; these windows are not considered further. We continue below to develop the tools required to treat windows with energy crossings.

We first seek a quantitative relationship between the number of quadrature points $N^{(\tau, \mathrm{HGL})}$, the energy difference $x=E_{a}-E_{b}$, and the fractional error of the quadrature for the case of energy windows with an energy crossing. The fractional quadrature error is defined as
$$
\begin{equation*}
\epsilon^{(q)}=\frac{\left|F(x)-\sum_{u=1}^{N^{(\tau, \mathrm{HGL})}} w_{u} \sin \left(\tau_{u} x\right)\right|}{|F(x)|} \tag{D1}
\end{equation*}
$$
where, again,
$$
F(x)=\operatorname{Im} \int_{0}^{\infty} d \tau e^{\left(-\tau-\tau^{2} / 2\right)} e^{i \tau x}
$$
and we have standardized the analysis by setting the energy scaling variable to unity ( $|\zeta|=\gamma=1$ ). Here, $F(x)$ is computed to very high accuracy via numerical integration or evaluation of the generalized error function. Figure 17 displays the function $\epsilon^{(q)}\left(x, N^{(\tau, \mathrm{HGL})}\right)$ : due to

\begin{figure}
\includegraphics[max width=\textwidth]{https://cdn.mathpix.com/cropped/59dcea53-65fc-493a-9d14-75ff36f4a405-22.jpg?height=907&width=581&top_left_y=178&top_left_x=1220}
\captionsetup{labelformat=empty}
\caption{FIG. 17. Hermite-Gauss-Laguerre quadrature error as function of $x$ and $N^{(\tau, \mathrm{HGL})}$ (see Eq. (D.1)). In the upper plot, the fractional quadrature error ( $\epsilon^{(q)}$ ) is indicated as blue dots along with the fit function, $\epsilon_{\text {fit }}^{(q)}$ (see Eq. D2). In the lower plot, the contour lines of $\epsilon_{\text {fit }}^{(q)}$ are shown. Each contour line can be represented with high fidelity using only quadratic function of $x$ (see Eq. D3).}
\end{figure}
the presence of the sine function in $\epsilon^{(q)}$, the quadrature error $\epsilon^{(q)}$ is oscillatory as a function of $x$ and finding a simple relationship between $\epsilon^{(q)}, x$ and $N^{(\tau, \mathrm{HGL})}$ is challenging.

We find that the function, $\epsilon_{\text {fit }}^{(q)}$,
$$
\begin{align*}
\epsilon_{\mathrm{fit}}^{(q)} & =\tanh \left(x^{2 N^{(\tau, \mathrm{HGL})}}\right) \times \\
& \exp \left[-\left(1+3.3 N^{(\tau, \mathrm{HGL})}\right) e^{-0.68 x^{2} / N^{(\tau, \mathrm{HGL})}}\right] \tag{D2}
\end{align*}
$$
which is also plotted in Fig. 17, provides a good fit to the data. Direct analytical inversion of Eq. (D2) to obtain $N^{(\tau, \mathrm{HGL})}$ as a function of $x$ and $\epsilon_{\text {fit }}^{(q)}$ is not feasible. However, a good estimate is
$$
\begin{equation*}
N^{(\tau, \mathrm{HGL})}\left(x ; \epsilon^{(q)}\right)=c_{2}\left(\epsilon^{(q)}\right) x^{2}+c_{1}\left(\epsilon^{(q)}\right) x+c_{0}\left(\epsilon^{(q)}\right) \tag{D3}
\end{equation*}
$$
where
$$
\begin{aligned}
& c_{2}=-0.0036 \ln \epsilon^{(q)}+0.11 \\
& c_{1}=-0.0043\left(\ln \epsilon^{(q)}\right)^{2}-0.13 \ln \epsilon^{(q)}+0.54 \\
& c_{0}=-0.204 \ln \epsilon^{(q)}-0.29
\end{aligned}
$$

\begin{figure}
\includegraphics[max width=\textwidth]{https://cdn.mathpix.com/cropped/59dcea53-65fc-493a-9d14-75ff36f4a405-23.jpg?height=906&width=611&top_left_y=197&top_left_x=296}
\captionsetup{labelformat=empty}
\caption{FIG. 18. The optimized cost function to compute the model dynamic $\Sigma^{(\text {model })}(\omega)$ of the text as a function of the number of energy windows for bulk silicon with $\omega=E_{v, \text { min }}$ (valence band minimum energy), 16 atoms and 399 bands. The upper and lower plots show the cost to compute the valence and the conduction band contributions to $\Sigma^{(\text {model })}(\omega)$, respectively. The position of the two minima are ( $N_{v_{w}}=2, N_{p_{w}}=7$ ), upper, and ( $N_{c_{w}}=1, N_{p_{w}}=3$ ), lower.}
\end{figure}

\section*{Appendix E: Treating systems with energy level crossings}

Here, the procedure to determine window placement for cases in which there is an energy crossing (e.g. in the computation of the self-energy $\Sigma(\omega)$ ), is described. In direct analogy with the static $P$ case, we write a cost function with separate energy windows for occupied (valence, $v)$ case $\omega-E_{v}+\omega_{p}$ and the unoccupied (conduction, $c)$ case $\omega-E_{c}-\omega_{p}$ in the band sums for the self-energy. We then allow the number of energy windows $N_{p_{w}}$ and $N_{v_{w}}$ or $N_{c_{w}}$ to range from 1 to 9, and for each such choice ( $N_{p_{w}}, N_{v_{w}}$ ) or ( $N_{p_{w}}, N_{c_{w}}$ ), the computational cost is minimized via a simple gradient descent method. When a window pair has an energy crossing, we simply employ Eq. (D2) to estimate the quadrature size, while for all other window pairs we employ Eq. (A3) to determine the size of the quadrature grid.

For a concrete example, consider the model self-energy
$$
\begin{equation*}
\Sigma^{(\text {model })}(\omega)=\sum_{v p} \frac{1}{\omega-E_{v}+\omega_{p}}+\sum_{c p} \frac{1}{\omega-E_{c}-\omega_{p}} . \tag{E1}
\end{equation*}
$$
using energies and plasmon frequencies from an 8 -atom crystalline Si supercell cell. A total of 32 valence bands,

382 conduction bands and and 425 plasmon modes are employed. The valence band ranges from -0.21 to 0.23 Ha , the conduction band from 0.25 to 2.29 Ha , and the plasmon modes from 0.31 to 45.5 Ha . Selecting $\omega=$ -0.21 , only the valence band sum for $\Sigma(\omega)$ has the signchange requiring the use of HGL quadrature. The conduction contribution to $\Sigma(\omega)$ does not change sign, and we simply utilize GL quadrature for all $\left\{\omega_{p}, E_{c}\right\}$ pairs. In Figure 18, we present the cost function minimized for $81\left\{N_{v_{w}}, N_{p_{w}}\right\}$ pairs (upper) and $81\left\{N_{c_{w}}, N_{p_{w}}\right\}$ pairs (lower) at error, $\epsilon^{(q)}=0.01$. For the valence band sum, the optimal number of windows is ( $N_{v_{w}}=2, N_{p_{w}}=7$ ), while for the conduction band, the optimal number of windows is ( $N_{c_{w}}=1, N_{p_{w}}=3$ ) (i.e., the position of the minimum in the upper and lower curves of Fig. 18, respectively).

\section*{Appendix F: Interpolation method}

\section*{1. Theory}

In real space, the static random phase approximation (RPA) irreducible polarizability matrix is
$$
\begin{equation*}
P_{r, r^{\prime}}=-2 \sum_{v}^{N_{v}} \sum_{c}^{N_{c}} \frac{\psi_{r, v}^{*} \psi_{r, c} \psi_{r^{\prime}, c}^{*} \psi_{r^{\prime}, v}}{E_{c}-E_{v}} \tag{F1}
\end{equation*}
$$

One advantage of working in a real-space basis is that the sum over products of wave functions is separable so one can come up with cubic scaling algorithms if one can make separable approximations to the energy denominator. We begin by rewriting $P$ as
$$
P_{r, r^{\prime}}=-2 \sum_{v} \psi_{r, v}^{*} A\left(E_{v}\right)_{r, r^{\prime}} \psi_{r^{\prime}, v}
$$
where the matrix $A$ is defined as
$$
A(z)_{r, r^{\prime}}=\sum_{c} \psi_{r, c} \psi_{r^{\prime}, c}^{*} /\left(E_{c}-z\right)
$$

For a system with an energy gap $E^{(\text {gap })}$, the denominator $E_{c}-E_{v}$ is always positive with a minimum value of the gap $E^{\text {(gap) }}$. Furthermore, the matrix $A$ must be evaluated only for energies $z$ within the range of valence band energies $E_{v}$. Hence, the calculation of $P$ uses $A(z)$ for values of $z$ where it is smooth in $z$. This means we can use interpolation: we first tabulate $A(z)$ for a range of $z$ values ranging over the valence band energies. This tabulation costs $N_{z} N_{c} N_{r}^{2}$ which is cubic since the valence bandwidth is an intensive quantity and the number of points $N_{z}$ needed for a fixed accuracy is a fixed, intensive number. Next, to compute $P$, we sum over $v$, and for each $E_{v}$ we interpolate $A$ to that energy by using the tabulated $A$. This calculation is also cubic and costs $N_{i} N_{v} N_{r}^{2}$ where $N_{i} \leq N_{z}$ is the number of tabulated $z$ values needed for interpolation (e.g. $N_{i}=2$ for linear interpolation).

An efficient interpolation scheme should require a small number of $z$-points $N_{z}$ as well as a modest interpolation cost $N_{i}$. In our case, the energy dependence requiring interpolation is given by $1 /\left(E_{c}-z\right)$ which is most rapidly changing for the largest values of $z$ near the top of the valence $E_{v}^{\text {(max) }}$ band and when $E_{c}$ takes on its smallest value at the conduction band minimum $E_{c}^{\text {(min) }}$. Hence, an efficient interpolation scheme will use a non-uniform $z$ grid that appropriately concentrates sampling points near $E_{v}^{(\text {max })}$.

The next section below describes the approach we use to find optimal interpolation grids $z_{j}$ for the case of linear interpolation (i.e., two-point nearest neighbor interpolation with $N_{i}=2$ ) when sampling over the entire range of valence band energies. We note higher order interpolation schemes with $N_{i}>2$ can be used as well that will reduce the number of grid points needed for a fixed error but require more work to perform the interpolation. In our experience, the higher order interpolations do not in the end improve performance at the same level of error when compared to the simpler linear interpolation method.

Regardless of the precise interpolation scheme used, all such interpolation methods will have errors that decrease as a power of the number of grid points, $n$. As the data presented in the main text shows, the Fourier-Laplace transform based methods turn out to have superior error properties (their errors fall off exponentially in $n$ ).

\section*{2. Energy grids for interpolation}

The function of $z$ that we wish to interpolate over $z$ is
$$
A(z)_{r, r^{\prime}}=\sum_{c}^{N_{c}} \frac{\psi_{r, c} \psi_{r^{\prime}, c}^{*}}{E_{c}-z}
$$

The function is steepest versus $z$ close to the top of the valence band $E_{v}^{\text {(max) }}$ when the energy difference in the denominator is small. In fact, we will consider the worse case scenario and focus on the stiffest and steepest term in the entire sum which is for the case $E_{c}=E_{c}^{(\mathrm{min})}$, the conduction band minimum energy. Hence the most difficult to interpolate term is given by the dimensionless function
$$
f(z)=\frac{E_{\text {gap }}}{E_{c}^{(\min )}-z} \equiv \frac{1}{1+x}
$$
where $z=E_{v}^{(\max )}-x E^{(\text {gap })}$, and the scaled energy variable $x$ satisfies $\left.0 \leq x \leq\left(E_{v}^{(\max )}-E_{v}^{(\min }\right)\right) / E^{(\text {gap })}$.

The question is how to pick a grid of $\left\{x_{j}\right\}$ values with $n$ points where $x_{1}=0$ and $x_{n}=\left(E_{v}^{(\text {max })}-E_{v}^{(\text {min })}\right) / E^{(\text {gap })}$. For simplicity, we will be using linear interpolation, so that given some $x$ between two grid points $x_{j} \leq x \leq x_{j+1}$, the linear interpolation is $f^{l}(x)=\left[f\left(x_{j}\right)\left(x_{j+1}-\right.\right.$
$\left.x)+f\left(x_{j+1}\right)\left(x-x_{j}\right)\right] / \Delta x_{j}$ where $\Delta x_{j}=x_{j+1}-x_{j}$. Calculus then provides an analytical expression for the maximum error $f^{l}(x)-f(x)$ in the interval $x_{j} \leq x \leq x_{j+1}$. For large $n$ and thus small spacings $\Delta x_{j}$, the lowest order term for the error is
$$
\left(f^{I}-f\right)_{\max } \approx \frac{\left(\Delta x_{j}\right)^{2}}{4\left(1+x_{j}\right)^{3}}
$$

We wish to bound this error by a fixed fractional error tolerance, $\epsilon^{(q)}$, for all $j$,
$$
\begin{equation*}
\frac{\left(\Delta x_{j}\right)^{2}}{4\left(1+x_{j}\right)^{3}} \leq \epsilon^{(q)} \tag{F2}
\end{equation*}
$$
which then in principle determines the grid points $x_{j}$. In practice, exact solution of this equation is very difficult, so we again appeal to the large $n$ limit where $x_{j}$ can be viewed as a function $x(j)$ of a continuous argument $j$ so we approximate $\Delta x_{j} \approx d x / d j$. Then Eq. (F2) turns into an ordinary differential equation with specified boundary conditions. The solution is
$$
x(j)=\frac{1}{\left(1-(j-1) \sqrt{\epsilon^{(q)}}\right)^{2}}-1
$$

Since $x(n)=\left(E_{v}^{(\max )}-E_{v}^{(\min )}\right) / E^{(\text {gap })}$ is known, this determines $n$ for each $\epsilon^{(q)}$. And finally we have $z_{j}= E_{v}^{(\text {max })}-x_{j} E^{(\text {gap })}$.

The above choice of grid bounds the error when evaluating the function once. However, when using the interpolation to compute $P$ from $A$, we will be evaluating the interpolation over many values across the valence band which approximate an integral. Hence, a more appropriate error control scheme will not only consider the error in interpolating $f(x)$ but also the fact that narrower intervals of $x$ will be sampled less often (assuming a smooth and roughly flat density of states). Hence we should instead bound the error in the function times the size of the interval:
$$
\Delta x_{j} \times \frac{\left(\Delta x_{j}\right)^{2}}{4\left(1+x_{j}\right)^{3}} \leq \epsilon^{(q)}
$$

Repeating the above exercise, the grid appropriate to this error bound is given by
$$
\begin{equation*}
x(j)=\exp \left(\left[4 \epsilon^{(q)}\right]^{1 / 3}(j-1)\right)-1 \tag{F3}
\end{equation*}
$$

As before, the fixed value of $x(n)$ then determines $n$ at fixed $\epsilon^{(q)}$, and we use the $x_{j}$ to get the energy grid points $z_{j}$. The results in the main text are based on use of this second (exponential) grid of Eq. (F3).

\section*{Appendix G: Details of KS-DFT and GW computations}

We have performed DFT simulations to obtain the single particle wave functions and energies employed as input to the GW calculations reported in the main text.

The plane-wave, non-local, norm-conserving pseudopotential, supercell approach was employed as implemented in the Quantum Espresso software application ${ }^{53}$.

To study cystalline Si , we employed the local density approximation (LDA) for exchange and correlation as parameterized by Perdew and Zunger ${ }^{54}$. The normconserving pseudopotential for Si was generated with the valence configuration of $3 s^{2} 3 p^{2} 3 d^{0}$ with cutoff radii of 1.75, 1.93, and 2.07 a.u. for $s, p$, and $d$ channels, respectively. The plane wave cutoff was taken to be 25 Ry , and the lattice parameter was set to the experimental value of $5.43 \AA$.

To study cystalline MgO , the GGA-PBE exchangecorrelation functional ${ }^{55}$ was employed. Both Mg and O were represented by norm-conserving pseudopotentials generated with valence configuration $3 \mathrm{~s}^{2}$ and $2 \mathrm{~s}^{2} 2 \mathrm{p}^{4} 3 \mathrm{~d}^{0} 4 \mathrm{f}^{0}$ for Mg and O , respectively. The plane wave cutoff was taken to be 50 Ry , and the lattice parameter was set to $8.42 \AA$.

To generate $\epsilon_{\infty}$ and the COHSEX band gap for crystalline Si and MgO , we sampled the $\Gamma$-point of the BZ in a 16 atom supercell for both cases. The reference $\mathrm{G}_{0} \mathrm{~W}_{0}$ prediction of the band gap of Si was obtained using a $4 \times 4 \times 4$ sampling of the primitive cell - equivalent to a $2 \times 2 \times 2$ sampling of the 16 atom supercell. The total number of bands in the 16 atom supercell was taken to be 399 and 433 for Si and MgO , respectively. For the CTSPW results, $\left\{N_{v_{w}}=1, N_{c_{w}}=4\right\}$ and $\left\{N_{v_{w}}=1, N_{c_{w}}=4\right\}$ was employed to treat both MgO and Si .

To create the data on computational load versus the number of atoms (Fig. 8 of the main text), we studied Si with the following $k$-point sampling and bands: 52 bands and $8 k$-points for the 2 -atom cell, 104 bands with $4 k$ points for the 4 -atom cell, 208 bands with $2 k$-points for the 8 -atom cell, and 416 bands with $1 k$-point for 16 -atom cell. For the CTSP-W method, $\left\{N_{v_{w}}=1, N_{c_{w}}=5\right\}$ were employed for all simulations.

To study crystalline Al , we employed the LDA for exchange and correlation as parameterized by Perdew and Zunger ${ }^{54}$. The plane wave cutoff was taken to be 50 Ry, and the lattice parameter was set to $3.99 \AA$. To obtain $P_{0,0}$, we employed a 16 atom supercell, sampled $2 k$-points and included a total 400 bands. Gaussian smearing was used to represent the occupation numbers with $\beta^{-1}=0.03 \mathrm{Ry}$. For the CTSP-W results, $\left\{N_{v_{w}}=1, N_{c_{w}}=7\right\}$ was employed in all cases.

\section*{Appendix H: Hermite-Gauss-Laguerre Quadrature}

The nodes and weights for the Hermite-GaussLaguerre (HGL) quadrature described in the main text can be obtained by employing the matlab functions provided below:
```
function [x,w]=GLquad(n)
% function [x,w]=GLagIntP(n)
% Gauss-Laguerre integration: return nodes x
```

```
% and weights w for a
% quadrature grid with n points
% This is basically the Golub-Welsch method
J=diag(1:2:2*n-1)+diag(1:n-1,1)+diag(1:n-1,-1);
[v,l]=eig(J);
[x,ix]=sort(diag(l));
w=v(1,ix)'.^2;
return
function [xmat,wmat] = myweightquad(n)
%function [xmat,wmat] = myweightquad(n)
% Return all nodes (xmat) and weights (wmat)
% for quadratures up to % n points for weight
% w(x)=exp(-x-x^2/2). These are organized in
% matrices. xmat are the nodes and wmat
% are the weights. Each column is for a
% quadrature size going from
% 1 to n (left to right). Thus the lower
% triangle is padded with zeros.
% Figure out number of grid points
% so that the biggest moment (2n)
% is well converged. We do
% Gauss-Laguerre quadrature to
% do these integrals over the weights!
Iold = 0;
for nx=round(10.^[1:.2:7])
    [xq,wq] = \rrGLquad(nx);
    weight = exp(-xq.^2/2);
    I = sum(wq.*weight.*xq.^(2*n));
    if Iold>0
        err = (I-Iold)/I;
        if abs(err)<1e-14
            break
        end
    else
    end
    Iold = I;
end
% Build polynomials as we go
% and figure out the recursion
% relation coefficients as we go
p = zeros(length(xq),n+1);
p(:,1) = 1;
a = zeros(n,1);
b = zeros(n,1);
for j=1:n
    xpp = sum(wq.*xq.*weight.*p(:,j).^2);
    pp = sum(wq.*weight.*p(:,j).^2);
    a(j) = xpp/pp;
    if j>1
        ppm1 = sum(wq.*weight.*p(:,j-1).^2);
        b(j) = pp/ppm1;
    end
    if j>1
        p(:,j+1) = ...
```

```
            (xq-a(j)).*p(:,j)-b(j)*p(:,j-1);
    else
        p(:,j+1) = (xq-a(j)).*p(:,j);
    end
end
% Prepare for Golub-Welsch
b = b(2:end);
b = sqrt(b);
mu0 = sum(wq.*weight);
% Build Golub-Welsch J matrix,
% eigen decompose it, and get weights and
% nodes for each value of j=1,...,n
% (i.e., all weights and nodes for
% quadratures up to size n)
J = diag(a) + diag(b,1) + diag(b,-1);
xmat = zeros(n,n);
```

```
wmat = zeros(n,n);
for j=1:n
    Jcut = J(1:j,1:j);
    [v,d] = eig(Jcut);
    d = diag(d);
    [^,is] = sort(d);
    d = d(is);
    v = v(:,is);
    x = d;
    w = v(1,:).^2*mu0;
    w = w’;
    xmat(:,j) = [x' zeros(1,n-j)]';
    wmat(:,j) = [w' zeros(1,n-j)]';
end
return
```

* sohrab.ismail-beigi@yale.edu
${ }^{1}$ P. Hohenberg and W. Kohn, Physical Review 136, B864 (1964).
${ }^{2}$ W. Kohn and L. J. Sham, Physical Review 140, A1133 (1965).
${ }^{3}$ J. P. Perdew and A. Zunger, Physical Review B 23, 5048 (1981).
${ }^{4}$ J. P. Perdew, J. A. Chevary, S. H. Vosko, K. A. Jackson, M. R. Pederson, D. J. Singh, and C. Fiolhais, Physical Review B 46, 6671 (1992).
${ }^{5}$ J. P. Perdew, R. G. Parr, M. Levy, and J. L. Balduz, Physical Review Letters 49, 1691 (1982).
${ }^{6}$ S. Lundqvist and N. H. March, Theory of the Inhomogeneous Electron Gas (Springer, 2013).
${ }^{7}$ V. I. Anisimov, F. Aryasetiawan, and A. I. Lichtenstein, Journal of Physics: Condensed Matter 9, 767 (1997).
${ }^{8}$ L. Hedin, Physical Review 139, A796 (1965).
${ }^{9}$ M. S. Hybertsen and S. G. Louie, Physical Review B 34, 5390 (1986).
${ }^{10}$ F. Aryasetiawan and O. Gunnarsson, Reports on Progress in Physics 61, 237 (1998).
${ }^{11}$ G. Onida, L. Reining, and A. Rubio, Reviews of Modern Physics 74, 601 (2002).
${ }^{12}$ H. F. Wilson, F. Gygi, and G. Galli, Physical Review B (Condensed Matter and Materials Physics) 78, 113303 (2008).
${ }^{13}$ H. F. Wilson, D. Lu, F. Gygi, and G. Galli, Physical Review B 79, 245106 (2009).
${ }^{14}$ D. Rocca, D. Lu, and G. Galli, The Journal of Chemical Physics 133, 164109 (2010).
${ }^{15}$ D. Lu, F. Gygi, and G. Galli, Physical Review Letters 100, 147601 (2008).
${ }^{16}$ F. Giustino, M. L. Cohen, and S. G. Louie, Physical Review B 81, 115105 (2010).
${ }^{17}$ P. Umari, G. Stenuit, and S. Baroni, Physical Review B 81, 115104 (2010).
${ }^{18}$ M. Govoni and G. Galli, Journal of Chemical Theory and Computation 11, 2680 (2015).
${ }^{19}$ F. Bruneval and X. Gonze, Physical Review B 78, 085125 (2008).
${ }^{20}$ J. A. Berger, L. Reining, and F. Sottile, Physical Review B 82, 041103 (2010).
${ }^{21}$ W. Gao, W. Xia, X. Gao, and P. Zhang, Scientific Reports 6, 36849 (2016).
${ }^{22}$ D. Foerster, P. Koval, and D. Sanchez-Portal, The Journal of Chemical Physics 135, 074105 (2011).
${ }^{23}$ P. Liu, M. Kaltak, J. Klimeš, and G. Kresse, Physical Review B 94, 165109 (2016).
${ }^{24}$ D. Neuhauser, Y. Gao, C. Arntsen, C. Karshenas, E. Rabani, and R. Baer, Physical Review Letters 113, 076402 (2014).
${ }^{25}$ J. C. Light and T. Carrington, "Discrete-variable representations and their utilization," in Advances in Chemical Physics (John Wiley and Sons, Inc., 2007) pp. 263-310.
${ }^{26}$ J. Deslippe, G. Samsonidze, D. A. Strubbe, M. Jain, M. L. Cohen, and S. G. Louie, Computer Physics Communications 183, 1269 (2012).
${ }^{27}$ M. Kim, S. Mandal, E. Mikida, K. Chandrasekar, E. Bohm, N. Jain, Q. Li, R. Kanakagiri, G. J. Martyna, L. Kale, and S. Ismail-Beigi, Computer Physics Communications (2019), https://doi.org/10.1016/j.cpc.2019.05.020.
${ }^{28}$ J. W. Negele and H. Orland, Quantum Many-particle Systems (Westview Press, 1998).
${ }^{29}$ L. Hedin and S. Lundqvist, in Advances in Research and Applications, Vol. Volume 23 (Academic Press, 1970) pp. 1-181.
${ }^{30}$ M. M. Rieger, L. Steinbeck, I. D. White, H. N. Rojas, and R. W. Godby, Computer Physics Communications 117, 211 (1999).
${ }^{31}$ M. Kaltak, J. Klimeš, and G. Kresse, Journal of Chemical Theory and Computation 10, 2498 (2014).
${ }^{32}$ D. Baye and P.-H. Heenen, Journal of Physics A: Mathematical and General 19, 2041 (1986).
${ }^{33}$ R. A. Friesner, The Journal of Chemical Physics 85, 1462 (1986).

34 " $s^{-1}=\int_{0}^{\infty} d t \exp (-s t)=\zeta \int_{0}^{\infty} d t \exp (-\zeta s t)$, ".
${ }^{35}$ M. S. Hybertsen and S. G. Louie, Physical Review B 34, 5390 (1986).
${ }^{36}$ J. Deslippe, G. Samsonidze, D. A. Strubbe, M. Jain, M. L.

Cohen, and S. G. Louie, Computer Physics Communications 183, 1269 (2012).
37 J. Williamson, Lebesgue Integration: Dover Books on Mathematics (Dover, 2014).
${ }^{38}$ M. Abramowitz and Stegun, eds., Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables, 10th ed. (U.S. Government Printing Office, 1972).
${ }^{39}$ C. L. Fu and K. M. Ho, Physical Review B 28, 5480 (1983).
${ }^{40}$ R. J. Needs, R. M. Martin, and O. H. Nielsen, Physical Review B 33, 3778 (1986).
${ }^{41}$ M. J. Gillan, Journal of Physics: Condensed Matter 1, 689 (1989).
${ }^{42}$ To avoid excessive memory use, one can compute the large matrix $\Sigma(\omega)_{r, r^{\prime}}$ for a fixed $\omega$ and then compute and only store the much smaller number of desired matrix elements $<n|\Sigma(\omega)| n^{\prime}>$ before moving to the next $\omega$ value.
${ }^{43}$ A. Gil, J. Segura, and N. M. Temme, Numerical Methods for Special Functions (SIAM, 2007).
${ }^{44}$ For systems with a small number of atoms, the CTSPW runs slightly slower per operation, $<2 \times$, due to the inefficient caching and pipelining of our untuned software.
${ }^{45}$ S. Zhang, C. I. Pelligra, G. Keskar, J. Jiang, P. W. Majewski, A. D. Taylor, S. Ismail-Beigi, L. D. Pfefferle, and
C. O. Osuji, Advanced Materials 24, 82 (2012).
${ }^{46}$ https://bluewaters.ncsa.illinois.edu/.
${ }^{47}$ F. Bruneval and X. Gonze, Physical Review Letters 78, 085125 (2008).
${ }^{48}$ D. Foerster, P. Koval, and D. Sanchez-Portal, Journal of Chemical Physics 135, 074105 (2011).
${ }^{49}$ P. Liu, M. Kaltak, J. Klimes, and G. Kresse, Physical Review B 94, 165109 (2016).
${ }^{50}$ K. S. D. Beach, R. J. Gooding, and F. Marsiglio, Physical Review B 61, 5147 (2000).
${ }^{51}$ S. Goedecker and L. Colombo, Physical Review Letters 73, 122 (1994).
${ }^{52}$ S. Goedecker, Reviews of Modern Physics 71, 1085 (1999).
${ }^{53}$ P. Giannozzi, S. Baroni, N. Bonini, M. Calandra, R. Car, C. Cavazzoni, D. Ceresoli, G. L. Chiarotti, M. Cococcioni, and I. Dabo, Journal of Physics: Condensed Matter 21, 395502 (2009).
54 J. P. Perdew and A. Zunger, Physical Review B 23, 5048 (1981).

55 J. P. Perdew, K. Burke, and M. Ernzerhof, Physical Review Letters 77, 3865 (1996).