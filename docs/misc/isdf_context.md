### ISDF - real space GW

The ISDF + GW method allows us to calculate the COHSEX and full-frequency quasiparticle energies of a material.

This package reads DFT wavefunctions in a plane wave basis from WFN.h5 according to the wfnreader.py interface, 
and works through the following steps to obtain quasiparticle energies:

1. The charge density of the valence + some conduction bands is calculated by get_charge_density.py, by FFT-ing
wavefunctions from G space to r space, squaring and summing them. It is saved to charge_density.h5.

2. ISDF interpolation points {r_mu} are calculated by the centroidal voronoi tesselation algorithm using the 
charge density as the functional weight. This is done in kmeans_isdf.py using a k-means clustering algorithm 
in A kmeans++ initialization is used, and the algorithm is run until convergence, when points are saved to 
centroids_frac.h5.

3. The main driver script is currently cohsex_isdf.py. First, the code calculates the Coulomb interaction v_q(G)
using BerkeleyGW conventions (analytical truncated form in 2D, and the divergent q=G=0 component is obtained by a 
monte carlo integration over its voronoi cell), and IFFT's the wavefunctions in the relevant band range to real 
space. A loop over each q-point in the Brillouin zone is performed to obtain the ISDF interpolation vectors 
zeta_q,mu(r) that minimize least squared errors in the expression:

rho_q(mnk,r) = psi\*_mk-q(r) psi_nk(r) = \sum_mu zeta_q,mu(r) psi\*_mk-q(r_mu) psi_nk(r_mu),

for all points r in the unit cell and m and n in the range of interest. As the wavefunctions are spinor 
wavefunctions, the exact expression used is actually:

rho_q(msns'k,r) = psi\*_mk-q,s(r) psi_nk,s'(r) = \sum_mu zeta_q,mu(r) psi\*_mk-q,s(r_mu) psi_nk,s'(r_mu),

where s and s' are the spinor indices, but zeta_q is spin-independent For the following few lines, m and n
will be shorthand for the combination indices ms and ns'. 

The array containing the psi_nk is repeatedly referred to as psi_r in the code, and psi_mk-q as psi_l. The 
least squares problem can actually be solved without ever constructing the extremely large matrix rho_q(mnk,r), 
which can be many TB, using:

P_k(r,r_mu) = \sum_n psi_nk(r) psi_nk(r_mu), P_muk(r_mu,r_nu) = \sum_n psi_nk(r_mu) psi_nk(r_nu)
Z_q(r_mu, r) = \sum_k P\*_k-q(r_mu,r) P_k(r_mu,r)
C_q(r_mu, r_nu) = \sum_k P\*_k-q(r_mu,r_nu) P_k(r_mu,r_nu)
Solve C_q zeta_q = Z_q by least squares.

After zeta_q,mu(r) is found, the code calculates the matrix elements <zeta_q,mu|V|zeta_q,nu> in G space by
FFT-ing the cell-periodic parts z_q,mu(r)=exp(-iqr)zeta_q,mu(r) and doing \sum_G z\*_q,mu(G) v_q(G) z_q,nu(G).
This is done in the q-loop so that the set of zeta_q,mu(r) for all q is never held in memory at once.

4. The code then calculates the Green's function G_k(s,r_mu,s',r_nu,t), which I will henceforth refer to
in a shorthand representing the order of the indices, G[t,kx,ky,kz,s,r,s',r']. 

G[t,kx,ky,kz,s,r,s',r'] = step(-t) sum_(n occupied) psi\*_nk,s(r_mu) psi_nk,s'(r_nu) * exp(-E_nk t) - step(t) sum_(m unoccupied) psi\*_mk,s(r_mu) psi_mk,s'(r_nu) * exp(-E_mk t).

Both energies are defined with respect to the Fermi level, and the step functions are s.t. step(0)=1. For
the COHSEX approximation, we use the static G[t=0]. Both a G_occ and a G_full are required for different 
contributions to the self energy, where G_full sums over all occupied and unoccupied bands.

5. Gk and Vq are IFFT'ed to GR and VR representations, by reordering so the kgrid dimensions are last:

G[t,kx,ky,kz,s,r,s',r'] => G[t,s,r,s',r',kx,ky,kz] => ...Rx,Ry,Rz
V[:,kx,ky,kz,:,r,:,r] => V[:,:,r,:,r',qx,qy,qz] => ...Rx,Ry,Rz

After the IFFT, the "exchange component of the self energy" sigmaX is given by the element-wise product:

sigmaX[t,s,r,s',r',Rx,Ry,Rz] = GR[t,s,r,s',r',Rx,Ry,Rz] * VR[:,:,r,:,r',Rx,Ry,Rz]

6. The matrix elements <mk|sigmaX|nk> are then calculated by FFT'ing sigmaX from R->k space and contracting 
with the wavefunction coefficients:

sigmaX[t,s,r,s',r',Rx,Ry,Rz] => sigmaX[t,Rx,Ry,Rz,s,r,s',r'] => kx,ky,kz
SigmaX[t,i,j,kx,ky,kz] = \sum_s,s',r_mu,r_nu psi\*_mk,s(r_mu) sigmaX[t,kx,ky,kz,s,r_mu,s',r_nu] psi_nk,s'(r_nu)

7. Though not implemented currently, there will be a transform from sigmaX[t] to sigmaX[w] by a sine/cosine 
transform procedure.

8. This is not the next step chronologically, but to get the zero-freq. screened interaction, we compute 
Chi[t,r,r',Rx,Ry,Rz] = \sum_ab G[t,Rx,Ry,Rz,a,r,b,r'] G[-t,-Rx,-Ry,-Rz,b,r',a,r]
(the formula can be interpreted as prob. of an electron propagating from ra->R+r'b and back. Since G has a step
function so that for t>0, G is a sum over valence bands and for t<0 it's a sum over conduction bands (Temp=0), we actually write the zero frequency Chi as G_occ * G_unocc, with no energy 
We then do the Fourier transform:
Chi[t,r,r',Rx,Ry,Rz] =>


