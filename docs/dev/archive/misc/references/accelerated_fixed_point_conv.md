\title{
ON THE CONVERGENCE OF CROP-ANDERSON ACCELERATION METHOD *
}

\author{
NING WAN ${ }^{\dagger}$ AND AGNIESZKA MIĘDLAR ${ }^{\dagger}$
}

\begin{abstract}
Anderson Acceleration is a well-established method that allows to speed up or encourage convergence of fixed-point iterations. It has been successfully used in a variety of applications, in particular within the Self-Consistent Field (SCF) iteration method for quantum chemistry and physics computations. In recent years, the Conjugate Residual with OPtimal trial vectors (CROP) algorithm was introduced and shown to have a better performance than the classical Anderson Acceleration with less storage needed. This paper aims to delve into the intricate connections between the classical Anderson Acceleration method and the CROP algorithm. Our objectives include a comprehensive study of their convergence properties, explaining the underlying relationships, and substantiating our findings through some numerical examples. Through this exploration, we contribute valuable insights that can enhance the understanding and application of acceleration methods in practical computations, as well as the developments of new and more efficient acceleration schemes.
\end{abstract}

Key words. fixed-point iteration, self-consistent field iteration, acceleration method, Anderson Acceleration, CROP

MSC codes. 65B05, 65B99, 65F10, 65H10
1. Introduction. Consider the following problem: Given a function $g: \mathbb{C}^{n} \rightarrow \mathbb{C}^{n}$ find $x \in \mathbb{C}^{n}$ such that
$$
\begin{equation*}
x=g(x), \quad \text { or alternatively } \quad f(x)=0 \tag{1.1}
\end{equation*}
$$
with $f(x):=g(x)-x$. Obviously, a simplest method of choice to solve this problem is the fixed-point iteration
$$
\begin{equation*}
x^{(k+1)}=g\left(x^{(k)}\right), \text { for all } k \in \mathbb{N} . \tag{1.2}
\end{equation*}
$$

Unfortunately, its convergence is often extremely slow.
Remark 1.1. Note that in the case of a fixed-point problem (1.1) with iteration function $g(x)$, the associated residual (error) function $f(x)$ is defined as $f(x):= g(x)-x$. However, this choice is not in any sense universal. If the problem of interest has a specific residual (error) function associated with it, then it will usually be used to define $f(x)$. For example, in the case of C-DIIS [35, 36] method for solving the HartreeFock equations, the density matrix $D$ is updated until it commutes with the associated Fock matrix $F(D)$, i.e., $F(D) D-D F(D)=0$. The new iterates $D^{(k+1)}=g\left(D^{(k)}\right)$ are computed by Roothaan SCF process. Hence, the residual (error) function $f(D)$ is defined as the commutator $f(D)=F(D) D-D F(F)$ instead of the difference between two consecutive iterates.

The problem of slow (or no) convergence of a sequence of iterates has been extensively studied by researchers since the early 20th century. Aitken's delta-squared process was introduced in 1926 [1] for nonlinear sequences, and since then, people have been investigating various extrapolation and convergence acceleration methods with Shanks transformation [40,7] providing one of the most important and fundamental ideas. Introduced as a generalization of the Aitken's delta-squared process, it laid the

\footnotetext{
* Funding: This work was supported by the National Science Foundation through the NSF CAREER Award DMS-2144181 and DMS-2324958.
${ }^{\dagger}$ Department of Mathematics, Virginia Tech, Blacksburg, VA (wning@vt.edu, amiedlar@vt.edu)
}
foundations for many acceleration schemes including the $\varepsilon$-algorithm. Some notable acceleration methods, including but not limited to, are: $\varepsilon$-algorithm, which contains scalar $\epsilon$-algorithm (SEA) [47], vector $\epsilon$-algorithm (VEA) [48], topological $\epsilon$-algorithm (TEA)[5], simplified TEA (STEA)[8]; polynomial methods, which contains minimal polynomial extrapalolation (MPE)[11], reduced rank extrapolation (RRE)[20], modified minimal polynomial extrapolation (MMPE)[41]. For further reading about extrapolation and acceleration methods, see $[27,6,10,9,7]$.

In the following, we will consider two mixing acceleration methods: the Anderson Acceleration [2, 3] (also referred to as Pulay mixing [35, 36] in computational chemistry) and the Conjugate Residual algorithm with OPtimal trial vectors (CROP) [49, 22]. The CROP method, introduced in [49], is a generalization of the Conjugate Residual (CR) method [39, Section 6.8], which is a well-known iterative algorithm for solving linear systems. Starting with investigating broad connections between Anderson Acceleration and CROP algorithm, our goal is to understand the convergence behavior of CROP algorithm and find out for when it may serve as an alternative to the well-established Anderson Acceleration method.

Contributions and Outline. In this paper, we discuss the connection between CROP algorithm and some other well-known methods, analyze its equivalence with Anderson Acceleration method and investigate convergence for linear and nonlinear problems. The specific contributions and novelties of this work are as follows:
- We summarize CROP algorithm and establish a unified Anderson-type framework and show the equivalence between Anderson Acceleration method and CROP algorithm.
- We compare CROP algorithm with some Krylov subspace methods for linear problems and with multisecant methods in the general case.
- We illustrate the connection between CROP algorithm and Anderson Acceleration method and explain the CROP-Anderson variant.
- We investigate the situations in which CROP and CROP-Anderson algorithms work better than Anderson Acceleration method.
- We derive the convergence results for CROP and CROP-Anderson algorithms for linear and nonlinear problems.
- We extend CROP and CROP-Anderson algorithms to rCROP and rCROPAnderson, respectively, by incorporating real residuals to make them work better for nonlinear problems.
Previous work. Anderson Acceleration method has a long history in mathematics literature, which goes back to Anderson's 1965 seminal paper [2]. Over the years, the method has been successfully applied to many challenging problems [12, 13, 28, 14]. An independent line of research on accelerating convergence of nonlinear solvers established by physicists and chemists has led to developments of techniques such as Pulay mixing [35, 36], also known as the Direct Inversion of the Iterative Subspace (DIIS) algorithm, which is instrumental in accelerating the self-consistent field iteration method in electronic structure calculations [29].

It is well-known that Anderson Acceleration method has connections with the Generalized Minimal Residual Method (GMRES) algorithm [39, Section 6.5] and is categorized as a multisecant method [44, 38, 24, 25]. The first convergence theory for Anderson Acceleration, under the assumption of a contraction mapping, appears in [43]. The convergence of Anderson(1), a topic of particular interest to many researchers, is discussed separately in works such as [37, 18, 19]. Various variants of Anderson Acceleration are explored in [4, 42, 17, 46, 16]. The acceleration properties of Anderson Acceleration are theoretically justified in [17, 23, 32, 34, 37]. Addition-
ally, discussions on the depth parameter can be found in [37, 15], while the damping parameter is examined in [32, 33, 46]. We further refer readers to [27, 6, 10, 9, 7] and references therein for detailed and more comprehensive presentation of history, theoretical and practical results on the acceleration methods and their applications.

The paper is organized as follows: Section 1 briefly explains the general idea of acceleration methods, provides historical context and presents the state-of-the-art results relevant to our studies. In Section 2 we establish necessary notation and review some background material on Anderson Acceleration method and CROP algorithm. We propose a novel unified framework that allows us to illustrate explicitly the connection between the two approaches, including the role of various parameters, and perform their theoretical analysis in Section 3. We present convergence analysis of CROP algorithm and discuss its truncated variants in Section 5. Finally, in Section 6, we present some numerical experiments to highlight the main results.
2. Background. In this section, we will collect some essential background information on Anderson Acceleration and CROP methods. In the discussion below, we use subscripts to denote iterates associated with a particular method, e.g., $x_{A}$ amd $x_{C}$ indicate Anderson and CROP iterates, respectively. The quantities with no subscript indicate any of the above methods. Throughout the paper, we will talk about equivalence of various methods, by which we mean that under additional assumptions the iterates of either algorithm can be obtained directly from the iterates of the other.
2.1. Anderson Acceleration. Given Anderson iterates $x_{A}^{(k)}, k=0,1, \ldots$ and corresponding residual (error) vectors, e.g., $f_{A}^{(k)}:=g\left(x_{A}^{(k)}\right)-x_{A}^{(k)}$, consider weighted averages of the prior iterates, i.e.,
$$
\begin{equation*}
\bar{x}_{A}^{(k)}:=\sum_{i=0}^{m_{A}^{(k)}} \alpha_{A, i}^{(k)} x_{A}^{\left(k-m_{A}^{(k)}+i\right)} \quad \text { and } \quad \bar{f}_{A}^{(k)}:=\sum_{i=0}^{m_{A}^{(k)}} \alpha_{A, i}^{(k)} f_{A}^{\left(k-m_{A}^{(k)}+i\right)} \tag{2.1}
\end{equation*}
$$
with weights $\alpha_{A, 0}^{(k)}, \ldots, \alpha_{A, m_{A}^{(k)}}^{(k)} \in \mathbb{R}$ satisfying $\sum_{i=0}^{m_{A}^{(k)}} \alpha_{A, i}^{(k)}=1$, a fixed depth (history or window size) parameter $m$ and a truncation parameter $m_{A}^{(k)}:=\min \{m, k\}$. Note that (2.1) can be written in the equivalent matrix form, i.e.,
$$
\begin{equation*}
\bar{x}_{A}^{(k)}:=X_{A}^{(k)} \alpha_{A}^{(k)} \quad \text { and } \quad \bar{f}_{A}^{(k)}:=F_{A}^{(k)} \alpha_{A}^{(k)} \tag{2.2}
\end{equation*}
$$
with $\mathbb{R}^{n \times\left(m_{A}^{(k)}+1\right)}$ matrices $X_{A}^{(k)}=\left[x_{A}^{\left(k-m_{A}^{(k)}\right)}, \ldots, x_{A}^{(k)}\right], F_{A}^{(k)}=\left[f_{A}^{\left(k-m_{A}^{(k)}\right)}, \ldots, f_{A}^{(k)}\right]$, and coefficient vector $\alpha_{A}^{(k)}=\left[\alpha_{A, 0}^{(k)}, \ldots, \alpha_{A, m_{A}^{(k)}}^{(k)}\right]^{T} \in \mathbb{R}_{A}^{m_{A}^{(k)}+1},\left\|\alpha_{A}^{(k)}\right\|_{1}=1$. Anderson Acceleration achieves a faster convergence than a simple fixed-point iteration by using the past information to generate new iterates as linear combinations of previous $m_{A}^{(k)}$ iterates [35, 36, 44], i.e.,
$$
\begin{align*}
x_{A}^{(k+1)} & =\bar{x}_{A}^{(k)}+\beta^{(k)} \bar{f}_{A}^{(k)} \\
& =\left(1-\beta^{(k)}\right) \sum_{i=0}^{m_{A}^{(k)}} \alpha_{A, i}^{(k)} x_{A}^{\left(k-m_{A}^{(k)}+i\right)}+\beta^{(k)} \sum_{i=0}^{m_{A}^{(k)}} \alpha_{A, i}^{(k)} g\left(x^{\left(k-m_{A}^{(k)}+i\right)}\right), \tag{2.3}
\end{align*}
$$
with given relaxation (or damping) parameters $\beta^{(k)} \in \mathbb{R}^{+}$and mixing coefficients $\alpha_{A, i}^{(k)} \in \mathbb{R}, i=0, \ldots, m_{A}^{(k)}$ selected to minimize the linearized residual (error) of a new
iterate within an affine space $\operatorname{Aff}\left\{f_{A}^{\left(k-m_{A}^{(k)}\right)}, \ldots, f_{A}^{(k)}\right\}$, i.e., obtained as a solution of the least-squares problem
$$
\begin{equation*}
\min _{\alpha \in \mathbb{R}^{m_{A}^{(k)}+1}}\left\|F_{A}^{(k)} \alpha\right\|_{2}^{2}=\min _{\alpha_{0}, \ldots, \alpha_{m_{A}}^{(k)}}\left\|\sum_{i=0}^{m_{A}^{(k)}} \alpha_{i} f_{A}^{\left(k-m_{A}^{(k)}+i\right)}\right\|_{2}^{2} \quad \text { s. t. } \quad \sum_{i=0}^{m_{A}^{(k)}} \alpha_{i}=1 \tag{2.4}
\end{equation*}
$$

Note that in the case of $\beta^{(k)}=1$ a general formulation (2.3) introduced in the original work of Anderson [2, 3] reduces to the Pulay mixing [35, 36], i.e.,
$$
\begin{equation*}
x_{A}^{(k+1)}=\sum_{i=0}^{m_{A}^{(k)}} \alpha_{A, i}^{(k)} g\left(x_{A}^{\left(k-m_{A}^{(k)}+i\right)}\right) . \tag{2.5}
\end{equation*}
$$

Therefore, Anderson Acceleration method can be summarized in Algorithm 2.1 and is often denoted as Anderson( $m$ ) method.
```
Algorithm 2.1 Anderson Acceleration Method (of fixed depth $m$ )
Input: Initial Anderson iterate $x_{A}^{(0)}$, a fixed depth $m \geq 1$ and a fixed damping pa-
    rameter $\beta$
    Compute $x_{A}^{(1)}=g\left(x_{A}^{(0)}\right)$
    for $k=1,2, \ldots$ until convergence do
        Set truncation and relaxation parameters $m_{A}^{(k)}, \beta^{(k)}$
            $/ *$ e.g. $\quad m_{A}^{(k)}=\min \{k, m\}, \quad \beta^{(k)}=\beta * /$
        Set Anderson residuals $F_{A}^{(k)}=\left[f_{A}^{\left(k-m_{A}^{(k)}\right)}, \ldots, f_{A}^{(k)}\right]$, with
            $f_{A}^{(i)}:=f\left(x_{A}^{(i)}\right)=g\left(x_{A}^{(i)}\right)-x_{A}^{(i)}$
        Determine mixing coefficients, i.e., $\alpha_{A}^{(k)}:=\left[\alpha_{A, 0}^{(k)}, \ldots, \alpha_{A, m_{A}^{(k)}}^{(k)}\right]^{T}$ that solves
    Problem (2.4)
        Set $x_{A}^{(k+1)}=\left(1-\beta^{(k)}\right) \sum_{i=0}^{m_{A}^{(k)}} \alpha_{A, i}^{(k)} x_{A}^{\left(k-m_{A}^{(k)}+i\right)}+\beta^{(k)} \sum_{i=0}^{m_{A}^{(k)}} \alpha_{A, i}^{(k)} g\left(x_{A}^{\left(k-m_{A}^{(k)}+i\right)}\right)$
    end for
Output: $x_{A}^{(k)}$ that solves $f(x)=0$.
```


Note that in what follows we consider the case of $\beta=1$, since $\beta \neq 1$ can be reduced to the latter by setting $f_{\beta}(x)=\beta f(x)$ and $g_{\beta}(x)=x+\beta f(x)$.
2.2. CROP Algorithm. Analogously, we consider iterates $x_{C}^{(k)}$, a sequence of recorded search directions $\Delta x_{C}^{(i)}:=x_{C}^{(i+1)}-x_{C}^{(i)}, i=k-m_{C}^{(k)}, \ldots, k-1$, and the residual (error) vectors $f_{C}^{(k)}$ generated by CROP algorithm outlined in Algorithm 2.2 also called $\operatorname{CROP}(m)$ algorithm. Then the new search direction $\Delta x_{C}^{(k)}=x_{C}^{(k+1)}-x_{C}^{(k)}$ is chosen in the space spanned by the prior $m_{C}^{(k)}$ search directions $\Delta x_{C}^{(i)}, i=k- m_{C}^{(k)}, \ldots, k-1$ and the most recent residual (error) vector $f_{C}^{(k)}$, i.e.,
$$
x_{C}^{(k+1)}=x_{C}^{(k)}+\sum_{i=k-m_{C}^{(k)}}^{k-1} \eta_{i} \Delta x_{C}^{(i)}+\eta_{k} f_{C}^{(k)}
$$
with some coefficients $\eta_{k-m_{C}^{(k)}}, \ldots, \eta_{k} \in \mathbb{R}$.

Let us assume we have carried $k$ steps of the CROP algorithm, i.e., we have the subspace of optimal vectors $\operatorname{span}\left\{x_{C}^{(1)}, \ldots, x_{C}^{(k)}\right\}$ at hand. From the residual vector $f_{C}^{(k)}$, we can introduce a preliminary improvement of the current iterate $x_{C}^{(k)}$, i.e.,
$$
\begin{equation*}
\widetilde{x}_{C}^{(k+1)}:=x_{C}^{(k)}+f_{C}^{(k)} . \tag{2.6}
\end{equation*}
$$

Now, since (2.6) is equivalent to $f_{C}^{(k)}=\widetilde{x}_{C}^{(k+1)}-x_{C}^{(k)}$, we can find the optimal vector $x_{C}^{(k+1)}$ within the affine subspace $\operatorname{span}\left\{x_{C}^{(1)}, \ldots, x_{C}^{(k)}, \widetilde{x}_{C}^{(k+1)}\right\}$, i.e.,
$$
\begin{equation*}
x_{C}^{(k+1)}=\sum_{i=0}^{m_{C}^{(k+1)}-1} \alpha_{C, i}^{(k+1)} x_{C}^{\left(k+1-m_{C}^{(k+1)}+i\right)}+\alpha_{C, m_{C}^{(k+1)}}^{(k+1)} \widetilde{x}_{C}^{(k+1)} \tag{2.7}
\end{equation*}
$$
with $\sum_{i=0}^{m_{C}^{(k+1)}} \alpha_{C, i}^{(k+1)}=1$. The estimated residual (error) $f_{C}^{(k+1)}$ corresponding to the iterate $x_{C}^{(k+1)}$ is constructed as the linear combination of the estimated residuals (errors) of each component in (2.7) with the same coefficients, i.e.,
$$
\begin{equation*}
f_{C}^{(k+1)}=\sum_{i=0}^{m_{C}^{(k+1)}-1} \alpha_{C, i}^{(k+1)} f_{C}^{\left(k+1-m_{C}^{(k+1)}+i\right)}+\alpha_{C, m_{C}^{(k+1)}}^{(k+1)} \widetilde{f}_{C}^{(k+1)} \tag{2.8}
\end{equation*}
$$

Note that in general, unlike for the Anderson Acceleration method, $f_{C}^{(k+1)} \neq f\left(x_{C}^{(k+1)}\right.$. As before, the updates $x_{C}^{(k+1)}$ and $f_{C}^{(k+1)}$ can be written in the matrix form, i.e.,
$$
\begin{equation*}
x_{C}^{(k+1)}=X_{C}^{(k+1)} \alpha_{C}^{(k+1)} \quad \text { and } \quad f_{C}^{(k+1)}=F_{C}^{(k+1)} \alpha_{C}^{(k+1)}, \tag{2.9}
\end{equation*}
$$
with $X_{C}^{(k)}=\left[x_{C}^{\left(k-m_{A}^{(k)}\right)}, \ldots, x_{C}^{(k)}, \widetilde{x}_{C}^{(k+1)}\right]$ and $F_{C}^{(k)}=\left[f_{C}^{\left(k-m_{C}^{(k)}\right)}, \ldots, f_{C}^{(k)}, \widetilde{f}_{C}^{(k+1)}\right]$ in $\mathbb{R}^{n \times\left(m_{C}^{(k+1)}+1\right)}$, and coefficients vector $\alpha_{C}^{(k+1)}=\left[\alpha_{C, 0}^{(k)}, \ldots, \alpha_{C, m_{C}^{(k+1)}}^{(k+1)}\right]^{T}$ in $\mathbb{R}^{m_{C}^{(k+1)}+1}$. Minimizing the norm of the residual (error) defined in (2.8) results in a constrained least-squares problem
$$
\begin{equation*}
\min _{\alpha}\left\|F_{C}^{(k)} \alpha\right\|_{2}^{2}=\min _{\alpha_{0}, \ldots, \alpha_{m_{C}^{(k+1)}}}\left\|\sum_{i=0}^{m_{C}^{(k+1)}-1} \alpha_{i} f_{C}^{\left(k+1-m_{C}^{(k+1)}+i\right)}+\alpha_{m_{C}^{(k+1)}} \widetilde{f}_{C}^{(k+1)}\right\|_{2}^{2} \tag{2.10}
\end{equation*}
$$
such that $\sum_{i=0}^{m_{C}^{(k+1)}} \alpha_{C, i}^{(k+1)}=1$, with a vector of mixing coefficients $\alpha_{C}^{(k+1)}$ as a solution.
In Algorithm 2.2 the superscript of the iterates in step $k$ is chosen as $k+1$ instead of $k$ for a reason. In this way, the case of $m_{C}^{(k)}=k$ indicates no truncation. Also, as we will see in Section 3, this choice enables us to understand the correspondence between classical Anderson Acceleration method and CROP algorithm.
2.3. The Least-Squares Problem. For both the classical Anderson Acceleration method and CROP algorithm, obtaining mixing coefficients $\alpha_{0}^{(k)}, \ldots, \alpha_{m^{(k)}}^{(k)}$ requires solving constrained least-squares problems, i.e., (2.4) and (2.10), of the same general form
$$
\begin{equation*}
\min _{\alpha_{0}^{(k)}, \ldots, \alpha_{m}^{(k)}}\left\|\sum_{i=0}^{m^{(k)}} \alpha_{i}^{(k)} f^{\left(k-m^{(k)}+i\right)}\right\|_{2}^{2} \quad \text { such that } \quad \sum_{i=0}^{m^{(k)}} \alpha_{i}^{(k)}=1 \tag{2.11}
\end{equation*}
$$
```
Algorithm 2.2 CROP Algorithm (of fixed depth $m$ )
Input: Initial CROP iterate $x_{C}^{(0)}$, initial fixed depth $m \geq 1$
    Compute $f_{C}^{(0)}=f\left(x_{C}^{(0)}\right)$
    for $k=0,1,2, \ldots$ until convergence do
        Set $\widetilde{x}_{C}^{(k+1)}=x_{C}^{(k)}+f_{C}^{(k)}$ and truncation parameter $m_{C}^{(k+1)}$
        $/ *$ e.g. $\quad m_{C}^{(k+1)}=\min \{k+1, m\} * /$
        Set $F_{C}^{(k+1)}=\left[f_{C}^{\left(k+1-m_{C}^{(k)}\right)}, \ldots, f_{C}^{(k)}, \widetilde{f}_{C}^{(k+1)}\right]$ with $\widetilde{f}_{C}^{(k+1)}=f\left(\widetilde{x}_{C}^{(k+1)}\right)$
        Determine mixing coefficients, i.e., $\alpha_{C}^{(k+1)}=\left[\alpha_{C, 0}^{(k+1)}, \ldots, \alpha_{C, m_{C}^{(k+1)}}^{(k+1)}\right]^{T}$ that
        solves Problem $\left({ }_{k+1}^{2.10}\right)$
        Set $x_{C}^{(k+1)}=\sum_{i=0}^{m_{C}^{(k+1)}-1} \alpha_{C, i}^{(k+1)} x_{C}^{\left(k+1-m_{C}^{(k+1)}+i\right)}+\alpha_{C, m_{C}^{(k+1)}}^{(k+1)} \widetilde{x}_{C}^{(k+1)}$
        Set CROP residuals $f_{C}^{(k+1)}=\sum_{i=0}^{m_{C}^{(k+1)}-1} \alpha_{C, i}^{(k+1)} f_{C}^{\left(k+1-m_{C}^{(k+1)}+i\right)}+\alpha_{C, m_{C}^{(k+1)}}^{(k+1)} \widetilde{f}_{C}^{(k+1)}$
    end for
Output: $x_{C}^{(k+1)}$ that solves $f(x)=0$.
```

defined within the affine subspace $\operatorname{Aff}\left\{x^{\left(k-m^{(k)}\right)}, \ldots, x^{(k)}\right\}$. In general, (2.11) can be solved by the method of Lagrange multipliers [35]. By changing the barycentric coordinates into the affine frame, (2.11) becomes
$$
\begin{equation*}
\min _{\gamma_{1}^{(k)}, \ldots, \gamma_{m^{(k)}}^{(k)}}\left\|f^{(k)}-\sum_{i=1}^{m^{(k)}} \gamma_{i}^{(k)} \Delta f^{\left(k-m^{(k)}+i\right)}\right\|_{2}^{2} \tag{2.12}
\end{equation*}
$$
where $\Delta f^{(i)}=f^{(i+1)}-f^{(i)}$. Also, $\alpha^{(k)}$ and $\gamma^{(k)}$ can be transformed, i.e., $\alpha_{0}^{(k)}=\gamma_{1}^{(k)}$, $\alpha_{i}^{(k)}=\gamma_{i+1}^{(k)}-\gamma_{i}^{(k)}, i=1, \ldots, m^{(k)}-1, \alpha_{m^{(k)}}^{(k)}=1-\gamma_{m^{(k)}}^{(k)}$. Now, (2.12) can be solved by minimal equations. Let $\mathscr{F}^{(k)}:=\left[\Delta f^{\left(k-m^{(k)}\right)}, \ldots, \Delta f^{(k-1)}\right] \in \mathbb{R}^{n \times m^{(k)}}$. Then, $\bar{f}^{(k)}=f^{(k)}-\mathscr{F}^{(k)} \gamma^{(k)}$ and the solution of (2.12) is given as
$$
\begin{equation*}
\gamma^{(k)}=\left[\left(\mathscr{F}^{(k)}\right)^{T} \mathscr{F}^{(k)}\right]^{-1}\left(\mathscr{F}^{(k)}\right)^{T} f_{k} . \tag{2.13}
\end{equation*}
$$

Using (2.12) and (2.13) we can write the update of Anderson Acceleration and CROP method as
$$
\begin{equation*}
x_{A}^{(k+1)}=x_{A}^{(k)}+\beta f_{A}^{(k)}-\left(\mathscr{X}_{A}^{(k)}+\beta \mathscr{F}_{A}^{(k)}\right)\left[\left(\mathscr{F}_{A}^{(k)}\right)^{T} \mathscr{F}_{A}^{(k)}\right]^{-1}\left(\mathscr{F}_{A}^{(k)}\right)^{T} f_{A}^{(k)}, \tag{2.14}
\end{equation*}
$$
and
$$
\begin{equation*}
x_{C}^{(k+1)}=\widetilde{x}_{C}^{(k)}-\mathscr{X}_{C}^{(k+1)}\left[\left(\mathscr{F}_{C}^{(k+1)}\right)^{T} \mathscr{F}_{C}^{(k+1)}\right]^{-1}\left(\mathscr{F}_{C}^{(k+1)}\right)^{T} \widetilde{f}_{C}^{(k+1)}, \tag{2.15}
\end{equation*}
$$
where
$$
\begin{gathered}
\mathscr{X}_{A}^{(k)}=\left[\Delta x_{A}^{\left(k-m_{A}^{(k)}\right)}, \ldots, \Delta x_{A}^{(k-1)}\right], \quad \mathscr{F}_{A}^{(k)}=\left[\Delta f_{A}^{\left(k-m_{A}^{(k)}\right)}, \ldots, \Delta f_{A}^{(k-1)}\right] \\
\mathscr{X}_{C}^{(k+1)}=\left[\Delta x_{C}^{\left(k+1-m_{C}^{(k+1)}\right)}, \ldots, \Delta x_{C}^{(k-1)}, \widetilde{x}_{C}^{(k+1)}-x_{C}^{(k)}\right] \\
\mathscr{F}_{C}^{(k+1)}=\left[\Delta f_{C}^{\left(k+1-m_{C}^{(k+1)}\right)}, \ldots, \Delta f_{C}^{(k-1)}, \widetilde{f}_{C}^{(k+1)}-f_{C}^{(k)}\right]
\end{gathered}
$$

In the practical implementation, the basis $\Delta f^{(k)}$ are often orthogonalized using a QR factorization which also enables the least-squares problem use the information from the previous iteration steps [44, 30]. In both Algorithm 2.1 and Algorithm 2.2, the least-squares problem (line 5) is solved using the QR factorization, however, if the least-squares problem is small, explicit pseudoinverse formulation of (2.14) and (2.15) is used instead.
2.4. The damping parameter $\beta$. It is well-known that a good choice of the damping parameter $\beta_{k}$ significantly influences the convergence of Anderson Acceleration method. In the case of a fixed damping, i.e., $\beta_{k}=\beta$, if $\beta \neq 1$, the update formula (2.5) has the form
$$
x_{A}^{(k+1)}=(1-\beta) \sum_{i=0}^{m_{A}^{(k)}} \alpha_{i}^{(k)} x_{A}^{\left(k-m_{A}^{(k)}+i\right)}+\beta \sum_{i=0}^{m_{A}^{(k)}} \alpha_{i}^{(k)} g\left(x_{A}^{\left(k-m_{A}^{(k)}+i\right)}\right) .
$$

Defining $g_{\beta}(x):=(1-\beta) x+\beta g(x)=x+\beta(g(x)-x)=x+\beta f(x)$ yields an equivalent formulation
$$
x_{A}^{(k+1)}=\sum_{i=0}^{m_{A}^{(k)}} \alpha_{i}^{(k)} g_{\beta}\left(x_{k-m_{A}^{(k)}+i}\right),
$$
which has the same form as the iterates in Algorithm 2.1. Hence, Anderson Acceleration method with a fixed depth parameter $\beta$ can be regarded as running Algorithm 2.1 with a new fixed-point iteration function $g_{\beta}$. It is worth mentioning, that the corresponding residual (error) $f_{\beta}(x)=g_{\beta}(x)-x=\beta(g(x)-x)=\beta f(x)$, although different than $f(x)$, has the same zeros as $f(x)$.

According to $\left[23\right.$, Propsition 4.3], the residual $f_{A}^{(k)}$ at step $k$ is bounded by $\beta_{k}$ in the following way:
$$
\begin{equation*}
\left\|f_{A}^{(k+1)}\right\|_{2} \leq \theta_{k+1}\left(\left(\left(1-\beta_{k}\right)+L_{g} \beta_{k}\right)\right)\left\|f_{A}^{(k)}\right\|+\sum_{j=0}^{m_{A}} \mathcal{O}\left(\left\|f_{A}^{(k-j)}\right\|_{2}^{2}\right) \tag{2.16}
\end{equation*}
$$
where $L_{g}$ is the Lipchitz constant of the mapping $g$, and $\theta_{k+1}=\left\|\bar{f}_{A}^{(k+1)}\right\|_{2} /\left\|f_{A}^{(k+1)}\right\|_{2}$.
2.5. The truncation parameter $m^{(k)}$. In general, the subspace truncation parameter $m^{(k)} \geq 1\left(m_{A}^{(k)}\right.$ or $\left.m_{C}^{(k)}\right)$ determines the dimension of the search space for the next trial vector $\left(x_{A}^{(k+1)}\right.$ or $\left.x_{C}^{(k+1)}\right)$, e.g., the size of the least-squares problem 2.11. Hence, the affine subspace in iteration step $k$ involves $m^{(k)}+1$ vectors. Obviously, $m^{(k)}=0$ corresponds to the fixed-point iteration method.

Note that for Anderson Acceleration method and CROP algorithm, the situations are slightly different. In Anderson Acceleration, at step $k$ iterate $x_{A}^{(k+1)}$ is computed from iterates $x_{A}^{\left(k-m^{(k)}\right)}, \ldots, x_{A}^{(k)}$, whereas in CROP algorithm, $x_{C}^{(k+1)}$ is computed using vectors $x_{C}^{\left(k+1-m^{(k+1)}\right)}, \ldots, x_{C}^{(k)}, \widetilde{x}_{C}^{(k+1)}$. Therefore, we can immediately see that Anderson(1) is a mixing method that needs the historical information in every step, while $\operatorname{CROP}(1)$ is a fixed-point iteration, i.e., in Anderson(1) step to compute $x_{A}^{(k+1)}$ we need access to $x_{A}^{(k-1)}$ and $x_{A}^{(k)}$, whereas in the case of CROP (1) we only need $x_{C}^{(k)}$ to determine next iterate $x_{C}^{(k+1)}$.

The proper choice of $m^{(k)}$ is very important as it affects the convergence speed, time and overall complexity of these methods. For the fixed depth methods, a fixed parameter $m$ (e.g. $m \leq 5$ ) is set before the iteration starts and the truncation parameter is chosen at each iteration step $k$ to be $m^{(k)}=\min \{k, m\}$. For further discussions
regarding various choices of $m^{(k)}$ see [37, 18]. Values $m \leq 5$ are often used in practice, in particular $m=1$. A larger $m$ is usually unnecessary and will cause the least-squares problem to be hard to solve. In what follows, we will briefly talk about values of $m$ for CROP method.

Restarting and adaptation are used to maintain the dimension of the subspace by choosing parameter $m^{(k)}$ in each step [17]. Also, some techniques like filtering [33] do not use the most recent $m^{(k)}$ trial vectors and residuals. However, they usually need to store a list of trial vectors and residuals of size $m^{(k)}$.
3. Anderson Acceleration vs CROP Method. The aim of this section is to establish a unified framework which will enable us to show that Anderson Acceleration method and CROP algorithm are equivalent. By showing the equivalence between the full versions of these two methods, some new variants, i.e., CROP-Anderson and rCROP, can be developed and modified to get the real residuals.

Theorem 3.1. Let us consider applying Anderson Acceleration method ( $\beta_{k}=1$ ) and $C R O P$ algorithm to the nonlinear problem $f(x)=0$ with initial values $x_{A}^{(0)}=x_{C}^{(0)}$ and no truncation ( $m^{(k)}=k$ ). Then, for $k=0,1, \ldots$ until convergence ( $f_{C}^{(0)} \neq 0$ )
$$
x_{C}^{(k)}=\bar{x}_{A}^{(k)}, f_{C}^{(k)}=\bar{f}_{A}^{(k)} \quad \text { and } \quad x_{A}^{(k+1)}=\widetilde{x}_{C}^{(k+1)}, f_{A}^{(k+1)}=\widetilde{f}_{C}^{(k+1)}
$$
with $\bar{x}_{A}^{(k)}, x_{A}^{(k+1)}$ defined as in (2.1) and (2.3), and $\widetilde{x}_{C}^{(k+1)}, x_{C}^{(k)}$ as in (2.6) and (2.7).
Proof. The proof follows by induction. Since $x_{A}^{(0)}=x_{C}^{(0)}$, then for $k=0, x_{C}^{(0)}= \bar{x}_{A}^{(0)}, f_{C}^{(0)}=f\left(x_{C}^{(0)}\right)=f\left(x_{A}^{(0)}\right)=f_{A}^{(0)}=\bar{f}_{A}^{(0)}$. Hence, $\widetilde{x}_{C}^{(1)}=x_{C}^{(0)}+f_{C}^{(0)}=x_{A}^{(0)}+f_{A}^{(0)}= x_{A}^{(1)}$ and $\widetilde{f}_{C}^{(1)}=f\left(\widetilde{x}_{C}^{(1)}\right)=f\left(x_{A}^{(1)}\right)=f_{A}^{(1)}$. Assume that for all $\ell \leq k, x_{C}^{(\ell)}=\bar{x}_{A}^{(\ell)}$, $f_{C}^{(\ell)}=\bar{f}_{A}^{(\ell)}, \widetilde{x}_{C}^{(\ell+1)}=x_{A}^{(\ell+1)}$ and $\widetilde{f}_{C}^{(\ell+1)}=f_{A}^{(\ell+1)}$. Then, for $k+1$
$$
\begin{aligned}
f_{C}^{(k+1)} & =\sum_{i=0}^{m^{(k+1)}-1} \alpha_{C, i}^{(k+1)} f_{C}^{\left(k+1-m_{C}^{(k+1)}+i\right)}+\alpha_{C, m^{(k+1)}}^{(k+1)} \widetilde{f}_{C}^{(k+1)} \\
& =\sum_{i=0}^{k} \alpha_{C, i}^{(k+1)} \bar{f}_{A}^{(i)}+\alpha_{C, k+1}^{(k+1)} f_{A}^{(k+1)}=\sum_{i=0}^{k} \alpha_{C, i}^{(k+1)} \sum_{j=0}^{i} \alpha_{A, j}^{(i)} f_{A}^{(j)}+\alpha_{C, k+1}^{(k+1)} f_{A}^{(k+1)} \\
& =\sum_{j=0}^{k} \sum_{i=j}^{k} \alpha_{C, i}^{(k+1)} \alpha_{A, j}^{(i)} f_{A}^{(j)}+\alpha_{C, k+1}^{(k+1)} f_{A}^{(k+1)} .
\end{aligned}
$$

Let $\widehat{\alpha}_{j}^{(k+1)}=\sum_{i=j}^{k} \alpha_{C, i}^{(k+1)} \alpha_{A, j}^{(i)}, j=1, \ldots, k$, and $\widehat{\alpha}_{k+1}^{(k+1)}=\alpha_{C, k+1}^{(k+1)}$, then
$$
f_{C}^{(k+1)}=\sum_{j=0}^{k+1} \widehat{\alpha}_{j}^{(k+1)} f_{A}^{(j)} \quad \text { and } \quad \sum_{j=0}^{k+1} \widehat{\alpha}_{j}^{(k+1)}=\sum_{i=0}^{k+1} \alpha_{C, i}^{(k+1)}=1 .
$$

Since $f_{C}^{(k+1)}=\sum_{i=0}^{k+1} \widehat{\alpha}_{i}^{(k+1)} f_{A}^{(i)}$ and $\bar{f}_{A}^{(k+1)}=\sum_{i=0}^{k+1} \alpha_{A, i}^{(k+1)} f_{A}^{(i)}$ are the solutions of the least-squares problem in the same affine space $\operatorname{Aff}\left\{f_{A}^{(0)}, \ldots, f_{A}^{(k+1)}\right\}$, we know that
$\widehat{\alpha}_{i}^{(k+1)}=\alpha_{A, i}^{(k+1)}$, and $f_{C}^{(k+1)}=\bar{f}_{A}^{(k+1)}$. Also,
$$
\begin{aligned}
& x_{C}^{(k+1)}=\sum_{i=0}^{k+1} \widehat{\alpha}_{i}^{(k+1)} x_{A}^{(i)}=\sum_{i=0}^{k+1} \alpha_{A, i}^{(k+1)} f_{A}^{(i)}=\bar{x}_{A}^{(k+1)} \\
& \tilde{x}_{C}^{(k+2)}=x_{C}^{(k+1)}+f_{C}^{(k+1)}=\bar{x}_{A}^{(k+1)}+\bar{f}_{A}^{(k+1)}=x_{A}^{(k+2)}
\end{aligned}
$$
and $\widetilde{f}_{C}^{(k+2)}=f\left(\widetilde{x}_{C}^{(k+2)}\right)=f\left(x_{A}^{(k+2)}\right)=f_{A}^{(k+2)}$. Therefore by induction $x_{C}^{(k)}=\bar{x}_{A}^{(k)}$, $f_{C}^{(k)}=\bar{f}_{A}^{(k)}, \widetilde{x}_{C}^{(k+1)}=x_{A}^{(k+1)}$ and $\widetilde{f}_{C}^{(k+1)}=f_{A}^{(k+1)}$ for all $k \in \mathbb{N}$.

Remark 3.2. Since the CROP residual $f_{C}^{(k)}$ may become exactly 0 , the Algorithm 2.2 can break down. However, before the actual breakdown occurs, Theorem 3.1 holds. If the CROP algorithm breaks down at step $k$ with $f_{C}^{(k)}=0$, it stops, whereas in the case of Anderson Acceleration $x_{A}^{(k+1)}=x_{A}^{(k)}$ and the stagnation occurs.
To illustrate a connection between Anderson Acceleration method and CROP algorithm, in Figure 1, we consider one and a half step of Anderson Acceleration from iterate $x_{A}^{(k)}$ to $\bar{x}_{A}^{(k+1)}$. By the equivalence of the least-squares problems in (2.4) and (2.10), block (II) in Figure 1 is equivalent to a single step of CROP algorithm. Changing the weighted averages of Anderson Acceleration to CROP type averages enables us to obtain a different scheme illustrated in Figure 2.
$$
\begin{gather*}
(I)\left\{\begin{array}{c}
x_{A}^{(k)}=\bar{x}_{A}^{(k-1)}+\bar{f}_{A}^{(k-1)} \\
\downarrow \\
f_{A}^{(k)}:=f\left(x_{A}^{(k)}\right) \\
\downarrow \\
\text { Find } \alpha_{A}^{(k)} \text { that minimizes }\left\|\bar{f}_{A}^{(k)}\right\|_{2} \\
\text { with } \bar{f}_{A}^{(k)}=\sum_{i=0}^{m_{A}^{(k)}} \alpha_{A, i}^{(k)} f_{A}^{\left(k-m_{A}^{(k)}+i\right)} \\
\downarrow \\
\bar{x}_{A}^{(k)}=\sum_{i=0}^{m_{A}^{(k)}} \alpha_{A, i}^{(k)} x_{A}^{\left(k-m_{A}^{(k)}+i\right)} \\
\downarrow \\
x_{A}^{(k+1)}=\bar{x}_{A}^{(k)}+\bar{f}_{A}^{(k)} \\
\downarrow \\
f_{A}^{(k+1)}:=f\left(x_{A}^{(k+1)}\right) \\
\downarrow \\
\text { Find } \alpha_{A}^{(k+1)} \text { that minimizes }\left\|\bar{f}_{A}^{(k+1)}\right\|_{2} \\
\text { with } \bar{f}_{A}^{(k+1)}=\sum_{i=0}^{m_{A}^{(k+1)}} \alpha_{A, i}^{(k+1)} f_{A}^{\left(k+1-m_{A}^{(k+1)}+i\right)} \\
\downarrow \\
\bar{x}_{A}^{(k+1)}=\sum_{i=0}^{m_{A}^{(k+1)}} \alpha_{A, i}^{(k+1)} x_{A}^{\left(k+1-m_{A}^{(k+1)}+i\right)}
\end{array}\right\} \tag{II}
\end{gather*}
$$

Fig. 1: Anderson Acceleration in Anderson notation.

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_08_2ce2904da8f9b738a883g-10.jpg?height=1001&width=1191&top_left_y=363&top_left_x=295}
\captionsetup{labelformat=empty}
\caption{Fig. 2: CROP method in CROP notation.}
\end{figure}

Hence, blocks (I - IV) in Figures 1 and 2 can be associated with four different algorithms: block (I) illustrates Anderson Acceleration steps (2.1)-(2.3); block (II) is equivalent to steps (2.6)-(2.7) of CROP algorithm; block (III) is equivalent to the Anderson Acceleration steps (2.1)-(2.3) and block (IV) illustrates CROP algorithm steps (2.6)-(2.7). Note that following the notation of Figures 1 and 2 allows us to use the same variables $x^{(k)}$ and $\bar{x}^{(k)}$ in all four different algorithms (blocks I - IV) and write them all in the unified framework in terms of previous iterates of Anderson Acceleration method. However, we can also express all quantities of interest in terms of CROP past information. Since CROP algorithm has better behavior while keeping less historical information, we can run Anderson Acceleration executing block (III), which is called CROP generalization of Anderson Acceleration [22] or, for the simplicity, CROP-Anderson method.

CROP-Anderson method, denoted as $C A$, follows the steps of Algorithm 2.2, with the only difference of checking the size of residuals (errors) at each iteration step and the final output result being $\widetilde{x}_{C}^{(k+1)}$. Although we can express the steps of CROP-Anderson method using CROP algorithm ( $C$ ) notation, we can easily use the alternative CROP-Anderson ( $C A$ ) formulation, i.e., we first set $x_{C A}^{(k)}=\widetilde{x}_{C}^{(k)}, f_{C A}^{(k)}= \widetilde{f}_{C}^{(k)}, \bar{x}_{C A}^{(k)}=x_{C}^{(k)}$ and $\bar{f}_{C A}^{(k)}=f_{C}^{(k)}$. Then by Theorem 3.1, we get the following corollary.
```
Algorithm 3.1 CROP-Anderson method (of fixed depth $m$ )
Input: Initial CROP-Anderson iterate $x^{(0)}$, initial fixed depth $m \geq 1$
    Compute $f_{C}^{(0)}=f\left(x_{C}^{(0)}\right)$
    for $k=0,1,2, \ldots$ do
        Set $\widetilde{x}_{C}^{(k+1)}=x_{C}^{(k)}+f_{C}^{(k)}$ and $\widetilde{f}_{C}^{(k+1)}=f\left(\widetilde{x}_{C}^{(k+1)}\right)$
        if $\widetilde{f}_{C}^{(k+1)}<$ tol then
            break
        end if
        Set truncation parameter $m_{C}^{(k+1)}$
        $/ *$ e.g. $\quad m_{C}^{(k+1)}=\min \{k+1, m\} \quad * /$
        Set $F_{C}^{(k+1)}=\left[f_{C}^{\left(k-m_{C}^{(k)}\right)}, \ldots, f_{C}^{(k)}, \widetilde{f}_{C}^{(k+1)}\right]$.
        Determine mixing coefficients, i.e., $\alpha_{C}^{(k+1)}=\left[\alpha_{C, 0}^{(k+1)}, \ldots, \alpha_{C, m_{C}^{(k+1)}}^{(k+1)}\right]^{T}$ that
        solves Problem (2.10)
        Set $x_{C}^{(k+1)}=\sum_{i=0}^{m_{C}^{(k+1)}-1} \alpha_{C, i}^{(k+1)} x_{C}^{\left(k+1-m_{C}^{(k+1)}+i\right)}+\alpha_{C, m_{C}^{(k+1)}}^{(k+1)} \widetilde{x}_{C}^{(k+1)}$
        Set $f_{C}^{(k+1)}=\sum_{i=0}^{m_{C}^{(k+1)}-1} \alpha_{C, i}^{(k+1)} f_{C}^{\left(k+1-m_{C}^{(k+1)}+i\right)}+\alpha_{C, m_{C}^{(k+1)}}^{(k+1)} \widetilde{f}_{C}^{(k+1)}$
    end for
Output: $\widetilde{x}_{C}^{(k+1)}$ that solves $f(x)=0$.
```


Corollary 3.3. Let us consider applying Anderson Acceleration method ( $\beta_{k}=$ 1) and $C R O P$-Anderson algorithm to the nonlinear problem $f(x)=0$ with initial values $x_{A}^{(0)}=x_{C}^{(0)}=x_{C A}^{(0)}$ and no truncation ( $m^{(k)}=k$ ). Then, for $k=0,1, \ldots$
$$
x_{C A}^{(k)}=x_{A}^{(k)}, f_{C A}^{(k)}=f_{A}^{(k)} \quad \text { and } \quad \bar{x}_{C A}^{(k)}=\bar{x}_{A}^{(k)}, \bar{f}_{C A}^{(k)}=\bar{f}_{A}^{(k)} .
$$

In CROP algorithm, updates $x_{C}^{(k)}$ have the corresponding residuals $f_{C}^{(k)}$ associated with the least-squares Problem 2.10. Although $f_{C}^{(k)}$ are used to terminate the iterations, they are still approximated residuals and are affected by the initial guess $x_{C}^{(0)}$. Thus, we call them control residuals. Relatively, $r_{C}^{(k)}=f\left(x_{C}^{(k)}\right)$ are the real residuals corresponding to iterates $x_{C}^{(k)}$, which obviously are never explicitly computed. Consequently, when solving nonlinear problems, the control residuals may not estimate the real residuals very well, and may cause a lot of problems, e.g. breakdowns. One remedy is to use CROP-Anderson method. Alternatively, the real residuals can be used instead of the control residuals. By changing line 7 of Algorithm 2.2 and the corresponding line in CROP-Anderson method (line 11 in Algorithm 3.1) into $f_{C}^{(k+1)}=f\left(x_{C}^{k+1}\right)$, CROP algorithm and CROP-Anderson method become completely different algorithms. In what follows, we will refer to them as rCROP and rCROP-Anderson. Note that in the case of Anderson Acceleration method, the control residuals and the real residuals are the same, i.e., $f_{A}^{(k)}=f\left(x_{A}^{(k)}\right)=r_{A}^{(k)}$.

The diagrams above are summarized in Table 1 which illustrates two steps of Anderson Acceleration method and CROP algorithm. The $k^{\text {th }}$ step of each method is highlighted in bold. Note that the "optimization" steps have different results for the two methods, but the "average $x$ " and the "iteration $k$ " steps should have the same value. Analogously, the "average $f$ " and the "residual $k$ " steps should be the
same. If we consider $\widetilde{x}_{C}$ and $\widetilde{f}_{C}$ from CROP algorithm as iterates and residuals, respectively, then we get steps of CROP-Anderson method (shaded part of CROP), which is equivalent to Anderson Acceleration method.

If in CROP and CROP-Anderson algorithms we use $f_{C}^{(k)}=f\left(x_{C}^{(k)}\right)$ instead of $f_{C}^{(k)}=F_{C}^{(k)} \alpha_{C}^{(k)}$ as $k^{\text {th }}$ residual, then we get real residual CROP (rCROP) and the corresponding rCROP-Anderson (shaded part of CROP method). The rCROP algorithm is presented along Anderson Acceleration method in Table 2.

\begin{table}
\begin{tabular}{|l|l|l|l|}
\hline & Anderson Acceleration & CROP Algorithm & \\
\hline iteration $k$ & $x_{A}^{(k)}=\bar{x}_{A}^{(k-1)}+\bar{f}_{A}^{(k-1)}$ & $\widetilde{x}_{C}^{(k)}=x_{C}^{(k-1)}+f_{C}^{(k-1)}$ & new direction \\
\hline residual $k$ & $f_{A}^{(k)}=f\left(x_{A}^{(k)}\right)$ & $\widetilde{f}_{C}^{(k)}=f\left(\widetilde{x}_{C}^{(k)}\right)$ & new direction \\
\hline optimization & $\alpha_{\mathbf{A}}^{(\mathbf{k})}=\arg \min \left\|\mathbf{F}_{\mathbf{A}}^{(\mathbf{k})} \alpha\right\|_{\mathbf{2}}$ & $\alpha_{C}^{(k)}=\arg \min \left\|F_{C}^{(k)} \alpha\right\|_{2}$ & optimization \\
\hline average $x$ & $\overline{\mathbf{x}}_{\mathbf{A}}^{(\mathbf{k})}=\mathbf{X}_{\mathbf{A}}^{(\mathbf{k})} \alpha_{\mathbf{A}}^{(\mathbf{k})}$ & $x_{C}^{(k)}=X_{C}^{(k)} \alpha_{C}^{(k)}$ & iteration $k$ \\
\hline average $f$ & $\overline{\mathbf{f}}_{\mathbf{A}}^{(\mathbf{k})}=\mathbf{F}_{\mathbf{A}}^{(\mathbf{k})} \alpha_{\mathbf{A}}^{(\mathbf{k})}$ & $f_{C}^{(k)}=F_{C}^{(k)} \alpha_{C}^{(k)}$ & residual $k$ \\
\hline iteration $k+1$ & $\mathrm{x}_{\mathrm{A}}^{(\mathrm{k}+1)}=\overline{\mathrm{x}}_{\mathrm{A}}^{(\mathrm{k})}+\overline{\mathrm{f}}_{\mathrm{A}}^{(\mathrm{k})}$ & $\widetilde{\mathbf{x}}_{\mathbf{C}}^{(\mathbf{k}+\mathbf{1})}=\mathbf{x}_{\mathbf{C}}^{(\mathbf{k})}+\mathbf{f}_{\mathbf{C}}^{(\mathbf{k})}$ & new direction \\
\hline residual $k+1$ & $\mathbf{f}_{\mathbf{A}}^{(\mathbf{k}+\mathbf{1})}=\mathbf{f}\left(\mathbf{x}_{\mathbf{A}}^{(\mathbf{k}+\mathbf{1})}\right)$ & $\widetilde{\mathbf{f}}_{\mathbf{C}}^{(\mathbf{k}+\mathbf{1})}=\mathbf{f}\left(\widetilde{\mathbf{x}}_{\mathbf{C}}^{(\mathbf{k}+\mathbf{1})}\right)$ & new direction \\
\hline optimization & $\alpha_{A}^{(k+1)}=\arg \min \left\|F_{A}^{(k+1)} \alpha\right\|_{2}$ & $\alpha_{\mathbf{C}}^{(\mathbf{k}+\mathbf{1})}=\arg \min \left\|\mathbf{F}_{\mathbf{C}}^{(\mathbf{k}+\mathbf{1})} \alpha\right\|_{\mathbf{2}}$ & optimization \\
\hline average $x$ & $\bar{x}_{A}^{(k+1)}=X_{A}^{(k+1)} \alpha_{A}^{(k+1)}$ & $\mathbf{x}_{\mathbf{C}}^{(\mathbf{k}+\mathbf{1})}=\mathbf{X}_{\mathbf{C}}^{(\mathbf{k}+\mathbf{1})} \alpha_{\mathbf{C}}^{(\mathbf{k}+\mathbf{1})}$ & iteration $k+1$ \\
\hline average $f$ & $\bar{f}_{A}^{(k+1)}=F_{A}^{(k+1)} \alpha_{A}^{(k+1)}$ & $\mathrm{f}_{\mathrm{C}}^{(\mathrm{k}+1)}=\mathrm{F}_{\mathrm{C}}^{(\mathrm{k}+1)} \alpha_{\mathrm{C}}^{(\mathrm{k}+1)}$ & residual $k+1$ \\
\hline
\end{tabular}
\captionsetup{labelformat=empty}
\caption{Table 1: Two iteration steps of Anderson Acceleration and CROP method.}
\end{table}

\begin{table}
\begin{tabular}{|l|l|l|l|}
\hline & Anderson Acceleration & rCROP Algorithm & \\
\hline iteration $k$ & $x_{A}^{(k)}=\bar{x}_{A}^{(k-1)}+\bar{f}_{A}^{(k-1)}$ & $\widetilde{x}_{C}^{(k)}=x_{C}^{(k-1)}+f_{C}^{(k-1)}$ & new direction \\
\hline residual $k$ & $f_{A}^{(k)}=f\left(x_{A}^{(k)}\right)$ & $\widetilde{f}_{C}^{(k)}=f\left(\widetilde{x}_{C}^{(k)}\right)$ & new direction \\
\hline optimization & $\alpha_{\mathbf{A}}^{(\mathbf{k})}=\arg \min \left\|\mathbf{F}_{\mathbf{A}}^{(\mathbf{k})} \alpha\right\|_{\mathbf{2}}$ & $\alpha_{C}^{(k)}=\arg \min \left\|F_{C}^{(k)} \alpha\right\|_{2}$ & optimization \\
\hline average $x$ & $\overline{\mathbf{x}}_{\mathbf{A}}^{(\mathbf{k})}=\mathbf{X}_{\mathbf{A}}^{(\mathbf{k})} \alpha_{\mathbf{A}}^{(\mathbf{k})}$ & $x_{C}^{(k)}=X_{C}^{(k)} \alpha_{C}^{(k)}$ & iteration $k$ \\
\hline average $f$ & $\overline{\mathbf{f}}_{\mathbf{A}}^{(\mathbf{k})}=\mathbf{F}_{\mathbf{A}}^{(\mathbf{k})} \alpha_{\mathbf{A}}^{(\mathbf{k})}$ & $\mathrm{f}_{\mathrm{C}}^{(\mathrm{k})}=\mathrm{f}\left(\mathrm{x}_{\mathrm{C}}^{(\mathrm{k})}\right)$ & residual $k$ \\
\hline iteration $k+1$ & $\mathrm{x}_{\mathrm{A}}^{(\mathrm{k}+1)}=\overline{\mathrm{x}}_{\mathrm{A}}^{(\mathrm{k})}+\overline{\mathrm{f}}_{\mathrm{A}}^{(\mathrm{k})}$ & $\widetilde{\mathbf{x}}_{\mathbf{C}}^{(\mathbf{k}+\mathbf{1})}=\mathbf{x}_{\mathbf{C}}^{(\mathbf{k})}+\mathbf{f}_{\mathbf{C}}^{(\mathbf{k})}$ & new direction \\
\hline residual $k+1$ & $\mathbf{f}_{\mathrm{A}}^{(\mathbf{k}+\mathbf{1})}=\mathbf{f}\left(\mathbf{x}_{\mathrm{A}}^{(\mathbf{k}+\mathbf{1})}\right)$ & $\widetilde{\mathbf{f}}_{\mathbf{C}}^{(\mathbf{k}+\mathbf{1})}=\mathbf{f}\left(\widetilde{\mathbf{x}}_{\mathbf{C}}^{(\mathbf{k}+\mathbf{1})}\right)$ & new direction \\
\hline optimization & $\alpha_{A}^{(k+1)}=\arg \min \left\|F_{A}^{(k+1)} \alpha\right\|_{2}$ & $\alpha_{\mathbf{C}}^{(\mathbf{k}+\mathbf{1})}=\arg \min \left\|\mathbf{F}_{\mathbf{C}}^{(\mathbf{k}+\mathbf{1})} \alpha\right\|_{\mathbf{2}}$ & optimization \\
\hline average $x$ & $\bar{x}_{A}^{(k+1)}=X_{A}^{(k+1)} \alpha_{A}^{(k+1)}$ & $\mathbf{x}_{\mathbf{C}}^{(\mathbf{k}+\mathbf{1})}=\mathbf{X}_{\mathbf{C}}^{(\mathbf{k}+\mathbf{1})} \alpha_{\mathbf{C}}^{(\mathbf{k}+\mathbf{1})}$ & iteration $k+1$ \\
\hline average $f$ & $\bar{f}_{A}^{(k+1)}=F_{A}^{(k+1)} \alpha_{A}^{(k+1)}$ & $\mathrm{f}_{\mathrm{C}}^{(\mathrm{k}+1)}=\mathrm{f}\left(\mathrm{x}_{\mathrm{C}}^{(\mathrm{k}+1)}\right)$ & residual $k+1$ \\
\hline
\end{tabular}
\captionsetup{labelformat=empty}
\caption{Table 2: Two iteration steps of Anderson Acceleration and rCROP method.}
\end{table}
4. Broader View of Anderson Acceleration Method and CROP Algorithm. This section discusses the connections between CROP algorithm and some other state-of-the-art iterative methods. Following our findings in Section 3, we explore links between CROP and Krylov subspace methods in Subsection 4.1, and multisecant methods in Subsection 4.2.

In [45, 31], a Krylov acceleration method equivalent to flexible GMRES was introduced, and its Jacobian-free version utilizes the least-squares similar to Equation (2.4). [38] pointed out the connection between Anderson Acceleration and the GMRES method, and details on the equivalence between Anderson Acceleration without truncation and the GMRES method were provided [29, 44]. [19] showed that truncated Anderson Acceleration is a multi-Krylov method.
4.1. Connection with Krylov methods. Let us consider applying CROP algorithm to the simple linear problem: Find $x \in \mathbb{R}^{n}$ such that $A x=b$, with a nonsingular $A \in \mathbb{R}^{n \times n}$ and $b \in \mathbb{R}^{n}$. Then, the associated residual (error) vectors can be chosen as $f(x)=b-A x$, with the corresponding $g(x)=b+(I-A) x$.

First, we will present some facts about CROP algorithm's application to the linear problem $A x=b$.

Lemma 4.1. Consider using CROP algorithm to solve the linear problem $A x=b$. Then, the real and the control residuals are equal, i.e., $r_{C}^{(k)}=f_{C}^{(k)}$ for any $k \in \mathbb{N}$.

Proof. The proof follows by induction. For the initial residuals we have $r_{C}^{(0)}= f_{C}^{(0)}$. Assume that $r_{C}^{(\ell)}=f_{C}^{(\ell)}$ for all $\ell \leq k$. Then, for $k+1$
$$
\begin{aligned}
f_{C}^{(k+1)} & =\sum_{i=0}^{m_{C}^{(k+1)}-1} \alpha_{C, i}^{(k+1)} f_{C}^{\left(k+1-m_{C}^{(k+1)}+i\right)}+\alpha_{C, m_{C}^{(k+1)}}^{(k+1)} \widetilde{f}_{C}^{(k+1)} \\
& =\sum_{i=0}^{m_{C}^{(k+1)}-1} \alpha_{C, i}^{(k+1)} r_{C}^{\left(k+1-m_{C}^{(k+1)}+i\right)}+\alpha_{C, m_{C}^{(k+1)}}^{(k+1)} \widetilde{f}_{C}^{(k+1)} \\
& =\sum_{i=0}^{m_{C}^{(k+1)}-1} \alpha_{C, i}^{(k+1)}\left(b-A x_{C}^{\left(k+1-m_{C}^{(k+1)}+i\right)}\right)+\alpha_{C, m_{C}^{(k+1)}}^{(k+1)}\left(b-A \widetilde{x}_{C}^{(k+1)}\right) \\
& =b-A\left(\sum_{i=0}^{m_{C}^{(k+1)}-1} \alpha_{C, i}^{(k+1)} x_{C}^{\left(k+1-m_{C}^{(k+1)}+i\right)}+\alpha_{C, m_{C}^{(k+1)}}^{(k+1)} \widetilde{x}_{C}^{(k+1)}\right) \\
& =b-A x_{C}^{(k+1)}=r_{C}^{(k+1)}
\end{aligned}
$$

Hence, by induction $r_{C}^{(k)}=f_{C}^{(k)}$ for all $k \in \mathbb{N}$.
Lemma 4.2. If $C R O P$ algorithm is used to solve the linear problem $A x=b$, then $\widetilde{f}_{C}^{(k+1)}=(I-A) f_{C}^{(k)}$ for any $k \in \mathbb{N}$.

Proof.
$$
\begin{aligned}
\widetilde{f}_{C}^{(k+1)} & =b-A \widetilde{x}_{C}^{(k+1)}=b-A\left(b+(I-A) x_{C}^{(k)}\right)=(I-A)\left(b-A x_{C}^{(k)}\right) \\
& =(I-A) r_{C}^{(k)}=(I-A) f_{C}^{(k)}
\end{aligned}
$$

The equivalence of Anderson Acceleration method and the GMRES method is presented in [29, 44]. A similar result for CROP algorithm can be proved.

Theorem 4.3. Consider using CROP algorithm and the GMRES method to solve the linear problem $A x=b$ under the following assumptions:
1. $A$ is nonsingular.
2. Run CROP algorithm with $f(x)=b-A x, g(x)=f(x)+x=b+(I-A) x$ and no truncation.
3. The initial values of the GMRES and CROP coincide, i.e., $x_{G}^{(0)}=x_{C}^{(0)}$.

Then, for $k>0, x_{C}^{(k)}=x_{G}^{(k)}, f_{C}^{(k)}=r_{G}^{(k)}$, where $r_{G}^{(k)}$ denote the GMRES residual defined as $r_{G}^{(k)}:=b-A x_{G}^{(k)}$.

Before proving Theorem 4.3, we first need the following lemma.
Lemma 4.4. Consider using CROP algorithm and the GMRES method to solve the linear problem $A x=b$ under the assumptions from Theorem 4.3, and let the Krylov subspace associated with matrix $A$ and vector $r_{G}^{(0)}$ be given as
$$
\mathcal{K}_{n}=\mathcal{K}_{n}\left(A, r_{G}^{(0)}\right)=\operatorname{span}\left\{r_{G}^{(0)}, A r_{G}^{(0)}, \ldots, A^{n-1} r_{G}^{(0)}\right\}
$$

Then, $\mathcal{K}_{n}=\operatorname{span}\left\{f_{C}^{(0)}, \ldots, f_{C}^{(n-1)}\right\}$ for all $n \in \mathbb{N}^{+}$
Proof. The proof proceeds by induction. For $n=1$, the Krylov subspace $\mathcal{K}_{1}= \operatorname{span}\left\{r_{G}^{(0)}\right\}=\operatorname{span}\left\{f_{C}^{(0)}\right\}$.
Assume that for $n=k+1$, the Krylov subspace $\mathcal{K}_{k+1}=\operatorname{span}\left\{f_{C}^{(0)}, \ldots, f_{C}^{(k)}\right\}$. Then, for $n=k+2$,
$$
\widetilde{f}_{C}^{(k+1)}=(I-A) f_{C}^{(k)}=f_{C}^{(k)}-A f_{C}^{(k)} \in \mathcal{K}_{k+2}
$$

Since $f_{C}^{(k+1)}$ is a linear combination of vectors $f_{C}^{(0)}, \ldots, f_{C}^{(k)}, \widetilde{f}_{C}^{(k+1)}$, thus $f_{C}^{(k+1)} \in \mathcal{K}_{k+2}$.
Finally, we are ready to prove Theorem 4.3.
Proof of Theorem 4.3. We show by induction that at step $k$ of CROP algorithm vectors $f_{C}^{(0)}, \ldots, f_{C}^{(k)}$ form Krylov subspace $\mathcal{K}_{k+1}$, i.e., $\mathcal{K}_{k+1}:=\operatorname{span}\left\{f_{C}^{(0)}, \ldots, f_{C}^{(k)}\right\}$. For $k=0$, Krylov subspace $\mathcal{K}_{1}=\operatorname{span}\left\{r_{G}^{(0)}\right\}=\operatorname{span}\left\{f_{C}^{(0)}\right\}$, which proves the basic case.
Assume that at step $k$ we have Krylov subspace $\mathcal{K}_{k+1}=\operatorname{span}\left\{f_{C}^{(0)}, \ldots, f_{C}^{(k)}\right\}$ as an induction hypothesis. Then at step $k+1$
$$
\widetilde{f}_{C}^{(k+1)}=(I-A) f_{C}^{(k)}=f_{C}^{(k)}-A f_{C}^{(k)} \in \mathcal{K}_{k+2}
$$

Since $f_{C}^{(k+1)}$ is a linear combination of vectors $f_{C}^{(0)}, \ldots, f_{C}^{(k)}, \widetilde{f}_{C}^{(k+1)}, f_{C}^{(k+1)} \in \mathcal{K}_{k+2}$. Moreover, we need to show that $f_{C}^{(k+1)} \notin \mathcal{K}_{k+1}$ which requires $\alpha_{C, m_{C}^{(k+1)}}^{(k+1)} \neq 0$ and $\widetilde{f}_{C}^{(k+1)} \notin \mathcal{K}_{k+1}$. These, however, are satisfied unless the algorithm stagnates. The equivalence of CROP algorithm and GMRES method follows directly from the fact that $f_{C}^{(k)}=\min _{v \in \mathcal{K}_{k+1}} v=r_{G}^{(k)}$.

Remark 4.5. Note that Theorem 4.3 can also be proved using Theorem 3.1 and the equivalence between Anderson Acceleration and GMRES method [44].

Next, we will show that, similarly to full CROP algorithm, there exists a truncated linear method equivalent to the truncated $\operatorname{CROP}(\mathrm{m})$ algorithm. In Theorem 4.3, we have shown the equivalence between CROP and GMRES method. Since GMRES is equivalent to Generalized Conjugate Residual (GCR) algorithm, let us consider the
truncated GCR, namely, the ORTHOMIN method [39, Section 6.9]). For general CROP $(m)$ method, the following theorem holds.

Theorem 4.6. Consider solving the linear system $A x=b$ by $\operatorname{CROP}(m)$ algorithm and ORTHOMIN $(m-1)$ method, under the following assumptions:
1. $A$ is nonsingular.
2. Run $\operatorname{CROP}(m)$ algorithm with $f(x)=b-A x$ and $g(x)=f(x)+x=b+(I-$ A) $x$.
3. CROP algorithm is truncated with parameter $m$, and ORTHOMIN with parameter $m-1$.
4. The initial values of $\operatorname{CROP}(m)$ and $\operatorname{ORTHOMIN}(m-1)$ coincide, i.e., $x_{O m i n(m-1)}^{(0)}=x_{C R O P(m)}^{(0)}$.

Then for $k>0, \quad x_{C R O P(m)}^{(k)}=x_{O m i n(m-1)}^{(k)} \quad$ and $\quad f_{C R O P(m)}^{(k)}=r_{O m i n(m-1)}^{(k)}$, with ORTHOMIN residuals $r_{\text {Omin }(m-1)}^{(k)}=b-A x^{(k)}$.

Proof. We proof Theorem 4.6 by induction on $m$.
Let us start with $m=1$. At step $k+1$ of $\operatorname{CROP}(1)$ iterate $x_{C}^{(k+1)}$ is determined directly from $x_{C}^{(k)}$ and $f_{C}^{(k)}$, hence making it a fixed-point iteration. According to (2.8), the control residuals in CROP(1) are given as
$$
\begin{equation*}
f_{C}^{(k+1)}=\alpha_{C, 0}^{(k+1)} f_{C}^{(k)}+\alpha_{C, 1}^{(k+1)} \widetilde{f}_{C}^{(k+1)} \tag{4.1}
\end{equation*}
$$

Since $\widetilde{f}_{C}^{(k+1)}=(I-A) f_{C}^{(k)}$ and $\alpha_{C, 0}^{(k+1)}=1-\alpha_{C, 1}^{(k+1)},(4.1)$ yields
$$
\begin{equation*}
f_{C}^{(k+1)}=\left(1-\alpha_{C, 1}^{(k+1)}\right) f_{C}^{(k)}+\alpha_{C, 1}^{k+1}(I-A) f_{C}^{(k)}=f_{C}^{(k)}-\alpha_{C, 1}^{(k+1)} A f_{C}^{(k)} . \tag{4.2}
\end{equation*}
$$

To obtain a solution of the least-squares problem (2.10) which minimizes $\left\|f_{C}^{(k+1)}\right\|_{2}$, we need $f_{C}^{(k+1)} \perp A f_{C}^{(k)}$ and thus
$$
\begin{equation*}
\alpha_{C, 1}^{(k+1)}=\frac{\left(f_{C}^{(k)}\right)^{T} A f_{C}^{(k)}}{\left(A f_{C}^{(k)}\right)^{T} A f_{C}^{(k)}} \tag{4.3}
\end{equation*}
$$

Therefore,
$$
\begin{equation*}
x_{C}^{(k+1)}=\left(1-\alpha_{C, 1}^{(k+1)}\right) x_{C}^{(k)}+\alpha_{C, 1}^{(k+1)}\left(x_{C}^{(k)}+f_{C}^{(k)}\right)=x_{C}^{(k)}+\alpha_{C, 1}^{(k+1)} f_{C}^{(k)} . \tag{4.4}
\end{equation*}
$$

Hence, iterates $x_{C}^{(k+1)}$ and residuals $f_{C}^{(k+1)}$ are updated according to (4.4) and (4.2), respectively, with $\alpha_{C, 1}^{(k+1)}$ computed as in (4.3). Moreover, they are exactly the same as those calculated by the ORTHOMIN( 0 ), i.e., $\operatorname{CROP}(1)=\operatorname{ORTHOMIN}(0)$.

Assume that $\operatorname{CROP}(\ell)=\operatorname{ORTHOMIN}(\ell-1)$ for $\ell=1, \ldots, m-1$. Then, we can prove $\operatorname{CROP}(m)=\operatorname{ORTHOMIN}(m-1)$ by induction on $k$.

Since $m_{C}^{(k+1)}=\min \{k+1, m\}$, for $k=0, m_{C}^{(k+1)}=1$ and $\operatorname{CROP}(m)=\operatorname{CROP}(1) =\operatorname{ORTHOMIN}(0)$, which is the same as the first step of ORTHOMIN $(m)$ for any $m>1$. For $k+1<m$, step $k+1$ of $\operatorname{CROP}(m)$ is the same as step $k+1$ of $\operatorname{CROP}(k+1)$, which is equivalent to step $k+1$ of $\operatorname{ORTHOMIN}(k)$ by the induction assumption, and thus is the same as step $k+1$ of ORTHOMIN $(m-1)$. Hence, we only need to consider the case of $k+1 \geq m$.

Assume that for some $k>0$ vectors $\left\{A \Delta x_{C}^{\left(k+1-m_{C}^{(k)}\right)}, \ldots, A \Delta x_{C}^{(k-1)}, f_{C}^{(k)}\right\}$ are pairwise orthogonal and $x_{C R O P(m)}^{(k)}=x_{O m i n(m-1)}^{(k)}$. Then, for general CROP (m), the $k+1$ residual is
$$
\begin{equation*}
f_{C}^{(k+1)}=\sum_{i=0}^{m-1} \alpha_{C, i}^{(k+1)} f_{C}^{(k+1-m)}+\alpha_{C, m}^{(k+1)} \widetilde{f}_{C}^{(k+1)} \tag{4.5}
\end{equation*}
$$

By Lemma 4.2 $\widetilde{f}_{C}^{(k+1)}=(I-A) f_{C}^{(k)}$. Then, with $\alpha_{C, 0}^{(k+1)}=\gamma_{C, 1}^{(k+1)}, \alpha_{C, i}^{(k+1)}=\gamma_{C, i+1}^{(k+1)}- \gamma_{C, i}^{(k+1)}$ and $\gamma_{C, m}^{(k+1)}=1-\alpha_{C, m}^{(k+1)}$, and noting that $\Delta f_{C}^{(k-m+i)}=-A \Delta x_{C}^{(k-m+i)}$, we get
$$
\begin{align*}
f_{C}^{(k+1)} & =\sum_{i=0}^{m} \alpha_{C, i}^{(k+1)} f_{C}^{(k+1-m)}+\alpha_{C, m}^{(k+1)}(I-A) f_{C}^{(k)} \\
& =f_{C}^{(k)}-\alpha_{C, m}^{k+1} A f_{C}^{(k)}-\sum_{i=1}^{m-1} \gamma_{C, i}^{(k+1)} A \Delta x_{C}^{(k-m+i)} \tag{4.6}
\end{align*}
$$

Since by the induction assumption $\left\{A \Delta x_{C}^{(k-m+1)}, \ldots, A \Delta x_{C}^{(k-1)}, f_{C}^{(k)}\right\}$ are pairwise orthogonal, if we orthogonalize $A f_{C}^{(k)}$ against all $A \Delta x_{C}^{(k-m+i)}$ for $i=1, \ldots, m-1$ and let
$$
\begin{equation*}
A p^{(k)}=A f_{C}^{(k)}-\sum_{i=1}^{m-1} \frac{\left(A f_{C}^{(k)}\right)^{T} A \Delta x_{C}^{(k-m+i)}}{\left(A \Delta x_{C}^{(k-m+i)}\right)^{T} A \Delta x_{C}^{(k-m+i)}} A \Delta x_{C}^{(k-m+i)} \tag{4.7}
\end{equation*}
$$
then (4.6) yields $f_{C}^{(k+1)}=f_{C}^{(k)}-\alpha_{C, m}^{k+1} A p^{(k)}+\sum_{i=1}^{m-1} c_{i} A \Delta x_{C}^{(k-m+i)}$. To obtain a solution of the least-squares problem (2.10) which minimizes $\left\|f_{C}^{(k+1)}\right\|_{2}$, we need $f_{C}^{(k+1)} \perp A p^{(k)}$ and $f_{C}^{(k+1)} \perp A \Delta x_{C}^{(k-m+i)}$ for all $i=1, \ldots, m-1$. Thus
$$
\begin{equation*}
\alpha_{C, m}^{(k+1)}=\frac{\left(f_{C}^{(k)}\right)^{T} A p^{(k)}}{\left(A p^{(k)}\right)^{T} A p^{(k)}} \tag{4.8}
\end{equation*}
$$
and $c_{i}=0$ for all $i=1, \ldots, m-1$. The actual updates of $x_{C}^{(k+1)}$ and $f_{C}^{(k+1)}$ are
$$
\begin{equation*}
x_{C}^{(k+1)}=x_{C}^{(k)}+\alpha_{C, m}^{(k+1)} p^{(k)} \quad \text { and } \quad f_{C}^{(k+1)}=f_{C}^{(k)}-\alpha_{C, m}^{(k+1)} A p^{(k)} . \tag{4.9}
\end{equation*}
$$

Since $\Delta x_{C}^{(k)}=\alpha_{C, m}^{(k+1)} p^{(k)}$ is parallel to $p^{(k)}$, we can rewrite (4.7) by replacing $\Delta x_{C}^{(k-m+i)}$ with $p^{(k-m+i)}$. Hence, the orthogonalization step (4.7) and the update formulas (4.9) are the same as those in ORTHOMIN $(m-1)$ method. Therefore, in the case of solving linear system, the $\operatorname{CROP}(m)$ algorithm is equivalent to ORTHOMIN $(m-1)$.

Remark 4.7. Since for symmetric matrix $A$, ORTHOMIN(0) is Minimal Residual (MINRES) method [39, Section 5.3.2], and ORTHOMIN(1) is the CR method, for symmetric linear systems $\operatorname{CROP}(1)$ is equivalent to MINRES and $\operatorname{CROP}(2)$ to the CR method.
4.2. Connection with Multisecant Methods. Generalized Broyden's second method is a multisecant method [25] equivalent to Anderson Acceleration method [24, 25] with iterates generated according to the update formula
$$
\begin{equation*}
x^{(k+1)}=x^{(k)}-G^{(k)} f^{(k)} \tag{4.10}
\end{equation*}
$$
where $G^{(k)}$ is approximated inverse of the Jacobian updated by
$$
\begin{equation*}
G^{(k)}=G^{(k-m)}+\left(\mathscr{X}^{(k)}-G^{(k-m)} \mathscr{F}^{(k)}\right)\left[\left(\mathscr{F}^{(k)}\right)^{T} \mathscr{F}^{(k)}\right]^{-1}\left(\mathscr{F}^{(k)}\right)^{T}, \tag{4.11}
\end{equation*}
$$
which minimizes $\left\|G^{(k)}-G^{(k-m)}\right\|_{F}$ subject to $G^{(k)} \mathscr{F}^{(k)}=\mathscr{X}^{(k)}$. Here, $\mathscr{X}^{(k)}$ and $\mathscr{F}^{(k)}$ represent the differences of iterates and residuals, respectively, i.e.,
$$
\mathscr{X}^{(k)}=\left[\Delta x^{(k-m)}, \ldots, \Delta x^{(k-1)}\right] \quad \text { and } \quad \mathscr{F}^{(k)}=\left[\Delta f^{(k-m)}, \ldots, \Delta f^{(k-1)}\right] .
$$

Therefore, the update formula (4.10) can be written as
$$
\begin{equation*}
x^{(k+1)}=x^{(k)}-G^{(k-m)} f^{(k)}-\left(\mathscr{X}^{(k)}-G^{(k-m)} \mathscr{F}^{(k)}\right)\left[\left(\mathscr{F}^{(k)}\right)^{T} \mathscr{F}^{(k)}\right]^{-1}\left(\mathscr{F}^{(k)}\right)^{T} f^{(k)} . \tag{4.12}
\end{equation*}
$$

When $G^{(k-m)}=-\beta I, \mathscr{X}^{(k)}=\mathscr{X}_{A}^{(k)}$ and $\mathscr{F}^{(k)}=\mathscr{F}_{A}^{(k)}$, (4.12) reduces to (2.14). Anderson Acceleration forms an approximate inverse of the Jacobian implicitly
$$
G_{A}^{(k)}=-\beta I+\left(\mathscr{X}^{(k)}+\beta \mathscr{F}_{A}^{(k)}\right)\left[\left(\mathscr{F}_{A}^{(k)}\right)^{T} \mathscr{F}_{A}^{(k)}\right]^{-1}\left(\mathscr{F}_{A}^{(k)}\right)^{T},
$$
that minimizes $\left\|G_{A}^{(k)}+\beta I\right\|_{F}$ subject to $G_{A}^{(k)} \mathscr{F}_{A}^{(k)}=\mathscr{X}_{A}^{(k)}$. Following the same notation, the updates of CROP algorithm can be written as
$$
\begin{equation*}
x_{C}^{(k+1)}=\widetilde{x}_{C}^{(k+1)}-\mathscr{X}_{C}^{(k+1)}\left[\left(\mathscr{F}_{C}^{(k+1)}\right)^{T} \mathscr{F}_{C}^{(k+1)}\right]^{-1}\left(\mathscr{F}_{C}^{(k+1)}\right)^{T} \widetilde{f}_{C}^{(k+1)} \tag{4.13}
\end{equation*}
$$

With a common framework in place, we can now describe the connection between CROP algorithm and multisecant methods. First, if we consider $\widetilde{x}_{C}^{(k)}$ as an iterate, then we can view (4.13) as an update of a generalized Broyden's second method. When $G^{(k-m)}=0, \mathscr{X}^{(k)}=\mathscr{X}_{C}^{(k)}$ and $\mathscr{F}^{(k)}=\mathscr{F}_{C}^{(k)}$, (4.12) reduces to (4.13). CROP algorithm forms implicitly an approximate inverse of the Jacobian
$$
G_{C}^{(k+1)}=\mathscr{X}_{C}^{(k+1)}\left[\left(\mathscr{F}_{C}^{(k+1)}\right)^{T} \mathscr{F}_{C}^{(k+1)}\right]^{-1}\left(\mathscr{F}_{C}^{(k+1)}\right)^{T},
$$
that minimizes $\left\|G_{C}^{(k+1)}\right\|_{F}$ subject to $G_{C}^{(k+1)} \mathscr{F}_{C}^{(k+1)}=\mathscr{X}_{C}^{(k+1)}$.
5. Convergence Theory for CROP Algorithm. In this section, we provide some initial results on the convergence of CROP algorithm. The first convergence result for Anderson Acceleration method was given in [43], under the assumption of a contraction mapping. Following this work, several other convergence results utilizing various different assumptions were established, see for example [17, 23, 32, 34, 37]. Most of the existing assumptions are needed to determine the existence and uniqueness of the exact solution $x^{*}$ of $f(x)=0$ in an open set. Mappings $f$ and $g$ are usually chosen to be Lipschitz continuous. In this section, we use the same assumptions and follow the same process as the one in [43]. We first prove in Subsection 5.1 that for the linear case CROP algorithm is q-linearly convergent. Then, in Subsection 5.2, we show that for the general nonlinear case, the convergence is r-linear.
5.1. Convergence of CROP Algorithm for Linear Problems. Let us consider applying the CROP algorithm to the simple linear problem $A x=b$, with a nonsingular matrix $A \in \mathbb{R}^{n \times n}$ and vector $b \in \mathbb{R}^{n}$. Then, the associated residual (error) vectors can be chosen as $f(x)=b-A x$, with the corresponding $g(x)=b+(I-A) x$.

Let us present our first convergence result.
Theorem 5.1. Let us consider solving the linear system $A x=b$. If $\|I-A\|= c<1$, then CROP algorithm converges to the exact solution $x^{*}=A^{-1} b$, and the control and real residuals converge $q$-linearly to zero with the $q$-factor $c$.

Proof. Since $f_{C}^{(k+1)}=\sum_{i=0}^{m_{C}^{(k+1)}-1} \alpha_{C, i}^{(k+1)} f_{C}^{\left(k+1-m_{C}^{(k+1)}+i\right)}+\alpha_{C, m_{C}^{(k+1)}}^{(k+1)} \widetilde{f}_{C}^{(k+1)}$ is the least-squares residual corresponding to (2.10), we have
$$
\left\|f_{C}^{(k+1)}\right\|_{2} \leq\left\|\widetilde{f}_{C}^{(k+1)}\right\|_{2}
$$

Moreover, by Lemma $4.2 \widetilde{f}_{C}^{(k+1)}=(I-A) f_{C}^{(k)}$. Finally by Lemma 4.1, we obtain
$$
\left\|f_{C}^{(k+1)}\right\|_{2} \leq\left\|\widetilde{f}_{C}^{(k+1)}\right\|_{2} \leq c\left\|f_{C}^{(k)}\right\|_{2} \quad \text { and } \quad\left\|r_{C}^{(k+1)}\right\|_{2} \leq c\left\|r_{C}^{(k)}\right\|_{2} .
$$

Now, if we set $e^{(k)}=x^{(k)}-x^{*}$, then $f\left(x^{(k)}\right)=b-A x^{(k)}=A\left(x^{*}-x^{(k)}\right)=-A e^{(k)}$. Consequently, following Theorem 5.1 yields
$$
(1-c)\left\|e^{(k)}\right\|_{2} \leq\left\|f\left(x^{(k)}\right)\right\|_{2} \leq c^{k}\left\|f\left(x^{(0)}\right)\right\|_{2} \leq c^{k}(1+c)\left\|e^{(0)}\right\|_{2}
$$

Hence,
$$
\left\|e^{(k)}\right\|_{2} \leq\left(\frac{1+c}{1-c}\right) c^{k}\left\|e^{(0)}\right\|_{2}
$$

To fully understand the convergence of CROP method for linear problems, we refer to the equivalence of CROP and GMRES, and CROP $(m)$ and ORTHOMIN $(m-1)$.

Remark 5.2. The convergence of ORTHOMIN is shown in [21]. Note that CROP algorithm with no truncation, ORTHOMIN and GMRES are all mathematically equivalent methods.
5.2. Convergence of the CROP Algorithm for Nonlinear Problems. In the case of nonlinear problems, investigating the convergence of fixed-depth CROP Algorithm 2.2 or $\operatorname{CROP}(m)$ algorithm, requires an assumption of the functions $f$ and $g$ to be good enough. Let us consider the following assumption.

Assumption 5.3. Consider a nonlinear problem (1.1) such that
1. there exists an $x^{*}$ such that $f\left(x^{*}\right)=0$ and $g\left(x^{*}\right)=x^{*}$,
2. function $g$ is Lipschitz continuously differentiable in the ball $\mathcal{B}_{\widehat{\rho}}\left(x^{*}\right)$ for some $\widehat{\rho}>0$ with Lipschitz constant $c \in(0,1)$, and
3. $f^{\prime}$ is Lipschitz continuous with a Lipschitz constant $L$.

In the case of no truncation, equivalence between CROP algorithm and Anderson Acceleration method was established in Theorem 3.1. For truncated CROP $(m)$ algorithm, we can still try to explore the relation between the two methods. Let us investigate the connection between $\widetilde{x}_{C}$ and $x_{C}, \widetilde{f}_{C}$ and $f_{C}$. Note that when $m \neq k$, $\widetilde{x}_{C}$ and $\widetilde{f}_{C}$ are not the iterates and residuals of Anderson Acceleration method.

For $\operatorname{CROP}(m)$ method, we have
$$
X_{C}^{(k)}=\widetilde{X}_{C}^{(k)} A_{0} A_{1} \cdots A_{k} \quad \text { and } \quad F_{C}^{(k)}=\widetilde{F}_{C}^{(k)} A_{0} A_{1} \cdots A_{k},
$$
with
$$
X_{C}^{(k)}=\left[\begin{array}{lll}
x_{C}^{(0)} & \cdots & x_{C}^{(k)}
\end{array}\right] \quad \text { and } \quad F_{C}^{(k)}=\left[\begin{array}{lll}
f_{C}^{(0)} & \cdots & f_{C}^{(k)}
\end{array}\right]
$$
and $A_{i}$ is the identity matrix with the ( $i+1$ )-th column changed to the coefficient vector $\alpha_{C}^{(i)}=\left[\alpha_{C, 0}^{(i)}, \ldots, \alpha_{C, m_{C}^{(i)}}^{(i)}\right]^{T}$ inserted in range of rows starting at index ( $i- \left.m_{C}^{(i)}+1\right)$ and ending at index $(i+1)$, i.e.,
$$
A_{0}=I, A_{1}=\left[\begin{array}{cccc}
1 & \alpha_{C, 0}^{(1)} & 0 & \ldots \\
0 & \alpha_{C, 1}^{(1)} & 0 & \ldots \\
0 & 0 & 1 & \ldots \\
\vdots & \vdots & \vdots & \ddots
\end{array}\right], A_{k}=\left[\begin{array}{ccccc}
1 & 0 & \cdots & 0 & 0 \\
\vdots & \ddots & \cdots & \vdots & \vdots \\
0 & 0 & \cdots & 0 & \alpha_{C, 0}^{(k)} \\
\vdots & \vdots & \vdots & \vdots & \vdots \\
0 & 0 & \cdots & 1 & \alpha_{C, m_{C}^{(k)}-1}^{(k)} \\
0 & 0 & \cdots & 0 & \alpha_{C, m_{C}^{(k)}}^{(k)}
\end{array}\right], k=0,1, \ldots
$$

Since matrices $A_{i}$ have column sum $1, x_{C}^{(k)}$ and $f_{C}^{(k)}$, as the last column of $X_{C}^{(k)}$ and $F_{C}^{(k)}$, respectively, can be written as
$$
x_{C}^{(k)}=\sum_{j=0}^{k} s_{j}^{(k)} \widetilde{x}_{C}^{(k)} \quad \text { and } \quad f_{C}^{(k)}=\sum_{j=0}^{k} s_{j}^{(k)} \widetilde{f}_{C}^{(k)}, \quad \text { with } \sum_{j=0}^{k} s_{j}^{(k)}=1
$$

Now, we impose the following assumption that will allow us to establish the convergence result for $\operatorname{CROP}(m)$ along the lines of [43, Theorem 2.3].

Assumption 5.4. For all $k>0$, there exists $M>0$ such that $\sum_{j=0}^{k}\left|s_{j}^{(k)}\right|<M$.
Let us consider Assumption 5.3 again. From the Lipschitz condition for function $g,\left\|g(x)-g\left(x^{*}\right)\right\|_{2} \leq c\left\|x-x^{*}\right\|_{2}$ or equivalently $\left\|f(x)-\left(x-x^{*}\right)\right\|_{2} \leq c\left\|x-x^{*}\right\|_{2}$. By the triangle inequality we get
$$
\begin{equation*}
(1-c)\left\|x-x^{*}\right\|_{2} \leq\|f(x)\|_{2} \leq(1+c)\left\|x-x^{*}\right\|_{2} \tag{5.1}
\end{equation*}
$$

Moreover, by the Lipschitz condition on $f^{\prime}$, there exists $\rho>0$ sufficiently small such that in the ball $\mathcal{B}_{\rho}\left(x^{*}\right)$ we have
$$
\begin{equation*}
\left\|f(x)-f^{\prime}\left(x^{*}\right)(x-x *)\right\|_{2} \leq \frac{L}{2}\left\|x-x^{*}\right\|_{2}^{2} \tag{5.2}
\end{equation*}
$$
which can be written in terms of $g$, as
$$
\begin{equation*}
\left\|g(x)-g^{\prime}\left(x^{*}\right)(x-x *)-x^{*}\right\|_{2} \leq \frac{L}{2}\left\|x-x^{*}\right\|_{2}^{2} \tag{5.3}
\end{equation*}
$$

Theorem 5.5. Let Assumption 5.3 and Assumption 5.4 hold and let $c<\widehat{c}<1$. Then, if $x_{C, m}^{(0)}$ is sufficiently close to $x^{*}$, the control residual $f_{C}^{(k)}$ of $\operatorname{CROP}(m)$ and CROP-Anderson(m) algorithm and the residual $f_{C A}^{(k)}=\widetilde{f}_{C}^{(k)}$ satisfy
$$
\left\|f_{C}^{(k)}\right\|_{2} \leq \widehat{c}^{k}\left\|f_{C}^{(0)}\right\|_{2} \quad \text { and } \quad\left\|f_{C A}^{(k)}\right\|_{2} \leq \widehat{c}^{k}\left\|f_{C A}^{(0)}\right\|_{2}
$$

Moreover, this implies $r$-linear convergence of $\operatorname{CROP}(m)$ and $\operatorname{CROP}-\operatorname{Anderson}(m)$ algorithm with $r$-factor no greater than $\widehat{c}$.

Proof. Since $\lim _{\rho \rightarrow 0} \frac{2(1-c) c \hat{c}^{k}+M L \rho}{2(1-c)-L \rho}=c \hat{c}^{k}<\hat{c}^{k+1}$, we can choose $\rho$ small enough such that $\frac{2(1-c) c \hat{c}^{k}+M L \rho}{2(1-c)-L \rho} \leq \hat{c}^{k+1}$. Suppose $x_{C}^{(0)} \in \mathcal{B}_{\rho}\left(x^{*}\right)$, and $x_{C}^{(0)}$ is sufficiently close to $x^{*}$ such that
$$
\frac{M(c+L \rho / 2)}{1-c}\left\|f\left(x_{C}^{(0)}\right)\right\|_{2} \leq \frac{M(1+c)(c+L \rho / 2)}{1-c}\left\|x_{C}^{(0)}-x^{*}\right\|_{2} \leq \rho
$$

Then, we can prove by induction on $k$ that $\left\|f\left(\widetilde{x}_{C}^{(k)}\right)\right\|_{2} \leq \widehat{c}^{k}\left\|f\left(x_{C}^{(0)}\right)\right\|_{2}$ and $\widetilde{x}_{C}^{(k)} \in \mathcal{B}_{\rho}\left(x^{*}\right)$ for all $k \geq 0$. For $k=0$, the conditions are satisfied because $\widetilde{x}_{C}^{(0)}=x_{C}^{(0)}$.

Now, assume that for all $n \leq k$, we have $\left\|f\left(\widetilde{x}_{C}^{(n)}\right)\right\|_{2} \leq \widehat{c}^{k}\left\|f\left(x_{C}^{(0)}\right)\right\|_{2}$ and $\widetilde{x}_{C}^{(n)} \in \mathcal{B}_{\rho}\left(x^{*}\right)$. By (5.3) $g\left(\widetilde{x}_{C}^{(n)}\right)=x^{*}+g^{\prime}\left(x^{*}\right)\left(\widetilde{x}_{C}^{(n)}-x^{*}\right)+\Delta^{(n)}$, with $\left\|\Delta^{(n)}\right\|_{2} \leq \frac{L}{2}\left\|\widetilde{x}_{C}^{(n)}-x^{*}\right\|_{2}$. Then, for $n=k+1$,
$$
\begin{aligned}
\widetilde{x}_{C}^{(k+1)} & =x^{*}+\sum_{j=0}^{k} s_{j}^{(k)}\left(g^{\prime}\left(x^{*}\right)\left(\widetilde{x}_{C}^{(j)}-x^{*}\right)+\Delta^{(j)}\right) \\
& =x^{*}+\sum_{j=0}^{k} s_{j}^{(k)} g^{\prime}\left(x^{*}\right)\left(\widetilde{x}_{C}^{(j)}-x^{*}\right)+\sum_{j=0}^{k} s_{j}^{(k)} \Delta^{(j)}
\end{aligned}
$$

Since $\left\|\sum_{j=0}^{k} s_{j}^{(k)} \Delta^{(j)}\right\|_{2}=\sum_{j=0}^{m_{C}^{(k)}}\left|s_{j}^{(k)}\right| \frac{L}{2}\left\|\widetilde{x}_{C}^{(j)}-x^{*}\right\|_{2}^{2}$, following (5.1) we get
$$
\left\|\widetilde{x}_{C}^{(j)}-x^{*}\right\|_{2}<\frac{1}{1-c}\left\|f\left(\widetilde{x}_{C}^{(j)}\right)\right\|_{2} \leq \frac{1}{1-c}\left\|f\left(x_{C}^{(0)}\right)\right\|_{2}
$$

Then $\left\|\widetilde{x}_{C}^{(j)}-x^{*}\right\|_{2}<\rho$ yields $\left\|\widetilde{x}_{C}^{(j)}-x^{*}\right\|_{2}^{2} \leq \frac{\rho}{1-c}\left\|f\left(x_{C}^{(0)}\right)\right\|_{2}$. Under Assumption 5.4, we have
$$
\begin{aligned}
\left\|\widetilde{x}_{C}^{(k+1)}-x^{*}\right\|_{2} & =\left\|\sum_{j=0}^{k} s_{j}^{(k)} g^{\prime}\left(x^{*}\right)\left(\widetilde{x}_{C}^{(j)}-x^{*}\right)+\sum_{j=0}^{k} s_{j}^{(k)} \Delta^{(j)}\right\|_{2} \\
& \leq \frac{M(c+L \rho / 2)}{1-c}\left\|f\left(x_{C}^{(0)}\right)\right\|_{2} \leq \rho
\end{aligned}
$$

Thus, $\widetilde{x}_{C}^{(k+1)} \in \mathcal{B}_{\rho}\left(x^{*}\right)$ and by (5.2) we obtain $f\left(\widetilde{x}_{C}^{(k+1)}\right)=f^{\prime}\left(x^{*}\right)\left(\widetilde{x}_{C}^{(k+1)}-x^{*}\right)+ \Delta^{(k+1)}$, with $\left\|\Delta^{(k+1)}\right\|_{2} \leq \frac{L}{2}\left\|\widetilde{x}_{C}^{(k+1)}-x^{*}\right\|_{2}^{2} \leq \frac{L \rho}{2(1-c)}\left\|f\left(\widetilde{x}_{C}^{(k+1)}\right)\right\|_{2}$. Since $f^{\prime}\left(x^{*}\right)=$
$g^{\prime}\left(x^{*}\right)-I$ and $g^{\prime}\left(x^{*}\right)$ commute,
$$
\begin{aligned}
f\left(\widetilde{x}_{C}^{(k+1)}\right) & =f^{\prime}\left(x^{*}\right) \sum_{j=0}^{k} s_{j}^{(k)} g^{\prime}\left(x^{*}\right)\left(\widetilde{x}_{C}^{(j)}-x^{*}\right)+f^{\prime}\left(x^{*}\right) \sum_{j=0}^{k} s_{j}^{(k)} \Delta^{(j)}+\Delta^{(k+1)} \\
& =g^{\prime}\left(x^{*}\right) \sum_{j=0}^{k} s_{j}^{(k)} f^{\prime}\left(x^{*}\right)\left(\widetilde{x}_{C}^{(j)}-x^{*}\right)+f^{\prime}\left(x^{*}\right) \sum_{j=0}^{k} s_{j}^{(k)} \Delta^{(j)}+\Delta^{(k+1)} \\
& =g^{\prime}\left(x^{*}\right) \sum_{j=0}^{k} s_{j}^{(k)}\left(f\left(\widetilde{x}_{C}^{(k)}\right)-\Delta^{(j)}\right)+f^{\prime}\left(x^{*}\right) \sum_{j=0}^{k} s_{j}^{(k)} \Delta^{(j)}+\Delta^{(k+1)} \\
& =g^{\prime}\left(x^{*}\right) \sum_{j=0}^{k} s_{j}^{(k)} f\left(\widetilde{x}_{C}^{(j)}\right)-\sum_{j=0}^{k} s_{j}^{(k)} \Delta^{(j)}+\Delta^{(k+1)} \\
& =g^{\prime}\left(x^{*}\right) f_{C}^{(k+1)}-\sum_{j=0}^{k} s_{j}^{(k)} \Delta^{(j)}+\Delta^{(k+1)}
\end{aligned}
$$

Now, from $\left\|f_{C}^{(k)}\right\|_{2} \leq\left\|\widetilde{f}_{C}^{(k)}\right\|_{2} \leq \widehat{c}^{k}\left\|f\left(x_{C}^{(0)}\right)\right\|_{2}$, we have
$$
\begin{aligned}
\left\|f\left(\widetilde{x}_{C}^{(k+1)}\right)\right\|_{2} & \leq\left\|g^{\prime}\left(x^{*}\right) f_{C}^{(k)}\right\|+\left\|\sum_{j=0}^{k} s_{j}^{(k)} \Delta^{(j)}\right\|+\left\|\Delta^{(k+1)}\right\|_{2} \\
& \leq c \widehat{c}^{k}\left\|f\left(x_{C}^{(0)}\right)\right\|+\frac{M L \rho}{2(1-c)}\left\|f\left(x_{C}^{(0)}\right)\right\|_{2}+\frac{L \rho}{2(1-c)}\left\|f\left(\widetilde{x}_{C}^{(k+1)}\right)\right\|_{2}
\end{aligned}
$$

Thus $\left\|f\left(\widetilde{x}_{C}^{(k+1)}\right)\right\|_{2} \leq \frac{2(1-c) c \hat{c}^{k}+M L \rho}{2(1-c)-L \rho}\left\|f\left(x_{C}^{(0)}\right)\right\|_{2} \leq \widehat{c}^{k+1}\left\|f\left(x_{C}^{(0)}\right)\right\|_{2}$.
Although the control residuals $f_{C}^{(k)}$ are not the real residuals $r_{C}^{(k)}$, which may cause a breakdown $\left(f_{C}^{(k)}=0\right)$ without finding a reasonable approximation, they do have some good properties.

THEOREM 5.6. The control residuals $f_{C}^{(k)}$ of $\operatorname{CROP}(m)$ for $m \geq 1$ are nonincreasing.

Proof. Since the coefficients $\alpha_{C, i}^{(k+1)}$ in
$$
\begin{equation*}
f_{C}^{(k+1)}=\sum_{i=0}^{m_{C}^{(k+1)}-1} \alpha_{C, i}^{(k+1)} f_{C}^{\left(k+1-m_{C}^{(k+1)}+i\right)}+\alpha_{C, m_{C}^{(k+1)}}^{(k+1)} \widetilde{f}_{C}^{(k+1)} \tag{5.4}
\end{equation*}
$$
are chosen to minimize $\left\|f_{C}^{(k+1)}\right\|$, and $f_{C}^{(k)}$ is itself an element in the sum on the right of (5.4), we must have $\left\|f_{C}^{(k+1)}\right\| \leq\left\|f_{C}^{(k)}\right\|$.
In the case of smaller values of $m$, the control residuals are approximating better the real residuals. Also, using CROP-Anderson or rCROP algorithm is a good way to avoid the breakdown. We will see some examples in Section 6.
6. Numerical Experiments. In this section, we present a small selection of numerical results illustrating some observations discussed in the previous sections. We consider both linear and nonlinear problems of various sizes. Details regarding implementation and some further numerical examples can be found in Supplementary Materials, Section SM. 1 and SM.2.
6.1. Linear Problems. First, we present numerical experiments to demonstrate the behavior of discussed methods when applied to a linear system $A x=b$ with a nonsingular matrix $A$.

Example 1 (Synthetic Problem).
Let us consider a nonsingular matrix $A \in \mathbb{R}^{100 \times 100}$ given as a tridiagonal matrix with entries $(1,-4,1)$ and a seven-diagonal matrix with entries $(0,0,1,-4,1,1,1)$, and vector $b \in \mathbb{R}^{100}$ with its first entry 1 and others 0 . We choose $f(x)=b-A x$, $g(x)=x+f(x)$, maxit $=100$, tol $=10^{-10}$, and run Anderson Acceleration method, CROP and CROP-Anderson algorithm with initial vector $x_{0}=0$ and different values of parameter $m$. The corresponding results are shown in Figure 3. In Figure 3a

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_08_2ce2904da8f9b738a883g-22.jpg?height=505&width=1253&top_left_y=859&top_left_x=238}
\captionsetup{labelformat=empty}
\caption{Fig. 3: A linear problem in Example (1) with (a) a tridiagonal and (b) a sevendiagonal matrix $A$.}
\end{figure}
we can see that convergence curve of CROP-Anderson method is parallel to the one of CROP algorithm. As we discussed in Subsection 5.1, CROP algorithm, CROP(2) and GMRES admit the same convergence similarly as Anderson Acceleration method, CROP-Anderson and CROP-Anderson(2) algorithm. Analogous behavior can be observed for CROP, CROP(4) and GMRES method, as well as for Anderson Acceleration method, CROP-Anderson algorithm and CROP-Anderson(4), see Figure 3b. More linear examples with matrices of different sizes and structures can be found in Supplementary Materials, Section SM.2.
6.2. Nonlinear Problems. As presented methods are primarily used to accelerate convergence in the case of nonlinear problems, in this section we introduce a variety of nonlinear examples. Starting with small and weakly nonlinear problems, we move towards more complex examples.

Example 2 (Dominant Linear Part Problem).
Consider a nonlinear problem $A x+\frac{\mu\|x\|^{2}}{n} x=b$ with $A=\operatorname{tridiag}(1,-4,1) \in \mathbb{R}^{n \times n}$, a right-hand side vector $b \in \mathbb{R}^{n}$ with first entry 1 and others $0, n=100$ and parameter $\mu=1 / 100$. We choose $f(x)=A x+\frac{\mu\|x\|^{2}}{n} x-b$ and $g(x)=x+f(x)$, parameters maxit $=100$ and tol $=10^{-10}$, and run Anderson Acceleration method,

CROP algorithm and CROP-Anderson method with initial vector $x^{(0)}=0$ and different values of parameter $m$. Figure 4 presents our findings. We can see that the

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_08_2ce2904da8f9b738a883g-23.jpg?height=424&width=581&top_left_y=455&top_left_x=577}
\captionsetup{labelformat=empty}
\caption{Fig. 4: Convergence for a nonlinear problem in Example (2).}
\end{figure}
first several steps of all methods are controlled by the linear part of the problem and can be compared with the results displayed in Figure 3a. The difference occurs when the residual reaches $\approx 10^{-6}$. We can see that $\operatorname{CROP}$ algorithm, $\operatorname{CROP}(2)$, and CROP-Anderson(2) method converge in 18, 19 and 21 steps, respectively, which is better than the other methods. It is worth mentioning that for CROP algorithm, although the control residuals get below the tolerance, the real residuals may not. Actually, in this example, $\left\|r_{C R O P}^{(18)}\right\|_{2}=6.28 \cdot 10^{-8},\left\|r_{C R O P(2)}^{(19)}\right\|_{2}=9.56 \cdot 10^{-11}$ and $\left\|r_{C R O P(1)}^{(32)}\right\|_{2}=5.19 \cdot 10^{-11}$. In the case of CROP algorithm with small values of $m$, the control residuals are good approximations of the real residuals. However, this does not have to be the case for the larger values of $m$. We will see some further disadvantages of the control residuals in Example (3).

Example 3 (A Small Nonlinear Problem).
Consider problem (1.1) [18, 19, Problem 2] with
$$
g\left(\left[\begin{array}{l}
x_{1} \\
x_{2}
\end{array}\right]\right)=\frac{1}{2}\left[\begin{array}{c}
x_{1}+x_{1}^{2}+x_{2}^{2} \\
x_{2}+x_{1}^{2}
\end{array}\right] \quad \text { and exact solution } \quad\left[\begin{array}{l}
x_{1}^{*} \\
x_{2}^{*}
\end{array}\right]=\left[\begin{array}{l}
0 \\
0
\end{array}\right] .
$$

We set maxit $=100$, tol $=10^{-10}$ and run fixed-point iteration, Anderson Acceleration method, CROP algorithm and CROP-Anderson method with $x^{(0)}=[0.1,0.1]^{T}$ and different values of parameter $m$. Note that all variants of CROP algorithm and CROP-Anderson method, except the full CROP-Anderson without truncation ( $m=$ maxit), are bad, see Figure 5. CROP algorithm and CROP(2) method break down in iteration 2, but rCROP and rCROP-Anderson converge well. rCROP(1) and rCROP(2) method converge in 4 iterations, while Anderson(2) method converges in 9 and the Anderson(1) method in 32 iterations. Even though using real residuals requires additional function evaluation in each iteration, Example 3 illustrates that this extra cost is worthy as it reduces the number of iterations from 32 in the case of Anderson Acceleration method to 8 steps for rCROP algorithm.

Example 4 (Bratu Problem).
In this example, we consider the Bratu Problem [26, Section 5.1]
$$
\begin{align*}
\Delta u+\lambda e^{u} & =0 \text { in } \Omega=(0,1) \times(0,1)  \tag{6.1}\\
u(x, y) & =0 \text { for }(x, y) \in \partial \Omega
\end{align*} .
$$

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_08_2ce2904da8f9b738a883g-24.jpg?height=499&width=1243&top_left_y=374&top_left_x=243}
\captionsetup{labelformat=empty}
\caption{Fig. 5: A small nonlinear problem in Example (3) with (a) control residuals and (b) real residuals.}
\end{figure}

Using finite difference method with grid size $100 \times 100$, the problem becomes $L x+ h^{2} \lambda \exp (x)=0$, where $L$ is the $10000 \times 100002 \mathrm{D}$ Laplace matrix and $h=1 / 101$. We choose $\lambda=0.5, f(x)=L x+h^{2} \lambda \exp (x), g(x)=x+f(x)$ as well as parameters maxit $=$ 400, tol $=10^{-10}, x^{(0)}=0$. We run Algorithm 2.1, Algorithm 2.2 and Algorithm 3.1 with $m=\infty, 1,2$. We also compare these methods with nlTGCR method introduced in [26]. nlTGCR method is an extension of the Generalized Conjugate Residual (GCR) [21] method to nonlinear problems by changing matrix $A$ to the Jacobian $-\Delta f$ in each step of the algorithm. Since the Bratu problem (6.1) is symmetric and has small nonlinearities, the rCROP results are almost the same as those of CROP algorithm, and confirm that $m=2$ is a good choice of the truncation parameter.

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_08_2ce2904da8f9b738a883g-24.jpg?height=573&width=1207&top_left_y=1653&top_left_x=260}
\captionsetup{labelformat=empty}
\caption{Fig. 6: Bratu problem from Example (4) with (a) control residuals and (b) real residuals.}
\end{figure}
7. Conclusions. Based on the initial convergence analysis and experiments presented in this paper, CROP algorithm emerges as an interesting approach for linear and nonlinear problems with weak nonlinearities. Although for highly nonlinear problems CROP algorithm often fails to converge, its variant rCROP can behave even better than Anderson Acceleration method. Further theoretical and numerical studies of CROP algorithm and its variants, in particular for challenging large scale computational chemistry problems, are the subject of an ongoing work.

Acknowledgments. The authors would like to thank Eric de Sturler, Tom Werner and Mark Embree for insightful discussions and helpful suggestions to this project. This work was supported by the National Science Foundation through the awards DMS-2144181 and DMS-2324958.

\section*{REFERENCES}
[1] A. C. Aitken, On Bernoulli's Numerical Solution of Algebraic Equations, Proc. R. Soc. Edinb., 46 (1926), pp. 289-305.
[2] D. G. Anderson, Iterative procedures for nonlinear integral equations, J. ACM, 12 (1965), pp. 547-560.
[3] D. G. M. Anderson, Comments on "Anderson acceleration, mixing and extrapolation", Numer. Algorithms, 80 (2019), pp. 135-234.
[4] A. S. Banerjee, P. Suryanarayana, and J. E. Pask, Periodic Pulay method for robust and efficient convergence acceleration of self-consistent field iterations, Chem. Phys. Lett., 647 (2016), pp. 31-35.
[5] C. Brezinski, Application de l'ɛ-algorithme à la résolution des systèmes non linéaires, C . R . Acad. Sci. Paris Sér. A-B, 271 (1970), pp. A1174-A1177.
[6] C. Brezinski, Convergence acceleration during the 20th century, in Numerical analysis 2000, Vol. II: Interpolation and extrapolation, vol. 122 of J. Comput. Appl. Math., 2000, pp. 1-21.
[7] C. Brezinski, S. Cipolla, M. Redivo-Zaglia, and Y. Saad, Shanks and Anderson-type acceleration techniques for systems of nonlinear equations, IMA J. Numer. Anal., 42 (2022), pp. 3058-3093.
[8] C. Brezinski and M. Redivo-Zaglia, The simplified topological $\varepsilon$-algorithms for accelerating sequences in a vector space, SIAM J. Sci. Comput., 36 (2014), pp. A2227-A2247.
[9] C. Brezinski and M. Redivo-Zaglia, Extrapolation and rational approximation, The Works of the Main Contributors, Springer Nature, Cham, Switzerland, (2020).
[10] C. Brezinski, M. Redivo-Zaglia, and Y. Saad, Shanks sequence transformations and Anderson acceleration, SIAM Rev., 60 (2018), pp. 646-669.
[11] S. Cabay and L. W. Jackson, A polynomial extrapolation method for finding limits and antilimits of vector sequences, SIAM J. Numer. Anal., 13 (1976), pp. 734-752.
[12] E. Cancès and C. L. Bris, Can we outperform the DIIS approach for electronic structure calculations?, Int. J. Quantum Chem., 79 (2000), pp. 82-90.
[13] E. Cancés and C. L. Bris, On the convergence of SCF algorithms for the Hartree-Fock equations, M2AN Math. Model. Numer. Anal., 34 (2000), pp. 749-774.
[14] E. Cancés, G. Kemlin, and A. Levitt, Convergence analysis of direct minimization and self-consistent iterations, SIAM J. Matrix Anal. Appl., 42 (2021), pp. 243-274.
[15] K. Chen and C. Vuik, Composite anderson acceleration method with two window sizes and optimized damping, Int. J. Numer. Meth. ENG., 123 (2022), pp. 5964-5985.
[16] X. Chen and C. T. Kelley, Convergence of the EDIIS algorithm for nonlinear equations, SIAM J. Sci. Comput., 41 (2019), pp. A365-A379.
[17] M. Chupin, M.-S. Dupuy, G. Legendre, and É. Séré, Convergence analysis of adaptive DIIS algorithms with application to electronic ground state calculations, ESAIM Math. Model. Numer. Anal., 55 (2021), pp. 2785-2825.
[18] H. De Sterck and Y. He, Linear asymptotic convergence of anderson acceleration: Fixedpoint analysis, SIAM J. Matrix Anal. Appl., 43 (2022), pp. 1755-1783.
[19] H. De Sterck, Y. He, and O. A. Krzysik, Anderson acceleration as a Krylov method with application to convergence analysis, J. Sci. Comput., 99 (2024), pp. Paper No. 12, 30.
[20] R. P. Eddy, Extrapolation to the limit of a vector sequence, in Information Linkage Between Applied Mathematics and Industry, P. C. C. Wang, ed., Academic Press, New York, 1979, pp. 387-396.
[21] S. C. Eisenstat, H. C. Elman, and M. H. Schultz, Variational iterative methods for nonsymmetric systems of linear equations, SIAM J. Numer. Anal., 20 (1983), pp. 345-357.
[22] P. Ettenhuber and P. Jørgensen, Discarding information from previous iterations in an optimal way to solve the coupled cluster amplitude equations, J. Chem. Theory. Comput., 11 (2015), pp. 1518-1524.
[23] C. Evans, S. Pollock, L. G. Rebholz, and M. Xiao, A proof that Anderson acceleration improves the convergence rate in linearly converging fixed-point methods (but not in those converging quadratically), SIAM J. Numer. Anal., 58 (2020), pp. 788-810.
[24] V. Eyert, A comparative study on methods for convergence acceleration of iterative vector sequences, J. Comput. Phys., 124 (1996), pp. 271-285.
[25] H. Fang and Y. Saad, Two classes of multisecant methods for nonlinear acceleration, Numer. Linear Algebra Appl., 16 (2009), pp. 197-221.
[26] H. He, Z. Tang, S. Zhao, Y. Saad, and Y. Xi, nltGCR: a class of nonlinear acceleration procedures based on conjugate residuals, SIAM J. Matrix Anal. Appl., 45 (2024), pp. 712743.
[27] K. Jbilou and H. Sadok, Vector extrapolation methods. applications and numerical comparison, J. Comput. Appl. Math., 122 (2000), pp. 149 - 165. Numerical Analysis in the 20th Century Vol. II: Interpolation and Extrapolation.
[28] L. Lin, J. Lu, and L. Ying, Numerical methods for Kohn-Sham density functional theory, Acta Numer., 28 (2019), pp. 405-539.
[29] P. NI, Anderson acceleration of fixed-point iteration with applications to electronic structure computations, PhD thesis, Worcester Polytechnic Institute, 2009.
[30] P. Ni and H. F. Walker, A linearly constrained least-squares problem in electronic structure computations, ICCES. v7 i1, (2010), pp. 43-49.
[31] C. W. Oosterlee and T. Washio, Krylov subspace acceleration of nonlinear multigrid with application to recirculating flows, J. Sci. Comput., 21 (2000), pp. 1670-1690.
[32] S. Pollock and L. G. Rebholz, Anderson acceleration for contractive and noncontractive operators, IMA J. Numer. Anal., 41 (2021), pp. 2841-2872.
[33] S. Pollock and L. G. Rebholz, Filtering for anderson acceleration, SIAM J. Sci. Comput., 45 (2023), pp. A1571-A1590.
[34] S. Pollock, L. G. Rebholz, and M. Xiao, Anderson-accelerated convergence of Picard iterations for incompressible Navier-Stokes equations, SIAM J. Numer. Anal., 57 (2019), pp. 615-637.
[35] P. Pulay, Convergence acceleration of iterative sequences. The case of SCF iteration, Chem. Phys. Lett., 73 (1980), pp. 393-398.
[36] P. Pulay, Improved SCF convergence acceleration, J. Comput. Chem., 3 (1982), pp. 556-560.
[37] L. G. Rebholz and M. Xiao, The effect of anderson acceleration on superlinear and sublinear convergence, SIAM J. Sci. Comput., 96 (2023).
[38] T. Rohwedder and R. Schneider, An analysis for the DIIS acceleration method used in quantum chemistry calculations, J. Math. Chem., 49 (2011), pp. 1889-1914.
[39] Y. Saad, Iterative methods for sparse linear systems, Society for Industrial and Applied Mathematics, Philadelphia, PA, second ed., 2003.
[40] D. Shanks, Non-linear transformations of divergent and slowly convergent sequences, J. Math. Phys, 34 (1955), pp. 1-42.
[41] A. Sidi, W. F. Ford, and D. A. Smith, Acceleration of convergence of vector sequences, SIAM J. Numer. Anal., 23 (1986), pp. 178-196.
[42] P. Suryanarayana, P. P. Pratapa, and J. E. Pask, Alternating Anderson-Richardson method: An efficient alternative to preconditioned Krylov methods for large, sparse linear systems, Comput. Phys. Commun., 234 (2019), pp. 278-285.
[43] A. Toth and C. T. Kelley, Convergence analysis for Anderson acceleration, SIAM J. Numer. Anal., 53 (2015), pp. 805-819.
[44] H. F. Walker and P. Ni, Anderson acceleration for fixed-point iterations, SIAM J. Numer. Anal., 49 (2011), pp. 1715-1735.
[45] T. Washio and C. W. Oosterlee, Krylov subspace acceleration for nonlinear multigrid schemes, Electron. Trans. Numer. Anal., 6 (1997), pp. 271-290.
[46] F. Wei, C. Bao, Y. Liu, and G. Yang, Convergence analysis for restarted anderson mixing and beyond, arXiv:2307.02062, (2023). https://arxiv.org/abs/2307.02062.
[47] P. Wynn, On a device for computing the $e_{m}\left(S_{n}\right)$ transformation, Mathematical Tables and Other Aids to Computation, 10 (1956), pp. 91-96.
[48] P. Wynn, Acceleration techniques for iterated vector and matrix problems, Math. Comp., 16 (1962), pp. 301-322.
[49] M. Ziólkowski, V. Weijo, P. Jørgensen, and J. Olsen, An efficient algorithm for solving nonlinear equations with a minimal number of trial vectors: Applications to atomic-orbital based coupled-cluster theory, J. Chem. Phys, 128 (2008), p. 204105.

\title{
SUPPLEMENTARY MATERIALS: ON THE CONVERGENCE OF CROP-ANDERSON ACCELERATION METHOD *
}

\author{
NING WAN ${ }^{\dagger}$ AND AGNIESZKA MIĘDLAR ${ }^{\dagger}$
}

This Supplementary Material provides some details regarding implementation and validation of discussed methods, as well as some additional numerical examples.

SM1. Implementation details. All numerical experiments presented in this section have been implemented in Julia 1.7.1 [SM2] and carried out on an Intel Xeon 8-Core 3.00 GHz machine with 32 GB memory. Here, we briefly discuss an adaptive variant of CROP algorithm for nonlinear problems that substitutes some rCROP steps with CROP steps when the approximation is good. Moreover, the idea of line search is discussed.

Restarted and Adaptive Methods on Real Residuals. Since real residuals require additional function evaluations, rCROP algorithm usually takes longer to execute the same number of steps than CROP algorithm. Following the idea presented in [SM6], an adaptive method is running an rCROP step every few iterations to check the accuracy of CROP algorithm step by measuring the angle $\theta^{(k)}$ between the control residual $f_{C}^{(k)}$ and the real residual $r_{C}^{(k)}$ of that step, i.e.,
$$
\cos \theta^{(k)}=\frac{\left(f_{C}^{(k)}, r_{C}^{(k)}\right)}{\left\|f_{C}^{(k)}\right\|\left\|r_{C}^{(k)}\right\|}
$$

If $\cos \theta^{(k)}>0.99$, the nonlinearity of the problem is considered small, and the result produced by CROP algorithm can be viewed as a good approximation of the exact solution. Otherwise, the problem is highly nonlinear and should be solved with rCROP algorithm instead.

Line Search Refinement. The line search is an iterative process used to find the solution of $f(x)=0$ when a descent search direction is available. One commonly used line search method is based on the Armijo condition [SM1]. Since the line search requires computing residuals explicitly, when used it should be employed with rCROP algorithm instead of CROP. For rCROP algorithm, the function evaluation gives the real residuals and thus the line search is possible.

Since CROP gives a Broyden-type approximated inverse Jacobian
$$
G_{C}^{(k+1)}=\mathscr{X}_{C}^{(k+1)}\left[\left(\mathscr{F}_{C}^{(k+1)}\right)^{T} \mathscr{F}_{C}^{(k+1)}\right]^{-1}\left(\mathscr{F}_{C}^{(k+1)}\right)^{T},
$$
we can do the line search with this approximated direction. However, for multisecant methods, we can not guarantee that the approximate Newton direction will be a descent direction, and therefore, a line search may fail. As the approximated direction can have a very large error, the Armijo condition may become meaningless. If the line search with approximated Jacobian fails, a better preconditioner is needed, or CROP algorithm without the line search must be used.

In fact, the line search works very badly for CROP algorithm. For linear and nonlinear problems with weak nonlinearities, rCROP has essentially the similar form

\footnotetext{
* Funding: This work was supported by the National Science Foundation through the NSF CAREER Award DMS-2144181 and DMS-2324958.
${ }^{\dagger}$ Department of Mathematics, Virginia Tech, Blacksburg, VA (wning@vt.edu, amiedlar@vt.edu)
}
as CROP algorithm, meaning that the obtained result is optimal in the search space and the explicit line search step is not needed.

Another way is to do a line search without derivatives, for example by using discrete line search methods like bisection or Li-Fukushima derivative-free line search. Li-Fukushima derivative-free line search [SM8] is designed to work with global convergent Broyden-type method, which update satisfies
$$
\begin{equation*}
\sigma_{0}\left\|\Delta x^{(k)}\right\|^{2} \leq\left(1+\eta_{k}\right)\left\|f\left(x^{(k)}\right)\right\|-\left\|f\left(x^{(k+1)}\right)\right\|, \tag{SM1.1}
\end{equation*}
$$
where $\sigma_{0}>0$ and $\sum_{k=0}^{\infty} \eta_{k} \leq \eta<\infty$.
Let $\Delta x^{(k)}=\lambda^{(k)} p^{(k)}$, where $\lambda^{(k)}$ is the step size, and $p^{(k)}$ is the search direction given as $p^{(k)}=G_{C}^{(k+1)} \widetilde{f}_{C}^{(k+1)}$. Then
$$
x^{(k)}=\widetilde{x}_{C}^{(k+1)} \text { and } x^{(k+1)}=x_{C}^{(k+1)} .
$$

\section*{SM2. Additional experimental results.}

Example 1 (A Real World Problem).
Let us now consider the same linear problem $A x=b$ with a symmetric positive definite matrix cfd1 from SuiteSparse Matrix Collection [SM3] and vector $b \in \mathbb{R}^{70656}$ with its first entry 1 and others 0 . The corresponding convergence results for different methods are shown in Figure SM1. Notice that choice of $m=2$ is the optimal for both CROP and CROP-Anderson algorithm.

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_08_2ce2904da8f9b738a883g-28.jpg?height=405&width=546&top_left_y=1393&top_left_x=265}
\captionsetup{labelformat=empty}
\caption{Fig. SM1: Convergence for a linear problem in Example (1).}
\end{figure}

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_08_2ce2904da8f9b738a883g-28.jpg?height=400&width=535&top_left_y=1398&top_left_x=937}
\captionsetup{labelformat=empty}
\caption{Fig. SM2: r-linear convergence factor for a small linear system in Example (2).}
\end{figure}

Example 2 (A Small Linear Problem).
In this example, we consider a linear system $A x=b[\mathrm{SM} 4, \mathrm{SM} 5$, Problem 1] with
$$
A=\left[\begin{array}{cc}
1 / 3 & -1 / 4  \tag{SM2.1}\\
0 & 2 / 3
\end{array}\right] \in \mathbb{R}^{2 \times 2} \quad \text { and } \quad b=\left[\begin{array}{l}
0 \\
0
\end{array}\right] \in \mathbb{R}^{2} .
$$

We choose $f:=b-A x$ and $g:=x+f=M x$ with
$$
M=\left[\begin{array}{cc}
2 / 3 & 1 / 4 \\
0 & 1 / 3
\end{array}\right] \in \mathbb{R}^{2 \times 2}
$$

We set maxit $=100$, tol $=10^{-16}$ and run fixed-point iteration, Anderson Acceleration, CROP, and CROP-Anderson algorithms with $m=1$. The components $x_{1}^{(0)}$ and $x_{2}^{(0)}$ of the initial vector $x^{(0)}$ are chosen randomly between $[-0.5,0.5]$. The r-linear convergence factors are shown in Figure SM2. The damping parameters $\gamma^{(k)}$ for Anderson and CROP/CROP-Anderson methods are presented in Figure SM3a and Figure SM3b. Note that they have different oscillation behaviors. For Anderson Acceleration method, $\gamma^{(k)}$ display the same oscillation pattern independent on the choice of initial vector, whereas two different kinds of oscillations are occurring for CROP algorithm.

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_08_2ce2904da8f9b738a883g-29.jpg?height=507&width=1221&top_left_y=786&top_left_x=262}
\captionsetup{labelformat=empty}
\caption{Fig. SM3: $\gamma_{k}$ for (a) Anderson Acceleration and (b) CROP Algorithm in Example (2).}
\end{figure}

We show the r-linear convergence factors of the fixed point iteration, Anderson acceleration, CROP, and CROP-Anderson algorithms in Figure SM4a, Figure SM4b, Figure SM4c and Figure SM4d. We can see that the r-linear convergence factor is the function of the angle $\arctan \frac{x_{2}^{(0)}}{x_{1}^{(0)}}$. The factor for CROP and CROP-Anderson oscillates much faster than the factor for Anderson acceleration.

Example 3 (Nonlinear Eigenvalue Problem).
In this example, we consider a time-delay system with a distributed delay discussed in [SM7, Example 2] where the distributed term is a Gaussian distribution, i.e.,
$$
\begin{align*}
\dot{x}(t)= & \frac{1}{10}\left(\begin{array}{ccc}
25 & 28 & -5 \\
18 & 3 & 3 \\
-23 & -14 & 35
\end{array}\right) x(t)+\frac{1}{10}\left(\begin{array}{ccc}
17 & 7 & -3 \\
-24 & -21 & -2 \\
20 & 7 & 4
\end{array}\right) x(t-\tau)  \tag{SM2.2}\\
& +\int_{-\tau}^{0}\left(\begin{array}{ccc}
14 & -13 & 4 \\
14 & 7 & 10 \\
6 & 16 & 17
\end{array}\right) \frac{e^{\left(s+\frac{1}{2}\right)^{2}}-e^{\frac{1}{4}}}{10} x(t+s) d s
\end{align*}
$$

The eigenvalues of (SM2.2) are given as the solutions of the nonlinear eigenvalue problem (NEP) $M(\lambda) v=0$ associated with a matrix-valued function
$$
M(\lambda)=-\lambda I+A_{0}+A_{1} e^{-\lambda \tau}+\int_{-\tau}^{0} F(s) e^{\lambda s} d s
$$

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_08_2ce2904da8f9b738a883g-30.jpg?height=890&width=1140&top_left_y=363&top_left_x=308}
\captionsetup{labelformat=empty}
\caption{Fig. SM4: r-linear convergence factors in Example (2).}
\end{figure}
where
$$
\begin{gathered}
A_{0}=\frac{1}{10}\left(\begin{array}{ccc}
25 & 28 & -5 \\
18 & 3 & 3 \\
-23 & -14 & 35
\end{array}\right), \quad A_{1}=\frac{1}{10}\left(\begin{array}{ccc}
17 & 7 & -3 \\
-24 & -21 & -2 \\
20 & 7 & 4
\end{array}\right) \\
F(s)=\left(\begin{array}{ccc}
14 & -13 & 4 \\
14 & 7 & 10 \\
6 & 16 & 17
\end{array}\right) \frac{e^{\left(s+\frac{1}{2}\right)^{2}}-e^{\frac{1}{4}}}{10} .
\end{gathered}
$$

Consider $x=\left[\begin{array}{ll}v & \lambda\end{array}\right]^{T}$. Then, the nonlinear eigenvalue problem associated with (SM2.2) can be written as a nonlinear equation of the form
$$
f(x)=f\left(\left[\begin{array}{l}
v \\
\lambda
\end{array}\right]\right):=\left[\begin{array}{c}
M(\lambda) v \\
c^{H} v-1
\end{array}\right]=0,
$$
where $c$ is used to normalize the eigenvector $v$ and we choose $c$ the vector of ones here. Let $g:=x+\beta f$ with $\beta=0.1$. We run Anderson Acceleration, CROP, CROPAnderson algorithms with maxit $=100$, tol $=10^{-10}, m=$ maxit, $m=3$ and $m=5$. The initial vector $x^{(0)}$ is chosen as the vector of ones. The convergence of all methods is shown in Figure SM5.

Figure SM5a illustrates the breakdown of CROP algorithm and the problems associated with using control residuals $f_{C}^{(k)}$. The eigenvector of this NEP is of length 3, and the residual vector of dimension $n=4$. The problem is nonlinear and needs more than $n$ iterations to converge. When the least-squares problem for finding the

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_08_2ce2904da8f9b738a883g-31.jpg?height=535&width=1215&top_left_y=374&top_left_x=265}
\captionsetup{labelformat=empty}
\caption{Fig. SM5: A nonlinear eigenvalue problem from Example (3) with (a) control residuals and (b) real residuals.}
\end{figure}
coefficients of the CROP iterates is of size $m_{C}^{(k)}=4$, it has a solution and which makes $f_{C}^{(k)}=0$. CROP algorithm for this problem breaks down at the 4 -th step. Meanwhile, rCROP and rCROP-Anderson method converge well when $m=3$ and $m=5$, respectively, see Figure SM5b.

\section*{REFERENCES}
[SM1] L. Armijo, Minimization of functions having Lipschitz continuous first partial derivatives., Pacific Journal of Mathematics, 16 (1966), pp. 1-3.
[SM2] J. Bezanson, A. Edelman, S. Karpinski, and V. B. Shah, Julia: A fresh approach to numerical computing, SIAM Review, 59 (2017), pp. 65-98, https://doi.org/10.1137/141000671, https://epubs.siam.org/doi/10.1137/141000671.
[SM3] T. A. Davis and Y. Hu, The university of florida sparse matrix collection, ACM Transactions on Mathematical Software (TOMS), 38 (2011), pp. 1-25.
[SM4] H. De Sterck and Y. He, Linear asymptotic convergence of anderson acceleration: Fixedpoint analysis, SIAM J. Matrix Anal. Appl., 43 (2022), pp. 1755-1783.
[SM5] H. De Sterck, Y. He, and O. A. Krzysik, Anderson acceleration as a Krylov method with application to convergence analysis, J. Sci. Comput., 99 (2024), pp. Paper No. 12, 30.
[SM6] H. He, Z. Tang, S. Zhao, Y. Saad, and Y. Xi, nltGCR: a class of nonlinear acceleration procedures based on conjugate residuals, SIAM J. Matrix Anal. Appl., 45 (2024), pp. 712743.
[SM7] E. Jarlebring, W. Michiels, and K. Meerbergen, The Infinite Arnoldi Method and an Application to Time-Delay Systems with Distributed Delays, Springer Berlin Heidelberg, Berlin, Heidelberg, 2012, pp. 229-239, https://doi.org/10.1007/978-3-642-25221-1_ 17, https://doi.org/10.1007/978-3-642-25221-1_17.
[SM8] D.-H. Li and M. Fukushima, A derivative-free line search and global convergence of broyden-like method for nonlinear equations, Optimization Methods and Software, 13 (2000), pp. 181-201, https://doi.org/10.1080/10556780008805782, https://doi.org/10.1080/ 10556780008805782 , https://arxiv.org/abs/https://doi.org/10.1080/10556780008805782.