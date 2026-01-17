## Closest Wannier method

Following the closest Wannier method, one can obtain
the closest Wannier functions (CWFs)
to the initial guesses in a Hilbert space without iterative calculations,
significantly reducing computational costs.
For the detail of the theoretical background, see Ref. [ozaki-prb24].

A main idea of the closest Wannier method is to
introduce a window function $w(\varepsilon)$ (See Eq. (4) in
Ref. [ozaki-prb24]) for the projection of the Bloch states
    $|\psi_{n\bf{k}}\rangle$ onto trial localised orbitals
    $|g_{n}\rangle$

$$
& A_{mn}^{(\bf{k})} = w(\varepsilon_{m\bf{k}}) \langle \psi_{m{\bf k}}|g_{n}\rangle,
\\
& w(\varepsilon) = \frac{1}{e^{(\varepsilon - \mu_{\rm max})/\sigma_{\rm max}} + 1}
- \frac{1}{e^{(\mu_{\rm min} - \varepsilon)/\sigma_{\rm min}} + 1} - 1 + \cwf_delta.
$$

$\mu_{\rm min}$ and $\mu_{\rm max}$ ($\mu_{\rm min} < \mu_{\rm max}$)
represent the bottom and top of the energy window and
$\sigma_{\rm min}$ and $\sigma_{\rm max}$ control
the degree of smearing around $\mu_{\rm min}$ and $\mu_{\rm max}$, respectively.
As a result of this smearing, all Bloch states are incorporated into
the projection with specific weights, where the weight is
particularly large inside the window
($\mu_{\rm min} < \varepsilon_{m\bf{k}} < \mu_{\rm max}$),
whereas it is small outside of the window.
$\cwf_delta$ is a small constant introduced to prevent the matrix
consisting of $A_{mn}^{(\bf{k})}$ from becoming ill-conditioned.

Note that the disentanglement of bands is naturally
taken into account by introducing a window function.
The unitary matrix is then obtained by
$\mathbf{U}^{(\mathbf{k})} = \mathbf{A}^{(\bf{k})}
(\mathbf{A}^{(\bf{k}) \dagger} \mathbf{A}^{(\bf{k})})^{-1/2}$
without iterative calculations for disentanglment of bands
and wannierisation (`dis_num_iter = 0` and `num_iter = 0`).
By properly choosing $\mu_{\rm min}$, $\mu_{\rm max}$,
$\sigma_{\rm min}$, $\sigma_{\rm max}$, and $\cwf_delta$,
one can obtain the Wannier functions _closest_ to the initial guesses in
a Hilbert space (see Eqs. (5)~(17) in Ref. [ozaki-prb24]).

!!! note
    It's worth noting that the CWFs can be treated as initial guesses to generate
    a set of MLWFs by specifying parameters for disentanglement and wannierisation.
