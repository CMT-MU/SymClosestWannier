# Phase conventions of real-space operators

This page fixes the conventions used for the real-space matrices written by `pw2cw` (`seedname_hr.dat.cw`, `seedname_tb.dat.cw`, …) and stored in `seedname.hdf5`, and states what `tb_gauge` changes.

## Real-space matrices

- $O_{ab}(\boldsymbol{R}) = \langle \phi_{a}(\boldsymbol{0})|O|\phi_{b}(\boldsymbol{R})\rangle$, where $\phi_{b}(\boldsymbol{R})$ is orbital $b$ in the cell at the lattice vector $\boldsymbol{R}$.
- $\boldsymbol{R} = n_1\boldsymbol{a}_1 + n_2\boldsymbol{a}_2 + n_3\boldsymbol{a}_3$ is given by the integers `irvec` = (n1, n2, n3) of the Wigner–Seitz supercell (`util/utility.py:wigner_seitz`); `ndegen` is the degeneracy of each $\boldsymbol{R}$.
- Arrays are indexed as `O[iR, a, b]` (vector operators: `O[i, iR, a, b]`, i = x, y, z), and the stored values are **not** divided by `ndegen`, as in wannier90.
- Lattice vectors `unit_cell_cart` are rows $\boldsymbol{a}_1, \boldsymbol{a}_2, \boldsymbol{a}_3$ in Å.
- The orbital order is the order of `ket_amn` (the projections in `seedname.win`). Symmetrized matrices (`*_sym`) use the SAMB ket order of MultiPie; `sort_ket_matrix` converts between the two.

## Fourier transform

The Bloch basis is $|\phi_{b}(\boldsymbol{k})\rangle = \sum_{\boldsymbol{R}} e^{i\boldsymbol{k}\cdot\boldsymbol{R}}|\phi_{b}(\boldsymbol{R})\rangle$, with no orbital position in the phase. This is the basis in which the closest Wannier construction gives $H(\boldsymbol{k})$, $S(\boldsymbol{k})$, … from `seedname.amn` and `seedname.eig`, and it is the convention of wannier90 (`_hr.dat`, `_tb.dat`). Then

$$
O_{ab}(\boldsymbol{k}) = \sum_{\boldsymbol{R}} \frac{1}{N_{\boldsymbol{R}}} e^{i\boldsymbol{k}\cdot\boldsymbol{R}} O_{ab}(\boldsymbol{R}), \qquad
O_{ab}(\boldsymbol{R}) = \frac{1}{N_k}\sum_{\boldsymbol{k}} e^{-i\boldsymbol{k}\cdot\boldsymbol{R}} O_{ab}(\boldsymbol{k}),
$$

with $N_{\boldsymbol{R}}$ = `ndegen` and the sum over the DFT k mesh (`fourier_transform_r_to_k` / `fourier_transform_k_to_r` without `atoms_frac`).

## tb_gauge

With `tb_gauge = true`, the Bloch basis contains the orbital positions $\boldsymbol{\tau}_b$ (the projection centre of orbital $b$ in `seedname.nnkp`, `util/utility.py:orbital_positions`; for projections on atoms this is the atom position), $|\phi^{\rm tb}_{b}(\boldsymbol{k})\rangle = e^{i\boldsymbol{k}\cdot\boldsymbol{\tau}_b}|\phi_{b}(\boldsymbol{k})\rangle$, and k-space matrices are
$$
O^{\rm tb}_{ab}(\boldsymbol{k}) = \sum_{\boldsymbol{R}} \frac{1}{N_{\boldsymbol{R}}} e^{i\boldsymbol{k}\cdot(\boldsymbol{R}+\boldsymbol{\tau}_b-\boldsymbol{\tau}_a)} O^{\rm tb}_{ab}(\boldsymbol{R})
$$
(`atoms_frac` argument of `fourier_transform_r_to_k`). The change of basis is a k-dependent unitary transformation, so eigenvalues do not change; eigenvectors and the Wannier-gauge matrices of the Berry connection and of k derivatives do.

All real-space matrices computed and written by `pw2cw` (`Hr`, `Sr`, `*_sym`, `seedname_hr.dat.cw`, `seedname_r.dat.cw`, `seedname_v.dat.cw`, `seedname.s.cw`, …) are in the wannier90 convention, whatever `tb_gauge` is. `seedname_tb.dat.cw` is written only with `tb_gauge = false`.

For `postcw` with `tb_gauge = true`, `Response` keeps the operators in the wannier90 convention and converts them when they are passed to the calculation (`util/get_oper_R.py:to_tb_gauge`, called from `Response.operators`), with $\boldsymbol{\tau}$ in Cartesian coordinates and $m$, $n$ the row and column orbitals:

| operator | conversion |
| --- | --- |
| `HH_R`, `SS_R`, `SH_R` | unchanged |
| `AA_R` ($\boldsymbol{r}$) | $A^{\rm tb}_{a,mn}(\boldsymbol{R}) = A_{a,mn}(\boldsymbol{R}) - \tau_{n,a}\delta_{mn}\delta_{\boldsymbol{R}\boldsymbol{0}}$ |
| `BB_R` ($H\boldsymbol{r}$) | $B^{\rm tb}_{a}(\boldsymbol{R}) = B_{a}(\boldsymbol{R}) - \tau_{n,a} H(\boldsymbol{R})$ |
| `CC_R` ($\boldsymbol{r}H\boldsymbol{r}$) | $C^{\rm tb}_{ab} = C_{ab} - \tau_{m,a} B_{b} - \tau_{n,b} B^{\dagger}_{a} + \tau_{m,a}\tau_{n,b} H$, with $B^{\dagger}_{a,mn}(\boldsymbol{R}) = B_{a,nm}(-\boldsymbol{R})^{*}$ |
| `SR_R` ($\sigma\boldsymbol{r}$), `SHR_R` ($\sigma H\boldsymbol{r}$) | $SR^{\rm tb}_{ab}(\boldsymbol{R}) = SR_{ab}(\boldsymbol{R}) - \tau_{n,b} SS_{a}(\boldsymbol{R})$, $SHR^{\rm tb}_{ab}(\boldsymbol{R}) = SHR_{ab}(\boldsymbol{R}) - \tau_{n,b} SH_{a}(\boldsymbol{R})$ ($a$: spin component) |

k derivatives are taken in the same basis, $\partial_{\boldsymbol{k}} O^{\rm tb}(\boldsymbol{k}) = \sum_{\boldsymbol{R}} i(\boldsymbol{R}+\boldsymbol{\tau}_b-\boldsymbol{\tau}_a)\,e^{i\boldsymbol{k}\cdot(\boldsymbol{R}+\boldsymbol{\tau}_b-\boldsymbol{\tau}_a)} O^{\rm tb}_{ab}(\boldsymbol{R})/N_{\boldsymbol{R}}$ (`fourier_transform_r_to_k_new`, `fourier_transform_r_to_k_vec` with `pseudo=True`, `get_v_R` with `atoms_frac`).

`to_tb_gauge` assumes `ndegen(-R) = ndegen(R)`, which holds for the Wigner–Seitz supercell.

With these conversions the operators in both gauges represent the same physical quantities, so the Wannier-interpolation formulas give the same Berry curvature, orbital magnetization (−2Im f, g, h) and spin current at each k point; only their split into the J0, J1 and J2 terms depends on the gauge (checked in `tests/test_tb_gauge.py`). This holds when `HH_R` is the Hamiltonian from which `BB_R` and `CC_R` are built. For the spin current of Qiao's method the gauges also agree if $SH(\boldsymbol{k}) = SS(\boldsymbol{k})\,H(\boldsymbol{k})$ (a product of k-space matrices, a convolution in $\boldsymbol{R}$), as in the model of the test. `SH_R` is built from all Bloch bands, $U^\dagger s H U$, which differs from $U^\dagger s U\,U^\dagger H U$ by $U^\dagger s(1-UU^\dagger)HU$; this is in general nonzero for disentangled models, and the spin Hall conductivity of Qiao's method can then depend on `tb_gauge`, as it does for the disentangled spinor model of the tests of `symcw-wberri`. wannier90 uses the convention of `tb_gauge = false`. `postcw` may use a different Hamiltonian (`Hr_sym` with symmetrization, `hr_input`); `BB_R` and `CC_R` are then still converted with their own Hamiltonian, but quantities combining them with the given Hamiltonian (orbital magnetization, gyrotropic K) are approximate in either gauge and differ between the gauges by terms of the order of the difference of the two Hamiltonians.

Quantities that use only eigenvalues (band dispersion, density of states, Lindhard function) do not depend on `tb_gauge`. Results do depend on it where the position operator is neglected: the velocity $\partial_{\boldsymbol{k}} H/\hbar$ used with `use_tb_approximation = true` (`get_v_R`) has interband matrix elements that depend on the orbital phases.
