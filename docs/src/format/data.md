# data for SymCW (* optional [default] or only for crystal value)
- Sk              : Overlap matrix elements in k-space (ndarray).",
- Hk              : Hamiltonian matrix elements in k-space (orthogonal) (ndarray).",
- Hk_nonortho     : Hamiltonian matrix elements in k-space (non-orthogonal) (ndarray).",
- Sr              : Overlap matrix elements in real-space (ndarray).",
- Hr              : Hamiltonian matrix elements in real-space (orthogonal) (ndarray).",
- Hr_nonortho     : Hamiltonian matrix elements in real-space (non-orthogonal) (ndarray).",
- s               : The expansion coefficients of Sk expressed by a linear combination of SAMBs (ndarray).",
- z               : The expansion coefficients of Hk expressed by a linear combination of SAMBs (ndarray).",
- z_nonortho      : The expansion coefficients of Hk_nonortho expressed by a linear combination of SAMBs (ndarray).",
- z_exp           : The expectation value of the SAMBs (dict),
- Sk_sym          : Symmetrized overlap matrix elements in k-space (ndarray).",
- Hk_sym          : Symmetrized Hamiltonian matrix elements in k-space (orthogonal) (ndarray).",
- Hk_nonortho_sym : Symmetrized Hamiltonian matrix elements in k-space (non-orthogonal) (ndarray).",
- Sr_sym          : Symmetrized overlap matrix elements in real-space (ndarray).",
- Hr_sym          : Symmetrized Hamiltonian matrix elements in real-space (orthogonal) (ndarray).",
- Hr_nonortho_sym : Symmetrized Hamiltonian matrix elements in real-space (non-orthogonal) (ndarray).",
- Ek_RMSE_grid    : mean squared error of eigen energies between symmetrized and non-symmetrized closed wannier TB model (grid) (float).",
- Ek_RMSE_path    : mean squared error of eigen energies between symmetrized and non-symmetrized closed wannier TB model (path) (float).",
- matrix_dict     : dictionary form of the real-space representation of symmetry-adapted multipole basis (SAMB) (dict).",