# SymClosestWannier

## data for SymCW (* optional [default] or only for crystal value)
- kpoints : reciprocal lattice points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
- rpoints : lattice points data, [[r1, r2, r3]] (crystal coordinate).
- kpoints_path* :  reciprocal lattice points along high symmetry line in Brillouin zonen, [[k1, k2, k3]] (crystal coordinate).
- k_linear* : reciprocal lattice points along high symmetry line, [k].
- Pk : projectability of each Kohn-Sham state in k-space.
- Hk : Hamiltonian matrix elements in k-space.
- Sk : Overlap matrix elements in k-space.
- matrix_dict* : dictionary form of the real-space representation of symmetry-adapted multipole basis (SAMB).
- z* : The expansion coefficients of the Hamiltonian matrix expressed by a linear combination of SAMBs
- s* : The expansion coefficients of the Overlap matrix expressed by a linear combination of SAMBs
- rpoints_mp* : lattice points data included in matrix_dict, [[r1, r2, r3]] (crystal coordinate).
- Ek_RMSE_grid* : mean squared error of eigen energies between symmetrized and non-symmetrized closed wannier TB model (grid).
- Ek_RMSE_path* :  mean squared error of eigen energies between symmetrized and non-symmetrized closed wannier TB model (path).
