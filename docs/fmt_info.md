# SymClosestWannier

## information for SymCW (* optional [default] or only for crystal value)
- restart* : the restart position (str), ["wannierize"].
- outdir* : input and output files are found in this directory (str), ["./"].
- seedname* : seedname for seedname.win and seedname.cwin files (str), ["cwannier"].
- disentangle* : disentagle bands ? (bool), [False].
- proj_min* minimum value of projectability: [0.0].
- dis_win_emax* : upper energy window (float), [None].
- dis_win_emin* : lower energy window (float), [None].
- smearing_temp_max* : smearing temperature for upper window (float), [5.0].
- smearing_temp_min* : smearing temperature for lower window (float), [0.01].
- delta* : small constant to avoid ill-conditioning of overlap matrices (float), [1e-12].
- svd* : implement singular value decomposition ? otherwise adopt Lowdin's orthogonalization method (bool), [False].
- verbose* : verbose parallel info (bool), [False].
- write_hr* : write seedname_hr.py ? (bool), [False].
- write_sr* : write seedname_sr.py ? (bool), [False].
- symmetrization* : symmetrize ? (bool), [False].
- mp_outdir* : output files for multipie are found in this directory (str). ["./"].
- mp_seedname* : seedname for seedname_model.py, seedname_samb.py and seedname_matrix.py files (str).
- ket_amn* : ket basis list in the seedname.amn file. The format of each ket must be same as the "ket" in sambname_model.py file. See sambname["info"]["ket"] in sambname_model.py file for the format (list), [None].
- irreps* : list of irreps to be considered (str/list), [None].
- a* : lattice parameter (in Ang) used to correct units of k points in reference band data (float), [None].
- N1* : number of divisions for high symmetry lines (int), [50].
- fermi_energy* : fermi energy used for band shift (float), [None].
- num_k : # of k points (int).
- num_bands : # of bands passed to the code (int).
- num_wann : # of CWFs (int).
- kpoint* : representative k points.
- kpoint_path* : high-symmetry line in k space.d
- unit_cell_cart* : transform matrix, [a1,a2,a3].
- atoms_frac* : atomic positions in fractional coordinates with respect to the lattice vectors, {atom: [r1,r2,r3]}.
- atoms_cart* : atomic positions in cartesian coordinates, {atom: [rx,ry,rz]}.