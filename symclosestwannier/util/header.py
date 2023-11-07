"""
headers.
"""

start_str = """
\n********************************************************************************
*                                                                              *
*  Create Closest Wannier Tight-Binding Model from Plane-Wave DFT Calculation  *
*                                                                              *
********************************************************************************\n
"""

end_str = """
\n********************************************************************************
*                                                                              *
*                            Successfully Completed                            *
*                                                                              *
********************************************************************************\n
"""


input_header_str = """
=== input parameters (* optional [default] or only for crystal value) ===
* For execution use: pw2scw [seedname]
    - If a seedname string is given the code will read its input from a file seedname.cwin. The default value is cwannier.
      One can also equivalently provide the string seedname.cwin instead of seedname.

* input parameters in seedname.cwin (same format as seedname.win) (* optional [default])
    - restart*           : the restart position (str), ["wannierize"].
    - outdir*            : input and output files are found in this directory (str), ["./"].
    - disentangle*       : disentagle bands ? (bool), [False].
    - proj_min*          : minimum value of projectability: [0.0].
    - dis_win_emax*      : upper energy window (float), [None].
    - dis_win_emin*      : lower energy window (float), [None].
    - smearing_temp_max* : smearing temperature for upper window (float), [5.0].
    - smearing_temp_min* : smearing temperature for lower window (float), [0.01].
    - delta*             : small constant to avoid ill-conditioning of overlap matrices (float), [1e-12].
    - svd*               : implement singular value decomposition ? otherwise adopt Lowdin's orthogonalization method (bool), [False].
    - verbose*           : verbose calculation info (bool, optional), [False].
    - write_hr*          : write seedname_hr.py ? (bool), [False].
    - write_sr*          : write seedname_sr.py ? (bool), [False].

  # only used for symmetrization.
    - symmetrization*    : symmetrize ? (bool), [False].
    - mp_outdir*         : output files for multipie are found in this directory (str). ["./"].
    - mp_seedname*       : seedname for seedname_model.py, seedname_samb.py and seedname_matrix.py files (str).
    - ket_amn*           : ket basis list in the seedname.amn file. The format of each ket must be same as the "ket" in sambname_model.py file. See sambname["info"]["ket"] in sambname_model.py file for the format (list), [None].
    - irreps*            : list of irreps to be considered (str/list), [None].

  # only used for band dispersion calculation.
    - a*                 : lattice parameter (in Ang) used to correct units of k points in reference band data, [1.0].
    - fermi_energy*      : fermi energy used for band shift, [None].
    - N1*                : number of divisions for high symmetry lines (int, optional), [50].
"""


info_header_str = """
=== information for SymCW (* optional [default] or only for crystal value) ===
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
- kpoint_path* : high-symmetry line in k space.
- unit_cell_cart* : transform matrix, [a1,a2,a3].
- atoms_frac* : atomic positions in fractional coordinates with respect to the lattice vectors, {atom: [r1,r2,r3]}.
- atoms_cart* : atomic positions in cartesian coordinates, {atom: [rx,ry,rz]}.
"""

data_header_str = """
=== data for SymCW (* optional [default] or only for crystal value) ===
- kpoints : reciprocal lattice points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
- rpoints : lattice points data, [[r1, r2, r3]] (crystal coordinate).
- kpoints_path* :  reciprocal lattice points along high symmetry line in Brillouin zonen, [[k1, k2, k3]] (crystal coordinate).
- k_linear* : reciprocal lattice points along high symmetry line, [k].
- k_dis_pos* : disconnected linear positions and labels, {disconnected linear position: label}.
- Pk : projectability of each Kohn-Sham state in k-space.
- Hk : Hamiltonian matrix elements in k-space.
- Sk : Overlap matrix elements in k-space.
- matrix_dict* : dictionary form of the real-space representation of symmetry-adapted multipole basis (SAMB).
- z* : The expansion coefficients of the Hamiltonian matrix expressed by a linear combination of SAMBs
- s* : The expansion coefficients of the Overlap matrix expressed by a linear combination of SAMBs
- rpoints_mp* : lattice points data included in matrix_dict, [[r1, r2, r3]] (crystal coordinate).
- Ek_RMSE_grid* : mean squared error of eigen energies between symmetrized and non-symmetrized closed wannier TB model (grid).
- Ek_RMSE_path* :  mean squared error of eigen energies between symmetrized and non-symmetrized closed wannier TB model (path).
"""


kpoints_header_str = "# reciprocal lattice points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate)."


rpoints_header_str = "# lattice points data, [[r1, r2, r3]] (crystal coordinate)."


hk_header_str = """
=== Hamiltonian matrix elements in k-space ===
- {(k2,k2,k3,a,b) = H_{ab}(k)}.
- H_{ab}(k) : <φ_{a}(k)|H|φ_{b}(k)>.
- φ_{a}(k) : orthogonalized pseudo atomic orbital.
- k = (k1,k2,k3) : k points (crystal coordinate).
"""


sk_header_str = """
=== Overlap matrix elements in k-space ===
- {(k2,k2,k3,a,b) = S_{ab}(k)}.
- S_{ab}(k) : <φ_{a}(k)|φ_{b}(k)>.
- φ_{a}(k) : non-orthogonalized pseudo atomic orbital.
- k = (k1,k2,k3) : k points (crystal coordinate).
"""


pk_header_str = """
=== Projectability of each Kohn-Sham state in k-space ===
- {(k2,k2,k3,n,n) = P_{n}(k)}.
- P_{n}(k) : Σ_{a} |<ψ_{n}(k)|φ_{a}(k)>|^2.
- ψ_{n}(k) : Kohn-Sham orbital corresponding to the n-th Kohn-Sham energy ε_{n}(k).
- φ_{a}(k) : non-orthogonalized pseudo atomic orbital.
- k = (k1,k2,k3) : k points (crystal coordinate).
"""


hr_header_str = """
=== Hamiltonian matrix elements in real-space ===
- n1 n2 n3 a b re(H_{ab}(R)) im(H_{ab}(R))
    - H_{ab}(R) : <φ_{a}(R)|H|φ_{b}(0)>.
    - φ_{a}(R) : orthogonalized pseudo atomic orbital.
    - R = (n1,n2,n3) : lattice points (crystal coordinate, nj: integer).
"""

sr_header_str = """
=== Overlap matrix elements in real-space ===
- n1 n2 n3 a b re(S_{ab}(R)) im(S_{ab}(R))
    - S_{ab}(R) : <φ_{a}(R)|φ_{b}(0)>.
    - φ_{a}(R) : non-orthogonalized pseudo atomic orbital.
    - R = (n1,n2,n3) : lattice points (crystal coordinate, nj: integer).
"""


z_header_str = """
=== The expansion coefficients of the Hamiltonian matrix expressed by a linear combination of SAMBs ===
- j z_j TagMultipole coefficient
    - H(R) ~ sum_{j} z_j Z_j(R)
    - z_j = \sum_{R} Tr[Z_j(R)*H(R)].
    - z_j is the expansion coefficients.
"""


s_header_str = """
=== The expansion coefficients of the Overlap matrix expressed by a linear combination of SAMBs ===
- j z_j TagMultipole coefficient
    - S(R) ~ sum_{j} z_j Z_j(R)
    - z_j = \sum_{R} Tr[Z_j(R)*S(R)].
    - z_j is the expansion coefficients.
"""
