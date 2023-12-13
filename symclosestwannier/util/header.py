"""
headers.
"""

cwin_info = {
    "seedname": "seedname (str), [cwannier].",
    "outdir": "input and output files are found in this directory (str), ['./'].",
    "restart": "the restart position 'cw'/'w90'/'sym' (str), ['wannierize'].",
    "disentangle": "disentagle bands ? (bool), [False].",
    "proj_min": "minimum value of projectability (float), [0.0].",
    "dis_win_emax": "upper energy window (float), [None].",
    "dis_win_emin": "lower energy window (float), [None].",
    "smearing_temp_max": "smearing temperature for upper window (float), [5.0].",
    "smearing_temp_min": "smearing temperature for lower window (float), [0.01].",
    "delta": "small constant to avoid ill-conditioning of overlap matrices (float), [1e-12].",
    "svd": "implement singular value decomposition ? otherwise adopt Lowdin's orthogonalization method (bool), [False].",
    "verbose": "verbose calculation info (bool, optional), [False].",
    "parallel": "use parallel code? (bool), [False].",
    "formatter": "format by using black? (bool), [False].",
    "write_hr": "write seedname_hr.dat.cw ? (bool), [False].",
    "write_sr": "write seedname_sr.dat.cw ? (bool), [False].",
    "write_u_matrices": "write seedname_u.dat.cw and seedname_u_dis.dat.cw ? (bool), [False].",
    "write_rmn": "write seedname_r.dat.cw ? (bool), [False].",
    "write_vmn": "write seedname_v.dat.cw ? (bool), [False].",
    "write_tb": "write seedname_tb.dat.cw ? (bool), [False].",
    "symmetrization": "symmetrize ? (bool), [False].",
    "mp_outdir": "output files for multipie are found in this directory (str). ['./'].",
    "mp_seedname": "seedname for seedname_model.py, seedname_samb.py and seedname_matrix.py files (str), ['default'].",
    "ket_amn": "ket basis list in the seedname.amn file. The format of each ket must be same as the 'ket' in sambname_model.py file. See sambname['info']['ket'] in sambname_model.py file for the format (list), [None].",
    "irreps": "list of irreps to be considered (str/list), [None].",
    "a": "lattice parameter (in Ang) used to correct units of k points in reference band data (float, optional), [1.0].",
    "N1": "number of divisions for high symmetry lines (int, optional), [50].",
    "fermi_energy": "fermi energy used for band shift (float, optional), [None].",
}

win_info = {
    "seedname": "seedname (str), [cwannier].",
    "num_k": "# of k points (int), [1].",
    "num_bands": "# of bands passed to the code (int), [1].",
    "num_wann": "# of WFs (int), [1].",
    #
    "dis_num_iter": "# of iterations for disentanglement (int), [0].",
    "num_iter": "# of iterations for maximal localization (int), [200].",
    "dis_froz_max": "top of the inner (frozen) energy window (float), [+100000].",
    "dis_froz_min": "bottom of the inner (frozen) energy window (float), [-100000].",
    "dis_win_max": "top of the outer energy window (float), [+100000].",
    "dis_win_min": "bottom of the outer energy window (float), [-100000].",
    "dis_mix_ratio": "mixing ratio during the disentanglement (float), [0.5].",
    #
    "mp_grid": "dimensions of the Monkhorst-Pack grid of k-points (list), [0, 0, 0].",
    "kpoints": "k-points, [[k1, k2, k3]] (crystal coordinate) (list), [[[0, 0, 0]]].",
    "kpoint": "representative k points (dict), [None].",
    "kpoint_path": "k-points along high symmetry line in Brillouin zonen, [[k1, k2, k3]] (crystal coordinate) (str), [None].",
    "unit_cell_cart": "transform matrix, [a1,a2,a3], [None].",
    "atoms_frac": "atomic positions in fractional coordinates with respect to the lattice vectors, {atom: [r1,r2,r3]} [None].",
    "atoms_cart": "atomic positions in cartesian coordinates, {atom: [rx,ry,rz]} [None].",
}

nnkp_info = {
    "A": "real lattice vectors, A = [a1,a2,a3] (list), [[[1,0,0], [0,1,0], [0,0,1]]].",
    "B": "reciprocal lattice vectors, B = [b1,b2,b3] (list), [[[2*pi,0,0], [0,2*pi,0], [0,0,2*pi]]].",
    "num_k": "# of k points (int), [1].",
    "num_wann": "# of WFs (int), [1].",
    "num_atom": "# of atoms (int), [1].",
    "num_b": "# of b-vectors (int), [1].",
    "kpoints": "k-points, [[k1, k2, k3]] (crystal coordinate) (list), [[[0, 0, 0]]].",
    "nnkpts": "nearest-neighbour k-points (list), [None].",
    "nw2n": "atom position index of each WFs (list), [None].",
    "nw2l": "l specifies the angular part Θlm(θ, φ) (list), [None].",
    "nw2m": "m specifies the angular part Θlm(θ, φ) (list), [None].",
    "nw2r": "r specifies the radial part Rr(r) (list), [None].",
    "atom_orb": "WFs indexes of each atom (list), [None].",
    "atom_pos": "atom position index of each atom (list), [None].",
    "atom_pos_r": "atom position of each atom in fractional coordinates with respect to the lattice vectors (list), [None].",
    "bvec_cart": "b-vectors (cartesian coordinate) (list), [None].",
    "bvec_crys": "b-vectors (crystal coordinate) (list), [None].",
    "wb": " weight for each k-points and nearest-neighbour k-points (list), [None].",
}


eig_info = {
    "num_k": "# of k points (int), [1].",
    "num_bands": "# of bands passed to the code (int), [1].",
    "Ek": "Kohn-Sham energies, E_{m}(k) (list), [None].",
}

amn_info = {
    "num_k": "# of k points (int), [1].",
    "num_bands": "# of bands passed to the code (int), [1].",
    "num_wann": "# of WFs (int), [1].",
    "Ak": "Overlap matrix elements, A_{mn}(k) = <ψ^{KS}_{m}(k)|φ_{n}(k)> (list), [None].",
}

mmn_info = {
    "num_k": "# of k points (int), [1].",
    "num_bands": "# of bands passed to the code (int), [1].",
    "num_b": "# of b-vectors (int), [1].",
    "nnkpts": "nearest-neighbour k-points (list), [None].",
    "Mkb": "Overlap matrix elements, M_{mn}(k,b) = <u^{KS}_{m}(k)|u^{KS}_{n}(k+b)> (list), [None].",
}

umat_info = {
    "num_k": "# of k points (int), [1].",
    "num_bands": "# of bands passed to the code (int), [1].",
    "num_wann": "# of WFs (int), [1].",
    "kpoints": "k-points, [[k1, k2, k3]] (crystal coordinate) (list), [[[0, 0, 0]]].",
    "Uoptk": "num_wann×num_wann full unitary matrix (ndarray), [None].",
    "Udisk": "num_wann×num_bands partial unitary matrix (ndarray), [None].",
    "Uk": "num_wann×num_bands full unitary matrix (ndarray), [None].",
}

cw_info = cwin_info | win_info | nnkp_info | eig_info | amn_info | mmn_info | umat_info

cwin_header = """
=== input parameters in the *.cwin file [default] === \n
* For execution use: pw2cw [seedname]
    - If a seedname string is given the code will read its input from a file seedname.cwin. The default value is cwannier.
      One can also equivalently provide the string seedname.cwin instead of seedname.

* input parameters in seedname.cwin (same format as seedname.win) [default] \n
"""
cwin_header += "\n".join(["    - {:<18} : {:<100} \n".format(k, v) for k, v in cwin_info.items()])


win_header = """
=== input parameters in the *.win file [default] === \n
"""
win_header += "\n".join(["    - {:<14} : {:<100} \n".format(k, v) for k, v in win_info.items()])


nnkp_header = """
=== data in the *.nnkp file [default] === \n
"""
nnkp_header += "\n".join(["    - {:<10} : {:<100} \n".format(k, v) for k, v in nnkp_info.items()])


eig_header = """
=== data in the *.eig file [default] === \n
"""
eig_header += "\n".join(["    - {:<9} : {:<100} \n".format(k, v) for k, v in eig_info.items()])


amn_header = """
=== data in the *.amn file [default] === \n
"""
amn_header += "\n".join(["    - {:<9} : {:<100} \n".format(k, v) for k, v in amn_info.items()])


mmn_header = """
=== data in the *.mmn file [default] === \n
"""
mmn_header += "\n".join(["    - {:<9} : {:<100} \n".format(k, v) for k, v in mmn_info.items()])


umat_header = """
=== data in the *_u.mat and *_u_dis.mat files [default] === \n
"""
umat_header += "\n".join(["    - {:<9} : {:<100} \n".format(k, v) for k, v in umat_info.items()])

cw_info_header = """
=== information in CWModel [default]===
"""
cw_info_header += "\n".join(["    - {:<9} : {:<100} \n".format(k, v) for k, v in cw_info.items()])


cw_data = {
    "Sk": "Overlap matrix elements in k-space (ndarray).",
    "Hk": "Hamiltonian matrix elements in k-space (orthogonal) (ndarray).",
    "Hk_nonortho": "Hamiltonian matrix elements in k-space (non-orthogonal) (ndarray).",
    #
    "Sr": "Overlap matrix elements in real-space (ndarray).",
    "Hr": "Hamiltonian matrix elements in real-space (orthogonal) (ndarray).",
    "Hr_nonortho": "Hamiltonian matrix elements in real-space (non-orthogonal) (ndarray).",
    #
    "s": "The expansion coefficients of Sk expressed by a linear combination of SAMBs (ndarray).",
    "z": "The expansion coefficients of Hk expressed by a linear combination of SAMBs (ndarray).",
    "z_nonortho": "The expansion coefficients of Hk_nonortho expressed by a linear combination of SAMBs (ndarray).",
    #
    "Sk_sym": "Symmetrized overlap matrix elements in k-space (ndarray).",
    "Hk_sym": "Symmetrized Hamiltonian matrix elements in k-space (orthogonal) (ndarray).",
    "Hk_nonortho_sym": "Symmetrized Hamiltonian matrix elements in k-space (non-orthogonal) (ndarray).",
    "Sr_sym": "Symmetrized overlap matrix elements in real-space (ndarray).",
    "Hr_sym": "Symmetrized Hamiltonian matrix elements in real-space (orthogonal) (ndarray).",
    "Hr_nonortho_sym": "Symmetrized Hamiltonian matrix elements in real-space (non-orthogonal) (ndarray).",
    #
    "rpoints_mp": "lattice points data included in matrix_dict, [[r1, r2, r3]] (crystal coordinate) (ndarray).",
    #
    "Ek_RMSE_grid": "mean squared error of eigen energies between symmetrized and non-symmetrized closed wannier TB model (grid).",
    "Ek_RMSE_path": "mean squared error of eigen energies between symmetrized and non-symmetrized closed wannier TB model (path).",
    #
    "matrix_dict": "SAMBs matrix",
}

cw_data_header = """
=== data in CWModel [default]===
"""
cw_data_header += "\n".join(["    - {:<15} : {:<100} \n".format(k, v) for k, v in cw_data.items()])


kpoints_header = "# k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate)."


rpoints_header = "# lattice points data, [[r1, r2, r3]] (crystal coordinate)."


hk_header = """
=== Hamiltonian matrix elements in k-space ===
- {(k2,k2,k3,a,b) = H_{ab}(k)}.
- H_{ab}(k)      : <φ_{a}(k)|H|φ_{b}(k)>.
- φ_{a}(k)       : orthogonalized pseudo atomic orbital.
- k = (k1,k2,k3) : k points (crystal coordinate).
"""


sk_header = """
=== Overlap matrix elements in k-space ===
- {(k2,k2,k3,a,b) = S_{ab}(k)}.
- S_{ab}(k)      : <φ_{a}(k)|φ_{b}(k)>.
- φ_{a}(k)       : non-orthogonalized pseudo atomic orbital.
- k = (k1,k2,k3) : k points (crystal coordinate).
"""


pk_header = """
=== Projectability of each Kohn-Sham state in k-space ===
- {(k2,k2,k3,n,n) = P_{n}(k)}.
- P_{n}(k)       : Σ_{a} |<ψ_{n}(k)|φ_{a}(k)>|^2.
- ψ_{n}(k)       : Kohn-Sham orbital corresponding to the n-th Kohn-Sham energy ε_{n}(k).
- φ_{a}(k)       : non-orthogonalized pseudo atomic orbital.
- k = (k1,k2,k3) : k points (crystal coordinate).
"""


hr_header = """
=== Hamiltonian matrix elements in real-space ===
- n1 n2 n3 a b re(H_{ab}(R)) im(H_{ab}(R))
    - H_{ab}(R)      : <φ_{a}(R)|H|φ_{b}(0)>.
    - φ_{a}(R)       : orthogonalized pseudo atomic orbital.
    - R = (n1,n2,n3) : lattice points (crystal coordinate, nj: integer).
"""

sr_header = """
=== Overlap matrix elements in real-space ===
- n1 n2 n3 a b re(S_{ab}(R)) im(S_{ab}(R))
    - S_{ab}(R)      : <φ_{a}(R)|φ_{b}(0)>.
    - φ_{a}(R)       : non-orthogonalized pseudo atomic orbital.
    - R = (n1,n2,n3) : lattice points (crystal coordinate, nj: integer).
"""


z_header = """
=== The expansion coefficients of the Hamiltonian matrix (orthogonal) H expressed by a linear combination of SAMBs ===
- j z_j TagMultipole coefficient
    - H(R) ~ sum_{j} z_j Z_j(R)
    - z_j = \sum_{R} Tr[Z_j(R)*H(R)].
    - z_j is the expansion coefficients.
"""

z_nonortho_header = """
=== The expansion coefficients of the Hamiltonian matrix (non-orthogonal) H' expressed by a linear combination of SAMBs ===
- j z_j TagMultipole coefficient
    - H'(R) ~ sum_{j} z_j Z_j(R)
    - z_j = \sum_{R} Tr[Z_j(R)*H'(R)].
    - z_j is the expansion coefficients.
"""


s_header = """
=== The expansion coefficients of the Overlap matrix expressed by a linear combination of SAMBs ===
- j z_j TagMultipole coefficient
    - S(R) ~ sum_{j} z_j Z_j(R)
    - z_j = \sum_{R} Tr[Z_j(R)*S(R)].
    - z_j is the expansion coefficients.
"""
