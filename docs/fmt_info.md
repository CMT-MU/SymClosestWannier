# SymClosestWannier

## information in CWModel [default]
- seedname          : seedname (str), [cwannier].
- restart           : the restart position 'cw'/'w90'/'sym' (str), ['cw'].
- outdir            : input and output files are found in this directory (str), ['./'].
- disentangle       : disentagle bands ? (bool), [False].
- proj_min          : minimum value of projectability (float), [0.0].
- dis_win_emax      : upper energy window (float), [None].
- dis_win_emin      : lower energy window (float), [None].
- smearing_temp_max : smearing temperature for upper window (float), [5.0].
- smearing_temp_min : smearing temperature for lower window (float), [0.01].
- delta             : small constant to avoid ill-conditioning of overlap matrices (float), [1e-12].
- svd               : implement singular value decomposition ? otherwise adopt Lowdin's orthogonalization method (bool), [False].
- verbose           : verbose calculation info (bool, optional), [False].
- parallel          : use parallel code? (bool), [False].
- formatter         : format by using black? (bool), [False].
- write_hr          : write seedname_hr.dat ? (bool), [False].
- write_sr          : write seedname_sr.dat ? (bool), [False].
- write_u_matrices  : write seedname_u.dat and seedname_u_dis.dat ? (bool), [False].
- write_rmn         : write seedname_r.dat ? (bool), [False].
- write_vmn         : write seedname_v.dat ? (bool), [False].
- write_tb          : write seedname_tb.dat ? (bool), [False].
- symmetrization    : symmetrize ? (bool), [False].
- mp_outdir         : output files for multipie are found in this directory (str). ['./'].
- mp_seedname       : seedname for seedname_model.py, seedname_samb.py and seedname_matrix.py files (str), ['default'].
- ket_amn           : ket basis list in the seedname.amn file. The format of each ket must be same as the 'ket' in sambname_model.py file. See sambname['info']['ket'] in sambname_model.py file for the format (list), [None].
- irreps            : list of irreps to be considered (str/list), [None].
- a                 : lattice parameter (in Ang) used to correct units of k points in reference band data (float, optional), [1.0].
- N1                : number of divisions for high symmetry lines (int, optional), [50].
- fermi_energy      : fermi energy used for band shift (float, optional), [None].
- num_k             : # of k points (int), [1].
- num_bands         : # of bands passed to the code (int), [1].
- num_wann          : # of WFs (int), [1].
- dis_num_iter      : # of iterations for disentanglement (int), [0].
- num_iter          : # of iterations for maximal localization (int), [200].
- dis_froz_max      : top of the inner (frozen) energy window (float), [+100000].
- dis_froz_min      : bottom of the inner (frozen) energy window (float), [-100000].
- dis_win_max       : top of the outer energy window (float), [+100000].
- dis_win_min       : bottom of the outer energy window (float), [-100000].
- dis_mix_ratio     : mixing ratio during the disentanglement (float), [0.5].
- mp_grid           : dimensions of the Monkhorst-Pack grid of k-points (list), [0, 0, 0].
- kpoints           : k-points, [[k1, k2, k3]] (crystal coordinate) (list), [[[0, 0, 0]]].
- kpoint            : representative k points (dict), [None].
- kpoint_path       : k-points along high symmetry line in Brillouin zonen, [[k1, k2, k3]] (crystal coordinate) (str), [None].
- unit_cell_cart    : transform matrix, [a1,a2,a3], [None].
- atoms_frac        : atomic positions in fractional coordinates with respect to the lattice vectors, {atom: [r1,r2,r3]} [None].
- atoms_cart        : atomic positions in cartesian coordinates, {atom: [rx,ry,rz]} [None].
- A                 : real lattice vectors, A = [a1,a2,a3] (list), [[[1,0,0], [0,1,0], [0,0,1]]].
- B                 : reciprocal lattice vectors, B = [b1,b2,b3] (list), [[[2*pi,0,0], [0,2*pi,0], [0,0,2*pi]]].
- num_atom          : # of atoms (int), [1].
- num_b             : # of b-vectors (int), [1].
- nnkpts            : nearest-neighbour k-points (list), [None].
- nw2n              : atom position index of each WFs (list), [None].
- nw2l              : l specifies the angular part Θlm(θ, φ) (list), [None].
- nw2m              : m specifies the angular part Θlm(θ, φ) (list), [None].
- nw2r              : r specifies the radial part Rr(r) (list), [None].
- atom_orb          : WFs indexes of each atom (list), [None].
- atom_pos          : atom position index of each atom (list), [None].
- atom_pos_r        : atom position of each atom in fractional coordinates with respect to the lattice vectors (list), [None].
- bvec_cart         : b-vectors (cartesian coordinate) (list), [None].
- bvec_crys         : b-vectors (crystal coordinate) (list), [None].
- wb                :  weight for each k-points and nearest-neighbour k-points (list), [None].
- Ek                : Kohn-Sham energies, E_{m}(k) (list), [None].
- Ak                : Overlap matrix elements, A_{mn}(k) = <ψ^{KS}_{m}(k)|φ_{n}(k)> (list), [None].
- Mkb               : Overlap matrix elements, M_{mn}(k,b) = <u^{KS}_{m}(k)|u^{KS}_{n}(k+b)> (list), [None].
- Uoptk             : num_wann×num_wann full unitary matrix (ndarray), [None].
- Udisk             : num_wann×num_bands partial unitary matrix (ndarray), [None].
- Uk                : num_wann×num_bands full unitary matrix (ndarray), [None].
