# seedname.cwin [default]
- For execution use: pw2cw [seedname]
  - If a seedname string is given the code will read its input from a file seedname.cwin. The default value is cwannier.
      One can also equivalently provide the string seedname.cwin instead of seedname.

- input parameters in seedname.cwin (same format as seedname.win) [default]
  - restart           : the restart position 'cw'/'w90' (str), ["cw"].
  - seedname          : seedname (str), ["cwannier"].
  - outdir            : output files are written to this directory, relative to the directory of seedname.cwin (str), ["./"].
  - disentangle       : disentagle bands ? (bool), [False].
  - proj_min          : minimum value of projectability: [0.0].
  - cwf_mu_max      : top of the energy window (float), [None].
  - cwf_mu_min      : bottom of the energy window (float), [None].
  - cwf_sigma_max : smearing temperature for the top of the energy window (float), [1.0].
  - cwf_sigma_min : smearing temperature for the bottom of the energy window (float), [0.0].
  - cwf_delta             : small constant to avoid ill-conditioning of overlap matrices (< 1e-5) (float), [0.0].
  - svd               : implement singular value decomposition ? otherwise adopt Lowdin's orthogonalization method (bool), [False].
  - verbose           : verbose calculation info (bool, optional), [False].
  - parallel          : use parallel code? (bool), [False].
  - formatter         : format by using black? (bool), [False].
  - calc_spreads      : calculate spreads? (bool), [False].
  - write_info_data   : write info and data to seedname.hdf5 (hdf5 format) ? (bool), [False].
  - transl_inv        : use Eq.(31) of Marzari&Vanderbilt PRB 56, 12847 (1997) for band-diagonal position matrix elements? (bool), [False].
  - use_degen_pert    : use degenerate perturbation theory when bands are degenerate and band derivatives are needed? (bool), [False].
  - degen_thr         : threshold to exclude degenerate bands from the calculation, [0.0001].
  - tb_gauge          : use tb gauge? (bool), [False].
  - write_hr          : write seedname_hr.py ? (bool), [False].
  - write_sr          : write seedname_sr.py ? (bool), [False].
  - write_u_matrices  : write seedname_u.dat and seedname_u_dis.dat ? (bool), [False].
  - write_rmn         : write seedname_r.dat ? (bool), [False].
  - write_vmn         : write seedname_v.dat ? (bool), [False].
  - write_tb          : write seedname_tb.dat ? (bool), [False].
  - write_eig         : write seedname.eig.cw ? (bool), [False].
  - write_amn         : write seedname.amn.cw ? (bool), [False].
  - write_mmn         : write seedname.mmn.cw ? (bool), [False].
  - write_spn         : write seedname.spn.cw ? (bool), [False].

- only used for symmetrization.
  - symmetrization    : symmetrize ? (bool), [False].
  - mp_outdir         : output files for multipie are written to this directory, relative to the directory of seedname.cwin (str). ["./"].
  - mp_seedname       : seedname for seedname_model.py, seedname_samb.py and seedname_matrix.py files (str).
  - ket_amn           : ket basis list in the seedname.amn file. If ket_amn == auto, the list of orbitals are set automatically, or it can be set manually. The format of each ket must be same as the "ket" in sambname_model.py file. See sambname["info"]["ket"] in sambname_model.py file for the format (list), [None].
  - irreps            : list of irreps to be considered (str/list), ["all"].

- only used for band dispersion calculation.
  - a                 : lattice parameter (in Ang) used to correct units of k points in reference band data, [None].
  - N1                : number of divisions for high symmetry lines (int, optional), [50].
  - calc_spin_2d      : add expectation value of spin operator given by *.spn in output file ? (bool). [False].

- only used for dos calculation.
  - calc_dos          : calculate dos? (bool), [False].
  - dos_kmesh         : dimensions of the Monkhorst-Pack grid of k-points for dos calculation (list), [[1, 1, 1]].
  - dos_num_fermi     : number of fermi energies (int), [50].
  - dos_smr_en_width  : Energy width for the smearing function for the DOS (The units are [eV]) (flaot), [0.001].
  - dos_emax          : maximun energy to be calculated (flaot), [None].
  - dos_emin          : minimum energy to be calculated (flaot), [None].

- only used for cohp calculation.
  - calc_cohp             : calculate cohp? (bool), [False].
  - calc_cohp_samb_decomp : decompose cohp by SAMBs? (bool), [False].
  - cohp_bond_length_max  : maximum bond length (ang) to be calculated (float), [6.0].
  - cohp_bond_length_min  : minimum bond length (ang) to be calculated (float), [0.0].
  - cohp_head_atom        : head atom to be calculated (float), [None].
  - cohp_head_atom_idx    : head atom index to be calculated (float), [None].
  - cohp_tail_atom        : tail atom to be calculated (float), [None].
  - cohp_tail_atom_idx    : tail atom index to be calculated (float), [None].
  - cohp_kmesh            : dimensions of the Monkhorst-Pack grid of k-points for cohp calculation (list), [[1, 1, 1]].
  - cohp_num_fermi        : number of fermi energies (int), [50].
  - cohp_smr_en_width     : Energy width for the smearing function for the COHP (The units are [eV]) (flaot), [0.001].
  - cohp_emax             : maximun energy to be calculated (flaot), [None].
  - cohp_emin             : minimum energy to be calculated (flaot), [None].


- only used for when zeeman interaction is considered.
  - zeeman_interaction   : consider zeeman interaction ? (bool), [False].
  - magnetic_field       : strength of the magnetic field (float), [0.0].
  - magnetic_field_theta : angle from the z-axis of the magnetic field (float), [0.0].
  - magnetic_field_phi   : angle from the x-axis of the magnetic field (float), [0.0].
  - g_factor             : spin g factor (float), [2.0].


- only used for fermi surface calculation.
  - fermi_surface        : calculate fermi surface? (bool), [False].
  - fermi_surface_kmesh  : 2d kmesh given by [[k1_min, k1_max, N1], [k2_min, k2_max, N2] ] (crystal coordinate), (list), [ [[-1, 1, 10], [-1, 1, 10]] ].
  - fermi_surface_view   : k3 direction (list), [ [0, 0, 1] ].
  - fermi_surface_const  : constant value for k3 axis [0.0].

- only used for lindhard function.
  - lindhard                    : calculate lindhard function? (bool), [False].
  - lindhard_freq               : frequency for computing the lindhard function. (The units are [eV]) (float), [0.0].
  - lindhard_smr_fixed_en_width : Overrides the smr_fixed_en_width global variable (float), [0.01].
  - lindhard_kmesh              : dimensions of the Monkhorst-Pack grid of k-points for lindhard function (list), [[1, 1, 1]].
  - lindhard_surface            : calculate lindhard function on a 2d q-plane? (bool), [False].
  - lindhard_surface_qmesh      : 2d qmesh given by [[q1_min, q1_max, N1], [q2_min, q2_max, N2] ] (crystal coordinate), (list), [ [[-1, 1, 10], [-1, 1, 10]] ].
  - lindhard_surface_view       : q3 direction (list), [ [0, 0, 1] ].
  - lindhard_surface_const      : constant value for q3 axis [0.0].
  - qpoint                      : representative q points (dict), [None].
  - qpoint_path                 : q-points along high symmetry line in Brillouin zone, [[k1, k2, k3]] (crystal coordinate) (str), [None].
  - Nq1                         : number of divisions for high symmetry lines (int, optional), [30].
  - filling                     : number of electrons per unit-cell, [None].
  - temperature                 : temperature for caluclating lindhard function (float), [0.0].

- only used for postcw calculation.
  - hr_input             : full filename of hr.dat file (str), [""].
  - use_tb_approximation : use tight-binding approximation? (bool), [False].
