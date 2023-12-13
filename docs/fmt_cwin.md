# SymClosestWannier

## input parameters (* optional [default] or only for crystal value)
- For execution use: pw2cw [seedname]
  - If a seedname string is given the code will read its input from a file seedname.cwin. The default value is cwannier.
      One can also equivalently provide the string seedname.cwin instead of seedname.

- input parameters in seedname.cwin (same format as seedname.win) (* optional [default])
  - restart*           : the restart position 'cw'/'w90'/'sym' (str), ["wannierize"].
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

- only used for symmetrization.
  - symmetrization*    : symmetrize ? (bool), [False].
  - mp_outdir*         : output files for multipie are found in this directory (str). ["./"].
  - mp_seedname*       : seedname for seedname_model.py, seedname_samb.py and seedname_matrix.py files (str).
  - ket_amn*           : ket basis list in the seedname.amn file. The format of each ket must be same as the "ket" in sambname_model.py file. See sambname["info"]["ket"] in sambname_model.py file for the format (list), [None].
  - irreps*            : list of irreps to be considered (str/list), [None].

- only used for band dispersion calculation.
  - a*                 : lattice parameter (in Ang) used to correct units of k points in reference band data, [1.0].
  - fermi_energy*      : fermi energy used for band shift, [None].
  - N1*                : number of divisions for high symmetry lines (int, optional), [50].
