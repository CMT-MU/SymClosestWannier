"""
CWin manages input file for pw2cw, seedname.cwin file.
"""

import os
import difflib

from symclosestwannier.util.exceptions import SymCWFileNotFoundError, SymCWInputError


# keywords given by space-separated integers, "list" or "range" ([[min_1, max_1, N1], [min_2, max_2, N2]]).
_mesh_keys = {
    "dos_kmesh": "list",
    "cohp_kmesh": "list",
    "lindhard_kmesh": "list",
    "fermi_surface_kmesh": "range",
    "fermi_surface_view": "list",
    "lindhard_surface_qmesh": "range",
    "lindhard_surface_view": "list",
}


_default = {
    "seedname": "cwannier",
    "outdir": "./",
    "restart": "cw",
    #
    "disentangle": False,
    "proj_min": 0.0,
    "cwf_mu_max": None,
    "cwf_mu_min": None,
    "cwf_sigma_max": 1.0,
    "cwf_sigma_min": 0.0,
    "cwf_delta": 0.0,
    "svd": False,
    #
    "verbose": False,
    "parallel": False,
    "formatter": False,
    #
    "calc_spreads": False,
    #
    "transl_inv": False,
    "use_degen_pert": False,
    "degen_thr": 1e-4,
    "tb_gauge": False,
    "tb_position": False,
    #
    "write_info_data": False,
    "write_hr": False,
    "write_sr": False,
    "write_u_matrices": False,
    "write_rmn": False,
    "write_vmn": False,
    "write_tb": False,
    "write_eig": False,
    "write_amn": False,
    "write_mmn": False,
    "write_spn": False,
    # symmetrization
    "symmetrization": False,
    "mp_outdir": "./",
    "mp_seedname": "default",
    "ket_amn": None,
    "irreps": "all",
    # band
    "a": None,
    "N1": 50,
    "calc_spin_2d": False,
    # dos
    "calc_dos": False,
    "dos_kmesh": [1, 1, 1],
    "dos_num_fermi": 50,
    "dos_smr_en_width": 0.001,
    "dos_emax": None,
    "dos_emin": None,
    # cohp
    "calc_cohp": False,
    "cohp_kmesh": [1, 1, 1],
    "cohp_bond_length_max": 6,
    "cohp_bond_length_min": 0,
    "cohp_head_atom": None,
    "cohp_tail_atom": None,
    "cohp_head_atom_idx": None,
    "cohp_tail_atom_idx": None,
    "cohp_num_fermi": 50,
    "cohp_smr_en_width": 0.001,
    "cohp_emax": None,
    "cohp_emin": None,
    "calc_cohp_samb_decomp": False,
    # zeeman interaction
    "zeeman_interaction": False,
    "magnetic_field": 0.0,
    "magnetic_field_theta": 0.0,
    "magnetic_field_phi": 0.0,
    "g_factor": 2.0,
    # fermi surface
    "fermi_surface": False,
    "fermi_surface_kmesh": [[-1, 1, 10], [-1, 1, 10]],
    "fermi_surface_view": [0, 0, 1],
    "fermi_surface_const": 0.0,
    # lindhard
    "lindhard": False,
    "lindhard_freq": 0.0,
    "lindhard_smr_fixed_en_width": 0.01,
    "lindhard_kmesh": [1, 1, 1],
    "qpoint": None,
    "qpoint_path": None,
    "Nq1": 30,
    "filling": None,
    "temperature": 0.0,
    # lindhard (surface)
    "lindhard_surface": False,
    "lindhard_surface_qmesh": [[-1, 1, 10], [-1, 1, 10]],
    "lindhard_surface_view": [0, 0, 1],
    "lindhard_surface_const": 0.0,
    # postcw
    "hr_input": "",
    "use_tb_approximation": False,
}


# ==================================================
class CWin(dict):
    """
    CWin manages input file for pw2cw, seedname.cwin file.

    Attributes:
        _topdir (str): top directory.
        _seedname (str): seedname.
    """

    # ==================================================
    def __init__(self, topdir=None, seedname="cwannier", dic=None):
        """
        CWin manages input file for pw2cw, seedname.cwin file.

        Args:
            topdir (str, optional): directory of seedname.cwin file.
            seedname (str, optional): seedname.
            dic (dict, optional): dictionary of CWin.
        """
        super().__init__()

        self._topdir = topdir
        self._seedname = seedname

        if dic is None:
            file_name = os.path.join(topdir, "{}.{}".format(seedname, "cwin"))
            self.update(self.read(file_name))
            self["seedname"] = seedname

            # output directories are relative to the directory of seedname.cwin.
            for key in ("outdir", "mp_outdir"):
                self[key] = os.path.normpath(os.path.join(os.path.abspath(topdir), self[key]))
        else:
            self.update(dic)

    # ==================================================
    def read(self, file_name="cwannier.cwin"):
        """
        read seedname.cwin file.

        Args:
            file_name (str, optional): file name.

        Returns:
            dict: dictionary form of seedname.cwin.
                - seedname          : seedname (str), ["cwannier"].
                - outdir            : output files are written to this directory, relative to the directory of seedname.cwin (str), ["./"].
                - restart           : the restart position 'cw'/'w90' (str), ["cw"].
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
                - transl_inv        : use Eq.(31) of Marzari&Vanderbilt PRB 56, 12847 (1997) for band-diagonal position matrix elements? (bool), [False].
                - use_degen_pert    : use degenerate perturbation theory when bands are degenerate and band derivatives are needed? (bool), [False].
                - degen_thr         : threshold to exclude degenerate bands from the calculation, [0.0001].
                - tb_gauge          : use tb gauge? (bool), [False].
                - tb_position       : approximate the position operator of seedname_r.dat and seedname_tb.dat by the projection centres, r_ab(R) = tau_a delta_ab delta_R0 (seedname.mmn is then not needed) ? (bool), [False].
                - write_info_data   : write info and data to seedname.hdf5 (hdf5 format) ? (bool), [False].
                - write_hr          : write seedname_hr.py ? (bool), [False].
                - write_sr          : write seedname_sr.py ? (bool), [False].
                - write_u_matrices  : write seedname_u.dat and seedname_u_dis.dat ? (bool), [False].
                - write_rmn         : write seedname_r.dat (needs seedname.mmn unless tb_position = true) ? (bool), [False].
                - write_vmn         : write seedname_v.dat ? (bool), [False].
                - write_tb          : write seedname_tb.dat (wannier90 format, needs seedname.mmn unless tb_position = true, not with tb_gauge = true) ? (bool), [False].
                - write_eig         : write seedname.eig.cw ? (bool), [False].
                - write_amn         : write seedname.amn.cw ? (bool), [False].
                - write_mmn         : write seedname.mmn.cw ? (bool), [False].
                - write_spn         : write seedname.spn.cw ? (bool), [False].

            # only used for symmetrization.
                - symmetrization    : symmetrize ? (bool), [False].
                - mp_outdir         : output files for multipie are written to this directory, relative to the directory of seedname.cwin (str). ["./"].
                - mp_seedname       : seedname for seedname_model.py, seedname_samb.py and seedname_matrix.py files (str), ["default"].
                - ket_amn           : ket basis list in the seedname.amn file. If ket_amn == auto, the list of orbitals are set automatically, or it can be set manually. The format of each ket must be same as the "ket" in sambname_model.py file. See sambname["info"]["ket"] in sambname_model.py file for the format (list), [None].
                - irreps            : list of irreps to be considered (str/list), ["all"].

            # only used for band dispersion calculation.
                - a                 : lattice parameter (in Ang) used to correct units of k points in reference band data, [None].
                - N1                : number of divisions for high symmetry lines (int, optional), [50].
                - calc_spin_2d      : add expectation value of spin operator given by *.spn in output file ? (bool). [False].

            # only used for dos calculation.
                - calc_dos          : calculate dos? (bool), [False].
                - dos_kmesh         : dimensions of the Monkhorst-Pack grid of k-points for dos calculation (list), [[1, 1, 1]].
                - dos_num_fermi     : number of fermi energies (int), [50].
                - dos_smr_en_width  : Energy width for the smearing function for the DOS (The units are [eV]) (flaot), [0.001].
                - dos_emax          : maximun energy to be calculated (flaot), [None].
                - dos_emin          : minimum energy to be calculated (flaot), [None].

            # only used for cohp calculation.
                - calc_cohp             : calculate cohp? (bool), [False].
                - cohp_kmesh            : dimensions of the Monkhorst-Pack grid of k-points for cohp calculation (list), [[1, 1, 1]].
                - cohp_bond_length_max  : maximum bond length (ang) to be calculated (float), [6.0].
                - cohp_bond_length_min  : minimum bond length (ang) to be calculated (float), [0.0].
                - cohp_head_atom        : head atom to be calculated (float), [None].
                - cohp_tail_atom        : tail atom to be calculated (float), [None].
                - cohp_head_atom_idx    : head atom index to be calculated (float), [None].
                - cohp_tail_atom_idx    : tail atom index to be calculated (float), [None].
                - cohp_num_fermi        : number of fermi energies (int), [50].
                - cohp_smr_en_width     : Energy width for the smearing function for the COHP (The units are [eV]) (flaot), [0.001].
                - cohp_emax             : maximun energy to be calculated (flaot), [None].
                - cohp_emin             : minimum energy to be calculated (flaot), [None].
                - calc_cohp_samb_decomp : decompose cohp by SAMBs? (bool), [False].

            # only used for when zeeman interaction is considered.
                - zeeman_interaction   : consider zeeman interaction ? (bool), [False].
                - magnetic_field       : strength of the magnetic field (float), [0.0].
                - magnetic_field_theta : angle from the z-axis of the magnetic field (float), [0.0].
                - magnetic_field_phi   : angle from the x-axis of the magnetic field (float), [0.0].
                - g_factor             : spin g factor (float), [2.0].

            # only used for fermi surface calculation.
                - fermi_surface        : calculate fermi surface? (bool), [False].
                - fermi_surface_kmesh  : 2d kmesh given by [[k1_min, k1_max, N1], [k2_min, k2_max, N2] ] (crystal coordinate), (list), [ [[-1, 1, 10], [-1, 1, 10]] ].
                - fermi_surface_view   : k3 direction (list), [ [0, 0, 1] ].
                - fermi_surface_const  : constant value for k3 axis [0.0].

            # only used for lindhard function.
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

            # only used for postcw calculation.
                - hr_input             : full filename of hr.dat file (str), [""].
                - use_tb_approximation : use tight-binding approximation? (bool), [False].

        """
        if os.path.exists(file_name):
            with open(file_name) as fp:
                cwin_data = fp.readlines()
        else:
            raise SymCWFileNotFoundError("cwin", file_name)

        cwin_data = [v.replace("\n", "") for v in cwin_data]
        cwin_data_lower = [v.lower().replace("\n", "") for v in cwin_data]

        # default
        d = CWin._default().copy()

        for line in cwin_data:
            line = line.replace("\n", "")
            line = line.lstrip()

            if len([vi for vi in line.split(" ") if vi != ""]) == 0:
                continue

            if "begin qpoint_path" in line:
                try:
                    q_data = cwin_data[
                        cwin_data_lower.index("begin qpoint_path") + 1 : cwin_data_lower.index("end qpoint_path")
                    ]
                    q_data = [[vi for vi in v.split()] for v in q_data]
                    qpoint = {}
                    qpoint_path = ""
                    cnt = 1
                    for X, qi1, qi2, qi3, Y, qf1, qf2, qf3 in q_data:
                        if cnt == 1:
                            qpoint_path += f"{X}-{Y}-"
                        else:
                            if qpoint_path.split("-")[-2] == X:
                                qpoint_path += f"{Y}-"
                            else:
                                qpoint_path = qpoint_path[:-1]
                                qpoint_path += f"|{X}-{Y}-"
                        if X not in qpoint:
                            qpoint[X] = [float(qi1), float(qi2), float(qi3)]
                        if Y not in qpoint:
                            qpoint[Y] = [float(qf1), float(qf2), float(qf3)]

                        cnt += 1

                    qpoint_path = qpoint_path[:-1]
                except (ValueError, IndexError) as e:
                    raise SymCWInputError(f"{file_name}: invalid qpoint_path block ({e}).") from e

                d["qpoint"] = qpoint
                d["qpoint_path"] = qpoint_path
                continue

            if "=" in line:
                key = line.split("=")[0]
            elif ":" in line:
                key = line.split(":")[0]
            else:
                continue

            key = key.replace(" ", "")

            if "!" in key or "#" in key:
                continue

            if "=" in line:
                v = line.split("=")[1].split("!")[0]
            elif ":" in line:
                v = line.split(":")[1].split("!")[0]

            try:
                if key in _mesh_keys:
                    d[key] = CWin._str_to_mesh(key, v)
                    continue

                v = v.replace(" ", "")

                if key == "ket_amn":
                    if "[" in v or "]" in v:
                        v = "".join(v)

                if key == "optimize_params_fixed":
                    if "[" in v or "]" in v:
                        v = "".join(v)

                d[key] = self._str_to(key, v)
            except SymCWInputError as e:
                raise SymCWInputError(f"{file_name}: {e}") from e
            except (ValueError, IndexError) as e:
                raise SymCWInputError(f"{file_name}: invalid value {key} = {v.strip()} ({e}).") from e

        if d["disentangle"] and (d["cwf_mu_max"] is None or d["cwf_mu_min"] is None):
            raise SymCWInputError(f"{file_name}: cwf_mu_max and cwf_mu_min must be specified when disentangle = true.")

        if d["cwf_mu_max"] is not None and d["cwf_mu_min"] is not None:
            if d["cwf_mu_max"] < d["cwf_mu_min"]:
                raise SymCWInputError(
                    f"{file_name}: check disentanglement windows, cwf_mu_max = {d['cwf_mu_max']} < cwf_mu_min = {d['cwf_mu_min']}."
                )

        if d["write_tb"] and d["tb_gauge"]:
            raise SymCWInputError(f"{file_name}: write_tb = true cannot be used with tb_gauge = true.")

        return d

    # ==================================================
    @classmethod
    def _str_to_mesh(cls, key, v):
        """
        convert space-separated integers of mesh/view keywords.

        Args:
            key (str): keyword in _mesh_keys.
            v (str): value, e.g. "1 1 1" or "-1 1 10 -1 1 10".

        Returns:
            list: [n1, n2, ...] or [[min_1, max_1, N1], [min_2, max_2, N2]].
        """
        v = [int(x) for x in v.split() if x != ""]
        if _mesh_keys[key] == "range":
            if len(v) != 6:
                raise SymCWInputError(f"{key} needs 6 integers, min_1 max_1 N1 min_2 max_2 N2.")
            v = [v[:3], v[3:]]

        return v

    # ==================================================
    def _str_to(self, key, v):
        v = str(v).replace("'", "").replace('"', "")

        if key not in CWin._default().keys():
            from symclosestwannier.cw.win import Win

            msg = f"invalid keyword = {key} was given."
            close = difflib.get_close_matches(key, CWin._default().keys(), n=3)
            if key in Win._default().keys():
                msg += f" {key} must be given in seedname.win."
            elif close:
                msg += " did you mean " + "/".join(f"'{k}'" for k in close) + "?"
            msg += " run 'pw2cw -i' to list the keywords."
            raise SymCWInputError(msg)
        elif key in ("seedname", "mp_seedname", "hr_input", "lindhard_smr_type", "cohp_head_atom", "cohp_tail_atom"):
            pass
        elif key in ("outdir", "mp_outdir"):
            v = v[:-1] if v[-1] == "/" else v
        elif key == "restart":
            if v not in ("cw", "w90"):
                raise SymCWInputError(f"invalid restart = {v} was given. choose from 'cw'/'w90'.")
        elif key in (
            "proj_min",
            "cwf_mu_max",
            "cwf_mu_min",
            "cwf_sigma_max",
            "cwf_sigma_min",
            "cwf_delta",
            "a",
            "dos_smr_en_width",
            "dos_emax",
            "dos_emin",
            "cohp_bond_length_max",
            "cohp_bond_length_min",
            "cohp_smr_en_width",
            "cohp_emax",
            "cohp_emin",
            "degen_thr",
            "fermi_surface_const",
            "magnetic_field",
            "magnetic_field_theta",
            "magnetic_field_phi",
            "g_factor",
            "lindhard_freq",
            "lindhard_smr_fixed_en_width",
            "filling",
            "temperature",
            "lindhard_surface_const",
        ):
            v = float(v)
            if key == "cwf_delta":
                if v > 1e-5:
                    raise SymCWInputError(f"cwf_delta is too large. cwf_delta must be less than 1e-5.")
            elif key == "lindhard_smr_fixed_en_width":
                pass
                # if v == 0.0:
                #     raise Exception(f"lindhard_smr_fixed_en_width must be > 0.0.")
            elif key == "temperature":
                if v < 0.0:
                    raise SymCWInputError(f"temperature must be positive value.")
            elif key == "filling":
                if v < 0.0:
                    raise SymCWInputError(f"filling must be positive value.")
        elif key in ("N1", "Nq1", "dos_num_fermi", "cohp_head_atom_idx", "cohp_tail_atom_idx", "cohp_num_fermi"):
            v = int(v)
        elif key == "ket_amn":
            v = v.replace(" ", "")
            if v == "auto":
                pass
            else:
                raw = v
                try:
                    if "(" in v or ")" in v:
                        v = [[oi.replace("]", "") for oi in o[1:].split(",")] for o in v[1:-1].split("],")]
                        v = [[str(o[0]), int(o[1]), int(o[2]), str(o[3] + "," + o[4])] for o in v]
                    else:
                        v = [[oi.replace("]", "") for oi in o[1:].split(",")] for o in v[1:-1].split("],")]
                        v = [[str(o[0]), int(o[1]), int(o[2]), str(o[3])] for o in v]
                except (ValueError, IndexError):
                    raise SymCWInputError(
                        f"invalid ket_amn = {raw} was given. each ket must be [atom, sublattice, rank, orbital], "
                        "e.g. [[C,1,1,pz],[C,2,1,pz]] or [[C,1,1,(pz,U)],[C,1,1,(pz,D)]] for spinful case, or use ket_amn = auto."
                    ) from None
        elif key == "optimize_params_fixed":
            if "[" in v or "]" in v:
                v = [str(o) for o in v[1:-1].split(",")]
            if len(v) > 0:
                for vi in v:
                    if vi not in ("cwf_mu_min", "cwf_mu_max", "cwf_sigma_min", "cwf_sigma_max", "cwf_delta"):
                        raise SymCWInputError(
                            f"invalid optimize_params_fixed: {vi} was given. choose from 'cwf_mu_min'/'cwf_mu_max'/'cwf_sigma_min'/'cwf_sigma_max'/'cwf_delta'."
                        )

        elif key == "irreps":
            if "[" in v and "]" in v:
                v = [str(o) for o in v[1:-1].split(",")]
            else:
                if v not in ("all", "full"):
                    raise SymCWInputError(f"invalid irreps = {v} was given. choose from 'all'/'full'.")
        else:
            if v.lower() in ("true", ".true."):
                v = True
            elif v.lower() in ("false", ".false."):
                v = False
            else:
                raise SymCWInputError(f"invalid {key} = {v} was given. choose from 'true'/'.true.'/'false'/'.false.'.")

        return v

    # ==================================================
    @classmethod
    def _default(cls):
        return _default
