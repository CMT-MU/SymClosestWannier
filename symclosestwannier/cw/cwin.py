"""
CWin manages input file for pw2cw, seedname.cwin file.
"""

import os


_default = {
    "seedname": "cwannier",
    "outdir": "./",
    "restart": "cw",
    #
    "disentangle": False,
    "proj_min": 0.0,
    "dis_win_emax": None,
    "dis_win_emin": None,
    "smearing_temp_max": 1.0,
    "smearing_temp_min": 0.0,
    "delta": 0.0,
    "svd": False,
    "optimize_params": False,
    "optimize_params_fixed": [],
    #
    "verbose": False,
    "parallel": False,
    "formatter": False,
    #
    "transl_inv": False,
    "use_degen_pert": False,
    "degen_thr": 1e-4,
    "tb_gauge": False,
    #
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
    #
    "symmetrization": False,
    "mp_outdir": "./",
    "mp_seedname": "default",
    "ket_amn": None,
    "irreps": "all",
    "z_exp": False,
    "z_exp_kmesh": [1, 1, 1],
    "z_exp_temperature": 0.0,
    # band
    "a": None,
    "N1": 50,
    "calc_spin_2d": False,
    # dos
    "calc_dos": False,
    "dos_kmesh": [1, 1, 1],
    "dos_num_fermi": 50,
    "dos_smr_en_width": 0.001,
    #
    "zeeman_interaction": False,
    "magnetic_field": 0.0,
    "magnetic_field_theta": 0.0,
    "magnetic_field_phi": 0.0,
    "g_factor": 2.0,
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
        else:
            self.update(dic)

    # ==================================================
    def read(self, file_name="cwannier.cwin"):
        """
        read seedname.cwin file.

        Args:d
            file_name (str, optional): file name.

        Returns:
            dict: dictionary form of seedname.cwin.
                - seedname          : seedname (str), ["cwannier"].
                - outdir            : output files are found in this directory (str), ["./"].
                - restart           : the restart position 'cw'/'w90' (str), ["cw"].
                - disentangle       : disentagle bands ? (bool), [False].
                - proj_min          : minimum value of projectability: [0.0].
                - dis_win_emax      : upper energy window (float), [None].
                - dis_win_emin      : lower energy window (float), [None].
                - smearing_temp_max : smearing temperature for upper window (float), [1.0].
                - smearing_temp_min : smearing temperature for lower window (float), [0.0].
                - delta             : small constant to avoid ill-conditioning of overlap matrices (< 1e-5) (float), [0.0].
                - svd               : implement singular value decomposition ? otherwise adopt Lowdin's orthogonalization method (bool), [False].
                - optimize_params": optimize the energy windows and smearing temperatures? (bool), [False].
                - optimize_params_fixed":  fixed parameters for optimization (ex: ['dis_win_emin', 'smearing_temp_min']), (list) [].
                - verbose           : verbose calculation info (bool, optional), [False].
                - parallel          : use parallel code? (bool), [False].
                - formatter         : format by using black? (bool), [False].
                - transl_inv        : use Eq.(31) of Marzari&Vanderbilt PRB 56, 12847 (1997) for band-diagonal position matrix elements? (bool), [False].
                - use_degen_pert    : use degenerate perturbation theory when bands are degenerate and band derivatives are needed? (bool), [False].
                - degen_thr         : threshold to exclude degenerate bands from the calculation, [0.0].
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

            # only used for symmetrization.
                - symmetrization    : symmetrize ? (bool), [False].
                - mp_outdir         : output files for multipie are found in this directory (str). ["./"].
                - mp_seedname       : seedname for seedname_model.py, seedname_samb.py and seedname_matrix.py files (str), ["default"].
                - ket_amn           : ket basis list in the seedname.amn file. If ket_amn == auto, the list of orbitals are set automatically, or it can be set manually. The format of each ket must be same as the "ket" in sambname_model.py file. See sambname["info"]["ket"] in sambname_model.py file for the format (list), [None].
                - irreps            : list of irreps to be considered (str/list), ["all"].
                - z_exp             : calculate the expectation value of the SAMB operators? (bool), [False].
                - z_exp_kmesh       : dimensions of the Monkhorst-Pack grid of k-points for z_exp calculation (list), [[1, 1, 1]].
                - z_exp_temperature : temperature for which we want to calculate the expectation value of the SAMB operators (float), [0.0].

            # only used for band dispersion calculation.
                - a                 : lattice parameter (in Ang) used to correct units of k points in reference band data, [None].
                - N1                : number of divisions for high symmetry lines (int, optional), [50].
                - calc_spin_2d      : add expectation value of spin operator given by *.spn in output file ? (bool). [False].

            # only used for dos calculation.
                - calc_dos          : calculate dos? (bool), [False].
                - dos_kmesh         : dimensions of the Monkhorst-Pack grid of k-points for dos calculation (list), [[1, 1, 1]].
                - dos_num_fermi     : number of fermi energies (int), [50].
                - dos_smr_en_width  : Energy width for the smearing function for the DOS (The units are [eV]) (flaot), [0.001].

            # only used for when zeeman interaction is considered.
                - zeeman_interaction   : consider zeeman interaction ? (bool), [False].
                - magnetic_field       : strength of the magnetic field (float), [0.0].
                - magnetic_field_theta : angle from the z-axis of the magnetic field (float), [0.0].
                - magnetic_field_phi   : angle from the x-axis of the magnetic field (float), [0.0].
                - g_factor             : spin g factor (float), [2.0].
        """
        if os.path.exists(file_name):
            with open(file_name) as fp:
                cwin_data = fp.readlines()
        else:
            raise Exception("failed to read cwin file: " + file_name)

        # default
        d = CWin._default().copy()

        for line in cwin_data:
            line = line.replace("\n", "")
            line = line.lstrip()

            if len([vi for vi in line.split(" ") if vi != ""]) == 0:
                continue

            key = line.split("=")[0]
            key = key.replace(" ", "")

            if "!" in key or "#" in key:
                continue

            if "=" in line:
                v = line.split("=")[1].split("!")[0]
            elif ":" in line:
                v = line.split(":")[1].split("!")[0]

            if key == "z_exp_kmesh":
                d["z_exp_kmesh"] = [int(x) for x in v.split() if x != ""]
                continue

            if key == "dos_kmesh":
                d["dos_kmesh"] = [int(x) for x in v.split() if x != ""]
                continue

            v = v.replace(" ", "")

            if key == "ket_amn":
                if "[" in v or "]" in v:
                    v = "".join(v)

            if key == "optimize_params_fixed":
                if "[" in v or "]" in v:
                    v = "".join(v)

            d[key] = self._str_to(key, v)
        # assert not (
        #    d["restart"] == "w90" and d["symmetrization"]
        # ), "Symmetrization cannot be performed when restart == w90."

        assert not (
            d["disentangle"] and (d["dis_win_emax"] is None or d["dis_win_emin"] is None)
        ), "dis_win_emax and dis_win_emin must be specified when disentangle == true."

        if d["dis_win_emax"] is not None and d["dis_win_emin"] is not None:
            assert not (
                d["dis_win_emax"] < d["dis_win_emin"]
            ), "check disentanglement windows (dis_win_emax < dis_win_emin !)"

        return d

    # ==================================================
    def _str_to(self, key, v):
        v = str(v).replace("'", "").replace('"', "")

        if key not in CWin._default().keys():
            raise Exception(f"invalid keyword = {key} was given.")
        elif key in ("seedname", "mp_seedname"):
            pass
        elif key in ("outdir", "mp_outdir"):
            v = v[:-1] if v[-1] == "/" else v
        elif key == "restart":
            if v not in ("cw", "w90"):
                raise Exception(f"invalid restart = {v} was given. choose from 'cw'/'w90'.")
        elif key in (
            "proj_min",
            "dis_win_emax",
            "dis_win_emin",
            "smearing_temp_max",
            "smearing_temp_min",
            "delta",
            "z_exp_temperature",
            "a",
            "dos_smr_en_width",
            "degen_thr",
            "magnetic_field",
            "magnetic_field_theta",
            "magnetic_field_phi",
            "g_factor",
        ):
            v = float(v)
            if key == "delta":
                if v > 1e-5:
                    raise Exception(f"delta is too large. delta must be less than 1e-5.")
        elif key in ("N1", "dos_num_fermi"):
            v = int(v)
        elif key == "ket_amn":
            if "[" in v or "]" in v:
                if "(" in str(v) and ")" in str(v):
                    v = [str(o) if i == 0 else f"({str(o)}" for i, o in enumerate(v[1:-1].split(",("))]
                else:
                    v = [str(o) for o in v[1:-1].split(",")]
        elif key == "optimize_params_fixed":
            if "[" in v or "]" in v:
                v = [str(o) for o in v[1:-1].split(",")]
            if len(v) > 0:
                for vi in v:
                    if vi not in ("dis_win_emin", "dis_win_emax", "smearing_temp_min", "smearing_temp_max", "delta"):
                        raise Exception(
                            f"invalid optimize_params_fixed: {vi} was given. choose from 'dis_win_emin'/'dis_win_emax'/'smearing_temp_min'/'smearing_temp_max'/'delta'."
                        )

        elif key == "irreps":
            if "[" in v and "]" in v:
                v = [str(o) for o in v[1:-1].split(",")]
            else:
                if v not in ("all", "full"):
                    raise Exception(f"invalid irreps = {v} was given. choose from 'all'/'full'.")
        else:
            if v.lower() in ("true", ".true."):
                v = True
            elif v.lower() in ("false", ".false."):
                v = False
            else:
                raise Exception(f"invalid {key} = {v} was given. choose from 'true'/'.true.'/'false'/'.false.'.")

        return v

    # ==================================================
    @classmethod
    def _default(cls):
        return _default
