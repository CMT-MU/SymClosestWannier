"""
CWin manages input file for pw2cw, seedname.cwin file.
"""
import os
import numpy as np


_default_cwin = {
    "seedname": "cwannier",
    "outdir": "./",
    "restart": "wannierise",
    #
    "disentangle": False,
    "proj_min": 0.0,
    "dis_win_emax": None,
    "dis_win_emin": None,
    "smearing_temp_max": 5.0,
    "smearing_temp_min": 0.01,
    "delta": 1e-12,
    "svd": False,
    #
    "verbose": False,
    "parallel": False,
    "formatter": False,
    #
    "write_hr": False,
    "write_sr": False,
    #
    "symmetrization": False,
    "mp_outdir": "./",
    "mp_seedname": "default",
    "ket_amn": None,
    "irreps": "all",
    #
    "a": None,
    "N1": 50,
    "fermi_energy": None,
}


# ==================================================
class CWin(dict):
    """
    CWin manages input file for pw2cw, seedname.cwin file.
    """

    # ==================================================
    def __init__(self, topdir=None, seedname=None, dic=None):
        """
        initialize the class.

        Args:
            topdir (str, optional): directory of seedname.cwin file.
            seedname (str, optional): seedname.
            dic (dict, optional): dictionary of CWin.
        """
        super().__init__()

        if dic is None:
            file_name = os.path.join(topdir, "{}.{}".format(seedname, "cwin"))
            self.update(self.read(file_name))
            self["seedname"] = seedname
        else:
            self.update(dic)

    # ==================================================
    def read(self, file_name):
        """
        read seedname.cwin file.

        Args:d
            file_name (str): file name.

        Returns:
            dict: dictionary form of seedname.cwin.
                - seedname          : seedname (str), ["cwannier"].
                - outdir            : output files are found in this directory (str), ["./"].
                - restart           : the restart position (str), ["wannierize"].
                - disentangle       : disentagle bands ? (bool), [False].
                - proj_min          : minimum value of projectability: [0.0].
                - dis_win_emax      : upper energy window (float), [None].
                - dis_win_emin      : lower energy window (float), [None].
                - smearing_temp_max : smearing temperature for upper window (float), [5.0].
                - smearing_temp_min : smearing temperature for lower window (float), [0.01].
                - delta             : small constant to avoid ill-conditioning of overlap matrices (float), [1e-12].
                - svd               : implement singular value decomposition ? otherwise adopt Lowdin's orthogonalization method (bool), [False].
                - verbose           : verbose calculation info (bool, optional), [False].
                - parallel          : use parallel code? (bool), [False].
                - formatter         : format by using black? (bool), [False].
                - write_hr          : write seedname_hr.py ? (bool), [False].
                - write_sr          : write seedname_sr.py ? (bool), [False].

            # only used for symmetrization.
                - symmetrization    : symmetrize ? (bool), [False].
                - mp_outdir         : output files for multipie are found in this directory (str). ["./"].
                - mp_seedname       : seedname for seedname_model.py, seedname_samb.py and seedname_matrix.py files (str), ["default"].
                - ket_amn           : ket basis list in the seedname.amn file. The format of each ket must be same as the "ket" in sambname_model.py file. See sambname["info"]["ket"] in sambname_model.py file for the format (list), [None].
                - irreps            : list of irreps to be considered (str/list), ["all"].

            # only used for band dispersion calculation.
                - a                 : lattice parameter (in Ang) used to correct units of k points in reference band data, [None].
                - fermi_energy      : fermi energy used for band shift, [None].
                - N1                : number of divisions for high symmetry lines (int, optional), [50].
        """
        if os.path.exists(file_name):
            with open(file_name) as fp:
                cwin_data = fp.readlines()
        else:
            raise Exception("failed to read cwin file: " + file_name)

        # default
        d = CWin._default_cwin().copy()

        for line in cwin_data:
            line = [vi for vi in line.replace("\n", "").split(" ") if vi != ""]
            if len(line) == 0:
                continue

            if "!" in line[0]:
                continue

            if "#" in line[0]:
                continue

            k = line[0]
            if k == "ket_amn":
                v = "".join(line[2:]) if line[1] == "=" else "".join(line[1:])
            else:
                v = line[2] if line[1] == "=" else line[1]

            d[k] = self._str_to(k, v)

        return d

    # ==================================================
    def _str_to(self, k, v):
        v = str(v).replace("'", "").replace('"', "")

        if k in ("seedname", "mp_seedname"):
            pass
        elif k in ("outdir", "mp_outdir"):
            v = v[:-1] if v[-1] == "/" else v
        elif k == "restart":
            if v not in ("wannierise", "symmetrization"):
                raise Exception(f"invalid restart = {v} was given. choose from 'wannierise'/'symmetrization'.")
        elif k in (
            "proj_min",
            "dis_win_emax",
            "dis_win_emin",
            "smearing_temp_max",
            "smearing_temp_min",
            "delta",
            "a",
            "fermi_energy",
        ):
            v = float(v)
        elif k == "N1":
            v = int(v)
        elif k == "ket_amn":
            if "(" in str(v) and ")" in str(v):
                v = [str(o) if i == 0 else f"({str(o)}" for i, o in enumerate(v[1:-1].split(",("))]
            else:
                v = [str(o) for o in v[1:-1].split(",")]
        elif k == "irreps":
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
                raise Exception(f"invalid {k} = {v} was given. choose from 'true'/'.true.'/'false'/'.false.'.")

        return v

    # ==================================================
    @classmethod
    def _default_cwin(cls):
        return _default_cwin
