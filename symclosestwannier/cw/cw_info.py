# ****************************************************************** #
#                                                                    #
# This file is distributed as part of the symclosestwannier code and #
#     under the terms of the GNU General Public License. See the     #
#     file LICENSE in the root directory of the symclosestwannier    #
#      distribution, or http://www.gnu.org/licenses/gpl-3.0.txt      #
#                                                                    #
#          The symclosestwannier code is hosted on GitHub:           #
#                                                                    #
#            https://github.com/CMT-MU/SymClosestWannier             #
#                                                                    #
#                            written by                              #
#                        Rikuto Oiwa, RIKEN                          #
#                                                                    #
# ------------------------------------------------------------------ #
#                                                                    #
#                 cw_info: information for CWModel                   #
#                                                                    #
# ****************************************************************** #

import numpy as np
from gcoreutils.nsarray import NSArray

from symclosestwannier.cw.cwin import CWin
from symclosestwannier.cw.win import Win
from symclosestwannier.cw.nnkp import Nnkp
from symclosestwannier.cw.eig import Eig
from symclosestwannier.cw.amn import Amn
from symclosestwannier.cw.mmn import Mmn, _sort_Mkb_nnkpts
from symclosestwannier.cw.umat import Umat
from symclosestwannier.cw.spn import Spn
from symclosestwannier.cw.uHu import UHu

from symclosestwannier.util.utility import wigner_seitz, convert_w90_orbital

_class_map = {
    "cwin": CWin,
    "win": Win,
    "nnkp": Nnkp,
    "eig": Eig,
    "amn": Amn,
    "mmn": Mmn,
    "umat": Umat,
    "spn": Spn,
    "uHu": UHu,
}


# ==================================================
class CWInfo(dict):
    """
    CWInfo manages information for CWModel, CWin, Win, Nnkp, Eig, Amn, Mmn, Umat, Spn, UHu.

    Attributes:
        _topdir (str): top directory.
        _seedname (str): seedname.
        _postcw (bool): postcw calculation?
    """

    # ==================================================
    def __init__(self, topdir=None, seedname="cwannier", dic=None, postcw=False):
        """
        CWInfo manages information for CWModel, CWin, Win, Nnkp, Eig, Amn, Mmn, Umat, Spn, UHu.

        Args:
            topdir (str, optional): directory of seedname.cwin file.
            seedname (str, optional): seedname.
            dic (dict, optional): dictionary of CWin.
            postcw (bool): postcw calculation?
        """
        super().__init__()

        self._topdir = topdir
        self._seedname = seedname
        self._postcw = postcw

        self.update(self.read(topdir, seedname, dic))

    # ==================================================
    def read(self, topdir, seedname, dic=None):
        """
        read seedname.cwin/win/nnkp/eig/amn/(mmn)/(umat)/(spn)/(uHu) files.

        Args:
            topdir (str): directory of seedname.cwin/win/eig/amn/mmn/nnkp files.
            seedname (str): seedname.
            dic (dict, optional): dictionary of CWin.

        Returns:
            dict: system information.
        """
        d = {}

        if dic is None:
            for C in _class_map.values():
                d.update(C._default())
        else:
            d = dic

        info_dict = {}
        for name, C in _class_map.items():
            if name in d:
                continue

            if name == "umat" and (d["restart"] != "w90"):
                continue
            if name == "mmn":
                if not np.any(
                    [d["write_mmn"], d["write_rmn"], d["write_vmn"], d["write_tb"], d["berry"], d["gyrotropic"]]
                ):
                    continue
            if name == "spn":
                if not np.any(
                    [
                        d["write_spn"],
                        d["spin_decomp"],
                        d["spin_moment"],
                        d["zeeman_interaction"],
                        d["berry_task"] == "shc",
                        info_dict["win"].eval_spn,
                    ]
                ):
                    continue

            if name == "uHu":
                if not (self._postcw and d["gyrotropic"]):
                    continue
                if not np.any(
                    [
                        info_dict["win"].eval_K,
                        info_dict["win"].eval_C,
                        info_dict["win"].eval_D,
                        info_dict["win"].eval_Dw,
                        info_dict["win"].eval_NOA,
                    ]
                ):
                    continue

                info = C(topdir, seedname, formatted=d["uHu_formatted"])
            else:
                info = C(topdir, seedname)

            for k, v in info.items():
                for name_, info_ in info_dict.items():
                    if k in info_:
                        v_ = info_[k]

                        if type(v) == list:
                            if not np.allclose(v, v_, 1e-6):
                                if k == "nnkpts":
                                    info = _sort_Mkb_nnkpts(info, info_dict["nnkp"])
                                    v = info[k]

                                    if np.allclose(v, v_, 1e-6):
                                        continue

                                msg = str(f"The values of {k} in {name} and {name_} files are inconsistent.")
                                raise Exception(msg)
                        else:
                            if v != v_:
                                msg = str(f"The values of {k} in {name} and {name_} files are inconsistent.")
                                raise Exception(msg)

            info_dict.update({name: info})
            d.update(info)

        if d["zeeman_interaction"]:
            if not d["spinors"]:
                raise Exception("WFs are not spinors.")

        #
        # additional information
        #
        d["A"] = np.array(d["unit_cell_cart"])
        d["B"] = 2 * np.pi * np.linalg.inv(d["A"]).T
        A, B = d["A"], d["B"]

        irvec, ndegen = wigner_seitz(A, d["mp_grid"])

        if d["kpoint"] is not None and d["kpoint_path"] is not None:
            kpoint = {i: NSArray(j, "vector", fmt="value") for i, j in d["kpoint"].items()}
            kpoint_path = d["kpoint_path"]
            N1 = d["N1"]
            B = NSArray(d["unit_cell_cart"], "matrix", fmt="value").inverse()
            kpoints_path, k_linear, k_dis_pos = NSArray.grid_path(kpoint, kpoint_path, N1, B)
            kpoints_path = kpoints_path.tolist()
            k_linear = k_linear.tolist()
        else:
            kpoints_path, k_linear, k_dis_pos = None, None, None

        d["unit_cell_volume"] = np.dot(A[0], np.cross(A[1], A[2]))
        d["irvec"] = irvec.tolist()
        d["ndegen"] = ndegen.tolist()
        d["kpoints_path"] = kpoints_path
        d["k_linear"] = k_linear
        d["k_dis_pos"] = k_dis_pos

        # ket
        if d["ket_amn"] == "auto":
            ket_amn = [] * d["num_wann"]
            for pos_idx, l, m, r, s in zip(d["nw2n"], d["nw2l"], d["nw2m"], d["nw2r"], d["nw2s"]):
                pos = d["atom_pos_r"][pos_idx]

                name = ""
                for name_, pos_ in d["atoms_frac"].items():
                    if np.allclose(np.array(pos), np.array(pos_), rtol=1e-04, atol=1e-04):
                        name = f"{name_[0]}_{name_[1]}"

                orb = convert_w90_orbital(l, m, r, s)
                ket_amn.append(f"{orb}@{name}")

            d["ket_amn"] = ket_amn

        return d

    # ==================================================
    def write(self, topdir, seedname):
        """
        read seedname.cwin/win/eig/amn/mmn/nnkp files.

        Args:
            topdir (str): directory of seedname.cwin/win/eig/amn/mmn/nnkp files.
            seedname (str): seedname.

        Returns:
            dict: system information.
        """
        pass

    # ==================================================
    @property
    def cwin(self):
        return CWin(dic={k: self[k] if k in self else v for k, v in CWin._default().items()})

    # ==================================================
    @property
    def win(self):
        return Win(dic={k: self[k] if k in self else v for k, v in Win._default().items()})

    # ==================================================
    @property
    def nnkp(self):
        return Nnkp(dic={k: self[k] if k in self else v for k, v in Nnkp._default().items()})

    # ==================================================
    @property
    def eig(self):
        return Eig(dic={k: self[k] if k in self else v for k, v in Eig._default().items()})

    # ==================================================
    @property
    def amn(self):
        return Amn(dic={k: self[k] if k in self else v for k, v in Amn._default().items()})

    # ==================================================
    @property
    def mmn(self):
        return Mmn(dic={k: self[k] if k in self else v for k, v in Mmn._default().items()})

    # ==================================================
    @property
    def umat(self):
        return Umat(dic={k: self[k] if k in self else v for k, v in Umat._default().items()})

    # ==================================================
    @property
    def spn(self):
        return Spn(dic={k: self[k] if k in self else v for k, v in Spn._default().items()})
