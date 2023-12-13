"""
CWInfo manages CWin, Win, Eig, Amn, Mmn, and Nnkp.
"""
import numpy as np
from gcoreutils.nsarray import NSArray

from symclosestwannier.system_info.cwin import CWin
from symclosestwannier.system_info.win import Win
from symclosestwannier.system_info.nnkp import Nnkp
from symclosestwannier.system_info.eig import Eig
from symclosestwannier.system_info.amn import Amn
from symclosestwannier.system_info.mmn import Mmn
from symclosestwannier.system_info.umat import Umat

_class_map = {"cwin": CWin, "win": Win, "nnkp": Nnkp, "eig": Eig, "amn": Amn, "mmn": Mmn, "umat": Umat}


# ==================================================
class CWInfo(dict):
    """
    CWInfo manages CWin, Win, Eig, Amn, Mmn, and Nnkp.

    Attributes:
        _topdir (str): top directory.
        _seedname (str): seedname.
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

        self._topdir = topdir
        self._seedname = seedname

        if dic is None:
            self.update(self.read(topdir, seedname))
        else:
            self.update(dic)

    # ==================================================
    def read(self, topdir, seedname):
        """
        read seedname.cwin/win/eig/amn/mmn/nnkp files.

        Args:
            topdir (str): directory of seedname.cwin/win/eig/amn/mmn/nnkp files.
            seedname (str): seedname.

        Returns:
            dict: system information.
        """
        d = {}
        for C in _class_map.values():
            d.update(C._default())

        info_list = []
        for name, C in _class_map.items():
            if name == "umat" and (d["restart"] != "w90"):
                continue

            if name == "mmn":
                continue

            info = C(topdir, seedname)
            for k, v in info.items():
                for name_, info_ in info_list:
                    if k in info_:
                        v_ = info_[k]
                        try:
                            if v != v_:
                                raise Exception(f"The value of {k} in {name} and {name_} is inconsistent.")
                        except:
                            if np.all(np.array(v) == np.array(v_)):
                                raise Exception(f"The value of {k} in {name} and {name_} is inconsistent.")

            info_list.append((name, info))
            d.update(info)

        # additional information
        A = np.array(d["A"])
        d["unit_cell_volume"] = np.dot(A[0], np.cross(A[1], A[2]))

        if d["kpoint"] is not None and d["kpoint_path"] is not None:
            kpoint = {i: NSArray(j, "vector", fmt="value") for i, j in d["kpoint"].items()}
            kpoint_path = d["kpoint_path"]
            N1 = d["N1"]
            B = NSArray(d["unit_cell_cart"], "matrix", fmt="value").inverse()
            kpoints_path, k_linear, k_dis_pos = NSArray.grid_path(kpoint, kpoint_path, N1, B)
        else:
            kpoints_path, k_linear, k_dis_pos = None, None, None

        d["kpoints_path"] = kpoints_path
        d["k_linear"] = k_linear
        d["k_dis_pos"] = k_dis_pos

        return d

    # ==================================================
    @property
    def cwin(self):
        return CWin(dic={k: self[k] for k in CWin._default().keys()})

    # ==================================================
    @property
    def win(self):
        return Win(dic={k: self[k] for k in Win._default().keys()})

    # ==================================================
    @property
    def nnkp(self):
        return Nnkp(dic={k: self[k] for k in Nnkp._default().keys()})

    # ==================================================
    @property
    def eig(self):
        return Eig(dic={k: self[k] for k in Eig._default().keys()})

    # ==================================================
    @property
    def amn(self):
        return Amn(dic={k: self[k] for k in Amn._default().keys()})

    # ==================================================
    @property
    def mmn(self):
        return Mmn(dic={k: self[k] for k in Mmn._default().keys()})

    # ==================================================
    @property
    def umat(self):
        return Umat(dic={k: self[k] for k in Umat._default().keys()})
