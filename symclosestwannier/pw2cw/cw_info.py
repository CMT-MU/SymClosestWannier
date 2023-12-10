"""
CWInfo manages CWin, Win, Eig, Amn, Mmn, and Nnkp.
"""
import numpy as np

from symclosestwannier.system_info.cwin import CWin
from symclosestwannier.system_info.win import Win
from symclosestwannier.system_info.eig import Eig
from symclosestwannier.system_info.amn import Amn
from symclosestwannier.system_info.mmn import Mmn
from symclosestwannier.system_info.nnkp import Nnkp

_class_map = {"cwin": CWin, "win": Win, "eig": Eig, "amn": Amn, "mmn": Mmn, "nnkp": Nnkp}


# ==================================================
class CWInfo(dict):
    """
    CWInfo manages CWin, Win, Eig, Amn, Mmn, and Nnkp.
    """

    # ==================================================
    def __init__(self, topdir=None, seedname=None, read_mmn=False, dic=None):
        """
        initialize the class.

        Args:
            topdir (str, optional): directory of seedname.cwin file.
            seedname (str, optional): seedname.
            read_mmn (bool, optional): read seedname.mmn file?
            dic (dict, optional): dictionary of CWin.
        """
        super().__init__()

        if dic is None:
            self.update(self.read(topdir, seedname, read_mmn))
        else:
            self.update(dic)

    # ==================================================
    def read(self, topdir, seedname, read_mmn=False):
        """
        read seedname.cwin/win/eig/amn/mmn/nnkp files.

        Args:
            topdir (str): directory of seedname.cwin/win/eig/amn/mmn/nnkp files.
            seedname (str): seedname.
            read_mmn (bool, optional): read seedname.mmn file?

        Returns:
            dict: system information.
        """
        d = {}
        info_list = []
        for name, C in _class_map.items():
            if name == "mmn" and not read_mmn:
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

        A = np.array(d["A"])
        d["unit_cell_volume"] = np.dot(A[0], np.cross(A[1], A[2]))

        return d

    # ==================================================
    @property
    def cwin(self):
        return CWin(dic={k: self[k] for k in CWin._default_cwin().keys()})

    # ==================================================
    @property
    def win(self):
        return Win(dic={k: self[k] for k in Win._default_win().keys()})

    # ==================================================
    @property
    def eig(self):
        return Eig(dic={k: self[k] for k in Eig._default_eig().keys()})

    # ==================================================
    @property
    def amn(self):
        return Amn(dic={k: self[k] for k in Amn._default_amn().keys()})

    # ==================================================
    @property
    def mmn(self):
        return Mmn(dic={k: self[k] for k in Mmn._default_mmn().keys()})

    # ==================================================
    @property
    def nnkp(self):
        return Nnkp(dic={k: self[k] for k in Nnkp._default_nnkp().keys()})
