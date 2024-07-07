"""
uHu manages the matrix elements in seedname.uHu file, H_{mn}(k,b1,b2) = <u^{KS}_{m}(k+b1)|H(k)|u^{KS}_{n}(k+b2)>.
- ψ^{KS}_{m}(k) = u^{KS}_{m}(k) e^{ik.r}.
"""

import os
import gzip
import tarfile
import itertools
import datetime
from itertools import islice

import numpy as np

from symclosestwannier.util.utility import FortranFileR


_default = {"num_k": 1, "num_bands": 1, "num_b": 1, "Hkb1b2": None}


# ==================================================
class UHu(dict):
    """
    UHu manages the matrix elements in seedname.uHu file, H_{mn}(k,b1,b2) = <u^{KS}_{m}(k+b1)|H(k)|u^{KS}_{n}(k+b2)>.

    pw2wannier90 writes data_pw2w90[n, m, ib1, ib2, ik] = <u^{KS}_{m}(k+b1)|H(k)|u^{KS}_{n}(k+b2)>
    in column-major order.
    Here, we read to have data[ik, ib1, ib2, m, n] = <u^{KS}_{m}(k+b1)|H(k)|u^{KS}_{n}(k+b2)>

    Attributes:
        _topdir (str): top directory.
        _seedname (str): seedname.
        _formatted (bool): formatted file?
    """

    # def __init__(self, seedname="wannier90", formatted=False, suffix="uHu"):
    def __init__(self, topdir=None, seedname="cwannier", formatted=False, dic=None):
        super().__init__()

        self._topdir = topdir
        self._seedname = seedname
        self._formatted = formatted

        if dic is None:
            file_name = os.path.join(topdir, "{}.{}".format(seedname, "uHu"))
            self.update(self.read(file_name))
        else:
            self.update(dic)

    # ==================================================
    def read(self, file_name="cwannier.uHu"):
        """
        read seedname.uHu file.

        Args:
            file_name (str, optional): file name.

        Returns:
            dict:
                - num_k     : # of k points (int), [1].
                - num_bands : # of bands passed to the code (int), [1].
                - num_b     : # of b-vectors (int), [1].
                - nnkpts    : nearest-neighbor k-points (list), [None].
                - Hkb1b2    : Overlap matrix elements, H_{mn}(k,b1,b2) = <u^{KS}_{m}(k+b1)|H(k)|u^{KS}_{n}(k+b2)>.
        """
        if os.path.exists(file_name):
            pass
        elif os.path.exists(file_name + ".gz"):
            pass
        elif os.path.exists(file_name + ".tar.gz"):
            pass
        else:
            raise Exception("failed to read uHu file: " + file_name)

        if self._formatted:
            f_uHu_in = open(file_name, "r")
            header = f_uHu_in.readline().strip()
            num_bands, num_k, num_b = (int(x) for x in f_uHu_in.readline().split())
        else:
            f_uHu_in = FortranFileR(file_name)
            header = "".join(c.decode("ascii") for c in f_uHu_in.read_record("c")).strip()
            num_bands, num_k, num_b = f_uHu_in.read_record("i4")

        Hkb1b2 = np.zeros((num_k, num_b, num_b, num_bands, num_bands), dtype=complex)

        if self._formatted:
            tmp = np.array(
                [f_uHu_in.readline().split() for i in range(num_k * num_b * num_b * num_bands * num_bands)], dtype=float
            )
            tmp_cplx = tmp[:, 0] + 1.0j * tmp[:, 1]
            Hkb1b2 = tmp_cplx.reshape(num_k, num_b, num_b, num_bands, num_bands).transpose(0, 2, 1, 3, 4)
        else:
            for ik in range(num_k):
                for ib2 in range(num_b):
                    for ib1 in range(num_b):
                        tmp = (
                            f_uHu_in.read_record("f8").reshape((2, num_bands, num_bands), order="F").transpose(2, 1, 0)
                        )
                        Hkb1b2[ik, ib1, ib2] = tmp[:, :, 0] + 1j * tmp[:, :, 1]

        f_uHu_in.close()

        d = {"num_k": num_k, "num_bands": num_bands, "num_b": num_b, "Hkb1b2": Hkb1b2.tolist()}

        return d

    # ==================================================
    @classmethod
    def _default(cls):
        return _default
