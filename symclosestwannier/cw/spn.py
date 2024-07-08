"""
Spn manages matrix elements of Pauli spin operator in seedname.spn file.
"""

import os

import numpy as np

from symclosestwannier.util.utility import FortranFileR

_default = {"num_k": 1, "num_bands": 1, "pauli_spn": None}


# ==================================================
class Spn(dict):
    """
    Spn manages matrix elements of Pauli spin operator in seedname.spn file.

    Attributes:
        _topdir (str): top directory.
        _seedname (str): seedname.
        _formatted (bool): formatted file?
    """

    # ==================================================
    def __init__(self, topdir=None, seedname="cwannier", formatted=False, dic=None):
        """
        Spn manages matrix elements of Pauli spin operator in seedname.spn file.

        Args:
            topdir (str, optional): directory of seedname.spn file.
            seedname (str, optional): seedname.
            formatted (bool, optional): formatted file?
            dic (dict, optional): dictionary of Spn.
        """
        super().__init__()

        self._topdir = topdir
        self._seedname = seedname
        self._formatted = formatted

        if dic is None:
            file_name = os.path.join(topdir, "{}.{}".format(seedname, "spn"))
            self.update(self.read(file_name))
        else:
            self.update(dic)

    # ==================================================
    def read(self, file_name="cwannier.spn"):
        """
        read seedname.spn file.

        Args:
            file_name (str, optional): file name.

        Returns:
            dict:
                - num_k     : # of k points (int), [1].
                - num_bands : # of bands passed to the code (int), [1].
                - pauli_spn : num_bands×num_bands matrix elements of Pauli spin operators (ndarray), [None].
        """
        if os.path.exists(file_name):
            pass
        elif os.path.exists(file_name + ".gz"):
            pass
        elif os.path.exists(file_name + ".tar.gz"):
            pass
        else:
            raise Exception("failed to read spn file: " + file_name)

        if self._formatted:
            f_spn_in = open(file_name, "r")
            SPNheader = f_spn_in.readline().strip()
            num_bands, num_k = (int(x) for x in f_spn_in.readline().split())
        else:
            f_spn_in = FortranFileR(file_name)
            SPNheader = f_spn_in.read_record(dtype="c")
            num_bands, num_k = f_spn_in.read_record(dtype=np.int32)
            SPNheader = "".join(a.decode("ascii") for a in SPNheader)

        indm, indn = np.tril_indices(num_bands)
        pauli_spn = np.zeros((3, num_k, num_bands, num_bands), dtype=complex)

        for ik in range(num_k):
            A = np.zeros((3, num_bands, num_bands), dtype=complex)
            if self._formatted:
                tmp = np.array(
                    [f_spn_in.readline().split() for i in range(3 * num_bands * (num_bands + 1) // 2)], dtype=float
                )
                tmp = tmp[:, 0] + 1.0j * tmp[:, 1]
            else:
                tmp = f_spn_in.read_record(dtype=np.complex128)
            A[:, indn, indm] = tmp.reshape(3, num_bands * (num_bands + 1) // 2, order="F")
            check = np.einsum("ijj->", np.abs(A.imag))
            A[:, indm, indn] = A[:, indn, indm].conj()
            if check > 1e-10:
                raise RuntimeError("REAL DIAG CHECK FAILED : {0}".format(check))

            pauli_spn[:, ik, :, :] = A

        d = {"num_k": num_k, "num_bands": num_bands, "pauli_spn": pauli_spn.tolist()}

        return d

    # ==================================================
    @classmethod
    def _default(cls):
        return _default
