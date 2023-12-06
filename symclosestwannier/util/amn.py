"""
Amn manages overlap matrix elements in seedname.amn file, A_{mn}(k) = <ψ^{KS}_{m}(k)|φ_{n}(k)>.
- ψ^{KS}_{m}(k): Kohn-Sham orbitals (KSOs).
- φ_{n}(k): pseudo atomic (PAOs) orbitals.
"""
import os
import gzip
import itertools

import numpy as np
import scipy.linalg


# ==================================================
class Amn(dict):
    """
    Amn manages overlap matrix elements in seedname.amn file, A_{mn}(k) = <ψ^{KS}_{m}(k)|φ_{n}(k)>.
    """

    # ==================================================
    def __init__(self, topdir, seedname, encoding="UTF-8"):
        """
        initialize the class.

        Args:
            topdir (str): directory of seedname.amn file.
            seedname (str): seedname.
            encoding (str, optional): encoding.
        """
        file_amn = topdir + "/" + seedname + ".amn"

        self.update(self.read(file_amn))

    # ==================================================
    def read(self, file_amn):
        """
        read seedname.amn file.

        Args:
            file_amn (str): file name.

        Returns:
            dict:
                num_k (int): # of k points.
                num_bands (int): # of bands passed to the code.
                num_wann (int): # of CWFs.
                amn (ndarray): Overlap matrix elements, A_{mn}(k) = <ψ^{KS}_{m}(k)|φ_{n}(k)>.
        """
        if os.path.exists(file_amn):
            with open(file_amn) as fp:
                amn_data = fp.readlines()
        elif os.path.exists(file_amn + ".gz"):
            with gzip.open(file_amn + ".gz", "rt") as fp:
                amn_data = fp.readlines()
        else:
            raise Exception("failed to read amn file: " + file_amn)

        num_bands, num_k, num_wann = [int(x) for x in amn_data[1].split()]
        amn_data = np.genfromtxt(amn_data[2:]).reshape(num_k, num_wann, num_bands, 5)
        Ak = np.transpose(amn_data[:, :, :, 3] + 1j * amn_data[:, :, :, 4], axes=(0, 2, 1))

        d = {"num_k": num_k, "num_bands": num_bands, "num_wann": num_wann, "Ak": Ak}

        return d

    # ==================================================
    def write(self, file_amn):
        """
        write amn data.

        Args:
            file_amn (str): file name.
        """
        with open(file_amn, "w") as fp:
            fp.write("Amn created by amn.py\n")
            fp.write("{} {} {}\n".format(self["num_k"], self["num_bands"], self["num_wann"]))
            for ik, m, n in itertools.product(range(self["num_k"]), range(self["num_bands"]), range(self["num_wann"])):
                fp.write(
                    "{0} {1} {2}  {3.real:18.12f}  {3.imag:18.12f}\n".format(n + 1, m + 1, ik + 1, self.Ak[ik, m, n])
                )
