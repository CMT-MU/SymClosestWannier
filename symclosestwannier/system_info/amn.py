"""
Amn manages overlap matrix elements in seedname.amn file, A_{mn}(k) = <ψ^{KS}_{m}(k)|φ_{n}(k)>.
- ψ^{KS}_{m}(k): Kohn-Sham orbitals (KSOs).
- φ_{n}(k): pseudo atomic (PAOs) orbitals.
"""
import os
import gzip
import tarfile
import itertools

import numpy as np
import scipy.linalg

_default_amn = {"num_k": 1, "num_bands": 1, "num_wann": 1, "Ak": None}


# ==================================================
class Amn(dict):
    """
    Amn manages overlap matrix elements in seedname.amn file, A_{mn}(k) = <ψ^{KS}_{m}(k)|φ_{n}(k)>.
    """

    # ==================================================
    def __init__(self, topdir=None, seedname=None, dic=None):
        """
        initialize the class.

        Args:
            topdir (str, optional): directory of seedname.amn file.
            seedname (str, optional): seedname.
            dic (dict, optional): dictionary of Amn.
        """
        super().__init__()

        if dic is None:
            file_name = os.path.join(topdir, "{}.{}".format(seedname, "amn"))
            self.update(self.read(file_name))
        else:
            self.update(dic)

    # ==================================================
    def read(self, file_name):
        """
        read seedname.amn file.

        Args:
            file_name (str): file name.

        Returns:
            dict:
                - num_k     : # of k points (int), [1].
                - num_bands : # of bands passed to the code (int), [1].
                - num_wann  : # of CWFs (int), [1].
                - Ak        : Overlap matrix elements, A_{mn}(k) = <ψ^{KS}_{m}(k)|φ_{n}(k)> (list), [None].
        """
        if os.path.exists(file_name):
            with open(file_name) as fp:
                amn_data = fp.readlines()
        elif os.path.exists(file_name + ".gz"):
            with gzip.open(file_name + ".gz", "rt") as fp:
                amn_data = fp.readlines()
        elif os.path.exists(file_name + ".tar.gz"):
            with tarfile.open(file_name + "tar.gz", "rt") as fp:
                amn_data = fp.readlines()
        else:
            raise Exception("failed to read amn file: " + file_name)

        num_bands, num_k, num_wann = [int(x) for x in amn_data[1].split()]
        amn_data = np.genfromtxt(amn_data[2:]).reshape(num_k, num_wann, num_bands, 5)
        Ak = np.transpose(amn_data[:, :, :, 3] + 1j * amn_data[:, :, :, 4], axes=(0, 2, 1))
        Ak = Ak.tolist()

        d = {"num_k": num_k, "num_bands": num_bands, "num_wann": num_wann, "Ak": Ak}

        return d

    # ==================================================
    def write(self, file_name="cwannier.amn"):
        """
        write amn data.

        Args:
            file_name (str, optional): file name.
        """
        Ak = np.array(self["Ak"])

        with open(file_name, "w") as fp:
            fp.write("# amn created by amn.py\n")
            fp.write("{} {} {}\n".format(self["num_k"], self["num_bands"], self["num_wann"]))
            for ik, m, n in itertools.product(range(self["num_k"]), range(self["num_bands"]), range(self["num_wann"])):
                fp.write("{0} {1} {2}  {3.real:18.12f}  {3.imag:18.12f}\n".format(n + 1, m + 1, ik + 1, Ak[ik, m, n]))

    # ==================================================
    @classmethod
    def _default_amn(cls):
        return _default_amn