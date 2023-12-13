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


_default = {"num_k": 1, "num_bands": 1, "num_wann": 1, "Ak": None}


# ==================================================
class Amn(dict):
    """
    Amn manages overlap matrix elements in seedname.amn file, A_{mn}(k) = <ψ^{KS}_{m}(k)|φ_{n}(k)>.

    Attributes:
        _topdir (str): top directory.
        _seedname (str): seedname.
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

        self._topdir = topdir
        self._seedname = seedname

        if dic is None:
            file_name = os.path.join(topdir, "{}.{}".format(seedname, "amn"))
            self.update(self.read(file_name))
        else:
            self.update(dic)

    # ==================================================
    def read(self, file_name="cwannier.amn"):
        """
        read seedname.amn file.

        Args:
            file_name (str, optional): file name.

        Returns:
            dict:
                - num_k     : # of k points (int), [1].
                - num_bands : # of bands passed to the code (int), [1].
                - num_wann  : # of WFs (int), [1].
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
    def write_u_dis_mat(self, file_name="cwannier.amn"):
        """
        write Udisk data.

        Args:
            file_name (str, optional): file name.
        """
        with open(file_name, "w") as fp:
            fp.write("# u_dis.mat created by umat.py\n")
            fp.write("{} {} {}\n\n".format(self["num_k"], self["num_wann"], self["num_bands"]))

            for k in range(self["num_k"]):
                kv = self["kpoints"][k]
                fp.write("{0:18.12f} {1:18.12f} {2:18.12f} \n".format(kv[0], kv[1], kv[2]))
                for i in range(self["num_wann"]):
                    for j in range(self["num_bands"]):
                        fp.write("{0.real:18.12f}  {0.imag:18.12f} \n".format(self["Udisk"][k][j][i]))

                fp.write("\n")

    # ==================================================
    @classmethod
    def _default(cls):
        return _default
