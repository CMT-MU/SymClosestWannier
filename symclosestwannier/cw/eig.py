"""
Eig manages Kohn-Sham energies in seedname.eig file, E_{m}(k).
"""

import os
import itertools
import datetime

import numpy as np

from symclosestwannier.util.input_file import open_input


_default = {"num_k": 1, "num_bands": 1, "Ek": None}


# ==================================================
class Eig(dict):
    """
    Eig manages Kohn-Sham energies in seedname.eig file, E_{m}(k).

    Attributes:
        _topdir (str): top directory.
        _seedname (str): seedname.
    """

    # ==================================================
    def __init__(self, topdir=None, seedname="cwannier", dic=None):
        """
        Eig manages Kohn-Sham energies in seedname.eig file, E_{m}(k).

        Args:
            topdir (str, optional): directory of seedname.eig file.
            seedname (str, optional): seedname.
            dic (dict, optional): dictionary of Eig.
        """
        super().__init__()

        self._topdir = topdir
        self._seedname = seedname

        if dic is None:
            file_name = os.path.join(topdir, "{}.{}".format(seedname, "eig"))
            self.update(self.read(file_name))
        else:
            self.update(dic)

    # ==================================================
    def read(self, file_name="cwannier.eig"):
        """
        read seedname.eig file.

        Args:
            file_name (str, optional): file name.

        Returns:
            dict:
                - num_k     : # of k points (int), [1].
                - num_bands : # of bands passed to the code (int), [1].
                - Ek        : Kohn-Sham energies, E_{m}(k) (list), [None].
        """
        with open_input(file_name, "eig") as fp:
            eig_data = fp.readlines()

        eig_data = [[v for v in lst.rstrip("\n").split(" ") if v != ""] for lst in eig_data]
        eig_data = [[float(v) if "." in v else int(v) for v in lst] for lst in eig_data]

        num_bands = np.max([v[0] for v in eig_data])
        num_k = np.max([v[1] for v in eig_data])
        Ek = [[eig_data[k * num_bands + m][2] for m in range(num_bands)] for k in range(num_k)]

        d = {"num_k": num_k, "num_bands": num_bands, "Ek": Ek}

        return d

    # ==================================================
    def write(self, file_name="cwannier.eig.cw"):
        """
        write eig data.

        Args:
            file_name (str, optional): file name.
        """
        Ek = np.array(self["Ek"])

        with open(file_name, "w") as fp:
            for ik, n in itertools.product(range(self["num_k"]), range(self["num_bands"])):
                fp.write("{:5d}{:5d}{:>18.12f}\n".format(n + 1, ik + 1, Ek[ik, n]))

        print(f"  * wrote '{file_name}'.")

    # ==================================================
    @classmethod
    def _default(cls):
        return _default
