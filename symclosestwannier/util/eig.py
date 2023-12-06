"""
Eig manages Kohn-Sham energies in seedname.eig file, E_{m}(k).
"""
import os
import gzip
import tarfile
import itertools

import numpy as np
import scipy.linalg


# ==================================================
class Eig(dict):
    """
    Eig manages Kohn-Sham energies in seedname.eig file, E_{m}(k).
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
        file_eig = os.path.join(topdir, "{}.{}".format(seedname, "eig"))

        self.update(self.read(file_eig))

    # ==================================================
    def read(self, file_eig):
        """
        read seedname.eig file.

        Args:
            file_eig (str): file name.

        Returns:
            dict:
                num_k (int): # of k points.
                num_bands (int): # of bands passed to the code.
                Ek (ndarray): Kohn-Sham energies, E_{m}(k).
        """
        if os.path.exists(file_eig):
            with open(file_eig) as fp:
                eig_data = fp.readlines()
        elif os.path.exists(file_eig + ".gz"):
            with gzip.open(file_eig + ".gz", "rt") as fp:
                eig_data = fp.readlines()
        elif os.path.exists(file_eig + ".tar.gz"):
            with tarfile.open(file_eig + "tar.gz", "rt") as fp:
                amn_data = fp.readlines()
        else:
            raise Exception("failed to read eig file: " + file_eig)

        eig_data = [[v for v in lst.rstrip("\n").split(" ") if v != ""] for lst in eig_data]
        eig_data = [[float(v) if "." in v else int(v) for v in lst] for lst in eig_data]

        num_bands = np.max([v[0] for v in eig_data])
        num_k = np.max([v[1] for v in eig_data])
        Ek = np.array([[eig_data[k * num_bands + m][2] for m in range(num_bands)] for k in range(num_k)])

        d = {"num_k": num_k, "num_bands": num_bands, "Ek": Ek}

        return d

    # ==================================================
    def write_eig(self, file_eig):
        """
        write eig data.

        Args:
            file_eig (str): file name.
        """
        with open(file_eig, "w") as fp:
            for ik, n in itertools.product(range(self["num_k"]), range(self["num_bands"])):
                fp.write("{:5d}{:5d}{:18.12f}\n".format(n + 1, ik + 1, self["Ek"][ik, n]))
