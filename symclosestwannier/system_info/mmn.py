"""
Mmn manages overlap matrix elements in seedname.mmn file, M_{mn}(k,b) = <u^{KS}_{m}(k)|u^{KS}_{n}(k+b)>.
- ψ^{KS}_{m}(k) = u^{KS}_{m}(k) e^{ik.r}.
"""
import os
import gzip
import tarfile
import itertools

import numpy as np


_default_mmn = {"num_k": 1, "num_bands": 1, "num_b": 1, "nnkpts": None, "Mkb": None}


# ==================================================
class Mmn(dict):
    """
    Mmn manages overlap matrix elements in seedname.mmn file, M_{mn}(k,b) = <u^{KS}_{m}(k)|u^{KS}_{n}(k+b)>.
    """

    # ==================================================
    def __init__(self, topdir=None, seedname=None, dic=None):
        """
        initialize the class.

        Args:
            topdir (str, optional): directory of seedname.mmn file.
            seedname (str, optional): seedname.
            dic (dict, optional): dictionary of Mmn.
        """
        super().__init__()

        if dic is None:
            file_name = os.path.join(topdir, "{}.{}".format(seedname, "mmn"))
            self.update(self.read(file_name))
        else:
            self.update(dic)

    # ==================================================
    def read(self, file_name):
        """
        read seedname.mmn file.

        Args:
            file_name (str): file name.

        Returns:
            dict:
                - num_k     : # of k points (int), [1].
                - num_bands : # of bands passed to the code (int), [1].
                - num_b     : # of b-vectors (int), [1].
                - nnkpts    : nearest-nmmnhbour k-points (list), [None].
                - Mkb       : Overlap matrix elements, M_{mn}(k,b) = <u^{KS}_{m}(k)|u^{KS}_{n}(k+b)> (list), [None].
        """
        if os.path.exists(file_name):
            fp = open(file_name, "r")
        elif os.path.exists(file_name + ".gz"):
            fp = gzip.open(file_name + ".gz", "rt")
        elif os.path.exists(file_name + ".tar.gz"):
            fp = tarfile.open(file_name + "tar.gz", "rt")
        else:
            raise Exception("failed to read mmn file: " + file_name)

        _ = fp.readline()  # first line

        num_bands, num_k, num_b = [int(x) for x in fp.readline().split()]

        Mkb = np.zeros([num_k, num_b, num_bands, num_bands], dtype=complex)
        nnkpts = np.zeros([num_k, num_b, 5], dtype=int)

        for ik, ib in itertools.product(range(num_k), range(num_b)):
            d = [int(x) for x in fp.readline().split()]
            assert ik == d[0] - 1, "{} {}".format(ik, d[0])
            nnkpts[ik, ib, :] = d
            for m, n in itertools.product(range(num_bands), repeat=2):
                dat = [float(x) for x in fp.readline().split()]
                Mkb[ik, ib, m, n] = dat[0] + 1j * dat[1]

        Mkb = Mkb.tolist()

        d = {"num_k": num_k, "num_bands": num_bands, "num_b": num_b, "nnkpts": nnkpts, "Mkb": Mkb}

        return d

    # ==================================================
    def write(self, file_name="cwannier.mmn"):
        """
        write mmn data.

        Args:
            file_name (str, optional): file name.
        """
        Mkb = np.array(self["Mkb"])

        with open(file_name, "w") as fp:
            fp.write("# mmn created by mmn.py\n")
            fp.write("{} {} {}\n".format(self["num_bands"], self["num_k"], self["num_b"]))
            for ik, ib in itertools.product(range(self["num_k"]), range(self["num_b"])):
                mkb = Mkb[ik, ib, :, :]
                ik, ib, g0, g1, g2 = self["nnkpts"][ik, ib, :]
                fp.write("{0}  {1}  {2}  {3}  {4}\n".format(ik, ib, g0, g1, g2))
                for m, n in itertools.product(range(self["num_bands"]), repeat=2):
                    fp.write("{0.real:18.12f}  {0.imag:18.12f}\n".format(mkb[m, n]))

    # ==================================================
    @classmethod
    def _default_mmn(cls):
        return _default_mmn
