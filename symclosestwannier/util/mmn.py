"""
Mmn manages overlap matrix elements in seedname.mmn file, M_{mn}(k,b) = <u^{KS}_{m}(k)|u^{KS}_{n}(k+b)>.
- ψ^{KS}_{m}(k) = u^{KS}_{m}(k) e^{ik.r}.
"""
import os
import gzip
import tarfile
import itertools

import numpy as np
import scipy.linalg


# ==================================================
class Mmn(dict):
    """
    Mmn manages overlap matrix elements in seedname.mmn file, M_{mn}(k,b) = <u^{KS}_{m}(k)|u^{KS}_{n}(k+b)>.
    """

    # ==================================================
    def __init__(self, topdir, seedname, encoding="UTF-8"):
        """
        initialize the class.

        Args:
            topdir (str): directory of seedname.mmn file.
            seedname (str): seedname.
            encoding (str, optional): encoding.
        """
        file_mmn = os.path.join(topdir, "{}.{}".format(seedname, "mmn"))

        self.update(self.read(file_mmn))

    # ==================================================
    def read(self, file_mmn):
        """
        read seedname.mmn file.

        Args:
            file_mmn (str): file name.

        Returns:
            dict:
        """
        if os.path.exists(file_mmn):
            fp = open(file_mmn, "r")
        elif os.path.exists(file_mmn + ".gz"):
            fp = gzip.open(file_mmn + ".gz", "rt")
        elif os.path.exists(file_mmn + ".tar.gz"):
            fp = tarfile.open(file_mmn + "tar.gz", "rt")
        else:
            raise Exception("failed to read mmn file: " + file_mmn)

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

        d = {"num_k": num_k, "num_bands": num_bands, "num_b": num_b, "nnkpts": nnkpts, "Mkb": Mkb}

        return d

    # ==================================================
    def write(self, file_mmn):
        """
        write mmn data.

        Args:
            file_mmn (str): file name.
        """
        with open(file_mmn, "w") as fp:
            fp.write("Mmn created by mmn.py\n")
            fp.write("{} {} {}\n".format(d["num_bands"], d["num_k"], d["num_b"]))
            for ik, ib in itertools.product(range(d["num_k"]), range(d["num_b"])):
                mkb = self["Mkb"][ik, ib, :, :]
                ik, ib, g0, g1, g2 = nnkpts[ik, ib, :]
                fp.write("{0}  {1}  {2}  {3}  {4}\n".format(ik, ib, g0, g1, g2))
                for m, n in itertools.product(range(d["num_bands"]), repeat=2):
                    fp.write("{0.real:18.12f}  {0.imag:18.12f}\n".format(mkb[m, n]))
