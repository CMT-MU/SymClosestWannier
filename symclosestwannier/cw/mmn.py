"""
Mmn manages overlap matrix elements in seedname.mmn file, M_{mn}(k,b) = <u^{KS}_{m}(k)|u^{KS}_{n}(k+b)>.
- ψ^{KS}_{m}(k) = u^{KS}_{m}(k) e^{ik.r}.
"""

import os
import gzip
import tarfile
import itertools
import datetime
import multiprocessing
from itertools import islice

import numpy as np


_default = {"num_k": 1, "num_bands": 1, "num_b": 1, "nnkpts": None, "Mkb": None}

_mult = 4


# ==================================================
def _convert(lst):
    return np.array([l.split() for l in lst], dtype=float)


# ==================================================
class Mmn(dict):
    """
    Mmn manages overlap matrix elements in seedname.mmn file, M_{mn}(k,b) = <u^{KS}_{m}(k)|u^{KS}_{n}(k+b)>.

    Attributes:
        _topdir (str): top directory.
        _seedname (str): seedname.
        _npar (int): # of cpu core.
    """

    # ==================================================
    def __init__(self, topdir=None, seedname="cwannier", dic=None, npar=multiprocessing.cpu_count()):
        """
        Mmn manages overlap matrix elements in seedname.mmn file, M_{mn}(k,b) = <u^{KS}_{m}(k)|u^{KS}_{n}(k+b)>.

        Args:
            topdir (str, optional): directory of seedname.mmn file.
            seedname (str, optional): seedname.
            dic (dict, optional): dictionary of Mmn.
            npar (int, optional): # of cpu core.
        """
        super().__init__()

        self._topdir = topdir
        self._seedname = seedname
        self._npar = npar

        if dic is None:
            file_name = os.path.join(topdir, "{}.{}".format(seedname, "mmn"))
            self.update(self.read(file_name))
        else:
            self.update(dic)

    # ==================================================
    def read(self, file_name="cwannier.mmn"):
        """
        read seedname.mmn file.

        Args:
            file_name (str, optional): file name.

        Returns:
            dict:
                - num_k     : # of k points (int), [1].
                - num_bands : # of bands passed to the code (int), [1].
                - num_b     : # of b-vectors (int), [1].
                - nnkpts    : nearest-neighbor k-points (list), [None].
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

        fp.readline()
        num_bands, num_k, num_b = np.array(fp.readline().split(), dtype=int)

        block = 1 + num_bands * num_bands

        Mkb_data = []
        nnkpts_data = []

        if self._npar > 0:
            pool = multiprocessing.Pool(self._npar)

        for j in range(0, num_b * num_k, self._npar * _mult):
            x = list(islice(fp, int(block * self._npar * _mult)))
            if len(x) == 0:
                break
            nnkpts_data += x[::block]
            y = [x[i * block + 1 : (i + 1) * block] for i in range(self._npar * _mult) if (i + 1) * block <= len(x)]
            if self._npar > 0:
                Mkb_data += pool.map(_convert, y)
            else:
                Mkb_data += [_convert(z) for z in y]

        if self._npar > 0:
            pool.close()
            pool.join()

        fp.close()

        Mkb_data = [d[:, 0] + 1j * d[:, 1] for d in Mkb_data]
        Mkb = np.array(Mkb_data).reshape(num_k, num_b, num_bands, num_bands).transpose((0, 1, 3, 2))
        nnkpts_data = np.array([s.split() for s in nnkpts_data], dtype=int).reshape(num_k, num_b, 5)

        assert np.all(nnkpts_data[:, :, 0] - 1 == np.arange(num_k)[:, None])

        nnkpts = nnkpts_data.tolist()

        d = {"num_k": num_k, "num_bands": num_bands, "num_b": num_b, "nnkpts": nnkpts, "Mkb": Mkb}

        return d

    # ==================================================
    def write(self, file_name="cwannier.mmn.cw"):
        """
        write mmn data.

        Args:
            file_name (str, optional): file name.
        """
        Mkb = np.array(self["Mkb"])

        with open(file_name, "w") as fp:
            fp.write("Created by mmn.py {}\n".format(datetime.datetime.now().strftime("on %d%b%Y at %H:%M:%S")))
            fp.write("       {:5d}       {:5d}       {:5d}\n".format(self["num_bands"], self["num_k"], self["num_b"]))
            for ik, ib in itertools.product(range(self["num_k"]), range(self["num_b"])):
                mkb = Mkb[ik, ib, :, :]
                ik, ib, g0, g1, g2 = self["nnkpts"][ik][ib][:]
                fp.write("    {0}    {1}    {2}    {3}    {4}\n".format(ik, ib, g0, g1, g2))
                for m, n in itertools.product(range(self["num_bands"]), repeat=2):
                    fp.write("{0.real:18.12f}{0.imag:18.12f}\n".format(mkb[n, m]))

        print(f"  * wrote '{file_name}'.")

    # ==================================================
    @classmethod
    def _default(cls):
        return _default
