"""
Berry manages berry phase/curvature and position/velocity operator.
"""
import os
import gzip
import tarfile
import itertools
import textwrap
import datetime
from docopt import docopt

import numpy as np
import scipy.linalg

from .utility import wigner_seitz
from symclosestwannier.util.amn import Amn
from symclosestwannier.util.eig import Eig
from symclosestwannier.util.win import Win


# ==================================================
class Berry(dict):
    """
    Berry manages berry phase/curvature and position operator.
    """

    # ==================================================
    def __init__(self, nnkp, mmn, win, Uk, topdir, seedname, encoding="UTF-8"):
        """
        initialize the class.

        Args:
            nnkp (dict): Nnkp.
            mmn (dict): Mmn.
            win (dict): Win.
            Uk (ndarray): unitary matrix.
            topdir (str): directory of seedname.amn file.
            seedname (str): seedname.
            encoding (str, optional): encoding.
        """
        num_k = nnkp["num_k"]
        num_wann = nnkp["num_wann"]
        kpoints = nnkp["kpoints"]
        A = nnkp["A"]
        bvec_cart = nnkp["bvec_cart"]
        wb = nnkp["wb"]

        Mkb = mmn["Mkb"]
        num_bands = mmn["num_bands"]

        mp_grid = win["mp_grid"]

        irvec, ndegen = wigner_seitz(A, mp_grid)

        kr = np.einsum("ka,ra->kr", kpoints, irvec, optimize=True)
        fac = np.exp(-2 * np.pi * 1j * kr)

        mnn = np.einsum("kbnn->kbn", Mkb, optimize=True)
        imlnmnn = np.log(mnn).imag
        rk_diag = -np.einsum("b,ba,kbn->akn", wb, bvec_cart, imlnmnn, optimize=True)
        rk_offdiag = 1j * np.einsum("b,ba,kbmn->akmn", wb, bvec_cart, Mkb, optimize=True)
        rk = np.zeros((3, num_k, num_bands, num_bands), dtype=complex)
        for ik in range(num_k):
            for i, j in itertools.product(range(num_bands), repeat=2):
                if i == j:
                    rk[:, ik, i, i] = rk_diag[:, ik, i]
                else:
                    rk[:, ik, i, j] = rk_offdiag[:, ik, i, j]

        Uk = np.array(Uk)

        #
        # position operator
        #
        # <mk|r|nk>
        rk = np.array(rk)

        # <wik|r|wjk>
        rk_w = np.array(
            [
                Uk[ik].transpose().conjugate()[np.newaxis, :, :] @ rk[:, ik, :, :] @ Uk[ik][np.newaxis, :, :]
                for ik in range(num_k)
            ]
        )
        rk_w = rk_w.transpose(1, 0, 2, 3)

        # <wi0|r|wjR>
        rR_w = np.einsum("akmn,kr->armn", rk_w, fac, optimize=True) / num_k

        #
        # berry phase
        #
        # i<mk|∇nk>
        ak = 1j * np.einsum("b,ba,kbmn->akmn", wb, bvec_cart, (Mkb - np.eye(num_bands)), optimize=True)

        # i<ik|∇wjk>
        ak_w = np.array(
            [
                Uk[ik].transpose().conjugate()[np.newaxis, :, :] @ ak[:, ik, :, :] @ Uk[ik][np.newaxis, :, :]
                for ik in range(num_k)
            ]
        )
        ak_w = ak_w.transpose(1, 0, 2, 3)

        # <wi0|a|wjR>
        aR_w = np.einsum("akmn,kr->armn", ak_w, fac, optimize=True) / num_k

        file_r = os.path.join(topdir, "{}.{}".format(f"{seedname}_r", "dat"))
        t = datetime.datetime.now()
        with open(file_r, "w") as fp:
            fp.write(" written {}\n".format(t.strftime("on %d%b%Y at %H:%M:%S")))
            fp.write(" {0[0]:15.8f} {0[1]:15.8f} {0[2]:15.8f}\n".format(nnkp["A"][0, :]))
            fp.write(" {0[0]:15.8f} {0[1]:15.8f} {0[2]:15.8f}\n".format(nnkp["A"][1, :]))
            fp.write(" {0[0]:15.8f} {0[1]:15.8f} {0[2]:15.8f}\n".format(nnkp["A"][2, :]))
            fp.write("{:12d}\n{:12d}\n".format(num_wann, len(ndegen)))
            fp.write(textwrap.fill("".join(["{:5d}".format(x) for x in ndegen]), 75, drop_whitespace=False))
            fp.write("\n")

            #
            # write <wi0|r|wjR>
            #
            for irpts in range(len(ndegen)):
                for i, j in itertools.product(range(num_wann), repeat=2):
                    line = "{:5d}{:5d}{:5d}{:5d}{:5d}  ".format(*irvec[irpts, :], j + 1, i + 1)
                    line += "".join([" {:15.8e} {:15.8e}".format(x.real, x.imag) for x in rR_w[:, irpts, i, j]])
                    fp.write(line + "\n")

        file_a = os.path.join(topdir, "{}.{}".format(f"{seedname}_a", "dat"))
        t = datetime.datetime.now()
        with open(file_a, "w") as fp:
            fp.write(" written {}\n".format(t.strftime("on %d%b%Y at %H:%M:%S")))
            fp.write(" {0[0]:15.8f} {0[1]:15.8f} {0[2]:15.8f}\n".format(nnkp["A"][0, :]))
            fp.write(" {0[0]:15.8f} {0[1]:15.8f} {0[2]:15.8f}\n".format(nnkp["A"][1, :]))
            fp.write(" {0[0]:15.8f} {0[1]:15.8f} {0[2]:15.8f}\n".format(nnkp["A"][2, :]))
            fp.write("{:12d}\n{:12d}\n".format(num_wann, len(ndegen)))
            fp.write(textwrap.fill("".join(["{:5d}".format(x) for x in ndegen]), 75, drop_whitespace=False))
            fp.write("\n")

            #
            # write <wi0|a|wjR>
            #
            for irpts in range(len(ndegen)):
                for i, j in itertools.product(range(num_wann), repeat=2):
                    line = "{:5d}{:5d}{:5d}{:5d}{:5d}  ".format(*irvec[irpts, :], j + 1, i + 1)
                    line += "".join([" {:15.8e} {:15.8e}".format(x.real, x.imag) for x in aR_w[:, irpts, i, j]])
                    fp.write(line + "\n")
