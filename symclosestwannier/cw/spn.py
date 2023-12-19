"""
Spn manages matrix elements of Spin operator in seedname.spn file.
"""
import os
import gzip
import tarfile

import numpy as np

from symclosestwannier.util._utility import FortranFileR, FortranFileW

_default = {"num_k": 1, "num_bands": 1, "Sk": None}


# ==================================================
class Spn(dict):
    """
    Spn manages matrix elements of Spin operator in seedname.spn file.

    Attributes:
        _topdir (str): top directory.
        _seedname (str): seedname.
        _spn_formatted (bool): formatted file?
    """

    # ==================================================
    def __init__(self, topdir=None, seedname="cwannier", spn_formatted=False, dic=None):
        """
        initialize the class.

        Args:
            topdir (str, optional): directory of seedname.spn file.
            seedname (str, optional): seedname.
            spn_formatted (bool, optional): formatted file?
            dic (dict, optional): dictionary of Spn.
        """
        super().__init__()

        self._topdir = topdir
        self._seedname = seedname
        self._spn_formatted = spn_formatted

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
                - num_wann  : # of WFs (int), [1].
                - k-points  : [[k1, k2, k3]] (crystal coordinate) (list), [[[0, 0, 0]]].,
                - Uoptk     : num_wann×num_wann full unitary matrix (ndarray), [None].
                - Udisk     : num_wann×num_bands partial unitary matrix (ndarray), [None].
                - Uk        : num_wann×num_bands full unitary matrix (ndarray), [None].
        """
        if os.path.exists(file_name):
            pass
        elif os.path.exists(file_name + ".gz"):
            pass
        elif os.path.exists(file_name + ".tar.gz"):
            pass
        else:
            raise Exception("failed to read spn file: " + file_name)

        if self._spn_formatted:
            f_spn_in = open(file_name, "r")
            SPNheader = f_spn_in.readline().strip()
            num_bands, num_k = (int(x) for x in f_spn_in.readline().split())
        else:
            f_spn_in = FortranFileR(file_name)
            SPNheader = f_spn_in.read_record(dtype="c")
            num_bands, num_k = f_spn_in.read_record(dtype=np.int32)
            SPNheader = "".join(a.decode("ascii") for a in SPNheader)

        indm, indn = np.tril_indices(num_bands)
        Sk = np.zeros((3, num_k, num_bands, num_bands), dtype=complex)

        for ik in range(num_k):
            A = np.zeros((3, num_bands, num_bands), dtype=complex)
            if self._spn_formatted:
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

            Sk[:, ik, :, :] = A

        d = {"num_k": num_k, "num_bands": num_bands, "Sk": Sk.tolist()}

        return d

    # # ==================================================
    # def write(self, file_name="cwannier.spn"):
    #     """
    #     write spn data.

    #     Args:
    #         file_name (str, optional): file name.
    #     """
    #     SPN = FortranFileW(file_name)
    #     header = "Created from wavecar at {0}".format(datetime.datetime.now().isoformat())
    #     header = header[:60]
    #     header += " " * (60 - len(header))
    #     SPN.write_record(bytearray(header, encoding="ascii"))
    #     SPN.write_record(np.array([NBout, NK], dtype=np.int32))

    #     for ik in range(NK):
    #         npw = int(record(2 + ik * (NBin + 1), 1))
    #         npw12 = npw // 2
    #         if npw != npw12 * 2:
    #             raise RuntimeError(f"odd number of coefs {npw}")
    #         print("k-point {0:3d} : {1:6d} plane waves".format(ik, npw))
    #         WF = np.zeros((npw, NBout), dtype=complex)
    #         for ib in range(NBout):
    #             WF[:, ib] = record(3 + ik * (NBin + 1) + ib + IBstart, npw, np.complex64)
    #         overlap = WF.conj().T.dot(WF)
    #         assert np.max(np.abs(overlap - overlap.T.conj())) < 1e-15

    #         if normalize == "norm":
    #             WF = WF / np.sqrt(np.abs(overlap.diagonal()))

    #         SIGMA = np.array(
    #             [
    #                 [
    #                     np.einsum(
    #                         "ki,kj->ij", WF.conj()[npw12 * i : npw12 * (i + 1), :], WF[npw12 * j : npw12 * (j + 1), :]
    #                     )
    #                     for j in (0, 1)
    #                 ]
    #                 for i in (0, 1)
    #             ]
    #         )
    #         SX = SIGMA[0, 1] + SIGMA[1, 0]
    #         SY = -1.0j * (SIGMA[0, 1] - SIGMA[1, 0])
    #         SZ = SIGMA[0, 0] - SIGMA[1, 1]
    #         A = np.array(
    #             [s[n, m] for m in range(NBout) for n in range(m + 1) for s in (SX, SY, SZ)], dtype=np.complex128
    #         )
    #         SPN.write_record(A)

    # ==================================================
    @classmethod
    def _default(cls):
        return _default
