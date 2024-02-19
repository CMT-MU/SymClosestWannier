"""
Umat manages unitary matrix elements in seedname_u.mat (Uopt(k)) and seedname_u_dis.mat (Udis(k)) files, U(k) = Uopt(k)@Udis(k).
- Uopt(k): num_wann×num_wann unitary matrix.
- Udis(k): num_wann×num_bands partial unitary matrix.
- U(k): num_wann×num_bands partial unitary matrix.
"""

import os
import gzip
import tarfile
import datetime

import numpy as np

from symclosestwannier.cw.win import Win
from symclosestwannier.cw.eig import Eig


_default = {
    "num_k": 1,
    "num_bands": 1,
    "num_wann": 1,
    "kpoints": [[0, 0, 0]],
    "kpoints_wo_shift": [[0, 0, 0]],
    "Uoptk": None,
    "Udisk": None,
    "Uk": None,
}


# ==================================================
class Umat(dict):
    """
    Umat manages unitary matrix elements in seedname_u.mat (Uopt(k)) and seedname_u_dis.mat (Udis(k)) files, U(k) = Uopt(k)@Udis(k).

    Attributes:
        _topdir (str): top directory.
        _seedname (str): seedname.
    """

    # ==================================================
    def __init__(self, topdir=None, seedname="cwannier", dic=None):
        """
        Umat manages unitary matrix elements in seedname_u.mat (Uopt(k)) and seedname_u_dis.mat (Udis(k)) files, U(k) = Uopt(k)@Udis(k).

        Args:
            topdir (str, optional): directory of seedname.amn file.
            seedname (str, optional): seedname.
            dic (dict, optional): dictionary of Amn.
        """
        super().__init__()

        self._topdir = topdir
        self._seedname = seedname

        if dic is None:
            u_file_name = os.path.join(topdir, "{}.{}".format(f"{seedname}_u", "mat"))
            u_dis_file_name = os.path.join(topdir, "{}.{}".format(f"{seedname}_u_dis", "mat"))
            self.update(self.read(u_file_name, u_dis_file_name))
        else:
            self.update(dic)

    # ==================================================
    def read(self, u_file_name="cwannier_u.mat", u_dis_file_name=None):
        """
        read seedname.amn file.

        Args:
            u_file_name (str, optional): file name for seedname_u.mat.
            u_dis_file_name (str, optional): file name for seedname_u_dis.mat.

        Returns:
            dict:
                - num_k            : # of k points (int), [1].
                - num_bands        : # of bands passed to the code (int), [1].
                - num_wann         : # of WFs (int), [1].
                - k-points         : [[k1, k2, k3]] (crystal coordinate) (list), [[[0, 0, 0]]].,
                - kpoints_wo_shift : k-points without shift, [[k1, k2, k3]] (crystal coordinate) (list), [[[0, 0, 0]]].
                - Uoptk            : num_wann×num_wann full unitary matrix (ndarray), [None].
                - Udisk            : num_wann×num_bands partial unitary matrix (ndarray), [None].
                - Uk               : num_wann×num_bands full unitary matrix (ndarray), [None].
        """
        # Uoptk
        if os.path.exists(u_file_name):
            with open(u_file_name) as fp:
                u_mat_data = fp.readlines()
        elif os.path.exists(u_file_name + ".gz"):
            with gzip.open(u_file_name + ".gz", "rt") as fp:
                u_mat_data = fp.readlines()
        elif os.path.exists(u_file_name + ".tar.gz"):
            with tarfile.open(u_file_name + "tar.gz", "rt") as fp:
                u_mat_data = fp.readlines()
        else:
            raise Exception("failed to read u.mat file: " + u_file_name)

        d = Umat._default().copy()

        num_k, num_wann, _ = [int(x) for x in u_mat_data[1].split()]

        kpoints_wo_shift = np.zeros([num_k, 3], dtype=float)
        Uoptk = np.zeros([num_k, num_wann, num_wann], dtype=complex)
        u_mat_data = u_mat_data[3:]
        for k in range(num_k):
            kpoints_wo_shift[k] = np.array(
                [float(vi) for vi in u_mat_data[k * (num_wann * num_wann + 2)].split() if vi != ""]
            )
            u = u_mat_data[k * (num_wann * num_wann + 2) + 1 : k * (num_wann * num_wann + 2) + 1 + num_wann * num_wann]
            for j in range(num_wann):  # col
                for i in range(num_wann):  # row
                    v = [float(vi) for vi in u[j * num_wann + i].split() if vi != ""]
                    Uoptk[k, i, j] = v[0] + 1j * v[1]

        # Udisk
        if u_dis_file_name is None:
            Udisk = np.array([np.identity(num_wann, dtype=complex)] * num_k)
        else:
            if os.path.exists(u_dis_file_name):
                with open(u_dis_file_name) as fp:
                    u_dis_mat_data = fp.readlines()
            elif os.path.exists(u_dis_file_name + ".gz"):
                with gzip.open(u_dis_file_name + ".gz", "rt") as fp:
                    u_dis_mat_data = fp.readlines()
            elif os.path.exists(u_dis_file_name + ".tar.gz"):
                with tarfile.open(u_dis_file_name + "tar.gz", "rt") as fp:
                    u_dis_mat_data = fp.readlines()
            else:
                raise Exception("failed to read u_dis.mat file: " + u_dis_file_name)

            num_k_, num_wann_, num_bands = [int(x) for x in u_dis_mat_data[1].split()]

            assert num_k == num_k_, "#k points must be identical for *.inp and *_u_dis.mat"
            assert num_wann == num_wann_, "#WFs must be identical for *_u.mat and *_hr.dat"

            Udisk = np.zeros([num_k, num_bands, num_wann], dtype=complex)
            u_dis_mat_data = u_dis_mat_data[3:]
            for k in range(num_k):
                u = u_dis_mat_data[
                    k * (num_wann * num_bands + 2) + 1 : k * (num_wann * num_bands + 2) + 1 + num_wann * num_bands
                ]
                for j in range(num_wann):  # col
                    for i in range(num_bands):  # row
                        v = [float(vi) for vi in u[j * num_bands + i].split() if vi != ""]
                        Udisk[k, i, j] = v[0] + 1j * v[1]

        """
        Since seedname_u_dis.mat contains only the bands inside the outer window and the rest is given as zeros,
        we need to put the entry for udis_data in the correct position in Udisk. In other words, reshape band_data by shifting it by the number of bands below dis_window_min.
        """
        win = Win(self._topdir, self._seedname)
        if win["dis_num_iter"] > 0:
            eig = Eig(self._topdir, self._seedname)
            Ek = np.array(eig["Ek"])

            inside_win_true_or_false = np.logical_and(Ek >= win["dis_win_min"], Ek <= win["dis_win_max"])
            num_inside_win_k = np.sum(inside_win_true_or_false, axis=1)

            Udisk_ = np.zeros([num_k, num_bands, num_wann], dtype=complex)
            for ik in range(num_k):
                if not np.allclose(Udisk[ik, num_inside_win_k[ik] :, :], 0):
                    raise ValueError(
                        "This error may be due to rounding of the band window in seedname.wout. Do not use the default outer window, use a wider one and make sure the outer window is not near the band energy."
                    )
                Udisk_[ik, inside_win_true_or_false[ik]] = Udisk[ik, : num_inside_win_k[ik], :]

            Udisk = Udisk_

        Uk = Udisk @ Uoptk

        kpoints = np.mod(kpoints_wo_shift, 1)  # 0 <= kj < 1.0

        d = {
            "num_k": num_k,
            "num_bands": num_bands,
            "num_wann": num_wann,
            "kpoints": kpoints.tolist(),
            "kpoints_wo_shift": kpoints_wo_shift.tolist(),
            "Uoptk": Uoptk.tolist(),
            "Udisk": Udisk.tolist(),
            "Uk": Uk.tolist(),
        }

        return d

    # ==================================================
    def write_u_opt_mat(self, file_name="cwannier_u.mat"):
        """
        write Uoptk data.

        Args:
            file_name (str, optional): file name.
        """
        with open(file_name, "w") as fp:
            fp.write("# created by umat.py\n")
            fp.write("# written {}\n".format(datetime.datetime.now().strftime("on %d%b%Y at %H:%M:%S")))
            fp.write("{} {} {}\n\n".format(self["num_k"], self["num_wann"], self["num_wann"]))

            for k in range(self["num_k"]):
                kv = self["kpoints_wo_shift"][k]
                fp.write("{0:18.12f} {1:18.12f} {2:18.12f} \n".format(kv[0], kv[1], kv[2]))
                for i in range(self["num_wann"]):
                    for j in range(self["num_wann"]):
                        fp.write("{0.real:18.12f}  {0.imag:18.12f} \n".format(self["Uoptk"][k][j][i]))

                fp.write("\n")

        print(f"  * wrote '{file_name}'.")

    # ==================================================
    def write_u_dis_mat(self, file_name="cwannier_u_dis.mat"):
        """
        write Udisk data.

        Args:
            file_name (str, optional): file name.
        """
        with open(file_name, "w") as fp:
            fp.write("# created by umat.py\n")
            fp.write("# written {}\n".format(datetime.datetime.now().strftime("on %d%b%Y at %H:%M:%S")))
            fp.write("{} {} {}\n\n".format(self["num_k"], self["num_wann"], self["num_bands"]))

            for k in range(self["num_k"]):
                kv = self["kpoints_wo_shift"][k]
                fp.write("{0:18.12f} {1:18.12f} {2:18.12f} \n".format(kv[0], kv[1], kv[2]))
                for i in range(self["num_wann"]):
                    for j in range(self["num_bands"]):
                        fp.write("{0.real:18.12f}  {0.imag:18.12f} \n".format(self["Udisk"][k][j][i]))

                fp.write("\n")

        print(f"  * wrote '{file_name}'.")

    # ==================================================
    def write(self, file_names=("cwannier_u.mat", "cwannier_u_dis.mat")):
        """
        write Uoptk and Udisk data.

        Args:
            file_names (str, optional): file names.
        """
        self.write_u_opt_mat(file_name=file_names[0])
        self.write_u_dis_mat(file_name=file_names[1])

    # ==================================================
    @classmethod
    def _default(cls):
        return _default
