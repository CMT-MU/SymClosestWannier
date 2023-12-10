"""
Umat manages unitary matrix elements in seedname_u.mat (Uopt(k)) and seedname_u_dis.mat (Udis(k)) files, U(k) = Uopt(k)@Udis(k).
- Uopt(k): num_wann×num_wann unitary matrix.
- Udis(k): num_wann×num_bands partial unitary matrix.
- U(k): num_wann×num_bands partial unitary matrix.
"""
import os
import gzip
import tarfile

import numpy as np

_default_umat = {
    "num_k": 1,
    "num_bands": 1,
    "num_wann": 1,
    "kpoints": [[0, 0, 0]],
    "Uoptk": None,
    "Udisk": None,
    "Uk": None,
}


# ==================================================
class Umat(dict):
    """
    Umat manages unitary matrix elements in seedname_u.mat (Uopt(k)) and seedname_u_dis.mat (Udis(k)) files, U(k) = Uopt(k)@Udis(k).
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
            u_file_name = os.path.join(topdir, "{}.{}".format(f"{seedname}_u", "mat"))
            u_dis_file_name = os.path.join(topdir, "{}.{}".format(f"{seedname}_u_dis", "mat"))
            self.update(self.read(u_file_name, u_dis_file_name))
        else:
            self.update(dic)

    # ==================================================
    def read(self, u_file_name, u_dis_file_name):
        """
        read seedname.amn file.

        Args:
            u_file_name (str): file name for seedname_u.mat.
            u_dis_file_name (str): file name for seedname_u_dis.mat.

        Returns:
            dict:
                - num_k     : # of k points (int), [1].
                - num_bands : # of bands passed to the code (int), [1].
                - num_wann  : # of CWFs (int), [1].
                - Uoptk     : num_wann×num_wann unitary matrix (ndarray), [None].
                - Udisk     : num_wann×num_bands partial unitary matrix (ndarray), [None].
                - Uk        : num_wann×num_bands partial unitary matrix (ndarray), [None].
        """
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

        d = Umat._default_umat().copy()

        num_k, num_wann, num_bands = [int(x) for x in u_dis_mat_data[1].split()]

        kpoints = np.zeros([num_k, 3], dtype=float)
        Uoptk = np.zeros([num_k, num_wann, num_wann], dtype=complex)
        u_mat_data = u_mat_data[3:]
        for k in range(num_k):
            kpoints[k] = np.array([float(vi) for vi in u_mat_data[k * (num_wann * num_wann + 2)].split() if vi != ""])
            u = u_mat_data[k * (num_wann * num_wann + 2) + 1 : k * (num_wann * num_wann + 2) + num_wann * num_wann + 1]
            for i in range(num_wann):
                for j in range(num_wann):
                    v = [float(vi) for vi in u[i * num_wann + j].split() if vi != ""]
                    Uoptk[k, i, j] = v[0] + 1j * v[1]

        Udisk = np.zeros([num_k, num_bands, num_wann], dtype=complex)
        u_dis_mat_data = u_dis_mat_data[3:]
        for k in range(num_k):
            u = u_dis_mat_data[
                k * (num_wann * num_bands + 2) + 1 : k * (num_wann * num_bands + 2) + num_wann * num_bands + 1
            ]
            for i in range(num_bands):
                for j in range(num_wann):
                    v = [float(vi) for vi in u[i * num_wann + j].split() if vi != ""]
                    Udisk[k, i, j] = v[0] + 1j * v[1]

        Uk = Udisk @ Uoptk

        d = {
            "num_k": num_k,
            "num_bands": num_bands,
            "num_wann": num_wann,
            "kpoints": kpoints.tolist(),
            "Uoptk": Uoptk.tolist(),
            "Udisk": Udisk.tolist(),
            "Uk": Uk.tolist(),
        }

        return d

    # ==================================================
    def write_u_mat(self, file_name="cwannier_u.mat"):
        """
        write Uoptk data.

        Args:
            file_name (str, optional): file name.
        """
        with open(file_name, "w") as fp:
            fp.write("# u.mat created by umat.py\n")
            fp.write("{} {} {}\n\n".format(self["num_k"], self["num_wann"], self["num_wann"]))

            for k in range(self["num_k"]):
                kv = self["kpoints"][k]
                fp.write("{0:18.12f} {1:18.12f} {2:18.12f} \n".format(kv[0], kv[1], kv[2]))
                for i in range(self["num_wann"]):
                    for j in range(self["num_wann"]):
                        fp.write("{0.real:18.12f}  {0.imag:18.12f} \n".format(self["Uoptk"][k][i][j]))

                fp.write("\n")

    # ==================================================
    def write_u_dis_mat(self, file_name="cwannier_u_dis.mat"):
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
                for i in range(self["num_bands"]):
                    for j in range(self["num_wann"]):
                        fp.write("{0.real:18.12f}  {0.imag:18.12f} \n".format(self["Udisk"][k][i][j]))

                fp.write("\n")

    # ==================================================
    @classmethod
    def _default_umat(cls):
        return _default_umat
