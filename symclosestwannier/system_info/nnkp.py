"""
Nnkp manages information needed to determine the required overlap elements Mmn(k,b) and projections A_{mn}(k).
"""
import os
import gzip
import tarfile
import itertools

import numpy as np
import scipy.linalg


_default_nnkp = {
    "A": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "B": [[2 * np.pi, 0, 0], [0, 2 * np.pi, 0], [0, 0, 2 * np.pi]],
    "num_k": 1,
    "num_wann": 1,
    "num_atom": 1,
    "num_b": 1,
    "kpoints": [[0, 0, 0]],
    "nnkpts": None,
    "nw2n": None,
    "nw2l": None,
    "nw2m": None,
    "nw2r": None,
    "atom_orb": None,
    "atom_pos": None,
    "atom_pos_r": None,
    "bvec_cart": None,
    "bvec_crys": None,
    "wb": None,
}


# ==================================================
class Nnkp(dict):
    """
    Nnkp manages information needed to determine the required overlap elements Mmn(k,b) and projections A_{mn}(k).
    """

    # ==================================================
    def __init__(self, topdir=None, seedname=None, dic=None):
        """
        initialize the class.

        Args:
            topdir (str, optional): directory of seedname.nnkp file.
            seedname (str, optional): seedname.
            dic (dict, optional): dictionary of Nnkp.
        """
        super().__init__()

        if dic is None:
            file_name = os.path.join(topdir, "{}.{}".format(seedname, "nnkp"))
            self.update(self.read(file_name))
        else:
            self.update(dic)

    # ==================================================
    def read(self, file_name):
        """
        read seedname.nnkp file.

        Args:
            file_name (str): file name.

        Returns:
            dict:
                - A          : real lattice vectors, A = [a1,a2,a3] (list), [[[1,0,0], [0,1,0], [0,0,1]]].
                - B          : reciprocal lattice vectors, B = [b1,b2,b3] (list), [[[2*pi,0,0], [0,2*pi,0], [0,0,2*pi]]].
                - num_k      : # of k points (int), [1].
                - num_wann   : # of CWFs (int), [1].
                - num_atom   : # of atoms (int), [1].
                - num_b      : # of b-vectors (int), [1].
                - kpoints    : k-points, [[k1, k2, k3]] (crystal coordinate) (list), [[[0, 0, 0]]].
                - nnkpts     : nearest-neighbour k-points (list), [None].
                - nw2n       : atom position index of each CWFs (list), [None].
                - nw2l       : l specifies the angular part Θlm(θ, φ) (list), [None].
                - nw2m       : m specifies the angular part Θlm(θ, φ) (list), [None].
                - nw2r       : r specifies the radial part Rr(r) (list), [None].
                - atom_orb   : CWFs indexes of each atom (list), [None].
                - atom_pos   : atom position index of each atom (list), [None].
                - atom_pos_r : atom position of each atom in fractional coordinates with respect to the lattice vectors (list), [None].
                - bvec_cart  : b-vectors (cartesian coordinate) (list), [None].
                - bvec_crys  : b-vectors (crystal coordinate) (list), [None].
                - wb         : weight for each k-points and nearest-neighbour k-points (list), [None].
        """
        if os.path.exists(file_name):
            with open(file_name) as fp:
                nnkp_data = fp.readlines()
        elif os.path.exists(file_name + ".gz"):
            with gzip.open(file_name + ".gz", "rt") as fp:
                nnkp_data = fp.readlines()
        elif os.path.exists(file_name + ".tar.gz"):
            with tarfile.open(file_name + "tar.gz", "rt") as fp:
                nnkp_data = fp.readlines()
        else:
            raise Exception("failed to read nnkp file: " + file_name)

        d = Nnkp._default_nnkp().copy()

        try:
            for i, line in enumerate(nnkp_data):
                if "begin real_lattice" in line:
                    d["A"] = np.genfromtxt(nnkp_data[i + 1 : i + 4], dtype=float).tolist()

                if "begin recip_lattice" in line:
                    d["B"] = np.genfromtxt(nnkp_data[i + 1 : i + 4], dtype=float).tolist()

                if "begin kpoints" in line:
                    d["num_k"] = int(nnkp_data[i + 1])
                    kpoints = np.genfromtxt(nnkp_data[i + 2 : i + 2 + d["num_k"]], dtype=float)
                    d["kpoints"] = np.mod(kpoints, 1).tolist()  # 0 <= kj < 1.0

                if "begin nnkpts" in line:
                    d["num_b"] = int(nnkp_data[i + 1])
                    dat = np.genfromtxt(nnkp_data[i + 2 : i + 2 + d["num_k"] * d["num_b"]], dtype=int)
                    d["nnkpts"] = dat.reshape(d["num_k"], d["num_b"], 5).tolist()

                if "begin projections" in line or "begin spinor_projections" in line:
                    spinors = "begin spinor_projections" in line
                    d["num_wann"] = int(nnkp_data[i + 1])
                    nw2n = np.zeros([d["num_wann"]], dtype=int)
                    nw2l = np.zeros([d["num_wann"]], dtype=int)
                    nw2m = np.zeros([d["num_wann"]], dtype=int)
                    nw2r = np.zeros([d["num_wann"]], dtype=int)
                    atom_orb_strlist = []
                    atom_pos_strlist = []
                    # read projections
                    for j in range(d["num_wann"]):
                        if spinors:
                            proj_str = nnkp_data[i + 2 + 3 * j]
                        else:
                            proj_str = nnkp_data[i + 2 + 2 * j]
                        proj_dat = proj_str.split()
                        nw2l[j] = int(proj_dat[3])
                        nw2m[j] = int(proj_dat[4])
                        nw2r[j] = int(proj_dat[5])
                        atom_orb_strlist.append(proj_str[0:40])
                        atom_pos_strlist.append(proj_str[0:35])
                    # set atom_pos_r, atom_pos, atom_orb
                    #   for example, Fe case
                    #   atom_pos_r: [[0.0, 0.0, 0.0]]
                    #   atom_pos: [[0, 1, 2]]
                    #   atom_orb: [[0, 1], [2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
                    atom_orb_uniq = sorted(set(atom_orb_strlist), key=atom_orb_strlist.index)
                    atom_pos_uniq = sorted(set(atom_pos_strlist), key=atom_pos_strlist.index)
                    atom_orb = []
                    for orb_str in atom_orb_uniq:
                        indexes = [j for j, x in enumerate(atom_orb_strlist) if x == orb_str]
                        atom_orb.append(indexes)
                    atom_pos = []
                    atom_pos_r = []
                    for pos_str in atom_pos_uniq:
                        indexes = [j for j, x in enumerate(atom_orb_uniq) if pos_str in x]
                        atom_pos.append(indexes)
                        atom_pos_r.append([float(x) for x in pos_str.split()[0:3]])
                    # print ("atom_pos_r: " + str(atom_pos_r))
                    # print ("atom_pos: " + str( atom_pos))
                    # print ("atom_orb: " + str(atom_orb))
                    num_atom = len(atom_pos_r)
                    for i, pos in enumerate(atom_pos):
                        for p in pos:
                            for n in atom_orb[p]:
                                nw2n[n] = i
                    # for j in range(d["num_wann"]):
                    #    print("nw {:3d} : n = {:3d}, l = {:3d}, m = {:3d}".format(j, nw2n[j], nw2l[j], nw2m[j]))

                    d["num_atom"] = num_atom
                    d["nw2n"] = nw2n.tolist()
                    d["nw2l"] = nw2l.tolist()
                    d["nw2m"] = nw2m.tolist()
                    d["nw2r"] = nw2r.tolist()
                    d["atom_orb"] = atom_orb
                    d["atom_pos"] = atom_pos
                    d["atom_pos_r"] = atom_pos_r

            # calculate b-vectors
            bvec_cart = np.zeros([d["num_b"], 3])
            bvec_crys = np.zeros([d["num_b"], 3])
            bbmat = np.zeros([d["num_b"], 9])
            for i in range(d["num_b"]):
                kv = d["nnkpts"][0][i]
                k = d["kpoints"][kv[0] - 1]
                k_b = d["kpoints"][kv[1] - 1]
                b = np.array(k_b) - np.array(k) + np.array(kv[2:5])
                bvec_cart[i, :] = self.k_crys2cart(b, d["B"])
                bvec_crys[i, :] = self.k_cart2crys(bvec_cart[i, :], d["A"])
                bbmat[i, :] = [bvec_cart[i, a] * bvec_cart[i, b] for a, b in itertools.product(range(3), range(3))]

            delta_ab = np.array([a == b for a, b in itertools.product(range(3), range(3))]).astype(int)
            wb = np.matmul(delta_ab, scipy.linalg.pinv(bbmat))

            d["bvec_cart"] = bvec_cart.tolist()
            d["bvec_crys"] = bvec_crys.tolist()
            d["wb"] = wb.tolist()

        except Exception as e:
            print("failed to read: " + file_name)
            print("type:" + str(type(e)))
            print("args:" + str(e.args))
            print(str(e))

        return d

    # ==================================================
    def bvec_num(self, b, type="cart"):
        for ib in range(self["num_b"]):
            if np.allclose(self["bvec_" + type][ib, :], b):
                return ib

        assert False, b

    # ==================================================
    def k_crys2cart(self, k, B):
        return np.matmul(k, B)

    # ==================================================
    def k_cart2crys(self, k, A):
        return np.matmul(A, k) / (2 * np.pi)

    # ==================================================
    @classmethod
    def _default_nnkp(cls):
        return _default_nnkp