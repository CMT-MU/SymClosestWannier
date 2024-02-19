"""
Nnkp manages information needed to determine the required overlap elements Mmn(k,b) and projections A_{mn}(k).
"""

import os
import gzip
import tarfile
import itertools

import numpy as np
import scipy.linalg


_default = {
    "A": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "B": [[2 * np.pi, 0, 0], [0, 2 * np.pi, 0], [0, 0, 2 * np.pi]],
    "num_k": 1,
    "num_wann": 1,
    "num_atom": 1,
    "num_b": 1,
    "kpoints": [[0, 0, 0]],
    "kpoints_wo_shift": [[0, 0, 0]],
    "nnkpts": None,
    "nw2n": None,
    "nw2l": None,
    "nw2m": None,
    "nw2r": None,
    "nw2s": None,
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

    Attributes:
        _topdir (str): top directory.
        _seedname (str): seedname.
    """

    # ==================================================
    def __init__(self, topdir=None, seedname="cwannier", dic=None):
        """
        Nnkp manages information needed to determine the required overlap elements Mmn(k,b) and projections A_{mn}(k).

        Args:
            topdir (str, optional): directory of seedname.nnkp file.
            seedname (str, optional): seedname.
            dic (dict, optional): dictionary of Nnkp.
        """
        super().__init__()

        self._topdir = topdir
        self._seedname = seedname

        if dic is None:
            file_name = os.path.join(topdir, "{}.{}".format(seedname, "nnkp"))
            self.update(self.read(file_name))
        else:
            self.update(dic)

    # ==================================================
    def read(self, file_name="cwannier.nnkp"):
        """
        read seedname.nnkp file.

        Args:
            file_name (str, optional): file name.

        Returns:
            dict:
                - A                : real lattice vectors, A = [a1,a2,a3] (list), [[[1,0,0], [0,1,0], [0,0,1]]].
                - B                : reciprocal lattice vectors, B = [b1,b2,b3] (list), [[[2*pi,0,0], [0,2*pi,0], [0,0,2*pi]]].
                - num_k            : # of k points (int), [1].
                - num_wann         : # of WFs (int), [1].
                - num_atom         : # of atoms (int), [1].
                - num_b            : # of b-vectors (int), [1].
                - kpoints          : k-points, [[k1, k2, k3]] (crystal coordinate) (list), [[[0, 0, 0]]].
                - kpoints_wo_shift : k-points without shift, [[k1, k2, k3]] (crystal coordinate) (list), [[[0, 0, 0]]].
                - nnkpts           : nearest-neighbour k-points (list), [None].
                - nw2n             : atom position index of each WFs (list), [None].
                - nw2l             : l specifies the angular part Θlm(θ, φ) (list), [None].
                - nw2m             : m specifies the angular part Θlm(θ, φ) (list), [None].
                - nw2r             : r specifies the radial part Rr(r) (list), [None].
                - nw2s             : s specifies the spin, 1(up)/-1(dn) (list), [None].
                - atom_orb         : WFs indexes of each atom (list), [None].
                - atom_pos         : atom position index of each atom (list), [None].
                - atom_pos_r       : atom position of each atom in fractional coordinates with respect to the lattice vectors (list), [None].
                - bvec_cart        : b-vectors (cartesian coordinate) (list), [None].
                - bvec_crys        : b-vectors (crystal coordinate) (list), [None].
                - wb               : weight for each k-points and nearest-neighbour k-points (list), [None].
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

        d = Nnkp._default().copy()

        try:
            for i, line in enumerate(nnkp_data):
                if "begin real_lattice" in line:
                    d["A"] = np.genfromtxt(nnkp_data[i + 1 : i + 4], dtype=float).tolist()

                if "begin recip_lattice" in line:
                    d["B"] = np.genfromtxt(nnkp_data[i + 1 : i + 4], dtype=float).tolist()

                if "begin kpoints" in line:
                    d["num_k"] = int(nnkp_data[i + 1])
                    kpoints_wo_shift = np.genfromtxt(nnkp_data[i + 2 : i + 2 + d["num_k"]], dtype=float)
                    kpoints = np.mod(kpoints_wo_shift, 1)  # 0 <= kj < 1.0
                    if kpoints.ndim == 1:
                        d["kpoints"] = [kpoints.tolist()]
                        d["kpoints_wo_shift"] = [kpoints_wo_shift.tolist()]
                    else:
                        d["kpoints"] = kpoints.tolist()
                        d["kpoints_wo_shift"] = kpoints_wo_shift.tolist()

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
                    nw2s = np.zeros([d["num_wann"]], dtype=int)
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
                        if spinors:
                            spn_dat = nnkp_data[i + 2 + 3 * j + 2].split()[0]
                            nw2s[j] = int(spn_dat)
                        atom_orb_strlist.append(proj_str[0:40])
                        atom_pos_strlist.append(proj_str[0:35])

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

                    num_atom = len(atom_pos_r)
                    for i, pos in enumerate(atom_pos):
                        for p in pos:
                            for n in atom_orb[p]:
                                nw2n[n] = i

                    d["num_atom"] = num_atom
                    d["nw2n"] = nw2n.tolist()
                    d["nw2l"] = nw2l.tolist()
                    d["nw2m"] = nw2m.tolist()
                    d["nw2r"] = nw2r.tolist()
                    d["nw2s"] = nw2s.tolist()
                    d["atom_orb"] = atom_orb
                    d["atom_pos"] = atom_pos
                    d["atom_pos_r"] = atom_pos_r

            bvec_cart = np.zeros([d["num_b"], 3])
            bvec_crys = np.zeros([d["num_b"], 3])
            bbmat = np.zeros([d["num_b"], 9])
            try:
                Gp_idx = d["kpoints_wo_shift"].index([0.0, 0.0, 0.0])
            except:
                raise Exception("Gamma point must be included.")

            for i in range(d["num_b"]):
                kv = d["nnkpts"][Gp_idx][i]
                k = d["kpoints_wo_shift"][kv[0] - 1]
                k_b = d["kpoints_wo_shift"][kv[1] - 1]
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
    def bvec_idx(self, b, type="cart"):
        for ib in range(self["num_b"]):
            if np.allclose(np.array(self["bvec_" + type][ib]), b, rtol=1e-05, atol=1e-05):
                return ib

        assert False, b

    # ==================================================
    def bvec(self, d, type="cart"):
        """
        d : array of integer [5]
        return : b vector in cartesian/fractional coordinates
        """
        k = np.array(self["kpoints_wo_shift"][d[0] - 1])
        kb = np.array(self["kpoints_wo_shift"][d[1] - 1])
        G = np.array(d[2:5])
        b_crys = kb + G - k

        if type == "cart":
            return self.k_crys2cart(b_crys, self["B"])
        else:
            return b_crys

    # ==================================================
    def bveck(self, type="cart"):
        bk = np.zeros([self["num_k"], self["num_b"], 3], dtype=float)
        for ik in range(self["num_k"]):
            for ib in range(self["num_b"]):
                bk[ik, ib] = self.bvec(self["nnkpts"][ik][ib], type)

        return bk

    # ==================================================
    def wk(self):
        wk = np.zeros([self["num_k"], self["num_b"]], dtype=float)
        for ik in range(self["num_k"]):
            for ib in range(self["num_b"]):
                bk = self.bvec(self["nnkpts"][ik][ib])
                wk[ik, ib] = self["wb"][self.bvec_idx(bk)]

        return wk

    # ==================================================
    def kb2k(self):
        kb2k = np.zeros([self["num_k"], self["num_b"]], dtype=int)
        for ik in range(self["num_k"]):
            for ib in range(self["num_b"]):
                kb2k[ik, ib] = self["nnkpts"][ik][ib][1] - 1

        return kb2k

    # ==================================================
    def k_crys2cart(self, k, B):
        return np.matmul(k, B)

    # ==================================================
    def k_cart2crys(self, k, A):
        return np.matmul(A, k) / (2 * np.pi)

    # ==================================================
    @classmethod
    def _default(cls):
        return _default
