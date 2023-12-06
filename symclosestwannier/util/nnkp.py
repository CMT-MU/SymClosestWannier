"""
Nnkp manages information needed to determine the required overlap elements Mmn(k,b) and projections A_{mn}(k).
"""
import os
import gzip
import tarfile
import itertools

import numpy as np
import scipy.linalg


# ==================================================
class Nnkp(dict):
    """
    Nnkp manages information needed to determine the required overlap elements Mmn(k,b) and projections A_{mn}(k).
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
        file_nnkp = os.path.join(topdir, "{}.{}".format(seedname, "nnkp"))

        self.update(self.read(file_nnkp))

    # ==================================================
    def read(self, file_nnkp):
        """
        read seedname.nnkp file.

        Args:
            file_nnkp (str): file name.

        Returns:
            dict:
                A (ndarray): real lattice vectors, A = [a1,a2,a3].
                B (ndarray): reciprocal lattice vectors, B = [b1,b2,b3].
                num_k (int): # of k points.
                num_wann (int): # of CWFs.
                num_atom (int): # of atoms.
                num_b (int): # of b-vectors.
                kpoints (ndarray) : k-points, [[k1, k2, k3]] (crystal coordinate).
                nnkpts (ndarray): nearest-neighbour k-points.
                nw2n (ndarray):
                nw2l (ndarray):
                nw2m (ndarray):
                nw2r (ndarray):
                atom_orb (ndarray):
                atom_pos (ndarray):
                atoms_frac (ndarray): atomic positions in fractional coordinates with respect to the lattice vectors, {atom: [r1,r2,r3]}.
                bvec_cart: b-vectors (cartesian coordinate).
                bvec_crys: b-vectors (crystal coordinate).
                wb (ndarray): weight for each k-points and nearest-neighbour k-points.
        """
        d = {}

        if os.path.exists(file_nnkp):
            with open(file_nnkp) as fp:
                nnkp_data = fp.readlines()
        elif os.path.exists(file_nnkp + ".gz"):
            with gzip.open(file_nnkp + ".gz", "rt") as fp:
                nnkp_data = fp.readlines()
        elif os.path.exists(file_nnkp + ".tar.gz"):
            with tarfile.open(file_nnkp + "tar.gz", "rt") as fp:
                amn_data = fp.readlines()
        else:
            raise Exception("failed to read nnkp file: " + file_nnkp)

        try:
            for i, line in enumerate(nnkp_data):
                if "begin real_lattice" in line:
                    d["A"] = np.genfromtxt(nnkp_data[i + 1 : i + 4], dtype=float)

                if "begin recip_lattice" in line:
                    d["B"] = np.genfromtxt(nnkp_data[i + 1 : i + 4], dtype=float)

                if "begin kpoints" in line:
                    d["num_k"] = int(nnkp_data[i + 1])
                    kpoints = np.genfromtxt(nnkp_data[i + 2 : i + 2 + d["num_k"]], dtype=float)
                    d["kpoints"] = np.mod(kpoints, 1)  # 0 <= kj < 1.0

                if "begin nnkpts" in line:
                    d["num_b"] = int(nnkp_data[i + 1])
                    dat = np.genfromtxt(nnkp_data[i + 2 : i + 2 + d["num_k"] * d["num_b"]], dtype=int)
                    d["nnkpts"] = dat.reshape(d["num_k"], d["num_b"], 5)

                if "begin projections" in line or "begin spinor_projections" in line:
                    spinors = "begin spinor_projections" in line
                    num_wann = int(nnkp_data[i + 1])
                    nw2n = np.zeros([num_wann], dtype=int)
                    nw2l = np.zeros([num_wann], dtype=int)
                    nw2m = np.zeros([num_wann], dtype=int)
                    nw2r = np.zeros([num_wann, 3], dtype=float)
                    atom_orb_strlist = []
                    atom_pos_strlist = []
                    # read projections
                    for j in range(num_wann):
                        if spinors:
                            proj_str = nnkp_data[i + 2 + 3 * j]
                        else:
                            proj_str = nnkp_data[i + 2 + 2 * j]
                        proj_dat = proj_str.split()
                        nw2l[j] = int(proj_dat[3])
                        nw2m[j] = int(proj_dat[4])
                        nw2r[j, :] = [float(x) for x in proj_dat[0:3]]
                        atom_orb_strlist.append(proj_str[0:40])
                        atom_pos_strlist.append(proj_str[0:35])
                    # set atoms_frac, atom_pos, atom_orb
                    #   for example, Fe case
                    #   atoms_frac: [[0.0, 0.0, 0.0]]
                    #   atom_pos: [[0, 1, 2]]
                    #   atom_orb: [[0, 1], [2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
                    atom_orb_uniq = sorted(set(atom_orb_strlist), key=atom_orb_strlist.index)
                    atom_pos_uniq = sorted(set(atom_pos_strlist), key=atom_pos_strlist.index)
                    atom_orb = []
                    for orb_str in atom_orb_uniq:
                        indexes = [j for j, x in enumerate(atom_orb_strlist) if x == orb_str]
                        atom_orb.append(indexes)
                    atom_pos = []
                    atoms_frac = []
                    for pos_str in atom_pos_uniq:
                        indexes = [j for j, x in enumerate(atom_orb_uniq) if pos_str in x]
                        atom_pos.append(indexes)
                        atoms_frac.append([float(x) for x in pos_str.split()[0:3]])
                    # print ("atoms_frac: " + str(atoms_frac))
                    # print ("atom_pos: " + str( atom_pos))
                    # print ("atom_orb: " + str(atom_orb))
                    num_atom = len(atoms_frac)
                    for i, pos in enumerate(atom_pos):
                        for p in pos:
                            for n in atom_orb[p]:
                                nw2n[n] = i
                    # for j in range(num_wann):
                    #    print("nw {:3d} : n = {:3d}, l = {:3d}, m = {:3d}".format(j, nw2n[j], nw2l[j], nw2m[j]))

                    d["nw2n"] = nw2n
                    d["nw2l"] = nw2l
                    d["nw2m"] = nw2m
                    d["nw2r"] = nw2r
                    d["atom_orb"] = atom_orb
                    d["atom_pos"] = atom_pos
                    d["atoms_frac"] = atoms_frac

            # calculate b-vectors
            bvec_cart = np.zeros([d["num_b"], 3])
            bvec_crys = np.zeros([d["num_b"], 3])
            bbmat = np.zeros([d["num_b"], 9])
            for i in range(d["num_b"]):
                kv = d["nnkpts"][0, i, :]
                k = d["kpoints"][kv[0] - 1]
                k_b = d["kpoints"][kv[1] - 1]
                b = k_b - k + np.array(kv[2:5])
                bvec_cart[i, :] = self.k_crys2cart(b, d["B"])
                bvec_crys[i, :] = self.k_cart2crys(bvec_cart[i, :], d["A"])
                bbmat[i, :] = [bvec_cart[i, a] * bvec_cart[i, b] for a, b in itertools.product(range(3), range(3))]

            delta_ab = np.array([a == b for a, b in itertools.product(range(3), range(3))]).astype(int)
            wb = np.matmul(delta_ab, scipy.linalg.pinv(bbmat))

            d["bvec_cart"] = bvec_cart
            d["bvec_crys"] = bvec_crys
            d["wb"] = wb

        except Exception as e:
            print("failed to read: " + file_nnkp)
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
