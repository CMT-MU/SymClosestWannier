"""
Symmetry-adapted Closest Wannier (SymCW) tight-binding model based on Symmetry-Adapted Multipole Basis (SAMB) and Plane-Wave (PW) DFT calculation.
"""
import os
import sys
import codecs
import time

import sympy as sp
import numpy as np
from numpy import linalg as npl
from scipy import linalg as spl

from gcoreutils.io_util import read_dict, write_dict
from gcoreutils.nsarray import NSArray
from multipie.tag.tag_multipole import TagMultipole

from symclosestwannier.util.reader import amn_reader, eig_reader, win_reader
from symclosestwannier.util.header import (
    start_str,
    end_str,
    info_header_str,
    data_header_str,
    kpoints_header_str,
    rpoints_header_str,
    hk_header_str,
    sk_header_str,
    pk_header_str,
    hr_header_str,
    sr_header_str,
    z_header_str,
    s_header_str,
)


# ==================================================
class SymCW(dict):
    """
    Symmetry-adapted Closest Wannier (SymCW) tight-binding model based on Symmetry-Adapted Multipole Basis (SAMB) and Plane-Wave (PW) DFT calculation.
    """

    # ==================================================
    def __init__(self, model_dict):
        """
        initialize the class.

        Args:
            model_dict (dict): minimal model information.
        """
        os.makedirs(os.path.abspath(model_dict["outdir"]), exist_ok=True)
        os.chdir(model_dict["outdir"])
        sys.path.append(os.getcwd())
        model_dict["outdir"] = os.getcwd().replace(os.sep, "/")

        self["info"] = model_dict

        self._print(start_str)
        start0 = time.time()

        self._print(f"* {self['info']['seedname']} \n", mode="a")

        #####

        if self["info"]["restart"] == "wannierise":
            self._print(" - reading output of DFT calculation ... ", end="", mode="a")
            start = time.time()

            # Kohn-Sham energy
            Ek = eig_reader(".", self["info"]["seedname"], encoding="utf-8")
            # overlap between Kohn-Sham orbitals and non-orthogonalized atomic orbitals
            Ak = amn_reader(".", self["info"]["seedname"], encoding="utf-8")
            # wannier input
            kpoints, kpoint, kpoint_path, unit_cell_cart, atoms_frac, atoms_cart = win_reader(
                ".", self["info"]["seedname"], encoding="utf-8"
            )

            # band calculation
            if kpoint is not None and kpoint_path is not None:
                kpoint = {i: NSArray(j, "vector", fmt="value") for i, j in kpoint.items()}
                N1 = self["info"]["N1"]
                A = NSArray(unit_cell_cart, "matrix", fmt="value")
                B = A.inverse()
                kpoints_path, k_linear, k_dis_pos = NSArray.grid_path(kpoint, kpoint_path, N1, B)
            else:
                kpoints_path = None
                k_linear = None
                k_dis_pos = None

            # number of k points
            num_k = Ak.shape[0]
            # number of Kohn-Sham orbitals
            num_bands = Ak.shape[1]
            # number of pseudo atomic orbitals
            num_wann = Ak.shape[2]

            if num_bands < num_wann:
                raise Exception("number of Kohn-Sham orbitals is smaller than that of the pseudo atomic orbitals.")

            # overlap matrix, Sk = Ak^† @ Ak (num_wann × num_wann matrix)
            Sk = Ak.transpose(0, 2, 1).conjugate() @ Ak

            # projectability
            Pk = np.real(np.diagonal(Ak @ Ak.transpose(0, 2, 1).conjugate(), axis1=1, axis2=2))

            end = time.time()
            self._print(f"done ({'{:.2f}'.format(end - start)} [sec])", mode="a")

            #####

            if self["info"]["proj_min"] > 0.0:
                self._print(
                    f" - eliminating bands with low projectability (proj_min = {self['info']['proj_min']}) ... ",
                    end="",
                    mode="a",
                )
                start = time.time()

                # band index for projection
                proj_band_idx = [
                    [n for n in range(num_bands) if Pk[k][n] > self["info"]["proj_min"]] for k in range(num_k)
                ]

                for k in range(num_k):
                    if len(proj_band_idx[k]) < num_wann:
                        raise Exception(
                            f"proj_min = {self['info']['proj_min']} is too large or PAOs are inappropriate."
                        )

                # eliminate bands with low projectability
                Ek = [Ek[k, proj_band_idx[k]] for k in range(num_k)]
                Ak = [Ak[k, proj_band_idx[k], :] for k in range(num_k)]

                end = time.time()
                self._print(f"done ({'{:.2f}'.format(end - start)} [sec])", mode="a")

            #####

            if self["info"]["disentangle"]:
                self._print(" - disentanglement ... ", end="", mode="a")
                start = time.time()

                # fermi-dirac function
                def fermi(x, T=0.01):
                    return 0.5 * (1.0 - np.tanh(0.5 * x / T))

                # weight function for disentanglement
                def weight(e, e0, e1, T0, T1, delta=10e-12):
                    return fermi(e0 - e, T0) + fermi(e - e1, T1) - 1.0 + delta

                Ak = [
                    np.array(
                        weight(
                            Ek[k],
                            self["info"]["dis_win_emin"],
                            self["info"]["dis_win_emax"],
                            self["info"]["smearing_temp_min"],
                            self["info"]["smearing_temp_max"],
                            self["info"]["delta"],
                        )[:, np.newaxis]
                        * Ak[k]
                    )
                    for k in range(num_k)
                ]

                end = time.time()
                self._print(f"done ({'{:.2f}'.format(end - start)} [sec])", mode="a")

            #####

            self._print(" - constructing TB Hamiltonian ... ", end="", mode="a")
            start = time.time()

            Sk = np.array([Ak[k].transpose().conjugate() @ Ak[k] for k in range(num_k)])

            if self["info"]["svd"]:  # orthogonalize PAOs by singular value decomposition (SVD)

                def U_mat(k):
                    u, _, vd = np.linalg.svd(Ak[k], full_matrices=False)
                    return u @ vd

                Uk = [U_mat(k) for k in range(num_k)]

            else:  # orthogonalize PAOs by Lowdin's method
                S2k_inv = np.array([npl.inv(spl.sqrtm(Sk[k])) for k in range(num_k)])
                Uk = [Ak[k] @ S2k_inv[k] for k in range(num_k)]

            # projection from KS energies to PAOs Hamiltonian
            diag_Ek = [np.diag(Ek[k]) for k in range(num_k)]
            Hk = np.array([Uk[k].transpose().conjugate() @ diag_Ek[k] @ Uk[k] for k in range(num_k)])

            end = time.time()
            self._print(f"done ({'{:.2f}'.format(end - start)} [sec])", mode="a")

            #####

            self["info"] |= {"num_k": num_k, "num_bands": num_bands, "num_wann": num_wann}

            if kpoint is not None:
                self["info"]["kpoint"] = {k: str(v.tolist()) for k, v in kpoint.items()}
            if kpoint_path is not None:
                self["info"]["kpoint_path"] = kpoint_path
            if unit_cell_cart is not None:
                self["info"]["unit_cell_cart"] = str(unit_cell_cart.tolist())
            if atoms_frac is not None:
                self["info"]["atoms_frac"] = {k: str(v.tolist()) for k, v in atoms_frac.items()}
            if atoms_cart is not None:
                self["info"]["atoms_cart"] = {k: str(v.tolist()) for k, v in atoms_cart.items()}

            self["data"] = {
                "kpoints": kpoints.tolist(),
                "rpoints": SymCW.kpoints_to_rpoints(kpoints).tolist(),
                #
                "Pk": Pk.tolist(),
                #
                "Hk": Hk.tolist(),
                "Sk": Sk.tolist(),
            }

            if kpoints_path is not None:
                self["data"]["kpoints_path"] = kpoints_path.tolist()
            if k_linear is not None:
                self["data"]["k_linear"] = k_linear.tolist()
            if k_dis_pos is not None:
                self["data"]["k_dis_pos"] = k_dis_pos

            self["data"]["matrix_dict"] = None
        else:
            self["info"] = self.read(f"{self['info']['seedname']}_info.py")
            self["data"] = self.read(f"{self['info']['seedname']}_data.py")

        if self["info"]["symmetrization"]:
            mp_outdir = self["info"]["mp_outdir"]
            mp_seedname = self["info"]["mp_seedname"]
            ket_amn = self["info"].get("ket_amn", None)
            irreps = self["info"].get("irreps", "all")
            self.symmetrize(mp_outdir, mp_seedname, ket_amn, irreps)

        #####

        end0 = time.time()
        self._print(f"\n - total elapsed_time: {'{:.2f}'.format(end0 - start0)} [sec]", mode="a")
        self._print(end_str, mode="a")

    # ==================================================
    @property
    def Hr(self):
        return SymCW.fourier_transform_k_to_r(self["data"]["Hk"], self["data"]["kpoints"])[0]

    # ==================================================
    @property
    def Sr(self):
        return SymCW.fourier_transform_k_to_r(self["data"]["Sk"], self["data"]["kpoints"])[0]

    # ==================================================
    @property
    def Hk_path(self):
        return SymCW.interpolate(
            self["data"]["Hk"], self["data"]["kpoints"], self["data"]["kpoints_path"], self["data"]["rpoints"]
        )

    # ==================================================
    @property
    def Sk_path(self):
        return SymCW.interpolate(
            self["data"]["Sk"], self["data"]["kpoints"], self["data"]["kpoints_path"], self["data"]["rpoints"]
        )

    # ==================================================
    @property
    def Hr_sym(self):
        if "z" in self["data"]:
            return self.construct_Or(list(self["data"]["z"].values()))
        else:
            return None

    # ==================================================
    @property
    def Sr_sym(self):
        if "s" in self["data"]:
            return self.construct_Or(list(self["data"]["s"].values()))
        else:
            return None

    # ==================================================
    @property
    def Hk_sym(self):
        if "z" in self["data"]:
            return self.construct_Ok(list(self["data"]["z"].values()), self["data"]["kpoints"])
        else:
            return None

    # ==================================================
    @property
    def Sk_sym(self):
        if "s" in self["data"]:
            return self.construct_Ok(list(self["data"]["s"].values()), self["data"]["kpoints"])
        else:
            return None

    # ==================================================
    @property
    def Hk_sym_path(self):
        if "z" in self["data"]:
            return self.construct_Ok(list(self["data"]["z"].values()), self["data"]["kpoints_path"])
        else:
            return None

    # ==================================================
    @property
    def Sk_sym_path(self):
        if "s" in self["data"]:
            return self.construct_Ok(list(self["data"]["s"].values()), self["data"]["kpoints_path"])
        else:
            return None

    # ==================================================
    def _print(self, msg, mode="w", *args, **kwargs):
        if self["info"]["verbose"]:
            print(msg, *args, **kwargs)

        print(msg, *args, **kwargs, file=codecs.open(f"{self['info']['seedname']}.cwout", mode, "utf-8"))

    # ==================================================
    def read(self, file_dict, dir=None):
        """
        read dict file or dict itself.

        Args:
            file_dict (str or dict): filename of dict. or dict.
            dir (str, optional): directory.

        Returns:
            dict: read dict.
        """
        if dir is None:
            dir = self["info"]["outdir"]

        dir = dir[:-1] if dir[-1] == "/" else dir

        if type(file_dict) == str:
            full = dir + "/" + file_dict
            if os.path.isfile(full):
                dic = read_dict(full)
                self._print(f"  * read '{full}'.", mode="a")
            else:
                raise Exception(f"cannot open {full}.")
        else:
            dic = file_dict

        return dic

    # ==================================================
    def write(self, filename, dic, header=None, var=None, dir=None):
        """
        write dict to file.

        Args:
            filename (str): file name.
            dic (dict): dict to write.
            header (str, optional): header of dict.
            var (str, optional): variable name for dict.
            dir (str, optional): directory.
        """
        if dir is None:
            dir = self["info"]["outdir"]

        dir = dir[:-1] if dir[-1] == "/" else dir

        full = dir + "/" + filename
        write_dict(full, dic, header, var)
        self._print(f"  * wrote '{filename}'.", mode="a")

    # ==================================================
    def symmetrize(self, mp_outdir="./", mp_seedname=None, ket_amn=None, irreps="all"):
        """
        write dict to file.

        Args:
            mp_outdir (str, optional): output files for multipie are found in this directory.
            mp_seedname (str, optional): seedname for seedname.win and seedname.cwin files.
            ket_amn (list): ket basis list in the seedname.amn file. The format of each ket must be same as the "ket" in sambname_model.py file. See sambname["info"]["ket"] in sambname_model.py file for the format.
            irreps (str/list, optional): list of irreps to be considered.
        """
        self._print(" - symmetrization ... ", end="\n", mode="a")
        start0 = time.time()

        if mp_seedname is None:
            mp_seedname = self["info"]["seedname"]

        #####

        rpoints = self["data"]["rpoints"]

        num_k = self["info"]["num_k"]
        num_wann = self["info"]["num_wann"]

        rpoints = self["data"]["rpoints"]
        kpoints = self["data"]["kpoints"]
        Hk = np.array(self["data"]["Hk"])

        Hr_dict = SymCW.matrix_dict_r(self.Hr, self["data"]["rpoints"])
        Sr_dict = SymCW.matrix_dict_r(self.Sr, self["data"]["rpoints"])

        #####

        self._print("   - reading output of multipie ... ", end="\n", mode="a")
        start = time.time()

        model = self.read(f"{self['info']['mp_seedname']}_model.py", dir=mp_outdir)
        samb = self.read(f"{self['info']['mp_seedname']}_samb.py", dir=mp_outdir)
        mat = self.read(f"{self['info']['mp_seedname']}_matrix.py", dir=mp_outdir)

        ket_samb = model["info"]["ket"]

        # sort orbitals
        if ket_amn is not None:
            idx_list = [ket_amn.index(o) for o in ket_samb]
            Hk = Hk[:, idx_list, :]
            Hk = Hk[:, :, idx_list]

            idx_list = [ket_samb.index(o) for o in ket_amn]
            Hr_dict = {(n1, n2, n3, idx_list[a], idx_list[b]): v for (n1, n2, n3, a, b), v in Hr_dict.items()}
            Sr_dict = {(n1, n2, n3, idx_list[a], idx_list[b]): v for (n1, n2, n3, a, b), v in Sr_dict.items()}

        if irreps == "all":
            irreps = model["info"]["generate"]["irrep"]
        elif irreps == "full":
            irreps = [model["info"]["generate"]["irrep"][0]]

        for zj, (tag, _) in samb["data"]["Z"].items():
            if TagMultipole(tag).irrep not in irreps:
                del mat["matrix"][zj]

        tag_dict = {zj: tag for zj, (tag, _) in samb["data"]["Z"].items()}
        Zr_dict = {
            (zj, tag_dict[zj]): {k: complex(sp.sympify(v)) for k, v in d.items()} for zj, d in mat["matrix"].items()
        }
        mat["matrix"] = {zj: {k: complex(sp.sympify(v)) for k, v in d.items()} for zj, d in mat["matrix"].items()}

        lattice = model["info"]["group"][1].split("/")[1].replace(" ", "")[0]
        if lattice != "P":
            cell_site = {}
            for site, v in mat["cell_site"].items():
                if "(" in site and ")" in site:
                    if "(1)" in site:
                        cell_site[site[:-3]] = v
                else:
                    cell_site[site] = v

            mat["cell_site"] = cell_site

        end = time.time()
        self._print(f"   done ({'{:.2f}'.format(end - start)} [sec])", mode="a")

        #####

        self._print(
            "   - decomposing Hamiltonian Hr as linear combination of SAMBs ... ",
            end="",
            mode="a",
        )
        start = time.time()

        z = SymCW.samb_decomp(Hr_dict, Zr_dict)

        end = time.time()
        self._print(f"done ({'{:.2f}'.format(end - start)} [sec])", mode="a")

        #####

        self._print("   - decomposing overlap Sr as linear combination of SAMBs ... ", end="", mode="a")
        start = time.time()

        s = SymCW.samb_decomp(Sr_dict, Zr_dict)

        end = time.time()
        self._print(f"done ({'{:.2f}'.format(end - start)} [sec])", mode="a")

        #####

        rpoints_mp = [(n1, n2, n3) for Zj_dict in Zr_dict.values() for (n1, n2, n3, _, _) in Zj_dict.keys()]
        rpoints_mp = sorted(list(set(rpoints_mp)), key=rpoints_mp.index)

        if not mat["molecule"]:
            kpoint = {i: NSArray(j, "vector", fmt="value") for i, j in self["info"]["kpoint"].items()}
            kpoint_path = self["info"]["kpoint_path"]
            N1 = self["info"]["N1"]
            A = model["detail"]["A"]
            B = NSArray(A, "matrix", fmt="value").T.inverse()
            kpoints_path, _, _ = NSArray.grid_path(kpoint, kpoint_path, N1, B)

        #####

        self["info"]["mp_outdir"] = mp_outdir
        self["info"]["mp_seedname"] = mp_seedname
        self["info"]["ket_amn"] = ket_amn
        self["info"]["irreps"] = irreps

        self["data"] = {
            "z": z,
            "s": s,
            #
            "rpoints_mp": rpoints_mp,
        } | self["data"]

        self["data"]["matrix_dict"] = mat

        #####

        self._print("   - evaluating fitting accuracy ... ", end="\n", mode="a")
        start = time.time()

        Ek_grid, _ = np.linalg.eigh(Hk)
        Ek_grid_sym, _ = np.linalg.eigh(self.Hk_sym)
        num_k, num_wann = Ek_grid_sym.shape
        Ek_RMSE_grid = np.sum(np.abs(Ek_grid_sym - Ek_grid)) / num_k / num_wann * 1000  # [meV]

        self._print(
            f"     * RMSE of eigen values between CW and Symmetry-Adapted CW models (grid) = {'{:.4f}'.format(Ek_RMSE_grid)} [meV]",
            end="\n",
            mode="a",
        )

        #####

        if not mat["molecule"]:
            Hk_path = SymCW.interpolate(Hk, kpoints, kpoints_path, rpoints)
            Ek_path, _ = np.linalg.eigh(Hk_path)
            Ek_path_sym, _ = np.linalg.eigh(self.Hk_sym_path)
            num_k, num_wann = Ek_path_sym.shape
            Ek_RMSE_path = np.sum(np.abs(Ek_path_sym - Ek_path)) / num_k / num_wann * 1000  # [meV]
            self._print(
                f"     * RMSE of eigen values between CW and Symmetry-Adapted CW models (path) = {'{:.4f}'.format(Ek_RMSE_path)} [meV]",
                end="\n",
                mode="a",
            )

        end = time.time()
        self._print(f"    done ({'{:.2f}'.format(end - start)} [sec])", mode="a")

        #####

        if Ek_RMSE_grid is not None:
            self["data"] = {"Ek_RMSE_grid": Ek_RMSE_grid} | self["data"]

        if not mat["molecule"]:
            self["data"] = {"Ek_RMSE_path": Ek_RMSE_path} | self["data"]

        self["info"]["symmetrization"] = True

        #####

        end0 = time.time()
        self._print(f"  done ({'{:.2f}'.format(end0 - start0)} [sec])", mode="a")

    # ==================================================
    def construct_Or(self, z):
        """
        arbitrary operator constructed by linear combination of SAMBs in real space representation.

        Args:
            z (list): parameter set, [z_j].

        Returns:
            ndarray: matrix, [#r, dim, dim].
        """
        num_wann = self["info"]["num_wann"]
        rpoints_mp = np.array(self["data"]["rpoints_mp"])

        Or_dict = {
            (n1, n2, n3, a, b): 0.0 for (n1, n2, n3) in rpoints_mp for a in range(num_wann) for b in range(num_wann)
        }
        for j, d in enumerate(self["data"]["matrix_dict"]["matrix"].values()):
            zj = z[j]
            for (n1, n2, n3, a, b), v in d.items():
                Or_dict[(n1, n2, n3, a, b)] += zj * v

        Or = np.array(
            [
                [[Or_dict.get((n1, n2, n3, a, b), 0.0) for b in range(num_wann)] for a in range(num_wann)]
                for (n1, n2, n3) in rpoints_mp
            ]
        )

        return Or

    # ==================================================
    def construct_Ok(self, z, kpoints):
        """
        arbitrary operator constructed by linear combination of SAMBs in k space representation.

        Args:
            z (list): parameter set, [z_j].

        Returns:
            ndarray: matrix, [#k, dim, dim].
        """
        rpoints_mp = np.array(self["data"]["rpoints_mp"])
        kpoints = np.array(kpoints)
        cell_site = self["data"]["matrix_dict"]["cell_site"]
        ket = self["data"]["matrix_dict"]["ket"]
        atoms_positions = [
            NSArray(cell_site[ket[a].split("@")[1]][0], style="vector", fmt="value").tolist() for a in range(len(ket))
        ]

        Or = self.construct_Or(z)
        Ok, _ = SymCW.fourier_transform_r_to_k(Or, rpoints_mp, kpoints, atoms_positions=atoms_positions)

        return Ok

    # ==================================================
    @classmethod
    def get_rpoints(cls, nr1, nr2, nr3, unit_cell_cart=np.eye(3)):
        """
        get lattice points, R = (R1, R2, R3).
        R = R1*a1 + R2*a2 + R3*a3
        Rj: lattice vector.

        Args:
            nr1 (int): # of lattice point a1 direction.
            nr2 (int): # of lattice point a2 direction.
            nr3 (int): # of lattice point a3 direction.
            unit_cell_cart (ndarray, optional): transform matrix, [a1,a2,a3].

        Returns:
            tuple: (R, Rfft, idx).
        """
        A = unit_cell_cart
        nrtot = nr1 * nr2 * nr3

        R = np.zeros((nrtot, 3), dtype=float)
        idx = np.zeros((nr1, nr2, nr3), dtype=int)
        Rfft = np.zeros((nr1, nr2, nr3, 3), dtype=float)

        for i in range(nr1):
            for j in range(nr2):
                for k in range(nr3):
                    n = k + j * nr3 + i * nr2 * nr3
                    R1 = float(i) / float(nr1)
                    R2 = float(j) / float(nr2)
                    R3 = float(k) / float(nr3)
                    if R1 >= 0.5:
                        R1 = R1 - 1.0
                    if R2 >= 0.5:
                        R2 = R2 - 1.0
                    if R3 >= 0.5:
                        R3 = R3 - 1.0
                    R1 -= int(R1)
                    R2 -= int(R2)
                    R3 -= int(R3)

                    R[n, :] = R1 * nr1 * A[0, :] + R2 * nr2 * A[1, :] + R3 * nr3 * A[2, :]
                    Rfft[i, j, k, :] = R[n, :]
                    idx[i, j, k] = n

        return R, Rfft, idx

    # ==================================================
    @classmethod
    def get_kpoints(cls, nk1, nk2, nk3):
        """
        get reciprocal lattice points (crystal coordinate), k = (k1, k2, k3).
        k = k1*b1 + k2*b2 + k3*b3
        bj: reciprocal lattice vector.

        Args:
            nk1 (int): # of lattice point b1 direction.
            nk2 (int): # of lattice point b2 direction.
            nk3 (int): # of lattice point b3 direction.

        Returns:
            ndarray: lattice points.
        """
        nktot = nk1 * nk2 * nk3

        Kint = np.zeros((nktot, 3), dtype=float)

        for i in range(nk1):
            for j in range(nk2):
                for k in range(nk3):
                    n = k + j * nk3 + i * nk2 * nk3
                    k1 = float(i) / float(nk1)
                    k2 = float(j) / float(nk2)
                    k3 = float(k) / float(nk3)
                    if k1 >= 0.5:
                        k1 = k1 - 1.0
                    if k2 >= 0.5:
                        k2 = k2 - 1.0
                    if k3 >= 0.5:
                        k3 = k3 - 1.0
                    k1 -= int(k1)
                    k2 -= int(k2)
                    k3 -= int(k3)

                    Kint[n] = k1, k2, k3

        return Kint

    # ==================================================
    @classmethod
    def kpoints_to_rpoints(cls, kpoints):
        """
        get lattice points corresponding to reciprocal lattice points.

        Args:
            kpoints (ndarray): reciprocal lattice points (crystal coordinate).
            nk3 (int): # of lattice point b3 direction.

        Returns:
            ndarray: reciprocal lattice points (crystal coordinate).
        """
        kpoints = np.array(kpoints, dtype=float)
        N1 = len(sorted(set(list(kpoints[:, 0]))))
        N2 = len(sorted(set(list(kpoints[:, 1]))))
        N3 = len(sorted(set(list(kpoints[:, 2]))))
        N1 = N1 - 1 if N1 % 2 == 0 else N1
        N2 = N2 - 1 if N2 % 2 == 0 else N2
        N3 = N3 - 1 if N3 % 2 == 0 else N3
        rpoints, _, _ = SymCW.get_rpoints(N1, N2, N3)

        return rpoints

    # ==================================================
    @classmethod
    def fourier_transform_k_to_r(cls, Ok, kpoints, rpoints=None, atoms_positions=None):
        """
        inverse fourier transformation of an arbitrary operator from k-space representation into real-space representation.

        Args:
            Ok (ndarray): arbitrary operator in k-space representation, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
            kpoints (ndarray): reciprocal lattice points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
            rpoints (ndarray, optional): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
            atoms_positions (ndarray, optional): atom's position in fractional coordinates.

        Returns:
            ndarray: real-space representation of the given operator, O_{ab}(R) = <φ_{a}(R)|O|φ_{b}(0)>.
        """
        # lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
        if rpoints is None:
            rpoints = SymCW.kpoints_to_rpoints(kpoints)

        Ok = np.array(Ok, dtype=complex)
        kpoints = np.array(kpoints, dtype=float)
        rpoints = np.array(rpoints, dtype=float)

        # number of k points
        num_k = kpoints.shape[0]
        # number of lattice points
        Nr = rpoints.shape[0]
        # number of pseudo atomic orbitalså
        num_wann = Ok.shape[1]

        if atoms_positions is not None:
            ap = np.array(atoms_positions)
            phase_ab = np.exp(
                [
                    [1.0j * (2 * np.pi * kpoints @ (ap[a, :] - ap[b, :]).transpose()) for b in range(num_wann)]
                    for a in range(num_wann)
                ]
            ).transpose(2, 0, 1)
            Ok = Ok * phase_ab

        phase = np.exp(1.0j * 2 * np.pi * kpoints @ rpoints.T)
        Or = np.array([np.sum(Ok[:, :, :] * phase[:, r, np.newaxis, np.newaxis], axis=0) for r in range(Nr)])
        Or /= num_k

        rpoints = np.array([[round(N1), round(N2), round(N3)] for N1, N2, N3 in rpoints], dtype=int)

        return Or, rpoints

    # ==================================================
    @classmethod
    def fourier_transform_r_to_k(cls, Or, rpoints, kpoints, atoms_positions=None):
        """
        fourier transformation of an arbitrary operator from real-space representation into k-space representation.

        Args:
            Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(R)|O|φ_{b}(0)>.
            rpoints (ndarray): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
            kpoints (ndarray): reciprocal lattice points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
            atoms_positions (ndarray, optional): atom's position in fractional coordinates.

        Returns:
            ndarray: k-space representation of the given operator, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
        """
        Or = np.array(Or, dtype=complex)
        rpoints = np.array(rpoints, dtype=float)
        kpoints = np.array(kpoints, dtype=float)

        # number of k points
        num_k = kpoints.shape[0]

        # number of pseudo atomic orbitalså
        num_wann = Or.shape[1]

        phase = np.exp(-1.0j * 2 * np.pi * kpoints @ rpoints.T)
        Ok = np.array([np.sum(Or[:, :, :] * phase[k, :, np.newaxis, np.newaxis], axis=0) for k in range(num_k)])

        if atoms_positions is not None:
            ap = np.array(atoms_positions)
            phase_ab = np.exp(
                [
                    [1.0j * (-2 * np.pi * kpoints @ (ap[a, :] - ap[b, :]).transpose()) for b in range(num_wann)]
                    for a in range(num_wann)
                ]
            ).transpose(2, 0, 1)
            Ok = Ok * phase_ab

        return Ok, kpoints

    # ==================================================
    @classmethod
    def interpolate(cls, Ok, kpoints_0, kpoints, rpoints=None, atoms_positions=None):
        """
        interpolate an arbitrary operator by implementing
        fourier transformation from real-space representation into k-space representation.

        Args:
            Ok (ndarray): arbitrary operator in k-space representation, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
            kpoints_0 (ndarray): k points before interpolated (crystal coordinate, [[k1,k2,k3]]).
            kpoints (ndarray): k points after interpolated (crystal coordinate, [[k1,k2,k3]]).
            rpoints (ndarray): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
            atoms_positions (ndarray, optional): atom's position in fractional coordinates.

        Returns:
            ndarray: matrix elements at each k point, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
        """
        Or, rpoints = SymCW.fourier_transform_k_to_r(Ok, kpoints_0, rpoints, atoms_positions)
        Ok_interpolated, _ = SymCW.fourier_transform_r_to_k(Or, rpoints, kpoints, atoms_positions)

        return Ok_interpolated

    # ==================================================
    @classmethod
    def matrix_dict_r(cls, Or, rpoints, diagonal=False):
        """
        dictionary form of an arbitrary operator matrix in real-space representation.

        Args:
            Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(R)|O|φ_{b}(0)>.
            rpoints (ndarray): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
            diagonal (bool, optional): diagonal matrix ?

        Returns:
            dict: real-space representation of the given operator, {(n2,n2,n3,a,b) = O_{ab}(R)}.
        """
        # number of pseudo atomic orbitals
        dim_r = len(Or[0])
        if not diagonal:
            dim_c = len(Or[0][0])

        Or_dict = {}

        r_list = [[r, round(n1), round(n2), round(n3)] for r, (n1, n2, n3) in enumerate(rpoints)]

        if diagonal:
            Or_dict = {(n1, n2, n3, a, a): Or[r][a] for r, n1, n2, n3 in r_list for a in range(dim_r)}
        else:
            Or_dict = {
                (n1, n2, n3, a, b): Or[r][a][b] for r, n1, n2, n3 in r_list for a in range(dim_r) for b in range(dim_c)
            }

        return Or_dict

    # ==================================================
    @classmethod
    def matrix_dict_k(cls, Ok, kpoints, diagonal=False):
        """
        dictionary form of an arbitrary operator matrix in k-space representation.

        Args:
            Ok (ndarray): arbitrary operator in k-space representation, O_{ab}(k) = <φ_{a}(k)|H|φ_{b}(k)>.
            kpoints (ndarray): reciprocal lattice points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
            diagonal (bool, optional): diagonal matrix ?

        Returns:
            dict: k-space representation of the given operator, {(k2,k2,k3,a,b) = O_{ab}(k)}.
        """
        # number of pseudo atomic orbitals
        dim_r = len(Ok[0])
        if not diagonal:
            dim_c = len(Ok[0][0])

        k_list = [[k, k1, k2, k3] for k, (k1, k2, k3) in enumerate(kpoints)]

        if diagonal:
            Ok_dict = {(k1, k2, k3, a, a): Ok[k][a] for k, k1, k2, k3 in k_list for a in range(dim_r)}
        else:
            Ok_dict = {
                (k1, k2, k3, a, b): Ok[k][a][b] for k, k1, k2, k3 in k_list for a in range(dim_r) for b in range(dim_c)
            }

        return Ok_dict

    # ==================================================
    @classmethod
    def dict_to_matrix(cls, dic):
        """
        convert dictionary form to matrix form of an arbitrary operator matrix.

        Args:
            dic (dict): dictionary form of an arbitrary operator matrix in reak-space/k-space representation.

        Returns:
            ndarray: matrix form of the given operator.
        """
        dim = max([a for (_, _, _, a, _) in dic.keys()]) + 1

        O_mat = [np.zeros((dim, dim), dtype=complex)]
        idx = 0
        g0 = list(dic.keys())[0][:3]

        for (g1, g2, g3, a, b), v in dic.items():
            g = (g1, g2, g3)
            if g != g0:
                O_mat.append(np.zeros((dim, dim), dtype=complex))
                idx += 1
                g0 = g

            O_mat[idx][a, b] = complex(v)

        return np.array(O_mat)

    # ==================================================
    @classmethod
    def samb_decomp(cls, Or_dict, Zr_dict):
        z = {
            k: np.real(np.sum([v * Or_dict.get((-k[0], -k[1], -k[2], k[4], k[3]), 0) for k, v in d.items()]))
            for k, d in Zr_dict.items()
        }

        return z

    # ==================================================
    @classmethod
    def _info_header(cls):
        return info_header_str

    # ==================================================
    @classmethod
    def _data_header(cls):
        return data_header_str

    # ==================================================
    @classmethod
    def _kpoints_header(cls):
        return kpoints_header_str

    # ==================================================
    @classmethod
    def _rpoints_header(cls):
        return rpoints_header_str

    # ==================================================
    @classmethod
    def _hk_header(cls):
        return hk_header_str

    # ==================================================
    @classmethod
    def _sk_header(cls):
        return sk_header_str

    # ==================================================
    @classmethod
    def _pk_header(cls):
        return pk_header_str

    # ==================================================
    @classmethod
    def _hr_header(cls):
        return hr_header_str

    # ==================================================
    @classmethod
    def _sr_header(cls):
        return sr_header_str

    # ==================================================
    @classmethod
    def _z_header(cls):
        return z_header_str

    # ==================================================
    @classmethod
    def _s_header(cls):
        return s_header_str
