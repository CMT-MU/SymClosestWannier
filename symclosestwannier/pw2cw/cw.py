"""
Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.
"""
import os
import sympy as sp
import numpy as np
from numpy import linalg as npl
from scipy import linalg as spl

from gcoreutils.nsarray import NSArray
from multipie.tag.tag_multipole import TagMultipole

from symclosestwannier.pw2cw.cw_info import CWInfo
from symclosestwannier.pw2cw.cw_manager import CWManager

from symclosestwannier.util.header import (
    info_header,
    data_header,
    kpoints_header,
    rpoints_header,
    hk_header,
    sk_header,
    pk_header,
    hr_header,
    sr_header,
    z_header,
    s_header,
)

from symclosestwannier.util.message import opening_msg, ending_msg, starting_msg, system_msg

from symclosestwannier.util.functions import (
    w_proj,
    get_rpoints,
    get_kpoints,
    kpoints_to_rpoints,
    fourier_transform_k_to_r,
    fourier_transform_r_to_k,
    interpolate,
    matrix_dict_r,
    matrix_dict_k,
    dict_to_matrix,
    samb_decomp,
    construct_Or,
    construct_Ok,
)


# ==================================================
class CW(dict):
    """
    Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.

    Attributes:
        _cwi (SystemInfo): CWInfo.
        _cwm (CWManager): CWManager.
        _outfile (str): output file, seedname.cwout.
    """

    # ==================================================
    def __init__(self, cwi, cwm):
        """
        initialize the class.

        Args:
            cwi (CWInfo): CWInfo.
            cwm (CWManager): CWManager.
        """
        self._cwi = cwi
        self._cwm = cwm
        self._outfile = f"{self._cwi['seedname']}.cwout"

        self._cwm.log(opening_msg(), stamp=None, end="\n", file=self._outfile, mode="w")

        self._cwm.log(system_msg(self._cwi), stamp=None, end="\n", file=self._outfile, mode="a")

        if self._cwi["restart"] == "wannierise":
            self._wannierize()
        else:
            self.update(self._cwm.read(f"{self._cwi['seedname']}_data.py"))

        msg = f"  * total elapsed_time:"
        self._cwm.log(msg, stamp="start", file=self._outfile, mode="a")

        self._cwm.log(ending_msg(), stamp=None, end="\n", file=self._outfile, mode="a")

    # ==================================================
    def _wannierize(self):
        """
        wannierization.

        Returns:
            tuple: Sk, Uk, Hk, Hk_nonortho, Sr, Hr, Hr_nonortho.
                - Sk (ndarray) : Overlap matrix elements in k-space.
                - Uk (ndarray) : Unitary matrix elements in k-space.
                - Hk (ndarray) : Hamiltonian matrix elements in k-space (orthogonal).
                - Hk_nonortho (ndarray) : Hamiltonian matrix elements in k-space (non-orthogonal).
                - Sr (ndarray) : Overlap matrix elements in real-space.
                - Hr (ndarray) : Hamiltonian matrix elements in real-space (orthogonal).
                - Hr_nonortho (ndarray) : Hamiltonian matrix elements in real-space (non-orthogonal).
        """
        self._cwm.log(starting_msg(self._cwi), stamp=None, end="\n", file=self._outfile, mode="a")

        Ek = np.array(self._cwi.eig["Ek"], dtype=float)
        Ak = np.array(self._cwi.amn["Ak"], dtype=complex)
        Pk = np.real(np.diagonal(Ak @ Ak.transpose(0, 2, 1).conjugate(), axis1=1, axis2=2))

        if self._cwi["proj_min"] > 0.0:
            msg = f"   - exluding bands with low projectability (proj_min = {self._cwi['proj_min']}) ... "
            self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
            self._cwm.set_stamp()

            Ek, Ak = self._exclude_bands(Pk, Ek, Ak)

            self._cwm.log("done", file=self._outfile, mode="a")

        if self._cwi["disentangle"]:
            msg = "   - disentanglement ... "
            self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
            self._cwm.set_stamp()

            Ak = self._disentangle(Ek, Ak)

            self._cwm.log("done", file=self._outfile, mode="a")

        msg = "   - constructing TB Hamiltonian ... "
        self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        Sk, Uk, Hk, Hk_nonortho, Sr, Hr, Hr_nonortho = self._construct_tb(Ek, Ak)

        self._cwm.log("done", file=self._outfile, mode="a")

        #####

        if self._cwi["kpoint"] is not None and self._cwi["kpoint_path"] is not None:
            kpoint = {i: NSArray(j, "vector", fmt="value") for i, j in self._cwi["kpoint"].items()}
            kpoint_path = self._cwi["kpoint_path"]
            N1 = self._cwi["N1"]
            B = NSArray(self._cwi["unit_cell_cart"], "matrix", fmt="value").inverse()
            kpoints_path, k_linear, k_dis_pos = NSArray.grid_path(kpoint, kpoint_path, N1, B)
        else:
            kpoints_path, k_linear, k_dis_pos = None, None, None

        self.update(
            {
                "kpoints_path": kpoints_path if kpoints_path is not None else None,
                "k_linear": k_linear.tolist() if k_linear is not None else None,
                "k_dis_pos": k_dis_pos if k_dis_pos is not None else None,
                #
                "Pk": Pk.tolist(),
                "Uk": [u.tolist() for u in Uk],
                "Sk": Sk.tolist(),
                "Hk": Hk.tolist(),
                "Hk_nonortho": Hk_nonortho.tolist(),
                #
                "Sr": Sr.tolist(),
                "Hr": Hr.tolist(),
                "Hr_nonortho": Hr_nonortho.tolist(),
            }
        )

        if self._cwi["symmetrization"]:
            msg = "   - symmetrization ... "
            self._cwm.log(msg, None, end="\n", file=self._outfile, mode="a")

            (
                s,
                z,
                z_nonortho,
                Sk_sym,
                Hk_sym,
                Hk_nonortho_sym,
                Sr_sym,
                Hr_sym,
                Hr_nonortho_sym,
                rpoints_mp,
                Ek_RMSE_grid,
                Ek_RMSE_path,
                matrix_dict,
            ) = self._symmetrize()

            self.update(
                {
                    "s": s,
                    "z": z,
                    "z_nonortho": z_nonortho,
                    #
                    "Sk_sym": Sk_sym,
                    "Hk_sym": Hk_sym,
                    "Hk_nonortho_sym": Hk_nonortho_sym,
                    "Sr_sym": Sr_sym,
                    "Hr_sym": Hr_sym,
                    "Hr_nonortho_sym": Hr_nonortho_sym,
                    #
                    "rpoints_mp": rpoints_mp,
                    #
                    "Ek_RMSE_grid": Ek_RMSE_grid,
                    "Ek_RMSE_path": Ek_RMSE_path,
                    #
                    "matrix_dict": matrix_dict,
                }
            )

    # ==================================================
    def _exclude_bands(self, Pk, Ek, Ak):
        """
        exlude bands with low projectability.

        Args:
            Pk (ndarray): projectability of each Kohn-Sham state in k-space.
            Ek (ndarray): Kohn-Sham energies.
            Ak (ndarray): Overlap matrix elements.

        Returns:
            tuple: Ek, Ak.
        """
        # band index for projection
        proj_band_idx = [
            [n for n in range(self._cwi["num_bands"]) if Pk[k][n] > self._cwi["proj_min"]]
            for k in range(self._cwi["num_k"])
        ]

        for k in range(self._cwi["num_k"]):
            if len(proj_band_idx[k]) < self._cwi["num_wann"]:
                raise Exception(f"proj_min = {self._cwi['proj_min']} is too large or PAOs are inappropriate.")

        # eliminate bands with low projectability
        Ek = [Ek[k, proj_band_idx[k]] for k in range(self._cwi["num_k"])]
        Ak = [Ak[k, proj_band_idx[k], :] for k in range(self._cwi["num_k"])]

        return Ek, Ak

    # ==================================================
    def _disentangle(self, Ek, Ak):
        """
        disentangle bands.

        Args:
            Ek (ndarray): Kohn-Sham energies.
            Ak (ndarray): Overlap matrix elements.

        Returns:
            ndarray: Ak.
        """
        Ak = [
            np.array(
                w_proj(
                    Ek[k],
                    self._cwi["dis_win_emin"],
                    self._cwi["dis_win_emax"],
                    self._cwi["smearing_temp_min"],
                    self._cwi["smearing_temp_max"],
                    self._cwi["delta"],
                )[:, np.newaxis]
                * Ak[k]
            )
            for k in range(self._cwi["num_k"])
        ]

        return Ak

    # ==================================================
    def _construct_tb(self, Ek, Ak):
        """
        construct CW TB Hamiltonian.

        Args:
            Ek (ndarray): Kohn-Sham energies.
            Ak (ndarray): Overlap matrix elements.

        Returns:
            tuple: Sk, Uk, Hk, Hk_nonortho, Sr, Hr, Hr_nonortho.
                - Sk (ndarray) : Overlap matrix elements in k-space.
                - Uk (ndarray) : Unitary matrix elements in k-space.
                - Hk (ndarray) : Hamiltonian matrix elements in k-space (orthogonal).
                - Hk_nonortho (ndarray) : Hamiltonian matrix elements in k-space (non-orthogonal).
                - Sr (ndarray) : Overlap matrix elements in real-space.
                - Hr (ndarray) : Hamiltonian matrix elements in real-space (orthogonal).
                - Hr_nonortho (ndarray) : Hamiltonian matrix elements in real-space (non-orthogonal).
        """
        Sk = np.array([Ak[k].transpose().conjugate() @ Ak[k] for k in range(self._cwi["num_k"])])

        if self._cwi["svd"]:  # orthogonalize PAOs by singular value decomposition (SVD)

            def U_mat(k):
                u, _, vd = np.linalg.svd(Ak[k], full_matrices=False)
                return u @ vd

            Uk = [U_mat(k) for k in range(self._cwi["num_k"])]

        else:  # orthogonalize PAOs by Lowdin's method
            S2k_inv = np.array([npl.inv(spl.sqrtm(Sk[k])) for k in range(self._cwi["num_k"])])
            Uk = [Ak[k] @ S2k_inv[k] for k in range(self._cwi["num_k"])]

        # projection from KS energies to PAOs Hamiltonian
        diag_Ek = [np.diag(Ek[k]) for k in range(self._cwi["num_k"])]
        Hk = np.array([Uk[k].transpose().conjugate() @ diag_Ek[k] @ Uk[k] for k in range(self._cwi["num_k"])])

        S2k = np.array([spl.sqrtm(Sk[k]) for k in range(self._cwi["num_k"])])
        Hk_nonortho = S2k @ Hk @ S2k

        Sr = CW.fourier_transform_k_to_r(Sk, self._cwi["kpoints"])[0]
        Hr = CW.fourier_transform_k_to_r(Hk, self._cwi["kpoints"])[0]
        Hr_nonortho = CW.fourier_transform_k_to_r(Hk_nonortho, self._cwi["kpoints"])[0]

        return Sk, Uk, Hk, Hk_nonortho, Sr, Hr, Hr_nonortho

    # ==================================================
    def _symmetrize(self):
        """
        symmetrize CW TB Hamiltonian.

        Returns:
            tuple:
        """
        Hk = np.array(self["Hk"])
        Hr_dict = CW.matrix_dict_r(self["Hr"], self._cwi["rpoints"])
        Sr_dict = CW.matrix_dict_r(self["Sr"], self._cwi["rpoints"])
        Hr_nonortho_dict = CW.matrix_dict_r(self["Hr_nonortho"], self._cwi["rpoints"])

        #####

        msg = "    - reading output of multipie ... "
        self._cwm.log(msg, None, end="\n", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        model = self._cwm.read(
            os.path.join(self._cwi["mp_outdir"], "{}".format(f"{self._cwi['mp_seedname']}_model.py"))
        )
        samb = self._cwm.read(os.path.join(self._cwi["mp_outdir"], "{}".format(f"{self._cwi['mp_seedname']}_samb.py")))

        try:
            mat = self._cwm.read(
                os.path.join(self._cwi["mp_outdir"], "{}".format(f"{self._cwi['mp_seedname']}_matrix.pkl"))
            )
        except:
            mat = self._cwm.read(
                os.path.join(self._cwi["mp_outdir"], "{}".format(f"{self._cwi['mp_seedname']}_matrix.py"))
            )

        ket_samb = model["info"]["ket"]
        ket_amn = self._cwi["ket_amn"]

        # sort orbitals
        if ket_amn is not None:
            idx_list = [ket_amn.index(o) for o in ket_samb]
            Hk = Hk[:, idx_list, :]
            Hk = Hk[:, :, idx_list]

            idx_list = [ket_samb.index(o) for o in ket_amn]
            Hr_dict = {(n1, n2, n3, idx_list[a], idx_list[b]): v for (n1, n2, n3, a, b), v in Hr_dict.items()}
            Sr_dict = {(n1, n2, n3, idx_list[a], idx_list[b]): v for (n1, n2, n3, a, b), v in Sr_dict.items()}

            Hr_nonortho_dict = {
                (n1, n2, n3, idx_list[a], idx_list[b]): v for (n1, n2, n3, a, b), v in Hr_nonortho_dict.items()
            }

        if self._cwi["irreps"] == "all":
            irreps = model["info"]["generate"]["irrep"]
        elif self._cwi["irreps"] == "full":
            irreps = [model["info"]["generate"]["irrep"][0]]
        else:
            irreps = self._cwi["irreps"]

        for zj, (tag, _) in samb["data"]["Z"].items():
            if TagMultipole(tag).irrep not in irreps:
                del mat["matrix"][zj]

        tag_dict = {zj: tag for zj, (tag, _) in samb["data"]["Z"].items()}
        Zr_dict = {
            (zj, tag_dict[zj]): {tuple(sp.sympify(k)): complex(sp.sympify(v)) for k, v in d.items()}
            for zj, d in mat["matrix"].items()
        }
        mat["matrix"] = {
            zj: {tuple(sp.sympify(k)): complex(sp.sympify(v)) for k, v in d.items()} for zj, d in mat["matrix"].items()
        }

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

        #####

        msg = "    - decomposing Hamiltonian as linear combination of SAMBs ... "
        self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        z = CW.samb_decomp(Hr_dict, Zr_dict)

        self._cwm.log("done", file=self._outfile, mode="a")

        #####

        msg = "    - decomposing overlap as linear combination of SAMBs ... "
        self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        s = CW.samb_decomp(Sr_dict, Zr_dict)

        self._cwm.log("done", file=self._outfile, mode="a")

        #####

        msg = "    - decomposing non-orthogonal Hamiltonian as linear combination of SAMBs ... "
        self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        z_nonortho = CW.samb_decomp(Hr_nonortho_dict, Zr_dict)

        self._cwm.log("done", file=self._outfile, mode="a")

        #####

        msg = "    - constructing symmetrized TB Hamiltonian ... "
        self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        rpoints_mp = [(n1, n2, n3) for Zj_dict in Zr_dict.values() for (n1, n2, n3, _, _) in Zj_dict.keys()]
        rpoints_mp = sorted(list(set(rpoints_mp)), key=rpoints_mp.index)

        Sr_sym = CW.construct_Or(list(s.values()), self._cwi["num_wann"], rpoints_mp, mat)
        Hr_sym = CW.construct_Or(list(z.values()), self._cwi["num_wann"], rpoints_mp, mat)
        Hr_nonortho_sym = CW.construct_Or(list(z_nonortho.values()), self._cwi["num_wann"], rpoints_mp, mat)

        atoms_frac = [self._cwi["atom_pos_r"][i] for i in self._cwi["nw2n"]]
        Sk_sym = CW.fourier_transform_r_to_k(Sr_sym, rpoints_mp, self._cwi["kpoints"], atoms_frac)[0]
        Hk_sym = CW.fourier_transform_r_to_k(Hr_sym, rpoints_mp, self._cwi["kpoints"], atoms_frac)[0]
        Hk_nonortho_sym = CW.fourier_transform_r_to_k(Hr_nonortho_sym, rpoints_mp, self._cwi["kpoints"], atoms_frac)[0]

        self._cwm.log("done", file=self._outfile, mode="a")

        #####

        msg = "    - evaluating fitting accuracy ... "
        self._cwm.log(msg, None, end="\n", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        Ek_grid, _ = np.linalg.eigh(Hk)
        Ek_grid_sym, _ = np.linalg.eigh(Hk_sym)

        num_k, num_wann = Ek_grid_sym.shape
        Ek_RMSE_grid = np.sum(np.abs(Ek_grid_sym - Ek_grid)) / num_k / num_wann * 1000  # [meV]

        msg = f"     * RMSE of eigen values between CW and Symmetry-Adapted CW models (grid) = {'{:.4f}'.format(Ek_RMSE_grid)} [meV]"
        self._cwm.log(msg, None, end="\n", file=self._outfile, mode="a")

        #####

        if not mat["molecule"]:
            Hk_path = CW.fourier_transform_r_to_k(self["Hr"], self._cwi["rpoints"], self["kpoints_path"])[0]
            Ek_path, _ = np.linalg.eigh(Hk_path)

            Hk_sym_path = CW.fourier_transform_r_to_k(Hr_sym, rpoints_mp, self["kpoints_path"], atoms_frac)[0]
            Ek_path_sym, _ = np.linalg.eigh(Hk_sym_path)

            num_k, num_wann = Ek_path_sym.shape
            Ek_RMSE_path = np.sum(np.abs(Ek_path_sym - Ek_path)) / num_k / num_wann * 1000  # [meV]

            msg = f"     * RMSE of eigen values between CW and Symmetry-Adapted CW models (path) = {'{:.4f}'.format(Ek_RMSE_path)} [meV]"
            self._cwm.log(msg, None, end="\n", file=self._outfile, mode="a")
        else:
            Ek_RMSE_path = None

        return (
            s,
            z,
            z_nonortho,
            Sk_sym,
            Hk_sym,
            Hk_nonortho_sym,
            Sr_sym,
            Hr_sym,
            Hr_nonortho_sym,
            rpoints_mp,
            Ek_RMSE_grid,
            Ek_RMSE_path,
            mat,
        )

    # ==================================================
    def write_or(self, Or, rpoints, filename, header=None):
        """
        write seedname_or.dat.

        Args:

        """
        Hr_dict = CW.matrix_dict_r(Or, rpoints)
        Hr_str = "".join(
            [
                f"{n1}  {n2}  {n3}  {a}  {b}  {'{:.8f}'.format(np.real(v))}  {'{:.8f}'.format(np.imag(v))}\n"
                for (n1, n2, n3, a, b), v in Hr_dict.items()
            ]
        )
        self._cwm.write(filename, Hr_str, header, None)

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
        return get_rpoints(nr1, nr2, nr3, unit_cell_cart)

    # ==================================================
    @classmethod
    def get_kpoints(cls, nk1, nk2, nk3):
        """
        get k-points (crystal coordinate), k = (k1, k2, k3).
        k = k1*b1 + k2*b2 + k3*b3
        bj: reciprocal lattice vector.

        Args:
            nk1 (int): # of lattice point b1 direction.
            nk2 (int): # of lattice point b2 direction.
            nk3 (int): # of lattice point b3 direction.

        Returns:
            ndarray: lattice points.
        """
        return get_kpoints(nk1, nk2, nk3)

    # ==================================================
    @classmethod
    def kpoints_to_rpoints(cls, kpoints):
        """
        get lattice points corresponding to k-points.

        Args:
            kpoints (ndarray): k-points (crystal coordinate).
            nk3 (int): # of lattice point b3 direction.

        Returns:
            ndarray: k-points (crystal coordinate).
        """
        return kpoints_to_rpoints(kpoints)

    # ==================================================
    @classmethod
    def fourier_transform_k_to_r(cls, Ok, kpoints, rpoints=None, atoms_frac=None):
        """
        inverse fourier transformation of an arbitrary operator from k-space representation into real-space representation.

        Args:
            Ok (ndarray): arbitrary operator in k-space representation, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
            kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
            rpoints (ndarray, optional): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
            atoms_frac (ndarray, optional): atom's position in fractional coordinates.

        Returns:
            (ndarray, ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(R)|O|φ_{b}(0)>, lattice points.
        """
        return fourier_transform_k_to_r(Ok, kpoints, rpoints, atoms_frac)

    # ==================================================
    @classmethod
    def fourier_transform_r_to_k(cls, Or, rpoints, kpoints, atoms_frac=None):
        """
        fourier transformation of an arbitrary operator from real-space representation into k-space representation.

        Args:
            Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(R)|O|φ_{b}(0)>.
            rpoints (ndarray): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
            kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
            atoms_frac (ndarray, optional): atom's position in fractional coordinates.

        Returns:
            ndarray: k-space representation of the given operator, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
        """
        return fourier_transform_r_to_k(Or, rpoints, kpoints, atoms_frac)

    # ==================================================
    @classmethod
    def interpolate(cls, Ok, kpoints_0, kpoints, rpoints=None, atoms_frac=None):
        """
        interpolate an arbitrary operator by implementing
        fourier transformation from real-space representation into k-space representation.

        Args:
            Ok (ndarray): arbitrary operator in k-space representation, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
            kpoints_0 (ndarray): k points before interpolated (crystal coordinate, [[k1,k2,k3]]).
            kpoints (ndarray): k points after interpolated (crystal coordinate, [[k1,k2,k3]]).
            rpoints (ndarray): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
            atoms_frac (ndarray, optional): atom's position in fractional coordinates.

        Returns:
            ndarray: matrix elements at each k point, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
        """
        return interpolate(Ok, kpoints_0, kpoints, rpoints, atoms_frac)

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
        return matrix_dict_r(Or, rpoints, diagonal)

    # ==================================================
    @classmethod
    def matrix_dict_k(cls, Ok, kpoints, diagonal=False):
        """
        dictionary form of an arbitrary operator matrix in k-space representation.

        Args:
            Ok (ndarray): arbitrary operator in k-space representation, O_{ab}(k) = <φ_{a}(k)|H|φ_{b}(k)>.
            kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
            diagonal (bool, optional): diagonal matrix ?

        Returns:
            dict: k-space representation of the given operator, {(k2,k2,k3,a,b) = O_{ab}(k)}.
        """
        return matrix_dict_k(Ok, kpoints, diagonal)

    # ==================================================
    @classmethod
    def dict_to_matrix(cls, Or_dict):
        """
        convert dictionary form to matrix form of an arbitrary operator matrix.

        Args:
            dic (dict): dictionary form of an arbitrary operator matrix in reak-space/k-space representation.

        Returns:
            ndarray: matrix form of the given operator.
        """
        return dict_to_matrix(Or_dict)

    # ==================================================
    @classmethod
    def samb_decomp(cls, Or_dict, Zr_dict):
        """
        decompose arbitrary operator into linear combination of SAMBs.

        Args:
            Or_dict (dict): dictionary form of an arbitrary operator matrix in reak-space/k-space representation.
            Zr_dict (dict): SAMBs

        Returns:
            z (list): parameter set, [z_j].
        """
        return samb_decomp(Or_dict, Zr_dict)

    # ==================================================
    @classmethod
    def construct_Or(cls, z, num_wann, rpoints, matrix_dict):
        """
        arbitrary operator constructed by linear combination of SAMBs in real space representation.

        Args:
            z (list): parameter set, [z_j].
            num_wann (int): # of CWFs.
            rpoints (ndarray, optional): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
            matrix_dict (dict): SAMBs.

        Returns:
            ndarray: matrix, [#r, dim, dim].
        """
        return construct_Or(z, num_wann, rpoints, matrix_dict)

    # ==================================================
    @classmethod
    def construct_Ok(cls, z, num_wann, kpoints, rpoints, matrix_dict):
        """
        arbitrary operator constructed by linear combination of SAMBs in k space representation.

        Args:
            z (list): parameter set, [z_j].
            num_wann (int): # of CWFs.
            kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
            rpoints (ndarray, optional): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
            matrix_dict (dict): SAMBs.

        Returns:
            ndarray: matrix, [#k, dim, dim].
        """
        return construct_Ok(z, num_wann, kpoints, rpoints, matrix_dict)

    # ==================================================
    @property
    def cwin(self):
        self._cwm.cwin

    # ==================================================
    @property
    def win(self):
        self._cwm.win

    # ==================================================
    @property
    def eig(self):
        self._cwm.eig

    # ==================================================
    @property
    def amn(self):
        self._cwm.amn

    # ==================================================
    @property
    def mmn(self):
        self._cwm.mmn

    # ==================================================
    @property
    def nnkp(self):
        self._cwm.nnkp

    # ==================================================
    @classmethod
    def _info_header(cls):
        return info_header

    # ==================================================
    @classmethod
    def _data_header(cls):
        return data_header

    # ==================================================
    @classmethod
    def _kpoints_header(cls):
        return kpoints_header

    # ==================================================
    @classmethod
    def _rpoints_header(cls):
        return rpoints_header

    # ==================================================
    @classmethod
    def _hk_header(cls):
        return hk_header

    # ==================================================
    @classmethod
    def _sk_header(cls):
        return sk_header

    # ==================================================
    @classmethod
    def _pk_header(cls):
        return pk_header

    # ==================================================
    @classmethod
    def _hr_header(cls):
        return hr_header

    # ==================================================
    @classmethod
    def _sr_header(cls):
        return sr_header

    # ==================================================
    @classmethod
    def _z_header(cls):
        return z_header

    # ==================================================
    @classmethod
    def _s_header(cls):
        return s_header
