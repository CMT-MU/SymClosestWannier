"""
Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.
"""
import os
import datetime
import itertools
import textwrap

import sympy as sp
import numpy as np
from numpy import linalg as npl
from scipy import linalg as spl

from multipie.tag.tag_multipole import TagMultipole

from symclosestwannier.cw.cw_info import CWInfo
from symclosestwannier.cw.cw_manager import CWManager

from symclosestwannier.util.message import opening_msg, ending_msg, starting_msg, system_msg, starting_msg_w90
from symclosestwannier.util.header import (
    cw_info_header,
    cw_data_header,
    kpoints_header,
    rpoints_header,
    hk_header,
    sk_header,
    pk_header,
    hr_header,
    sr_header,
    z_header,
    z_nonortho_header,
    s_header,
)
from symclosestwannier.util._utility import (
    weight_proj,
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

_default = {
    "Sk": None,
    "Hk": None,
    "Hk_nonortho": None,
    #
    "Sr": None,
    "Hr": None,
    "Hr_nonortho": None,
    #
    "s": None,
    "z": None,
    "z_nonortho": None,
    #
    "Sk_sym": None,
    "Hk_sym": None,
    "Hk_nonortho_sym": None,
    "Sr_sym": None,
    "Hr_sym": None,
    "Hr_nonortho_sym": None,
    #
    "rpoints_mp": None,
    #
    "Ek_RMSE_grid": None,
    "Ek_RMSE_path": None,
    #
    "matrix_dict": None,
}


# ==================================================
class CWModel(dict):
    """
    Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.

    Attributes:
        _cwi (SystemInfo): CWInfo.
        _cwm (CWManager): CWManager.
        _outfile (str): output file, seedname.cwout.
    """

    # ==================================================
    def __init__(self, seedname):
        """
        initialize the class.

        Args:
            seedname (str, optional): seedname.
        """
        super().__init__()

        self.update(_default)

        self._cwi = CWInfo(".", seedname)
        self._cwm = CWManager(
            topdir=self._cwi["outdir"],
            verbose=self._cwi["verbose"],
            parallel=self._cwi["parallel"],
            formatter=self._cwi["formatter"],
        )
        self._outfile = f"{self._cwi['seedname']}.cwout"

        self._cwm.log(opening_msg(), stamp=None, end="\n", file=self._outfile, mode="w")

        self._cwm.log(system_msg(self._cwi), stamp=None, end="\n", file=self._outfile, mode="a")

        if self._cwi["restart"] == "cw":
            self._cw()
        elif self._cwi["restart"] == "w90":
            self._w90()
        else:
            self.update(self._cwm.read(f"{self._cwi['seedname']}_data.py"))

        self._cwm.log(f"  * total elapsed_time:", stamp="start", file=self._outfile, mode="a")

        self._cwm.log(ending_msg(), stamp=None, end="\n", file=self._outfile, mode="a")

    # ==================================================
    def _w90(self):
        """
        construct Wannier TB model by using wannier90 outputs.
        """
        self._cwm.log(starting_msg_w90(self._cwi["seedname"]), stamp=None, end="\n", file=self._outfile, mode="a")

        Ek = np.array(self._cwi["Ek"], dtype=float)
        Ak = np.array(self._cwi["Ak"], dtype=complex)
        Pk = np.real(np.diagonal(Ak @ Ak.transpose(0, 2, 1).conjugate(), axis1=1, axis2=2))
        Uk = np.array(self._cwi["Uk"], dtype=complex)
        Sk = Ak.transpose(0, 2, 1).conjugate() @ Ak
        Sk = 0.5 * (Sk + np.einsum("kmn->knm", Sk).conj())

        diag_Ek = np.array([np.diag(Ek[k]) for k in range(self._cwi["num_k"])])
        Hk = Uk.transpose(0, 2, 1).conjugate() @ diag_Ek @ Uk
        Hk = 0.5 * (Hk + np.einsum("kmn->knm", Hk).conj())

        S2k = np.array([spl.sqrtm(Sk[k]) for k in range(self._cwi["num_k"])])
        Hk_nonortho = S2k @ Hk @ S2k

        Sr = CWModel.fourier_transform_k_to_r(Sk, self._cwi["kpoints"], self._cwi["irvec"])
        Hr = CWModel.fourier_transform_k_to_r(Hk, self._cwi["kpoints"], self._cwi["irvec"])
        Hr_nonortho = CWModel.fourier_transform_k_to_r(Hk_nonortho, self._cwi["kpoints"], self._cwi["irvec"])

        #####

        self.update(
            {
                "Sk": Sk.tolist(),
                "Hk": Hk.tolist(),
                "Hk_nonortho": Hk_nonortho.tolist(),
                #
                "Sr": Sr.tolist(),
                "Hr": Hr.tolist(),
                "Hr_nonortho": Hr_nonortho.tolist(),
            }
        )

    # ==================================================
    def _cw(self):
        """
        construct CW TB model.
        """
        self._cwm.log(starting_msg(self._cwi["seedname"]), stamp=None, end="\n", file=self._outfile, mode="a")

        Ek = np.array(self._cwi["Ek"], dtype=float)
        Ak = np.array(self._cwi["Ak"], dtype=complex)
        Pk = np.real(np.diagonal(Ak @ Ak.transpose(0, 2, 1).conjugate(), axis1=1, axis2=2))

        if self._cwi["proj_min"] > 0.0:
            msg = f"   - exluding bands with low projectability (proj_min = {self._cwi['proj_min']}) ... "
            self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
            self._cwm.set_stamp()

            Ak = self._exclude_bands(Pk, Ak)

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

        self._cwi["Ak"] = Ak
        self._cwi["Uoptk"] = np.array([np.identity(self._cwi["num_wann"], dtype=complex)] * self._cwi["num_k"])
        self._cwi["Udisk"] = Uk
        self._cwi["Uk"] = Uk

        self.update(
            {
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
    def _exclude_bands(self, Pk, Ak):
        """
        exlude bands by setting the matrix elements of Ak with low projectability to zero.

        Args:
            Pk (ndarray): projectability of each Kohn-Sham state in k-space.
            Ak (ndarray): Overlap matrix elements.

        Returns:
            ndarray: Ak.
        """
        num_k = self._cwi["num_k"]
        proj_min = self._cwi["proj_min"]
        exclude_bands_idx = np.array([np.sum(Pk[:, n]) / num_k < proj_min for n in range(self._cwi["num_bands"])])

        if np.sum(exclude_bands_idx) < self._cwi["num_wann"]:
            raise Exception(f"proj_min = {proj_min} is too large or selected orbitals are inappropriate.")

        Ak[:, exclude_bands_idx, :] = 0.0

        return Ak

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
        w = weight_proj(
            Ek,
            self._cwi["dis_win_emin"],
            self._cwi["dis_win_emax"],
            self._cwi["smearing_temp_min"],
            self._cwi["smearing_temp_max"],
            self._cwi["delta"],
        )

        Ak = w[:, :, np.newaxis] * Ak

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
        Sk = Ak.transpose(0, 2, 1).conjugate() @ Ak
        Sk = 0.5 * (Sk + np.einsum("kmn->knm", Sk).conj())

        if self._cwi["svd"]:  # orthogonalize orbitals by singular value decomposition (SVD)

            def U_mat(k):
                u, _, vd = np.linalg.svd(Ak[k], full_matrices=False)
                return u @ vd

            Uk = np.array([U_mat(k) for k in range(self._cwi["num_k"])])

        else:  # orthogonalize orbitals by Lowdin's method
            S2k_inv = np.array([npl.inv(spl.sqrtm(Sk[k])) for k in range(self._cwi["num_k"])])
            Uk = Ak @ S2k_inv

        # projection from KS energies to Closest Wannnier Hamiltonian
        diag_Ek = np.array([np.diag(Ek[k]) for k in range(self._cwi["num_k"])])
        Hk = Uk.transpose(0, 2, 1).conjugate() @ diag_Ek @ Uk
        Hk = 0.5 * (Hk + np.einsum("kmn->knm", Hk).conj())

        S2k = np.array([spl.sqrtm(Sk[k]) for k in range(self._cwi["num_k"])])
        Hk_nonortho = S2k @ Hk @ S2k

        Sr = CWModel.fourier_transform_k_to_r(Sk, self._cwi["kpoints"], self._cwi["irvec"])
        Hr = CWModel.fourier_transform_k_to_r(Hk, self._cwi["kpoints"], self._cwi["irvec"])
        Hr_nonortho = CWModel.fourier_transform_k_to_r(Hk_nonortho, self._cwi["kpoints"], self._cwi["irvec"])

        return Sk, Uk, Hk, Hk_nonortho, Sr, Hr, Hr_nonortho

    # ==================================================
    def _symmetrize(self):
        """
        symmetrize CW TB Hamiltonian.

        Returns:
            tuple:
        """
        Hk = np.array(self["Hk"])
        Hr_dict = CWModel.matrix_dict_r(self["Hr"], self._cwi["irvec"])
        Sr_dict = CWModel.matrix_dict_r(self["Sr"], self._cwi["irvec"])
        Hr_nonortho_dict = CWModel.matrix_dict_r(self["Hr_nonortho"], self._cwi["irvec"])

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

        z = CWModel.samb_decomp(Hr_dict, Zr_dict)

        self._cwm.log("done", file=self._outfile, mode="a")

        #####

        msg = "    - decomposing overlap as linear combination of SAMBs ... "
        self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        s = CWModel.samb_decomp(Sr_dict, Zr_dict)

        self._cwm.log("done", file=self._outfile, mode="a")

        #####

        msg = "    - decomposing non-orthogonal Hamiltonian as linear combination of SAMBs ... "
        self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        z_nonortho = CWModel.samb_decomp(Hr_nonortho_dict, Zr_dict)

        self._cwm.log("done", file=self._outfile, mode="a")

        #####

        msg = "    - constructing symmetrized TB Hamiltonian ... "
        self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        rpoints_mp = [(n1, n2, n3) for Zj_dict in Zr_dict.values() for (n1, n2, n3, _, _) in Zj_dict.keys()]
        rpoints_mp = sorted(list(set(rpoints_mp)), key=rpoints_mp.index)

        Sr_sym = CWModel.construct_Or(list(s.values()), self._cwi["num_wann"], rpoints_mp, mat)
        Hr_sym = CWModel.construct_Or(list(z.values()), self._cwi["num_wann"], rpoints_mp, mat)
        Hr_nonortho_sym = CWModel.construct_Or(list(z_nonortho.values()), self._cwi["num_wann"], rpoints_mp, mat)

        atoms_frac = [self._cwi["atom_pos_r"][i] for i in self._cwi["nw2n"]]
        Sk_sym = CWModel.fourier_transform_r_to_k(Sr_sym, self._cwi["kpoints"], rpoints_mp, atoms_frac)
        Hk_sym = CWModel.fourier_transform_r_to_k(Hr_sym, self._cwi["kpoints"], rpoints_mp, atoms_frac)
        Hk_nonortho_sym = CWModel.fourier_transform_r_to_k(
            Hr_nonortho_sym, self._cwi["kpoints"], rpoints_mp, atoms_frac
        )

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
            Hk_path = CWModel.fourier_transform_r_to_k(
                self["Hr"], self._cwi["kpoints_path"], self._cwi["irvec"], self._cwi["ndegen"]
            )
            Ek_path, _ = np.linalg.eigh(Hk_path)

            Hk_sym_path = CWModel.fourier_transform_r_to_k(Hr_sym, self._cwi["kpoints_path"], rpoints_mp, atoms_frac)
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
    @classmethod
    def fourier_transform_k_to_r(cls, Ok, kpoints, irvec, atoms_frac=None):
        """
        inverse fourier transformation of an arbitrary operator from k-space representation into real-space representation.

        Args:
            Ok (ndarray): arbitrary operator in k-space representation, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
            kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
            irvec (ndarray, optional): irreducible R vectors (crystal coordinate, [[n1,n2,n3]], nj: integer).
            atoms_frac (ndarray, optional): atom's position in fractional coordinates.

        Returns:
            (ndarray, ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(R)|O|φ_{b}(0)>, lattice points.
        """
        return fourier_transform_k_to_r(Ok, kpoints, irvec, atoms_frac)

    # ==================================================
    @classmethod
    def fourier_transform_r_to_k(cls, Or, kpoints, irvec, ndegen=None, atoms_frac=None):
        """
        fourier transformation of an arbitrary operator from real-space representation into k-space representation.

        Args:
        Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(R)|O|φ_{b}(0)>.
        kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
        irvec (ndarray): irreducible R vectors (crystal coordinate, [[n1,n2,n3]], nj: integer).
        ndegen (ndarray, optional): number of degeneracy at each R.
        atoms_frac (ndarray, optional): atom's position in fractional coordinates.

        Returns:
            ndarray: k-space representation of the given operator, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
        """
        return fourier_transform_r_to_k(Or, kpoints, irvec, ndegen, atoms_frac)

    # ==================================================
    @classmethod
    def interpolate(cls, Ok, kpoints_0, kpoints, irvec, ndegen=None, atoms_frac=None):
        """
        interpolate an arbitrary operator by implementing
        fourier transformation from real-space representation into k-space representation.

        Args:
            Ok (ndarray): arbitrary operator in k-space representation, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
            kpoints_0 (ndarray): k points before interpolated (crystal coordinate, [[k1,k2,k3]]).
            kpoints (ndarray): k points after interpolated (crystal coordinate, [[k1,k2,k3]]).
            irvec (ndarray): irreducible R vectors (crystal coordinate, [[n1,n2,n3]], nj: integer).
            ndegen (ndarray, optional): number of degeneracy at each R.
            atoms_frac (ndarray, optional): atom's position in fractional coordinates.

        Returns:
            ndarray: matrix elements at each k point, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
        """
        return interpolate(Ok, kpoints_0, kpoints, irvec, ndegen, atoms_frac)

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
        arbitrary operator constructed by linear combination of SAMBs in real-space representation.

        Args:
            z (list): parameter set, [z_j].
            num_wann (int): # of WFs.
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
        arbitrary operator constructed by linear combination of SAMBs in k-space representation.

        Args:
            z (list): parameter set, [z_j].
            num_wann (int): # of WFs.
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
        self._cwi.cwin

    # ==================================================
    @property
    def win(self):
        self._cwi.win

    # ==================================================
    @property
    def eig(self):
        self._cwi.eig

    # ==================================================
    @property
    def amn(self):
        self._cwi.amn

    # ==================================================
    @property
    def mmn(self):
        self._cwi.mmn

    # ==================================================
    @property
    def nnkp(self):
        self._cwi.nnkp

    # ==================================================
    @classmethod
    def _cw_info_header(cls):
        return cw_info_header

    # ==================================================
    @classmethod
    def _cw_data_header(cls):
        return cw_data_header

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

    # ==================================================
    def write_or(self, Or, filename, header=None, vec=False):
        """
        write seedname_or.dat.

        Args:
            Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(R)|O|φ_{b}(0)>.
            filename (str): file name.
            header (str, optional): header.
            vec (bool, optional): vector ?
        """
        irvec = self._cwi["irvec"]
        ndegen = self._cwi["ndegen"]
        num_wann = self._cwi["num_wann"]
        unit_cell_cart = np.array(self._cwi["unit_cell_cart"])
        Or = np.array(Or)

        t = datetime.datetime.now()

        Or_str = "# *_or.dat created by berry.py\n"
        Or_str += "# written {}\n".format(t.strftime("on %d%b%Y at %H:%M:%S"))
        Or_str += " {0[0]:15.8f} {0[1]:15.8f} {0[2]:15.8f}\n".format(unit_cell_cart[0, :])
        Or_str += " {0[0]:15.8f} {0[1]:15.8f} {0[2]:15.8f}\n".format(unit_cell_cart[1, :])
        Or_str += " {0[0]:15.8f} {0[1]:15.8f} {0[2]:15.8f}\n".format(unit_cell_cart[2, :])
        Or_str += "{:12d}\n{:12d}\n".format(num_wann, len(ndegen))
        Or_str += textwrap.fill("".join(["{:5d}".format(x) for x in ndegen]), 75, drop_whitespace=False)
        Or_str += "\n"

        for irpts in range(len(ndegen)):
            for i, j in itertools.product(range(num_wann), repeat=2):
                line = "{:5d}{:5d}{:5d}{:5d}{:5d}  ".format(*irvec[irpts, :], j + 1, i + 1)
                if vec:
                    line += "".join([" {:>12.8f} {:>12.8f}".format(x.real, x.imag) for x in Or[:, irpts, j, i]])
                else:
                    x = Or[irpts, j, i]
                    line += " {:>12.8f} {:>12.8f}".format(x.real, x.imag)
                line += "\n"

                Or_str += line

        self._cwm.write(filename, Or_str, header, None)

    # ==================================================
    def write_samb_coeffs(self, filename, type="z"):
        """
        write seedname_or.dat.

        Args:
            filename (str): file name.
            type (str): 'z'/'z_nonortho'/'s'.
        """
        assert type in ("z", "z_nonortho", "s"), f"invalid type = {type} was given. choose from 'z'/'z_nonortho'/'s'."

        if type == "z":
            header = z_header
            o = self["z"]
        elif type == "z_nonortho":
            header = z_nonortho_header
            o = self["z_nonortho"]
        elif type == "s":
            header = s_header
            o = self["s"]
        else:
            raise Exception(f"invalid type = {type} was given. choose from 'z'/'z_nonortho'/'s'.")

        o_str = "".join(
            [
                " {:>7d}   {:>15}   {:>15}   {:>12.8f}".format(j + 1, zj, tag, v)
                for j, ((zj, tag), v) in enumerate(o.items())
            ]
        )

        self._cwm.write(filename, o_str, header, None)