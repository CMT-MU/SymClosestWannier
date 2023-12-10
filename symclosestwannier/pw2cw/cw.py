"""
Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.
"""
import numpy as np
from numpy import linalg as npl
from scipy import linalg as spl

from gcoreutils.nsarray import NSArray

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
            dic = self._initialize()
        else:
            dic = self._cwm.read(f"{self._cwi['seedname']}_data.py")

        self.update(dic)

        msg = f"  * total elapsed_time:"
        self._cwm.log(msg, stamp="start", file=self._outfile, mode="a")

        self._cwm.log(ending_msg(), stamp=None, end="\n", file=self._outfile, mode="a")

    # ==================================================
    def _initialize(self):
        """
        initilize the class.

        Returns:
            dict:
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

        if self._cwi["symmetrization"]:
            self._symmetrize()

        return {
            "kpoints": self._cwi["kpoints"],
            "rpoints": CW.kpoints_to_rpoints(self._cwi["kpoints"]).tolist(),
            "kpoints_path": kpoints_path if kpoints_path is not None else None,
            "k_linear": k_linear if k_linear is not None else None,
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
    def write_hr(self):
        """
        write seedname_hr.dat.
        """
        Hr_dict = CW.matrix_dict_r(self["Hr"], self["rpoints"])
        Hr_str = "".join(
            [
                f"{n1}  {n2}  {n3}  {a}  {b}  {'{:.8f}'.format(np.real(v))}  {'{:.8f}'.format(np.imag(v))}\n"
                for (n1, n2, n3, a, b), v in Hr_dict.items()
            ]
        )
        self._cwm.write(f"{self._cwi['seedname']}_hr.dat", Hr_str, CW._hr_header(), None)

    # ==================================================
    def write_sr(self):
        """
        write seedname_sr.dat.
        """
        Sr_dict = CW.matrix_dict_r(self["Sr"], self["rpoints"])
        Sr_str = "".join(
            [
                f"{n1}  {n2}  {n3}  {a}  {b}  {'{:.8f}'.format(np.real(v))}  {'{:.8f}'.format(np.imag(v))}\n"
                for (n1, n2, n3, a, b), v in Sr_dict.items()
            ]
        )
        self._cwm.write(f"{self._cwi['seedname']}_sr.dat", Sr_str, CW._sr_header(), None)

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
    def dict_to_matrix(cls, dic):
        """
        convert dictionary form to matrix form of an arbitrary operator matrix.

        Args:
            dic (dict): dictionary form of an arbitrary operator matrix in reak-space/k-space representation.

        Returns:
            ndarray: matrix form of the given operator.
        """
        return dict_to_matrix(dic)

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
