# ****************************************************************** #
#                                                                    #
# This file is distributed as part of the symclosestwannier code and #
#     under the terms of the GNU General Public License. See the     #
#     file LICENSE in the root directory of the symclosestwannier    #
#      distribution, or http://www.gnu.org/licenses/gpl-3.0.txt      #
#                                                                    #
#          The symclosestwannier code is hosted on GitHub:           #
#                                                                    #
#            https://github.com/CMT-MU/SymClosestWannier             #
#                                                                    #
#                            written by                              #
#                        Rikuto Oiwa, RIKEN                          #
#                                                                    #
# ------------------------------------------------------------------ #
#                                                                    #
#                cw_model: Closest Wannier TB model                  #
#                                                                    #
# ****************************************************************** #

import os
import h5py
import ast
import datetime
import itertools
import textwrap

import numpy as np
from numpy import linalg as npl
from scipy import linalg as spl

from symclosestwannier.util.message import cw_start_msg, cw_start_msg_w90
from symclosestwannier.util.header import (
    cw_info_header,
    cw_data_header,
    kpoints_header,
    rpoints_header,
    hk_header,
    sk_header,
    nk_header,
    pk_header,
    hr_header,
    sr_header,
    nr_header,
    z_header,
    z_nonortho_header,
    s_header,
    n_header,
    sx_header,
    sy_header,
    sz_header,
    O_R_dependence_header,
)
from symclosestwannier.util.utility import (
    fermi,
    weight_proj,
    band_distance,
    get_wannier_center_spread,
    get_spreads,
    fourier_transform_k_to_r,
    fourier_transform_r_to_k,
    fourier_transform_r_to_k_vec,
    interpolate,
    matrix_dict_r,
    matrix_dict_k,
    dict_to_matrix,
    sort_ket_matrix,
    samb_decomp_operator,
    O_R_dependence,
    construct_Or,
    construct_Ok,
    spin_zeeman_interaction,
    su2_that_maps_z_to_n,
    embed_spin_unitary,
    Kelvin_to_eV,
)

_default = {
    "Sk": None,
    "nk": None,
    "Hk": None,
    "Hk_nonortho": None,
    #
    "Sr": None,
    "nr": None,
    "Hr": None,
    "Hr_nonortho": None,
    #
    "s": None,
    "n": None,
    "z": None,
    "z_nonortho": None,
    #
    "Sk_sym": None,
    "nk_sym": None,
    "Hk_sym": None,
    "Hk_nonortho_sym": None,
    "Sr_sym": None,
    "nr_sym": None,
    "Hr_sym": None,
    "Hr_nonortho_sym": None,
    #
    "Ek_MAE_grid": None,
    "Ek_MAE_path": None,
}


# ==================================================
class CWModel(dict):
    """
    Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.

    Attributes:
        _cwi (CWInfo): CWInfo.
        _cwm (CWManager): CWManager.
        _samb_info (dict): SAMB info.
        _outfile (str): output file, seedname.cwout.
    """

    # ==================================================
    def __init__(self, cwi, cwm, samb_info={}, dic=None):
        """
        Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.

        Args:
            cwi (CWInfo, optional): CWInfo.
            cwm (CWManager, optional): CWManager.
            samb_info (dict, optional): SAMB info.
            dic (dict, optional): dictionary of data.
        """
        super().__init__()

        self._cwi = cwi
        self._cwm = cwm
        self._samb_info = samb_info
        self._outfile = f"{self._cwi['seedname']}.cwout"

        if dic is not None:
            self.update(dic)
        if dic is None:
            self.update(_default)
            if self._cwi["restart"] == "cw":
                self._cw()
            elif self._cwi["restart"] == "w90":
                self._w90()
            else:
                raise Exception(f"invalid restart = {self._cwi['restart']} was given. choose from 'cw'/'w90'.")

    # ==================================================
    def _w90(self):
        """
        construct Wannier TB model by using wannier90 outputs.
        """
        self._cwm.log(cw_start_msg_w90(self._cwi["seedname"]), stamp=None, end="\n", file=self._outfile, mode="a")

        Ek = np.array(self._cwi["Ek"], dtype=float)
        Ak = np.array(self._cwi["Ak"], dtype=complex)
        Uk = np.array(self._cwi["Uk"], dtype=complex)
        Sk = Ak.transpose(0, 2, 1).conjugate() @ Ak
        Sk = 0.5 * (Sk + np.einsum("kmn->knm", Sk).conj())

        Hk = np.einsum("klm,kl,kln->kmn", np.conj(Uk), Ek, Uk, optimize=True)
        Hk = 0.5 * (Hk + np.einsum("kmn->knm", Hk).conj())

        # zeeman interaction
        if self._cwi["zeeman_interaction"]:
            B = self._cwi["magnetic_field"]
            theta = self._cwi["magnetic_field_theta"]
            phi = self._cwi["magnetic_field_phi"]
            g_factor = self._cwi["g_factor"]

            pauli_spin = Uk.transpose(0, 2, 1).conjugate() @ self._cwi["pauli_spn"] @ Uk
            H_zeeman = spin_zeeman_interaction(B, theta, phi, pauli_spin, g_factor, self._cwi["num_wann"])
            Hk += H_zeeman

        S2k = np.array([spl.sqrtm(Sk[k]) for k in range(self._cwi["num_k"])])
        Hk_nonortho = S2k @ Hk @ S2k

        # electronic density matrix elements
        ef = self._cwi["fermi_energy"]
        fk = np.array([np.diag(fermi(eki - ef, T=0.0)) for eki in Ek], dtype=float)
        nk = Uk.transpose(0, 2, 1).conjugate() @ fk @ Uk

        if self._cwi["calc_spin_2d"]:
            SSk = np.array(self._cwi["pauli_spn"])
            SSk = np.array([(fk @ SSk[a] + SSk[a] @ fk) / 2 for a in range(3)])
            SSk = np.einsum("klm,aklp,kpn->akmn", np.conj(Uk), SSk, Uk, optimize=True)
            SSk = 0.5 * (SSk + np.einsum("akmn->aknm", SSk).conj())
            SSr = np.array(
                [CWModel.fourier_transform_k_to_r(SSk[a], self._cwi["kpoints"], self._cwi["irvec"]) for a in range(3)]
            )
        else:
            SSk = None
            SSr = None

        Sr = CWModel.fourier_transform_k_to_r(Sk, self._cwi["kpoints"], self._cwi["irvec"])
        nr = CWModel.fourier_transform_k_to_r(nk, self._cwi["kpoints"], self._cwi["irvec"])
        Hr = CWModel.fourier_transform_k_to_r(Hk, self._cwi["kpoints"], self._cwi["irvec"])
        Hr_nonortho = CWModel.fourier_transform_k_to_r(Hk_nonortho, self._cwi["kpoints"], self._cwi["irvec"])

        #####

        self.update(
            {
                "Sk": Sk.tolist(),
                "nk": nk.tolist(),
                "Hk": Hk.tolist(),
                "Hk_nonortho": Hk_nonortho.tolist(),
                "SSk": SSk.tolist() if SSk is not None else SSk,
                #
                "Sr": Sr.tolist(),
                "nr": nr.tolist(),
                "Hr": Hr.tolist(),
                "Hr_nonortho": Hr_nonortho.tolist(),
                "SSr": SSr.tolist() if SSk is not None else SSk,
            }
        )

        # band distance
        self._cwm.log("* band distance between DFT and Wannier bands:", None, file=self._outfile, mode="a")
        self._cwm.set_stamp()

        eta_0, eta_0_max, eta_2, eta_2_max, eta_5, eta_5_max = band_distance(Ak, Ek, Hk, ef=self._cwi["fermi_energy"])

        self._cwm.log(f" - eta_0     = {eta_0} [meV]", file=self._outfile, mode="a")
        self._cwm.log(f" - eta_0_max = {eta_0_max} [meV]", file=self._outfile, mode="a")
        self._cwm.log(f" - eta_2     = {eta_2} [meV]", file=self._outfile, mode="a")
        self._cwm.log(f" - eta_2_max = {eta_2_max} [meV]", file=self._outfile, mode="a")
        self._cwm.log(f" - eta_5     = {eta_5} [meV]", file=self._outfile, mode="a")
        self._cwm.log(f" - eta_5_max = {eta_5_max} [meV]", file=self._outfile, mode="a")

        self._cwm.log("done", file=self._outfile, mode="a")

        # spreads
        if self._cwi["calc_spreads"] and self._cwi.mmn["Mkb"] is not None:

            #
            self._cwm.set_stamp()

            self._cwm.log("\n    * WF center and Spread:", None, file=self._outfile, mode="a")

            r, Omega = get_wannier_center_spread(self._cwi)

            self._cwm.log(
                "     idx        center (cartesian)                             Spread (Ang^2)",
                None,
                file=self._outfile,
                mode="a",
            )

            for m in range(self._cwi["num_wann"]):
                self._cwm.log(
                    "  {0:6d}        ( {1[0]:12.6f}, {1[1]:12.6f}, {1[2]:12.6f} )  {2:15.8f}".format(
                        m + 1, r[m].real, Omega[m].real
                    ),
                    None,
                    file=self._outfile,
                    mode="a",
                )

            self._cwm.log(
                "     Sum        ( {0[0]:12.6f}, {0[1]:12.6f}, {0[2]:12.6f} )  {1:15.8f}".format(
                    np.sum(r, axis=0).real, np.sum(Omega).real
                ),
                None,
                file=self._outfile,
                mode="a",
            )

            #
            self._cwm.log("\n    * Spreads (Ang^2):", None, file=self._outfile, mode="a")

            OmegaI, OmegaD, OmegaOD = get_spreads(self._cwi)

            self._cwm.log("     - Omega I      = {0:15.8f}".format(OmegaI), None, file=self._outfile, mode="a")
            self._cwm.log("     - Omega D      = {0:15.8f}".format(OmegaD), None, file=self._outfile, mode="a")
            self._cwm.log("     - Omega OD     = {0:15.8f}".format(OmegaOD), None, file=self._outfile, mode="a")
            self._cwm.log(
                "     - Omega Total  = {0:15.8f}".format(OmegaI + OmegaD + OmegaOD),
                None,
                file=self._outfile,
                mode="a",
            )

            self._cwm.log("done", file=self._outfile, mode="a")

        if self._cwi["symmetrization"]:
            msg = "   - symmetrization ... "
            self._cwm.log(msg, None, end="\n", file=self._outfile, mode="a")
            self._sym()

    # ==================================================
    def _cw(self):
        """
        construct CW TB model.
        """
        self._cwm.log(cw_start_msg(self._cwi["seedname"]), stamp=None, end="\n", file=self._outfile, mode="a")

        Ek = np.array(self._cwi["Ek"], dtype=float)
        Ak = np.array(self._cwi["Ak"], dtype=complex)

        # 30 orbitals
        # Ak_tmp = np.zeros(Ak.shape, dtype=complex)
        # Ak_tmp =  Ak.copy()
        # 0-4
        # 5-9
        # 10-14
        # #
        # 15-17
        # #
        # 18-20
        # Ak_tmp[:,:,18] = (Ak[:,:,18] + Ak[:,:,21]) / np.sqrt(2) # (pz@Sb4h1 + pz@Sb4h2)/sqrt(2)
        # Ak_tmp[:,:,19] = (Ak[:,:,19] - Ak[:,:,22]) / np.sqrt(2) # (px@Sb4h1 - px@Sb4h2)/sqrt(2)
        # Ak_tmp[:,:,20] = (Ak[:,:,20] - Ak[:,:,23]) / np.sqrt(2) # (py@Sb4h1 - py@Sb4h2)/sqrt(2)
        # 21-23
        # Ak_tmp[:,:,21] = (Ak[:,:,18] - Ak[:,:,21]) / np.sqrt(2) # (pz@Sb4h1 - pz@Sb4h2)/sqrt(2)
        # Ak_tmp[:,:,22] = (Ak[:,:,19] + Ak[:,:,22]) / np.sqrt(2) # (px@Sb4h1 + px@Sb4h2)/sqrt(2)
        # Ak_tmp[:,:,23] = (Ak[:,:,20] + Ak[:,:,23]) / np.sqrt(2) # (py@Sb4h1 + py@Sb4h2)/sqrt(2)
        # 24-26
        # Ak_tmp[:,:,24] = (Ak[:,:,24] + Ak[:,:,27]) / np.sqrt(2) # (pz@Sb4h1 + pz@Sb4h2)/sqrt(2)
        # Ak_tmp[:,:,25] = (Ak[:,:,25] - Ak[:,:,28]) / np.sqrt(2) # (px@Sb4h1 - px@Sb4h2)/sqrt(2)
        # Ak_tmp[:,:,26] = (Ak[:,:,26] - Ak[:,:,29]) / np.sqrt(2) # (py@Sb4h1 - py@Sb4h2)/sqrt(2)
        # 27-29
        # Ak_tmp[:,:,27] = (Ak[:,:,24] - Ak[:,:,27]) / np.sqrt(2) # (pz@Sb4h1 - pz@Sb4h2)/sqrt(2)
        # Ak_tmp[:,:,28] = (Ak[:,:,25] + Ak[:,:,28]) / np.sqrt(2) # (px@Sb4h1 + px@Sb4h2)/sqrt(2)
        # Ak_tmp[:,:,29] = (Ak[:,:,26] + Ak[:,:,29]) / np.sqrt(2) # (py@Sb4h1 + py@Sb4h2)/sqrt(2)
        # #Ak_tmp[:,:,12:] = 0.0
        # Ak = Ak_tmp
        # self._cwi["Ak"] = Akf
        # Ak = np.zeros((Ak.shape[0],Ak.shape[1],12))
        # Ak = Ak_tmp[:,:,:12]
        # self._cwi["num_wann"] = 12

        # Mz-odd p orbital (12 orbitals)
        # Ak_tmp = np.zeros(Ak.shape, dtype=complex)
        # Ak_tmp = Ak.copy()
        # Ak[:, :, 6] = (Ak_tmp[:, :, 6] + Ak_tmp[:, :, 9]) / np.sqrt(2)  # (pz@Sb4h1 + pz@Sb4h2)/sqrt(2)
        # Ak[:, :, 7] = (Ak_tmp[:, :, 7] - Ak_tmp[:, :, 10]) / np.sqrt(2)  # (px@Sb4h1 - px@Sb4h2)/sqrt(2)
        # Ak[:, :, 8] = (Ak_tmp[:, :, 8] - Ak_tmp[:, :, 11]) / np.sqrt(2)  # (py@Sb4h1 - py@Sb4h2)/sqrt(2)
        # Ak[:, :, 9] = (Ak_tmp[:, :, 12] + Ak_tmp[:, :, 15]) / np.sqrt(2)  # (pz@Sb4h3 + pz@Sb4h4)/sqrt(2)
        # Ak[:, :, 10] = (Ak_tmp[:, :, 13] - Ak_tmp[:, :, 16]) / np.sqrt(2)  # (px@Sb4h3 - px@Sb4h4)/sqrt(2)
        # Ak[:, :, 11] = (Ak_tmp[:, :, 14] - Ak_tmp[:, :, 17]) / np.sqrt(2)  # (py@Sb4h3 - py@Sb4h4)/sqrt(2)
        # Ak = Ak[:, :, :12]
        # self._cwi["num_wann"] = 12
        # self._cwi["Ak"] = Ak

        # Mz-odd p orbital (10 orbitals)
        # Ak_tmp = np.zeros(Ak.shape, dtype=complex)
        # Ak_tmp = Ak.copy()
        # Ak[:, :, 6] = (Ak_tmp[:, :, 7] - Ak_tmp[:, :, 10]) / np.sqrt(2)  # (px@Sb4h1 - px@Sb4h2)/sqrt(2)
        # Ak[:, :, 7] = (Ak_tmp[:, :, 8] - Ak_tmp[:, :, 11]) / np.sqrt(2)  # (py@Sb4h1 - py@Sb4h2)/sqrt(2)
        # Ak[:, :, 8] = (Ak_tmp[:, :, 13] - Ak_tmp[:, :, 16]) / np.sqrt(2)  # (px@Sb4h3 - px@Sb4h4)/sqrt(2)
        # Ak[:, :, 9] = (Ak_tmp[:, :, 14] - Ak_tmp[:, :, 17]) / np.sqrt(2)  # (py@Sb4h3 - py@Sb4h4)/sqrt(2)
        # Ak = Ak[:, :, :10]
        # self._cwi["num_wann"] = 10
        # self._cwi["Ak"] = Ak

        # Mz-odd p orbital (13 orbitals)
        # Ak_tmp = np.zeros(Ak.shape, dtype=complex)
        # Ak_tmp = np.zeros(Ak.shape, dtype=complex)
        # Ak_tmp = Ak.copy()
        # Ak[:, :, 6] = (Ak_tmp[:, :, 6] + Ak_tmp[:, :, 9]) / np.sqrt(2)  # (pz@Sb4h1 + pz@Sb4h2)/sqrt(2)
        # Ak[:, :, 7] = (Ak_tmp[:, :, 7] - Ak_tmp[:, :, 10]) / np.sqrt(2)  # (px@Sb4h1 - px@Sb4h2)/sqrt(2)
        # Ak[:, :, 8] = (Ak_tmp[:, :, 8] - Ak_tmp[:, :, 11]) / np.sqrt(2)  # (py@Sb4h1 - py@Sb4h2)/sqrt(2)
        # Ak[:, :, 9] = (Ak_tmp[:, :, 12] + Ak_tmp[:, :, 15]) / np.sqrt(2)  # (pz@Sb4h3 + pz@Sb4h4)/sqrt(2)
        # Ak[:, :, 10] = (Ak_tmp[:, :, 13] - Ak_tmp[:, :, 16]) / np.sqrt(2)  # (px@Sb4h3 - px@Sb4h4)/sqrt(2)
        # Ak[:, :, 11] = (Ak_tmp[:, :, 14] - Ak_tmp[:, :, 17]) / np.sqrt(2)  # (py@Sb4h3 - py@Sb4h4)/sqrt(2)
        # Ak[:, :, 12] = Ak_tmp[:, :, 18]
        # Ak = Ak[:, :, :13]
        # self._cwi["num_wann"] = 13
        # self._cwi["Ak"] = Ak

        # Mz-even p orbital
        # Ak_tmp = np.zeros(Ak.shape, dtype=complex)
        # Ak_tmp = Ak.copy()
        # Ak[:, :, 0] = Ak_tmp[:, :, 0]  # V1: du
        # Ak[:, :, 1] = Ak_tmp[:, :, 3]  # V2: du
        # Ak[:, :, 2] = Ak_tmp[:, :, 6]  # V3: du
        # Ak[:, :, 3] = (Ak_tmp[:, :, 11] + Ak_tmp[:, :, 13]) / np.sqrt(2)  # (px@Sb4h1 + px@Sb4h2)/sqrt(2)
        # Ak[:, :, 4] = (Ak_tmp[:, :, 12] + Ak_tmp[:, :, 14]) / np.sqrt(2)  # (py@Sb4h1 + py@Sb4h2)/sqrt(2)
        # Ak[:, :, 5] = (Ak_tmp[:, :, 15] + Ak_tmp[:, :, 17]) / np.sqrt(2)  # (px@Sb4h3 + px@Sb4h4)/sqrt(2)
        # Ak[:, :, 6] = (Ak_tmp[:, :, 16] + Ak_tmp[:, :, 18]) / np.sqrt(2)  # (py@Sb4h3 + py@Sb4h4)/sqrt(2)
        # Ak = Ak[:, :, :7]
        # self._cwi["num_wann"] = 7
        # self._cwi["Ak"] = Ak

        # H4
        # Ak_tmp = np.zeros(Ak.shape, dtype=complex)
        # Ak_tmp =  Ak.copy()
        # Ak = np.zeros((Ak.shape[0],Ak.shape[1],1))
        # Ak[:,:,0] = Ak_tmp[:,:,0]
        # self._cwi["Ak"] = Ak
        # self._cwi["num_wann"] = 1

        # GdCo5
        # Ak_tmp = np.zeros(Ak.shape, dtype=complex)
        # Ak_tmp = Ak.copy()
        # Ak = np.zeros((Ak.shape[0],Ak.shape[1],25), dtype=complex)
        # Ak[:,:,0:5] = Ak_tmp[:,:,17:22]
        # Ak[:,:,5:10] = Ak_tmp[:,:,26:31]
        # Ak[:,:,10:15] = Ak_tmp[:,:,35:40]
        # Ak[:,:,15:20] = Ak_tmp[:,:,44:49]
        # Ak[:,:,20:25] = Ak_tmp[:,:,53:58]
        # self._cwi["num_wann"] = 25
        # self._cwi["Ak"] = Ak

        if self._cwi["proj_min"] > 0.0:
            msg = f"   - excluding bands with low projectability (proj_min = {self._cwi['proj_min']}) ... "
            self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
            self._cwm.set_stamp()

            Ak = self._exclude_bands(Ak)

            self._cwm.log("done", file=self._outfile, mode="a")

        if self._cwi["disentangle"]:
            msg = "   - disentanglement ... "
            self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
            self._cwm.set_stamp()

            cwf_mu_min, cwf_mu_max, cwf_sigma_min, cwf_sigma_max = (
                self._cwi["cwf_mu_min"],
                self._cwi["cwf_mu_max"],
                self._cwi["cwf_sigma_min"],
                self._cwi["cwf_sigma_max"],
            )
            Ak = self._disentangle(Ek, Ak, cwf_mu_min, cwf_mu_max, cwf_sigma_min, cwf_sigma_max)

            self._cwm.log("done", file=self._outfile, mode="a")

        with open(f"{self._cwi['seedname']}.amn_window.cw", "w") as fp:

            Ak_ = Ak
            # Ak_ = np.array(self._cwi["Ak"], dtype=complex)

            def U_mat(k):
                u, _, vd = np.linalg.svd(Ak_[k], full_matrices=False)
                return u @ vd

            Uk = np.array([U_mat(k) for k in range(self._cwi["num_k"])])

            fp.write("Created by amn.py {}\n".format(datetime.datetime.now().strftime("on %d%b%Y at %H:%M:%S")))
            fp.write(
                "       {:5d}       {:5d}       {:5d}\n".format(
                    self._cwi["num_bands"], self._cwi["num_k"], self._cwi["num_wann"]
                )
            )
            for ik, m, n in itertools.product(
                range(self._cwi["num_k"]), range(self._cwi["num_wann"]), range(self._cwi["num_bands"])
            ):
                # num_bands, num_wann, nk
                fp.write(
                    "{0:5d}{1:5d}{2:5d}{3.real:18.12f}{3.imag:18.12f}\n".format(n + 1, m + 1, ik + 1, Uk[ik, n, m])
                )

        msg = "   - constructing TB Hamiltonian ... "
        self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        Sk, Uk, Hk, Sr, Hr, Hk_nonortho, Hr_nonortho = self._construct_tb(Ek, Ak)

        # electronic density matrix elements
        ef = self._cwi["fermi_energy"]

        fk = np.array([np.diag(fermi(eki - ef, T=0.0)) for eki in Ek], dtype=float)
        nk = Uk.transpose(0, 2, 1).conjugate() @ fk @ Uk
        nr = CWModel.fourier_transform_k_to_r(nk, self._cwi["kpoints"], self._cwi["irvec"])

        if self._cwi["calc_spin_2d"] and self._cwi["pauli_spn"] is not None:
            SSk = np.array(self._cwi["pauli_spn"])
            SSk = np.array([(fk @ SSk[a] + SSk[a] @ fk) / 2 for a in range(3)])
            SSk = np.einsum("klm,aklp,kpn->akmn", np.conj(Uk), SSk, Uk, optimize=True)
            SSk = 0.5 * (SSk + np.einsum("akmn->aknm", SSk).conj())
            SSr = np.array(
                [CWModel.fourier_transform_k_to_r(SSk[a], self._cwi["kpoints"], self._cwi["irvec"]) for a in range(3)]
            )
        else:
            SSk = None
            SSr = None

        self._cwm.log("done", file=self._outfile, mode="a")

        #####

        def Hr_2d(m):
            d = {}
            for i, (n1, n2, n3) in enumerate(self._cwi["irvec"]):
                if (n1, n2) not in d:
                    d[(n1, n2)] = m[i]
                else:
                    d[(n1, n2)] += m[i]

            m_2d = np.zeros(m.shape, dtype=complex)
            for i, (n1, n2, n3) in enumerate(self._cwi["irvec"]):
                if n3 == 0:
                    m_2d[i] = d[(n1, n2)]

            return m_2d

        # Hr = Hr_2d(Hr)
        # Hr_nonortho = Hr_2d(Hr_nonortho)

        #####

        self._cwi["Ak"] = Ak
        self._cwi["Uoptk"] = np.array([np.identity(self._cwi["num_wann"], dtype=complex)] * self._cwi["num_k"])
        self._cwi["Udisk"] = Uk
        self._cwi["Uk"] = Uk

        self.update(
            {
                "Sk": Sk.tolist(),
                "nk": nk.tolist(),
                "Hk": Hk.tolist(),
                "Hk_nonortho": Hk_nonortho.tolist(),
                "SSk": SSk.tolist() if SSk is not None else SSk,
                #
                "Sr": Sr.tolist(),
                "nr": nr.tolist(),
                "Hr": Hr.tolist(),
                "Hr_nonortho": Hr_nonortho.tolist(),
                "SSr": SSr.tolist() if SSk is not None else SSk,
            }
        )

        # band distance
        self._cwm.log("\n    * band distance between DFT and Wannier bands:", None, file=self._outfile, mode="a")
        self._cwm.set_stamp()

        eta_0, eta_0_max, eta_2, eta_2_max, eta_5, eta_5_max = band_distance(Ak, Ek, Hk, ef=self._cwi["fermi_energy"])

        # self._cwm.log(f"     - bottom_band_idx = {bottom_band_idx}", None, file=self._outfile, mode="a")
        self._cwm.log(f"     - eta_0     = {eta_0} [meV]", None, file=self._outfile, mode="a")
        self._cwm.log(f"     - eta_0_max = {eta_0_max} [meV]", None, file=self._outfile, mode="a")
        self._cwm.log(f"     - eta_2     = {eta_2} [meV]", None, file=self._outfile, mode="a")
        self._cwm.log(f"     - eta_2_max = {eta_2_max} [meV]", None, file=self._outfile, mode="a")
        self._cwm.log(f"     - eta_5     = {eta_5} [meV]", None, file=self._outfile, mode="a")
        self._cwm.log(f"     - eta_5_max = {eta_5_max} [meV]", None, file=self._outfile, mode="a")

        self._cwm.log("done", file=self._outfile, mode="a")

        # spreads
        if self._cwi["calc_spreads"] and self._cwi.mmn["Mkb"] is not None:

            #
            self._cwm.set_stamp()

            self._cwm.log("\n    * WF center and Spread:", None, file=self._outfile, mode="a")

            r, Omega = get_wannier_center_spread(self._cwi)

            self._cwm.log(
                "     idx        center (cartesian)                             Spread (Ang^2)",
                None,
                file=self._outfile,
                mode="a",
            )

            for m in range(self._cwi["num_wann"]):
                self._cwm.log(
                    "  {0:6d}        ( {1[0]:12.6f}, {1[1]:12.6f}, {1[2]:12.6f} )  {2:15.8f}".format(
                        m + 1, r[m].real, Omega[m].real
                    ),
                    None,
                    file=self._outfile,
                    mode="a",
                )

            self._cwm.log(
                "     Sum        ( {0[0]:12.6f}, {0[1]:12.6f}, {0[2]:12.6f} )  {1:15.8f}".format(
                    np.sum(r, axis=0).real, np.sum(Omega).real
                ),
                None,
                file=self._outfile,
                mode="a",
            )

            #
            self._cwm.log("\n    * Spreads (Ang^2):", None, file=self._outfile, mode="a")

            OmegaI, OmegaD, OmegaOD = get_spreads(self._cwi)

            self._cwm.log("     - Omega I      = {0:15.8f}".format(OmegaI), None, file=self._outfile, mode="a")
            self._cwm.log("     - Omega D      = {0:15.8f}".format(OmegaD), None, file=self._outfile, mode="a")
            self._cwm.log("     - Omega OD     = {0:15.8f}".format(OmegaOD), None, file=self._outfile, mode="a")
            self._cwm.log(
                "     - Omega Total  = {0:15.8f}".format(OmegaI + OmegaD + OmegaOD),
                None,
                file=self._outfile,
                mode="a",
            )

            self._cwm.log("done", file=self._outfile, mode="a")

        # symmetrization
        if self._cwi["symmetrization"]:
            msg = "   - symmetrization ... "
            self._cwm.log(msg, None, end="\n", file=self._outfile, mode="a")
            self._sym()

    # ==================================================
    def _exclude_bands(self, Ak):
        """
        exclude bands by setting the matrix elements of Ak with low projectability to zero.

        Args:
            Ak (ndarray): Overlap matrix elements.

        Returns:
            ndarray: Ak.
        """
        num_k = self._cwi["num_k"]
        num_bands = self._cwi["num_bands"]
        num_wann = self._cwi["num_wann"]
        proj_min = self._cwi["proj_min"]

        # projectability of each Kohn-Sham state in k-space.
        Pk = np.real(np.diagonal(Ak @ Ak.transpose(0, 2, 1).conjugate(), axis1=1, axis2=2))

        # band index for projection
        proj_band_idx = [[n for n in range(num_bands) if Pk[k][n] > proj_min] for k in range(num_k)]

        for k in range(num_k):
            if len(proj_band_idx[k]) < num_wann:
                raise Exception(f"proj_min = {proj_min} is too large or PAOs are inappropriate.")
            for n in range(num_bands):
                if n not in proj_band_idx[k]:
                    Ak[k, n, :] = 0

        return Ak

    # ==================================================
    def _disentangle(self, Ek, Ak, cwf_mu_min, cwf_mu_max, cwf_sigma_min, cwf_sigma_max):
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
            cwf_mu_min,
            cwf_mu_max,
            cwf_sigma_min,
            cwf_sigma_max,
            self._cwi["cwf_delta"],
        )

        return w[:, :, np.newaxis] * Ak

    # ==================================================
    def _construct_tb(self, Ek, Ak):
        """
        construct CW TB Hamiltonian.

        Args:
            Ek (ndarray): Kohn-Sham energies.
            Ak (ndarray): Overlap matrix elements.

        Returns:
            tuple: Sk, Uk, Hk, Sr, Hr.
                - Sk (ndarray) : Overlap matrix elements in k-space.
                - Uk (ndarray) : Unitary matrix elements in k-space.
                - Hk (ndarray) : Hamiltonian matrix elements in k-space (orthogonal).
                - Sr (ndarray) : Overlap matrix elements in real-space.
                - Hr (ndarray) : Hamiltonian matrix elements in real-space (orthogonal).
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
        Hk = np.einsum("klm,kl,kln->kmn", np.conj(Uk), Ek, Uk, optimize=True)
        Hk = 0.5 * (Hk + np.einsum("kmn->knm", Hk).conj())

        # Hk = Hk - self._cwi["fermi_energy"] * np.eye(Hk.shape[-1])[np.newaxis, :, :]

        # zeeman interaction
        if self._cwi["zeeman_interaction"]:
            B = self._cwi["magnetic_field"]
            theta = self._cwi["magnetic_field_theta"]
            phi = self._cwi["magnetic_field_phi"]
            g_factor = self._cwi["g_factor"]

            pauli_spin = Uk.transpose(0, 2, 1).conjugate() @ self._cwi["pauli_spn"] @ Uk
            H_zeeman = spin_zeeman_interaction(B, theta, phi, pauli_spin, g_factor, self._cwi["num_wann"])
            Hk += H_zeeman

        S2k = np.array([spl.sqrtm(Sk[k]) for k in range(self._cwi["num_k"])])
        Hk_nonortho = S2k @ Hk @ S2k

        Sr = CWModel.fourier_transform_k_to_r(Sk, self._cwi["kpoints"], self._cwi["irvec"])
        Hr = CWModel.fourier_transform_k_to_r(Hk, self._cwi["kpoints"], self._cwi["irvec"])
        Hr_nonortho = CWModel.fourier_transform_k_to_r(Hk_nonortho, self._cwi["kpoints"], self._cwi["irvec"])

        return Sk, Uk, Hk, Sr, Hr, Hk_nonortho, Hr_nonortho

    # ==================================================
    def _sym(self):
        """
        symmetrize CW TB Hamiltonian.
        """
        Hk = np.array(self["Hk"])
        Sk = np.array(self["Sk"])
        nk = np.array(self["nk"])
        Hr_dict = CWModel.matrix_dict_r(self["Hr"], self._cwi["irvec"])
        Sr_dict = CWModel.matrix_dict_r(self["Sr"], self._cwi["irvec"])
        nr_dict = CWModel.matrix_dict_r(self["nr"], self._cwi["irvec"])
        Hr_nonortho_dict = CWModel.matrix_dict_r(self["Hr_nonortho"], self._cwi["irvec"])

        if self._cwi["calc_spin_2d"] and self._cwi["pauli_spn"] is not None:
            SSr_dict = np.array([CWModel.matrix_dict_r(self["SSr"][a], self._cwi["irvec"]) for a in range(3)])

        #####

        msg = "    - creating combined samb matrix ... "
        self._cwm.log(msg, None, end="\n", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        if self._cwi["irreps"] == "all":
            select = {}
        else:
            select = {"Gamma": self._cwi["irreps"]}

        combined_samb_matrix = self._cwi._mm.get_combined_samb_matrix(fmt="value", digit=15, **select)

        ### sign chagne for odd-parity site- and bond-cluster multipoles (L-handed CoSi) ###
        # for _, zj, d in res:
        #     tag = tag_dict[zj]
        #     coeff, amp, sbmp = samb["data"]["Z"][zj][1][0]
        #     if sbmp in samb["data"]["site_cluster"]:
        #         s_tag = samb["data"]["site_cluster"][sbmp][0]
        #         if TagMultipole(s_tag).rank % 2 == 1:
        #             d = {k: -v for k, v in d.items()}
        #     elif sbmp in samb["data"]["bond_cluster"]:
        #         b_tag = samb["data"]["bond_cluster"][sbmp][0]
        #         if TagMultipole(b_tag).rank % 2 == 1:
        #             d = {k: -v for k, v in d.items()}

        #     Zr_dict[(zj, tag)] = d
        #     mat["matrix"][zj] = d
        ### sign chagne for odd-parity site- and bond-cluster multipoles (L-handed CoSi) ###

        ### change spin quantization axis
        if self._cwi["spinors"]:
            saxis = self._cwi["nw2saxis"][0]
            if saxis != [0, 0, 1.0]:
                U = su2_that_maps_z_to_n(saxis)
                U = embed_spin_unitary(self._cwi["num_wann"], U)

                for zj, d in combined_samb_matrix.items():
                    m, rpoints = CWModel.dict_to_matrix(d, dim=self._cwi["num_wann"])
                    m = U.conj().T @ m[:] @ U
                    d = CWModel.matrix_dict_r(m, rpoints)
                    d = {k: v for k, v in d.items() if v != 0.0}

                    combined_samb_matrix[zj] = d
        ### change spin quantization axis

        self._cwm.log("done", file=self._outfile, mode="a")

        ###

        ket_samb = self._cwi._mm["full_matrix"]["ket"]
        ket_amn = self._cwi.get("ket_amn", ket_samb)

        ###

        msg = "    - decomposing Hamiltonian as linear combination of SAMBs ... "
        self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        z = CWModel.samb_decomp_operator(Hr_dict, combined_samb_matrix, ket=ket_amn, ket_samb=ket_samb)

        self._cwm.log("done", file=self._outfile, mode="a")

        #####

        msg = "    - decomposing overlap as linear combination of SAMBs ... "
        self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        s = CWModel.samb_decomp_operator(Sr_dict, combined_samb_matrix, ket=ket_amn, ket_samb=ket_samb)

        self._cwm.log("done", file=self._outfile, mode="a")

        #####

        msg = "    - decomposing non-orthogonal Hamiltonian as linear combination of SAMBs ... "
        self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        z_nonortho = CWModel.samb_decomp_operator(
            Hr_nonortho_dict, combined_samb_matrix, ket=ket_amn, ket_samb=ket_samb
        )

        self._cwm.log("done", file=self._outfile, mode="a")

        #####

        msg = "    - decomposing electronic density as linear combination of SAMBs ... "
        self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        n = CWModel.samb_decomp_operator(nr_dict, combined_samb_matrix, ket=ket_amn, ket_samb=ket_samb)

        self._cwm.log("done", file=self._outfile, mode="a")
        ###

        ###

        if self._cwi["calc_spin_2d"] and self._cwi["pauli_spn"] is not None:
            msg = "    - decomposing spin density as linear combination of SAMBs ... "
            self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
            self._cwm.set_stamp()

            ss = [
                CWModel.samb_decomp_operator(SSr_dict[a], combined_samb_matrix, ket=ket_amn, ket_samb=ket_samb)
                for a in range(3)
            ]

            self._cwm.log("done", file=self._outfile, mode="a")
        else:
            ss = None

        if self._cwi["calc_cohp_samb_decomp"]:
            msg = "    - decomposing electronic density as linear combination of SAMBs ... "

            # ==================================================
            def _lorentzian(e, g=0.001):
                return (1.0 / np.pi) * g / (g**2 + e**2)

            cohp_num_fermi = self._cwi["cohp_num_fermi"]
            cohp_smr_en_width = self._cwi["cohp_smr_en_width"]
            cohp_emax = self._cwi["cohp_emax"]
            cohp_emin = self._cwi["cohp_emin"]

            # electronic density matrix elements
            Ek = np.array(self._cwi["Ek"])
            Uk = np.array(self._cwi["Uk"])
            ef_shift = self._cwi["fermi_energy"]

            emax = np.max(Ek) if cohp_emax is None else cohp_emax
            emin = np.min(Ek) if cohp_emin is None else cohp_emin
            offset = (emax - emin) * 0.1
            ef_max = emax + offset
            ef_min = emin - offset

            num_k, num_wann = Ek.shape
            dE = (ef_max - ef_min) / cohp_num_fermi
            fermi_energy_list = [ef_min + i * dE for i in range(cohp_num_fermi + 1)]

            # fk = np.array([np.diag(fermi(eki - ef_shift, T=0.0)) for eki in Ek], dtype=float)
            n_list = []
            for ief in range(cohp_num_fermi + 1):
                print(f"{ief+1}/{cohp_num_fermi + 1}")
                ef = fermi_energy_list[ief]
                # if i > 19:
                #     continue
                delta_func = _lorentzian(Ek - ef, cohp_smr_en_width)
                delta_func = np.array([np.diag(delta_func_k) for delta_func_k in delta_func], dtype=float)
                # nk_delta_func = Uk.transpose(0, 2, 1).conjugate() @ fk @ delta_func @ Uk
                nk_delta_func = Uk.transpose(0, 2, 1).conjugate() @ delta_func @ Uk
                nr_delta_func = CWModel.fourier_transform_k_to_r(
                    nk_delta_func, self._cwi["kpoints"], self._cwi["irvec"]
                )
                nr_delta_dict = CWModel.matrix_dict_r(nr_delta_func, self._cwi["irvec"])

                n_i = CWModel.samb_decomp_operator(nr_delta_dict, combined_samb_matrix, ket_amn, ket_samb)
                n_list.append(n_i)

            n_list_integrated = []
            for ief in range(cohp_num_fermi + 1):
                d_integated = {k: 0.0 for k in n_list[0].keys()}
                for d in n_list[: ief + 1]:
                    for k, v in d.items():
                        d_integated[k] += v * dE
                n_list_integrated.append(d_integated)

            z_n_list = [{k: z[k] * nj for k, nj in n_list[ief].items()} for ief in range(cohp_num_fermi + 1)]
            z_n_list_integrated = [
                {k: z[k] * nj for k, nj in n_list_integrated[ief].items()} for ief in range(cohp_num_fermi + 1)
            ]
            cohp_str = ""
            icohp_str = ""
            for ief in range(cohp_num_fermi + 1):
                ef = fermi_energy_list[ief]
                z_n_i = z_n_list[ief]
                z_n_integrated_i = z_n_list_integrated[ief]
                cohp = np.sum([zj_nj for zj_nj in z_n_i.values()])
                icohp = np.sum([zj_nj for zj_nj in z_n_integrated_i.values()])
                cohp_str += str(ef - ef_shift) + "  " + str(cohp)
                icohp_str += str(ef - ef_shift) + "  " + str(icohp)

                for v, v_integrated in zip(z_n_i.values(), z_n_integrated_i.values()):
                    cohp_str += "  " + str(v)
                    icohp_str += "  " + str(v_integrated)

                cohp_str += "\n "
                icohp_str += "\n "

            filename = os.path.join(self._cwi["mp_outdir"], "{}".format(f"{self._cwi['mp_seedname']}_cohp.dat.cw"))
            self._cwm.write(filename, cohp_str, None, None)

            filename = os.path.join(self._cwi["mp_outdir"], "{}".format(f"{self._cwi['mp_seedname']}_icohp.dat.cw"))
            self._cwm.write(filename, icohp_str, None, None)

            n_ = n_list[-1]
            filename = os.path.join(self._cwi["mp_outdir"], "{}".format(f"{self._cwi['mp_seedname']}_n_.dat.cw"))
            self.write_samb_coeffs(filename, type="", o=n_)

            z_n_ = z_n_list[-1]
            filename = os.path.join(self._cwi["mp_outdir"], "{}".format(f"{self._cwi['mp_seedname']}_z_n_.dat.cw"))
            self.write_samb_coeffs(filename, type="", o=z_n_)

        self._cwm.log("done", file=self._outfile, mode="a")

        #####

        site_dict = {
            k + "_" + str(vi.sublattice): vi.position_primitive.tolist()
            for k, v in self._cwi._mm["site"]["cell"].items()
            for vi in v
            if vi.plus_set == 1
        }
        atoms_frac_samb = [
            site_dict[atom + "_" + str(sl)] for atom, sl, rank, orbital in self._cwi._mm["full_matrix"]["ket"]
        ]

        msg = "    - constructing symmetrized TB Hamiltonian ... "
        self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        Sr_sym = CWModel.construct_Or(s, self._cwi["num_wann"], self._cwi["irvec"], combined_samb_matrix)
        Hr_sym = CWModel.construct_Or(z, self._cwi["num_wann"], self._cwi["irvec"], combined_samb_matrix)
        Hr_nonortho_sym = CWModel.construct_Or(
            z_nonortho, self._cwi["num_wann"], self._cwi["irvec"], combined_samb_matrix
        )
        nr_sym = CWModel.construct_Or(n, self._cwi["num_wann"], self._cwi["irvec"], combined_samb_matrix)

        if self._cwi["tb_gauge"]:
            Sk_sym = CWModel.fourier_transform_r_to_k(
                Sr_sym, self._cwi["kpoints"], self._cwi["irvec"], self._cwi["ndegen"], atoms_frac=atoms_frac_samb
            )
            Hk_sym = CWModel.fourier_transform_r_to_k(
                Hr_sym, self._cwi["kpoints"], self._cwi["irvec"], self._cwi["ndegen"], atoms_frac=atoms_frac_samb
            )
            Hk_nonortho_sym = CWModel.fourier_transform_r_to_k(
                Hr_nonortho_sym,
                self._cwi["kpoints"],
                self._cwi["irvec"],
                self._cwi["ndegen"],
                atoms_frac=atoms_frac_samb,
            )
            nk_sym = CWModel.fourier_transform_r_to_k(
                nr_sym, self._cwi["kpoints"], self._cwi["irvec"], self._cwi["ndegen"], atoms_frac=atoms_frac_samb
            )
        else:
            Sk_sym = CWModel.fourier_transform_r_to_k(
                Sr_sym, self._cwi["kpoints"], self._cwi["irvec"], self._cwi["ndegen"]
            )
            Hk_sym = CWModel.fourier_transform_r_to_k(
                Hr_sym, self._cwi["kpoints"], self._cwi["irvec"], self._cwi["ndegen"]
            )
            Hk_nonortho_sym = CWModel.fourier_transform_r_to_k(
                Hr_nonortho_sym, self._cwi["kpoints"], self._cwi["irvec"], self._cwi["ndegen"]
            )
            nk_sym = CWModel.fourier_transform_r_to_k(
                nr_sym, self._cwi["kpoints"], self._cwi["irvec"], self._cwi["ndegen"]
            )

        self._cwm.log("done", file=self._outfile, mode="a")

        #####

        msg = "    - evaluating fitting accuracy ... "
        self._cwm.log(msg, None, end="\n", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        # sort orbitals
        Hk = sort_ket_matrix(Hk, ket_amn, ket_samb)
        Sk = sort_ket_matrix(Sk, ket_amn, ket_samb)
        nk = sort_ket_matrix(nk, ket_amn, ket_samb)

        Ek_grid, _ = np.linalg.eigh(Hk)
        Ek_grid_sym, _ = np.linalg.eigh(Hk_sym)

        num_k, num_wann = Ek_grid_sym.shape
        Ek_MAE_grid = np.sum(np.abs(Ek_grid_sym - Ek_grid)) / num_k / num_wann * 1000  # [meV]

        # projectability of each Kohn-Sham state in k-space.
        from heapq import nlargest

        Ek = np.array(self._cwi["Ek"], dtype=float)
        Ak = np.array(self._cwi["Ak"], dtype=complex)
        Pk = np.real(np.diagonal(Ak @ Ak.transpose(0, 2, 1).conjugate(), axis1=1, axis2=2))

        Ek_ref = np.zeros((num_k, num_wann))
        for k in range(num_k):
            Pk_ = [(pnk, n) for n, pnk in enumerate(Pk[k])]
            Pk_max_idx_list = sorted([n for _, n in nlargest(num_wann, Pk_)])
            # print(Pk_max_idx_list, Ek[k, Pk_max_idx_list[0]])
            for i, m in enumerate(Pk_max_idx_list):
                Ek_ref[k, i] = Ek[k, m]

        Ek_MAE_grid_DFT = np.sum(np.abs(Ek_grid_sym - Ek_ref)) / num_k / num_wann * 1000  # [meV]

        msg = f"     * MAE of eigen values between CW and Symmetry-Adapted CW models (grid) = {'{:.4f}'.format(Ek_MAE_grid)} [meV] \n"
        msg += f"     * MAE of eigen values between DFT and Symmetry-Adapted CW models (grid) = {'{:.4f}'.format(Ek_MAE_grid_DFT)} [meV] \n"

        molecule = self._cwi._mm.group.is_point_group

        if not molecule:
            Hk_path = CWModel.fourier_transform_r_to_k(
                self["Hr"], self._cwi["kpoints_path"], self._cwi["irvec"], self._cwi["ndegen"]
            )
            Ek_path, _ = np.linalg.eigh(Hk_path)

            if self._cwi["tb_gauge"]:
                Hk_sym_path = CWModel.fourier_transform_r_to_k(
                    Hr_sym,
                    self._cwi["kpoints_path"],
                    self._cwi["irvec"],
                    self._cwi["ndegen"],
                    atoms_frac=atoms_frac_samb,
                )
            else:
                Hk_sym_path = CWModel.fourier_transform_r_to_k(
                    Hr_sym, self._cwi["kpoints_path"], self._cwi["irvec"], self._cwi["ndegen"]
                )

            Ek_path_sym, _ = np.linalg.eigh(Hk_sym_path)

            num_k, num_wann = Ek_path_sym.shape
            Ek_MAE_path = np.sum(np.abs(Ek_path_sym - Ek_path)) / num_k / num_wann * 1000  # [meV]

            msg += f"     * MAE of eigen values between CW and Symmetry-Adapted CW models (path) = {'{:.4f}'.format(Ek_MAE_path)} [meV]\n"
        else:
            Ek_MAE_path = None

        self._cwm.log(msg, None, end="\n", file=self._outfile, mode="a")

        #####
        msg = "    - band energy \n"
        self._cwm.log(msg, None, end="", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        # electronic density matrix elements
        ef = self._cwi["fermi_energy"]
        fk = np.array([fermi(eki - ef, T=0.0) for eki in Ek], dtype=float)
        E_dft = np.sum(Ek * fk) / num_k
        msg = f"     * DFT: {E_dft} \n"

        fk_grid = np.array([fermi(eki - ef, T=0.0) for eki in Ek_grid], dtype=float)
        E_cw = np.sum(Ek_grid * fk_grid) / num_k
        msg = f"     * CW: {E_cw} \n"
        fk_grid_sym = np.array([fermi(eki - ef, T=0.0) for eki in Ek_grid_sym], dtype=float)
        E_scw = np.sum(Ek_grid_sym * fk_grid_sym) / num_k
        msg += f"     * SCW: {E_scw} \n"
        E_scw = np.sum(np.array(list(z.values())) * np.array(list(n.values())))
        msg += f"     * SCW2: {E_scw} \n"

        self._cwm.log(msg, None, end="\n", file=self._outfile, mode="a")
        #####

        #####

        self.update(
            {
                "s": s,
                "n": n,
                "z": z,
                "z_nonortho": z_nonortho,
                "ss": ss,
                #
                "Sk_sym": Sk_sym,
                "nk_sym": nk_sym,
                "Hk_sym": Hk_sym,
                "Hk_nonortho_sym": Hk_nonortho_sym,
                "Sr_sym": Sr_sym,
                "nr_sym": nr_sym,
                "Hr_sym": Hr_sym,
                "Hr_nonortho_sym": Hr_nonortho_sym,
                #
                "Ek_MAE_grid": Ek_MAE_grid,
                "Ek_MAE_path": Ek_MAE_path,
            }
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
            (ndarray, ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(0)|O|φ_{b}(R)>, lattice points.
        """
        return fourier_transform_k_to_r(Ok, kpoints, irvec, atoms_frac)

    # ==================================================
    @classmethod
    def fourier_transform_r_to_k(cls, Or, kpoints, irvec, ndegen=None, atoms_frac=None):
        """
        fourier transformation of an arbitrary operator from real-space representation into k-space representation.

        Args:
        Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(0)|O|φ_{b}(R)>.
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
    def fourier_transform_r_to_k_vec(
        cls, Or_vec, kpoints, irvec, ndegen=None, atoms_frac=None, unit_cell_cart=None, pseudo=False
    ):
        """
        fourier transformation of an arbitrary operator from real-space representation into k-space representation.

        Args:
            Or_vec (ndarray): real-space representation of the given operator, [O_{ab}^{x}(R), O_{ab}^{y}(R), O_{ab}^{z}(R)].
            kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
            irvec (ndarray): irreducible R vectors (crystal coordinate, [[n1,n2,n3]], nj: integer).
            ndegen (ndarray, optional): number of degeneracy at each R.
            atoms_frac (ndarray, optional): atom's position in fractional coordinates.
            unit_cell_cart (ndarray): transform matrix, [a1,a2,a3], [None].
            pseudo (bool, optional): calculate pseudo vector?

        Returns:
            ndarray: k-space representation of the given operator, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
        """
        return fourier_transform_r_to_k_vec(Or_vec, kpoints, irvec, ndegen, atoms_frac, unit_cell_cart, pseudo)

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
            Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(0)|O|φ_{b}(R)>.
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
    def dict_to_matrix(cls, Or_dict, dim=None):
        """
        convert dictionary form to matrix form of an arbitrary operator matrix.

        Args:
            dic (dict): dictionary form of an arbitrary operator matrix in reak-space/k-space representation.
            dim (int, optional): Matrix dimension, [None].

        Returns:
            ndarray: matrix form of the given operator.
            ndarray: lattice or k points.
        """
        return dict_to_matrix(Or_dict, dim)

    # ==================================================
    @classmethod
    def samb_decomp_operator(cls, Or_dict, Zr_dict, ket=None, ket_samb=None):
        """
        decompose arbitrary operator into linear combination of SAMBs.

        Args:
            Or_dict (dict): dictionary form of an arbitrary operator matrix in reak-space/k-space representation.
            Zr_dict (dict): dictionary form of SAMBs.
            ket (list, optional): ket basis list, orbital@site.
            ket_samb (list, optional): ket basis list for SAMBs, orbital@site.

        Returns:
            z (dict): parameter set, {tag: z_j}.
        """
        return samb_decomp_operator(Or_dict, Zr_dict, ket, ket_samb)

    # ==================================================
    @classmethod
    def construct_Or(cls, coeff, num_wann, rpoints, matrix_dict):
        """
        arbitrary operator constructed by linear combination of SAMBs in real-space representation.

        Args:
            coeff (dict): coefficients, {zj: coeff}.
            num_wann (int): # of WFs.
            rpoints (ndarray, optional): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
            matrix_dict (dict): SAMBs.

        Returns:
            ndarray: matrix, [#r, dim, dim].
        """
        return construct_Or(coeff, num_wann, rpoints, matrix_dict)

    # ==================================================
    @classmethod
    def construct_Ok(cls, z, num_wann, kpoints, rpoints, matrix_dict, atoms_frac=None):
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
        return construct_Ok(z, num_wann, kpoints, rpoints, matrix_dict, atoms_frac)

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
    def nnkp(self):
        self._cwi.nnkp

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
    def umat(self):
        self._cwi.umat

    # ==================================================
    @property
    def spn(self):
        self._cwi.spn

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
    def _nk_header(cls):
        return nk_header

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
    def _nr_header(cls):
        return nr_header

    # ==================================================
    @classmethod
    def _z_header(cls):
        return z_header

    # ==================================================
    @classmethod
    def _s_header(cls):
        return s_header

    # ==================================================
    @classmethod
    def _n_header(cls):
        return n_header

    # ==================================================
    @classmethod
    def _O_R_dependence_header(cls):
        return O_R_dependence_header

    # ==================================================
    def write_info_data(self, filename):
        """
        write info and data to seedname.hdf5 (hdf5 format).

        Args:
            filename (str): file name.
        """
        with h5py.File(filename, "w") as hf:
            for group in ("info", "data", "samb_info"):

                if group == "info":
                    data = self._cwi
                elif group == "data":
                    data = self
                else:
                    data = self._samb_info

                group = hf.create_group(group)

                for k, v in data.items():
                    try:
                        if type(v) in (str, list, np.ndarray):
                            dset = group.create_dataset(k, data=v)
                        elif type(v) == bool:
                            dset = group.create_dataset(k, data=v, dtype=bool)
                        else:
                            dset = group.create_dataset(k, data=str(v))
                    except:
                        dset = group.create_dataset(k, data=str(v))

    # ==================================================
    @classmethod
    def read_info_data(self, filename):
        """
        read info and data from seedname.hdf5 (hdf5 format).

        Args:
            filename (str): file name.

        Returns:
            tuple: CWInfo, dictionary of data, dictionary of SAMB info.
        """
        info = {}
        data = {}
        samb_info = {}

        with h5py.File(filename, "r") as hf:
            for d in ("info", "data", "samb_info"):
                for k, v in hf[d].items():
                    v = v[()]

                    if type(v) == bytes:
                        v = v.decode("utf-8")
                        try:
                            v = ast.literal_eval(v)
                        except:
                            v = v

                    if type(v) == np.bool_:
                        v = bool(v)
                    elif type(v) == np.float64:
                        v = float(v)
                    elif type(v) in (list, np.ndarray):
                        v = [vi.decode("utf-8") if type(vi) == bytes else vi for vi in v]

                    if d == "info":
                        info[k] = v
                    elif d == "data":
                        data[k] = v
                    else:
                        samb_info[k] = v

            return info, data, samb_info

    # ==================================================
    def write_or(self, Or, filename, rpoints=None, header=None, vec=False):
        """
        write seedname_or.dat.

        Args:
            Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(0)|O|φ_{b}(R)>.
            filename (str): file name.
            rpoints (ndarray): rpoints.
            header (str, optional): header
            vec (bool, optional): vector ?
        """
        num_wann = self._cwi["num_wann"]
        unit_cell_cart = np.array(self._cwi["unit_cell_cart"])
        Or = np.array(Or)
        Or_str = "# written {}  (created by pw2cw)\n".format(datetime.datetime.now().strftime("on %d%b%Y at %H:%M:%S"))

        # Or_str += " {0[0]:18.15f} {0[1]:18.15f} {0[2]:18.15f}\n".format(unit_cell_cart[0, :])
        # Or_str += " {0[0]:18.15f} {0[1]:18.15f} {0[2]:18.15f}\n".format(unit_cell_cart[1, :])
        # Or_str += " {0[0]:18.15f} {0[1]:18.15f} {0[2]:18.15f}\n".format(unit_cell_cart[2, :])

        if rpoints is None:
            rpoints = np.array(self._cwi["irvec"])
            ndegen = np.array(self._cwi["ndegen"])
            Or_str += "{:12d}\n{:12d}\n".format(num_wann, len(ndegen))
            Or_str += textwrap.fill("".join(["{:5d}".format(x) for x in ndegen]), 75, drop_whitespace=False)
            Or_str += "\n"

        else:
            rpoints = np.array(rpoints)
            ndegen = None
            Or_str += "{:12d}\n".format(num_wann)

        for irpts in range(len(rpoints)):
            for i, j in itertools.product(range(num_wann), repeat=2):
                v = rpoints[irpts, :]
                line = "{:5d}{:5d}{:5d}{:5d}{:5d}  ".format(
                    int(round(v[0])), int(round(v[1])), int(round(v[2])), j + 1, i + 1
                )
                if vec:
                    line += "".join([" {:>15.8E}  {:>15.8E}".format(x.real, x.imag) for x in Or[:, irpts, j, i]])
                else:
                    x = Or[irpts, j, i]
                    line += " {:>15.8E}  {:>15.8E}".format(x.real, x.imag)
                line += "\n"

                Or_str += line

        self._cwm.write(filename, Or_str, header, None)

    # ==================================================
    def write_O_R_dependence(self, Or, filename, header=None):
        """
        write Bond length ||R|| (the 2-norm of lattice vector) dependence of the Frobenius norm of the operator ||O(R)||.
        and the decay length τ [Ang] defined by Exponential-form fitting ||O(R)|| = ||O(0)|| exp(-||R||/τ).

        Args:
            Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(0)|O|φ_{b}(R)>.
            filename (str): file name.
            header (str, optional): header.
        """
        Or = np.array(Or)

        A = self._cwi["A"]
        irvec = self._cwi["irvec"]
        ndegen = self._cwi["ndegen"]
        ef = self._cwi["fermi_energy"]

        # [||R||], [||O(R)||], [max(|O(R)|)], ||O(Rmin)||, ||Rmin||, τ
        R_2_norm_lst, OR_F_norm_lst, OR_abs_max_lst, Omin_F_norm, R_2_norm_min, tau = O_R_dependence(
            Or, A, irvec, ndegen, ef
        )

        O_R_dep_str = "# created by pw2cw \n"
        O_R_dep_str += "# written {}\n".format(datetime.datetime.now().strftime("on %d%b%Y at %H:%M:%S"))
        O_R_dep_str += "# ||Rmin||    = {:<12.8f} [eV] \n".format(R_2_norm_min)
        O_R_dep_str += "# ||O(Rmin)|| = {:<12.8f} [eV] \n".format(Omin_F_norm)
        O_R_dep_str += "# τ           = {:<12.8f} [Ang] \n".format(tau)

        for iR, R_2_norm in enumerate(R_2_norm_lst):
            OR_F_norm = OR_F_norm_lst[iR]
            OR_abs_max = OR_abs_max_lst[iR]

            O_R_dep_str += " {:>15.8E}  {:>15.8E}  {:>15.8E}  \n".format(R_2_norm - R_2_norm_min, OR_F_norm, OR_abs_max)

        self._cwm.write(filename, O_R_dep_str, header, None)

    # ==================================================
    def write_tb(self, Hr, Ar, filename, rpoints=None):
        """
        write seedname_or.dat.

        Args:
            Hr (ndarray): real-space representation of the Hamiltonian, H_{ab}(R) = <φ_{a}(0)|H|φ_{b}(R)>.
            Ar (ndarray): real-space representation of the Hamiltonian, A_{ab}(R) = <φ_{a}(0)|r|φ_{b}(R)>.
            filename (str): file name.
            rpoints (ndarray): rpoints.
        """
        num_wann = self._cwi["num_wann"]
        unit_cell_cart = np.array(self._cwi["unit_cell_cart"])
        Hr = np.array(Hr)
        tb_str = "# written {}  (created by pw2cw)\n".format(datetime.datetime.now().strftime("on %d%b%Y at %H:%M:%S"))

        # tb_str += " {0[0]:18.15f} {0[1]:18.15f} {0[2]:18.15f}\n".format(unit_cell_cart[0, :])
        # tb_str += " {0[0]:18.15f} {0[1]:18.15f} {0[2]:18.15f}\n".format(unit_cell_cart[1, :])
        # tb_str += " {0[0]:18.15f} {0[1]:18.15f} {0[2]:18.15f}\n".format(unit_cell_cart[2, :])

        if rpoints is None:
            rpoints = np.array(self._cwi["irvec"])
            ndegen = np.array(self._cwi["ndegen"])
            tb_str += "{:12d}\n{:12d}\n".format(num_wann, len(ndegen))
            tb_str += textwrap.fill("".join(["{:5d}".format(x) for x in ndegen]), 75, drop_whitespace=False)
            tb_str += "\n\n"

        else:
            rpoints = np.array(rpoints)
            ndegen = None
            tb_str += "{:12d}\n".format(num_wann)

        # _hr
        for irpts in range(len(rpoints)):
            v = rpoints[irpts, :]
            tb_str += "{:5d}{:5d}{:5d}".format(int(round(v[0])), int(round(v[1])), int(round(v[2])))
            tb_str += "\n"
            for i, j in itertools.product(range(num_wann), repeat=2):
                v = rpoints[irpts, :]
                line = "{:5d}{:5d}  ".format(j + 1, i + 1)
                x = Hr[irpts, j, i]
                line += " {:>15.8E}  {:>15.8E}".format(x.real, x.imag)
                line += "\n"

                tb_str += line

            tb_str += "\n"

        # _r
        for irpts in range(len(rpoints)):
            v = rpoints[irpts, :]
            tb_str += "{:5d}{:5d}{:5d}".format(int(round(v[0])), int(round(v[1])), int(round(v[2])))
            tb_str += "\n"
            for i, j in itertools.product(range(num_wann), repeat=2):
                v = rpoints[irpts, :]
                line = "{:5d}{:5d}  ".format(j + 1, i + 1)
                line += "".join([" {:>15.8E}  {:>15.8E}".format(x.real, x.imag) for x in Ar[:, irpts, j, i]])
                line += "\n"

                tb_str += line

            tb_str += "\n"

        self._cwm.write(filename, tb_str, None, None)

    # ==================================================
    def write_samb_coeffs(self, filename, type="", o=None):
        """
        write seedname_z.dat.

        Args:
            filename (str): file name.
            type (str, optional): 'z'/'z_nonortho'/'s'.
            o (dict, optional): coefficients of SAMBs.
        """
        assert type in (
            "",
            "z",
            "z_nonortho",
            "s",
            "n",
            "sx",
            "sy",
            "sz",
        ), f"invalid type = {type} was given. choose from ''/'z'/'z_nonortho'/'s'/'n'."

        if type == "" and o is not None:
            header = ""
            o = o
        elif type == "z":
            header = z_header
            o = self["z"]
        elif type == "z_nonortho":
            header = z_nonortho_header
            o = self["z_nonortho"]
        elif type == "s":
            header = s_header
            o = self["s"]
        elif type == "n":
            header = n_header
            o = self["n"]
        elif type == "sx":
            header = sx_header
            o = self["ss"][0]
        elif type == "sy":
            header = sy_header
            o = self["ss"][1]
        elif type == "sz":
            header = sz_header
            o = self["ss"][2]
        else:
            raise Exception(f"invalid type = {type} was given. choose from 'z'/'z_nonortho'/'s'/'n'/'sx'/'sy'/'sz'.")

        o_str = "# created by pw2cw \n"
        o_str += "# written {}\n".format(datetime.datetime.now().strftime("on %d%b%Y at %H:%M:%S"))

        o_str = "".join(
            [
                "{:>7d}   {:>15}   {:>15}   {:>15.8E} \n ".format(j + 1, zj, self._cwi._mm["combined_id"][zj][0], v)
                for j, (zj, v) in enumerate(o.items())
            ]
        )

        self._cwm.write(filename, o_str, header, None)
