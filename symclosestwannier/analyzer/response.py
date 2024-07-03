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
#               response: external response properties               #
#                                                                    #
# ****************************************************************** #

import numpy as np

from symclosestwannier.util.get_oper_R import get_oper_R
from symclosestwannier.analyzer.get_response import berry_main, boltzwann_main, gyrotropic_main, spin_moment_main

from symclosestwannier.util.message import (
    cw_start_set_operators_msg,
    cw_end_set_operators_msg,
    cw_start_response_msg,
    cw_end_response_msg,
    cw_start_expectation_msg,
    cw_end_expectation_msg,
)


# ==================================================
class Response(dict):
    """
    Analyze external responses of Closest Wannier (CW) tight-binding (TB) model.

    Attributes:
        _cwi (CWInfo): CWInfo.
        _cwm (CWManager): CWManager.
        _outfile (str): output file, seedname.cwout.
    """

    # ==================================================
    def __init__(self, cwi, cwm, HH_R=None):
        """
        Analyze external responses of Closest Wannier (CW) tight-binding (TB) model.

        Args:
            cwi (CWInfo): CWInfo.
            cwm (CWManager): CWManager.
            HH_R (ndarray): matrix elements of real-space Hamiltonian, <0n|H|Rm>.
        """
        super().__init__()

        self._cwi = cwi
        self._cwm = cwm
        self._outfile = f"{self._cwi['seedname']}.cwpout"

        # operators
        self["HH_R"] = HH_R  # <0n|H|Rm>
        self["AA_R"] = None  # <0n|r|Rm>
        self["BB_R"] = None  # <0|H(r-R)|R>
        self["CC_R"] = None  # <0|r_alpha.H(r-R)_beta|R>
        self["SS_R"] = None  # <0n|sigma_x,y,z|Rm>
        self["SR_R"] = None  # <0n|sigma_x,y,z.(r-R)_alpha|Rm>
        self["SHR_R"] = None  # <0n|sigma_x,y,z.H.(r-R)_alpha|Rm>
        self["SH_R"] = None  # <0n|sigma_x,y,z.H|Rm>
        self["SAA_R"] = None  # <0n|sigma_x,y,z.(r-R)_alpha|Rm>
        self["SBB_R"] = None  # <0n|sigma_x,y,z.H.(r-R)_alpha|Rm>

        # responses
        # kubo
        self["kubo_H"] = None
        self["kubo_H_spn"] = None
        self["kubo_AH"] = None
        self["kubo_AH_spn"] = None

        # ahc
        self["ahc_list"] = None

        # me
        self["me_H_spn"] = None
        self["me_H_orb"] = None
        self["me_AH_spn"] = None
        self["me_AH_orb"] = None

        # expectation values
        self["Ms_x"] = None
        self["Ms_y"] = None
        self["Ms_z"] = None

        self.set_operators()

        self.calc_response()

        self.calc_spin_moment()

    # ==================================================
    def set_operators(self):
        """
        Wannier matrix elements, allocations and initializations.
        """
        self._cwm.log(cw_start_set_operators_msg(), stamp=None, end="\n", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        # (ahc)  Anomalous Hall conductivity (from Berry curvature)
        if self._cwi["berry_task"] == "ahc":
            if self["HH_R"] is None:
                self["HH_R"] = get_oper_R("HH_R", self._cwi)
            if self["AA_R"] is None:
                self["AA_R"] = get_oper_R("AA_R", self._cwi)

        # (morb) Orbital magnetization
        if self._cwi["berry_task"] == "morb":
            if self["HH_R"] is None:
                self["HH_R"] = get_oper_R("HH_R", self._cwi)
            if self["AA_R"] is None:
                self["AA_R"] = get_oper_R("AA_R", self._cwi)
            if self["BB_R"] is None:
                self["BB_R"] = get_oper_R("BB_R", self._cwi)
            if self["CC_R"] is None:
                self["CC_R"] = get_oper_R("CC_R", self._cwi)

        # (kubo) Complex optical conductivity (Kubo-Greenwood) & JDOS
        if self._cwi["berry_task"] == "kubo":
            if self["HH_R"] is None:
                self["HH_R"] = get_oper_R("HH_R", self._cwi)
            if self["AA_R"] is None:
                self["AA_R"] = get_oper_R("AA_R", self._cwi)
            if self._cwi["spin_decomp"] and self["SS_R"] is None:
                self["SS_R"] = get_oper_R("SS_R", self._cwi)

        # (sc)   Nonlinear shift current
        if self._cwi["berry_task"] == "sc":
            if self["HH_R"] is None:
                self["HH_R"] = get_oper_R("HH_R", self._cwi)
            if self["AA_R"] is None:
                self["AA_R"] = get_oper_R("AA_R", self._cwi)

        # (shc)  Spin Hall conductivity
        if self._cwi["berry_task"] == "shc":
            if self["HH_R"] is None:
                self["HH_R"] = get_oper_R("HH_R", self._cwi)
            if self["AA_R"] is None:
                self["AA_R"] = get_oper_R("AA_R", self._cwi)
            if self["SS_R"] is None:
                self["SS_R"] = get_oper_R("SS_R", self._cwi)

            if self._cwi["shc_method"] == "qiao":
                if self["SHC_R"] is None:
                    self["SHC_R"] = get_oper_R("SHC_R", self._cwi)
            else:  # ryoo
                if self["SAA_R"] is None:
                    self["SAA_R"] = get_oper_R("SAA_R", self._cwi)
                if self["SBB_R"] is None:
                    self["SBB_R"] = get_oper_R("SBB_R", self._cwi)

        if self._cwi["berry_task"] == "kdotp":
            if self["HH_R"] is None:
                self["HH_R"] = get_oper_R("HH_R", self._cwi)

        # (me) magnetoelectric tensor
        if self._cwi["berry_task"] == "me":
            if self["HH_R"] is None:
                self["HH_R"] = get_oper_R("HH_R", self._cwi)
            if self["AA_R"] is None:
                self["AA_R"] = get_oper_R("AA_R", self._cwi)
            if self["SS_R"] is None:
                self["SS_R"] = get_oper_R("SS_R", self._cwi)

        if self._cwi["spin_moment"]:
            if self["SS_R"] is None:
                self["SS_R"] = get_oper_R("SS_R", self._cwi)

        self._cwm.log(cw_end_set_operators_msg(), stamp=None, end="\n", file=self._outfile, mode="a")

    # ==================================================
    def calc_response(self):
        self._cwm.log(cw_start_response_msg(), stamp=None, end="\n", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        if self._cwi["berry"]:
            self.update(berry_main(self._cwi, self.operators))

        if self._cwi["gyrotropic"]:
            self.update(gyrotropic_main(self._cwi, self.operators))

        if self._cwi["boltzwann"]:
            self.update(boltzwann_main(self._cwi, self.operators))

        self._cwm.log(cw_end_response_msg(), stamp=None, end="\n", file=self._outfile, mode="a")

    # ==================================================
    def calc_spin_moment(self):
        self._cwm.log(cw_start_expectation_msg(), stamp=None, end="\n", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        if self._cwi["spin_moment"]:
            self.update(spin_moment_main(self._cwi, self.operators))

        self._cwm.log(cw_end_expectation_msg(), stamp=None, end="\n", file=self._outfile, mode="a")

    # ==================================================
    @property
    def operators(self):
        return {k: self[k] for k in ("HH_R", "AA_R", "BB_R", "CC_R", "SS_R", "SR_R", "SHR_R", "SH_R", "SAA_R", "SBB_R")}

    # ==================================================
    def write_ahc(self):
        """
        write seedname-ahc.dat, seedname-ahc-fermiscan.dat.
        """
        if self._cwi["num_fermi"] == 0:
            pass
        elif self._cwi["num_fermi"] == 1:
            ahc_list = self["ahc_list"]
            ahc_str = "AHC (S/cm)       x          y          z \n"
            ahc_str += "==========\n"
            ahc_str += "J0 term :" + "{:>15.8f}   {:>15.8f}   {:>15.8f} \n".format(
                ahc_list[0, 0, 0], ahc_list[0, 0, 1], ahc_list[0, 0, 2]
            )
            ahc_str += "J1 term :" + "{:>15.8f}   {:>15.8f}   {:>15.8f} \n".format(
                ahc_list[0, 1, 0], ahc_list[0, 1, 1], ahc_list[0, 1, 2]
            )
            ahc_str += "J2 term :" + "{:>15.8f}   {:>15.8f}   {:>15.8f} \n".format(
                ahc_list[0, 2, 0], ahc_list[0, 2, 1], ahc_list[0, 2, 2]
            )
            ahc_str += "-------------------------------------------\n"
            ahc_str += "Total   :" + "{:>15.8f}   {:>15.8f}   {:>15.8f} \n".format(
                sum(ahc_list[0, :, 0]), sum(ahc_list[0, :, 1]), sum(ahc_list[0, :, 2])
            )

            filename = f"{self._cwi['seedname']}-ahc.dat"
            self._cwm.write(filename, ahc_str, None, None)
        else:
            ahc_list = self["ahc_list"]
            num_fermi = self._cwi["num_fermi"]
            fermi_energy_list = self._cwi["fermi_energy_list"]
            ahc_str = ""
            for ife in range(num_fermi):
                ahc_str += "{:>15.8f}   {:>15.8f}   {:>15.8f}   {:>15.8f} \n".format(
                    fermi_energy_list[ife], sum(ahc_list[ife, :, 0]), sum(ahc_list[ife, :, 1]), sum(ahc_list[ife, :, 2])
                )

            filename = f"{self._cwi['seedname']}-ahc-fermiscan.dat"
            self._cwm.write(filename, ahc_str, None, None)

    # ==================================================
    def write_kubo(self):
        """
        write seedname-kubo_H_*.dat, seedname-kubo_A_*.dat.

        Args:
            filename (str): file name.
        """
        kubo_freq_list = np.arange(self._cwi["kubo_freq_min"], self._cwi["kubo_freq_max"], self._cwi["kubo_freq_step"])

        d = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}

        for k, (i, j) in d.items():
            kubo_H_ij = self["kubo_H"][:, i, j]
            kubo_H_ji = self["kubo_H"][:, j, i]
            kubo_AH_ij = self["kubo_AH"][:, i, j]
            kubo_AH_ji = self["kubo_AH"][:, j, i]

            kubo_S_str = "".join(
                [
                    "{:>15.8f}   {:>15.8f}   {:>15.8f} \n ".format(
                        o, 0.5 * (H_ij + H_ji).real, 0.5 * (AH_ij + AH_ji).imag
                    )
                    for o, H_ij, H_ji, AH_ij, AH_ji in zip(kubo_freq_list, kubo_H_ij, kubo_H_ji, kubo_AH_ij, kubo_AH_ji)
                ]
            )

            filename_S = f"{self._cwi['seedname']}-kubo_S_{k}.dat"
            self._cwm.write(filename_S, kubo_S_str, None, None)

        d = {"yz": (1, 2), "zx": (2, 0), "xy": (0, 1)}

        for k, (i, j) in d.items():
            kubo_H_ij = self["kubo_H"][:, i, j]
            kubo_H_ji = self["kubo_H"][:, j, i]
            kubo_AH_ij = self["kubo_AH"][:, i, j]
            kubo_AH_ji = self["kubo_AH"][:, j, i]

            kubo_A_str = "".join(
                [
                    "{:>15.8f}   {:>15.8f}   {:>15.8f} \n ".format(
                        o, 0.5 * (AH_ij - AH_ji).real, 0.5 * (H_ij - H_ji).imag
                    )
                    for o, H_ij, H_ji, AH_ij, AH_ji in zip(kubo_freq_list, kubo_H_ij, kubo_H_ji, kubo_AH_ij, kubo_AH_ji)
                ]
            )

            filename_A = f"{self._cwi['seedname']}-kubo_A_{k}.dat"
            self._cwm.write(filename_A, kubo_A_str, None, None)

    # ==================================================
    def write_spin(self):
        """
        write seedname-spin.dat.

        Args:
            filename (str): file name.
        """
        spin_str = "{:>15.8f}   {:>15.8f}   {:>15.8f} \n ".format(self["Ms_x"], self["Ms_y"], self["Ms_z"])
        filename = f"{self._cwi['seedname']}-spin.dat"
        self._cwm.write(filename, spin_str, None, None)
