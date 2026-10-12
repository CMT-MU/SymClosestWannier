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

from symclosestwannier.util.get_oper_R import get_oper_R, to_tb_gauge
from symclosestwannier.util.utility import tb_gauge_positions
from symclosestwannier.util.exceptions import SymCWInputError
from symclosestwannier.analyzer.get_response import (
    berry_main,
    boltzwann_main,
    gyrotropic_main,
    kubo_frequencies,
    spin_moment_main,
)

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
        # a given HH_R (Hr_sym, hr_input, ...) may differ from the Hamiltonian from which BB_R and CC_R are built.
        self._HH_R_given = HH_R is not None
        self._HH_R_oper = None

        # operators
        self["HH_R"] = HH_R  # <0n|H|Rm>
        self["AA_R"] = None  # <0n|r|Rm>
        self["v_R"] = None  # <0n|v|Rm>
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

        # self.calc_response()

        # self.calc_spin_moment()

    # ==================================================
    @property
    def operators(self):
        """
        operators in the gauge used by get_response.

        The stored operators are in the wannier90 convention (v_R is built in the gauge given by tb_gauge); for
        tb_gauge = true they are converted here, so that operators added by a later set_operators are converted as well.
        """
        d = {k: self[k] for k in ("HH_R", "AA_R", "v_R", "BB_R", "CC_R", "SS_R", "SR_R", "SHR_R", "SH_R", "SAA_R", "SBB_R")}

        if not self._cwi["tb_gauge"]:
            return d

        keys = ("AA_R", "BB_R", "CC_R", "SS_R", "SR_R", "SHR_R", "SH_R")
        ops = {k: d[k] for k in keys}
        if d["BB_R"] is not None or d["CC_R"] is not None:
            # the corrections of BB_R and CC_R need the Hamiltonian they are built from.
            if not self._HH_R_given:
                ops["HH_R"] = d["HH_R"]
            else:
                if self._HH_R_oper is None:
                    self._HH_R_oper = get_oper_R("HH_R", self._cwi)
                ops["HH_R"] = self._HH_R_oper

        tau = tb_gauge_positions(self._cwi)
        ops = to_tb_gauge(ops, self._cwi["irvec"], self._cwi["unit_cell_cart"], tau, self._cwi["ndegen"])
        d.update({k: ops[k] for k in keys})

        return d

    # ==================================================
    def set_operators(self):
        """
        Wannier matrix elements, allocations and initializations.
        """
        self._cwm.log(cw_start_set_operators_msg(), stamp=None, end="\n", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        if self._cwi["use_tb_approximation"]:
            # only the optical conductivity has a tight-binding version (berry_get_kubo_tb).
            if self._cwi["berry"] and self._cwi["berry_task"] != "kubo":
                raise SymCWInputError(
                    f"use_tb_approximation = true is available only for berry_task = kubo, not {self._cwi['berry_task']}."
                )
            if self._cwi["gyrotropic"]:
                raise SymCWInputError("use_tb_approximation = true is not available for gyrotropic.")

        if self._cwi["use_tb_approximation"]:
            self["HH_R"] = get_oper_R("HH_R", self._cwi) if self["HH_R"] is None else self["HH_R"]

            if self._cwi["berry_task"] == "kubo":
                self["v_R"] = (
                    get_oper_R("v_R", self._cwi, self["HH_R"], tb_gauge_positions(self._cwi))
                    if self["v_R"] is None
                    else self["v_R"]
                )
                self["SS_R"] = (
                    get_oper_R("SS_R", self._cwi) if self._cwi["spin_decomp"] and self["SS_R"] is None else self["SS_R"]
                )

        else:
            # berry
            if self._cwi["berry"]:
                self["HH_R"] = get_oper_R("HH_R", self._cwi) if self["HH_R"] is None else self["HH_R"]

                # (ahc)  Anomalous Hall conductivity (from Berry curvature)
                if self._cwi["berry_task"] == "ahc":
                    self["AA_R"] = get_oper_R("AA_R", self._cwi) if self["AA_R"] is None else self["AA_R"]

                # (morb) Orbital magnetization
                if self._cwi["berry_task"] == "morb":
                    self["AA_R"] = get_oper_R("AA_R", self._cwi) if self["AA_R"] is None else self["AA_R"]
                    self["BB_R"] = get_oper_R("BB_R", self._cwi) if self["BB_R"] is None else self["BB_R"]
                    self["CC_R"] = get_oper_R("CC_R", self._cwi) if self["CC_R"] is None else self["CC_R"]

                # (kubo) Complex optical conductivity (Kubo-Greenwood) & JDOS
                if self._cwi["berry_task"] == "kubo":
                    self["AA_R"] = get_oper_R("AA_R", self._cwi) if self["AA_R"] is None else self["AA_R"]
                    self["SS_R"] = (
                        get_oper_R("SS_R", self._cwi)
                        if self._cwi["spin_decomp"] and self["SS_R"] is None
                        else self["SS_R"]
                    )

                # (sc)   Nonlinear shift current
                if self._cwi["berry_task"] == "sc":
                    self["AA_R"] = get_oper_R("AA_R", self._cwi) if self["AA_R"] is None else self["AA_R"]

                # (shc)  Spin Hall conductivity
                if self._cwi["berry_task"] == "shc":
                    self["AA_R"] = get_oper_R("AA_R", self._cwi) if self["AA_R"] is None else self["AA_R"]
                    self["SS_R"] = get_oper_R("SS_R", self._cwi) if self["SS_R"] is None else self["SS_R"]

                    # if self._cwi["shc_method"] == "qiao":
                    if self["SR_R"] is None:
                        SR_R, SHR_R, SH_R = get_oper_R("SHC_R", self._cwi)
                        self["SR_R"] = SR_R
                        self["SHR_R"] = SHR_R
                        self["SH_R"] = SH_R

                # (kdotp) joint density of states
                if self._cwi["berry_task"] == "kdotp":
                    pass

                # (me) magnetoelectric tensor
                if self._cwi["berry_task"] == "me":
                    self["AA_R"] = get_oper_R("AA_R", self._cwi) if self["AA_R"] is None else self["AA_R"]
                    self["SS_R"] = get_oper_R("SS_R", self._cwi) if self["SS_R"] is None else self["SS_R"]

            # gyrotropic
            if self._cwi["gyrotropic"]:
                win = self._cwi.win

                self["HH_R"] = get_oper_R("HH_R", self._cwi) if self["HH_R"] is None else self["HH_R"]

                if win.eval_D or win.eval_Dw or win.eval_K or win.eval_NOA:
                    self["AA_R"] = get_oper_R("AA_R", self._cwi) if self["AA_R"] is None else self["AA_R"]

                if win.eval_spn:
                    self["SS_R"] = get_oper_R("SS_R", self._cwi) if self["SS_R"] is None else self["SS_R"]

                if win.eval_K:
                    self["BB_R"] = get_oper_R("BB_R", self._cwi) if self["BB_R"] is None else self["BB_R"]
                    self["CC_R"] = get_oper_R("CC_R", self._cwi) if self["CC_R"] is None else self["CC_R"]

        # the Zeeman term needs the spin operator in every response.
        if self._cwi.get("zeeman_interaction") and self["SS_R"] is None:
            self["SS_R"] = get_oper_R("SS_R", self._cwi)

        # spin magnetic moment
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
            ahc_str += "J0 term :" + "{:>20.15f}   {:>20.15f}   {:>20.15f} \n".format(
                ahc_list[0, 0, 0], ahc_list[0, 0, 1], ahc_list[0, 0, 2]
            )
            ahc_str += "J1 term :" + "{:>20.15f}   {:>20.15f}   {:>20.15f} \n".format(
                ahc_list[0, 1, 0], ahc_list[0, 1, 1], ahc_list[0, 1, 2]
            )
            ahc_str += "J2 term :" + "{:>20.15f}   {:>20.15f}   {:>20.15f} \n".format(
                ahc_list[0, 2, 0], ahc_list[0, 2, 1], ahc_list[0, 2, 2]
            )
            ahc_str += "-------------------------------------------\n"
            ahc_str += "Total   :" + "{:>20.15f}   {:>20.15f}   {:>20.15f} \n".format(
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
                ahc_str += "{:>20.15f}   {:>20.15f}   {:>20.15f}   {:>20.15f} \n".format(
                    fermi_energy_list[ife], sum(ahc_list[ife, :, 0]), sum(ahc_list[ife, :, 1]), sum(ahc_list[ife, :, 2])
                )

            filename = f"{self._cwi['seedname']}-ahc-fermiscan.dat"
            self._cwm.write(filename, ahc_str, None, None)

    # ==================================================
    def write_morb(self):
        """
        write seedname-morb.dat (one Fermi energy) or seedname-morb-fermiscan.dat, in Bohr magnetons per cell.
        """
        fermi_energy_list = self._cwi["fermi_energy_list"]
        LC, IC, M = (self[k].sum(axis=1) for k in ("morb_LC", "morb_IC", "morb"))

        if len(fermi_energy_list) == 1:
            morb_str = "M_orb (bohr magn/cell)        x                      y                      z \n"
            morb_str += "======================\n"
            for label, v in (("Local circulation :", LC[0]), ("Itinerant circulation:", IC[0]), ("Total   :", M[0])):
                morb_str += "{:<22s}".format(label) + "".join(["{:>22.15f} ".format(x) for x in v]) + "\n"
            filename = f"{self._cwi['seedname']}-morb.dat"
        else:
            morb_str = "".join(
                ["{:>20.15f}   {:>20.15f}   {:>20.15f}   {:>20.15f} \n".format(ef, *v) for ef, v in zip(fermi_energy_list, M)]
            )
            filename = f"{self._cwi['seedname']}-morb-fermiscan.dat"

        self._cwm.write(filename, morb_str, None, None)

    # ==================================================
    def write_sc(self):
        """
        write seedname-sc_abc.dat, the shift current sigma_abc(omega) in A/V^2 (sigma_abc = sigma_acb).
        """
        xyz = "xyz"
        for a in range(3):
            for bc, (b, c) in enumerate(zip((0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2))):
                sc_str = "".join(["{:>20.15f}   {:>24.15E} \n".format(w, v) for w, v in zip(self["sc_freq"], self["sc"][a, bc])])
                filename = f"{self._cwi['seedname']}-sc_{xyz[a]}{xyz[b]}{xyz[c]}.dat"
                self._cwm.write(filename, sc_str, None, None)

    # ==================================================
    def write_kubo(self):
        """
        write seedname-kubo_H_*.dat, seedname-kubo_A_*.dat.

        """
        kubo_freq_list = kubo_frequencies(self._cwi)

        d = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}

        for k, (i, j) in d.items():
            kubo_H_ij = self["kubo_H"][:, i, j]
            kubo_H_ji = self["kubo_H"][:, j, i]
            kubo_AH_ij = self["kubo_AH"][:, i, j]
            kubo_AH_ji = self["kubo_AH"][:, j, i]

            kubo_S_str = "".join(
                [
                    "{:>20.15f}   {:>20.15f}   {:>20.15f} \n ".format(
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
                    "{:>20.15f}   {:>20.15f}   {:>20.15f} \n ".format(
                        o, 0.5 * (AH_ij - AH_ji).real, 0.5 * (H_ij - H_ji).imag
                    )
                    for o, H_ij, H_ji, AH_ij, AH_ji in zip(kubo_freq_list, kubo_H_ij, kubo_H_ji, kubo_AH_ij, kubo_AH_ji)
                ]
            )

            filename_A = f"{self._cwi['seedname']}-kubo_A_{k}.dat"
            self._cwm.write(filename_A, kubo_A_str, None, None)

    # ==================================================
    def write_shc(self):
        """
        write seedname-shc.dat, seedname-shc-fermiscan.dat.
        """
        if self._cwi["shc_freq_scan"]:
            shc_freq = self["shc_freq"]
            kubo_freq_list = kubo_frequencies(self._cwi)
            kubo_nfreq = len(kubo_freq_list)

            shc_str = "#No.   Frequency(eV)   Re(sigma)((hbar/e)*S/cm)   Im(sigma)((hbar/e)*S/cm) \n"
            for ifreq in range(kubo_nfreq):
                shc_str += "{:>20.15f}   {:>20.15f}   {:>20.15f}   {:>20.15f} \n".format(
                    ifreq, np.real(kubo_freq_list[ifreq]), np.real(shc_freq[ifreq]), np.imag(shc_freq[ifreq])
                )

            filename = f"{self._cwi['seedname']}-shc-freqscan.dat"
            self._cwm.write(filename, shc_str, None, None)
        else:
            shc_fermi = self["shc_fermi"]

            shc_str = "#No.   Fermi energy(eV)   SHC((hbar/e)*S/cm) \n"
            num_fermi = self._cwi["num_fermi"]
            fermi_energy_list = self._cwi["fermi_energy_list"]
            for ife in range(num_fermi):
                shc_str += "{:>20.15f}   {:>20.15f}   {:>20.15f} \n".format(
                    ife, fermi_energy_list[ife], np.real(shc_fermi[ife])
                )

            filename = f"{self._cwi['seedname']}-shc-fermiscan.dat"
            self._cwm.write(filename, shc_str, None, None)

    # ==================================================
    def write_gyro_K(self):
        """
        write *.dat.
        """

        def write(f_out_name_tmp, units_tmp, comment_tmp, spin=False):
            if spin:
                gyro_K = self[f"gyro_K_spn"]
            else:
                gyro_K = self[f"gyro_K_orb"]

            if gyro_K is not None:
                filename = f"{self._cwi['seedname']}-gyrotropic-{f_out_name_tmp}.dat"
                gyro_K_str = f"# {comment_tmp} \n"
                gyro_K_str += f"# in units of [ {units_tmp} ] \n"

                # Symmetric part
                xx = gyro_K[0, 0, :]
                yy = gyro_K[1, 1, :]
                zz = gyro_K[2, 2, :]
                xy = (gyro_K[0, 1, :] + gyro_K[1, 0, :]) * 0.5
                xz = (gyro_K[0, 2, :] + gyro_K[2, 0, :]) * 0.5
                yz = (gyro_K[1, 2, :] + gyro_K[2, 1, :]) * 0.5
                # Antisymmetric part, in polar-vector form
                x = (gyro_K[1, 2, :] - gyro_K[2, 1, :]) * 0.5
                y = (gyro_K[2, 0, :] - gyro_K[0, 2, :]) * 0.5
                z = (gyro_K[0, 1, :] - gyro_K[1, 0, :]) * 0.5

                gyro_K_str += f"#                             |                                      symmetric part                                     ||              asymmetric part              | \n"
                gyro_K_str += "# EFERMI(eV)      omega(eV)             xx             yy             zz             xy             xz             yz              x              y              z \n"

                for i in range(self._cwi["num_fermi"]):
                    gyro_K_str += "{:>20.15e}   {:>20.15e}   {:>20.15e}   {:>20.15e}   {:>20.15e}   {:>20.15e}   {:>20.15e}   {:>20.15e}   {:>20.15e}   {:>20.15e}   {:>20.15e}\n".format(
                        self._cwi["fermi_energy_list"][i],
                        0.0,
                        xx[i],
                        yy[i],
                        zz[i],
                        xy[i],
                        xz[i],
                        yz[i],
                        x[i],
                        y[i],
                        z[i],
                    )

                self._cwm.write(filename, gyro_K_str, None, None)

        f_out_name_tmp = "K_spin"
        units_tmp = "Ampere"
        comment_tmp = "spin part of the K tensor -- Eq. 3 of TAS17"
        write(f_out_name_tmp, units_tmp, comment_tmp, spin=True)

        f_out_name_tmp = "K_orb"
        units_tmp = "Ampere"
        comment_tmp = "orbital part of the K tensor -- Eq. 3 of TAS17"
        write(f_out_name_tmp, units_tmp, comment_tmp, spin=False)

    # ==================================================
    def write_spin(self):
        """
        write seedname-spin.dat.

        Args:
            filename (str): file name.
        """
        spin_str = "{:>20.15f}   {:>20.15f}   {:>20.15f} \n ".format(self["Ms_x"], self["Ms_y"], self["Ms_z"])
        filename = f"{self._cwi['seedname']}-spin.dat"
        self._cwm.write(filename, spin_str, None, None)
