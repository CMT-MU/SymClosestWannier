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

from symclosestwannier.cw.cw_info import CWInfo
from symclosestwannier.cw.cw_manager import CWManager
from symclosestwannier.util.get_oper_R import get_oper_R
from symclosestwannier.analyzer.get_response import berry_main, boltzwann_main, gyrotropic_main

from symclosestwannier.util.message import (
    cw_start_set_operators_msg,
    cw_end_set_operators_msg,
    cw_start_response_msg,
    cw_end_response_msg,
)

# default parameters
parameters = {
    "berry": False,
    "berry_task": "",
    "berry_kmesh": None,
    "fermi_energy": None,
    "fermi_energy_min": None,
    "fermi_energy_max": None,
    "fermi_energy_step": None,
    "use_ws_distance": True,
    "transl_inv": True,
    "__wb_fft_lib": "fftw",
}


# ==================================================
class Response(dict):
    """
    Analyze external responses of Closest Wannier (CW) tight-binding (TB) model.

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
            cwi (SystemInfo): CWInfo.
            cwm (CWManager): CWManager.
        """
        super().__init__()

        self._cwi = cwi
        self._cwm = cwm
        self._outfile = f"{self._cwi['seedname']}.cwpout"

        # operators
        self["HH_R"] = None  # <0n|H|Rm>
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
        self["kubo_H_k"] = None
        self["kubo_H"] = None
        self["kubo_AH_k"] = None
        self["kubo_AH"] = None

        self.set_operators()

        self.calc_response()

    # ==================================================
    def set_operators(self):
        """
        Wannier matrix elements, allocations and initializations.
        """
        self._cwm.log(cw_start_set_operators_msg(), stamp=None, end="\n", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        # (ahc)  Anomalous Hall conductivity (from Berry curvature)
        if self._cwi["berry_task"] == "ahc":
            self["HH_R"] = get_oper_R("HH_R", self._cwi, tb_gauge=True) if self["HH_R"] is None else None
            self["AA_R"] = get_oper_R("AA_R", self._cwi, tb_gauge=True) if self["AA_R"] is None else None

        # (morb) Orbital magnetization
        if self._cwi["berry_task"] == "morb":
            self["HH_R"] = get_oper_R("HH_R", self._cwi, tb_gauge=True) if self["HH_R"] is None else None
            self["AA_R"] = get_oper_R("AA_R", self._cwi, tb_gauge=True) if self["AA_R"] is None else None
            self["BB_R"] = get_oper_R("BB_R", self._cwi, tb_gauge=True) if self["BB_R"] is None else None
            self["CC_R"] = get_oper_R("CC_R", self._cwi, tb_gauge=True) if self["CC_R"] is None else None

        # (kubo) Complex optical conductivity (Kubo-Greenwood) & JDOS
        if self._cwi["berry_task"] == "kubo":
            self["HH_R"] = get_oper_R("HH_R", self._cwi, tb_gauge=True) if self["HH_R"] is None else None
            self["AA_R"] = get_oper_R("AA_R", self._cwi, tb_gauge=True) if self["AA_R"] is None else None
            if self._cwi["spin_decomp"]:
                self["SS_R"] = get_oper_R("SS_R", self._cwi, tb_gauge=True) if self["SS_R"] is None else None

        # (sc)   Nonlinear shift current
        if self._cwi["berry_task"] == "sc":
            self["HH_R"] = get_oper_R("HH_R", self._cwi, tb_gauge=True) if self["HH_R"] is None else None
            self["AA_R"] = get_oper_R("AA_R", self._cwi, tb_gauge=True) if self["AA_R"] is None else None

        # (shc)  Spin Hall conductivity
        if self._cwi["berry_task"] == "shc":
            self["HH_R"] = get_oper_R("HH_R", self._cwi, tb_gauge=True) if self["HH_R"] is None else None
            self["AA_R"] = get_oper_R("AA_R", self._cwi, tb_gauge=True) if self["AA_R"] is None else None
            self["SS_R"] = get_oper_R("SS_R", self._cwi, tb_gauge=True) if self["SS_R"] is None else None

            if self._cwi["shc_method"] == "qiao":
                self["SHC_R"] = get_oper_R("SHC_R", self._cwi, tb_gauge=True) if self["SHC_R"] is None else None
            else:  # ryoo
                self["SAA_R"] = get_oper_R("SAA_R", self._cwi, tb_gauge=True) if self["SAA_R"] is None else None
                self["SBB_R"] = get_oper_R("SBB_R", self._cwi, tb_gauge=True) if self["SBB_R"] is None else None

        if self._cwi["berry_task"] == "kdotp":
            self["HH_R"] = get_oper_R("HH_R", self._cwi, tb_gauge=True) if self["HH_R"] is None else None

        self._cwm.log(cw_end_set_operators_msg(), stamp=None, end="\n", file=self._outfile, mode="a")

    # ==================================================
    def calc_response(self):
        self._cwm.log(cw_start_response_msg(), stamp=None, end="\n", file=self._outfile, mode="a")
        self._cwm.set_stamp()

        if self._cwi["berry"]:
            _ = berry_main(self._cwi, self.operators)

        if self._cwi["gyrotropic"]:
            boltzwann_main()

        if self._cwi["boltzwann"]:
            gyrotropic_main()

        self._cwm.log(cw_end_response_msg(), stamp=None, end="\n", file=self._outfile, mode="a")

    # ==================================================
    @property
    def operators(self):
        return {k: self[k] for k in ("HH_R", "AA_R", "BB_R", "CC_R", "SS_R", "SR_R", "SHR_R", "SH_R", "SAA_R", "SBB_R")}
