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
#           cw_analyzer: analyze Closest Wannier TB model            #
#                                                                    #
# ****************************************************************** #

import numpy as np

from symclosestwannier.cw.cw_info import CWInfo
from symclosestwannier.cw.cw_manager import CWManager
from symclosestwannier.cw.get_matrix_R import get_HH_R, get_AA_R, get_SS_R
from symclosestwannier.cw.get_response import berry_main, boltzwann_main, gyrotropic_main

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
class CWResponse(dict):
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
        self._cwi = cwi
        self._cwm = cwm
        self._outfile = f"{self._cwi['seedname']}.cwpout"

        self.set_operators()

        self.calc_response()

    # ==================================================
    def set_operators(self):
        # <0n|H|Rm>
        self.HH_R = get_HH_R(self._cwi)

        # <0n|r|Rm>
        self.AA_R = get_AA_R(self._cwi)

        # <0|H(r-R)|R>
        self.BB_R = None

        # <0|r_alpha.H(r-R)_beta|R>
        self.CC_R = None

        # <0n|sigma_x,y,z|Rm>
        if self._cwi.get("Sk", None):
            self.SS_R = get_SS_R(self._cwi)

        # <0n|sigma_x,y,z.(r-R)_alpha|Rm>
        self.SR_R = None

        # <0n|sigma_x,y,z.H.(r-R)_alpha|Rm>
        self.SHR_R = None

        # <0n|sigma_x,y,z.H|Rm>
        self.SH_R = None

        # <0n|sigma_x,y,z.(r-R)_alpha|Rm>
        self.SAA_R = None

        # <0n|sigma_x,y,z.H.(r-R)_alpha|Rm>
        self.SBB_R = None

    # ==================================================
    def calc_response(self):
        """Now perform one or more of the following tasks"""

        if self._cwi["berry"]:
            berry_main()

        if self._cwi["gyrotropic"]:
            boltzwann_main()

        if self._cwi["boltzwann"]:
            gyrotropic_main()
