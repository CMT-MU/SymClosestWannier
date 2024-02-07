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
#                       band: band properties                        #
#                                                                    #
# ****************************************************************** #

import numpy as np

from symclosestwannier.cw.cw_info import CWInfo
from symclosestwannier.cw.cw_manager import CWManager
from symclosestwannier.util.get_oper_R import get_HH_R, get_AA_R, get_SS_R
from symclosestwannier.analyzer.get_band import dos_main, k_path, k_slice, spin_get_moment, geninterp_main


# ==================================================
class Band(dict):
    """
    Analyze band related properties of Closest Wannier (CW) tight-binding (TB) model.

    Attributes:
        _cwi (CWInfo): CWInfo.
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
        self._outfile = f"{self._cwi['seedname']}.cwpout"

        self.set_operators()

        self.calc_band()

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
    def calc_band(self):
        """Now perform one or more of the following tasks"""

        # *************************************************************************** #
        #       Density of states calculated using a uniform interpolation mesh       #
        # *************************************************************************** #
        dos_main()

        # *************************************************************************** #
        #     Bands, Berry curvature, or orbital magnetization plot along a k-path    #
        # *************************************************************************** #
        k_path()

        # *************************************************************************** #
        # Bands, Berry curvature, or orbital magnetization plot on a slice in k-space #
        # *************************************************************************** #
        k_slice()

        # *************************************************************************** #
        #                            Spin magnetic moment                             #
        # *************************************************************************** #
        spin_get_moment()

        # *************************************************************************** #
        #                             Bamd interpolation                              #
        # *************************************************************************** #
        geninterp_main()
