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

import os
import numpy as np

from symclosestwannier.cw.cwin import CWin
from symclosestwannier.cw.cw_info import CWInfo
from symclosestwannier.cw.cw_manager import CWManager
from symclosestwannier.cw.cw_model import CWModel
from symclosestwannier.cw.cw_response import CWResponse
from symclosestwannier.cw.cw_band import CWBand

from symclosestwannier.util.message import (
    cw_start_response_msg,
    cw_end_response_msg,
    cw_start_band_msg,
    cw_end_band_msg,
)


# ==================================================
def cw_analyzer(seedname="cwannier"):
    """
    Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.
    CW TB model can be symmetrized by using Symmetry-Adapted Multipole Basis (SAMB).

    Args:
        seedname (str, optional): seedname.
    """
    cwin = CWin(".", seedname)
    filename = os.path.join(cwin["outdir"], "{}".format(f"{seedname}.hdf5"))

    cwi, dic = CWModel.read_info_data(filename)
    cwi = CWInfo("./", seedname="cwannier", dic=cwi)
    cwm = CWManager(topdir=cwi["outdir"], verbose=cwi["verbose"], parallel=cwi["parallel"], formatter=cwi["formatter"])
    cw_model = CWModel(cwi, cwm, dic)
    cwi = cw_model._cwi

    outfile = f"{cwi['seedname']}.cwpout"

    # ******************** #
    #       Response       #
    # ******************** #

    cwm.log(cw_start_response_msg(), stamp=None, end="\n", file=outfile, mode="w")

    cwr = CWResponse(cwi, cwm)

    cwm.log(cw_end_response_msg(), stamp=None, end="\n", file=outfile, mode="a")

    # ******************** #
    #         Band         #
    # ******************** #

    cwm.log(cw_start_band_msg(), stamp=None, end="\n", file=outfile, mode="a")

    cwb = CWBand(cwi, cwm)

    cwm.log(cw_end_band_msg(), stamp=None, end="\n", file=outfile, mode="a")
