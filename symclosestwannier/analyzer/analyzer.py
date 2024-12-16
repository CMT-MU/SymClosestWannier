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
#                 analyzer: analyze Wannier TB model                 #
#                                                                    #
# ****************************************************************** #

import os
import numpy as np

from symclosestwannier.cw.win import Win
from symclosestwannier.cw.cwin import CWin
from symclosestwannier.cw.cw_info import CWInfo
from symclosestwannier.cw.cw_manager import CWManager
from symclosestwannier.cw.cw_model import CWModel
from symclosestwannier.analyzer.response import Response
from symclosestwannier.analyzer.band import Band

from symclosestwannier.util.message import (
    cw_open_msg,
    postcw_end_msg,
    system_msg,
    cw_start_output_msg,
    cw_end_output_msg,
)

from symclosestwannier.util.utility import sort_ket_matrix


# ==================================================
def analyzer(seedname="cwannier"):
    """
    Analyze Wannier TB model.

    Args:
        seedname (str, optional): seedname.
    """
    cwin = CWin(".", seedname)
    cwm = CWManager(
        topdir=cwin["outdir"], verbose=cwin["verbose"], parallel=cwin["parallel"], formatter=cwin["formatter"]
    )

    filename = os.path.join(cwin["outdir"], "{}".format(f"{seedname}.hdf5"))
    info, data, samb_info = CWModel.read_info_data(filename)

    cwi = CWInfo("./", seedname, dic=info, postcw=True)
    cwi |= cwin | Win(".", seedname)

    cw_model = CWModel(cwi, cwm, samb_info, dic=data)
    cwi = cw_model._cwi

    outfile = f"{seedname}.cwpout"

    cwm.log(cw_open_msg(), stamp=None, end="\n", file=outfile, mode="w")
    cwm.log(system_msg(cwi), stamp=None, end="\n", file=outfile, mode="a")

    # ******************** #
    #       Response       #
    # ******************** #

    if type(cw_model["Hr"]) == np.ndarray:
        Hr = np.array(cw_model["Hr"], dtype=np.complex128)
    else:
        Hr = None

    if cwi["symmetrization"]:
        if cw_model["Hr_sym"] is not None:
            Hr = np.array(cw_model["Hr_sym"], dtype=np.complex128)
            ket_samb = samb_info["ket"]
            ket_amn = cwi.get("ket_amn", ket_samb)
            Hr = sort_ket_matrix(Hr, ket_samb, ket_amn)

    res = Response(cwi, cwm, HH_R=Hr)

    # ******************** #
    #         Band         #
    # ******************** #

    band = Band(cwi, cwm)

    # ******************** #
    #        Output        #
    # ******************** #

    cwm.log(cw_start_output_msg(), stamp=None, end="\n", file=outfile, mode="a")
    cwm.set_stamp()

    if cwi["berry"]:
        if cwi["berry_task"] == "ahc":
            res.write_ahc()

        if cwi["berry_task"] == "kubo":
            res.write_kubo()

        if cwi["berry_task"] == "shc":
            res.write_shc()

    if cwi["gyrotropic"]:
        if cwi.win.eval_K or cwi.win.eval_spn:
            res.write_gyro_K()

    if cwi["spin_moment"]:
        res.write_spin()

    cwm.log(f"\n\n  * total elapsed_time:", file=outfile, mode="a")
    cwm.log(cw_end_output_msg(), stamp=None, end="\n", file=outfile, mode="a")

    cwm.log(f"  * total elapsed_time:", stamp="start", file=outfile, mode="a")
    cwm.log(postcw_end_msg(), stamp=None, end="\n", file=outfile, mode="a")

    return res, band
