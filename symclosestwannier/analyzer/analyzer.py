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
from symclosestwannier.util.hr_utility import read_hr
from symclosestwannier.util.exceptions import SymCWInputError


# ==================================================
def analyzer(seedname="cwannier"):
    """
    Analyze Wannier TB model.

    Args:
        seedname (str, optional): seedname.

    Returns:
        tuple: Response, Band.
    """
    # input files are read from the current directory, CWManager moves to outdir.
    # the current directory is restored afterwards.
    indir = os.getcwd()
    try:
        return _analyzer(seedname, indir)
    finally:
        os.chdir(indir)


# ==================================================
def _analyzer(seedname, indir):
    """
    Args:
        seedname (str): seedname.
        indir (str): directory of input files.
    """
    cwin = CWin(indir, seedname)
    cwm = CWManager(
        topdir=cwin["outdir"], verbose=cwin["verbose"], parallel=cwin["parallel"], formatter=cwin["formatter"]
    )

    filename = os.path.join(cwin["outdir"], "{}".format(f"{seedname}.hdf5"))
    info, data, samb_info = CWModel.read_info_data(filename)

    cwi = CWInfo(indir, seedname, dic=info, postcw=True)
    cwi |= cwin | Win(indir, seedname)

    cw_model = CWModel(cwi, cwm, samb_info, dic=data)
    cwi = cw_model._cwi

    outfile = f"{seedname}.cwpout"

    cwm.log(cw_open_msg(), stamp=None, end="\n", file=outfile, mode="w")
    cwm.log(system_msg(cwi), stamp=None, end="\n", file=outfile, mode="a")

    # ******************** #
    #      Hamiltonian     #
    # ******************** #

    Hr = None

    if isinstance(cw_model["Hr"], np.ndarray):
        Hr = np.array(cw_model["Hr"], dtype=np.complex128)

    if cwi["symmetrization"]:
        if cw_model["Hr_sym"] is not None:
            Hr = np.array(cw_model["Hr_sym"], dtype=np.complex128)
            ket_samb = samb_info["ket"]
            ket_amn = cwi.get("ket_amn", ket_samb)
            Hr = sort_ket_matrix(Hr, ket_samb, ket_amn)

    if cwi["hr_input"] != "":
        Hr, irvec, ndegen = read_hr(os.path.join(indir, cwi["hr_input"]), orb_dict=None, encoding="UTF-8")
        if not np.array_equal(irvec, cwi["irvec"]) or not np.array_equal(ndegen, cwi["ndegen"]):
            raise SymCWInputError(f"the R vectors in hr_input = {cwi['hr_input']} are inconsistent with those of the CW model.")

    # ******************** #
    #       Response       #
    # ******************** #

    res = Response(cwi, cwm, HH_R=Hr)
    res.calc_response()
    res.calc_spin_moment()

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

        if cwi["berry_task"] == "morb":
            res.write_morb()

        if cwi["berry_task"] == "kubo":
            res.write_kubo()

        if cwi["berry_task"] == "sc":
            res.write_sc()

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
