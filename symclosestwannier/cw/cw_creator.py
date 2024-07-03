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
#           cw_analyzer: create Closest Wannier TB model             #
#                                                                    #
# ****************************************************************** #

import os
import numpy as np

from gcoreutils.nsarray import NSArray

from symclosestwannier.cw.win import Win
from symclosestwannier.cw.cwin import CWin
from symclosestwannier.cw.cw_info import CWInfo
from symclosestwannier.cw.cw_manager import CWManager
from symclosestwannier.cw.cw_model import CWModel
from symclosestwannier.util.band import output_linear_dispersion


from symclosestwannier.util.message import (
    cw_open_msg,
    cw_end_msg,
    system_msg,
    cw_start_output_msg,
    cw_end_output_msg,
)

from symclosestwannier.util.get_oper_R import get_oper_R


# ==================================================
def cw_creator(seedname="cwannier"):
    """
    Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.
    CW TB model can be symmetrized by using Symmetry-Adapted Multipole Basis (SAMB).

    Args:
        seedname (str, optional): seedname.
    """
    win = Win(".", seedname)
    cwin = CWin("./", seedname)
    cwm = CWManager(
        topdir=cwin["outdir"], verbose=cwin["verbose"], parallel=cwin["parallel"], formatter=cwin["formatter"]
    )

    outfile = f"{seedname}.cwout"

    if cwin["restart"] == "sym":
        filename = os.path.join(cwin["outdir"], "{}".format(f"{seedname}.hdf5"))
        info, data = CWModel.read_info_data(filename)
        cwi = CWInfo("./", seedname, dic=info)
        cwi |= cwin | win
        dic = data
    else:
        cwi = CWInfo("./", seedname)
        dic = None

    cwm.log(cw_open_msg(), stamp=None, end="\n", file=outfile, mode="w")
    cwm.log(system_msg(cwi), stamp=None, end="\n", file=outfile, mode="a")

    cw_model = CWModel(cwi, cwm, dic)
    cwi = cw_model._cwi

    cwm.log(cw_start_output_msg(), stamp=None, end="\n", file=outfile, mode="a")
    cwm.set_stamp()

    filename = os.path.join(cwi["outdir"], "{}".format(f"{cwi['seedname']}.hdf5"))
    cw_model.write_info_data(filename)

    if cwi["write_hr"]:
        filename = f"{cwi['seedname']}_hr.dat.cw"
        cw_model.write_or(cw_model["Hr"], filename, header=CWModel._hr_header())
        filename = f"{cwi['seedname']}_hr_nonortho.dat.cw"
        cw_model.write_or(cw_model["Hr_nonortho"], filename, header=CWModel._hr_header())

    if cwi["write_sr"]:
        filename = f"{cwi['seedname']}_sr.dat.cw"
        cw_model.write_or(cw_model["Sr"], filename, header=CWModel._sr_header())

    if cwi["write_u_matrices"] and cwi["restart"] != "w90":
        file_names = (f"{cwi['seedname']}_u.mat.cw", f"{cwi['seedname']}_u_dis.mat.cw")
        cwi.umat.write(file_names)

    if cwi["write_rmn"]:
        AA_R = get_oper_R("AA_R", cwi)
        filename = f"{cwi['seedname']}_r.dat.cw"
        cw_model.write_or(AA_R, filename, vec=True)

    if cwi["write_tb"]:
        AA_R = get_oper_R("AA_R", cwi)
        filename = f"{cwi['seedname']}_tb.dat.cw"
        cw_model.write_tb(cw_model["Hr"], AA_R, filename)

    if cwi["write_vmn"]:
        pass

    if cwi["write_eig"]:
        filename = f"{cwi['seedname']}.eig.cw"
        cwi.eig.write(filename)

    if cwi["write_amn"]:
        filename = f"{cwi['seedname']}.amn.cw"
        cwi.amn.write(filename)

    if cwi["write_mmn"]:
        filename = f"{cwi['seedname']}.mmn.cw"
        cwi.mmn.write(filename)

    if cwi["write_spn"]:
        pass

    if cwi["symmetrization"]:
        if cwi["write_hr"]:
            filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_hr_sym.dat.cw"))
            cw_model.write_or(cw_model["Hr_sym"], filename, header=CWModel._hr_header())

            filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_hr_nonortho_sym.dat.cw"))
            cw_model.write_or(cw_model["Hr_nonortho_sym"], filename, header=CWModel._hr_header())

        if cwi["write_sr"]:
            filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_sr_sym.dat.cw"))
            cw_model.write_or(cw_model["Sr_sym"], filename, header=CWModel._sr_header())

        if cwi["write_tb"]:
            AA_R = get_oper_R("AA_R", cwi)
            filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_tb_sym.dat.cw"))
            cw_model.write_tb(cw_model["Hr_sym"], AA_R, filename)

        filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_z.dat.cw"))
        cw_model.write_samb_coeffs(filename, type="z")

        filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_z_nonortho.dat.cw"))
        cw_model.write_samb_coeffs(filename, type="z_nonortho")

        filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_s.dat.cw"))
        cw_model.write_samb_coeffs(filename, type="s")

        filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_z_exp.dat.cw"))
        cw_model.write_samb_exp(filename)

    # the order of atoms are different from that of SAMBs
    atoms_list = list(cw_model._cwi["atoms_frac_shift"].values())
    atoms_frac = [atoms_list[i] for i in cw_model._cwi["nw2n"]]

    # band calculation
    if cwi["kpoint"] is not None and cwi["kpoint_path"] is not None:
        k_linear = NSArray(cwi["k_linear"], "vector", fmt="value")
        k_dis_pos = cwi["k_dis_pos"]

        if os.path.isfile(f"{seedname}.band.gnu"):
            ref_filename = f"{seedname}.band.gnu"
        elif os.path.isfile(f"{seedname}.band.gnu.dat"):
            ref_filename = f"{seedname}.band.gnu.dat"
        else:
            ref_filename = None

        a = cwi["a"]
        if a is None:
            A = NSArray(cwi["unit_cell_cart"], "matrix", fmt="value")
            a = A[0].norm()

        Hk_path = cw_model.fourier_transform_r_to_k(
            cw_model["Hr"], cwi["kpoints_path"], cwi["irvec"], cwi["ndegen"], atoms_frac
        )
        Ek, Uk = np.linalg.eigh(Hk_path)

        ef = cwi["fermi_energy"]

        output_linear_dispersion(
            ".", seedname + "_band.txt", k_linear, Ek, Uk, ref_filename=ref_filename, a=a, ef=ef, k_dis_pos=k_dis_pos
        )

        if cwi["symmetrization"]:
            ket_samb = cw_model["matrix_dict"]["ket"]
            cell_site = cw_model["matrix_dict"]["cell_site"]
            atoms_frac = [
                NSArray(cell_site[ket_samb[a].split("@")[1]][0], style="vector", fmt="value").tolist()
                for a in range(cw_model._cwi["num_wann"])
            ]

            rel = os.path.relpath(cwi["outdir"], cwi["mp_outdir"])

            if os.path.isfile(f"{seedname}.band.gnu"):
                ref_filename = f"{rel}/{seedname}.band.gnu"
            elif os.path.isfile(f"{seedname}.band.gnu.dat"):
                ref_filename = f"{rel}/{seedname}.band.gnu.dat"
            else:
                ref_filename = None

            Hk_sym_path = cw_model.fourier_transform_r_to_k(
                cw_model["Hr_sym"], cwi["kpoints_path"], cwi["irvec"], cwi["ndegen"], atoms_frac=atoms_frac
            )
            Ek, Uk = np.linalg.eigh(Hk_sym_path)

            output_linear_dispersion(
                cwi["mp_outdir"],
                cwi["mp_seedname"] + "_band.txt",
                k_linear,
                Ek,
                Uk,
                ref_filename=ref_filename,
                a=a,
                ef=ef,
                k_dis_pos=k_dis_pos,
            )

    cwm.log(f"\n\n  * total elapsed_time:", file=outfile, mode="a")
    cwm.log(cw_end_output_msg(), stamp=None, end="\n", file=outfile, mode="a")

    cwm.log(f"  * total elapsed_time:", stamp="start", file=outfile, mode="a")
    cwm.log(cw_end_msg(), stamp=None, end="\n", file=outfile, mode="a")
