"""
create Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.
CW TB model can be symmetrized by using Symmetry-Adapted Multipole Basis (SAMB).
"""
import os

import numpy as np

from gcoreutils.nsarray import NSArray

from symclosestwannier.cw.cw_info import CWInfo
from symclosestwannier.cw.cw_manager import CWManager
from symclosestwannier.cw.cw_model import CWModel
from symclosestwannier.util.band import output_linear_dispersion


from symclosestwannier.util.message import (
    cw_start_output_msg,
    cw_end_output_msg,
)

from symclosestwannier.cw.get_matrix_R import get_HH_R, get_AA_R, get_SS_R


# ==================================================
def cw_creator(seedname="cwannier"):
    """
    Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.
    CW TB model can be symmetrized by using Symmetry-Adapted Multipole Basis (SAMB).

    Args:
        seedname (str, optional): seedname.
    """
    cwi = CWInfo(".", seedname)
    cwm = CWManager(topdir=cwi["outdir"], verbose=cwi["verbose"], parallel=cwi["parallel"], formatter=cwi["formatter"])
    cw_model = CWModel(cwi, cwm)

    cw_model._cwm.log(cw_start_output_msg(), stamp=None, end="\n", file=cw_model._outfile, mode="w")

    cw_model._cwm.write(f"{seedname}_info.py", cw_model._cwi.copy(), CWModel._cw_info_header(), seedname)
    cw_model._cwm.write(f"{seedname}_data.py", cw_model.copy(), CWModel._cw_data_header(), seedname)

    if cw_model._cwi["write_hr"]:
        filename = f"{cw_model._cwi['seedname']}_hr.dat"
        cw_model.write_or(cw_model["Hr"], filename, CWModel._hr_header())
        filename = f"{cw_model._cwi['seedname']}_hr_nonortho.dat"
        cw_model.write_or(cw_model["Hr_nonortho"], filename, CWModel._hr_header())

    if cw_model._cwi["write_sr"]:
        filename = f"{cw_model._cwi['seedname']}_sr.dat"
        cw_model.write_or(cw_model["Sr"], filename, CWModel._sr_header())

    if cw_model._cwi["write_u_matrices"] and cw_model._cwi["restart"] != "w90":
        file_names = (f"{cw_model._cwi['seedname']}_u.mat", f"{cw_model._cwi['seedname']}_u_dis.mat")
        cw_model.umat.write(file_names)

    if cw_model._cwi["write_rmn"]:
        AA_R = get_AA_R(cw_model._cwi)
        filename = f"{cw_model._cwi['seedname']}_r.dat"
        cw_model.write_or(AA_R, filename, vec=True)

    if cw_model._cwi["write_vmn"]:
        pass

    if cw_model._cwi["write_tb"]:
        pass

    if cw_model._cwi["write_spn"]:
        pass

    if cw_model._cwi["symmetrization"]:
        if cw_model._cwi["write_hr"]:
            filename = os.path.join(
                cw_model._cwi["mp_outdir"], "{}".format(f"{cw_model._cwi['mp_seedname']}_hr_sym.dat")
            )
            cw_model.write_or(cw_model["Hr_sym"], filename, CWModel._hr_header())

            filename = os.path.join(
                cw_model._cwi["mp_outdir"], "{}".format(f"{cw_model._cwi['mp_seedname']}_hr_nonortho_sym.dat")
            )
            cw_model.write_or(cw_model["Hr_nonortho_sym"], filename, CWModel._hr_header())

        if cw_model._cwi["write_sr"]:
            filename = os.path.join(
                cw_model._cwi["mp_outdir"], "{}".format(f"{cw_model._cwi['mp_seedname']}_sr_sym.dat")
            )
            cw_model.write_or(cw_model["Sr_sym"], filename, CWModel._sr_header())

        filename = os.path.join(cw_model._cwi["mp_outdir"], "{}".format(f"{cw_model._cwi['mp_seedname']}_z.dat"))
        cw_model._cwm.write_samb_coeffs(filename, type="z")

        filename = os.path.join(
            cw_model._cwi["mp_outdir"], "{}".format(f"{cw_model._cwi['mp_seedname']}_z_nonortho.dat")
        )
        cw_model._cwm.write_samb_coeffs(filename, type="z_nonortho")

        filename = os.path.join(cw_model._cwi["mp_outdir"], "{}".format(f"{cw_model._cwi['mp_seedname']}_s.dat"))
        cw_model._cwm.write_samb_coeffs(filename, type="s")

    # band calculation
    if cw_model._cwi["kpoint"] is not None and cw_model._cwi["kpoint_path"] is not None:
        k_linear = NSArray(cw_model._cwi["k_linear"], "vector", fmt="value")
        k_dis_pos = cw_model._cwi["k_dis_pos"]

        if os.path.isfile(f"{seedname}.band.gnu"):
            ref_filename = f"{seedname}.band.gnu"
        elif os.path.isfile(f"{seedname}.band.gnu.dat"):
            ref_filename = f"{seedname}.band.gnu.dat"
        else:
            ref_filename = None

        a = cw_model._cwi["a"]
        if a is None:
            A = NSArray(cw_model._cwi["unit_cell_cart"], "matrix", fmt="value")
            a = A[0].norm()

        Hk_path = cw_model.fourier_transform_r_to_k(
            cw_model["Hr"], cw_model._cwi["kpoints_path"], cw_model._cwi["irvec"], cw_model._cwi["ndegen"]
        )
        Ek, Uk = np.linalg.eigh(Hk_path)

        ef = cw_model._cwi["fermi_energy"]

        output_linear_dispersion(
            ".", seedname + "_band.txt", k_linear, Ek, Uk, ref_filename=ref_filename, a=a, ef=ef, k_dis_pos=k_dis_pos
        )

        if cw_model._cwi["symmetrization"]:
            rel = os.path.relpath(cw_model._cwi["outdir"], cw_model._cwi["mp_outdir"])

            if os.path.isfile(f"{seedname}.band.gnu"):
                ref_filename = f"{rel}/{seedname}.band.gnu"
            elif os.path.isfile(f"{seedname}.band.gnu.dat"):
                ref_filename = f"{rel}/{seedname}.band.gnu.dat"
            else:
                ref_filename = None

            Hk_sym_path = cw_model.fourier_transform_r_to_k(
                cw_model["Hr_sym"], cw_model._cwi["kpoints_path"], cw_model["rpoints_mp"]
            )
            Ek, Uk = np.linalg.eigh(Hk_sym_path)

            output_linear_dispersion(
                cw_model._cwi["mp_outdir"],
                cw_model._cwi["mp_seedname"] + "_band.txt",
                k_linear,
                Ek,
                Uk,
                ref_filename=ref_filename,
                a=a,
                ef=ef,
                k_dis_pos=k_dis_pos,
            )

    cw_model._cwm.log(f"\n\n  * total elapsed_time:", stamp="start", file=cw_model._outfile, mode="a")

    cw_model._cwm.log(cw_end_output_msg(), stamp=None, end="\n", file=cw_model._outfile, mode="a")
