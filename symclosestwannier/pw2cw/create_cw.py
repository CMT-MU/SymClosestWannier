"""
create Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.
CW TB model can be symmetrized by using Symmetry-Adapted Multipole Basis (SAMB).
"""
import os

import numpy as np

from gcoreutils.nsarray import NSArray

from symclosestwannier.pw2cw.cw import CW
from symclosestwannier.pw2cw.cw_manager import CWManager
from symclosestwannier.pw2cw.cw_info import CWInfo
from symclosestwannier.util.band import output_linear_dispersion


# ==================================================
def create_cw(seedname="cwannier"):
    """
    Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.
    CW TB model can be symmetrized by using Symmetry-Adapted Multipole Basis (SAMB).

    Args:
        seedname (str, optional): seedname.
    """
    cwi = CWInfo(".", seedname)
    cwm = CWManager(topdir=cwi["outdir"], verbose=cwi["verbose"], parallel=cwi["parallel"], formatter=cwi["formatter"])
    cw = CW(cwi, cwm)

    cw._cwm.write(f"{seedname}_info.py", cw._cwi.copy(), CW._info_header(), seedname)
    cw._cwm.write(f"{seedname}_data.py", cw.copy(), CW._data_header(), seedname)

    if cw._cwi["write_hr"]:
        cw.write_or(cw["Hr"], cw._cwi["rpoints"], f"{cw._cwi['seedname']}_hr.dat", CW._hr_header())

    if cw._cwi["write_sr"]:
        cw.write_or(cw["Sr"], cw._cwi["rpoints"], f"{cw._cwi['seedname']}_sr.dat", CW._sr_header())

    if cw._cwi["symmetrization"]:
        if cw._cwi["write_hr"]:
            filename = os.path.join(cw._cwi["mp_outdir"], "{}".format(f"{cw._cwi['mp_seedname']}_hr_sym.dat"))
            cw.write_or(cw["Hr_sym"], cw["rpoints_mp"], filename, CW._hr_header())

        if cw._cwi["write_sr"]:
            filename = os.path.join(cw._cwi["mp_outdir"], "{}".format(f"{cw._cwi['mp_seedname']}_sr_sym.dat"))
            cw.write_or(cw["Sr_sym"], cw["rpoints_mp"], filename, CW._sr_header())

        z = "".join(
            [f"{j+1}    {zj}    {tag}    {'{:.8f}'.format(v)}  \n " for j, ((zj, tag), v) in enumerate(cw["z"].items())]
        )
        filename = os.path.join(cw._cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_z.dat"))
        cw._cwm.write(filename, z, CW._z_header(), None)

        s = "".join(
            [f"{j+1}    {zj}    {tag}    {'{:.8f}'.format(v)}  \n " for j, ((zj, tag), v) in enumerate(cw["s"].items())]
        )
        filename = os.path.join(cw._cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_s.dat"))
        cw._cwm.write(filename, s, CW._s_header(), None)

        z_nonortho = "".join(
            [
                f"{j+1}    {zj}    {tag}    {'{:.8f}'.format(v)}  \n "
                for j, ((zj, tag), v) in enumerate(cw["z_nonortho"].items())
            ]
        )
        filename = os.path.join(cw._cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_z_nonortho.dat"))
        cw._cwm.write(filename, z_nonortho, CW._z_header(), None)

    # band calculation
    if cw._cwi["kpoint"] is not None and cw._cwi["kpoint_path"] is not None:
        k_linear = NSArray(cw._cwi["k_linear"], "vector", fmt="value")
        k_dis_pos = cw._cwi["k_dis_pos"]

        if os.path.isfile(f"{seedname}.band.gnu"):
            ref_filename = f"{seedname}.band.gnu"
        elif os.path.isfile(f"{seedname}.band.gnu.dat"):
            ref_filename = f"{seedname}.band.gnu.dat"
        else:
            ref_filename = None

        a = cw._cwi["a"]
        if a is None:
            A = NSArray(cw._cwi["unit_cell_cart"], "matrix", fmt="value")
            a = A[0].norm()

        Hk_path = cw.fourier_transform_r_to_k(cw["Hr"], cw._cwi["rpoints"], cw._cwi["kpoints_path"])[0]
        Ek, Uk = np.linalg.eigh(Hk_path)

        ef = cw._cwi["fermi_energy"]

        output_linear_dispersion(
            ".", seedname + "_band.txt", k_linear, Ek, Uk, ref_filename=ref_filename, a=a, ef=ef, k_dis_pos=k_dis_pos
        )

        if cw._cwi["symmetrization"]:
            rel = os.path.relpath(cw._cwi["outdir"], cw._cwi["mp_outdir"])

            if os.path.isfile(f"{seedname}.band.gnu"):
                ref_filename = f"{rel}/{seedname}.band.gnu"
            elif os.path.isfile(f"{seedname}.band.gnu.dat"):
                ref_filename = f"{rel}/{seedname}.band.gnu.dat"
            else:
                ref_filename = None

            Hk_sym_path = cw.fourier_transform_r_to_k(cw["Hr_sym"], cw["rpoints_mp"], cw._cwi["kpoints_path"])[0]
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
