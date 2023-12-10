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
    cwi = CWInfo(".", seedname, read_mmn=False)
    cwm = CWManager(topdir=cwi["outdir"], verbose=cwi["verbose"], parallel=cwi["parallel"], formatter=cwi["formatter"])

    cw = CW(cwi, cwm)

    cw._cwm.write(f"{seedname}_info.py", cw._cwi.copy(), CW._info_header(), seedname)
    cw._cwm.write(f"{seedname}_data.py", cw.copy(), CW._data_header(), seedname)

    if cw._cwi["write_hr"]:
        cw.write_hr()

    if cw._cwi["write_sr"]:
        cw.write_sr()

    if cw._cwi["symmetrization"]:
        if cw._cwi["write_hr"]:
            Hr_sym_dict = CW.matrix_dict_r(cw.Hr_sym, cw["rpoints_mp"])
            Hr_sym_str = "".join(
                [
                    f"{n1}  {n2}  {n3}  {a}  {b}  {'{:.8f}'.format(np.real(v))}  {'{:.8f}'.format(np.imag(v))}\n"
                    for (n1, n2, n3, a, b), v in Hr_sym_dict.items()
                ]
            )
            cw._cwm.write(f"{seedname}_hr_sym.dat", Hr_sym_str, CW._hr_header(), None, dir=cwi["mp_outdir"])

        if cw._cwi["write_sr"]:
            Sr_sym_dict = CW.matrix_dict_r(cw.Sr_sym, cw["rpoints_mp"])
            Sr_sym_str = "".join(
                [
                    f"{n1}  {n2}  {n3}  {a}  {b}  {'{:.8f}'.format(np.real(v))}  {'{:.8f}'.format(np.imag(v))}\n"
                    for (n1, n2, n3, a, b), v in Sr_sym_dict.items()
                ]
            )
            cw._cwm.write(f"{seedname}_sr_sym.dat", Sr_sym_str, CW._sr_header(), None, dir=cwi["mp_outdir"])

        z = "".join(
            [f"{j+1}    {zj}    {tag}    {'{:.8f}'.format(v)}  \n " for j, ((zj, tag), v) in enumerate(cw["z"].items())]
        )
        cw._cwm.write(f"{cwi['mp_seedname']}_z.dat", z, CW._z_header(), None, dir=cwi["mp_outdir"])

        s = "".join(
            [f"{j+1}    {zj}    {tag}    {'{:.8f}'.format(v)}  \n " for j, ((zj, tag), v) in enumerate(cw["s"].items())]
        )
        cw._cwm.write(f"{cwi['mp_seedname']}_s.dat", s, CW._s_header(), None, dir=cwi["mp_outdir"])

        z_nonortho = "".join(
            [
                f"{j+1}    {zj}    {tag}    {'{:.8f}'.format(v)}  \n "
                for j, ((zj, tag), v) in enumerate(cw["z_nonortho"].items())
            ]
        )
        cw._cwm.write(f"{cwi['mp_seedname']}_z_nonortho.dat", z_nonortho, CW._z_header(), None, dir=cwi["mp_outdir"])

    # band calculation
    if "kpoint" in cw._cwi and "kpoint_path" in cw._cwi:
        k_linear = NSArray(cw["k_linear"], "vector", fmt="value")
        k_dis_pos = cw["k_dis_pos"]

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

        Hk_path = cw.fourier_transform_r_to_k(cw["Hr"], cw["rpoints"], cw["kpoints_path"])[0]
        Ek, Uk = np.linalg.eigh(Hk_path)

        ef = cw._cwi["fermi_energy"]

        output_linear_dispersion(
            ".", seedname + "_band.txt", k_linear, Ek, Uk, ref_filename=ref_filename, a=a, ef=ef, k_dis_pos=k_dis_pos
        )

        if cw._cwi["symmetrization"]:
            rel = os.path.relpath(cwi["outdir"], cwi["mp_outdir"])

            if os.path.isfile(f"{seedname}.band.gnu"):
                ref_filename = f"{rel}/{seedname}.band.gnu"
            elif os.path.isfile(f"{seedname}.band.gnu.dat"):
                ref_filename = f"{rel}/{seedname}.band.gnu.dat"
            else:
                ref_filename = None

            Ek, Uk = np.linalg.eigh(cw.Hk_sym_path)

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
