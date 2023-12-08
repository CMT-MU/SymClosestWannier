"""
create Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.
CW TB model can be symmetrized by using Symmetry-Adapted Multipole Basis (SAMB).
"""
import os

import numpy as np

from gcoreutils.nsarray import NSArray

from symclosestwannier.pw2cw.cw import CW
from symclosestwannier.util.reader import cwin_reader
from symclosestwannier.util.band import output_linear_dispersion


# ==================================================
def create_cw(seedname="cwannier"):
    """
    Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.
    CW TB model can be symmetrized by using Symmetry-Adapted Multipole Basis (SAMB).

    Args:
        seedname (str, optional): seedname.
    """
    model_dict = cwin_reader(".", seedname)

    cw = CW(model_dict)

    outdir = cw["info"]["outdir"]
    mp_outdir = cw["info"]["mp_outdir"]
    mp_seedname = cw["info"]["mp_seedname"]

    cw.write(f"{seedname}_info.py", cw["info"], CW._info_header(), seedname)

    d = cw["data"].copy()
    del d["kpoints"]
    del d["rpoints"]
    del d["Pk"]
    del d["Uk"]
    del d["Hk"]
    del d["Sk"]
    del d["Hk_nonortho"]
    del d["matrix_dict"]
    cw.write(f"{seedname}_data.py", d, CW._data_header(), seedname)

    if model_dict["write_hr"]:
        Hr_dict = CW.matrix_dict_r(cw.Hr, cw["data"]["rpoints"])
        Hr_str = "".join(
            [
                f"{n1}  {n2}  {n3}  {a}  {b}  {'{:.6f}'.format(np.real(v))}  {'{:.6f}'.format(np.imag(v))}\n"
                for (n1, n2, n3, a, b), v in Hr_dict.items()
            ]
        )
        cw.write(f"{seedname}_hr.dat", Hr_str, CW._hr_header(), None)

    if model_dict["write_sr"]:
        Sr_dict = CW.matrix_dict_r(cw.Sr, cw["data"]["rpoints"])
        Sr_str = "".join(
            [
                f"{n1}  {n2}  {n3}  {a}  {b}  {'{:.6f}'.format(np.real(v))}  {'{:.6f}'.format(np.imag(v))}\n"
                for (n1, n2, n3, a, b), v in Sr_dict.items()
            ]
        )
        cw.write(f"{seedname}_sr.dat", Sr_str, CW._sr_header(), None)

    if cw["info"]["symmetrization"]:
        if model_dict["write_hr"]:
            Hr_sym_dict = CW.matrix_dict_r(cw.Hr_sym, cw["data"]["rpoints_mp"])
            Hr_sym_str = "".join(
                [
                    f"{n1}  {n2}  {n3}  {a}  {b}  {'{:.6f}'.format(np.real(v))}  {'{:.6f}'.format(np.imag(v))}\n"
                    for (n1, n2, n3, a, b), v in Hr_sym_dict.items()
                ]
            )
            cw.write(f"{seedname}_hr_sym.dat", Hr_sym_str, CW._hr_header(), None, dir=mp_outdir)

        if model_dict["write_sr"]:
            Sr_sym_dict = CW.matrix_dict_r(cw.Sr_sym, cw["data"]["rpoints_mp"])
            Sr_sym_str = "".join(
                [
                    f"{n1}  {n2}  {n3}  {a}  {b}  {'{:.6f}'.format(np.real(v))}  {'{:.6f}'.format(np.imag(v))}\n"
                    for (n1, n2, n3, a, b), v in Sr_sym_dict.items()
                ]
            )
            cw.write(f"{seedname}_sr_sym.dat", Sr_sym_str, CW._sr_header(), None, dir=mp_outdir)

        z = "".join(
            [
                f"{j+1}    {zj}    {tag}    {'{:.6f}'.format(v)}  \n "
                for j, ((zj, tag), v) in enumerate(cw["data"]["z"].items())
            ]
        )
        cw.write(f"{mp_seedname}_z.dat", z, CW._z_header(), None, dir=mp_outdir)

        s = "".join(
            [
                f"{j+1}    {zj}    {tag}    {'{:.6f}'.format(v)}  \n "
                for j, ((zj, tag), v) in enumerate(cw["data"]["s"].items())
            ]
        )
        cw.write(f"{mp_seedname}_s.dat", s, CW._s_header(), None, dir=mp_outdir)

        z_nonortho = "".join(
            [
                f"{j+1}    {zj}    {tag}    {'{:.6f}'.format(v)}  \n "
                for j, ((zj, tag), v) in enumerate(cw["data"]["z_nonortho"].items())
            ]
        )
        cw.write(f"{mp_seedname}_z_nonortho.dat", z_nonortho, CW._z_header(), None, dir=mp_outdir)

    # band calculation
    if "kpoint" in cw["info"] and "kpoint_path" in cw["info"]:
        k_linear = NSArray(cw["data"]["k_linear"], "vector", fmt="value")
        k_dis_pos = cw["data"]["k_dis_pos"]

        if os.path.isfile(f"{seedname}.band.gnu"):
            ref_filename = f"{seedname}.band.gnu"
        elif os.path.isfile(f"{seedname}.band.gnu.dat"):
            ref_filename = f"{seedname}.band.gnu.dat"
        else:
            ref_filename = None

        a = model_dict["a"]
        if a is None:
            A = NSArray(cw["info"]["unit_cell_cart"], "matrix", fmt="value")
            a = A[0].norm()

        Ek, Uk = np.linalg.eigh(cw.Hk_path)

        ef = cw["info"]["fermi_energy"]

        output_linear_dispersion(
            ".", seedname + "_band.txt", k_linear, Ek, Uk, ref_filename=ref_filename, a=a, ef=ef, k_dis_pos=k_dis_pos
        )

        if cw["info"]["symmetrization"]:
            rel = os.path.relpath(outdir, mp_outdir)

            if os.path.isfile(f"{seedname}.band.gnu"):
                ref_filename = f"{rel}/{seedname}.band.gnu"
            elif os.path.isfile(f"{seedname}.band.gnu.dat"):
                ref_filename = f"{rel}/{seedname}.band.gnu.dat"
            else:
                ref_filename = None

            Ek, Uk = np.linalg.eigh(cw.Hk_sym_path)

            output_linear_dispersion(
                mp_outdir,
                mp_seedname + "_band.txt",
                k_linear,
                Ek,
                Uk,
                ref_filename=ref_filename,
                a=a,
                ef=ef,
                k_dis_pos=k_dis_pos,
            )
