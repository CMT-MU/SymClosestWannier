"""
create Symmetry-adapted Closest Wannier (SymCW) tight-binding model based on Symmetry-Adapted Multipole Basis (SAMB) and Plane-Wave (PW) DFT calculation.
"""
import os

import numpy as np

from gcoreutils.nsarray import NSArray

from symclosestwannier.pw2scw.scw import SymCW
from symclosestwannier.util.reader import cwin_reader
from symclosestwannier.util.band import output_linear_dispersion


# ==================================================
def create_scw(seedname="cwannier"):
    """
    create Symmetry-adapted Closest Wannier (SymCW) tight-binding model based on Symmetry-Adapted Multipole Basis (SAMB) and Plane-Wave (PW) DFT calculation.

    Args:
        seedname (str, optional): seedname.
    """
    model_dict = cwin_reader(".", seedname)

    scw = SymCW(model_dict)

    outdir = scw["info"]["outdir"]
    mp_outdir = scw["info"]["mp_outdir"]
    mp_seedname = scw["info"]["mp_seedname"]

    scw.write(f"{seedname}_info.py", scw["info"], SymCW._info_header(), seedname)
    scw.write(f"{seedname}_data.py", scw["data"], SymCW._data_header(), seedname)

    if model_dict["write_hr"]:
        Hr_dict = SymCW.matrix_dict_r(scw.Hr, scw["data"]["rpoints"])
        Hr_str = "".join(
            [f"{n1}  {n2}  {n3}  {a}  {b}  {np.real(v)}  {np.imag(v)}\n" for (n1, n2, n3, a, b), v in Hr_dict.items()]
        )
        scw.write(f"{seedname}_hr.dat", Hr_str, SymCW._hr_header(), None)

    if model_dict["write_sr"]:
        Sr_dict = SymCW.matrix_dict_r(scw.Sr, scw["data"]["rpoints"])
        Sr_str = "".join(
            [f"{n1}  {n2}  {n3}  {a}  {b}  {np.real(v)}  {np.imag(v)}\n" for (n1, n2, n3, a, b), v in Sr_dict.items()]
        )
        scw.write(f"{seedname}_sr.dat", Sr_str, SymCW._sr_header(), None)

    if scw["info"]["symmetrization"]:
        if model_dict["write_hr"]:
            Hr_sym_dict = SymCW.matrix_dict_r(scw.Hr_sym, scw["data"]["rpoints_mp"])
            Hr_sym_str = "".join(
                [
                    f"{n1}  {n2}  {n3}  {a}  {b}  {np.real(v)}  {np.imag(v)}\n"
                    for (n1, n2, n3, a, b), v in Hr_sym_dict.items()
                ]
            )
            scw.write(f"{seedname}_hr_sym.dat", Hr_sym_str, SymCW._hr_header(), None, dir=mp_outdir)

        if model_dict["write_sr"]:
            Sr_sym_dict = SymCW.matrix_dict_r(scw.Sr_sym, scw["data"]["rpoints_mp"])
            Sr_sym_str = "".join(
                [
                    f"{n1}  {n2}  {n3}  {a}  {b}  {np.real(v)}  {np.imag(v)}\n"
                    for (n1, n2, n3, a, b), v in Sr_sym_dict.items()
                ]
            )
            scw.write(f"{seedname}_sr_sym.dat", Sr_sym_str, SymCW._sr_header(), None, dir=mp_outdir)

        z = "".join([f"{j+1}    {zj}    {tag}    {v} \n " for j, ((zj, tag), v) in enumerate(scw["data"]["z"].items())])
        scw.write(f"{mp_seedname}_z.dat", z, SymCW._z_header(), None, dir=mp_outdir)

        s = "".join([f"{j+1}    {zj}    {tag}    {v} \n " for j, ((zj, tag), v) in enumerate(scw["data"]["s"].items())])
        scw.write(f"{mp_seedname}_s.dat", s, SymCW._s_header(), None, dir=mp_outdir)

    # band calculation
    if "kpoint" in scw["info"] and "kpoint_path" in scw["info"]:
        k_linear = NSArray(scw["data"]["k_linear"], "vector", fmt="value")
        k_dis_pos = scw["data"]["k_dis_pos"]

        if os.path.isfile(f"{seedname}.band.gnu"):
            ref_filename = f"{seedname}.band.gnu"
        elif os.path.isfile(f"{seedname}.band.gnu.dat"):
            ref_filename = f"{seedname}.band.gnu.dat"
        else:
            ref_filename = None

        a = model_dict["a"]
        if a is None:
            A = NSArray(scw["info"]["unit_cell_cart"], "matrix", fmt="value")
            a = A[0].norm()

        Ek, Uk = np.linalg.eigh(scw.Hk_path)

        ef = scw["info"]["fermi_energy"]

        output_linear_dispersion(
            ".", seedname + "_band.txt", k_linear, Ek, Uk, ref_filename=ref_filename, a=a, ef=ef, k_dis_pos=k_dis_pos
        )

        if scw["info"]["symmetrization"]:
            rel = os.path.relpath(outdir, mp_outdir)

            if os.path.isfile(f"{seedname}.band.gnu"):
                ref_filename = f"{rel}/{seedname}.band.gnu"
            elif os.path.isfile(f"{seedname}.band.gnu.dat"):
                ref_filename = f"{rel}/{seedname}.band.gnu.dat"
            else:
                ref_filename = None

            Ek, Uk = np.linalg.eigh(scw.Hk_sym_path)

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
