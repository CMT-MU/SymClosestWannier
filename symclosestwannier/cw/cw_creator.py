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
from symclosestwannier.util.band import output_linear_dispersion, output_linear_dispersion_eig
from symclosestwannier.util.fermi_surface import output_fermi_surface_eig
from symclosestwannier.util.dos import output_dos
from symclosestwannier.util.cohp import output_cohp
from symclosestwannier.util.lindhard import get_lindhard, output_lindhard, output_lindhard_surface


from symclosestwannier.util.message import (
    cw_open_msg,
    cw_end_msg,
    system_msg,
    cwin_msg,
    cw_start_output_msg,
    cw_end_output_msg,
)

from symclosestwannier.util.get_oper_R import get_oper_R

from symclosestwannier.util.utility import ket_samb_list, sort_ket_matrix, tune_fermi_level


# ==================================================
def _ref_band_filename(indir, seedname, plotdir):
    """
    reference DFT band file (seedname.band.gnu(.dat) in indir) relative to the directory of the gnuplot script.

    Args:
        indir (str): directory of input files.
        seedname (str): seedname.
        plotdir (str): directory where the gnuplot script is written.

    Returns:
        str or None: relative file name, or None if not found.
    """
    for ext in ("band.gnu", "band.gnu.dat"):
        full = os.path.join(indir, f"{seedname}.{ext}")
        if os.path.isfile(full):
            return os.path.relpath(full, os.path.abspath(plotdir))

    return None

# ==================================================
def cw_creator(seedname="cwannier"):
    """
    Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.
    CW TB model can be symmetrized by using Symmetry-Adapted Multipole Basis (SAMB).

    Args:
        seedname (str, optional): seedname.
    """
    # input files are read from the current directory, CWManager moves to outdir.
    # the current directory is restored afterwards.
    indir = os.getcwd()
    try:
        _cw_creator(seedname, indir)
    finally:
        os.chdir(indir)

# ==================================================
def _cw_creator(seedname, indir):
    """
    Args:
        seedname (str): seedname.
        indir (str): directory of input files.
    """
    cwin = CWin(indir, seedname)
    cwm = CWManager(
        topdir=cwin["outdir"], verbose=cwin["verbose"], parallel=cwin["parallel"], formatter=cwin["formatter"]
    )

    outfile = f"{seedname}.cwout"

    cwi = CWInfo(indir, seedname)

    cwm.log(cw_open_msg(), stamp=None, end="\n", file=outfile, mode="w")
    cwm.log(system_msg(cwi), stamp=None, end="\n", file=outfile, mode="a")
    cwm.log(cwin_msg(cwi), stamp=None, end="\n", file=outfile, mode="a")

    cw_model = CWModel(cwi, cwm)
    cwi = cw_model._cwi

    cwm.log(cw_start_output_msg(), stamp=None, end="\n", file=outfile, mode="a")
    cwm.set_stamp()

    if cwi["write_info_data"]:
        filename = os.path.join(cwi["outdir"], "{}".format(f"{cwi['seedname']}.hdf5"))
        cw_model.write_info_data(filename)

    if cwi["write_hr"]:
        filename = f"{cwi['seedname']}_hr.dat.cw"
        cw_model.write_or(cw_model["Hr"], filename)  # , header=CWModel._hr_header())

        filename = f"{cwi['seedname']}_hr_R_dep.dat.cw"
        cw_model.write_O_R_dependence(cw_model["Hr"], filename, header=CWModel._O_R_dependence_header())

        filename = f"{cwi['seedname']}_nr.dat.cw"
        cw_model.write_or(cw_model["nr"], filename)  # , header=CWModel._hr_header())

        filename = f"{cwi['seedname']}_nr_R_dep.dat.cw"
        cw_model.write_O_R_dependence(cw_model["nr"], filename, header=CWModel._O_R_dependence_header())

    if cwi["write_sr"]:
        filename = f"{cwi['seedname']}_sr.dat.cw"
        cw_model.write_or(cw_model["Sr"], filename)  # , header=CWModel._sr_header())

        filename = f"{cwi['seedname']}_sr_R_dep.dat.cw"
        cw_model.write_O_R_dependence(cw_model["Sr"], filename, header=CWModel._O_R_dependence_header())

    if cwi["write_u_matrices"] and cwi["restart"] != "w90":
        file_names = (f"{cwi['seedname']}_u.mat.cw", f"{cwi['seedname']}_u_dis.mat.cw")
        cwi.umat.write(file_names)

    position = "AA_R_tb" if cwi["tb_position"] else "AA_R"

    if cwi["write_rmn"]:
        AA_R = get_oper_R(position, cwi)
        filename = f"{cwi['seedname']}_r.dat.cw"
        cw_model.write_or(AA_R, filename, vec=True)

        # with tb_position the position operator is on-site and diagonal: no dependence on R to show.
        if not cwi["tb_position"]:
            filename = f"{cwi['seedname']}_rx_R_dep.dat.cw"
            cw_model.write_O_R_dependence(AA_R[0], filename, header=CWModel._O_R_dependence_header())
            filename = f"{cwi['seedname']}_ry_R_dep.dat.cw"
            cw_model.write_O_R_dependence(AA_R[1], filename, header=CWModel._O_R_dependence_header())
            filename = f"{cwi['seedname']}_rz_R_dep.dat.cw"
            cw_model.write_O_R_dependence(AA_R[2], filename, header=CWModel._O_R_dependence_header())

    if cwi["write_tb"]:
        AA_R = get_oper_R(position, cwi)
        filename = f"{cwi['seedname']}_tb.dat.cw"
        cw_model.write_tb(cw_model["Hr"], AA_R, filename)

    if cwi["write_vmn"]:
        v_R = get_oper_R("v_R", cwi)
        filename = f"{cwi['seedname']}_v.dat.cw"
        cw_model.write_or(v_R, filename, vec=True)

        filename = f"{cwi['seedname']}_vx_R_dep.dat.cw"
        cw_model.write_O_R_dependence(v_R[0], filename, header=CWModel._O_R_dependence_header())
        filename = f"{cwi['seedname']}_vy_R_dep.dat.cw"
        cw_model.write_O_R_dependence(v_R[1], filename, header=CWModel._O_R_dependence_header())
        filename = f"{cwi['seedname']}_vz_R_dep.dat.cw"
        cw_model.write_O_R_dependence(v_R[2], filename, header=CWModel._O_R_dependence_header())

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
        SS_R = get_oper_R("SS_R", cwi)
        filename = f"{cwi['seedname']}.s.cw"
        cw_model.write_or(SS_R, filename, vec=True)

    if cwi["symmetrization"]:
        if cwi["write_hr"]:
            filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_hr_sym.dat.cw"))
            cw_model.write_or(cw_model["Hr_sym"], filename, header=CWModel._hr_header())

            filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_hr_nonortho_sym.dat.cw"))
            cw_model.write_or(cw_model["Hr_nonortho_sym"], filename, header=CWModel._hr_header())

            filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_nr_sym.dat.cw"))
            cw_model.write_or(cw_model["nr_sym"], filename, header=CWModel._hr_header())

        if cwi["write_sr"]:
            filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_sr_sym.dat.cw"))
            cw_model.write_or(cw_model["Sr_sym"], filename, header=CWModel._sr_header())

        if cwi["write_tb"]:
            # Hr_sym is in the ket order of the SAMBs, AA_R in that of ket_amn.
            ket_samb = cwi._mm["full_matrix"]["ket"]
            ket_amn = cwi.get("ket_amn", ket_samb)
            AA_R = get_oper_R(position, cwi)
            AA_R = np.array([sort_ket_matrix(AA_R[a], ket_amn, ket_samb) for a in range(3)])
            filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_tb_sym.dat.cw"))
            cw_model.write_tb(cw_model["Hr_sym"], AA_R, filename)

        filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_z.dat.cw"))
        cw_model.write_samb_coeffs(filename, type="z")

        filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_z_nonortho.dat.cw"))
        cw_model.write_samb_coeffs(filename, type="z_nonortho")

        filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_s.dat.cw"))
        cw_model.write_samb_coeffs(filename, type="s")

        filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_n.dat.cw"))
        cw_model.write_samb_coeffs(filename, type="n")

        if cwi["calc_spin_2d"] and cwi["pauli_spn"] is not None:
            filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_sx.dat.cw"))
            cw_model.write_samb_coeffs(filename, type="sx")

            filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_sy.dat.cw"))
            cw_model.write_samb_coeffs(filename, type="sy")

            filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_sz.dat.cw"))
            cw_model.write_samb_coeffs(filename, type="sz")

    # # the order of atoms are different from that of SAMBs
    atoms_list = list(cw_model._cwi["atoms_frac_shift"].values())
    atoms_frac = [atoms_list[i] for i in cw_model._cwi["nw2n"]]

    if cwi["tb_gauge"]:
        atoms_list = list(cw_model._cwi["atoms_frac_shift"].values())
        atoms_frac = [atoms_list[i] for i in cw_model._cwi["nw2n"]]
    else:
        atoms_frac = None

    # band calculation
    if cwi["kpoint"] is not None and cwi["kpoint_path"] is not None:
        cwm.log("\n  * calculating band dispersion ... ", None, end="", file=outfile, mode="a")

        k_linear = NSArray(cwi["k_linear"], "vector", fmt="value")
        k_dis_pos = cwi["k_dis_pos"]

        ref_filename = _ref_band_filename(indir, seedname, os.getcwd())

        a = cwi["a"]
        if a is None:
            A = NSArray(cwi["unit_cell_cart"], "matrix", fmt="value")
            a = A[0].norm()

        Hk_path = cw_model.fourier_transform_r_to_k(
            cw_model["Hr"], cwi["kpoints_path"], cwi["irvec"], cwi["ndegen"], atoms_frac=atoms_frac
        )

        Ek, Uk = np.linalg.eigh(Hk_path)

        ef = cwi["fermi_energy"]

        if cwi["calc_spin_2d"]:
            SS_R = get_oper_R("SS_R", cwi)
            SS_k = cw_model.fourier_transform_r_to_k_vec(
                SS_R, cwi["kpoints_path"], cwi["irvec"], cwi["ndegen"], atoms_frac=atoms_frac
            )
            SS_H = np.array([Uk.transpose(0, 2, 1).conjugate() @ SS_k[a] @ Uk for a in range(3)])
            Sk = np.real(np.diagonal(SS_H, axis1=2, axis2=3))
            Sk = Sk.transpose(2, 1, 0)
        else:
            Sk = None

        output_linear_dispersion_eig(
            ".",
            seedname + "_band.txt",
            k_linear,
            e=Ek,
            ref_filename=ref_filename,
            a=a,
            ef=ef,
            k_dis_pos=k_dis_pos,
        )

        output_linear_dispersion(
            ".",
            seedname + "_band_detail.txt",
            k_linear,
            e=Ek,
            u=Uk,
            ref_filename=ref_filename,
            a=a,
            ef=ef,
            k_dis_pos=k_dis_pos,
        )

        if cwi["symmetrization"]:
            ket_samb = ket_samb_list(cwi._mm)

            if cwi["tb_gauge"]:
                site_dict = {
                    k + "_" + str(vi.sublattice): vi.position_primitive.tolist()
                    for k, v in cwi._mm["site"]["cell"].items()
                    for vi in v
                    if vi.plus_set == 1
                }
                atoms_frac = [site_dict[atom + "_" + str(sl)] for atom, sl, rank, orbital in ket_samb]
            else:
                atoms_frac = None

            ref_filename = _ref_band_filename(indir, seedname, cwi["mp_outdir"])

            Hk_sym_path = cw_model.fourier_transform_r_to_k(
                cw_model["Hr_sym"], cwi["kpoints_path"], cwi["irvec"], cwi["ndegen"], atoms_frac=atoms_frac
            )
            Ek, Uk = np.linalg.eigh(Hk_sym_path)

            if cwi["calc_spin_2d"]:
                SS_R = get_oper_R("SS_R", cwi)
                SS_k = cw_model.fourier_transform_r_to_k_vec(
                    SS_R, cwi["kpoints_path"], cwi["irvec"], cwi["ndegen"], atoms_frac=atoms_frac
                )
                ket_amn = cwi.get("ket_amn", ket_samb)
                SS_k = np.array([sort_ket_matrix(SS_k[a], ket_amn, ket_samb) for a in range(3)])
                SS_H = np.array([Uk.transpose(0, 2, 1).conjugate() @ SS_k[a] @ Uk for a in range(3)])
                Sk = np.real(np.diagonal(SS_H, axis1=2, axis2=3))
                Sk = Sk.transpose(2, 1, 0)
            else:
                Sk = None

            output_linear_dispersion_eig(
                cwi["mp_outdir"],
                cwi["mp_seedname"] + "_band.txt",
                k_linear,
                e=Ek,
                o=Sk,
                ref_filename=ref_filename,
                a=a,
                ef=ef,
                k_dis_pos=k_dis_pos,
            )

            output_linear_dispersion(
                cwi["mp_outdir"],
                cwi["mp_seedname"] + "_band_detail.txt",
                k_linear,
                e=Ek,
                u=Uk,
                ref_filename=ref_filename,
                a=a,
                ef=ef,
                k_dis_pos=k_dis_pos,
            )

        cwm.log("done", end="\n", file=outfile, mode="a")

    # band calculation
    if cwi["fermi_surface"]:
        cwm.log("\n  * calculating fermi surface ... ", None, end="", file=outfile, mode="a")
        kpoints = np.array(cwi["fermi_surface_grid"], dtype=float)

        Hk = CWModel.fourier_transform_r_to_k(cw_model["Hr"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac=None)
        Ek, Uk = np.linalg.eigh(Hk)
        ef = cwi["fermi_energy"]

        kpoints_2d = np.array(cwi["fermi_surface_grid_2d"], dtype=float)
        output_fermi_surface_eig(".", seedname, kpoints_2d, e=Ek, ef=ef)

        if cwi["symmetrization"]:
            ket_samb = ket_samb_list(cwi._mm)

            if cwi["tb_gauge"]:
                site_dict = {
                    k + "_" + str(vi.sublattice): vi.position_primitive.tolist()
                    for k, v in cwi._mm["site"]["cell"].items()
                    for vi in v
                    if vi.plus_set == 1
                }
                atoms_frac = [site_dict[atom + "_" + str(sl)] for atom, sl, rank, orbital in ket_samb]
            else:
                atoms_frac = None

            Hk_sym = cw_model.fourier_transform_r_to_k(
                cw_model["Hr_sym"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac=atoms_frac
            )
            Ek, Uk = np.linalg.eigh(Hk_sym)

            output_fermi_surface_eig(cwi["mp_outdir"], seedname, kpoints_2d, e=Ek, ef=ef)

    #####

    if cwi["calc_dos"]:
        cwm.log("\n  * calculating DOS ... ", None, end="", file=outfile, mode="a")
        cwm.set_stamp()

        N1, N2, N3 = cwi["dos_kmesh"]
        kpoints = np.array(
            [[i / float(N1), j / float(N2), k / float(N3)] for i in range(N1) for j in range(N2) for k in range(N3)]
        )

        Hk_grid = CWModel.fourier_transform_r_to_k(
            cw_model["Hr"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac=None
        )
        Ek, Uk = np.linalg.eigh(Hk_grid)

        ef_shift = cwi["fermi_energy"]
        dos_num_fermi = cwi["dos_num_fermi"]
        dos_smr_en_width = cwi["dos_smr_en_width"]

        dos_emax = cwi["dos_emax"]
        dos_emin = cwi["dos_emin"]

        output_dos(".", seedname + "_dos.txt", Ek, Uk, ef_shift, dos_num_fermi, dos_smr_en_width, dos_emax, dos_emin)

        if cwi["symmetrization"]:
            ket_samb = ket_samb_list(cwi._mm)

            if cwi["tb_gauge"]:
                site_dict = {
                    k + "_" + str(vi.sublattice): vi.position_primitive.tolist()
                    for k, v in cwi._mm["site"]["cell"].items()
                    for vi in v
                    if vi.plus_set == 1
                }
                atoms_frac = [site_dict[atom + "_" + str(sl)] for atom, sl, rank, orbital in ket_samb]
            else:
                atoms_frac = None

            Hk_sym_grid = cw_model.fourier_transform_r_to_k(
                cw_model["Hr_sym"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac=atoms_frac
            )
            Ek, Uk = np.linalg.eigh(Hk_sym_grid)

            output_dos(
                cwi["mp_outdir"],
                cwi["mp_seedname"] + "_dos.txt",
                Ek,
                Uk,
                ef_shift,
                dos_num_fermi,
                dos_smr_en_width,
                dos_emax,
                dos_emin,
            )

        cwm.log("done", end="\n", file=outfile, mode="a")

    #####
    if cwi["calc_cohp"]:
        cwm.log("\n  * calculating COHP ... ", None, end="", file=outfile, mode="a")
        cwm.set_stamp()

        N1, N2, N3 = cwi["cohp_kmesh"]
        kpoints = np.array(
            [[i / float(N1), j / float(N2), k / float(N3)] for i in range(N1) for j in range(N2) for k in range(N3)]
        )

        A = NSArray(cwi["unit_cell_cart"], "matrix", fmt="value")

        ef_shift = cwi["fermi_energy"]
        cohp_bond_length_max = cwi["cohp_bond_length_max"]
        cohp_bond_length_min = cwi["cohp_bond_length_min"]
        cohp_head_atom = cwi["cohp_head_atom"]
        cohp_tail_atom = cwi["cohp_tail_atom"]
        cohp_head_atom_idx = cwi["cohp_head_atom_idx"]
        cohp_tail_atom_idx = cwi["cohp_tail_atom_idx"]
        cohp_num_fermi = cwi["cohp_num_fermi"]
        cohp_smr_en_width = cwi["cohp_smr_en_width"]
        cohp_emax = cwi["cohp_emax"]
        cohp_emin = cwi["cohp_emin"]

        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])

        output_cohp(
            ".",
            seedname + "_cohp.txt",
            cw_model["Hr"],
            cwi["Ek"],
            cwi["Uk"],
            cwi["kpoints"],
            cwi["irvec"],
            cwi["ndegen"],
            cwi["atoms_frac"],
            cwi["nw2n"],
            cwi["nw2l"],
            cwi["nw2m"],
            cwi["nw2r"],
            cwi["nw2s"],
            A,
            cohp_bond_length_max,
            cohp_bond_length_min,
            cohp_head_atom,
            cohp_tail_atom,
            cohp_head_atom_idx,
            cohp_tail_atom_idx,
            cohp_num_fermi,
            cohp_smr_en_width,
            cohp_emax,
            cohp_emin,
            ef_shift,
        )

    #####

    if cwi["lindhard"]:
        cwm.log("\n  * calculating lindhard ... ", None, end="", file=outfile, mode="a")
        cwm.set_stamp()

        qpoints = cwi["qpoints_path"]
        omega = cwi["lindhard_freq"]
        ef = cwi["fermi_energy"]
        T = cwi["temperature"]
        cwf_delta = cwi["lindhard_smr_fixed_en_width"]

        if cwi["filling"] is not None:
            N1, N2, N3 = cwi["lindhard_kmesh"]
            kpoints = np.array(
                [[i / float(N1), j / float(N2), k / float(N3)] for i in range(N1) for j in range(N2) for k in range(N3)]
            )

            Hk_grid = CWModel.fourier_transform_r_to_k(
                cw_model["Hr"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac=None
            )
            Ek, _ = np.linalg.eigh(Hk_grid)

            ef_calculated = tune_fermi_level(Ek, filling=cwi["filling"], T=T, threshold=1e-8)
            cwm.log(f"\n  * ef_calculated = {ef_calculated} (ef(input) = {ef})", None, end="", file=outfile, mode="a")

            ef = ef_calculated

        lindhard_re, lindhard_im_om0 = get_lindhard(cwi, cw_model["Hr"], qpoints, omega, ef, T, delta)

        q = cwi["q_linear"]
        q_dis_pos = cwi["q_dis_pos"]

        output_lindhard(".", seedname + "_lindhard_re.txt", omega, q, lindhard_re, q_dis_pos=q_dis_pos, ef=ef)
        output_lindhard(".", seedname + "_lindhard_im_om0.txt", omega, q, lindhard_im_om0, q_dis_pos=q_dis_pos, ef=ef)

        cwm.log("done", end="\n", file=outfile, mode="a")

    #####
    if cwi["lindhard_surface"]:
        cwm.log("\n  * calculating lindhard ... ", None, end="", file=outfile, mode="a")
        cwm.set_stamp()

        qpoints = cwi["qpoints_surface_grid"]
        omega = cwi["lindhard_freq"]
        ef = cwi["fermi_energy"]
        T = cwi["temperature"]
        delta = cwi["lindhard_smr_fixed_en_width"]

        if cwi["filling"] is not None:
            N1, N2, N3 = cwi["lindhard_kmesh"]
            kpoints = np.array(
                [[i / float(N1), j / float(N2), k / float(N3)] for i in range(N1) for j in range(N2) for k in range(N3)]
            )

            Hk_grid = CWModel.fourier_transform_r_to_k(
                cw_model["Hr"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac=None
            )
            Ek, _ = np.linalg.eigh(Hk_grid)

            ef_calculated = tune_fermi_level(Ek, filling=cwi["filling"], T=T, threshold=1e-8)
            cwm.log(f"\n  * ef_calculated = {ef_calculated} (ef(input) = {ef})", None, end="", file=outfile, mode="a")

            ef = ef_calculated

        lindhard_re, lindhard_im_om0 = get_lindhard(cwi, cw_model["Hr"], qpoints, omega, ef, T, delta)

        qpoints_2d = cwi["qpoints_surface_grid_2d"]

        output_lindhard_surface(
            ".", seedname + "_lindhard_surface.txt", omega, qpoints_2d, lindhard_re, lindhard_im_om0, ef=ef
        )

        cwm.log("done", end="\n", file=outfile, mode="a")

    #####

    cwm.log(f"\n\n  * total elapsed_time:", file=outfile, mode="a")
    cwm.log(cw_end_output_msg(), stamp=None, end="\n", file=outfile, mode="a")

    cwm.log(f"  * total elapsed_time:", stamp="start", file=outfile, mode="a")
    cwm.log(cw_end_msg(), stamp=None, end="\n", file=outfile, mode="a")
