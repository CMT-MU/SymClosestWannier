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
from symclosestwannier.util.dos import output_dos


from symclosestwannier.util.message import (
    cw_open_msg,
    cw_end_msg,
    system_msg,
    cw_start_output_msg,
    cw_end_output_msg,
)

from symclosestwannier.util.get_oper_R import get_oper_R

from symclosestwannier.util.utility import sort_ket_matrix


# ==================================================
def cw_creator(seedname="cwannier"):
    """
    Closest Wannier (CW) tight-binding (TB) model based on Plane-Wave (PW) DFT calculation.
    CW TB model can be symmetrized by using Symmetry-Adapted Multipole Basis (SAMB).

    Args:
        seedname (str, optional): seedname.
    """
    cwin = CWin("./", seedname)
    cwm = CWManager(
        topdir=cwin["outdir"], verbose=cwin["verbose"], parallel=cwin["parallel"], formatter=cwin["formatter"]
    )

    outfile = f"{seedname}.cwout"

    cwi = CWInfo("./", seedname)

    cwm.log(cw_open_msg(), stamp=None, end="\n", file=outfile, mode="w")
    cwm.log(system_msg(cwi), stamp=None, end="\n", file=outfile, mode="a")

    cw_model = CWModel(cwi, cwm)
    cwi = cw_model._cwi
    samb_info = cw_model._samb_info

    cwm.log(cw_start_output_msg(), stamp=None, end="\n", file=outfile, mode="a")
    cwm.set_stamp()

    filename = os.path.join(cwi["outdir"], "{}".format(f"{cwi['seedname']}.hdf5"))
    cw_model.write_info_data(filename)

    if cwi["write_hr"]:
        filename = f"{cwi['seedname']}_hr.dat.cw"
        cw_model.write_or(cw_model["Hr"], filename, header=CWModel._hr_header())
        # filename = f"{cwi['seedname']}_hr_nonortho.dat.cw"
        # cw_model.write_or(cw_model["Hr_nonortho"], filename, header=CWModel._hr_header())

        filename = f"{cwi['seedname']}_hr_R_dep.dat.cw"
        cw_model.write_O_R_dependence(cw_model["Hr"], filename, header=CWModel._O_R_dependence_header())

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
        SS_R = get_oper_R("SS_R", cwi)
        filename = f"{cwi['seedname']}.s.cw"
        cw_model.write_or(SS_R, filename, vec=True)

    if cwi["symmetrization"]:
        if cwi["write_hr"]:
            filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_hr_sym.dat.cw"))
            cw_model.write_or(cw_model["Hr_sym"], filename, header=CWModel._hr_header())

            # filename = os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_hr_nonortho_sym.dat.cw"))
            # cw_model.write_or(cw_model["Hr_nonortho_sym"], filename, header=CWModel._hr_header())

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

    # # the order of atoms are different from that of SAMBs
    # atoms_list = list(cw_model._cwi["atoms_frac_shift"].values())
    # atoms_frac = [atoms_list[i] for i in cw_model._cwi["nw2n"]]

    # band calculation
    if cwi["kpoint"] is not None and cwi["kpoint_path"] is not None:
        cwm.log("\n  * calculating band dispersion ... ", None, end="", file=outfile, mode="a")

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

        if cwi["tb_gauge"]:
            atoms_list = list(cwi["atoms_frac"].values())
            atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
        else:
            atoms_frac = None

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

        ### orbital angular momentum ###
        # Lx = np.array(
        #    [
        #        [0, 0, 0, 0],
        #        [0, 0, 0, 1.0j],
        #        [0, 0, 0, 0],
        #        [0, -1.0j, 0, 0],
        #    ]
        # )
        # Ly = np.array(
        #    [
        #        [0, 0, 0, 0],
        #        [0, 0, -1.0j, 0],
        #        [0, 1.0j, 0, 0],
        #        [0, 0, 0, 0],
        #    ]
        # )
        # Lz = np.array(
        #    [
        #        [0, 0, 0, 0],
        #        [0, 0, 0, 0],
        #        [0, 0, 0, -1.0j],
        #        [0, 0, 1.0j, 0],
        #    ]
        # )
        # from numpy import linalg as npl
        # from scipy import linalg as spl

        # Sk_mat = cw_model.fourier_transform_r_to_k(
        #    cw_model["Sr"], cwi["kpoints_path"], cwi["irvec"], cwi["ndegen"], atoms_frac=atoms_frac
        # )
        # S2k_inv = np.array([npl.inv(spl.sqrtm(Sk_mat[k])) for k in range(len(cwi["kpoints_path"]))])

        # identity_2 = np.eye(2)
        # identity_3 = np.eye(3)
        # Lx_ = np.kron(identity_3, np.kron(Lx, identity_2))
        # Ly_ = np.kron(identity_3, np.kron(Ly, identity_2))
        # Lz_ = np.kron(identity_3, np.kron(Lz, identity_2))
        ## Lx_ = np.kron(identity_3, Lx)
        ## Ly_ = np.kron(identity_3, Ly)
        ## Lz_ = np.kron(identity_3, Lz)
        # L = [Lx_, Ly_, Lz_]
        # Lk = np.array([[L[a] for _ in cwi["kpoints_path"]] for a in range(3)])
        ## Lk_H = np.array([Uk.transpose(0, 2, 1).conjugate() @ S2k_inv @ Lk[a] @ S2k_inv @ Uk for a in range(3)])
        # Lk_H = np.array([Uk.transpose(0, 2, 1).conjugate() @ Lk[a] @ Uk for a in range(3)])
        # Lk = np.real(np.diagonal(Lk_H, axis1=2, axis2=3))
        # Lk = Lk.transpose(2, 1, 0)
        #### orbital angular momentum ###

        #### sublattice angular momentum ###
        # C3_site = np.array(
        #    [
        #        [0, 1, 0],
        #        [0, 0, 1],
        #        [1, 0, 0],
        #    ]
        # )

        # identity_4 = np.eye(4)
        # C3 = np.kron(C3_site, np.kron(identity_4, identity_2))
        # Lz_site = 1.0j / np.sqrt(3) * (C3 - np.linalg.inv(C3))
        # Lzk_site = np.array([Lz_site for _ in cwi["kpoints_path"]])
        # Lzk_site_H = Uk.transpose(0, 2, 1).conjugate() @ Lzk_site @ Uk
        # Lzk_site = np.real(np.diagonal(Lzk_site_H, axis1=1, axis2=2))
        # Lzk_site = Lzk_site.transpose(1, 0)
        #### sublattice angular momentum ###

        #### crystal angular momentum (spinless) ###
        # C3_site = np.array(
        #    [
        #        [0, 1, 0],
        #        [0, 0, 1],
        #        [1, 0, 0],
        #    ]
        # )
        # C3_orb = np.array(
        #    [
        #        [1, 0, 0, 0],
        #        [0, 1, 0, 0],
        #        [0, 0, -1.0 / 2, -np.sqrt(3) / 2],
        #        [0, 0, np.sqrt(3) / 2, -1.0 / 2],
        #    ]
        # )

        # C3 = np.kron(C3_site, np.kron(C3_orb, identity_2))
        # Jz_as_c = 1.0j / np.sqrt(3) * (C3 - np.linalg.inv(C3))
        # Jzk_as_c = np.array([Jz_as_c for _ in cwi["kpoints_path"]])
        # Jzk_as_c_H = Uk.transpose(0, 2, 1).conjugate() @ Jzk_as_c @ Uk
        # Jzk_as_c = np.real(np.diagonal(Jzk_as_c_H, axis1=1, axis2=2))
        # Jzk_as_c = Jzk_as_c.transpose(1, 0)
        #### crystal angular momentum (spinless) ###

        #### crystal angular momentum ###
        # C3_site = np.array(
        #    [
        #        [0, 1, 0],
        #        [0, 0, 1],
        #        [1, 0, 0],
        #    ]
        # )
        # C3_orb = np.array(
        #    [
        #        [1, 0, 0, 0],
        #        [0, 1, 0, 0],
        #        [0, 0, -1.0 / 2, -np.sqrt(3) / 2],
        #        [0, 0, np.sqrt(3) / 2, -1.0 / 2],
        #    ]
        # )
        # C3_spn = np.array(
        #    [
        #        [1.0 / 2 - 1.0j * np.sqrt(3) / 2, 0],
        #        [0, 1.0 / 2 + 1.0j * np.sqrt(3) / 2],
        #    ]
        # )

        # C3 = np.kron(C3_site, np.kron(C3_orb, C3_spn))
        # Jz_c = 1.0j / np.sqrt(3) * (C3 - np.linalg.inv(C3))
        # Jzk_c = np.array([Jz_c for _ in cwi["kpoints_path"]])
        # Jzk_c_H = Uk.transpose(0, 2, 1).conjugate() @ Jzk_c @ Uk
        # Jzk_c = np.real(np.diagonal(Jzk_c_H, axis1=1, axis2=2))
        # Jzk_c = Jzk_c.transpose(1, 0)
        #### crystal angular momentum ###

        #### ET dipole ###
        # sigma_x = np.array([[0, 1.0], [1.0, 0.0]]) / np.sqrt(2)
        # sigma_y = np.array([[0, -1.0j], [1.0j, 0.0]]) / np.sqrt(2)
        # Q0_s = np.array(
        #    [
        #        [1, 0, 0],
        #        [0, 1, 0],
        #        [0, 0, 1],
        #    ]
        # )
        # Q0_u = np.array(
        #    [
        #        [0, 1, 1],
        #        [1, 0, 1],
        #        [1, 1, 0],
        #    ]
        # )
        # Gz_ = np.kron(Q0_s, np.kron(Lx, sigma_y)) - np.kron(
        #    Q0_s,
        #    np.kron(Ly, sigma_x),
        # )
        # Gzk = np.array([Gz_ for _ in cwi["kpoints_path"]])
        # Gzk_H = Uk.transpose(0, 2, 1).conjugate() @ Gzk @ Uk
        # Gzk = np.real(np.diagonal(Gzk_H, axis1=1, axis2=2))
        # Gzk = Gzk.transpose(1, 0)
        #### ET dipole ###

        #### Lzk_atomic_Q0u ###
        # Q0_u = np.array(
        #    [
        #        [0, 1, 1],
        #        [1, 0, 1],
        #        [1, 1, 0],
        #    ]
        # )
        # LazQ0u = np.kron(Q0_u, np.kron(Lz, identity_2))
        # LazQ0u = np.array([LazQ0u for _ in cwi["kpoints_path"]])
        # LazQ0u_H = Uk.transpose(0, 2, 1).conjugate() @ LazQ0u @ Uk
        # LazQ0u = np.real(np.diagonal(LazQ0u_H, axis1=1, axis2=2))
        # LazQ0u = LazQ0u.transpose(1, 0)
        #### Lzk_atomic_Q0u ###

        #### Lzk_site_Qua ###
        # C3_site = np.array(
        #    [
        #        [0, 1, 0],
        #        [0, 0, 1],
        #        [1, 0, 0],
        #    ]
        # )

        # identity_4 = np.eye(4)
        # C3 = np.kron(C3_site, np.kron(identity_4, identity_2))
        # Lz_site = 1.0j / np.sqrt(3) * (C3 - np.linalg.inv(C3))

        # Qu_a = np.array(
        #    [
        #        [0, 0, 0, 0],
        #        [0, 2, 0, 0],
        #        [0, 0, -1, 0],
        #        [0, 0, 0, -1],
        #    ]
        # )

        # LszQua = Lz_site @ np.kron(identity_3, np.kron(Qu_a, identity_2))

        # LszQua = np.array([LszQua for _ in cwi["kpoints_path"]])
        # LszQua_H = Uk.transpose(0, 2, 1).conjugate() @ LszQua @ Uk
        # LszQua = np.real(np.diagonal(LszQua_H, axis1=1, axis2=2))
        # LszQua = LszQua.transpose(1, 0)
        #### Lzk_site_Qua ###

        # S_L_k = np.array([Sk[:, :, a] for a in range(3)] + [Lk[:, :, a] for a in range(3)])
        # S_L_k = S_L_k.transpose(1, 2, 0)

        # S_La_Ls_J_k = np.array([Sk[:, :, a] for a in range(3)] + [Lk[:, :, a] for a in range(3)] + [Lzk_site] + [Jzk_c])
        # S_La_Ls_J_k = S_La_Ls_J_k.transpose(1, 2, 0)

        # S_La_Ls_J_Gz_Jas_LazQ0u_LzsQua_k = np.array(
        #    [Sk[:, :, a] for a in range(3)]
        #    + [Lk[:, :, a] for a in range(3)]
        #    + [Lzk_site]
        #    # + [Jzk_c]
        #    + [Lk[:, :, 2] + 0.5 * Sk[:, :, 2]]
        #    + [Gzk]
        #    + [Jzk_as_c]
        #    + [LazQ0u]
        #    + [LszQua]
        # )
        # S_La_Ls_J_Gz_Jas_LazQ0u_LzsQua_k = S_La_Ls_J_Gz_Jas_LazQ0u_LzsQua_k.transpose(1, 2, 0)

        output_linear_dispersion_eig(
            ".",
            seedname + "_band.txt",
            k_linear,
            e=Ek,
            # o=Lk,
            # o=S_La_Ls_J_Gz_Jas_LazQ0u_LzsQua_k,
            o=Sk,
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
            ket_samb = samb_info["ket"]
            cell_site = samb_info["cell_site"]

            if cwi["tb_gauge"]:
                atoms_frac = [
                    NSArray(cell_site[ket_samb[a].split("@")[1]][0], style="vector", fmt="value").tolist()
                    for a in range(cw_model._cwi["num_wann"])
                ]
            else:
                atoms_frac = None

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

            if cwi["calc_spin_2d"]:
                SS_R = get_oper_R("SS_R", cwi)
                SS_k = cw_model.fourier_transform_r_to_k_vec(
                    SS_R, cwi["kpoints_path"], cwi["irvec"], cwi["ndegen"], atoms_frac=atoms_frac
                )
                ket_samb = samb_info["ket"]
                ket_amn = cwi.get("ket_amn", ket_samb)
                SS_k = np.array([sort_ket_matrix(SS_k[a], ket_amn, ket_samb) for a in range(3)])
                SS_H = np.array([Uk.transpose(0, 2, 1).conjugate() @ SS_k[a] @ Uk for a in range(3)])
                Sk = np.real(np.diagonal(SS_H, axis1=2, axis2=3))
                Sk = Sk.transpose(2, 1, 0)
            else:
                Sk = None

            # Lx = np.array(
            #    [
            #        [0, 0, 0, 0],
            #        [0, 0, 0, 0],
            #        [0, 0, 0, -1.0j],
            #        [0, 0, 1.0j, 0],
            #    ]
            # )
            # Ly = np.array(
            #    [
            #        [0, 0, 0, 0],
            #        [0, 0, 0, 1.0j],
            #        [0, 0, 0, 0],
            #        [0, -1.0j, 0, 0],
            #    ]
            # )
            # Lz = np.array(
            #    [
            #        [0, 0, 0, 0],
            #        [0, 0, -1.0j, 0],
            #        [0, 1.0j, 0, 0],
            #        [0, 0, 0, 0],
            #    ]
            # )

            # identity_2 = np.eye(2)
            # identity_3 = np.eye(3)
            # Lx = np.kron(identity_3, np.kron(Lx, identity_2))
            # Ly = np.kron(identity_3, np.kron(Ly, identity_2))
            # Lz = np.kron(identity_3, np.kron(Lz, identity_2))
            ## Lx = np.kron(identity_3, Lx)
            ## Ly = np.kron(identity_3, Ly)
            ## Lz = np.kron(identity_3, Lz)
            # L = [Lx, Ly, Lz]
            # Lk = np.array([[L[a] for _ in cwi["kpoints_path"]] for a in range(3)])
            # Lk_H = np.array([Uk.transpose(0, 2, 1).conjugate() @ Lk[a] @ Uk for a in range(3)])
            # Lk = np.real(np.diagonal(Lk_H, axis1=2, axis2=3))
            # Lk = Lk.transpose(2, 1, 0)
            ##### orbital angular momentum ###

            ##
            ##
            ##

            # model = cwm.read(os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_model.py")))
            # samb = cwm.read(os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_samb.py")))

            # try:
            #    mat = cwm.read(os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_matrix.pkl")))
            # except:
            #    mat = cwm.read(os.path.join(cwi["mp_outdir"], "{}".format(f"{cwi['mp_seedname']}_matrix.py")))

            # if cwi["irreps"] == "all":
            #    irreps = model["info"]["generate"]["irrep"]
            # elif cwi["irreps"] == "full":
            #    irreps = [model["info"]["generate"]["irrep"][0]]
            # else:
            #    irreps = cwi["irreps"]

            # from multipie.tag.tag_multipole import TagMultipole

            # for zj, (tag, _) in samb["data"]["Z"].items():
            #    if TagMultipole(tag).irrep not in irreps:
            #        del mat["matrix"][zj]

            # tag_dict = {zj: tag for zj, (tag, _) in samb["data"]["Z"].items()}

            ## Zr_dict = {
            ##     (zj, tag_dict[zj]): {tuple(sp.sympify(k)): complex(sp.sympify(v)) for k, v in d.items()}
            ##     for zj, d in mat["matrix"].items()
            ## }
            ## mat["matrix"] = {
            ##     zj: {tuple(sp.sympify(k)): complex(sp.sympify(v)) for k, v in d.items()} for zj, d in mat["matrix"].items()
            ## }

            #### kuniyoshi (24/08/20) ###
            # import sympy as sp
            # import multiprocessing
            # from joblib import Parallel, delayed, wrap_non_picklable_objects
            # from symclosestwannier.util.utility import construct_Ok

            # _num_proc = multiprocessing.cpu_count()

            # def proc(j, zj, d):
            #    return j, zj, {tuple(sp.sympify(k)): complex(sp.sympify(v)) for k, v in d.items()}

            # res = Parallel(n_jobs=_num_proc, verbose=1)(
            #    delayed(proc)(j, zj, d) for j, (zj, d) in enumerate(mat["matrix"].items())
            # )
            # res = sorted(res, key=lambda x: x[0])

            # Zr_dict = {}
            # for _, zj, d in res:
            #    Zr_dict[(zj, tag_dict[zj])] = d
            #    mat["matrix"][zj] = d

            # atoms_list = list(cwi["atoms_frac"].values())
            # atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])

            # atoms_frac_samb = [
            #    NSArray(mat["cell_site"][ket_samb[a].split("@")[1]][0], style="vector", fmt="value").tolist()
            #    for a in range(cwi["num_wann"])
            # ]

            #### kuniyoshi (24/08/20) ###

            # dic = mat.copy()
            #### Gu local (i = 9) ###
            ## Gu local (i = 9)
            # tag, d = list(Zr_dict.items())[9]
            # dic["matrix"] = {tag: d}
            # print(tag)
            # if cwi["tb_gauge"]:
            #    Gu_local_k = construct_Ok(
            #        [1], cwi["num_wann"], cwi["kpoints_path"], cwi["irvec"], dic, atoms_frac=atoms_frac_samb
            #    )
            # else:
            #    Gu_local_k = construct_Ok([1], cwi["num_wann"], cwi["kpoints_path"], cwi["irvec"], dic)

            # print((Gu_local_k[0]).tolist())

            # Gu_local_k_H = Uk.transpose(0, 2, 1).conjugate() @ Gu_local_k @ Uk
            # Gu_local_k = np.real(np.diagonal(Gu_local_k_H, axis1=1, axis2=2))
            # Gu_local_k = Gu_local_k.transpose(1, 0)
            #### Gu local (i = 9) ###

            #### Gu B1 (i = 33) ###
            # tag, d = list(Zr_dict.items())[33]
            # dic["matrix"] = {tag: d}
            # print(tag)
            # if cwi["tb_gauge"]:
            #    Gu_B1_k = construct_Ok(
            #        [1], cwi["num_wann"], cwi["kpoints_path"], cwi["irvec"], dic, atoms_frac=atoms_frac_samb
            #    )
            # else:
            #    Gu_B1_k = construct_Ok([1], cwi["num_wann"], cwi["kpoints_path"], cwi["irvec"], dic)

            # print((Gu_B1_k[0]).tolist())

            # Gu_B1_k_H = Uk.transpose(0, 2, 1).conjugate() @ Gu_B1_k @ Uk
            # Gu_B1_k = np.real(np.diagonal(Gu_B1_k_H, axis1=1, axis2=2))
            # Gu_B1_k = Gu_B1_k.transpose(1, 0)
            #### Gu B1 (i = 33) ###

            ##
            ##
            ##

            # S_L_Gu_local_Gu_B1_k = np.array(
            #    [Sk[:, :, a] for a in range(3)] + [Lk[:, :, a] for a in range(3)] + [Gu_local_k] + [Gu_B1_k]
            # )
            # S_L_Gu_local_Gu_B1_k = S_L_Gu_local_Gu_B1_k.transpose(1, 2, 0)

            # output_linear_dispersion(
            output_linear_dispersion_eig(
                cwi["mp_outdir"],
                cwi["mp_seedname"] + "_band.txt",
                k_linear,
                e=Ek,
                o=Sk,
                # o=Lk,
                # o=S_L_k,
                # o=S_L_Gu_local_Gu_B1_k,
                ref_filename=ref_filename,
                a=a,
                ef=ef,
                k_dis_pos=k_dis_pos,
            )

            # output_linear_dispersion(
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

        output_dos(".", seedname + "_dos.txt", Ek, Uk, ef_shift, dos_num_fermi, dos_smr_en_width)

        if cwi["symmetrization"]:
            ket_samb = samb_info["ket"]
            cell_site = samb_info["cell_site"]
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

            Hk_sym_grid = cw_model.fourier_transform_r_to_k(
                cw_model["Hr_sym"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac=atoms_frac
            )
            Ek, Uk = np.linalg.eigh(Hk_sym_grid)

            # output_linear_dispersion(
            output_dos(
                cwi["mp_outdir"], cwi["mp_seedname"] + "_dos.txt", Ek, Uk, ef_shift, dos_num_fermi, dos_smr_en_width
            )

        cwm.log("done", end="\n", file=outfile, mode="a")

    #####

    cwm.log(f"\n\n  * total elapsed_time:", file=outfile, mode="a")
    cwm.log(cw_end_output_msg(), stamp=None, end="\n", file=outfile, mode="a")

    cwm.log(f"  * total elapsed_time:", stamp="start", file=outfile, mode="a")
    cwm.log(cw_end_msg(), stamp=None, end="\n", file=outfile, mode="a")
