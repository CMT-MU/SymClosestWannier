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
#         get_matrix_R: matrix elements of various operators         #
#                                                                    #
# ****************************************************************** #

import numpy as np

from symclosestwannier.util.utility import fourier_transform_k_to_r


# ==================================================
def get_oper_R(name, cwi):
    """
    wrapper for getting matrix elements of the operator.

    Args:
        cwi (CWInfo): CWInfo.

    Returns:
        ndarray: matrix elements of the operator.
    """
    d = {
        "HH_R": get_HH_R,  # <0n|H|Rm>
        "AA_R": get_AA_R,  # <0n|r|Rm>
        "BB_R": get_BB_R,  # <0|H(r-R)|R>
        "CC_R": get_CC_R,  # <0|r_alpha.H(r-R)_beta|R>
        "SS_R": get_SS_R,  # <0n|sigma_x,y,z|Rm>
        "get_SHC_R": get_SHC_R,  # <0n|sigma_x,y,z.(r-R)_alpha|Rm>, <0n|sigma_x,y,z.H.(r-R)_alpha|Rm>, <0n|sigma_x,y,z.H|Rm>
        "SAA_R": get_SAA_R,  # <0n|sigma_x,y,z.(r-R)_alpha|Rm>
        "SBB_R": get_SBB_R,  # <0n|sigma_x,y,z.H.(r-R)_alpha|Rm>
    }

    return d[name](cwi)


# ==================================================
def get_HH_R(cwi):
    """
    matrix elements of real-space Hamiltonian, <0n|H|Rm>.

    Args:
        cwi (CWInfo): CWInfo.

    Returns:
        ndarray: Hamiltonian, HH_R(len(irvec), num_wann, num_wann).
    """
    Ek = np.array(cwi["Ek"])
    Uk = np.array(cwi["Uk"])

    HH_k = np.einsum("klm,kl,kln->kmn", np.conj(Uk), Ek, Uk, optimize=True)
    HH_k = 0.5 * (HH_k + np.einsum("kmn->knm", HH_k).conj())

    kpoints = np.array(cwi["kpoints"])
    irvec = np.array(cwi["irvec"])

    if cwi["tb_gauge"]:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    HH_R = fourier_transform_k_to_r(HH_k, kpoints, irvec, atoms_frac)

    return HH_R


# ==================================================
def get_AA_R(cwi):
    """
    matrix elements of real-space position operator, <0n|r|Rm>.

    Args:
        cwi (CWInfo): CWInfo.

    Returns:
        ndarray: position operator, AA_R(3, len(irvec), num_wann, num_wann).
    """
    Mkb = np.array(cwi["Mkb"])
    Uk = np.array(cwi["Uk"])

    kb2k = cwi.nnkp.kb2k()
    bveck = cwi.nnkp.bveck()
    wk = cwi.nnkp.wk()
    wb = cwi["wb"]

    kpoints = np.array(cwi["kpoints"])
    irvec = np.array(cwi["irvec"])

    ### Unitary transform Mkb ###
    Mkb_w = np.einsum("klm, kblp, kbpn->kbmn", np.conj(Uk), Mkb, Uk[kb2k[:, :], :, :], optimize=True)  # Eq. (61)
    AA_k = 1.0j * np.einsum("b,kba,kbmn->akmn", wb, bveck, Mkb_w, optimize=True)

    # Use Eq.(31) of Marzari&Vanderbilt PRB 56, 12847 (1997) for band-diagonal position matrix.
    if cwi["transl_inv"]:
        AA_k_diag = -np.einsum("b,kba,kbnn->akn", wb, bveck, np.imag(np.log(Mkb_w)), optimize=True)
        np.einsum("aknn->akn", AA_k)[:] = AA_k_diag

    AA_k = 0.5 * (AA_k + np.einsum("akmn->aknm", AA_k).conj())

    if cwi["tb_gauge"]:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    AA_R = np.array([fourier_transform_k_to_r(AA_k[i], kpoints, irvec, atoms_frac) for i in range(3)])

    return AA_R


# ==================================================
def get_BB_R(cwi):
    """
    matrix elements of real-space BB operator,
        BB_a(R)=<0n|H(r-R)_a|Rm>

    BB_a(R) is the Fourier transform of
        BB_a(k) = i<u|H|del_a u> (a=x,y,z)

    Args:
        cwi (CWInfo): CWInfo.

    Returns:
        ndarray: position operator, BB_R(3, len(irvec), num_wann, num_wann).
    """
    if abs(cwi.get("scissors_shift", 0.0)) > 1.0e-7:
        raise Exception("Error: scissors correction not yet implemented for BB_R")

    num_bands = cwi["num_bands"]
    num_wann = cwi["num_wann"]
    num_k = cwi["num_k"]
    kpoints = np.array(cwi["kpoints"])
    irvec = np.array(cwi["irvec"])

    kb2k = cwi.nnkp.kb2k()
    bveck = cwi.nnkp.bveck()
    wk = cwi.nnkp.wk()
    wb = cwi["wb"]

    Ek = np.array(cwi["Ek"])
    Uk = np.array(cwi["Uk"])
    Mkb = np.array(cwi["Mkb"])

    H_o = np.array([np.diag(Ek[k]) for k in range(num_k)])

    HM_o = np.einsum("kml, kbln->kbmn", H_o, Mkb, optimize=True)
    H_k_kb = np.einsum("klm, kblp, kbpn->kbmn", np.conj(Uk), HM_o, Uk[kb2k[:, :], :, :], optimize=True)
    BB_k = 1.0j * np.einsum("b,kbc,kbmn->ckmn", wb, bveck, H_k_kb, optimize=True)

    if cwi["tb_gauge"]:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    BB_R = np.array([fourier_transform_k_to_r(BB_k[i], kpoints, irvec, atoms_frac) for i in range(3)])

    return BB_R


# ==================================================
def get_CC_R(cwi):
    """
    matrix elements of real-space CC operator,
        CC_ab(R)=<0n|r_a.H.(r-R)_b|Rm>

    CC_ab(R) is the Fourier transform of
        CC_ab(k) = <del_a u|H|del_b u> (a,b=x,y,z)

    Args:
        cwi (CWInfo): CWInfo.

    Returns:
        ndarray: position operator, CC_R(3, 3, len(irvec), num_wann, num_wann).

    """
    if abs(cwi.get("scissors_shift", 0.0)) > 1.0e-7:
        raise Exception("Error: scissors correction not yet implemented for CC_R")

    num_bands = cwi["num_bands"]
    num_wann = cwi["num_wann"]
    num_k = cwi["num_k"]
    kpoints = np.array(cwi["kpoints"])
    irvec = np.array(cwi["irvec"])

    kb2k = cwi.nnkp.kb2k()
    bveck = cwi.nnkp.bveck()
    wk = cwi.nnkp.wk()
    wb = cwi["wb"]

    Ek = np.array(cwi["Ek"])
    Uk = np.array(cwi["Uk"])
    Mkb = np.array(cwi["Mkb"])
    Hkb1b2 = np.array(cwi["Hkb1b2"])

    H_o = np.array([np.diag(Ek[k]) for k in range(num_k)])

    Hkb1b2 = np.einsum(
        "kblm, kbdlp, kdpn->kbdmn", np.conj(Uk[kb2k[:, :], :, :]), Hkb1b2, Uk[kb2k[:, :], :, :], optimize=True
    )
    CC_k = np.einsum("b,kbi,d,kdj,kbdmn->ijkmn", wb, bveck, wb, bveck, Hkb1b2, optimize=True)

    if cwi["tb_gauge"]:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    CC_R = np.array(
        [[fourier_transform_k_to_r(CC_k[i, j], kpoints, irvec, atoms_frac) for j in range(3)] for i in range(3)]
    )

    return CC_R


# ==================================================
def get_SS_R(cwi):
    """
    matrix elements of real-space spin operator, <0n|sigma_x,y,z|Rm>.

    Args:
        cwi (CWInfo): CWInfo.

    Returns:
        ndarray: spin operator, SS_R(3, len(irvec), num_wann, num_wann).
    """
    pauli_spn = np.array(cwi["pauli_spn"])
    Uk = np.array(cwi["Uk"])

    SS_k = np.einsum("klm,aklp,kpn->akmn", np.conj(Uk), pauli_spn, Uk, optimize=True)

    SS_k = 0.5 * (SS_k + np.einsum("akmn->aknm", SS_k).conj())

    kpoints = np.array(cwi["kpoints"])
    irvec = np.array(cwi["irvec"])

    if cwi["tb_gauge"]:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    SS_R = np.array([fourier_transform_k_to_r(SS_k[i], kpoints, irvec, atoms_frac) for i in range(3)])

    return SS_R


# ==================================================
def get_SHC_R(cwi):
    """
    Compute several matrices for spin Hall conductivity
        - SR_R  = <0n|sigma_{x,y,z}.(r-R)_alpha|Rm>
        - SHR_R = <0n|sigma_{x,y,z}.H.(r-R)_alpha|Rm>
        - SH_R  = <0n|sigma_{x,y,z}.H|Rm>

    Args:
        cwi (CWInfo): CWInfo.

    Returns:
        tuple: SR_R(3, 3, len(irvec), num_wann, num_wann), SHR_R(3, 3, len(irvec), num_wann, num_wann), SH_R(3, len(irvec), num_wann, num_wann).
    """
    kpoints = np.array(cwi["kpoints"])
    irvec = np.array(cwi["irvec"])
    num_wann = cwi["num_wann"]
    num_bands = cwi["num_bands"]
    num_k = cwi["num_k"]

    if cwi["tb_gauge"]:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    Ek = np.array(cwi["Ek"])
    Uk = np.array(cwi["Uk"])

    # spin operator
    spn_o = np.array(cwi["pauli_spn"])
    SS_k = np.einsum("klm,aklp,kpn->akmn", np.conj(Uk), spn_o, Uk, optimize=True)

    # get_HH_R
    shc_bandshift = cwi["shc_bandshift"]
    shc_bandshift_firstband = cwi["shc_bandshift_firstband"]
    shc_bandshift_energyshift = cwi["shc_bandshift_energyshift"]

    if shc_bandshift:
        Ek[:, shc_bandshift_firstband:] += shc_bandshift_energyshift

    H_o = np.array([np.diag(Ek[k]) for k in range(num_k)])

    # get_AA_R
    Mkb = np.array(cwi["Mkb"])
    kb2k = cwi.nnkp.kb2k()
    bveck = cwi.nnkp.bveck()
    wk = cwi.nnkp.wk()
    wb = cwi["wb"]

    #! QZYZ18 Eq.(48)
    SH_o = spn_o @ H_o[np.newaxis, :, :, :]
    SH_k = np.einsum("klm, aklp, kpn->akmn", np.conj(Uk), SH_o, Uk, optimize=True)

    #! QZYZ18 Eq.(50)
    SM_o = np.einsum("akml, kbln->akbmn", spn_o, Mkb, optimize=True)
    SM_k = np.einsum("klm, akblp, kbpn->akbmn", np.conj(Uk), SM_o, Uk[kb2k[:, :], :, :], optimize=True)
    # SR_k = np.einsum("kb,kbc,akbmn->ackmn", wk, bveck, SM_k, optimize=True) - np.einsum(
    #     "kb,kbc,akmn->ackmn", wk, bveck, SS_k, optimize=True
    # )
    SR_k = np.einsum("b,kbc,akbmn->ackmn", wb, bveck, SM_k, optimize=True) - np.einsum(
        "b,kbc,akmn->ackmn", wb, bveck, SS_k, optimize=True
    )

    #! QZYZ18 Eq.(51)
    SHM_o = np.einsum("akml, kbln->akbmn", SH_o, Mkb, optimize=True)
    SHM_k = np.einsum("klm, akblp, kbpn->akbmn", np.conj(Uk), SHM_o, Uk[kb2k[:, :], :, :], optimize=True)
    # SHR_k = np.einsum("kb,kbc,akbmn->ackmn", wk, bveck, SHM_k, optimize=True) - np.einsum(
    #     "kb,kbc,akmn->ackmn", wk, bveck, SH_k, optimize=True
    # )
    SHR_k = np.einsum("b,kbc,akbmn->ackmn", wb, bveck, SHM_k, optimize=True) - np.einsum(
        "b,kbc,akmn->ackmn", wb, bveck, SH_k, optimize=True
    )

    SH_R = np.array([fourier_transform_k_to_r(SH_k[i], kpoints, irvec, atoms_frac) for i in range(3)])
    SR_R = np.array(
        [[fourier_transform_k_to_r(SR_k[i][j], kpoints, irvec, atoms_frac) for j in range(3)] for i in range(3)]
    )
    SHR_R = np.array(
        [[fourier_transform_k_to_r(SHR_k[i][j], kpoints, irvec, atoms_frac) for j in range(3)] for i in range(3)]
    )

    SR_R = 1.0j * SR_R
    SHR_R = 1.0j * SHR_R

    return SR_R, SHR_R, SH_R


# ==================================================
def get_SAA_R(cwi):
    """<0n|sigma_x,y,z.(r-R)_alpha|Rm>"""
    pass


# ==================================================
def get_SBB_R(cwi):
    """<0n|sigma_x,y,z.H.(r-R)_alpha|Rm>"""
    pass


# ******************************************************************
# ******************************************************************
# ******************************************************************


# ==================================================
def get_berry_phase_R(cwi):
    """
    matrix elements of berry phase, <0n|A_x,y,z|Rm>.

    Args:
        cwi (CWInfo): CWInfo.

    Returns:
        ndarray: spin operator, SS_R(3, len(irvec), num_wann, num_wann).
    """
    Mkb = np.array(cwi["Mkb"])
    Uk = np.array(cwi["Uk"])
    num_wann = cwi["num_wann"]

    ### Unitary transform Mkb ###
    kb2k = cwi.nnkp.kb2k()
    Mkb_w = np.einsum("klm, kblp, kbpn->kbmn", np.conj(Uk), Mkb, Uk[kb2k[:, :], :, :], optimize=True)  # Eq. (61)

    bveck = cwi.nnkp.bveck()
    wk = cwi.nnkp.wk()

    # i<wik|∇wjk>
    a_k = 1j * np.einsum("kb,kba,kbmn->akmn", wk, bveck, (Mkb_w - np.eye(num_wann)), optimize=True)
    a_k = 0.5 * (a_k + np.einsum("akmn->aknm", a_k.conj()))

    kpoints = np.array(cwi["kpoints"])
    irvec = np.array(cwi["irvec"])

    if cwi["tb_gauge"]:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    a_R = np.array([fourier_transform_k_to_r(a_k[i], kpoints, irvec, atoms_frac) for i in range(3)])

    return a_R


# ==================================================
def get_berry_Curvature_R(cwi):
    """<0n|Ω|Rm>"""
    pass


# ==================================================
def get_der_berry_Curvature_Rcwi():
    """<0n|∇Ω|Rm>"""

    pass


# ==================================================
def get_orbital_moment_R(cwi):
    """<0n|Morb|Rm>"""

    pass


# ==================================================
def get_der_orbital_moment_R(cwi):
    """<0n|∇Morb|Rm>"""

    pass


# ==================================================
def get_velocity_R(cwi):
    """<0n|v|Rm>"""
    pass
