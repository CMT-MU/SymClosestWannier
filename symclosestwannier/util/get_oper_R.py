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

from symclosestwannier.util._utility import fourier_transform_k_to_r


# ==================================================
def get_oper_R(name, cwi, tb_gauge=False):
    """
    wrapper for getting matrix elements of the operator.

    Args:
        cwi (SystemInfo): CWInfo.
        tb_gauge (bool, optional): tb gauge?

    Returns:
        ndarray: matrix elements of the operator.
    """
    d = {
        "HH_R": get_HH_R,  # <0n|H|Rm>
        "AA_R": get_AA_R,  # <0n|r|Rm>
        "BB_R": get_BB_R,  # <0|H(r-R)|R>
        "CC_R": get_CC_R,  # <0|r_alpha.H(r-R)_beta|R>
        "SS_R": get_SS_R,  # <0n|sigma_x,y,z|Rm>
        "SR_R": get_SR_R,  # <0n|sigma_x,y,z.(r-R)_alpha|Rm>
        "SHR_R": get_SHR_R,  # <0n|sigma_x,y,z.H.(r-R)_alpha|Rm>
        "SH_R": get_SH_R,  # <0n|sigma_x,y,z.H|Rm>
        "SAA_R": get_SAA_R,  # <0n|sigma_x,y,z.(r-R)_alpha|Rm>
        "SBB_R": get_SBB_R,  # <0n|sigma_x,y,z.H.(r-R)_alpha|Rm>
    }

    return d[name](cwi, tb_gauge)


# ==================================================
def get_HH_R(cwi, tb_gauge=False):
    """
    matrix elements of real-space Hamiltonian, <0n|H|Rm>.

    Args:
        cwi (SystemInfo): CWInfo.
        tb_gauge (bool, optional): tb gauge?

    Returns:
        ndarray: Hamiltonian, HH_R(len(irvec), num_wann, num_wann).
    """
    Ek = np.array(cwi["Ek"])
    Uk = np.array(cwi["Uk"])
    num_k = cwi["num_k"]

    diag_Ek = np.array([np.diag(Ek[k]) for k in range(num_k)])
    HH_k = Uk.transpose(0, 2, 1).conjugate() @ diag_Ek @ Uk
    HH_k = 0.5 * (HH_k + np.einsum("kmn->knm", HH_k).conj())

    kpoints = np.array(cwi["kpoints"])
    irvec = np.array(cwi["irvec"])

    if tb_gauge:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    HH_R = fourier_transform_k_to_r(HH_k, kpoints, irvec, atoms_frac)

    return HH_R


# ==================================================
def get_AA_R(cwi, tb_gauge=False):
    """
    matrix elements of real-space position operator, <0n|r|Rm>.

    Args:
        cwi (SystemInfo): CWInfo.
        tb_gauge (bool, optional): tb gauge?

    Returns:
        ndarray: position operator, AA_R(3, len(irvec), num_wann, num_wann).
    """
    Mkb = np.array(cwi["Mkb"])
    Uk = np.array(cwi["Uk"])

    kb2k = cwi.nnkp.kb2k()
    bveck = cwi.nnkp.bveck()
    wk = cwi.nnkp.wk()

    kpoints = np.array(cwi["kpoints"])
    irvec = np.array(cwi["irvec"])

    ### Unitary transform Mkb ###
    Mkb_w = np.einsum("klm, kblp, kbpn->kbmn", np.conj(Uk), Mkb, Uk[kb2k[:, :], :, :], optimize=True)  # Eq. (61)

    AA_k = 1j * np.einsum("kb,kba,kbmn->akmn", wk, bveck, Mkb_w, optimize=True)
    AA_k_diag = -1 * np.einsum("kb,kba,kbnn->akn", wk, bveck, np.imag(np.log(Mkb_w)), optimize=True)
    np.einsum("aknn->akn", AA_k)[:] = AA_k_diag

    AA_k = 0.5 * (AA_k + np.einsum("akmn->aknm", AA_k).conj())

    if tb_gauge:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    AA_R = np.array([fourier_transform_k_to_r(AA_k[i], kpoints, irvec, atoms_frac) for i in range(3)])

    return AA_R


# ==================================================
def get_BB_R():
    """<0|H(r-R)|R>"""
    pass


# ==================================================
def get_CC_R():
    """<0|r_alpha.H(r-R)_beta|R>"""
    pass


# ==================================================
def get_SS_R(cwi, tb_gauge=False):
    """
    matrix elements of real-space spin operator, <0n|sigma_x,y,z|Rm>.

    Args:
        cwi (SystemInfo): CWInfo.
        tb_gauge (bool, optional): tb gauge?

    Returns:
        ndarray: spin operator, SS_R(3, len(irvec), num_wann, num_wann).
    """
    Sk = np.array(cwi["Sk"])
    Uk = np.array(cwi["Uk"])

    SS_k = Uk.transpose(0, 2, 1).conjugate() @ Sk @ Uk
    SS_k = 0.5 * (SS_k + np.einsum("akmn->aknm", SS_k).conj())

    kpoints = np.array(cwi["kpoints"])
    irvec = np.array(cwi["irvec"])

    if tb_gauge:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    SS_R = np.array([fourier_transform_k_to_r(SS_k[i], kpoints, irvec, atoms_frac) for i in range(3)])

    return SS_R


# ==================================================
def get_SR_R():
    """<0n|sigma_x,y,z.(r-R)_alpha|Rm>"""
    pass


# ==================================================
def get_SHR_R():
    """<0n|sigma_x,y,z.H.(r-R)_alpha|Rm>"""
    pass


# ==================================================
def get_SH_R():
    """<0n|sigma_x,y,z.H|Rm>"""
    pass


# ==================================================
def get_SAA_R():
    """<0n|sigma_x,y,z.(r-R)_alpha|Rm>"""
    pass


# ==================================================
def get_SBB_R():
    """<0n|sigma_x,y,z.H.(r-R)_alpha|Rm>"""
    pass


# ******************************************************************
# ******************************************************************
# ******************************************************************


# ==================================================
def get_berry_phase_R(cwi, tb_gauge=False):
    """
    matrix elements of real-space spin operator, <0n|A_x,y,z|Rm>.

    Args:
        cwi (SystemInfo): CWInfo.
        tb_gauge (bool, optional): tb gauge?

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

    if tb_gauge:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
        # atoms_frac = np.array([cwi["atom_pos_r"][idx] for idx in cwi["nw2n"]])
    else:
        atoms_frac = None

    a_R = np.array([fourier_transform_k_to_r(a_k[i], kpoints, irvec, atoms_frac) for i in range(3)])

    return a_R


# ==================================================
def get_berry_Curvature_R():
    """<0n|Ω|Rm>"""
    pass


# ==================================================
def get_der_berry_Curvature_R():
    """<0n|∇Ω|Rm>"""

    pass


# ==================================================
def get_orbital_moment_R():
    """<0n|Morb|Rm>"""

    pass


# ==================================================
def get_der_orbital_moment_R():
    """<0n|∇Morb|Rm>"""

    pass


# ==================================================
def get_velocity_R():
    """<0n|v|Rm>"""
    pass


# ==================================================
def get_spin_velocity_R():
    """<0n|{s,v}/2|Rm>"""

    pass
