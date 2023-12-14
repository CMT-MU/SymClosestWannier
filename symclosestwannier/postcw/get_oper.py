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
#          get_oper: matrix elements of various operators            #
#                                                                    #
# ****************************************************************** #

from symclosestwannier.util._utility import (
    weight_proj,
    fourier_transform_k_to_r,
    fourier_transform_r_to_k,
    interpolate,
    matrix_dict_r,
    matrix_dict_k,
    dict_to_matrix,
    samb_decomp,
    construct_Or,
    construct_Ok,
)


# ==================================================
def get_HH_R(cwi):
    """
    matrix elements of real-space Hamiltonian, <0n|H|Rm>.

    Args:
        cwi (SystemInfo): CWInfo.

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
    atoms_frac = np.array([cwi["atom_pos_r"][idx] for idx in cwi["nw2n"]])

    HH_R = fourier_transform_k_to_r(HH_k, kpoints, irvec, atoms_frac)

    return HH_R


# ==================================================
def get_AA_R(cwi):
    """
    matrix elements of real-space position operator, <0n|r|Rm>.

    Args:
        cwi (SystemInfo): CWInfo.

    Returns:
        ndarray: position operator, AA_R(3, len(irvec), num_wann, num_wann).
    """
    Mkb = np.array(cwi["Mkb"])
    Uk = np.array(cwi["Uk"])
    num_k = cwi["num_k"]

    ### Unitary transform Mkb ###
    kb2k = cwi.nnkp.kb2k()
    Mkb_w = np.einsum("klm, kblp, kbpn->kbmn", np.conj(Uk), Mkb, Uk[kb2k[:, :], :, :], optimize=True)  # Eq. (61)

    bveck = cwi.nnkp.bveck()
    wk = cwi.nnkp.wk()

    AA_k = 1j * np.einsum("kb,kba,kbmn->akmn", wk, bveck, Mkb_w, optimize=True)
    AA_k_diag = -1 * np.einsum("kb,kba,kbnn->akn", wk, bveck, np.imag(np.log(Mkb_w)), optimize=True)
    np.einsum("aknn->akn", AA_k)[:] = AA_k_diag
    AA_k = 0.5 * (AA_k + np.einsum("akmn->aknm", AA_k.conj()))

    kpoints = np.array(cwi["kpoints"])
    irvec = np.array(cwi["irvec"])
    atoms_frac = np.array([cwi["atom_pos_r"][idx] for idx in cwi["nw2n"]])

    AA_R = fourier_transform_k_to_r(AA_k, kpoints, irvec, atoms_frac)

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
def get_SS_R():
    """
    matrix elements of real-space spin operator, <0n|sigma_x,y,z|Rm>.

    Args:
        cwi (SystemInfo): CWInfo.

    Returns:
        ndarray: spin operator, SS_R(3, len(irvec), num_wann, num_wann).
    """
    Sk = np.array(cwi["Sk"])
    Uk = np.array(cwi["Uk"])
    num_k = cwi["num_k"]

    SS_k = Uk.transpose(0, 2, 1).conjugate() @ Sk @ Uk
    SS_k = 0.5 * (SS_k + np.einsum("kmn->knm", SS_k).conj())

    kpoints = np.array(cwi["kpoints"])
    irvec = np.array(cwi["irvec"])
    atoms_frac = np.array([cwi["atom_pos_r"][idx] for idx in cwi["nw2n"]])

    SS_R = fourier_transform_k_to_r(SS_k, kpoints, irvec, atoms_frac)

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
