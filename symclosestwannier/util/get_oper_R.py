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

from symclosestwannier.util.constants import elem_charge_SI, hbar_SI
from symclosestwannier.util.utility import fourier_transform_r_to_k, fourier_transform_k_to_r, orbital_positions
from symclosestwannier.util.exceptions import SymCWInputError


# ==================================================
def _require(cwi, key, ext, name):
    """
    check that the data read from seedname.ext is available.

    Args:
        cwi (CWInfo): CWInfo.
        key (str): key of the data.
        ext (str): extension of the file.
        name (str): name of the operator to be calculated.
    """
    if cwi.get(key) is None:
        raise SymCWInputError(f"{name} requires {cwi['seedname']}.{ext}, which is not found or not read.")


# ==================================================
def get_oper_R(name, cwi, *args):
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
        "AA_R_tb": get_AA_R_tb,  # <0n|r|Rm> ≈ τ_n δ_nm δ_R0
        "BB_R": get_BB_R,  # <0|H(r-R)|R>
        "CC_R": get_CC_R,  # <0|r_alpha.H(r-R)_beta|R>
        "SS_R": get_SS_R,  # <0n|sigma_x,y,z|Rm>
        "SHC_R": get_SHC_R,  # <0n|sigma_x,y,z.(r-R)_alpha|Rm>, <0n|sigma_x,y,z.H.(r-R)_alpha|Rm>, <0n|sigma_x,y,z.H|Rm>
        "SAA_R": get_SAA_R,  # <0n|sigma_x,y,z.(r-R)_alpha|Rm>
        "SBB_R": get_SBB_R,  # <0n|sigma_x,y,z.H.(r-R)_alpha|Rm>
        "v_R": get_v_R,
    }

    return d[name](cwi, *args)


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

    HH_R = fourier_transform_k_to_r(HH_k, kpoints, irvec)

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
    _require(cwi, "Mkb", "mmn", "position operator (AA_R)")
    Mkb = np.array(cwi["Mkb"])
    Uk = np.array(cwi["Uk"])

    kb2k = cwi.nnkp.kb2k()
    bveck = cwi.nnkp.bveck()
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

    AA_R = np.array([fourier_transform_k_to_r(AA_k[i], kpoints, irvec) for i in range(3)])

    return AA_R


# ==================================================
def get_AA_R_tb(cwi):
    """
    position operator in the tight-binding approximation, <0n|r|Rm> = τ_n δ_nm δ_R0,
    with the projection centres τ_n (wannier90 convention). seedname.mmn is not needed.

    Args:
        cwi (CWInfo): CWInfo.

    Returns:
        ndarray: position operator, AA_R(3, len(irvec), num_wann, num_wann).
    """
    irvec = np.array(cwi["irvec"])
    num_wann = cwi["num_wann"]
    tau = orbital_positions(cwi) @ np.array(cwi["unit_cell_cart"], dtype=float)

    R0 = np.where(np.all(irvec == 0, axis=1))[0]
    if len(R0) != 1:
        raise ValueError("R = 0 is not in irvec.")

    AA_R = np.zeros((3, len(irvec), num_wann, num_wann), dtype=complex)
    AA_R[:, R0[0], np.arange(num_wann), np.arange(num_wann)] = tau.T

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
        raise NotImplementedError("scissors correction not yet implemented for BB_R")

    num_k = cwi["num_k"]
    kpoints = np.array(cwi["kpoints"])
    irvec = np.array(cwi["irvec"])

    kb2k = cwi.nnkp.kb2k()
    bveck = cwi.nnkp.bveck()
    wb = cwi["wb"]

    Ek = np.array(cwi["Ek"])
    Uk = np.array(cwi["Uk"])
    _require(cwi, "Mkb", "mmn", "BB_R")
    Mkb = np.array(cwi["Mkb"])

    H_o = np.array([np.diag(Ek[k]) for k in range(num_k)])

    HM_o = np.einsum("kml, kbln->kbmn", H_o, Mkb, optimize=True)
    H_k_kb = np.einsum("klm, kblp, kbpn->kbmn", np.conj(Uk), HM_o, Uk[kb2k[:, :], :, :], optimize=True)
    BB_k = 1.0j * np.einsum("b,kbc,kbmn->ckmn", wb, bveck, H_k_kb, optimize=True)

    BB_R = np.array([fourier_transform_k_to_r(BB_k[i], kpoints, irvec) for i in range(3)])

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
        raise NotImplementedError("scissors correction not yet implemented for CC_R")

    kpoints = np.array(cwi["kpoints"])
    irvec = np.array(cwi["irvec"])

    kb2k = cwi.nnkp.kb2k()
    bveck = cwi.nnkp.bveck()
    wb = cwi["wb"]

    Uk = np.array(cwi["Uk"])
    _require(cwi, "Hkb1b2", "uHu", "CC_R")
    Hkb1b2 = np.array(cwi["Hkb1b2"])

    Hkb1b2 = np.einsum(
        "kblm, kbdlp, kdpn->kbdmn", np.conj(Uk[kb2k[:, :], :, :]), Hkb1b2, Uk[kb2k[:, :], :, :], optimize=True
    )
    CC_k = np.einsum("b,kbi,d,kdj,kbdmn->ijkmn", wb, bveck, wb, bveck, Hkb1b2, optimize=True)

    # CC_ab(k) is not hermitian for a != b; only CC_ba(k) = CC_ab(k)^† holds (as in wannier90 get_oper.F90).
    for i in range(3):
        CC_k[i, i] = 0.5 * (CC_k[i, i] + np.einsum("kmn->knm", CC_k[i, i]).conj())
        for j in range(i + 1, 3):
            CC_k[j, i] = np.einsum("kmn->knm", CC_k[i, j]).conj()

    CC_R = np.array(
        [[fourier_transform_k_to_r(CC_k[i, j], kpoints, irvec) for j in range(3)] for i in range(3)]
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
    _require(cwi, "pauli_spn", "spn", "spin operator (SS_R)")
    pauli_spn = np.array(cwi["pauli_spn"])
    Uk = np.array(cwi["Uk"])

    SS_k = np.einsum("klm,aklp,kpn->akmn", np.conj(Uk), pauli_spn, Uk, optimize=True)

    SS_k = 0.5 * (SS_k + np.einsum("akmn->aknm", SS_k).conj())

    kpoints = np.array(cwi["kpoints"])
    irvec = np.array(cwi["irvec"])

    SS_R = np.array([fourier_transform_k_to_r(SS_k[i], kpoints, irvec) for i in range(3)])

    return SS_R


# ==================================================
def shc_bandshift(E, cwi):
    """
    Shift the bands shc_bandshift_firstband, ..., num_bands (1-based, as in wannier90) by shc_bandshift_energyshift.

    Args:
        E (ndarray): eigenvalues, (num_k, num_bands).
        cwi (CWInfo): CWInfo.

    Returns:
        ndarray: shifted copy of E, or E itself if shc_bandshift = false.
    """
    if not cwi["shc_bandshift"]:
        return E

    E = np.array(E)
    E[:, cwi["shc_bandshift_firstband"] - 1 :] += cwi["shc_bandshift_energyshift"]

    return E


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
    num_k = cwi["num_k"]

    Ek = np.array(cwi["Ek"])
    Uk = np.array(cwi["Uk"])

    # spin operator
    spn_o = np.array(cwi["pauli_spn"])
    SS_k = np.einsum("klm,aklp,kpn->akmn", np.conj(Uk), spn_o, Uk, optimize=True)

    # get_HH_R
    Ek = shc_bandshift(Ek, cwi)

    H_o = np.array([np.diag(Ek[k]) for k in range(num_k)])

    # get_AA_R
    _require(cwi, "Mkb", "mmn", "spin Hall conductivity")
    Mkb = np.array(cwi["Mkb"])
    kb2k = cwi.nnkp.kb2k()
    bveck = cwi.nnkp.bveck()
    wb = cwi["wb"]

    #! QZYZ18 Eq.(48)
    SH_o = spn_o @ H_o[np.newaxis, :, :, :]
    SH_k = np.einsum("klm, aklp, kpn->akmn", np.conj(Uk), SH_o, Uk, optimize=True)

    #! QZYZ18 Eq.(50)
    SM_o = np.einsum("akml, kbln->akbmn", spn_o, Mkb, optimize=True)
    SM_k = np.einsum("klm, akblp, kbpn->akbmn", np.conj(Uk), SM_o, Uk[kb2k[:, :], :, :], optimize=True)
    SR_k = np.einsum("b,kbc,akbmn->ackmn", wb, bveck, SM_k, optimize=True) - np.einsum(
        "b,kbc,akmn->ackmn", wb, bveck, SS_k, optimize=True
    )

    #! QZYZ18 Eq.(51)
    SHM_o = np.einsum("akml, kbln->akbmn", SH_o, Mkb, optimize=True)
    SHM_k = np.einsum("klm, akblp, kbpn->akbmn", np.conj(Uk), SHM_o, Uk[kb2k[:, :], :, :], optimize=True)
    SHR_k = np.einsum("b,kbc,akbmn->ackmn", wb, bveck, SHM_k, optimize=True) - np.einsum(
        "b,kbc,akmn->ackmn", wb, bveck, SH_k, optimize=True
    )

    SH_R = np.array([fourier_transform_k_to_r(SH_k[i], kpoints, irvec) for i in range(3)])
    SR_R = np.array(
        [[fourier_transform_k_to_r(SR_k[i][j], kpoints, irvec) for j in range(3)] for i in range(3)]
    )
    SHR_R = np.array(
        [[fourier_transform_k_to_r(SHR_k[i][j], kpoints, irvec) for j in range(3)] for i in range(3)]
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
    _require(cwi, "Mkb", "mmn", "berry phase")
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

    a_R = np.array([fourier_transform_k_to_r(a_k[i], kpoints, irvec) for i in range(3)])

    return a_R


# ==================================================
def get_v_R(cwi, HH_R=None, atoms_frac=None):
    """
    matrix elements of real-space velocity operator in TB approximation, <0n|v|Rm>.
    v_k^a = 1 / (h/2π) ∇_k^a H_k [Angstrom / s]

    The derivative is taken in the gauge given by atoms_frac: without atoms_frac (wannier90 convention) v_R = i R H(R),
    with atoms_frac (tb_gauge = true) v_R = i (R + τ_n - τ_m) H(R), to be transformed to k space with the same atoms_frac.

    Args:
        cwi (CWInfo): CWInfo.
        HH_R (ndarray, optional): Hamiltonian, HH_R(len(irvec), num_wann, num_wann).
        atoms_frac (ndarray, optional): orbital positions in fractional coordinates, (num_wann, 3).

    Returns:
        ndarray: velocity operator, v_R(3, len(irvec), num_wann, num_wann).
    """
    if HH_R is None:
        HH_R = get_HH_R(cwi)

    A = np.array(cwi["unit_cell_cart"])
    irvec = cwi["irvec"]

    irvec_cart = np.array([np.array(R) @ np.array(A) for R in irvec])

    if atoms_frac is not None:
        atoms_cart = np.array(atoms_frac) @ A
        bond_cart = irvec_cart[:, None, None, :] + atoms_cart[None, None, :, :] - atoms_cart[None, :, None, :]
        v_R_x, v_R_y, v_R_z = 1.0j * np.einsum("Rmna,Rmn->aRmn", bond_cart, HH_R, optimize=True)
    else:
        v_R_x, v_R_y, v_R_z = 1.0j * np.einsum("Ra,Rmn->aRmn", irvec_cart, HH_R, optimize=True)

    v_R = np.array([v_R_x, v_R_y, v_R_z], dtype=complex)

    """
    --------------------------------------------------------------------
    Convert to Angstrom eV / (J s)

    fac = 1.0 / hbar [1 / (J s)]

    'hbar' in SI units
    --------------------------------------------------------------------
    """

    fac = 1.0 / hbar_SI

    v_R = fac * v_R

    return v_R


# ==================================================
def to_tb_gauge(operators, irvec, unit_cell_cart, atoms_frac, ndegen=None):
    """
    convert real-space operators from the wannier90 convention (no orbital position in the Fourier phase)
    to the tb gauge (orbital positions τ in the phase), so that the Fourier transform with atoms_frac gives
    the k-space operators of the Bloch basis e^{ik(R+τ_b)} |φ_b(R)>.

        AA_a(R)      -> AA_a(R) - τ_{n,a} δ_{mn} δ_{R0}
        BB_a(R)      -> BB_a(R) - τ_{n,a} HH(R)
        CC_ab(R)     -> CC_ab(R) - τ_{m,a} BB_b(R) - τ_{n,b} BB_a^†(R) + τ_{m,a} τ_{n,b} HH(R), BB_a^†(R)_{mn} = BB_a(-R)_{nm}^*
        SR_ab(R)     -> SR_ab(R) - τ_{n,b} SS_a(R)
        SHR_ab(R)    -> SHR_ab(R) - τ_{n,b} SH_a(R)

    (m, n: row and column orbitals, τ in Cartesian coordinates).
    HH_R, SS_R and SH_R are the same in both conventions; v_R is not converted (see get_v_R).
    HH_R must be the Hamiltonian from which BB_R and CC_R are built.
    BB_a^†(R) is the Fourier coefficient of BB_a(k)^† only if ndegen(-R) = ndegen(R), as for a Wigner-Seitz supercell.

    Args:
        operators (dict): real-space operators in the wannier90 convention (None for those not calculated).
        irvec (ndarray): lattice points, [[n1,n2,n3]].
        unit_cell_cart (ndarray): lattice vectors (rows), [a1,a2,a3].
        atoms_frac (ndarray): orbital positions in fractional coordinates, (num_wann, 3).
        ndegen (ndarray, optional): degeneracy of each R, checked to be symmetric under R -> -R.

    Returns:
        dict: converted operators (only the keys given in operators, the input is not modified).
    """
    irvec = np.array(irvec, dtype=int)
    tau = np.array(atoms_frac, dtype=float) @ np.array(unit_cell_cart, dtype=float)

    def require(name, *keys):
        for key in keys:
            if operators.get(key) is None:
                raise ValueError(f"{key} is required to convert {name} to the tb gauge.")

    index = {tuple(R): ir for ir, R in enumerate(irvec)}
    if (0, 0, 0) not in index:
        raise ValueError("R = 0 is not in irvec.")

    d = dict(operators)

    if operators.get("AA_R") is not None:
        AA_R = np.array(operators["AA_R"], dtype=complex)
        num_wann = AA_R.shape[-1]
        AA_R[:, index[(0, 0, 0)], np.arange(num_wann), np.arange(num_wann)] -= tau.T
        d["AA_R"] = AA_R

    if operators.get("BB_R") is not None:
        require("BB_R", "HH_R")
        HH_R = np.array(operators["HH_R"], dtype=complex)
        BB_R = np.array(operators["BB_R"], dtype=complex)
        d["BB_R"] = BB_R - np.einsum("na,Rmn->aRmn", tau, HH_R)

    if operators.get("CC_R") is not None:
        require("CC_R", "HH_R", "BB_R")
        HH_R = np.array(operators["HH_R"], dtype=complex)
        BB_R = np.array(operators["BB_R"], dtype=complex)
        CC_R = np.array(operators["CC_R"], dtype=complex)

        try:
            minus_R = [index[tuple(-R)] for R in irvec]
        except KeyError:
            raise ValueError("irvec must contain -R for every R to convert CC_R to the tb gauge.")
        if ndegen is not None and not np.array_equal(np.asarray(ndegen)[minus_R], np.asarray(ndegen)):
            raise ValueError("ndegen must satisfy ndegen(-R) = ndegen(R) to convert CC_R to the tb gauge.")
        BB_R_dag = BB_R[:, minus_R].transpose(0, 1, 3, 2).conj()

        d["CC_R"] = (
            CC_R
            - np.einsum("ma,bRmn->abRmn", tau, BB_R)
            - np.einsum("nb,aRmn->abRmn", tau, BB_R_dag)
            + np.einsum("ma,nb,Rmn->abRmn", tau, tau, HH_R)
        )

    if operators.get("SR_R") is not None:
        require("SR_R", "SS_R")
        SS_R = np.array(operators["SS_R"], dtype=complex)
        d["SR_R"] = np.array(operators["SR_R"], dtype=complex) - np.einsum("nb,aRmn->abRmn", tau, SS_R)

    if operators.get("SHR_R") is not None:
        require("SHR_R", "SH_R")
        SH_R = np.array(operators["SH_R"], dtype=complex)
        d["SHR_R"] = np.array(operators["SHR_R"], dtype=complex) - np.einsum("nb,aRmn->abRmn", tau, SH_R)

    return d


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
