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
#             get_response: external response properties             #
#                                                                    #
# ****************************************************************** #

# This module computes various "Berry phase" related properties
#
# Key REFERENCES
#
# *  WYSV06 = PRB 74, 195118 (2006)  (anomalous Hall conductivity - AHC)
# *  YWVS07 = PRB 75, 195121 (2007)  (Kubo frequency-dependent conductivity)
# *  LVTS12 = PRB 85, 014435 (2012)  (orbital magnetization and AHC)
# *  CTVR06 = PRB 74, 024408 (2006)  (  "          "       )
# *  IATS18 = PRB 97, 245143 (2018)  (nonlinear shift current)
# *  QZYZ18 = PRB 98, 214402 (2018)  (spin Hall conductivity - SHC)
# *  RPS19  = PRB 99, 235113 (2019)  (spin Hall conductivity - SHC)
# *  IAdJS19 = arXiv:1910.06172 (2019) (quasi-degenerate k.p)
# ---------------------------------------------------------------

import numpy as np

from symclosestwannier.util._utility import (
    fermi,
    fourier_transform_r_to_k,
    fourier_transform_r_to_k_new,
    fourier_transform_r_to_k_vec,
)


# ==================================================
def berry_main(cwi, operators):
    """
    Computes the following quantities:
     (ahc)  Anomalous Hall conductivity (from Berry curvature)
     (kubo) Complex optical conductivity (Kubo-Greenwood) & JDOS
     (morb) Orbital magnetization
     (sc)   Nonlinear shift current
     (shc)  Spin Hall conductivity

    Args:
        cwi (CWInfo): CWInfo.
        operators (dict): operators.
    """
    print("Properties calculated in module  b e r r y \n ------------------------------------------")

    # initialization
    d = {
        # ahc
        # morb
        # kubo
        "kubo_H": 0.0,
        "kubo_AH": 0.0,
        "kubo_H_spn": 0.0,
        "kubo_AH_spn": 0.0,
        # sc
        # shc
        # kdotp
    }

    # (ahc)  Anomalous Hall conductivity (from Berry curvature)
    if cwi["berry_task"] == "ahc":
        pass

    # (morb) Orbital magnetization
    if cwi["berry_task"] == "morb":
        pass

    # (kubo) Complex optical conductivity (Kubo-Greenwood) & JDOS
    if cwi["berry_task"] == "kubo":
        kubo_H, kubo_AH, kubo_H_spn, kubo_AH_spn = berry_get_kubo(cwi, operators["HH_R"], operators["AA_R"])

        d["kubo_H"] = kubo_H
        d["kubo_AH"] = kubo_AH
        d["kubo_H_spn"] = kubo_H_spn
        d["kubo_AH_spn"] = kubo_AH_spn

    # (sc)   Nonlinear shift current
    if cwi["berry_task"] == "sc":
        kubo_nfreq = round((cwi["kubo_freq_max"] - cwi["kubo_freq_min"]) / cwi["kubo_freq_step"]) + 1
        sc_k_list = np.zeros((3, 6, kubo_nfreq))
        sc_list = np.zeros((3, 6, kubo_nfreq))
        # allocate (shc_fermi(fermi_n))
        # allocate (shc_k_fermi(fermi_n))
        # allocate (shc_k_fermi_dummy(fermi_n))
        # shc_fermi = 0.0_dp
        # shc_k_fermi = 0.0_dp
        # !only used for fermiscan & adpt kmesh
        # shc_k_fermi_dummy = 0.0_dp
        # adpt_counter_list = 0

    # (shc)  Spin Hall conductivity
    if cwi["berry_task"] == "shc":
        pass

    if cwi["berry_task"] == "kdotp":
        pass

    # (me) magnetoelectric tensor
    if cwi["berry_task"] == "me":
        pass
        # SS = fourier_transform_r_to_k_vec(operators["SS_R"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac)
        # Sx = U.transpose(0, 2, 1).conjugate() @ SS[0] @ U
        # Sy = U.transpose(0, 2, 1).conjugate() @ SS[1] @ U
        # Sz = U.transpose(0, 2, 1).conjugate() @ SS[2] @ U
        # S = np.array([Sx, Sy, Sz])

        # me_H_spn, me_H_orb, me_AH_spn, me_AH_orb = berry_get_me(cwi, E, A, S)

    return d


# ==================================================
def wham_get_D_h(delHH, E, U):
    """
    Compute D^H_a=UU^dag.del_a UU (a=x,y,z)
    using Eq.(24) of WYSV06
    """
    num_k = E.shape[0]
    num_wann = E.shape[1]

    delHH_Band = U.transpose(0, 2, 1).conjugate()[np.newaxis, :, :, :] @ delHH @ U[np.newaxis, :, :, :]
    fac = np.array(
        [
            [
                [0.0 if n == m or abs(E[k, m] - E[k, n]) < 1e-7 else 1.0 / (E[k, n] - E[k, m]) for n in range(num_wann)]
                for m in range(num_wann)
            ]
            for k in range(num_k)
        ]
    )
    D_h = delHH_Band * fac[np.newaxis, :, :, :]

    return D_h


# ==================================================
def wham_get_deleig_a(delHH_a, E, U, use_degen_pert=False, degen_thr=0.0):
    """
    Compute band derivatives dE/dk_a.
    """
    if use_degen_pert:
        delHH_bar_a = U.transpose(0, 2, 1).conjugate() @ delHH_a @ U
        num_k = U.shape[0]
        num_wann = U.shape[1]

        deleig_a = np.zeros((num_k, num_wann))

        for k in range(num_k):
            i = 0
            while i <= num_wann:
                i = i + 1
                if i + 1 <= num_wann:
                    diff = E[k, i + 1] - E[k, i]
                else:
                    #
                    # i-th is the highest band, and it is non-degenerate
                    #
                    diff = degen_thr + 1.0

                if diff < degen_thr:
                    #
                    # Bands i and i+1 are degenerate
                    #
                    degen_min = i
                    degen_max = degen_min + 1
                    #
                    # See if any higher bands are in the same degenerate group
                    #
                    while degen_max + 1 <= num_wann and diff < degen_thr:
                        diff = E[k, degen_max + 1] - E[k, degen_max]
                        if diff < degen_thr:
                            degen_max = degen_max + 1

                    #
                    # Bands from degen_min to degen_max are degenerate. Diagonalize
                    # the submatrix in Eq.(31) YWVS07 over this degenerate subspace.
                    # The eigenvalues are the band gradients
                    #
                    dim = degen_max - degen_min + 1
                    deleig_a[k, i : i + dim] = np.linalg.eigh(
                        delHH_bar_a[degen_min : degen_max + 1, degen_min : degen_max + 1]
                    )[0]

                    #
                    # Scanned bands up to degen_max
                    #
                    i = degen_max
                else:
                    #
                    # Use non-degenerate form [Eq.(27) YWVS07] for current (i-th) band
                    #
                    deleig_a[k, i] = np.real(delHH_bar_a[k, i, i])
    else:
        deleig_a = np.real(np.diagonal(U.transpose(0, 2, 1).conjugate() @ delHH_a @ U, axis1=1, axis2=2))

    return deleig_a


# ==================================================
def wham_get_deleig(delHH, E, U, use_degen_pert=False, degen_thr=0.0):
    """
    This function returns derivatives of the eigenvalues dE/dk_a, using wham_get_deleig_a.
    """
    delE = np.array([wham_get_deleig_a(delHH[a, :, :, :], E, U, use_degen_pert, degen_thr) for a in range(3)])

    return delE


# =======================================================================
def kmesh_spacing_mesh(mesh, B):
    """
    Set up the value of the interpolation mesh spacing, needed for
    adaptive smearing [see Eqs. (34-35) YWVS07]. Choose it as the largest of
    the three Delta_k's for each of the primitive translations b1, b2, and b3
    """
    B = np.array(B)
    Delta_k_i = np.array([np.sqrt(np.dot(B[i, :], B[i, :])) / mesh[i] for i in range(3)])

    kmesh_spacing_mesh = np.max(Delta_k_i)

    return kmesh_spacing_mesh


# ==================================================
def utility_w0gauss(x, n):
    """
    the derivative of utility_wgauss:  an approximation to the delta function

    (n>=0) : derivative of the corresponding Methfessel-Paxton utility_wgauss

    (n=-1 ): derivative of cold smearing:
                 1/sqrt(pi)*exp(-(x-1/sqrt(2))**2)*(2-sqrt(2)*x)

    (n=-99): derivative of Fermi-Dirac function: 0.5/(1.0+cosh(x))
    """
    utility_w0gauss = 0.0  # in case of error return

    sqrtpm1 = 1.0 / np.sqrt(np.pi)

    # cold smearing  (Marzari-Vanderbilt)
    if n == -1:
        arg = min(200, (x - 1.0 / np.sqrt(2)) ** 2)
        utility_w0gauss = sqrtpm1 * np.exp(-arg) * (2.0 - np.sqrt(2.0) * x)
    # Fermi-Dirac smearing
    elif n == -99:
        if np.abs(x) <= 36.0:
            utility_w0gauss = 1.0 / (2.0 + np.exp(-x) + np.exp(+x))
        else:
            utility_w0gauss = 0.0
    # Gaussian
    elif n == 0:
        arg = min(200, x**2)
        utility_w0gauss = np.exp(-arg) * sqrtpm1
    elif n > 10 or n < 0:
        raise Exception("utility_w0gauss higher order (n>10) smearing is untested and unstable")
    # Methfessel-Paxton
    else:
        arg = min(200, x**2)
        utility_w0gauss = np.exp(-arg) * sqrtpm1
        hd = 0.0
        hp = np.exp(-arg)
        ni = 0
        a = sqrtpm1
        for i in range(1, n + 1):
            hd = 2.0 * x * hp - 2.0 * float(ni) * hd
            ni += 1
            a = -a / (float(i) * 4.0)
            hp = 2.0 * x * hd - 2.0 * float(ni) * hp
            ni += 1
            utility_w0gauss += a * hp

    return utility_w0gauss


# ==================================================
def berry_get_imfgh_klist(kpt, num_fermi, img=False, imh=False, occ=None, ladpt=None):
    """
    Calculates the three quantities needed for the orbital magnetization:
        * -2Im[f(k)] [Eq.33 CTVR06, Eq.6 LVTS12]
        * -2Im[g(k)] [Eq.34 CTVR06, Eq.7 LVTS12]
        * -2Im[h(k)] [Eq.35 CTVR06, Eq.8 LVTS12]

    They are calculated together (to reduce the number of Fourier calls)
    for a list of Fermi energies, and stored in axial-vector form.

    Args:
        num_fermi (int):
        img (bool, optional): calculate -2Im[g(k)] ?
        imh (bool, optional): calculate -2Im[h(k)] ?
        occ (ndarray, optional): occupancy.
        ladpt (ndarray, optional): .
    """
    pass
    # if occ is not None:
    #     num_fermi_loc = 1
    # else:
    #     num_fermi_loc = num_fermi

    # if ladpt sis not None:
    #     todo = ladpt
    # else:
    #     todo = True

    #
    # Gather W-gauge matrix objects
    #

    # # E, U, JJp_list, JJm_list
    # # f_list, g_list
    # if occ is not None:
    #     call wham_get_eig_UU_HH_JJlist(kpt, eig, UU, HH, JJp_list, JJm_list, occ=occ)
    #     call wham_get_occ_mat_list(UU, f_list, g_list, occ=occ)
    # else:
    #     call wham_get_eig_UU_HH_JJlist(kpt, eig, UU, HH, JJp_list, JJm_list)
    #     call wham_get_occ_mat_list(UU, f_list, g_list, eig=eig)
    # endif

    # call pw90common_fourier_R_to_k_vec(kpt, AA_R, OO_true=AA, OO_pseudo=OOmega)

    # if (present(imf_k_list)) then
    #   ! Trace formula for -2Im[f], Eq.(51) LVTS12
    #   !
    #   do ife = 1, num_fermi_loc
    #     if (todo(ife)) then
    #       do i = 1, 3
    #         !
    #         ! J0 term (Omega_bar term of WYSV06)
    #         imf_k_list(1, i, ife) = &
    #           utility_re_tr_prod(f_list(:, :, ife), OOmega(:, :, i))
    #         !
    #         ! J1 term (DA term of WYSV06)
    #         imf_k_list(2, i, ife) = -2.0_dp* &
    #                                 ( &
    #                                 utility_im_tr_prod(AA(:, :, alpha_A(i)), JJp_list(:, :, ife, beta_A(i))) &
    #                                 + utility_im_tr_prod(JJm_list(:, :, ife, alpha_A(i)), AA(:, :, beta_A(i))) &
    #                                 )
    #         !
    #         ! J2 term (DD of WYSV06)
    #         imf_k_list(3, i, ife) = -2.0_dp* &
    #                                 utility_im_tr_prod(JJm_list(:, :, ife, alpha_A(i)), JJp_list(:, :, ife, beta_A(i)))
    #       end do
    #     endif
    #   end do
    # end if

    # if (present(img_k_list)) img_k_list = 0.0_dp
    # if (present(imh_k_list)) imh_k_list = 0.0_dp

    # if (present(img_k_list) .and. present(imh_k_list)) then
    #   allocate (BB(num_wann, num_wann, 3))
    #   allocate (CC(num_wann, num_wann, 3, 3))

    #   allocate (tmp(num_wann, num_wann, 5))
    #   ! tmp(:,:,1:3) ... not dependent on inner loop variables
    #   ! tmp(:,:,1) ..... HH . AA(:,:,alpha_A(i))
    #   ! tmp(:,:,2) ..... LLambda_ij [Eq. (37) LVTS12] expressed as a pseudovector
    #   ! tmp(:,:,3) ..... HH . OOmega(:,:,i)
    #   ! tmp(:,:,4:5) ... working matrices for matrix products of inner loop

    #   call pw90common_fourier_R_to_k_vec(kpt, BB_R, OO_true=BB)
    #   do j = 1, 3
    #     do i = 1, j
    #       call pw90common_fourier_R_to_k(kpt, CC_R(:, :, :, i, j), CC(:, :, i, j), 0)
    #       CC(:, :, j, i) = conjg(transpose(CC(:, :, i, j)))
    #     end do
    #   end do

    #   ! Trace formula for -2Im[g], Eq.(66) LVTS12
    #   ! Trace formula for -2Im[h], Eq.(56) LVTS12
    #   !
    #   do i = 1, 3
    #     call utility_zgemm_new(HH, AA(:, :, alpha_A(i)), tmp(:, :, 1))
    #     call utility_zgemm_new(HH, OOmega(:, :, i), tmp(:, :, 3))
    #     !
    #     ! LLambda_ij [Eq. (37) LVTS12] expressed as a pseudovector
    #     tmp(:, :, 2) = cmplx_i*(CC(:, :, alpha_A(i), beta_A(i)) &
    #                             - conjg(transpose(CC(:, :, alpha_A(i), beta_A(i)))))

    #     do ife = 1, num_fermi_loc
    #       !
    #       ! J0 terms for -2Im[g] and -2Im[h]
    #       !
    #       ! tmp(:,:,5) = HH . AA(:,:,alpha_A(i)) . f_list(:,:,ife) . AA(:,:,beta_A(i))
    #       call utility_zgemm_new(tmp(:, :, 1), f_list(:, :, ife), tmp(:, :, 4))
    #       call utility_zgemm_new(tmp(:, :, 4), AA(:, :, beta_A(i)), tmp(:, :, 5))

    #       s = 2.0_dp*utility_im_tr_prod(f_list(:, :, ife), tmp(:, :, 5));
    #       img_k_list(1, i, ife) = utility_re_tr_prod(f_list(:, :, ife), tmp(:, :, 2)) - s
    #       imh_k_list(1, i, ife) = utility_re_tr_prod(f_list(:, :, ife), tmp(:, :, 3)) + s

    #       !
    #       ! J1 terms for -2Im[g] and -2Im[h]
    #       !
    #       ! tmp(:,:,1) = HH . AA(:,:,alpha_A(i))
    #       ! tmp(:,:,4) = HH . JJm_list(:,:,ife,alpha_A(i))
    #       call utility_zgemm_new(HH, JJm_list(:, :, ife, alpha_A(i)), tmp(:, :, 4))

    #       img_k_list(2, i, ife) = -2.0_dp* &
    #                               ( &
    #                               utility_im_tr_prod(JJm_list(:, :, ife, alpha_A(i)), BB(:, :, beta_A(i))) &
    #                               - utility_im_tr_prod(JJm_list(:, :, ife, beta_A(i)), BB(:, :, alpha_A(i))) &
    #                               )
    #       imh_k_list(2, i, ife) = -2.0_dp* &
    #                               ( &
    #                               utility_im_tr_prod(tmp(:, :, 1), JJp_list(:, :, ife, beta_A(i))) &
    #                               + utility_im_tr_prod(tmp(:, :, 4), AA(:, :, beta_A(i))) &
    #                               )

    #       !
    #       ! J2 terms for -2Im[g] and -2Im[h]
    #       !
    #       ! tmp(:,:,4) = JJm_list(:,:,ife,alpha_A(i)) . HH
    #       ! tmp(:,:,5) = HH . JJm_list(:,:,ife,alpha_A(i))
    #       call utility_zgemm_new(JJm_list(:, :, ife, alpha_A(i)), HH, tmp(:, :, 4))
    #       call utility_zgemm_new(HH, JJm_list(:, :, ife, alpha_A(i)), tmp(:, :, 5))

    #       img_k_list(3, i, ife) = -2.0_dp* &
    #                               utility_im_tr_prod(tmp(:, :, 4), JJp_list(:, :, ife, beta_A(i)))
    #       imh_k_list(3, i, ife) = -2.0_dp* &
    #                               utility_im_tr_prod(tmp(:, :, 5), JJp_list(:, :, ife, beta_A(i)))
    #     end do
    #   end do
    #   deallocate (tmp)
    # end if


# ==================================================
def berry_get_imf_klist(kpt, occ=None, ladpt=None):
    """
    Calculates the Berry curvature traced over the occupied
    states, -2Im[f(k)] [Eq.33 CTVR06, Eq.6 LVTS12] for a list
    of Fermi energies, and stores it in axial-vector form
    """
    # if occ is not None:
    #     imf_k_list = berry_get_imfgh_klist(kpt, occ=occ)
    # else:
    #     if ladpt is not None:
    #         imf_k_list = berry_get_imfgh_klist(kpt, ladpt=ladpt)
    #     else:
    #         imf_k_list = berry_get_imfgh_klist(kpt)

    # return imf_k_list
    pass


# ==================================================
def berry_get_ahc(cwi, HH_R, AA_R):
    """
    Anomalous Hall conductivity, in S/cm.
    The three independent components σx = σyz, σy = σzx, and σz = σxy are computed.
    The real part Re[σ^AH_αβ] describes the anomalous Hall conductivity (AHC), and remains finite in the static limit,
    while the imaginary part Im[σ^H_αβ] describes magnetic circular dichroism, and vanishes as ω → 0.

    Args:
        cwi (CWInfo): CWInfo.
        HH_R (ndarray): matrix elements of real-space Hamiltonian, <0n|H|Rm>.
        AA_R (ndarray): matrix elements of real-space position operator, <0n|r|Rm>.

    Returns:
        tuple: .
    """
    # imf_k_list = np.array([berry_get_imf_klist(k) for k in range(num_k)])

    # ladpt = [False]*num_fermi

    # for i in range(num_fermi):
    #     vdum = np.array([sum(imf_k_list[:, 0, i]) for a in range(3)])

    #     if berry_curv_unit == 'bohr2':
    #         vdum = vdum/bohr**2
    #     rdum = np.sqrt(np.dot(vdum, vdum))
    #     if rdum > berry_curv_adpt_kmesh_thresh:
    #         adpt_counter_list[i] = adpt_counter_list[i] + 1
    #         ladpt[i] = True
    #     else:
    #         imf_list[:,:,i] = imf_list[:,:,i] + imf_k_list[:,:,i]*kweight

    #     if np.any(ladpt):
    #         for loop_adpt in range(berry_curv_adpt_kmesh**3):
    #             # Using imf_k_list here would corrupt values for other
    #             # frequencies, hence dummy. Only i-th element is used
    #             imf_k_list_dummy = berry_get_imf_klist(kpt + adkpt(:, loop_adpt), imf_k_list_dummy, ladpt=ladpt)

    #             if = 1, num_fermi
    #             if (ladpt(if)) then
    #               imf_list(:, :, if) = imf_list(:, :, if) &
    #                                    + imf_k_list_dummy(:, :, if)*kweight_adpt
    #             endif
    #           enddo
    #         end do
    #       endif
    #     end if
    pass


# ==================================================
def berry_get_kubo_k(cwi, HH_R, AA_R):
    """
    calculate
    Complex interband optical conductivity, in S/cm,
    separated into Hermitian (Kubo_H) and anti-Hermitian (Kubo_AH) parts.

    Args:
        cwi (CWInfo): CWInfo.
        HH_R (ndarray): matrix elements of real-space Hamiltonian, <0n|H|Rm>.
        AA_R (ndarray): matrix elements of real-space position operator, <0n|r|Rm>.

    Returns:
        tuple: Kubo_H, Kubo_AH, Kubo_H_spn, Kubo_AH_spn.
    """
    pass


# ==================================================
def berry_get_kubo(cwi, HH_R, AA_R):
    """
    Complex interband optical conductivity, in S/cm,
    separated into Hermitian (Kubo_H) and anti-Hermitian (Kubo_AH) parts.

    Args:
        cwi (CWInfo): CWInfo.
        HH_R (ndarray): matrix elements of real-space Hamiltonian, <0n|H|Rm>.
        AA_R (ndarray): matrix elements of real-space position operator, <0n|r|Rm>.

    Returns:
        tuple: Kubo_H, Kubo_AH, Kubo_H_spn, Kubo_AH_spn.
    """
    N1, N2, N3 = cwi["berry_kmesh"]
    kpoints = np.array(
        [[i / float(N1), j / float(N2), k / float(N3)] for i in range(N1) for j in range(N2) for k in range(N3)]
    )

    if cwi["tb_gauge"]:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    HH, delHH = fourier_transform_r_to_k_new(
        HH_R, kpoints, cwi["unit_cell_cart"], cwi["irvec"], cwi["ndegen"], atoms_frac
    )

    E, U = np.linalg.eigh(HH)

    D_h = wham_get_D_h(delHH, E, U)
    AA = fourier_transform_r_to_k_vec(AA_R, kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac)
    Avec = np.array([U.transpose(0, 2, 1).conjugate() @ AA[i] @ U for i in range(3)])
    A = Avec + 1.0j * D_h  # Eq.(25) WYSV06

    ef = cwi["fermi_energy"]
    berry_kmesh = cwi["berry_kmesh"]
    num_k = np.prod(berry_kmesh)
    num_wann = cwi["num_wann"]

    spin_decomp = cwi["spin_decomp"]
    spn_nk = np.zeros(num_wann)

    kubo_adpt_smr = cwi["kubo_adpt_smr"]
    kubo_adpt_smr_fac = cwi["kubo_adpt_smr_fac"]
    kubo_adpt_smr_max = cwi["kubo_adpt_smr_max"]
    kubo_smr_fixed_en_width = cwi["kubo_smr_fixed_en_width"]

    use_degen_pert = cwi["use_degen_pert"]
    degen_thr = cwi["degen_thr"]

    if cwi["kubo_smr_type"] == "gauss":
        kubo_smr_type_idx = 0
    elif "m-p" in cwi["kubo_smr_type"]:
        m_pN = cwi["kubo_smr_type"]
        kubo_smr_type_idx = m_pN[2:]
    elif cwi["kubo_smr_type"] == "m-v" or cwi["kubo_smr_type"] == "cold":
        kubo_smr_type_idx = -1
    elif cwi["kubo_smr_type"] == "f-d":
        kubo_smr_type_idx = -99

    if cwi["kubo_eigval_max"] < +100000:
        kubo_eigval_max = cwi["kubo_eigval_max"]
    elif cwi["dis_froz_max"] < +100000:
        kubo_eigval_max = cwi["dis_froz_max"] + 0.6667
    else:
        kubo_eigval_max = np.max(E) + 0.6667

    if kubo_adpt_smr:
        delE = wham_get_deleig(delHH, E, U, use_degen_pert, degen_thr)
        Delta_k = kmesh_spacing_mesh(berry_kmesh, cwi["B"])

    kubo_freq_list = np.arange(cwi["kubo_freq_min"], cwi["kubo_freq_max"], cwi["kubo_freq_step"])
    # Replace imaginary part of frequency with a fixed value
    if not kubo_adpt_smr and kubo_smr_fixed_en_width != 0.0:
        kubo_freq_list = np.real(kubo_freq_list) + 1.0j * kubo_smr_fixed_en_width

    kubo_nfreq = len(kubo_freq_list)

    kubo_H = 1.0j * np.zeros((kubo_nfreq, 3, 3))
    kubo_AH = 1.0j * np.zeros((kubo_nfreq, 3, 3))

    if spin_decomp:
        kubo_H_spn = 1.0j * np.zeros((kubo_nfreq, 3, 3, 3))
        kubo_AH_spn = 1.0j * np.zeros((kubo_nfreq, 3, 3, 3))
    else:
        kubo_H_spn = 0.0
        kubo_AH_spn = 0.0

    k_m_n_list = [
        (k, m, n)
        for k in range(num_k)
        for m in range(num_wann)
        for n in range(num_wann)
        if m != n and E[k, m] < kubo_eigval_max and E[k, n] < kubo_eigval_max
    ]

    num = len(k_m_n_list)
    for cnt, (k, m, n) in enumerate(k_m_n_list):
        if (cnt + 1) % 1000 == 0:
            print(f"{cnt+1}/{num}")
        ekm = E[k, m]
        ekn = E[k, n]
        fkm = 1.0 if E[k, m] < ef else 0.0
        fkn = 1.0 if E[k, n] < ef else 0.0

        if spin_decomp:
            if spn_nk[n] >= 0 and spn_nk[m] >= 0:
                ispn = 0  # up --> up transition
            elif spn_nk[n] < 0 and spn_nk[m] < 0:
                ispn = 1  # down --> down
            else:
                ispn = 2  # spin-flip

        if kubo_adpt_smr:
            # Eq.(35) YWVS07
            vdum = delE[:, k, m] - delE[:, k, n]
            joint_level_spacing = np.sqrt(np.dot(vdum, vdum)) * Delta_k
            eta_smr = min(joint_level_spacing * kubo_adpt_smr_fac, kubo_adpt_smr_max)
        else:
            eta_smr = kubo_smr_fixed_en_width

        # Complex frequency for the anti-Hermitian conductivity
        if kubo_adpt_smr:
            omega_list = np.real(kubo_freq_list) + 1.0j * eta_smr
        else:
            omega_list = kubo_freq_list

        # Broadened delta function for the Hermitian conductivity and JDOS
        arg = (ekm - ekn - np.real(omega_list)) / eta_smr
        delta = np.array([utility_w0gauss(arg[ifreq], kubo_smr_type_idx) for ifreq in range(kubo_nfreq)]) / eta_smr

        rfac1 = (fkm - fkn) * (ekm - ekn)
        rfac2 = -np.pi * rfac1 * delta
        cfac = 1.0j * rfac1 / (ekm - ekn - omega_list)

        aiknm_ajkmn = np.array([[A[i, k, n, m] * A[j, k, m, n] for j in range(3)] for i in range(3)])

        kubo_H += rfac2[:, np.newaxis, np.newaxis] * aiknm_ajkmn[np.newaxis, :, :]
        kubo_AH += cfac[:, np.newaxis, np.newaxis] * aiknm_ajkmn[np.newaxis, :, :]

        if spin_decomp:
            kubo_H_spn += rfac2[:, np.newaxis, np.newaxis] * aiknm_ajkmn[np.newaxis, :, :]
            kubo_AH_spn += cfac[:, np.newaxis, np.newaxis] * aiknm_ajkmn[np.newaxis, :, :]

    # Convert to S/cm
    # ==================================================
    # fac = e^2/(hbar.V_c*10^-8)
    # ==================================================
    #
    # with 'V_c' in Angstroms^3, and 'e', 'hbar' in SI units
    # --------------------------------------------------------------------

    cell_volume = cwi["unit_cell_volume"]
    hbar_SI = 1.054571726e-34  # wannier90
    elem_charge_SI = 1.602176565e-19  # wannier90

    fac = 1.0e8 * elem_charge_SI**2 / (hbar_SI * cell_volume) / num_k

    kubo_H *= fac
    kubo_AH *= fac

    if spin_decomp:
        kubo_H_spn *= fac
        kubo_AH_spn *= fac

    return kubo_H, kubo_AH, kubo_H_spn, kubo_AH_spn


# ==================================================
def berry_get_me(cwi, E, A):
    """
    Contribution from point k to the complex interband optical
    conductivity, separated into Hermitian (H) and anti-Hermitian (AH)
    parts.

    Args:

    Returns:
    """
    ef = cwi["fermi_energy"]
    berry_kmesh = cwi["berry_kmesh"]
    me_eigval_max = cwi["me_eigval_max"]
    num_k = len(berry_kmesh)
    num_wann = cwi["num_wann"]

    spin_decomp = cwi["spin_decomp"]
    spn_nk = np.zeros(num_wann)

    me_adpt_smr = cwi["me_adpt_smr"]
    me_adpt_smr_fac = cwi["me_adpt_smr_fac"]
    me_adpt_smr_max = cwi["me_adpt_smr_max"]

    me_smr_fixed_en_width = cwi["me_smr_fixed_en_width"]
    eta_smr = me_smr_fixed_en_width

    me_freq_list = np.arange(cwi["me_freq_min"], cwi["me_freq_max"], cwi["me_freq_step"])
    me_nfreq = len(me_freq_list)
    if cwi["me_smr_type"] == "gauss":
        me_smr_type_idx = 0
    elif "m-p" in cwi["me_smr_type"]:
        m_pN = cwi["me_smr_type"]
        me_smr_type_idx = m_pN[2:]
    elif cwi["me_smr_type"] == "m-v" or cwi["me_smr_type"] == "cold":
        me_smr_type_idx = -1
    elif cwi["me_smr_type"] == "f-d":
        me_smr_type_idx = -99

    occ = np.array([[fermi(E[k, m] - ef, T=0.0) for m in range(num_wann)] for k in range(num_k)])

    me_H_spn = 1.0j * np.zeros((me_nfreq, 3, 3))
    me_H_orb = 1.0j * np.zeros((me_nfreq, 3, 3))
    me_AH_spn = 1.0j * np.zeros((me_nfreq, 3, 3))
    me_AH_orb = 1.0j * np.zeros((me_nfreq, 3, 3))

    for k in range(num_k):
        ek = E[k]
        fk = occ[k]
        ak = A[k]
        sk = S[k]

        for m in range(num_wann):
            for n in range(num_wann):
                if n == m:
                    continue
                if ek[m] > me_eigval_max or ek[n] > me_eigval_max:
                    continue
                if spin_decomp:
                    if spn_nk[n] >= 0 and spn_nk[m] >= 0:
                        ispn = 0  # up --> up transition
                    elif spn_nk[n] < 0 and spn_nk[m] < 0:
                        ispn = 1  # down --> down
                    else:
                        ispn = 2  # spin-flip
                if me_adpt_smr:  # Eq.(35) YWVS07
                    # vdum[:] = del_ek[m, :] - del_ek[n, :]
                    # joint_level_spacing = np.sqrt(np.dot(vdum, vdum))*Delta_k
                    # eta_smr = min(joint_level_spacing*me_adpt_smr_fac, me_adpt_smr_max)
                    pass

                rfac1 = (fk[m] - fk[n]) * (ek[m] - ek[n])

                for ifreq in range(me_nfreq):
                    #
                    # Complex frequency for the anti-Hermitian conductivity
                    #
                    if me_adpt_smr:
                        omega = me_freq_list[ifreq] + 1j * eta_smr
                    else:
                        omega = me_freq_list[ifreq] + 1j * me_smr_fixed_en_width
                    #
                    # Broadened delta function for the Hermitian conductivity and JDOS
                    #
                    arg = (ek[m] - ek[n] - np.real(omega)) / eta_smr
                    # If only Hermitean part were computed, could speed up
                    # by inserting here 'if(abs(arg)>10.0_dp) cycle'
                    delta = utility_w0gauss(arg, me_smr_type_idx) / eta_smr

                    #
                    # Lorentzian shape (for testing purposes)
                    # delta=1.0_dp/(1.0_dp+arg*arg)/pi
                    # delta=delta/eta_smr
                    #
                    cfac = 1j * rfac1 / (ek[m] - ek[n] - omega)
                    rfac2 = -np.pi * rfac1 * delta
                    for j in range(3):
                        for i in range(j, 3):
                            me_H[ifreq, i, j] += rfac2 * ak[i, n, m] * ak[j, m, n]
                            me_AH[ifreq, i, j] += cfac * ak[i, n, m] * ak[j, m, n]
                            if spin_decomp:
                                me_H_spn[ifreq, i, j, ispn] += rfac2 * ak[i, n, m] * ak[j, m, n]
                                me_AH_spn[ifreq, i, j, ispn] += cfac * ak[i, n, m] * ak[j, m, n]

    me_H_spn /= num_k
    me_H_orb /= num_k
    me_AH_spn /= num_k
    me_AH_orb /= num_k

    return me_H_spn, me_H_orb, me_AH_spn, me_AH_orb


# *************************************************************************** #
#     dc Anomalous Hall conductivity and eventually (if 'mcd' string also     #
#       present in addition to 'ahe', e.g., 'ahe+mcd') dichroic optical       #
#     conductivity, both calculated on the same (adaptively-refined) mesh     #
# *************************************************************************** #
# ==================================================
def absorptive_dichroic_optical_cond_main():
    """
    Absorptive dichroic optical conductivity & JDOS on uniform mesh
    """
    pass


# ==================================================
def absorptive_dichroic_optical_cond_main():
    """
    Absorptive dichroic optical conductivity & JDOS on uniform mesh
    """
    pass


# ==================================================
def abs_ordinary_optical_cond_main():
    """
    Absorptive ordinary optical conductivity & JDOS on a uniform mesh
    """
    pass


# ==================================================
def orbital_magnetization_main():
    """
    Orbital magnetization
    """
    pass


# ==================================================
def boltzwann_main():
    """
    Boltzmann transport coefficients (BoltzWann module)
    """
    pass


# ==================================================
def gyrotropic_main():
    """
    Gyrotropic transport coefficients (Gyrotropic module)
    """
    pass
