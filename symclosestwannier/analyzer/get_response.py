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
import multiprocessing
from joblib import Parallel, delayed

from symclosestwannier.util._utility import (
    fourier_transform_r_to_k,
    fourier_transform_r_to_k_new,
    fourier_transform_r_to_k_vec,
)

# ==================================================
# number of cpu cores
_cpu_num = multiprocessing.cpu_count()


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
def berry_main(cwi, operators):
    """
    Computes the following quantities:
     (ahc)  Anomalous Hall conductivity (from Berry curvature)
     (kubo) Complex optical conductivity (Kubo-Greenwood) & JDOS
     (morb) Orbital magnetization
     (sc)   Nonlinear shift current
     (shc)  Spin Hall conductivity

    Args:
        cwi (SystemInfo): CWInfo.
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

    if cwi["tb_gauge"]:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    N1, N2, N3 = cwi["berry_kmesh"]
    kpoints = np.array(
        [[i / float(N1), j / float(N2), k / float(N3)] for i in range(N1) for j in range(N2) for k in range(N3)]
    )

    HH, delHH = fourier_transform_r_to_k_new(
        operators["HH_R"], kpoints, cwi["unit_cell_cart"], cwi["irvec"], cwi["ndegen"], atoms_frac
    )

    E, U = np.linalg.eigh(HH)

    D_h = wham_get_D_h(delHH, E, U)

    AA = fourier_transform_r_to_k_vec(operators["AA_R"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac)

    Avec = np.array([U.transpose(0, 2, 1).conjugate() @ AA[i] @ U for i in range(3)])

    A = Avec + 1.0j * D_h  # Eq.(25) WYSV06

    # (ahc)  Anomalous Hall conductivity (from Berry curvature)
    if cwi["berry_task"] == "ahc":
        pass

    # (morb) Orbital magnetization
    if cwi["berry_task"] == "morb":
        pass

    # (kubo) Complex optical conductivity (Kubo-Greenwood) & JDOS
    if cwi["berry_task"] == "kubo":
        kubo_H, kubo_AH, kubo_H_spn, kubo_AH_spn = berry_get_kubo(cwi, E, A)

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
        SS = fourier_transform_r_to_k_vec(operators["SS_R"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac)
        Sx = U.transpose(0, 2, 1).conjugate() @ SS[0] @ U
        Sy = U.transpose(0, 2, 1).conjugate() @ SS[1] @ U
        Sz = U.transpose(0, 2, 1).conjugate() @ SS[2] @ U
        S = np.array([Sx, Sy, Sz])

        me_H_spn, me_H_orb, me_AH_spn, me_AH_orb = berry_get_me(cwi, E, A, S)

    return d


# ==================================================
def berry_get_kubo(cwi, E, A):
    """
    Contribution from point k to the complex interband optical
    conductivity, separated into Hermitian (H) and anti-Hermitian (AH)
    parts.

    Args:

    Returns:
    """
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
    eta_smr = kubo_smr_fixed_en_width

    kubo_freq_list = np.arange(cwi["kubo_freq_min"], cwi["kubo_freq_max"], cwi["kubo_freq_step"])
    if not kubo_adpt_smr and kubo_smr_fixed_en_width != 0.0:
        kubo_freq_list = kubo_freq_list + 1.0j * kubo_smr_fixed_en_width

    kubo_nfreq = len(kubo_freq_list)
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

    def proc(k, m, n):
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

        if kubo_adpt_smr:  # Eq.(35) YWVS07
            # vdum[:] = del_ek[m, :] - del_ek[n, :]
            # joint_level_spacing = np.sqrt(np.dot(vdum, vdum))*Delta_k
            # eta_smr = min(joint_level_spacing*kubo_adpt_smr_fac, kubo_adpt_smr_max)
            eta_smr = kubo_smr_fixed_en_width
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

        kubo_H_ = rfac2[:, np.newaxis, np.newaxis] * aiknm_ajkmn[np.newaxis, :, :]
        kubo_AH_ = cfac[:, np.newaxis, np.newaxis] * aiknm_ajkmn[np.newaxis, :, :]

        if spin_decomp:
            kubo_H_spn_ = rfac2[:, np.newaxis, np.newaxis] * aiknm_ajkmn[np.newaxis, :, :]
            kubo_AH_spn_ = cfac[:, np.newaxis, np.newaxis] * aiknm_ajkmn[np.newaxis, :, :]
        else:
            kubo_H_spn_ = 0
            kubo_AH_spn_ = 0

        return kubo_H_, kubo_AH_, kubo_H_spn_, kubo_AH_spn_

    n_jobs = _cpu_num - 2  # if self._mpm.parallel else 1
    print(len(k_m_n_list))
    res_lst = Parallel(n_jobs=n_jobs, verbose=10)(delayed(proc)(k, m, n) for k, m, n in k_m_n_list)

    for kubo_H_, kubo_AH_, kubo_H_spn_, kubo_AH_spn_ in res_lst:
        kubo_H += kubo_H_
        kubo_AH += kubo_AH_
        kubo_H_spn += kubo_H_spn_
        kubo_AH_spn += kubo_AH_spn_

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

    occ = np.array([[1.0 if E[k, m] < ef else 0.0 for m in range(num_wann)] for k in range(num_k)])

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
