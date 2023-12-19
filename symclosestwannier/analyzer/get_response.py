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

import numpy as np


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

    # (ahc)  Anomalous Hall conductivity (from Berry curvature)
    if cwi["berry_task"] == "ahc":
        pass

    # (morb) Orbital magnetization
    if cwi["berry_task"] == "morb":
        pass

    # (kubo) Complex optical conductivity (Kubo-Greenwood) & JDOS
    if cwi["berry_task"] == "kubo":
        _ = berry_get_kubo_k(
            cwi, HH_R, AA_R, )

        if cwi["spin_decomp"]:
            kubo_H_k_spn = np.zeros((3, 3, kubo_nfreq))
            kubo_H_spn = np.zeros((3, 3, kubo_nfreq))
            kubo_AH_k_spn = np.zeros((3, 3, kubo_nfreq))
            kubo_AH_spn = np.zeros((3, 3, kubo_nfreq))

            _ = berry_get_kubo_k(
                pw90_berry,
                dis_manifold,
                fermi_energy_list,
                kpt_latt,
                pw90_band_deriv_degen,
                pw90_spin,
                ws_region,
                print_output,
                wannier_data,
                ws_distance,
                wigner_seitz,
                AA_R,
                HH_R,
                kubo_AH_k,
                kubo_H_k,
                SS_R,
                u_matrix,
                v_matrix,
                eigval,
                kpt,
                real_lattice,
                jdos_k,
                scissors_shift,
                mp_grid,
                num_bands,
                num_kpts,
                num_wann,
                num_valence_bands,
                effective_model,
                have_disentangled,
                spin_decomp,
                seedname,
                stdout,
                timer,
                error,
                comm,
                kubo_AH_k_spn,
                kubo_H_k_spn,
                jdos_k_spn,
            )

        kubo_H = kubo_H + kubo_H_k * kweight
        kubo_AH = kubo_AH + kubo_AH_k * kweight

        if cwi["spin_decomp"]:
            kubo_H_spn = kubo_H_spn + kubo_H_k_spn * kweight
            kubo_AH_spn = kubo_AH_spn + kubo_AH_k_spn * kweight

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

    return 0


# ==================================================
def berry_get_kubo_k():
    """
    Contribution from point k to the complex interband optical
    conductivity, separated into Hermitian (H) and anti-Hermitian (AH)
    parts.

    Args:

    Returns:
    """
    num_wann = 1
    kubo_eigval_max = 1
    eig = np.zeros(num_wann)
    occ = np.zeros(num_wann)
    spin_decomp = False
    spn_nk = np.zeros(num_wann)
    kubo_adpt_smr = False
    vdum = np.zeros(3)
    del_eig = np.zeros(num_wann, 3)
    Delta_k = 1.0
    kubo_adpt_smr_fac = 1.0
    kubo_adpt_smr_max = 1.0
    kubo_smr_fixed_en_width = 1.0
    kubo_nfreq = 2
    kubo_freq_list = [1.0, 2.0]

    kubo_smr_type_index = 0
    # (n>=0) : Methfessel-Paxton case. See PRB 40, 3616 (1989).
    #
    # (n=-1 ): Cold smearing (Marzari-Vanderbilt). See PRL 82, 3296 (1999)
    #       1/2*erf(x-1/sqrt(2)) + 1/sqrt(2*pi)*exp(-(x-1/sqrt(2))**2) + 1/2
    #
    # (n=-99): Fermi-Dirac case: 1.0/(1.0+exp(-x)).

    for m in range(num_wann):
        for n in range(num_wann):
            if n == m:
                  continue
            if eig[m] > kubo_eigval_max or eig[n] > kubo_eigval_max:
                  continue
            if spin_decomp:
                if spn_nk[n] >= 0 and spn_nk[m] >= 0:
                    ispn = 1 # up --> up transition
                elif spn_nk[n] < 0 and spn_nk[m] < 0:
                    ispn = 2 # down --> down
                else:
                    ispn = 3 # spin-flip
            if kubo_adpt_smr: # Eq.(35) YWVS07
                vdum[:] = del_eig[m, :] - del_eig[n, :]
                joint_level_spacing = np.sqrt(np.dot(vdum, vdum))*Delta_k
                eta_smr = np.min(joint_level_spacing*kubo_adpt_smr_fac, kubo_adpt_smr_max)
            else:
                eta_smr = kubo_smr_fixed_en_width
            rfac1 = (occ(m) - occ(n))*(eig(m) - eig(n))
            occ_prod = occ(n)*(1.0 - occ(m))

            for ifreq in range(kubo_nfreq):
                #
                # Complex frequency for the anti-Hermitian conductivity
                #
                if kubo_adpt_smr:
                    omega = kubo_freq_list[ifreq] + 1j*eta_smr
                else:
                    omega = kubo_freq_list[ifreq]
                #
                # Broadened delta function for the Hermitian conductivity and JDOS
                #
                arg = (eig[m] - eig[n] - omega) / eta_smr
                # If only Hermitean part were computed, could speed up
                # by inserting here 'if(abs(arg)>10.0_dp) cycle'
                delta = utility_w0gauss(arg, kubo_smr_type_index, error, comm)/eta_smr
                if (allocated(error)):
                    return
                #
                # Lorentzian shape (for testing purposes)
                # delta=1.0_dp/(1.0_dp+arg*arg)/pi
                # delta=delta/eta_smr
                #
                cfac = 1j*rfac1/(eig[m] - eig[n] - omega)
                rfac2 = -np.pi*rfac1*delta
                for j in range(3):
                    for i in range(3):
                        kubo_H_k[i, j, ifreq]  += rfac2*AA[n, m, i]*AA[m, n, j]
                        kubo_AH_k[i, j, ifreq] += cfac*AA[n, m, i]*AA[m, n, j]
                        if (spin_decomp) then
                            kubo_H_k_spn(i, j, ispn, ifreq) = &
                            kubo_H_k_spn(i, j, ispn, ifreq) &
                            + rfac2*AA[n, m, i]*AA[m, n, j]
                            kubo_AH_k_spn(i, j, ispn, ifreq) = &
                            kubo_AH_k_spn(i, j, ispn, ifreq) &
                            + cfac*AA[n, m, i]*AA[m, n, j]



# *************************************************************************** #
#     dc Anomalous Hall conductivity and eventually (if 'mcd' string also     #
#       present in addition to 'ahe', e.g., 'ahe+mcd') dichroic optical       #
#     conductivity, both calculated on the same (adaptively-refined) mesh     #
# *************************************************************************** #

# *************************************************************************** #
#       Absorptive dichroic optical conductivity & JDOS on uniform mesh       #
# *************************************************************************** #

# *************************************************************************** #
#      Absorptive ordinary optical conductivity & JDOS on a uniform mesh      #
# *************************************************************************** #


# *************************************************************************** #
#                            Orbital magnetization                            #
# *************************************************************************** #


# ==================================================
# *************************************************************************** #
#           Boltzmann transport coefficients (BoltzWann module)               #
# *************************************************************************** #
def boltzwann_main():
    pass


# ==================================================
# *************************************************************************** #
#           Gyrotropic transport coefficients (Gyrotropic module)             #
# *************************************************************************** #
def gyrotropic_main():
    pass
