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
from joblib import Parallel, delayed, wrap_non_picklable_objects


from symclosestwannier.util.utility import (
    fermi,
    fourier_transform_r_to_k,
    fourier_transform_r_to_k_new,
    fourier_transform_r_to_k_vec,
    spin_zeeman_interaction,
    spn_operator,
    thermal_avg,
)

from symclosestwannier.util.constants import elec_mass_SI, elem_charge_SI, hbar_SI, bohr, bohr_magn_SI, joul_to_eV

_num_proc = multiprocessing.cpu_count()

_alpha_A = [1, 2, 0]
_beta_A = [2, 0, 1]


# ==================================================
def spin_moment_main(cwi, operators):
    """
    Computes the spin magnetic moments, Ms_x, Ms_y, Ms_z.

    Args:
        cwi (CWInfo): CWInfo.
        operators (dict): operators.

    Returns:
        dict: Ms_x, Ms_y, Ms_z.
    """
    d = {"Ms_x": 0.0, "Ms_y": 0.0, "Ms_z": 0.0}

    if not cwi["spinors"]:
        return d
    else:
        if cwi["tb_gauge"]:
            atoms_list = list(cwi["atoms_frac"].values())
            atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
        else:
            atoms_frac = None

        B = cwi["magnetic_field"]
        theta = cwi["magnetic_field_theta"]
        phi = cwi["magnetic_field_phi"]
        g_factor = cwi["g_factor"]
        dim = cwi["num_wann"]

        N1, N2, N3 = cwi["kmesh"]
        kpoints = np.array(
            [[i / float(N1), j / float(N2), k / float(N3)] for i in range(N1) for j in range(N2) for k in range(N3)]
        )

        num_k = np.prod(cwi["kmesh"])

        mu_B = bohr_magn_SI * joul_to_eV

        # ==================================================
        @wrap_non_picklable_objects
        def spin_moment_main_k(kpt):
            HH = fourier_transform_r_to_k(operators["HH_R"], kpt, cwi["irvec"], cwi["ndegen"], atoms_frac)

            pauli_spn = fourier_transform_r_to_k_vec(operators["SS_R"], kpt, cwi["irvec"], cwi["ndegen"], atoms_frac)

            if cwi["zeeman_interaction"]:
                H_zeeman = spin_zeeman_interaction(B, theta, phi, pauli_spn, g_factor, cwi["num_wann"])
                HH += H_zeeman

            E, U = np.linalg.eigh(HH)
            HH = None

            spn_x, spn_y, spn_z = spn_operator(pauli_spn, g_factor, dim) / mu_B

            return thermal_avg([spn_x, spn_y, spn_z], E, U, cwi["fermi_energy"], T_Kelvin=0.0, num_k=num_k)

        # ==================================================
        kpoints_chunks = np.split(kpoints, [j for j in range(1000, len(kpoints), 1000)])

        for idx, kpoints in enumerate(kpoints_chunks):
            print(f"* {idx+1}/{len(kpoints_chunks)}")
            Ms_x_k, Ms_y_k, Ms_z_k = spin_moment_main_k(kpoints)
            d["Ms_x"] += Ms_x_k
            d["Ms_y"] += Ms_y_k
            d["Ms_z"] += Ms_z_k

        return d


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

    Returns:
        dict: dictionary of results.
    """
    if cwi["num_fermi"] == 0:
        raise Exception("Must specify one or more Fermi levels when berry=true")

    print("Properties calculated in berry_main \n ------------------------------------------")

    # initialization
    d = {
        "ahc_list": None,
        "morb": None,
        "kubo": None,
        "kubo_H": None,
        "kubo_AH": None,
        "kubo_H_spn": None,
        "kubo_AH_spn": None,
        "sc": None,
        "shc": None,
    }

    # (ahc)  Anomalous Hall conductivity (from Berry curvature)
    if cwi["berry_task"] == "ahc":
        print("* Anomalous Hall conductivity")

    # (morb) Orbital magnetization
    if cwi["berry_task"] == "morb":
        if cwi["transl_inv"]:
            raise Exception("transl_inv=T disabled for morb")

        print("* Orbital magnetization")

    # (kubo) Complex optical conductivity (Kubo-Greenwood) & JDOS
    if cwi["berry_task"] == "kubo":
        if cwi["spin_decomp"]:
            print("* Complex optical conductivity and its spin-decomposition")
            print("* Joint density of states and its spin-decomposition")
        else:
            print("* Complex optical conductivity")
            print("* Joint density of states")

    # (sc)   Nonlinear shift current
    if cwi["berry_task"] == "sc":
        print("* Shift current")

    # (shc)  Spin Hall conductivity
    if cwi["berry_task"] == "shc":
        print("* Spin Hall Conductivity")
        if cwi["shc_freq_scan"]:
            print("  Frequency scan")
        else:
            print("  Fermi energy scan")

    if cwi["transl_inv"]:
        print(
            "Using a translationally-invariant discretization for the band-diagonal Wannier matrix elements of r, etc."
        )

    # --- #

    if cwi["berry_task"] == "ahc":
        d["ahc_list"] = berry_get_ahc(cwi, operators)

    if cwi["berry_task"] == "kubo":
        kubo_H, kubo_AH, kubo_H_spn, kubo_AH_spn = berry_get_kubo(cwi, operators)

        d["kubo_H"] = kubo_H
        d["kubo_AH"] = kubo_AH
        d["kubo_H_spn"] = kubo_H_spn
        d["kubo_AH_spn"] = kubo_AH_spn

    if cwi["berry_task"] == "sc":
        kubo_nfreq = round((cwi["kubo_freq_max"] - cwi["kubo_freq_min"]) / cwi["kubo_freq_step"]) + 1
        sc_k_list = np.zeros((3, 6, kubo_nfreq))
        sc_list = np.zeros((3, 6, kubo_nfreq))

    if cwi["berry_task"] == "shc":
        if cwi["shc_freq_scan"]:
            d["shc_freq"] = berry_get_shc(cwi, operators)
        else:
            d["shc_fermi"] = berry_get_shc(cwi, operators)

    return d


# ==================================================
def gyrotropic_main(cwi, operators):
    """
    Computes the following quantities:
     (-d0)   D tensor (the Berry curvature dipole).
     (-dw)   D(w) tensor (the finite-frequency generalization of the Berry curvature dipole).
     (-k)    K tensor (orbital component of the kinetic magnetoelectric effect (kME)).
     (-spin) K tensor (spin component of the kinetic magnetoelectric effect (kME)).
     (-c)    C tensor (the ohmic conductivity).
     (-noa)  the interband contributionto the natural optical activity.
     (-dos)  the density of states.
     (-all)  computes all.

    Args:
        cwi (CWInfo): CWInfo.
        operators (dict): operators.

    Returns:
        dict: dictionary of results.
    """
    if cwi["num_fermi"] == 0:
        raise Exception("Must specify one or more Fermi levels when gyrotropic=true")

    print("Properties calculated in gyrotropic_main \n ------------------------------------------")

    # initialization
    d = {
        "gyro_K_orb": None,
        "gyro_K_spn": None,
        "gyro_DOS": None,
        "gyro_C": None,
        "gyro_D": None,
        "gyro_Dw": None,
        "gyro_NOA_orb": None,
        "gyro_NOA_spn": None,
    }

    win = cwi.win

    # (K tensor)  orbital component of the kinetic magnetoelectric effect (kME)
    if win.eval_K:
        if cwi["transl_inv"]:
            raise Exception("transl_inv=T disabled for K-tensor")

        print("* K-tensor  --- Eq.3 of TAS17 ")
        if win.eval_spn:
            print("    * including spin component ")
        else:
            print("    * excluding spin component ")

    # D(w) tensor (the finite-frequency generalization of the Berry curvature dipole).
    if win.eval_Dw:
        print("* Dw-tensor  --- Eq.12 of TAS17 ")

    if win.eval_C:
        print("* C-tensor  --- Eq.B6 of TAS17 ")

    if win.eval_NOA:
        print("* gamma-tensor of NOA --- Eq.C12 of TAS17 ")
        if win.eval_spn:
            print("    * including spin component ")
        else:
            print("    * excluding spin component ")

    if cwi["transl_inv"]:
        print(
            "Using a translationally-invariant discretization for the band-diagonal Wannier matrix elements of r, etc."
        )

    # --- #

    if win.eval_K:
        d["gyro_K_orb"], d["gyro_K_spn"] = gyrotropic_get_K(cwi, operators)

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
    w0gauss = 0.0  # in case of error return

    sqrtpm1 = 1.0 / np.sqrt(np.pi)

    # cold smearing  (Marzari-Vanderbilt)
    if n == -1:
        arg = min(200, (x - 1.0 / np.sqrt(2)) ** 2)
        w0gauss = sqrtpm1 * np.exp(-arg) * (2.0 - np.sqrt(2.0) * x)
    # Fermi-Dirac smearing
    elif n == -99:
        if np.abs(x) <= 36.0:
            w0gauss = 1.0 / (2.0 + np.exp(-x) + np.exp(+x))
        else:
            w0gauss = 0.0
    # Gaussian
    elif n == 0:
        arg = min(200, x**2)
        w0gauss = np.exp(-arg) * sqrtpm1
    elif n > 10 or n < 0:
        raise Exception("utility_w0gauss higher order (n>10) smearing is untested and unstable")
    # Methfessel-Paxton
    else:
        arg = min(200, x**2)
        w0gauss = np.exp(-arg) * sqrtpm1
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
            w0gauss += a * hp

    return w0gauss


# ==================================================
def wham_get_JJp_JJm_list(cwi, delHH, E, U, occ=None):
    """
    Compute JJ^+_a and JJ^-_a (a=Cartesian index)
    for a list of Fermi energies

    This routine is a replacement for
    wham_get_JJp_list and wham_getJJm_list.
    It computes both lists at once in a more
    efficient manner.
    """
    if occ is not None:
        nfermi_loc = 1
    else:
        nfermi_loc = cwi["num_fermi"]

    fermi_energy_list = cwi["fermi_energy_list"]
    num_k = E.shape[0]
    num_wann = cwi["num_wann"]

    if occ is not None:
        occ_list = np.array([occ])
    else:
        occ_list = np.array([fermi(E - ef, T=0.0) for ef in fermi_energy_list])

    if occ is not None:
        fac_m = np.array(
            [
                [
                    [
                        [
                            (
                                1.0 / (E[k, m] - E[k, n])
                                if occ_list[ife, k, m] < 0.5 and occ_list[ife, k, n] > 0.5
                                else 0.0
                            )
                            for m in range(num_wann)
                        ]
                        for n in range(num_wann)
                    ]
                    for k in range(num_k)
                ]
                for ife in range(nfermi_loc)
            ],
            dtype=float,
        )
        fac_p = np.array(
            [
                [
                    [
                        [
                            (
                                1.0 / (E[k, n] - E[k, m])
                                if occ_list[ife, k, m] < 0.5 and occ_list[ife, k, n] > 0.5
                                else 0.0
                            )
                            for n in range(num_wann)
                        ]
                        for m in range(num_wann)
                    ]
                    for k in range(num_k)
                ]
                for ife in range(nfermi_loc)
            ],
            dtype=float,
        )
    else:
        fac_m = np.array(
            [
                [
                    [
                        [
                            (
                                1.0 / (E[k, n] - E[k, m])
                                if E[k, n] > fermi_energy_list[ife] and E[k, m] < fermi_energy_list[ife]
                                else 0.0
                            )
                            for n in range(num_wann)
                        ]
                        for m in range(num_wann)
                    ]
                    for k in range(num_k)
                ]
                for ife in range(nfermi_loc)
            ],
            dtype=float,
        )
        fac_p = np.array(
            [
                [
                    [
                        [
                            (
                                1.0 / (E[k, m] - E[k, n])
                                if E[k, n] > fermi_energy_list[ife] and E[k, m] < fermi_energy_list[ife]
                                else 0.0
                            )
                            for m in range(num_wann)
                        ]
                        for n in range(num_wann)
                    ]
                    for k in range(num_k)
                ]
                for ife in range(nfermi_loc)
            ],
            dtype=float,
        )

    del E

    delHH_Band = U.transpose(0, 2, 1).conjugate()[np.newaxis, :, :, :] @ delHH @ U[np.newaxis, :, :, :]

    JJp_list = 1.0j * np.einsum("akmn,wkmn->awkmn", delHH_Band, fac_p, optimize=True)
    JJm_list = 1.0j * np.einsum("akmn,wkmn->awkmn", delHH_Band, fac_m, optimize=True)

    del fac_p
    del fac_m
    del delHH_Band

    JJp_list = (
        U[np.newaxis, np.newaxis, :, :, :]
        @ JJp_list
        @ U.transpose(0, 2, 1).conjugate()[np.newaxis, np.newaxis, :, :, :]
    )
    JJm_list = (
        U[np.newaxis, np.newaxis, :, :, :]
        @ JJm_list
        @ U.transpose(0, 2, 1).conjugate()[np.newaxis, np.newaxis, :, :, :]
    )

    return JJp_list, JJm_list


# ==================================================
def wham_get_eig_UU_HH_JJlist(cwi, delHH, E, U, occ=None):
    """
    Wrapper routine used to reduce number of Fourier calls
    Added the optional occ parameter
    """
    if occ is not None:
        JJp_list, JJm_list = wham_get_JJp_JJm_list(cwi, delHH, E, U, occ=occ)
    else:
        JJp_list, JJm_list = wham_get_JJp_JJm_list(cwi, delHH, E, U)

    return JJp_list, JJm_list


# ==================================================
def wham_get_occ_mat_list(cwi, U, E=None, occ=None):
    """
    Occupation matrix f, and g=1-f
    for a list of Fermi energies
    Tsirkin: !now optionally either E or occ parameters may be supplied
    (Changed consistently the calls from the Berry module)
    """
    if occ is not None:
        nfermi_loc = 1
    else:
        nfermi_loc = cwi["num_fermi"]

    fermi_energy_list = cwi["fermi_energy_list"]
    num_k = U.shape[0]
    num_wann = cwi["num_wann"]

    if occ is not None and E is not None:
        raise Exception("occ_list and eig cannot be both arguments in get_occ_mat_list")
    elif occ is None and E is None:
        raise Exception("either occ_list or eig must be passed as arguments to get_occ_mat_list")

    if occ is not None:
        occ_list = [occ]
    else:
        occ_list = np.array([fermi(E - ef, T=0.0) for ef in fermi_energy_list])

    f_list = np.zeros((nfermi_loc, num_k, num_wann, num_wann), dtype=np.complex128)
    g_list = np.zeros((nfermi_loc, num_k, num_wann, num_wann), dtype=np.complex128)

    f_list = np.einsum("kni,wki,kmi->wknm", U, occ_list, np.conjugate(U), optimize=True)

    g_list = -f_list
    for n in range(num_wann):
        g_list[:, :, n, n] += 1.0

    return f_list, g_list


# ==================================================
def berry_get_imfgh_klist(cwi, operators, kpoints, imf=False, img=False, imh=False, occ=None, ladpt=None):
    """
    Calculates the three quantities needed for the orbital magnetization:
        * -2Im[f(k)] [Eq.33 CTVR06, Eq.6 LVTS12]
        * -2Im[g(k)] [Eq.34 CTVR06, Eq.7 LVTS12]
        * -2Im[h(k)] [Eq.35 CTVR06, Eq.8 LVTS12]

    They are calculated together (to reduce the number of Fourier calls)
    for a list of Fermi energies, and stored in axial-vector form.

    Args:
        imf (bool, optional): calculate -2Im[f(k)] ?
        img (bool, optional): calculate -2Im[g(k)] ?
        imh (bool, optional): calculate -2Im[h(k)] ?
        occ (ndarray, optional): occupancy.
        ladpt (ndarray, optional): .
    """
    if cwi["tb_gauge"]:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    if kpoints.ndim == 1:
        kpoints = np.array([kpoints])

    if occ is not None:
        num_fermi_loc = 1
    else:
        num_fermi_loc = cwi["num_fermi"]

    if ladpt is not None:
        todo = ladpt
    else:
        todo = [True] * num_fermi_loc

    num_wann = cwi["num_wann"]

    HH, delHH = fourier_transform_r_to_k_new(
        operators["HH_R"], kpoints, cwi["unit_cell_cart"], cwi["irvec"], cwi["ndegen"], atoms_frac
    )

    if cwi["zeeman_interaction"]:
        B = cwi["magnetic_field"]
        theta = cwi["magnetic_field_theta"]
        phi = cwi["magnetic_field_phi"]
        g_factor = cwi["g_factor"]

        pauli_spin = fourier_transform_r_to_k_vec(operators["SS_R"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac)
        H_zeeman = spin_zeeman_interaction(B, theta, phi, pauli_spin, g_factor, cwi["num_wann"])
        HH += H_zeeman

    E, U = np.linalg.eigh(HH)

    #
    # Gather W-gauge matrix objects
    #
    if occ is not None:
        JJp_list, JJm_list = wham_get_eig_UU_HH_JJlist(cwi, delHH, E, U, occ=occ)
        f_list, g_list = wham_get_occ_mat_list(cwi, U, occ=occ)
    else:
        JJp_list, JJm_list = wham_get_eig_UU_HH_JJlist(cwi, delHH, E, U)
        f_list, g_list = wham_get_occ_mat_list(cwi, U, E=E)

    del delHH
    del E
    del U
    del occ

    AA, OOmega = fourier_transform_r_to_k_vec(
        operators["AA_R"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac, cwi["unit_cell_cart"], pseudo=True
    )

    if imf:
        imf_k_list = np.zeros((num_fermi_loc, len(kpoints), 3, 3))
        # Trace formula for -2Im[f], Eq.(51) LVTS12
        for ife in range(num_fermi_loc):
            if todo[ife]:
                for i in range(3):
                    # J0 term (Omega_bar term of WYSV06)
                    imf_k_list[ife, :, 0, i] = np.real(
                        np.trace(f_list[ife, :, :, :] @ OOmega[i, :, :, :], axis1=1, axis2=2)
                    )
                    #
                    # J1 term (DA term of WYSV06)
                    imf_k_list[ife, :, 1, i] = -2.0 * np.imag(
                        np.trace(AA[_alpha_A[i], :, :, :] @ JJp_list[_beta_A[i], ife, :, :, :], axis1=1, axis2=2)
                        + np.trace(JJm_list[_alpha_A[i], ife, :, :, :] @ AA[_beta_A[i], :, :, :], axis1=1, axis2=2)
                    )
                    #
                    # J2 term (DD of WYSV06)
                    imf_k_list[ife, :, 2, i] = -2.0 * np.imag(
                        np.trace(
                            JJm_list[_alpha_A[i], ife, :, :, :] @ JJp_list[_beta_A[i], ife, :, :, :], axis1=1, axis2=2
                        )
                    )

    else:
        imf_k_list = None

    if img and imh:
        img_k_list = np.zeros((num_fermi_loc, len(kpoints), 3, 3))
        imh_k_list = np.zeros((num_fermi_loc, len(kpoints), 3, 3))

        BB = fourier_transform_r_to_k_vec(
            operators["BB_R"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac, cwi["unit_cell_cart"]
        )

        CC = np.array(
            [
                [
                    fourier_transform_r_to_k(operators["CC_R"][i, j], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac)
                    for j in range(3)
                ]
                for i in range(3)
            ]
        )

        tmp = np.zeros((5, len(kpoints), num_wann, num_wann), dtype=complex)
        # tmp[0:2,:,:,:] ... not dependent on inner loop variables
        # tmp[0.:,:,:] ..... HH . AA(:,:,alpha_A(i))
        # tmp[1,:,:,:] ..... LLambda_ij [Eq. (37) LVTS12] expressed as a pseudovector
        # tmp[2,:,:,:] ..... HH . OOmega(:,:,i)
        # tmp[3:4,:,:,:] ... working matrices for matrix products of inner loop

        # Trace formula for -2Im[g], Eq.(66) LVTS12
        # Trace formula for -2Im[h], Eq.(56) LVTS12

        for i in range(3):
            tmp[0] = HH @ AA[_alpha_A[i]]
            tmp[2] = HH @ OOmega[i]
            #
            # LLambda_ij [Eq. (37) LVTS12] expressed as a pseudovector
            tmp[1] = 1.0j * (CC[_alpha_A[i], _beta_A[i]] - CC[_alpha_A[i], _beta_A[i]].transpose(0, 2, 1).conjugate())

            for ife in range(num_fermi_loc):
                #
                # J0 terms for -2Im[g] and -2Im[h]
                #
                tmp[3] = tmp[0] @ f_list[ife]
                tmp[4] = tmp[3] @ AA[_beta_A[i]]

                s = 2.0 * np.imag(np.trace(f_list[ife] @ tmp[4], axis1=1, axis2=2))
                img_k_list[ife, :, 0, i] = np.imag(np.trace(f_list[ife] @ tmp[1], axis1=1, axis2=2)) - s
                imh_k_list[ife, :, 0, i] = np.imag(np.trace(f_list[ife] @ tmp[2], axis1=1, axis2=2)) + s

                #
                # J1 terms for -2Im[g] and -2Im[h]
                #
                # tmp(:,:,1) = HH . AA(:,:,alpha_A(i))
                # tmp(:,:,4) = HH . JJm_list(:,:,ife,alpha_A(i))
                tmp[3] = HH @ JJm_list[_alpha_A[i], ife]

                img_k_list[ife, :, 1, i] = -2.0 * (
                    np.imag(np.trace(JJm_list[_alpha_A[i], ife] @ BB[_beta_A[i]], axis1=1, axis2=2))
                    - np.imag(np.trace(JJm_list[_beta_A[i], ife] @ BB[_alpha_A[i]], axis1=1, axis2=2))
                )
                imh_k_list[ife, :, 1, i] = -2.0 * (
                    np.imag(np.trace(tmp[0] @ JJp_list[_beta_A[i], ife], axis1=1, axis2=2))
                    + np.imag(np.trace(tmp[3] @ AA[_beta_A[i]], axis1=1, axis2=2))
                )

                #
                # J2 terms for -2Im[g] and -2Im[h]
                #
                # tmp(:,:,4) = JJm_list(:,:,ife,alpha_A(i)) . HH
                # tmp(:,:,5) = HH . JJm_list(:,:,ife,alpha_A(i))
                tmp[3] = JJm_list[_alpha_A[i], ife] @ HH
                tmp[4] = HH @ JJm_list[_alpha_A[i], ife]

                img_k_list[ife, :, 2, i] = -2.0 * np.imag(
                    np.trace(tmp[3] @ JJp_list[_beta_A[i], ife], axis1=1, axis2=2)
                )
                imh_k_list[ife, :, 2, i] = -2.0 * np.imag(
                    np.trace(tmp[4] @ JJp_list[_beta_A[i], ife], axis1=1, axis2=2)
                )

    return imf_k_list, img_k_list, imh_k_list


# ==================================================
def berry_get_imf_klist(cwi, operators, kpoints, occ=None, ladpt=None):
    """
    Calculates the Berry curvature traced over the occupied
    states, -2Im[f(k)] [Eq.33 CTVR06, Eq.6 LVTS12] for a list
    of Fermi energies, and stores it in axial-vector form
    """
    if occ is not None:
        imf_k_list, _, _ = berry_get_imfgh_klist(cwi, operators, kpoints, imf=True, occ=occ)
    else:
        if ladpt is not None:
            imf_k_list, _, _ = berry_get_imfgh_klist(cwi, operators, kpoints, imf=True, ladpt=ladpt)
        else:
            imf_k_list, _, _ = berry_get_imfgh_klist(cwi, operators, kpoints, imf=True)

    return imf_k_list


# ==================================================
def berry_get_ahc(cwi, operators):
    """
    Anomalous Hall conductivity, in S/cm.
    The three independent components σx = σyz, σy = σzx, and σz = σxy are computed.
    The real part Re[σ^AH_αβ] describes the anomalous Hall conductivity (AHC), and remains finite in the static limit,
    while the imaginary part Im[σ^H_αβ] describes magnetic circular dichroism, and vanishes as ω → 0.

    Args:
        cwi (CWInfo): CWInfo.
        operators (dict): operators.

    Returns:
        ndarray: Anomalous Hall conductivity.
    """
    num_fermi = cwi["num_fermi"]

    berry_kmesh = cwi["berry_kmesh"]

    berry_curv_unit = cwi["berry_curv_unit"]
    berry_curv_adpt_kmesh = cwi["berry_curv_adpt_kmesh"]
    berry_curv_adpt_kmesh_thresh = cwi["berry_curv_adpt_kmesh_thresh"]

    # Mesh spacing in reduced coordinates
    db1 = 1.0 / float(berry_kmesh[0])
    db2 = 1.0 / float(berry_kmesh[1])
    db3 = 1.0 / float(berry_kmesh[2])

    kweight = db1 * db2 * db3
    kweight_adpt = kweight / berry_curv_adpt_kmesh**3

    adkpt = np.zeros((3, berry_curv_adpt_kmesh**3))
    ikpt = 0
    for i in range(berry_curv_adpt_kmesh):
        for j in range(berry_curv_adpt_kmesh):
            for k in range(berry_curv_adpt_kmesh):
                adkpt[0, ikpt] = db1 * ((i + 0.5) / berry_curv_adpt_kmesh - 0.5)
                adkpt[1, ikpt] = db2 * ((j + 0.5) / berry_curv_adpt_kmesh - 0.5)
                adkpt[2, ikpt] = db3 * ((k + 0.5) / berry_curv_adpt_kmesh - 0.5)
                ikpt = ikpt + 1

    # ==================================================
    @wrap_non_picklable_objects
    def berry_get_ahc_k(kpoints):
        """
        berry_get_imf_klist

        """
        imf_k_list = berry_get_imf_klist(cwi, operators, kpoints)
        imf_list = np.zeros((num_fermi, 3, 3))

        for k in range(len(kpoints)):
            kpt = kpoints[k]
            ladpt = [False] * num_fermi
            adpt_counter_list = [0] * num_fermi

            for ife in range(num_fermi):
                vdum = np.array([sum(imf_k_list[ife, k, :, a]) for a in range(3)])

                if berry_curv_unit == "bohr2":
                    vdum = vdum / bohr**2

                rdum = np.sqrt(np.dot(vdum, vdum))
                if rdum > berry_curv_adpt_kmesh_thresh:
                    adpt_counter_list[ife] = adpt_counter_list[ife] + 1
                    ladpt[ife] = True
                else:
                    imf_list[ife, :, :] += imf_k_list[ife, k, :, :] * kweight

            if np.any(ladpt):
                # for loop_adpt in range(berry_curv_adpt_kmesh**3):
                # Using imf_k_list here would corrupt values for other
                # frequencies, hence dummy. Only i-th element is used

                imf_k_list_dummy = berry_get_imf_klist(cwi, operators, kpt[np.newaxis, :] + adkpt.T, ladpt=ladpt)

                for ife_ in range(num_fermi):
                    if ladpt[ife_]:
                        for loop_adpt in range(berry_curv_adpt_kmesh**3):
                            imf_list[ife_, :, :] += imf_k_list_dummy[ife_, loop_adpt, :, :] * kweight_adpt

        del imf_k_list

        return imf_list

    # ==================================================
    N1, N2, N3 = cwi["berry_kmesh"]
    kpoints = np.array(
        [[i / float(N1), j / float(N2), k / float(N3)] for i in range(N1) for j in range(N2) for k in range(N3)]
    )

    num_k = np.prod(cwi["berry_kmesh"])

    kpoints_chunks = np.split(kpoints, [j for j in range(100000, len(kpoints), 100000)])

    imf_list = np.zeros((num_fermi, 3, 3), dtype=float)

    res = Parallel(n_jobs=_num_proc, verbose=10)(delayed(berry_get_ahc_k)(kpoints) for kpoints in kpoints_chunks)

    for v in res:
        imf_list += v

    """
    --------------------------------------------------------------------
     At this point imf contains

     (1/N) sum_k Omega_{alpha beta}(k),

     an approximation to

     V_c.int dk/(2.pi)^3 Omega_{alpha beta}(k) dk

     (V_c is the cell volume). We want

     sigma_{alpha beta}=-(e^2/hbar) int dk/(2.pi)^3 Omega(k) dk

     Hence need to multiply by -(e^2/hbar.V_c).
     To get a conductivity in units of S/cm,

     (i)   Divide by V_c to obtain (1/N) sum_k omega(k)/V_c, with units
           of [L]^{-1} (Berry curvature Omega(k) has units of [L]^2)
     (ii)  [L] = Angstrom. Multiply by 10^8 to convert to (cm)^{-1}
     (iii) Multiply by -e^2/hbar in SI, with has units ofconductance,
           (Ohm)^{-1}, or Siemens (S), to get the final result in S/cm

     ===========================
     fac = -e^2/(hbar.V_c*10^-8)
     ===========================

     with 'V_c' in Angstroms^3, and 'e', 'hbar' in SI units
     -------------------------------------------------------------------
    """

    cell_volume = cwi["unit_cell_volume"]

    fac = -1.0e8 * elem_charge_SI**2 / (hbar_SI * cell_volume)

    ahc_list = imf_list * fac

    return ahc_list


# ==================================================
def berry_get_kubo(cwi, operators):
    """
    Complex interband optical conductivity, in S/cm,
    separated into Hermitian (Kubo_H) and anti-Hermitian (Kubo_AH) parts.

    Args:
        cwi (CWInfo): CWInfo.
        operators (dict):

    Returns:
        tuple: Kubo_H, Kubo_AH, Kubo_H_spn, Kubo_AH_spn.
    """
    if cwi["tb_gauge"]:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    ef = cwi["fermi_energy"]
    berry_kmesh = cwi["berry_kmesh"]
    num_wann = cwi["num_wann"]

    spin_decomp = cwi["spin_decomp"]
    spn_nk = np.zeros(num_wann)

    kubo_adpt_smr = cwi["kubo_adpt_smr"]
    kubo_adpt_smr_fac = cwi["kubo_adpt_smr_fac"]
    kubo_adpt_smr_max = cwi["kubo_adpt_smr_max"]
    kubo_smr_fixed_en_width = cwi["kubo_smr_fixed_en_width"]

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
        kubo_eigval_max = 100000

    use_degen_pert = cwi["use_degen_pert"]
    degen_thr = cwi["degen_thr"]

    kubo_freq_list = np.arange(cwi["kubo_freq_min"], cwi["kubo_freq_max"], cwi["kubo_freq_step"])
    # Replace imaginary part of frequency with a fixed value
    if not kubo_adpt_smr and kubo_smr_fixed_en_width != 0.0:
        kubo_freq_list = np.real(kubo_freq_list) + 1.0j * kubo_smr_fixed_en_width

    kubo_nfreq = len(kubo_freq_list)

    # ==================================================
    @wrap_non_picklable_objects
    def berry_get_kubo_k(kpt):
        """
        calculate
        Complex interband optical conductivity, in S/cm,
        separated into Hermitian (Kubo_H) and anti-Hermitian (Kubo_AH) parts.

        Args:
            kpt (ndarray): kpoint.

        Returns:
            tuple: Kubo_H, Kubo_AH, Kubo_H_spn, Kubo_AH_spn.
        """
        if kpt.ndim == 1:
            kpt = np.array([kpt])

        HH, delHH = fourier_transform_r_to_k_new(
            operators["HH_R"], kpt, cwi["unit_cell_cart"], cwi["irvec"], cwi["ndegen"], atoms_frac
        )

        if cwi["zeeman_interaction"]:
            B = cwi["magnetic_field"]
            theta = cwi["magnetic_field_theta"]
            phi = cwi["magnetic_field_phi"]
            g_factor = cwi["g_factor"]

            pauli_spin = fourier_transform_r_to_k_vec(operators["SS_R"], kpt, cwi["irvec"], cwi["ndegen"], atoms_frac)
            H_zeeman = spin_zeeman_interaction(B, theta, phi, pauli_spin, g_factor, cwi["num_wann"])
            HH += H_zeeman

        E, U = np.linalg.eigh(HH)
        HH = None
        D_h = wham_get_D_h(delHH, E, U)
        AA = fourier_transform_r_to_k_vec(operators["AA_R"], kpt, cwi["irvec"], cwi["ndegen"], atoms_frac)
        Avec = np.array([U.transpose(0, 2, 1).conjugate() @ AA[i] @ U for i in range(3)])
        AA = None
        A = Avec + 1.0j * D_h  # Eq.(25) WYSV06
        Avec = None
        D_h = None

        if kubo_adpt_smr:
            delE = wham_get_deleig(delHH, E, U, use_degen_pert, degen_thr)
            Delta_k = kmesh_spacing_mesh(berry_kmesh, cwi["B"])

        delHH = None
        U = None

        kubo_H = 1.0j * np.zeros((kubo_nfreq, 3, 3))
        kubo_AH = 1.0j * np.zeros((kubo_nfreq, 3, 3))

        if spin_decomp:
            kubo_H_spn = 1.0j * np.zeros((kubo_nfreq, 3, 3, 3))
            kubo_AH_spn = 1.0j * np.zeros((kubo_nfreq, 3, 3, 3))
        else:
            kubo_H_spn = 0.0
            kubo_AH_spn = 0.0

        occ = fermi(E - ef, T=0.0)

        for k in range(len(kpt)):
            for m in range(num_wann):
                for n in range(num_wann):
                    if m == n:
                        continue

                    if E[k, m] > kubo_eigval_max or E[k, n] > kubo_eigval_max:
                        continue

                    ekm = E[k, m]
                    ekn = E[k, n]
                    fkm = occ[k, m]
                    fkn = occ[k, n]

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
                        # if eta_smr < 1e-6:
                        #     eta_smr = 1e-6
                    else:
                        eta_smr = kubo_smr_fixed_en_width

                    # Complex frequency for the anti-Hermitian conductivity
                    if kubo_adpt_smr:
                        omega_list = np.real(kubo_freq_list) + 1.0j * eta_smr
                    else:
                        omega_list = kubo_freq_list

                    # Broadened delta function for the Hermitian conductivity and JDOS
                    arg = (ekm - ekn - np.real(omega_list)) / eta_smr
                    delta = (
                        np.array([utility_w0gauss(arg[ifreq], kubo_smr_type_idx) for ifreq in range(kubo_nfreq)])
                        / eta_smr
                    )

                    rfac1 = (fkm - fkn) * (ekm - ekn)
                    rfac2 = -np.pi * rfac1 * delta
                    cfac = 1.0j * rfac1 / (ekm - ekn - omega_list)

                    aiknm_ajkmn = np.array([[A[i, k, n, m] * A[j, k, m, n] for j in range(3)] for i in range(3)])

                    kubo_H += rfac2[:, np.newaxis, np.newaxis] * aiknm_ajkmn[np.newaxis, :, :]
                    kubo_AH += cfac[:, np.newaxis, np.newaxis] * aiknm_ajkmn[np.newaxis, :, :]

                    if spin_decomp:
                        kubo_H_spn += rfac2[:, np.newaxis, np.newaxis] * aiknm_ajkmn[np.newaxis, :, :]
                        kubo_AH_spn += cfac[:, np.newaxis, np.newaxis] * aiknm_ajkmn[np.newaxis, :, :]

        return kubo_H, kubo_AH, kubo_H_spn, kubo_AH_spn

    # ==================================================
    N1, N2, N3 = cwi["berry_kmesh"]
    kpoints = np.array(
        [[i / float(N1), j / float(N2), k / float(N3)] for i in range(N1) for j in range(N2) for k in range(N3)]
    )

    num_k = np.prod(cwi["berry_kmesh"])

    kpoints_chunks = np.split(kpoints, [j for j in range(100000, len(kpoints), 100000)])

    res = Parallel(n_jobs=_num_proc, verbose=10)(delayed(berry_get_kubo_k)(kpt) for kpt in kpoints_chunks)

    kubo_H = 1.0j * np.zeros((kubo_nfreq, 3, 3))
    kubo_AH = 1.0j * np.zeros((kubo_nfreq, 3, 3))

    if spin_decomp:
        kubo_H_spn = 1.0j * np.zeros((kubo_nfreq, 3, 3, 3))
        kubo_AH_spn = 1.0j * np.zeros((kubo_nfreq, 3, 3, 3))
    else:
        kubo_H_spn = 0.0
        kubo_AH_spn = 0.0

    for kubo_H_k, kubo_AH_k, kubo_H_spn_k, kubo_AH_spn_k in res:
        kubo_H += kubo_H_k
        kubo_AH += kubo_AH_k
        kubo_H_spn += kubo_H_spn_k
        kubo_AH_spn += kubo_AH_spn_k

    """
    --------------------------------------------------------------------
    Convert to S/cm

    fac = e^2/(hbar.V_c*10^-8)

    with 'V_c' in Angstroms^3, and 'e', 'hbar' in SI units
    --------------------------------------------------------------------
    """

    cell_volume = cwi["unit_cell_volume"]

    fac = 1.0e8 * elem_charge_SI**2 / (hbar_SI * cell_volume) / num_k

    kubo_H *= fac
    kubo_AH *= fac

    if spin_decomp:
        kubo_H_spn *= fac
        kubo_AH_spn *= fac

    return kubo_H, kubo_AH, kubo_H_spn, kubo_AH_spn


# ==================================================
def berry_get_js_k(cwi, operators, kpoints, E, del_alpha_E, D_alpha_h, U):
    """
    ontribution from point k to the
    <psi_k | 1/2*(sigma_gamma*v_alpha + v_alpha*sigma_gamma) | psi_k>

    QZYZ18 Eq.(23) without hbar/2 (required by spin operator) and
    not divided by hbar (required by velocity operator)

    Junfeng Qiao (8/7/2018)
    """
    if cwi["tb_gauge"]:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    shc_alpha = cwi["shc_alpha"]
    shc_gamma = cwi["shc_gamma"]

    # =========== S_k ===========
    #  < u_k | sigma_gamma | u_k >, QZYZ18 Eq.(25)
    #  QZYZ18 Eq.(36)
    S_gamma_w = fourier_transform_r_to_k(
        operators["SS_R"][shc_gamma - 1], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac
    )
    #  QZYZ18 Eq.(30)
    S_gamma_k = U.transpose(0, 2, 1).conjugate() @ S_gamma_w @ U

    # =========== K_k ===========
    #  < u_k | sigma_gamma | \partial_alpha u_k >, QZYZ18 Eq.(26)
    #  QZYZ18 Eq.(37)
    S_gamma_R_alpha_w = fourier_transform_r_to_k(
        operators["SR_R"][shc_gamma - 1, shc_alpha - 1], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac
    )
    #   ! QZYZ18 Eq.(31)
    S_gamma_R_alpha_k = -1.0j * U.transpose(0, 2, 1).conjugate() @ S_gamma_R_alpha_w @ U
    K_k = S_gamma_R_alpha_k + S_gamma_k @ D_alpha_h

    # =========== L_k ===========
    # < u_k | sigma_gamma.H | \partial_alpha u_k >, QZYZ18 Eq.(27)
    # QZYZ18 Eq.(38)
    S_gamma_HR_alpha_w = fourier_transform_r_to_k(
        operators["SHR_R"][shc_gamma - 1, shc_alpha - 1], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac
    )
    # QZYZ18 Eq.(39)
    S_gamma_H_w = fourier_transform_r_to_k(
        operators["SH_R"][shc_gamma - 1], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac
    )
    # QZYZ18 Eq.(32)
    S_gamma_HR_alpha_k = -1.0j * U.transpose(0, 2, 1).conjugate() @ S_gamma_HR_alpha_w @ U
    S_gamma_H_k = U.transpose(0, 2, 1).conjugate() @ S_gamma_H_w @ U
    L_k = S_gamma_HR_alpha_k + S_gamma_H_k @ D_alpha_h

    # =========== B_k ===========
    #  < \psi_nk | sigma_gamma v_alpha | \psi_mk >, QZYZ18 Eq.(24)
    B_k = (
        np.einsum("km, knm->knm", del_alpha_E, S_gamma_k, optimize=True)
        + np.einsum("km, knm->knm", E, K_k, optimize=True)
        - L_k
    )

    # =========== js_k ===========
    #  QZYZ18 Eq.(23)
    #  note the S in SR_R,SHR_R,SH_R of get_SHC_R is sigma,
    #  to get spin current, we need to multiply it by hbar/2,
    #  also we need to divide it by hbar to recover the velocity
    #  operator, these are done outside of this subroutine
    js_k = 1.0 / 2.0 * (B_k + B_k.transpose(0, 2, 1).conjugate())

    return js_k


# ==================================================
def berry_get_shc_klist(cwi, operators, kpoints, band=False):
    """
    Contribution from a k-point to the spin Hall conductivity on a list
    of Fermi energies or a list of frequencies or a list of energy bands
      sigma_{alpha,beta}^{gamma}(k), alpha, beta, gamma = 1, 2, 3
                                                         (x, y, z, respectively)
    i.e. the Berry curvature-like term of QZYZ18 Eq.(3) & (4).
    The unit is angstrom^2, similar to that of Berry curvature of AHC.

    Note the berry_get_js_k() has not been multiplied by hbar/2 (as
    required by spin operator) and not been divided by hbar (as required
    by the velocity operator). The second velocity operator has not been
    divided by hbar as well. But these two hbar required by velocity
    operators are canceled by the preceding hbar^2 of QZYZ18 Eq.(3).

    - shc_k_fermi: return a list for different Fermi energies
    - shc_k_freq:  return a list for different frequencies
    - shc_k_band:  return a list for each energy band

    Junfeng Qiao (18/8/2018)
    """
    if kpoints.ndim == 1:
        kpoints = np.array([kpoints])

    if cwi["tb_gauge"]:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    kubo_adpt_smr = cwi["kubo_adpt_smr"]
    kubo_adpt_smr_fac = cwi["kubo_adpt_smr_fac"]
    kubo_adpt_smr_max = cwi["kubo_adpt_smr_max"]
    kubo_smr_fixed_en_width = cwi["kubo_smr_fixed_en_width"]

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
        kubo_eigval_max = 100000

    kubo_freq_list = np.arange(cwi["kubo_freq_min"], cwi["kubo_freq_max"], cwi["kubo_freq_step"])
    # Replace imaginary part of frequency with a fixed value
    if not kubo_adpt_smr and kubo_smr_fixed_en_width != 0.0:
        kubo_freq_list = np.real(kubo_freq_list) + 1.0j * kubo_smr_fixed_en_width

    kubo_nfreq = len(kubo_freq_list)

    # Hamiltonian

    HH, delHH = fourier_transform_r_to_k_new(
        operators["HH_R"], kpoints, cwi["unit_cell_cart"], cwi["irvec"], cwi["ndegen"], atoms_frac
    )

    # if cwi["zeeman_interaction"]:
    #     B = cwi["magnetic_field"]
    #     theta = cwi["magnetic_field_theta"]
    #     phi = cwi["magnetic_field_phi"]
    #     g_factor = cwi["g_factor"]

    #     pauli_spin = fourier_transform_r_to_k_vec(operators["SS_R"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac)
    #     H_zeeman = spin_zeeman_interaction(B, theta, phi, pauli_spin, g_factor, cwi["num_wann"])
    #     HH += H_zeeman

    E, U = np.linalg.eigh(HH)
    HH = None

    use_degen_pert = cwi["use_degen_pert"]
    degen_thr = cwi["degen_thr"]

    delE = wham_get_deleig(delHH, E, U, use_degen_pert, degen_thr)
    D_h = wham_get_D_h(delHH, E, U)

    AA = fourier_transform_r_to_k_vec(operators["AA_R"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac)
    Avec = np.array([U.transpose(0, 2, 1).conjugate() @ AA[i] @ U for i in range(3)])
    AA = None
    AA = Avec + 1.0j * D_h  # Eq.(25) WYSV06
    Avec = None

    # SHC

    shc_alpha = cwi["shc_alpha"]
    shc_beta = cwi["shc_beta"]

    shc_bandshift = cwi["shc_bandshift"]
    shc_bandshift_firstband = cwi["shc_bandshift_firstband"]
    shc_bandshift_energyshift = cwi["shc_bandshift_energyshift"]

    if shc_bandshift:
        E[:, shc_bandshift_firstband:] += shc_bandshift_energyshift

    del_alpha_E = delE[shc_alpha - 1]
    D_alpha_h = D_h[shc_alpha - 1]
    js_k = berry_get_js_k(cwi, operators, kpoints, E, del_alpha_E, D_alpha_h, U)

    kubo_adpt_smr = cwi["kubo_adpt_smr"]
    berry_kmesh = cwi["berry_kmesh"]
    if kubo_adpt_smr:
        Delta_k = kmesh_spacing_mesh(berry_kmesh, cwi["B"])

    lfreq = False
    lfermi = False
    if cwi["shc_freq_scan"]:
        shc_k_freq = np.zeros((kubo_nfreq, len(kpoints)))
        lfreq = True
    else:
        shc_k_fermi = np.zeros((cwi["num_fermi"], len(kpoints)))
        lfermi = True

    lband = False
    # if cwi["kpath_bands_colour"] == "shc" and band:
    if band:
        shc_k_band = np.zeros(cwi["num_wann"])
        lband = True
        lfreq = False
        lfermi = False

    if lfreq:
        ef = cwi["fermi_energy"]
        occ_freq = fermi(E - ef, T=0.0)
    elif lfermi:
        fermi_energy_list = cwi["fermi_energy_list"]
        occ_fermi = np.array([fermi(E - ef, T=0.0) for ef in fermi_energy_list])

    num_wann = cwi["num_wann"]

    for k in range(len(kpoints)):
        for n in range(num_wann):
            # get Omega_{n,alpha beta}^{gamma}
            if lfreq:
                omega_list = np.zeros(kubo_nfreq)
            elif lfermi or lband:
                omega = 0.0

            for m in range(num_wann):
                if m == n:
                    continue

                if E[k, m] > kubo_eigval_max or E[k, n] > kubo_eigval_max:
                    continue

                rfac = E[k, m] - E[k, n]

                # this will calculate AHC
                # prod = -rfac*cmplx_i*AA(n, m, shc_alpha) * rfac*cmplx_i*AA(m, n, shc_beta)
                prod = js_k[k, n, m] * 1.0j * rfac * AA[shc_beta - 1, k, m, n]
                if kubo_adpt_smr:
                    # Eq.(35) YWVS07
                    vdum = delE[:, k, m] - delE[:, k, n]
                    joint_level_spacing = np.sqrt(np.dot(vdum, vdum)) * Delta_k
                    eta_smr = min(joint_level_spacing * kubo_adpt_smr_fac, kubo_adpt_smr_max)
                    if eta_smr < 1e-6:
                        eta_smr = 1e-6
                else:
                    eta_smr = kubo_smr_fixed_en_width

                if lfreq:
                    for ifreq in range(kubo_nfreq):
                        cdum = np.real(kubo_freq_list[ifreq]) + 1.0j * eta_smr
                        cfac = -2.0 / (rfac**2 - cdum**2)
                        omega_list[ifreq] += cfac * np.imag(prod)
                elif lfermi or lband:
                    rfac = -2.0 / (rfac**2 + eta_smr**2)
                    omega += rfac * np.imag(prod)

            if lfermi:
                shc_k_fermi[:, k] += occ_fermi[:, k, n] * omega
            elif lfreq:
                shc_k_freq[:, k] += occ_freq[k, n] * omega_list
            elif lband:
                shc_k_band[n] = omega

    if lfermi:
        return shc_k_fermi
    elif lfreq:
        return shc_k_freq
    elif lband:
        return shc_k_band


# ==================================================
def berry_get_shc(cwi, operators):
    """
    Spin Hall conductivity, in (hbar/e)*S/cm.

    Args:
        cwi (CWInfo): CWInfo.
        operators (dict): operators.

    Returns:
        ndarray: Spin Hall conductivity.
    """
    num_fermi = cwi["num_fermi"]

    berry_kmesh = cwi["berry_kmesh"]

    berry_curv_unit = cwi["berry_curv_unit"]
    berry_curv_adpt_kmesh = cwi["berry_curv_adpt_kmesh"]
    berry_curv_adpt_kmesh_thresh = cwi["berry_curv_adpt_kmesh_thresh"]

    # Mesh spacing in reduced coordinates
    db1 = 1.0 / float(berry_kmesh[0])
    db2 = 1.0 / float(berry_kmesh[1])
    db3 = 1.0 / float(berry_kmesh[2])

    # Do not read 'kpoint.dat'. Loop over a regular grid in the full BZ
    kweight = db1 * db2 * db3
    kweight_adpt = kweight / berry_curv_adpt_kmesh**3

    adkpt = np.zeros((3, berry_curv_adpt_kmesh**3))
    ikpt = 0
    for i in range(berry_curv_adpt_kmesh):
        for j in range(berry_curv_adpt_kmesh):
            for k in range(berry_curv_adpt_kmesh):
                adkpt[0, ikpt] = db1 * ((i + 0.5) / berry_curv_adpt_kmesh - 0.5)
                adkpt[1, ikpt] = db2 * ((j + 0.5) / berry_curv_adpt_kmesh - 0.5)
                adkpt[2, ikpt] = db3 * ((k + 0.5) / berry_curv_adpt_kmesh - 0.5)
                ikpt = ikpt + 1

    # ==================================================
    @wrap_non_picklable_objects
    def berry_get_shc_k(kpoints):
        """
        berry_get_imf_klist
        """
        shc_k_list = berry_get_shc_klist(cwi, operators, kpoints)

        if cwi["shc_freq_scan"]:
            return np.sum(shc_k_list, axis=1) * kweight
        else:
            shc_list = np.zeros((num_fermi))

            for k in range(len(kpoints)):
                kpt = kpoints[k]
                ladpt = [False] * num_fermi
                adpt_counter_list = [0] * num_fermi

                for ife in range(num_fermi):
                    rdum = abs(shc_k_list[ife, k])

                    if berry_curv_unit == "bohr2":
                        rdum = rdum / bohr**2

                    if rdum > berry_curv_adpt_kmesh_thresh:
                        adpt_counter_list[ife] = adpt_counter_list[ife] + 1
                        ladpt[ife] = True
                    else:
                        shc_list[ife] += shc_k_list[ife, k] * kweight

                if np.any(ladpt):
                    # for loop_adpt in range(berry_curv_adpt_kmesh**3):
                    # !Using shc_k here would corrupt values for other
                    # !kpt, hence dummy. Only if-th element is used.
                    shc_k_list_dummy = berry_get_shc_klist(cwi, operators, kpt[np.newaxis, :] + adkpt.T)

                    for ife_ in range(num_fermi):
                        if ladpt[ife_]:
                            for loop_adpt in range(berry_curv_adpt_kmesh**3):
                                shc_list[ife_] += shc_k_list_dummy[ife_, loop_adpt] * kweight_adpt

            return shc_list

    # ==================================================
    N1, N2, N3 = cwi["berry_kmesh"]
    kpoints = np.array(
        [[i / float(N1), j / float(N2), k / float(N3)] for i in range(N1) for j in range(N2) for k in range(N3)]
    )

    num_k = np.prod(cwi["berry_kmesh"])

    kpoints_chunks = np.split(kpoints, [j for j in range(100000, len(kpoints), 100000)])

    res = Parallel(n_jobs=_num_proc, verbose=10)(delayed(berry_get_shc_k)(kpoints) for kpoints in kpoints_chunks)

    shc_list = np.sum(res, axis=0)

    """
    --------------------------------------------------------------------
    Convert to the unit: (hbar/e) S/cm

    at this point, we need to
        (i)   multiply -e^2/hbar/(V*N_k) as in the QZYZ18 Eq.(5),
              note 1/N_k has already been applied by the kweight

        (ii)  convert charge current to spin current:
              divide the result by -e and multiply hbar/2 to
              recover the spin current, so the overall
              effect is -hbar/2/e

        (iii) multiply 1e8 to convert it to the unit S/cm

    So, the overall factor is

    fac = 1.0e8 * e^2 / hbar / V / 2.0

    and the final unit of spin Hall conductivity is (hbar/e)S/cm
     -------------------------------------------------------------------
    """

    cell_volume = cwi["unit_cell_volume"]

    fac = 1.0e8 * elem_charge_SI**2 / (hbar_SI * cell_volume) / 2.0

    shc_list *= fac

    return shc_list


# ==================================================
def gyrotropic_get_K(cwi, operators):
    """
    Computes the following quantities:

    - gyro_K_orb = delta(E_kn-E_f).(d E_{kn}/d k_i).(2.hbar/e).m^orb_{kn,j}
      [units of (length^3)*energy]

    - gyro_K_spn = delta(E_kn-E_f).(d E_{kn}/d k_i).sigma_{kn,j} (sigma = Pauli matrix)
      [units of length]

    Args:
        cwi (CWInfo): CWInfo.
        operators (dict): operators.

    Returns:
        ndarray: Spin Hall conductivity.
    """
    if cwi["tb_gauge"]:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    gyrotropic_kmesh = cwi["gyrotropic_kmesh"]
    gyrotropic_box = cwi.win.gyrotropic_box
    gyrotropic_box_corner = cwi.win.gyrotropic_box_corner

    # Mesh spacing in reduced coordinates
    db1 = 1.0 / float(gyrotropic_kmesh[0])
    db2 = 1.0 / float(gyrotropic_kmesh[1])
    db3 = 1.0 / float(gyrotropic_kmesh[2])

    # Do not read 'kpoint.dat'. Loop over a regular grid in the full BZ
    kweight = db1 * db2 * db3 * np.linalg.det(gyrotropic_box)

    gyrotropic_degen_thresh = cwi["gyrotropic_degen_thresh"]
    gyrotropic_smr_max_arg = cwi["gyrotropic_smr_max_arg"]
    eta_smr = cwi["gyrotropic_smr_fixed_en_width"]
    use_degen_pert = cwi["use_degen_pert"]
    degen_thr = cwi["degen_thr"]
    num_wann = cwi["num_wann"]

    if cwi["gyrotropic_band_list"] is None:
        gyrotropic_band_list = [n for n in range(cwi["num_wann"])]
        gyrotropic_num_bands = cwi["num_wann"]
    else:
        gyrotropic_band_list = [int(n) for n in gyrotropic_band_list.split(",")]
        gyrotropic_num_bands = len(gyrotropic_band_list)

    mum_fermi = cwi["num_fermi"]
    fermi_energy_list = cwi["fermi_energy_list"]

    if cwi["gyrotropic_smr_type"] == "gauss":
        gyrotropic_smr_type_idx = 0
    elif "m-p" in cwi["gyrotropic_smr_type"]:
        m_pN = cwi["gyrotropic_smr_type"]
        gyrotropic_smr_type_idx = m_pN[2:]
    elif cwi["gyrotropic_smr_type"] == "m-v" or cwi["gyrotropic_smr_type"] == "cold":
        gyrotropic_smr_type_idx = -1
    elif cwi["gyrotropic_smr_type"] == "f-d":
        gyrotropic_smr_type_idx = -99

    # ==================================================
    @wrap_non_picklable_objects
    def gyrotropic_get_K_k(kpoints):
        """
        berry_get_imf_klist
        """
        if kpoints.ndim == 1:
            kpoints = np.array([kpoints])

        kpoints = np.array([gyrotropic_box_corner + gyrotropic_box @ k for k in kpoints])

        HH, delHH = fourier_transform_r_to_k_new(
            operators["HH_R"], kpoints, cwi["unit_cell_cart"], cwi["irvec"], cwi["ndegen"], atoms_frac
        )

        if cwi["zeeman_interaction"]:
            B = cwi["magnetic_field"]
            theta = cwi["magnetic_field_theta"]
            phi = cwi["magnetic_field_phi"]
            g_factor = cwi["g_factor"]

            pauli_spin = fourier_transform_r_to_k_vec(
                operators["SS_R"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac
            )
            H_zeeman = spin_zeeman_interaction(B, theta, phi, pauli_spin, g_factor, cwi["num_wann"])
            HH += H_zeeman

        E, U = np.linalg.eigh(HH)
        HH = None

        delE = wham_get_deleig(delHH, E, U, use_degen_pert, degen_thr)

        gyro_K_orb = np.zeros((3, 3, mum_fermi))

        if cwi.win.eval_spn:
            S_w = fourier_transform_r_to_k_vec(operators["SS_R"], kpoints, cwi["irvec"], cwi["ndegen"], atoms_frac)
            S_k_diag = np.einsum("kln,aklp,kpn->akn", U.conj(), S_w, U, optimize=True)
            gyro_K_spn = np.zeros((3, 3, mum_fermi), dtype=float)
        else:
            gyro_K_spn = None

        for k in range(len(kpoints)):
            kpt = kpoints[k]

            for n1 in range(gyrotropic_num_bands):
                n = gyrotropic_band_list[n1]

                if n > 0 and E[k, n] - E[k, n - 1] <= gyrotropic_degen_thresh:
                    continue

                if n < num_wann - 1 and E[k, n + 1] - E[k, n] <= gyrotropic_degen_thresh:
                    continue

                S = np.zeros(3)
                orb_nk = np.zeros(3)
                got_orb_n = False
                for ifermi in range(mum_fermi):
                    arg = (E[k, n] - fermi_energy_list[ifermi]) / eta_smr
                    #
                    # To save time: far from the Fermi surface, negligible contribution
                    #
                    # -------------------------
                    if np.abs(arg) > gyrotropic_smr_max_arg:
                        continue

                    if cwi.win.eval_spn:
                        S = S_k_diag[:, k, n]

                    # Orbital quantities are computed for each band separately
                    if not got_orb_n:
                        if cwi.win.eval_K:
                            # Fake occupations: band n occupied, others empty
                            occ = np.zeros((1, num_wann))
                            occ[:, n] = 1.0
                            imf_k, img_k, imh_k = berry_get_imfgh_klist(
                                cwi, operators, np.array([kpt]), imf=True, img=True, imh=True, occ=occ
                            )
                            for i in range(3):
                                orb_nk[i] = np.sum(imh_k[0, 0, :, i]) - np.sum(img_k[0, 0, :, i])
                            # curv_nk(i) = sum(imf_k(:, i, 1))
                            # enddo
                        elif cwi.win.eval_D:
                            pass
                            # occ = 0.0
                            # occ(n) = 1.0
                            # call berry_get_imf_klist(kpt, imf_k, occ)
                            # do i = 1, 3
                            # curv_nk(i) = sum(imf_k(:, i, 1))
                            # enddo
                            # got_orb_n = .true. ! Do it for only one value of ifermi

                        if cwi.win.eval_Dw:
                            # gyrotropic_get_curv_w_k(eig, AA, curv_w_nk)
                            pass

                        got_orb_n = True  # Do it for only one value of ifermi

                    delta = (
                        utility_w0gauss(arg, gyrotropic_smr_type_idx) / eta_smr * kweight
                    )  # Broadened delta(E_nk-E_f)
                    #
                    # Loop over Cartesian tensor components
                    #

                    for j in range(3):
                        if cwi.win.eval_K and cwi.win.eval_spn:
                            gyro_K_spn[:, j, ifermi] += np.real(delE[:, k, n] * S[j] * delta)
                        if cwi.win.eval_K:
                            gyro_K_orb[:, j, ifermi] += np.real(delE[:, k, n] * orb_nk[j] * delta)
                        # if cwi.win.eval_D:
                        #     gyro_D[:, j, ifermi] += delE[:, k, n] * curv_nk(j)*delta
                        # if cwi.win.eval_Dw:
                        #     for i in range(3):
                        #         gyro_Dw[i, j, ifermi, :] += delE[i, k, n]*delta*curv_w_nk[n, :, j]

                        # if cwi.win.eval_C:
                        #     gyro_C[:, j, ifermi] += delE[:, k, n] * delE[j, k, n] * delta

                    # if cwi.win.eval_dos:
                    #     gyro_DOS[ifermi] += delta

            # if cwi.win.eval_NOA:
            #     if cwi.win.eval_spn:
            #         gyrotropic_get_NOA_k(kpt, kweight, eig, del_eig, AA, UU, gyro_NOA_orb, gyro_NOA_spn)
            #     else:
            #         gyrotropic_get_NOA_k(kpt, kweight, eig, del_eig, AA, UU, gyro_NOA_orb)

        return gyro_K_orb, gyro_K_spn

    # ==================================================
    N1, N2, N3 = cwi["gyrotropic_kmesh"]
    kpoints = np.array(
        [[i / float(N1), j / float(N2), k / float(N3)] for i in range(N1) for j in range(N2) for k in range(N3)]
    )

    kpoints_chunks = np.split(kpoints, [j for j in range(100000, len(kpoints), 100000)])

    res = Parallel(n_jobs=_num_proc, verbose=10)(delayed(gyrotropic_get_K_k)(kpoints) for kpoints in kpoints_chunks)

    gyro_K_orb = np.sum([gyro_K_orb for gyro_K_orb, _ in res], axis=0)
    gyro_K_spn = np.sum([gyro_K_spn for _, gyro_K_spn in res], axis=0)

    """
    --------------------------------------------------------------------
    At this point gme_orb_list contains

    (1/N)sum_{k,n} delta(E_kn-E_f).(d E_{kn}/d k_i).Im[<del_k u_kn| x (H_k-E_kn)|del_k u_kn>]
    (units of energy times length^3) in eV.Ang^3.

    To get K  in units of Ampere do the following:
        * Divide by V_c in Ang^3 to get a quantity with units of eV
        * Multiply by 'e' in SI to convert to SI (Joules)
        * Multiply by e/(2.hbar) to get K in Ampere

    fac = e^2/(2.hbar.V_c)
     -------------------------------------------------------------------
    """
    cell_volume = cwi["unit_cell_volume"]

    fac = elem_charge_SI**2 / (2.0 * hbar_SI * cell_volume)

    gyro_K_orb *= fac

    """
    --------------------------------------------------------------------
    At this point gyro_K_spn contains

    (1/N) sum_k delta(E_kn-E_f).(d E_{kn}/d k_i).sigma_{kn,j}
    (units of length) in Angstroms.

    To get K in units of Ampere do the following:
        * Divide by V_c in Ang^3 to get a quantity with units of [L]^{-2}
        * Multiply by 10^20 to convert to SI
        * Multiply by -g_s.e.hbar/(4m_e) \simeq e.hbar/(2.m_e) in SI units

    fac = 10^20*e*hbar/(2.m_e.V_c)
     -------------------------------------------------------------------
    """
    fac = -1.0e20 * elem_charge_SI * hbar_SI / (2.0 * elec_mass_SI * cell_volume)

    gyro_K_spn *= fac

    return gyro_K_orb, gyro_K_spn


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
