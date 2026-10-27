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
#                 spin_moment: spin magnetic moment                  #
#                                                                    #
# ****************************************************************** #

# The spin magnetic moment of SymClosestWannier (with the Zeeman term), and the check of a single Fermi energy shared by
# the responses. Moved from get_response.py unchanged.
import numpy as np
from joblib import wrap_non_picklable_objects

from symclosestwannier.util.utility import (
    fourier_transform_r_to_k,
    fourier_transform_r_to_k_vec,
    spin_zeeman_interaction,
    spn_operator,
    tb_gauge_positions,
    thermal_avg,
)
from symclosestwannier.util.constants import bohr_magn_SI, joul_to_eV
from symclosestwannier.util.exceptions import SymCWInputError


# ==================================================
def _single_fermi_energy(cwi, what):
    """
    the Fermi energy of the calculations at a single Fermi energy, fermi_energy_list[0] as in wannier90; a Fermi energy
    scan is rejected.

    Args:
        cwi (CWInfo): CWInfo.
        what (str): the calculation, for the error message.

    Returns:
        float: Fermi energy (eV).
    """
    if cwi["num_fermi"] != 1:
        raise SymCWInputError(f"{what} needs a single Fermi energy (fermi_energy in seedname.win), not a scan.")
    return cwi["fermi_energy_list"][0]


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
    ef = _single_fermi_energy(cwi, "the spin moment")

    d = {"Ms_x": 0.0, "Ms_y": 0.0, "Ms_z": 0.0}

    if not cwi["spinors"]:
        return d
    else:
        atoms_frac = tb_gauge_positions(cwi)

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

            return thermal_avg([spn_x, spn_y, spn_z], E, U, ef, T_Kelvin=0.0, num_k=num_k)

        # ==================================================
        kpoints_chunks = np.split(kpoints, [j for j in range(1000, len(kpoints), 1000)])

        for idx, kpoints in enumerate(kpoints_chunks):
            print(f"* {idx+1}/{len(kpoints_chunks)}")
            Ms_x_k, Ms_y_k, Ms_z_k = spin_moment_main_k(kpoints)
            d["Ms_x"] += Ms_x_k
            d["Ms_y"] += Ms_y_k
            d["Ms_z"] += Ms_z_k

        return d
