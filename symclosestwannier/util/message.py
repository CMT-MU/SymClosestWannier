import datetime


# ==================================================
def cw_open_msg():
    dt_now = datetime.datetime.now()
    now = dt_now.strftime("%Y/%m/%d %H:%M:%S")

    msg = f"""
             +---------------------------------------------------+
             |                                                   |
             |                SymClosestWwannier                 |
             |                                                   |
             +---------------------------------------------------+
             |                                                   |
             |         Welcome to the Symmetry-Adapted           |
             |        Closest Wannier Tight-Binding code         |
             |    https://github.com/CMT-MU/SymClosestWannier    |
             |                                                   |
             |                                                   |
             |        SymClosestWwannier Developer Group:        |
             |          Rikuto Oiwa      (RIKEN)                 |
             |          Hiroaki Kusunose (Meiji Univ.)           |
             |                                                   |
             |    For more detailed information,                 |
             |    please check the code documentation and the    |
             |    README on the GitHub page of the code          |
             |                                                   |
             |  Please cite                                      |
             |                                                   |
             |  [ref] "",                                        |
             |        Rikuto Oiwa, Akane Inda, Satoru Hayami,    |
             |        Takuya Nomoto, Ryotaro Arita,              |
             |        Hiroaki Kusunose, in preparation.          |
             |        url                                        |
             |                                                   |
             |  in any publications arising from the use of      |
             |  this code. For the method please cite            |
             |                                                   |
             |  [ref] "Closest Wannier functions to              |
             |         a given set of localized orbitals"        |
             |         Taisuke Ozaki,                            |
             |         arXiv:2306.15296, (2023).                 |
             |                                                   |
             |  [ref] "Symmetry-adapted modeling for             |
             |         molecules and crystals"                   |
             |  Hiroaki Kusunose, Rikuto Oiwa, and Satoru Hayami |
             |         Phys. Rev. B 107, 195118 (2023)           |
             |                                                   |
             |                                                   |
             | Copyright (c) 2023-                               |
             |     The SymClosestWwannier Developer Group and    |
             |        individual contributors                    |
             |                                                   |
             |       Release: 1.2.6       8th July 2024          |
             |                                                   |
             | This program is free software; you can            |
             | redistribute it and/or modify it under the terms  |
             | of the GNU General Public License as published by |
             | the Free Software Foundation; either version 2 of |
             | the License, or (at your option) any later version|
             |                                                   |
             | This program is distributed in the hope that it   |
             | will be useful, but WITHOUT ANY WARRANTY; without |
             | even the implied warranty of MERCHANTABILITY or   |
             | FITNESS FOR A PARTICULAR PURPOSE. See the GNU     |
             | General Public License for more details.          |
             |                                                   |
             | You should have received a copy of the GNU General|
             | Public License along with this program; if not,   |
             | write to the Free Software Foundation, Inc.,      |
             | 675 Mass Ave, Cambridge, MA 02139, USA.           |
             |                                                   |
             +---------------------------------------------------+
             |    Execution started on {now}       |
             +---------------------------------------------------+
    """

    return msg


# ==================================================
def cw_start_msg(seedname):
    msg = f"""

*----------------------------------------------------------------------------*
       Starting a new SymClosestWannier calculation of ''{seedname}''
*----------------------------------------------------------------------------*

"""

    return msg


# ==================================================
def cw_end_msg():
    msg = """

*----------------------------------------------------------------------------*
                             cw_model: All done
*----------------------------------------------------------------------------*

"""

    return msg


# ==================================================
def cw_start_output_msg():
    msg = """

*----------------------------------------------------------------------------*
                               Output results
*----------------------------------------------------------------------------*

"""

    return msg


# ==================================================
def cw_end_output_msg():
    msg = """

*----------------------------------------------------------------------------*
                              Output: All done
*----------------------------------------------------------------------------*

"""

    return msg


# ==================================================
def system_msg(cwi):
    A = cwi["A"]
    B = cwi["B"]
    V = cwi["unit_cell_volume"]

    atoms_frac = cwi["atoms_frac"]
    atoms_cart = cwi["atoms_cart"]

    mp_grid = cwi["mp_grid"]
    num_k = cwi["num_k"]

    msg = f"""

                                   ------
                                   SYSTEM
                                   ------

                            Lattice Vectors (Ang)
                a_1    {'{:>.6f}'.format(A[0][0])}   {'{:>.6f}'.format(A[0][1])}   {'{:>.6f}'.format(A[0][2])}
                a_2    {'{:>.6f}'.format(A[1][0])}   {'{:>.6f}'.format(A[1][1])}   {'{:>.6f}'.format(A[1][2])}
                a_3    {'{:>.6f}'.format(A[2][0])}   {'{:>.6f}'.format(A[2][1])}   {'{:>.6f}'.format(A[2][2])}

                Unit Cell Volume:      {'{:>.5f}'.format(V)}  (Ang^3)

                      Reciprocal-Space Vectors (Ang^-1)
                b_1    {'{:>.6f}'.format(B[0][0])}   {'{:>.6f}'.format(B[0][1])}   {'{:>.6f}'.format(B[0][2])}
                b_2    {'{:>.6f}'.format(B[1][0])}   {'{:>.6f}'.format(B[1][1])}   {'{:>.6f}'.format(B[1][2])}
                b_3    {'{:>.6f}'.format(B[2][0])}   {'{:>.6f}'.format(B[2][1])}   {'{:>.6f}'.format(B[2][2])}

    """

    msg += """
*----------------------------------------------------------------------------*
|   Site       Fractional Coordinate          Cartesian Coordinate (Ang)     |
+----------------------------------------------------------------------------+
    """

    msg += "".join(
        [
            f"""
  {X}   {i}   {'{:>.5f}'.format(vf[0])}   {'{:>.5f}'.format(vf[1])}   {'{:>.5f}'.format(vf[2])}    |    {'{:>.5f}'.format(vc[0])}   {'{:>.5f}'.format(vc[1])}   {'{:>.5f}'.format(vc[2])}
    """
            for ((X, i), vf), vc in zip(atoms_frac.items(), atoms_cart.values())
        ]
    )

    msg += """
*----------------------------------------------------------------------------*
"""

    msg += f"""

                                ------------
                                K-POINT GRID
                                ------------

              Grid size =  {mp_grid[0]} x {mp_grid[1]} x {mp_grid[2]}      Total points = {num_k}
"""

    return msg


# ==================================================
def cw_start_msg_w90(seedname):
    msg = f"""

*----------------------------------------------------------------------------*
     Starting a new calculation of ''{seedname}'' from Wannier90 output
*----------------------------------------------------------------------------*

"""

    return msg


# ==================================================
def cw_start_set_operators_msg():
    msg = f"""

*----------------------------------------------------------------------------*
                             Setting operators
*----------------------------------------------------------------------------*

"""

    return msg


# ==================================================
def cw_end_set_operators_msg():
    msg = f"""

*----------------------------------------------------------------------------*
                         Setting operators: All done
*----------------------------------------------------------------------------*

"""

    return msg


# ==================================================
def cw_start_response_msg():
    msg = """

*----------------------------------------------------------------------------*
                                cw_response
*----------------------------------------------------------------------------*

"""

    return msg


# ==================================================
def cw_end_response_msg():
    msg = """

*----------------------------------------------------------------------------*
                            cw_response: All done
*----------------------------------------------------------------------------*

"""

    return msg


# ==================================================
def cw_start_expectation_msg():
    msg = """

*----------------------------------------------------------------------------*
                                cw_expectation
*----------------------------------------------------------------------------*

"""

    return msg


# ==================================================
def cw_end_expectation_msg():
    msg = """

*----------------------------------------------------------------------------*
                            cw_expectation: All done
*----------------------------------------------------------------------------*

"""

    return msg


# ==================================================
def cw_start_band_msg():
    msg = """

*----------------------------------------------------------------------------*
                                   cw_band
*----------------------------------------------------------------------------*

"""

    return msg


# ==================================================
def cw_end_band_msg():
    msg = """

*----------------------------------------------------------------------------*
                              cw_band: All done
*----------------------------------------------------------------------------*

"""

    return msg


# ==================================================
def postcw_end_msg():
    msg = """

*----------------------------------------------------------------------------*
                              postcw: All done
*----------------------------------------------------------------------------*

"""

    return msg
