import datetime


# ==================================================
def opening_msg():
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
             |      Release: 1.0.0       10th December 2023      |
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
def starting_msg(cwi):
    msg = f"""

*----------------------------------------------------------------------------*
       Starting a new SymClosestWannier calculation of ''{cwi['seedname']}''
*----------------------------------------------------------------------------*

"""

    return msg


# ==================================================
def ending_msg():
    msg = """

*----------------------------------------------------------------------------*
                           All done: pw2cw exiting
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
                a_1    {'{:.6f}'.format(A[0][0])}   {'{:.6f}'.format(A[0][1])}   {'{:.6f}'.format(A[0][2])}
                a_2    {'{:.6f}'.format(A[1][0])}   {'{:.6f}'.format(A[1][1])}   {'{:.6f}'.format(A[1][2])}
                a_3    {'{:.6f}'.format(A[2][0])}   {'{:.6f}'.format(A[2][1])}   {'{:.6f}'.format(A[2][2])}

                Unit Cell Volume:      {'{:.5f}'.format(V)}  (Ang^3)

                      Reciprocal-Space Vectors (Ang^-1)
                b_1    {'{:.6f}'.format(B[0][0])}   {'{:.6f}'.format(B[0][1])}   {'{:.6f}'.format(B[0][2])}
                b_2    {'{:.6f}'.format(B[1][0])}   {'{:.6f}'.format(B[1][1])}   {'{:.6f}'.format(B[1][2])}
                b_3    {'{:.6f}'.format(B[2][0])}   {'{:.6f}'.format(B[2][1])}   {'{:.6f}'.format(B[2][2])}

    """

    msg += """
*----------------------------------------------------------------------------*
|   Site       Fractional Coordinate          Cartesian Coordinate (Ang)     |
+----------------------------------------------------------------------------+
    """

    msg += "".join(
        [
            f"""
  {X}   {i}   {'{:.5f}'.format(vf[0])}   {'{:.5f}'.format(vf[1])}   {'{:.5f}'.format(vf[2])}    |    {'{:.5f}'.format(vc[0])}   {'{:.5f}'.format(vc[1])}   {'{:.5f}'.format(vc[2])}
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