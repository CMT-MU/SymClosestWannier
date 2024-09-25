# standard output
Example standard output for [graphene](../../example/graphene/graphene.cwin) is given here.
```
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
             |         , Phys. Rev. B 110, 125115, (2024).                 |
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
             |       Release: 1.1.19       22th January 2024      |
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
             |    Execution started on 2024/02/19 18:24:57       |
             +---------------------------------------------------+



                                   ------
                                   SYSTEM
                                   ------

                            Lattice Vectors (Ang)
                a_1    2.435000   0.000000   0.000000
                a_2    -1.217500   2.108772   0.000000
                a_3    0.000000   0.000000   9.739999

                Unit Cell Volume:      50.01352  (Ang^3)

                      Reciprocal-Space Vectors (Ang^-1)
                b_1    2.580364   1.489774   0.000000
                b_2    0.000000   2.979547   0.000000
                b_3    0.000000   0.000000   0.645091


*----------------------------------------------------------------------------*
|   Site       Fractional Coordinate          Cartesian Coordinate (Ang)     |
+----------------------------------------------------------------------------+

  C   1   0.66667   0.33333   0.00000    |    1.21750   0.70292   0.00000

  C   2   0.33333   0.66667   0.00000    |    0.00000   1.40585   0.00000

*----------------------------------------------------------------------------*


                                ------------
                                K-POINT GRID
                                ------------

              Grid size =  19 x 19 x 1      Total points = 361



*----------------------------------------------------------------------------*
       Starting a new SymClosestWannier calculation of ''graphene_pz''
*----------------------------------------------------------------------------*


   - disentanglement ... done ( 0.0 [sec] ).
   - constructing TB Hamiltonian ... done ( 0.068 [sec] ).
   - symmetrization ...
    - reading output of multipie ...
  * read './sym/graphene_pz_20_model.py'.
  * read './sym/graphene_pz_20_samb.py'.
  * read './sym/graphene_pz_20_matrix.py'.
    - decomposing Hamiltonian as linear combination of SAMBs ... done ( 0.008 [sec] ).
    - decomposing overlap as linear combination of SAMBs ... done ( 0.0 [sec] ).
    - decomposing non-orthogonal Hamiltonian as linear combination of SAMBs ... done ( 0.0 [sec] ).
    - constructing symmetrized TB Hamiltonian ... done ( 0.025 [sec] ).
    - evaluating fitting accuracy ...
     * RMSE of eigen values between CW and Symmetry-Adapted CW models (grid) = 2.6820 [meV]
     * RMSE of eigen values between CW and Symmetry-Adapted CW models (path) = 3.7451 [meV]
    - evaluating expectation value of {Zj} at T = 0 ...
done


*----------------------------------------------------------------------------*
                               Output results
*----------------------------------------------------------------------------*


  * wrote 'graphene_pz_hr.dat.cw'.
  * wrote 'graphene_pz_sr.dat.cw'.
  * wrote 'graphene_pz_r.dat.cw'.
  * wrote './sym/graphene_pz_20_hr_sym.dat.cw'.
  * wrote './sym/graphene_pz_20_sr_sym.dat.cw'.
  * wrote './sym/graphene_pz_20_z.dat.cw'.
  * wrote './sym/graphene_pz_20_s.dat.cw'.
  * wrote './sym/graphene_pz_20_z_exp.dat.cw'.


  * total elapsed_time: ( 1.175 [sec] ).


*----------------------------------------------------------------------------*
                              Output: All done
*----------------------------------------------------------------------------*


  * total elapsed_time: ( 5.717 [sec] ).


*----------------------------------------------------------------------------*
                             cw_model: All done
*----------------------------------------------------------------------------*
```
