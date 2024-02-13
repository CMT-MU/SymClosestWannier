# standard output
Example standard output for [graphene](../../example/graphene/graphene.cwin) is given here.
```
********************************************************************************
*                                                                              *
*  Create Closest Wannier Tight-Binding Model from Plane-Wave DFT Calculation  *
*                                                                              *
********************************************************************************


* graphene

 - reading output of DFT calculation ... done (0.05 [sec])
 - eliminating bands with low projectability (proj_min = 0.2) ... done (0.00 [sec])
 - disentanglement ... done (0.00 [sec])
 - constructing TB Hamiltonian ... done (0.01 [sec])
 - symmetrization ...
   - reading output of multiple ...
  * read './graphene_model.py'.
  * read './graphene_samb.py'.
  * read './graphene_matrix.py'.
done (0.03 [sec])
   - decomposing Hamiltonian Hr as linear combination of SAMBs ... done (0.00 [sec])
   - decomposing overlap Sr as linear combination of SAMBs ... done (0.00 [sec])
   - evaluating fitting accuracy ...
     * RMSE of eigen values between CW and Symmetry-Adapted CW models (grid) = 47.6916 [meV]
     * RMSE of eigen values between CW and Symmetry-Adapted CW models (path) = 51.6997 [meV]
    done (0.04 [sec])
  done (0.14 [sec])

 - total elapsed_time: 0.21 [sec]


********************************************************************************
*                                                                              *
*                            Successfully Completed                            *
*                                                                              *
********************************************************************************


  * wrote 'graphene_info.py'.
  * wrote 'graphene_data.py'.
  * wrote 'graphene_hr.dat'.
  * wrote 'graphene_sr.dat'.
  * wrote 'graphene_hr_sym.dat'.
  * wrote 'graphene_sr_sym.dat'.
  * wrote 'graphene_z.dat'.
  * wrote 'graphene_s.dat'.
```
