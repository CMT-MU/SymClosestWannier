# Construction of SAMBs by using MultiPie

**Note**: if symmetrization is not required, this procedure is not necessary

1. Prepare input file. Example input files for [CH4](https://github.com/CMT-MU/SymClosestWannier/blob/main/tutorials/tutorial01/ch4.py) molecule and [graphene](https://github.com/CMT-MU/SymClosestWannier/blob/main/tutorials/tutorial02/graphene.py) are given here.
    - CH4 molecule
    ```{literalinclude} ../../tutorials/tutorial01/ch4.py
    ```
    - graphene
    ```{literalinclude} ../../tutorials/tutorial02/graphene.py
    ```

2. At the folder where these two input files exist, do the following to create SAMB.
For more details, run "**mp_create --help**".
    ```
    $ mp_create ch4 graphene
    ```

3. The following files are created in **ch4** and **graphene** folders.
    - ch4.pkl, ch4.qtdw, ch4.pdf
    - graphene.pkl, graphene.qtdw, graphene.pdf

    Here, **.pdf** is created if [TeXLive](https://www.tug.org/texlive/) is installed, and **.qtdw** is created if [QtDraw](https://github.com/CMT-MU/QtDraw) is installed.

    Each file contains
    - **.pkl** : model information including SAMB (binary), which is read by `pw2cw` through `mp_seedname`.
    - **.pdf** : detailed information on the model and SAMB in PDF format.
    - **.qtdw** : molecular or crystal structure file for QtDraw.

Create the models with the MultiPie installed with SymClosestWannier: a model (**.pkl**) created with MultiPie Ver.2 is not read by Ver.3 (see [Changes from Ver.2](https://cmt-mu.github.io/MultiPie/changes.html)). SymClosestWannier works with both versions, and the names of the f orbitals of the Wannier functions follow the installed MultiPie (`f3x`, `f3y`, `f2`, `f1` in Ver.2 are `fx`, `fy`, `fc`, `fs` in Ver.3). In Ver.3, the bonds are given by **bond** only (**max_neighbor** and **search_cell_range** are removed).

See [MultiPie](https://github.com/CMT-MU/MultiPie) for more detailed file formats.
