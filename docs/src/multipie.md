# Construction of SAMBs by using MultiPie

**Note**: if symmetrization is not required, this procedure is not necessary

1. Prepare input file. Example input files for [CH4](../../tutorials/tutorial01/ch4.py) molecule and [graphene](../../tutorials/tutorial02/graphene.py) are given here.
    - CH4 molecule
    ```{literalinclude} ../../tutorials/tutorial01/ch4.py
    ```
    - graphene
    ```{literalinclude} ../../tutorials/tutorial02/graphene.py
    ```

2. At the folder where these two input files exist, do the following to create SAMB.
See for more detail, try "**create_samb --help**" command.
    ```
    $ create_samb ch4 graphene
    ```

3. The following files are created in **ch4** and **graphene** folders.
    - ch4_model.py, ch4_samb.py, ch4_matrix.py, ch4_samb.tex, ch4_samb.pdf, ch4_view.qtdw
    - graphene_model.py, graphene_samb.py, graphene_matrix.py, graphene_samb.tex, graphene_samb.pdf, graphene_view.qtdw

    Here, **.tex** and **.pdf** are created if [TeXLive](https://www.tug.org/texlive/) is installed, and **.qtdw** is created if [QtDraw](https://github.com/CMT-MU/QtDraw) is installed.

    Each file contains
    - **_model.py** : model information.
    - **_samb.py** : detailed information on SAMB.
    - **_matrix.py** : full-matrix form of SAMB.
    - **_samb.tex** and **_samb.pdf** : detailed information on SAMB in LaTeX and PDF format.
    - **_view.qtdw** : molecular or crystal structure file for QtDraw.

See for more detaied file formats [MultiPie](https://github.com/CMT-MU/MultiPie).
