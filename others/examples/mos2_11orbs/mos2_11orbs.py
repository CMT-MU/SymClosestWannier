# mos2_11orbs.py
"""
input file for monolayer MoS2 (Mo d and S p orbitals).
"""

mos2_11orbs = {
    "model": "mos2_11orbs",  # name of model.
    "group": 187,  # No. of space group.
    # a = 5.983 bohr as in mos2_11orbs.win; set large enough interlayer distance.
    "cell": {"a": 3.16607, "b": 3.16607, "c": 1000},
    #
    # Mo at z = 1/2, and S at 1.5735 A (0.0497 c of mos2_11orbs.win) above and below it. The atoms are shifted by c/2
    # from mos2_11orbs.win, so that all of them are in the same unit cell, as in mos2_11orbs.win (where the S below Mo
    # is at z < 0): the lattice vectors R of H(R) are then the same as those of Wannier90.
    "site": {"Mo": ("[2/3,1/3,1/2]", "d"), "S": ("[1/3,2/3,0.5015735]", "p")},
    # Mo-Mo, Mo-S and S-S bonds up to 50th neighbors.
    # fmt: off
    "bond": [
        ("Mo", "Mo", [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        ]),
        ("Mo", "S", [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        ]),
        ("S", "S", [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        ]),
    ],
    # fmt: on
    #
    "spinful": False,  # spinless.
}
