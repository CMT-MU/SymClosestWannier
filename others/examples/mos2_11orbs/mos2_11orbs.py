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
    # Mo at the origin and S at z = +-1.5735 A (+-0.0497 c of mos2_11orbs.win), as in mos2_11orbs.win.
    # MultiPie puts the S below Mo at z ~ 1; SymCW matches it to the S at z < 0 of mos2_11orbs.nnkp.
    "site": {"Mo": ("[2/3,1/3,0]", "d"), "S": ("[1/3,2/3,0.0015735]", "p")},
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
